#pragma once

#include "native/env.h"
#include "storage/weights/moe/cached_source_internal.h"
#include <limits>

// Seed real compressed transfer-cache fields without launching the unrelated
// FFN consumer graph; this fixture can also run under Compute Sanitizer.
struct MoeResidencyTestAccess {
    static MoeResidencyManager& install_manager(MoeExpertCache& cache,std::unique_ptr<MoeResidencyManager> manager) {
        cache.moe_residency_=std::move(manager);return *cache.moe_residency_;
    }
    static void seed_full_exchange_loan(MoeExpertCache& cache) {
        cache.kv_expert_loan_=std::make_shared<mfq::cuda::KvExpertLoan>();
        auto arena=tb::zeros({32<<20},tb::TensorOptions().device(tb::kCUDA).dtype(tb::kUInt8));
        cache.kv_expert_loan_->add(arena,arena.numel(),1,true,1,true,true);
        for(const auto& source:cache.sources_) {
            auto lease=source->quant_source_->acquire_expert(1);
            const auto fields=moe_cache_fields(lease->weights);
            const auto& sizes=source->cohorts_[source->expert_to_cohort_[1]].bytes_per_expert;
            bool hit=false;
            if(!cache.kv_expert_loan_->pin_projection(source->id_,1,sizes,mfq_current_cuda_stream(),hit,
                [&](void* destination,const auto& offsets) {
                    for(std::size_t f=0;f<fields.size();++f)if(sizes[f])MFQ_CUDA_CHECK(cudaMemcpyAsync(
                        static_cast<uint8_t*>(destination)+offsets[f],fields[f].data_ptr(),std::size_t(sizes[f]),
                        cudaMemcpyHostToDevice,mfq_current_cuda_stream()));
                }))throw std::runtime_error("full exchange loan could not store a projection");
        }
        MFQ_CUDA_CHECK(cudaStreamSynchronize(mfq_current_cuda_stream()));
    }
    static void pressure_exchange_loan(MoeExpertCache& cache) {
        auto& loan=*cache.kv_expert_loan_;
        for(const auto& source:cache.sources_)if(source->layer_id_<0)
            loan.retire_expert(source->id_,2,int(source->cohorts_[source->expert_to_cohort_[2]].bytes_per_expert.size()));
        loan.next_packet();
        auto bytes=tb::full({mfq::cuda::KvExpertLoan::page},37,
            tb::TensorOptions().device(tb::kCPU).dtype(tb::kUInt8));
        auto guard=loan.retain_exchange_reads();bool full=false;
        for(int i=0;i<1024;++i) {
            bool hit=false;
            if(!loan.pin({99999,i,0},bytes.data_ptr(),bytes.nbytes(),mfq_current_cuda_stream(),hit)){full=true;break;}
        }
        MFQ_CUDA_CHECK(cudaStreamSynchronize(mfq_current_cuda_stream()));
        if(!full)throw std::runtime_error("predictor pressure fixture did not fill its cache");
    }
    static void check_predictor_cached(MoeExpertCache& cache) {
        auto& source=*cache.sources_[3];
        const auto& sizes=source.cohorts_[source.expert_to_cohort_[2]].bytes_per_expert;
        auto lease=source.quant_source_->acquire_expert(2);const auto fields=moe_cache_fields(lease->weights);
        bool hit=false;
        const auto* pointer=cache.kv_expert_loan_->pin_projection(source.id_,2,sizes,mfq_current_cuda_stream(),hit,
            [&](void* destination,const auto& offsets) {
                for(std::size_t f=0;f<fields.size();++f)if(sizes[f])MFQ_CUDA_CHECK(cudaMemcpyAsync(
                    static_cast<uint8_t*>(destination)+offsets[f],fields[f].data_ptr(),std::size_t(sizes[f]),
                    cudaMemcpyHostToDevice,mfq_current_cuda_stream()));
            });
        MFQ_CUDA_CHECK(cudaStreamSynchronize(mfq_current_cuda_stream()));
        if(!pointer || !cache.kv_expert_loan_->find_for_exchange({source.id_,2,0},std::size_t(sizes[0])))
            throw std::runtime_error("unrelated exchange pins forced predictor misses through wire instead of caching them");
    }
    static void bundles(MoeExpertCache& cache,std::vector<std::array<int,3>> bundles) {
        cache.pipeline_bundles_=std::move(bundles);
    }
    static int64_t prefill_copy_jobs(const MoeExpertCache& cache){return cache.stats_.pipeline_prefill_cpu_copy_jobs;}
    static int prefill_copy_pending(const MoeExpertCache& cache){return cache.pipeline_prefill_copy_pending_;}
    static void verify_prefill_copy_failure_cleanup(MoeExpertCache& cache) {
        if(cache.pipeline_prefill_copy_pending_)throw std::runtime_error("test injected over an active prefill copy");
        std::atomic<bool> joined=false;
        cache.pipeline_prefill_copy_tasks_[0]=std::async(std::launch::async,[]()->MoeExpertCache::PrefillCopySubmission {
            throw std::runtime_error("injected CPU prefill submission failure");
        });
        cache.pipeline_prefill_copy_tasks_[1]=std::async(std::launch::async,[&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));joined=true;
            MoeExpertCache::PrefillCopySubmission copy;copy.slot=1;copy.layer=999;return copy;
        });
        cache.pipeline_prefill_copy_pending_=2;bool failed=false;
        try{release_moe_prefill_buffers(&cache);}catch(const std::runtime_error& error) {
            if(std::string(error.what())!="injected CPU prefill submission failure")throw;
            failed=true;
        }
        if(!failed || !joined || cache.pipeline_prefill_copy_pending_)
            throw std::runtime_error("prefill copy failure did not drain every future");
        release_moe_prefill_buffers(&cache);
    }
    static void seed_prefill_shadow(MoeExpertCache& cache,bool stale,bool reclaim,bool permuted=false,bool refresh=false) {
        for(const auto& source:cache.sources_)for(const auto& cohort:source->cohorts_) {
            auto& arena=*cohort.arena;if(arena.prefill_backup_key && !refresh)continue;
            const mfq::engine::MoeCacheKey key=arena.prefill_backup_key.value_or(mfq::engine::MoeCacheKey{-2-source->id_,cohort.index,0});
            std::vector<tb::Tensor> fields;std::size_t image_bytes=arena.slots*3*sizeof(int32_t);
            for(const auto& field:arena.fields) {
                auto host=field.cpu();
                if(permuted) {
                    auto reordered=tb::empty_like(host);const auto bytes=std::size_t(tensor_nbytes(host)/arena.slots);
                    for(int row=0;row<arena.slots;++row)
                        std::memcpy(static_cast<uint8_t*>(reordered.data_ptr())+row*bytes,
                            static_cast<const uint8_t*>(host.data_ptr())+((row+1)%arena.slots)*bytes,bytes);
                    host=std::move(reordered);
                }
                fields.push_back(std::move(host));image_bytes+=tensor_nbytes(field);
            }
            auto image=moe_replace_quant_fields(*cohort.cpu,fields);
            image.expert_local=tb::empty({arena.slots,3},tb::TensorOptions().device(tb::kCPU).dtype(tb::kInt32));
            auto* identities=image.expert_local.data_ptr<int32_t>();
            for(int slot=0;slot<arena.slots;++slot) {
                const auto& owner=arena.book->key_for(permuted?(slot+1)%arena.slots:slot);
                identities[slot*3]=owner?owner->source:-1;
                identities[slot*3+1]=owner?owner->cohort:-1;
                identities[slot*3+2]=owner?owner->expert+(stale?100:0):-1;
            }
            if(refresh) {
                const auto owner=cache.host_experts_->find(key);
                if(!owner)throw std::runtime_error("shadow refresh lost its optional owner");
                auto current=moe_cache_fields(owner->weights);
                for(std::size_t f=0;f<fields.size();++f)current[f].copy_(fields[f]);
                auto identities=owner->weights.expert_local;identities.copy_(image.expert_local);
            } else cache.host_experts_->acquire(key,image_bytes,[image]{return image;});
            arena.prefill_backup_key=key;
            ++arena.prefill_backup_generation;
        }
        if(reclaim)cache.host_experts_->reclaim_optional(std::numeric_limits<std::size_t>::max());
    }
    static void reclaim_shadows(MoeExpertCache& cache) {
        cache.host_experts_->reclaim_optional(std::numeric_limits<std::size_t>::max());
        if(cache.host_experts_->stats().optional_bytes)
            throw std::runtime_error("completed shadow lookup retained optional weight ownership");
    }
    static void seed_kv_refill(MoeExpertCache& cache) {
        cache.kv_expert_loan_=std::make_shared<mfq::cuda::KvExpertLoan>();
        auto arena=tb::zeros({16<<20},tb::TensorOptions().device(tb::kCUDA).dtype(tb::kUInt8));
        cache.kv_expert_loan_->add(arena,4096,4096,false);
        for(const auto& source:cache.sources_) {
            const auto lease=source->quant_source_->acquire_expert(1);
            const auto fields=moe_cache_fields(lease->weights);
            for(std::size_t f=0;f<fields.size();++f) {
                const auto bytes=tensor_nbytes(fields[f]);if(bytes<int64_t(mfq::cuda::KvExpertLoan::page))continue;
                bool hit=false;
                if(!cache.kv_expert_loan_->pin({source->id_,1,int64_t(f)},fields[f].data_ptr(),
                        std::size_t(bytes),mfq_current_cuda_stream(),hit))
                    throw std::runtime_error("KV refill fixture did not retain an expert field");
            }
        }
        MFQ_CUDA_CHECK(cudaStreamSynchronize(mfq_current_cuda_stream()));
    }
    static void seed_exchange_wire(MoeExpertCache& cache,int64_t bytes) {
        cache.pipeline_wire_gpu_=tb::full({bytes+4096},91,tb::TensorOptions().device(tb::kCUDA).dtype(tb::kUInt8));
        MFQ_CUDA_CHECK(cudaStreamSynchronize(mfq_current_cuda_stream()));
    }
    static void check_exchange_wire(const MoeExpertCache& cache) {
        const auto& wire=cache.pipeline_wire_gpu_;
        if(!wire.defined())return;
        for(const auto offset:{int64_t(0),wire.numel()-256}) {
            std::array<uint8_t,256> bytes;
            MFQ_CUDA_CHECK(cudaMemcpy(bytes.data(),wire.data_ptr<uint8_t>()+offset,bytes.size(),cudaMemcpyDeviceToHost));
            if(std::any_of(bytes.begin(),bytes.end(),[](auto v){return v!=91;}))
                throw std::runtime_error("residency exchange overwrote wire guards");
        }
    }
    static bool wire_payload_used(const MoeExpertCache& cache) {
        std::array<uint8_t,256> payload;
        MFQ_CUDA_CHECK(cudaMemcpy(payload.data(),cache.pipeline_wire_gpu_.data_ptr<uint8_t>()+256,
            payload.size(),cudaMemcpyDeviceToHost));
        return std::any_of(payload.begin(),payload.end(),[](auto v){return v!=91;});
    }
    static bool transfer_enabled(const MoeExpertCache& cache){return cache.pipeline_transfer_cache_;}
    static void verify_transfer_fields(MoeExpertCache& cache) {
        for(const auto& source:cache.sources_)for(const auto& cohort:source->cohorts_) {
            const auto key=source->projection_role_+":"+cohort.arena->signature;
            auto found=cache.pipeline_stages_.find(key);
            if(found==cache.pipeline_stages_.end() || !found->second->book)continue;
            const auto& arena=*found->second;
            for(int e=0;e<source->n_experts();++e) {
                const int slot=arena.book->slot_for({source->id_,cohort.index,e});if(slot<0)continue;
                const auto held=source->quant_source_->acquire_expert(e);
                const auto fields=moe_cache_fields(held->weights);
                for(std::size_t f=0;f<fields.size();++f) {
                    const auto bytes=cohort.bytes_per_expert[f];if(!bytes)continue;
                    std::vector<uint8_t> actual(static_cast<std::size_t>(bytes));
                    MFQ_CUDA_CHECK(cudaMemcpy(actual.data(),static_cast<const uint8_t*>(arena.fields[f].data_ptr())+slot*bytes,
                        actual.size(),cudaMemcpyDeviceToHost));
                    if(std::memcmp(actual.data(),fields[f].data_ptr(),actual.size()))
                        throw std::runtime_error("transfer field bytes differ: source="+std::to_string(source->id_)+
                            " expert="+std::to_string(e)+" field="+std::to_string(f));
                }
            }
        }
    }
    static void verify_defaults_guard(MoeExpertCache& cache) {
        const auto saved=cache.ram_pcie_fraction_;
        cache.ram_pcie_fraction_=0.5;
        if(MoeResidencyManager(&cache).stats().exchange_defaults)throw std::runtime_error("mixed CPU routing enabled exchange defaults");
        cache.ram_pcie_fraction_=saved;
        cache.pipeline_shared_cpu_cost_=true;
        if(MoeResidencyManager(&cache).stats().exchange_defaults)throw std::runtime_error("CPU calibration enabled exchange defaults");
        cache.pipeline_shared_cpu_cost_=false;
    }
    static void seed(MoeExpertCache& cache,int projection,int expert) {
        auto& source=*cache.sources_.at(projection);
        const int c=source.expert_to_cohort_.at(expert);auto& cohort=source.cohorts_.at(c);
        const auto key=std::to_string(projection)+":"+cohort.arena->signature;
        auto& arena=cache.pipeline_stages_[key];
        if(!arena) {
            arena=std::make_unique<MoeGpuArena>();arena->signature=key;
            arena->slot_bytes=cohort.arena->slot_bytes;arena->slots=1;arena->layouts=cohort.arena->layouts;
            arena->book=std::make_unique<mfq::engine::MoeCacheSlotBook>(1);
        }
        const auto held=source.quant_source_->acquire_expert(expert);
        const auto fields=moe_cache_fields(held->weights);
        if(arena->fields.empty())for(const auto& field:fields)arena->fields.push_back(field.to(tb::kCUDA));
        else for(std::size_t f=0;f<fields.size();++f)arena->fields[f].copy_(fields[f].to(tb::kCUDA));
        arena->book->acquire(source.quant_source_->host_key(expert));
        MFQ_CUDA_CHECK(cudaStreamSynchronize(mfq_current_cuda_stream()));
    }
    static void compare_gpu(MoeExpertCache& cache,int projection,int expert,const MixedMoePool& original) {
        auto& source=*cache.sources_.at(projection);
        const int c=source.expert_to_cohort_.at(expert);const auto& cohort=source.cohorts_.at(c);
        const int slot=cohort.host_map.at(expert);
        if(slot<0 || slot!=cohort.arena->book->slot_for(source.quant_source_->host_key(expert)))
            throw std::runtime_error("direct exchange GPU map/book ownership mismatch");
        std::vector<int32_t> device_map(cohort.host_map.size());
        MFQ_CUDA_CHECK(cudaMemcpy(device_map.data(),cohort.active.expert_local.data_ptr<int32_t>(),
            device_map.size()*sizeof(int32_t),cudaMemcpyDeviceToHost));
        if(device_map!=cohort.host_map)throw std::runtime_error("direct exchange published a stale device map");
        const auto fields=moe_cache_fields(original);
        for(std::size_t f=0;f<fields.size();++f) {
            const auto bytes=cohort.bytes_per_expert[f];if(!bytes)continue;
            std::vector<uint8_t> actual(static_cast<std::size_t>(bytes));
            const auto* gpu=static_cast<const uint8_t*>(cohort.arena->fields[f].data_ptr())+int64_t(slot)*bytes;
            MFQ_CUDA_CHECK(cudaMemcpy(actual.data(),gpu,actual.size(),cudaMemcpyDeviceToHost));
            if(std::memcmp(actual.data(),fields[f].data_ptr(),actual.size()))
                throw std::runtime_error("direct exchange changed a GPU weight byte");
        }
    }
};

void verify_predictor_exchange_overlap(const std::filesystem::path& root) {
    int cases=0;
    for(bool shared_arena:{false,true})for(bool full_cache:{false,true})
    for(const std::string failure:{"none","direct_refill","publish"})for(bool pressure:{false,true}) {
        if((shared_arena || !full_cache) && failure!="none")continue;
        if(pressure && (shared_arena || !full_cache || failure!="none"))continue;
        CudaExecutionContext execution;auto& cfg=execution.config;
        cfg.moe_pipeline=true;cfg.moe_preload_all=true;cfg.moe_assert_resident=true;
        cfg.moe_hybrid_cpu=false;cfg.moe_direct_ram=true;cfg.moe_ram_pcie=true;cfg.moe_ram_pcie_fraction=1.;
        cfg.moe_residency_adapt=true;cfg.moe_residency_warm=true;cfg.moe_residency_projection_heat=true;
        std::array<std::shared_ptr<MoeQuantRangeSource>,6> sources;
        std::array<std::string,6> paths;int64_t budget=0;
        for(int i=0;i<6;++i) {
            const auto format=i<3 || shared_arena?"nint5":"nvq3j-512";
            paths[i]=format+std::string(i%3==2?"-2560-640.mfq":"-640-2560.mfq");
            sources[i]=source(root/paths[i]);budget+=bytes(sources[i]->metadata()->pools[0]);
        }
        cfg.moe_host_cache_bytes=8*budget+(1<<20);
        auto cache=make_moe_expert_cache(budget,cfg);
        std::array<MoeFfnForward,2> forward;
        for(int layer=0;layer<2;++layer) {
            std::array<std::shared_ptr<MfeWeight>,3> weights;
            for(int p=0;p<3;++p)weights[p]=std::make_shared<MfeWeight>(cache_quant_moe_weight(cache,
                "overlap/"+std::to_string(layer)+"/"+std::to_string(p),sources[layer*3+p],1,layer?-1:0,std::to_string(p)));
            forward[layer]=make_moe_ffn_pipeline({weights[0],weights[1]},{weights[2]});
        }
        finalize_moe_expert_cache(cache);
        for(const auto& item:sources)if(!item->gpu_resident(0) || item->gpu_resident(1))
            throw std::runtime_error("predictor overlap fixture requires expert0 on GPU and expert1 in RAM");
        if(full_cache)MoeResidencyTestAccess::seed_full_exchange_loan(*cache);
        else MoeResidencyTestAccess::seed_kv_refill(*cache);
        auto x=tb::full({1,2560},.07,tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat16));
        auto ids=tb::tensor(std::vector<int32_t>{0,pressure?2:1}).reshape({1,2}).to(tb::kCUDA);
        auto weights=tb::full({1,2},.5,tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat32));
        auto expected=forward[1](execution,x,ids,weights).cpu().contiguous();
        if(pressure)MoeResidencyTestAccess::pressure_exchange_loan(*cache);
        std::mutex mutex;std::condition_variable changed;bool release=false,held=false;
        const auto* enabled=mfq::cuda::env_lookup("MFQ_MOE_PREDICTOR_EXCHANGE_OVERLAP");
        const auto* keyed=mfq::cuda::env_lookup("MFQ_MOE_KEYED_EXCHANGE");
        const bool expect_overlap=!shared_arena && full_cache && (!enabled || enabled[0]!='0') && (!keyed || keyed[0]=='1');
        auto& manager=MoeResidencyTestAccess::install_manager(*cache,std::make_unique<MoeResidencyManager>(cache.get(),
            [&](const char* event,std::size_t index) {
                if(!index && std::string(event)=="predictor_overlap_ready") {
                    std::unique_lock<std::mutex> lock(mutex);held=true;changed.notify_all();
                    if(!changed.wait_for(lock,std::chrono::seconds(5),[&]{return release;}))
                        throw std::runtime_error("unrelated predictor remained blocked by the held exchange");
                }
                if(!index && failure==event)throw std::runtime_error("injected predictor overlap failure");
            }));
        // One complete adaptation period, using only backbone observations.
        for(int i=0;i<4;++i)manager.after_layer(0,std::vector<int32_t>(10,1),1);
        bool rejected=false;
        try {
            manager.prepare_predictor();
            if(expect_overlap) {
                {std::lock_guard<std::mutex> lock(mutex);if(!held || release)
                    throw std::runtime_error("predictor did not enter a live held transaction");}
                auto actual=forward[1](execution,x,ids,weights).cpu().contiguous();
                if(std::memcmp(actual.data_ptr(),expected.data_ptr(),actual.nbytes()))
                    throw std::runtime_error("concurrent predictor changed an FFN output byte");
                if(pressure)MoeResidencyTestAccess::check_predictor_cached(*cache);
            }
            {std::lock_guard<std::mutex> lock(mutex);release=true;}changed.notify_all();
            manager.apply_pending();
        }catch(const std::runtime_error& error) {
            {std::lock_guard<std::mutex> lock(mutex);release=true;}changed.notify_all();
            if(std::string(error.what())!="injected predictor overlap failure")throw;
            rejected=true;
        }
        if(rejected!=(failure!="none") || bool(manager.stats().predictor_overlap_serves)!=expect_overlap)
            throw std::runtime_error("predictor overlap eligibility or rollback disagrees with its guard");
        for(int i=0;i<6;++i)for(int expert=0;expert<3;++expert) {
            auto original=source(root/paths[i])->read_expert(expert);
            if(sources[i]->gpu_resident(expert))MoeResidencyTestAccess::compare_gpu(*cache,i,expert,original);
            else {
                auto lease=sources[i]->acquire_expert(expert);
                const auto a=moe_cache_fields(lease->weights),b=moe_cache_fields(original);
                for(std::size_t f=0;f<a.size();++f)if(a[f].nbytes()!=b[f].nbytes() ||
                    (a[f].nbytes() && std::memcmp(a[f].data_ptr(),b[f].data_ptr(),a[f].nbytes())))
                    throw std::runtime_error("predictor overlap changed canonical RAM bytes");
            }
        }
        ++cases;
    }
    std::cout<<"predictor_exchange_overlap cases="<<cases
        <<" live-held transaction, actual predictor FFN, source/arena/cache guards and rollback exact PASS\n";
}

void verify_direct_exchange(const std::filesystem::path& root) {
    int cases=0,repeats=0;
    const char* batching_env=mfq::cuda::env_lookup("MFQ_MOE_RESIDENCY_CACHED_REFILL");
    const bool defaults=mfq::cuda::env_lookup("MFQ_TEST_RESIDENCY_EXPECT_DEFAULTS")!=nullptr;
    const bool batching=defaults || (batching_env && std::string(batching_env)=="1");
    const char* shadow=mfq::cuda::env_lookup("MFQ_TEST_RESIDENCY_SHADOW");
    for(const std::string format:{"nint5","nvq3j-512"})
    for(const int cached_projections:{0,1,3})
    for(const bool gpu_wire:{false,true})
    for(const std::string phase:{"none","direct_backup","direct_backup_batch","direct_refill","direct_refill_batch","upload","map","host","publish","reader","window","cancel","prefill"}) {
        if(shadow && phase=="direct_backup_batch")continue;
        if(phase=="direct_refill_batch" && (!batching || (!cached_projections && !gpu_wire)))continue;
        const auto* small_backup=mfq::cuda::env_lookup("MFQ_MOE_RESIDENCY_SMALL_BACKUP_BYTES");
        if(phase=="direct_backup_batch" && small_backup && std::atoll(small_backup)<=0)continue;
        CudaExecutionContext execution;execution.config.moe_residency_warm=true;
        execution.config.moe_residency_projection_heat=true;execution.config.moe_ram_pcie_fraction=1.0;
        execution.config.moe_direct_ram=true;execution.config.moe_two_stage_ffn=true;
        std::array<std::shared_ptr<MoeQuantRangeSource>,3> sources;
        std::array<std::shared_ptr<MfeWeight>,3> weights;int64_t budget=0;
        for(int p=0;p<3;++p) {
            sources[p]=source(root/(format+(p==2?"-2560-640.mfq":"-640-2560.mfq")));
            budget+=bytes(sources[p]->metadata()->pools[0]);
        }
        execution.config.moe_host_cache_bytes=shadow?4*budget+(1<<20):3*budget;
        auto cache=make_moe_expert_cache(budget,execution.config);
        for(int p=0;p<3;++p)weights[p]=std::make_shared<MfeWeight>(cache_quant_moe_weight(
            cache,std::to_string(p),sources[p],1,0,std::to_string(p)));
        auto forward=make_moe_ffn_pipeline({weights[0],weights[1]},{weights[2]});
        finalize_moe_expert_cache(cache);
        if(shadow)MoeResidencyTestAccess::seed_prefill_shadow(*cache,
            std::string(shadow)=="stale",std::string(shadow)=="reclaimed");
        if(gpu_wire) {
            int64_t incoming_bytes=budget;
            for(int p=0;p<cached_projections;++p)incoming_bytes-=bytes(sources[p]->metadata()->pools[0]);
            MoeResidencyTestAccess::seed_exchange_wire(*cache,incoming_bytes);
        }
        if(!MoeResidencyTestAccess::transfer_enabled(*cache))throw std::runtime_error("direct fixture needs transfer caching");
        for(int p=0;p<cached_projections;++p)MoeResidencyTestAccess::seed(*cache,p,1);
        const bool loan_refill=mfq::cuda::env_lookup("MFQ_TEST_RESIDENCY_KV_REFILL")!=nullptr;
        if(loan_refill)MoeResidencyTestAccess::seed_kv_refill(*cache);
        if(defaults)MoeResidencyTestAccess::verify_defaults_guard(*cache);
        MoeHostExpertCache::Lease reader;std::atomic_bool complete=false;
        auto manager=std::make_unique<MoeResidencyManager>(cache.get(),[&](const char* event,std::size_t index) {
            if(std::string(event)=="copies_complete")complete=true;
            if(index)return;
            if(phase==event)throw std::runtime_error("injected direct exchange failure");
            if(phase=="reader" && std::string(event)=="before_direct_exchange")reader=sources[0]->acquire_expert(1);
        });
        if(defaults && !manager->stats().exchange_defaults)throw std::runtime_error("eligible fixture did not enable exchange defaults");
        const bool window=phase=="window" || phase=="cancel";
        if(window) {
            manager->before_layer(0);
            manager->record_window_expert_fence(0,mfq_current_cuda_stream());
            MFQ_CUDA_CHECK(cudaStreamSynchronize(mfq_current_cuda_stream()));
        }
        if(phase=="prefill") {
            auto old_routes=tb::full({32,1},2,tb::TensorOptions().device(tb::kCUDA).dtype(tb::kInt32));
            manager->record_prefill_routes(0,old_routes.data_ptr<int32_t>(),32,1,mfq_current_cuda_stream());
            auto wide_routes=tb::full({32,2},2,tb::TensorOptions().device(tb::kCUDA).dtype(tb::kInt32));
            manager->record_prefill_routes(0,wide_routes.data_ptr<int32_t>(),32,2,mfq_current_cuda_stream());
            auto routes=tb::full({32,1},1,tb::TensorOptions().device(tb::kCUDA).dtype(tb::kInt32));
            manager->record_prefill_routes(0,routes.data_ptr<int32_t>(),32,1,mfq_current_cuda_stream());
            if(manager->stats().committed_projections)throw std::runtime_error("prefill routes exchanged before decode boundary");
            manager->prepare_decode();
            manager->prepare_decode(); // No second transaction without a new prompt.
        } else manager->after_layer(0,std::vector<int32_t>(35,1),35,window);
        if(window) {
            const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(10);
            while(!complete.load()) {
                if(std::chrono::steady_clock::now()>deadline) {
                    manager->finish_window(0);throw std::runtime_error("direct exchange did not reach its publication barrier");
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            if(manager->stats().committed_projections)throw std::runtime_error("direct exchange published before window finish");
            for(const auto& source:sources)if(!source->gpu_resident(0) || source->gpu_resident(1))
                throw std::runtime_error("direct exchange changed early tier ownership");
            if(phase=="cancel")manager.reset();else manager->finish_window(0);
        }
        bool rejected=false;
        if(manager)try {manager->apply_pending();}catch(const std::runtime_error& error) {
            if(std::string(error.what())!="injected direct exchange failure")throw;
            rejected=true;
        }
        const bool injected=phase=="direct_backup" || phase=="direct_backup_batch" || phase=="direct_refill" || phase=="upload" ||
            phase=="direct_refill_batch" || phase=="map" || phase=="host" || phase=="publish";
        if(rejected!=injected)throw std::runtime_error("direct exchange failure was not propagated");
        int resident=(phase=="none" || phase=="window" || phase=="prefill")?1:0;
        if(manager && resident==1 && (!manager->stats().direct_exchange_bytes || manager->stats().direct_exchange_rounds!=1))
            throw std::runtime_error("direct exchange fixture did not execute the candidate");
        if(manager && resident==1 && loan_refill && cached_projections<3 && !manager->stats().kv_refill_bytes)
            throw std::runtime_error("direct exchange did not reuse borrowed KV expert bytes");
        if(manager && resident==1 && shadow) {
            const bool eligible=std::string(shadow)=="valid" && (cached_projections>0 || loan_refill);
            if(bool(manager->stats().shadow_backup_bytes)!=eligible)
                throw std::runtime_error("prefill shadow reuse did not follow identity/cache eligibility");
        }
        if(manager && resident==1 && batching && (cached_projections || gpu_wire) && !manager->stats().cached_copy_batches)
            throw std::runtime_error("direct exchange fixture did not batch cached refills");
        if(resident==1 && gpu_wire && cached_projections<3 && !MoeResidencyTestAccess::wire_payload_used(*cache))
            throw std::runtime_error("compact incoming weights did not use available GPU wire");
        if(phase=="reader" && !reader)throw std::runtime_error("direct exchange did not reject the arriving reader");
        reader.reset();
        const auto compare_all=[&] {
            MoeResidencyTestAccess::check_exchange_wire(*cache);
            for(int p=0;p<3;++p)for(int expert=0;expert<3;++expert) {
                if(sources[p]->gpu_resident(expert)!=(expert==resident))
                    throw std::runtime_error("direct exchange ownership differs after completion or rollback");
                auto original=source(root/(format+(p==2?"-2560-640.mfq":"-640-2560.mfq")))->read_expert(expert);
                if(expert==resident)MoeResidencyTestAccess::compare_gpu(*cache,p,expert,original);
                else {
                    auto held=sources[p]->acquire_expert(expert);
                    const auto actual=moe_cache_fields(held->weights),expected=moe_cache_fields(original);
                    for(std::size_t f=0;f<actual.size();++f) {
                        const auto size=std::size_t(actual[f].numel()*actual[f].element_size());
                        if(size && std::memcmp(actual[f].data_ptr(),expected[f].data_ptr(),size))
                            throw std::runtime_error("direct exchange changed a RAM weight byte");
                    }
                }
                if(sources[p]->expert_disk_reads_after_preload())throw std::runtime_error("direct exchange read sealed storage");
            }
        };
        compare_all();
        if(phase=="none") {
            for(int p=0;p<cached_projections;++p)MoeResidencyTestAccess::seed(*cache,p,2);
            manager->after_layer(0,std::vector<int32_t>(70,2),70);manager->apply_pending();
            resident=2;compare_all();
            if(manager->stats().direct_exchange_rounds!=2)throw std::runtime_error("direct exchange failed to repeat with new fields");
            ++repeats;
        }
        ++cases;
    }
    std::cout<<"direct_exchange_cases="<<cases<<" repeat_cases="<<repeats
        <<" all RAM/GPU bytes, partial backup/refill failures, map/host/publication rollback, reader exclusion, delayed publication and cancellation exact PASS\n";
}

void verify_indexed_shadow_exchange(const std::filesystem::path& root) {
    int cases=0,refreshes=0;
    const auto* option=mfq::cuda::env_lookup("MFQ_MOE_RESIDENCY_PREFILL_SHADOW_INDEX");
    const bool indexed=!option || option[0]!='0';
    for(const std::string format:{"nint5","nvq3j-512"})
    for(int cached_projections:{0,1,3})for(bool gpu_wire:{false,true})
    for(const std::string image:{"permuted","stale","reclaimed"})
    for(const std::string phase:{"none","direct_backup","direct_refill","upload","map","host","publish"}) {
        CudaExecutionContext execution;execution.config.moe_residency_warm=true;
        execution.config.moe_residency_projection_heat=true;execution.config.moe_ram_pcie_fraction=1.0;
        execution.config.moe_direct_ram=true;execution.config.moe_two_stage_ffn=true;
        std::array<std::shared_ptr<MoeQuantRangeSource>,3> sources;
        std::array<std::shared_ptr<MfeWeight>,3> weights;int64_t budget=0;
        const auto filename=[&](int projection) {
            return root/(format+(projection==2?"-2560-640.mfq":"-640-2560.mfq"));
        };
        for(int p=0;p<3;++p){sources[p]=source(filename(p));budget+=bytes(sources[p]->metadata()->pools[0]);}
        execution.config.moe_host_cache_bytes=6*budget+(1<<20);
        auto cache=make_moe_expert_cache(2*budget,execution.config);
        for(int p=0;p<3;++p)weights[p]=std::make_shared<MfeWeight>(cache_quant_moe_weight(
            cache,std::to_string(p),sources[p],1,0,std::to_string(p)));
        auto forward=make_moe_ffn_pipeline({weights[0],weights[1]},{weights[2]});
        finalize_moe_expert_cache(cache);
        for(const auto& source:sources)
            if(!source->gpu_resident(0) || !source->gpu_resident(1) || source->gpu_resident(2))
                throw std::runtime_error("indexed shadow fixture requires two GPU experts");
        MoeResidencyTestAccess::seed_prefill_shadow(*cache,image=="stale",image=="reclaimed",true);
        if(gpu_wire) {
            int64_t incoming=budget;
            for(int p=0;p<cached_projections;++p)incoming-=bytes(sources[p]->metadata()->pools[0]);
            MoeResidencyTestAccess::seed_exchange_wire(*cache,incoming);
        }
        for(int p=0;p<cached_projections;++p)MoeResidencyTestAccess::seed(*cache,p,2);
        MoeResidencyManager manager(cache.get(),[&](const char* event,std::size_t index) {
            if(!index && phase==event)throw std::runtime_error("injected indexed shadow failure");
        });
        bool rejected=false;
        try {manager.after_layer(0,std::vector<int32_t>(70,2),70);manager.apply_pending();}
        catch(const std::runtime_error& error) {
            if(std::string(error.what())!="injected indexed shadow failure")throw;
            rejected=true;
        }
        if(rejected!=(phase!="none"))throw std::runtime_error("indexed shadow failure was not propagated");
        const auto compare=[&] {
            MoeResidencyTestAccess::check_exchange_wire(*cache);
            for(int p=0;p<3;++p)for(int expert=0;expert<3;++expert) {
                auto expected=source(filename(p))->read_expert(expert);
                if(sources[p]->gpu_resident(expert))MoeResidencyTestAccess::compare_gpu(*cache,p,expert,expected);
                else {
                    auto held=sources[p]->acquire_expert(expert);
                    const auto a=moe_cache_fields(held->weights),b=moe_cache_fields(expected);
                    for(std::size_t f=0;f<a.size();++f)if(a[f].nbytes()!=b[f].nbytes() ||
                        (a[f].nbytes() && std::memcmp(a[f].data_ptr(),b[f].data_ptr(),a[f].nbytes())))
                        throw std::runtime_error("indexed shadow changed a RAM payload byte");
                }
                if(sources[p]->expert_disk_reads_after_preload())throw std::runtime_error("indexed shadow read sealed storage");
            }
        };
        compare();
        for(const auto& source:sources)if(source->gpu_resident(2)==rejected)
            throw std::runtime_error("indexed shadow published incorrect GPU ownership");
        if(phase=="none") {
            const bool reuse=indexed && image=="permuted" && cached_projections>0;
            if(bool(manager.stats().shadow_backup_bytes)!=reuse)
                throw std::runtime_error("permuted shadow eligibility/cache dependency mismatch");
            if(image=="permuted") {
                const auto before=manager.stats().shadow_backup_bytes;
                MoeResidencyTestAccess::seed_prefill_shadow(*cache,false,false,true,true);
                const int next=sources[0]->gpu_resident(0)?1:0;
                for(const auto& source:sources)if(source->gpu_resident(next))
                    throw std::runtime_error("indexed shadow refresh needs a common cold expert");
                for(int p=0;p<cached_projections;++p)MoeResidencyTestAccess::seed(*cache,p,next);
                manager.after_layer(0,std::vector<int32_t>(140,next),140);manager.apply_pending();compare();
                if(reuse && manager.stats().shadow_backup_bytes==before)
                    throw std::runtime_error("shadow refresh did not revalidate its identity index");
                MoeResidencyTestAccess::reclaim_shadows(*cache);++refreshes;
            }
        }
        ++cases;
    }
    std::cout<<"indexed_shadow_exchange_cases="<<cases<<" refreshed_and_reclaimed="<<refreshes
        <<" index_enabled="<<indexed<<" all RAM/GPU bytes, identity, refresh, reclamation and rollback exact PASS\n";
}

void verify_prefill_cpu_prefetch(const std::filesystem::path& root) {
    struct Setting {
        std::string previous;
        Setting(){if(const auto* p=mfq::cuda::env_lookup("MFQ_PREFILL_CPU_PREFETCH"))previous=p;}
        static void set(const char* value) {
#ifdef _WIN32
            _putenv_s("MFQ_PREFILL_CPU_PREFETCH",value);
#else
            if(*value)setenv("MFQ_PREFILL_CPU_PREFETCH",value,1);else unsetenv("MFQ_PREFILL_CPU_PREFETCH");
#endif
        }
        ~Setting(){set(previous.c_str());}
    } setting;
    CudaExecutionContext execution;auto& cfg=execution.config;
    cfg.moe_pipeline=true;cfg.moe_preload_all=true;cfg.moe_assert_resident=true;
    cfg.moe_hybrid_cpu=false;cfg.moe_direct_ram=true;cfg.moe_ram_pcie=true;cfg.moe_ram_pcie_fraction=1.;
    const std::array<std::array<std::string,3>,3> formats={
        std::array<std::string,3>{"nint5","nint5","nint5"},
        {"nvq3j-512","nvq3j-512","nvq3j-512"},{"nint6","nvq3j-l","nvq2j-xl"}};
    int64_t unit=0;
    for(int p=0;p<3;++p)unit+=bytes(source(root/(formats[0][p]+(p==2?"-2560-640.mfq":"-640-2560.mfq")))->metadata()->pools[0]);
    cfg.moe_host_cache_bytes=36*unit;
    auto cache=make_moe_expert_cache(3*unit,cfg);
    std::array<MoeFfnForward,3> forward;
    for(int layer=0;layer<3;++layer) {
        std::array<std::shared_ptr<MfeWeight>,3> weights;
        for(int p=0;p<3;++p)weights[p]=std::make_shared<MfeWeight>(cache_quant_moe_weight(cache,
            "cpu-prefetch/"+std::to_string(layer)+"/"+std::to_string(p),
            source(root/(formats[layer][p]+(p==2?"-2560-640.mfq":"-640-2560.mfq"))),1,layer,std::to_string(p)));
        forward[layer]=make_moe_ffn_pipeline({weights[0],weights[1]},{weights[2]});
    }
    finalize_moe_expert_cache(cache);int cases=0;
    for(int tokens:{16,64,257})for(int step=0;step<2;++step) {
        std::vector<float> values(tokens*2560),weights(tokens*3);std::vector<int32_t> routes(tokens*3);
        for(std::size_t i=0;i<values.size();++i)values[i]=std::sin(float(i+step*37)*.017f)*.11f;
        for(std::size_t i=0;i<routes.size();++i){routes[i]=(int(i)+step)%3;weights[i]=.13f+float(i%7)*.01f;}
        auto x=tb::tensor(values).reshape({tokens,2560}).to(tb::kCUDA,tb::kFloat16);
        auto ids=tb::tensor(routes).reshape({tokens,3}).to(tb::kCUDA);
        auto scales=tb::tensor(weights).reshape({tokens,3}).to(tb::kCUDA);
        const bool split=step==1 && tokens>=64;
        const auto run=[&](int layer,bool check_pending) {
            if(!split)return forward[layer](execution,x,ids,scales).clone();
            const int first=tokens/2;
            auto a=forward[layer](execution,x.narrow(0,0,first),ids.narrow(0,0,first),scales.narrow(0,0,first)).clone();
            auto b=forward[layer](execution,x.narrow(0,first,tokens-first),ids.narrow(0,first,tokens-first),
                scales.narrow(0,first,tokens-first)).clone();
            if(check_pending && layer<2 && MoeResidencyTestAccess::prefill_copy_pending(*cache)!=1)
                throw std::runtime_error("same-layer expert batch consumed next-layer prefetch early");
            return tb::cat({a,b},0);
        };
        std::array<tb::Tensor,3> expected;
        Setting::set("0");
        for(int layer=0;layer<3;++layer)expected[layer]=run(layer,false);
        release_moe_prefill_buffers(cache);Setting::set("1");
        const auto before=MoeResidencyTestAccess::prefill_copy_jobs(*cache);
        execution.profiler.enabled=tokens==64 && step==0;
        for(int layer=0;layer<3;++layer) {
            auto actual=run(layer,true).cpu().contiguous();
            auto reference=expected[layer].cpu().contiguous();
            if(actual.sizes()!=reference.sizes() || std::memcmp(actual.data_ptr(),reference.data_ptr(),actual.nbytes()))
                throw std::runtime_error("CPU prefill worker changed an FFN output byte");
            ++cases;
        }
        if(MoeResidencyTestAccess::prefill_copy_jobs(*cache)-before<2)
            throw std::runtime_error("CPU prefill copy worker did not dispatch across three layers");
        if(execution.profiler.enabled) {
            if(execution.profiler.stats.at("prefill.expert_h2d").calls!=3)
                throw std::runtime_error("CPU prefill worker lost background profiling events");
            execution.profiler.report("prefill_cpu_worker_test");execution.profiler.reset();
        }
        execution.profiler.enabled=false;release_moe_prefill_buffers(cache);
        // Cancel with a submitted next layer; cleanup must join its host work
        // and its copy stream before destroying either transfer image.
        (void)forward[0](execution,x,ids,scales);
        if(MoeResidencyTestAccess::prefill_copy_pending(*cache)!=1)
            throw std::runtime_error("prefill cancellation fixture has no pending next layer");
        release_moe_prefill_buffers(cache);
        if(MoeResidencyTestAccess::prefill_copy_pending(*cache))
            throw std::runtime_error("prefill cancellation retained a background copy");
    }
    MoeResidencyTestAccess::verify_prefill_copy_failure_cleanup(*cache);
    std::cout<<"prefill_cpu_prefetch_cases="<<cases
        <<" mixed formats, three-layer slot reuse, changing inputs, profiler merge, cancellation and failure cleanup exact PASS\n";
}
