#include "../runtime/execution_options.h"
#include "moe_cached_source_internal.h"
#include <map>
#include <cmath>

std::shared_ptr<MoeCachedSource> MoeExpertCache::register_source(
        const std::string & name,
        std::shared_ptr<MixedMoeRuntime> cpu,
        int minimum_slots,
        int layer_id,
        std::string projection_role) {
    const int id = static_cast<int>(sources_.size());
    auto source = std::make_shared<MoeCachedSource>(
        this, id, name, std::move(cpu), minimum_slots,
        layer_id, std::move(projection_role), nullptr);
    host_bytes_ += source->host_bytes();
    sources_.push_back(source);
    return source;
}

std::shared_ptr<MoeCachedSource> MoeExpertCache::register_range_source(
        const std::string & name,
        std::shared_ptr<MixedMoeRuntime> metadata,
        std::shared_ptr<mfq::cuda::MfeMxfp4ExpertStore> store,
        int minimum_slots,
        int layer_id,
        std::string projection_role) {
    const int id = static_cast<int>(sources_.size());
    auto source = std::make_shared<MoeCachedSource>(
        this, id, name, std::move(metadata), minimum_slots,
        layer_id, std::move(projection_role), std::move(store));
    host_bytes_ += source->host_bytes();
    sources_.push_back(source);
    return source;
}

std::shared_ptr<MoeCachedSource> MoeExpertCache::register_quant_range_source(
        const std::string& name,std::shared_ptr<MoeQuantRangeSource> range,
        int minimum_slots,int layer_id,std::string projection_role) {
    const int id=static_cast<int>(sources_.size());
    range->bind_host_cache(host_experts_,id);
    auto runtime=range->metadata();
    auto source=std::make_shared<MoeCachedSource>(this,id,name,std::move(runtime),minimum_slots,
        layer_id,std::move(projection_role),nullptr,std::move(range));
    host_bytes_+=source->host_bytes();
    sources_.push_back(source);
    return source;
}

void MoeExpertCache::finalize() {
    if (finalized_) return;
    if (sources_.empty()) {
        throw std::runtime_error(
            "MoE cache has no registered expert sources");
    }
    const auto cuda_context=mfq::cuda::default_context();
    if(cuda_context->memory_stats().limit)cuda_context->stream().synchronize();
    const auto memory=cuda_context->memory_stats();
    if(memory.limit) {
        const auto reserve=mfq::cuda::runtime_options::workspace_reserve_bytes();
        const auto model_bytes=std::max(memory.allocated,cuda_context->local_memory_usage());
        if(model_bytes>=memory.limit || reserve>=memory.limit-model_bytes)
            throw std::runtime_error("CUDA VRAM limit leaves no room for expert arenas");
        const auto available=memory.limit-model_bytes-reserve;
        const auto requested=budget_bytes_;
        budget_bytes_=static_cast<int64_t>(std::min<std::size_t>(static_cast<std::size_t>(budget_bytes_),available));
        std::cerr<<"moe_cache_vram_limit limit_bytes="<<memory.limit<<" model_bytes="<<model_bytes
            <<" workspace_reserve_bytes="<<reserve<<" requested_expert_bytes="<<requested
            <<" expert_budget_bytes="<<budget_bytes_<<std::endl;
    }
    std::vector<mfq::MoeArenaDemand> demands;
    demands.reserve(arenas_.size());
    for (const auto & item : arenas_) {
        const auto & arena = *item.second;
        demands.push_back({
            arena.signature,
            arena.slot_bytes,
            arena.minimum_slots,
            arena.registered_experts,
        });
    }

    if (profile_.has_value()) {
        std::unordered_map<
            int, std::vector<std::shared_ptr<MoeCachedSource>>>
            layer_sources;
        std::unordered_map<int, int> layer_experts;
        std::unordered_map<int, std::unordered_set<std::string>>
            layer_roles;
        for (const auto & source : sources_) {
            if (source->layer_id() < 0) continue;
            if (source->projection_role().empty()) {
                throw std::runtime_error(
                    "profiled MoE cache source has no projection role");
            }
            if (!layer_roles[source->layer_id()]
                     .insert(source->projection_role()).second) {
                throw std::runtime_error(
                    "profiled MoE cache layer has a duplicate projection role");
            }
            const auto inserted = layer_experts.emplace(
                source->layer_id(), source->n_experts());
            if (!inserted.second &&
                    inserted.first->second != source->n_experts()) {
                throw std::runtime_error(
                    "profiled MoE cache layer sources disagree on expert count");
            }
            layer_sources[source->layer_id()].push_back(source);
        }
        mfq::validate_moe_cache_profile(
            *profile_, layer_experts);

        std::unordered_map<std::string, int> required_slots;
        std::unordered_map<std::string, int> selected_slots;
        int64_t required_bytes = 0;
        for (const auto & demand : demands) {
            required_slots[demand.signature] =
                demand.minimum_slots;
            required_bytes +=
                static_cast<int64_t>(demand.minimum_slots) *
                demand.slot_bytes;
        }
        if (required_bytes > budget_bytes_) {
            throw std::runtime_error(
                "MoE cache budget is below the minimum decode working set");
        }

        auto bundle_bytes = [&](int layer, int expert) {
            int64_t bytes = 0;
            for (const auto & source : layer_sources.at(layer)) {
                const int64_t source_bytes =
                    source->bytes_for_expert(expert);
                if (source_bytes >
                        std::numeric_limits<int64_t>::max() - bytes) {
                    throw std::overflow_error(
                        "MoE cache prewarm bundle byte count overflows int64");
                }
                bytes += source_bytes;
            }
            return bytes;
        };

        auto select_candidate = [&](
                const mfq::MoeProfileCandidate & candidate,
                int64_t layer_limit,
                std::unordered_map<int, int64_t> * layer_used) {
            const auto source_it =
                layer_sources.find(candidate.layer);
            if (source_it == layer_sources.end() ||
                    source_it->second.empty()) {
                return false;
            }
            std::unordered_map<std::string, int> needed;
            for (const auto & source : source_it->second) {
                ++needed[
                    source->arena_for_expert(
                        candidate.expert)->signature];
            }
            int64_t incremental_bytes = 0;
            for (const auto & item : needed) {
                const auto arena_it = arenas_.find(item.first);
                if (arena_it == arenas_.end()) {
                    throw std::runtime_error(
                        "MoE cache profile references an unknown arena");
                }
                const auto & arena = *arena_it->second;
                const int selected =
                    selected_slots[item.first] + item.second;
                if (selected > arena.registered_experts) {
                    return false;
                }
                const int new_required = std::max(
                    required_slots[item.first], selected);
                incremental_bytes +=
                    static_cast<int64_t>(
                        new_required -
                        required_slots[item.first]) *
                    arena.slot_bytes;
            }
            const int64_t bytes =
                bundle_bytes(candidate.layer, candidate.expert);
            if (layer_used != nullptr &&
                    bytes >
                        layer_limit -
                        (*layer_used)[candidate.layer]) {
                return false;
            }
            if (incremental_bytes >
                    budget_bytes_ - required_bytes) {
                return false;
            }
            for (const auto & item : needed) {
                selected_slots[item.first] += item.second;
                required_slots[item.first] = std::max(
                    required_slots[item.first],
                    selected_slots[item.first]);
            }
            required_bytes += incremental_bytes;
            if (layer_used != nullptr) {
                (*layer_used)[candidate.layer] += bytes;
            }
            prewarm_selected_.push_back(candidate);
            return true;
        };

        const auto candidates =
            mfq::order_moe_profile_candidates(
                *profile_, false);
        int64_t frequency_bytes = 0;
        std::unordered_set<int> ranking_layers;
        for (const auto & candidate : candidates) {
            if (candidate.has_frequency) {
                if (select_candidate(
                        candidate,
                        std::numeric_limits<int64_t>::max(),
                        nullptr)) {
                    frequency_bytes +=
                        bundle_bytes(
                            candidate.layer,
                            candidate.expert);
                }
            } else {
                ranking_layers.insert(candidate.layer);
            }
        }

        std::unordered_map<int, int64_t> rank_layer_total;
        int64_t rank_model_total = 0;
        for (int layer : ranking_layers) {
            int64_t total = 0;
            for (int expert = 0;
                 expert < layer_experts.at(layer);
                 ++expert) {
                total += bundle_bytes(layer, expert);
            }
            rank_layer_total[layer] = total;
            rank_model_total += total;
        }
        const int64_t rank_budget =
            std::max<int64_t>(
                0, budget_bytes_ - frequency_bytes);
        std::unordered_map<int, int64_t> rank_layer_limit;
        for (const auto & item : rank_layer_total) {
            const long double fraction =
                rank_model_total > 0
                ? static_cast<long double>(item.second) /
                    static_cast<long double>(rank_model_total)
                : 0.0L;
            rank_layer_limit[item.first] =
                static_cast<int64_t>(
                    static_cast<long double>(rank_budget) *
                    fraction);
        }
        std::unordered_map<int, int64_t> rank_layer_used;
        for (const auto & candidate : candidates) {
            if (candidate.has_frequency) continue;
            (void)select_candidate(
                candidate,
                rank_layer_limit.at(candidate.layer),
                &rank_layer_used);
        }

        for (auto & demand : demands) {
            demand.minimum_slots = std::max(
                demand.minimum_slots,
                required_slots.at(demand.signature));
        }
    }
    const auto plan =
        mfq::plan_moe_arena_slots(budget_bytes_, demands);
    for (auto & item : arenas_) {
        auto & arena = *item.second;
        arena.slots = plan.at(arena.signature);
        arena.fields.reserve(arena.layouts.size());
        for (size_t index = 0;
             index < arena.layouts.size();
             ++index) {
            const auto & layout = arena.layouts[index];
            if (layout.slot_shape.empty()) {
                throw std::runtime_error(
                    "MoE cache field has no expert-major leading dimension");
            }
            auto shape = layout.slot_shape;
            shape[0] *= arena.slots;
            arena.fields.push_back(mfq_tensor_backend::empty(
                shape,
                mfq_tensor_backend::TensorOptions()
                    .device(mfq_tensor_backend::kCUDA)
                    .dtype(layout.scalar_type)));
        }
        arena.book =
            std::make_unique<mfq::MoeCacheSlotBook>(
                arena.slots);
        allocated_bytes_ +=
            static_cast<int64_t>(arena.slots) *
            arena.slot_bytes;
        std::cerr
            << "moe_cache_arena"
            << " signature=" << arena.signature
            << " slots=" << arena.slots
            << " slot_bytes=" << arena.slot_bytes
            << " bytes="
            << static_cast<int64_t>(arena.slots) *
                arena.slot_bytes
            << std::endl;
    }
    if (allocated_bytes_ > budget_bytes_) {
        throw std::runtime_error(
            "MoE cache arena allocation exceeded its budget");
    }
    for (auto & source : sources_) source->finalize();
    finalized_ = true;
    std::cerr
        << "moe_cache_ready"
        << " sources=" << sources_.size()
        << " host_bytes=" << host_bytes_
        << " budget_bytes=" << budget_bytes_
        << " allocated_bytes=" << allocated_bytes_
        << " mapped_gather=" << (mapped_gather_enabled_ ? 1 : 0)
        << " mapped_registered_bytes=" << mapped_registered_bytes_
        << " mapped_copy_blocks=" << mapped_copy_blocks_
        << std::endl;
    calibrate_pipeline_pcie();
    if (profile_.has_value()) prewarm();
    if (config_.moe_preload_all) preload_complete_residency();
}

void MoeExpertCache::preload_complete_residency() {
    if (!finalized_) throw std::logic_error("expert residency requires finalized GPU arenas");
    const auto started=std::chrono::steady_clock::now();
    std::uint64_t total_bytes=0,total_experts=0;
    for (const auto& source:sources_) {
        if (!source->quant_source_) throw std::runtime_error("complete preload requires canonical quantized expert sources");
        for (int expert=0; expert<source->n_experts(); ++expert) {
            total_bytes+=source->quant_source_->expert_field_bytes(expert);
            ++total_experts;
        }
    }
    const auto cold_bytes=total_bytes-std::min<std::uint64_t>(total_bytes,allocated_bytes_);
    const auto host_budget=host_experts_->stats().budget_bytes;
    std::cerr << "moe_residency_plan experts=" << total_experts << " runtime_bytes=" << total_bytes
        << " gpu_bytes=" << allocated_bytes_ << " cold_bytes=" << cold_bytes
        << " host_field_budget_bytes=" << host_budget << std::endl;
    if (cold_bytes>host_budget) throw std::runtime_error("complete expert residency exceeds RAM field budget");
    // Fill shared GPU arenas fairly across sources, preserving profile-hot
    // slots. No initial placement may replace an already valid GPU expert.
    std::unordered_map<MoeGpuArena*,int> remaining;
    for (const auto& item:arenas_) remaining[item.second.get()]=item.second->slots-item.second->book->size();
    std::vector<std::vector<bool>> warm(sources_.size());
    int maximum=0;
    for (const auto& source:sources_) {
        warm[source->id_].resize(source->n_experts());
        maximum=std::max(maximum,source->n_experts());
    }
    for (int expert=0; expert<maximum; ++expert) for (const auto& source:sources_) {
        if (expert>=source->n_experts() || source->quant_source_->gpu_resident(expert)) continue;
        auto* arena=source->cohorts_[source->expert_to_cohort_[expert]].arena;
        if (remaining[arena]>0) { warm[source->id_][expert]=true; --remaining[arena]; }
    }
    for (const auto& item:remaining) if (item.second) throw std::logic_error("expert residency left allocatable GPU slots unfilled");
    std::vector<std::vector<int64_t>> ram_offsets(sources_.size());
    std::map<int,std::vector<MoeCachedSource*>> layer_sources;
    for(const auto& source:sources_) {
        ram_offsets[source->id_].assign(source->n_experts(),-1);
        layer_sources[source->layer_id_].push_back(source.get());
    }
    int64_t ram_allocated=0;
    const auto layer_option=mfq::cuda::runtime_options::moe_prefill_layer();
    pipeline_prefill_layer_layout_=layer_option && *layer_option!=0;
    std::vector<std::vector<std::vector<int64_t>>> ram_field_offsets;
    if(pipeline_prefill_layer_layout_) {
        ram_field_offsets.resize(sources_.size());
        for(const auto& source:sources_)ram_field_offsets[source->id_].resize(source->n_experts());
        // Loader role strings can name the model architecture. The registered
        // FFN bundle defines Down independently of those strings and load order.
        std::vector<bool> down_sources(sources_.size(),false);
        for(const auto& bundle:pipeline_bundles_)down_sources.at(bundle[2])=true;
        for(auto& layer:layer_sources) {
            ram_allocated=(ram_allocated+255)&~int64_t(255);
            const auto begin=ram_allocated;
            const auto down=[&](const MoeCachedSource* source) {
                return down_sources.at(source->id_);
            };
            std::stable_sort(layer.second.begin(),layer.second.end(),[&](const auto* a,const auto* b) {
                return down(a)<down(b);
            });
            int64_t gate_up_end=-1;
            for(auto* source:layer.second)for(auto& cohort:source->cohorts_) {
                if(down(source) && gate_up_end<0)gate_up_end=ram_allocated-begin;
                for(int expert=0;expert<source->n_experts();++expert)
                    if(source->expert_to_cohort_[expert]==cohort.index &&
                            !source->quant_source_->gpu_resident(expert) && !warm[source->id_][expert]) {
                        cohort.prefill_ram_experts.push_back(expert);
                        ram_field_offsets[source->id_][expert].resize(cohort.bytes_per_expert.size());
                    }
                for(std::size_t f=0;f<cohort.bytes_per_expert.size();++f) {
                    ram_allocated=(ram_allocated+255)&~int64_t(255);
                    cohort.prefill_ram_fields.push_back(ram_allocated);
                    for(const auto expert:cohort.prefill_ram_experts) {
                        ram_field_offsets[source->id_][expert][f]=ram_allocated;
                        if(f==0)ram_offsets[source->id_][expert]=ram_allocated;
                        if(cohort.bytes_per_expert[f]>std::numeric_limits<int64_t>::max()-ram_allocated)
                            throw std::overflow_error("RAM layer size overflow");
                        ram_allocated+=cohort.bytes_per_expert[f];
                    }
                }
            }
            ram_allocated=(ram_allocated+255)&~int64_t(255);
            pipeline_prefill_layers_[layer.first]={begin,ram_allocated-begin};
            pipeline_prefill_layer_gate_up_bytes_[layer.first]=gate_up_end<0?ram_allocated-begin:gate_up_end;
        }
    }
    // Keep each expert's missing projection fields together. CUDA can transfer
    // an immutable RAM interval without first copying it to another host arena.
    if(!pipeline_prefill_layer_layout_)for(const auto& layer:layer_sources) {
        int count=0;for(const auto* source:layer.second)count=std::max(count,source->n_experts());
        for(int expert=0;expert<count;++expert)for(const auto* source:layer.second) {
            if(expert>=source->n_experts())continue;
            if(source->quant_source_->gpu_resident(expert) || warm[source->id_][expert])continue;
            const auto& cohort=source->cohorts_[source->expert_to_cohort_[expert]];
            ram_allocated=(ram_allocated+15)&~int64_t(15);ram_offsets[source->id_][expert]=ram_allocated;
            for(const auto count:cohort.bytes_per_expert) {
                ram_allocated=(ram_allocated+15)&~int64_t(15);
                if(count>std::numeric_limits<int64_t>::max()-ram_allocated)throw std::overflow_error("RAM complement size overflow");
                ram_allocated+=count;
            }
        }
    }
    pipeline_ram_complement_=std::make_shared<mfq_tensor_backend::Tensor>(mfq_tensor_backend::empty(
        {ram_allocated},mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCPU).dtype(mfq_tensor_backend::kUInt8)));
    host_experts_->preserve_resident_experts();
    measuring_preload_h2d_=true;
    std::uint64_t completed=0;
    for (const auto& source:sources_) {
        auto& quant=*source->quant_source_;
        for (std::size_t pool=0; pool<quant.store().pool_count(); ++pool) {
            std::vector<std::int32_t> upload;
            quant.store().visit_pool_experts(pool,[&](int expert,mfq::MfeQuantExpert encoded) {
                if (quant.gpu_resident(expert)) { ++completed; return; }
                const auto key=quant.host_key(expert);
                auto held=host_experts_->acquire(key,quant.expert_field_bytes(expert),[&] {
                    auto decoded=quant.decode_expert(expert,encoded);
                    auto offset=ram_offsets[source->id_][expert];
                    if(offset<0)return decoded;
                    auto fields=moe_cache_fields(decoded);
                    for(std::size_t f=0;f<fields.size();++f) {
                        auto& field=fields[f];
                        offset=pipeline_prefill_layer_layout_ ? ram_field_offsets[source->id_][expert][f]
                                                             : (offset+15)&~int64_t(15);
                        auto view=moe_owned_host_view(pipeline_ram_complement_,static_cast<std::size_t>(offset),field);
                        const auto count=tensor_nbytes(field);
                        std::memcpy(view.data_ptr(),field.data_ptr(),static_cast<std::size_t>(count));
                        offset+=count;field=std::move(view);
                    }
                    return moe_replace_quant_fields(std::move(decoded),fields);
                });
                if (!host_experts_->contains(key))
                    throw std::runtime_error("preloaded expert was not retained in RAM");
                if (warm[source->id_][expert]) upload.push_back(expert);
                ++completed;
            });
            if (!upload.empty() && !prepare(*source,upload,true))
                throw std::logic_error("expert preload GPU placement exceeded arena capacity");
        }
        const auto seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
        std::cerr << "moe_residency_progress experts=" << completed << '/' << total_experts
            << " host_bytes=" << host_experts_->stats().resident_bytes
            << " elapsed_s=" << seconds << " experts_s=" << completed/seconds << std::endl;
    }
    if (transfer_ready_recorded_) MFQ_CUDA_CHECK(cudaEventSynchronize(transfer_ready_));
    std::uint64_t gpu_experts=0,ram_experts=0;
    for (const auto& source:sources_) {
        auto& quant=*source->quant_source_;
        for (int expert=0; expert<source->n_experts(); ++expert) {
            if (quant.gpu_resident(expert)) ++gpu_experts;
            else if (host_experts_->contains(quant.host_key(expert))) ++ram_experts;
            else throw std::logic_error("expert has no RAM or VRAM copy after preload");
        }
        quant.mark_preload_complete(config_.moe_assert_resident);
        initialize_mixed_nvq_dispatch(*source->active_,config_);
    }
    measuring_preload_h2d_=false;
    std::int64_t h2d_bytes=0;
    double h2d_ms=0;
    for (const auto& timing:preload_h2d_timings_) {
        float milliseconds=0;
        MFQ_CUDA_CHECK(cudaEventElapsedTime(&milliseconds,timing->begin,timing->end));
        h2d_bytes+=timing->bytes; h2d_ms+=milliseconds;
    }
    preload_h2d_timings_.clear();
    if(config_.moe_pipeline && config_.moe_direct_ram && ram_allocated) {
        registered_host_fields_.reserve(registered_host_fields_.size()+1);
        auto* address=pipeline_ram_complement_->data_ptr();
        MFQ_CUDA_CHECK(cudaHostRegister(address,static_cast<std::size_t>(ram_allocated),cudaHostRegisterMapped));
        registered_host_fields_.push_back({address,ram_allocated,true});
        pipeline_ram_registered_bytes_=ram_allocated;
    }
    std::cerr << "moe_residency_h2d bytes=" << h2d_bytes << " cuda_ms=" << h2d_ms << std::endl;
    std::uint64_t dense_materializations=0;
    for(const auto& source:sources_)dense_materializations+=source->quant_source_->dense_materializations();
    const auto host=host_experts_->stats();
    complete_residency_=true;
    std::cerr << "moe_residency_ready experts=" << total_experts << " gpu_experts=" << gpu_experts
        << " ram_experts=" << ram_experts << " ram_bytes=" << host.resident_bytes
        << " ram_arena_bytes=" << ram_allocated
        << " ram_registered_bytes=" << pipeline_ram_registered_bytes_
        << " prefill_layer_layout=" << pipeline_prefill_layer_layout_
        << " residency_fixed_prefill_layout=" << pipeline_prefill_layer_layout_
        << " nvq_dense_materializations=" << dense_materializations
        << " disk_reads_sealed=" << sources_.front()->quant_source_->expert_disk_sealed() << " elapsed_s="
        << std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count() << std::endl;
    stats_={};
}

std::uint64_t MoeExpertCache::expert_disk_reads_after_preload() const {
    std::uint64_t reads=0;
    for (const auto& source:sources_) if (source->quant_source_)
        reads+=source->quant_source_->expert_disk_reads_after_preload();
    return reads;
}

void MoeExpertCache::calibrate_pipeline_pcie() {
    if(!config_.moe_pipeline && !config_.moe_ram_pcie)return;
    if(config_.moe_ram_pcie_fraction) {
        const auto fraction=*config_.moe_ram_pcie_fraction;
        ram_pcie_fraction_=fraction;std::cerr<<"moe_ram_pcie_override fraction="<<fraction<<std::endl;return;
    }
    namespace tb=mfq_tensor_backend;
    tb::Tensor target;
    for(const auto& [signature,arena]:arenas_)for(const auto& field:arena->fields)
        if(!target.defined() || tensor_nbytes(field)>tensor_nbytes(target))target=field;
    const auto bytes=std::min<int64_t>(256ll<<20,tensor_nbytes(target));
    if(!bytes)throw std::runtime_error("MFQ PCIe calibration has no allocated expert arena");
    const MfqCudaGuard guard(target.device());
    auto host=tb::empty({bytes},tb::TensorOptions().device(tb::kCPU).dtype(tb::kUInt8).pinned_memory(true));
    std::memset(host.data_ptr(),0,static_cast<std::size_t>(bytes));
    const auto stream=weight_stream_;
    struct Events {
        cudaEvent_t values[5]{};
        ~Events(){for(auto e:values)if(e)cudaEventDestroy(e);}
    } events;
    for(auto& e:events.values)MFQ_CUDA_CHECK(cudaEventCreate(&e));
    MFQ_CUDA_CHECK(cudaMemcpyAsync(target.data_ptr(),host.data_ptr(),static_cast<std::size_t>(bytes),cudaMemcpyHostToDevice,stream));
    MFQ_CUDA_CHECK(cudaEventRecord(events.values[0],stream));
    for(int burst=0;burst<4;++burst) {
        MFQ_CUDA_CHECK(cudaMemcpyAsync(target.data_ptr(),host.data_ptr(),static_cast<std::size_t>(bytes),cudaMemcpyHostToDevice,stream));
        MFQ_CUDA_CHECK(cudaEventRecord(events.values[burst+1],stream));
    }
    MFQ_CUDA_CHECK(cudaEventSynchronize(events.values[4]));
    std::cerr<<"moe_ram_pcie_calibration burst_bytes="<<bytes<<" gbps_samples=";
    for(int burst=0;burst<4;++burst) {
        float milliseconds=0;MFQ_CUDA_CHECK(cudaEventElapsedTime(&milliseconds,events.values[burst],events.values[burst+1]));
        if(!(milliseconds>0))throw std::runtime_error("MFQ PCIe calibration interval is empty");
        const auto gbps=bytes/milliseconds/1e6;pipeline_pcie_gbps_=std::max(pipeline_pcie_gbps_,gbps);
        std::cerr<<(burst ? "," : "")<<gbps;
    }
    ram_pcie_fraction_=mfq::measured_pcie_share(pipeline_pcie_gbps_);
    std::cerr<<" best_gbps="<<pipeline_pcie_gbps_<<" fraction="<<ram_pcie_fraction_<<std::endl;
}

void MoeExpertCache::prewarm() {
    if (!finalized_ || !profile_.has_value()) return;

    std::unordered_map<
        int, std::vector<std::shared_ptr<MoeCachedSource>>>
        layer_sources;
    for (const auto & source : sources_) {
        if (source->layer_id() < 0) continue;
        layer_sources[source->layer_id()].push_back(source);
    }

    const auto started = std::chrono::steady_clock::now();
    std::unordered_map<
        MoeCachedSource *, std::vector<int32_t>>
        source_experts;
    for (auto selected_it = prewarm_selected_.rbegin();
         selected_it != prewarm_selected_.rend();
         ++selected_it) {
        for (const auto & source :
             layer_sources.at(selected_it->layer)) {
            source_experts[source.get()].push_back(
                selected_it->expert);
        }
    }
    prewarming_ = true;
    try {
        for (const auto & source : sources_) {
            const auto found = source_experts.find(source.get());
            if (found == source_experts.end()) continue;
            prepare(*source, found->second, true);
        }
        if (transfer_ready_recorded_) {
            MFQ_CUDA_CHECK(
                cudaEventSynchronize(transfer_ready_));
        }
        prewarming_ = false;
    } catch (...) {
        prewarming_ = false;
        throw;
    }
    for (auto selected_it = prewarm_selected_.rbegin();
         selected_it != prewarm_selected_.rend();
         ++selected_it) {
        for (const auto & source :
             layer_sources.at(selected_it->layer)) {
            source->touch_expert(
                selected_it->expert);
        }
    }
    const auto stopped = std::chrono::steady_clock::now();
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(
            stopped - started).count();
    const int64_t prewarm_h2d_bytes = stats_.h2d_bytes;
    const int64_t prewarm_range_read_bytes = stats_.range_read_bytes;
    const int64_t prewarm_range_read_calls = stats_.range_read_calls;
    const int64_t prewarm_range_file_opens = stats_.range_file_opens;
    const double prewarm_range_read_ms =
        static_cast<double>(stats_.range_read_nanoseconds) / 1.0e6;
    const int64_t projection_entries =
        stats_.prefetch_misses;
    const int64_t unfilled_bytes =
        std::max<int64_t>(
            0, allocated_bytes_ - prewarm_h2d_bytes);
    std::cout
        << "moe_cache_prewarm"
        << " profile=" << profile_->path
        << " expert_bundles=" << prewarm_selected_.size()
        << " projection_entries=" << projection_entries
        << " h2d_bytes=" << prewarm_h2d_bytes
        << " range_read_bytes=" << prewarm_range_read_bytes
        << " range_read_calls=" << prewarm_range_read_calls
        << " range_file_opens=" << prewarm_range_file_opens
        << " range_read_ms=" << prewarm_range_read_ms
        << " time_ms=" << elapsed_ms
        << " unfilled_bytes=" << unfilled_bytes
        << "\n";
    stats_ = {};
}

void MoeExpertCache::invalidate(
        const mfq::MoeCacheKey & key,
        int slot,bool retain_cold) {
    if (key.source < 0 ||
        key.source >= static_cast<int>(sources_.size())) {
        throw std::runtime_error(
            "MoE cache eviction references an invalid source");
    }
    auto& source=*sources_.at(static_cast<size_t>(key.source));
    source.invalidate(key.cohort,key.expert,slot);
    if (!source.quant_source_) return;
    source.quant_source_->mark_gpu_resident(key.expert,false);
    if (!retain_cold) return; // A rolled-back slot may contain incomplete fields.
    auto& cohort=source.cohorts_.at(key.cohort);
    auto& arena=*cohort.arena;
    bool copied=false;
    const bool retained=source.quant_source_->demote_expert(key.expert,[&] {
        // An unprotected lease can still have DMA or GPU kernels in flight.
        // Copy the victim only after both producer streams have finished.
        const auto stream=mfq_get_current_cuda_stream().stream();
        if (transfer_ready_recorded_) MFQ_CUDA_CHECK(cudaStreamWaitEvent(stream,transfer_ready_,0));
        if (compute_done_recorded_) MFQ_CUDA_CHECK(cudaStreamWaitEvent(stream,compute_done_,0));
        auto pool=*cohort.cpu;
        pool.local_experts=1;
        pool.expert_local={};
        std::vector<mfq_tensor_backend::Tensor> fields;
        for (std::size_t i=0; i<arena.fields.size(); ++i) {
            const auto rows=arena.layouts[i].slot_shape[0];
            fields.push_back(arena.fields[i].narrow(0,slot*rows,rows).to(mfq_tensor_backend::kCPU).contiguous());
        }
        if (pool.family==MixedMoeFamily::Nint) {
            pool.nint.q_packed=fields[0]; pool.nint.row_q_bits=fields[1]; pool.nint.row_q_bit_offsets=fields[2];
            pool.nint.sub_scale=fields[3]; pool.nint.sub_min=fields[4];
            pool.nint.neuron_scale=fields[5]; pool.nint.neuron_min=fields[6];
        } else {
            pool.nvq.indices_packed=fields[0]; pool.nvq.aux_packed=fields[1];
            pool.nvq.sub_scale_packed=fields[2]; pool.nvq.neuron_scale=fields[3];
        }
        copied=true;
        return pool;
    });
    if (complete_residency_ && !retained) throw std::runtime_error("GPU eviction would lose a resident expert");
    if (copied) stats_.gpu_demote_bytes+=arena.slot_bytes;
}

void MoeExpertCache::publish_quant_promotions(const std::vector<MoeCacheNewLease>& leases) {
    for (const auto& lease:leases) {
        auto& source=*sources_.at(lease.key.source);
        if (source.quant_source_) source.quant_source_->mark_gpu_resident(lease.key.expert,true);
    }
}

void MoeExpertCache::append_source_transfers(
        MoeCachedSource & source,
        const std::vector<int32_t> & experts,
        bool prefetch,
        std::vector<MoeCacheTransfer> & transfers,
        bool & replaced_occupied,
        std::vector<std::pair<mfq::MoeCacheSlotBook *, int>> * held_slots,
        std::vector<MoeCacheNewLease> * new_leases) {
    for (int expert : experts) {
        const int cohort_index =
            source.expert_to_cohort_.at(
                static_cast<size_t>(expert));
        const int local_index =
            source.expert_to_local_.at(
                static_cast<size_t>(expert));
        auto & cohort =
            source.cohorts_.at(
                static_cast<size_t>(cohort_index));
        auto & arena = *cohort.arena;
        const mfq::MoeCacheKey key{
            source.id_, cohort_index, expert};
        auto lease = arena.book->acquire(key);
        if (held_slots != nullptr &&
                !arena.book->inflight(lease.slot)) {
            arena.book->mark_inflight(lease.slot);
            held_slots->emplace_back(arena.book.get(), lease.slot);
        }
        if (lease.hit) {
            if (prefetch) {
                ++stats_.prefetch_hits;
            } else {
                ++stats_.demand_hits;
            }
        } else {
            if (new_leases != nullptr) {
                new_leases->push_back({
                    arena.book.get(),
                    key,
                    lease.slot,
                    lease.generation,
                });
            }
            if (prefetch) {
                ++stats_.prefetch_misses;
            } else {
                ++stats_.demand_misses;
            }
            if (lease.replaced.has_value()) {
                ++stats_.evictions;
                replaced_occupied = true;
                invalidate(*lease.replaced, lease.slot);
            }
            std::vector<mfq_tensor_backend::Tensor> quant_fields;
            MoeHostExpertCache::Lease quant_owner;
            if (source.quant_source_) {
                quant_owner=source.quant_source_->acquire_expert(expert);
                quant_fields=moe_cache_fields(quant_owner->weights);
            }
            for (size_t field = 0;
                 field < cohort.bytes_per_expert.size();
                 ++field) {
                const int64_t nbytes =
                    cohort.bytes_per_expert[field];
                if (nbytes == 0) continue;
                auto & gpu_field =
                    arena.fields[field];
                if (source.quant_source_) {
                    const auto& owned=quant_fields.at(field);
                    if (tensor_nbytes(owned)!=nbytes || !owned.is_cpu() || !owned.is_contiguous())
                        throw std::runtime_error("quantized expert fields exceed registered cache layout");
                    transfers.push_back({reinterpret_cast<const std::uint8_t*>(owned.data_ptr()),
                        reinterpret_cast<std::uint8_t*>(gpu_field.data_ptr())+static_cast<std::int64_t>(lease.slot)*nbytes,
                        nbytes,true,nullptr,nullptr,nullptr,quant_owner});
                    continue;
                }
                if (cohort.range_store) {
                    const auto & part =
                        cohort.range_store->part(expert, field);
                    transfers.push_back({
                        nullptr,
                        reinterpret_cast<uint8_t *>(
                            gpu_field.data_ptr()) +
                            static_cast<int64_t>(lease.slot) * nbytes,
                        nbytes,
                        true,
                        nullptr,
                        cohort.range_store.get(),
                        &part,
                    });
                    continue;
                }
                const auto & cpu_field =
                    cohort.cpu_fields[field];
                transfers.push_back({
                    reinterpret_cast<const uint8_t *>(
                        cpu_field.data_ptr()) +
                        static_cast<int64_t>(local_index) *
                        nbytes,
                    reinterpret_cast<uint8_t *>(
                        gpu_field.data_ptr()) +
                        static_cast<int64_t>(lease.slot) *
                        nbytes,
                    nbytes,
                    true,
                    cohort.mapped_fields.at(field) == nullptr
                        ? nullptr
                        : cohort.mapped_fields.at(field) +
                            static_cast<int64_t>(local_index) * nbytes,
                });
            }
        }
        if (cohort.host_map[
                static_cast<size_t>(expert)] != lease.slot) {
            cohort.host_map[
                static_cast<size_t>(expert)] = lease.slot;
            cohort.map_dirty = true;
        }
    }

    for (auto & cohort : source.cohorts_) {
        if (!cohort.map_dirty) continue;
        auto & active =
            source.active_->pools.at(
                static_cast<size_t>(cohort.index));
        transfers.push_back({
            reinterpret_cast<const uint8_t *>(
                cohort.host_map.data()),
            reinterpret_cast<uint8_t *>(
                active.expert_local.data_ptr<int32_t>()),
            static_cast<int64_t>(
                cohort.host_map.size() *
                sizeof(int32_t)),
            false,
        });
        cohort.map_dirty = false;
    }
}

void MoeExpertCache::rollback_preparation(
        const std::vector<MoeCacheNewLease> & new_leases,
        const std::vector<
            std::pair<mfq::MoeCacheSlotBook *, int>> & held_slots) noexcept {
    (void)cudaStreamSynchronize(weight_stream_);
    for (auto lease = new_leases.rbegin();
         lease != new_leases.rend();
         ++lease) {
        try {
            invalidate(lease->key, lease->slot,false);
            (void)lease->book->discard(
                lease->key, lease->slot, lease->generation);
        } catch (...) {
        }
    }
    for (const auto & held : held_slots) {
        try {
            held.first->clear_inflight(held.second);
        } catch (...) {
        }
    }
}

bool MoeExpertCache::begin_deferred_range_read(
        MoeCachedSource & source,
        const std::vector<int32_t> & experts) {
    if (!range_overlap_enabled_ || prewarming_ || !source.range_store_ ||
            experts.empty()) {
        return false;
    }
    finish_deferred_range_read(source);

    std::unordered_map<
        MoeGpuArena *,
        std::unordered_set<mfq::MoeCacheKey, mfq::MoeCacheKeyHash>>
        arena_demands;
    for (int expert : experts) {
        if (expert < 0 || expert >= source.n_experts()) return false;
        const int cohort_index = source.expert_to_cohort_.at(
            static_cast<size_t>(expert));
        auto & cohort = source.cohorts_.at(
            static_cast<size_t>(cohort_index));
        arena_demands[cohort.arena].insert(
            {source.id_, cohort_index, expert});
    }
    for (const auto & item : arena_demands) {
        if (item.second.size() >
                static_cast<size_t>(item.first->book->capacity())) {
            return false;
        }
    }

    auto pending = std::make_unique<MoePendingRangeRead>();
    for (const auto & item : arena_demands) {
        auto * book = item.first->book.get();
        for (const auto & key : item.second) {
            const int slot = book->slot_for(key);
            if (slot >= 0 && !book->inflight(slot)) {
                book->mark_inflight(slot);
                pending->held_slots.emplace_back(book, slot);
            }
        }
    }

    try {
        append_source_transfers(
            source,
            experts,
            true,
            pending->transfers,
            pending->replaced_occupied,
            &pending->held_slots,
            &pending->new_leases);

        int64_t range_bytes = 0;
        size_t range_count = 0;
        for (const auto & transfer : pending->transfers) {
            if (transfer.range_store == nullptr) continue;
            range_bytes = (range_bytes + 15) & ~int64_t{15};
            if (transfer.nbytes >
                    std::numeric_limits<int64_t>::max() - range_bytes) {
                throw std::overflow_error(
                    "deferred MoE range byte count overflows int64");
            }
            range_bytes += transfer.nbytes;
            ++range_count;
        }
        if (range_count == 0) {
            submit_transfers(
                pending->transfers,
                pending->replaced_occupied,
                false);
            for (const auto & held : pending->held_slots) {
                held.first->clear_inflight(held.second);
            }
            return true;
        }

        pending->host = mfq_tensor_backend::empty(
            {range_bytes},
            mfq_tensor_backend::TensorOptions()
                .device(mfq_tensor_backend::kCPU)
                .dtype(mfq_tensor_backend::kUInt8)
                .pinned_memory(true));
        auto * staging = pending->host.data_ptr<uint8_t>();
        std::vector<mfq::cuda::MfeMxfp4ReadRequest> requests;
        requests.reserve(range_count);
        int64_t offset = 0;
        for (auto & transfer : pending->transfers) {
            if (transfer.range_store == nullptr) continue;
            offset = (offset + 15) & ~int64_t{15};
            requests.push_back({
                transfer.range_store,
                transfer.range_part,
                std::span<uint8_t>(
                    staging + offset,
                    static_cast<size_t>(transfer.nbytes)),
            });
            transfer.source = staging + offset;
            transfer.range_store = nullptr;
            transfer.range_part = nullptr;
            offset += transfer.nbytes;
        }
        pending->ticket = range_read_pool_->submit(requests);
        const auto inserted = pending_range_reads_.try_emplace(
            source.id_, std::move(pending));
        if (!inserted.second) {
            throw std::runtime_error(
                "MoE source already has a deferred range read");
        }
        ++stats_.range_overlap_batches;
        return true;
    } catch (...) {
        if (pending) {
            rollback_preparation(
                pending->new_leases, pending->held_slots);
        }
        throw;
    }
}

void MoeExpertCache::finish_deferred_range_read(
        MoeCachedSource & source) {
    const auto found = pending_range_reads_.find(source.id_);
    if (found == pending_range_reads_.end()) return;
    auto pending = std::move(found->second);
    pending_range_reads_.erase(found);
    try {
        const auto wait_begin = std::chrono::steady_clock::now();
        const auto range_stats = pending->ticket.wait();
        stats_.range_overlap_wait_nanoseconds +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - wait_begin).count();
        record_range_read(range_stats);
        submit_transfers(
            pending->transfers,
            pending->replaced_occupied,
            true);
    } catch (...) {
        rollback_preparation(
            pending->new_leases, pending->held_slots);
        throw;
    }
    for (const auto & held : pending->held_slots) {
        held.first->clear_inflight(held.second);
    }
}

bool MoeExpertCache::prepare(
        MoeCachedSource & source,
        const std::vector<int32_t> & experts,
        bool prefetch) {
    if (!finalized_) {
        throw std::runtime_error(
            "MoE cache must be finalized before inference");
    }
    finish_deferred_range_read(source);
    if (experts.empty()) return false;

    std::unordered_map<
        MoeGpuArena *,
        std::unordered_set<mfq::MoeCacheKey, mfq::MoeCacheKeyHash>>
        arena_demands;
    for (int expert : experts) {
        if (expert < 0 || expert >= source.n_experts()) return false;
        const int cohort_index = source.expert_to_cohort_.at(
            static_cast<size_t>(expert));
        auto & cohort = source.cohorts_.at(
            static_cast<size_t>(cohort_index));
        arena_demands[cohort.arena].insert(
            {source.id_, cohort_index, expert});
    }
    for (const auto & item : arena_demands) {
        if (item.second.size() >
                static_cast<size_t>(item.first->book->capacity())) {
            return false;
        }
    }

    std::vector<std::pair<mfq::MoeCacheSlotBook *, int>> held_slots;
    for (const auto & item : arena_demands) {
        auto * book = item.first->book.get();
        for (const auto & key : item.second) {
            const int slot = book->slot_for(key);
            if (slot >= 0 && !book->inflight(slot)) {
                book->mark_inflight(slot);
                held_slots.emplace_back(book, slot);
            }
        }
    }
    std::vector<MoeCacheTransfer> transfers;
    std::vector<MoeCacheNewLease> new_leases;
    bool replaced_occupied = false;
    try {
        append_source_transfers(
            source, experts, prefetch, transfers,
            replaced_occupied, &held_slots, &new_leases);
        submit_transfers(
            transfers,
            replaced_occupied,
            !prefetch);
        publish_quant_promotions(new_leases);
    } catch (...) {
        rollback_preparation(new_leases, held_slots);
        throw;
    }
    for (const auto & held : held_slots) {
        held.first->clear_inflight(held.second);
    }
    return true;
}

bool MoeExpertCache::prepare_bundle(
        const std::vector<MoeCachedSource *> & sources,
        const std::vector<int32_t> & experts) {
    if (!finalized_) {
        throw std::runtime_error(
            "MoE cache must be finalized before inference");
    }
    for (auto * source : sources) {
        if (source != nullptr && source->cache_ == this) {
            finish_deferred_range_read(*source);
        }
    }
    if (sources.empty() || experts.empty()) return false;

    std::unordered_map<
        MoeGpuArena *,
        std::unordered_set<mfq::MoeCacheKey, mfq::MoeCacheKeyHash>>
        arena_demands;
    for (auto * source : sources) {
        if (source == nullptr || source->cache_ != this) return false;
        for (int expert : experts) {
            if (expert < 0 || expert >= source->n_experts()) return false;
            const int cohort_index = source->expert_to_cohort_.at(
                static_cast<size_t>(expert));
            auto & cohort = source->cohorts_.at(
                static_cast<size_t>(cohort_index));
            arena_demands[cohort.arena].insert(
                {source->id_, cohort_index, expert});
        }
    }
    for (const auto & item : arena_demands) {
        if (item.second.size() >
                static_cast<size_t>(item.first->book->capacity())) {
            return false;
        }
    }

    std::vector<std::pair<mfq::MoeCacheSlotBook *, int>> held_slots;
    for (const auto & item : arena_demands) {
        auto * book = item.first->book.get();
        for (const auto & key : item.second) {
            const int slot = book->slot_for(key);
            if (slot >= 0 && !book->inflight(slot)) {
                book->mark_inflight(slot);
                held_slots.emplace_back(book, slot);
            }
        }
    }
    std::vector<MoeCacheTransfer> transfers;
    std::vector<MoeCacheNewLease> new_leases;
    bool replaced_occupied = false;
    try {
        for (auto * source : sources) {
            append_source_transfers(
                *source, experts, true, transfers,
                replaced_occupied, &held_slots, &new_leases);
        }
        submit_transfers(
            transfers, replaced_occupied, false);
        publish_quant_promotions(new_leases);
    } catch (...) {
        rollback_preparation(new_leases, held_slots);
        throw;
    }
    for (const auto & held : held_slots) {
        held.first->clear_inflight(held.second);
    }
    return true;
}

bool MoeExpertCache::prepare_bundle_deferred(
        const std::vector<MoeCachedSource *> & ready_sources,
        MoeCachedSource & deferred_source,
        const std::vector<int32_t> & experts) {
    if (!range_overlap_enabled_ || !deferred_source.range_store_) {
        auto sources = ready_sources;
        sources.push_back(&deferred_source);
        return prepare_bundle(sources, experts);
    }
    if (!prepare_bundle(ready_sources, experts)) return false;
    return begin_deferred_range_read(deferred_source, experts);
}

static MfeWeight wrap_cached_moe_source(
        const std::shared_ptr<MoeCachedSource> & source,
        const std::shared_ptr<MixedMoeRuntime> & cpu) {
    MfeWeight result =
        cpu_mixed_moe_metadata(cpu);
    result.mixed_weight_bytes = source->logical_weight_bytes();
    result.cached_source = source;
    result.cache_prefetch = [source](
            const MoeRoutePlan & route) {
        source->prefetch(route);
    };
    result.cache_prefetch_begin = [source](
            const MoeRoutePlan & route) {
        source->begin_prefetch(route);
    };
    result.mixed_forward = [source](
            CudaExecutionContext& execution,
            mfq_tensor_backend::Tensor x,
            const MoeRoutePlan & route) {
        return source->forward(execution, x, route);
    };
    result.mixed_prequantized_forward = [source](
            CudaExecutionContext& execution,
            mfq_tensor_backend::Tensor x,
            const MoeRoutePlan & route) {
        return source->forward_prequantized(execution, x, route);
    };
    result.mixed_glu_output_forward = [source](
            CudaExecutionContext& execution,
            mfq_tensor_backend::Tensor x,
            const MoeRoutePlan & route,
            bool gelu) {
        return source->forward_glu_output(
            execution, x, route, gelu);
    };
    result.mixed_glu_forward = [source](
            CudaExecutionContext& execution,
            mfq_tensor_backend::Tensor gate_up,
            const MoeRoutePlan & route,
            bool gelu) {
        return source->forward_glu(
            execution, gate_up, route, gelu);
    };
    if (source->supports_clamped_swiglu()) {
        result.mixed_clamped_swiglu_forward = [source](
                CudaExecutionContext& execution,
                mfq_tensor_backend::Tensor gate_up,
                const MoeRoutePlan & route,
                double limit) {
            return source->forward_clamped_swiglu(
                execution, gate_up, route, limit);
        };
    }
    return result;
}

bool prefetch_cached_moe_projection_bundle(
        const MfeWeight & gate,
        const MfeWeight & up,
        const MfeWeight & down,
        const MoeRoutePlan & route) {
    if (!gate.cached_source ||
            !up.cached_source ||
            !down.cached_source) {
        return false;
    }
    return MoeCachedSource::prefetch_bundle(
        {gate.cached_source, up.cached_source, down.cached_source},
        route);
}

bool prefetch_cached_moe_projection_bundle(
        const MfeWeight & gate_up,
        const MfeWeight & down,
        const MoeRoutePlan & route) {
    if (!gate_up.cached_source || !down.cached_source) {
        return false;
    }
    return MoeCachedSource::prefetch_bundle(
        {gate_up.cached_source, down.cached_source}, route);
}

MfeWeight cache_moe_weight(
        const std::shared_ptr<MoeExpertCache>& cache,
        const std::string& name,
        const std::shared_ptr<MixedMoeRuntime>& runtime,
        int minimum_slots,
        int layer_id,
        const std::string& projection_role,
        std::shared_ptr<mfq::cuda::MfeMxfp4ExpertStore> range_store) {
    auto source = range_store
        ? cache->register_range_source(name, runtime, std::move(range_store),
                                       minimum_slots, layer_id, projection_role)
        : cache->register_source(name, runtime, minimum_slots, layer_id, projection_role);
    // The cache owns registered sources. Export an alias that retains the cache,
    // so weights and copied forward/prefetch operations cannot outlive its arenas.
    return wrap_cached_moe_source(
        std::shared_ptr<MoeCachedSource>(cache, source.get()), runtime);
}

std::shared_ptr<MoeExpertCache> make_moe_expert_cache(
        std::int64_t bytes,
        const CudaExecutionConfig& config) {
    return std::make_shared<MoeExpertCache>(bytes, config);
}

MfeWeight cache_quant_moe_weight(const std::shared_ptr<MoeExpertCache>& cache,
        const std::string& name,std::shared_ptr<MoeQuantRangeSource> range,
        int minimum_slots,int layer_id,const std::string& role) {
    auto runtime=range->metadata();
    auto source=cache->register_quant_range_source(name,std::move(range),minimum_slots,layer_id,role);
    return wrap_cached_moe_source(std::shared_ptr<MoeCachedSource>(cache,source.get()),runtime);
}

bool moe_expert_cache_has_sources(
        const std::shared_ptr<MoeExpertCache>& cache) {
    return cache && cache->has_sources();
}

std::vector<std::pair<std::string, double>> moe_expert_memory_metrics(
        const std::shared_ptr<MoeExpertCache>& cache) {
    return cache ? cache->memory_metrics() : std::vector<std::pair<std::string, double>>{
        {"ram_expert_enabled", 0.0}, {"ram_expert_payload_bytes", 0.0},
        {"ram_expert_pcie_read_bytes", 0.0}};
}

bool moe_expert_cache_finalized(
        const std::shared_ptr<MoeExpertCache>& cache) {
    return cache && cache->finalized();
}

void finalize_moe_expert_cache(
        const std::shared_ptr<MoeExpertCache>& cache) {
    if (cache && !cache->finalized()) cache->finalize();
}

void print_moe_expert_cache_stats(
        const std::shared_ptr<MoeExpertCache>& cache,
        std::ostream& output) {
    if (cache) {
        cache->finish_pipeline_exchanges();
        cache->print_stats(output);
    }
}
double moe_expert_cache_ram_pcie_fraction(const std::shared_ptr<MoeExpertCache>& cache) {
    return cache ? cache->ram_pcie_fraction() : 0;
}
void release_moe_prefill_buffers(MoeExpertCache* cache) {
    if(!cache || (cache->pipeline_prefill_release_.empty() &&
            !cache->pipeline_prefill_layer_buffer_.defined() && !cache->pipeline_prefill_pool_))return;
    cudaStreamCaptureStatus capture;
    const auto stream=mfq_current_cuda_stream();
    MFQ_CUDA_CHECK(cudaStreamIsCapturing(stream,&capture));
    if(capture!=cudaStreamCaptureStatusNone)
        throw std::runtime_error("prefill buffers must be released before decode graph capture");
    MFQ_CUDA_CHECK(cudaStreamSynchronize(stream));
    MFQ_CUDA_CHECK(cudaStreamSynchronize(cache->weight_stream_));
    for(auto& release:cache->pipeline_prefill_release_)release();
    cache->pipeline_prefill_release_.clear();
    for(auto i=cache->pipeline_stages_.begin();i!=cache->pipeline_stages_.end();)
        if(i->first.starts_with("prefill16:"))i=cache->pipeline_stages_.erase(i);else ++i;
    cache->pipeline_prefill_activations_.reset();
    cache->pipeline_prefill_ring_={};
    cache->pipeline_prefill_layer_buffer_={};
    cache->pipeline_prefill_layer_stride_=0;
    cache->pipeline_prefill_layer_loaded_={};
    cache->pipeline_prefill_layer_last_served_=-1;
    cache->pipeline_prefill_layer_read_recorded_={};
    if(cache->pipeline_prefill_pool_) {
        MFQ_CUDA_CHECK(cudaMemPoolSetAttribute(cache->pipeline_prefill_pool_,cudaMemPoolAttrReleaseThreshold,
            &cache->pipeline_prefill_pool_previous_threshold_));
        MFQ_CUDA_CHECK(cudaStreamSynchronize(stream));
        MFQ_CUDA_CHECK(cudaMemPoolTrimTo(cache->pipeline_prefill_pool_,0));
        cache->pipeline_prefill_pool_=nullptr;
    }
    cache->pipeline_prefill_dma_metadata_.reset();cache->pipeline_dma_leases_.clear();
    cache->transfer_ready_recorded_=false;
}

void finish_moe_expert_exchanges(const std::shared_ptr<MoeExpertCache>& cache) {
    if(cache)cache->finish_pipeline_exchanges();
}


void prepare_moe_pipeline_comparison(const std::shared_ptr<MoeExpertCache>& cache,bool replay) {
    if(!cache || !cache->complete_residency_ || cache->config_.moe_residency_adapt ||
        !cache->pipeline_transfer_cache_ || cache->pipeline_mapped_copy_)
        throw std::logic_error("MFQ pipeline comparison requires complete residency, fixed primary tiers and DMA transfer caching");
    if(replay && !cache->pipeline_dispatch_capture_required_)
        throw std::logic_error("MFQ pipeline comparison has no captured baseline");
    if(!replay && !cache->pipeline_dispatch_replay_.empty())
        throw std::logic_error("MFQ pipeline comparison cannot combine an external dispatch replay");
    if(replay) {
        finish_moe_pipeline_comparison(cache);
        if(cache->pipeline_comparison_records_.empty())
            cache->pipeline_comparison_records_=cache->pipeline_dispatch_records_;
    }
    cache->finish_pipeline_exchanges();
    if(cache->compute_done_recorded_)MFQ_CUDA_CHECK(cudaEventSynchronize(cache->compute_done_));
    MFQ_CUDA_CHECK(cudaStreamSynchronize(cache->weight_stream_));
    cache->transfer_ready_recorded_=false;
    cache->pipeline_dma_leases_.clear();
    for(const auto& held:cache->pipeline_transfer_slots_)
        if(held.book->slot_for(held.key)==held.slot)held.book->clear_inflight(held.slot);
    cache->pipeline_transfer_slots_.clear();
    cache->pipeline_prefetched_keys_.clear();
    for(auto& [key,arena]:cache->pipeline_stages_)
        if(arena->book)arena->book=std::make_unique<mfq::MoeCacheSlotBook>(arena->slots);
    if(cache->pipeline_wire_gpu_.defined()) {
        MFQ_CUDA_CHECK(cudaMemsetAsync(cache->pipeline_wire_gpu_.data_ptr(),0,32,cache->weight_stream_));
        MFQ_CUDA_CHECK(cudaStreamSynchronize(cache->weight_stream_));
    }
    if(!replay)cache->pipeline_comparison_records_.clear();
    cache->pipeline_dispatch_replay_=replay?cache->pipeline_comparison_records_:std::vector<MoeExpertCache::DispatchRecord>{};
    cache->pipeline_dispatch_replay_cursor_=0;
    cache->pipeline_dispatch_records_.clear();
    cache->pipeline_dispatch_capture_required_=true;
    cache->pipeline_comparison_cpu_policy_=false;
    cache->pipeline_cpu_policy_changed_routes_=0;
}
void prepare_moe_cpu_budget_comparison(const std::shared_ptr<MoeExpertCache>& cache,
        bool replay,bool transfer_budget) {
    prepare_moe_pipeline_comparison(cache,replay);
    if(!cache->pipeline_shared_cpu_cost_)
        throw std::logic_error("CPU budget comparison needs shared measured CPU costs");
    cache->pipeline_cpu_transfer_budget_=transfer_budget;
    cache->pipeline_comparison_cpu_policy_=replay && transfer_budget;
    // Existing CPU/GPU arithmetic can change downstream router IDs. Run the
    // measured policy naturally and retain the first baseline for final replay.
    if(cache->pipeline_comparison_cpu_policy_)cache->pipeline_dispatch_replay_.clear();
}
void prepare_moe_cpu_calibration_comparison(const std::shared_ptr<MoeExpertCache>& cache,bool replay) {
    prepare_moe_cpu_budget_comparison(cache,replay,true);
    cache->pipeline_cpu_calibration_.reset(cache->pipeline_cpu_cost_);
    cache->pipeline_cpu_cost_={};
}
std::size_t finish_moe_pipeline_comparison(const std::shared_ptr<MoeExpertCache>& cache) {
    if(!cache || !cache->pipeline_dispatch_capture_required_ || cache->pipeline_dispatch_records_.empty())
        throw std::logic_error("MFQ pipeline comparison did not capture dispatch records");
    const auto& expected=cache->pipeline_comparison_cpu_policy_
        ?cache->pipeline_comparison_records_:cache->pipeline_dispatch_replay_;
    if(!expected.empty()) {
        const auto& actual=cache->pipeline_dispatch_records_;
        if(actual.size()!=expected.size() || (!cache->pipeline_comparison_cpu_policy_ &&
            cache->pipeline_dispatch_replay_cursor_!=expected.size()))
            throw std::runtime_error("MFQ pipeline comparison dispatch count mismatch");
        std::uint64_t changed=0;
        for(std::size_t i=0;i<actual.size();++i) {
            if(actual[i].layer!=expected[i].layer || actual[i].tokens!=expected[i].tokens ||
                actual[i].ids.size()!=expected[i].ids.size() ||
                (!cache->pipeline_comparison_cpu_policy_ &&
                    (actual[i].ids!=expected[i].ids || actual[i].cpu!=expected[i].cpu)))
                throw std::runtime_error("MFQ pipeline comparison dispatch mismatch");
            changed+=actual[i].ids!=expected[i].ids;
        }
        cache->pipeline_cpu_policy_changed_routes_=changed;
    }
    return cache->pipeline_dispatch_records_.size();
}

void set_moe_expert_cache_profile(
        MoeExpertCache& cache,
        mfq::MoeCacheProfile profile) {
    cache.set_profile(std::move(profile));
}
