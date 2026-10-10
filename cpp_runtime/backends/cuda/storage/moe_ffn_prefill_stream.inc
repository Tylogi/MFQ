    // Two compact expert groups keep the copy engine ahead of arithmetic.
    // The group-local route space avoids a full-layer transfer/output workspace.
    static constexpr int prefill_group_capacity=16;
    struct PrefillBuffer {
        std::array<MixedMoeRuntime,3> projections;
        tb::Tensor maps;
        std::array<std::vector<MoeGpuArena*>,3> stages;
        mfq::cuda::Event gu_ready{cudaEventDisableTiming};
        mfq::cuda::Event down_ready{cudaEventDisableTiming};
        mfq::cuda::Event used{cudaEventDisableTiming};
    };
    struct PrefillStream {
        std::array<PrefillBuffer,2> buffers;
        std::unique_ptr<Mapped> publication;
        int capacity=0,group_size=prefill_group_capacity;
    };
    std::unique_ptr<PrefillStream> prefill_stream_;

    static int64_t prefill_align(int64_t bytes) {return (bytes+255)&~int64_t(255);}
    static int64_t prefill_cohort_bytes(const MoeCachedCohort& cohort) {
        const int slots=2*std::min(prefill_group_capacity,cohort.cpu->local_experts);
        int64_t bytes=0;
        for(const auto size:cohort.bytes_per_expert)bytes+=prefill_align(slots*size);
        return bytes;
    }
    MoeGpuArena& prefill_stage(int projection,MoeCachedCohort& cohort,int64_t& offset) {
        const auto key="prefill16:"+std::to_string(sources_[projection]->id_)+":"+std::to_string(offset);
        auto& arena=cache_->pipeline_stages_[key];
        if(arena) {offset+=prefill_cohort_bytes(cohort);return *arena;}
        auto prepared=std::make_unique<MoeGpuArena>();
        prepared->signature=key;prepared->slot_bytes=cohort.arena->slot_bytes;
        prepared->layouts=cohort.arena->layouts;
        prepared->slots=2*std::min(prefill_group_capacity,cohort.cpu->local_experts);
        for(std::size_t f=0;f<prepared->layouts.size();++f) {
            const auto& layout=prepared->layouts[f];
            auto shape=layout.slot_shape;shape[0]*=prepared->slots;
            const auto bytes=prefill_align(prepared->slots*cohort.bytes_per_expert[f]);
            if(offset+bytes>cache_->pipeline_prefill_ring_.numel())
                throw std::logic_error("MFQ prefill ring budget is smaller than its layer layout");
            auto* pointer=cache_->pipeline_prefill_ring_.data_ptr<uint8_t>()+offset;
            prepared->fields.push_back(tb::from_blob(pointer,shape,
                tb::TensorOptions().device(tb::kCUDA).dtype(layout.scalar_type)));
            offset+=bytes;
        }
        arena=std::move(prepared);return *arena;
    }
    void initialize_prefill_stream(CudaExecutionContext& execution,const tb::Tensor& ids) {
        if(prefill_stream_)return;
        auto prepared=std::make_unique<PrefillStream>();
        if(const auto* value=std::getenv("MFQ_MOE_PREFILL_GROUP_SIZE"))
            prepared->group_size=std::clamp(std::atoi(value),1,prefill_group_capacity);
        prepared->group_size=std::min(prepared->group_size,experts_);
        const int prefill_group_size=prepared->group_size;
        if(!cache_->pipeline_prefill_ring_.defined()) {
            std::unordered_map<int,int64_t> layers;
            for(const auto& source:cache_->sources_)for(const auto& cohort:source->cohorts_)
                layers[source->layer_id_]+=prefill_cohort_bytes(cohort);
            int64_t bytes=0;for(const auto& layer:layers)bytes=std::max(bytes,layer.second);
            cache_->pipeline_prefill_ring_=tb::empty({bytes},ids.options().dtype(tb::kUInt8));
        }
        std::array<std::vector<MoeGpuArena*>,3> stages;int64_t stage_offset=0;
        for(int p=0;p<3;++p)for(auto& cohort:sources_[p]->cohorts_)
            stages[p].push_back(&prefill_stage(p,cohort,stage_offset));
        if(execution.config.moe_ffn_fused_activation && !cache_->pipeline_sigmoid_table_.defined())
            cache_->pipeline_sigmoid_table_=moe_swiglu_sigmoid_table_cuda();
        int map_count=0;
        for(const auto& source:sources_)map_count+=2*int(source->cohorts_.size())*prefill_group_size;
        for(auto& buffer:prepared->buffers)buffer.maps=tb::full({map_count},-1,ids.options());
        for(int b=0;b<2;++b)for(int p=0;p<3;++p) {
            auto& buffer=prepared->buffers[b];auto& runtime=buffer.projections[p];
            auto& source=*sources_[p];
            runtime.n_experts=prefill_group_size;runtime.out_per_expert=source.cpu_->out_per_expert;
            runtime.neuron_len=source.cpu_->neuron_len;
            runtime.prefill_only=true;
            // Each routed row has exactly one owner across the hot/cold pools.
            runtime.partial_experts=false;
            int map_base=0;
            for(int previous=0;previous<p;++previous)map_base+=2*int(sources_[previous]->cohorts_.size())*prefill_group_size;
            int cohort_index=0;
            for(auto& cohort:source.cohorts_) {
                auto& arena=*stages[p][cohort_index];buffer.stages[p].push_back(&arena);
                std::vector<tb::Tensor> fields;
                for(std::size_t f=0;f<arena.fields.size();++f) {
                    const auto rows=arena.layouts[f].slot_shape[0];
                    const int slots=arena.slots/2;
                    fields.push_back(arena.fields[f].narrow(0,b*slots*rows,slots*rows));
                }
                auto hot=cohort.active;
                hot.expert_local=buffer.maps.narrow(0,map_base+2*cohort_index*prefill_group_size,prefill_group_size);
                auto cold=fields_pool(cohort.active,fields,arena.slots/2,runtime.out_per_expert);
                cold.expert_local=buffer.maps.narrow(0,map_base+(2*cohort_index+1)*prefill_group_size,prefill_group_size);
                runtime.pools.push_back(std::move(hot));runtime.pools.push_back(std::move(cold));++cohort_index;
            }
            const auto* mma=std::getenv("MFQ_MOE_PREFILL_COMPACT_MMA");
            initialize_mixed_nvq_dispatch(runtime,execution.config,!mma || std::atoi(mma)!=0);
        }
        prefill_stream_=std::move(prepared);
        cache_->pipeline_prefill_release_.push_back([owner=weak_from_this()] {
            if(auto pipeline=owner.lock())pipeline->prefill_stream_.reset();
        });
    }
    tb::Tensor streamed_prefill(CudaExecutionContext& execution,const tb::Tensor& input,
            const tb::Tensor& ids,const tb::Tensor& weights) {
        using Clock=std::chrono::steady_clock;
        const auto elapsed=[](Clock::time_point started) {
            return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-started).count();
        };
        initialize_prefill_stream(execution,ids);
        auto& stream=*prefill_stream_;auto& stats=cache_->stats_;
        const int prefill_group_size=stream.group_size;
        const int tokens=int(ids.size(0)),routes=int(ids.size(1)),entries=tokens*routes;
        if(stream.capacity<entries) {
            MFQ_CUDA_CHECK(cudaStreamSynchronize(stream_));
            stream.publication=std::make_unique<Mapped>(64+std::size_t(entries)*sizeof(int32_t));
            stream.capacity=entries;
        }
        auto* published=stream.publication->h<uint32_t>();*published=0;
        mfq::cuda::moe_publish_routes(nullptr,ids.data_ptr<int32_t>(),nullptr,
            stream.publication->d<int32_t>()+16,stream.publication->d<uint32_t>(),0,entries,stream_);
        auto shared=shared_?shared_(execution,input):tb::Tensor{};
        const auto route_started=Clock::now();
        mfq::cuda::wait_route_publication(published,stream_);
        stats.pipeline_route_wait_ns+=elapsed(route_started);++stats.pipeline_serves;++stats.pipeline_prefill_group_serves;
        // Publication follows every previous consumer of the shared ring and
        // pinned metadata. A completed previous DMA is implied by that consumer.
        if(cache_->transfer_ready_recorded_)MFQ_CUDA_CHECK(cudaEventSynchronize(cache_->transfer_ready_));
        cache_->pipeline_dma_leases_.clear();cache_->pipeline_prefill_dma_metadata_.reset();
        const auto plan_started=Clock::now();
        const auto* host_ids=stream.publication->h<int32_t>()+16;
        std::vector<std::vector<int32_t>> positions(experts_);
        for(int i=0;i<entries;++i) {
            const int e=host_ids[i];
            if(e<0 || e>=experts_)throw std::runtime_error("MFQ prefill expert outside route geometry");
            positions[e].push_back(i);
        }
        std::vector<int> order;
        for(int e=0;e<experts_;++e)if(!positions[e].empty())order.push_back(e);
        const int groups=(int(order.size())+prefill_group_size-1)/prefill_group_size;
        struct Upload {int cohort,slot;std::vector<tb::Tensor> fields;};
        struct Group {
            int begin=0,rows=0;
            int route_offset=0,mma_tile_m=8,wide_tile_m=8;
            std::array<std::vector<Upload>,3> uploads;
        };
        std::vector<Group> work(groups);stats.pipeline_prefill_groups+=groups;
        std::array<int,3> map_offset{};int map_stride=0;
        for(int p=0;p<3;++p) {
            map_offset[p]=map_stride;
            map_stride+=int(stream.buffers[0].projections[p].pools.size())*prefill_group_size;
        }
        // One immutable pinned allocation supplies all asynchronous metadata
        // copies. It survives until the following layer publishes its routes.
        const std::size_t index_bytes=std::size_t(entries)*2*sizeof(int64_t);
        const std::size_t id_bytes=std::size_t(entries)*sizeof(int32_t);
        const std::size_t map_bytes=std::size_t(groups)*map_stride*sizeof(int32_t);
        const int bounds_size=prefill_group_size+1;
        const int group_fixed=4*bounds_size+prefill_group_size;
        const std::size_t route_words=std::size_t(entries)*4+std::size_t(groups)*group_fixed;
        auto metadata=std::make_shared<Mapped>(index_bytes+id_bytes+map_bytes+route_words*sizeof(int32_t));
        cache_->pipeline_prefill_dma_metadata_=metadata;
        auto* token_indices=metadata->h<int64_t>();
        auto* pair_indices=token_indices+entries;
        auto* group_ids=reinterpret_cast<int32_t*>(pair_indices+entries);
        auto* maps=group_ids+entries;std::fill(maps,maps+std::size_t(groups)*map_stride,-1);
        auto* route_metadata=maps+std::size_t(groups)*map_stride;
        int row=0;
        for(int g=0;g<groups;++g) {
            auto& group=work[g];group.begin=row;
            std::array<std::vector<int>,3> next;
            for(int p=0;p<3;++p)next[p].resize(sources_[p]->cohorts_.size());
            const int count=std::min(prefill_group_size,int(order.size())-g*prefill_group_size);
            for(int q=0;q<count;++q) {
                const int e=order[g*prefill_group_size+q];
                ++stats.hybrid_gpu_experts;bool cold_expert=false;
                for(int position:positions[e]) {
                    token_indices[row]=position/routes;pair_indices[row]=position;group_ids[row]=q;++row;
                }
                for(int p=0;p<3;++p) {
                    auto& source=*sources_[p];const int c=source.expert_to_cohort_[e];
                    const auto& cohort=source.cohorts_[c];
                    const int hot=cohort.arena->book->slot_for({source.id_,c,e});
                    auto* map=maps+g*map_stride+map_offset[p]+2*c*prefill_group_size;
                    if(hot>=0) {map[q]=hot;continue;}
                    cold_expert=true;auto lease=source.quant_source_->acquire_expert(e);
                    const int slot=next[p][c]++;map[prefill_group_size+q]=slot;
                    group.uploads[p].push_back({c,slot,moe_cache_fields(lease->weights)});
                    cache_->pipeline_dma_leases_.push_back(std::move(lease));
                }
                stats.ram_pcie_experts+=cold_expert;
            }
            group.rows=row-group.begin;
        }
        if(row!=entries)throw std::logic_error("MFQ prefill grouping lost routed rows");
        // The host has already sorted rows by expert for the DMA plan. Reuse
        // that ordering instead of running histogram/prefix/scatter kernels
        // and allocating three separate maps for every group.
        int route_offset=0;
        for(auto& group:work) {
            group.route_offset=route_offset;
            const int density=std::max(1,(group.rows+prefill_group_size-1)/prefill_group_size);
            group.mma_tile_m=density<4?8:(density<=16?16:(density<=32?32:64));
            group.wide_tile_m=density>64?128:group.mma_tile_m;
            auto* destination=route_metadata+route_offset;
            auto* bounds=destination+group.rows;
            auto* counts=bounds+bounds_size;
            std::fill(counts,counts+prefill_group_size,0);
            for(int i=0;i<group.rows;++i) {destination[i]=i;++counts[group_ids[group.begin+i]];}
            bounds[0]=0;
            for(int q=0;q<prefill_group_size;++q)bounds[q+1]=bounds[q]+counts[q];
            auto* tile_bounds=counts+prefill_group_size;
            auto* tile_experts=tile_bounds+bounds_size;
            for(int tile_m:{8,group.mma_tile_m,group.wide_tile_m}) {
                tile_bounds[0]=0;std::fill(tile_experts,tile_experts+group.rows,0);
                for(int q=0;q<prefill_group_size;++q) {
                    tile_bounds[q+1]=tile_bounds[q]+(counts[q]+tile_m-1)/tile_m;
                    std::fill(tile_experts+tile_bounds[q],tile_experts+tile_bounds[q+1],q);
                }
                tile_bounds=tile_experts+group.rows;tile_experts=tile_bounds+bounds_size;
            }
            route_offset+=4*group.rows+group_fixed;
        }
        if(std::size_t(route_offset)!=route_words)throw std::logic_error("MFQ prefill route metadata geometry differs");
        stats.pipeline_plan_ns+=elapsed(plan_started);
        auto token_device=tb::empty({entries},ids.options().dtype(tb::kInt64));
        auto pair_device=tb::empty({entries},ids.options().dtype(tb::kInt64));
        auto route_device=tb::empty({entries},ids.options());
        auto map_device=tb::empty({int64_t(route_words)},ids.options());
        MFQ_CUDA_CHECK(cudaMemcpyAsync(token_device.data_ptr(),token_indices,std::size_t(entries)*sizeof(int64_t),cudaMemcpyHostToDevice,stream_));
        MFQ_CUDA_CHECK(cudaMemcpyAsync(pair_device.data_ptr(),pair_indices,std::size_t(entries)*sizeof(int64_t),cudaMemcpyHostToDevice,stream_));
        MFQ_CUDA_CHECK(cudaMemcpyAsync(route_device.data_ptr(),group_ids,id_bytes,cudaMemcpyHostToDevice,stream_));
        MFQ_CUDA_CHECK(cudaMemcpyAsync(map_device.data_ptr(),route_metadata,route_words*sizeof(int32_t),cudaMemcpyHostToDevice,stream_));
        auto parts=tb::empty({entries,width_},input.options());
        const auto ram_begin=reinterpret_cast<std::uintptr_t>(cache_->pipeline_ram_complement_->data_ptr());
        const auto ram_end=ram_begin+cache_->pipeline_ram_registered_bytes_;
        const auto stage_group=[&](int g) {
            const auto started=Clock::now();auto& buffer=stream.buffers[g%2];
            if(g>=2)MFQ_CUDA_CHECK(cudaStreamWaitEvent(cache_->weight_stream_,buffer.used.get(),0));
            for(int p=0;p<3;++p) {
                const auto& uploads=work[g].uploads[p];auto& source=*sources_[p];
                for(std::size_t c=0;c<source.cohorts_.size();++c) {
                    std::vector<const Upload*> cohort_uploads;
                    for(const auto& upload:uploads)if(upload.cohort==int(c))cohort_uploads.push_back(&upload);
                    const auto& cohort=source.cohorts_[c];
                    const auto destination_fields=moe_cache_fields(buffer.projections[p].pools[2*c+1]);
                    for(std::size_t f=0;f<cohort.bytes_per_expert.size();++f) {
                        const auto bytes=std::size_t(cohort.bytes_per_expert[f]);if(!bytes)continue;
                        auto* destination_base=static_cast<uint8_t*>(destination_fields[f].data_ptr());
                        for(std::size_t first=0;first<cohort_uploads.size();) {
                            const auto start=reinterpret_cast<std::uintptr_t>(cohort_uploads[first]->fields[f].data_ptr());
                            if(start<ram_begin || start>ram_end || bytes>ram_end-start)
                                throw std::logic_error("MFQ streamed prefill needs registered RAM fields");
                            std::size_t last=first+1,pitch=0;
                            if(last<cohort_uploads.size()) {
                                const auto next=reinterpret_cast<std::uintptr_t>(cohort_uploads[last]->fields[f].data_ptr());
                                if(next>start && next-start>=bytes && next-start<=std::size_t(INT_MAX))pitch=next-start;
                            }
                            if(pitch)while(last<cohort_uploads.size()) {
                                const auto next=reinterpret_cast<std::uintptr_t>(cohort_uploads[last]->fields[f].data_ptr());
                                if(next!=start+(last-first)*pitch || next>ram_end || bytes>ram_end-next)break;
                                ++last;
                            }
                            const auto rows=last-first,total=bytes*rows;
                            auto* destination=destination_base+std::size_t(cohort_uploads[first]->slot)*bytes;
                            if(rows==1 || pitch==bytes)MFQ_CUDA_CHECK(cudaMemcpyAsync(destination,reinterpret_cast<const void*>(start),total,cudaMemcpyHostToDevice,cache_->weight_stream_));
                            else MFQ_CUDA_CHECK(cudaMemcpy2DAsync(destination,bytes,reinterpret_cast<const void*>(start),pitch,bytes,rows,cudaMemcpyHostToDevice,cache_->weight_stream_));
                            stats.ram_pcie_bytes+=total;stats.pipeline_direct_ram_bytes+=total;
                            if(p<2)stats.pipeline_gate_up_dma_bytes+=total;else stats.pipeline_down_dma_bytes+=total;
                            ++stats.pipeline_direct_ram_copies;++stats.pipeline_dma_copies;first=last;
                        }
                    }
                }
                if(p==1)buffer.gu_ready.record(cache_->weight_stream_);
            }
            buffer.down_ready.record(cache_->weight_stream_);
            stats.pipeline_fetch_ns+=elapsed(started);
        };
        if(groups)stage_group(0);
        if(groups>1)stage_group(1);
        for(int g=0;g<groups;++g) {
            auto& buffer=stream.buffers[g%2];const auto& group=work[g];
            MFQ_CUDA_CHECK(cudaStreamWaitEvent(stream_,buffer.gu_ready.get(),0));
            MFQ_CUDA_CHECK(cudaMemcpyAsync(buffer.maps.data_ptr(),maps+g*map_stride,
                std::size_t(map_stride)*sizeof(int32_t),cudaMemcpyHostToDevice,stream_));
            auto x=input.index_select(0,token_device.narrow(0,group.begin,group.rows));
            MoeRoutePlan route;
            route.ids=route_device.narrow(0,group.begin,group.rows).reshape({group.rows,1});
            route.n_experts=prefill_group_size;route.map_ready=true;
            int offset=group.route_offset;
            const auto take=[&](int count) {auto value=map_device.narrow(0,offset,count);offset+=count;return value;};
            route.ids_dst=take(group.rows);route.expert_bounds=take(bounds_size);route.counts=take(prefill_group_size);
            route.tile_bounds=take(bounds_size);route.tile_experts=take(group.rows);
            route.mma_tile_bounds=take(bounds_size);route.mma_tile_experts=take(group.rows);
            route.wide_tile_bounds=take(bounds_size);route.wide_tile_experts=take(group.rows);
            route.mma_tile_m=group.mma_tile_m;route.wide_tile_m=group.wide_tile_m;
            const auto projection=[&](int p,const tb::Tensor& x) {
                auto& runtime=buffer.projections[p];
                if(!cache_->pipeline_prefill_activations_)
                    cache_->pipeline_prefill_activations_=std::make_shared<MixedMoeRuntime>();
                for(const auto& geometry:runtime.activation_geometry()) {
                    const MixedMoeTransformKey transform{geometry.transform_block,geometry.transform_seed};
                    const MixedMoeActivationKey key{group.rows,geometry.groups,geometry.gs,x.get_device(),transform};
                    runtime.activation_workspaces.insert_or_assign(key,
                        cache_->pipeline_prefill_activations_->prefill_activation_workspace(
                            x,group.rows,geometry.groups,geometry.gs,transform));
                }
                auto output=runtime.forward(execution.config,execution.kl_mmq,execution.force_moe_prefill_mma_off,
                    execution.force_moe_pool_path,x,route);
                // Variable group densities must not retain one q8 scratch
                // allocation per row count across all prompt chunks.
                runtime.nint_input_plans.clear();runtime.nvq_active_plans.clear();
                runtime.activation_workspaces.clear();return output;
            };
            auto gate=projection(0,x),up=projection(1,x);
            auto hidden=execution.config.moe_ffn_fused_activation
                ? moe_swiglu_rounded_cuda(gate,up,cache_->pipeline_sigmoid_table_)
                : (gate*tb::sigmoid(gate))*up;
            MFQ_CUDA_CHECK(cudaStreamWaitEvent(stream_,buffer.down_ready.get(),0));
            auto down=projection(2,hidden).reshape({group.rows,width_});
            parts.index_copy_(0,pair_device.narrow(0,group.begin,group.rows),down);
            buffer.used.record(stream_);
            if(g+2<groups)stage_group(g+2);
        }
        MFQ_CUDA_CHECK(cudaEventRecord(cache_->transfer_ready_,cache_->weight_stream_));
        cache_->transfer_ready_recorded_=true;
        auto output=parts.reshape({tokens,routes,width_});
        if(shared.defined())output=moe_weighted_reduce_cuda(output,weights)+shared;
        cache_->record_compute_use();
        cache_->moe_residency_->after_layer(sources_[0]->layer_id_,std::vector<int32_t>(host_ids,host_ids+entries),tokens);
        MFQ_CUDA_CHECK(cudaGetLastError());return output;
    }
