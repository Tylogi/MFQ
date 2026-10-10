    struct LayerPrefill {
        std::array<MixedMoeRuntime,3> projections;
        int layer_id=0,slot=0;
        int64_t ram_begin=0,bytes=0;
    };
    std::unique_ptr<LayerPrefill> prefill_layer_;
    bool prefill_layer_flush() const {
        const auto* option=std::getenv("MFQ_MOE_PREFILL_LAYER_FLUSH");
        return !option || std::atoi(option)!=0;
    }
    bool prefill_layer_async() const {
        const auto* option=std::getenv("MFQ_MOE_PREFILL_LAYER_ASYNC");
        return !option || std::atoi(option)!=0;
    }
    bool prefill_layer_phased() const {
        const auto* option=std::getenv("MFQ_MOE_PREFILL_LAYER_PHASED");
        return !option || std::atoi(option)!=0;
    }

    void retain_prefill_pool(const tb::Tensor& ids) {
#ifdef MFQ_NATIVE_CUDA_RUNTIME
        const auto* retain=std::getenv("MFQ_MOE_PREFILL_POOL_RETAIN");
        if((retain && std::atoi(retain)==0) || cache_->pipeline_prefill_pool_)return;
        int supported=0;
        MFQ_CUDA_CHECK(cudaDeviceGetAttribute(&supported,cudaDevAttrMemoryPoolsSupported,ids.get_device()));
        if(!supported)return;
        cudaMemPool_t pool=nullptr;
        uint64_t previous=0;
        MFQ_CUDA_CHECK(cudaDeviceGetDefaultMemPool(&pool,ids.get_device()));
        MFQ_CUDA_CHECK(cudaMemPoolGetAttribute(pool,cudaMemPoolAttrReleaseThreshold,&previous));
        const auto limit=mfq::cuda::default_context(ids.get_device())->memory_stats().limit;
        uint64_t threshold=limit?limit:std::numeric_limits<uint64_t>::max();
        MFQ_CUDA_CHECK(cudaMemPoolSetAttribute(pool,cudaMemPoolAttrReleaseThreshold,&threshold));
        cache_->pipeline_prefill_pool_previous_threshold_=previous;
        cache_->pipeline_prefill_pool_=pool;
#endif
    }

    void initialize_prefill_layer(CudaExecutionContext& execution,const tb::Tensor& ids) {
        if(prefill_layer_)return;
        if(!cache_->pipeline_prefill_layer_layout_)
            throw std::logic_error("whole-layer prefill requires field-major preload");
        const auto layout=cache_->pipeline_prefill_layers_.at(sources_[0]->layer_id_);
        if(!cache_->pipeline_prefill_layer_buffer_.defined()) {
            int64_t maximum=0;
            for(const auto& layer:cache_->pipeline_prefill_layers_)maximum=std::max(maximum,layer.second.second);
            const auto* overlap=std::getenv("MFQ_MOE_PREFILL_LAYER_OVERLAP");
            cache_->pipeline_prefill_layer_overlap_=!overlap || std::atoi(overlap)!=0;
            cache_->pipeline_prefill_layer_stride_=maximum;
            cache_->pipeline_prefill_layer_buffer_=tb::empty(
                {maximum*(cache_->pipeline_prefill_layer_overlap_?2:1)},ids.options().dtype(tb::kUInt8));
            if(cache_->pipeline_prefill_layer_overlap_) {
                for(auto& event:cache_->pipeline_prefill_layer_ready_)
                    if(!event)MFQ_CUDA_CHECK(cudaEventCreateWithFlags(&event,cudaEventDisableTiming));
                for(auto& event:cache_->pipeline_prefill_layer_gate_up_ready_)
                    if(!event)MFQ_CUDA_CHECK(cudaEventCreateWithFlags(&event,cudaEventDisableTiming));
                for(auto& event:cache_->pipeline_prefill_layer_read_done_)
                    if(!event)MFQ_CUDA_CHECK(cudaEventCreateWithFlags(&event,cudaEventDisableTiming));
                // The allocation is stream ordered; copy readers wait for it.
                MFQ_CUDA_CHECK(cudaEventRecord(cache_->compute_done_,stream_));
                MFQ_CUDA_CHECK(cudaStreamWaitEvent(cache_->weight_stream_,cache_->compute_done_,0));
            }
        }
        auto prepared=std::make_unique<LayerPrefill>();
        prepared->layer_id=sources_[0]->layer_id_;
        prepared->slot=cache_->pipeline_prefill_layer_overlap_?
            int(std::distance(cache_->pipeline_prefill_layers_.begin(),
                cache_->pipeline_prefill_layers_.find(prepared->layer_id)))%2:0;
        prepared->ram_begin=layout.first;prepared->bytes=layout.second;
        auto* device=cache_->pipeline_prefill_layer_buffer_.data_ptr<uint8_t>()+
            prepared->slot*cache_->pipeline_prefill_layer_stride_;
        for(int p=0;p<3;++p) {
            const auto& source=*sources_[p];auto& runtime=prepared->projections[p];
            runtime.n_experts=experts_;runtime.neuron_len=source.cpu_->neuron_len;
            runtime.out_per_expert=source.cpu_->out_per_expert;
            runtime.prefill_only=true;runtime.partial_experts=false;
            for(const auto& cohort:source.cohorts_) {
                std::vector<int32_t> hot_map(experts_,-1),cold_map(experts_,-1);
                for(int e=0;e<experts_;++e)if(source.expert_to_cohort_[e]==cohort.index)
                    hot_map[e]=cohort.arena->book->slot_for({source.id_,cohort.index,e});
                auto hot=cohort.active;hot.expert_local=tb::tensor(hot_map).to(tb::kCUDA);
                runtime.pools.push_back(std::move(hot));
                const int slots=int(cohort.prefill_ram_experts.size());
                if(!slots)continue;
                std::vector<tb::Tensor> fields;
                for(std::size_t f=0;f<cohort.bytes_per_expert.size();++f) {
                    const auto& field=cohort.arena->layouts[f];auto shape=field.slot_shape;shape[0]*=slots;
                    const auto offset=cohort.prefill_ram_fields[f]-layout.first;
                    const auto bytes=slots*cohort.bytes_per_expert[f];
                    if(offset<0 || offset>layout.second || bytes>layout.second-offset)
                        throw std::logic_error("prefill field exceeds its layer image");
                    fields.push_back(tb::from_blob(device+offset,shape,
                        tb::TensorOptions().device(tb::kCUDA).dtype(field.scalar_type)));
                }
                for(int slot=0;slot<slots;++slot)cold_map[cohort.prefill_ram_experts[slot]]=slot;
                auto cold=fields_pool(cohort.active,fields,slots,runtime.out_per_expert);
                cold.expert_local=tb::tensor(cold_map).to(tb::kCUDA);runtime.pools.push_back(std::move(cold));
            }
            const auto* compact=std::getenv("MFQ_MOE_PREFILL_COMPACT_MMA");
            initialize_mixed_nvq_dispatch(runtime,execution.config,!compact || std::atoi(compact)!=0);
        }
        if(execution.config.moe_ffn_fused_activation && !cache_->pipeline_sigmoid_table_.defined())
            cache_->pipeline_sigmoid_table_=moe_swiglu_sigmoid_table_cuda();
        prefill_layer_=std::move(prepared);
        cache_->pipeline_prefill_release_.push_back([owner=weak_from_this()] {
            if(auto pipeline=owner.lock())pipeline->prefill_layer_.reset();
        });
    }

    void submit_prefill_layer_copy(CudaExecutionContext& execution,int layer_id,int slot) {
        if(cache_->pipeline_prefill_layer_loaded_[slot]==layer_id)return;
        const auto layout=cache_->pipeline_prefill_layers_.at(layer_id);
        const auto copy_stream=cache_->pipeline_prefill_layer_overlap_?cache_->weight_stream_:stream_;
        const auto gate_up_bytes=cache_->pipeline_prefill_layer_overlap_ && prefill_layer_phased()?
            cache_->pipeline_prefill_layer_gate_up_bytes_.at(layer_id):layout.second;
        if(gate_up_bytes<0 || gate_up_bytes>layout.second)
            throw std::logic_error("prefill Gate/Up copy exceeds layer image");
        if(cache_->pipeline_prefill_layer_overlap_ && cache_->pipeline_prefill_layer_read_recorded_[slot])
            MFQ_CUDA_CHECK(cudaStreamWaitEvent(copy_stream,cache_->pipeline_prefill_layer_read_done_[slot],0));
        {
#ifdef MFQ_NATIVE_CUDA_RUNTIME
        MfqCudaStreamGuard guard(MfqCudaStream(mfq_current_cuda_device(),copy_stream));
#else
        MfqCudaStreamGuard guard(at::cuda::getStreamFromExternal(copy_stream,mfq_current_cuda_device()));
#endif
        execution.profiler.measure("prefill.expert_h2d",[&] {
            auto* destination=cache_->pipeline_prefill_layer_buffer_.data_ptr<uint8_t>()+
                slot*cache_->pipeline_prefill_layer_stride_;
            const auto* source=cache_->pipeline_ram_complement_->data_ptr<uint8_t>()+layout.first;
            if(gate_up_bytes)MFQ_CUDA_CHECK(cudaMemcpyAsync(destination,source,std::size_t(gate_up_bytes),
                cudaMemcpyHostToDevice,copy_stream));
            if(cache_->pipeline_prefill_layer_overlap_)
                MFQ_CUDA_CHECK(cudaEventRecord(cache_->pipeline_prefill_layer_gate_up_ready_[slot],copy_stream));
            if(layout.second>gate_up_bytes)MFQ_CUDA_CHECK(cudaMemcpyAsync(destination+gate_up_bytes,
                source+gate_up_bytes,std::size_t(layout.second-gate_up_bytes),cudaMemcpyHostToDevice,copy_stream));
            return 0;
        });
        }
        if(cache_->pipeline_prefill_layer_overlap_)
            MFQ_CUDA_CHECK(cudaEventRecord(cache_->pipeline_prefill_layer_ready_[slot],copy_stream));
        cache_->pipeline_prefill_layer_loaded_[slot]=layer_id;
        if(cache_->pipeline_prefill_layer_overlap_ && prefill_layer_flush())
            mfq::cuda::check_stream_progress(copy_stream);
        auto& stats=cache_->stats_;
        const auto copies=int(gate_up_bytes>0)+int(layout.second>gate_up_bytes);
        stats.pipeline_direct_ram_copies+=copies;stats.pipeline_dma_copies+=copies;
        stats.pipeline_direct_ram_bytes+=layout.second;stats.ram_pcie_bytes+=layout.second;
        if(gate_up_bytes<layout.second) {
            ++stats.pipeline_phased_transfer_serves;
            stats.pipeline_gate_up_dma_bytes+=gate_up_bytes;
            stats.pipeline_down_dma_bytes+=layout.second-gate_up_bytes;
        }
    }

    tb::Tensor whole_layer_prefill(CudaExecutionContext& execution,const tb::Tensor& input,
            const tb::Tensor& ids,const tb::Tensor& weights) {
        // Bound submission to two layers of lookahead. The CPU only waits for
        // the old reader of this slot, while current GR/attention remain queued.
        // Copy-stream fences protect the other slot until its reader retires.
        const auto* reuse_option=std::getenv("MFQ_MOE_PREFILL_LAYER_REUSE_ASYNC");
        const bool same_layer=cache_->pipeline_prefill_layer_last_served_==sources_[0]->layer_id_ &&
            (!reuse_option || std::atoi(reuse_option)!=0);
        if(!cache_->pipeline_prefill_layer_buffer_.defined() || !cache_->pipeline_prefill_layer_overlap_ ||
                (!prefill_layer_async() && !same_layer))MFQ_CUDA_CHECK(cudaStreamSynchronize(stream_));
        initialize_prefill_layer(execution,ids);
        auto& layer=*prefill_layer_;
        if(cache_->pipeline_prefill_layer_overlap_ && prefill_layer_async() && !same_layer &&
                cache_->pipeline_prefill_layer_read_recorded_[layer.slot])
            MFQ_CUDA_CHECK(cudaEventSynchronize(cache_->pipeline_prefill_layer_read_done_[layer.slot]));
        submit_prefill_layer_copy(execution,layer.layer_id,layer.slot);
        if(cache_->pipeline_prefill_layer_overlap_) {
            MFQ_CUDA_CHECK(cudaStreamWaitEvent(stream_,cache_->pipeline_prefill_layer_gate_up_ready_[layer.slot],0));
            const auto next=std::next(cache_->pipeline_prefill_layers_.find(layer.layer_id));
            if(next!=cache_->pipeline_prefill_layers_.end())
                submit_prefill_layer_copy(execution,next->first,layer.slot^1);
        }
        ++cache_->stats_.pipeline_serves;
        auto shared=shared_?shared_(execution,input):tb::Tensor{};
        auto route=build_moe_route_plan(ids,experts_);
        const auto projection=[&](int p,const tb::Tensor& x) {
            auto& runtime=layer.projections[p];
            if(!cache_->pipeline_prefill_activations_)
                cache_->pipeline_prefill_activations_=std::make_shared<MixedMoeRuntime>();
            const int rows=int(x.numel()/x.size(-1));
            // Direct FP16 MMA consumes x; allocate shared Q8 scratch only when
            // the integer path actually needs it.
            if(!execution.config.moe_prefill_mma || execution.force_moe_prefill_mma_off)
            for(const auto& geometry:runtime.activation_geometry()) {
                const MixedMoeTransformKey transform{geometry.transform_block,geometry.transform_seed};
                const MixedMoeActivationKey key{rows,geometry.groups,geometry.gs,x.get_device(),transform};
                runtime.activation_workspaces.insert_or_assign(key,
                    cache_->pipeline_prefill_activations_->prefill_activation_workspace(
                        x,rows,geometry.groups,geometry.gs,transform));
            }
            auto output=execution.profiler.measure(p==0?"prefill.expert_gate":p==1?"prefill.expert_up":"prefill.expert_down",[&] {
                return runtime.forward(execution.config,execution.kl_mmq,execution.force_moe_prefill_mma_off,
                    execution.force_moe_pool_path,x,route);
            });
            runtime.activation_workspaces.clear();return output;
        };
        auto gate=projection(0,input),up=projection(1,input);
        auto hidden=execution.config.moe_ffn_fused_activation
            ? moe_swiglu_rounded_cuda(gate,up,cache_->pipeline_sigmoid_table_)
            : (gate*tb::sigmoid(gate))*up;
        if(cache_->pipeline_prefill_layer_overlap_) {
            // Submit Gate/Up independently of Down's completion on WDDM.
            if(prefill_layer_flush())mfq::cuda::check_stream_progress(stream_);
            MFQ_CUDA_CHECK(cudaStreamWaitEvent(stream_,cache_->pipeline_prefill_layer_ready_[layer.slot],0));
        }
        auto output=projection(2,hidden);
        if(shared.defined())output=moe_weighted_reduce_cuda(output,weights)+shared;
        if(cache_->pipeline_prefill_layer_overlap_) {
            MFQ_CUDA_CHECK(cudaEventRecord(cache_->pipeline_prefill_layer_read_done_[layer.slot],stream_));
            cache_->pipeline_prefill_layer_read_recorded_[layer.slot]=true;
        }
        if(cache_->pipeline_prefill_layer_overlap_ && prefill_layer_flush())
            mfq::cuda::check_stream_progress(stream_);
        cache_->pipeline_prefill_layer_last_served_=layer.layer_id;
        return output;
    }
