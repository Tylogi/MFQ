// Keep each compressed expert image resident while advancing all prompt chunks
// through that layer. Attention and PLE still advance in causal token order.
bool qwen4_layer_major_prefill_eligible(const Qwen4CausalLm& model, int64_t tokens) {
    const auto* enabled=std::getenv("MFQ_MOE_PREFILL_LAYER_MAJOR");
    const auto* whole=std::getenv("MFQ_MOE_PREFILL_LAYER");
    if((enabled && std::atoi(enabled)==0) || !whole || std::atoi(whole)==0 || tokens<=1024 ||
       model.cache_pos!=0 || model.speculative_start>=0 || !model.execution ||
       !model.execution->config.moe_pipeline || !model.execution->moe_expert_cache ||
       model.execution->dense_cpu_layer_count || model.execution->tensor_parallel.enabled() ||
       model.execution->expert_parallel.enabled())return false;
    const auto primary=model.execution->layer_placement.primary_device();
    for(const auto& block:model.blocks)
        if(block->cpu_offloaded || block->cuda_device!=primary)return false;
    return true;
}

mfq_tensor_backend::Tensor qwen4_layer_major_prefill_hidden(
        Qwen4CausalLm& model,mfq_tensor_backend::Tensor ids,int64_t compute_chunk,
        bool last_only,const std::function<bool()>& cancelled) {
    namespace tb=mfq_tensor_backend;
    using Tensor=tb::Tensor;
    MFQ_RUNTIME_CHECK(ids.dim()==2 && ids.size(0)==1 &&
        qwen4_layer_major_prefill_eligible(model,ids.size(1)),
        "layer-major prefill requires a fresh single-device Qwen4 prompt and whole-layer cache");
    MFQ_RUNTIME_CHECK(compute_chunk>8,"layer-major compute chunk must exceed decode width");
    auto scope=model.execution_scope();
    ids=model.device_ids(std::move(ids));
    const auto tokens=ids.size(1);
    model.adapter_validate_forward(1,tokens,0,false,false,false);
    model.reset(1);
    model.adapter_begin_forward(false);
    auto current=tb::arange(tokens,ids.options());
    if(model.decode_position_delta)current=current+model.decode_position_delta;
    auto prepared=model.adapter_prepare_positions(std::move(current),1,tokens);
    model.adapter_validate_positions(prepared.positions,1,tokens,false);
    auto cpu_ids=ids.to(tb::kCPU,tb::kInt64).contiguous();
    Tensor hidden=tb::empty({1,tokens,model.hidden_size()*model.hc_mult()},
                           ids.options().dtype(tb::kFloat16));
    for(int64_t begin=0;begin<tokens;begin+=compute_chunk) {
        if(cancelled && cancelled())return {};
        const auto count=std::min(compute_chunk,tokens-begin);
        auto input=model.adapter_prepare_hidden(model.embed_tokens(ids.narrow(1,begin,count)),1,count);
        MFQ_RUNTIME_CHECK(input.scalar_type()==hidden.scalar_type() &&
            input.size(-1)==hidden.size(-1),"layer-major embedding geometry changed");
        hidden.narrow(1,begin,count).copy_(input);
    }
    auto& profiler=model.execution->profiler;
    struct ProfileRestore {
        CudaProfiler& profiler;bool enabled;
        ~ProfileRestore(){profiler.enabled=enabled;}
    } restore{profiler,profiler.enabled};
    const auto* trace_option=std::getenv("MFQ_PREFILL_LAYER_MAJOR_TRACE");
    const bool trace=trace_option && std::atoi(trace_option)!=0;
    const auto* batch_option=std::getenv("MFQ_MOE_PREFILL_EXPERT_BATCH");
    const bool batch_experts=!batch_option || std::atoi(batch_option)!=0;
    const qwen4_exp::QwenPrefillReferenceRowsScope reference_rows(batch_experts?std::min<int64_t>(1024,compute_chunk):0);
    const auto* batch_tokens_option=std::getenv("MFQ_MOE_PREFILL_EXPERT_BATCH_TOKENS");
    const int64_t batch_tokens=batch_tokens_option?std::atoll(batch_tokens_option):8192;
    MFQ_RUNTIME_CHECK(!batch_experts || batch_tokens>=1024,"expert prefill batch must cover at least 1024 tokens");
    const auto* audit_option=std::getenv("MFQ_PREFILL_EXPERT_BATCH_AUDIT");
    const bool audit_batch=audit_option && std::atoi(audit_option)!=0;
    const auto started=std::chrono::steady_clock::now();
    int layer=0;
    for(auto& block:model.blocks) {
        Tensor next_hidden=hidden;
        Tensor ffn_inputs,ffn_injection;
        for(int64_t begin=0;begin<tokens;begin+=compute_chunk) {
            if(cancelled && cancelled())return {};
            const auto count=std::min(compute_chunk,tokens-begin);
            profiler.enabled=restore.enabled;
            Block::Context context;
            context.token_ids=ids.narrow(1,begin,count);
            context.host_token_ids=cpu_ids.narrow(1,begin,count);
            context.positions=prepared.positions.narrow(-1,begin,count);
            context.full_positions=prepared.full_positions.narrow(-1,0,begin+count);
            context.cache_position=begin;
            block->set_token_ids(context.token_ids);
            block->prefetch_token_ids(context.host_token_ids);
            auto input=hidden.narrow(1,begin,count);
            Tensor output;
            if(batch_experts) {
                auto& qwen_block=static_cast<qwen4_exp::Qwen4Block&>(*block);
                auto mix=qwen_block.prefill_attention(*model.execution,input,context);
                if(!ffn_inputs.defined()) {
                    ffn_inputs=tb::empty({1,tokens,model.hidden_size()},mix[0].options().dtype(tb::kFloat16));
                    ffn_injection=tb::empty({1,tokens,mix[2].size(-1)},mix[2].options());
                }
                // qwen_moe casts the router/FFN source to half at this boundary.
                ffn_inputs.narrow(1,begin,count).copy_(mix[0].to(tb::kFloat16));
                ffn_injection.narrow(1,begin,count).copy_(mix[2]);
                output=std::move(mix[1]);
            } else output=block->forward_context(*model.execution,input,context,model.rope);
            // Retain the original rounding at every layer boundary. The next
            // chunk is independent of this input once its cache writes retire.
            MFQ_RUNTIME_CHECK(output.sizes()==input.sizes(),"layer-major decoder changed hidden shape");
            if(begin==0 && output.scalar_type()!=hidden.scalar_type()) {
                next_hidden=tb::empty(hidden.sizes(),output.options());
                if(trace)std::cout<<"prefill_layer_major_hidden layer="<<layer
                    <<" input_bytes="<<hidden.nbytes()<<" output_bytes="<<next_hidden.nbytes()<<std::endl;
            }
            MFQ_RUNTIME_CHECK(output.scalar_type()==next_hidden.scalar_type(),
                "layer-major decoder output dtype changed within a layer");
            next_hidden.narrow(1,begin,count).copy_(output);
        }
        if(batch_experts) {
            if(cancelled && cancelled())return {};
            auto& qwen_block=static_cast<qwen4_exp::Qwen4Block&>(*block);
            profiler.enabled=restore.enabled;
            auto branch=profiler.measure("prefill.ffn",[&] {
                if(tokens<=batch_tokens)return qwen_block.ffn(*model.execution,ffn_inputs);
                Tensor result;
                for(int64_t begin=0;begin<tokens;begin+=batch_tokens) {
                    if(cancelled && cancelled())return Tensor{};
                    const auto count=std::min(batch_tokens,tokens-begin);
                    auto part=qwen_block.ffn(*model.execution,ffn_inputs.narrow(1,begin,count));
                    if(!result.defined()) {
                        auto shape=part.sizes().vec();shape[1]=tokens;
                        result=tb::empty(shape,part.options());
                    }
                    result.narrow(1,begin,count).copy_(part);
                }
                return result;
            });
            if(!branch.defined())return {};
            if(audit_batch) {
                auto expected=tb::empty(branch.sizes(),branch.options());
                for(int64_t begin=0;begin<tokens;begin+=compute_chunk) {
                    const auto count=std::min(compute_chunk,tokens-begin);
                    expected.narrow(1,begin,count).copy_(
                        qwen_block.ffn(*model.execution,ffn_inputs.narrow(1,begin,count)));
                }
                auto a=branch.to(tb::kCPU,tb::kFloat32).contiguous();
                auto b=expected.to(tb::kCPU,tb::kFloat32).contiguous();
                double error=0,scale=0,maximum=0;
                for(int64_t i=0;i<a.numel();++i) {
                    const double delta=double(a.data_ptr<float>()[i])-b.data_ptr<float>()[i];
                    error+=delta*delta;scale+=double(b.data_ptr<float>()[i])*b.data_ptr<float>()[i];
                    maximum=std::max(maximum,std::abs(delta));
                }
                std::cout<<"prefill_expert_batch_layer_oracle layer="<<layer<<" tokens="<<tokens
                    <<" relative_rms="<<std::sqrt(error/std::max(1e-30,scale))
                    <<" maximum_absolute="<<maximum<<std::endl;
                // Diagnostic only: continue with the reference so later
                // layers are tested with their original input distribution.
                branch=std::move(expected);
            }
            Tensor post_hidden=next_hidden;
            for(int64_t begin=0;begin<tokens;begin+=compute_chunk) {
                const auto count=std::min(compute_chunk,tokens-begin);
                std::vector<Tensor> mix{{},next_hidden.narrow(1,begin,count),ffn_injection.narrow(1,begin,count)};
                auto output=qwen_block.ffn_gr.post(branch.narrow(1,begin,count),mix);
                if(begin==0 && output.scalar_type()!=next_hidden.scalar_type())
                    post_hidden=tb::empty(next_hidden.sizes(),output.options());
                MFQ_RUNTIME_CHECK(output.scalar_type()==post_hidden.scalar_type(),
                    "expert-batched residual dtype changed");
                post_hidden.narrow(1,begin,count).copy_(output);
            }
            next_hidden=std::move(post_hidden);
        }
        hidden=std::move(next_hidden);
        ++layer;
        if(trace && (layer%4==0 || layer==int(model.blocks.size()))) {
            MFQ_CUDA_CHECK(cudaStreamSynchronize(mfq_get_current_cuda_stream()));
#ifdef MFQ_NATIVE_CUDA_RUNTIME
            const auto memory=mfq::cuda::default_context(ids.get_device())->memory_stats();
            const auto local=mfq::cuda::default_context(ids.get_device())->local_memory_usage();
            cudaMemPool_t pool=nullptr;uint64_t reserved=0,used=0;
            MFQ_CUDA_CHECK(cudaDeviceGetDefaultMemPool(&pool,ids.get_device()));
            MFQ_CUDA_CHECK(cudaMemPoolGetAttribute(pool,cudaMemPoolAttrReservedMemCurrent,&reserved));
            MFQ_CUDA_CHECK(cudaMemPoolGetAttribute(pool,cudaMemPoolAttrUsedMemCurrent,&used));
            std::cout<<"prefill_layer_major_memory tokens="<<tokens<<" chunk="<<compute_chunk
                <<" completed_layers="<<layer<<" owned_bytes="<<memory.allocated
                <<" process_local_bytes="<<local<<" pool_used_bytes="<<used<<" pool_reserved_bytes="<<reserved<<std::endl;
#endif
            std::cout<<"prefill_layer_major_progress tokens="<<tokens<<" chunk="<<compute_chunk
                <<" completed_layers="<<layer<<" total_layers="<<model.blocks.size()
                <<" elapsed_ms="<<std::chrono::duration<double,std::milli>(
                    std::chrono::steady_clock::now()-started).count()<<std::endl;
        }
    }
    profiler.enabled=restore.enabled;
    model.adapter_finish_forward(prepared.full_positions,1,tokens);
    model.cache_pos=tokens;
    if(last_only) {
        // Use the same final GR dispatch and rounding as chunk-major prefill.
        const auto begin=((tokens-1)/compute_chunk)*compute_chunk;
        auto finalized=model.finalize_hidden(hidden.narrow(1,begin,tokens-begin),1,tokens-begin);
        return model.last_hidden(std::move(finalized)).unsqueeze(1);
    }
    Tensor finalized;
    for(int64_t begin=0;begin<tokens;begin+=compute_chunk) {
        const auto count=std::min(compute_chunk,tokens-begin);
        auto part=model.finalize_hidden(hidden.narrow(1,begin,count),1,count);
        if(!finalized.defined())finalized=tb::empty({1,tokens,part.size(-1)},part.options());
        finalized.narrow(1,begin,count).copy_(part);
    }
    return finalized;
}
