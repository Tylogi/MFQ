void verify_compact_prefill_mma(const std::filesystem::path& root) {
    CudaExecutionContext execution;
    int cases=0;float maximum_relative=0;
    for(const auto* name:{"nvq2j","nvq2j-l","nvq2j-xl","nvq3j","nvq3j-512","nvq3j-l"})
        for(bool down:{false,true}) {
        auto model=mfq::open_model_source((root/(std::string(name)+(down?"-2560-640.mfq":"-640-2560.mfq"))).string());
        auto cpu=load_mfe_cpu(*model,"linear.weight");
        auto compact=make_mixed_moe_runtime(cpu,false,execution.config);
        std::size_t canonical_bytes=0,compact_bytes=0;
        for(auto& pool:compact->pools) {
            canonical_bytes+=bytes(pool);
            if(!prepare_nvq_dense_groups(pool.nvq))throw std::runtime_error("compact prefill fixture did not pack NVQ");
            compact_bytes+=bytes(pool);
            auto& w=pool.nvq;
            for(auto* field:{&w.indices_packed,&w.aux_packed,&w.sub_scale_packed,&w.neuron_scale,&w.codebook})
                *field=copy_cpu_weight_to_cuda(*field);
            pool.expert_local=pool.expert_local.to(tb::kCUDA);
        }
        if(compact_bytes>canonical_bytes)throw std::runtime_error("compact prefill expanded packed weights");
        initialize_mixed_nvq_dispatch(*compact,execution.config,true);
        if(!compact->nvq_dispatch || !compact->nvq_dispatch->dense_groups)
            throw std::runtime_error("compact prefill Tensor Core dispatch absent");
        for(int tokens:{64,257,512})for(bool routed:{false,true}) {
            constexpr int routes=3;
            const int width=cpu.neuron_len,output=cpu.out_per_expert;
            std::vector<float> values(std::size_t(tokens)*(routed?routes:1)*width);
            std::vector<int32_t> ids(tokens*routes);
            for(std::size_t i=0;i<values.size();++i)values[i]=std::sin(float(i)*.017f)*.11f;
            for(std::size_t i=0;i<ids.size();++i)ids[i]=int(i)%cpu.n_experts;
            auto x=tb::tensor(values).to(tb::kFloat16).to(tb::kCUDA);
            x=routed?x.reshape({tokens,routes,width}):x.reshape({tokens,width});
            const auto route=build_moe_route_plan(tb::tensor(ids).reshape({tokens,routes}).to(tb::kCUDA),cpu.n_experts);
            auto actual=compact->forward(execution.config,execution.kl_mmq,false,false,x,route)
                .reshape({tokens*routes,output}).cpu().to(tb::kFloat32);
            // Decode the canonical packed weights independently, then use
            // cuBLAS FP16 GEMM. No compact-reader code participates here.
            auto expected=mfe_dense_reference(*model,"linear.weight",x,ids,tokens,routes,routed)
                .cpu().to(tb::kFloat32);
            double squared=0,reference_squared=0;float max_difference=0,max_reference=0;
            for(int64_t i=0;i<actual.numel();++i) {
                const float a=actual.data_ptr<float>()[i],b=expected.data_ptr<float>()[i];
                if(!std::isfinite(a)||!std::isfinite(b))throw std::runtime_error("nonfinite compact prefill projection");
                const float d=std::abs(a-b);max_difference=std::max(max_difference,d);
                max_reference=std::max(max_reference,std::abs(b));squared+=double(d)*d;reference_squared+=double(b)*b;
            }
            const float relative=float(std::sqrt(squared/std::max(reference_squared,1e-20)));
            maximum_relative=std::max(maximum_relative,relative);
            std::cout<<"compact_prefill_oracle format="<<name<<" down="<<down<<" tokens="<<tokens<<" routed="<<routed
                     <<" max_abs="<<max_difference<<" relative_rms="<<relative<<"\n";
            if(relative>.002f || max_difference>.004f+.002f*max_reference)
                throw std::runtime_error("compact prefill differs from canonical-weight FP16 GEMM");
            ++cases;
        }
    }
    std::cout<<"compact_prefill_mma_cases="<<cases<<" max_relative_rms="<<maximum_relative
             <<" packed_weight_bytes_unchanged=1 PASS\n";
}
