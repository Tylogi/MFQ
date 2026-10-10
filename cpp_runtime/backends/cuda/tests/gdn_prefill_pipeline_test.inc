class GdnPipelineScope {
    std::string old_;
    bool present_;
public:
    explicit GdnPipelineScope(int mode):present_(std::getenv("MFQ_GDN_PREFILL_PIPELINED")!=nullptr) {
        if(present_)old_=std::getenv("MFQ_GDN_PREFILL_PIPELINED");set(mode);
    }
    static void set(int mode) {
        const auto value=mode<0?std::string{}:std::to_string(mode);
#ifdef _WIN32
        _putenv_s("MFQ_GDN_PREFILL_PIPELINED",value.c_str());
#else
        if(value.empty())unsetenv("MFQ_GDN_PREFILL_PIPELINED");else setenv("MFQ_GDN_PREFILL_PIPELINED",value.c_str(),1);
#endif
    }
    ~GdnPipelineScope() {
#ifdef _WIN32
        _putenv_s("MFQ_GDN_PREFILL_PIPELINED",present_?old_.c_str():"");
#else
        if(present_)setenv("MFQ_GDN_PREFILL_PIPELINED",old_.c_str(),1);else unsetenv("MFQ_GDN_PREFILL_PIPELINED");
#endif
    }
};
void gdn_prefill_pipeline_case(bool defaults=false) {
    GdnPipelineScope option(1);
    gdn_prefill_columns_case();
    GdnColumnsScope columns(4);
    for(int nv:{48,64})for(bool transposed:{false,true}) {
        constexpr int nk=16,width=128,tokens=1024;
        auto q=data({1,nk,tokens,width},391,.04f).to(tb::kFloat32),k=data({1,nk,tokens,width},392,.04f).to(tb::kFloat32);
        auto v=data({1,nv,tokens,width},393,.19f).to(tb::kFloat32);
        auto g=tb::ones({1,nv,tokens},q.options())*(-.07f),beta=tb::ones({1,nv,tokens},q.options())*.21f;
        auto state=data({1,nv,width,width},394,.03f).to(tb::kFloat32);
        std::vector<float> times[2];std::vector<Tensor> reference;
        for(int repeat=0;repeat<8;++repeat)for(int order=0;order<2;++order) {
            const int mode=order^(repeat&1);GdnPipelineScope::set(mode?(defaults?-1:1):0);
            const auto call=[&] {return (transposed?gdn_transposed_cuda:gdn_cuda)(q,k,v,g,beta,state);};
            auto result=call();result=call();mfq_cuda_synchronize();
            cudaEvent_t begin,end;MFQ_CUDA_CHECK(cudaEventCreate(&begin));MFQ_CUDA_CHECK(cudaEventCreate(&end));
            MFQ_CUDA_CHECK(cudaEventRecord(begin,mfq_current_cuda_stream()));
            for(int i=0;i<8;++i)result=call();
            MFQ_CUDA_CHECK(cudaEventRecord(end,mfq_current_cuda_stream()));MFQ_CUDA_CHECK(cudaEventSynchronize(end));
            float elapsed=0;MFQ_CUDA_CHECK(cudaEventElapsedTime(&elapsed,begin,end));cudaEventDestroy(begin);cudaEventDestroy(end);
            for(int i=0;i<2;++i) {
                auto checked=result[i].cpu().contiguous();
                if(reference.size()<2)reference.push_back(checked.clone());
                if(std::memcmp(checked.data_ptr(),reference[i].data_ptr(),checked.nbytes()))
                    throw std::runtime_error("GDN pipeline changed output/state bits");
            }
            if(repeat>=2)times[mode].push_back(elapsed/8);
        }
        for(auto& values:times)std::sort(values.begin(),values.end());
        std::cout<<"gdn_prefill_pipeline_pair key_heads="<<nk<<" value_heads="<<nv
            <<" tokens="<<tokens<<" transposed="<<transposed<<" baseline_ms="<<times[0][times[0].size()/2]
            <<" candidate_ms="<<times[1][times[1].size()/2]<<" output_state_bits_equal=1"<<std::endl;
    }
}
