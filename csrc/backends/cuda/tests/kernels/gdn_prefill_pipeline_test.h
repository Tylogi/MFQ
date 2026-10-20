#pragma once
#include "native/env.h"

class GdnPipelineScope {
    std::string old_;
    bool present_;
    const char* key_;
public:
    explicit GdnPipelineScope(int mode,const char* key="MFQ_GDN_PREFILL_PIPELINED"):
        present_(mfq::cuda::env_lookup(key)!=nullptr),key_(key) {
        if(present_)old_=mfq::cuda::env_lookup(key);set(mode,key);
    }
    static void set(int mode,const char* key="MFQ_GDN_PREFILL_PIPELINED") {
        const auto value=mode<0?std::string{}:std::to_string(mode);
        if(value.empty())mfq::cuda::env_clear(key);else mfq::cuda::env_set(key,value.c_str());
    }
    ~GdnPipelineScope() {
        if(present_)mfq::cuda::env_set(key_,old_.c_str());else mfq::cuda::env_clear(key_);
    }
};
void gdn_prefill_pipeline_case(bool defaults=false,bool prepared_decay=false) {
    GdnPipelineScope pipeline(1);
    const char* option_key=prepared_decay?"MFQ_GDN_PREFILL_PREPARED_DECAY":"MFQ_GDN_PREFILL_PIPELINED";
    GdnPipelineScope option(1,option_key);
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
            const int mode=order^(repeat&1);GdnPipelineScope::set(mode?(defaults?-1:1):0,option_key);
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
            <<" prepared_decay="<<prepared_decay
            <<" tokens="<<tokens<<" transposed="<<transposed<<" baseline_ms="<<times[0][times[0].size()/2]
            <<" candidate_ms="<<times[1][times[1].size()/2]<<" output_state_bits_equal=1"<<std::endl;
    }
}

void gdn_prefill_staged_case(bool benchmark) {
    GdnPipelineScope staged(0,"MFQ_GDN_PREFILL_STAGED");
    GdnPipelineScope pipeline(1);
    GdnColumnsScope columns(4);
    const auto bits=[](const Tensor& actual,const Tensor& expected) {
        auto a=actual.cpu().contiguous(),b=expected.cpu().contiguous();
        if(a.sizes()!=b.sizes() || std::memcmp(a.data_ptr(),b.data_ptr(),a.nbytes()))
            throw std::runtime_error("staged GDN changed output/state bits");
    };
    int cases=0;
    for(int batch:{1,2})for(int tokens:{9,15,16,17,31,129,1024})
    for(bool transposed:{false,true})for(bool tiled:{false,true})for(bool initial:{false,true}) {
        const int nk=2,nv=6,D=128;
        auto q=data({batch,nk,tokens,D},401,.04f).to(tb::kFloat32),k=data({batch,nk,tokens,D},402,.04f).to(tb::kFloat32);
        auto v=data({batch,nv,tokens,D},403,.19f).to(tb::kFloat32);
        auto g=data({batch,nv,tokens},404,.3f).to(tb::kFloat32)-.5f;
        auto beta=data({batch,nv,tokens},405,.3f).to(tb::kFloat32)+.5f;
        auto state=data({batch,nv,D,D},406,.03f).to(tb::kFloat32);
        const auto call=[&] {
            if(tiled) {
                auto s=initial?state.clone():tb::zeros(state.sizes(),state.options());
                return (transposed?gdn_inplace_transposed_tiled_cuda:gdn_inplace_tiled_cuda)(q,k,v,g,beta,s);
            }
            MfqOptional<Tensor> input;if(initial)input=state;
            return (transposed?gdn_transposed_cuda:gdn_cuda)(q,k,v,g,beta,input);
        };
        GdnPipelineScope::set(0,"MFQ_GDN_PREFILL_STAGED");auto expected=call();
        GdnPipelineScope::set(1,"MFQ_GDN_PREFILL_STAGED");auto actual=call();
        bits(actual[0],expected[0]);bits(actual[1],expected[1]);++cases;
    }
    if(benchmark)for(int tokens:{1024,4048,8192})for(int nv:{48,64}) {
        constexpr int nk=16,D=128;
        auto q=data({1,nk,tokens,D},411,.04f).to(tb::kFloat32),k=data({1,nk,tokens,D},412,.04f).to(tb::kFloat32);
        auto v=data({1,nv,tokens,D},413,.19f).to(tb::kFloat32);
        auto g=data({1,nv,tokens},414,.3f).to(tb::kFloat32)-.5f;
        auto beta=data({1,nv,tokens},415,.3f).to(tb::kFloat32)+.5f;
        auto state=data({1,nv,D,D},416,.03f).to(tb::kFloat32);
        std::vector<float> times[2];std::vector<Tensor> reference;
        for(int sample=0;sample<7;++sample)for(int order=0;order<2;++order) {
            const int mode=(sample+order)%2;GdnPipelineScope::set(mode,"MFQ_GDN_PREFILL_STAGED");
            auto result=gdn_transposed_cuda(q,k,v,g,beta,state);mfq_cuda_synchronize();
            cudaEvent_t start,end;MFQ_CUDA_CHECK(cudaEventCreate(&start));MFQ_CUDA_CHECK(cudaEventCreate(&end));
            const auto stream=mfq_current_cuda_stream();MFQ_CUDA_CHECK(cudaEventRecord(start,stream));
            for(int i=0;i<4;++i)result=gdn_transposed_cuda(q,k,v,g,beta,state);
            MFQ_CUDA_CHECK(cudaEventRecord(end,stream));MFQ_CUDA_CHECK(cudaEventSynchronize(end));
            float ms=0;MFQ_CUDA_CHECK(cudaEventElapsedTime(&ms,start,end));cudaEventDestroy(start);cudaEventDestroy(end);
            if(sample)times[mode].push_back(ms/4);
            if(reference.empty())reference={result[0].clone(),result[1].clone()};
            else {bits(result[0],reference[0]);bits(result[1],reference[1]);}
        }
        for(auto& values:times)std::sort(values.begin(),values.end());
        std::cout<<"gdn_prefill_staged tokens="<<tokens<<" heads="<<nv<<" baseline_ms="<<times[0][3]
            <<" staged_ms="<<times[1][3]<<" output_state_bits_equal=1"<<std::endl;
    }
    std::cout<<"GDN staged exact cases="<<cases<<" PASS\n";
}

void gdn_prefill_staged_graph_case() {
    GdnPipelineScope staged(0,"MFQ_GDN_PREFILL_STAGED");
    GdnColumnsScope columns(4);
    const auto bits=[](const Tensor& actual,const Tensor& expected) {
        auto a=actual.cpu().contiguous(),b=expected.cpu().contiguous();
        if(a.sizes()!=b.sizes() || std::memcmp(a.data_ptr(),b.data_ptr(),a.nbytes()))
            throw std::runtime_error("staged GDN changing graph changed output/state bits");
    };
    int replays=0;
    for(int tokens:{17,129,4099})for(int nv:{48,64})
    for(bool transposed:{false,true})for(bool tiled:{false,true})for(bool inplace:{false,true}) {
        if(tokens==4099 && !(transposed && !tiled && inplace))continue;
        constexpr int nk=16,D=128;
        auto q=data({1,nk,tokens,D},421,.04f).to(tb::kFloat32),k=data({1,nk,tokens,D},422,.04f).to(tb::kFloat32);
        auto q0=q.clone(),k0=k.clone(),v=data({1,nv,tokens,D},423,.19f).to(tb::kFloat32);
        auto g=data({1,nv,tokens},424,.3f).to(tb::kFloat32)-.5f;
        auto beta=data({1,nv,tokens},425,.3f).to(tb::kFloat32)+.5f;
        auto initial=data({1,nv,D,D},426,.03f).to(tb::kFloat32),state=initial.clone();
        const auto call=[&](Tensor& s) {
            if(tiled)return (transposed?gdn_inplace_transposed_tiled_cuda:gdn_inplace_tiled_cuda)(q,k,v,g,beta,s);
            if(inplace)return (transposed?gdn_inplace_transposed_cuda:gdn_inplace_cuda)(q,k,v,g,beta,s);
            return (transposed?gdn_transposed_cuda:gdn_cuda)(q,k,v,g,beta,s);
        };
        GdnPipelineScope::set(tokens==17?1:-1,"MFQ_GDN_PREFILL_STAGED");
        std::vector<Tensor> actual;DecodeWindow window(mfq_current_cuda_stream());
        window.capture([&]{actual=call(state);},[&]{state.copy_(initial);},[&]{actual.clear();});
        for(float amplitude:{.5f,1.f,1.5f}) {
            q.copy_(q0*amplitude);k.copy_(k0*amplitude);state.copy_(initial);
            GdnPipelineScope::set(0,"MFQ_GDN_PREFILL_STAGED");auto reference_state=initial.clone();auto expected=call(reference_state);
            window.run();bits(actual[0],expected[0]);bits(actual[1],expected[1]);
            if(inplace && actual[1].data_ptr()!=state.data_ptr())throw std::runtime_error("staged GDN replaced inplace state");
            ++replays;
        }
    }
    std::cout<<"GDN staged changing graph/default/inplace exact replays="<<replays<<" PASS\n";
}
