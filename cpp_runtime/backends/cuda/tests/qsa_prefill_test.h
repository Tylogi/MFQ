#pragma once

void qsa_prefill_fusion_case() {
    struct Setting {
        std::string old;
        Setting(){if(const auto* p=std::getenv("MFQ_QSA_PREFILL_FUSED"))old=p;}
        static void set(const char* p) {
#ifdef _WIN32
            _putenv_s("MFQ_QSA_PREFILL_FUSED",p);
#else
            if(*p)setenv("MFQ_QSA_PREFILL_FUSED",p,1);else unsetenv("MFQ_QSA_PREFILL_FUSED");
#endif
        }
        ~Setting(){set(old.c_str());}
    } setting;
    int cases=0,grouped_calls=0;
    for(int width:{8,256})for(bool grouped:{false,true}) {
        CudaExecutionContext execution;
        constexpr int hidden=32,kv=2,iw=8,ih=2,pool=4,maximum=2048,budget=512;
        const int heads=width==256?24:4;
        auto rotary=std::make_shared<RotaryEmbedding>(std::min(width,8),maximum,10000.);
        auto norm=data({width},1,0),inorm=data({iw},1,0);
        QsaWeights weights{linear(hidden,heads*2*width,1),linear(hidden,kv*width,2),linear(hidden,kv*width,3),
            linear(heads*width,hidden,4),linear(hidden,(ih+1)*iw,5),norm,norm,inorm,inorm};
        if(grouped)weights.input_projection=[q=weights.query,k=weights.key,v=weights.value,i=weights.index_query_key,&grouped_calls]
                (CudaExecutionContext& context,const Tensor& x) {
            ++grouped_calls;return std::vector<Tensor>{q(context,x),k(context,x),v(context,x),i(context,x)};
        };
        QsaConfig cfg{heads,kv,width,ih,iw,pool,budget,maximum,1e-6};
        Qsa actual(weights,cfg,rotary),oracle(weights,cfg,rotary);
        auto inputs=data({1,maximum,hidden},17,.4f);
        int64_t offset=0;
        for(int tokens:{128,257,511,512,1}) {
            auto positions=tb::arange(offset,offset+tokens,inputs.options().dtype(tb::kInt64));
            auto history=tb::arange(offset+tokens,positions.options());
            auto x=inputs.narrow(1,offset,tokens);
            Setting::set("0");auto expected=oracle.forward(execution,x,positions,history,true);
            Setting::set("1");auto result=actual.forward(execution,x,positions,history,true);
            equal(result,expected,"QSA prefill fused projection/normalization/RoPE");
            ++cases;offset+=tokens;
        }
        actual.truncate(385);oracle.truncate(385);
        auto positions=tb::arange(385,642,inputs.options().dtype(tb::kInt64));
        auto history=tb::arange(642,positions.options());auto x=inputs.narrow(1,385,257);
        Setting::set("0");auto expected=oracle.forward(execution,x,positions,history,true);
        Setting::set("1");auto result=actual.forward(execution,x,positions,history,true);
        equal(result,expected,"QSA prefill cache truncation and replay");++cases;
    }
    if(grouped_calls!=10)throw std::runtime_error("QSA prefill grouped projections were not used for every large chunk");
    std::cout<<"QSA prefill fusion cases="<<cases<<" grouped_calls="<<grouped_calls
             <<" dense/sparse prefixes, changing chunk sizes, decode transition, cache truncation PASS\n";
}
