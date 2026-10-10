#pragma once

class GdnColumnsScope {
    std::string previous_;
    bool present_;
public:
    explicit GdnColumnsScope(int columns):present_(std::getenv("MFQ_GDN_PREFILL_COLUMNS")!=nullptr) {
        if(present_)previous_=std::getenv("MFQ_GDN_PREFILL_COLUMNS");
        set(columns);
    }
    static void set(int columns) {
        const auto value=std::to_string(columns);
#ifdef _WIN32
        if(_putenv_s("MFQ_GDN_PREFILL_COLUMNS",value.c_str()))throw std::runtime_error("GDN columns option");
#else
        if(setenv("MFQ_GDN_PREFILL_COLUMNS",value.c_str(),1))throw std::runtime_error("GDN columns option");
#endif
    }
    ~GdnColumnsScope() {
#ifdef _WIN32
        _putenv_s("MFQ_GDN_PREFILL_COLUMNS",present_?previous_.c_str():"");
#else
        if(present_)setenv("MFQ_GDN_PREFILL_COLUMNS",previous_.c_str(),1);
        else unsetenv("MFQ_GDN_PREFILL_COLUMNS");
#endif
    }
};

void gdn_prefill_columns_case() {
    GdnColumnsScope option(0);
    const auto bits=[](const Tensor& a,const Tensor& b,const char* label) {
        auto x=a.cpu().contiguous(),y=b.cpu().contiguous();
        if(x.sizes()!=y.sizes() || std::memcmp(x.data_ptr(),y.data_ptr(),x.numel()*x.element_size()))
            throw std::runtime_error(std::string(label)+" differs");
    };
    int cases=0;
    for(int batch:{1,2})for(int tokens:{1,8,31,1024})
    for(bool transposed:{false,true})for(bool tiled:{false,true})for(bool initial:{false,true}) {
        constexpr int width=128,nk=2,nv=6;
        auto q=data({batch,nk,tokens,width},361,.04f).to(tb::kFloat32);
        auto k=data({batch,nk,tokens,width},362,.04f).to(tb::kFloat32);
        auto v=data({batch,nv,tokens,width},363,.19f).to(tb::kFloat32);
        auto g=tb::ones({batch,nv,tokens},q.options())*(-.07f);
        auto beta=tb::ones({batch,nv,tokens},q.options())*.21f;
        auto state=data({batch,nv,width,width},364,.03f).to(tb::kFloat32);
        const auto call=[&](bool inplace) {
            if(tiled) {
                auto s=initial?state.clone():tb::zeros(state.sizes(),state.options());
                return (transposed?gdn_inplace_transposed_tiled_cuda:gdn_inplace_tiled_cuda)(q,k,v,g,beta,s);
            }
            if(inplace) {
                auto s=state.clone();
                return (transposed?gdn_inplace_transposed_cuda:gdn_inplace_cuda)(q,k,v,g,beta,s);
            }
            MfqOptional<Tensor> input;
            if(initial)input=state;
            return (transposed?gdn_transposed_cuda:gdn_cuda)(q,k,v,g,beta,input);
        };
        GdnColumnsScope::set(0);auto expected=call(false);
        for(int columns:{2,4}) {
            GdnColumnsScope::set(columns);auto actual=call(false);
            bits(actual[0],expected[0],"GDN multi-column output");
            bits(actual[1],expected[1],"GDN multi-column state");++cases;
            if(initial && !tiled) {
                actual=call(true);bits(actual[0],expected[0],"GDN inplace output");
                bits(actual[1],expected[1],"GDN inplace state");++cases;
            }
        }
    }
    // Independent FP64 recurrence: decay, dot, rank-one update, and query dot.
    constexpr int D=128,Hq=2,Hv=6,T=17;
    auto q=data({1,Hq,T,D},371,.04f).to(tb::kFloat32),k=data({1,Hq,T,D},372,.04f).to(tb::kFloat32);
    auto v=data({1,Hv,T,D},373,.19f).to(tb::kFloat32),state=data({1,Hv,D,D},374,.03f).to(tb::kFloat32);
    auto g=tb::ones({1,Hv,T},q.options())*(-.07f),beta=tb::ones({1,Hv,T},q.options())*.21f;
    auto qc=q.cpu(),kc=k.cpu(),vc=v.cpu(),sc=state.cpu(),gc=g.cpu(),bc=beta.cpu();
    std::vector<float> expected_output(Hv*T*D),expected_state(Hv*D*D);
    for(int h=0;h<Hv;++h) {
        const int kh=h/(Hv/Hq);std::vector<double> s(D*D);
        for(int i=0;i<D*D;++i)s[i]=sc.data_ptr<float>()[h*D*D+i];
        for(int t=0;t<T;++t) {
            const double decay=std::exp(double(gc.data_ptr<float>()[h*T+t]));
            for(int col=0;col<D;++col) {
                double dot=0;
                for(int row=0;row<D;++row)dot+=s[row*D+col]*kc.data_ptr<float>()[(kh*T+t)*D+row];
                const double delta=(vc.data_ptr<float>()[(h*T+t)*D+col]-decay*dot)*bc.data_ptr<float>()[h*T+t];
                double attended=0;
                for(int row=0;row<D;++row) {
                    s[row*D+col]=decay*s[row*D+col]+kc.data_ptr<float>()[(kh*T+t)*D+row]*delta;
                    attended+=s[row*D+col]*qc.data_ptr<float>()[(kh*T+t)*D+row];
                }
                expected_output[(h*T+t)*D+col]=float(attended/std::sqrt(double(D)));
            }
        }
        for(int i=0;i<D*D;++i)expected_state[h*D*D+i]=float(s[i]);
    }
    for(int columns:{2,4}) {
        GdnColumnsScope::set(columns);auto actual=gdn_cuda(q,k,v,g,beta,state);
        equal(actual[0],tb::tensor(expected_output).reshape({1,Hv,T,D}),"GDN FP64 output oracle",2e-7f,5e-5f);
        equal(actual[1],tb::tensor(expected_state).reshape({1,Hv,D,D}),"GDN FP64 state oracle",2e-7f,5e-5f);
    }
    std::cout<<"GDN prefill multi-column exact cases="<<cases<<" plus FP64 output/state oracle PASS\n";
}

void gdn_prefill_columns_benchmark() {
    GdnColumnsScope option(0);
    for(int nv:{48,64}) {
        constexpr int nk=16,width=128,tokens=1024;
        auto q=data({1,nk,tokens,width},381,.04f).to(tb::kFloat32),k=data({1,nk,tokens,width},382,.04f).to(tb::kFloat32);
        auto v=data({1,nv,tokens,width},383,.19f).to(tb::kFloat32);
        auto g=tb::ones({1,nv,tokens},q.options())*(-.07f),beta=tb::ones({1,nv,tokens},q.options())*.21f;
        auto state=data({1,nv,width,width},384,.03f).to(tb::kFloat32);
        for(int columns:{0,2,4}) {
            GdnColumnsScope::set(columns);std::vector<Tensor> result;
            for(int i=0;i<3;++i)result=gdn_transposed_cuda(q,k,v,g,beta,state);
            cudaEvent_t begin,end;cudaEventCreate(&begin);cudaEventCreate(&end);
            cudaEventRecord(begin,mfq_current_cuda_stream());
            for(int i=0;i<30;++i)result=gdn_transposed_cuda(q,k,v,g,beta,state);
            cudaEventRecord(end,mfq_current_cuda_stream());cudaEventSynchronize(end);
            float elapsed=0;cudaEventElapsedTime(&elapsed,begin,end);cudaEventDestroy(begin);cudaEventDestroy(end);
            std::cout<<"gdn_prefill_columns_bench key_heads="<<nk<<" value_heads="<<nv
                <<" tokens="<<tokens<<" columns="<<columns<<" milliseconds="<<elapsed/30<<'\n';
        }
    }
}
