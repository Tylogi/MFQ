#pragma once
#include "mfe_ffn_runtime.h"
#include "mfe_expert_plan_test.h"
#include "storage/weight_loader.h"

namespace mfe_expert_plan_test {
struct Option {
    std::string previous;
    bool present=false;
    Option() {
        if(const char* value=std::getenv("MFQ_MFE_EXPERT_PLAN")) {
            previous=value;
            present=true;
        }
    }
    static void select(const char* value) {
#ifdef _WIN32
        _putenv_s("MFQ_MFE_EXPERT_PLAN",value?value:"");
#else
        if(value)setenv("MFQ_MFE_EXPERT_PLAN",value,1);
        else unsetenv("MFQ_MFE_EXPERT_PLAN");
#endif
    }
    ~Option(){select(present?previous.c_str():nullptr);}
};
inline void verify_ffn(const std::filesystem::path& root,bool timings) {
    namespace tb=mfq_tensor_backend;
    Option option;
    CudaExecutionContext execution;
    auto context=mfq::cuda::default_context(mfq_current_cuda_device());
    const auto stream=mfq_current_cuda_stream();
    context->begin_graph_pool(stream);
    const auto sigmoid=moe_swiglu_sigmoid_table_cuda();
    int cases=0;
    for(const char* family:{"nint","nvq","mxfp4"}) {
        std::array<std::shared_ptr<MixedMoeRuntime>,3> projections;
        for(int p=0;p<3;++p) {
            auto model=mfq::open_model_source((root/(std::string("plan-")+family+
                (p==2?"-down.mfq":"-up.mfq"))).string());
            projections[p]=make_mixed_moe_runtime(load_mfe_cpu(*model,"linear.weight"),true,execution.config);
        }
        constexpr int width=384,intermediate=96;
        const int experts=projections[0]->n_experts;
        for(int tokens:{2,8,16,32})for(int routes:{8,22})for(int pattern:{0,1}) {
            const auto options=tb::TensorOptions().device(tb::kCUDA);
            auto input=tb::zeros({tokens,width},options.dtype(tb::kFloat16));
            auto ids=tb::zeros({tokens,routes},options.dtype(tb::kInt32));
            auto weights=tb::full({tokens,routes},1.0/routes,options.dtype(tb::kFloat32));
            MfeFfnRuntime runtime({projections[0].get(),projections[1].get(),projections[2].get()},
                input,ids,weights,sigmoid,{},{},width,intermediate,nullptr,false,false,8);
            std::array<tb::Tensor,2> expected;
            for(bool planned:{false,true}) {
                Option::select(planned?"1":"0");
                mfq::cuda::DecodeWindow window(stream);
                window.capture([&]{runtime.run();},[]{},[]{});
                for(int step=0;step<2;++step) {
                    std::vector<float> values(tokens*width);
                    std::vector<int32_t> selected(tokens*routes);
                    for(std::size_t i=0;i<values.size();++i)values[i]=std::sin(float(i+step*31)*.017f)*.13f;
                    for(int pair=0;pair<tokens*routes;++pair)
                        selected[pair]=((pattern?pair:pair%routes)+step*37)%experts;
                    input.copy_(tb::tensor(values).reshape({tokens,width}).to(tb::kFloat16).to(tb::kCUDA));
                    ids.copy_(tb::tensor(selected).reshape({tokens,routes}).to(tb::kCUDA));
                    window.run();
                    auto actual=runtime.output().cpu().contiguous();
                    if(!planned)expected[step]=actual.clone();
                    else require(!std::memcmp(actual.data_ptr(),expected[step].data_ptr(),actual.nbytes()),
                        "expert-major plan changed complete FFN output on graph replay");
                }
                if(timings) {
                    const double prepare=time([&]{runtime.prepare();},stream);
                    const double ffn=time([&]{runtime.run();},stream,64,32);
                    std::cout<<"mfe_expert_ffn family="<<family<<" tokens="<<tokens<<" routes="<<routes
                        <<" pattern="<<pattern<<" planned="<<planned<<" prepare_us="<<prepare
                        <<" ffn_us="<<ffn<<'\n';
                }
                if(planned) {
                    Option::select(nullptr);
                    runtime.run();
                    auto automatic=runtime.output().cpu().contiguous();
                    require(!std::memcmp(automatic.data_ptr(),expected[1].data_ptr(),automatic.nbytes()),
                        "automatic expert-plan dispatch changed FFN output");
                    if(timings) {
                        const double prepare=time([&]{runtime.prepare();},stream);
                        const double ffn=time([&]{runtime.run();},stream,64,32);
                        std::cout<<"mfe_expert_ffn family="<<family<<" tokens="<<tokens<<" routes="<<routes
                            <<" pattern="<<pattern<<" planned=auto prepare_us="<<prepare<<" ffn_us="<<ffn<<'\n';
                    }
                }
                ++cases;
            }
        }
    }
    MFQ_CUDA_CHECK(cudaStreamSynchronize(stream));
    context->end_graph_pool(stream);
    std::cout<<"mfe_expert_plan_ffn_cases="<<cases<<" PASS\n";
}
} // namespace mfe_expert_plan_test
