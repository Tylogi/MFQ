#include "timing.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cuda.h"
#include "common.cuh"
#include <memory>

namespace fb=format_bench;
static std::vector<float> read_floats(const std::string& path,std::size_t count) {
    std::ifstream file(path,std::ios::binary);std::vector<float> result(count);
    file.read(reinterpret_cast<char*>(result.data()),count*sizeof(float));
    if(!file || file.peek()!=std::char_traits<char>::eof())
        throw std::runtime_error("invalid float fixture: "+path);
    for(float x:result)if(!std::isfinite(x))throw std::runtime_error("nonfinite fixture");
    return result;
}
static ggml_type corresponding(const fb::Case& c) {
    if(c.label.find("NINT4")==0)return GGML_TYPE_Q4_K;
    if(c.label.find("NINT5")==0)return GGML_TYPE_Q5_K;
    if(c.label.find("NINT6")==0)return GGML_TYPE_Q6_K;
    if(c.label.find("NQ1")!=std::string::npos)return GGML_TYPE_IQ1_S;
    if(c.label.find("E8-256")!=std::string::npos)return GGML_TYPE_IQ2_XXS;
    if(c.label.find("E8-1024")!=std::string::npos)return GGML_TYPE_IQ2_XS;
    if(c.label.find("E8-4096")!=std::string::npos)return GGML_TYPE_Q2_K;
    if(c.label.find("D4-256")!=std::string::npos)return GGML_TYPE_IQ3_XXS;
    if(c.label.find("D4-512")!=std::string::npos || c.label.find("D4-1024")!=std::string::npos)
        return GGML_TYPE_IQ3_S;
    throw std::runtime_error("missing bpw counterpart: "+c.label);
}
struct Bank {
    ggml_context* context=nullptr;
    ggml_backend_buffer_t buffer=nullptr;
    ggml_cgraph* graph=nullptr;
    ggml_tensor* output=nullptr;
    Bank(ggml_backend_t backend,ggml_type type,int n,int k,int m,
         const std::vector<uint8_t>& packed,const std::vector<float>& input,int routes=1,bool routed_input=false) {
        ggml_init_params params{};
        params.mem_size=ggml_tensor_overhead()*6+ggml_graph_overhead_custom(32,false);
        params.no_alloc=true;context=ggml_init(params);
        if(!context)throw std::runtime_error("ggml metadata allocation failed");
        auto* weight=ggml_new_tensor_3d(context,type,k,n,routes);
        auto* x=routes==1 ? ggml_new_tensor_2d(context,GGML_TYPE_F32,k,m) :
            ggml_new_tensor_3d(context,GGML_TYPE_F32,k,routed_input?routes:1,m);
        auto* ids=routes==1 ? nullptr : ggml_new_tensor_2d(context,GGML_TYPE_I32,routes,m);
        output=ids ? ggml_mul_mat_id(context,weight,x,ids) : ggml_mul_mat(context,weight,x);
        graph=ggml_new_graph_custom(context,32,false);ggml_build_forward_expand(graph,output);
        buffer=ggml_backend_alloc_ctx_tensors(context,backend);
        if(!buffer)throw std::runtime_error("ggml device allocation failed");
        ggml_backend_tensor_set(weight,packed.data(),0,packed.size());
        ggml_backend_tensor_set(x,input.data(),0,input.size()*sizeof(float));
        if(ids) {
            std::vector<int32_t> selected(routes*m);
            for(size_t i=0;i<selected.size();++i)selected[i]=int32_t(i%routes);
            ggml_backend_tensor_set(ids,selected.data(),0,selected.size()*sizeof(int32_t));
        }
    }
    ~Bank(){if(buffer)ggml_backend_buffer_free(buffer);if(context)ggml_free(context);}
};
struct Replay {
    cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;
    std::size_t calls;
    Replay(ggml_backend_t backend,const std::vector<std::unique_ptr<Bank>>& banks,cudaStream_t stream):calls(banks.size()*8) {
        auto compute=[&](std::size_t i){
            if(ggml_backend_graph_compute_async(backend,banks[i%banks.size()]->graph)!=GGML_STATUS_SUCCESS)
                throw std::runtime_error("ggml compute failed");
        };
        for(std::size_t i=0;i<calls;++i)compute(i);
        fb::check(cudaStreamSynchronize(stream));
        fb::check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        for(std::size_t i=0;i<calls;++i)compute(i);
        fb::check(cudaStreamEndCapture(stream,&graph));
        fb::check(cudaGraphInstantiate(&exec,graph,0));
        fb::check(cudaGraphUpload(exec,stream));
    }
    ~Replay(){if(exec)cudaGraphExecDestroy(exec);if(graph)cudaGraphDestroy(graph);}
};
int main(int argc,char** argv)try {
    if(argc<3 || argc>4)throw std::runtime_error("usage: llama-format-bench cases.tsv mfq-fixture-dir [case-filter]");
    const std::string root=argv[2];
    const int routes=std::getenv("MFQ_BENCH_ROUTES") ? std::atoi(std::getenv("MFQ_BENCH_ROUTES")) : 1;
    if(routes!=1 && routes!=10)throw std::runtime_error("MFQ_BENCH_ROUTES must be 1 or 10");
    ggml_backend_t backend=ggml_backend_cuda_init(0);
    if(!backend)throw std::runtime_error("CUDA backend unavailable");
    auto* context=static_cast<ggml_backend_cuda_context*>(backend->context);
    const auto stream=context->stream();
    cudaDeviceProp properties{};fb::check(cudaGetDeviceProperties(&properties,0));
    for(const auto& c:fb::read_cases(argv[1])) {
        if(argc==4 && c.name().find(argv[3])==std::string::npos)continue;
        const auto type=corresponding(c);
        const int block=int(ggml_blck_size(type)),kp=(c.k+block-1)/block*block;
        const auto dense=read_floats(root+"/"+c.name()+(routes>1?".moe.weight.f32":".weight.f32"),std::size_t(c.n)*c.k*routes);
        std::vector<float> padded(std::size_t(c.n)*kp*routes,0.f);
        for(int row=0;row<c.n*routes;++row)std::copy_n(dense.data()+std::size_t(row)*c.k,c.k,padded.data()+std::size_t(row)*kp);
        std::vector<float> importance(kp,1.f);
        const std::size_t bytes=ggml_row_size(type,kp)*c.n*routes;
        std::vector<uint8_t> packed(bytes);
        if(ggml_quantize_chunk(type,padded.data(),packed.data(),0,c.n*routes,kp,importance.data())!=bytes)
            throw std::runtime_error("ggml quantization byte count mismatch");
        const auto count=std::max<std::size_t>(2,(4*std::size_t(properties.l2CacheSize)+bytes-1)/bytes);
        std::vector<int> batches=routes>1 ? std::vector<int>{1} : std::vector<int>{1,4,16,64};
        if(const char* selected=std::getenv("MFQ_BENCH_M"))batches={std::stoi(selected)};
        for(int m:batches) {
            const int rows=m*(routes>1 && c.k==640 ? routes : 1);
            auto original=read_floats(root+"/"+c.name()+(routes>1 ? ".moe.x.f32" : ".x"+std::to_string(m)+".f32"),std::size_t(rows)*c.k);
            std::vector<float> input(std::size_t(rows)*kp,0.f);
            for(int row=0;row<rows;++row)std::copy_n(original.data()+std::size_t(row)*c.k,c.k,input.data()+std::size_t(row)*kp);
            std::vector<std::unique_ptr<Bank>> banks;
            for(std::size_t i=0;i<count;++i)banks.push_back(std::make_unique<Bank>(backend,type,c.n,kp,m,packed,input,routes,c.k==640));
            Replay replay(backend,banks,stream);
            auto samples=fb::measure([&](){fb::check(cudaGraphLaunch(replay.exec,stream));},stream,replay.calls);
            std::vector<float> result(std::size_t(m)*c.n*routes);
            ggml_backend_tensor_get(banks.front()->output,result.data(),0,result.size()*sizeof(float));
            for(float value:result)if(!std::isfinite(value))throw std::runtime_error("nonfinite ggml output");
            fb::report(c,"llama.cpp",ggml_type_name(type),m,kp,bytes,count,properties.l2CacheSize,samples,routes);
            if(std::getenv("MFQ_BENCH_PROFILE_KERNELS"))
                fb::profile_kernels(replay.graph,stream,replay.calls,c.name());
        }
    }
    ggml_backend_free(backend);ggml_quantize_free();return 0;
}catch(const std::exception& error){std::cerr<<"FAIL: "<<error.what()<<'\n';return 1;}
