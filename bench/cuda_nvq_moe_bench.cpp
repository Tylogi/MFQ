#include "packed_bench_utils.h"
#include "mfq_cuda_moe_ops.h"
#include "moe.h"
#include "mfq/model_source.h"
#include <array>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>

namespace {
namespace tb=mfq_tensor_backend;
const std::array<int,6> tiles={32,16,8,4,2,1};
void select(int rows) {
#ifdef _WIN32
    _putenv_s("MFQ_NVQ_MOE_ROW_TILE",std::to_string(rows).c_str());
#else
    setenv("MFQ_NVQ_MOE_ROW_TILE",std::to_string(rows).c_str(),1);
#endif
}
void exact(const tb::Tensor& output,const tb::Tensor& expected) {
    const auto actual=output.to(tb::kCPU).contiguous();
    if(actual.sizes()!=expected.sizes() || actual.scalar_type()!=expected.scalar_type() ||
       std::memcmp(actual.data_ptr(),expected.data_ptr(),actual.numel()*actual.element_size()))
        throw std::runtime_error("routed NVQ row tile changed original output bits");
}
struct Bank {
    MixedMoeRuntime runtime;
    std::size_t bytes=0,selected_weight_bytes=0;
    Bank(const std::filesystem::path& root,int rows,int width,bool mixed) {
        runtime.n_experts=512;runtime.out_per_expert=rows;runtime.neuron_len=width;runtime.partial_experts=true;
        const std::vector<std::string> formats=mixed
            ? std::vector<std::string>{"nvq1-s","nvq1-l","nvq2j","nvq2j-l","nvq2j-xl","nvq3j","nvq3j-512","nvq3j-l"}
            : std::vector<std::string>{"nvq3j-l","nvq3j-l","nvq3j-l","nvq3j-l"};
        CudaExecutionConfig config;int group=0;
        for(const auto& format:formats) {
            auto source=mfq::open_model_source((root/(format+"-"+std::to_string(rows)+"-"+std::to_string(width)+".mfq")).string());
            auto single=make_mixed_moe_runtime(load_mfe_cpu(*source,"linear.weight"),true,config);
            for(auto& pool:single->pools) {
                if(pool.family!=MixedMoeFamily::Nvq)throw std::runtime_error("benchmark fixture is not NVQ");
                if(pool.local_experts!=3)throw std::runtime_error("benchmark fixture must contain three experts");
                std::vector<int32_t> local(512,-1);for(int e=0;e<3;++e)local[group*3+e]=e;
                pool.expert_local=tb::tensor(local).to(tb::kCUDA);
                for(const auto* field:{&pool.nvq.indices_packed,&pool.nvq.aux_packed,&pool.nvq.sub_scale_packed,&pool.nvq.neuron_scale,&pool.nvq.codebook})
                    bytes+=std::size_t(field->numel()*field->element_size());
                const int selected=mixed ? (group<2?2:1) : (group<3?3:1);
                // Count selected expert fields only. Excluding shared codebooks
                // gives a conservative bound for the rotating weight footprint.
                for(const auto* field:{&pool.nvq.indices_packed,&pool.nvq.aux_packed,&pool.nvq.sub_scale_packed,&pool.nvq.neuron_scale})
                    selected_weight_bytes+=std::size_t(field->numel()*field->element_size())*selected/3;
                runtime.pools.push_back(std::move(pool));
            }
            ++group;
        }
        initialize_mixed_nvq_dispatch(runtime,config);
        if(!runtime.nvq_dispatch)throw std::runtime_error("benchmark NVQ dispatch unavailable");
    }
    explicit Bank(const Bank& original):runtime(original.runtime),bytes(original.bytes),selected_weight_bytes(original.selected_weight_bytes) {
        // Independent physical fields make a rotating working set; tensor
        // values stay identical to the retained legal compressed fixtures.
        for(auto& pool:runtime.pools) {
            pool.nvq.indices_packed=pool.nvq.indices_packed.clone();pool.nvq.aux_packed=pool.nvq.aux_packed.clone();
            pool.nvq.sub_scale_packed=pool.nvq.sub_scale_packed.clone();pool.nvq.neuron_scale=pool.nvq.neuron_scale.clone();
            pool.nvq.codebook=pool.nvq.codebook.clone();pool.expert_local=pool.expert_local.clone();
        }
        initialize_mixed_nvq_dispatch(runtime,CudaExecutionConfig{});
    }
};
struct Call {
    Bank& bank;
    int rows,width,tokens;
    bool routed;
    tb::Tensor x,ids,out,qx,xscale;
    Call(Bank& b,int n,int k,int m,bool route):bank(b),rows(n),width(k),tokens(m),routed(route) {
        const auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat16);
        x=tb::zeros(routed?std::vector<int64_t>{m,10,k}:std::vector<int64_t>{m,k},options);
        ids=tb::zeros({m,10},options.dtype(tb::kInt32));out=tb::zeros({m,10,n},options);
        const int ng=(k+23)/24,input_rows=routed?m*10:m;
        qx=tb::empty({input_rows,ng*24},options.dtype(tb::kInt8));
        xscale=tb::empty({input_rows,ng},options.dtype(tb::kFloat32));
        update(0);
    }
    void update(int step) {
        std::vector<float> values(x.numel());
        for(std::size_t i=0;i<values.size();++i)values[i]=std::sin(float(i+step*19)*.071f)*.13f;
        x.copy_(tb::tensor(values).reshape(x.sizes().vec()).to(tb::kFloat16).to(tb::kCUDA));
        std::vector<int32_t> selected(tokens*10);
        const bool mixed=bank.runtime.pools.size()==8;
        const int pattern[10]={0,3,6,9,12,15,18,21,1,4};
        for(int i=0;i<tokens*10;++i) {
            int expert=mixed?pattern[(i+step)%10]:(i+step)%12;
            if(step==1)expert=511;
            if(step==2 && i%10==7)expert=-1;
            if(step==2 && i%10==8)expert=512;
            if(step==2 && i%10==9)expert=selected[i-3];
            selected[i]=expert;
        }
        ids.copy_(tb::tensor(selected).reshape({tokens,10}).to(tb::kCUDA));
        std::vector<int32_t> local(512,-1),pools(512,-1);
        for(std::size_t p=0;p<bank.runtime.pools.size();++p)for(int e=0;e<3;++e) {
            const auto expert=p*3+e;
            if(step==3 && e==0)continue;
            pools[expert]=int32_t(p);local[expert]=e;
        }
        bank.runtime.nvq_dispatch->expert_local.copy_(tb::tensor(local).to(tb::kCUDA));
        bank.runtime.nvq_dispatch->expert_pool.copy_(tb::tensor(pools).to(tb::kCUDA));
    }
    void run() {
        const auto& d=*bank.runtime.nvq_dispatch;
        nvq_moe_grouped_matmul_hetero_ws_cuda(d.weight_ptrs,d.weight_sizes,d.pool_params,d.expert_pool,d.expert_local,
            x,ids,512,rows,width,false,out,qx,xscale);
    }
};
class Graph {
    mfq::cuda::StreamHandle stream_=mfq_get_stream_from_pool(false);
    cudaGraph_t graph_=nullptr,kernel_graph_=nullptr;
    cudaGraphExec_t exec_=nullptr,kernel_exec_=nullptr;
    std::vector<std::unique_ptr<Call>> calls_;
public:
    int registers=0,local_bytes=0,shared_bytes=0;
    std::size_t count=0,kernel_nodes=0;
    Graph(std::vector<std::unique_ptr<Bank>>& banks,std::size_t sets,int rows,int width,bool routed,int tile) {
        MfqCudaStreamGuard guard(stream_);select(tile);count=sets*8;
        for(std::size_t i=0;i<count;++i)calls_.push_back(std::make_unique<Call>(*banks[i%sets],rows,width,1,routed));
        for(auto& call:calls_)call->run();MFQ_CUDA_CHECK(cudaStreamSynchronize(stream_.stream()));
        MFQ_CUDA_CHECK(cudaStreamBeginCapture(stream_.stream(),cudaStreamCaptureModeThreadLocal));
        for(auto& call:calls_)call->run();
        MFQ_CUDA_CHECK(cudaStreamEndCapture(stream_.stream(),&graph_));
        MFQ_CUDA_CHECK(cudaGraphInstantiate(&exec_,graph_,0));
        std::size_t nodes_count=0;MFQ_CUDA_CHECK(cudaGraphGetNodes(graph_,nullptr,&nodes_count));
        std::vector<cudaGraphNode_t> nodes(nodes_count);MFQ_CUDA_CHECK(cudaGraphGetNodes(graph_,nodes.data(),&nodes_count));
        std::vector<std::pair<std::size_t,cudaKernelNodeParams>> kernels;
        for(auto node:nodes) {
            cudaGraphNodeType type;MFQ_CUDA_CHECK(cudaGraphNodeGetType(node,&type));if(type!=cudaGraphNodeTypeKernel)continue;
            ++kernel_nodes;cudaKernelNodeParams params{};MFQ_CUDA_CHECK(cudaGraphKernelNodeGetParams(node,&params));
            const char* name=nullptr;MFQ_CUDA_CHECK(cudaFuncGetName(&name,params.func));
            if(!name || !std::strstr(name,"nvq_moe_mmvq_hetero_kernel"))continue;
            const auto marker=std::string("ILi4ELi")+std::to_string(tile)+"E";
            if(!std::strstr(name,marker.c_str()))throw std::runtime_error("routed NVQ selected wrong row tile");
            cudaFuncAttributes attrs{};MFQ_CUDA_CHECK(cudaFuncGetAttributes(&attrs,params.func));
            registers=attrs.numRegs;local_bytes=int(attrs.localSizeBytes);shared_bytes=int(attrs.sharedSizeBytes);
            const auto output=*static_cast<void**>(params.kernelParams[8]);std::size_t index=0;
            while(index<count && calls_[index]->out.data_ptr()!=output)++index;
            if(index==count)throw std::runtime_error("routed NVQ graph output ownership invalid");
            kernels.emplace_back(index,params);
        }
        if(kernels.size()!=count || kernel_nodes!=count*2)throw std::runtime_error("routed NVQ graph coverage incomplete");
        std::sort(kernels.begin(),kernels.end(),[](const auto& a,const auto& b){return a.first<b.first;});
        MFQ_CUDA_CHECK(cudaGraphCreate(&kernel_graph_,0));cudaGraphNode_t previous=nullptr;
        for(std::size_t i=0;i<count;++i) {
            if(kernels[i].first!=i)throw std::runtime_error("routed NVQ graph order invalid");
            cudaGraphNode_t node;MFQ_CUDA_CHECK(cudaGraphAddKernelNode(&node,kernel_graph_,previous?&previous:nullptr,previous?1:0,&kernels[i].second));previous=node;
        }
        MFQ_CUDA_CHECK(cudaGraphInstantiate(&kernel_exec_,kernel_graph_,0));
        MFQ_CUDA_CHECK(cudaGraphUpload(exec_,stream_.stream()));MFQ_CUDA_CHECK(cudaGraphUpload(kernel_exec_,stream_.stream()));
        MFQ_CUDA_CHECK(cudaGraphLaunch(exec_,stream_.stream()));MFQ_CUDA_CHECK(cudaStreamSynchronize(stream_.stream()));
        validate();
    }
    ~Graph() {
        if(kernel_exec_)cudaGraphExecDestroy(kernel_exec_);if(exec_)cudaGraphExecDestroy(exec_);
        if(kernel_graph_)cudaGraphDestroy(kernel_graph_);if(graph_)cudaGraphDestroy(graph_);
    }
    void validate() {
        MfqCudaStreamGuard guard(stream_);select(2);
        for(auto& call:calls_) {
            const auto observed=call->out.to(tb::kCPU).contiguous();call->run();exact(call->out,observed);
        }
    }
    double measure(bool pure,std::size_t repeats) {
        cudaEvent_t begin=nullptr,end=nullptr;MFQ_CUDA_CHECK(cudaEventCreate(&begin));MFQ_CUDA_CHECK(cudaEventCreate(&end));
        MFQ_CUDA_CHECK(cudaEventRecord(begin,stream_.stream()));
        for(std::size_t i=0;i<repeats;++i)MFQ_CUDA_CHECK(cudaGraphLaunch(pure?kernel_exec_:exec_,stream_.stream()));
        MFQ_CUDA_CHECK(cudaEventRecord(end,stream_.stream()));MFQ_CUDA_CHECK(cudaEventSynchronize(end));
        float ms=0;MFQ_CUDA_CHECK(cudaEventElapsedTime(&ms,begin,end));
        MFQ_CUDA_CHECK(cudaEventDestroy(begin));MFQ_CUDA_CHECK(cudaEventDestroy(end));return ms;
    }
};
int check(Bank& bank,int rows,int width,bool routed) {
    int cases=0;auto stream=mfq_get_stream_from_pool(false);MfqCudaStreamGuard guard(stream);
    for(int output:{rows,rows-1})for(int tokens:{1,3,8})for(int tile:tiles) {
        Call call(bank,output,width,tokens,routed);select(tile);cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;
        MFQ_CUDA_CHECK(cudaStreamBeginCapture(stream.stream(),cudaStreamCaptureModeThreadLocal));
        call.out.zero_();call.run();MFQ_CUDA_CHECK(cudaStreamEndCapture(stream.stream(),&graph));
        MFQ_CUDA_CHECK(cudaGraphInstantiate(&exec,graph,0));
        for(int step=0;step<4;++step) {
            call.update(step);select(2);call.out.zero_();call.run();const auto expected=call.out.to(tb::kCPU).contiguous();
            select(tile);call.out.zero_();call.run();exact(call.out,expected);++cases;
            MFQ_CUDA_CHECK(cudaGraphLaunch(exec,stream.stream()));MFQ_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
            exact(call.out,expected);++cases;
        }
        MFQ_CUDA_CHECK(cudaGraphExecDestroy(exec));MFQ_CUDA_CHECK(cudaGraphDestroy(graph));
    }
    return cases;
}
void benchmark(std::vector<std::unique_ptr<Bank>>& banks,int rows,int width,bool mixed,bool routed,int l2) {
    for(bool rotating:{false,true}) {
        const auto count=rotating?banks.size():std::size_t(1);
        Graph original(banks,count,rows,width,routed,2);
        for(int tile:tiles) {
            Graph current(banks,count,rows,width,routed,tile);
            for(bool pure:{true,false}) {
                original.measure(pure,1);current.measure(pure,1);std::size_t repeats=1;
                while(std::min(original.measure(pure,repeats),current.measure(pure,repeats))<20) {
                    if(repeats>std::numeric_limits<std::size_t>::max()/2)throw std::runtime_error("benchmark repeat count overflow");
                    repeats*=2;
                }
                std::vector<double> before,after;
                for(int sample=0;sample<7;++sample) {
                    double a,b;if(sample%2){b=current.measure(pure,repeats);a=original.measure(pure,repeats);}
                    else {a=original.measure(pure,repeats);b=current.measure(pure,repeats);}
                    before.push_back(a*1000/(repeats*original.count));after.push_back(b*1000/(repeats*current.count));
                }
                // Verify the actual retained timing outputs after the clock.
                current.validate();original.validate();
                std::cout<<std::setprecision(9)<<"{\"backend\":\"cuda\",\"operator\":\"routed_nvq\",\"rows\":"<<rows
                    <<",\"width\":"<<width<<",\"M\":1,\"routes\":10,\"old_row_tile\":2,\"row_tile\":"<<tile
                    <<",\"formats\":\""<<(mixed?"mixed8":"nvq3j_l")<<"\",\"input\":\""<<(routed?"routed":"shared")
                    <<"\",\"cache\":\""<<(rotating?"rotating":"fixed")<<"\",\"working_sets\":"<<count
                    <<",\"working_bytes\":"<<banks[0]->bytes*count<<",\"device_l2_bytes\":"<<l2
                    <<",\"selected_weight_bytes\":"<<banks[0]->selected_weight_bytes*count
                    <<",\"timing_scope\":\""<<(pure?"kernel_device":"quantization_and_gemv_device")
                    <<"\",\"calls_per_graph\":"<<current.count<<",\"kernel_nodes\":"<<current.kernel_nodes
                    <<",\"registers\":"<<current.registers<<",\"local_bytes\":"<<current.local_bytes
                    <<",\"shared_bytes\":"<<current.shared_bytes<<",\"samples\":7,\"repeats\":"<<repeats
                    <<",\"exact_original_bits\":true";
                mfq::bench::timing(before,after);std::cout<<"}\n"<<std::flush;
            }
        }
    }
}
}
int main(int argc,char** argv)try {
    if(argc<2 || argc>3)throw std::invalid_argument("usage: mfq-cuda-nvq-moe-bench fixture-dir [--check-only|--bench-only]");
    const bool check_only=argc==3 && std::string(argv[2])=="--check-only";
    const bool bench_only=argc==3 && std::string(argv[2])=="--bench-only";
    if(argc==3 && !check_only && !bench_only)throw std::invalid_argument("unknown benchmark mode");
    auto context=mfq::cuda::default_context(mfq_current_cuda_device());
    cudaDeviceProp properties{};MFQ_CUDA_CHECK(cudaGetDeviceProperties(&properties,0));int cases=0;
    for(const auto shape:{std::pair<int,int>{640,2560},{2560,640}})for(bool mixed:{false,true}) {
        std::vector<std::unique_ptr<Bank>> banks;banks.push_back(std::make_unique<Bank>(argv[1],shape.first,shape.second,mixed));
        if(!bench_only)for(bool routed:{false,true})cases+=check(*banks[0],shape.first,shape.second,routed);
        if(check_only)continue;
        const auto count=std::max<std::size_t>(2,(std::size_t(properties.l2CacheSize)*4+banks[0]->selected_weight_bytes-1)/banks[0]->selected_weight_bytes);
        for(std::size_t i=1;i<count;++i)banks.push_back(std::make_unique<Bank>(*banks[0]));
        for(bool routed:{false,true})benchmark(banks,shape.first,shape.second,mixed,routed,properties.l2CacheSize);
    }
    select(2);
    if(!bench_only)std::cout<<"NVQ original Half bits ordinary/changing graphs cases="<<cases<<" PASS\n";
    else std::cout<<"NVQ timed output bits exact configurations=192 PASS\n";
    return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
