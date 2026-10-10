#include "packed_bench_utils.h"
#include "mfq_cuda_moe_ops.h"
#include "mfe_ffn_runtime.h"
#include "mfq/model_source.h"
#include <array>
#include <atomic>
#include <cstring>
#include <filesystem>
#include <functional>
#include <limits>
#include <memory>
#include <thread>

namespace {
namespace tb=mfq_tensor_backend;
constexpr int hidden=2560,intermediate=640,routes=10,experts=512;
const std::array<std::string,12> formats={"nint4","nint5","nint6","nint8","nvq1-s","nvq1-l",
    "nvq2j","nvq2j-l","nvq2j-xl","nvq3j","nvq3j-512","nvq3j-l"};
void exact(const tb::Tensor& actual,const tb::Tensor& expected,const std::string& label) {
    const auto a=actual.to(tb::kCPU).contiguous(),b=expected.to(tb::kCPU).contiguous();
    if(a.sizes()!=b.sizes() || a.scalar_type()!=b.scalar_type())throw std::runtime_error(label+" shape/dtype differs");
    if(!std::memcmp(a.data_ptr(),b.data_ptr(),a.numel()*a.element_size()))return;
    const auto* x=static_cast<const unsigned char*>(a.data_ptr());const auto* y=static_cast<const unsigned char*>(b.data_ptr());
    std::size_t i=0;while(i<std::size_t(a.numel()*a.element_size()) && x[i]==y[i])++i;
    const auto af=a.to(tb::kFloat32),bf=b.to(tb::kFloat32);const auto index=i/a.element_size();
    throw std::runtime_error(label+" differs at element "+std::to_string(index)+": "+
        std::to_string(af.data_ptr<float>()[index])+" versus "+std::to_string(bf.data_ptr<float>()[index]));
}
std::vector<tb::Tensor*> fields(MixedMoePool& pool) {
    if(pool.family==MixedMoeFamily::Nint)return {&pool.nint.q_packed,&pool.nint.row_q_bits,&pool.nint.row_q_bit_offsets,
        &pool.nint.sub_scale,&pool.nint.sub_min,&pool.nint.neuron_scale,&pool.nint.neuron_min};
    return {&pool.nvq.indices_packed,&pool.nvq.aux_packed,&pool.nvq.sub_scale_packed,&pool.nvq.neuron_scale,&pool.nvq.codebook};
}
NintWeight first_expert(NintWeight w,int rows) {
    w.out=rows;w.shape={rows,w.neuron_len};w.workspaces.clear();
    w.q_packed=w.q_packed.narrow(0,0,1).contiguous().reshape({-1});
    for(auto* field:{&w.row_q_bits,&w.row_q_bit_offsets,&w.sub_scale,&w.sub_min,&w.neuron_scale,&w.neuron_min})
        *field=field->narrow(0,0,rows).contiguous();
    return w;
}
struct Bank {
    std::array<MixedMoeRuntime,3> projections;
    std::array<NintWeight,3> shared;
    std::size_t bytes=0,selected_weight_bytes=0;
    int groups=0;
    Bank(const std::filesystem::path& root,const std::string& layout) {
        groups=layout=="mixed12"?12:4;CudaExecutionConfig config;
        for(int p=0;p<3;++p) {
            auto& runtime=projections[p];runtime.n_experts=experts;
            runtime.out_per_expert=p==2?hidden:intermediate;runtime.neuron_len=p==2?intermediate:hidden;
            runtime.partial_experts=true;
            for(int group=0;group<groups;++group) {
                const auto format=layout=="mixed12"?formats[(group+p*4)%formats.size()]:layout;
                const auto filename=format+"-"+std::to_string(runtime.out_per_expert)+"-"+std::to_string(runtime.neuron_len)+".mfq";
                auto source=mfq::open_model_source((root/filename).string());
                auto single=make_mixed_moe_runtime(load_mfe_cpu(*source,"linear.weight"),true,config);
                if(single->pools.size()!=1 || single->pools[0].local_experts!=3)
                    throw std::runtime_error("MFE benchmark needs legal three-expert single-format fixtures");
                auto pool=std::move(single->pools[0]);std::vector<int32_t> local(experts,-1);
                for(int e=0;e<3;++e)local[group*3+e]=e;
                pool.expert_local=tb::tensor(local).to(tb::kCUDA);
                int selected=0;for(int route=0;route<routes;++route)if(selected_expert(route)/3==group)++selected;
                for(auto* field:fields(pool)) {
                    const auto n=std::size_t(field->numel()*field->element_size());bytes+=n;
                    if(pool.family==MixedMoeFamily::Nint || field!=&pool.nvq.codebook)selected_weight_bytes+=n*selected/3;
                }
                runtime.pools.push_back(std::move(pool));
            }
            initialize_mixed_nvq_dispatch(runtime,config);
            const auto shared_name="nint5-"+std::to_string(runtime.out_per_expert)+"-"+std::to_string(runtime.neuron_len)+".mfq";
            auto source=mfq::open_model_source((root/shared_name).string());
            auto single=make_mixed_moe_runtime(load_mfe_cpu(*source,"linear.weight"),true,config);
            shared[p]=first_expert(single->pools[0].nint,runtime.out_per_expert);
        }
    }
    Bank(const Bank& other):projections(other.projections),shared(other.shared),bytes(other.bytes),
            selected_weight_bytes(other.selected_weight_bytes),groups(other.groups) {
        for(auto& runtime:projections) {
            runtime.activation_workspaces.clear();runtime.nint_input_plans.clear();runtime.nvq_active_plans.clear();
            for(auto& pool:runtime.pools) {
                for(auto* field:fields(pool))*field=field->clone();pool.expert_local=pool.expert_local.clone();
                pool.nint.workspaces.clear();pool.nvq.workspaces.clear();
            }
            initialize_mixed_nvq_dispatch(runtime,CudaExecutionConfig{});
        }
        for(auto& w:shared) {
            for(auto* field:{&w.q_packed,&w.row_q_bits,&w.row_q_bit_offsets,&w.sub_scale,&w.sub_min,&w.neuron_scale,&w.neuron_min})
                *field=field->clone();w.workspaces.clear();
        }
    }
    int selected_expert(int route)const {
        const int uniform[10]={0,3,6,9,1,4,7,10,2,5};
        return groups==12?route*3:uniform[route];
    }
    void maps(int step) {
        for(auto& runtime:projections)for(int group=0;group<groups;++group) {
            std::vector<int32_t> local(experts,-1);
            for(int e=0;e<3;++e)if(step!=3 || e!=0)local[group*3+e]=e;
            runtime.pools[group].expert_local.copy_(tb::tensor(local).to(tb::kCUDA));
        }
    }
};
struct Call {
    Bank& bank;int tokens,output_width,ffn_width;bool with_shared,cpu;
    CudaExecutionContext execution;
    tb::Tensor input,ids,weights,shared_gate,kinds,cpu_pairs,cpu_mask,quant_descriptors,oracle_hidden,oracle_output;
    int quant_groups=0;
    std::unique_ptr<MfeFfnRuntime> fused;
    Call(Bank& b,int m,bool shared,bool cpu_result,bool cropped=false,bool shared_float=true):bank(b),tokens(m),
            output_width(hidden-(cropped?1:0)),ffn_width(intermediate-(cropped?1:0)),with_shared(shared),cpu(cpu_result) {
        const auto opts=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat16);
        input=tb::zeros({tokens,hidden},opts);ids=tb::zeros({tokens,routes},opts.dtype(tb::kInt32));
        weights=tb::zeros({tokens,routes},opts.dtype(tb::kFloat32));
        shared_gate=tb::zeros({tokens,1},opts.dtype(shared_float?tb::kFloat32:tb::kFloat16));
        kinds=tb::ones({experts},opts.dtype(tb::kInt32));cpu_pairs=tb::zeros({tokens,routes,output_width},opts);
        cpu_mask=tb::zeros({tokens,routes,1},opts.dtype(tb::kBool));
        std::vector<MoeActivationGeometry> geometries;
        for(int p=0;p<2;++p)for(const auto& g:bank.projections[p].activation_geometry())
            if(std::find(geometries.begin(),geometries.end(),g)==geometries.end())geometries.push_back(g);
        std::vector<int64_t> pointers;
        for(const auto& g:geometries) {
            auto workspace=bank.projections[0].activation_workspace(input,tokens,g.groups,g.gs,{});
            bank.projections[1].activation_workspaces.insert_or_assign({tokens,g.groups,g.gs,input.get_device(),{}},workspace);
            pointers.insert(pointers.end(),{reinterpret_cast<int64_t>(workspace.qx.data_ptr()),
                reinterpret_cast<int64_t>(workspace.xscale.data_ptr()),g.groups,g.gs});quant_groups+=g.groups;
        }
        quant_descriptors=tb::tensor(pointers).reshape({int64_t(geometries.size()),4}).to(tb::kCUDA);
        std::array<const NintWeight*,3> sw{};if(shared)for(int p=0;p<3;++p)sw[p]=&bank.shared[p];
        const auto table=moe_swiglu_sigmoid_table_cuda();
        fused=std::make_unique<MfeFfnRuntime>(std::array<MixedMoeRuntime*,3>{&bank.projections[0],&bank.projections[1],&bank.projections[2]},
            input,ids,weights,table,sw,shared?shared_gate:tb::Tensor{},output_width,ffn_width);
        if(cpu)fused->cpu_results(kinds,cpu_pairs);update(0);
    }
    void update(int step) {
        std::vector<float> values(input.numel());
        for(std::size_t i=0;i<values.size();++i)values[i]=i%hidden<64?0.0f:std::sin(float(i+step*19)*.071f)*.13f;
        input.copy_(tb::tensor(values).reshape({tokens,hidden}).to(tb::kFloat16).to(tb::kCUDA));
        std::vector<int32_t> selected(tokens*routes),kind(experts,1);std::vector<float> route_weights(tokens*routes),gate(tokens);
        std::vector<uint8_t> mask(tokens*routes);std::vector<float> pairs(cpu_pairs.numel());
        for(int i=0;i<tokens*routes;++i) {
            int expert=step==0?bank.selected_expert(i%routes):(i*5+step*7)%(bank.groups*3);
            if(step==1)expert=511;
            if(step==2 && i%routes==7)expert=-1;
            if(step==2 && i%routes==8)expert=512;
            if(step==2 && i%routes==9)expert=selected[i-3];
            selected[i]=expert;route_weights[i]=.013f*float(1+(i*7+step)%19);
            if(cpu && expert>=0 && expert<experts && (step==4 || expert%2==0))kind[expert]=0;
        }
        for(int i=0;i<tokens*routes;++i)mask[i]=cpu && selected[i]>=0 && selected[i]<experts && kind[selected[i]]==0;
        for(std::size_t i=0;i<pairs.size();++i)pairs[i]=std::sin(float(i+step*11)*.041f)*.033f;
        for(int t=0;t<tokens;++t)gate[t]=.17f+.08f*float((t+step)%7);
        ids.copy_(tb::tensor(selected).reshape({tokens,routes}).to(tb::kCUDA));
        weights.copy_(tb::tensor(route_weights).reshape({tokens,routes}).to(tb::kCUDA));
        shared_gate.copy_(tb::tensor(gate).reshape({tokens,1}).to(shared_gate.scalar_type()).to(tb::kCUDA));
        kinds.copy_(tb::tensor(kind).to(tb::kCUDA));
        cpu_pairs.copy_(tb::tensor(pairs).reshape({tokens,routes,output_width}).to(tb::kFloat16).to(tb::kCUDA));
        cpu_mask.copy_(tb::tensor(mask).reshape({tokens,routes,1}).to(tb::kBool).to(tb::kCUDA));bank.maps(step);
    }
    void original() {
        auto route=build_moe_route_plan(ids,experts);
        moe_quantize_shared_input_cuda(input,quant_descriptors,quant_groups);
        auto g=bank.projections[0].forward(execution.config,execution.kl_mmq,true,false,input,route,true);
        auto u=bank.projections[1].forward(execution.config,execution.kl_mmq,true,false,input,route,true);
        oracle_hidden=moe_swiglu_rounded_cuda(g,u,moe_table());
        if(ffn_width!=intermediate)oracle_hidden.narrow(2,ffn_width,intermediate-ffn_width).zero_();
        auto pairs=bank.projections[2].forward(execution.config,execution.kl_mmq,true,false,oracle_hidden,route);
        if(output_width!=hidden)pairs=pairs.narrow(2,0,output_width).contiguous();
        if(cpu)pairs=tb::where(cpu_mask,cpu_pairs,pairs);
        oracle_output=moe_weighted_reduce_cuda(pairs,weights);
        if(with_shared) {
            auto sg=nint_matmul(execution.profiler,bank.shared[0],input);
            auto su=nint_matmul(execution.profiler,bank.shared[1],input);
            auto sh=(sg*tb::sigmoid(sg))*su;
            auto sd=nint_matmul(execution.profiler,bank.shared[2],sh);
            if(output_width!=hidden)sd=sd.narrow(1,0,output_width).contiguous();
            oracle_output=oracle_output+shared_gate*sd;
        }
    }
    tb::Tensor table;
    const tb::Tensor& moe_table(){if(!table.defined())table=moe_swiglu_sigmoid_table_cuda();return table;}
    void verify(const std::string& label) {
        exact(fused->output(),oracle_output,label+" output");
        const int stride=routes+(with_shared?1:0);
        const auto f=fused->hidden().reshape({tokens,stride,fused->batch().hidden_stride});
        // CPU-selected slots are intentionally not computed by the GPU.
        if(!cpu && !with_shared && ffn_width==intermediate) {
            const auto selected=ids.to(tb::kCPU).contiguous();
            for(int t=0;t<tokens;++t)for(int r=0;r<routes;++r) {
                const int e=selected.data_ptr<int32_t>()[t*routes+r];if(e<0 || e>=experts)continue;
                exact(f.narrow(0,t,1).narrow(1,r,1),oracle_hidden.narrow(0,t,1).narrow(1,r,1),label+" activation");
            }
        }
    }
};
class Graph {
    cudaGraph_t graph_=nullptr;cudaGraphExec_t exec_=nullptr;cudaStream_t stream_=nullptr;
public:
    int kernels=0,stage1_registers=0,stage2_registers=0,stage1_local=0,stage2_local=0;
    int stage2_warps=0,stage2_active_blocks=0;
    Graph(const std::function<void()>& run,cudaStream_t stream,int fused_calls=0,
            const std::function<void()>& release={}):stream_(stream) {
        run();MFQ_CUDA_CHECK(cudaStreamSynchronize(stream));
        auto context=mfq::cuda::default_context(mfq_current_cuda_device());
        if(!fused_calls) {
            if(release)release();
            {mfq::cuda::GraphWarmupScope warm(context,stream);run();}
            MFQ_CUDA_CHECK(cudaStreamSynchronize(stream));if(release)release();context->begin_graph_capture(stream);
        }
        try {
            MFQ_CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));run();
            MFQ_CUDA_CHECK(cudaStreamEndCapture(stream,&graph_));
        }catch(...) {
            if(!fused_calls)context->end_graph_capture(stream);
            cudaGraph_t discarded=nullptr;(void)cudaStreamEndCapture(stream,&discarded);
            if(discarded)cudaGraphDestroy(discarded);throw;
        }
        if(!fused_calls)context->end_graph_capture(stream);
        MFQ_CUDA_CHECK(cudaGraphInstantiate(&exec_,graph_,0));
        std::size_t count=0;MFQ_CUDA_CHECK(cudaGraphGetNodes(graph_,nullptr,&count));std::vector<cudaGraphNode_t> nodes(count);
        MFQ_CUDA_CHECK(cudaGraphGetNodes(graph_,nodes.data(),&count));int prepare=0,first=0,second=0;
        for(auto node:nodes) {
            cudaGraphNodeType type;MFQ_CUDA_CHECK(cudaGraphNodeGetType(node,&type));if(type!=cudaGraphNodeTypeKernel)continue;
            ++kernels;cudaKernelNodeParams params{};MFQ_CUDA_CHECK(cudaGraphKernelNodeGetParams(node,&params));
            const char* name=nullptr;MFQ_CUDA_CHECK(cudaFuncGetName(&name,params.func));if(!name)continue;
            cudaFuncAttributes attrs{};MFQ_CUDA_CHECK(cudaFuncGetAttributes(&attrs,params.func));
            if(std::strstr(name,"mfe_ffn_prepare_kernel"))++prepare;
            if(std::strstr(name,"mfe_ffn_gate_up_kernel")){++first;stage1_registers=attrs.numRegs;stage1_local=int(attrs.localSizeBytes);}
            if(std::strstr(name,"mfe_ffn_down_reduce_kernel")) {
                ++second;stage2_registers=attrs.numRegs;stage2_local=int(attrs.localSizeBytes);
                stage2_warps=int(params.blockDim.y);
                MFQ_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &stage2_active_blocks,params.func,int(params.blockDim.x*params.blockDim.y),0));
            }
        }
        if(fused_calls && (kernels!=3*fused_calls || prepare!=fused_calls || first!=fused_calls || second!=fused_calls))
            throw std::runtime_error("two-stage MFE graph contains unexpected kernels");
        MFQ_CUDA_CHECK(cudaGraphUpload(exec_,stream));launch();MFQ_CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    ~Graph(){if(exec_)cudaGraphExecDestroy(exec_);if(graph_)cudaGraphDestroy(graph_);}
    void launch(){MFQ_CUDA_CHECK(cudaGraphLaunch(exec_,stream_));}
    double measure(std::size_t repeats) {
        cudaEvent_t begin=nullptr,end=nullptr;MFQ_CUDA_CHECK(cudaEventCreate(&begin));MFQ_CUDA_CHECK(cudaEventCreate(&end));
        MFQ_CUDA_CHECK(cudaEventRecord(begin,stream_));for(std::size_t i=0;i<repeats;++i)launch();
        MFQ_CUDA_CHECK(cudaEventRecord(end,stream_));MFQ_CUDA_CHECK(cudaEventSynchronize(end));
        float ms=0;MFQ_CUDA_CHECK(cudaEventElapsedTime(&ms,begin,end));cudaEventDestroy(begin);cudaEventDestroy(end);return ms;
    }
};
int check(Bank& bank,const std::string& layout) {
    int cases=0;const auto stream=mfq_current_cuda_stream();
    for(int tokens:{1,3,6})for(int shared_mode:{0,1,2})for(bool cpu:{false,true})for(bool cropped:{false,true}) {
        Call call(bank,tokens,shared_mode!=0,cpu,cropped,shared_mode!=2);call.original();
        Graph graph([&]{call.fused->run();},stream,1);
        for(int step=0;step<6;++step) {
            call.update(step);call.original();call.fused->run();
            const auto label=layout+" M"+std::to_string(tokens)+" shared"+std::to_string(shared_mode)+
                " cpu"+std::to_string(cpu)+" crop"+std::to_string(cropped)+" step"+std::to_string(step);
            call.verify(label+" ordinary");++cases;
            graph.launch();call.verify(label+" graph");++cases;
        }
    }
    return cases;
}
int check_overlap(Bank& bank,const std::string& layout) {
    const auto stream=mfq_current_cuda_stream();int cases=0;
    struct Control {
        uint32_t* flags=nullptr;int32_t* kinds=nullptr;int32_t* indices=nullptr;
        Control() {
            MFQ_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&flags),4*sizeof(uint32_t),cudaHostAllocMapped));
            MFQ_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&kinds),experts*sizeof(int32_t),cudaHostAllocMapped));
            MFQ_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&indices),experts*sizeof(int32_t),cudaHostAllocMapped));
        }
        ~Control(){cudaFreeHost(indices);cudaFreeHost(kinds);cudaFreeHost(flags);}
    } control;
    for(int tokens:{1,3,6})for(int shared_mode:{0,1,2})for(bool graph:{false,true}) {
        bank.maps(0);
        Call call(bank,tokens,shared_mode!=0,true,false,shared_mode!=2);
        control.flags[0]=1;control.flags[1]=0;control.flags[2]=1;control.flags[3]=0;
        call.fused->asynchronous(control.kinds,control.flags,control.flags+1,control.flags+2,
            control.flags+3,control.indices,call.cpu_pairs.data_ptr());
        std::unique_ptr<Graph> resident_graph;
        for(int step:{0,2}) {
            bank.maps(step);call.update(step);call.original();
            auto kinds=call.kinds.cpu().contiguous();
            std::memcpy(control.kinds,kinds.data_ptr(),experts*sizeof(int32_t));
            std::fill(control.indices,control.indices+experts,-1);
            std::vector<mfq::cuda::MfePackedProjection> views;
            for(int expert=0;expert<bank.groups*3;++expert)if(control.kinds[expert]>0 && expert%4==3) {
                control.kinds[expert]=2;control.indices[expert]=int(views.size()/3);
                for(int p=0;p<3;++p) {
                    auto& pool=bank.projections[p].pools[expert/3];std::vector<int64_t> bytes;
                    const auto tensors=fields(pool);
                    const int count=pool.family==MixedMoeFamily::Nint?7:4;
                    for(int i=0;i<count;++i)bytes.push_back(tensors[i]->numel()*tensors[i]->element_size()/3);
                    views.push_back(call.fused->expert_view(p,expert,bytes,expert%3));
                }
            }
            if(call.fused->batch().resident_plan_overlap) {
                for(int p=0;p<3;++p)for(int group=0;group<bank.groups;++group) {
                    std::vector<int32_t> local(experts,-1);
                    for(int e=0;e<3;++e)if(control.kinds[group*3+e]==1 || p!=(group*3+e)%3)
                        local[group*3+e]=e;
                    bank.projections[p].pools[group].expert_local.copy_(tb::tensor(local).to(tb::kCUDA));
                }
            }
            if(views.empty())throw std::runtime_error("overlap fixture has no transferred experts");
            auto storage=call.fused->transfer_buffer();
            MFQ_CUDA_CHECK(cudaMemsetAsync(storage.data_ptr(),0,storage.numel(),stream));
            auto hidden_view=call.fused->hidden();hidden_view.fill_(17);
            control.flags[1]=0;control.flags[2]=0;control.flags[3]=0;
            const auto resident=[&]{call.fused->prepare();call.fused->resident();};
            if(graph) {
                if(!resident_graph)resident_graph=std::make_unique<Graph>(resident,stream,1);
                resident_graph->launch();
            }else resident();
            cudaEvent_t done=nullptr;MFQ_CUDA_CHECK(cudaEventCreateWithFlags(&done,cudaEventDisableTiming));
            MFQ_CUDA_CHECK(cudaEventRecord(done,stream));
            const auto start=std::chrono::steady_clock::now();
            auto status=cudaEventQuery(done);
            while(status==cudaErrorNotReady && std::chrono::steady_clock::now()-start<std::chrono::seconds(2)) {
                std::this_thread::yield();status=cudaEventQuery(done);
            }
            if(status!=cudaSuccess) {
                control.flags[3]=1;control.flags[1]=1;control.flags[2]=1;
                (void)cudaStreamSynchronize(stream);cudaEventDestroy(done);
                throw std::runtime_error("resident FFN blocked on unreleased transfer or CPU result");
            }
            cudaEventDestroy(done);
            const auto hidden_cpu=call.fused->hidden().cpu().contiguous();
            const auto selected=call.ids.cpu().contiguous();
            const int stride=routes+(shared_mode!=0);
            for(int token=0;token<tokens;++token)for(int route=0;route<routes;++route) {
                const int expert=selected.data_ptr<int32_t>()[token*routes+route];
                if(expert<0 || expert>=experts || control.kinds[expert]!=2)continue;
                const auto* values=hidden_cpu.data_ptr<mfq_half>()+(token*stride+route)*call.fused->batch().hidden_stride;
                for(int column=0;column<intermediate;++column)
                    if(float(values[column])!=17)throw std::runtime_error("cold expert computed before transfer readiness");
            }
            MFQ_CUDA_CHECK(cudaMemcpyAsync(storage.data_ptr(),views.data(),views.size()*sizeof(views[0]),cudaMemcpyHostToDevice,stream));
            MFQ_CUDA_CHECK(cudaStreamSynchronize(stream));
            std::atomic_thread_fence(std::memory_order_seq_cst);
            control.flags[1]=1;control.flags[2]=1;
            call.fused->wait_transfer();call.fused->transferred();call.fused->down_reduce();
            call.verify("asynchronous overlap "+layout+" step"+std::to_string(step));++cases;
        }
    }
    bank.maps(0);
    return cases;
}
void benchmark(std::vector<std::unique_ptr<Bank>>& banks,const std::string& layout,int l2) {
    const auto stream=mfq_current_cuda_stream();auto context=mfq::cuda::default_context(mfq_current_cuda_device());
    for(bool shared:{false,true})for(bool rotating:{false,true}) {
        const auto count=rotating?banks.size():std::size_t(1);std::vector<std::unique_ptr<Call>> calls;
        for(std::size_t i=0;i<count;++i)calls.push_back(std::make_unique<Call>(*banks[i],1,shared,false));
        context->begin_graph_pool(stream);
        {
            Graph original([&]{for(auto& call:calls)call->original();},stream,0,
                [&]{for(auto& call:calls){call->oracle_hidden={};call->oracle_output={};}});
            Graph current([&]{for(auto& call:calls)call->fused->run();},stream,int(count));
            for(auto& call:calls)call->verify("pre-timing "+layout);
            std::size_t repeats=1;
            while(std::min(original.measure(repeats),current.measure(repeats))<20) {
                if(repeats>std::numeric_limits<std::size_t>::max()/2)throw std::runtime_error("MFE benchmark repeat overflow");
                repeats*=2;
            }
            std::vector<double> before,after;
            for(int sample=0;sample<7;++sample) {
                double a,b;if(sample%2){b=current.measure(repeats);a=original.measure(repeats);}
                else {a=original.measure(repeats);b=current.measure(repeats);}
                before.push_back(a*1000/(repeats*count));after.push_back(b*1000/(repeats*count));
            }
            for(auto& call:calls)call->verify("post-timing "+layout);
            std::cout<<std::setprecision(9)<<"{\"operator\":\"mfe_ffn_two_stage\",\"layout\":\""<<layout
                <<"\",\"M\":1,\"routes\":10,\"hidden\":2560,\"intermediate\":640,\"shared\":"<<(shared?"true":"false")
                <<",\"cpu\":false,\"cache\":\""<<(rotating?"rotating":"fixed")<<"\",\"working_sets\":"<<count
                <<",\"selected_weight_bytes\":"<<banks[0]->selected_weight_bytes*count<<",\"device_l2_bytes\":"<<l2
                <<",\"off_kernel_nodes\":"<<original.kernels<<",\"on_kernel_nodes\":"<<current.kernels
                <<",\"stage1_registers\":"<<current.stage1_registers<<",\"stage2_registers\":"<<current.stage2_registers
                <<",\"stage1_local_bytes\":"<<current.stage1_local<<",\"stage2_local_bytes\":"<<current.stage2_local
                <<",\"stage2_warps\":"<<current.stage2_warps<<",\"stage2_active_blocks\":"<<current.stage2_active_blocks
                <<",\"samples\":7,\"repeats\":"<<repeats<<",\"exact_original_bits\":true";
            mfq::bench::timing(before,after);std::cout<<"}\n"<<std::flush;
        }
        context->end_graph_pool(stream);
    }
}
}
int main(int argc,char** argv)try {
    if(argc<2 || argc>3)throw std::invalid_argument("usage: mfq-cuda-mfe-ffn-bench fixture-dir [--check-only|--bench-only]");
    const bool check_only=argc==3 && std::string(argv[2])=="--check-only";
    const bool bench_only=argc==3 && std::string(argv[2])=="--bench-only";
    if(argc==3 && !check_only && !bench_only)throw std::invalid_argument("invalid MFE benchmark mode");
    auto context=mfq::cuda::default_context(mfq_current_cuda_device());cudaDeviceProp properties{};
    MFQ_CUDA_CHECK(cudaGetDeviceProperties(&properties,0));int cases=0,overlap_cases=0;
    for(const std::string layout:{"nint5","nvq3j-l","mixed12"}) {
        std::vector<std::unique_ptr<Bank>> banks;banks.push_back(std::make_unique<Bank>(argv[1],layout));
        if(!bench_only){cases+=check(*banks[0],layout);overlap_cases+=check_overlap(*banks[0],layout);
            std::cout<<"CHECK "<<layout<<" cases="<<cases<<" overlap_cases="<<overlap_cases<<" PASS\n"<<std::flush;}
        if(!check_only) {
            const auto selected=banks[0]->selected_weight_bytes;
            while(banks.size()*selected<std::size_t(properties.l2CacheSize)*4)banks.push_back(std::make_unique<Bank>(*banks[0]));
            benchmark(banks,layout,properties.l2CacheSize);
        }
    }
    if(!bench_only)std::cout<<"MFE original-bit cases="<<cases<<" PASS\n";
    if(!bench_only)std::cout<<"MFE asynchronous-overlap cases="<<overlap_cases<<" PASS\n";
    return 0;
}catch(const std::exception& error){std::cerr<<"FAIL: "<<error.what()<<'\n';return 1;}
