#include "packed_bench_utils.h"
#include "mfq_cuda_moe_ops.h"
#include "mfe_ffn_runtime.h"
#include "storage/moe_quant_range_source.h"
#include "storage/moe_cache_types_internal.h"
#include "storage/weight_loader.h"
#include "mfq/model_source.h"
#include <array>
#include <cstring>
#include <memory>
#include <set>
#include <fstream>
#include <filesystem>

namespace {
namespace tb=mfq_tensor_backend;
constexpr int hidden=2560,intermediate=640,routes=10,experts=512;
const std::array<const char*,3> roles={"gate","up","down"};

void export_shared_fixtures(const char* filename,const std::filesystem::path& root) {
    const auto model=mfq::open_model_source(filename);
    std::filesystem::create_directories(root);
    for(const auto* role:roles) {
        const auto name=std::string("model.block.0.mlp.shared_expert.")+role+".weight";
        const auto* metadata=model->find_tensor(name);
        if(!metadata || metadata->dtype!="NINT")throw std::runtime_error("shared fixture is not canonical NINT: "+name);
        const auto blob=model->read(name);
        const auto target=root/(std::string("shared-s2-")+role+".nint");
        std::ofstream output(target,std::ios::binary);
        if(!output || !output.write(reinterpret_cast<const char*>(blob.data()),std::streamsize(blob.size())))
            throw std::runtime_error("cannot write shared fixture: "+target.string());
        std::cout<<"shared_fixture tensor="<<name<<" bytes="<<blob.size()<<" path="<<target.string()<<'\n';
    }
}

void cache_topology(const char* filename) {
    const auto model=mfq::open_model_source(filename);
    std::size_t read_calls=0,read_bytes=0;
    std::cout<<"{\"sources\":[";
    bool first=true;
    for(int layer=0;;++layer) {
        const auto prefix="model.block."+std::to_string(layer)+".mlp.experts.";
        if(!model->find_tensor(prefix+roles[0]+".weight"))break;
        for(int p=0;p<3;++p) {
            const auto name=prefix+roles[p]+".weight";
            const auto* tensor=model->find_tensor(name);
            if(!tensor)throw std::runtime_error("missing topology projection "+name);
            const auto reader=model->tensor_reader(name);
            auto store=std::make_shared<mfq::MfeQuantExpertStore>(std::size_t(tensor->nbytes),
                [&,reader](std::size_t offset,uint8_t* dst,std::size_t size) {
                    ++read_calls;read_bytes+=size;
                    reader(offset,reinterpret_cast<std::byte*>(dst),size);
                });
            MoeQuantRangeSource source(store);
            if(!first)std::cout<<',';first=false;
            std::cout<<"{\"layer\":"<<layer<<",\"projection\":"<<p
                <<",\"experts\":"<<store->num_experts()<<",\"pools\":[";
            for(std::size_t i=0;i<store->pool_count();++i) {
                auto prototype=source.metadata()->pools[i];prototype.local_experts=1;
                const auto layouts=moe_cache_field_layouts(prototype);
                const auto signature=moe_cache_signature(prototype,store->out_per_expert(),store->neuron_len(),layouts);
                if(i)std::cout<<',';
                std::cout<<"{\"signature\":\""<<signature<<"\",\"field_bytes\":[";
                for(std::size_t f=0;f<layouts.size();++f) {
                    if(f)std::cout<<',';
                    std::cout<<layouts[f].elements*layouts[f].element_size;
                }
                std::cout<<"],\"ids\":[";
                const auto& ids=store->pool_expert_ids(i);
                for(std::size_t e=0;e<ids.size();++e){if(e)std::cout<<',';std::cout<<ids[e];}
                std::cout<<"]}";
            }
            std::cout<<"]}";
        }
    }
    std::cout<<"],\"read_calls\":"<<read_calls<<",\"read_bytes\":"<<read_bytes
        <<",\"full_model_loaded\":false}\n";
}

void exact(const tb::Tensor& actual,const tb::Tensor& expected,const std::string& label) {
    auto a=actual.cpu().contiguous(),b=expected.cpu().contiguous();
    if(a.sizes()!=b.sizes() || a.scalar_type()!=b.scalar_type())
        throw std::runtime_error(label+" shape/dtype mismatch");
    const auto bytes=std::size_t(a.numel()*a.element_size());
    if(!std::memcmp(a.data_ptr(),b.data_ptr(),bytes))return;
    const auto* x=static_cast<const uint8_t*>(a.data_ptr());
    const auto* y=static_cast<const uint8_t*>(b.data_ptr());
    std::size_t i=0;while(i<bytes && x[i]==y[i])++i;
    const auto index=i/a.element_size();auto af=a.to(tb::kFloat32),bf=b.to(tb::kFloat32);
    std::ostringstream detail;detail.precision(9);
    detail<<label<<" differs at "<<index<<": "<<af.data_ptr<float>()[index]
        <<" versus "<<bf.data_ptr<float>()[index];
    throw std::runtime_error(detail.str());
}
void finite(const tb::Tensor& value,const std::string& label) {
    const auto host=value.cpu().to(tb::kFloat32).contiguous();
    for(int64_t i=0;i<host.numel();++i)if(!std::isfinite(host.data_ptr<float>()[i]))
        throw std::runtime_error(label+" has nonfinite element "+std::to_string(i));
}
void upload(MixedMoePool& pool) {
    std::vector<tb::Tensor*> fields;
    if(pool.family==MixedMoeFamily::Nint)
        fields={&pool.nint.q_packed,&pool.nint.row_q_bits,&pool.nint.row_q_bit_offsets,
            &pool.nint.sub_scale,&pool.nint.sub_min,&pool.nint.neuron_scale,&pool.nint.neuron_min};
    else fields={&pool.nvq.indices_packed,&pool.nvq.aux_packed,&pool.nvq.sub_scale_packed,
            &pool.nvq.neuron_scale,&pool.nvq.codebook};
    for(auto* field:fields)*field=field->to(tb::kCUDA).contiguous();
}
struct Bank {
    std::shared_ptr<const mfq::ModelSource> model;
    std::array<std::shared_ptr<MoeQuantRangeSource>,3> sources;
    std::array<MixedMoeRuntime,3> projections;
    std::array<QuantLinear,3> shared;
    QuantLinear shared_router;
    std::vector<int> selected;
    std::array<std::size_t,3> read_calls{},read_bytes{};
    Bank(const char* filename,int layer) {
        model=mfq::open_model_source(filename);CudaExecutionContext execution;
        const auto prefix="model.block."+std::to_string(layer)+".mlp.";
        std::set<int> ids;
        for(int p=0;p<3;++p) {
            const auto name=prefix+"experts."+roles[p]+".weight";
            const auto* tensor=model->find_tensor(name);
            if(!tensor)throw std::runtime_error("missing released tensor "+name);
            auto reader=model->tensor_reader(name);
            auto store=std::make_shared<mfq::MfeQuantExpertStore>(std::size_t(tensor->nbytes),
                [this,p,reader](std::size_t offset,uint8_t* dst,std::size_t size) {
                    ++read_calls[p];read_bytes[p]+=size;
                    reader(offset,reinterpret_cast<std::byte*>(dst),size);
                });
            sources[p]=std::make_shared<MoeQuantRangeSource>(store);
            for(std::size_t cohort=0;cohort<store->pool_count();++cohort) {
                const auto& members=store->pool_expert_ids(cohort);
                ids.insert(members.front());ids.insert(members.back());
                std::cout<<"COHORT projection="<<roles[p]<<" index="<<cohort
                    <<" dtype="<<store->expert_dtype(members.front())<<" experts="<<members.size()<<'\n';
            }
        }
        selected.assign(ids.begin(),ids.end());
        for(int p=0;p<3;++p) {
            auto& runtime=projections[p];runtime.n_experts=experts;
            runtime.out_per_expert=p==2?hidden:intermediate;
            runtime.neuron_len=p==2?intermediate:hidden;runtime.partial_experts=true;
            for(int expert:selected) {
                auto pool=sources[p]->read_expert(expert);upload(pool);
                std::vector<int32_t> local(experts,-1);local[expert]=0;
                pool.expert_local=tb::tensor(local).to(tb::kCUDA);
                runtime.pools.push_back(std::move(pool));
            }
            initialize_mixed_nvq_dispatch(runtime,execution.config);
            shared[p]=load_quant_linear(execution,*model,prefix+"shared_expert."+roles[p]+".weight");
            if(!shared[p].is_nint())throw std::runtime_error("released shared projection is not NINT");
            std::cout<<"SOURCE projection="<<roles[p]<<" calls="<<read_calls[p]<<" bytes="<<read_bytes[p]
                <<" shared_bits="<<shared[p].nint.bits<<" shared_gs="<<shared[p].nint.gs<<'\n';
        }
        shared_router=load_quant_linear(execution,*model,prefix+"shared_expert.router.weight");
        std::cout<<"RELEASED layer="<<layer<<" selected_experts="<<selected.size()
            <<" full_model_loaded=0 synthetic_inputs=1\n"<<std::flush;
    }
};
struct Call {
    Bank& bank;int tokens;bool with_shared;
    CudaExecutionContext execution;
    tb::Tensor input,ids,weights,shared_gate,table;
    tb::Tensor expected_hidden,expected_shared_hidden,expected_output;
    std::unique_ptr<MfeFfnRuntime> fused;
    Call(Bank& b,int m,bool shared):bank(b),tokens(m),with_shared(shared) {
        const auto opts=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat16);
        input=tb::zeros({tokens,hidden},opts);ids=tb::zeros({tokens,routes},opts.dtype(tb::kInt32));
        weights=tb::zeros({tokens,routes},opts.dtype(tb::kFloat32));
        table=moe_swiglu_sigmoid_table_cuda();
        std::array<const NintWeight*,3> sw{};
        if(shared) {
            shared_gate=tb::sigmoid(bank.shared_router.forward(execution,input));
            for(int p=0;p<3;++p)sw[p]=&bank.shared[p].nint;
        }
        fused=std::make_unique<MfeFfnRuntime>(std::array<MixedMoeRuntime*,3>{
            &bank.projections[0],&bank.projections[1],&bank.projections[2]},input,ids,weights,table,sw,shared_gate);
    }
    void update(float amplitude,int step) {
        std::vector<float> values(tokens*hidden),route_weights(tokens*routes);
        std::vector<int32_t> chosen(tokens*routes);
        for(std::size_t i=0;i<values.size();++i)
            values[i]=amplitude*std::sin(float(i+step*31)*.017f);
        for(int i=0;i<tokens*routes;++i) {
            chosen[i]=bank.selected[(i+step*routes)%bank.selected.size()];
            route_weights[i]=float((i*7+step*3)%17+1)/97;
        }
        input.copy_(tb::tensor(values).reshape({tokens,hidden}).to(tb::kFloat16).to(tb::kCUDA));
        ids.copy_(tb::tensor(chosen).reshape({tokens,routes}).to(tb::kCUDA));
        weights.copy_(tb::tensor(route_weights).reshape({tokens,routes}).to(tb::kCUDA));
        if(with_shared)shared_gate.copy_(tb::sigmoid(bank.shared_router.forward(execution,input)));
    }
    void original() {
        auto route=build_moe_route_plan(ids,experts);
        auto gate=bank.projections[0].forward(execution.config,execution.kl_mmq,true,false,input,route);
        auto up=bank.projections[1].forward(execution.config,execution.kl_mmq,true,false,input,route);
        expected_hidden=moe_swiglu_rounded_cuda(gate,up,table);
        auto pairs=bank.projections[2].forward(execution.config,execution.kl_mmq,true,false,expected_hidden,route);
        expected_output=moe_weighted_reduce_cuda(pairs,weights);
        if(with_shared) {
            auto sg=bank.shared[0].forward(execution,input),su=bank.shared[1].forward(execution,input);
            expected_shared_hidden=(sg*tb::sigmoid(sg))*su;
            auto sd=bank.shared[2].forward(execution,expected_shared_hidden);
            expected_output=expected_output+shared_gate*sd;
        }
    }
    void verify(const std::string& label) {
        const auto f=fused->hidden().reshape({tokens,routes+(with_shared?1:0),fused->batch().hidden_stride});
        auto routed=f.narrow(1,0,routes).narrow(2,0,intermediate).contiguous();
        exact(routed,expected_hidden,label+" routed_activation");
        if(with_shared)exact(f.select(1,routes).narrow(1,0,intermediate).contiguous(),
            expected_shared_hidden,label+" shared_activation");
        exact(fused->output(),expected_output,label+" output");
        finite(expected_hidden,label+" routed_activation");finite(expected_output,label+" output");
    }
};
class Graph {
    cudaGraph_t graph_=nullptr;cudaGraphExec_t exec_=nullptr;cudaStream_t stream_;
public:
    explicit Graph(Call& call):stream_(mfq_current_cuda_stream()) {
        call.fused->run();MFQ_CUDA_CHECK(cudaStreamSynchronize(stream_));
        MFQ_CUDA_CHECK(cudaStreamBeginCapture(stream_,cudaStreamCaptureModeThreadLocal));
        call.fused->run();MFQ_CUDA_CHECK(cudaStreamEndCapture(stream_,&graph_));
        MFQ_CUDA_CHECK(cudaGraphInstantiate(&exec_,graph_,0));
        std::size_t count=0;MFQ_CUDA_CHECK(cudaGraphGetNodes(graph_,nullptr,&count));
        if(count!=3)throw std::runtime_error("released two-stage graph requires three kernel nodes");
    }
    ~Graph(){if(exec_)cudaGraphExecDestroy(exec_);if(graph_)cudaGraphDestroy(graph_);}
    void run(){MFQ_CUDA_CHECK(cudaGraphLaunch(exec_,stream_));}
};
}
int main(int argc,char** argv)try {
    if(argc==4 && std::string(argv[2])=="--export-shared-fixtures") {
        export_shared_fixtures(argv[1],argv[3]);return 0;
    }
    if(argc<2 || argc>3)throw std::runtime_error("usage: mfq-cuda-mfe-released-test model-shard [layer|--cache-topology|--export-shared-fixtures DIR]");
    if(argc==3 && std::string(argv[2])=="--cache-topology"){cache_topology(argv[1]);return 0;}
    auto context=mfq::cuda::default_context(mfq_current_cuda_device());
    const int layer=argc==3?std::stoi(argv[2]):0;Bank bank(argv[1],layer);
    const int steps=2*int((bank.selected.size()+routes-1)/routes);int cases=0;
    for(int tokens:{1,3,6})for(bool shared:{false,true}) {
        Call call(bank,tokens,shared);Graph graph(call);
        for(float amplitude:{.13f,1.0f,4.0f,8.0f})for(int step=0;step<steps;++step) {
            call.update(amplitude,step);call.original();
            const auto label="M="+std::to_string(tokens)+" shared="+std::to_string(shared)+
                " amplitude="+std::to_string(amplitude)+" step="+std::to_string(step);
            call.fused->run();call.verify(label+" ordinary");++cases;
            graph.run();call.verify(label+" graph");++cases;
        }
        std::cout<<"CHECK M="<<tokens<<" shared="<<shared<<" cases="<<cases<<" PASS\n"<<std::flush;
    }
    std::cout<<"MFE released-weight exact cases="<<cases<<" PASS\n";return 0;
}catch(const std::exception& error){std::cerr<<"FAIL: "<<error.what()<<'\n';return 1;}
