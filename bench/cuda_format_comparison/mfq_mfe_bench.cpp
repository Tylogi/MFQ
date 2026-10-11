#include "weights.h"
#include "mfe_ffn_runtime.h"
#include "mfq_cuda_moe_ops.h"
#include "quant_linear.h"
#include "storage/weight_loader.h"
#include "e8_dense_group_pack.h"
#include <array>
#include <functional>

constexpr int hidden=2560,intermediate=640,routes=10,experts=512;
static const std::array<const char*,3> roles{"gate","up","down"};
static void pack_routes(bool enabled) {
#ifdef _WIN32
    _putenv_s("MFQ_MFE_DOWN_PACK_ROUTES",enabled?"1":"0");
#else
    setenv("MFQ_MFE_DOWN_PACK_ROUTES",enabled?"1":"0",1);
#endif
}
static void specialize_formats(bool enabled) {
#ifdef _WIN32
    _putenv_s("MFQ_MFE_FORMAT_SPECIALIZE",enabled?"1":"0");
#else
    setenv("MFQ_MFE_FORMAT_SPECIALIZE",enabled?"1":"0",1);
#endif
}
static void multi_sums(bool enabled) {
#ifdef _WIN32
    _putenv_s("MFQ_CUDA_WARP_MULTI_SUM",enabled?"1":"0");
#else
    setenv("MFQ_CUDA_WARP_MULTI_SUM",enabled?"1":"0",1);
#endif
}

struct Layer {
    int index;
    std::array<MixedMoeRuntime,3> projections;
    std::array<QuantLinear,3> shared;
    QuantLinear shared_router;
    std::array<size_t,3> bytes{},fused_bytes{};
    size_t shared_bytes=0;
    std::array<std::size_t,3> shared_projection_bytes{};
    std::vector<int32_t> selected;
    std::map<const void*,tb::Tensor> dense_e8_streams;
    bool enlarge_e8_arena() {
        for(auto& projection:projections)for(auto& pool:projection.pools) {
            if(pool.family!=MixedMoeFamily::Nvq)continue;
            auto& w=pool.nvq;
            if(w.kernel_format!=5 && w.kernel_format!=13 && w.kernel_format!=14)continue;
            const auto old_rows=w.out;
            const int bits=index_bits(int(w.kernel_format));
            const auto expert_bits=old_rows*((w.neuron_len+7)/8)*bits;
            const int slot=int(INT32_MAX/expert_bits)+1,count=slot+1;
            for(auto* field:{&w.indices_packed,&w.aux_packed,&w.sub_scale_packed,&w.neuron_scale}) {
                const auto elements=field->numel();
                auto arena=tb::empty({elements*count},field->options());
                arena.narrow(0,elements*slot,elements).copy_(*field);
                *field=std::move(arena);
            }
            auto map=pool.expert_local.to(tb::kCPU);
            for(int64_t i=0;i<map.numel();++i)if(map.data_ptr<int32_t>()[i]>=0)map.data_ptr<int32_t>()[i]=slot;
            pool.expert_local=map.to(tb::kCUDA);pool.local_experts=count;
            w.out=old_rows*count;w.shape={w.out,w.neuron_len};
            std::cout<<"E8_LARGE_ARENA layer="<<index<<" format="<<w.kernel_format<<" experts="<<count
                <<" selected_slot="<<slot<<" selected_index_bit="<<expert_bits*slot<<'\n';
            return true;
        }
        return false;
    }
    int enable_dense_nvq(MfeFfnRuntime& runtime,bool e8,bool d4) {
        if(const auto* grouped=std::getenv("MFQ_MFE_GROUP_DOT");grouped && grouped[0]=='0')
            throw std::runtime_error("dense NVQ probe requires grouped MFE arithmetic");
        const auto selected=[&](int format) {
            return (e8 && (format==5 || format==13 || format==14)) || (d4 && dense_d4_format(format));
        };
        for(auto& projection:projections)for(auto& pool:projection.pools) {
            if(pool.family!=MixedMoeFamily::Nvq)continue;
            const auto& w=pool.nvq;
            if(!selected(int(w.kernel_format)) || dense_e8_streams.count(w.indices_packed.data_ptr()))continue;
            auto host=w;host.indices_packed=w.indices_packed.to(tb::kCPU);
            host.aux_packed=w.aux_packed.to(tb::kCPU);host.sub_scale_packed=w.sub_scale_packed.to(tb::kCPU);
            auto packed=pack_dense(host);
            if(packed.size()!=std::size_t(w.indices_packed.numel()+w.aux_packed.numel()+w.sub_scale_packed.numel()))
                throw std::runtime_error("real MFE dense NVQ changed payload size");
            dense_e8_streams.emplace(w.indices_packed.data_ptr(),tb::tensor(packed).to(tb::kCUDA));
        }
        const auto& batch=runtime.batch();
        const int count=batch.projection_counts[0]+batch.projection_counts[1]+batch.projection_counts[2];
        std::vector<mfq::cuda::MfePackedProjection> views(count);
        fb::check(cudaMemcpyAsync(views.data(),batch.projections,views.size()*sizeof(views[0]),
            cudaMemcpyDeviceToHost,mfq_current_cuda_stream()));
        fb::check(cudaStreamSynchronize(mfq_current_cuda_stream()));
        int converted=0;
        for(auto& view:views)if(view.family==2 && selected(view.format)) {
            const auto& packed=dense_e8_streams.at(view.fields[0]);
            view.fields[0]=packed.data_ptr();view.sizes[0]=packed.numel();
            view.fields[1]=view.fields[2]=nullptr;view.sizes[1]=view.sizes[2]=0;
            if(dense_d4_format(view.format))view.d4_dense_groups=true;
            else view.e8_dense_groups=true;
            ++converted;
        }
        // Some real layers contain only D4/NVQ1/NINT. Keep those as exact
        // compatibility controls rather than claiming E8 execution coverage.
        // Benchmark owns both representations for exact A/B comparison. Only
        // the selected descriptor's bytes are read by the timed MFE kernel.
        fb::check(cudaMemcpyAsync(const_cast<mfq::cuda::MfePackedProjection*>(batch.projections),views.data(),
            views.size()*sizeof(views[0]),cudaMemcpyHostToDevice,mfq_current_cuda_stream()));
        fb::check(cudaStreamSynchronize(mfq_current_cuda_stream()));
        return converted;
    }
    Layer(const mfq::ModelSource& model,int layer):index(layer) {
        CudaExecutionContext execution;
        for(int route=0;route<routes;++route)selected.push_back((route*53+layer*17)%experts);
        const auto prefix="model.block."+std::to_string(layer)+".mlp.";
        for(int p=0;p<3;++p) {
            const auto name=prefix+"experts."+roles[p]+".weight";
            const auto* record=model.find_tensor(name);
            if(!record)throw std::runtime_error("missing MFE tensor: "+name);
            auto reader=model.tensor_reader(name);
            auto store=std::make_shared<mfq::MfeQuantExpertStore>(size_t(record->nbytes),
                [reader](size_t offset,uint8_t* data,size_t size){reader(offset,reinterpret_cast<std::byte*>(data),size);});
            MoeQuantRangeSource source(store);
            auto& runtime=projections[p];runtime.n_experts=experts;
            runtime.out_per_expert=p==2 ? hidden : intermediate;
            runtime.neuron_len=p==2 ? intermediate : hidden;runtime.partial_experts=true;
            std::map<const void*,tb::Tensor> codebooks;
            std::vector<MixedMoePool> selected_weights;
            for(int expert:selected)selected_weights.push_back(source.read_expert(expert));
            prepare_bench_execution_layout_batch(selected_weights);
            for(size_t route=0;route<selected.size();++route) {
                const int expert=selected[route];auto& cpu=selected_weights[route];bytes[p]+=storage(cpu);
                auto& book=codebooks[cpu.nvq.codebook.defined() ? cpu.nvq.codebook.data_ptr() : nullptr];
                auto pool=upload(cpu,book,false);
                if(pool.family==MixedMoeFamily::Nvq)book=pool.nvq.codebook;
                std::vector<int32_t> local(experts,-1);local[expert]=0;
                pool.expert_local=tb::tensor(local).to(tb::kCUDA);
                runtime.pools.push_back(std::move(pool));
            }
            initialize_mixed_nvq_dispatch(runtime,execution.config);
            bytes[p]=0;
            for(auto& pool:runtime.pools)bytes[p]+=storage(pool);
            fused_bytes[p]=bytes[p];
            shared[p]=load_quant_linear(execution,model,prefix+"shared_expert."+roles[p]+".weight");
            if(!shared[p].is_nint())throw std::runtime_error("shared expert format is not NINT");
            auto& w=shared[p].nint;
            for(auto* field:{&w.q_packed,&w.row_q_bits,&w.row_q_bit_offsets,&w.sub_scale,&w.sub_min,&w.neuron_scale,&w.neuron_min})
                shared_projection_bytes[p]+=field->numel()*field->element_size();
            shared_bytes+=shared_projection_bytes[p];
        }
        shared_router=load_quant_linear(execution,model,prefix+"shared_expert.router.weight");
        std::cout<<"LAYER "<<layer<<" expert_ids=";
        for(auto id:selected)std::cout<<id<<',';
        std::cout<<" gate_bytes="<<bytes[0]<<" up_bytes="<<bytes[1]<<" down_bytes="<<bytes[2]
            <<" shared_bytes="<<shared_bytes<<'\n';
        std::cout<<"FORMATS "<<layer;
        for(int p=0;p<3;++p) {
            uint32_t mask=0;
            for(const auto& pool:projections[p].pools)if(pool.family==MixedMoeFamily::Nvq)
                mask|=1u<<pool.nvq.kernel_format;
            std::cout<<' '<<roles[p]<<'='<<mask;
        }
        std::cout<<'\n';
    }
};
struct Call {
    Layer& layer;
    int tokens;
    CudaExecutionContext execution;
    tb::Tensor x,xd,ids,weights,table,shared_gate;
    MoeRoutePlan route;
    std::unique_ptr<MfeFfnRuntime> fused;
    int dense_views=0;
    Call(Layer& bank,bool shared,int batch=1):layer(bank),tokens(batch) {
        x=tb::tensor(fb::input(tokens,hidden)).reshape({tokens,hidden}).to(tb::kCUDA,tb::kFloat16);
        xd=tb::tensor(fb::input(tokens*routes,intermediate)).reshape({tokens,routes,intermediate}).to(tb::kCUDA,tb::kFloat16);
        std::vector<int32_t> selected;
        for(int i=0;i<tokens;++i)selected.insert(selected.end(),layer.selected.begin(),layer.selected.end());
        ids=tb::tensor(selected).reshape({tokens,routes}).to(tb::kCUDA);
        weights=tb::full({tokens,routes},1.f/routes,x.options().dtype(tb::kFloat32));
        route=build_moe_route_plan(ids,experts);table=moe_swiglu_sigmoid_table_cuda();
        std::array<const NintWeight*,3> sw{};
        if(shared) {
            shared_gate=tb::sigmoid(layer.shared_router.forward(execution,x));
            for(int p=0;p<3;++p)sw[p]=&layer.shared[p].nint;
        }
        fused=std::make_unique<MfeFfnRuntime>(std::array<MixedMoeRuntime*,3>{&layer.projections[0],&layer.projections[1],&layer.projections[2]},
            x,ids,weights,table,sw,shared_gate);
        const auto enabled=[](const char* name){const auto* value=std::getenv(name);return value && value[0]=='1';};
        const bool all_dense=enabled("MFQ_BENCH_NVQ_DENSE");
        const bool e8_dense=all_dense || enabled("MFQ_BENCH_E8_DENSE");
        const bool d4_dense=all_dense || enabled("MFQ_BENCH_D4_DENSE");
        if(e8_dense || d4_dense)dense_views=layer.enable_dense_nvq(*fused,e8_dense,d4_dense);
        fused->run();
    }
    tb::Tensor projection(int p,const tb::Tensor& input) {
        return layer.projections[p].forward(execution.config,execution.kl_mmq,true,false,input,route);
    }
    double verify(bool shared) {
        auto gate=projection(0,x),up=projection(1,x);
        auto h=moe_swiglu_rounded_cuda(gate,up,table);
        auto expected=moe_weighted_reduce_cuda(projection(2,h),weights);
        if(shared) {
            auto g=layer.shared[0].forward(execution,x),u=layer.shared[1].forward(execution,x);
            expected=expected+shared_gate*layer.shared[2].forward(execution,(g*tb::sigmoid(g))*u);
        }
        fused->run();
        auto a=expected.to(tb::kCPU,tb::kFloat32),b=fused->output().to(tb::kCPU,tb::kFloat32);
        double error=0,norm=0;
        for(int i=0;i<tokens*hidden;++i) {
            double av=a.data_ptr<float>()[i],bv=b.data_ptr<float>()[i];
            if(!std::isfinite(bv))throw std::runtime_error("nonfinite MFE output");
            error+=(av-bv)*(av-bv);norm+=av*av;
        }
        const double relative=std::sqrt(error/std::max(norm,1e-30));
        if(relative>2e-3)throw std::runtime_error("MFE reference error exceeds 0.2%: "+std::to_string(relative));
        return relative;
    }
    tb::Tensor run(int stage) {
        if(stage<3)return projection(stage,stage==2 ? xd : x);
        if(stage==3){fused->prepare();fused->gate_up();return fused->hidden();}
        if(stage==4){fused->down_reduce();return fused->output();}
        fused->run();return fused->output();
    }
};
static void verify_packing(std::vector<std::unique_ptr<Layer>>& layers,bool compare_formats=false) {
    size_t checks=0;double worst=0;
    for(int tokens:{1,3,8})for(bool shared:{false,true})for(auto& layer:layers) {
        pack_routes(true);if(compare_formats)specialize_formats(true);
        Call call(*layer,shared,tokens);MfqCudaGraph graph;
        mfq_prepare_cuda_graph_memory(graph);call.fused->run();
        fb::check(cudaStreamSynchronize(mfq_current_cuda_stream()));
        graph.capture_begin();call.fused->run();graph.capture_end();
        for(int step=0;step<3;++step) {
            std::vector<int32_t> selected(tokens*routes);
            for(int i=0;i<tokens*routes;++i) {
                selected[i]=layer->selected[(i/(step==1?2:1)+step)%routes];
                if(step==2 && i%3==0)selected[i]=-1;
                if(step==2 && i%3==1)selected[i]=experts;
            }
            call.ids.copy_(tb::tensor(selected).reshape({tokens,routes}).to(tb::kCUDA));
            pack_routes(false);if(compare_formats)specialize_formats(false);
            call.fused->run();auto expected=call.fused->output().to(tb::kCPU);
            if(compare_formats)specialize_formats(true);
            pack_routes(true);graph.replay();auto actual=call.fused->output().to(tb::kCPU);
            if(actual.sizes()!=expected.sizes() || std::memcmp(actual.data_ptr(),expected.data_ptr(),actual.numel()*actual.element_size()))
                throw std::runtime_error("MFE route packing/format specialization changed output bits");
            worst=std::max(worst,call.verify(shared));++checks;
        }
    }
    std::cout<<(compare_formats?"VERIFY_MFE_FORMATS checks=":"VERIFY_MFE_PACKING checks=")<<checks<<" max_reference_relative_l2="<<worst
        <<" old/new bitexact, tokens=1/3/8, shared, duplicate/masked routes, graph replay PASS\n";
}
static void set_geometry(const char* name,const char* value);
static void verify_nvq_option(std::vector<std::unique_ptr<Layer>>& layers,const char* option,const char* label) {
    size_t checks=0,hidden_checks=0,eligible=0,compatibility=0;
    const bool check_narrow=std::strcmp(option,"MFQ_MFE_E8_NARROW")==0;
    const bool check_dense=std::strcmp(option,"MFQ_BENCH_E8_DENSE")==0 ||
        std::strcmp(option,"MFQ_BENCH_D4_DENSE")==0 || std::strcmp(option,"MFQ_BENCH_NVQ_DENSE")==0;
    const bool check_math=check_narrow || check_dense || std::strcmp(option,"MFQ_MFE_DEFAULT_MATH")==0;
    for(int tokens:{1,3,8})for(bool shared:{false,true})for(auto& layer:layers) {
        set_geometry(option,"0");Call reference(*layer,shared,tokens);
        set_geometry(option,"1");Call candidate(*layer,shared,tokens);
        if(check_math) {
            if(check_dense?candidate.dense_views>0:
                candidate.fused->batch().default_math && (!check_narrow || candidate.fused->batch().e8_narrow))++eligible;
            else ++compatibility;
        }
        MfqCudaGraph graph;mfq_prepare_cuda_graph_memory(graph);
        candidate.fused->run();fb::check(cudaStreamSynchronize(mfq_current_cuda_stream()));
        graph.capture_begin();candidate.fused->run();graph.capture_end();
        for(int step=0;step<3;++step) {
            std::vector<int32_t> selected(tokens*routes);
            for(int i=0;i<tokens*routes;++i) {
                selected[i]=layer->selected[(i/(step==1?2:1)+step)%routes];
                if(step==2 && i%3==0)selected[i]=-1;
                if(step==2 && i%3==1)selected[i]=experts;
            }
            auto ids=tb::tensor(selected).reshape({tokens,routes}).to(tb::kCUDA);
            reference.ids.copy_(ids);candidate.ids.copy_(ids);
            auto values=fb::input(tokens,hidden);
            for(size_t i=0;i<values.size();++i)values[i]=step==0?0.f:
                values[i]*float(step)+std::sin(float(i)*.019f+step)*.21f;
            auto input=tb::tensor(values).reshape({tokens,hidden}).to(tb::kCUDA,tb::kFloat16);
            reference.x.copy_(input);candidate.x.copy_(input);
            // A launch-time switch must be reset for the reference; the graph
            // keeps the enabled kernel captured above across environment changes.
            set_geometry(option,"0");
            reference.fused->run();auto expected=reference.fused->output().to(tb::kCPU);
            auto expected_hidden=check_math?reference.fused->hidden().to(tb::kCPU):tb::Tensor{};
            graph.replay();auto actual=candidate.fused->output().to(tb::kCPU);
            if(actual.sizes()!=expected.sizes() || actual.scalar_type()!=expected.scalar_type() ||
                std::memcmp(actual.data_ptr(),expected.data_ptr(),actual.numel()*actual.element_size()))
                throw std::runtime_error(std::string(option)+" changed MFE output bits");
            if(check_math) {
                auto actual_hidden=candidate.fused->hidden().to(tb::kCPU);
                if(actual_hidden.sizes()!=expected_hidden.sizes())
                    throw std::runtime_error("default math changed hidden geometry");
                const auto& b=candidate.fused->batch();
                for(int token=0;token<tokens;++token)for(int route=0;route<routes+int(shared);++route) {
                    if(route<routes && unsigned(selected[token*routes+route])>=unsigned(experts))continue;
                    const size_t slot=token*(routes+int(shared))+route;
                    const size_t offset=slot*b.hidden_stride*actual_hidden.element_size();
                    const size_t width=route==routes?b.shared_intermediate:b.intermediate;
                    if(std::memcmp(static_cast<const char*>(actual_hidden.data_ptr())+offset,
                        static_cast<const char*>(expected_hidden.data_ptr())+offset,
                        width*actual_hidden.element_size()))
                        throw std::runtime_error("default math changed hidden output bits");
                }
                ++hidden_checks;
            }
            ++checks;
        }
    }
    set_geometry(option,nullptr);
    std::cout<<label<<" checks="<<checks
        <<" bytes exact, tokens=1/3/8, shared, duplicate/masked routes, changing graph replay PASS\n";
    const char* dense_label=std::strcmp(option,"MFQ_BENCH_E8_DENSE")==0?
        "MFE_E8_DENSE hidden_checks=":"MFE_NVQ_DENSE hidden_checks=";
    if(check_math)std::cout<<(check_dense?dense_label:check_narrow?"MFE_E8_NARROW hidden_checks=":"MFE_DEFAULT_MATH hidden_checks=")<<hidden_checks
        <<" eligible="<<eligible<<" compatibility="<<compatibility<<'\n';
}
static void verify_default_math(std::vector<std::unique_ptr<Layer>>& layers) {
    for(int mode=0;mode<4;++mode) {
        set_geometry("MFQ_NINT_GROUP_DOT",mode==3?"0":"1");
        set_geometry("MFQ_MFE_NINT_LATE_SCALE",mode==1?"1":"0");
        set_geometry("MFQ_MFE_SHARED_DENSE_MATH",mode==2?"1":"0");
        std::cout<<"MFE_DEFAULT_MATH mode="<<mode<<'\n';
        verify_nvq_option(layers,"MFQ_MFE_DEFAULT_MATH","VERIFY_MFE_DEFAULT_MATH");
    }
}
static void verify_integer_delta(std::vector<std::unique_ptr<Layer>>& layers) {
    size_t checks=0,books=0;double worst=0;
    for(int tokens:{1,3,8})for(bool shared:{false,true})for(auto& layer:layers) {
        std::vector<tb::Tensor> saved;
        for(auto& projection:layer->projections)for(auto& pool:projection.pools) {
            saved.push_back(pool.nvq.integer_codebook);saved.push_back(pool.nvq.decode_records);
            if(pool.nvq.integer_codebook.defined() || pool.nvq.decode_records.defined())++books;
            pool.nvq.integer_codebook={};
            pool.nvq.decode_records={};
        }
        Call canonical(*layer,shared,tokens);
        size_t index=0;
        for(auto& projection:layer->projections)for(auto& pool:projection.pools) {
            pool.nvq.integer_codebook=saved[index++];pool.nvq.decode_records=saved[index++];
        }
        Call optimized(*layer,shared,tokens);MfqCudaGraph graph;
        mfq_prepare_cuda_graph_memory(graph);optimized.fused->run();
        fb::check(cudaStreamSynchronize(mfq_current_cuda_stream()));
        graph.capture_begin();optimized.fused->run();graph.capture_end();
        for(int step=0;step<3;++step) {
            std::vector<int32_t> selected(tokens*routes);
            for(int i=0;i<tokens*routes;++i) {
                selected[i]=layer->selected[(i/(step==1?2:1)+step)%routes];
                if(step==2 && i%3==0)selected[i]=-1;
                if(step==2 && i%3==1)selected[i]=experts;
            }
            auto ids=tb::tensor(selected).reshape({tokens,routes}).to(tb::kCUDA);
            canonical.ids.copy_(ids);optimized.ids.copy_(ids);
            canonical.fused->run();auto expected=canonical.fused->output().to(tb::kCPU);
            graph.replay();auto actual=optimized.fused->output().to(tb::kCPU);
            if(actual.sizes()!=expected.sizes() || std::memcmp(actual.data_ptr(),expected.data_ptr(),actual.numel()*actual.element_size()))
                throw std::runtime_error("NVQ1 integer delta/record decode changed MFE output bits");
            worst=std::max(worst,optimized.verify(shared));++checks;
        }
    }
    if(!books)throw std::runtime_error("NVQ1 integer delta verification exercised no transformed books");
    std::cout<<"VERIFY_NVQ1_DECODE checks="<<checks<<" books="<<books
        <<" max_reference_relative_l2="<<worst
        <<" canonical/integer bitexact, tokens=1/3/8, shared, duplicate/masked routes, graph replay PASS\n";
}
static void verify_multi_sums(std::vector<std::unique_ptr<Layer>>& layers) {
    size_t checks=0;double worst=0;
    for(int tokens:{1,3,8})for(bool shared:{false,true})for(auto& layer:layers) {
        multi_sums(false);Call reference(*layer,shared,tokens);
        multi_sums(true);Call candidate(*layer,shared,tokens);
        MfqCudaGraph graph;mfq_prepare_cuda_graph_memory(graph);
        candidate.fused->run();fb::check(cudaStreamSynchronize(mfq_current_cuda_stream()));
        graph.capture_begin();candidate.fused->run();graph.capture_end();
        for(int step=0;step<3;++step) {
            std::vector<int32_t> selected(tokens*routes);
            for(int i=0;i<tokens*routes;++i) {
                selected[i]=layer->selected[(i/(step==1?2:1)+step)%routes];
                if(step==2 && i%3==0)selected[i]=-1;
                if(step==2 && i%3==1)selected[i]=experts;
            }
            auto ids=tb::tensor(selected).reshape({tokens,routes}).to(tb::kCUDA);
            reference.ids.copy_(ids);candidate.ids.copy_(ids);
            reference.fused->run();auto expected=reference.fused->output().to(tb::kCPU);
            for(int replay=0;replay<2;++replay) {
                graph.replay();auto actual=candidate.fused->output().to(tb::kCPU);
                if(actual.sizes()!=expected.sizes() || actual.scalar_type()!=expected.scalar_type() ||
                        std::memcmp(actual.data_ptr(),expected.data_ptr(),actual.numel()*actual.element_size()))
                    throw std::runtime_error("Strata multi-row reduction changed MFE output bits");
                ++checks;
            }
            worst=std::max(worst,candidate.verify(shared));
        }
    }
    multi_sums(false);
    std::cout<<"VERIFY_MFE_MULTI_SUM checks="<<checks<<" max_reference_relative_l2="<<worst
        <<" old/new bitexact, tokens=1/3/8, shared, duplicate/masked routes, graph replay PASS\n";
}
static void set_geometry(const char* name,const char* value) {
#ifdef _WIN32
    _putenv_s(name,value?value:"");
#else
    if(value)setenv(name,value,1);else unsetenv(name);
#endif
}
static void verify_geometry(std::vector<std::unique_ptr<Layer>>& layers) {
    std::size_t output_checks=0,hidden_checks=0;
    const auto exact=[](const tb::Tensor& actual,const tb::Tensor& expected,const char* name) {
        if(actual.sizes()!=expected.sizes() || actual.scalar_type()!=expected.scalar_type() ||
            std::memcmp(actual.data_ptr(),expected.data_ptr(),actual.numel()*actual.element_size()))
            throw std::runtime_error(std::string("MFE launch geometry changed ")+name+" bits");
    };
    for(int tokens:{1,3,6,8})for(bool shared:{false,true})for(auto& layer:layers) {
        set_geometry("MFQ_MFE_GU_WARPS","8");set_geometry("MFQ_MFE_DOWN_WARPS",nullptr);
        set_geometry("MFQ_MFE_GPU_BUNDLE","1");
        Call call(*layer,shared,tokens);
        const auto source=call.x.clone();
        struct Geometry {const char* gu;const char* down;const char* bundle;};
        for(const auto& geometry:std::array<Geometry,12>{{
            {"4",nullptr,"1"},{"16",nullptr,"1"},{"8","1","1"},{"8","2","1"},{"8","3","1"},
            {"8","4","1"},{"8","5","1"},{"8","6","1"},{"8","8","1"},{"8","11","1"},
            {"8",nullptr,"0"},{"16",nullptr,"0"}}}) {
            set_geometry("MFQ_MFE_GU_WARPS",geometry.gu);
            set_geometry("MFQ_MFE_DOWN_WARPS",geometry.down);
            set_geometry("MFQ_MFE_GPU_BUNDLE",geometry.bundle);
            MfqCudaGraph graph;mfq_prepare_cuda_graph_memory(graph);
            call.fused->run();fb::check(cudaStreamSynchronize(mfq_current_cuda_stream()));
            graph.capture_begin();call.fused->run();graph.capture_end();
            for(int step=0;step<3;++step) {
                std::vector<int32_t> selected(tokens*routes);
                for(int i=0;i<tokens*routes;++i) {
                    selected[i]=layer->selected[(i/(step==1?2:1)+step)%routes];
                    if(step==2 && i%3==0)selected[i]=-1;
                    if(step==2 && i%3==1)selected[i]=experts;
                }
                call.ids.copy_(tb::tensor(selected).reshape({tokens,routes}).to(tb::kCUDA));
                call.x.copy_((source*(.25f+.625f*step)).to(tb::kFloat16));
                set_geometry("MFQ_MFE_GU_WARPS","8");set_geometry("MFQ_MFE_DOWN_WARPS",nullptr);
                set_geometry("MFQ_MFE_GPU_BUNDLE","1");
                call.fused->run();const auto expected=call.fused->output().cpu();
                const auto hidden_expected=step<2?call.fused->hidden().cpu():tb::Tensor{};
                for(int replay=0;replay<2;++replay) {
                    graph.replay();exact(call.fused->output().cpu(),expected,"output");++output_checks;
                    if(hidden_expected.defined()) {
                        exact(call.fused->hidden().cpu(),hidden_expected,"active hidden");++hidden_checks;
                    }
                }
            }
        }
    }
    set_geometry("MFQ_MFE_GU_WARPS",nullptr);set_geometry("MFQ_MFE_DOWN_WARPS",nullptr);
    set_geometry("MFQ_MFE_GPU_BUNDLE",nullptr);
    std::cout<<"VERIFY_MFE_GEOMETRY output_checks="<<output_checks<<" hidden_checks="<<hidden_checks
        <<" tokens=1/3/6/8, shared, GU4/8/16, Down1/2/3/4/5/6/8/11, bundle0/1, changing inputs, duplicate/masked routes, repeated graph replay bitexact PASS\n";
}
struct Replay {
    MfqCudaGraph graph;
    std::vector<tb::Tensor> outputs;
    size_t calls;
    Replay(std::vector<std::unique_ptr<Call>>& banks,int stage):calls(banks.size()*8) {
        mfq_prepare_cuda_graph_memory(graph);outputs.reserve(calls);
        for(size_t i=0;i<calls;++i)outputs.push_back(banks[i%banks.size()]->run(stage));
        fb::check(cudaStreamSynchronize(mfq_current_cuda_stream()));outputs.clear();
        graph.capture_begin();
        for(size_t i=0;i<calls;++i)outputs.push_back(banks[i%banks.size()]->run(stage));
        graph.capture_end();graph.replay();fb::check(cudaStreamSynchronize(mfq_current_cuda_stream()));
    }
};
int main(int argc,char** argv)try {
    if(argc!=3)throw std::runtime_error("usage: mfq-cuda-real-mfe-bench model-first-shard profile-label");
    auto context=mfq::cuda::default_context(0);auto stream=mfq_get_stream_from_pool(false);MfqCudaStreamGuard guard(stream);
    auto model=mfq::open_model_source(argv[1]);
    std::vector<std::unique_ptr<Layer>> layers;
    for(int layer:{0,8,16,24,32,40})layers.push_back(std::make_unique<Layer>(*model,layer));
    if(std::getenv("MFQ_BENCH_E8_LARGE_ARENA")) {
        bool enlarged=false;for(auto& layer:layers)if(layer->enlarge_e8_arena()){enlarged=true;break;}
        if(!enlarged || !std::getenv("MFQ_BENCH_VERIFY_E8_NARROW"))
            throw std::runtime_error("large E8 arena requires the narrow arithmetic verification mode");
    }
    if(std::getenv("MFQ_BENCH_VERIFY_E8_DENSE")){
        verify_nvq_option(layers,"MFQ_BENCH_E8_DENSE","VERIFY_MFE_E8_DENSE");return 0;
    }
    if(std::getenv("MFQ_BENCH_VERIFY_D4_DENSE")){
        verify_nvq_option(layers,"MFQ_BENCH_D4_DENSE","VERIFY_MFE_D4_DENSE");return 0;
    }
    if(std::getenv("MFQ_BENCH_VERIFY_NVQ_DENSE")){
        verify_nvq_option(layers,"MFQ_BENCH_NVQ_DENSE","VERIFY_MFE_NVQ_DENSE");return 0;
    }
    if(std::getenv("MFQ_BENCH_VERIFY_E8_NARROW")){
        verify_nvq_option(layers,"MFQ_MFE_E8_NARROW","VERIFY_MFE_E8_NARROW");return 0;
    }
    if(std::getenv("MFQ_BENCH_VERIFY_DEFAULT_MATH")){verify_default_math(layers);return 0;}
    if(std::getenv("MFQ_BENCH_VERIFY_GEOMETRY")){verify_geometry(layers);return 0;}
    if(std::getenv("MFQ_BENCH_VERIFY_MULTI_SUM")){verify_multi_sums(layers);return 0;}
    if(std::getenv("MFQ_BENCH_VERIFY_NVQ_WORD_SIGNS")){verify_nvq_option(layers,"MFQ_MFE_NVQ_WORD_SIGNS","VERIFY_NVQ_WORD_SIGNS");return 0;}
    if(std::getenv("MFQ_BENCH_VERIFY_NVQ1_DELTA")){verify_integer_delta(layers);return 0;}
    if(std::getenv("MFQ_BENCH_VERIFY_MFE_FORMATS")){verify_packing(layers,true);return 0;}
    if(std::getenv("MFQ_BENCH_VERIFY_MFE_ROUTES")){verify_packing(layers);return 0;}
    const int tokens=std::getenv("MFQ_BENCH_M")?std::atoi(std::getenv("MFQ_BENCH_M")):1;
    if(tokens<1 || tokens>8)throw std::runtime_error("MFE benchmark supports tokens=1..8");
    const char* stage_setting=std::getenv("MFQ_BENCH_STAGE");
    const int selected_stage=stage_setting?std::atoi(stage_setting):-1;
    if(selected_stage < -1 || selected_stage>5 || (stage_setting && selected_stage<0))
        throw std::runtime_error("MFE benchmark stage must be 0..5");
    cudaDeviceProp properties{};fb::check(cudaGetDeviceProperties(&properties,0));
    const char* names[]{"gate_10_experts","up_10_experts","down_10_experts","fused_gate_up","fused_down_reduce","full_mfe_ffn"};
    for(bool shared:{false,true}) {
        std::vector<std::unique_ptr<Call>> calls;double worst=0;
        for(auto& layer:layers) {
            calls.push_back(std::make_unique<Call>(*layer,shared,tokens));
            worst=std::max(worst,calls.back()->verify(shared));
        }
        const int first_stage=selected_stage>=0?selected_stage:(shared || std::getenv("MFQ_BENCH_FULL_ONLY")?5:0);
        for(int stage=first_stage;stage<(selected_stage>=0?selected_stage+1:6);++stage) {
            size_t bytes=0;
            for(const auto& layer:layers) {
                if(stage<3)bytes+=layer->bytes[stage];
                else if(stage==3)bytes+=layer->fused_bytes[0]+layer->fused_bytes[1]+(shared?
                    layer->shared_projection_bytes[0]+layer->shared_projection_bytes[1]:0);
                else if(stage==4)bytes+=layer->fused_bytes[2]+(shared?layer->shared_projection_bytes[2]:0);
                else bytes+=layer->fused_bytes[0]+layer->fused_bytes[1]+layer->fused_bytes[2]+(shared?layer->shared_bytes:0);
            }
            if(bytes<4*size_t(properties.l2CacheSize))throw std::runtime_error("MFE rotating footprint below four L2 caches");
            Replay replay(calls,stage);
            auto samples=fb::measure([&](){replay.graph.replay();},stream.stream(),replay.calls);
            const double us=fb::median(samples),per_call=double(bytes)/layers.size();
            std::cout<<std::setprecision(10)<<"{\"profile\":\""<<argv[2]<<"\",\"stage\":\""<<names[stage]
                <<"\",\"tokens\":"<<tokens<<",\"routes\":10,\"layers\":6,\"shared\":"<<(shared?"true":"false")
                <<",\"us\":"<<us<<",\"weight_bytes_per_call\":"<<per_call<<",\"working_bytes\":"<<bytes
                <<",\"effective_GBs\":"<<per_call/(us*1000.)<<",\"reference_relative_l2\":"<<worst
                <<",\"inputs\":\"synthetic\",\"routing\":\"fixed_distributed_ids\",\"scope\":\"gpu_resident_ffn_cuda_graph\",\"samples_us\":[";
            for(size_t i=0;i<samples.size();++i)std::cout<<(i?",":"")<<samples[i];
            std::cout<<"]}\n"<<std::flush;
        }
    }
    return 0;
}catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
