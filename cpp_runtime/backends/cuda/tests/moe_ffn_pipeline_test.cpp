#include "storage/moe_ffn_pipeline.h"
#include "runtime/decode_window.h"
#include "runtime/moe_pipeline.h"
#include "runtime/router_lookahead_audit.h"
#include "mfq_cuda_moe_ops.h"
#include "runtime/moe_residency.h"
#include "storage/moe_quant_range_source.h"
#include "storage/moe_expert_cache.h"
#include "storage/weight_loader.h"
#include "moe.h"
#include "mfq/moe_dispatch_plan.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <regex>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
namespace tb=mfq_tensor_backend;
namespace {
std::size_t bytes(const MixedMoePool& p) {
    std::vector<tb::Tensor> fields=p.family==MixedMoeFamily::Nint ?
        std::vector<tb::Tensor>{p.nint.q_packed,p.nint.row_q_bits,p.nint.row_q_bit_offsets,p.nint.sub_scale,p.nint.sub_min,p.nint.neuron_scale,p.nint.neuron_min} :
        std::vector<tb::Tensor>{p.nvq.indices_packed,p.nvq.aux_packed,p.nvq.sub_scale_packed,p.nvq.neuron_scale};
    std::size_t n=0;for(const auto& f:fields)n+=f.numel()*f.element_size();return n;
}
std::shared_ptr<MoeQuantRangeSource> source(const std::filesystem::path& path) {
    auto model=mfq::open_model_source(path.string());auto read=model->tensor_reader("linear.weight");
    auto store=std::make_shared<mfq::MfeQuantExpertStore>(require_tensor(*model,"linear.weight").nbytes,
        [read](std::size_t off,uint8_t* dst,std::size_t n){read(off,reinterpret_cast<std::byte*>(dst),n);});
    return std::make_shared<MoeQuantRangeSource>(store);
}
QuantLinear linear(const MixedMoePool& pool,int output,int width) {
    QuantLinear l;l.logical_out=output;l.logical_neuron_len=width;
    if(pool.family==MixedMoeFamily::Nint){l.kind=QuantLinearKind::Nint;l.nint=pool.nint;}
    else {l.kind=QuantLinearKind::Nvq;l.nvq=pool.nvq;}return l;
}

int verify_shared_input(const std::filesystem::path& root) {
    CudaExecutionContext execution;
    auto load=[&](const std::array<std::string,4>& formats) {
        MixedMoeRuntime result;result.n_experts=12;result.out_per_expert=640;result.neuron_len=2560;
        result.partial_experts=true;
        for(int i=0;i<4;++i) {
            auto model=mfq::open_model_source((root/(formats[i]+"-640-2560.mfq")).string());
            auto single=make_mixed_moe_runtime(load_mfe_cpu(*model,"linear.weight"),true,execution.config);
            for(auto& pool:single->pools) {
                std::vector<int32_t> local(12,-1);
                for(int expert=0;expert<3;++expert)local[i*3+expert]=expert;
                pool.expert_local=tb::tensor(local).to(tb::kCUDA);result.pools.push_back(std::move(pool));
            }
        }
        initialize_mixed_nvq_dispatch(result,execution.config);return result;
    };
    auto reference_gate=load({"nint4","nint5","nvq2j","nvq3j-512"});
    auto reference_up=load({"nint6","nint8","nvq1-l","nvq2j-l"});
    auto gate=reference_gate,up=reference_up;
    auto geometries=gate.activation_geometry();
    const auto gate_geometries=geometries;
    for(const auto& geometry:up.activation_geometry())
        if(std::find(geometries.begin(),geometries.end(),geometry)==geometries.end())geometries.push_back(geometry);
    if(geometries.size()<=gate_geometries.size())throw std::runtime_error("shared-input fixture lacks Up-only geometry");
    const auto same=[](const tb::Tensor& a,const tb::Tensor& b,const char* label) {
        auto x=a.to(tb::kCPU).contiguous(),y=b.to(tb::kCPU).contiguous();
        if(x.sizes()!=y.sizes() || x.scalar_type()!=y.scalar_type() ||
            std::memcmp(x.data_ptr(),y.data_ptr(),x.numel()*x.element_size()))
            throw std::runtime_error(std::string("shared-input ")+label+" is not byte exact");
    };
    const auto forward=[&](MixedMoeRuntime& runtime,const tb::Tensor& x,const MoeRoutePlan& route,bool prepared) {
        return runtime.forward(execution.config,execution.kl_mmq,false,false,x,route,prepared);
    };
    const auto stream=mfq_current_cuda_stream();auto context=mfq::cuda::default_context(mfq_current_cuda_device());
    context->begin_graph_pool(stream);int cases=0;
    for(int tokens:{1,3,8}) {
        auto x=tb::zeros({tokens,2560},tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat16));
        auto ids=tb::zeros({tokens,10},x.options().dtype(tb::kInt32));
        std::vector<int64_t> pointers;int groups=0;
        for(const auto& geometry:geometries) {
            auto workspace=gate.activation_workspace(x,tokens,geometry.groups,geometry.gs,{});
            up.activation_workspaces.insert_or_assign({tokens,geometry.groups,geometry.gs,x.get_device(),{}},workspace);
            pointers.insert(pointers.end(),{reinterpret_cast<int64_t>(workspace.qx.data_ptr()),
                reinterpret_cast<int64_t>(workspace.xscale.data_ptr()),geometry.groups,geometry.gs});
            groups+=geometry.groups;
        }
        auto descriptors=tb::tensor(pointers).reshape({int64_t(geometries.size()),4}).to(tb::kCUDA);
        tb::Tensor actual_gate,actual_up;mfq::cuda::DecodeWindow window(stream);
        window.capture([&] {
            auto route=build_moe_route_plan(ids,12);
            moe_quantize_shared_input_cuda(x,descriptors,groups);
            actual_gate=forward(gate,x,route,true);actual_up=forward(up,x,route,true);
        },[]{},[&]{actual_gate={};actual_up={};});
        for(int step=0;step<3;++step) {
            std::vector<float> input(tokens*2560);
            for(int t=0;t<tokens;++t)for(int k=0;k<2560;++k)
                input[t*2560+k]=k<64?0.0f:std::sin(float(k+t*17+step*13)*.173f)*.13f;
            std::vector<int32_t> selected(tokens*10);
            for(int i=0;i<tokens*10;++i)selected[i]=(i*5+step*3)%12;
            x.copy_(tb::tensor(input).reshape({tokens,2560}).to(tb::kFloat16).to(tb::kCUDA));
            ids.copy_(tb::tensor(selected).reshape({tokens,10}).to(tb::kCUDA));
            auto route=build_moe_route_plan(ids,12);
            execution.config.moe_nint_heterogeneous_decode=false;
            auto expected_gate=forward(reference_gate,x,route,false),expected_up=forward(reference_up,x,route,false);
            execution.config.moe_nint_heterogeneous_decode=true;
            window.run();same(actual_gate,expected_gate,"mixed Gate output");same(actual_up,expected_up,"mixed Up output");
            for(const auto& geometry:geometries) {
                const bool in_gate=std::find(gate_geometries.begin(),gate_geometries.end(),geometry)!=gate_geometries.end();
                auto& expected=(in_gate?reference_gate:reference_up).activation_workspace(x,tokens,geometry.groups,geometry.gs,{});
                auto& actual=gate.activation_workspace(x,tokens,geometry.groups,geometry.gs,{});
                same(actual.qx,expected.qx,"quantized bytes/tail padding");same(actual.xscale,expected.xscale,"group scales");
            }
            ++cases;
        }
    }
    MFQ_CUDA_CHECK(cudaStreamSynchronize(stream));context->end_graph_pool(stream);return cases;
}

int verify_nint_dispatch(const std::filesystem::path& root) {
    CudaExecutionContext execution;int cases=0;
    auto context=mfq::cuda::default_context(mfq_current_cuda_device());
    const auto stream=mfq_current_cuda_stream();
    context->begin_graph_pool(stream);
    const auto same=[](const tb::Tensor& a,const tb::Tensor& b) {
        auto x=a.cpu().contiguous(),y=b.cpu().contiguous();
        if(x.sizes()!=y.sizes() || x.scalar_type()!=y.scalar_type() ||
            std::memcmp(x.data_ptr(),y.data_ptr(),x.numel()*x.element_size()))
            throw std::runtime_error("grouped NINT output differs from original pool kernels");
    };
    for(bool down:{false,true}) {
        const int width=down?640:2560,output=down?2560:640;
        MixedMoeRuntime runtime;runtime.n_experts=512;runtime.neuron_len=width;
        runtime.out_per_expert=output;runtime.partial_experts=true;
        int index=0;
        for(const std::string format:{"nint4","nint5","nint6","nint8"}) {
            auto model=mfq::open_model_source((root/(format+(down?"-2560-640.mfq":"-640-2560.mfq"))).string());
            auto single=make_mixed_moe_runtime(load_mfe_cpu(*model,"linear.weight"),true,execution.config);
            for(auto& pool:single->pools) {
                std::vector<int32_t> local(512,-1);for(int e=0;e<3;++e)local[index*3+e]=e;
                pool.expert_local=tb::tensor(local).to(tb::kCUDA);runtime.pools.push_back(std::move(pool));
            }
            ++index;
        }
        initialize_mixed_nvq_dispatch(runtime,execution.config);
        if(!runtime.nint_dispatch || runtime.nint_dispatch->pools.size()!=4)
            throw std::runtime_error("mixed NINT fixture did not initialize four format pools");
        for(int tokens:{1,3,8}) {
            auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat16);
            auto x=tb::zeros(down?std::vector<int64_t>{tokens,10,width}:std::vector<int64_t>{tokens,width},options);
            auto ids=tb::zeros({tokens,10},options.dtype(tb::kInt32));tb::Tensor actual;
            const auto forward=[&] {
                return runtime.forward(execution.config,execution.kl_mmq,false,false,x,build_moe_route_plan(ids,512));
            };
            execution.config.moe_nint_heterogeneous_decode=true;
            mfq::cuda::DecodeWindow window(stream);
            window.capture([&]{actual=forward();},[]{},[&]{actual={};});
            for(int step=0;step<3;++step) {
                std::vector<float> values(x.numel());std::vector<int32_t> selected(tokens*10);
                for(std::size_t i=0;i<values.size();++i)values[i]=std::sin(float(i+step*31)*.017f)*.3f;
                for(int i=0;i<tokens*10;++i)selected[i]=i%10==9?511:(i*5+step*3)%12;
                for(std::size_t p=0;p<runtime.pools.size();++p) {
                    std::vector<int32_t> map(512,-1);
                    for(int e=0;e<3;++e)if(e!=step)map[p*3+e]=e;
                    runtime.pools[p].expert_local.copy_(tb::tensor(map).to(tb::kCUDA));
                }
                x.copy_(tb::tensor(values).reshape(x.sizes().vec()).to(tb::kFloat16).to(tb::kCUDA));
                ids.copy_(tb::tensor(selected).reshape({tokens,10}).to(tb::kCUDA));
                execution.config.moe_nint_heterogeneous_decode=false;auto expected=forward();
                execution.config.moe_nint_heterogeneous_decode=true;window.run();same(actual,expected);
                ++cases;
            }
        }
    }
    std::cout<<"nint_dispatch_cases="<<cases<<" real640/2560 four-format original-pool bytes, shared/routed inputs and changing ownership graphs exact PASS\n";
    MFQ_CUDA_CHECK(cudaStreamSynchronize(stream));context->end_graph_pool(stream);
    return cases;
}

int verify_phased_wire_bytes() {
    const std::vector<int64_t> sizes={3,1280,2564,640*107*16+7};
    const std::vector<int64_t> prefixes={1,16,4,0};
    const std::size_t header=(32+24*sizes.size()+15)&~std::size_t(15);
    std::vector<uint8_t> packed(header,0);
    std::vector<tb::Tensor> outputs;
    std::vector<std::vector<uint8_t>> initial,expected;
    std::vector<uint64_t> descriptors;
    std::size_t boundary=0;
    for(std::size_t field=0;field<sizes.size();++field) {
        initial.emplace_back(sizes[field]+prefixes[field]+32,0xa7);expected.push_back(initial.back());
        outputs.push_back(tb::tensor(initial.back()).to(tb::kCUDA));
        packed.resize((packed.size()+15)&~std::size_t(15),0);
        if(field==2)boundary=packed.size();
        descriptors.push_back(reinterpret_cast<uint64_t>(outputs.back().data_ptr<uint8_t>()+prefixes[field]));
        descriptors.push_back(packed.size());descriptors.push_back(sizes[field]);
        for(int64_t i=0;i<sizes[field];++i) {
            const auto value=static_cast<uint8_t>(i*29+field*71);
            packed.push_back(value);expected.back()[prefixes[field]+i]=value;
        }
    }
    const uint64_t words[4]={sizes.size(),header,packed.size(),boundary};
    std::memcpy(packed.data(),words,sizeof(words));std::memcpy(packed.data()+32,descriptors.data(),descriptors.size()*8);
    auto wire=tb::tensor(packed).to(tb::kCUDA);int blocks=0,threads=0;
    MFQ_CUDA_CHECK(mfq::cuda::moe_wire_launch_geometry(&blocks,&threads));
    auto cancelled=tb::zeros({1},wire.options().dtype(tb::kInt32));int cases=0;
    for(bool abort:{false,true}) {
        for(std::size_t f=0;f<outputs.size();++f)outputs[f].copy_(tb::tensor(initial[f]).to(tb::kCUDA));
        cancelled.fill_(abort?1:0);
        for(int phase=1;phase<=2;++phase) {
            mfq::cuda::moe_scatter_wire_phase(wire.data_ptr(),reinterpret_cast<const uint32_t*>(cancelled.data_ptr<int32_t>()),
                blocks,threads,phase,mfq_current_cuda_stream());
            for(std::size_t f=0;f<outputs.size();++f) {
                const auto& oracle=abort || (phase==1 && f>=2)?initial[f]:expected[f];
                auto actual=outputs[f].cpu().contiguous();
                if(std::memcmp(actual.data_ptr(),oracle.data(),oracle.size()))
                    throw std::runtime_error("phased scatter crossed the Gate/Up/Down boundary or neighboring byte guards");
            }
            ++cases;
        }
    }
    // Empty Gate/Up and empty Down each preserve the complete wire contract.
    for(int empty:{1,2}) {
        auto modified=packed;const uint64_t split=empty==1?header:(packed.size()+15)&~std::size_t(15);
        std::memcpy(modified.data()+24,&split,8);wire.copy_(tb::tensor(modified).to(tb::kCUDA));cancelled.zero_();
        for(std::size_t f=0;f<outputs.size();++f)outputs[f].copy_(tb::tensor(initial[f]).to(tb::kCUDA));
        for(int phase=1;phase<=2;++phase)mfq::cuda::moe_scatter_wire_phase(wire.data_ptr(),
            reinterpret_cast<const uint32_t*>(cancelled.data_ptr<int32_t>()),blocks,threads,phase,mfq_current_cuda_stream());
        for(std::size_t f=0;f<outputs.size();++f) {
            auto actual=outputs[f].cpu().contiguous();
            if(std::memcmp(actual.data_ptr(),expected[f].data(),expected[f].size()))
                throw std::runtime_error("empty phased packet changed original payload bytes");
        }
        ++cases;
    }
    std::cout<<"phased_wire_cases="<<cases<<" boundary and tail bytes, cancelled/empty packets exact PASS\n";
    return cases;
}

int verify_wire_bytes() {
    const std::vector<int64_t> sizes={3,1280,2564,640*107*16+7};
    const std::vector<int64_t> prefixes={1,16,4,0};
    const std::size_t header=(32+24*sizes.size()+15)&~std::size_t(15);
    std::vector<uint8_t> packed(header,0);
    std::vector<tb::Tensor> outputs;
    std::vector<std::vector<uint8_t>> expected;
    std::vector<uint64_t> descriptors;
    for(std::size_t field=0;field<sizes.size();++field) {
        std::vector<uint8_t> initial(sizes[field]+prefixes[field]+32,0xa7);
        outputs.push_back(tb::tensor(initial).to(tb::kCUDA));expected.push_back(initial);
        packed.resize((packed.size()+15)&~std::size_t(15),0);
        descriptors.push_back(reinterpret_cast<uint64_t>(outputs.back().data_ptr<uint8_t>()+prefixes[field]));
        descriptors.push_back(packed.size());descriptors.push_back(sizes[field]);
        for(int64_t i=0;i<sizes[field];++i) {
            const auto value=static_cast<uint8_t>(i*29+field*71);
            packed.push_back(value);expected.back()[prefixes[field]+i]=value;
        }
    }
    const uint64_t words[4]={sizes.size(),header,packed.size(),0};
    std::memcpy(packed.data(),words,sizeof(words));
    std::memcpy(packed.data()+32,descriptors.data(),descriptors.size()*sizeof(uint64_t));
    auto wire=tb::tensor(packed).to(tb::kCUDA);int blocks=0,threads=0;
    auto cancelled=tb::zeros({1},wire.options().dtype(tb::kInt32));
    mfq::cuda::HostBuffer flags(128,true);std::memset(flags.data(),0,flags.size());
    auto* host_flag=static_cast<uint32_t*>(flags.data());uint32_t* mapped_flag=nullptr;
    MFQ_CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&mapped_flag),host_flag,0));
    auto slot=tb::ones({1},cancelled.options());
    auto kind=tb::empty({1},cancelled.options());host_flag[2]=1;
    mfq::cuda::publish_mapped_flag(host_flag);
    std::vector<tb::Tensor> maps;
    for(int i=0;i<3;++i)maps.push_back(tb::empty({1},cancelled.options()));
    std::vector<int64_t> pointers(3,reinterpret_cast<int64_t>(slot.data_ptr()));
    for(const auto& map:maps)pointers.push_back(reinterpret_cast<int64_t>(map.data_ptr()));
    auto dispatch=tb::tensor(pointers).to(tb::kCUDA);
    MFQ_CUDA_CHECK(mfq::cuda::moe_wire_launch_geometry(&blocks,&threads));
    for(int replay=0;replay<3;++replay) {
        if(replay==2)mfq::cuda::publish_mapped_flag(host_flag+1);
        mfq::cuda::wait_mapped_plan(mapped_flag,reinterpret_cast<const int32_t*>(mapped_flag+2),kind.data_ptr<int32_t>(),
            1,mapped_flag+1,reinterpret_cast<uint32_t*>(cancelled.data_ptr<int32_t>()),mfq_current_cuda_stream());
        mfq::cuda::moe_update_dispatch_maps(reinterpret_cast<const uint64_t*>(dispatch.data_ptr<int64_t>()),
            1,1,kind.data_ptr<int32_t>(),reinterpret_cast<const uint32_t*>(cancelled.data_ptr<int32_t>()),
            mfq_current_cuda_stream());
        mfq::cuda::moe_scatter_wire(wire.data_ptr(),reinterpret_cast<const uint32_t*>(cancelled.data_ptr<int32_t>()),
            blocks,threads,mfq_current_cuda_stream());
        MFQ_CUDA_CHECK(cudaStreamSynchronize(mfq_current_cuda_stream()));
        if(cancelled.to(tb::kCPU).data_ptr<int32_t>()[0]!=(replay==2))
            throw std::runtime_error("mapped cancellation snapshot differs from host publication");
        if(kind.to(tb::kCPU).data_ptr<int32_t>()[0]!=1)
            throw std::runtime_error("mapped dispatch snapshot differs from host publication");
        for(int i=0;i<3;++i)if(maps[i].to(tb::kCPU).data_ptr<int32_t>()[0]!=(i==0 && replay!=2 ? 1 : -1))
            throw std::runtime_error("dispatch maps differ from cancellation snapshot");
        for(std::size_t field=0;field<sizes.size();++field) {
            auto actual=outputs[field].to(tb::kCPU).contiguous();
            if(std::memcmp(actual.data_ptr(),expected[field].data(),expected[field].size()))
                throw std::runtime_error("packed DMA scatter changed field bytes or neighboring guards");
        }
        if(!replay) {
            std::fill_n(packed.data(),8,uint8_t(0));
            for(std::size_t i=header;i<packed.size();++i)packed[i]^=0x5a;
            wire.copy_(tb::tensor(packed).to(tb::kCUDA));
        }
        if(replay==1) {
            std::memcpy(packed.data(),words,sizeof(words));wire.copy_(tb::tensor(packed).to(tb::kCUDA));
        }
    }
    return 3;
}
}
int verify_warm_residency(const std::filesystem::path& root) {
    int cases=0,phase_cases=0,async_cases=0,window_cases=0;
    const auto wait_preparation=[](const std::atomic_bool& prepared) {
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(10);
        while(!prepared.load(std::memory_order_acquire)) {
            if(std::chrono::steady_clock::now()>deadline)
                throw std::runtime_error("window residency preparation did not progress before graph finish");
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    };
    for(const std::string format:{"nint5","nvq3j-512"})for(int mode:{0,1,2}) {
        CudaExecutionContext execution;execution.config.moe_residency_warm=true;
        const bool limited=mode==1,shared=mode==2;
        if(shared && !execution.config.moe_direct_ram)continue;
        execution.config.moe_host_physical_bytes=limited?1024:0;
        const int count=mode?2:33;
        std::array<std::shared_ptr<MoeQuantRangeSource>,3> original={source(root/(format+"-640-2560.mfq")),
            source(root/(format+"-640-2560.mfq")),source(root/(format+"-2560-640.mfq"))};
        std::size_t bundle_bytes=0;for(const auto& s:original)bundle_bytes+=bytes(s->metadata()->pools[0]);
        execution.config.moe_host_cache_bytes=3*count*bundle_bytes;
        auto cache=make_moe_expert_cache(count*bundle_bytes,execution.config);
        std::vector<std::array<std::shared_ptr<MoeQuantRangeSource>,3>> sources(count);
        std::vector<std::array<std::shared_ptr<MfeWeight>,3>> weights(count);
        std::vector<MoeFfnForward> forwards;
        for(int layer=0;layer<count;++layer) {
            for(int i=0;i<3;++i) {
                sources[layer][i]=source(root/(format+(i==2?"-2560-640.mfq":"-640-2560.mfq")));
                weights[layer][i]=std::make_shared<MfeWeight>(cache_quant_moe_weight(cache,
                    std::to_string(layer)+"/"+std::to_string(i),sources[layer][i],1,layer,std::to_string(i)));
            }
            forwards.push_back(make_moe_ffn_pipeline({weights[layer][0],weights[layer][1]},{weights[layer][2]}));
        }
        finalize_moe_expert_cache(cache);
        if(shared) {
            std::vector<float> values(2560);for(int j=0;j<2560;++j)values[j]=std::sin(float(j)*.013f);
            auto x=tb::tensor(values).reshape({1,2560}).to(tb::kFloat16).to(tb::kCUDA);
            auto ids=tb::zeros({1,1},x.options().dtype(tb::kInt32));
            auto route_weights=tb::ones({1,1},x.options());
            auto reference=forwards[0](execution,x,ids,route_weights).cpu().contiguous();
            MoeResidencyManager failed(cache.get(),[](const char* phase,std::size_t) {
                if(std::string(phase)=="publish")throw std::runtime_error("shared backup publication failure");
            });
            for(int layer=0;layer<count;++layer)failed.after_layer(layer,std::vector<int32_t>(35,1),35);
            bool rejected=false;
            try {failed.apply_pending();}catch(const std::runtime_error& e) {
                rejected=std::string(e.what())=="shared backup publication failure";
            }
            if(!rejected || !failed.stats().shared_backup_bytes)
                throw std::runtime_error("shared residency backup rollback was not exercised");
            if(failed.prepare_during_window()) {
                std::atomic_bool prepared=false;
                {
                    MoeResidencyManager cancelled(cache.get(),[&](const char* phase,std::size_t index) {
                        if(std::string(phase)=="upload" && index==std::size_t(count*3-1))
                            prepared.store(true,std::memory_order_release);
                    });
                    for(int layer=0;layer<count;++layer)
                        cancelled.after_layer(layer,std::vector<int32_t>(35,1),35,true);
                    wait_preparation(prepared);
                    if(cancelled.stats().committed_projections)
                        throw std::runtime_error("unfinished window published its residency transaction");
                    // Destruction must wake a deferred publisher and restore
                    // every touched GPU field without exchanging RAM leases.
                }
                ++window_cases;
            }
            auto actual=forwards[0](execution,x,ids,route_weights).cpu().contiguous();
            if(std::memcmp(actual.data_ptr(),reference.data_ptr(),actual.numel()*actual.element_size()))
                throw std::runtime_error("shared residency backup corrupted GPU weights");
            for(int layer=0;layer<count;++layer)for(int i=0;i<3;++i) {
                if(!sources[layer][i]->gpu_resident(0) || sources[layer][i]->gpu_resident(1))
                    throw std::runtime_error("shared backup failure changed tier ownership");
                const int width=i==2?640:2560,output=i==2?2560:640;
                auto input=tb::tensor(std::vector<float>(values.begin(),values.begin()+width)).reshape({1,width});
                auto lease=sources[layer][i]->acquire_expert(1);
                auto a=linear(lease->weights,output,width).forward(execution,input).to(tb::kFloat32).contiguous();
                auto b=linear(original[i]->read_expert(1),output,width).forward(execution,input).to(tb::kFloat32).contiguous();
                if(std::memcmp(a.data_ptr(),b.data_ptr(),a.numel()*sizeof(float)))
                    throw std::runtime_error("shared backup failure changed RAM weights");
            }
        }
        auto memory=limited?MoeResidencyManager::MemoryProbe([]{return MoeResidencyManager::MemorySample{1000,1ull<<30};})
            :MoeResidencyManager::MemoryProbe{};
        std::atomic_bool prepared=false;
        MoeResidencyManager manager(cache.get(),[&](const char* phase,std::size_t index) {
            if(std::string(phase)=="upload" && index==std::size_t(count*3-1))
                prepared.store(true,std::memory_order_release);
        },std::move(memory));
        const bool in_window=manager.prepare_during_window() && !limited;
        for(int layer=0;layer<count;++layer)manager.after_layer(layer,std::vector<int32_t>(35,1),35,in_window);
        if(in_window) {
            wait_preparation(prepared);
            if(manager.stats().committed_projections || manager.stats().window_prepares!=1)
                throw std::runtime_error("window preparation crossed its publication barrier");
            for(int layer=0;layer<count;++layer)for(int p=0;p<3;++p)
                if(!sources[layer][p]->gpu_resident(0) || sources[layer][p]->gpu_resident(1))
                    throw std::runtime_error("window preparation changed expert ownership before graph finish");
            manager.finish_window(count-1);++window_cases;
        }
        if(manager.stats().async_publish && !limited) {
            // Publication must progress without a call on the model thread.
            // The remaining checks verify both GPU ownership and RAM bytes.
            const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(10);
            while(manager.stats().committed_bundles!=std::uint64_t(count)) {
                if(std::chrono::steady_clock::now()>deadline)
                    throw std::runtime_error("residency publication still requires the model thread");
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            ++async_cases;
        }
        manager.apply_pending();const auto stats=manager.stats();
        if(stats.batched && !limited && (stats.backup_copies!=1 || stats.batch_descriptor_copies!=1 ||
                !stats.batch_device_bytes || !stats.upload_copies))
            throw std::runtime_error("warm residency did not batch the complete backup and field descriptors");
        if(stats.observed_routes!=std::uint64_t(count*35) || stats.scheduled_rounds!=1)
            throw std::runtime_error("warm residency did not observe the complete prefill");
        if(limited) {
            if(stats.committed_bundles || stats.planned_bundles || stats.memory_rejections!=std::uint64_t(count))
                throw std::runtime_error("warm residency exceeded injected physical memory bound");
        } else if(stats.committed_bundles!=std::uint64_t(count) || stats.planned_bundles!=std::uint64_t(count))
            throw std::runtime_error("warm residency retained an arbitrary expert count cap");
        if(shared && (!stats.shared_backup_bytes || stats.backup_peak_bytes!=stats.shared_backup_bytes))
            throw std::runtime_error("warm residency did not reuse the pinned transfer workspace");
        for(int layer=0;layer<count;++layer)for(int i=0;i<3;++i) {
            if(sources[layer][i]->gpu_resident(1)==limited || sources[layer][i]->gpu_resident(0)!=limited ||
                sources[layer][i]->expert_disk_reads_after_preload())
                throw std::runtime_error("warm residency publication or disk sealing changed");
            if(!limited) {
                const int width=i==2?640:2560,output=i==2?2560:640;
                std::vector<float> input_values(width);for(int j=0;j<width;++j)input_values[j]=std::sin(float(j)*.013f);
                auto input=tb::tensor(input_values).reshape({1,width});
                auto lease=sources[layer][i]->acquire_expert(0);
                auto actual=linear(lease->weights,output,width).forward(execution,input).to(tb::kFloat32).contiguous();
                auto expected=linear(original[i]->read_expert(0),output,width).forward(execution,input).to(tb::kFloat32).contiguous();
                if(std::memcmp(actual.data_ptr(),expected.data_ptr(),actual.numel()*sizeof(float)))
                    throw std::runtime_error("warm residency changed demoted RAM expert arithmetic");
            }
        }
        if(!limited && execution.config.moe_direct_ram && (stats.host_pack_bytes || !stats.direct_upload_bytes))
            throw std::runtime_error("warm residency did not use registered RAM directly");
        if(stats.batched && !limited) {
            for(int layer=0;layer<count;++layer)
                manager.after_layer(layer,layer?std::vector<int32_t>{}:std::vector<int32_t>(35,2),1);
            manager.apply_pending();const auto next=manager.stats();
            if(!next.batch_device_bytes || next.batch_device_bytes>=stats.batch_device_bytes)
                throw std::runtime_error("decode retained the larger prefill device copy workspace");
            for(int p=0;p<3;++p) {
                if(!sources[0][p]->gpu_resident(2) || sources[0][p]->gpu_resident(1))
                    throw std::runtime_error("workspace resize changed incoming expert ownership");
                const int width=p==2?640:2560,output=p==2?2560:640;
                auto input=tb::ones({1,width},tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32));
                auto lease=sources[0][p]->acquire_expert(1);
                auto actual=linear(lease->weights,output,width).forward(execution,input).contiguous();
                auto expected=linear(original[p]->read_expert(1),output,width).forward(execution,input).contiguous();
                if(std::memcmp(actual.data_ptr(),expected.data_ptr(),actual.numel()*actual.element_size()))
                    throw std::runtime_error("workspace resize changed the demoted RAM expert bytes");
            }
            ++phase_cases;
        }
        ++cases;
    }
    std::cout<<"residency_workspace_phase_cases="<<phase_cases<<" changing copy workspace/ownership/demoted RAM exact PASS\n";
    std::cout<<"residency_async_publish_cases="<<async_cases<<" worker publishes GPU/RAM transaction before model-thread join PASS\n";
    std::cout<<"residency_window_prepare_cases="<<window_cases<<" prepare before graph finish/publication barrier/cancellation rollback PASS\n";
    std::cout<<"warm_residency_cases="<<cases<<" 33bundles/singleprefill/physicalbudget/shared-backup-rollback/RAMexact PASS\n";
    return cases;
}
int verify_projection_heat(const std::filesystem::path& root) {
    int cases=0;
    const std::array<std::array<std::string,3>,2> formats={
        std::array<std::string,3>{"nint5","nvq3j-512","nint5"},
        std::array<std::string,3>{"nint5","nint6","nvq3j-512"}};
    for(const std::string phase:{"disabled","none","upload","map","host","publish","late-reader"}) {
        CudaExecutionContext execution;execution.config.moe_residency_warm=true;
        execution.config.moe_residency_projection_heat=phase!="disabled";
        execution.config.moe_ram_pcie_fraction=1.0;
        std::array<std::array<std::shared_ptr<MoeQuantRangeSource>,3>,2> sources;
        std::array<std::array<std::shared_ptr<MfeWeight>,3>,2> weights;
        std::array<MoeFfnForward,2> forwards;
        std::array<MfeWeight,3> baseline;
        std::size_t gpu_bytes=0;
        for(int layer=0;layer<2;++layer)for(int p=0;p<3;++p) {
            const auto path=root/(formats[layer][p]+(p==2?"-2560-640.mfq":"-640-2560.mfq"));
            sources[layer][p]=source(path);gpu_bytes+=bytes(sources[layer][p]->metadata()->pools[0]);
            if(!layer) {
                auto model=mfq::open_model_source(path.string());
                baseline[p]=stage_cpu_mixed_moe(make_mixed_moe_runtime(load_mfe_cpu(*model,"linear.weight"),false),execution.config);
            }
        }
        execution.config.moe_host_cache_bytes=3*gpu_bytes;
        auto cache=make_moe_expert_cache(gpu_bytes,execution.config);
        for(int layer=0;layer<2;++layer) {
            for(int p=0;p<3;++p)weights[layer][p]=std::make_shared<MfeWeight>(cache_quant_moe_weight(cache,
                std::to_string(layer)+"/"+std::to_string(p),sources[layer][p],1,layer,std::to_string(p)));
            forwards[layer]=make_moe_ffn_pipeline({weights[layer][0],weights[layer][1]},{weights[layer][2]});
        }
        finalize_moe_expert_cache(cache);
        for(const auto& layer:sources)for(const auto& s:layer)
            if(!s->gpu_resident(0) || s->gpu_resident(1))throw std::runtime_error("projection-heat initial tier differs");
        const auto host_before=sources[0][0]->host_cache()->stats();
        std::vector<float> values(2560);for(int i=0;i<2560;++i)values[i]=std::sin(float(i)*.023f)*.3f;
        auto x=tb::tensor(values).reshape({1,2560}).to(tb::kFloat16).to(tb::kCUDA);
        auto ids=tb::ones({1,1},x.options().dtype(tb::kInt32)),route_weights=tb::ones({1,1},x.options().dtype(tb::kFloat32));
        tb::Tensor actual;mfq::cuda::DecodeWindow window(mfq_current_cuda_stream());
        window.capture([&]{actual=forwards[0](execution,x,ids,route_weights);},[]{},[&]{actual={};});
        const auto verify_gpu=[&] {
            auto route=build_moe_route_plan(ids,3);
            auto g=baseline[0].forward(execution,x,route),u=baseline[1].forward(execution,x,route);
            auto expected=baseline[2].forward(execution,(g*tb::sigmoid(g))*u,route).reshape({1,2560}).cpu().to(tb::kFloat32).contiguous();
            window.run();auto output=actual.cpu().to(tb::kFloat32).contiguous();
            if(std::memcmp(output.data_ptr(),expected.data_ptr(),output.numel()*sizeof(float)))
                throw std::runtime_error("mixed partial projection GPU graph changed canonical output bytes");
        };
        verify_gpu();
        MoeHostExpertCache::Lease reader;
        MoeResidencyManager manager(cache.get(),[&](const char* event,std::size_t index) {
            if(index)return;
            if(phase==event)throw std::runtime_error("injected projection exchange failure");
            if(phase=="late-reader" && std::string(event)=="upload")reader=sources[0][0]->acquire_expert(1);
        });
        std::vector<int32_t> heat(120,0);std::fill(heat.begin()+80,heat.end(),1);
        manager.after_layer(0,heat,120);manager.after_layer(1,{},120);
        bool rejected=false;
        try {manager.apply_pending();}catch(const std::runtime_error& e) {
            if(std::string(e.what())!="injected projection exchange failure")throw;rejected=true;
        }
        const auto stats=manager.stats();const bool changed=phase=="none";
        if((stats.committed_projections!=0)!=changed || stats.committed_bundles ||
            (phase=="disabled" ? stats.candidate_projections!=0 : stats.planned_projections!=1))
            throw std::runtime_error("projection heat did not select exactly one compatible cold projection");
        if((phase=="upload" || phase=="map" || phase=="host" || phase=="publish")!=rejected)
            throw std::runtime_error("projection heat failure was not propagated");
        if(phase=="late-reader" && !reader)throw std::runtime_error("projection heat late lease missing");
        for(int layer=0;layer<2;++layer)for(int p=0;p<3;++p) {
            if(sources[layer][p]->gpu_resident(0)!=(changed && layer==1 && p==0?false:true) ||
                sources[layer][p]->gpu_resident(1)!=(changed && layer==0 && p==0) ||
                sources[layer][p]->expert_disk_reads_after_preload())
                throw std::runtime_error("projection heat changed an unrelated projection or read SSD");
        }
        const int layer=changed?1:0,expert=changed?0:1;
        auto lease=sources[layer][0]->acquire_expert(expert);
        auto original=source(root/(formats[layer][0]+"-640-2560.mfq"));
        auto cpu=x.cpu().to(tb::kFloat32);
        auto a=linear(lease->weights,640,2560).forward(execution,cpu).to(tb::kFloat32).contiguous();
        auto b=linear(original->read_expert(expert),640,2560).forward(execution,cpu).to(tb::kFloat32).contiguous();
        if(std::memcmp(a.data_ptr(),b.data_ptr(),a.numel()*sizeof(float)))throw std::runtime_error("partial projection RAM transaction changed bytes");
        lease.reset();reader.reset();
        const auto host_after=sources[0][0]->host_cache()->stats();
        if(host_before.managed_bytes!=host_after.managed_bytes || host_before.managed_peak_bytes!=host_after.managed_peak_bytes)
            throw std::runtime_error("partial projection transaction duplicated RAM residency");
        for(int step=0;step<3;++step) {
            x.copy_(tb::tensor(values).reshape({1,2560}).to(tb::kFloat16).to(tb::kCUDA)*(1.0+step*.25));
            ids.copy_(tb::tensor(std::vector<int32_t>{step%2}).reshape({1,1}).to(tb::kCUDA));verify_gpu();
        }
        ++cases;
    }
    std::cout<<"projection_heat_cases="<<cases<<" incompatible GU/Down triples, one-projection cross-layer swap, RAM bytes, original GPU graph bytes and upload/map/host/publish/late-lease rollback exact PASS\n";
    return cases;
}
int verify_rounded_swiglu() {
    const auto context=mfq::cuda::default_context(mfq_current_cuda_device());
    const auto stream=mfq_current_cuda_stream();
    auto table=moe_swiglu_sigmoid_table_cuda();
    context->begin_graph_pool(stream);
    int cases=0;
    std::vector<std::vector<int64_t>> shapes;
    for(int tokens:{1,3,8,35})for(int routes:{1,10})for(int width:{640,641})
        shapes.push_back({tokens,routes,width});
    shapes.push_back({1,1,65536});shapes.push_back({1,2,32768});
    for(const auto& shape:shapes) {
        auto gate=tb::zeros(shape,table.options()),up=tb::zeros(shape,table.options());
        tb::Tensor result;mfq::cuda::DecodeWindow window(stream);
        window.capture([&]{result=moe_swiglu_rounded_cuda(gate,up,table);},[]{},[&]{result={};});
        for(int step=0;step<2;++step) {
            const bool exhaustive=gate.numel()==65536;
            auto cpu_gate=tb::empty(shape,tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat16));
            auto cpu_up=tb::empty(shape,cpu_gate.options());
            auto* g=reinterpret_cast<uint16_t*>(cpu_gate.data_ptr());
            auto* u=reinterpret_cast<uint16_t*>(cpu_up.data_ptr());
            const uint16_t special[8]={0,0x8000,0x3c00,0xbc00,0x3555,0x0400,0x7bff,0x7c00};
            for(int64_t i=0;i<gate.numel();++i) {
                if(exhaustive)g[i]=static_cast<uint16_t>(i);
                else {
                    const mfq_half value(std::sin(float(i+step*17)*.071f)*12.0f);
                    std::memcpy(g+i,&value,sizeof(value));
                    if(i%127==0)g[i]=special[(i/127+step)%8];
                }
                u[i]=step==0?0x3c00:special[(i+step)%8];
            }
            gate.copy_(cpu_gate.to(tb::kCUDA));up.copy_(cpu_up.to(tb::kCUDA));
            auto joined=tb::cat({gate,up},-1);
            auto old_gate=joined.narrow(-1,0,shape[2]),old_up=joined.narrow(-1,shape[2],shape[2]);
            auto expected=((old_gate*tb::sigmoid(old_gate))*old_up).cpu().contiguous();
            for(bool graph:{false,true}) {
                tb::Tensor output;
                if(graph){window.run();output=result;}
                else output=moe_swiglu_rounded_cuda(gate,up,table);
                auto actual=output.cpu().contiguous();
                if(actual.sizes()!=expected.sizes() || actual.scalar_type()!=expected.scalar_type() ||
                    std::memcmp(actual.data_ptr(),expected.data_ptr(),actual.numel()*actual.element_size()))
                    throw std::runtime_error("rounded SwiGLU changed concatenated sigmoid/multiply Half bits");
                ++cases;
            }
        }
        MFQ_CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    context->end_graph_pool(stream);
    std::cout<<"rounded_swiglu_cases="<<cases<<" all65536Half gate patterns, contiguous/strided sigmoid,1/3/8/35tokens,1/10routes,640/641width, signed-zero/inf/nan and changing ordinary/graph outputs exact PASS\n";
    return cases;
}
int verify_resident_classification() {
    CudaExecutionContext execution;
    constexpr int experts=512;
    const std::array<int,3> counts={3,2,4};
    const int maps=counts[0]+counts[1]+counts[2];
    const auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kInt32);
    auto kind=tb::empty({experts},options),cancelled=tb::empty({1},options);
    std::vector<std::array<tb::Tensor,6>> fields(maps);
    std::vector<int64_t> pointers;
    for(auto& map:fields)for(auto& field:map) {
        field=tb::full({experts},-1,options);
        pointers.push_back(reinterpret_cast<int64_t>(field.data_ptr()));
    }
    auto dispatch=tb::tensor(pointers).to(tb::kCUDA);
    const auto stream=mfq_current_cuda_stream();
    const auto run=[&] {
        mfq::cuda::moe_classify_resident(reinterpret_cast<const uint64_t*>(dispatch.data_ptr<int64_t>()),
            counts[0],counts[1],counts[2],experts,kind.data_ptr<int32_t>(),
            reinterpret_cast<uint32_t*>(cancelled.data_ptr<int32_t>()),stream);
        mfq::cuda::moe_update_dispatch_maps(reinterpret_cast<const uint64_t*>(dispatch.data_ptr<int64_t>()),
            maps,experts,kind.data_ptr<int32_t>(),reinterpret_cast<const uint32_t*>(cancelled.data_ptr<int32_t>()),stream);
        MFQ_CUDA_CHECK(cudaGetLastError());
    };
    auto context=mfq::cuda::default_context(mfq_current_cuda_device());
    context->begin_graph_pool(stream);
    mfq::cuda::DecodeWindow window(stream);window.capture(run,[]{},[]{});
    int cases=0;
    for(int step=0;step<8;++step) {
        std::vector<std::vector<int32_t>> slots(maps,std::vector<int32_t>(experts,-1));
        int offset=0;
        for(int projection=0;projection<3;++projection) {
            for(int e=0;e<experts;++e) {
                const bool resident=step==0 || (step!=1 && (e+step+projection)%5!=0);
                if(resident)slots[offset+(e+step)%counts[projection]][e]=(e+step)%17;
            }
            offset+=counts[projection];
        }
        for(int map=0;map<maps;++map)fields[map][0].copy_(tb::tensor(slots[map]).to(tb::kCUDA));
        std::vector<int32_t> expected(experts,1);offset=0;
        for(int projection=0;projection<3;++projection) {
            for(int e=0;e<experts;++e) {
                bool present=false;
                for(int map=offset;map<offset+counts[projection];++map)present|=slots[map][e]>=0;
                expected[e]&=present;
            }
            offset+=counts[projection];
        }
        for(bool graph:{false,true}) {
            cancelled.fill_(1);if(graph)window.run();else run();
            auto actual=kind.cpu().contiguous();
            if(std::memcmp(actual.data_ptr(),expected.data(),experts*sizeof(int32_t)) ||
                cancelled.cpu().data_ptr<int32_t>()[0]!=0)
                throw std::runtime_error("GPU resident classification differs from independent projection intersection");
            for(int map=0;map<maps;++map)for(int field=3;field<6;++field) {
                auto output=fields[map][field].cpu().contiguous();
                for(int e=0;e<experts;++e)if(output.data_ptr<int32_t>()[e]!=
                    (field==3 && expected[e]?slots[map][e]:-1))
                    throw std::runtime_error("GPU resident dispatch read or enabled a cold projection");
            }
            ++cases;
        }
    }
    context->end_graph_pool(stream);
    std::cout<<"resident_classification_cases="<<cases<<" 512experts,3/2/4format cohorts, all/empty/partial/changing projection ownership, ordinary and graph CPU oracle exact PASS\n";
    return cases;
}
int verify_active_nvq(const std::filesystem::path& root) {
    CudaExecutionContext execution;
    const auto same=[](const tb::Tensor& a,const tb::Tensor& b) {
        auto x=a.to(tb::kCPU).contiguous(),y=b.to(tb::kCPU).contiguous();
        if(x.sizes()!=y.sizes() || x.scalar_type()!=y.scalar_type() ||
            std::memcmp(x.data_ptr(),y.data_ptr(),x.numel()*x.element_size()))
            throw std::runtime_error("active NVQ differs from original heterogeneous kernel bits");
    };
    const auto stream=mfq_current_cuda_stream();
    auto context=mfq::cuda::default_context(mfq_current_cuda_device());
    context->begin_graph_pool(stream);int cases=0;
    const std::array<std::string,8> formats={"nvq1-s","nvq1-l","nvq2j","nvq2j-l",
        "nvq2j-xl","nvq3j","nvq3j-512","nvq3j-l"};
    for(const auto shape:{std::pair<int,int>{640,2560},{2560,640}})for(bool routed:{false,true}) {
        MixedMoeRuntime actual;actual.n_experts=512;actual.out_per_expert=shape.first;
        actual.neuron_len=shape.second;actual.partial_experts=true;
        int group=0;
        for(const auto& format:formats) {
            auto model=mfq::open_model_source((root/(format+"-"+std::to_string(shape.first)+"-"+
                std::to_string(shape.second)+".mfq")).string());
            auto single=make_mixed_moe_runtime(load_mfe_cpu(*model,"linear.weight"),true,execution.config);
            for(auto& pool:single->pools) {
                std::vector<int32_t> slots(512,-1);
                for(int e=0;e<3;++e)slots[group*3+e]=e;
                pool.expert_local=tb::tensor(slots).to(tb::kCUDA);actual.pools.push_back(std::move(pool));
            }
            ++group;
        }
        initialize_mixed_nvq_dispatch(actual,execution.config);auto original=actual;
        for(int tokens:{1,3,8}) {
            auto input=tb::zeros(routed?std::vector<int64_t>{tokens,10,shape.second}:
                std::vector<int64_t>{tokens,shape.second},tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat16));
            auto ids=tb::zeros({tokens,10},input.options().dtype(tb::kInt32));
            const auto forward=[&](MixedMoeRuntime& runtime) {
                MoeRoutePlan route;route.ids=ids;route.n_experts=512;
                return runtime.forward(execution.config,execution.kl_mmq,true,false,input,route);
            };
            tb::Tensor result;mfq::cuda::DecodeWindow window(stream);
            execution.config.moe_nvq_active_decode=true;
            window.capture([&]{result=forward(actual);},[]{},[&]{result={};});
            for(int step=0;step<4;++step) {
                std::vector<float> values(input.numel());
                for(size_t i=0;i<values.size();++i)values[i]=std::sin(float(i+step*19)*.071f)*.13f;
                input.copy_(tb::tensor(values).reshape(input.sizes().vec()).to(tb::kFloat16).to(tb::kCUDA));
                std::vector<int32_t> selected(tokens*10);
                const int pattern[10]={0,3,6,9,12,15,18,21,0,511};
                for(int i=0;i<tokens*10;++i) {
                    int expert=pattern[(i+step)%10];
                    if(step==1)expert=511;
                    if(step==2 && i%10==7)expert=-1;
                    if(step==2 && i%10==8)expert=512;
                    if(step==3 && expert<24)++expert;
                    selected[i]=expert;
                }
                ids.copy_(tb::tensor(selected).reshape({tokens,10}).to(tb::kCUDA));
                for(size_t p=0;p<actual.pools.size();++p) {
                    std::vector<int32_t> slots(512,-1);
                    for(int e=0;e<3;++e)if(step!=3 || e!=0)slots[int(p)*3+e]=e;
                    actual.pools[p].expert_local.copy_(tb::tensor(slots).to(tb::kCUDA));
                }
                execution.config.moe_nvq_active_decode=false;auto expected=forward(original);
                execution.config.moe_nvq_active_decode=true;window.run();same(result,expected);
                auto plan=actual.nvq_active_plans.at(tokens*10).cpu().contiguous();
                const auto* data=plan.data_ptr<int32_t>();int count=0;
                for(int pair=0;pair<tokens*10;++pair) {
                    const int expert=selected[pair];
                    const bool valid=expert>=0 && expert<int(formats.size())*3 && (step!=3 || expert%3!=0);
                    if(valid) {
                        if(data[1+2*tokens*10+count]!=pair || data[1+pair]!=expert/3 ||
                                data[1+tokens*10+pair]!=expert%3)
                            throw std::runtime_error("active NVQ plan differs from CPU route/slot oracle");
                        ++count;
                    }
                }
                if(data[0]!=count)throw std::runtime_error("active NVQ plan count differs from CPU oracle");
                ++cases;
            }
            MFQ_CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    }
    context->end_graph_pool(stream);
    std::cout<<"active_nvq_cases="<<cases<<" active_nvq_formats="<<formats.size()<<" actual640/2560, shared/routed input,1/3/8tokens, repeated/invalid/empty routes, changing ownership and original heterogeneous kernel bytes exact PASS\n";
    return cases;
}
int verify_two_stage_shared(const std::filesystem::path& root,bool transfer_cache=false,bool comparison=false) {
    struct RestoreEnvironment {
        std::string previous;
        bool present=false;
        static void set(const char* value) {
#ifdef _WIN32
            _putenv_s("MFQ_MOE_TRANSFER_CACHE",value);
#else
            if(*value)setenv("MFQ_MOE_TRANSFER_CACHE",value,1);else unsetenv("MFQ_MOE_TRANSFER_CACHE");
#endif
        }
        explicit RestoreEnvironment(bool enabled) {
            if(const auto* value=std::getenv("MFQ_MOE_TRANSFER_CACHE")){previous=value;present=true;}
            set(enabled?"1":"0");
        }
        ~RestoreEnvironment(){set(present?previous.c_str():"");}
    } environment(transfer_cache);
    const std::array<std::string,3> formats={"nint5","nvq3j-l","nvq2j-xl"};
    std::array<std::shared_ptr<const QuantLinear>,3> shared;
    for(int p=0;p<3;++p) {
        const int output=p==2?2560:640,width=p==2?640:2560;
        const std::array<const char*,3> roles={"gate","up","down"};
        std::ifstream stream(root/(std::string("shared-s2-")+roles[p]+".nint"),std::ios::binary);
        if(!stream)throw std::runtime_error("missing released shared FFN tensor fixture");
        const std::vector<uint8_t> blob((std::istreambuf_iterator<char>(stream)),std::istreambuf_iterator<char>());
        const auto cpu=unpack_nint(blob);
        if(cpu.gs!=24 || cpu.bits!=6 || cpu.neuron_len!=width || cpu.out!=output)
            throw std::runtime_error("released shared FFN fixture geometry mismatch");
        QuantLinear weight;weight.nint=to_gpu_nint(cpu);weight.logical_out=output;weight.logical_neuron_len=width;
        shared[p]=std::make_shared<QuantLinear>(std::move(weight));
    }
    const auto counter=[](const std::shared_ptr<MoeExpertCache>& cache,const char* name) {
        std::ostringstream stream;print_moe_expert_cache_stats(cache,stream);
        std::smatch match;const auto text=stream.str();
        if(!std::regex_search(text,match,std::regex(std::string(name)+"=([0-9]+)")))
            throw std::runtime_error("missing two-stage integration counter");
        return std::stoll(match[1]);
    };
    MFQ_CUDA_CHECK(cudaStreamSynchronize(mfq_current_cuda_stream()));
    const std::array<MfqCudaStream,2> streams={mfq_get_stream_from_pool(),mfq_get_current_cuda_stream()};
    int cases=0;std::int64_t cpu=0,transfers=0,resident=0,mapped_serves=0,mapped_bytes=0,mapped_overlaps=0,phased_serves=0;
    for(bool graph:{false,true})for(auto gate_type:{tb::kFloat16,tb::kBFloat16,tb::kFloat32})
    for(int tokens:{1,3,6})for(int routes:{3,10}) {
        std::array<CudaExecutionContext,2> executions;
        std::array<std::shared_ptr<MoeExpertCache>,2> caches;
        std::array<std::array<std::shared_ptr<MoeQuantRangeSource>,3>,2> sources;
        std::array<MoeFfnForward,2> forwards;
        std::array<mfq::MoeDispatchPlan,2> plans;
        std::vector<std::shared_ptr<MfeWeight>> comparison_gu,comparison_down;
        auto shared_gate=[gate_type](CudaExecutionContext&,const tb::Tensor& x) {
            return tb::sigmoid(x.mean(1).unsqueeze(-1).to(gate_type));
        };
        auto original_shared=[shared,shared_gate](CudaExecutionContext& execution,const tb::Tensor& x) {
            auto g=shared[0]->forward(execution,x),u=shared[1]->forward(execution,x);
            return shared_gate(execution,x)*shared[2]->forward(execution,(g*tb::sigmoid(g))*u);
        };
        for(int mode=0;mode<2;++mode) {
            MfqCudaStreamGuard stream_guard(streams[mode]);
            auto& config=executions[mode].config;
            config.moe_two_stage_ffn=mode==1;
            config.moe_ram_pcie_fraction=transfer_cache && !comparison?1.0:.5;
            if(comparison)config.moe_residency_adapt=false;
            if(transfer_cache)config.moe_direct_ram=true;
            config.moe_preload_all=true;config.moe_assert_resident=true;
            config.moe_pipeline=true;config.moe_ram_pcie=true;
            config.moe_residency_projection_heat=false;
            std::size_t bundle=0,extra=0;
            for(int p=0;p<3;++p) {
                sources[mode][p]=source(root/(formats[p]+(p==2?"-2560-640.mfq":"-640-2560.mfq")));
                const auto size=bytes(sources[mode][p]->metadata()->pools[0]);
                bundle+=size;if(!p)extra=size;
            }
            config.moe_host_cache_bytes=3*bundle;
            caches[mode]=make_moe_expert_cache(bundle+extra,config);
            std::vector<std::shared_ptr<MfeWeight>> gu,down;
            for(int p=0;p<3;++p) {
                auto weight=std::make_shared<MfeWeight>(cache_quant_moe_weight(caches[mode],
                    std::to_string(p),sources[mode][p],1,0,std::to_string(p)));
                (p==2?down:gu).push_back(weight);
            }
            forwards[mode]=make_moe_ffn_pipeline(gu,down,original_shared,
                [&,mode](const mfq::MoeDispatchPlan& plan){plans[mode]=plan;},
                MoeFfnSharedWeights{shared,shared_gate});
            if(comparison && mode==1){comparison_gu=gu;comparison_down=down;}
            finalize_moe_expert_cache(caches[mode]);
            bool partial=false;
            for(int e=0;e<3;++e) {
                int hits=0;for(const auto& s:sources[mode])hits+=s->gpu_resident(e);
                partial=partial || (hits>0 && hits<3);
            }
            if(!partial)throw std::runtime_error("weighted shared fixture lacks partially resident projection");
        }
        auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat16);
        auto input=tb::zeros({tokens,2560},options);
        auto ids=tb::zeros({tokens,routes},options.dtype(tb::kInt32));
        auto weights=tb::zeros({tokens,routes},options.dtype(tb::kFloat32));
        std::int64_t comparison_dma=-1;
        for(int pass=0;pass<(comparison?3:1);++pass) {
        if(comparison) {
            executions[1].config.moe_ffn_transfer_phases=pass==1;
            if(pass) {
                forwards[1]={};
                forwards[1]=make_moe_ffn_pipeline(comparison_gu,comparison_down,original_shared,
                    [&](const mfq::MoeDispatchPlan& plan){plans[1]=plan;},
                    MoeFfnSharedWeights{shared,shared_gate});
            }
            if(!pass) {
                bool rejected=false;
                try {prepare_moe_pipeline_comparison(caches[1],true);}
                catch(const std::logic_error&){rejected=true;}
                if(!rejected)throw std::runtime_error("empty comparison replay was accepted");
            }
            prepare_moe_pipeline_comparison(caches[1],pass>0);
        }
        const auto dma_before=comparison?counter(caches[1],"ram_pcie_bytes"):0;
        tb::Tensor actual;mfq::cuda::DecodeWindow window(mfq_current_cuda_stream());
        if(graph)window.capture([&]{actual=forwards[1](executions[1],input,ids,weights);},[]{},[&]{actual={};});
        for(int step=0;step<4;++step) {
            for(const auto& cache:caches)finish_moe_expert_exchanges(cache);
            std::vector<float> values(tokens*2560),route_weights(tokens*routes);
            std::vector<int32_t> selected(tokens*routes);
            for(std::size_t i=0;i<values.size();++i)values[i]=std::sin(float(i+step*31)*.017f)*.13f;
            for(int i=0;i<tokens*routes;++i) {
                selected[i]=(i+step)%3;
                route_weights[i]=float((i*7+step*3)%17+1)/37;
            }
            input.copy_(tb::tensor(values).reshape({tokens,2560}).to(tb::kFloat16).to(tb::kCUDA));
            ids.copy_(tb::tensor(selected).reshape({tokens,routes}).to(tb::kCUDA));
            weights.copy_(tb::tensor(route_weights).reshape({tokens,routes}).to(tb::kCUDA));
            MFQ_CUDA_CHECK(cudaStreamSynchronize(streams[1]));
            tb::Tensor expected;
            {MfqCudaStreamGuard stream_guard(streams[0]);
                expected=forwards[0](executions[0],input,ids,weights).cpu().contiguous();}
            if(graph)window.run();else actual=forwards[1](executions[1],input,ids,weights);
            auto result=actual.cpu().contiguous();
            if(plans[0].kinds!=plans[1].kinds)throw std::runtime_error("two-stage fusion changed CPU/GPU dispatch");
            for(const auto kind:plans[1].kinds) {
                cpu+=kind==mfq::MoeDispatchKind::Cpu;
                transfers+=kind==mfq::MoeDispatchKind::GpuTransfer;
                resident+=kind==mfq::MoeDispatchKind::GpuResident;
            }
            if(result.sizes()!=expected.sizes() || result.scalar_type()!=expected.scalar_type() ||
                std::memcmp(result.data_ptr(),expected.data_ptr(),result.numel()*result.element_size()))
                throw std::runtime_error("weighted shared two-stage output is not byte exact, tokens="+
                    std::to_string(tokens)+" routes="+std::to_string(routes)+" step="+std::to_string(step)+
                    " graph="+std::to_string(graph)+" gate_dtype="+std::to_string(int(gate_type)));
            for(const auto& mode:sources)for(const auto& s:mode)
                if(s->expert_disk_reads_after_preload())throw std::runtime_error("weighted shared fusion read expert SSD after preload");
            ++cases;
        }
        if(comparison) {
            if(finish_moe_pipeline_comparison(caches[1])!=4)
                throw std::runtime_error("comparison did not retain all original CPU assignments");
            const auto dma=counter(caches[1],"ram_pcie_bytes")-dma_before;
            if(comparison_dma<0)comparison_dma=dma;
            if(dma!=comparison_dma || !dma || counter(caches[1],"residency_committed_projections"))
                throw std::runtime_error("comparison changed weight transfers or primary ownership");
        }
        }
        if(counter(caches[0],"pipeline_two_stage_serves") || counter(caches[1],"pipeline_two_stage_serves")!=(comparison?12:4))
            throw std::runtime_error("weighted shared test did not compare old and two-stage serving paths");
        if(transfer_cache && (!counter(caches[1],"pipeline_transfer_cache_hits") ||
            !counter(caches[1],"pipeline_transfer_cache_misses") || !counter(caches[1],"pipeline_transfer_cache_saved_bytes")))
            throw std::runtime_error("retained transfer fields were not reused by changing shared FFN graphs");
        if(transfer_cache) {
            const auto* phases=std::getenv("MFQ_MOE_FFN_TRANSFER_PHASES");
            const auto* mapped=std::getenv("MFQ_MOE_MAPPED_COPY");
            const bool enabled_phase=(!phases || std::string(phases)=="1") && !(mapped && std::string(mapped)=="1");
            const auto actual_phases=counter(caches[1],"pipeline_phased_transfer_serves");
            if(actual_phases!=(comparison?4:enabled_phase?4:0) || counter(caches[0],"pipeline_phased_transfer_serves"))
                throw std::runtime_error("phased-transfer native arm was not exercised");
            if((enabled_phase || comparison) && (!counter(caches[1],"pipeline_gate_up_dma_bytes") || !counter(caches[1],"pipeline_down_dma_bytes")))
                throw std::runtime_error("phased-transfer test omitted an original projection payload");
            phased_serves+=actual_phases;
            const auto* enabled=std::getenv("MFQ_MOE_MAPPED_COPY");
            if(enabled && std::string(enabled)=="1") {
                const auto serves=counter(caches[1],"pipeline_mapped_copy_serves");
                const auto bytes=counter(caches[1],"pipeline_mapped_copy_bytes");
                if(serves!=4 || !bytes || counter(caches[0],"pipeline_mapped_copy_serves"))
                    throw std::runtime_error("shared graph did not compare DMA with mapped RAM transfer");
                const auto* overlap=std::getenv("MFQ_MOE_MAPPED_COPY_OVERLAP");
                const auto overlap_serves=counter(caches[1],"pipeline_mapped_overlap_serves");
                if(overlap_serves!=(overlap && std::string(overlap)=="1"?4:0))
                    throw std::runtime_error("mapped-copy overlap arm was not exercised");
                mapped_overlaps+=overlap_serves;
                mapped_serves+=serves;mapped_bytes+=bytes;
            }
        }
    }
    if(((!transfer_cache || comparison) && !cpu) || !transfers || !resident)throw std::runtime_error("weighted shared fusion omitted a dispatch tier");
    std::cout<<(transfer_cache?"transfer_cache_shared_cases=":"two_stage_shared_cases=")<<cases<<" cpu_positions="<<cpu<<" transfer_positions="<<transfers
        <<" resident_positions="<<resident<<" mixed G/U/Down, released shared adaptive NINT6 gs24/sub7, Float/Half/BFloat gate, changing weighted graphs and partial VRAM/RAM exact PASS\n";
    if(comparison)std::cout<<"transfer_comparison_shared_cases="<<cases<<" passes=108 replay_guards=36 equal_dma=1 fixed_primary=1 PASS\n";
    if(phased_serves)std::cout<<"phased_transfer_shared_cases="<<phased_serves<<" PASS\n";
    if(mapped_serves)std::cout<<"mapped_copy_shared_cases="<<mapped_serves<<" mapped_bytes="<<mapped_bytes<<" PASS\n";
    if(mapped_overlaps)std::cout<<"mapped_overlap_shared_cases="<<mapped_overlaps<<" PASS\n";
    return cases;
}
#include "../../../../bench/moe_residency_copy_bench.h"
int verify_transfer_hit_reservation(const std::filesystem::path& root) {
    const std::array<MfqCudaStream,2> streams={mfq_get_stream_from_pool(),mfq_get_current_cuda_stream()};
    const auto counter=[](const std::shared_ptr<MoeExpertCache>& cache,const char* name) {
        std::ostringstream stream;print_moe_expert_cache_stats(cache,stream);const auto text=stream.str();
        std::smatch match;
        if(!std::regex_search(text,match,std::regex(std::string(name)+"=([0-9]+)")))
            throw std::runtime_error("missing transfer reservation counter");
        return std::stoll(match[1]);
    };
    struct Environment {
        std::string old;
        Environment(){const auto* p=std::getenv("MFQ_MOE_TRANSFER_CACHE");if(p)old=p;}
        static void set(const char* p) {
#if defined(_WIN32)
            _putenv_s("MFQ_MOE_TRANSFER_CACHE",p);
#else
            if(*p)setenv("MFQ_MOE_TRANSFER_CACHE",p,1);else unsetenv("MFQ_MOE_TRANSFER_CACHE");
#endif
        }
        ~Environment(){set(old.c_str());}
    } environment;
    int cases=0;
    for(const std::string format:{"nint5","nvq3j-512"}) {
        std::array<std::shared_ptr<MoeExpertCache>,2> caches;
        std::array<std::array<MoeFfnForward,2>,2> forwards;
        std::array<CudaExecutionContext,2> executions;
        for(int mode=0;mode<2;++mode) {
            MfqCudaStreamGuard guard(streams[mode]);
            Environment::set(mode?"1":"0");auto& execution=executions[mode];
            execution.config.moe_hybrid_cpu=false;execution.config.moe_direct_ram=true;execution.config.moe_two_stage_ffn=true;
            execution.config.moe_ram_pcie_fraction=1.0;
            auto proto=source(root/(format+"-640-2560.mfq"));
            auto down=source(root/(format+"-2560-640.mfq"));
            const auto bundle_bytes=2*bytes(proto->metadata()->pools[0])+bytes(down->metadata()->pools[0]);
            execution.config.moe_host_cache_bytes=6*bundle_bytes;
            caches[mode]=make_moe_expert_cache(2*bundle_bytes,execution.config);
            for(int layer=0;layer<2;++layer) {
                std::vector<std::shared_ptr<MfeWeight>> gates,downs;
                for(int p=0;p<3;++p) {
                    auto s=source(root/(format+(p==2?"-2560-640.mfq":"-640-2560.mfq")));
                    auto weight=std::make_shared<MfeWeight>(cache_quant_moe_weight(caches[mode],
                        std::to_string(layer)+"/"+std::to_string(p),s,1,layer,std::to_string(p)));
                    (p==2?downs:gates).push_back(weight);
                }
                forwards[mode][layer]=make_moe_ffn_pipeline(gates,downs);
            }
            finalize_moe_expert_cache(caches[mode]);
        }
        std::vector<float> values(2560);for(int i=0;i<2560;++i)values[i]=std::sin(float(i)*.019f)*.1f;
        auto x=tb::tensor(values).reshape({1,2560}).to(tb::kFloat16).to(tb::kCUDA);
        const std::array<std::vector<int32_t>,3> routes={std::vector<int32_t>{2},{1,2},{1,2}};
        for(int step=0;step<3;++step) {
            const auto before=counter(caches[1],"pipeline_transfer_cache_hits");
            auto ids=tb::tensor(routes[step]).reshape({1,int64_t(routes[step].size())}).to(tb::kCUDA);
            auto weights=tb::ones(ids.sizes().vec(),x.options().dtype(tb::kFloat32));
            const int layer=step==1?1:0;
            const auto run=[&](int mode){MfqCudaStreamGuard guard(streams[mode]);
                return forwards[mode][layer](executions[mode],x,ids,weights).cpu().contiguous();};
            auto expected=run(0),actual=run(1);
            if(actual.sizes()!=expected.sizes() || actual.scalar_type()!=expected.scalar_type() ||
                    std::memcmp(actual.data_ptr(),expected.data_ptr(),actual.numel()*actual.element_size()))
                throw std::runtime_error("reserved transfer hits changed GPU output bytes");
            if(step==2 && counter(caches[1],"pipeline_transfer_cache_hits")-before!=3)
                throw std::runtime_error("an earlier transfer miss evicted a later route hit, actual_hits="+
                    std::to_string(counter(caches[1],"pipeline_transfer_cache_hits")-before)+
                    " two_stage="+std::to_string(counter(caches[1],"pipeline_two_stage_serves")));
            ++cases;
        }
    }
    std::cout<<"transfer_hit_reservation_cases="<<cases<<" two sources sharing full staging arenas, miss-before-hit and original output bytes exact PASS\n";
    return cases;
}
void verify_router_prefetch(const std::filesystem::path& root) {
    struct Environment {
        std::string old;
        Environment() {if(auto* p=std::getenv("MFQ_MOE_TRANSFER_CACHE"))old=p;set("1");}
        static void set(const char* p) {
#ifdef _WIN32
            _putenv_s("MFQ_MOE_TRANSFER_CACHE",p);
#else
            if(*p)setenv("MFQ_MOE_TRANSFER_CACHE",p,1);else unsetenv("MFQ_MOE_TRANSFER_CACHE");
#endif
        }
        ~Environment(){set(old.c_str());}
    } environment;
    const auto counter=[](const std::shared_ptr<MoeExpertCache>& cache,const char* name) {
        std::ostringstream out;print_moe_expert_cache_stats(cache,out);
        const auto text=out.str();std::smatch match;
        if(!std::regex_search(text,match,std::regex(std::string(name)+"=([0-9]+)")))
            throw std::runtime_error("missing router prefetch counter");
        return std::stoll(match[1]);
    };
    int cases=0,guards=0,recovered=0;int64_t transferred=0,hits=0,busy=0;
    for(const auto& formats:std::array<std::array<std::string,3>,3>{
        std::array<std::string,3>{"nint5","nint5","nint5"},
        {"nvq3j-512","nvq3j-512","nvq3j-512"},
        {"nint5","nvq3j-l","nvq2j-xl"}}) {
        std::array<CudaExecutionContext,2> executions;
        std::array<std::shared_ptr<MoeExpertCache>,2> caches;
        std::array<std::array<MoeFfnForward,2>,2> forward;
        std::array<std::array<std::array<std::shared_ptr<MoeQuantRangeSource>,3>,2>,2> sources;
        std::array<MfqCudaStream,2> streams={mfq_get_stream_from_pool(),mfq_get_current_cuda_stream()};
        MoeFfnPrefetch prefetch;
        for(int mode=0;mode<2;++mode) {
            MfqCudaStreamGuard guard(streams[mode]);auto& cfg=executions[mode].config;
            cfg.moe_preload_all=true;cfg.moe_assert_resident=true;cfg.moe_pipeline=true;
            cfg.moe_hybrid_cpu=false;cfg.moe_direct_ram=true;cfg.moe_two_stage_ffn=true;
            cfg.moe_ram_pcie=true;cfg.moe_ram_pcie_fraction=1.;cfg.moe_residency_adapt=false;
            int64_t bundle=0;
            for(int p=0;p<3;++p) {
                auto proto=source(root/(formats[p]+(p==2?"-2560-640.mfq":"-640-2560.mfq")));
                bundle+=bytes(proto->metadata()->pools[0]);
            }
            cfg.moe_host_cache_bytes=6*bundle;caches[mode]=make_moe_expert_cache(2*bundle,cfg);
            for(int layer=0;layer<2;++layer) {
                std::vector<std::shared_ptr<MfeWeight>> gu,down;
                for(int p=0;p<3;++p) {
                    auto& src=sources[mode][layer][p];
                    src=source(root/(formats[p]+(p==2?"-2560-640.mfq":"-640-2560.mfq")));
                    auto weight=std::make_shared<MfeWeight>(cache_quant_moe_weight(caches[mode],
                        "prefetch/"+std::to_string(layer)+"/"+std::to_string(p),src,1,layer,std::to_string(p)));
                    (p==2?down:gu).push_back(weight);
                }
                forward[mode][layer]=make_moe_ffn_pipeline(gu,down,{},{},{},mode==1 && layer==1?&prefetch:nullptr);
            }
            finalize_moe_expert_cache(caches[mode]);
        }
        auto opt=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat16);
        auto input=tb::zeros({1,2560},opt),first_ids=tb::zeros({1,3},opt.dtype(tb::kInt32)),
            next_ids=tb::zeros({1,2},opt.dtype(tb::kInt32)),predicted=tb::zeros({1,2},opt.dtype(tb::kInt32));
        auto first_weights=tb::ones({1,3},opt.dtype(tb::kFloat32))*.17,
            next_weights=tb::ones({1,2},opt.dtype(tb::kFloat32))*.23;
        std::array<tb::Tensor,2> output;
        std::array<std::unique_ptr<mfq::cuda::DecodeWindow>,2> windows;
        for(int mode=0;mode<2;++mode) {
            MfqCudaStreamGuard guard(streams[mode]);windows[mode]=std::make_unique<mfq::cuda::DecodeWindow>(streams[mode].stream());
            windows[mode]->capture([&,mode] {
                mfq::cuda::DecodeWindow::Task prediction{};
                if(mode)prediction=prefetch(predicted);
                auto first=forward[mode][0](executions[mode],input,first_ids,first_weights);
                if(mode)windows[mode]->enroll(std::move(prediction));
                auto hidden=(tb::tanh(first.mean(1))*.0001).to(tb::kFloat16);
                output[mode]=forward[mode][1](executions[mode],hidden,next_ids,next_weights);
            },[]{},[&,mode]{output[mode]={};});
        }
        const std::array<std::array<int32_t,3>,6> first={{{0,0,0},{1,1,1},{2,2,2},{1,2,1},{0,0,0},{0,1,2}}};
        const std::array<std::array<int32_t,2>,6> next={{{2,2},{2,2},{1,1},{1,2},{2,2},{1,2}}};
        const std::array<std::array<int32_t,2>,6> guesses={{{2,2},{1,1},{0,2},{1,2},{2,2},{1,2}}};
        for(int step=0;step<18;++step) {
            std::vector<float> values(2560);for(int i=0;i<2560;++i)values[i]=std::sin(float(i+step*23)*.017)*.13;
            input.copy_(tb::tensor(values).reshape({1,2560}).to(tb::kFloat16).to(tb::kCUDA));
            const int s=step%6;
            first_ids.copy_(tb::tensor(std::vector<int32_t>(first[s].begin(),first[s].end())).reshape({1,3}).to(tb::kCUDA));
            next_ids.copy_(tb::tensor(std::vector<int32_t>(next[s].begin(),next[s].end())).reshape({1,2}).to(tb::kCUDA));
            predicted.copy_(tb::tensor(std::vector<int32_t>(guesses[s].begin(),guesses[s].end())).reshape({1,2}).to(tb::kCUDA));
            MFQ_CUDA_CHECK(cudaStreamSynchronize(mfq_current_cuda_stream()));
            std::array<tb::Tensor,2> host;
            for(int mode=0;mode<2;++mode) {MfqCudaStreamGuard guard(streams[mode]);windows[mode]->run();host[mode]=output[mode].cpu().contiguous();}
            if(host[0].sizes()!=host[1].sizes() || host[0].scalar_type()!=host[1].scalar_type() ||
                std::memcmp(host[0].data_ptr(),host[1].data_ptr(),host[0].numel()*host[0].element_size()))
                throw std::runtime_error("router prefetch changed original output bytes at step "+std::to_string(step));
            for(const auto& mode:sources)for(const auto& layer:mode)for(const auto& src:layer)
                if(src->expert_disk_reads_after_preload())throw std::runtime_error("router prefetch read expert SSD");
            ++cases;
        }
        transferred+=counter(caches[1],"pipeline_prefetch_bytes");
        hits+=counter(caches[1],"pipeline_prefetch_hit_bytes");busy+=counter(caches[1],"pipeline_prefetch_busy_skips");
        if(counter(caches[0],"pipeline_prefetch_bytes") || counter(caches[1],"hybrid_cpu_experts"))
            throw std::runtime_error("router prefetch native mode changed its serving policy");
        bool rejected=false;
        try{prefetch(predicted);}catch(const std::invalid_argument&){rejected=true;++guards;}
        if(!rejected)throw std::runtime_error("router prefetch accepted an unrecorded call");
        // Fail after reserving one cold expert, then reuse the same cache in a
        // fresh window. The incomplete speculative entry must be discarded.
        prepare_moe_pipeline_comparison(caches[1],false);
        first_ids.copy_(tb::zeros({1,3},first_ids.options()));
        next_ids.copy_(tb::tensor(std::vector<int32_t>{2,2}).reshape({1,2}).to(tb::kCUDA));
        predicted.copy_(tb::tensor(std::vector<int32_t>{2,3}).reshape({1,2}).to(tb::kCUDA));
        rejected=false;
        try{windows[1]->run();}catch(const std::runtime_error& e) {
            rejected=std::string(e.what())=="MFQ prefetch expert outside router geometry";if(rejected)++guards;
        }
        if(!rejected)throw std::runtime_error("router prefetch failed to reject an invalid expert");
        rejected=false;
        try{windows[1]->run();}catch(const std::runtime_error&){rejected=true;++guards;}
        if(!rejected)throw std::runtime_error("router prefetch reused its failed window");
        predicted.copy_(next_ids);MFQ_CUDA_CHECK(cudaStreamSynchronize(mfq_current_cuda_stream()));
        {MfqCudaStreamGuard guard(streams[0]);windows[0]->run();}
        mfq::cuda::DecodeWindow fresh(streams[1].stream());tb::Tensor recovery;
        fresh.capture([&] {
            auto prediction=prefetch(predicted);
            auto first=forward[1][0](executions[1],input,first_ids,first_weights);
            fresh.enroll(std::move(prediction));
            recovery=forward[1][1](executions[1],(tb::tanh(first.mean(1))*.0001).to(tb::kFloat16),next_ids,next_weights);
        },[]{},[&]{recovery={};});
        fresh.run();auto expected=output[0].cpu().contiguous(),actual=recovery.cpu().contiguous();
        if(expected.sizes()!=actual.sizes() || std::memcmp(expected.data_ptr(),actual.data_ptr(),actual.numel()*actual.element_size()))
            throw std::runtime_error("router prefetch retained incomplete fields after failure");
        ++recovered;
    }
    if(!transferred || !hits || !busy)throw std::runtime_error("router prefetch omitted speculative DMA, actual hits or busy-slot protection");
    std::cout<<"router_prefetch_exact_cases="<<cases<<" bytes="<<transferred<<" hit_bytes="<<hits
        <<" busy_skips="<<busy<<" expert_SSD=0 PASS\n";
    if(guards!=9 || recovered!=3)throw std::runtime_error("router prefetch failure checks incomplete");
    std::cout<<"router_prefetch_failure_guards="<<guards<<" recovered_exact_cases="<<recovered<<" PASS\n";
}
void verify_router_lookahead_inputs() {
    auto context=mfq::cuda::default_context(mfq_current_cuda_device());
    const auto stream=mfq_current_cuda_stream();context->begin_graph_pool(stream);
    struct Pool {
        std::shared_ptr<mfq::cuda::Context> context;cudaStream_t stream;
        ~Pool(){context->end_graph_pool(stream);}
    } pool{context,stream};
    mfq::cuda::RouterLookaheadInputs audit(3,7,3);
    auto input=tb::zeros({3,7},tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat16));
    mfq::cuda::DecodeWindow window(mfq_current_cuda_stream());
    window.capture([&] {
        auto snapshot=input.clone();
        for(int layer=0;layer<3;++layer)audit.input(layer,snapshot.narrow(0,layer,1));
        window.enroll({&audit,[]{},[]{},[&]{audit.finish_sample();},{}});
    },[]{},[]{});
    if(audit.samples())throw std::runtime_error("lookahead audit recorded capture warmup as decode");
    std::vector<tb::Tensor> originals;
    for(int step=0;step<3;++step) {
        std::vector<float> values(21);for(int i=0;i<21;++i)values[i]=float(i+step*32)/16;
        values[step]=-0.0f;
        auto original=tb::tensor(values).to(tb::kFloat16).reshape({3,7}).contiguous();
        input.copy_(original.to(tb::kCUDA));window.run();originals.push_back(original);
    }
    for(int layer=0;layer<3;++layer) {
        std::vector<tb::Tensor> rows;for(const auto& original:originals)rows.push_back(original.narrow(0,layer,1));
        auto expected=tb::cat(rows,0).contiguous(),actual=audit.rows(layer);
        if(actual.sizes()!=expected.sizes() || std::memcmp(actual.data_ptr(),expected.data_ptr(),expected.numel()*expected.element_size()))
            throw std::runtime_error("lookahead audit changed input bytes or sample/layer order");
    }
    int guards=0;
    try {audit.finish_sample();}catch(const std::runtime_error&){++guards;}
    try {audit.input(3,input.narrow(0,0,1));}catch(const std::invalid_argument&){++guards;}
    try {audit.input(0,input);}catch(const std::invalid_argument&){++guards;}
    try {audit.rows(-1);}catch(const std::invalid_argument&){++guards;}
    if(guards!=4)throw std::runtime_error("lookahead audit shape/sample guards incomplete");
    std::cout<<"router_lookahead_input_samples=3 layer_rows=9 warmup_samples=0 guards=4 original_bytes_exact=1 PASS\n";
}
int main(int argc,char** argv)try {
    if(argc!=2 && argc!=3)throw std::runtime_error("expected real-shape range fixture directory");
    const std::filesystem::path root(argv[1]);
    auto cuda_context=mfq::cuda::default_context(mfq_current_cuda_device());
    if(argc==3) {
        if(std::string(argv[2])=="--router-prefetch-check") {
            verify_router_prefetch(root);return 0;
        }
        if(std::string(argv[2])=="--router-lookahead-input-check") {
            verify_router_lookahead_inputs();return 0;
        }
        if(std::string(argv[2])=="--transfer-comparison-check") {
            verify_two_stage_shared(root,true,true);return 0;
        }
        if(std::string(argv[2])=="--phased-transfer-check") {
            verify_phased_wire_bytes();verify_two_stage_shared(root,true);return 0;
        }
        if(std::string(argv[2])=="--transfer-reservation-check") {
            verify_transfer_hit_reservation(root);return 0;
        }
        if(std::string(argv[2])!="--residency-copy-bench")throw std::runtime_error("unknown residency benchmark mode");
        residency_copy_benchmark(root);return 0;
    }
    verify_transfer_hit_reservation(root);
    verify_rounded_swiglu();
    verify_resident_classification();
    verify_active_nvq(root);
    const int shared_input_cases=verify_shared_input(root);
    const int warm_residency_cases=verify_warm_residency(root);
    const int nint_dispatch_cases=verify_nint_dispatch(root);
    const int projection_heat_cases=verify_projection_heat(root);
    const int two_stage_shared_cases=verify_two_stage_shared(root);
    const int transfer_cache_shared_cases=verify_two_stage_shared(root,true);
    const std::vector<std::string> formats={"nint4","nint5","nint6","nint8","nvq1-s","nvq1-l","nvq2j","nvq2j-l","nvq2j-xl","nvq3j","nvq3j-512","nvq3j-l"};
    int cases=0,transactions=0,adaptive_cases=0,registered_ram_cases=0,cpu_profile_cases=0;
    int dma_profile_cases=0;
    std::int64_t dma_profile_samples=0,dma_profile_window_samples=0;
    std::int64_t dma_profile_window_copies=0,dma_profile_window_bytes=0;
    int shared_cost_cases=0,shared_cost_warm_cases=0,two_stage_format_cases=0;std::int64_t shared_cost_ready_queries=0;
    const int wire_checks=verify_wire_bytes();
    for(const auto& format:formats) {
        CudaExecutionContext execution;
        std::array<std::shared_ptr<MoeQuantRangeSource>,3> cold={source(root/(format+"-640-2560.mfq")),
            source(root/(format+"-640-2560.mfq")),source(root/(format+"-2560-640.mfq"))};
        std::array<std::shared_ptr<MoeQuantRangeSource>,3> oracle={source(root/(format+"-640-2560.mfq")),
            source(root/(format+"-640-2560.mfq")),source(root/(format+"-2560-640.mfq"))};
        std::size_t gpu_bytes=0;for(const auto& s:cold)gpu_bytes+=bytes(s->metadata()->pools[0]);
        execution.config.moe_host_cache_bytes=3*gpu_bytes;
        auto cache=make_moe_expert_cache(gpu_bytes,execution.config);
        std::vector<std::shared_ptr<MfeWeight>> gu,down;
        std::array<MfeWeight,3> baseline;
        for(int i=0;i<3;++i) {
            auto model=mfq::open_model_source((root/(format+(i==2 ? "-2560-640.mfq" : "-640-2560.mfq"))).string());
            baseline[i]=stage_cpu_mixed_moe(make_mixed_moe_runtime(load_mfe_cpu(*model,"linear.weight"),false),execution.config);
            auto weight=std::make_shared<MfeWeight>(cache_quant_moe_weight(cache,std::to_string(i),cold[i],1,0,std::to_string(i)));
            (i==2 ? down : gu).push_back(weight);
        }
        finalize_moe_expert_cache(cache);
        const auto initial_host=cold[0]->host_cache()->stats();
        auto ffn=make_moe_ffn_pipeline(gu,down);if(!ffn)throw std::runtime_error("MFQ FFN was not enabled");
        std::ifstream f(root/(format+"-640-2560.input.f32"),std::ios::binary);
        std::vector<float> inputs(3*2560);f.read(reinterpret_cast<char*>(inputs.data()),inputs.size()*sizeof(float));
        if(!f)throw std::runtime_error("missing real input fixture");
        auto verify_original=[&] {
            for(int i=0;i<3;++i) {
                if(!cold[i]->gpu_resident(0) || cold[i]->gpu_resident(2) ||
                    !cold[i]->host_cache()->contains(cold[i]->host_key(2)) ||
                    cold[i]->host_cache()->contains(cold[i]->host_key(0)))
                    throw std::runtime_error("failed bundle exchange changed tier ownership");
                const int width=i==2 ? 640 : 2560,output=i==2 ? 2560 : 640;
                auto x=tb::tensor(std::vector<float>(inputs.begin(),inputs.begin()+width)).reshape({1,width}).to(tb::kFloat16).to(tb::kCUDA);
                auto ids=tb::zeros({1,1},x.options().dtype(tb::kInt32));auto route=build_moe_route_plan(ids,3);
                auto actual=(i==2 ? down[0] : gu[i])->forward(execution,x,route).to(tb::kCPU).to(tb::kFloat32).contiguous();
                auto expected=baseline[i].forward(execution,x,route).to(tb::kCPU).to(tb::kFloat32).contiguous();
                for(int64_t j=0;j<actual.numel();++j)if(!std::isfinite(actual.data_ptr<float>()[j]) ||
                    std::abs(actual.data_ptr<float>()[j]-expected.data_ptr<float>()[j])>2e-5+1e-3*std::abs(expected.data_ptr<float>()[j]))
                    throw std::runtime_error("failed bundle exchange corrupted an outgoing GPU expert");
                auto lease=cold[i]->acquire_expert(2);auto cpu=x.to(tb::kCPU).to(tb::kFloat32);
                actual=linear(lease->weights,output,width).forward(execution,cpu).to(tb::kFloat32).contiguous();
                expected=linear(oracle[i]->read_expert(2),output,width).forward(execution,cpu).to(tb::kFloat32).contiguous();
                if(std::memcmp(actual.data_ptr(),expected.data_ptr(),actual.numel()*sizeof(float)))
                    throw std::runtime_error("failed bundle exchange corrupted an incoming RAM expert");
                if(cold[i]->expert_disk_reads_after_preload())throw std::runtime_error("bundle exchange rollback reread SSD");
            }
            const auto host=cold[0]->host_cache()->stats();
            if(host.managed_bytes!=initial_host.managed_bytes || host.managed_peak_bytes!=initial_host.managed_peak_bytes)
                throw std::runtime_error("bundle exchange rollback duplicated RAM experts");
        };
        for(const std::string phase:{"upload","map","host","publish"})for(int fail_at=0;fail_at<(phase=="publish" ? 1 : 3);++fail_at) {
            MoeResidencyManager manager(cache.get(),[&](const char* event,std::size_t index) {
                if(phase==event && index==std::size_t(fail_at))throw std::runtime_error("injected bundle exchange failure");
            });
            for(int step=0;step<4;++step)manager.after_layer(0,{2,2,2},1);
            bool rejected=false;try{manager.apply_pending();}catch(const std::runtime_error& e){rejected=std::string(e.what())=="injected bundle exchange failure";}
            if(!rejected)throw std::runtime_error("bundle exchange swallowed an injected failure");
            verify_original();++transactions;
        }
        {
            MoeHostExpertCache::Lease reader;
            MoeResidencyManager manager(cache.get(),[&](const char* event,std::size_t index) {
                if(std::string(event)=="upload" && index==2)reader=cold[1]->acquire_expert(2);
            });
            for(int step=0;step<4;++step)manager.after_layer(0,{2,2,2},1);
            manager.apply_pending();if(!reader)throw std::runtime_error("late CPU reader was not exercised");
            verify_original();reader.reset();++transactions;
        }
        {
            CudaExecutionContext cross;cross.config.moe_host_cache_bytes=6*gpu_bytes;
            auto tier=make_moe_expert_cache(2*gpu_bytes,cross.config);
            std::array<std::array<std::shared_ptr<MoeQuantRangeSource>,3>,2> layers;
            std::array<MoeFfnForward,2> forwards;
            for(int layer=0;layer<2;++layer) {
                std::vector<std::shared_ptr<MfeWeight>> gates,downs;
                for(int i=0;i<3;++i) {
                    layers[layer][i]=source(root/(format+(i==2 ? "-2560-640.mfq" : "-640-2560.mfq")));
                    auto weight=std::make_shared<MfeWeight>(cache_quant_moe_weight(tier,
                        std::to_string(layer)+"/"+std::to_string(i),layers[layer][i],1,layer,std::to_string(i)));
                    (i==2 ? downs : gates).push_back(weight);
                }
                forwards[layer]=make_moe_ffn_pipeline(gates,downs);
            }
            finalize_moe_expert_cache(tier);const auto host_before=layers[0][0]->host_cache()->stats();
            MoeResidencyManager manager(tier.get());
            for(int step=0;step<4;++step) {
                manager.after_layer(0,{0},1);
                manager.after_layer(1,{1,1,1,1,1,1,2,2,2,2,2,2},1);
            }
            manager.apply_pending();
            for(int i=0;i<3;++i)if(layers[0][i]->gpu_resident(0) || layers[1][i]->gpu_resident(0) ||
                    !layers[1][i]->gpu_resident(1) || !layers[1][i]->gpu_resident(2))
                throw std::runtime_error("cross-layer heat did not move whole expert bundles to the busy layer");
            auto x=tb::tensor(std::vector<float>(inputs.begin(),inputs.begin()+2560)).reshape({1,2560}).to(tb::kFloat16).to(tb::kCUDA);
            auto ids=tb::tensor(std::vector<int32_t>{1,2}).reshape({1,2}).to(tb::kCUDA);
            auto weights=tb::ones({1,2},x.options().dtype(tb::kFloat32));auto route=build_moe_route_plan(ids,3);
            auto actual=forwards[1](cross,x,ids,weights).to(tb::kCPU).to(tb::kFloat32).contiguous();
            auto g=baseline[0].forward(cross,x,route),u=baseline[1].forward(cross,x,route);
            auto expected=baseline[2].forward(cross,(g*tb::sigmoid(g))*u,route).to(tb::kCPU).to(tb::kFloat32).contiguous();
            for(int64_t j=0;j<actual.numel();++j)if(!std::isfinite(actual.data_ptr<float>()[j]) ||
                    std::abs(actual.data_ptr<float>()[j]-expected.data_ptr<float>()[j])>2e-5+1e-3*std::abs(expected.data_ptr<float>()[j]))
                throw std::runtime_error("cross-layer bundle exchange changed GPU arithmetic");
            for(const auto& layer:layers)for(int i=0;i<3;++i) {
                const int width=i==2 ? 640 : 2560,output=i==2 ? 2560 : 640;
                auto cpu=tb::tensor(std::vector<float>(inputs.begin(),inputs.begin()+width)).reshape({1,width});
                auto held=layer[i]->acquire_expert(0);
                actual=linear(held->weights,output,width).forward(cross,cpu).to(tb::kFloat32).contiguous();
                expected=linear(oracle[i]->read_expert(0),output,width).forward(cross,cpu).to(tb::kFloat32).contiguous();
                if(std::memcmp(actual.data_ptr(),expected.data_ptr(),actual.numel()*sizeof(float)))
                    throw std::runtime_error("cross-layer demotion changed CPU arithmetic");
                if(layer[i]->expert_disk_reads_after_preload())throw std::runtime_error("cross-layer exchange reread expert SSD");
            }
            const auto host_after=layers[0][0]->host_cache()->stats();
            if(host_after.managed_bytes!=host_before.managed_bytes || host_after.managed_peak_bytes!=host_before.managed_peak_bytes)
                throw std::runtime_error("cross-layer exchange duplicated RAM weights");
            ++transactions;
        }
        {
            CudaExecutionContext partial;partial.config.moe_host_cache_bytes=3*gpu_bytes;
            auto tier=make_moe_expert_cache(gpu_bytes+bytes(oracle[0]->metadata()->pools[0]),partial.config);
            std::array<std::shared_ptr<MoeQuantRangeSource>,3> parts;
            std::vector<std::shared_ptr<MfeWeight>> gates,downs;
            for(int i=0;i<3;++i) {
                parts[i]=source(root/(format+(i==2 ? "-2560-640.mfq" : "-640-2560.mfq")));
                auto weight=std::make_shared<MfeWeight>(cache_quant_moe_weight(tier,std::to_string(i),parts[i],1,0,std::to_string(i)));
                (i==2 ? downs : gates).push_back(weight);
            }
            auto forward=make_moe_ffn_pipeline(gates,downs);finalize_moe_expert_cache(tier);
            int mixed=-1;
            for(int expert=0;expert<3;++expert) {
                int hits=0;for(const auto& part:parts)hits+=part->gpu_resident(expert);
                if(hits && hits!=3){mixed=expert;break;}
            }
            if(mixed<0)throw std::runtime_error("partial-residency fixture did not create a split expert");
            auto x=tb::tensor(std::vector<float>(inputs.begin(),inputs.begin()+2560)).reshape({1,2560}).to(tb::kFloat16).to(tb::kCUDA);
            auto ids=tb::tensor(std::vector<int32_t>{mixed,mixed,mixed}).reshape({1,3}).to(tb::kCUDA);
            auto weights=tb::ones({1,3},x.options().dtype(tb::kFloat32));auto route=build_moe_route_plan(ids,3);
            auto actual=forward(partial,x,ids,weights).to(tb::kCPU).to(tb::kFloat32).contiguous();
            auto g=baseline[0].forward(partial,x,route),u=baseline[1].forward(partial,x,route);
            auto expected=baseline[2].forward(partial,(g*tb::sigmoid(g))*u,route).to(tb::kCPU).to(tb::kFloat32).contiguous();
            for(int64_t j=0;j<actual.numel();++j)if(!std::isfinite(actual.data_ptr<float>()[j]) ||
                    std::abs(actual.data_ptr<float>()[j]-expected.data_ptr<float>()[j])>2e-5+1e-3*std::abs(expected.data_ptr<float>()[j]))
                throw std::runtime_error("partial expert GPU/RAM assembly changed arithmetic");
            for(const auto& part:parts)if(part->expert_disk_reads_after_preload())
                throw std::runtime_error("partial expert reread SSD instead of using its GPU/RAM fields");
            ++transactions;
        }
        for(int tokens:{1,3,12})for(int iteration=0;iteration<8;++iteration) {
            finish_moe_expert_exchanges(cache);
            if(tokens==1 && iteration==4) {
                for(const auto& s:cold)if(!s->gpu_resident(2) || s->gpu_resident(0) ||
                    !s->host_cache()->contains(s->host_key(0)) || s->host_cache()->contains(s->host_key(2)))
                    throw std::runtime_error(format+" routing heat did not exchange VRAM and RAM ownership");
            }
            std::vector<float> input(tokens*2560);for(int t=0;t<tokens;++t)std::copy_n(inputs.data()+(t%3)*2560,2560,input.data()+t*2560);
            auto x=tb::tensor(input).reshape({tokens,2560}).to(tb::kFloat16).to(tb::kCUDA);
            std::vector<int32_t> selected(tokens*3);
            for(int p=0;p<tokens*3;++p)selected[p]=iteration<2 ? (p+iteration)%3 : iteration<6 ? 2 : 0;
            auto ids=tb::tensor(selected).reshape({tokens,3}).to(tb::kCUDA);
            auto weights=tb::ones({tokens,3},x.options().dtype(tb::kFloat32));
            auto resident=[&](int e){return std::all_of(cold.begin(),cold.end(),[&](const auto& s){return s->gpu_resident(e);});};
            auto dispatch=mfq::plan_moe_dispatch(selected,3,0,resident,[](int){return true;});
            int misses=0;for(const auto& group:dispatch.groups)misses+=group.kind!=mfq::MoeDispatchKind::GpuResident;
            dispatch=mfq::plan_moe_dispatch(selected,3,tokens>8 ? misses : static_cast<std::size_t>(misses*moe_expert_cache_ram_pcie_fraction(cache)),resident,[](int){return true;});
            auto actual=ffn(execution,x,ids,weights).to(tb::kCPU).to(tb::kFloat32).contiguous();
            auto route=build_moe_route_plan(ids,3);
            auto gu_gpu=tb::cat({baseline[0].forward(execution,x,route),baseline[1].forward(execution,x,route)},-1);
            auto g=gu_gpu.narrow(-1,0,640),u=gu_gpu.narrow(-1,640,640);
            auto expected_gpu=baseline[2].forward(execution,(g*tb::sigmoid(g))*u,route).to(tb::kCPU).to(tb::kFloat32).contiguous();
            for(int t=0;t<tokens;++t)for(int r=0;r<3;++r) {
                const int e=selected[t*3+r];
                const bool gpu=dispatch.kinds[t*3+r]!=mfq::MoeDispatchKind::Cpu;
                tb::Tensor expected;
                if(!gpu) {
                    auto input_cpu=x.narrow(0,t,1).to(tb::kCPU).to(tb::kFloat32).contiguous();
                    auto g=linear(oracle[0]->read_expert(e),640,2560).forward(execution,input_cpu);
                    auto u=linear(oracle[1]->read_expert(e),640,2560).forward(execution,input_cpu);
                    std::vector<float> activated(640);
                    for(int i=0;i<640;++i) {
                        const float gate=float(g.data_ptr<mfq_half>()[i]),up=float(u.data_ptr<mfq_half>()[i]);
                        const float sigmoid=float(mfq_half(1.0/(1.0+std::exp(-double(gate)))));
                        activated[i]=float(mfq_half(float(mfq_half(gate*sigmoid))*up));
                    }
                    auto hidden=tb::tensor(activated).reshape({1,640});
                    expected=linear(oracle[2]->read_expert(e),2560,640).forward(execution,hidden.to(tb::kFloat32)).to(tb::kFloat32);
                }
                for(int column=0;column<2560;++column) {
                    const float reference=gpu ? expected_gpu.data_ptr<float>()[(t*3+r)*2560+column] : expected.data_ptr<float>()[column];
                    const float value=actual.data_ptr<float>()[(t*3+r)*2560+column];
                    if(!std::isfinite(value) || std::abs(value-reference)>2e-5+1e-3*std::abs(reference))
                        throw std::runtime_error(format+" FFN graph/tier mismatch at "+std::to_string(t*3+r)+","+std::to_string(column)+" value="+std::to_string(value)+" expected="+std::to_string(reference));
                }
            }
            for(const auto& s:cold)if(s->expert_disk_reads_after_preload()!=0)throw std::runtime_error("FFN replay reread expert SSD");
            const auto host=cold[0]->host_cache()->stats();
            if(host.managed_bytes!=initial_host.managed_bytes || host.managed_peak_bytes!=initial_host.managed_peak_bytes)
                throw std::runtime_error("adaptive exchange allocated another retained RAM expert");
            ++cases;
        }
        {
            auto adaptive_stream=mfq_get_stream_from_pool();MfqCudaStreamGuard adaptive_guard(adaptive_stream);
            CudaExecutionContext adaptive;adaptive.config.moe_ram_pcie_fraction.reset();
            const auto* shared_cost=std::getenv("MFQ_MOE_SHARED_CPU_COST");
            const bool shared_cost_enabled=shared_cost && std::strcmp(shared_cost,"1")==0;
            // Keep whole RAM experts eligible while obtaining repeated cost
            // observations; projection-wise exchanges are covered separately.
            if(shared_cost_enabled)adaptive.config.moe_residency_projection_heat=false;
            adaptive.config.moe_host_cache_bytes=3*gpu_bytes;
            auto tier=make_moe_expert_cache(gpu_bytes,adaptive.config);
            std::array<std::shared_ptr<MoeQuantRangeSource>,3> parts;
            std::vector<std::shared_ptr<MfeWeight>> gates,downs;
            for(int i=0;i<3;++i) {
                parts[i]=source(root/(format+(i==2 ? "-2560-640.mfq" : "-640-2560.mfq")));
                auto weight=std::make_shared<MfeWeight>(cache_quant_moe_weight(tier,
                    "adaptive/"+std::to_string(i),parts[i],1,0,std::to_string(i)));
                (i==2 ? downs : gates).push_back(weight);
            }
            mfq::MoeDispatchPlan observed;
            auto forward=make_moe_ffn_pipeline(gates,downs,{},[&](const mfq::MoeDispatchPlan& plan){observed=plan;});
            finalize_moe_expert_cache(tier);
            auto x=tb::tensor(std::vector<float>(inputs.begin(),inputs.begin()+2560)).reshape({1,2560}).to(tb::kFloat16).to(tb::kCUDA);
            auto ids=tb::zeros({1,5},x.options().dtype(tb::kInt32));
            auto weights=tb::ones({1,5},x.options().dtype(tb::kFloat32));
            mfq::cuda::DecodeWindow window(mfq_current_cuda_stream());tb::Tensor output;
            window.capture([&]{output=forward(adaptive,x,ids,weights);},[]{},[&]{output={};});
            for(int step=shared_cost_enabled?-4:0;step<8;++step) {
                finish_moe_expert_exchanges(tier);
                std::vector<int32_t> selected={step<0?0:step%3,1,2,1,2};
                ids.copy_(tb::tensor(selected).reshape({1,5}).to(tb::kCUDA));
                window.run();
                auto actual=output.to(tb::kCPU).to(tb::kFloat32).contiguous();
                const auto route=build_moe_route_plan(ids,3);
                auto g=baseline[0].forward(adaptive,x,route),u=baseline[1].forward(adaptive,x,route);
                auto expected=baseline[2].forward(adaptive,(g*tb::sigmoid(g))*u,route).to(tb::kCPU).to(tb::kFloat32).contiguous();
                if(observed.kinds.size()!=selected.size())throw std::runtime_error("adaptive dispatch was not observed");
                for(std::size_t r=0;r<selected.size();++r) {
                    tb::Tensor expected_cpu;
                    const bool cpu=observed.kinds[r]==mfq::MoeDispatchKind::Cpu;
                    if(cpu) {
                        const int e=selected[r];
                        auto input_cpu=x.to(tb::kCPU).to(tb::kFloat32).contiguous();
                        auto gate=linear(oracle[0]->read_expert(e),640,2560).forward(adaptive,input_cpu);
                        auto up=linear(oracle[1]->read_expert(e),640,2560).forward(adaptive,input_cpu);
                        std::vector<float> activated(640);
                        for(int i=0;i<640;++i) {
                            const float gv=float(gate.data_ptr<mfq_half>()[i]),uv=float(up.data_ptr<mfq_half>()[i]);
                            const float sigmoid=float(mfq_half(1.0/(1.0+std::exp(-double(gv)))));
                            activated[i]=float(mfq_half(float(mfq_half(gv*sigmoid))*uv));
                        }
                        auto hidden=tb::tensor(activated).reshape({1,640});
                        expected_cpu=linear(oracle[2]->read_expert(e),2560,640).forward(adaptive,hidden).to(tb::kFloat32).contiguous();
                    }
                    for(int column=0;column<2560;++column) {
                        const float reference=cpu ? expected_cpu.data_ptr<float>()[column] : expected.data_ptr<float>()[r*2560+column];
                        const float value=actual.data_ptr<float>()[r*2560+column];
                        if(!std::isfinite(value) || std::abs(value-reference)>2e-5+1e-3*std::abs(reference))
                            throw std::runtime_error(format+" measured adaptive window changed canonical output step="+
                                std::to_string(step)+" route="+std::to_string(r)+" expert="+std::to_string(selected[r])+
                                " cpu="+std::to_string(cpu)+" column="+std::to_string(column)+
                                " value="+std::to_string(value)+" expected="+std::to_string(reference));
                    }
                }
                for(const auto& part:parts)if(part->expert_disk_reads_after_preload())
                    throw std::runtime_error("measured adaptive window reread expert SSD");
                if(step<0)++shared_cost_warm_cases;else ++adaptive_cases;
            }
            std::ostringstream cost_stats;print_moe_expert_cache_stats(tier,cost_stats);
            const auto cost_counter=[&](const char* name) {
                const auto text=cost_stats.str(),field=std::string(name)+"=";
                const auto position=text.find(field);
                if(position==std::string::npos)throw std::runtime_error("missing shared CPU cost evidence");
                return std::stoll(text.substr(position+field.size()));
            };
            const auto* enabled=std::getenv("MFQ_MOE_SHARED_CPU_COST");
            if(enabled && std::strcmp(enabled,"1")==0) {
                if(cost_counter("pipeline_shared_cpu_cost_enabled")!=1 ||
                   cost_counter("pipeline_cpu_cost_queries")<=0 || cost_counter("pipeline_cpu_cost_samples")<=0)
                    throw std::runtime_error("shared CPU cost mode did not observe real adaptive work");
                const auto* background=std::getenv("MFQ_MOE_CPU_BACKGROUND_CALIBRATION");
                const auto* budget=std::getenv("MFQ_MOE_CPU_TRANSFER_BUDGET");
                const bool expected_background=(!background || std::strcmp(background,"1")==0) &&
                    (!budget || budget[0]!='0');
                const auto jobs=cost_counter("pipeline_cpu_calibration_jobs");
                if((expected_background && jobs<=0) || (!expected_background && jobs) ||
                   jobs!=cost_counter("pipeline_cpu_calibration_observations"))
                    throw std::runtime_error("CPU calibration execution/completion differs from selected mode");
                shared_cost_ready_queries+=cost_counter("pipeline_cpu_cost_ready_queries");
                ++shared_cost_cases;
            } else if(cost_counter("pipeline_shared_cpu_cost_enabled") || cost_counter("pipeline_cpu_cost_samples"))
                throw std::runtime_error("disabled shared CPU cost mode collected samples");
        }
        // Two successive expert layers share transfer arenas. Changing input
        // and routes must reach both layers in one launch without a wait on an
        // event queued behind the same, still-in-flight graph.
        auto second=make_moe_ffn_pipeline(gu,down);
        for(int tokens:{1,3,12}) {
            auto x=tb::zeros({tokens,2560},tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat16));
            auto ids=tb::zeros({tokens,3},x.options().dtype(tb::kInt32));
            auto weights=tb::ones({tokens,3},x.options().dtype(tb::kFloat32));
            mfq::cuda::DecodeWindow window(mfq_current_cuda_stream());tb::Tensor output;
            for(int step=0;step<3;++step) {
                finish_moe_expert_exchanges(cache);
                std::vector<float> input(tokens*2560);for(int i=0;i<tokens*2560;++i)input[i]=inputs[(i+step*17)%inputs.size()];
                std::vector<int32_t> selected(tokens*3);for(int i=0;i<tokens*3;++i)selected[i]=(i+step)%3;
                x.copy_(tb::tensor(input).reshape({tokens,2560}).to(tb::kFloat16).to(tb::kCUDA));
                ids.copy_(tb::tensor(selected).reshape({tokens,3}).to(tb::kCUDA));
                auto first=ffn(execution,x,ids,weights);
                // Synthetic fixture weights amplify coherent inputs strongly.
                // Bound the inter-layer input while keeping its graph dependency.
                auto expected=second(execution,tb::tanh(first.mean(1))*.0001,ids,weights).to(tb::kCPU).to(tb::kFloat32).contiguous();
                window.capture([&]{auto a=ffn(execution,x,ids,weights);output=second(execution,tb::tanh(a.mean(1))*.0001,ids,weights);},[]{},[&]{output={};});
                window.run();auto actual=output.to(tb::kCPU).to(tb::kFloat32).contiguous();
                for(int64_t i=0;i<actual.numel();++i)if(!std::isfinite(actual.data_ptr<float>()[i]) ||
                    std::abs(actual.data_ptr<float>()[i]-expected.data_ptr<float>()[i])>2e-5+1e-3*std::abs(expected.data_ptr<float>()[i]))
                    throw std::runtime_error(format+" two-layer window mismatch tokens="+std::to_string(tokens)+" step="+std::to_string(step)+" at "+std::to_string(i)+" actual="+std::to_string(actual.data_ptr<float>()[i])+" expected="+std::to_string(expected.data_ptr<float>()[i]));
                for(const auto& s:cold)if(s->expert_disk_reads_after_preload())throw std::runtime_error("window reread resident expert SSD");
                ++cases;
            }
        }
        {
            mfq::cuda::DecodeWindow failed(mfq_current_cuda_stream());
            auto x=tb::zeros({1,2560},tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat16));
            auto ids=tb::tensor(std::vector<int32_t>{0,1,2}).reshape({1,3}).to(tb::kCUDA);
            auto weights=tb::ones({1,3},x.options().dtype(tb::kFloat32));
            failed.capture([&]{failed.enroll({&failed,[]{},[]{throw std::runtime_error("injected host read failure");},{},{}});ffn(execution,x,ids,weights);},[]{});
            bool rejected=false;try{failed.run();}catch(const std::runtime_error& e){rejected=std::string(e.what())=="injected host read failure";}
            if(!rejected)throw std::runtime_error("window swallowed its host failure");
            MFQ_CUDA_CHECK(cudaStreamSynchronize(mfq_current_cuda_stream()));
            rejected=false;try{failed.run();}catch(const std::runtime_error&){rejected=true;}
            if(!rejected)throw std::runtime_error("failed window accepted another replay");
        }
        std::ostringstream statistics;print_moe_expert_cache_stats(cache,statistics);
        auto counter=[&](const std::string& name) {
            std::smatch match;const auto text=statistics.str();
            if(!std::regex_search(text,match,std::regex(name+"=([0-9]+)")))
                throw std::runtime_error("missing RAM transfer counter "+name);
            return std::stoll(match[1]);
        };
        if(const auto* value=std::getenv("MFQ_TRACE_CPU_ROWS");value && std::strcmp(value,"1")==0) {
            const auto text=statistics.str();
            const auto* fail=std::getenv("MFQ_TRACE_CPU_ROWS_FAIL_ALLOC");
            if(fail && std::strcmp(fail,"1")==0) {
                if(text.find("moe_cpu_profile enabled=0 dropped=1 samples=0")==std::string::npos)
                    throw std::runtime_error("optional CPU trace allocation failure did not disable tracing");
            } else {
                if(text.find("moe_cpu_profile enabled=1 dropped=0")==std::string::npos)
                    throw std::runtime_error("CPU trace lost its observations");
                std::istringstream lines(text);std::string line;std::int64_t projections=0;
                while(std::getline(lines,line))if(line.rfind("moe_cpu_projection ",0)==0) {
                    const auto field=[&](const char* name) {
                        std::smatch found;
                        if(!std::regex_search(line,found,std::regex(std::string(name)+"=([0-9]+)")))
                            throw std::runtime_error("CPU projection observation missing field");
                        return std::stoll(found[1]);
                    };
                    if(field("rows")!=field("outputs") || field("positions")<=0 || field("callbacks")<=0)
                        throw std::runtime_error("CPU trace omitted or duplicated projection rows");
                    ++projections;
                }
                if(!projections || projections!=3*counter("hybrid_cpu_experts"))
                    throw std::runtime_error("CPU trace projection count differs from real dispatch");
            }
            ++cpu_profile_cases;
        }
        if(const auto* value=std::getenv("MFQ_TRACE_MOE_DMA");value && std::strcmp(value,"1")==0) {
            const auto text=statistics.str();const auto* fail=std::getenv("MFQ_TRACE_MOE_DMA_FAIL_ALLOC");
            if(fail && std::strcmp(fail,"1")==0) {
                if(text.find("moe_dma_profile enabled=0 dropped=1 samples=0")==std::string::npos)
                    throw std::runtime_error("optional DMA trace allocation failure did not disable tracing");
            } else {
                std::smatch head;
                if(!std::regex_search(text,head,std::regex("moe_dma_profile enabled=1 dropped=0 samples=([0-9]+) skipped=([0-9]+)")))
                    throw std::runtime_error("DMA trace lost its observations");
                std::istringstream lines(text);std::string line;
                std::int64_t samples=0,windows=0,copies=0,bytes=0;
                std::vector<std::int64_t> serves;
                while(std::getline(lines,line))if(line.rfind("moe_dma_call ",0)==0) {
                    const auto field=[&](const char* name) {
                        std::smatch found;
                        if(!std::regex_search(line,found,std::regex(std::string("(?:^| )")+name+"=(-?[0-9]+)")))
                            throw std::runtime_error("DMA observation missing field");
                        return std::stoll(found[1]);
                    };
                    const auto serve=field("serve");
                    if(std::find(serves.begin(),serves.end(),serve)!=serves.end())
                        throw std::runtime_error("DMA trace duplicated a served call");
                    serves.push_back(serve);
                    // CUDA event timestamps have approximately 0.5 us resolution.
                    // Independently measured intervals must agree within 2 us.
                    if(field("copies")<=0 || field("copy_bytes")<field("payload_bytes") ||
                        field("first_copy_bytes")<=0 || field("first_copy_bytes")>field("copy_bytes") ||
                        field("copy_ns")<0 || field("notice_ns")<0 || field("hot_ns")<=0 ||
                        std::abs(field("copy_ns")-(field("copy_end_after_hot_ns")-field("copy_begin_after_hot_ns")))>2000 ||
                        std::abs(field("notice_ns")-(field("notice_end_after_hot_ns")-field("copy_end_after_hot_ns")))>2000 ||
                        field("prepare_ns")+field("enqueue_ns")>field("fetch_ns"))
                        throw std::runtime_error("DMA trace has inconsistent transfer timestamps or bytes");
                    ++samples;
                    if(field("window")) {++windows;copies+=field("copies");bytes+=field("copy_bytes");}
                }
                if(samples!=std::stoll(head[1]) || (!samples && std::stoll(head[2])==0))
                    throw std::runtime_error("DMA trace sample count differs from recorded calls");
                if(std::stoll(head[2])==0 &&
                    (copies!=counter("pipeline_window_dma_copies") || bytes!=counter("pipeline_window_dma_bytes")))
                    throw std::runtime_error("DMA trace window byte/copy counts differ from real submissions");
                dma_profile_samples+=samples;dma_profile_window_samples+=windows;
                dma_profile_window_copies+=copies;dma_profile_window_bytes+=bytes;
            }
            ++dma_profile_cases;
        } else if(statistics.str().find("moe_dma_")!=std::string::npos)
            throw std::runtime_error("disabled DMA profiling still produced observations");
        if(execution.config.moe_two_stage_ffn) {
            if(counter("pipeline_two_stage_serves")<=0)throw std::runtime_error(format+" did not serve the two-stage candidate");
            ++two_stage_format_cases;
        }
        if(execution.config.moe_direct_ram) {
            if(counter("pipeline_ram_registered_bytes")<=0 || counter("pipeline_direct_ram_bytes")<=0 ||
                counter("pipeline_staged_ram_bytes")!=0)
                throw std::runtime_error(format+" did not retain registered RAM transfers across cache exchanges");
            ++registered_ram_cases;
        }else if(counter("pipeline_ram_registered_bytes") || counter("pipeline_direct_ram_bytes"))
            throw std::runtime_error("disabled direct RAM transfer still registered memory");
    }
    std::cout<<"MFQ FFN canonical graph/tier/heat exchange checks PASS cases="<<cases
        <<" transactions="<<transactions<<" adaptive="<<adaptive_cases<<" wire="<<wire_checks<<'\n';
    std::cout<<"registered_ram_cases="<<registered_ram_cases<<'\n';
    std::cout<<"two_stage_format_cases="<<two_stage_format_cases<<" two_stage_shared_cases="<<two_stage_shared_cases<<'\n';
    std::cout<<"transfer_cache_shared_exact_cases="<<transfer_cache_shared_cases<<'\n';
    std::cout<<"cpu_profile_cases="<<cpu_profile_cases<<" optional observations/failure continuity checked\n";
    std::cout<<"dma_profile_cases="<<dma_profile_cases<<" dma_profile_samples="<<dma_profile_samples
        <<" dma_profile_window_samples="<<dma_profile_window_samples
        <<" dma_profile_window_copies="<<dma_profile_window_copies
        <<" dma_profile_window_bytes="<<dma_profile_window_bytes<<" optional observations/failure continuity checked\n";
    if(shared_cost_cases && !shared_cost_ready_queries)throw std::runtime_error("adaptive dispatch never consumed a calibrated shared CPU rate");
    std::cout<<"shared_cpu_cost_cases="<<shared_cost_cases<<" calibrated_queries="<<shared_cost_ready_queries
        <<" numeric_warm_cases="<<shared_cost_warm_cases<<'\n';
    std::cout<<"shared_input_cases="<<shared_input_cases<<" mixed Gate/Up outputs, bytes and scales exact PASS\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
