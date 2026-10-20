#define main mfq_original_mfe_bench_main
#include "mfq_mfe_bench.cpp"
#undef main
#include "runtime/moe_pipeline.h"
#include "mfe_ffn_compact.h"
#include "mfq_cuda_shared_gate.h"
#include <iostream>
#include <iomanip>

using View=mfq::cuda::MfePackedProjection;
static void compact_runtime(Call& call) {
    auto projections=call.layer.projections;
    for(auto& runtime:projections)for(auto& pool:runtime.pools)if(pool.family==MixedMoeFamily::Nvq) {
        auto& w=pool.nvq;const auto found=call.layer.dense_e8_streams.find(w.indices_packed.data_ptr());
        if(found==call.layer.dense_e8_streams.end())continue;
        w.indices_packed=found->second;w.aux_packed=tb::empty({0},w.indices_packed.options());
        w.sub_scale_packed=tb::empty({0},w.indices_packed.options());w.dense_groups=true;
    }
    std::array<const NintWeight*,3> shared{};
    if(call.shared_gate.defined())for(int p=0;p<3;++p)shared[p]=&call.layer.shared[p].nint;
    call.fused=std::make_unique<MfeFfnRuntime>(std::array<MixedMoeRuntime*,3>{&projections[0],&projections[1],&projections[2]},
        call.x,call.ids,call.weights,call.table,shared,call.shared_gate);
    if(!call.fused->batch().compact_nvq)throw std::runtime_error("probe did not construct production compact views");
    call.fused->run();
}

struct CachedDispatch {
    Call call;
    mfq::cuda::HostBuffer host;
    mfq::cuda::HostBuffer cache_host;
    uint8_t* mapped=nullptr;
    uint8_t* cache_mapped=nullptr;
    tb::Tensor masked_map,snapshot,wire;
    tb::Tensor old_payload,new_payload;
    int replacement_expert=-1;
    std::size_t replacement_bytes=0;
    const void* original_payload=nullptr;
    bool gate_fusion=false;
    bool evaluate_gate=false;
    tb::Tensor gate_quantized,gate_scales;
    int mode,blocks=0,threads=0;
    std::size_t flags_bytes=6*64,kind_offset,index_offset,packet_offset,table_offset,view_offset,half=64;
    CachedDispatch(Layer& layer,int tokens,int arm,bool shared,bool fuse_gate=false,bool include_gate=false):call(layer,shared,tokens),
        host(6*64+2*experts*4+tokens*routes*8+128+tokens*routes*3*sizeof(View),true),
        cache_host(experts*sizeof(mfq::cuda::MfeCachedExpert)+tokens*routes*4,true),mode(arm) {
        compact_runtime(call);
        std::memset(host.data(),0,host.size());
        fb::check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&mapped),host.data(),0));
        std::memset(cache_host.data(),0,cache_host.size());
        fb::check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&cache_mapped),cache_host.data(),0));
        kind_offset=flags_bytes;index_offset=kind_offset+experts*4;packet_offset=index_offset+experts*4;
        table_offset=packet_offset+tokens*routes*8;view_offset=table_offset+half*2;
        auto* base=static_cast<uint8_t*>(host.data());auto* flags=reinterpret_cast<uint32_t*>(base);
        flags[16]=flags[32]=flags[48]=flags[80]=1;
        auto* kinds=reinterpret_cast<int32_t*>(base+kind_offset);auto* indices=reinterpret_cast<int32_t*>(base+index_offset);
        auto* packet=reinterpret_cast<int32_t*>(base+packet_offset);auto* transfer=reinterpret_cast<View*>(base+view_offset);
        std::fill_n(indices,experts,-1);std::vector<int32_t> local(experts,-1);
        std::vector<View> primary;
        const auto& initial=call.fused->batch();
        const int count=initial.projection_counts[0]+initial.projection_counts[1]+initial.projection_counts[2];
        primary.resize(count+int(initial.shared!=nullptr)*3);
        fb::check(cudaMemcpyAsync(primary.data(),initial.projections,primary.size()*sizeof(View),cudaMemcpyDeviceToHost,mfq_current_cuda_stream()));
        fb::check(cudaStreamSynchronize(mfq_current_cuda_stream()));
        std::vector<int32_t> cohorts(3*experts);
        fb::check(cudaMemcpyAsync(cohorts.data(),initial.cohort_by_expert,cohorts.size()*4,cudaMemcpyDeviceToHost,mfq_current_cuda_stream()));
        fb::check(cudaStreamSynchronize(mfq_current_cuda_stream()));
        int cached=0;
        std::array<std::vector<std::vector<int64_t>>,3> cohort_bytes;
        for(int p=0;p<3;++p)cohort_bytes[p].resize(initial.projection_counts[p]);
        for(int r=0;r<routes;++r) {
            const int expert=layer.selected[r];const bool cold=mode==1 || mode==3 || mode==5 || mode==7 || ((mode==2 || mode==4 || mode==6 || mode==8) && (r&1));
            const bool transferred=cold && (mode<3 || mode>=5);
            const bool complete_cache=mode==7 && !(r&1);
            kinds[expert]=transferred && !complete_cache?2:1;indices[expert]=r;local[expert]=cold?-1:0;cached+=transferred && !complete_cache;
            for(int p=0;p<3;++p) {
                const auto& d=primary[initial.projection_offsets[p]+cohorts[p*experts+expert]];
                auto& pool=layer.projections[p].pools[r];std::vector<int64_t> bytes;
                if(pool.family==MixedMoeFamily::Nint)for(auto* field:fields(pool))bytes.push_back(field->numel()*field->element_size());
                else bytes={d.sizes[0],d.sizes[1],d.sizes[2],int64_t(pool.nvq.neuron_scale.numel()*pool.nvq.neuron_scale.element_size())};
                transfer[r*3+p]=call.fused->expert_view(p,expert,bytes,0);
                cohort_bytes[p][cohorts[p*experts+expert]]=bytes;
                auto& state=static_cast<mfq::cuda::MfeCachedExpert*>(cache_host.data())[expert];
                for(int f=0;f<(d.family==1?7:4);++f)state.fields[p][f]=reinterpret_cast<uint64_t>(transfer[r*3+p].fields[f]);
                state.ready_mask=mode>=7?(cold && (mode==8 || (r&1))?0:7):mode>=5?3:7;
                if(mode>=3 && p==0 && r==((mode==4 || mode==6)?1:0) && bytes[0]) {
                    replacement_expert=expert;replacement_bytes=std::size_t(bytes[0]);original_payload=transfer[r*3+p].fields[0];
                    old_payload=tb::empty({bytes[0]},call.x.options().dtype(tb::kUInt8));
                    fb::check(cudaMemcpyAsync(old_payload.data_ptr(),original_payload,replacement_bytes,cudaMemcpyDeviceToDevice,mfq_current_cuda_stream()));
                    state.fields[p][0]=reinterpret_cast<uint64_t>(old_payload.data_ptr());
                }
            }
        }
        for(int t=0;t<tokens;++t)for(int r=0;r<routes;++r) {
            const int expert=layer.selected[r];packet[(t*routes+r)*2]=kinds[expert];packet[(t*routes+r)*2+1]=indices[expert];
        }
        masked_map=tb::tensor(local).to(tb::kCUDA);
        if(mode<3) {
            for(int i=0;i<count;++i)primary[i].expert_local=masked_map.data_ptr<int32_t>();
            fb::check(cudaMemcpyAsync(const_cast<View*>(initial.projections),primary.data(),primary.size()*sizeof(View),cudaMemcpyHostToDevice,mfq_current_cuda_stream()));
        }
        call.fused->asynchronous(reinterpret_cast<int32_t*>(mapped+kind_offset),reinterpret_cast<uint32_t*>(mapped)+16,
            reinterpret_cast<uint32_t*>(mapped)+32,reinterpret_cast<uint32_t*>(mapped)+48,
            reinterpret_cast<uint32_t*>(mapped)+64,reinterpret_cast<int32_t*>(mapped+index_offset),mapped);
        evaluate_gate=fuse_gate || include_gate;
        if(evaluate_gate)std::tie(gate_quantized,gate_scales)=call.fused->reserve_input_quantization(int(layer.shared_router.nint.ng),48);
        if(!call.fused->batch().resident_plan_overlap)throw std::runtime_error("probe needs resident-plan overlap");
        if(mode>=3) {
            int original_counts[3];std::copy_n(initial.projection_counts,3,original_counts);
            call.fused->enable_cached_dispatch(reinterpret_cast<mfq::cuda::MfeCachedExpert*>(cache_mapped),cohort_bytes,reinterpret_cast<uint32_t*>(mapped)+4);
            if(fuse_gate) {
                if(!layer.shared_router.is_nint() || !call.fused->try_fuse_prepare_shared_gate(layer.shared_router.nint,call.shared_gate))
                    throw std::runtime_error("real scalar gate did not qualify for prepare fusion");
                gate_fusion=true;
            }
            const auto& enabled=call.fused->batch();
            const int bytes=(enabled.projection_offsets[2]+enabled.projection_counts[2]+int(enabled.shared!=nullptr)*3)*int(sizeof(View));
            std::vector<View> views(bytes/sizeof(View));
            fb::check(cudaMemcpyAsync(views.data(),enabled.projections,bytes,cudaMemcpyDeviceToHost,mfq_current_cuda_stream()));
            fb::check(cudaStreamSynchronize(mfq_current_cuda_stream()));
            for(int p=0;p<3;++p)for(int c=0;c<original_counts[p];++c)
                views[enabled.projection_offsets[p]+c].expert_local=masked_map.data_ptr<int32_t>();
            fb::check(cudaMemcpyAsync(const_cast<View*>(enabled.projections),views.data(),bytes,cudaMemcpyHostToDevice,mfq_current_cuda_stream()));
            fb::check(cudaStreamSynchronize(mfq_current_cuda_stream()));
        }
        call.fused->bind_transfer_route_plan(reinterpret_cast<int32_t*>(mapped+packet_offset));
        for(int phase=0;phase<2;++phase) {
            auto* header=reinterpret_cast<uint64_t*>(base+table_offset+phase*half);
            header[0]=~uint64_t(0);header[2]=!phase && cached?uint64_t(routes*3*sizeof(View)/8):0;
        }
        snapshot=tb::empty({int64_t(half*2)},call.x.options().dtype(tb::kUInt8));
        wire=tb::zeros({32},snapshot.options());
        fb::check(mfq::cuda::moe_wire_launch_geometry(&blocks,&threads));
        fb::check(cudaStreamSynchronize(mfq_current_cuda_stream()));
    }
    void prepare() {
        if(mode>=3)call.fused->prepare(reinterpret_cast<int32_t*>(cache_mapped+experts*sizeof(mfq::cuda::MfeCachedExpert)),reinterpret_cast<uint32_t*>(mapped));
        else call.fused->prepare();
        if(evaluate_gate && !gate_fusion) {
            call.fused->bind_shared_gate({});
            const auto gate=try_nint_shared_gate_sigmoid_cuda(call.x,call.layer.shared_router.nint,gate_quantized,gate_scales);
            if(!gate)throw std::runtime_error("unfused prepared scalar gate rejected real weights");
            call.fused->bind_shared_gate(*gate);
        }
    }
    void change_cached_address() {
        if(replacement_expert<0)throw std::runtime_error("generation check lacks a nonempty compressed field");
        new_payload=tb::empty({int64_t(replacement_bytes)},old_payload.options());
        fb::check(cudaMemcpyAsync(new_payload.data_ptr(),original_payload,replacement_bytes,cudaMemcpyDeviceToDevice,mfq_current_cuda_stream()));
        auto& state=static_cast<mfq::cuda::MfeCachedExpert*>(cache_host.data())[replacement_expert];
        state.fields[0][0]=reinterpret_cast<uint64_t>(new_payload.data_ptr());state.ready_mask+=8;
        old_payload.zero_();
    }
    void run() {
        const auto stream=mfq_current_cuda_stream();const auto& b=call.fused->batch();
        prepare();call.fused->resident();
        call.fused->wait_transfer(mapped+table_offset,snapshot.data_ptr(),1,
            mapped+view_offset,uint64_t(call.tokens)*routes*3*sizeof(View)/8);
        mfq::cuda::moe_scatter_wire_and_loan_phase(wire.data_ptr(),snapshot.data_ptr(),1,b.aborted,blocks,threads,1,stream);
        call.fused->transferred();
        mfq::cuda::wait_mapped_plan_snapshot(reinterpret_cast<uint32_t*>(mapped)+80,
            reinterpret_cast<uint32_t*>(mapped)+64,b.aborted,mapped+table_offset+half,
            snapshot.data_ptr<uint8_t>()+half,1,stream);
        mfq::cuda::moe_scatter_wire_and_loan_phase(wire.data_ptr(),snapshot.data_ptr<uint8_t>()+half,1,b.aborted,blocks,threads,2,stream);
        call.fused->down_reduce_compute();
    }
    void phase(int stage) {
        const auto stream=mfq_current_cuda_stream();const auto& b=call.fused->batch();
        if(stage==0)prepare();
        else if(stage==1)call.fused->resident();
        else if(stage==2) {
            call.fused->wait_transfer(mapped+table_offset,snapshot.data_ptr(),1,
                mapped+view_offset,uint64_t(call.tokens)*routes*3*sizeof(View)/8);
            mfq::cuda::moe_scatter_wire_and_loan_phase(wire.data_ptr(),snapshot.data_ptr(),1,b.aborted,blocks,threads,1,stream);
        }else if(stage==3)call.fused->transferred();
        else if(stage==4) {
            mfq::cuda::wait_mapped_plan_snapshot(reinterpret_cast<uint32_t*>(mapped)+80,
                reinterpret_cast<uint32_t*>(mapped)+64,b.aborted,mapped+table_offset+half,
                snapshot.data_ptr<uint8_t>()+half,1,stream);
            mfq::cuda::moe_scatter_wire_and_loan_phase(wire.data_ptr(),snapshot.data_ptr<uint8_t>()+half,1,b.aborted,blocks,threads,2,stream);
        }else call.fused->down_reduce_compute();
    }
};

int main(int argc,char** argv)try {
    std::cout.setf(std::ios::unitbuf);
    if(argc!=2 && argc!=3)throw std::runtime_error("expected MFQ model path and optional --inspect");
    const bool inspect=argc==3 && std::string(argv[2])=="--inspect";
    const bool profile=argc==3 && std::string(argv[2])=="--profile";
    const bool in_graph=argc==3 && std::string(argv[2])=="--in-graph-profile";
    const bool cached_check=argc==3 && std::string(argv[2])=="--cached-resident-check";
    const bool cached_profile=argc==3 && std::string(argv[2])=="--cached-resident-profile";
    const bool generation_check=argc==3 && std::string(argv[2])=="--cached-generation-check";
    const bool cached_gu=argc==3 && (std::string(argv[2])=="--cached-gu-check" || std::string(argv[2])=="--cached-gu-generation-check");
    const bool gu_generation=argc==3 && std::string(argv[2])=="--cached-gu-generation-check";
    const bool prepare_gate=argc==3 && std::string(argv[2])=="--prepare-gate-check";
    const bool gate_bench=argc==3 && std::string(argv[2])=="--prepare-gate-bench";
    const bool transfer_fused=argc==3 && std::string(argv[2])=="--transfer-hidden-fused-check";
    const bool prequantized_gu=transfer_fused || (argc==3 && std::string(argv[2])=="--prequantized-gu-check");
    const bool parity_read=prequantized_gu || (argc==3 && std::string(argv[2])=="--parity-read-check");
    auto context=mfq::cuda::default_context(0);auto stream=mfq_get_stream_from_pool(false);MfqCudaStreamGuard guard(stream);
    auto model=mfq::open_model_source(argv[1]);std::vector<std::unique_ptr<Layer>> layers;
    for(int layer:{0,8,16,24,32,40})layers.push_back(std::make_unique<Layer>(*model,layer));
    cudaDeviceProp properties{};fb::check(cudaGetDeviceProperties(&properties,0));
    for(int tokens:{1,2,6})for(bool shared:{false,true}) {
        if(std::getenv("MFQ_CACHED_PROBE_M1_ONLY") && tokens!=1)continue;
        if(prepare_gate && !shared)continue;
        if(gate_bench && (tokens!=1 || !shared))continue;
        if((profile || in_graph || cached_profile || generation_check || gu_generation) && (tokens!=1 || !shared))continue;
        std::vector<tb::Tensor> expected;size_t total_bytes=0;
        for(auto& layer:layers) {
            if(parity_read)_putenv_s("MFQ_MFE_PARITY_READ","0");
            Call reference(*layer,shared,tokens);compact_runtime(reference);
            expected.push_back(reference.fused->output().cpu());
            for(auto bytes:layer->bytes)total_bytes+=bytes;
            if(shared)total_bytes+=layer->shared_bytes;
        }
        if(total_bytes<4*size_t(properties.l2CacheSize))throw std::runtime_error("working set is too small for a memory-streaming comparison");
        if(parity_read)_putenv_s("MFQ_MFE_PARITY_READ","1");
        for(int mode:transfer_fused?std::vector<int>{7,8,5,6}:parity_read?std::vector<int>{3,4,5,6}:cached_gu?std::vector<int>{5,6}:cached_check || cached_profile || generation_check || prepare_gate || gate_bench?std::vector<int>{3,4}:std::vector<int>{0,1,2}) {
            if((profile || in_graph) && mode==2)continue;
            std::vector<std::unique_ptr<CachedDispatch>> calls;
            for(size_t l=0;l<layers.size();++l) {
                calls.push_back(std::make_unique<CachedDispatch>(*layers[l],tokens,gate_bench?3:mode,shared,prepare_gate || (parity_read && shared) || (gate_bench && mode==4),gate_bench));calls.back()->run();
                const auto actual=calls.back()->call.fused->output().cpu();
                if(transfer_fused && tokens==1 && mode>=7) {
                    std::vector<int32_t> kinds(experts);
                    fb::check(cudaMemcpy(kinds.data(),calls.back()->call.fused->batch().kinds,experts*4,cudaMemcpyDeviceToHost));
                    int active=0;for(int r=0;r<routes;++r)active+=kinds[layers[l]->selected[r]]==2;
                    if(active!=routes/2)throw std::runtime_error("fused transfer GU fixture did not execute five cold routes");
                    std::cout<<"transfer_fused_active_cold_routes="<<active<<" mode="<<mode<<" layer="<<layers[l]->index<<std::endl;
                }
                if(cached_gu) {
                    std::vector<int32_t> kinds(experts);
                    fb::check(cudaMemcpy(kinds.data(),calls.back()->call.fused->batch().kinds,experts*4,cudaMemcpyDeviceToHost));
                    for(int r=0;r<routes;++r)if(kinds[layers[l]->selected[r]]!=(mode==5 || (r&1)?6:1))
                        throw std::runtime_error("cached GU did not enter the early compute partition");
                }
                if(actual.nbytes()!=expected[l].nbytes() || std::memcmp(actual.data_ptr(),expected[l].data_ptr(),actual.nbytes())) {
                    std::cerr<<"MISMATCH layer="<<layers[l]->index<<" tokens="<<tokens<<" shared="<<shared<<" mode="<<mode<<std::endl;
                    const auto& b=calls.back()->call.fused->batch();
                    std::vector<int32_t> kinds(experts),maps(experts*3);
                    fb::check(cudaMemcpy(kinds.data(),b.kinds,experts*4,cudaMemcpyDeviceToHost));
                    fb::check(cudaMemcpy(maps.data(),b.cohort_by_expert,experts*3*4,cudaMemcpyDeviceToHost));
                    for(int r=0;r<routes;++r) {
                        const int e=layers[l]->selected[r];std::cerr<<"EXPERT "<<e<<" kind="<<kinds[e];
                        for(int p=0;p<3;++p) {
                            View view{};fb::check(cudaMemcpy(&view,b.projections+b.projection_offsets[p]+maps[p*experts+e],sizeof(View),cudaMemcpyDeviceToHost));
                            const auto& source=reinterpret_cast<View*>(static_cast<uint8_t*>(calls.back()->host.data())+calls.back()->view_offset)[r*3+p];
                            std::cerr<<" p="<<p<<" map="<<maps[p*experts+e]<<" equal="<<!std::memcmp(&view,&source,sizeof(View));
                            if(r==0) {
                                std::cerr<<" family="<<view.family<<" rows="<<view.output_rows<<" locals="<<view.local_experts;
                                for(int f=0;f<9;++f)if(view.fields[f]!=source.fields[f])std::cerr<<" field"<<f<<"="<<view.fields[f]<<"/"<<source.fields[f];
                                for(int i=72;i<184;++i)if(reinterpret_cast<const uint8_t*>(&view)[i]!=reinterpret_cast<const uint8_t*>(&source)[i])std::cerr<<" byte"<<i<<"="<<int(reinterpret_cast<const uint8_t*>(&view)[i])<<"/"<<int(reinterpret_cast<const uint8_t*>(&source)[i]);
                            }
                        }
                        std::cerr<<std::endl;
                    }
                    for(int i=0;i<actual.numel();++i)if(std::memcmp(static_cast<const uint8_t*>(actual.data_ptr())+i*actual.element_size(),static_cast<const uint8_t*>(expected[l].data_ptr())+i*actual.element_size(),actual.element_size())) {
                        std::cerr<<"FIRST_DIFF index="<<i<<" actual="<<actual.to(tb::kFloat32).data_ptr<float>()[i]<<" expected="<<expected[l].to(tb::kFloat32).data_ptr<float>()[i]<<std::endl;break;
                    }
                    throw std::runtime_error("resident/cached-transfer dispatch changed output bits");
                }
            }
            if(generation_check || gu_generation) {
                MfqCudaGraph graph;mfq_prepare_cuda_graph_memory(graph);graph.capture_begin();
                for(auto& call:calls)call->run();graph.capture_end();
                graph.replay();fb::check(cudaStreamSynchronize(mfq_current_cuda_stream()));
                for(auto& call:calls)call->change_cached_address();
                graph.replay();fb::check(cudaStreamSynchronize(mfq_current_cuda_stream()));
                for(size_t l=0;l<calls.size();++l) {
                    const auto actual=calls[l]->call.fused->output().cpu();
                    if(std::memcmp(actual.data_ptr(),expected[l].data_ptr(),actual.nbytes()))
                        throw std::runtime_error("new cache generation reused an overwritten field pointer");
                }
                std::cout<<"cached_generation mode="<<mode<<" replay_new_address_old_payload_zeroed exact PASS"<<std::endl;
                continue;
            }
            if(prepare_gate || parity_read) {
                MfqCudaGraph graph;mfq_prepare_cuda_graph_memory(graph);graph.capture_begin();
                for(auto& call:calls)call->run();graph.capture_end();
                int checks=0;
                for(int step=0;step<(prequantized_gu?6:4);++step) {
                    std::vector<tb::Tensor> references;
                    for(size_t l=0;l<calls.size();++l) {
                        auto values=fb::input(tokens,hidden);
                        for(auto& value:values)value*=step==0?0.f:step==1?-1.f:step==2?1.25f:.375f;
                        const auto input=tb::tensor(values).reshape({tokens,hidden}).to(tb::kCUDA,tb::kFloat16);
                        calls[l]->call.x.copy_(input);
                        std::vector<int32_t> selected(tokens*routes);
                        for(int i=0;i<tokens*routes;++i) {
                            selected[i]=layers[l]->selected[step>=4?(i/2+step)%routes:i%routes];
                            if(step==5 && i%3==0)selected[i]=-1;
                            if(step==5 && i%3==1)selected[i]=experts;
                        }
                        const auto ids=tb::tensor(selected).reshape({tokens,routes}).to(tb::kCUDA);
                        calls[l]->call.ids.copy_(ids);
                        auto* base=static_cast<uint8_t*>(calls[l]->host.data());
                        const auto* kinds=reinterpret_cast<const int32_t*>(base+calls[l]->kind_offset);
                        const auto* indices=reinterpret_cast<const int32_t*>(base+calls[l]->index_offset);
                        auto* packet=reinterpret_cast<int32_t*>(base+calls[l]->packet_offset);
                        for(int i=0;i<tokens*routes;++i) {
                            const int e=selected[i];
                            packet[i*2]=unsigned(e)<unsigned(experts)?kinds[e]:-1;
                            packet[i*2+1]=unsigned(e)<unsigned(experts)?indices[e]:-1;
                        }
                        if(parity_read)_putenv_s("MFQ_MFE_PARITY_READ","0");
                        Call reference(*layers[l],shared,tokens);reference.x.copy_(input);reference.ids.copy_(ids);
                        if(shared)reference.shared_gate.copy_(tb::sigmoid(layers[l]->shared_router.forward(reference.execution,reference.x)));
                        compact_runtime(reference);references.push_back(reference.fused->output().cpu());
                    }
                    if(parity_read)_putenv_s("MFQ_MFE_PARITY_READ","1");
                    graph.replay();fb::check(cudaStreamSynchronize(mfq_current_cuda_stream()));
                    for(size_t l=0;l<calls.size();++l) {
                        const auto actual=calls[l]->call.fused->output().cpu();
                        if(std::memcmp(actual.data_ptr(),references[l].data_ptr(),actual.nbytes()))
                            throw std::runtime_error("fused scalar gate changed output on reused graph: M="+std::to_string(tokens)+" step="+std::to_string(step)+" layer="+std::to_string(layers[l]->index));
                        if(transfer_fused && tokens==1) {
                            const auto& runtime=*calls[l]->call.fused;
                            const auto packed=runtime.cached_transfer_routes().cpu();
                            const auto ids=calls[l]->call.ids.cpu();
                            std::vector<int32_t> kinds(experts),expected_routes;
                            fb::check(cudaMemcpy(kinds.data(),runtime.batch().kinds,experts*4,cudaMemcpyDeviceToHost));
                            for(int r=0;r<routes;++r) {
                                const int e=ids.data_ptr<int32_t>()[r];
                                if(unsigned(e)<unsigned(experts) && kinds[e]==2)expected_routes.push_back(r);
                            }
                            const auto* list=packed.data_ptr<int32_t>();
                            if(list[0]!=int(expected_routes.size()) ||
                                    !std::equal(expected_routes.begin(),expected_routes.end(),list+1))
                                throw std::runtime_error("cold route list lost an original slot, retained an early GU, or reused a stale count");
                        }
                        ++checks;
                    }
                }
                std::cout<<(parity_read?"parity_read":"prepare_gate")<<" tokens="<<tokens<<" shared="<<shared<<" mode="<<mode<<" changed_input_replays="<<checks<<" exact PASS"<<std::endl;
                continue;
            }
            if(cached_check || cached_gu)continue;
            if(inspect) {
                for(auto& item:calls) {
                    const auto& b=item->call.fused->batch();
                    std::cout<<"CONTRACT layer="<<item->call.layer.index<<" tokens="<<tokens<<" shared="<<shared
                        <<" mode="<<mode<<" compact="<<b.compact_nvq<<" default_math="<<b.default_math
                        <<" compact_gu="<<mfq::cuda::mfe_ffn_compact_requested(b,false)
                        <<" compact_down="<<mfq::cuda::mfe_ffn_compact_requested(b,true)
                        <<" e8_narrow="<<b.e8_narrow<<" mask="<<b.nvq_format_mask
                        <<" context_allocated="<<context->memory_stats().allocated<<std::endl;
                }
                continue;
            }
            if(profile) {
                const char* names[]={"prepare","resident","gu_wait","transferred_gu","down_wait","down_reduce"};
                for(int phase=0;phase<6;++phase) {
                    MfqCudaGraph graph;mfq_prepare_cuda_graph_memory(graph);graph.capture_begin();
                    constexpr int repeats=16;
                    for(int i=0;i<int(calls.size())*repeats;++i)calls[i%calls.size()]->phase(phase);
                    graph.capture_end();graph.replay();fb::check(cudaStreamSynchronize(mfq_current_cuda_stream()));
                    const auto samples=fb::measure([&]{graph.replay();},stream.stream(),calls.size()*repeats);
                    std::cout<<std::setprecision(10)<<"{\"mode\":"<<mode<<",\"phase\":\""<<names[phase]
                        <<"\",\"us\":"<<fb::median(samples)<<",\"graph_nodes\":"<<graph.nodes()<<"}"<<std::endl;
                }
                continue;
            }
            MfqCudaGraph graph;mfq_prepare_cuda_graph_memory(graph);
            auto timestamps=in_graph || cached_profile?tb::empty({int64_t(calls.size()*7)},calls[0]->call.x.options().dtype(tb::kInt64)):tb::Tensor{};
            constexpr int repeats=16;
            if(gate_bench && mode==3) {
                for(int i=0;i<int(calls.size())*repeats;++i)calls[i%calls.size()]->run();
                fb::check(cudaStreamSynchronize(mfq_current_cuda_stream()));
                for(auto& call:calls)call->call.fused->bind_shared_gate({});
            }
            graph.capture_begin();
            for(int i=0;i<int(calls.size())*repeats;++i) {
                if((in_graph || cached_profile) && i<int(calls.size())) {
                    for(int stage=0;stage<6;++stage) {
                        mfq::cuda::record_gpu_timestamp(reinterpret_cast<uint64_t*>(timestamps.data_ptr()),i*7+stage,stream.stream());
                        calls[i]->phase(stage);
                    }
                    mfq::cuda::record_gpu_timestamp(reinterpret_cast<uint64_t*>(timestamps.data_ptr()),i*7+6,stream.stream());
                }else calls[i%calls.size()]->run();
            }
            graph.capture_end();
            graph.replay();fb::check(cudaStreamSynchronize(mfq_current_cuda_stream()));
            const auto samples=fb::measure([&]{graph.replay();},stream.stream(),calls.size()*repeats);
            if(in_graph || cached_profile) {
                const auto values=timestamps.cpu();const auto* data=reinterpret_cast<const uint64_t*>(values.data_ptr());
                const char* names[]={"prepare","resident","gu_wait","transferred_gu","down_wait","down_reduce"};
                for(int stage=0;stage<6;++stage) {
                    double ns=0;for(size_t l=0;l<calls.size();++l)ns+=double(data[l*7+stage+1]-data[l*7+stage]);
                    std::cout<<"IN_GRAPH mode="<<mode<<" phase="<<names[stage]<<" us="<<ns/calls.size()/1000.<<std::endl;
                }
            }
            std::cout<<std::setprecision(10)<<"{\"tokens\":"<<tokens<<",\"shared\":"<<int(shared)<<",\"mode\":"<<mode
                <<",\"prepare_gate_fused\":"<<int(gate_bench && mode==4)
                <<",\"experts\":10,\"layers\":6,\"us\":"<<fb::median(samples)<<",\"graph_nodes\":"<<graph.nodes()
                <<",\"streamed_bytes_per_layer\":"<<total_bytes/layers.size()<<",\"output_exact\":true}"<<std::endl;
        }
    }
    std::cout<<"cached_dispatch 10 real experts, six layers, M1/2/6, shared, resident/mixed/cache-only exact PASS"<<std::endl;
    return 0;
}catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<std::endl;return 1;}
