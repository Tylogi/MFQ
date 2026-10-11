// Reuse real weights, routes, and production runtime without modifying its API.
#define main mfq_mfe_bench_original_main
#include "mfq_mfe_bench.cpp"
#undef main

namespace mfq::cuda {
void mfe_gu_register_probe(const MfeFfnBatch&,cudaStream_t,int);
cudaFuncAttributes mfe_gu_register_probe_attributes(uint32_t,int);
}
static void probe_check_contract(Call& call) {
#ifdef MFQ_GU_COMPACT_ONLY
    const auto& b=call.fused->batch();
    const int count=b.projection_counts[0]+b.projection_counts[1]+b.projection_counts[2];
    std::vector<mfq::cuda::MfePackedProjection> views(count);
    fb::check(cudaMemcpyAsync(views.data(),b.projections,views.size()*sizeof(views[0]),
        cudaMemcpyDeviceToHost,mfq_current_cuda_stream()));
    fb::check(cudaStreamSynchronize(mfq_current_cuda_stream()));
    for(const auto& d:views)if(d.family==2) {
        if((d.format==5 || d.format==13 || d.format==14) && !d.e8_dense_groups)
            throw std::runtime_error("compact GU probe received canonical E8");
        if((d.format==10 || d.format==11 || d.format==12 || d.format==15) && !d.d4_dense_groups)
            throw std::runtime_error("compact GU probe received canonical D4");
    }
#endif
}
static void probe_run(Call& call,int variant) {
    if(variant<0){call.fused->run();return;}
    call.fused->prepare();
    mfq::cuda::mfe_gu_register_probe(call.fused->batch(),mfq_current_cuda_stream(),variant);
    call.fused->down_reduce();
}
static void probe_exact(const tb::Tensor& expected,const tb::Tensor& actual) {
    if(expected.sizes()!=actual.sizes() || expected.scalar_type()!=actual.scalar_type() ||
        std::memcmp(expected.data_ptr(),actual.data_ptr(),expected.numel()*expected.element_size()))
        throw std::runtime_error("GU register probe changed final output bits");
}
static void probe_verify(std::vector<std::unique_ptr<Layer>>& layers) {
    size_t final_checks=0,hidden_checks=0;
    for(int tokens:{1,3,8})for(bool shared:{false,true})for(auto& layer:layers) {
        Call call(*layer,shared,tokens);
        probe_check_contract(call);
        for(int variant=0;variant<3;++variant) {
            MfqCudaGraph graph;mfq_prepare_cuda_graph_memory(graph);
            probe_run(call,variant);fb::check(cudaStreamSynchronize(mfq_current_cuda_stream()));
            graph.capture_begin();probe_run(call,variant);graph.capture_end();
            for(int step=0;step<3;++step) {
                std::vector<int32_t> selected(tokens*routes);
                for(int i=0;i<tokens*routes;++i) {
                    selected[i]=layer->selected[(i/(step==1?2:1)+step)%routes];
                    if(step==2 && i%3==0)selected[i]=-1;
                    if(step==2 && i%3==1)selected[i]=experts;
                }
                call.ids.copy_(tb::tensor(selected).reshape({tokens,routes}).to(tb::kCUDA));
                auto values=fb::input(tokens,hidden);
                for(size_t i=0;i<values.size();++i)values[i]=step==0?0.f:
                    values[i]*float(step)+std::sin(float(i)*.019f+step)*.21f;
                call.x.copy_(tb::tensor(values).reshape({tokens,hidden}).to(tb::kCUDA,tb::kFloat16));
                probe_run(call,-1);
                auto expected=call.fused->output().cpu(),expected_hidden=call.fused->hidden().cpu();
                graph.replay();auto actual=call.fused->output().cpu(),actual_hidden=call.fused->hidden().cpu();
                probe_exact(expected,actual);++final_checks;
                const auto& b=call.fused->batch();
                for(int token=0;token<tokens;++token)for(int route=0;route<routes+int(shared);++route) {
                    if(route<routes && unsigned(selected[token*routes+route])>=unsigned(experts))continue;
                    const size_t slot=token*(routes+int(shared))+route;
                    const size_t offset=slot*b.hidden_stride*actual_hidden.element_size();
                    const size_t width=route==routes?b.shared_intermediate:b.intermediate;
                    if(std::memcmp(static_cast<const char*>(actual_hidden.data_ptr())+offset,
                        static_cast<const char*>(expected_hidden.data_ptr())+offset,width*actual_hidden.element_size()))
                        throw std::runtime_error("GU register probe changed active hidden bits");
                }
                ++hidden_checks;
            }
        }
    }
    std::cout<<"VERIFY_GU_REGISTER final="<<final_checks<<" hidden="<<hidden_checks<<" PASS\n"<<std::flush;
}
int main(int argc,char** argv)try {
    if(argc!=3)throw std::runtime_error("usage: mfe-gu-register-probe model-first-shard label");
    // These checks make the generated launcher's deliberately narrow contract explicit.
    for(const char* name:{"MFQ_MFE_DEFAULT_MATH","MFQ_MFE_E8_NARROW","MFQ_MFE_NVQ_DENSE_NARROW"}) {
        const char* value=std::getenv(name);
        if(!value || std::strcmp(value,"0"))throw std::runtime_error(std::string(name)+" must be 0");
    }
    auto context=mfq::cuda::default_context(0);auto stream=mfq_get_stream_from_pool(false);MfqCudaStreamGuard guard(stream);
    auto model=mfq::open_model_source(argv[1]);std::vector<std::unique_ptr<Layer>> layers;
    for(int layer:{0,8,16,24,32,40})layers.push_back(std::make_unique<Layer>(*model,layer));
    for(uint32_t mask:{0x3f820u,0x3f922u})for(int variant=0;variant<3;++variant) {
        const auto a=mfq::cuda::mfe_gu_register_probe_attributes(mask,variant);
        std::cout<<"GU_RESOURCE mask="<<mask<<" variant="<<variant<<" registers="<<a.numRegs
            <<" local_bytes="<<a.localSizeBytes<<" shared_bytes="<<a.sharedSizeBytes<<'\n';
    }
    if(std::getenv("MFQ_BENCH_GU_VERIFY")){probe_verify(layers);return 0;}
    cudaDeviceProp properties{};fb::check(cudaGetDeviceProperties(&properties,0));
    for(bool shared:{false,true}) {
        std::vector<std::unique_ptr<Call>> calls;size_t bytes=0;
        for(auto& layer:layers) {
            calls.push_back(std::make_unique<Call>(*layer,shared,1));calls.back()->verify(shared);
            probe_check_contract(*calls.back());
            bytes+=layer->fused_bytes[0]+layer->fused_bytes[1]+layer->fused_bytes[2]+(shared?layer->shared_bytes:0);
        }
        if(bytes<4*size_t(properties.l2CacheSize))throw std::runtime_error("probe working set too small");
        for(int variant:{-1,0,1,2,0,-1}) {
            MfqCudaGraph graph;mfq_prepare_cuda_graph_memory(graph);
            for(auto& call:calls)probe_run(*call,variant);
            fb::check(cudaStreamSynchronize(mfq_current_cuda_stream()));
            constexpr size_t repeats=16;
            graph.capture_begin();
            for(size_t i=0;i<calls.size()*repeats;++i)probe_run(*calls[i%calls.size()],variant);
            graph.capture_end();graph.replay();fb::check(cudaStreamSynchronize(mfq_current_cuda_stream()));
            auto samples=fb::measure([&](){graph.replay();},stream.stream(),calls.size()*repeats);
            const double us=fb::median(samples);
            std::cout<<std::setprecision(10)<<"{\"profile\":\""<<argv[2]<<"\",\"variant\":"<<variant
                <<",\"shared\":"<<(shared?"true":"false")<<",\"us\":"<<us
                <<",\"effective_GBs\":"<<double(bytes)/calls.size()/(us*1000.)<<",\"samples_us\":[";
            for(size_t i=0;i<samples.size();++i)std::cout<<(i?",":"")<<samples[i];
            std::cout<<"]}\n"<<std::flush;
        }
    }
    return 0;
}catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
