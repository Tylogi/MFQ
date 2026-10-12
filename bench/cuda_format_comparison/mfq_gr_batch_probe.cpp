// Actual GR weights; the reference uses the service's unfused composition.
#define main single_row_gr_main
#include "mfq_gr_bench.cpp"
#undef main

namespace wl=mfq::cuda::weight_loader;

struct BatchGr {
    Bank weights;
    wl::ResidualLinear down,up,inject;
    BatchGr(CudaExecutionContext& execution,const mfq::ModelSource& model,int layer,const char* role)
        :weights(execution,model,layer,role) {
        const auto prefix="model.block."+std::to_string(layer)+"."+role+".mhc.";
        down=wl::residual_linear(execution,model,prefix+"pre.down.weight");
        up=wl::residual_linear(execution,model,prefix+"pre.up.weight");
        inject=wl::residual_linear(execution,model,prefix+"post.inject.weight");
    }
    std::vector<tb::Tensor> run(CudaExecutionContext& execution,const tb::Tensor& branch,
            const tb::Tensor& residual,const tb::Tensor& prior,bool after,bool fused) const {
        if(fused)return gated_residual_two_stage_cuda(branch,residual,prior,weights.norm,
            down.weight->nint,up.weight->nint,inject.weight->is_nint()?&inject.weight->nint:nullptr,
            4,1e-6,after,inject.weight->is_dense()?inject.weight->dense:tb::Tensor{},inject.prepared_right);
        if(after)return gated_residual_pre_after_projected(execution,branch,residual,prior,weights.norm,
            down.forward,up.forward,inject.forward,2560,4,1e-6,up.mixed,{},down.activated,inject.activated);
        return gated_residual_pre_projected(execution,residual,weights.norm,down.forward,up.forward,
            inject.forward,2560,4,1e-6,up.mixed,{},down.activated,inject.activated);
    }
};

int main(int argc,char** argv)try {
    std::cout<<std::unitbuf;
    if(argc!=3 && !(argc==5 && std::string(argv[3])=="--rows"))
        throw std::runtime_error("expected model-first-shard and --check/--bench [--rows N]");
    const bool verify=std::string(argv[2])=="--check";
    if(!verify && std::string(argv[2])!="--bench")throw std::runtime_error("unknown batch GR mode");
    // At 32 rows the old dispatcher selects prefill arithmetic. Correctness
    // compares its ordered FP32 alternative; timings retain old dispatch.
    if(verify) {
#ifdef _WIN32
        if(_putenv_s("MFQ_GR_PREFILL_MATMUL","0"))throw std::runtime_error("GR reference option");
#else
        if(setenv("MFQ_GR_PREFILL_MATMUL","0",1))throw std::runtime_error("GR reference option");
#endif
    }
    auto context=mfq::cuda::default_context(0);
    auto stream=mfq_get_stream_from_pool(false);MfqCudaStreamGuard guard(stream);
    auto model=mfq::open_model_source(argv[1]);CudaExecutionContext execution;
    std::vector<std::unique_ptr<BatchGr>> banks;
    for(int layer:{0,8,16,24,32,40})for(const char* role:{"attention","mlp"})
        banks.push_back(std::make_unique<BatchGr>(execution,*model,layer,role));
    context->begin_graph_pool(stream.stream());
    int cases=0,checks=0;
    const auto batches=argc==5?std::vector<int>{std::stoi(argv[4])}:
        verify?std::vector<int>{9,16,17,32}:std::vector<int>{16,32};
    for(int batch:batches)
    for(bool half:{false,true})for(bool after:{false,true}) {
        auto residual=tb::tensor(fb::input(batch,10240)).reshape({batch,1,10240})
            .to(tb::kCUDA,half?tb::kFloat16:tb::kFloat32);
        auto branch=tb::tensor(fb::input(batch,2560)).reshape({batch,1,2560}).to(tb::kCUDA,tb::kFloat16);
        auto prior=tb::tensor(fb::input(batch,4)).reshape({batch,1,4}).to(tb::kCUDA,residual.scalar_type());
        const auto source=residual.clone(),branch_source=branch.clone();
        std::array<std::vector<std::vector<tb::Tensor>>,2> outputs;
        std::array<MfqCudaGraph,2> graphs;context->begin_graph_warmup(stream.stream());
        const auto body=[&](int fused) {
            outputs[fused].clear();
            for(const auto& bank:banks)outputs[fused].push_back(bank->run(execution,branch,residual,prior,after,bool(fused)));
        };
        for(int fused=0;fused<2;++fused) {
            body(fused);fb::check(cudaStreamSynchronize(stream.stream()));
        }
        for(auto& output:outputs)output.clear();
        for(int fused=0;fused<2;++fused) {
            graphs[fused].capture_begin_shared();
            body(fused);graphs[fused].capture_end();
        }
        for(int step=0;step<(verify?3:1);++step) {
            residual.copy_((source*(step==0?0.:.25+.5*step)).to(residual.scalar_type()));
            branch.copy_((branch_source*(1.+.25*step)).to(branch.scalar_type()));
            for(int fused=0;fused<2;++fused) {
                graphs[fused].replay();
                fb::check(cudaStreamSynchronize(stream.stream()));
            }
            fb::check(cudaStreamSynchronize(stream.stream()));
            if(verify)for(size_t bank=0;bank<banks.size();++bank)for(size_t out=0;out<3;++out) {
                exact(outputs[1][bank][out],outputs[0][bank][out].cpu(),"continuous GR batch="+
                    std::to_string(batch)+" bank="+std::to_string(bank)+" output="+std::to_string(out));++checks;
            }
            ++cases;
        }
        if(!verify) {
            residual.copy_(source);branch.copy_(branch_source);
            std::array<std::vector<double>,2> times;
            for(int repeat=0;repeat<3;++repeat)for(int offset=0;offset<2;++offset) {
                const int candidate=(repeat+offset)%2;
                times[candidate].push_back(fb::median(fb::measure([&]{graphs[candidate].replay();},
                    stream.stream(),banks.size())));
            }
            std::cout<<"actual_gr_batch batch="<<batch<<" half="<<half<<" after="<<after
                <<" old_us="<<fb::median(times[0])<<" fused_us="<<fb::median(times[1])<<"\n"<<std::flush;
        }
    }
    context->end_graph_pool(stream.stream());
    std::cout<<"actual_gr_batch cases="<<cases<<" output_checks="<<checks
        <<(verify?" BYTE_EXACT PASS\n":" TIMING PASS\n");
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
