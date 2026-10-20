#pragma once
#include "kernels/mfe/ffn_launch.h"
#include "kernels/mfe/ffn_compact.h"

void verify_mfe_shape_dispatch(const std::filesystem::path& root) {
    MfeWideOption grouped("MFQ_MFE_TOKEN_GROUPS","0");
    MfeWideOption compact("MFQ_MFE_NVQ_COMPACT","0"),narrow("MFQ_MFE_E8_NARROW","0");
    CudaExecutionContext execution;
    const auto stream=mfq_current_cuda_stream();
    const auto context=mfq::cuda::default_context(mfq_current_cuda_device());
    context->begin_graph_pool(stream);
    const auto sigmoid=moe_swiglu_sigmoid_table_cuda();
    int cases=0,default_dispatches=0;
    for(const char* format:{"nvq2j-xl","nvq3j-l"})for(bool reverse:{false,true}) {
        const int width=reverse?640:2560,intermediate=reverse?2560:640;
        std::array<MixedMoeRuntime,3> projections;
        std::array<NintWeight,3> shared_weights;
        for(int p=0;p<3;++p) {
            const int output=p==2?width:intermediate,input=p==2?intermediate:width;
            const auto suffix="-"+std::to_string(output)+"-"+std::to_string(input)+".mfq";
            auto model=mfq::open_model_source((root/(std::string(format)+suffix)).string());
            const auto encoded=load_mfe_cpu(*model,"linear.weight");
            auto cpu=make_mixed_moe_runtime(encoded,false,execution.config);
            auto gpu=make_mixed_moe_runtime(encoded,true,execution.config);
            auto w=cpu->pools[0].nvq;
            if(!prepare_nvq_dense_groups(w))throw std::runtime_error("generic shape fixture cannot pack NVQ groups");
            for(auto* field:{&w.indices_packed,&w.aux_packed,&w.sub_scale_packed,&w.neuron_scale,&w.codebook})
                *field=field->to(tb::kCUDA);
            gpu->pools[0].nvq=std::move(w);projections[p]=std::move(*gpu);
            auto shared_model=mfq::open_model_source((root/("nint5"+suffix)).string());
            auto shared=make_mixed_moe_runtime(load_mfe_cpu(*shared_model,"linear.weight"),true,execution.config);
            auto& n=shared_weights[p];n=shared->pools[0].nint;n.out=output;n.shape={output,input};n.workspaces.clear();
            for(auto* field:{&n.row_q_bits,&n.row_q_bit_offsets,&n.sub_scale,&n.sub_min,&n.neuron_scale,&n.neuron_min})
                *field=field->narrow(0,0,output).contiguous();
        }
        for(bool shared:{false,true})for(int tokens:{1,3,9})for(int routes:{1,7,31})for(bool tail:{false,true}) {
            auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat16);
            auto x=tb::zeros({tokens,width},options);
            std::vector<int32_t> selected(tokens*routes);
            for(int i=0;i<tokens*routes;++i)selected[i]=i%9==8?-1:i%3;
            auto ids=tb::tensor(selected).reshape({tokens,routes}).to(tb::kCUDA);
            auto weights=tb::tensor(std::vector<float>(tokens*routes,1.f/routes)).reshape({tokens,routes}).to(tb::kCUDA);
            auto gate=shared?tb::full({tokens},.375,options):tb::Tensor{};
            std::array<const NintWeight*,3> shared_views{};
            if(shared)for(int p=0;p<3;++p)shared_views[p]=&shared_weights[p];
            std::array<tb::Tensor,2> expected;
            for(int mode=0;mode<2;++mode) {
                compact.set(mode?"":"0");narrow.set(mode?"":"0");
                MfeFfnRuntime fused({&projections[0],&projections[1],&projections[2]},x,ids,weights,
                    sigmoid,shared_views,gate,width-int(tail)*3,intermediate-int(tail)*2);
                if(mode) {
                    const auto& b=fused.batch();
                    if(!mfq::cuda::mfe_ffn_compact_requested(b,false) || !mfq::cuda::mfe_ffn_compact_requested(b,true))
                        throw std::runtime_error("generic compact default rejected legal matrix geometry");
                    ++default_dispatches;
                }
                tb::Tensor actual;
                mfq::cuda::DecodeWindow window(stream);
                window.capture([&]{fused.prepare();fused.gate_up();fused.down_reduce();actual=fused.output();},[]{},[&]{actual={};});
                for(int step=0;step<2;++step) {
                    std::vector<float> values(tokens*width);
                    for(std::size_t i=0;i<values.size();++i)values[i]=std::sin(float(i+step*37)*.017f)*.17f;
                    x.copy_(tb::tensor(values).reshape({tokens,width}).to(tb::kFloat16).to(tb::kCUDA));
                    window.run();auto result=actual.cpu().contiguous();
                    if(!mode)expected[step]=result.clone();
                    else {
                        if(result.sizes()!=expected[step].sizes() || result.scalar_type()!=expected[step].scalar_type() ||
                                std::memcmp(result.data_ptr(),expected[step].data_ptr(),result.numel()*result.element_size()))
                            throw std::runtime_error("generic compact shape changed output bytes: "+std::string(format)+
                                " width="+std::to_string(width)+" tokens="+std::to_string(tokens)+" routes="+std::to_string(routes));
                        ++cases;
                    }
                }
            }
        }
    }
    std::cout<<"mfe_generic_shape_cases="<<cases<<" default_dispatches="<<default_dispatches
        <<" E8/D4 reversed matrices, top1/7/31, M1/3/9, shared and row tails graph byte-exact PASS\n";
    MFQ_CUDA_CHECK(cudaStreamSynchronize(stream));context->end_graph_pool(stream);
}
