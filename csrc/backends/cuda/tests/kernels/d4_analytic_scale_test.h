#pragma once
#include "native/env.h"

void verify_d4_analytic_scales(const std::filesystem::path& root,bool benchmark=true) {
    MfeWideOption capacity("MFQ_MFE_TOKEN_GROUP_CAPACITY","2");
    CudaExecutionContext execution;
    const auto context=mfq::cuda::default_context(mfq_current_cuda_device());
    const auto stream=mfq_current_cuda_stream();context->begin_graph_pool(stream);
    const auto sigmoid=moe_swiglu_sigmoid_table_cuda();int checks=0;
    for(const char* format:{"nvq3j-512","nvq3j-l"}) {
        std::array<MixedMoeRuntime,3> projections[2];
        for(int p=0;p<3;++p) {
            auto source=mfq::open_model_source((root/(std::string(format)+(p==2?"-2560-640.mfq":"-640-2560.mfq"))).string());
            const auto encoded=load_mfe_cpu(*source,"linear.weight");
            auto gpu=make_mixed_moe_runtime(encoded,true,execution.config);
            auto cpu=make_mixed_moe_runtime(encoded,false,execution.config);
            if(cpu->pools.size()!=1)throw std::runtime_error("analytic scale fixture requires a single pool");
            auto weight=cpu->pools[0].nvq;
            if(!prepare_nvq_dense_groups(weight))throw std::runtime_error("analytic scale fixture missed compact representation");
            // Existing synthetic fixtures use learned four-bank tables. Build
            // a legal two-bank analytic table from their first two codebooks;
            // compressed indices and signs stay unchanged. Real S2-L metadata
            // is checked separately by the header-only model inspector.
            const int bank_bytes=(std::string(format)=="nvq3j-512"?512:1024)*4;
            auto table=weight.codebook.narrow(0,0,64+2*bank_bytes).clone();
            auto* metadata=table.data_ptr<int8_t>();metadata[1]=2;metadata[3]=1;
            for(int state=0;state<16;++state) {
                const mfq_half alpha(float((state>>1)+1));
                std::memcpy(metadata+4+state*2,&alpha,sizeof(alpha));metadata[36+state]=int8_t(state&1);
            }
            weight.bank_map=0x44444444u;
            weight.indices_packed=weight.indices_packed.to(tb::kCUDA);
            weight.aux_packed=weight.aux_packed.to(tb::kCUDA);
            weight.sub_scale_packed=weight.sub_scale_packed.to(tb::kCUDA);
            weight.neuron_scale=weight.neuron_scale.to(tb::kCUDA);
            for(int mode=0;mode<2;++mode) {
                auto header=table.clone();header.data_ptr<int8_t>()[3]=int8_t(mode);
                // Mode 0 is legal with the same exact LUT and bank mapping.
                // Only the tiny metadata/codebook allocation differs.
                weight.codebook=header.to(tb::kCUDA);
                auto copy=*gpu;copy.pools[0].nvq=weight;
                projections[mode][p]=std::move(copy);
            }
        }
        for(int tokens:{1,2,3,4,6,8}) {
            if(const auto* minimum=mfq::cuda::env_lookup("MFQ_D4_SCALE_MIN_TOKENS");minimum && tokens<std::atoi(minimum))continue;
            if(!benchmark && tokens!=1 && tokens!=3 && tokens!=8)continue;
            capacity.set(tokens>=6?"8":tokens>=3?"4":"2");
            std::vector<float> values(tokens*2560),weights(tokens*10,.1f);
            std::vector<int32_t> routes(tokens*10);
            for(std::size_t i=0;i<values.size();++i)values[i]=std::sin(float(i)*.017f)*.13f;
            for(int i=0;i<tokens*10;++i)routes[i]=i%3;
            auto x=tb::tensor(values).reshape({tokens,2560}).to(tb::kFloat16).to(tb::kCUDA),initial=x.clone();
            auto ids=tb::tensor(routes).reshape({tokens,10}).to(tb::kCUDA);
            auto route_weights=tb::tensor(weights).reshape({tokens,10}).to(tb::kCUDA);
            std::unique_ptr<MfeFfnRuntime> runtime[2];mfq::cuda::Graph graph[2];
            for(int mode=0;mode<2;++mode) {
                auto& p=projections[mode];
                runtime[mode]=std::make_unique<MfeFfnRuntime>(
                    std::array<MixedMoeRuntime*,3>{&p[0],&p[1],&p[2]},x,ids,route_weights,sigmoid);
                auto& r=*runtime[mode];r.run();
                if(!r.batch().compact_nvq || (tokens>1 && r.batch().token_group_capacity!=(tokens>=6?8:tokens>=3?4:2)))
                    throw std::runtime_error("analytic scale fixture missed the intended grouped decoder");
                {mfq::cuda::GraphWarmupScope warmup(context,stream);r.run();}
                mfq_cuda_synchronize();graph[mode].capture_begin_shared();r.run();graph[mode].capture_end();
            }
            if(graph[0].nodes()!=graph[1].nodes())throw std::runtime_error("analytic scale graph scheduling changed");
            for(int step=0;step<3;++step) {
                x.copy_(initial*float(step+1));
                for(int i=0;i<tokens*10;++i)routes[i]=step==2 && i%7==0?-1:(i+step)%3;
                ids.copy_(tb::tensor(routes).reshape({tokens,10}).to(tb::kCUDA));
                graph[0].replay();graph[1].replay();
                auto a=runtime[0]->output().cpu().contiguous(),b=runtime[1]->output().cpu().contiguous();
                if(std::memcmp(a.data_ptr(),b.data_ptr(),a.nbytes()))throw std::runtime_error("analytic D4 scale changed output bits");
                a=runtime[0]->hidden().cpu().contiguous();b=runtime[1]->hidden().cpu().contiguous();
                const auto stride=runtime[0]->batch().hidden_stride;
                for(int i=0;i<tokens*10;++i)if(routes[i]>=0 && std::memcmp(
                    a.data_ptr<mfq_half>()+i*stride,b.data_ptr<mfq_half>()+i*stride,640*sizeof(mfq_half)))
                    throw std::runtime_error("analytic D4 scale changed rounded hidden bits");
                ++checks;
            }
            if(!benchmark)continue;
            std::vector<float> timing[2];
            for(int sample=0;sample<9;++sample)for(int order=0;order<2;++order) {
                const int mode=(sample+order)%2;
                mfq::cuda::Event begin(cudaEventDefault),end(cudaEventDefault);begin.record(stream);
                for(int i=0;i<64;++i)graph[mode].replay();end.record(stream);end.synchronize();
                float ms=0;MFQ_CUDA_CHECK(cudaEventElapsedTime(&ms,begin.get(),end.get()));
                if(sample)timing[mode].push_back(ms*1000/64);
            }
            std::cout<<"d4_analytic_scale format="<<format<<" tokens="<<tokens<<" nodes="<<graph[0].nodes();
            for(int mode=0;mode<2;++mode) {
                auto sorted=timing[mode];std::sort(sorted.begin(),sorted.end());
                std::cout<<(mode?" analytic_us=":" lut_us=")<<sorted[4]<<(mode?" analytic_samples=":" lut_samples=");
                for(auto v:timing[mode])std::cout<<v<<',';
            }
            std::cout<<'\n';
        }
    }
    std::cout<<"D4 analytic scales changing_graph="<<checks<<" exact PASS\n";
    mfq_cuda_synchronize();context->end_graph_pool(stream);
}
