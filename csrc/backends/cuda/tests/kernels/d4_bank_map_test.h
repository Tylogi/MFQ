#pragma once
#include "native/env.h"
#include "ops/moe/mfe_ffn_runtime.h"

// Compare descriptor-cached bank selection with the original metadata load in
// complete production FFNs, including grouped routed tokens and graph replay.
void verify_d4_bank_map(const std::filesystem::path& root,int benchmark=1) {
    struct Option {
        std::string old;bool present=mfq::cuda::env_lookup("MFQ_MFE_NVQ_BANK_MAP")!=nullptr;
        Option(){if(present)old=mfq::cuda::env_lookup("MFQ_MFE_NVQ_BANK_MAP");}
        static void set(const char* value) {
#ifdef _WIN32
            _putenv_s("MFQ_MFE_NVQ_BANK_MAP",value);
#else
            if(*value)setenv("MFQ_MFE_NVQ_BANK_MAP",value,1);else unsetenv("MFQ_MFE_NVQ_BANK_MAP");
#endif
        }
        ~Option(){set(present?old.c_str():"");}
    } option;
    CudaExecutionContext execution;
    auto context=mfq::cuda::default_context(mfq_current_cuda_device());
    const auto stream=mfq_current_cuda_stream();context->begin_graph_pool(stream);
    const auto sigmoid=moe_swiglu_sigmoid_table_cuda();int checks=0;
    for(const char* format:{"nvq3j","nvq3j-512","nvq3j-l"}) {
        std::array<MixedMoeRuntime,3> projections;
        for(int p=0;p<3;++p) {
            auto model=mfq::open_model_source((root/(std::string(format)+(p==2?"-2560-640.mfq":"-640-2560.mfq"))).string());
            const auto encoded=load_mfe_cpu(*model,"linear.weight");
            auto gpu=make_mixed_moe_runtime(encoded,true,execution.config);
            auto cpu=make_mixed_moe_runtime(encoded,false,execution.config);
            for(std::size_t c=0;c<cpu->pools.size();++c) {
                auto weight=cpu->pools[c].nvq;
                if(!weight.bank_map || !prepare_nvq_dense_groups(weight))throw std::runtime_error("D4 bank fixture lacks packed metadata");
                weight.indices_packed=weight.indices_packed.to(tb::kCUDA);
                weight.aux_packed=weight.aux_packed.to(tb::kCUDA);
                weight.sub_scale_packed=weight.sub_scale_packed.to(tb::kCUDA);
                weight.neuron_scale=weight.neuron_scale.to(tb::kCUDA);
                weight.codebook=weight.codebook.to(tb::kCUDA);
                gpu->pools[c].nvq=std::move(weight);
            }
            projections[p]=std::move(*gpu);
        }
        for(int tokens:{1,2,3,4,6,8}) {
            if(benchmark==2 && tokens!=1 && tokens!=3 && tokens!=8)continue;
            std::vector<float> values(tokens*2560),weights(tokens*10,.1f);
            std::vector<int32_t> routes(tokens*10);
            for(std::size_t i=0;i<values.size();++i)values[i]=std::sin(float(i)*.017f)*.13f;
            for(int i=0;i<tokens*10;++i)routes[i]=i%3;
            auto x=tb::tensor(values).reshape({tokens,2560}).to(tb::kFloat16).to(tb::kCUDA),initial=x.clone();
            auto ids=tb::tensor(routes).reshape({tokens,10}).to(tb::kCUDA);
            auto route_weights=tb::tensor(weights).reshape({tokens,10}).to(tb::kCUDA);
            std::unique_ptr<MfeFfnRuntime> runtime[2];
            std::unique_ptr<mfq::cuda::DecodeWindow> graph[2];
            for(int mode=0;mode<2;++mode) {
                Option::set(mode?"1":"0");
                runtime[mode]=std::make_unique<MfeFfnRuntime>(
                    std::array<MixedMoeRuntime*,3>{&projections[0],&projections[1],&projections[2]},x,ids,route_weights,sigmoid);
                auto& r=*runtime[mode];r.run();const auto& batch=r.batch();
                if(!batch.compact_nvq)throw std::runtime_error("D4 bank fixture missed production compact dispatch");
                const int count=batch.projection_counts[0]+batch.projection_counts[1]+batch.projection_counts[2];
                std::vector<mfq::cuda::MfePackedProjection> descriptors(count);
                MFQ_CUDA_CHECK(cudaMemcpyAsync(descriptors.data(),batch.projections,count*sizeof(descriptors[0]),cudaMemcpyDeviceToHost,stream));
                MFQ_CUDA_CHECK(cudaStreamSynchronize(stream));
                for(const auto& d:descriptors)if(d.family==2 && bool(d.nvq_bank_map)!=bool(mode))
                    throw std::runtime_error("D4 bank map did not follow descriptor option");
                graph[mode]=std::make_unique<mfq::cuda::DecodeWindow>(stream);
                graph[mode]->capture([&r]{r.run();},[]{},[]{});
            }
            for(int step=0;step<3;++step) {
                x.copy_(initial*float(step+1));
                for(int i=0;i<tokens*10;++i)routes[i]=step==2 && i%7==0?-1:(i+step)%3;
                ids.copy_(tb::tensor(routes).reshape({tokens,10}).to(tb::kCUDA));
                graph[0]->run();graph[1]->run();
                auto a=runtime[0]->output().cpu().contiguous(),b=runtime[1]->output().cpu().contiguous();
                if(std::memcmp(a.data_ptr(),b.data_ptr(),a.nbytes()))throw std::runtime_error("D4 bank map changed output bits");
                a=runtime[0]->hidden().cpu().contiguous();b=runtime[1]->hidden().cpu().contiguous();
                const auto stride=runtime[0]->batch().hidden_stride;
                for(int i=0;i<tokens*10;++i)if(routes[i]>=0 && std::memcmp(
                    a.data_ptr<mfq_half>()+i*stride,b.data_ptr<mfq_half>()+i*stride,640*sizeof(mfq_half)))
                    throw std::runtime_error("D4 bank map changed rounded hidden bits");
                ++checks;
            }
            if(!benchmark)continue;
            if(benchmark==2) {
                for(int stage=0;stage<2;++stage) {
                    mfq::cuda::Graph part[2];std::vector<float> timing[2];
                    for(int mode=0;mode<2;++mode) {
                        auto& r=*runtime[mode];
                        const auto compute=[&]{if(!stage){r.prepare();r.gate_up();}else r.down_reduce();};
                        {mfq::cuda::GraphWarmupScope warmup(context,stream);compute();}
                        mfq_cuda_synchronize();part[mode].capture_begin_shared();compute();part[mode].capture_end();
                    }
                    if(part[0].nodes()!=part[1].nodes())throw std::runtime_error("D4 bank stage graph node count differs");
                    for(int sample=0;sample<9;++sample)for(int order=0;order<2;++order) {
                        const int mode=(sample+order)%2;
                        mfq::cuda::Event begin(cudaEventDefault),end(cudaEventDefault);begin.record(stream);
                        for(int i=0;i<64;++i)part[mode].replay();end.record(stream);end.synchronize();
                        float ms=0;MFQ_CUDA_CHECK(cudaEventElapsedTime(&ms,begin.get(),end.get()));
                        if(sample)timing[mode].push_back(ms*1000/64);
                    }
                    for(auto& values:timing)std::sort(values.begin(),values.end());
                    std::cout<<"d4_bank_gpu_stage format="<<format<<" tokens="<<tokens<<" stage="<<(stage?"down":"gu")<<" nodes="<<part[0].nodes()
                        <<" load_us="<<timing[0][4]<<" cached_us="<<timing[1][4]<<" load_samples=";
                    for(auto v:timing[0])std::cout<<v<<',';
                    std::cout<<" cached_samples=";for(auto v:timing[1])std::cout<<v<<',';
                    std::cout<<'\n';
                }
                continue;
            }
            mfq::cuda::Graph raw[2];
            for(int mode=0;mode<2;++mode) {
                {mfq::cuda::GraphWarmupScope warmup(context,stream);runtime[mode]->run();}
                mfq_cuda_synchronize();raw[mode].capture_begin_shared();runtime[mode]->run();raw[mode].capture_end();
            }
            if(raw[0].nodes()!=raw[1].nodes())throw std::runtime_error("D4 bank FFN graph node count differs");
            std::vector<float> timing[2];
            for(int sample=0;sample<7;++sample)for(int order=0;order<2;++order) {
                const int mode=(sample+order)%2;
                mfq::cuda::Event begin(cudaEventDefault),end(cudaEventDefault);begin.record(stream);
                for(int i=0;i<32;++i)raw[mode].replay();end.record(stream);end.synchronize();
                float ms=0;MFQ_CUDA_CHECK(cudaEventElapsedTime(&ms,begin.get(),end.get()));
                if(sample)timing[mode].push_back(ms*1000/32);
            }
            for(auto& values:timing)std::sort(values.begin(),values.end());
            std::cout<<"d4_bank_gpu_map format="<<format<<" tokens="<<tokens<<" nodes="<<raw[0].nodes()<<" load_us="<<timing[0][3]
                <<" cached_us="<<timing[1][3]<<" output_hidden_bits_equal=1 load_samples=";
            for(auto v:timing[0])std::cout<<v<<',';
            std::cout<<" cached_samples=";for(auto v:timing[1])std::cout<<v<<',';
            std::cout<<'\n';
        }
    }
    std::cout<<"D4 cached bank map changing_graph="<<checks<<" PASS\n";
    mfq_cuda_synchronize();context->end_graph_pool(stream);
}
