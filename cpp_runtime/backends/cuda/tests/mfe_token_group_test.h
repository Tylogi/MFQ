#pragma once
#include "mfe_ffn_runtime.h"
#include "mfe_expert_ffn_test.h"
#include <cstring>

void verify_mfe_token_groups(const std::filesystem::path& root) {
    struct Environment {
        std::string previous;
        Environment(){if(const auto* s=std::getenv("MFQ_MFE_TOKEN_GROUPS"))previous=s;}
        static void select(bool enabled) {
#ifdef _WIN32
            _putenv_s("MFQ_MFE_TOKEN_GROUPS",enabled?"1":"0");
#else
            setenv("MFQ_MFE_TOKEN_GROUPS",enabled?"1":"0",1);
#endif
        }
        ~Environment(){
#ifdef _WIN32
            _putenv_s("MFQ_MFE_TOKEN_GROUPS",previous.c_str());
#else
            if(previous.empty())unsetenv("MFQ_MFE_TOKEN_GROUPS");else setenv("MFQ_MFE_TOKEN_GROUPS",previous.c_str(),1);
#endif
        }
    } environment;
    CudaExecutionContext execution;
    auto context=mfq::cuda::default_context(mfq_current_cuda_device());
    const auto stream=mfq_current_cuda_stream();context->begin_graph_pool(stream);
    const auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat16);
    const auto sigmoid=moe_swiglu_sigmoid_table_cuda();
    int cases=0;
    for(int family:{0,1,2}) {
        const bool nvq=family!=0;
        std::array<MixedMoeRuntime,3> projections;
        const std::array<std::string,4> formats=family==2?
            std::array<std::string,4>{"nvq1-l","nvq1-s","nvq2j-xl","nvq3j-l"}:nvq?
            std::array<std::string,4>{"nvq2j-l","nvq2j-xl","nvq3j","nvq3j-l"}:
            std::array<std::string,4>{"nint4","nint5","nint6","nint8"};
        for(int p=0;p<3;++p) {
            auto& runtime=projections[p];runtime.n_experts=384;runtime.partial_experts=true;
            runtime.out_per_expert=p==2?2560:640;runtime.neuron_len=p==2?640:2560;
            for(int cohort=0;cohort<4;++cohort) {
                auto model=mfq::open_model_source((root/(formats[(cohort+p)%4]+(p==2?"-2560-640.mfq":"-640-2560.mfq"))).string());
                const auto encoded=load_mfe_cpu(*model,"linear.weight");
                auto single=make_mixed_moe_runtime(encoded,true,execution.config);
                if(nvq) {
                    auto cpu=make_mixed_moe_runtime(encoded,false,execution.config);
                    for(std::size_t i=0;i<cpu->pools.size();++i) {
                        auto weight=cpu->pools[i].nvq;
                        if(weight.kernel_format==1 || weight.kernel_format==8)continue;
                        if(!prepare_nvq_dense_groups(weight))throw std::runtime_error("token-group dense fixture not convertible");
                        weight.indices_packed=weight.indices_packed.to(tb::kCUDA);
                        weight.aux_packed=weight.aux_packed.to(tb::kCUDA);
                        weight.sub_scale_packed=weight.sub_scale_packed.to(tb::kCUDA);
                        weight.neuron_scale=weight.neuron_scale.to(tb::kCUDA);
                        weight.codebook=weight.codebook.to(tb::kCUDA);
                        single->pools[i].nvq=std::move(weight);
                    }
                }
                for(auto& pool:single->pools) {
                    // Each synthetic expert owns distinct compressed storage;
                    // the no-sharing benchmark must not alias twelve weights.
                    const auto repeat=[](const tb::Tensor& field) {
                        if(!field.numel())return field;
                        return tb::cat(std::vector<tb::Tensor>(32,field),0);
                    };
                    if(pool.family==MixedMoeFamily::Nint) {
                        auto& w=pool.nint;
                        const auto packed_bytes=w.q_packed.numel();
                        if(!w.q_expert_stride) {
                            std::vector<tb::Tensor> offsets;
                            for(int copy=0;copy<32;++copy)offsets.push_back(w.row_q_bit_offsets+int64_t(copy)*packed_bytes*8);
                            w.row_q_bit_offsets=tb::cat(offsets,0);
                        }else w.row_q_bit_offsets=repeat(w.row_q_bit_offsets);
                        w.q_packed=repeat(w.q_packed);w.row_q_bits=repeat(w.row_q_bits);
                        w.sub_scale=repeat(w.sub_scale);w.sub_min=repeat(w.sub_min);
                        w.neuron_scale=repeat(w.neuron_scale);w.neuron_min=repeat(w.neuron_min);w.out*=32;
                    }else {
                        auto& w=pool.nvq;
                        w.indices_packed=repeat(w.indices_packed);w.aux_packed=repeat(w.aux_packed);
                        w.sub_scale_packed=repeat(w.sub_scale_packed);w.neuron_scale=repeat(w.neuron_scale);
                        if(w.decode_records.defined())w.decode_records=repeat(w.decode_records);
                        w.out*=32;
                    }
                    pool.local_experts=96;
                    std::vector<int32_t> local(384,-1);
                    for(int e=0;e<96;++e)local[cohort*96+e]=e;
                    pool.expert_local=tb::tensor(local).to(tb::kCUDA);runtime.pools.push_back(std::move(pool));
                }
            }
        }
        for(int tokens:{1,2,3,4,5,6,8,16,32})for(int pattern:{0,1,2})for(bool shared:{false,true}) {
            std::vector<float> values(tokens*2560),weights(tokens*10);
            std::vector<int32_t> ids(tokens*10);
            for(std::size_t i=0;i<values.size();++i)values[i]=std::sin(float(i+pattern*31)*.017f)*.13f;
            for(int t=0;t<tokens;++t)for(int r=0;r<10;++r) {
                const int i=t*10+r;
                // Full sharing, partial sharing, and no shared route IDs.
                const int logical=pattern==0?r:pattern==1?(r+t*3)%16:i;
                ids[i]=(logical%4)*96+logical/4;
                if(pattern==1 && r==9)ids[i]=t%2?-1:384;
                weights[i]=float((i*7+pattern*3)%17+1)/37;
            }
            auto x=tb::tensor(values).reshape({tokens,2560}).to(tb::kFloat16).to(tb::kCUDA);
            auto selected=tb::tensor(ids).reshape({tokens,10}).to(tb::kCUDA);
            auto route_weights=tb::tensor(weights).reshape({tokens,10}).to(tb::kCUDA);
            std::array<NintWeight,3> shared_weights;
            std::array<const NintWeight*,3> shared_views{};
            if(shared)for(int p=0;p<3;++p) {
                const std::array<const char*,3> roles={"gate","up","down"};
                std::ifstream stream(root/(std::string("shared-s2-")+roles[p]+".nint"),std::ios::binary);
                if(!stream)throw std::runtime_error("missing shared expert-group fixture");
                const std::vector<uint8_t> blob((std::istreambuf_iterator<char>(stream)),std::istreambuf_iterator<char>());
                shared_weights[p]=to_gpu_nint(unpack_nint(blob));shared_views[p]=&shared_weights[p];
            }
            auto shared_gate=shared?tb::sigmoid(x.mean(1)):tb::Tensor{};
            MfeFfnRuntime runtime({&projections[0],&projections[1],&projections[2]},x,selected,route_weights,
                sigmoid,shared_views,shared_gate,2560,640,nullptr,false,false,tokens==8?8:0);
            if(tokens==8 && runtime.batch().token_group_capacity!=8)
                throw std::runtime_error("independent decode hint did not prepare eight-row expert groups");
            std::array<double,2> elapsed{};
            std::array<std::array<double,2>,2> stages{};
            tb::Tensor expected,expected_hidden;
            for(int mode=0;mode<2;++mode) {
                Environment::select(mode!=0);
                runtime.run();auto actual=runtime.output().cpu().contiguous();
                if(!mode){expected=actual.clone();expected_hidden=runtime.hidden().cpu().contiguous().clone();}
                else if(actual.sizes()!=expected.sizes() || actual.scalar_type()!=expected.scalar_type() ||
                    std::memcmp(actual.data_ptr(),expected.data_ptr(),actual.numel()*actual.element_size()))
                    throw std::runtime_error("expert-group output differs: family="+std::to_string(family)+
                        " M="+std::to_string(tokens)+" pattern="+std::to_string(pattern)+" shared="+std::to_string(shared));
                if(mode) {
                    auto hidden=runtime.hidden().cpu().contiguous();
                    if(std::memcmp(hidden.data_ptr(),expected_hidden.data_ptr(),hidden.numel()*hidden.element_size()))
                        throw std::runtime_error("token-group rounded SwiGLU differs");
                }
                mfq::cuda::DecodeWindow window(mfq_current_cuda_stream());
                window.capture([&]{runtime.run();},[]{},[]{});
                for(int i=0;i<3;++i)window.run();mfq_cuda_synchronize();
                cudaEvent_t begin,end;MFQ_CUDA_CHECK(cudaEventCreate(&begin));MFQ_CUDA_CHECK(cudaEventCreate(&end));
                MFQ_CUDA_CHECK(cudaEventRecord(begin,mfq_current_cuda_stream()));
                for(int i=0;i<20;++i)window.run();
                MFQ_CUDA_CHECK(cudaEventRecord(end,mfq_current_cuda_stream()));MFQ_CUDA_CHECK(cudaEventSynchronize(end));
                float milliseconds=0;MFQ_CUDA_CHECK(cudaEventElapsedTime(&milliseconds,begin,end));
                cudaEventDestroy(begin);cudaEventDestroy(end);elapsed[mode]=milliseconds/20;
                window.run();actual=runtime.output().cpu().contiguous();
                if(std::memcmp(actual.data_ptr(),expected.data_ptr(),actual.numel()*actual.element_size()))
                    throw std::runtime_error("expert-group graph replay differs");
                for(int stage=0;stage<2;++stage) {
                    mfq::cuda::DecodeWindow part(mfq_current_cuda_stream());
                    part.capture([&]{if(!stage){runtime.prepare();runtime.gate_up();}else runtime.down_reduce();},[]{},[]{});
                    for(int i=0;i<3;++i)part.run();mfq_cuda_synchronize();
                    MFQ_CUDA_CHECK(cudaEventCreate(&begin));MFQ_CUDA_CHECK(cudaEventCreate(&end));
                    MFQ_CUDA_CHECK(cudaEventRecord(begin,mfq_current_cuda_stream()));
                    for(int i=0;i<20;++i)part.run();
                    MFQ_CUDA_CHECK(cudaEventRecord(end,mfq_current_cuda_stream()));MFQ_CUDA_CHECK(cudaEventSynchronize(end));
                    MFQ_CUDA_CHECK(cudaEventElapsedTime(&milliseconds,begin,end));
                    cudaEventDestroy(begin);cudaEventDestroy(end);stages[mode][stage]=milliseconds/20;
                }
            }
            std::cout<<"token_group family="<<family<<" M="<<tokens<<" sharing="<<pattern<<" shared="<<shared
                <<" old_ms="<<elapsed[0]<<" grouped_ms="<<elapsed[1]<<" speedup="<<elapsed[0]/elapsed[1]
                <<" old_gu_ms="<<stages[0][0]<<" grouped_gu_ms="<<stages[1][0]
                <<" old_down_ms="<<stages[0][1]<<" grouped_down_ms="<<stages[1][1]<<" exact=1\n";
            ++cases;
        }
    }
    std::cout<<"mfe_token_group_cases="<<cases<<" PASS\n";
    mfq_cuda_synchronize();context->end_graph_pool(stream);
}

void verify_mfe_expert_plan(const std::filesystem::path& root) {
    mfe_expert_plan_test::verify(false,mfq_current_cuda_stream());
    mfe_expert_plan_test::verify_cache(false,mfq_current_cuda_stream());
    mfe_expert_plan_test::verify_ffn(root,false);
}

void verify_mfe_expert_plan_pipeline(const std::filesystem::path& root) {
    mfe_expert_plan_test::Option option;
    const auto counter=[](const std::shared_ptr<MoeExpertCache>& cache,const char* name) {
        std::ostringstream output;
        print_moe_expert_cache_stats(cache,output);
        std::smatch match;
        const auto text=output.str();
        if(!std::regex_search(text,match,std::regex(std::string(name)+"=([0-9]+)")))
            throw std::runtime_error("missing expert-plan pipeline counter: "+std::string(name));
        return std::stoll(match[1]);
    };
    int cases=0;
    int64_t dma=0,hits=0,misses=0;
    for(const char* family:{"nint","nvq","mxfp4"})for(int tokens:{2,8,16,32})for(int routes:{8,22})
    for(bool graph:{false,true}) {
        const std::array<MfqCudaStream,2> streams={mfq_get_stream_from_pool(),mfq_get_current_cuda_stream()};
        std::array<CudaExecutionContext,2> executions;
        std::array<std::shared_ptr<MoeExpertCache>,2> caches;
        std::array<std::array<std::shared_ptr<MoeQuantRangeSource>,3>,2> sources;
        std::array<MoeFfnForward,2> forwards;
        std::array<mfq::MoeDispatchPlan,2> plans;
        for(int mode=0;mode<2;++mode) {
            MfqCudaStreamGuard guard(streams[mode]);
            auto& config=executions[mode].config;
            config.moe_two_stage_ffn=true;
            config.moe_preload_all=true;
            config.moe_assert_resident=true;
            config.moe_pipeline=true;
            config.moe_ram_pcie=true;
            config.moe_ram_pcie_fraction=1;
            config.moe_direct_ram=true;
            config.moe_residency_adapt=false;
            config.moe_residency_projection_heat=false;
            executions[mode].continuous_batch_cache_serial=tokens>8;
            std::size_t bundle=0;
            for(int p=0;p<3;++p) {
                sources[mode][p]=source(root/(std::string("plan-")+family+(p==2?"-down.mfq":"-up.mfq")));
                bundle+=sources[mode][p]->expert_field_bytes(0);
            }
            config.moe_host_cache_bytes=768*bundle;
            caches[mode]=make_moe_expert_cache(2*bundle,config);
            std::vector<std::shared_ptr<MfeWeight>> gu,down;
            for(int p=0;p<3;++p) {
                auto weight=std::make_shared<MfeWeight>(cache_quant_moe_weight(caches[mode],
                    std::to_string(p),sources[mode][p],1,0,std::to_string(p)));
                (p==2?down:gu).push_back(weight);
            }
            forwards[mode]=make_moe_ffn_pipeline(gu,down,{},
                [&,mode](const mfq::MoeDispatchPlan& plan){plans[mode]=plan;});
            finalize_moe_expert_cache(caches[mode]);
        }
        const auto options=tb::TensorOptions().device(tb::kCUDA);
        auto input=tb::full({tokens,384},.03125,options.dtype(tb::kFloat16));
        auto ids=tb::zeros({tokens,routes},options.dtype(tb::kInt32));
        auto weights=tb::full({tokens,routes},1.0/routes,options.dtype(tb::kFloat32));
        MFQ_CUDA_CHECK(cudaStreamSynchronize(streams[1]));
        std::array<tb::Tensor,2> outputs;
        std::array<std::unique_ptr<mfq::cuda::DecodeWindow>,2> windows;
        if(graph)for(int mode=0;mode<2;++mode) {
            MfqCudaStreamGuard guard(streams[mode]);
            mfe_expert_plan_test::Option::select(mode?"1":"0");
            windows[mode]=std::make_unique<mfq::cuda::DecodeWindow>(mfq_current_cuda_stream());
            windows[mode]->capture([&,mode]{outputs[mode]=forwards[mode](executions[mode],input,ids,weights);},
                []{},[&,mode]{outputs[mode]={};});
        }
        for(int step=0;step<4;++step) {
            std::vector<int32_t> selected(tokens*routes);
            for(int pair=0;pair<tokens*routes;++pair)
                selected[pair]=pair%routes==0?0:2+(pair%(step<2?routes:tokens*routes));
            ids.copy_(tb::tensor(selected).reshape({tokens,routes}).to(tb::kCUDA));
            MFQ_CUDA_CHECK(cudaStreamSynchronize(streams[1]));
            mfe_expert_plan_test::Option::select("0");
            tb::Tensor expected;
            {
                MfqCudaStreamGuard guard(streams[0]);
                if(graph)windows[0]->run();
                else outputs[0]=forwards[0](executions[0],input,ids,weights);
                expected=outputs[0].cpu().contiguous();
            }
            mfe_expert_plan_test::Option::select("1");
            if(graph)windows[1]->run();
            else outputs[1]=forwards[1](executions[1],input,ids,weights);
            auto result=outputs[1].cpu().contiguous();
            if(plans[0].kinds!=plans[1].kinds)
                throw std::runtime_error("expert plan changed pipeline route ownership");
            if(result.sizes()!=expected.sizes() || std::memcmp(result.data_ptr(),expected.data_ptr(),result.nbytes()))
                throw std::runtime_error("expert plan changed pipeline FFN output");
            for(const char* metric:{"ram_pcie_bytes","pipeline_transfer_cache_hits","pipeline_transfer_cache_misses",
                    "pipeline_transfer_cache_saved_bytes","residency_committed_projections"})
                if(counter(caches[0],metric)!=counter(caches[1],metric))
                    throw std::runtime_error("expert plan changed pipeline "+std::string(metric));
            ++cases;
        }
        dma+=counter(caches[1],"ram_pcie_bytes");
        hits+=counter(caches[1],"pipeline_transfer_cache_hits");
        misses+=counter(caches[1],"pipeline_transfer_cache_misses");
        for(const auto& mode:sources)for(const auto& source:mode)
            if(source->expert_disk_reads_after_preload())
                throw std::runtime_error("expert-plan pipeline reread a sealed weight bank");
    }
    if(!dma || !hits || !misses)throw std::runtime_error("expert-plan pipeline omitted DMA or cache hit/miss");
    std::cout<<"mfe_expert_plan_pipeline_cases="<<cases<<" dma_bytes="<<dma<<" hits="<<hits<<" misses="<<misses
        <<" equal_dma=1 equal_cache=1 equal_ownership=1 PASS\n";
}
