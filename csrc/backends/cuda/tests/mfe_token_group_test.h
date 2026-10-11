#pragma once
#include "mfe_ffn_runtime.h"
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
                sigmoid,shared_views,shared_gate,2560,640);
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
