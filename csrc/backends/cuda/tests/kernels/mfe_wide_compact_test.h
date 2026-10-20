#pragma once
#include "native/env.h"
#include "ops/moe/mfe_ffn_runtime.h"
#include "kernels/mfe/ffn_token_groups.h"

class MfeWideOption {
    const char* key_;std::string old_;bool present_;
public:
    explicit MfeWideOption(const char* key,const char* value):key_(key),present_(mfq::cuda::env_lookup(key)!=nullptr) {
        if(present_)old_=mfq::cuda::env_lookup(key);set(value);
    }
    void set(const char* value)const {
#ifdef _WIN32
        _putenv_s(key_,value);
#else
        if(*value)setenv(key_,value,1);else unsetenv(key_);
#endif
    }
    ~MfeWideOption(){set(present_?old_.c_str():"");}
};

void verify_mfe_wide_compact(const std::filesystem::path& root,bool benchmark=true,bool aligned_probe=false,bool legacy_only=false,bool shared_quad_probe=false,bool shared_split_probe=false,bool aligned_warps_probe=false,bool nonaligned_quad_probe=false) {
    MfeWideOption capacity("MFQ_MFE_TOKEN_GROUP_CAPACITY","2"),wide("MFQ_MFE_TOKEN_WIDE_COMPACT","0");
    MfeWideOption grouped("MFQ_MFE_TOKEN_GROUPS","1"),compact("MFQ_MFE_TOKEN_COMPACT","1");
    MfeWideOption aligned("MFQ_MFE_ALIGNED_COMPACT",aligned_probe?"0":"");
    MfeWideOption alignment("MFQ_MFE_TOKEN_ALIGNED_GU","");
    MfeWideOption split_shared("MFQ_MFE_SPLIT_SHARED_GROUPS","0");
    MfeWideOption aligned_warps("MFQ_MFE_ALIGNED_GU_WARPS","16");
    CudaExecutionContext execution;const auto context=mfq::cuda::default_context(mfq_current_cuda_device());
    const auto stream=mfq_current_cuda_stream();context->begin_graph_pool(stream);
    const auto sigmoid=moe_swiglu_sigmoid_table_cuda();int checks=0,wide_buckets=0;
    int case_index=0;const auto* start_option=mfq::cuda::env_lookup("MFQ_MFE_WIDE_START_CASE");
    const int start_case=start_option?std::atoi(start_option):0;
    std::array<NintWeight,3> shared_weights;
    for(int p=0;p<3;++p) {
        const std::array<const char*,3> roles={"gate","up","down"};
        std::ifstream f(root/(std::string("shared-s2-")+roles[p]+".nint"),std::ios::binary);
        if(!f)throw std::runtime_error("missing wide-group shared fixture");
        const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)),std::istreambuf_iterator<char>());
        shared_weights[p]=to_gpu_nint(unpack_nint(bytes));
    }
    for(const char* format:{"nvq2j","nvq2j-l","nvq2j-xl","nvq3j","nvq3j-512","nvq3j-l","nvq1-l","nvq1-s","nint5"}) {
        const std::string name(format);const bool nint=name=="nint5",nvq1=name=="nvq1-l" || name=="nvq1-s";
        if(legacy_only && !(nint || nvq1))continue;
        if(!aligned_probe && !legacy_only && (nint || nvq1))continue;
        if(!benchmark && !legacy_only && name!="nvq2j-l" && name!="nvq3j-512" && !(aligned_probe && (nint || name=="nvq1-s")))continue;
        std::array<MixedMoeRuntime,3> projections;
        for(int p=0;p<3;++p) {
            auto model=mfq::open_model_source((root/(std::string(format)+(p==2?"-2560-640.mfq":"-640-2560.mfq"))).string());
            const auto encoded=load_mfe_cpu(*model,"linear.weight");
            auto gpu=make_mixed_moe_runtime(encoded,true,execution.config),cpu=make_mixed_moe_runtime(encoded,false,execution.config);
            if(cpu->pools.size()!=1 || cpu->n_experts!=3)throw std::runtime_error("wide-group fixture requires three original experts");
            const int copies=aligned_probe?16:4,experts=3*copies;
            const auto repeat=[&](const tb::Tensor& t){return t.numel()?tb::cat(std::vector<tb::Tensor>(copies,t),0):t;};
            auto& pool=gpu->pools[0];
            if(nint) {
                auto& w=pool.nint;const auto packed_bytes=w.q_packed.numel();
                if(!w.q_expert_stride) {
                    std::vector<tb::Tensor> offsets;
                    for(int copy=0;copy<copies;++copy)offsets.push_back(w.row_q_bit_offsets+int64_t(copy)*packed_bytes*8);
                    w.row_q_bit_offsets=tb::cat(offsets,0);
                }else w.row_q_bit_offsets=repeat(w.row_q_bit_offsets);
                w.q_packed=repeat(w.q_packed);w.row_q_bits=repeat(w.row_q_bits);
                w.sub_scale=repeat(w.sub_scale);w.sub_min=repeat(w.sub_min);
                w.neuron_scale=repeat(w.neuron_scale);w.neuron_min=repeat(w.neuron_min);w.out*=copies;
            }else {
                auto weight=nvq1?pool.nvq:cpu->pools[0].nvq;
                if(!nvq1 && !prepare_nvq_dense_groups(weight))throw std::runtime_error("wide-group fixture cannot use compact representation");
                weight.indices_packed=repeat(weight.indices_packed).to(tb::kCUDA);
                weight.aux_packed=repeat(weight.aux_packed).to(tb::kCUDA);
                weight.sub_scale_packed=repeat(weight.sub_scale_packed).to(tb::kCUDA);
                weight.neuron_scale=repeat(weight.neuron_scale).to(tb::kCUDA);
                weight.codebook=weight.codebook.to(tb::kCUDA);weight.out*=copies;
                if(nvq1 && weight.decode_records.defined())weight.decode_records=repeat(weight.decode_records);
                pool.nvq=std::move(weight);
            }
            pool.local_experts*=copies;
            std::vector<int32_t> local(experts);for(int i=0;i<experts;++i)local[i]=i;
            pool.expert_local=tb::tensor(local).to(tb::kCUDA);gpu->n_experts=experts;
            projections[p]=std::move(*gpu);
        }
        for(int tokens:{2,3,4,6,8})for(bool shared:{false,true}) {
            if(aligned_probe && tokens!=4 && !(shared_quad_probe && tokens==3))continue;
            if(shared_quad_probe && !shared)continue;
            if(!benchmark && tokens!=3 && tokens!=4 && tokens!=6 && !(legacy_only && tokens==8))continue;
            if(case_index++<start_case)continue;
            std::vector<float> values(tokens*2560),weights(tokens*10,.1f);
            std::vector<int32_t> selected(tokens*10);
            const int experts=aligned_probe?48:12;
            for(std::size_t i=0;i<values.size();++i)values[i]=std::sin(float(i)*.017f)*.13f;
            const auto routes=[&](int pattern) {
                for(int t=0;t<tokens;++t)for(int r=0;r<10;++r)
                    selected[t*10+r]=pattern==0?r:pattern==1?(aligned_probe?t*10+r:(r+t*3)%experts):
                        pattern==3?(r+t*3)%experts:r==9?(t%2?-1:experts):r%5;
            };
            routes(0);
            auto x=tb::tensor(values).reshape({tokens,2560}).to(tb::kFloat16).to(tb::kCUDA),initial=x.clone();
            auto ids=tb::tensor(selected).reshape({tokens,10}).to(tb::kCUDA);
            auto route_weights=tb::tensor(weights).reshape({tokens,10}).to(tb::kCUDA);
            std::array<const NintWeight*,3> shared_views{};
            if(shared)for(int p=0;p<3;++p)shared_views[p]=&shared_weights[p];
            auto shared_gate=shared?tb::sigmoid(x.mean(1)):tb::Tensor{};
            constexpr int variants=5;
            const std::array<int,variants> capacities=nonaligned_quad_probe?std::array<int,variants>{2,4,2,4,4}:
                shared_split_probe?std::array<int,variants>{2,4,20,4,4}:
                shared_quad_probe?std::array<int,variants>{2,4,20,2,4}:
                aligned_probe?std::array<int,variants>{2,2,4,4,4}:std::array<int,variants>{2,4,4,8,8};
            std::unique_ptr<MfeFfnRuntime> runtime[variants];mfq::cuda::Graph graph[variants];
            std::vector<tb::Tensor> expected_output,expected_hidden;
            for(int variant=0;variant<variants;++variant) {
                capacity.set(capacities[variant]==20?"shared4":capacities[variant]==2?"2":capacities[variant]==4?"4":"8");
                if(aligned_probe) {
                    aligned.set(shared_quad_probe?(variant==0 || (!shared_split_probe && variant==3)?"0":"1"):
                        (variant==1 || variant>=3?"1":"0"));wide.set("1");
                    alignment.set(shared_quad_probe && !shared_split_probe && variant==3?"0":"1");
                    if(variant==4)capacity.set("");
                }else wide.set(variant==2 || variant==4?"1":"0");
                split_shared.set(shared_split_probe && variant>=3?"1":"0");
                if(aligned_warps_probe) {
                    aligned.set("1");wide.set("1");
                    aligned_warps.set(variant==1 || variant>=3?"12":"16");
                }
                if(nonaligned_quad_probe) {
                    aligned.set("1");wide.set("1");aligned_warps.set("16");
                    alignment.set(variant==4?"":variant==2 || variant==3?"0":"1");
                }
                routes(0);
                ids.copy_(tb::tensor(selected).reshape({tokens,10}).to(tb::kCUDA));x.copy_(initial);
                runtime[variant]=std::make_unique<MfeFfnRuntime>(
                    std::array<MixedMoeRuntime*,3>{&projections[0],&projections[1],&projections[2]},x,ids,route_weights,sigmoid,shared_views,shared_gate,
                    0,0,nullptr,false,false,nint?4:0);
                auto& r=*runtime[variant];r.run();const auto& b=r.batch();
                if(!b.token_group_compact || (!(nvq1 || nint) && !b.compact_nvq) || b.token_group_capacity!=capacities[variant])
                    throw std::runtime_error("wide compact fixture missed its descriptor/capacity contract");
                std::vector<int32_t> buckets(4*(b.token_groups+1));
                MFQ_CUDA_CHECK(cudaMemcpyAsync(buckets.data(),b.token_group_buckets,buckets.size()*sizeof(int32_t),cudaMemcpyDeviceToHost,stream));
                MFQ_CUDA_CHECK(cudaStreamSynchronize(stream));
                if(capacities[variant]>=4 && capacities[variant]<=8 && tokens>=3) {
                    const int bucket=capacities[variant]==8 && tokens>4?3:2;
                    if(!buckets[bucket*(b.token_groups+1)])throw std::runtime_error("wide routed bucket remained empty");
                    if(aligned_probe?variant>=2:(variant==2 || variant==4))++wide_buckets;
                }
                if(capacities[variant]==20) {
                    if(b.token_group_aligned_gu)throw std::runtime_error("shared quad skipped its required hidden quantizer");
                    std::vector<int32_t> counts(b.token_groups);
                    MFQ_CUDA_CHECK(cudaMemcpyAsync(counts.data(),b.token_group_counts,counts.size()*sizeof(int32_t),cudaMemcpyDeviceToHost,stream));
                    MFQ_CUDA_CHECK(cudaStreamSynchronize(stream));
                    for(int i=0;i<tokens*10;++i)if(counts[i]>2)throw std::runtime_error("shared-only plan widened routed groups");
                    if(counts[tokens*10]!=tokens)throw std::runtime_error("shared-only plan did not group all shared tokens");
                }
                if(shared_split_probe && variant>=3) {
                    std::vector<int32_t> counts(b.token_groups);
                    MFQ_CUDA_CHECK(cudaMemcpyAsync(counts.data(),b.token_group_counts,counts.size()*sizeof(int32_t),cudaMemcpyDeviceToHost,stream));
                    MFQ_CUDA_CHECK(cudaStreamSynchronize(stream));
                    if(b.token_groups<tokens*10+2 || counts[tokens*10]!=2 || counts[tokens*10+1]!=tokens-2)
                        throw std::runtime_error("shared pair plan did not retain routed quads and shared pairs");
                }
                {mfq::cuda::GraphWarmupScope warmup(context,stream);r.run();}
                mfq_cuda_synchronize();graph[variant].capture_begin_shared();r.run();graph[variant].capture_end();
                for(int pattern=0;pattern<(aligned_probe?4:3);++pattern) {
                    routes(pattern);ids.copy_(tb::tensor(selected).reshape({tokens,10}).to(tb::kCUDA));x.copy_(initial*float(pattern+1));
                    graph[variant].replay();auto actual=r.output().cpu().contiguous(),hidden=r.hidden().cpu().contiguous();
                    if(!variant){expected_output.push_back(actual.clone());expected_hidden.push_back(hidden.clone());}
                    else {
                        const auto& expected=expected_output[pattern];const auto& eh=expected_hidden[pattern];
                        if(actual.sizes()!=expected.sizes() || std::memcmp(actual.data_ptr(),expected.data_ptr(),actual.nbytes()))
                            throw std::runtime_error("wide compact FFN output bits differ");
                        for(int t=0;t<tokens;++t)for(int route=0;route<10+int(shared);++route) {
                            if(route<10 && unsigned(selected[t*10+route])>=unsigned(experts))continue;
                            const int width=route==10?b.shared_intermediate:b.intermediate;
                            const int64_t offset=int64_t(t*(10+int(shared))+route)*b.hidden_stride;
                            if(std::memcmp(hidden.data_ptr<mfq_half>()+offset,eh.data_ptr<mfq_half>()+offset,width*sizeof(mfq_half)))
                                throw std::runtime_error("wide compact rounded hidden bits differ");
                        }
                    }
                    if(aligned_probe && pattern==1) {
                        int singles=0;MFQ_CUDA_CHECK(cudaMemcpyAsync(&singles,b.token_group_buckets,sizeof(singles),cudaMemcpyDeviceToHost,stream));
                        MFQ_CUDA_CHECK(cudaStreamSynchronize(stream));
                        if(!singles)throw std::runtime_error("aligned compact single bucket remained empty");
                    }
                    ++checks;
                }
            }
            if(aligned_warps_probe)for(int down_gs:{28,32})for(int base_variant:{0,2}) {
                // Gate/Up only: change the destination quantization geometry,
                // without pretending that packed Down weights use another GS.
                auto b=runtime[base_variant]->batch();
                const int view_count=b.projection_offsets[2]+b.projection_counts[2]+(shared?3:0);
                std::vector<mfq::cuda::MfePackedProjection> views(view_count);
                MFQ_CUDA_CHECK(cudaMemcpyAsync(views.data(),b.projections,
                    views.size()*sizeof(views[0]),cudaMemcpyDeviceToHost,stream));
                MFQ_CUDA_CHECK(cudaStreamSynchronize(stream));
                for(int i=0;i<b.projection_counts[2];++i)
                    views[b.projection_offsets[2]+i].group_size=down_gs;
                if(shared)views.back().group_size=down_gs;
                auto descriptors=tb::empty({int64_t(views.size()*sizeof(views[0]))},x.options().dtype(tb::kUInt8));
                MFQ_CUDA_CHECK(cudaMemcpyAsync(descriptors.data_ptr(),views.data(),descriptors.nbytes(),cudaMemcpyHostToDevice,stream));
                b.projections=reinterpret_cast<const mfq::cuda::MfePackedProjection*>(descriptors.data_ptr());
                if(shared)b.shared=b.projections+view_count-3;
                b.scale_stride=(std::max(b.intermediate,b.shared_intermediate)+down_gs-1)/down_gs;
                b.quantized_stride=b.scale_stride*down_gs;
                std::array<tb::Tensor,3> expected;
                for(int warps:{16,12}) {
                    auto hidden=tb::zeros({tokens*(10+int(shared)),b.hidden_stride},x.options());
                    auto quantized=tb::zeros({tokens*(10+int(shared)),b.quantized_stride},x.options().dtype(tb::kInt8));
                    auto scales=tb::zeros({tokens*(10+int(shared)),b.scale_stride},x.options().dtype(tb::kFloat32));
                    b.hidden=reinterpret_cast<__half*>(hidden.data_ptr());
                    b.hidden_quantized=quantized.data_ptr<int8_t>();b.hidden_scales=scales.data_ptr<float>();
                    aligned_warps.set(warps==12?"12":"16");
                    if(!mfq::cuda::mfe_ffn_token_group_gate_up(b,0,std::max(b.intermediate,b.shared_intermediate),stream))
                        throw std::runtime_error("aligned GS boundary missed grouped GU dispatch");
                    MFQ_CUDA_CHECK(cudaStreamSynchronize(stream));
                    std::array<tb::Tensor,3> actual={hidden.cpu().contiguous(),quantized.cpu().contiguous(),scales.cpu().contiguous()};
                    for(int i=0;i<3;++i)if(warps==16)expected[i]=actual[i].clone();
                        else if(std::memcmp(actual[i].data_ptr(),expected[i].data_ptr(),actual[i].nbytes()))
                            throw std::runtime_error("aligned GS28/32 hidden or quantization bits differ");
                    if(warps==12)++checks;
                }
            }
            if(shared_quad_probe?(graph[nonaligned_quad_probe?(nint && tokens==4?1:3):shared_split_probe?3:1].nodes()!=graph[4].nodes()):
                aligned_probe?(graph[0].nodes()!=graph[1].nodes() || graph[2].nodes()!=graph[3].nodes() || graph[3].nodes()!=graph[4].nodes()):
                (graph[1].nodes()!=graph[2].nodes() || graph[3].nodes()!=graph[4].nodes()))
                throw std::runtime_error("wide compact specialization changed FFN scheduling: "+name+
                    " T="+std::to_string(tokens)+" case="+std::to_string(case_index-1));
            if(!benchmark) {
                std::cout<<"mfe_case_pass index="<<case_index-1<<" format="<<name<<" tokens="<<tokens<<" shared="<<shared<<'\n';
                continue;
            }
            const int timed_variants=shared_quad_probe && !shared_split_probe && !nonaligned_quad_probe && tokens==3?3:aligned_probe?4:variants;
            for(int sample_pattern=0;sample_pattern<(aligned_probe?3:2);++sample_pattern) {
                const int pattern=sample_pattern==2?3:sample_pattern;
                routes(pattern);ids.copy_(tb::tensor(selected).reshape({tokens,10}).to(tb::kCUDA));x.copy_(initial);
                std::vector<float> timing[variants];
                for(int sample=0;sample<7;++sample)for(int order=0;order<timed_variants;++order) {
                    const int variant=(sample+order)%timed_variants;
                    mfq::cuda::Event begin(cudaEventDefault),end(cudaEventDefault);begin.record(stream);
                    for(int i=0;i<32;++i)graph[variant].replay();end.record(stream);end.synchronize();
                    float ms=0;MFQ_CUDA_CHECK(cudaEventElapsedTime(&ms,begin.get(),end.get()));
                    if(sample)timing[variant].push_back(ms*1000/32);
                }
                std::cout<<(nonaligned_quad_probe?"mfe_nonaligned_quad":aligned_warps_probe?"mfe_aligned_warps":shared_split_probe?"mfe_shared_split":shared_quad_probe?"mfe_shared_quad":aligned_probe?"mfe_aligned_compact":"mfe_wide_compact")<<" format="<<format<<" tokens="<<tokens<<" shared="<<shared<<" pattern="<<pattern;
                for(int variant=0;variant<timed_variants;++variant) {
                    auto sorted=timing[variant];std::sort(sorted.begin(),sorted.end());
                    std::cout<<" v"<<variant<<"_us="<<sorted[3]<<" v"<<variant<<"_nodes="<<graph[variant].nodes()<<" v"<<variant<<"_samples=";
                    for(auto v:timing[variant])std::cout<<v<<',';
                }
                std::cout<<'\n';
            }
        }
    }
    std::cout<<(nonaligned_quad_probe?"MFE nonaligned quad":aligned_warps_probe?"MFE aligned warps":shared_split_probe?"MFE shared split":shared_quad_probe?"MFE shared quad":aligned_probe?"MFE aligned compact":"MFE wide compact")<<" executions="<<checks<<" active_wide_buckets="<<wide_buckets<<" PASS\n";
    mfq_cuda_synchronize();context->end_graph_pool(stream);
}
