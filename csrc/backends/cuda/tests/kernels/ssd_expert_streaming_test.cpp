#include "native/env.h"
#include "storage/weights/moe/expert_cache.h"
#include "storage/weights/moe/cached_source_internal.h"
#include "storage/weights/moe/ffn_pipeline.h"
#include "storage/weights/moe/quant_range_source.h"
#include "storage/weights/moe/moe_pinned_expert_arena.h"
#include "storage/weights/loaders/weight.h"
#include "ops/decode_window.h"
#include "storage/weights/moe/native_dma_batch.h"
#include "native/context.h"
#include "ops/quant/nint_prefill_cache.h"
#include "ops/moe/mfe_ffn_runtime.h"
#include "models/common/ffn.h"
#include "kernels/activation_ops.h"
#include "ops/format.h"
#include <array>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <filesystem>
#include <exception>
#include <future>
#include <iostream>
#include <map>
#include <mutex>

namespace tb=mfq_tensor_backend;
struct MoeResidencyTestAccess {
    static void progressive_read_batches() {
        MoeExpertCache::SsdExpertReads reads;
        reads.gate_up_requests=2;reads.gate_up_remaining=2;reads.down_remaining=2;
        std::promise<void> finish_down;
        auto release=finish_down.get_future();
        reads.ready=std::async(std::launch::async,[&] {
            reads.publish_read(0);reads.publish_read(2);reads.publish_read(1);
            release.wait_for(std::chrono::seconds(2));
            reads.publish_read(3);
        });
        std::vector<std::size_t> batch,gate_up,down;
        while(reads.next_projection_batch(batch,false))gate_up.insert(gate_up.end(),batch.begin(),batch.end());
        const bool first=reads.next_projection_batch(batch,true);
        down.insert(down.end(),batch.begin(),batch.end());
        const bool overlapped=reads.ready.wait_for(std::chrono::seconds(0))!=std::future_status::ready;
        finish_down.set_value();
        while(reads.next_projection_batch(batch,true))down.insert(down.end(),batch.begin(),batch.end());
        reads.wait();
        if(!first || !overlapped || gate_up!=std::vector<std::size_t>{0,1} || down!=std::vector<std::size_t>{2,3})
            throw std::runtime_error("Down batches did not publish before the complete SSD read joined");
        MoeExpertCache::SsdExpertReads failed;
        failed.down_remaining=1;
        failed.ready=std::async(std::launch::async,[&] {
            {std::lock_guard lock(failed.mutex);
             failed.failed=std::make_exception_ptr(std::runtime_error("controlled Down failure"));}
            failed.changed.notify_all();
        });
        bool caught=false;
        try{failed.next_projection_batch(batch,true);}catch(const std::runtime_error&){caught=true;}
        failed.wait();
        if(!caught)throw std::runtime_error("failed Down read did not release its batch waiter");
    }
    static tb::Tensor direct_cache(const std::shared_ptr<MoeExpertCache>& cache) {
        return cache->pipeline_direct_cache_storage_;
    }
    static tb::Tensor wire(const std::shared_ptr<MoeExpertCache>& cache) {
        return cache->pipeline_wire_gpu_;
    }
};
namespace {
void check(bool value,const char* message){if(!value)throw std::runtime_error(message);}
void check_packed_metadata() {
    for(int bits=1;bits<=8;++bits)for(std::size_t count=0;count<=257;++count) {
        const auto bytes=(count*bits+7)/8;
        std::vector<uint8_t> encoded(bytes+12,0),expected(count);
        for(std::size_t i=0;i<count;++i) {
            expected[i]=uint8_t((i*13+count*7)&((1u<<bits)-1));
            for(int bit=0;bit<bits;++bit)
                encoded[3+(i*bits+bit)/8]|=((expected[i]>>bit)&1u)<<((i*bits+bit)&7);
        }
        std::fill(encoded.begin()+3+bytes,encoded.end(),0xa5);
        std::size_t offset=3;
        check(mfq::cuda::quant_format::unpack_bits(encoded,offset,count,bits)==expected && offset==3+bytes,
            "packed metadata unpack changed values or consumed its neighbor");
        if(bytes) {
            encoded.resize(3+bytes-1);offset=3;bool rejected=false;
            try{(void)mfq::cuda::quant_format::unpack_bits(encoded,offset,count,bits);}
            catch(const std::runtime_error&){rejected=true;}
            check(rejected && offset==3,"truncated packed metadata was accepted");
        }
    }
}
void check_pinned_arena() {
    auto arena=std::make_shared<MoePinnedExpertArena>(1024);
    auto a=arena->allocate(255),b=arena->allocate(256),c=arena->allocate(512);
    check(a && b && c && arena->used_bytes()==1024,"pinned expert allocations were not bounded and aligned");
    check(!arena->allocate(16),"pinned expert arena exceeded its capacity");
    a.reset();c.reset();
    check(!arena->allocate(768),"pinned arena reused an interval protected by a live lease");
    b.reset();
    auto joined=arena->allocate(1024);
    check(joined && joined->data()==arena->data(),"pinned expert intervals did not coalesce");
    auto storage=arena->tensor();arena.reset();
    std::memset(joined->data(),0x5a,joined->bytes());joined.reset();
    check(static_cast<const uint8_t*>(storage.data_ptr())[1023]==0x5a,"pinned expert view outlived its storage owner");
    {
        auto fragmented=std::make_shared<MoePinnedExpertArena>(1024);
        auto x=fragmented->allocate(256),guard=fragmented->allocate(128),y=fragmented->allocate(128),tail=fragmented->allocate(512);
        auto* small=y->data();auto* large=x->data();x.reset();y.reset();
        auto picked=fragmented->allocate(128),next=fragmented->allocate(256);
        check(picked && next && picked->data()==small && next->data()==large,
            "pinned expert allocation fragmented an available larger interval");
    }
    if(mfq::cuda::NativeDmaBatch::available()) {
        auto gpu=tb::empty({1024},tb::TensorOptions().device(tb::kCUDA).dtype(tb::kUInt8));
        void* alias=nullptr;MFQ_CUDA_CHECK(cudaHostGetDevicePointer(&alias,storage.data_ptr(),0));
        const std::vector<mfq::cuda::DmaCopy> copies{{gpu.data_ptr(),storage.data_ptr(),1024}};
        check(mfq::cuda::NativeDmaBatch::submit(copies,mfq_current_cuda_stream(),1,
            [&](const void*,std::size_t){return reinterpret_cast<CUdeviceptr>(alias);}),
            "single merged projection bypassed the native DMA path");
        auto result=gpu.cpu().contiguous();
        check(std::memcmp(result.data_ptr(),storage.data_ptr(),1024)==0,"single native DMA copy changed the payload");
    }
}
void check_reused_expert_pages() {
    using mfq::cuda::KvExpertLoan;
    for(bool enabled:{false,true}) {
        KvExpertLoan loan;loan.set_reuse_priority(enabled);
        auto memory=tb::empty({int64_t(8*KvExpertLoan::page)},tb::TensorOptions().device(tb::kCUDA).dtype(tb::kUInt8));
        loan.add(memory,8,KvExpertLoan::page,true);
        const auto advance=[&] {MFQ_CUDA_CHECK(cudaStreamSynchronize(mfq_current_cuda_stream()));loan.next_packet();};
        const auto put=[&](int expert,int source=17) {
            bool hit=false;
            check(loan.pin_projection(source,expert,{int64_t(KvExpertLoan::page)},mfq_current_cuda_stream(),hit,
                [&](void* dst,const auto&) {MFQ_CUDA_CHECK(cudaMemsetAsync(dst,expert,KvExpertLoan::page,mfq_current_cuda_stream()));}),
                "reuse cache could not allocate an unprotected page");
        };
        const auto hit=[&](int expert) {
            std::array<std::size_t,8> offsets{};std::size_t bytes=0;
            check(loan.pin_existing_projection(17,expert,{int64_t(KvExpertLoan::page)},offsets,bytes),
                "reuse fixture lost a hot projection");
        };
        put(1);put(2);advance();
        if(enabled) {
            loan.set_reuse_learning(false);hit(1);hit(2);advance();
            check(!loan.reuse_stats().promotions,"synthetic accesses promoted GPU cache entries");
            loan.set_reuse_learning(true);
        }
        hit(1);hit(2);advance();
        for(int expert=10;expert<42;++expert){put(expert,99);advance();}
        check(bool(loan.find_for_exchange({17,1,0},KvExpertLoan::page))==enabled,
            "a one-use source scan displaced a protected recurring expert");
        if(!enabled)continue;
        auto stats=loan.reuse_stats();
        check(stats.promotions==2 && stats.protected_pages==2 && stats.probation_evictions>0,
            "reuse cache did not distinguish repeat hits from one-use insertions");
        // Rejected speculative hits must not promote a cold expert.
        put(3);put(4);advance();loan.begin_speculative(2);
        const std::array<int32_t,2> ids{3,4};loan.next_packet({17,18,19},ids.data(),2,1,64);
        hit(3);hit(4);loan.commit_speculative(1);advance();
        loan.begin_speculative(2);
        const std::array<int32_t,2> cancelled{4,3};loan.next_packet({17,18,19},cancelled.data(),2,1,64);
        hit(4);hit(3);loan.discard_speculative();advance();
        loan.begin_speculative(1);
        const int32_t rejected=4;loan.next_packet({17,18,19},&rejected,1,1,64);
        hit(4);loan.commit_speculative(0);advance();
        for(int expert=42;expert<74;++expert){put(expert,99);advance();}
        check(loan.find_for_exchange({17,3,0},KvExpertLoan::page) && !loan.find_for_exchange({17,4,0},KvExpertLoan::page),
            "rejected draft routes gained cache protection");
        for(int expert=5;expert<17;++expert){put(expert);advance();hit(expert);advance();}
        check(loan.reuse_stats().protected_pages<=6,"reuse protection consumed the probationary space");
        loan.retire_expert(17,16,1);advance();
        check(!loan.find_for_exchange({17,16,0},KvExpertLoan::page),"retired reused expert survived its reader fence");
        loan.ensure(8);
        check(!loan.bytes() && !loan.reuse_stats().protected_pages,"KV growth retained protected expert pages");
        const auto region=loan.add(memory,8,KvExpertLoan::page,true,1,true,true);
        put(1);advance();hit(1);advance();
        loan.set_reuse_priority(false);loan.set_reuse_priority(true);hit(1);advance();
        loan.remove(region);
        check(!loan.bytes() && !loan.reuse_stats().protected_pages,"removed workspace retained reused expert links");
    }
    std::cout<<"SSD reused expert pages: scan/eviction/speculation/retirement/KV growth PASS\n";
}
std::shared_ptr<MoeQuantRangeSource> source(const std::filesystem::path& path,const std::string& name="linear.weight") {
    auto model=mfq::open_model_source(path.string());auto read=model->tensor_reader(name);
    auto store=std::make_shared<mfq::MfeQuantExpertStore>(require_tensor(*model,name).nbytes,
        [read](std::size_t offset,uint8_t* dst,std::size_t n){read(offset,reinterpret_cast<std::byte*>(dst),n);},
        model->tensor_batch_reader(name));
    return std::make_shared<MoeQuantRangeSource>(std::move(store));
}
struct Run {
    CudaExecutionContext execution;
    std::shared_ptr<MoeExpertCache> cache;
    std::vector<std::shared_ptr<MoeQuantRangeSource>> sources;
    MoeFfnForward forward;
    bool hybrid=false;
    Run(const std::filesystem::path& path,bool streaming,bool automatic=false,bool combined=false,const std::string& prefix="") {
        auto& c=execution.config;
        c.moe_pipeline=true;c.moe_two_stage_ffn=true;c.moe_ram_pcie=true;c.moe_direct_ram=true;
        c.moe_ram_pcie_fraction=1.;c.moe_preload_all=!streaming || automatic;c.moe_assert_resident=!streaming || automatic;
        if(const auto* fraction=mfq::cuda::env_lookup("MFQ_SSD_EXPERT_TEST_PCIE_FRACTION")) {
            c.moe_ram_pcie_fraction=std::stod(fraction);hybrid=*c.moe_ram_pcie_fraction<1;
        }
        const bool mixed=path.stem()=="mixed";
        const bool phase_wide=path.stem()=="phase-wide";
        c.moe_host_cache_bytes=prefix.empty()?(streaming?(phase_wide?4<<20:mixed?512*1024:32768):(phase_wide?64:8)*1024*1024):(streaming?32*1024*1024:4LL*1024*1024*1024);
        // CPU/GPU arithmetic parity needs the same placement in both runs.
        // The GPU-only variant separately exercises adaptive exchanges.
        c.moe_residency_adapt=streaming && !hybrid;c.moe_residency_warm=streaming && !hybrid;c.moe_residency_projection_heat=true;
        cache=make_moe_expert_cache(prefix.empty()?((mixed || phase_wide)?512*1024:32768):64*1024*1024,c);
        const auto separator=prefix.find(';');
        std::vector<std::string> prefixes=separator==std::string::npos?std::vector<std::string>{prefix}:
            std::vector<std::string>{prefix.substr(0,separator),prefix.substr(separator+1)};
        sources.resize(3*prefixes.size());
        std::vector<MoeFfnForward> forwards;
        for(int layer=0;layer<int(prefixes.size());++layer) {
          std::vector<std::shared_ptr<MfeWeight>> gu,down;
          for(int p=0;p<3;++p) {
            const auto id=layer*3+p;
            if(combined && p==1){sources[id]=sources[id-1];continue;}
            sources[id]=prefix.empty()?source(path.parent_path()/(path.stem().string()+"-"+std::to_string(combined && p==0?3:p)+".mfq")):
                source(path,prefixes[layer]+std::array<std::string,3>{".gate.weight",".up.weight",".down.weight"}[p]);
            auto weight=std::make_shared<MfeWeight>(cache_quant_moe_weight(cache,
                std::to_string(id),sources[id],1,combined?-1:layer,std::to_string(p)));
            (p==2?down:gu).push_back(std::move(weight));
          }
          if(prefix.empty())forwards.push_back(make_moe_ffn_pipeline(gu,down));
          else {
            auto model=mfq::open_model_source(path.string());
            const auto shared_prefix=prefixes[layer].substr(0,prefixes[layer].size()-8)+".shared_expert";
            std::array<std::shared_ptr<const QuantLinear>,3> shared;
            for(int p=0;p<3;++p)shared[p]=std::make_shared<QuantLinear>(load_quant_linear(execution,*model,
                shared_prefix+std::array<std::string,3>{".gate.weight",".up.weight",".down.weight"}[p]));
            auto scalar=std::make_shared<QuantLinear>(load_quant_linear(execution,*model,shared_prefix+".router.weight"));
            auto gate=[scalar](CudaExecutionContext& e,const tb::Tensor& x){return tb::sigmoid(scalar->forward(e,x));};
            auto run_shared=[shared,gate](CudaExecutionContext& e,const tb::Tensor& x){
                auto g=shared[0]->forward(e,x),u=shared[1]->forward(e,x);
                return gate(e,x)*shared[2]->forward(e,(g*tb::sigmoid(g))*u);
            };
            forwards.push_back(make_moe_ffn_pipeline(gu,down,run_shared,{},MoeFfnSharedWeights{shared,gate,scalar}));
          }
        }
        forward=[forwards=std::move(forwards)](CudaExecutionContext& e,const tb::Tensor& x,const tb::Tensor& ids,const tb::Tensor& weights) {
            auto input=x;tb::Tensor output;
            for(std::size_t l=0;l<forwards.size();++l) {
                output=forwards[l](e,input,ids,weights);
                if(l+1<forwards.size())input=(tb::tanh(output.dim()==3?output.mean(1):output)*.1).to(tb::kFloat16);
            }
            return output;
        };
        finalize_moe_expert_cache(cache);
        if(!prefix.empty() || phase_wide) {
            auto loan=std::make_shared<mfq::cuda::KvExpertLoan>();
            auto storage=tb::empty({32*1024*1024},tb::TensorOptions().device(tb::kCUDA).dtype(tb::kUInt8));
            loan->add(storage,storage.numel(),1,true,1,true,true);
            set_moe_kv_expert_loan(cache,std::move(loan));
        }
        check(cache->ssd_experts()==streaming,"expert tier selection changed");
    }
    std::map<std::string,double> metrics()const {
        const auto values=moe_expert_memory_metrics(cache);return {values.begin(),values.end()};
    }
};
void check_output(const tb::Tensor& actual,const tb::Tensor& expected) {
    auto a=actual.to(tb::kCPU).to(tb::kFloat32).contiguous();
    auto e=expected.to(tb::kCPU).to(tb::kFloat32).contiguous();
    check(a.sizes()==e.sizes(),"SSD expert output shape differs");
    for(int64_t i=0;i<a.numel();++i) {
        const auto x=a.data_ptr<float>()[i],y=e.data_ptr<float>()[i];
        if(!std::isfinite(x) || std::abs(x-y)>5e-5f+.005f*std::abs(y))
            throw std::runtime_error("SSD expert result differs from full-RAM pipeline at "+std::to_string(i)+
                " actual="+std::to_string(x)+" expected="+std::to_string(y));
    }
}
void check_token_reuse(const std::filesystem::path& root,const std::string& format) {
    CudaExecutionContext execution;
    std::array<std::shared_ptr<MixedMoeRuntime>,3> projections;
    for(int p=0;p<3;++p) {
        auto model=mfq::open_model_source((root/(format+"-"+std::to_string(p)+".mfq")).string());
        projections[p]=make_mixed_moe_runtime(load_mfe_cpu(*model,"linear.weight"),true,execution.config);
        if(format=="nvq")for(const auto& pool:projections[p]->pools)
            check(!pool.nvq.dense_groups,"raw NVQ benchmark silently used a repacked layout");
    }
    const auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat16);
    const auto sigmoid=moe_swiglu_sigmoid_table_cuda();
    const int width=projections[0]->neuron_len,experts=projections[0]->n_experts;
    const auto* previous=mfq::cuda::env_lookup("MFQ_MFE_TOKEN_GROUPS");
    const std::string old=previous?previous:"";
    const auto select=[](const char* mode) {
#ifdef _WIN32
        _putenv_s("MFQ_MFE_TOKEN_GROUPS",mode);
#else
        if(*mode)setenv("MFQ_MFE_TOKEN_GROUPS",mode,1);else unsetenv("MFQ_MFE_TOKEN_GROUPS");
#endif
    };
    int cases=0;
    const auto* rows_option=mfq::cuda::env_lookup("MFQ_TOKEN_REUSE_TEST_ROWS");
    const int only_rows=rows_option?std::atoi(rows_option):0;
    for(int rows:{2,3,4,6,8,16,32})for(int sharing:{0,1,2}) {
        if(only_rows && rows!=only_rows)continue;
        std::vector<float> values(rows*width),weights(rows*3,1.f/3);
        std::vector<int32_t> ids(rows*3);
        for(std::size_t i=0;i<values.size();++i)values[i]=std::sin(float(i)*.017f)*.03f;
        for(int t=0;t<rows;++t)for(int r=0;r<3;++r)
            ids[t*3+r]=sharing==0?r:sharing==1?(t+r)%experts:(t*3+r)%experts;
        if(sharing==2)ids.back()=-1;
        auto x=tb::tensor(values).reshape({rows,width}).to(options);
        auto routed=tb::tensor(ids).reshape({rows,3}).to(tb::kCUDA);
        auto route_weights=tb::tensor(weights).reshape({rows,3}).to(tb::kCUDA);
        MfeFfnRuntime runtime({projections[0].get(),projections[1].get(),projections[2].get()},x,routed,route_weights,sigmoid,
            {},{},0,0,nullptr,false,false,0,10);
        check(runtime.batch().token_group_counts!=nullptr,"packed expert format omitted token grouping");
        if(!cases) {
            select("1");for(int warm=0;warm<100;++warm)runtime.run();mfq_cuda_synchronize();
        }
        std::array<float,2> elapsed{};tb::Tensor expected;
        for(int mode=0;mode<2;++mode) {
            select(mode?"1":"0");runtime.run();mfq_cuda_synchronize();
            if(!mode)expected=runtime.output().cpu().clone();else check_output(runtime.output(),expected);
            cudaGraph_t graph;cudaGraphExec_t replay;
            MFQ_CUDA_CHECK(cudaStreamBeginCapture(mfq_current_cuda_stream(),cudaStreamCaptureModeThreadLocal));
            runtime.run();MFQ_CUDA_CHECK(cudaStreamEndCapture(mfq_current_cuda_stream(),&graph));
            MFQ_CUDA_CHECK(cudaGraphInstantiate(&replay,graph,0));
            for(int warm=0;warm<3;++warm)MFQ_CUDA_CHECK(cudaGraphLaunch(replay,mfq_current_cuda_stream()));
            mfq_cuda_synchronize();
            cudaEvent_t begin,end;
            MFQ_CUDA_CHECK(cudaEventCreate(&begin));MFQ_CUDA_CHECK(cudaEventCreate(&end));
            MFQ_CUDA_CHECK(cudaEventRecord(begin,mfq_current_cuda_stream()));
            for(int repeat=0;repeat<20;++repeat)MFQ_CUDA_CHECK(cudaGraphLaunch(replay,mfq_current_cuda_stream()));
            MFQ_CUDA_CHECK(cudaEventRecord(end,mfq_current_cuda_stream()));
            MFQ_CUDA_CHECK(cudaEventSynchronize(end));
            MFQ_CUDA_CHECK(cudaEventElapsedTime(&elapsed[mode],begin,end));
            cudaEventDestroy(begin);cudaEventDestroy(end);
            cudaGraphExecDestroy(replay);cudaGraphDestroy(graph);
            elapsed[mode]/=20;
        }
        std::cout<<format<<"_token_reuse M="<<rows<<" sharing="<<sharing
            <<" ordinary_ms="<<elapsed[0]<<" grouped_ms="<<elapsed[1]
            <<" speedup="<<elapsed[0]/elapsed[1]<<" graph_replay=1\n";++cases;
    }
    select(old.c_str());
    std::cout<<format<<"_token_reuse_cases="<<cases<<" PASS\n";
}
void check_clipped_pipeline(const std::filesystem::path& root) {
    const auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat16);
    std::vector<float> values{-65504,-100,-11,-10,-1,-0.0f,0,1,9.99f,10,11,100,65504};
    auto gate=tb::tensor(values).to(options);
    std::reverse(values.begin(),values.end());
    auto up=tb::tensor(values).to(options);
    auto expected_gate=tb::clamp_max(gate.to(tb::kFloat32),10);
    auto expected_up=tb::clamp(up.to(tb::kFloat32),-10,10);
    check_output(clipped_silu_mul_cuda(gate,up,10),
        (tb::silu(expected_gate)*expected_up).to(tb::kFloat16));
    int cases=0;
    for(const char* format:{"nint","nvq3j","mxfp4","mixed"}) {
        if(const auto* selected=mfq::cuda::env_lookup("MFQ_SSD_EXPERT_TEST_FORMAT");selected &&
                std::string(selected).find(format)==std::string::npos)continue;
        if(!std::filesystem::exists(root/(std::string(format)+"-0.mfq")))continue;
        std::map<std::tuple<double,int,int>,tb::Tensor> references;
        for(bool ssd:{false,true})for(double limit:{.01,10.}) {
            CudaExecutionContext execution;auto& config=execution.config;
            config.moe_pipeline=true;config.moe_two_stage_ffn=true;
            config.moe_ram_pcie=true;config.moe_direct_ram=true;config.moe_ram_pcie_fraction=1.;
            config.moe_preload_all=!ssd;config.moe_assert_resident=!ssd;
            config.moe_residency_adapt=false;config.moe_residency_warm=false;
            config.moe_host_cache_bytes=ssd?32768:16*1024*1024;
            const bool mixed=std::string(format)=="mixed";
            auto cache=make_moe_expert_cache(mixed?1<<20:32768,config);
            std::array<MfeWeight,3> resident;
            if(!ssd)for(int p=0;p<3;++p) {
                auto model=mfq::open_model_source((root/(std::string(format)+"-"+std::to_string(p)+".mfq")).string());
                resident[p]=load_mfe_gpu(execution,*model,"linear.weight");
            }
            std::array<FFN,2> layers;
            int width=0,experts=0;
            for(int layer=0;layer<2;++layer) {
                auto& ffn=layers[layer];ffn.is_moe=true;ffn.moe_split_gate_up=true;
                ffn.moe_layer=layer;ffn.moe_top_k=2;ffn.moe_normalize=true;ffn.moe_delayed_softmax=false;
                ffn.moe_shared_ungated=true;ffn.swiglu_limit=limit;
                for(int p=0;p<3;++p) {
                    auto quant=source(root/(std::string(format)+"-"+std::to_string(p)+".mfq"));
                    auto weight=cache_quant_moe_weight(cache,std::to_string(layer)+"/"+std::to_string(p),
                        quant,1,layer,std::to_string(p));
                    if(p==0){width=quant->metadata()->neuron_len;experts=quant->metadata()->n_experts;}
                    (p==0?ffn.moe_gate:p==1?ffn.moe_up:ffn.moe_down)=std::move(weight);
                }
                ffn.moe_router=tb::zeros({experts,width},options.dtype(tb::kFloat32));
                const int base=mixed?6:0;
                ffn.moe_hash_ids=tb::tensor(std::vector<int32_t>{base,base+1,base+1,base+2,base+2,base}).reshape({3,2}).to(tb::kCUDA);
                ffn.shared=std::make_unique<FFN>();ffn.shared->swiglu_limit=limit;
                auto dense=[&](int out,int in) {
                    QuantLinear value;value.kind=QuantLinearKind::Dense;
                    value.logical_out=out;value.logical_neuron_len=in;
                    value.dense=tb::full({out,in},.003,options);return value;
                };
                ffn.shared->gate_up.layers={dense(16,width),dense(16,width)};
                ffn.shared->gate_up.outs={16,16};ffn.shared->down=dense(width,16);
                ffn.prepare_moe_pipeline(execution);
                check(bool(ffn.moe_pipeline_forward),"generic FFN did not register its SSD pipeline");
            }
            finalize_moe_expert_cache(cache);
            std::shared_ptr<mfq::cuda::KvExpertLoan> loan;
            tb::Tensor loan_storage;
            uint64_t loan_region=0;
            if(ssd && mixed) {
                loan=std::make_shared<mfq::cuda::KvExpertLoan>();
                loan_storage=tb::empty({4<<20},options.dtype(tb::kUInt8));
                loan_region=loan->add(loan_storage,loan_storage.numel(),1,true,1,true,true);
                set_moe_kv_expert_loan(cache,loan);
            }
            bool saw_prefill=false,loan_revoked=false;
            bool retained=false,restored=false;
            uint64_t retained_captures=0;
            tb::Tensor old_direct_cache,old_wire;
            for(int tokens:{1,2,3,4,5,6,1,11,143,1,1})for(auto& ffn:layers) {
                if(ssd && tokens==11 && ffn.moe_layer==0) {
                    old_direct_cache=MoeResidencyTestAccess::direct_cache(cache);
                    old_wire=MoeResidencyTestAccess::wire(cache);
                    retained_captures=moe_expert_graph_captures(cache);
                    retained=lend_moe_decode_cache_for_prefill(cache,0);
                    const auto* fallback=mfq::cuda::env_lookup("MFQ_MOE_WIRE_FALLBACK");
                    const bool indirect=loan && mfq::cuda::NativeDmaBatch::available() &&
                        (!fallback || fallback[0]!='0');
                    check(retained==indirect,"private FFN graph lending capability mismatch");
                    if(!retained)release_moe_decode_buffers(cache);
                    if(retained) {
                        check(old_direct_cache.defined(),"indirect FFN has no expert cache to return");
                        // Hold the old allocation so restoring the cache must
                        // choose another address, then poison every old page.
                        MFQ_CUDA_CHECK(cudaMemsetAsync(old_direct_cache.data_ptr(),0xa5,
                            old_direct_cache.numel(),mfq_current_cuda_stream()));
                        MFQ_CUDA_CHECK(cudaMemsetAsync(old_wire.data_ptr(),0xa5,
                            old_wire.numel(),mfq_current_cuda_stream()));
                    } else {old_direct_cache={};old_wire={};}
                }
                std::fprintf(stderr,"clipped_check format=%s ssd=%d limit=%g tokens=%d layer=%d phase=reference\n",
                    format,int(ssd),limit,tokens,ffn.moe_layer);
                std::vector<float> input(tokens*width);
                const float amplitude=std::string(format)=="nvq3j" || mixed?.003f:.3f;
                for(std::size_t i=0;i<input.size();++i)input[i]=std::sin(float(i)*.13f)*amplitude;
                auto x=tb::tensor(input).reshape({tokens,width}).to(options);
                std::vector<int32_t> selected(tokens*2);
                std::vector<int64_t> input_ids(tokens);
                for(int t=0;t<tokens;++t) {
                    input_ids[t]=(t+tokens+ffn.moe_layer)%3;
                    for(int r=0;r<2;++r)selected[t*2+r]=(mixed?6:0)+int(input_ids[t]+r)%3;
                }
                const auto key=std::make_tuple(limit,tokens,ffn.moe_layer);
                if(!ssd) {
                    auto ids=tb::tensor(selected).reshape({tokens,2}).to(tb::kCUDA);
                    auto route=build_moe_route_plan(ids,experts);
                    auto weights=moe_sqrtsoftplus_weights_cuda(tb::zeros({tokens,experts},options.dtype(tb::kFloat32)),ids,1e-20,1);
                    auto g=resident[0].forward(execution,x,route),u=resident[1].forward(execution,x,route);
                    tb::Tensor hidden;
                    if(limit>0) {
                        g=tb::clamp_max(g.to(tb::kFloat32),limit);
                        u=tb::clamp(u.to(tb::kFloat32),-limit,limit);
                        hidden=(tb::silu(g)*u).to(tb::kFloat16).contiguous();
                    } else hidden=moe_swiglu_split_cuda(tb::cat({g,u},-1).contiguous());
                    auto routed=moe_weighted_reduce_cuda(resident[2].forward(execution,hidden,route),weights);
                    references[key]=(routed+ffn.shared->forward(execution,x)).cpu().clone();
                }
                execution.force_moe_materialized_swiglu=false;
                std::fprintf(stderr,"clipped_check phase=pipeline\n");
                const bool speculative=ssd && tokens<=6;
                if(speculative && ffn.moe_layer==0)begin_moe_speculative_routes(cache,tokens);
                check_output(ffn.forward(execution,x,tb::tensor(input_ids).to(tb::kCUDA)),references.at(key));++cases;
                if(retained && saw_prefill && tokens==1) {
                    const auto cache_storage=MoeResidencyTestAccess::direct_cache(cache);
                    check(cache_storage.defined() && cache_storage.data_ptr()!=old_direct_cache.data_ptr(),
                        "retained FFN did not restore its returned cache at a new address");
                    const auto wire=MoeResidencyTestAccess::wire(cache);
                    check(wire.defined() && wire.data_ptr()!=old_wire.data_ptr(),
                        "retained FFN did not restore its returned wire at a new address");
                    check(moe_expert_graph_captures(cache)==retained_captures,
                        "prefill recaptured an indirect private FFN graph");
                    restored=true;
                }
                if(speculative && ffn.moe_layer==1) {
                    if(tokens==6)discard_moe_speculative_routes(cache);
                    else commit_moe_speculative_routes(cache,tokens-1);
                }
                if(tokens==143)saw_prefill=true;
                if(loan && saw_prefill && tokens==1 && ffn.moe_layer==1 && !loan_revoked) {
                    // The final M=1 replay must refresh descriptors after KV
                    // growth revokes every borrowed expert page.
                    loan->remove(loan_region);
                    MFQ_CUDA_CHECK(cudaMemsetAsync(loan_storage.data_ptr(),0xa5,
                        loan_storage.numel(),mfq_current_cuda_stream()));
                    loan_revoked=true;
                }
            }
            check(cache->stats().pipeline_two_stage_serves>=16,
                "clipped FFN did not execute its two-stage kernels");
            check(cache->stats().pipeline_multi_route_positions[0]>cache->stats().pipeline_multi_unique_routes[0] &&
                    cache->stats().pipeline_multi_unique_routes[0]>0,
                "multi-token route sharing was not measured");
            check(!retained || restored,"retained private FFN was not replayed after prefill");
            if(retained) {
                // A different loan owns different publication observers. Its
                // first decode must rebuild graphs instead of reading the old
                // mapped dispatch table retained by the previous loan.
                auto replacement=std::make_shared<mfq::cuda::KvExpertLoan>();
                auto storage=tb::empty({4<<20},options.dtype(tb::kUInt8));
                replacement->add(storage,storage.numel(),1,true,1,true,true);
                set_moe_kv_expert_loan(cache,replacement);
                check(!MoeResidencyTestAccess::direct_cache(cache).defined(),
                    "replacing the loan retained its old expert cache");
                for(auto& ffn:layers) {
                    std::vector<float> input(width);
                    for(int i=0;i<width;++i)input[i]=std::sin(float(i)*.13f)*.003f;
                    auto x=tb::tensor(input).reshape({1,width}).to(options);
                    auto id=tb::tensor(std::vector<int64_t>{(1+ffn.moe_layer)%3}).to(tb::kCUDA);
                    check_output(ffn.forward(execution,x,id),references.at({limit,1,ffn.moe_layer}));
                }
                check(moe_expert_graph_captures(cache)>retained_captures,
                    "a replacement loan reused stale cached-dispatch observers");
            }
            check(moe_expert_graph_captures(cache)>0,
                "private FFN initialization was not visible to shared MTP policy");
            if(ssd && mixed && mfq::cuda::NativeDmaBatch::available()) {
                const auto spec=loan->speculative_stats();
                check(spec.windows==9 && spec.cancels==1 && spec.disabled==0,
                    "private FFN graphs lost speculative cache commit/cancel accounting");
                check(cache->stats().pipeline_host_packet_serves>0,"private FFN graphs did not use host cache packets");
                check(cache->stats().pipeline_kv_projection_hits>0,"private FFN graphs did not reuse borrowed GPU expert copies");
                const auto* fallback=mfq::cuda::env_lookup("MFQ_MOE_WIRE_FALLBACK");
                if(fallback && fallback[0]=='0')for(const auto& [name,bytes]:moe_expert_memory_metrics(cache))
                    if(name=="expert_transfer_wire_bytes")
                        check(bytes<=16384,"direct FFN packets retained an unused full payload wire");
            }
            release_moe_prefill_buffers(cache);
        }
    }
    check(cases>0,"missing clipped FFN fixtures");
    std::cout<<"generic_clipped_ffn_cases="<<cases<<" ordinary/pipeline RAM/SSD multi-layer PASS\n";
}
void check_ssd_predictor_overlap(const std::filesystem::path& root) {
    for(int variant:{0,1,2,3}) {
        const bool shared_arena=variant!=0,wide=variant>=2,fail_image=variant==3;
        const int width=wide?384:96;
        CudaExecutionContext execution;auto& config=execution.config;
        config.moe_pipeline=true;config.moe_two_stage_ffn=true;config.moe_direct_ram=true;
        config.moe_ram_pcie=true;config.moe_ram_pcie_fraction=1.;
        config.moe_preload_all=false;config.moe_assert_resident=false;
        config.moe_residency_adapt=true;config.moe_residency_warm=true;config.moe_residency_projection_heat=true;
        config.moe_host_cache_bytes=wide?4*1024*1024:128*1024;
        auto cache=make_moe_expert_cache(wide?512*1024:64*1024,config);
        mfq::engine::MoeCacheProfile profile;profile.path="ssd-fixture";
        profile.layers[0].ranking={0,1};profile.layers[0].frequencies={{0,.6},{1,.4}};
        set_moe_expert_cache_profile(*cache,std::move(profile));
        std::vector<std::shared_ptr<MoeQuantRangeSource>> quant_sources;
        std::array<MoeFfnForward,2> forward;
        for(int layer=0;layer<2;++layer) {
            std::vector<std::shared_ptr<MfeWeight>> gu,down;
            for(int p=0;p<3;++p) {
                const auto family=wide?"phase-wide":layer && !shared_arena?"nint":"nvq3j";
                auto quant=source(root/(std::string(family)+"-"+std::to_string(p)+".mfq"));
                quant_sources.push_back(quant);
                auto weight=std::make_shared<MfeWeight>(cache_quant_moe_weight(cache,
                    "ssd-overlap/"+std::to_string(layer)+"/"+std::to_string(p),quant,1,layer?-1:0,std::to_string(p)));
                (p==2?down:gu).push_back(std::move(weight));
            }
            forward[layer]=make_moe_ffn_pipeline(gu,down);
        }
        finalize_moe_expert_cache(cache);
        auto loan=std::make_shared<mfq::cuda::KvExpertLoan>();
        auto loan_storage=tb::empty({wide?8<<20:2<<20},tb::TensorOptions().device(tb::kCUDA).dtype(tb::kUInt8));
        loan->add(loan_storage,loan_storage.numel(),1,true,1,true,true);
        set_moe_kv_expert_loan(cache,loan);
        bool pageable_target=false;
        for(auto it=quant_sources.rbegin();it!=quant_sources.rend();++it) {
            const auto& quant=*it;
            const auto lease=quant->acquire_expert(33);
            const auto fields=moe_cache_field_refs(lease->weights);
            if(quant->host_key(33).source<3)for(const auto& field:fields)
                pageable_target|=field.nbytes()>0 && !field.options().pinned();
            std::vector<int64_t> sizes;for(const auto& field:fields)sizes.push_back(field.nbytes());
            // The tiny fixture is below the cache's 64 KiB admission floor.
            // Add a private tail while retaining the real fields' geometry.
            std::size_t packed=0;for(const auto size:sizes)packed=((packed+15)&~std::size_t(15))+size;
            if(packed<(64<<10))sizes.push_back((64<<10)-((packed+15)&~std::size_t(15)));
            bool hit=false;
            const auto* cached=loan->pin_projection(quant->host_key(33).source,33,sizes,mfq_current_cuda_stream(),hit,
                [&](void* target,const std::array<std::size_t,8>& offsets) {
                    for(std::size_t f=0;f<fields.size();++f)if(sizes[f])
                        MFQ_CUDA_CHECK(cudaMemcpyAsync(static_cast<uint8_t*>(target)+offsets[f],fields[f].data_ptr(),
                            std::size_t(sizes[f]),cudaMemcpyHostToDevice,mfq_current_cuda_stream()));
                    if(sizes.size()>fields.size())MFQ_CUDA_CHECK(cudaMemsetAsync(
                        static_cast<uint8_t*>(target)+offsets[fields.size()],0,std::size_t(sizes.back()),mfq_current_cuda_stream()));
                });
            check(cached!=nullptr,"SSD GPU refill fixture did not enter the cache");
            MFQ_CUDA_CHECK(cudaStreamSynchronize(mfq_current_cuda_stream()));
            if(wide)quant->host_cache()->erase(quant->host_key(33));
        }
        if(const auto* cap=mfq::cuda::env_lookup("MFQ_MOE_RAM_PIN_GIB");cap && std::atof(cap)*(1LL<<30)==16384)
            check(pageable_target,"SSD partial pin fixture did not include pageable experts");
        auto x=tb::full({1,width},.017,tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat16));
        auto ids=tb::tensor(std::vector<int32_t>{31,32}).reshape({1,2}).to(tb::kCUDA);
        auto weights=tb::full({1,2},.5,x.options().dtype(tb::kFloat32));
        const auto target=forward[0](execution,x,ids,weights).cpu().clone();
        const auto predictor=forward[1](execution,x,ids,weights).cpu().clone();
        std::mutex mutex;std::condition_variable changed;bool held=false,release=false;
        MoeResidencyManager manager(cache.get(),[&](const char* event,std::size_t) {
            if(fail_image && std::string(event)=="gpu_image")throw std::runtime_error("injected GPU image failure");
            if(std::string(event)!="ssd_predictor_overlap_ready")return;
            std::unique_lock lock(mutex);held=true;changed.notify_all();
            if(!changed.wait_for(lock,std::chrono::seconds(5),[&]{return release;}))
                throw std::runtime_error("SSD predictor waited for a held background read");
        });
        for(int i=0;i<8;++i)manager.after_layer(0,std::vector<int32_t>(10,33),1);
        bool failed=false;
        try {manager.prepare_predictor();}
        catch(const std::runtime_error& error) {
            if(std::string(error.what())!="injected GPU image failure")throw;
            failed=true;
        }
        if(!shared_arena) {
            {std::unique_lock lock(mutex);
                check(changed.wait_for(lock,std::chrono::seconds(5),[&]{return held;}),"SSD predictor did not overlap a background read");}
            check_output(forward[1](execution,x,ids,weights),predictor);
        }
        {std::lock_guard lock(mutex);release=true;}changed.notify_all();
        try {manager.apply_pending();}
        catch(const std::runtime_error& error) {
            if(std::string(error.what())!="injected GPU image failure")throw;
            failed=true;
        }
        check(failed==fail_image,"GPU image failure did not propagate");
        if(!fail_image)check(manager.stats().kv_refill_bytes>0,"SSD expert promotion reread an existing GPU cache copy");
        if(wide && !fail_image)check(manager.stats().ssd_gpu_image_bytes>0,"SSD exchange reloaded a complete GPU image from disk");
        if(wide && !fail_image)check(manager.stats().ssd_gpu_image_copies==manager.stats().ssd_gpu_images,
            "contiguous GPU expert fields were not copied as one image");
        if(const auto* option=mfq::cuda::env_lookup("MFQ_MOE_RESIDENCY_NATIVE_DMA");option && option[0]=='1' &&
                mfq::cuda::NativeDmaBatch::available() && !fail_image)
            check(manager.stats().native_dma_batches>0,"SSD expert exchange bypassed native DMA batching");
        check(bool(manager.stats().predictor_overlap_serves)==!shared_arena,
            "SSD predictor overlap ignored shared arena ownership");
        check_output(forward[0](execution,x,ids,weights),target);
        check_output(forward[1](execution,x,ids,weights),predictor);
        std::array<tb::Tensor,2> prefill_expected;
        auto prompt=tb::full({143,width},.017,x.options());
        std::vector<int32_t> prompt_routes(286);
        for(std::size_t i=0;i<prompt_routes.size();++i)prompt_routes[i]=31+int(i%2);
        auto prompt_ids=tb::tensor(prompt_routes).reshape({143,2}).to(tb::kCUDA);
        auto prompt_weights=tb::full({143,2},.5,weights.options());
        for(int phase:{0,1,0,1}) {
            auto result=forward[phase](execution,prompt,prompt_ids,prompt_weights);
            const auto raw=moe_expert_memory_metrics(cache);
            const std::map<std::string,double> metrics(raw.begin(),raw.end());
            check(metrics.at("ssd_prefill_group_size")>0 && metrics.at("ssd_prefill_predictor")==double(phase),
                "SSD prefill reused the other phase's ring layout");
            if(prefill_expected[phase].defined())check_output(result,prefill_expected[phase]);
            else prefill_expected[phase]=result.cpu().clone();
        }
        release_moe_prefill_buffers(cache);
    }
}
}
int main(int argc,char** argv)try {
    MoeResidencyTestAccess::progressive_read_batches();
    if(argc==3 && std::string(argv[2])=="--mxfp4-token-reuse") {
        check_token_reuse(argv[1],"mx4");return 0;
    }
    if(argc==3 && std::string(argv[2])=="--nvq-token-reuse") {
        check_token_reuse(argv[1],"nvq");return 0;
    }
    if(argc==3 && std::string(argv[2])=="--clipped-check") {
        check_clipped_pipeline(argv[1]);return 0;
    }
    std::set_terminate([] {
        if(auto error=std::current_exception())try {std::rethrow_exception(error);}
        catch(const std::exception& value){std::fprintf(stderr,"SSD test terminated: %s\n",value.what());}
        catch(...){std::fprintf(stderr,"SSD test terminated: unknown exception\n");}
        else std::fprintf(stderr,"SSD test terminated without an exception\n");
        std::fflush(stderr);std::abort();
    });
    check_packed_metadata();
    check_pinned_arena();
    check_reused_expert_pages();
    check(argc==2 || argc==3,"expected SSD expert fixture directory or model and expert tensor prefix");
    const auto stream=mfq_current_cuda_stream();
    auto context=mfq::cuda::default_context(mfq_current_cuda_device());
    int cases=0;
    if(argc==2)check_ssd_predictor_overlap(argv[1]);
    for(const char* format:{"nint","nvq2j","nvq3j","nvq2j-l","nvq2j-xl","nvq3j-l","nint-combined","mixed","phase-wide"}) {
        if(const auto* selected=mfq::cuda::env_lookup("MFQ_SSD_EXPERT_TEST_FORMAT");selected && std::string(selected)!=format)continue;
        if(argc==3 && std::string(format)!="nint")continue;
        const bool combined=std::string(format)=="nint-combined";
        const auto path=argc==3?std::filesystem::path(argv[1]):std::filesystem::path(argv[1])/(std::string(combined?"nint":format)+".mfq");
        const std::string prefix=argc==3?argv[2]:"";
        Run reference(path,false,false,combined,prefix);
        Run candidate(path,true,std::string(format)!="nvq2j",combined,prefix);
        const int width=reference.sources[0]->metadata()->neuron_len;
        const int experts=reference.sources[0]->metadata()->n_experts;
        if(prefix.empty() && std::string(format)=="nint") {
            auto* quant=candidate.sources[0].get();bool failed=false;
            try{candidate.cache->queue_ssd_expert_copies({{quant,31,{}}}).get();}
            catch(const std::invalid_argument&){failed=true;}
            check(failed,"SSD prefill worker lost a read failure");
            auto held=quant->acquire_expert(31);
            const auto before=candidate.metrics();
            const auto cached=candidate.cache->read_ssd_experts({{quant,31},{quant,31}});
            const auto after=candidate.metrics();
            check(cached.size()==2 && cached[0]==held && cached[1]==held &&
                after.at("ssd_ram_read_fast_hits")==before.at("ssd_ram_read_fast_hits")+2 &&
                after.at("ssd_scheduled_read_jobs")==before.at("ssd_scheduled_read_jobs"),
                "RAM hits launched SSD worker jobs or lost their shared lease");
            for(std::size_t gate_up_count:{0,1,2}) {
                auto ready=candidate.cache->begin_ssd_expert_reads({{quant,31},{quant,31}},gate_up_count);
                check(!ready->ready.valid(),"RAM-only phase batch launched a coordinator job");
                std::vector<std::size_t> batch,gu,down;
                while(ready->next_projection_batch(batch,false))gu.insert(gu.end(),batch.begin(),batch.end());
                while(ready->next_projection_batch(batch,true))down.insert(down.end(),batch.begin(),batch.end());
                check(gu.size()==gate_up_count && down.size()==2-gate_up_count,
                    "RAM projection phase hit was not published exactly once");
                ready->wait();
                check(ready->leases[0]==held && ready->leases[1]==held,"phase batch changed a RAM hit's lease");
            }
            for(int expert:{45,46,47})quant->host_cache()->erase(quant->host_key(expert));
            {
                auto reads=candidate.cache->begin_ssd_expert_reads({{quant,45},{quant,46},{quant,47}},2);
                check(reads->ready.valid(),"cold phase batch did not start asynchronous reads");
                std::vector<std::size_t> batch;std::array<int,2> publications{};
                while(reads->next_projection_batch(batch,false))for(const auto i:batch) {
                    check(i<2 && ++publications.at(i)==1 && bool(reads->leases[i]),
                        "GU batch duplicated a row or preceded its lease publication");
                }
                check(publications==std::array<int,2>{1,1},"GU batch omitted a completed read");
                std::vector<std::size_t> down;
                while(reads->next_projection_batch(batch,true))down.insert(down.end(),batch.begin(),batch.end());
                check(down==std::vector<std::size_t>{2},"Down batch omitted or duplicated a completed read");
                reads->wait();check(bool(reads->leases[2]),"Down read lost its lease");
                for(int i=0;i<3;++i) {
                    const auto expected=moe_cache_fields(quant->read_expert(45+i));
                    const auto actual=moe_cache_fields(reads->leases[i]->weights);
                    for(std::size_t f=0;f<expected.size();++f)check(std::memcmp(actual[f].data_ptr(),expected[f].data_ptr(),expected[f].nbytes())==0,
                        "asynchronous SSD phase read changed an expert field");
                }
            }
            failed=false;
            try {
                auto reads=candidate.cache->begin_ssd_expert_reads({{quant,experts+1},{quant,47}},1);
                std::vector<std::size_t> batch;
                while(reads->next_projection_batch(batch,false)){}
                reads->wait();
            }catch(const std::exception&){failed=true;}
            check(failed,"failed asynchronous SSD GU read did not release its waiter");
            // Dropping a packet still drains its queued read before freeing
            // leases and completion state captured by the retained worker.
            candidate.cache->begin_ssd_expert_reads({{quant,48},{quant,49}},1).reset();
            const auto expected=moe_cache_fields(quant->read_expert(31));
            std::array<std::vector<tb::Tensor>,2> targets;
            std::array<std::future<void>,2> reads;
            for(int i=0;i<2;++i) {
                for(const auto& field:expected) {
                    auto target=field.clone();if(target.nbytes())std::memset(target.data_ptr(),0xa5,target.nbytes());
                    targets[i].push_back(std::move(target));
                }
                reads[i]=candidate.cache->queue_ssd_expert_copies({{quant,31,targets[i]}});
            }
            for(int i=0;i<2;++i) {
                reads[i].get();
                for(std::size_t f=0;f<expected.size();++f)check(std::memcmp(targets[i][f].data_ptr(),expected[f].data_ptr(),expected[f].nbytes())==0,
                    "SSD prefill worker did not recover or changed a queued packet");
            }
        }
        for(const auto& quant:candidate.sources)for(std::size_t pool=0;pool<quant->store().pool_count();++pool) {
            if(quant->metadata()->pools[pool].family!=MixedMoeFamily::Nvq)continue;
            const auto& members=quant->store().pool_expert_ids(pool);
            if(members.size()<2)continue;
            const std::vector<int> selected{members[0],members[1]};
            const auto prototype=moe_cache_fields(quant->read_expert(selected[0]));
            std::array<tb::Tensor,4> storage,targets;
            for(int f=0;f<4;++f) {
                storage[f]=tb::full({int64_t(prototype[f].nbytes()*2+32)},0xa5,tb::TensorOptions().device(tb::kCPU).dtype(tb::kUInt8));
                targets[f]=tb::from_blob(storage[f].data_ptr<uint8_t>()+16,{prototype[f].numel()*2},prototype[f].options());
            }
            check(quant->copy_nvq_expert_fields(selected,targets),"aligned NVQ experts could not share their canonical reads");
            for(int i=0;i<2;++i) {
                const auto expected=moe_cache_fields(quant->decode_expert(selected[i],quant->store().read_expert(selected[i])));
                for(int f=0;f<4;++f)check(std::memcmp(static_cast<const uint8_t*>(targets[f].data_ptr())+i*prototype[f].nbytes(),
                    expected[f].data_ptr(),prototype[f].nbytes())==0,"bulk NVQ runtime fields changed canonical bytes");
            }
            for(int f=0;f<4;++f) {
                const auto* bytes=storage[f].data_ptr<uint8_t>();
                for(int i=0;i<16;++i)check(bytes[i]==0xa5 && bytes[storage[f].numel()-1-i]==0xa5,"bulk NVQ runtime copy overwrote guards");
            }
        }
        {
            auto pageable=std::make_shared<MoeQuantRangeSource>(
                std::make_shared<mfq::MfeQuantExpertStore>(candidate.sources[0]->store()));
            pageable->bind_pinned_arena(std::make_shared<MoePinnedExpertArena>(16));
            const auto expert=31%experts;
            const auto actual=moe_cache_fields(pageable->read_expert(expert));
            const auto decoded=moe_cache_fields(pageable->decode_expert(expert,pageable->store().read_expert(expert)));
            check(actual.size()==decoded.size(),"pageable expert field count differs");
            for(std::size_t f=0;f<actual.size();++f)
                check(actual[f].numel()==decoded[f].numel() &&
                    actual[f].scalar_type()==decoded[f].scalar_type() && (!actual[f].numel() ||
                    !std::memcmp(actual[f].data_ptr(),decoded[f].data_ptr(),actual[f].numel()*actual[f].element_size())),
                    "pageable expert fields changed when the pinned arena could not fit an expert");
        }
        {
            auto partial=std::make_shared<MoeQuantRangeSource>(
                std::make_shared<mfq::MfeQuantExpertStore>(candidate.sources[0]->store()));
            const auto expert=31%experts;
            const auto decoded=moe_cache_fields(partial->decode_expert(expert,partial->store().read_expert(expert)));
            std::size_t largest=0;
            for(const auto& field:decoded)largest=std::max(largest,std::size_t(field.nbytes()));
            auto arena=std::make_shared<MoePinnedExpertArena>((largest+15)&~std::size_t(15));
            partial->bind_pinned_arena(arena);
            {
                const auto actual=moe_cache_fields(partial->read_expert(expert));
                check(actual.size()==decoded.size(),"partially pinned expert lost a field");
                bool pinned=false,pageable=false;
                for(std::size_t f=0;f<actual.size();++f) {
                    if(!actual[f].nbytes())continue;
                    pinned|=actual[f].options().pinned();pageable|=!actual[f].options().pinned();
                    check(actual[f].nbytes()==decoded[f].nbytes() &&
                          !std::memcmp(actual[f].data_ptr(),decoded[f].data_ptr(),actual[f].nbytes()),
                          "partially pinned expert changed decoded field bytes");
                }
                check(pinned && pageable && arena->used_bytes()<=arena->bytes(),
                      "fragmented expert allocation did not use bounded mixed storage");
            }
            check(arena->used_bytes()==0,"partially pinned expert leaked field leases");
        }
        if(candidate.hybrid)for(std::size_t p=0;p<candidate.sources.size();++p)for(const int e:{31%experts,32%experts}) {
            const auto& source=candidate.sources[p];
            const auto direct=source->read_expert(e);
            const auto decoded=source->decode_expert(e,source->store().read_expert(e));
            const auto resident=reference.sources[p]->acquire_expert(e);
            const auto a=moe_cache_fields(direct),b=moe_cache_fields(decoded);
            const auto c=moe_cache_fields(resident->weights);
            check(a.size()==b.size(),"direct CPU expert field count differs");
            for(std::size_t f=0;f<a.size();++f) {
                check(a[f].numel()==b[f].numel() && a[f].scalar_type()==b[f].scalar_type(),"direct CPU expert field shape differs");
                check(!a[f].numel() || !std::memcmp(a[f].data_ptr(),b[f].data_ptr(),a[f].numel()*a[f].element_size()),
                    "direct CPU expert fields differ from full codec");
                check(a[f].numel()==c[f].numel() && (!a[f].numel() ||
                    !std::memcmp(a[f].data_ptr(),c[f].data_ptr(),a[f].numel()*a[f].element_size())),
                    "SSD CPU fields differ from resident RAM fields");
            }
        }
        if(std::string(format)=="phase-wide") {
            bool reserved=false;
            const auto requested=mfq::cuda::default_context()->available_memory()+(std::size_t(1)<<30);
            set_moe_workspace_reservation(candidate.cache,[&](std::size_t bytes){
                check(bytes==requested,"SSD prefill changed its workspace reservation");reserved=true;
            });
            {
                MoePrefillWorkspaceLoan loan(candidate.cache);
                check(!loan.begin(requested) && reserved,"SSD prefill skipped cache reclamation when primary expert borrowing was unavailable");
            }
            set_moe_workspace_reservation(candidate.cache,{});
            const auto suspended=candidate.cache->suspend_residency(true);
            auto input=tb::full({1,width},.015,tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat16));
            auto ids=tb::full({1,2},51,input.options().dtype(tb::kInt32));
            auto weights=tb::full({1,2},.5,input.options().dtype(tb::kFloat32));
            const auto expected=reference.forward(reference.execution,input,ids,weights).cpu().clone();
            release_moe_decode_buffers(reference.cache);
            ids.fill_(0);
            candidate.forward(candidate.execution,input,ids,weights);
            MFQ_CUDA_CHECK(cudaStreamSynchronize(stream));
            for(const auto& source:candidate.sources) {
                check(!source->gpu_resident(51),"one-use RAM priority fixture unexpectedly selected a primary expert");
                source->host_cache()->erase(source->host_key(51));
            }
            ids.fill_(51);
            const auto before=candidate.sources[0]->host_cache()->stats().reuse_promotions;
            check_output(candidate.forward(candidate.execution,input,ids,weights),expected);
            check(candidate.sources[0]->host_cache()->stats().reuse_promotions==before,
                "one SSD demand promoted its fresh RAM image as a second use");
            candidate.cache->suspend_residency(suspended);
            release_moe_decode_buffers(reference.cache);release_moe_decode_buffers(candidate.cache);
        }
        for(const int rows:{1,2,3,4,5,6,8,32,1024,143,1,1024})for(const bool graph:{false,true}) {
            if(graph && rows>32)continue;
            std::cerr<<"ssd_case format="<<format<<" rows="<<rows<<" graph="<<graph<<'\n';
            reference.execution.continuous_batch_cache_serial=rows==32;
            candidate.execution.continuous_batch_cache_serial=rows==32;
            auto x=tb::zeros({rows,width},tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat16));
            auto ids=tb::zeros({rows,2},x.options().dtype(tb::kInt32));
            auto weights=tb::full({rows,2},.5, x.options().dtype(tb::kFloat32));
            const auto set_input=[&](int step) {
                std::vector<float> input(rows*width);
                for(std::size_t i=0;i<input.size();++i)input[i]=(prefix.empty()?.001f:.1f)*std::sin(float(i+step*17)*.027f);
                std::vector<int32_t> routes(rows*2);
                for(int i=0;i<rows*2;++i)routes[i]=rows>32?(i+step*19)%experts:
                    (31+(i%2)+(step==2?11:0)+(graph && std::string(format)=="phase-wide"?rows*2:0))%experts;
                x.copy_(tb::tensor(input).reshape({rows,width}).to(tb::kFloat16).to(tb::kCUDA));
                ids.copy_(tb::tensor(routes).reshape({rows,2}).to(tb::kCUDA));
            };
            std::vector<tb::Tensor> expected;
            for(int step=0;step<3;++step) {
                set_input(step);
                expected.push_back(reference.forward(reference.execution,x,ids,weights).cpu().clone());
            }
            release_moe_decode_buffers(reference.cache);
            tb::Tensor result;
            std::unique_ptr<mfq::cuda::DecodeWindow> window;
            if(graph) {
                window=std::make_unique<mfq::cuda::DecodeWindow>(stream,false,rows==32);
                window->capture([&]{result=candidate.forward(candidate.execution,x,ids,weights);},[]{},[&]{result={};});
            }
            for(int step=0;step<3;++step) {
                set_input(step);
                const auto before=candidate.metrics();
                if(graph)window->run();else result=candidate.forward(candidate.execution,x,ids,weights);
                check_output(result,expected[step]);
                finish_moe_expert_exchanges(candidate.cache);
                const auto after=candidate.metrics();
                const auto promotion_reads=after.at("residency_ssd_load_requests")-before.at("residency_ssd_load_requests");
                if(rows<=32)check(after.at("ssd_expert_read_count")-before.at("ssd_expert_read_count")<=2*candidate.sources.size()+promotion_reads,
                    "one window reread an expert for repeated token routes");
                check(after.at("ssd_expert_enabled")==1 && after.at("ssd_expert_payload_bytes")>0,
                    "SSD expert residency was not reported");
                check(candidate.sources[0]->host_cache()->stats().managed_peak_bytes<=candidate.execution.config.moe_host_cache_bytes,
                    "SSD expert LRU exceeded its RAM budget");
            }
            window.reset();result={};
            if(rows<=32)release_moe_decode_buffers(candidate.cache);
            release_moe_decode_buffers(reference.cache);
            ++cases;
        }
        {
            const auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat16);
            auto small=tb::full({1,width},.015,options),large=tb::full({1024,width},.015,options);
            std::vector<int32_t> single(10);
            for(std::size_t i=0;i<single.size();++i)single[i]=(31+int(i))%experts;
            auto small_ids=tb::tensor(single).reshape({1,10}).to(tb::kCUDA);
            std::vector<int32_t> routes(10240);
            for(std::size_t i=0;i<routes.size();++i)routes[i]=int(i)%experts;
            auto large_ids=tb::tensor(routes).reshape({1024,10}).to(tb::kCUDA);
            auto small_weights=tb::full({1,10},.1,options.dtype(tb::kFloat32));
            auto large_weights=tb::full({1024,10},.1,options.dtype(tb::kFloat32));
            reference.execution.continuous_batch_cache_serial=false;
            candidate.execution.continuous_batch_cache_serial=false;
            auto expected_small=reference.forward(reference.execution,small,small_ids,small_weights).cpu().clone();
            auto expected_large=reference.forward(reference.execution,large,large_ids,large_weights).cpu().clone();
            release_moe_decode_buffers(reference.cache);
            release_moe_decode_buffers(candidate.cache);
            tb::Tensor result;
            mfq::cuda::DecodeWindow retained(stream,false);
            retained.capture([&]{result=candidate.forward(candidate.execution,small,small_ids,small_weights);},[]{},[&]{result={};});
            for(int request=0;request<3;++request) {
                std::cerr<<"ssd_retained_case format="<<format<<" request="<<request<<'\n';
                {
                    NintPrefillDecodeCacheScope reuse(true);
                    check_output(candidate.forward(candidate.execution,large,large_ids,large_weights),expected_large);
                }
                retained.run();check_output(result,expected_small);
                finish_moe_expert_exchanges(candidate.cache);
                cases+=2;
            }
        }
        release_moe_decode_buffers(candidate.cache);
        if(prefix.empty() && !combined && !candidate.hybrid) {
            int cold=-1;
            for(int e=0;e<experts && cold<0;++e)for(const auto& source:candidate.sources)
                if(!source->gpu_resident(e)){cold=e;break;}
            check(cold>=0,"SSD promotion fixture has no cold expert");
            const auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat16);
            auto input=tb::full({143,width},.015,options);
            auto ids=tb::full({143,2},cold,options.dtype(tb::kInt32));
            auto weights=tb::full({143,2},.5,options.dtype(tb::kFloat32));
            check_output(candidate.forward(candidate.execution,input,ids,weights),
                reference.forward(reference.execution,input,ids,weights));
            release_moe_prefill_buffers(candidate.cache);
            MFQ_CUDA_CHECK(cudaStreamSynchronize(stream));
            for(const auto& source:candidate.sources)for(int e=0;e<experts;++e)
                source->host_cache()->erase(source->host_key(e));
            const auto before=candidate.metrics().at("residency_committed_projections");
            prepare_moe_decode_residency(candidate.cache);
            check(candidate.metrics().at("residency_committed_projections")>before,
                "hot experts evicted from RAM could not be promoted from SSD");
            check(candidate.sources[0]->host_cache()->stats().managed_peak_bytes<=candidate.execution.config.moe_host_cache_bytes,
                "SSD promotion exceeded the managed RAM budget");
            check_output(candidate.forward(candidate.execution,input,ids,weights),
                reference.forward(reference.execution,input,ids,weights));
            release_moe_prefill_buffers(candidate.cache);
        }
        if(std::string(format)=="mixed") {
            finish_moe_expert_exchanges(candidate.cache);
            for(const auto& source:candidate.sources)for(int expert=0;expert<experts;++expert)
                if(!source->gpu_resident(expert)) {
                    const auto lease=source->acquire_expert(expert);
                    check(bool(lease),"mixed SSD working set lost a cold expert");
                }
        }
        const auto metrics=candidate.metrics();
        check(metrics.at("ram_expert_pinned_bytes")>0,"SSD RAM cache did not allocate reusable pinned storage");
        check(metrics.at("ssd_expert_bytes_read")>0,"SSD expert path did not read checkpoint payloads");
        check(metrics.at("ram_expert_cache_evictions")>0,"tiny RAM cache did not exercise eviction");
        if(!candidate.hybrid)check(metrics.at("residency_committed_projections")>0,"SSD cache never adapted GPU expert residency");
        if(candidate.hybrid) {
            std::uint64_t cpu=0;for(const auto& source:candidate.sources)cpu+=source->cpu_projections();
            check(cpu>0,"SSD hybrid test did not execute CPU expert projections");
        }
        if(prefix.empty() && std::string(format).starts_with("nvq") && std::string(format)!="nvq2j-xl")
            check(metrics.at("ssd_direct_field_reads")>0,"NVQ fields were not read directly into pinned targets");
        std::cout<<"ssd_experts format="<<format<<" reads="<<metrics.at("ssd_expert_read_count")
            <<" bytes="<<metrics.at("ssd_expert_bytes_read")<<" bounded_RAM=1 graph=1 PASS\n";
        release_moe_decode_buffers(candidate.cache);
    }
    std::cout<<"SSD expert streaming cases="<<cases<<" PASS\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
