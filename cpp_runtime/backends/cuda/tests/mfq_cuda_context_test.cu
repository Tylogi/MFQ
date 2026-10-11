#include "mfq_cuda_context.h"
#include "mfq_cuda_resident_allocations.h"

#include <cuda_runtime_api.h>

#include <array>
#include <cassert>
#include <cstdint>
#include <memory>
#include <cstdlib>
#include <iostream>
#include <string>

int memory_limit_cases() {
    constexpr auto mib=std::size_t(1024)*1024;
    auto first=std::make_shared<mfq::cuda::Context>(0);
    auto second=std::make_shared<mfq::cuda::Context>(0);
    mfq::cuda::StreamGuard active(mfq::cuda::StreamHandle(0,first->stream().get()));
    assert(first->memory_stats().limit==1024*mib);
    assert(first->memory_stats().allocated==0);
    {
        mfq::cuda::Buffer model(first,48*mib);
        {
            mfq::cuda::Buffer cache(second,16*mib);
            bool rejected=false;
            try {mfq::cuda::Buffer overflow(first,first->memory_stats().limit);}catch(const mfq::cuda::Error& error) {
                rejected=std::string(error.what()).find("VRAM limit exceeded")!=std::string::npos;
            }
            assert(rejected && second->memory_stats().allocated==64*mib);
        }
        assert(first->memory_stats().allocated==48*mib);
        const auto stream=first->stream().get();first->begin_graph_pool(stream);
        void* warmed=nullptr;
        {
            mfq::cuda::GraphWarmupScope warm(first,stream);
            mfq::cuda::Buffer workspace(first,16*mib);warmed=workspace.data();
        }
        // Released tensors still own their warmed graph allocations.
        assert(second->memory_stats().allocated==64*mib);
        first->begin_graph_capture(stream);
        {
            mfq::cuda::Buffer reuse(first,16*mib);
            assert(reuse.data()==warmed && first->memory_stats().allocated==64*mib);
        }
        first->end_graph_capture(stream);
        bool retained_rejected=false;
        try {mfq::cuda::Buffer overflow(second,second->memory_stats().limit);}catch(const mfq::cuda::Error&) {retained_rejected=true;}
        assert(retained_rejected && first->memory_stats().allocated==64*mib);
        first->end_graph_pool(stream);
        assert(second->memory_stats().allocated==48*mib);
    }
    first->stream().synchronize();second->stream().synchronize();
    assert(first->memory_stats().allocated==0 && first->memory_stats().peak==64*mib);
    std::cout<<"cuda_memory_limit_cases=5 limit_bytes="<<first->memory_stats().limit
        <<" peak_bytes="<<first->memory_stats().peak<<" allocated_after_release=0\n";
    return 0;
}

int memory_pool_reuse_case() {
    auto context=std::make_shared<mfq::cuda::Context>(0);
    mfq::cuda::StreamGuard active(mfq::cuda::StreamHandle(0,context->stream().get()));
    if(!context->supports_async_allocations())throw std::runtime_error("async allocator required");
    const auto limit=context->memory_stats().limit,baseline=context->local_memory_usage();
    if(limit<=baseline+128*1024*1024)throw std::runtime_error("insufficient test memory budget");
    const auto retained=(limit-baseline)*3/4;
    cudaMemPool_t pool=nullptr;MFQ_NATIVE_CUDA_CHECK(cudaDeviceGetDefaultMemPool(&pool,0));
    std::uint64_t old_threshold=0,keep=~std::uint64_t(0);
    MFQ_NATIVE_CUDA_CHECK(cudaMemPoolGetAttribute(pool,cudaMemPoolAttrReleaseThreshold,&old_threshold));
    struct Restore {
        cudaMemPool_t pool;std::uint64_t threshold;
        ~Restore(){cudaMemPoolSetAttribute(pool,cudaMemPoolAttrReleaseThreshold,&threshold);cudaMemPoolTrimTo(pool,0);}
    } restore{pool,old_threshold};
    MFQ_NATIVE_CUDA_CHECK(cudaMemPoolSetAttribute(pool,cudaMemPoolAttrReleaseThreshold,&keep));
    {mfq::cuda::Buffer released(context,retained);}
    context->stream().synchronize();
    std::uint64_t reserved=0,used=0;
    MFQ_NATIVE_CUDA_CHECK(cudaMemPoolGetAttribute(pool,cudaMemPoolAttrReservedMemCurrent,&reserved));
    MFQ_NATIVE_CUDA_CHECK(cudaMemPoolGetAttribute(pool,cudaMemPoolAttrUsedMemCurrent,&used));
    if(reserved<retained || used!=0 || context->memory_stats().allocated!=0)
        throw std::runtime_error("test did not retain an idle allocator pool");
#ifdef _WIN32
    if(context->local_memory_usage()+retained<=limit)
        throw std::runtime_error("test did not exercise the WDDM pool accounting boundary");
#endif
    {mfq::cuda::Buffer reused(context,retained);context->stream().synchronize();}
    context->stream().synchronize();
    if(context->memory_stats().allocated!=0)throw std::runtime_error("pool reuse leaked an allocation");
    std::cout<<"cuda_memory_pool_reuse retained_bytes="<<reserved<<" allocation_bytes="<<retained<<" PASS\n";
    return 0;
}

int memory_telemetry_case() {
    constexpr auto mib=std::size_t(1024)*1024;
    auto context=std::make_shared<mfq::cuda::Context>(0);
    const auto stream=context->stream().get();
    mfq::cuda::StreamGuard active(mfq::cuda::StreamHandle(0,stream));
    context->begin_graph_pool(stream);
    void* retained=nullptr;
    {
        mfq::cuda::GraphWarmupScope warm(context,stream);
        mfq::cuda::Buffer graph(context,16*mib);retained=graph.data();
    }
    {mfq::cuda::Buffer temporary(context,32*mib);}
    context->stream().synchronize();
    if(context->memory_stats().allocated!=16*mib || context->resident_memory_usage()<16*mib)
        throw std::runtime_error("memory telemetry lost retained graph allocations");
    context->reclaim_idle_memory();
    context->begin_graph_capture(stream);
    {
        mfq::cuda::Buffer replay(context,16*mib);
        if(replay.data()!=retained)throw std::runtime_error("memory reclamation freed live graph storage");
        MFQ_NATIVE_CUDA_CHECK(cudaMemsetAsync(replay.data(),0,replay.size(),stream));
    }
    context->end_graph_capture(stream);context->stream().synchronize();
    context->end_graph_pool(stream);context->reclaim_idle_memory();
    if(context->memory_stats().allocated!=0)throw std::runtime_error("memory reclamation leaked graph storage");
    std::cout<<"cuda_memory_telemetry retained_graph_reuse=PASS allocated_after_release=0\n";
    return 0;
}

int graph_pool_isolation_case() {
    constexpr auto mib=std::size_t(1024)*1024;
    auto context=std::make_shared<mfq::cuda::Context>(0);
    auto observer=std::make_shared<mfq::cuda::Context>(0);
    const auto stream=context->stream().get();
    mfq::cuda::StreamGuard active(mfq::cuda::StreamHandle(0,stream));
    mfq::cuda::Buffer resident(context,16*mib);
    context->stream().synchronize();
    cudaMemPool_t default_pool=nullptr;
    MFQ_NATIVE_CUDA_CHECK(cudaDeviceGetDefaultMemPool(&default_pool,0));
    std::uint64_t baseline=0;
    MFQ_NATIVE_CUDA_CHECK(cudaMemPoolGetAttribute(default_pool,cudaMemPoolAttrUsedMemCurrent,&baseline));
    context->begin_graph_pool(stream);
    {
        mfq::cuda::Buffer outside_capture(context,8*mib);context->stream().synchronize();
        std::uint64_t ordinary=0;
        MFQ_NATIVE_CUDA_CHECK(cudaMemPoolGetAttribute(default_pool,cudaMemPoolAttrUsedMemCurrent,&ordinary));
        if(ordinary<baseline+8*mib)
            throw std::runtime_error("ordinary request storage entered the graph scratch pool");
    }
    std::unique_ptr<mfq::cuda::Buffer> persistent;
    void* warmed=nullptr;
    {
        mfq::cuda::GraphWarmupScope warm(context,stream);
        persistent=std::make_unique<mfq::cuda::Buffer>(context,32*mib);
        mfq::cuda::Buffer scratch(context,16*mib);warmed=scratch.data();
    }
    context->stream().synchronize();
    std::uint64_t current=0;
    MFQ_NATIVE_CUDA_CHECK(cudaMemPoolGetAttribute(default_pool,cudaMemPoolAttrUsedMemCurrent,&current));
    const auto [used,reserved]=observer->pool_memory_stats();
    if(current!=baseline || used<baseline+48*mib || reserved<used)
        throw std::runtime_error("graph pages were mixed with resident weights or omitted across contexts");
    context->reclaim_idle_memory();context->begin_graph_capture(stream);
    {
        mfq::cuda::Buffer reuse(context,16*mib);
        if(reuse.data()!=warmed)throw std::runtime_error("private graph pool lost warmed storage");
        MFQ_NATIVE_CUDA_CHECK(cudaMemsetAsync(reuse.data(),0,reuse.size(),stream));
    }
    context->end_graph_capture(stream);context->stream().synchronize();
    context->end_graph_pool(stream);
    // Persistent request workspaces can outlive a membership change.
    MFQ_NATIVE_CUDA_CHECK(cudaMemsetAsync(persistent->data(),0x35,4,stream));
    std::uint32_t value=0;
    MFQ_NATIVE_CUDA_CHECK(cudaMemcpyAsync(&value,persistent->data(),4,cudaMemcpyDeviceToHost,stream));
    context->stream().synchronize();
    if(value!=0x35353535 || context->memory_stats().allocated!=48*mib)
        throw std::runtime_error("retiring graph pages damaged a persistent request workspace");
    persistent.reset();context->reclaim_idle_memory();
    if(context->memory_stats().allocated!=16*mib)
        throw std::runtime_error("private graph pool leaked its allocation budget");
    for(int i=0;i<128;++i) {
        context->begin_graph_pool(stream);
        {mfq::cuda::GraphWarmupScope warmup(context,stream);
            persistent=std::make_unique<mfq::cuda::Buffer>(context,mib);}
        context->end_graph_pool(stream);
        persistent.reset();context->stream().synchronize();
    }
    context->reclaim_idle_memory();
    if(context->memory_stats().allocated!=16*mib)
        throw std::runtime_error("retired graph pools leaked after repeated captures");
    std::cout<<"cuda_graph_pool_isolation default_pool_unchanged=PASS shared_telemetry=PASS persistent_lifetime=PASS repeated_pools=128\n";
    return 0;
}

int resident_allocation_case() {
    constexpr auto mib=std::size_t(1024)*1024;
    auto context=std::make_shared<mfq::cuda::Context>(0);
    const auto stream=context->stream().get();
    mfq::cuda::StreamGuard active(mfq::cuda::StreamHandle(0,stream));
    context->reclaim_idle_memory();
    cudaMemPool_t temporary_pool=nullptr;
    MFQ_NATIVE_CUDA_CHECK(cudaDeviceGetDefaultMemPool(&temporary_pool,0));
    const auto temporary_used=[&] {
        std::uint64_t used=0;
        MFQ_NATIVE_CUDA_CHECK(cudaMemPoolGetAttribute(temporary_pool,cudaMemPoolAttrUsedMemCurrent,&used));
        return used;
    };
    const auto baseline=temporary_used();
    std::unique_ptr<mfq::cuda::Buffer> resident;
    {
        mfq::cuda::ResidentAllocationScope load;
        resident=std::make_unique<mfq::cuda::Buffer>(context,16*mib);
        context->stream().synchronize();
        if(temporary_used()!=baseline || context->pool_memory_stats().first<baseline+16*mib)
            throw std::runtime_error("resident allocation entered the temporary pool");
        {
            mfq::cuda::ResidentAllocationScope temporary(false);
            mfq::cuda::Buffer dynamic(context,8*mib);context->stream().synchronize();
            if(temporary_used()<baseline+8*mib)
                throw std::runtime_error("nested allocation scope did not restore the async path");
        }
        mfq::cuda::Buffer nested(context,4*mib);context->stream().synchronize();
        if(context->memory_stats().allocated!=20*mib)
            throw std::runtime_error("resident allocation budget is incorrect");
    }
    MFQ_NATIVE_CUDA_CHECK(cudaMemsetAsync(resident->data(),0x39,4,stream));
    std::uint32_t value=0;
    MFQ_NATIVE_CUDA_CHECK(cudaMemcpyAsync(&value,resident->data(),4,cudaMemcpyDeviceToHost,stream));
    context->stream().synchronize();resident.reset();context->reclaim_idle_memory();
    if(value!=0x39393939 || context->memory_stats().allocated)
        throw std::runtime_error("resident storage or stream-ordered release is invalid");
    std::cout<<"cuda_resident_allocation pool_isolation=PASS nested_scope=PASS async_release=PASS\n";
    return 0;
}

int main(int argc,char** argv) {
    int devices = 0;
    MFQ_NATIVE_CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices == 0) {
        return 77;
    }
    if(argc==2 && std::string(argv[1])=="--memory-limit")return memory_limit_cases();
    if(argc==2 && std::string(argv[1])=="--memory-pool-reuse")return memory_pool_reuse_case();
    if(argc==2 && std::string(argv[1])=="--memory-telemetry")return memory_telemetry_case();
    if(argc==2 && std::string(argv[1])=="--graph-pool-isolation") {
        try {return graph_pool_isolation_case();}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc==2 && std::string(argv[1])=="--resident-allocation") {
        try {return resident_allocation_case();}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }

    auto context = std::make_shared<mfq::cuda::Context>(0);
    mfq::cuda::Buffer device(context, sizeof(std::uint32_t) * 4);
    mfq::cuda::HostBuffer host(sizeof(std::uint32_t) * 4);
    auto* values = static_cast<std::uint32_t*>(host.data());
    values[0] = 1;
    values[1] = 2;
    values[2] = 3;
    values[3] = 4;

    MFQ_NATIVE_CUDA_CHECK(cudaMemcpyAsync(
        device.data(),
        host.data(),
        host.size(),
        cudaMemcpyHostToDevice,
        context->stream().get()));
    mfq::cuda::Event copied;
    copied.record(context->stream().get());
    copied.synchronize();
    assert(copied.ready());

    values[0] = 0;
    values[1] = 0;
    values[2] = 0;
    values[3] = 0;
    MFQ_NATIVE_CUDA_CHECK(cudaMemcpyAsync(
        host.data(),
        device.data(),
        host.size(),
        cudaMemcpyDeviceToHost,
        context->stream().get()));
    context->stream().synchronize();
    assert(values[0] == 1);
    assert(values[1] == 2);
    assert(values[2] == 3);
    assert(values[3] == 4);

    context->trim();
}
