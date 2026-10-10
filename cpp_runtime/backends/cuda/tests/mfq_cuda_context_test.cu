#include "mfq_cuda_context.h"

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

int main(int argc,char** argv) {
    int devices = 0;
    MFQ_NATIVE_CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices == 0) {
        return 77;
    }
    if(argc==2 && std::string(argv[1])=="--memory-limit")return memory_limit_cases();
    if(argc==2 && std::string(argv[1])=="--memory-pool-reuse")return memory_pool_reuse_case();

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
