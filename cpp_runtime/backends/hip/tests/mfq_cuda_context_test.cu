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

int main(int argc,char** argv) {
    int devices = 0;
    MFQ_NATIVE_CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices == 0) {
        return 77;
    }
    if(argc==2 && std::string(argv[1])=="--memory-limit")return memory_limit_cases();

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
