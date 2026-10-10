#pragma once
#include "quant_linear.h"
#include "cuda_execution.h"
#include "mfq_cuda_context.h"
#include <cstring>
#include <memory>

// Storage strategy adapted from Strata NativeEmbed::load/gather_dev:
// references/strata/src/core/native_head.cpp, MIT license in
// cpp_runtime/third_party/strata.LICENSE. Preserve MFQ's native floating dtype
// and gather kernel; only the read-only embedding table changes placement.
inline QuantLinear map_dense_embedding(const mfq_tensor_backend::Tensor& cpu,int device,
        bool device_read_control=false) {
    namespace tb=mfq_tensor_backend;
    MFQ_RUNTIME_CHECK(cpu.is_cpu() && cpu.is_contiguous() && cpu.dim()==2 &&
        (cpu.scalar_type()==tb::kFloat16 || cpu.scalar_type()==tb::kBFloat16 || cpu.scalar_type()==tb::kFloat32),
        "mapped embedding requires a contiguous rank-2 floating CPU table");
    const auto bytes=std::size_t(cpu.numel())*cpu.element_size();
    MFQ_RUNTIME_CHECK(bytes>0,"mapped embedding table is empty");
    struct Owner {
        mfq::cuda::HostBuffer host;
        tb::Tensor device_copy;
        std::size_t bytes;
        explicit Owner(std::size_t count):host(count,true),bytes(count) {
#ifdef MFQ_NATIVE_CUDA_RUNTIME
            mfq::cuda::charge_tensor_host_bytes(bytes);
#endif
        }
        ~Owner() {
#ifdef MFQ_NATIVE_CUDA_RUNTIME
            mfq::cuda::tensor_host_bytes.fetch_sub(bytes);
#endif
        }
    };
    MfqCudaGuard guard(device);
    auto owner=std::make_shared<Owner>(bytes);
    std::memcpy(owner->host.data(),cpu.data_ptr(),bytes);
    void* alias=nullptr;
    MFQ_CUDA_CHECK(cudaHostGetDevicePointer(&alias,owner->host.data(),0));
    MFQ_RUNTIME_CHECK(alias,"mapped embedding has no device alias");
    const auto gpu_device=tb::Device(tb::kCUDA,device);
    // Diagnostic control: retain the identical mapped allocation, but gather
    // from a GPU copy. This separates pinning/placement from remote GPU reads.
    if(device_read_control) {
        owner->device_copy=cpu.to(gpu_device);
        alias=owner->device_copy.data_ptr();
    }
    QuantLinear result;
    result.kind=QuantLinearKind::Dense;
    result.tensor_parallel_axis=TensorParallelAxis::Mirrored;
    result.logical_out=cpu.size(0);result.logical_neuron_len=cpu.size(1);
#ifdef MFQ_NATIVE_CUDA_RUNTIME
    auto storage=std::make_shared<tb::TensorStorage>();
    storage->owner=std::move(owner);storage->base=alias;storage->bytes=bytes;storage->device=gpu_device;
    result.dense=tb::Tensor(storage,tb::make_contiguous_view(alias,cpu.sizes(),cpu.scalar_type(),gpu_device));
#else
    result.dense=tb::from_blob(alias,cpu.sizes(),[owner](void*){},cpu.options().device(gpu_device));
#endif
    return result;
}
