#pragma once
#include "ops/quant/quant_linear.h"
#include "ops/execution/cuda_execution.h"
#include "native/context.h"
#include <cstring>
#include <memory>

// Storage strategy adapted from Strata NativeEmbed::load/gather_dev:
// references/strata/src/core/native_head.cpp, MIT license in
// csrc/third_party/strata.LICENSE. Preserve MFQ's native floating dtype
// and gather kernel; only the read-only embedding table changes placement.
inline mfq_tensor_backend::Tensor map_embedding_storage(const mfq_tensor_backend::Tensor& cpu,int device,
        bool device_read_control=false) {
    namespace tb=mfq_tensor_backend;
    MFQ_RUNTIME_CHECK(cpu.is_cpu() && cpu.is_contiguous(),
        "mapped embedding requires contiguous CPU storage");
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
#ifdef MFQ_NATIVE_CUDA_RUNTIME
    auto storage=std::make_shared<tb::TensorStorage>();
    storage->owner=std::move(owner);storage->base=alias;storage->bytes=bytes;storage->device=gpu_device;
    return tb::Tensor(storage,tb::make_contiguous_view(alias,cpu.sizes(),cpu.scalar_type(),gpu_device));
#else
    return tb::from_blob(alias,cpu.sizes(),[owner](void*){},cpu.options().device(gpu_device));
#endif
}

inline QuantLinear map_dense_embedding(const mfq_tensor_backend::Tensor& cpu,int device,
        bool device_read_control=false) {
    namespace tb=mfq_tensor_backend;
    MFQ_RUNTIME_CHECK(cpu.dim()==2 &&
        (cpu.scalar_type()==tb::kFloat16 || cpu.scalar_type()==tb::kBFloat16 || cpu.scalar_type()==tb::kFloat32),
        "mapped embedding requires a rank-2 floating table");
    QuantLinear result;
    result.kind=QuantLinearKind::Dense;
    result.tensor_parallel_axis=TensorParallelAxis::Mirrored;
    result.logical_out=cpu.size(0);result.logical_neuron_len=cpu.size(1);
    result.dense=map_embedding_storage(cpu,device,device_read_control);
    return result;
}

inline QuantLinear map_nint_embedding(QuantLinear result,int device,bool device_read_control=false) {
    namespace tb=mfq_tensor_backend;
    MFQ_RUNTIME_CHECK(result.is_nint() && !result.tensor_parallel() && result.nint.q_packed.is_cpu(),
        "mapped NINT embedding requires an unsharded CPU table");
    auto& w=result.nint;
    // Only selected vocabulary rows cross PCIe. Keep small row descriptors on
    // the GPU so their dependent loads do not add host round trips.
    for(auto* field:{&w.q_packed,&w.sub_scale,&w.sub_min,&w.q8_zero_scale})
        if(field->defined())*field=map_embedding_storage(*field,device,device_read_control);
    for(auto* field:{&w.row_q_bits,&w.row_q_bit_offsets,&w.neuron_scale,&w.neuron_min})
        if(field->defined())*field=field->to(tb::Device(tb::kCUDA,device));
    return result;
}
