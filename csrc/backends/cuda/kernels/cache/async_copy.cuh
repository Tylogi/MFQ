#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace mfq::cuda_detail {

__device__ __forceinline__ void copy_16_async(
        __half * destination,
        const __half * source) {
#if __CUDA_ARCH__ >= 800
    const uint32_t shared_address = static_cast<uint32_t>(
        __cvta_generic_to_shared(destination));
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16;\n"
        :: "r"(shared_address), "l"(source) : "memory");
#else
    *reinterpret_cast<int4 *>(destination) =
        *reinterpret_cast<const int4 *>(source);
#endif
}

__device__ __forceinline__ void copy_async_commit() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::: "memory");
#endif
}

__device__ __forceinline__ void copy_async_wait() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
#endif
}

}  // namespace mfq::cuda_detail
