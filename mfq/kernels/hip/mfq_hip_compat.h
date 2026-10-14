// !!! Compatibility shims for hipify-translated CUDA kernels on AMD GPUs.
//
// The hipify pass leaves several CUDA-only constructs in place. This header
// centralizes the translations that are safe to express as preprocessor or
// inline shims. Include it after the regular hip includes of a translation
// unit. It only takes effect on AMD (__HIP_PLATFORM_AMD__).
#pragma once

#ifdef __HIP_PLATFORM_AMD__

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

// gfx9-class GPUs (MI300X) execute 64-lane waves. Passing a partial 32-bit
// mask to a _sync shuffle traps at runtime (hardware exception), and a full
// 64-bit mask would bleed reductions across the two 32-lane halves of the
// wave. HIP's width-parameterized shuffles provide exactly the 32-lane
// segment semantics the translated CUDA code expects (verified on gfx942).
//
// The mask is dropped: the translated kernels only exchange data within
// converged 32-lane groups, which the non-sync variants already satisfy.
#ifndef MFQ_HIP_COMPAT_SHFL
#define MFQ_HIP_COMPAT_SHFL
namespace mfq::hip_compat {
template<typename T>
__device__ __forceinline__ T shuffle(T value, int source, int width = 32) {
    return __shfl(value, source, width);
}
template<typename T>
__device__ __forceinline__ T shuffle_down(T value, unsigned delta, int width = 32) {
    return __shfl_down(value, delta, width);
}
template<typename T>
__device__ __forceinline__ T shuffle_up(T value, unsigned delta, int width = 32) {
    return __shfl_up(value, delta, width);
}
template<typename T>
__device__ __forceinline__ T shuffle_xor(T value, int lane, int width = 32) {
    return __shfl_xor(value, lane, width);
}
}
#define __shfl_sync(mask, ...) ::mfq::hip_compat::shuffle(__VA_ARGS__)
#define __shfl_down_sync(mask, ...) ::mfq::hip_compat::shuffle_down(__VA_ARGS__)
#define __shfl_up_sync(mask, ...) ::mfq::hip_compat::shuffle_up(__VA_ARGS__)
#define __shfl_xor_sync(mask, ...) ::mfq::hip_compat::shuffle_xor(__VA_ARGS__)
#endif

// CUDA bf16 conversion intrinsic absent from hip_bf16.h. HIP's
// __float2bfloat16 already rounds to nearest even.
#if !defined(MFQ_HIP_COMPAT_BALLOT)
#define MFQ_HIP_COMPAT_BALLOT
namespace mfq::hip_compat {
__device__ __forceinline__ unsigned ballot(unsigned mask, unsigned long long votes,
        unsigned thread, unsigned wave_width) {
    return static_cast<unsigned>(votes >> ((thread % wave_width) & 32u)) & mask;
}
}
#define __ballot_sync(mask, predicate) ::mfq::hip_compat::ballot(mask, __ballot(predicate), \
    threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z), warpSize)
#endif

// cuBLAS enum names left behind by partial hipify passes map onto the
// hipBLAS equivalents (identical values).
#if !defined(MFQ_HIP_COMPAT_CUBLAS_ENUMS)
#define MFQ_HIP_COMPAT_CUBLAS_ENUMS
#define CUBLAS_COMPUTE_16F HIPBLAS_COMPUTE_16F
#define CUBLAS_COMPUTE_32F HIPBLAS_COMPUTE_32F
#define CUBLAS_GEMM_DEFAULT HIPBLAS_GEMM_DEFAULT
#define CUBLAS_OP_N HIPBLAS_OP_N
#define CUBLAS_OP_T HIPBLAS_OP_T
#endif

// Preserve CUDA's signed-byte dot product and wrapping accumulator.
#if !defined(MFQ_HIP_COMPAT_DP4A)
#define MFQ_HIP_COMPAT_DP4A
__device__ __forceinline__ int __dp4a(int a, int b, int c) {
#if defined(__HIP_DEVICE_COMPILE__) && \
    (defined(__gfx908__) || defined(__gfx90a__) || defined(__gfx942__)) && \
    __has_builtin(__builtin_amdgcn_sdot4)
    return __builtin_amdgcn_sdot4(a, b, c, false);
#else
    const signed char* av = reinterpret_cast<const signed char*>(&a);
    const signed char* bv = reinterpret_cast<const signed char*>(&b);
    const int dot = av[0] * bv[0] + av[1] * bv[1] + av[2] * bv[2] + av[3] * bv[3];
    return static_cast<int>(static_cast<unsigned>(c) + static_cast<unsigned>(dot));
#endif
}
#endif

// CUDA SIMD per-byte subtract (modular, no saturation) absent from HIP.
// CUDA backoff intrinsic absent from HIP; map to the GCN sleep instruction.
#if !defined(MFQ_HIP_COMPAT_NANOSLEEP)
#define MFQ_HIP_COMPAT_NANOSLEEP
#define __nanosleep(nanoseconds) \
    __builtin_amdgcn_s_sleep(static_cast<unsigned>((nanoseconds) / 64u + 1u))
#endif

#if !defined(MFQ_HIP_COMPAT_VSUB4)
#define MFQ_HIP_COMPAT_VSUB4
__device__ __forceinline__ unsigned int __vsub4(unsigned int a, unsigned int b) {
    const unsigned char* av = reinterpret_cast<const unsigned char*>(&a);
    const unsigned char* bv = reinterpret_cast<const unsigned char*>(&b);
    unsigned char out[4];
    for (int i = 0; i < 4; ++i) out[i] = static_cast<unsigned char>(av[i] - bv[i]);
    unsigned int result;
    __builtin_memcpy(&result, out, sizeof(result));
    return result;
}
#endif

#if !defined(__HIP_NO_BF16_CONVERSIONS__) && !defined(__float2bfloat16_rn)
#define __float2bfloat16_rn(value) __float2bfloat16(value)
#endif


// OCKL sdot4 takes the packed-byte vectors as raw 32-bit registers.
__device__ __attribute__((const)) int __ockl_sdot4_pv(int, int, int, bool);

// CUDA SIMD intrinsics absent from HIP's device functions.
#if !defined(MFQ_HIP_COMPAT_SIMD)
#define MFQ_HIP_COMPAT_SIMD
__device__ __forceinline__ unsigned int __vsubss4(unsigned int a, unsigned int b) {
    const unsigned char* av = reinterpret_cast<const unsigned char*>(&a);
    const unsigned char* bv = reinterpret_cast<const unsigned char*>(&b);
    unsigned char out[4];
    for (int i = 0; i < 4; ++i) {
        int d = int(av[i]) - int(bv[i]);
        out[i] = static_cast<unsigned char>(d < 0 ? 0 : (d > 255 ? 255 : d));
    }
    unsigned int result;
    __builtin_memcpy(&result, out, sizeof(result));
    return result;
}
__device__ __forceinline__ unsigned int __vcmpne4(unsigned int a, unsigned int b) {
    const unsigned char* av = reinterpret_cast<const unsigned char*>(&a);
    const unsigned char* bv = reinterpret_cast<const unsigned char*>(&b);
    unsigned int result = 0;
    for (int i = 0; i < 4; ++i) {
        result |= (av[i] != bv[i]) ? (0xFFu << (8 * i)) : 0;
    }
    return result;
}
__device__ __forceinline__ void __trap() {
    __builtin_trap();
}
#endif
#endif // __HIP_PLATFORM_AMD__
