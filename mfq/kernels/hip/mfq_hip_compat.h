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
// Call sites that pass an explicit sub-warp width (8/16) must be rewritten
// to the direct width form (__shfl_down(var, delta, 8)); they cannot go
// through these macros.
#ifndef MFQ_HIP_COMPAT_SHFL
#define MFQ_HIP_COMPAT_SHFL
#define __shfl_sync(mask, var, src, ...) __shfl(var, src, 32)
#define __shfl_down_sync(mask, var, delta, ...) __shfl_down(var, delta, 32)
#define __shfl_up_sync(mask, var, delta, ...) __shfl_up(var, delta, 32)
#define __shfl_xor_sync(mask, var, lane, ...) __shfl_xor(var, lane, 32)
#endif

// CUDA bf16 conversion intrinsic absent from hip_bf16.h. HIP's
// __float2bfloat16 already rounds to nearest even.
// Ballot over the wave, truncated to the low 32 lanes, which matches the
// 32-lane warp groups of the translated CUDA code.
#if !defined(MFQ_HIP_COMPAT_BALLOT)
#define MFQ_HIP_COMPAT_BALLOT
#define __ballot_sync(mask, predicate) ((unsigned)__ballot(predicate))
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

// ROCm does not provide CUDA's __dp4a intrinsic; emulate it in device code.
#if !defined(MFQ_HIP_COMPAT_DP4A)
#define MFQ_HIP_COMPAT_DP4A
__device__ __forceinline__ int __dp4a(int a, int b, int c) {
    // Scalar byte dot product; correct but compute-limited. The hardware
    // v_dot4_i32_i8 exists on gfx942 but its inline-asm encoding differs
    // from the documented form (produced wrong results). The OCKL
    // __ockl_sdot4 intrinsic requires a header chain not on our path.
    // Closing the ~15% decode gap to BF16 needs the hardware instruction.
    const signed char* av = reinterpret_cast<const signed char*>(&a);
    const signed char* bv = reinterpret_cast<const signed char*>(&b);
    return c + av[0] * bv[0] + av[1] * bv[1] + av[2] * bv[2] + av[3] * bv[3];
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
