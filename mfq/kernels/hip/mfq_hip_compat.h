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
#if !defined(__HIP_NO_BF16_CONVERSIONS__) && !defined(__float2bfloat16_rn)
#define __float2bfloat16_rn(value) __float2bfloat16(value)
#endif

#endif // __HIP_PLATFORM_AMD__
