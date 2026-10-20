#pragma once
// Shim: ggml headers reference CUDA's nv_bfloat16 types; map to HIP's.
#include <hip/hip_bf16.h>
using nv_bfloat16 = __hip_bfloat16;
using nv_bfloat162 = __hip_bfloat162;
