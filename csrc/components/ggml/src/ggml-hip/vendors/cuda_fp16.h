#pragma once
// Shim: ggml-common.h includes <cuda_fp16.h>; on HIP it maps to hip_fp16.
#include <hip/hip_fp16.h>
