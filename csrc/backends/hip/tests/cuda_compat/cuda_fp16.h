#pragma once
// Test-facing fp16 shim that also pulls the cuda_* runtime renames, for
// test sources that include only <cuda_fp16.h>.
#include "hip/hip_fp16.h"
#include "cuda_runtime.h"
