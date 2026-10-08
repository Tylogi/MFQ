#pragma once
#include "nint.h"
#include <vector>

// Two fused residual kernels with runtime geometry and quantization metadata.
// BF16 injection uses the original FP32 cuBLAS projection between these kernels.
std::vector<mfq_tensor_backend::Tensor> gated_residual_two_stage_cuda(
    const mfq_tensor_backend::Tensor& branch,
    const mfq_tensor_backend::Tensor& residual,
    const mfq_tensor_backend::Tensor& prior_injection,
    const mfq_tensor_backend::Tensor& norm,
    const NintWeight& down,const NintWeight& up,const NintWeight* injection,
    int64_t streams,double eps,bool after,
    const mfq_tensor_backend::Tensor& dense_injection={},
    const mfq_tensor_backend::Tensor& prepared_injection_right={});
