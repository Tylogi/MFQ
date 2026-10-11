#pragma once
#include "nint.h"
#include <functional>
#include <vector>

// Replace the existing F32 [column, stream] buffer with an equal-size
// [column/4, stream, item] buffer. Its rank explicitly identifies the layout.
mfq_tensor_backend::Tensor prepare_gr_dense_vector_right(
    const mfq_tensor_backend::Tensor& right);

// Fused norm, Down/injection, and Up/mix residual stages. Markers instrument
// the same stages used by ordinary decode without replacing their arithmetic.
std::vector<mfq_tensor_backend::Tensor> gated_residual_two_stage_cuda(
    const mfq_tensor_backend::Tensor& branch,
    const mfq_tensor_backend::Tensor& residual,
    const mfq_tensor_backend::Tensor& prior_injection,
    const mfq_tensor_backend::Tensor& norm,
    const NintWeight& down,const NintWeight& up,const NintWeight* injection,
    int64_t streams,double eps,bool after,
    const mfq_tensor_backend::Tensor& dense_injection={},
    const mfq_tensor_backend::Tensor& prepared_injection_right={},
    const std::function<void(const char*)>& marker={});
