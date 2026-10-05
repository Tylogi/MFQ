#pragma once
#include "mfq_tensor_backend.h"
#include <functional>
#include <vector>

struct CudaExecutionContext;
using GatedResidualProjection = std::function<mfq_tensor_backend::Tensor(
    CudaExecutionContext&, const mfq_tensor_backend::Tensor&)>;

std::vector<mfq_tensor_backend::Tensor> gated_residual_pre_projected(
    CudaExecutionContext&, const mfq_tensor_backend::Tensor& input,
    const mfq_tensor_backend::Tensor& norm, const GatedResidualProjection& down,
    const GatedResidualProjection& up, const GatedResidualProjection& injection,
    int64_t hidden, int64_t streams, double eps);
