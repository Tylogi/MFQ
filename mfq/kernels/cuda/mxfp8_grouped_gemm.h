#pragma once
#include "mfq_tensor_backend.h"
#include <cstdint>

mfq_tensor_backend::Tensor mxfp8_groupwise_gemm_f16_cuda(
    mfq_tensor_backend::Tensor values,mfq_tensor_backend::Tensor scales,
    mfq_tensor_backend::Tensor input,std::int64_t groups);
mfq_tensor_backend::Tensor mxfp8_groupwise_gemm_f32_cuda(
    mfq_tensor_backend::Tensor values,mfq_tensor_backend::Tensor scales,
    mfq_tensor_backend::Tensor input,std::int64_t groups);
