#pragma once

#include "native/tensor/backend.h"
#include <vector>

mfq_tensor_backend::Tensor deepseek_v41_engram_rows_cuda(
    mfq_tensor_backend::Tensor packed, mfq_tensor_backend::Tensor inverse, int64_t width);

mfq_tensor_backend::Tensor deepseek_v41_engram_gate_add_cuda(
    mfq_tensor_backend::Tensor hidden, mfq_tensor_backend::Tensor projected,
    mfq_tensor_backend::Tensor query_key_weight, double epsilon);

std::vector<mfq_tensor_backend::Tensor> deepseek_v41_hc_prepare_cuda(
    mfq_tensor_backend::Tensor residual,
    mfq_tensor_backend::Tensor previous_pre,
    mfq_tensor_backend::Tensor norm,
    double epsilon);

mfq_tensor_backend::Tensor deepseek_v41_hc_input_cuda(
    mfq_tensor_backend::Tensor residual,
    mfq_tensor_backend::Tensor previous_pre,
    mfq_tensor_backend::Tensor norm,
    double epsilon);

mfq_tensor_backend::Tensor deepseek_v41_mxfp8_e4m3_sim_cuda(
    mfq_tensor_backend::Tensor input);

mfq_tensor_backend::Tensor deepseek_v41_mxfp4_e4m3_scale_sim_cuda(
    mfq_tensor_backend::Tensor input);

mfq_tensor_backend::Tensor deepseek_v41_hc_project_cuda(
    mfq_tensor_backend::Tensor residual, mfq_tensor_backend::Tensor previous,
    mfq_tensor_backend::Tensor function);

std::vector<mfq_tensor_backend::Tensor> deepseek_v41_hc_finalize_cuda(
    mfq_tensor_backend::Tensor residual, mfq_tensor_backend::Tensor previous,
    mfq_tensor_backend::Tensor norm, mfq_tensor_backend::Tensor partials,
    mfq_tensor_backend::Tensor scale, mfq_tensor_backend::Tensor base,
    int64_t iterations, double rms_epsilon, double hc_epsilon);
