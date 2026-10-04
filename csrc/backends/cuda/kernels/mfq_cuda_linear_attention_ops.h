#pragma once

#include "../native/tensor_backend.h"

#include <cstdint>
#include <vector>

std::vector<mfq_tensor_backend::Tensor> linear_gate_beta_cuda(
    mfq_tensor_backend::Tensor alpha, mfq_tensor_backend::Tensor beta, mfq_tensor_backend::Tensor dt_bias, mfq_tensor_backend::Tensor a_log);
std::vector<mfq_tensor_backend::Tensor> gdn_cuda(mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor k, mfq_tensor_backend::Tensor v,
                                    mfq_tensor_backend::Tensor g, mfq_tensor_backend::Tensor beta, MfqOptional<mfq_tensor_backend::Tensor> state);
std::vector<mfq_tensor_backend::Tensor> gdn_inplace_cuda(mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor k, mfq_tensor_backend::Tensor v,
                                             mfq_tensor_backend::Tensor g, mfq_tensor_backend::Tensor beta, mfq_tensor_backend::Tensor state);
std::vector<mfq_tensor_backend::Tensor> gdn_inplace_transposed_cuda(
    mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor k, mfq_tensor_backend::Tensor v,
    mfq_tensor_backend::Tensor g, mfq_tensor_backend::Tensor beta, mfq_tensor_backend::Tensor state);
std::vector<mfq_tensor_backend::Tensor> gdn_inplace_tiled_cuda(
    mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor k, mfq_tensor_backend::Tensor v,
    mfq_tensor_backend::Tensor g, mfq_tensor_backend::Tensor beta, mfq_tensor_backend::Tensor state);
std::vector<mfq_tensor_backend::Tensor> gdn_inplace_transposed_tiled_cuda(
    mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor k, mfq_tensor_backend::Tensor v,
    mfq_tensor_backend::Tensor g, mfq_tensor_backend::Tensor beta, mfq_tensor_backend::Tensor state);
mfq_tensor_backend::Tensor ssm_conv_silu_cuda(mfq_tensor_backend::Tensor conv_input, mfq_tensor_backend::Tensor weight, mfq_tensor_backend::Tensor bias, int64_t n_tokens);
mfq_tensor_backend::Tensor ssm_conv_silu_decode_cuda(mfq_tensor_backend::Tensor state, mfq_tensor_backend::Tensor x, mfq_tensor_backend::Tensor weight, mfq_tensor_backend::Tensor bias);
std::vector<mfq_tensor_backend::Tensor> linear_conv_qkv_decode_cuda(
    mfq_tensor_backend::Tensor state, mfq_tensor_backend::Tensor qk, mfq_tensor_backend::Tensor v, mfq_tensor_backend::Tensor weight, mfq_tensor_backend::Tensor bias,
    int64_t nk, int64_t nv, int64_t dk, int64_t dv, double eps);
std::vector<mfq_tensor_backend::Tensor> linear_conv_qkv_prefill_cuda(
    mfq_tensor_backend::Tensor state, mfq_tensor_backend::Tensor qk, mfq_tensor_backend::Tensor v, mfq_tensor_backend::Tensor weight, mfq_tensor_backend::Tensor bias,
    int64_t nk, int64_t nv, int64_t dk, int64_t dv, double eps);
