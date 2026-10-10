#pragma once

#include "mfq_tensor_backend.h"

#include <cstdint>
#include <vector>

std::vector<mfq_tensor_backend::Tensor> linear_gate_beta_cuda(
    mfq_tensor_backend::Tensor alpha, mfq_tensor_backend::Tensor beta, mfq_tensor_backend::Tensor dt_bias, mfq_tensor_backend::Tensor a_log,
    bool stable_softplus = false);
std::vector<mfq_tensor_backend::Tensor> gdn_cuda(mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor k, mfq_tensor_backend::Tensor v,
                                    mfq_tensor_backend::Tensor g, mfq_tensor_backend::Tensor beta, MfqOptional<mfq_tensor_backend::Tensor> state);
std::vector<mfq_tensor_backend::Tensor> gdn_transposed_cuda(
    mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor k, mfq_tensor_backend::Tensor v,
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
std::vector<mfq_tensor_backend::Tensor> linear_conv_qkv_gate_decode_cuda(
    mfq_tensor_backend::Tensor state, mfq_tensor_backend::Tensor qk, mfq_tensor_backend::Tensor v,
    mfq_tensor_backend::Tensor weight, mfq_tensor_backend::Tensor alpha, mfq_tensor_backend::Tensor beta,
    mfq_tensor_backend::Tensor dt_bias, mfq_tensor_backend::Tensor a_log,
    int64_t nk, int64_t nv, int64_t dk, int64_t dv, double eps);

// Workspace counters start at zero and reset after each completed decode.
// With inplace_state, every CTA owns disjoint recurrent columns and returns the
// updated input storage. The caller must exclusively own that recurrent state.
std::vector<mfq_tensor_backend::Tensor> gdn_decode_core_cuda(
    mfq_tensor_backend::Tensor qkv, mfq_tensor_backend::Tensor output_gate,
    mfq_tensor_backend::Tensor alpha, mfq_tensor_backend::Tensor beta,
    mfq_tensor_backend::Tensor convolution_state, mfq_tensor_backend::Tensor recurrent_state,
    mfq_tensor_backend::Tensor convolution_weight, mfq_tensor_backend::Tensor dt_bias,
    mfq_tensor_backend::Tensor a_log, mfq_tensor_backend::Tensor norm_weight,
    int64_t key_heads, int64_t value_heads, int64_t width,
    double convolution_eps, double norm_eps, bool silu_gate, bool output_half,
    bool transposed_state,
    mfq_tensor_backend::Tensor workspace={},bool inplace_state=false);
