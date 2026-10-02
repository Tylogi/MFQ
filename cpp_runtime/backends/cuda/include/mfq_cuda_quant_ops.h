#pragma once

#include "mfq_tensor_backend.h"

#include <cstdint>
#include <vector>

mfq_tensor_backend::Tensor embedding_lookup_cuda(
    mfq_tensor_backend::Tensor weight, mfq_tensor_backend::Tensor token_ids);
mfq_tensor_backend::Tensor nepq_hadamard_input_cuda(
    mfq_tensor_backend::Tensor input, mfq_tensor_backend::Tensor signs, int64_t block_size);
mfq_tensor_backend::Tensor nint_embedding_cuda(
    mfq_tensor_backend::Tensor q_packed, mfq_tensor_backend::Tensor row_q_bits,
    mfq_tensor_backend::Tensor row_q_bit_offsets,
    mfq_tensor_backend::Tensor sub_scale, mfq_tensor_backend::Tensor sub_min,
    mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor neuron_min,
    mfq_tensor_backend::Tensor token_ids, int64_t neuron_len, int64_t gs);
mfq_tensor_backend::Tensor nint8_zero_embedding_lookup_cuda(
    mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor scale, mfq_tensor_backend::Tensor token_ids,
    int64_t neuron_len);
std::vector<mfq_tensor_backend::Tensor> nint8_one_quantize_reconstruct_cuda(
    mfq_tensor_backend::Tensor x);
mfq_tensor_backend::Tensor nint8_zero_mmq_f16_packed_cuda(
    mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor scale, mfq_tensor_backend::Tensor x,
    int64_t neuron_len);
mfq_tensor_backend::Tensor nint8_zero_mmq_f32_packed_cuda(
    mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor scale, mfq_tensor_backend::Tensor x,
    int64_t neuron_len);
mfq_tensor_backend::Tensor nint_matmul_ws_cuda(
    mfq_tensor_backend::Tensor q_packed, mfq_tensor_backend::Tensor row_q_bits,
    mfq_tensor_backend::Tensor row_q_bit_offsets,
    mfq_tensor_backend::Tensor sub_scale, mfq_tensor_backend::Tensor sub_min,
    mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor neuron_min,
    mfq_tensor_backend::Tensor x, int64_t gs,
    mfq_tensor_backend::Tensor qx, mfq_tensor_backend::Tensor xscale);
// Caller proves every row is q8 with a 32-bit-aligned row bit offset.
// Misaligned storage or a non-four-element group uses the generic kernel.
mfq_tensor_backend::Tensor nint_matmul_q8_ws_cuda(
    mfq_tensor_backend::Tensor q_packed, mfq_tensor_backend::Tensor row_q_bits,
    mfq_tensor_backend::Tensor row_q_bit_offsets,
    mfq_tensor_backend::Tensor sub_scale, mfq_tensor_backend::Tensor sub_min,
    mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor neuron_min,
    mfq_tensor_backend::Tensor x, int64_t gs,
    mfq_tensor_backend::Tensor qx, mfq_tensor_backend::Tensor xscale);
mfq_tensor_backend::Tensor nint_matmul_input_mul_ws_cuda(
    mfq_tensor_backend::Tensor q_packed, mfq_tensor_backend::Tensor row_q_bits,
    mfq_tensor_backend::Tensor row_q_bit_offsets,
    mfq_tensor_backend::Tensor sub_scale, mfq_tensor_backend::Tensor sub_min,
    mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor neuron_min,
    mfq_tensor_backend::Tensor x, mfq_tensor_backend::Tensor gate,
    int64_t activation_mode, int64_t gs,
    mfq_tensor_backend::Tensor qx, mfq_tensor_backend::Tensor xscale);
// Same all-row q8 proof and storage fallback as nint_matmul_q8_ws_cuda.
mfq_tensor_backend::Tensor nint_matmul_input_mul_q8_ws_cuda(
    mfq_tensor_backend::Tensor q_packed, mfq_tensor_backend::Tensor row_q_bits,
    mfq_tensor_backend::Tensor row_q_bit_offsets,
    mfq_tensor_backend::Tensor sub_scale, mfq_tensor_backend::Tensor sub_min,
    mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor neuron_min,
    mfq_tensor_backend::Tensor x, mfq_tensor_backend::Tensor gate,
    int64_t activation_mode, int64_t gs,
    mfq_tensor_backend::Tensor qx, mfq_tensor_backend::Tensor xscale);
mfq_tensor_backend::Tensor nint8_zero_gemv_ws_cuda(
    mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor scale, mfq_tensor_backend::Tensor x,
    mfq_tensor_backend::Tensor qx, mfq_tensor_backend::Tensor xscale);
mfq_tensor_backend::Tensor nint8_zero_dequant_cuda(
    mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor scale, int64_t neuron_len);
mfq_tensor_backend::Tensor nint_decode_cuda(
    mfq_tensor_backend::Tensor q_packed, mfq_tensor_backend::Tensor row_q_bits,
    mfq_tensor_backend::Tensor row_q_bit_offsets,
    mfq_tensor_backend::Tensor sub_scale, mfq_tensor_backend::Tensor sub_min,
    mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor neuron_min,
    int64_t neuron_len, int64_t gs);
mfq_tensor_backend::Tensor nint_cublas_gemm_nt_f32acc_cuda(mfq_tensor_backend::Tensor x, mfq_tensor_backend::Tensor w);
mfq_tensor_backend::Tensor mxfp8_dequant_cuda(
    mfq_tensor_backend::Tensor values, mfq_tensor_backend::Tensor scales);
mfq_tensor_backend::Tensor mxfp8_embedding_lookup_cuda(
    mfq_tensor_backend::Tensor values, mfq_tensor_backend::Tensor scales, mfq_tensor_backend::Tensor token_ids);
mfq_tensor_backend::Tensor mxfp8_small_m_cuda(
    mfq_tensor_backend::Tensor values, mfq_tensor_backend::Tensor scales, mfq_tensor_backend::Tensor x);
mfq_tensor_backend::Tensor mxfp8_matmul_f16_cuda(
    mfq_tensor_backend::Tensor values, mfq_tensor_backend::Tensor scales, mfq_tensor_backend::Tensor input);
mfq_tensor_backend::Tensor mxfp8_small_m_f32_cuda(
    mfq_tensor_backend::Tensor values, mfq_tensor_backend::Tensor scales, mfq_tensor_backend::Tensor x);
mfq_tensor_backend::Tensor mxfp8_gemm_f32_cuda(
    mfq_tensor_backend::Tensor values, mfq_tensor_backend::Tensor scales, mfq_tensor_backend::Tensor x);
mfq_tensor_backend::Tensor mxfp8_groupwise_small_m_cuda(
    mfq_tensor_backend::Tensor values, mfq_tensor_backend::Tensor scales,
    mfq_tensor_backend::Tensor x, int64_t groups);
mfq_tensor_backend::Tensor mxfp8_groupwise_small_m_f32_cuda(
    mfq_tensor_backend::Tensor values, mfq_tensor_backend::Tensor scales,
    mfq_tensor_backend::Tensor x, int64_t groups);
mfq_tensor_backend::Tensor mxfp4_dequant_cuda(
    mfq_tensor_backend::Tensor values, mfq_tensor_backend::Tensor scales);
mfq_tensor_backend::Tensor mxfp4_embedding_lookup_cuda(
    mfq_tensor_backend::Tensor values, mfq_tensor_backend::Tensor scales, mfq_tensor_backend::Tensor token_ids);
mfq_tensor_backend::Tensor mxfp4_matmul_f16_cuda(
    mfq_tensor_backend::Tensor values, mfq_tensor_backend::Tensor scales, mfq_tensor_backend::Tensor input);
mfq_tensor_backend::Tensor nvq_dequant_cuda(
    mfq_tensor_backend::Tensor indices, mfq_tensor_backend::Tensor aux, mfq_tensor_backend::Tensor sub_scale,
    mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor codebook,
    int64_t neuron_len, int64_t gs, int64_t sub_bits, int64_t format, int64_t sign_mode);
mfq_tensor_backend::Tensor nepq_dequant_cuda(
    mfq_tensor_backend::Tensor indices, mfq_tensor_backend::Tensor aux, mfq_tensor_backend::Tensor sub_scale,
    mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor table_pool,
    mfq_tensor_backend::Tensor bank_ids, int64_t neuron_len,
    int64_t sub_bits, int64_t format);
mfq_tensor_backend::Tensor nepq_sparse_residual_dequant_cuda(
    mfq_tensor_backend::Tensor dictionary, mfq_tensor_backend::Tensor first, mfq_tensor_backend::Tensor second,
    int64_t position_bits, int64_t block_vectors, mfq_tensor_backend::Tensor weight);
mfq_tensor_backend::Tensor nvq_gemm_f16_cuda(
    mfq_tensor_backend::Tensor indices, mfq_tensor_backend::Tensor aux, mfq_tensor_backend::Tensor sub_scale,
    mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor codebook, mfq_tensor_backend::Tensor x,
    int64_t neuron_len, int64_t gs, int64_t sub_bits, int64_t format, int64_t sign_mode);
mfq_tensor_backend::Tensor nvq_gemv_ws_cuda(
    mfq_tensor_backend::Tensor indices, mfq_tensor_backend::Tensor aux, mfq_tensor_backend::Tensor sub_scale,
    mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor codebook, mfq_tensor_backend::Tensor x,
    int64_t neuron_len, int64_t gs, int64_t sub_bits, int64_t format, int64_t sign_mode,
    mfq_tensor_backend::Tensor qx, mfq_tensor_backend::Tensor xscale);
mfq_tensor_backend::Tensor nvq_gemv_qx_ws_cuda(
    mfq_tensor_backend::Tensor indices, mfq_tensor_backend::Tensor aux, mfq_tensor_backend::Tensor sub_scale,
    mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor codebook,
    int64_t neuron_len, int64_t gs, int64_t sub_bits, int64_t format, int64_t sign_mode,
    mfq_tensor_backend::Tensor qx, mfq_tensor_backend::Tensor xscale);
mfq_tensor_backend::Tensor nvq_gemv_qx_residual_ws_cuda(
    mfq_tensor_backend::Tensor indices, mfq_tensor_backend::Tensor aux, mfq_tensor_backend::Tensor sub_scale,
    mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor codebook,
    int64_t neuron_len, int64_t gs, int64_t sub_bits, int64_t format,
    int64_t sign_mode, mfq_tensor_backend::Tensor qx, mfq_tensor_backend::Tensor xscale,
    mfq_tensor_backend::Tensor residual);
mfq_tensor_backend::Tensor nvq_gemv_multi2_ws_cuda(
    mfq_tensor_backend::Tensor first_indices, mfq_tensor_backend::Tensor first_aux, mfq_tensor_backend::Tensor first_sub_scale,
    mfq_tensor_backend::Tensor first_neuron_scale, mfq_tensor_backend::Tensor first_codebook,
    mfq_tensor_backend::Tensor second_indices, mfq_tensor_backend::Tensor second_aux, mfq_tensor_backend::Tensor second_sub_scale,
    mfq_tensor_backend::Tensor second_neuron_scale, mfq_tensor_backend::Tensor second_codebook,
    mfq_tensor_backend::Tensor x, int64_t neuron_len, int64_t gs,
    int64_t first_sub_bits, int64_t first_format, int64_t first_sign_mode,
    int64_t second_sub_bits, int64_t second_format, int64_t second_sign_mode,
    mfq_tensor_backend::Tensor qx, mfq_tensor_backend::Tensor xscale);
mfq_tensor_backend::Tensor nvq_gemv_swiglu_ws_cuda(
    mfq_tensor_backend::Tensor gate_indices, mfq_tensor_backend::Tensor gate_aux, mfq_tensor_backend::Tensor gate_sub_scale,
    mfq_tensor_backend::Tensor gate_neuron_scale, mfq_tensor_backend::Tensor gate_codebook,
    mfq_tensor_backend::Tensor up_indices, mfq_tensor_backend::Tensor up_aux, mfq_tensor_backend::Tensor up_sub_scale,
    mfq_tensor_backend::Tensor up_neuron_scale, mfq_tensor_backend::Tensor up_codebook,
    mfq_tensor_backend::Tensor x, int64_t neuron_len, int64_t gs,
    int64_t gate_sub_bits, int64_t gate_format, int64_t gate_sign_mode,
    int64_t up_sub_bits, int64_t up_format, int64_t up_sign_mode,
    mfq_tensor_backend::Tensor qx, mfq_tensor_backend::Tensor xscale);
void nvq_ffn_swiglu_quant_ws_cuda(
    mfq_tensor_backend::Tensor gate_indices, mfq_tensor_backend::Tensor gate_aux, mfq_tensor_backend::Tensor gate_sub_scale,
    mfq_tensor_backend::Tensor gate_neuron_scale, mfq_tensor_backend::Tensor gate_codebook,
    mfq_tensor_backend::Tensor up_indices, mfq_tensor_backend::Tensor up_aux, mfq_tensor_backend::Tensor up_sub_scale,
    mfq_tensor_backend::Tensor up_neuron_scale, mfq_tensor_backend::Tensor up_codebook,
    mfq_tensor_backend::Tensor x, int64_t neuron_len, int64_t gs,
    int64_t gate_sub_bits, int64_t gate_format, int64_t gate_sign_mode,
    int64_t up_sub_bits, int64_t up_format, int64_t up_sign_mode, int64_t down_gs,
    mfq_tensor_backend::Tensor input_qx, mfq_tensor_backend::Tensor input_xscale,
    mfq_tensor_backend::Tensor output_qx, mfq_tensor_backend::Tensor output_xscale, mfq_tensor_backend::Tensor swiglu_scratch);
mfq_tensor_backend::Tensor nvq_gemv_gate_ws_cuda(
    mfq_tensor_backend::Tensor indices, mfq_tensor_backend::Tensor aux, mfq_tensor_backend::Tensor sub_scale,
    mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor codebook, mfq_tensor_backend::Tensor x, mfq_tensor_backend::Tensor gate,
    int64_t neuron_len, int64_t gs, int64_t sub_bits, int64_t format, int64_t sign_mode,
    int64_t mode, mfq_tensor_backend::Tensor qx, mfq_tensor_backend::Tensor xscale);
mfq_tensor_backend::Tensor nvq_mmq_ws_cuda(
    mfq_tensor_backend::Tensor indices, mfq_tensor_backend::Tensor aux, mfq_tensor_backend::Tensor sub_scale,
    mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor codebook, mfq_tensor_backend::Tensor x,
    int64_t neuron_len, int64_t gs, int64_t sub_bits, int64_t format, int64_t sign_mode,
    mfq_tensor_backend::Tensor qx, mfq_tensor_backend::Tensor xscale);
mfq_tensor_backend::Tensor nvq_mmq_gate_ws_cuda(
    mfq_tensor_backend::Tensor indices, mfq_tensor_backend::Tensor aux, mfq_tensor_backend::Tensor sub_scale,
    mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor codebook, mfq_tensor_backend::Tensor x, mfq_tensor_backend::Tensor gate,
    int64_t neuron_len, int64_t gs, int64_t sub_bits, int64_t format, int64_t sign_mode,
    int64_t mode, mfq_tensor_backend::Tensor qx, mfq_tensor_backend::Tensor xscale);
mfq_tensor_backend::Tensor nvq_embedding_lookup_cuda(
    mfq_tensor_backend::Tensor indices, mfq_tensor_backend::Tensor aux, mfq_tensor_backend::Tensor sub_scale,
    mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor codebook, mfq_tensor_backend::Tensor token_ids,
    int64_t neuron_len, int64_t gs, int64_t sub_bits, int64_t format, int64_t sign_mode);
