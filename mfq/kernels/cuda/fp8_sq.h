#pragma once

#include <cstdint>

#include "cpp_runtime/backends/cuda/include/mfq_tensor_backend.h"
#include "mfq/fp8_sq_blob.h"

// The CPU loader validates the self-describing wire payload and expands the
// compact three-bit q descriptor into row_q/row_symbol_byte_offsets.  CUDA
// never reads metadata back from the device and never keeps a full decoded
// weight resident.
mfq_tensor_backend::Tensor mxfp8_sq_dequant_cuda(
    mfq_tensor_backend::Tensor blob,
    mfq_tensor_backend::Tensor row_q,
    mfq_tensor_backend::Tensor row_symbol_byte_offsets,
    std::int64_t outputs, std::int64_t width,
    std::int64_t block_rows, std::int64_t block_columns,
    std::int64_t scale_rows, std::int64_t scale_columns,
    std::int64_t palettes_offset, std::int64_t symbols_offset,
    std::int64_t scales_offset, bool fp32);

mfq_tensor_backend::Tensor fp8_128_sq_dequant_cuda(
    mfq_tensor_backend::Tensor blob,
    mfq_tensor_backend::Tensor row_q,
    mfq_tensor_backend::Tensor row_symbol_byte_offsets,
    std::int64_t outputs, std::int64_t width,
    std::int64_t scale_kind,
    std::int64_t palettes_offset, std::int64_t symbols_offset,
    std::int64_t scales_offset, bool fp32);

mfq_tensor_backend::Tensor mxfp8_sq_matmul_cuda(
    mfq_tensor_backend::Tensor blob,
    mfq_tensor_backend::Tensor row_q,
    mfq_tensor_backend::Tensor row_symbol_byte_offsets,
    mfq_tensor_backend::Tensor input,
    std::int64_t outputs, std::int64_t width,
    std::int64_t block_rows, std::int64_t block_columns,
    std::int64_t scale_rows, std::int64_t scale_columns,
    std::int64_t palettes_offset, std::int64_t symbols_offset,
    std::int64_t scales_offset);

mfq_tensor_backend::Tensor fp8_128_sq_matmul_cuda(
    mfq_tensor_backend::Tensor blob,
    mfq_tensor_backend::Tensor row_q,
    mfq_tensor_backend::Tensor row_symbol_byte_offsets,
    mfq_tensor_backend::Tensor input,
    std::int64_t outputs, std::int64_t width,
    std::int64_t scale_kind,
    std::int64_t palettes_offset, std::int64_t symbols_offset,
    std::int64_t scales_offset);

mfq_tensor_backend::Tensor fp8_128_sq_swiglu_m5_cuda(
    mfq_tensor_backend::Tensor gate_blob,
    mfq_tensor_backend::Tensor gate_row_q,
    mfq_tensor_backend::Tensor gate_row_symbol_byte_offsets,
    mfq_tensor_backend::Tensor up_blob,
    mfq_tensor_backend::Tensor up_row_q,
    mfq_tensor_backend::Tensor up_row_symbol_byte_offsets,
    mfq_tensor_backend::Tensor input,
    std::int64_t outputs, std::int64_t width,
    std::int64_t scale_kind,
    std::int64_t gate_palettes_offset,
    std::int64_t gate_symbols_offset,
    std::int64_t gate_scales_offset,
    std::int64_t up_palettes_offset,
    std::int64_t up_symbols_offset,
    std::int64_t up_scales_offset);

mfq_tensor_backend::Tensor mxfp8_sq_backward_input_cuda(
    mfq_tensor_backend::Tensor blob,
    mfq_tensor_backend::Tensor row_q,
    mfq_tensor_backend::Tensor row_symbol_byte_offsets,
    mfq_tensor_backend::Tensor output_gradient,
    std::int64_t outputs, std::int64_t width,
    std::int64_t block_rows, std::int64_t block_columns,
    std::int64_t scale_rows, std::int64_t scale_columns,
    std::int64_t palettes_offset, std::int64_t symbols_offset,
    std::int64_t scales_offset);

mfq_tensor_backend::Tensor fp8_128_sq_backward_input_cuda(
    mfq_tensor_backend::Tensor blob,
    mfq_tensor_backend::Tensor row_q,
    mfq_tensor_backend::Tensor row_symbol_byte_offsets,
    mfq_tensor_backend::Tensor output_gradient,
    std::int64_t outputs, std::int64_t width,
    std::int64_t scale_kind,
    std::int64_t palettes_offset, std::int64_t symbols_offset,
    std::int64_t scales_offset);

void mxfp8_sq_moe_matmul_cuda(
    mfq_tensor_backend::Tensor blob,
    mfq_tensor_backend::Tensor row_q,
    mfq_tensor_backend::Tensor row_symbol_byte_offsets,
    mfq_tensor_backend::Tensor input,
    mfq_tensor_backend::Tensor expert_ids,
    mfq_tensor_backend::Tensor expert_local,
    std::int64_t n_experts, std::int64_t local_experts,
    std::int64_t out_per_expert, std::int64_t width,
    std::int64_t block_rows, std::int64_t block_columns,
    std::int64_t scale_rows, std::int64_t scale_columns,
    std::int64_t palettes_offset, std::int64_t symbols_offset,
    std::int64_t scales_offset,
    mfq_tensor_backend::Tensor output);

void fp8_128_sq_moe_matmul_cuda(
    mfq_tensor_backend::Tensor blob,
    mfq_tensor_backend::Tensor row_q,
    mfq_tensor_backend::Tensor row_symbol_byte_offsets,
    mfq_tensor_backend::Tensor input,
    mfq_tensor_backend::Tensor expert_ids,
    mfq_tensor_backend::Tensor expert_local,
    std::int64_t n_experts, std::int64_t local_experts,
    std::int64_t out_per_expert, std::int64_t width,
    std::int64_t scale_kind,
    std::int64_t palettes_offset, std::int64_t symbols_offset,
    std::int64_t scales_offset,
    mfq_tensor_backend::Tensor output);
