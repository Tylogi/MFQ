#pragma once
#include "csrc/backends/cuda/native/tensor_backend.h"
#include "csrc/core/include/mfq/mxfp4_sq_blob.h"

// Parse the self-describing blob on CPU before upload.  These capture-safe
// entry points consume loader-expanded row metadata without GPU-to-CPU reads.
mfq_tensor_backend::Tensor mxfp4_sq_dequant_cuda(
    mfq_tensor_backend::Tensor blob,
    mfq_tensor_backend::Tensor row_q,
    mfq_tensor_backend::Tensor row_symbol_byte_offsets,
    mfq_tensor_backend::Tensor row_auxiliary,
    std::int64_t bits, std::int64_t outputs,
    std::int64_t width, std::int64_t base,
    std::int64_t q_sum, std::int64_t sq4_rows, bool fp32);
mfq_tensor_backend::Tensor mxfp4_sq_matmul_cuda(
    mfq_tensor_backend::Tensor blob,
    mfq_tensor_backend::Tensor row_q,
    mfq_tensor_backend::Tensor row_symbol_byte_offsets,
    mfq_tensor_backend::Tensor row_auxiliary,
    mfq_tensor_backend::Tensor input,
    std::int64_t bits, std::int64_t outputs,
    std::int64_t width, std::int64_t base,
    std::int64_t q_sum, std::int64_t sq4_rows);
void mxfp4_sq_moe_matmul_cuda(
    mfq_tensor_backend::Tensor blob,
    mfq_tensor_backend::Tensor row_q,
    mfq_tensor_backend::Tensor row_symbol_byte_offsets,
    mfq_tensor_backend::Tensor row_auxiliary,
    mfq_tensor_backend::Tensor input,
    mfq_tensor_backend::Tensor expert_ids,
    mfq_tensor_backend::Tensor expert_local,
    std::int64_t bits,
    std::int64_t n_experts,
    std::int64_t local_experts,
    std::int64_t out_per_expert,
    std::int64_t width,
    std::int64_t base,
    std::int64_t q_sum,
    std::int64_t sq4_rows,
    mfq_tensor_backend::Tensor output);
mfq_tensor_backend::Tensor mxfp4_sq_backward_input_cuda(
    mfq_tensor_backend::Tensor blob,
    mfq_tensor_backend::Tensor row_q,
    mfq_tensor_backend::Tensor row_symbol_byte_offsets,
    mfq_tensor_backend::Tensor row_auxiliary,
    mfq_tensor_backend::Tensor output_gradient,
    std::int64_t bits, std::int64_t outputs, std::int64_t width,
    std::int64_t base, std::int64_t q_sum, std::int64_t sq4_rows);
