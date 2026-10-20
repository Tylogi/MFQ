#pragma once

#include "native/tensor/view.h"

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace mfq::cuda::kernels {

void clipped_silu_mul(const TensorView& gate, const TensorView& up,
    const TensorView& output, double limit, cudaStream_t stream);

void silu_mul(
    const TensorView& gate,
    const TensorView& up,
    const TensorView& output,
    cudaStream_t stream);

void gelu_mul(
    const TensorView& gate,
    const TensorView& up,
    const TensorView& output,
    cudaStream_t stream);

void linear_gate_beta(
    const TensorView& alpha,
    const TensorView& beta,
    const TensorView& dt_bias,
    const TensorView& a_log,
    const TensorView& gate_t,
    const TensorView& beta_t,
    cudaStream_t stream,
    bool stable_softplus = false);

// Embedding lookups stay flat: the kernels only need the token count, the row
// width and the row pointers, so no TensorView shape is imposed on the token
// IDs.  The caller owns validation, allocation and the launch stream; these
// launchers never consume a CUDA launch error.  Width and group size stay
// int64_t so the grid geometry is computed from the caller's value; the int
// truncation happens only in the kernel arguments, as it did before.
void dense_embedding_lookup(
    const void* weight,
    ScalarType weight_type,
    const std::int64_t* token_ids,
    void* output,
    int token_count,
    int width,
    int vocabulary,
    cudaStream_t stream);

void nint_embedding_lookup(
    const std::uint8_t* bitstream,
    const std::uint8_t* row_q_bits,
    const std::int64_t* row_q_bit_offsets,
    const std::uint8_t* subgroup_scale,
    const std::uint8_t* subgroup_minimum,
    const float* neuron_scale,
    const float* neuron_minimum,
    const std::int64_t* token_ids,
    __half* output,
    int token_count,
    int vocabulary,
    int groups,
    std::int64_t group_size,
    std::int64_t width,
    cudaStream_t stream);

// inverse is optional: null selects descriptor row index/width for output row.
void nint_selected_rows(
    const std::uint8_t* packed,
    const std::uint32_t* descriptors,
    const std::int64_t* inverse,
    __half* output,
    std::size_t total,
    int width,
    cudaStream_t stream);

// quantized is the NINT8-0 uint8 storage reinterpreted as int8, scale is Half.
void nint8_zero_embedding_lookup(
    const std::int8_t* quantized,
    const __half* scale,
    const std::int64_t* token_ids,
    __half* output,
    int token_count,
    int vocabulary,
    int groups,
    std::int64_t width,
    cudaStream_t stream);

// Callers validate cache/input views and check launch errors. Extents retain
// their original host-side int casts; launchers do not consume CUDA errors.
void kv_cache_write(
    void* k_cache,
    void* v_cache,
    const void* k,
    const void* v,
    const std::int64_t* positions,
    ScalarType dtype,
    int B,
    int H,
    int T,
    int D,
    int max_seq,
    int pos_dim,
    cudaStream_t stream);

void kv_cache_write_ring(
    void* k_cache,
    void* v_cache,
    const void* k,
    const void* v,
    ScalarType dtype,
    std::int64_t position_start,
    int B,
    int H,
    int T,
    int D,
    int capacity,
    cudaStream_t stream);

void kv_cache_write_ring_positions(
    void* k_cache,
    void* v_cache,
    const void* k,
    const void* v,
    const std::int64_t* positions,
    ScalarType dtype,
    int B,
    int H,
    int T,
    int D,
    int capacity,
    cudaStream_t stream);

}  // namespace mfq::cuda::kernels
