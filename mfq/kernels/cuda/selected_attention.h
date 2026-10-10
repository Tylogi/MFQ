#pragma once

#include "mfq_tensor_backend.h"

#include <initializer_list>

namespace mfq_selected_attention {
using mfq_tensor_backend::Tensor;

void values(std::initializer_list<const Tensor*> tensors);
Tensor promoted_matmul(const Tensor& left, const Tensor& right);
Tensor dots(const Tensor& query, const Tensor& keys);
Tensor dense(const Tensor& query, const Tensor& key, const Tensor& value,
    int64_t offset, double scale);
Tensor sparse(const Tensor& query, const Tensor& key, const Tensor& value,
    const Tensor& indices, double scale, bool key_is_cache);
Tensor sparse_gated(const Tensor& query, const Tensor& key, const Tensor& value,
    const Tensor& indices, const Tensor& gate, double scale, bool half_output);
Tensor block_indices_to_tokens(const Tensor& blocks, const Tensor& positions,
    int64_t pool, int64_t budget);
Tensor gated(const Tensor& attended, const Tensor& gate, bool half_output);
Tensor causal_gated(const Tensor& query, const Tensor& key, const Tensor& value,
    const Tensor& positions, const Tensor& gate, int64_t columns, double scale,
    bool half_output, bool fused_mma_gate = true, bool fused_mma_prepare = true);
Tensor mma_reduce_gated(const Tensor& partial_output, const Tensor& metadata,
    const Tensor& gate, bool half_output, int kv_heads, int total_blocks,
    int blocks_per_tile, int query_columns, int head_columns);

} // namespace mfq_selected_attention
