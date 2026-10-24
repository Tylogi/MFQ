#pragma once

#include "mfq_tensor_backend.h"

#include <cstdint>

namespace mfq::cuda {

// Destination of the fused K/V store.
enum class QkKvStoreMode : int {
    None = 0,   // do not touch a cache; only the dense key/value outputs are written
    Linear = 1, // contiguous [batch, kv_heads, capacity, head_dim] addressed by positions
    Ring = 2,   // same layout, the slot is the position modulo capacity
    Paged = 3,  // chunked physical pages addressed through a page table
};

struct QkKvStore {
    QkKvStoreMode mode = QkKvStoreMode::None;
    // Linear/Ring destination; both caches must use the input storage dtype.
    mfq_tensor_backend::Tensor key_cache;
    mfq_tensor_backend::Tensor value_cache;
    // int64 [tokens] or [batch, tokens]; negative slots are skipped.
    mfq_tensor_backend::Tensor positions;
    // Paged destination: int32 [batch, logical_pages], int64 [chunks] addresses.
    mfq_tensor_backend::Tensor page_table;
    mfq_tensor_backend::Tensor key_chunks;
    mfq_tensor_backend::Tensor value_chunks;
    int64_t page_size = 0;
    int64_t pages_per_chunk = 0;
};

// Axis order of the q/k/v sources. The projection layout is token major; a
// caller that already holds head-major [batch, heads, tokens, head_dim] heads
// passes kHeadMajorAxes instead.
struct QkSourceAxes {
    int batch = 0;
    int tokens = 1;
    int heads = 2;
    int columns = 3;
};

inline constexpr QkSourceAxes kTokenMajorAxes{0, 1, 2, 3};
inline constexpr QkSourceAxes kHeadMajorAxes{0, 2, 1, 3};

// Everything the shared Q/K prepare operator needs; no geometry is implied.
struct QkRopeKvArgs {
    // Sources with the same axis order; strides are read from the tensors.
    mfq_tensor_backend::Tensor query;
    mfq_tensor_backend::Tensor key;
    mfq_tensor_backend::Tensor value;
    QkSourceAxes source_axes = kTokenMajorAxes;
    // f32 [head_dim] RMSNorm weights with an optional additive offset; an
    // undefined weight skips normalization for that tensor.
    mfq_tensor_backend::Tensor query_weight;
    mfq_tensor_backend::Tensor key_weight;
    double eps = 1e-6;
    double weight_offset = 0.0;
    // Rotary positions: [tokens], [batch, tokens], [axes, tokens] or
    // [axes, batch, tokens] in int32/int64.
    mfq_tensor_backend::Tensor positions;
    // f32 [table_length, rotary_dim / 2] rotate-half tables.
    mfq_tensor_backend::Tensor cosine;
    mfq_tensor_backend::Tensor sine;
    // CPU int64 with either no entries or three MRoPE axis lengths.
    mfq_tensor_backend::Tensor sections;
    int64_t rotary_dim = 0;
    QkKvStore store;
    // Also emit the processed K/V rows as contiguous [batch, kv_heads, tokens,
    // head_dim]; required by callers that attend the new tokens directly.
    bool dense_key_value = false;
};

struct QkRopeKvOutput {
    // Contiguous [batch, heads, tokens, head_dim] normalized and rotated query.
    mfq_tensor_backend::Tensor query;
    // Contiguous [batch, kv_heads, tokens, head_dim]; empty unless requested.
    mfq_tensor_backend::Tensor key;
    mfq_tensor_backend::Tensor value;
};

// Fused RMSNorm(weight + offset) -> rotate-half RoPE (full, partial or MRoPE)
// -> K/V cache store for the new tokens of one attention layer.
// Only CUDA contiguous BF16 or FP16 inputs are accepted: BF16 reproduces the
// per-product rounding of the separate BF16 norm/rotary kernels, FP16 keeps
// FP32 intermediates and rounds once when a value is stored.
QkRopeKvOutput qk_norm_rope_kv_cuda(const QkRopeKvArgs& args);

} // namespace mfq::cuda
