// Fused Q/K RMSNorm(+offset), rotate-half RoPE (full, partial or MRoPE) and
// K/V cache store for the new tokens of one attention layer.
//
// The operator is shared by every FullBlock user: head counts, head width,
// rotary width, position axes, source strides and the cache destination
// (linear, ring or paged) are parameters instead of a fixed geometry.
//
// Rounding follows the storage dtype so the fused span keeps the numeric
// contract of the separate stages:
//   * BF16 reproduces the Qwen BF16 kernels, which round the normalized value,
//     the norm scale, the rotary tables and both rotary products to BF16.
//   * FP16 keeps FP32 intermediates (paired FP16 RMSNorm plus table rotary) and
//     rounds once when a value is stored.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "mfq_cuda_qk_rope_kv.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <type_traits>

namespace {

constexpr int kQkMaxThreads = 256;

// Strides of one [batch, tokens, heads, head_dim] source view.
struct QkRowLayout {
    int64_t batch = 0;
    int64_t token = 0;
    int64_t head = 0;
    int64_t column = 0;
};

// K/V store destination; mode matches mfq::cuda::QkKvStoreMode.
struct QkCachePlan {
    int mode = 0;
    void* key = nullptr;
    void* value = nullptr;
    const int64_t* positions = nullptr;
    int position_rows = 1;
    int64_t capacity = 0;
    const int32_t* page_table = nullptr;
    const int64_t* key_chunks = nullptr;
    const int64_t* value_chunks = nullptr;
    int logical_pages = 0;
    int page_size = 0;
    int pages_per_chunk = 0;
    int chunks = 0;
};

struct QkLayerPlan {
    const void* query = nullptr;
    const void* key = nullptr;
    const void* value = nullptr;
    const float* query_weight = nullptr;
    const float* key_weight = nullptr;
    void* query_target = nullptr;
    void* key_target = nullptr;
    void* value_target = nullptr;
    const void* positions = nullptr;
    const float* cosine = nullptr;
    const float* sine = nullptr;
    float eps = 0.0f;
    float weight_offset = 0.0f;
    int width = 0;
    int pairs = 0;
    int table_length = 0;
    int heads = 0;
    int query_heads = 0;
    int kv_heads = 0;
    int tokens = 0;
    int position_axes = 1;
    int position_batches = 1;
    int section0 = 0;
    int section1 = 0;
    int section2 = 0;
    int threads = 32;
    bool has_sections = false;
    bool wide_positions = false;
    QkRowLayout query_layout;
    QkRowLayout key_layout;
    QkRowLayout value_layout;
    QkCachePlan store;
};

template <class Value>
__device__ __forceinline__ float qk_to_float(Value value) {
    if constexpr (std::is_same_v<Value, __nv_bfloat16>) {
        return __bfloat162float(value);
    } else {
        return __half2float(value);
    }
}

template <class Value>
__device__ __forceinline__ Value qk_from_float(float value) {
    if constexpr (std::is_same_v<Value, __nv_bfloat16>) {
        return __float2bfloat16_rn(value);
    } else {
        return __float2half_rn(value);
    }
}

// BF16 stores and reloads every intermediate, FP16 keeps FP32.
template <class Value>
__device__ __forceinline__ float qk_round(float value) {
    if constexpr (std::is_same_v<Value, __nv_bfloat16>) {
        return __bfloat162float(__float2bfloat16_rn(value));
    } else {
        return value;
    }
}

// RMS inverse scale over one row. The caller passes the shared staging area so
// a block can normalize the Q row and the K row in sequence.
template <class Value>
__device__ __forceinline__ float qk_row_inverse(const Value* source, const QkRowLayout& layout, int width,
                                                float eps, float* partial) {
    float sum = 0.0f;
    for (int column = threadIdx.x; column < width; column += int(blockDim.x)) {
        const float value = qk_to_float(source[int64_t(column) * layout.column]);
        sum = __fadd_rn(sum, __fmul_rn(value, value));
    }
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = int(blockDim.x) / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) partial[threadIdx.x] = __fadd_rn(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    const float mean = __fdiv_rn(partial[0], float(width));
    const float inverse = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(mean, eps)));
    __syncthreads();
    return inverse;
}

template <class Value>
__device__ __forceinline__ float qk_normalized(const Value* source, const QkRowLayout& layout, int column,
                                               float inverse, const float* weight, float weight_offset) {
    const float value = qk_to_float(source[int64_t(column) * layout.column]);
    if (weight == nullptr) return value;
    const float normalized = qk_round<Value>(__fmul_rn(value, inverse));
    const float scale = weight_offset != 0.0f
                            ? qk_round<Value>(__fadd_rn(qk_round<Value>(weight[column]), weight_offset))
                            : qk_round<Value>(weight[column]);
    return qk_round<Value>(__fmul_rn(scale, normalized));
}

// Table position for one rotary pair; axes follow the MRoPE sections.
__device__ __forceinline__ int qk_rotary_position(const QkLayerPlan& plan, int column, int batch, int token) {
    int axis = 0;
    if (plan.has_sections) {
        axis = column < plan.section0 ? 0 : (column < plan.section0 + plan.section1 ? 1 : 2);
        if (axis >= plan.position_axes) axis = 0;
    }
    const int64_t batch_index = plan.position_batches == 1 ? 0 : batch;
    const int64_t index = (int64_t(axis) * plan.position_batches + batch_index) * plan.tokens + token;
    const int64_t raw = plan.wide_positions
                            ? int64_t(static_cast<const int64_t*>(plan.positions)[index])
                            : int64_t(static_cast<const int32_t*>(plan.positions)[index]);
    const int position = int(raw);
    return position < 0 ? 0 : (position >= plan.table_length ? plan.table_length - 1 : position);
}

__device__ __forceinline__ int64_t qk_store_position(const QkCachePlan& store, int tokens, int batch, int token) {
    const int64_t index = int64_t(store.position_rows == 1 ? 0 : batch) * tokens + token;
    return store.positions[index];
}

// Row base of the K or V cache slot, or nullptr when the position is outside
// the cache (matching the separate store, which skips such slots).
template <class Value>
__device__ __forceinline__ Value* qk_cache_row(const QkLayerPlan& plan, bool key, int batch, int head, int token) {
    if (plan.store.mode == 0) return nullptr;
    const int64_t position = qk_store_position(plan.store, plan.tokens, batch, token);
    if (position < 0) return nullptr;
    if (plan.store.mode == 3) {
        const int logical_page = int(position / plan.store.page_size);
        if (logical_page >= plan.store.logical_pages) return nullptr;
        const int physical_page = plan.store.page_table[int64_t(batch) * plan.store.logical_pages + logical_page];
        if (physical_page < 0) return nullptr;
        const int chunk = physical_page / plan.store.pages_per_chunk;
        if (chunk >= plan.store.chunks) return nullptr;
        const int64_t* chunk_table = key ? plan.store.key_chunks : plan.store.value_chunks;
        if (chunk_table[chunk] == 0) return nullptr;
        const int64_t page_elements = int64_t(plan.kv_heads) * plan.store.page_size * plan.width;
        auto* page = reinterpret_cast<Value*>(static_cast<uintptr_t>(chunk_table[chunk])) +
                     int64_t(physical_page - chunk * plan.store.pages_per_chunk) * page_elements;
        return page + int64_t(head * plan.store.page_size + position % plan.store.page_size) * plan.width;
    }
    const int64_t slot = plan.store.mode == 2 ? position % plan.store.capacity : position;
    if (slot >= plan.store.capacity) return nullptr;
    auto* base = static_cast<Value*>(key ? plan.store.key : plan.store.value);
    return base + ((int64_t(batch) * plan.kv_heads + head) * plan.store.capacity + slot) * plan.width;
}

// Normalize (when a weight is present) and rotate one row, then write it to the
// contiguous target row and/or the cache slot. `target` uses the head-major
// [batch, heads, tokens, head_dim] layout the attention kernels consume.
template <class Value>
__device__ __forceinline__ void qk_norm_rope_row(const QkLayerPlan& plan, const void* source,
                                                 const QkRowLayout& layout, const float* weight, int batch,
                                                 int head, int token, float* partial, Value* target,
                                                 Value* cache) {
    const auto* input = static_cast<const Value*>(source) +
                        int64_t(batch) * layout.batch + int64_t(token) * layout.token + int64_t(head) * layout.head;
    const float inverse =
        weight != nullptr ? qk_row_inverse<Value>(input, layout, plan.width, plan.eps, partial) : 0.0f;
    const auto store = [&](int column, float value) {
        const Value rounded = qk_from_float<Value>(value);
        if (target != nullptr) target[column] = rounded;
        if (cache != nullptr) cache[column] = rounded;
    };
    for (int index = threadIdx.x; index < plan.width - plan.pairs; index += int(blockDim.x)) {
        if (index >= plan.pairs) {
            const int column = index + plan.pairs;
            store(column, qk_normalized<Value>(input, layout, column, inverse, weight, plan.weight_offset));
            continue;
        }
        const int table = qk_rotary_position(plan, index, batch, token);
        const float cosine = qk_round<Value>(plan.cosine[int64_t(table) * plan.pairs + index]);
        const float sine = qk_round<Value>(plan.sine[int64_t(table) * plan.pairs + index]);
        const float first = qk_normalized<Value>(input, layout, index, inverse, weight, plan.weight_offset);
        const float second =
            qk_normalized<Value>(input, layout, index + plan.pairs, inverse, weight, plan.weight_offset);
        const float rotated_first =
            qk_round<Value>(__fsub_rn(qk_round<Value>(__fmul_rn(first, cosine)),
                                      qk_round<Value>(__fmul_rn(second, sine))));
        const float rotated_second =
            qk_round<Value>(__fadd_rn(qk_round<Value>(__fmul_rn(second, cosine)),
                                      qk_round<Value>(__fmul_rn(first, sine))));
        store(index, rotated_first);
        store(index + plan.pairs, rotated_second);
    }
}

// Plain (optionally narrowed) row copy used for the value projection.
template <class Value>
__device__ __forceinline__ void qk_copy_row(const QkLayerPlan& plan, const void* source, const QkRowLayout& layout,
                                            int batch, int head, int token, Value* target, Value* cache) {
    const auto* input = static_cast<const Value*>(source) +
                        int64_t(batch) * layout.batch + int64_t(token) * layout.token + int64_t(head) * layout.head;
    for (int column = threadIdx.x; column < plan.width; column += int(blockDim.x)) {
        const Value value = qk_from_float<Value>(qk_to_float(input[int64_t(column) * layout.column]));
        if (target != nullptr) target[column] = value;
        if (cache != nullptr) cache[column] = value;
    }
}

template <class Value>
__global__ void qk_norm_rope_kv_kernel(QkLayerPlan plan) {
    __shared__ float partial[kQkMaxThreads];
    const int token = int(blockIdx.x) % plan.tokens;
    const int head = (int(blockIdx.x) / plan.tokens) % plan.heads;
    const int batch = int(blockIdx.x) / (plan.tokens * plan.heads);
    if (head < plan.query_heads) {
        auto* target = static_cast<Value*>(plan.query_target) +
                       ((int64_t(batch) * plan.query_heads + head) * plan.tokens + token) * plan.width;
        qk_norm_rope_row<Value>(plan, plan.query, plan.query_layout, plan.query_weight, batch, head, token, partial,
                                target, nullptr);
    }
    // One block also serves the matching K/V head, which keeps the fused span a
    // single launch for every query-head to KV-head ratio.
    if (head < plan.kv_heads) {
        auto* key_target =
            plan.key_target != nullptr
                ? static_cast<Value*>(plan.key_target) +
                      ((int64_t(batch) * plan.kv_heads + head) * plan.tokens + token) * plan.width
                : nullptr;
        qk_norm_rope_row<Value>(plan, plan.key, plan.key_layout, plan.key_weight, batch, head, token, partial,
                                key_target, qk_cache_row<Value>(plan, true, batch, head, token));
        auto* value_target =
            plan.value_target != nullptr
                ? static_cast<Value*>(plan.value_target) +
                      ((int64_t(batch) * plan.kv_heads + head) * plan.tokens + token) * plan.width
                : nullptr;
        qk_copy_row<Value>(plan, plan.value, plan.value_layout, batch, head, token, value_target,
                           qk_cache_row<Value>(plan, false, batch, head, token));
    }
}

QkRowLayout row_layout(const mfq_tensor_backend::Tensor& value, const mfq::cuda::QkSourceAxes& axes) {
    return QkRowLayout{value.stride(axes.batch), value.stride(axes.tokens), value.stride(axes.heads),
                       value.stride(axes.columns)};
}

int launch_threads(int64_t width) {
    return width <= 64 ? 64 : (width <= 128 ? 128 : kQkMaxThreads);
}

// Sections are a CPU int64 vector, so the axis lengths travel by value. An
// empty tensor means "no MRoPE" and must not be dereferenced.
std::array<int, 3> section_lengths(const mfq_tensor_backend::Tensor& sections) {
    auto lengths = std::array<int, 3>{0, 0, 0};
    if (!sections.defined() || sections.numel() != 3) return lengths;
    const int64_t* entries = sections.data_ptr<int64_t>();
    for (int axis = 0; axis < 3; ++axis) lengths[axis] = int(entries[axis]);
    return lengths;
}

void fill_cache_plan(const mfq::cuda::QkKvStore& store, QkCachePlan& plan) {
    plan.mode = int(store.mode);
    if (plan.mode == 0) return;
    namespace tb = mfq_tensor_backend;
    MFQ_RUNTIME_CHECK(store.positions.is_cuda() && store.positions.is_contiguous() &&
                          store.positions.scalar_type() == tb::kInt64 &&
                          (store.positions.dim() == 1 || store.positions.dim() == 2),
                      "fused QK store requires contiguous CUDA int64 positions");
    plan.positions = store.positions.data_ptr<int64_t>();
    plan.position_rows = int(store.positions.dim() == 1 ? 1 : store.positions.size(0));
    if (plan.mode == 3) {
        plan.page_table = store.page_table.data_ptr<int32_t>();
        plan.key_chunks = store.key_chunks.data_ptr<int64_t>();
        plan.value_chunks = store.value_chunks.data_ptr<int64_t>();
        plan.logical_pages = int(store.page_table.size(1));
        plan.chunks = int(store.key_chunks.numel());
        plan.page_size = int(store.page_size);
        plan.pages_per_chunk = int(store.pages_per_chunk);
        return;
    }
    plan.key = store.key_cache.data_ptr();
    plan.value = store.value_cache.data_ptr();
    plan.capacity = store.key_cache.size(2);
}

} // namespace

namespace mfq::cuda {
namespace tb = mfq_tensor_backend;

QkRopeKvOutput qk_norm_rope_kv_cuda(const QkRopeKvArgs& args) {
    MFQ_RUNTIME_CHECK(args.query.is_cuda() && args.key.is_cuda() && args.value.is_cuda(),
                      "fused QK rotary requires CUDA q/k/v");
    MFQ_RUNTIME_CHECK(args.query.dim() == 4 && args.key.dim() == 4 && args.value.dim() == 4,
                      "fused QK rotary requires four-dimensional q/k/v");
    const auto& axes = args.source_axes;
    MFQ_RUNTIME_CHECK(axes.batch >= 0 && axes.batch < 4 && axes.tokens >= 0 && axes.tokens < 4 &&
                          axes.heads >= 0 && axes.heads < 4 && axes.columns >= 0 && axes.columns < 4 &&
                          axes.batch != axes.tokens && axes.batch != axes.heads &&
                          axes.batch != axes.columns && axes.tokens != axes.heads &&
                          axes.tokens != axes.columns && axes.heads != axes.columns,
                      "fused QK rotary source axes must be a permutation");
    MFQ_RUNTIME_CHECK(args.query.scalar_type() == tb::kBFloat16 || args.query.scalar_type() == tb::kFloat16,
                      "fused QK rotary requires BF16 or FP16 q/k/v");
    MFQ_RUNTIME_CHECK(args.key.scalar_type() == args.query.scalar_type() &&
                          args.value.scalar_type() == args.query.scalar_type(),
                      "fused QK rotary requires one q/k/v storage dtype");
    const auto batches = args.query.size(axes.batch), tokens = args.query.size(axes.tokens),
               heads = args.query.size(axes.heads), width = args.query.size(axes.columns);
    const auto kv_heads = args.key.size(axes.heads);
    MFQ_RUNTIME_CHECK(batches > 0 && tokens > 0 && heads > 0 && kv_heads > 0 && width > 0 &&
                          batches <= INT_MAX && tokens <= INT_MAX && heads <= INT_MAX && kv_heads <= INT_MAX &&
                          width <= INT_MAX,
                      "fused QK rotary geometry is invalid");
    MFQ_RUNTIME_CHECK(args.key.size(axes.batch) == batches && args.key.size(axes.tokens) == tokens &&
                          args.key.size(axes.columns) == width && args.value.sizes() == args.key.sizes(),
                      "fused QK rotary key/value geometry must match the query");
    MFQ_RUNTIME_CHECK(args.rotary_dim > 0 && args.rotary_dim <= width && args.rotary_dim % 2 == 0,
                      "fused QK rotary width is invalid");
    MFQ_RUNTIME_CHECK(args.cosine.is_cuda() && args.sine.is_cuda() && args.cosine.is_contiguous() &&
                          args.sine.is_contiguous() && args.cosine.scalar_type() == tb::kFloat32 &&
                          args.sine.scalar_type() == tb::kFloat32 && args.cosine.sizes() == args.sine.sizes() &&
                          args.cosine.dim() == 2 && args.cosine.size(0) > 0 && args.cosine.size(0) <= INT_MAX &&
                          args.cosine.size(1) == args.rotary_dim / 2,
                      "fused QK rotary requires f32 [table, rotary_dim/2] cos/sin tables");
    MFQ_RUNTIME_CHECK(args.cosine.device() == args.query.device() && args.sine.device() == args.query.device(),
                      "fused QK rotary tables must share the input device");
    const auto weight = [&](const tb::Tensor& value, const char* name) -> const float* {
        if (!value.defined()) return nullptr;
        MFQ_RUNTIME_CHECK(value.is_cuda() && value.is_contiguous() && value.scalar_type() == tb::kFloat32 &&
                              value.dim() == 1 && value.numel() == width,
                          name, " must be a CUDA contiguous f32[head_dim] vector");
        return value.data_ptr<float>();
    };
    const float* query_weight = weight(args.query_weight, "fused QK rotary query norm weight");
    const float* key_weight = weight(args.key_weight, "fused QK rotary key norm weight");
    MFQ_RUNTIME_CHECK(args.positions.is_cuda() && args.positions.is_contiguous() &&
                          (args.positions.scalar_type() == tb::kInt32 ||
                           args.positions.scalar_type() == tb::kInt64) &&
                          args.positions.dim() >= 1 && args.positions.dim() <= 3 &&
                          args.positions.size(-1) == tokens,
                      "fused QK rotary positions are invalid");
    const bool has_sections = args.sections.defined() && args.sections.numel() == 3;
    MFQ_RUNTIME_CHECK(!args.sections.defined() || args.sections.numel() == 0 || has_sections,
                      "fused QK rotary sections must be empty or three MRoPE lengths");
    MFQ_RUNTIME_CHECK(!args.sections.defined() || (!args.sections.is_cuda() &&
                                                  args.sections.scalar_type() == tb::kInt64),
                      "fused QK rotary sections must be CPU int64");
    // MRoPE always carries three axis positions, one per section length.
    MFQ_RUNTIME_CHECK(!has_sections || (args.positions.dim() >= 2 && args.positions.size(0) == 3),
                      "fused QK rotary MRoPE requires three position axes");
    const int position_axes = args.positions.dim() == 1 ? 1 : (has_sections || args.positions.dim() == 3
                                                                  ? int(args.positions.size(0))
                                                                  : 1);
    const int position_batches = args.positions.dim() == 1
                                     ? 1
                                     : (args.positions.dim() == 3 ? int(args.positions.size(1))
                                                                  : (has_sections ? 1 : int(args.positions.size(0))));
    MFQ_RUNTIME_CHECK(position_axes > 0 && position_batches > 0 &&
                          (position_batches == 1 || position_batches == int(batches)),
                      "fused QK rotary positions do not cover the batch");
    const auto sections = section_lengths(args.sections);
    MFQ_RUNTIME_CHECK(!has_sections || sections[0] + sections[1] + sections[2] == int(args.rotary_dim / 2),
                      "fused QK rotary sections disagree with the rotary width");

    QkLayerPlan plan;
    plan.query = args.query.data_ptr();
    plan.key = args.key.data_ptr();
    plan.value = args.value.data_ptr();
    plan.query_weight = query_weight;
    plan.key_weight = key_weight;
    plan.positions = args.positions.data_ptr();
    plan.wide_positions = args.positions.scalar_type() == tb::kInt64;
    plan.cosine = args.cosine.data_ptr<float>();
    plan.sine = args.sine.data_ptr<float>();
    plan.eps = float(args.eps);
    plan.weight_offset = float(args.weight_offset);
    plan.width = int(width);
    plan.pairs = int(args.rotary_dim / 2);
    plan.table_length = int(args.cosine.size(0));
    plan.heads = int(std::max(heads, kv_heads));
    plan.query_heads = int(heads);
    plan.kv_heads = int(kv_heads);
    plan.tokens = int(tokens);
    plan.position_axes = position_axes;
    plan.position_batches = position_batches;
    plan.section0 = sections[0];
    plan.section1 = sections[1];
    plan.section2 = sections[2];
    plan.has_sections = has_sections;
    plan.threads = launch_threads(width);
    plan.query_layout = row_layout(args.query, axes);
    plan.key_layout = row_layout(args.key, axes);
    plan.value_layout = row_layout(args.value, axes);

    const auto store_mode = args.store.mode;
    MFQ_RUNTIME_CHECK(store_mode == QkKvStoreMode::None || args.store.positions.defined(),
                      "fused QK store requires cache positions");
    if (store_mode == QkKvStoreMode::Linear || store_mode == QkKvStoreMode::Ring) {
        const auto& key_cache = args.store.key_cache;
        const auto& value_cache = args.store.value_cache;
        MFQ_RUNTIME_CHECK(key_cache.is_cuda() && key_cache.is_contiguous() && value_cache.is_cuda() &&
                              value_cache.is_contiguous() && key_cache.sizes() == value_cache.sizes() &&
                              key_cache.dim() == 4 && key_cache.size(0) == batches &&
                              key_cache.size(1) == kv_heads && key_cache.size(2) > 0 &&
                              key_cache.size(2) <= INT_MAX && key_cache.size(3) == width &&
                              key_cache.scalar_type() == args.query.scalar_type(),
                          "fused QK store requires matching contiguous [batch, kv_heads, capacity, head_dim] caches");
        MFQ_RUNTIME_CHECK(args.store.positions.scalar_type() == tb::kInt64,
                          "fused QK store requires int64 cache positions");
    }
    if (store_mode == QkKvStoreMode::Paged) {
        MFQ_RUNTIME_CHECK(args.store.page_table.is_cuda() && args.store.page_table.is_contiguous() &&
                              args.store.page_table.scalar_type() == tb::kInt32 && args.store.page_table.dim() == 2 &&
                              args.store.page_table.size(0) >= batches && args.store.page_table.size(1) > 0 &&
                              args.store.page_table.size(1) <= INT_MAX,
                          "fused QK paged store requires a contiguous int32 page table");
        MFQ_RUNTIME_CHECK(args.store.key_chunks.is_cuda() && args.store.key_chunks.is_contiguous() &&
                              args.store.key_chunks.scalar_type() == tb::kInt64 && args.store.key_chunks.dim() == 1 &&
                              args.store.key_chunks.numel() > 0 && args.store.key_chunks.numel() <= INT_MAX &&
                              args.store.value_chunks.is_cuda() && args.store.value_chunks.is_contiguous() &&
                              args.store.value_chunks.scalar_type() == tb::kInt64 &&
                              args.store.value_chunks.sizes() == args.store.key_chunks.sizes(),
                          "fused QK paged store requires contiguous int64 chunk tables");
        MFQ_RUNTIME_CHECK(args.store.page_size > 0 && args.store.page_size <= INT_MAX &&
                              args.store.pages_per_chunk > 0 && args.store.pages_per_chunk <= INT_MAX,
                          "fused QK paged store geometry is invalid");
    }
    fill_cache_plan(args.store, plan.store);

    MfqCudaGuard guard(args.query.device());
    auto options = args.query.options();
    QkRopeKvOutput output;
    output.query = tb::empty({batches, heads, tokens, width}, options);
    if (args.dense_key_value) {
        output.key = tb::empty({batches, kv_heads, tokens, width}, args.key.options());
        output.value = tb::empty({batches, kv_heads, tokens, width}, args.value.options());
        plan.key_target = output.key.data_ptr();
        plan.value_target = output.value.data_ptr();
    }
    plan.query_target = output.query.data_ptr();
    const auto blocks = unsigned(batches * plan.heads * tokens);
    if (args.query.scalar_type() == tb::kBFloat16) {
        qk_norm_rope_kv_kernel<__nv_bfloat16><<<blocks, plan.threads, 0, mfq_current_cuda_stream()>>>(plan);
    } else {
        qk_norm_rope_kv_kernel<__half><<<blocks, plan.threads, 0, mfq_current_cuda_stream()>>>(plan);
    }
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

} // namespace mfq::cuda
