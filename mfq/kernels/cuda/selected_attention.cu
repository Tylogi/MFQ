// Shared dense and index-selected attention mechanics.
#include "selected_attention.h"
#include "mfq_sparse_attn_mma_f16.cuh"

#include <cmath>
#include <limits>

namespace mfq_selected_attention {
namespace tb = mfq_tensor_backend;

void values(std::initializer_list<const Tensor*> tensors) {
    const auto* first = *tensors.begin();
    for (const auto* value : tensors) {
        MFQ_RUNTIME_CHECK(value->defined() && value->is_cuda(),
            "selected attention requires CUDA tensors");
        MFQ_RUNTIME_CHECK(value->device() == first->device(),
            "selected-attention tensors must share one CUDA device");
        const auto dtype = value->scalar_type();
        MFQ_RUNTIME_CHECK(dtype == tb::kFloat32 || dtype == tb::kFloat16 ||
            dtype == tb::kBFloat16, "selected attention requires F32/F16/BF16 values");
    }
}

Tensor promoted_matmul(const Tensor& left, const Tensor& right) {
    // MLX matmul promotes mixed floating operands, unlike ATen matmul.
    const auto dtype = left.scalar_type() == right.scalar_type()
        ? left.scalar_type() : tb::kFloat32;
    return tb::matmul(left.to(dtype), right.to(dtype));
}

Tensor dots(const Tensor& query, const Tensor& pooled) {
    values({&query, &pooled});
    MFQ_RUNTIME_CHECK(query.dim() == 4 && pooled.dim() == 3,
        "selected-attention scores require [B,T,H,D] query and [B,P,D] keys");
    MFQ_RUNTIME_CHECK(query.size(0) == pooled.size(0) &&
        query.size(3) == pooled.size(2) && query.size(2) > 0 && query.size(3) > 0,
        "selected-attention score dimensions disagree");
    // QSA rotates [B,H,T,D] then permutes back to [B,T,H,D]. Native
    // batched GEMM requires the leading batch axes to have compact strides.
    return tb::matmul(query.to(tb::kFloat32).contiguous(),
        pooled.to(tb::kFloat32).transpose(-1, -2).unsqueeze(1)).clamp_min(0.0);
}

void attention_shapes(const Tensor& query, const Tensor& key, const Tensor& value) {
    values({&query, &key, &value});
    MFQ_RUNTIME_CHECK(query.dim() == 4 && key.dim() == 4 && value.sizes() == key.sizes(),
        "selected attention requires [B,H,T,D] query/key/value");
    MFQ_RUNTIME_CHECK(query.size(0) == key.size(0) && key.size(1) > 0 &&
        query.size(1) > 0 && query.size(1) % key.size(1) == 0 &&
        query.size(3) == key.size(3) && query.size(3) > 0,
        "selected-attention dimensions disagree");
}

__global__ void prepare_sparse_indices_kernel(
    const int32_t * __restrict__ source,
    int32_t * __restrict__ indices,
    half * __restrict__ mask,
    int64_t rows,
    int source_count,
    int padded_count,
    int cache_length) {
    const int64_t total = rows * padded_count;
    for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int slot = static_cast<int>(linear % padded_count);
        const int64_t row = linear / padded_count;
        const int index = slot < source_count
            ? source[row * source_count + slot]
            : -1;
        const bool valid = index >= 0 && index < cache_length;
        indices[linear] = valid ? index : 0;
        mask[linear] = valid ? __float2half(0.0f) : __float2half(-INFINITY);
    }
}

// Each CTA retains the Metal kernel's eight warp partials and online-softmax
// recurrence. The generic wide-D path stores accumulators in its output only;
// it imposes no extra head-width limit. Published D=256/512 stay in registers.
template<int Slices>
__global__ void sparse_attention_kernel(
    const float* query, const half* key, const half* value, const int32_t* indices,
    float* output, int64_t heads, int64_t kv_heads, int64_t tokens,
    int64_t width, int64_t cache_length, int64_t topk, float scale) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int64_t head = blockIdx.x % heads;
    const int64_t row = blockIdx.x / heads;
    const int64_t token = row % tokens;
    const int64_t batch = row / tokens;
    const int64_t kv_head = head / (heads / kv_heads);
    const int64_t qb = ((batch * heads + head) * tokens + token) * width;
    const int64_t ob = ((batch * tokens + token) * heads + head) * width;
    __shared__ float partials[8];
    __shared__ float online[4];
    float accum[Slices > 0 ? Slices : 1] = {};
    if constexpr (Slices == 0) {
        for (int64_t d = tid; d < width; d += 256) output[ob + d] = 0.f;
    }
    if (tid == 0) {
        online[0] = -INFINITY;
        online[1] = online[2] = online[3] = 0.f;
    }
    __syncthreads();
    for (int64_t selected = 0; selected < topk; ++selected) {
        const int64_t index = indices[row * topk + selected];
        const bool valid = index >= 0 && index < cache_length;
        const int64_t cb = ((batch * kv_heads + kv_head) * cache_length +
            (valid ? index : 0)) * width;
        float dot = 0.f;
        if (valid) {
            for (int64_t d = tid; d < width; d += 256) {
                dot += query[qb + d] * __half2float(key[cb + d]);
            }
        }
        for (int offset = 16; offset > 0; offset >>= 1)
            dot += __shfl_down_sync(0xffffffff, dot, offset);
        if (lane == 0) partials[warp] = dot;
        __syncthreads();
        if (tid == 0) {
            float score = 0.f;
            for (int group = 0; group < 8; ++group) score += partials[group];
            score = valid ? score * scale : -INFINITY;
            const float old_max = online[0];
            const float next_max = fmaxf(old_max, score);
            const float old_scale = isfinite(old_max) ? expf(old_max - next_max) : 0.f;
            const float next_scale = valid ? expf(score - next_max) : 0.f;
            online[0] = next_max;
            online[1] = online[1] * old_scale + next_scale;
            online[2] = old_scale;
            online[3] = next_scale;
        }
        __syncthreads();
        if constexpr (Slices > 0) {
            #pragma unroll
            for (int slice = 0; slice < Slices; ++slice) {
                const int64_t d = tid + slice * 256;
                const float v = valid && d < width ? __half2float(value[cb + d]) : 0.f;
                accum[slice] = accum[slice] * online[2] + v * online[3];
            }
        } else {
            for (int64_t d = tid; d < width; d += 256) {
                const float v = valid ? __half2float(value[cb + d]) : 0.f;
                output[ob + d] = output[ob + d] * online[2] + v * online[3];
            }
        }
        __syncthreads();
    }
    const float inverse = online[1] > 0.f ? 1.f / online[1] : 0.f;
    if constexpr (Slices > 0) {
        #pragma unroll
        for (int slice = 0; slice < Slices; ++slice) {
            const int64_t d = tid + slice * 256;
            if (d < width) output[ob + d] = accum[slice] * inverse;
        }
    } else {
        for (int64_t d = tid; d < width; d += 256) output[ob + d] *= inverse;
    }
}

Tensor sparse(const Tensor& q, const Tensor& k, const Tensor& v,
                        const Tensor& selected, double scale,
                        bool v_is_k_view = false) {
    attention_shapes(q, k, v);
    MFQ_RUNTIME_CHECK(selected.defined() && selected.is_cuda() &&
        selected.device() == q.device() && selected.dim() == 3 &&
        selected.size(0) == q.size(0) && selected.size(1) == q.size(2) &&
        selected.size(2) > 0, "sparse indices must have [B,T,topk] shape");
    MFQ_RUNTIME_CHECK(selected.scalar_type() == tb::kInt32 ||
        selected.scalar_type() == tb::kInt64, "sparse indices must be integer");
    MFQ_RUNTIME_CHECK(std::isfinite(scale), "attention scale must be finite");
    MfqCudaGuard guard(q.device());
    auto query = q.to(tb::kFloat32).contiguous();
    auto key = k.to(tb::kFloat16).contiguous();
    auto value = v_is_k_view ? key : v.to(tb::kFloat16).contiguous();
    auto indices = selected.to(tb::kInt32).contiguous();
    if (q.size(0) == 0 || q.size(2) == 0) {
        return tb::empty({q.size(0), q.size(2), q.size(1), q.size(3)},
            q.options().dtype(tb::kFloat32));
    }
    if (key.size(2) == 0) {
        return tb::zeros({q.size(0), q.size(2), q.size(1), q.size(3)},
            q.options().dtype(tb::kFloat32));
    }

    const bool use_qsa_mma = !v_is_k_view && q.size(3) == 256;
    const bool use_mla_mma = v_is_k_view && q.size(3) == 512;
    if (use_qsa_mma || use_mla_mma) {
        constexpr int index_tile = 32;
        const int64_t source_count = indices.size(2);
        MFQ_RUNTIME_CHECK(
            source_count <= std::numeric_limits<int>::max() - index_tile &&
            key.size(2) <= std::numeric_limits<int>::max(),
            "sparse attention exceeds 32-bit index geometry");
        const int64_t padded_count =
            ((source_count + index_tile - 1) / index_tile) * index_tile;
        auto shape = indices.sizes().vec();
        shape[2] = padded_count;
        auto safe_indices = tb::empty(shape, indices.options());
        auto mask = tb::empty(shape, query.options().dtype(tb::kFloat16));
        MFQ_RUNTIME_CHECK(
            indices.size(0) <= std::numeric_limits<int64_t>::max() /
                indices.size(1),
            "sparse attention row count is too large");
        const int64_t rows = indices.size(0) * indices.size(1);
        MFQ_RUNTIME_CHECK(
            rows <= std::numeric_limits<int64_t>::max() / padded_count,
            "sparse attention index workspace is too large");
        const int64_t total = rows * padded_count;
        const int blocks = static_cast<int>(std::min<int64_t>(
            65535, std::max<int64_t>(1, (total + 255) / 256)));
        prepare_sparse_indices_kernel<<<blocks, 256, 0, mfq_current_cuda_stream()>>>(
            indices.data_ptr<int32_t>(), safe_indices.data_ptr<int32_t>(),
            reinterpret_cast<half *>(mask.data_ptr<mfq_half>()), rows,
            static_cast<int>(source_count), static_cast<int>(padded_count),
            static_cast<int>(key.size(2)));
        MFQ_CUDA_KERNEL_LAUNCH_CHECK();
        if (use_qsa_mma) {
            return mfq_sparse_attn_mma::launch<256, 256, 1, 16, false>(
                query, key, value, safe_indices, mask, Tensor{}, Tensor{},
                scale, "qwen4_sparse_gqa_attention");
        }
        return mfq_sparse_attn_mma::launch<512, 512, 1, 16, true>(
            query, key, key, safe_indices, mask, Tensor{}, Tensor{},
            scale, "glm5_sparse_mla_attention");
    }

    auto output = tb::empty({q.size(0), q.size(2), q.size(1), q.size(3)},
        q.options().dtype(tb::kFloat32));
    if (output.numel() == 0) return output;
    const int64_t blocks = output.numel() / q.size(3);
    MFQ_RUNTIME_CHECK(blocks <= std::numeric_limits<int>::max(),
        "sparse attention exceeds CUDA grid size");
    const auto stream = mfq_current_cuda_stream();
    #define MFQ_LAUNCH_SPARSE(S) sparse_attention_kernel<S><<<unsigned(blocks), 256, 0, stream>>>( \
        query.data_ptr<float>(), reinterpret_cast<const half*>(key.data_ptr()), \
        reinterpret_cast<const half*>(value.data_ptr()), indices.data_ptr<int32_t>(), \
        output.data_ptr<float>(), q.size(1), k.size(1), q.size(2), q.size(3), \
        k.size(2), selected.size(2), static_cast<float>(scale))
    if (q.size(3) <= 256) { MFQ_LAUNCH_SPARSE(1); }
    else if (q.size(3) <= 512) { MFQ_LAUNCH_SPARSE(2); }
    else if (q.size(3) <= 1024) { MFQ_LAUNCH_SPARSE(4); }
    else { MFQ_LAUNCH_SPARSE(0); }
    #undef MFQ_LAUNCH_SPARSE
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

Tensor dense(const Tensor& query, const Tensor& key, const Tensor& value,
                       int64_t offset, double scale) {
    attention_shapes(query, key, value);
    const auto count = key.size(2);
    MFQ_RUNTIME_CHECK(count > 0 && offset >= 0 && offset <= count &&
        query.size(2) <= count - offset, "attention query range is outside the cache");
    MFQ_RUNTIME_CHECK(std::isfinite(scale), "attention scale must be finite");
    MfqCudaGuard guard(query.device());
    const auto options = query.options().dtype(tb::kInt64);
    const auto kp = tb::arange(count, options).unsqueeze(0);
    const auto qp = tb::arange(query.size(2), options).unsqueeze(1) + offset;
    auto mask = (kp <= qp).unsqueeze(0).unsqueeze(0);
    return tb::scaled_dot_product_attention(query, key.to(query.scalar_type()),
        value.to(query.scalar_type()), mask, 0.0, false, scale, true).permute({0, 2, 1, 3});
}

} // namespace mfq_selected_attention
