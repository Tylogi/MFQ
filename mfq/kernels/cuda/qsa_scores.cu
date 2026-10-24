#include "qsa_scores.h"
#include "selected_attention.h"

#include <cmath>
#include <cstdlib>
#include <limits>

namespace mfq_qsa {
namespace tb = mfq_tensor_backend;
using tb::Tensor;
namespace {
constexpr int tile_m = 32;
constexpr int tile_n = 64;
constexpr int tile_k = 32;
constexpr int threads = 512;
constexpr int dot_chains = 16;

struct ScoreLayout {
    int64_t qb, qt, qh, qd, kb, kp, kd;
    int rows, pools, heads, width;
};

template<class Query, class Key, class Storage>
__device__ void load_tile(const Query* query, const Key* keys, Storage& shared,
    ScoreLayout layout, int row_base, int pool_base, int begin) {
    const int query_rows = tile_m / layout.heads;
    for (int i = threadIdx.x; i < tile_m * tile_k; i += threads) {
        const int m = i / tile_k, d = i % tile_k;
        const int row = row_base + m / layout.heads, head = m % layout.heads;
        const bool valid = m < query_rows * layout.heads && row < layout.rows && begin + d < layout.width;
        shared.q[m][d] = valid ? query[int64_t(blockIdx.z) * layout.qb + int64_t(row) * layout.qt +
            int64_t(head) * layout.qh + int64_t(begin + d) * layout.qd] : Query(0.f);
    }
    for (int i = threadIdx.x; i < tile_n * tile_k; i += threads) {
        const int n = i / tile_k, d = i % tile_k;
        const bool valid = pool_base + n < layout.pools && begin + d < layout.width;
        shared.k[n][d] = valid ? keys[int64_t(blockIdx.z) * layout.kb +
            int64_t(pool_base + n) * layout.kp + int64_t(begin + d) * layout.kd] : Key(0.f);
    }
}

__device__ void reduce_heads(const float* products, float* output, ScoreLayout layout,
    int row_base, int pool_base, double divisor) {
    const int rows = tile_m / layout.heads;
    for (int i = threadIdx.x; i < rows * tile_n; i += threads) {
        const int row = i / tile_n, pool = i % tile_n;
        if (row_base + row >= layout.rows || pool_base + pool >= layout.pools) continue;
        float score = 0.f;
        // QSA's head weights are one. Preserve its FP32, ascending-head sum.
        for (int head = 0; head < layout.heads; ++head)
            score += fmaxf(products[(row * layout.heads + head) * tile_n + pool], 0.f);
        output[(int64_t(blockIdx.z) * layout.rows + row_base + row) * layout.pools + pool_base + pool] =
            float(double(score) / divisor);
    }
}

// Four dots per thread reuse a 32x64 tile without rounding FP32 operands.
// Sixteen independent FP32 chains followed by an ascending sum retain the
// batched GEMM's reduction at close top-k boundaries. Products stay on chip.
template<class Query, class Key>
__global__ void simt_scores(const Query* query, const Key* keys, float* output,
    ScoreLayout layout, double divisor) {
    union SharedTile {
        struct {
            float q[tile_m][tile_k + 1];
            float k[tile_n][tile_k + 1];
        } input;
        float products[tile_m][tile_n];
    };
    __shared__ SharedTile shared;
    const int row_base = int(blockIdx.y) * (tile_m / layout.heads);
    const int pool_base = int(blockIdx.x) * tile_n;
    const int m = threadIdx.x / 32, n = threadIdx.x % 32;
    float accum[dot_chains][2][2]{};
    for (int begin = 0; begin < layout.width; begin += tile_k) {
        load_tile(query, keys, shared.input, layout, row_base, pool_base, begin);
        __syncthreads();
        #pragma unroll
        for (int d = 0; d < tile_k; ++d) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                const float key = shared.input.k[n + j * 32][d];
                #pragma unroll
                for (int i = 0; i < 2; ++i)
                    accum[d % dot_chains][i][j] =
                        fmaf(shared.input.q[m + i * 16][d], key, accum[d % dot_chains][i][j]);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            float dot = accum[0][i][j];
            #pragma unroll
            for (int part = 1; part < dot_chains; ++part) dot += accum[part][i][j];
            shared.products[m + i * 16][n + j * 32] = dot;
        }
    }
    __syncthreads();
    reduce_heads(&shared.products[0][0], output, layout, row_base, pool_base, divisor);
}

Tensor legacy_scores(const Tensor& query, const Tensor& keys) {
    const auto* option = std::getenv("MFQ_QSA_PREFILL_SCORE_GEMM");
    if (query.size(1) >= 32 && option && option[0] == '1') {
        auto q = query.to(tb::kFloat32).permute({0, 2, 1, 3}).contiguous();
        auto k = keys.to(tb::kFloat32).transpose(-1, -2).unsqueeze(1);
        auto products = tb::matmul(q, k).permute({0, 2, 1, 3}).clamp_min(0.0);
        return products.sum(-2) / std::sqrt(double(query.size(-1)));
    }
    return mfq_selected_attention::dots(query, keys).sum(-2) / std::sqrt(double(query.size(-1)));
}
} // namespace

Tensor block_scores(const Tensor& query, const Tensor& keys) {
    mfq_selected_attention::values({&query, &keys});
    MFQ_RUNTIME_CHECK(query.dim() == 4 && keys.dim() == 3 && query.size(0) == keys.size(0) &&
        query.size(3) == keys.size(2) && query.size(2) > 0 && query.size(3) > 0,
        "QSA scores require [B,T,H,D] query and [B,P,D] keys");
    MfqCudaGuard guard(query.device());
    const auto* option = std::getenv("MFQ_QSA_PREFILL_TILED_SCORES");
    const auto* gemm = std::getenv("MFQ_QSA_PREFILL_SCORE_GEMM");
    // Small/decode rows and large heads/widths keep their original GEMM path.
    // Tiles support arbitrary strides and partial T/P/D tiles, including odd H.
    const bool supported = query.size(1) >= 32 && query.size(2) <= tile_m && query.size(3) <= 256 &&
        query.size(0) <= 65535 && (query.size(1) + tile_m / query.size(2) - 1) /
        (tile_m / query.size(2)) <= 65535 && keys.size(1) <= std::numeric_limits<int>::max();
    // An explicit legacy GEMM request still wins unless tiling is explicit.
    if (!supported || (option && option[0] == '0') || (!option && gemm && gemm[0] == '1'))
        return legacy_scores(query, keys);
    auto output = tb::empty({query.size(0), query.size(1), keys.size(1)}, query.options().dtype(tb::kFloat32));
    if (!output.numel()) return output;
    const ScoreLayout layout{query.stride(0), query.stride(1), query.stride(2), query.stride(3),
        keys.stride(0), keys.stride(1), keys.stride(2), int(query.size(1)), int(keys.size(1)),
        int(query.size(2)), int(query.size(3))};
    const int rows = tile_m / layout.heads;
    const dim3 grid((int64_t(layout.pools) + tile_n - 1) / tile_n,
        (layout.rows + rows - 1) / rows, query.size(0));
    const double divisor = std::sqrt(double(layout.width));
    const auto launch = [&](auto* q, auto* k) {
        simt_scores<<<grid, threads, 0, mfq_current_cuda_stream()>>>(q, k,
            output.data_ptr<float>(), layout, divisor);
    };
    const auto dispatch_key = [&](auto* q) {
        if (keys.scalar_type() == tb::kFloat32) launch(q, keys.data_ptr<float>());
        else if (keys.scalar_type() == tb::kFloat16) launch(q, keys.data_ptr<mfq_half>());
        else launch(q, keys.data_ptr<mfq_bfloat16>());
    };
    if (query.scalar_type() == tb::kFloat32) dispatch_key(query.data_ptr<float>());
    else if (query.scalar_type() == tb::kFloat16) dispatch_key(query.data_ptr<mfq_half>());
    else dispatch_key(query.data_ptr<mfq_bfloat16>());
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
} // namespace mfq_qsa
