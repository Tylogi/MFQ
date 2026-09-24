#pragma once

#include "../../runtime/cuda_transformer.h"
#include "models/glm_dsa.h"

#include <memory>

namespace mfq::cuda::glm_dsa {
using Config = mfq::models::glm_dsa::Config;
}

struct GlmDsaSharedState {
    mfq_tensor_backend::Tensor topk_indices;
    int64_t dense_prefix_rows = 0;
    std::unordered_map<int64_t, MoeRoutePlan> head_routes;
    MoeRoutePlan transient_head_route;
    int64_t transient_head_route_key = -1;
    mfq_tensor_backend::Tensor dense_mask;
    mfq_tensor_backend::Tensor decode_mask;
    mfq_tensor_backend::Tensor kv_max;
    mfq_tensor_backend::Tensor attention_meta;

    void reset() {
        topk_indices = mfq_tensor_backend::Tensor();
        dense_prefix_rows = 0;
    }

    const MoeRoutePlan & head_route(int64_t rows, int heads) {
        const int64_t key = rows * 4096 + heads;
        if (rows != heads) {
            if (transient_head_route_key != key) {
                auto options = mfq_tensor_backend::TensorOptions()
                    .device(mfq_tensor_backend::kCUDA).dtype(mfq_tensor_backend::kInt32);
                auto ids = mfq_tensor_backend::remainder(mfq_tensor_backend::arange(rows, options), heads)
                    .reshape({rows, 1}).contiguous();
                transient_head_route = build_moe_route_plan(ids, heads);
                transient_head_route_key = key;
            }
            return transient_head_route;
        }
        auto found = head_routes.find(key);
        if (found != head_routes.end()) return found->second;
        auto options = mfq_tensor_backend::TensorOptions()
            .device(mfq_tensor_backend::kCUDA).dtype(mfq_tensor_backend::kInt32);
        auto ids = mfq_tensor_backend::remainder(mfq_tensor_backend::arange(rows, options), heads)
            .reshape({rows, 1}).contiguous();
        return head_routes.emplace(
            key, build_moe_route_plan(ids, heads)).first->second;
    }

    void ensure_meta() {
        constexpr int64_t kMetaFloats = 8 * 1024 * 1024;
        auto cuda = mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA);
        if (!attention_meta.defined() || attention_meta.numel() < kMetaFloats) {
            attention_meta = mfq_tensor_backend::empty(
                {kMetaFloats}, cuda.dtype(mfq_tensor_backend::kFloat32));
        }
    }

    void ensure_dense_workspace(int64_t B, int64_t M, int64_t logical_len) {
        const int64_t stride = (logical_len + 63) / 64 * 64;
        auto cuda = mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA);
        if (!dense_mask.defined() || dense_mask.size(0) < M ||
            dense_mask.size(1) < stride) {
            dense_mask = mfq_tensor_backend::empty(
                {M, stride}, cuda.dtype(mfq_tensor_backend::kFloat16));
        }
        if (!kv_max.defined() || kv_max.numel() < B * M) {
            kv_max = mfq_tensor_backend::empty({B * M}, cuda.dtype(mfq_tensor_backend::kInt32));
        }
        ensure_meta();
    }

    void ensure_decode_workspace(int64_t B, int64_t planned_len) {
        const int64_t stride = (planned_len + 63) / 64 * 64;
        auto cuda = mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA);
        if (!decode_mask.defined() || decode_mask.size(0) != B ||
            decode_mask.size(1) < stride) {
            decode_mask = mfq_tensor_backend::empty(
                {B, stride}, cuda.dtype(mfq_tensor_backend::kFloat16));
        }
        if (!kv_max.defined() || kv_max.numel() < B) {
            kv_max = mfq_tensor_backend::empty({B}, cuda.dtype(mfq_tensor_backend::kInt32));
        }
        ensure_meta();
    }
};

struct GlmDsaBlock : Block {
    mfq::cuda::glm_dsa::Config config;
    int layer = -1;
    bool full_indexer = false;
    std::shared_ptr<GlmDsaSharedState> shared_state;
    mfq_tensor_backend::Tensor attn_norm, ffn_norm;
    mfq_tensor_backend::Tensor q_a_norm, kv_a_norm;
    mfq_tensor_backend::Tensor index_k_norm, index_k_bias;
    QuantLinearGroup input_proj;
    QuantLinearGroup q_proj;
    MfeWeight embed_q;
    MfeWeight unembed_out;
    QuantLinear o_proj;
    FFN ffn;
    mfq_tensor_backend::Tensor kv_cache;
    mfq_tensor_backend::Tensor index_cache;

    void reset(int64_t B) override {
        shared_state->reset();
        if (kv_cache.defined() && kv_cache.size(0) == B) return;
        kv_cache = mfq_tensor_backend::Tensor();
        index_cache = mfq_tensor_backend::Tensor();
    }

    mfq_tensor_backend::Tensor headwise_project(
        const MfeWeight & weight, mfq_tensor_backend::Tensor x,
        int64_t B, int64_t T, int64_t heads) const {
        if (x.dim() != 4 || x.size(0) != B || x.size(1) != T ||
            x.size(2) != heads || x.size(3) != weight.neuron_len ||
            weight.n_experts != heads) {
            throw std::runtime_error("GLM head-wise MFE projection shape mismatch");
        }
        const int64_t rows = B * T * heads;
        const auto & route = shared_state->head_route(rows, static_cast<int>(heads));
        auto y = weight.forward(x.contiguous().reshape({rows, weight.neuron_len}), route);
        return y.reshape({B, T, heads, weight.out_per_expert});
    }

    mfq_tensor_backend::Tensor dense_attention(
        mfq_tensor_backend::Tensor q, int64_t logical_len, int64_t B,
        const MfqOptional<mfq_tensor_backend::Tensor> & seq_len,
        double scale, int64_t planned_kv_length) const {
        if (seq_len.has_value() && q.size(2) == 1 && B == 1) {
            const int64_t planned_len = planned_kv_length > 0
                ? planned_kv_length : logical_len;
            shared_state->ensure_decode_workspace(B, planned_len);
            return attention_glm_mla576_decode_cuda(
                q, kv_cache, seq_len.value(), scale, planned_len,
                shared_state->decode_mask, shared_state->kv_max,
                shared_state->attention_meta);
        }
        shared_state->ensure_dense_workspace(B, q.size(2), logical_len);
        return attention_glm_mla576_cached_cuda(
            q, kv_cache, logical_len, shared_state->dense_mask,
            shared_state->kv_max, shared_state->attention_meta, scale);
    }

    void update_indexer(
        mfq_tensor_backend::Tensor index_q, mfq_tensor_backend::Tensor index_weights,
        int64_t B, int64_t T, int64_t cache_pos,
        const MfqOptional<mfq_tensor_backend::Tensor> & seq_len,
        int64_t planned_kv_length) const {
        const auto& c = config;
        const int64_t logical_len = cache_pos + T;
        if (logical_len <= c.index_topk) {
            shared_state->topk_indices = mfq_tensor_backend::Tensor();
            shared_state->dense_prefix_rows = T;
            return;
        }

        if (seq_len.has_value() && T == 1 && B == 1) {
            const int64_t planned_len = planned_kv_length > 0
                ? planned_kv_length : logical_len;
            auto scores = g_profiler.measure("glm.indexer_scores", [&]() {
                return glm_dsa_indexer_scores_decode_cuda(
                    index_q, index_cache, index_weights,
                    seq_len.value(), planned_len);
            });
            auto selected = g_profiler.measure("glm.indexer_topk", [&]() {
                return mfq_tensor_backend::topk(scores, c.index_topk, -1, true, false);
            });
            shared_state->topk_indices =
                std::get<1>(selected).to(mfq_tensor_backend::kInt32).contiguous();
            shared_state->dense_prefix_rows = 0;
            return;
        }

        const int64_t prefix_rows = std::max<int64_t>(
            0, std::min<int64_t>(T, c.index_topk - cache_pos));
        const int64_t sparse_rows = T - prefix_rows;
        if (sparse_rows <= 0) {
            shared_state->topk_indices = mfq_tensor_backend::Tensor();
            shared_state->dense_prefix_rows = T;
            return;
        }
        auto indices = mfq_tensor_backend::empty(
            {B, sparse_rows, c.index_topk},
            mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA).dtype(mfq_tensor_backend::kInt32));
        constexpr int64_t kMaxScoreElements = 32 * 1024 * 1024;
        int64_t rows_per_chunk = std::max<int64_t>(
            1, kMaxScoreElements / std::max<int64_t>(1, B * logical_len));
        rows_per_chunk = std::min<int64_t>(rows_per_chunk, 256);
        if (rows_per_chunk >= 64) rows_per_chunk = rows_per_chunk / 64 * 64;
        for (int64_t start = 0; start < sparse_rows; start += rows_per_chunk) {
            const int64_t count = std::min<int64_t>(rows_per_chunk, sparse_rows - start);
            auto q_chunk = index_q.narrow(1, prefix_rows + start, count).contiguous();
            auto w_chunk = index_weights.narrow(1, prefix_rows + start, count).contiguous();
            auto scores = g_profiler.measure("glm.indexer_scores", [&]() {
                return glm_dsa_indexer_scores_cuda(
                    q_chunk, index_cache, w_chunk,
                    cache_pos + prefix_rows + start, logical_len);
            });
            auto selected = g_profiler.measure("glm.indexer_topk", [&]() {
                return mfq_tensor_backend::topk(scores, c.index_topk, -1, true, false);
            });
            indices.narrow(1, start, count).copy_(
                std::get<1>(selected).to(mfq_tensor_backend::kInt32));
        }
        shared_state->topk_indices = indices;
        shared_state->dense_prefix_rows = prefix_rows;
    }

    mfq_tensor_backend::Tensor forward(
        mfq_tensor_backend::Tensor x, mfq_tensor_backend::Tensor pos, int64_t cache_pos,
        const MfqOptional<mfq_tensor_backend::Tensor> & seq_len,
        const RopeCache & rope,
        const MfqOptional<mfq_tensor_backend::Tensor> & cache_positions = mfq_nullopt,
        const MfqOptional<mfq_tensor_backend::Tensor> & attention_mask = mfq_nullopt) override {
        return forward_impl(
            std::move(x), std::move(pos), cache_pos, seq_len, rope,
            cache_positions, attention_mask, 0);
    }

    mfq_tensor_backend::Tensor forward_context(
            mfq_tensor_backend::Tensor x,
            const Context& context,
            const RopeCache& rope) override {
        MFQ_RUNTIME_CHECK(
            context.confirmed_prefix == 0,
            "GLM DSA does not support speculative verification");
        return forward_impl(
            std::move(x), context.positions, context.cache_position,
            context.sequence_lengths, rope, context.cache_positions,
            context.attention_mask, context.planned_kv_length);
    }

    mfq_tensor_backend::Tensor forward_impl(
        mfq_tensor_backend::Tensor x, mfq_tensor_backend::Tensor pos,
        int64_t cache_pos,
        const MfqOptional<mfq_tensor_backend::Tensor>& seq_len,
        const RopeCache& rope,
        const MfqOptional<mfq_tensor_backend::Tensor>& cache_positions,
        const MfqOptional<mfq_tensor_backend::Tensor>& attention_mask,
        int64_t planned_kv_length) {
        const auto& c = config;
        (void)cache_positions;
        (void)attention_mask;
        const int64_t B = x.size(0);
        const int64_t T = x.size(1);
        const int64_t H = x.size(2);
        constexpr int64_t kHeads = 64;
        constexpr int64_t kNope = 192;
        constexpr int64_t kRope = 64;
        constexpr int64_t kLatent = 512;
        constexpr int64_t kMlaWidth = kLatent + kRope;
        constexpr int64_t kValue = 256;
        const int64_t logical_len = cache_pos + T;
        if (!kv_cache.defined()) {
            auto options = mfq_tensor_backend::TensorOptions()
                .device(mfq_tensor_backend::kCUDA).dtype(mfq_tensor_backend::kFloat16);
            kv_cache = mfq_tensor_backend::empty(
                {B, 1, c.max_position_embeddings, kMlaWidth}, options);
            if (full_indexer) {
                index_cache = mfq_tensor_backend::empty(
                    {B, c.max_position_embeddings, c.index_head_dim}, options);
            }
        }

        auto residual = x.scalar_type() == mfq_tensor_backend::kFloat16
            ? x.contiguous() : x.to(mfq_tensor_backend::kFloat16).contiguous();
        auto xn = g_profiler.measure("glm.attn_norm", [&]() {
            return rms_norm_f16_cuda(
                residual.reshape({B * T, H}), attn_norm,
                c.rms_norm_eps, 0.0).reshape({B, T, H});
        });
        auto first = g_profiler.measure("glm.input_proj", [&]() {
            return input_proj.forward(xn);
        });
        const size_t expected_first = full_indexer ? 4u : 2u;
        if (first.size() != expected_first) {
            throw std::runtime_error("GLM DSA input projection count mismatch");
        }
        auto qr = g_profiler.measure("glm.q_a_norm", [&]() {
            return rms_norm_f16_cuda(
                first[0].reshape({B * T, c.q_lora_rank}).to(mfq_tensor_backend::kFloat16).contiguous(),
                q_a_norm, 1e-6, 0.0).reshape({B, T, c.q_lora_rank});
        });
        auto second = g_profiler.measure("glm.q_proj", [&]() {
            return q_proj.forward(qr);
        });
        const size_t expected_second = full_indexer ? 2u : 1u;
        if (second.size() != expected_second) {
            throw std::runtime_error("GLM DSA q projection count mismatch");
        }

        auto q_main = second[0].to(mfq_tensor_backend::kFloat16)
            .reshape({B, T, kHeads, kNope + kRope});
        auto q_nope = q_main.index({Slice(), Slice(), Slice(), Slice(0, kNope)})
            .contiguous();
        auto q_pe = q_main.index({Slice(), Slice(), Slice(), Slice(kNope, kNope + kRope)})
            .permute({0, 2, 1, 3}).contiguous();
        q_pe = g_profiler.measure("glm.q_rope", [&]() {
            return glm_interleaved_rope_cuda(
                q_pe, pos.contiguous(), rope.cos, rope.sin, kRope);
        });

        auto compressed = first[1].to(mfq_tensor_backend::kFloat16)
            .reshape({B, T, kLatent + kRope});
        auto kv_latent = g_profiler.measure("glm.kv_a_norm", [&]() {
            auto raw = compressed.index({Slice(), Slice(), Slice(0, kLatent)})
                .contiguous();
            return rms_norm_f16_cuda(
                raw.reshape({B * T, kLatent}), kv_a_norm,
                1e-6, 0.0).reshape({B, T, kLatent});
        });
        auto k_pe = compressed.index({Slice(), Slice(), Slice(kLatent, kLatent + kRope)})
            .reshape({B, T, 1, kRope}).permute({0, 2, 1, 3}).contiguous();
        k_pe = g_profiler.measure("glm.k_rope", [&]() {
            return glm_interleaved_rope_cuda(
                k_pe, pos.contiguous(), rope.cos, rope.sin, kRope);
        });
        auto kv_rows = mfq_tensor_backend::cat({
            kv_latent,
            k_pe.permute({0, 2, 1, 3}).reshape({B, T, kRope})}, -1)
            .contiguous();
        g_profiler.measure("glm.kv_write", [&]() {
            return glm_dsa_cache_write_cuda(
                kv_cache.view({B, c.max_position_embeddings, kMlaWidth}),
                kv_rows, pos.contiguous());
        });

        if (full_indexer) {
            auto index_k = g_profiler.measure("glm.indexer_k_norm", [&]() {
                return glm_dsa_indexer_layer_norm_cuda(
                    first[2].to(mfq_tensor_backend::kFloat16).reshape({B, T, c.index_head_dim}).contiguous(),
                    index_k_norm, index_k_bias, 1e-5);
            });
            index_k = glm_interleaved_rope_cuda(
                index_k.reshape({B, T, 1, c.index_head_dim})
                    .permute({0, 2, 1, 3}).contiguous(),
                pos.contiguous(), rope.cos, rope.sin, kRope)
                .permute({0, 2, 1, 3}).reshape({B, T, c.index_head_dim}).contiguous();
            g_profiler.measure("glm.indexer_k_write", [&]() {
                return glm_dsa_cache_write_cuda(
                    index_cache, index_k, pos.contiguous());
            });
            auto index_q = second[1].to(mfq_tensor_backend::kFloat16)
                .reshape({B, T, c.index_n_heads, c.index_head_dim})
                .permute({0, 2, 1, 3}).contiguous();
            index_q = glm_interleaved_rope_cuda(
                index_q, pos.contiguous(), rope.cos, rope.sin, kRope)
                .permute({0, 2, 1, 3}).contiguous();
            auto index_weights = first[3].reshape({B, T, c.index_n_heads})
                .to(mfq_tensor_backend::kFloat32).contiguous();
            update_indexer(
                index_q, index_weights, B, T, cache_pos, seq_len,
                planned_kv_length);
        }

        auto q_absorbed = g_profiler.measure("glm.embed_q", [&]() {
            return headwise_project(embed_q, q_nope, B, T, kHeads);
        }).permute({0, 2, 1, 3}).contiguous();
        auto q_mla = mfq_tensor_backend::cat({q_absorbed, q_pe}, -1)
            .to(mfq_tensor_backend::kFloat32).contiguous();
        const double scale = 1.0 / std::sqrt(
            static_cast<double>(kNope + kRope));
        mfq_tensor_backend::Tensor attended;
        if (!shared_state->topk_indices.defined()) {
            attended = g_profiler.measure("glm.attention_dense", [&]() {
                return dense_attention(
                    q_mla, logical_len, B, seq_len, scale,
                    planned_kv_length);
            });
        } else {
            const int64_t prefix = shared_state->dense_prefix_rows;
            const int64_t sparse_rows = shared_state->topk_indices.size(1);
            if (prefix + sparse_rows != T) {
                throw std::runtime_error("GLM DSA shared index state has the wrong row count");
            }
            mfq_tensor_backend::Tensor dense_out;
            if (prefix > 0) {
                dense_out = g_profiler.measure("glm.attention_dense_prefix", [&]() {
                    return dense_attention(
                        q_mla.narrow(2, 0, prefix).contiguous(),
                        cache_pos + prefix, B, mfq_nullopt, scale,
                        planned_kv_length);
                });
            }
            shared_state->ensure_meta();
            auto sparse_out = g_profiler.measure("glm.attention_sparse", [&]() {
                return attention_glm_mla_sparse_cuda(
                    q_mla.narrow(2, prefix, sparse_rows).contiguous(),
                    kv_cache.view({B, c.max_position_embeddings, kMlaWidth}),
                    shared_state->topk_indices,
                    shared_state->attention_meta, scale);
            });
            attended = prefix > 0
                ? mfq_tensor_backend::cat({dense_out, sparse_out}, 1) : sparse_out;
        }
        auto value_heads = g_profiler.measure("glm.unembed_out", [&]() {
            return headwise_project(
                unembed_out, attended.to(mfq_tensor_backend::kFloat16).contiguous(),
                B, T, kHeads);
        });
        auto attn_out = g_profiler.measure("glm.o_proj", [&]() {
            return o_proj.forward(
                value_heads.reshape({B, T, kHeads * kValue}).contiguous());
        });
        auto attn_pair = g_profiler.measure("glm.attn_residual_ffn_norm", [&]() {
            return acc_rms_norm_f16_cuda(
                residual.reshape({B * T, H}),
                attn_out.reshape({B * T, H}).to(mfq_tensor_backend::kFloat16).contiguous(),
                ffn_norm, c.rms_norm_eps, 0.0);
        });
        auto hidden = attn_pair[0].reshape({B, T, H});
        auto ffn_input = attn_pair[1].reshape({B * T, H});
        auto ffn_out = ffn.forward(ffn_input).reshape({B * T, H});
        return g_profiler.measure("glm.ffn_residual", [&]() {
            return acc_cuda(hidden.reshape({B * T, H}), ffn_out)
                .reshape({B, T, H});
        });
    }
};
namespace mfq::cuda::glm_dsa {

void load_ffn(
    const mfq::ModelSource& source,
    const Config& config,
    int layer,
    ::FFN& ffn);
std::unique_ptr<::Block> load_block(
    const mfq::ModelSource& source,
    const Config& config,
    int layer,
    const std::string& type,
    const std::shared_ptr<::GlmDsaSharedState>& state);

} // namespace mfq::cuda::glm_dsa
