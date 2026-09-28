#pragma once

#include "quant_linear.h"
#include "models/include/model_config.h"
#include "cuda_execution.h"
#include "storage/moe_expert_cache.h"
#include "mfq_cuda_ops.h"
#include "mfq_cuda_paged_kv.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using mfq_tensor_backend::indexing::Slice;

std::string layer_name(const std::string & templ, int i);

mfq_tensor_backend::Tensor qwen_rms_norm(
    mfq_tensor_backend::Tensor x,
    mfq_tensor_backend::Tensor weight,
    double eps,
    double weight_offset);

mfq_tensor_backend::Tensor qwen_rms_norm_bf16(
    mfq_tensor_backend::Tensor x,
    mfq_tensor_backend::Tensor weight,
    double eps,
    double weight_offset);

mfq_tensor_backend::Tensor gemma_rms_norm_f16(
    mfq_tensor_backend::Tensor x,
    mfq_tensor_backend::Tensor weight,
    double eps,
    double weight_offset);

struct RopeCache {
    mfq_tensor_backend::Tensor cos;
    mfq_tensor_backend::Tensor sin;
    mfq_tensor_backend::Tensor sections;
    mfq_tensor_backend::Tensor empty_sections;
    mfq_tensor_backend::Tensor interleaved_order;
    mfq_tensor_backend::Tensor interleaved_inverse;
    int64_t rotary_dim = 0;

    RopeCache() = default;
    RopeCache(
        int64_t max_positions,
        int64_t dim,
        double base,
        int64_t frequency_dim = 0,
        int64_t active_pairs = -1,
        mfq_tensor_backend::Device device = mfq_tensor_backend::Device(mfq_tensor_backend::kCUDA),
        bool official_reciprocal_frequencies = false) : rotary_dim(dim) {
        int64_t half = rotary_dim / 2;
        const int64_t denominator = frequency_dim > 0 ? frequency_dim : rotary_dim;
        if (active_pairs < 0) active_pairs = half;
        if (active_pairs > half) {
            throw std::runtime_error("RoPE active pair count exceeds rotary dimension");
        }
        auto opts = mfq_tensor_backend::TensorOptions().device(device).dtype(mfq_tensor_backend::kFloat32);
        auto pos = mfq_tensor_backend::arange(max_positions, opts);
        auto ar = mfq_tensor_backend::arange(0, rotary_dim, 2, opts);
        auto freq = mfq_tensor_backend::pow(
            mfq_tensor_backend::full({half}, base, opts),
            -ar / (double)denominator);
        if (official_reciprocal_frequencies) {
            auto cpu_opts = mfq_tensor_backend::TensorOptions()
                .device(mfq_tensor_backend::kCPU).dtype(mfq_tensor_backend::kFloat32);
#ifdef MFQ_NATIVE_CUDA_RUNTIME
            std::vector<float> official_values(static_cast<size_t>(half));
            const float official_base = static_cast<float>(base);
            const float official_denominator = static_cast<float>(denominator);
            for (int64_t index = 0; index < half; ++index) {
                const float exponent =
                    static_cast<float>(index * 2) / official_denominator;
                official_values[static_cast<size_t>(index)] =
                    1.0f / std::pow(official_base, exponent);
            }
            auto official_freq = mfq_tensor_backend::tensor(
                official_values, cpu_opts);
#else
            auto exponent = mfq_tensor_backend::arange(0, rotary_dim, 2, cpu_opts) /
                (double)denominator;
            auto official_freq = mfq_tensor_backend::reciprocal(
                mfq_tensor_backend::pow(mfq_tensor_backend::full({half}, base, cpu_opts), exponent));
#endif
            freq.copy_(official_freq);
        }
        if (active_pairs < half) {
            auto pair = mfq_tensor_backend::arange(half, opts);
            freq = mfq_tensor_backend::where(pair < active_pairs, freq, mfq_tensor_backend::zeros_like(freq));
        }
        auto ang = pos.unsqueeze(1) * freq.unsqueeze(0);
        cos = mfq_tensor_backend::cos(ang).contiguous();
        sin = mfq_tensor_backend::sin(ang).contiguous();
        sections = mfq_tensor_backend::empty({0}, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64).device(mfq_tensor_backend::kCPU));
        empty_sections = sections;
    }
    void configure_mrope(
            const std::vector<int64_t>& configured_sections,
            bool interleaved,
            int64_t configured_rotary_dim,
            mfq_tensor_backend::Device device);
    mfq_tensor_backend::Tensor apply(
            mfq_tensor_backend::Tensor x,
            mfq_tensor_backend::Tensor pos,
            bool grid_mrope_positions = false) const;

    mfq_tensor_backend::Tensor apply_bf16(mfq_tensor_backend::Tensor x, mfq_tensor_backend::Tensor pos) const;
};

struct FFN {
    QuantLinearGroup gate_up;
    QuantLinear down;
    std::unique_ptr<FFN> important_neurons;
    mutable std::shared_ptr<CudaIndependentBranchExecutor>
        important_neuron_executor =
            std::make_shared<CudaIndependentBranchExecutor>();
    bool geglu = false;
    bool is_moe = false;
    bool moe_split_gate_up = false;
    MfeWeight moe_gate_up;
    MfeWeight moe_gate;
    MfeWeight moe_up;
    MfeWeight moe_down;
    std::shared_ptr<MixedMoeRuntime> cpu_moe_gate_up;
    std::shared_ptr<MixedMoeRuntime> cpu_moe_gate;
    std::shared_ptr<MixedMoeRuntime> cpu_moe_up;
    std::shared_ptr<MixedMoeRuntime> cpu_moe_down;
    mfq_tensor_backend::Tensor moe_router;
    mfq_tensor_backend::Tensor moe_shared_gate;
    mfq_tensor_backend::Tensor moe_router_bias;
    mfq_tensor_backend::Tensor moe_hash_ids;
    std::unique_ptr<FFN> shared;
    int moe_top_k = 0;
    bool moe_use_sigmoid = false;
    bool moe_use_sqrt_softplus = false;
    bool moe_normalize = false;
    bool moe_delayed_softmax = true;
    bool moe_shared_ungated = false;
    double moe_router_scale = 1.0;
    double swiglu_limit = 0.0;
    int moe_layer = -1;

    bool uses_moe_expert_cache() const;

    bool tensor_parallel_dense_compatible() const;

    mfq_tensor_backend::Tensor forward_tensor_parallel_dense(
            mfq_tensor_backend::Tensor xh) const;

    bool expert_parallel_moe_compatible() const;

    mfq_tensor_backend::Tensor forward_expert_parallel_moe(
            mfq_tensor_backend::Tensor x,
            const MoeRoutePlan & route,
            mfq_tensor_backend::Tensor route_weights) const;

    mfq_tensor_backend::Tensor forward_dense_f32_down_kld(
            mfq_tensor_backend::Tensor xh) const;

    mfq_tensor_backend::Tensor forward_impl(
        mfq_tensor_backend::Tensor x,
        MfqOptional<mfq_tensor_backend::Tensor> input_ids,
        bool allow_important_neurons) const;

    bool can_forward_fused_residual(
        const mfq_tensor_backend::Tensor & x,
        const mfq_tensor_backend::Tensor & residual) const;

    mfq_tensor_backend::Tensor forward_fused_residual(
        mfq_tensor_backend::Tensor x,
        mfq_tensor_backend::Tensor residual) const;

    mfq_tensor_backend::Tensor forward(
        mfq_tensor_backend::Tensor x,
        MfqOptional<mfq_tensor_backend::Tensor> input_ids =
            mfq_nullopt) const;
};

struct KVCache {
    mfq_tensor_backend::Tensor k;
    mfq_tensor_backend::Tensor v;
    mfq_tensor_backend::Tensor k_chunk_ptrs;
    mfq_tensor_backend::Tensor v_chunk_ptrs;
    mfq_tensor_backend::Tensor page_table;
    bool ring = false;
    int64_t paged_batch = 0;
    int64_t paged_heads = 0;
    int64_t paged_head_dim = 0;
    int64_t page_size = 0;
    int64_t pages_per_chunk = 0;
    mfq_tensor_backend::ScalarType paged_dtype =
        mfq_tensor_backend::kFloat16;
    KVCache() = default;
    KVCache(
            int64_t B,
            int64_t H,
            int64_t max_seq,
            int64_t D,
            bool use_ring = false,
            mfq_tensor_backend::Device device = mfq_tensor_backend::Device(mfq_tensor_backend::kCUDA),
            mfq_tensor_backend::ScalarType dtype = mfq_tensor_backend::kFloat16)
        : ring(use_ring) {
        auto opts = mfq_tensor_backend::TensorOptions().device(device).dtype(dtype);
        k = mfq_tensor_backend::zeros({B, H, max_seq, D}, opts);
        v = mfq_tensor_backend::zeros({B, H, max_seq, D}, opts);
    }

    static KVCache paged_view(
            mfq_tensor_backend::Tensor key_chunks,
            mfq_tensor_backend::Tensor value_chunks,
            mfq_tensor_backend::Tensor pages,
            int64_t batch, int64_t heads, int64_t head_dim,
            int64_t tokens_per_page, int64_t chunk_pages,
            mfq_tensor_backend::ScalarType dtype) {
        KVCache result;
        result.k_chunk_ptrs = std::move(key_chunks);
        result.v_chunk_ptrs = std::move(value_chunks);
        result.page_table = std::move(pages);
        result.paged_batch = batch;
        result.paged_heads = heads;
        result.paged_head_dim = head_dim;
        result.page_size = tokens_per_page;
        result.pages_per_chunk = chunk_pages;
        result.paged_dtype = dtype;
        return result;
    }

    bool is_paged() const noexcept { return page_size > 0; }

    bool defined() const noexcept {
        return is_paged()
            ? k_chunk_ptrs.defined() && v_chunk_ptrs.defined() &&
                page_table.defined()
            : k.defined() && v.defined();
    }

    int64_t batch_size() const noexcept {
        return is_paged() ? paged_batch : (k.defined() ? k.size(0) : 0);
    }

    mfq_tensor_backend::ScalarType scalar_type() const {
        return is_paged() ? paged_dtype : k.scalar_type();
    }

    std::pair<mfq_tensor_backend::Tensor, mfq_tensor_backend::Tensor> append(
            mfq_tensor_backend::Tensor kk, mfq_tensor_backend::Tensor vv, mfq_tensor_backend::Tensor pos,
            int64_t start_pos, int64_t end_pos,
            bool contiguous_prefill_prefix = false);
};

struct Block {
    struct Context {
        mfq_tensor_backend::Tensor token_ids;
        mfq_tensor_backend::Tensor positions;
        mfq_tensor_backend::Tensor full_positions;
        int64_t cache_position = 0;
        int64_t confirmed_prefix = 0;
        int64_t planned_kv_length = 0;
        int64_t decode_attention_parts = 0;
        MfqOptional<mfq_tensor_backend::Tensor> sequence_lengths = mfq_nullopt;
        MfqOptional<mfq_tensor_backend::Tensor> cache_positions = mfq_nullopt;
        MfqOptional<mfq_tensor_backend::Tensor> attention_mask = mfq_nullopt;
    };
    int cuda_device = 0;
    bool cpu_offloaded = false;
    virtual ~Block() = default;
    virtual void reset(int64_t B) = 0;
    virtual void set_token_ids(const mfq_tensor_backend::Tensor &) {}
    virtual bool supports_speculation() const noexcept { return false; }
    virtual void begin_speculative(int64_t) {}
    virtual void commit_speculative() {}
    virtual void rollback_speculative(int64_t) {}
    virtual mfq_tensor_backend::Tensor forward(
            mfq_tensor_backend::Tensor x,
            mfq_tensor_backend::Tensor pos,
            int64_t cache_pos,
            const MfqOptional<mfq_tensor_backend::Tensor> & seq_len,
            const RopeCache & rope,
            const MfqOptional<mfq_tensor_backend::Tensor> & cache_positions = mfq_nullopt,
            const MfqOptional<mfq_tensor_backend::Tensor> & attention_mask = mfq_nullopt) = 0;
    virtual mfq_tensor_backend::Tensor forward_context(
            mfq_tensor_backend::Tensor x,
            const Context & context,
            const RopeCache & rope) {
        MFQ_RUNTIME_CHECK(
            context.confirmed_prefix == 0 || supports_speculation(),
            "block does not support speculative verification");
        return forward(
            std::move(x), context.positions, context.cache_position,
            context.sequence_lengths, rope, context.cache_positions,
            context.attention_mask);
    }
};

struct FullBlock : Block {
    int layer = -1;
    bool gemma4 = false;
    bool gemma4_moe = false;
    bool sliding = false;
    bool value_equals_key = false;
    int64_t attention_heads = 0;
    int64_t kv_heads = 0;
    int64_t attention_head_dim = 0;
    int64_t attention_rotary_dim = 0;
    int64_t attention_window = 0;
    int64_t max_position_embeddings = 0;
    double attention_scale = 0.0;
    double rms_norm_eps = 1e-6;
    double norm_weight_offset = 1.0;
    bool official_bf16 = false;
    RopeCache attention_rope;

    bool supports_speculation() const noexcept override { return !sliding; }
    mfq_tensor_backend::Tensor attn_norm, ffn_norm, q_norm, k_norm;
    mfq_tensor_backend::Tensor v_norm, attn_post_norm;
    mfq_tensor_backend::Tensor ffn_post_norm, ffn_post_norm_1, ffn_pre_norm_2, ffn_post_norm_2;
    mfq_tensor_backend::Tensor layer_scale;
    QuantLinearGroup qkv;
    bool attention_output_gate = false;
    bool split_q_kv_projections = false;
    QuantLinear q_projection;
    QuantLinear k_projection;
    QuantLinear v_projection;
    QuantLinear o;
    FFN ffn;
    MfeWeight gemma_moe_gate_up;
    MfeWeight gemma_moe_down;
    mfq_tensor_backend::Tensor gemma_router;
    mfq_tensor_backend::Tensor gemma_router_norm_scale;
    mfq_tensor_backend::Tensor gemma_expert_scale;
    int gemma_top_k = 0;
    KVCache cache;
    mfq_tensor_backend::Tensor decode_partial_o, decode_partial_m, decode_partial_l;
    mfq_tensor_backend::Tensor decode_mma_mask, decode_mma_kv_max, decode_mma_meta;

    static constexpr int64_t kDecodeAttentionMaxParts = 16;

    void reset(int64_t B) override;

    mfq_tensor_backend::Tensor forward(
            mfq_tensor_backend::Tensor x,
            mfq_tensor_backend::Tensor pos,
            int64_t cache_pos,
            const MfqOptional<mfq_tensor_backend::Tensor> & seq_len,
            const RopeCache & rope,
            const MfqOptional<mfq_tensor_backend::Tensor> & cache_positions = mfq_nullopt,
            const MfqOptional<mfq_tensor_backend::Tensor> & attention_mask = mfq_nullopt) override;

    mfq_tensor_backend::Tensor forward_context(
            mfq_tensor_backend::Tensor x,
            const Context& context,
            const RopeCache& rope) override;

    mfq_tensor_backend::Tensor forward_impl(
            mfq_tensor_backend::Tensor x,
            mfq_tensor_backend::Tensor pos,
            int64_t cache_pos,
            const MfqOptional<mfq_tensor_backend::Tensor>& seq_len,
            const RopeCache& rope,
            const MfqOptional<mfq_tensor_backend::Tensor>& cache_positions,
            const MfqOptional<mfq_tensor_backend::Tensor>& attention_mask,
            int64_t planned_kv_length,
            int64_t decode_attention_parts);
};

void prepare_ffn_workspaces(FFN & f);

FFN load_ffn(
    const mfq::ModelSource& source,
    const mfq::models::ModelConfig& config,
    int layer,
    bool minicpmo45 = false,
    std::string_view tensor_root = "model");

std::unique_ptr<Block> load_transformer_block(
    const mfq::ModelSource& source,
    const mfq::models::ModelConfig& config,
    int layer,
    const std::string& type,
    bool minicpmo45 = false,
    std::string_view tensor_root = "model");

void load_important_neuron_branch(
        const mfq::ModelSource & mfq,
        int64_t hidden_size,
        int64_t intermediate_size,
        FFN & f,
        const std::string & down_name,
        const std::string & gate_name,
        const std::string & up_name);
