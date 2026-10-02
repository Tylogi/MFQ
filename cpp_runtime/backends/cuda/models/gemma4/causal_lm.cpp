#include "causal_lm.h"
#include "../causal_lm_impl.h"

#include "models/full_block.h"

#include <bit>
#include <cmath>

namespace mfq::cuda::gemma4 {

struct Gemma4Block final : ::FullBlock {
    bool gemma4_moe = false;
    mfq_tensor_backend::Tensor attn_post_norm;
    mfq_tensor_backend::Tensor ffn_post_norm;
    mfq_tensor_backend::Tensor ffn_post_norm_1;
    mfq_tensor_backend::Tensor ffn_pre_norm_2;
    mfq_tensor_backend::Tensor ffn_post_norm_2;
    mfq_tensor_backend::Tensor layer_scale;
    MfeWeight gemma_moe_gate_up;
    MfeWeight gemma_moe_down;
    mfq_tensor_backend::Tensor gemma_router;
    mfq_tensor_backend::Tensor gemma_router_norm_scale;
    mfq_tensor_backend::Tensor gemma_expert_scale;
    int gemma_top_k = 0;

    mfq_tensor_backend::Tensor forward_ffn(
        CudaExecutionContext& execution,
        mfq_tensor_backend::Tensor residual,
        mfq_tensor_backend::Tensor attention_output,
        int64_t batch,
        int64_t tokens,
        int64_t hidden) override;
};

static mfq_tensor_backend::Tensor gemma_rms_norm_f16(
        mfq_tensor_backend::Tensor x,
        mfq_tensor_backend::Tensor weight,
        double eps,
        double weight_offset) {
    MFQ_RUNTIME_CHECK(
        x.scalar_type() == mfq_tensor_backend::kFloat16,
        "gemma_rms_norm_f16: activation must remain f16");
    return rms_norm_f16_cuda(
        x.contiguous(), weight, eps, weight_offset);
}

mfq_tensor_backend::Tensor Gemma4Block::forward_ffn(
        CudaExecutionContext& execution,
        mfq_tensor_backend::Tensor residual,
        mfq_tensor_backend::Tensor oo,
        int64_t B,
        int64_t T,
        int64_t H) {
    auto& profiler = execution.profiler;
    mfq_tensor_backend::Tensor x;
    trace_gemma_stage(execution, layer, "attention_output", oo);
    const bool fused_norms = gemma4_moe &&
        execution.config.gemma4_fused_norms &&
        execution.gemma_stage_trace == nullptr &&
        layer_scale.defined();
    mfq_tensor_backend::Tensor dense_input;
    mfq_tensor_backend::Tensor router_input;
    mfq_tensor_backend::Tensor moe_input;
    if (fused_norms) {
        auto prepared = profiler.measure("gemma.attn_residual_pre_norms", [&]() {
            return gemma4_attn_residual_pre_norms_f16_cuda(
                residual.reshape({B * T, H}), oo.reshape({B * T, H}),
                attn_post_norm, ffn_norm, gemma_router_norm_scale,
                ffn_pre_norm_2, rms_norm_eps);
        });
        x = prepared[0].reshape({B, T, H});
        residual = x;
        dense_input = prepared[1];
        router_input = prepared[2];
        moe_input = prepared[3];
    } else {
        auto attn_post = profiler.measure("gemma.attn_post_norm", [&]() {
            return gemma_rms_norm_f16(
                oo.reshape({B * T, H}), attn_post_norm,
                rms_norm_eps, norm_weight_offset);
        });
        x = profiler.measure("gemma.attn_residual", [&]() {
            return acc_cuda(residual.reshape({B * T, H}), attn_post).reshape({B, T, H});
        });
        trace_gemma_stage(
            execution, layer, "attention_residual", x);
        residual = x;
        dense_input = profiler.measure("gemma.ffn_pre_norm", [&]() {
            return gemma_rms_norm_f16(
                x.reshape({B * T, H}), ffn_norm,
                rms_norm_eps, norm_weight_offset);
        });
        if (gemma4_moe) {
            router_input = profiler.measure("gemma.router_norm", [&]() {
                return qwen_rms_norm(
                    x.reshape({B * T, H})
                        .to(mfq_tensor_backend::kFloat32),
                    gemma_router_norm_scale,
                    rms_norm_eps, norm_weight_offset);
            });
            moe_input = profiler.measure("gemma.ffn_pre_norm_2", [&]() {
                return gemma_rms_norm_f16(
                    x.reshape({B * T, H}), ffn_pre_norm_2,
                    rms_norm_eps, norm_weight_offset);
            });
        }
    }
    auto dense_output = profiler.measure("gemma.ffn_dense", [&]() {
        return ffn.forward(execution, dense_input)
            .reshape({B * T, H});
    });
    if (!gemma4_moe) {
        auto dense_post = profiler.measure("gemma.ffn_post_norm", [&]() {
            return dense_output.scalar_type() == mfq_tensor_backend::kFloat16
                ? gemma_rms_norm_f16(
                    dense_output, ffn_post_norm,
                    rms_norm_eps, norm_weight_offset)
                : qwen_rms_norm(
                    dense_output.to(mfq_tensor_backend::kFloat32),
                    ffn_post_norm,
                    rms_norm_eps, norm_weight_offset)
                    .to(mfq_tensor_backend::kFloat16)
                    .contiguous();
        });
        auto result = profiler.measure("gemma.ffn_residual", [&]() {
            return acc_cuda(
                residual.reshape({B * T, H}), dense_post)
                .reshape({B, T, H});
        });
        if (layer_scale.defined()) {
            result = profiler.measure("gemma.layer_scale", [&]() {
                return result * layer_scale;
            });
        }
        trace_gemma_stage(
            execution, layer, "layer_output", result);
        return result;
    }
    if (!fused_norms) {
        dense_output = profiler.measure("gemma.ffn_post_norm_1", [&]() {
            return gemma_rms_norm_f16(
                dense_output, ffn_post_norm_1,
                rms_norm_eps, norm_weight_offset);
        });
        trace_gemma_stage(
            execution, layer, "dense_output", dense_output);
    }
    auto router_logits = profiler.measure("gemma.router", [&]() {
        return mfq_tensor_backend::matmul(router_input, gemma_router.transpose(0, 1));
    });
    auto selected = profiler.measure("gemma.topk", [&]() {
        return moe_topk_cuda(
            router_logits.contiguous(), gemma_top_k,
            false, false, false, true, mfq_nullopt, 1e-20, 1.0);
    });
    trace_gemma_stage(
        execution, layer, "route_ids", selected.at(0));
    trace_gemma_stage(
        execution, layer, "route_weights_before_scale", selected.at(1));
    profiler.measure("gemma.route_scale", [&]() {
        return moe_apply_expert_scale_cuda(
            selected.at(1), selected.at(0), gemma_expert_scale);
    });
    trace_gemma_stage(
        execution, layer, "route_weights", selected.at(1));
    auto route = profiler.measure("gemma.route_map", [&]() {
        return build_moe_route_plan(selected.at(0), gemma_moe_gate_up.n_experts);
    });
    const bool projection_bundle_prefetched =
        prefetch_cached_moe_projection_bundle(
            gemma_moe_gate_up, gemma_moe_down, route);
    mfq_tensor_backend::Tensor down_pair;
    const bool tracing_layer =
        execution.gemma_stage_trace != nullptr &&
        layer == execution.gemma_trace_layer;
    if (!tracing_layer &&
            gemma_moe_gate_up
                .supports_projection_glu_epilogue()) {
        auto moe_hidden = profiler.measure("gemma.moe_gate_up_geglu", [&]() {
            return gemma_moe_gate_up.forward_glu_output(
                execution, moe_input, route, true);
        });
        if (!projection_bundle_prefetched) {
            gemma_moe_down.prefetch(route);
        }
        down_pair = profiler.measure("gemma.moe_down", [&]() {
            return gemma_moe_down.forward(
                execution, moe_hidden, route);
        });
    } else {
        auto gate_up_pair = profiler.measure("gemma.moe_gate_up", [&]() {
            return gemma_moe_gate_up.forward(
                execution, moe_input, route);
        });
        if (!projection_bundle_prefetched) {
            gemma_moe_down.prefetch(route);
        }
        trace_gemma_stage(
            execution, layer, "moe_gate_up", gate_up_pair);
        if (tracing_layer || gate_up_pair.size(0) > 4) {
            auto moe_hidden = profiler.measure("gemma.moe_geglu", [&]() {
                return moe_geglu_split_cuda(gate_up_pair);
            });
            if (tracing_layer) {
                trace_gemma_stage(
                    execution, layer, "moe_hidden", moe_hidden);
            }
            down_pair = profiler.measure("gemma.moe_down", [&]() {
                return gemma_moe_down.forward(
                    execution, moe_hidden, route);
            });
        } else {
            down_pair = profiler.measure("gemma.moe_geglu_down", [&]() {
                return gemma_moe_down.forward_geglu(
                    execution, gate_up_pair, route);
            });
        }
    }
    trace_gemma_stage(execution, layer, "moe_down", down_pair);
    auto moe_output = profiler.measure("gemma.moe_reduce", [&]() {
        return moe_weighted_reduce_cuda(down_pair, selected.at(1));
    });
    trace_gemma_stage(execution, layer, "moe_reduce", moe_output);
    if (fused_norms) {
        auto result = profiler.measure("gemma.ffn_merge", [&]() {
            return gemma4_ffn_merge_f16_cuda(
                dense_output, moe_output, residual.reshape({B * T, H}),
                ffn_post_norm_1, ffn_post_norm_2, ffn_post_norm,
                layer_scale, rms_norm_eps).reshape({B, T, H});
        });
        return result;
    }
    moe_output = profiler.measure("gemma.ffn_post_norm_2", [&]() {
        return gemma_rms_norm_f16(
            moe_output, ffn_post_norm_2,
            rms_norm_eps, norm_weight_offset);
    });
    auto combined = profiler.measure("gemma.ffn_combine", [&]() {
        return dense_output + moe_output;
    });
    auto post = profiler.measure("gemma.ffn_post_norm", [&]() {
        return gemma_rms_norm_f16(
            combined, ffn_post_norm,
            rms_norm_eps, norm_weight_offset);
    });
    auto result = profiler.measure("gemma.ffn_residual", [&]() {
        return acc_cuda(residual.reshape({B * T, H}), post).reshape({B, T, H});
    });
    if (layer_scale.defined()) {
        result = profiler.measure("gemma.layer_scale", [&]() {
            return result * layer_scale;
        });
    }
    trace_gemma_stage(
        execution, layer, "layer_output", result);
    return result;
}

std::unique_ptr<::Block> load_block(
        CudaExecutionContext& execution,
        const mfq::ModelSource& mfq,
        const Config& config,
        int i,
        const std::string& type) {
        const auto& c = config;
        if (type != "full_attention" && type != "sliding_attention") {
            throw std::runtime_error("unsupported Gemma4 layer type: " + type);
        }
        const std::string lp =
            "model.block." + std::to_string(i) + ".";
        const std::string ap = lp + "attention.";
        auto b = std::make_unique<Gemma4Block>();
        b->layer = i;
        b->gemma4_moe = c.num_experts > 0;
        b->max_position_embeddings = c.max_position_embeddings;
        b->rms_norm_eps = c.rms_norm_eps;
        b->norm_weight_offset = 0.0;
        b->sliding = type == "sliding_attention";
        b->value_equals_key =
            !b->sliding && config.attention_key_equals_value;
        b->attention_heads = c.num_attention_heads;
        b->kv_heads = b->sliding
            ? c.num_key_value_heads : config.num_global_key_value_heads;
        b->attention_head_dim = b->sliding
            ? c.head_dim : config.global_head_dim;
        b->attention_rotary_dim = b->attention_head_dim;
        b->attention_window = b->sliding ? config.sliding_window : 0;
        b->attention_scale = 1.0;
        b->attention_rope = RopeCache(
            c.max_position_embeddings, b->attention_rotary_dim,
            b->sliding ? config.sliding_rope_base : c.rope_base,
            b->attention_head_dim,
            b->sliding ? -1 : (int64_t)std::llround(
                c.full_rotary_factor * (double)b->attention_head_dim / 2.0));

        b->attn_norm = load_dense_gpu(execution, mfq, ap + "norm.weight");
        b->attn_post_norm = load_dense_gpu(execution,
            mfq, ap + "output_norm.weight");
        b->q_norm = load_dense_gpu(execution, mfq, ap + "query_norm.weight");
        b->k_norm = load_dense_gpu(execution, mfq, ap + "key_norm.weight");
        b->v_norm = mfq_tensor_backend::ones(
            {b->attention_head_dim},
            mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA).dtype(mfq_tensor_backend::kFloat32));
        std::vector<std::string> projections = {
            ap + "query.weight", ap + "key.weight"};
        if (!b->value_equals_key) projections.push_back(ap + "value.weight");
        b->qkv = load_quant_group(execution, mfq, projections, 2);
        b->o = load_quant_linear(execution, mfq, ap + "output.weight");

        b->ffn_norm = load_dense_gpu(execution,
            mfq, lp + "mlp.dense.input_norm.weight");
        b->ffn_post_norm = load_dense_gpu(execution,
            mfq, lp + "mlp.output_norm.weight");
        b->layer_scale = load_dense_gpu(execution, mfq, lp + "output_scale")
            .to(mfq_tensor_backend::kFloat16).contiguous();
        if (b->gemma4_moe) {
            b->ffn_post_norm_1 = load_dense_gpu(execution,
                mfq, lp + "mlp.dense.output_norm.weight");
            b->ffn_pre_norm_2 = load_dense_gpu(execution,
                mfq, lp + "mlp.experts.input_norm.weight");
            b->ffn_post_norm_2 = load_dense_gpu(execution,
                mfq, lp + "mlp.experts.output_norm.weight");
        }

        const std::string mp = lp + "mlp.";
        b->ffn.geglu = true;
        b->ffn.down = load_quant_linear(execution, mfq, mp + "down.weight");
        b->ffn.gate_up = load_paired_gate_up(execution, mfq, {
            mp + "gate.weight", mp + "up.weight"},
            b->ffn.down);
        prepare_ffn_workspaces(execution, b->ffn);

        if (!b->gemma4_moe) return b;

        b->gemma_moe_gate_up = load_mfe_gpu(execution,
            mfq, mp + "experts.gate_up.weight",
            true, i, "gate_up");
        b->gemma_moe_down = load_mfe_gpu(execution,
            mfq, mp + "experts.down.weight",
            true, i, "down");
        b->gemma_router = load_dense_gpu(execution,
            mfq, mp + "router.weight").to(mfq_tensor_backend::kFloat32).contiguous();
        b->gemma_router_norm_scale = (
            load_dense_gpu(execution, mfq, mp + "router.norm.weight").to(mfq_tensor_backend::kFloat32) /
            std::sqrt((double)c.hidden_size)).contiguous();
        b->gemma_expert_scale = load_dense_gpu(execution,
            mfq, mp + "router.expert_scale").to(mfq_tensor_backend::kFloat32).contiguous();
        b->gemma_top_k = static_cast<int>(c.num_experts_per_tok);

        if (b->gemma_moe_gate_up.n_experts != c.num_experts ||
            b->gemma_moe_down.n_experts != c.num_experts ||
            b->gemma_moe_gate_up.neuron_len != c.hidden_size ||
            b->gemma_moe_gate_up.out_per_expert != 2 * c.moe_intermediate_size ||
            b->gemma_moe_down.neuron_len != c.moe_intermediate_size ||
            b->gemma_moe_down.out_per_expert != c.hidden_size ||
            b->gemma_router.dim() != 2 ||
            b->gemma_router.size(0) != c.num_experts ||
            b->gemma_router.size(1) != c.hidden_size ||
            b->gemma_router_norm_scale.numel() != c.hidden_size ||
            b->gemma_expert_scale.numel() != c.num_experts) {
            throw std::runtime_error(
                "Gemma4 MoE tensor shapes disagree with config at layer " +
                std::to_string(i));
        }
        return b;
}


} // namespace mfq::cuda::gemma4

namespace mfq::cuda {

void Gemma4Model::adapter_load_config(
        std::string_view payload,
        const mfq::ModelGraph&,
        const mfq::ModelSource&) {
    config = gemma4::Config::from_json(payload);
    auto bits = std::bit_cast<std::uint32_t>(static_cast<float>(
        std::sqrt(static_cast<double>(config.hidden_size))));
    bits += 0x7fffU + ((bits >> 16U) & 1U);
    embed_scale = std::bit_cast<float>(bits & 0xffff0000U);
    metadata.vocab_size = config.vocab_size;
    metadata.hidden_size = config.hidden_size;
    metadata.num_hidden_layers = config.num_hidden_layers;
    metadata.num_attention_heads = config.num_attention_heads;
    metadata.num_key_value_heads = config.num_key_value_heads;
    metadata.head_dim = config.head_dim;
    metadata.max_position_embeddings = config.max_position_embeddings;
    metadata.rotary_dim = config.rotary_dim;
    metadata.num_experts = config.num_experts;
    metadata.rope_base = config.rope_base;
    metadata.rms_norm_eps = config.rms_norm_eps;
    metadata.final_logit_softcapping = config.final_logit_softcapping;
    metadata.embedding_scale = embed_scale;
    metadata.tie_word_embeddings = config.tie_word_embeddings;
    metadata.gemma4 = true;
    metadata.decode_graph_double_warmup = true;
    metadata.model_type = config.model_type;
    metadata.layer_types = config.layer_types;
}

std::unique_ptr<Block>
Gemma4Model::adapter_load_block(
        const mfq::ModelSource& source,
        int layer,
        int,
        const std::string& type) {
    return gemma4::load_block(
        *execution, source, config, layer, type);
}

mfq_tensor_backend::Tensor
Gemma4Model::adapter_prepare_hidden(
        mfq_tensor_backend::Tensor hidden,
        int64_t,
        int64_t) const {
    return execution->profiler.measure("model.embed_scale", [&]() {
        return hidden * embed_scale;
    });
}

} // namespace mfq::cuda

namespace mfq::cuda {

template struct CudaSessionCodec<Gemma4Model>;
template struct CausalLm<Gemma4Model>;

} // namespace mfq::cuda
