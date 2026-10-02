#include "storage/weight_loader.h"
#include "ops.h"
#include "../session_codec_impl.h"
#include "models/common/transformer_layer.h"
#include "models/gemma4/causal_lm.h"

#include "models/full_block.h"

#include "models/common/gated_mlp.h"
#include <array>
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

    mfq_tensor_backend::Tensor forward_ffn(CudaExecutionContext &execution,
                                           mfq_tensor_backend::Tensor residual,
                                           mfq_tensor_backend::Tensor attention_output,
                                           int64_t batch, int64_t tokens, int64_t hidden) override;
};

static mfq_tensor_backend::Tensor gemma_rms_norm_f16(mfq_tensor_backend::Tensor x,
                                                     mfq_tensor_backend::Tensor weight, double eps,
                                                     double weight_offset) {
    MFQ_RUNTIME_CHECK(x.scalar_type() == mfq_tensor_backend::kFloat16,
                      "gemma_rms_norm_f16: activation must remain f16");
    return rms_norm_f16_cuda(x.contiguous(), weight, eps, weight_offset);
}

mfq_tensor_backend::Tensor Gemma4Block::forward_ffn(CudaExecutionContext &execution,
                                                    mfq_tensor_backend::Tensor residual,
                                                    mfq_tensor_backend::Tensor oo, int64_t B,
                                                    int64_t T, int64_t H) {
    auto &profiler = execution.profiler;
    trace_gemma_stage(execution, layer, "attention_output", oo);
    const bool fused_norms = gemma4_moe && execution.config.gemma4_fused_norms &&
                             execution.gemma_stage_trace == nullptr && layer_scale.defined();
    using Tensor = mfq_tensor_backend::Tensor;
    using Inputs = mfq::models::gemma4::FfnInputs<Tensor>;
    auto inputs = mfq::models::gemma4::prepare_ffn(
        residual, oo, gemma4_moe,
        [&](Tensor residual, Tensor oo) -> std::optional<Inputs> {
            if (!fused_norms)
                return {};
            auto prepared = profiler.measure("gemma.attn_residual_pre_norms", [&]() {
                return gemma4_attn_residual_pre_norms_f16_cuda(
                    residual.reshape({B * T, H}), oo.reshape({B * T, H}), attn_post_norm, ffn_norm,
                    gemma_router_norm_scale, ffn_pre_norm_2, rms_norm_eps);
            });
            return Inputs{prepared[0].reshape({B, T, H}), prepared[1], prepared[2], prepared[3]};
        },
        [&](Tensor oo) {
            return profiler.measure("gemma.attn_post_norm", [&]() {
                return gemma_rms_norm_f16(oo.reshape({B * T, H}), attn_post_norm, rms_norm_eps,
                                          norm_weight_offset);
            });
        },
        [&](Tensor residual, Tensor attn_post) {
            auto x = profiler.measure("gemma.attn_residual", [&] {
                return acc_cuda(residual.reshape({B * T, H}), attn_post).reshape({B, T, H});
            });
            trace_gemma_stage(execution, layer, "attention_residual", x);
            return x;
        },
        [&](Tensor x) {
            return profiler.measure("gemma.ffn_pre_norm", [&] {
                return gemma_rms_norm_f16(x.reshape({B * T, H}), ffn_norm, rms_norm_eps,
                                          norm_weight_offset);
            });
        },
        [&](Tensor x) {
            return profiler.measure("gemma.router_norm", [&]() {
                return qwen_rms_norm(x.reshape({B * T, H}).to(mfq_tensor_backend::kFloat32),
                                     gemma_router_norm_scale, rms_norm_eps, norm_weight_offset);
            });
        },
        [&](Tensor x) {
            return profiler.measure("gemma.ffn_pre_norm_2", [&]() {
                return gemma_rms_norm_f16(x.reshape({B * T, H}), ffn_pre_norm_2, rms_norm_eps,
                                          norm_weight_offset);
            });
        });
    residual = inputs.residual;
    auto dense_input = inputs.dense, router_input = inputs.router, moe_input = inputs.experts;
    auto result = mfq::models::gemma4::feed_forward(
        gemma4_moe, layer_scale.defined(), fused_norms, residual,
        [&] {
            return profiler.measure("gemma.ffn_dense", [&] {
                return ffn.forward(execution, dense_input).reshape({B * T, H});
            });
        },
        [&] {
            return mfq::models::gemma4::experts(
                [&] {
                    return profiler.measure("gemma.router", [&]() {
                        return mfq_tensor_backend::matmul(router_input,
                                                          gemma_router.transpose(0, 1));
                    });
                },
                [&](Tensor router_logits) {
                    return profiler.measure("gemma.topk", [&]() {
                        return moe_topk_cuda(router_logits.contiguous(), gemma_top_k, false, false,
                                             false, true, mfq_nullopt, 1e-20, 1.0);
                    });
                },
                [&](auto &selected) {
                    trace_gemma_stage(execution, layer, "route_ids", selected.at(0));
                    trace_gemma_stage(execution, layer, "route_weights_before_scale",
                                      selected.at(1));
                    profiler.measure("gemma.route_scale", [&]() {
                        return moe_apply_expert_scale_cuda(selected.at(1), selected.at(0),
                                                           gemma_expert_scale);
                    });
                    trace_gemma_stage(execution, layer, "route_weights", selected.at(1));
                },
                [&](const auto &selected) {
                    auto route = profiler.measure("gemma.route_map", [&]() {
                        return build_moe_route_plan(selected.at(0), gemma_moe_gate_up.n_experts);
                    });
                    const bool projection_bundle_prefetched = prefetch_cached_moe_projection_bundle(
                        gemma_moe_gate_up, gemma_moe_down, route);
                    const bool tracing = execution.gemma_stage_trace != nullptr &&
                                         layer == execution.gemma_trace_layer;
                    Tensor packed;
                    auto prefetch_down = [&] {
                        if (!projection_bundle_prefetched)
                            gemma_moe_down.prefetch(route);
                    };
                    auto down_pair = mfq::models::gated_mlp(
                        moe_input, true, 0.0,
                        [](const auto &...) { return std::optional<Tensor>{}; },
                        [&](const Tensor &input, auto, double) -> std::optional<Tensor> {
                            if (tracing || !gemma_moe_gate_up.supports_projection_glu_epilogue())
                                return {};
                            auto hidden = profiler.measure("gemma.moe_gate_up_geglu", [&] {
                                return gemma_moe_gate_up.forward_glu_output(execution, input, route,
                                                                            true);
                            });
                            prefetch_down();
                            return hidden;
                        },
                        [&](Tensor input) {
                            packed = profiler.measure("gemma.moe_gate_up", [&] {
                                return gemma_moe_gate_up.forward(execution, input, route);
                            });
                            prefetch_down();
                            trace_gemma_stage(execution, layer, "moe_gate_up", packed);
                            const auto width = packed.size(-1) / 2;
                            return std::array<Tensor, 2>{packed.narrow(-1, 0, width),
                                                         packed.narrow(-1, width, width)};
                        },
                        [&](const Tensor &, const Tensor &, auto, double) {
                            auto hidden = profiler.measure(
                                "gemma.moe_geglu", [&] { return moe_geglu_split_cuda(packed); });
                            if (tracing)
                                trace_gemma_stage(execution, layer, "moe_hidden", hidden);
                            return hidden;
                        },
                        [&](Tensor hidden) {
                            return profiler.measure("gemma.moe_down", [&] {
                                return gemma_moe_down.forward(execution, hidden, route);
                            });
                        },
                        [&](const Tensor &, const Tensor &, auto, double) -> std::optional<Tensor> {
                            if (tracing || packed.size(0) > 4)
                                return {};
                            return profiler.measure("gemma.moe_geglu_down", [&] {
                                return gemma_moe_down.forward_geglu(execution, packed, route);
                            });
                        });
                    trace_gemma_stage(execution, layer, "moe_down", down_pair);
                    return down_pair;
                },
                [&](Tensor down_pair, const auto &selected) {
                    auto moe_output = profiler.measure("gemma.moe_reduce", [&]() {
                        return moe_weighted_reduce_cuda(down_pair, selected.at(1));
                    });
                    trace_gemma_stage(execution, layer, "moe_reduce", moe_output);
                    return moe_output;
                });
        },
        [&](Tensor value, mfq::models::gemma4::FfnNorm role) {
            using Norm = mfq::models::gemma4::FfnNorm;
            const auto &weight = role == Norm::dense     ? ffn_post_norm_1
                                 : role == Norm::experts ? ffn_post_norm_2
                                                         : ffn_post_norm;
            const char *name = role == Norm::dense     ? "gemma.ffn_post_norm_1"
                               : role == Norm::experts ? "gemma.ffn_post_norm_2"
                                                       : "gemma.ffn_post_norm";
            auto normalized = profiler.measure(name, [&] {
                return value.scalar_type() == mfq_tensor_backend::kFloat16
                           ? gemma_rms_norm_f16(value, weight, rms_norm_eps, norm_weight_offset)
                           : qwen_rms_norm(value.to(mfq_tensor_backend::kFloat32), weight,
                                           rms_norm_eps, norm_weight_offset)
                                 .to(mfq_tensor_backend::kFloat16)
                                 .contiguous();
            });
            if (role == Norm::dense)
                trace_gemma_stage(execution, layer, "dense_output", normalized);
            return normalized;
        },
        [&](Tensor dense, Tensor routed) {
            return profiler.measure("gemma.ffn_combine", [&] { return dense + routed; });
        },
        [&](Tensor skip, Tensor value) {
            return profiler.measure("gemma.ffn_residual", [&] {
                return acc_cuda(skip.reshape({B * T, H}), value).reshape({B, T, H});
            });
        },
        [&](Tensor value) {
            return profiler.measure("gemma.layer_scale", [&] { return value * layer_scale; });
        },
        [&](Tensor dense, Tensor routed, Tensor skip) {
            return profiler.measure("gemma.ffn_merge", [&] {
                return gemma4_ffn_merge_f16_cuda(dense, routed, skip.reshape({B * T, H}),
                                                 ffn_post_norm_1, ffn_post_norm_2, ffn_post_norm,
                                                 layer_scale, rms_norm_eps)
                    .reshape({B, T, H});
            });
        });
    trace_gemma_stage(execution, layer, "layer_output", result);
    return result;
}

struct BlockLoader {
    CudaExecutionContext &execution;
    const mfq::ModelSource &source;
    auto dense(const std::string &name) { return load_dense_gpu(execution, source, name); }
    auto linear(const std::string &name) { return load_quant_linear(execution, source, name); }
    auto projections(const std::vector<std::string> &names) {
        return load_quant_group(execution, source, names, 2);
    }
    auto rope(int64_t capacity, const mfq::models::gemma4::LayerSpec &spec) {
        return RopeCache(capacity, spec.head_dim, spec.rope_base, spec.head_dim, spec.rotary_pairs);
    }
    auto ones(int64_t width) {
        return mfq_tensor_backend::ones({width}, mfq_tensor_backend::TensorOptions()
                                                     .device(mfq_tensor_backend::kCUDA)
                                                     .dtype(mfq_tensor_backend::kFloat32));
    }
    auto layer_scale(const std::string &name) {
        return dense(name).to(mfq_tensor_backend::kFloat16).contiguous();
    }
    auto router_parameter(const std::string &name) {
        return dense(name).to(mfq_tensor_backend::kFloat32).contiguous();
    }
    auto normalized_router_parameter(const std::string &name, double divisor) {
        return (router_parameter(name) / divisor).contiguous();
    }
    auto gate_up(const std::vector<std::string> &names, const QuantLinear &down) {
        return load_paired_gate_up(execution, source, names, down);
    }
    void workspace(FFN &ffn) { prepare_ffn_workspaces(execution, ffn); }
    auto experts(const std::string &name, int layer, const char *projection) {
        return load_mfe_gpu(execution, source, name, true, layer, projection);
    }
    static auto expert_shape(const MfeWeight &weight) {
        return std::array<int64_t, 3>{weight.n_experts, weight.out_per_expert, weight.neuron_len};
    }
    static auto shape(const mfq_tensor_backend::Tensor &t) { return t.sizes().vec(); }
    static auto elements(const mfq_tensor_backend::Tensor &t) { return t.numel(); }
};

std::unique_ptr<::Block> load_block(CudaExecutionContext &execution, const mfq::ModelSource &source,
                                    const Config &config, int layer, const std::string &type) {
    auto block = std::make_unique<Gemma4Block>();
    BlockLoader loader{execution, source};
    mfq::models::gemma4::load_block(*block, loader, config, layer, type);
    return block;
}

} // namespace mfq::cuda::gemma4

namespace mfq::cuda {

std::unique_ptr<Block> Gemma4Model::adapter_load_block(const mfq::ModelSource &source, int layer,
                                                       int, const std::string &type) {
    return gemma4::load_block(*execution, source, config, layer, type);
}

} // namespace mfq::cuda

namespace mfq::cuda {

template struct CudaSessionCodec<Gemma4Model>;

} // namespace mfq::cuda

namespace mfq::models {
template struct gemma4::CausalLm<cuda::CudaCausalOps<cuda::Gemma4Model>>;
} // namespace mfq::models
