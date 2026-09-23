#pragma once

#include "quant_linear.h"
#include "models/model_config.h"
#include "cuda_execution.h"
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
            mfq_tensor_backend::Device device) {
        if (configured_sections.empty()) return;
        auto execution_sections = configured_sections;
        if (interleaved) {
            std::array<std::vector<int64_t>, 3> pairs;
            for (int64_t pair = 0; pair < configured_rotary_dim / 2; ++pair) {
                const int64_t residue = pair % 3;
                const int axis = residue == 1 &&
                        pair < configured_sections[1] * 3
                    ? 1
                    : (residue == 2 && pair < configured_sections[2] * 3
                        ? 2 : 0);
                pairs[static_cast<size_t>(axis)].push_back(pair);
            }
            execution_sections = {
                static_cast<int64_t>(pairs[0].size()),
                static_cast<int64_t>(pairs[1].size()),
                static_cast<int64_t>(pairs[2].size())};
            std::vector<int64_t> order;
            for (const auto& axis : pairs) {
                order.insert(order.end(), axis.begin(), axis.end());
            }
            std::vector<int64_t> inverse(order.size());
            for (size_t index = 0; index < order.size(); ++index) {
                inverse[static_cast<size_t>(order[index])] =
                    static_cast<int64_t>(index);
            }
            interleaved_order = mfq_tensor_backend::tensor(
                order, mfq_tensor_backend::TensorOptions()
                    .dtype(mfq_tensor_backend::kInt64)
                    .device(mfq_tensor_backend::kCPU)).to(device);
            interleaved_inverse = mfq_tensor_backend::tensor(
                inverse, mfq_tensor_backend::TensorOptions()
                    .dtype(mfq_tensor_backend::kInt64)
                    .device(mfq_tensor_backend::kCPU)).to(device);
        }
        sections = mfq_tensor_backend::tensor(
            execution_sections,
            mfq_tensor_backend::TensorOptions()
                .dtype(mfq_tensor_backend::kInt64)
                .device(mfq_tensor_backend::kCPU));
    }
    mfq_tensor_backend::Tensor apply(
            mfq_tensor_backend::Tensor x,
            mfq_tensor_backend::Tensor pos,
            bool grid_mrope_positions = false) const {
        if (!x.is_cuda()) {
            MFQ_RUNTIME_CHECK(
                !cos.is_cuda() && !sin.is_cuda(),
                "CPU RoPE requires a CPU-resident table");
            auto positions = pos.contiguous().to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt64)
                .clamp(0, cos.size(0) - 1);
            mfq_tensor_backend::Tensor selected_cos;
            mfq_tensor_backend::Tensor selected_sin;
            if (positions.dim() == 1) {
                selected_cos = cos.index_select(0, positions);
                selected_sin = sin.index_select(0, positions);
            } else if (!grid_mrope_positions && positions.dim() == 2) {
                const int64_t batches = positions.size(0);
                const int64_t tokens = positions.size(1);
                selected_cos = cos.index_select(0, positions.reshape({-1}))
                    .reshape({batches, tokens, rotary_dim / 2})
                    .unsqueeze(1);
                selected_sin = sin.index_select(0, positions.reshape({-1}))
                    .reshape({batches, tokens, rotary_dim / 2})
                    .unsqueeze(1);
            } else if (grid_mrope_positions && sections.numel() == 3) {
                const int64_t * section = sections.data_ptr<int64_t>();
                std::vector<mfq_tensor_backend::Tensor> cos_parts;
                std::vector<mfq_tensor_backend::Tensor> sin_parts;
                int64_t offset = 0;
                for (int axis = 0; axis < 3; ++axis) {
                    const int64_t count = section[axis];
                    auto active = positions.select(0, axis);
                    cos_parts.push_back(
                        cos.index_select(0, active).narrow(1, offset, count));
                    sin_parts.push_back(
                        sin.index_select(0, active).narrow(1, offset, count));
                    offset += count;
                }
                selected_cos = mfq_tensor_backend::cat(cos_parts, 1);
                selected_sin = mfq_tensor_backend::cat(sin_parts, 1);
            } else {
                throw std::runtime_error(
                    "multi-axis CPU RoPE requires three position sections");
            }
            auto xf = x.contiguous().to(mfq_tensor_backend::kFloat32);
            const int64_t half = rotary_dim / 2;
            auto first = xf.narrow(-1, 0, half);
            auto second = xf.narrow(-1, half, half);
            auto output = xf.clone();
            output.narrow(-1, 0, half).copy_(
                first * selected_cos - second * selected_sin);
            output.narrow(-1, half, half).copy_(
                second * selected_cos + first * selected_sin);
            return output;
        }
        auto source = x.contiguous().to(mfq_tensor_backend::kFloat32);
        auto active_cos = cos;
        auto active_sin = sin;
        const bool interleaved = grid_mrope_positions &&
            interleaved_order.defined();
        if (interleaved) {
            const int64_t half = rotary_dim / 2;
            std::vector<mfq_tensor_backend::Tensor> parts{
                source.narrow(-1, 0, half).index_select(-1, interleaved_order),
                source.narrow(-1, half, half).index_select(-1, interleaved_order)};
            if (source.size(-1) > rotary_dim) {
                parts.push_back(source.narrow(
                    -1, rotary_dim, source.size(-1) - rotary_dim));
            }
            source = mfq_tensor_backend::cat(parts, -1).contiguous();
            active_cos = cos.index_select(1, interleaved_order).contiguous();
            active_sin = sin.index_select(1, interleaved_order).contiguous();
        }
        auto output = rope_table_cuda(
            source,
            pos.contiguous().to(
                mfq_tensor_backend::kCUDA, mfq_tensor_backend::kInt64),
            active_cos, active_sin, rotary_dim,
            grid_mrope_positions ? sections : empty_sections);
        if (interleaved) {
            const int64_t half = rotary_dim / 2;
            std::vector<mfq_tensor_backend::Tensor> parts{
                output.narrow(-1, 0, half).index_select(
                    -1, interleaved_inverse),
                output.narrow(-1, half, half).index_select(
                    -1, interleaved_inverse)};
            if (output.size(-1) > rotary_dim) {
                parts.push_back(output.narrow(
                    -1, rotary_dim, output.size(-1) - rotary_dim));
            }
            output = mfq_tensor_backend::cat(parts, -1).contiguous();
        }
        return output;
    }

    mfq_tensor_backend::Tensor apply_bf16(mfq_tensor_backend::Tensor x, mfq_tensor_backend::Tensor pos) const {
        MFQ_RUNTIME_CHECK(
            sections.numel() == 0,
            "Qwen3 BF16 RoPE does not support multi-axis sections");
        const char * fused_env =
            std::getenv("MFQ_MINICPM_FUSED_BF16_ROPE");
        if (x.is_cuda() && pos.dim() == 1 &&
                (fused_env == nullptr || fused_env[0] != '0')) {
            return rope_table_bf16_cuda(
                x.contiguous().to(mfq_tensor_backend::kBFloat16),
                pos.contiguous().to(cos.device(), mfq_tensor_backend::kInt64),
                cos, sin, rotary_dim);
        }
        auto positions = pos.contiguous().to(cos.device(), mfq_tensor_backend::kInt64)
            .clamp(0, cos.size(0) - 1);
        mfq_tensor_backend::Tensor selected_cos;
        mfq_tensor_backend::Tensor selected_sin;
        if (positions.dim() == 1) {
            selected_cos = cos.index_select(0, positions)
                .to(mfq_tensor_backend::kBFloat16).unsqueeze(0).unsqueeze(0);
            selected_sin = sin.index_select(0, positions)
                .to(mfq_tensor_backend::kBFloat16).unsqueeze(0).unsqueeze(0);
        } else if (positions.dim() == 2) {
            const int64_t batches = positions.size(0);
            const int64_t tokens = positions.size(1);
            selected_cos = cos.index_select(0, positions.reshape({-1}))
                .reshape({batches, tokens, rotary_dim / 2})
                .to(mfq_tensor_backend::kBFloat16).unsqueeze(1);
            selected_sin = sin.index_select(0, positions.reshape({-1}))
                .reshape({batches, tokens, rotary_dim / 2})
                .to(mfq_tensor_backend::kBFloat16).unsqueeze(1);
        } else {
            throw std::runtime_error(
                "Qwen3 BF16 RoPE positions must be rank one or two");
        }
        auto xb = x.contiguous().to(mfq_tensor_backend::kBFloat16);
        const int64_t half = rotary_dim / 2;
        auto first = xb.narrow(-1, 0, half);
        auto second = xb.narrow(-1, half, half);
        auto output = xb.clone();
        output.narrow(-1, 0, half).copy_(
            first * selected_cos - second * selected_sin);
        output.narrow(-1, half, half).copy_(
            second * selected_cos + first * selected_sin);
        return output.contiguous();
    }
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

    bool uses_moe_expert_cache() const {
        if (!is_moe) return false;
        if (moe_split_gate_up) {
            return moe_gate.cached_source ||
                moe_up.cached_source ||
                moe_down.cached_source;
        }
        return moe_gate_up.cached_source || moe_down.cached_source;
    }

    bool tensor_parallel_dense_compatible() const {
        if (is_moe ||
            gate_up.layers.size() != 2 ||
            !down.tensor_parallel() ||
            down.tensor_parallel_axis !=
                TensorParallelAxis::Input) {
            return false;
        }
        const auto & gate = gate_up.layers[0];
        const auto & up = gate_up.layers[1];
        if (!gate.tensor_parallel() ||
            !up.tensor_parallel() ||
            gate.tensor_parallel_axis !=
                TensorParallelAxis::Output ||
            up.tensor_parallel_axis !=
                TensorParallelAxis::Output ||
            gate.tensor_parallel_shards.size() !=
                up.tensor_parallel_shards.size() ||
            gate.tensor_parallel_shards.size() !=
                down.tensor_parallel_shards.size()) {
            return false;
        }
        for (size_t index = 0;
             index < gate.tensor_parallel_shards.size();
             ++index) {
            const auto & gate_shard =
                gate.tensor_parallel_shards[index];
            const auto & up_shard =
                up.tensor_parallel_shards[index];
            const auto & down_shard =
                down.tensor_parallel_shards[index];
            if (gate_shard.device != up_shard.device ||
                gate_shard.device != down_shard.device ||
                gate_shard.output_begin !=
                    up_shard.output_begin ||
                gate_shard.output_end !=
                    up_shard.output_end ||
                gate_shard.output_begin !=
                    down_shard.input_begin ||
                gate_shard.output_end !=
                    down_shard.input_end) {
                return false;
            }
        }
        return true;
    }

    mfq_tensor_backend::Tensor forward_tensor_parallel_dense(
            mfq_tensor_backend::Tensor xh) const {
        auto shape = xh.sizes().vec();
        auto flat = xh.reshape(
            {-1, xh.size(-1)});
        const size_t shard_count = down.tensor_parallel_shards.size();
        std::vector<mfq_tensor_backend::Tensor> partials(shard_count);
        for (size_t launch_position = 0;
             launch_position < shard_count;
             ++launch_position) {
            const size_t index = model_parallel_launch_index(
                launch_position, shard_count);
            const auto & gate_shard =
                gate_up.layers[0]
                    .tensor_parallel_shards[index];
            const auto & up_shard =
                gate_up.layers[1]
                    .tensor_parallel_shards[index];
            const auto & down_shard =
                down.tensor_parallel_shards[index];
            MfqCudaGuard guard(
                gate_shard.device);
            auto local_x =
                tensor_to_cuda_device(
                    flat, gate_shard.device);
            auto gate_output =
                run_quant_linear_shard(
                    gate_shard, local_x);
            auto up_output =
                run_quant_linear_shard(
                    up_shard, local_x);
            mfq_tensor_backend::Tensor activation;
            if (geglu) {
                activation = gelu_mul_cuda(
                    gate_output.contiguous(),
                    up_output.contiguous());
            } else if (swiglu_limit > 0.0) {
                auto clipped_gate =
                    mfq_tensor_backend::clamp_max(
                        gate_output.to(
                            mfq_tensor_backend::kFloat32),
                        swiglu_limit);
                auto clipped_up = mfq_tensor_backend::clamp(
                    up_output.to(
                            mfq_tensor_backend::kFloat32),
                        -swiglu_limit,
                        swiglu_limit);
                activation =
                    (mfq_tensor_backend::silu(clipped_gate) *
                     clipped_up)
                        .to(mfq_tensor_backend::kFloat16)
                        .contiguous();
            } else {
                activation =
                    (mfq_tensor_backend::silu(gate_output) *
                     up_output).contiguous();
            }
            partials[index] = run_quant_linear_shard(
                down_shard, activation);
        }
        auto output =
            reduce_model_parallel_outputs(
                std::move(partials));
        shape.back() = output.size(-1);
        return output.reshape(shape);
    }

    bool expert_parallel_moe_compatible() const {
        if (!is_moe || moe_split_gate_up ||
                moe_gate_up.expert_parallel_shards.size() !=
                    moe_down.expert_parallel_shards.size() ||
                moe_gate_up.expert_parallel_shards.empty()) {
            return false;
        }
        for (size_t index = 0;
             index < moe_gate_up.expert_parallel_shards.size(); ++index) {
            const auto & gate = moe_gate_up.expert_parallel_shards[index];
            const auto & down = moe_down.expert_parallel_shards[index];
            if (!gate.weight || !down.weight ||
                    gate.device != down.device ||
                    gate.expert_begin != down.expert_begin ||
                    gate.expert_end != down.expert_end) {
                return false;
            }
        }
        return true;
    }

    mfq_tensor_backend::Tensor forward_expert_parallel_moe(
            mfq_tensor_backend::Tensor x,
            const MoeRoutePlan & route,
            mfq_tensor_backend::Tensor route_weights) const {
        const size_t shard_count =
            moe_gate_up.expert_parallel_shards.size();
        std::vector<mfq_tensor_backend::Tensor> routed_partials(
            shard_count);
        std::vector<mfq_tensor_backend::Tensor> down_partials;
        const bool collect_output_energy =
            moe_route_stats_path() != nullptr &&
            moe_route_output_energy_enabled();
        if (collect_output_energy) {
            down_partials.resize(shard_count);
        }
        static const bool disable_swiglu_quant_fusion = [] {
            const char * value = std::getenv(
                "MFQ_DISABLE_MOE_SWIGLU_QUANT_FUSION");
            return value != nullptr && std::atoi(value) != 0;
        }();
        for (size_t launch_position = 0;
             launch_position < shard_count; ++launch_position) {
            const size_t index = model_parallel_launch_index(
                launch_position, shard_count);
            const auto & gate_shard =
                moe_gate_up.expert_parallel_shards[index];
            const auto & down_shard =
                moe_down.expert_parallel_shards[index];
            MfqCudaGuard guard(gate_shard.device);
            auto local_x = tensor_to_cuda_device(x, gate_shard.device);
            auto local_weights = tensor_to_cuda_device(
                route_weights, gate_shard.device);
            auto local_route = moe_route_to_device(
                route, gate_shard.device);
            mfq_tensor_backend::Tensor gate_up_pair;
            mfq_tensor_backend::Tensor projected_hidden;
            if (swiglu_limit <= 0.0 &&
                    !g_force_moe_materialized_swiglu &&
                    gate_shard.weight
                        ->supports_projection_glu_epilogue()) {
                projected_hidden = gate_shard.weight->forward_glu_output(
                    local_x, local_route, false);
            } else {
                gate_up_pair = gate_shard.weight->forward(
                    local_x, local_route);
            }
            mfq_tensor_backend::Tensor down_pair;
            const bool allow_fusion =
                gate_up_pair.defined() &&
                !g_force_moe_materialized_swiglu &&
                !disable_swiglu_quant_fusion &&
                moe_small_glu_path_enabled(
                    static_cast<int>(gate_up_pair.size(0)));
            if (projected_hidden.defined()) {
                down_pair = down_shard.weight->forward(
                    projected_hidden, local_route);
            } else if (swiglu_limit <= 0.0 && allow_fusion) {
                down_pair = down_shard.weight->forward_swiglu(
                    gate_up_pair, local_route);
            } else if (swiglu_limit > 0.0 && allow_fusion &&
                    down_shard.weight->supports_clamped_swiglu()) {
                down_pair = down_shard.weight->forward_clamped_swiglu(
                    gate_up_pair, local_route, swiglu_limit);
            } else {
                mfq_tensor_backend::Tensor hidden;
                if (swiglu_limit <= 0.0) {
                    hidden = moe_swiglu_split_cuda(gate_up_pair);
                } else {
                    const int64_t width = gate_up_pair.size(-1) / 2;
                    auto gate = mfq_tensor_backend::clamp_max(
                        gate_up_pair.slice(-1, 0, width)
                            .to(mfq_tensor_backend::kFloat32),
                        swiglu_limit);
                    auto up = mfq_tensor_backend::clamp(
                        gate_up_pair.slice(-1, width, 2 * width)
                            .to(mfq_tensor_backend::kFloat32),
                        -swiglu_limit, swiglu_limit);
                    hidden = (mfq_tensor_backend::silu(gate) * up)
                        .to(mfq_tensor_backend::kFloat16).contiguous();
                }
                down_pair = down_shard.weight->forward(
                    hidden, local_route);
            }
            if (collect_output_energy) {
                down_partials[index] = down_pair;
            }
            routed_partials[index] = moe_weighted_reduce_cuda(
                down_pair, local_weights);
        }
        if (moe_route_stats_path() != nullptr) {
            mfq_tensor_backend::Tensor complete_down;
            if (collect_output_energy) {
                complete_down = reduce_model_parallel_outputs(
                    std::move(down_partials));
            }
            record_moe_route_stats(
                moe_layer, route.ids, route_weights,
                complete_down, moe_gate_up.n_experts);
        }
        return reduce_model_parallel_outputs(
            std::move(routed_partials));
    }

    mfq_tensor_backend::Tensor forward_dense_f32_down_kld(
            mfq_tensor_backend::Tensor xh) const {
        MFQ_RUNTIME_CHECK(
            !is_moe && !tensor_parallel_dense_compatible() &&
            !geglu && swiglu_limit <= 0.0,
            "FP32-output IN diagnostic requires a local dense SiLU FFN");
        auto parts = g_profiler.measure(
            "ffn.gate_up", [&]() {
                return gate_up.forward(xh);
            });
        return g_profiler.measure(
            "ffn.down.fp32_output", [&]() {
                return down.forward_input_mul_f32_kld(
                    parts[1], parts[0], 2);
            });
    }

    mfq_tensor_backend::Tensor forward_impl(
        mfq_tensor_backend::Tensor x,
        MfqOptional<mfq_tensor_backend::Tensor> input_ids,
        bool allow_important_neurons) const {
        mfq_tensor_backend::Tensor xh;
        if (x.scalar_type() == mfq_tensor_backend::kFloat16) {
            xh = x;
        } else {
            xh = g_profiler.measure("ffn.input_cast", [&]() { return x.to(mfq_tensor_backend::kFloat16); });
        }
        if (allow_important_neurons && important_neurons) {
            if (is_moe) {
                throw std::runtime_error(
                    "important-neuron branches are only supported for dense FFNs");
            }
            const int64_t rows =
                xh.numel() / xh.size(-1);
            const char * f32_down =
                std::getenv(
                    "MFQ_DIAGNOSTIC_IN_F32_DOWN");
            const bool use_f32_down =
                rows >= 16 &&
                g_kl_mmq_mode == KlMmqMode::Fp16 &&
                f32_down != nullptr &&
                f32_down[0] == '1';
            if (use_f32_down) {
                auto low =
                    forward_dense_f32_down_kld(xh);
                auto high =
                    important_neurons
                        ->forward_dense_f32_down_kld(xh);
                return g_profiler.measure(
                    "ffn.in.combine.fp32", [&]() {
                        return (low + high)
                            .to(mfq_tensor_backend::kFloat16)
                            .contiguous();
                    });
            }
            auto run_branch = [&](size_t index) {
                return index == 0
                    ? forward_impl(xh, input_ids, false)
                    : important_neurons->forward_impl(
                        xh, input_ids, false);
            };
            std::vector<mfq_tensor_backend::Tensor> outputs;
            const char * disable_parallel =
                std::getenv("MFQ_DISABLE_IN_BRANCH_PARALLEL");
            const bool parallel =
                decode_branch_parallel_enabled(rows) &&
                (disable_parallel == nullptr ||
                 disable_parallel[0] != '1') &&
                important_neuron_executor->run(
                    2, run_branch, outputs);
            auto low = parallel ? outputs[0] : run_branch(0);
            auto high = parallel ? outputs[1] : run_branch(1);
            return g_profiler.measure(
                "ffn.in.combine", [&]() {
                    return acc_cuda(
                        low.contiguous(), high.contiguous());
                });
        }
        if (tensor_parallel_dense_compatible()) {
            return g_profiler.measure(
                "ffn.tensor_parallel", [&]() {
                    return forward_tensor_parallel_dense(
                        xh);
                });
        }
        if (is_moe) {
            const int64_t rows = xh.numel() / xh.size(-1);
            if (g_moe_continuous_batch_cache_serial &&
                    uses_moe_expert_cache() && rows > 1) {
                // A bounded expert cache cannot safely admit the union of an
                // arbitrary request batch: a miss in that union otherwise
                // falls back to staging the complete projection. Keep the
                // outer request/KV/attention batch intact and execute only
                // each routed FFN row independently, so cache admission is
                // bounded by one token's top-k experts.
                auto flat = xh.reshape({rows, xh.size(-1)}).contiguous();
                std::vector<mfq_tensor_backend::Tensor> outputs;
                outputs.reserve(static_cast<size_t>(rows));
                for (int64_t row = 0; row < rows; ++row) {
                    auto row_input = flat.narrow(0, row, 1).contiguous();
                    if (input_ids.has_value()) {
                        MFQ_RUNTIME_CHECK(
                            input_ids.value().numel() == rows,
                            "continuous-batch MoE token ids do not match rows");
                        auto row_ids = input_ids.value().reshape({rows})
                            .narrow(0, row, 1).contiguous();
                        outputs.push_back(forward_impl(
                            row_input, row_ids, false));
                    } else {
                        outputs.push_back(forward_impl(
                            row_input, mfq_nullopt, false));
                    }
                }
                return mfq_tensor_backend::cat(outputs, 0)
                    .reshape(xh.sizes());
            }
            if (!shared || moe_top_k <= 0 || !moe_router.defined() ||
                (!moe_shared_ungated && !moe_shared_gate.defined())) {
                throw std::runtime_error("incomplete MoE FFN state");
            }
            auto xf = xh.reshape({-1, xh.size(-1)}).contiguous();
            auto xf32 = g_profiler.measure("moe.input_f32", [&]() {
                return xf.to(mfq_tensor_backend::kFloat32);
            });
            auto router_logits = g_profiler.measure("moe.router", [&]() {
                return mfq_tensor_backend::matmul(xf32, moe_router.transpose(0, 1));
            });
            mfq_tensor_backend::Tensor shared_gate_logits;
            std::vector<mfq_tensor_backend::Tensor> selected;
            if (moe_hash_ids.defined()) {
                if (!input_ids.has_value()) {
                    throw std::runtime_error(
                        "hash-routed MoE requires the current token ids");
                }
                selected = g_profiler.measure("moe.hash_route", [&]() {
                    auto ids = moe_hash_ids.index_select(
                        0, input_ids.value().reshape({-1})
                               .to(mfq_tensor_backend::kCUDA, mfq_tensor_backend::kInt64))
                        .to(mfq_tensor_backend::kInt32).contiguous();
                    auto weights = moe_sqrtsoftplus_weights_cuda(
                        router_logits.contiguous(), ids, 1e-20,
                        moe_router_scale);
                    return std::vector<mfq_tensor_backend::Tensor>{ids, weights};
                });
            } else {
                selected = g_profiler.measure("moe.topk", [&]() {
                    return moe_topk_cuda(
                        router_logits.contiguous(), moe_top_k,
                        moe_use_sigmoid, moe_use_sqrt_softplus,
                        moe_normalize, moe_delayed_softmax,
                        moe_router_bias.defined()
                            ? MfqOptional<mfq_tensor_backend::Tensor>(moe_router_bias)
                            : mfq_nullopt,
                        1e-20, moe_router_scale);
                });
            }
            auto route = g_profiler.measure("moe.route_map", [&]() {
                return build_moe_route_plan(
                    selected.at(0),
                    moe_split_gate_up ? moe_gate.n_experts : moe_gate_up.n_experts);
            });
            if (expert_parallel_moe_compatible()) {
                auto routed = g_profiler.measure(
                    "moe.expert_parallel", [&]() {
                        return forward_expert_parallel_moe(
                            xf, route, selected.at(1));
                    });
                auto shared_output = shared->forward(xf);
                auto shared_half = shared_output
                    .reshape({xf.size(0), xf.size(1)})
                    .contiguous().to(mfq_tensor_backend::kFloat16);
                if (moe_shared_ungated) {
                    return g_profiler.measure("moe.combine", [&]() {
                        return acc_cuda(
                            routed.contiguous(), shared_half);
                    });
                }
                shared_gate_logits = g_profiler.measure(
                    "moe.shared_gate", [&]() {
                        return mfq_tensor_backend::matmul(
                            xf32,
                            moe_shared_gate.transpose(0, 1))
                            .contiguous();
                    });
                return g_profiler.measure("moe.combine", [&]() {
                    return moe_add_shared_gate_cuda(
                        routed.contiguous(), shared_half,
                        shared_gate_logits);
                });
            }

            // Start the tiny route readback before the shared expert.  The
            // cache consumes it after the shared branch has been enqueued,
            // then launches H2D on its separate stream.  The projection
            // forward below still performs the normal cache check and stream
            // wait, preserving the existing execution semantics.
            static const bool delayed_route_readback = [] {
                const char * value =
                    std::getenv("MFQ_MOE_DELAYED_ROUTE_READBACK");
                return value == nullptr || std::atoi(value) != 0;
            }();
            static const bool disable_projection_bundle = [] {
                const char * value = std::getenv(
                    "MFQ_DISABLE_MOE_PROJECTION_BUNDLE_PREFETCH");
                return value != nullptr && std::atoi(value) != 0;
            }();
            auto prefetch_projection_bundle = [&]() {
                if (disable_projection_bundle || cpu_moe_down ||
                        g_moe_continuous_batch_cache_serial) {
                    return false;
                }
                if (moe_split_gate_up) {
                    if (cpu_moe_gate || cpu_moe_up) return false;
                    return prefetch_cached_moe_projection_bundle(
                        moe_gate, moe_up, moe_down, route);
                }
                if (cpu_moe_gate_up) return false;
                return prefetch_cached_moe_projection_bundle(
                    moe_gate_up, moe_down, route);
            };
            bool projection_bundle_prefetched = false;
            if (moe_split_gate_up) {
                if (delayed_route_readback) {
                    moe_gate.prefetch_begin(route);
                } else {
                    projection_bundle_prefetched =
                        prefetch_projection_bundle();
                    if (!projection_bundle_prefetched) {
                        moe_gate.prefetch(route);
                    }
                }
            } else {
                if (delayed_route_readback) {
                    moe_gate_up.prefetch_begin(route);
                } else {
                    projection_bundle_prefetched =
                        prefetch_projection_bundle();
                    if (!projection_bundle_prefetched) {
                        moe_gate_up.prefetch(route);
                    }
                }
            }
            auto shared_output = g_profiler.measure(
                "moe.shared", [&]() {
                    return shared->forward(xf);
                });
            if (!moe_split_gate_up && !moe_shared_ungated) {
                shared_gate_logits = g_profiler.measure(
                    "moe.shared_gate", [&]() {
                    return mfq_tensor_backend::matmul(
                        xf32,
                        moe_shared_gate.transpose(0, 1))
                        .contiguous();
                });
            }
            if (delayed_route_readback) {
                projection_bundle_prefetched =
                    prefetch_projection_bundle();
                if (!projection_bundle_prefetched) {
                    if (moe_split_gate_up) {
                        moe_gate.prefetch(route);
                    } else {
                        moe_gate_up.prefetch(route);
                    }
                }
            }
            std::optional<MfeWeight> staged_gate_up;
            std::optional<MfeWeight> staged_gate;
            std::optional<MfeWeight> staged_up;
            const MfeWeight * active_gate_up = &moe_gate_up;
            const MfeWeight * active_gate = &moe_gate;
            const MfeWeight * active_up = &moe_up;
            mfq_tensor_backend::Tensor gate_up_pair;
            mfq_tensor_backend::Tensor projected_hidden;
            if (moe_split_gate_up) {
                if (cpu_moe_gate) {
                    staged_gate.emplace(g_profiler.measure(
                        "moe.cpu_offload_gate_h2d", [&]() {
                            return stage_cpu_mixed_moe(cpu_moe_gate);
                        }));
                    active_gate = &staged_gate.value();
                }
                if (cpu_moe_up) {
                    staged_up.emplace(g_profiler.measure(
                        "moe.cpu_offload_up_h2d", [&]() {
                            return stage_cpu_mixed_moe(cpu_moe_up);
                        }));
                    active_up = &staged_up.value();
                }
                gate_up_pair = g_profiler.measure("moe.gate_up_split", [&]() {
                    static const bool disable_activation_reuse = [] {
                        const char * value = std::getenv(
                            "MFQ_DISABLE_SPLIT_MOE_ACTIVATION_REUSE");
                        return value != nullptr && std::atoi(value) != 0;
                    }();
                    const bool reuse_gate_activation =
                        !disable_activation_reuse &&
                        g_kl_mmq_mode == KlMmqMode::Default &&
                        xf.size(0) <= 8 &&
                        active_gate->can_reuse_activation_for(*active_up);
                    auto gate = active_gate->forward(xf, route);
                    if (!projection_bundle_prefetched) {
                        active_up->prefetch(route);
                    }
                    if (!moe_shared_ungated &&
                            !shared_gate_logits.defined()) {
                        shared_gate_logits = g_profiler.measure(
                            "moe.shared_gate", [&]() {
                                return mfq_tensor_backend::matmul(
                                    xf32,
                                    moe_shared_gate.transpose(0, 1))
                                    .contiguous();
                            });
                    }
                    auto up = reuse_gate_activation
                        ? active_up->forward_prequantized(xf, route)
                        : active_up->forward(xf, route);
                    return mfq_tensor_backend::cat({gate, up}, -1).contiguous();
                });
            } else {
                if (cpu_moe_gate_up) {
                    staged_gate_up.emplace(g_profiler.measure(
                        "moe.cpu_offload_gate_up_h2d", [&]() {
                            return stage_cpu_mixed_moe(cpu_moe_gate_up);
                        }));
                    active_gate_up = &staged_gate_up.value();
                }
                const bool fuse_projection_glu =
                    swiglu_limit <= 0.0 &&
                    !g_force_moe_materialized_swiglu &&
                    active_gate_up->supports_projection_glu_epilogue();
                if (fuse_projection_glu) {
                    projected_hidden = g_profiler.measure(
                        "moe.gate_up_swiglu", [&]() {
                            return active_gate_up->forward_glu_output(
                                xf, route, false);
                        });
                } else {
                    gate_up_pair = g_profiler.measure("moe.gate_up", [&]() {
                        return active_gate_up->forward(xf, route);
                    });
                }
            }
            staged_gate_up.reset();
            staged_gate.reset();
            staged_up.reset();
            if (!projection_bundle_prefetched) {
                moe_down.prefetch(route);
            }
            std::optional<MfeWeight> staged_down;
            const MfeWeight * active_down = &moe_down;
            if (cpu_moe_down) {
                staged_down.emplace(g_profiler.measure(
                    "moe.cpu_offload_down_h2d", [&]() {
                        return stage_cpu_mixed_moe(cpu_moe_down);
                    }));
                active_down = &staged_down.value();
            }
            static const bool disable_swiglu_quant_fusion = [] {
                const char * value = std::getenv("MFQ_DISABLE_MOE_SWIGLU_QUANT_FUSION");
                return value != nullptr && std::atoi(value) != 0;
            }();
            mfq_tensor_backend::Tensor down_pair;
            const bool allow_swiglu_quant_fusion =
                gate_up_pair.defined() &&
                !g_force_moe_materialized_swiglu &&
                !disable_swiglu_quant_fusion &&
                moe_small_glu_path_enabled(
                    static_cast<int>(gate_up_pair.size(0)));
            if (projected_hidden.defined()) {
                down_pair = g_profiler.measure("moe.down", [&]() {
                    return active_down->forward(projected_hidden, route);
                });
            } else if (swiglu_limit <= 0.0 && allow_swiglu_quant_fusion) {
                down_pair = g_profiler.measure("moe.swiglu_down", [&]() {
                    return active_down->forward_swiglu(gate_up_pair, route);
                });
            } else if (swiglu_limit > 0.0 && allow_swiglu_quant_fusion &&
                    active_down->supports_clamped_swiglu()) {
                down_pair = g_profiler.measure("moe.swiglu_down", [&]() {
                    return active_down->forward_clamped_swiglu(
                        gate_up_pair, route, swiglu_limit);
                });
            } else {
                auto hidden = g_profiler.measure("moe.swiglu", [&]() {
                    if (swiglu_limit <= 0.0) {
                        return moe_swiglu_split_cuda(gate_up_pair);
                    }
                    const int64_t width = gate_up_pair.size(-1) / 2;
                    auto gate = mfq_tensor_backend::clamp_max(
                        gate_up_pair.slice(-1, 0, width).to(mfq_tensor_backend::kFloat32),
                        swiglu_limit);
                    auto up = mfq_tensor_backend::clamp(
                        gate_up_pair.slice(-1, width, 2 * width)
                            .to(mfq_tensor_backend::kFloat32),
                        -swiglu_limit, swiglu_limit);
                    return (mfq_tensor_backend::silu(gate) * up)
                        .to(mfq_tensor_backend::kFloat16).contiguous();
                });
                down_pair = g_profiler.measure("moe.down", [&]() {
                    return active_down->forward(hidden, route);
                });
            }
            record_moe_route_stats(
                moe_layer, selected.at(0), selected.at(1), down_pair,
                moe_split_gate_up ? moe_gate.n_experts : moe_gate_up.n_experts);
            staged_down.reset();
            static const bool disable_reduce_gate_fusion = [] {
                const char * value = std::getenv("MFQ_DISABLE_MOE_REDUCE_GATE_FUSION");
                return value != nullptr && std::atoi(value) != 0;
            }();
            const bool fuse_reduce_gate = !g_force_moe_unfused_reduce &&
                !disable_reduce_gate_fusion && !moe_shared_ungated &&
                down_pair.size(0) <= 8;
            mfq_tensor_backend::Tensor routed;
            if (!fuse_reduce_gate) {
                routed = g_profiler.measure("moe.reduce", [&]() {
                    return moe_weighted_reduce_cuda(down_pair, selected.at(1));
                });
            }
            if (!moe_shared_ungated && !shared_gate_logits.defined()) {
                shared_gate_logits = g_profiler.measure("moe.shared_gate", [&]() {
                    return mfq_tensor_backend::matmul(xf32, moe_shared_gate.transpose(0, 1)).contiguous();
                });
            }
            auto shared_half = shared_output.reshape({xf.size(0), xf.size(1)}).contiguous().to(mfq_tensor_backend::kFloat16);
            if (moe_shared_ungated) {
                return g_profiler.measure("moe.combine", [&]() {
                    return acc_cuda(routed.contiguous(), shared_half);
                });
            }
            if (fuse_reduce_gate) {
                return g_profiler.measure("moe.reduce_combine", [&]() {
                    return moe_weighted_reduce_shared_gate_cuda(
                        down_pair, selected.at(1), shared_half, shared_gate_logits);
                });
            }
            return g_profiler.measure("moe.combine", [&]() {
                return moe_add_shared_gate_cuda(
                    routed.contiguous(), shared_half,
                    shared_gate_logits);
            });
        }
        if (geglu) {
            const char * disable_geglu = std::getenv("MFQ_DISABLE_FFN_GEGLU_FUSION");
            const bool geglu_fusion_enabled =
                disable_geglu == nullptr || disable_geglu[0] != '1';
            if (geglu_fusion_enabled && xh.numel() / xh.size(-1) == 1 && gate_up.nint_grouped &&
                gate_up.nint.split_w.empty()) {
                auto act = g_profiler.measure("ffn.gate_up_geglu", [&]() {
                    return gate_up.forward_geglu(xh);
                });
                return g_profiler.measure("ffn.down", [&]() { return down.forward(act); });
            }
            auto parts = g_profiler.measure("ffn.gate_up", [&]() { return gate_up.forward(xh); });
            auto act = g_profiler.measure("ffn.geglu", [&]() {
                return gelu_mul_cuda(parts[0].contiguous(), parts[1].contiguous());
            });
            return g_profiler.measure("ffn.down", [&]() { return down.forward(act); });
        }
        if (swiglu_limit > 0.0) {
            auto parts = g_profiler.measure("ffn.gate_up", [&]() {
                return gate_up.forward(xh);
            });
            auto act = g_profiler.measure("ffn.swiglu_clamped", [&]() {
                auto gate = mfq_tensor_backend::clamp_max(
                    parts[0].to(mfq_tensor_backend::kFloat32), swiglu_limit);
                auto up = mfq_tensor_backend::clamp(
                    parts[1].to(mfq_tensor_backend::kFloat32),
                    -swiglu_limit, swiglu_limit);
                return (mfq_tensor_backend::silu(gate) * up)
                    .to(mfq_tensor_backend::kFloat16).contiguous();
            });
            return g_profiler.measure("ffn.down", [&]() {
                return down.forward(act);
            });
        }
        if (nvq_fusion_enabled() && xh.numel() / xh.size(-1) == 1 &&
            gate_up.nvq_prefix2 && gate_up.layers.size() == 2 &&
            gate_up.layers[0].is_nvq() && gate_up.layers[1].is_nvq() && down.is_nvq() &&
            gate_up.outs.size() == 2 && gate_up.outs[0] == gate_up.outs[1] &&
            gate_up.outs[0] == down.nvq.w.neuron_len &&
            (down.nvq.w.gs == 24 || down.nvq.w.gs == 28 || down.nvq.w.gs == 32)) {
            auto shape = xh.sizes().vec();
            shape.back() = down.nvq.w.out;
            auto y = nvq_ffn_swiglu_down(
                gate_up.layers[0].nvq.w, gate_up.layers[1].nvq.w, down.nvq.w,
                xh.reshape({-1, xh.size(-1)}));
            return y.reshape(shape);
        }
        const char* disable_swiglu = std::getenv("MFQ_DISABLE_FFN_SWIGLU_FUSION");
        if ((disable_swiglu == nullptr || disable_swiglu[0] != '1') &&
            xh.numel() / xh.size(-1) >= 1 && xh.numel() / xh.size(-1) <= 6 &&
            gate_up.nint_grouped && gate_up.nint.split_w.empty() &&
            gate_up.outs.size() == 2 && gate_up.outs[0] == gate_up.outs[1]) {
            auto act = g_profiler.measure("ffn.gate_up_swiglu", [&]() { return gate_up.forward_swiglu(xh); });
            return g_profiler.measure("ffn.down", [&]() { return down.forward(act); });
        }
        auto parts = g_profiler.measure("ffn.gate_up", [&]() { return gate_up.forward(xh); });
        if (down.is_dense() || down.is_mxfp8()) {
            auto activation = g_profiler.measure("ffn.swiglu", [&]() {
                return (parts[1] * mfq_tensor_backend::silu(parts[0])).contiguous();
            });
            return g_profiler.measure("ffn.down", [&]() {
                return down.forward(activation);
            });
        }
        return g_profiler.measure("ffn.down", [&]() { return down.forward_input_mul(parts[1], parts[0], 2); });
    }

    bool can_forward_fused_residual(
        const mfq_tensor_backend::Tensor & x,
        const mfq_tensor_backend::Tensor & residual) const {
        const char * fp32_residual_env =
            std::getenv("MFQ_DIAGNOSTIC_FP32_RESIDUAL");
        if (fp32_residual_env != nullptr && fp32_residual_env[0] == '1') {
            return false;
        }
        if (!nvq_fusion_enabled() || is_moe || geglu ||
            swiglu_limit > 0.0 || important_neurons ||
            tensor_parallel_dense_compatible() ||
            !x.is_cuda() || !residual.is_cuda() ||
            x.scalar_type() != mfq_tensor_backend::kFloat16 ||
            residual.scalar_type() != mfq_tensor_backend::kFloat16 ||
            !x.is_contiguous() || !residual.is_contiguous() ||
            x.dim() < 1 || residual.dim() < 1 ||
            x.numel() / x.size(-1) != 1 ||
            residual.numel() / residual.size(-1) != 1 ||
            gate_up.layers.size() != 2 || !gate_up.nvq_prefix2 ||
            !gate_up.layers[0].is_nvq() ||
            !gate_up.layers[1].is_nvq() || !down.is_nvq() ||
            gate_up.layers[0].tensor_parallel() ||
            gate_up.layers[1].tensor_parallel() || down.tensor_parallel() ||
            gate_up.outs.size() != 2 ||
            gate_up.outs[0] != gate_up.outs[1] ||
            gate_up.outs[0] != down.nvq.w.neuron_len ||
            x.size(-1) != gate_up.layers[0].nvq.w.neuron_len ||
            residual.size(-1) != down.nvq.w.out) {
            return false;
        }
        return nvq_fused_residual_format(
            down.nvq.w.kernel_format);
    }

    mfq_tensor_backend::Tensor forward_fused_residual(
        mfq_tensor_backend::Tensor x,
        mfq_tensor_backend::Tensor residual) const {
        MFQ_RUNTIME_CHECK(
            can_forward_fused_residual(x, residual),
            "FFN fused residual requires a compatible single-token NVQ FFN");
        auto shape = x.sizes().vec();
        shape.back() = down.nvq.w.out;
        auto output = nvq_ffn_swiglu_down(
            gate_up.layers[0].nvq.w, gate_up.layers[1].nvq.w,
            down.nvq.w, x.reshape({1, x.size(-1)}),
            residual.reshape({1, residual.size(-1)}));
        return output.reshape(shape);
    }

    mfq_tensor_backend::Tensor forward(
        mfq_tensor_backend::Tensor x,
        MfqOptional<mfq_tensor_backend::Tensor> input_ids =
            mfq_nullopt) const {
        return forward_impl(
            std::move(x), input_ids, true);
    }
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
            bool contiguous_prefill_prefix = false) {
        (void)start_pos;
        auto kh = kk.to(scalar_type()).contiguous();
        auto vh = vv.to(scalar_type()).contiguous();
        if (is_paged()) {
            MFQ_RUNTIME_CHECK(
                paged_batch == kh.size(0) && paged_heads == kh.size(1) &&
                paged_head_dim == kh.size(3) &&
                page_size > 0 && pages_per_chunk > 0,
                "Paged KV cache geometry does not match the write");
            paged_kv_cache_write_cuda(
                k_chunk_ptrs, v_chunk_ptrs, page_table,
                kh, vh, pos, page_size, pages_per_chunk);
            // Later prefill chunks must attend to the complete previous
            // prefix, including a final one-token chunk. Decode reads pages
            // directly and does not materialize a contiguous prefix.
            if (contiguous_prefill_prefix && start_pos > 0) {
                auto options = mfq_tensor_backend::TensorOptions()
                    .device(kh.device()).dtype(kh.scalar_type());
                auto prefix_k = mfq_tensor_backend::empty(
                    {paged_batch, paged_heads, end_pos, paged_head_dim}, options);
                auto prefix_v = mfq_tensor_backend::empty(
                    {paged_batch, paged_heads, end_pos, paged_head_dim}, options);
                paged_kv_cache_gather_cuda(k_chunk_ptrs, v_chunk_ptrs,
                    page_table, prefix_k, prefix_v, page_size, pages_per_chunk);
                return {prefix_k, prefix_v};
            }
            return {kh, vh};
        }
        MFQ_RUNTIME_CHECK(
            kh.dim() == 4 && vh.sizes() == kh.sizes() &&
            kh.size(0) == k.size(0) && kh.size(1) == k.size(1) &&
            kh.size(3) == k.size(3) &&
            ((pos.dim() == 1 && pos.numel() == kh.size(2)) ||
             (pos.dim() == 2 && pos.size(0) == kh.size(0) &&
              pos.size(1) == kh.size(2))),
            "KV cache write requires positions [T] or [B,T]");
        const char * aten_write_env = std::getenv("MFQ_KV_CACHE_WRITE_ATEN");
#ifdef MFQ_NATIVE_CUDA_RUNTIME
        const bool aten_write =
            aten_write_env != nullptr && aten_write_env[0] == '1';
#else
        const bool aten_write = k.scalar_type() != mfq_tensor_backend::kFloat16 ||
            (aten_write_env != nullptr && aten_write_env[0] == '1');
#endif
        if (k.is_cuda() && !aten_write && !ring) {
            auto out = kv_cache_write_cuda(k, v, kh, vh, pos);
            (void)out;
        } else {
            const int64_t batches = pos.dim() == 1 ? 1 : pos.size(0);
            for (int64_t batch = 0; batch < batches; ++batch) {
                auto cache_k = pos.dim() == 1 ? k : k.narrow(0, batch, 1);
                auto cache_v = pos.dim() == 1 ? v : v.narrow(0, batch, 1);
                auto source_k = pos.dim() == 1 ? kh : kh.narrow(0, batch, 1);
                auto source_v = pos.dim() == 1 ? vh : vh.narrow(0, batch, 1);
                auto write_pos = pos.dim() == 1
                    ? pos : pos.narrow(0, batch, 1).reshape({-1});
                if (!k.is_cuda() || aten_write) {
                    auto slots = ring
                        ? mfq_tensor_backend::remainder(write_pos, k.size(2))
                        : write_pos;
                    slots = slots.to(mfq_tensor_backend::kInt64).contiguous();
                    cache_k.index_copy_(2, slots, source_k);
                    cache_v.index_copy_(2, slots, source_v);
                } else {
                    auto out = ring
                        ? kv_cache_write_ring_positions_cuda(
                            cache_k, cache_v, source_k, source_v, write_pos)
                        : kv_cache_write_cuda(
                            cache_k, cache_v, source_k, source_v, write_pos);
                    (void)out;
                }
            }
        }
        if (ring) return {k, v};
        return {k.index({Slice(), Slice(), Slice(0, end_pos), Slice()}),
                v.index({Slice(), Slice(), Slice(0, end_pos), Slice()})};
    }
};

struct Block {
    struct Context {
        mfq_tensor_backend::Tensor token_ids;
        mfq_tensor_backend::Tensor positions;
        mfq_tensor_backend::Tensor full_positions;
        int64_t cache_position = 0;
        int64_t confirmed_prefix = 0;
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

    void reset(int64_t B) override {
        if (cache.defined() && cache.batch_size() == B) return;
        cache = KVCache();
        decode_partial_o = mfq_tensor_backend::Tensor();
        decode_partial_m = mfq_tensor_backend::Tensor();
        decode_partial_l = mfq_tensor_backend::Tensor();
        decode_mma_mask = mfq_tensor_backend::Tensor();
        decode_mma_kv_max = mfq_tensor_backend::Tensor();
        decode_mma_meta = mfq_tensor_backend::Tensor();
    }

    mfq_tensor_backend::Tensor forward(
            mfq_tensor_backend::Tensor x,
            mfq_tensor_backend::Tensor pos,
            int64_t cache_pos,
            const MfqOptional<mfq_tensor_backend::Tensor> & seq_len,
            const RopeCache & rope,
            const MfqOptional<mfq_tensor_backend::Tensor> & cache_positions = mfq_nullopt,
            const MfqOptional<mfq_tensor_backend::Tensor> & attention_mask = mfq_nullopt) override {
        int64_t B = x.size(0), T = x.size(1), H = x.size(2);
        auto trace_qwen_stage = [&](const char* name, const mfq_tensor_backend::Tensor& value,
                                    int token_axis = 1) {
            if (!attention_output_gate || g_gemma_stage_trace == nullptr ||
                    layer != g_gemma_trace_layer) return;
            auto ordered = token_axis == 1 ? value : value.transpose(1, token_axis).contiguous();
            trace_gemma_stage(layer, name, ordered.reshape({B, T, -1}));
        };
        if (official_bf16 &&
                x.scalar_type() != mfq_tensor_backend::kBFloat16) {
            x = x.to(mfq_tensor_backend::kBFloat16).contiguous();
        }
        const int64_t nh = attention_heads;
        const int64_t nkh = kv_heads;
        const int64_t hd = attention_head_dim;
        const int64_t attn_width = nh * hd;
        const int64_t cache_capacity = sliding
            ? attention_window
            : (g_kl_kv_cache_capacity > 0
                   ? std::max<int64_t>(
                         cache_pos + T, g_kl_kv_cache_capacity)
                   : max_position_embeddings);
        const RopeCache & active_rope = attention_rope.cos.defined() ? attention_rope : rope;
        if (!cache.defined()) {
            cache = KVCache(
                B, nkh, cache_capacity, hd, sliding, x.device(),
                official_bf16 ? mfq_tensor_backend::kBFloat16 : mfq_tensor_backend::kFloat16);
        }
        if (x.is_cuda()) {
            const int64_t total = B * nh;
            if (!decode_partial_o.defined() ||
                    decode_partial_o.size(0) < total ||
                    decode_partial_o.size(1) != kDecodeAttentionMaxParts ||
                    decode_partial_o.size(2) != hd) {
                auto opts = mfq_tensor_backend::TensorOptions()
                    .device(x.device()).dtype(mfq_tensor_backend::kFloat32);
                decode_partial_o = mfq_tensor_backend::empty(
                    {total, kDecodeAttentionMaxParts, hd}, opts);
                decode_partial_m = mfq_tensor_backend::empty(
                    {total, kDecodeAttentionMaxParts}, opts);
                decode_partial_l = mfq_tensor_backend::empty(
                    {total, kDecodeAttentionMaxParts}, opts);
            }
        }
        auto residual = x;
        auto xn = g_profiler.measure("full.attn_norm", [&]() {
            return official_bf16
                ? qwen_rms_norm_bf16(
                    x.reshape({B * T, H}), attn_norm,
                    rms_norm_eps, norm_weight_offset)
                    .reshape({B, T, H})
                : qwen_rms_norm(
                    x.reshape({B * T, H})
                        .to(mfq_tensor_backend::kFloat32),
                    attn_norm, rms_norm_eps, norm_weight_offset)
                    .reshape({B, T, H});
        });
        trace_qwen_stage("qwen.attn_norm", xn);
        auto parts = g_profiler.measure("full.qkv", [&]() {
            if (!split_q_kv_projections) return qkv.forward(xn);
            std::vector<mfq_tensor_backend::Tensor> result;
            result.reserve(3);
            result.push_back(q_projection.forward(xn));
            result.push_back(k_projection.forward(xn));
            result.push_back(v_projection.forward(xn));
            return result;
        });
        if (parts.size() != (value_equals_key ? 2u : 3u)) {
            throw std::runtime_error("attention projection group has the wrong output count");
        }
        auto q_full = parts[0], k_full = parts[1];
        auto v_full = value_equals_key ? k_full : parts[2];
        trace_qwen_stage("qwen.q_projection", q_full);
        trace_qwen_stage("qwen.k_projection", k_full);
        trace_qwen_stage("qwen.v_projection", v_full);
        if (official_bf16) {
            q_full = q_full.to(mfq_tensor_backend::kBFloat16).contiguous();
            k_full = k_full.to(mfq_tensor_backend::kBFloat16).contiguous();
            v_full = v_full.to(mfq_tensor_backend::kBFloat16).contiguous();
        }
        mfq_tensor_backend::Tensor q_raw, q_gate;
        g_profiler.measure("full.qkv_view", [&]() {
            if (attention_output_gate) {
            auto qp = q_full.reshape({B, T, nh, hd * 2});
            auto chunks = qp.chunk(2, -1);
            q_raw = chunks[0];
            q_gate = chunks[1];
            } else {
            q_raw = q_full.reshape({B, T, nh, hd});
            }
            return q_raw;
        });
        auto q = g_profiler.measure("full.q_view", [&]() { return q_raw.transpose(1, 2).contiguous(); });
        auto k = g_profiler.measure("full.k_view", [&]() { return k_full.reshape({B, T, nkh, hd}).transpose(1, 2).contiguous(); });
        auto v = g_profiler.measure("full.v_view", [&]() { return v_full.reshape({B, T, nkh, hd}).transpose(1, 2).contiguous(); });
        auto write_positions = cache_positions.has_value()
            ? cache_positions.value().to(x.device(), mfq_tensor_backend::kInt64).contiguous()
            : pos.to(x.device(), mfq_tensor_backend::kInt64).contiguous();
        if (!((write_positions.dim() == 1 && write_positions.numel() == T) ||
              (write_positions.dim() == 2 && write_positions.size(0) == B &&
               write_positions.size(1) == T))) {
            throw std::runtime_error(
                "KV cache positions must have shape [tokens] or [batch,tokens]");
        }
        // A leading extent of three is semantic axes only for the prepared
        // single-image path. B=3 ordinary batching must remain batch-major.
        const bool grid_mrope_positions = B == 1 &&
            cache_positions.has_value() && pos.dim() == 2 &&
            pos.size(0) == 3;
        const char * fused_qk_rope_kv_env =
            std::getenv("MFQ_MINICPM_FUSED_QK_NORM_ROPE_KV");
        const bool fused_qk_rope_kv = official_bf16 && x.is_cuda() &&
            write_positions.dim() == 1 && pos.dim() == 1 && T == 1 &&
            !cache.is_paged() && !cache.ring && !v_norm.defined() &&
            q_norm.defined() && k_norm.defined() &&
            active_rope.rotary_dim == 128 &&
            active_rope.sections.numel() == 0 && nh == 32 && nkh == 8 &&
            hd == 128 && cache.scalar_type() == mfq_tensor_backend::kBFloat16 &&
            (fused_qk_rope_kv_env == nullptr ||
             fused_qk_rope_kv_env[0] != '0');
        std::pair<mfq_tensor_backend::Tensor, mfq_tensor_backend::Tensor> kv;
        if (fused_qk_rope_kv) {
            q = g_profiler.measure("full.qk_norm_rope_kv_write", [&]() {
                return minicpm_qk_norm_rope_cache_write_bf16_cuda(
                    q.contiguous(), k.contiguous(), v.contiguous(),
                    q_norm, k_norm,
                    pos.contiguous().to(x.device(), mfq_tensor_backend::kInt64),
                    write_positions, active_rope.cos, active_rope.sin,
                    cache.k, cache.v, rms_norm_eps,
                    norm_weight_offset);
            });
            kv = {
                cache.k.index({Slice(), Slice(), Slice(0, cache_pos + T), Slice()}),
                cache.v.index({Slice(), Slice(), Slice(0, cache_pos + T), Slice()})};
        } else {
        const char * fused_bf16_norm_env =
            std::getenv("MFQ_MINICPM_FUSED_BF16_RMSNORM");
        const bool fused_bf16_norm = official_bf16 &&
            (fused_bf16_norm_env == nullptr ||
             fused_bf16_norm_env[0] != '0');
        if (fused_bf16_norm && q_norm.defined() && k_norm.defined() &&
                q.scalar_type() == mfq_tensor_backend::kBFloat16 &&
                k.scalar_type() == mfq_tensor_backend::kBFloat16) {
            auto normalized = g_profiler.measure("full.qk_norm", [&]() {
                return qwen_rms_norm_pair_bf16_cuda(
                    q, k, q_norm, k_norm,
                    rms_norm_eps, norm_weight_offset);
            });
            q = normalized[0].reshape_as(q);
            k = normalized[1].reshape_as(k);
        } else if (!official_bf16 && q_norm.defined() && k_norm.defined() &&
            q.scalar_type() == mfq_tensor_backend::kFloat16 &&
            k.scalar_type() == mfq_tensor_backend::kFloat16) {
            auto normalized = g_profiler.measure("full.qk_norm", [&]() {
                return rms_norm_pair_f16_f32_offset_cuda(
                    q, k, q_norm, k_norm,
                    rms_norm_eps, norm_weight_offset);
            });
            q = normalized[0].reshape_as(q);
            k = normalized[1].reshape_as(k);
        } else {
            if (q_norm.defined()) q = g_profiler.measure("full.q_norm", [&]() {
                return official_bf16
                    ? qwen_rms_norm_bf16(
                        q.reshape({-1, hd}), q_norm,
                        rms_norm_eps, norm_weight_offset).reshape_as(q)
                    : qwen_rms_norm(
                        q.reshape({-1, hd})
                            .to(mfq_tensor_backend::kFloat32),
                        q_norm, rms_norm_eps, norm_weight_offset)
                        .reshape_as(q);
            });
            if (k_norm.defined()) k = g_profiler.measure("full.k_norm", [&]() {
                return official_bf16
                    ? qwen_rms_norm_bf16(
                        k.reshape({-1, hd}), k_norm,
                        rms_norm_eps, norm_weight_offset).reshape_as(k)
                    : qwen_rms_norm(
                        k.reshape({-1, hd})
                            .to(mfq_tensor_backend::kFloat32),
                        k_norm, rms_norm_eps, norm_weight_offset)
                        .reshape_as(k);
            });
        }
        if (v_norm.defined()) v = g_profiler.measure("full.v_norm", [&]() {
            return official_bf16
                ? qwen_rms_norm_bf16(
                    v.reshape({-1, hd}), v_norm,
                    rms_norm_eps, norm_weight_offset).reshape_as(v)
                : qwen_rms_norm(
                    v.reshape({-1, hd}).to(mfq_tensor_backend::kFloat32),
                    v_norm, rms_norm_eps, norm_weight_offset)
                    .reshape_as(v);
        });
        const char * fused_rope_kv_env =
            std::getenv("MFQ_MINICPM_FUSED_ROPE_KV");
        const bool fused_rope_kv = official_bf16 && x.is_cuda() &&
            write_positions.dim() == 1 && pos.dim() == 1 && T == 1 &&
            !cache.is_paged() && !cache.ring && active_rope.rotary_dim == 128 &&
            active_rope.sections.numel() == 0 && nh == 32 && nkh == 8 &&
            hd == 128 && cache.scalar_type() == mfq_tensor_backend::kBFloat16 &&
            (fused_rope_kv_env == nullptr || fused_rope_kv_env[0] != '0');
        if (fused_rope_kv) {
            q = g_profiler.measure("full.rope_kv_write", [&]() {
                return minicpm_bf16_rope_cache_write_cuda(
                    q.contiguous(), k.contiguous(), v.contiguous(),
                    pos.contiguous().to(x.device(), mfq_tensor_backend::kInt64),
                    write_positions, active_rope.cos, active_rope.sin,
                    cache.k, cache.v, active_rope.rotary_dim);
            });
            kv = {
                cache.k.index({Slice(), Slice(), Slice(0, cache_pos + T), Slice()}),
                cache.v.index({Slice(), Slice(), Slice(0, cache_pos + T), Slice()})};
        } else {
            q = g_profiler.measure("full.q_rope", [&]() {
                return official_bf16
                    ? active_rope.apply_bf16(q, pos)
                    : active_rope.apply(q, pos, grid_mrope_positions);
            });
            k = g_profiler.measure("full.k_rope", [&]() {
                return official_bf16
                    ? active_rope.apply_bf16(k, pos)
                    : active_rope.apply(k, pos, grid_mrope_positions);
            });
            kv = g_profiler.measure("full.kv_write", [&]() {
                return cache.append(
                    k, v, write_positions, cache_pos, cache_pos + T,
                    !seq_len.has_value());
            });
        }
        }
        trace_qwen_stage("qwen.q_rope", q, 2);
        trace_qwen_stage("qwen.k_rope", k, 2);
        auto minicpmo45_attention_mask = [&](int64_t visible_len,
                                              bool explicit_causal) {
            MfqOptional<mfq_tensor_backend::Tensor> result = mfq_nullopt;
            if (!official_bf16 ||
                    (!explicit_causal && !seq_len.has_value() &&
                     !attention_mask.has_value())) {
                return result;
            }
            auto options = mfq_tensor_backend::TensorOptions()
                .device(x.device()).dtype(mfq_tensor_backend::kInt64);
            auto key_positions = mfq_tensor_backend::arange(visible_len, options);
            auto query_positions = write_positions
                .to(x.device(), mfq_tensor_backend::kInt64);
            if (query_positions.dim() == 1) {
                query_positions = query_positions
                    .reshape({1, T}).expand({B, T});
            }
            auto allowed = key_positions.reshape({1, 1, visible_len}) <=
                query_positions.unsqueeze(-1);
            if (seq_len.has_value()) {
                auto lengths = seq_len.value()
                    .to(x.device(), mfq_tensor_backend::kInt64).reshape({B, 1, 1});
                allowed = allowed &
                    (key_positions.reshape({1, 1, visible_len}) < lengths);
            }
            if (attention_mask.has_value()) {
                auto valid = attention_mask.value().to(x.device());
                if (valid.dim() != 2 || valid.size(0) != B ||
                        valid.size(1) < visible_len) {
                    throw std::runtime_error(
                        "MiniCPM-o attention_mask must cover [batch,visible_tokens]");
                }
                valid = valid.narrow(1, 0, visible_len).ne(0);
                allowed = allowed & valid.unsqueeze(1);
                auto attended = allowed.any(-1, true);
                allowed = mfq_tensor_backend::where(
                    attended, allowed, mfq_tensor_backend::ones_like(allowed));
            }
            const auto mask_dtype = official_bf16
                ? mfq_tensor_backend::kBFloat16 : x.scalar_type();
            auto additive = mfq_tensor_backend::zeros(
                {B, 1, T, visible_len},
                mfq_tensor_backend::TensorOptions().device(x.device())
                    .dtype(mask_dtype));
            const double mask_min = static_cast<double>(
                std::numeric_limits<mfq_bfloat16>::lowest());
            additive.masked_fill_(
                allowed.logical_not().unsqueeze(1),
                mask_min);
            result = additive.contiguous();
            return result;
        };
        double attn_scale = attention_scale > 0.0 ? attention_scale : 1.0 / std::sqrt((double)hd);
        mfq_tensor_backend::Tensor a;
        bool attention_token_major = false;
        a = g_profiler.measure("full.attention", [&]() {
            if (!x.is_cuda()) {
                MFQ_RUNTIME_CHECK(!sliding, "CPU dense offload does not support sliding attention");
                const auto cpu_attention_dtype = official_bf16
                    ? mfq_tensor_backend::kBFloat16 : mfq_tensor_backend::kFloat32;
                auto qh = q.to(cpu_attention_dtype).contiguous();
                auto kh = kv.first.to(cpu_attention_dtype).contiguous();
                auto vh = kv.second.to(cpu_attention_dtype).contiguous();
                if (nkh != nh) {
                    MFQ_RUNTIME_CHECK(nh % nkh == 0, "CPU attention head ratio is invalid");
                    const int64_t repeat = nh / nkh;
                    kh = kh.repeat_interleave(repeat, 1);
                    vh = vh.repeat_interleave(repeat, 1);
                }
                MfqOptional<mfq_tensor_backend::Tensor> mask = mfq_nullopt;
                if (official_bf16) {
                    mask = minicpmo45_attention_mask(
                        kh.size(2), T > 1);
                } else if (T > 1 || seq_len.has_value()) {
                    const int64_t visible = kh.size(2);
                    auto key_positions = mfq_tensor_backend::arange(
                        visible,
                        mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCPU).dtype(mfq_tensor_backend::kInt64));
                    mfq_tensor_backend::Tensor allowed;
                    if (T > 1) {
                        auto query_positions = pos.dim() == 1
                            ? pos.to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt64)
                            : pos.select(0, 0).to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt64);
                        allowed = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1);
                        allowed = allowed.unsqueeze(0).expand({B, T, visible});
                    } else {
                        allowed = mfq_tensor_backend::ones(
                            {B, T, visible},
                            mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCPU).dtype(mfq_tensor_backend::kBool));
                    }
                    if (seq_len.has_value()) {
                        auto lengths = seq_len.value().to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt64)
                            .reshape({B, 1, 1});
                        allowed = allowed & (key_positions.reshape({1, 1, visible}) < lengths);
                    }
                    mask = allowed.unsqueeze(1);
                }
                return mfq_scaled_dot_product_attention(
                    qh, kh, vh, mask, 0.0, false, attn_scale, false);
            }
            const auto attention_dtype = official_bf16
                ? mfq_tensor_backend::kBFloat16 : mfq_tensor_backend::kFloat16;
            auto qh = q.to(attention_dtype).contiguous();
            auto kh = k.to(attention_dtype).contiguous();
            auto vh = v.to(attention_dtype).contiguous();
            if (cache_pos == 0 && T > 1) {
                if (official_bf16) {
                    const char* bf16_flash128_disabled =
                        std::getenv("MFQ_DISABLE_MINICPM_BF16_FLASH128");
                    const bool bf16_flash128 = !sliding && hd == 128 &&
                        nh == 4 * nkh && !seq_len.has_value() &&
                        !attention_mask.has_value() &&
                        (bf16_flash128_disabled == nullptr ||
                         bf16_flash128_disabled[0] != '1');
                    if (bf16_flash128) {
                        const char* specialized_casts_disabled = std::getenv(
                            "MFQ_DISABLE_MINICPM_FLASH128_SPECIALIZED_CASTS");
                        const bool specialized_casts =
                            specialized_casts_disabled == nullptr ||
                            specialized_casts_disabled[0] != '1';
                        auto flash_q = g_profiler.measure(
                            "full.flash128_q_cast", [&]() {
                                return specialized_casts
                                    ? minicpm_flash128_q_cast_cuda(qh)
                                    : qh.to(mfq_tensor_backend::kFloat32)
                                        .contiguous();
                            });
                        auto flash_kv = g_profiler.measure(
                            "full.flash128_kv_cast", [&]() {
                                if (specialized_casts) {
                                    return minicpm_flash128_kv_cast_cuda(kh, vh);
                                }
                                return std::vector<mfq_tensor_backend::Tensor>{
                                    kh.to(mfq_tensor_backend::kFloat16)
                                        .contiguous(),
                                    vh.to(mfq_tensor_backend::kFloat16)
                                        .contiguous()};
                            });
                        a = g_profiler.measure(
                            "full.flash128_kernel", [&]() {
                                return mfq_attention_mma128_cuda(
                                    flash_q, flash_kv[0], flash_kv[1],
                                    attn_scale);
                            });
                        a = g_profiler.measure(
                            "full.flash128_output_cast", [&]() {
                                return specialized_casts
                                    ? minicpm_flash128_output_cast_cuda(a)
                                    : a.to(mfq_tensor_backend::kBFloat16)
                                        .contiguous();
                            });
                        attention_token_major = true;
                    } else {
                        auto repeated_k = kh;
                        auto repeated_v = vh;
                        if (nkh != nh) {
                            MFQ_RUNTIME_CHECK(
                                nh % nkh == 0,
                                "MiniCPM-o Qwen3 attention head ratio is invalid");
                            const int64_t repeat = nh / nkh;
                            repeated_k = kh.repeat_interleave(repeat, 1).contiguous();
                            repeated_v = vh.repeat_interleave(repeat, 1).contiguous();
                        }
                        auto mask = minicpmo45_attention_mask(T, false);
                        a = mfq_scaled_dot_product_attention(
                            qh, repeated_k, repeated_v, mask,
                            0.0, !mask.has_value(), attn_scale, false);
                    }
                } else {
                const char * mma_attention_env = std::getenv("MFQ_MMA_ATTENTION");
                const bool mma_attention_enabled =
                    mma_attention_env == nullptr || mma_attention_env[0] != '0';
                if (!sliding && T % 256 == 0 && hd == 512 && nh == 8 * nkh &&
                    mma_attention_enabled) {
                    a = mfq_attention_mma512_cuda(
                        q.to(mfq_tensor_backend::kFloat32).contiguous(), kh, vh, attn_scale);
                    attention_token_major = true;
                } else if (sliding && T >= 32 && hd == 256 && nh == 2 * nkh &&
                    mma_attention_enabled) {
                    a = mfq_attention_mma256_swa_cuda(
                        q.to(mfq_tensor_backend::kFloat32).contiguous(), kh, vh,
                        attn_scale, attention_window);
                    attention_token_major = true;
                } else if (!sliding && T >= 32 && hd == 256 &&
                           (nh == 4 * nkh || nh == 8 * nkh) &&
                           mma_attention_enabled) {
                    a = mfq_attention_mma256_cuda(
                        q.to(mfq_tensor_backend::kFloat32).contiguous(), kh, vh, attn_scale);
                    attention_token_major = true;
                } else if (sliding) {
                    a = attention_swa_cuda(qh, kh, vh, attn_scale, attention_window);
                } else {
                    a = mfq_scaled_dot_product_attention(
                        qh, kh, vh, std::nullopt, 0.0, true, attn_scale, true);
                }
                }
            } else if (seq_len.has_value()) {
                const int64_t planned_len = g_decode_graph_attention_kv_len > 0
                    ? g_decode_graph_attention_kv_len : cache_pos + T;
                const char * aten_decode_env = std::getenv("MFQ_ATTENTION_DECODE_ATEN");
                const char * bf16_gqa_env = std::getenv("MFQ_MINICPM_BF16_GQA_DECODE");
                const bool bf16_gqa_decode = official_bf16 && T == 1 &&
                    (bf16_gqa_env == nullptr || bf16_gqa_env[0] != '0');
                const bool aten_decode_enabled = (official_bf16 && !bf16_gqa_decode) ||
                    (aten_decode_env != nullptr && aten_decode_env[0] == '1');
                const char * mma_decode_env = std::getenv("MFQ_MMA_ATTENTION_DECODE");
                const bool mma_decode_enabled =
                    mma_decode_env == nullptr || mma_decode_env[0] != '0';
                auto prepare_mma_decode_workspace = [&](int64_t visible_len, int64_t kv_tile) {
                    const int64_t mask_stride = (visible_len + kv_tile - 1) / kv_tile * kv_tile;
                    const int64_t ntiles_kv = (visible_len + kv_tile - 1) / kv_tile;
                    const int64_t max_blocks = B * nkh * ntiles_kv;
                    const int64_t meta_float2 = max_blocks * 8 * (2 + hd / 2);
                    auto cuda = mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA);
                    if (!decode_mma_mask.defined() || decode_mma_mask.size(0) != B ||
                        decode_mma_mask.size(1) < mask_stride) {
                        decode_mma_mask = mfq_tensor_backend::empty(
                            {B, mask_stride}, cuda.dtype(mfq_tensor_backend::kFloat16));
                    }
                    if (!decode_mma_kv_max.defined() || decode_mma_kv_max.numel() < B) {
                        decode_mma_kv_max = mfq_tensor_backend::empty({B}, cuda.dtype(mfq_tensor_backend::kInt32));
                    }
                    if (!decode_mma_meta.defined() || decode_mma_meta.numel() < 2 * meta_float2) {
                        decode_mma_meta = mfq_tensor_backend::empty(
                            {2 * meta_float2}, cuda.dtype(mfq_tensor_backend::kFloat32));
                    }
                };
                if (cache.is_paged()) {
                    const char * split_env =
                        std::getenv("MFQ_ATTENTION_DECODE_SPLITK");
                    const bool split_enabled =
                        split_env == nullptr || split_env[0] != '0';
                    int64_t parts = split_enabled && cache_pos >= 192
                        ? (cache_pos + 127) / 128 : 1;
                    const bool dynamic_parts =
                        split_enabled && g_decode_graph_attention_parts > 1;
                    if (dynamic_parts) parts = g_decode_graph_attention_parts;
                    parts = std::min<int64_t>(
                        parts, kDecodeAttentionMaxParts);
                    a = attention_paged_cache_decode_cuda(
                        qh, cache.k_chunk_ptrs, cache.v_chunk_ptrs,
                        cache.page_table, seq_len.value(), attn_scale,
                        cache.page_size, cache.pages_per_chunk,
                        cache.paged_heads,
                        decode_partial_o, decode_partial_m, decode_partial_l,
                        parts, dynamic_parts);
                } else if (aten_decode_enabled) {
                    const int64_t visible_len = sliding
                        ? std::min<int64_t>(attention_window, cache_pos + T)
                        : cache_pos + T;
                    auto cached_k = cache.k.index({
                        Slice(), Slice(), Slice(0, visible_len), Slice()}).contiguous();
                    auto cached_v = cache.v.index({
                        Slice(), Slice(), Slice(0, visible_len), Slice()}).contiguous();
                    auto mask = minicpmo45_attention_mask(
                        visible_len, cache_pos > 0 && T > 1);
                    if (official_bf16 && nkh != nh) {
                        MFQ_RUNTIME_CHECK(
                            nh % nkh == 0,
                            "MiniCPM-o Qwen3 attention head ratio is invalid");
                        const int64_t repeat = nh / nkh;
                        cached_k = cached_k.repeat_interleave(repeat, 1).contiguous();
                        cached_v = cached_v.repeat_interleave(repeat, 1).contiguous();
                    }
                    a = mfq_scaled_dot_product_attention(
                        qh, cached_k, cached_v, mask,
                        0.0, false, attn_scale, !official_bf16);
                } else if (sliding && T == 1 && mma_decode_enabled && hd == 256 && nh == 2 * nkh) {
                    const int64_t visible_len = std::min<int64_t>(attention_window, planned_len);
                    prepare_mma_decode_workspace(visible_len, 64);
                    a = mfq_attention_mma256_swa_decode_cuda(
                        q.to(mfq_tensor_backend::kFloat32).contiguous(), cache.k, cache.v,
                        seq_len.value(), attn_scale, visible_len,
                        decode_mma_mask, decode_mma_kv_max, decode_mma_meta);
                    attention_token_major = true;
                } else if (!sliding && T == 1 && mma_decode_enabled &&
                           nh == 8 * nkh && (hd == 256 || hd == 512)) {
                    const int64_t kv_tile = hd == 512 ? 32 : 64;
                    prepare_mma_decode_workspace(planned_len, kv_tile);
                    a = hd == 512
                        ? mfq_attention_mma512_decode_cuda(
                            q.to(mfq_tensor_backend::kFloat32).contiguous(), cache.k, cache.v,
                            seq_len.value(), attn_scale, planned_len,
                            decode_mma_mask, decode_mma_kv_max, decode_mma_meta)
                        : mfq_attention_mma256_decode_cuda(
                            q.to(mfq_tensor_backend::kFloat32).contiguous(), cache.k, cache.v,
                            seq_len.value(), attn_scale, planned_len,
                            decode_mma_mask, decode_mma_kv_max, decode_mma_meta);
                    attention_token_major = true;
                } else if (sliding) {
                    a = attention_cache_swa_planned_cuda(
                        qh, cache.k, cache.v, seq_len.value(),
                        attn_scale, attention_window, planned_len);
                } else {
                        const char * split_env = std::getenv("MFQ_ATTENTION_DECODE_SPLITK");
                        const bool split_enabled =
                            split_env == nullptr || split_env[0] != '0';
                        int64_t parts = split_enabled && cache_pos >= 192
                            ? (cache_pos + 127) / 128 : 1;
                        if (split_enabled && g_decode_graph_attention_parts > 0) {
                            parts = g_decode_graph_attention_parts;
                        }
                        parts = std::min<int64_t>(parts, kDecodeAttentionMaxParts);
                        a = g_decode_graph_attention_parts > 1
                            ? attention_cache_decode_dynamic_cuda(
                                qh, cache.k, cache.v, seq_len.value(), attn_scale,
                                decode_partial_o, decode_partial_m, decode_partial_l,
                                parts)
                            : parts > 1
                            ? attention_cache_decode_split_cuda(
                                qh, cache.k, cache.v, seq_len.value(), attn_scale,
                                decode_partial_o, decode_partial_m, decode_partial_l, parts)
                            : attention_cache_decode_cuda(
                                qh, cache.k, cache.v, seq_len.value(), attn_scale);
                }
            } else {
                if (official_bf16) {
                    auto cached_k = kv.first.contiguous();
                    auto cached_v = kv.second.contiguous();
                    if (nkh != nh) {
                        MFQ_RUNTIME_CHECK(
                            nh % nkh == 0,
                            "MiniCPM-o Qwen3 attention head ratio is invalid");
                        const int64_t repeat = nh / nkh;
                        cached_k = cached_k.repeat_interleave(repeat, 1).contiguous();
                        cached_v = cached_v.repeat_interleave(repeat, 1).contiguous();
                    }
                    auto mask = minicpmo45_attention_mask(
                        cached_k.size(2), cache_pos > 0 && T > 1);
                    a = mfq_scaled_dot_product_attention(
                        qh, cached_k, cached_v, mask,
                        0.0, !mask.has_value() && cache_pos == 0 && T > 1,
                        attn_scale, false);
                } else {
                    a = sliding
                        ? attention_swa_cuda(
                            qh, kv.first.contiguous(), kv.second.contiguous(),
                            attn_scale, attention_window)
                        : attention_cuda(
                            qh, kv.first.contiguous(), kv.second.contiguous(),
                            attn_scale, true);
                }
            }
            return a;
        });
        mfq_tensor_backend::Tensor oo;
        if (q_gate.defined()) {
            trace_qwen_stage("qwen.attention", a, attention_token_major ? 1 : 2);
            auto af = g_profiler.measure("full.attn_out_view", [&]() {
                return attention_token_major ? a.reshape({B, T, attn_width}) :
                    a.transpose(1, 2).contiguous().reshape({B, T, attn_width});
            });
            auto gf = g_profiler.measure("full.q_gate_view", [&]() { return q_gate.contiguous().reshape({B, T, attn_width}); });
            oo = g_profiler.measure("full.o_proj_gate", [&]() { return o.forward_input_mul(af, gf, 1); });
        } else {
            auto af = g_profiler.measure("full.attn_out_view", [&]() {
                return attention_token_major ? a.reshape({B, T, attn_width}) :
                    a.transpose(1, 2).contiguous().reshape({B, T, attn_width});
            });
            oo = g_profiler.measure("full.o_proj", [&]() {
                return official_bf16
                    ? o.forward_bf16_output(af)
                    : o.forward(af);
            });
        }
        trace_qwen_stage("qwen.o_projection", oo);
        if (official_bf16) {
            oo = oo.to(mfq_tensor_backend::kBFloat16).contiguous();
        }
        if (gemma4) {
            trace_gemma_stage(layer, "attention_output", oo);
            const bool fused_norms = gemma4_moe &&
                gemma4_fused_norms_enabled() &&
                g_gemma_stage_trace == nullptr && layer_scale.defined();
            mfq_tensor_backend::Tensor dense_input;
            mfq_tensor_backend::Tensor router_input;
            mfq_tensor_backend::Tensor moe_input;
            if (fused_norms) {
                auto prepared = g_profiler.measure("gemma.attn_residual_pre_norms", [&]() {
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
                auto attn_post = g_profiler.measure("gemma.attn_post_norm", [&]() {
                    return gemma_rms_norm_f16(
                        oo.reshape({B * T, H}), attn_post_norm,
                        rms_norm_eps, norm_weight_offset);
                });
                x = g_profiler.measure("gemma.attn_residual", [&]() {
                    return acc_cuda(residual.reshape({B * T, H}), attn_post).reshape({B, T, H});
                });
                trace_gemma_stage(layer, "attention_residual", x);
                residual = x;
                dense_input = g_profiler.measure("gemma.ffn_pre_norm", [&]() {
                    return gemma_rms_norm_f16(
                        x.reshape({B * T, H}), ffn_norm,
                        rms_norm_eps, norm_weight_offset);
                });
                if (gemma4_moe) {
                    router_input = g_profiler.measure("gemma.router_norm", [&]() {
                        return qwen_rms_norm(
                            x.reshape({B * T, H})
                                .to(mfq_tensor_backend::kFloat32),
                            gemma_router_norm_scale,
                            rms_norm_eps, norm_weight_offset);
                    });
                    moe_input = g_profiler.measure("gemma.ffn_pre_norm_2", [&]() {
                        return gemma_rms_norm_f16(
                            x.reshape({B * T, H}), ffn_pre_norm_2,
                            rms_norm_eps, norm_weight_offset);
                    });
                }
            }
            auto dense_output = g_profiler.measure("gemma.ffn_dense", [&]() {
                return ffn.forward(dense_input).reshape({B * T, H});
            });
            if (!gemma4_moe) {
                auto dense_post = g_profiler.measure("gemma.ffn_post_norm", [&]() {
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
                auto result = g_profiler.measure("gemma.ffn_residual", [&]() {
                    return acc_cuda(
                        residual.reshape({B * T, H}), dense_post)
                        .reshape({B, T, H});
                });
                if (layer_scale.defined()) {
                    result = g_profiler.measure("gemma.layer_scale", [&]() {
                        return result * layer_scale;
                    });
                }
                trace_gemma_stage(layer, "layer_output", result);
                return result;
            }
            if (!fused_norms) {
                dense_output = g_profiler.measure("gemma.ffn_post_norm_1", [&]() {
                    return gemma_rms_norm_f16(
                        dense_output, ffn_post_norm_1,
                        rms_norm_eps, norm_weight_offset);
                });
                trace_gemma_stage(layer, "dense_output", dense_output);
            }
            auto router_logits = g_profiler.measure("gemma.router", [&]() {
                return mfq_tensor_backend::matmul(router_input, gemma_router.transpose(0, 1));
            });
            auto selected = g_profiler.measure("gemma.topk", [&]() {
                return moe_topk_cuda(
                    router_logits.contiguous(), gemma_top_k,
                    false, false, false, true, mfq_nullopt, 1e-20, 1.0);
            });
            trace_gemma_stage(layer, "route_ids", selected.at(0));
            trace_gemma_stage(layer, "route_weights_before_scale", selected.at(1));
            g_profiler.measure("gemma.route_scale", [&]() {
                return moe_apply_expert_scale_cuda(
                    selected.at(1), selected.at(0), gemma_expert_scale);
            });
            trace_gemma_stage(layer, "route_weights", selected.at(1));
            auto route = g_profiler.measure("gemma.route_map", [&]() {
                return build_moe_route_plan(selected.at(0), gemma_moe_gate_up.n_experts);
            });
            const bool projection_bundle_prefetched =
                prefetch_cached_moe_projection_bundle(
                    gemma_moe_gate_up, gemma_moe_down, route);
            mfq_tensor_backend::Tensor down_pair;
            const bool tracing_layer =
                g_gemma_stage_trace != nullptr && layer == g_gemma_trace_layer;
            if (!tracing_layer &&
                    gemma_moe_gate_up
                        .supports_projection_glu_epilogue()) {
                auto moe_hidden = g_profiler.measure("gemma.moe_gate_up_geglu", [&]() {
                    return gemma_moe_gate_up.forward_glu_output(moe_input, route, true);
                });
                if (!projection_bundle_prefetched) {
                    gemma_moe_down.prefetch(route);
                }
                down_pair = g_profiler.measure("gemma.moe_down", [&]() {
                    return gemma_moe_down.forward(moe_hidden, route);
                });
            } else {
                auto gate_up_pair = g_profiler.measure("gemma.moe_gate_up", [&]() {
                    return gemma_moe_gate_up.forward(moe_input, route);
                });
                if (!projection_bundle_prefetched) {
                    gemma_moe_down.prefetch(route);
                }
                trace_gemma_stage(layer, "moe_gate_up", gate_up_pair);
                if (tracing_layer || gate_up_pair.size(0) > 4) {
                    auto moe_hidden = g_profiler.measure("gemma.moe_geglu", [&]() {
                        return moe_geglu_split_cuda(gate_up_pair);
                    });
                    if (tracing_layer) trace_gemma_stage(layer, "moe_hidden", moe_hidden);
                    down_pair = g_profiler.measure("gemma.moe_down", [&]() {
                        return gemma_moe_down.forward(moe_hidden, route);
                    });
                } else {
                    down_pair = g_profiler.measure("gemma.moe_geglu_down", [&]() {
                        return gemma_moe_down.forward_geglu(gate_up_pair, route);
                    });
                }
            }
            trace_gemma_stage(layer, "moe_down", down_pair);
            auto moe_output = g_profiler.measure("gemma.moe_reduce", [&]() {
                return moe_weighted_reduce_cuda(down_pair, selected.at(1));
            });
            trace_gemma_stage(layer, "moe_reduce", moe_output);
            if (fused_norms) {
                auto result = g_profiler.measure("gemma.ffn_merge", [&]() {
                    return gemma4_ffn_merge_f16_cuda(
                        dense_output, moe_output, residual.reshape({B * T, H}),
                        ffn_post_norm_1, ffn_post_norm_2, ffn_post_norm,
                        layer_scale, rms_norm_eps).reshape({B, T, H});
                });
                return result;
            }
            moe_output = g_profiler.measure("gemma.ffn_post_norm_2", [&]() {
                return gemma_rms_norm_f16(
                    moe_output, ffn_post_norm_2,
                    rms_norm_eps, norm_weight_offset);
            });
            auto combined = g_profiler.measure("gemma.ffn_combine", [&]() {
                return dense_output + moe_output;
            });
            auto post = g_profiler.measure("gemma.ffn_post_norm", [&]() {
                return gemma_rms_norm_f16(
                    combined, ffn_post_norm,
                    rms_norm_eps, norm_weight_offset);
            });
            auto result = g_profiler.measure("gemma.ffn_residual", [&]() {
                return acc_cuda(residual.reshape({B * T, H}), post).reshape({B, T, H});
            });
            if (layer_scale.defined()) {
                result = g_profiler.measure("gemma.layer_scale", [&]() {
                    return result * layer_scale;
                });
            }
            trace_gemma_stage(layer, "layer_output", result);
            return result;
        }
        auto attn_pair = g_profiler.measure("full.attn_residual_ffn_norm", [&]() {
            auto rr = residual.reshape({-1, H});
            auto oo2 = oo.reshape({-1, H});
            if (!rr.is_cuda()) {
                auto summed = (rr.to(mfq_tensor_backend::kFloat32) + oo2.to(mfq_tensor_backend::kFloat32))
                    .to(rr.scalar_type()).contiguous();
                auto normalized = official_bf16
                    ? qwen_rms_norm_bf16(
                        summed, ffn_norm, rms_norm_eps, norm_weight_offset)
                    : qwen_rms_norm(
                        summed.to(mfq_tensor_backend::kFloat32), ffn_norm,
                        rms_norm_eps, norm_weight_offset);
                return std::vector<mfq_tensor_backend::Tensor>{summed, normalized};
            }
            if (oo2.scalar_type() != rr.scalar_type()) {
                oo2 = oo2.to(rr.scalar_type()).contiguous();
            }
            const char * fp32_residual_env =
                std::getenv("MFQ_DIAGNOSTIC_FP32_RESIDUAL");
            if (fp32_residual_env != nullptr &&
                    fp32_residual_env[0] == '1') {
                return acc_rms_norm_cuda(
                    rr.to(mfq_tensor_backend::kFloat32),
                    oo2.to(mfq_tensor_backend::kFloat32),
                    ffn_norm, rms_norm_eps,
                    norm_weight_offset);
            }
            if (rr.scalar_type() == mfq_tensor_backend::kFloat16 && oo2.scalar_type() == mfq_tensor_backend::kFloat16) {
                return acc_rms_norm_f16_cuda(
                    rr, oo2, ffn_norm, rms_norm_eps, norm_weight_offset);
            }
            if (rr.scalar_type() == mfq_tensor_backend::kBFloat16 &&
                    oo2.scalar_type() == mfq_tensor_backend::kBFloat16) {
                if (official_bf16) {
                    return acc_rms_norm_bf16_cuda(
                        rr, oo2, ffn_norm, rms_norm_eps,
                        norm_weight_offset);
                }
                auto sum = (rr + oo2).contiguous();
                auto norm = qwen_rms_norm(
                    sum.to(mfq_tensor_backend::kFloat32), ffn_norm,
                    rms_norm_eps, norm_weight_offset)
                    .to(mfq_tensor_backend::kBFloat16)
                    .contiguous();
                return std::vector<mfq_tensor_backend::Tensor>{sum, norm};
            }
            return acc_rms_norm_cuda(
                rr, oo2, ffn_norm, rms_norm_eps, norm_weight_offset);
        });
        x = attn_pair[0].reshape({B, T, H});
        residual = x;
        xn = attn_pair[1].reshape({B, T, H});
        trace_qwen_stage("qwen.ffn_norm", xn);
        if (!official_bf16) {
            auto ffn_input = xn.reshape({B * T, H});
            auto residual_flat = residual.reshape({B * T, H});
            if (ffn.can_forward_fused_residual(ffn_input, residual_flat)) {
                return g_profiler.measure("full.ffn_down_residual", [&]() {
                    return ffn.forward_fused_residual(
                        ffn_input, residual_flat).reshape({B, T, H});
                });
            }
        }
        mfq_tensor_backend::Tensor ff;
        if (official_bf16) {
            auto ffn_input = xn.reshape({B * T, H});
            auto gate_up = g_profiler.measure(
                "full.minicpmo45_ffn_gate_up",
                [&]() { return ffn.gate_up.forward(ffn_input); });
            MFQ_RUNTIME_CHECK(
                gate_up.size() == 2,
                "MiniCPM-o Qwen3 FFN requires separate Gate and Up outputs");
            auto gate = gate_up[0].to(mfq_tensor_backend::kBFloat16).contiguous();
            auto up = gate_up[1].to(mfq_tensor_backend::kBFloat16).contiguous();
            auto activation = g_profiler.measure(
                "full.minicpmo45_ffn_swiglu",
                [&]() {
                    const char * disabled = std::getenv(
                        "MFQ_DISABLE_MINICPM_BF16_SWIGLU_FUSION");
                    if (disabled == nullptr || disabled[0] != '1') {
                        return silu_mul_cuda(gate, up);
                    }
                    return (mfq_tensor_backend::silu(gate) * up).contiguous();
                });
            ff = g_profiler.measure(
                "full.minicpmo45_ffn_down",
                [&]() {
                    return ffn.down.forward_bf16_output(
                        activation);
                })
                .reshape({B, T, H})
                .to(mfq_tensor_backend::kBFloat16)
                .contiguous();
        } else {
            ff = ffn.forward(xn.reshape({B * T, H})).reshape({B, T, H});
        }
        trace_qwen_stage("qwen.ffn_output", ff);
        auto output = g_profiler.measure("full.ffn_residual", [&]() {
            auto rr = residual.reshape({-1, H});
            auto ff2 = ff.reshape({-1, H});
            if (!rr.is_cuda()) {
                return (rr.to(mfq_tensor_backend::kFloat32) + ff2.to(mfq_tensor_backend::kFloat32))
                    .to(rr.scalar_type()).reshape({B, T, H}).contiguous();
            }
            const char * fp32_residual_env =
                std::getenv("MFQ_DIAGNOSTIC_FP32_RESIDUAL");
            if (fp32_residual_env != nullptr &&
                    fp32_residual_env[0] == '1') {
                rr = rr.to(mfq_tensor_backend::kFloat32);
                ff2 = ff2.to(mfq_tensor_backend::kFloat32);
            } else if (ff2.scalar_type() != rr.scalar_type()) {
                ff2 = ff2.to(rr.scalar_type()).contiguous();
            }
            if (rr.scalar_type() == mfq_tensor_backend::kBFloat16 &&
                    ff2.scalar_type() == mfq_tensor_backend::kBFloat16) {
                const char* specialized_acc_disabled =
                    std::getenv("MFQ_DISABLE_MINICPM_BF16_RESIDUAL_ACC");
                if (specialized_acc_disabled == nullptr ||
                        specialized_acc_disabled[0] != '1') {
                    return acc_cuda(rr, ff2).reshape({B, T, H});
                }
                return (rr + ff2).contiguous().reshape({B, T, H});
            }
            return acc_cuda(rr, ff2).reshape({B, T, H});
        });
        return output;
    }
};

void prepare_ffn_workspaces(FFN & f);

FFN load_ffn(
    const mfq::ModelSource& source,
    const mfq::models::ModelConfig& config,
    int layer,
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
