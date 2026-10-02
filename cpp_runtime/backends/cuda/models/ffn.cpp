#include "ffn.h"
#include "models/common/gated_mlp.h"

#include "models/common/model_config.h"
#include "storage/moe_expert_cache.h"
#include "mfq_cuda_ops.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

bool FFN::uses_moe_expert_cache() const {
        if (!is_moe) return false;
        if (moe_split_gate_up) {
            return moe_gate.cached_source ||
                moe_up.cached_source ||
                moe_down.cached_source;
        }
        return moe_gate_up.cached_source || moe_down.cached_source;
    }

bool FFN::tensor_parallel_dense_compatible() const {
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

mfq_tensor_backend::Tensor FFN::forward_tensor_parallel_dense(
        CudaExecutionContext& execution,
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
                execution.config, launch_position, shard_count);
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
                    execution.model_parallel_collectives,
                    flat, gate_shard.device);
            auto gate_output =
                run_quant_linear_shard(
                    execution.profiler, execution.kl_mmq,
                    gate_shard, local_x);
            auto up_output =
                run_quant_linear_shard(
                    execution.profiler, execution.kl_mmq,
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
                execution.profiler, execution.kl_mmq,
                down_shard, activation);
        }
        auto output =
            reduce_model_parallel_outputs(
                execution, std::move(partials));
        shape.back() = output.size(-1);
        return output.reshape(shape);
    }

bool FFN::expert_parallel_moe_compatible() const {
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

mfq_tensor_backend::Tensor FFN::forward_expert_parallel_moe(
        CudaExecutionContext& execution,
        mfq_tensor_backend::Tensor x,
        const MoeRoutePlan & route,
        mfq_tensor_backend::Tensor route_weights) const {
        const size_t shard_count =
            moe_gate_up.expert_parallel_shards.size();
        std::vector<mfq_tensor_backend::Tensor> routed_partials(
            shard_count);
        std::vector<mfq_tensor_backend::Tensor> down_partials;
        const bool collect_output_energy =
            !execution.config.moe_route_stats_path.empty() &&
            execution.config.moe_route_output_energy;
        if (collect_output_energy) {
            down_partials.resize(shard_count);
        }
        for (size_t launch_position = 0;
             launch_position < shard_count; ++launch_position) {
            const size_t index = model_parallel_launch_index(
                execution.config, launch_position, shard_count);
            const auto & gate_shard =
                moe_gate_up.expert_parallel_shards[index];
            const auto & down_shard =
                moe_down.expert_parallel_shards[index];
            MfqCudaGuard guard(gate_shard.device);
            auto local_x = tensor_to_cuda_device(
                execution.model_parallel_collectives,
                x, gate_shard.device);
            auto local_weights = tensor_to_cuda_device(
                execution.model_parallel_collectives,
                route_weights, gate_shard.device);
            const auto& local_route = moe_route_to_device(
                execution.model_parallel_collectives,
                route, gate_shard.device);
            mfq_tensor_backend::Tensor gate_up_pair;
            mfq_tensor_backend::Tensor projected_hidden;
            if (swiglu_limit <= 0.0 &&
                    !execution.force_moe_materialized_swiglu &&
                    gate_shard.weight
                        ->supports_projection_glu_epilogue()) {
                projected_hidden = gate_shard.weight->forward_glu_output(
                    execution, local_x, local_route, false);
            } else {
                gate_up_pair = gate_shard.weight->forward(
                    execution, local_x, local_route);
            }
            mfq_tensor_backend::Tensor down_pair;
            const bool allow_fusion =
                gate_up_pair.defined() &&
                !execution.force_moe_materialized_swiglu &&
                execution.config.moe_swiglu_quant_fusion &&
                (gate_up_pair.size(0) == 1 ||
                 (gate_up_pair.size(0) <= 4 &&
                  execution.config.moe_small_heterogeneous));
            if (projected_hidden.defined()) {
                down_pair = down_shard.weight->forward(
                    execution, projected_hidden, local_route);
            } else if (swiglu_limit <= 0.0 && allow_fusion) {
                down_pair = down_shard.weight->forward_swiglu(
                    execution, gate_up_pair, local_route);
            } else if (swiglu_limit > 0.0 && allow_fusion &&
                    down_shard.weight->supports_clamped_swiglu()) {
                down_pair = down_shard.weight->forward_clamped_swiglu(
                    execution, gate_up_pair, local_route, swiglu_limit);
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
                    execution, hidden, local_route);
            }
            if (collect_output_energy) {
                down_partials[index] = down_pair;
            }
            routed_partials[index] = moe_weighted_reduce_cuda(
                down_pair, local_weights);
        }
        if (!execution.config.moe_route_stats_path.empty()) {
            mfq_tensor_backend::Tensor complete_down;
            if (collect_output_energy) {
                complete_down = reduce_model_parallel_outputs(
                    execution, std::move(down_partials));
            }
            record_moe_route_stats(
                execution, moe_layer, route.ids, route_weights,
                complete_down, moe_gate_up.n_experts);
        }
        return reduce_model_parallel_outputs(
            execution, std::move(routed_partials));
    }

mfq_tensor_backend::Tensor FFN::forward_dense_f32_down_kld(
        CudaExecutionContext& execution,
        mfq_tensor_backend::Tensor xh) const {
        auto& profiler = execution.profiler;
        MFQ_RUNTIME_CHECK(
            !is_moe && !tensor_parallel_dense_compatible() &&
            !geglu && swiglu_limit <= 0.0,
            "FP32-output IN diagnostic requires a local dense SiLU FFN");
        auto parts = profiler.measure(
            "ffn.gate_up", [&]() {
                return gate_up.forward(execution, xh);
            });
        return profiler.measure(
            "ffn.down.fp32_output", [&]() {
                return down.forward_input_mul_f32_kld(
                    execution, parts[1], parts[0], 2);
            });
    }

mfq_tensor_backend::Tensor FFN::forward_impl(
    CudaExecutionContext& execution,
    mfq_tensor_backend::Tensor x,
    MfqOptional<mfq_tensor_backend::Tensor> input_ids,
    bool allow_important_neurons) const {
        auto& profiler = execution.profiler;
        mfq_tensor_backend::Tensor xh;
        if (x.scalar_type() == mfq_tensor_backend::kFloat16) {
            xh = x;
        } else {
            xh = profiler.measure("ffn.input_cast", [&]() { return x.to(mfq_tensor_backend::kFloat16); });
        }
        if (allow_important_neurons && important_neurons) {
            if (is_moe) {
                throw std::runtime_error(
                    "important-neuron branches are only supported for dense FFNs");
            }
            const int64_t rows =
                xh.numel() / xh.size(-1);
            const bool use_f32_down =
                rows >= 16 &&
                execution.kl_mmq.mode == KlMmqMode::Fp16 &&
                execution.config.diagnostic_in_f32_down;
            if (use_f32_down) {
                auto low =
                    forward_dense_f32_down_kld(execution, xh);
                auto high =
                    important_neurons
                        ->forward_dense_f32_down_kld(execution, xh);
                return profiler.measure(
                    "ffn.in.combine.fp32", [&]() {
                        return (low + high)
                            .to(mfq_tensor_backend::kFloat16)
                            .contiguous();
                    });
            }
            auto run_branch = [&](size_t index) {
                return index == 0
                    ? forward_impl(execution, xh, input_ids, false)
                    : important_neurons->forward_impl(
                        execution, xh, input_ids, false);
            };
            std::vector<mfq_tensor_backend::Tensor> outputs;
            const bool parallel =
                decode_branch_parallel_enabled(
                    execution.config, execution.decode_graph_serial_branches, rows) &&
                execution.config.important_neuron_branch_parallel &&
                important_neuron_executor->run(
                    2, run_branch, outputs);
            auto low = parallel ? outputs[0] : run_branch(0);
            auto high = parallel ? outputs[1] : run_branch(1);
            return profiler.measure(
                "ffn.in.combine", [&]() {
                    return acc_cuda(
                        low.contiguous(), high.contiguous());
                });
        }
        if (tensor_parallel_dense_compatible()) {
            return profiler.measure(
                "ffn.tensor_parallel", [&]() {
                    return forward_tensor_parallel_dense(execution, xh);
                });
        }
        if (is_moe) {
            const int64_t rows = xh.numel() / xh.size(-1);
            if (execution.continuous_batch_cache_serial &&
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
                            execution, row_input, row_ids, false));
                    } else {
                        outputs.push_back(forward_impl(
                            execution, row_input, mfq_nullopt, false));
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
            auto xf32 = profiler.measure("moe.input_f32", [&]() {
                return xf.to(mfq_tensor_backend::kFloat32);
            });
            auto router_logits = profiler.measure("moe.router", [&]() {
                return mfq_tensor_backend::matmul(xf32, moe_router.transpose(0, 1));
            });
            mfq_tensor_backend::Tensor shared_gate_logits;
            std::vector<mfq_tensor_backend::Tensor> selected;
            if (moe_hash_ids.defined()) {
                if (!input_ids.has_value()) {
                    throw std::runtime_error(
                        "hash-routed MoE requires the current token ids");
                }
                selected = profiler.measure("moe.hash_route", [&]() {
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
                selected = profiler.measure("moe.topk", [&]() {
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
            auto route = profiler.measure("moe.route_map", [&]() {
                return build_moe_route_plan(
                    selected.at(0),
                    moe_split_gate_up ? moe_gate.n_experts : moe_gate_up.n_experts);
            });
            if (expert_parallel_moe_compatible()) {
                auto routed = profiler.measure(
                    "moe.expert_parallel", [&]() {
                        return forward_expert_parallel_moe(
                            execution, xf, route, selected.at(1));
                    });
                auto shared_output = shared->forward(execution, xf);
                auto shared_half = shared_output
                    .reshape({xf.size(0), xf.size(1)})
                    .contiguous().to(mfq_tensor_backend::kFloat16);
                if (moe_shared_ungated) {
                    return profiler.measure("moe.combine", [&]() {
                        return acc_cuda(
                            routed.contiguous(), shared_half);
                    });
                }
                shared_gate_logits = profiler.measure(
                    "moe.shared_gate", [&]() {
                        return mfq_tensor_backend::matmul(
                            xf32,
                            moe_shared_gate.transpose(0, 1))
                            .contiguous();
                    });
                return profiler.measure("moe.combine", [&]() {
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
            auto prefetch_projection_bundle = [&]() {
                if (!execution.config.moe_projection_bundle_prefetch ||
                        cpu_moe_down ||
                        execution.continuous_batch_cache_serial) {
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
                if (execution.config.moe_delayed_route_readback) {
                    moe_gate.prefetch_begin(route);
                } else {
                    projection_bundle_prefetched =
                        prefetch_projection_bundle();
                    if (!projection_bundle_prefetched) {
                        moe_gate.prefetch(route);
                    }
                }
            } else {
                if (execution.config.moe_delayed_route_readback) {
                    moe_gate_up.prefetch_begin(route);
                } else {
                    projection_bundle_prefetched =
                        prefetch_projection_bundle();
                    if (!projection_bundle_prefetched) {
                        moe_gate_up.prefetch(route);
                    }
                }
            }
            auto shared_output = profiler.measure(
                "moe.shared", [&]() {
                    return shared->forward(execution, xf);
                });
            if (!moe_split_gate_up && !moe_shared_ungated) {
                shared_gate_logits = profiler.measure(
                    "moe.shared_gate", [&]() {
                    return mfq_tensor_backend::matmul(
                        xf32,
                        moe_shared_gate.transpose(0, 1))
                        .contiguous();
                });
            }
            if (execution.config.moe_delayed_route_readback) {
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
                    staged_gate.emplace(profiler.measure(
                        "moe.cpu_offload_gate_h2d", [&]() {
                            return stage_cpu_mixed_moe(cpu_moe_gate, execution.config);
                        }));
                    active_gate = &staged_gate.value();
                }
                if (cpu_moe_up) {
                    staged_up.emplace(profiler.measure(
                        "moe.cpu_offload_up_h2d", [&]() {
                            return stage_cpu_mixed_moe(cpu_moe_up, execution.config);
                        }));
                    active_up = &staged_up.value();
                }
                gate_up_pair = profiler.measure("moe.gate_up_split", [&]() {
                    const bool reuse_gate_activation =
                        execution.config.split_moe_activation_reuse &&
                        execution.kl_mmq.mode ==
                            KlMmqMode::Default &&
                        xf.size(0) <= 8 &&
                        active_gate->can_reuse_activation_for(*active_up);
                    auto gate = active_gate->forward(execution, xf, route);
                    if (!projection_bundle_prefetched) {
                        active_up->prefetch(route);
                    }
                    if (!moe_shared_ungated &&
                            !shared_gate_logits.defined()) {
                        shared_gate_logits = profiler.measure(
                            "moe.shared_gate", [&]() {
                                return mfq_tensor_backend::matmul(
                                    xf32,
                                    moe_shared_gate.transpose(0, 1))
                                    .contiguous();
                            });
                    }
                    auto up = reuse_gate_activation
                        ? active_up->forward_prequantized(execution, xf, route)
                        : active_up->forward(execution, xf, route);
                    return mfq_tensor_backend::cat({gate, up}, -1).contiguous();
                });
            } else {
                if (cpu_moe_gate_up) {
                    staged_gate_up.emplace(profiler.measure(
                        "moe.cpu_offload_gate_up_h2d", [&]() {
                            return stage_cpu_mixed_moe(cpu_moe_gate_up, execution.config);
                        }));
                    active_gate_up = &staged_gate_up.value();
                }
                const bool fuse_projection_glu =
                    swiglu_limit <= 0.0 &&
                    !execution.force_moe_materialized_swiglu &&
                    active_gate_up->supports_projection_glu_epilogue();
                if (fuse_projection_glu) {
                    projected_hidden = profiler.measure(
                        "moe.gate_up_swiglu", [&]() {
                            return active_gate_up->forward_glu_output(
                                execution, xf, route, false);
                        });
                } else {
                    gate_up_pair = profiler.measure("moe.gate_up", [&]() {
                        return active_gate_up->forward(
                            execution, xf, route);
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
                staged_down.emplace(profiler.measure(
                    "moe.cpu_offload_down_h2d", [&]() {
                        return stage_cpu_mixed_moe(cpu_moe_down, execution.config);
                    }));
                active_down = &staged_down.value();
            }
            mfq_tensor_backend::Tensor down_pair;
            const bool allow_swiglu_quant_fusion =
                gate_up_pair.defined() &&
                !execution.force_moe_materialized_swiglu &&
                execution.config.moe_swiglu_quant_fusion &&
                (gate_up_pair.size(0) == 1 ||
                 (gate_up_pair.size(0) <= 4 &&
                  execution.config.moe_small_heterogeneous));
            if (projected_hidden.defined()) {
                down_pair = profiler.measure("moe.down", [&]() {
                    return active_down->forward(
                        execution, projected_hidden, route);
                });
            } else if (swiglu_limit <= 0.0 && allow_swiglu_quant_fusion) {
                down_pair = profiler.measure("moe.swiglu_down", [&]() {
                    return active_down->forward_swiglu(
                        execution, gate_up_pair, route);
                });
            } else if (swiglu_limit > 0.0 && allow_swiglu_quant_fusion &&
                    active_down->supports_clamped_swiglu()) {
                down_pair = profiler.measure("moe.swiglu_down", [&]() {
                    return active_down->forward_clamped_swiglu(
                        execution, gate_up_pair, route, swiglu_limit);
                });
            } else {
                auto hidden = profiler.measure("moe.swiglu", [&]() {
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
                down_pair = profiler.measure("moe.down", [&]() {
                    return active_down->forward(execution, hidden, route);
                });
            }
            record_moe_route_stats(
                execution, moe_layer, selected.at(0), selected.at(1), down_pair,
                moe_split_gate_up ? moe_gate.n_experts : moe_gate_up.n_experts);
            staged_down.reset();
            const bool fuse_reduce_gate =
                !execution.force_moe_unfused_reduce &&
                execution.config.moe_reduce_gate_fusion && !moe_shared_ungated &&
                down_pair.size(0) <= 8;
            mfq_tensor_backend::Tensor routed;
            if (!fuse_reduce_gate) {
                routed = profiler.measure("moe.reduce", [&]() {
                    return moe_weighted_reduce_cuda(down_pair, selected.at(1));
                });
            }
            if (!moe_shared_ungated && !shared_gate_logits.defined()) {
                shared_gate_logits = profiler.measure("moe.shared_gate", [&]() {
                    return mfq_tensor_backend::matmul(xf32, moe_shared_gate.transpose(0, 1)).contiguous();
                });
            }
            auto shared_half = shared_output.reshape({xf.size(0), xf.size(1)}).contiguous().to(mfq_tensor_backend::kFloat16);
            if (moe_shared_ungated) {
                return profiler.measure("moe.combine", [&]() {
                    return acc_cuda(routed.contiguous(), shared_half);
                });
            }
            if (fuse_reduce_gate) {
                return profiler.measure("moe.reduce_combine", [&]() {
                    return moe_weighted_reduce_shared_gate_cuda(
                        down_pair, selected.at(1), shared_half, shared_gate_logits);
                });
            }
            return profiler.measure("moe.combine", [&]() {
                return moe_add_shared_gate_cuda(
                    routed.contiguous(), shared_half,
                    shared_gate_logits);
            });
        }
        using Tensor = mfq_tensor_backend::Tensor;
        using Activation = mfq::models::GatedActivation;
        return mfq::models::gated_mlp(xh, geglu, swiglu_limit,
            [&](const Tensor& input, Activation activation, double limit) -> std::optional<Tensor> {
                if (activation != Activation::silu || limit > 0.0) return {};
                if (nvq_fusion_enabled(execution.config) && input.numel() / input.size(-1) == 1 &&
                    gate_up.nvq_prefix2 && gate_up.layers.size() == 2 &&
                    gate_up.layers[0].is_nvq() && gate_up.layers[1].is_nvq() && down.is_nvq() &&
                    gate_up.outs.size() == 2 && gate_up.outs[0] == gate_up.outs[1] &&
                    gate_up.outs[0] == down.nvq.neuron_len &&
                    (down.nvq.gs == 24 || down.nvq.gs == 28 || down.nvq.gs == 32)) {
                    auto shape = input.sizes().vec();
                    shape.back() = down.nvq.out;
                    return nvq_ffn_swiglu_down(profiler, gate_up.layers[0].nvq,
                        gate_up.layers[1].nvq, down.nvq, input.reshape({-1, input.size(-1)})).reshape(shape);
                }
                return {};
            },
            [&](const Tensor& input, Activation activation, double limit) -> std::optional<Tensor> {
                if (limit > 0.0 || !gate_up.nint_grouped || !gate_up.nint.split_w.empty()) return {};
                const auto rows = input.numel() / input.size(-1);
                if (activation == Activation::gelu && execution.config.ffn_geglu_fusion && rows == 1)
                    return profiler.measure("ffn.gate_up_geglu", [&] { return gate_up.forward_geglu(execution, input); });
                if (activation == Activation::silu && execution.config.ffn_swiglu_fusion && rows >= 1 && rows <= 6 &&
                    gate_up.outs.size() == 2 && gate_up.outs[0] == gate_up.outs[1])
                    return profiler.measure("ffn.gate_up_swiglu", [&] { return gate_up.forward_swiglu(execution, input); });
                return {};
            },
            [&](Tensor input) { return profiler.measure("ffn.gate_up", [&] { return gate_up.forward(execution, input); }); },
            [&](Tensor gate, Tensor up, Activation activation, double limit) {
                if (activation == Activation::gelu)
                    return profiler.measure("ffn.geglu", [&] { return gelu_mul_cuda(gate.contiguous(), up.contiguous()); });
                if (limit > 0.0)
                    return profiler.measure("ffn.swiglu_clamped", [&] {
                        auto g = mfq_tensor_backend::clamp_max(gate.to(mfq_tensor_backend::kFloat32), limit);
                        auto u = mfq_tensor_backend::clamp(up.to(mfq_tensor_backend::kFloat32), -limit, limit);
                        return (mfq_tensor_backend::silu(g) * u).to(mfq_tensor_backend::kFloat16).contiguous();
                    });
                return profiler.measure("ffn.swiglu", [&] { return (up * mfq_tensor_backend::silu(gate)).contiguous(); });
            },
            [&](Tensor hidden) { return profiler.measure("ffn.down", [&] { return down.forward(execution, hidden); }); },
            [&](const Tensor& gate, const Tensor& up, Activation activation, double limit) -> std::optional<Tensor> {
                if (activation != Activation::silu || limit > 0.0 || down.is_dense() || down.is_mxfp8()) return {};
                return profiler.measure("ffn.down", [&] { return down.forward_input_mul(execution, up, gate, 2); });
            });
    }

bool FFN::can_forward_fused_residual(
    const CudaExecutionConfig& config,
    const mfq_tensor_backend::Tensor & x,
    const mfq_tensor_backend::Tensor & residual) const {
        if (config.diagnostic_fp32_residual) return false;
        if (!nvq_fusion_enabled(config) || is_moe || geglu ||
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
            gate_up.outs[0] != down.nvq.neuron_len ||
            x.size(-1) != gate_up.layers[0].nvq.neuron_len ||
            residual.size(-1) != down.nvq.out) {
            return false;
        }
        return nvq_fused_residual_format(
            down.nvq.kernel_format);
    }

mfq_tensor_backend::Tensor FFN::forward_fused_residual(
    CudaProfiler& profiler,
    const CudaExecutionConfig& config,
    mfq_tensor_backend::Tensor x,
    mfq_tensor_backend::Tensor residual) const {
        MFQ_RUNTIME_CHECK(
            can_forward_fused_residual(config, x, residual),
            "FFN fused residual requires a compatible single-token NVQ FFN");
        auto shape = x.sizes().vec();
        shape.back() = down.nvq.out;
        auto output = nvq_ffn_swiglu_down(
            profiler, gate_up.layers[0].nvq,
            gate_up.layers[1].nvq, down.nvq,
            x.reshape({1, x.size(-1)}),
            residual.reshape({1, residual.size(-1)}));
        return output.reshape(shape);
    }

mfq_tensor_backend::Tensor FFN::forward(
    CudaExecutionContext& execution,
    mfq_tensor_backend::Tensor x,
    MfqOptional<mfq_tensor_backend::Tensor> input_ids) const {
        return forward_impl(
            execution, std::move(x), input_ids, true);
    }

void prepare_ffn_workspaces(
        CudaExecutionContext& execution,
        FFN& f) {
    if (execution.loading_cpu_layer) return;
    if (f.down.tensor_parallel()) return;
    if (f.gate_up.nvq_prefix2 && f.gate_up.layers.size() == 2 && f.down.is_nvq() &&
        f.gate_up.outs.size() == 2 && f.gate_up.outs[0] == f.gate_up.outs[1] &&
        f.gate_up.outs[0] == f.down.nvq.neuron_len) {
        NvqWorkspace & ws = f.gate_up.layers[0].nvq.workspace(1);
        ws.swiglu_scratch = mfq_tensor_backend::empty(
            {f.gate_up.outs[0]}, mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA).dtype(mfq_tensor_backend::kFloat32));
        (void)f.down.nvq.workspace(1);
    }
    if (f.important_neurons) {
        prepare_ffn_workspaces(execution, *f.important_neurons);
    }
}

void load_important_neuron_branch(
        CudaExecutionContext& execution,
        const mfq::ModelSource & mfq,
        int64_t hidden_size,
        int64_t intermediate_size,
        FFN & f,
        const std::string & down_name,
        const std::string & gate_name,
        const std::string & up_name) {
    const std::string down_high = down_name + ".in_high";
    const std::string gate_high = gate_name + ".in_high";
    const std::string up_high = up_name + ".in_high";
    const bool has_down = has_tensor(mfq, down_high);
    const bool has_gate = has_tensor(mfq, gate_high);
    const bool has_up = has_tensor(mfq, up_high);
    if (!has_down && !has_gate && !has_up) {
        return;
    }
    if (!has_down || !has_gate || !has_up) {
        throw std::runtime_error(
            "important-neuron FFN requires matching gate/up/down .in_high records");
    }
    if (f.is_moe) {
        throw std::runtime_error(
            "important-neuron records are unsupported on routed MoE FFNs");
    }

    auto high = std::make_unique<FFN>();
    high->down = load_quant_linear(execution, mfq, down_high, TensorParallelAxis::Input);
    high->gate_up = load_paired_gate_up(execution, mfq, {gate_high, up_high}, high->down);
    high->geglu = f.geglu;
    high->swiglu_limit = f.swiglu_limit;

    if (f.gate_up.outs.size() != 2 ||
        high->gate_up.outs.size() != 2 ||
        f.gate_up.outs[0] != f.gate_up.outs[1] ||
        high->gate_up.outs[0] != high->gate_up.outs[1] ||
        f.down.out() != hidden_size ||
        high->down.out() != hidden_size ||
        f.down.neuron_len() != f.gate_up.outs[0] ||
        high->down.neuron_len() != high->gate_up.outs[0] ||
        f.down.neuron_len() + high->down.neuron_len() !=
            intermediate_size) {
        throw std::runtime_error(
            "important-neuron FFN tensor shapes disagree with model config");
    }
    f.important_neurons = std::move(high);
}

FFN load_moe_weights(
        CudaExecutionContext& execution,
        const mfq::ModelSource& source,
        std::string_view prefix,
        const MoeWeightLoadOptions& options) {
    const std::string base(prefix);
    const std::string gate_up = base + "experts.gate_up.weight";
    const std::string gate = base + "experts.gate.weight";
    const std::string up = base + "experts.up.weight";
    const std::string down = base + "experts.down.weight";
    const bool fused = has_tensor(source, gate_up);
    const bool split_gate = has_tensor(source, gate);
    const bool split_up = has_tensor(source, up);
    if (split_gate != split_up || fused == split_gate ||
            !has_tensor(source, down)) {
        throw std::runtime_error(
            "routed MoE requires down and exactly one fused or split "
            "Gate/Up representation at layer " +
            std::to_string(options.layer));
    }

    FFN result;
    result.is_moe = true;
    result.moe_split_gate_up = split_gate;
    result.moe_layer = options.layer;
    if (options.cpu_offloaded) {
        if (split_gate) {
            result.cpu_moe_gate = load_mfe_cpu_offloaded(source, gate);
            result.cpu_moe_up = load_mfe_cpu_offloaded(source, up);
            result.moe_gate = cpu_mixed_moe_metadata(result.cpu_moe_gate);
            result.moe_up = cpu_mixed_moe_metadata(result.cpu_moe_up);
        } else {
            result.cpu_moe_gate_up = load_mfe_cpu_offloaded(source, gate_up);
            result.moe_gate_up =
                cpu_mixed_moe_metadata(result.cpu_moe_gate_up);
        }
        result.cpu_moe_down = load_mfe_cpu_offloaded(source, down);
        result.moe_down = cpu_mixed_moe_metadata(result.cpu_moe_down);
    } else {
        if (split_gate) {
            result.moe_gate = load_mfe_gpu(
                execution, source, gate, true, options.layer, "gate");
            result.moe_up = load_mfe_gpu(
                execution, source, up, true, options.layer, "up");
        } else {
            result.moe_gate_up = load_mfe_gpu(
                execution, source, gate_up, true, options.layer, "gate_up");
        }
        result.moe_down = load_mfe_gpu(
            execution, source, down, true, options.layer, "down");
    }
    result.moe_router = load_dense_gpu(
        execution, source, base + "router.weight")
        .to(mfq_tensor_backend::kFloat32).contiguous();
    const std::string router_bias = base + "router.bias";
    if (options.router_bias_required || has_tensor(source, router_bias)) {
        result.moe_router_bias = load_dense_gpu(
            execution, source, router_bias)
            .to(mfq_tensor_backend::kFloat32).contiguous();
    }
    result.shared = std::make_unique<FFN>();
    result.shared->down = load_quant_linear(
        execution, source, base + "shared_expert.down.weight");
    result.shared->gate_up = load_paired_gate_up(
        execution, source,
        {base + "shared_expert.gate.weight",
         base + "shared_expert.up.weight"},
        result.shared->down,
        options.shared_gate_up_compatible_prefix);
    prepare_ffn_workspaces(execution, *result.shared);
    return result;
}

FFN load_ffn(
        CudaExecutionContext& execution,
        const mfq::ModelSource& source,
        const mfq::models::ModelConfig& config,
        int layer,
        bool minicpmo45,
        std::string_view tensor_root) {
    FFN ffn;
    const std::string prefix =
        std::string(tensor_root) + ".block." +
        std::to_string(layer) + ".mlp.";
    const std::string down = prefix + "down.weight";
    const std::string gate = prefix + "gate.weight";
    const std::string up = prefix + "up.weight";
    ffn.down = load_quant_linear(execution, source, down);
    ffn.gate_up = load_paired_gate_up(execution, source, {gate, up}, ffn.down, 2, minicpmo45);
    load_important_neuron_branch(
        execution, source, config.hidden_size, config.intermediate_size,
        ffn, down, gate, up);
    prepare_ffn_workspaces(execution, ffn);
    return ffn;
}
