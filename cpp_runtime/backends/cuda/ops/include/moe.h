#pragma once

#include "cuda_execution.h"
#include "fp8_sq.h"
#include "mx.h"
#include "mxfp4_sq.h"
#include "nint.h"
#include "vq.h"
#include "../legacy/tpq/tpq.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mfq::cuda {
class MfeMxfp4ExpertStore;
}

class MoeCachedSource;
struct MixedMoeRuntime;
struct MoeRoutePlan;

mfq_tensor_backend::Tensor moe_tensor_to_device(
    mfq_tensor_backend::Tensor value,
    int device);
MoeRoutePlan moe_route_to_device(const MoeRoutePlan& route, int device);
mfq_tensor_backend::Tensor reduce_model_parallel_outputs(
    std::vector<mfq_tensor_backend::Tensor> outputs);
int select_nint_prefill_route_tile(
    const MoeRoutePlan& route, int tokens, int routes, int experts);
const mfq_tensor_backend::Tensor& nint_route_tile_bounds(
    const MoeRoutePlan& route, int tile_m);
const mfq_tensor_backend::Tensor& nint_route_tile_experts(
    const MoeRoutePlan& route, int tile_m);

struct MfePoolWeight {
    NintWeight weight;
    mfq_tensor_backend::Tensor expert_local;
    int local_experts = 0;
};

struct MoeActivationKey {
    int input_rows = 0;
    int groups = 0;
    int gs = 0;
    int device = 0;

    bool operator==(const MoeActivationKey & other) const {
        return input_rows == other.input_rows && groups == other.groups &&
               gs == other.gs && device == other.device;
    }
};

struct MoeActivationKeyHash {
    size_t operator()(const MoeActivationKey & key) const {
        size_t value = (size_t)key.input_rows;
        value = value * 1315423911u + (size_t)key.groups;
        value = value * 1315423911u + (size_t)key.gs;
        return value * 1315423911u + (size_t)key.device;
    }
};

struct MoeActivationWorkspace {
    mfq_tensor_backend::Tensor qx;
    mfq_tensor_backend::Tensor xscale;
};

struct MoeActivationGeometry {
    int groups = 0;
    int gs = 0;
    int transform_block = 0;
    uint64_t transform_seed = 0;

    bool operator==(const MoeActivationGeometry & other) const {
        return groups == other.groups && gs == other.gs &&
            transform_block == other.transform_block &&
            transform_seed == other.transform_seed;
    }
};

struct MoeRoutePlan {
    mfq_tensor_backend::Tensor ids;
    mfq_tensor_backend::Tensor ids_dst;
    mfq_tensor_backend::Tensor expert_bounds;
    mfq_tensor_backend::Tensor tile_bounds;
    mfq_tensor_backend::Tensor tile_experts;
    mfq_tensor_backend::Tensor mma_tile_bounds;
    mfq_tensor_backend::Tensor mma_tile_experts;
    mfq_tensor_backend::Tensor wide_tile_bounds;
    mfq_tensor_backend::Tensor wide_tile_experts;
    mfq_tensor_backend::Tensor counts;
    mfq_tensor_backend::Tensor cursors;
    int n_experts = 0;
    int mma_tile_m = 8;
    int wide_tile_m = 8;
    bool map_ready = false;
    uint64_t generation = 0;
    mutable std::shared_ptr<std::vector<int32_t>>
        host_unique_experts;
};

struct MfeWeight {
    struct ExpertParallelShard {
        int device = 0;
        int64_t expert_begin = 0;
        int64_t expert_end = 0;
        std::shared_ptr<MfeWeight> weight;
    };

    int n_experts = 0;
    int out_per_expert = 0;
    int neuron_len = 0;
    std::vector<MfePoolWeight> pools;
    int64_t mixed_weight_bytes = 0;
    bool partial_experts = false;
    bool unified_nint_projection = false;
    std::vector<ExpertParallelShard>
        expert_parallel_shards;
    std::function<void(const MoeRoutePlan &)> cache_prefetch;
    std::function<void(const MoeRoutePlan &)> cache_prefetch_begin;
    std::function<mfq_tensor_backend::Tensor(mfq_tensor_backend::Tensor, const MoeRoutePlan &)> mixed_forward;
    std::function<mfq_tensor_backend::Tensor(mfq_tensor_backend::Tensor, const MoeRoutePlan &)>
        mixed_prequantized_forward;
    std::function<mfq_tensor_backend::Tensor(
        mfq_tensor_backend::Tensor, const MoeRoutePlan &, bool)> mixed_glu_output_forward;
    std::function<mfq_tensor_backend::Tensor(
        mfq_tensor_backend::Tensor, const MoeRoutePlan &, bool)> mixed_glu_forward;
    std::function<mfq_tensor_backend::Tensor(
        mfq_tensor_backend::Tensor, const MoeRoutePlan &, double)> mixed_clamped_swiglu_forward;
    std::vector<MoeActivationGeometry> activation_geometries;
    int activation_workspace_domain = 0;
    std::shared_ptr<MoeCachedSource> cached_source;

    MoeActivationWorkspace & activation_workspace(
            mfq_tensor_backend::Tensor x, int input_rows, int groups, int gs) const {
        // Decode Gate and Up consume the same activation.  A process thread
        // therefore owns one stable workspace per activation geometry; the
        // explicit forward_prequantized() entry point below is the only path
        // allowed to reuse its contents without launching quantization again.
        static thread_local std::unordered_map<
            MoeActivationKey, MoeActivationWorkspace, MoeActivationKeyHash>
            shared_activation_workspaces;
        MoeActivationKey key{input_rows, groups, gs, x.get_device()};
        auto it = shared_activation_workspaces.find(key);
        if (it != shared_activation_workspaces.end()) return it->second;
        MoeActivationWorkspace workspace;
        workspace.qx = mfq_tensor_backend::empty(
            {input_rows, groups * gs}, x.options().dtype(mfq_tensor_backend::kInt8));
        workspace.xscale = mfq_tensor_backend::empty(
            {input_rows, groups}, x.options().dtype(mfq_tensor_backend::kFloat32));
        return shared_activation_workspaces.emplace(
            key, std::move(workspace)).first->second;
    }

    bool expert_parallel() const {
        return !expert_parallel_shards.empty();
    }

    bool supports_projection_glu_epilogue() const {
        if (!expert_parallel()) {
            return unified_nint_projection || !pools.empty();
        }
        return std::all_of(
            expert_parallel_shards.begin(),
            expert_parallel_shards.end(),
            [](const ExpertParallelShard & shard) {
                return shard.weight &&
                    shard.weight->supports_projection_glu_epilogue();
            });
    }

    template <typename Forward>
    mfq_tensor_backend::Tensor forward_expert_parallel(
            mfq_tensor_backend::Tensor x,
            const MoeRoutePlan & route,
            Forward && forward) const {
        std::vector<mfq_tensor_backend::Tensor> outputs(
            expert_parallel_shards.size());
        for (size_t launch_position = 0;
             launch_position < expert_parallel_shards.size();
             ++launch_position) {
            const size_t index = model_parallel_launch_index(
                launch_position, expert_parallel_shards.size());
            const auto & shard = expert_parallel_shards[index];
            if (!shard.weight) {
                throw std::runtime_error(
                    "expert-parallel MoE shard is missing");
            }
            MfqCudaGuard guard(shard.device);
            auto local_x =
                moe_tensor_to_device(x, shard.device);
            auto local_route =
                moe_route_to_device(route, shard.device);
            outputs[index] = forward(
                *shard.weight,
                local_x,
                local_route);
        }
        return reduce_model_parallel_outputs(
            std::move(outputs));
    }

    void prefetch(const MoeRoutePlan & route) const {
        if (cache_prefetch) cache_prefetch(route);
    }

    void prefetch_begin(const MoeRoutePlan & route) const {
        if (cache_prefetch_begin) cache_prefetch_begin(route);
    }

    bool supports_prequantized_input() const {
        return !expert_parallel() &&
            (!pools.empty() || static_cast<bool>(mixed_prequantized_forward));
    }

    bool can_reuse_activation_for(
            const MfeWeight & consumer) const {
        if (!supports_prequantized_input() ||
                !consumer.supports_prequantized_input() ||
                neuron_len != consumer.neuron_len ||
                activation_workspace_domain == 0 ||
                activation_workspace_domain !=
                    consumer.activation_workspace_domain) {
            return false;
        }
        if (activation_geometries.empty() ||
                consumer.activation_geometries.empty()) {
            return false;
        }
        return std::all_of(
            consumer.activation_geometries.begin(),
            consumer.activation_geometries.end(),
            [&](const MoeActivationGeometry & required) {
                return std::find(
                    activation_geometries.begin(),
                    activation_geometries.end(),
                    required) != activation_geometries.end();
            });
    }

    mfq_tensor_backend::Tensor forward(
            mfq_tensor_backend::Tensor x,
            const MoeRoutePlan & route) const {
        return forward_impl(x, route, false);
    }

    mfq_tensor_backend::Tensor forward_prequantized(
            mfq_tensor_backend::Tensor x,
            const MoeRoutePlan & route) const {
        return forward_impl(x, route, true);
    }

    mfq_tensor_backend::Tensor forward_impl(
            mfq_tensor_backend::Tensor x,
            const MoeRoutePlan & route,
            bool input_prequantized,
            int epilogue_mode = 0) const {
        if (expert_parallel()) {
            if (input_prequantized) {
                throw std::runtime_error(
                    "prequantized MoE activation reuse is unavailable with expert parallelism");
            }
            return forward_expert_parallel(
                x, route,
                [epilogue_mode](const MfeWeight & shard,
                   mfq_tensor_backend::Tensor local_x,
                   const MoeRoutePlan & local_route) {
                    return shard.forward_impl(
                        local_x, local_route, false, epilogue_mode);
                });
        }
        if (epilogue_mode < 0 || epilogue_mode > 2 ||
                (epilogue_mode != 0 && (out_per_expert % 2) != 0)) {
            throw std::runtime_error("invalid MFE NINT projection epilogue");
        }
        if (input_prequantized && mixed_prequantized_forward) {
            if (epilogue_mode != 0) {
                throw std::runtime_error(
                    "mixed prequantized MFE GLU epilogue is unavailable");
            }
            return mixed_prequantized_forward(x, route);
        }
        if (mixed_forward) {
            if (epilogue_mode != 0) {
                throw std::runtime_error(
                    "mixed MFE GLU epilogue must use its native dispatch");
            }
            return mixed_forward(x, route);
        }
        if (!x.is_cuda() || !x.is_contiguous() || x.scalar_type() != mfq_tensor_backend::kFloat16 ||
            (x.dim() != 2 && x.dim() != 3) || x.size(-1) != neuron_len) {
            throw std::runtime_error("MFE input must be contiguous CUDA f16 with exact K");
        }
        int tokens = (int)route.ids.size(0);
        int routes = (int)route.ids.size(1);
        if (route.n_experts != n_experts || x.size(0) != tokens ||
            (x.dim() == 3 && x.size(1) != routes)) {
            throw std::runtime_error("MFE input and route shape mismatch");
        }
        int input_rows = x.dim() == 3 ? tokens * routes : tokens;
        const int result_width = epilogue_mode == 0
            ? out_per_expert
            : out_per_expert / 2;
        auto output = partial_experts
            ? mfq_tensor_backend::zeros(
                {tokens, routes, result_width},
                x.options().dtype(mfq_tensor_backend::kFloat16))
            : mfq_tensor_backend::empty(
                {tokens, routes, result_width},
                x.options().dtype(mfq_tensor_backend::kFloat16));
        std::unordered_set<MoeActivationKey, MoeActivationKeyHash> quantized;
        int nint_pool_phase = 0;
        for (const auto & pool : pools) {
            int groups = (int)pool.weight.ng;
            int gs = (int)pool.weight.gs;
            MoeActivationKey key{input_rows, groups, gs, x.get_device()};
            auto & workspace = activation_workspace(x, input_rows, groups, gs);
            bool input_quantized = input_prequantized ||
                quantized.find(key) != quantized.end();
            const int route_tile_m = select_nint_prefill_route_tile(
                route, tokens, routes, n_experts);
            mfe_nint_matmul_ws_cuda(
                pool.weight.q_packed, pool.weight.row_q_bits,
                pool.weight.row_q_bit_offsets, pool.weight.sub_scale,
                pool.weight.sub_min, pool.weight.neuron_scale,
                pool.weight.neuron_min, x, route.ids, pool.expert_local,
                n_experts, pool.local_experts, out_per_expert, gs,
                epilogue_mode, route.map_ready, input_quantized,
                output, workspace.qx,
                workspace.xscale, route.ids_dst, route.expert_bounds,
                nint_route_tile_bounds(route, route_tile_m),
                nint_route_tile_experts(route, route_tile_m),
                route_tile_m, nint_pool_phase++);
            if (input_quantized || route_tile_m == 8) {
                quantized.insert(key);
            }
        }
        return output;
    }

    mfq_tensor_backend::Tensor forward_glu_output(
            mfq_tensor_backend::Tensor x, const MoeRoutePlan & route, bool gelu) const {
        if (expert_parallel()) {
            return forward_expert_parallel(
                x, route,
                [gelu](
                    const MfeWeight & shard,
                    mfq_tensor_backend::Tensor local_x,
                    const MoeRoutePlan & local_route) {
                    return shard.forward_glu_output(
                        local_x, local_route, gelu);
                });
        }
        if (mixed_glu_output_forward) {
            return mixed_glu_output_forward(x, route, gelu);
        }
        if (mixed_forward) {
            auto gate_up = mixed_forward(x, route);
            return gelu
                ? moe_geglu_split_cuda(gate_up)
                : moe_swiglu_split_cuda(gate_up);
        }
        return forward_impl(x, route, false, gelu ? 2 : 1);
    }

    mfq_tensor_backend::Tensor forward_glu(
            mfq_tensor_backend::Tensor gate_up, const MoeRoutePlan & route, bool gelu) const {
        if (expert_parallel()) {
            return forward_expert_parallel(
                gate_up, route,
                [gelu](
                    const MfeWeight & shard,
                    mfq_tensor_backend::Tensor local_gate_up,
                    const MoeRoutePlan & local_route) {
                    return shard.forward_glu(
                        local_gate_up,
                        local_route, gelu);
                });
        }
        if (mixed_glu_forward) {
            return mixed_glu_forward(gate_up, route, gelu);
        }
        if (!gate_up.is_cuda() || !gate_up.is_contiguous() ||
                gate_up.scalar_type() != mfq_tensor_backend::kFloat16 ||
                gate_up.dim() != 3 || gate_up.size(2) != 2 * neuron_len) {
            throw std::runtime_error("fused MFE GLU input has an unsupported layout");
        }
        auto activation = gelu
            ? moe_geglu_split_cuda(gate_up)
            : moe_swiglu_split_cuda(gate_up);
        return forward(activation, route);
    }

    mfq_tensor_backend::Tensor forward_swiglu(mfq_tensor_backend::Tensor gate_up, const MoeRoutePlan & route) const {
        return forward_glu(gate_up, route, false);
    }

    bool supports_clamped_swiglu() const {
        if (expert_parallel()) {
            return std::all_of(
                expert_parallel_shards.begin(),
                expert_parallel_shards.end(),
                [](const ExpertParallelShard & shard) {
                    return shard.weight &&
                        shard.weight
                            ->supports_clamped_swiglu();
                });
        }
        return !pools.empty() ||
            static_cast<bool>(mixed_clamped_swiglu_forward);
    }

    mfq_tensor_backend::Tensor forward_clamped_swiglu(
            mfq_tensor_backend::Tensor gate_up,
            const MoeRoutePlan & route,
            double limit) const {
        if (expert_parallel()) {
            return forward_expert_parallel(
                gate_up, route,
                [limit](
                    const MfeWeight & shard,
                    mfq_tensor_backend::Tensor local_gate_up,
                    const MoeRoutePlan & local_route) {
                    return shard.forward_clamped_swiglu(
                        local_gate_up,
                        local_route, limit);
                });
        }
        if (mixed_clamped_swiglu_forward) {
            return mixed_clamped_swiglu_forward(
                gate_up, route, limit);
        }
        if (pools.empty() || !gate_up.is_cuda() ||
                !gate_up.is_contiguous() ||
                gate_up.scalar_type() != mfq_tensor_backend::kFloat16 ||
                gate_up.dim() != 3 ||
                gate_up.size(2) != 2 * neuron_len || limit <= 0.0) {
            throw std::runtime_error(
                "clamped SwiGLU is unavailable for this MFE tensor");
        }
        const int64_t width = gate_up.size(2) / 2;
        auto gate = mfq_tensor_backend::clamp_max(
            gate_up.slice(2, 0, width)
                .to(mfq_tensor_backend::kFloat32),
            limit);
        auto up = mfq_tensor_backend::clamp(
            gate_up.slice(2, width, 2 * width)
                .to(mfq_tensor_backend::kFloat32),
            -limit, limit);
        auto activation = (mfq_tensor_backend::silu(gate) * up)
            .to(mfq_tensor_backend::kFloat16).contiguous();
        return forward(activation, route);
    }

    mfq_tensor_backend::Tensor forward_geglu(mfq_tensor_backend::Tensor gate_up, const MoeRoutePlan & route) const {
        return forward_glu(gate_up, route, true);
    }
};

struct MfeCpuPool {
    std::vector<int32_t> expert_ids;
    std::string dtype;
    std::vector<uint8_t> payload;
    std::vector<uint8_t> runtime_payload;
    NintCpu weight;
    Nint8ZeroCpu q8_zero;
    Mxfp4Cpu mxfp4;
    TpqCpu tpq;
};

struct MfeCpu {
    int n_experts = 0;
    int out_per_expert = 0;
    int neuron_len = 0;
    std::vector<MfeCpuPool> pools;
};

enum class MixedMoeFamily {
    Nint,
    Nint8Zero,
    Mxfp4,
    Mxfp4Sq,
    Fp8Sq,
    Tpq,
    Nvq,
    Nepq,
};

struct MixedMoePool {
    MixedMoeFamily family = MixedMoeFamily::Nint;
    NintWeight nint;
    NintWeight q8_zero;
    Mxfp4Weight mxfp4;
    Mxfp4SqWeight mxfp4_sq;
    Fp8SqWeight fp8_sq;
    TpqWeight tpq;
    NvqWeight nvq;
    NepqWeight nepq;
    mfq_tensor_backend::Tensor expert_local;
    int local_experts = 0;
};

void fp8_sq_moe_matmul(
    const Fp8SqWeight& weight,
    mfq_tensor_backend::Tensor input,
    mfq_tensor_backend::Tensor expert_ids,
    mfq_tensor_backend::Tensor expert_local,
    int n_experts,
    int local_experts,
    int out_per_expert,
    int neuron_len,
    mfq_tensor_backend::Tensor output);

struct MixedMoeTransformKey {
    int block = 0;
    uint64_t seed = 0;

    bool operator==(const MixedMoeTransformKey & other) const {
        return block == other.block && seed == other.seed;
    }
};

struct MixedMoeTransformKeyHash {
    size_t operator()(const MixedMoeTransformKey & key) const {
        return ((size_t)key.block * 1315423911u) ^
            (size_t)(key.seed ^ (key.seed >> 32));
    }
};

struct MixedMoeActivationKey {
    int input_rows = 0;
    int groups = 0;
    int gs = 0;
    int device = 0;
    MixedMoeTransformKey transform;

    bool operator==(const MixedMoeActivationKey & other) const {
        return input_rows == other.input_rows && groups == other.groups &&
            gs == other.gs && device == other.device &&
            transform == other.transform;
    }
};

struct MixedMoeActivationKeyHash {
    size_t operator()(const MixedMoeActivationKey & key) const {
        size_t value = (size_t)key.input_rows;
        value = value * 1315423911u + (size_t)key.groups;
        value = value * 1315423911u + (size_t)key.gs;
        value = value * 1315423911u + (size_t)key.device;
        return value * 1315423911u +
            MixedMoeTransformKeyHash{}(key.transform);
    }
};

enum class MixedNvqF16FormatGroup : int {
    All = 0,
    Standard = 1,
    Extended = 2,
    Legacy = 3,
};

MixedNvqF16FormatGroup mixed_nvq_f16_format_group(int format);

struct MixedNvqDispatch {
    mfq_tensor_backend::Tensor weight_ptrs;
    mfq_tensor_backend::Tensor weight_sizes;
    mfq_tensor_backend::Tensor pool_params;
    mfq_tensor_backend::Tensor expert_pool;
    mfq_tensor_backend::Tensor expert_local;
    int pool_count = 0;
    MixedNvqF16FormatGroup f16_format_group =
        MixedNvqF16FormatGroup::All;
    bool masked_experts = false;
};

struct MixedMoeRuntime {
    int n_experts = 0;
    int out_per_expert = 0;
    int neuron_len = 0;
    bool partial_experts = false;
    std::vector<MixedMoePool> pools;
    std::shared_ptr<MixedNvqDispatch> nvq_dispatch;

    bool nint_only() const {
        return !pools.empty() && std::all_of(
            pools.begin(), pools.end(), [](const MixedMoePool & pool) {
                return pool.family == MixedMoeFamily::Nint;
            });
    }
    MoeActivationWorkspace & activation_workspace(
            mfq_tensor_backend::Tensor x, int input_rows, int groups, int gs,
            MixedMoeTransformKey transform) const {
        static thread_local std::unordered_map<
            MixedMoeActivationKey, MoeActivationWorkspace,
            MixedMoeActivationKeyHash> shared_activation_workspaces;
        MixedMoeActivationKey key{
            input_rows, groups, gs, x.get_device(), transform};
        auto found = shared_activation_workspaces.find(key);
        if (found != shared_activation_workspaces.end()) return found->second;
        MoeActivationWorkspace value;
        value.qx = mfq_tensor_backend::empty(
            {input_rows, groups * gs}, x.options().dtype(mfq_tensor_backend::kInt8));
        value.xscale = mfq_tensor_backend::empty(
            {input_rows, groups}, x.options().dtype(mfq_tensor_backend::kFloat32));
        return shared_activation_workspaces.emplace(
            key, std::move(value)).first->second;
    }

    std::vector<MoeActivationGeometry> activation_geometry() const {
        std::vector<MoeActivationGeometry> result;
        for (const auto & pool : pools) {
            int groups = 0;
            int gs = 24;
            int transform_block = 0;
            uint64_t transform_seed = 0;
            if (pool.family == MixedMoeFamily::Nint) {
                groups = static_cast<int>(pool.nint.ng);
                gs = static_cast<int>(pool.nint.gs);
            } else if (pool.family == MixedMoeFamily::Nint8Zero) {
                groups = static_cast<int>(pool.q8_zero.ng);
                gs = 32;
            } else if (pool.family == MixedMoeFamily::Nvq) {
                groups = static_cast<int>(pool.nvq.ng);
                gs = static_cast<int>(pool.nvq.gs);
            } else if (pool.family == MixedMoeFamily::Mxfp4 ||
                    pool.family == MixedMoeFamily::Mxfp4Sq ||
                    pool.family == MixedMoeFamily::Fp8Sq ||
                    pool.family == MixedMoeFamily::Tpq) {
                continue;
            } else {
                groups = pool.nepq.ng;
                transform_block = pool.nepq.rotation_block;
                transform_seed = pool.nepq.rotation_seed;
            }
            const MoeActivationGeometry geometry{
                groups, gs, transform_block, transform_seed};
            if (std::find(result.begin(), result.end(), geometry) == result.end()) {
                result.push_back(geometry);
            }
        }
        return result;
    }

    mfq_tensor_backend::Tensor forward(
            mfq_tensor_backend::Tensor x,
            const MoeRoutePlan & route,
            bool input_prequantized = false,
            int epilogue_mode = 0) const {
        if (epilogue_mode < 0 || epilogue_mode > 2 ||
                (epilogue_mode != 0 &&
                 (!nint_only() || (out_per_expert % 2) != 0))) {
            throw std::runtime_error(
                "mixed MFE projection epilogue requires only canonical NINT");
        }
        if (!x.is_cuda() || !x.is_contiguous() ||
            x.scalar_type() != mfq_tensor_backend::kFloat16 ||
            (x.dim() != 2 && x.dim() != 3) || x.size(-1) != neuron_len) {
            throw std::runtime_error(
                "mixed MFE input must be contiguous CUDA f16 with exact K");
        }
        const int tokens = (int)route.ids.size(0);
        const int routes = (int)route.ids.size(1);
        if (route.n_experts != n_experts || x.size(0) != tokens ||
            (x.dim() == 3 && x.size(1) != routes)) {
            throw std::runtime_error("mixed MFE input and route shape mismatch");
        }
        const int input_rows = x.dim() == 3 ? tokens * routes : tokens;
        const int result_width = epilogue_mode == 0
            ? out_per_expert
            : out_per_expert / 2;
        auto output = partial_experts
            ? mfq_tensor_backend::zeros(
                {tokens, routes, result_width},
                x.options().dtype(mfq_tensor_backend::kFloat16))
            : mfq_tensor_backend::empty(
                {tokens, routes, result_width},
                x.options().dtype(mfq_tensor_backend::kFloat16));
        std::unordered_map<
            MixedMoeTransformKey, mfq_tensor_backend::Tensor,
            MixedMoeTransformKeyHash> prepared_inputs;
        const MixedMoeTransformKey identity{};
        prepared_inputs.emplace(identity, x);
        std::unordered_set<
            MixedMoeActivationKey, MixedMoeActivationKeyHash> quantized;
        static const bool disable_prefill_mma = [] {
            const char * value = std::getenv("MFQ_DISABLE_MOE_PREFILL_MMA");
            return value != nullptr && std::atoi(value) != 0;
        }();
        static const int prefill_mma_min_tokens = [] {
            const char * value = std::getenv("MFQ_MOE_PREFILL_MMA_MIN_TOKENS");
            return value == nullptr ? 9 : std::max(9, std::atoi(value));
        }();
        static const bool disable_nvq_hetero_decode = [] {
            const char * disabled =
                std::getenv("MFQ_DISABLE_MOE_NVQ_HETERO_DECODE");
            const char * exact =
                std::getenv("MFQ_NVQ_MOE_EXACT_REDUCTION");
            const char * rows =
                std::getenv("MFQ_NVQ_MOE_ROWS_PER_BLOCK");
            const char * warps =
                std::getenv("MFQ_NVQ_MOE_WARPS");
            const char * shared =
                std::getenv("MFQ_NVQ_MOE_SHARE_GROUP_STATE");
            return (disabled != nullptr && std::atoi(disabled) != 0) ||
                (exact != nullptr && std::atoi(exact) != 0) ||
                rows != nullptr || warps != nullptr || shared != nullptr;
        }();
        const bool use_f16_mma =
            !disable_prefill_mma && !g_force_moe_prefill_mma_off &&
            tokens >= prefill_mma_min_tokens && route.map_ready &&
            route.ids_dst.numel() == route.ids.numel();
        const bool use_kl_mmq = g_kl_mmq_mode != KlMmqMode::Default;
        const bool nvq_hetero_prefill_ready = nvq_dispatch &&
            (nvq_dispatch->pool_count > 1 ||
             (out_per_expert >= 128 &&
              static_cast<int64_t>(tokens) * routes >
                  static_cast<int64_t>(n_experts) * 16));
        const bool use_nvq_prefill =
            use_f16_mma && !use_kl_mmq && nvq_hetero_prefill_ready;
        const bool use_nvq_decode =
            !use_f16_mma && !use_kl_mmq && nvq_dispatch &&
            nvq_dispatch->pool_count > 1 &&
            tokens <= 8 && !g_force_moe_pool_path &&
            !disable_nvq_hetero_decode;
        int nint_pool_phase = 0;
        if (input_prequantized && use_kl_mmq) {
            throw std::runtime_error(
                "mixed prequantized activation reuse is unavailable in KLD MMQ mode");
        }
        if (use_kl_mmq) {
            MFQ_RUNTIME_CHECK(
                route.map_ready && route.ids_dst.numel() == route.ids.numel(),
                "KLD mixed routed FP16 requires the compact route map");
        }

        if (use_nvq_prefill) {
            const int routed_rows_per_expert = std::max(
                1, (tokens * routes + n_experts - 1) / n_experts);
            const bool use_coarse_nvq_tiles =
                routed_rows_per_expert > 64 && route.wide_tile_m > 8;
            const int nvq_tile_m = use_coarse_nvq_tiles
                ? route.wide_tile_m : 8;
            const auto & nvq_tile_bounds = use_coarse_nvq_tiles
                ? route.wide_tile_bounds : route.tile_bounds;
            const auto & nvq_tile_experts = use_coarse_nvq_tiles
                ? route.wide_tile_experts : route.tile_experts;
            nvq_moe_grouped_matmul_hetero_f16_cuda(
                nvq_dispatch->weight_ptrs,
                nvq_dispatch->weight_sizes,
                nvq_dispatch->pool_params,
                nvq_dispatch->expert_pool,
                nvq_dispatch->expert_local,
                x, n_experts, out_per_expert, neuron_len,
                nvq_tile_m, output,
                route.ids_dst, route.expert_bounds,
                nvq_tile_bounds, nvq_tile_experts,
                static_cast<int>(nvq_dispatch->f16_format_group),
                nvq_dispatch->masked_experts);
        }

        if (use_nvq_decode) {
            const int groups = (neuron_len + 23) / 24;
            const MixedMoeActivationKey activation_key{
                input_rows, groups, 24, x.get_device(), identity};
            auto & workspace = activation_workspace(
                x, input_rows, groups, 24, identity);
            auto qx = workspace.qx;
            auto xscale = workspace.xscale;
            bool input_quantized = input_prequantized;
            nvq_moe_grouped_matmul_hetero_ws_cuda(
                nvq_dispatch->weight_ptrs,
                nvq_dispatch->weight_sizes,
                nvq_dispatch->pool_params,
                nvq_dispatch->expert_pool,
                nvq_dispatch->expert_local,
                x, route.ids, n_experts, out_per_expert, neuron_len,
                input_quantized, output, qx, xscale);
            quantized.insert(activation_key);
        }

        for (const auto & pool : pools) {
            if (pool.family == MixedMoeFamily::Nvq &&
                    (use_nvq_prefill || use_nvq_decode)) {
                continue;
            }
            int gs = 24;
            int groups = 0;
            MixedMoeTransformKey transform{};
            mfq_tensor_backend::Tensor value = x;
            if (pool.family == MixedMoeFamily::Nint) {
                gs = (int)pool.nint.gs;
                groups = (int)pool.nint.ng;
            } else if (pool.family == MixedMoeFamily::Nint8Zero) {
                gs = 32;
                groups = (int)pool.q8_zero.ng;
            } else if (pool.family == MixedMoeFamily::Nvq) {
                gs = (int)pool.nvq.gs;
                groups = (int)pool.nvq.ng;
            } else if (pool.family == MixedMoeFamily::Mxfp4 ||
                    pool.family == MixedMoeFamily::Mxfp4Sq ||
                    pool.family == MixedMoeFamily::Fp8Sq ||
                    pool.family == MixedMoeFamily::Tpq) {
                groups = 0;
            } else {
                groups = pool.nepq.ng;
                transform = {
                    pool.nepq.rotation_block,
                    pool.nepq.rotation_seed,
                };
                if (transform.block != 0 && !input_prequantized) {
                    auto found = prepared_inputs.find(transform);
                    if (found == prepared_inputs.end()) {
                        auto flat = x.reshape({input_rows, neuron_len}).contiguous();
                        auto rotated = nepq_hadamard_input_cuda(
                            flat, pool.nepq.rotation_signs, transform.block);
                        value = x.dim() == 3
                            ? rotated.reshape({tokens, routes, neuron_len})
                            : rotated;
                        prepared_inputs.emplace(transform, value);
                    } else {
                        value = found->second;
                    }
                }
            }
            if (use_kl_mmq) {
                value = kl_mmq_prepare_activation(value);
                ++g_kl_mmq_moe_calls;
                if (pool.family == MixedMoeFamily::Mxfp4) {
                    mxfp4_moe_grouped_matmul_pool_f16_cuda(
                        pool.mxfp4.values, pool.mxfp4.scales, value,
                        route.ids, pool.expert_local, n_experts,
                        pool.local_experts, out_per_expert, neuron_len,
                        output, route.ids_dst, route.expert_bounds,
                        route.tile_bounds, route.tile_experts);
                    continue;
                }
                if (pool.family == MixedMoeFamily::Mxfp4Sq) {
                    mxfp4_sq_moe_matmul_cuda(
                        pool.mxfp4_sq.blob,
                        pool.mxfp4_sq.row_q,
                        pool.mxfp4_sq.row_symbol_byte_offsets,
                        pool.mxfp4_sq.row_auxiliary,
                        value, route.ids,
                        pool.expert_local, pool.mxfp4_sq.bits,
                        n_experts, pool.local_experts, out_per_expert,
                        neuron_len, pool.mxfp4_sq.matrix_scale_base,
                        pool.mxfp4_sq.q_sum,
                        pool.mxfp4_sq.sq4_rows,
                        output);
                    continue;
                }
                if (pool.family == MixedMoeFamily::Fp8Sq) {
                    fp8_sq_moe_matmul(
                        pool.fp8_sq, value, route.ids, pool.expert_local,
                        n_experts, pool.local_experts, out_per_expert,
                        neuron_len, output);
                    continue;
                }
                if (pool.family == MixedMoeFamily::Tpq) {
                    tpq_pq_moe_grouped_matmul_pool_f16_cuda(
                        pool.tpq.packed, pool.tpq.codebook, value,
                        route.ids, pool.expert_local, n_experts,
                        pool.local_experts, out_per_expert, neuron_len,
                        pool.tpq.vector_size, pool.tpq.index_bits, output,
                        route.ids_dst, route.expert_bounds,
                        route.tile_bounds, route.tile_experts);
                    continue;
                }
                if (pool.family == MixedMoeFamily::Nvq) {
                    nvq_moe_grouped_matmul_pool_f16_cuda(
                        pool.nvq.indices_packed, pool.nvq.aux_packed,
                        pool.nvq.sub_scale_packed, pool.nvq.neuron_scale,
                        pool.nvq.codebook, value, pool.expert_local,
                        n_experts, pool.local_experts, out_per_expert,
                        neuron_len, pool.nvq.gs, pool.nvq.sub_bits,
                        pool.nvq.kernel_format, pool.nvq.sign_mode, output,
                        route.ids_dst, route.expert_bounds,
                        route.tile_bounds, route.tile_experts);
                    continue;
                }
                if (pool.family == MixedMoeFamily::Nepq) {
                    nepq_moe_grouped_matmul_pool_f16_cuda(
                        pool.nepq.indices_packed, pool.nepq.aux_packed,
                        pool.nepq.state_packed, pool.nepq.neuron_scale,
                        pool.nepq.table_pool, pool.nepq.bank_ids, value,
                        pool.expert_local, n_experts, pool.local_experts,
                        out_per_expert, neuron_len, pool.nepq.state_bits,
                        pool.nepq.format, output, route.ids_dst,
                        route.expert_bounds, route.tile_bounds,
                        route.tile_experts);
                    if (pool.nepq.residual) {
                        nepq_sparse_residual_grouped_cuda(
                            pool.nepq.residual_codebook,
                            pool.nepq.residual_first,
                            pool.nepq.residual_second,
                            value, route.ids, pool.expert_local,
                            out_per_expert,
                            pool.nepq.residual_position_bits,
                            pool.nepq.residual_block_vectors,
                            output);
                    }
                    continue;
                }
                ++g_kl_mmq_fallback_calls;
                throw std::runtime_error(
                    "KLD mixed routed FP16 encountered a non-VQ pool");
            }
            if (pool.family == MixedMoeFamily::Mxfp4) {
                mxfp4_moe_grouped_matmul_pool_f16_cuda(
                    pool.mxfp4.values, pool.mxfp4.scales, value,
                    route.ids, pool.expert_local, n_experts,
                    pool.local_experts, out_per_expert, neuron_len,
                    output, route.ids_dst, route.expert_bounds,
                    route.tile_bounds, route.tile_experts);
                continue;
            }
            if (pool.family == MixedMoeFamily::Mxfp4Sq) {
                mxfp4_sq_moe_matmul_cuda(
                    pool.mxfp4_sq.blob,
                    pool.mxfp4_sq.row_q,
                    pool.mxfp4_sq.row_symbol_byte_offsets,
                    pool.mxfp4_sq.row_auxiliary,
                    value, route.ids,
                    pool.expert_local, pool.mxfp4_sq.bits,
                    n_experts, pool.local_experts, out_per_expert,
                    neuron_len, pool.mxfp4_sq.matrix_scale_base,
                    pool.mxfp4_sq.q_sum,
                    pool.mxfp4_sq.sq4_rows,
                    output);
                continue;
            }
            if (pool.family == MixedMoeFamily::Fp8Sq) {
                fp8_sq_moe_matmul(
                    pool.fp8_sq, value, route.ids, pool.expert_local,
                    n_experts, pool.local_experts, out_per_expert,
                    neuron_len, output);
                continue;
            }
            if (pool.family == MixedMoeFamily::Tpq) {
                tpq_pq_moe_grouped_matmul_pool_f16_cuda(
                    pool.tpq.packed, pool.tpq.codebook, value,
                    route.ids, pool.expert_local, n_experts,
                    pool.local_experts, out_per_expert, neuron_len,
                    pool.tpq.vector_size, pool.tpq.index_bits, output,
                    route.ids_dst, route.expert_bounds,
                    route.tile_bounds, route.tile_experts);
                continue;
            }
            if (pool.family == MixedMoeFamily::Nvq && use_f16_mma) {
                nvq_moe_grouped_matmul_pool_f16_cuda(
                    pool.nvq.indices_packed, pool.nvq.aux_packed,
                    pool.nvq.sub_scale_packed, pool.nvq.neuron_scale,
                    pool.nvq.codebook, value, pool.expert_local,
                    n_experts, pool.local_experts, out_per_expert,
                    neuron_len, pool.nvq.gs, pool.nvq.sub_bits,
                    pool.nvq.kernel_format, pool.nvq.sign_mode, output,
                    route.ids_dst, route.expert_bounds,
                    route.tile_bounds, route.tile_experts);
                continue;
            }
            MixedMoeActivationKey activation_key{
                input_rows, groups, gs, x.get_device(), transform};
            auto & workspace = activation_workspace(
                value, input_rows, groups, gs, transform);
            bool input_quantized = input_prequantized ||
                quantized.find(activation_key) != quantized.end();
            bool activation_quantized_after_call = true;
            auto qx = workspace.qx;
            auto xscale = workspace.xscale;
            if (pool.family == MixedMoeFamily::Nint) {
                const int route_tile_m = select_nint_prefill_route_tile(
                    route, tokens, routes, n_experts);
                mfe_nint_matmul_ws_cuda(
                    pool.nint.q_packed, pool.nint.row_q_bits,
                    pool.nint.row_q_bit_offsets, pool.nint.sub_scale,
                    pool.nint.sub_min, pool.nint.neuron_scale,
                    pool.nint.neuron_min, value, route.ids,
                    pool.expert_local, n_experts, pool.local_experts,
                    out_per_expert, gs, epilogue_mode, route.map_ready,
                    input_quantized,
                    output, qx, xscale, route.ids_dst,
                    route.expert_bounds,
                    nint_route_tile_bounds(route, route_tile_m),
                    nint_route_tile_experts(route, route_tile_m),
                    route_tile_m, nint_pool_phase++);
                activation_quantized_after_call =
                    input_quantized || route_tile_m == 8;
            } else if (pool.family == MixedMoeFamily::Nint8Zero) {
                const int routed_rows_per_expert = std::max(
                    1, (tokens * routes + n_experts - 1) / n_experts);
                const bool use_q8_f16_mma = use_f16_mma &&
                    (routed_rows_per_expert >= 5 ||
                     tokens >= n_experts * 4);
                const bool use_coarse_q8_tiles = use_q8_f16_mma &&
                    route.mma_tile_m == 64 && routed_rows_per_expert > 32;
                nint8_zero_moe_grouped_matmul_pool_ws_cuda(
                    pool.q8_zero.q_packed, pool.q8_zero.q8_zero_scale, value,
                    route.ids, pool.expert_local, n_experts,
                    pool.local_experts, out_per_expert, route.map_ready,
                    input_quantized, use_q8_f16_mma, output,
                    qx, xscale,
                    route.counts, route.cursors, route.ids_dst,
                    route.expert_bounds,
                    use_coarse_q8_tiles
                        ? route.mma_tile_bounds : route.tile_bounds,
                    use_coarse_q8_tiles
                        ? route.mma_tile_experts : route.tile_experts,
                    use_coarse_q8_tiles ? route.mma_tile_m : 8);
            } else if (pool.family == MixedMoeFamily::Nvq) {
                nvq_moe_grouped_matmul_pool_ws_cuda(
                    pool.nvq.indices_packed, pool.nvq.aux_packed,
                    pool.nvq.sub_scale_packed, pool.nvq.neuron_scale,
                    pool.nvq.codebook, value, route.ids, pool.expert_local,
                    n_experts, pool.local_experts, out_per_expert, neuron_len,
                    pool.nvq.gs, pool.nvq.sub_bits, pool.nvq.kernel_format,
                    pool.nvq.sign_mode, input_quantized, output,
                    qx, xscale, route.ids_dst,
                    route.expert_bounds, route.tile_bounds, route.tile_experts);
            } else {
                nepq_moe_grouped_matmul_pool_ws_cuda(
                    pool.nepq.indices_packed, pool.nepq.aux_packed,
                    pool.nepq.state_packed, pool.nepq.neuron_scale,
                    pool.nepq.table_pool, pool.nepq.bank_ids,
                    pool.nepq.grouped_table_pool, value, route.ids,
                    pool.expert_local, n_experts, pool.local_experts,
                    out_per_expert, neuron_len, pool.nepq.state_bits,
                    pool.nepq.format, input_quantized, output,
                    qx, xscale, route.ids_dst,
                    route.expert_bounds, route.tile_bounds, route.tile_experts);
                if (pool.nepq.residual) {
                    nepq_sparse_residual_grouped_cuda(
                        pool.nepq.residual_codebook,
                        pool.nepq.residual_first,
                        pool.nepq.residual_second,
                        value, route.ids, pool.expert_local,
                        out_per_expert,
                        pool.nepq.residual_position_bits,
                        pool.nepq.residual_block_vectors,
                        output);
                }
            }
            if (activation_quantized_after_call) {
                quantized.insert(activation_key);
            }
        }
        return output;
    }

    mfq_tensor_backend::Tensor forward_glu_output(
            mfq_tensor_backend::Tensor x,
            const MoeRoutePlan & route,
            bool gelu) const {
        if (nint_only() && g_kl_mmq_mode == KlMmqMode::Default) {
            return forward(x, route, false, gelu ? 2 : 1);
        }
        auto gate_up = forward(x, route);
        return gelu
            ? moe_geglu_split_cuda(gate_up)
            : moe_swiglu_split_cuda(gate_up);
    }

    bool supports_clamped_swiglu() const {
        return !pools.empty();
    }

    mfq_tensor_backend::Tensor forward_clamped_swiglu(
            mfq_tensor_backend::Tensor gate_up,
            const MoeRoutePlan & route,
            double limit) const {
        if (!supports_clamped_swiglu() ||
            !gate_up.is_cuda() || !gate_up.is_contiguous() ||
            gate_up.scalar_type() != mfq_tensor_backend::kFloat16 ||
            gate_up.dim() != 3 || gate_up.size(2) != 2 * neuron_len) {
            throw std::runtime_error(
                "mixed MFE clamped SwiGLU input is unsupported");
        }
        const int tokens = static_cast<int>(route.ids.size(0));
        const int routes = static_cast<int>(route.ids.size(1));
        if (route.n_experts != n_experts ||
            gate_up.size(0) != tokens || gate_up.size(1) != routes) {
            throw std::runtime_error(
                "mixed MFE clamped SwiGLU route shape mismatch");
        }
        auto parts = gate_up.split_with_sizes(
            {neuron_len, neuron_len}, -1);
        auto gate = mfq_tensor_backend::clamp_max(parts[0], limit);
        auto up = mfq_tensor_backend::clamp(parts[1], -limit, limit);
        auto activation =
            (mfq_tensor_backend::silu(gate) * up).contiguous();
        return forward(activation, route);
    }
};

MfeCpu unpack_mfe(const std::vector<uint8_t>& blob);
MfeCpu load_mfe_cpu(
    const mfq::ModelSource& source, const std::string& name);
MfeWeight to_gpu_mfe(const MfeCpu& source);
std::shared_ptr<MixedMoeRuntime> make_mixed_moe_runtime(
    const MfeCpu& source, bool cuda);
int64_t mixed_moe_storage_bytes(const MixedMoeRuntime& runtime);
mfq_tensor_backend::Tensor copy_cpu_weight_to_cuda(
    const mfq_tensor_backend::Tensor& source);
MfeWeight to_gpu_mixed_moe(const MfeCpu& source);
MfeWeight to_cuda_device_moe_expert_slice(
    const MfeCpu& source,
    int64_t expert_begin,
    int64_t expert_end,
    int device);
std::shared_ptr<MixedMoeRuntime> make_mxfp4_range_runtime(
    const mfq::cuda::MfeMxfp4ExpertStore& store);
std::vector<mfq::TensorParallelSlice> plan_moe_expert_parallel_slices(
    int64_t extent, const std::string& name);
