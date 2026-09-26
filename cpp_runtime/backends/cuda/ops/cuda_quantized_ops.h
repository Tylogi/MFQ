#pragma once

#include "cuda_execution.h"
#include "mfq_cuda_ops.h"
#include "mfq/kernels/cuda/fp8_sq.h"
#include "mfq/kernels/cuda/mxfp4_sq.h"
#include "mfq/model_source.h"
#include "moe_cache_profile.h"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

class MoeCachedSource;
class MoeExpertCache;
struct MixedMoeRuntime;
struct MoeRoutePlan;
struct MfeWeight;
struct NintWeight;
struct NvqWeight;
struct Mxfp4Weight;
struct Mxfp8Weight;
struct TpqWeight;
struct QuantLinear;
struct QuantLinearGroup;
struct DenseLinearGroup;

extern thread_local bool g_decode_graph_serial_branches;
extern thread_local bool g_decode_graph_tp_projection_major;
extern bool g_mfq_drop_file_cache;
extern std::shared_ptr<MoeExpertCache> g_moe_expert_cache;

struct MfqDropFileCacheGuard {
    bool previous;

    explicit MfqDropFileCacheGuard(bool enabled)
        : previous(g_mfq_drop_file_cache) {
        g_mfq_drop_file_cache = enabled;
    }

    ~MfqDropFileCacheGuard() {
        g_mfq_drop_file_cache = previous;
    }
};

std::shared_ptr<MoeExpertCache> make_moe_expert_cache(std::int64_t bytes);
bool moe_expert_cache_has_sources();
bool moe_expert_cache_finalized();
void finalize_moe_expert_cache();
void print_moe_expert_cache_stats(std::ostream& output);
void set_moe_expert_cache_profile(mfq::MoeCacheProfile profile);

bool decode_branch_parallel_enabled(std::int64_t rows);
bool nvq_fusion_enabled();
mfq_tensor_backend::Tensor nvq_matmul_multi2(
    const NvqWeight& first,
    const NvqWeight& second,
    mfq_tensor_backend::Tensor input);
mfq_tensor_backend::Tensor nvq_matmul_swiglu(
    const NvqWeight& gate,
    const NvqWeight& up,
    mfq_tensor_backend::Tensor input);
mfq_tensor_backend::Tensor mxfp8_groupwise_matmul(
    const Mxfp8Weight& weight,
    mfq_tensor_backend::Tensor input,
    std::int64_t groups);
mfq_tensor_backend::Tensor mxfp8_groupwise_matmul_f32(
    const Mxfp8Weight& weight,
    mfq_tensor_backend::Tensor input,
    std::int64_t groups);

mfq_tensor_backend::Tensor tensor_to_cuda_device(
    mfq_tensor_backend::Tensor value,
    int device,
    mfq_tensor_backend::Tensor reusable = {});
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

mfq_tensor_backend::Tensor nint_matmul(
    const NintWeight& weight, mfq_tensor_backend::Tensor input);
mfq_tensor_backend::Tensor nint_matmul_bf16_output(
    const NintWeight& weight, mfq_tensor_backend::Tensor input);
mfq_tensor_backend::Tensor nint_matmul_input_mul(
    const NintWeight& weight,
    mfq_tensor_backend::Tensor input,
    mfq_tensor_backend::Tensor gate,
    int mode);
mfq_tensor_backend::Tensor nint_matmul_input_mul_f32_kld(
    const NintWeight& weight,
    mfq_tensor_backend::Tensor input,
    mfq_tensor_backend::Tensor gate,
    int mode);
mfq_tensor_backend::Tensor nint_matmul_swiglu(
    const NintWeight& weight, mfq_tensor_backend::Tensor input);
mfq_tensor_backend::Tensor nint_matmul_geglu(
    const NintWeight& weight, mfq_tensor_backend::Tensor input);
mfq_tensor_backend::Tensor nvq_matmul(
    const NvqWeight& weight, mfq_tensor_backend::Tensor input);
mfq_tensor_backend::Tensor nvq_matmul_input_mul(
    const NvqWeight& weight,
    mfq_tensor_backend::Tensor input,
    mfq_tensor_backend::Tensor gate,
    int mode);
mfq_tensor_backend::Tensor mxfp8_matmul(
    const Mxfp8Weight& weight, mfq_tensor_backend::Tensor input);
mfq_tensor_backend::Tensor mxfp8_matmul_f32(
    const Mxfp8Weight& weight, mfq_tensor_backend::Tensor input);
mfq_tensor_backend::Tensor mxfp4_matmul(
    const Mxfp4Weight& weight, mfq_tensor_backend::Tensor input);
mfq_tensor_backend::Tensor tpq_matmul(
    const TpqWeight& weight, mfq_tensor_backend::Tensor input);
mfq_tensor_backend::Tensor run_quant_linear_shard(
    const struct QuantLinearShard& shard,
    mfq_tensor_backend::Tensor input,
    MfqOptional<mfq_tensor_backend::Tensor> gate = mfq_nullopt,
    int gate_mode = 0);

struct Mxfp8Weight {
    mfq_tensor_backend::Tensor values;
    mfq_tensor_backend::Tensor scales;
    int64_t out = 0;
    int64_t neuron_len = 0;
};

struct Workspace {
    int M = 0;
    int K_pad = 0;
    mfq_tensor_backend::Tensor qx;
    mfq_tensor_backend::Tensor xscale;
};

struct NintWeight {
    mfq_tensor_backend::Tensor q_packed;
    mfq_tensor_backend::Tensor row_q_bits;
    mfq_tensor_backend::Tensor row_q_bit_offsets;
    mfq_tensor_backend::Tensor q8_zero_scale;
    mfq_tensor_backend::Tensor sub_scale;
    mfq_tensor_backend::Tensor sub_min;
    mfq_tensor_backend::Tensor neuron_scale;
    mfq_tensor_backend::Tensor neuron_min;
    int64_t out = 0;
    int64_t ng = 0;
    int64_t gs = 0;
    int64_t bits = 0;
    int64_t neuron_len = 0;
    int64_t q_expert_stride = 0;
    int format_version = 2;
    double aggregate_bpw = 0.0;
    double distribution_entropy = 0.0;
    bool q8_zero = false;
    std::vector<int64_t> shape;
    mutable std::unordered_map<int, Workspace> workspaces;

    Workspace & workspace(int M) const {
        int K_pad = (int)(ng * gs);
        auto it = workspaces.find(M);
        if (it != workspaces.end() && it->second.K_pad == K_pad) return it->second;
        Workspace ws;
        ws.M = M;
        ws.K_pad = K_pad;
        const auto workspace_options = q_packed.options();
        ws.qx = mfq_tensor_backend::empty(
            {M, K_pad}, workspace_options.dtype(mfq_tensor_backend::kInt8));
        ws.xscale = mfq_tensor_backend::empty(
            {M, ng}, workspace_options.dtype(mfq_tensor_backend::kFloat32));
        auto res = workspaces.emplace(M, std::move(ws));
        return res.first->second;
    }

};

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

struct Mxfp4Weight {
    mfq_tensor_backend::Tensor values;
    mfq_tensor_backend::Tensor scales;
    int64_t out = 0;
    int64_t neuron_len = 0;
};

struct Mxfp4SqWeight {
    mfq_tensor_backend::Tensor blob;
    mfq_tensor_backend::Tensor row_q;
    mfq_tensor_backend::Tensor row_symbol_byte_offsets;
    mfq_tensor_backend::Tensor row_auxiliary;
    int64_t bits = 0;
    int64_t out = 0;
    int64_t neuron_len = 0;
    int64_t matrix_scale_base = 0;
    int64_t q_sum = 0;
    int64_t sq4_rows = 0;
    int format_version = 0;
    double aggregate_bpw = 0.0;
    double distribution_entropy = 0.0;
};

struct Fp8SqWeight {
    std::string dtype;
    mfq_tensor_backend::Tensor blob;
    mfq_tensor_backend::Tensor row_q;
    mfq_tensor_backend::Tensor row_symbol_byte_offsets;
    int64_t out = 0;
    int64_t neuron_len = 0;
    int64_t block_rows = 0;
    int64_t block_columns = 0;
    int64_t scale_rows = 0;
    int64_t scale_columns = 0;
    int64_t scale_kind = 0;
    int64_t palettes_offset = 0;
    int64_t symbols_offset = 0;
    int64_t scales_offset = 0;
    int format_version = 0;
    double aggregate_bpw = 0.0;
    double distribution_entropy = 0.0;
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

struct NvqWorkspace {
    int M = 0;
    int K_pad = 0;
    mfq_tensor_backend::Tensor qx;
    mfq_tensor_backend::Tensor xscale;
    mfq_tensor_backend::Tensor swiglu_scratch;
};

struct NvqWeight {
    mfq_tensor_backend::Tensor indices_packed;
    mfq_tensor_backend::Tensor aux_packed;
    mfq_tensor_backend::Tensor sub_scale_packed;
    mfq_tensor_backend::Tensor neuron_scale;
    mfq_tensor_backend::Tensor codebook;
    int64_t format = 0;
    int64_t kernel_format = 0;
    int64_t sign_mode = 0;
    int64_t sub_bits = 0;
    int64_t gs = 0;
    int64_t out = 0;
    int64_t ng = 0;
    int64_t neuron_len = 0;
    std::vector<int64_t> shape;
    mutable std::unordered_map<int, NvqWorkspace> workspaces;

    NvqWorkspace & workspace(int M) const {
        int K_pad = (int)(ng * gs);
        auto it = workspaces.find(M);
        if (it != workspaces.end() && it->second.K_pad == K_pad) return it->second;
        NvqWorkspace ws;
        ws.M = M;
        ws.K_pad = K_pad;
        const auto workspace_options = indices_packed.options();
        ws.qx = mfq_tensor_backend::empty(
            {M, K_pad}, workspace_options.dtype(mfq_tensor_backend::kInt8));
        ws.xscale = mfq_tensor_backend::empty(
            {M, ng}, workspace_options.dtype(mfq_tensor_backend::kFloat32));
        return workspaces.emplace(M, std::move(ws)).first->second;
    }
};

struct TpqWeight {
    bool int4 = false;
    mfq_tensor_backend::Tensor packed;
    mfq_tensor_backend::Tensor scales;
    mfq_tensor_backend::Tensor codebook;
    int64_t out = 0;
    int64_t neuron_len = 0;
    int group_size = 0;
    int vector_size = 0;
    int index_bits = 0;
};

struct NintLinear {
    NintWeight w;
    mfq_tensor_backend::Tensor forward(mfq_tensor_backend::Tensor x) const {
        auto shape = x.sizes().vec();
        int64_t last = shape.back();
        (void)last;
        auto y = nint_matmul(w, x.reshape({-1, x.size(-1)}));
        shape.back() = y.size(-1);
        return y.reshape(shape);
    }
    mfq_tensor_backend::Tensor forward_bf16_output(mfq_tensor_backend::Tensor x) const {
        return nint_matmul_bf16_output(w, x);
    }
    mfq_tensor_backend::Tensor forward_input_mul(mfq_tensor_backend::Tensor x, mfq_tensor_backend::Tensor gate, int mode) const {
        auto shape = x.sizes().vec();
        auto y = nint_matmul_input_mul(w, x.reshape({-1, x.size(-1)}), gate.reshape({-1, gate.size(-1)}), mode);
        shape.back() = y.size(-1);
        return y.reshape(shape);
    }
    mfq_tensor_backend::Tensor forward_input_mul_f32_kld(
            mfq_tensor_backend::Tensor x,
            mfq_tensor_backend::Tensor gate,
            int mode) const {
        auto shape = x.sizes().vec();
        auto y = nint_matmul_input_mul_f32_kld(
            w,
            x.reshape({-1, x.size(-1)}),
            gate.reshape({-1, gate.size(-1)}),
            mode);
        shape.back() = y.size(-1);
        return y.reshape(shape);
    }
};

struct DecodeGraphBranchScope {
    bool previous = g_decode_graph_serial_branches;
    DecodeGraphBranchScope() { g_decode_graph_serial_branches = true; }
    ~DecodeGraphBranchScope() { g_decode_graph_serial_branches = previous; }
};

struct DecodeGraphTpProjectionScope {
    bool previous = g_decode_graph_tp_projection_major;
    DecodeGraphTpProjectionScope() {
        g_decode_graph_tp_projection_major = true;
    }
    ~DecodeGraphTpProjectionScope() {
        g_decode_graph_tp_projection_major = previous;
    }
};

struct CudaIndependentBranchExecutor {
    using Stream =
        decltype(mfq_get_stream_from_pool(false));

    int device = -1;
    cudaEvent_t ready = nullptr;
    std::vector<Stream> streams;
    std::vector<cudaEvent_t> completed;

    ~CudaIndependentBranchExecutor() {
        for (cudaEvent_t event : completed) {
            if (event != nullptr) {
                (void)cudaEventDestroy(event);
            }
        }
        if (ready != nullptr) {
            (void)cudaEventDestroy(ready);
        }
    }

    bool ensure(size_t branches, const Stream & parent) {
        const int parent_device = parent.device_index();
        if (device >= 0 && device != parent_device) {
            return false;
        }
        if (streams.size() >= branches) {
            return true;
        }
        cudaStreamCaptureStatus capture_status =
            cudaStreamCaptureStatusNone;
        MFQ_CUDA_CHECK(cudaStreamIsCapturing(
            parent.stream(), &capture_status));
        if (capture_status != cudaStreamCaptureStatusNone) {
            return false;
        }
        device = parent_device;
        if (ready == nullptr) {
            MFQ_CUDA_CHECK(cudaEventCreateWithFlags(
                &ready, cudaEventDisableTiming));
        }
        while (streams.size() < branches) {
            streams.push_back(
                mfq_get_stream_from_pool(false, device));
            cudaEvent_t event = nullptr;
            MFQ_CUDA_CHECK(cudaEventCreateWithFlags(
                &event, cudaEventDisableTiming));
            completed.push_back(event);
        }
        return true;
    }

    template <typename Fn>
    bool run(
            size_t branches,
            Fn && fn,
            std::vector<mfq_tensor_backend::Tensor> & outputs) {
        if (branches < 2) {
            return false;
        }
        const Stream parent =
            mfq_get_current_cuda_stream();
        if (!ensure(branches, parent)) {
            return false;
        }

        outputs.resize(branches);
        MFQ_CUDA_CHECK(cudaEventRecord(
            ready, parent.stream()));
        for (size_t index = 0; index < branches; ++index) {
            const Stream branch_stream = streams[index];
            MFQ_CUDA_CHECK(cudaStreamWaitEvent(
                branch_stream.stream(), ready, 0));
            {
                MfqCudaStreamGuard guard(
                    branch_stream);
                outputs[index] = fn(index);
            }
            MFQ_CUDA_CHECK(cudaEventRecord(
                completed[index], branch_stream.stream()));
        }
        for (size_t index = 0; index < branches; ++index) {
            MFQ_CUDA_CHECK(cudaStreamWaitEvent(
                parent.stream(), completed[index], 0));
            if (outputs[index].defined()) {
                mfq_cuda_record_stream(outputs[index], parent);
            }
        }
        return true;
    }
};

struct NintLinearGroup {
    NintWeight w;
    std::vector<NintWeight> projection_w;
    std::vector<NintWeight> split_w;
    std::vector<std::vector<int64_t>> split_outs;
    std::vector<int64_t> outs;
    mutable std::shared_ptr<CudaIndependentBranchExecutor>
        branch_executor =
            std::make_shared<CudaIndependentBranchExecutor>();
    std::vector<mfq_tensor_backend::Tensor> forward(mfq_tensor_backend::Tensor x) const {
        auto shape = x.sizes().vec();
        std::vector<mfq_tensor_backend::Tensor> parts;
        auto xf = x.reshape({-1, x.size(-1)});
        if (w.q8_zero && xf.size(0) > 64 &&
                projection_w.size() == outs.size()) {
            parts.reserve(projection_w.size());
            for (const auto & projection : projection_w) {
                parts.push_back(nint_matmul(projection, xf));
            }
        } else if (!split_w.empty()) {
            parts.reserve(outs.size());
            std::vector<mfq_tensor_backend::Tensor> grouped_outputs;
            const bool parallel =
                decode_branch_parallel_enabled(xf.size(0)) &&
                branch_executor->run(
                    split_w.size(),
                    [&](size_t index) {
                        return nint_matmul(
                            split_w[index], xf);
                    },
                    grouped_outputs);
            for (size_t i = 0; i < split_w.size(); ++i) {
                auto y = parallel
                    ? grouped_outputs[i]
                    : nint_matmul(split_w[i], xf);
                auto ys = y.split_with_sizes(split_outs[i], -1);
                for (auto & p : ys) parts.push_back(p);
            }
        } else {
            auto y = nint_matmul(w, x.reshape({-1, x.size(-1)}));
            parts = y.split_with_sizes(outs, -1);
        }
        for (auto & p : parts) {
            auto s = shape;
            s.back() = p.size(-1);
            p = p.reshape(s);
        }
        return parts;
    }
    mfq_tensor_backend::Tensor forward_swiglu(mfq_tensor_backend::Tensor x) const {
        if (!split_w.empty() || outs.size() != 2 || outs[0] != outs[1]) {
            throw std::runtime_error("NINT SwiGLU fusion requires one packed [gate, up] group");
        }
        auto shape = x.sizes().vec();
        auto y = nint_matmul_swiglu(w, x.reshape({-1, x.size(-1)}));
        shape.back() = y.size(-1);
        return y.reshape(shape);
    }
    mfq_tensor_backend::Tensor forward_geglu(mfq_tensor_backend::Tensor x) const {
        if (!split_w.empty() || outs.size() != 2 || outs[0] != outs[1]) {
            throw std::runtime_error("NINT GeGLU fusion requires one packed [gate, up] group");
        }
        auto shape = x.sizes().vec();
        auto y = nint_matmul_geglu(w, x.reshape({-1, x.size(-1)}));
        shape.back() = y.size(-1);
        return y.reshape(shape);
    }
};

struct NvqLinear {
    NvqWeight w;
    mfq_tensor_backend::Tensor forward(mfq_tensor_backend::Tensor x) const {
        auto shape = x.sizes().vec();
        auto y = nvq_matmul(w, x.reshape({-1, x.size(-1)}));
        shape.back() = y.size(-1);
        return y.reshape(shape);
    }
    mfq_tensor_backend::Tensor forward_input_mul(mfq_tensor_backend::Tensor x, mfq_tensor_backend::Tensor gate, int mode) const {
        auto shape = x.sizes().vec();
        auto y = nvq_matmul_input_mul(
            w, x.reshape({-1, x.size(-1)}), gate.reshape({-1, gate.size(-1)}), mode);
        shape.back() = y.size(-1);
        return y.reshape(shape);
    }
};

struct TpqLinear {
    TpqWeight weight;

    mfq_tensor_backend::Tensor forward(mfq_tensor_backend::Tensor x) const {
        auto shape = x.sizes().vec();
        auto y = tpq_matmul(
            weight, x.reshape({-1, x.size(-1)}));
        shape.back() = y.size(-1);
        return y.reshape(shape);
    }
};

struct Mxfp4Linear {
    Mxfp4Weight weight;

    mfq_tensor_backend::Tensor forward(mfq_tensor_backend::Tensor x) const {
        auto shape = x.sizes().vec();
        auto y = mxfp4_matmul(
            weight, x.reshape({-1, x.size(-1)}));
        shape.back() = y.size(-1);
        return y.reshape(shape);
    }
};

struct Mxfp4SqLinear {
    Mxfp4SqWeight weight;

    mfq_tensor_backend::Tensor forward(
            mfq_tensor_backend::Tensor x) const {
        MFQ_RUNTIME_CHECK(
            weight.blob.is_cuda(),
            "MXFP4-SQ does not support dense CPU-layer offload");
        auto shape = x.sizes().vec();
        const auto original_dtype = x.scalar_type();
        auto source = x.reshape({-1, x.size(-1)});
        if (source.scalar_type() != mfq_tensor_backend::kFloat16 &&
                source.scalar_type() != mfq_tensor_backend::kFloat32) {
            source = source.to(mfq_tensor_backend::kFloat16).contiguous();
        } else {
            source = source.contiguous();
        }
        auto output = mxfp4_sq_matmul_cuda(
            weight.blob,
            weight.row_q,
            weight.row_symbol_byte_offsets,
            weight.row_auxiliary,
            source,
            weight.bits,
            weight.out,
            weight.neuron_len,
            weight.matrix_scale_base,
            weight.q_sum,
            weight.sq4_rows);
        if (original_dtype == mfq_tensor_backend::kBFloat16) {
            output = output.to(original_dtype).contiguous();
        }
        shape.back() = output.size(-1);
        return output.reshape(shape);
    }
};

struct Fp8SqLinear {
    Fp8SqWeight weight;

    mfq_tensor_backend::Tensor forward(
            mfq_tensor_backend::Tensor x) const {
        MFQ_RUNTIME_CHECK(
            weight.blob.is_cuda(),
            "FP8-SQ does not support dense CPU-layer offload");
        auto shape = x.sizes().vec();
        const auto original_dtype = x.scalar_type();
        auto source = x.reshape({-1, x.size(-1)});
        if (source.scalar_type() != mfq_tensor_backend::kFloat16 &&
                source.scalar_type() != mfq_tensor_backend::kFloat32) {
            source = source.to(mfq_tensor_backend::kFloat16).contiguous();
        } else {
            source = source.contiguous();
        }
        mfq_tensor_backend::Tensor output;
        if (weight.dtype == "MXFP8-SQ") {
            output = mxfp8_sq_matmul_cuda(
                weight.blob, weight.row_q,
                weight.row_symbol_byte_offsets, source,
                weight.out, weight.neuron_len,
                weight.block_rows, weight.block_columns,
                weight.scale_rows, weight.scale_columns,
                weight.palettes_offset, weight.symbols_offset,
                weight.scales_offset);
        } else if (weight.dtype == "FP8-128SQ") {
            output = fp8_128_sq_matmul_cuda(
                weight.blob, weight.row_q,
                weight.row_symbol_byte_offsets, source,
                weight.out, weight.neuron_len, weight.scale_kind,
                weight.palettes_offset, weight.symbols_offset,
                weight.scales_offset);
        } else {
            throw std::runtime_error("unsupported FP8-SQ linear dtype");
        }
        if (original_dtype == mfq_tensor_backend::kBFloat16) {
            output = output.to(original_dtype).contiguous();
        }
        shape.back() = output.size(-1);
        return output.reshape(shape);
    }

    mfq_tensor_backend::Tensor forward_swiglu_m5(
            const Fp8SqLinear & up,
            mfq_tensor_backend::Tensor x) const {
        auto shape = x.sizes().vec();
        auto source = x.reshape({-1, x.size(-1)});
        if (source.scalar_type() != mfq_tensor_backend::kFloat16) {
            source = source.to(mfq_tensor_backend::kFloat16).contiguous();
        } else {
            source = source.contiguous();
        }
        auto output = fp8_128_sq_swiglu_m5_cuda(
            weight.blob, weight.row_q,
            weight.row_symbol_byte_offsets,
            up.weight.blob, up.weight.row_q,
            up.weight.row_symbol_byte_offsets,
            source, weight.out, weight.neuron_len, weight.scale_kind,
            weight.palettes_offset, weight.symbols_offset,
            weight.scales_offset, up.weight.palettes_offset,
            up.weight.symbols_offset, up.weight.scales_offset);
        shape.back() = output.size(-1);
        return output.reshape(shape);
    }
};

struct Mxfp8Linear {
    Mxfp8Weight weight;

    mfq_tensor_backend::Tensor forward(mfq_tensor_backend::Tensor x) const {
        auto shape = x.sizes().vec();
        auto y = mxfp8_matmul(
            weight, x.reshape({-1, x.size(-1)}));
        shape.back() = y.size(-1);
        return y.reshape(shape);
    }
};

enum class QuantLinearKind {
    Nint,
    Nvq,
    Mxfp4,
    Mxfp4Sq,
    Fp8Sq,
    Mxfp8,
    Tpq,
    Dense,
};

struct QuantLinearShard {
    int device = 0;
    int64_t input_begin = 0;
    int64_t input_end = 0;
    int64_t output_begin = 0;
    int64_t output_end = 0;
    QuantLinearKind kind = QuantLinearKind::Nint;
    NintWeight nint;
    NvqWeight nvq;
    Mxfp4Weight mxfp4;
    Mxfp8Weight mxfp8;
    TpqWeight tpq;
    mfq_tensor_backend::Tensor dense;
};

struct QuantLinear {
    QuantLinearKind kind = QuantLinearKind::Nint;
    NintLinear nint;
    NvqLinear nvq;
    Mxfp4Linear mxfp4;
    Mxfp4SqLinear mxfp4_sq;
    Fp8SqLinear fp8_sq;
    Mxfp8Linear mxfp8;
    TpqLinear tpq;
    mfq_tensor_backend::Tensor dense;
    bool dense_small_m_rowwise = false;
    TensorParallelAxis tensor_parallel_axis =
        TensorParallelAxis::Mirrored;
    std::vector<QuantLinearShard> tensor_parallel_shards;
    int64_t logical_out = 0;
    int64_t logical_neuron_len = 0;

    bool tensor_parallel() const {
        return !tensor_parallel_shards.empty();
    }

    bool is_nint() const { return kind == QuantLinearKind::Nint; }
    bool is_nvq() const { return kind == QuantLinearKind::Nvq; }
    bool is_mxfp4() const { return kind == QuantLinearKind::Mxfp4; }
    bool is_mxfp4_sq() const { return kind == QuantLinearKind::Mxfp4Sq; }
    bool is_fp8_sq() const { return kind == QuantLinearKind::Fp8Sq; }
    bool is_mxfp8() const { return kind == QuantLinearKind::Mxfp8; }
    bool is_tpq() const { return kind == QuantLinearKind::Tpq; }
    bool is_dense() const { return kind == QuantLinearKind::Dense; }

    mfq_tensor_backend::Tensor forward_tensor_parallel_flat(
            mfq_tensor_backend::Tensor x,
            MfqOptional<mfq_tensor_backend::Tensor> gate,
            int gate_mode) const {
        MFQ_RUNTIME_CHECK(
            tensor_parallel(),
            "tensor-parallel linear has no shards");
        MFQ_RUNTIME_CHECK(
            tensor_parallel_axis == TensorParallelAxis::Output ||
            tensor_parallel_axis == TensorParallelAxis::Input,
            "tensor-parallel linear has an invalid axis");
        std::vector<mfq_tensor_backend::Tensor> local_outputs(
            tensor_parallel_shards.size());
        for (size_t launch_position = 0;
             launch_position < tensor_parallel_shards.size();
             ++launch_position) {
            const size_t index = model_parallel_launch_index(
                launch_position, tensor_parallel_shards.size());
            const auto & shard = tensor_parallel_shards[index];
            MfqCudaGuard guard(shard.device);
            mfq_tensor_backend::Tensor local_x = x;
            mfq_tensor_backend::Tensor local_gate;
            if (tensor_parallel_axis == TensorParallelAxis::Input) {
                local_x = x.narrow(
                    -1, shard.input_begin,
                    shard.input_end - shard.input_begin);
                if (gate.has_value()) {
                    local_gate = gate.value().narrow(
                        -1, shard.input_begin,
                        shard.input_end - shard.input_begin);
                }
            } else if (gate.has_value()) {
                local_gate = gate.value();
            }
            local_x = tensor_to_cuda_device(local_x, shard.device);
            if (gate.has_value()) {
                local_gate =
                    tensor_to_cuda_device(local_gate, shard.device);
            }
            if (is_mxfp8() &&
                    tensor_parallel_axis == TensorParallelAxis::Input) {
                MFQ_RUNTIME_CHECK(
                    !gate.has_value(),
                    "MXFP8 input-axis tensor parallelism does not support gating");
                local_outputs[index] =
                    mxfp8_matmul_f32(shard.mxfp8, local_x);
            } else {
                local_outputs[index] =
                    run_quant_linear_shard(
                        shard, local_x,
                        gate.has_value()
                            ? MfqOptional<mfq_tensor_backend::Tensor>(
                                local_gate)
                            : mfq_nullopt,
                        gate_mode);
            }
        }

        const int primary = model_parallel_primary_device();
        MfqCudaGuard primary_guard(primary);
        if (tensor_parallel_axis == TensorParallelAxis::Output) {
            std::vector<mfq_tensor_backend::Tensor> gathered;
            gathered.reserve(local_outputs.size());
            for (auto & output : local_outputs) {
                gathered.push_back(
                    tensor_to_cuda_device(output, primary));
            }
            return mfq_tensor_backend::cat(gathered, -1).contiguous();
        }

        auto reduced = reduce_model_parallel_outputs(
            std::move(local_outputs));
        return is_mxfp8()
            ? reduced.to(x.scalar_type()).contiguous()
            : reduced;
    }

    mfq_tensor_backend::Tensor forward_dense(mfq_tensor_backend::Tensor x) const {
        auto input = x.to(dense.scalar_type());
        const int64_t rows = input.numel() / input.size(-1);
        if (dense_small_m_rowwise && rows > 1 && rows <= 6) {
            auto shape = input.sizes().vec();
            shape.back() = dense.size(0);
            // Native matmul issues the same M=1 cuBLAS operation per row.
            return mfq_tensor_backend::matmul(
                input.reshape({rows, 1, input.size(-1)}), dense.transpose(0, 1))
                .reshape(shape);
        }
        return mfq_tensor_backend::matmul(input, dense.transpose(0, 1));
    }

    mfq_tensor_backend::Tensor forward(mfq_tensor_backend::Tensor x) const {
        if (tensor_parallel()) {
            auto shape = x.sizes().vec();
            auto y = forward_tensor_parallel_flat(
                x.reshape({-1, x.size(-1)}),
                mfq_nullopt, 0);
            shape.back() = y.size(-1);
            return y.reshape(shape);
        }
        if (is_nint()) return nint.forward(x);
        if (is_nvq()) return nvq.forward(x);
        if (is_mxfp4()) return mxfp4.forward(x);
        if (is_mxfp4_sq()) return mxfp4_sq.forward(x);
        if (is_fp8_sq()) return fp8_sq.forward(x);
        if (is_tpq()) return tpq.forward(x);
        if (is_dense()) return forward_dense(x);
        return mxfp8.forward(x);
    }
    mfq_tensor_backend::Tensor forward_bf16_output(mfq_tensor_backend::Tensor x) const {
        if (!tensor_parallel() && is_nint()) {
            return nint.forward_bf16_output(x);
        }
        return forward(x).to(mfq_tensor_backend::kBFloat16).contiguous();
    }
    mfq_tensor_backend::Tensor forward_mxfp8_groupwise(
            mfq_tensor_backend::Tensor grouped,
            int64_t groups) const {
        MFQ_RUNTIME_CHECK(
            is_mxfp8(),
            "groupwise MXFP8 projection requires an MXFP8 tensor");
        if (!tensor_parallel()) {
            return mxfp8_groupwise_matmul(
                mxfp8.weight, grouped, groups);
        }
        MFQ_RUNTIME_CHECK(
            tensor_parallel_axis == TensorParallelAxis::Input,
            "groupwise MXFP8 tensor parallelism requires input-axis shards");
        std::vector<mfq_tensor_backend::Tensor> partials;
        partials.reserve(tensor_parallel_shards.size());
        for (const auto & shard : tensor_parallel_shards) {
            MFQ_RUNTIME_CHECK(
                shard.kind == QuantLinearKind::Mxfp8,
                "groupwise MXFP8 tensor-parallel shard kind mismatch");
            MfqCudaGuard guard(shard.device);
            auto local = grouped.narrow(
                -1, shard.input_begin,
                shard.input_end - shard.input_begin);
            local = tensor_to_cuda_device(
                local, shard.device);
            partials.push_back(mxfp8_groupwise_matmul_f32(
                shard.mxfp8, local, groups));
        }
        return reduce_model_parallel_outputs(
            std::move(partials))
            .to(grouped.scalar_type()).contiguous();
    }
    mfq_tensor_backend::Tensor forward_input_mul(mfq_tensor_backend::Tensor x, mfq_tensor_backend::Tensor gate, int mode) const {
        if (tensor_parallel()) {
            auto shape = x.sizes().vec();
            auto y = forward_tensor_parallel_flat(
                x.reshape({-1, x.size(-1)}),
                gate.reshape({-1, gate.size(-1)}),
                mode);
            shape.back() = y.size(-1);
            return y.reshape(shape);
        }
        if (is_nint()) return nint.forward_input_mul(x, gate, mode);
        if (is_nvq()) return nvq.forward_input_mul(x, gate, mode);
        if (is_dense()) {
            MFQ_RUNTIME_CHECK(mode == 1 || mode == 2,
                "dense input gate mode must be sigmoid or SiLU");
            // Match the existing dense shard path, including dtype rounding.
            auto local = x.to(dense.scalar_type());
            auto local_gate = gate.to(dense.scalar_type());
            auto gated = mode == 1
                ? local * mfq_tensor_backend::sigmoid(local_gate)
                : local * mfq_tensor_backend::silu(local_gate);
            return forward_dense(gated);
        }
        MFQ_RUNTIME_CHECK(mode == 1 || mode == 2,
            "input gate mode must be sigmoid or SiLU");
        auto local_gate = gate.to(x.scalar_type());
        auto gated = mode == 1
            ? x * mfq_tensor_backend::sigmoid(local_gate)
            : x * mfq_tensor_backend::silu(local_gate);
        return forward(gated);
    }
    mfq_tensor_backend::Tensor forward_input_mul_f32_kld(
            mfq_tensor_backend::Tensor x,
            mfq_tensor_backend::Tensor gate,
            int mode) const {
        MFQ_RUNTIME_CHECK(
            !tensor_parallel() && is_nint(),
            "FP32-output KLD down projection requires a local NINT tensor");
        return nint.forward_input_mul_f32_kld(
            x, gate, mode);
    }
    int64_t out() const {
        if (tensor_parallel()) return logical_out;
        if (is_nint()) return nint.w.out;
        if (is_nvq()) return nvq.w.out;
        if (is_mxfp4()) return mxfp4.weight.out;
        if (is_mxfp4_sq()) return mxfp4_sq.weight.out;
        if (is_fp8_sq()) return fp8_sq.weight.out;
        if (is_tpq()) return tpq.weight.out;
        if (is_dense()) return dense.size(0);
        return mxfp8.weight.out;
    }
    int64_t neuron_len() const {
        if (tensor_parallel()) return logical_neuron_len;
        if (is_nint()) return nint.w.neuron_len;
        if (is_nvq()) return nvq.w.neuron_len;
        if (is_mxfp4()) return mxfp4.weight.neuron_len;
        if (is_mxfp4_sq()) return mxfp4_sq.weight.neuron_len;
        if (is_fp8_sq()) return fp8_sq.weight.neuron_len;
        if (is_tpq()) return tpq.weight.neuron_len;
        if (is_dense()) return dense.size(1);
        return mxfp8.weight.neuron_len;
    }
};

using QuantLinearProjectionRefs =
    std::vector<const QuantLinear*>;

bool tensor_parallel_output_projections_compatible(
    const QuantLinearProjectionRefs& projections);
std::vector<mfq_tensor_backend::Tensor>
forward_tensor_parallel_output_projections(
    mfq_tensor_backend::Tensor input,
    const QuantLinearProjectionRefs& projections);

struct QuantLinearGroup {
    bool nint_grouped = false;
    bool nvq_prefix2 = false;
    bool decode_branch_parallel = true;
    NintLinearGroup nint;
    std::vector<QuantLinear> layers;
    std::vector<int64_t> outs;
    mutable std::shared_ptr<CudaIndependentBranchExecutor>
        branch_executor =
            std::make_shared<CudaIndependentBranchExecutor>();

    QuantLinearProjectionRefs tensor_parallel_output_projections() const {
        QuantLinearProjectionRefs projections;
        projections.reserve(layers.size());
        for (const auto & layer : layers) {
            projections.push_back(&layer);
        }
        return projections;
    }

    bool tensor_parallel_output_compatible() const {
        return tensor_parallel_output_projections_compatible(
            tensor_parallel_output_projections());
    }

    std::vector<mfq_tensor_backend::Tensor>
    forward_tensor_parallel_output_group(
            mfq_tensor_backend::Tensor x) const {
        return forward_tensor_parallel_output_projections(
            x, tensor_parallel_output_projections());
    }

    std::vector<mfq_tensor_backend::Tensor> forward(mfq_tensor_backend::Tensor x) const {
        if (!x.is_cuda()) {
            MFQ_RUNTIME_CHECK(
                !nint_grouped,
                "CPU dense offload requires separate compact linear weights");
            std::vector<mfq_tensor_backend::Tensor> result;
            result.reserve(layers.size());
            for (const auto & layer : layers) {
                result.push_back(layer.forward(x));
            }
            return result;
        }
        if (tensor_parallel_grouped_projections_enabled() &&
                tensor_parallel_output_compatible()) {
            return forward_tensor_parallel_output_group(x);
        }
        if (nint_grouped) return nint.forward(x);
        if (g_kl_mmq_mode == KlMmqMode::Default &&
                nvq_prefix2 && nvq_fusion_enabled()) {
            auto shape = x.sizes().vec();
            const auto flat =
                x.reshape({-1, x.size(-1)});
            std::vector<mfq_tensor_backend::Tensor> branch_outputs;
            const bool parallel =
                decode_branch_parallel &&
                decode_branch_parallel_enabled(flat.size(0)) &&
                layers.size() > 2 &&
                branch_executor->run(
                    layers.size() - 1,
                    [&](size_t branch) {
                        if (branch == 0) {
                            return nvq_matmul_multi2(
                                layers[0].nvq.w,
                                layers[1].nvq.w,
                                flat);
                        }
                        return layers[branch + 1].forward(x);
                    },
                    branch_outputs);
            auto combined = parallel
                ? branch_outputs[0]
                : nvq_matmul_multi2(
                    layers[0].nvq.w, layers[1].nvq.w,
                    flat);
            auto pair = combined.split_with_sizes({outs[0], outs[1]}, -1);
            std::vector<mfq_tensor_backend::Tensor> result;
            result.reserve(layers.size());
            for (size_t i = 0; i < 2; ++i) {
                auto part_shape = shape;
                part_shape.back() = outs[i];
                result.push_back(pair[i].reshape(part_shape));
            }
            for (size_t i = 2; i < layers.size(); ++i) {
                result.push_back(
                    parallel
                        ? branch_outputs[i - 1]
                        : layers[i].forward(x));
            }
            return result;
        }
        std::vector<mfq_tensor_backend::Tensor> result;
        if (g_kl_mmq_mode == KlMmqMode::Default &&
                decode_branch_parallel &&
                decode_branch_parallel_enabled(
                    x.numel() / x.size(-1)) &&
                branch_executor->run(
                    layers.size(),
                    [&](size_t index) {
                        return layers[index].forward(x);
                    },
                    result)) {
            return result;
        }
        result.reserve(layers.size());
        for (const auto & layer : layers) {
            result.push_back(layer.forward(x));
        }
        return result;
    }
    mfq_tensor_backend::Tensor forward_swiglu(mfq_tensor_backend::Tensor x) const {
        if (g_kl_mmq_mode == KlMmqMode::Default &&
                layers.size() == 2 &&
                x.numel() / x.size(-1) == 5 &&
                layers[0].is_fp8_sq() && layers[1].is_fp8_sq() &&
                layers[0].fp8_sq.weight.dtype == "FP8-128SQ" &&
                layers[1].fp8_sq.weight.dtype == "FP8-128SQ" &&
                layers[0].fp8_sq.weight.out == layers[1].fp8_sq.weight.out &&
                layers[0].fp8_sq.weight.neuron_len ==
                    layers[1].fp8_sq.weight.neuron_len &&
                layers[0].fp8_sq.weight.scale_kind ==
                    layers[1].fp8_sq.weight.scale_kind) {
            return layers[0].fp8_sq.forward_swiglu_m5(
                layers[1].fp8_sq, x);
        }
        if (g_kl_mmq_mode == KlMmqMode::Default &&
                nint_grouped && nint.split_w.empty() &&
                x.numel() / x.size(-1) >= 1 && x.numel() / x.size(-1) <= 6) {
            return nint.forward_swiglu(x);
        }
        if (g_kl_mmq_mode == KlMmqMode::Default &&
                nvq_prefix2 && layers.size() == 2 &&
                nvq_fusion_enabled()) {
            auto shape = x.sizes().vec();
            auto y = nvq_matmul_swiglu(
                layers[0].nvq.w, layers[1].nvq.w,
                x.reshape({-1, x.size(-1)}));
            shape.back() = y.size(-1);
            return y.reshape(shape);
        }
        if (outs.size() != 2 || outs[0] != outs[1]) {
            throw std::runtime_error("SwiGLU requires equal gate/up output widths");
        }
        auto parts = forward(x);
        return silu_mul_cuda(
            parts[0].contiguous(), parts[1].contiguous());
    }
    mfq_tensor_backend::Tensor forward_geglu(mfq_tensor_backend::Tensor x) const {
        if (g_kl_mmq_mode == KlMmqMode::Default &&
                nint_grouped && nint.split_w.empty() &&
                x.numel() / x.size(-1) == 1) {
            return nint.forward_geglu(x);
        }
        if (outs.size() != 2 || outs[0] != outs[1]) {
            throw std::runtime_error("GeGLU requires equal gate/up output widths");
        }
        auto parts = forward(x);
        return gelu_mul_cuda(parts[0].contiguous(), parts[1].contiguous());
    }
};

struct DenseLinearGroup {
    mfq_tensor_backend::Tensor w;
    std::vector<int64_t> outs;

    std::vector<mfq_tensor_backend::Tensor> forward(mfq_tensor_backend::Tensor x) const {
        auto shape = x.sizes().vec();
        auto y = mfq_tensor_backend::matmul(x.reshape({-1, x.size(-1)}).to(mfq_tensor_backend::kFloat32), w.transpose(0, 1));
        auto parts = y.split_with_sizes(outs, -1);
        for (auto & p : parts) {
            auto s = shape;
            s.back() = p.size(-1);
            p = p.reshape(s);
        }
        return parts;
    }
};


const mfq::TensorMetadata& require_tensor(
    const mfq::ModelSource& source, std::string_view name);
bool has_tensor(const mfq::ModelSource& source, std::string_view name) noexcept;
bool has_tensor_prefix(const mfq::ModelSource& source, std::string_view prefix);
std::vector<std::uint8_t> read_tensor(
    const mfq::ModelSource& source, std::string_view name);
std::vector<std::uint8_t> read_asset(
    const mfq::ModelSource& source, std::string_view name);
std::string read_asset_text(
    const mfq::ModelSource& source, std::string_view name);

MoeRoutePlan build_moe_route_plan(
    mfq_tensor_backend::Tensor ids, int n_experts);
MfeWeight load_mfe_gpu(
    const mfq::ModelSource& source,
    const std::string& name,
    bool cacheable = false,
    int layer_id = -1,
    const std::string& projection_role = {});
std::shared_ptr<MixedMoeRuntime> load_mfe_cpu_offloaded(
    const mfq::ModelSource& source, const std::string& name);
MfeWeight cpu_mixed_moe_metadata(
    const std::shared_ptr<MixedMoeRuntime>& runtime);
mfq_tensor_backend::Tensor load_dense_gpu(
    const mfq::ModelSource& source, const std::string& name);
QuantLinear load_quant_linear(
    const mfq::ModelSource& source,
    const std::string& name,
    std::optional<TensorParallelAxis> axis_override = std::nullopt,
    const std::vector<mfq::TensorParallelSlice>* slices_override = nullptr);
QuantLinearGroup load_quant_group(
    const mfq::ModelSource& source,
    const std::vector<std::string>& names,
    size_t required_compatible_prefix = 0,
    const std::vector<mfq::TensorParallelSlice>* slices_override = nullptr,
    bool preserve_projection_boundaries = false);
QuantLinearGroup load_paired_gate_up(
    const mfq::ModelSource& source,
    const std::vector<std::string>& names,
    const QuantLinear& down,
    size_t required_compatible_prefix = 2,
    bool preserve_projection_boundaries = false);
mfq_tensor_backend::Tensor quant_embedding_lookup(
    const QuantLinear& weight, mfq_tensor_backend::Tensor token_ids);
DenseLinearGroup make_fp32_quant_group(QuantLinearGroup group);

std::int64_t read_i64_from(
    const std::vector<std::uint8_t>& bytes, std::size_t& offset);
std::uint32_t read_u32_from(
    const std::vector<std::uint8_t>& bytes, std::size_t& offset);
MfeWeight stage_cpu_mixed_moe(
    const std::shared_ptr<MixedMoeRuntime>& runtime);
bool prefetch_cached_moe_projection_bundle(
    const MfeWeight& gate,
    const MfeWeight& up,
    const MfeWeight& down,
    const MoeRoutePlan& route);
bool prefetch_cached_moe_projection_bundle(
    const MfeWeight& gate_up,
    const MfeWeight& down,
    const MoeRoutePlan& route);
mfq_tensor_backend::Tensor nvq_ffn_swiglu_down(
    const NvqWeight& gate,
    const NvqWeight& up,
    const NvqWeight& down,
    mfq_tensor_backend::Tensor input,
    MfqOptional<mfq_tensor_backend::Tensor> residual = mfq_nullopt);
bool nvq_fused_residual_format(std::int64_t kernel_format);
mfq_tensor_backend::Tensor nint_matmul_groupwise_u8(
    const NintWeight& weight,
    mfq_tensor_backend::Tensor input,
    std::int64_t groups);
bool is_quant_dtype(const std::string& dtype);
QuantLinearGroup make_quant_group(
    std::vector<QuantLinear> layers,
    bool preserve_projection_boundaries = false);
DenseLinearGroup make_dense_group(
    const std::vector<mfq_tensor_backend::Tensor>& weights);
mfq_tensor_backend::Tensor quant_linear_reference_weight(
    const QuantLinear& linear);
mfq_tensor_backend::Tensor mfe_dense_reference(
    const mfq::ModelSource& source,
    const std::string& name,
    mfq_tensor_backend::Tensor input,
    const std::vector<std::int32_t>& expert_ids,
    int tokens,
    int routes,
    bool routed_input);
mfq_tensor_backend::Tensor materialize_mfe_dense(
    const mfq::ModelSource& source, const std::string& name);
