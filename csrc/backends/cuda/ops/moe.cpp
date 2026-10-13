#include "execution_options.h"
#include "kernels/moe/ops.h"
#include "kernels/quant_ops.h"
#include "kernels/nint_route_hint.h"
#include "kernels/nvq1_decode.h"
#include "moe.h"

#include "quant_linear.h"
#include "format.h"
#include "mfq_format_compat.h"
#include "kernels/moe/pipeline.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <limits>
#include <unordered_set>

using mfq_tensor_backend::indexing::Slice;
using namespace mfq::cuda::quant_format;

static std::atomic<uint64_t> g_moe_route_generation{1};

mfq_tensor_backend::Tensor moe_tensor_to_device(
        ModelParallelCollectiveRuntime& collectives,
        mfq_tensor_backend::Tensor value, int device) {
    return value.defined()
        ? tensor_to_cuda_device(collectives, std::move(value), device)
        : value;
}

static void copy_route_tensor(
        ModelParallelCollectiveRuntime& collectives,
        mfq_tensor_backend::Tensor& destination,
        const mfq_tensor_backend::Tensor& source,
        int device) {
    if (!source.defined()) {
        destination = mfq_tensor_backend::Tensor();
    } else if (source.is_cuda() && source.get_device() == device) {
        destination = source.contiguous();
    } else {
        destination = tensor_to_cuda_device(
            collectives, source, device, std::move(destination));
    }
}

const MoeRoutePlan& moe_route_to_device(
        ModelParallelCollectiveRuntime& collectives,
        const MoeRoutePlan& source,
        int device) {
    auto& replica = (*source.device_replicas)[device];
    if (!replica) replica = std::make_shared<MoeRoutePlan>();
    if (replica->generation == source.generation) return *replica;
    copy_route_tensor(collectives, replica->ids, source.ids, device);
    copy_route_tensor(collectives, replica->ids_dst, source.ids_dst, device);
    copy_route_tensor(collectives, replica->expert_bounds, source.expert_bounds, device);
    copy_route_tensor(collectives, replica->tile_bounds, source.tile_bounds, device);
    copy_route_tensor(collectives, replica->tile_experts, source.tile_experts, device);
    if (source.mma_tile_m == 8) {
        replica->mma_tile_bounds = replica->tile_bounds;
        replica->mma_tile_experts = replica->tile_experts;
    } else {
        copy_route_tensor(collectives, replica->mma_tile_bounds, source.mma_tile_bounds, device);
        copy_route_tensor(collectives, replica->mma_tile_experts, source.mma_tile_experts, device);
    }
    if (source.wide_tile_m == 8) {
        replica->wide_tile_bounds = replica->tile_bounds;
        replica->wide_tile_experts = replica->tile_experts;
    } else if (source.wide_tile_m == source.mma_tile_m) {
        replica->wide_tile_bounds = replica->mma_tile_bounds;
        replica->wide_tile_experts = replica->mma_tile_experts;
    } else {
        copy_route_tensor(collectives, replica->wide_tile_bounds, source.wide_tile_bounds, device);
        copy_route_tensor(collectives, replica->wide_tile_experts, source.wide_tile_experts, device);
    }
    copy_route_tensor(collectives, replica->counts, source.counts, device);
    copy_route_tensor(collectives, replica->cursors, source.cursors, device);
    replica->n_experts = source.n_experts;
    replica->mma_tile_m = source.mma_tile_m;
    replica->wide_tile_m = source.wide_tile_m;
    replica->map_ready = source.map_ready;
    replica->generation = source.generation;
    replica->host_unique_experts = source.host_unique_experts;
    return *replica;
}

MoeRoutePlan build_moe_route_plan(mfq_tensor_backend::Tensor ids, int n_experts) {
    if (!ids.is_cuda() || ids.scalar_type() != mfq_tensor_backend::kInt32 || ids.dim() != 2) {
        throw std::runtime_error("MoE ids must be CUDA int32 [tokens, routes]");
    }
    MoeRoutePlan result;
    result.generation = g_moe_route_generation.fetch_add(
        1, std::memory_order_relaxed);
    result.ids = ids.contiguous();
    result.n_experts = n_experts;
    auto empty = mfq_tensor_backend::empty({0}, result.ids.options());
    result.ids_dst = empty;
    result.expert_bounds = empty;
    result.tile_bounds = empty;
    result.tile_experts = empty;
    result.mma_tile_bounds = empty;
    result.mma_tile_experts = empty;
    result.wide_tile_bounds = empty;
    result.wide_tile_experts = empty;
    result.counts = empty;
    result.cursors = empty;
    if (result.ids.size(0) > 8) {
        const int64_t rows_per_expert = std::max<int64_t>(
            1, (result.ids.numel() + n_experts - 1) / n_experts);
        const bool use_coarse_mma = rows_per_expert >= 4;
        const int mma_tile_m = rows_per_expert <= 16
            ? 16 : (rows_per_expert <= 32 ? 32 : 64);
        auto mapped = !use_coarse_mma
            ? moe_build_expert_map_cuda(result.ids, n_experts, 8)
            : (rows_per_expert > 64
                ? moe_build_expert_maps_cuda(
                    result.ids, n_experts, 8, mma_tile_m, 128)
                : moe_build_expert_maps_cuda(
                    result.ids, n_experts, 8, mma_tile_m, 0));
        result.ids_dst = mapped.at(0);
        result.expert_bounds = mapped.at(1);
        result.tile_bounds = mapped.at(2);
        result.tile_experts = mapped.at(3);
        result.counts = mapped.at(4);
        if (use_coarse_mma) {
            result.mma_tile_bounds = mapped.at(5);
            result.mma_tile_experts = mapped.at(6);
            result.mma_tile_m = mma_tile_m;
            if (rows_per_expert > 64) {
                result.wide_tile_bounds = mapped.at(7);
                result.wide_tile_experts = mapped.at(8);
                result.wide_tile_m = 128;
            } else {
                result.wide_tile_bounds = result.mma_tile_bounds;
                result.wide_tile_experts = result.mma_tile_experts;
                result.wide_tile_m = result.mma_tile_m;
            }
        } else {
            result.mma_tile_bounds = result.tile_bounds;
            result.mma_tile_experts = result.tile_experts;
            result.mma_tile_m = 8;
            result.wide_tile_bounds = result.tile_bounds;
            result.wide_tile_experts = result.tile_experts;
            result.wide_tile_m = 8;
        }
    }
    result.map_ready = result.ids.size(0) <= 8 ||
        result.ids_dst.numel() == result.ids.numel();
    return result;
}

int select_nint_prefill_route_tile(
        const MoeRoutePlan & route,
        int tokens,
        int routes,
        int experts) {
    const int64_t pairs = static_cast<int64_t>(tokens) * routes;
    const int64_t rows_per_expert = std::max<int64_t>(
        1, (pairs + experts - 1) / experts);
    if (rows_per_expert < 4) {
        return 8;
    }
    if (rows_per_expert > 64 && route.wide_tile_m == 128) {
        return 128;
    }
    return route.mma_tile_m == 16 || route.mma_tile_m == 32 ||
            route.mma_tile_m == 64
        ? route.mma_tile_m
        : 8;
}

const mfq_tensor_backend::Tensor & nint_route_tile_bounds(
        const MoeRoutePlan & route, int tile_m) {
    return tile_m == 128
        ? route.wide_tile_bounds
        : (tile_m == 8 ? route.tile_bounds : route.mma_tile_bounds);
}

const mfq_tensor_backend::Tensor & nint_route_tile_experts(
        const MoeRoutePlan & route, int tile_m) {
    return tile_m == 128
        ? route.wide_tile_experts
        : (tile_m == 8 ? route.tile_experts : route.mma_tile_experts);
}
void fp8_sq_moe_matmul(
        const Fp8SqWeight & weight,
        mfq_tensor_backend::Tensor input,
        mfq_tensor_backend::Tensor expert_ids,
        mfq_tensor_backend::Tensor expert_local,
        int n_experts,
        int local_experts,
        int out_per_expert,
        int neuron_len,
        mfq_tensor_backend::Tensor output) {
    if (weight.dtype == "MXFP8-SQ") {
        mxfp8_sq_moe_matmul_cuda(
            weight.blob, weight.row_q, weight.row_symbol_byte_offsets,
            std::move(input), std::move(expert_ids),
            std::move(expert_local), n_experts, local_experts,
            out_per_expert, neuron_len, weight.block_rows,
            weight.block_columns, weight.scale_rows, weight.scale_columns,
            weight.palettes_offset, weight.symbols_offset,
            weight.scales_offset, std::move(output));
        return;
    }
    if (weight.dtype == "FP8-128SQ") {
        fp8_128_sq_moe_matmul_cuda(
            weight.blob, weight.row_q, weight.row_symbol_byte_offsets,
            std::move(input), std::move(expert_ids),
            std::move(expert_local), n_experts, local_experts,
            out_per_expert, neuron_len, weight.scale_kind,
            weight.palettes_offset, weight.symbols_offset,
            weight.scales_offset, std::move(output));
        return;
    }
    throw std::runtime_error("unsupported FP8-SQ MoE dtype");
}

MixedNvqF16FormatGroup mixed_nvq_f16_format_group(int format) {
    switch (format) {
        case 2:
        case 3:
        case 4:
        case 5:
        case 6:
        case 10:
        case 11:
            return MixedNvqF16FormatGroup::Standard;
        case 12:
        case 13:
        case 14:
        case 15:
        case 16:
        case 17:
            return MixedNvqF16FormatGroup::Extended;
        case 1:
        case 7:
        case 8:
        case 9:
            return MixedNvqF16FormatGroup::Legacy;
        default:
            return MixedNvqF16FormatGroup::All;
    }
}

static void initialize_mixed_nint_dispatch(MixedMoeRuntime& runtime) {
    namespace tb=mfq_tensor_backend;
    runtime.nint_dispatch.reset();runtime.nint_input_plans.clear();
    if(runtime.n_experts<=0 || runtime.n_experts>4096 || runtime.neuron_len<=0 ||
        runtime.out_per_expert<=0 || runtime.out_per_expert>INT_MAX)return;
    auto dispatch=std::make_shared<MixedNintDispatch>();
    std::vector<int64_t> maps;std::vector<int32_t> params;
    tb::Device device(tb::kCUDA,mfq_current_cuda_device());
    for(std::size_t i=0;i<runtime.pools.size();++i) {
        const auto& pool=runtime.pools[i];if(pool.family!=MixedMoeFamily::Nint)continue;
        const auto& w=pool.nint;
        if(!w.q_packed.is_cuda() || pool.local_experts<=0 || w.q8_zero ||
            w.out!=int64_t(pool.local_experts)*runtime.out_per_expert || w.neuron_len!=runtime.neuron_len ||
            w.ng<=0 || w.ng>INT_MAX || w.gs<=0 || w.gs>128 || w.ng*w.gs>INT_MAX ||
            w.ng!=(runtime.neuron_len+w.gs-1)/w.gs ||
            w.q_packed.numel()%pool.local_experts || w.q_packed.numel()/pool.local_experts>INT_MAX)return;
        if(dispatch->pools.empty())device=w.q_packed.device();
        const auto field_ok=[&](const tb::Tensor& field,auto dtype,int64_t count) {
            return field.defined() && field.is_cuda() && field.device()==device && field.is_contiguous() &&
                field.scalar_type()==dtype && field.numel()>=count;
        };
        if(!field_ok(w.q_packed,tb::kUInt8,1) || !field_ok(pool.expert_local,tb::kInt32,runtime.n_experts) ||
            !field_ok(w.row_q_bits,tb::kUInt8,w.out) || !field_ok(w.row_q_bit_offsets,tb::kInt64,w.out) ||
            !field_ok(w.sub_scale,tb::kUInt8,w.out*w.ng) || !field_ok(w.sub_min,tb::kUInt8,w.out*w.ng) ||
            !field_ok(w.neuron_scale,tb::kFloat32,w.out) || !field_ok(w.neuron_min,tb::kFloat32,w.out))return;
        dispatch->pools.push_back(int(i));maps.push_back(reinterpret_cast<int64_t>(pool.expert_local.data_ptr<int32_t>()));
        params.insert(params.end(),{int32_t(w.ng),int32_t(w.gs),int32_t(w.q_packed.numel()/pool.local_experts)});
    }
    if(dispatch->pools.empty())return;
    dispatch->map_ptrs=tb::tensor(maps).to(device);
    dispatch->pool_params=tb::tensor(params).reshape({int64_t(dispatch->pools.size()),3}).to(device);
    dispatch->expert_pool=tb::full({runtime.n_experts},-1,dispatch->pool_params.options());
    dispatch->expert_local=tb::full({runtime.n_experts},-1,dispatch->pool_params.options());
    runtime.nint_dispatch=std::move(dispatch);
}

void initialize_mixed_nvq_dispatch(
        MixedMoeRuntime & runtime,
        const CudaExecutionConfig& config,bool dense_prefill,
        const std::vector<std::vector<int32_t>>* expert_local_host) {
    if(expert_local_host && expert_local_host->size()!=runtime.pools.size())
        throw std::invalid_argument("mixed NVQ host-map pool count mismatch");
    std::unordered_map<const void*,mfq_tensor_backend::Tensor> integer_books;
    for(auto& pool:runtime.pools)if(pool.family==MixedMoeFamily::Nvq) {
        auto& w=pool.nvq;
        prepare_nvq_decode_records(w);
        const auto* key=w.codebook.data_ptr();
        const auto found=integer_books.find(key);
        if(found!=integer_books.end())w.integer_codebook=found->second;
        else {
            prepare_nvq_integer_codebook(w);
            if(w.integer_codebook.defined())integer_books.emplace(key,w.integer_codebook);
        }
    }
    initialize_mixed_nint_dispatch(runtime);
    runtime.nvq_dispatch.reset();
    if (!config.moe_nvq_heterogeneous) return;
    bool has_dense_groups=false;
    for(const auto& pool:runtime.pools)
        if(pool.family==MixedMoeFamily::Nvq && pool.nvq.dense_groups)has_dense_groups=true;
    // Prefill's FP16 weight reader handles compact records directly. Preserve
    // the existing decode dispatch until its separate readers are validated.
    if(has_dense_groups && !dense_prefill)return;

    int nvq_pools = 0;
    for (const auto & pool : runtime.pools) {
        if (pool.family == MixedMoeFamily::Nvq) ++nvq_pools;
    }
    if (nvq_pools == 0) return;

    std::vector<int64_t> weight_ptrs;
    std::vector<int64_t> map_ptrs;
    std::vector<int64_t> weight_sizes;
    std::vector<int32_t> pool_params;
    std::vector<int32_t> expert_pool(
        static_cast<size_t>(runtime.n_experts), -1);
    std::vector<int32_t> expert_local(
        static_cast<size_t>(runtime.n_experts), -1);
    weight_ptrs.reserve(static_cast<size_t>(nvq_pools) * 5);
    weight_sizes.reserve(static_cast<size_t>(nvq_pools) * 3);
    pool_params.reserve(static_cast<size_t>(nvq_pools) * 7);

    mfq_tensor_backend::Device target = mfq_tensor_backend::Device(
        mfq_tensor_backend::kCUDA, mfq_current_cuda_device());
    int dispatch_pool = 0;
    int owned_experts = 0;
    MixedNvqF16FormatGroup f16_format_group =
        MixedNvqF16FormatGroup::All;
    bool first_nvq_format = true;
    for (std::size_t pool_index=0;pool_index<runtime.pools.size();++pool_index) {
        const auto& pool=runtime.pools[pool_index];
        if (pool.family != MixedMoeFamily::Nvq) continue;
        const auto & weight = pool.nvq;
        map_ptrs.push_back(static_cast<int64_t>(reinterpret_cast<uintptr_t>(pool.expert_local.data_ptr<int32_t>())));
        if (weight.gs != 24 || weight.ng <= 0 ||
                weight.ng > std::numeric_limits<int32_t>::max() ||
                weight.sub_bits < 1 || weight.sub_bits > 8 ||
                weight.kernel_format < 1 || weight.kernel_format > 17 ||
                weight.sign_mode < 0 || weight.sign_mode > 1) {
            return;
        }
        if (dispatch_pool == 0) target = weight.indices_packed.device();
        if (weight.indices_packed.device() != target ||
                weight.aux_packed.device() != target ||
                weight.sub_scale_packed.device() != target ||
                weight.neuron_scale.device() != target ||
                weight.codebook.device() != target) {
            return;
        }
        weight_ptrs.push_back(static_cast<int64_t>(
            reinterpret_cast<uintptr_t>(
                weight.indices_packed.data_ptr<uint8_t>())));
        weight_ptrs.push_back(static_cast<int64_t>(
            reinterpret_cast<uintptr_t>(
                weight.aux_packed.data_ptr<uint8_t>())));
        weight_ptrs.push_back(static_cast<int64_t>(
            reinterpret_cast<uintptr_t>(
                weight.sub_scale_packed.data_ptr<uint8_t>())));
        weight_ptrs.push_back(static_cast<int64_t>(
            reinterpret_cast<uintptr_t>(
                weight.neuron_scale.data_ptr<float>())));
        weight_ptrs.push_back(static_cast<int64_t>(
            reinterpret_cast<uintptr_t>(
                weight.codebook.data_ptr<int8_t>())));
        weight_sizes.push_back(weight.indices_packed.numel());
        weight_sizes.push_back(weight.aux_packed.numel());
        weight_sizes.push_back(weight.sub_scale_packed.numel());
        const int format = static_cast<int>(weight.kernel_format);
        const auto pool_format_group = mixed_nvq_f16_format_group(format);
        if (first_nvq_format) {
            f16_format_group = pool_format_group;
            first_nvq_format = false;
        } else if (f16_format_group != pool_format_group) {
            f16_format_group = MixedNvqF16FormatGroup::All;
        }
        const bool d4 =
            format == 3 || format == 10 || format == 11 ||
            format == 12 || format == 15 || format == 17;
        const int nvec =
            (runtime.neuron_len + (d4 ? 3 : 7)) / (d4 ? 4 : 8);
        const int nsign = (runtime.neuron_len + 7) / 8;
        pool_params.push_back(pool.local_experts);
        pool_params.push_back(static_cast<int32_t>(weight.ng));
        pool_params.push_back(nvec);
        pool_params.push_back(nsign);
        pool_params.push_back(static_cast<int32_t>(weight.sub_bits));
        pool_params.push_back(static_cast<int32_t>(weight.sign_mode));
        pool_params.push_back(format);

        mfq_tensor_backend::Tensor local_host;
        const int32_t* local=nullptr;
        if(expert_local_host) {
            const auto& host=expert_local_host->at(pool_index);
            if(host.size()!=std::size_t(runtime.n_experts))
                throw std::invalid_argument("mixed NVQ host-map expert count mismatch");
            local=host.data();
        } else {
            local_host=pool.expert_local.to(mfq_tensor_backend::kCPU,mfq_tensor_backend::kInt32).contiguous();
            local=local_host.data_ptr<int32_t>();
        }
        for (int expert = 0; expert < runtime.n_experts; ++expert) {
            if (local[expert] < 0) continue;
            if (local[expert] >= pool.local_experts ||
                    expert_pool[static_cast<size_t>(expert)] >= 0) {
                throw std::runtime_error(
                    "mixed NVQ prefill has invalid expert ownership");
            }
            expert_pool[static_cast<size_t>(expert)] = dispatch_pool;
            expert_local[static_cast<size_t>(expert)] = local[expert];
            ++owned_experts;
        }
        ++dispatch_pool;
    }

    auto dispatch = std::make_shared<MixedNvqDispatch>();
    dispatch->pool_count = dispatch_pool;
    dispatch->map_ptrs=mfq_tensor_backend::tensor(map_ptrs).to(target).contiguous();
    dispatch->f16_format_group = f16_format_group;
    dispatch->masked_experts = owned_experts < runtime.n_experts;
    dispatch->dense_groups = has_dense_groups;
    dispatch->weight_ptrs = mfq_tensor_backend::from_blob(
        weight_ptrs.data(), {dispatch_pool, 5},
        mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64))
        .clone().to(target).contiguous();
    dispatch->weight_sizes = mfq_tensor_backend::from_blob(
        weight_sizes.data(), {dispatch_pool, 3},
        mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64))
        .clone().to(target).contiguous();
    dispatch->pool_params = mfq_tensor_backend::from_blob(
        pool_params.data(), {dispatch_pool, 7},
        mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt32))
        .clone().to(target).contiguous();
    dispatch->expert_pool = mfq_tensor_backend::from_blob(
        expert_pool.data(), {runtime.n_experts},
        mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt32))
        .clone().to(target).contiguous();
    dispatch->expert_local = mfq_tensor_backend::from_blob(
        expert_local.data(), {runtime.n_experts},
        mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt32))
        .clone().to(target).contiguous();
    runtime.nvq_dispatch = std::move(dispatch);
}

bool MixedMoeRuntime::nint_only() const {
    return !pools.empty() && std::all_of(
        pools.begin(), pools.end(), [](const MixedMoePool & pool) {
            return pool.family == MixedMoeFamily::Nint;
        });
}

MoeActivationWorkspace & MixedMoeRuntime::activation_workspace(
        mfq_tensor_backend::Tensor x, int input_rows, int groups, int gs,
        MixedMoeTransformKey transform) const {
    MixedMoeActivationKey key{
        input_rows, groups, gs, x.get_device(), transform};
    auto found = activation_workspaces.find(key);
    if (found != activation_workspaces.end()) return found->second;
    MoeActivationWorkspace value;
    value.qx = mfq_tensor_backend::empty(
        {input_rows, groups * gs}, x.options().dtype(mfq_tensor_backend::kInt8));
    value.xscale = mfq_tensor_backend::empty(
        {input_rows, groups}, x.options().dtype(mfq_tensor_backend::kFloat32));
    return activation_workspaces.emplace(
        key, std::move(value)).first->second;
}

MoeActivationWorkspace MixedMoeRuntime::prefill_activation_workspace(
        mfq_tensor_backend::Tensor x,int input_rows,int groups,int gs,
        MixedMoeTransformKey transform) const {
    // All prompt groups execute on the same compute stream. One growing
    // allocation per geometry can therefore serve every layer and density.
    const MixedMoeActivationKey key{0,groups,gs,x.get_device(),transform};
    auto& value=activation_workspaces[key];
    if(!value.qx.defined() || value.qx.size(0)<input_rows) {
        const int capacity=((input_rows+255)/256)*256;
        value.qx=mfq_tensor_backend::empty({capacity,groups*gs},x.options().dtype(mfq_tensor_backend::kInt8));
        value.xscale=mfq_tensor_backend::empty({capacity,groups},x.options().dtype(mfq_tensor_backend::kFloat32));
    }
    return {value.qx.narrow(0,0,input_rows),value.xscale.narrow(0,0,input_rows)};
}

std::vector<MoeActivationGeometry> MixedMoeRuntime::activation_geometry() const {
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
                pool.family == MixedMoeFamily::Fp8Sq) {
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

mfq_tensor_backend::Tensor MixedMoeRuntime::forward(
        const CudaExecutionConfig& config,
        KlMmqState& kl_mmq,
        bool force_prefill_mma_off,
        bool force_pool_path,
        mfq_tensor_backend::Tensor x,
        const MoeRoutePlan & route,
        bool input_prequantized,
        int epilogue_mode) const {
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
    const bool use_f16_mma =
        config.moe_prefill_mma &&
        !force_prefill_mma_off &&
        (tokens >= config.moe_prefill_mma_min_tokens || prefill_only) &&
        route.map_ready &&
        route.ids_dst.numel() == route.ids.numel();
    const bool use_kl_mmq =
        kl_mmq.mode != KlMmqMode::Default;
    const bool nvq_hetero_prefill_ready = nvq_dispatch &&
        (nvq_dispatch->pool_count > 1 ||
         (out_per_expert >= 128 &&
          static_cast<int64_t>(tokens) * routes >
              static_cast<int64_t>(n_experts) * 16));
    const bool use_nvq_prefill =
        use_f16_mma && tokens>8 && !use_kl_mmq && nvq_hetero_prefill_ready;
    const bool use_nvq_decode =
        !use_f16_mma && !use_kl_mmq && nvq_dispatch && !nvq_dispatch->dense_groups &&
        nvq_dispatch->pool_count > 1 &&
        tokens <= 8 && !prefill_only && !force_pool_path &&
        config.moe_nvq_heterogeneous_decode;
    const bool use_active_nvq=use_nvq_decode && config.moe_nvq_active_decode &&
        route.ids.numel()>0 && route.ids.numel()<=1024;
    mfq_tensor_backend::Tensor active_nvq_plan;
    if(use_active_nvq) {
        if (!route.ids.is_cuda() || !route.ids.is_contiguous() ||
            route.ids.scalar_type() != mfq_tensor_backend::kInt32 ||
            route.ids.get_device() != x.get_device()) {
            throw std::runtime_error("active NVQ routes require contiguous int32 IDs on the input CUDA device");
        }
        const int pairs=int(route.ids.numel());
        auto found=nvq_active_plans.find(pairs);
        if(found==nvq_active_plans.end())found=nvq_active_plans.emplace(pairs,
            mfq_tensor_backend::empty({1+int64_t(pairs)*3},route.ids.options())).first;
        active_nvq_plan=found->second;
        mfq::cuda::moe_plan_nvq_routes(reinterpret_cast<const uint64_t*>(nvq_dispatch->map_ptrs.data_ptr<int64_t>()),
            nvq_dispatch->pool_count,n_experts,route.ids.data_ptr<int32_t>(),pairs,
            active_nvq_plan.data_ptr<int32_t>(),mfq_current_cuda_stream());
        MFQ_CUDA_CHECK(cudaGetLastError());
    } else if(use_nvq_prefill || use_nvq_decode) {
        mfq::cuda::moe_update_nvq_maps(reinterpret_cast<const uint64_t*>(nvq_dispatch->map_ptrs.data_ptr<int64_t>()),
            nvq_dispatch->pool_count,n_experts,nvq_dispatch->expert_pool.data_ptr<int32_t>(),
            nvq_dispatch->expert_local.data_ptr<int32_t>(),mfq_current_cuda_stream());
        MFQ_CUDA_CHECK(cudaGetLastError());
    }
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
        if(use_active_nvq)nvq_moe_grouped_matmul_active_ws_cuda(
            nvq_dispatch->weight_ptrs,nvq_dispatch->weight_sizes,nvq_dispatch->pool_params,
            nvq_dispatch->expert_pool,nvq_dispatch->expert_local,x,route.ids,n_experts,
            out_per_expert,neuron_len,input_quantized,output,qx,xscale,active_nvq_plan);
        else nvq_moe_grouped_matmul_hetero_ws_cuda(
            nvq_dispatch->weight_ptrs,
            nvq_dispatch->weight_sizes,
            nvq_dispatch->pool_params,
            nvq_dispatch->expert_pool,
            nvq_dispatch->expert_local,
            x, route.ids, n_experts, out_per_expert, neuron_len,
            input_quantized, output, qx, xscale);
        quantized.insert(activation_key);
    }

    const bool use_nint_decode=config.moe_nint_heterogeneous_decode && nint_dispatch &&
        !use_f16_mma && !use_kl_mmq && !prefill_only && !force_pool_path && tokens<=8 &&
        int64_t(tokens)*routes<=65535 && epilogue_mode==0;
    if(use_nint_decode) {
        auto found=nint_input_plans.find(input_rows);
        if(found==nint_input_plans.end()) {
            std::vector<int64_t> pointers,quantize;
            std::unordered_set<MixedMoeActivationKey,MixedMoeActivationKeyHash> geometries;
            MixedNintInputPlan plan;
            for(int index:nint_dispatch->pools) {
                const auto& w=pools[index].nint;
                auto& workspace=activation_workspace(x,input_rows,int(w.ng),int(w.gs),identity);
                pointers.insert(pointers.end(),{reinterpret_cast<int64_t>(w.q_packed.data_ptr()),
                    reinterpret_cast<int64_t>(w.row_q_bits.data_ptr()),reinterpret_cast<int64_t>(w.row_q_bit_offsets.data_ptr()),
                    reinterpret_cast<int64_t>(w.sub_scale.data_ptr()),reinterpret_cast<int64_t>(w.sub_min.data_ptr()),
                    reinterpret_cast<int64_t>(w.neuron_scale.data_ptr()),reinterpret_cast<int64_t>(w.neuron_min.data_ptr()),
                    reinterpret_cast<int64_t>(workspace.qx.data_ptr()),reinterpret_cast<int64_t>(workspace.xscale.data_ptr())});
                const MixedMoeActivationKey key{input_rows,int(w.ng),int(w.gs),x.get_device(),identity};
                if(geometries.insert(key).second) {
                    quantize.insert(quantize.end(),{reinterpret_cast<int64_t>(workspace.qx.data_ptr()),
                        reinterpret_cast<int64_t>(workspace.xscale.data_ptr()),w.ng,w.gs});
                    plan.total_groups+=int(w.ng);
                }
            }
            plan.weight_ptrs=mfq_tensor_backend::tensor(pointers).reshape({int64_t(nint_dispatch->pools.size()),9}).to(x.device());
            plan.quantize_descriptors=mfq_tensor_backend::tensor(quantize).reshape({int64_t(geometries.size()),4}).to(x.device());
            found=nint_input_plans.emplace(input_rows,std::move(plan)).first;
        }
        auto& plan=found->second;
        const auto row_setting=mfq::cuda::runtime_options::nint_route_row_workspace();
        const auto direct_setting=mfq::cuda::runtime_options::nint_direct_pool();
        const auto group_setting=mfq::cuda::runtime_options::nint_group_dot();
        const auto hint_setting=mfq::cuda::runtime_options::nint_route_hint();
        const auto warp_setting=mfq::cuda::runtime_options::nint_route_warps();
        bool request_rows=row_setting && *row_setting;
        if(!row_setting && tokens==1 && nint_dispatch->pools.size()==1) {
            const auto& w=pools[nint_dispatch->pools.front()].nint;
            if(neuron_len==2560 && out_per_expert==640 && w.gs==28 && w.bits==5 &&
                    nint_late_scale_enabled(neuron_len,int(w.bits),x.get_device())) {
                thread_local int cached_device=-1,major=0,minor=0;
                if(cached_device!=x.get_device()) {
                    MFQ_CUDA_CHECK(cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,x.get_device()));
                    MFQ_CUDA_CHECK(cudaDeviceGetAttribute(&minor,cudaDevAttrComputeCapabilityMinor,x.get_device()));
                    cached_device=x.get_device();
                }
                request_rows=major==8 && minor==6;
            }
        }
        bool prepare_rows=request_rows && nint_dispatch->pools.size()==1 &&
            (!direct_setting || *direct_setting) && (!group_setting || *group_setting) &&
            (!hint_setting || *hint_setting) && (!warp_setting || *warp_setting!=4);
        if(prepare_rows) {
            const auto& pool=pools[nint_dispatch->pools.front()];const auto& w=pool.nint;
            prepare_rows=(w.gs==24 || w.gs==28) && w.bits>=4 && w.bits<=6;
            if(prepare_rows) {
                if(!plan.row_metadata_workspace.defined())plan.row_metadata_workspace=mfq_tensor_backend::empty(
                    {w.out,4},w.q_packed.options().dtype(mfq_tensor_backend::kInt32));
                moe_quantize_shared_nint_rows_cuda(x.reshape({input_rows,neuron_len}),plan.quantize_descriptors,
                    plan.total_groups,!input_prequantized,w.row_q_bit_offsets,w.neuron_scale,w.neuron_min,
                    plan.row_metadata_workspace,pool.expert_local,route.ids,out_per_expert,pool.local_experts);
            }
        }
        if(!prepare_rows && !input_prequantized)moe_quantize_shared_input_cuda(x.reshape({input_rows,neuron_len}),
            plan.quantize_descriptors,plan.total_groups);
        for(int index:nint_dispatch->pools) {
            const auto& w=pools[index].nint;
            quantized.insert({input_rows,int(w.ng),int(w.gs),x.get_device(),identity});
        }
        bool direct=false;
        if(nint_dispatch->pools.size()==1) {
            const auto& pool=pools[nint_dispatch->pools.front()];const auto& w=pool.nint;
            auto& workspace=activation_workspace(x,input_rows,int(w.ng),int(w.gs),identity);
            const auto& row_metadata=prepare_rows?plan.row_metadata_workspace:w.route_metadata;
            const NintSingleRouteWeight view{{reinterpret_cast<int64_t>(w.q_packed.data_ptr()),
                reinterpret_cast<int64_t>(w.row_q_bits.data_ptr()),reinterpret_cast<int64_t>(w.row_q_bit_offsets.data_ptr()),
                reinterpret_cast<int64_t>(w.sub_scale.data_ptr()),reinterpret_cast<int64_t>(w.sub_min.data_ptr()),
                reinterpret_cast<int64_t>(w.neuron_scale.data_ptr()),reinterpret_cast<int64_t>(w.neuron_min.data_ptr()),
                reinterpret_cast<int64_t>(workspace.qx.data_ptr()),reinterpret_cast<int64_t>(workspace.xscale.data_ptr())},
                {int(w.ng),int(w.gs),int(w.q_packed.numel()/pool.local_experts)},pool.local_experts,
                row_metadata.defined()?reinterpret_cast<int64_t>(row_metadata.data_ptr()):0};
            direct=nint_try_single_route_cuda(view,pool.expert_local,route.ids,output,x.dim()==3,int(w.bits),prepare_rows);
        }
        if(!direct) {
        mfq::cuda::moe_update_nvq_maps(reinterpret_cast<const uint64_t*>(nint_dispatch->map_ptrs.data_ptr<int64_t>()),
            int(nint_dispatch->pools.size()),n_experts,nint_dispatch->expert_pool.data_ptr<int32_t>(),
            nint_dispatch->expert_local.data_ptr<int32_t>(),mfq_current_cuda_stream());
        MFQ_CUDA_CHECK(cudaGetLastError());
        const auto& hint=pools[nint_dispatch->pools.front()].nint;
        nint_moe_grouped_matmul_hinted_cuda(plan.weight_ptrs,nint_dispatch->pool_params,
            nint_dispatch->expert_pool,nint_dispatch->expert_local,route.ids,output,neuron_len,x.dim()==3,
            nint_dispatch->pools.size()==1?int(hint.gs):0,int(hint.bits));
        }
    }

    for (const auto & pool : pools) {
        if(pool.family==MixedMoeFamily::Nint && use_nint_decode)continue;
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
                pool.family == MixedMoeFamily::Fp8Sq) {
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
            value = kl_mmq.prepare_activation(value);
            ++kl_mmq.moe_calls;
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
            ++kl_mmq.fallback_calls;
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
            const int route_tile_m = tokens<=8 ? 8 : select_nint_prefill_route_tile(
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
            bool decode_done=false;
            if(gs==24 && epilogue_mode==0 &&
               ((pool.nvq.kernel_format==1 && pool.nvq.sub_bits==3) ||
                (pool.nvq.kernel_format==8 && pool.nvq.sub_bits==4))) {
                const auto& w=pool.nvq;Nvq1DecodeView view;
                const std::array<mfq_tensor_backend::Tensor,7> fields={w.indices_packed,w.aux_packed,
                    w.sub_scale_packed,w.neuron_scale,w.codebook,w.decode_records,w.integer_codebook};
                for(int i=0;i<7;++i)if(fields[i].defined())view.pointers[i]=reinterpret_cast<int64_t>(fields[i].data_ptr());
                view.sizes[0]=w.indices_packed.numel();view.sizes[1]=w.aux_packed.numel();
                view.sizes[2]=w.sub_scale_packed.numel();view.sizes[3]=w.decode_records.defined()?w.decode_records.numel():0;
                view.groups=int(w.ng);view.vectors=int((neuron_len+7)/8);
                view.format=int(w.kernel_format);view.local_experts=pool.local_experts;
                decode_done=nvq1_try_decode_cuda(view,value,route.ids,pool.expert_local,qx,xscale,
                    output,n_experts,out_per_expert,input_quantized);
            }
            if(decode_done)activation_quantized_after_call=true;
            else
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

mfq_tensor_backend::Tensor MixedMoeRuntime::forward_glu_output(
        const CudaExecutionConfig& config,
        KlMmqState& kl_mmq,
        bool force_prefill_mma_off,
        bool force_pool_path,
        mfq_tensor_backend::Tensor x,
        const MoeRoutePlan & route,
        bool gelu) const {
    if (nint_only() && kl_mmq.mode == KlMmqMode::Default) {
        return forward(
            config, kl_mmq, force_prefill_mma_off, force_pool_path,
            x, route, false, gelu ? 2 : 1);
    }
    auto gate_up = forward(
        config, kl_mmq, force_prefill_mma_off, force_pool_path,
        x, route);
    return gelu
        ? moe_geglu_split_cuda(gate_up)
        : moe_swiglu_split_cuda(gate_up);
}

bool MixedMoeRuntime::supports_clamped_swiglu() const {
    return !pools.empty();
}

mfq_tensor_backend::Tensor MixedMoeRuntime::forward_clamped_swiglu(
        const CudaExecutionConfig& config,
        KlMmqState& kl_mmq,
        bool force_prefill_mma_off,
        bool force_pool_path,
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
    return forward(
        config, kl_mmq, force_prefill_mma_off, force_pool_path,
        activation, route);
}
