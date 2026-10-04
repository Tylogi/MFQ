#include "weight_loader.h"
#include "../ops/format.h"
#include "../kernels/mfq_cuda_quant_ops.h"
#include "mfq/mxfp4_sq_decode.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <numeric>

#include "moe_expert_cache.h"
#include "mfe_expert_store.h"

const mfq::TensorMetadata& require_tensor(
        const mfq::ModelSource& source,
        std::string_view name) {
    const auto* tensor = source.find_tensor(name);
    if (tensor == nullptr) {
        throw std::runtime_error("missing tensor: " + std::string(name));
    }
    return *tensor;
}

bool has_tensor(
        const mfq::ModelSource& source,
        std::string_view name) noexcept {
    return source.find_tensor(name) != nullptr;
}

bool has_tensor_prefix(
        const mfq::ModelSource& source,
        std::string_view prefix) {
    for (const auto& tensor : source.tensors()) {
        if (tensor.name == prefix ||
            (tensor.name.size() > prefix.size() &&
             tensor.name.compare(0, prefix.size(), prefix) == 0 &&
             tensor.name[prefix.size()] == '.')) {
            return true;
        }
    }
    return false;
}

std::vector<std::uint8_t> read_tensor(
        const mfq::ModelSource& source,
        std::string_view name) {
    const auto& tensor = require_tensor(source, name);
    if (tensor.nbytes > std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error("model tensor is too large to read");
    }
    std::vector<std::uint8_t> result(
        static_cast<std::size_t>(tensor.nbytes));
    source.read_range_into(
        name, 0, reinterpret_cast<std::byte*>(result.data()), result.size());
    return result;
}

static std::vector<std::uint8_t> read_tensor(
        bool drop_file_cache,
        const mfq::ModelSource& source,
        std::string_view name) {
    auto result = read_tensor(source, name);
    if (drop_file_cache) source.drop_file_cache();
    return result;
}

std::vector<std::uint8_t> read_asset(
        const mfq::ModelSource& source,
        std::string_view name) {
    const auto bytes = source.read_asset(name);
    std::vector<std::uint8_t> result(bytes.size());
    if (!bytes.empty()) {
        std::memcpy(result.data(), bytes.data(), bytes.size());
    }
    return result;
}

std::string read_asset_text(
        const mfq::ModelSource& source,
        std::string_view name) {
    const auto bytes = source.read_asset(name);
    return {
        reinterpret_cast<const char*>(bytes.data()), bytes.size()};
}

mfq_tensor_backend::Tensor load_dense_gpu(
        CudaExecutionContext& execution,
        const mfq::ModelSource& mfq,
        const std::string& name) {
    const bool cpu_layer = execution.loading_cpu_layer;
    MfqCudaGuard guard(active_weight_load_device(execution));
    const auto & rec = require_tensor(mfq, name);
    if (rec.dtype != "NINT8-0" && rec.dtype != "NINT") {
        auto value = load_dense_cpu(execution, mfq, name);
        if (rec.dtype == "BF16" || rec.dtype == "F16")
            value = value.to(mfq_tensor_backend::kFloat32);
        return cpu_layer ? value : value.to(mfq_tensor_backend::kCUDA).contiguous();
    }
    auto blob = read_tensor(execution.drop_file_cache, mfq, name);
    if (rec.dtype == "NINT8-0") {
        const auto source = unpack_nint8_zero(blob);
        if (cpu_layer) {
            return dequant_nint8_zero_cpu(source);
        }
        const auto packed = to_gpu_nint8_zero(source);
        auto dense = nint8_zero_dequant_cuda(
            packed.q_packed,
            packed.q8_zero_scale,
            packed.neuron_len)
            .to(mfq_tensor_backend::kFloat32)
            .contiguous();
        if (packed.shape.size() != 2 ||
            dense.size(0) != packed.shape[0] ||
            dense.size(1) != packed.shape[1]) {
            throw std::runtime_error(
                "NINT8-0 dense tensor shape mismatch: " + name);
        }
        return dense;
    }
    auto dense = dequant_nint_dense_f32(to_gpu_nint(unpack_nint(blob)));
    return cpu_layer
        ? dense.cpu().contiguous()
        : dense;
}

static bool is_nint_linear_dtype(const std::string & dtype) {
    return dtype == "NINT";
}

static bool is_nvq_linear_dtype(const std::string & dtype) {
    return dtype == "NVQ" || dtype == "NPQ";
}

static TensorParallelAxis infer_tensor_parallel_axis(
        const std::string & name) {
    const auto ends_with = [&name](std::string_view suffix) {
        return name.size() >= suffix.size() &&
            std::string_view(name).substr(name.size() - suffix.size()) == suffix;
    };
    if (name == "model.token_embedding.weight" ||
        ends_with(".token_embedding.weight") ||
        ends_with(".text_embedding.weight") ||
        (name.find(".code_embedding.") != std::string::npos &&
         ends_with(".weight"))) {
        return TensorParallelAxis::Mirrored;
    }
    if (name == "model.output.weight" ||
        name.find(".code_output.") != std::string::npos) {
        return TensorParallelAxis::Output;
    }
    if (ends_with(".mlp.down.weight") ||
        ends_with(".mlp.shared_expert.down.weight") ||
        ends_with(".attention.output.weight") ||
        ends_with(".linear_attention.output.weight") ||
        ends_with(".attention.output_a.weight") ||
        ends_with(".attention.output_b.weight") ||
        ends_with(".projector.output.weight")) {
        return TensorParallelAxis::Input;
    }
    return TensorParallelAxis::Output;
}

static std::vector<mfq::TensorParallelSlice>
plan_parallel_slices(
        int64_t extent,
        int64_t preferred_granularity,
        const ParallelConfig & config,
        const std::string & name) {
    (void)name;
    if (!config.enabled()) {
        throw std::runtime_error(
            "parallel slice planning requires at least two ranks");
    }
    int64_t granularity = std::max<int64_t>(
        1, std::min<int64_t>(
            preferred_granularity,
            extent / static_cast<int64_t>(config.devices.size())));
    while (granularity > 1 &&
           extent < static_cast<int64_t>(config.devices.size()) *
               granularity) {
        granularity /= 2;
    }
    auto slices = mfq::plan_tensor_parallel_slices(
        extent, granularity, config.devices, config.split);
    mfq::validate_tensor_parallel_slices(
        slices, extent, granularity);
    return slices;
}

static std::vector<mfq::TensorParallelSlice>
plan_moe_expert_parallel_slices(
        const ParallelConfig& parallel,
        int64_t extent,
        const std::string & name) {
    return plan_parallel_slices(extent, 1, parallel, name);
}

QuantLinear load_quant_linear(
        CudaExecutionContext& execution,
        const mfq::ModelSource & mfq,
        const std::string & name,
        std::optional<TensorParallelAxis> axis_override,
        const std::vector<mfq::TensorParallelSlice> *
            slices_override) {
    const bool cpu_layer = execution.loading_cpu_layer;
    const auto & dtype = require_tensor(mfq, name).dtype;
    QuantLinear result;
    const TensorParallelAxis axis =
        axis_override.value_or(
            infer_tensor_parallel_axis(name));
    if (slices_override != nullptr &&
        axis != TensorParallelAxis::Output) {
        throw std::runtime_error(
            "explicit tensor-parallel slices require an output-axis weight");
    }
    auto select_slices = [&](
            int64_t extent,
            int64_t preferred) {
        if (slices_override == nullptr) {
            return plan_parallel_slices(
                extent, preferred, execution.tensor_parallel, name);
        }
        auto slices = *slices_override;
        mfq::validate_tensor_parallel_slices(
            slices, extent, 1);
        if (slices.size() !=
            execution.tensor_parallel.devices.size()) {
            throw std::runtime_error(
                "explicit tensor-parallel slice count mismatch");
        }
        for (size_t index = 0;
             index < slices.size(); ++index) {
            if (slices[index].device !=
                execution.tensor_parallel.devices[index]) {
                throw std::runtime_error(
                    "explicit tensor-parallel device order mismatch");
            }
        }
        return slices;
    };
    result.tensor_parallel_axis = axis;
    if (is_nint_linear_dtype(dtype)) {
        result.kind = QuantLinearKind::Nint;
        const auto blob = read_tensor(
            execution.drop_file_cache, mfq, name);
        if (dtype == "NINT8-0") {
            const auto cpu = unpack_nint8_zero(blob);
            result.logical_out = cpu.out;
            result.logical_neuron_len = cpu.neuron_len;
            if (execution.tensor_parallel.enabled() &&
                axis != TensorParallelAxis::Mirrored) {
                const int64_t extent =
                    axis == TensorParallelAxis::Output
                    ? cpu.out : cpu.ng;
                const int64_t preferred =
                    axis == TensorParallelAxis::Output
                    ? 128
                    : 4;
                for (const auto & slice :
                     select_slices(
                         extent, preferred)) {
                    auto shard_cpu =
                        axis == TensorParallelAxis::Output
                        ? slice_nint8_zero_cpu_output(
                            cpu, slice.begin, slice.end)
                        : slice_nint8_zero_cpu_input_groups(
                            cpu, slice.begin, slice.end);
                    QuantLinearShard shard;
                    shard.device = slice.device;
                    shard.kind = QuantLinearKind::Nint;
                    shard.output_begin =
                        axis == TensorParallelAxis::Output
                        ? slice.begin : 0;
                    shard.output_end =
                        axis == TensorParallelAxis::Output
                        ? slice.end : cpu.out;
                    shard.input_begin =
                        axis == TensorParallelAxis::Input
                        ? slice.begin * 32 : 0;
                    shard.input_end =
                        axis == TensorParallelAxis::Input
                        ? std::min<int64_t>(
                            slice.end * 32,
                            cpu.neuron_len)
                        : cpu.neuron_len;
                    shard.nint =
                        to_cuda_device_nint8_zero(
                            shard_cpu, slice.device);
                    result.tensor_parallel_shards.push_back(
                        std::move(shard));
                }
            } else {
                if (cpu_layer) {
                    result.nint = to_device_nint8_zero(cpu, false);
                } else {
                    MfqCudaGuard guard(active_weight_load_device(execution));
                    result.nint =
                        to_device_nint8_zero(cpu, true);
                }
            }
        } else {
            const auto cpu = unpack_nint(blob);
            result.logical_out = cpu.out;
            result.logical_neuron_len = cpu.neuron_len;
            if (execution.tensor_parallel.enabled() &&
                axis != TensorParallelAxis::Mirrored) {
                const int64_t extent =
                    axis == TensorParallelAxis::Output
                    ? cpu.out : cpu.ng;
                const int64_t preferred =
                    axis == TensorParallelAxis::Output
                    ? 128
                    : std::lcm<int64_t>(cpu.gs, 128) / cpu.gs;
                for (const auto & slice :
                     select_slices(
                         extent, preferred)) {
                    auto shard_cpu =
                        axis == TensorParallelAxis::Output
                        ? slice_nint_cpu_output(
                            cpu, slice.begin, slice.end)
                        : slice_nint_cpu_input_groups(
                            cpu, slice.begin, slice.end);
                    QuantLinearShard shard;
                    shard.device = slice.device;
                    shard.kind = QuantLinearKind::Nint;
                    shard.output_begin =
                        axis == TensorParallelAxis::Output
                        ? slice.begin : 0;
                    shard.output_end =
                        axis == TensorParallelAxis::Output
                        ? slice.end : cpu.out;
                    shard.input_begin =
                        axis == TensorParallelAxis::Input
                        ? slice.begin * cpu.gs : 0;
                    shard.input_end =
                        axis == TensorParallelAxis::Input
                        ? std::min<int64_t>(
                            slice.end * cpu.gs,
                            cpu.neuron_len)
                        : cpu.neuron_len;
                    shard.nint = to_cuda_device_nint(
                        shard_cpu, slice.device);
                    result.tensor_parallel_shards.push_back(
                        std::move(shard));
                }
            } else {
                if (cpu_layer) {
                    result.nint = to_device_nint(cpu, false);
                } else {
                    MfqCudaGuard guard(active_weight_load_device(execution));
                    result.nint =
                        to_device_nint(cpu, true);
                }
            }
        }
    } else if (is_nvq_linear_dtype(dtype)) {
        result.kind = QuantLinearKind::Nvq;
        const auto cpu = unpack_nvq(
            read_tensor(execution.drop_file_cache, mfq, name), dtype);
        result.logical_out = cpu.out;
        result.logical_neuron_len = cpu.neuron_len;
        if (execution.tensor_parallel.enabled() &&
            axis != TensorParallelAxis::Mirrored) {
            const int64_t extent =
                axis == TensorParallelAxis::Output
                ? cpu.out : cpu.ng;
            const int64_t preferred =
                axis == TensorParallelAxis::Output
                ? 128
                : std::lcm<int64_t>(cpu.gs, 128) / cpu.gs;
            for (const auto & slice :
                 select_slices(
                     extent, preferred)) {
                auto shard_cpu = slice_nvq_cpu(
                    cpu, axis, slice.begin, slice.end);
                QuantLinearShard shard;
                shard.device = slice.device;
                shard.kind = QuantLinearKind::Nvq;
                shard.output_begin =
                    axis == TensorParallelAxis::Output
                    ? slice.begin : 0;
                shard.output_end =
                    axis == TensorParallelAxis::Output
                    ? slice.end : cpu.out;
                shard.input_begin =
                    axis == TensorParallelAxis::Input
                    ? slice.begin * cpu.gs : 0;
                shard.input_end =
                    axis == TensorParallelAxis::Input
                    ? std::min<int64_t>(
                        slice.end * cpu.gs,
                        cpu.neuron_len)
                    : cpu.neuron_len;
                shard.nvq = to_cuda_device_nvq(
                    shard_cpu, slice.device);
                result.tensor_parallel_shards.push_back(
                    std::move(shard));
            }
        } else {
            if (cpu_layer) {
                result.nvq = to_device_nvq(cpu, false, execution.config);
            } else {
                MfqCudaGuard guard(active_weight_load_device(execution));
                result.nvq = to_device_nvq(cpu, true, execution.config);
            }
        }
    } else if (dtype == "MXFP4-SQ") {
        result.kind = QuantLinearKind::Mxfp4Sq;
        const auto payload = read_tensor(
            execution.drop_file_cache, mfq, name);
        const auto layout = mfq::sq::parse(payload.data(), payload.size());
        result.logical_out = layout.outputs;
        result.logical_neuron_len = layout.width;
        if (!cpu_layer && execution.tensor_parallel.enabled() &&
                axis != TensorParallelAxis::Mirrored) {
            const bool output_axis = axis == TensorParallelAxis::Output;
            for (const auto& slice : select_slices(
                    output_axis ? layout.outputs : layout.width,
                    output_axis ? 128 : 32)) {
                QuantLinearShard shard;
                shard.kind = result.kind;
                shard.device = slice.device;
                shard.output_begin = output_axis ? slice.begin : 0;
                shard.output_end = output_axis ? slice.end : layout.outputs;
                shard.input_begin = output_axis ? 0 : slice.begin;
                shard.input_end = output_axis ? layout.width : slice.end;
                std::vector<int64_t> rows(
                    shard.output_end - shard.output_begin);
                std::iota(rows.begin(), rows.end(), shard.output_begin);
                shard.mxfp4_sq.weight = to_device_mxfp4_sq(
                    mfq::sq::select_rows(
                        payload, rows, shard.input_begin, shard.input_end),
                    true, slice.device);
                result.tensor_parallel_shards.push_back(std::move(shard));
            }
        } else {
            result.mxfp4_sq.weight = to_device_mxfp4_sq(
                payload, !cpu_layer,
                cpu_layer ? -1 : active_weight_load_device(execution));
        }
    } else if (mfq::fp8sq::is_dtype(dtype)) {
        result.kind = QuantLinearKind::Fp8Sq;
        const auto payload = read_tensor(
            execution.drop_file_cache, mfq, name);
        const auto layout = mfq::fp8sq::parse(
            dtype, payload.data(), payload.size());
        result.logical_out = layout.outputs;
        result.logical_neuron_len = layout.width;
        if (!cpu_layer && execution.tensor_parallel.enabled() &&
                axis != TensorParallelAxis::Mirrored) {
            const bool output_axis = axis == TensorParallelAxis::Output;
            const auto alignment = output_axis
                ? layout.block_rows : layout.block_columns;
            for (const auto& slice : select_slices(
                    output_axis ? layout.outputs : layout.width, alignment)) {
                QuantLinearShard shard;
                shard.kind = result.kind;
                shard.device = slice.device;
                shard.output_begin = output_axis ? slice.begin : 0;
                shard.output_end = output_axis ? slice.end : layout.outputs;
                shard.input_begin = output_axis ? 0 : slice.begin;
                shard.input_end = output_axis ? layout.width : slice.end;
                shard.fp8_sq.weight = to_device_fp8_sq(
                    dtype, mfq::fp8sq::slice(
                        dtype, payload,
                        shard.output_begin, shard.output_end,
                        shard.input_begin, shard.input_end),
                    true, slice.device);
                result.tensor_parallel_shards.push_back(std::move(shard));
            }
        } else {
            result.fp8_sq.weight = to_device_fp8_sq(
                dtype, payload, !cpu_layer,
                cpu_layer ? -1 : active_weight_load_device(execution));
        }
    } else if (dtype == "MXFP4") {
        result.kind = QuantLinearKind::Mxfp4;
        const auto cpu = unpack_mxfp4(
            read_tensor(execution.drop_file_cache, mfq, name));
        result.logical_out = cpu.out;
        result.logical_neuron_len = cpu.neuron_len;
        if (execution.tensor_parallel.enabled() &&
                axis != TensorParallelAxis::Mirrored) {
            const int64_t extent = axis == TensorParallelAxis::Output
                ? cpu.out : cpu.neuron_len;
            const int64_t preferred = axis == TensorParallelAxis::Output
                ? 128 : 32;
            for (const auto & slice : select_slices(extent, preferred)) {
                auto shard_cpu = slice_mxfp4_cpu(
                    cpu, axis, slice.begin, slice.end);
                QuantLinearShard shard;
                shard.device = slice.device;
                shard.kind = QuantLinearKind::Mxfp4;
                shard.output_begin = axis == TensorParallelAxis::Output
                    ? slice.begin : 0;
                shard.output_end = axis == TensorParallelAxis::Output
                    ? slice.end : cpu.out;
                shard.input_begin = axis == TensorParallelAxis::Input
                    ? slice.begin : 0;
                shard.input_end = axis == TensorParallelAxis::Input
                    ? slice.end : cpu.neuron_len;
                shard.mxfp4 = to_device_mxfp4(
                    shard_cpu, true, slice.device);
                result.tensor_parallel_shards.push_back(std::move(shard));
            }
        } else {
            result.mxfp4.weight = to_device_mxfp4(
                cpu, !cpu_layer,
                cpu_layer ? -1 : active_weight_load_device(execution));
        }
    } else if (dtype == "MXFP8") {
        result.kind = QuantLinearKind::Mxfp8;
        const auto cpu = unpack_mxfp8(
            read_tensor(execution.drop_file_cache, mfq, name));
        result.logical_out = cpu.out;
        result.logical_neuron_len = cpu.neuron_len;
        if (execution.tensor_parallel.enabled() &&
                axis != TensorParallelAxis::Mirrored) {
            const int64_t extent = axis == TensorParallelAxis::Output
                ? cpu.out : cpu.neuron_len;
            for (const auto & slice : select_slices(extent, 128)) {
                auto shard_cpu = slice_mxfp8_cpu(
                    cpu, axis, slice.begin, slice.end);
                QuantLinearShard shard;
                shard.device = slice.device;
                shard.kind = QuantLinearKind::Mxfp8;
                shard.output_begin = axis == TensorParallelAxis::Output
                    ? slice.begin : 0;
                shard.output_end = axis == TensorParallelAxis::Output
                    ? slice.end : cpu.out;
                shard.input_begin = axis == TensorParallelAxis::Input
                    ? slice.begin : 0;
                shard.input_end = axis == TensorParallelAxis::Input
                    ? slice.end : cpu.neuron_len;
                shard.mxfp8 = to_cuda_device_mxfp8(
                    shard_cpu, slice.device);
                result.tensor_parallel_shards.push_back(
                    std::move(shard));
            }
        } else if (cpu_layer) {
            result.mxfp8.weight = to_device_mxfp8(cpu, false);
        } else {
            result.mxfp8.weight = to_cuda_device_mxfp8(
                cpu, active_weight_load_device(execution));
        }
    } else if (dtype == "BF16" || dtype == "F16" || dtype == "F32") {
        result.kind = QuantLinearKind::Dense;
        auto cpu = load_dense_cpu(execution, mfq, name);
        MFQ_RUNTIME_CHECK(cpu.dim() == 2, "dense linear tensor must be rank 2: ", name);
        result.logical_out = cpu.size(0);
        result.logical_neuron_len = cpu.size(1);
        // Keep native floating-point linears whole on the primary TP rank.
        // Splitting these matrices changes the cuBLAS GEMM geometry and causes
        // materially larger drift than the weight formats under test. They
        // are a small fraction of the model; routed experts and MXFP8/NINT/NVQ
        // weights remain sharded.
        const bool shard_native_float =
            execution.config.tensor_parallel_shard_native_float;
        if (shard_native_float && execution.tensor_parallel.enabled() &&
                axis != TensorParallelAxis::Mirrored) {
            const int64_t extent = axis == TensorParallelAxis::Output
                ? cpu.size(0) : cpu.size(1);
            for (const auto & slice : select_slices(extent, 128)) {
                QuantLinearShard shard;
                shard.device = slice.device;
                shard.kind = QuantLinearKind::Dense;
                shard.output_begin = axis == TensorParallelAxis::Output
                    ? slice.begin : 0;
                shard.output_end = axis == TensorParallelAxis::Output
                    ? slice.end : cpu.size(0);
                shard.input_begin = axis == TensorParallelAxis::Input
                    ? slice.begin : 0;
                shard.input_end = axis == TensorParallelAxis::Input
                    ? slice.end : cpu.size(1);
                const int64_t dimension =
                    axis == TensorParallelAxis::Output ? 0 : 1;
                MfqCudaGuard guard(slice.device);
                shard.dense = cpu.narrow(
                        dimension, slice.begin,
                        slice.end - slice.begin)
                    .to(mfq_tensor_backend::Device(mfq_tensor_backend::kCUDA, slice.device))
                    .contiguous();
                result.tensor_parallel_shards.push_back(
                    std::move(shard));
            }
        } else if (cpu_layer) {
            result.dense = cpu.contiguous();
        } else {
            MfqCudaGuard guard(active_weight_load_device(execution));
            result.dense = cpu.to(mfq_tensor_backend::kCUDA).contiguous();
        }
    } else {
        throw std::runtime_error(
            "linear tensor must be NINT/NVQ/MXFP4-SQ/MXFP8-SQ/FP8-128SQ/MXFP4/MXFP8/BF16/F16/F32: " +
            name + " dtype=" + dtype);
    }
    return result;
}

bool is_quant_dtype(const std::string & dtype) {
    return is_nint_linear_dtype(dtype) ||
        is_nvq_linear_dtype(dtype) || dtype == "MXFP4-SQ" ||
        mfq::fp8sq::is_dtype(dtype) ||
        dtype == "MXFP4" || dtype == "MXFP8";
}

QuantLinearGroup load_quant_group(
    CudaExecutionContext& execution,
    const mfq::ModelSource & mfq, const std::vector<std::string> & names,
    size_t required_compatible_prefix,
    const std::vector<mfq::TensorParallelSlice> *
        slices_override,
    bool preserve_projection_boundaries) {
    std::vector<QuantLinear> layers;
    layers.reserve(names.size());
    for (const auto & name : names) {
        layers.push_back(
            load_quant_linear(
                execution, mfq, name,
                slices_override != nullptr
                    ? std::optional<TensorParallelAxis>(
                        TensorParallelAxis::Output)
                    : std::nullopt,
                slices_override));
    }
    if (required_compatible_prefix > layers.size()) {
        throw std::runtime_error("invalid required quantized-group prefix length");
    }
    // A compatible prefix is fused opportunistically by make_quant_group().
    // Mixed-precision Q/K and gate/up pairs remain separate QuantLinear
    // branches and preserve the recipe-selected layouts.
    return make_quant_group(
        execution, std::move(layers), preserve_projection_boundaries);
}

static std::vector<mfq::TensorParallelSlice>
tensor_parallel_output_slices_for_input(
        const QuantLinear & input_parallel) {
    if (!input_parallel.tensor_parallel() ||
        input_parallel.tensor_parallel_axis !=
            TensorParallelAxis::Input) {
        return {};
    }
    std::vector<mfq::TensorParallelSlice> result;
    result.reserve(
        input_parallel.tensor_parallel_shards.size());
    for (const auto & shard :
         input_parallel.tensor_parallel_shards) {
        result.push_back({
            shard.device,
            shard.input_begin,
            shard.input_end,
        });
    }
    mfq::validate_tensor_parallel_slices(
        result,
        input_parallel.neuron_len(),
        1);
    return result;
}

QuantLinearGroup load_paired_gate_up(
        CudaExecutionContext& execution,
        const mfq::ModelSource & mfq,
        const std::vector<std::string> & names,
        const QuantLinear & down,
        size_t required_compatible_prefix,
        bool preserve_projection_boundaries) {
    auto slices =
        tensor_parallel_output_slices_for_input(
            down);
    return slices.empty()
        ? load_quant_group(
            execution, mfq, names,
            required_compatible_prefix,
            nullptr,
            preserve_projection_boundaries)
        : load_quant_group(
            execution, mfq, names,
            required_compatible_prefix,
            &slices,
            preserve_projection_boundaries);
}

MfeCpu load_mfe_cpu(
        const mfq::ModelSource & mfq, const std::string & name) {
    if (require_tensor(mfq, name).dtype != "MFE") {
        throw std::runtime_error("expert tensor must use MFE: " + name);
    }
    return unpack_mfe(read_tensor(mfq, name));
}

std::shared_ptr<MixedMoeRuntime> make_mxfp4_range_runtime(
        const mfq::cuda::MfeMxfp4ExpertStore & store) {
    auto runtime = std::make_shared<MixedMoeRuntime>();
    runtime->n_experts = store.num_experts();
    runtime->out_per_expert = store.out_per_expert();
    runtime->neuron_len = store.neuron_len();
    MixedMoePool pool;
    pool.family = MixedMoeFamily::Mxfp4;
    pool.local_experts = store.num_experts();
    pool.mxfp4.out =
        static_cast<int64_t>(store.num_experts()) * store.out_per_expert();
    pool.mxfp4.neuron_len = store.neuron_len();
    std::vector<int32_t> local(static_cast<size_t>(store.num_experts()));
    std::iota(local.begin(), local.end(), int32_t{0});
    pool.expert_local = mfq_tensor_backend::from_blob(
        local.data(),
        {static_cast<int64_t>(local.size())},
        mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt32))
        .clone();
    runtime->pools.push_back(std::move(pool));
    return runtime;
}

std::shared_ptr<mfq::NintRows> load_nint_row_table(
        const mfq::ModelSource& source, const std::string& name) {
    const auto& metadata = require_tensor(source, name);
    MFQ_RUNTIME_CHECK(metadata.dtype == "NINT", "NINT row table requires packed NINT");
    MFQ_RUNTIME_CHECK(metadata.nbytes <= std::numeric_limits<std::size_t>::max(),
        "NINT row table size overflow");
    auto read = source.tensor_reader(name);
    return std::make_shared<mfq::NintRows>(static_cast<std::size_t>(metadata.nbytes),
        [read = std::move(read)](std::size_t offset, std::uint8_t* out, std::size_t count) {
            read(offset, reinterpret_cast<std::byte*>(out), count);
        });
}

MfeWeight load_mfe_gpu(
        CudaExecutionContext& execution,
        const mfq::ModelSource & mfq, const std::string & name,
        bool cacheable,
        int layer_id,
        const std::string & projection_role) {
    auto& cache = execution.moe_expert_cache;
    if (cache && cacheable &&
            !moe_parallel_config(execution).enabled() &&
            execution.config.moe_ssd_ranges) {
        const auto & record = require_tensor(mfq, name);
        try {
            auto store =
                std::make_shared<mfq::cuda::MfeMxfp4ExpertStore>(
                    mfq::cuda::MfqRecordRange{
                        name,
                        record.dtype,
                        {},
                        0,
                        record.nbytes,
                        [&mfq, name](
                                std::uint64_t offset,
                                std::span<std::uint8_t> destination) {
                            mfq.read_range_into(
                                name,
                                offset,
                                reinterpret_cast<std::byte*>(destination.data()),
                                destination.size());
                        },
                    });
            auto runtime = make_mxfp4_range_runtime(*store);
            return cache_moe_weight(
                cache,
                name,
                runtime,
                std::min(
                    execution.moe_cache_registration_min_slots,
                    runtime->n_experts),
                layer_id,
                projection_role, std::move(store));
        } catch (const mfq::cuda::MfeMxfp4Unsupported &) {
        }
    }
    auto cpu = load_mfe_cpu(mfq, name);
    const bool has_matrix_local_sq = std::any_of(
        cpu.pools.begin(), cpu.pools.end(), [](const MfeCpuPool & pool) {
            return pool.dtype == "MXFP4-SQ" ||
                mfq::fp8sq::is_dtype(pool.dtype);
        });
    if (moe_parallel_config(execution).enabled()) {
        auto slices = plan_moe_expert_parallel_slices(
            moe_parallel_config(execution), cpu.n_experts, name);
        MfeWeight result;
        result.n_experts = cpu.n_experts;
        result.out_per_expert =
            cpu.out_per_expert;
        result.neuron_len =
            cpu.neuron_len;
        for (const auto & slice : slices) {
            auto shard =
                std::make_shared<MfeWeight>(
                    to_cuda_device_moe_expert_slice(
                        cpu, slice.begin,
                        slice.end,
                        slice.device,
                        execution.config));
            result.expert_parallel_shards.push_back({
                slice.device,
                slice.begin,
                slice.end,
                std::move(shard),
            });
        }
        return result;
    }
    if (cache && cacheable && !has_matrix_local_sq) {
        auto runtime =
            make_mixed_moe_runtime(cpu, false);
        return cache_moe_weight(
            cache,
            name, runtime,
            std::min(
                execution.moe_cache_registration_min_slots,
                runtime->n_experts),
            layer_id,
            projection_role);
    }
    const bool all_nint = std::all_of(
        cpu.pools.begin(), cpu.pools.end(), [](const MfeCpuPool & pool) {
            return pool.dtype == "NINT";
        });
    return all_nint
        ? to_gpu_mfe(cpu)
        : to_gpu_mixed_moe(cpu, execution.config);
}

std::shared_ptr<MixedMoeRuntime> load_mfe_cpu_offloaded(
        const mfq::ModelSource & mfq, const std::string & name) {
    return make_mixed_moe_runtime(load_mfe_cpu(mfq, name), false);
}

namespace mfq::cuda::weight_loader {
namespace tb = mfq_tensor_backend;

Linear linear(CudaExecutionContext& execution, const mfq::ModelSource& file, const std::string& name) {
    auto weight=std::make_shared<QuantLinear>(load_quant_linear(execution, file,name));
    return [weight](CudaExecutionContext& execution, const Tensor& x) {
        return weight->forward(execution, x);
    };
}

Tensor dense(CudaExecutionContext& execution, const mfq::ModelSource& file, const std::string& name) {
    const auto& dtype=require_tensor(file, name).dtype;
    MFQ_RUNTIME_CHECK(dtype=="F32" || dtype=="F16" || dtype=="BF16",
        "expected a dense parameter: ",name);
    auto value=load_dense_gpu(execution, file,name);
    return value.to(dtype=="F16" ? tb::kFloat16 : dtype=="BF16" ? tb::kBFloat16 : tb::kFloat32);
}

Routed routed(CudaExecutionContext& execution, const mfq::ModelSource& file, const std::string& name, int layer,
    int64_t experts, int64_t output, int64_t input, const std::string& role) {
    const auto& dtype=require_tensor(file, name).dtype;
    if (dtype=="F32" || dtype=="F16" || dtype=="BF16") {
        MFQ_RUNTIME_CHECK(!moe_parallel_config(execution).enabled(),
            "expert parallelism requires packed routed tensors: ",name);
        auto w=dense(execution,file,name);
        MFQ_RUNTIME_CHECK(w.sizes().vec()==std::vector<int64_t>({experts,output,input}),
            "dense expert tensor shape mismatch: ",name);
        return [w,output,input](CudaExecutionContext&, const Tensor& x,const Tensor& ids) {
            const auto rows=ids.size(0), routes=ids.size(1);
            auto selected=w.index_select(0,ids.reshape({-1}).to(tb::kInt64)).reshape({rows,routes,output,input});
            auto source=x.dim()==2 ? x.unsqueeze(1).expand({rows,routes,input}) : x;
            return tb::matmul(selected,source.to(w.scalar_type()).unsqueeze(-1)).squeeze(-1);
        };
    }
    auto w=std::make_shared<MfeWeight>(load_mfe_gpu(execution, file,name,true,layer,role));
    MFQ_RUNTIME_CHECK(w->n_experts==experts && w->out_per_expert==output && w->neuron_len==input,
        "routed tensor shape mismatch: ",name);
    return [w,experts](CudaExecutionContext& execution, const Tensor& x,const Tensor& ids) {
        auto route=build_moe_route_plan(ids.to(tb::kInt32).contiguous(),int(experts));
        return w->forward(execution,x.contiguous(),route);
    };
}

Routed routed_gate_up(CudaExecutionContext& execution, const mfq::ModelSource& file, const std::string& mlp_prefix,
    int layer, int64_t experts, int64_t width, int64_t input, const std::string& role) {
    const auto base=mlp_prefix+".experts";
    const auto gate_name=base+".gate.weight",up_name=base+".up.weight";
    const bool has_gate=has_tensor(file, gate_name),has_up=has_tensor(file, up_name);
    MFQ_RUNTIME_CHECK(has_gate==has_up,"incomplete routed Gate/Up pair under ",base);
    if (!has_gate) return routed(execution,file,base+".gate_up.weight",layer,experts,2*width,input,role);
    auto gate=routed(execution,file,gate_name,layer,experts,width,input,role);
    auto up=routed(execution,file,up_name,layer,experts,width,input,role);
    return [gate=std::move(gate),up=std::move(up)](CudaExecutionContext& execution, const Tensor& x,const Tensor& ids) {
        return tb::cat({gate(execution,x,ids),up(execution,x,ids)},-1);
    };
}

void validate_load_options(
        const CudaExecutionContext& execution) {
    if (execution.tensor_parallel.enabled() ||
            execution.layer_placement.enabled() ||
            execution.n_gpu_layers >= 0 || execution.moe_expert_cache) {
        throw std::runtime_error(
            "native attention adapter supports expert parallelism, but "
            "tensor/layer parallelism and offload require a different placement path");
    }
}

} // namespace mfq::cuda::weight_loader

mfq_tensor_backend::Tensor load_dense_cpu(const CudaExecutionContext &execution,
                                          const mfq::ModelSource &source,
                                          const std::string &name) {
    namespace tb = mfq_tensor_backend;
    const auto &record = require_tensor(source, name);
    tb::ScalarType dtype;
    size_t item_size;
    if (record.dtype == "BF16") { dtype = tb::kBFloat16; item_size = 2; }
    else if (record.dtype == "F16") { dtype = tb::kFloat16; item_size = 2; }
    else if (record.dtype == "F32") { dtype = tb::kFloat32; item_size = 4; }
    else if (record.dtype == "I64") { dtype = tb::kInt64; item_size = 8; }
    else if (record.dtype == "I32") { dtype = tb::kInt32; item_size = 4; }
    else throw std::runtime_error("unsupported dense dtype: " + record.dtype + " tensor " + name);

    auto blob = read_tensor(source, name);
    if (execution.drop_file_cache) source.drop_file_cache();
    MFQ_RUNTIME_CHECK(blob.size() >= sizeof(uint32_t), "truncated dense tensor header: ", name);
    size_t offset = 0;
    const auto rank = read_u32_from(blob, offset);
    MFQ_RUNTIME_CHECK(rank <= (blob.size() - offset) / sizeof(int64_t),
                      "truncated dense tensor shape: ", name);
    std::vector<int64_t> shape(rank);
    int64_t numel = 1;
    for (auto &extent : shape) {
        extent = read_i64_from(blob, offset);
        MFQ_RUNTIME_CHECK(extent >= 0 &&
                              (extent == 0 || numel <= std::numeric_limits<int64_t>::max() / extent),
                          "invalid dense tensor shape: ", name);
        numel *= extent;
    }
    MFQ_RUNTIME_CHECK(!record.shape || *record.shape == shape,
                      "dense tensor metadata shape mismatch: ", name);
    const auto payload = blob.size() - offset;
    MFQ_RUNTIME_CHECK(payload % item_size == 0 && uint64_t(numel) == payload / item_size,
                      "dense tensor payload size mismatch: ", name);
    return tb::from_blob(blob.data() + offset, shape, tb::TensorOptions().dtype(dtype)).clone();
}

mfq_tensor_backend::Tensor load_dense_native_gpu(CudaExecutionContext &execution,
                                                 const mfq::ModelSource &source,
                                                 const std::string &name) {
    MfqCudaGuard guard(active_weight_load_device(execution));
    return load_dense_cpu(execution, source, name).to(mfq_tensor_backend::kCUDA).contiguous();
}

namespace mfq::cuda {

std::string load_model_config_json(
        const mfq::ModelSource& source,
        const std::string& external_path) {
    if (external_path.empty()) return source.model_config_json();
    std::ifstream input(external_path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open config: " + external_path);
    return {std::istreambuf_iterator<char>(input), {}};
}

void validate_model_source(const mfq::ModelSource& source) {
    if (source.find_tensor("model.token_embedding.weight") == nullptr) {
        throw std::runtime_error(
            "canonical text tensor inventory has no token embedding");
    }
    for (const auto& component : source.resolved_model_graph().components) {
        if (component.tensor_root != "runtime" &&
                !has_tensor_prefix(source, component.tensor_root)) {
            throw std::runtime_error(
                "model graph declares " + component.kind +
                " but its canonical tensor inventory is missing");
        }
    }
}

} // namespace mfq::cuda
