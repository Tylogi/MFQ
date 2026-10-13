#include "moe_weight_loader.h"

#include "ops/execution_options.h"
#include "ops/format.h"
#include "kernels/moe_ops.h"
#include "kernels/quant_ops.h"
#include "mfq_format_compat.h"
#include "ops/quant_linear.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <numeric>
#include "storage/quant_weight_loader.h"

using mfq_tensor_backend::indexing::Slice;
using namespace mfq::cuda::quant_format;

static MfeCpu unpack_mfe_impl(
        const std::vector<uint8_t> & blob,
        bool allow_partial) {
    if (blob.size() < 20) {
        throw std::runtime_error("invalid MFE header");
    }
    const bool version1 = std::memcmp(blob.data(), "NIM1", 4) == 0;
    const bool delta =
        std::memcmp(blob.data(), "MFD1", 4) == 0 ||
        std::memcmp(blob.data(), "NID2", 4) == 0;
    const bool version2 =
        std::memcmp(blob.data(), "MFE1", 4) == 0 ||
        std::memcmp(blob.data(), "NIM2", 4) == 0 || delta;
    if (!version1 && !version2) throw std::runtime_error("invalid MFE header");
    if (delta && !allow_partial) {
        throw std::runtime_error("MFE delta cannot be loaded as a full tensor");
    }
    size_t off = 4;
    MfeCpu result;
    result.n_experts = (int)read_u32_from(blob, off);
    result.out_per_expert = (int)read_u32_from(blob, off);
    result.neuron_len = (int)read_u32_from(blob, off);
    int pool_count = (int)read_u32_from(blob, off);
    if (result.n_experts <= 0 || result.out_per_expert <= 0 || result.neuron_len <= 0 ||
        pool_count <= 0 || pool_count > result.n_experts) {
        throw std::runtime_error("invalid MFE dimensions");
    }
    std::vector<int> owners((size_t)result.n_experts, -1);
    result.pools.reserve((size_t)pool_count);
    for (int pool_index = 0; pool_index < pool_count; ++pool_index) {
        const size_t pool_header_bytes = version1 ? 12 : 24;
        if (off + pool_header_bytes > blob.size()) {
            throw std::runtime_error("truncated MFE pool header");
        }
        uint32_t expert_count = read_u32_from(blob, off);
        uint32_t dtype_nbytes = version1 ? 0 : read_u32_from(blob, off);
        uint64_t payload_nbytes = read_u64_from(blob, off);
        uint64_t runtime_nbytes = version1 ? 0 : read_u64_from(blob, off);
        if (expert_count == 0 || expert_count > static_cast<uint32_t>(result.n_experts) ||
                expert_count > static_cast<uint32_t>(
                    std::numeric_limits<int>::max() / result.out_per_expert) ||
                dtype_nbytes > 32) {
            throw std::runtime_error("invalid MFE pool dimensions");
        }
        size_t ids_nbytes = (size_t)expert_count * sizeof(int32_t);
        size_t remaining = blob.size() - off;
        if (ids_nbytes > remaining) {
            throw std::runtime_error("truncated MFE pool payload");
        }
        remaining -= ids_nbytes;
        if (dtype_nbytes > remaining) {
            throw std::runtime_error("truncated MFE pool payload");
        }
        remaining -= dtype_nbytes;
        if (runtime_nbytes > remaining) {
            throw std::runtime_error("truncated MFE pool payload");
        }
        remaining -= static_cast<size_t>(runtime_nbytes);
        if (payload_nbytes > remaining) {
            throw std::runtime_error("truncated MFE pool payload");
        }
        MfeCpuPool pool;
        pool.expert_ids.resize(expert_count);
        std::memcpy(pool.expert_ids.data(), blob.data() + off, ids_nbytes);
        off += ids_nbytes;
        for (int32_t expert : pool.expert_ids) {
            if (expert < 0 || expert >= result.n_experts || owners[(size_t)expert] >= 0) {
                throw std::runtime_error("invalid or duplicate MFE expert id");
            }
            owners[(size_t)expert] = pool_index;
        }
        if (version2) {
            if (dtype_nbytes == 0) throw std::runtime_error("empty MFE pool dtype");
            pool.dtype.assign(
                reinterpret_cast<const char *>(blob.data() + off), dtype_nbytes);
            pool.dtype = std::string(
                mfq::canonical_format_dtype(pool.dtype));
            off += dtype_nbytes;
            pool.runtime_payload.assign(
                blob.begin() + (ptrdiff_t)off,
                blob.begin() + (ptrdiff_t)(off + (size_t)runtime_nbytes));
            off += (size_t)runtime_nbytes;
        }
        size_t payload_end = off + (size_t)payload_nbytes;
        pool.payload.assign(
            blob.begin() + (ptrdiff_t)off, blob.begin() + (ptrdiff_t)payload_end);
        off = payload_end;
        if (!version1 && mfq::fp8sq::is_dtype(pool.dtype)) {
            if (!pool.runtime_payload.empty()) {
                throw std::runtime_error(
                    "unexpected MFE FP8-SQ runtime metadata");
            }
            const auto layout = mfq::fp8sq::parse(
                pool.dtype, pool.payload.data(), pool.payload.size());
            const int expected_rows =
                static_cast<int>(expert_count) * result.out_per_expert;
            if (layout.outputs != expected_rows ||
                    layout.width != result.neuron_len) {
                throw std::runtime_error(
                    "MFE FP8-SQ pool weight shape mismatch");
            }
        } else if (!version1 && pool.dtype == "MXFP4-SQ") {
            if (!pool.runtime_payload.empty()) {
                throw std::runtime_error(
                    "unexpected MFE MXFP4-SQ runtime metadata");
            }
            const auto layout = mfq::sq::parse(
                pool.payload.data(), pool.payload.size());
            const int expected_rows =
                static_cast<int>(expert_count) * result.out_per_expert;
            if (layout.outputs != expected_rows ||
                    layout.width != result.neuron_len) {
                throw std::runtime_error(
                    "MFE MXFP4-SQ pool weight shape mismatch");
            }
        } else if (!version1 && pool.dtype == "MXFP4") {
            pool.mxfp4 = unpack_mxfp4(pool.payload);
            const int expected_rows =
                static_cast<int>(expert_count) * result.out_per_expert;
            if (pool.mxfp4.out != expected_rows ||
                    pool.mxfp4.neuron_len != result.neuron_len) {
                throw std::runtime_error(
                    "MFE MXFP4 pool weight shape mismatch");
            }
        } else if (!version1 && pool.dtype == "NINT8-0") {
            pool.q8_zero = unpack_nint8_zero(pool.payload);
            const int expected_rows =
                static_cast<int>(expert_count) * result.out_per_expert;
            if (pool.q8_zero.axis != 0 ||
                pool.q8_zero.out != expected_rows ||
                pool.q8_zero.neuron_len != result.neuron_len ||
                pool.q8_zero.shape.size() != 2 ||
                pool.q8_zero.shape[0] != expected_rows ||
                pool.q8_zero.shape[1] != result.neuron_len) {
                throw std::runtime_error(
                    "MFE NINT8-0 pool weight shape mismatch");
            }
        } else if (version1 || pool.dtype == "NINT") {
            pool.weight = unpack_nint(pool.payload);
            if (version1) pool.dtype = "NINT";
            int expected_rows = (int)expert_count * result.out_per_expert;
            if (pool.weight.axis != 0 || pool.weight.out != expected_rows ||
                pool.weight.neuron_len != result.neuron_len ||
                pool.weight.shape.size() != 2 ||
                pool.weight.shape[0] != expected_rows ||
                pool.weight.shape[1] != result.neuron_len) {
                throw std::runtime_error("MFE pool weight shape mismatch");
            }
        }
        result.pools.push_back(std::move(pool));
    }
    if (off != blob.size() ||
        (!allow_partial &&
         std::find(owners.begin(), owners.end(), -1) != owners.end())) {
        throw std::runtime_error("MFE expert coverage or tail mismatch");
    }
    return result;
}

MfeCpu unpack_mfe(const std::vector<uint8_t> & blob) {
    return unpack_mfe_impl(blob, false);
}

static MfeCpu unpack_mfe_delta(
        const std::vector<uint8_t> & blob) {
    return unpack_mfe_impl(blob, true);
}


static void initialize_mfe_nint_runtime(MfeWeight & result) {
    if (result.pools.empty()) {
        throw std::runtime_error("MFE NINT runtime requires at least one pool");
    }
    result.activation_geometries.clear();
    result.activation_workspace_domain = 1;
    std::unordered_set<int64_t> quantized_shapes;
    const auto target = result.pools.front().weight.q_packed.device();
    for (const auto & pool : result.pools) {
        MFQ_RUNTIME_CHECK(
            pool.weight.q_packed.is_cuda() &&
            pool.weight.q_packed.device() == target,
            "MFE NINT pools must share one CUDA device");
        const int64_t activation_key =
            (pool.weight.gs << 32) ^ pool.weight.ng;
        if (quantized_shapes.insert(activation_key).second) {
            result.activation_geometries.push_back({
                static_cast<int>(pool.weight.ng),
                static_cast<int>(pool.weight.gs),
                0,
                0,
            });
        }
    }
}

MfeWeight to_gpu_mfe(const MfeCpu & cpu) {
    MfeWeight result;
    result.unified_nint_projection = true;
    result.n_experts = cpu.n_experts;
    result.out_per_expert = cpu.out_per_expert;
    result.neuron_len = cpu.neuron_len;
    result.pools.reserve(cpu.pools.size());
    for (const auto & source_pool : cpu.pools) {
        if (source_pool.dtype != "NINT") {
            throw std::runtime_error("non-NINT cohort reached the NINT-only loader");
        }
        MfePoolWeight pool;
        pool.local_experts = (int)source_pool.expert_ids.size();
        pool.weight = to_device_mfe_nint(
            source_pool.weight, pool.local_experts,
            cpu.out_per_expert, true);
        std::vector<int32_t> local((size_t)cpu.n_experts, -1);
        for (int index = 0; index < pool.local_experts; ++index) {
            const int expert = source_pool.expert_ids.at(static_cast<size_t>(index));
            local[static_cast<size_t>(expert)] = index;
        }
        pool.expert_local = mfq_tensor_backend::from_blob(
            local.data(), {(int64_t)local.size()}, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt32))
            .clone().to(pool.weight.q_packed.device()).contiguous();
        result.pools.push_back(std::move(pool));
    }
    initialize_mfe_nint_runtime(result);
    return result;
}


static int64_t tensor_storage_bytes(const mfq_tensor_backend::Tensor & value) {
    return value.defined()
        ? value.numel() * (int64_t)value.element_size()
        : 0;
}

int64_t mixed_moe_storage_bytes(const MixedMoeRuntime & runtime) {
    int64_t bytes = 0;
    if(runtime.nint_dispatch) {
        const auto& d=*runtime.nint_dispatch;
        for(const auto& t:{d.map_ptrs,d.pool_params,d.expert_pool,d.expert_local})bytes+=tensor_storage_bytes(t);
    }
    for(const auto& item:runtime.nint_input_plans) {
        bytes+=tensor_storage_bytes(item.second.weight_ptrs)+tensor_storage_bytes(item.second.quantize_descriptors);
    }
    if (runtime.nvq_dispatch) {
        bytes += tensor_storage_bytes(runtime.nvq_dispatch->weight_ptrs);
        bytes += tensor_storage_bytes(runtime.nvq_dispatch->weight_sizes);
        bytes += tensor_storage_bytes(runtime.nvq_dispatch->pool_params);
        bytes += tensor_storage_bytes(runtime.nvq_dispatch->expert_pool);
        bytes += tensor_storage_bytes(runtime.nvq_dispatch->expert_local);
    }
    for (const auto & pool : runtime.pools) {
        bytes += tensor_storage_bytes(pool.expert_local);
        if (pool.family == MixedMoeFamily::Nint) {
            bytes += tensor_storage_bytes(pool.nint.q_packed);
            bytes += tensor_storage_bytes(pool.nint.row_q_bits);
            bytes += tensor_storage_bytes(pool.nint.row_q_bit_offsets);
            bytes += tensor_storage_bytes(pool.nint.sub_scale);
            bytes += tensor_storage_bytes(pool.nint.sub_min);
            bytes += tensor_storage_bytes(pool.nint.neuron_scale);
            bytes += tensor_storage_bytes(pool.nint.neuron_min);
        } else if (pool.family == MixedMoeFamily::Nint8Zero) {
            bytes += tensor_storage_bytes(pool.q8_zero.q_packed);
            bytes += tensor_storage_bytes(pool.q8_zero.q8_zero_scale);
        } else if (pool.family == MixedMoeFamily::Mxfp4) {
            bytes += tensor_storage_bytes(pool.mxfp4.values);
            bytes += tensor_storage_bytes(pool.mxfp4.scales);
        } else if (pool.family == MixedMoeFamily::Mxfp4Sq) {
            bytes += tensor_storage_bytes(pool.mxfp4_sq.blob);
            bytes += tensor_storage_bytes(pool.mxfp4_sq.row_q);
            bytes += tensor_storage_bytes(
                pool.mxfp4_sq.row_symbol_byte_offsets);
            bytes += tensor_storage_bytes(pool.mxfp4_sq.row_auxiliary);
        } else if (pool.family == MixedMoeFamily::Fp8Sq) {
            bytes += tensor_storage_bytes(pool.fp8_sq.blob);
            bytes += tensor_storage_bytes(pool.fp8_sq.row_q);
            bytes += tensor_storage_bytes(
                pool.fp8_sq.row_symbol_byte_offsets);
        } else if (pool.family == MixedMoeFamily::Nvq) {
            bytes += tensor_storage_bytes(pool.nvq.indices_packed);
            bytes += tensor_storage_bytes(pool.nvq.aux_packed);
            bytes += tensor_storage_bytes(pool.nvq.sub_scale_packed);
            bytes += tensor_storage_bytes(pool.nvq.neuron_scale);
            bytes += tensor_storage_bytes(pool.nvq.codebook);
            bytes += tensor_storage_bytes(pool.nvq.integer_codebook);
            if(pool.nvq.decode_records.defined() && pool.nvq.decode_records.data_ptr()!=pool.nvq.indices_packed.data_ptr())
                bytes += tensor_storage_bytes(pool.nvq.decode_records);
        } else {
            bytes += tensor_storage_bytes(pool.nepq.indices_packed);
            bytes += tensor_storage_bytes(pool.nepq.aux_packed);
            bytes += tensor_storage_bytes(pool.nepq.state_packed);
            bytes += tensor_storage_bytes(pool.nepq.neuron_scale);
            bytes += tensor_storage_bytes(pool.nepq.table_pool);
            bytes += tensor_storage_bytes(pool.nepq.grouped_table_pool);
            bytes += tensor_storage_bytes(pool.nepq.bank_ids);
            bytes += tensor_storage_bytes(pool.nepq.rotation_signs);
            bytes += tensor_storage_bytes(pool.nepq.residual_codebook);
            bytes += tensor_storage_bytes(pool.nepq.residual_first);
            bytes += tensor_storage_bytes(pool.nepq.residual_second);
        }
    }
    return bytes;
}

std::shared_ptr<MixedMoeRuntime> make_mixed_moe_runtime(
        const MfeCpu & cpu,
        bool cuda,
        const CudaExecutionConfig& config) {
    auto runtime = std::make_shared<MixedMoeRuntime>();
    runtime->n_experts = cpu.n_experts;
    runtime->out_per_expert = cpu.out_per_expert;
    runtime->neuron_len = cpu.neuron_len;
    runtime->pools.reserve(cpu.pools.size());
    for (const auto & source : cpu.pools) {
        MixedMoePool pool;
        pool.local_experts = (int)source.expert_ids.size();
        std::vector<int32_t> local((size_t)cpu.n_experts, -1);
        for (int index = 0; index < pool.local_experts; ++index) {
            local[(size_t)source.expert_ids[(size_t)index]] = index;
        }
        pool.expert_local = mfq_tensor_backend::from_blob(
            local.data(), {(int64_t)local.size()},
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt32))
            .clone();
        if (cuda) {
            const auto target = mfq_tensor_backend::Device(
                mfq_tensor_backend::kCUDA,
                mfq_current_cuda_device());
            pool.expert_local =
                pool.expert_local.to(target).contiguous();
        }
        const int expected_rows = pool.local_experts * cpu.out_per_expert;
        if (source.dtype == "NINT8-0") {
            pool.family = MixedMoeFamily::Nint8Zero;
            pool.q8_zero = cuda
                ? to_gpu_nint8_zero(source.q8_zero)
                : to_cpu_nint8_zero(source.q8_zero);
            if (pool.q8_zero.out != expected_rows ||
                pool.q8_zero.neuron_len != cpu.neuron_len) {
                throw std::runtime_error(
                    "mixed NINT8-0 cohort shape mismatch");
            }
        } else if (source.dtype == "MXFP4") {
            pool.family = MixedMoeFamily::Mxfp4;
            pool.mxfp4 = to_device_mxfp4(source.mxfp4, cuda);
            if (pool.mxfp4.out != expected_rows ||
                    pool.mxfp4.neuron_len != cpu.neuron_len) {
                throw std::runtime_error(
                    "mixed MXFP4 cohort shape mismatch");
            }
        } else if (source.dtype == "MXFP4-SQ") {
            pool.family = MixedMoeFamily::Mxfp4Sq;
            pool.mxfp4_sq = to_device_mxfp4_sq(
                source.payload, cuda);
            if (pool.mxfp4_sq.out != expected_rows ||
                    pool.mxfp4_sq.neuron_len != cpu.neuron_len) {
                throw std::runtime_error(
                    "mixed MXFP4-SQ cohort shape mismatch");
            }
        } else if (mfq::fp8sq::is_dtype(source.dtype)) {
            pool.family = MixedMoeFamily::Fp8Sq;
            pool.fp8_sq = to_device_fp8_sq(
                source.dtype, source.payload, cuda);
            if (pool.fp8_sq.out != expected_rows ||
                    pool.fp8_sq.neuron_len != cpu.neuron_len) {
                throw std::runtime_error(
                    "mixed FP8-SQ cohort shape mismatch");
            }
        } else if (source.dtype == "NINT") {
            pool.family = MixedMoeFamily::Nint;
            pool.nint = to_device_mfe_nint(
                source.weight, pool.local_experts,
                cpu.out_per_expert, cuda);
            if (pool.nint.out != expected_rows ||
                pool.nint.neuron_len != cpu.neuron_len) {
                throw std::runtime_error("mixed NINT cohort shape mismatch");
            }
        } else if (source.dtype == "NEPQ") {
            pool.family = MixedMoeFamily::Nepq;
            auto parsed = unpack_nepq(
                source.payload, source.dtype, source.runtime_payload);
            if (parsed.n_experts != pool.local_experts ||
                parsed.out_per_expert != cpu.out_per_expert ||
                parsed.neuron_len != cpu.neuron_len) {
                throw std::runtime_error("mixed NEPQ cohort shape mismatch");
            }
            pool.nepq = cuda
                ? to_gpu_nepq(parsed)
                : to_cpu_nepq(parsed);
        } else {
            pool.family = MixedMoeFamily::Nvq;
            auto parsed = unpack_nvq(source.payload, source.dtype);
            if (parsed.out != expected_rows || parsed.neuron_len != cpu.neuron_len) {
                throw std::runtime_error("mixed NVQ/NPQ cohort shape mismatch");
            }
            pool.nvq = cuda
                ? to_gpu_nvq(parsed, config)
                : to_cpu_nvq(parsed);
        }
        runtime->pools.push_back(std::move(pool));
    }
    if (cuda) {
        initialize_mixed_nvq_dispatch(*runtime, config);
    }
    return runtime;
}

static MfeWeight wrap_mixed_moe_runtime(
        const std::shared_ptr<MixedMoeRuntime> & runtime) {
    MfeWeight result;
    result.unified_nint_projection = runtime->nint_only();
    result.n_experts = runtime->n_experts;
    result.out_per_expert = runtime->out_per_expert;
    result.neuron_len = runtime->neuron_len;
    result.partial_experts = runtime->partial_experts;
    result.mixed_weight_bytes = mixed_moe_storage_bytes(*runtime);
    result.activation_geometries = runtime->activation_geometry();
    result.activation_workspace_domain = 2;
    result.mixed_forward = [runtime](
            CudaExecutionContext& execution,
            mfq_tensor_backend::Tensor x, const MoeRoutePlan & route) {
        return runtime->forward(
            execution.config, execution.kl_mmq,
            execution.force_moe_prefill_mma_off,
            execution.force_moe_pool_path, x, route);
    };
    result.mixed_prequantized_forward = [runtime](
            CudaExecutionContext& execution,
            mfq_tensor_backend::Tensor x, const MoeRoutePlan & route) {
        return runtime->forward(
            execution.config, execution.kl_mmq,
            execution.force_moe_prefill_mma_off,
            execution.force_moe_pool_path, x, route, true);
    };
    result.mixed_glu_output_forward = [runtime](
            CudaExecutionContext& execution,
            mfq_tensor_backend::Tensor x,
            const MoeRoutePlan & route,
            bool gelu) {
        return runtime->forward_glu_output(
            execution.config, execution.kl_mmq,
            execution.force_moe_prefill_mma_off,
            execution.force_moe_pool_path, x, route, gelu);
    };
    if (runtime->supports_clamped_swiglu()) {
        result.mixed_clamped_swiglu_forward = [runtime](
                CudaExecutionContext& execution,
                mfq_tensor_backend::Tensor gate_up,
                const MoeRoutePlan & route,
                double limit) {
            return runtime->forward_clamped_swiglu(
                execution.config, execution.kl_mmq,
                execution.force_moe_prefill_mma_off,
                execution.force_moe_pool_path,
                gate_up, route, limit);
        };
    }
    return result;
}

MfeWeight to_gpu_mixed_moe(
        const MfeCpu & cpu,
        const CudaExecutionConfig& config) {
    return wrap_mixed_moe_runtime(
        make_mixed_moe_runtime(cpu, true, config));
}

MfeWeight to_cuda_device_moe_expert_slice(
        const MfeCpu & cpu,
        int64_t expert_begin,
        int64_t expert_end,
        int device,
        const CudaExecutionConfig& config) {
    if (expert_begin < 0 || expert_begin >= expert_end ||
            expert_end > cpu.n_experts) {
        throw std::runtime_error(
            "invalid expert-parallel MoE shard");
    }
    const bool all_nint = std::all_of(
        cpu.pools.begin(), cpu.pools.end(),
        [](const MfeCpuPool & pool) {
            return pool.dtype == "NINT";
        });
    MfqCudaGuard guard(device);
    if (all_nint) {
        MfeCpu sliced = cpu;
        sliced.pools.clear();
        for (const auto & source : cpu.pools) {
            MfeCpuPool destination = source;
            destination.expert_ids.clear();
            std::vector<int64_t> rows;
            for (size_t local = 0;
                 local < source.expert_ids.size(); ++local) {
                const int expert = source.expert_ids[local];
                if (expert < expert_begin || expert >= expert_end) continue;
                destination.expert_ids.push_back(expert);
                for (int row = 0; row < cpu.out_per_expert; ++row) {
                    rows.push_back(
                        static_cast<int64_t>(local) * cpu.out_per_expert + row);
                }
            }
            if (rows.empty()) continue;
            destination.weight = select_nint_cpu_rows(
                source.weight, rows);
            sliced.pools.push_back(std::move(destination));
        }
        auto result = to_gpu_mfe(sliced);
        result.partial_experts = true;
        return result;
    }

    auto runtime = std::make_shared<MixedMoeRuntime>();
    runtime->n_experts = cpu.n_experts;
    runtime->out_per_expert = cpu.out_per_expert;
    runtime->neuron_len = cpu.neuron_len;
    runtime->partial_experts = true;
    for (const auto & source : cpu.pools) {
        std::vector<int> expert_ids;
        std::vector<int64_t> rows;
        for (size_t local = 0;
             local < source.expert_ids.size(); ++local) {
            const int expert = source.expert_ids[local];
            if (expert < expert_begin || expert >= expert_end) continue;
            expert_ids.push_back(expert);
            for (int row = 0; row < cpu.out_per_expert; ++row) {
                rows.push_back(
                    static_cast<int64_t>(local) * cpu.out_per_expert + row);
            }
        }
        if (rows.empty()) continue;

        MixedMoePool pool;
        pool.local_experts = static_cast<int>(expert_ids.size());
        std::vector<int32_t> local_map(
            static_cast<size_t>(cpu.n_experts), -1);
        for (size_t local = 0; local < expert_ids.size(); ++local) {
            local_map[static_cast<size_t>(expert_ids[local])] =
                static_cast<int32_t>(local);
        }
        pool.expert_local = mfq_tensor_backend::from_blob(
            local_map.data(),
            {static_cast<int64_t>(local_map.size())},
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt32))
            .clone().to(mfq_tensor_backend::Device(
                mfq_tensor_backend::kCUDA,
                mfq_current_cuda_device())).contiguous();
        if (source.dtype == "NINT8-0") {
            pool.family = MixedMoeFamily::Nint8Zero;
            pool.q8_zero = to_gpu_nint8_zero(
                select_nint8_zero_cpu_rows(source.q8_zero, rows));
        } else if (source.dtype == "NINT") {
            pool.family = MixedMoeFamily::Nint;
            const auto selected = select_nint_cpu_rows(source.weight, rows);
            pool.nint = to_device_mfe_nint(
                selected, pool.local_experts,
                cpu.out_per_expert, true);
        } else if (source.dtype == "MXFP4") {
            pool.family = MixedMoeFamily::Mxfp4;
            pool.mxfp4 = to_device_mxfp4(
                select_mxfp4_cpu_rows(source.mxfp4, rows), true);
        } else if (source.dtype == "MXFP4-SQ") {
            pool.family = MixedMoeFamily::Mxfp4Sq;
            pool.mxfp4_sq = to_device_mxfp4_sq(
                mfq::sq::select_rows(source.payload, rows), true);
        } else if (mfq::fp8sq::is_dtype(source.dtype)) {
            if (rows.size() != source.expert_ids.size() *
                    static_cast<size_t>(cpu.out_per_expert)) {
                throw std::runtime_error(
                    "FP8-SQ MFE pools must remain matrix-local across devices");
            }
            pool.family = MixedMoeFamily::Fp8Sq;
            pool.fp8_sq = to_device_fp8_sq(
                source.dtype, source.payload, true);
        } else if (source.dtype == "NEPQ") {
            pool.family = MixedMoeFamily::Nepq;
            auto parsed = unpack_nepq(
                source.payload, source.dtype, source.runtime_payload);
            pool.nepq = to_gpu_nepq(select_nepq_cpu_rows(
                parsed, rows, cpu.out_per_expert,
                pool.local_experts));
        } else {
            pool.family = MixedMoeFamily::Nvq;
            auto parsed = unpack_nvq(source.payload, source.dtype);
            pool.nvq = to_gpu_nvq(
                select_nvq_cpu_rows(parsed, rows), config);
        }
        runtime->pools.push_back(std::move(pool));
    }
    if (runtime->pools.empty()) {
        throw std::runtime_error(
            "expert-parallel MoE shard has no owned experts");
    }
    initialize_mixed_nvq_dispatch(*runtime, config);
    return wrap_mixed_moe_runtime(runtime);
}

mfq_tensor_backend::Tensor copy_cpu_weight_to_cuda(
        const mfq_tensor_backend::Tensor & source) {
    return source.defined()
        ? source.to(mfq_tensor_backend::Device(
            mfq_tensor_backend::kCUDA,
            mfq_current_cuda_device())).contiguous()
        : mfq_tensor_backend::Tensor();
}

static NintWeight copy_cpu_nint_to_cuda(const NintWeight & source) {
    NintWeight result = source;
    result.workspaces.clear();
    result.q_packed = copy_cpu_weight_to_cuda(source.q_packed);
    result.row_q_bits = copy_cpu_weight_to_cuda(source.row_q_bits);
    result.row_q_bit_offsets =
        copy_cpu_weight_to_cuda(source.row_q_bit_offsets);
    result.q8_zero_scale =
        copy_cpu_weight_to_cuda(source.q8_zero_scale);
    result.sub_scale = copy_cpu_weight_to_cuda(source.sub_scale);
    result.sub_min = copy_cpu_weight_to_cuda(source.sub_min);
    result.neuron_scale =
        copy_cpu_weight_to_cuda(source.neuron_scale);
    result.neuron_min = copy_cpu_weight_to_cuda(source.neuron_min);
    return result;
}

static NvqWeight copy_cpu_nvq_to_cuda(const NvqWeight & source) {
    NvqWeight result = source;
    result.workspaces.clear();
    result.indices_packed =
        copy_cpu_weight_to_cuda(source.indices_packed);
    result.aux_packed = copy_cpu_weight_to_cuda(source.aux_packed);
    result.sub_scale_packed =
        copy_cpu_weight_to_cuda(source.sub_scale_packed);
    result.neuron_scale =
        copy_cpu_weight_to_cuda(source.neuron_scale);
    result.codebook = copy_cpu_weight_to_cuda(source.codebook);
    result.integer_codebook = copy_cpu_weight_to_cuda(source.integer_codebook);
    result.decode_records = source.decode_records.defined() && source.decode_records.data_ptr()==source.indices_packed.data_ptr()
        ? result.indices_packed : copy_cpu_weight_to_cuda(source.decode_records);
    return result;
}

static Mxfp4Weight copy_cpu_mxfp4_to_cuda(const Mxfp4Weight & source) {
    Mxfp4Weight result = source;
    result.values = copy_cpu_weight_to_cuda(source.values);
    result.scales = copy_cpu_weight_to_cuda(source.scales);
    return result;
}

static Mxfp4SqWeight copy_cpu_mxfp4_sq_to_cuda(
        const Mxfp4SqWeight & source) {
    Mxfp4SqWeight result = source;
    result.blob = copy_cpu_weight_to_cuda(source.blob);
    result.row_q = copy_cpu_weight_to_cuda(source.row_q);
    result.row_symbol_byte_offsets =
        copy_cpu_weight_to_cuda(source.row_symbol_byte_offsets);
    result.row_auxiliary =
        copy_cpu_weight_to_cuda(source.row_auxiliary);
    return result;
}

static Fp8SqWeight copy_cpu_fp8_sq_to_cuda(
        const Fp8SqWeight & source) {
    Fp8SqWeight result = source;
    result.blob = copy_cpu_weight_to_cuda(source.blob);
    result.row_q = copy_cpu_weight_to_cuda(source.row_q);
    result.row_symbol_byte_offsets =
        copy_cpu_weight_to_cuda(source.row_symbol_byte_offsets);
    return result;
}

static NepqWeight copy_cpu_nepq_to_cuda(const NepqWeight & source) {
    NepqWeight result = source;
    result.indices_packed =
        copy_cpu_weight_to_cuda(source.indices_packed);
    result.aux_packed = copy_cpu_weight_to_cuda(source.aux_packed);
    result.state_packed =
        copy_cpu_weight_to_cuda(source.state_packed);
    result.neuron_scale =
        copy_cpu_weight_to_cuda(source.neuron_scale);
    result.table_pool = copy_cpu_weight_to_cuda(source.table_pool);
    result.grouped_table_pool =
        copy_cpu_weight_to_cuda(source.grouped_table_pool);
    result.bank_ids = copy_cpu_weight_to_cuda(source.bank_ids);
    result.rotation_signs =
        copy_cpu_weight_to_cuda(source.rotation_signs);
    if (source.residual) {
        result.residual_codebook =
            copy_cpu_weight_to_cuda(source.residual_codebook);
        result.residual_first =
            copy_cpu_weight_to_cuda(source.residual_first);
        result.residual_second =
            copy_cpu_weight_to_cuda(source.residual_second);
    }
    return result;
}

MfeWeight stage_cpu_mixed_moe(
        const std::shared_ptr<MixedMoeRuntime> & cpu,
        const CudaExecutionConfig& config) {
    if (!cpu) throw std::runtime_error("missing CPU-offloaded MoE state");
    auto runtime = std::make_shared<MixedMoeRuntime>();
    runtime->n_experts = cpu->n_experts;
    runtime->out_per_expert = cpu->out_per_expert;
    runtime->neuron_len = cpu->neuron_len;
    runtime->pools.reserve(cpu->pools.size());
    for (const auto & source : cpu->pools) {
        MixedMoePool pool;
        pool.family = source.family;
        pool.local_experts = source.local_experts;
        pool.expert_local =
            copy_cpu_weight_to_cuda(source.expert_local);
        if (pool.family == MixedMoeFamily::Nint) {
            pool.nint = copy_cpu_nint_to_cuda(source.nint);
        } else if (pool.family == MixedMoeFamily::Nint8Zero) {
            pool.q8_zero = copy_cpu_nint_to_cuda(source.q8_zero);
        } else if (pool.family == MixedMoeFamily::Mxfp4) {
            pool.mxfp4 = copy_cpu_mxfp4_to_cuda(source.mxfp4);
        } else if (pool.family == MixedMoeFamily::Mxfp4Sq) {
            pool.mxfp4_sq = copy_cpu_mxfp4_sq_to_cuda(
                source.mxfp4_sq);
        } else if (pool.family == MixedMoeFamily::Fp8Sq) {
            pool.fp8_sq = copy_cpu_fp8_sq_to_cuda(source.fp8_sq);
        } else if (pool.family == MixedMoeFamily::Nvq) {
            pool.nvq = copy_cpu_nvq_to_cuda(source.nvq);
        } else {
            pool.nepq = copy_cpu_nepq_to_cuda(source.nepq);
        }
        runtime->pools.push_back(std::move(pool));
    }
    initialize_mixed_nvq_dispatch(*runtime, config);
    return wrap_mixed_moe_runtime(runtime);
}

MfeWeight cpu_mixed_moe_metadata(
        const std::shared_ptr<MixedMoeRuntime> & runtime) {
    MfeWeight result;
    result.unified_nint_projection = runtime->nint_only();
    result.n_experts = runtime->n_experts;
    result.out_per_expert = runtime->out_per_expert;
    result.neuron_len = runtime->neuron_len;
    result.mixed_weight_bytes = mixed_moe_storage_bytes(*runtime);
    result.activation_geometries = runtime->activation_geometry();
    result.activation_workspace_domain = 2;
    return result;
}

mfq_tensor_backend::Tensor mfe_dense_reference(
        const MfeCpu& cpu,
        mfq_tensor_backend::Tensor input,
        const std::vector<std::int32_t>& expert_ids,
        int tokens,
        int routes,
        bool routed_input) {
    auto reference = mfq_tensor_backend::empty(
        {tokens * routes, cpu.out_per_expert},
        input.options().dtype(mfq_tensor_backend::kFloat16));
    for (const auto& pool : cpu.pools) {
        mfq_tensor_backend::Tensor dense_flat;
        int rotation_block = 0;
        mfq_tensor_backend::Tensor rotation_signs;
        if (pool.dtype == "NINT8-0") {
            auto packed = to_gpu_nint8_zero(pool.q8_zero);
            dense_flat = nint8_zero_dequant_cuda(
                packed.q_packed, packed.q8_zero_scale,
                packed.neuron_len);
        } else if (pool.dtype == "NINT") {
            auto packed = to_gpu_nint(pool.weight);
            dense_flat = nint_decode_cuda(
                packed.q_packed, packed.row_q_bits,
                packed.row_q_bit_offsets, packed.sub_scale,
                packed.sub_min, packed.neuron_scale,
                packed.neuron_min, packed.neuron_len, packed.gs);
        } else if (pool.dtype == "MXFP4") {
            dense_flat = dequant_mxfp4_cpu(pool.mxfp4)
                .to(mfq_tensor_backend::kCUDA).contiguous();
        } else if (pool.dtype == "NEPQ") {
            auto packed = to_gpu_nepq(unpack_nepq(
                pool.payload, pool.dtype, pool.runtime_payload));
            dense_flat = nepq_dequant_cuda(
                packed.indices_packed, packed.aux_packed,
                packed.state_packed, packed.neuron_scale,
                packed.table_pool, packed.bank_ids,
                packed.neuron_len, packed.state_bits,
                packed.format);
            if (packed.residual) {
                dense_flat = nepq_sparse_residual_dequant_cuda(
                    packed.residual_codebook,
                    packed.residual_first,
                    packed.residual_second,
                    packed.residual_position_bits,
                    packed.residual_block_vectors,
                    dense_flat.reshape({
                        packed.n_experts * packed.out_per_expert,
                        packed.neuron_len}));
            }
            rotation_block = packed.rotation_block;
            rotation_signs = packed.rotation_signs;
        } else {
            dense_flat = nvq_dequant(to_gpu_nvq(
                unpack_nvq(pool.payload, pool.dtype)));
        }
        auto dense = dense_flat.reshape({
            static_cast<int64_t>(pool.expert_ids.size()),
            cpu.out_per_expert, cpu.neuron_len});
        for (size_t local = 0; local < pool.expert_ids.size(); ++local) {
            const int expert = pool.expert_ids[local];
            std::vector<int64_t> pair_indices;
            std::vector<int64_t> token_indices;
            for (int pair = 0; pair < tokens * routes; ++pair) {
                if (expert_ids[static_cast<size_t>(pair)] == expert) {
                    pair_indices.push_back(pair);
                    token_indices.push_back(pair / routes);
                }
            }
            if (pair_indices.empty()) continue;
            auto pair_index = mfq_tensor_backend::from_blob(
                pair_indices.data(),
                {static_cast<int64_t>(pair_indices.size())},
                mfq_tensor_backend::TensorOptions()
                    .dtype(mfq_tensor_backend::kInt64))
                .clone().to(mfq_tensor_backend::kCUDA);
            auto token_index = mfq_tensor_backend::from_blob(
                token_indices.data(),
                {static_cast<int64_t>(token_indices.size())},
                mfq_tensor_backend::TensorOptions()
                    .dtype(mfq_tensor_backend::kInt64))
                .clone().to(mfq_tensor_backend::kCUDA);
            auto selected = routed_input
                ? input.reshape({tokens * routes, cpu.neuron_len})
                      .index_select(0, pair_index)
                : input.index_select(0, token_index);
            if (rotation_block != 0) {
                selected = nepq_hadamard_input_cuda(
                    selected.contiguous(), rotation_signs,
                    rotation_block);
            }
            auto expected = mfq_tensor_backend::matmul(
                selected,
                dense.index({static_cast<int64_t>(local)})
                    .transpose(0, 1));
            reference.index_copy_(0, pair_index, expected);
        }
    }
    return reference;
}

mfq_tensor_backend::Tensor materialize_mfe_dense(
        const MfeCpu& cpu) {
    auto dense = mfq_tensor_backend::empty(
        {cpu.n_experts, cpu.out_per_expert, cpu.neuron_len},
        mfq_tensor_backend::TensorOptions()
            .device(mfq_tensor_backend::kCUDA)
            .dtype(mfq_tensor_backend::kFloat16));
    for (const auto& pool : cpu.pools) {
        mfq_tensor_backend::Tensor local_flat;
        if (pool.dtype == "NINT8-0") {
            auto packed = to_gpu_nint8_zero(pool.q8_zero);
            local_flat = nint8_zero_dequant_cuda(
                packed.q_packed, packed.q8_zero_scale,
                packed.neuron_len);
        } else if (pool.dtype == "NINT") {
            auto packed = to_gpu_nint(pool.weight);
            local_flat = nint_decode_cuda(
                packed.q_packed, packed.row_q_bits,
                packed.row_q_bit_offsets, packed.sub_scale,
                packed.sub_min, packed.neuron_scale,
                packed.neuron_min, packed.neuron_len, packed.gs);
        } else if (pool.dtype == "MXFP4") {
            local_flat = dequant_mxfp4_cpu(pool.mxfp4)
                .to(mfq_tensor_backend::kCUDA).contiguous();
        } else {
            throw std::runtime_error(
                "dense MoE reference requires NINT or MXFP4 cohorts");
        }
        auto local = local_flat.reshape({
            static_cast<int64_t>(pool.expert_ids.size()),
            cpu.out_per_expert, cpu.neuron_len});
        auto expert_index = mfq_tensor_backend::from_blob(
            const_cast<int32_t*>(pool.expert_ids.data()),
            {static_cast<int64_t>(pool.expert_ids.size())},
            mfq_tensor_backend::TensorOptions()
                .dtype(mfq_tensor_backend::kInt32))
            .clone().to(mfq_tensor_backend::kCUDA)
            .to(mfq_tensor_backend::kInt64);
        dense.index_copy_(0, expert_index, local);
    }
    return dense;
}


