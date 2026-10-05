#include "weight_loader.h"
#include "float_projection.h"
#include "selected_attention.h"
#include <limits>
#include <fstream>
#include <iterator>
#include "moe.h"
#include "moe_expert_cache.h"
#include "mfe_expert_store.h"
#include "moe_quant_range_source.h"

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
        mfq::ModelSource::TensorReader reader;
        try {
            reader=mfq.tensor_reader(name);
        } catch (const mfq::TensorReaderUnsupported&) {
            // Sources without retained readers keep the existing CPU loader.
        }
        if (reader) {
            try {
                auto store =
                    std::make_shared<mfq::cuda::MfeMxfp4ExpertStore>(
                        mfq::cuda::MfqRecordRange{
                            name,
                            record.dtype,
                            {},
                            0,
                            record.nbytes,
                            [reader](
                                    std::uint64_t offset,
                                    std::span<std::uint8_t> destination) {
                                reader(offset,
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
                try {
                    auto store=std::make_shared<mfq::MfeQuantExpertStore>(record.nbytes,
                        [reader](std::size_t offset,std::uint8_t* output,std::size_t bytes) {
                            reader(offset,reinterpret_cast<std::byte*>(output),bytes);
                        });
                    const int minimum=std::min(execution.moe_cache_registration_min_slots,store->num_experts());
                    return cache_quant_moe_weight(cache,name,std::make_shared<MoeQuantRangeSource>(std::move(store)),
                        minimum,layer_id,projection_role);
                } catch (const mfq::MfeQuantRangeUnsupported&) {
                    // Other canonical pool families retain the existing CPU path.
                }
            }
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

Linear residual_linear(CudaExecutionContext& execution, const mfq::ModelSource& file, const std::string& name) {
    auto weight = std::make_shared<QuantLinear>(load_quant_linear(execution, file, name));
    MFQ_RUNTIME_CHECK(weight->is_dense() || weight->is_nint(),
        "floating residual projection requires dense or NINT weights: ", name);
    return [weight](CudaExecutionContext& execution, const Tensor& input) {
        if (weight->is_dense())
            return mfq_selected_attention::promoted_matmul(input, weight->dense.transpose(-1, -2));
        auto output = execution.profiler.measure("nint.float_projection", [&] {
            return nint_float_projection_cuda(weight->nint, input);
        });
        // Match Metal's promotion of NINT's logical F16 weight dtype.
        return input.scalar_type() == tb::kFloat16 ? output.to(tb::kFloat16) : output;
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
            execution.n_gpu_layers >= 0) {
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
