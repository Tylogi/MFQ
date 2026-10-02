#pragma once

#include "quant_linear.h"
#include "mfe_weight.h"
#include "mfq/model_source.h"

struct MfqDropFileCacheGuard {
    bool& setting;
    bool previous;

    MfqDropFileCacheGuard(bool enabled, bool& value)
        : setting(value), previous(value) {
        setting = enabled;
    }

    ~MfqDropFileCacheGuard() {
        setting = previous;
    }
};

bool has_tensor(const mfq::ModelSource& source, std::string_view name) noexcept;

bool has_tensor_prefix(const mfq::ModelSource& source, std::string_view prefix);

std::vector<std::uint8_t> read_asset(
    const mfq::ModelSource& source, std::string_view name);

std::string read_asset_text(
    const mfq::ModelSource& source, std::string_view name);

MfeWeight load_mfe_gpu(
    CudaExecutionContext& execution,
    const mfq::ModelSource& source,
    const std::string& name,
    bool cacheable = false,
    int layer_id = -1,
    const std::string& projection_role = {});

std::shared_ptr<MixedMoeRuntime> load_mfe_cpu_offloaded(
    const mfq::ModelSource& source,
    const std::string& name);

mfq_tensor_backend::Tensor load_dense_gpu(
    CudaExecutionContext& execution,
    const mfq::ModelSource& source,
    const std::string& name);

// Preserve the stored dtype; load_dense_gpu promotes half-precision floats to F32.
mfq_tensor_backend::Tensor load_dense_cpu(
    const CudaExecutionContext& execution,
    const mfq::ModelSource& source,
    const std::string& name);

mfq_tensor_backend::Tensor load_dense_native_gpu(
    CudaExecutionContext& execution,
    const mfq::ModelSource& source,
    const std::string& name);

QuantLinear load_quant_linear(
    CudaExecutionContext& execution,
    const mfq::ModelSource& source,
    const std::string& name,
    std::optional<TensorParallelAxis> axis_override = std::nullopt,
    const std::vector<mfq::TensorParallelSlice>* slices_override = nullptr);

QuantLinearGroup load_quant_group(
    CudaExecutionContext& execution,
    const mfq::ModelSource& source,
    const std::vector<std::string>& names,
    size_t required_compatible_prefix = 0,
    const std::vector<mfq::TensorParallelSlice>* slices_override = nullptr,
    bool preserve_projection_boundaries = false);

QuantLinearGroup load_paired_gate_up(
    CudaExecutionContext& execution,
    const mfq::ModelSource& source,
    const std::vector<std::string>& names,
    const QuantLinear& down,
    size_t required_compatible_prefix = 2,
    bool preserve_projection_boundaries = false);

mfq_tensor_backend::Tensor mfe_dense_reference(
    const mfq::ModelSource& source,
    const std::string& name,
    mfq_tensor_backend::Tensor input,
    const std::vector<std::int32_t>& expert_ids,
    int tokens,
    int routes,
    bool routed_input);

mfq_tensor_backend::Tensor materialize_mfe_dense(
    const mfq::ModelSource& source,
    const std::string& name);

namespace mfq::cuda::weight_loader {
using Tensor = mfq_tensor_backend::Tensor;
using Linear = std::function<Tensor(CudaExecutionContext&, const Tensor&)>;
using Routed = std::function<Tensor(CudaExecutionContext&, const Tensor&, const Tensor&)>;

Linear linear(CudaExecutionContext& execution, const mfq::ModelSource& file, const std::string& name);

Tensor dense(CudaExecutionContext& execution, const mfq::ModelSource& file, const std::string& name);

Routed routed(CudaExecutionContext& execution, const mfq::ModelSource& file, const std::string& name, int layer,
    int64_t experts, int64_t output, int64_t input, const std::string& role);

Routed routed_gate_up(CudaExecutionContext& execution, const mfq::ModelSource& file, const std::string& mlp_prefix,
    int layer, int64_t experts, int64_t width, int64_t input, const std::string& role);

void validate_load_options(
        const CudaExecutionContext& execution);
} // namespace mfq::cuda::weight_loader
