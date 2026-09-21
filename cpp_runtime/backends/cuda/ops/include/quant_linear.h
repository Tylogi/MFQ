#pragma once

#include "format.h"
#include "fp8_sq.h"
#include "mx.h"
#include "mxfp4_sq.h"
#include "moe.h"
#include "nint.h"
#include "vq.h"
#include "../../legacy/tpq/tpq.h"

#include "cuda_execution.h"
#include "mfq_cuda_ops.h"
#include "mfq/model_source.h"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

struct QuantLinear;
struct QuantLinearGroup;
struct DenseLinearGroup;

extern thread_local bool g_decode_graph_serial_branches;
extern thread_local bool g_decode_graph_tp_projection_major;
extern bool g_mfq_drop_file_cache;

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

bool decode_branch_parallel_enabled(std::int64_t rows);

mfq_tensor_backend::Tensor tensor_to_cuda_device(
    mfq_tensor_backend::Tensor value,
    int device,
    mfq_tensor_backend::Tensor reusable = {});
mfq_tensor_backend::Tensor run_quant_linear_shard(
    const struct QuantLinearShard& shard,
    mfq_tensor_backend::Tensor input,
    MfqOptional<mfq_tensor_backend::Tensor> gate = mfq_nullopt,
    int gate_mode = 0);

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
        return mfq_tensor_backend::silu(parts[0]) * parts[1];
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

bool has_tensor(const mfq::ModelSource& source, std::string_view name) noexcept;
bool has_tensor_prefix(const mfq::ModelSource& source, std::string_view prefix);
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
