#pragma once

#include "nint_linear_group.h"

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
    mfq_tensor_backend::Tensor dense;
};

struct QuantLinear {
    QuantLinearKind kind = QuantLinearKind::Nint;
    NintWeight nint;
    NvqWeight nvq;
    Mxfp4Linear mxfp4;
    Mxfp4SqLinear mxfp4_sq;
    Fp8SqLinear fp8_sq;
    Mxfp8Linear mxfp8;
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
    bool is_dense() const { return kind == QuantLinearKind::Dense; }

    mfq_tensor_backend::Tensor forward_tensor_parallel_flat(
            CudaExecutionContext& execution,
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
                execution.config, launch_position, tensor_parallel_shards.size());
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
            local_x = tensor_to_cuda_device(
                execution, local_x, shard.device);
            if (gate.has_value()) {
                local_gate =
                    tensor_to_cuda_device(
                        execution, local_gate, shard.device);
            }
            if (is_mxfp8() &&
                    tensor_parallel_axis == TensorParallelAxis::Input) {
                MFQ_RUNTIME_CHECK(
                    !gate.has_value(),
                    "MXFP8 input-axis tensor parallelism does not support gating");
                local_outputs[index] =
                    mxfp8_matmul_f32(
                        execution.profiler, shard.mxfp8, local_x);
            } else {
                local_outputs[index] =
                    run_quant_linear_shard(
                        execution, shard, local_x,
                        gate.has_value()
                            ? MfqOptional<mfq_tensor_backend::Tensor>(
                                local_gate)
                            : mfq_nullopt,
                        gate_mode);
            }
        }

        const int primary = model_parallel_primary_device(execution);
        MfqCudaGuard primary_guard(primary);
        if (tensor_parallel_axis == TensorParallelAxis::Output) {
            std::vector<mfq_tensor_backend::Tensor> gathered;
            gathered.reserve(local_outputs.size());
            for (auto & output : local_outputs) {
                gathered.push_back(
                    tensor_to_cuda_device(execution, output, primary));
            }
            return mfq_tensor_backend::cat(gathered, -1).contiguous();
        }

        auto reduced = reduce_model_parallel_outputs(
            execution, std::move(local_outputs));
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

    mfq_tensor_backend::Tensor forward(
            CudaExecutionContext& execution,
            mfq_tensor_backend::Tensor x) const {
        if (tensor_parallel() || is_nint() || is_nvq()) {
            auto shape = x.sizes().vec();
            auto flat = x.reshape({-1, x.size(-1)});
            auto y = tensor_parallel()
                ? forward_tensor_parallel_flat(execution, flat, mfq_nullopt, 0)
                : is_nint() ? run_nint_linear(execution, nint, flat)
                            : run_nvq_linear(execution, nvq, flat);
            shape.back() = y.size(-1);
            return y.reshape(shape);
        }
        if (is_mxfp4()) return mxfp4.forward(x);
        if (is_mxfp4_sq()) return mxfp4_sq.forward(x);
        if (is_fp8_sq()) return fp8_sq.forward(x);
        if (is_dense()) return forward_dense(x);
        return mxfp8.forward(execution.profiler, x);
    }
    mfq_tensor_backend::Tensor forward_bf16_output(
            CudaExecutionContext& execution,
            mfq_tensor_backend::Tensor x) const {
        return forward(execution, x)
            .to(mfq_tensor_backend::kBFloat16).contiguous();
    }
    mfq_tensor_backend::Tensor forward_mxfp8_groupwise(
            CudaExecutionContext& execution,
            mfq_tensor_backend::Tensor grouped,
            int64_t groups) const {
        MFQ_RUNTIME_CHECK(
            is_mxfp8(),
            "groupwise MXFP8 projection requires an MXFP8 tensor");
        if (!tensor_parallel()) {
            return mxfp8_groupwise_matmul(
                execution.profiler, mxfp8.weight, grouped, groups);
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
                execution, local, shard.device);
            partials.push_back(mxfp8_groupwise_matmul_f32(
                execution.profiler, shard.mxfp8, local, groups));
        }
        return reduce_model_parallel_outputs(execution, std::move(partials))
            .to(grouped.scalar_type()).contiguous();
    }
    mfq_tensor_backend::Tensor forward_input_mul(
            CudaExecutionContext& execution,
            mfq_tensor_backend::Tensor x,
            mfq_tensor_backend::Tensor gate,
            int mode) const {
        if (tensor_parallel() || is_nint() || is_nvq()) {
            auto shape = x.sizes().vec();
            auto flat = x.reshape({-1, x.size(-1)});
            auto flat_gate = gate.reshape({-1, gate.size(-1)});
            auto y = tensor_parallel()
                ? forward_tensor_parallel_flat(execution, flat, flat_gate, mode)
                : is_nint() ? run_nint_linear(execution, nint, flat, flat_gate, mode)
                            : run_nvq_linear(execution, nvq, flat, flat_gate, mode);
            shape.back() = y.size(-1);
            return y.reshape(shape);
        }
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
        return forward(execution, gated);
    }
    mfq_tensor_backend::Tensor forward_input_mul_f32_kld(
            CudaExecutionContext& execution,
            mfq_tensor_backend::Tensor x,
            mfq_tensor_backend::Tensor gate,
            int mode) const {
        MFQ_RUNTIME_CHECK(
            !tensor_parallel() && is_nint(),
            "FP32-output KLD down projection requires a local NINT tensor");
        MFQ_RUNTIME_CHECK(
            execution.kl_mmq_mode == KlMmqMode::Fp16,
            "FP32-output NINT MMQ is restricted to the FP16 KLD path");
        auto shape = x.sizes().vec();
        auto y = nint_matmul_input_mul_f32(
            execution.profiler, nint, x.reshape({-1, x.size(-1)}),
            gate.reshape({-1, gate.size(-1)}), mode);
        if (nint.q8_zero) ++execution.kl_mmq_dense_calls;
        shape.back() = y.size(-1);
        return y.reshape(shape);
    }
    int64_t out() const {
        if (tensor_parallel()) return logical_out;
        if (is_nint()) return nint.out;
        if (is_nvq()) return nvq.out;
        if (is_mxfp4()) return mxfp4.weight.out;
        if (is_mxfp4_sq()) return mxfp4_sq.weight.out;
        if (is_fp8_sq()) return fp8_sq.weight.out;
        if (is_dense()) return dense.size(0);
        return mxfp8.weight.out;
    }
    int64_t neuron_len() const {
        if (tensor_parallel()) return logical_neuron_len;
        if (is_nint()) return nint.neuron_len;
        if (is_nvq()) return nvq.neuron_len;
        if (is_mxfp4()) return mxfp4.weight.neuron_len;
        if (is_mxfp4_sq()) return mxfp4_sq.weight.neuron_len;
        if (is_fp8_sq()) return fp8_sq.weight.neuron_len;
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
    CudaExecutionContext& execution,
    mfq_tensor_backend::Tensor input,
    const QuantLinearProjectionRefs& projections);

