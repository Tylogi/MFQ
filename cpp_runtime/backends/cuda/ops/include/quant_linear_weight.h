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
    Mxfp4SqLinear mxfp4_sq;
    Fp8SqLinear fp8_sq;
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
            int gate_mode) const;

    mfq_tensor_backend::Tensor forward_dense(mfq_tensor_backend::Tensor x) const;

    mfq_tensor_backend::Tensor forward(
            CudaExecutionContext& execution,
            mfq_tensor_backend::Tensor x) const;
    mfq_tensor_backend::Tensor forward_bf16_output(
            CudaExecutionContext& execution,
            mfq_tensor_backend::Tensor x) const;
    mfq_tensor_backend::Tensor forward_mxfp8_groupwise(
            CudaExecutionContext& execution,
            mfq_tensor_backend::Tensor grouped,
            int64_t groups) const;
    mfq_tensor_backend::Tensor forward_input_mul(
            CudaExecutionContext& execution,
            mfq_tensor_backend::Tensor x,
            mfq_tensor_backend::Tensor gate,
            int mode) const;
    mfq_tensor_backend::Tensor forward_input_mul_f32_kld(
            CudaExecutionContext& execution,
            mfq_tensor_backend::Tensor x,
            mfq_tensor_backend::Tensor gate,
            int mode) const;
    int64_t out() const;
    int64_t neuron_len() const;
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

