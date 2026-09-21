#pragma once

#include "cuda_execution.h"
#include "mfq/kernels/cuda/mxfp4_sq.h"

#include <cstdint>
#include <vector>

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

Mxfp4SqWeight to_device_mxfp4_sq(
    const std::vector<uint8_t>& payload,
    bool cuda,
    int device = -1);
