#pragma once

#include "cuda_execution.h"
#include "mfq/kernels/cuda/fp8_sq.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

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
};

Fp8SqWeight to_device_fp8_sq(
    std::string_view dtype,
    const std::vector<uint8_t>& payload,
    bool cuda,
    int device = -1);
mfq_tensor_backend::Tensor dequant_fp8_sq(
    const Fp8SqWeight& weight, bool fp32);
