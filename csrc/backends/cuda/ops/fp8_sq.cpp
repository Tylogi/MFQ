#include "fp8_sq.h"
#include "format.h"

using namespace mfq::cuda::quant_format;

mfq_tensor_backend::Tensor dequant_fp8_sq(
        const Fp8SqWeight & weight,
        bool fp32) {
    if (!weight.blob.is_cuda()) {
        auto output = mfq_tensor_backend::empty(
            {weight.out, weight.neuron_len},
            weight.blob.options().dtype(mfq_tensor_backend::kFloat32));
        const auto* blob = weight.blob.data_ptr<uint8_t>();
        const auto layout = mfq::fp8sq::parse(
            weight.dtype, blob, weight.blob.numel());
        mfq_parallel_for(0, weight.out, 1, [&](int64_t begin, int64_t end) {
            for (auto row = begin; row < end; ++row) {
                for (int64_t column = 0;
                        column < weight.neuron_len; ++column) {
                    output.data_ptr<float>()[row * weight.neuron_len + column] =
                        mfq::fp8sq::decode_cpu(
                            blob, layout,
                            weight.row_q.data_ptr<uint8_t>()[row],
                            weight.row_symbol_byte_offsets.data_ptr<int32_t>()[row],
                            row, column);
                }
            }
        });
        return output.to(fp32
            ? mfq_tensor_backend::kFloat32
            : mfq_tensor_backend::kFloat16);
    }
    if (weight.dtype == "MXFP8-SQ") {
        return mxfp8_sq_dequant_cuda(
            weight.blob, weight.row_q, weight.row_symbol_byte_offsets,
            weight.out, weight.neuron_len,
            weight.block_rows, weight.block_columns,
            weight.scale_rows, weight.scale_columns,
            weight.palettes_offset, weight.symbols_offset,
            weight.scales_offset, fp32);
    }
    if (weight.dtype == "FP8-128SQ") {
        return fp8_128_sq_dequant_cuda(
            weight.blob, weight.row_q, weight.row_symbol_byte_offsets,
            weight.out, weight.neuron_len, weight.scale_kind,
            weight.palettes_offset, weight.symbols_offset,
            weight.scales_offset, fp32);
    }
    throw std::runtime_error("unsupported FP8-SQ dequant dtype");
}
