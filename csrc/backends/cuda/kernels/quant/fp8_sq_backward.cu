// Tool-only FP8-SQ / FP8-128SQ input backward. The native inference library
// compiles fp8_sq.cu alone; the Python calibration extension adds this unit.
// The dense weight comes from the shared inference dequant entry point, so the
// packed decode stays one implementation.
#include "kernels/quant/fp8_sq.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

#include "fp8_sq_impl.cuh"
#include "packed_backward.cuh"

namespace {

template <bool MXFP8>
mfq_tensor_backend::Tensor dequant_weight(
        const mfq_tensor_backend::Tensor& blob,
        const mfq_tensor_backend::Tensor& row_q,
        const mfq_tensor_backend::Tensor& row_symbol_byte_offsets,
        const Layout& layout) {
    if constexpr (MXFP8) {
        return mxfp8_sq_dequant_cuda(
            blob, row_q, row_symbol_byte_offsets,
            layout.outputs, layout.width, layout.block_rows,
            layout.block_columns, layout.scale_rows, layout.scale_columns,
            static_cast<std::int64_t>(layout.palettes),
            static_cast<std::int64_t>(layout.symbols),
            static_cast<std::int64_t>(layout.scales), false);
    }
    return fp8_128_sq_dequant_cuda(
        blob, row_q, row_symbol_byte_offsets,
        layout.outputs, layout.width, layout.scale_kind,
        static_cast<std::int64_t>(layout.palettes),
        static_cast<std::int64_t>(layout.symbols),
        static_cast<std::int64_t>(layout.scales), false);
}

template <bool MXFP8>
mfq_tensor_backend::Tensor backward_input(
        mfq_tensor_backend::Tensor blob,
        mfq_tensor_backend::Tensor row_q,
        mfq_tensor_backend::Tensor row_symbol_byte_offsets,
        mfq_tensor_backend::Tensor output_gradient,
        Layout layout) {
    validate_metadata(blob, row_q, row_symbol_byte_offsets, layout, MXFP8);
    MFQ_RUNTIME_CHECK(
        output_gradient.is_cuda() &&
        output_gradient.get_device() == blob.get_device() &&
        output_gradient.dim() == 2 && output_gradient.is_contiguous() &&
        output_gradient.size(1) == layout.outputs &&
        output_gradient.size(0) <= std::numeric_limits<int>::max() &&
        (output_gradient.scalar_type() == mfq_tensor_backend::kFloat16 ||
         output_gradient.scalar_type() == mfq_tensor_backend::kBFloat16 ||
         output_gradient.scalar_type() == mfq_tensor_backend::kFloat32),
        "FP8-SQ backward requires contiguous CUDA FP16/BF16/FP32 [M,N]");
    const MfqCudaGuard guard(blob.device());
    auto result = mfq_tensor_backend::empty(
        {output_gradient.size(0), layout.width}, output_gradient.options());
    if (output_gradient.size(0) == 0) return result;
    auto weight = dequant_weight<MXFP8>(
        blob, row_q, row_symbol_byte_offsets, layout);
    mfq_packed_backward::launch_dense_half_weight(
        output_gradient, weight, result,
        static_cast<int>(output_gradient.size(0)),
        layout.outputs, layout.width, mfq_current_cuda_stream());
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}

}  // namespace

mfq_tensor_backend::Tensor mxfp8_sq_backward_input_cuda(
        mfq_tensor_backend::Tensor blob,
        mfq_tensor_backend::Tensor row_q,
        mfq_tensor_backend::Tensor row_symbol_byte_offsets,
        mfq_tensor_backend::Tensor output_gradient,
        std::int64_t outputs, std::int64_t width,
        std::int64_t block_rows, std::int64_t block_columns,
        std::int64_t scale_rows, std::int64_t scale_columns,
        std::int64_t palettes_offset, std::int64_t symbols_offset,
        std::int64_t scales_offset) {
    return backward_input<true>(
        std::move(blob), std::move(row_q),
        std::move(row_symbol_byte_offsets), std::move(output_gradient),
        make_layout(outputs, width, block_rows, block_columns,
                    scale_rows, scale_columns, 1,
                    palettes_offset, symbols_offset, scales_offset));
}

mfq_tensor_backend::Tensor fp8_128_sq_backward_input_cuda(
        mfq_tensor_backend::Tensor blob,
        mfq_tensor_backend::Tensor row_q,
        mfq_tensor_backend::Tensor row_symbol_byte_offsets,
        mfq_tensor_backend::Tensor output_gradient,
        std::int64_t outputs, std::int64_t width,
        std::int64_t scale_kind,
        std::int64_t palettes_offset, std::int64_t symbols_offset,
        std::int64_t scales_offset) {
    return backward_input<false>(
        std::move(blob), std::move(row_q),
        std::move(row_symbol_byte_offsets), std::move(output_gradient),
        make_layout(outputs, width, 128, 128,
                    (outputs + 127) / 128, (width + 127) / 128,
                    scale_kind, palettes_offset, symbols_offset, scales_offset));
}
