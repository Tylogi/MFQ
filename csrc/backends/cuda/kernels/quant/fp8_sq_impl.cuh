#pragma once

// Shared by the FP8-SQ inference unit and the tool-only backward unit: the
// blob layout both sides build and the metadata checks both sides run.

#include "kernels/quant/fp8_sq.h"

#include <cstddef>
#include <cstdint>
#include <limits>

namespace {

struct Layout {
    int outputs = 0;
    int width = 0;
    int block_rows = 0;
    int block_columns = 0;
    int scale_rows = 0;
    int scale_columns = 0;
    int scale_kind = 0;
    std::size_t palettes = 0;
    std::size_t symbols = 0;
    std::size_t scales = 0;
};

void validate_metadata(
        const mfq_tensor_backend::Tensor& blob,
        const mfq_tensor_backend::Tensor& row_q,
        const mfq_tensor_backend::Tensor& row_symbol_byte_offsets,
        const Layout& layout,
        bool mxfp8) {
    MFQ_RUNTIME_CHECK(
        blob.is_cuda() && blob.scalar_type() == mfq_tensor_backend::kUInt8 &&
        blob.dim() == 1 && blob.is_contiguous(),
        "FP8-SQ blob must be contiguous rank-1 CUDA uint8");
    MFQ_RUNTIME_CHECK(
        row_q.is_cuda() && row_q.get_device() == blob.get_device() &&
        row_q.scalar_type() == mfq_tensor_backend::kUInt8 &&
        row_q.dim() == 1 && row_q.is_contiguous() &&
        row_q.numel() == layout.outputs &&
        row_symbol_byte_offsets.is_cuda() &&
        row_symbol_byte_offsets.get_device() == blob.get_device() &&
        row_symbol_byte_offsets.scalar_type() == mfq_tensor_backend::kInt32 &&
        row_symbol_byte_offsets.dim() == 1 &&
        row_symbol_byte_offsets.is_contiguous() &&
        row_symbol_byte_offsets.numel() == layout.outputs,
        "FP8-SQ row metadata must be matching CUDA arrays");
    MFQ_RUNTIME_CHECK(
        layout.outputs > 0 && layout.width > 0 &&
        layout.block_rows > 0 && layout.block_columns > 0 &&
        layout.scale_rows == (layout.outputs + layout.block_rows - 1) /
            layout.block_rows &&
        layout.scale_columns == (layout.width + layout.block_columns - 1) /
            layout.block_columns &&
        layout.palettes < layout.symbols && layout.symbols <= layout.scales &&
        layout.scales < static_cast<std::size_t>(blob.numel()),
        "FP8-SQ layout metadata is inconsistent");
    const std::size_t scale_count =
        static_cast<std::size_t>(layout.scale_rows) * layout.scale_columns;
    const std::size_t scale_itemsize = mxfp8 || layout.scale_kind == 1
        ? 1
        : layout.scale_kind == 4 ? 4 : 2;
    MFQ_RUNTIME_CHECK(
        scale_count <=
            (static_cast<std::size_t>(blob.numel()) - layout.scales) /
                scale_itemsize,
        "FP8-SQ scale payload exceeds its blob");
    if (mxfp8) {
        MFQ_RUNTIME_CHECK(
            layout.scale_kind == 1 &&
            ((layout.block_rows == 1 && layout.block_columns == 32) ||
             (layout.block_rows == 32 && layout.block_columns == 32) ||
             (layout.block_rows == 128 && layout.block_columns == 128)),
            "MXFP8-SQ scale contract is invalid");
    } else {
        MFQ_RUNTIME_CHECK(
            layout.block_rows == 128 && layout.block_columns == 128 &&
            (layout.scale_kind == 2 || layout.scale_kind == 3 ||
             layout.scale_kind == 4),
            "FP8-128SQ scale contract is invalid");
    }
}

Layout make_layout(
        std::int64_t outputs,
        std::int64_t width,
        std::int64_t block_rows,
        std::int64_t block_columns,
        std::int64_t scale_rows,
        std::int64_t scale_columns,
        std::int64_t scale_kind,
        std::int64_t palettes,
        std::int64_t symbols,
        std::int64_t scales) {
    MFQ_RUNTIME_CHECK(
        outputs > 0 && outputs <= std::numeric_limits<int>::max() &&
        width > 0 && width <= std::numeric_limits<int>::max() &&
        block_rows > 0 && block_rows <= std::numeric_limits<int>::max() &&
        block_columns > 0 && block_columns <= std::numeric_limits<int>::max() &&
        scale_rows > 0 && scale_rows <= std::numeric_limits<int>::max() &&
        scale_columns > 0 && scale_columns <= std::numeric_limits<int>::max() &&
        palettes >= 0 && symbols >= 0 && scales >= 0,
        "FP8-SQ host layout exceeds CUDA limits");
    return {
        static_cast<int>(outputs),
        static_cast<int>(width),
        static_cast<int>(block_rows),
        static_cast<int>(block_columns),
        static_cast<int>(scale_rows),
        static_cast<int>(scale_columns),
        static_cast<int>(scale_kind),
        static_cast<std::size_t>(palettes),
        static_cast<std::size_t>(symbols),
        static_cast<std::size_t>(scales),
    };
}

}  // namespace
