#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <mlx/mlx.h>

#include "mfq/fp8_sq_blob.h"

namespace mfq::metal {

bool is_fp8_sq_dtype(std::string_view dtype) noexcept;

struct Fp8SqDescriptor {
    int format_version = 1;
    double aggregate_bpw = 0.0;
    double distribution_entropy = 0.0;
};

// MXFP8-SQ and FP8-128SQ deliberately retain distinct public dtype and scale
// contracts.  This runtime object shares their q-stream plumbing, then
// dispatches separate Metal kernel entry points for the two scale families.
class MlxFp8SqWeight {
public:
    static MlxFp8SqWeight from_blob(
        std::string_view dtype,
        const std::vector<std::uint8_t>& blob);
    static MlxFp8SqWeight from_blob(
        std::string_view dtype,
        std::span<const std::uint8_t> blob);

    mlx::core::array dequantize(
        mlx::core::Dtype dtype = mlx::core::float16) const;
    mlx::core::array embedding(
        const mlx::core::array& rows,
        mlx::core::Dtype dtype = mlx::core::float16) const;
    mlx::core::array matmul(const mlx::core::array& input) const;
    mlx::core::array routed_matmul(
        const mlx::core::array& input,
        const mlx::core::array& expert_ids,
        const mlx::core::array& expert_map,
        int out_per_expert) const;
    mlx::core::array backward_input(
        const mlx::core::array& output_gradient) const;

    // Projection-fused Q/K/V and gate/up execution.  MXFP8-SQ and FP8-128SQ
    // remain distinct scale contracts and compile under distinct kernel keys;
    // grouping never rewrites either packed representation.
    static std::vector<mlx::core::array> projection_group_matmul(
        std::span<const MlxFp8SqWeight> weights,
        const mlx::core::array& input);

    const std::string& dtype() const noexcept { return dtype_; }
    int input_size() const noexcept { return layout_.width; }
    int output_size() const noexcept { return layout_.outputs; }
    int block_rows() const noexcept { return layout_.block_rows; }
    int block_columns() const noexcept { return layout_.block_columns; }
    const mfq::fp8sq::Layout& wire_layout() const noexcept { return layout_; }
    const Fp8SqDescriptor& descriptor() const noexcept { return descriptor_; }
    bool has_uniform_q() const noexcept {
        return descriptor_.distribution_entropy == 0.0;
    }
    std::size_t packed_nbytes() const noexcept { return blob_.nbytes(); }

private:
    MlxFp8SqWeight(
        std::string dtype,
        mlx::core::array blob,
        mlx::core::array row_q,
        mlx::core::array row_symbol_byte_offsets,
        mfq::fp8sq::Layout layout,
        Fp8SqDescriptor descriptor);

    std::string dtype_;
    mlx::core::array blob_;
    mlx::core::array row_q_;
    mlx::core::array row_symbol_byte_offsets_;
    mfq::fp8sq::Layout layout_;
    Fp8SqDescriptor descriptor_;
};

} // namespace mfq::metal
