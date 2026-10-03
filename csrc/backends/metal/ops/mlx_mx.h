#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include <mlx/mlx.h>

namespace mfq::metal {

bool is_mx_dtype(std::string_view dtype) noexcept;

// Model-neutral activation fake-quantization boundaries used by native-QAT
// graphs. The returned values are unpacked F16, but every element is exactly
// representable by the corresponding native MX format.
mlx::core::array mlx_mxfp8_sim(
    const mlx::core::array& input);
mlx::core::array mlx_mxfp4_e4m3_scale_sim(
    const mlx::core::array& input);

// Weighted RMSNorm, adjacent-pair tail RoPE, and MXFP8 activation simulation.
// A device-specialized fused kernel is selected by operator geometry; all
// other inputs use the composition of the same model-neutral primitives.
mlx::core::array mlx_weighted_rms_rope_mxfp8_sim(
    const mlx::core::array& input,
    const mlx::core::array& norm_weight,
    float eps,
    int rotary_dimension,
    const mlx::core::array& cosine,
    const mlx::core::array& sine);

class MlxMxWeight {
public:
    static MlxMxWeight from_blob(
        std::string_view dtype,
        const std::vector<std::uint8_t>& blob);
    static MlxMxWeight from_blob(
        std::string_view dtype,
        std::span<const std::uint8_t> blob);
    static MlxMxWeight from_arrays(
        std::string_view dtype,
        mlx::core::array values,
        mlx::core::array scales,
        int input_size,
        int output_size);

    mlx::core::array matmul(const mlx::core::array& input) const;
    // Zero-copy projection group for native row/block-32 MXFP8 weights.
    // This is the storage used by QAT Attention projections: all members keep
    // their own packed payload and expanded E8M0 sidecar, but share one Metal
    // submission over the common activation.
    static std::vector<mlx::core::array> projection_group_matmul(
        std::span<const MlxMxWeight> weights,
        const mlx::core::array& input);
    mlx::core::array dequantize(
        mlx::core::Dtype dtype = mlx::core::float16) const;
    mlx::core::array embedding(
        const mlx::core::array& token_ids,
        mlx::core::Dtype dtype = mlx::core::float16) const;
    mlx::core::array grouped_row_matmul(
        const mlx::core::array& input,
        int group_count) const;
    mlx::core::array grouped_row_matmul_inverse_rope(
        const mlx::core::array& input,
        int group_count,
        const mlx::core::array& cosine,
        const mlx::core::array& sine,
        int head_dimension,
        int rotary_dimension) const;

    int bits() const noexcept { return bits_; }
    int input_size() const noexcept { return input_size_; }
    int output_size() const noexcept { return output_size_; }
    int scale_row_block_size() const noexcept {
        return mxfp8_scale_row_block_size_;
    }
    int scale_column_block_size() const noexcept {
        return mxfp8_scale_column_block_size_;
    }
    std::size_t packed_nbytes() const noexcept;

    // Read-only packed storage used by fused/grouped Metal kernels.
    const mlx::core::array& packed_values() const noexcept {
        return values_;
    }
    const mlx::core::array& block_scales() const noexcept {
        return scales_;
    }

private:
    MlxMxWeight(
        mlx::core::array values,
        mlx::core::array scales,
        int bits,
        int input_size,
        int output_size);

    mlx::core::array values_;
    mlx::core::array scales_;
    std::optional<mlx::core::array> expanded_mxfp8_scales_;
    int mxfp8_scale_row_block_size_ = 0;
    int mxfp8_scale_column_block_size_ = 0;
    int bits_ = 0;
    int input_size_ = 0;
    int output_size_ = 0;
};

} // namespace mfq::metal
