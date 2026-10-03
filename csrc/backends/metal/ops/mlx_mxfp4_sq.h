#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include <mlx/mlx.h>

#include "mfq/mxfp4_sq_blob.h"

namespace mfq::metal {

bool is_mxfp4_sq_dtype(std::string_view dtype) noexcept;

// Frozen format-level palettes shared by the SQ1/SQ2/SQ3 row profiles. Every
// entry is an ordinary E2M1 nibble; the tables are not stored per tensor and
// do not define additional runtime formats. SQ4 stores native E2M1 nibbles.
inline constexpr std::array<std::uint8_t, 64> kMxfp4Sq1PaletteNibbles{
    15, 6, 14, 7, 13, 7, 15, 5, 15, 7, 11, 6, 14, 3, 12, 6,
    14, 6, 14, 4, 10, 6, 14, 2, 13, 6, 14, 5, 9, 6, 14, 1,
    15, 2, 11, 7, 10, 7, 15, 3, 9, 7, 15, 1, 14, 0, 13, 5,
    12, 4, 11, 3, 10, 2, 9, 1, 13, 4, 12, 5, 11, 4, 12, 3,
};

inline constexpr std::array<std::uint8_t, 128> kMxfp4Sq2PaletteNibbles{
    15, 13, 0,  5,  15, 13, 1,  6,  15, 12, 2,  6,  14, 11, 0,  3,  14, 11, 1,
    5,  14, 11, 2,  6,  14, 10, 1,  4,  14, 10, 1,  5,  14, 10, 3,  6,  14, 10,
    4,  7,  14, 9,  5,  7,  13, 10, 1,  4,  13, 9,  3,  6,  13, 0,  5,  7,  12,
    9,  2,  5,  12, 9,  2,  6,  15, 12, 3,  7,  11, 0,  3,  6,  12, 9,  3,  6,
    13, 9,  2,  6,  14, 10, 3,  7,  15, 11, 4,  7,  15, 13, 9,  4,  15, 11, 1,
    5,  15, 11, 2,  6,  15, 12, 1,  6,  15, 12, 2,  7,  15, 13, 0,  6,  14, 11,
    0,  4,  13, 1,  5,  7,  15, 14, 13, 12, 15, 14, 13, 11,
};

struct Mxfp4SqDescriptor {
    int format_version = 2;
    double aggregate_bpw = 0.0;
    double distribution_entropy = 0.0;
};

inline constexpr std::array<std::uint8_t, 256> kMxfp4Sq3PaletteNibbles{
    15, 14, 13, 11, 0,  3,  5,  7,  15, 13, 11, 0,  3,  5,  6,  7,  15, 14, 13,
    11, 1,  4,  6,  7,  13, 11, 9,  0,  1,  2,  3,  6,  15, 14, 12, 9,  3,  5,
    6,  7,  15, 13, 12, 10, 0,  2,  4,  5,  13, 12, 10, 0,  2,  4,  5,  7,  14,
    12, 10, 0,  2,  4,  5,  7,  15, 14, 12, 10, 0,  3,  5,  7,  15, 13, 11, 0,
    2,  4,  5,  6,  15, 13, 11, 0,  2,  4,  6,  7,  15, 14, 12, 10, 0,  2,  4,
    5,  15, 14, 12, 10, 0,  2,  4,  6,  14, 13, 11, 0,  2,  4,  5,  6,  13, 12,
    10, 0,  3,  5,  6,  7,  15, 13, 12, 10, 0,  2,  4,  6,  14, 13, 12, 10, 0,
    3,  5,  6,  15, 13, 12, 10, 0,  3,  5,  6,  13, 12, 10, 0,  2,  4,  6,  7,
    14, 13, 11, 0,  2,  4,  5,  7,  13, 12, 9,  0,  2,  4,  5,  6,  14, 12, 10,
    0,  2,  4,  6,  7,  15, 14, 13, 11, 0,  3,  6,  7,  15, 14, 11, 0,  3,  5,
    6,  7,  14, 13, 12, 10, 0,  2,  4,  5,  14, 13, 12, 10, 0,  3,  5,  7,  15,
    14, 11, 0,  2,  4,  6,  7,  14, 11, 10, 9,  0,  1,  2,  3,  13, 11, 0,  2,
    4,  5,  6,  7,  15, 13, 12, 10, 0,  2,  4,  7,  15, 12, 10, 0,  2,  4,  5,
    7,  15, 13, 12, 10, 1,  4,  6,  7,
};

// One public MXFP4-SQ runtime weight. SQ1/SQ2/SQ3/SQ4 are per-neuron profiles
// selected by the self-describing blob; ordinary Linear and routed MFE both
// execute through the same metadata-driven Metal matmul kernel.
class MlxMxfp4SqWeight {
public:
    static MlxMxfp4SqWeight from_blob(
        const std::vector<std::uint8_t>& blob);
    static MlxMxfp4SqWeight from_blob(
        std::span<const std::uint8_t> blob);

    mlx::core::array dequantize(
        mlx::core::Dtype dtype = mlx::core::float16) const;
    mlx::core::array embedding(
        const mlx::core::array& rows,
        mlx::core::Dtype dtype = mlx::core::float16) const;
    mlx::core::array matmul(const mlx::core::array& input) const;
    // Route a cohort packed as [local_experts * out_per_expert, K].
    // expert_map maps global/physical expert IDs to cohort-local IDs and
    // uses -1 for experts owned by another cohort.  This is another launch
    // mode of the same packed matmul kernel used by ordinary Linear.
    mlx::core::array routed_matmul(
        const mlx::core::array& input,
        const mlx::core::array& expert_ids,
        const mlx::core::array& expert_map,
        int out_per_expert) const;
    mlx::core::array backward_input(
        const mlx::core::array& output_gradient) const;

    // Projection-fused Q/K/V and gate/up execution.  Every member retains its
    // own self-describing q=1..4 stream; the common Metal dispatch only shares
    // the activation read and launch, so no profile-specific tensor format is
    // introduced by grouping.
    static std::vector<mlx::core::array> projection_group_matmul(
        std::span<const MlxMxfp4SqWeight> weights,
        const mlx::core::array& input);

    int bits() const noexcept {
        return layout_.bits;
    }
    int format_version() const noexcept {
        return layout_.version;
    }
    int input_size() const noexcept {
        return layout_.width;
    }
    int output_size() const noexcept {
        return layout_.outputs;
    }
    std::uint8_t matrix_scale_base() const noexcept {
        return static_cast<std::uint8_t>(layout_.base);
    }
    const Mxfp4SqDescriptor& descriptor() const noexcept {
        return descriptor_;
    }
    bool has_uniform_q() const noexcept {
        return descriptor_.distribution_entropy == 0.0;
    }
    const mfq::sq::Layout& wire_layout() const noexcept {
        return layout_;
    }
    std::size_t packed_nbytes() const noexcept {
        return blob_.nbytes();
    }

private:
    MlxMxfp4SqWeight(
        mlx::core::array blob,
        mlx::core::array row_q,
        mlx::core::array row_symbol_byte_offsets,
        mlx::core::array row_auxiliary,
        mfq::sq::Layout layout,
        Mxfp4SqDescriptor descriptor);

    mlx::core::array blob_;
    mlx::core::array row_q_;
    mlx::core::array row_symbol_byte_offsets_;
    mlx::core::array row_auxiliary_;
    mfq::sq::Layout layout_;
    Mxfp4SqDescriptor descriptor_;
};

} // namespace mfq::metal
