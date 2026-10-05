#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include <mlx/mlx.h>

namespace mfq::metal {

bool is_nint_dtype(std::string_view dtype) noexcept;

namespace detail {
std::string_view nint_matmul_metal_header() noexcept;
std::string_view nint_matmul_metal_body() noexcept;
}

struct NintDescriptor {
    int format_version = 2;
    double aggregate_bpw = 0.0;
    double distribution_entropy = 0.0;
};

class MlxNintWeight {
public:
    static MlxNintWeight from_blob(
        std::span<const std::uint8_t> blob);

    mlx::core::array matmul(const mlx::core::array& input) const;
    mlx::core::array matmul_packed(const mlx::core::array& input) const;
    mlx::core::array matmul_add(
        const mlx::core::array& input,
        const mlx::core::array& residual) const;
    // Decode/small-M epilogue for a shared expert. The NINT projection keeps
    // its output in registers, while routed expert rows are reduced and the
    // sigmoid shared gate is applied before the single final write.
    mlx::core::array matmul_moe_shared(
        const mlx::core::array& input,
        const mlx::core::array& routed_pairs,
        const mlx::core::array& route_weights,
        const mlx::core::array& gate_logits) const;
    // Routed MFE projection over this packed expert cohort.  This is another
    // invocation mode of the ordinary NINT matmul kernel, not a separate MoE
    // decoder. expert_map maps global expert IDs to cohort-local rows and -1
    // for experts owned by another MFE cohort.
    mlx::core::array routed_matmul(
        const mlx::core::array& input,
        const mlx::core::array& expert_ids,
        const mlx::core::array& expert_map,
        int out_per_expert) const;
    // Decode-only fast path for a single FP16 row. Supported layouts compute
    // the LM-head projection and greedy argmax without materializing logits.
    std::optional<mlx::core::array> greedy_argmax(
        const mlx::core::array& input) const;
    bool can_fuse_swiglu(
        const MlxNintWeight& up) const noexcept;
    mlx::core::array swiglu(
        const MlxNintWeight& up,
        const mlx::core::array& input) const;
    mlx::core::array dequantize(
        mlx::core::Dtype dtype = mlx::core::float16) const;
    mlx::core::array embedding(
        const mlx::core::array& token_ids,
        mlx::core::Dtype dtype = mlx::core::float16) const;
    // O-LoRA-style grouped projection:
    // [..., M, G, K] x [G * O, K] -> [..., M, G, O].
    // Reuses the routed mode of the same metadata-driven NINT matmul kernel;
    // the optional return type is retained for the packed-linear interface.
    std::optional<mlx::core::array> grouped_row_matmul(
        const mlx::core::array& input,
        int group_count) const;

    int bits() const noexcept {
        return bits_;
    }
    int group_size() const noexcept {
        return group_size_;
    }
    int groups() const noexcept {
        return groups_;
    }
    int input_size() const noexcept {
        return input_size_;
    }
    int output_size() const noexcept {
        return output_size_;
    }
    const NintDescriptor& descriptor() const noexcept {
        return descriptor_;
    }
    std::size_t packed_nbytes() const noexcept;

    // Read-only packed storage views used by fused/grouped Metal kernels.
    // The arrays remain owned by this weight.
    const mlx::core::array& packed_values() const noexcept {
        return q_packed_;
    }
    const mlx::core::array& sub_scales() const noexcept {
        return sub_scale_;
    }
    const mlx::core::array& sub_mins() const noexcept {
        return sub_min_;
    }
    const mlx::core::array& neuron_scales() const noexcept {
        return neuron_scale_;
    }
    const mlx::core::array& neuron_mins() const noexcept {
        return neuron_min_;
    }
    const mlx::core::array& row_q_layout() const noexcept {
        return row_q_layout_;
    }
    const mlx::core::array& row_q_byte_offsets() const noexcept {
        return row_q_byte_offsets_;
    }
    const mlx::core::array& row_metadata() const noexcept {
        return row_metadata_;
    }
    bool has_uniform_q_bits() const noexcept {
        return uniform_q_bits_;
    }
private:
    mlx::core::array matmul_impl(
        const mlx::core::array& input,
        const mlx::core::array* residual,
        const mlx::core::array* routed_pairs = nullptr,
        const mlx::core::array* route_weights = nullptr,
        const mlx::core::array* gate_logits = nullptr,
        bool allow_dequantize = true) const;

    MlxNintWeight(
        mlx::core::array q_packed,
        mlx::core::array sub_scale,
        mlx::core::array sub_min,
        mlx::core::array neuron_scale,
        mlx::core::array neuron_min,
        mlx::core::array row_q_layout,
        mlx::core::array row_q_byte_offsets,
        mlx::core::array row_metadata,
        int bits,
        int group_size,
        int groups,
        int input_size,
        int output_size,
        NintDescriptor descriptor,
        bool uniform_q_bits);

    mlx::core::array q_packed_;
    mlx::core::array sub_scale_;
    mlx::core::array sub_min_;
    mlx::core::array neuron_scale_;
    mlx::core::array neuron_min_;
    mlx::core::array row_q_layout_;
    mlx::core::array row_q_byte_offsets_;
    // Runtime-only packed row descriptor: layout, byte offset, neuron scale,
    // and neuron minimum. Dense kernels bind this single leaf instead of four
    // independent MLX arrays; the canonical wire representation is unchanged.
    mlx::core::array row_metadata_;
    int bits_ = 0;
    int group_size_ = 0;
    int groups_ = 0;
    int input_size_ = 0;
    int output_size_ = 0;
    NintDescriptor descriptor_;
    bool uniform_q_bits_ = false;
};

// Load-time packed view of two shape-compatible NINT projections. The
// combined buffers let a fused Gate/Up kernel bind both projections without
// exceeding Metal's buffer-slot limit. Canonical weights remain unchanged.
class MlxNintSwiGluPair {
public:
    static std::optional<MlxNintSwiGluPair> from_weights(
        const MlxNintWeight& gate,
        const MlxNintWeight& up);

    const mlx::core::array& packed_values() const noexcept {
        return q_packed_;
    }
    const mlx::core::array& row_metadata() const noexcept {
        return row_metadata_;
    }
    const mlx::core::array& sub_scales() const noexcept {
        return sub_scale_;
    }
    const mlx::core::array& sub_mins() const noexcept {
        return sub_min_;
    }
    int group_size() const noexcept { return group_size_; }
    int groups() const noexcept { return groups_; }
    int input_size() const noexcept { return input_size_; }
    int output_size() const noexcept { return output_size_; }

private:
    MlxNintSwiGluPair(
        mlx::core::array q_packed,
        mlx::core::array row_metadata,
        mlx::core::array sub_scale,
        mlx::core::array sub_min,
        int group_size,
        int groups,
        int input_size,
        int output_size)
        : q_packed_(std::move(q_packed)),
          row_metadata_(std::move(row_metadata)),
          sub_scale_(std::move(sub_scale)),
          sub_min_(std::move(sub_min)),
          group_size_(group_size),
          groups_(groups),
          input_size_(input_size),
          output_size_(output_size) {}

    mlx::core::array q_packed_;
    mlx::core::array row_metadata_;
    mlx::core::array sub_scale_;
    mlx::core::array sub_min_;
    int group_size_ = 0;
    int groups_ = 0;
    int input_size_ = 0;
    int output_size_ = 0;
};

} // namespace mfq::metal
