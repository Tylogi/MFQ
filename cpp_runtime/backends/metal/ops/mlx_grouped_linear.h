#pragma once

#include "mlx_fp8_sq.h"
#include "mlx_mxfp4_sq.h"
#include "mlx_mx.h"
#include "mlx_nint.h"
#include "mlx_nint8_zero.h"
#include "mlx_vq.h"

#include <memory>
#include <stdexcept>
#include <variant>
#include <vector>

#include <mlx/mlx.h>

namespace mfq::metal {

// A projection which can participate in the ordinary grouped Q/K/V or
// gate/up Metal kernel. MlxGroupedLinear retains MLX array handles, so both
// packed and dense storage remain shared with the source weight.
using MlxGroupedLinearWeightRef = std::variant<
    const MlxNintWeight*,
    const MlxNint8ZeroWeight*,
    const MlxVqWeight*,
    const MlxFp8SqWeight*,
    const MlxMxfp4SqWeight*,
    const MlxMxWeight*,
    const mlx::core::array*>;

class MlxGroupedLinearUnsupported : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Packed projection coordinator.
//
// All projections share an input width but may use different output widths.
// NINT QKV and gate/up groups use one metadata-driven grouped operator;
// per-neuron q/k choices never create profile-specific NINT kernels. Other
// established formats may retain their format-level grouped paths.
// Expert-shaped/rotated NEPQ belongs to the MoE path and is rejected here.
class MlxGroupedLinear {
public:
    static constexpr int max_rows() noexcept {
        return 16;
    }

    explicit MlxGroupedLinear(
        std::vector<MlxGroupedLinearWeightRef> weights);

    bool supports(const mlx::core::array& input) const noexcept;

    std::vector<mlx::core::array> matmul(
        const mlx::core::array& input) const;

    // Decode-only two-projection SwiGLU. NINT reuses its metadata-driven
    // decoder, including adaptive q/k rows; MXFP8 retains its native path.
    bool supports_single_row_swiglu(
        const mlx::core::array& input) const noexcept;
    mlx::core::array single_row_swiglu(
        const mlx::core::array& input,
        float limit) const;

    // MXFP8 verifier counterpart to single_row_swiglu. It retains the same
    // two 16-lane reduction trees for each of M=2..6 independent rows.
    bool supports_small_m_swiglu(
        const mlx::core::array& input) const noexcept;
    mlx::core::array small_m_swiglu(
        const mlx::core::array& input,
        float limit) const;
    std::vector<mlx::core::array> operator()(
        const mlx::core::array& input) const {
        return matmul(input);
    }

    int input_size() const noexcept;
    int total_output_size() const noexcept;
    std::size_t projection_count() const noexcept;
    const std::vector<int>& output_sizes() const noexcept;

    // Logical bytes in the referenced packed weights. This does not imply
    // that the grouped object owns another copy of those bytes.
    std::size_t packed_nbytes() const noexcept;

    // NINT projection groups bind the retained metadata streams directly.
    // Other production groups may bind each source array to a format-level
    // Metal dispatch; groups which exceed the direct buffer limit and contain
    // VQ or MX are unsupported.
    bool uses_zero_copy_storage() const noexcept;
    std::size_t copied_packed_nbytes() const noexcept;

    // True when a single-row projection group is executed by one fused
    // operator instead of replaying the member projections independently.
    bool supports_single_row_projection_fusion() const noexcept;

    // True when this coordinator owns a real multi-projection Metal dispatch
    // for at least one supported row count. Graph-only retained groups must not
    // hide smaller compatible groups from MlxProjectionBatch's partitioner.
    bool has_projection_fusion() const noexcept;

    // True when a float16, single-row invocation can use the MXFP8
    // projection-fused decode kernel.
    bool has_single_row_mxfp8_fast_path() const noexcept;

private:
    mlx::core::array run_nint_projection_group(
        const mlx::core::array& source,
        std::size_t rows,
        bool swiglu,
        float limit) const;

    struct Impl;
    std::shared_ptr<Impl> impl_;
};

} // namespace mfq::metal
