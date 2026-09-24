#include "mlx_moe.h"

#include "mfq_container.h"
#include "mfq_mfe_prefill_embedded.h"
#include "mlx_nint.h"
#include "mlx_nint8_zero.h"
#include "mlx_mx.h"
#include "mlx_fp8_sq.h"
#include "mlx_eval_timing.h"
#include "mlx_mxfp4_sq.h"
#include "mlx_platform.h"
#include "mlx_reference.h"
#include "mlx_staging_allocator.h"
#include "mlx_vq.h"

#include <mlx/allocator.h>
#include <mlx/backend/metal/device.h>
#include <mlx/memory.h>
#include <mlx/primitives.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <future>
#include <list>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <sstream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

namespace mfq::metal {
namespace {

using mlx::core::CompileOptions;
using mlx::core::Dtype;
using mlx::core::MathMode;
using mlx::core::Shape;
using mlx::core::array;

constexpr int kDescriptorSize = 32;

constexpr int kFamily = 0;
constexpr int kLocalExpert = 1;
constexpr int kOut = 2;
constexpr int kInput = 3;

constexpr int kFamilyNint = 0;
constexpr int kNintBits = 4;
constexpr int kNintGroupSize = 5;
constexpr int kNintGroups = 6;
constexpr int kNintQOffset = 7;
constexpr int kNintSubOffset = 8;
constexpr int kNintAnchorOffset = 9;
constexpr int kNintQ5Execution = 10;
constexpr int kNintRowQLayoutOffset = 11;
constexpr int kNintRowQByteOffsetsOffset = 12;
constexpr int kNintV2 = 13;

// Keep the same family value and fields as the Python Metal descriptor.  The
// unused value 1 remains available for the VQ-family extension.
constexpr int kFamilyNint8Zero = 2;
constexpr int kQ8Groups = 4;
constexpr int kQ8QOffset = 5;
constexpr int kQ8ScaleOffset = 6;

constexpr int kFamilyMxfp4 = 3;
constexpr int kFamilyMxfp8 = 4;
constexpr int kFamilyBf16 = 5;
constexpr int kFamilyF16 = 6;
constexpr int kMxGroups = 4;
constexpr int kMxValueOffset = 5;
constexpr int kMxScaleOffset = 6;
constexpr int kDenseValueOffset = 4;

bool apple_m5_family() noexcept {
    return mlx_apple_chip_starts_with("Apple M5");
}

bool apple_m3_ultra() noexcept {
    return mlx_apple_chip_is("Apple M3 Ultra");
}

bool mixed_grouped_nax_enabled(int route_count) noexcept {
    // The heterogeneous NAX path is a large-M prefill kernel.  Keep small-M
    // and short-tail shapes on the compatibility kernel, whose boundary
    // handling is both cheaper and already exhaustive.
    constexpr int kMinRoutes = 1024;
    if (route_count < kMinRoutes) {
        return false;
    }
    const char* value = std::getenv(
        "MFQ_METAL_MIXED_PREFILL_NAX");
    if (value == nullptr) {
        return apple_m5_family();
    }
    return std::string_view(value) != "0"
        && std::string_view(value) != "false"
        && std::string_view(value) != "off";
}

int grouped_mmq_tile_columns(
    int block_rows,
    int output_width) noexcept {
    const char* value = std::getenv(
        "MFQ_METAL_GROUPED_MMQ_TILE_COLUMNS");
    if (value != nullptr) {
        if (block_rows == 32 && std::string_view(value) == "128") {
            return 128;
        }
        if (std::string_view(value) == "96") {
            if (block_rows == 48 || block_rows == 64) {
                return 96;
            }
        }
        if (std::string_view(value) == "64") {
            return 64;
        }
    }
    // Use the largest column tile that remains within the 32-KiB
    // threadgroup-memory budget for this row tile. Wider column tiles reduce
    // repeated activation staging; retain BN64 for genuinely narrow outputs.
    if (block_rows == 32 && output_width >= 512) {
        return 128;
    }
    if (block_rows == 48 && output_width >= 512) {
        return 96;
    }
    return block_rows == 64 && output_width >= 1024 ? 96 : 64;
}

bool mxfp4_nax_prefill_enabled(
    int route_count,
    int experts,
    int input_width,
    int output_width,
    bool allow_automatic) noexcept {
    constexpr int kDefaultMinRoutes = 1024;
    const bool native_256_expert_geometry = experts == 256 && (
        (input_width == 4096 &&
         (output_width == 2048 || output_width == 4096)) ||
        (input_width == 2048 && output_width == 4096));
    const bool native_384_expert_geometry = experts == 384 && (
        (input_width == 5120 &&
         (output_width == 2304 || output_width == 4608)) ||
        (input_width == 2304 && output_width == 5120));
    const bool native_optimized_geometry =
        native_256_expert_geometry || native_384_expert_geometry;
    // M5 uses MFQ's unified grouped-NAX path below.  MLX gather_qmm is a
    // win only for the measured M3 Ultra native expert geometries; selecting
    // it ahead of grouped-NAX on M5 more than doubles routed-projection time
    // at the 256-expert, 4096<->2048, top-6 production geometry.
    const bool automatic = route_count >= kDefaultMinRoutes
        && apple_m3_ultra() && native_optimized_geometry;
    const char* value = std::getenv(
        "MFQ_METAL_MFE_PREFILL_NAX");
    if (value == nullptr) {
        return allow_automatic && automatic;
    }
    const auto setting = std::string_view(value);
    if (
        setting == "1"
        || setting == "true"
        || setting == "on"
    ) {
        return true;
    }
    return setting == "auto" && allow_automatic && automatic;
}

bool mxfp4_nax_smallm_preferred(
    const array& expert_ids,
    int tokens,
    int addressable_experts,
    int logical_experts,
    int input_width,
    int output_width) noexcept {
    // Historical naming: this policy selects MLX gather_qmm, not an
    // unconditional hardware-NAX kernel. MLX dispatches the ordinary
    // fp_gather_qmm kernels on pre-NAX devices (including M3 Ultra) and may
    // select its NAX implementation only on hardware that supports it.
    const char* value = std::getenv(
        "MFQ_METAL_MFE_SMALLM_NAX");
    const auto setting = value == nullptr
        ? std::string_view("auto")
        : std::string_view(value);
    const bool force =
        setting == "1"
        || setting == "true"
        || setting == "on";
    const bool m3_ultra_native_geometry =
        apple_m3_ultra() && logical_experts == 256 && (
            (input_width == 4096 &&
             (output_width == 2048 || output_width == 4096)) ||
            (input_width == 2048 && output_width == 4096));
    if (
        tokens < 4
        || tokens > 6
        || (!force && setting != "auto" && setting != "adaptive")
        || (!force && !apple_m5_family() && !m3_ultra_native_geometry)
        || expert_ids.dtype() != mlx::core::int32
        || !expert_ids.flags().row_contiguous
        || !expert_ids.is_available()
    ) {
        return false;
    }
    const auto* ids = expert_ids.data<std::int32_t>();
    bool has_duplicate = false;
    for (std::size_t index = 0; index < expert_ids.size(); ++index) {
        if (ids[index] < 0 || ids[index] >= addressable_experts) {
            return false;
        }
        for (std::size_t previous = 0; previous < index; ++previous) {
            if (ids[index] == ids[previous]) {
                has_duplicate = true;
                break;
            }
        }
    }
    return has_duplicate;
}

bool mxfp4_decode_down_reduce_enabled() noexcept {
    const char* value = std::getenv(
        "MFQ_METAL_MFE_DECODE_DOWN_REDUCE");
    if (value == nullptr) {
        return true;
    }
    const auto setting = std::string_view(value);
    return setting != "0"
        && setting != "false"
        && setting != "off";
}

int mxfp4_decode_down_reduce_rows() noexcept {
    const char* value = std::getenv(
        "MFQ_METAL_MFE_DECODE_DOWN_REDUCE_ROWS");
    if (value != nullptr) {
        const auto setting = std::string_view(value);
        if (setting == "1") return 1;
        if (setting == "4") return 4;
    }
    return 2;
}

int mxfp4_decode_rows_per_simd() noexcept {
    const char* value = std::getenv(
        "MFQ_METAL_MFE_DECODE_ROWS_PER_SIMD");
    if (value != nullptr) {
        const auto setting = std::string_view(value);
        if (setting == "1") return 1;
        if (setting == "4") return 4;
    }
    return 2;
}

bool routed_sort_enabled(int tokens) noexcept {
    if (tokens <= 1) {
        return false;
    }
    const char* value = std::getenv(
        "MFQ_METAL_MFE_SORT_ROUTES");
    if (value == nullptr) {
        return true;
    }
    const auto setting = std::string_view(value);
    return !(
        setting == "0"
        || setting == "false"
        || setting == "off");
}

constexpr int kFamilyVq = 1;
constexpr int kVqGroupSize = 4;
constexpr int kVqGroups = 5;
constexpr int kVqVectorSize = 6;
constexpr int kVqVectors = 7;
constexpr int kVqIndexBits = 8;
constexpr int kVqStateBits = 9;
constexpr int kVqStates = 10;
constexpr int kVqEntries = 11;
constexpr int kVqCodeBanks = 12;
constexpr int kVqAuxMode = 13;
constexpr int kVqCodeBankMode = 14;
constexpr int kVqHasTableBanks = 15;
constexpr int kVqGroupsPerSuper = 16;
constexpr int kVqSupergroups = 17;
constexpr int kVqIndicesOffset = 18;
constexpr int kVqStateOffset = 19;
constexpr int kVqAuxOffset = 20;
constexpr int kVqAnchorOffset = 21;
constexpr int kVqCodebookOffset = 22;
constexpr int kVqScaleOffset = 23;
constexpr int kVqStateBankOffset = 24;
constexpr int kVqBankOffset = 25;
constexpr int kVqParameterOffset = 26;
constexpr int kVqRotationVariant = 27;
constexpr int kVqProfile = 28;
constexpr int kVqJscExecution = 29;
constexpr int kVqResidualCodebookOffset = 30;
constexpr int kVqResidualRecordOffset = 31;
constexpr int kVqProfileGeneric = 0;
constexpr int kVqProfileJsc4 = 1;
constexpr int kVqProfileNpqS = 2;
constexpr int kVqProfileNvq1L = 3;
constexpr int kVqProfileJsc8 = 4;
constexpr int kVqProfileNpqL = 5;
constexpr int kVqProfileNvq1S = 6;
constexpr int kVqProfileJscExtended8 = 7;
constexpr std::uint32_t kGroupedVqVectorProfileMask =
    (std::uint32_t{1} << kVqProfileGeneric)
    | (std::uint32_t{1} << kVqProfileNpqS)
    | (std::uint32_t{1} << kVqProfileNvq1L)
    | (std::uint32_t{1} << kVqProfileNpqL)
    | (std::uint32_t{1} << kVqProfileNvq1S);
constexpr std::uint32_t kGroupedJscExtendedProfileMask =
    std::uint32_t{1} << kVqProfileJscExtended8;

constexpr const char* kMoeHeader = R"METAL(
template <typename Stream>
inline uint mfq_moe_read_bits(
    Stream stream,
    uint value_index,
    uint bits
) {
    uint residual_bits = (value_index & 7u) * bits;
    uint byte_index =
        (value_index >> 3) * bits + (residual_bits >> 3);
    uint shift = residual_bits & 7u;
    uint packed = uint(stream[byte_index]);
    if (shift + bits > 8u) {
        packed |= uint(stream[byte_index + 1u]) << 8;
    }
    if (shift + bits > 16u) {
        packed |= uint(stream[byte_index + 2u]) << 16;
    }
    return (packed >> shift)
        & ((1u << bits) - 1u);
}

inline ushort4 mfq_moe_read_nint_row_quad(
    device const uchar* stream,
    uint row_byte_offset,
    uint row_bit_shift,
    uint value_index,
    uint bits
) {
    uint row_relative_bits = row_bit_shift + value_index * bits;
    uint byte_index = row_byte_offset + (row_relative_bits >> 3u);
    uint shift = row_relative_bits & 7u;
    uint required_bits = shift + 4u * bits;
    packed_uchar4 bytes =
        *reinterpret_cast<device const packed_uchar4*>(stream + byte_index);
    uint packed = as_type<uint>(bytes);
    if (shift != 0u) {
        packed = (packed >> shift)
            | (required_bits > 32u
                ? uint(stream[byte_index + 4u]) << (32u - shift)
                : 0u);
    }
    uint mask = (1u << bits) - 1u;
    return ushort4(
        packed & mask,
        (packed >> bits) & mask,
        (packed >> (2u * bits)) & mask,
        (packed >> (3u * bits)) & mask);
}

template <uint BITS, typename Stream>
inline uint3 mfq_moe_read_npq_group_indices(
    Stream stream,
    uint row,
    uint group,
    uint vectors
) {
    const uint first = group * 3u;
    if (first + 2u >= vectors) {
        return uint3(
            first < vectors
                ? mfq_moe_read_bits(
                      stream,
                      row * vectors + first,
                      BITS)
                : 0u,
            first + 1u < vectors
                ? mfq_moe_read_bits(
                      stream,
                      row * vectors + first + 1u,
                      BITS)
                : 0u,
            0u);
    }
    const uint linear = row * vectors + first;
    const uint bit = linear * BITS;
    const uint byte = bit >> 3;
    const uint shift = bit & 7u;
    uint packed = uint(stream[byte])
        | (uint(stream[byte + 1u]) << 8)
        | (uint(stream[byte + 2u]) << 16);
    if (shift + 3u * BITS > 24u) {
        packed |= uint(stream[byte + 3u]) << 24;
    }
    const uint mask = (1u << BITS) - 1u;
    return uint3(
        (packed >> shift) & mask,
        (packed >> (shift + BITS)) & mask,
        (packed >> (shift + 2u * BITS)) & mask);
}

constant constexpr float mfq_moe_mxfp4_lut[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

inline float mfq_moe_mxfp4_value(uchar code) {
    return mfq_moe_mxfp4_lut[uint(code & 15u)];
}

inline float mfq_moe_e8m0(uchar raw) {
    if (raw == 255u) {
        return NAN;
    }
    uint bits = raw == 0u ? 0x00400000u : uint(raw) << 23u;
    return as_type<float>(bits);
}

inline float mfq_moe_mxfp8_value(uchar raw) {
    uint magnitude = uint(raw & 0x7fu);
    uint exponent = magnitude >> 3u;
    uint mantissa = magnitude & 7u;
    if (exponent == 15u && mantissa == 7u) {
        return NAN;
    }
    float value = exponent == 0u
        ? float(mantissa) * 0.001953125f
        : as_type<float>((exponent + 120u) << 23u)
            * (1.0f + float(mantissa) * 0.125f);
    return (raw & 0x80u) == 0u ? value : -value;
}

inline float4 mfq_moe_load_code4(
    device const int8_t* stream,
    uint offset
) {
    return float4(
        *(device const char4*)(stream + offset));
}

inline float4 mfq_moe_load_code4(
    constant const int8_t* stream,
    uint offset
) {
    return float4(
        *(constant const char4*)(stream + offset));
}

template <
    uint VECTOR_SIZE,
    uint MATRIX_ROWS,
    uint EXECUTION_LAYOUT,
    typename XStream,
    typename IndexStream,
    typename StateStream,
    typename AuxStream,
    typename ScaleStream,
    typename StateBankStream,
    typename CodebookStream
>
inline void mfq_moe_jsc_profile(
    XStream x,
    IndexStream indices_stream,
    StateStream state_stream,
    AuxStream aux_stream,
    ScaleStream scale_stream,
    StateBankStream state_bank_stream,
    CodebookStream codebook_stream,
    uint x_offset,
    thread const uint* outputs,
    thread const float* row_anchors,
    thread float* accumulators,
    uint groups,
    uint vectors,
    uint index_bits,
    uint entries,
    uint indices_offset,
    uint state_offset,
    uint aux_offset,
    uint codebook_offset,
    uint scale_offset,
    uint state_bank_offset,
    uint signs,
    uint k_lane,
    uint k_lanes,
    uint k_size,
    uint execution_layout
) {
    constexpr uint VECTOR_SHIFT =
        VECTOR_SIZE == 4u ? 2u : 3u;
    constexpr uint BYTES_PER_SIGN =
        VECTOR_SIZE == 4u ? 3u : 2u;
    for (
        uint group = k_lane;
        group < groups;
        group += k_lanes
    ) {
        uint selected_code_banks[MATRIX_ROWS];
        float weight_scales[MATRIX_ROWS];
        uint2 group_records[MATRIX_ROWS];
        for (uint row = 0u; row < MATRIX_ROWS; ++row) {
            uint state_index = outputs[row] * groups + group;
            uint state;
            if (EXECUTION_LAYOUT != 0u && execution_layout == 2u) {
                uint record_offset = indices_offset + state_index * 8u;
                group_records[row] = uint2(
                    uint(indices_stream[record_offset])
                        | (uint(indices_stream[record_offset + 1u]) << 8u)
                        | (uint(indices_stream[record_offset + 2u]) << 16u)
                        | (uint(indices_stream[record_offset + 3u]) << 24u),
                    uint(indices_stream[record_offset + 4u])
                        | (uint(indices_stream[record_offset + 5u]) << 8u)
                        | (uint(indices_stream[record_offset + 6u]) << 16u)
                        | (uint(indices_stream[record_offset + 7u]) << 24u));
                state = group_records[row].y >> 28u;
            } else {
                uint state_byte = uint(state_stream[
                    state_offset + (state_index >> 1)
                ]);
                state = (
                    state_byte >> ((state_index & 1u) * 4u)
                ) & 15u;
            }
            selected_code_banks[row] = uint(
                state_bank_stream[state_bank_offset + state]);
            weight_scales[row] =
                row_anchors[row]
                * scale_stream[scale_offset + state];
        }
        for (
            uint sign_block = 0u;
            sign_block < 3u;
            ++sign_block
        ) {
            uint column_base = group * 24u + sign_block * 8u;
            if (column_base >= k_size) {
                break;
            }
            float4 activation0 = float4(
                float(x[x_offset + column_base]),
                float(x[x_offset + column_base + 1u]),
                float(x[x_offset + column_base + 2u]),
                float(x[x_offset + column_base + 3u]));
            float4 activation1 = float4(
                float(x[x_offset + column_base + 4u]),
                float(x[x_offset + column_base + 5u]),
                float(x[x_offset + column_base + 6u]),
                float(x[x_offset + column_base + 7u]));
            uint first_vector = column_base >> VECTOR_SHIFT;
            for (uint row = 0u; row < MATRIX_ROWS; ++row) {
                uint index0;
                uint index1 = 0u;
                uint sign_value;
                if (EXECUTION_LAYOUT != 0u && execution_layout == 2u) {
                    uint2 record = group_records[row];
                    uint segment = sign_block == 0u
                        ? record.x & 0xfffffu
                        : (sign_block == 1u
                            ? ((record.x >> 20u) | (record.y << 12u))
                                & 0xfffffu
                            : (record.y >> 8u) & 0xfffffu);
                    index0 = segment & 4095u;
                    sign_value = segment >> 12u;
                } else if (EXECUTION_LAYOUT != 0u && execution_layout == 1u) {
                    uint execution_offset = indices_offset + (
                        outputs[row] * signs + column_base / 8u
                    ) * BYTES_PER_SIGN;
                    index0 = uint(indices_stream[execution_offset]);
                    if (VECTOR_SIZE == 4u) {
                        index1 = uint(indices_stream[
                            execution_offset + 1u]);
                    }
                    sign_value = uint(indices_stream[
                        execution_offset + BYTES_PER_SIGN - 1u]);
                } else {
                    uint index_linear =
                        outputs[row] * vectors + first_vector;
                    index0 = index_bits == 8u
                        ? uint(indices_stream[
                              indices_offset + index_linear])
                        : mfq_moe_read_bits(
                              indices_stream + indices_offset,
                              index_linear,
                              index_bits);
                    if (VECTOR_SIZE == 4u) {
                        index1 = index_bits == 8u
                            ? uint(indices_stream[
                                  indices_offset
                                  + index_linear + 1u])
                            : mfq_moe_read_bits(
                                  indices_stream + indices_offset,
                                  index_linear + 1u,
                                  index_bits);
                    }
                    sign_value = mfq_moe_read_bits(
                        aux_stream + aux_offset,
                        outputs[row] * signs + column_base / 8u,
                        7u);
                }
                uint bank_base =
                    selected_code_banks[row] * entries;
                uint code0_offset =
                    (bank_base + index0) * VECTOR_SIZE;
                uint code1_offset = VECTOR_SIZE == 4u
                    ? (bank_base + index1) * 4u
                    : code0_offset + 4u;
                bool4 negative0 = bool4(
                    (sign_value & 1u) != 0u,
                    (sign_value & 2u) != 0u,
                    (sign_value & 4u) != 0u,
                    (sign_value & 8u) != 0u);
                bool4 negative1 = bool4(
                    (sign_value & 16u) != 0u,
                    (sign_value & 32u) != 0u,
                    (sign_value & 64u) != 0u,
                    EXECUTION_LAYOUT != 0u && execution_layout != 0u
                        ? (sign_value & 128u) != 0u
                        : (popcount(sign_value) & 1u) != 0u);
                float4 code0 = mfq_moe_load_code4(
                    codebook_stream,
                    codebook_offset + code0_offset);
                float4 code1 = mfq_moe_load_code4(
                    codebook_stream,
                    codebook_offset + code1_offset);
                code0 = select(code0, -code0, negative0);
                code1 = select(code1, -code1, negative1);
                float code_dot =
                    dot(activation0, code0)
                    + dot(activation1, code1);
                accumulators[row] = fma(
                    weight_scales[row],
                    code_dot,
                    accumulators[row]);
            }
        }
    }
}
)METAL";

constexpr const char* kMoeSource = R"METAL(
    constexpr uint SIMD_GROUPS = 2u;
    constexpr uint K_LANES = uint(K_LANES_VALUE);
    constexpr uint LANE_GROUPS = 32u / K_LANES;
    constexpr uint ROWS_PER_SIMD = uint(ROWS_PER_SIMD_VALUE);
    constexpr uint MATRIX_ROWS =
        uint(FUSED_SWIGLU) != 0u
            ? 2u * ROWS_PER_SIMD
            : ROWS_PER_SIMD;
    constexpr uint ROWS_PER_PHYSICAL_SIMD =
        LANE_GROUPS * ROWS_PER_SIMD;
    constexpr uint ROWS_PER_TG =
        SIMD_GROUPS * ROWS_PER_PHYSICAL_SIMD;
    constexpr uint OUTPUT_TILES =
        (uint(OUT) + ROWS_PER_TG - 1u) / ROWS_PER_TG;

    uint lane = thread_index_in_simdgroup;
    uint k_lane = lane & (K_LANES - 1u);
    uint lane_group = lane / K_LANES;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint workgroup = threadgroup_position_in_grid.x;
    uint output_tile = workgroup % OUTPUT_TILES;
    uint projection_index = workgroup / OUTPUT_TILES;
    uint projection =
        projection_index % uint(PROJECTIONS);
    uint sorted_slot =
        projection_index / uint(PROJECTIONS);
    uint route_index = uint(SORTED_ROUTES) != 0u
        ? uint(route_order[sorted_slot])
        : sorted_slot;
    uint route = route_index % uint(ROUTES);
    uint token = route_index / uint(ROUTES);
    if (token >= uint(TOKENS)) {
        return;
    }

    uint output_base =
        output_tile * ROWS_PER_TG
        + simd_group * ROWS_PER_PHYSICAL_SIMD
        + lane_group * ROWS_PER_SIMD;
    int logical_expert =
        int(expert_ids[token * uint(ROUTES) + route]);
    int expert = logical_expert;
    if (PACKED_EXPERT_IDS != 0) {
        expert = (logical_expert >> 8) - 1;
    } else if (EXPERT_MAP_SIZE != 0) {
        expert = logical_expert >= 0 &&
                logical_expert < int(EXPERT_MAP_SIZE)
            ? expert_map[logical_expert]
            : -1;
    }
    if (expert < 0 || expert >= int(EXPERTS)) {
        for (
            uint row = 0u;
            row < ROWS_PER_SIMD;
            ++row
        ) {
            uint output = output_base + row;
            if (k_lane == 0u && output < uint(OUT)) {
                y[
                    (
                        (
                            token * uint(ROUTES) + route
                        ) * uint(PROJECTIONS)
                        + projection
                    ) * uint(OUT)
                    + output
                ] = T(0.0f);
            }
        }
        return;
    }

    uint descriptor_base = (
        uint(expert) * uint(PROJECTIONS)
        + projection
    ) * uint(DESCRIPTOR_SIZE);
    uint family =
        uint(descriptors[descriptor_base]);
    uint local_expert =
        uint(descriptors[descriptor_base + 1u]);
    uint rotation_variant =
        uint(descriptors[descriptor_base + 27u]);
    uint x_offset = (
        rotation_variant * uint(VARIANT_STRIDE)
        + (
            uint(SHARED_INPUT) != 0u
                ? token
                : token * uint(ROUTES) + route
        )
    ) * uint(K);
    float accumulators[MATRIX_ROWS] = {0.0f};

    if (
        (uint(FAMILY_MASK) & 1u) != 0u
        && family == 0u
    ) {
        uint group_size =
            uint(descriptors[descriptor_base + 5u]);
        uint groups =
            uint(descriptors[descriptor_base + 6u]);
        uint q_offset =
            uint(descriptors[descriptor_base + 7u]);
        uint sub_offset =
            uint(descriptors[descriptor_base + 8u]);
        uint anchor_offset =
            uint(descriptors[descriptor_base + 9u]);
        uint row_layout_offset =
            uint(descriptors[descriptor_base + 11u]);
        uint row_byte_offsets_offset =
            uint(descriptors[descriptor_base + 12u]);
        device const uint* row_byte_offsets =
            reinterpret_cast<device const uint*>(
                nint_q + row_byte_offsets_offset);

        uint outputs[MATRIX_ROWS];
        uint q_widths[MATRIX_ROWS];
        uint q_row_byte_offsets[MATRIX_ROWS];
        uint q_row_bit_shifts[MATRIX_ROWS];
        float neuron_scales[MATRIX_ROWS];
        float neuron_minimums[MATRIX_ROWS];
        for (uint row = 0u; row < MATRIX_ROWS; ++row) {
            uint output = min(
                output_base + (
                    uint(FUSED_SWIGLU) != 0u
                        ? (row / ROWS_PER_SIMD) * uint(OUT)
                            + row % ROWS_PER_SIMD
                        : row
                ),
                uint(MATRIX_OUT) - 1u);
            uint pool_output =
                local_expert * uint(MATRIX_OUT) + output;
            uint layout = uint(nint_q[
                row_layout_offset + pool_output]);
            outputs[row] = pool_output;
            q_widths[row] = layout & 15u;
            q_row_byte_offsets[row] =
                row_byte_offsets[pool_output];
            q_row_bit_shifts[row] = layout >> 4u;
            neuron_scales[row] =
                nint_anchor_scale[anchor_offset + pool_output];
            neuron_minimums[row] =
                nint_anchor_min[anchor_offset + pool_output];
        }

        for (uint group = k_lane; group < groups; group += K_LANES) {
            float scales[MATRIX_ROWS];
            float minimums[MATRIX_ROWS];
            for (uint row = 0u; row < MATRIX_ROWS; ++row) {
                uint metadata = outputs[row] * groups + group;
                scales[row] = neuron_scales[row]
                    * float(nint_sub_scale[sub_offset + metadata]);
                minimums[row] = neuron_minimums[row]
                    * float(nint_sub_min[sub_offset + metadata]);
            }
            uint column_base = group * group_size;
            for (uint element = 0u; element < group_size; element += 4u) {
                uint column = column_base + element;
                float4 activations;
                if (element + 3u < group_size && column + 3u < uint(K)) {
                    activations = float4(
                        x[x_offset + column],
                        x[x_offset + column + 1u],
                        x[x_offset + column + 2u],
                        x[x_offset + column + 3u]);
                } else {
                    activations = float4(
                        element < group_size && column < uint(K)
                            ? float(x[x_offset + column]) : 0.0f,
                        element + 1u < group_size && column + 1u < uint(K)
                            ? float(x[x_offset + column + 1u]) : 0.0f,
                        element + 2u < group_size && column + 2u < uint(K)
                            ? float(x[x_offset + column + 2u]) : 0.0f,
                        element + 3u < group_size && column + 3u < uint(K)
                            ? float(x[x_offset + column + 3u]) : 0.0f);
                }
                for (uint row = 0u; row < MATRIX_ROWS; ++row) {
                    ushort4 quantized = mfq_moe_read_nint_row_quad(
                        nint_q + q_offset,
                        q_row_byte_offsets[row],
                        q_row_bit_shifts[row],
                        column,
                        q_widths[row]);
                    float4 decoded = scales[row] * float4(quantized)
                        - minimums[row];
                    accumulators[row] += dot(activations, decoded);
                }
            }
        }
    } else if (
        (uint(FAMILY_MASK) & 2u) != 0u
        && family == 1u
    ) {
        uint group_size =
            uint(descriptors[descriptor_base + 4u]);
        uint groups =
            uint(descriptors[descriptor_base + 5u]);
        uint vector_size =
            uint(descriptors[descriptor_base + 6u]);
        uint vectors =
            uint(descriptors[descriptor_base + 7u]);
        uint index_bits =
            uint(descriptors[descriptor_base + 8u]);
        uint state_bits =
            uint(descriptors[descriptor_base + 9u]);
        uint states =
            uint(descriptors[descriptor_base + 10u]);
        uint entries =
            uint(descriptors[descriptor_base + 11u]);
        uint code_banks =
            uint(descriptors[descriptor_base + 12u]);
        uint aux_mode =
            uint(descriptors[descriptor_base + 13u]);
        uint code_bank_mode =
            uint(descriptors[descriptor_base + 14u]);
        uint has_table_banks =
            uint(descriptors[descriptor_base + 15u]);
        uint groups_per_super =
            uint(descriptors[descriptor_base + 16u]);
        uint supergroups =
            uint(descriptors[descriptor_base + 17u]);
        uint indices_offset =
            uint(descriptors[descriptor_base + 18u]);
        uint state_offset =
            uint(descriptors[descriptor_base + 19u]);
        uint aux_offset =
            uint(descriptors[descriptor_base + 20u]);
        uint anchor_offset =
            uint(descriptors[descriptor_base + 21u]);
        uint codebook_offset =
            uint(descriptors[descriptor_base + 22u]);
        uint scale_offset =
            uint(descriptors[descriptor_base + 23u]);
        uint state_bank_offset =
            uint(descriptors[descriptor_base + 24u]);
        uint bank_offset =
            uint(descriptors[descriptor_base + 25u]);
        uint parameter_offset =
            uint(descriptors[descriptor_base + 26u]);
        uint vectors_per_group =
            (
                group_size + vector_size - 1u
            ) / vector_size;
        uint signs = (uint(K) + 7u) / 8u;

        uint outputs[MATRIX_ROWS];
        float row_anchors[MATRIX_ROWS];
        for (
            uint row = 0u;
            row < MATRIX_ROWS;
            ++row
        ) {
            uint output = min(
                output_base + (
                    uint(FUSED_SWIGLU) != 0u
                        ? (row / ROWS_PER_SIMD) * uint(OUT)
                            + row % ROWS_PER_SIMD
                        : row
                ),
                uint(MATRIX_OUT) - 1u);
            uint pool_output =
                local_expert * uint(MATRIX_OUT) + output;
            outputs[row] = pool_output;
            row_anchors[row] =
                vq_anchors[anchor_offset + pool_output];
        }

        uint profile =
            uint(descriptors[descriptor_base + 28u]) & 255u;
        uint cohort_jsc_layout =
            uint(descriptors[descriptor_base + 29u]);
        if (
            (uint(VQ_PROFILE_MASK) & 2u) != 0u
            && profile == 1u
        ) {
            for (
                uint group = k_lane;
                group < groups;
                group += K_LANES
            ) {
                uint selected_code_banks[MATRIX_ROWS];
                float weight_scales[MATRIX_ROWS];
                for (
                    uint row = 0u;
                    row < MATRIX_ROWS;
                    ++row
                ) {
                    uint state_index =
                        outputs[row] * groups + group;
                    uint state_byte = uint(vq_state[
                        state_offset + (state_index >> 1)
                    ]);
                    uint state = (
                        state_byte
                        >> ((state_index & 1u) * 4u)
                    ) & 15u;
                    selected_code_banks[row] = uint(
                        vq_state_to_codebank[
                            state_bank_offset + state
                        ]);
                    weight_scales[row] =
                        row_anchors[row]
                        * vq_scales[scale_offset + state];
                }
                constexpr uint vector_shift = 2u;
                constexpr uint vectors_per_sign = 2u;
                for (
                    uint sign_block = 0u;
                    sign_block < 3u;
                    ++sign_block
                ) {
                    uint column_base =
                        group * 24u + sign_block * 8u;
                    if (column_base >= uint(K)) {
                        break;
                    }
                    float4 activation0 = float4(
                        float(x[x_offset + column_base]),
                        float(x[x_offset + column_base + 1u]),
                        float(x[x_offset + column_base + 2u]),
                        float(x[x_offset + column_base + 3u]));
                    float4 activation1 = float4(
                        float(x[x_offset + column_base + 4u]),
                        float(x[x_offset + column_base + 5u]),
                        float(x[x_offset + column_base + 6u]),
                        float(x[x_offset + column_base + 7u]));
                    uint first_vector =
                        column_base >> vector_shift;
                    for (
                        uint row = 0u;
                        row < MATRIX_ROWS;
                        ++row
                    ) {
                        uint indices[2] = {0u, 0u};
                        uint sign_value;
                        if (uint(JSC_EXECUTION_LAYOUT) != 0u) {
                            constexpr uint bytes_per_sign = 3u;
                            uint execution_offset =
                                indices_offset
                                + (
                                    outputs[row] * signs
                                    + column_base / 8u
                                ) * bytes_per_sign;
                            indices[0] = uint(vq_indices[
                                execution_offset]);
                            indices[1] = uint(vq_indices[
                                execution_offset + 1u]);
                            sign_value = uint(vq_indices[
                                execution_offset
                                + bytes_per_sign - 1u]);
                        } else {
                            for (
                                uint local_vector = 0u;
                                local_vector < vectors_per_sign;
                                ++local_vector
                            ) {
                                indices[local_vector] = uint(vq_indices[
                                    indices_offset
                                    + outputs[row] * vectors
                                    + first_vector + local_vector
                                ]);
                            }
                            sign_value = mfq_moe_read_bits(
                                vq_aux + aux_offset,
                                outputs[row] * signs
                                    + column_base / 8u,
                                7u);
                        }
                        uint bank_base =
                            selected_code_banks[row] * 256u;
                        uint code0_offset =
                            (bank_base + indices[0]) * 4u;
                        uint code1_offset =
                            (bank_base + indices[1]) * 4u;
                        bool4 negative0 = bool4(
                            (sign_value & 1u) != 0u,
                            (sign_value & 2u) != 0u,
                            (sign_value & 4u) != 0u,
                            (sign_value & 8u) != 0u);
                        bool4 negative1 = bool4(
                            (sign_value & 16u) != 0u,
                            (sign_value & 32u) != 0u,
                            (sign_value & 64u) != 0u,
                            uint(JSC_EXECUTION_LAYOUT) != 0u
                                ? (sign_value & 128u) != 0u
                                : (popcount(sign_value) & 1u) != 0u);
                        float4 code0 = mfq_moe_load_code4(
                            vq_codebooks,
                            codebook_offset + code0_offset);
                        float4 code1 = mfq_moe_load_code4(
                            vq_codebooks,
                            codebook_offset + code1_offset);
                        code0 = select(code0, -code0, negative0);
                        code1 = select(code1, -code1, negative1);
                        float code_dot =
                            dot(activation0, code0)
                            + dot(activation1, code1);
                        accumulators[row] = fma(
                            weight_scales[row],
                            code_dot,
                            accumulators[row]);
                    }
                }
            }
        } else if (
            (uint(VQ_PROFILE_MASK) & 16u) != 0u
            && profile == 4u
        ) {
            mfq_moe_jsc_profile<
                8u,
                MATRIX_ROWS,
                JSC_EXECUTION_LAYOUT
            >(
                x,
                vq_indices,
                vq_state,
                vq_aux,
                vq_scales,
                vq_state_to_codebank,
                vq_codebooks,
                x_offset,
                outputs,
                row_anchors,
                accumulators,
                groups,
                vectors,
                index_bits,
                entries,
                indices_offset,
                state_offset,
                aux_offset,
                codebook_offset,
                scale_offset,
                state_bank_offset,
                signs,
                k_lane,
                K_LANES,
                uint(K),
                cohort_jsc_layout);
        } else if (
            (uint(VQ_PROFILE_MASK) & 128u) != 0u
            && profile == 7u
        ) {
            // Extended E8 JSC profiles keep 10- or 12-bit indices packed.
            // They otherwise share the vectorized 24-column execution shape
            // with NVQ2J, including its banked codebooks and parity sign.
            mfq_moe_jsc_profile<
                8u,
                MATRIX_ROWS,
                JSC_EXECUTION_LAYOUT
            >(
                x,
                vq_indices,
                vq_state,
                vq_aux,
                vq_scales,
                vq_state_to_codebank,
                vq_codebooks,
                x_offset,
                outputs,
                row_anchors,
                accumulators,
                groups,
                vectors,
                index_bits,
                entries,
                indices_offset,
                state_offset,
                aux_offset,
                codebook_offset,
                scale_offset,
                state_bank_offset,
                signs,
                k_lane,
                K_LANES,
                uint(K),
                cohort_jsc_layout);
        } else if (
            (uint(VQ_PROFILE_MASK) & 4u) != 0u
            && profile == 2u
        ) {
            for (
                uint group = k_lane;
                group < groups;
                group += K_LANES
            ) {
                uint states_by_row[MATRIX_ROWS];
                float weight_scales[MATRIX_ROWS];
                for (
                    uint row = 0u;
                    row < MATRIX_ROWS;
                    ++row
                ) {
                    uint state_index =
                        outputs[row] * groups + group;
                    uint state = mfq_moe_read_bits(
                        vq_state + state_offset,
                        state_index,
                        2u);
                    states_by_row[row] = state;
                    weight_scales[row] =
                        row_anchors[row]
                        * vq_scales[scale_offset + state];
                }
                uint3 indices_by_row[MATRIX_ROWS];
                if (uint(NPQ_GROUPED_INDICES) != 0u) {
                    for (
                        uint row = 0u;
                        row < MATRIX_ROWS;
                        ++row
                    ) {
                        indices_by_row[row] =
                            mfq_moe_read_npq_group_indices<6u>(
                                vq_indices + indices_offset,
                                outputs[row],
                                group,
                                vectors);
                    }
                }

                for (
                    uint local_vector = 0u;
                    local_vector < 3u;
                    ++local_vector
                ) {
                    uint column_base =
                        group * 24u + local_vector * 8u;
                    if (column_base >= uint(K)) {
                        break;
                    }
                    float4 activation0 = float4(
                        float(x[x_offset + column_base]),
                        float(x[x_offset + column_base + 1u]),
                        float(x[x_offset + column_base + 2u]),
                        float(x[x_offset + column_base + 3u]));
                    float4 activation1 = float4(
                        float(x[x_offset + column_base + 4u]),
                        float(x[x_offset + column_base + 5u]),
                        float(x[x_offset + column_base + 6u]),
                        float(x[x_offset + column_base + 7u]));
                    uint vector = column_base >> 3;
                    for (
                        uint row = 0u;
                        row < MATRIX_ROWS;
                        ++row
                    ) {
                        uint index =
                            uint(NPQ_GROUPED_INDICES) != 0u
                            ? indices_by_row[row][local_vector]
                            : mfq_moe_read_bits(
                                  vq_indices + indices_offset,
                                  outputs[row] * vectors + vector,
                                  6u);
                        uint code_base = codebook_offset + (
                            states_by_row[row] * 64u + index
                        ) * 8u;
                        float4 code0 = mfq_moe_load_code4(
                            vq_codebooks, code_base);
                        float4 code1 = mfq_moe_load_code4(
                            vq_codebooks, code_base + 4u);
                        float code_dot =
                            dot(activation0, code0)
                            + dot(activation1, code1);
                        accumulators[row] = fma(
                            weight_scales[row],
                            code_dot,
                            accumulators[row]);
                    }
                }
            }
        } else if (
            (uint(VQ_PROFILE_MASK) & 32u) != 0u
            && profile == 5u
        ) {
            for (
                uint group = k_lane;
                group < groups;
                group += K_LANES
            ) {
                uint states_by_row[MATRIX_ROWS];
                float weight_scales[MATRIX_ROWS];
                for (
                    uint row = 0u;
                    row < MATRIX_ROWS;
                    ++row
                ) {
                    uint state_index =
                        outputs[row] * groups + group;
                    uint state = mfq_moe_read_bits(
                        vq_state + state_offset,
                        state_index,
                        3u);
                    states_by_row[row] = state;
                    weight_scales[row] =
                        row_anchors[row]
                        * vq_scales[scale_offset + state];
                }
                uint3 indices_by_row[MATRIX_ROWS];
                if (uint(NPQ_GROUPED_INDICES) != 0u) {
                    for (
                        uint row = 0u;
                        row < MATRIX_ROWS;
                        ++row
                    ) {
                        indices_by_row[row] =
                            mfq_moe_read_npq_group_indices<7u>(
                                vq_indices + indices_offset,
                                outputs[row],
                                group,
                                vectors);
                    }
                }

                for (
                    uint local_vector = 0u;
                    local_vector < 3u;
                    ++local_vector
                ) {
                    uint column_base =
                        group * 24u + local_vector * 8u;
                    if (column_base >= uint(K)) {
                        break;
                    }
                    float4 activation0 = float4(
                        float(x[x_offset + column_base]),
                        float(x[x_offset + column_base + 1u]),
                        float(x[x_offset + column_base + 2u]),
                        float(x[x_offset + column_base + 3u]));
                    float4 activation1 = float4(
                        float(x[x_offset + column_base + 4u]),
                        float(x[x_offset + column_base + 5u]),
                        float(x[x_offset + column_base + 6u]),
                        float(x[x_offset + column_base + 7u]));
                    uint vector = column_base >> 3;
                    for (
                        uint row = 0u;
                        row < MATRIX_ROWS;
                        ++row
                    ) {
                        uint index =
                            uint(NPQ_GROUPED_INDICES) != 0u
                            ? indices_by_row[row][local_vector]
                            : mfq_moe_read_bits(
                                  vq_indices + indices_offset,
                                  outputs[row] * vectors + vector,
                                  7u);
                        uint code_base = codebook_offset + (
                            states_by_row[row] * 128u + index
                        ) * 8u;
                        float4 code0 = mfq_moe_load_code4(
                            vq_codebooks, code_base);
                        float4 code1 = mfq_moe_load_code4(
                            vq_codebooks, code_base + 4u);
                        float code_dot =
                            dot(activation0, code0)
                            + dot(activation1, code1);
                        accumulators[row] = fma(
                            weight_scales[row],
                            code_dot,
                            accumulators[row]);
                    }
                }
            }
        } else if (
            (uint(VQ_PROFILE_MASK) & 64u) != 0u
            && profile == 6u
        ) {
            // NVQ1-S stores three 8-wide vectors per 24-column group.  Keep
            // its two-codebook/delta semantics, but avoid the scalar generic
            // VQ inner loop and redundant packed-bit address calculations.
            float delta = vq_parameters[parameter_offset];
            for (
                uint group = k_lane;
                group < groups;
                group += K_LANES
            ) {
                uint delta_by_row[MATRIX_ROWS];
                float weight_scales[MATRIX_ROWS];
                for (
                    uint row = 0u;
                    row < MATRIX_ROWS;
                    ++row
                ) {
                    uint state_index =
                        outputs[row] * groups + group;
                    uint state_byte = uint(vq_state[
                        state_offset + (state_index >> 1)
                    ]);
                    uint state = (
                        state_byte
                        >> ((state_index & 1u) * 4u)
                    ) & 15u;
                    delta_by_row[row] = mfq_moe_read_bits(
                        vq_aux + aux_offset,
                        state_index,
                        1u);
                    weight_scales[row] =
                        row_anchors[row]
                        * vq_scales[scale_offset + state];
                }

                for (
                    uint local_vector = 0u;
                    local_vector < 3u;
                    ++local_vector
                ) {
                    uint column_base =
                        group * 24u + local_vector * 8u;
                    if (column_base >= uint(K)) {
                        break;
                    }
                    float4 activation0 = float4(
                        float(x[x_offset + column_base]),
                        float(x[x_offset + column_base + 1u]),
                        float(x[x_offset + column_base + 2u]),
                        float(x[x_offset + column_base + 3u]));
                    float4 activation1 = float4(
                        float(x[x_offset + column_base + 4u]),
                        float(x[x_offset + column_base + 5u]),
                        float(x[x_offset + column_base + 6u]),
                        float(x[x_offset + column_base + 7u]));
                    uint vector = column_base >> 3;
                    for (
                        uint row = 0u;
                        row < MATRIX_ROWS;
                        ++row
                    ) {
                        uint index = mfq_moe_read_bits(
                            vq_indices + indices_offset,
                            outputs[row] * vectors + vector,
                            9u);
                        uint bank = delta_by_row[row];
                        float signed_delta =
                            bank != 0u ? -delta : delta;
                        uint code_base = codebook_offset + (
                            bank * 512u + index
                        ) * 8u;
                        float4 code0 = mfq_moe_load_code4(
                            vq_codebooks, code_base)
                            + float4(signed_delta);
                        float4 code1 = mfq_moe_load_code4(
                            vq_codebooks, code_base + 4u)
                            + float4(signed_delta);
                        float code_dot =
                            dot(activation0, code0)
                            + dot(activation1, code1);
                        accumulators[row] = fma(
                            weight_scales[row],
                            code_dot,
                            accumulators[row]);
                    }
                }
            }
        } else if (
            (uint(VQ_PROFILE_MASK) & 8u) != 0u
            && profile == 3u
        ) {
            float delta = vq_parameters[parameter_offset];
            for (
                uint group = k_lane;
                group < groups;
                group += K_LANES
            ) {
                uint delta_by_row[MATRIX_ROWS];
                float weight_scales[MATRIX_ROWS];
                for (
                    uint row = 0u;
                    row < MATRIX_ROWS;
                    ++row
                ) {
                    uint state_index =
                        outputs[row] * groups + group;
                    uint state = mfq_moe_read_bits(
                        vq_state + state_offset,
                        state_index,
                        state_bits);
                    delta_by_row[row] = mfq_moe_read_bits(
                        vq_aux + aux_offset,
                        state_index,
                        1u);
                    weight_scales[row] =
                        row_anchors[row]
                        * vq_scales[scale_offset + state];
                }

                for (
                    uint local_vector = 0u;
                    local_vector < 3u;
                    ++local_vector
                ) {
                    uint column_base =
                        group * 24u + local_vector * 8u;
                    if (column_base >= uint(K)) {
                        break;
                    }
                    float4 activation0 = float4(
                        float(x[x_offset + column_base]),
                        float(x[x_offset + column_base + 1u]),
                        float(x[x_offset + column_base + 2u]),
                        float(x[x_offset + column_base + 3u]));
                    float4 activation1 = float4(
                        float(x[x_offset + column_base + 4u]),
                        float(x[x_offset + column_base + 5u]),
                        float(x[x_offset + column_base + 6u]),
                        float(x[x_offset + column_base + 7u]));
                    uint vector = column_base >> 3;
                    for (
                        uint row = 0u;
                        row < MATRIX_ROWS;
                        ++row
                    ) {
                        uint index = mfq_moe_read_bits(
                            vq_indices + indices_offset,
                            outputs[row] * vectors + vector,
                            11u);
                        float signed_delta =
                            delta_by_row[row] != 0u
                            ? -delta
                            : delta;
                        uint code_base =
                            codebook_offset + index * 8u;
                        float4 code0 = mfq_moe_load_code4(
                            vq_codebooks, code_base)
                            + float4(signed_delta);
                        float4 code1 = mfq_moe_load_code4(
                            vq_codebooks, code_base + 4u)
                            + float4(signed_delta);
                        float code_dot =
                            dot(activation0, code0)
                            + dot(activation1, code1);
                        accumulators[row] = fma(
                            weight_scales[row],
                            code_dot,
                            accumulators[row]);
                    }
                }
            }
        } else if (
            (uint(VQ_PROFILE_MASK) & 1u) != 0u
        ) {
        for (
            uint group = k_lane;
            group < groups;
            group += K_LANES
        ) {
            uint table_banks[MATRIX_ROWS];
            uint selected_code_banks[MATRIX_ROWS];
            uint delta_values[MATRIX_ROWS];
            float weight_scales[MATRIX_ROWS];
            for (
                uint row = 0u;
                row < MATRIX_ROWS;
                ++row
            ) {
                uint pool_output = outputs[row];
                uint state_index =
                    pool_output * groups + group;
                uint state = mfq_moe_read_bits(
                    vq_state + state_offset,
                    state_index,
                    state_bits);
                uint table_bank =
                    has_table_banks != 0u
                    ? uint(vq_banks[
                          bank_offset
                          + pool_output * supergroups
                          + group / groups_per_super
                      ])
                    : 0u;
                uint delta_value =
                    aux_mode == 3u
                    ? mfq_moe_read_bits(
                          vq_aux + aux_offset,
                          state_index,
                          1u)
                    : 0u;
                uint selected_code_bank = 0u;
                if (code_bank_mode == 1u) {
                    selected_code_bank = uint(
                        vq_state_to_codebank[
                            state_bank_offset + state
                        ]);
                } else if (code_bank_mode == 2u) {
                    selected_code_bank = delta_value;
                }
                table_banks[row] = table_bank;
                selected_code_banks[row] =
                    selected_code_bank;
                delta_values[row] = delta_value;
                weight_scales[row] =
                    row_anchors[row] * vq_scales[
                        scale_offset
                        + table_bank * states + state
                    ];
            }

            for (
                uint local_vector = 0u;
                local_vector < vectors_per_group;
                ++local_vector
            ) {
                uint column_base =
                    group * group_size
                    + local_vector * vector_size;
                if (column_base >= uint(K)) {
                    break;
                }
                float activations[8];
                for (
                    uint component = 0u;
                    component < 8u;
                    ++component
                ) {
                    uint column =
                        column_base + component;
                    activations[component] =
                        component < vector_size
                                && column < uint(K)
                        ? float(x[x_offset + column])
                        : 0.0f;
                }
                uint vector =
                    column_base / vector_size;
                for (
                    uint row = 0u;
                    row < MATRIX_ROWS;
                    ++row
                ) {
                    uint index = mfq_moe_read_bits(
                        vq_indices + indices_offset,
                        outputs[row] * vectors + vector,
                        index_bits);
                    uint aux_value = 0u;
                    if (
                        aux_mode == 1u
                        || aux_mode == 2u
                    ) {
                        aux_value =
                            mfq_moe_read_bits(
                                vq_aux + aux_offset,
                                outputs[row] * signs
                                    + column_base / 8u,
                                7u);
                    }
                    for (
                        uint component = 0u;
                        component < 8u;
                        ++component
                    ) {
                        if (component >= vector_size) {
                            break;
                        }
                        uint column =
                            column_base + component;
                        if (column >= uint(K)) {
                            break;
                        }
                        uint code_offset = (
                            (
                                (
                                    table_banks[row]
                                        * code_banks
                                    + selected_code_banks[
                                        row]
                                )
                                * entries + index
                            )
                            * vector_size + component
                        );
                        float code = float(vq_codebooks[
                            codebook_offset
                            + code_offset
                        ]);
                        if (
                            aux_mode == 1u
                            || aux_mode == 2u
                        ) {
                            uint sign_position =
                                column & 7u;
                            uint negative =
                                sign_position < 7u
                                ? (
                                    (
                                        aux_value
                                        >> sign_position
                                    ) & 1u
                                )
                                : (
                                    popcount(aux_value)
                                    & 1u
                                );
                            if (
                                aux_mode == 2u
                                && sign_position == 7u
                            ) {
                                negative ^=
                                    (index >> 7u) & 1u;
                            }
                            code = negative != 0u
                                ? -code
                                : code;
                        } else if (aux_mode == 3u) {
                            float delta = vq_parameters[
                                parameter_offset
                            ];
                            code +=
                                delta_values[row] != 0u
                                ? -delta
                                : delta;
                        }
                        accumulators[row] = fma(
                            activations[component],
                            weight_scales[row] * code,
                            accumulators[row]);
                    }
                }
            }
        }
        }
    } else if (
        (uint(FAMILY_MASK) & 4u) != 0u
        && family == 2u
    ) {
        uint groups =
            uint(descriptors[descriptor_base + 4u]);
        uint q_offset =
            uint(descriptors[descriptor_base + 5u]);
        uint scale_offset =
            uint(descriptors[descriptor_base + 6u]);
        for (
            uint group = k_lane;
            group < groups;
            group += K_LANES
        ) {
            uint column_base = group * 32u;
            uint outputs[MATRIX_ROWS];
            float scales[MATRIX_ROWS];
            for (
                uint row = 0u;
                row < MATRIX_ROWS;
                ++row
            ) {
                uint output = min(
                    output_base + (
                        uint(FUSED_SWIGLU) != 0u
                            ? (row / ROWS_PER_SIMD) * uint(OUT)
                                + row % ROWS_PER_SIMD
                            : row
                    ),
                    uint(MATRIX_OUT) - 1u);
                uint pool_output =
                    local_expert * uint(MATRIX_OUT) + output;
                outputs[row] = pool_output;
                scales[row] = float(q8_scales[
                    scale_offset
                    + pool_output * groups + group
                ]);
            }
            for (
                uint component = 0u;
                component < 32u;
                ++component
            ) {
                uint column =
                    column_base + component;
                float activation =
                    column < uint(K)
                        ? float(x[x_offset + column])
                        : 0.0f;
                for (
                    uint row = 0u;
                    row < MATRIX_ROWS;
                    ++row
                ) {
                    uint quantized_index = (
                        outputs[row] * groups + group
                    ) * 32u + component;
                    accumulators[row] = fma(
                        activation,
                        scales[row]
                            * float(q8_q[
                                q_offset
                                + quantized_index
                            ]),
                        accumulators[row]);
                }
            }
        }
    } else if (
        (uint(FAMILY_MASK) & 8u) != 0u
        && family == 3u
    ) {
        uint groups =
            uint(descriptors[descriptor_base + 4u]);
        uint value_offset =
            uint(descriptors[descriptor_base + 5u]);
        uint scale_offset =
            uint(descriptors[descriptor_base + 6u]);
        for (
            uint group = k_lane;
            group < groups;
            group += K_LANES
        ) {
            uint column_base = group * 32u;
            uint outputs[MATRIX_ROWS];
            float scales[MATRIX_ROWS];
            for (
                uint row = 0u;
                row < MATRIX_ROWS;
                ++row
            ) {
                uint output = min(
                    output_base + (
                        uint(FUSED_SWIGLU) != 0u
                            ? (row / ROWS_PER_SIMD) * uint(OUT)
                                + row % ROWS_PER_SIMD
                            : row
                    ),
                    uint(MATRIX_OUT) - 1u);
                ulong pool_output =
                    ulong(local_expert) * ulong(MATRIX_OUT) + ulong(output);
                outputs[row] = pool_output;
                scales[row] = mfq_moe_e8m0(mx_scales[
                    ulong(scale_offset)
                    + pool_output * ulong(groups) + ulong(group)
                ]);
            }
            for (
                uint component = 0u;
                component < 32u;
                component += 16u
            ) {
                uint column = column_base + component;
                vec<T, 4> source0 = *reinterpret_cast<
                    device const vec<T, 4>*>(
                        x + x_offset + column);
                vec<T, 4> source1 = *reinterpret_cast<
                    device const vec<T, 4>*>(
                        x + x_offset + column + 4u);
                vec<T, 4> source2 = *reinterpret_cast<
                    device const vec<T, 4>*>(
                        x + x_offset + column + 8u);
                vec<T, 4> source3 = *reinterpret_cast<
                    device const vec<T, 4>*>(
                        x + x_offset + column + 12u);
                float activations[16] = {
                    float(source0.x),
                    float(source0.y),
                    float(source0.z),
                    float(source0.w),
                    float(source1.x),
                    float(source1.y),
                    float(source1.z),
                    float(source1.w),
                    float(source2.x),
                    float(source2.y),
                    float(source2.z),
                    float(source2.w),
                    float(source3.x),
                    float(source3.y),
                    float(source3.z),
                    float(source3.w),
                };
                for (
                    uint row = 0u;
                    row < MATRIX_ROWS;
                    ++row
                ) {
                    ulong packed_offset = ulong(value_offset)
                        + outputs[row] * (ulong(K) >> 1u)
                        + (ulong(column) >> 1u);
                    uint2 packed = *reinterpret_cast<device const uint2*>(
                        mx_values + packed_offset);
                    for (uint pair = 0u; pair < 8u; ++pair) {
                        uint word = pair < 4u ? packed.x : packed.y;
                        uchar codes = uchar(
                            word >> ((pair & 3u) * 8u));
                        accumulators[row] = fma(
                            activations[pair * 2u],
                            scales[row]
                                * mfq_moe_mxfp4_value(codes & 15u),
                            accumulators[row]);
                        accumulators[row] = fma(
                            activations[pair * 2u + 1u],
                            scales[row]
                                * mfq_moe_mxfp4_value(codes >> 4u),
                            accumulators[row]);
                    }
                }
            }
        }
    } else if (
        (uint(FAMILY_MASK) & 16u) != 0u
        && family == 4u
    ) {
        uint groups =
            uint(descriptors[descriptor_base + 4u]);
        uint value_offset =
            uint(descriptors[descriptor_base + 5u]);
        uint scale_offset =
            uint(descriptors[descriptor_base + 6u]);
        for (
            uint group = k_lane;
            group < groups;
            group += K_LANES
        ) {
            uint column_base = group * 128u;
            ulong outputs[MATRIX_ROWS];
            float scales[MATRIX_ROWS];
            for (uint row = 0u; row < MATRIX_ROWS; ++row) {
                uint output = min(
                    output_base + (
                        uint(FUSED_SWIGLU) != 0u
                            ? (row / ROWS_PER_SIMD) * uint(OUT)
                                + row % ROWS_PER_SIMD
                            : row
                    ),
                    uint(MATRIX_OUT) - 1u);
                ulong pool_output =
                    ulong(local_expert) * ulong(MATRIX_OUT) + ulong(output);
                outputs[row] = pool_output;
                scales[row] = mfq_moe_e8m0(mx_scales[
                    ulong(scale_offset)
                    + (pool_output >> 7u) * ulong(groups)
                    + ulong(group)
                ]);
            }
            for (uint component = 0u; component < 128u; ++component) {
                uint column = column_base + component;
                float activation = column < uint(K)
                    ? float(x[x_offset + column])
                    : 0.0f;
                for (uint row = 0u; row < MATRIX_ROWS; ++row) {
                    uchar code = mx_values[
                        ulong(value_offset)
                        + outputs[row] * ulong(K)
                        + ulong(column)];
                    accumulators[row] = fma(
                        activation,
                        scales[row] * mfq_moe_mxfp8_value(code),
                        accumulators[row]);
                }
            }
        }
    } else if (
        (uint(FAMILY_MASK) & 96u) != 0u
        && (family == 5u || family == 6u)
    ) {
        uint value_offset =
            uint(descriptors[descriptor_base + 4u]);
        for (
            uint column = k_lane;
            column < uint(K);
            column += K_LANES
        ) {
            float activation = float(x[x_offset + column]);
            for (uint row = 0u; row < MATRIX_ROWS; ++row) {
                uint output = min(
                    output_base + (
                        uint(FUSED_SWIGLU) != 0u
                            ? (row / ROWS_PER_SIMD) * uint(OUT)
                                + row % ROWS_PER_SIMD
                            : row
                    ),
                    uint(MATRIX_OUT) - 1u);
                ulong pool_output =
                    ulong(local_expert) * ulong(MATRIX_OUT) + ulong(output);
                ulong byte_offset = ulong(value_offset)
                    + (pool_output * ulong(K) + ulong(column)) * 2u;
                ushort raw = *reinterpret_cast<device const ushort*>(
                    q8_q + byte_offset);
                float weight = family == 5u
                    ? as_type<float>(uint(raw) << 16u)
                    : float(as_type<half>(raw));
                accumulators[row] = fma(
                    activation,
                    weight,
                    accumulators[row]);
            }
        }
    }

    if (
        uint(HAS_NEPQ_RESIDUAL) != 0u
        && family == 1u
    ) {
        uint residual_profile =
            uint(descriptors[descriptor_base + 28u]);
        uint position_bits = (residual_profile >> 8u) & 255u;
        uint block_vectors = (residual_profile >> 16u) & 255u;
        if (position_bits != 0u && block_vectors != 0u) {
            uint vectors =
                uint(descriptors[descriptor_base + 7u]);
            uint residual_blocks =
                (vectors + block_vectors - 1u) / block_vectors;
            uint residual_codebook_offset =
                uint(descriptors[descriptor_base + 30u]);
            uint residual_record_offset =
                uint(descriptors[descriptor_base + 31u]);
            uint position_mask = (1u << position_bits) - 1u;
            for (
                uint block = k_lane;
                block < residual_blocks;
                block += K_LANES
            ) {
                for (uint row = 0u; row < MATRIX_ROWS; ++row) {
                    uint output = min(
                        output_base + (
                            uint(FUSED_SWIGLU) != 0u
                                ? (row / ROWS_PER_SIMD) * uint(OUT)
                                    + row % ROWS_PER_SIMD
                                : row
                        ),
                        uint(MATRIX_OUT) - 1u);
                    uint pool_output =
                        local_expert * uint(MATRIX_OUT) + output;
                    uint record_index = residual_record_offset
                        + pool_output * residual_blocks + block;
                    short records[2] = {
                        vq_residual_first[record_index],
                        vq_residual_second[record_index],
                    };
                    for (uint stream = 0u; stream < 2u; ++stream) {
                        int record = int(records[stream]);
                        if (record < 0) {
                            continue;
                        }
                        uint position = uint(record) & position_mask;
                        uint dictionary_id = uint(record) >> position_bits;
                        uint vector = block * block_vectors + position;
                        if (dictionary_id >= 1024u || vector >= vectors) {
                            continue;
                        }
                        uint input_offset = x_offset + vector * 8u;
                        uint dictionary_offset = residual_codebook_offset
                            + dictionary_id * 8u;
                        for (uint component = 0u; component < 8u; ++component) {
                            accumulators[row] = fma(
                                float(x[input_offset + component]),
                                vq_residual_codebooks[
                                    dictionary_offset + component],
                                accumulators[row]);
                        }
                    }
                }
            }
        }
    }

    if (uint(FUSED_SWIGLU) != 0u) {
        for (
            uint row = 0u;
            row < ROWS_PER_SIMD;
            ++row
        ) {
            float gate = accumulators[row];
            float up = accumulators[ROWS_PER_SIMD + row];
            for (
                uint offset = K_LANES >> 1;
                offset > 0u;
                offset >>= 1
            ) {
                gate += simd_shuffle_down(gate, offset);
                up += simd_shuffle_down(up, offset);
            }
            uint output = output_base + row;
            if (k_lane == 0u && output < uint(OUT)) {
                // Match the unfused graph: both projections round to the
                // activation dtype before the elementwise SwiGLU.
                gate = float(T(gate));
                up = float(T(up));
                if (params[0] > 0.0f) {
                    gate = min(gate, params[0]);
                    up = clamp(up, -params[0], params[0]);
                }
                float activated = gate / (1.0f + exp(-gate));
                y[
                    (token * uint(ROUTES) + route)
                        * uint(OUT)
                    + output
                ] = T(activated * up);
            }
        }
    } else {
        for (
            uint row = 0u;
            row < ROWS_PER_SIMD;
            ++row
        ) {
            float total = accumulators[row];
            for (
                uint offset = K_LANES >> 1;
                offset > 0u;
                offset >>= 1
            ) {
                total += simd_shuffle_down(total, offset);
            }
            uint output = output_base + row;
            if (k_lane == 0u && output < uint(OUT)) {
                y[
                    (
                        (
                            token * uint(ROUTES) + route
                        ) * uint(PROJECTIONS)
                        + projection
                    ) * uint(OUT)
                    + output
                ] = T(total);
            }
        }
    }
)METAL";

constexpr const char* kMoeHadamardSource = R"METAL(
    uint row = thread_position_in_grid.x / 256u;
    uint lane = thread_index_in_threadgroup;
    if (row >= uint(M)) {
        return;
    }

    threadgroup float values[BLOCK];
    for (
        uint local_block = 0u;
        local_block < uint(K) / uint(BLOCK);
        ++local_block
    ) {
        uint column_base =
            local_block * uint(BLOCK);
        for (
            uint index = lane;
            index < uint(BLOCK);
            index += 256u
        ) {
            uint column = column_base + index;
            values[index] =
                float(x[row * uint(K) + column])
                * float(signs[column]);
        }
        threadgroup_barrier(
            mem_flags::mem_threadgroup);
        for (
            uint stride = 1u;
            stride < uint(BLOCK);
            stride <<= 1u
        ) {
            for (
                uint pair = lane;
                pair < uint(BLOCK) / 2u;
                pair += 256u
            ) {
                uint pair_block = pair / stride;
                uint within =
                    pair - pair_block * stride;
                uint first =
                    pair_block * (stride << 1u)
                    + within;
                uint second = first + stride;
                float first_value = values[first];
                float second_value = values[second];
                values[first] =
                    first_value + second_value;
                values[second] =
                    first_value - second_value;
            }
            threadgroup_barrier(
                mem_flags::mem_threadgroup);
        }
        float inverse = rsqrt(float(BLOCK));
        for (
            uint index = lane;
            index < uint(BLOCK);
            index += 256u
        ) {
            uint column = column_base + index;
            y[row * uint(K) + column] =
                T(values[index] * inverse);
        }
        threadgroup_barrier(
            mem_flags::mem_threadgroup);
    }
)METAL";

class BlobCursor {
public:
    explicit BlobCursor(
        std::span<const std::uint8_t> blob)
        : blob_(blob) {}

    template <typename T>
    T scalar(const char* name) {
        require(sizeof(T), name);
        T value{};
        std::memcpy(
            &value,
            blob_.data() + offset_,
            sizeof(T));
        offset_ += sizeof(T);
        return value;
    }

    std::span<const std::uint8_t> bytes(
        std::size_t count,
        const char* name) {
        require(count, name);
        auto result = blob_.subspan(offset_, count);
        offset_ += count;
        return result;
    }

    std::size_t remaining() const noexcept {
        return blob_.size() - offset_;
    }

private:
    void require(
        std::size_t count,
        const char* name) const {
        if (count > blob_.size() - offset_) {
            throw std::runtime_error(
                std::string("truncated MFE ") + name);
        }
    }

    std::span<const std::uint8_t> blob_;
    std::size_t offset_ = 0;
};

std::size_t checked_size(
    std::uint64_t value,
    const char* name) {
    if (
        value
        > static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max())
    ) {
        throw std::runtime_error(
            std::string("MFE ") + name
            + " exceeds addressable memory");
    }
    return static_cast<std::size_t>(value);
}

std::size_t checked_product(
    std::size_t left,
    std::size_t right,
    const char* name) {
    if (
        right != 0
        && left
            > std::numeric_limits<std::size_t>::max()
                / right
    ) {
        throw std::runtime_error(
            std::string("MFE ") + name
            + " overflows");
    }
    return left * right;
}

std::size_t checked_add(
    std::size_t left,
    std::size_t right,
    const char* name) {
    if (
        right
        > std::numeric_limits<std::size_t>::max()
            - left
    ) {
        throw std::runtime_error(
            std::string("MFE ") + name
            + " overflows");
    }
    return left + right;
}

std::size_t checked_packed_size(
    std::size_t count,
    int bits,
    const char* name) {
    const auto bit_count = checked_product(
        count,
        static_cast<std::size_t>(bits),
        name);
    return checked_add(bit_count, 7, name) / 8;
}

std::int32_t checked_int(
    std::size_t value,
    const char* name) {
    if (
        value
        > static_cast<std::size_t>(
            std::numeric_limits<std::int32_t>::max())
    ) {
        throw std::runtime_error(
            std::string("MFE ") + name
            + " exceeds int32 range");
    }
    return static_cast<std::int32_t>(value);
}

constexpr std::int32_t descriptor_u32_bits(
    std::uint32_t value) noexcept {
    return std::bit_cast<std::int32_t>(value);
}

constexpr std::uint32_t descriptor_u32_value(
    std::int32_t value) noexcept {
    return std::bit_cast<std::uint32_t>(value);
}

static_assert(
    descriptor_u32_value(
        descriptor_u32_bits(std::uint32_t{0x80000000u}))
    == std::uint32_t{0x80000000u});
static_assert(
    descriptor_u32_value(
        descriptor_u32_bits(std::uint32_t{0xffffffffu}))
    == std::uint32_t{0xffffffffu});

std::int32_t checked_descriptor_u32(
    std::size_t value,
    const char* name) {
    if (
        value
        > static_cast<std::size_t>(
            std::numeric_limits<std::uint32_t>::max())
    ) {
        throw std::runtime_error(
            std::string("MFE ") + name
            + " exceeds uint32 descriptor range");
    }
    return descriptor_u32_bits(
        static_cast<std::uint32_t>(value));
}

int checked_positive(
    std::uint32_t value,
    const char* name) {
    if (
        value == 0
        || value
            > static_cast<std::uint32_t>(
                std::numeric_limits<std::int32_t>::max())
    ) {
        throw std::runtime_error(
            std::string("invalid MFE ") + name);
    }
    return static_cast<int>(value);
}

template <typename Allocator>
void append_raw(
    std::vector<std::uint8_t, Allocator>& target,
    const array& source,
    Dtype expected,
    const char* name) {
    if (
        source.dtype() != expected
        || !source.flags().row_contiguous
    ) {
        throw std::runtime_error(
            std::string("invalid MFE packed ") + name);
    }
    auto evaluated = source;
    evaluated.eval();
    if (
        evaluated.nbytes()
        > std::numeric_limits<std::size_t>::max()
            - target.size()
    ) {
        throw std::runtime_error(
            "MFE packed stream size overflows");
    }
    const auto previous = target.size();
    target.resize(previous + evaluated.nbytes());
    std::memcpy(
        target.data() + previous,
        evaluated.data<std::uint8_t>(),
        evaluated.nbytes());
}

template <typename Allocator>
void append_bytes(
    std::vector<std::uint8_t, Allocator>& target,
    std::span<const std::uint8_t> source,
    const char* name) {
    if (
        source.size()
        > std::numeric_limits<std::size_t>::max()
            - target.size()
    ) {
        throw std::runtime_error(
            std::string("MFE packed ") + name
            + " stream size overflows");
    }
    target.insert(
        target.end(),
        source.begin(),
        source.end());
}

template <typename Allocator>
array make_raw_array(
    std::vector<std::uint8_t, Allocator> bytes,
    Dtype dtype) {
    if (bytes.empty()) {
        bytes.resize(dtype.size(), 0);
    }
    if (bytes.size() % dtype.size() != 0) {
        throw std::runtime_error(
            "MFE packed stream is misaligned");
    }
    const auto layout = detail::packed_storage_layout(
        bytes.size() / dtype.size());
    const Shape shape = layout.is_matrix()
        ? Shape{layout.rows, layout.columns}
        : Shape{layout.columns};
    auto result = array(
        mlx::core::allocator::malloc(bytes.size()),
        shape,
        dtype);
    std::memcpy(
        result.data<std::uint8_t>(),
        bytes.data(),
        bytes.size());
    return result;
}

array make_int32_array(
    const std::vector<std::int32_t>& values,
    Shape shape) {
    return array(values.begin(), std::move(shape));
}

struct NativeMoeConfig {
    Dtype dtype;
    Shape output_shape;
    int tokens = 0;
    int routes = 0;
    int experts = 0;
    int output_width = 0;
    int matrix_output_width = 0;
    int projections = 0;
    int fused_swiglu = 0;
    int input_width = 0;
    int k_lanes = 0;
    int rows_per_simd = 1;
    int descriptor_size = 0;
    int variant_stride = 0;
    int shared_input = 0;
    int jsc_execution_layout = 0;
    int family_mask = 0;
    int vq_profile_mask = 0;
    int has_nepq_residual = 0;
    int npq_grouped_indices = 0;
    int sorted_routes = 0;
    int expert_map_size = 0;
    int packed_expert_ids = 0;
    int workgroups = 0;
};

struct Mxfp4DecodeReduceConfig {
    Dtype dtype;
    Shape output_shape;
    int tokens = 0;
    int routes = 0;
    int experts = 0;
    int output_width = 0;
    int input_width = 0;
    int descriptor_size = 0;
    int rows_per_lane_group = 2;
    int workgroups = 0;
};

struct Mxfp4PairSwiGluConfig {
    Dtype dtype;
    Shape output_shape;
    int tokens = 0;
    int routes = 0;
    int experts = 0;
    int output_width = 0;
    int input_width = 0;
    int descriptor_size = 0;
    int workgroups = 0;
};

struct GroupedMmqConfig {
    Shape output_shape;
    int route_count = 0;
    int max_blocks = 0;
    int block_rows = 32;
    int tile_columns = 64;
    int tokens = 0;
    int routes = 0;
    int experts = 0;
    int output_width = 0;
    int matrix_output_width = 0;
    int projections = 1;
    int input_width = 0;
    int descriptor_size = 0;
    int variant_stride = 0;
    int shared_input = 0;
    int input_sorted = 0;
    int fused_swiglu = 0;
    int has_nepq_residual = 0;
    int family_mask = 127;
    int vq_profile_mask = 0;
    bool use_nax = false;
    bool direct_nax = false;
    float swiglu_limit = 0.0f;
};

struct GroupedMmqParameters {
    int route_count = 0;
    int tokens = 0;
    int routes = 0;
    int experts = 0;
    int output_width = 0;
    int matrix_output_width = 0;
    int projections = 1;
    int input_width = 0;
    int descriptor_size = 0;
    int variant_stride = 0;
    int shared_input = 0;
    int input_sorted = 0;
    float swiglu_limit = 0.0f;
};

static_assert(sizeof(GroupedMmqParameters) == 52);

struct Mxfp4BlocksConfig {
    Shape output_shape;
    int route_count = 0;
    int max_blocks = 0;
    int experts = 0;
    int output_width = 0;
    int input_width = 0;
    int block_rows = 32;
    bool paired = false;
};

// Adapted from oMLX's DeepSeek-V4 block-list builder.  The expert IDs have
// already been sorted, so one GPU thread can find each expert's contiguous
// range and split it into independently schedulable BM-row blocks.
const mlx::core::fast::CustomKernelFunction&
grouped_mmq_block_builder(bool cohort_ordered) {
    const auto make_builder = [](bool ordered) {
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        std::vector<std::string> inputs{"indices"};
        if (ordered) {
            inputs.emplace_back("expert_order");
        }
        std::string source = ordered
            ? "#define MFQ_SCHEDULED_EXPERT(index) expert_order[index]\n"
            : "#define MFQ_SCHEDULED_EXPERT(index) int(index)\n";
        source +=
            R"METAL(
                const uint schedule_index = thread_index_in_threadgroup;
                const uint expert = uint(
                    MFQ_SCHEDULED_EXPERT(schedule_index));

                threadgroup atomic_int local_count;
                threadgroup int expert_starts[NUM_EXPERTS];
                threadgroup int expert_rows[NUM_EXPERTS];
                threadgroup int use_block_chunks;
                if constexpr (BLOCK_CHUNK == 0) {
                    if (expert == 0) {
                        atomic_store_explicit(
                            &local_count,
                            0,
                            memory_order_relaxed);
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }

                int lo = 0;
                int hi = M;
                while (lo < hi) {
                    int mid = (lo + hi) >> 1;
                    if (indices[mid] < int(expert)) {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }
                const int start = lo;

                hi = M;
                while (lo < hi) {
                    int mid = (lo + hi) >> 1;
                    if (indices[mid] <= int(expert)) {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }
                const int end = lo;

                if constexpr (BLOCK_CHUNK == 0) {
                    for (int row = start; row < end; row += BM) {
                        const int rows = min(BM, end - row);
                        const int slot = atomic_fetch_add_explicit(
                            &local_count,
                            1,
                            memory_order_relaxed);
                        if (slot < MAX_BLOCKS) {
                            block_meta[slot * 3 + 0] = row;
                            block_meta[slot * 3 + 1] = int(expert);
                            block_meta[slot * 3 + 2] = rows;
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    if (expert == 0) {
                        block_count[0] = atomic_load_explicit(
                            &local_count,
                            memory_order_relaxed);
                    }
                } else {
                    expert_starts[schedule_index] = start;
                    expert_rows[schedule_index] = end - start;
                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    if (expert == 0) {
                        int active_experts = 0;
                        for (uint item = 0; item < NUM_EXPERTS; ++item) {
                            active_experts += expert_rows[item] > 0 ? 1 : 0;
                        }
                        // A heavily skewed route set already reuses a small
                        // working set and benefits from maximum interleave.
                        // Chunk only when at least half the pool is active.
                        use_block_chunks =
                            active_experts * 2 >= int(NUM_EXPERTS) ? 1 : 0;
                        if (use_block_chunks == 0) {
                            atomic_store_explicit(
                                &local_count,
                                0,
                                memory_order_relaxed);
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    if (use_block_chunks == 0) {
                        for (int row = start; row < end; row += BM) {
                            const int rows = min(BM, end - row);
                            const int slot = atomic_fetch_add_explicit(
                                &local_count,
                                1,
                                memory_order_relaxed);
                            if (slot < MAX_BLOCKS) {
                                block_meta[slot * 3 + 0] = row;
                                block_meta[slot * 3 + 1] = int(expert);
                                block_meta[slot * 3 + 2] = rows;
                            }
                        }
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                        if (expert == 0) {
                            block_count[0] = atomic_load_explicit(
                                &local_count,
                                memory_order_relaxed);
                        }
                    } else if (expert == 0) {
                        int maximum_blocks = 0;
                        for (uint item = 0; item < NUM_EXPERTS; ++item) {
                            maximum_blocks = max(
                                maximum_blocks,
                                (expert_rows[item] + BM - 1) / BM);
                        }
                        int slot = 0;
                        for (int first = 0;
                             first < maximum_blocks;
                             first += BLOCK_CHUNK) {
                            for (uint item = 0; item < NUM_EXPERTS; ++item) {
                                const int scheduled_expert = int(
                                    MFQ_SCHEDULED_EXPERT(item));
                                int count =
                                    (expert_rows[item] + BM - 1) / BM;
                                for (int local = 0;
                                     local < BLOCK_CHUNK
                                         && first + local < count;
                                     ++local) {
                                    int row = expert_starts[item]
                                        + (first + local) * BM;
                                    if (slot < MAX_BLOCKS) {
                                        block_meta[slot * 3 + 0] = row;
                                        block_meta[slot * 3 + 1] =
                                            scheduled_expert;
                                        block_meta[slot * 3 + 2] = min(
                                            BM,
                                            expert_starts[item]
                                                + expert_rows[item] - row);
                                    }
                                    ++slot;
                                }
                            }
                        }
                        block_count[0] = slot;
                    }
                }
            )METAL";
        return mlx::core::fast::metal_kernel(
            ordered
                ? "mfq_grouped_mmq_block_builder_cohort"
                : "mfq_grouped_mmq_block_builder",
            std::move(inputs),
            {"block_meta", "block_count"},
            std::move(source),
            "",
            true,
            false,
            options);
    };
    static const auto unordered_builder = make_builder(false);
    static const auto ordered_builder = make_builder(true);
    return cohort_ordered ? ordered_builder : unordered_builder;
}

MlxGroupedMmqPlan make_grouped_mmq_plan(
    const array& expert_ids,
    const array& route_order_value,
    const array& expert_order,
    int experts,
    int block_rows = 32) {
    if (block_rows != 32 && block_rows != 48 && block_rows != 64
        && block_rows != 80 && block_rows != 96) {
        throw std::invalid_argument(
            "grouped MMQ block rows must be 32, 48, 64, 80, or 96");
    }
    if (experts <= 0 || experts > 1024) {
        throw std::invalid_argument(
            "grouped MMQ block builder requires 1..1024 experts; received " +
            std::to_string(experts));
    }
    if (
        expert_order.dtype() != mlx::core::int32
        || expert_order.ndim() != 1
        || expert_order.size() != static_cast<std::size_t>(experts)
    ) {
        throw std::invalid_argument(
            "grouped MMQ expert order must contain every expert");
    }
    auto ids = mlx::core::contiguous(
        mlx::core::astype(expert_ids, mlx::core::int32));
    if (ids.ndim() != 2) {
        throw std::invalid_argument(
            "routed expert IDs must have [tokens,routes] shape");
    }
    const int route_count = checked_int(
        checked_product(
            static_cast<std::size_t>(ids.shape(0)),
            static_cast<std::size_t>(ids.shape(1)),
            "route count"),
        "route count");
    auto route_order = mlx::core::contiguous(
        mlx::core::astype(route_order_value, mlx::core::int32));
    if (
        route_order.ndim() != 1
        || route_order.size()
            != static_cast<std::size_t>(route_count)
    ) {
        throw std::invalid_argument(
            "route order must contain one index per routed row");
    }
    const int max_blocks = checked_int(
        static_cast<std::size_t>(
            (route_count + block_rows - 1) / block_rows)
            + static_cast<std::size_t>(experts),
        "grouped MMQ max blocks");
    auto sorted_ids = mlx::core::contiguous(
        mlx::core::take(
            mlx::core::reshape(ids, Shape{route_count}),
            route_order,
            0));
    const bool cohort_ordered = experts >= 128
        && route_count > experts * 96;
    const int block_chunk = cohort_ordered ? 2 : 0;
    std::vector<array> builder_inputs{
        std::move(sorted_ids),
    };
    if (cohort_ordered) {
        builder_inputs.push_back(expert_order);
    }
    auto outputs = grouped_mmq_block_builder(cohort_ordered)(
        std::move(builder_inputs),
        {
            Shape{max_blocks, 3},
            Shape{1},
        },
        {
            mlx::core::int32,
            mlx::core::int32,
        },
        {experts, 1, 1},
        {experts, 1, 1},
        {
            {"NUM_EXPERTS", experts},
            {"BM", block_rows},
            {"M", route_count},
            {"MAX_BLOCKS", max_blocks},
            {
                "BLOCK_CHUNK",
                // Above 96 mean routes, pair adjacent blocks in deterministic
                // cohort order to retain packed-weight locality without
                // serializing an entire expert. Smaller pools and route sets
                // keep the original parallel atomic path.
                block_chunk,
            },
        },
        std::nullopt,
        false,
        {});
    return MlxGroupedMmqPlan{
        .block_meta = std::move(outputs.at(0)),
        .block_count = std::move(outputs.at(1)),
        .max_blocks = max_blocks,
        .block_rows = block_rows,
        .route_count = route_count,
        .experts = experts,
    };
}

std::string native_moe_kernel_name(
    const NativeMoeConfig& config) {
    std::ostringstream name;
    name
        << "mfq_native_mfe_v3_"
        << (config.dtype == mlx::core::float16 ? "f16" : "f32")
        << "_t" << config.tokens
        << "_r" << config.routes
        << "_e" << config.experts
        << "_o" << config.output_width
        << "_mo" << config.matrix_output_width
        << "_p" << config.projections
        << "_sw" << config.fused_swiglu
        << "_k" << config.input_width
        << "_kl" << config.k_lanes
        << "_rs" << config.rows_per_simd
        << "_vs" << config.variant_stride
        << "_si" << config.shared_input
        << "_jl" << config.jsc_execution_layout
        << "_fm" << config.family_mask
        << "_vm" << config.vq_profile_mask
        << "_nr" << config.has_nepq_residual
        << "_ni" << config.npq_grouped_indices;
    name << "_sr" << config.sorted_routes;
    name << "_em" << config.expert_map_size;
    name << "_pe" << config.packed_expert_ids;
    return name.str();
}

std::string make_native_moe_source(
    const NativeMoeConfig& config,
    const std::string& kernel_name) {
    const bool has_expert_map = config.expert_map_size != 0;
    std::ostringstream source;
    source
        << "#include <metal_stdlib>\n"
        << "using namespace metal;\n"
        << "using T = "
        << (config.dtype == mlx::core::float16 ? "half" : "float")
        << ";\n"
        << kMoeHeader
        << "\nkernel void " << kernel_name << "(\n"
        << "device const int* descriptors [[buffer(0)]],\n"
        << "device const uchar* nint_q [[buffer(1)]],\n"
        << "device const uchar* nint_sub_scale [[buffer(2)]],\n"
        << "device const uchar* nint_sub_min [[buffer(3)]],\n"
        << "device const float* nint_anchor_scale [[buffer(4)]],\n"
        << "device const float* nint_anchor_min [[buffer(5)]],\n"
        << "device const int8_t* q8_q [[buffer(6)]],\n"
        << "device const half* q8_scales [[buffer(7)]],\n"
        << "device const uchar* vq_indices [[buffer(8)]],\n"
        << "device const uchar* vq_state [[buffer(9)]],\n"
        << "device const uchar* vq_aux [[buffer(10)]],\n"
        << "device const float* vq_anchors [[buffer(11)]],\n"
        << "device const int8_t* vq_codebooks [[buffer(12)]],\n"
        << "device const float* vq_scales [[buffer(13)]],\n"
        << "device const uchar* vq_state_to_codebank [[buffer(14)]],\n"
        << "device const uchar* vq_banks [[buffer(15)]],\n"
        << "device const float* vq_parameters [[buffer(16)]],\n"
        << "device const float* vq_residual_codebooks [[buffer(17)]],\n"
        << "device const short* vq_residual_first [[buffer(18)]],\n"
        << "device const short* vq_residual_second [[buffer(19)]],\n"
        << "device const uchar* mx_values [[buffer(20)]],\n"
        << "device const uchar* mx_scales [[buffer(21)]],\n"
        << "device const T* x [[buffer(22)]],\n"
        << "device const int* expert_ids [[buffer(23)]],\n"
        << "device const int* route_order [[buffer(24)]],\n"
        << "device const float* params [[buffer(25)]],\n";
    if (has_expert_map) {
        source << "device const int* expert_map [[buffer(26)]],\n";
    }
    source
        << "device T* y [[buffer("
        << (has_expert_map ? 27 : 26)
        << ")]],\n"
        << "uint thread_index_in_simdgroup "
           "[[thread_index_in_simdgroup]],\n"
        << "uint simdgroup_index_in_threadgroup "
           "[[simdgroup_index_in_threadgroup]],\n"
        << "uint3 threadgroup_position_in_grid "
           "[[threadgroup_position_in_grid]]) {\n"
        << "constexpr int TOKENS = " << config.tokens << ";\n"
        << "constexpr int ROUTES = " << config.routes << ";\n"
        << "constexpr int EXPERTS = " << config.experts << ";\n"
        << "constexpr int OUT = " << config.output_width << ";\n"
        << "constexpr int MATRIX_OUT = "
        << config.matrix_output_width << ";\n"
        << "constexpr int PROJECTIONS = "
        << config.projections << ";\n"
        << "constexpr int FUSED_SWIGLU = "
        << config.fused_swiglu << ";\n"
        << "constexpr int K = " << config.input_width << ";\n"
        << "constexpr int K_LANES_VALUE = "
        << config.k_lanes << ";\n"
        << "constexpr int ROWS_PER_SIMD_VALUE = "
        << config.rows_per_simd << ";\n"
        << "constexpr int DESCRIPTOR_SIZE = "
        << config.descriptor_size << ";\n"
        << "constexpr int VARIANT_STRIDE = "
        << config.variant_stride << ";\n"
        << "constexpr int SHARED_INPUT = "
        << config.shared_input << ";\n"
        << "constexpr int JSC_EXECUTION_LAYOUT = "
        << config.jsc_execution_layout << ";\n"
        << "constexpr int FAMILY_MASK = "
        << config.family_mask << ";\n"
        << "constexpr int VQ_PROFILE_MASK = "
        << config.vq_profile_mask << ";\n"
        << "constexpr int HAS_NEPQ_RESIDUAL = "
        << config.has_nepq_residual << ";\n"
        << "constexpr int NPQ_GROUPED_INDICES = "
        << config.npq_grouped_indices << ";\n"
        << "constexpr int SORTED_ROUTES = "
        << config.sorted_routes << ";\n"
        << "constexpr int EXPERT_MAP_SIZE = "
        << config.expert_map_size << ";\n"
        << "constexpr int PACKED_EXPERT_IDS = "
        << config.packed_expert_ids << ";\n";
    if (!has_expert_map) {
        source << "device const int* expert_map = expert_ids;\n";
    }
    source
        << kMoeSource
        << "\n}\n";
    return source.str();
}

class NativeMfePrimitive final
    : public mlx::core::UnaryPrimitive {
public:
    NativeMfePrimitive(
        mlx::core::Stream stream,
        NativeMoeConfig config)
        : UnaryPrimitive(stream),
          config_(std::move(config)),
          kernel_name_(native_moe_kernel_name(config_)) {}

    void eval_cpu(
        const std::vector<array>&,
        array&) override {
        throw std::runtime_error(
            "native MFE primitive has no CPU path");
    }

    void eval_gpu(
        const std::vector<array>& inputs,
        array& output) override {
        const int input_count =
            config_.expert_map_size != 0 ? 27 : 26;
        if (inputs.size() != static_cast<std::size_t>(input_count)) {
            throw std::logic_error(
                "native MFE primitive input count mismatch");
        }
        output.set_data(
            mlx::core::allocator::malloc(output.nbytes()));
        auto& selected_stream = stream();
        auto& device = mlx::core::metal::device(
            selected_stream.device);
        CompileOptions compile_options;
        compile_options.math_mode = MathMode::Fast;
        auto* library = device.get_library(
            kernel_name_,
            compile_options,
            [config = config_, name = kernel_name_] {
                return make_native_moe_source(
                    config,
                    name);
            });
        auto* kernel = device.get_kernel(
            kernel_name_,
            library);
        auto& encoder =
            mlx::core::metal::get_command_encoder(
                selected_stream);
        encoder.set_compute_pipeline_state(kernel);
        for (int index = 0; index < input_count; ++index) {
            encoder.set_input_array(
                inputs[static_cast<std::size_t>(index)],
                index);
        }
        encoder.set_output_array(output, input_count);
        encoder.dispatch_threadgroups(
            MTL::Size(config_.workgroups, 1, 1),
            MTL::Size(64, 1, 1));
    }

    const char* name() const override {
        return "NativeMfePrimitive";
    }

    bool is_equivalent(
        const mlx::core::Primitive& other) const override {
        const auto* primitive =
            dynamic_cast<const NativeMfePrimitive*>(&other);
        return primitive != nullptr
            && primitive->kernel_name_ == kernel_name_;
    }

    std::vector<Shape> output_shapes(
        const std::vector<array>&) override {
        return {config_.output_shape};
    }

private:
    NativeMoeConfig config_;
    std::string kernel_name_;
};

array native_moe_dispatch(
    std::vector<array> inputs,
    NativeMoeConfig config) {
    auto stream = mlx::core::default_stream(
        mlx::core::default_device());
    if (stream.device != mlx::core::Device::gpu) {
        throw std::invalid_argument(
            "native MFE primitive requires the Metal device");
    }
    auto shape = config.output_shape;
    auto dtype = config.dtype;
    return array(
        std::move(shape),
        dtype,
        std::make_shared<NativeMfePrimitive>(
            stream,
            std::move(config)),
        std::move(inputs));
}

std::string mxfp4_decode_reduce_kernel_name(
    const Mxfp4DecodeReduceConfig& config) {
    std::ostringstream name;
    name
        << "mfq_mxfp4_decode_down_reduce_"
        << (config.dtype == mlx::core::float16 ? "f16" : "f32")
        << "_t" << config.tokens
        << "_r" << config.routes
        << "_e" << config.experts
        << "_o" << config.output_width
        << "_k" << config.input_width
        << "_rr" << config.rows_per_lane_group;
    return name.str();
}

std::string make_mxfp4_decode_reduce_source(
    const Mxfp4DecodeReduceConfig& config,
    const std::string& kernel_name) {
    std::ostringstream source;
    source
        << "#include <metal_stdlib>\n"
        << "using namespace metal;\n"
        << "using T = "
        << (config.dtype == mlx::core::float16 ? "half" : "float")
        << ";\n"
        << "constant constexpr float MXFP4_LUT[16] = {\n"
           "    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,\n"
           "    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,\n"
           "};\n"
        << "kernel void " << kernel_name << "(\n"
        << "device const int* descriptors [[buffer(0)]],\n"
        << "device const uchar* mx_values [[buffer(1)]],\n"
        << "device const uchar* mx_scales [[buffer(2)]],\n"
        << "device const T* x [[buffer(3)]],\n"
        << "device const int* expert_ids [[buffer(4)]],\n"
        << "device const float* route_weights [[buffer(5)]],\n"
        << "device T* y [[buffer(6)]],\n"
        << "uint thread_index_in_simdgroup "
           "[[thread_index_in_simdgroup]],\n"
        << "uint simdgroup_index_in_threadgroup "
           "[[simdgroup_index_in_threadgroup]],\n"
        << "uint3 threadgroup_position_in_grid "
           "[[threadgroup_position_in_grid]]) {\n"
        << "constexpr uint TOKENS = " << config.tokens << "u;\n"
        << "constexpr uint ROUTES = " << config.routes << "u;\n"
        << "constexpr uint EXPERTS = " << config.experts << "u;\n"
        << "constexpr uint OUT = " << config.output_width << "u;\n"
        << "constexpr uint K = " << config.input_width << "u;\n"
        << "constexpr uint DESCRIPTOR_SIZE = "
        << config.descriptor_size << "u;\n"
        << "constexpr uint ROWS_PER_LANE_GROUP = "
        << config.rows_per_lane_group << "u;\n"
        << R"METAL(
    constexpr uint K_LANES = 8u;
    constexpr uint LANE_GROUPS = 32u / K_LANES;
    constexpr uint OUTPUTS_PER_TG = LANE_GROUPS * ROWS_PER_LANE_GROUP;
    threadgroup float route_outputs[ROUTES * OUTPUTS_PER_TG];

    const uint lane = thread_index_in_simdgroup;
    const uint k_lane = lane & (K_LANES - 1u);
    const uint lane_group = lane / K_LANES;
    const uint route = simdgroup_index_in_threadgroup;
    const uint output_tiles = (OUT + OUTPUTS_PER_TG - 1u)
        / OUTPUTS_PER_TG;
    const uint token = threadgroup_position_in_grid.x / output_tiles;
    const uint output_tile = threadgroup_position_in_grid.x
        - token * output_tiles;
    if (token >= TOKENS) return;
    const uint output_base = output_tile * OUTPUTS_PER_TG
        + lane_group * ROWS_PER_LANE_GROUP;
    uint bounded_outputs[ROWS_PER_LANE_GROUP];
    float accumulators[ROWS_PER_LANE_GROUP] = {0.0f};
    for (uint row = 0u; row < ROWS_PER_LANE_GROUP; ++row) {
        bounded_outputs[row] = min(output_base + row, OUT - 1u);
    }

    const int expert = expert_ids[token * ROUTES + route];
    if (expert >= 0 && expert < int(EXPERTS)) {
        const uint descriptor_base = uint(expert) * DESCRIPTOR_SIZE;
        // This primitive is selected only for a pure MXFP4 projection.
        const uint local_expert = uint(descriptors[descriptor_base + 1u]);
        const uint groups = uint(descriptors[descriptor_base + 4u]);
        const uint value_offset = uint(descriptors[descriptor_base + 5u]);
        const uint scale_offset = uint(descriptors[descriptor_base + 6u]);

        for (uint group = k_lane; group < groups; group += K_LANES) {
            const uint column_base = group * 32u;
            ulong pool_outputs[ROWS_PER_LANE_GROUP];
            float scales[ROWS_PER_LANE_GROUP];
            for (uint row = 0u; row < ROWS_PER_LANE_GROUP; ++row) {
                pool_outputs[row] = ulong(local_expert) * ulong(OUT)
                    + ulong(bounded_outputs[row]);
                const uchar raw_scale = mx_scales[
                    ulong(scale_offset)
                        + pool_outputs[row] * ulong(groups)
                        + ulong(group)];
                const uint scale_bits = raw_scale == 0u
                    ? 0x00400000u
                    : uint(raw_scale) << 23u;
                scales[row] = raw_scale == 255u
                    ? NAN
                    : as_type<float>(scale_bits);
            }

            for (uint component = 0u; component < 32u; component += 16u) {
                const uint column = column_base + component;
                const vec<T, 4> source0 = *reinterpret_cast<
                    device const vec<T, 4>*>(
                        x + (token * ROUTES + route) * K + column);
                const vec<T, 4> source1 = *reinterpret_cast<
                    device const vec<T, 4>*>(
                        x + (token * ROUTES + route) * K + column + 4u);
                const vec<T, 4> source2 = *reinterpret_cast<
                    device const vec<T, 4>*>(
                        x + (token * ROUTES + route) * K + column + 8u);
                const vec<T, 4> source3 = *reinterpret_cast<
                    device const vec<T, 4>*>(
                        x + (token * ROUTES + route) * K + column + 12u);
                const float activations[16] = {
                    float(source0.x), float(source0.y),
                    float(source0.z), float(source0.w),
                    float(source1.x), float(source1.y),
                    float(source1.z), float(source1.w),
                    float(source2.x), float(source2.y),
                    float(source2.z), float(source2.w),
                    float(source3.x), float(source3.y),
                    float(source3.z), float(source3.w),
                };
                for (uint row = 0u; row < ROWS_PER_LANE_GROUP; ++row) {
                    const ulong packed_offset = ulong(value_offset)
                        + pool_outputs[row] * (ulong(K) >> 1u)
                        + (ulong(column) >> 1u);
                    const uint2 packed = *reinterpret_cast<
                        device const uint2*>(mx_values + packed_offset);
                    for (uint pair = 0u; pair < 8u; ++pair) {
                        const uint word = pair < 4u ? packed.x : packed.y;
                        const uchar codes = uchar(
                            word >> ((pair & 3u) * 8u));
                        accumulators[row] = fma(
                            activations[pair * 2u],
                            scales[row] * MXFP4_LUT[uint(codes & 15u)],
                            accumulators[row]);
                        accumulators[row] = fma(
                            activations[pair * 2u + 1u],
                            scales[row] * MXFP4_LUT[uint(codes >> 4u)],
                            accumulators[row]);
                    }
                }
            }
        }
        for (uint offset = K_LANES >> 1u; offset > 0u; offset >>= 1u) {
            for (uint row = 0u; row < ROWS_PER_LANE_GROUP; ++row) {
                accumulators[row] += simd_shuffle_down(
                    accumulators[row], offset);
            }
        }
    }
    if (k_lane == 0u) {
        for (uint row = 0u; row < ROWS_PER_LANE_GROUP; ++row) {
            // Preserve the original two-dispatch contract: each down result
            // first rounds to the activation dtype before route weighting.
            const float down = expert >= 0 && expert < int(EXPERTS)
                ? float(T(accumulators[row]))
                : 0.0f;
            route_outputs[
                route * OUTPUTS_PER_TG
                    + lane_group * ROWS_PER_LANE_GROUP + row
            ] = down * route_weights[token * ROUTES + route];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // One SIMD group performs the tiny deterministic Top-6 reduction while
    // the other route groups finish together at the same barrier.
    if (simdgroup_index_in_threadgroup == 0u && lane < OUTPUTS_PER_TG) {
        const uint output = output_tile * OUTPUTS_PER_TG + lane;
        float total = 0.0f;
        for (uint selected = 0u; selected < ROUTES; ++selected) {
            total += route_outputs[selected * OUTPUTS_PER_TG + lane];
        }
        if (output < OUT) {
            y[token * OUT + output] = T(total);
        }
    }
}
)METAL";
    return source.str();
}

class Mxfp4DecodeReducePrimitive final
    : public mlx::core::UnaryPrimitive {
public:
    Mxfp4DecodeReducePrimitive(
        mlx::core::Stream stream,
        Mxfp4DecodeReduceConfig config)
        : UnaryPrimitive(stream),
          config_(std::move(config)),
          kernel_name_(mxfp4_decode_reduce_kernel_name(config_)) {}

    void eval_cpu(
        const std::vector<array>&,
        array&) override {
        throw std::runtime_error(
            "MXFP4 decode down-reduce has no CPU path");
    }

    void eval_gpu(
        const std::vector<array>& inputs,
        array& output) override {
        if (inputs.size() != 6) {
            throw std::logic_error(
                "MXFP4 decode down-reduce input count mismatch");
        }
        output.set_data(
            mlx::core::allocator::malloc(output.nbytes()));
        auto& selected_stream = stream();
        auto& device = mlx::core::metal::device(
            selected_stream.device);
        CompileOptions compile_options;
        compile_options.math_mode = MathMode::Fast;
        auto* library = device.get_library(
            kernel_name_,
            compile_options,
            [config = config_, name = kernel_name_] {
                return make_mxfp4_decode_reduce_source(config, name);
            });
        auto* kernel = device.get_kernel(kernel_name_, library);
        auto& encoder = mlx::core::metal::get_command_encoder(
            selected_stream);
        encoder.set_compute_pipeline_state(kernel);
        for (int index = 0; index < 6; ++index) {
            encoder.set_input_array(
                inputs[static_cast<std::size_t>(index)],
                index);
        }
        encoder.set_output_array(output, 6);
        encoder.dispatch_threadgroups(
            MTL::Size(config_.workgroups, 1, 1),
            MTL::Size(config_.routes * 32, 1, 1));
    }

    const char* name() const override {
        return "Mxfp4DecodeReducePrimitive";
    }

    bool is_equivalent(
        const mlx::core::Primitive& other) const override {
        const auto* primitive =
            dynamic_cast<const Mxfp4DecodeReducePrimitive*>(&other);
        return primitive != nullptr
            && primitive->kernel_name_ == kernel_name_;
    }

    std::vector<Shape> output_shapes(
        const std::vector<array>&) override {
        return {config_.output_shape};
    }

private:
    Mxfp4DecodeReduceConfig config_;
    std::string kernel_name_;
};

array mxfp4_decode_reduce_dispatch(
    std::vector<array> inputs,
    Mxfp4DecodeReduceConfig config) {
    auto stream = mlx::core::default_stream(
        mlx::core::default_device());
    if (stream.device != mlx::core::Device::gpu) {
        throw std::invalid_argument(
            "MXFP4 decode down-reduce requires the Metal device");
    }
    auto shape = config.output_shape;
    const auto dtype = config.dtype;
    return array(
        std::move(shape),
        dtype,
        std::make_shared<Mxfp4DecodeReducePrimitive>(
            stream,
            std::move(config)),
        std::move(inputs));
}

std::string mxfp4_pair_swiglu_kernel_name(
    const Mxfp4PairSwiGluConfig& config) {
    std::ostringstream name;
    name
        << "mfq_mxfp4_pair_swiglu_"
        << (config.dtype == mlx::core::float16 ? "f16" : "f32")
        << "_t" << config.tokens
        << "_r" << config.routes
        << "_e" << config.experts
        << "_o" << config.output_width
        << "_k" << config.input_width;
    return name.str();
}

std::string make_mxfp4_pair_swiglu_source(
    const Mxfp4PairSwiGluConfig& config,
    const std::string& kernel_name) {
    std::ostringstream source;
    source
        << "#include <metal_stdlib>\n"
        << "using namespace metal;\n"
        << "using T = "
        << (config.dtype == mlx::core::float16 ? "half" : "float")
        << ";\n"
        << "constant constexpr float MXFP4_LUT[16] = {\n"
           "    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,\n"
           "    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,\n"
           "};\n"
        << "inline float decode_e8m0(uchar raw) {\n"
           "    if (raw == 255u) return NAN;\n"
           "    uint bits = raw == 0u ? 0x00400000u : uint(raw) << 23u;\n"
           "    return as_type<float>(bits);\n"
           "}\n"
        << "kernel void " << kernel_name << "(\n"
        << "device const int* gate_descriptors [[buffer(0)]],\n"
        << "device const uchar* gate_values [[buffer(1)]],\n"
        << "device const uchar* gate_scales [[buffer(2)]],\n"
        << "device const int* up_descriptors [[buffer(3)]],\n"
        << "device const uchar* up_values [[buffer(4)]],\n"
        << "device const uchar* up_scales [[buffer(5)]],\n"
        << "device const T* x [[buffer(6)]],\n"
        << "device const int* expert_ids [[buffer(7)]],\n"
        << "device const float* params [[buffer(8)]],\n"
        << "device T* y [[buffer(9)]],\n"
        << "uint thread_index_in_simdgroup "
           "[[thread_index_in_simdgroup]],\n"
        << "uint simdgroup_index_in_threadgroup "
           "[[simdgroup_index_in_threadgroup]],\n"
        << "uint3 threadgroup_position_in_grid "
           "[[threadgroup_position_in_grid]]) {\n"
        << "constexpr uint TOKENS = " << config.tokens << "u;\n"
        << "constexpr uint ROUTES = " << config.routes << "u;\n"
        << "constexpr uint EXPERTS = " << config.experts << "u;\n"
        << "constexpr uint OUT = " << config.output_width << "u;\n"
        << "constexpr uint K = " << config.input_width << "u;\n"
        << "constexpr uint DESCRIPTOR_SIZE = "
        << config.descriptor_size << "u;\n"
        << R"METAL(
    constexpr uint K_LANES = 8u;
    constexpr uint LANE_GROUPS = 32u / K_LANES;
    constexpr uint SIMD_GROUPS = 2u;
    constexpr uint OUTPUTS_PER_TG = LANE_GROUPS * SIMD_GROUPS;
    constexpr uint OUTPUT_TILES = (OUT + OUTPUTS_PER_TG - 1u)
        / OUTPUTS_PER_TG;

    const uint lane = thread_index_in_simdgroup;
    const uint k_lane = lane & (K_LANES - 1u);
    const uint lane_group = lane / K_LANES;
    const uint workgroup = threadgroup_position_in_grid.x;
    const uint output_tile = workgroup % OUTPUT_TILES;
    const uint routed_row = workgroup / OUTPUT_TILES;
    const uint route = routed_row % ROUTES;
    const uint token = routed_row / ROUTES;
    if (token >= TOKENS) return;

    const uint output = output_tile * OUTPUTS_PER_TG
        + simdgroup_index_in_threadgroup * LANE_GROUPS
        + lane_group;
    const uint bounded_output = min(output, OUT - 1u);
    const int expert = expert_ids[token * ROUTES + route];
    float gate_accumulator = 0.0f;
    float up_accumulator = 0.0f;
    if (expert >= 0 && expert < int(EXPERTS)) {
        const uint descriptor_base = uint(expert) * DESCRIPTOR_SIZE;
        const uint gate_local = uint(gate_descriptors[descriptor_base + 1u]);
        const uint up_local = uint(up_descriptors[descriptor_base + 1u]);
        const uint groups = uint(gate_descriptors[descriptor_base + 4u]);
        const uint gate_value_offset = uint(
            gate_descriptors[descriptor_base + 5u]);
        const uint gate_scale_offset = uint(
            gate_descriptors[descriptor_base + 6u]);
        const uint up_value_offset = uint(
            up_descriptors[descriptor_base + 5u]);
        const uint up_scale_offset = uint(
            up_descriptors[descriptor_base + 6u]);
        const ulong gate_pool_output = ulong(gate_local) * ulong(OUT)
            + ulong(bounded_output);
        const ulong up_pool_output = ulong(up_local) * ulong(OUT)
            + ulong(bounded_output);

        for (uint group = 0u; group < groups; ++group) {
            const uint column = group * 32u + k_lane * 4u;
            const vec<T, 4> raw_activations = *reinterpret_cast<
                device const vec<T, 4>*>(x + token * K + column);
            const float4 activations = float4(raw_activations);
            const ulong gate_packed_offset = ulong(gate_value_offset)
                + gate_pool_output * (ulong(K) >> 1u)
                + (ulong(column) >> 1u);
            const ulong up_packed_offset = ulong(up_value_offset)
                + up_pool_output * (ulong(K) >> 1u)
                + (ulong(column) >> 1u);
            const uchar2 gate_codes = *reinterpret_cast<
                device const uchar2*>(gate_values + gate_packed_offset);
            const uchar2 up_codes = *reinterpret_cast<
                device const uchar2*>(up_values + up_packed_offset);
            const float gate_scale = decode_e8m0(gate_scales[
                ulong(gate_scale_offset)
                    + gate_pool_output * ulong(groups) + ulong(group)]);
            const float up_scale = decode_e8m0(up_scales[
                ulong(up_scale_offset)
                    + up_pool_output * ulong(groups) + ulong(group)]);
            const float4 gate_weights = gate_scale * float4(
                MXFP4_LUT[uint(gate_codes.x & 15u)],
                MXFP4_LUT[uint(gate_codes.x >> 4u)],
                MXFP4_LUT[uint(gate_codes.y & 15u)],
                MXFP4_LUT[uint(gate_codes.y >> 4u)]);
            const float4 up_weights = up_scale * float4(
                MXFP4_LUT[uint(up_codes.x & 15u)],
                MXFP4_LUT[uint(up_codes.x >> 4u)],
                MXFP4_LUT[uint(up_codes.y & 15u)],
                MXFP4_LUT[uint(up_codes.y >> 4u)]);
            gate_accumulator += dot(activations, gate_weights);
            up_accumulator += dot(activations, up_weights);
        }
        for (uint offset = K_LANES >> 1u; offset > 0u; offset >>= 1u) {
            gate_accumulator += simd_shuffle_down(
                gate_accumulator, offset);
            up_accumulator += simd_shuffle_down(
                up_accumulator, offset);
        }
    }

    if (k_lane == 0u && output < OUT) {
        if (expert < 0 || expert >= int(EXPERTS)) {
            y[routed_row * OUT + output] = T(0.0f);
            return;
        }
        // Preserve the unfused graph's projection-boundary rounding.
        float gate = float(T(gate_accumulator));
        float up = float(T(up_accumulator));
        if (params[0] > 0.0f) {
            gate = min(gate, params[0]);
            up = clamp(up, -params[0], params[0]);
        }
        const float activated = gate / (1.0f + exp(-gate));
        y[routed_row * OUT + output] = T(activated * up);
    }
}
)METAL";
    return source.str();
}

class Mxfp4PairSwiGluPrimitive final
    : public mlx::core::UnaryPrimitive {
public:
    Mxfp4PairSwiGluPrimitive(
        mlx::core::Stream stream,
        Mxfp4PairSwiGluConfig config)
        : UnaryPrimitive(stream),
          config_(std::move(config)),
          kernel_name_(mxfp4_pair_swiglu_kernel_name(config_)) {}

    void eval_cpu(
        const std::vector<array>&,
        array&) override {
        throw std::runtime_error(
            "MXFP4 pair SwiGLU has no CPU path");
    }

    void eval_gpu(
        const std::vector<array>& inputs,
        array& output) override {
        if (inputs.size() != 9) {
            throw std::logic_error(
                "MXFP4 pair SwiGLU input count mismatch");
        }
        output.set_data(
            mlx::core::allocator::malloc(output.nbytes()));
        auto& selected_stream = stream();
        auto& device = mlx::core::metal::device(
            selected_stream.device);
        CompileOptions compile_options;
        compile_options.math_mode = MathMode::Fast;
        auto* library = device.get_library(
            kernel_name_,
            compile_options,
            [config = config_, name = kernel_name_] {
                return make_mxfp4_pair_swiglu_source(config, name);
            });
        auto* kernel = device.get_kernel(kernel_name_, library);
        auto& encoder = mlx::core::metal::get_command_encoder(
            selected_stream);
        encoder.set_compute_pipeline_state(kernel);
        for (int index = 0; index < 9; ++index) {
            encoder.set_input_array(
                inputs[static_cast<std::size_t>(index)],
                index);
        }
        encoder.set_output_array(output, 9);
        encoder.dispatch_threadgroups(
            MTL::Size(config_.workgroups, 1, 1),
            MTL::Size(64, 1, 1));
    }

    const char* name() const override {
        return "Mxfp4PairSwiGluPrimitive";
    }

    bool is_equivalent(
        const mlx::core::Primitive& other) const override {
        const auto* primitive =
            dynamic_cast<const Mxfp4PairSwiGluPrimitive*>(&other);
        return primitive != nullptr
            && primitive->kernel_name_ == kernel_name_;
    }

    std::vector<Shape> output_shapes(
        const std::vector<array>&) override {
        return {config_.output_shape};
    }

private:
    Mxfp4PairSwiGluConfig config_;
    std::string kernel_name_;
};

array mxfp4_pair_swiglu_dispatch(
    std::vector<array> inputs,
    Mxfp4PairSwiGluConfig config) {
    auto stream = mlx::core::default_stream(
        mlx::core::default_device());
    if (stream.device != mlx::core::Device::gpu) {
        throw std::invalid_argument(
            "MXFP4 pair SwiGLU requires the Metal device");
    }
    auto shape = config.output_shape;
    const auto dtype = config.dtype;
    return array(
        std::move(shape),
        dtype,
        std::make_shared<Mxfp4PairSwiGluPrimitive>(
            stream,
            std::move(config)),
        std::move(inputs));
}

class Mxfp4BlocksPrimitive final
    : public mlx::core::UnaryPrimitive {
public:
    Mxfp4BlocksPrimitive(
        mlx::core::Stream stream,
        Mxfp4BlocksConfig config)
        : UnaryPrimitive(stream),
          config_(std::move(config)) {}

    void eval_cpu(
        const std::vector<array>&,
        array&) override {
        throw std::runtime_error(
            "MXFP4 block primitive has no CPU path");
    }

    void eval_gpu(
        const std::vector<array>& inputs,
        array& output) override {
        if (inputs.size() != 9) {
            throw std::logic_error(
                "MXFP4 block primitive input count mismatch");
        }
        output.set_data(
            mlx::core::allocator::malloc(output.nbytes()));
        auto& selected_stream = stream();
        auto& device = mlx::core::metal::device(
            selected_stream.device);
        CompileOptions compile_options;
        compile_options.math_mode = MathMode::Fast;
        auto* library = device.get_library(
            "mfq_dsv4_mxfp4_blocks_v5",
            compile_options,
            [] {
                std::string source;
                source.reserve(
                    sizeof(detail::kSteelMmaSource)
                    + sizeof(detail::kMfePrefillSource)
                    + 128);
                source += "#include <metal_stdlib>\n";
                source += "#include <metal_simdgroup>\n";
                source += "#include <metal_simdgroup_matrix>\n";
                source += "#define MFQ_ENABLE_DSV4_MXFP4_BLOCKS 1\n";
                source += "using namespace metal;\n";
                source += detail::kSteelMmaSource;
                source += detail::kMfePrefillSource;
                return source;
            });
        const char* kernel_name = config_.paired
            ? "mfq_dsv4_mxfp4_pair_concat_f16_bm32_bn32_bk32"
            : "mfq_dsv4_mxfp4_single_f16_bm32_bn32_bk32";
        auto* kernel = device.get_kernel(
            kernel_name,
            library);
        auto& encoder =
            mlx::core::metal::get_command_encoder(
                selected_stream);
        encoder.set_compute_pipeline_state(kernel);
        for (int index = 0; index < 9; ++index) {
            encoder.set_input_array(
                inputs[static_cast<std::size_t>(index)],
                index);
        }
        encoder.set_output_array(output, 9);
        encoder.set_bytes(config_.max_blocks, 10);
        encoder.set_bytes(config_.output_width, 11);
        encoder.set_bytes(config_.input_width, 12);
        encoder.dispatch_threadgroups(
            MTL::Size(
                ((config_.paired ? 2 : 1) * config_.output_width + 31) / 32,
                config_.max_blocks,
                1),
            MTL::Size(64, 1, 1));
    }

    const char* name() const override {
        return "Mxfp4BlocksPrimitive";
    }

    bool is_equivalent(
        const mlx::core::Primitive& other) const override {
        const auto* primitive =
            dynamic_cast<const Mxfp4BlocksPrimitive*>(&other);
        return primitive != nullptr
            && primitive->config_.route_count == config_.route_count
            && primitive->config_.max_blocks == config_.max_blocks
            && primitive->config_.experts == config_.experts
            && primitive->config_.output_width == config_.output_width
            && primitive->config_.input_width == config_.input_width
            && primitive->config_.block_rows == config_.block_rows
            && primitive->config_.paired == config_.paired;
    }

    std::vector<Shape> output_shapes(
        const std::vector<array>&) override {
        return {config_.output_shape};
    }

private:
    Mxfp4BlocksConfig config_;
};

array mxfp4_blocks_dispatch(
    std::vector<array> inputs,
    Mxfp4BlocksConfig config) {
    auto stream = mlx::core::default_stream(
        mlx::core::default_device());
    if (stream.device != mlx::core::Device::gpu) {
        throw std::invalid_argument(
            "MXFP4 blocks require the Metal device");
    }
    auto shape = config.output_shape;
    return array(
        std::move(shape),
        mlx::core::float16,
        std::make_shared<Mxfp4BlocksPrimitive>(
            stream,
            std::move(config)),
        std::move(inputs));
}

class GroupedMmqPrimitive final
    : public mlx::core::UnaryPrimitive {
public:
    GroupedMmqPrimitive(
        mlx::core::Stream stream,
        GroupedMmqConfig config)
        : UnaryPrimitive(stream),
          config_(std::move(config)) {}

    void eval_cpu(
        const std::vector<array>&,
        array&) override {
        throw std::runtime_error(
            "grouped MMQ primitive has no CPU path");
    }

    void eval_gpu(
        const std::vector<array>& inputs,
        array& output) override {
        if (inputs.size() != 28) {
            throw std::logic_error(
                "grouped MMQ primitive input count mismatch");
        }
        output.set_data(
            mlx::core::allocator::malloc(output.nbytes()));
        auto& selected_stream = stream();
        auto& device = mlx::core::metal::device(
            selected_stream.device);
        CompileOptions compile_options;
        compile_options.math_mode = MathMode::Fast;
        const bool vector_vq =
            (static_cast<std::uint32_t>(config_.vq_profile_mask)
                & kGroupedVqVectorProfileMask) != 0;
        const bool vector_jsc_extended =
            (static_cast<std::uint32_t>(config_.vq_profile_mask)
                & kGroupedJscExtendedProfileMask) != 0;
        std::string library_name;
        if (config_.use_nax) {
            library_name = vector_vq
                ? (vector_jsc_extended
                    ? "mfq_grouped_mfe_nax_v2_legacy_vq_jsc_extended"
                    : "mfq_grouped_mfe_nax_v2_legacy_vq_vector")
                : (vector_jsc_extended
                    ? "mfq_grouped_mfe_nax_v2_jsc_extended"
                    : "mfq_grouped_mfe_nax_v2");
        } else {
            library_name = vector_vq
                ? (vector_jsc_extended
                    ? "mfq_grouped_mmq_v13_legacy_vq_jsc_extended"
                    : "mfq_grouped_mmq_v13_legacy_vq_vector")
                : (vector_jsc_extended
                    ? "mfq_grouped_mmq_v13_jsc_extended"
                    : "mfq_grouped_mmq_v13");
        }
        if (config_.use_nax) {
            library_name += "_fm" + std::to_string(config_.family_mask);
        }
        auto* library = device.get_library(
            library_name,
            compile_options,
            [
                use_nax = config_.use_nax,
                vector_vq,
                vector_jsc_extended,
                family_mask = config_.use_nax
                    ? config_.family_mask
                    : 127
            ] {
                std::string source;
                source.reserve(
                    (use_nax
                        ? sizeof(detail::kSteelNaxSource)
                        : sizeof(detail::kSteelMmaSource))
                    + sizeof(detail::kMfePrefillSource)
                    + 256);
                source += "#include <metal_stdlib>\n";
                source += "#include <metal_simdgroup>\n";
                source += "#include <metal_simdgroup_matrix>\n";
                if (use_nax) {
                    source += "#include <MetalPerformancePrimitives/"
                        "MetalPerformancePrimitives.h>\n";
                    source += "#define MFQ_ENABLE_NAX 1\n";
                }
                if (vector_vq) {
                    source +=
                        "#define MFQ_ENABLE_LEGACY_VQ_VECTOR 1\n";
                }
                if (vector_jsc_extended) {
                    source +=
                        "#define MFQ_ENABLE_JSC_EXTENDED_VECTOR 1\n";
                }
                if (use_nax) {
                    source += "#define MFQ_GROUPED_FAMILY_MASK ";
                    source += std::to_string(family_mask);
                    source += "\n";
                }
                source += "using namespace metal;\n";
                source += "using bfloat16_t = bfloat;\n";
                source += use_nax
                    ? detail::kSteelNaxSource
                    : detail::kSteelMmaSource;
                source += detail::kMfePrefillSource;
                return source;
            });
        auto& encoder =
            mlx::core::metal::get_command_encoder(
                selected_stream);
        encoder.set_input_array(inputs[0], 0);
        for (int source = 8; source <= 16; ++source) {
            encoder.set_input_array(
                inputs[static_cast<std::size_t>(source)],
                source - 7);
        }
        for (int source = 22; source <= 24; ++source) {
            encoder.set_input_array(
                inputs[static_cast<std::size_t>(source)],
                source - 12);
        }
        const GroupedMmqParameters parameters{
            .route_count = config_.route_count,
            .tokens = config_.tokens,
            .routes = config_.routes,
            .experts = config_.experts,
            .output_width = config_.output_width,
            .matrix_output_width = config_.matrix_output_width,
            .projections = config_.projections,
            .input_width = config_.input_width,
            .descriptor_size = config_.descriptor_size,
            .variant_stride = config_.variant_stride,
            .shared_input = config_.shared_input,
            .input_sorted = config_.input_sorted,
            .swiglu_limit = config_.swiglu_limit,
        };
        encoder.set_output_array(output, 13);
        encoder.set_bytes(parameters, 14);
        encoder.set_input_array(inputs[17], 15);
        encoder.set_input_array(inputs[18], 16);
        encoder.set_input_array(inputs[19], 17);
        encoder.set_input_array(inputs[26], 18);
        encoder.set_input_array(inputs[27], 19);
        encoder.set_input_array(inputs[20], 20);
        encoder.set_input_array(inputs[21], 21);
        for (int source = 1; source <= 5; ++source) {
            encoder.set_input_array(
                inputs[static_cast<std::size_t>(source)],
                source + 21);
        }
        encoder.set_input_array(inputs[6], 27);
        encoder.set_input_array(inputs[7], 28);
        const bool block48 = config_.block_rows == 48;
        const bool block64 = config_.block_rows == 64;
        const bool block80 = config_.block_rows == 80;
        const bool block96 = config_.block_rows == 96;
        const bool columns96 = config_.tile_columns == 96;
        const bool columns128 = config_.tile_columns == 128;
        const char* kernel_name = config_.use_nax
            ? (config_.direct_nax
                ? (config_.fused_swiglu != 0
                    ? "mfq_grouped_mfe_nax_direct_swiglu_f16_bm32_bn64_bk96"
                    : "mfq_grouped_mfe_nax_direct_f16_bm32_bn64_bk96")
                : (config_.fused_swiglu != 0
                    ? (columns96
                        ? (block48
                            ? "mfq_grouped_mfe_nax_swiglu_f16_bm48_bn96_bk96"
                            : "mfq_grouped_mfe_nax_swiglu_f16_bm64_bn96_bk96")
                        : columns128
                        ? "mfq_grouped_mfe_nax_swiglu_f16_bm32_bn128_bk96"
                        : block96
                        ? "mfq_grouped_mfe_nax_swiglu_f16_bm96_bn64_bk96"
                        : block80
                        ? "mfq_grouped_mfe_nax_swiglu_f16_bm80_bn64_bk96"
                        : block48
                        ? "mfq_grouped_mfe_nax_swiglu_f16_bm48_bn64_bk96"
                        : block64
                        ? "mfq_grouped_mfe_nax_swiglu_f16_bm64_bn64_bk96"
                        : "mfq_grouped_mfe_nax_swiglu_f16_bm32_bn64_bk96")
                    : (columns96
                        ? (block48
                            ? "mfq_grouped_mfe_nax_f16_bm48_bn96_bk96"
                            : "mfq_grouped_mfe_nax_f16_bm64_bn96_bk96")
                        : columns128
                        ? "mfq_grouped_mfe_nax_f16_bm32_bn128_bk96"
                        : block96
                        ? "mfq_grouped_mfe_nax_f16_bm96_bn64_bk96"
                        : block80
                        ? "mfq_grouped_mfe_nax_f16_bm80_bn64_bk96"
                        : block48
                        ? "mfq_grouped_mfe_nax_f16_bm48_bn64_bk96"
                        : block64
                        ? "mfq_grouped_mfe_nax_f16_bm64_bn64_bk96"
                        : "mfq_grouped_mfe_nax_f16_bm32_bn64_bk96")))
            : (config_.fused_swiglu != 0
                ? (config_.has_nepq_residual != 0
                    ? "mfq_grouped_mmq_swiglu_f16_bm32_bn64_bk96_nr"
                    : "mfq_grouped_mmq_swiglu_f16_bm32_bn64_bk96")
                : (config_.has_nepq_residual != 0
                    ? "mfq_grouped_mmq_f16_bm32_bn64_bk96_nr"
                    : "mfq_grouped_mmq_f16_bm32_bn64_bk96"));
        auto* kernel = device.get_kernel(
            kernel_name,
            library);
        encoder.set_compute_pipeline_state(kernel);
        encoder.dispatch_threadgroups(
            MTL::Size(
                (config_.output_width + config_.tile_columns - 1)
                    / config_.tile_columns,
                config_.max_blocks,
                1),
            MTL::Size(
                config_.use_nax
                    ? config_.block_rows * config_.tile_columns / 16
                    : 256,
                1,
                1));
    }

    const char* name() const override {
        return "GroupedMmqPrimitive";
    }

    bool is_equivalent(
        const mlx::core::Primitive& other) const override {
        const auto* primitive =
            dynamic_cast<const GroupedMmqPrimitive*>(&other);
        return primitive != nullptr
            && primitive->config_.route_count == config_.route_count
            && primitive->config_.max_blocks == config_.max_blocks
            && primitive->config_.block_rows == config_.block_rows
            && primitive->config_.tile_columns == config_.tile_columns
            && primitive->config_.experts == config_.experts
            && primitive->config_.output_width == config_.output_width
            && primitive->config_.projections == config_.projections
            && primitive->config_.input_width == config_.input_width
            && primitive->config_.variant_stride == config_.variant_stride
            && primitive->config_.shared_input == config_.shared_input
            && primitive->config_.input_sorted == config_.input_sorted
            && primitive->config_.fused_swiglu == config_.fused_swiglu
            && primitive->config_.has_nepq_residual
                == config_.has_nepq_residual
            && primitive->config_.family_mask == config_.family_mask
            && primitive->config_.vq_profile_mask == config_.vq_profile_mask
            && primitive->config_.use_nax == config_.use_nax
            && primitive->config_.direct_nax == config_.direct_nax
            && primitive->config_.swiglu_limit == config_.swiglu_limit;
    }

    std::vector<Shape> output_shapes(
        const std::vector<array>&) override {
        return {config_.output_shape};
    }

private:
    GroupedMmqConfig config_;
};

array grouped_mmq_dispatch(
    std::vector<array> inputs,
    GroupedMmqConfig config) {
    auto stream = mlx::core::default_stream(
        mlx::core::default_device());
    if (stream.device != mlx::core::Device::gpu) {
        throw std::invalid_argument(
            "grouped MMQ requires the Metal device");
    }
    auto shape = config.output_shape;
    return array(
        std::move(shape),
        mlx::core::float16,
        std::make_shared<GroupedMmqPrimitive>(
            stream,
            std::move(config)),
        std::move(inputs));
}

mlx::core::fast::CustomKernelFunction
make_moe_kernel() {
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    return mlx::core::fast::metal_kernel(
        "mfq_cpp_heterogeneous_mfe",
        {
            "descriptors",
            "nint_q",
            "nint_sub_scale",
            "nint_sub_min",
            "nint_anchor_scale",
            "nint_anchor_min",
            "q8_q",
            "q8_scales",
            "vq_indices",
            "vq_state",
            "vq_aux",
            "vq_anchors",
            "vq_codebooks",
            "vq_scales",
            "vq_state_to_codebank",
            "vq_banks",
            "vq_parameters",
            "vq_residual_codebooks",
            "vq_residual_first",
            "vq_residual_second",
            "mx_values",
            "mx_scales",
            "x",
            "expert_ids",
            "route_order",
            "params",
            "expert_map",
        },
        {"y"},
        kMoeSource,
        kMoeHeader,
        true,
        false,
        options);
}

const mlx::core::fast::CustomKernelFunction&
moe_kernel() {
    static const auto kernel = make_moe_kernel();
    return kernel;
}

mlx::core::fast::CustomKernelFunction
make_moe_hadamard_kernel() {
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    return mlx::core::fast::metal_kernel(
        "mfq_cpp_moe_signed_hadamard",
        {"x", "signs"},
        {"y"},
        kMoeHadamardSource,
        "",
        true,
        false,
        options);
}

const mlx::core::fast::CustomKernelFunction&
moe_hadamard_kernel() {
    static const auto kernel =
        make_moe_hadamard_kernel();
    return kernel;
}

struct RotationSpec {
    array signs;
    std::vector<std::int8_t> sign_values;
    int block = 0;
    std::uint64_t seed = 0;
};

struct PackedStreams {
    detail::StagingVector<std::uint8_t> nint_q;
    detail::StagingVector<std::uint8_t> nint_sub_scale;
    detail::StagingVector<std::uint8_t> nint_sub_min;
    detail::StagingVector<std::uint8_t> nint_anchor_scale;
    detail::StagingVector<std::uint8_t> nint_anchor_min;
    detail::StagingVector<std::uint8_t> q8_q;
    detail::StagingVector<std::uint8_t> q8_scales;
    detail::StagingVector<std::uint8_t> vq_indices;
    detail::StagingVector<std::uint8_t> vq_state;
    detail::StagingVector<std::uint8_t> vq_aux;
    detail::StagingVector<std::uint8_t> vq_anchors;
    detail::StagingVector<std::uint8_t> vq_codebooks;
    detail::StagingVector<std::uint8_t> vq_scales;
    detail::StagingVector<std::uint8_t>
        vq_state_to_codebank;
    detail::StagingVector<std::uint8_t> vq_banks;
    detail::StagingVector<std::uint8_t> vq_parameters;
    detail::StagingVector<std::uint8_t> vq_residual_codebooks;
    detail::StagingVector<std::uint8_t> vq_residual_first;
    detail::StagingVector<std::uint8_t> vq_residual_second;
    detail::StagingVector<std::uint8_t> mx_values;
    detail::StagingVector<std::uint8_t> mx_scales;
};

struct DenseReferenceMoeWeight {
    array values;

    array embedding(
        const array& rows,
        Dtype dtype) const {
        return mlx::core::astype(
            mlx::core::take(values, rows, 0),
            dtype);
    }

    std::size_t packed_nbytes() const noexcept {
        return values.nbytes();
    }
};

using ReferenceMoeWeight = std::variant<
    MlxNintWeight,
    MlxNint8ZeroWeight,
    MlxVqWeight,
    MlxMxWeight,
    MlxFp8SqWeight,
    MlxMxfp4SqWeight,
    DenseReferenceMoeWeight>;

struct ReferenceMoeCohort {
    std::vector<std::int32_t> expert_ids;
    ReferenceMoeWeight weight;
};

struct Mxfp4SqMoeCohort {
    array expert_map;
    MlxMxfp4SqWeight weight;
};

struct Fp8SqMoeCohort {
    array expert_map;
    MlxFp8SqWeight weight;
};

void validate_nint_payload_shape(
    std::span<const std::uint8_t> payload,
    int expected_rows,
    int expected_columns) {
    BlobCursor cursor(payload);
    const int raw_bits = static_cast<int>(
        cursor.scalar<std::uint8_t>("NINT bits"));
    const bool is_nint_v2 = (raw_bits & 0x80) != 0;
    const int bits = raw_bits & 0x7f;
    const int sub_bits = static_cast<int>(
        cursor.scalar<std::uint8_t>(
            "NINT sub bits"));
    const int group_size =
        cursor.scalar<std::int32_t>(
            "NINT group size");
    const int axis =
        cursor.scalar<std::int32_t>("NINT axis");
    const int columns =
        cursor.scalar<std::int32_t>(
            "NINT neuron length");
    const auto dimensions =
        cursor.scalar<std::uint32_t>(
            "NINT dimension count");
    if (
        bits < 1
        || bits > 8
        || sub_bits < 1
        || sub_bits > 8
        || group_size <= 0
        || axis != 0
        || columns != expected_columns
        || dimensions != 2
    ) {
        throw std::runtime_error(
            "MFE NINT cohort shape is inconsistent");
    }
    const auto rows =
        cursor.scalar<std::int64_t>(
            "NINT output shape");
    const auto shape_columns =
        cursor.scalar<std::int64_t>(
            "NINT input shape");
    const auto output_size =
        cursor.scalar<std::uint32_t>(
            "NINT output size");
    const auto groups =
        cursor.scalar<std::uint32_t>(
            "NINT group count");
    const auto expected_groups =
        (
            static_cast<std::uint64_t>(
                expected_columns)
            + static_cast<std::uint64_t>(
                group_size)
            - 1
        ) / static_cast<std::uint64_t>(
            group_size);
    if (
        rows != expected_rows
        || shape_columns != expected_columns
        || output_size
            != static_cast<std::uint32_t>(
                expected_rows)
        || groups != expected_groups
    ) {
        throw std::runtime_error(
            "MFE NINT cohort shape is inconsistent");
    }

    const auto metadata_count = checked_product(
        static_cast<std::size_t>(output_size),
        static_cast<std::size_t>(groups),
        "NINT metadata count");
    const auto value_count = checked_product(
        metadata_count,
        static_cast<std::size_t>(group_size),
        "NINT value count");
    const auto anchor_bytes = checked_product(
        static_cast<std::size_t>(output_size),
        2 * sizeof(std::uint16_t),
        "NINT anchor bytes");
    const auto packed_metadata =
        checked_packed_size(
            metadata_count,
            sub_bits,
            "NINT metadata bytes");
    const auto packed_values =
        checked_packed_size(
            value_count,
            bits,
            "NINT value bytes");
    const auto packed_tail = checked_add(
        checked_product(
            packed_metadata,
            2,
            "NINT metadata bytes"),
        packed_values,
        "NINT packed tail");
    const auto old_tail = checked_add(
        checked_product(
            metadata_count,
            2,
            "legacy NINT metadata bytes"),
        value_count,
        "legacy NINT tail");
    const auto remaining = cursor.remaining();
    if (is_nint_v2) {
        const auto k_selector_bytes = checked_packed_size(
            static_cast<std::size_t>(output_size),
            2,
            "NINTv2 k-selector bytes");
        const auto q_selector_bytes = checked_packed_size(
            static_cast<std::size_t>(output_size),
            3,
            "NINTv2 q-selector bytes");
        const auto minimum_v2 = checked_add(
            anchor_bytes,
            checked_add(
                k_selector_bytes,
                q_selector_bytes,
                "NINTv2 selector bytes"),
            "NINTv2 minimum payload");
        if (remaining < minimum_v2) {
            throw std::runtime_error(
                "invalid MFE NINTv2 cohort payload length");
        }
        return;
    }
    if (
        remaining
            != checked_add(
                anchor_bytes,
                packed_tail,
                "NINT payload bytes")
        && remaining
            != checked_add(
                anchor_bytes,
                old_tail,
                "legacy NINT payload bytes")
    ) {
        throw std::runtime_error(
            "invalid MFE NINT cohort payload length");
    }
}

void claim_experts(
    const std::vector<std::int32_t>& expert_ids,
    std::vector<int>& owners,
    int pool) {
    for (const auto expert : expert_ids) {
        if (
            expert < 0
            || expert
                >= static_cast<std::int32_t>(
                    owners.size())
        ) {
            throw std::runtime_error(
                "MFE pool contains an invalid expert id");
        }
        if (owners[static_cast<std::size_t>(expert)] >= 0) {
            throw std::runtime_error(
                "an expert belongs to multiple MFE pools");
        }
        owners[static_cast<std::size_t>(expert)] = pool;
    }
}

MlxNintWeight add_nint_pool(
    std::span<const std::uint8_t> payload,
    const std::vector<std::int32_t>& expert_ids,
    int out_per_expert,
    int neuron_len,
    PackedStreams& streams,
    std::vector<std::int32_t>& descriptors,
    bool pack_execution = true) {
    const auto expected_rows = checked_product(
        expert_ids.size(),
        static_cast<std::size_t>(out_per_expert),
        "NINT cohort row count");
    const int checked_rows = checked_int(
        expected_rows,
        "NINT cohort row count");
    validate_nint_payload_shape(
        payload,
        checked_rows,
        neuron_len);
    auto weight = MlxNintWeight::from_blob(payload);
    if (
        weight.output_size() != checked_rows
        || weight.input_size() != neuron_len
        || weight.groups()
            != (
                neuron_len + weight.group_size() - 1
            ) / weight.group_size()
    ) {
        throw std::runtime_error(
            "MFE NINT cohort shape is inconsistent");
    }

    if (!pack_execution) {
        return weight;
    }

    const int row_layout_offset = checked_int(
        streams.nint_q.size(),
        "NINT row-layout offset");
    append_raw(
        streams.nint_q,
        weight.row_q_layout(),
        mlx::core::uint8,
        "NINT row layouts");
    while ((streams.nint_q.size() & 3u) != 0u) {
        streams.nint_q.push_back(0);
    }
    const int row_byte_offsets_offset = checked_int(
        streams.nint_q.size(),
        "NINT row-byte-offset offset");
    append_raw(
        streams.nint_q,
        weight.row_q_byte_offsets(),
        mlx::core::uint32,
        "NINT row byte offsets");
    const int q_offset = checked_int(
        streams.nint_q.size(),
        "NINT q offset");
    const int sub_offset = checked_int(
        streams.nint_sub_scale.size(),
        "NINT sub offset");
    const int anchor_offset = checked_int(
        streams.nint_anchor_scale.size() / sizeof(float),
        "NINT anchor offset");

    for (std::size_t local_expert = 0;
         local_expert < expert_ids.size(); ++local_expert) {
        const int expert = expert_ids[local_expert];
        const auto base = checked_product(
            static_cast<std::size_t>(expert),
            static_cast<std::size_t>(kDescriptorSize),
            "descriptor offset");
        descriptors[base + kFamily] = kFamilyNint;
        descriptors[base + kLocalExpert] =
            checked_int(local_expert, "local expert");
        descriptors[base + kOut] = out_per_expert;
        descriptors[base + kInput] = neuron_len;
        descriptors[base + kNintBits] = weight.bits();
        descriptors[base + kNintGroupSize] = weight.group_size();
        descriptors[base + kNintGroups] = weight.groups();
        descriptors[base + kNintQOffset] = q_offset;
        descriptors[base + kNintSubOffset] = sub_offset;
        descriptors[base + kNintAnchorOffset] = anchor_offset;
        descriptors[base + kNintQ5Execution] = 0;
        descriptors[base + kNintRowQLayoutOffset] = row_layout_offset;
        descriptors[base + kNintRowQByteOffsetsOffset] =
            row_byte_offsets_offset;
        descriptors[base + kNintV2] = 1;
    }

    append_raw(
        streams.nint_q,
        weight.packed_values(),
        mlx::core::uint8,
        "NINT values");
    append_raw(
        streams.nint_sub_scale,
        weight.sub_scales(),
        mlx::core::uint8,
        "NINT sub scales");
    append_raw(
        streams.nint_sub_min,
        weight.sub_mins(),
        mlx::core::uint8,
        "NINT sub minima");
    append_raw(
        streams.nint_anchor_scale,
        weight.neuron_scales(),
        mlx::core::float32,
        "NINT neuron scales");
    append_raw(
        streams.nint_anchor_min,
        weight.neuron_mins(),
        mlx::core::float32,
        "NINT neuron minima");
    return weight;
}

MlxNint8ZeroWeight add_q8_pool(
    std::span<const std::uint8_t> payload,
    const std::vector<std::int32_t>& expert_ids,
    int out_per_expert,
    int neuron_len,
    PackedStreams& streams,
    std::vector<std::int32_t>& descriptors,
    bool pack_execution = true) {
    auto weight =
        MlxNint8ZeroWeight::from_blob(payload);
    const auto expected_rows = checked_product(
        expert_ids.size(),
        static_cast<std::size_t>(out_per_expert),
        "NINT8-0 cohort row count");
    if (
        weight.output_size()
            != checked_int(
                expected_rows,
                "NINT8-0 cohort row count")
        || weight.input_size() != neuron_len
        || weight.groups() != neuron_len / 32
    ) {
        throw std::runtime_error(
            "MFE NINT8-0 cohort shape is inconsistent");
    }

    if (!pack_execution) {
        return weight;
    }

    const int q_offset =
        checked_int(
            streams.q8_q.size(),
            "NINT8-0 q offset");
    const int scale_offset =
        checked_int(
            streams.q8_scales.size()
                / sizeof(std::uint16_t),
            "NINT8-0 scale offset");
    for (
        std::size_t local_expert = 0;
        local_expert < expert_ids.size();
        ++local_expert
    ) {
        const int expert =
            expert_ids[local_expert];
        const auto base =
            checked_product(
                static_cast<std::size_t>(expert),
                static_cast<std::size_t>(
                    kDescriptorSize),
                "descriptor offset");
        descriptors[base + kFamily] =
            kFamilyNint8Zero;
        descriptors[base + kLocalExpert] =
            checked_int(local_expert, "local expert");
        descriptors[base + kOut] = out_per_expert;
        descriptors[base + kInput] = neuron_len;
        descriptors[base + kQ8Groups] =
            weight.groups();
        descriptors[base + kQ8QOffset] = q_offset;
        descriptors[base + kQ8ScaleOffset] =
            scale_offset;
    }

    append_raw(
        streams.q8_q,
        weight.quantized_values(),
        mlx::core::int8,
        "NINT8-0 values");
    append_raw(
        streams.q8_scales,
        weight.scales(),
        mlx::core::float16,
        "NINT8-0 scales");
    return weight;
}

std::optional<MlxMxWeight> add_mx_pool(
    std::string_view dtype,
    std::span<const std::uint8_t> payload,
    const std::vector<std::int32_t>& expert_ids,
    int out_per_expert,
    int neuron_len,
    PackedStreams& streams,
    std::vector<std::int32_t>& descriptors,
    bool pack_execution = true) {
    const bool mxfp4 = dtype == "MXFP4";
    const bool mxfp8 = dtype == "MXFP8";
    if (!mxfp4 && !mxfp8) {
        throw std::invalid_argument("unsupported MFE MX cohort dtype");
    }
    const std::string label(dtype);
    const auto field = [&](const char* suffix) {
        return label + " " + suffix;
    };
    BlobCursor cursor(payload);
    const auto magic_name = field("magic");
    const auto magic = cursor.bytes(4, magic_name.c_str());
    const auto version_name = field("version");
    const auto kind_name = field("kind");
    const auto reserved_name = field("reserved");
    if (
        magic.size() != 4
        || std::memcmp(magic.data(), "MXT1", 4) != 0
        || cursor.scalar<std::uint8_t>(version_name.c_str()) != 1
        || cursor.scalar<std::uint8_t>(kind_name.c_str())
            != static_cast<std::uint8_t>(mxfp4 ? 4 : 8)
        || cursor.scalar<std::uint16_t>(reserved_name.c_str()) != 0
    ) {
        throw std::runtime_error(
            "invalid MFE " + label + " payload header");
    }
    const auto rows_name = field("rows");
    const auto columns_name = field("columns");
    const auto storage_rows_name = field("storage rows");
    const auto storage_columns_name = field("storage columns");
    const auto scale_rows_name = field("scale rows");
    const auto scale_columns_name = field("scale columns");
    const auto rows = cursor.scalar<std::uint64_t>(rows_name.c_str());
    const auto columns = cursor.scalar<std::uint64_t>(columns_name.c_str());
    const auto storage_rows =
        cursor.scalar<std::uint64_t>(storage_rows_name.c_str());
    const auto storage_columns =
        cursor.scalar<std::uint64_t>(storage_columns_name.c_str());
    const auto scale_rows =
        cursor.scalar<std::uint64_t>(scale_rows_name.c_str());
    const auto scale_columns =
        cursor.scalar<std::uint64_t>(scale_columns_name.c_str());
    const auto expected_rows = checked_product(
        expert_ids.size(),
        static_cast<std::size_t>(out_per_expert),
        "MX cohort row count");
    const auto block = static_cast<std::uint64_t>(mxfp4 ? 32 : 128);
    if (
        neuron_len % static_cast<int>(block) != 0
        || rows != expected_rows
        || columns != static_cast<std::uint64_t>(neuron_len)
        || storage_rows != rows
        || storage_columns != (mxfp4 ? columns / 2 : columns)
        || scale_rows != (mxfp4 ? rows : (rows + 127) / 128)
        || scale_columns != columns / block
    ) {
        throw std::runtime_error(
            "MFE " + label + " cohort shape is inconsistent");
    }
    const auto value_count = checked_product(
        checked_size(storage_rows, "MX storage rows"),
        checked_size(storage_columns, "MX storage columns"),
        "MX value bytes");
    const auto scale_count = checked_product(
        checked_size(scale_rows, "MX scale rows"),
        checked_size(scale_columns, "MX scale columns"),
        "MX scale bytes");
    const auto values_name = field("values");
    const auto scales_name = field("scales");
    const auto values = cursor.bytes(value_count, values_name.c_str());
    const auto scales = cursor.bytes(scale_count, scales_name.c_str());
    if (std::find(scales.begin(), scales.end(), std::uint8_t{255})
        != scales.end()) {
        throw std::runtime_error(
            "MFE " + label + " contains an E8M0 NaN scale");
    }
    if (
        mxfp8
        && std::any_of(
            values.begin(),
            values.end(),
            [](std::uint8_t value) {
                return (value & 0x7fu) == 0x7fu;
            })
    ) {
        throw std::runtime_error(
            "MFE MXFP8 contains an E4M3 NaN code");
    }
    if (cursor.remaining() != 0) {
        throw std::runtime_error(
            "trailing bytes in MFE " + label + " cohort");
    }
    if (!pack_execution) {
        return MlxMxWeight::from_blob(label, payload);
    }

    const std::int32_t value_offset = checked_descriptor_u32(
        streams.mx_values.size(),
        "MX value offset");
    const std::int32_t scale_offset = checked_descriptor_u32(
        streams.mx_scales.size(),
        "MX scale offset");
    const int groups = neuron_len / static_cast<int>(block);
    for (
        std::size_t local_expert = 0;
        local_expert < expert_ids.size();
        ++local_expert
    ) {
        const int expert = expert_ids[local_expert];
        const auto base = checked_product(
            static_cast<std::size_t>(expert),
            static_cast<std::size_t>(kDescriptorSize),
            "descriptor offset");
        descriptors[base + kFamily] =
            mxfp4 ? kFamilyMxfp4 : kFamilyMxfp8;
        descriptors[base + kLocalExpert] =
            checked_int(local_expert, "local expert");
        descriptors[base + kOut] = out_per_expert;
        descriptors[base + kInput] = neuron_len;
        descriptors[base + kMxGroups] = groups;
        descriptors[base + kMxValueOffset] = value_offset;
        descriptors[base + kMxScaleOffset] = scale_offset;
    }
    append_bytes(streams.mx_values, values, "MX values");
    append_bytes(streams.mx_scales, scales, "MX scales");
    return std::nullopt;
}

std::optional<DenseReferenceMoeWeight> add_dense_pool(
    std::string_view dtype,
    std::span<const std::uint8_t> payload,
    const std::vector<std::int32_t>& expert_ids,
    int out_per_expert,
    int neuron_len,
    PackedStreams& streams,
    std::vector<std::int32_t>& descriptors,
    bool pack_execution = true) {
    const bool bf16 = dtype == "BF16";
    if (!bf16 && dtype != "F16") {
        throw std::invalid_argument("unsupported MFE dense cohort dtype");
    }
    BlobCursor cursor(payload);
    if (cursor.scalar<std::uint32_t>("dense dimension count") != 2) {
        throw std::runtime_error(
            "MFE dense expert cohort must be rank 2");
    }
    const auto rows = cursor.scalar<std::int64_t>("dense rows");
    const auto columns = cursor.scalar<std::int64_t>("dense columns");
    const auto expected_rows = checked_product(
        expert_ids.size(),
        static_cast<std::size_t>(out_per_expert),
        "dense cohort row count");
    const auto value_count = checked_product(
        expected_rows,
        static_cast<std::size_t>(neuron_len),
        "dense cohort value count");
    const auto value_bytes = checked_product(
        value_count,
        sizeof(std::uint16_t),
        "dense cohort value bytes");
    if (
        rows != static_cast<std::int64_t>(expected_rows)
        || columns != neuron_len
        || cursor.remaining() != value_bytes
    ) {
        throw std::runtime_error(
            "MFE dense expert cohort shape is inconsistent");
    }
    const auto values = cursor.bytes(value_bytes, "dense values");
    if (!pack_execution) {
        std::vector<std::uint8_t> copied(values.begin(), values.end());
        auto array_value = make_raw_array(
            std::move(copied),
            bf16 ? mlx::core::bfloat16 : mlx::core::float16);
        return DenseReferenceMoeWeight{
            mlx::core::reshape(
                std::move(array_value),
                Shape{
                    checked_int(expected_rows, "dense cohort rows"),
                    neuron_len,
                }),
        };
    }

    if ((streams.q8_q.size() & 1u) != 0u) {
        streams.q8_q.push_back(0);
    }
    const int value_offset = checked_int(
        streams.q8_q.size(),
        "dense value offset");
    for (
        std::size_t local_expert = 0;
        local_expert < expert_ids.size();
        ++local_expert
    ) {
        const int expert = expert_ids[local_expert];
        const auto base = checked_product(
            static_cast<std::size_t>(expert),
            static_cast<std::size_t>(kDescriptorSize),
            "descriptor offset");
        descriptors[base + kFamily] = bf16 ? kFamilyBf16 : kFamilyF16;
        descriptors[base + kLocalExpert] =
            checked_int(local_expert, "local expert");
        descriptors[base + kOut] = out_per_expert;
        descriptors[base + kInput] = neuron_len;
        descriptors[base + kDenseValueOffset] = value_offset;
    }
    append_bytes(streams.q8_q, values, "dense values");
    return std::nullopt;
}

std::vector<std::int8_t> int8_values(
    const array& source,
    const char* name) {
    if (
        source.dtype() != mlx::core::int8
        || !source.flags().row_contiguous
    ) {
        throw std::runtime_error(
            std::string("invalid MFE packed ") + name);
    }
    auto evaluated = source;
    evaluated.eval();
    return {
        evaluated.data<std::int8_t>(),
        evaluated.data<std::int8_t>()
            + static_cast<std::ptrdiff_t>(
                evaluated.size()),
    };
}

int rotation_variant(
    const MlxVqWeight& weight,
    std::vector<RotationSpec>& rotations) {
    if (weight.rotation_block() == 0) {
        return 0;
    }
    auto values = int8_values(
        weight.rotation_signs(),
        "VQ rotation signs");
    if (
        values.size()
        != static_cast<std::size_t>(
            weight.input_size())
    ) {
        throw std::runtime_error(
            "rotated MFE VQ sign width mismatch");
    }
    for (
        std::size_t index = 0;
        index < rotations.size();
        ++index
    ) {
        const auto& rotation = rotations[index];
        if (
            rotation.block == weight.rotation_block()
            && rotation.seed == weight.rotation_seed()
        ) {
            if (rotation.sign_values != values) {
                throw std::runtime_error(
                    "conflicting MFE HSG1 sign "
                    "vectors share one rotation key");
            }
            return checked_int(
                index + 1,
                "rotation variant");
        }
    }
    rotations.push_back({
        weight.rotation_signs(),
        std::move(values),
        weight.rotation_block(),
        weight.rotation_seed(),
    });
    return checked_int(
        rotations.size(),
        "rotation variant");
}

int rotation_variant(
    const RotationSpec& source,
    std::vector<RotationSpec>& rotations) {
    for (
        std::size_t index = 0;
        index < rotations.size();
        ++index
    ) {
        const auto& rotation = rotations[index];
        if (
            rotation.block == source.block
            && rotation.seed == source.seed
        ) {
            if (
                rotation.sign_values
                != source.sign_values
            ) {
                throw std::runtime_error(
                    "conflicting MFE HSG1 sign "
                    "vectors share one rotation key");
            }
            return checked_int(
                index + 1,
                "rotation variant");
        }
    }
    rotations.push_back(source);
    return checked_int(
        rotations.size(),
        "rotation variant");
}

MlxVqWeight add_vq_pool(
    std::string_view dtype,
    std::span<const std::uint8_t> payload,
    std::span<const std::uint8_t> runtime,
    const std::vector<std::int32_t>& expert_ids,
    int out_per_expert,
    int neuron_len,
    PackedStreams& streams,
    std::vector<RotationSpec>& rotations,
    std::vector<std::int32_t>& descriptors,
    bool pack_execution = true) {
    const auto expected_rows = checked_product(
        expert_ids.size(),
        static_cast<std::size_t>(out_per_expert),
        "VQ cohort row count");
    const int checked_rows = checked_int(
        expected_rows,
        "VQ cohort row count");
    const auto metadata = inspect_vq_blob(
        dtype,
        payload,
        runtime);
    const auto& output_shape = metadata.output_shape;
    const bool cross_expert =
        output_shape.size() == 2;
    if (
        metadata.output_size != checked_rows
        || metadata.input_size != neuron_len
        || (
            cross_expert
            && (
                output_shape[0]
                    != static_cast<int>(
                        expert_ids.size())
                || output_shape[1]
                    != out_per_expert
            )
        )
        || (
            !cross_expert
            && (
                output_shape.size() != 1
                || output_shape.front()
                    != checked_rows
            )
        )
    ) {
        throw std::runtime_error(
            "MFE VQ cohort shape is inconsistent");
    }
    // Every supported VQ profile stores one FP16 anchor per flattened output
    // row.  Reject impossible dimensions before the full parser allocates
    // canonical execution arrays from an attacker-controlled header.
    if (
        expected_rows
        > payload.size() / sizeof(std::uint16_t)
    ) {
        throw std::runtime_error(
            "MFE VQ cohort dimensions exceed "
            "its payload");
    }
    auto weight = MlxVqWeight::from_blob(
        dtype,
        payload,
        runtime);
    if (
        weight.output_size() != metadata.output_size
        || weight.input_size() != metadata.input_size
        || weight.output_shape() != metadata.output_shape
        || weight.rotation_block()
            != metadata.rotation_block
        || weight.rotation_seed()
            != metadata.rotation_seed
    ) {
        throw std::runtime_error(
            "MFE VQ header/full parse mismatch");
    }
    if (!pack_execution) {
        return weight;
    }

    // CUDA's NVQ2 execution layout merges one byte index and one byte sign
    // mask.  The same idea also applies to NVQ3J by placing its two D4
    // indices beside the parity-completed sign mask.  This trades at most one
    // padding bit per eight weights for one sequential read in the decode
    // kernel instead of independent packed index and 7-bit auxiliary reads.
    const char* jsc_exec_env = std::getenv(
        "MFQ_METAL_MFE_JSC_EXEC");
    const auto& profile = weight.format_label();
    const bool packed_jsc_execution =
        (profile == "NVQ2J" || profile == "NVQ3J")
        && (
            jsc_exec_env == nullptr
            || std::string_view(jsc_exec_env) != "0"
        );
    const bool group64_execution =
        profile == "NVQ2J-XL"
        && weight.execution_layout() == 1;
    const bool jsc_execution =
        packed_jsc_execution || group64_execution;
    detail::StagingVector<std::uint8_t>
        jsc_execution_indices;
    if (packed_jsc_execution) {
        auto packed_indices = weight.packed_indices();
        auto packed_aux = weight.packed_auxiliary();
        packed_indices.eval();
        packed_aux.eval();
        const auto* index_data =
            packed_indices.data<std::uint8_t>();
        const auto* aux_data =
            packed_aux.data<std::uint8_t>();
        const int signs = (neuron_len + 7) / 8;
        const int bytes_per_sign =
            weight.vector_size() == 4 ? 3 : 2;
        const auto execution_bytes = checked_product(
            checked_product(
                expected_rows,
                static_cast<std::size_t>(signs),
                "VQ JSC execution stream size"),
            static_cast<std::size_t>(bytes_per_sign),
            "VQ JSC execution stream size");
        jsc_execution_indices.resize(
            execution_bytes);
        const auto read_aux = [&](std::size_t linear) {
            const auto bit = linear * 7;
            const auto byte = bit >> 3;
            const int shift = bit & 7;
            std::uint32_t packed = aux_data[byte];
            if (byte + 1 < packed_aux.size()) {
                packed |= static_cast<std::uint32_t>(
                    aux_data[byte + 1]) << 8;
            }
            return (packed >> shift) & 0x7fu;
        };
        const auto pack_rows = [&](std::size_t begin, std::size_t end) {
            for (std::size_t row = begin; row < end; ++row) {
                for (int sign = 0; sign < signs; ++sign) {
                    const auto source_sign =
                        row * signs + sign;
                    const auto target =
                        source_sign * bytes_per_sign;
                    const auto first_vector =
                        static_cast<std::size_t>(
                            sign)
                        * (weight.vector_size() == 4 ? 2 : 1);
                    jsc_execution_indices[target] =
                        index_data[
                            row * weight.vectors()
                            + first_vector];
                    if (weight.vector_size() == 4) {
                        jsc_execution_indices[target + 1] =
                            index_data[
                                row * weight.vectors()
                                + first_vector + 1];
                    }
                    const auto mask7 = read_aux(
                        source_sign);
                    const auto parity =
                        std::popcount(mask7) & 1u;
                    jsc_execution_indices[
                        target + bytes_per_sign - 1] =
                        static_cast<std::uint8_t>(
                            mask7 | (parity << 7));
                }
            }
        };
        constexpr std::size_t kParallelThreshold =
            8u * 1024u * 1024u;
        const auto worker_count = execution_bytes
                >= kParallelThreshold
            ? std::min<std::size_t>(8, expected_rows)
            : 1;
        if (worker_count == 1) {
            pack_rows(0, expected_rows);
        } else {
            std::vector<std::thread> workers;
            workers.reserve(worker_count);
            for (
                std::size_t worker = 0;
                worker < worker_count;
                ++worker
            ) {
                const auto begin =
                    expected_rows * worker / worker_count;
                const auto end =
                    expected_rows * (worker + 1) / worker_count;
                workers.emplace_back(pack_rows, begin, end);
            }
            for (auto& worker : workers) {
                worker.join();
            }
        }
    }

    if (group64_execution) {
        while ((streams.vq_indices.size() & 7u) != 0u) {
            streams.vq_indices.push_back(0);
        }
    }
    const int indices_offset = checked_int(
        streams.vq_indices.size(),
        "VQ index offset");
    const int state_offset = checked_int(
        streams.vq_state.size(),
        "VQ state offset");
    const int aux_offset = checked_int(
        streams.vq_aux.size(),
        "VQ auxiliary offset");
    const int anchor_offset = checked_int(
        streams.vq_anchors.size() / sizeof(float),
        "VQ anchor offset");
    const int codebook_offset = checked_int(
        streams.vq_codebooks.size(),
        "VQ codebook offset");
    const int scale_offset = checked_int(
        streams.vq_scales.size() / sizeof(float),
        "VQ scale offset");
    const int state_bank_offset = checked_int(
        streams.vq_state_to_codebank.size(),
        "VQ state-bank offset");
    const int bank_offset = checked_int(
        streams.vq_banks.size(),
        "VQ bank offset");
    const int parameter_offset = checked_int(
        streams.vq_parameters.size()
            / sizeof(float),
        "VQ parameter offset");
    const int residual_codebook_offset = checked_int(
        streams.vq_residual_codebooks.size()
            / sizeof(float),
        "VQ residual codebook offset");
    const int residual_record_offset = checked_int(
        streams.vq_residual_first.size()
            / sizeof(std::int16_t),
        "VQ residual record offset");
    const int rotation =
        rotation_variant(weight, rotations);

    for (
        std::size_t local_expert = 0;
        local_expert < expert_ids.size();
        ++local_expert
    ) {
        const int expert = expert_ids[local_expert];
        const auto base = checked_product(
            static_cast<std::size_t>(expert),
            static_cast<std::size_t>(kDescriptorSize),
            "descriptor offset");
        descriptors[base + kFamily] = kFamilyVq;
        descriptors[base + kLocalExpert] =
            checked_int(local_expert, "local expert");
        descriptors[base + kOut] = out_per_expert;
        descriptors[base + kInput] = neuron_len;
        descriptors[base + kVqGroupSize] =
            weight.group_size();
        descriptors[base + kVqGroups] =
            weight.groups();
        descriptors[base + kVqVectorSize] =
            weight.vector_size();
        descriptors[base + kVqVectors] =
            weight.vectors();
        descriptors[base + kVqIndexBits] =
            weight.index_bits();
        descriptors[base + kVqStateBits] =
            weight.state_bits();
        descriptors[base + kVqStates] =
            weight.states();
        descriptors[base + kVqEntries] =
            weight.entries();
        descriptors[base + kVqCodeBanks] =
            weight.code_banks();
        descriptors[base + kVqAuxMode] =
            weight.aux_mode();
        descriptors[base + kVqCodeBankMode] =
            weight.code_bank_mode();
        descriptors[base + kVqHasTableBanks] =
            static_cast<int>(
                weight.table_banks() > 1);
        descriptors[base + kVqGroupsPerSuper] =
            weight.groups_per_supergroup();
        descriptors[base + kVqSupergroups] =
            weight.supergroups();
        descriptors[base + kVqIndicesOffset] =
            indices_offset;
        descriptors[base + kVqStateOffset] =
            state_offset;
        descriptors[base + kVqAuxOffset] =
            aux_offset;
        descriptors[base + kVqAnchorOffset] =
            anchor_offset;
        descriptors[base + kVqCodebookOffset] =
            codebook_offset;
        descriptors[base + kVqScaleOffset] =
            scale_offset;
        descriptors[base + kVqStateBankOffset] =
            state_bank_offset;
        descriptors[base + kVqBankOffset] =
            bank_offset;
        descriptors[base + kVqParameterOffset] =
            parameter_offset;
        descriptors[base + kVqRotationVariant] =
            rotation;
        const auto& profile = weight.format_label();
        const int execution_profile =
            profile == "NVQ2J-L" || profile == "NVQ2J-XL"
                ? kVqProfileJscExtended8
                : profile == "NVQ2J" || profile == "NVQ3J"
                ? (
                    weight.vector_size() == 4
                        ? kVqProfileJsc4
                        : kVqProfileJsc8
                )
                : (
                    profile == "NPQ0-S" || profile == "NPQ0-L"
                        ? (
                            profile == "NPQ0-S"
                                ? kVqProfileNpqS
                                : kVqProfileNpqL
                        )
                        : (
                            profile == "NVQ1-L"
                                    && weight.group_size() == 24
                                    && weight.vector_size() == 8
                                    && weight.index_bits() == 11
                                ? kVqProfileNvq1L
                                : (
                                    profile == "NVQ1-S"
                                            && weight.group_size() == 24
                                            && weight.vector_size() == 8
                                            && weight.index_bits() == 9
                                        ? kVqProfileNvq1S
                                        : kVqProfileGeneric
                                )
                        )
                );
        descriptors[base + kVqProfile] =
            execution_profile
            | (weight.residual_position_bits() << 8)
            | (weight.residual_block_vectors() << 16);
        descriptors[base + kVqJscExecution] =
            group64_execution
                ? 2
                : static_cast<int>(packed_jsc_execution);
        descriptors[base + kVqResidualCodebookOffset] =
            residual_codebook_offset;
        descriptors[base + kVqResidualRecordOffset] =
            residual_record_offset;
    }

    if (group64_execution) {
        append_raw(
            streams.vq_indices,
            weight.packed_indices(),
            mlx::core::uint8,
            "VQ group64 values");
    } else if (packed_jsc_execution) {
        streams.vq_indices.insert(
            streams.vq_indices.end(),
            jsc_execution_indices.begin(),
            jsc_execution_indices.end());
    } else {
        append_raw(
            streams.vq_indices,
            weight.packed_indices(),
            mlx::core::uint8,
            "VQ indices");
    }
    append_raw(
        streams.vq_state,
        weight.packed_states(),
        mlx::core::uint8,
        "VQ states");
    if (!jsc_execution) {
        append_raw(
            streams.vq_aux,
            weight.packed_auxiliary(),
            mlx::core::uint8,
            "VQ auxiliary values");
    }
    append_raw(
        streams.vq_anchors,
        weight.anchors(),
        mlx::core::float32,
        "VQ anchors");
    append_raw(
        streams.vq_codebooks,
        weight.codebooks(),
        mlx::core::int8,
        "VQ codebooks");
    append_raw(
        streams.vq_scales,
        weight.scale_lut(),
        mlx::core::float32,
        "VQ scales");
    append_raw(
        streams.vq_state_to_codebank,
        weight.state_to_codebank(),
        mlx::core::uint8,
        "VQ state banks");
    append_raw(
        streams.vq_banks,
        weight.bank_ids(),
        mlx::core::uint8,
        "VQ table banks");
    append_raw(
        streams.vq_parameters,
        weight.parameters(),
        mlx::core::float32,
        "VQ parameters");
    if (weight.residual_position_bits() != 0) {
        append_raw(
            streams.vq_residual_codebooks,
            weight.residual_codebook(),
            mlx::core::float32,
            "VQ residual codebooks");
        append_raw(
            streams.vq_residual_first,
            weight.residual_first(),
            mlx::core::int16,
            "VQ first residual records");
        append_raw(
            streams.vq_residual_second,
            weight.residual_second(),
            mlx::core::int16,
            "VQ second residual records");
    }
    return weight;
}

bool is_ascii(
    std::span<const std::uint8_t> value) {
    return std::all_of(
        value.begin(),
        value.end(),
        [](std::uint8_t byte) {
            return byte <= 0x7fu;
        });
}

array concatenate_1d(
    std::vector<array> values) {
    const bool all_vectors = std::all_of(
        values.begin(),
        values.end(),
        [](const array& value) {
            return value.ndim() == 1;
        });
    if (!all_vectors) {
        if (values.empty()) {
            throw std::invalid_argument(
                "cannot concatenate an empty packed stream list");
        }
        const auto dtype = values.front().dtype();
        std::size_t total_bytes = 0;
        for (auto& value : values) {
            if (value.dtype() != dtype || !value.flags().row_contiguous) {
                throw std::invalid_argument(
                    "packed stream concatenation requires contiguous matching dtypes");
            }
            value.eval();
            total_bytes = checked_add(
                total_bytes,
                value.nbytes(),
                "packed concatenation bytes");
        }
        detail::StagingVector<std::uint8_t> bytes(total_bytes);
        std::size_t offset = 0;
        for (const auto& value : values) {
            std::memcpy(
                bytes.data() + offset,
                value.data<std::uint8_t>(),
                value.nbytes());
            offset += value.nbytes();
        }
        return make_raw_array(std::move(bytes), dtype);
    }
    return mlx::core::contiguous(
        mlx::core::concatenate(
            std::move(values),
            0));
}

array apply_rotation(
    const array& source,
    const RotationSpec& rotation,
    int rows,
    int width) {
    if (
        source.ndim() != 2
        || source.shape(0) != rows
        || source.shape(1) != width
        || rotation.block <= 0
        || rotation.block > 8192
        || (
            rotation.block
            & (rotation.block - 1)
        ) != 0
        || width % rotation.block != 0
        || rotation.signs.dtype()
            != mlx::core::int8
        || rotation.signs.size()
            != static_cast<std::size_t>(width)
    ) {
        throw std::runtime_error(
            "invalid MFE HSG1 rotation layout");
    }
    const auto grid = checked_product(
        static_cast<std::size_t>(rows),
        256,
        "HSG1 Metal grid");
    if (
        grid
        > static_cast<std::size_t>(
            std::numeric_limits<int>::max())
    ) {
        throw std::runtime_error(
            "MFE HSG1 Metal grid exceeds MLX limits");
    }
    auto outputs = moe_hadamard_kernel()(
        {source, rotation.signs},
        {Shape{rows, width}},
        {source.dtype()},
        {
            static_cast<int>(grid),
            1,
            1,
        },
        {256, 1, 1},
        {
            {"T", source.dtype()},
            {"M", rows},
            {"K", width},
            {"BLOCK", rotation.block},
        },
        std::nullopt,
        false,
        {});
    return std::move(outputs.front());
}

int descriptor_with_offset(
    int value,
    std::size_t offset,
    const char* name) {
    if (value < 0) {
        throw std::runtime_error(
            std::string("invalid MFE ") + name);
    }
    return checked_int(
        checked_add(
            static_cast<std::size_t>(value),
            offset,
            name),
        name);
}

std::int32_t descriptor_u32_with_offset(
    std::int32_t value,
    std::size_t offset,
    const char* name) {
    return checked_descriptor_u32(
        checked_add(
            static_cast<std::size_t>(
                descriptor_u32_value(value)),
            offset,
            name),
        name);
}

std::uint64_t checked_range_add(
    std::uint64_t left,
    std::uint64_t right,
    const char* name) {
    if (
        left
        > std::numeric_limits<std::uint64_t>::max()
            - right
    ) {
        throw std::runtime_error(
            std::string("streamed MFE ") + name
            + " overflows");
    }
    return left + right;
}

std::uint64_t checked_range_product(
    std::uint64_t left,
    std::uint64_t right,
    const char* name) {
    if (
        left != 0
        && right
            > std::numeric_limits<std::uint64_t>::max()
                / left
    ) {
        throw std::runtime_error(
            std::string("streamed MFE ") + name
            + " overflows");
    }
    return left * right;
}

template <typename T>
T read_scalar(
    const std::vector<std::uint8_t>& bytes,
    std::size_t offset,
    const char* name) {
    if (
        offset > bytes.size()
        || sizeof(T) > bytes.size() - offset
    ) {
        throw std::runtime_error(
            std::string("truncated streamed MFE ") + name);
    }
    T value{};
    std::memcpy(
        &value,
        bytes.data() + offset,
        sizeof(T));
    return value;
}

std::string read_ascii(
    const std::vector<std::uint8_t>& bytes,
    std::size_t offset,
    std::size_t count,
    const char* name) {
    if (
        offset > bytes.size()
        || count > bytes.size() - offset
    ) {
        throw std::runtime_error(
            std::string("truncated streamed MFE ") + name);
    }
    const auto begin =
        bytes.begin()
        + static_cast<std::ptrdiff_t>(offset);
    const auto end =
        begin + static_cast<std::ptrdiff_t>(count);
    if (
        std::any_of(
            begin,
            end,
            [](std::uint8_t value) {
                return value > 0x7fu;
            })
    ) {
        throw std::runtime_error(
            std::string("streamed MFE ") + name
            + " is not ASCII");
    }
    return {
        reinterpret_cast<const char*>(
            bytes.data() + offset),
        count,
    };
}

std::uint32_t read_packed(
    const std::vector<std::uint8_t>& bytes,
    std::size_t bit_offset,
    int bits) {
    const auto byte_offset = bit_offset / 8;
    const auto shift =
        static_cast<unsigned>(bit_offset & 7);
    std::uint32_t packed = 0;
    for (unsigned index = 0; index < 3; ++index) {
        const auto source = byte_offset + index;
        if (source < bytes.size()) {
            packed |=
                static_cast<std::uint32_t>(
                    bytes[source])
                << (8 * index);
        }
    }
    return (
        packed >> shift
    ) & ((std::uint32_t{1} << bits) - 1u);
}

void write_packed(
    std::vector<std::uint8_t>& bytes,
    std::size_t value_index,
    int bits,
    std::uint32_t value) {
    const auto bit_offset =
        value_index * static_cast<std::size_t>(bits);
    for (int bit = 0; bit < bits; ++bit) {
        if (((value >> bit) & 1u) == 0u) {
            continue;
        }
        const auto target =
            bit_offset
            + static_cast<std::size_t>(bit);
        bytes[target / 8] |=
            static_cast<std::uint8_t>(
                1u << (target & 7));
    }
}

class MfeStreamUnsupported final
    : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

template <typename T>
void append_scalar_bytes(
    std::vector<std::uint8_t>& target,
    T value) {
    const auto previous = target.size();
    target.resize(previous + sizeof(T));
    std::memcpy(target.data() + previous, &value, sizeof(T));
}

template <typename T>
void patch_scalar_bytes(
    std::vector<std::uint8_t>& target,
    std::size_t offset,
    T value,
    const char* name) {
    if (offset > target.size() || sizeof(T) > target.size() - offset) {
        throw std::runtime_error(
            std::string("MFE streamed ") + name + " patch is out of range");
    }
    std::memcpy(target.data() + offset, &value, sizeof(T));
}

std::vector<std::uint8_t> unpack_small_selectors(
    const std::vector<std::uint8_t>& packed,
    std::size_t count,
    int bits,
    const char* name) {
    std::vector<std::uint8_t> result(count);
    for (std::size_t index = 0; index < count; ++index) {
        const auto value = read_packed(
            packed,
            index * static_cast<std::size_t>(bits),
            bits);
        if (value >= (std::uint32_t{1} << bits)) {
            throw std::runtime_error(
                std::string("invalid streamed MFE ") + name);
        }
        result[index] = static_cast<std::uint8_t>(value);
    }
    return result;
}

std::vector<std::uint8_t> pack_small_selectors(
    std::span<const std::uint8_t> values,
    int bits) {
    std::vector<std::uint8_t> result(
        checked_packed_size(values.size(), bits, "selector bytes"),
        0);
    for (std::size_t index = 0; index < values.size(); ++index) {
        write_packed(result, index, bits, values[index]);
    }
    return result;
}

std::vector<std::uint8_t> read_packed_value_slice(
    const MfqContainer& model,
    const std::string& name,
    std::uint64_t stream_offset,
    std::size_t first_value,
    std::size_t value_count,
    int bits) {
    if (value_count == 0) {
        return {};
    }
    const auto source_bit = checked_product(
        first_value,
        static_cast<std::size_t>(bits),
        "streamed source bit offset");
    const auto bit_count = checked_product(
        value_count,
        static_cast<std::size_t>(bits),
        "streamed value bit count");
    if ((source_bit & 7u) == 0u && (bit_count & 7u) == 0u) {
        return model.read_range(
            name,
            checked_range_add(
                stream_offset,
                source_bit / 8,
                "streamed aligned source offset"),
            bit_count / 8);
    }
    const auto source_byte = source_bit / 8;
    const auto shift = source_bit & 7u;
    const auto source_bytes = checked_add(
        checked_add(shift, bit_count, "streamed source bits"),
        7,
        "streamed source rounding") / 8;
    const auto raw = model.read_range(
        name,
        checked_range_add(
            stream_offset,
            source_byte,
            "streamed packed source offset"),
        source_bytes);
    std::vector<std::uint8_t> result(
        checked_packed_size(value_count, bits, "streamed packed slice"),
        0);
    for (std::size_t index = 0; index < value_count; ++index) {
        write_packed(
            result,
            index,
            bits,
            read_packed(
                raw,
                shift + index * static_cast<std::size_t>(bits),
                bits));
    }
    return result;
}

void append_record_range(
    std::vector<std::uint8_t>& target,
    const MfqContainer& model,
    const std::string& name,
    std::uint64_t offset,
    std::uint64_t nbytes) {
    const auto bytes = model.read_range(name, offset, nbytes);
    target.insert(target.end(), bytes.begin(), bytes.end());
}

struct MfeNintStreamLayout {
    int rows = 0;
    int groups = 0;
    int group_size = 0;
    int q_bits = 0;
    int nominal_sub_bits = 0;
    bool adaptive_storage = false;
    bool legacy_unpacked = false;
    std::uint64_t payload_offset = 0;
    std::uint64_t anchor_scale_offset = 0;
    std::uint64_t anchor_min_offset = 0;
    std::uint64_t legacy_sub_scale_offset = 0;
    std::uint64_t legacy_sub_min_offset = 0;
    std::uint64_t legacy_q_offset = 0;
    std::uint64_t k_selector_offset = 0;
    std::array<std::uint64_t, 4> sub_scale_offsets{};
    std::array<std::uint64_t, 4> sub_min_offsets{};
    std::uint64_t q_selector_offset = 0;
    std::array<std::uint64_t, 8> q_offsets{};
    std::vector<std::uint8_t> k_selectors;
    std::vector<std::uint8_t> q_selectors;
};

struct MfeNvqJscStreamLayout {
    int rows = 0;
    int groups = 0;
    int vectors = 0;
    int signs = 0;
    int state_bits = 0;
    int index_bits = 0;
    bool group64 = false;
    std::uint64_t payload_offset = 0;
    std::uint64_t prefix_bytes = 0;
    std::uint64_t anchors_offset = 0;
    std::uint64_t state_offset = 0;
    std::uint64_t indices_offset = 0;
    std::uint64_t signs_offset = 0;
};

struct MfeMxStreamLayout {
    int rows = 0;
    int columns = 0;
    int bits = 0;
    std::uint64_t payload_offset = 0;
    std::uint64_t values_offset = 0;
    std::uint64_t scales_offset = 0;
};

using MfeStreamLayout = std::variant<
    MfeNintStreamLayout,
    MfeNvqJscStreamLayout,
    MfeMxStreamLayout>;

struct MfeStreamPool {
    std::string dtype;
    int expert_count = 0;
    std::uint64_t runtime_offset = 0;
    std::uint64_t runtime_bytes = 0;
    std::uint64_t payload_offset = 0;
    std::uint64_t payload_bytes = 0;
    MfeStreamLayout layout;
};

struct MfeStreamExpertLocation {
    std::shared_ptr<const MfeStreamPool> pool;
    int local_expert = 0;
};

struct MfeStreamProjection {
    int experts = 0;
    int out_per_expert = 0;
    int neuron_len = 0;
    std::vector<std::optional<MfeStreamExpertLocation>> experts_by_id;
};

MfeNintStreamLayout parse_streamed_nint_layout(
    const MfqContainer& model,
    const std::string& name,
    std::uint64_t payload_offset,
    std::uint64_t payload_bytes,
    int expected_rows,
    int expected_columns) {
    constexpr std::uint64_t kHeaderBytes = 42;
    if (payload_bytes < kHeaderBytes) {
        throw std::runtime_error("truncated streamed NINT payload: " + name);
    }
    const auto header = model.read_range(
        name, payload_offset, kHeaderBytes);
    const int raw_bits = header[0];
    const int nominal_sub_bits = header[1];
    const auto group_size = read_scalar<std::int32_t>(
        header, 2, "NINT group size");
    const auto axis = read_scalar<std::int32_t>(
        header, 6, "NINT axis");
    const auto columns = read_scalar<std::int32_t>(
        header, 10, "NINT width");
    const auto dimensions = read_scalar<std::uint32_t>(
        header, 14, "NINT dimensions");
    const auto shape_rows = read_scalar<std::int64_t>(
        header, 18, "NINT rows");
    const auto shape_columns = read_scalar<std::int64_t>(
        header, 26, "NINT columns");
    const auto rows = read_scalar<std::uint32_t>(
        header, 34, "NINT output size");
    const auto groups = read_scalar<std::uint32_t>(
        header, 38, "NINT group count");
    const bool adaptive_storage = (raw_bits & 0x80) != 0;
    const int q_bits = raw_bits & 0x7f;
    if (q_bits < 1
        || q_bits > 8 || nominal_sub_bits < 1
        || nominal_sub_bits > 8 || group_size <= 0 || axis != 0
        || dimensions != 2 || columns != expected_columns
        || shape_rows != expected_rows || shape_columns != expected_columns
        || rows != static_cast<std::uint32_t>(expected_rows)
        || groups != static_cast<std::uint32_t>(
            (expected_columns + group_size - 1) / group_size)) {
        throw std::runtime_error(
            "unsupported streamed NINTv2 geometry: " + name);
    }

    MfeNintStreamLayout result;
    result.rows = expected_rows;
    result.groups = static_cast<int>(groups);
    result.group_size = group_size;
    result.q_bits = q_bits;
    result.nominal_sub_bits = nominal_sub_bits;
    result.adaptive_storage = adaptive_storage;
    result.payload_offset = payload_offset;
    result.anchor_scale_offset = payload_offset + kHeaderBytes;
    result.anchor_min_offset = result.anchor_scale_offset
        + static_cast<std::uint64_t>(expected_rows) * 2;
    std::uint64_t cursor = result.anchor_min_offset
        + static_cast<std::uint64_t>(expected_rows) * 2;
    if (!adaptive_storage) {
        const auto metadata_count = checked_product(
            static_cast<std::size_t>(expected_rows),
            static_cast<std::size_t>(groups),
            "legacy NINT metadata count");
        const auto values_per_row = checked_product(
            static_cast<std::size_t>(groups),
            static_cast<std::size_t>(group_size),
            "legacy NINT values per row");
        const auto q_count = checked_product(
            static_cast<std::size_t>(expected_rows),
            values_per_row,
            "legacy NINT q count");
        const auto packed_metadata_bytes = checked_packed_size(
            metadata_count, nominal_sub_bits,
            "legacy NINT metadata bytes");
        const auto packed_q_bytes = checked_packed_size(
            q_count, q_bits, "legacy NINT q bytes");
        const auto packed_tail = checked_add(
            checked_product(
                packed_metadata_bytes, std::size_t{2},
                "legacy NINT packed metadata"),
            packed_q_bytes,
            "legacy NINT packed tail");
        const auto unpacked_tail = checked_add(
            checked_product(
                metadata_count, std::size_t{2},
                "legacy NINT unpacked metadata"),
            q_count,
            "legacy NINT unpacked tail");
        const auto payload_end = checked_range_add(
            payload_offset, payload_bytes, "legacy NINT payload end");
        if (cursor > payload_end) {
            throw std::runtime_error(
                "truncated streamed legacy NINT payload: " + name);
        }
        const auto remaining = payload_end - cursor;
        if (remaining == unpacked_tail) {
            result.legacy_unpacked = true;
            result.legacy_sub_scale_offset = cursor;
            result.legacy_sub_min_offset = cursor + metadata_count;
            result.legacy_q_offset =
                result.legacy_sub_min_offset + metadata_count;
        } else if (remaining == packed_tail) {
            result.legacy_sub_scale_offset = cursor;
            result.legacy_sub_min_offset = cursor + packed_metadata_bytes;
            result.legacy_q_offset =
                result.legacy_sub_min_offset + packed_metadata_bytes;
        } else {
            throw std::runtime_error(
                "invalid streamed legacy NINT payload length: " + name);
        }
        return result;
    }
    const auto k_selector_bytes = checked_packed_size(
        static_cast<std::size_t>(expected_rows), 2,
        "NINTv2 k selectors");
    result.k_selector_offset = cursor;
    result.k_selectors = unpack_small_selectors(
        model.read_range(name, cursor, k_selector_bytes),
        static_cast<std::size_t>(expected_rows),
        2,
        "NINTv2 k selector");
    cursor += k_selector_bytes;
    for (int selector = 0; selector < 4; ++selector) {
        const int row_bits = nominal_sub_bits - 1 + selector;
        const auto selected = static_cast<std::size_t>(std::count(
            result.k_selectors.begin(), result.k_selectors.end(),
            static_cast<std::uint8_t>(selector)));
        if (selected != 0 && (row_bits < 1 || row_bits > 8)) {
            throw std::runtime_error(
                "invalid streamed NINTv2 subgroup width: " + name);
        }
        const auto bytes = checked_packed_size(
            checked_product(selected, static_cast<std::size_t>(groups),
                "NINTv2 subgroup values"),
            std::max(row_bits, 1),
            "NINTv2 subgroup bytes");
        result.sub_scale_offsets[selector] = cursor;
        cursor += bytes;
        result.sub_min_offsets[selector] = cursor;
        cursor += bytes;
    }
    const auto q_selector_bytes = checked_packed_size(
        static_cast<std::size_t>(expected_rows), 3,
        "NINTv2 q selectors");
    result.q_selector_offset = cursor;
    result.q_selectors = unpack_small_selectors(
        model.read_range(name, cursor, q_selector_bytes),
        static_cast<std::size_t>(expected_rows),
        3,
        "NINTv2 q selector");
    cursor += q_selector_bytes;
    const auto values_per_row = checked_product(
        static_cast<std::size_t>(groups),
        static_cast<std::size_t>(group_size),
        "NINTv2 values per row");
    for (int selector = 0; selector < 8; ++selector) {
        const int row_bits = selector + 1;
        const auto selected = static_cast<std::size_t>(std::count(
            result.q_selectors.begin(), result.q_selectors.end(),
            static_cast<std::uint8_t>(selector)));
        result.q_offsets[selector] = cursor;
        cursor += checked_packed_size(
            checked_product(selected, values_per_row,
                "NINTv2 selected values"),
            row_bits,
            "NINTv2 q bytes");
    }
    if (cursor != payload_offset + payload_bytes) {
        throw std::runtime_error(
            "invalid streamed NINTv2 payload length: " + name);
    }
    return result;
}

MfeNvqJscStreamLayout parse_streamed_nvq_jsc_layout(
    const MfqContainer& model,
    const std::string& name,
    std::uint64_t payload_offset,
    std::uint64_t payload_bytes,
    int expected_rows,
    int expected_columns) {
    constexpr std::uint64_t kMatrixHeaderBytes = 40;
    constexpr std::uint64_t kJscHeaderBytes = 64;
    if (payload_bytes < kMatrixHeaderBytes + kJscHeaderBytes) {
        throw std::runtime_error("truncated streamed NVQ-JSC payload: " + name);
    }
    const auto header = model.read_range(
        name, payload_offset, kMatrixHeaderBytes + kJscHeaderBytes);
    const std::string_view magic(
        reinterpret_cast<const char*>(header.data()), 4);
    const int profile_flags = header[4];
    const int profile = profile_flags & ~(0x80 | 0x40 | 0x20);
    const int state_bits = header[5];
    const auto group_size = read_scalar<std::uint16_t>(
        header, 6, "NVQ group size");
    const auto axis = read_scalar<std::int32_t>(
        header, 8, "NVQ axis");
    const auto columns = read_scalar<std::int32_t>(
        header, 12, "NVQ width");
    const auto dimensions = read_scalar<std::uint32_t>(
        header, 16, "NVQ dimensions");
    const auto shape_rows = read_scalar<std::int64_t>(
        header, 20, "NVQ rows");
    const auto shape_columns = read_scalar<std::int64_t>(
        header, 28, "NVQ columns");
    const auto rows = read_scalar<std::uint32_t>(
        header, 36, "NVQ output size");
    const int vector_size = profile == 1 || profile == 4 || profile == 5
        ? 8 : profile == 2 || profile == 3 || profile == 6 ? 4 : 0;
    const int index_bits = profile == 1 || profile == 2 ? 8
        : profile == 3 ? 9
        : profile == 4 || profile == 6 ? 10
        : profile == 5 ? 12 : 0;
    const int banks = header[kMatrixHeaderBytes + 1];
    const int states = header[kMatrixHeaderBytes + 2];
    const int storage_layout = header[kMatrixHeaderBytes + 52];
    if ((magic != "NVQ1" && magic != "NIQ1")
        || (profile_flags & 0x20) == 0
        || (profile_flags & (0x80 | 0x40)) != 0
        || state_bits != 4 || group_size != 24 || axis != 0
        || dimensions != 2 || columns != expected_columns
        || shape_rows != expected_rows || shape_columns != expected_columns
        || rows != static_cast<std::uint32_t>(expected_rows)
        || vector_size == 0 || index_bits == 0
        || (banks != 1 && banks != 2 && banks != 4)
        || states != 16 || (storage_layout != 0 && storage_layout != 1)) {
        throw MfeStreamUnsupported(
            "MFE VQ cohort is not a streamable NVQ-JSC layout");
    }
    const auto entries = std::uint64_t{1} << index_bits;
    const auto codebook_bytes = checked_range_product(
        checked_range_product(
            static_cast<std::uint64_t>(banks), entries,
            "NVQ-JSC codebook entries"),
        static_cast<std::uint64_t>(vector_size),
        "NVQ-JSC codebook bytes");
    MfeNvqJscStreamLayout result;
    result.rows = expected_rows;
    result.groups = (expected_columns + group_size - 1) / group_size;
    result.vectors = (expected_columns + vector_size - 1) / vector_size;
    result.signs = (expected_columns + 7) / 8;
    result.state_bits = state_bits;
    result.index_bits = index_bits;
    result.group64 = storage_layout == 1;
    result.payload_offset = payload_offset;
    result.prefix_bytes = checked_range_add(
        kMatrixHeaderBytes + kJscHeaderBytes,
        codebook_bytes,
        "NVQ-JSC prefix bytes");
    result.anchors_offset = payload_offset + result.prefix_bytes;
    result.state_offset = result.anchors_offset
        + static_cast<std::uint64_t>(expected_rows) * 2;
    if (result.group64) {
        result.indices_offset = result.state_offset;
        const auto records = checked_range_product(
            static_cast<std::uint64_t>(expected_rows),
            static_cast<std::uint64_t>(result.groups),
            "NVQ-JSC group64 records");
        result.signs_offset = checked_range_add(
            result.indices_offset,
            checked_range_product(records, 8, "NVQ-JSC group64 bytes"),
            "NVQ-JSC group64 end");
    } else {
        const auto state_bytes = checked_packed_size(
            checked_product(
                static_cast<std::size_t>(expected_rows),
                static_cast<std::size_t>(result.groups),
                "NVQ-JSC state count"),
            state_bits,
            "NVQ-JSC state bytes");
        result.indices_offset = result.state_offset + state_bytes;
        const auto index_bytes = checked_packed_size(
            checked_product(
                static_cast<std::size_t>(expected_rows),
                static_cast<std::size_t>(result.vectors),
                "NVQ-JSC index count"),
            index_bits,
            "NVQ-JSC index bytes");
        result.signs_offset = result.indices_offset + index_bytes;
    }
    const auto end = result.group64
        ? result.signs_offset
        : result.signs_offset + checked_packed_size(
              checked_product(
                  static_cast<std::size_t>(expected_rows),
                  static_cast<std::size_t>(result.signs),
                  "NVQ-JSC sign count"),
              7,
              "NVQ-JSC sign bytes");
    if (end != payload_offset + payload_bytes) {
        throw std::runtime_error(
            "invalid streamed NVQ-JSC payload length: " + name);
    }
    return result;
}

MfeMxStreamLayout parse_streamed_mx_layout(
    const MfqContainer& model,
    const std::string& name,
    std::uint64_t payload_offset,
    std::uint64_t payload_bytes,
    int expected_rows,
    int expected_columns,
    int expected_bits) {
    constexpr std::uint64_t kHeaderBytes = 56;
    if (payload_bytes < kHeaderBytes) {
        throw std::runtime_error("truncated streamed MX payload: " + name);
    }
    const auto header = model.read_range(
        name, payload_offset, kHeaderBytes);
    const std::string_view magic(
        reinterpret_cast<const char*>(header.data()), 4);
    const int version = header[4];
    const int bits = header[5];
    const auto reserved = read_scalar<std::uint16_t>(
        header, 6, "MX reserved");
    const auto rows = read_scalar<std::uint64_t>(header, 8, "MX rows");
    const auto columns = read_scalar<std::uint64_t>(header, 16, "MX columns");
    const auto storage_rows = read_scalar<std::uint64_t>(
        header, 24, "MX storage rows");
    const auto storage_columns = read_scalar<std::uint64_t>(
        header, 32, "MX storage columns");
    const auto scale_rows = read_scalar<std::uint64_t>(
        header, 40, "MX scale rows");
    const auto scale_columns = read_scalar<std::uint64_t>(
        header, 48, "MX scale columns");
    const auto expected_storage_columns = expected_bits == 4
        ? static_cast<std::uint64_t>(expected_columns / 2)
        : static_cast<std::uint64_t>(expected_columns);
    const auto expected_scale_rows = expected_bits == 4
        ? static_cast<std::uint64_t>(expected_rows)
        : static_cast<std::uint64_t>((expected_rows + 127) / 128);
    const auto expected_scale_columns = expected_bits == 4
        ? static_cast<std::uint64_t>(expected_columns / 32)
        : static_cast<std::uint64_t>(expected_columns / 128);
    if (magic != "MXT1" || version != 1 || bits != expected_bits
        || reserved != 0 || rows != static_cast<std::uint64_t>(expected_rows)
        || columns != static_cast<std::uint64_t>(expected_columns)
        || storage_rows != rows || storage_columns != expected_storage_columns
        || scale_rows != expected_scale_rows
        || scale_columns != expected_scale_columns) {
        throw std::runtime_error(
            "unsupported streamed MX geometry: " + name);
    }
    const auto value_bytes = checked_range_product(
        storage_rows, storage_columns, "MX value bytes");
    const auto scale_bytes = checked_range_product(
        scale_rows, scale_columns, "MX scale bytes");
    if (kHeaderBytes + value_bytes + scale_bytes != payload_bytes) {
        throw std::runtime_error(
            "invalid streamed MX payload length: " + name);
    }
    return {
        expected_rows,
        expected_columns,
        expected_bits,
        payload_offset,
        payload_offset + kHeaderBytes,
        payload_offset + kHeaderBytes + value_bytes,
    };
}

std::vector<std::uint8_t> slice_streamed_nint_expert(
    const MfqContainer& model,
    const std::string& name,
    const MfeNintStreamLayout& layout,
    int local_expert,
    int rows_per_expert) {
    const auto begin = checked_product(
        static_cast<std::size_t>(local_expert),
        static_cast<std::size_t>(rows_per_expert),
        "NINT expert row offset");
    const auto end = checked_add(
        begin,
        static_cast<std::size_t>(rows_per_expert),
        "NINT expert row end");
    if (end > static_cast<std::size_t>(layout.rows)) {
        throw std::out_of_range("streamed NINT expert is out of range");
    }
    std::vector<std::uint8_t> result = model.read_range(
        name, layout.payload_offset, 42);
    patch_scalar_bytes<std::int64_t>(
        result, 18, rows_per_expert, "NINTv2 shape row");
    patch_scalar_bytes<std::uint32_t>(
        result, 34, static_cast<std::uint32_t>(rows_per_expert),
        "NINTv2 output size");
    append_record_range(
        result, model, name,
        layout.anchor_scale_offset + begin * 2,
        static_cast<std::uint64_t>(rows_per_expert) * 2);
    append_record_range(
        result, model, name,
        layout.anchor_min_offset + begin * 2,
        static_cast<std::uint64_t>(rows_per_expert) * 2);
    if (!layout.adaptive_storage) {
        const auto metadata_begin = checked_product(
            begin,
            static_cast<std::size_t>(layout.groups),
            "legacy NINT metadata prefix");
        const auto metadata_count = checked_product(
            static_cast<std::size_t>(rows_per_expert),
            static_cast<std::size_t>(layout.groups),
            "legacy NINT metadata slice");
        const auto append_legacy = [&](std::uint64_t offset,
                                       std::size_t first,
                                       std::size_t count,
                                       int bits) {
            if (layout.legacy_unpacked) {
                append_record_range(
                    result, model, name, offset + first, count);
            } else {
                const auto packed = read_packed_value_slice(
                    model, name, offset, first, count, bits);
                result.insert(result.end(), packed.begin(), packed.end());
            }
        };
        append_legacy(
            layout.legacy_sub_scale_offset,
            metadata_begin,
            metadata_count,
            layout.nominal_sub_bits);
        append_legacy(
            layout.legacy_sub_min_offset,
            metadata_begin,
            metadata_count,
            layout.nominal_sub_bits);
        const auto values_per_row = checked_product(
            static_cast<std::size_t>(layout.groups),
            static_cast<std::size_t>(layout.group_size),
            "legacy NINT values per row");
        append_legacy(
            layout.legacy_q_offset,
            checked_product(begin, values_per_row,
                "legacy NINT q prefix"),
            checked_product(
                static_cast<std::size_t>(rows_per_expert),
                values_per_row,
                "legacy NINT q slice"),
            layout.q_bits);
        return result;
    }
    const auto k_slice = std::span<const std::uint8_t>(
        layout.k_selectors).subspan(begin, rows_per_expert);
    const auto packed_k = pack_small_selectors(k_slice, 2);
    result.insert(result.end(), packed_k.begin(), packed_k.end());
    for (int selector = 0; selector < 4; ++selector) {
        const auto selected_before = static_cast<std::size_t>(std::count(
            layout.k_selectors.begin(),
            layout.k_selectors.begin() + static_cast<std::ptrdiff_t>(begin),
            static_cast<std::uint8_t>(selector)));
        const auto selected_here = static_cast<std::size_t>(std::count(
            k_slice.begin(), k_slice.end(),
            static_cast<std::uint8_t>(selector)));
        const int bits = layout.nominal_sub_bits - 1 + selector;
        for (const auto offset : {
                 layout.sub_scale_offsets[selector],
                 layout.sub_min_offsets[selector]}) {
            const auto packed = read_packed_value_slice(
                model, name, offset,
                checked_product(selected_before,
                    static_cast<std::size_t>(layout.groups),
                    "NINTv2 subgroup prefix"),
                checked_product(selected_here,
                    static_cast<std::size_t>(layout.groups),
                    "NINTv2 subgroup slice"),
                bits);
            result.insert(result.end(), packed.begin(), packed.end());
        }
    }
    const auto q_slice = std::span<const std::uint8_t>(
        layout.q_selectors).subspan(begin, rows_per_expert);
    const auto packed_q_selectors = pack_small_selectors(q_slice, 3);
    result.insert(
        result.end(),
        packed_q_selectors.begin(),
        packed_q_selectors.end());
    const auto values_per_row = checked_product(
        static_cast<std::size_t>(layout.groups),
        static_cast<std::size_t>(layout.group_size),
        "NINTv2 values per row");
    for (int selector = 0; selector < 8; ++selector) {
        const auto selected_before = static_cast<std::size_t>(std::count(
            layout.q_selectors.begin(),
            layout.q_selectors.begin() + static_cast<std::ptrdiff_t>(begin),
            static_cast<std::uint8_t>(selector)));
        const auto selected_here = static_cast<std::size_t>(std::count(
            q_slice.begin(), q_slice.end(),
            static_cast<std::uint8_t>(selector)));
        const auto packed = read_packed_value_slice(
            model, name, layout.q_offsets[selector],
            checked_product(selected_before, values_per_row,
                "NINTv2 q prefix"),
            checked_product(selected_here, values_per_row,
                "NINTv2 q slice"),
            selector + 1);
        result.insert(result.end(), packed.begin(), packed.end());
    }
    return result;
}

std::vector<std::uint8_t> slice_streamed_nvq_expert(
    const MfqContainer& model,
    const std::string& name,
    const MfeNvqJscStreamLayout& layout,
    int local_expert,
    int rows_per_expert) {
    const auto begin = checked_product(
        static_cast<std::size_t>(local_expert),
        static_cast<std::size_t>(rows_per_expert),
        "NVQ-JSC expert row offset");
    const auto end = checked_add(
        begin,
        static_cast<std::size_t>(rows_per_expert),
        "NVQ-JSC expert row end");
    if (end > static_cast<std::size_t>(layout.rows)) {
        throw std::out_of_range("streamed NVQ-JSC expert is out of range");
    }
    auto result = model.read_range(
        name, layout.payload_offset, layout.prefix_bytes);
    patch_scalar_bytes<std::int64_t>(
        result, 20, rows_per_expert, "NVQ-JSC shape row");
    patch_scalar_bytes<std::uint32_t>(
        result, 36, static_cast<std::uint32_t>(rows_per_expert),
        "NVQ-JSC output size");
    append_record_range(
        result, model, name,
        layout.anchors_offset + begin * 2,
        static_cast<std::uint64_t>(rows_per_expert) * 2);
    if (layout.group64) {
        append_record_range(
            result, model, name,
            layout.indices_offset
                + begin * static_cast<std::size_t>(layout.groups) * 8,
            static_cast<std::uint64_t>(rows_per_expert)
                * static_cast<std::uint64_t>(layout.groups) * 8);
        return result;
    }
    const auto append_bits = [&](std::uint64_t offset,
                                 std::size_t values_per_row,
                                 int bits) {
        const auto packed = read_packed_value_slice(
            model, name, offset,
            checked_product(begin, values_per_row,
                "NVQ-JSC stream prefix"),
            checked_product(
                static_cast<std::size_t>(rows_per_expert),
                values_per_row,
                "NVQ-JSC stream slice"),
            bits);
        result.insert(result.end(), packed.begin(), packed.end());
    };
    append_bits(layout.state_offset, layout.groups, layout.state_bits);
    append_bits(layout.indices_offset, layout.vectors, layout.index_bits);
    append_bits(layout.signs_offset, layout.signs, 7);
    return result;
}

std::vector<std::uint8_t> slice_streamed_mx_expert(
    const MfqContainer& model,
    const std::string& name,
    const MfeMxStreamLayout& layout,
    int local_expert,
    int rows_per_expert) {
    const auto begin = checked_product(
        static_cast<std::size_t>(local_expert),
        static_cast<std::size_t>(rows_per_expert),
        "MX expert row offset");
    const auto end = checked_add(
        begin,
        static_cast<std::size_t>(rows_per_expert),
        "MX expert row end");
    if (end > static_cast<std::size_t>(layout.rows)) {
        throw std::out_of_range("streamed MX expert is out of range");
    }
    auto result = model.read_range(name, layout.payload_offset, 56);
    patch_scalar_bytes<std::uint64_t>(
        result, 8, rows_per_expert, "MX logical rows");
    patch_scalar_bytes<std::uint64_t>(
        result, 24, rows_per_expert, "MX storage rows");
    const auto value_stride = static_cast<std::uint64_t>(
        layout.bits == 4 ? layout.columns / 2 : layout.columns);
    append_record_range(
        result, model, name,
        layout.values_offset + begin * value_stride,
        static_cast<std::uint64_t>(rows_per_expert) * value_stride);
    if (layout.bits == 4) {
        const auto scale_stride = static_cast<std::uint64_t>(
            layout.columns / 32);
        patch_scalar_bytes<std::uint64_t>(
            result, 40, rows_per_expert, "MXFP4 scale rows");
        append_record_range(
            result, model, name,
            layout.scales_offset + begin * scale_stride,
            static_cast<std::uint64_t>(rows_per_expert) * scale_stride);
    } else {
        if ((begin & 127u) != 0u) {
            throw MfeStreamUnsupported(
                "MXFP8 expert rows are not block-128 aligned");
        }
        const auto scale_rows =
            (static_cast<std::size_t>(rows_per_expert) + 127) / 128;
        const auto scale_stride = static_cast<std::uint64_t>(
            layout.columns / 128);
        patch_scalar_bytes<std::uint64_t>(
            result, 40, scale_rows, "MXFP8 scale rows");
        append_record_range(
            result, model, name,
            layout.scales_offset + (begin / 128) * scale_stride,
            static_cast<std::uint64_t>(scale_rows) * scale_stride);
    }
    return result;
}

std::vector<std::uint8_t> slice_streamed_mfe_expert_payload(
    const MfqContainer& model,
    const std::string& name,
    const MfeStreamPool& pool,
    int local_expert,
    int rows_per_expert) {
    return std::visit(
        [&](const auto& layout) {
            using Layout = std::decay_t<decltype(layout)>;
            if constexpr (std::is_same_v<Layout, MfeNintStreamLayout>) {
                return slice_streamed_nint_expert(
                    model, name, layout, local_expert, rows_per_expert);
            } else if constexpr (
                std::is_same_v<Layout, MfeNvqJscStreamLayout>) {
                return slice_streamed_nvq_expert(
                    model, name, layout, local_expert, rows_per_expert);
            } else {
                return slice_streamed_mx_expert(
                    model, name, layout, local_expert, rows_per_expert);
            }
        },
        pool.layout);
}

} // namespace

struct MlxMfeOffloadCache::Impl {
    struct Key {
        std::string name;
        std::int32_t expert = 0;

        bool operator==(const Key& other) const noexcept {
            return expert == other.expert
                && name == other.name;
        }
    };

    struct KeyHash {
        std::size_t operator()(
            const Key& value) const noexcept {
            const auto first =
                std::hash<std::string>{}(value.name);
            const auto second =
                std::hash<std::int32_t>{}(
                    value.expert);
            return first
                ^ (
                    second
                    + std::size_t{
                        0x9e3779b97f4a7c15ull}
                    + (first << 6)
                    + (first >> 2)
                );
        }
    };

    struct MfeCachedExpert {
        MfeCachedExpert(
            std::int32_t global,
            MlxMfeWeight value)
            : expert(global),
              weight(std::move(value)),
              packed_nbytes(weight.packed_nbytes()) {}

        std::int32_t expert = 0;
        MlxMfeWeight weight;
        std::size_t packed_nbytes = 0;
    };

    struct MfeCacheValue {
        Key key;
        std::shared_ptr<const MfeCachedExpert> weight;
    };

    using MfeProjectionCache = std::unordered_map<
        std::string,
        std::shared_ptr<const MfeStreamProjection>>;
    using MfeLru = std::list<MfeCacheValue>;
    using MfeExpertCache = std::unordered_map<
        Key,
        MfeLru::iterator,
        KeyHash>;

    Impl(
        const MfqContainer& source,
        std::size_t limit,
        int expert_count)
        : model(source),
          cache_limit(limit),
          experts(expert_count) {
        if (experts < 0) {
            throw std::invalid_argument(
                "MFE offload expert count cannot be negative");
        }
    }

    std::shared_ptr<const MfeStreamProjection>
    parse_mfe_projection(
        const std::string& name) {
        const auto& record = model.record(name);
        if (record.dtype != "MFE") {
            throw MfeStreamUnsupported(
                "expert record is not MFE: " + name);
        }
        constexpr std::uint64_t header_size = 20;
        constexpr std::uint64_t pool_header_size = 24;
        if (record.nbytes < header_size) {
            throw std::runtime_error(
                "truncated streamed MFE header: " + name);
        }
        const auto header = model.read_range(name, 0, header_size);
        const std::string_view magic(
            reinterpret_cast<const char*>(header.data()), 4);
        if (magic == "NIM1") {
            throw MfeStreamUnsupported(
                "legacy NIM1 records are not streamable");
        }
        if (magic != "MFE1" && magic != "NIM2") {
            throw std::runtime_error(
                "invalid streamed MFE magic: " + name);
        }
        const auto record_experts = read_scalar<std::uint32_t>(
            header, 4, "MFE expert count");
        const auto rows_per_expert = read_scalar<std::uint32_t>(
            header, 8, "MFE output width");
        const auto columns = read_scalar<std::uint32_t>(
            header, 12, "MFE input width");
        const auto pool_count = read_scalar<std::uint32_t>(
            header, 16, "MFE pool count");
        if (record_experts == 0
            || record_experts > static_cast<std::uint32_t>(
                std::numeric_limits<int>::max())
            || (experts > 0
                && record_experts != static_cast<std::uint32_t>(experts))
            || rows_per_expert == 0 || columns == 0
            || rows_per_expert > static_cast<std::uint32_t>(
                std::numeric_limits<int>::max())
            || columns > static_cast<std::uint32_t>(
                std::numeric_limits<int>::max())
            || pool_count == 0 || pool_count > record_experts) {
            throw std::runtime_error(
                "invalid streamed MFE dimensions: " + name);
        }
        const int projection_experts = static_cast<int>(record_experts);

        std::vector<std::optional<MfeStreamExpertLocation>> locations(
            static_cast<std::size_t>(projection_experts));
        std::uint64_t offset = header_size;
        for (std::uint32_t pool_index = 0;
             pool_index < pool_count;
             ++pool_index) {
            if (offset > record.nbytes
                || pool_header_size > record.nbytes - offset) {
                throw std::runtime_error(
                    "truncated streamed MFE pool header: " + name);
            }
            const auto pool_header = model.read_range(
                name, offset, pool_header_size);
            const auto pool_experts = read_scalar<std::uint32_t>(
                pool_header, 0, "MFE pool expert count");
            const auto dtype_bytes = read_scalar<std::uint32_t>(
                pool_header, 4, "MFE pool dtype length");
            const auto payload_bytes = read_scalar<std::uint64_t>(
                pool_header, 8, "MFE pool payload length");
            const auto runtime_bytes = read_scalar<std::uint64_t>(
                pool_header, 16, "MFE pool runtime length");
            if (pool_experts == 0 || pool_experts > record_experts
                || dtype_bytes == 0 || dtype_bytes > 32) {
                throw std::runtime_error(
                    "invalid streamed MFE pool metadata: " + name);
            }
            offset = checked_range_add(
                offset, pool_header_size, "MFE pool metadata offset");
            const auto ids_bytes = checked_range_product(
                pool_experts, sizeof(std::int32_t), "MFE expert IDs");
            const auto metadata_bytes = checked_range_add(
                ids_bytes, dtype_bytes, "MFE pool metadata bytes");
            const auto runtime_offset = checked_range_add(
                offset, metadata_bytes, "MFE runtime offset");
            const auto payload_offset = checked_range_add(
                runtime_offset, runtime_bytes, "MFE payload offset");
            const auto payload_end = checked_range_add(
                payload_offset, payload_bytes, "MFE payload end");
            if (payload_end > record.nbytes) {
                throw std::runtime_error(
                    "truncated streamed MFE pool: " + name);
            }
            const auto metadata = model.read_range(
                name, offset, metadata_bytes);
            const auto dtype = read_ascii(
                metadata,
                static_cast<std::size_t>(ids_bytes),
                dtype_bytes,
                "MFE pool dtype");
            const auto canonical = std::string(
                mfq::canonical_format_dtype(dtype));
            const auto pool_rows_u64 = checked_range_product(
                pool_experts,
                rows_per_expert,
                "MFE pool rows");
            if (pool_rows_u64 > static_cast<std::uint64_t>(
                    std::numeric_limits<int>::max())) {
                throw std::runtime_error(
                    "streamed MFE pool row count is too large: " + name);
            }
            const int pool_rows = static_cast<int>(pool_rows_u64);
            MfeStreamLayout layout;
            if (is_nint_dtype(canonical)) {
                if (runtime_bytes != 0) {
                    throw std::runtime_error(
                        "streamed NINTv2 has unexpected runtime metadata: "
                        + name);
                }
                layout = parse_streamed_nint_layout(
                    model, name, payload_offset, payload_bytes,
                    pool_rows, static_cast<int>(columns));
            } else if (canonical == "MXFP4" || canonical == "MXFP8") {
                if (runtime_bytes != 0) {
                    throw std::runtime_error(
                        "streamed MX has unexpected runtime metadata: "
                        + name);
                }
                layout = parse_streamed_mx_layout(
                    model, name, payload_offset, payload_bytes,
                    pool_rows, static_cast<int>(columns),
                    canonical == "MXFP4" ? 4 : 8);
            } else if (is_vq_dtype(canonical)) {
                layout = parse_streamed_nvq_jsc_layout(
                    model, name, payload_offset, payload_bytes,
                    pool_rows, static_cast<int>(columns));
            } else {
                throw MfeStreamUnsupported(
                    "MFE contains a cohort without expert slicing support: "
                    + dtype);
            }

            auto pool = std::make_shared<MfeStreamPool>();
            pool->dtype = dtype;
            pool->expert_count = static_cast<int>(pool_experts);
            pool->runtime_offset = runtime_offset;
            pool->runtime_bytes = runtime_bytes;
            pool->payload_offset = payload_offset;
            pool->payload_bytes = payload_bytes;
            pool->layout = std::move(layout);
            for (std::uint32_t local = 0; local < pool_experts; ++local) {
                const auto expert = read_scalar<std::int32_t>(
                    metadata,
                    static_cast<std::size_t>(local) * sizeof(std::int32_t),
                    "MFE global expert ID");
                if (expert < 0 || expert >= projection_experts
                    || locations[static_cast<std::size_t>(expert)].has_value()) {
                    throw std::runtime_error(
                        "invalid or duplicate streamed MFE expert ID: "
                        + name);
                }
                locations[static_cast<std::size_t>(expert)] =
                    MfeStreamExpertLocation{
                        pool,
                        static_cast<int>(local),
                    };
            }
            offset = payload_end;
        }
        if (offset != record.nbytes
            || std::any_of(
                locations.begin(), locations.end(),
                [](const auto& value) { return !value.has_value(); })) {
            throw std::runtime_error(
                "streamed MFE pools do not exactly cover the record: "
                + name);
        }
        return std::make_shared<MfeStreamProjection>(
            MfeStreamProjection{
                projection_experts,
                static_cast<int>(rows_per_expert),
                static_cast<int>(columns),
                std::move(locations),
            });
    }

    std::shared_ptr<const MfeStreamProjection>
    mfe_projection_locked(const std::string& name) {
        const auto found = mfe_projections.find(name);
        if (found != mfe_projections.end()) {
            return found->second;
        }
        auto result = parse_mfe_projection(name);
        mfe_projections.emplace(name, result);
        return result;
    }

    std::vector<std::uint8_t> load_mfe_expert_blob(
        const std::string& name,
        const MfeStreamProjection& projection,
        const MfeStreamExpertLocation& location) {
        const auto payload = slice_streamed_mfe_expert_payload(
            model,
            name,
            *location.pool,
            location.local_expert,
            projection.out_per_expert);
        const auto runtime = model.read_range(
            name,
            location.pool->runtime_offset,
            location.pool->runtime_bytes);
        std::vector<std::uint8_t> blob;
        const auto reserve_bytes = checked_add(
            checked_add(payload.size(), runtime.size(),
                "streamed expert payload"),
            checked_add(
                std::size_t{20 + 24 + sizeof(std::int32_t)},
                location.pool->dtype.size(),
                "streamed expert metadata"),
            "streamed expert blob");
        blob.reserve(reserve_bytes);
        blob.insert(blob.end(), {'M', 'F', 'E', '1'});
        append_scalar_bytes<std::uint32_t>(blob, 1);
        append_scalar_bytes<std::uint32_t>(
            blob, static_cast<std::uint32_t>(projection.out_per_expert));
        append_scalar_bytes<std::uint32_t>(
            blob, static_cast<std::uint32_t>(projection.neuron_len));
        append_scalar_bytes<std::uint32_t>(blob, 1);
        append_scalar_bytes<std::uint32_t>(blob, 1);
        append_scalar_bytes<std::uint32_t>(
            blob,
            static_cast<std::uint32_t>(location.pool->dtype.size()));
        append_scalar_bytes<std::uint64_t>(blob, payload.size());
        append_scalar_bytes<std::uint64_t>(blob, runtime.size());
        append_scalar_bytes<std::int32_t>(blob, 0);
        blob.insert(
            blob.end(),
            location.pool->dtype.begin(),
            location.pool->dtype.end());
        blob.insert(blob.end(), runtime.begin(), runtime.end());
        blob.insert(blob.end(), payload.begin(), payload.end());
        return blob;
    }

    // Own the record table and source paths.  Streamed layers frequently
    // outlive the MfqContainer object used by their load call.
    MfqContainer model;
    const std::size_t cache_limit = 0;
    const int experts = 0;
    mutable std::mutex mutex;
    std::unordered_map<
        std::string,
        std::shared_ptr<const MfeStreamProjection>> mfe_projections;
    MfeLru mfe_lru;
    MfeExpertCache mfe_cache;
    std::size_t resident_bytes = 0;
};

MlxMfeOffloadCache::MlxMfeOffloadCache(
    const MfqContainer& model,
    std::size_t cache_limit_bytes,
    int experts)
    : impl_(std::make_unique<Impl>(
          model,
          cache_limit_bytes,
          experts)) {}

MlxMfeOffloadCache::~MlxMfeOffloadCache() =
    default;

bool MlxMfeOffloadCache::can_offload(
    const std::string& name) {
    std::lock_guard lock(impl_->mutex);
    try {
        (void)impl_->mfe_projection_locked(name);
        return true;
    } catch (const MfeStreamUnsupported&) {
        return false;
    }
}

bool MlxMfeOffloadCache::can_group_mfe(
    const std::string& name) {
    std::lock_guard lock(impl_->mutex);
    try {
        (void)impl_->mfe_projection_locked(name);
        return true;
    } catch (const MfeStreamUnsupported&) {
        return false;
    }
}

MlxMfeProjectionInfo
MlxMfeOffloadCache::projection_info(
    const std::string& name) {
    std::lock_guard lock(impl_->mutex);
    const auto projection = impl_->mfe_projection_locked(name);
    MlxMfeProjectionInfo result;
    result.experts = projection->experts;
    result.out_per_expert = projection->out_per_expert;
    result.neuron_len = projection->neuron_len;
    for (int expert = 0; expert < projection->experts; ++expert) {
        if (projection->experts_by_id[
                static_cast<std::size_t>(expert)].has_value()) {
            result.available_experts.push_back(expert);
        }
    }
    return result;
}

std::vector<std::uint8_t>
MlxMfeOffloadCache::availability(
    const std::string& name) {
    std::lock_guard lock(impl_->mutex);
    const auto projection = impl_->mfe_projection_locked(name);
    std::vector<std::uint8_t> result(
        static_cast<std::size_t>(projection->experts), 0);
    for (int expert = 0; expert < projection->experts; ++expert) {
        result[static_cast<std::size_t>(expert)] =
            static_cast<std::uint8_t>(projection->experts_by_id[
                static_cast<std::size_t>(expert)].has_value());
    }
    return result;
}

MlxMfeWeight MlxMfeOffloadCache::grouped_mfe(
    const std::string& name,
    const std::vector<std::int32_t>& active_experts) {
    std::lock_guard lock(impl_->mutex);
    const auto projection = impl_->mfe_projection_locked(name);
    if (active_experts.empty()) {
        throw std::invalid_argument(
            "streamed MFE active expert set cannot be empty");
    }

    std::vector<Impl::Key> ordered_keys;
    ordered_keys.reserve(active_experts.size());
    std::unordered_set<Impl::Key, Impl::KeyHash> active_keys;
    active_keys.reserve(active_experts.size());
    for (const auto expert : active_experts) {
        if (expert < 0 || expert >= projection->experts) {
            throw std::out_of_range(
                "streamed MFE global expert ID is out of range");
        }
        if (!projection->experts_by_id[
                static_cast<std::size_t>(expert)].has_value()) {
            throw std::runtime_error(
                "streamed MFE global expert is unavailable: "
                + std::to_string(expert));
        }
        Impl::Key key{name, expert};
        if (active_keys.emplace(key).second) {
            ordered_keys.push_back(std::move(key));
        }
    }

    struct ActivePage {
        Impl::Key key;
        std::shared_ptr<const Impl::MfeCachedExpert> value;
        bool staged = false;
    };
    struct PendingPage {
        std::size_t page = 0;
        std::int32_t expert = 0;
        std::future<std::vector<std::uint8_t>> blob;
    };
    std::vector<ActivePage> pages;
    pages.reserve(ordered_keys.size());
    std::vector<PendingPage> pending;
    pending.reserve(ordered_keys.size());
    std::size_t staged_bytes = 0;
    for (const auto& key : ordered_keys) {
        const auto cached = impl_->mfe_cache.find(key);
        if (cached != impl_->mfe_cache.end()) {
            pages.push_back({key, cached->second->weight, false});
            continue;
        }
        const auto location = *projection->experts_by_id[
            static_cast<std::size_t>(key.expert)];
        const auto page = pages.size();
        pages.push_back({key, nullptr, true});
        pending.push_back({
            page,
            key.expert,
            std::async(
                std::launch::async,
                [source = impl_.get(), name, projection, location] {
                    return source->load_mfe_expert_blob(
                        name, *projection, location);
                }),
        });
    }
    for (auto& request : pending) {
        auto blob = request.blob.get();
        auto value = std::make_shared<Impl::MfeCachedExpert>(
            request.expert,
            MlxMfeWeight::from_blob(blob));
        staged_bytes = checked_add(
            staged_bytes,
            value->packed_nbytes,
            "staged MFE expert bytes");
        pages[request.page].value = std::move(value);
    }

    std::vector<MlxMfeWeight> weights;
    weights.reserve(pages.size());
    for (const auto& page : pages) {
        weights.push_back(page.value->weight);
    }
    auto result = MlxMfeWeight::concatenate_experts(weights);

    impl_->mfe_cache.reserve(checked_add(
        impl_->mfe_cache.size(),
        static_cast<std::size_t>(std::count_if(
            pages.begin(), pages.end(),
            [](const auto& page) { return page.staged; })),
        "resident MFE cache entries"));
    for (const auto& page : pages) {
        if (!page.staged) {
            continue;
        }
        impl_->mfe_lru.push_back({page.key, page.value});
        const auto inserted = std::prev(impl_->mfe_lru.end());
        if (!impl_->mfe_cache.emplace(inserted->key, inserted).second) {
            impl_->mfe_lru.erase(inserted);
            throw std::logic_error(
                "duplicate staged MFE expert cache key");
        }
    }
    for (const auto& key : ordered_keys) {
        const auto cached = impl_->mfe_cache.find(key);
        impl_->mfe_lru.splice(
            impl_->mfe_lru.end(),
            impl_->mfe_lru,
            cached->second);
    }
    auto committed_bytes = checked_add(
        impl_->resident_bytes,
        staged_bytes,
        "resident MFE expert bytes");
    while (committed_bytes > impl_->cache_limit
           && !impl_->mfe_lru.empty()) {
        const auto candidate = std::find_if(
            impl_->mfe_lru.begin(),
            impl_->mfe_lru.end(),
            [&](const auto& item) {
                return active_keys.find(item.key) == active_keys.end();
            });
        if (candidate == impl_->mfe_lru.end()) {
            break;
        }
        committed_bytes -= candidate->weight->packed_nbytes;
        impl_->mfe_cache.erase(candidate->key);
        impl_->mfe_lru.erase(candidate);
    }
    impl_->resident_bytes = committed_bytes;
    return result;
}

std::size_t
MlxMfeOffloadCache::cache_limit_bytes() const noexcept {
    return impl_->cache_limit;
}

std::size_t
MlxMfeOffloadCache::resident_packed_bytes() const {
    std::lock_guard lock(impl_->mutex);
    return impl_->resident_bytes;
}

std::size_t
MlxMfeOffloadCache::cached_expert_count() const {
    std::lock_guard lock(impl_->mutex);
    return impl_->mfe_cache.size();
}

void MlxMfeOffloadCache::discard_record(
    const std::string& name) noexcept {
    std::lock_guard lock(impl_->mutex);
    for (auto item = impl_->mfe_lru.begin();
         item != impl_->mfe_lru.end();) {
        if (item->key.name != name) {
            ++item;
            continue;
        }
        impl_->resident_bytes -= item->weight->packed_nbytes;
        impl_->mfe_cache.erase(item->key);
        item = impl_->mfe_lru.erase(item);
    }
    impl_->mfe_projections.erase(name);
}

void MlxMfeOffloadCache::clear() {
    std::lock_guard lock(impl_->mutex);
    impl_->mfe_cache.clear();
    impl_->mfe_lru.clear();
    impl_->mfe_projections.clear();
    impl_->resident_bytes = 0;
}

struct MlxMfeWeight::Impl {
    array descriptors;
    array nint_q;
    array nint_sub_scale;
    array nint_sub_min;
    array nint_anchor_scale;
    array nint_anchor_min;
    array q8_q;
    array q8_scales;
    array vq_indices;
    array vq_state;
    array vq_aux;
    array vq_anchors;
    array vq_codebooks;
    array vq_scales;
    array vq_state_to_codebank;
    array vq_banks;
    array vq_parameters;
    array vq_residual_codebooks;
    array vq_residual_first;
    array vq_residual_second;
    array mx_values;
    array mx_scales;
    std::vector<RotationSpec> rotations;
    std::vector<std::int32_t> descriptor_values;
    std::vector<std::shared_ptr<const Impl>> projection_views;
    std::optional<array> expert_order;
    std::vector<ReferenceMoeCohort> reference_cohorts;
    std::vector<Mxfp4SqMoeCohort> mxfp4_sq_cohorts;
    std::vector<Fp8SqMoeCohort> fp8_sq_cohorts;
    std::optional<array> standalone_owner;
    bool has_generic_cohorts = false;
    int experts = 0;
    int out_per_expert = 0;
    int neuron_len = 0;
    int projections = 0;
    // Physical SSD arenas may contain many more rows than one model layer.
    // Keep the logical expert geometry separately so hardware policy does not
    // confuse cache capacity with the routed operator it is executing.
    int logical_experts = 0;
    std::uint32_t family_mask = 0;
    std::uint32_t vq_profile_mask = 0;
    bool jsc_execution_layout = false;
    bool npq_grouped_indices = true;
    bool native_primitive = true;
    bool grouped_mmq = false;
    bool has_nepq_residual = false;
    std::optional<array> mxfp4_slot_ids;
    bool mxfp4_slot_ids_sorted = false;
    bool automatic_mxfp4_nax_prefill = true;
    int k_lanes_override = 0;
    std::size_t packed_bytes = 0;

    Impl(
        array descriptor_array,
        array nint_q_array,
        array nint_sub_scale_array,
        array nint_sub_min_array,
        array nint_anchor_scale_array,
        array nint_anchor_min_array,
        array q8_q_array,
        array q8_scale_array,
        array vq_indices_array,
        array vq_state_array,
        array vq_aux_array,
        array vq_anchor_array,
        array vq_codebook_array,
        array vq_scale_array,
        array vq_state_bank_array,
        array vq_bank_array,
        array vq_parameter_array,
        array vq_residual_codebook_array,
        array vq_residual_first_array,
        array vq_residual_second_array,
        array mx_value_array,
        array mx_scale_array,
        std::vector<RotationSpec> rotation_values,
        std::vector<std::int32_t> values,
        int expert_count,
        int output_width,
        int input_width,
        int projection_count)
        : descriptors(std::move(descriptor_array)),
          nint_q(std::move(nint_q_array)),
          nint_sub_scale(
              std::move(nint_sub_scale_array)),
          nint_sub_min(
              std::move(nint_sub_min_array)),
          nint_anchor_scale(
              std::move(nint_anchor_scale_array)),
          nint_anchor_min(
              std::move(nint_anchor_min_array)),
          q8_q(std::move(q8_q_array)),
          q8_scales(std::move(q8_scale_array)),
          vq_indices(std::move(vq_indices_array)),
          vq_state(std::move(vq_state_array)),
          vq_aux(std::move(vq_aux_array)),
          vq_anchors(std::move(vq_anchor_array)),
          vq_codebooks(std::move(vq_codebook_array)),
          vq_scales(std::move(vq_scale_array)),
          vq_state_to_codebank(
              std::move(vq_state_bank_array)),
          vq_banks(std::move(vq_bank_array)),
          vq_parameters(
              std::move(vq_parameter_array)),
          vq_residual_codebooks(
              std::move(vq_residual_codebook_array)),
          vq_residual_first(
              std::move(vq_residual_first_array)),
          vq_residual_second(
              std::move(vq_residual_second_array)),
          mx_values(std::move(mx_value_array)),
          mx_scales(std::move(mx_scale_array)),
          rotations(std::move(rotation_values)),
          descriptor_values(std::move(values)),
          experts(expert_count),
          out_per_expert(output_width),
          neuron_len(input_width),
          projections(projection_count) {
        grouped_mmq = !descriptor_values.empty();
        for (
            std::size_t base = 0;
            base + kDescriptorSize
                <= descriptor_values.size();
            base += kDescriptorSize
        ) {
            const auto family = descriptor_values[
                base + kFamily];
            if (family >= 0 && family < 7) {
                family_mask |= std::uint32_t{1}
                    << static_cast<unsigned>(family);
            }
            if (family == kFamilyVq) {
                const auto profile = descriptor_values[
                    base + kVqProfile] & 255;
                if (profile >= 0 && profile < 8) {
                    vq_profile_mask |= std::uint32_t{1}
                        << static_cast<unsigned>(profile);
                }
                if ((descriptor_values[base + kVqProfile] >> 8) != 0) {
                    has_nepq_residual = true;
                }
            }
            const bool supported_nint =
                family == kFamilyNint
                && descriptor_values[base + kNintGroupSize] > 0
                && descriptor_values[base + kNintGroups]
                    == (neuron_len
                        + descriptor_values[base + kNintGroupSize] - 1)
                        / descriptor_values[base + kNintGroupSize]
                && descriptor_values[base + kNintV2] != 0;
            const bool supported_q8 =
                family == kFamilyNint8Zero
                && descriptor_values[base + kQ8Groups]
                    == neuron_len / 32;
            const bool supported_vq =
                family == kFamilyVq
                && descriptor_values[base + kVqGroupSize] == 24
                && (
                    descriptor_values[base + kVqVectorSize] == 4
                    || descriptor_values[base + kVqVectorSize] == 8
                );
            const bool supported_mxfp4 =
                family == kFamilyMxfp4
                && descriptor_values[base + kMxGroups]
                    == neuron_len / 32;
            const bool supported_mxfp8 =
                family == kFamilyMxfp8
                && neuron_len % 128 == 0
                && descriptor_values[base + kMxGroups]
                    == neuron_len / 128;
            const bool supported_dense =
                family == kFamilyBf16 || family == kFamilyF16;
            if (!supported_nint && !supported_q8
                && !supported_vq && !supported_mxfp4
                && !supported_mxfp8 && !supported_dense) {
                grouped_mmq = false;
            }
            if (
                descriptor_values[
                    base + kVqJscExecution] != 0
            ) {
                jsc_execution_layout = true;
            }
        }
        if (
            descriptor_values.size()
                == static_cast<std::size_t>(experts) * kDescriptorSize
        ) {
            std::vector<std::int32_t> schedule(
                static_cast<std::size_t>(experts));
            std::iota(schedule.begin(), schedule.end(), 0);
            // Keep adjacent threadgroups on one exact decoder cohort and walk
            // each cohort's packed rows monotonically. Quantization assigns
            // formats independently of global expert ID, so the default ID
            // order otherwise jumps among unrelated packed streams on every
            // block. Row destinations remain route-order based; this only
            // changes execution order.
            std::stable_sort(
                schedule.begin(),
                schedule.end(),
                [this](std::int32_t left, std::int32_t right) {
                    const auto left_base = static_cast<std::size_t>(left)
                        * kDescriptorSize;
                    const auto right_base = static_cast<std::size_t>(right)
                        * kDescriptorSize;
                    const int left_family =
                        descriptor_values[left_base + kFamily];
                    const int right_family =
                        descriptor_values[right_base + kFamily];
                    const auto compare_field = [&](int field) {
                        if (
                            left_family == right_family
                            && (
                                left_family == kFamilyMxfp4
                                || left_family == kFamilyMxfp8
                            )
                            && (
                                field == kMxValueOffset
                                || field == kMxScaleOffset
                            )
                        ) {
                            const auto left_value = descriptor_u32_value(
                                descriptor_values[left_base + field]);
                            const auto right_value = descriptor_u32_value(
                                descriptor_values[right_base + field]);
                            return left_value < right_value
                                ? -1
                                : left_value > right_value ? 1 : 0;
                        }
                        return descriptor_values[left_base + field]
                            < descriptor_values[right_base + field]
                            ? -1
                            : descriptor_values[left_base + field]
                                > descriptor_values[right_base + field]
                            ? 1
                            : 0;
                    };
                    if (const int family = compare_field(kFamily);
                        family != 0) {
                        return family < 0;
                    }
                    for (int field = 4; field < kDescriptorSize; ++field) {
                        if (const int value = compare_field(field);
                            value != 0) {
                            return value < 0;
                        }
                    }
                    return descriptor_values[left_base + kLocalExpert]
                        < descriptor_values[right_base + kLocalExpert];
                });
            expert_order.emplace(make_int32_array(
                schedule,
                Shape{experts}));
        }
        grouped_mmq = grouped_mmq && expert_order.has_value();
        if (
            family_mask == (std::uint32_t{1} << kFamilyMxfp4)
            && descriptor_values.size()
                == static_cast<std::size_t>(experts) * kDescriptorSize
        ) {
            const auto value_stride = checked_product(
                static_cast<std::size_t>(out_per_expert),
                static_cast<std::size_t>(neuron_len / 2),
                "MXFP4 slot value stride");
            const auto scale_stride = checked_product(
                static_cast<std::size_t>(out_per_expert),
                static_cast<std::size_t>(neuron_len / 32),
                "MXFP4 slot scale stride");
            std::vector<std::int32_t> slots(
                static_cast<std::size_t>(experts));
            bool valid_slots = value_stride != 0 && scale_stride != 0
                && mx_values.size() % value_stride == 0
                && mx_scales.size() % scale_stride == 0
                && mx_values.size() / value_stride
                    == mx_scales.size() / scale_stride;
            for (int expert = 0; expert < experts; ++expert) {
                const auto base = static_cast<std::size_t>(expert)
                    * kDescriptorSize;
                const int local = descriptor_values[base + kLocalExpert];
                const auto value_offset = descriptor_u32_value(
                    descriptor_values[base + kMxValueOffset]);
                const auto scale_offset = descriptor_u32_value(
                    descriptor_values[base + kMxScaleOffset]);
                if (!valid_slots || local < 0
                    || static_cast<std::size_t>(value_offset) % value_stride
                        != 0
                    || static_cast<std::size_t>(scale_offset) % scale_stride
                        != 0) {
                    valid_slots = false;
                    break;
                }
                const auto value_slot =
                    static_cast<std::size_t>(value_offset) / value_stride
                    + static_cast<std::size_t>(local);
                const auto scale_slot =
                    static_cast<std::size_t>(scale_offset) / scale_stride
                    + static_cast<std::size_t>(local);
                if (value_slot != scale_slot
                    || value_slot >= mx_values.size() / value_stride) {
                    valid_slots = false;
                    break;
                }
                slots[static_cast<std::size_t>(expert)] = checked_int(
                    value_slot,
                    "MXFP4 physical expert slot");
            }
            if (valid_slots) {
                mxfp4_slot_ids_sorted =
                    std::is_sorted(slots.begin(), slots.end());
                mxfp4_slot_ids.emplace(make_int32_array(
                    slots,
                    Shape{experts}));
            }
        }
        const char* specialize_env = std::getenv(
            "MFQ_METAL_MFE_SPECIALIZE");
        if (
            specialize_env != nullptr
            && std::string_view(specialize_env) == "0"
        ) {
            family_mask = 126;
            vq_profile_mask = 255;
        }
        const char* npq_indices_env = std::getenv(
            "MFQ_METAL_MFE_NPQ_INDICES");
        npq_grouped_indices =
            npq_indices_env == nullptr
            || std::string_view(npq_indices_env) != "0";
        const char* native_env = std::getenv(
            "MFQ_METAL_MFE_NATIVE_PRIMITIVE");
        native_primitive = native_env == nullptr
            || std::string_view(native_env) != "0";
        const char* k_lanes_env = std::getenv(
            "MFQ_METAL_MFE_K_LANES");
        if (k_lanes_env != nullptr) {
            const auto value = std::string_view(k_lanes_env);
            if (value == "8") {
                k_lanes_override = 8;
            } else if (value == "16") {
                k_lanes_override = 16;
            } else if (value == "32") {
                k_lanes_override = 32;
            }
        }
        packed_bytes =
            descriptors.nbytes()
            + nint_q.nbytes()
            + nint_sub_scale.nbytes()
            + nint_sub_min.nbytes()
            + nint_anchor_scale.nbytes()
            + nint_anchor_min.nbytes()
            + q8_q.nbytes()
            + q8_scales.nbytes()
            + vq_indices.nbytes()
            + vq_state.nbytes()
            + vq_aux.nbytes()
            + vq_anchors.nbytes()
            + vq_codebooks.nbytes()
            + vq_scales.nbytes()
            + vq_state_to_codebank.nbytes()
            + vq_banks.nbytes()
            + vq_parameters.nbytes()
            + vq_residual_codebooks.nbytes()
            + vq_residual_first.nbytes()
            + vq_residual_second.nbytes()
            + mx_values.nbytes()
            + mx_scales.nbytes();
        for (const auto& rotation : rotations) {
            packed_bytes += rotation.signs.nbytes();
        }
    }
};

MlxMfeWeight::MlxMfeWeight(
    std::shared_ptr<const Impl> impl)
    : impl_(std::move(impl)) {
    if (!impl_) {
        throw std::invalid_argument(
            "MFE implementation cannot be null");
    }
}

MlxMfeWeight MlxMfeWeight::from_blob(
    std::span<const std::uint8_t> blob) {
    BlobCursor cursor(blob);
    const auto magic_bytes = cursor.bytes(4, "header");
    const std::string_view magic(
        reinterpret_cast<const char*>(
            magic_bytes.data()),
        magic_bytes.size());
    if (magic != "MFE1" && magic != "NIM1" && magic != "NIM2") {
        throw std::runtime_error("invalid MFE magic");
    }

    const int expert_count = checked_positive(
        cursor.scalar<std::uint32_t>("expert count"),
        "expert count");
    const int output_width = checked_positive(
        cursor.scalar<std::uint32_t>(
            "output width"),
        "output width");
    const int input_width = checked_positive(
        cursor.scalar<std::uint32_t>(
            "neuron length"),
        "neuron length");
    const auto pool_count =
        cursor.scalar<std::uint32_t>("pool count");
    if (
        pool_count == 0
        || pool_count
            > static_cast<std::uint32_t>(
                expert_count)
    ) {
        throw std::runtime_error(
            "invalid MFE pool count");
    }
    if (
        static_cast<std::size_t>(expert_count)
        > cursor.remaining() / sizeof(std::int32_t)
    ) {
        throw std::runtime_error(
            "MFE expert count exceeds its payload");
    }

    std::vector<std::int32_t> descriptors(
        checked_product(
            static_cast<std::size_t>(expert_count),
            static_cast<std::size_t>(kDescriptorSize),
            "descriptor count"),
        0);
    std::vector<int> owners(
        static_cast<std::size_t>(expert_count),
        -1);
    PackedStreams streams;
    std::vector<RotationSpec> rotations;
    std::vector<ReferenceMoeCohort>
        reference_cohorts;
    std::vector<Mxfp4SqMoeCohort>
        mxfp4_sq_cohorts;
    std::vector<Fp8SqMoeCohort>
        fp8_sq_cohorts;
    std::vector<std::int32_t> standalone_owner(
        static_cast<std::size_t>(expert_count), -1);
    bool has_generic_cohorts = false;
    const bool reference =
        mlx_reference_enabled();

    for (
        std::uint32_t pool = 0;
        pool < pool_count;
        ++pool
    ) {
        std::uint32_t count = 0;
        std::uint32_t dtype_bytes = 0;
        std::uint64_t payload_bytes = 0;
        std::uint64_t runtime_bytes = 0;
        if (magic == "NIM1") {
            count =
                cursor.scalar<std::uint32_t>(
                    "pool header");
            payload_bytes =
                cursor.scalar<std::uint64_t>(
                    "pool header");
        } else {
            count =
                cursor.scalar<std::uint32_t>(
                    "v2 pool header");
            dtype_bytes =
                cursor.scalar<std::uint32_t>(
                    "v2 pool header");
            payload_bytes =
                cursor.scalar<std::uint64_t>(
                    "v2 pool header");
            runtime_bytes =
                cursor.scalar<std::uint64_t>(
                    "v2 pool header");
        }
        if (
            count == 0
            || count
                > static_cast<std::uint32_t>(
                    expert_count)
            || (
                magic != "NIM1"
                && (
                    dtype_bytes == 0
                    || dtype_bytes > 32
                )
            )
        ) {
            throw std::runtime_error(
                "invalid MFE pool metadata");
        }

        std::vector<std::int32_t> expert_ids(count);
        for (auto& expert : expert_ids) {
            expert =
                cursor.scalar<std::int32_t>(
                    "expert IDs");
        }
        claim_experts(
            expert_ids,
            owners,
            static_cast<int>(pool));

        std::string dtype = "NINT";
        if (magic != "NIM1") {
            const auto raw_dtype =
                cursor.bytes(
                    dtype_bytes,
                    "cohort dtype");
            if (!is_ascii(raw_dtype)) {
                throw std::runtime_error(
                    "MFE cohort dtype must be ASCII");
            }
            dtype.assign(
                reinterpret_cast<const char*>(
                    raw_dtype.data()),
                raw_dtype.size());
            dtype = std::string(
                mfq::canonical_format_dtype(dtype));
        }

        auto runtime = cursor.bytes(
            checked_size(
                runtime_bytes,
                "cohort runtime metadata"),
            "cohort runtime metadata");
        auto payload = cursor.bytes(
            checked_size(
                payload_bytes,
                "cohort payload"),
            "cohort payload");

        if (is_nint_dtype(dtype)) {
            has_generic_cohorts = true;
            if (!runtime.empty()) {
                throw std::runtime_error(
                    "unexpected MFE NINT "
                    "runtime metadata");
            }
            auto weight = add_nint_pool(
                payload,
                expert_ids,
                output_width,
                input_width,
                streams,
                descriptors,
                !reference);
            if (reference) {
                reference_cohorts.push_back({
                    expert_ids,
                    std::move(weight),
                });
            }
        } else if (is_nint8_zero_dtype(dtype)) {
            has_generic_cohorts = true;
            if (!runtime.empty()) {
                throw std::runtime_error(
                    "unexpected MFE NINT8-0 "
                    "runtime metadata");
            }
            auto weight = add_q8_pool(
                payload,
                expert_ids,
                output_width,
                input_width,
                streams,
                descriptors,
                !reference);
            if (reference) {
                reference_cohorts.push_back({
                    expert_ids,
                    std::move(weight),
                });
            }
        } else if (dtype == "MXFP4" || dtype == "MXFP8") {
            has_generic_cohorts = true;
            if (!runtime.empty()) {
                throw std::runtime_error(
                    "unexpected MFE MX runtime metadata");
            }
            auto weight = add_mx_pool(
                dtype,
                payload,
                expert_ids,
                output_width,
                input_width,
                streams,
                descriptors,
                !reference);
            if (reference) {
                reference_cohorts.push_back({
                    expert_ids,
                    std::move(*weight),
                });
            }
        } else if (dtype == "BF16" || dtype == "F16") {
            has_generic_cohorts = true;
            if (!runtime.empty()) {
                throw std::runtime_error(
                    "unexpected MFE dense runtime metadata");
            }
            auto weight = add_dense_pool(
                dtype,
                payload,
                expert_ids,
                output_width,
                input_width,
                streams,
                descriptors,
                !reference);
            if (reference) {
                reference_cohorts.push_back({
                    expert_ids,
                    std::move(*weight),
                });
            }
        } else if (is_vq_dtype(dtype)) {
            has_generic_cohorts = true;
            auto weight = add_vq_pool(
                dtype,
                payload,
                runtime,
                expert_ids,
                output_width,
                input_width,
                streams,
                rotations,
                descriptors,
                !reference);
            if (reference) {
                reference_cohorts.push_back({
                    expert_ids,
                    std::move(weight),
                });
            }
        } else if (is_mxfp4_sq_dtype(dtype)) {
            if (!runtime.empty()) {
                throw std::runtime_error(
                    "unexpected MFE MXFP4-SQ runtime metadata");
            }
            auto weight = MlxMxfp4SqWeight::from_blob(payload);
            if (weight.input_size() != input_width ||
                weight.output_size() != checked_int(
                    checked_product(
                        static_cast<std::size_t>(count),
                        static_cast<std::size_t>(output_width),
                        "MXFP4-SQ cohort output width"),
                    "MXFP4-SQ cohort output width")) {
                throw std::runtime_error(
                    "MFE MXFP4-SQ cohort shape is inconsistent");
            }
            std::vector<std::int32_t> local_map(
                static_cast<std::size_t>(expert_count), -1);
            const int cohort_index = checked_int(
                mxfp4_sq_cohorts.size(),
                "MXFP4-SQ cohort count");
            for (std::size_t local = 0; local < expert_ids.size(); ++local) {
                const auto expert = expert_ids[local];
                local_map[static_cast<std::size_t>(expert)] =
                    checked_int(local, "MXFP4-SQ local expert");
                standalone_owner[static_cast<std::size_t>(expert)] =
                    cohort_index;
            }
            if (reference) {
                reference_cohorts.push_back({expert_ids, weight});
            }
            mxfp4_sq_cohorts.push_back({
                make_int32_array(local_map, Shape{expert_count}),
                std::move(weight),
            });
        } else if (is_fp8_sq_dtype(dtype)) {
            if (!runtime.empty()) {
                throw std::runtime_error(
                    "unexpected MFE native-FP8 SQ runtime metadata");
            }
            auto weight = MlxFp8SqWeight::from_blob(dtype, payload);
            if (weight.input_size() != input_width ||
                weight.output_size() != checked_int(
                    checked_product(
                        static_cast<std::size_t>(count),
                        static_cast<std::size_t>(output_width),
                        "native-FP8 SQ cohort output width"),
                    "native-FP8 SQ cohort output width")) {
                throw std::runtime_error(
                    "MFE native-FP8 SQ cohort shape is inconsistent");
            }
            std::vector<std::int32_t> local_map(
                static_cast<std::size_t>(expert_count), -1);
            const int cohort_index = checked_int(
                fp8_sq_cohorts.size(),
                "native-FP8 SQ cohort count");
            for (std::size_t local = 0; local < expert_ids.size(); ++local) {
                const auto expert = expert_ids[local];
                local_map[static_cast<std::size_t>(expert)] =
                    checked_int(local, "native-FP8 SQ local expert");
                standalone_owner[static_cast<std::size_t>(expert)] =
                    cohort_index;
            }
            if (reference) {
                reference_cohorts.push_back({expert_ids, weight});
            }
            fp8_sq_cohorts.push_back({
                make_int32_array(local_map, Shape{expert_count}),
                std::move(weight),
            });
        } else {
            throw std::runtime_error(
                "unsupported nested MFE cohort dtype: "
                + dtype);
        }
    }

    if (cursor.remaining() != 0) {
        throw std::runtime_error(
            "trailing bytes in MFE tensor");
    }
    const auto missing = std::find(
        owners.begin(),
        owners.end(),
        -1);
    if (missing != owners.end()) {
        throw std::runtime_error(
            "MFE pools do not cover every expert");
    }

    const Shape descriptor_shape{
        expert_count,
        kDescriptorSize,
    };
    auto impl = std::make_shared<Impl>(
        make_int32_array(
            descriptors,
            descriptor_shape),
        make_raw_array(
            std::move(streams.nint_q),
            mlx::core::uint8),
        make_raw_array(
            std::move(streams.nint_sub_scale),
            mlx::core::uint8),
        make_raw_array(
            std::move(streams.nint_sub_min),
            mlx::core::uint8),
        make_raw_array(
            std::move(streams.nint_anchor_scale),
            mlx::core::float32),
        make_raw_array(
            std::move(streams.nint_anchor_min),
            mlx::core::float32),
        make_raw_array(
            std::move(streams.q8_q),
            mlx::core::int8),
        make_raw_array(
            std::move(streams.q8_scales),
            mlx::core::float16),
        make_raw_array(
            std::move(streams.vq_indices),
            mlx::core::uint8),
        make_raw_array(
            std::move(streams.vq_state),
            mlx::core::uint8),
        make_raw_array(
            std::move(streams.vq_aux),
            mlx::core::uint8),
        make_raw_array(
            std::move(streams.vq_anchors),
            mlx::core::float32),
        make_raw_array(
            std::move(streams.vq_codebooks),
            mlx::core::int8),
        make_raw_array(
            std::move(streams.vq_scales),
            mlx::core::float32),
        make_raw_array(
            std::move(
                streams.vq_state_to_codebank),
            mlx::core::uint8),
        make_raw_array(
            std::move(streams.vq_banks),
            mlx::core::uint8),
        make_raw_array(
            std::move(streams.vq_parameters),
            mlx::core::float32),
        make_raw_array(
            std::move(streams.vq_residual_codebooks),
            mlx::core::float32),
        make_raw_array(
            std::move(streams.vq_residual_first),
            mlx::core::int16),
        make_raw_array(
            std::move(streams.vq_residual_second),
            mlx::core::int16),
        make_raw_array(
            std::move(streams.mx_values),
            mlx::core::uint8),
        make_raw_array(
            std::move(streams.mx_scales),
            mlx::core::uint8),
        std::move(rotations),
        std::move(descriptors),
        expert_count,
        output_width,
        input_width,
        1);
    impl->reference_cohorts =
        std::move(reference_cohorts);
    impl->mxfp4_sq_cohorts =
        std::move(mxfp4_sq_cohorts);
    impl->fp8_sq_cohorts =
        std::move(fp8_sq_cohorts);
    impl->has_generic_cohorts = has_generic_cohorts;
    if (!impl->mxfp4_sq_cohorts.empty()
        || !impl->fp8_sq_cohorts.empty()) {
        // These cohorts reuse their standalone packed kernels; none is
        // decoded by the heterogeneous MFE kernels.
        impl->grouped_mmq = false;
        impl->standalone_owner.emplace(make_int32_array(
            standalone_owner,
            Shape{expert_count}));
        if (!reference) {
            for (const auto& cohort : impl->mxfp4_sq_cohorts) {
                impl->packed_bytes += cohort.weight.packed_nbytes();
            }
            for (const auto& cohort : impl->fp8_sq_cohorts) {
                impl->packed_bytes += cohort.weight.packed_nbytes();
            }
        }
    }
    if (reference) {
        impl->grouped_mmq = false;
        impl->native_primitive = false;
        impl->packed_bytes = 0;
        for (const auto& cohort :
             impl->reference_cohorts) {
            impl->packed_bytes += std::visit(
                [](const auto& weight) {
                    return weight.packed_nbytes();
                },
                cohort.weight);
        }
    }
    // VQ/MX/dense cohorts are parsed through standalone loaders and repacked
    // into heterogeneous execution streams. Their temporary MLX arrays are
    // dead now; NINT/SQ arrays remain live as the shared-kernel cohorts.
    mlx::core::clear_cache();
    return MlxMfeWeight(std::move(impl));
}

MlxMfeWeight MlxMfeWeight::from_mxfp4_slots(
    int experts,
    int out_per_expert,
    int neuron_len,
    const std::vector<std::int32_t>& slot_for_expert,
    array packed_values,
    array block_scales,
    int logical_experts) {
    if (experts <= 0 || out_per_expert <= 0 || neuron_len <= 0 ||
        neuron_len % 32 != 0 ||
        slot_for_expert.size() != static_cast<std::size_t>(experts) ||
        logical_experts < 0) {
        throw std::invalid_argument("invalid MXFP4 slot-view dimensions");
    }
    if (packed_values.dtype() != mlx::core::uint8 ||
        block_scales.dtype() != mlx::core::uint8 ||
        !packed_values.flags().row_contiguous ||
        !block_scales.flags().row_contiguous) {
        throw std::invalid_argument(
            "MXFP4 slot-view banks must be contiguous uint8 arrays");
    }
    const auto value_stride = checked_product(
        static_cast<std::size_t>(out_per_expert),
        static_cast<std::size_t>(neuron_len / 2),
        "MXFP4 slot value stride");
    const auto scale_stride = checked_product(
        static_cast<std::size_t>(out_per_expert),
        static_cast<std::size_t>(neuron_len / 32),
        "MXFP4 slot scale stride");
    if (packed_values.size() % value_stride != 0 ||
        block_scales.size() % scale_stride != 0 ||
        packed_values.size() / value_stride !=
            block_scales.size() / scale_stride) {
        throw std::invalid_argument(
            "MXFP4 slot-view bank geometry mismatch");
    }
    const auto slots = packed_values.size() / value_stride;
    if (slots == 0 || slots > static_cast<std::size_t>(
        std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("invalid MXFP4 arena slot count");
    }
    std::vector<std::int32_t> descriptors(
        checked_product(
            static_cast<std::size_t>(experts),
            static_cast<std::size_t>(kDescriptorSize),
            "MXFP4 slot descriptors"),
        0);
    for (int expert = 0; expert < experts; ++expert) {
        const auto slot = slot_for_expert[static_cast<std::size_t>(expert)];
        if (slot < 0 || static_cast<std::size_t>(slot) >= slots) {
            throw std::invalid_argument(
                "MXFP4 slot map contains an invalid arena slot");
        }
        const auto base = static_cast<std::size_t>(expert) * kDescriptorSize;
        descriptors[base + kFamily] = kFamilyMxfp4;
        descriptors[base + kLocalExpert] = slot;
        descriptors[base + kOut] = out_per_expert;
        descriptors[base + kInput] = neuron_len;
        descriptors[base + kMxGroups] = neuron_len / 32;
        descriptors[base + kMxValueOffset] = 0;
        descriptors[base + kMxScaleOffset] = 0;
    }

    auto empty_u8 = [] {
        return make_raw_array(std::vector<std::uint8_t>{}, mlx::core::uint8);
    };
    auto empty_i8 = [] {
        return make_raw_array(std::vector<std::uint8_t>{}, mlx::core::int8);
    };
    auto empty_i16 = [] {
        return make_raw_array(std::vector<std::uint8_t>{}, mlx::core::int16);
    };
    auto empty_f16 = [] {
        return make_raw_array(std::vector<std::uint8_t>{}, mlx::core::float16);
    };
    auto empty_f32 = [] {
        return make_raw_array(std::vector<std::uint8_t>{}, mlx::core::float32);
    };
    auto impl = std::make_shared<Impl>(
        make_int32_array(descriptors, Shape{experts, kDescriptorSize}),
        empty_u8(),
        empty_u8(),
        empty_u8(),
        empty_f32(),
        empty_f32(),
        empty_i8(),
        empty_f16(),
        empty_u8(),
        empty_u8(),
        empty_u8(),
        empty_f32(),
        empty_i8(),
        empty_f32(),
        empty_u8(),
        empty_u8(),
        empty_f32(),
        empty_f32(),
        empty_i16(),
        empty_i16(),
        std::move(packed_values),
        std::move(block_scales),
        std::vector<RotationSpec>{},
        std::move(descriptors),
        experts,
        out_per_expert,
        neuron_len,
        1);
    impl->logical_experts = logical_experts > 0
        ? logical_experts
        : experts;
    return MlxMfeWeight(std::move(impl));
}

MlxMfeWeight MlxMfeWeight::concatenate_projections(
    const std::vector<MlxMfeWeight>& weights) {
    if (weights.empty()) {
        throw std::invalid_argument(
            "at least one MFE projection is required");
    }
    const auto& first = *weights.front().impl_;
    for (const auto& weight : weights) {
        const auto& current = *weight.impl_;
        if (
            current.projections != 1
            || current.experts != first.experts
            || current.out_per_expert
                != first.out_per_expert
            || current.neuron_len != first.neuron_len
        ) {
            throw std::invalid_argument(
                "MFE projections have incompatible shapes");
        }
    }

    const auto descriptor_count = checked_product(
        checked_product(
            static_cast<std::size_t>(first.experts),
            weights.size(),
            "projection descriptor count"),
        static_cast<std::size_t>(kDescriptorSize),
        "projection descriptor count");
    std::vector<std::int32_t> descriptors(
        descriptor_count,
        0);

    std::size_t nint_q_offset = 0;
    std::size_t nint_sub_offset = 0;
    std::size_t nint_anchor_offset = 0;
    std::size_t q8_q_offset = 0;
    std::size_t q8_scale_offset = 0;
    std::size_t vq_indices_offset = 0;
    std::size_t vq_state_offset = 0;
    std::size_t vq_aux_offset = 0;
    std::size_t vq_anchor_offset = 0;
    std::size_t vq_codebook_offset = 0;
    std::size_t vq_scale_offset = 0;
    std::size_t vq_state_bank_offset = 0;
    std::size_t vq_bank_offset = 0;
    std::size_t vq_parameter_offset = 0;
    std::size_t vq_residual_codebook_offset = 0;
    std::size_t vq_residual_record_offset = 0;
    std::size_t mx_value_offset = 0;
    std::size_t mx_scale_offset = 0;

    std::vector<array> nint_q_arrays;
    std::vector<array> nint_sub_scale_arrays;
    std::vector<array> nint_sub_min_arrays;
    std::vector<array> nint_anchor_scale_arrays;
    std::vector<array> nint_anchor_min_arrays;
    std::vector<array> q8_q_arrays;
    std::vector<array> q8_scale_arrays;
    std::vector<array> vq_indices_arrays;
    std::vector<array> vq_state_arrays;
    std::vector<array> vq_aux_arrays;
    std::vector<array> vq_anchor_arrays;
    std::vector<array> vq_codebook_arrays;
    std::vector<array> vq_scale_arrays;
    std::vector<array> vq_state_bank_arrays;
    std::vector<array> vq_bank_arrays;
    std::vector<array> vq_parameter_arrays;
    std::vector<array> vq_residual_codebook_arrays;
    std::vector<array> vq_residual_first_arrays;
    std::vector<array> vq_residual_second_arrays;
    std::vector<array> mx_value_arrays;
    std::vector<array> mx_scale_arrays;
    std::vector<RotationSpec> combined_rotations;
    for (
        std::size_t projection = 0;
        projection < weights.size();
        ++projection
    ) {
        const auto& source = *weights[projection].impl_;
        std::vector<int> rotation_map(
            source.rotations.size() + 1,
            0);
        for (
            std::size_t index = 0;
            index < source.rotations.size();
            ++index
        ) {
            rotation_map[index + 1] =
                rotation_variant(
                    source.rotations[index],
                    combined_rotations);
        }
        if ((q8_q_offset & 1u) != 0u) {
            q8_q_arrays.push_back(make_raw_array(
                std::vector<std::uint8_t>{0},
                mlx::core::int8));
            ++q8_q_offset;
        }
        for (
            int expert = 0;
            expert < first.experts;
            ++expert
        ) {
            const auto source_base =
                static_cast<std::size_t>(expert)
                * kDescriptorSize;
            const auto target_base = (
                static_cast<std::size_t>(expert)
                    * weights.size()
                + projection
            ) * kDescriptorSize;
            std::copy_n(
                source.descriptor_values.begin()
                    + static_cast<std::ptrdiff_t>(
                        source_base),
                kDescriptorSize,
                descriptors.begin()
                    + static_cast<std::ptrdiff_t>(
                        target_base));
            auto* descriptor =
                descriptors.data() + target_base;
            if (descriptor[kFamily] == kFamilyNint) {
                descriptor[kNintQOffset] =
                    descriptor_with_offset(
                        descriptor[kNintQOffset],
                        nint_q_offset,
                        "NINT q offset");
                descriptor[kNintSubOffset] =
                    descriptor_with_offset(
                        descriptor[kNintSubOffset],
                        nint_sub_offset,
                        "NINT sub offset");
                descriptor[kNintAnchorOffset] =
                    descriptor_with_offset(
                        descriptor[
                            kNintAnchorOffset],
                        nint_anchor_offset,
                        "NINT anchor offset");
                if (descriptor[kNintV2] != 0) {
                    descriptor[kNintRowQLayoutOffset] =
                        descriptor_with_offset(
                            descriptor[kNintRowQLayoutOffset],
                            nint_q_offset,
                            "NINTv2 q-layout offset");
                    descriptor[kNintRowQByteOffsetsOffset] =
                        descriptor_with_offset(
                            descriptor[kNintRowQByteOffsetsOffset],
                            nint_q_offset,
                            "NINTv2 q-byte-offset offset");
                }
            } else if (
                descriptor[kFamily]
                == kFamilyNint8Zero
            ) {
                descriptor[kQ8QOffset] =
                    descriptor_with_offset(
                        descriptor[kQ8QOffset],
                        q8_q_offset,
                        "NINT8-0 q offset");
                descriptor[kQ8ScaleOffset] =
                    descriptor_with_offset(
                        descriptor[kQ8ScaleOffset],
                        q8_scale_offset,
                        "NINT8-0 scale offset");
            } else if (
                descriptor[kFamily] == kFamilyBf16
                || descriptor[kFamily] == kFamilyF16
            ) {
                descriptor[kDenseValueOffset] =
                    descriptor_with_offset(
                        descriptor[kDenseValueOffset],
                        q8_q_offset,
                        "dense value offset");
            } else if (
                descriptor[kFamily] == kFamilyVq
            ) {
                descriptor[kVqIndicesOffset] =
                    descriptor_with_offset(
                        descriptor[kVqIndicesOffset],
                        vq_indices_offset,
                        "VQ index offset");
                descriptor[kVqStateOffset] =
                    descriptor_with_offset(
                        descriptor[kVqStateOffset],
                        vq_state_offset,
                        "VQ state offset");
                descriptor[kVqAuxOffset] =
                    descriptor_with_offset(
                        descriptor[kVqAuxOffset],
                        vq_aux_offset,
                        "VQ auxiliary offset");
                descriptor[kVqAnchorOffset] =
                    descriptor_with_offset(
                        descriptor[kVqAnchorOffset],
                        vq_anchor_offset,
                        "VQ anchor offset");
                descriptor[kVqCodebookOffset] =
                    descriptor_with_offset(
                        descriptor[kVqCodebookOffset],
                        vq_codebook_offset,
                        "VQ codebook offset");
                descriptor[kVqScaleOffset] =
                    descriptor_with_offset(
                        descriptor[kVqScaleOffset],
                        vq_scale_offset,
                        "VQ scale offset");
                descriptor[kVqStateBankOffset] =
                    descriptor_with_offset(
                        descriptor[
                            kVqStateBankOffset],
                        vq_state_bank_offset,
                        "VQ state-bank offset");
                descriptor[kVqBankOffset] =
                    descriptor_with_offset(
                        descriptor[kVqBankOffset],
                        vq_bank_offset,
                        "VQ bank offset");
                descriptor[kVqParameterOffset] =
                    descriptor_with_offset(
                        descriptor[
                            kVqParameterOffset],
                        vq_parameter_offset,
                        "VQ parameter offset");
                descriptor[kVqResidualCodebookOffset] =
                    descriptor_with_offset(
                        descriptor[kVqResidualCodebookOffset],
                        vq_residual_codebook_offset,
                        "VQ residual codebook offset");
                descriptor[kVqResidualRecordOffset] =
                    descriptor_with_offset(
                        descriptor[kVqResidualRecordOffset],
                        vq_residual_record_offset,
                        "VQ residual record offset");
                const int local_rotation =
                    descriptor[
                        kVqRotationVariant];
                if (
                    local_rotation < 0
                    || static_cast<std::size_t>(
                           local_rotation)
                        >= rotation_map.size()
                ) {
                    throw std::runtime_error(
                        "invalid MFE VQ rotation "
                        "variant");
                }
                descriptor[kVqRotationVariant] =
                    rotation_map[
                        static_cast<std::size_t>(
                            local_rotation)];
            } else if (
                descriptor[kFamily] == kFamilyMxfp4
                || descriptor[kFamily] == kFamilyMxfp8
            ) {
                descriptor[kMxValueOffset] =
                    descriptor_u32_with_offset(
                        descriptor[kMxValueOffset],
                        mx_value_offset,
                        "MX value offset");
                descriptor[kMxScaleOffset] =
                    descriptor_u32_with_offset(
                        descriptor[kMxScaleOffset],
                        mx_scale_offset,
                        "MX scale offset");
            } else {
                throw std::runtime_error(
                    "unsupported MFE descriptor family");
            }
        }

        nint_q_arrays.push_back(source.nint_q);
        nint_sub_scale_arrays.push_back(
            source.nint_sub_scale);
        nint_sub_min_arrays.push_back(
            source.nint_sub_min);
        nint_anchor_scale_arrays.push_back(
            source.nint_anchor_scale);
        nint_anchor_min_arrays.push_back(
            source.nint_anchor_min);
        q8_q_arrays.push_back(source.q8_q);
        q8_scale_arrays.push_back(source.q8_scales);
        vq_indices_arrays.push_back(
            source.vq_indices);
        vq_state_arrays.push_back(source.vq_state);
        vq_aux_arrays.push_back(source.vq_aux);
        vq_anchor_arrays.push_back(
            source.vq_anchors);
        vq_codebook_arrays.push_back(
            source.vq_codebooks);
        vq_scale_arrays.push_back(source.vq_scales);
        vq_state_bank_arrays.push_back(
            source.vq_state_to_codebank);
        vq_bank_arrays.push_back(source.vq_banks);
        vq_parameter_arrays.push_back(
            source.vq_parameters);
        vq_residual_codebook_arrays.push_back(
            source.vq_residual_codebooks);
        vq_residual_first_arrays.push_back(
            source.vq_residual_first);
        vq_residual_second_arrays.push_back(
            source.vq_residual_second);
        mx_value_arrays.push_back(source.mx_values);
        mx_scale_arrays.push_back(source.mx_scales);

        nint_q_offset = checked_add(
            nint_q_offset,
            source.nint_q.size(),
            "NINT q stream size");
        nint_sub_offset = checked_add(
            nint_sub_offset,
            source.nint_sub_scale.size(),
            "NINT sub stream size");
        nint_anchor_offset = checked_add(
            nint_anchor_offset,
            source.nint_anchor_scale.size(),
            "NINT anchor stream size");
        q8_q_offset = checked_add(
            q8_q_offset,
            source.q8_q.size(),
            "NINT8-0 q stream size");
        q8_scale_offset = checked_add(
            q8_scale_offset,
            source.q8_scales.size(),
            "NINT8-0 scale stream size");
        vq_indices_offset = checked_add(
            vq_indices_offset,
            source.vq_indices.size(),
            "VQ index stream size");
        vq_state_offset = checked_add(
            vq_state_offset,
            source.vq_state.size(),
            "VQ state stream size");
        vq_aux_offset = checked_add(
            vq_aux_offset,
            source.vq_aux.size(),
            "VQ auxiliary stream size");
        vq_anchor_offset = checked_add(
            vq_anchor_offset,
            source.vq_anchors.size(),
            "VQ anchor stream size");
        vq_codebook_offset = checked_add(
            vq_codebook_offset,
            source.vq_codebooks.size(),
            "VQ codebook stream size");
        vq_scale_offset = checked_add(
            vq_scale_offset,
            source.vq_scales.size(),
            "VQ scale stream size");
        vq_state_bank_offset = checked_add(
            vq_state_bank_offset,
            source.vq_state_to_codebank.size(),
            "VQ state-bank stream size");
        vq_bank_offset = checked_add(
            vq_bank_offset,
            source.vq_banks.size(),
            "VQ bank stream size");
        vq_parameter_offset = checked_add(
            vq_parameter_offset,
            source.vq_parameters.size(),
            "VQ parameter stream size");
        vq_residual_codebook_offset = checked_add(
            vq_residual_codebook_offset,
            source.vq_residual_codebooks.size(),
            "VQ residual codebook stream size");
        vq_residual_record_offset = checked_add(
            vq_residual_record_offset,
            source.vq_residual_first.size(),
            "VQ residual record stream size");
        mx_value_offset = checked_add(
            mx_value_offset,
            source.mx_values.size(),
            "MXFP4 value stream size");
        mx_scale_offset = checked_add(
            mx_scale_offset,
            source.mx_scales.size(),
            "MXFP4 scale stream size");
    }

    const int projection_count =
        checked_int(weights.size(), "projection count");
    auto impl = std::make_shared<Impl>(
        make_int32_array(
            descriptors,
            Shape{
                first.experts * projection_count,
                kDescriptorSize,
            }),
        concatenate_1d(std::move(nint_q_arrays)),
        concatenate_1d(
            std::move(nint_sub_scale_arrays)),
        concatenate_1d(
            std::move(nint_sub_min_arrays)),
        concatenate_1d(
            std::move(nint_anchor_scale_arrays)),
        concatenate_1d(
            std::move(nint_anchor_min_arrays)),
        concatenate_1d(std::move(q8_q_arrays)),
        concatenate_1d(std::move(q8_scale_arrays)),
        concatenate_1d(
            std::move(vq_indices_arrays)),
        concatenate_1d(
            std::move(vq_state_arrays)),
        concatenate_1d(std::move(vq_aux_arrays)),
        concatenate_1d(
            std::move(vq_anchor_arrays)),
        concatenate_1d(
            std::move(vq_codebook_arrays)),
        concatenate_1d(
            std::move(vq_scale_arrays)),
        concatenate_1d(
            std::move(vq_state_bank_arrays)),
        concatenate_1d(
            std::move(vq_bank_arrays)),
        concatenate_1d(
            std::move(vq_parameter_arrays)),
        concatenate_1d(
            std::move(vq_residual_codebook_arrays)),
        concatenate_1d(
            std::move(vq_residual_first_arrays)),
        concatenate_1d(
            std::move(vq_residual_second_arrays)),
        concatenate_1d(std::move(mx_value_arrays)),
        concatenate_1d(std::move(mx_scale_arrays)),
        std::move(combined_rotations),
        std::move(descriptors),
        first.experts,
        first.out_per_expert,
        first.neuron_len,
        projection_count);
    impl->automatic_mxfp4_nax_prefill = std::all_of(
        weights.begin(),
        weights.end(),
        [](const MlxMfeWeight& weight) {
            return weight.impl_->automatic_mxfp4_nax_prefill;
        });
    impl->logical_experts = first.logical_experts > 0
        ? first.logical_experts
        : first.experts;
    impl->projection_views.reserve(weights.size());
    const bool split_standalone_projections = std::any_of(
        weights.begin(),
        weights.end(),
        [](const MlxMfeWeight& weight) {
            return !weight.impl_->mxfp4_sq_cohorts.empty()
                || !weight.impl_->fp8_sq_cohorts.empty();
        });
    if (split_standalone_projections) {
        impl->packed_bytes = 0;
    }
    for (const auto& weight : weights) {
        // Keep each projection on its original packed pool.  Evaluating the
        // combined stream would otherwise copy every expert's Gate and Up
        // payload before a one-token decode can touch its selected experts.
        impl->projection_views.push_back(weight.impl_);
        if (split_standalone_projections) {
            impl->packed_bytes = checked_add(
                impl->packed_bytes,
                weight.impl_->packed_bytes,
                "split MFE projection bytes");
        }
    }
    return MlxMfeWeight(std::move(impl));
}

MlxMfeWeight MlxMfeWeight::concatenate_experts(
    const std::vector<MlxMfeWeight>& weights) {
    if (weights.empty()) {
        throw std::invalid_argument(
            "at least one MFE expert page is required");
    }
    const auto& first = *weights.front().impl_;
    for (const auto& weight : weights) {
        const auto& current = *weight.impl_;
        if (
            current.projections != 1
            || current.experts != 1
            || current.out_per_expert != first.out_per_expert
            || current.neuron_len != first.neuron_len
            || !current.reference_cohorts.empty()
            || !current.mxfp4_sq_cohorts.empty()
            || !current.fp8_sq_cohorts.empty()
            || current.standalone_owner.has_value()
        ) {
            throw std::invalid_argument(
                "MFE expert pages have incompatible layouts");
        }
    }

    const int expert_count = checked_int(
        weights.size(),
        "resident expert count");
    std::vector<std::int32_t> descriptors(
        checked_product(
            weights.size(),
            static_cast<std::size_t>(kDescriptorSize),
            "resident expert descriptor count"),
        0);

    std::size_t nint_q_offset = 0;
    std::size_t nint_sub_offset = 0;
    std::size_t nint_anchor_offset = 0;
    std::size_t q8_q_offset = 0;
    std::size_t q8_scale_offset = 0;
    std::size_t vq_indices_offset = 0;
    std::size_t vq_state_offset = 0;
    std::size_t vq_aux_offset = 0;
    std::size_t vq_anchor_offset = 0;
    std::size_t vq_codebook_offset = 0;
    std::size_t vq_scale_offset = 0;
    std::size_t vq_state_bank_offset = 0;
    std::size_t vq_bank_offset = 0;
    std::size_t vq_parameter_offset = 0;
    std::size_t vq_residual_codebook_offset = 0;
    std::size_t vq_residual_record_offset = 0;
    std::size_t mx_value_offset = 0;
    std::size_t mx_scale_offset = 0;

    std::vector<array> nint_q_arrays;
    std::vector<array> nint_sub_scale_arrays;
    std::vector<array> nint_sub_min_arrays;
    std::vector<array> nint_anchor_scale_arrays;
    std::vector<array> nint_anchor_min_arrays;
    std::vector<array> q8_q_arrays;
    std::vector<array> q8_scale_arrays;
    std::vector<array> vq_indices_arrays;
    std::vector<array> vq_state_arrays;
    std::vector<array> vq_aux_arrays;
    std::vector<array> vq_anchor_arrays;
    std::vector<array> vq_codebook_arrays;
    std::vector<array> vq_scale_arrays;
    std::vector<array> vq_state_bank_arrays;
    std::vector<array> vq_bank_arrays;
    std::vector<array> vq_parameter_arrays;
    std::vector<array> vq_residual_codebook_arrays;
    std::vector<array> vq_residual_first_arrays;
    std::vector<array> vq_residual_second_arrays;
    std::vector<array> mx_value_arrays;
    std::vector<array> mx_scale_arrays;
    std::vector<RotationSpec> combined_rotations;

    for (std::size_t expert = 0; expert < weights.size(); ++expert) {
        const auto& source = *weights[expert].impl_;
        std::vector<int> rotation_map(
            source.rotations.size() + 1,
            0);
        for (std::size_t index = 0;
             index < source.rotations.size();
             ++index) {
            rotation_map[index + 1] = rotation_variant(
                source.rotations[index],
                combined_rotations);
        }
        if ((q8_q_offset & 1u) != 0u) {
            q8_q_arrays.push_back(make_raw_array(
                std::vector<std::uint8_t>{0},
                mlx::core::int8));
            ++q8_q_offset;
        }

        const auto target_base = expert * kDescriptorSize;
        std::copy_n(
            source.descriptor_values.begin(),
            kDescriptorSize,
            descriptors.begin()
                + static_cast<std::ptrdiff_t>(target_base));
        auto* descriptor = descriptors.data() + target_base;
        if (descriptor[kFamily] == kFamilyNint) {
            descriptor[kNintQOffset] = descriptor_with_offset(
                descriptor[kNintQOffset], nint_q_offset,
                "NINT q offset");
            descriptor[kNintSubOffset] = descriptor_with_offset(
                descriptor[kNintSubOffset], nint_sub_offset,
                "NINT sub offset");
            descriptor[kNintAnchorOffset] = descriptor_with_offset(
                descriptor[kNintAnchorOffset], nint_anchor_offset,
                "NINT anchor offset");
            if (descriptor[kNintV2] != 0) {
                descriptor[kNintRowQLayoutOffset] = descriptor_with_offset(
                    descriptor[kNintRowQLayoutOffset], nint_q_offset,
                    "NINTv2 q-layout offset");
                descriptor[kNintRowQByteOffsetsOffset] = descriptor_with_offset(
                    descriptor[kNintRowQByteOffsetsOffset], nint_q_offset,
                    "NINTv2 q-byte-offset offset");
            }
        } else if (descriptor[kFamily] == kFamilyNint8Zero) {
            descriptor[kQ8QOffset] = descriptor_with_offset(
                descriptor[kQ8QOffset], q8_q_offset,
                "NINT8-0 q offset");
            descriptor[kQ8ScaleOffset] = descriptor_with_offset(
                descriptor[kQ8ScaleOffset], q8_scale_offset,
                "NINT8-0 scale offset");
        } else if (
            descriptor[kFamily] == kFamilyBf16
            || descriptor[kFamily] == kFamilyF16
        ) {
            descriptor[kDenseValueOffset] = descriptor_with_offset(
                descriptor[kDenseValueOffset], q8_q_offset,
                "dense value offset");
        } else if (descriptor[kFamily] == kFamilyVq) {
            descriptor[kVqIndicesOffset] = descriptor_with_offset(
                descriptor[kVqIndicesOffset], vq_indices_offset,
                "VQ index offset");
            descriptor[kVqStateOffset] = descriptor_with_offset(
                descriptor[kVqStateOffset], vq_state_offset,
                "VQ state offset");
            descriptor[kVqAuxOffset] = descriptor_with_offset(
                descriptor[kVqAuxOffset], vq_aux_offset,
                "VQ auxiliary offset");
            descriptor[kVqAnchorOffset] = descriptor_with_offset(
                descriptor[kVqAnchorOffset], vq_anchor_offset,
                "VQ anchor offset");
            descriptor[kVqCodebookOffset] = descriptor_with_offset(
                descriptor[kVqCodebookOffset], vq_codebook_offset,
                "VQ codebook offset");
            descriptor[kVqScaleOffset] = descriptor_with_offset(
                descriptor[kVqScaleOffset], vq_scale_offset,
                "VQ scale offset");
            descriptor[kVqStateBankOffset] = descriptor_with_offset(
                descriptor[kVqStateBankOffset], vq_state_bank_offset,
                "VQ state-bank offset");
            descriptor[kVqBankOffset] = descriptor_with_offset(
                descriptor[kVqBankOffset], vq_bank_offset,
                "VQ bank offset");
            descriptor[kVqParameterOffset] = descriptor_with_offset(
                descriptor[kVqParameterOffset], vq_parameter_offset,
                "VQ parameter offset");
            descriptor[kVqResidualCodebookOffset] = descriptor_with_offset(
                descriptor[kVqResidualCodebookOffset],
                vq_residual_codebook_offset,
                "VQ residual codebook offset");
            descriptor[kVqResidualRecordOffset] = descriptor_with_offset(
                descriptor[kVqResidualRecordOffset],
                vq_residual_record_offset,
                "VQ residual record offset");
            const int local_rotation =
                descriptor[kVqRotationVariant];
            if (local_rotation < 0
                || static_cast<std::size_t>(local_rotation)
                    >= rotation_map.size()) {
                throw std::runtime_error(
                    "invalid MFE VQ rotation variant");
            }
            descriptor[kVqRotationVariant] = rotation_map[
                static_cast<std::size_t>(local_rotation)];
        } else if (
            descriptor[kFamily] == kFamilyMxfp4
            || descriptor[kFamily] == kFamilyMxfp8
        ) {
            descriptor[kMxValueOffset] = descriptor_u32_with_offset(
                descriptor[kMxValueOffset], mx_value_offset,
                "MX value offset");
            descriptor[kMxScaleOffset] = descriptor_u32_with_offset(
                descriptor[kMxScaleOffset], mx_scale_offset,
                "MX scale offset");
        } else {
            throw std::runtime_error(
                "unsupported MFE resident expert family");
        }

        const int family = descriptor[kFamily];
        if (family == kFamilyNint) {
            nint_q_arrays.push_back(source.nint_q);
            nint_sub_scale_arrays.push_back(source.nint_sub_scale);
            nint_sub_min_arrays.push_back(source.nint_sub_min);
            nint_anchor_scale_arrays.push_back(source.nint_anchor_scale);
            nint_anchor_min_arrays.push_back(source.nint_anchor_min);
            nint_q_offset = checked_add(
                nint_q_offset, source.nint_q.size(),
                "NINT q stream size");
            nint_sub_offset = checked_add(
                nint_sub_offset, source.nint_sub_scale.size(),
                "NINT sub stream size");
            nint_anchor_offset = checked_add(
                nint_anchor_offset, source.nint_anchor_scale.size(),
                "NINT anchor stream size");
        } else if (family == kFamilyNint8Zero) {
            q8_q_arrays.push_back(source.q8_q);
            q8_scale_arrays.push_back(source.q8_scales);
            q8_q_offset = checked_add(
                q8_q_offset, source.q8_q.size(),
                "NINT8-0 q stream size");
            q8_scale_offset = checked_add(
                q8_scale_offset, source.q8_scales.size(),
                "NINT8-0 scale stream size");
        } else if (family == kFamilyBf16 || family == kFamilyF16) {
            q8_q_arrays.push_back(source.q8_q);
            q8_q_offset = checked_add(
                q8_q_offset, source.q8_q.size(),
                "dense value stream size");
        } else if (family == kFamilyVq) {
            vq_indices_arrays.push_back(source.vq_indices);
            vq_state_arrays.push_back(source.vq_state);
            vq_aux_arrays.push_back(source.vq_aux);
            vq_anchor_arrays.push_back(source.vq_anchors);
            vq_codebook_arrays.push_back(source.vq_codebooks);
            vq_scale_arrays.push_back(source.vq_scales);
            vq_state_bank_arrays.push_back(source.vq_state_to_codebank);
            vq_bank_arrays.push_back(source.vq_banks);
            vq_parameter_arrays.push_back(source.vq_parameters);
            vq_residual_codebook_arrays.push_back(
                source.vq_residual_codebooks);
            vq_residual_first_arrays.push_back(source.vq_residual_first);
            vq_residual_second_arrays.push_back(source.vq_residual_second);
            vq_indices_offset = checked_add(
                vq_indices_offset, source.vq_indices.size(),
                "VQ index stream size");
            vq_state_offset = checked_add(
                vq_state_offset, source.vq_state.size(),
                "VQ state stream size");
            vq_aux_offset = checked_add(
                vq_aux_offset, source.vq_aux.size(),
                "VQ auxiliary stream size");
            vq_anchor_offset = checked_add(
                vq_anchor_offset, source.vq_anchors.size(),
                "VQ anchor stream size");
            vq_codebook_offset = checked_add(
                vq_codebook_offset, source.vq_codebooks.size(),
                "VQ codebook stream size");
            vq_scale_offset = checked_add(
                vq_scale_offset, source.vq_scales.size(),
                "VQ scale stream size");
            vq_state_bank_offset = checked_add(
                vq_state_bank_offset, source.vq_state_to_codebank.size(),
                "VQ state-bank stream size");
            vq_bank_offset = checked_add(
                vq_bank_offset, source.vq_banks.size(),
                "VQ bank stream size");
            vq_parameter_offset = checked_add(
                vq_parameter_offset, source.vq_parameters.size(),
                "VQ parameter stream size");
            vq_residual_codebook_offset = checked_add(
                vq_residual_codebook_offset,
                source.vq_residual_codebooks.size(),
                "VQ residual codebook stream size");
            vq_residual_record_offset = checked_add(
                vq_residual_record_offset,
                source.vq_residual_first.size(),
                "VQ residual record stream size");
        } else {
            mx_value_arrays.push_back(source.mx_values);
            mx_scale_arrays.push_back(source.mx_scales);
            mx_value_offset = checked_add(
                mx_value_offset, source.mx_values.size(),
                "MX value stream size");
            mx_scale_offset = checked_add(
                mx_scale_offset, source.mx_scales.size(),
                "MX scale stream size");
        }
    }

    const auto concatenate_or_empty = [](
        std::vector<array> values,
        Dtype dtype) {
        return values.empty()
            ? make_raw_array(
                  std::vector<std::uint8_t>{},
                  dtype)
            : concatenate_1d(std::move(values));
    };
    auto impl = std::make_shared<Impl>(
        make_int32_array(
            descriptors,
            Shape{expert_count, kDescriptorSize}),
        concatenate_or_empty(std::move(nint_q_arrays), mlx::core::uint8),
        concatenate_or_empty(
            std::move(nint_sub_scale_arrays), mlx::core::uint8),
        concatenate_or_empty(
            std::move(nint_sub_min_arrays), mlx::core::uint8),
        concatenate_or_empty(
            std::move(nint_anchor_scale_arrays), mlx::core::float32),
        concatenate_or_empty(
            std::move(nint_anchor_min_arrays), mlx::core::float32),
        concatenate_or_empty(std::move(q8_q_arrays), mlx::core::int8),
        concatenate_or_empty(
            std::move(q8_scale_arrays), mlx::core::float16),
        concatenate_or_empty(
            std::move(vq_indices_arrays), mlx::core::uint8),
        concatenate_or_empty(std::move(vq_state_arrays), mlx::core::uint8),
        concatenate_or_empty(std::move(vq_aux_arrays), mlx::core::uint8),
        concatenate_or_empty(
            std::move(vq_anchor_arrays), mlx::core::float32),
        concatenate_or_empty(
            std::move(vq_codebook_arrays), mlx::core::int8),
        concatenate_or_empty(
            std::move(vq_scale_arrays), mlx::core::float32),
        concatenate_or_empty(
            std::move(vq_state_bank_arrays), mlx::core::uint8),
        concatenate_or_empty(std::move(vq_bank_arrays), mlx::core::uint8),
        concatenate_or_empty(
            std::move(vq_parameter_arrays), mlx::core::float32),
        concatenate_or_empty(
            std::move(vq_residual_codebook_arrays), mlx::core::float32),
        concatenate_or_empty(
            std::move(vq_residual_first_arrays), mlx::core::int16),
        concatenate_or_empty(
            std::move(vq_residual_second_arrays), mlx::core::int16),
        concatenate_or_empty(std::move(mx_value_arrays), mlx::core::uint8),
        concatenate_or_empty(std::move(mx_scale_arrays), mlx::core::uint8),
        std::move(combined_rotations),
        std::move(descriptors),
        expert_count,
        first.out_per_expert,
        first.neuron_len,
        1);
    impl->automatic_mxfp4_nax_prefill = std::all_of(
        weights.begin(),
        weights.end(),
        [](const MlxMfeWeight& weight) {
            return weight.impl_->automatic_mxfp4_nax_prefill;
        });
    return MlxMfeWeight(std::move(impl));
}

MlxMfeWeight MlxMfeWeight::with_automatic_mxfp4_nax_prefill(
    bool enabled) const {
    auto impl = std::make_shared<Impl>(*impl_);
    impl->automatic_mxfp4_nax_prefill = enabled;
    for (auto& projection : impl->projection_views) {
        projection = MlxMfeWeight(projection)
            .with_automatic_mxfp4_nax_prefill(enabled)
            .impl_;
    }
    return MlxMfeWeight(std::move(impl));
}

MlxMoeWeight load_routed_gate_up_weight(
    const MfqContainer& model,
    const std::string& mlp_prefix) {
    const auto base = mlp_prefix + ".experts";
    const auto gate_name = base + ".gate.weight";
    const auto up_name = base + ".up.weight";
    const bool has_gate = model.contains(gate_name);
    const bool has_up = model.contains(up_name);
    if (has_gate != has_up) {
        throw std::runtime_error(
            "incomplete routed Gate/Up pair under " + base);
    }
    const auto load = [&model](const std::string& name) {
        if (model.record(name).dtype != "MFE") {
            throw std::runtime_error(
                "routed expert tensor must use MFE: " + name);
        }
        const auto mapped = model.map_record(name);
        return MlxMoeWeight::from_blob(mapped.view());
    };
    if (has_gate) {
        return MlxMoeWeight::concatenate_projections(
            std::vector<MlxMoeWeight>{load(gate_name), load(up_name)});
    }
    return load(base + ".gate_up.weight");
}

array MlxMfeWeight::routed_matmul(
    const array& input,
    const array& expert_ids) const {
    return routed_matmul_impl(
        input,
        expert_ids,
        false,
        0.0f);
}

array MlxMfeWeight::routed_matmul_mapped(
    const array& input,
    const array& expert_ids,
    const array& expert_map) const {
    return routed_matmul_impl(
        input,
        expert_ids,
        false,
        0.0f,
        &expert_map);
}

array MlxMfeWeight::routed_matmul_packed(
    const array& input,
    const array& packed_expert_ids) const {
    return routed_matmul_impl(
        input,
        packed_expert_ids,
        false,
        0.0f,
        nullptr,
        true);
}

array MlxMfeWeight::routed_swiglu(
    const array& input,
    const array& expert_ids,
    float limit) const {
    const bool split_gate_up = impl_->projections == 2;
    if (
        (!split_gate_up && impl_->projections != 1)
        || impl_->out_per_expert <= 0
        || (!split_gate_up && (impl_->out_per_expert & 1) != 0)
    ) {
        throw std::invalid_argument(
            "MFE SwiGLU requires one fused or two split gate/up projections");
    }
    if (!std::isfinite(limit) || limit < 0.0f) {
        throw std::invalid_argument(
            "MFE SwiGLU limit must be finite and non-negative");
    }
    if (split_gate_up) {
        if (
            input.ndim() >= 2
            && input.shape(0) >= 32
            && impl_->grouped_mmq
            && impl_->rotations.empty()
        ) {
            return routed_matmul_impl(
                input,
                expert_ids,
                true,
                limit);
        }
        auto gate_up = routed_matmul_impl(
            input,
            expert_ids,
            false,
            0.0f);
        return limit > 0.0f
            ? moe_limited_swiglu_split(gate_up, limit)
            : moe_swiglu_split(gate_up);
    }
    return routed_matmul_impl(
        input,
        expert_ids,
        true,
        limit);
}

array MlxMfeWeight::routed_swiglu_mapped(
    const array& input,
    const array& expert_ids,
    const array& expert_map,
    float limit) const {
    const bool split_gate_up = impl_->projections == 2;
    if (
        (!split_gate_up && impl_->projections != 1)
        || impl_->out_per_expert <= 0
        || (!split_gate_up && (impl_->out_per_expert & 1) != 0)
    ) {
        throw std::invalid_argument(
            "MFE SwiGLU requires one fused or two split gate/up projections");
    }
    if (!std::isfinite(limit) || limit < 0.0f) {
        throw std::invalid_argument(
            "MFE SwiGLU limit must be finite and non-negative");
    }
    if (split_gate_up) {
        auto gate_up = routed_matmul_impl(
            input,
            expert_ids,
            false,
            0.0f,
            &expert_map);
        return limit > 0.0f
            ? moe_limited_swiglu_split(gate_up, limit)
            : moe_swiglu_split(gate_up);
    }
    return routed_matmul_impl(
        input,
        expert_ids,
        true,
        limit,
        &expert_map);
}

array MlxMfeWeight::routed_swiglu_packed(
    const array& input,
    const array& packed_expert_ids,
    float limit) const {
    const bool split_gate_up = impl_->projections == 2;
    if (
        (!split_gate_up && impl_->projections != 1)
        || impl_->out_per_expert <= 0
        || (!split_gate_up && (impl_->out_per_expert & 1) != 0)
    ) {
        throw std::invalid_argument(
            "MFE SwiGLU requires one fused or two split gate/up projections");
    }
    if (!std::isfinite(limit) || limit < 0.0f) {
        throw std::invalid_argument(
            "MFE SwiGLU limit must be finite and non-negative");
    }
    if (split_gate_up) {
        auto gate_up = routed_matmul_impl(
            input,
            packed_expert_ids,
            false,
            0.0f,
            nullptr,
            true);
        return limit > 0.0f
            ? moe_limited_swiglu_split(gate_up, limit)
            : moe_swiglu_split(gate_up);
    }
    return routed_matmul_impl(
        input,
        packed_expert_ids,
        true,
        limit,
        nullptr,
        true);
}

array MlxMfeWeight::routed_swiglu_pair(
    const MlxMfeWeight& up,
    const array& input,
    const array& expert_ids,
    float limit) const {
    if (!std::isfinite(limit) || limit < 0.0f) {
        throw std::invalid_argument(
            "paired MFE SwiGLU limit must be finite and non-negative");
    }
    if (impl_->experts != up.impl_->experts
        || impl_->out_per_expert != up.impl_->out_per_expert
        || impl_->neuron_len != up.impl_->neuron_len
        || impl_->projections != 1
        || up.impl_->projections != 1) {
        throw std::invalid_argument(
            "paired MFE Gate and Up weights are incompatible");
    }
    const auto fallback = [&]() {
        auto gate_output = routed_matmul(input, expert_ids);
        auto up_output = up.routed_matmul(input, expert_ids);
        if (limit > 0.0f) {
            return moe_limited_swiglu_pair(
                std::move(gate_output),
                std::move(up_output),
                limit);
        }
        return moe_swiglu_split(mlx::core::concatenate(
            {std::move(gate_output), std::move(up_output)},
            -1));
    };
    const bool native_pair =
        impl_->native_primitive
        && up.impl_->native_primitive
        && impl_->family_mask
            == (std::uint32_t{1} << kFamilyMxfp4)
        && up.impl_->family_mask
            == (std::uint32_t{1} << kFamilyMxfp4)
        && impl_->rotations.empty()
        && up.impl_->rotations.empty()
        && (impl_->k_lanes_override == 0
            || impl_->k_lanes_override == 8)
        && (up.impl_->k_lanes_override == 0
            || up.impl_->k_lanes_override == 8)
        && input.ndim() == 2
        && expert_ids.ndim() == 2
        && input.shape(0) >= 1
        && input.shape(0) <= 6
        && expert_ids.shape(0) == input.shape(0)
        && expert_ids.shape(1) >= 1
        && expert_ids.shape(1) <= 16
        && input.shape(1) == impl_->neuron_len
        && impl_->out_per_expert > 0
        && impl_->neuron_len > 0
        && impl_->neuron_len % 32 == 0
        && impl_->descriptor_values.size()
            == static_cast<std::size_t>(impl_->experts)
                * kDescriptorSize
        && up.impl_->descriptor_values.size()
            == static_cast<std::size_t>(up.impl_->experts)
                * kDescriptorSize;
    if (!native_pair || mlx_reference_enabled()) {
        detail::profile_marker("mfe.dispatch.mxfp4_pair_fallback");
        return fallback();
    }

    detail::profile_marker("mfe.dispatch.mxfp4_pair_native");

    auto source = input;
    if (source.dtype() != mlx::core::float16
        && source.dtype() != mlx::core::float32) {
        source = mlx::core::astype(source, mlx::core::float16);
    }
    source = mlx::core::contiguous(std::move(source));
    auto ids = mlx::core::contiguous(
        mlx::core::astype(expert_ids, mlx::core::int32));
    const int tokens = ids.shape(0);
    const int routes = ids.shape(1);
    constexpr int kOutputsPerWorkgroup = 8;
    const auto output_tiles =
        (static_cast<std::size_t>(impl_->out_per_expert)
             + kOutputsPerWorkgroup - 1u)
        / kOutputsPerWorkgroup;
    const auto routed_rows = checked_product(
        static_cast<std::size_t>(tokens),
        static_cast<std::size_t>(routes),
        "MXFP4 pair routed rows");
    const int workgroups = checked_int(
        checked_product(
            routed_rows,
            output_tiles,
            "MXFP4 pair workgroups"),
        "MXFP4 pair workgroups");
    return mxfp4_pair_swiglu_dispatch(
        {
            impl_->descriptors,
            impl_->mx_values,
            impl_->mx_scales,
            up.impl_->descriptors,
            up.impl_->mx_values,
            up.impl_->mx_scales,
            std::move(source),
            std::move(ids),
            array({limit}, mlx::core::float32),
        },
        Mxfp4PairSwiGluConfig{
            .dtype = input.dtype() == mlx::core::float32
                ? mlx::core::float32
                : mlx::core::float16,
            .output_shape = Shape{
                tokens,
                routes,
                impl_->out_per_expert,
            },
            .tokens = tokens,
            .routes = routes,
            .experts = impl_->experts,
            .output_width = impl_->out_per_expert,
            .input_width = impl_->neuron_len,
            .descriptor_size = kDescriptorSize,
            .workgroups = workgroups,
        });
}

array MlxMfeWeight::routed_matmul_reduce(
    const array& input,
    const array& expert_ids,
    const array& route_weights) const {
    const auto fallback = [&]() {
        return moe_weighted_reduce(
            routed_matmul(input, expert_ids),
            route_weights);
    };
    if (!mxfp4_decode_down_reduce_enabled()
        || mlx_reference_enabled()
        || !impl_->native_primitive
        || impl_->family_mask
            != (std::uint32_t{1} << kFamilyMxfp4)
        || impl_->projections != 1
        || !impl_->rotations.empty()
        || (impl_->k_lanes_override != 0
            && impl_->k_lanes_override != 8)
        || input.ndim() != 3
        || expert_ids.ndim() != 2
        || route_weights.ndim() != 2
        || input.shape(0) < 1
        || input.shape(0) > 6
        || expert_ids.shape(0) != input.shape(0)
        || route_weights.shape(0) != input.shape(0)
        || input.shape(1) != expert_ids.shape(1)
        || route_weights.shape(1) != expert_ids.shape(1)
        || input.shape(2) != impl_->neuron_len
        || expert_ids.shape(1) <= 0
        || expert_ids.shape(1) > 16
        || impl_->out_per_expert <= 0
        || impl_->neuron_len <= 0
        || impl_->neuron_len % 32 != 0
        || impl_->descriptor_values.size()
            != static_cast<std::size_t>(impl_->experts)
                * kDescriptorSize) {
        detail::profile_marker("mfe.dispatch.mxfp4_down_reduce_fallback");
        return fallback();
    }

    detail::profile_marker("mfe.dispatch.mxfp4_down_reduce_native");

    auto source = input;
    if (source.dtype() != mlx::core::float16
        && source.dtype() != mlx::core::float32) {
        source = mlx::core::astype(source, mlx::core::float16);
    }
    source = mlx::core::contiguous(std::move(source));
    auto ids = mlx::core::contiguous(
        mlx::core::astype(expert_ids, mlx::core::int32));
    auto weights = mlx::core::contiguous(
        mlx::core::astype(route_weights, mlx::core::float32));
    const int tokens = ids.shape(0);
    const int routes = ids.shape(1);
    const int rows_per_lane_group =
        mxfp4_decode_down_reduce_rows();
    const int outputs_per_workgroup =
        4 * rows_per_lane_group;
    const int workgroups = checked_int(
        checked_product(
            static_cast<std::size_t>(tokens),
            (static_cast<std::size_t>(impl_->out_per_expert)
                 + static_cast<std::size_t>(outputs_per_workgroup) - 1u)
                / static_cast<std::size_t>(outputs_per_workgroup),
            "MXFP4 decode down-reduce workgroups"),
        "MXFP4 decode down-reduce workgroups");
    return mxfp4_decode_reduce_dispatch(
        {
            impl_->descriptors,
            impl_->mx_values,
            impl_->mx_scales,
            std::move(source),
            std::move(ids),
            std::move(weights),
        },
        Mxfp4DecodeReduceConfig{
            .dtype = input.dtype() == mlx::core::float32
                ? mlx::core::float32
                : mlx::core::float16,
            .output_shape = Shape{tokens, impl_->out_per_expert},
            .tokens = tokens,
            .routes = routes,
            .experts = impl_->experts,
            .output_width = impl_->out_per_expert,
            .input_width = impl_->neuron_len,
            .descriptor_size = kDescriptorSize,
            .rows_per_lane_group = rows_per_lane_group,
            .workgroups = workgroups,
        });
}

array MlxMfeWeight::routed_matmul_reduce_packed(
    const array& input,
    const array& packed_expert_ids,
    const array& route_weights) const {
    // Packed cache routes store the physical arena slot in the upper bits.
    // Decode the slots on device, then reuse the same fused MXFP4
    // down-projection/reduction primitive as the host-remapped path.
    auto physical_ids = mlx::core::subtract(
        mlx::core::right_shift(
            mlx::core::contiguous(mlx::core::astype(
                packed_expert_ids,
                mlx::core::int32)),
            array(8, mlx::core::int32)),
        array(1, mlx::core::int32));
    return routed_matmul_reduce(
        input,
        mlx::core::contiguous(std::move(physical_ids)),
        route_weights);
}

bool MlxMfeWeight::supports_grouped_mmq() const noexcept {
    // The block-list builder uses one Metal thread and one threadgroup-array
    // entry per addressable expert. Large shared SSD arenas can expose more
    // than the Metal 1024-thread limit even though a model routes to only a
    // small subset. They must use the ordinary mapped routed kernel until the
    // builder itself is made multi-threadgroup.
    if (impl_->experts > 1024) return false;
    if (impl_->projections == 1) return impl_->grouped_mmq;
    return !impl_->projection_views.empty()
        && std::all_of(
            impl_->projection_views.begin(),
            impl_->projection_views.end(),
            [](const auto& projection) {
                return projection->grouped_mmq;
            });
}

bool MlxMfeWeight::prefers_mxfp4_smallm_nax(
    const array& expert_ids) const noexcept {
    const int policy_experts = impl_->logical_experts > 0
        ? impl_->logical_experts
        : impl_->experts;
    return impl_->mxfp4_slot_ids.has_value()
        && impl_->projections == 1
        && impl_->rotations.empty()
        && expert_ids.ndim() == 2
        && mxfp4_nax_smallm_preferred(
            expert_ids,
            expert_ids.shape(0),
            impl_->experts,
            policy_experts,
            impl_->neuron_len,
            impl_->out_per_expert);
}

bool MlxMfeWeight::prefers_mxfp4_nax_prefill(
    int route_count) const noexcept {
    return route_count > 0
        && impl_->mxfp4_slot_ids.has_value()
        && impl_->projections == 1
        && impl_->rotations.empty()
        && mxfp4_nax_prefill_enabled(
            route_count,
            impl_->experts,
            impl_->neuron_len,
            impl_->out_per_expert,
            impl_->automatic_mxfp4_nax_prefill);
}

bool MlxMfeWeight::supports_mxfp4_blocks() const noexcept {
    return impl_->mxfp4_slot_ids.has_value()
        && impl_->mxfp4_slot_ids->dtype() == mlx::core::int32
        && impl_->mxfp4_slot_ids->size()
            == static_cast<std::size_t>(impl_->experts)
        && impl_->projections == 1
        && impl_->rotations.empty()
        && impl_->family_mask
            == (std::uint32_t{1} << kFamilyMxfp4);
}

bool MlxMfeWeight::supports_mxfp4_pair_blocks(
    const MlxMfeWeight& other) const noexcept {
    return supports_mxfp4_blocks()
        && other.supports_mxfp4_blocks()
        && impl_->experts == other.impl_->experts
        && impl_->out_per_expert == other.impl_->out_per_expert
        && impl_->neuron_len == other.impl_->neuron_len;
}

array MlxMfeWeight::mxfp4_blocks_sorted_impl(
    const MlxMfeWeight* other,
    const array& sorted_input,
    const MlxGroupedMmqPlan& plan) const {
    const bool paired = other != nullptr;
    if (
        sorted_input.ndim() != 2
        || sorted_input.shape(0) != plan.route_count
        || sorted_input.shape(1) != impl_->neuron_len
        || plan.experts != impl_->experts
        || plan.max_blocks <= 0
        || plan.block_rows != 32
    ) {
        throw std::invalid_argument(
            "MXFP4 block plan or sorted input is incompatible");
    }

    const auto packed_view = [](const Impl& impl) {
        const auto stride = checked_product(
            static_cast<std::size_t>(impl.out_per_expert),
            static_cast<std::size_t>(impl.neuron_len / 2),
            "MXFP4 block packed stride");
        if (stride == 0 || impl.mx_values.size() % stride != 0) {
            throw std::runtime_error(
                "MXFP4 block packed arena geometry is invalid");
        }
        const auto slots = impl.mx_values.size() / stride;
        return mlx::core::reshape(
            mlx::core::view(impl.mx_values, mlx::core::uint32),
            Shape{
                checked_int(slots, "MXFP4 block slot count"),
                impl.out_per_expert,
                impl.neuron_len / 8,
            });
    };
    const auto scale_view = [](const Impl& impl) {
        const auto stride = checked_product(
            static_cast<std::size_t>(impl.out_per_expert),
            static_cast<std::size_t>(impl.neuron_len / 32),
            "MXFP4 block scale stride");
        if (stride == 0 || impl.mx_scales.size() % stride != 0) {
            throw std::runtime_error(
                "MXFP4 block scale arena geometry is invalid");
        }
        const auto slots = impl.mx_scales.size() / stride;
        return mlx::core::reshape(
            impl.mx_scales,
            Shape{
                checked_int(slots, "MXFP4 block scale slot count"),
                impl.out_per_expert,
                impl.neuron_len / 32,
            });
    };
    // Match oMLX's DeepSeek-V4 MXFP4 path: the following SwiGLU stage already
    // consumes FP16, so running the block GEMM in BF16 only adds cost before
    // an unavoidable cast and does not preserve additional output precision.
    auto source = sorted_input.dtype() == mlx::core::float16
        ? sorted_input
        : mlx::core::astype(sorted_input, mlx::core::float16);
    source = mlx::core::expand_dims(
        mlx::core::contiguous(std::move(source)),
        1);
    const auto& second = paired ? *other->impl_ : *impl_;
    return mxfp4_blocks_dispatch(
        {
            std::move(source),
            packed_view(*impl_),
            scale_view(*impl_),
            *impl_->mxfp4_slot_ids,
            packed_view(second),
            scale_view(second),
            *second.mxfp4_slot_ids,
            plan.block_meta,
            plan.block_count,
        },
        Mxfp4BlocksConfig{
            .output_shape = Shape{
                plan.route_count,
                (paired ? 2 : 1) * impl_->out_per_expert,
            },
            .route_count = plan.route_count,
            .max_blocks = plan.max_blocks,
            .experts = impl_->experts,
            .output_width = impl_->out_per_expert,
            .input_width = impl_->neuron_len,
            .block_rows = plan.block_rows,
            .paired = paired,
        });
}

array MlxMfeWeight::mxfp4_block_matmul_sorted(
    const array& sorted_input,
    const MlxGroupedMmqPlan& plan) const {
    if (!supports_mxfp4_blocks()) {
        throw std::invalid_argument(
            "MXFP4 blocks require a pure-MXFP4 weight");
    }
    return mxfp4_blocks_sorted_impl(nullptr, sorted_input, plan);
}

array MlxMfeWeight::mxfp4_pair_swiglu_sorted(
    const MlxMfeWeight& other,
    const array& sorted_input,
    const MlxGroupedMmqPlan& plan,
    float limit) const {
    if (!supports_mxfp4_pair_blocks(other)) {
        throw std::invalid_argument(
            "MXFP4 pair blocks require compatible pure-MXFP4 weights");
    }
    if (!std::isfinite(limit) || limit < 0.0f) {
        throw std::invalid_argument(
            "MXFP4 pair SwiGLU limit must be finite and non-negative");
    }
    auto pair = mxfp4_blocks_sorted_impl(&other, sorted_input, plan);
    return limit > 0.0f
        ? moe_limited_swiglu_split(std::move(pair), limit)
        : moe_swiglu_split(std::move(pair));
}

int MlxMfeWeight::recommended_mxfp4_nax_prefill_tokens(
    int routes_per_token) const noexcept {
    if (routes_per_token <= 0) return 0;
    if (!impl_->projection_views.empty()) {
        int recommendation = 0;
        for (const auto& projection : impl_->projection_views) {
            const int candidate = MlxMfeWeight(projection)
                .recommended_mxfp4_nax_prefill_tokens(routes_per_token);
            if (candidate == 0) return 0;
            recommendation = recommendation == 0
                ? candidate
                : std::min(recommendation, candidate);
        }
        return recommendation;
    }
    if (!impl_->mxfp4_slot_ids.has_value() ||
        impl_->projections != 1 || !impl_->rotations.empty()) {
        return 0;
    }
    // MLX 0.32's sorted NAX kernel is valid through 32768 routed rows.
    // Align token chunks to 32 for the packed prefill tiling. Top-k=6
    // therefore yields a 5440-token recommendation.
    constexpr int kMaximumSortedRows = 32768;
    constexpr int kTokenAlignment = 32;
    const int maximum_tokens =
        (kMaximumSortedRows / routes_per_token / kTokenAlignment) *
        kTokenAlignment;
    if (maximum_tokens <= 0 ||
        maximum_tokens >
            std::numeric_limits<int>::max() / routes_per_token) {
        return 0;
    }
    // The 256-expert 4096<->2048 geometry performs best with balanced 4096
    // token chunks on M3 Ultra; approaching the sorted-row ceiling regresses
    // its second chunk on long prompts. Keep this recommendation based on
    // operator geometry and device capabilities, not on a model identifier.
    const bool m3_ultra_256_geometry = apple_m3_ultra() &&
        impl_->experts == 256 && (
            (impl_->neuron_len == 4096 &&
             (impl_->out_per_expert == 2048 ||
              impl_->out_per_expert == 4096)) ||
            (impl_->neuron_len == 2048 &&
             impl_->out_per_expert == 4096));
    const int recommended_tokens = m3_ultra_256_geometry
        ? std::min(maximum_tokens, 4096)
        : maximum_tokens;
    const int route_count = recommended_tokens * routes_per_token;
    return mxfp4_nax_prefill_enabled(
               route_count,
               impl_->experts,
               impl_->neuron_len,
               impl_->out_per_expert,
               impl_->automatic_mxfp4_nax_prefill)
        ? recommended_tokens
        : 0;
}

int MlxMfeWeight::recommended_grouped_mmq_block_rows(
    int route_count,
    bool fused_swiglu) const noexcept {
    const bool fused_split_projections =
        fused_swiglu
        && impl_->projections == 2
        && impl_->grouped_mmq
        && impl_->rotations.empty();
    if (!impl_->projection_views.empty() && !fused_split_projections) {
        int selected = 0;
        for (const auto& projection : impl_->projection_views) {
            const int candidate = MlxMfeWeight(projection)
                .recommended_grouped_mmq_block_rows(route_count, false);
            if (selected != 0 && selected != candidate) return 32;
            selected = candidate;
        }
        return selected == 0 ? 32 : selected;
    }
    const bool use_nax = !impl_->has_nepq_residual
        && mixed_grouped_nax_enabled(route_count);
    constexpr bool direct_nax = false;
    if (!use_nax || direct_nax || impl_->experts <= 0) {
        return 32;
    }
    if (const char* value = std::getenv(
            "MFQ_METAL_GROUPED_MMQ_BLOCK_ROWS")) {
        if (std::string_view(value) == "64") {
            return 64;
        }
        if (std::string_view(value) == "48") {
            return 48;
        }
        if (std::string_view(value) == "96") {
            return 96;
        }
        if (std::string_view(value) == "80") {
            return 80;
        }
        if (std::string_view(value) == "32") {
            return 32;
        }
    }
    const int mean_routes = (route_count + impl_->experts - 1)
        / impl_->experts;
    // Up to 96 mean routes, use one 16-row-aligned block per expert to avoid
    // wasting SIMD rows. Once multiple blocks are inevitable, choose between
    // the two high-occupancy plans by their actual padded-row cost. This keeps
    // the decision independent of model family, projection role, and packed
    // weight format while still preferring BM64 when both plans do equal work.
    if (mean_routes <= 32) {
        return 32;
    }
    if (mean_routes <= 48) {
        return 48;
    }
    if (mean_routes <= 64) {
        return 64;
    }
    if (mean_routes <= 80) {
        return 80;
    }
    if (mean_routes <= 96) {
        return 96;
    }
    (void)fused_swiglu;
    const auto padded_rows = [mean_routes](int rows) {
        return (
            (static_cast<std::int64_t>(mean_routes) + rows - 1)
            / rows
        ) * rows;
    };
    return padded_rows(80) < padded_rows(64) ? 80 : 64;
}

MlxGroupedMmqPlan MlxMfeWeight::build_grouped_mmq_plan(
    const array& expert_ids,
    const array& route_order,
    int block_rows) const {
    if (!supports_grouped_mmq()) {
        throw std::invalid_argument(
            "weight does not support grouped MMQ");
    }
    if (!impl_->projection_views.empty()) {
        return MlxMfeWeight(impl_->projection_views.front())
            .build_grouped_mmq_plan(
                expert_ids,
                route_order,
                block_rows);
    }
    return make_grouped_mmq_plan(
        expert_ids,
        route_order,
        *impl_->expert_order,
        impl_->experts,
        block_rows);
}

array MlxMfeWeight::routed_matmul_sorted(
    const array& input,
    const array& expert_ids,
    const array& route_order_value,
    bool input_is_sorted,
    bool fused_swiglu,
    float swiglu_limit,
    const MlxGroupedMmqPlan* plan,
    bool force_mxfp4_nax) const {
    const bool fused_split_projections =
        fused_swiglu
        && impl_->projections == 2
        && impl_->grouped_mmq
        && impl_->rotations.empty();
    if (!impl_->projection_views.empty() && !fused_split_projections) {
        if (fused_swiglu && impl_->projection_views.size() != 2) {
            throw std::invalid_argument(
                "split grouped SwiGLU requires exactly two projections");
        }
        if (!std::isfinite(swiglu_limit) || swiglu_limit < 0.0f) {
            throw std::invalid_argument(
                "grouped SwiGLU limit must be finite and non-negative");
        }
        std::vector<array> outputs;
        outputs.reserve(impl_->projection_views.size());
        for (const auto& projection : impl_->projection_views) {
            outputs.push_back(
                MlxMfeWeight(projection).routed_matmul_sorted(
                    input,
                    expert_ids,
                    route_order_value,
                    input_is_sorted,
                    false,
                    0.0f,
                    plan,
                    force_mxfp4_nax));
        }
        auto result = mlx::core::concatenate(std::move(outputs), 1);
        if (!fused_swiglu) return result;
        return swiglu_limit > 0.0f
            ? moe_limited_swiglu_split(result, swiglu_limit)
            : moe_swiglu_split(result);
    }
    const bool direct_mxfp4_nax =
        force_mxfp4_nax &&
        impl_->mxfp4_slot_ids.has_value() &&
        impl_->projections == 1 &&
        impl_->rotations.empty();
    if (!supports_grouped_mmq() && !direct_mxfp4_nax) {
        throw std::invalid_argument(
            "weight does not support grouped MMQ");
    }
    if (
        fused_swiglu
        && (
            impl_->out_per_expert <= 0
            || (impl_->projections == 1
                && (impl_->out_per_expert & 1) != 0)
            || (impl_->projections != 1
                && impl_->projections != 2)
            || !std::isfinite(swiglu_limit)
            || swiglu_limit < 0.0f
        )
    ) {
        throw std::invalid_argument(
            "fused grouped VQ SwiGLU parameters are invalid");
    }
    auto ids = mlx::core::contiguous(
        mlx::core::astype(
            expert_ids,
            mlx::core::int32));
    if (ids.ndim() != 2) {
        throw std::invalid_argument(
            "routed expert IDs must have [tokens,routes] shape");
    }
    const int tokens = ids.shape(0);
    const int routes = ids.shape(1);
    const auto route_count_size = checked_product(
        static_cast<std::size_t>(tokens),
        static_cast<std::size_t>(routes),
        "route count");
    const int route_count = checked_int(
        route_count_size,
        "route count");
    auto route_order = mlx::core::contiguous(
        mlx::core::astype(
            route_order_value,
            mlx::core::int32));
    if (
        route_order.ndim() != 1
        || route_order.size() != route_count_size
    ) {
        throw std::invalid_argument(
            "route order must contain one index per routed row");
    }

    bool shared_input = false;
    if (input_is_sorted) {
        if (
            input.ndim() != 2
            || input.shape(0) != route_count
            || input.shape(1) != impl_->neuron_len
        ) {
            throw std::invalid_argument(
                "sorted routed input must have [tokens*routes,K] shape");
        }
    } else if (
        input.ndim() == 2
        && input.shape(0) == tokens
        && input.shape(1) == impl_->neuron_len
    ) {
        shared_input = true;
    } else if (
        input.ndim() != 3
        || input.shape(0) != tokens
        || input.shape(1) != routes
        || input.shape(2) != impl_->neuron_len
    ) {
        throw std::invalid_argument(
            "routed input must have [tokens,K] or [tokens,routes,K] shape");
    }

    // This is a native MLX gather-QMM route. The old "NAX" identifiers are
    // retained at the public/internal compatibility boundary, but do not
    // imply that the current device must provide hardware NAX instructions.
    const bool use_mxfp4_nax =
        (mxfp4_nax_prefill_enabled(
             route_count,
             impl_->experts,
             impl_->neuron_len,
             impl_->out_per_expert,
             impl_->automatic_mxfp4_nax_prefill) ||
         force_mxfp4_nax)
        && impl_->mxfp4_slot_ids.has_value()
        && impl_->projections == 1
        && impl_->rotations.empty();
    // MLX's native gather-QMM accepts both BF16 and FP16 activations. Keep
    // BF16 intact on that path so raw-HF DeepSeek-V4 matches the checkpoint's
    // activation dtype. The heterogeneous Metal kernels still consume FP16,
    // so only the exact native MXFP4 path may bypass the existing cast.
    const bool preserve_bfloat16 =
        use_mxfp4_nax && input.dtype() == mlx::core::bfloat16;
    auto source = input.dtype() == mlx::core::float16 || preserve_bfloat16
        ? input
        : mlx::core::astype(input, mlx::core::float16);
    source = mlx::core::contiguous(source);
    const int variant_stride = route_count;
    if (!impl_->rotations.empty()) {
        if (shared_input) {
            source = mlx::core::broadcast_to(
                mlx::core::reshape(
                    source,
                    Shape{tokens, 1, impl_->neuron_len}),
                Shape{tokens, routes, impl_->neuron_len});
        }
        source = mlx::core::contiguous(
            mlx::core::reshape(
                source,
                Shape{route_count, impl_->neuron_len}));
        std::vector<array> variants;
        variants.reserve(impl_->rotations.size() + 1);
        variants.push_back(source);
        for (const auto& rotation : impl_->rotations) {
            variants.push_back(
                apply_rotation(
                    source,
                    rotation,
                    variant_stride,
                    impl_->neuron_len));
        }
        source = mlx::core::contiguous(
            mlx::core::concatenate(
                std::move(variants),
                0));
        shared_input = false;
    }

    // Prefill-sized pure-MXFP4 projections can use MLX's batched gather-QMM
    // without repacking. The automatic policy is selected from device and
    // tensor geometry; model identity is deliberately irrelevant.
    if (use_mxfp4_nax) {
        array sorted_source = [&]() {
            if (input_is_sorted) {
                return source;
            }
            if (shared_input) {
                auto source_rows = mlx::core::floor_divide(
                    route_order,
                    array(routes, mlx::core::int32));
                return mlx::core::take(
                    source,
                    source_rows,
                    0);
            }
            return mlx::core::take(
                mlx::core::reshape(
                    source,
                    Shape{route_count, impl_->neuron_len}),
                route_order,
                0);
        }();
        sorted_source = mlx::core::expand_dims(
            mlx::core::contiguous(std::move(sorted_source)),
            1);
        auto sorted_expert_ids = mlx::core::take(
            mlx::core::reshape(ids, Shape{route_count}),
            route_order,
            0);
        auto sorted_slot_ids = mlx::core::take(
            *impl_->mxfp4_slot_ids,
            sorted_expert_ids,
            0);
        const auto value_stride = checked_product(
            static_cast<std::size_t>(impl_->out_per_expert),
            static_cast<std::size_t>(impl_->neuron_len / 2),
            "MXFP4 NAX value stride");
        const auto scale_stride = checked_product(
            static_cast<std::size_t>(impl_->out_per_expert),
            static_cast<std::size_t>(impl_->neuron_len / 32),
            "MXFP4 NAX scale stride");
        const auto slots = impl_->mx_values.size() / value_stride;
        if (
            slots == 0
            || impl_->mx_values.size() % value_stride != 0
            || impl_->mx_scales.size() != slots * scale_stride
        ) {
            throw std::runtime_error(
                "MXFP4 NAX arena geometry is invalid");
        }
        auto packed = mlx::core::reshape(
            mlx::core::view(impl_->mx_values, mlx::core::uint32),
            Shape{
                checked_int(slots, "MXFP4 NAX slot count"),
                impl_->out_per_expert,
                impl_->neuron_len / 8,
            });
        auto scales = mlx::core::reshape(
            impl_->mx_scales,
            Shape{
                checked_int(slots, "MXFP4 NAX slot count"),
                impl_->out_per_expert,
                impl_->neuron_len / 32,
            });
        // MLX 0.32.0's sorted NAX kernel has row-offset overflow above 32768
        // routes and assumes K is divisible by 64.  The unsorted dispatch is
        // still valid for the already sorted rows and avoids both defects.
        const bool safe_sorted_indices =
            impl_->mxfp4_slot_ids_sorted
            && impl_->neuron_len % 64 == 0
            && route_count <= 32768;
        auto projected = mlx::core::squeeze(
            mlx::core::gather_qmm(
                sorted_source,
                packed,
                scales,
                std::nullopt,
                std::nullopt,
                sorted_slot_ids,
                true,
                32,
                4,
                "mxfp4",
                safe_sorted_indices),
            1);
        return fused_swiglu
            ? moe_limited_swiglu_split(
                  std::move(projected),
                  swiglu_limit)
            : projected;
    }

    const array params({0.0f}, mlx::core::float32);
    std::vector<array> kernel_inputs{
        impl_->descriptors,
        impl_->nint_q,
        impl_->nint_sub_scale,
        impl_->nint_sub_min,
        impl_->nint_anchor_scale,
        impl_->nint_anchor_min,
        impl_->q8_q,
        impl_->q8_scales,
        impl_->vq_indices,
        impl_->vq_state,
        impl_->vq_aux,
        impl_->vq_anchors,
        impl_->vq_codebooks,
        impl_->vq_scales,
        impl_->vq_state_to_codebank,
        impl_->vq_banks,
        impl_->vq_parameters,
        impl_->vq_residual_codebooks,
        impl_->vq_residual_first,
        impl_->vq_residual_second,
        impl_->mx_values,
        impl_->mx_scales,
        source,
        ids,
        route_order,
        params,
    };
    const int output_width = fused_swiglu
        ? (impl_->projections == 2
            ? impl_->out_per_expert
            : impl_->out_per_expert / 2)
        : impl_->out_per_expert;
    const bool use_grouped_nax = !impl_->has_nepq_residual
        && mixed_grouped_nax_enabled(route_count);
    constexpr bool use_direct_nax = false;
    auto owned_plan = plan == nullptr
        ? std::optional<MlxGroupedMmqPlan>(
            make_grouped_mmq_plan(
                ids,
                route_order,
                *impl_->expert_order,
                impl_->experts,
                recommended_grouped_mmq_block_rows(
                    route_count,
                    fused_swiglu)))
        : std::nullopt;
    const auto& selected_plan = plan != nullptr
        ? *plan
        : *owned_plan;
    if (
        selected_plan.route_count != route_count
        || selected_plan.experts != impl_->experts
        || (selected_plan.block_rows != 32
            && selected_plan.block_rows != 48
            && selected_plan.block_rows != 64
            && selected_plan.block_rows != 80
            && selected_plan.block_rows != 96)
        || (selected_plan.block_rows != 32
            && (!use_grouped_nax || use_direct_nax))
        || selected_plan.max_blocks <= 0
    ) {
        throw std::invalid_argument(
            "grouped MMQ block plan does not match routed projection");
    }
    kernel_inputs.push_back(selected_plan.block_meta);
    kernel_inputs.push_back(selected_plan.block_count);
    return grouped_mmq_dispatch(
        std::move(kernel_inputs),
        GroupedMmqConfig{
            .output_shape = Shape{
                route_count,
                output_width,
            },
            .route_count = route_count,
            .max_blocks = selected_plan.max_blocks,
            .block_rows = selected_plan.block_rows,
            .tile_columns = use_grouped_nax && !use_direct_nax
                ? grouped_mmq_tile_columns(
                    selected_plan.block_rows,
                    output_width)
                : 64,
            .tokens = tokens,
            .routes = routes,
            .experts = impl_->experts,
            .output_width = output_width,
            .matrix_output_width = impl_->out_per_expert,
            .projections = impl_->projections,
            .input_width = impl_->neuron_len,
            .descriptor_size = kDescriptorSize,
            .variant_stride = variant_stride,
            .shared_input = static_cast<int>(shared_input),
            .input_sorted = static_cast<int>(input_is_sorted),
            .fused_swiglu = static_cast<int>(fused_swiglu),
            .has_nepq_residual = static_cast<int>(
                impl_->has_nepq_residual),
            .family_mask = static_cast<int>(impl_->family_mask),
            .vq_profile_mask = static_cast<int>(
                impl_->vq_profile_mask),
            .use_nax = use_grouped_nax,
            .direct_nax = use_direct_nax,
            .swiglu_limit = swiglu_limit,
        });
}

array MlxMfeWeight::routed_matmul_impl(
    const array& input,
    const array& expert_ids,
    bool fused_swiglu,
    float swiglu_limit,
    const array* expert_map,
    bool packed_expert_ids) const {
    if (impl_->projections > 1 && !impl_->projection_views.empty() &&
        std::any_of(
            impl_->projection_views.begin(),
            impl_->projection_views.end(),
            [](const auto& projection) {
                return !projection->mxfp4_sq_cohorts.empty()
                    || !projection->fp8_sq_cohorts.empty();
            })) {
        if (fused_swiglu) {
            throw std::logic_error(
                "split standalone-kernel projections must use routed_swiglu");
        }
        std::vector<array> outputs;
        outputs.reserve(impl_->projection_views.size());
        for (const auto& projection : impl_->projection_views) {
            outputs.push_back(MlxMfeWeight(projection).routed_matmul_impl(
                input,
                expert_ids,
                false,
                0.0f,
                expert_map,
                packed_expert_ids));
        }
        return mlx::core::concatenate(std::move(outputs), -1);
    }
    if (mlx_reference_enabled()) {
        if (expert_map != nullptr) {
            auto mapped_ids = mlx::core::take(
                mlx::core::contiguous(mlx::core::astype(
                    *expert_map,
                    mlx::core::int32)),
                expert_ids,
                0);
            return routed_bf16_reference(
                input,
                mapped_ids,
                fused_swiglu,
                swiglu_limit);
        }
        if (packed_expert_ids) {
            auto mapped_ids =
                mlx::core::floor_divide(
                    expert_ids,
                    array(256, mlx::core::int32)) -
                array(1, mlx::core::int32);
            return routed_bf16_reference(
                input,
                mapped_ids,
                fused_swiglu,
                swiglu_limit);
        }
        return routed_bf16_reference(
            input,
            expert_ids,
            fused_swiglu,
            swiglu_limit);
    }
    auto ids = mlx::core::contiguous(
        mlx::core::astype(
            expert_ids,
            mlx::core::int32));
    auto map = expert_map != nullptr
        ? mlx::core::contiguous(
              mlx::core::astype(*expert_map, mlx::core::int32))
        : ids;
    int expert_map_size = expert_map != nullptr
        ? checked_int(map.size(), "expert page-table size")
        : 0;
    if (expert_map != nullptr && map.ndim() != 1) {
        throw std::invalid_argument(
            "expert page table must be one-dimensional");
    }
    if (ids.ndim() != 2) {
        throw std::invalid_argument(
            "routed expert IDs must have "
            "[tokens,routes] shape");
    }
    const int tokens = ids.shape(0);
    const int routes = ids.shape(1);
    if (tokens < 0 || routes < 0) {
        throw std::invalid_argument(
            "routed expert dimensions cannot be negative");
    }
    bool shared_input = false;
    if (
        input.ndim() == 2
        && input.shape(0) == tokens
        && input.shape(1) == impl_->neuron_len
    ) {
        shared_input = true;
    } else if (
        input.ndim() != 3
        || input.shape(0) != tokens
        || input.shape(1) != routes
        || input.shape(2) != impl_->neuron_len
    ) {
        throw std::invalid_argument(
            "routed input must have [tokens,K] or "
            "[tokens,routes,K] shape");
    }

    auto source = input;
    if (
        source.dtype() != mlx::core::float16
        && source.dtype() != mlx::core::float32
    ) {
        source =
            mlx::core::astype(
                source,
                mlx::core::float16);
    }
    source = mlx::core::contiguous(source);
    const int logical_output_width = fused_swiglu
        ? (impl_->projections == 2
            ? impl_->out_per_expert
            : impl_->out_per_expert / 2)
        : impl_->out_per_expert;
    const auto output_width = fused_swiglu
        ? static_cast<std::size_t>(logical_output_width)
        : checked_product(
              static_cast<std::size_t>(impl_->projections),
              static_cast<std::size_t>(impl_->out_per_expert),
              "routed output width");
    const Shape output_shape{
        tokens,
        routes,
        checked_int(
            output_width,
            "routed output width"),
    };
    if (tokens == 0 || routes == 0) {
        return mlx::core::zeros(
            output_shape,
            source.dtype());
    }

    std::optional<array> standalone_output;
    if (!impl_->mxfp4_sq_cohorts.empty()
        || !impl_->fp8_sq_cohorts.empty()) {
        auto physical_ids = ids;
        if (packed_expert_ids) {
            physical_ids = mlx::core::floor_divide(
                ids,
                array(256, mlx::core::int32)) -
                array(1, mlx::core::int32);
        } else if (expert_map != nullptr) {
            if (expert_map_size <= 0) {
                physical_ids = mlx::core::full(
                    ids.shape(), -1, mlx::core::int32);
            } else {
                auto valid = mlx::core::logical_and(
                    mlx::core::greater_equal(
                        ids, array(0, mlx::core::int32)),
                    mlx::core::less(
                        ids,
                        array(expert_map_size, mlx::core::int32)));
                auto safe = mlx::core::minimum(
                    mlx::core::maximum(
                        ids, array(0, mlx::core::int32)),
                    array(expert_map_size - 1, mlx::core::int32));
                physical_ids = mlx::core::where(
                    valid,
                    mlx::core::take(map, safe, 0),
                    mlx::core::full(
                        ids.shape(), -1, mlx::core::int32));
            }
        }
        physical_ids = mlx::core::contiguous(
            mlx::core::astype(physical_ids, mlx::core::int32));
        const auto append_standalone = [&](array projected) {
            if (fused_swiglu) {
                projected = moe_limited_swiglu_split(
                    projected, swiglu_limit);
            }
            if (projected.dtype() != source.dtype()) {
                projected = mlx::core::astype(
                    projected, source.dtype());
            }
            standalone_output = standalone_output.has_value()
                ? *standalone_output + projected
                : std::move(projected);
        };
        for (const auto& cohort : impl_->mxfp4_sq_cohorts) {
            append_standalone(cohort.weight.routed_matmul(
                source,
                physical_ids,
                cohort.expert_map,
                impl_->out_per_expert));
        }
        for (const auto& cohort : impl_->fp8_sq_cohorts) {
            append_standalone(cohort.weight.routed_matmul(
                source,
                physical_ids,
                cohort.expert_map,
                impl_->out_per_expert));
        }
        if (!impl_->has_generic_cohorts) {
            return std::move(*standalone_output);
        }

        auto valid = mlx::core::logical_and(
            mlx::core::greater_equal(
                physical_ids, array(0, mlx::core::int32)),
            mlx::core::less(
                physical_ids,
                array(impl_->experts, mlx::core::int32)));
        auto safe = mlx::core::minimum(
            mlx::core::maximum(
                physical_ids, array(0, mlx::core::int32)),
            array(impl_->experts - 1, mlx::core::int32));
        auto owner = mlx::core::take(
            *impl_->standalone_owner, safe, 0);
        auto is_standalone = mlx::core::logical_and(
            valid,
            mlx::core::greater_equal(
                owner, array(0, mlx::core::int32)));
        ids = mlx::core::contiguous(mlx::core::where(
            is_standalone,
            mlx::core::full(ids.shape(), -1, mlx::core::int32),
            physical_ids));
        map = ids;
        expert_map = nullptr;
        expert_map_size = 0;
        packed_expert_ids = false;
    }

    const auto merge_standalone = [&](array output) {
        return standalone_output.has_value()
            ? output + *standalone_output
            : output;
    };

    const auto route_count_size = checked_product(
        static_cast<std::size_t>(tokens),
        static_cast<std::size_t>(routes),
        "route count");
    const int variant_stride = checked_int(
        route_count_size,
        "rotation variant stride");
    const bool sorted_routes = routed_sort_enabled(tokens);
    auto route_order = mlx::core::zeros(
        Shape{1},
        mlx::core::int32);
    if (sorted_routes) {
        route_order = mlx::core::contiguous(
            mlx::core::astype(
                mlx::core::argsort(
                    mlx::core::reshape(
                        ids,
                        Shape{variant_stride})),
                mlx::core::int32));
    }
    if (!impl_->rotations.empty()) {
        if (shared_input) {
            source = mlx::core::broadcast_to(
                mlx::core::reshape(
                    source,
                    Shape{
                        tokens,
                        1,
                        impl_->neuron_len,
                    }),
                Shape{
                    tokens,
                    routes,
                    impl_->neuron_len,
                });
        }
        source = mlx::core::contiguous(
            mlx::core::reshape(
                source,
                Shape{
                    variant_stride,
                    impl_->neuron_len,
                }));
        std::vector<array> variants;
        variants.reserve(
            impl_->rotations.size() + 1);
        variants.push_back(source);
        for (const auto& rotation : impl_->rotations) {
            variants.push_back(
                apply_rotation(
                    source,
                    rotation,
                    variant_stride,
                    impl_->neuron_len));
        }
        source = mlx::core::contiguous(
            mlx::core::concatenate(
                std::move(variants),
                0));
        shared_input = false;
    }

    const int k_lanes = impl_->k_lanes_override != 0
        ? impl_->k_lanes_override
        : (fused_swiglu
            && impl_->family_mask !=
                (std::uint32_t{1} << kFamilyMxfp4)
            ? 16
            : 8);
    const int rows_per_simd =
        tokens == 1
            && impl_->projections == 1
            && impl_->family_mask
                == (std::uint32_t{1} << kFamilyMxfp4)
            && impl_->rotations.empty()
            && k_lanes == 8
        ? mxfp4_decode_rows_per_simd()
        : 1;
    const auto rows_per_workgroup =
        static_cast<std::size_t>(64 / k_lanes * rows_per_simd);
    const auto output_tiles =
        (
            static_cast<std::size_t>(
                logical_output_width)
            + rows_per_workgroup - 1
        ) / rows_per_workgroup;
    auto workgroups = route_count_size;
    workgroups = checked_product(
        workgroups,
        static_cast<std::size_t>(
            fused_swiglu ? 1 : impl_->projections),
        "routed projection count");
    workgroups = checked_product(
        workgroups,
        output_tiles,
        "routed workgroup count");
    const array params(
        {swiglu_limit},
        mlx::core::float32);
    std::vector<array> kernel_inputs{
        impl_->descriptors,
        impl_->nint_q,
        impl_->nint_sub_scale,
        impl_->nint_sub_min,
        impl_->nint_anchor_scale,
        impl_->nint_anchor_min,
        impl_->q8_q,
        impl_->q8_scales,
        impl_->vq_indices,
        impl_->vq_state,
        impl_->vq_aux,
        impl_->vq_anchors,
        impl_->vq_codebooks,
        impl_->vq_scales,
        impl_->vq_state_to_codebank,
        impl_->vq_banks,
        impl_->vq_parameters,
        impl_->vq_residual_codebooks,
        impl_->vq_residual_first,
        impl_->vq_residual_second,
        impl_->mx_values,
        impl_->mx_scales,
        source,
        ids,
        route_order,
        params,
    };
    if (
        sorted_routes
        && tokens >= 32
        && source.dtype() == mlx::core::float16
        && (impl_->projection_views.empty()
            || (fused_swiglu
                && impl_->projections == 2
                && impl_->rotations.empty()))
        && expert_map == nullptr
        && !packed_expert_ids
        && supports_grouped_mmq()
    ) {
        const bool use_grouped_nax = !impl_->has_nepq_residual
            && mixed_grouped_nax_enabled(variant_stride);
        constexpr bool use_direct_nax = false;
        auto plan = make_grouped_mmq_plan(
            ids,
            route_order,
            *impl_->expert_order,
            impl_->experts,
            recommended_grouped_mmq_block_rows(
                variant_stride,
                fused_swiglu));
        kernel_inputs.push_back(plan.block_meta);
        kernel_inputs.push_back(plan.block_count);
        auto sorted_outputs = grouped_mmq_dispatch(
            std::move(kernel_inputs),
            GroupedMmqConfig{
                .output_shape = Shape{
                    variant_stride,
                    logical_output_width,
                },
                .route_count = variant_stride,
                .max_blocks = plan.max_blocks,
                .block_rows = plan.block_rows,
                .tile_columns = use_grouped_nax && !use_direct_nax
                    ? grouped_mmq_tile_columns(
                        plan.block_rows,
                        logical_output_width)
                    : 64,
                .tokens = tokens,
                .routes = routes,
                .experts = impl_->experts,
                .output_width = logical_output_width,
                .matrix_output_width = impl_->out_per_expert,
                .projections = impl_->projections,
                .input_width = impl_->neuron_len,
                .descriptor_size = kDescriptorSize,
                .variant_stride = variant_stride,
                .shared_input = static_cast<int>(shared_input),
                .fused_swiglu = static_cast<int>(fused_swiglu),
                .has_nepq_residual = static_cast<int>(
                    impl_->has_nepq_residual),
                .family_mask = static_cast<int>(impl_->family_mask),
                .vq_profile_mask = static_cast<int>(
                    impl_->vq_profile_mask),
                .use_nax = use_grouped_nax,
                .direct_nax = use_direct_nax,
                .swiglu_limit = swiglu_limit,
            });
        auto inverse_order = moe_inverse_permutation(route_order);
        auto restored = mlx::core::take(
            std::move(sorted_outputs),
            inverse_order,
            0);
        auto valid = mlx::core::expand_dims(
            mlx::core::greater_equal(
                ids,
                array(0, mlx::core::int32)),
            -1);
        return merge_standalone(mlx::core::where(
            valid,
            mlx::core::reshape(
                std::move(restored),
                output_shape),
            mlx::core::zeros(
                output_shape,
                source.dtype())));
    }
    if (impl_->native_primitive) {
        if (expert_map != nullptr) {
            kernel_inputs.push_back(map);
        }
        return merge_standalone(native_moe_dispatch(
            std::move(kernel_inputs),
            NativeMoeConfig{
                .dtype = source.dtype(),
                .output_shape = output_shape,
                .tokens = tokens,
                .routes = routes,
                .experts = impl_->experts,
                .output_width = logical_output_width,
                .matrix_output_width = impl_->out_per_expert,
                .projections = impl_->projections,
                .fused_swiglu = static_cast<int>(fused_swiglu),
                .input_width = impl_->neuron_len,
                .k_lanes = k_lanes,
                .rows_per_simd = rows_per_simd,
                .descriptor_size = kDescriptorSize,
                .variant_stride = variant_stride,
                .shared_input = static_cast<int>(shared_input),
                .jsc_execution_layout = static_cast<int>(
                    impl_->jsc_execution_layout),
                .family_mask = static_cast<int>(impl_->family_mask),
                .vq_profile_mask = static_cast<int>(
                    impl_->vq_profile_mask),
                .has_nepq_residual = static_cast<int>(
                    impl_->has_nepq_residual),
                .npq_grouped_indices = static_cast<int>(
                    impl_->npq_grouped_indices),
                .sorted_routes = static_cast<int>(
                    sorted_routes),
                .expert_map_size = expert_map_size,
                .packed_expert_ids = static_cast<int>(packed_expert_ids),
                .workgroups = static_cast<int>(workgroups),
            }));
    }
    const auto grid = checked_product(
        workgroups,
        64,
        "Metal grid");
    if (
        grid
        > static_cast<std::size_t>(
            std::numeric_limits<int>::max())
    ) {
        throw std::runtime_error(
            "MFE compatibility grid exceeds MLX limits");
    }
    kernel_inputs.push_back(map);
    auto outputs = moe_kernel()(
        std::move(kernel_inputs),
        {output_shape},
        {source.dtype()},
        {
            static_cast<int>(grid),
            1,
            1,
        },
        {64, 1, 1},
        {
            {"T", source.dtype()},
            {"TOKENS", tokens},
            {"ROUTES", routes},
            {"EXPERTS", impl_->experts},
            {"OUT", logical_output_width},
            {"MATRIX_OUT", impl_->out_per_expert},
            {"PROJECTIONS", impl_->projections},
            {
                "FUSED_SWIGLU",
                static_cast<int>(fused_swiglu),
            },
            {"K", impl_->neuron_len},
            {"K_LANES_VALUE", k_lanes},
            {"ROWS_PER_SIMD_VALUE", rows_per_simd},
            {"DESCRIPTOR_SIZE", kDescriptorSize},
            {"VARIANT_STRIDE", variant_stride},
            {
                "SHARED_INPUT",
                static_cast<int>(shared_input),
            },
            {
                "JSC_EXECUTION_LAYOUT",
                static_cast<int>(
                    impl_->jsc_execution_layout),
            },
            {
                "FAMILY_MASK",
                static_cast<int>(impl_->family_mask),
            },
            {
                "VQ_PROFILE_MASK",
                static_cast<int>(
                    impl_->vq_profile_mask),
            },
            {
                "HAS_NEPQ_RESIDUAL",
                static_cast<int>(
                    impl_->has_nepq_residual),
            },
            {
                "NPQ_GROUPED_INDICES",
                static_cast<int>(
                    impl_->npq_grouped_indices),
            },
            {
                "SORTED_ROUTES",
                static_cast<int>(sorted_routes),
            },
            {"EXPERT_MAP_SIZE", expert_map_size},
            {
                "PACKED_EXPERT_IDS",
                static_cast<int>(packed_expert_ids),
            },
        },
        std::nullopt,
        false,
        {});
    return merge_standalone(std::move(outputs.front()));
}

array MlxMfeWeight::routed_bf16_reference(
    const array& input,
    const array& expert_ids,
    bool fused_swiglu,
    float swiglu_limit) const {
    auto ids = mlx::core::contiguous(
        mlx::core::astype(
            expert_ids,
            mlx::core::int32));
    if (ids.ndim() != 2) {
        throw std::invalid_argument(
            "unpacked reference expert IDs must be [tokens,routes]");
    }
    const int tokens = ids.shape(0);
    const int routes = ids.shape(1);
    const bool shared_input =
        input.ndim() == 2
        && input.shape(0) == tokens
        && input.shape(1) == impl_->neuron_len;
    if (!shared_input && (
        input.ndim() != 3
        || input.shape(0) != tokens
        || input.shape(1) != routes
        || input.shape(2) != impl_->neuron_len)) {
        throw std::invalid_argument(
            "unpacked reference routed input shape mismatch");
    }
    if (impl_->projections != 1
        || impl_->reference_cohorts.empty()) {
        throw std::runtime_error(
            "unpacked reference MFE cohorts are unavailable");
    }

    ids.eval();
    std::vector<bool> selected(
        static_cast<std::size_t>(impl_->experts),
        false);
    const auto* id_values = ids.data<std::int32_t>();
    for (std::size_t index = 0; index < ids.size(); ++index) {
        const int expert = id_values[index];
        if (expert >= 0 && expert < impl_->experts) {
            selected[static_cast<std::size_t>(expert)] = true;
        }
    }

    const auto reference_dtype = mlx::core::float16;
    auto source = input.dtype() == reference_dtype
        ? input
        : mlx::core::astype(
              input,
              reference_dtype);
    auto output = mlx::core::zeros(
        Shape{
            tokens,
            routes,
            impl_->out_per_expert,
        },
        reference_dtype);
    for (int expert = 0; expert < impl_->experts; ++expert) {
        if (!selected[static_cast<std::size_t>(expert)]) {
            continue;
        }
        const ReferenceMoeCohort* cohort = nullptr;
        int local_expert = -1;
        for (const auto& candidate :
             impl_->reference_cohorts) {
            const auto found = std::find(
                candidate.expert_ids.begin(),
                candidate.expert_ids.end(),
                expert);
            if (found != candidate.expert_ids.end()) {
                cohort = &candidate;
                local_expert = static_cast<int>(
                    found - candidate.expert_ids.begin());
                break;
            }
        }
        if (cohort == nullptr) {
            throw std::runtime_error(
                "dense reference cannot locate selected expert");
        }
        const int row_begin =
            local_expert * impl_->out_per_expert;
        const auto rows = mlx::core::arange(
            row_begin,
            row_begin + impl_->out_per_expert,
            1,
            mlx::core::int32);
        auto dense = std::visit(
            [&](const auto& weight) {
                return mlx::core::astype(
                    weight.embedding(
                        rows,
                        mlx::core::float32),
                    reference_dtype);
            },
            cohort->weight);
        auto projected = mlx::core::matmul(
            source,
            mlx::core::transpose(dense));
        if (shared_input) {
            projected = mlx::core::expand_dims(
                projected,
                1);
        }
        auto mask = mlx::core::expand_dims(
            mlx::core::equal(
                ids,
                array(expert, mlx::core::int32)),
            -1);
        output = mlx::core::where(
            mask,
            projected,
            output);
        output.eval();
        mlx::core::clear_cache();
    }
    return fused_swiglu
        ? moe_limited_swiglu_split(
              output,
              swiglu_limit)
        : output;
}

int MlxMfeWeight::experts() const noexcept {
    return impl_->experts;
}

int MlxMfeWeight::out_per_expert() const noexcept {
    return impl_->out_per_expert;
}

int MlxMfeWeight::neuron_len() const noexcept {
    return impl_->neuron_len;
}

int MlxMfeWeight::projections() const noexcept {
    return impl_->projections;
}

std::size_t MlxMfeWeight::packed_nbytes() const noexcept {
    return impl_->packed_bytes;
}

MlxRoutedLinear::MlxRoutedLinear(
    MlxMoeWeight weight)
    : weight_(std::move(weight)) {
    if (weight_.projections() != 1) {
        throw std::invalid_argument(
            "routed linear requires one MFE projection");
    }
}

MlxRoutedLinear MlxRoutedLinear::from_blob(
    std::span<const std::uint8_t> blob) {
    return MlxRoutedLinear(
        MlxMoeWeight::from_blob(blob));
}

array MlxRoutedLinear::forward(
    const array& input,
    const array& expert_ids) const {
    return weight_.routed_matmul(input, expert_ids);
}

array MlxRoutedLinear::forward_mapped(
    const array& input,
    const array& expert_ids,
    const array& expert_map) const {
    return weight_.routed_matmul_mapped(
        input,
        expert_ids,
        expert_map);
}

array MlxRoutedLinear::forward_packed(
    const array& input,
    const array& packed_expert_ids) const {
    return weight_.routed_matmul_packed(
        input,
        packed_expert_ids);
}

array MlxRoutedLinear::swiglu(
    const array& input,
    const array& expert_ids,
    float limit) const {
    return weight_.routed_swiglu(
        input,
        expert_ids,
        limit);
}

array MlxRoutedLinear::swiglu_mapped(
    const array& input,
    const array& expert_ids,
    const array& expert_map,
    float limit) const {
    return weight_.routed_swiglu_mapped(
        input,
        expert_ids,
        expert_map,
        limit);
}

array MlxRoutedLinear::swiglu_packed(
    const array& input,
    const array& packed_expert_ids,
    float limit) const {
    return weight_.routed_swiglu_packed(
        input,
        packed_expert_ids,
        limit);
}

array MlxRoutedLinear::swiglu_pair(
    const MlxRoutedLinear& up,
    const array& input,
    const array& expert_ids,
    float limit) const {
    return weight_.routed_swiglu_pair(
        up.weight_,
        input,
        expert_ids,
        limit);
}

array MlxRoutedLinear::combine(
    const array& input,
    const array& expert_ids,
    const array& route_weights) const {
    return weight_.routed_matmul_reduce(
        input,
        expert_ids,
        route_weights);
}

array MlxRoutedLinear::combine_mapped(
    const array& input,
    const array& expert_ids,
    const array& expert_map,
    const array& route_weights) const {
    return moe_weighted_reduce(
        forward_mapped(input, expert_ids, expert_map),
        route_weights);
}

array MlxRoutedLinear::combine_packed(
    const array& input,
    const array& packed_expert_ids,
    const array& route_weights) const {
    return weight_.routed_matmul_reduce_packed(
        input,
        packed_expert_ids,
        route_weights);
}

MlxRoutedSwiGluFfn::MlxRoutedSwiGluFfn(
    MlxMoeWeight gate,
    MlxMoeWeight up,
    MlxMoeWeight down)
    : gate_up_(
          MlxMoeWeight::concatenate_projections(
              {std::move(gate), std::move(up)})),
      down_(std::move(down)) {
    if (
        gate_up_.experts() != down_.experts()
        || gate_up_.out_per_expert()
            != down_.neuron_len()
        || gate_up_.neuron_len()
            != down_.out_per_expert()
        || down_.projections() != 1
    ) {
        throw std::invalid_argument(
            "routed SwiGLU gate/up/down shapes "
            "are incompatible");
    }
}

MlxRoutedSwiGluFfn
MlxRoutedSwiGluFfn::from_blobs(
    std::span<const std::uint8_t> gate,
    std::span<const std::uint8_t> up,
    std::span<const std::uint8_t> down) {
    return MlxRoutedSwiGluFfn(
        MlxMoeWeight::from_blob(gate),
        MlxMoeWeight::from_blob(up),
        MlxMoeWeight::from_blob(down));
}

array MlxRoutedSwiGluFfn::forward(
    const array& input,
    const array& expert_ids,
    const array& route_weights) const {
    auto gate_up =
        gate_up_.routed_matmul(
            input,
            expert_ids);
    auto hidden =
        moe_swiglu_split(gate_up);
    auto pair_output =
        down_.routed_matmul(
            hidden,
            expert_ids);
    return moe_weighted_reduce(
        pair_output,
        route_weights);
}

MlxRoutedFfnResult
MlxRoutedSwiGluFfn::forward_from_logits(
    const array& input,
    const array& router_logits,
    int top_k,
    bool use_sigmoid,
    bool use_sqrt_softplus,
    bool normalize,
    bool delayed_softmax,
    const std::optional<array>& bias,
    const std::optional<array>& available,
    float norm_floor,
    float scale) const {
    auto routes = moe_topk(
        router_logits,
        top_k,
        use_sigmoid,
        use_sqrt_softplus,
        normalize,
        delayed_softmax,
        bias,
        available,
        norm_floor,
        scale);
    auto output = forward(
        input,
        routes.ids,
        routes.weights);
    return {
        std::move(output),
        std::move(routes.ids),
        std::move(routes.weights),
    };
}

} // namespace mfq::metal
