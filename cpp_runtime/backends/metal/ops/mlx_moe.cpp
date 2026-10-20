#include "mlx_moe.h"
#include "mlx_moe_ops.h"

#include <coroutine>
#include "mlx_kernel_prepare.h"
#include "mlx_builtin_kernels.h"
#include "mlx_weight_residency.h"
#include "mlx_nvq_decode.h"
#include "mlx_resource_telemetry.h"
#include "mlx_resident_budget.h"

#include "mfq_container.h"
#include "mfq/host_parallel.h"
#include "mfq_mfe_prefill_embedded.h"
#include "mlx_nint.h"
#include "mlx_nint_metadata.h"
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
#include <atomic>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <dispatch/dispatch.h>
#include <exception>
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
#include <sys/mman.h>
#include <unistd.h>

namespace mfq::metal {
namespace {

using mlx::core::CompileOptions;
using mlx::core::Dtype;
using mlx::core::MathMode;
using mlx::core::Shape;
using mlx::core::array;

constexpr int kDescriptorSize = 32;
constexpr unsigned kMfeSmallMReadWorkers = 16;
constexpr std::size_t kMfeDirectRangeMaxExperts = 2;
std::atomic<std::size_t> uncached_packed_storage_bytes{0};
constexpr int kMfeFoldedGateSimdPairs = 4;

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
constexpr int kNintExecution = 10;
constexpr int kNintRowMetadataOffset = 11;
constexpr int kNintQDispatch = 12;
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
    return !mlx_metal_nax_disabled() && mlx_apple_chip_starts_with("Apple M5");
}

bool apple_m3_ultra() noexcept {
    return mlx_apple_chip_is("Apple M3 Ultra");
}

bool mixed_grouped_nax_enabled(int route_count) noexcept {
    if (mlx_metal_nax_disabled()) return false;
    // The heterogeneous NAX path is a large-M prefill kernel.  Keep small-M
    // and short-tail shapes on the compatibility kernel, whose boundary
    // handling is both cheaper and already exhaustive.
    constexpr int kMinRoutes = 512;
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
    int output_width,
    bool fused_swiglu) noexcept {
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
    if (block_rows == 128 && !fused_swiglu) {
        return 32;
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

int mfe_decode_rows_per_simd(int automatic_rows) noexcept {
    const char* value = std::getenv(
        "MFQ_METAL_MFE_DECODE_ROWS_PER_SIMD");
    if (value != nullptr) {
        const auto setting = std::string_view(value);
        if (setting == "1") return 1;
        if (setting == "2") return 2;
        if (setting == "4") return 4;
    }
    return automatic_rows;
}

bool mfe_narrow_bindings_enabled() noexcept {
    const char* value = std::getenv(
        "MFQ_METAL_MFE_NARROW_BINDINGS");
    if (value == nullptr) {
        return true;
    }
    const auto setting = std::string_view(value);
    return setting != "0"
        && setting != "false"
        && setting != "off";
}

bool mfe_split_swiglu_enabled() noexcept {
    const char* value = std::getenv(
        "MFQ_METAL_MFE_SPLIT_SWIGLU");
    if (value == nullptr) {
        return true;
    }
    const auto setting = std::string_view(value);
    return setting != "0"
        && setting != "false"
        && setting != "off";
}

bool routed_sort_enabled(int tokens) noexcept {
    if (tokens <= 1) {
        return false;
    }
    const char* value = std::getenv(
        "MFQ_METAL_MFE_SORT_ROUTES");
    if (value == nullptr) {
        return tokens > 6;
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
constexpr int kVqExecutionLayout = 29;
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
constexpr int kVqProfileJscExtended4 = 8;
constexpr std::uint32_t kGroupedVqVectorProfileMask =
    (std::uint32_t{1} << kVqProfileGeneric)
    | (std::uint32_t{1} << kVqProfileNpqS)
    | (std::uint32_t{1} << kVqProfileNvq1L)
    | (std::uint32_t{1} << kVqProfileNpqL)
    | (std::uint32_t{1} << kVqProfileNvq1S)
    | (std::uint32_t{1} << kVqProfileJscExtended4);
constexpr std::uint32_t kGroupedJscExtendedProfileMask =
    (std::uint32_t{1} << kVqProfileJscExtended8)
    | (std::uint32_t{1} << kVqProfileJscExtended4);

const std::string kMoeHeader = std::string(R"METAL(
#ifndef MFQ_MOE_ROUTE_GROUPS
#define MFQ_MOE_ROUTE_GROUPS 0
#endif
#ifndef MFQ_MOE_VQ_LAYOUTS
#define MFQ_MOE_VQ_LAYOUTS 0xfffffffffffffffful
#endif
template <uint K, typename T>
struct MfqMoeVqInputs {
    device const T* values;
    thread const uint* offsets;
    T operator[](uint index) const {
        return values[offsets[index / K] + index % K];
    }
};

template <uint ROUTED_WIDTH, uint TOKEN_STRIDE, typename T>
struct MfqMoeVqOutput {
    device T* values;
    device T& operator[](uint index) const {
        return values[(index / ROUTED_WIDTH) * TOKEN_STRIDE + index % ROUTED_WIDTH];
    }
};

template <uint K, typename T>
inline device const T* mfq_nvq_input_row(MfqMoeVqInputs<K, T> x, uint row) {
    return x.values + x.offsets[row];
}
)METAL") + detail::kNintMetadataSource + kNvqDecodeHeader + R"METAL(
#ifndef MFQ_NINT_Q_DISPATCH
#define MFQ_NINT_Q_DISPATCH 0
#endif

#define MFQ_MFE_NINT_PROFILE(GS, EXPANSION, TAIL, Q) \
    mfq_moe_nint_profile<GS, MATRIX_ROWS, uint(K), K_LANES, EXPANSION, TAIL, 1u, Q>( \
        x, nint_q, nint_sub_scale, nint_sub_min, x_offset, q_offset, sub_offset, \
        metadata_rows, q_widths, q_row_byte_offsets, q_row_bit_shifts, neuron_scales, \
        neuron_minimums, accumulators, groups, k_lane)
#define MFQ_MFE_NINT_CALL_Q(GS, Q) \
    if constexpr (OUT > K) { MFQ_MFE_NINT_PROFILE(GS, true, true, Q); } \
    else { \
        MFQ_MFE_NINT_PROFILE(GS, false, false, Q); \
        if constexpr (uint(K) % GS != 0u) { MFQ_MFE_NINT_PROFILE(GS, false, true, Q); } \
    }
#define MFQ_MFE_NINT_CALL_WITH(GS, ENABLED) \
    if constexpr (ENABLED && MATRIX_ROWS == 1u) { \
        switch (metadata_rows[0].q_bits) { \
            case 4u: { MFQ_MFE_NINT_CALL_Q(GS, 4u); break; } \
            case 5u: { MFQ_MFE_NINT_CALL_Q(GS, 5u); break; } \
            case 6u: { MFQ_MFE_NINT_CALL_Q(GS, 6u); break; } \
            case 8u: { MFQ_MFE_NINT_CALL_Q(GS, 8u); break; } \
            default: { MFQ_MFE_NINT_CALL_Q(GS, 0u); break; } \
        } \
    } else { MFQ_MFE_NINT_CALL_Q(GS, 0u); }
#define MFQ_MFE_NINT_CALL(GS) MFQ_MFE_NINT_CALL_WITH(GS, MFQ_NINT_Q_DISPATCH)
#ifndef MFQ_MFE_NINT_DISPATCH_WITH
#define MFQ_MFE_NINT_DISPATCH_WITH(CALL) \
    if (group_size == 24u) { CALL(24u); } \
    else if (group_size == 28u) { CALL(28u); } else
#endif
#define MFQ_MFE_NINT_DISPATCH MFQ_MFE_NINT_DISPATCH_WITH(MFQ_MFE_NINT_CALL)

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

template <bool ALIGNED_GROUP = true, bool WORD_QUAD = false>
inline ushort4 mfq_moe_decode_nint_quad_at(
    device const uchar* stream,
    uint byte_index,
    uint shift,
    uint bits
) {
    uint required_bits = shift + 4u * bits;
    packed_uchar4 bytes =
        *reinterpret_cast<device const packed_uchar4*>(stream + byte_index);
    uint packed = as_type<uint>(bytes);
    if constexpr (WORD_QUAD) {
        if constexpr (ALIGNED_GROUP) {
            if (shift != 0u) {
                packed >>= shift;
            }
        } else {
            packed >>= shift;
        }
    } else {
        if (shift != 0u) {
            packed = (packed >> shift)
                | (required_bits > 32u
                    ? uint(stream[byte_index + 4u]) << (32u - shift)
                    : 0u);
        }
    }
    uint mask = (1u << bits) - 1u;
    return ushort4(
        packed & mask,
        (packed >> bits) & mask,
        (packed >> (2u * bits)) & mask,
        (packed >> (3u * bits)) & mask);
}

inline ushort4 mfq_moe_read_nint_row_quad(
    device const uchar* stream,
    uint row_byte_offset,
    uint row_bit_shift,
    uint value_index,
    uint bits
) {
    uint row_relative_bits = row_bit_shift + value_index * bits;
    return mfq_moe_decode_nint_quad_at(
        stream,
        row_byte_offset + (row_relative_bits >> 3u),
        row_relative_bits & 7u,
        bits);
}

struct MfqMoeNintOctet {
    ushort4 low;
    ushort4 high;
};

template <bool ALIGNED_EIGHT>
inline MfqMoeNintOctet mfq_moe_decode_nint_octet_at(
    device const uchar* stream, uint byte_index, uint shift, uint bits
) {
    if constexpr (ALIGNED_EIGHT) shift = 0u;
    const uint word0 = as_type<uint>(
        *reinterpret_cast<device const packed_uchar4*>(stream + byte_index));
    const uint word1 = bits > 4u ? as_type<uint>(
        *reinterpret_cast<device const packed_uchar4*>(stream + byte_index + 4u)) : 0u;
    const uint first = shift == 0u ? word0 : (word0 >> 4u) | (word1 << 28u);
    const uint bit = 4u * bits + shift;
    const uint second = bit == 32u ? word1
        : (word0 >> bit) | (word1 << (32u - bit));
    const uint mask = (1u << bits) - 1u;
    return {
        ushort4(first & mask, (first >> bits) & mask,
            (first >> (2u * bits)) & mask, (first >> (3u * bits)) & mask),
        ushort4(second & mask, (second >> bits) & mask,
            (second >> (2u * bits)) & mask, (second >> (3u * bits)) & mask)
    };
}

template <
    uint GROUP_SIZE,
    uint MATRIX_ROWS,
    uint K,
    uint K_LANES,
    bool EXPANSION,
    bool TAIL,
    uint INPUT_ROWS = 1u,
    uint FIXED_Q = 0u,
    typename XStream,
    typename SubScaleStream,
    typename SubMinStream
>
inline void mfq_moe_nint_profile(
    XStream x,
    device const uchar* q_stream,
    SubScaleStream sub_scale_stream,
    SubMinStream sub_min_stream,
    uint x_offset,
    uint q_offset,
    uint sub_offset,
    thread const MfqNintRow* metadata_rows,
    thread const uint* q_widths,
    thread const uint* q_row_byte_offsets,
    thread const uint* q_row_bit_shifts,
    thread const float* neuron_scales,
    thread const float* neuron_minimums,
    thread float* accumulators,
    uint groups,
    uint k_lane,
    thread const uint* input_offsets = nullptr
) {
    constexpr uint OCTETS = (!EXPANSION || GROUP_SIZE == 24u || GROUP_SIZE == 28u)
        && GROUP_SIZE % 4u == 0u ? GROUP_SIZE / 8u : 0u;
    const uint first_group = !EXPANSION && TAIL
        ? K / GROUP_SIZE + (k_lane + K_LANES - (K / GROUP_SIZE) % K_LANES) % K_LANES
        : k_lane;
    constexpr uint END_GROUP = TAIL ? (K + GROUP_SIZE - 1u) / GROUP_SIZE : K / GROUP_SIZE;
    static_assert(K_LANES % 8u == 0u);
    uint4 sub_cursors[MATRIX_ROWS];
    uint3 q_group_cursors[MATRIX_ROWS];
    for (uint row = 0u; row < MATRIX_ROWS; ++row) {
        const MfqNintRow metadata = metadata_rows[row];
        const uint bit = metadata.sub_shift + (first_group & 7u) * metadata.sub_bits;
        sub_cursors[row] = uint4(
            sub_offset + metadata.sub_offset
                + (first_group >> 3u) * metadata.sub_bits + (bit >> 3u),
            bit & 7u,
            (1u << metadata.sub_bits) - 1u,
            (K_LANES / 8u) * metadata.sub_bits);
        const uint column = first_group * GROUP_SIZE;
        const uint bits = FIXED_Q != 0u ? FIXED_Q : q_widths[row];
        const uint q_bit = q_row_bit_shifts[row] + (column & 7u) * bits;
        q_group_cursors[row] = uint3(
            q_row_byte_offsets[row] + (column >> 3u) * bits + (q_bit >> 3u),
            q_bit & 7u,
            GROUP_SIZE * (K_LANES / 8u) * bits);
    }
    for (
        uint group = first_group;
        group < END_GROUP;
        group += K_LANES
    ) {
        float scales[MATRIX_ROWS];
        float minimums[MATRIX_ROWS];
        uint q_byte_cursors[MATRIX_ROWS];
        uint q_bit_cursors[MATRIX_ROWS];
        uint column_base = group * GROUP_SIZE;
        for (uint row = 0u; row < MATRIX_ROWS; ++row) {
            const uint4 sub = sub_cursors[row];
            const uint scale = mfq_nint_sub_word(sub_scale_stream, sub.x);
            const uint minimum = mfq_nint_sub_word(sub_min_stream, sub.x);
            scales[row] = neuron_scales[row]
                * float((scale >> sub.y) & sub.z);
            minimums[row] = neuron_minimums[row]
                * float((minimum >> sub.y) & sub.z);
            sub_cursors[row].x += sub.w;
            q_byte_cursors[row] = q_group_cursors[row].x;
            q_bit_cursors[row] = q_group_cursors[row].y;
            q_group_cursors[row].x += q_group_cursors[row].z;
        }
        float activation_sum[INPUT_ROWS] = {0.0f};
        float quantized_dots[MATRIX_ROWS * INPUT_ROWS] = {0.0f};
        for (uint block = 0u; block < OCTETS; ++block) {
            const uint column = column_base + block * 8u;
            if (TAIL && column >= K) break;
            float4 activation0[INPUT_ROWS];
            float4 activation1[INPUT_ROWS];
            for (uint input_row = 0u; input_row < INPUT_ROWS; ++input_row) {
            const uint offset = INPUT_ROWS == 1u ? x_offset : input_offsets[input_row];
            activation0[input_row] = float4(0.0f);
            activation1[input_row] = float4(0.0f);
            if (!TAIL || column + 7u < K) {
                activation0[input_row] = float4(x[offset + column], x[offset + column + 1u],
                    x[offset + column + 2u], x[offset + column + 3u]);
                activation1[input_row] = float4(x[offset + column + 4u], x[offset + column + 5u],
                    x[offset + column + 6u], x[offset + column + 7u]);
            } else {
                for (uint item = 0u; item < 4u; ++item) {
                    if (column + item < K) activation0[input_row][item] = float(x[offset + column + item]);
                    if (column + 4u + item < K) activation1[input_row][item] = float(x[offset + column + 4u + item]);
                }
            }
            activation_sum[input_row] += activation0[input_row].x + activation0[input_row].y
                + activation0[input_row].z + activation0[input_row].w
                + activation1[input_row].x + activation1[input_row].y
                + activation1[input_row].z + activation1[input_row].w;
            }
            for (uint row = 0u; row < MATRIX_ROWS; ++row) {
                const auto quantized = mfq_moe_decode_nint_octet_at<GROUP_SIZE % 8u == 0u>(
                    q_stream + q_offset, q_byte_cursors[row], q_bit_cursors[row], FIXED_Q != 0u ? FIXED_Q : q_widths[row]);
                for (uint input_row = 0u; input_row < INPUT_ROWS; ++input_row)
                    quantized_dots[row * INPUT_ROWS + input_row] +=
                        dot(activation0[input_row], float4(quantized.low))
                        + dot(activation1[input_row], float4(quantized.high));
                q_byte_cursors[row] += FIXED_Q != 0u ? FIXED_Q : q_widths[row];
            }
        }
        for (
            uint element = OCTETS * 8u;
            element < GROUP_SIZE;
            element += 4u
        ) {
            uint column = column_base + element;
            if (TAIL && column >= K) {
                break;
            }
            float4 activations[INPUT_ROWS];
            for (uint input_row = 0u; input_row < INPUT_ROWS; ++input_row) {
            const uint offset = INPUT_ROWS == 1u ? x_offset : input_offsets[input_row];
            if (element + 3u < GROUP_SIZE && (!TAIL || column + 3u < K)) {
                activations[input_row] = float4(
                    x[offset + column],
                    x[offset + column + 1u],
                    x[offset + column + 2u],
                    x[offset + column + 3u]);
            } else {
                activations[input_row] = float4(
                    !TAIL || column < K
                        ? float(x[offset + column]) : 0.0f,
                    element + 1u < GROUP_SIZE && (!TAIL || column + 1u < K)
                        ? float(x[offset + column + 1u]) : 0.0f,
                    element + 2u < GROUP_SIZE && (!TAIL || column + 2u < K)
                        ? float(x[offset + column + 2u]) : 0.0f,
                    element + 3u < GROUP_SIZE && column + 3u < K
                        ? float(x[offset + column + 3u]) : 0.0f);
            }
            activation_sum[input_row] += activations[input_row].x + activations[input_row].y
                + activations[input_row].z + activations[input_row].w;
            }
            for (uint row = 0u; row < MATRIX_ROWS; ++row) {
                ushort4 quantized = mfq_moe_decode_nint_quad_at<
                    GROUP_SIZE % 8u == 0u, GROUP_SIZE % 4u == 0u>(
                    q_stream + q_offset,
                    q_byte_cursors[row],
                    q_bit_cursors[row],
                    FIXED_Q != 0u ? FIXED_Q : q_widths[row]);
                for (uint input_row = 0u; input_row < INPUT_ROWS; ++input_row)
                    quantized_dots[row * INPUT_ROWS + input_row] +=
                        dot(activations[input_row], float4(quantized));
                uint next_bit = q_bit_cursors[row]
                    + 4u * (FIXED_Q != 0u ? FIXED_Q : q_widths[row]);
                q_byte_cursors[row] += next_bit >> 3u;
                q_bit_cursors[row] = next_bit & 7u;
            }
        }
        for (uint row = 0u; row < MATRIX_ROWS; ++row) {
            for (uint input_row = 0u; input_row < INPUT_ROWS; ++input_row) {
            const uint index = row * INPUT_ROWS + input_row;
            accumulators[index] = fma(
                scales[row],
                quantized_dots[index],
                fma(
                    -minimums[row],
                    activation_sum[input_row],
                    accumulators[index]));
            }
        }
    }
}

#define MFQ_MFE_NINT_PAIR_PROFILE(GS, EXPANSION, TAIL, Q) \
    mfq_moe_nint_profile<GS, MATRIX_ROWS, K, K_LANES, EXPANSION, TAIL, 2u, Q>( \
        x, q_stream, sub_scale_stream, sub_min_stream, 0u, uint(d[7]), uint(d[8]), \
        metadata_rows, q_widths, q_row_byte_offsets, q_row_bit_shifts, neuron_scales, \
        neuron_minimums, values, uint(d[6]), k_lane, input_offsets)
#define MFQ_MFE_NINT_PAIR_CALL_Q(GS, Q) \
    if constexpr (OUT > K) { MFQ_MFE_NINT_PAIR_PROFILE(GS, true, true, Q); } \
    else { \
        MFQ_MFE_NINT_PAIR_PROFILE(GS, false, false, Q); \
        if constexpr (K % GS != 0u) { MFQ_MFE_NINT_PAIR_PROFILE(GS, false, true, Q); } \
    }
#define MFQ_MFE_NINT_PAIR_CALL(GS) \
    if (!leader) return true; \
    if constexpr (MFQ_NINT_Q_DISPATCH && MATRIX_ROWS == 1u) { \
        switch (metadata_rows[0].q_bits) { \
            case 4u: { MFQ_MFE_NINT_PAIR_CALL_Q(GS, 4u); break; } \
            case 5u: { MFQ_MFE_NINT_PAIR_CALL_Q(GS, 5u); break; } \
            case 6u: { MFQ_MFE_NINT_PAIR_CALL_Q(GS, 6u); break; } \
            case 8u: { MFQ_MFE_NINT_PAIR_CALL_Q(GS, 8u); break; } \
            default: { MFQ_MFE_NINT_PAIR_CALL_Q(GS, 0u); break; } \
        } \
    } else { MFQ_MFE_NINT_PAIR_CALL_Q(GS, 0u); }
#define MFQ_MFE_NINT_SUPPORTED(GS) return true
inline bool mfq_moe_nint_supported_gs(uint group_size) {
    MFQ_MFE_NINT_DISPATCH_WITH(MFQ_MFE_NINT_SUPPORTED) { return false; }
}

template <uint MATRIX_ROWS, uint ROWS_PER_SIMD, uint K, uint K_LANES,
    uint OUT, uint MATRIX_OUT, uint PROJECTIONS, uint ROUTES,
    bool SHARED_INPUT, bool SPLIT_SWIGLU, bool PACKED_SWIGLU,
    typename RouteStream, typename YStream, typename T>
inline bool mfq_moe_nint_pair(
    device const T* x, YStream y, device const int* d,
    device const uchar* q_stream, device const uchar* sub_scale_stream,
    device const uchar* sub_min_stream, RouteStream route_rows,
    threadgroup T* split_values, uint output_base, uint lane_group,
    uint k_lane, uint projection, uint variant_stride, float swiglu_limit,
    bool leader
) {
    uint input_offsets[2];
    for (uint row = 0u; row < 2u; ++row)
        input_offsets[row] = (uint(d[27]) * variant_stride
            + (SHARED_INPUT ? uint(route_rows[row]) / ROUTES : uint(route_rows[row]))) * K;
    device const uint* row_metadata = reinterpret_cast<device const uint*>(q_stream + uint(d[11]));
    MfqNintRow metadata_rows[MATRIX_ROWS];
    uint q_widths[MATRIX_ROWS];
    uint q_row_byte_offsets[MATRIX_ROWS];
    uint q_row_bit_shifts[MATRIX_ROWS];
    float neuron_scales[MATRIX_ROWS];
    float neuron_minimums[MATRIX_ROWS];
    for (uint row = 0u; row < MATRIX_ROWS; ++row) {
        const uint output = min(output_base + (PACKED_SWIGLU
            ? (row / ROWS_PER_SIMD) * OUT + row % ROWS_PER_SIMD : row), MATRIX_OUT - 1u);
        const MfqNintRow metadata = mfq_nint_row(row_metadata, uint(d[1]) * MATRIX_OUT + output);
        metadata_rows[row] = metadata;
        q_widths[row] = metadata.q_bits;
        q_row_byte_offsets[row] = metadata.q_offset;
        q_row_bit_shifts[row] = metadata.q_shift;
        neuron_scales[row] = metadata.scale;
        neuron_minimums[row] = metadata.minimum;
    }
    float values[MATRIX_ROWS * 2u] = {0.0f};
    const uint group_size = uint(d[5]);
    MFQ_MFE_NINT_DISPATCH_WITH(MFQ_MFE_NINT_PAIR_CALL) { return false; }
    constexpr uint PHYSICAL_ROWS = (32u / K_LANES) * ROWS_PER_SIMD;
    for (uint row = 0u; row < MATRIX_ROWS; ++row) {
        for (uint input_row = 0u; input_row < 2u; ++input_row) {
            const uint index = row * 2u + input_row;
            for (uint offset = K_LANES >> 1u; offset > 0u; offset >>= 1u)
                values[index] += simd_shuffle_down(values[index], offset);
            if constexpr (SPLIT_SWIGLU) {
                if (k_lane == 0u)
                    split_values[(input_row * 2u + projection) * PHYSICAL_ROWS
                        + lane_group * ROWS_PER_SIMD + row] = T(values[index]);
            }
        }
    }
    if constexpr (SPLIT_SWIGLU) threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {
        const uint output = output_base + row;
        for (uint input_row = 0u; input_row < 2u; ++input_row) {
            if (k_lane != 0u || output >= OUT || (SPLIT_SWIGLU && projection != 0u)) continue;
            float value = values[row * 2u + input_row];
            if constexpr (SPLIT_SWIGLU || PACKED_SWIGLU) {
                const uint local = lane_group * ROWS_PER_SIMD + row;
                float gate = SPLIT_SWIGLU
                    ? float(split_values[input_row * 2u * PHYSICAL_ROWS + local]) : float(T(value));
                float up = SPLIT_SWIGLU
                    ? float(split_values[(input_row * 2u + 1u) * PHYSICAL_ROWS + local])
                    : float(T(values[(row + ROWS_PER_SIMD) * 2u + input_row]));
                if (swiglu_limit > 0.0f) {
                    gate = min(gate, swiglu_limit);
                    up = clamp(up, -swiglu_limit, swiglu_limit);
                }
                value = gate / (1.0f + exp(-gate)) * up;
            }
            const uint destination = SPLIT_SWIGLU || PACKED_SWIGLU
                ? uint(route_rows[input_row]) * OUT + output
                : (uint(route_rows[input_row]) * PROJECTIONS + projection) * OUT + output;
            y[destination] = T(value);
        }
    }
    return true;
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

inline bool mfq_moe_reusable_vq_profile(uint profile) {
    return profile == 1u || profile == 3u || profile == 4u
        || profile == 6u || profile == 7u || profile == 8u;
}

template <uint PROFILE, uint EXCLUDED = 0u>
inline uint mfq_moe_vq_layout(device const int* descriptor) {
    constexpr uint layouts = uint((MFQ_MOE_VQ_LAYOUTS >> (PROFILE * 7u)) & 127u) & ~EXCLUDED;
    if constexpr (layouts != 0u && (layouts & (layouts - 1u)) == 0u) return ctz(layouts);
    else return uint(descriptor[29]);
}

template <uint INPUT_ROWS, uint MATRIX_ROWS, uint ROWS_PER_SIMD,
    uint K, uint K_LANES, uint OUT, uint MATRIX_OUT, uint PROJECTIONS,
    uint ROUTES, uint PROFILE_MASK, bool SHARED_INPUT, bool SPLIT_SWIGLU, bool PACKED_SWIGLU, bool MXFP4,
    typename RouteStream, typename YStream, typename T>
inline void mfq_moe_reused_rows(
    device const T* x, YStream y, device const int* d,
    device const uchar* indices, device const uchar* states,
    device const uchar* aux, device const float* anchors,
    device const int8_t* codebooks, device const float* scales,
    device const uchar* state_banks, device const float* parameters,
    device const uchar* mx_values, device const uchar* mx_scales,
    RouteStream route_rows, threadgroup T* split_values,
    uint output_base, uint lane_group, uint k_lane, uint projection,
    uint variant_stride, float swiglu_limit
) {
    uint input_offsets[INPUT_ROWS];
    for (uint row = 0u; row < INPUT_ROWS; ++row) {
        uint route = route_rows[row];
        input_offsets[row] = (uint(d[27]) * variant_stride
            + (SHARED_INPUT ? route / ROUTES : route)) * K;
    }
    MfqMoeVqInputs<K, T> inputs{x, input_offsets};
    uint outputs[MATRIX_ROWS];
    float row_anchors[MATRIX_ROWS];
    for (uint row = 0u; row < MATRIX_ROWS; ++row) {
        uint output = min(output_base + (PACKED_SWIGLU
            ? (row / ROWS_PER_SIMD) * OUT + row % ROWS_PER_SIMD : row), MATRIX_OUT - 1u);
        outputs[row] = uint(d[1]) * MATRIX_OUT + output;
        row_anchors[row] = MXFP4 ? 0.0f : anchors[uint(d[21]) + outputs[row]];
    }
    float values[MATRIX_ROWS * INPUT_ROWS] = {0.0f};
    uint profile = uint(d[28]) & 255u;
    if constexpr (MXFP4) {
        const uint groups = uint(d[4]);
        for (uint group = k_lane; group < groups; group += K_LANES) {
            for (uint row = 0u; row < MATRIX_ROWS; ++row) {
                const float scale = mfq_moe_e8m0(mx_scales[ulong(uint(d[6]))
                    + ulong(outputs[row]) * groups + group]);
                for (uint component = 0u; component < 32u; component += 16u) {
                    const uint column = group * 32u + component;
                    const ulong offset = ulong(uint(d[5])) + ulong(outputs[row]) * (K / 2u) + column / 2u;
                    const uint2 packed = *reinterpret_cast<device const uint2*>(mx_values + offset);
                    for (uint pair = 0u; pair < 8u; ++pair) {
                        const uint word = pair < 4u ? packed.x : packed.y;
                        const uchar code = uchar(word >> ((pair & 3u) * 8u));
                        const float first = scale * mfq_moe_mxfp4_value(code & 15u);
                        const float second = scale * mfq_moe_mxfp4_value(code >> 4u);
                        for (uint input_row = 0u; input_row < INPUT_ROWS; ++input_row) {
                            const uint xbase = input_offsets[input_row] + column + pair * 2u;
                            const uint index = row * INPUT_ROWS + input_row;
                            values[index] = fma(float(x[xbase]), first, values[index]);
                            values[index] = fma(float(x[xbase + 1u]), second, values[index]);
                        }
                    }
                }
            }
        }
    } else if ((PROFILE_MASK & 8u) != 0u && profile == 3u) {
        mfq_nvq1_profile<MATRIX_ROWS, K, K_LANES, 11u, INPUT_ROWS>(
            inputs, indices, states, aux, scales, codebooks, outputs, row_anchors, values,
            parameters[uint(d[26])], 0u, uint(d[18]), uint(d[19]), uint(d[20]), uint(d[23]), uint(d[22]), k_lane);
    } else if ((PROFILE_MASK & 64u) != 0u && profile == 6u) {
        mfq_nvq1_profile<MATRIX_ROWS, K, K_LANES, 9u, INPUT_ROWS>(
            inputs, indices, states, aux, scales, codebooks, outputs, row_anchors, values,
            parameters[uint(d[26])], 0u, uint(d[18]), uint(d[19]), uint(d[20]), uint(d[23]), uint(d[22]), k_lane);
    } else if ((PROFILE_MASK & 128u) != 0u && profile == 7u && mfq_moe_vq_layout<7u>(d) == 0u) {
        mfq_nvq_banked_profile<MATRIX_ROWS, K, K_LANES, (OUT > K), 8u, INPUT_ROWS>(
            inputs, indices, states, aux, state_banks, scales, codebooks, outputs, row_anchors, values,
            0u, uint(d[18]), uint(d[19]), uint(d[20]), uint(d[24]), uint(d[23]), uint(d[22]), k_lane, uint(d[8]));
    } else if ((PROFILE_MASK & 256u) != 0u && profile == 8u) {
        mfq_nvq_banked_profile<MATRIX_ROWS, K, K_LANES, (OUT > K), 4u, INPUT_ROWS>(
            inputs, indices, states, aux, state_banks, scales, codebooks, outputs, row_anchors, values,
            0u, uint(d[18]), uint(d[19]), uint(d[20]), uint(d[24]), uint(d[23]), uint(d[22]), k_lane, uint(d[8]));
    } else if ((PROFILE_MASK & 2u) != 0u && profile == 1u) {
        mfq_nvq_banked_profile<MATRIX_ROWS, K, K_LANES, (OUT > K), 4u, INPUT_ROWS, 8u>(
            inputs, indices, states, aux, state_banks, scales, codebooks, outputs, row_anchors, values,
            0u, uint(d[18]), uint(d[19]), uint(d[20]), uint(d[24]), uint(d[23]), uint(d[22]),
            k_lane, 8u, uint(d[13]));
    } else if ((PROFILE_MASK & 16u) != 0u && profile == 4u) {
        mfq_nvq_banked_profile<MATRIX_ROWS, K, K_LANES, (OUT > K), 8u, INPUT_ROWS, 8u>(
            inputs, indices, states, aux, state_banks, scales, codebooks, outputs, row_anchors, values,
            0u, uint(d[18]), uint(d[19]), uint(d[20]), uint(d[24]), uint(d[23]), uint(d[22]),
            k_lane, 8u, uint(d[13]));
    } else if ((PROFILE_MASK & 128u) != 0u && profile == 7u) {
        mfq_nvq_jsc_profile<8u, K, MATRIX_ROWS, 1u, false, 0u, INPUT_ROWS>(
            inputs, indices, states, aux, scales, state_banks, codebooks,
            0u, outputs, row_anchors, values, uint(d[8]), uint(d[11]), uint(d[18]),
            uint(d[19]), uint(d[20]), uint(d[22]), uint(d[23]), uint(d[24]),
            k_lane, K_LANES, mfq_moe_vq_layout<7u, 65u>(d));
    }
    constexpr uint PHYSICAL_ROWS = (32u / K_LANES) * ROWS_PER_SIMD;
    for (uint row = 0u; row < MATRIX_ROWS; ++row) {
        for (uint input_row = 0u; input_row < INPUT_ROWS; ++input_row) {
            uint index = row * INPUT_ROWS + input_row;
            for (uint offset = K_LANES >> 1u; offset > 0u; offset >>= 1u)
                values[index] += simd_shuffle_down(values[index], offset);
            if constexpr (SPLIT_SWIGLU) {
                if (k_lane == 0u)
                    split_values[(input_row * 2u + projection) * PHYSICAL_ROWS
                        + lane_group * ROWS_PER_SIMD + row] = T(values[index]);
            }
        }
    }
    if constexpr (SPLIT_SWIGLU) return;
    for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {
        uint output = output_base + row;
        for (uint input_row = 0u; input_row < INPUT_ROWS; ++input_row) {
            if (k_lane != 0u || output >= OUT || (SPLIT_SWIGLU && projection != 0u)) continue;
            float value = values[row * INPUT_ROWS + input_row];
            if constexpr (SPLIT_SWIGLU || PACKED_SWIGLU) {
                uint local = lane_group * ROWS_PER_SIMD + row;
                float gate = SPLIT_SWIGLU
                    ? float(split_values[input_row * 2u * PHYSICAL_ROWS + local]) : float(T(value));
                float up = SPLIT_SWIGLU
                    ? float(split_values[(input_row * 2u + 1u) * PHYSICAL_ROWS + local])
                    : float(T(values[(row + ROWS_PER_SIMD) * INPUT_ROWS + input_row]));
                if (swiglu_limit > 0.0f) {
                    gate = min(gate, swiglu_limit);
                    up = clamp(up, -swiglu_limit, swiglu_limit);
                }
                value = gate / (1.0f + exp(-gate)) * up;
            }
            uint destination = SPLIT_SWIGLU || PACKED_SWIGLU
                ? route_rows[input_row] * OUT + output
                : (route_rows[input_row] * PROJECTIONS + projection) * OUT + output;
            y[destination] = T(value);
        }
    }
}

template <uint INPUT_ROWS, uint ROWS_PER_SIMD, uint K_LANES, uint OUT,
    typename RouteStream, typename YStream, typename T>
inline void mfq_moe_finish_reused_swiglu(
    YStream y, RouteStream route_rows, threadgroup T* values,
    uint output_base, uint lane_group, uint k_lane, uint projection, float limit
) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    constexpr uint PHYSICAL_ROWS = (32u / K_LANES) * ROWS_PER_SIMD;
    if (projection != 0u || k_lane != 0u) return;
    for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {
        const uint output = output_base + row;
        if (output >= OUT) continue;
        const uint local = lane_group * ROWS_PER_SIMD + row;
        for (uint input_row = 0u; input_row < INPUT_ROWS; ++input_row) {
            float gate = float(values[input_row * 2u * PHYSICAL_ROWS + local]);
            float up = float(values[(input_row * 2u + 1u) * PHYSICAL_ROWS + local]);
            if (limit > 0.0f) {
                gate = min(gate, limit);
                up = clamp(up, -limit, limit);
            }
            y[route_rows[input_row] * OUT + output] = T(gate / (1.0f + exp(-gate)) * up);
        }
    }
}

#define MFQ_MOE_FORMAT_ROWS(N, LIMIT, ROWS, MX) \
    mfq_moe_reused_rows<N, MATRIX_ROWS, ROWS_PER_SIMD, uint(K), K_LANES, uint(OUT), \
        uint(MATRIX_OUT), uint(PROJECTIONS), uint(ROUTES), uint(VQ_PROFILE_MASK), SHARED_INPUT != 0, \
        SPLIT_SWIGLU, PACKED_SWIGLU, MX>(x, y, descriptors + descriptor_base, \
        vq_indices, vq_state, vq_aux, vq_anchors, vq_codebooks, vq_scales, \
        vq_state_to_codebank, vq_parameters, mx_values, mx_scales, ROWS, split_values, \
        output_base, lane_group, k_lane, projection, uint(VARIANT_STRIDE), LIMIT)

#define MFQ_MOE_SHARED_ROWS(N, LIMIT, ROWS) \
    do { \
    if (family == 3u) { MFQ_MOE_FORMAT_ROWS(N, LIMIT, ROWS, true); } \
    else { MFQ_MOE_FORMAT_ROWS(N, LIMIT, ROWS, false); } \
    if constexpr (SPLIT_SWIGLU) mfq_moe_finish_reused_swiglu<N, ROWS_PER_SIMD, K_LANES, uint(OUT)>( \
        y, ROWS, split_values, output_base, lane_group, k_lane, projection, LIMIT); \
    } while (false)

)METAL";

constexpr const char* kMoeSource = R"METAL(
#ifndef MFQ_MOE_DUAL_PAGES
#define MFQ_MOE_DUAL_PAGES 0
#endif
#ifndef MFQ_MOE_OTHER_ROW
#define MFQ_MOE_OTHER_ROW (uint(expert) * 2u + (projection ^ 1u))
#define MFQ_MOE_OTHER_DESCRIPTORS descriptors
#endif
    constexpr bool HAS_ROUTE_GROUPS = MFQ_MOE_ROUTE_GROUPS != 0;
#if !MFQ_MOE_ROUTE_GROUPS
    device const int* route_groups = nullptr;
#endif
    constexpr uint SIMD_GROUPS = 2u;
    constexpr uint K_LANES = uint(K_LANES_VALUE);
    constexpr uint LANE_GROUPS = 32u / K_LANES;
    constexpr uint ROWS_PER_SIMD = uint(ROWS_PER_SIMD_VALUE);
    // A split Gate/Up MFE stores the two projections under independent
    // descriptors, and the descriptors may name completely different
    // decoder families.  Pair the two physical SIMD groups instead of
    // pretending the Up rows are the second half of the Gate descriptor.
    constexpr bool SPLIT_SWIGLU =
        uint(FUSED_SWIGLU) != 0u && uint(PROJECTIONS) == 2u;
    constexpr bool PACKED_SWIGLU =
        uint(FUSED_SWIGLU) != 0u && uint(PROJECTIONS) == 1u;
    constexpr uint MATRIX_ROWS =
        PACKED_SWIGLU
            ? 2u * ROWS_PER_SIMD
            : ROWS_PER_SIMD;
    constexpr uint ROWS_PER_PHYSICAL_SIMD =
        LANE_GROUPS * ROWS_PER_SIMD;
    constexpr uint ROWS_PER_TG =
        SPLIT_SWIGLU
            ? ROWS_PER_PHYSICAL_SIMD
            : SIMD_GROUPS * ROWS_PER_PHYSICAL_SIMD;
    constexpr uint OUTPUT_TILES =
        (uint(OUT) + ROWS_PER_TG - 1u) / ROWS_PER_TG;

    uint lane = thread_index_in_simdgroup;
    uint k_lane = lane & (K_LANES - 1u);
    uint lane_group = lane / K_LANES;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint workgroup = threadgroup_position_in_grid.x;
    uint output_tile = workgroup % OUTPUT_TILES;
    uint projection_index = workgroup / OUTPUT_TILES;
    uint projection = SPLIT_SWIGLU
        ? simd_group
        : projection_index % uint(PROJECTIONS);
    uint sorted_slot = SPLIT_SWIGLU
        ? projection_index
        : projection_index / uint(PROJECTIONS);
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
        + (SPLIT_SWIGLU ? 0u : simd_group * ROWS_PER_PHYSICAL_SIMD)
        + lane_group * ROWS_PER_SIMD;
    uint2 route_group = HAS_ROUTE_GROUPS
        ? uint2(*(device const packed_uint2*)(route_groups + route_index * 8u))
        : uint2(uint(expert_ids[route_index]), 0u);
    int logical_expert = int(route_group.x);
#if MFQ_MOE_DUAL_PAGES
    const bool valid_id = logical_expert >= 0 && logical_expert < EXPERT_MAP_SIZE;
    const int gate_expert = EXPERT_MAP_SIZE != 0
        ? (valid_id ? gate_map[logical_expert] : -1) : logical_expert;
    const int up_expert = EXPERT_MAP_SIZE != 0
        ? (valid_id ? up_map[logical_expert] : -1) : logical_expert;
    int expert = projection == 0u ? gate_expert : up_expert;
    int other_expert = projection == 0u ? up_expert : gate_expert;
    const bool invalid_expert = gate_expert < 0 || gate_expert >= EXPERTS
        || up_expert < 0 || up_expert >= geometry[4];
#else
    int expert = logical_expert;
    if (PACKED_EXPERT_IDS != 0) {
        expert = (logical_expert >> 8) - 1;
    } else if (EXPERT_MAP_SIZE != 0) {
        expert = logical_expert >= 0 &&
                logical_expert < int(EXPERT_MAP_SIZE)
            ? expert_map[logical_expert]
            : -1;
    }
    const bool invalid_expert = expert < 0 || expert >= int(EXPERTS);
#endif
    if (invalid_expert) {
        for (
            uint row = 0u;
            row < ROWS_PER_SIMD;
            ++row
        ) {
            uint output = output_base + row;
            if (
                k_lane == 0u
                && output < uint(OUT)
                && (!SPLIT_SWIGLU || projection == 0u)
            ) {
                uint destination = SPLIT_SWIGLU
                    ? (token * uint(ROUTES) + route) * uint(OUT) + output
                    : (
                        (
                            (token * uint(ROUTES) + route)
                                * uint(PROJECTIONS)
                            + projection
                        ) * uint(OUT)
                        + output
                    );
                y[destination] = T(0.0f);
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
    threadgroup T split_values[
        12u * ROWS_PER_PHYSICAL_SIMD];
    threadgroup uint grouped_route_rows[6];
    if constexpr (HAS_ROUTE_GROUPS) {
        if (family == 0u && route_group.y > 1u && route_group.y <= 6u) {
            bool reusable = mfq_moe_nint_supported_gs(uint(descriptors[descriptor_base + 5u]));
            if constexpr (SPLIT_SWIGLU) {
                const uint other = MFQ_MOE_OTHER_ROW * uint(DESCRIPTOR_SIZE);
                reusable = reusable && MFQ_MOE_OTHER_DESCRIPTORS[other] == 0
                    && mfq_moe_nint_supported_gs(uint(MFQ_MOE_OTHER_DESCRIPTORS[other + 5u]));
            }
            if (reusable) {
                device const int* rows = route_groups + route_index * 8u + 2u;
                uint position = 0u;
                while (position < route_group.y && uint(rows[position]) != route_index) ++position;
                const uint first = position & ~1u;
                if (first + 1u < route_group.y) {
                    mfq_moe_nint_pair<MATRIX_ROWS, ROWS_PER_SIMD, uint(K), K_LANES,
                        uint(OUT), uint(MATRIX_OUT), uint(PROJECTIONS), uint(ROUTES),
                        SHARED_INPUT != 0, SPLIT_SWIGLU, PACKED_SWIGLU>(
                        x, y, descriptors + descriptor_base, nint_q, nint_sub_scale, nint_sub_min,
                        rows + first, split_values, output_base, lane_group, k_lane, projection,
                        uint(VARIANT_STRIDE), params[0], position == first);
                    return;
                }
            }
        }
    }
    if (((uint(VQ_PROFILE_MASK) & 474u) != 0u || (uint(FAMILY_MASK) & 8u) != 0u)
            && (!HAS_ROUTE_GROUPS || route_group.y > 1u)
            && (HAS_ROUTE_GROUPS || (uint(TOKENS) > 1u && uint(TOKENS) <= 6u))
            && (family == 1u || family == 3u)) {
        uint route_count = uint(TOKENS) * uint(ROUTES);
        uint preceding = 0u;
        uint matches = 0u;
        if constexpr (HAS_ROUTE_GROUPS) {
            matches = route_group.y;
        } else {
            for (uint index = lane; index < route_count; index += 32u) {
                uint match = uint(expert_ids[index] == logical_expert);
                preceding += uint(index < route_index) * match;
                matches += match;
            }
            preceding = simd_sum(preceding);
            matches = simd_sum(matches);
        }
        if (matches > 1u) {
            bool reusable_vq = family == 3u || mfq_moe_reusable_vq_profile(
                uint(descriptors[descriptor_base + 28u]) & 255u);
            if constexpr (SPLIT_SWIGLU) {
                uint other = MFQ_MOE_OTHER_ROW * uint(DESCRIPTOR_SIZE);
                const uint other_family = uint(MFQ_MOE_OTHER_DESCRIPTORS[other]);
                reusable_vq = reusable_vq && (other_family == 3u
                    || (other_family == 1u && mfq_moe_reusable_vq_profile(uint(MFQ_MOE_OTHER_DESCRIPTORS[other + 28u]) & 255u)));
            }
            if (reusable_vq) {
                if constexpr (HAS_ROUTE_GROUPS) {
                    device const int* rows = route_groups + route_index * 8u + 2u;
                    if (uint(rows[0]) != route_index) return;
                    switch (matches) {
                        case 2u: MFQ_MOE_SHARED_ROWS(2u, params[0], rows); return;
                        case 3u: MFQ_MOE_SHARED_ROWS(3u, params[0], rows); return;
                        case 4u: MFQ_MOE_SHARED_ROWS(4u, params[0], rows); return;
                        case 5u: MFQ_MOE_SHARED_ROWS(5u, params[0], rows); return;
                        case 6u: MFQ_MOE_SHARED_ROWS(6u, params[0], rows); return;
                    }
                } else {
                if (preceding % 6u != 0u) return;
                if (simd_group == 0u) {
                    uint matched = 0u;
                    for (uint base = 0u; base < route_count; base += 32u) {
                        uint index = base + lane;
                        uint match = uint(index < route_count && expert_ids[index] == logical_expert);
                        uint position = matched + simd_prefix_exclusive_sum(match);
                        if (match != 0u && position >= preceding && position < preceding + 6u)
                            grouped_route_rows[position - preceding] = index;
                        matched += simd_sum(match);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                switch (min(matches - preceding, 6u)) {
                    case 2u: MFQ_MOE_SHARED_ROWS(2u, params[0], grouped_route_rows); return;
                    case 3u: MFQ_MOE_SHARED_ROWS(3u, params[0], grouped_route_rows); return;
                    case 4u: MFQ_MOE_SHARED_ROWS(4u, params[0], grouped_route_rows); return;
                    case 5u: MFQ_MOE_SHARED_ROWS(5u, params[0], grouped_route_rows); return;
                    case 6u: MFQ_MOE_SHARED_ROWS(6u, params[0], grouped_route_rows); return;
                }
                }
            }
        }
    }

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
        uint row_metadata_offset =
            uint(descriptors[descriptor_base + 11u]);
        device const uint* row_metadata =
            reinterpret_cast<device const uint*>(
                nint_q + row_metadata_offset);

        MfqNintRow metadata_rows[MATRIX_ROWS];
        uint q_widths[MATRIX_ROWS];
        uint q_row_byte_offsets[MATRIX_ROWS];
        uint q_row_bit_shifts[MATRIX_ROWS];
        float neuron_scales[MATRIX_ROWS];
        float neuron_minimums[MATRIX_ROWS];
        for (uint row = 0u; row < MATRIX_ROWS; ++row) {
            uint output = min(
                output_base + (
                    PACKED_SWIGLU
                        ? (row / ROWS_PER_SIMD) * uint(OUT)
                            + row % ROWS_PER_SIMD
                        : row
                ),
                uint(MATRIX_OUT) - 1u);
            uint pool_output =
                local_expert * uint(MATRIX_OUT) + output;
            const MfqNintRow metadata = mfq_nint_row(row_metadata, pool_output);
            metadata_rows[row] = metadata;
            q_widths[row] = metadata.q_bits;
            q_row_byte_offsets[row] = metadata.q_offset;
            q_row_bit_shifts[row] = metadata.q_shift;
            neuron_scales[row] = metadata.scale;
            neuron_minimums[row] = metadata.minimum;
        }

        MFQ_MFE_NINT_DISPATCH {
        for (uint group = k_lane; group < groups; group += K_LANES) {
            float scales[MATRIX_ROWS];
            float minimums[MATRIX_ROWS];
            for (uint row = 0u; row < MATRIX_ROWS; ++row) {
                const MfqNintRow metadata = metadata_rows[row];
                scales[row] = neuron_scales[row]
                    * float(mfq_nint_sub_value(nint_sub_scale,
                        sub_offset + metadata.sub_offset, metadata.sub_shift,
                        metadata.sub_bits, group));
                minimums[row] = neuron_minimums[row]
                    * float(mfq_nint_sub_value(nint_sub_min,
                        sub_offset + metadata.sub_offset, metadata.sub_shift,
                        metadata.sub_bits, group));
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
                    PACKED_SWIGLU
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
        if (
            (uint(VQ_PROFILE_MASK) & 2u) != 0u
            && profile == 1u
        ) {
            mfq_nvq_banked_profile<MATRIX_ROWS, uint(K), K_LANES, (OUT > K), 4u, 1u, 8u>(
                x, vq_indices, vq_state, vq_aux, vq_state_to_codebank, vq_scales, vq_codebooks,
                outputs, row_anchors, accumulators, x_offset,
                indices_offset, state_offset, aux_offset, state_bank_offset,
                scale_offset, codebook_offset, k_lane, 8u,
                uint(descriptors[descriptor_base + 13u]));
        } else if (
            (uint(VQ_PROFILE_MASK) & 16u) != 0u
            && profile == 4u
        ) {
            mfq_nvq_banked_profile<MATRIX_ROWS, uint(K), K_LANES, (OUT > K), 8u, 1u, 8u>(
                x, vq_indices, vq_state, vq_aux, vq_state_to_codebank, vq_scales, vq_codebooks,
                outputs, row_anchors, accumulators, x_offset,
                indices_offset, state_offset, aux_offset, state_bank_offset,
                scale_offset, codebook_offset, k_lane, 8u,
                uint(descriptors[descriptor_base + 13u]));
        } else if (
            (uint(VQ_PROFILE_MASK) & 128u) != 0u
            && profile == 7u
        ) {
            uint cohort_execution_layout = mfq_moe_vq_layout<7u>(descriptors + descriptor_base);
            if (cohort_execution_layout == 0u) {
                mfq_nvq_banked_profile<MATRIX_ROWS, uint(K), K_LANES, (OUT > K), 8u>(
                    x, vq_indices, vq_state, vq_aux, vq_state_to_codebank, vq_scales, vq_codebooks,
                    outputs, row_anchors, accumulators, x_offset,
                    indices_offset, state_offset, aux_offset, state_bank_offset,
                    scale_offset, codebook_offset, k_lane, index_bits);
            } else {
                mfq_nvq_jsc_profile<8u, uint(K), MATRIX_ROWS, VQ_EXECUTION_LAYOUT, false, 0u>(
                    x, vq_indices, vq_state, vq_aux, vq_scales,
                    vq_state_to_codebank, vq_codebooks, x_offset,
                    outputs, row_anchors, accumulators,
                    index_bits, entries, indices_offset, state_offset,
                    aux_offset, codebook_offset, scale_offset,
                    state_bank_offset, k_lane, K_LANES,
                    mfq_moe_vq_layout<7u, 65u>(descriptors + descriptor_base));
            }
        } else if (
            (uint(VQ_PROFILE_MASK) & 256u) != 0u
            && profile == 8u
        ) {
            mfq_nvq_banked_profile<MATRIX_ROWS, uint(K), K_LANES, (OUT > K)>(
                x, vq_indices, vq_state, vq_aux, vq_state_to_codebank, vq_scales, vq_codebooks,
                outputs, row_anchors, accumulators, x_offset,
                indices_offset, state_offset, aux_offset, state_bank_offset,
                scale_offset, codebook_offset, k_lane, index_bits);
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
            (uint(VQ_PROFILE_MASK) & 64u) != 0u && profile == 6u
        ) {
            mfq_nvq1_profile<MATRIX_ROWS, uint(K), K_LANES, 9u>(
                x, vq_indices, vq_state, vq_aux, vq_scales, vq_codebooks, outputs, row_anchors,
                accumulators, vq_parameters[parameter_offset], x_offset,
                indices_offset, state_offset, aux_offset, scale_offset, codebook_offset, k_lane);
        } else if (
            (uint(VQ_PROFILE_MASK) & 8u) != 0u && profile == 3u
        ) {
            mfq_nvq1_profile<MATRIX_ROWS, uint(K), K_LANES, 11u>(
                x, vq_indices, vq_state, vq_aux, vq_scales, vq_codebooks, outputs, row_anchors,
                accumulators, vq_parameters[parameter_offset], x_offset,
                indices_offset, state_offset, aux_offset, scale_offset, codebook_offset, k_lane);
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
                float4 activations[2] = {
                    mfq_nvq_load_input4<uint(K)>(x, x_offset + column_base, column_base),
                    vector_size == 8u && column_base + 4u < uint(K)
                        ? mfq_nvq_load_input4<uint(K)>(x, x_offset + column_base + 4u, column_base + 4u)
                        : float4(0.0f),
                };
                uint vector = column_base >> (vector_size == 8u ? 3u : 2u);
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
                    uint code_base = codebook_offset + ((table_banks[row] * code_banks
                        + selected_code_banks[row]) * entries + index) * vector_size;
                    uint parity = (popcount(aux_value) & 1u)
                        ^ (aux_mode == 2u ? (index >> 7u) & 1u : 0u);
                    uint full_sign = aux_value | (parity << 7u);
                    float value = 0.0f;
                    #pragma clang loop unroll(full)
                    for (uint quad = 0u; quad < 8u; quad += 4u) {
                        if (quad >= vector_size) break;
                        float4 codes = mfq_moe_load_code4(vq_codebooks, code_base + quad);
                        if (aux_mode == 1u || aux_mode == 2u) {
                            uint sign = full_sign >> ((column_base + quad) & 7u);
                            codes = select(codes, -codes, bool4((sign & 1u) != 0u,
                                (sign & 2u) != 0u, (sign & 4u) != 0u, (sign & 8u) != 0u));
                        } else if (aux_mode == 3u) {
                            float delta = vq_parameters[parameter_offset];
                            codes += delta_values[row] != 0u ? -delta : delta;
                        }
                        value += dot(activations[quad / 4u], codes);
                    }
                    accumulators[row] = fma(weight_scales[row], value, accumulators[row]);
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
                        PACKED_SWIGLU
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
                        PACKED_SWIGLU
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
                        PACKED_SWIGLU
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
                        PACKED_SWIGLU
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
                            PACKED_SWIGLU
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

    if (SPLIT_SWIGLU) {
        for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {
            float total = accumulators[row];
            for (
                uint offset = K_LANES >> 1;
                offset > 0u;
                offset >>= 1
            ) {
                total += simd_shuffle_down(total, offset);
            }
            if (k_lane == 0u) {
                split_values[
                    projection * ROWS_PER_PHYSICAL_SIMD
                    + lane_group * ROWS_PER_SIMD
                    + row
                ] = T(total);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_group == 0u && k_lane == 0u) {
            for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {
                uint local_output =
                    lane_group * ROWS_PER_SIMD + row;
                uint output = output_tile * ROWS_PER_TG + local_output;
                if (output < uint(OUT)) {
                    // Preserve the exact unfused graph boundary: both
                    // projections round before the elementwise activation.
                    float gate = float(split_values[local_output]);
                    float up = float(split_values[
                        ROWS_PER_PHYSICAL_SIMD + local_output]);
                    if (params[0] > 0.0f) {
                        gate = min(gate, params[0]);
                        up = clamp(up, -params[0], params[0]);
                    }
                    float activated = gate / (1.0f + exp(-gate));
                    y[
                        (token * uint(ROUTES) + route) * uint(OUT)
                        + output
                    ] = T(activated * up);
                }
            }
        }
    } else if (uint(FUSED_SWIGLU) != 0u) {
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
    const uint M = uint(x_shape[0]);
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

array allocate_packed_array(std::size_t bytes, Dtype dtype) {
    if (bytes == 0 || bytes % dtype.size() != 0) {
        throw std::runtime_error("MFE packed stream is misaligned");
    }
    const auto layout = detail::packed_storage_layout(bytes / dtype.size());
    const Shape shape = layout.is_matrix()
        ? Shape{layout.rows, layout.columns} : Shape{layout.columns};
    auto result = array(mlx::core::allocator::malloc(bytes), shape, dtype);
    MlxWeightResidency::track(result);
    return result;
}

template <typename Array = array>
Array wrap_wire_buffer(mlx::core::allocator::Buffer buffer, const Shape& shape,
    std::function<void(mlx::core::allocator::Buffer)> deleter) {
    if constexpr (requires { Array(buffer, shape, mlx::core::uint8, deleter); })
        return Array(buffer, shape, mlx::core::uint8, std::move(deleter));
    else
        return Array(typename Array::Data(buffer, std::move(deleter)), shape, mlx::core::uint8);
}

array allocate_wire_array(std::size_t bytes, bool account = false) {
    struct Storage {
        std::uint8_t* data = nullptr;
        std::size_t bytes = 0;
        bool accounted = false;
        mlx::core::allocator::Buffer buffer{nullptr};
        ~Storage() {
            if (buffer.ptr()) mlx::core::allocator::release(buffer);
            detail::MmapAllocator<std::uint8_t>{}.deallocate(data, bytes);
            if (accounted) uncached_packed_storage_bytes.fetch_sub(bytes, std::memory_order_relaxed);
        }
    };
    const auto layout = detail::packed_storage_layout(bytes);
    const Shape shape = layout.is_matrix()
        ? Shape{layout.rows, layout.columns} : Shape{layout.columns};
    const auto page = static_cast<std::size_t>(getpagesize());
    auto storage = std::make_shared<Storage>();
    storage->bytes = checked_add(bytes, page - 1, "MFE wire allocation") / page * page;
    storage->data = detail::MmapAllocator<std::uint8_t>{}.allocate(storage->bytes);
    storage->buffer = mlx::core::allocator::make_buffer(storage->data, storage->bytes);
    if (!storage->buffer.ptr()) return allocate_packed_array(bytes, mlx::core::uint8);
    if (account) {
        uncached_packed_storage_bytes.fetch_add(storage->bytes, std::memory_order_relaxed);
        storage->accounted = true;
    }
    const auto buffer = storage->buffer;
    return wrap_wire_buffer(buffer, shape,
        [storage = std::move(storage)](mlx::core::allocator::Buffer) {});
}

array empty_packed_array(Dtype dtype) {
    struct Entry {
        std::once_flag initialized;
        std::optional<array> value;
    };
    static std::array<Entry, static_cast<std::size_t>(Dtype::Val::complex64) + 1> values;
    auto& entry = values.at(static_cast<std::size_t>(dtype.val()));
    std::call_once(entry.initialized, [&] {
        auto value = allocate_packed_array(dtype.size(), dtype);
        std::memset(value.data<std::uint8_t>(), 0, dtype.size());
        entry.value.emplace(std::move(value));
    });
    return *entry.value;
}

struct MfeDirectReadTarget {
    std::uint64_t offset;
    std::uint8_t* wire;
    std::size_t bytes;
    std::uint8_t* destination = nullptr;
};
thread_local std::vector<MfeDirectReadTarget>* direct_read_targets = nullptr;
enum class MfeMetadataRead { Planned, Eager, DeferredMxScales };
thread_local MfeMetadataRead direct_read_metadata = MfeMetadataRead::Planned;
struct MfeDirectReadScope {
    explicit MfeDirectReadScope(std::vector<MfeDirectReadTarget>& targets)
        : previous(std::exchange(direct_read_targets, &targets)) {}
    ~MfeDirectReadScope() { direct_read_targets = previous; }
    std::vector<MfeDirectReadTarget>* previous;
};
bool redirect_direct_read(std::span<const std::uint8_t> source, std::uint8_t* destination) {
    if (!direct_read_targets) return false;
    const auto begin = reinterpret_cast<std::uintptr_t>(source.data());
    std::size_t copied = 0;
    bool redirected = false;
    for (auto& range : *direct_read_targets) {
        const auto address = reinterpret_cast<std::uintptr_t>(range.wire);
        if (address < begin || address - begin > source.size()
            || range.bytes > source.size() - (address - begin)) continue;
        const auto offset = static_cast<std::size_t>(address - begin);
        if (offset < copied || range.destination) throw std::runtime_error("overlapping direct MFE read");
        std::memcpy(destination + copied, source.data() + copied, offset - copied);
        range.destination = destination + offset;
        copied = offset + range.bytes;
        redirected = true;
    }
    if (redirected) std::memcpy(destination + copied, source.data() + copied, source.size() - copied);
    return redirected;
}

// Keep source ownership, not a second model-sized byte vector. Once all chunks
// are known, write them directly into one exact-sized Metal allocation.
class PackedStream {
public:
    PackedStream() = default;
    PackedStream(PackedStream&&) = default;
    PackedStream& operator=(PackedStream&&) = default;
    PackedStream(const PackedStream&) = delete;
    PackedStream& operator=(const PackedStream&) = delete;

    std::size_t size() const noexcept { return size_; }

    void retain_source(const array& source) { source_ = source; }

    void append(array source) {
        const auto bytes = source.nbytes();
        append_chunk(std::move(source), bytes);
    }

    void append(std::span<const std::uint8_t> source) {
        // Borrowed wire views live until from_blob finishes materialization.
        append_chunk(source, source.size());
    }

    template <typename T>
    void append(detail::StagingVector<T> source) {
        const auto bytes = checked_product(source.size(), sizeof(T), "packed stream size");
        append_chunk(std::move(source), bytes);
    }

    void append(float value) {
        std::array<std::uint8_t, sizeof(float)> bytes;
        std::memcpy(bytes.data(), &value, sizeof(value));
        append_chunk(bytes, bytes.size());
    }

    void push_back(std::uint8_t value) {
        append_chunk(value, 1);
    }

    array materialize(Dtype dtype, const array* arena = nullptr, std::size_t arena_offset = 0) && {
        if (size_ == 0) return empty_packed_array(dtype);
        const auto bytes = size_;
        if (bytes % dtype.size() != 0) {
            throw std::runtime_error("MFE packed stream is misaligned");
        }
        const auto layout = detail::packed_storage_layout(bytes / dtype.size());
        const Shape shape = layout.is_matrix()
            ? Shape{layout.rows, layout.columns} : Shape{layout.columns};
        if (chunks_.size() == 1) {
            if (const auto* source = std::get_if<array>(&chunks_.front());
                source != nullptr && source->dtype() == dtype) {
                auto result = mlx::core::reshape(*source, shape);
                result.eval();
                MlxWeightResidency::track(result);
                return result;
            }
        }
        if (source_ && bytes >= (std::size_t{8} << 20)
            && source_->nbytes() <= std::size_t(std::numeric_limits<int>::max())
            && bytes <= std::size_t(std::numeric_limits<int>::max())) {
            const auto wire = mlx::core::reshape(*source_, Shape{static_cast<int>(source_->nbytes())});
            const auto base = reinterpret_cast<std::uintptr_t>(source_->data<std::uint8_t>());
            std::vector<array> parts;
            for (const auto& chunk : chunks_) {
                std::visit([&](const auto& source) {
                    using T = std::decay_t<decltype(source)>;
                    if constexpr (std::is_same_v<T, std::uint8_t>) {
                        parts.emplace_back(std::initializer_list<std::uint8_t>{source}, mlx::core::uint8);
                    } else if constexpr (std::is_same_v<T, array>) {
                        parts.push_back(mlx::core::reshape(mlx::core::view(source, mlx::core::uint8),
                            Shape{checked_int(source.nbytes(), "packed source bytes")}));
                    } else {
                        const auto count = source.size() * sizeof(typename T::value_type);
                        const auto* data = reinterpret_cast<const std::uint8_t*>(source.data());
                        if constexpr (std::is_same_v<T, std::span<const std::uint8_t>>) {
                            const auto address = reinterpret_cast<std::uintptr_t>(data);
                            if (address >= base && address - base <= source_->nbytes()
                                && count <= source_->nbytes() - (address - base)) {
                                const auto start = static_cast<int>(address - base);
                                parts.push_back(mlx::core::slice(wire, {start},
                                    {start + static_cast<int>(count)}));
                                return;
                            }
                        }
                        parts.emplace_back(data, Shape{checked_int(count, "packed source bytes")}, mlx::core::uint8);
                    }
                }, chunk);
            }
            auto result = mlx::core::reshape(mlx::core::view(
                mlx::core::concatenate(std::move(parts), 0, mlx::core::Device::gpu), dtype), shape);
            result.eval();
            MlxWeightResidency::track(result);
            return result;
        }
        auto result = arena ? array(shape, dtype, nullptr, {}) : allocate_packed_array(bytes, dtype);
        if (arena) {
            if (arena_offset > arena->nbytes() || bytes > arena->nbytes() - arena_offset)
                throw std::out_of_range("MFE packed arena is too small");
            result.copy_shared_buffer(*arena, result.strides(), result.flags(), bytes / dtype.size(), arena_offset / dtype.size());
            result.set_status(array::Status::available);
        }
        auto* destination = result.data<std::uint8_t>();
        std::size_t offset = 0;
        for (const auto& chunk : chunks_) {
            std::visit([&](const auto& source) {
                using T = std::decay_t<decltype(source)>;
                if constexpr (std::is_same_v<T, std::uint8_t>) {
                    destination[offset++] = source;
                } else {
                    const auto count = [&] {
                        if constexpr (std::is_same_v<T, array>) return source.nbytes();
                        else return source.size() * sizeof(typename T::value_type);
                    }();
                    const auto* data = [&] {
                        if constexpr (std::is_same_v<T, array>) return source.template data<std::uint8_t>();
                        else return source.data();
                    }();
                    if constexpr (std::is_same_v<T, std::span<const std::uint8_t>>) {
                        if (redirect_direct_read(source, destination + offset)) {
                            offset += count;
                            return;
                        }
                    }
                    std::memcpy(destination + offset, data, count);
                    offset += count;
                }
            }, chunk);
        }
        return result;
    }

private:
    using Chunk = std::variant<array, std::span<const std::uint8_t>, std::uint8_t,
        detail::StagingVector<std::uint8_t>, detail::StagingVector<std::uint32_t>,
        detail::StagingVector<float>, detail::StagingVector<std::int8_t>,
        detail::StagingVector<std::int16_t>, std::array<std::uint8_t, sizeof(float)>>;

    template <typename T>
    void append_chunk(T source, std::size_t bytes) {
        if (bytes != 0) {
            const auto total = checked_add(size_, bytes, "packed stream size");
            chunks_.emplace_back(std::move(source));
            size_ = total;
        }
    }

    std::vector<Chunk> chunks_;
    std::optional<array> source_;
    std::size_t size_ = 0;
};

void append_raw(
    PackedStream& target,
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
    target.append(std::move(evaluated));
}

void append_bytes(
    PackedStream& target,
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
    target.append(source);
}

array make_raw_array(PackedStream bytes, Dtype dtype) {
    return std::move(bytes).materialize(dtype);
}

template <typename Allocator>
array make_raw_array(
    std::vector<std::uint8_t, Allocator> bytes,
    Dtype dtype) {
    if (bytes.empty()) {
        return empty_packed_array(dtype);
    }
    auto result = allocate_packed_array(bytes.size(), dtype);
    std::memcpy(
        result.template data<std::uint8_t>(),
        bytes.data(),
        bytes.size());
    return result;
}

void align_packed_stream(
    std::vector<array>& chunks, std::size_t& offset, std::size_t alignment) {
    const auto padding = (alignment - offset % alignment) % alignment;
    if (padding != 0) {
        chunks.push_back(make_raw_array(std::vector<std::uint8_t>(padding, 0), mlx::core::uint8));
        offset = checked_add(offset, padding, "packed stream alignment");
    }
}

array make_int32_array(
    const std::vector<std::int32_t>& values,
    Shape shape,
    const array* arena = nullptr) {
    if (values.size() == 1 && values.front() == 0)
        return mlx::core::reshape(empty_packed_array(mlx::core::int32), std::move(shape));
    if (arena) {
        const auto bytes = checked_product(values.size(), sizeof(std::int32_t), "MFE metadata bytes");
        if (bytes > arena->nbytes()) throw std::out_of_range("MFE metadata arena is too small");
        auto result = array(std::move(shape), mlx::core::int32, nullptr, {});
        result.copy_shared_buffer(*arena, result.strides(), result.flags(), values.size());
        result.set_status(array::Status::available);
        std::memcpy(result.data<std::int32_t>(), values.data(), bytes);
        return result;
    }
    auto result = array(values.begin(), std::move(shape));
    MlxWeightResidency::track(result);
    return result;
}

constexpr std::size_t kPackedCopyInputs = 24;
std::atomic<std::size_t> packed_copy_submissions{0};
std::atomic<std::size_t> indirect_expert_submissions{0};
std::atomic<std::size_t> packed_pinned_bytes{0}, packed_pin_errors{0};
std::atomic<std::size_t> pooled_expert_staging{0};
using PackedCopySpec = std::array<std::uint32_t, 4>;

struct PackedCopyBatch {
    std::uint32_t first;
    std::uint32_t count;
    std::uint32_t source;
    bool operator==(const PackedCopyBatch&) const = default;
};

class PackedCopyPrimitive final : public mlx::core::UnaryPrimitive, public MlxPreparableKernel {
public:
    PackedCopyPrimitive(mlx::core::Stream stream, Shape shape, std::vector<PackedCopyBatch> batches)
        : UnaryPrimitive(stream), shape_(std::move(shape)), batches_(std::move(batches)) {}
    std::string preparation_key() const override { return "mfq_mfe_packed_copy_v2"; }
    void prepare_gpu() override { (void)kernel(); }
    void eval_cpu(const std::vector<array>&, array&) override {
        throw std::runtime_error("packed expert copy requires Metal");
    }
    void eval_gpu(const std::vector<array>& inputs, array& output) override {
        if (inputs.size() < 2) throw std::logic_error("packed expert copy has no sources");
        if (output.nbytes() >= (std::size_t{64} << 10)) {
            auto storage = allocate_wire_array(output.nbytes(), true);
            output.copy_shared_buffer(storage, output.strides(), output.flags(), output.size());
        } else output.set_data(mlx::core::allocator::malloc(output.nbytes()));
        auto& encoder = mlx::core::metal::get_command_encoder(stream());
        encoder.set_compute_pipeline_state(kernel());
        encoder.set_input_array(inputs.back(), kPackedCopyInputs);
        encoder.set_output_array(output, kPackedCopyInputs + 1);
        const auto empty = empty_packed_array(mlx::core::uint8);
        for (const auto& batch : batches_) {
            for (std::size_t index = 0; index < kPackedCopyInputs; ++index) {
                const auto source = batch.source + index;
                encoder.set_input_array(source < inputs.size() - 1 ? inputs[source] : empty, index);
            }
            const std::array<std::uint32_t, 2> geometry{batch.first, batch.source};
            encoder.set_bytes(geometry, kPackedCopyInputs + 2);
            encoder.dispatch_threadgroups(MTL::Size(batch.count, 1, 1), MTL::Size(256, 1, 1));
        }
        packed_copy_submissions.fetch_add(1, std::memory_order_relaxed);
    }
    const char* name() const override { return "PackedCopyPrimitive"; }
    bool is_equivalent(const mlx::core::Primitive& other) const override {
        const auto* value = dynamic_cast<const PackedCopyPrimitive*>(&other);
        return value && value->shape_ == shape_ && value->batches_ == batches_;
    }
    std::vector<Shape> output_shapes(const std::vector<array>&) override { return {shape_}; }
private:
    MTL::ComputePipelineState* kernel() {
        auto& device = kernel_device(stream().device);
        auto* library = device.get_library(preparation_key(), [] {
            std::ostringstream source;
            source << "#include <metal_stdlib>\nusing namespace metal;\n"
                "kernel void mfq_mfe_packed_copy_v2(\n";
            for (std::size_t index = 0; index < kPackedCopyInputs; ++index)
                source << "device const uchar* p" << index << " [[buffer(" << index << ")]],\n";
            source << R"(device const uint4* moves [[buffer(24)]],
device uchar* output [[buffer(25)]], constant uint* geometry [[buffer(26)]],
uint3 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    const uint4 move = moves[geometry[0] + group.x];
    const uint bytes = move.w & 0x0fffffffu;
    if (move.x == 0xffffffffu) {
        for (uint index = lane; index < bytes; index += 256) output[move.z + index] = 0;
        return;
    }
    device const uchar* input = nullptr;
    switch (move.x - geometry[1]) {
)";
            for (std::size_t index = 0; index < kPackedCopyInputs; ++index)
                source << "case " << index << ": input = p" << index << "; break;\n";
            source << R"(default: return;
    }
    input += move.y;
    device uchar* target = output + move.z;
    const uint alignment = move.y | move.z | (move.w >> 28);
    if ((alignment & 3u) == 0) {
        device const uint* words = reinterpret_cast<device const uint*>(input);
        device uint* destination = reinterpret_cast<device uint*>(target);
        for (uint index = lane; index < bytes / 4; index += 256) destination[index] = words[index];
        for (uint index = (bytes & ~3u) + lane; index < bytes; index += 256) target[index] = input[index];
    } else {
        for (uint index = lane; index < bytes; index += 256) target[index] = input[index];
    }
}
)";
            return source.str();
        });
        return device.get_kernel(preparation_key(), library);
    }
    Shape shape_;
    std::vector<PackedCopyBatch> batches_;
};

array packed_expert_copy(std::vector<array> sources,
    const std::vector<std::vector<PackedCopySpec>>& copies, std::size_t bytes) {
    if (sources.empty() || bytes == 0 || bytes > std::size_t(std::numeric_limits<int>::max()))
        throw std::invalid_argument("invalid packed expert copy extent");
    std::vector<std::uint32_t> moves;
    std::vector<PackedCopyBatch> batches;
    for (std::size_t index = 0; index < copies.size(); ++index) {
        if (copies[index].empty()) continue;
        batches.push_back({static_cast<std::uint32_t>(moves.size() / 4),
            static_cast<std::uint32_t>(copies[index].size()),
            static_cast<std::uint32_t>(index * kPackedCopyInputs)});
        for (const auto& copy : copies[index]) moves.insert(moves.end(), copy.begin(), copy.end());
    }
    if (moves.empty()) throw std::invalid_argument("packed expert copy has no ranges");
    sources.emplace_back(moves.begin(), Shape{checked_int(moves.size() / 4, "packed copy ranges"), 4});
    const Shape shape{static_cast<int>(bytes)};
    return array(shape, mlx::core::uint8,
        std::make_shared<PackedCopyPrimitive>(mlx::core::default_stream(mlx::core::Device::gpu), shape, std::move(batches)),
        std::move(sources));
}

struct PackedExpertArena {
    array storage;
    struct Stream {
        const std::vector<array>* values;
        std::size_t offset;
        std::size_t bytes;
    };
    std::vector<Stream> streams;
};

std::optional<PackedExpertArena> pack_expert_streams(std::vector<array> sources,
    std::span<const std::vector<array>* const> streams) {
    std::unordered_map<const void*, std::uint32_t> owners;
    std::unordered_map<std::uintptr_t, std::uint32_t> external;
    std::vector<array> unique;
    for (const auto& source : sources) {
        if (!source.is_available() || !source.flags().row_contiguous)
            continue;
        if (owners.emplace(source.data_shared_ptr().get(), static_cast<std::uint32_t>(unique.size())).second) {
            if (source.dtype() == mlx::core::uint8
                && source.buffer_size() <= std::size_t(std::numeric_limits<int>::max())) {
                auto backing = array(Shape{static_cast<int>(source.buffer_size())}, mlx::core::uint8, nullptr, {});
                backing.copy_shared_buffer(source, backing.strides(), backing.flags(), backing.size(), -source.offset());
                backing.set_status(array::Status::available);
                unique.push_back(std::move(backing));
            } else unique.push_back(source);
        }
    }
    std::vector<std::vector<PackedCopySpec>> copies(std::max<std::size_t>(1,
        (unique.size() + kPackedCopyInputs - 1) / kPackedCopyInputs));
    std::vector<PackedExpertArena::Stream> views;
    std::size_t bytes = 0;
    constexpr std::size_t tile = std::size_t{64} << 10;
    for (const auto* values : streams) {
        if (values->empty()) continue;
        const auto empty = empty_packed_array(values->front().dtype()).data_shared_ptr();
        if (std::all_of(values->begin(), values->end(), [&](const auto& value) {
                return value.data_shared_ptr() == empty;
            })) continue;
        bytes = checked_add(bytes, 15, "packed copy alignment") & ~std::size_t{15};
        const auto first = bytes;
        for (const auto& value : *values) {
            if (!value.is_available() || !value.flags().row_contiguous) return std::nullopt;
            const auto next = checked_add(bytes, value.nbytes(), "packed copy output");
            if (next > std::size_t(std::numeric_limits<int>::max())) return std::nullopt;
            const auto found = owners.find(value.data_shared_ptr().get());
            std::optional<std::pair<std::uint32_t, std::size_t>> range;
            if (found != owners.end()) {
                const auto& source = unique[found->second];
                const auto base = reinterpret_cast<std::uintptr_t>(source.data<std::uint8_t>());
                const auto address = reinterpret_cast<std::uintptr_t>(value.data<std::uint8_t>());
                if (address >= base && address - base <= source.nbytes()
                    && value.nbytes() <= source.nbytes() - (address - base))
                    range.emplace(found->second, address - base);
            }
            if (!range) {
                const auto* data = value.data<std::uint8_t>();
                if (value.nbytes() <= 16 && std::all_of(data, data + value.nbytes(), [](auto v) { return v == 0; })) {
                    copies.front().push_back({std::numeric_limits<std::uint32_t>::max(), 0,
                        static_cast<std::uint32_t>(bytes), static_cast<std::uint32_t>(value.nbytes())});
                } else {
                    const auto address = reinterpret_cast<std::uintptr_t>(data);
                    const auto found = external.find(address);
                    if (found != external.end() && unique[found->second].nbytes() >= value.nbytes())
                        range.emplace(found->second, 0);
                    else {
                        const auto index = static_cast<std::uint32_t>(unique.size());
                        unique.push_back(value);
                        external[address] = index;
                        copies.resize((unique.size() + kPackedCopyInputs - 1) / kPackedCopyInputs);
                        range.emplace(index, 0);
                    }
                }
            }
            if (range) {
                const auto alignment = static_cast<std::uint32_t>(
                    reinterpret_cast<std::uintptr_t>(unique[range->first].data<std::uint8_t>()) & 15u) << 28;
                for (std::size_t offset = 0; offset < value.nbytes(); offset += tile)
                    copies[range->first / kPackedCopyInputs].push_back({range->first,
                        static_cast<std::uint32_t>(range->second + offset),
                        static_cast<std::uint32_t>(bytes + offset),
                        static_cast<std::uint32_t>(std::min(tile, value.nbytes() - offset)) | alignment});
            }
            bytes = next;
        }
        views.push_back({values, first, bytes - first});
    }
    if (!bytes) return std::nullopt;
    if (unique.empty()) unique.push_back(empty_packed_array(mlx::core::uint8));
    return PackedExpertArena{packed_expert_copy(std::move(unique), copies, bytes), std::move(views)};
}

struct MfePageResources {
    NS::SharedPtr<MTL::ResidencySet> residency;
    std::vector<array> owners;
};

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
    std::shared_ptr<const std::vector<int>> nint_group_sizes;
    int descriptor_size = 0;
    int variant_stride = 0;
    int shared_input = 0;
    int vq_execution_layout = 0;
    std::uint64_t vq_layouts = UINT64_MAX;
    int family_mask = 0;
    int vq_profile_mask = 0;
    int has_nepq_residual = 0;
    int npq_grouped_indices = 0;
    int sorted_routes = 0;
    int expert_map_size = 0;
    int packed_expert_ids = 0;
    bool route_groups = false;
    bool nint_q_dispatch = false;
    bool indirect_pages = false;
    bool dual_pages = false;
    bool weighted_reduce = false;
    int direct_page_mask = 0;
    int other_experts = 0;
    int workgroups = 0;
    std::shared_ptr<const MfePageResources> page_resources;
    std::shared_ptr<const MfePageResources> other_page_resources;
};

constexpr std::array<std::pair<const char*, const char*>, 21> kExpertStreams{{
    {"nint_q", "uchar"}, {"nint_sub_scale", "uchar"}, {"nint_sub_min", "uchar"},
    {"nint_anchor_scale", "float"}, {"nint_anchor_min", "float"},
    {"q8_q", "int8_t"}, {"q8_scales", "half"},
    {"vq_indices", "uchar"}, {"vq_state", "uchar"}, {"vq_aux", "uchar"},
    {"vq_anchors", "float"}, {"vq_codebooks", "int8_t"}, {"vq_scales", "float"},
    {"vq_state_to_codebank", "uchar"}, {"vq_banks", "uchar"}, {"vq_parameters", "float"},
    {"vq_residual_codebooks", "float"}, {"vq_residual_first", "short"},
    {"vq_residual_second", "short"}, {"mx_values", "uchar"}, {"mx_scales", "uchar"}
}};

bool uses_indirect_expert_pages(std::uint32_t families) {
    if (families == (1u << kFamilyMxfp4) || mlx_reference_enabled()) return false;
    if (__builtin_available(macOS 15, *))
        return mlx::core::metal::device(mlx::core::Device::gpu).mtl_device()->supportsFamily(MTL::GPUFamilyMetal3);
    return false;
}

void retain_indirect_resources(mlx::core::metal::CommandEncoder& encoder, std::span<const array> sources) {
    struct Residency {
        NS::SharedPtr<MTL::ResidencySet> set;
        std::unordered_set<const MTL::Buffer*> buffers;
    };
    static thread_local std::unordered_map<MTL::CommandBuffer*, std::weak_ptr<Residency>> pending;
    auto* command = encoder.get_command_buffer();
    auto state = pending[command].lock();
    if (!state) {
        for (auto it = pending.begin(); it != pending.end();)
            if (it->second.expired()) it = pending.erase(it); else ++it;
        state = std::make_shared<Residency>();
        auto descriptor = NS::TransferPtr(MTL::ResidencySetDescriptor::alloc()->init());
        descriptor->setInitialCapacity(sources.size());
        NS::Error* error = nullptr;
        auto& device = mlx::core::metal::device(mlx::core::Device::gpu);
        state->set = NS::TransferPtr(device.mtl_device()->newResidencySet(descriptor.get(), &error));
        if (!state->set) throw std::runtime_error(error
            ? error->localizedDescription()->utf8String() : "expert residency set unavailable");
        pending[command] = state;
        command->useResidencySet(state->set.get());
        command->addCompletedHandler([state](MTL::CommandBuffer*) {});
    }
    bool changed = false;
    for (const auto& source : sources) {
        auto* buffer = static_cast<const MTL::Buffer*>(source.buffer().ptr());
        if (state->buffers.insert(buffer).second) {
            state->set->addAllocation(buffer);
            encoder.add_temporary(source);
            changed = true;
        }
    }
    if (changed) state->set->commit();
}

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
    std::shared_ptr<const std::vector<int>> nint_group_sizes;
    std::shared_ptr<const std::vector<std::tuple<int, int, int>>> cohort_populations;
    bool use_nax = false;
    bool direct_nax = false;
    bool indirect_pages = false;
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
        if (ordered) inputs.push_back("expert_order");
        std::string source = ordered
            ? "#define MFQ_SCHEDULED_EXPERT(index) expert_order[index]\n"
            : "#define MFQ_SCHEDULED_EXPERT(index) int(index)\n";
        source +=
            R"METAL(
                const int M = indices_shape[0];
                const int NUM_EXPERTS = int(threads_per_threadgroup.x);
                const int MAX_BLOCKS = min(max(M, 1), M / BM + int(M % BM != 0) + NUM_EXPERTS);
                const uint schedule_index = thread_index_in_threadgroup;
                const uint expert = uint(
                    MFQ_SCHEDULED_EXPERT(schedule_index));

                threadgroup atomic_int local_count;
                threadgroup int expert_starts[1024];
                threadgroup int expert_rows[1024];
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
        && block_rows != 80 && block_rows != 96 && block_rows != 128) {
        throw std::invalid_argument(
            "grouped MMQ block rows must be 32, 48, 64, 80, 96, or 128");
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
    const auto population_bound =
        (static_cast<std::size_t>(route_count) + block_rows - 1) / block_rows
        + static_cast<std::size_t>(experts);
    const int max_blocks = checked_int(
        std::min(static_cast<std::size_t>(std::max(route_count, 1)), population_bound),
        "grouped MMQ max blocks");
    auto sorted_ids = mlx::core::contiguous(
        mlx::core::take(
            mlx::core::reshape(ids, Shape{route_count}),
            route_order,
            0));
    const bool cohort_ordered = experts >= 128
        && route_count > experts * 96;
    const int block_chunk = cohort_ordered ? 2 : 0;
    std::vector<array> builder_inputs{std::move(sorted_ids)};
    if (cohort_ordered) builder_inputs.push_back(expert_order);
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
            {"BM", block_rows},
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

std::string nint_group_specialization(
    const std::shared_ptr<const std::vector<int>>& group_sizes) {
    if (!group_sizes || group_sizes->empty()) return {};
    std::ostringstream source;
    source << "#define MFQ_MFE_NINT_DISPATCH_WITH(CALL) ";
    for (std::size_t index = 0; index < group_sizes->size(); ++index) {
        const auto group_size = (*group_sizes)[index];
        source << (index == 0 ? "if" : "else if")
            << " (group_size == " << group_size << "u) { CALL("
            << group_size << "u); } ";
    }
    source << "else\n";
    return source.str();
}

void append_nint_group_key(
    std::ostringstream& name,
    const std::shared_ptr<const std::vector<int>>& group_sizes) {
    if (group_sizes)
        for (const auto group_size : *group_sizes) name << '_' << group_size;
}

std::string native_moe_kernel_name(
    const NativeMoeConfig& config) {
    std::ostringstream name;
    name
        << "mfq_native_mfe_v16_"
        << (config.dtype == mlx::core::float16 ? "f16" : "f32")
        << (config.tokens == 1 ? "_decode" : "_batch")
        << "_r" << config.routes
        << "_o" << config.output_width
        << "_mo" << config.matrix_output_width
        << "_p" << config.projections
        << "_sw" << config.fused_swiglu
        << "_k" << config.input_width
        << "_kl" << config.k_lanes
        << "_rs" << config.rows_per_simd
        << "_si" << config.shared_input
        << "_vl" << config.vq_execution_layout
        << "_vls" << config.vq_layouts
        << "_fm" << config.family_mask
        << "_vm" << config.vq_profile_mask
        << "_nr" << config.has_nepq_residual
        << "_ni" << config.npq_grouped_indices;
    name << "_sr" << config.sorted_routes;
    name << "_em" << (config.expert_map_size != 0);
    name << "_pe" << config.packed_expert_ids;
    name << "_rg" << config.route_groups;
    name << "_qd" << config.nint_q_dispatch;
    if (config.indirect_pages) name << "_page_addresses";
    if (config.dual_pages) name << "_dual" << config.direct_page_mask;
    if (config.weighted_reduce) name << "_reduce_v1";
    if (config.projections == 2 && config.fused_swiglu
        && (config.family_mask & (1 << kFamilyMxfp4)) && (config.family_mask & (1 << kFamilyVq)))
        name << "_mixed_pair_v2";
    if (config.route_groups) name << "_pairq";
    name << "_ng";
    append_nint_group_key(name, config.nint_group_sizes);
    return name.str();
}

std::string make_mfe_reduce_body();

std::string make_native_moe_source(
    const NativeMoeConfig& config,
    const std::string& kernel_name) {
    const bool has_expert_map = config.expert_map_size != 0 && !config.dual_pages;
    std::ostringstream source;
    source
        << "#include <metal_stdlib>\n"
        << "using namespace metal;\n"
        << "using T = "
        << (config.dtype == mlx::core::float16 ? "half" : "float")
        << ";\n"
        << nint_group_specialization(config.nint_group_sizes)
        << "#define MFQ_MOE_ROUTE_GROUPS " << config.route_groups << "\n"
        << "#define MFQ_MOE_DUAL_PAGES " << config.dual_pages << "\n"
        << "#define MFQ_NINT_Q_DISPATCH " << config.nint_q_dispatch << "\n"
        << "#define MFQ_MOE_VQ_LAYOUTS " << config.vq_layouts << "ul\n"
        << kMoeHeader;
    if (config.indirect_pages) {
        source << "struct MfqExpertStreams {\n";
        for (const auto& [name, type] : kExpertStreams)
            source << "device const " << type << "* " << name << ";\n";
        source << "};\n";
    }
    source << "\nkernel void " << kernel_name << "(\n"
        << (config.dual_pages ? "device const int* gate_descriptors [[buffer(0)]],\n"
            : "device const int* descriptors [[buffer(0)]],\n")
        << (config.dual_pages ? "device const MfqExpertStreams* gate_streams [[buffer(1)]],\n"
            : config.indirect_pages ? "device const MfqExpertStreams* expert_streams [[buffer(1)]],\n"
            : "device const uchar* nint_q [[buffer(1)]],\n")
        << (config.dual_pages ? "device const int* up_descriptors [[buffer(2)]],\n"
            : "device const uchar* nint_sub_scale [[buffer(2)]],\n")
        << (config.dual_pages ? "device const MfqExpertStreams* up_streams [[buffer(3)]],\n"
            : "device const uchar* nint_sub_min [[buffer(3)]],\n")
        << (config.dual_pages ? "device const int* gate_map [[buffer(4)]],\n"
            : "device const float* nint_anchor_scale [[buffer(4)]],\n")
        << (config.dual_pages ? "device const int* up_map [[buffer(5)]],\n"
            : "device const float* nint_anchor_min [[buffer(5)]],\n")
        << (config.dual_pages ? "device const T* up_x [[buffer(6)]],\n"
            : "device const int8_t* q8_q [[buffer(6)]],\n")
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
        << (config.dual_pages ? "device const T* gate_x [[buffer(22)]],\n"
            : "device const T* x [[buffer(22)]],\n")
        << "device const int* expert_ids [[buffer(23)]],\n"
        << "device const int* route_order [[buffer(24)]],\n"
        << "device const float* params [[buffer(25)]],\n";
    if (has_expert_map) {
        source << "device const int* expert_map [[buffer(26)]],\n";
    }
    const int input_count = 26 + static_cast<int>(has_expert_map) + static_cast<int>(config.route_groups)
        + static_cast<int>(config.weighted_reduce);
    if (config.route_groups)
        source << "device const int* route_groups [[buffer(" << input_count - 1 << ")]],\n";
    if (config.weighted_reduce)
        source << "device const float* route_weights [[buffer(" << input_count - 1 << ")]],\n";
    source
        << "device T* y [[buffer("
        << input_count
        << ")]],\n"
        << "constant int* geometry [[buffer("
        << input_count + 1
        << ")]],\n"
        << "uint thread_index_in_simdgroup "
           "[[thread_index_in_simdgroup]],\n"
        << "uint simdgroup_index_in_threadgroup "
           "[[simdgroup_index_in_threadgroup]],\n"
        << "uint3 threadgroup_position_in_grid "
           "[[threadgroup_position_in_grid]]) {\n"
        << (config.tokens == 1
            ? "constexpr int TOKENS = 1;\n"
            : "const int TOKENS = geometry[0];\n")
        << "constexpr int ROUTES = " << config.routes << ";\n"
        << "const int EXPERTS = geometry[2];\n"
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
        << (config.tokens == 1
            ? "constexpr int VARIANT_STRIDE = " + std::to_string(config.variant_stride) + ";\n"
            : "const int VARIANT_STRIDE = geometry[1];\n")
        << "constexpr int SHARED_INPUT = "
        << config.shared_input << ";\n"
        << "constexpr int VQ_EXECUTION_LAYOUT = "
        << config.vq_execution_layout << ";\n"
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
        << "const int EXPERT_MAP_SIZE = geometry[3];\n"
        << "constexpr int PACKED_EXPERT_IDS = "
        << config.packed_expert_ids << ";\n";
    source << "#define MFQ_MOE_OTHER_ROW "
        << (config.dual_pages ? "uint(other_expert)" : "(uint(expert) * 2u + (projection ^ 1u))") << "\n"
        << "#define MFQ_MOE_OTHER_DESCRIPTORS "
        << (config.dual_pages ? "(projection == 0u ? up_descriptors : gate_descriptors)" : "descriptors") << "\n";
    if (!has_expert_map) {
        source << "device const int* expert_map = expert_ids;\n";
    }
    std::string body = config.weighted_reduce ? make_mfe_reduce_body() : kMoeSource;
    if (config.indirect_pages) {
        const std::string marker = "    uint descriptor_base = (\n"
            "        uint(expert) * uint(PROJECTIONS)\n"
            "        + projection\n"
            "    ) * uint(DESCRIPTOR_SIZE);";
        const auto position = body.find(marker);
        if (position == std::string::npos) throw std::logic_error("MFE page binding marker is missing");
        if (config.dual_pages) body.replace(position, marker.size(),
            "    uint descriptor_base = uint(expert) * uint(DESCRIPTOR_SIZE);");
        std::ostringstream bindings;
        if (config.dual_pages) {
            bindings << "device const int* descriptors = projection == 0u ? gate_descriptors : up_descriptors;\n"
                << "device const T* x = projection == 0u ? gate_x : up_x;\n"
                << "device const MfqExpertStreams* selected_stream = projection == 0u ? gate_streams + "
                << ((config.direct_page_mask & 1) ? "0u" : "uint(expert)") << " : up_streams + "
                << ((config.direct_page_mask & 2) ? "0u" : "uint(expert)") << ";\n";
            for (const auto& [name, type] : kExpertStreams)
                bindings << "#define " << name << " selected_stream->" << name << "\n";
        } else for (const auto& [name, type] : kExpertStreams)
            bindings << "#define " << name << " expert_streams[uint(expert)]." << name << "\n";
        body.insert(position, bindings.str());
    }
    source << body << "\n}\n";
    return source.str();
}

class NativeMfePrimitive final
    : public mlx::core::UnaryPrimitive, public MlxPreparableKernel {
public:
    NativeMfePrimitive(
        mlx::core::Stream stream,
        NativeMoeConfig config)
        : UnaryPrimitive(stream),
          config_(std::move(config)),
          kernel_name_(native_moe_kernel_name(config_)) {}

    std::string preparation_key() const override { return kernel_name_; }
    void prepare_gpu() override { (void)prepared_kernel(); }

    MTL::ComputePipelineState* prepared_kernel() {
        auto& selected_stream = stream();
        auto& device = kernel_device(
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
        return device.get_kernel(kernel_name_, library);
    }

    void eval_cpu(
        const std::vector<array>&,
        array&) override {
        throw std::runtime_error(
            "native MFE primitive has no CPU path");
    }

    void eval_gpu(
        const std::vector<array>& inputs,
        array& output) override {
        const int input_count = 26 + static_cast<int>(config_.expert_map_size != 0 && !config_.dual_pages)
            + static_cast<int>(config_.route_groups) + static_cast<int>(config_.weighted_reduce);
        if (inputs.size() < static_cast<std::size_t>(input_count)
            || (!config_.indirect_pages && inputs.size() != static_cast<std::size_t>(input_count))) {
            throw std::logic_error(
                "native MFE primitive input count mismatch");
        }
        output.set_data(
            mlx::core::allocator::malloc(output.nbytes()));
        auto& selected_stream = stream();
        auto* kernel = prepared_kernel();
        auto& encoder =
            mlx::core::metal::get_command_encoder(
                selected_stream);
        encoder.set_compute_pipeline_state(kernel);
        if (config_.indirect_pages) {
            if (config_.page_resources && (!config_.dual_pages || config_.other_page_resources)) {
                auto* command = encoder.get_command_buffer();
                command->useResidencySet(config_.page_resources->residency.get());
                command->addCompletedHandler([resources = config_.page_resources](MTL::CommandBuffer*) {});
                if (config_.other_page_resources) {
                    command->useResidencySet(config_.other_page_resources->residency.get());
                    command->addCompletedHandler([resources = config_.other_page_resources](MTL::CommandBuffer*) {});
                }
            } else {
                for (std::size_t index = input_count; index < inputs.size(); ++index)
                    encoder.set_input_array(inputs[index], 1);
                retain_indirect_resources(encoder, std::span(inputs).subspan(input_count));
            }
        }
        for (int index = 0; index < input_count; ++index) {
            encoder.set_input_array(
                inputs[static_cast<std::size_t>(index)],
                index);
        }
        encoder.set_output_array(output, input_count);
        const std::array<int, 5> geometry{
            config_.tokens, config_.variant_stride, config_.experts, config_.expert_map_size, config_.other_experts};
        encoder.set_bytes(geometry, input_count + 1);
        encoder.dispatch_threadgroups(
            MTL::Size(config_.workgroups, 1, 1),
            MTL::Size(config_.weighted_reduce ? config_.routes * 32 : 64, 1, 1));
        if (config_.indirect_pages) indirect_expert_submissions.fetch_add(1, std::memory_order_relaxed);
    }

    const char* name() const override {
        return "NativeMfePrimitive";
    }

    bool is_equivalent(
        const mlx::core::Primitive& other) const override {
        const auto* primitive =
            dynamic_cast<const NativeMfePrimitive*>(&other);
        return primitive != nullptr
            && primitive->kernel_name_ == kernel_name_
            && primitive->config_.tokens == config_.tokens
            && primitive->config_.variant_stride == config_.variant_stride
            && primitive->config_.experts == config_.experts
            && primitive->config_.other_experts == config_.other_experts
            && primitive->config_.expert_map_size == config_.expert_map_size
            && primitive->config_.workgroups == config_.workgroups
            && primitive->config_.page_resources == config_.page_resources
            && primitive->config_.other_page_resources == config_.other_page_resources;
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

// Adapted from oMLX's two-launch Qwen routed-decode topology (Apache-2.0):
// the first expert launch runs routed and shared Gate/Up with their SwiGLU
// epilogues plus the shared gate; the second assigns one SIMD group to every
// routed expert and one to the shared expert, then combines in-threadgroup.
// The one-token launch uses oMLX's three-candidate lane cache for folded top-k.

struct MfeNintDecodeConfig {
    Dtype dtype;
    Dtype shared_gate_dtype;
    bool shared_gate_is_logits = false;
    int shared_router_group_size = 0;
    bool packed_gate_up = false;
    bool route_groups = false;
    bool folded_topk = false;
    bool gate_q_dispatch = false;
    bool down_q_dispatch = false;
    bool shared_gate_q_dispatch = false;
    bool shared_down_q_dispatch = false;
    Shape stage_one_shape;
    Shape output_shape;
    int tokens = 1;
    int routes = 0;
    int experts = 0;
    int hidden = 0;
    int intermediate = 0;
    int descriptor_size = 0;
    int gate_family_mask = 0;
    int gate_vq_profile_mask = 0;
    int gate_vq_execution_layout = 0;
    std::uint64_t gate_vq_layouts = UINT64_MAX;
    int gate_has_nepq_residual = 0;
    int gate_npq_grouped_indices = 0;
    std::shared_ptr<const std::vector<int>> gate_nint_group_sizes;
    int down_family_mask = 0;
    int down_vq_profile_mask = 0;
    int down_vq_execution_layout = 0;
    std::uint64_t down_vq_layouts = UINT64_MAX;
    int down_has_nepq_residual = 0;
    int down_npq_grouped_indices = 0;
    std::shared_ptr<const std::vector<int>> down_nint_group_sizes;
    int shared_gate_group_size = 0;
    int shared_gate_groups = 0;
    int shared_up_group_size = 0;
    int shared_up_groups = 0;
    int shared_down_group_size = 0;
    int shared_down_groups = 0;
    int k_lanes = 16;
    int down_k_lanes = 16;
    int down_rows_per_simd = 1;
    int stage_one_workgroups = 0;
    int stage_two_workgroups = 0;
};

bool mfe_paired_nint_down(const MfeNintDecodeConfig& config) {
    return config.tokens == 1 && config.down_k_lanes == 16 &&
        config.down_rows_per_simd == 2 && config.hidden > config.intermediate;
}

std::string metal_activation_type(Dtype dtype) {
    if (dtype == mlx::core::float16) return "half";
    if (dtype == mlx::core::bfloat16) return "bfloat";
    if (dtype == mlx::core::float32) return "float";
    throw std::invalid_argument("unsupported fused MoE activation dtype");
}

std::string mfe_decode_kernel_name(
    const MfeNintDecodeConfig& config,
    int stage) {
    std::string name;
    name.reserve(256);
    name.append("mfq_mfe_nint_decode_stage");
    const auto integer = [&](int value) {
        char buffer[12];
        const auto result = std::to_chars(buffer, buffer + sizeof(buffer), value);
        name.append(buffer, result.ptr);
    };
    integer(stage);
    if (stage == 1 && config.folded_topk) {
        name.append("_gp");
        integer(kMfeFoldedGateSimdPairs);
    }
    for (const auto& field : std::array<std::pair<const char*, int>, 15>{{
             {"_r", config.routes}, {"_e", config.experts},
             {"_h", config.hidden}, {"_i", config.intermediate},
             {"_gf", config.gate_family_mask}, {"_gv", config.gate_vq_profile_mask},
             {"_df", config.down_family_mask}, {"_dv", config.down_vq_profile_mask},
             {"_ggs", config.shared_gate_group_size}, {"_gg", config.shared_gate_groups},
             {"_dgs", config.shared_down_group_size}, {"_dg", config.shared_down_groups},
             {"_kl", stage == 1 ? config.k_lanes : config.down_k_lanes}, {"_dr", config.down_rows_per_simd},
             {"_gt", static_cast<int>((stage == 2 && config.shared_router_group_size
                 ? mlx::core::float16 : config.shared_gate_dtype).val())}}}) {
        name.append(field.first);
        integer(field.second);
    }
    name.append("_gng");
    if (config.gate_nint_group_sizes)
        for (const auto value : *config.gate_nint_group_sizes) {
            name.push_back('_'); integer(value);
        }
    name.append("_dng");
    if (config.down_nint_group_sizes)
        for (const auto value : *config.down_nint_group_sizes) {
            name.push_back('_'); integer(value);
        }
    if (config.tokens > 1) name.append("_batch");
    name.append("_vls").append(std::to_string(stage == 1 ? config.gate_vq_layouts : config.down_vq_layouts));
    if (stage == 1 && config.route_groups) name.append("_rg1");
    if (config.shared_gate_is_logits) name.append("_gate_logits");
    if (stage == 1 && config.shared_router_group_size)
        name.append("_srg").append(std::to_string(config.shared_router_group_size));
    if (stage == 1 && config.shared_up_group_size != config.shared_gate_group_size)
        name.append("_ugs").append(std::to_string(config.shared_up_group_size))
            .append("_ug").append(std::to_string(config.shared_up_groups));
    if (config.packed_gate_up) name.append("_packed_gu");
    if (config.folded_topk) name.append("_folded_topk");
    if (stage == 2 && mfe_paired_nint_down(config)) name.append("_nint_pair");
    name.append("_rowq_auto");
    name.append(stage == 1 ? (config.gate_q_dispatch ? "1" : "0")
                           : (config.down_q_dispatch ? "1" : "0"));
    name.append(stage == 1 ? (config.shared_gate_q_dispatch ? "s1" : "s0")
                           : (config.shared_down_q_dispatch ? "s1" : "s0"));
    if (stage == 1 && config.route_groups) name.append("_pairq");
    return name;
}

void append_mfe_pool_arguments(std::ostringstream& source) {
    source
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
        << "device const uchar* mx_scales [[buffer(21)]],\n";
}

std::string_view mfe_decode_body() {
    static const std::string body = [] {
        const std::string_view source(kMoeSource);
        const auto begin = source.find("    uint descriptor_base = (\n");
        const auto end = source.find("    if (SPLIT_SWIGLU) {\n", begin);
        if (begin == std::string_view::npos || end == std::string_view::npos) {
            throw std::logic_error("MFE decoder body markers are missing");
        }
        std::string body(source.substr(begin, end - begin));
        const auto routes_begin = body.find("    threadgroup uint grouped_route_rows[6];\n");
        const auto routes_end = body.find("    if (\n        (uint(FAMILY_MASK) & 1u)", routes_begin);
        if (routes_begin == std::string::npos || routes_end == std::string::npos)
            throw std::logic_error("MFE route grouping markers are missing");
        body.erase(routes_begin, routes_end - routes_begin);
        return body;
    }();
    return body;
}

std::string make_mfe_reduce_body() {
    std::string body = R"METAL(
    constexpr uint K_LANES = uint(K_LANES_VALUE);
    constexpr uint ROWS_PER_SIMD = uint(ROWS_PER_SIMD_VALUE);
    constexpr uint LANE_GROUPS = 32u / K_LANES;
    constexpr uint ROWS_PER_PHYSICAL_SIMD = LANE_GROUPS * ROWS_PER_SIMD;
    constexpr uint MATRIX_ROWS = ROWS_PER_SIMD;
    constexpr bool SPLIT_SWIGLU = false;
    constexpr bool PACKED_SWIGLU = false;
    constexpr uint OUTPUT_TILES = (uint(OUT) + ROWS_PER_PHYSICAL_SIMD - 1u) / ROWS_PER_PHYSICAL_SIMD;
    const uint lane = thread_index_in_simdgroup;
    const uint k_lane = lane & (K_LANES - 1u);
    const uint lane_group = lane / K_LANES;
    const uint simd_group = simdgroup_index_in_threadgroup;
    const uint workgroup = threadgroup_position_in_grid.x;
    const uint output_tile = workgroup % OUTPUT_TILES;
    const uint token = workgroup / OUTPUT_TILES;
    const uint route = simd_group;
    const uint route_index = token * uint(ROUTES) + route;
    const uint projection = 0u;
    const uint output_base = output_tile * ROWS_PER_PHYSICAL_SIMD + lane_group * ROWS_PER_SIMD;
    threadgroup T partials[uint(ROUTES) * ROWS_PER_PHYSICAL_SIMD];
    const int logical_expert = expert_ids[route_index];
    int expert = logical_expert;
    if (PACKED_EXPERT_IDS != 0) expert = (logical_expert >> 8) - 1;
    else if (EXPERT_MAP_SIZE != 0) expert = logical_expert >= 0 && logical_expert < EXPERT_MAP_SIZE
        ? expert_map[logical_expert] : -1;
    if (expert >= 0 && expert < EXPERTS) {
)METAL";
    body += mfe_decode_body();
    body += R"METAL(
        for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {
            float total = accumulators[row];
            for (uint offset = K_LANES >> 1u; offset > 0u; offset >>= 1u)
                total += simd_shuffle_down(total, offset);
            if (k_lane == 0u) partials[route * ROWS_PER_PHYSICAL_SIMD
                + lane_group * ROWS_PER_SIMD + row] = T(total);
        }
    } else if (k_lane == 0u) {
        for (uint row = 0u; row < ROWS_PER_SIMD; ++row)
            partials[route * ROWS_PER_PHYSICAL_SIMD + lane_group * ROWS_PER_SIMD + row] = T(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0u && lane < ROWS_PER_PHYSICAL_SIMD) {
        float total = 0.0f;
        for (uint selected = 0u; selected < uint(ROUTES); ++selected)
            total += float(partials[selected * ROWS_PER_PHYSICAL_SIMD + lane])
                * route_weights[token * uint(ROUTES) + selected];
        const uint output = output_tile * ROWS_PER_PHYSICAL_SIMD + lane;
        if (output < uint(OUT)) y[token * uint(OUT) + output] = T(total);
    }
)METAL";
    return body;
}

std::string mfe_swiglu_tail() {
    const std::string_view source(kMoeSource);
    const auto begin = source.find("    if (SPLIT_SWIGLU) {\n");
    if (begin == std::string_view::npos) {
        throw std::logic_error("MFE SwiGLU tail marker is missing");
    }
    std::string tail(source.substr(begin));
    const std::string needle = "params[0]";
    std::size_t offset = 0;
    while ((offset = tail.find(needle, offset)) != std::string::npos) {
        tail.replace(offset, needle.size(), "SWIGLU_LIMIT");
        offset += std::string_view("SWIGLU_LIMIT").size();
    }
    return tail;
}

void append_mfe_decode_constants(
    std::ostringstream& source,
    const MfeNintDecodeConfig& config,
    bool gate_up) {
    source
        << "constexpr int TOKENS = 1;\n"
        << "constexpr int ROUTES = " << config.routes << ";\n"
        << "constexpr int EXPERTS = " << config.experts << ";\n"
        << "constexpr int OUT = "
        << (gate_up ? config.intermediate : config.hidden) << ";\n"
        << "constexpr int MATRIX_OUT = OUT" << (gate_up && config.packed_gate_up ? " * 2" : "") << ";\n"
        << "constexpr int PROJECTIONS = " << (gate_up && !config.packed_gate_up ? 2 : 1) << ";\n"
        << "constexpr int FUSED_SWIGLU = " << (gate_up ? 1 : 0) << ";\n"
        << "constexpr int K = "
        << (gate_up ? config.hidden : config.intermediate) << ";\n"
        << "constexpr int K_LANES_VALUE = "
        << (gate_up ? config.k_lanes : config.down_k_lanes) << ";\n"
        << "constexpr int ROWS_PER_SIMD_VALUE = "
        << (gate_up ? 1 : config.down_rows_per_simd) << ";\n"
        << "constexpr int DESCRIPTOR_SIZE = "
        << config.descriptor_size << ";\n"
        << "constexpr int VARIANT_STRIDE = ROUTES;\n"
        << "constexpr int SHARED_INPUT = " << (gate_up ? 1 : 0) << ";\n"
        << "constexpr int VQ_EXECUTION_LAYOUT = "
        << (gate_up ? config.gate_vq_execution_layout
                    : config.down_vq_execution_layout) << ";\n"
        << "constexpr int FAMILY_MASK = "
        << (gate_up ? config.gate_family_mask
                    : config.down_family_mask) << ";\n"
        << "constexpr int VQ_PROFILE_MASK = "
        << (gate_up ? config.gate_vq_profile_mask
                    : config.down_vq_profile_mask) << ";\n"
        << "constexpr int HAS_NEPQ_RESIDUAL = "
        << (gate_up ? config.gate_has_nepq_residual
                    : config.down_has_nepq_residual) << ";\n"
        << "constexpr int NPQ_GROUPED_INDICES = "
        << (gate_up ? config.gate_npq_grouped_indices
                    : config.down_npq_grouped_indices) << ";\n"
        << "constexpr int SORTED_ROUTES = 0;\n"
        << "constexpr int EXPERT_MAP_SIZE = 0;\n"
        << "constexpr int PACKED_EXPERT_IDS = 0;\n";
}

void append_mfe_shared_nint_decoder(
    std::ostringstream& source,
    const MfeNintDecodeConfig& config,
    bool gate_up) {
    const int group_size = gate_up
        ? config.shared_gate_group_size : config.shared_down_group_size;
    const int groups = gate_up
        ? config.shared_gate_groups : config.shared_down_groups;
    source
        << "  MfqNintRow metadata_rows[ROWS_PER_SIMD];\n"
        << "  uint q_widths[ROWS_PER_SIMD];\n"
        << "  uint q_row_byte_offsets[ROWS_PER_SIMD], q_row_bit_shifts[ROWS_PER_SIMD];\n"
        << "  float neuron_scales[ROWS_PER_SIMD], neuron_minimums[ROWS_PER_SIMD];\n"
        << "  for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {\n"
        << "    uint output = min(output_base + row, uint(OUT) - 1u);\n"
        << "    uint physical = " << (gate_up ? "projection * uint(OUT) + output" : "output") << ";\n"
        << "    MfqNintRow metadata = mfq_nint_row(shared_row_metadata, physical);\n"
        << "    metadata_rows[row] = metadata; q_widths[row] = metadata.q_bits;\n"
        << "    q_row_byte_offsets[row] = metadata.q_offset;\n"
        << "    q_row_bit_shifts[row] = metadata.q_shift;\n"
        << "    neuron_scales[row] = metadata.scale;\n"
        << "    neuron_minimums[row] = metadata.minimum;\n"
        << "  }\n"
        << "  device const uchar* nint_q = shared_q;\n"
        << "  device const uchar* nint_sub_scale = shared_sub_scale;\n"
        << "  device const uchar* nint_sub_min = shared_sub_min;\n"
        << "  uint x_offset = " << (gate_up ? "0u" : "SHARED_OFFSET") << ";\n";
    if (gate_up && config.shared_up_group_size != group_size) {
        source << "  uint q_offset = 0u, sub_offset = 0u;\n"
            << "  if (projection == 0u) { uint groups = " << groups << "u;\n"
            << "    MFQ_MFE_NINT_CALL_WITH(" << group_size << "u, " << config.shared_gate_q_dispatch << ");\n"
            << "  } else { uint groups = " << config.shared_up_groups << "u;\n"
            << "    MFQ_MFE_NINT_CALL_WITH(" << config.shared_up_group_size << "u, " << config.shared_gate_q_dispatch << ");\n"
            << "  }\n";
    } else {
        source << "  uint q_offset = 0u, sub_offset = 0u, groups = " << groups << "u;\n"
            << "  MFQ_MFE_NINT_CALL_WITH(" << group_size << "u, "
            << (gate_up ? config.shared_gate_q_dispatch : config.shared_down_q_dispatch) << ");\n";
    }
}

constexpr const char* kFoldedTopKHeader = R"METAL(
inline void mfq_topk_keys(device const half* logits, uint lane, thread uint* keys, uint experts) {
    for (uint local = 0u; local < experts / 32u; ++local) {
        uint expert = lane + local * 32u;
        float value = float(logits[expert]);
        value = isnan(value) ? -FLT_MAX : value;
        if (value == 0.0f) value = 0.0f;
        uint bits = as_type<uint>(value);
        uint ordered = bits & 0x80000000u ? ~bits : bits ^ 0x80000000u;
        keys[local] = ((ordered >> 13u) << 12u) | (4095u - expert);
    }
}
template <uint PER>
inline uint mfq_topk_step(thread const uint* keys, thread uint* head, thread uint& below, thread uint& left) {
    if (left == 0u) {
        uint a = 0u, b = 0u, c = 0u;
        for (uint local = 0u; local < PER; ++local) {
            uint v = keys[local] < below ? keys[local] : 0u;
            bool ga = v > a, gb = v > b, gc = v > c;
            c = gb ? b : (gc ? v : c);
            b = ga ? a : (gb ? v : b);
            a = ga ? v : a;
        }
        head[0] = a; head[1] = b; head[2] = c; left = 3u;
    }
    uint selected = simd_max(head[0]);
    if (head[0] == selected) {
        head[0] = head[1]; head[1] = head[2]; head[2] = 0u;
        below = selected; --left;
    }
    return 4095u - (selected & 4095u);
}
template <uint EXPERTS>
inline uint mfq_topk_nth(device const half* logits, uint lane, uint rank) {
    uint keys[EXPERTS / 32u], head[3] = {0u};
    mfq_topk_keys(logits, lane, keys, EXPERTS);
    uint below = 0xffffffffu, left = 0u, result = 0u;
    for (uint i = 0u; i <= rank; ++i)
        result = mfq_topk_step<EXPERTS / 32u>(keys, head, below, left);
    return result;
}
template <uint EXPERTS, uint ROUTES>
inline void mfq_topk_write(device const half* logits, uint lane, device float* routing) {
    uint keys[EXPERTS / 32u], head[3] = {0u}, ids[ROUTES];
    mfq_topk_keys(logits, lane, keys, EXPERTS);
    uint below = 0xffffffffu, left = 0u;
    for (uint i = 0u; i < ROUTES; ++i)
        ids[i] = mfq_topk_step<EXPERTS / 32u>(keys, head, below, left);
    if (lane == 0u) {
        float selected[ROUTES], denominator = 0.0f;
        for (uint i = 0u; i < ROUTES; ++i) {
            float value = float(logits[ids[i]]);
            selected[i] = isnan(value) ? -FLT_MAX : value;
        }
        float maximum = selected[0];
        for (uint i = 0u; i < ROUTES; ++i) {
            selected[i] = exp(selected[i] - maximum);
            denominator += selected[i];
        }
        for (uint i = 0u; i < ROUTES; ++i) {
            routing[i] = float(ids[i]);
            routing[ROUTES + i] = selected[i] / denominator;
        }
    }
}
)METAL";

std::string make_mfe_nint_gate_up_source(
    const MfeNintDecodeConfig& config,
    const std::string& kernel_name) {
    std::ostringstream source;
    source
        << "#include <metal_stdlib>\nusing namespace metal;\n"
        << "using T = " << metal_activation_type(config.dtype) << ";\n"
        << "using SharedGateT = "
        << (config.shared_router_group_size ? "uchar" : metal_activation_type(config.shared_gate_dtype)) << ";\n"
        << nint_group_specialization(config.gate_nint_group_sizes)
        << "#define MFQ_NINT_Q_DISPATCH " << config.gate_q_dispatch << "\n"
        << "#define MFQ_MOE_VQ_LAYOUTS " << config.gate_vq_layouts << "ul\n"
        << kMoeHeader
        << (config.folded_topk ? kFoldedTopKHeader : "");
    if (config.shared_router_group_size) {
        source << detail::nint_matmul_metal_header().substr(sizeof(detail::kNintMetadataSource) - 1)
            << "struct MfqSharedRouterInput { device const T* base; uint offset; T operator[](uint index) const { return base[offset + index]; } };\n"
            << "inline float4 mfq_nint_load_input4(MfqSharedRouterInput input, uint offset) { return mfq_nint_load_input4(input.base, input.offset + offset); }\n"
            << "template <bool SMALL> inline void mfq_shared_router(device const uchar* packed, device const T* values, device T* y, uint thread_index_in_simdgroup, uint input_offset) {\n"
            << "MfqSharedRouterInput x{values - input_offset, input_offset};\n"
            << "device const uint* header = reinterpret_cast<device const uint*>(packed);\n"
            << "device const uchar* q_packed = packed + header[0];\n"
            << "device const uchar* sub_scale = packed + header[1];\n"
            << "device const uchar* sub_min = packed + header[2];\n"
            << "device const uint* row_metadata = header + 4u;\n"
            << "device const int* expert_ids = reinterpret_cast<device const int*>(row_metadata);\n"
            << "device const int* expert_map = expert_ids;\n"
            << "uint simdgroup_index_in_threadgroup = 0u; uint3 threadgroup_position_in_grid = uint3(0u);\n";
        for (const auto& [name, value] : std::array<std::pair<const char*, int>, 14>{{
                 {"GS", config.shared_router_group_size},
                 {"NG", (config.hidden + config.shared_router_group_size - 1) / config.shared_router_group_size},
                 {"K", config.hidden}, {"OUT", 1}, {"M", 1}, {"TILE_M", 1}, {"OPS_PER_SIMD", 1},
                 {"ROUTED", 0}, {"ROUTES", 1}, {"SHARED_INPUT", 0}, {"EXPERT_MAP_SIZE", 1},
                 {"LOCAL_EXPERTS", 1}, {"OUT_PER_EXPERT", 1}, {"LOGICAL_OUT", 1}}})
            source << "#define " << name << " " << value << "\n";
        source << "#define SMALL_M SMALL\n#define MFQ_NINT_STORE_OUTPUT(index, value) y[index] = T(value)\n"
            << detail::nint_matmul_metal_body()
            << "\n#undef MFQ_NINT_STORE_OUTPUT\n#undef SMALL_M\n#undef GS\n#undef NG\n#undef K\n#undef OUT\n#undef M\n#undef TILE_M\n#undef OPS_PER_SIMD\n#undef ROUTED\n#undef ROUTES\n#undef SHARED_INPUT\n#undef EXPERT_MAP_SIZE\n#undef LOCAL_EXPERTS\n#undef OUT_PER_EXPERT\n#undef LOGICAL_OUT\n}\n";
    }
    source << "kernel void " << kernel_name << "(\n";
    append_mfe_pool_arguments(source);
    source
        << "device const T* x [[buffer(22)]],\n"
        << (config.folded_topk ? "device const half* router_logits [[buffer(23)]],\n"
            : "device const int* expert_ids [[buffer(23)]],\n")
        << "device const uchar* shared_q [[buffer(24)]],\n"
        << "device const uint* shared_row_metadata [[buffer(25)]],\n"
        << "device const uchar* shared_sub_scale [[buffer(26)]],\n"
        << "device const uchar* shared_sub_min [[buffer(27)]],\n"
        << "device const SharedGateT* shared_gate_weight [[buffer(28)]],\n"
        << (config.route_groups ? "device const int* route_groups [[buffer(29)]],\n" : "")
        << "device T* y [[buffer(" << (config.route_groups ? 30 : 29) << ")]],\n"
        << (config.folded_topk ? "device float* routing [[buffer(30)]],\n" : "")
        << "uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],\n"
        << "uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],\n"
        << "uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],\n"
        << "uint3 group_count [[threadgroups_per_grid]]) {\n";
    append_mfe_decode_constants(source, config, true);
    if (config.route_groups)
        source << "device const T* batch_x = x; device T* batch_y = y;\n";
    if (config.tokens > 1) {
        source
            << "uint token_row = threadgroup_position_in_grid.x;\n"
            << "x += token_row * uint(" << config.hidden << ");\n"
            << "expert_ids += token_row * uint(" << config.routes << ");\n"
            << "y += token_row * uint(" << (config.routes + 1) * config.intermediate + 1 << ");\n";
    }
    source
        << "constexpr float SWIGLU_LIMIT = 0.0f;\n"
        << "constexpr uint SIMD_GROUPS = 2u;\n"
        << "constexpr uint K_LANES = uint(K_LANES_VALUE);\n"
        << "constexpr uint LANE_GROUPS = 32u / K_LANES;\n"
        << "constexpr uint ROWS_PER_SIMD = uint(ROWS_PER_SIMD_VALUE);\n"
        << "constexpr bool SPLIT_SWIGLU = true;\n"
        << "constexpr bool PACKED_SWIGLU = false;\n"
        << "constexpr uint MATRIX_ROWS = ROWS_PER_SIMD;\n"
        << "constexpr uint ROWS_PER_PHYSICAL_SIMD = "
        << (config.folded_topk ? kMfeFoldedGateSimdPairs : 1)
        << "u * LANE_GROUPS * ROWS_PER_SIMD;\n"
        << "constexpr uint ROWS_PER_TG = ROWS_PER_PHYSICAL_SIMD;\n"
        << "constexpr uint OUTPUT_TILES = (uint(OUT) + ROWS_PER_TG - 1u) / ROWS_PER_TG;\n"
        << "constexpr uint ROUTED_SIZE = uint(ROUTES) * uint(OUT);\n"
        << "constexpr uint SHARED_OFFSET = ROUTED_SIZE;\n"
        << "constexpr uint GATE_OFFSET = SHARED_OFFSET + uint(OUT);\n"
        << "uint lane = thread_index_in_simdgroup;\n"
        << "uint k_lane = lane & (K_LANES - 1u);\n"
        << "uint lane_group = lane / K_LANES + (simdgroup_index_in_threadgroup >> 1u) * LANE_GROUPS;\n"
        << "uint simd_group = simdgroup_index_in_threadgroup & 1u;\n"
        << "uint workgroup = threadgroup_position_in_grid."
        << (config.tokens > 1 ? "y" : "x") << ";\n"
        << "if (workgroup == 0u) {\n"
        << (config.folded_topk ? "  if (simdgroup_index_in_threadgroup == 1u) mfq_topk_write<uint(EXPERTS), uint(ROUTES)>(router_logits, lane, routing);\n" : "")
        << "  if (simdgroup_index_in_threadgroup == 0u) {\n";
    if (config.shared_router_group_size) {
        if (config.tokens == 1)
            source << "    mfq_shared_router<true>(shared_gate_weight, x, y + GATE_OFFSET, lane, 0u);\n";
        else source << "    if (group_count.x <= 5u) mfq_shared_router<true>(shared_gate_weight, x, y + GATE_OFFSET, lane, token_row * uint(K));\n"
                    << "    else mfq_shared_router<false>(shared_gate_weight, x, y + GATE_OFFSET, lane, token_row * uint(K));\n";
    } else if (config.shared_gate_is_logits) {
        source << "    if (lane == 0u) y[GATE_OFFSET] = T(shared_gate_weight["
               << (config.tokens > 1 ? "token_row" : "0u") << "]);\n";
    } else {
        source
            << "    float gate = 0.0f;\n"
            << "    for (uint column = lane; column < uint(K); column += 32u)\n"
            << "      gate = fma(float(x[column]), float(shared_gate_weight[column]), gate);\n"
            << "    gate = simd_sum(gate);\n"
            << "    if (lane == 0u) y[GATE_OFFSET] = T(gate);\n";
    }
    source
        << "  }\n"
        << "  return;\n"
        << "}\n"
        << "if (workgroup <= OUTPUT_TILES) {\n"
        << "  uint output_tile = workgroup - 1u;\n"
        << "  uint output_base = output_tile * ROWS_PER_TG + lane_group * ROWS_PER_SIMD;\n"
        << "  float accumulators[ROWS_PER_SIMD] = {0.0f};\n"
        << "  uint projection = simd_group;\n";
    append_mfe_shared_nint_decoder(source, config, true);
    source
        << "  threadgroup T values[2u * ROWS_PER_PHYSICAL_SIMD];\n"
        << "  for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {\n"
        << "    float total = accumulators[row];\n"
        << "    for (uint offset = K_LANES >> 1; offset > 0u; offset >>= 1u)\n"
        << "      total += simd_shuffle_down(total, offset);\n"
        << "    if (k_lane == 0u) values[projection * ROWS_PER_PHYSICAL_SIMD + lane_group * ROWS_PER_SIMD + row] = T(total);\n"
        << "  }\n"
        << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        << "  if (simd_group == 0u && k_lane == 0u) {\n"
        << "    for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {\n"
        << "      uint local = lane_group * ROWS_PER_SIMD + row;\n"
        << "      uint output = output_tile * ROWS_PER_TG + local;\n"
        << "      if (output < uint(OUT)) {\n"
        << "        T gate = values[local]; T up = values[ROWS_PER_PHYSICAL_SIMD + local];\n"
        << "        y[SHARED_OFFSET + output] = T((float(gate) / (1.0f + exp(-float(gate)))) * float(up));\n"
        << "      }\n"
        << "    }\n"
        << "  }\n"
        << "  return;\n"
        << "}\n"
        << "workgroup -= 1u + OUTPUT_TILES;\n"
        << "uint output_tile = workgroup % OUTPUT_TILES;\n"
        << "uint projection_index = workgroup / OUTPUT_TILES;\n"
        << "uint projection = simd_group;\n"
        << "uint sorted_slot = projection_index;\n"
        << "uint route_index = sorted_slot;\n"
        << "uint route = route_index;\n"
        << "uint token = 0u;\n"
        << "uint output_base = output_tile * ROWS_PER_TG + lane_group * ROWS_PER_SIMD;\n";
    if (config.route_groups)
        source << "uint grouped_route = token_row * uint(ROUTES) + route;\n"
               << "uint2 route_group = uint2(*(device const packed_uint2*)(route_groups + grouped_route * 8u));\n"
               << "int logical_expert = int(route_group.x);\n";
    else if (config.folded_topk)
        source << "threadgroup uint selected_expert;\n"
               << "if (simdgroup_index_in_threadgroup == 0u) {\n"
               << "  uint selected = mfq_topk_nth<uint(EXPERTS)>(router_logits, lane, route);\n"
               << "  if (lane == 0u) selected_expert = selected;\n"
               << "}\n"
               << "threadgroup_barrier(mem_flags::mem_threadgroup);\n"
               << "int logical_expert = int(selected_expert);\n";
    else source << "int logical_expert = expert_ids[route];\n";
    source << "int expert = logical_expert;\n";
    if (config.packed_gate_up) {
        source
            << "output_base += projection * uint(OUT);\n"
            << "projection = 0u;\n";
    }
    if (config.route_groups) {
        if (!config.packed_gate_up)
            source
                << "if (route_group.y > 1u && route_group.y <= 6u) {\n"
                << "  uint descriptor = (uint(expert) * 2u + projection) * uint(DESCRIPTOR_SIZE);\n"
                << "  uint other = (uint(expert) * 2u + (projection ^ 1u)) * uint(DESCRIPTOR_SIZE);\n"
                << "  if (descriptors[descriptor] == 0 && descriptors[other] == 0\n"
                << "      && mfq_moe_nint_supported_gs(uint(descriptors[descriptor + 5u]))\n"
                << "      && mfq_moe_nint_supported_gs(uint(descriptors[other + 5u]))) {\n"
                << "    device const int* rows = route_groups + grouped_route * 8u + 2u;\n"
                << "    uint position = 0u;\n"
                << "    while (position < route_group.y && uint(rows[position]) != grouped_route) ++position;\n"
                << "    uint first = position & ~1u;\n"
                << "    if (first + 1u < route_group.y) {\n"
                << "      MfqMoeVqOutput<uint(ROUTES) * uint(OUT), (uint(ROUTES) + 1u) * uint(OUT) + 1u, T> targets{batch_y};\n"
                << "      threadgroup T grouped_values[4u * ROWS_PER_PHYSICAL_SIMD];\n"
                << "      mfq_moe_nint_pair<1u, 1u, uint(K), K_LANES, uint(OUT), uint(MATRIX_OUT),\n"
                << "          2u, uint(ROUTES), true, true, false>(batch_x, targets, descriptors + descriptor,\n"
                << "          nint_q, nint_sub_scale, nint_sub_min, rows + first, grouped_values,\n"
                << "          output_base, lane_group, k_lane, simd_group, uint(VARIANT_STRIDE), 0.0f, position == first);\n"
                << "      return;\n"
                << "    }\n"
                << "  }\n"
                << "}\n";
        source
            << "uint grouped_count = route_group.y;\n"
            << "if (grouped_count > 1u && grouped_count <= 6u) {\n"
            << "  uint grouped_descriptor = (uint(expert) * uint(PROJECTIONS) + projection) * uint(DESCRIPTOR_SIZE);\n"
            << "  bool reusable = descriptors[grouped_descriptor] == 1 && mfq_moe_reusable_vq_profile(uint(descriptors[grouped_descriptor + 28u]) & 255u);\n";
        if (!config.packed_gate_up)
            source
                << "  uint other = (uint(expert) * 2u + (projection ^ 1u)) * uint(DESCRIPTOR_SIZE);\n"
                << "  reusable = reusable && descriptors[other] == 1 && mfq_moe_reusable_vq_profile(uint(descriptors[other + 28u]) & 255u);\n";
        source
            << "  if (reusable) {\n"
            << "    device const int* rows = route_groups + grouped_route * 8u + 2u;\n"
            << "    if (uint(rows[0]) != grouped_route) return;\n"
            << "    MfqMoeVqOutput<uint(ROUTES) * uint(OUT), (uint(ROUTES) + 1u) * uint(OUT) + 1u, T> targets{batch_y};\n"
            << "    threadgroup T grouped_values[12u * ROWS_PER_PHYSICAL_SIMD];\n"
            << "    switch (grouped_count) {\n";
        for (int rows = 2; rows <= 6; ++rows)
            source << "      case " << rows << "u: { mfq_moe_reused_rows<" << rows
                << "u, 1u, 1u, uint(K), K_LANES, uint(OUT), uint(MATRIX_OUT), uint(PROJECTIONS), uint(ROUTES), uint(VQ_PROFILE_MASK), true, true, false, false>("
                << "batch_x, targets, descriptors + grouped_descriptor, vq_indices, vq_state, vq_aux, vq_anchors, vq_codebooks, vq_scales, vq_state_to_codebank, vq_parameters, mx_values, mx_scales, rows, grouped_values, output_base, lane_group, k_lane, simd_group, uint(VARIANT_STRIDE), 0.0f);\n"
                << "mfq_moe_finish_reused_swiglu<" << rows << "u, 1u, K_LANES, uint(OUT)>("
                << "targets, rows, grouped_values, output_base, lane_group, k_lane, simd_group, 0.0f); return; }\n";
        source << "    }\n  }\n}\n";
    }
    source << mfe_decode_body();
    if (config.packed_gate_up) source << "projection = simd_group;\n";
    source << mfe_swiglu_tail();
    source << "\n}\n";
    return source.str();
}

std::string make_mfe_nint_down_source(
    const MfeNintDecodeConfig& config,
    const std::string& kernel_name) {
    const bool paired_nint = mfe_paired_nint_down(config);
    std::ostringstream source;
    source
        << "#include <metal_stdlib>\nusing namespace metal;\n"
        << "using T = " << metal_activation_type(config.dtype) << ";\n"
        << nint_group_specialization(config.down_nint_group_sizes)
        << "#define MFQ_NINT_Q_DISPATCH " << config.down_q_dispatch << "\n"
        << "#define MFQ_MOE_VQ_LAYOUTS " << config.down_vq_layouts << "ul\n"
        << kMoeHeader
        << "kernel void " << kernel_name << "(\n";
    append_mfe_pool_arguments(source);
    source
        << "device const T* x [[buffer(22)]],\n"
        << "device const uchar* shared_q [[buffer(23)]],\n"
        << "device const uint* shared_row_metadata [[buffer(24)]],\n"
        << "device const uchar* shared_sub_scale [[buffer(25)]],\n"
        << "device const uchar* shared_sub_min [[buffer(26)]],\n"
        << (config.folded_topk ? "device const float* expert_ids [[buffer(27)]],\n"
            : "device const int* expert_ids [[buffer(27)]],\n")
        << "device const float* route_weights [[buffer(28)]],\n"
        << "device T* y [[buffer(29)]],\n"
        << "uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],\n"
        << "uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],\n"
        << "uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {\n";
    if (config.folded_topk) source << "route_weights += " << config.routes << "u;\n";
    append_mfe_decode_constants(source, config, false);
    if (config.tokens > 1) {
        source
            << "uint token_row = threadgroup_position_in_grid.x;\n"
            << "x += token_row * uint(" << (config.routes + 1) * config.intermediate + 1 << ");\n"
            << "expert_ids += token_row * uint(" << config.routes << ");\n"
            << "route_weights += token_row * uint(" << config.routes << ");\n"
            << "y += token_row * uint(" << config.hidden << ");\n";
    }
    source
        << "constexpr uint K_LANES = uint(K_LANES_VALUE);\n"
        << "constexpr uint LANE_GROUPS = 32u / K_LANES;\n"
        << "constexpr uint ROWS_PER_SIMD = uint(ROWS_PER_SIMD_VALUE);\n"
        << "constexpr bool SPLIT_SWIGLU = false;\n"
        << "constexpr bool PACKED_SWIGLU = false;\n"
        << "constexpr uint MATRIX_ROWS = ROWS_PER_SIMD;\n"
        << "constexpr uint ROWS_PER_PHYSICAL_SIMD = LANE_GROUPS * ROWS_PER_SIMD;\n"
        << "constexpr uint OUTPUT_TILES = (uint(OUT) + ROWS_PER_PHYSICAL_SIMD - 1u) / ROWS_PER_PHYSICAL_SIMD;\n"
        << "constexpr uint ROUTED_SIZE = uint(ROUTES) * uint(K);\n"
        << "constexpr uint SHARED_OFFSET = ROUTED_SIZE;\n"
        << "constexpr uint GATE_OFFSET = SHARED_OFFSET + uint(K);\n"
        << "uint lane = thread_index_in_simdgroup;\n"
        << "uint k_lane = lane & (K_LANES - 1u);\n"
        << "uint lane_group = lane / K_LANES;\n"
        << "uint route = simdgroup_index_in_threadgroup;\n"
        << "uint output_tile = threadgroup_position_in_grid."
        << (config.tokens > 1 ? "y" : "x") << ";\n"
        << "uint output_base = output_tile * ROWS_PER_PHYSICAL_SIMD + lane_group * ROWS_PER_SIMD;\n"
        << "threadgroup T partials[(ROUTES + 1) * ROWS_PER_PHYSICAL_SIMD];\n"
        << "if (route < uint(ROUTES)) {\n"
        << "  int expert = expert_ids[route];\n"
        << "  uint projection = 0u; uint token = 0u;\n"
        << "  if (expert >= 0 && expert < EXPERTS) {\n";
    if (paired_nint) source << R"METAL(
    device const int* selected = descriptors + uint(expert) * uint(DESCRIPTOR_SIZE);
    if (selected[0] == 0 && mfq_moe_nint_supported_gs(uint(selected[5]))) {
        uint narrow_lane = lane & 7u;
        uint narrow_group = lane >> 3u;
        uint output = min(output_tile * 4u + narrow_group, uint(OUT) - 1u);
        uint physical = uint(selected[1]) * uint(OUT) + output;
        device const uint* metadata = reinterpret_cast<device const uint*>(nint_q + uint(selected[11]));
        MfqNintRow metadata_rows[1] = {mfq_nint_row(metadata, physical)};
        uint q_widths[1] = {metadata_rows[0].q_bits};
        uint q_row_byte_offsets[1] = {metadata_rows[0].q_offset};
        uint q_row_bit_shifts[1] = {metadata_rows[0].q_shift};
        float neuron_scales[1] = {metadata_rows[0].scale};
        float neuron_minimums[1] = {metadata_rows[0].minimum};
        float low[1] = {0.0f}, high[1] = {0.0f};
        uint x_offset = route * uint(K), q_offset = uint(selected[7]), sub_offset = uint(selected[8]);
        uint group_size = uint(selected[5]), groups = uint(selected[6]);
        #define MFQ_PAIRED_DOWN_Q(GS, Q) \
            mfq_moe_nint_profile<GS, 1u, uint(K), 16u, true, true, 1u, Q>( \
                x, nint_q, nint_sub_scale, nint_sub_min, x_offset, q_offset, sub_offset, \
                metadata_rows, q_widths, q_row_byte_offsets, q_row_bit_shifts, neuron_scales, \
                neuron_minimums, low, groups, narrow_lane); \
            mfq_moe_nint_profile<GS, 1u, uint(K), 16u, true, true, 1u, Q>( \
                x, nint_q, nint_sub_scale, nint_sub_min, x_offset, q_offset, sub_offset, \
                metadata_rows, q_widths, q_row_byte_offsets, q_row_bit_shifts, neuron_scales, \
                neuron_minimums, high, groups, narrow_lane + 8u)
        #define MFQ_PAIRED_DOWN_GS(GS) \
            if constexpr (MFQ_NINT_Q_DISPATCH) { \
            switch (metadata_rows[0].q_bits) { \
                case 4u: { MFQ_PAIRED_DOWN_Q(GS, 4u); break; } \
                case 5u: { MFQ_PAIRED_DOWN_Q(GS, 5u); break; } \
                case 6u: { MFQ_PAIRED_DOWN_Q(GS, 6u); break; } \
                case 8u: { MFQ_PAIRED_DOWN_Q(GS, 8u); break; } \
                default: { MFQ_PAIRED_DOWN_Q(GS, 0u); break; } \
            } \
            } else { MFQ_PAIRED_DOWN_Q(GS, 0u); }
        MFQ_MFE_NINT_DISPATCH_WITH(MFQ_PAIRED_DOWN_GS) { }
        #undef MFQ_PAIRED_DOWN_GS
        #undef MFQ_PAIRED_DOWN_Q
        float total = low[0] + high[0];
        for (uint offset = 4u; offset > 0u; offset >>= 1u)
            total += simd_shuffle_down(total, offset);
        if (narrow_lane == 0u) partials[route * 4u + narrow_group] = T(total);
    } else {
)METAL";
    source << mfe_decode_body();
    source
        << "    for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {\n"
        << "      float total = accumulators[row];\n"
        << "      for (uint offset = K_LANES >> 1; offset > 0u; offset >>= 1u) total += simd_shuffle_down(total, offset);\n"
        << "      if (k_lane == 0u) partials[route * ROWS_PER_PHYSICAL_SIMD + lane_group * ROWS_PER_SIMD + row] = T(total);\n"
        << "    }\n";
    if (paired_nint) source << "    }\n";
    source
        << "  } else if (k_lane == 0u) {\n"
        << "    for (uint row = 0u; row < ROWS_PER_SIMD; ++row) partials[route * ROWS_PER_PHYSICAL_SIMD + lane_group * ROWS_PER_SIMD + row] = T(0.0f);\n"
        << "  }\n"
        << "} else {\n"
        << "  float accumulators[ROWS_PER_SIMD] = {0.0f};\n";
    append_mfe_shared_nint_decoder(source, config, false);
    source
        << "  for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {\n"
        << "    float total = accumulators[row];\n"
        << "    for (uint offset = K_LANES >> 1; offset > 0u; offset >>= 1u) total += simd_shuffle_down(total, offset);\n"
        << "    if (k_lane == 0u) partials[uint(ROUTES) * ROWS_PER_PHYSICAL_SIMD + lane_group * ROWS_PER_SIMD + row] = T(total);\n"
        << "  }\n"
        << "}\n"
        << "threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        << "if (route == 0u && k_lane == 0u) {\n"
        << "  float gate = float(x[GATE_OFFSET]); float shared_scale = 1.0f / (1.0f + exp(-gate));\n"
        << "  for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {\n"
        << "    uint local = lane_group * ROWS_PER_SIMD + row; uint output = output_tile * ROWS_PER_PHYSICAL_SIMD + local;\n"
        << "    if (output < uint(OUT)) {\n"
        << "      float total = 0.0f;\n"
        << "      for (uint selected = 0u; selected < uint(ROUTES); ++selected)\n"
        << "        total = fma(float(partials[selected * ROWS_PER_PHYSICAL_SIMD + local]), route_weights[selected], total);\n"
        << "      total += shared_scale * float(partials[uint(ROUTES) * ROWS_PER_PHYSICAL_SIMD + local]);\n"
        << "      y[output] = T(total);\n"
        << "    }\n"
        << "  }\n"
        << "}\n}\n";
    return source.str();
}

class MfeNintDecodePrimitive final : public mlx::core::Primitive, public MlxPreparableKernel {
public:
    MfeNintDecodePrimitive(
        mlx::core::Stream stream,
        MfeNintDecodeConfig config,
        int stage)
        : Primitive(stream),
          config_(std::move(config)),
          stage_(stage),
          kernel_name_(mfe_decode_kernel_name(config_, stage_)) {}

    std::string preparation_key() const override { return kernel_name_; }
    void prepare_gpu() override { (void)prepared_kernel(); }

    MTL::ComputePipelineState* prepared_kernel() {
        auto& selected_stream = stream();
        auto& device = kernel_device(selected_stream.device);
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        auto* library = device.get_library(
            kernel_name_,
            options,
            [config = config_, name = kernel_name_, stage = stage_] {
                return stage == 1
                    ? make_mfe_nint_gate_up_source(config, name)
                    : make_mfe_nint_down_source(config, name);
            });
        return device.get_kernel(kernel_name_, library);
    }

    void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
        throw std::runtime_error(
            "two-stage MFE/NINT decode has no CPU path");
    }

    void eval_gpu(
        const std::vector<array>& inputs,
        std::vector<array>& outputs) override {
        const int input_count = 29 + static_cast<int>(stage_ == 1 && config_.route_groups);
        const bool routing_output = stage_ == 1 && config_.folded_topk;
        if (inputs.size() != static_cast<std::size_t>(input_count) ||
            outputs.size() != (routing_output ? 2u : 1u)) {
            throw std::logic_error(
                "two-stage MFE/NINT decode input count mismatch");
        }
        for (auto& output : outputs)
            output.set_data(mlx::core::allocator::malloc(output.nbytes()));
        auto& selected_stream = stream();
        auto* kernel = prepared_kernel();
        auto& encoder = mlx::core::metal::get_command_encoder(
            selected_stream);
        encoder.set_compute_pipeline_state(kernel);
        for (int index = 0; index < input_count; ++index) {
            encoder.set_input_array(
                inputs[static_cast<std::size_t>(index)], index);
        }
        encoder.set_output_array(outputs[0], input_count);
        if (routing_output) encoder.set_output_array(outputs[1], input_count + 1);
        const int workgroups = stage_ == 1
            ? config_.stage_one_workgroups : config_.stage_two_workgroups;
        encoder.dispatch_threadgroups(
            MTL::Size(
                config_.tokens > 1 ? config_.tokens : workgroups,
                config_.tokens > 1 ? workgroups : 1,
                1),
            MTL::Size(
                stage_ == 1
                    ? 64 * (config_.folded_topk ? kMfeFoldedGateSimdPairs : 1)
                    : (config_.routes + 1) * 32,
                1,
                1));
    }

    const char* name() const override {
        return "MfeNintDecodePrimitive";
    }

    bool is_equivalent(const mlx::core::Primitive& other) const override {
        const auto* primitive =
            dynamic_cast<const MfeNintDecodePrimitive*>(&other);
        return primitive != nullptr &&
            primitive->kernel_name_ == kernel_name_ &&
            primitive->config_.tokens == config_.tokens;
    }

    std::vector<Shape> output_shapes(
        const std::vector<array>&) override {
        std::vector<Shape> shapes{stage_ == 1
            ? config_.stage_one_shape
            : config_.output_shape};
        if (stage_ == 1 && config_.folded_topk)
            shapes.emplace_back(Shape{2, config_.routes});
        return shapes;
    }

private:
    MfeNintDecodeConfig config_;
    int stage_ = 0;
    std::string kernel_name_;
};

std::vector<array> mfe_nint_decode_dispatch(
    std::vector<array> inputs,
    const MfeNintDecodeConfig& config,
    int stage) {
    auto stream = mlx::core::default_stream(
        mlx::core::default_device());
    if (stream.device != mlx::core::Device::gpu) {
        throw std::invalid_argument(
            "two-stage MFE/NINT decode requires the Metal device");
    }
    std::vector<Shape> shapes{stage == 1 ? config.stage_one_shape : config.output_shape};
    std::vector<Dtype> dtypes{config.dtype};
    if (stage == 1 && config.folded_topk) {
        shapes.emplace_back(Shape{2, config.routes});
        dtypes.push_back(mlx::core::float32);
    }
    return array::make_arrays(
        shapes, dtypes,
        std::make_shared<MfeNintDecodePrimitive>(
            stream, config, stage),
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
    constexpr uint K_LANES = 16u;
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
    : public mlx::core::UnaryPrimitive, public MlxPreparableKernel {
public:
    Mxfp4DecodeReducePrimitive(
        mlx::core::Stream stream,
        Mxfp4DecodeReduceConfig config)
        : UnaryPrimitive(stream),
          config_(std::move(config)),
          kernel_name_(mxfp4_decode_reduce_kernel_name(config_)) {}

    std::string preparation_key() const override { return kernel_name_; }
    void prepare_gpu() override { (void)prepared_kernel(); }

    MTL::ComputePipelineState* prepared_kernel() {
        auto& selected_stream = stream();
        auto& device = kernel_device(
            selected_stream.device);
        CompileOptions compile_options;
        compile_options.math_mode = MathMode::Fast;
        auto* library = device.get_library(
            kernel_name_,
            compile_options,
            [config = config_, name = kernel_name_] {
                return make_mxfp4_decode_reduce_source(config, name);
            });
        return device.get_kernel(kernel_name_, library);
    }

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
        auto* kernel = prepared_kernel();
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
    constexpr uint K_LANES = 16u;
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
            const uint column = group * 32u + k_lane * 2u;
            const vec<T, 2> raw_activations = *reinterpret_cast<
                device const vec<T, 2>*>(x + token * K + column);
            const float2 activations = float2(raw_activations);
            const ulong gate_packed_offset = ulong(gate_value_offset)
                + gate_pool_output * (ulong(K) >> 1u)
                + (ulong(column) >> 1u);
            const ulong up_packed_offset = ulong(up_value_offset)
                + up_pool_output * (ulong(K) >> 1u)
                + (ulong(column) >> 1u);
            const uchar gate_codes = gate_values[gate_packed_offset];
            const uchar up_codes = up_values[up_packed_offset];
            const float gate_scale = decode_e8m0(gate_scales[
                ulong(gate_scale_offset)
                    + gate_pool_output * ulong(groups) + ulong(group)]);
            const float up_scale = decode_e8m0(up_scales[
                ulong(up_scale_offset)
                    + up_pool_output * ulong(groups) + ulong(group)]);
            const float2 gate_weights = gate_scale * float2(
                MXFP4_LUT[uint(gate_codes & 15u)],
                MXFP4_LUT[uint(gate_codes >> 4u)]);
            const float2 up_weights = up_scale * float2(
                MXFP4_LUT[uint(up_codes & 15u)],
                MXFP4_LUT[uint(up_codes >> 4u)]);
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
    : public mlx::core::UnaryPrimitive, public MlxPreparableKernel {
public:
    Mxfp4PairSwiGluPrimitive(
        mlx::core::Stream stream,
        Mxfp4PairSwiGluConfig config)
        : UnaryPrimitive(stream),
          config_(std::move(config)),
          kernel_name_(mxfp4_pair_swiglu_kernel_name(config_)) {}

    std::string preparation_key() const override { return kernel_name_; }
    void prepare_gpu() override { (void)prepared_kernel(); }

    MTL::ComputePipelineState* prepared_kernel() {
        auto& selected_stream = stream();
        auto& device = kernel_device(
            selected_stream.device);
        CompileOptions compile_options;
        compile_options.math_mode = MathMode::Fast;
        auto* library = device.get_library(
            kernel_name_,
            compile_options,
            [config = config_, name = kernel_name_] {
                return make_mxfp4_pair_swiglu_source(config, name);
            });
        return device.get_kernel(kernel_name_, library);
    }

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
        auto* kernel = prepared_kernel();
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
    : public mlx::core::UnaryPrimitive, public MlxPreparableKernel {
public:
    Mxfp4BlocksPrimitive(
        mlx::core::Stream stream,
        Mxfp4BlocksConfig config)
        : UnaryPrimitive(stream),
          config_(std::move(config)) {}

    std::string preparation_key() const override { return config_.paired ? "mfq_mxfp4_blocks_pair" : "mfq_mxfp4_blocks_single"; }
    void prepare_gpu() override { (void)prepared_kernel(); }

    MTL::ComputePipelineState* prepared_kernel() {
        auto& device = kernel_device(stream().device);
        CompileOptions compile_options;
        compile_options.math_mode = MathMode::Fast;
        auto* library = device.get_library(
            "mfq_dsv4_mxfp4_blocks_v5",
            compile_options,
            dsv4_mxfp4_blocks_source);
        const char* kernel_name = config_.paired
            ? "mfq_dsv4_mxfp4_pair_concat_f16_bm32_bn32_bk32"
            : "mfq_dsv4_mxfp4_single_f16_bm32_bn32_bk32";
        return device.get_kernel(kernel_name, library);
    }

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
        auto& device = kernel_device(
            selected_stream.device);
        auto* kernel = prepared_kernel();
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
    : public mlx::core::UnaryPrimitive, public MlxPreparableKernel {
public:
    GroupedMmqPrimitive(
        mlx::core::Stream stream,
        GroupedMmqConfig config)
        : UnaryPrimitive(stream),
          config_(std::move(config)) {
        if (config_.use_nax && !config_.direct_nax && config_.projections == 1
            && config_.fused_swiglu == 0 && config_.cohort_populations
            && config_.cohort_populations->size() > 1) {
            int mask = 0;
            int experts = 0;
            for (const auto& [family, group, count] : *config_.cohort_populations) {
                mask |= family;
                experts += count;
            }
            compact_blocks_ = mask == config_.family_mask && experts == config_.experts;
        }
    }

    std::string preparation_key() const override {
        std::string key = "grouped_mmq";
        for (const int value : {config_.block_rows, config_.tile_columns, config_.projections,
                config_.input_width, config_.output_width, config_.matrix_output_width,
                config_.fused_swiglu, config_.has_nepq_residual, config_.family_mask,
                config_.vq_profile_mask, static_cast<int>(config_.use_nax), static_cast<int>(config_.direct_nax),
                static_cast<int>(config_.indirect_pages)})
            key += "_" + std::to_string(value);
        if (config_.nint_group_sizes)
            for (const int group : *config_.nint_group_sizes) key += "_gs" + std::to_string(group);
        if (compact_blocks()) key += "_compact_blocks_p"
            + std::to_string(config_.cohort_populations->size());
        return key;
    }
    void prepare_gpu() override {
        (void)prepared_kernels();
        if (compact_blocks()) (void)prepared_partition();
    }

    bool compact_blocks() const {
        return compact_blocks_;
    }

    MTL::ComputePipelineState* prepared_partition() {
        auto& device = kernel_device(stream().device);
        const int passes = static_cast<int>(config_.cohort_populations->size());
        const std::string name = "mfq_grouped_mfe_partition_v1_p" + std::to_string(passes);
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        auto* library = device.get_library(name, options, [passes] {
            return std::string("#include <metal_stdlib>\nusing namespace metal;\n#define PASSES ")
                + std::to_string(passes) + R"METAL(
kernel void mfq_grouped_mfe_partition(
    device const int* descriptors [[buffer(0)]],
    device const int* source [[buffer(1)]],
    device const int* count [[buffer(2)]],
    device int* destination [[buffer(3)]],
    device int* counts [[buffer(4)]],
    constant int4& params [[buffer(5)]],
    constant int2* keys [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup atomic_int local_counts[PASSES];
    for (int pass = int(tid); pass < PASSES; pass += 256)
        atomic_store_explicit(local_counts + pass, 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int block = int(tid); block < count[0]; block += 256) {
        const int expert = source[block * 3 + 1];
        if (expert < 0 || expert >= params.y || source[block * 3 + 2] <= 0) continue;
        device const int* descriptor = descriptors + expert * params.z;
        const int family = descriptor[0];
        for (int pass = 0; pass < PASSES; ++pass) {
            if ((keys[pass].x & (1 << family)) == 0
                || (keys[pass].y != 0 && descriptor[5] != keys[pass].y)) continue;
            const int slot = atomic_fetch_add_explicit(local_counts + pass, 1, memory_order_relaxed);
            const int offset = (pass * params.x + slot) * 3;
            destination[offset] = source[block * 3];
            destination[offset + 1] = expert;
            destination[offset + 2] = source[block * 3 + 2];
            break;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int pass = int(tid); pass < PASSES; pass += 256)
        counts[pass] = atomic_load_explicit(local_counts + pass, memory_order_relaxed);
}
)METAL";
        });
        return device.get_kernel("mfq_grouped_mfe_partition", library);
    }

    std::vector<std::tuple<MTL::ComputePipelineState*, int, int, int, int>> prepared_kernels() {
        auto& selected_stream = stream();
        auto& device = kernel_device(
            selected_stream.device);
        CompileOptions compile_options;
        compile_options.math_mode = MathMode::Fast;
        const bool vector_vq =
            (static_cast<std::uint32_t>(config_.vq_profile_mask)
                & kGroupedVqVectorProfileMask) != 0;
        const bool vector_jsc_extended =
            (static_cast<std::uint32_t>(config_.vq_profile_mask)
                & kGroupedJscExtendedProfileMask) != 0;
        const bool aligned_input = config_.tile_columns <= 64
            && !config_.direct_nax && config_.input_width % 32 == 0;
        const auto get_library = [&](int family_mask, int group_size, int simd_rows) {
            const bool uses_vq = (family_mask & (1 << kFamilyVq)) != 0;
            const bool selected_vector_vq = vector_vq && uses_vq;
            const bool selected_extended = vector_jsc_extended && uses_vq;
            std::string library_name;
            if (config_.use_nax) {
                library_name = selected_vector_vq
                    ? (selected_extended
                        ? "mfq_grouped_mfe_nax_v34_legacy_vq_jsc_extended"
                        : "mfq_grouped_mfe_nax_v34_legacy_vq_vector")
                    : (selected_extended
                        ? "mfq_grouped_mfe_nax_v34_jsc_extended"
                        : "mfq_grouped_mfe_nax_v34");
            } else {
                library_name = selected_vector_vq
                    ? (selected_extended
                        ? "mfq_grouped_mmq_v28_legacy_vq_jsc_extended"
                        : "mfq_grouped_mmq_v28_legacy_vq_vector")
                    : (selected_extended
                        ? "mfq_grouped_mmq_v28_jsc_extended"
                        : "mfq_grouped_mmq_v28");
            }
            library_name += "_fm" + std::to_string(family_mask)
                + "_gs" + std::to_string(group_size);
            if (compact_blocks()) library_name += "_compact";
            if (config_.indirect_pages) library_name += "_pages";
            if (!config_.use_nax)
                library_name += "_bm" + std::to_string(config_.block_rows);
            if (config_.use_nax) {
                library_name += "_bm" + std::to_string(config_.block_rows)
                    + "_bn" + std::to_string(config_.tile_columns)
                    + "_ak" + std::to_string(aligned_input)
                    + "_sm" + std::to_string(simd_rows)
                    + "_sw" + std::to_string(config_.fused_swiglu)
                    + "_di" + std::to_string(config_.direct_nax)
                    + "_iw" + std::to_string(config_.input_width)
                    + "_ow" + std::to_string(config_.output_width)
                    + "_mw" + std::to_string(config_.matrix_output_width);
            }
            return device.get_library(
                library_name,
                compile_options,
                [
                    use_nax = config_.use_nax,
                    indirect_pages = config_.indirect_pages,
                    vector_vq = selected_vector_vq,
                    vector_jsc_extended = selected_extended,
                    family_mask,
                    group_size,
                    block_rows = config_.block_rows,
                    tile_columns = config_.tile_columns,
                    aligned_input,
                    simd_rows,
                    fused = config_.fused_swiglu,
                    direct = config_.direct_nax,
                    input_width = config_.input_width,
                    output_width = config_.output_width,
                    matrix_output_width = config_.matrix_output_width
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
                    source += "#define MFQ_GROUPED_FAMILY_MASK "
                        + std::to_string(family_mask) + "\n";
                    source += "#define MFQ_GROUPED_NINT_GROUP_SIZE "
                        + std::to_string(group_size) + "\n";
                    source += "#define MFQ_GROUPED_MMQ_BM "
                        + std::to_string(block_rows) + "\n";
                    if (use_nax) {
                        source += "#define MFQ_GROUPED_NAX_BM "
                            + std::to_string(block_rows) + "\n";
                        source += "#define MFQ_GROUPED_NAX_BN "
                            + std::to_string(tile_columns) + "\n";
                        source += "#define MFQ_GROUPED_NAX_ALIGNED_INPUT "
                            + std::to_string(aligned_input) + "\n";
                        source += "#define MFQ_GROUPED_NAX_SIMD_ROWS "
                            + std::to_string(simd_rows) + "\n";
                        source += "#define MFQ_GROUPED_NAX_INPUT_WIDTH "
                            + std::to_string(input_width) + "\n";
                        source += "#define MFQ_GROUPED_NAX_OUTPUT_WIDTH "
                            + std::to_string(output_width) + "\n";
                        source += "#define MFQ_GROUPED_NAX_MATRIX_OUTPUT_WIDTH "
                            + std::to_string(matrix_output_width) + "\n";
                        const bool aligned_nint = family_mask == (1 << kFamilyNint)
                            && group_size == 28 && tile_columns <= 64
                            && !fused && !direct;
                        const bool balanced_loader = tile_columns <= 64
                            && block_rows == 128 && !fused && !direct
                            && ((family_mask == (1 << kFamilyNint)
                                    && group_size == 24 && simd_rows == 16)
                                || (family_mask == (1 << kFamilyVq)
                                    && simd_rows == 16));
                        source += "#define MFQ_GROUPED_NAX_BK "
                            + std::to_string(aligned_nint ? 224
                                : balanced_loader ? 192 : 96) + "\n";
                        const int x_stride = aligned_nint ? 228
                            : balanced_loader ? 196 : tile_columns == 128
                            || (tile_columns <= 64 && block_rows != 32
                                && block_rows != 64) ? 100 : 104;
                        const int w_stride = tile_columns <= 64 ? x_stride : 100;
                        source += "#define MFQ_GROUPED_NAX_X_STRIDE "
                            + std::to_string(x_stride) + "\n";
                        source += "#define MFQ_GROUPED_NAX_W_STRIDE "
                            + std::to_string(w_stride) + "\n";
                        source += "#define MFQ_GROUPED_NAX_FUSED "
                            + std::to_string(fused != 0) + "\n";
                        source += "#define MFQ_GROUPED_NAX_DIRECT "
                            + std::to_string(direct) + "\n";
                    }
                    source += "using namespace metal;\n";
                    source += "using bfloat16_t = bfloat;\n";
                    if (indirect_pages) {
                        source += "#define MFQ_GROUPED_INDIRECT_PAGES 1\nstruct MfqExpertStreams {\n";
                        for (const auto& [name, type] : kExpertStreams)
                            source += "device const " + std::string(type) + "* " + name + ";\n";
                        source += "};\n";
                    }
                    source += detail::kNintMetadataSource;
                    source += use_nax
                        ? detail::kSteelNaxSource
                        : detail::kSteelMmaSource;
                    source += detail::kMfePrefillSource;
                    return source;
                });
        };
        const char* kernel_name = config_.use_nax
            ? "mfq_grouped_mfe_nax_f16_specialized"
            : (config_.fused_swiglu != 0
                ? (config_.has_nepq_residual != 0
                    ? "mfq_grouped_mmq_swiglu_f16_specialized_nr"
                    : "mfq_grouped_mmq_swiglu_f16_specialized")
                : (config_.has_nepq_residual != 0
                    ? "mfq_grouped_mmq_f16_specialized_nr"
                    : "mfq_grouped_mmq_f16_specialized"));
        std::vector<std::tuple<MTL::ComputePipelineState*, int, int, int, int>> kernels;
        std::vector<std::pair<int, int>> passes;
        if (!config_.direct_nax
            && config_.projections == 1 && config_.fused_swiglu == 0) {
            for (int family = 0; family < 7; ++family) {
                const int mask = 1 << family;
                if ((config_.family_mask & mask) == 0) continue;
                if (family == kFamilyNint && config_.nint_group_sizes
                    && !config_.nint_group_sizes->empty()) {
                    for (const auto group_size : *config_.nint_group_sizes) {
                        passes.emplace_back(mask, group_size);
                    }
                } else {
                    passes.emplace_back(mask, 0);
                }
            }
        } else {
            const int group_size = !config_.use_nax && config_.nint_group_sizes
                && config_.nint_group_sizes->size() == 1 ? config_.nint_group_sizes->front() : 0;
            passes.emplace_back(config_.family_mask, group_size);
        }
        for (const auto& [family_mask, group_size] : passes) {
            const int simd_rows = config_.use_nax
                    && (family_mask == (1 << kFamilyVq)
                        || (family_mask == (1 << kFamilyNint) && group_size == 24))
                    && config_.block_rows == 128 && config_.tile_columns == 32
                    && config_.output_width > config_.input_width
                    && aligned_input && !config_.direct_nax
                    && config_.fused_swiglu == 0
                ? 32 : 16;
            auto* library = get_library(family_mask, group_size, simd_rows);
            auto* kernel = device.get_kernel(kernel_name, library);
            const int tile_columns = config_.use_nax ? config_.tile_columns
                : config_.block_rows >= 80
                    && !(family_mask == (1 << kFamilyVq)
                        || (family_mask == (1 << kFamilyNint)
                            && (group_size == 24 || group_size == 28))) ? 32 : 64;
            kernels.emplace_back(kernel, simd_rows, tile_columns, family_mask, group_size);
            if (!config_.use_nax && config_.block_rows >= 64) {
                const auto mid_name = std::string(kernel_name) + "_mid";
                kernels.emplace_back(device.get_kernel(mid_name, library), simd_rows, 64, family_mask, group_size);
                if (config_.block_rows > 80) {
                    const auto mid80_name = std::string(kernel_name) + "_mid80";
                    kernels.emplace_back(device.get_kernel(mid80_name, library), simd_rows, tile_columns, family_mask, group_size);
                }
                const auto tail_name = std::string(kernel_name) + "_tail";
                kernels.emplace_back(device.get_kernel(tail_name, library), simd_rows, 64, family_mask, group_size);
            }
        }
        return kernels;
    }

    void eval_cpu(
        const std::vector<array>&,
        array&) override {
        throw std::runtime_error(
            "grouped MMQ primitive has no CPU path");
    }

    void eval_gpu(
        const std::vector<array>& inputs,
        array& output) override {
        if (inputs.size() < (config_.indirect_pages ? 29u : 28u)
            || (!config_.indirect_pages && inputs.size() != 28)) {
            throw std::logic_error(
                "grouped MMQ primitive input count mismatch");
        }
        output.set_data(
            mlx::core::allocator::malloc(output.nbytes()));
        auto& selected_stream = stream();
        auto& encoder =
            mlx::core::metal::get_command_encoder(
                selected_stream);
        if (config_.indirect_pages) {
            for (std::size_t index = 29; index < inputs.size(); ++index)
                encoder.set_input_array(inputs[index], 29);
            retain_indirect_resources(encoder, std::span(inputs).subspan(29));
            encoder.set_input_array(inputs[28], 29);
        }
        const auto kernels = prepared_kernels();
        std::optional<array> compact_meta;
        std::optional<array> compact_count;
        if (compact_blocks()) {
            const int passes = static_cast<int>(config_.cohort_populations->size());
            compact_meta.emplace(Shape{passes, config_.max_blocks, 3}, mlx::core::int32, nullptr, std::vector<array>{});
            compact_count.emplace(Shape{passes}, mlx::core::int32, nullptr, std::vector<array>{});
            for (auto* temporary : {&*compact_meta, &*compact_count}) {
                temporary->set_data(mlx::core::allocator::malloc(temporary->nbytes()));
                encoder.add_temporary(*temporary);
            }
            std::vector<std::array<int, 2>> keys;
            for (const auto& [mask, group, population] : *config_.cohort_populations)
                keys.push_back({mask, group});
            encoder.set_compute_pipeline_state(prepared_partition());
            encoder.set_input_array(inputs[0], 0);
            encoder.set_input_array(inputs[26], 1);
            encoder.set_input_array(inputs[27], 2);
            encoder.set_output_array(*compact_meta, 3);
            encoder.set_output_array(*compact_count, 4);
            encoder.set_bytes(std::array<int, 4>{config_.max_blocks, config_.experts,
                config_.descriptor_size, 0}, 5);
            encoder.set_vector_bytes(keys, 6);
            encoder.dispatch_threadgroups(MTL::Size(1, 1, 1), MTL::Size(256, 1, 1));
        }
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
        const int rows_x = config_.use_nax && config_.block_rows == 128
            ? 1 : std::min(32, config_.max_blocks);
        for (const auto& [kernel, simd_rows, tile_columns, family_mask, group_size] : kernels) {
            const int columns = (config_.output_width + tile_columns - 1) / tile_columns;
            int pass_blocks = config_.max_blocks;
            if (compact_blocks()) {
                const auto& populations = *config_.cohort_populations;
                const auto found = std::find_if(populations.begin(), populations.end(),
                    [family = family_mask, group = group_size](const auto& population) {
                        return std::get<0>(population) == family && std::get<1>(population) == group;
                    });
                if (found == populations.end()) throw std::logic_error("grouped MFE cohort is missing");
                const auto pass = static_cast<std::size_t>(found - populations.begin());
                pass_blocks = std::min(config_.max_blocks,
                    (config_.route_count + config_.block_rows - 1) / config_.block_rows + std::get<2>(*found));
                encoder.set_input_array(*compact_meta, 18,
                    pass * static_cast<std::size_t>(config_.max_blocks) * 3 * sizeof(int));
                encoder.set_input_array(*compact_count, 19, pass * sizeof(int));
            }
            encoder.set_compute_pipeline_state(kernel);
            encoder.dispatch_threadgroups(
                config_.use_nax ? MTL::Size(compact_blocks() ? 1 : rows_x,
                    std::size_t(compact_blocks() ? pass_blocks : (config_.max_blocks + rows_x - 1) / rows_x) * columns, 1)
                    : MTL::Size(columns, config_.max_blocks, 1),
                MTL::Size(
                    config_.use_nax
                        ? config_.block_rows * config_.tile_columns / simd_rows
                        : 256,
                    1,
                    1));
        }
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
            && primitive->config_.nint_group_sizes == config_.nint_group_sizes
            && primitive->config_.cohort_populations == config_.cohort_populations
            && primitive->config_.use_nax == config_.use_nax
            && primitive->config_.direct_nax == config_.direct_nax
            && primitive->config_.indirect_pages == config_.indirect_pages
            && primitive->config_.swiglu_limit == config_.swiglu_limit;
    }

    std::vector<Shape> output_shapes(
        const std::vector<array>&) override {
        return {config_.output_shape};
    }

private:
    GroupedMmqConfig config_;
    bool compact_blocks_ = false;
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
        std::string(
            "const int TOKENS = expert_ids_shape[0];\n"
            "const int VARIANT_STRIDE = SHARED_INPUT != 0 ? TOKENS : TOKENS * ROUTES;\n") + kMoeSource,
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
    PackedStream nint_q;
    PackedStream nint_sub_scale;
    PackedStream nint_sub_min;
    PackedStream nint_anchor_scale;
    PackedStream nint_anchor_min;
    PackedStream q8_q;
    PackedStream q8_scales;
    PackedStream vq_indices;
    PackedStream vq_state;
    PackedStream vq_aux;
    PackedStream vq_anchors;
    PackedStream vq_codebooks;
    PackedStream vq_scales;
    PackedStream vq_state_to_codebank;
    PackedStream vq_banks;
    PackedStream vq_parameters;
    PackedStream vq_residual_codebooks;
    PackedStream vq_residual_first;
    PackedStream vq_residual_second;
    PackedStream mx_values;
    PackedStream mx_scales;

    void retain_source(const array& source) {
        for (auto* stream : {&nint_q, &nint_sub_scale, &nint_sub_min,
            &nint_anchor_scale, &nint_anchor_min, &q8_q, &q8_scales, &vq_indices,
            &vq_state, &vq_aux, &vq_anchors, &vq_codebooks, &vq_scales,
            &vq_state_to_codebank, &vq_banks, &vq_parameters, &vq_residual_codebooks,
            &vq_residual_first, &vq_residual_second, &mx_values, &mx_scales}) {
            stream->retain_source(source);
        }
    }

    std::size_t size() const {
        std::size_t bytes = 0;
        for (const auto count : {
                 nint_q.size(), nint_sub_scale.size(), nint_sub_min.size(),
                 nint_anchor_scale.size(), nint_anchor_min.size(), q8_q.size(),
                 q8_scales.size(), vq_indices.size(), vq_state.size(), vq_aux.size(),
                 vq_anchors.size(), vq_codebooks.size(), vq_scales.size(),
                 vq_state_to_codebank.size(), vq_banks.size(), vq_parameters.size(),
                 vq_residual_codebooks.size(), vq_residual_first.size(),
                 vq_residual_second.size(), mx_values.size(), mx_scales.size()}) {
            bytes = checked_add(bytes, count, "packed streams size");
        }
        return bytes;
    }
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
        || (dimensions != 2 && dimensions != 3)
    ) {
        throw std::runtime_error(
            "MFE NINT cohort shape is inconsistent");
    }
    auto rows =
        cursor.scalar<std::int64_t>(
            "NINT output shape");
    if (dimensions == 3) {
        const auto output = cursor.scalar<std::int64_t>("NINT expert output shape");
        if (rows <= 0 || output <= 0 || rows > expected_rows / output)
            throw std::runtime_error("MFE NINT cohort row dimensions are inconsistent");
        rows *= output;
    }
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

std::optional<MlxNintWeight> add_nint_pool(
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
    if (!pack_execution) {
        return MlxNintWeight::from_blob(payload);
    }
    auto storage = detail::parse_nint_blob(payload, true);
    if (
        storage.output_size != checked_rows
        || storage.input_size != neuron_len
        || storage.groups
            != (
                neuron_len + storage.group_size - 1
            ) / storage.group_size
    ) {
        throw std::runtime_error(
            "MFE NINT cohort shape is inconsistent");
    }

    while ((streams.nint_q.size() & 15u) != 0u) {
        streams.nint_q.push_back(0);
    }
    const int row_metadata_offset = checked_int(
        streams.nint_q.size(),
        "NINT row-metadata offset");
    const int q_offset = checked_int(
        checked_add(streams.nint_q.size(),
            checked_product(storage.row_metadata.size(), sizeof(std::uint32_t), "NINT metadata bytes"),
            "NINT q offset"),
        "NINT q offset");
    const int sub_offset = checked_int(
        streams.nint_sub_scale.size(),
        "NINT sub offset");

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
        descriptors[base + kNintBits] = storage.bits;
        descriptors[base + kNintGroupSize] = storage.group_size;
        descriptors[base + kNintGroups] = storage.groups;
        descriptors[base + kNintQOffset] = q_offset;
        descriptors[base + kNintSubOffset] = sub_offset;
        descriptors[base + kNintAnchorOffset] = 0;
        descriptors[base + kNintExecution] = 0;
        descriptors[base + kNintRowMetadataOffset] = row_metadata_offset;
        const auto first_row = local_expert * static_cast<std::size_t>(out_per_expert);
        const auto first_q = storage.row_metadata[first_row * 4u] & 15u;
        bool uniform_q = first_q == 4u || first_q == 5u || first_q == 6u || first_q == 8u;
        for (int row = 1; uniform_q && row < out_per_expert; ++row)
            uniform_q = (storage.row_metadata[(first_row + row) * 4u] & 15u) == first_q;
        descriptors[base + kNintQDispatch] = static_cast<int>(uniform_q);
        descriptors[base + kNintV2] = 1;
    }

    streams.nint_q.append(std::move(storage.row_metadata));
    streams.nint_q.append(storage.borrowed_q);
    streams.nint_q.append(std::move(storage.q_packed));
    while ((streams.nint_q.size() & 15u) != 0u) {
        streams.nint_q.push_back(0);
    }
    for (const auto view : storage.borrowed_sub_scale) streams.nint_sub_scale.append(view);
    for (const auto view : storage.borrowed_sub_min) streams.nint_sub_min.append(view);
    streams.nint_sub_scale.append(std::move(storage.sub_scale));
    streams.nint_sub_min.append(std::move(storage.sub_min));
    return std::nullopt;
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
    const detail::ParsedVqStorage& weight,
    std::vector<RotationSpec>& rotations) {
    if (weight.rotation_block == 0) {
        return 0;
    }
    std::vector<std::int8_t> values(weight.rotation_signs.begin(), weight.rotation_signs.end());
    if (
        values.size()
        != static_cast<std::size_t>(
            weight.input_size)
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
            rotation.block == weight.rotation_block
            && rotation.seed == weight.rotation_seed
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
        make_raw_array(std::vector<std::uint8_t>(values.begin(), values.end()), mlx::core::int8),
        std::move(values),
        weight.rotation_block,
        weight.rotation_seed,
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

std::optional<MlxVqWeight> add_vq_pool(
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
    if (!pack_execution) {
        return MlxVqWeight::from_blob(dtype, payload, runtime);
    }
    auto weight = detail::parse_vq_blob(dtype, payload, runtime, true);
    if (
        weight.output_size != metadata.output_size
        || weight.input_size != metadata.input_size
        || weight.output_shape != metadata.output_shape
        || weight.rotation_block
            != metadata.rotation_block
        || weight.rotation_seed
            != metadata.rotation_seed
    ) {
        throw std::runtime_error(
            "MFE VQ header/full parse mismatch");
    }
    const auto& profile = weight.label;
    const bool group64_execution =
        profile == "NVQ2J-XL"
        && weight.execution_layout == 1;
    const bool native_jsc = detail::vq_native_jsc(weight.execution_layout, weight.group_size, weight.state_bits,
        weight.table_banks, weight.index_bits, weight.vector_size, weight.input_size,
        weight.aux_mode, weight.code_bank_mode);
    const bool native_nvq1 = detail::vq_native_nvq1(weight.execution_layout, weight.group_size,
        weight.vector_size, weight.label, weight.index_bits, weight.state_bits,
        weight.code_bank_mode, weight.aux_mode, weight.table_banks);
    const bool packed_nvq1s_execution =
        profile == "NVQ1-S" && native_nvq1;
    const bool packed_nvq1l_execution =
        profile == "NVQ1-L" && native_nvq1;

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
            weight.group_size;
        descriptors[base + kVqGroups] =
            weight.groups;
        descriptors[base + kVqVectorSize] =
            weight.vector_size;
        descriptors[base + kVqVectors] =
            weight.vectors;
        descriptors[base + kVqIndexBits] =
            weight.index_bits;
        descriptors[base + kVqStateBits] =
            weight.state_bits;
        descriptors[base + kVqStates] =
            weight.states;
        descriptors[base + kVqEntries] =
            weight.entries;
        descriptors[base + kVqCodeBanks] =
            weight.code_banks;
        descriptors[base + kVqAuxMode] =
            weight.aux_mode;
        descriptors[base + kVqCodeBankMode] =
            weight.code_bank_mode;
        descriptors[base + kVqHasTableBanks] =
            static_cast<int>(
                weight.table_banks > 1);
        descriptors[base + kVqGroupsPerSuper] =
            weight.groups_per_supergroup;
        descriptors[base + kVqSupergroups] =
            weight.supergroups;
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
        const auto& profile = weight.label;
        const int execution_profile =
            profile == "NVQ2J-L" || profile == "NVQ2J-XL"
                ? kVqProfileJscExtended8
                : profile == "NVQ3J-512" || profile == "NVQ3J-L"
                ? kVqProfileJscExtended4
                : profile == "NVQ2J" || profile == "NVQ3J"
                    || (native_jsc && weight.index_bits == 8)
                ? (
                    weight.vector_size == 4
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
                            packed_nvq1l_execution
                                ? kVqProfileNvq1L
                                : (
                                    packed_nvq1s_execution
                                        ? kVqProfileNvq1S
                                        : kVqProfileGeneric
                                )
                        )
                );
        descriptors[base + kVqProfile] =
            execution_profile
            | (weight.residual_position_bits << 8)
            | (weight.residual_block_vectors << 16);
        descriptors[base + kVqExecutionLayout] =
            group64_execution
                ? 2 : 0;
        descriptors[base + kVqResidualCodebookOffset] =
            residual_codebook_offset;
        descriptors[base + kVqResidualRecordOffset] =
            residual_record_offset;
    }

    streams.vq_indices.append(weight.borrowed_indices);
    streams.vq_indices.append(std::move(weight.indices));
    streams.vq_state.append(weight.borrowed_state);
    streams.vq_state.append(std::move(weight.states_packed));
    if (!group64_execution) {
        streams.vq_aux.append(weight.borrowed_auxiliary);
        streams.vq_aux.append(std::move(weight.auxiliary));
    }
    streams.vq_anchors.append(std::move(weight.anchors));
    streams.vq_codebooks.append(std::span(
        reinterpret_cast<const std::uint8_t*>(weight.borrowed_codebooks.data()),
        weight.borrowed_codebooks.size()));
    streams.vq_codebooks.append(std::move(weight.codebooks));
    streams.vq_scales.append(std::move(weight.scales));
    streams.vq_state_to_codebank.append(std::move(weight.state_to_codebank));
    streams.vq_banks.append(std::move(weight.bank_ids));
    streams.vq_parameters.append(weight.parameter);
    if (weight.residual_position_bits != 0) {
        streams.vq_residual_codebooks.append(std::move(weight.residual_codebook));
        streams.vq_residual_first.append(std::move(weight.residual_first));
        streams.vq_residual_second.append(std::move(weight.residual_second));
    }
    return std::nullopt;
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

struct MfePoolView {
    std::vector<std::int32_t> expert_ids;
    std::string dtype;
    std::span<const std::uint8_t> runtime;
    std::span<const std::uint8_t> payload;
};

struct MfeProjectionView {
    int experts;
    int output;
    int input;
    std::vector<MfePoolView> pools;
};

MfeProjectionView inspect_nint_projection(std::span<const std::uint8_t> blob) {
    BlobCursor cursor(blob);
    const auto bits = cursor.scalar<std::uint8_t>("NINT bits") & 127;
    const auto sub_bits = cursor.scalar<std::uint8_t>("NINT sub bits");
    const auto group_size = cursor.scalar<std::int32_t>("NINT group size");
    const auto axis = cursor.scalar<std::int32_t>("NINT axis");
    const auto input = cursor.scalar<std::int32_t>("NINT input width");
    const auto rank = cursor.scalar<std::uint32_t>("NINT rank");
    if (bits < 1 || bits > 8 || sub_bits < 1 || sub_bits > 8 ||
        group_size <= 0 || rank != 3 || axis != 0 || input <= 0)
        throw std::runtime_error("routed NINT tensor must use [experts,out,in]");
    const int experts = checked_int(checked_size(cursor.scalar<std::int64_t>("NINT experts"), "NINT experts"), "NINT experts");
    const int output = checked_int(checked_size(cursor.scalar<std::int64_t>("NINT output"), "NINT output"), "NINT output");
    if (experts <= 0 || output <= 0 || cursor.scalar<std::int64_t>("NINT input") != input)
        throw std::runtime_error("routed NINT tensor shape disagrees");
    const auto rows = checked_int(checked_product(static_cast<std::size_t>(experts),
        static_cast<std::size_t>(output), "routed NINT rows"), "routed NINT rows");
    if (cursor.scalar<std::uint32_t>("NINT rows") != rows ||
        cursor.scalar<std::uint32_t>("NINT groups") != (static_cast<std::int64_t>(input) + group_size - 1) / group_size ||
        static_cast<std::size_t>(rows) > cursor.remaining() / 4)
        throw std::runtime_error("routed NINT tensor row metadata disagrees");
    MfePoolView pool;
    pool.expert_ids.resize(experts);
    std::iota(pool.expert_ids.begin(), pool.expert_ids.end(), 0);
    pool.dtype = "NINT";
    pool.payload = blob;
    return {experts, output, input, {std::move(pool)}};
}

MfeProjectionView inspect_mfe_projection(std::span<const std::uint8_t> blob) {
    BlobCursor cursor(blob);
    const auto magic_bytes = cursor.bytes(4, "header");
    const std::string_view magic(
        reinterpret_cast<const char*>(magic_bytes.data()), magic_bytes.size());
    if (magic != "MFE1" && magic != "NIM1" && magic != "NIM2") {
        throw std::runtime_error("invalid MFE magic");
    }
    MfeProjectionView result{
        checked_positive(cursor.scalar<std::uint32_t>("expert count"), "expert count"),
        checked_positive(cursor.scalar<std::uint32_t>("output width"), "output width"),
        checked_positive(cursor.scalar<std::uint32_t>("neuron length"), "neuron length"),
        {},
    };
    const auto pool_count = cursor.scalar<std::uint32_t>("pool count");
    if (pool_count == 0 || pool_count > static_cast<std::uint32_t>(result.experts)) {
        throw std::runtime_error("invalid MFE pool count");
    }
    if (static_cast<std::size_t>(result.experts) > cursor.remaining() / sizeof(std::int32_t)) {
        throw std::runtime_error("MFE expert count exceeds its payload");
    }
    std::vector<int> owners(static_cast<std::size_t>(result.experts), -1);
    for (std::uint32_t pool = 0; pool < pool_count; ++pool) {
        const auto count = cursor.scalar<std::uint32_t>("pool header");
        const auto dtype_bytes = magic == "NIM1" ? 0u
            : cursor.scalar<std::uint32_t>("v2 pool header");
        const auto payload_bytes = cursor.scalar<std::uint64_t>("pool header");
        const auto runtime_bytes = magic == "NIM1" ? std::uint64_t{0}
            : cursor.scalar<std::uint64_t>("v2 pool header");
        if (count == 0 || count > static_cast<std::uint32_t>(result.experts) ||
            (magic != "NIM1" && (dtype_bytes == 0 || dtype_bytes > 32))) {
            throw std::runtime_error("invalid MFE pool metadata");
        }
        MfePoolView current;
        current.expert_ids.resize(count);
        for (auto& expert : current.expert_ids) {
            expert = cursor.scalar<std::int32_t>("expert IDs");
        }
        claim_experts(current.expert_ids, owners, static_cast<int>(pool));
        current.dtype = "NINT";
        if (magic != "NIM1") {
            const auto raw_dtype = cursor.bytes(dtype_bytes, "cohort dtype");
            if (!is_ascii(raw_dtype)) {
                throw std::runtime_error("MFE cohort dtype must be ASCII");
            }
            current.dtype = std::string(mfq::canonical_format_dtype(std::string_view(
                reinterpret_cast<const char*>(raw_dtype.data()), raw_dtype.size())));
        }
        current.runtime = cursor.bytes(
            checked_size(runtime_bytes, "cohort runtime metadata"), "cohort runtime metadata");
        current.payload = cursor.bytes(
            checked_size(payload_bytes, "cohort payload"), "cohort payload");
        result.pools.push_back(std::move(current));
    }
    if (cursor.remaining() != 0) {
        throw std::runtime_error("trailing bytes in MFE tensor");
    }
    if (std::find(owners.begin(), owners.end(), -1) != owners.end()) {
        throw std::runtime_error("MFE pools do not cover every expert");
    }
    return result;
}

struct ProjectionStorageInfo {
    std::size_t packed_bytes;
    bool automatic_mxfp4_nax_prefill;
    int logical_experts;
};

array concatenate_1d(
    std::vector<array> values) {
    if (values.empty()) {
        throw std::invalid_argument("cannot concatenate an empty packed stream list");
    }
    const auto empty = empty_packed_array(values.front().dtype());
    if (std::all_of(values.begin(), values.end(), [&](const array& value) {
            return value.data_shared_ptr() == empty.data_shared_ptr();
        })) {
        return empty;
    }
    const bool all_vectors = std::all_of(
        values.begin(),
        values.end(),
        [](const array& value) {
            return value.ndim() == 1;
        });
    if (!all_vectors) {
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
struct MfeReadAllocator : std::allocator<T> {
    template <typename U> struct rebind { using other = MfeReadAllocator<U>; };
    MfeReadAllocator() = default;
    template <typename U> MfeReadAllocator(const MfeReadAllocator<U>&) {}
    void construct(T* value) { ::new (static_cast<void*>(value)) T; }
    template <typename U, typename... Args>
    void construct(U* value, Args&&... args) { std::construct_at(value, std::forward<Args>(args)...); }
    template <typename U> bool operator==(const MfeReadAllocator<U>&) const { return true; }
};
using MfeReadBytes = std::vector<std::uint8_t, MfeReadAllocator<std::uint8_t>>;

template <typename T, typename Bytes>
void append_scalar_bytes(
    Bytes& target,
    T value) {
    const auto previous = target.size();
    target.resize(previous + sizeof(T));
    std::memcpy(target.data() + previous, &value, sizeof(T));
}

template <typename T, typename Bytes>
void patch_scalar_bytes(
    Bytes& target,
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

struct MfeReadCounters {
    std::size_t bytes = 0, calls = 0, parallel_body_batches = 0;
    double seconds = 0;
};
struct MfeReadRange {
    std::uint64_t source;
    std::size_t destination;
    std::size_t bytes;
};
using MfeReadPlan = std::vector<MfeReadRange>;
thread_local MfeReadPlan* current_mfe_plan = nullptr;
struct MfeReadPlanScope {
    explicit MfeReadPlanScope(MfeReadPlan* value)
        : previous(std::exchange(current_mfe_plan, value)) {}
    ~MfeReadPlanScope() { current_mfe_plan = previous; }
    MfeReadPlan* previous;
};
thread_local MfeReadCounters* current_mfe_reads = nullptr;
struct MfeReadScope {
    explicit MfeReadScope(MfeReadCounters* value)
        : previous(std::exchange(current_mfe_reads, value)) {}
    ~MfeReadScope() { current_mfe_reads = previous; }
    MfeReadCounters* previous;
};
void record_mfe_read(std::size_t bytes, std::chrono::steady_clock::time_point started) {
    if (!current_mfe_reads || !bytes) return;
    current_mfe_reads->bytes += bytes;
    ++current_mfe_reads->calls;
    current_mfe_reads->seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
}

void read_mfe_direct_targets(const MfqContainer& model, const std::string& name,
    std::span<const MfeDirectReadTarget> targets, bool complete, std::size_t missing) {
    if (missing > kMfeDirectRangeMaxExperts || targets.size() < 2) {
        for (const auto& range : targets) {
            const auto started = std::chrono::steady_clock::now();
            auto* destination = complete ? range.destination : range.wire;
            model.read_range_into(name, range.offset,
                std::as_writable_bytes(std::span(destination, range.bytes)));
            record_mfe_read(range.bytes, started);
        }
        return;
    }
    struct Context {
        const MfqContainer& model;
        const std::string& name;
        std::span<const MfeDirectReadTarget> targets;
        bool complete;
        std::mutex mutex;
        std::exception_ptr failure;
    } context{model, name, targets, complete, {}, {}};
    const auto started = std::chrono::steady_clock::now();
    dispatch_apply_f(targets.size(), dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
        &context, [](void* raw, std::size_t index) {
            auto& work = *static_cast<Context*>(raw);
            const auto& range = work.targets[index];
            try {
                auto* destination = work.complete ? range.destination : range.wire;
                work.model.read_range_into(work.name, range.offset,
                    std::as_writable_bytes(std::span(destination, range.bytes)));
            } catch (...) {
                std::scoped_lock lock(work.mutex);
                if (!work.failure) work.failure = std::current_exception();
            }
        });
    if (context.failure) std::rethrow_exception(context.failure);
    if (current_mfe_reads) {
        for (const auto& range : targets) {
            current_mfe_reads->bytes += range.bytes;
            ++current_mfe_reads->calls;
        }
        ++current_mfe_reads->parallel_body_batches;
        current_mfe_reads->seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
    }
}

template <typename Bytes>
void append_record_range(
    Bytes& target,
    const MfqContainer& model,
    const std::string& name,
    std::uint64_t offset,
    std::uint64_t nbytes) {
    const auto begin = target.size();
    target.resize(checked_add(begin, nbytes, "streamed appended range"));
    if (current_mfe_plan) {
        if (nbytes) current_mfe_plan->push_back({offset, begin, static_cast<std::size_t>(nbytes)});
        return;
    }
    const auto started = current_mfe_reads ? std::chrono::steady_clock::now()
        : std::chrono::steady_clock::time_point{};
    model.read_range_into(name, offset,
        std::as_writable_bytes(std::span<std::uint8_t>(target).subspan(begin)));
    record_mfe_read(nbytes, started);
}

void read_mfe_batch(const MfqContainer& model, const std::string& name,
    std::span<MfeReadBytes> blobs, std::span<const MfeReadPlan> plans) {
    struct Range {
        std::uint64_t source;
        std::span<std::uint8_t> target;
    };
    std::vector<Range> ranges;
    for (std::size_t index = 0; index < plans.size(); ++index)
        for (const auto& range : plans[index]) ranges.push_back({range.source,
            std::span(blobs[index]).subspan(range.destination, range.bytes)});
    std::sort(ranges.begin(), ranges.end(), [](const auto& a, const auto& b) { return a.source < b.source; });
    constexpr std::size_t max_read = std::size_t{8} << 20;
    MfeReadBytes scratch;
    std::vector<std::span<std::byte>> destinations;
    for (std::size_t first = 0; first < ranges.size();) {
        const auto offset = ranges[first].source;
        auto end = checked_range_add(offset, ranges[first].target.size(), "MFE batch range");
        auto last = first + 1;
        bool contiguous = true;
        while (last < ranges.size() && ranges[last].source <= end) {
            const auto next = checked_range_add(ranges[last].source, ranges[last].target.size(), "MFE batch range");
            if (next - offset > max_read) break;
            contiguous = contiguous && ranges[last].source == end;
            end = std::max(end, next);
            ++last;
        }
        const auto bytes = static_cast<std::size_t>(end - offset);
        const auto started = std::chrono::steady_clock::now();
        if (last == first + 1) {
            model.read_range_into(name, offset, std::as_writable_bytes(ranges[first].target));
            record_mfe_read(bytes, started);
        } else if (contiguous) {
            destinations.clear();
            for (auto index = first; index < last; ++index)
                destinations.push_back(std::as_writable_bytes(ranges[index].target));
            model.readv_range(name, offset, destinations);
            record_mfe_read(bytes, started);
        } else {
            scratch.resize(bytes);
            model.read_range_into(name, offset, std::as_writable_bytes(std::span(scratch)));
            record_mfe_read(bytes, started);
            for (auto index = first; index < last; ++index)
                std::memcpy(ranges[index].target.data(), scratch.data() + ranges[index].source - offset,
                    ranges[index].target.size());
        }
        first = last;
    }
}

template <typename Bytes>
void append_packed_value_slice(
    Bytes& target,
    const MfqContainer& model,
    const std::string& name,
    std::uint64_t stream_offset,
    std::size_t first_value,
    std::size_t value_count,
    int bits) {
    if (value_count == 0) {
        return;
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
        append_record_range(
            target, model, name,
            checked_range_add(
                stream_offset,
                source_bit / 8,
                "streamed aligned source offset"),
            bit_count / 8);
        return;
    }
    const auto source_byte = source_bit / 8;
    const auto shift = source_bit & 7u;
    const auto source_bytes = checked_add(
        checked_add(shift, bit_count, "streamed source bits"),
        7,
        "streamed source rounding") / 8;
    const auto started = current_mfe_reads ? std::chrono::steady_clock::now()
        : std::chrono::steady_clock::time_point{};
    const auto raw = model.read_range(
        name,
        checked_range_add(
            stream_offset,
            source_byte,
            "streamed packed source offset"),
        source_bytes);
    record_mfe_read(source_bytes, started);
    const auto begin = target.size();
    const auto bytes = checked_packed_size(value_count, bits, "streamed packed slice");
    target.resize(checked_add(begin, bytes, "streamed packed append"));
    for (std::size_t index = 0; index < bytes; ++index) {
        const auto upper = index + 1 < raw.size() ? std::uint32_t{raw[index + 1]} : 0u;
        target[begin + index] = static_cast<std::uint8_t>(
            (std::uint32_t{raw[index]} | (upper << 8)) >> shift);
    }
    if ((bit_count & 7u) != 0u)
        target.back() &= static_cast<std::uint8_t>((1u << (bit_count & 7u)) - 1u);
}

template <std::size_t Selectors>
std::vector<std::array<std::uint32_t, Selectors>> selector_expert_prefixes(
    std::span<const std::uint8_t> selectors,
    int rows_per_expert) {
    if (rows_per_expert <= 0 || selectors.size() % rows_per_expert != 0) {
        throw std::runtime_error("invalid streamed NINT expert selector geometry");
    }
    std::vector<std::array<std::uint32_t, Selectors>> result(
        selectors.size() / rows_per_expert + 1);
    std::array<std::uint32_t, Selectors> counts{};
    for (std::size_t row = 0; row < selectors.size(); ++row) {
        if (selectors[row] >= Selectors) {
            throw std::runtime_error("invalid streamed NINT selector");
        }
        ++counts[selectors[row]];
        if ((row + 1) % rows_per_expert == 0) {
            result[(row + 1) / rows_per_expert] = counts;
        }
    }
    return result;
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
    std::vector<std::uint8_t> header;
    std::vector<std::array<std::uint32_t, 4>> k_expert_prefixes;
    std::vector<std::array<std::uint32_t, 8>> q_expert_prefixes;
};

struct MfeNvqJscStreamLayout {
    int rows = 0;
    int groups = 0;
    int vectors = 0;
    int signs = 0;
    int state_bits = 0;
    int index_bits = 0;
    int auxiliary_bits = 7;
    bool group64 = false;
    int decoder_profile = kVqProfileGeneric;
    int execution_layout = 0;
    std::uint64_t payload_offset = 0;
    std::uint64_t prefix_bytes = 0;
    std::uint64_t anchors_offset = 0;
    std::uint64_t state_offset = 0;
    std::uint64_t indices_offset = 0;
    std::uint64_t signs_offset = 0;
    std::vector<std::uint8_t> prefix;
};

struct MfeMxStreamLayout {
    int rows = 0;
    int columns = 0;
    int bits = 0;
    std::uint64_t payload_offset = 0;
    std::uint64_t values_offset = 0;
    std::uint64_t scales_offset = 0;
    std::vector<std::uint8_t> header;
};

using MfeStreamLayout = std::variant<
    MfeNintStreamLayout,
    MfeNvqJscStreamLayout,
    MfeMxStreamLayout>;

struct MfeStreamPool {
    std::string dtype;
    int expert_count = 0;
    std::uint64_t runtime_bytes = 0;
    std::uint64_t payload_offset = 0;
    std::uint64_t payload_bytes = 0;
    MfeStreamLayout layout;
    std::vector<std::uint8_t> runtime;
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
    std::shared_ptr<const MlxMfeKernelProfile> kernel_profile;
};

std::size_t streamed_expert_storage_bound(
    const MfeStreamProjection& projection, const MfeStreamExpertLocation& location) {
    const auto rows = static_cast<std::size_t>(projection.out_per_expert);
    const auto packed = [&](std::size_t count, int bits) {
        return checked_packed_size(count, bits, "streamed expert working storage");
    };
    const auto product = [&](std::size_t first, std::size_t second) {
        return checked_product(first, second, "streamed expert working storage");
    };
    const auto add = [](std::size_t first, std::size_t second) {
        return checked_add(first, second, "streamed expert working storage");
    };
    auto bytes = std::visit([&](const auto& layout) -> std::size_t {
        using Layout = std::decay_t<decltype(layout)>;
        if constexpr (std::is_same_v<Layout, MfeNintStreamLayout>) {
            const auto groups = static_cast<std::size_t>(layout.groups);
            const auto width = product(groups, layout.group_size);
            auto result = add(layout.header.size(), product(rows, 4));
            if (layout.adaptive_storage) {
                result = add(result, add(packed(rows, 2), packed(rows, 3)));
                for (int index = 0; index < 4; ++index) {
                    const auto count = layout.k_expert_prefixes[location.local_expert + 1][index]
                        - layout.k_expert_prefixes[location.local_expert][index];
                    result = add(result, product(2, packed(product(count, groups),
                        layout.nominal_sub_bits - 1 + index)));
                }
                for (int index = 0; index < 8; ++index) {
                    const auto count = layout.q_expert_prefixes[location.local_expert + 1][index]
                        - layout.q_expert_prefixes[location.local_expert][index];
                    result = add(result, packed(product(count, width), index + 1));
                }
            } else {
                const auto metadata = product(rows, groups);
                const auto values = product(rows, width);
                result = add(result, layout.legacy_unpacked
                    ? add(product(metadata, 2), values)
                    : add(product(2, packed(metadata, layout.nominal_sub_bits)), packed(values, layout.q_bits)));
            }
            // Four uint32 row descriptors and per-row byte/alignment tails.
            return add(result, product(rows, 32));
        } else if constexpr (std::is_same_v<Layout, MfeNvqJscStreamLayout>) {
            auto result = add(layout.prefix_bytes, product(rows, 2));
            if (layout.group64) result = add(result, product(product(rows, layout.groups), 8));
            else {
                result = add(result, packed(product(rows, layout.groups), layout.state_bits));
                result = add(result, packed(product(rows, layout.vectors), layout.index_bits));
                result = add(result, packed(product(rows, layout.signs), layout.auxiliary_bits));
            }
            // Packed indices/states remain packed; anchors and small lookup
            // tables may widen. NVQ1-L's built-in 2048x8 table has no wire bytes.
            return add(add(result, product(rows, 2)), add(
                std::max<std::size_t>(16384, product(layout.prefix_bytes, 2)), 1024));
        } else {
            const auto values = product(rows, layout.bits == 4 ? layout.columns / 2 : layout.columns);
            const auto scales = layout.bits == 4 ? product(rows, layout.columns / 32)
                : product((rows + 127) / 128, layout.columns / 128);
            return add(layout.header.size(), add(values, scales));
        }
    }, location.pool->layout);
    bytes = add(bytes, product(location.pool->runtime_bytes, 2));
    return add(bytes, sizeof(std::int32_t) * kDescriptorSize + 21 * 15
        + static_cast<std::size_t>(getpagesize()));
}

MfeNintStreamLayout parse_streamed_nint_layout(
    const MfqContainer& model,
    const std::string& name,
    std::uint64_t payload_offset,
    std::uint64_t payload_bytes,
    int expected_rows,
    int expected_columns,
    int rows_per_expert,
    int expected_experts = 0) {
    constexpr std::uint64_t kHeaderBytes = 42;
    if (payload_bytes < kHeaderBytes) {
        throw std::runtime_error("truncated streamed NINT payload: " + name);
    }
    auto header = model.read_range(
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
    if (dimensions == 3 && expected_experts > 0) {
        if (payload_bytes < kHeaderBytes + 8)
            throw std::runtime_error("truncated streamed NINT expert header: " + name);
        header = model.read_range(name, payload_offset, kHeaderBytes + 8);
    } else if (dimensions != 2 || expected_experts > 0) {
        throw std::runtime_error("unsupported streamed NINT rank: " + name);
    }
    const auto shape_offset = dimensions == 3 ? 26u : 18u;
    const auto shape_rows = read_scalar<std::int64_t>(header, shape_offset, "NINT rows");
    const auto shape_columns = read_scalar<std::int64_t>(
        header, shape_offset + 8, "NINT columns");
    const auto rows = read_scalar<std::uint32_t>(
        header, shape_offset + 16, "NINT output size");
    const auto groups = read_scalar<std::uint32_t>(
        header, shape_offset + 20, "NINT group count");
    const bool shape_valid = dimensions == 3
        ? read_scalar<std::int64_t>(header, 18, "NINT experts") == expected_experts
            && shape_rows == rows_per_expert
            && std::int64_t(expected_experts) * rows_per_expert == expected_rows
        : shape_rows == expected_rows;
    const bool adaptive_storage = (raw_bits & 0x80) != 0;
    const int q_bits = raw_bits & 0x7f;
    if (q_bits < 1
        || q_bits > 8 || nominal_sub_bits < 1
        || nominal_sub_bits > 8 || group_size <= 0 || axis != 0
        || !shape_valid || columns != expected_columns
        || shape_columns != expected_columns
        || rows != static_cast<std::uint32_t>(expected_rows)
        || groups != static_cast<std::uint32_t>(
            (expected_columns - 1) / group_size + 1)) {
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
    result.header.assign(header.begin(), header.begin() + 18);
    patch_scalar_bytes<std::uint32_t>(result.header, 14, 2, "NINT matrix rank");
    append_scalar_bytes<std::int64_t>(result.header, expected_rows);
    append_scalar_bytes<std::int64_t>(result.header, expected_columns);
    append_scalar_bytes<std::uint32_t>(result.header, rows);
    append_scalar_bytes<std::uint32_t>(result.header, groups);
    result.payload_offset = payload_offset;
    result.anchor_scale_offset = payload_offset + header.size();
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
    result.k_expert_prefixes = selector_expert_prefixes<4>(
        result.k_selectors, rows_per_expert);
    cursor += k_selector_bytes;
    for (int selector = 0; selector < 4; ++selector) {
        const int row_bits = nominal_sub_bits - 1 + selector;
        const auto selected = result.k_expert_prefixes.back()[selector];
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
    result.q_expert_prefixes = selector_expert_prefixes<8>(
        result.q_selectors, rows_per_expert);
    cursor += q_selector_bytes;
    const auto values_per_row = checked_product(
        static_cast<std::size_t>(groups),
        static_cast<std::size_t>(group_size),
        "NINTv2 values per row");
    for (int selector = 0; selector < 8; ++selector) {
        const int row_bits = selector + 1;
        const auto selected = result.q_expert_prefixes.back()[selector];
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
    result.decoder_profile = index_bits > 8
        ? (vector_size == 4 ? kVqProfileJscExtended4 : kVqProfileJscExtended8)
        : (vector_size == 4 ? kVqProfileJsc4 : kVqProfileJsc8);
    result.execution_layout = profile == 5 && result.group64 ? 2 : 0;
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
    result.prefix = model.read_range(name, payload_offset, result.prefix_bytes);
    return result;
}

// NVQ1-S/L have the same row-major anchor/state/index/auxiliary structure as
// planar JSC, but use a one-bit delta per group instead of seven-bit signs.
// Keep the codebook prefix intact and slice packed streams at bit boundaries.
MfeNvqJscStreamLayout parse_streamed_nvq1_layout(
    const MfqContainer& model,
    const std::string& name,
    std::uint64_t payload_offset,
    std::uint64_t payload_bytes,
    int expected_rows,
    int expected_columns) {
    constexpr std::size_t header_bytes = 40;
    if (payload_bytes < header_bytes) {
        throw std::runtime_error("truncated streamed NVQ1 header: " + name);
    }
    const auto header = model.read_range(name, payload_offset, header_bytes);
    const bool short_profile = std::memcmp(header.data(), "NQ1S", 4) == 0;
    const bool long_profile = std::memcmp(header.data(), "NQ1L", 4) == 0;
    const int profile = header[4];
    const int state_bits = header[5];
    const auto group_size = read_scalar<std::uint16_t>(header, 6, "NVQ1 group size");
    if ((!short_profile && !long_profile) ||
        (short_profile && (profile != 1 || state_bits != 4 ||
                           group_size != 24 || expected_columns % 8 != 0)) ||
        (long_profile && ((profile != 1 && profile != 2) ||
                          state_bits < 1 || state_bits > 8 ||
                          group_size == 0 || group_size % 8 != 0)) ||
        read_scalar<std::int32_t>(header, 8, "NVQ1 axis") != 0 ||
        read_scalar<std::int32_t>(header, 12, "NVQ1 width") != expected_columns ||
        read_scalar<std::uint32_t>(header, 16, "NVQ1 dimensions") != 2 ||
        read_scalar<std::int64_t>(header, 20, "NVQ1 rows") != expected_rows ||
        read_scalar<std::int64_t>(header, 28, "NVQ1 columns") != expected_columns ||
        read_scalar<std::uint32_t>(header, 36, "NVQ1 output size") !=
            static_cast<std::uint32_t>(expected_rows)) {
        throw std::runtime_error("invalid streamed NVQ1 geometry: " + name);
    }
    MfeNvqJscStreamLayout result;
    result.rows = expected_rows;
    result.groups = (expected_columns - 1) / group_size + 1;
    result.vectors = (expected_columns - 1) / 8 + 1;
    result.signs = result.groups;
    result.state_bits = state_bits;
    result.index_bits = short_profile ? 9 : 11;
    result.auxiliary_bits = 1;
    if (detail::vq_native_nvq1(0, group_size, 8,
            short_profile ? "NVQ1-S" : "NVQ1-L", result.index_bits, state_bits,
            short_profile ? 2 : 0, 3, 1))
        result.decoder_profile = short_profile ? kVqProfileNvq1S : kVqProfileNvq1L;
    result.payload_offset = payload_offset;
    result.prefix_bytes = header_bytes + (short_profile ? 2048 : profile == 2 ? 4096 : 0);
    auto cursor = checked_range_add(payload_offset, result.prefix_bytes, "NVQ1 prefix");
    result.anchors_offset = cursor;
    cursor = checked_range_add(cursor, checked_range_product(
        expected_rows, 2, "NVQ1 anchors"), "NVQ1 anchors end");
    const auto advance = [&](int values_per_row, int bits) {
        const auto offset = cursor;
        cursor = checked_range_add(cursor, checked_packed_size(
            checked_product(static_cast<std::size_t>(expected_rows),
                static_cast<std::size_t>(values_per_row), "NVQ1 stream values"),
            bits, "NVQ1 packed stream"), "NVQ1 stream end");
        return offset;
    };
    result.state_offset = advance(result.groups, state_bits);
    result.indices_offset = advance(result.vectors, result.index_bits);
    result.signs_offset = advance(result.groups, 1);
    if (cursor != checked_range_add(payload_offset, payload_bytes, "NVQ1 payload end")) {
        throw std::runtime_error("invalid streamed NVQ1 payload length: " + name);
    }
    result.prefix = model.read_range(name, payload_offset, result.prefix_bytes);
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
        header,
    };
}

template <typename Bytes>
void append_streamed_nint_expert(
    Bytes& result,
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
    const auto payload_begin = result.size();
    result.insert(result.end(), layout.header.begin(), layout.header.end());
    patch_scalar_bytes<std::int64_t>(
        result, payload_begin + 18, rows_per_expert, "NINTv2 shape row");
    patch_scalar_bytes<std::uint32_t>(
        result, payload_begin + 34, static_cast<std::uint32_t>(rows_per_expert),
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
                append_packed_value_slice(result, model, name, offset, first, count, bits);
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
        return;
    }
    const auto k_slice = std::span<const std::uint8_t>(
        layout.k_selectors).subspan(begin, rows_per_expert);
    const auto packed_k = pack_small_selectors(k_slice, 2);
    result.insert(result.end(), packed_k.begin(), packed_k.end());
    for (int selector = 0; selector < 4; ++selector) {
        const auto selected_before = layout.k_expert_prefixes[local_expert][selector];
        const auto selected_here = layout.k_expert_prefixes[local_expert + 1][selector] - selected_before;
        const int bits = layout.nominal_sub_bits - 1 + selector;
        for (const auto offset : {
                 layout.sub_scale_offsets[selector],
                 layout.sub_min_offsets[selector]}) {
            append_packed_value_slice(
                result, model, name, offset,
                checked_product(selected_before,
                    static_cast<std::size_t>(layout.groups),
                    "NINTv2 subgroup prefix"),
                checked_product(selected_here,
                    static_cast<std::size_t>(layout.groups),
                    "NINTv2 subgroup slice"),
                bits);
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
        const auto selected_before = layout.q_expert_prefixes[local_expert][selector];
        const auto selected_here = layout.q_expert_prefixes[local_expert + 1][selector] - selected_before;
        append_packed_value_slice(
            result, model, name, layout.q_offsets[selector],
            checked_product(selected_before, values_per_row,
                "NINTv2 q prefix"),
            checked_product(selected_here, values_per_row,
                "NINTv2 q slice"),
            selector + 1);
    }
}

template <typename Bytes>
void append_streamed_nvq_expert(
    Bytes& result,
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
    const auto payload_begin = result.size();
    result.insert(result.end(), layout.prefix.begin(), layout.prefix.end());
    patch_scalar_bytes<std::int64_t>(
        result, payload_begin + 20, rows_per_expert, "NVQ-JSC shape row");
    patch_scalar_bytes<std::uint32_t>(
        result, payload_begin + 36, static_cast<std::uint32_t>(rows_per_expert),
        "NVQ-JSC output size");
    {
        MfeReadPlanScope immediate(direct_read_metadata != MfeMetadataRead::Planned ? nullptr : current_mfe_plan);
        append_record_range(
            result, model, name,
            layout.anchors_offset + begin * 2,
            static_cast<std::uint64_t>(rows_per_expert) * 2);
    }
    if (layout.group64) {
        append_record_range(
            result, model, name,
            layout.indices_offset
                + begin * static_cast<std::size_t>(layout.groups) * 8,
            static_cast<std::uint64_t>(rows_per_expert)
                * static_cast<std::uint64_t>(layout.groups) * 8);
        return;
    }
    const auto append_bits = [&](std::uint64_t offset,
                                 std::size_t values_per_row,
                                 int bits) {
        append_packed_value_slice(
            result, model, name, offset,
            checked_product(begin, values_per_row,
                "NVQ-JSC stream prefix"),
            checked_product(
                static_cast<std::size_t>(rows_per_expert),
                values_per_row,
                "NVQ-JSC stream slice"),
            bits);
    };
    append_bits(layout.state_offset, layout.groups, layout.state_bits);
    append_bits(layout.indices_offset, layout.vectors, layout.index_bits);
    append_bits(layout.signs_offset, layout.signs, layout.auxiliary_bits);
}

template <typename Bytes>
void append_streamed_mx_expert(
    Bytes& result,
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
    const auto payload_begin = result.size();
    result.insert(result.end(), layout.header.begin(), layout.header.end());
    patch_scalar_bytes<std::uint64_t>(
        result, payload_begin + 8, rows_per_expert, "MX logical rows");
    patch_scalar_bytes<std::uint64_t>(
        result, payload_begin + 24, rows_per_expert, "MX storage rows");
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
            result, payload_begin + 40, rows_per_expert, "MXFP4 scale rows");
        const bool deferred = direct_read_metadata == MfeMetadataRead::DeferredMxScales;
        MfeReadPlanScope immediate(direct_read_metadata == MfeMetadataRead::Eager ? nullptr : current_mfe_plan);
        const auto scale_begin = result.size();
        append_record_range(
            result, model, name,
            layout.scales_offset + begin * scale_stride,
            static_cast<std::uint64_t>(rows_per_expert) * scale_stride);
        if (deferred) std::fill(result.begin() + scale_begin, result.end(), std::uint8_t{0});
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
            result, payload_begin + 40, scale_rows, "MXFP8 scale rows");
        append_record_range(
            result, model, name,
            layout.scales_offset + (begin / 128) * scale_stride,
            static_cast<std::uint64_t>(scale_rows) * scale_stride);
    }
}

template <typename Bytes>
void append_streamed_mfe_expert_payload(
    Bytes& result,
    const MfqContainer& model,
    const std::string& name,
    const MfeStreamPool& pool,
    int local_expert,
    int rows_per_expert) {
    std::visit(
        [&](const auto& layout) {
            using Layout = std::decay_t<decltype(layout)>;
            if constexpr (std::is_same_v<Layout, MfeNintStreamLayout>) {
                append_streamed_nint_expert(
                    result, model, name, layout, local_expert, rows_per_expert);
            } else if constexpr (
                std::is_same_v<Layout, MfeNvqJscStreamLayout>) {
                append_streamed_nvq_expert(
                    result, model, name, layout, local_expert, rows_per_expert);
            } else {
                append_streamed_mx_expert(
                    result, model, name, layout, local_expert, rows_per_expert);
            }
        },
        pool.layout);
}

} // namespace

namespace {

struct MfeGroupWork {
    std::size_t count;
    std::function<void(std::int64_t, std::int64_t)> load;
};

struct MfeGroupTask {
    struct promise_type {
        std::optional<MfeGroupWork> work;
        std::optional<MlxMfeWeight> result;
        std::exception_ptr failure;
        MfeGroupTask get_return_object() {
            return MfeGroupTask{std::coroutine_handle<promise_type>::from_promise(*this)};
        }
        std::suspend_always initial_suspend() noexcept { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }
        std::suspend_always yield_value(MfeGroupWork value) {
            work.emplace(std::move(value));
            return {};
        }
        void return_value(MlxMfeWeight value) { result.emplace(std::move(value)); }
        void unhandled_exception() { failure = std::current_exception(); }
    };
    std::coroutine_handle<promise_type> handle;
    explicit MfeGroupTask(std::coroutine_handle<promise_type> value) : handle(value) {}
    MfeGroupTask(MfeGroupTask&& other) noexcept : handle(std::exchange(other.handle, {})) {}
    MfeGroupTask(const MfeGroupTask&) = delete;
    ~MfeGroupTask() { if (handle) handle.destroy(); }
    void resume() {
        handle.promise().work.reset();
        handle.resume();
        if (handle.promise().failure) std::rethrow_exception(handle.promise().failure);
    }
};

}

struct MlxMfeOffloadCache::Impl {
    MfeGroupTask prepare_grouped(const std::string& name,
        const std::vector<std::int32_t>& active_experts,
        std::span<const std::uint32_t> activation_counts, bool prefill_scan);
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
            MlxMfeWeight value,
            bool temporary = false)
            : expert(global),
              weight(std::move(value)),
              packed_nbytes(weight.packed_nbytes()),
              transient(temporary) {}

        std::int32_t expert = 0;
        MlxMfeWeight weight;
        std::size_t packed_nbytes = 0;
        bool transient = false;
    };

    struct MfeCacheValue {
        Key key;
        std::shared_ptr<const MfeCachedExpert> weight;
        std::uint8_t credit = 0;
        std::shared_ptr<void> pin;
        bool pin_attempted = false;
    };

    struct PreparedProjection {
        MlxMfeWeight weight;
        std::unordered_map<std::int32_t, std::int32_t> local_ids;
    };

    struct InFlight {
        bool complete = false;
        std::optional<PreparedProjection> result;
        std::vector<std::shared_ptr<const MfeCachedExpert>> pages;
    };

    struct PinRequest {
        Key key;
        std::weak_ptr<const MfeCachedExpert> page;
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
        PackedCopyPrimitive(mlx::core::default_stream(mlx::core::Device::gpu), Shape{1}, {}).prepare_gpu();
        resources.bind([this] {
            std::lock_guard lock(mutex);
            std::size_t payload = 0;
            for (const auto& item : backing_records) payload += item.second;
            return MlxResourceUsage{0, 0, resident_bytes, payload, 0};
        });
    }

    ~Impl() {
        {
            std::lock_guard lock(pin_mutex);
            pin_stopping = true;
            pin_queue.clear();
            pin_pending.clear();
        }
        pin_ready.notify_one();
        if (pin_worker.joinable()) pin_worker.join();
    }

    bool enqueue_pin(const Key& key, const std::shared_ptr<const MfeCachedExpert>& page) noexcept {
        if (page->packed_nbytes < (std::size_t{64} << 10)) return true;
        try {
            std::lock_guard lock(pin_mutex);
            if (pin_stopping) return false;
            if (pin_pending.contains(key)) return true;
            if (!pin_worker.joinable()) pin_worker = std::thread([this] { pin_loop(); });
            pin_queue.push_back({key, page});
            try {
                pin_pending.emplace(key, std::prev(pin_queue.end()));
            } catch (...) {
                pin_queue.pop_back();
                throw;
            }
            pin_ready.notify_one();
            return true;
        } catch (...) {
            return false;
        }
    }

    void cancel_pin(const Key& key) {
        std::lock_guard lock(pin_mutex);
        if (const auto found = pin_pending.find(key); found != pin_pending.end()) {
            pin_queue.erase(found->second);
            pin_pending.erase(found);
        }
    }

    void pin_loop() {
        for (;;) {
            PinRequest request;
            {
                std::unique_lock lock(pin_mutex);
                pin_ready.wait(lock, [&] { return pin_stopping || !pin_queue.empty(); });
                if (pin_stopping) return;
                request = std::move(pin_queue.front());
                pin_queue.pop_front();
                pin_pending.erase(request.key);
            }
            std::shared_ptr<const MfeCachedExpert> page;
            {
                std::lock_guard lock(mutex);
                const auto cached = mfe_cache.find(request.key);
                page = request.page.lock();
                if (!page || cached == mfe_cache.end() || cached->second->weight != page) continue;
            }
            auto pin = page->weight.pin_packed_storage();
            {
                std::lock_guard lock(mutex);
                const auto cached = mfe_cache.find(request.key);
                if (cached != mfe_cache.end() && cached->second->weight == page)
                    cached->second->pin = std::move(pin);
            }
        }
    }

    static std::shared_ptr<const MfeStreamProjection> make_projection(
        int projection_experts, int rows_per_expert, int columns,
        std::vector<std::optional<MfeStreamExpertLocation>> locations) {
        auto profile = std::make_shared<MlxMfeKernelProfile>();
        profile->nint_q_dispatch = true;
        std::vector<int> group_sizes;
        std::unordered_set<const MfeStreamPool*> visited;
        for (const auto& location : locations) {
            const auto& pool = *location->pool;
            if (!visited.insert(&pool).second) continue;
            if (const auto* nint = std::get_if<MfeNintStreamLayout>(&pool.layout)) {
                profile->family_mask |= 1u << kFamilyNint;
                group_sizes.push_back(nint->group_size);
                if (nint->adaptive_storage) {
                    for (std::size_t first = 0; first < nint->q_selectors.size(); first += rows_per_expert) {
                        const auto q = nint->q_selectors[first] + 1u;
                        profile->nint_q_dispatch &= q == 4u || q == 5u || q == 6u || q == 8u;
                        for (std::size_t row = 1; profile->nint_q_dispatch && row < rows_per_expert; ++row)
                            profile->nint_q_dispatch &= nint->q_selectors[first + row] == nint->q_selectors[first];
                    }
                } else {
                    profile->nint_q_dispatch &= nint->q_bits == 4 || nint->q_bits == 5
                        || nint->q_bits == 6 || nint->q_bits == 8;
                }
            } else if (const auto* mx = std::get_if<MfeMxStreamLayout>(&pool.layout)) {
                profile->family_mask |= 1u << (mx->bits == 4 ? kFamilyMxfp4 : kFamilyMxfp8);
            } else {
                const auto& vq = std::get<MfeNvqJscStreamLayout>(pool.layout);
                profile->family_mask |= 1u << kFamilyVq;
                profile->vq_profile_mask |= 1u << vq.decoder_profile;
                profile->vq_layouts |= std::uint64_t{1}
                    << (vq.decoder_profile * 7 + vq.execution_layout);
                profile->vq_execution_layout |= vq.execution_layout != 0;
            }
        }
        std::sort(group_sizes.begin(), group_sizes.end());
        group_sizes.erase(std::unique(group_sizes.begin(), group_sizes.end()), group_sizes.end());
        profile->nint_group_sizes = std::make_shared<const std::vector<int>>(std::move(group_sizes));
        return std::make_shared<MfeStreamProjection>(MfeStreamProjection{
            projection_experts, rows_per_expert, columns, std::move(locations), std::move(profile)});
    }

    std::shared_ptr<const MfeStreamProjection> parse_nint_projection(const std::string& name) {
        const auto& record = model.record(name);
        if (record.nbytes < 50)
            throw std::runtime_error("truncated streamed NINT expert header: " + name);
        const auto header = model.read_range(name, 0, 50);
        const auto count = read_scalar<std::int64_t>(header, 18, "NINT expert count");
        const auto output = read_scalar<std::int64_t>(header, 26, "NINT expert output");
        const auto input = read_scalar<std::int64_t>(header, 34, "NINT expert input");
        const auto limit = std::numeric_limits<int>::max();
        if (read_scalar<std::uint32_t>(header, 14, "NINT rank") != 3
            || count <= 0 || count > limit || (experts > 0 && count != experts)
            || output <= 0 || output > limit || input <= 0 || input > limit
            || count > limit / output)
            throw std::runtime_error("invalid streamed NINT expert geometry: " + name);
        auto pool = std::make_shared<MfeStreamPool>();
        pool->dtype = std::string(mfq::canonical_format_dtype(record.dtype));
        pool->expert_count = static_cast<int>(count);
        pool->payload_bytes = record.nbytes;
        pool->layout = parse_streamed_nint_layout(model, name, 0, record.nbytes,
            static_cast<int>(count * output), static_cast<int>(input),
            static_cast<int>(output), static_cast<int>(count));
        std::vector<std::optional<MfeStreamExpertLocation>> locations(static_cast<std::size_t>(count));
        for (int expert = 0; expert < count; ++expert)
            locations[expert] = MfeStreamExpertLocation{pool, expert};
        return make_projection(static_cast<int>(count), static_cast<int>(output),
            static_cast<int>(input), std::move(locations));
    }

    std::shared_ptr<const MfeStreamProjection>
    parse_mfe_projection(
        const std::string& name) {
        const auto& record = model.record(name);
        if (is_nint_dtype(record.dtype)) return parse_nint_projection(name);
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
                    pool_rows, static_cast<int>(columns),
                    static_cast<int>(rows_per_expert));
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
                if (payload_bytes < 4) {
                    throw std::runtime_error("truncated streamed VQ magic: " + name);
                }
                const auto magic = model.read_range(name, payload_offset, 4);
                if (std::memcmp(magic.data(), "NQ1S", 4) == 0 ||
                    std::memcmp(magic.data(), "NQ1L", 4) == 0) {
                    if (canonical != mfq::kNvqDtype || runtime_bytes != 0) {
                        throw std::runtime_error("invalid streamed NVQ1 cohort metadata: " + name);
                    }
                    layout = parse_streamed_nvq1_layout(
                        model, name, payload_offset, payload_bytes,
                        pool_rows, static_cast<int>(columns));
                } else {
                    layout = parse_streamed_nvq_jsc_layout(
                        model, name, payload_offset, payload_bytes,
                        pool_rows, static_cast<int>(columns));
                }
            } else {
                throw MfeStreamUnsupported(
                    "MFE contains a cohort without expert slicing support: "
                    + dtype);
            }

            auto pool = std::make_shared<MfeStreamPool>();
            pool->dtype = dtype;
            pool->expert_count = static_cast<int>(pool_experts);
            pool->runtime_bytes = runtime_bytes;
            pool->payload_offset = payload_offset;
            pool->payload_bytes = payload_bytes;
            pool->layout = std::move(layout);
            pool->runtime = model.read_range(name, runtime_offset, runtime_bytes);
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
        return make_projection(projection_experts, static_cast<int>(rows_per_expert),
            static_cast<int>(columns), std::move(locations));
    }

    std::shared_ptr<const MfeStreamProjection>
    mfe_projection_locked(const std::string& name) {
        const auto found = mfe_projections.find(name);
        if (found != mfe_projections.end()) {
            return found->second;
        }
        auto result = parse_mfe_projection(name);
        mfe_projections.emplace(name, result);
        model.record_prepared(name);
        return result;
    }

    MfeReadBytes load_mfe_expert_blob(
        const std::string& name,
        const MfeStreamProjection& projection,
        const MfeStreamExpertLocation& location,
        MfeReadCounters* reads = nullptr, MfeReadPlan* plan = nullptr) {
        MfeReadScope observation(reads);
        MfeReadPlanScope planned(plan);
        const auto& runtime = location.pool->runtime;
        MfeReadBytes blob;
        const auto prefix_bytes = std::visit([](const auto& layout) {
            if constexpr (std::is_same_v<std::decay_t<decltype(layout)>, MfeNvqJscStreamLayout>)
                return layout.prefix.size();
            else return layout.header.size();
        }, location.pool->layout);
        const auto reserve_bytes = checked_add(
            checked_add(location.pool->payload_bytes / location.pool->expert_count,
                checked_add(runtime.size(), prefix_bytes + 8, "streamed expert prefix"),
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
        append_scalar_bytes<std::uint64_t>(blob, 0);
        append_scalar_bytes<std::uint64_t>(blob, runtime.size());
        append_scalar_bytes<std::int32_t>(blob, 0);
        blob.insert(
            blob.end(),
            location.pool->dtype.begin(),
            location.pool->dtype.end());
        blob.insert(blob.end(), runtime.begin(), runtime.end());
        const auto payload_begin = blob.size();
        append_streamed_mfe_expert_payload(blob, model, name, *location.pool,
            location.local_expert, projection.out_per_expert);
        patch_scalar_bytes<std::uint64_t>(blob, 28, blob.size() - payload_begin,
            "streamed MFE payload size");
        return blob;
    }

    // Own the record table and source paths.  Streamed layers frequently
    // outlive the MfqContainer object used by their load call.
    MfqContainer model;
    std::size_t cache_limit = 0;
    const int experts = 0;
    mutable std::mutex mutex;
    std::unordered_map<
        std::string,
        std::shared_ptr<const MfeStreamProjection>> mfe_projections;
    MfeLru mfe_lru;
    MfeExpertCache mfe_cache;
    std::unordered_set<std::string> prefill_priority;
    std::size_t resident_bytes = 0;
    std::unordered_map<std::string, std::size_t> backing_records;
    MlxResourceTelemetry resources;
    mfq::HostParallelPool io_workers{{}, false};
    std::unordered_map<std::string, PreparedProjection> prepared;
    std::unordered_map<std::string,
        std::vector<std::shared_ptr<const MfeCachedExpert>>> resident_pages;
    std::unordered_map<std::string, std::shared_ptr<const MlxMfeCachedProjection>> cached_views;
    std::size_t page_generation = 0;
    std::unordered_map<std::string, std::uint8_t> route_confidence;
    std::unordered_map<std::string, std::shared_ptr<InFlight>> loading;
    std::condition_variable loaded;
    std::size_t epoch = 0;
    MlxMfeCacheReuseStats stats;
    std::mutex pin_mutex;
    std::condition_variable pin_ready;
    std::list<PinRequest> pin_queue;
    std::unordered_map<Key, std::list<PinRequest>::iterator, KeyHash> pin_pending;
    bool pin_stopping = false;
    std::thread pin_worker;

    void drop_prepared(const std::string& name) {
        const auto found = prepared.find(name);
        if (found == prepared.end()) return;
        resident_bytes -= found->second.weight.packed_nbytes();
        prepared.erase(found);
    }

    void drop_cached_view(const std::string& name) {
        ++page_generation;
        if (const auto found = cached_views.find(name); found != cached_views.end()) {
            resident_bytes -= found->second->metadata_bytes;
            cached_views.erase(found);
        }
    }

    void trim() {
        while (resident_bytes > cache_limit && !prepared.empty())
            drop_prepared(prepared.begin()->first);
        while (resident_bytes > cache_limit && !mfe_lru.empty()) {
            const auto victim = mfe_lru.begin();
            if (victim->credit > 0) {
                --victim->credit;
                mfe_lru.splice(mfe_lru.end(), mfe_lru, victim);
                continue;
            }
            drop_cached_view(victim->key.name);
            resident_bytes -= victim->weight->packed_nbytes;
            resident_pages.at(victim->key.name)[victim->key.expert].reset();
            ++stats.evictions;
            cancel_pin(victim->key);
            mfe_cache.erase(victim->key);
            mfe_lru.erase(victim);
        }
    }
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

void MlxMfeOffloadCache::activate_record(const std::string& name) {
    std::lock_guard lock(impl_->mutex);
    (void)impl_->mfe_projection_locked(name);
    impl_->backing_records.emplace(name, impl_->model.record(name).nbytes);
}

void MlxMfeOffloadCache::prepare_projection(
    const std::string& name, bool shared_input, bool fused_swiglu) {
    auto* preparation = MlxKernelPreparation::current();
    if (!preparation || preparation->routes() <= 0) return;
    std::shared_ptr<const MfeStreamProjection> projection;
    {
        std::lock_guard lock(impl_->mutex);
        projection = impl_->mfe_projection_locked(name);
    }
    preparation->collect([&] {
        preparation->add(packed_expert_copy({empty_packed_array(mlx::core::uint8)},
            {{PackedCopySpec{0, 0, 0, 1}}}, 1));
        const auto blob = impl_->load_mfe_expert_blob(
            name, *projection, *projection->experts_by_id.front());
        const auto weight = MlxMfeWeight::from_blob(blob)
            .with_kernel_profile(projection->kernel_profile);
        const auto indirect = MlxMfeWeight::concatenate_expert_pages({weight, weight}, projection->kernel_profile)
            .with_kernel_profile(projection->kernel_profile);
        const int routes = preparation->routes();
        for (const int rows : {1, 2, 3, 4, 5, 6, 8}) {
            const auto input = mlx::core::zeros(shared_input
                ? Shape{rows, projection->neuron_len}
                : Shape{rows, routes, projection->neuron_len}, mlx::core::float16);
            const auto ids = mlx::core::zeros(Shape{rows, routes}, mlx::core::int32);
            const auto add = [&](const array* groups) {
                preparation->add(fused_swiglu
                    ? weight.routed_swiglu(input, ids, 0.0f, groups)
                    : weight.routed_matmul(input, ids, groups));
                preparation->add(fused_swiglu
                    ? indirect.routed_swiglu(input, ids, 0.0f, groups)
                    : indirect.routed_matmul(input, ids, groups));
            };
            add(nullptr);
            if (rows > 1 && rows <= 24) {
                const auto groups = mlx::core::zeros(Shape{rows, routes, 8}, mlx::core::int32);
                add(&groups);
            }
        }
        const auto maximum_rows = *std::max_element(preparation->row_buckets().begin(),
            preparation->row_buckets().end());
        if (maximum_rows < 32 || !indirect.supports_grouped_mmq()
            || !uses_indirect_expert_pages(projection->kernel_profile->family_mask)) return;
        for (const int block_rows : {32, 48, 64, 80, 96, 128}) {
            for (const int rows : {32, maximum_rows}) {
                const auto ids = mlx::core::zeros(Shape{rows, routes}, mlx::core::int32);
                const auto order = mlx::core::arange(rows * routes, mlx::core::int32);
                const auto experts = mlx::core::arange(projection->experts, mlx::core::int32);
                preparation->add(make_grouped_mmq_plan(ids, order, experts,
                    projection->experts, block_rows).block_meta);
            }
        }
        const auto& profile = *projection->kernel_profile;
        std::vector<std::tuple<int, int, int>> cohorts;
        for (int family = 0; family < 7; ++family) {
            if (!(profile.family_mask & (1u << family))) continue;
            if (family == kFamilyNint && profile.nint_group_sizes && !profile.nint_group_sizes->empty()) {
                for (const int group : *profile.nint_group_sizes) cohorts.emplace_back(1 << family, group, 1);
            } else cohorts.emplace_back(1 << family, 0, 1);
        }
        const int width = fused_swiglu ? projection->out_per_expert / 2 : projection->out_per_expert;
        for (const bool nax : {false, true}) {
            if (nax && !mixed_grouped_nax_enabled(checked_int(checked_product(
                    maximum_rows, routes, "prefill preparation routes"), "prefill preparation routes"))) continue;
            for (const int block_rows : {32, 48, 64, 80, 96, 128}) {
                if (!nax && (block_rows == 48 || block_rows == 128)) continue;
                GroupedMmqConfig config{
                    .output_shape = Shape{1, width},
                    .block_rows = block_rows,
                    .tile_columns = nax ? grouped_mmq_tile_columns(block_rows, width, fused_swiglu) : 64,
                    .experts = 1,
                    .output_width = width,
                    .matrix_output_width = projection->out_per_expert,
                    .input_width = projection->neuron_len,
                    .fused_swiglu = static_cast<int>(fused_swiglu),
                    .family_mask = static_cast<int>(profile.family_mask),
                    .vq_profile_mask = static_cast<int>(profile.vq_profile_mask),
                    .nint_group_sizes = profile.nint_group_sizes,
                    .use_nax = nax,
                    .indirect_pages = true,
                };
                preparation->add(grouped_mmq_dispatch({}, config));
                if (!nax || fused_swiglu) continue;
                for (std::size_t count = 2; count <= cohorts.size(); ++count) {
                    auto passes = std::make_shared<const std::vector<std::tuple<int, int, int>>>(
                        cohorts.begin(), cohorts.begin() + count);
                    config.family_mask = 0;
                    for (const auto& [family, group, population] : *passes) config.family_mask |= family;
                    config.experts = static_cast<int>(count);
                    config.cohort_populations = std::move(passes);
                    preparation->add(grouped_mmq_dispatch({}, config));
                }
            }
        }
    });
}

void MlxMfeOffloadCache::prepare_swiglu_pair(
    const std::string& gate, const std::string& up, int routes) {
    auto* preparation = MlxKernelPreparation::current();
    if (!preparation || routes <= 0 || routes > 16) return;
    std::array<std::shared_ptr<const MfeStreamProjection>, 2> projections;
    {
        std::lock_guard lock(impl_->mutex);
        projections = {impl_->mfe_projection_locked(gate), impl_->mfe_projection_locked(up)};
    }
    if (!uses_indirect_expert_pages(projections[0]->kernel_profile->family_mask
            | projections[1]->kernel_profile->family_mask)) return;
    preparation->collect([&] {
        std::vector<MlxMfeWeight> weights;
        for (std::size_t index = 0; index < projections.size(); ++index) {
            const auto& projection = projections[index];
            const auto blob = impl_->load_mfe_expert_blob(index == 0 ? gate : up,
                *projection, *projection->experts_by_id.front());
            const auto weight = MlxMfeWeight::from_blob(blob).with_kernel_profile(projection->kernel_profile);
            weights.push_back(MlxMfeWeight::concatenate_expert_pages(
                {weight, weight}, projection->kernel_profile).with_kernel_profile(projection->kernel_profile));
        }
        for (const int rows : {1, 2}) {
            const auto input = mlx::core::zeros(Shape{rows, projections[0]->neuron_len}, mlx::core::float16);
            const std::vector<std::int32_t> host_ids(static_cast<std::size_t>(rows) * routes, 0);
            const auto ids = array(host_ids.begin(), Shape{rows, routes});
            const auto groups = rows > 1 ? moe_route_groups_host(host_ids, rows, routes) : std::nullopt;
            preparation->add(weights[0].routed_swiglu_pair(weights[1], input, ids, 0.0f,
                groups ? &*groups : nullptr));
            if (rows == 1) {
                const auto map = array({0}, mlx::core::int32);
                preparation->add(weights[0].routed_swiglu_pair(weights[1], input, ids,
                    0.0f, nullptr, &map, &map));
            }
        }
    });
}

std::size_t MlxMfeOffloadCache::backing_payload_bytes() const {
    std::lock_guard lock(impl_->mutex);
    std::size_t bytes = 0;
    for (const auto& record : impl_->backing_records) bytes += record.second;
    return bytes;
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

std::optional<std::vector<MlxMfeWeight>> MlxMfeOffloadCache::layer_mfe(
    std::span<const std::string> names,
    std::span<const std::int32_t> active_experts) {
    if (names.empty()) throw std::invalid_argument("MFE layer records cannot be empty");
    std::size_t bytes = 0;
    std::size_t dense_bytes = 0;
    std::size_t missing_bytes = 0;
    {
        std::lock_guard lock(impl_->mutex);
        for (const auto& name : names) {
            const auto projection = impl_->mfe_projection_locked(name);
            bytes = checked_add(bytes, impl_->model.record(name).nbytes, "MFE layer bytes");
            const auto elements = checked_product(checked_product(projection->experts,
                projection->out_per_expert, "MFE layer rows"), projection->neuron_len,
                "MFE layer elements");
            dense_bytes = checked_add(dense_bytes, checked_product(elements, 2, "MFE layer F16 bytes"),
                "MFE layer execution bytes");
            std::unordered_set<std::int32_t> seen;
            for (const auto expert : active_experts) {
                if (expert < 0 || expert >= projection->experts || !projection->experts_by_id[expert])
                    throw std::out_of_range("bulk MFE expert is unavailable");
                if (!seen.insert(expert).second || impl_->mfe_cache.contains({name, expert})) continue;
                const auto& pool = projection->experts_by_id[expert]->pool;
                missing_bytes = checked_add(missing_bytes, pool->payload_bytes / pool->expert_count,
                    "bulk MFE missing bytes");
            }
        }
        if (bytes > impl_->cache_limit / 2) return std::nullopt;
        if (!active_experts.empty() && missing_bytes < bytes / 2) return std::nullopt;
    }
    const auto staging = checked_add(checked_add(bytes, bytes, "MFE layer staging"),
        mlx_reference_enabled() ? dense_bytes : 0, "MFE layer staging and execution");
    if (!MlxResidentBudgetScope::permits_optional(staging)) return std::nullopt;
    MlxResidentBudgetScope::reserve(staging);
    std::size_t epoch = 0;
    {
        std::lock_guard lock(impl_->mutex);
        if (bytes > impl_->cache_limit / 2) return std::nullopt;
        epoch = impl_->epoch;
    }
    std::vector<MlxMfeWeight> result;
    result.reserve(names.size());
    constexpr std::size_t chunk = std::size_t{8} << 20;
    const auto read_record = [&](const std::string& name, array storage) {
        const std::span blob(storage.data<std::uint8_t>(), storage.nbytes());
        const auto pieces = (blob.size() + chunk - 1) / chunk;
        std::vector<double> durations(pieces);
        impl_->io_workers.run(0, pieces, 1,
            static_cast<int>(std::min(8u, std::max(1u, std::thread::hardware_concurrency()))),
            [&](std::int64_t begin, std::int64_t end) {
                for (auto index = begin; index < end; ++index) {
                    const auto offset = static_cast<std::size_t>(index) * chunk;
                    const auto started = std::chrono::steady_clock::now();
                    impl_->model.read_range_into(name, offset, std::as_writable_bytes(
                        std::span(blob.data() + offset, std::min(chunk, blob.size() - offset))));
                    durations[index] = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - started).count();
                }
            });
        std::lock_guard lock(impl_->mutex);
        impl_->stats.bytes_read += blob.size();
        impl_->stats.read_calls += pieces;
        impl_->stats.io_seconds += std::accumulate(durations.begin(), durations.end(), 0.0);
        return storage;
    };
    auto storage = read_record(names.front(), allocate_wire_array(
        static_cast<std::size_t>(impl_->model.record(names.front()).nbytes)));
    for (std::size_t index = 0; index < names.size(); ++index) {
        std::future<decltype(storage)> next;
        if (index + 1 < names.size()) {
            auto pending = allocate_wire_array(
                static_cast<std::size_t>(impl_->model.record(names[index + 1]).nbytes));
            next = std::async(std::launch::async,
                [&read_record, &names, index, pending = std::move(pending)]() mutable {
                return read_record(names[index + 1], std::move(pending));
            });
        }
        result.push_back(is_nint_dtype(impl_->model.record(names[index]).dtype)
            ? MlxMfeWeight::from_nint_blob(storage) : MlxMfeWeight::from_blob(storage));
        if (next.valid()) storage = next.get();
    }
    {
        std::lock_guard lock(impl_->mutex);
        if (epoch == impl_->epoch)
            for (const auto& name : names)
                impl_->backing_records.emplace(name, impl_->model.record(name).nbytes);
    }
    return result;
}

MfeGroupTask MlxMfeOffloadCache::Impl::prepare_grouped(
    const std::string& name,
    const std::vector<std::int32_t>& active_experts,
    std::span<const std::uint32_t> activation_counts, bool prefill_scan) {
    auto* const impl_ = this;
    if (active_experts.empty())
        throw std::invalid_argument("streamed MFE active expert set cannot be empty");
    if (!activation_counts.empty() && activation_counts.size() != active_experts.size())
        throw std::invalid_argument("MFE activation counts must match the active experts");
    if (std::any_of(activation_counts.begin(), activation_counts.end(),
        [](auto count) { return count == 0; }))
        throw std::invalid_argument("MFE activation counts must be positive");
    std::optional<MlxResidentBudgetHold> staging;
    std::vector<std::shared_ptr<const Impl::MfeCachedExpert>> protected_pages;
    std::optional<std::size_t> admission_epoch;
    if (MlxResidentBudgetScope::enabled()) {
        std::size_t working_bytes = 0;
        {
            std::lock_guard lock(impl_->mutex);
            admission_epoch = impl_->epoch;
            const auto projection = impl_->mfe_projection_locked(name);
            protected_pages.reserve(std::min(active_experts.size(), static_cast<std::size_t>(projection->experts)));
            const bool indirect_pages = uses_indirect_expert_pages(projection->kernel_profile->family_mask);
            if (indirect_pages) working_bytes = std::size_t{3} * getpagesize();
            std::size_t missing_bytes = 0, largest_missing = 0, missing_count = 0;
            std::unordered_set<std::int32_t> seen;
            for (const auto expert : active_experts) {
                if (!seen.insert(expert).second) continue;
                if (expert < 0 || expert >= projection->experts || !projection->experts_by_id[expert])
                    throw std::out_of_range("streamed MFE expert is unavailable");
                const auto cached = impl_->mfe_cache.find({name, expert});
                const bool missing = cached == impl_->mfe_cache.end();
                if (!missing) protected_pages.push_back(cached->second->weight);
                const auto bytes = missing
                    ? streamed_expert_storage_bound(*projection, *projection->experts_by_id[expert])
                    : indirect_pages ? sizeof(std::int32_t) * (kDescriptorSize + 1)
                        + 21 * sizeof(std::uint64_t) : cached->second->weight->packed_nbytes;
                const auto copies = missing && !indirect_pages ? std::size_t{5} : std::size_t{1};
                working_bytes = checked_add(working_bytes, checked_product(bytes, copies,
                    "expert packed staging"), "active expert working set");
                if (missing && indirect_pages) {
                    missing_bytes = checked_add(missing_bytes, bytes, "missing expert storage");
                    largest_missing = std::max(largest_missing, bytes);
                    ++missing_count;
                }
            }
            if (missing_count) {
                const auto workers = std::max(1u, std::thread::hardware_concurrency());
                const auto concurrent = std::min(missing_count,
                    std::size_t(prefill_scan ? workers : std::min(kMfeSmallMReadWorkers, workers)));
                const auto batch_bytes = prefill_scan
                    ? std::max(largest_missing, std::size_t{8} << 20) : largest_missing;
                const auto in_flight = std::min(missing_bytes,
                    checked_product(concurrent, batch_bytes, "expert reader working set"));
                working_bytes = checked_add(working_bytes,
                    checked_product(in_flight, std::size_t{3}, "expert reader staging"),
                    "active expert working set");
            }
        }
        staging.emplace(working_bytes);
    }
    std::unique_lock lock(impl_->mutex);
    std::shared_ptr<Impl::InFlight> completed;
    if (const auto found = impl_->loading.find(name); found != impl_->loading.end()) {
        completed = found->second;
        ++impl_->stats.load_waits;
        const auto started = std::chrono::steady_clock::now();
        impl_->loaded.wait(lock, [&] { return completed->complete; });
        impl_->stats.wait_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
    }
    const auto projection = impl_->mfe_projection_locked(name);

    std::vector<Impl::Key> ordered_keys;
    ordered_keys.reserve(active_experts.size());
    std::unordered_set<Impl::Key, Impl::KeyHash> active_keys;
    active_keys.reserve(active_experts.size());
    std::unordered_map<std::int32_t, std::uint32_t> counts;
    for (std::size_t index = 0; index < active_experts.size(); ++index) {
        const auto expert = active_experts[index];
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
        auto& count = counts[expert];
        count = std::min<std::uint64_t>(3, static_cast<std::uint64_t>(count)
            + (activation_counts.empty() ? 1 : activation_counts[index]));
    }

    impl_->backing_records.emplace(name, impl_->model.record(name).nbytes);

    const bool priority_scan = prefill_scan && impl_->prefill_priority.contains(name);
    const auto touch = [&](const Impl::Key& key, bool hit) {
        if (prefill_scan && !priority_scan) return;
        const auto cached = impl_->mfe_cache.find(key);
        if (cached == impl_->mfe_cache.end()) return;
        auto& value = *cached->second;
        value.credit = static_cast<std::uint8_t>(std::min<unsigned>(3,
            value.credit + counts.at(key.expert) - (hit ? 0 : 1)));
        impl_->mfe_lru.splice(impl_->mfe_lru.end(), impl_->mfe_lru, cached->second);
    };
    const auto reuse_packed = [&](const Impl::PreparedProjection& packed)
        -> std::optional<MlxMfeWeight> {
        std::vector<std::int32_t> local;
        local.reserve(ordered_keys.size());
        for (const auto& key : ordered_keys) {
            const auto found = packed.local_ids.find(key.expert);
            if (found == packed.local_ids.end()) return std::nullopt;
            local.push_back(found->second);
        }
        auto weight = packed.weight;
        for (const auto& key : ordered_keys) touch(key, true);
        impl_->stats.expert_hits += ordered_keys.size();
        ++impl_->stats.packed_hits;
        lock.unlock();
        return weight.select_experts(local);
    };
    if (completed && completed->result)
        if (auto result = reuse_packed(*completed->result)) co_return std::move(*result);
    const auto prepared = impl_->prepared.find(name);
    if (prepared != impl_->prepared.end())
        if (auto result = reuse_packed(prepared->second)) co_return std::move(*result);
    impl_->drop_prepared(name);

    struct ActivePage {
        Impl::Key key;
        std::shared_ptr<const Impl::MfeCachedExpert> value;
        bool staged = false;
    };
    struct PendingPage {
        std::size_t page = 0;
        std::int32_t expert = 0;
        MfeReadCounters reads;
        double prepare_seconds = 0;
        std::optional<array> destination;
    };
    std::vector<ActivePage> pages;
    pages.reserve(ordered_keys.size());
    std::vector<PendingPage> pending;
    pending.reserve(ordered_keys.size());
    for (const auto& key : ordered_keys) {
        const auto cached = impl_->mfe_cache.find(key);
        if (cached != impl_->mfe_cache.end()) {
            pages.push_back({key, cached->second->weight, false});
            ++impl_->stats.expert_hits;
            continue;
        }
        if (const auto protected_page = std::find_if(protected_pages.begin(), protected_pages.end(),
                [&](const auto& page) { return page->expert == key.expert; }); protected_page != protected_pages.end()) {
            pages.push_back({key, *protected_page, false});
            ++impl_->stats.expert_hits;
            continue;
        }
        if (completed) {
            const auto found = std::find_if(completed->pages.begin(), completed->pages.end(),
                [&](const auto& page) { return page->expert == key.expert; });
            if (found != completed->pages.end()) {
                pages.push_back({key, *found, false});
                ++impl_->stats.expert_hits;
                continue;
            }
        }
        const auto page = pages.size();
        pages.push_back({key, nullptr, true});
        pending.push_back({page, key.expert, {}});
    }

    std::vector<std::shared_ptr<const Impl::MfeCachedExpert>> bank_pages;
    std::size_t bank_bytes = 0;
    const bool indirect_pages = uses_indirect_expert_pages(projection->kernel_profile->family_mask);
    if (!prefill_scan) {
        auto& confidence = impl_->route_confidence[name];
        confidence = pending.empty() ? static_cast<std::uint8_t>(std::min<unsigned>(15, confidence + 1)) : 0;
    }
    const auto resident = impl_->resident_pages.find(name);
    if (!indirect_pages && pending.empty() && resident != impl_->resident_pages.end()) {
        for (const auto& page : resident->second) {
            if (!page) continue;
            bank_pages.push_back(page);
            bank_bytes = checked_add(bank_bytes, page->packed_nbytes + 128,
                "prepared MFE expert bytes");
        }
    }
    if (!indirect_pages && pending.empty()) {
        for (const auto& page : pages) {
            if (impl_->mfe_cache.contains(page.key)) continue;
            bank_pages.push_back(page.value);
            bank_bytes = checked_add(bank_bytes, page.value->packed_nbytes + 128,
                "in-flight MFE expert bytes");
        }
    }
    const auto spare = impl_->cache_limit - std::min(impl_->cache_limit, impl_->resident_bytes);
    bool prepare = pending.empty() && bank_pages.size() > 1 && bank_bytes <= spare;
    const auto epoch = admission_epoch.value_or(impl_->epoch);
    auto in_flight = std::make_shared<Impl::InFlight>();
    impl_->loading.emplace(name, in_flight);
    struct FinishLoading {
        Impl& impl;
        const std::string& name;
        std::unique_lock<std::mutex>& lock;
        std::shared_ptr<Impl::InFlight> request;
        ~FinishLoading() {
            if (!lock.owns_lock()) lock.lock();
            request->complete = true;
            impl.loading.erase(name);
            impl.loaded.notify_all();
        }
    } finish{*impl_, name, lock, in_flight};
    lock.unlock();
    if (prepare) {
        prepare = MlxResidentBudgetScope::permits_optional(bank_bytes);
        if (prepare) MlxResidentBudgetScope::reserve(bank_bytes);
    }
    if (!prepare) bank_pages.clear();
    std::optional<array> transient_arena;
    if (prefill_scan && pending.size() > 1 && !mlx_reference_enabled()) {
        std::vector<std::size_t> offsets{0};
        for (const auto& request : pending) {
            const auto bytes = streamed_expert_storage_bound(*projection,
                *projection->experts_by_id[request.expert]);
            offsets.push_back(checked_add(checked_add(offsets.back(), bytes,
                "prefill expert arena"), 15, "prefill expert alignment") & ~std::size_t{15});
        }
        if (offsets.back() > spare && offsets.back() <= std::size_t(std::numeric_limits<int>::max())) {
            transient_arena = offsets.back() >= (std::size_t{64} << 10)
                ? allocate_wire_array(offsets.back())
                : allocate_packed_array(offsets.back(), mlx::core::uint8);
            for (std::size_t index = 0; index < pending.size(); ++index) {
                auto view = array(Shape{static_cast<int>(offsets[index + 1] - offsets[index])}, mlx::core::uint8, nullptr, {});
                view.copy_shared_buffer(*transient_arena, view.strides(), view.flags(), view.size(), offsets[index]);
                view.set_status(array::Status::available);
                pending[index].destination.emplace(std::move(view));
            }
            pooled_expert_staging.fetch_add(1, std::memory_order_relaxed);
        }
    }
    if (prefill_scan && pending.size() > 1) {
        std::vector<std::size_t> read_order(pending.size());
        std::iota(read_order.begin(), read_order.end(), 0);
        std::sort(read_order.begin(), read_order.end(), [&](auto a, auto b) {
            const auto& left = *projection->experts_by_id[pending[a].expert];
            const auto& right = *projection->experts_by_id[pending[b].expert];
            return std::tie(left.pool->payload_offset, left.local_expert)
                < std::tie(right.pool->payload_offset, right.local_expert);
        });
        std::vector<std::vector<std::size_t>> batches;
        std::size_t batch_bytes = 0;
        for (const auto index : read_order) {
            const auto& location = *projection->experts_by_id[pending[index].expert];
            const auto bytes = streamed_expert_storage_bound(*projection, location);
            const auto* previous = batches.empty() ? nullptr
                : &*projection->experts_by_id[pending[batches.back().back()].expert];
            if (!previous || previous->pool != location.pool
                || previous->local_expert + 1 != location.local_expert
                || bytes > (std::size_t{8} << 20) - std::min(batch_bytes, std::size_t{8} << 20)) {
                batches.emplace_back();
                batch_bytes = 0;
            }
            batches.back().push_back(index);
            batch_bytes += bytes;
        }
        co_yield MfeGroupWork{batches.size(),
            [&](std::int64_t begin, std::int64_t end) {
                for (auto batch_index = begin; batch_index < end; ++batch_index) {
                    const auto& batch = batches[batch_index];
                    const auto started = std::chrono::steady_clock::now();
                    std::vector<MfeReadBytes> wire;
                    std::vector<MfeReadPlan> plans(batch.size());
                    wire.reserve(batch.size());
                    auto& reads = pending[batch.front()].reads;
                    for (std::size_t index = 0; index < batch.size(); ++index) {
                        const auto& location = *projection->experts_by_id[pending[batch[index]].expert];
                        wire.push_back(impl_->load_mfe_expert_blob(name, *projection, location, &reads,
                            batch.size() > 1 ? &plans[index] : nullptr));
                    }
                    if (batch.size() > 1) {
                        MfeReadScope observation(&reads);
                        read_mfe_batch(impl_->model, name, wire, plans);
                    }
                    for (std::size_t index = 0; index < batch.size(); ++index) {
                        auto& request = pending[batch[index]];
                        const std::array blobs{std::span<const std::uint8_t>(wire[index])};
                        pages[request.page].value = std::make_shared<Impl::MfeCachedExpert>(
                            request.expert, request.destination
                                ? MlxMfeWeight::from_projection_blobs_impl(blobs, false, nullptr, &*request.destination)
                                : MlxMfeWeight::from_blob(wire[index]), request.destination.has_value());
                    }
                    pending[batch.front()].prepare_seconds = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - started).count() - reads.seconds;
                }
            }};
    } else if (!pending.empty()) co_yield MfeGroupWork{pending.size(),
        [&](std::int64_t begin, std::int64_t end) {
            for (auto index = begin; index < end; ++index) {
                auto& request = pending[index];
                const auto& location = *projection->experts_by_id[request.expert];
                const auto started = std::chrono::steady_clock::now();
                const auto eligible = [&] {
                    if (mlx_reference_enabled()) return false;
                    if (const auto* mx = std::get_if<MfeMxStreamLayout>(&location.pool->layout))
                        return mx->bits == 4;
                    const auto* vq = std::get_if<MfeNvqJscStreamLayout>(&location.pool->layout);
                    if (!vq || vq->group64 || vq->execution_layout != 0
                        || vq->decoder_profile == kVqProfileGeneric
                        || vq->decoder_profile == kVqProfileNpqS || vq->decoder_profile == kVqProfileNpqL) return false;
                    const auto rows = static_cast<std::uint64_t>(projection->out_per_expert);
                    return rows * vq->groups * vq->state_bits % 8 == 0
                        && rows * vq->vectors * vq->index_bits % 8 == 0
                        && rows * vq->signs * vq->auxiliary_bits % 8 == 0;
                }();
                const auto* mx = std::get_if<MfeMxStreamLayout>(&location.pool->layout);
                const bool deferred_scales = eligible && mx && !prefill_scan
                    && pending.size() <= kMfeDirectRangeMaxExperts;
                MfeReadPlan plan;
                struct MetadataScope {
                    MfeMetadataRead previous;
                    MetadataScope(MfeMetadataRead value) : previous(std::exchange(direct_read_metadata, value)) {}
                    ~MetadataScope() { direct_read_metadata = previous; }
                } metadata_scope(deferred_scales ? MfeMetadataRead::DeferredMxScales
                    : eligible ? MfeMetadataRead::Eager : MfeMetadataRead::Planned);
                auto blob = impl_->load_mfe_expert_blob(name, *projection, location, &request.reads,
                    eligible ? &plan : nullptr);
                std::vector<MfeDirectReadTarget> targets;
                for (const auto& range : plan)
                    targets.push_back({range.source, blob.data() + range.destination, range.bytes});
                std::optional<MlxMfeWeight> parsed;
                const auto parse = [&] {
                    const std::array blobs{std::span<const std::uint8_t>(blob)};
                    return request.destination
                        ? MlxMfeWeight::from_projection_blobs_impl(blobs, false, nullptr, &*request.destination)
                        : MlxMfeWeight::from_blob(blob);
                };
                if (eligible) {
                    MfeDirectReadScope redirect(targets);
                    parsed.emplace(parse());
                } else parsed.emplace(parse());
                const bool complete = std::all_of(targets.begin(), targets.end(),
                    [](const auto& range) { return range.destination != nullptr; });
                {
                    MfeReadScope observation(&request.reads);
                    read_mfe_direct_targets(impl_->model, name, targets, complete, pending.size());
                }
                if (deferred_scales && complete) {
                    const auto scale_offset = mx->scales_offset
                        + static_cast<std::uint64_t>(location.local_expert)
                            * projection->out_per_expert * (mx->columns / 32);
                    const auto scale = std::find_if(targets.begin(), targets.end(),
                        [&](const auto& range) { return range.offset == scale_offset; });
                    if (scale == targets.end()) throw std::logic_error("deferred MX scale range is missing");
                    if (std::find(scale->destination, scale->destination + scale->bytes, std::uint8_t{255})
                        != scale->destination + scale->bytes)
                        throw std::runtime_error("MFE MXFP4 contains an E8M0 NaN scale");
                }
                if (!complete) {
                    parsed.emplace(parse());
                }
                pages[request.page].value = std::make_shared<Impl::MfeCachedExpert>(
                    request.expert, std::move(*parsed), request.destination.has_value());
                request.prepare_seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - started).count() - request.reads.seconds;
            }
        }};

    const auto view_started = std::chrono::steady_clock::now();
    std::vector<MlxMfeWeight> weights;
    weights.reserve(pages.size());
    for (const auto& page : pages) {
        weights.push_back(page.value->weight);
    }
    std::optional<Impl::PreparedProjection> bank;
    const auto assemble = [&](const std::vector<MlxMfeWeight>& values) {
        return MlxMfeWeight::concatenate_expert_pages(values, projection->kernel_profile);
    };
    if (prepare) {
        std::vector<MlxMfeWeight> bank_weights;
        bank_weights.reserve(bank_pages.size());
        std::unordered_map<std::int32_t, std::int32_t> local;
        for (const auto& page : bank_pages) {
            local.emplace(page->expert, static_cast<std::int32_t>(bank_weights.size()));
            bank_weights.push_back(page->weight);
        }
        bank.emplace(Impl::PreparedProjection{
            assemble(bank_weights)
                .with_kernel_profile(projection->kernel_profile), std::move(local)});
    }
    auto result = [&] {
        if (!bank) return assemble(weights)
            .with_kernel_profile(projection->kernel_profile);
        std::vector<std::int32_t> local;
        for (const auto& key : ordered_keys) local.push_back(bank->local_ids.at(key.expert));
        return bank->weight.select_experts(local);
    }();
    if (!bank && pages.size() > 1 && !indirect_pages) {
        std::unordered_map<std::int32_t, std::int32_t> local;
        for (std::size_t index = 0; index < pages.size(); ++index)
            local.emplace(pages[index].key.expert, static_cast<std::int32_t>(index));
        bank.emplace(Impl::PreparedProjection{result, std::move(local)});
    }

    const auto view_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - view_started).count();
    lock.lock();
    impl_->stats.view_seconds += view_seconds;
    for (const auto& request : pending) {
        impl_->stats.bytes_read += request.reads.bytes;
        impl_->stats.read_calls += request.reads.calls;
        impl_->stats.parallel_body_read_batches += request.reads.parallel_body_batches;
        impl_->stats.io_seconds += request.reads.seconds;
        impl_->stats.prepare_seconds += request.prepare_seconds;
    }
    std::unordered_map<std::int32_t, std::int32_t> result_ids;
    for (std::size_t index = 0; index < pages.size(); ++index) {
        result_ids.emplace(pages[index].key.expert, static_cast<std::int32_t>(index));
        in_flight->pages.push_back(pages[index].value);
    }
    in_flight->result.emplace(Impl::PreparedProjection{result, std::move(result_ids)});
    impl_->stats.expert_loads += pending.size();
    if (epoch != impl_->epoch) co_return result;

    if (std::any_of(pages.begin(), pages.end(), [&](const auto& page) {
        return !page.value->transient && !impl_->mfe_cache.contains(page.key);
    })) impl_->drop_cached_view(name);

    impl_->mfe_cache.reserve(checked_add(
        impl_->mfe_cache.size(),
        static_cast<std::size_t>(std::count_if(
            pages.begin(), pages.end(),
            [](const auto& page) { return page.staged; })),
        "resident MFE cache entries"));
    for (const auto& page : pages) {
        if (page.value->transient) continue;
        if (impl_->mfe_cache.contains(page.key)) continue;
        auto& by_id = impl_->resident_pages[name];
        if (by_id.empty()) by_id.resize(projection->experts);
        const auto bytes = checked_add(impl_->resident_bytes,
            page.value->packed_nbytes, "resident MFE expert bytes");
        const auto inserted = prefill_scan && !priority_scan
            ? impl_->mfe_lru.insert(impl_->mfe_lru.begin(), {page.key, page.value})
            : impl_->mfe_lru.insert(impl_->mfe_lru.end(), {page.key, page.value});
        try {
            if (!impl_->mfe_cache.emplace(inserted->key, inserted).second)
                throw std::logic_error("duplicate staged MFE expert cache key");
        } catch (...) {
            impl_->mfe_lru.erase(inserted);
            throw;
        }
        impl_->resident_bytes = bytes;
        by_id[page.key.expert] = page.value;
    }
    for (const auto& page : pages) touch(page.key, !page.staged);
    if (bank && bank->weight.packed_nbytes() <= impl_->cache_limit
        - std::min(impl_->cache_limit, impl_->resident_bytes)) {
        const auto bytes = bank->weight.packed_nbytes();
        impl_->prepared.emplace(name, std::move(*bank));
        impl_->resident_bytes += bytes;
    }
    impl_->trim();
    for (const auto& page : pages) {
        const auto cached = impl_->mfe_cache.find(page.key);
        if (cached == impl_->mfe_cache.end()) continue;
        auto& value = *cached->second;
        if (!value.pin_attempted) value.pin_attempted = impl_->enqueue_pin(page.key, value.weight);
    }
    co_return result;
}

MlxMfeWeight MlxMfeOffloadCache::grouped_mfe(
    const std::string& name,
    const std::vector<std::int32_t>& active_experts,
    std::span<const std::uint32_t> activation_counts, bool prefill_scan) {
    const std::array<std::string, 1> names{name};
    return grouped_mfe(names, active_experts, activation_counts, prefill_scan).front();
}

std::vector<MlxMfeWeight> MlxMfeOffloadCache::grouped_mfe(
    std::span<const std::string> names,
    const std::vector<std::int32_t>& active_experts,
    std::span<const std::uint32_t> activation_counts, bool prefill_scan,
    const std::function<void(const std::vector<MlxMfeWeight>&)>& prefix_ready) {
    if (names.empty()) throw std::invalid_argument("streamed MFE projections cannot be empty");
    std::unordered_set<std::string> unique;
    for (const auto& name : names)
        if (!unique.insert(name).second)
            throw std::invalid_argument("streamed MFE projections must be distinct");
    const auto workers = std::max(1u, std::thread::hardware_concurrency());
    if (prefix_ready) {
        if (names.size() != 3)
            throw std::invalid_argument("streamed MFE prefix submission requires three projections");
        bool parallel = !prefill_scan;
        std::array<std::size_t, 2> missing{};
        if (parallel) {
            std::lock_guard lock(impl_->mutex);
            for (std::size_t index = 0; index < names.size(); ++index) {
                (void)impl_->mfe_projection_locked(names[index]);
                for (const auto expert : active_experts)
                    missing[index == 2] += !impl_->mfe_cache.contains({names[index], expert});
            }
            parallel = missing[0] && missing[1] && missing[0] + missing[1] <= std::min(8u, workers);
        }
        if (!parallel) {
            auto result = grouped_mfe(names.first(2), active_experts, activation_counts, prefill_scan);
            prefix_ready(result);
            result.push_back(grouped_mfe(names.back(), active_experts, activation_counts, prefill_scan));
            return result;
        }
    }
    std::vector<MfeGroupTask> tasks;
    std::vector<MfeGroupWork*> jobs;
    std::vector<std::size_t> offsets{0};
    std::vector<std::size_t> order(names.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](auto left, auto right) { return names[left] < names[right]; });
    tasks.reserve(names.size());
    for (const auto index : order) {
        tasks.push_back(impl_->prepare_grouped(names[index], active_experts, activation_counts, prefill_scan));
        tasks.back().resume();
        if (auto& work = tasks.back().handle.promise().work; work) {
            jobs.push_back(&*work);
            offsets.push_back(checked_add(offsets.back(), work->count, "MFE grouped expert tasks"));
        }
    }
    // Prefill has wide CPU parsing work; small route sets mostly contend on
    // the allocator's resource/residency lock once the I/O queue is saturated.
    if (prefix_ready) {
        std::vector<std::pair<std::size_t, MfeGroupWork*>> prioritized;
        for (std::size_t index = 0; index < tasks.size(); ++index)
            if (auto& work = tasks[index].handle.promise().work; work)
                prioritized.emplace_back(order[index], &*work);
        std::sort(prioritized.begin(), prioritized.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });
        jobs.clear();
        offsets.assign(1, 0);
        std::size_t prefix_jobs = 0;
        for (const auto& [index, job] : prioritized) {
            jobs.push_back(job);
            offsets.push_back(checked_add(offsets.back(), job->count, "MFE grouped expert tasks"));
            if (index < 2) ++prefix_jobs;
        }
        std::atomic<std::size_t> prefix_left{offsets[prefix_jobs]};
        std::mutex ready_mutex;
        std::condition_variable ready_condition;
        bool io_done = false;
        auto background = std::async(std::launch::async, [&] {
            struct Finish {
                std::mutex& mutex;
                std::condition_variable& condition;
                bool& done;
                ~Finish() {
                    { std::lock_guard lock(mutex); done = true; }
                    condition.notify_all();
                }
            } finish{ready_mutex, ready_condition, io_done};
            impl_->io_workers.run(0, offsets.back(), 1,
                static_cast<int>(std::min<std::size_t>(offsets.back(), std::min(8u, workers))),
                [&](std::int64_t begin, std::int64_t end) {
                    while (begin < end) {
                        const auto index = std::upper_bound(offsets.begin(), offsets.end(),
                            static_cast<std::size_t>(begin)) - offsets.begin() - 1;
                        const auto next = std::min<std::int64_t>(end, offsets[index + 1]);
                        jobs[index]->load(begin - offsets[index], next - offsets[index]);
                        const auto count = static_cast<std::size_t>(next - begin);
                        if (index < prefix_jobs && prefix_left.fetch_sub(count) == count) {
                            std::lock_guard lock(ready_mutex);
                            ready_condition.notify_all();
                        }
                        begin = next;
                    }
                });
        });
        {
            std::unique_lock lock(ready_mutex);
            ready_condition.wait(lock, [&] { return prefix_left.load() == 0 || io_done; });
        }
        if (prefix_left.load() != 0) background.get();
        std::array<std::optional<MlxMfeWeight>, 2> selected;
        for (std::size_t index = 0; index < tasks.size(); ++index) if (order[index] < 2) {
            auto& task = tasks[index];
            if (task.handle.promise().work) task.resume();
            if (!task.handle.promise().result)
                throw std::logic_error("streamed MFE prefix preparation did not finish");
            selected[order[index]] = *task.handle.promise().result;
        }
        const std::vector<MlxMfeWeight> prefix{*selected[0], *selected[1]};
        prefix_ready(prefix);
        bool submitted_early = false;
        { std::lock_guard lock(ready_mutex); submitted_early = !io_done; }
        if (prefix_jobs && prefix_jobs < jobs.size()) {
            std::lock_guard lock(impl_->mutex);
            ++impl_->stats.overlapped_read_batches;
            impl_->stats.overlapped_read_submissions += submitted_early;
        }
        background.get();
    } else impl_->io_workers.run(0, offsets.back(), 1,
        static_cast<int>(std::min(std::max<std::size_t>(1, offsets.back()),
            std::size_t(prefill_scan ? workers : std::min(kMfeSmallMReadWorkers, workers)))),
        [&](std::int64_t begin, std::int64_t end) {
            while (begin < end) {
                const auto index = std::upper_bound(offsets.begin(), offsets.end(),
                    static_cast<std::size_t>(begin)) - offsets.begin() - 1;
                const auto next = std::min<std::int64_t>(end, offsets[index + 1]);
                jobs[index]->load(begin - offsets[index], next - offsets[index]);
                begin = next;
            }
        });
    std::vector<std::optional<MlxMfeWeight>> ordered(tasks.size());
    for (std::size_t index = 0; index < tasks.size(); ++index) {
        auto& task = tasks[index];
        if (task.handle.promise().work) task.resume();
        if (!task.handle.promise().result)
            throw std::logic_error("streamed MFE preparation did not finish");
        ordered[order[index]].emplace(std::move(*task.handle.promise().result));
    }
    std::vector<MlxMfeWeight> result;
    result.reserve(ordered.size());
    for (auto& weight : ordered) result.push_back(std::move(*weight));
    return result;
}

std::size_t
MlxMfeOffloadCache::cache_limit_bytes() const noexcept {
    std::lock_guard lock(impl_->mutex);
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

MlxMfeCacheReuseStats MlxMfeOffloadCache::reuse_stats() const {
    std::lock_guard lock(impl_->mutex);
    return impl_->stats;
}

std::size_t MlxMfeOffloadCache::set_cache_limit(std::size_t bytes) {
    std::lock_guard lock(impl_->mutex);
    const auto before = impl_->resident_bytes;
    if (impl_->cache_limit != bytes) {
        impl_->cache_limit = bytes;
        ++impl_->epoch;
    }
    impl_->trim();
    return before - impl_->resident_bytes;
}

void MlxMfeOffloadCache::set_prefill_priority(std::span<const std::string> names) {
    std::lock_guard lock(impl_->mutex);
    std::unordered_set<std::string> priority(names.begin(), names.end());
    if (priority == impl_->prefill_priority) return;
    impl_->prefill_priority = std::move(priority);
    if (impl_->prefill_priority.empty()) return;
    auto item = impl_->mfe_lru.begin();
    for (std::size_t count = impl_->mfe_lru.size(); count > 0; --count) {
        const auto current = item++;
        const bool preferred = impl_->prefill_priority.contains(current->key.name);
        current->credit = preferred ? 3 : 0;
        if (preferred) impl_->mfe_lru.splice(impl_->mfe_lru.end(), impl_->mfe_lru, current);
    }
}

void MlxMfeOffloadCache::discard_record(
    const std::string& name) noexcept {
    std::lock_guard lock(impl_->mutex);
    ++impl_->epoch;
    impl_->drop_prepared(name);
    impl_->drop_cached_view(name);
    impl_->route_confidence.erase(name);
    for (auto item = impl_->mfe_lru.begin();
         item != impl_->mfe_lru.end();) {
        if (item->key.name != name) {
            ++item;
            continue;
        }
        impl_->resident_bytes -= item->weight->packed_nbytes;
        impl_->cancel_pin(item->key);
        impl_->mfe_cache.erase(item->key);
        item = impl_->mfe_lru.erase(item);
    }
    impl_->mfe_projections.erase(name);
    impl_->resident_pages.erase(name);
    impl_->backing_records.erase(name);
}

void MlxMfeOffloadCache::clear() {
    std::lock_guard lock(impl_->mutex);
    ++impl_->epoch;
    impl_->prepared.clear();
    impl_->cached_views.clear();
    ++impl_->page_generation;
    impl_->route_confidence.clear();
    impl_->resident_pages.clear();
    {
        std::lock_guard pin_lock(impl_->pin_mutex);
        impl_->pin_queue.clear();
        impl_->pin_pending.clear();
    }
    impl_->mfe_cache.clear();
    impl_->mfe_lru.clear();
    impl_->mfe_projections.clear();
    impl_->resident_bytes = 0;
}

struct MlxMfeWeight::Impl {
    array descriptors;
    std::shared_ptr<const MfePageResources> page_resources;
    std::optional<array> indirect_addresses;
    std::vector<array> indirect_owners;
    std::shared_ptr<const std::vector<MlxMfeWeight>> indirect_sources;
    std::optional<array> packed_arena;
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
    // Metal kernels address packed pools through descriptor offsets, so their
    // logical MLX shape is not part of the ABI. Prefix views retain the same
    // backing buffer and base address while preventing MLX's command-buffer
    // heuristic from counting every resident expert as touched by each
    // decode launch.
    std::vector<array> narrow_bindings;
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
    std::shared_ptr<const std::vector<int>> nint_group_sizes;
    std::shared_ptr<const MlxMfeKernelProfile> kernel_profile;
    std::shared_ptr<const std::vector<std::tuple<int, int, int>>> cohort_populations;
    bool vq_execution_layout = false;
    std::uint64_t vq_layouts = 0;
    bool npq_grouped_indices = true;
    bool native_primitive = true;
    bool grouped_mmq = false;
    bool has_nepq_residual = false;
    std::optional<array> mxfp4_slot_ids;
    bool mxfp4_slot_ids_sorted = false;
    bool automatic_mxfp4_nax_prefill = true;
    int k_lanes_override = 0;
    bool nint_q_dispatch = true;
    std::size_t packed_bytes = 0;

    MlxMfeKernelProfile compilation_profile() const {
        return kernel_profile ? *kernel_profile
            : MlxMfeKernelProfile{family_mask, vq_profile_mask, vq_layouts,
                nint_group_sizes, vq_execution_layout, nint_q_dispatch};
    }

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
        std::vector<int> group_sizes;
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
            if (family == kFamilyNint) {
                nint_q_dispatch &= descriptor_values[base + kNintQDispatch] != 0;
                const auto group_size = descriptor_values[base + kNintGroupSize];
                if (group_size > 0 && std::find(group_sizes.begin(), group_sizes.end(), group_size)
                    == group_sizes.end()) group_sizes.push_back(group_size);
            }
            if (family == kFamilyVq) {
                const auto profile = descriptor_values[
                    base + kVqProfile] & 255;
                if (profile >= 0 && profile < 9) {
                    const auto shift = static_cast<unsigned>(profile) * 7u;
                    const auto layout = descriptor_values[base + kVqExecutionLayout];
                    vq_layouts |= (layout >= 0 && layout < 7
                        ? std::uint64_t{1} << layout : std::uint64_t{127}) << shift;
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
                    base + kVqExecutionLayout] != 0
            ) {
                vq_execution_layout = true;
            }
        }
        std::sort(group_sizes.begin(), group_sizes.end());
        nint_group_sizes = std::make_shared<const std::vector<int>>(std::move(group_sizes));
        std::vector<std::tuple<int, int, int>> populations;
        if (projections == 1) {
            for (int family = 0; family < 7; ++family) {
                if ((family_mask & (std::uint32_t{1} << family)) == 0) continue;
                const auto groups = family == kFamilyNint ? *nint_group_sizes : std::vector<int>{0};
                for (const int group : groups) {
                    int count = 0;
                    for (std::size_t base = 0; base + kDescriptorSize <= descriptor_values.size(); base += kDescriptorSize)
                        if (descriptor_values[base + kFamily] == family
                            && (group == 0 || descriptor_values[base + kNintGroupSize] == group)) ++count;
                    populations.emplace_back(1 << family, group, count);
                }
            }
        }
        cohort_populations = std::make_shared<const std::vector<std::tuple<int, int, int>>>(std::move(populations));
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
            vq_profile_mask = 511;
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
            if (value == "16") {
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

        const auto narrow = [](const array& source) {
            if (source.ndim() == 0 || source.size() <= 1) {
                return source;
            }
            Shape begin(static_cast<std::size_t>(source.ndim()), 0);
            Shape end = source.shape();
            end.front() = 1;
            return mlx::core::slice(source, std::move(begin), std::move(end));
        };
        narrow_bindings = {
            descriptors,
            narrow(nint_q),
            narrow(nint_sub_scale),
            narrow(nint_sub_min),
            narrow(nint_anchor_scale),
            narrow(nint_anchor_min),
            narrow(q8_q),
            narrow(q8_scales),
            narrow(vq_indices),
            narrow(vq_state),
            narrow(vq_aux),
            narrow(vq_anchors),
            narrow(vq_codebooks),
            narrow(vq_scales),
            narrow(vq_state_to_codebank),
            narrow(vq_banks),
            narrow(vq_parameters),
            narrow(vq_residual_codebooks),
            narrow(vq_residual_first),
            narrow(vq_residual_second),
            narrow(mx_values),
            narrow(mx_scales),
        };
    }

    void bind_projection_views(const std::vector<ProjectionStorageInfo>& sources) {
        projection_views.clear();
        projection_views.reserve(sources.size());
        for (std::size_t projection = 0; projection < sources.size(); ++projection) {
            std::vector<std::int32_t> values;
            values.reserve(static_cast<std::size_t>(experts) * kDescriptorSize);
            for (int expert = 0; expert < experts; ++expert) {
                const auto begin = descriptor_values.begin() +
                    (static_cast<std::size_t>(expert) * sources.size() + projection) * kDescriptorSize;
                values.insert(values.end(), begin, begin + kDescriptorSize);
            }
            auto view = std::make_shared<Impl>(
                make_int32_array(values, Shape{experts, kDescriptorSize}),
                nint_q, nint_sub_scale, nint_sub_min, nint_anchor_scale, nint_anchor_min,
                q8_q, q8_scales, vq_indices, vq_state, vq_aux, vq_anchors,
                vq_codebooks, vq_scales, vq_state_to_codebank, vq_banks, vq_parameters,
                vq_residual_codebooks, vq_residual_first, vq_residual_second,
                mx_values, mx_scales, rotations, std::move(values),
                experts, out_per_expert, neuron_len, 1);
            view->automatic_mxfp4_nax_prefill = sources[projection].automatic_mxfp4_nax_prefill;
            view->logical_experts = sources[projection].logical_experts;
            view->packed_bytes = sources[projection].packed_bytes;
            projection_views.push_back(std::move(view));
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
    if (auto* preparation = MlxKernelPreparation::current(); preparation && !preparation->model_paths() && preparation->routes() > 0
        && impl_->reference_cohorts.empty() && !mlx_reference_enabled()) {
        preparation->collect([&] {
            const int routes = preparation->routes();
            const auto prepare = [&](const MlxMfeWeight& weight, bool projection_view) {
                for (const auto rows : preparation->row_buckets()) {
                    if (projection_view && rows > 8) continue;
                    const auto ids = mlx::core::zeros(Shape{rows, routes}, mlx::core::int32);
                    const auto shared = mlx::core::zeros(Shape{rows, weight.neuron_len()}, mlx::core::float16);
                    const auto routed = mlx::core::zeros(Shape{rows, routes, weight.neuron_len()}, mlx::core::float16);
                    preparation->add(weight.routed_matmul(shared, ids));
                    if (!projection_view) preparation->add(weight.routed_matmul(routed, ids));
                    if (weight.projections() == 2) preparation->add(weight.routed_swiglu(shared, ids));
                    if (weight.projections() == 1 && rows > 1 && rows <= 24) {
                        const auto groups = mlx::core::zeros(Shape{rows, routes, 8}, mlx::core::int32);
                        preparation->add(weight.routed_matmul(shared, ids, &groups));
                        if (!projection_view) preparation->add(weight.routed_matmul(routed, ids, &groups));
                    }
                    if (!projection_view && weight.supports_fused_routed_reduce()) preparation->add(weight.routed_matmul_reduce(routed, ids,
                        mlx::core::zeros(Shape{rows, routes}, mlx::core::float32)));
                }
            };
            prepare(*this, false);
            for (const auto& view : impl_->projection_views) prepare(MlxMfeWeight(view), true);
        });
    }
}

MlxMfeWeight MlxMfeWeight::from_blob(
    std::span<const std::uint8_t> blob) {
    const std::array blobs{blob};
    return from_projection_blobs(blobs);
}

MlxMfeWeight MlxMfeWeight::from_blob(const array& blob) {
    if (blob.dtype() != mlx::core::uint8 || !blob.flags().row_contiguous)
        throw std::invalid_argument("MFE wire storage must be contiguous bytes");
    auto storage = blob;
    storage.eval();
    const std::array blobs{std::span<const std::uint8_t>(storage.data<std::uint8_t>(), storage.nbytes())};
    return from_projection_blobs_impl(blobs, false, &storage);
}

MlxMfeWeight MlxMfeWeight::from_projection_blobs(
    std::span<const std::span<const std::uint8_t>> blobs) {
    return from_projection_blobs_impl(blobs, false);
}

MlxMfeWeight MlxMfeWeight::from_nint_blob(
    std::span<const std::uint8_t> blob) {
    const std::array blobs{blob};
    return from_projection_blobs_impl(blobs, true);
}

MlxMfeWeight MlxMfeWeight::from_nint_blob(const array& blob) {
    if (blob.dtype() != mlx::core::uint8 || !blob.flags().row_contiguous)
        throw std::invalid_argument("NINT wire storage must be contiguous bytes");
    auto storage = blob;
    storage.eval();
    const std::array blobs{std::span<const std::uint8_t>(storage.data<std::uint8_t>(), storage.nbytes())};
    return from_projection_blobs_impl(blobs, true, &storage);
}

MlxMfeWeight MlxMfeWeight::from_projection_blobs_impl(
    std::span<const std::span<const std::uint8_t>> blobs,
    bool nint_tensor,
    const array* storage,
    const array* destination) {
    if (blobs.empty()) {
        throw std::invalid_argument("at least one MFE projection is required");
    }
    const int projection_count = checked_int(blobs.size(), "projection count");
    std::vector<MfeProjectionView> projections;
    projections.reserve(blobs.size());
    for (const auto blob : blobs) {
        projections.push_back(nint_tensor
            ? inspect_nint_projection(blob) : inspect_mfe_projection(blob));
    }
    const int expert_count = projections.front().experts;
    const int output_width = projections.front().output;
    const int input_width = projections.front().input;
    for (const auto& projection : projections) {
        if (projection.experts != expert_count || projection.output != output_width ||
            projection.input != input_width) {
            throw std::invalid_argument("MFE projections have incompatible shapes");
        }
    }
    const bool reference = mlx_reference_enabled();
    if (projection_count > 1 && (reference || std::any_of(
        projections.begin(), projections.end(), [](const auto& projection) {
            return std::any_of(projection.pools.begin(), projection.pools.end(),
                [](const auto& pool) {
                    return !pool.runtime.empty() || is_mxfp4_sq_dtype(pool.dtype) ||
                        is_fp8_sq_dtype(pool.dtype);
                });
        }))) {
        // Standalone SQ/reference and rotated cohorts keep their established
        // ownership/dispatch path, without parsing or repacking weights twice.
        std::vector<MlxMfeWeight> weights;
        weights.reserve(blobs.size());
        for (const auto blob : blobs) weights.push_back(from_blob(blob));
        return concatenate_projections(weights).materialize_packed_projections();
    }

    const int descriptor_rows = checked_int(checked_product(
        static_cast<std::size_t>(expert_count), blobs.size(), "projection rows"),
        "projection rows");
    std::vector<std::int32_t> descriptors(checked_product(
        static_cast<std::size_t>(descriptor_rows),
        static_cast<std::size_t>(kDescriptorSize), "descriptor count"), 0);
    PackedStreams streams;
    if (storage) streams.retain_source(*storage);
    std::vector<RotationSpec> rotations;
    std::vector<ReferenceMoeCohort> reference_cohorts;
    std::vector<Mxfp4SqMoeCohort> mxfp4_sq_cohorts;
    std::vector<Fp8SqMoeCohort> fp8_sq_cohorts;
    std::vector<std::int32_t> standalone_owner(
        static_cast<std::size_t>(expert_count), -1);
    std::vector<ProjectionStorageInfo> projection_info;
    bool has_generic_cohorts = true;

    for (std::size_t projection = 0; projection < projections.size(); ++projection) {
        if ((streams.q8_q.size() & 1u) != 0u) streams.q8_q.push_back(0);
        const auto before = streams.size();
        std::vector<std::int32_t> local_descriptors(checked_product(
            static_cast<std::size_t>(expert_count), static_cast<std::size_t>(kDescriptorSize),
            "descriptor count"), 0);
        bool projection_generic = false;
        for (const auto& pool : projections[projection].pools) {
            const auto& expert_ids = pool.expert_ids;
            const auto count = expert_ids.size();
            const auto& dtype = pool.dtype;
            const auto runtime = pool.runtime;
            const auto payload = pool.payload;
            if (is_nint_dtype(dtype)) {
                projection_generic = true;
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
                    local_descriptors,
                    !reference);
                if (reference) {
                    reference_cohorts.push_back({
                        expert_ids,
                        std::move(*weight),
                    });
                }
            } else if (is_nint8_zero_dtype(dtype)) {
                projection_generic = true;
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
                    local_descriptors,
                    !reference);
                if (reference) {
                    reference_cohorts.push_back({
                        expert_ids,
                        std::move(weight),
                    });
                }
            } else if (dtype == "MXFP4" || dtype == "MXFP8") {
                projection_generic = true;
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
                    local_descriptors,
                    !reference);
                if (reference) {
                    reference_cohorts.push_back({
                        expert_ids,
                        std::move(*weight),
                    });
                }
            } else if (dtype == "BF16" || dtype == "F16") {
                projection_generic = true;
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
                    local_descriptors,
                    !reference);
                if (reference) {
                    reference_cohorts.push_back({
                        expert_ids,
                        std::move(*weight),
                    });
                }
            } else if (is_vq_dtype(dtype)) {
                projection_generic = true;
                auto weight = add_vq_pool(
                    dtype,
                    payload,
                    runtime,
                    expert_ids,
                    output_width,
                    input_width,
                    streams,
                    rotations,
                    local_descriptors,
                    !reference);
                if (reference) {
                    reference_cohorts.push_back({
                        expert_ids,
                        std::move(*weight),
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
        has_generic_cohorts = has_generic_cohorts && projection_generic;
        projection_info.push_back({
            checked_add(streams.size() - before,
                local_descriptors.size() * sizeof(std::int32_t), "projection bytes"),
            true, 0,
        });
        for (int expert = 0; expert < expert_count; ++expert) {
            const auto source = static_cast<std::size_t>(expert) * kDescriptorSize;
            const auto target = (static_cast<std::size_t>(expert) * projections.size()
                + projection) * kDescriptorSize;
            std::copy_n(local_descriptors.begin() + source, kDescriptorSize,
                descriptors.begin() + target);
        }
    }

    const Shape descriptor_shape{
        descriptor_rows,
        kDescriptorSize,
    };
    std::optional<array> arena = destination ? std::optional<array>(*destination) : std::nullopt;
    const auto stream_bytes = streams.size();
    const auto descriptor_bytes = checked_product(descriptors.size(), sizeof(std::int32_t), "MFE descriptor bytes");
    if (destination && (expert_count != 1 || reference || destination->dtype() != mlx::core::uint8
        || !destination->is_available() || !destination->flags().row_contiguous
        || destination->nbytes() < stream_bytes + 21 * 15 + descriptor_bytes + 15))
        throw std::invalid_argument("MFE expert staging arena is too small or incompatible");
    if (!destination && expert_count == 1 && !reference && stream_bytes > 0 && stream_bytes <= (std::size_t{8} << 20))
        arena = allocate_packed_array(stream_bytes + 21 * 15 + descriptor_bytes + 15, mlx::core::uint8);
    std::size_t arena_offset = arena ? descriptor_bytes : 0;
    const auto materialize = [&](PackedStream stream, Dtype dtype) {
        arena_offset = (arena_offset + 15) & ~std::size_t{15};
        const auto begin = arena_offset;
        arena_offset += stream.size();
        return std::move(stream).materialize(dtype, arena ? &*arena : nullptr, begin);
    };
    auto impl = std::make_shared<Impl>(
        make_int32_array(
            descriptors,
            descriptor_shape, arena ? &*arena : nullptr),
        materialize(
            std::move(streams.nint_q),
            mlx::core::uint8),
        materialize(
            std::move(streams.nint_sub_scale),
            mlx::core::uint8),
        materialize(
            std::move(streams.nint_sub_min),
            mlx::core::uint8),
        materialize(
            std::move(streams.nint_anchor_scale),
            mlx::core::float32),
        materialize(
            std::move(streams.nint_anchor_min),
            mlx::core::float32),
        materialize(
            std::move(streams.q8_q),
            mlx::core::int8),
        materialize(
            std::move(streams.q8_scales),
            mlx::core::float16),
        materialize(
            std::move(streams.vq_indices),
            mlx::core::uint8),
        materialize(
            std::move(streams.vq_state),
            mlx::core::uint8),
        materialize(
            std::move(streams.vq_aux),
            mlx::core::uint8),
        materialize(
            std::move(streams.vq_anchors),
            mlx::core::float32),
        materialize(
            std::move(streams.vq_codebooks),
            mlx::core::int8),
        materialize(
            std::move(streams.vq_scales),
            mlx::core::float32),
        materialize(
            std::move(
                streams.vq_state_to_codebank),
            mlx::core::uint8),
        materialize(
            std::move(streams.vq_banks),
            mlx::core::uint8),
        materialize(
            std::move(streams.vq_parameters),
            mlx::core::float32),
        materialize(
            std::move(streams.vq_residual_codebooks),
            mlx::core::float32),
        materialize(
            std::move(streams.vq_residual_first),
            mlx::core::int16),
        materialize(
            std::move(streams.vq_residual_second),
            mlx::core::int16),
        materialize(
            std::move(streams.mx_values),
            mlx::core::uint8),
        materialize(
            std::move(streams.mx_scales),
            mlx::core::uint8),
        std::move(rotations),
        std::move(descriptors),
        expert_count,
        output_width,
        input_width,
        projection_count);
    impl->packed_arena = arena;
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
    if (projection_count > 1) {
        impl->bind_projection_views(projection_info);
    }
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
    if (std::any_of(weights.begin(), weights.end(), [](const auto& weight) { return weight.impl_->indirect_sources != nullptr; })) {
        std::vector<MlxMfeWeight> packed;
        for (const auto& weight : weights)
            packed.push_back(weight.impl_->indirect_sources
                ? concatenate_experts(*weight.impl_->indirect_sources).with_kernel_profile(weight.impl_->kernel_profile) : weight);
        return concatenate_projections(packed);
    }
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
        // Preserve typed row-offset/group64 alignment when rebasing arenas.
        if ((source.family_mask & (1u << kFamilyNint)) != 0u) {
            align_packed_stream(nint_q_arrays, nint_q_offset, 16);
        }
        if ((source.family_mask & (1u << kFamilyVq)) != 0u) {
            align_packed_stream(vq_indices_arrays, vq_indices_offset, 8);
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
                    descriptor[kNintRowMetadataOffset] =
                        descriptor_with_offset(
                            descriptor[kNintRowMetadataOffset],
                            nint_q_offset,
                            "NINTv2 row-metadata offset");
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
    impl->has_generic_cohorts = std::all_of(
        weights.begin(),
        weights.end(),
        [](const MlxMfeWeight& weight) {
            return weight.impl_->has_generic_cohorts;
        });
    impl->logical_experts = first.logical_experts > 0
        ? first.logical_experts
        : first.experts;
    if (std::any_of(weights.begin(), weights.end(),
            [](const auto& weight) { return weight.impl_->kernel_profile != nullptr; })) {
        auto profile = std::make_shared<MlxMfeKernelProfile>();
        profile->nint_q_dispatch = true;
        std::vector<int> groups;
        for (const auto& weight : weights) {
            const auto source = weight.impl_->compilation_profile();
            profile->family_mask |= source.family_mask;
            profile->vq_profile_mask |= source.vq_profile_mask;
            profile->vq_layouts |= source.vq_layouts;
            profile->vq_execution_layout |= source.vq_execution_layout;
            profile->nint_q_dispatch &= source.nint_q_dispatch;
            groups.insert(groups.end(), source.nint_group_sizes->begin(), source.nint_group_sizes->end());
        }
        std::sort(groups.begin(), groups.end());
        groups.erase(std::unique(groups.begin(), groups.end()), groups.end());
        profile->nint_group_sizes = std::make_shared<const std::vector<int>>(std::move(groups));
        impl->kernel_profile = std::move(profile);
    }
    impl->projection_views.reserve(weights.size());
    impl->packed_bytes = 0;
    for (const auto& weight : weights) {
        // Keep each projection on its original packed pool.  Evaluating the
        // combined stream would otherwise copy every expert's Gate and Up
        // payload before a one-token decode can touch its selected experts.
        impl->projection_views.push_back(weight.impl_);
        impl->packed_bytes = checked_add(
            impl->packed_bytes,
            weight.impl_->packed_bytes,
            "split MFE projection bytes");
    }
    return MlxMfeWeight(std::move(impl));
}

MlxMfeWeight MlxMfeWeight::materialize_packed_projections() const {
    if (impl_->projections != 2 || impl_->projection_views.empty() ||
        !impl_->rotations.empty() ||
        std::any_of(impl_->projection_views.begin(), impl_->projection_views.end(),
            [](const auto& source) {
                return !source->reference_cohorts.empty() ||
                    !source->mxfp4_sq_cohorts.empty() || !source->fp8_sq_cohorts.empty();
            })) {
        return *this;
    }
    auto impl = std::make_shared<Impl>(*impl_);
    mlx::core::eval(std::vector<array>{
        impl->descriptors, impl->nint_q, impl->nint_sub_scale, impl->nint_sub_min,
        impl->nint_anchor_scale, impl->nint_anchor_min, impl->q8_q, impl->q8_scales,
        impl->vq_indices, impl->vq_state, impl->vq_aux, impl->vq_anchors,
        impl->vq_codebooks, impl->vq_scales, impl->vq_state_to_codebank, impl->vq_banks,
        impl->vq_parameters, impl->vq_residual_codebooks, impl->vq_residual_first,
        impl->vq_residual_second, impl->mx_values, impl->mx_scales,
    });
    // Preserve split dispatch and its arithmetic while releasing source pools.
    std::vector<ProjectionStorageInfo> sources;
    for (const auto& source : impl->projection_views) {
        sources.push_back({source->packed_bytes, source->automatic_mxfp4_nax_prefill,
            source->logical_experts});
    }
    impl->bind_projection_views(sources);
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

    if (weights.size() == 1) return weights.front();
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
        if ((source.family_mask & (1u << kFamilyNint)) != 0u) {
            align_packed_stream(nint_q_arrays, nint_q_offset, 16);
        }
        if ((source.family_mask & (1u << kFamilyVq)) != 0u) {
            align_packed_stream(vq_indices_arrays, vq_indices_offset, 8);
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
                descriptor[kNintRowMetadataOffset] = descriptor_with_offset(
                    descriptor[kNintRowMetadataOffset], nint_q_offset,
                    "NINTv2 row-metadata offset");
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

    const std::array<const std::vector<array>*, 21> stream_arrays{&nint_q_arrays, &nint_sub_scale_arrays, &nint_sub_min_arrays,
        &nint_anchor_scale_arrays, &nint_anchor_min_arrays, &q8_q_arrays, &q8_scale_arrays,
        &vq_indices_arrays, &vq_state_arrays, &vq_aux_arrays, &vq_anchor_arrays,
        &vq_codebook_arrays, &vq_scale_arrays, &vq_state_bank_arrays, &vq_bank_arrays,
        &vq_parameter_arrays, &vq_residual_codebook_arrays, &vq_residual_first_arrays,
        &vq_residual_second_arrays, &mx_value_arrays, &mx_scale_arrays};
    std::vector<array> sources;
    for (const auto& weight : weights) {
        if (weight.impl_->packed_arena) sources.push_back(*weight.impl_->packed_arena);
    }
    const auto execution_arena = pack_expert_streams(std::move(sources), stream_arrays);
    const auto host_stream_bytes = [](const std::vector<array>& values) {
        if (values.size() < 2) return std::size_t{0};
        const auto dtype = values.front().dtype();
        const auto empty = empty_packed_array(dtype).data_shared_ptr();
        bool all_empty = true;
        std::size_t bytes = 0;
        for (const auto& value : values) {
            if (value.dtype() != dtype || !value.flags().row_contiguous || !value.is_available())
                return std::size_t{0};
            bytes = checked_add(bytes, value.nbytes(), "expert metadata bytes");
            if (bytes > (std::size_t{128} << 10)) return std::size_t{0};
            all_empty &= value.data_shared_ptr() == empty;
        }
        return all_empty ? std::size_t{0} : bytes;
    };
    std::size_t metadata_bytes = 0;
    std::vector<std::pair<const std::vector<array>*, std::size_t>> metadata_streams;
    for (const auto* values : stream_arrays) {
        if (execution_arena) break;
        const auto bytes = host_stream_bytes(*values);
        if (!bytes) continue;
        metadata_streams.emplace_back(values, bytes);
        metadata_bytes = checked_add(metadata_bytes, bytes, "expert metadata arena");
    }
    const auto descriptor_bytes = checked_product(descriptors.size(), sizeof(std::int32_t), "expert descriptors");
    std::optional<array> metadata_arena;
    if (metadata_bytes)
        metadata_arena = allocate_packed_array(metadata_bytes + descriptor_bytes + 22 * 15, mlx::core::uint8);
    std::size_t metadata_offset = descriptor_bytes;
    const auto concatenate_or_empty = [&](
        std::vector<array>&& values,
        Dtype dtype) {
        if (execution_arena) {
            const auto found = std::find_if(execution_arena->streams.begin(), execution_arena->streams.end(),
                [&](const auto& entry) { return entry.values == &values; });
            if (found == execution_arena->streams.end()) return empty_packed_array(dtype);
            return mlx::core::view(mlx::core::slice(execution_arena->storage,
                {static_cast<int>(found->offset)}, {static_cast<int>(found->offset + found->bytes)}), dtype);
        }
        if (metadata_arena) {
            const auto found = std::find_if(metadata_streams.begin(), metadata_streams.end(),
                [&](const auto& entry) { return entry.first == &values; });
            if (found != metadata_streams.end()) {
                PackedStream stream;
                for (auto& value : values) stream.append(std::move(value));
                metadata_offset = (metadata_offset + 15) & ~std::size_t{15};
                const auto begin = metadata_offset;
                metadata_offset += found->second;
                return std::move(stream).materialize(dtype, &*metadata_arena, begin);
            }
        }
        return values.empty()
            ? make_raw_array(
                  std::vector<std::uint8_t>{},
                  dtype)
            : concatenate_1d(std::move(values));
    };
    auto impl = std::make_shared<Impl>(
        make_int32_array(
            descriptors,
            Shape{expert_count, kDescriptorSize}, metadata_arena ? &*metadata_arena : nullptr),
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
    impl->has_generic_cohorts = std::all_of(weights.begin(), weights.end(),
        [](const auto& weight) { return weight.impl_->has_generic_cohorts; });
    if (execution_arena) impl->packed_arena = execution_arena->storage;
    impl->automatic_mxfp4_nax_prefill = std::all_of(
        weights.begin(),
        weights.end(),
        [](const MlxMfeWeight& weight) {
            return weight.impl_->automatic_mxfp4_nax_prefill;
        });
    return MlxMfeWeight(std::move(impl));
}

MlxMfeWeight MlxMfeWeight::concatenate_expert_pages(const std::vector<MlxMfeWeight>& weights,
    std::shared_ptr<const MlxMfeKernelProfile> profile) {
    if (weights.empty() || mlx_reference_enabled()) return concatenate_experts(weights);
    const auto streams = [](const Impl& value) {
        return std::array<const array*, 21>{&value.nint_q, &value.nint_sub_scale, &value.nint_sub_min,
            &value.nint_anchor_scale, &value.nint_anchor_min, &value.q8_q, &value.q8_scales,
            &value.vq_indices, &value.vq_state, &value.vq_aux, &value.vq_anchors,
            &value.vq_codebooks, &value.vq_scales, &value.vq_state_to_codebank, &value.vq_banks,
            &value.vq_parameters, &value.vq_residual_codebooks, &value.vq_residual_first,
            &value.vq_residual_second, &value.mx_values, &value.mx_scales};
    };
    const auto& first = *weights.front().impl_;
    std::uint32_t families = 0;
    for (const auto& weight : weights) {
        const auto& value = *weight.impl_;
        if (!value.native_primitive || value.projections != 1 || value.experts != 1
            || value.out_per_expert != first.out_per_expert || value.neuron_len != first.neuron_len
            || !value.reference_cohorts.empty() || !value.mxfp4_sq_cohorts.empty()
            || !value.fp8_sq_cohorts.empty() || value.standalone_owner || value.indirect_sources)
            return concatenate_experts(weights);
        for (const auto* stream : streams(value))
            if (!stream->is_available() || !stream->flags().row_contiguous) return concatenate_experts(weights);
        families |= value.family_mask;
    }
    if (!uses_indirect_expert_pages(profile ? profile->family_mask : families)) return concatenate_experts(weights);
    std::vector<std::int32_t> descriptors;
    std::vector<std::uint64_t> addresses;
    std::vector<array> owners;
    std::unordered_set<const MTL::Buffer*> buffers;
    std::vector<RotationSpec> rotations;
    std::size_t bytes = 0;
    for (const auto& weight : weights) {
        const auto& value = *weight.impl_;
        const auto begin = descriptors.size();
        descriptors.insert(descriptors.end(), value.descriptor_values.begin(), value.descriptor_values.end());
        const int variant = descriptors[begin + kVqRotationVariant];
        if (variant) descriptors[begin + kVqRotationVariant] = rotation_variant(value.rotations.at(variant - 1), rotations);
        for (const auto* stream : streams(value)) {
            auto* buffer = static_cast<const MTL::Buffer*>(stream->buffer().ptr());
            addresses.push_back(buffer->gpuAddress() + stream->offset());
            if (buffers.insert(buffer).second) owners.push_back(*stream);
        }
        bytes = checked_add(bytes, weight.packed_nbytes(), "indirect expert bytes");
    }
    const int experts = checked_int(weights.size(), "indirect expert count");
    auto impl = std::make_shared<Impl>(make_int32_array(descriptors, Shape{experts, kDescriptorSize}),
        first.nint_q, first.nint_sub_scale, first.nint_sub_min, first.nint_anchor_scale, first.nint_anchor_min,
        first.q8_q, first.q8_scales, first.vq_indices, first.vq_state, first.vq_aux, first.vq_anchors,
        first.vq_codebooks, first.vq_scales, first.vq_state_to_codebank, first.vq_banks, first.vq_parameters,
        first.vq_residual_codebooks, first.vq_residual_first, first.vq_residual_second,
        first.mx_values, first.mx_scales, std::move(rotations), std::move(descriptors),
        experts, first.out_per_expert, first.neuron_len, 1);
    impl->indirect_addresses.emplace(addresses.begin(), Shape{experts, 21}, mlx::core::uint64);
    impl->mxfp4_slot_ids.reset();
    impl->mxfp4_slot_ids_sorted = false;
    impl->indirect_owners = std::move(owners);
    impl->indirect_sources = std::make_shared<const std::vector<MlxMfeWeight>>(weights);
    impl->has_generic_cohorts = true;
    impl->kernel_profile = std::move(profile);
    impl->packed_bytes = checked_add(bytes, checked_add(impl->descriptors.nbytes(),
        impl->indirect_addresses->nbytes(), "indirect expert metadata"), "indirect expert storage");
    return MlxMfeWeight(std::move(impl));
}

MlxMfeWeight MlxMfeWeight::select_experts(
    std::span<const std::int32_t> experts) const {
    if (experts.empty() || impl_->projections != 1 || !impl_->projection_views.empty()
        || !impl_->reference_cohorts.empty() || !impl_->mxfp4_sq_cohorts.empty()
        || !impl_->fp8_sq_cohorts.empty() || impl_->standalone_owner)
        throw std::invalid_argument("MFE expert views require one native packed projection");
    std::vector<std::int32_t> values;
    values.reserve(experts.size() * kDescriptorSize);
    bool identity = experts.size() == static_cast<std::size_t>(impl_->experts);
    for (std::size_t index = 0; index < experts.size(); ++index) {
        const auto expert = experts[index];
        if (expert < 0 || expert >= impl_->experts)
            throw std::out_of_range("MFE view expert ID is out of range");
        identity &= expert == static_cast<std::int32_t>(index);
        const auto first = impl_->descriptor_values.begin() + expert * kDescriptorSize;
        values.insert(values.end(), first, first + kDescriptorSize);
    }
    if (identity) return *this;
    if (impl_->indirect_sources) {
        std::vector<MlxMfeWeight> pages;
        for (const auto expert : experts) pages.push_back(impl_->indirect_sources->at(expert));
        return concatenate_expert_pages(pages, impl_->kernel_profile);
    }
    auto descriptors = make_int32_array(values,
        Shape{checked_int(experts.size(), "MFE view experts"), kDescriptorSize});
    auto impl = std::make_shared<Impl>(std::move(descriptors),
        impl_->nint_q, impl_->nint_sub_scale, impl_->nint_sub_min,
        impl_->nint_anchor_scale, impl_->nint_anchor_min, impl_->q8_q, impl_->q8_scales,
        impl_->vq_indices, impl_->vq_state, impl_->vq_aux, impl_->vq_anchors,
        impl_->vq_codebooks, impl_->vq_scales, impl_->vq_state_to_codebank,
        impl_->vq_banks, impl_->vq_parameters, impl_->vq_residual_codebooks,
        impl_->vq_residual_first, impl_->vq_residual_second,
        impl_->mx_values, impl_->mx_scales, impl_->rotations, std::move(values),
        checked_int(experts.size(), "MFE view experts"), impl_->out_per_expert,
        impl_->neuron_len, 1);
    impl->automatic_mxfp4_nax_prefill = impl_->automatic_mxfp4_nax_prefill;
    impl->packed_arena = impl_->packed_arena;
    impl->has_generic_cohorts = impl_->has_generic_cohorts;
    impl->kernel_profile = impl_->kernel_profile;
    return MlxMfeWeight(std::move(impl));
}

MlxMfeWeight MlxMfeWeight::with_kernel_profile(
    std::shared_ptr<const MlxMfeKernelProfile> profile) const {
    if (impl_->kernel_profile == profile) return *this;
    auto impl = std::make_shared<Impl>(*impl_);
    impl->kernel_profile = std::move(profile);
    return MlxMfeWeight(std::move(impl));
}

MlxMfeWeight MlxMfeWeight::with_prepared_page_resources() const {
    if (!impl_->indirect_addresses || impl_->page_resources) return *this;
    auto resources = std::make_shared<MfePageResources>();
    resources->owners = impl_->indirect_owners;
    auto descriptor = NS::TransferPtr(MTL::ResidencySetDescriptor::alloc()->init());
    descriptor->setInitialCapacity(resources->owners.size());
    NS::Error* error = nullptr;
    auto& device = mlx::core::metal::device(mlx::core::Device::gpu);
    resources->residency = NS::TransferPtr(device.mtl_device()->newResidencySet(descriptor.get(), &error));
    if (!resources->residency) throw std::runtime_error(error
        ? error->localizedDescription()->utf8String() : "expert residency set unavailable");
    for (const auto& owner : resources->owners) {
        if (!owner.is_available()) throw std::logic_error("prepared expert page is not ready");
        resources->residency->addAllocation(static_cast<const MTL::Buffer*>(owner.buffer().ptr()));
    }
    resources->residency->commit();
    auto impl = std::make_shared<Impl>(*impl_);
    impl->page_resources = std::move(resources);
    return MlxMfeWeight(std::move(impl));
}

std::shared_ptr<const MlxMfeCachedProjection> MlxMfeOffloadCache::cached_projection(const std::string& name) {
    std::unique_lock lock(impl_->mutex);
    const auto confidence = impl_->route_confidence.find(name);
    if (confidence == impl_->route_confidence.end() || confidence->second < 8) return {};
    if (const auto cached = impl_->cached_views.find(name); cached != impl_->cached_views.end()) return cached->second;
    const auto projection = impl_->mfe_projection_locked(name);
    const auto resident = impl_->resident_pages.find(name);
    if (resident == impl_->resident_pages.end() || !uses_indirect_expert_pages(projection->kernel_profile->family_mask)) return {};
    const auto count = std::count_if(resident->second.begin(), resident->second.end(), [](const auto& page) { return bool(page); });
    if (!count) return {};
    const auto estimate = checked_add(std::size_t{3} * getpagesize(), checked_product(count,
        sizeof(std::int32_t) * (kDescriptorSize + 2) + 21 * sizeof(std::uint64_t), "cached expert metadata"),
        "cached expert metadata");
    lock.unlock();
    if (!MlxResidentBudgetScope::permits_optional(estimate)) return {};
    MlxResidentBudgetScope::reserve(estimate);
    lock.lock();
    if (const auto cached = impl_->cached_views.find(name); cached != impl_->cached_views.end()) return cached->second;
    const auto current = impl_->resident_pages.find(name);
    if (current == impl_->resident_pages.end()) return {};
    const auto epoch = impl_->epoch;
    const auto generation = impl_->page_generation;
    std::vector<MlxMfeWeight> pages;
    std::vector<std::int32_t> map(projection->experts, -1);
    for (std::size_t expert = 0; expert < current->second.size(); ++expert) if (const auto& page = current->second[expert]) {
        map[expert] = static_cast<std::int32_t>(pages.size());
        pages.push_back(page->weight);
    }
    if (pages.empty()) return {};
    lock.unlock();
    auto weight = MlxMfeWeight::concatenate_expert_pages(pages, projection->kernel_profile).with_prepared_page_resources();
    if (!weight.impl_->page_resources) return {};
    array expert_map(map.begin(), Shape{projection->experts}, mlx::core::int32);
    const auto metadata = weight.impl_->descriptors.nbytes() + weight.impl_->indirect_addresses->nbytes()
        + (weight.impl_->expert_order ? weight.impl_->expert_order->nbytes() : 0) + expert_map.nbytes();
    auto view = std::make_shared<const MlxMfeCachedProjection>(MlxMfeCachedProjection{
        std::move(weight), std::move(expert_map), std::move(map), name, metadata});
    lock.lock();
    if (epoch == impl_->epoch && generation == impl_->page_generation && metadata <= impl_->cache_limit) {
        if (const auto cached = impl_->cached_views.find(name); cached != impl_->cached_views.end()) return cached->second;
        const auto bytes = checked_add(impl_->resident_bytes, metadata, "cached expert metadata");
        impl_->cached_views.emplace(name, view);
        impl_->resident_bytes = bytes;
        impl_->trim();
    }
    return view;
}

bool MlxMfeOffloadCache::resolve_cached_projections(std::span<const std::shared_ptr<const MlxMfeCachedProjection>> views,
    std::span<const std::int32_t> experts) {
    if (experts.empty() || views.empty()) throw std::invalid_argument("cached expert routes cannot be empty");
    std::unordered_map<std::int32_t, unsigned> counts;
    std::vector<std::int32_t> ordered;
    counts.reserve(experts.size());
    ordered.reserve(experts.size());
    for (const auto expert : experts) {
        const auto [found, inserted] = counts.try_emplace(expert, 0);
        if (inserted) ordered.push_back(expert);
        auto& count = found->second;
        count = std::min(3u, count + 1);
    }
    std::vector<bool> hits(views.size(), true);
    for (std::size_t index = 0; index < views.size(); ++index) {
        if (!views[index]) throw std::invalid_argument("cached projection is missing");
        for (const auto expert : ordered) {
            if (expert < 0 || static_cast<std::size_t>(expert) >= views[index]->host_expert_map.size())
                throw std::out_of_range("cached expert route is out of range");
            hits[index] = hits[index] && views[index]->host_expert_map[expert] >= 0;
        }
    }
    std::lock_guard lock(impl_->mutex);
    if (std::any_of(hits.begin(), hits.end(), [](bool hit) { return !hit; })) {
        for (std::size_t index = 0; index < views.size(); ++index) if (!hits[index]) {
            impl_->route_confidence[views[index]->name] = 0;
        }
        ++impl_->stats.cached_route_misses;
        return false;
    }
    ++impl_->stats.cached_route_hits;
    for (const auto& view : views) {
        auto& confidence = impl_->route_confidence[view->name];
        confidence = static_cast<std::uint8_t>(std::min<unsigned>(15, confidence + 1));
        impl_->stats.expert_hits += counts.size();
        for (const auto expert : ordered) {
            const auto count = counts.at(expert);
            const Impl::Key key{view->name, expert};
            const auto cached = impl_->mfe_cache.find(key);
            if (cached == impl_->mfe_cache.end()) continue;
            auto& value = *cached->second;
            value.credit = static_cast<std::uint8_t>(std::min<unsigned>(3, value.credit + count));
            if (value.credit >= 2 && !value.pin_attempted) value.pin_attempted = impl_->enqueue_pin(key, value.weight);
            impl_->mfe_lru.splice(impl_->mfe_lru.end(), impl_->mfe_lru, cached->second);
        }
    }
    return true;
}

array MlxMfeCachedRoute::forward(const array& source, const array& routes, float swiglu_limit,
    const array* reduce_weights) {
    ids = routes;
    const auto down = projections.size() - 1;
    auto intermediate = [&]() {
        if (projections.size() != 3)
            return projections[0]->weight.routed_swiglu(source,
                mlx::core::take(projections[0]->expert_map, routes), swiglu_limit);
        return projections[0]->weight.routed_swiglu_pair(projections[1]->weight,
            source, routes, swiglu_limit, nullptr, &projections[0]->expert_map, &projections[1]->expert_map);
    }();
    auto down_ids = mlx::core::take(projections[down]->expert_map, routes);
    return reduce_weights ? projections[down]->weight.routed_matmul_reduce(intermediate, down_ids, *reduce_weights)
        : projections[down]->weight.routed_matmul(intermediate, down_ids);
}

std::span<const std::int32_t> MlxMfeCachedRoute::host_ids() {
    if (!ids) throw std::logic_error("cached route was not evaluated");
    ids->eval();
    if (ids->dtype() != mlx::core::int32 || !ids->flags().row_contiguous) {
        ids = mlx::core::contiguous(mlx::core::astype(*ids, mlx::core::int32));
        ids->eval();
    }
    return {ids->data<std::int32_t>(), ids->size()};
}

bool MlxMfeCachedRoute::all_resident() {
    const auto routes = host_ids();
    for (const auto& projection : projections) for (const auto expert : routes) {
        if (expert < 0 || static_cast<std::size_t>(expert) >= projection->host_expert_map.size())
            throw std::out_of_range("cached expert route is out of range");
        if (projection->host_expert_map[expert] < 0) return false;
    }
    return true;
}

bool MlxMfeCachedRoute::resolve() {
    return cache->resolve_cached_projections(projections, host_ids());
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
        if (model.record(gate_name).dtype != "MFE" || model.record(up_name).dtype != "MFE") {
            throw std::runtime_error("routed expert tensors must use MFE under " + base);
        }
        const auto gate = model.map_record(gate_name);
        const auto up = model.map_record(up_name);
        const std::array blobs{gate.view(), up.view()};
        return MlxMoeWeight::from_projection_blobs(blobs);
    }
    return load(base + ".gate_up.weight");
}

array MlxMfeWeight::routed_matmul(
    const array& input,
    const array& expert_ids,
    const array* route_groups) const {
    return routed_matmul_impl(
        input,
        expert_ids,
        false,
        0.0f, nullptr, false, route_groups);
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
    float limit,
    const array* route_groups) const {
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
                limit, nullptr, false, route_groups);
        }
        const bool has_standalone_projection = std::any_of(
            impl_->projection_views.begin(),
            impl_->projection_views.end(),
            [](const auto& projection) {
                return !projection->reference_cohorts.empty()
                    || !projection->mxfp4_sq_cohorts.empty()
                    || !projection->fp8_sq_cohorts.empty();
            });
        if (mfe_split_swiglu_enabled()
            && impl_->native_primitive
            && !has_standalone_projection) {
            return routed_matmul_impl(
                input,
                expert_ids,
                true,
                limit, nullptr, false, route_groups);
        }
        auto gate_up = routed_matmul_impl(
            input,
            expert_ids,
            false,
            0.0f, nullptr, false, route_groups);
        return limit > 0.0f
            ? moe_limited_swiglu_split(gate_up, limit)
            : moe_swiglu_split(gate_up);
    }
    return routed_matmul_impl(
        input,
        expert_ids,
        true,
        limit, nullptr, false, route_groups);
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
    float limit,
    const array* route_groups,
    const array* gate_map,
    const array* up_map) const {
    if (!std::isfinite(limit) || limit < 0.0f) {
        throw std::invalid_argument(
            "paired MFE SwiGLU limit must be finite and non-negative");
    }
    if ((gate_map == nullptr) != (up_map == nullptr)
        || (!gate_map && impl_->experts != up.impl_->experts)
        || impl_->out_per_expert != up.impl_->out_per_expert
        || impl_->neuron_len != up.impl_->neuron_len
        || impl_->projections != 1
        || up.impl_->projections != 1) {
        throw std::invalid_argument(
            "paired MFE Gate and Up weights are incompatible");
    }
    if (gate_map && (gate_map->ndim() != 1 || up_map->ndim() != 1
            || gate_map->size() != up_map->size() || gate_map->size() == 0
            || gate_map->dtype() != mlx::core::int32 || up_map->dtype() != mlx::core::int32
            || !gate_map->flags().row_contiguous || !up_map->flags().row_contiguous)) {
        throw std::invalid_argument("paired MFE maps must be equal-sized contiguous int32 vectors");
    }
    if (route_groups && (expert_ids.ndim() != 2 || expert_ids.shape(0) < 2 || expert_ids.shape(0) > 24
            || route_groups->dtype() != mlx::core::int32 || !route_groups->flags().row_contiguous
            || route_groups->shape() != Shape{expert_ids.shape(0), expert_ids.shape(1), 8})) {
        throw std::invalid_argument("paired MFE route groups must be contiguous int32 [tokens,routes,8]");
    }
    const auto fallback = [&]() {
        auto gate_output = gate_map ? routed_matmul_mapped(input, expert_ids, *gate_map)
            : routed_matmul(input, expert_ids, route_groups);
        auto up_output = up_map ? up.routed_matmul_mapped(input, expert_ids, *up_map)
            : up.routed_matmul(input, expert_ids, route_groups);
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
        !gate_map &&
        !impl_->indirect_sources && !up.impl_->indirect_sources &&
        impl_->native_primitive
        && up.impl_->native_primitive
        && impl_->family_mask
            == (std::uint32_t{1} << kFamilyMxfp4)
        && up.impl_->family_mask
            == (std::uint32_t{1} << kFamilyMxfp4)
        && impl_->rotations.empty()
        && up.impl_->rotations.empty()
        && (impl_->k_lanes_override == 0
            || impl_->k_lanes_override == 16)
        && (up.impl_->k_lanes_override == 0
            || up.impl_->k_lanes_override == 16)
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
        bool reusable_routes = input.ndim() == 2 && input.shape(0) == 1;
        if (!reusable_routes && route_groups && route_groups->is_available()) {
            const auto* groups = route_groups->data<std::int32_t>();
            reusable_routes = true;
            bool any = false;
            for (std::size_t index = 0; index < expert_ids.size(); ++index) {
                if (groups[index * 8] < 0) continue;
                any = true;
                reusable_routes &= groups[index * 8 + 1] > 1;
            }
            reusable_routes &= any;
        }
        const auto eligible = [](const Impl& value) {
            return value.native_primitive && value.reference_cohorts.empty()
                && value.mxfp4_sq_cohorts.empty() && value.fp8_sq_cohorts.empty()
                && !value.standalone_owner && value.k_lanes_override != 32;
        };
        if (reusable_routes && !mlx_reference_enabled() && mfe_split_swiglu_enabled()
            && input.ndim() == 2 && input.shape(0) >= 1 && input.shape(0) <= 6
            && expert_ids.ndim() == 2 && expert_ids.shape(0) == input.shape(0)
            && expert_ids.shape(1) > 0 && expert_ids.shape(1) <= 16
            && input.shape(1) == impl_->neuron_len
            && eligible(*impl_) && eligible(*up.impl_)
            && uses_indirect_expert_pages(impl_->compilation_profile().family_mask
                | up.impl_->compilation_profile().family_mask)) {
            const auto streams = [](const Impl& value) {
                return std::array<const array*, 21>{&value.nint_q, &value.nint_sub_scale, &value.nint_sub_min,
                    &value.nint_anchor_scale, &value.nint_anchor_min, &value.q8_q, &value.q8_scales,
                    &value.vq_indices, &value.vq_state, &value.vq_aux, &value.vq_anchors,
                    &value.vq_codebooks, &value.vq_scales, &value.vq_state_to_codebank, &value.vq_banks,
                    &value.vq_parameters, &value.vq_residual_codebooks, &value.vq_residual_first,
                    &value.vq_residual_second, &value.mx_values, &value.mx_scales};
            };
            const std::array sources{impl_, up.impl_};
            bool available = true;
            for (const auto& source : sources) {
                available &= source->descriptor_values.size() == std::size_t(source->experts) * kDescriptorSize;
                if (!source->indirect_addresses) for (const auto* value : streams(*source))
                    available &= value->is_available() && value->flags().row_contiguous;
            }
            if (available && input.shape(0) > 1 && !gate_map) {
                for (int expert = 0; expert < impl_->experts; ++expert) {
                    const auto* gate = impl_->descriptor_values.data() + expert * kDescriptorSize;
                    const auto* other = up.impl_->descriptor_values.data() + expert * kDescriptorSize;
                    const auto reusable = [](const auto* descriptor) {
                        const auto profile = descriptor[kVqProfile] & 255;
                        return descriptor[kFamily] == kFamilyMxfp4
                            || (descriptor[kFamily] == kFamilyVq
                                && profile < 32 && (474u & (1u << profile)) != 0u);
                    };
                    if ((gate[kFamily] != kFamilyMxfp4 && gate[kFamily] != kFamilyVq)
                        || (other[kFamily] != kFamilyMxfp4 && other[kFamily] != kFamilyVq)
                        || reusable(gate) != reusable(other)) {
                        available = false;
                        break;
                    }
                }
            }
            if (available && gate_map && input.shape(0) > 1) {
                for (const auto& source : sources) {
                    const auto profile = source->compilation_profile();
                    available &= (profile.family_mask & ~((1u << kFamilyMxfp4) | (1u << kFamilyVq))) == 0
                        && (profile.vq_profile_mask & ~474u) == 0;
                }
            }
            if (available) {
                const auto page_addresses = [&](const auto& source) {
                    if (source->indirect_addresses) return *source->indirect_addresses;
                    std::vector<std::uint64_t> addresses;
                    addresses.reserve(21);
                    for (const auto* value : streams(*source)) {
                        const auto* buffer = static_cast<const MTL::Buffer*>(value->buffer().ptr());
                        addresses.push_back(buffer->gpuAddress() + value->offset());
                    }
                    return array(addresses.begin(), Shape{1, 21}, mlx::core::uint64);
                };
                auto source = mlx::core::contiguous(input.dtype() == mlx::core::float16
                    || input.dtype() == mlx::core::float32 ? input : mlx::core::astype(input, mlx::core::float16));
                const int tokens = input.shape(0), routes = expert_ids.shape(1);
                const auto rotated_source = [&](const auto& weight) {
                    if (weight->rotations.empty()) return source;
                    std::vector<array> variants{source};
                    for (const auto& rotation : weight->rotations)
                        variants.push_back(apply_rotation(source, rotation, tokens, impl_->neuron_len));
                    return mlx::core::contiguous(mlx::core::concatenate(variants, 0));
                };
                auto inputs = impl_->narrow_bindings;
                inputs[1] = page_addresses(impl_);
                inputs[2] = up.impl_->descriptors;
                inputs[3] = page_addresses(up.impl_);
                inputs[4] = gate_map ? *gate_map : empty_packed_array(mlx::core::int32);
                inputs[5] = up_map ? *up_map : empty_packed_array(mlx::core::int32);
                inputs[6] = rotated_source(up.impl_);
                inputs.push_back(rotated_source(impl_));
                inputs.push_back(mlx::core::contiguous(mlx::core::astype(expert_ids, mlx::core::int32)));
                inputs.push_back(empty_packed_array(mlx::core::int32));
                inputs.emplace_back(std::initializer_list<float>{limit}, mlx::core::float32);
                if (route_groups) inputs.push_back(*route_groups);
                const bool prepared = impl_->page_resources && up.impl_->page_resources;
                if (!prepared) for (const auto& weight : sources) {
                    if (weight->indirect_addresses)
                        inputs.insert(inputs.end(), weight->indirect_owners.begin(), weight->indirect_owners.end());
                    else for (const auto* value : streams(*weight)) inputs.push_back(*value);
                }
                const auto gate_profile = impl_->compilation_profile();
                const auto up_profile = up.impl_->compilation_profile();
                std::vector<int> groups = *gate_profile.nint_group_sizes;
                groups.insert(groups.end(), up_profile.nint_group_sizes->begin(), up_profile.nint_group_sizes->end());
                std::sort(groups.begin(), groups.end());
                groups.erase(std::unique(groups.begin(), groups.end()), groups.end());
                const auto families = gate_profile.family_mask | up_profile.family_mask;
                const int rows_per_simd = tokens == 1
                    && (families & ~((1u << kFamilyMxfp4) | (1u << kFamilyVq))) == 0 ? 2 : 1;
                const auto output_rows = std::size_t{2} * rows_per_simd;
                const int workgroups = checked_int(checked_product(checked_product(tokens, routes, "paired MFE routes"),
                    (std::size_t(impl_->out_per_expert) + output_rows - 1) / output_rows,
                    "paired MFE workgroups"), "paired MFE workgroups");
                detail::profile_marker("mfe.dispatch.indirect_pair_native");
                return native_moe_dispatch(std::move(inputs), NativeMoeConfig{
                    .dtype = source.dtype(),
                    .output_shape = Shape{tokens, routes, impl_->out_per_expert},
                    .tokens = tokens,
                    .routes = routes,
                    .experts = impl_->experts,
                    .output_width = impl_->out_per_expert,
                    .matrix_output_width = impl_->out_per_expert,
                    .projections = 2,
                    .fused_swiglu = 1,
                    .input_width = impl_->neuron_len,
                    .k_lanes = 16,
                    .rows_per_simd = rows_per_simd,
                    .nint_group_sizes = std::make_shared<const std::vector<int>>(std::move(groups)),
                    .descriptor_size = kDescriptorSize,
                    .variant_stride = tokens,
                    .shared_input = 1,
                    .vq_execution_layout = gate_profile.vq_execution_layout || up_profile.vq_execution_layout,
                    .vq_layouts = gate_profile.vq_layouts | up_profile.vq_layouts,
                    .family_mask = static_cast<int>(gate_profile.family_mask | up_profile.family_mask),
                    .vq_profile_mask = static_cast<int>(gate_profile.vq_profile_mask | up_profile.vq_profile_mask),
                    .has_nepq_residual = impl_->has_nepq_residual || up.impl_->has_nepq_residual,
                    .npq_grouped_indices = impl_->npq_grouped_indices && up.impl_->npq_grouped_indices,
                    .expert_map_size = gate_map ? checked_int(gate_map->size(), "paired MFE expert map") : 0,
                    .route_groups = route_groups != nullptr,
                    .nint_q_dispatch = gate_profile.nint_q_dispatch && up_profile.nint_q_dispatch,
                    .indirect_pages = true,
                    .dual_pages = true,
                    .direct_page_mask = (impl_->indirect_addresses ? 0 : 1) | (up.impl_->indirect_addresses ? 0 : 2),
                    .other_experts = up.impl_->experts,
                    .workgroups = workgroups,
                    .page_resources = prepared ? impl_->page_resources : nullptr,
                    .other_page_resources = prepared ? up.impl_->page_resources : nullptr,
                });
            }
        }
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
    constexpr int kOutputsPerWorkgroup = 4;
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

std::optional<array> MlxMfeWeight::decode_nint_shared(
    const MlxMfeWeight& down,
    const MlxNintSwiGluPair& shared_gate_up,
    const MlxNintWeight& shared_down,
    const array& shared_gate_weight,
    const array& input,
    const array& expert_ids,
    const array& route_weights,
    bool shared_gate_is_logits,
    const array* route_groups,
    const array* router_logits) const {
    const auto supported_impl = [](const Impl& impl, int projections) {
        return impl.native_primitive && !impl.indirect_sources &&
            impl.projections == projections &&
            impl.rotations.empty() &&
            impl.reference_cohorts.empty() &&
            impl.mxfp4_sq_cohorts.empty() &&
            impl.fp8_sq_cohorts.empty() &&
            !impl.standalone_owner.has_value() &&
            impl.has_generic_cohorts &&
            impl.descriptor_values.size() ==
                static_cast<std::size_t>(impl.experts) *
                    static_cast<std::size_t>(projections) *
                    kDescriptorSize &&
            impl.narrow_bindings.size() == 22;
    };
    const bool activation16 =
        input.dtype() == mlx::core::float16 ||
        input.dtype() == mlx::core::bfloat16;
    const bool packed_shared_router = shared_gate_weight.dtype() == mlx::core::uint8 &&
        shared_gate_weight.ndim() == 1 && shared_gate_weight.size() >= 36;
    const bool shared_gate_dtype_supported = packed_shared_router ||
        shared_gate_weight.dtype() == mlx::core::float16 ||
        shared_gate_weight.dtype() == mlx::core::bfloat16 ||
        (shared_gate_is_logits && shared_gate_weight.dtype() == mlx::core::float32);
    const int routes = expert_ids.ndim() == 2
        ? expert_ids.shape(1) : 0;
    const int tokens = input.ndim() == 2 ? input.shape(0) : 0;
    const bool shared_gate_shape_supported = packed_shared_router || (shared_gate_is_logits
        ? shared_gate_weight.size() == static_cast<std::size_t>(tokens)
        : shared_gate_weight.ndim() == 2 && shared_gate_weight.shape(0) == 1 &&
            shared_gate_weight.shape(1) == impl_->neuron_len);
    const bool packed_gate_up = impl_->projections == 1;
    const int intermediate = down.impl_->neuron_len;
    const bool supported = !mlx_reference_enabled() &&
        routes > 0 && routes <= 16 &&
        (routes + 1) * 32 <= 1024 &&
        impl_->experts > 0 && impl_->experts % 32 == 0 &&
        routes <= impl_->experts &&
        supported_impl(*impl_, packed_gate_up ? 1 : 2) &&
        supported_impl(*down.impl_, 1) &&
        impl_->experts == down.impl_->experts &&
        impl_->out_per_expert > 0 &&
        impl_->neuron_len > 0 &&
        down.impl_->out_per_expert == impl_->neuron_len &&
        intermediate > 0 &&
        impl_->out_per_expert == intermediate * (packed_gate_up ? 2 : 1) &&
        tokens > 0 &&
        input.shape(1) == impl_->neuron_len && activation16 &&
        expert_ids.dtype() == mlx::core::int32 &&
        expert_ids.shape() == Shape{tokens, routes} &&
        route_weights.dtype() == mlx::core::float32 &&
        route_weights.shape() == Shape{tokens, routes} &&
        shared_gate_up.input_size() == impl_->neuron_len &&
        shared_gate_up.output_size() == intermediate &&
        shared_down.input_size() == intermediate &&
        shared_down.output_size() == impl_->neuron_len &&
        shared_gate_shape_supported &&
        shared_gate_weight.flags().row_contiguous && shared_gate_dtype_supported;
    if (!supported) {
        return std::nullopt;
    }
    int shared_router_group_size = 0;
    if (packed_shared_router) {
        auto packed = shared_gate_weight;
        packed.eval();
        std::array<std::uint32_t, 9> header{};
        std::memcpy(header.data(), packed.data<std::uint8_t>(), sizeof(header));
        shared_router_group_size = static_cast<int>(header[3]);
        const auto q = header[4] & 15u, k = (header[4] >> 8u) & 15u;
        if (shared_router_group_size <= 0 || header[8] != std::uint32_t(impl_->neuron_len) ||
            q < 1 || q > 8 || k < 1 || k > 8 ||
            header[0] != sizeof(header) || header[1] < header[0] ||
            header[2] < header[1] || header[2] > packed.size() ||
            (header[6] & 0x7c00u) == 0x7c00u || ((header[6] >> 16u) & 0x7c00u) == 0x7c00u)
            throw std::invalid_argument("invalid packed NINT router header");
        const auto groups = (std::uint64_t(impl_->neuron_len) + header[3] - 1) / header[3];
        const auto q_bytes = header[5] + (((header[4] >> 4u) & 7u) + groups * header[3] * q + 7) / 8;
        const auto sub_bytes = header[7] + (((header[4] >> 12u) & 7u) + groups * k + 7) / 8;
        if (q_bytes >= header[1] - header[0] || sub_bytes >= header[2] - header[1] ||
            sub_bytes >= packed.size() - header[2])
            throw std::invalid_argument("truncated packed NINT router streams");
    }
    const bool folded = router_logits != nullptr && tokens == 1 && route_groups == nullptr &&
        impl_->experts <= 4096 && router_logits->dtype() == mlx::core::float16 &&
        router_logits->shape() == Shape{1, impl_->experts};
    if (route_groups != nullptr && (tokens < 2 || tokens > 24 || route_groups->dtype() != mlx::core::int32 ||
            route_groups->shape() != Shape{tokens, routes, 8} || !route_groups->flags().row_contiguous))
        throw std::invalid_argument("two-stage MFE route groups must be contiguous int32 [tokens,routes,8] for M=2..24");

    auto source = mlx::core::contiguous(mlx::core::reshape(
        input.dtype() == mlx::core::float16
            ? input
            : mlx::core::astype(input, mlx::core::float16),
        Shape{tokens, impl_->neuron_len}));
    auto ids = mlx::core::contiguous(mlx::core::reshape(
        expert_ids, Shape{tokens * routes}));
    auto weights = mlx::core::contiguous(mlx::core::reshape(
        route_weights, Shape{tokens * routes}));
    auto shared_gate = mlx::core::contiguous(shared_gate_weight);

    const int decode_k_lanes = impl_->k_lanes_override != 0
        ? impl_->k_lanes_override
        : 16;
    constexpr int kGateRowsPerSimd = 1;
    const int gate_rows_per_workgroup =
        (folded ? kMfeFoldedGateSimdPairs : 1) *
        (32 / decode_k_lanes) * kGateRowsPerSimd;
    const int gate_tiles =
        (intermediate + gate_rows_per_workgroup - 1) /
        gate_rows_per_workgroup;
    const bool mixed_decode_two_rows = apple_m5_family() &&
        down.impl_->family_mask ==
            ((std::uint32_t{1} << kFamilyNint) |
             (std::uint32_t{1} << kFamilyVq));
    const int down_rows_per_simd = mixed_decode_two_rows ? 2 : 1;
    const int down_k_lanes = down.impl_->k_lanes_override != 0
        ? down.impl_->k_lanes_override
        : (tokens == 1 && impl_->neuron_len > intermediate && down_rows_per_simd > 1
            ? 32 : decode_k_lanes);
    const int down_rows_per_workgroup =
        (32 / down_k_lanes) * down_rows_per_simd;
    const int down_tiles =
        (down.impl_->out_per_expert + down_rows_per_workgroup - 1) /
        down_rows_per_workgroup;
    const int stage_one_width =
        (routes + 1) * intermediate + 1;

    MfeNintDecodeConfig config{
        .dtype = mlx::core::float16,
        .shared_gate_dtype = shared_gate.dtype(),
        .shared_gate_is_logits = shared_gate_is_logits,
        .shared_router_group_size = shared_router_group_size,
        .packed_gate_up = packed_gate_up,
        .route_groups = route_groups != nullptr,
        .folded_topk = folded,
        .gate_q_dispatch = impl_->compilation_profile().nint_q_dispatch,
        .down_q_dispatch = down.impl_->compilation_profile().nint_q_dispatch,
        .shared_gate_q_dispatch = shared_gate_up.has_uniform_q_bits(),
        .shared_down_q_dispatch = shared_down.has_uniform_q_bits(),
        .stage_one_shape = Shape{tokens, stage_one_width},
        .output_shape = Shape{tokens, impl_->neuron_len},
        .tokens = tokens,
        .routes = routes,
        .experts = impl_->experts,
        .hidden = impl_->neuron_len,
        .intermediate = intermediate,
        .descriptor_size = kDescriptorSize,
        .gate_family_mask = static_cast<int>(impl_->family_mask),
        .gate_vq_profile_mask = static_cast<int>(impl_->vq_profile_mask),
        .gate_vq_execution_layout =
            static_cast<int>(impl_->vq_execution_layout),
        .gate_vq_layouts = impl_->vq_layouts,
        .gate_has_nepq_residual =
            static_cast<int>(impl_->has_nepq_residual),
        .gate_npq_grouped_indices =
            static_cast<int>(impl_->npq_grouped_indices),
        .gate_nint_group_sizes = impl_->nint_group_sizes,
        .down_family_mask = static_cast<int>(down.impl_->family_mask),
        .down_vq_profile_mask =
            static_cast<int>(down.impl_->vq_profile_mask),
        .down_vq_execution_layout =
            static_cast<int>(down.impl_->vq_execution_layout),
        .down_vq_layouts = down.impl_->vq_layouts,
        .down_has_nepq_residual =
            static_cast<int>(down.impl_->has_nepq_residual),
        .down_npq_grouped_indices =
            static_cast<int>(down.impl_->npq_grouped_indices),
        .down_nint_group_sizes = down.impl_->nint_group_sizes,
        .shared_gate_group_size = shared_gate_up.group_size(),
        .shared_gate_groups = shared_gate_up.groups(),
        .shared_up_group_size = shared_gate_up.up_group_size(),
        .shared_up_groups = shared_gate_up.up_groups(),
        .shared_down_group_size = shared_down.group_size(),
        .shared_down_groups = shared_down.groups(),
        .k_lanes = decode_k_lanes,
        .down_k_lanes = down_k_lanes,
        .down_rows_per_simd = down_rows_per_simd,
        .stage_one_workgroups = 1 + (routes + 1) * gate_tiles,
        .stage_two_workgroups = down_tiles,
    };

    auto stage_one_inputs = impl_->narrow_bindings;
    stage_one_inputs.reserve(29 + static_cast<int>(route_groups != nullptr));
    stage_one_inputs.push_back(std::move(source));
    stage_one_inputs.push_back(folded ? mlx::core::contiguous(*router_logits) : ids);
    stage_one_inputs.push_back(shared_gate_up.packed_values());
    stage_one_inputs.push_back(shared_gate_up.row_metadata());
    stage_one_inputs.push_back(shared_gate_up.sub_scales());
    stage_one_inputs.push_back(shared_gate_up.sub_mins());
    stage_one_inputs.push_back(std::move(shared_gate));
    if (route_groups != nullptr) stage_one_inputs.push_back(*route_groups);
    auto stage_one_outputs = mfe_nint_decode_dispatch(
        std::move(stage_one_inputs), config, 1);
    auto stage_one = std::move(stage_one_outputs[0]);
    if (folded) {
        ids = stage_one_outputs[1];
        weights = stage_one_outputs[1];
    }

    auto stage_two_inputs = down.impl_->narrow_bindings;
    stage_two_inputs.reserve(29);
    stage_two_inputs.push_back(std::move(stage_one));
    stage_two_inputs.push_back(shared_down.packed_values());
    stage_two_inputs.push_back(shared_down.row_metadata());
    stage_two_inputs.push_back(shared_down.sub_scales());
    stage_two_inputs.push_back(shared_down.sub_mins());
    stage_two_inputs.push_back(std::move(ids));
    stage_two_inputs.push_back(std::move(weights));
    detail::profile_marker("mfe.dispatch.two_stage_nint_shared");
    auto outputs = mfe_nint_decode_dispatch(std::move(stage_two_inputs), config, 2);
    return std::move(outputs.front());
}

bool MlxMfeWeight::supports_fused_routed_reduce(const array* route_groups) const noexcept {
    if (route_groups && (impl_->indirect_sources || impl_->rotations.size()
            || impl_->family_mask != (std::uint32_t{1} << kFamilyMxfp4))) {
        if (!route_groups->is_available() || route_groups->dtype() != mlx::core::int32
            || !route_groups->flags().row_contiguous || route_groups->ndim() != 3
            || route_groups->shape(2) != 8) return false;
        const auto* groups = route_groups->data<std::int32_t>();
        for (std::size_t index = 0; index < route_groups->size() / 8; ++index)
            if (groups[index * 8 + 1] > 1) return false;
    }
    return mxfp4_decode_down_reduce_enabled()
        && !mlx_reference_enabled()
        && impl_->native_primitive
        && impl_->projections == 1
        && impl_->reference_cohorts.empty() && impl_->mxfp4_sq_cohorts.empty()
        && impl_->fp8_sq_cohorts.empty() && !impl_->standalone_owner
        && (impl_->k_lanes_override == 0 || impl_->k_lanes_override == 16)
        && impl_->out_per_expert > 0
        && impl_->neuron_len > 0
        && impl_->neuron_len % 32 == 0
        && impl_->descriptor_values.size()
            == static_cast<std::size_t>(impl_->experts) * kDescriptorSize;
}

array MlxMfeWeight::routed_matmul_reduce(
    const array& input,
    const array& expert_ids,
    const array& route_weights,
    const array* route_groups) const {
    const auto fallback = [&]() {
        return moe_weighted_reduce(
            routed_matmul(input, expert_ids, route_groups),
            route_weights);
    };
    if (!supports_fused_routed_reduce()
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
        || expert_ids.shape(1) > 16) {
        detail::profile_marker("mfe.dispatch.mxfp4_down_reduce_fallback");
        return fallback();
    }

    const bool pure_mxfp4 = !impl_->indirect_sources && impl_->rotations.empty()
        && impl_->family_mask == (std::uint32_t{1} << kFamilyMxfp4);
    if (!pure_mxfp4) {
        if (route_groups) {
            if (route_groups->dtype() != mlx::core::int32
                || route_groups->shape() != Shape{input.shape(0), expert_ids.shape(1), 8}
                || !route_groups->flags().row_contiguous)
                throw std::invalid_argument("Down route groups must be contiguous int32 [tokens,routes,8]");
            if (!route_groups->is_available()) return fallback();
            const auto* groups = route_groups->data<std::int32_t>();
            for (std::size_t index = 0; index < expert_ids.size(); ++index)
                if (groups[index * 8 + 1] > 1) return fallback();
        }
        detail::profile_marker("mfe.dispatch.mixed_down_reduce_native");
        return routed_matmul_impl(input, expert_ids, false, 0.0f, nullptr, false, nullptr, &route_weights);
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
        2 * rows_per_lane_group;
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
        if (!use_nax && impl_->experts > 0 && impl_->neuron_len >= 96
            && impl_->out_per_expert >= (fused_swiglu && impl_->projections == 1 ? 128 : 64)) {
            const std::int64_t mean_routes =
                (std::int64_t(route_count) + impl_->experts - 1) / impl_->experts;
            if (mean_routes > 64) return 96;
            if (mean_routes > 32) return 64;
        }
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
    if (!fused_swiglu && mean_routes > 64
        && impl_->out_per_expert >= 64 && impl_->neuron_len >= 192) {
        return 128;
    }
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
        if (!fused_swiglu) return mlx::core::concatenate(std::move(outputs), 1);
        return swiglu_limit > 0.0f
            ? moe_limited_swiglu_pair(outputs.at(0), outputs.at(1), swiglu_limit)
            : moe_swiglu_pair(outputs.at(0), outputs.at(1));
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
    const int variant_stride = shared_input ? tokens : route_count;
    if (!impl_->rotations.empty()) {
        source = mlx::core::contiguous(
            mlx::core::reshape(
                source,
                Shape{variant_stride, impl_->neuron_len}));
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
                std::nullopt,
                safe_sorted_indices),
            1);
        return fused_swiglu
            ? moe_limited_swiglu_split(
                  std::move(projected),
                  swiglu_limit)
            : projected;
    }

    const array params({0.0f}, mlx::core::float32);
    std::vector<array> kernel_inputs = mfe_narrow_bindings_enabled() && impl_->native_primitive
        ? impl_->narrow_bindings
        : std::vector<array>{
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
          };
    kernel_inputs.push_back(source);
    kernel_inputs.push_back(ids);
    kernel_inputs.push_back(route_order);
    kernel_inputs.push_back(params);
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
            && selected_plan.block_rows != 96
            && selected_plan.block_rows != 128)
        || (!use_grouped_nax && selected_plan.block_rows != 32
            && selected_plan.block_rows != 64 && selected_plan.block_rows != 80
            && selected_plan.block_rows != 96)
        || (use_direct_nax && selected_plan.block_rows != 32)
        || selected_plan.max_blocks <= 0
    ) {
        throw std::invalid_argument(
            "grouped MMQ block plan does not match routed projection");
    }
    kernel_inputs.push_back(selected_plan.block_meta);
    kernel_inputs.push_back(selected_plan.block_count);
    if (impl_->indirect_addresses) {
        kernel_inputs.push_back(*impl_->indirect_addresses);
        kernel_inputs.insert(kernel_inputs.end(), impl_->indirect_owners.begin(), impl_->indirect_owners.end());
    }
    const auto compilation = impl_->compilation_profile();
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
                    output_width,
                    fused_swiglu)
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
            .vq_profile_mask = static_cast<int>(compilation.vq_profile_mask),
            .nint_group_sizes = impl_->nint_group_sizes,
            .cohort_populations = impl_->cohort_populations,
            .use_nax = use_grouped_nax,
            .direct_nax = use_direct_nax,
            .indirect_pages = impl_->indirect_addresses.has_value(),
            .swiglu_limit = swiglu_limit,
        });
}

array MlxMfeWeight::routed_matmul_impl(
    const array& input,
    const array& expert_ids,
    bool fused_swiglu,
    float swiglu_limit,
    const array* expert_map,
    bool packed_expert_ids,
    const array* route_groups,
    const array* reduce_weights) const {
    if (impl_->indirect_sources && mlx_reference_enabled())
        return concatenate_experts(*impl_->indirect_sources).with_kernel_profile(impl_->kernel_profile)
            .routed_matmul_impl(input, expert_ids, fused_swiglu, swiglu_limit, expert_map, packed_expert_ids, route_groups);
    if (impl_->projections > 1 && !impl_->projection_views.empty() &&
        std::any_of(
            impl_->projection_views.begin(),
            impl_->projection_views.end(),
            [](const auto& projection) {
                return !projection->reference_cohorts.empty()
                    || !projection->mxfp4_sq_cohorts.empty()
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
                packed_expert_ids, route_groups));
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
    if (route_groups != nullptr && (tokens < 2 || tokens > 24 ||
            route_groups->dtype() != mlx::core::int32 ||
            route_groups->shape() != Shape{tokens, routes, 8} ||
            !route_groups->flags().row_contiguous)) {
        throw std::invalid_argument("MFE route groups must be contiguous int32 [tokens,routes,8] for M=2..24");
    }
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
    const Shape output_shape = reduce_weights ? Shape{tokens, impl_->out_per_expert} : Shape{
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
    const int route_count = checked_int(
        route_count_size,
        "route count");
    const int variant_stride = shared_input ? tokens : route_count;
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
                        Shape{route_count})),
                mlx::core::int32));
    }
    if (!impl_->rotations.empty()) {
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
    }

    const bool split_fused_swiglu =
        fused_swiglu && impl_->projections == 2;
    const int k_lanes = impl_->k_lanes_override != 0
        ? impl_->k_lanes_override
        : 16;
    // Two rows reuse activation loads on the measured M5 mixed NINT/VQ
    // 640 -> 2560 decode geometry. Do not apply this to the inverse gate/up
    // projection: its larger K/register footprint regresses some cohorts.
    const auto compilation = impl_->compilation_profile();
    const bool mixed_decode_two_rows =
        impl_->projections == 1 && impl_->neuron_len == 640 &&
        impl_->out_per_expert == 2560 &&
        compilation.family_mask == ((std::uint32_t{1} << kFamilyNint) |
                               (std::uint32_t{1} << kFamilyVq));
    const int automatic_rows = mixed_decode_two_rows ||
        (reduce_weights && impl_->out_per_expert > impl_->neuron_len) ||
        (impl_->projections == 1 && compilation.family_mask ==
            (std::uint32_t{1} << kFamilyMxfp4)) ? 2 : 1;
    const int rows_per_simd =
        split_fused_swiglu
        ? 1
        : (tokens == 1 || (tokens <= 6 && mixed_decode_two_rows))
            && (impl_->rotations.empty() || reduce_weights)
        ? mfe_decode_rows_per_simd(automatic_rows)
        : 1;
    const auto rows_per_workgroup = static_cast<std::size_t>(
        (split_fused_swiglu || reduce_weights ? 32 : 64)
        / k_lanes * rows_per_simd);
    const auto output_tiles =
        (
            static_cast<std::size_t>(
                logical_output_width)
            + rows_per_workgroup - 1
        ) / rows_per_workgroup;
    auto workgroups = reduce_weights ? static_cast<std::size_t>(tokens) : route_count_size;
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
    std::vector<array> kernel_inputs = mfe_narrow_bindings_enabled() && impl_->native_primitive
        ? impl_->narrow_bindings
        : std::vector<array>{
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
          };
    if (!impl_->native_primitive) {
        constexpr int min_device_elements = 8;
        for (auto& bank : kernel_inputs) {
            if (bank.size() < min_device_elements) {
                const int elements = static_cast<int>(bank.size());
                bank = mlx::core::concatenate({
                    mlx::core::reshape(bank, Shape{elements}),
                    mlx::core::zeros(Shape{min_device_elements - elements}, bank.dtype()),
                });
            }
        }
    }
    kernel_inputs.push_back(source);
    kernel_inputs.push_back(ids);
    kernel_inputs.push_back(route_order);
    kernel_inputs.push_back(params);
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
            && mixed_grouped_nax_enabled(route_count);
        constexpr bool use_direct_nax = false;
        auto plan = build_grouped_mmq_plan(
            ids,
            route_order,
            recommended_grouped_mmq_block_rows(
                route_count,
                fused_swiglu));
        kernel_inputs.push_back(plan.block_meta);
        kernel_inputs.push_back(plan.block_count);
        if (impl_->indirect_addresses) {
            kernel_inputs.push_back(*impl_->indirect_addresses);
            kernel_inputs.insert(kernel_inputs.end(), impl_->indirect_owners.begin(), impl_->indirect_owners.end());
        }
        auto sorted_outputs = grouped_mmq_dispatch(
            std::move(kernel_inputs),
            GroupedMmqConfig{
                .output_shape = Shape{
                    route_count,
                    logical_output_width,
                },
                .route_count = route_count,
                .max_blocks = plan.max_blocks,
                .block_rows = plan.block_rows,
                .tile_columns = use_grouped_nax && !use_direct_nax
                    ? grouped_mmq_tile_columns(
                        plan.block_rows,
                        logical_output_width,
                        fused_swiglu)
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
                .vq_profile_mask = static_cast<int>(compilation.vq_profile_mask),
                .nint_group_sizes = impl_->nint_group_sizes,
                .cohort_populations = impl_->cohort_populations,
                .use_nax = use_grouped_nax,
                .direct_nax = use_direct_nax,
                .indirect_pages = impl_->indirect_addresses.has_value(),
                .swiglu_limit = swiglu_limit,
            });
        auto restored = moe_restore_routed_rows(sorted_outputs,route_order,ids);
        return merge_standalone(mlx::core::reshape(std::move(restored),output_shape));
    }
    if (impl_->native_primitive) {
        if (expert_map != nullptr) {
            kernel_inputs.push_back(map);
        }
        if (route_groups != nullptr) kernel_inputs.push_back(*route_groups);
        if (reduce_weights) kernel_inputs.push_back(mlx::core::contiguous(
            mlx::core::astype(*reduce_weights, mlx::core::float32)));
        if (impl_->indirect_addresses) {
            kernel_inputs[1] = *impl_->indirect_addresses;
            if (!impl_->page_resources)
                kernel_inputs.insert(kernel_inputs.end(), impl_->indirect_owners.begin(), impl_->indirect_owners.end());
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
                .nint_group_sizes = compilation.nint_group_sizes,
                .descriptor_size = kDescriptorSize,
                .variant_stride = variant_stride,
                .shared_input = static_cast<int>(shared_input),
                .vq_execution_layout = static_cast<int>(compilation.vq_execution_layout),
                .vq_layouts = compilation.vq_layouts,
                .family_mask = static_cast<int>(compilation.family_mask),
                .vq_profile_mask = static_cast<int>(
                    compilation.vq_profile_mask),
                .has_nepq_residual = static_cast<int>(
                    impl_->has_nepq_residual),
                .npq_grouped_indices = static_cast<int>(
                    impl_->npq_grouped_indices),
                .sorted_routes = static_cast<int>(
                    sorted_routes),
                .expert_map_size = expert_map_size,
                .packed_expert_ids = static_cast<int>(packed_expert_ids),
                .route_groups = route_groups != nullptr,
                .nint_q_dispatch = compilation.nint_q_dispatch,
                .indirect_pages = impl_->indirect_addresses.has_value(),
                .weighted_reduce = reduce_weights != nullptr,
                .workgroups = static_cast<int>(workgroups),
                .page_resources = impl_->page_resources,
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
            {
                "SHARED_INPUT",
                static_cast<int>(shared_input),
            },
            {
                "VQ_EXECUTION_LAYOUT",
                static_cast<int>(
                    impl_->vq_execution_layout),
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

std::size_t MlxMfeWeight::packed_copy_count() noexcept {
    return packed_copy_submissions.load(std::memory_order_relaxed);
}

std::size_t MlxMfeWeight::indirect_dispatch_count() noexcept {
    return indirect_expert_submissions.load(std::memory_order_relaxed);
}
std::size_t MlxMfeWeight::uncached_packed_bytes() noexcept {
    return uncached_packed_storage_bytes.load(std::memory_order_relaxed);
}

std::size_t MlxMfeWeight::pinned_packed_bytes() noexcept {
    return packed_pinned_bytes.load(std::memory_order_relaxed);
}

std::size_t MlxMfeWeight::packed_pin_failures() noexcept {
    return packed_pin_errors.load(std::memory_order_relaxed);
}

std::size_t MlxMfeWeight::pooled_staging_count() noexcept {
    return pooled_expert_staging.load(std::memory_order_relaxed);
}

std::shared_ptr<void> MlxMfeWeight::pin_packed_storage() const noexcept {
    struct Pin {
        explicit Pin(array value) : storage(std::move(value)) {}
        ~Pin() {
            if (!bytes) return;
            munlock(storage.data<std::uint8_t>(), bytes);
            packed_pinned_bytes.fetch_sub(bytes, std::memory_order_relaxed);
        }
        array storage;
        std::size_t bytes = 0;
    };
    try {
        if (!impl_->packed_arena || !impl_->packed_arena->is_available()
            || impl_->packed_arena->nbytes() < (std::size_t{64} << 10)) return {};
        const auto* data = impl_->packed_arena->data<std::uint8_t>();
        if (reinterpret_cast<std::uintptr_t>(data) % getpagesize()) return {};
        auto pin = std::make_shared<Pin>(*impl_->packed_arena);
        const auto bytes = pin->storage.buffer_size();
        if (mlock(data, bytes) != 0) {
            packed_pin_errors.fetch_add(1, std::memory_order_relaxed);
            return {};
        }
        pin->bytes = bytes;
        packed_pinned_bytes.fetch_add(bytes, std::memory_order_relaxed);
        return pin;
    } catch (...) {
        packed_pin_errors.fetch_add(1, std::memory_order_relaxed);
        return {};
    }
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
