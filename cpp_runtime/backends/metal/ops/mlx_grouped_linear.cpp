#include "mlx_grouped_linear.h"
#include "mlx_staging_allocator.h"

#include <mlx/allocator.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>

namespace mfq::metal {
namespace {

using mlx::core::CompileOptions;
using mlx::core::Dtype;
using mlx::core::MathMode;
using mlx::core::Shape;
using mlx::core::array;

constexpr int kDescriptorSize = 12;
constexpr int kFamily = 0;
constexpr int kOutput = 1;

constexpr int kFamilyNint8Zero = 1;
constexpr int kQ8Groups = 2;
constexpr int kQ8QOffset = 3;
constexpr int kQ8ScaleOffset = 4;

constexpr int kFamilyVq = 2;
constexpr int kFamilyMx = 3;

struct DirectProjectionLayout {
    int family = kFamilyNint8Zero;
    int output_size = 0;
    int bits = 0;
    int group_size = 0;
    int groups = 0;
    int vector_size = 0;
    int vectors = 0;
    int index_bits = 0;
    int state_bits = 0;
    int states = 0;
    int entries = 0;
    int code_banks = 0;
    int aux_mode = 0;
    int code_bank_mode = 0;
    int execution_layout = 0;
    int table_banks = 0;
    int groups_per_supergroup = 0;
    int supergroups = 0;
    int blocks = 0;
    int tile_begin = 0;
    int tile_end = 0;
    int output_offset = 0;
};

using RetainedProjection = std::variant<
    MlxNintWeight,
    MlxNint8ZeroWeight,
    MlxVqWeight,
    MlxFp8SqWeight,
    MlxMxfp4SqWeight,
    MlxMxWeight>;

RetainedProjection retain_projection(
    const MlxGroupedLinearWeightRef& value) {
    return std::visit(
        [](const auto* weight) -> RetainedProjection {
            if (weight == nullptr) {
                throw std::invalid_argument(
                    "grouped linear weight cannot be null");
            }
            using Weight = std::remove_cv_t<
                std::remove_pointer_t<decltype(weight)>>;
            if constexpr (std::is_same_v<Weight, array>) {
                throw MlxGroupedLinearUnsupported(
                    "dense projections require the dense grouped kernel");
            } else {
                return RetainedProjection(
                    std::in_place_type<Weight>,
                    *weight);
            }
        },
        value);
}

constexpr const char* kNintProjectionHeader = R"METAL(
template <typename Stream>
inline uint4 mfq_grouped_nint_read_row_value4(
    Stream stream,
    uint row_byte_offset,
    uint row_bit_shift,
    uint value_index,
    uint bits
) {
    uint row_relative_bits = row_bit_shift + value_index * bits;
    uint byte_index = row_byte_offset + (row_relative_bits >> 3u);
    uint shift = row_relative_bits & 7u;
    uint required_bits = shift + 4u * bits;
    const packed_uchar4 bytes =
        *reinterpret_cast<device const packed_uchar4*>(stream + byte_index);
    uint packed = as_type<uint>(bytes);
    if (shift != 0u) {
        packed = (packed >> shift)
            | (required_bits > 32u
                ? uint(stream[byte_index + 4u]) << (32u - shift)
                : 0u);
    }
    uint mask = (1u << bits) - 1u;
    return uint4(
        packed & mask,
        (packed >> bits) & mask,
        (packed >> (2u * bits)) & mask,
        (packed >> (3u * bits)) & mask);
}
)METAL";

constexpr const char* kGroupedHeader = R"METAL(
template <typename Stream>
inline uint mfq_grouped_vq_read_bits(
    Stream stream,
    uint value_index,
    uint bits
) {
    uint residual_bits = (value_index & 7u) * bits;
    uint byte_index =
        (value_index >> 3) * bits + (residual_bits >> 3);
    uint shift = residual_bits & 7u;
    uint packed = uint(stream[byte_index])
        | (uint(stream[byte_index + 1u]) << 8)
        | (uint(stream[byte_index + 2u]) << 16);
    return (packed >> shift)
        & ((1u << bits) - 1u);
}

template <typename Stream>
inline uint mfq_grouped_vq_read_4(
    Stream stream,
    uint value_index
) {
    uint packed = uint(stream[value_index >> 1u]);
    return (packed >> ((value_index & 1u) * 4u)) & 15u;
}

template <typename Stream>
inline uint mfq_grouped_vq_read_8(
    Stream stream,
    uint value_index
) {
    return uint(stream[value_index]);
}

template <typename Stream>
inline uint mfq_grouped_vq_read_12(
    Stream stream,
    uint value_index
) {
    uint odd = value_index & 1u;
    uint byte_index = (value_index >> 1u) * 3u + odd;
    uint packed = uint(stream[byte_index])
        | (uint(stream[byte_index + 1u]) << 8u);
    return (packed >> (odd * 4u)) & 4095u;
}

template <typename Stream>
inline uint2 mfq_grouped_vq_read_group64(
    Stream stream,
    uint record_index
) {
    return *(device const uint2*)(stream + record_index * 8u);
}

inline uint mfq_grouped_vq_group64_segment(
    uint2 record,
    uint local_vector
) {
    if (local_vector == 0u) {
        return record.x & 0xfffffu;
    }
    if (local_vector == 1u) {
        return ((record.x >> 20u) | (record.y << 12u)) & 0xfffffu;
    }
    return (record.y >> 8u) & 0xfffffu;
}

inline float mfq_grouped_mx_e8m0(uchar raw) {
    if (raw == 255u) {
        return NAN;
    }
    uint bits = raw == 0u ? 0x00400000u : uint(raw) << 23u;
    return as_type<float>(bits);
}

inline float mfq_grouped_mx_fp4(uchar raw) {
    uchar magnitude = raw & 7u;
    float value = magnitude == 0u ? 0.0f
        : (magnitude == 1u ? 0.5f
        : (magnitude == 2u ? 1.0f
        : (magnitude == 3u ? 1.5f
        : (magnitude == 4u ? 2.0f
        : (magnitude == 5u ? 3.0f
        : (magnitude == 6u ? 4.0f : 6.0f))))));
    return (raw & 8u) == 0u ? value : -value;
}

constant ushort mfq_grouped_mx_fp8_half_lut[256] = {
    0x0000u, 0x1800u, 0x1c00u, 0x1e00u, 0x2000u, 0x2100u, 0x2200u, 0x2300u,
    0x2400u, 0x2480u, 0x2500u, 0x2580u, 0x2600u, 0x2680u, 0x2700u, 0x2780u,
    0x2800u, 0x2880u, 0x2900u, 0x2980u, 0x2a00u, 0x2a80u, 0x2b00u, 0x2b80u,
    0x2c00u, 0x2c80u, 0x2d00u, 0x2d80u, 0x2e00u, 0x2e80u, 0x2f00u, 0x2f80u,
    0x3000u, 0x3080u, 0x3100u, 0x3180u, 0x3200u, 0x3280u, 0x3300u, 0x3380u,
    0x3400u, 0x3480u, 0x3500u, 0x3580u, 0x3600u, 0x3680u, 0x3700u, 0x3780u,
    0x3800u, 0x3880u, 0x3900u, 0x3980u, 0x3a00u, 0x3a80u, 0x3b00u, 0x3b80u,
    0x3c00u, 0x3c80u, 0x3d00u, 0x3d80u, 0x3e00u, 0x3e80u, 0x3f00u, 0x3f80u,
    0x4000u, 0x4080u, 0x4100u, 0x4180u, 0x4200u, 0x4280u, 0x4300u, 0x4380u,
    0x4400u, 0x4480u, 0x4500u, 0x4580u, 0x4600u, 0x4680u, 0x4700u, 0x4780u,
    0x4800u, 0x4880u, 0x4900u, 0x4980u, 0x4a00u, 0x4a80u, 0x4b00u, 0x4b80u,
    0x4c00u, 0x4c80u, 0x4d00u, 0x4d80u, 0x4e00u, 0x4e80u, 0x4f00u, 0x4f80u,
    0x5000u, 0x5080u, 0x5100u, 0x5180u, 0x5200u, 0x5280u, 0x5300u, 0x5380u,
    0x5400u, 0x5480u, 0x5500u, 0x5580u, 0x5600u, 0x5680u, 0x5700u, 0x5780u,
    0x5800u, 0x5880u, 0x5900u, 0x5980u, 0x5a00u, 0x5a80u, 0x5b00u, 0x5b80u,
    0x5c00u, 0x5c80u, 0x5d00u, 0x5d80u, 0x5e00u, 0x5e80u, 0x5f00u, 0x7e00u,
    0x8000u, 0x9800u, 0x9c00u, 0x9e00u, 0xa000u, 0xa100u, 0xa200u, 0xa300u,
    0xa400u, 0xa480u, 0xa500u, 0xa580u, 0xa600u, 0xa680u, 0xa700u, 0xa780u,
    0xa800u, 0xa880u, 0xa900u, 0xa980u, 0xaa00u, 0xaa80u, 0xab00u, 0xab80u,
    0xac00u, 0xac80u, 0xad00u, 0xad80u, 0xae00u, 0xae80u, 0xaf00u, 0xaf80u,
    0xb000u, 0xb080u, 0xb100u, 0xb180u, 0xb200u, 0xb280u, 0xb300u, 0xb380u,
    0xb400u, 0xb480u, 0xb500u, 0xb580u, 0xb600u, 0xb680u, 0xb700u, 0xb780u,
    0xb800u, 0xb880u, 0xb900u, 0xb980u, 0xba00u, 0xba80u, 0xbb00u, 0xbb80u,
    0xbc00u, 0xbc80u, 0xbd00u, 0xbd80u, 0xbe00u, 0xbe80u, 0xbf00u, 0xbf80u,
    0xc000u, 0xc080u, 0xc100u, 0xc180u, 0xc200u, 0xc280u, 0xc300u, 0xc380u,
    0xc400u, 0xc480u, 0xc500u, 0xc580u, 0xc600u, 0xc680u, 0xc700u, 0xc780u,
    0xc800u, 0xc880u, 0xc900u, 0xc980u, 0xca00u, 0xca80u, 0xcb00u, 0xcb80u,
    0xcc00u, 0xcc80u, 0xcd00u, 0xcd80u, 0xce00u, 0xce80u, 0xcf00u, 0xcf80u,
    0xd000u, 0xd080u, 0xd100u, 0xd180u, 0xd200u, 0xd280u, 0xd300u, 0xd380u,
    0xd400u, 0xd480u, 0xd500u, 0xd580u, 0xd600u, 0xd680u, 0xd700u, 0xd780u,
    0xd800u, 0xd880u, 0xd900u, 0xd980u, 0xda00u, 0xda80u, 0xdb00u, 0xdb80u,
    0xdc00u, 0xdc80u, 0xdd00u, 0xdd80u, 0xde00u, 0xde80u, 0xdf00u, 0xfe00u,
};

inline float mfq_grouped_mx_fp8(uchar raw) {
    return float(as_type<half>(
        mfq_grouped_mx_fp8_half_lut[uint(raw)]));
}

template <typename ValueStream, typename ScaleStream>
inline float mfq_grouped_mx_weight(
    ValueStream values,
    ScaleStream scales,
    uint output,
    uint column,
    uint bits,
    uint width
) {
    if (bits == 4u) {
        uchar packed = values[output * (width / 2u) + (column >> 1u)];
        uchar code = (column & 1u) == 0u
            ? packed & 15u
            : packed >> 4u;
        uchar scale = scales[output * (width / 32u) + column / 32u];
        return mfq_grouped_mx_fp4(code)
            * mfq_grouped_mx_e8m0(scale);
    }
    uchar code = values[output * width + column];
    uchar scale = scales[
        (output / 128u) * (width / 128u) + column / 128u];
    return mfq_grouped_mx_fp8(code)
        * mfq_grouped_mx_e8m0(scale);
}

template <
    typename IndicesPtr,
    typename StatePtr,
    typename AuxPtr,
    typename AnchorPtr,
    typename CodebookPtr,
    typename ScalePtr,
    typename StateBankPtr,
    typename BankPtr,
    typename ParameterPtr
>
inline float mfq_grouped_vq_decode_weight(
    IndicesPtr indices_packed,
    StatePtr state_packed,
    AuxPtr aux_packed,
    AnchorPtr anchors,
    CodebookPtr codebooks,
    ScalePtr scale_lut,
    StateBankPtr state_to_codebank,
    BankPtr bank_ids,
    ParameterPtr parameters,
    uint output,
    uint column,
    uint group_size,
    uint groups,
    uint vector_size,
    uint vectors,
    uint index_bits,
    uint state_bits,
    uint states,
    uint entries,
    uint code_banks,
    uint aux_mode,
    uint code_bank_mode,
    uint execution_layout,
    uint has_table_banks,
    uint groups_per_supergroup,
    uint supergroups,
    uint sign_groups
) {
    uint group = column / group_size;
    uint vector = column / vector_size;
    uint component =
        column - vector * vector_size;
    uint state_index =
        output * groups + group;
    uint state;
    uint index;
    uint auxiliary = 0u;
    if (execution_layout == 1u) {
        uint2 record = mfq_grouped_vq_read_group64(
            indices_packed,
            state_index);
        uint segment = mfq_grouped_vq_group64_segment(
            record,
            (column - group * group_size) / vector_size);
        state = record.y >> 28u;
        index = segment & 4095u;
        auxiliary = segment >> 12u;
    } else {
        state = mfq_grouped_vq_read_bits(
            state_packed,
            state_index,
            state_bits);
        index = mfq_grouped_vq_read_bits(
            indices_packed,
            output * vectors + vector,
            index_bits);
        if (aux_mode == 1u || aux_mode == 2u) {
            auxiliary = mfq_grouped_vq_read_bits(
                aux_packed,
                output * sign_groups + column / 8u,
                7u);
        } else if (aux_mode == 3u) {
            auxiliary = mfq_grouped_vq_read_bits(
                aux_packed,
                state_index,
                1u);
        }
    }
    uint table_bank = 0u;
    if (has_table_banks != 0u) {
        table_bank = uint(bank_ids[
            output * supergroups
            + group / groups_per_supergroup
        ]);
    }
    uint code_bank = 0u;
    if (code_bank_mode == 1u) {
        code_bank =
            uint(state_to_codebank[state]);
    } else if (code_bank_mode == 2u) {
        code_bank = auxiliary;
    }
    uint code_offset = (
        (
            (
                table_bank * code_banks
                + code_bank
            )
            * entries + index
        )
        * vector_size + component
    );
    float code = float(codebooks[code_offset]);
    if (aux_mode == 1u || aux_mode == 2u) {
        uint sign_position = column & 7u;
        uint negative = execution_layout == 1u
            ? ((auxiliary >> sign_position) & 1u)
            : (sign_position < 7u
                ? ((auxiliary >> sign_position) & 1u)
                : (popcount(auxiliary) & 1u));
        if (
            aux_mode == 2u
            && sign_position == 7u
        ) {
            negative ^= (index >> 7u) & 1u;
        }
        code = negative != 0u ? -code : code;
    } else if (aux_mode == 3u) {
        code += auxiliary != 0u
            ? -parameters[0]
            : parameters[0];
    }
    return anchors[output]
        * scale_lut[table_bank * states + state]
        * code;
}
)METAL";

constexpr const char* kGroupedSource = R"METAL(
    constexpr uint ROWS_PER_SIMD = 4u;
    constexpr uint ROWS_PER_TG = 8u;

    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint workgroup = threadgroup_position_in_grid.x;
    uint input_row = workgroup / uint(TOTAL_TILES);
    uint global_tile =
        workgroup - input_row * uint(TOTAL_TILES);
    if (input_row >= uint(ROWS)) {
        return;
    }

    uint projection = 0u;
    while (
        projection + 1u < uint(PROJECTIONS)
        && global_tile >=
            uint(projection_tile_offsets[projection + 1u])
    ) {
        ++projection;
    }
    uint local_tile =
        global_tile - uint(projection_tile_offsets[projection]);
    uint descriptor_base =
        projection * uint(DESCRIPTOR_SIZE);
    uint output_width =
        uint(descriptors[descriptor_base + 1u]);
    uint output_base =
        local_tile * ROWS_PER_TG
        + simd_group * ROWS_PER_SIMD;
    uint output_offset =
        uint(projection_output_offsets[projection]);
    float accumulators[ROWS_PER_SIMD] = {0.0f};
    for (uint column = lane; column < uint(K); column += 32u) {
        float activation =
            float(x[input_row * uint(K) + column]);
        for (
            uint local_row = 0u;
            local_row < ROWS_PER_SIMD;
            ++local_row
        ) {
            uint output = output_base + local_row;
            if (output >= output_width) {
                continue;
            }

            uint groups =
                uint(descriptors[descriptor_base + 2u]);
            uint group = column >> 5;
            float weight =
                float(q8_scales[
                    uint(descriptors[descriptor_base + 4u])
                    + output * groups + group])
                * float(q8_q[
                    uint(descriptors[descriptor_base + 3u])
                    + output * uint(K) + column]);
            accumulators[local_row] = fma(
                activation,
                weight,
                accumulators[local_row]);
        }
    }

    for (
        uint local_row = 0u;
        local_row < ROWS_PER_SIMD;
        ++local_row
    ) {
        float total = simd_sum(accumulators[local_row]);
        uint output = output_base + local_row;
        if (lane == 0u && output < output_width) {
            y[
                input_row * uint(TOTAL_OUT)
                + output_offset + output
            ] = T(total);
        }
    }
)METAL";

std::int32_t checked_int(
    std::size_t value,
    const char* name) {
    if (value >
        static_cast<std::size_t>(
            std::numeric_limits<std::int32_t>::max())) {
        throw MlxGroupedLinearUnsupported(
            std::string("grouped linear ") + name
            + " exceeds int32 range");
    }
    return static_cast<std::int32_t>(value);
}

void append_raw(
    std::vector<std::uint8_t>& target,
    const array& source,
    Dtype expected,
    const char* name) {
    if (source.dtype() != expected ||
        !source.flags().row_contiguous) {
        throw std::runtime_error(
            std::string("invalid grouped linear packed ") + name);
    }
    auto evaluated = source;
    evaluated.eval();
    const auto previous = target.size();
    if (evaluated.nbytes() >
        std::numeric_limits<std::size_t>::max() - previous) {
        throw MlxGroupedLinearUnsupported(
            "grouped linear packed stream size overflows");
    }
    target.resize(previous + evaluated.nbytes());
    std::memcpy(
        target.data() + previous,
        evaluated.data<std::uint8_t>(),
        evaluated.nbytes());
}

array make_raw_array(
    std::vector<std::uint8_t> bytes,
    Dtype dtype) {
    if (bytes.empty()) {
        bytes.resize(dtype.size(), 0);
    }
    if (bytes.size() % dtype.size() != 0) {
        throw std::runtime_error(
            "grouped linear packed stream is misaligned");
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

mlx::core::fast::CustomKernelFunction make_grouped_kernel() {
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    return mlx::core::fast::metal_kernel(
        "mfq_cpp_heterogeneous_grouped_linear",
        {
            "descriptors",
            "projection_tile_offsets",
            "projection_output_offsets",
            "q8_q",
            "q8_scales",
            "x",
        },
        {"y"},
        kGroupedSource,
        kGroupedHeader,
        true,
        false,
        options);
}

const mlx::core::fast::CustomKernelFunction& grouped_kernel() {
    static const auto kernel = make_grouped_kernel();
    return kernel;
}

std::vector<std::string> nint_projection_group_input_names(
    std::size_t projections) {
    std::vector<std::string> names;
    names.reserve(projections * 7 + 1);
    for (std::size_t projection = 0;
         projection < projections;
         ++projection) {
        const auto suffix = std::to_string(projection);
        names.push_back("q_packed_" + suffix);
        names.push_back("row_q_layout_" + suffix);
        names.push_back("row_q_byte_offsets_" + suffix);
        names.push_back("sub_scale_" + suffix);
        names.push_back("sub_min_" + suffix);
        names.push_back("neuron_scale_" + suffix);
        names.push_back("neuron_min_" + suffix);
    }
    names.emplace_back("x");
    names.emplace_back("params");
    return names;
}

std::string make_nint_projection_group_source(
    std::size_t projections) {
    std::string source = R"METAL(
    constexpr uint SIMD_GROUPS = 8u;
    constexpr uint OUTPUTS_PER_SIMD = 2u;
    constexpr uint OUTPUTS_PER_TG = SIMD_GROUPS * OUTPUTS_PER_SIMD;
    constexpr uint CHUNKS = (uint(GS) + 3u) / 4u;

    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint output_base =
        threadgroup_position_in_grid.x * OUTPUTS_PER_TG
        + simd_group * OUTPUTS_PER_SIMD;
    uint first_row = threadgroup_position_in_grid.y * uint(TILE_M);
    if (output_base >= uint(MAX_OUT) || first_row >= uint(M)) {
        return;
    }
)METAL";

    for (std::size_t projection = 0;
         projection < projections;
         ++projection) {
        const auto suffix = std::to_string(projection);
        source +=
            "    bool active_" + suffix
            + " = output_base < uint(P" + suffix + "_OUT);\n"
            "    uint outputs_" + suffix
            + "[OUTPUTS_PER_SIMD];\n"
            "    uint metadata_bases_" + suffix
            + "[OUTPUTS_PER_SIMD];\n"
            "    uint q_widths_" + suffix
            + "[OUTPUTS_PER_SIMD];\n"
            "    uint q_byte_offsets_" + suffix
            + "[OUTPUTS_PER_SIMD];\n"
            "    uint q_bit_shifts_" + suffix
            + "[OUTPUTS_PER_SIMD];\n"
            "    float neuron_scales_" + suffix
            + "[OUTPUTS_PER_SIMD];\n"
            "    float neuron_minimums_" + suffix
            + "[OUTPUTS_PER_SIMD];\n"
            "    float accumulators_" + suffix
            + "[OUTPUTS_PER_SIMD][TILE_M];\n"
            "    for (uint output_row = 0u;\n"
            "         output_row < OUTPUTS_PER_SIMD;\n"
            "         ++output_row) {\n"
            "        uint output = min(\n"
            "            output_base + output_row,\n"
            "            uint(P" + suffix + "_OUT) - 1u);\n"
            "        outputs_" + suffix + "[output_row] = output;\n"
            "        metadata_bases_" + suffix
            + "[output_row] = output * uint(NG);\n"
            "        uint row_layout = uint(row_q_layout_" + suffix
            + "[output]);\n"
            "        q_widths_" + suffix
            + "[output_row] = row_layout & 15u;\n"
            "        q_byte_offsets_" + suffix
            + "[output_row] = row_q_byte_offsets_" + suffix
            + "[output];\n"
            "        q_bit_shifts_" + suffix
            + "[output_row] = row_layout >> 4u;\n"
            "        neuron_scales_" + suffix
            + "[output_row] = neuron_scale_" + suffix
            + "[output];\n"
            "        neuron_minimums_" + suffix
            + "[output_row] = neuron_min_" + suffix
            + "[output];\n"
            "        for (uint local_row = 0u;\n"
            "             local_row < uint(TILE_M);\n"
            "             ++local_row) {\n"
            "            accumulators_" + suffix
            + "[output_row][local_row] = 0.0f;\n"
            "        }\n"
            "    }\n";
    }

    source += R"METAL(
    for (uint group = lane; group < uint(NG); group += 32u) {
        float activation_sums[TILE_M];
        for (uint local_row = 0u;
             local_row < uint(TILE_M);
             ++local_row) {
            activation_sums[local_row] = 0.0f;
        }
)METAL";

    for (std::size_t projection = 0;
         projection < projections;
         ++projection) {
        const auto suffix = std::to_string(projection);
        source +=
            "        float quantized_dots_" + suffix
            + "[OUTPUTS_PER_SIMD][TILE_M];\n"
            "        for (uint output_row = 0u;\n"
            "             output_row < OUTPUTS_PER_SIMD;\n"
            "             ++output_row) {\n"
            "            for (uint local_row = 0u;\n"
            "                 local_row < uint(TILE_M);\n"
            "                 ++local_row) {\n"
            "                quantized_dots_" + suffix
            + "[output_row][local_row] = 0.0f;\n"
            "            }\n"
            "        }\n";
    }

    source += R"METAL(
        for (uint chunk = 0u; chunk < CHUNKS; ++chunk) {
            uint group_element = chunk * 4u;
            uint column = group * uint(GS) + group_element;
)METAL";

    for (std::size_t projection = 0;
         projection < projections;
         ++projection) {
        const auto suffix = std::to_string(projection);
        source +=
            "            uint4 codes_" + suffix
            + "[OUTPUTS_PER_SIMD];\n"
            "            if (active_" + suffix + ") {\n"
            "                for (uint output_row = 0u;\n"
            "                     output_row < OUTPUTS_PER_SIMD;\n"
            "                     ++output_row) {\n"
            "                    codes_" + suffix
            + "[output_row] = mfq_grouped_nint_read_row_value4(\n"
            "                        q_packed_" + suffix + ",\n"
            "                        q_byte_offsets_" + suffix
            + "[output_row],\n"
            "                        q_bit_shifts_" + suffix
            + "[output_row],\n"
            "                        column,\n"
            "                        q_widths_" + suffix
            + "[output_row]);\n"
            "                }\n"
            "            }\n";
    }

    source += R"METAL(
            for (uint local_row = 0u;
                 local_row < uint(TILE_M);
                 ++local_row) {
                uint row = first_row + local_row;
                float4 activation = float4(0.0f);
                if (row < uint(M)) {
                    uint input_base = row * uint(K) + column;
                    if (group_element + 3u < uint(GS) &&
                        column + 3u < uint(K)) {
                        const vec<T, 4> packed_activation =
                            *reinterpret_cast<device const vec<T, 4>*>(
                                x + input_base);
                        activation = float4(packed_activation);
                    } else {
                        activation.x = column < uint(K)
                            ? float(x[input_base]) : 0.0f;
                        activation.y = group_element + 1u < uint(GS) &&
                                column + 1u < uint(K)
                            ? float(x[input_base + 1u]) : 0.0f;
                        activation.z = group_element + 2u < uint(GS) &&
                                column + 2u < uint(K)
                            ? float(x[input_base + 2u]) : 0.0f;
                        activation.w = group_element + 3u < uint(GS) &&
                                column + 3u < uint(K)
                            ? float(x[input_base + 3u]) : 0.0f;
                    }
                }
                activation_sums[local_row] +=
                    activation.x + activation.y
                    + activation.z + activation.w;
)METAL";

    for (std::size_t projection = 0;
         projection < projections;
         ++projection) {
        const auto suffix = std::to_string(projection);
        source +=
            "                if (active_" + suffix + ") {\n"
            "                    for (uint output_row = 0u;\n"
            "                         output_row < OUTPUTS_PER_SIMD;\n"
            "                         ++output_row) {\n"
            "                        quantized_dots_" + suffix
            + "[output_row][local_row] += dot(\n"
            "                            activation,\n"
            "                            float4(codes_" + suffix
            + "[output_row]));\n"
            "                    }\n"
            "                }\n";
    }

    source += R"METAL(
            }
        }
)METAL";

    for (std::size_t projection = 0;
         projection < projections;
         ++projection) {
        const auto suffix = std::to_string(projection);
        source +=
            "        if (active_" + suffix + ") {\n"
            "            for (uint output_row = 0u;\n"
            "                 output_row < OUTPUTS_PER_SIMD;\n"
            "                 ++output_row) {\n"
            "                uint metadata_index = metadata_bases_"
            + suffix + "[output_row] + group;\n"
            "                float scale = neuron_scales_" + suffix
            + "[output_row]\n"
            "                    * float(sub_scale_" + suffix
            + "[metadata_index]);\n"
            "                float minimum = neuron_minimums_" + suffix
            + "[output_row]\n"
            "                    * float(sub_min_" + suffix
            + "[metadata_index]);\n"
            "                for (uint local_row = 0u;\n"
            "                     local_row < uint(TILE_M);\n"
            "                     ++local_row) {\n"
            "                    accumulators_" + suffix
            + "[output_row][local_row] = fma(\n"
            "                        scale,\n"
            "                        quantized_dots_" + suffix
            + "[output_row][local_row],\n"
            "                        fma(\n"
            "                            -minimum,\n"
            "                            activation_sums[local_row],\n"
            "                            accumulators_" + suffix
            + "[output_row][local_row]));\n"
            "                }\n"
            "            }\n"
            "        }\n";
    }

    source += "    }\n";

    if (projections == 2) {
        source += R"METAL(
    if (uint(SWIGLU) != 0u) {
        if (active_0 && active_1) {
            for (uint output_row = 0u;
                 output_row < OUTPUTS_PER_SIMD;
                 ++output_row) {
                uint output = output_base + output_row;
                for (uint local_row = 0u;
                     local_row < uint(TILE_M);
                     ++local_row) {
                    float gate = float(T(simd_sum(
                        accumulators_0[output_row][local_row])));
                    float up = float(T(simd_sum(
                        accumulators_1[output_row][local_row])));
                    uint row = first_row + local_row;
                    if (lane == 0u &&
                        output < uint(P0_OUT) &&
                        row < uint(M)) {
                        if (params[0] > 0.0f) {
                            gate = min(gate, params[0]);
                            up = clamp(up, -params[0], params[0]);
                        }
                        y[row * uint(P0_OUT) + output] =
                            T((gate / (1.0f + exp(-gate))) * up);
                    }
                }
            }
        }
        return;
    }
)METAL";
    }

    for (std::size_t projection = 0;
         projection < projections;
         ++projection) {
        const auto suffix = std::to_string(projection);
        source +=
            "    if (active_" + suffix + ") {\n"
            "        for (uint output_row = 0u;\n"
            "             output_row < OUTPUTS_PER_SIMD;\n"
            "             ++output_row) {\n"
            "            uint output = output_base + output_row;\n"
            "            for (uint local_row = 0u;\n"
            "                 local_row < uint(TILE_M);\n"
            "                 ++local_row) {\n"
            "                float total = simd_sum(\n"
            "                    accumulators_" + suffix
            + "[output_row][local_row]);\n"
            "                uint row = first_row + local_row;\n"
            "                if (lane == 0u &&\n"
            "                    output < uint(P" + suffix + "_OUT) &&\n"
            "                    row < uint(M)) {\n"
            "                    y[row * uint(TOTAL_OUT)\n"
            "                        + uint(P" + suffix + "_OFFSET)\n"
            "                        + output] = T(total);\n"
            "                }\n"
            "            }\n"
            "        }\n"
            "    }\n";
    }
    return source;
}

mlx::core::fast::CustomKernelFunction nint_projection_group_kernel(
    std::size_t projections) {
    static std::mutex mutex;
    static std::unordered_map<
        std::size_t,
        mlx::core::fast::CustomKernelFunction> kernels;
    std::lock_guard<std::mutex> lock(mutex);
    const auto found = kernels.find(projections);
    if (found != kernels.end()) {
        return found->second;
    }
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    auto kernel = mlx::core::fast::metal_kernel(
        "mfq_cpp_nint_metadata_grouped_p" +
            std::to_string(projections),
        nint_projection_group_input_names(projections),
        {"y"},
        make_nint_projection_group_source(projections),
        kNintProjectionHeader,
        true,
        false,
        options);
    kernels.emplace(projections, kernel);
    return kernel;
}

std::vector<std::string> dense_projection_group_input_names(
    std::size_t projections) {
    std::vector<std::string> names;
    names.reserve(projections + 1);
    for (std::size_t projection = 0;
         projection < projections;
         ++projection) {
        names.push_back("w_" + std::to_string(projection));
    }
    names.push_back("x");
    return names;
}

std::string make_dense_projection_group_source(
    std::size_t projections) {
    std::string source = R"METAL(
    uint tile = threadgroup_position_in_grid.x;
    uint lane = thread_index_in_simdgroup;
    uint output_lane = simdgroup_index_in_threadgroup;
)METAL";
    for (std::size_t projection = 0;
         projection < projections;
         ++projection) {
        const auto suffix = std::to_string(projection);
        source +=
            "    if (tile >= uint(P" + suffix + "_TILE_BEGIN) &&\n"
            "        tile < uint(P" + suffix + "_TILE_END)) {\n"
            "        uint output = (tile - uint(P" + suffix
            + "_TILE_BEGIN)) * 4u + output_lane;\n"
            "        if (output >= uint(P" + suffix + "_OUT)) return;\n"
            "        float accumulators[M];\n"
            "        for (uint row = 0u; row < uint(M); ++row) {\n"
            "            accumulators[row] = 0.0f;\n"
            "        }\n"
            "        for (uint column = lane; column < uint(K); "
            "column += 32u) {\n"
            "            float weight = float(w_" + suffix
            + "[output * uint(K) + column]);\n"
            "            for (uint row = 0u; row < uint(M); ++row) {\n"
            "                accumulators[row] = fma(\n"
            "                    float(x[row * uint(K) + column]),\n"
            "                    weight,\n"
            "                    accumulators[row]);\n"
            "            }\n"
            "        }\n"
            "        for (uint row = 0u; row < uint(M); ++row) {\n"
            "            float total = simd_sum(accumulators[row]);\n"
            "            if (lane == 0u) {\n"
            "                y[row * uint(TOTAL_OUT) + uint(P" + suffix
            + "_OFFSET) + output] = T(total);\n"
            "            }\n"
            "        }\n"
            "        return;\n"
            "    }\n";
    }
    return source;
}

mlx::core::fast::CustomKernelFunction dense_projection_group_kernel(
    std::size_t projections) {
    static std::mutex mutex;
    static std::unordered_map<
        std::size_t,
        mlx::core::fast::CustomKernelFunction> kernels;
    std::lock_guard<std::mutex> lock(mutex);
    const auto found = kernels.find(projections);
    if (found != kernels.end()) {
        return found->second;
    }
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    auto kernel = mlx::core::fast::metal_kernel(
        "mfq_cpp_dense_projection_group_p" +
            std::to_string(projections),
        dense_projection_group_input_names(projections),
        {"y"},
        make_dense_projection_group_source(projections),
        "",
        true,
        false,
        options);
    kernels.emplace(projections, kernel);
    return kernel;
}

std::string direct_kernel_key(
    const std::vector<DirectProjectionLayout>& layouts) {
    std::string key;
    for (const auto& layout : layouts) {
        if (!key.empty()) {
            key.push_back('_');
        }
        if (layout.family == kFamilyNint8Zero) {
            key += "q8";
        } else if (layout.family == kFamilyVq) {
            key += layout.execution_layout == 1 ? "vqg64" : "vq";
        } else if (layout.family == kFamilyMx) {
            key += "mx" + std::to_string(layout.bits);
        } else {
            key += "unknown";
        }
    }
    return key;
}

bool supports_single_row_mxfp8_fast_path(
    const std::vector<DirectProjectionLayout>& layouts) noexcept {
    if (layouts.size() < 2 || layouts.size() > 14) {
        return false;
    }
    bool found_mxfp8 = false;
    for (const auto& layout : layouts) {
        if (layout.family == kFamilyMx && layout.bits == 8) {
            found_mxfp8 = true;
            continue;
        }
        if (layout.family != kFamilyNint8Zero) {
            return false;
        }
    }
    return found_mxfp8;
}

bool supports_small_m_group64_output_tile(
    const std::vector<DirectProjectionLayout>& layouts) noexcept {
    if (layouts.size() != 2) {
        return false;
    }
    for (const auto& layout : layouts) {
        if (layout.family != kFamilyVq ||
            layout.execution_layout != 1 ||
            layout.group_size != 24 ||
            layout.vector_size != 8 ||
            layout.index_bits != 12 ||
            layout.state_bits != 4 ||
            layout.entries != 4096 ||
            layout.code_bank_mode == 2 ||
            layout.aux_mode < 1 || layout.aux_mode > 2) {
            return false;
        }
    }
    return true;
}

std::vector<std::string> direct_input_names(
    const std::vector<DirectProjectionLayout>& layouts) {
    std::vector<std::string> names;
    names.reserve(layouts.size() * 9 + 1);
    for (
        std::size_t projection = 0;
        projection < layouts.size();
        ++projection
    ) {
        const auto suffix = std::to_string(projection);
        if (
            layouts[projection].family
                == kFamilyNint8Zero
        ) {
            names.push_back("q8_q_" + suffix);
            names.push_back("q8_scales_" + suffix);
        } else if (
            layouts[projection].family == kFamilyVq
        ) {
            names.push_back("vq_indices_" + suffix);
            names.push_back("vq_state_" + suffix);
            names.push_back("vq_aux_" + suffix);
            names.push_back("vq_anchors_" + suffix);
            names.push_back("vq_codebooks_" + suffix);
            names.push_back("vq_scales_" + suffix);
            names.push_back("vq_state_banks_" + suffix);
            names.push_back("vq_bank_ids_" + suffix);
            names.push_back("vq_parameters_" + suffix);
        } else if (
            layouts[projection].family == kFamilyMx
        ) {
            names.push_back("mx_values_" + suffix);
            names.push_back("mx_scales_" + suffix);
        }
    }
    names.emplace_back("x");
    return names;
}

std::string make_direct_small_m_source(
    const std::vector<DirectProjectionLayout>& layouts) {
    std::string source = R"METAL(
    constexpr uint ROWS_PER_SIMD = 4u;
    constexpr uint ROWS_PER_TG = 8u;

    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint global_tile = threadgroup_position_in_grid.x;
)METAL";

    for (std::size_t projection = 0;
         projection < layouts.size();
         ++projection) {
        const auto suffix = std::to_string(projection);
        source += projection == 0 ? "    if (" : "    else if (";
        source += "global_tile >= uint(P" + suffix
            + "_TILE_BEGIN) && global_tile < uint(P" + suffix
            + "_TILE_END)) {\n";
        source +=
            "        uint local_tile = global_tile - uint(P" + suffix
            + "_TILE_BEGIN);\n"
            "        uint output_base = local_tile * ROWS_PER_TG"
            " + simd_group * ROWS_PER_SIMD;\n"
            "        float accumulators[ROWS][ROWS_PER_SIMD];\n"
            "        for (uint input_row = 0u; input_row < uint(ROWS);"
            " ++input_row) {\n"
            "            for (uint local_row = 0u;"
            " local_row < ROWS_PER_SIMD; ++local_row) {\n"
            "                accumulators[input_row][local_row] = 0.0f;\n"
            "            }\n"
            "        }\n"
            "        for (uint column = lane; column < uint(K);"
            " column += 32u) {\n"
            "            float activations[ROWS];\n"
            "            for (uint input_row = 0u;"
            " input_row < uint(ROWS); ++input_row) {\n"
            "                activations[input_row] ="
            " float(x[input_row * uint(K) + column]);\n"
            "            }\n"
            "            for (uint local_row = 0u;"
            " local_row < ROWS_PER_SIMD; ++local_row) {\n"
            "                uint output = output_base + local_row;\n"
            "                if (output >= uint(P" + suffix
            + "_OUT)) { continue; }\n"
            "                float weight = 0.0f;\n";

        if (layouts[projection].family == kFamilyNint8Zero) {
            source +=
                "                uint group = column >> 5;\n"
                "                weight = float(q8_scales_" + suffix
                + "[output * uint(P" + suffix + "_NG) + group])"
                " * float(q8_q_" + suffix
                + "[output * uint(K) + column]);\n";
        } else {
            source +=
                "                uint group = column / uint(P" + suffix
                + "_GS);\n"
                "                uint vector = column / uint(P" + suffix
                + "_VECTOR_SIZE);\n"
                "                uint component = column - vector"
                " * uint(P" + suffix + "_VECTOR_SIZE);\n"
                "                uint state_index = output * uint(P"
                + suffix + "_NG) + group;\n"
                "                uint state = ";
            if (layouts[projection].state_bits == 4) {
                source += "mfq_grouped_vq_read_4(vq_state_" + suffix
                    + ", state_index);\n";
            } else if (layouts[projection].state_bits == 8) {
                source += "mfq_grouped_vq_read_8(vq_state_" + suffix
                    + ", state_index);\n";
            } else {
                source +=
                    "mfq_grouped_vq_read_bits(vq_state_" + suffix
                    + ", state_index, uint(P" + suffix
                    + "_STATE_BITS));\n";
            }
            source +=
                "                uint table_bank = 0u;\n";
            if (layouts[projection].table_banks > 1) {
                source +=
                    "                table_bank = uint(vq_bank_ids_"
                    + suffix + "[output * uint(P" + suffix
                    + "_NSUPER) + group / uint(P" + suffix
                    + "_GROUPS_PER_SUPER)]);\n";
            }
            source += "                uint auxiliary = 0u;\n";
            if (layouts[projection].aux_mode == 3) {
                source +=
                    "                auxiliary ="
                    " mfq_grouped_vq_read_bits(\n"
                    "                    vq_aux_" + suffix + ",\n"
                    "                    state_index, 1u);\n";
            }
            source +=
                "                uint index_position = output * uint(P"
                + suffix + "_NVEC) + vector;\n"
                "                uint index = ";
            if (layouts[projection].index_bits == 4) {
                source += "mfq_grouped_vq_read_4(vq_indices_" + suffix
                    + ", index_position);\n";
            } else if (layouts[projection].index_bits == 8) {
                source += "mfq_grouped_vq_read_8(vq_indices_" + suffix
                    + ", index_position);\n";
            } else if (layouts[projection].index_bits == 12) {
                source += "mfq_grouped_vq_read_12(vq_indices_" + suffix
                    + ", index_position);\n";
            } else {
                source +=
                    "mfq_grouped_vq_read_bits(vq_indices_" + suffix
                    + ", index_position, uint(P" + suffix
                    + "_INDEX_BITS));\n";
            }
            if (layouts[projection].aux_mode == 1
                || layouts[projection].aux_mode == 2) {
                source +=
                    "                auxiliary ="
                    " mfq_grouped_vq_read_bits(\n"
                    "                    vq_aux_" + suffix + ",\n"
                    "                    output * ((uint(K) + 7u) / 8u)"
                    " + column / 8u, 7u);\n";
            }
            source += "                uint code_bank = 0u;\n";
            if (layouts[projection].code_bank_mode == 1) {
                source +=
                    "                code_bank = uint(vq_state_banks_"
                    + suffix + "[state]);\n";
            } else if (layouts[projection].code_bank_mode == 2) {
                source +=
                    "                code_bank = auxiliary;\n";
            }
            source +=
                "                uint code_offset = (((table_bank"
                " * uint(P" + suffix + "_CODE_BANKS) + code_bank)"
                " * uint(P" + suffix + "_ENTRIES) + index)"
                " * uint(P" + suffix + "_VECTOR_SIZE) + component);\n"
                "                float code ="
                " float(vq_codebooks_" + suffix + "[code_offset]);\n";
            if (layouts[projection].aux_mode == 1
                || layouts[projection].aux_mode == 2) {
                source +=
                    "                uint sign_position = column & 7u;\n"
                    "                uint negative = sign_position < 7u"
                    " ? ((auxiliary >> sign_position) & 1u)"
                    " : (popcount(auxiliary) & 1u);\n";
                if (layouts[projection].aux_mode == 2) {
                    source +=
                        "                if (sign_position == 7u) {"
                        " negative ^= (index >> 7u) & 1u; }\n";
                }
                source +=
                    "                code = negative != 0u"
                    " ? -code : code;\n";
            } else if (layouts[projection].aux_mode == 3) {
                source +=
                    "                code += auxiliary != 0u"
                    " ? -vq_parameters_" + suffix + "[0]"
                    " : vq_parameters_" + suffix + "[0];\n";
            }
            source +=
                "                weight = vq_anchors_" + suffix
                + "[output] * vq_scales_" + suffix
                + "[table_bank * uint(P" + suffix
                + "_STATES) + state] * code;\n";
        }

        source +=
            "                for (uint input_row = 0u;"
            " input_row < uint(ROWS); ++input_row) {\n"
            "                    accumulators[input_row][local_row] = fma(\n"
            "                        activations[input_row], weight,\n"
            "                        accumulators[input_row][local_row]);\n"
            "                }\n"
            "            }\n"
            "        }\n"
            "        for (uint input_row = 0u;"
            " input_row < uint(ROWS); ++input_row) {\n"
            "            for (uint local_row = 0u;"
            " local_row < ROWS_PER_SIMD; ++local_row) {\n"
            "                float total ="
            " simd_sum(accumulators[input_row][local_row]);\n"
            "                uint output = output_base + local_row;\n"
            "                if (lane == 0u && output < uint(P" + suffix
            + "_OUT)) {\n"
            "                    y[input_row * uint(TOTAL_OUT) + uint(P"
            + suffix + "_OUT_OFFSET) + output] = T(total);\n"
            "                }\n"
            "            }\n"
            "        }\n"
            "    }\n";
    }
    return source;
}

std::string make_direct_source(
    const std::vector<DirectProjectionLayout>& layouts,
    bool batch_rows) {
    const bool supports_small_m_specialization =
        batch_rows && std::all_of(
            layouts.begin(),
            layouts.end(),
            [](const DirectProjectionLayout& layout) {
                return (layout.family == kFamilyNint8Zero
                    || layout.family == kFamilyVq)
                    && layout.execution_layout == 0;
            });
    if (supports_small_m_specialization) {
        return make_direct_small_m_source(layouts);
    }
    std::string source = batch_rows ? R"METAL(
    constexpr uint ROWS_PER_SIMD = 4u;
    constexpr uint ROWS_PER_TG = 8u;

    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint global_tile = threadgroup_position_in_grid.x;

    uint projection = 0u;
    uint local_tile = 0u;
    uint output_width = 0u;
    uint output_offset = 0u;
)METAL" : R"METAL(
    constexpr uint ROWS_PER_SIMD = 4u;
    constexpr uint ROWS_PER_TG = 8u;

    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint workgroup = threadgroup_position_in_grid.x;
    uint input_row = workgroup / uint(TOTAL_TILES);
    uint global_tile =
        workgroup - input_row * uint(TOTAL_TILES);
    if (input_row >= uint(ROWS)) {
        return;
    }

    uint projection = 0u;
    uint local_tile = 0u;
    uint output_width = 0u;
    uint output_offset = 0u;
)METAL";

    for (
        std::size_t projection = 0;
        projection < layouts.size();
        ++projection
    ) {
        const auto suffix = std::to_string(projection);
        if (projection == 0) {
            source += "    if (";
        } else if (projection + 1 < layouts.size()) {
            source += " else if (";
        } else {
            source += " else {\n";
            source += "        projection = "
                + suffix + "u;\n";
            source += "        local_tile = global_tile"
                " - uint(P" + suffix + "_TILE_BEGIN);\n";
            source += "        output_width = uint(P"
                + suffix + "_OUT);\n";
            source += "        output_offset = uint(P"
                + suffix + "_OUT_OFFSET);\n";
            source += "    }\n";
            continue;
        }
        source += "global_tile < uint(P"
            + suffix + "_TILE_END)) {\n";
        source += "        projection = "
            + suffix + "u;\n";
        source += "        local_tile = global_tile"
            " - uint(P" + suffix + "_TILE_BEGIN);\n";
        source += "        output_width = uint(P"
            + suffix + "_OUT);\n";
        source += "        output_offset = uint(P"
            + suffix + "_OUT_OFFSET);\n";
        source += "    }";
    }

    source += batch_rows ? R"METAL(
    uint output_base =
        local_tile * ROWS_PER_TG
        + simd_group * ROWS_PER_SIMD;

    float accumulators[ROWS][ROWS_PER_SIMD];
    for (uint input_row = 0u;
         input_row < uint(ROWS);
         ++input_row) {
        for (uint local_row = 0u;
             local_row < ROWS_PER_SIMD;
             ++local_row) {
            accumulators[input_row][local_row] = 0.0f;
        }
    }
    for (uint column = lane; column < uint(K); column += 32u) {
        float activations[ROWS];
        for (uint input_row = 0u;
             input_row < uint(ROWS);
             ++input_row) {
            activations[input_row] =
                float(x[input_row * uint(K) + column]);
        }
        for (
            uint local_row = 0u;
            local_row < ROWS_PER_SIMD;
            ++local_row
        ) {
            uint output = output_base + local_row;
            if (output >= output_width) {
                continue;
            }

            float weight = 0.0f;
)METAL" : R"METAL(
    uint output_base =
        local_tile * ROWS_PER_TG
        + simd_group * ROWS_PER_SIMD;

    float accumulators[ROWS_PER_SIMD] = {0.0f};
    for (uint column = lane; column < uint(K); column += 32u) {
        float activation =
            float(x[input_row * uint(K) + column]);
        for (
            uint local_row = 0u;
            local_row < ROWS_PER_SIMD;
            ++local_row
        ) {
            uint output = output_base + local_row;
            if (output >= output_width) {
                continue;
            }

            float weight = 0.0f;
)METAL";

    for (
        std::size_t projection = 0;
        projection < layouts.size();
        ++projection
    ) {
        const auto suffix = std::to_string(projection);
        source += projection == 0
            ? "            if ("
            : " else if (";
        source += "projection == " + suffix + "u) {\n";
        if (
            layouts[projection].family
                == kFamilyNint8Zero
        ) {
            source +=
                "                uint group = column >> 5;\n"
                "                weight = float(q8_scales_" + suffix
                + "[output * uint(P" + suffix
                + "_NG) + group])"
                " * float(q8_q_" + suffix
                + "[output * uint(K) + column]);\n";
        } else if (
            layouts[projection].family
                == kFamilyVq
        ) {
            source +=
                "                weight ="
                " mfq_grouped_vq_decode_weight(\n"
                "                    vq_indices_" + suffix + ",\n"
                "                    vq_state_" + suffix + ",\n"
                "                    vq_aux_" + suffix + ",\n"
                "                    vq_anchors_" + suffix + ",\n"
                "                    vq_codebooks_" + suffix + ",\n"
                "                    vq_scales_" + suffix + ",\n"
                "                    vq_state_banks_" + suffix + ",\n"
                "                    vq_bank_ids_" + suffix + ",\n"
                "                    vq_parameters_" + suffix + ",\n"
                "                    output,\n"
                "                    column,\n"
                "                    uint(P" + suffix + "_GS),\n"
                "                    uint(P" + suffix + "_NG),\n"
                "                    uint(P" + suffix + "_VECTOR_SIZE),\n"
                "                    uint(P" + suffix + "_NVEC),\n"
                "                    uint(P" + suffix + "_INDEX_BITS),\n"
                "                    uint(P" + suffix + "_STATE_BITS),\n"
                "                    uint(P" + suffix + "_STATES),\n"
                "                    uint(P" + suffix + "_ENTRIES),\n"
                "                    uint(P" + suffix + "_CODE_BANKS),\n"
                "                    uint(P" + suffix + "_AUX_MODE),\n"
                "                    uint(P" + suffix + "_CODE_BANK_MODE),\n"
                "                    uint(P" + suffix + "_EXECUTION_LAYOUT),\n"
                "                    uint(P" + suffix + "_HAS_TABLE_BANKS),\n"
                "                    uint(P" + suffix + "_GROUPS_PER_SUPER),\n"
                "                    uint(P" + suffix + "_NSUPER),\n"
                "                    (uint(K) + 7u) / 8u);\n";
        } else if (
            layouts[projection].family == kFamilyMx
        ) {
            source +=
                "                weight = mfq_grouped_mx_weight(\n"
                "                    mx_values_" + suffix + ",\n"
                "                    mx_scales_" + suffix + ",\n"
                "                    output, column,\n"
                "                    uint(P" + suffix + "_BITS),\n"
                "                    uint(K));\n";
        }
        source += "            }";
    }

    source += batch_rows ? R"METAL(
            for (uint input_row = 0u;
                 input_row < uint(ROWS);
                 ++input_row) {
                accumulators[input_row][local_row] = fma(
                    activations[input_row],
                    weight,
                    accumulators[input_row][local_row]);
            }
        }
    }

    for (uint input_row = 0u;
         input_row < uint(ROWS);
         ++input_row) {
        for (
            uint local_row = 0u;
            local_row < ROWS_PER_SIMD;
            ++local_row
        ) {
            float total = simd_sum(
                accumulators[input_row][local_row]);
            uint output = output_base + local_row;
            if (lane == 0u && output < output_width) {
                y[
                    input_row * uint(TOTAL_OUT)
                    + output_offset + output
                ] = T(total);
            }
        }
    }
)METAL" : R"METAL(
            accumulators[local_row] = fma(
                activation,
                weight,
                accumulators[local_row]);
        }
    }

    for (
        uint local_row = 0u;
        local_row < ROWS_PER_SIMD;
        ++local_row
    ) {
        float total = simd_sum(accumulators[local_row]);
        uint output = output_base + local_row;
        if (lane == 0u && output < output_width) {
            y[
                input_row * uint(TOTAL_OUT)
                + output_offset + output
            ] = T(total);
        }
    }
)METAL";
    return source;
}

void append_blockwise_vq_read(
    std::string& source,
    const std::string& target,
    const std::string& stream,
    const std::string& position,
    int bits) {
    source += "                uint " + target + " = ";
    if (bits == 4) {
        source += "mfq_grouped_vq_read_4(" + stream + ", "
            + position + ");\n";
    } else if (bits == 8) {
        source += "mfq_grouped_vq_read_8(" + stream + ", "
            + position + ");\n";
    } else if (bits == 12) {
        source += "mfq_grouped_vq_read_12(" + stream + ", "
            + position + ");\n";
    } else {
        source += "mfq_grouped_vq_read_bits(" + stream + ", "
            + position + ", " + std::to_string(bits) + "u);\n";
    }
}

std::string make_direct_small_m_blockwise_source(
    const std::vector<DirectProjectionLayout>& layouts,
    bool vectorized_fp16) {
    std::string source = R"METAL(
    constexpr uint SIMD_GROUPS = 2u;
    constexpr uint K_LANES = 8u;
    constexpr uint ROWS_PER_SIMD = 4u;
    constexpr uint ROWS_PER_TG = SIMD_GROUPS * ROWS_PER_SIMD;

    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint k_lane = lane & (K_LANES - 1u);
    uint simd_row = lane / K_LANES;
    uint global_tile = threadgroup_position_in_grid.x;
)METAL";

    for (std::size_t projection = 0;
         projection < layouts.size();
         ++projection) {
        const auto& layout = layouts[projection];
        const auto suffix = std::to_string(projection);
        source += projection == 0 ? "    if (" : "    else if (";
        source += "global_tile >= uint(P" + suffix
            + "_TILE_BEGIN) && global_tile < uint(P" + suffix
            + "_TILE_END)) {\n";
        source +=
            "        uint local_tile = global_tile - uint(P" + suffix
            + "_TILE_BEGIN);\n"
            "        uint output_index = local_tile * ROWS_PER_TG"
            " + simd_group * ROWS_PER_SIMD + simd_row;\n"
            "        uint output = min(output_index, uint(P" + suffix
            + "_OUT) - 1u);\n"
            "        float accumulators[ROWS];\n"
            "        for (uint row = 0u; row < uint(ROWS); ++row) {\n"
            "            accumulators[row] = 0.0f;\n"
            "        }\n";

        if (layout.family == kFamilyVq) {
            const int vectors_per_group =
                (layout.group_size + layout.vector_size - 1)
                / layout.vector_size;
            const bool use_wide_vq4 =
                vectorized_fp16 &&
                layout.group_size == 24 &&
                layout.vector_size == 4 &&
                layout.index_bits == 8 &&
                vectors_per_group == 6 &&
                (layout.aux_mode == 1 || layout.aux_mode == 2);
            const bool use_grouped_signs_vq8 =
                vectorized_fp16 &&
                layout.group_size == 24 &&
                layout.vector_size == 8 &&
                vectors_per_group == 3 &&
                (layout.aux_mode == 1 || layout.aux_mode == 2);
            const bool use_grouped_indices_vq8 =
                use_grouped_signs_vq8 && layout.index_bits == 12;
            const bool use_group64 = layout.execution_layout == 1;
            source +=
                "        uint output_group_base = output * uint(P" + suffix
                + "_NG);\n"
                "        uint output_vector_base = output * uint(P" + suffix
                + "_NVEC);\n"
                "        uint output_sign_base = output"
                " * ((uint(K) + 7u) / 8u);\n"
                "        uint output_super_base = output * uint(P" + suffix
                + "_NSUPER);\n"
                "        float output_anchor = vq_anchors_" + suffix
                + "[output];\n";
            if (use_wide_vq4) {
                source +=
                    "        #pragma clang loop unroll_count(2)\n";
            }
            source +=
                "        for (uint group = k_lane; group < uint(P" + suffix
                + "_NG); group += K_LANES) {\n"
                "            uint state_index = output_group_base + group;\n";
            if (use_group64) {
                source +=
                    "            uint2 group_record ="
                    " mfq_grouped_vq_read_group64(vq_indices_" + suffix
                    + ", state_index);\n"
                    "            uint state = group_record.y >> 28u;\n";
            } else {
                append_blockwise_vq_read(
                    source,
                    "state",
                    "vq_state_" + suffix,
                    "state_index",
                    layout.state_bits);
            }
            source += "            uint table_bank = 0u;\n";
            if (layout.table_banks > 1) {
                source +=
                    "            table_bank = uint(vq_bank_ids_" + suffix
                    + "[output_super_base + group / uint(P" + suffix
                    + "_GROUPS_PER_SUPER)]);\n";
            }
            source += "            uint auxiliary = 0u;\n";
            if (layout.aux_mode == 3) {
                source +=
                    "            auxiliary = mfq_grouped_vq_read_bits("
                    "vq_aux_" + suffix + ", state_index, 1u);\n";
            }
            source += "            uint code_bank = 0u;\n";
            if (layout.code_bank_mode == 1) {
                source +=
                    "            code_bank = uint(vq_state_banks_" + suffix
                    + "[state]);\n";
            } else if (layout.code_bank_mode == 2) {
                source += "            code_bank = auxiliary;\n";
            }
            source +=
                "            float weight_scale = output_anchor"
                " * vq_scales_" + suffix + "[table_bank * uint(P" + suffix
                + "_STATES) + state];\n";
            if (use_wide_vq4) {
                source +=
                    "            uint sign_bit ="
                    " (output_sign_base + group * 3u) * 7u;\n"
                    "            uint sign_byte = sign_bit >> 3u;\n"
                    "            uint packed_signs ="
                    " uint(vq_aux_" + suffix + "[sign_byte])"
                    " | (uint(vq_aux_" + suffix
                    + "[sign_byte + 1u]) << 8u)"
                    " | (uint(vq_aux_" + suffix
                    + "[sign_byte + 2u]) << 16u)"
                    " | (uint(vq_aux_" + suffix
                    + "[sign_byte + 3u]) << 24u);\n"
                    "            uint group_signs ="
                    " packed_signs >> (sign_bit & 7u);\n"
                    "            #pragma clang loop unroll(full)\n"
                    "            for (uint local_pair = 0u;"
                    " local_pair < 3u; ++local_pair) {\n"
                    "                uint column_base ="
                    " group * uint(P" + suffix
                    + "_GS) + local_pair * 8u;\n"
                    "                if (column_base >= uint(K))"
                    " { break; }\n"
                    "                uint vector = column_base >> 2u;\n"
                    "                uint index_position ="
                    " output_vector_base + vector;\n"
                    "                uchar2 pair_indices ="
                    " *(device const uchar2*)(vq_indices_" + suffix
                    + " + index_position);\n"
                    "                uint index0 = uint(pair_indices.x);\n"
                    "                uint index1 = uint(pair_indices.y);\n"
                    "                uint sign_value ="
                    " (group_signs >> (local_pair * 7u)) & 127u;\n"
                    "                uint code_base0 ="
                    " (((table_bank * uint(P" + suffix
                    + "_CODE_BANKS) + code_bank) * uint(P" + suffix
                    + "_ENTRIES) + index0) * 4u);\n"
                    "                uint code_base1 ="
                    " (((table_bank * uint(P" + suffix
                    + "_CODE_BANKS) + code_bank) * uint(P" + suffix
                    + "_ENTRIES) + index1) * 4u);\n"
                    "                char4 packed_codes0 ="
                    " *(device const char4*)(vq_codebooks_" + suffix
                    + " + code_base0);\n"
                    "                char4 packed_codes1 ="
                    " *(device const char4*)(vq_codebooks_" + suffix
                    + " + code_base1);\n"
                    "                float4 codes0 = float4("
                    "float(packed_codes0.x), float(packed_codes0.y),"
                    " float(packed_codes0.z), float(packed_codes0.w));\n"
                    "                float4 codes1 = float4("
                    "float(packed_codes1.x), float(packed_codes1.y),"
                    " float(packed_codes1.z), float(packed_codes1.w));\n"
                    "                for (uint component = 0u;"
                    " component < 4u; ++component) {\n"
                    "                    uint negative ="
                    " (sign_value >> component) & 1u;\n"
                    "                    codes0[component] ="
                    " negative != 0u ? -codes0[component]"
                    " : codes0[component];\n"
                    "                }\n"
                    "                for (uint component = 0u;"
                    " component < 4u; ++component) {\n"
                    "                    uint sign_position ="
                    " component + 4u;\n"
                    "                    uint negative ="
                    " sign_position < 7u"
                    " ? ((sign_value >> sign_position) & 1u)"
                    " : (popcount(sign_value) & 1u);\n";
                if (layout.aux_mode == 2) {
                    source +=
                        "                    if (sign_position == 7u)"
                        " { negative ^= (index1 >> 7u) & 1u; }\n";
                }
                source +=
                    "                    codes1[component] ="
                    " negative != 0u ? -codes1[component]"
                    " : codes1[component];\n"
                    "                }\n"
                    "                float4 weights0 ="
                    " weight_scale * codes0;\n"
                    "                float4 weights1 ="
                    " weight_scale * codes1;\n"
                    "                #pragma clang loop unroll(full)\n"
                    "                for (uint row = 0u;"
                    " row < uint(ROWS); ++row) {\n"
                    "                    uint input_base ="
                    " row * uint(K) + column_base;\n"
                    "                    float4 activation0 = float4("
                    "*(device const half4*)(x + input_base));\n"
                    "                    float4 activation1 = float4("
                    "*(device const half4*)(x + input_base + 4u));\n"
                    "                    if (uint(ROWS) <= 3u) {\n"
                    "                        accumulators[row] +="
                    " dot(activation0, weights0);\n"
                    "                        accumulators[row] +="
                    " dot(activation1, weights1);\n"
                    "                    } else {\n"
                    "                        accumulators[row] +="
                    " activation0.x * weights0.x;\n"
                    "                        accumulators[row] +="
                    " activation0.y * weights0.y;\n"
                    "                        accumulators[row] +="
                    " activation0.z * weights0.z;\n"
                    "                        accumulators[row] +="
                    " activation0.w * weights0.w;\n"
                    "                        accumulators[row] +="
                    " activation1.x * weights1.x;\n"
                    "                        accumulators[row] +="
                    " activation1.y * weights1.y;\n"
                    "                        accumulators[row] +="
                    " activation1.z * weights1.z;\n"
                    "                        accumulators[row] +="
                    " activation1.w * weights1.w;\n"
                    "                    }\n"
                    "                }\n";
            } else {
                if (use_grouped_signs_vq8 && !use_group64) {
                    source +=
                        "            uint group_signs = 0u;\n"
                        "            if (uint(ROWS) == 2u) {\n"
                        "                uint sign_bit ="
                        " (output_sign_base + group * 3u) * 7u;\n"
                        "                uint sign_byte = sign_bit >> 3u;\n"
                        "                uint packed_signs ="
                        " uint(vq_aux_" + suffix + "[sign_byte])"
                        " | (uint(vq_aux_" + suffix
                        + "[sign_byte + 1u]) << 8u)"
                        " | (uint(vq_aux_" + suffix
                        + "[sign_byte + 2u]) << 16u)"
                        " | (uint(vq_aux_" + suffix
                        + "[sign_byte + 3u]) << 24u);\n"
                        "                group_signs ="
                        " packed_signs >> (sign_bit & 7u);\n"
                        "            }\n";
                }
                if (use_grouped_indices_vq8 && !use_group64) {
                    source +=
                        "            uint grouped_indices = 0u;\n"
                        "            uint grouped_index_high = 0u;\n"
                        "            uint grouped_index_shift = 0u;\n"
                        "            if (uint(ROWS) == 2u) {\n"
                        "                uint index_bit ="
                        " (output_vector_base + group * 3u) * 12u;\n"
                        "                uint index_byte = index_bit >> 3u;\n"
                        "                grouped_index_shift ="
                        " index_bit & 7u;\n"
                        "                grouped_indices ="
                        " uint(vq_indices_" + suffix + "[index_byte])"
                        " | (uint(vq_indices_" + suffix
                        + "[index_byte + 1u]) << 8u)"
                        " | (uint(vq_indices_" + suffix
                        + "[index_byte + 2u]) << 16u)"
                        " | (uint(vq_indices_" + suffix
                        + "[index_byte + 3u]) << 24u);\n"
                        "                grouped_index_high ="
                        " uint(vq_indices_" + suffix
                        + "[index_byte + 4u]);\n"
                        "            }\n";
                }
                source +=
                "            for (uint local_vector = 0u; local_vector < "
                + std::to_string(vectors_per_group)
                + "u; ++local_vector) {\n"
                "                uint column_base = group * uint(P" + suffix
                + "_GS) + local_vector * uint(P" + suffix
                + "_VECTOR_SIZE);\n"
                "                if (column_base >= uint(K)) { break; }\n"
                "                uint vector = column_base / uint(P" + suffix
                + "_VECTOR_SIZE);\n"
                "                uint index_position = output_vector_base"
                " + vector;\n";
            if (use_group64) {
                source +=
                    "                uint group_segment ="
                    " mfq_grouped_vq_group64_segment(group_record,"
                    " local_vector);\n"
                    "                uint index = group_segment & 4095u;\n";
            } else if (use_grouped_indices_vq8) {
                source +=
                    "                uint index;\n"
                    "                if (uint(ROWS) == 2u) {\n"
                    "                    uint local_index_bit ="
                    " grouped_index_shift + local_vector * 12u;\n"
                    "                    if (local_index_bit < 24u) {\n"
                    "                        index = (grouped_indices >>"
                    " local_index_bit) & 4095u;\n"
                    "                    } else {\n"
                    "                        index = ((grouped_indices >>"
                    " local_index_bit) | (grouped_index_high <<"
                    " (32u - local_index_bit))) & 4095u;\n"
                    "                    }\n"
                    "                } else {\n"
                    "                    index = mfq_grouped_vq_read_12("
                    "vq_indices_" + suffix + ", index_position);\n"
                    "                }\n";
            } else {
                append_blockwise_vq_read(
                    source,
                    "index",
                    "vq_indices_" + suffix,
                    "index_position",
                    layout.index_bits);
            }
            if (use_group64) {
                source +=
                    "                uint sign_value ="
                    " group_segment >> 12u;\n";
            } else {
                source += "                uint sign_value = 0u;\n";
            }
            if (layout.aux_mode == 1 || layout.aux_mode == 2) {
                if (use_group64) {
                    // The 8-bit group64 sign mask already includes the even
                    // parity completion bit used by component seven.
                } else if (use_grouped_signs_vq8) {
                    source +=
                        "                if (uint(ROWS) == 2u) {\n"
                        "                    sign_value ="
                        " (group_signs >> (local_vector * 7u)) & 127u;\n"
                        "                } else {\n"
                        "                    sign_value ="
                        " mfq_grouped_vq_read_bits(vq_aux_" + suffix
                        + ", output_sign_base + column_base / 8u, 7u);\n"
                        "                }\n";
                } else {
                    source +=
                        "                sign_value ="
                        " mfq_grouped_vq_read_bits(vq_aux_" + suffix
                        + ", output_sign_base + column_base / 8u, 7u);\n";
                }
            }
            source +=
                "                uint code_base = (((table_bank * uint(P"
                + suffix + "_CODE_BANKS) + code_bank) * uint(P" + suffix
                + "_ENTRIES) + index) * uint(P" + suffix
                + "_VECTOR_SIZE));\n";
            if (vectorized_fp16 && layout.vector_size == 8 &&
                (layout.aux_mode == 1 || layout.aux_mode == 2)) {
                source +=
                    "                uint2 packed_words ="
                    " *(device const uint2*)(vq_codebooks_" + suffix
                    + " + code_base);\n"
                    "                char4 packed_codes0 ="
                    " as_type<char4>(packed_words.x);\n"
                    "                char4 packed_codes1 ="
                    " as_type<char4>(packed_words.y);\n"
                    "                float4 codes0 = float4("
                    "float(packed_codes0.x), float(packed_codes0.y),"
                    " float(packed_codes0.z), float(packed_codes0.w));\n"
                    "                float4 codes1 = float4("
                    "float(packed_codes1.x), float(packed_codes1.y),"
                    " float(packed_codes1.z), float(packed_codes1.w));\n"
                    "                for (uint component = 0u;"
                    " component < 4u; ++component) {\n"
                    "                    uint negative ="
                    " (sign_value >> component) & 1u;\n"
                    "                    codes0[component] ="
                    " negative != 0u ? -codes0[component]"
                    " : codes0[component];\n"
                    "                }\n"
                    "                for (uint component = 0u;"
                    " component < 4u; ++component) {\n"
                    "                    uint sign_position ="
                    " component + 4u;\n"
                    "                    uint negative =";
                if (use_group64) {
                    source +=
                        " (sign_value >> sign_position) & 1u;\n";
                } else {
                    source +=
                        " sign_position < 7u"
                        " ? ((sign_value >> sign_position) & 1u)"
                        " : (popcount(sign_value) & 1u);\n";
                }
                if (layout.aux_mode == 2) {
                    source +=
                        "                    if (sign_position == 7u)"
                        " { negative ^= (index >> 7u) & 1u; }\n";
                }
                source +=
                    "                    codes1[component] ="
                    " negative != 0u ? -codes1[component]"
                    " : codes1[component];\n"
                    "                }\n"
                    "                float4 weights0 ="
                    " weight_scale * codes0;\n"
                    "                float4 weights1 ="
                    " weight_scale * codes1;\n"
                    "                for (uint row = 0u;"
                    " row < uint(ROWS); ++row) {\n"
                    "                    uint input_base ="
                    " row * uint(K) + column_base;\n"
                    "                    float4 activation0 = float4("
                    "*(device const half4*)(x + input_base));\n"
                    "                    float4 activation1 = float4("
                    "*(device const half4*)(x + input_base + 4u));\n"
                    "                    accumulators[row] +="
                    " dot(activation0, weights0);\n"
                    "                    accumulators[row] +="
                    " dot(activation1, weights1);\n"
                    "                }\n";
            } else if ((layout.vector_size % 4) == 0) {
                source +=
                    "                for (uint component_base = 0u;"
                    " component_base < uint(P" + suffix
                    + "_VECTOR_SIZE); component_base += 4u) {\n"
                    "                    uint column = column_base"
                    " + component_base;\n"
                    "                    char4 packed_codes ="
                    " *(device const char4*)(vq_codebooks_" + suffix
                    + " + code_base + component_base);\n"
                    "                    float4 codes = float4("
                    "float(packed_codes.x), float(packed_codes.y),"
                    " float(packed_codes.z), float(packed_codes.w));\n";
                if (layout.aux_mode == 1 || layout.aux_mode == 2) {
                    source +=
                        "                    for (uint packed_component = 0u;"
                        " packed_component < 4u; ++packed_component) {\n"
                        "                        uint sign_position ="
                        " (column + packed_component) & 7u;\n"
                        "                        uint negative =";
                    if (use_group64) {
                        source +=
                            " (sign_value >> sign_position) & 1u;\n";
                    } else {
                        source +=
                            " sign_position < 7u"
                            " ? ((sign_value >> sign_position) & 1u)"
                            " : (popcount(sign_value) & 1u);\n";
                    }
                    if (layout.aux_mode == 2) {
                        source +=
                            "                        if (sign_position == 7u)"
                            " { negative ^= (index >> 7u) & 1u; }\n";
                    }
                    source +=
                        "                        codes[packed_component] ="
                        " negative != 0u ? -codes[packed_component]"
                        " : codes[packed_component];\n"
                        "                    }\n";
                } else if (layout.aux_mode == 3) {
                    source +=
                        "                    codes += float4("
                        "auxiliary != 0u ? -vq_parameters_" + suffix
                        + "[0] : vq_parameters_" + suffix + "[0]);\n";
                }
                source +=
                    "                    float4 weights ="
                    " weight_scale * codes;\n"
                    "                    for (uint row = 0u;"
                    " row < uint(ROWS); ++row) {\n"
                    "                        uint input_base ="
                    " row * uint(K) + column;\n";
                if (vectorized_fp16) {
                    source +=
                        "                        float4 activation;\n"
                        "                        if (column + 3u < uint(K)) {\n"
                        "                            activation = float4("
                        "*(device const half4*)(x + input_base));\n"
                        "                        } else {\n"
                        "                            activation = float4(\n"
                        "                                column < uint(K)"
                        " ? float(x[input_base]) : 0.0f,\n"
                        "                                column + 1u < uint(K)"
                        " ? float(x[input_base + 1u]) : 0.0f,\n"
                        "                                column + 2u < uint(K)"
                        " ? float(x[input_base + 2u]) : 0.0f,\n"
                        "                                column + 3u < uint(K)"
                        " ? float(x[input_base + 3u]) : 0.0f);\n"
                        "                        }\n";
                } else {
                    source +=
                        "                        float4 activation = float4(\n"
                        "                            column < uint(K)"
                        " ? float(x[input_base]) : 0.0f,\n"
                        "                            column + 1u < uint(K)"
                        " ? float(x[input_base + 1u]) : 0.0f,\n"
                        "                            column + 2u < uint(K)"
                        " ? float(x[input_base + 2u]) : 0.0f,\n"
                        "                            column + 3u < uint(K)"
                        " ? float(x[input_base + 3u]) : 0.0f);\n";
                }
                if (vectorized_fp16 && layout.vector_size == 8) {
                    source +=
                        "                        accumulators[row] +="
                        " dot(activation, weights);\n";
                } else {
                    source +=
                        "                        accumulators[row] +="
                        " activation.x * weights.x;\n"
                        "                        if (column + 1u < uint(K))"
                        " accumulators[row] += activation.y * weights.y;\n"
                        "                        if (column + 2u < uint(K))"
                        " accumulators[row] += activation.z * weights.z;\n"
                        "                        if (column + 3u < uint(K))"
                        " accumulators[row] += activation.w * weights.w;\n";
                }
                source +=
                    "                    }\n"
                    "                }\n";
            } else {
                source +=
                    "                for (uint component = 0u;"
                    " component < uint(P" + suffix
                    + "_VECTOR_SIZE); ++component) {\n"
                    "                    uint column = column_base"
                    " + component;\n"
                    "                    if (column >= uint(K))"
                    " { break; }\n"
                    "                    float code = float(vq_codebooks_"
                    + suffix + "[code_base + component]);\n";
                if (layout.aux_mode == 1 || layout.aux_mode == 2) {
                    source +=
                        "                    uint sign_position ="
                        " column & 7u;\n"
                        "                    uint negative ="
                        " sign_position < 7u"
                        " ? ((sign_value >> sign_position) & 1u)"
                        " : (popcount(sign_value) & 1u);\n";
                    if (layout.aux_mode == 2) {
                        source +=
                            "                    if (sign_position == 7u)"
                            " { negative ^= (index >> 7u) & 1u; }\n";
                    }
                    source +=
                        "                    code = negative != 0u"
                        " ? -code : code;\n";
                } else if (layout.aux_mode == 3) {
                    source +=
                        "                    code += auxiliary != 0u"
                        " ? -vq_parameters_" + suffix + "[0]"
                        " : vq_parameters_" + suffix + "[0];\n";
                }
                source +=
                    "                    float weight ="
                    " weight_scale * code;\n"
                    "                    for (uint row = 0u;"
                    " row < uint(ROWS); ++row) {\n"
                    "                        accumulators[row] += float("
                    "x[row * uint(K) + column]) * weight;\n"
                    "                    }\n"
                    "                }\n";
            }
            }
            source +=
                "            }\n"
                "        }\n";
        } else if (layout.family == kFamilyMx) {
            const int block = layout.bits == 4 ? 32 : 128;
            const int vectors = block / 4;
            source +=
                "        constexpr uint MX_BLOCK = "
                + std::to_string(block) + "u;\n"
                "        constexpr uint MX_BLOCKS = uint(K) / MX_BLOCK;\n"
                "        for (uint block = k_lane; block < MX_BLOCKS;"
                " block += K_LANES) {\n"
                "            uint column_base = block * MX_BLOCK;\n"
                "            float mx_accumulators[ROWS];\n"
                "            for (uint row = 0u; row < uint(ROWS); ++row) {\n"
                "                mx_accumulators[row] = 0.0f;\n"
                "            }\n";
            if (layout.bits == 4) {
                source +=
                    "            uint value_base = output * (uint(K) / 2u);\n";
            } else {
                source +=
                    "            uint value_base = output * uint(K);\n";
            }
            source +=
                "            for (uint vector = 0u; vector < "
                + std::to_string(vectors) + "u; ++vector) {\n"
                "                uint column = column_base + vector * 4u;\n";
            if (layout.bits == 4) {
                source +=
                    "                uint packed = uint(*(device const ushort*)("
                    "mx_values_" + suffix
                    + " + value_base + (column >> 1u)));\n"
                    "                float4 weights = float4(\n"
                    "                    mfq_grouped_mx_fp4(uchar(packed & 15u)),\n"
                    "                    mfq_grouped_mx_fp4(uchar((packed >> 4u) & 15u)),\n"
                    "                    mfq_grouped_mx_fp4(uchar((packed >> 8u) & 15u)),\n"
                    "                    mfq_grouped_mx_fp4(uchar((packed >> 12u) & 15u)));\n";
            } else {
                source +=
                    "                uchar4 codes = as_type<uchar4>("
                    "*(device const uint*)(mx_values_" + suffix
                    + " + value_base + column));\n"
                    "                float4 weights = float4(\n"
                    "                    mfq_grouped_mx_fp8(codes.x),\n"
                    "                    mfq_grouped_mx_fp8(codes.y),\n"
                    "                    mfq_grouped_mx_fp8(codes.z),\n"
                    "                    mfq_grouped_mx_fp8(codes.w));\n";
            }
            source +=
                "                for (uint row = 0u; row < uint(ROWS); ++row) {\n"
                "                    half4 activation = *(device const half4*)(\n"
                "                        x + row * uint(K) + column);\n"
                "                    mx_accumulators[row] = fma(\n"
                "                        float(activation.x), weights.x,\n"
                "                        mx_accumulators[row]);\n"
                "                    mx_accumulators[row] = fma(\n"
                "                        float(activation.y), weights.y,\n"
                "                        mx_accumulators[row]);\n"
                "                    mx_accumulators[row] = fma(\n"
                "                        float(activation.z), weights.z,\n"
                "                        mx_accumulators[row]);\n"
                "                    mx_accumulators[row] = fma(\n"
                "                        float(activation.w), weights.w,\n"
                "                        mx_accumulators[row]);\n"
                "                }\n"
                "            }\n"
                "            float scale = mfq_grouped_mx_e8m0(";
            if (layout.bits == 4) {
                source +=
                    "mx_scales_" + suffix
                    + "[output * MX_BLOCKS + block]);\n";
            } else {
                source +=
                    "mx_scales_" + suffix
                    + "[(output / 128u) * MX_BLOCKS + block]);\n";
            }
            source +=
                "            for (uint row = 0u; row < uint(ROWS); ++row) {\n"
                "                accumulators[row] = fma(\n"
                "                    scale, mx_accumulators[row],\n"
                "                    accumulators[row]);\n"
                "            }\n"
                "        }\n";
        } else {
            source +=
                "        uint output_group_base = output * uint(P" + suffix
                + "_NG);\n"
                "        for (uint group = k_lane; group < uint(P" + suffix
                + "_NG); group += K_LANES) {\n"
                "            float scale = float(q8_scales_" + suffix
                + "[output_group_base + group]);\n"
                "            for (uint element = 0u; element < 32u;"
                " ++element) {\n"
                "                uint column = group * 32u + element;\n"
                "                if (column >= uint(K)) { break; }\n"
                "                float weight = scale * float(q8_q_" + suffix
                + "[output * uint(K) + column]);\n"
                "                for (uint row = 0u; row < uint(ROWS);"
                " ++row) {\n"
                "                    accumulators[row] += float("
                "x[row * uint(K) + column]) * weight;\n"
                "                }\n"
                "            }\n"
                "        }\n";
        }

        source += R"METAL(
        for (uint row = 0u; row < uint(ROWS); ++row) {
            accumulators[row] += simd_shuffle_down(accumulators[row], 4);
            accumulators[row] += simd_shuffle_down(accumulators[row], 2);
            accumulators[row] += simd_shuffle_down(accumulators[row], 1);
)METAL";
        source +=
            "            if (k_lane == 0u && output_index < uint(P" + suffix
            + "_OUT)) {\n"
            "                y[row * uint(TOTAL_OUT) + uint(P" + suffix
            + "_OUT_OFFSET) + output_index] = T(accumulators[row]);\n"
            "            }\n"
            "        }\n"
            "    }\n";
    }
    return source;
}

std::string make_direct_small_m_group64_output_tile_source(
    const std::vector<DirectProjectionLayout>& layouts,
    int outputs_per_simd,
    int simd_groups) {
    std::string source =
        "    constexpr uint SIMD_GROUPS = "
        + std::to_string(simd_groups) + "u;\n";
    source += "    constexpr uint OUTPUTS_PER_SIMD = "
        + std::to_string(outputs_per_simd) + "u;\n";
    source += R"METAL(
    constexpr uint OUTPUTS_PER_TG = SIMD_GROUPS * OUTPUTS_PER_SIMD;

    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint global_tile = threadgroup_position_in_grid.x;
)METAL";

    for (std::size_t projection = 0;
         projection < layouts.size();
         ++projection) {
        const auto& layout = layouts[projection];
        const auto suffix = std::to_string(projection);
        source += projection == 0 ? "    if (" : "    else if (";
        source += "global_tile >= uint(P" + suffix
            + "_TILE_BEGIN) && global_tile < uint(P" + suffix
            + "_TILE_END)) {\n";
        source +=
            "        uint local_tile = global_tile - uint(P" + suffix
            + "_TILE_BEGIN);\n"
            "        uint output_base = local_tile * OUTPUTS_PER_TG"
            " + simd_group * OUTPUTS_PER_SIMD;\n"
            "        uint output_group_bases[OUTPUTS_PER_SIMD];\n"
            "        uint output_super_bases[OUTPUTS_PER_SIMD];\n"
            "        float output_anchors[OUTPUTS_PER_SIMD];\n"
            "        float accumulators[OUTPUTS_PER_SIMD][ROWS];\n"
            "        for (uint output_row = 0u;"
            " output_row < OUTPUTS_PER_SIMD; ++output_row) {\n"
            "            uint output = min(output_base + output_row,"
            " uint(P" + suffix + "_OUT) - 1u);\n"
            "            output_group_bases[output_row] ="
            " output * uint(P" + suffix + "_NG);\n"
            "            output_super_bases[output_row] ="
            " output * uint(P" + suffix + "_NSUPER);\n"
            "            output_anchors[output_row] ="
            " vq_anchors_" + suffix + "[output];\n"
            "            for (uint row = 0u; row < uint(ROWS); ++row) {\n"
            "                accumulators[output_row][row] = 0.0f;\n"
            "            }\n"
            "        }\n"
            "        for (uint group = lane; group < uint(P" + suffix
            + "_NG); group += 32u) {\n"
            "            uint2 records[OUTPUTS_PER_SIMD];\n"
            "            float weight_scales[OUTPUTS_PER_SIMD];\n"
            "            uint table_banks[OUTPUTS_PER_SIMD];\n"
            "            uint code_banks[OUTPUTS_PER_SIMD];\n"
            "            for (uint output_row = 0u;"
            " output_row < OUTPUTS_PER_SIMD; ++output_row) {\n"
            "                uint2 record = mfq_grouped_vq_read_group64("
            "vq_indices_" + suffix
            + ", output_group_bases[output_row] + group);\n"
            "                records[output_row] = record;\n"
            "                uint state = record.y >> 28u;\n"
            "                uint table_bank = 0u;\n";
        if (layout.table_banks > 1) {
            source +=
                "                table_bank = uint(vq_bank_ids_" + suffix
                + "[output_super_bases[output_row]"
                " + group / uint(P" + suffix
                + "_GROUPS_PER_SUPER)]);\n";
        }
        source +=
            "                table_banks[output_row] = table_bank;\n"
            "                uint code_bank = 0u;\n";
        if (layout.code_bank_mode == 1) {
            source +=
                "                code_bank = uint(vq_state_banks_" + suffix
                + "[state]);\n";
        } else if (layout.code_bank_mode == 2) {
            source +=
                "                code_bank ="
                " mfq_grouped_vq_group64_segment(record, 0u) >> 12u;\n";
        }
        source +=
            "                code_banks[output_row] = code_bank;\n"
            "                weight_scales[output_row] ="
            " output_anchors[output_row] * vq_scales_" + suffix
            + "[table_bank * uint(P" + suffix
            + "_STATES) + state];\n"
            "            }\n"
            "            #pragma clang loop unroll(full)\n"
            "            for (uint local_vector = 0u;"
            " local_vector < 3u; ++local_vector) {\n"
            "                uint column_base = group * 24u"
            " + local_vector * 8u;\n"
            "                if (column_base >= uint(K)) { break; }\n"
            "                for (uint output_row = 0u;"
            " output_row < OUTPUTS_PER_SIMD; ++output_row) {\n"
            "                    uint segment ="
            " mfq_grouped_vq_group64_segment("
            "records[output_row], local_vector);\n"
            "                    uint index = segment & 4095u;\n"
            "                    uint sign_value = segment >> 12u;\n"
            "                    uint code_base = ((("
            "table_banks[output_row] * uint(P" + suffix
            + "_CODE_BANKS) + code_banks[output_row])"
            " * uint(P" + suffix + "_ENTRIES) + index) * 8u);\n"
            "                    uint2 packed_words ="
            " *(device const uint2*)(vq_codebooks_" + suffix
            + " + code_base);\n"
            "                    char4 packed_codes0 ="
            " as_type<char4>(packed_words.x);\n"
            "                    char4 packed_codes1 ="
            " as_type<char4>(packed_words.y);\n"
            "                    float4 codes0 = float4("
            "float(packed_codes0.x), float(packed_codes0.y),"
            " float(packed_codes0.z), float(packed_codes0.w));\n"
            "                    float4 codes1 = float4("
            "float(packed_codes1.x), float(packed_codes1.y),"
            " float(packed_codes1.z), float(packed_codes1.w));\n"
            "                    for (uint component = 0u;"
            " component < 4u; ++component) {\n"
            "                        if (((sign_value >> component)"
            " & 1u) != 0u) { codes0[component] = -codes0[component]; }\n"
            "                    }\n"
            "                    for (uint component = 0u;"
            " component < 4u; ++component) {\n"
            "                        uint sign_position = component + 4u;\n"
            "                        uint negative ="
            " (sign_value >> sign_position) & 1u;\n";
        if (layout.aux_mode == 2) {
            source +=
                "                        if (sign_position == 7u)"
                " { negative ^= (index >> 7u) & 1u; }\n";
        }
        source +=
            "                        if (negative != 0u)"
            " { codes1[component] = -codes1[component]; }\n"
            "                    }\n"
            "                    float scale = weight_scales[output_row];\n"
            "                    float4 weights0 = scale * codes0;\n"
            "                    float4 weights1 = scale * codes1;\n"
            "                    for (uint row = 0u;"
            " row < uint(ROWS); ++row) {\n"
            "                        uint input_base ="
            " row * uint(K) + column_base;\n"
            "                        float4 activation0 = float4("
            "*(device const half4*)(x + input_base));\n"
            "                        float4 activation1 = float4("
            "*(device const half4*)(x + input_base + 4u));\n"
            "                        accumulators[output_row][row] +="
            " dot(activation0, weights0);\n"
            "                        accumulators[output_row][row] +="
            " dot(activation1, weights1);\n"
            "                    }\n"
            "                }\n"
            "            }\n"
            "        }\n"
            "        for (uint output_row = 0u;"
            " output_row < OUTPUTS_PER_SIMD; ++output_row) {\n"
            "            uint output_index = output_base + output_row;\n"
            "            for (uint row = 0u; row < uint(ROWS); ++row) {\n"
            "                float total ="
            " simd_sum(accumulators[output_row][row]);\n"
            "                if (lane == 0u &&"
            " output_index < uint(P" + suffix + "_OUT)) {\n"
            "                    y[row * uint(TOTAL_OUT) + uint(P" + suffix
            + "_OUT_OFFSET) + output_index] = T(total);\n"
            "                }\n"
            "            }\n"
            "        }\n"
            "    }\n";
    }
    return source;
}

mlx::core::fast::CustomKernelFunction make_direct_kernel(
    const std::vector<DirectProjectionLayout>& layouts,
    bool batch_rows,
    bool blockwise,
    bool vectorized_fp16,
    int group64_outputs_per_simd,
    int group64_simd_groups) {
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    const auto key = direct_kernel_key(layouts)
        + (group64_outputs_per_simd > 0
            ? "_m2_6_group64_o" + std::to_string(group64_outputs_per_simd)
                + "s" + std::to_string(group64_simd_groups)
            : (batch_rows
            ? (blockwise
                ? (vectorized_fp16
                    ? "_m2_6_block_vec"
                    : "_m2_6_block")
                : "_m2_6")
            : "_rows"));
    return mlx::core::fast::metal_kernel(
        "mfq_cpp_zero_copy_grouped_linear_" + key,
        direct_input_names(layouts),
        {"y"},
        group64_outputs_per_simd > 0
            ? make_direct_small_m_group64_output_tile_source(
                layouts,
                group64_outputs_per_simd,
                group64_simd_groups)
            : (blockwise
            ? make_direct_small_m_blockwise_source(
                layouts,
                vectorized_fp16)
            : make_direct_source(layouts, batch_rows)),
        kGroupedHeader,
        true,
        false,
        options);
}

mlx::core::fast::CustomKernelFunction direct_kernel(
    const std::vector<DirectProjectionLayout>& layouts,
    bool batch_rows,
    bool blockwise = false,
    bool vectorized_fp16 = false,
    int group64_outputs_per_simd = 0,
    int group64_simd_groups = 2) {
    static std::mutex mutex;
    static std::unordered_map<
        std::string,
        mlx::core::fast::CustomKernelFunction> kernels;

    const auto key = direct_kernel_key(layouts)
        + (group64_outputs_per_simd > 0
            ? "_m2_6_group64_o" + std::to_string(group64_outputs_per_simd)
                + "s" + std::to_string(group64_simd_groups)
            : (batch_rows
            ? (blockwise
                ? (vectorized_fp16
                    ? "_m2_6_block_vec"
                    : "_m2_6_block")
                : "_m2_6")
            : "_rows"));
    std::lock_guard<std::mutex> lock(mutex);
    const auto found = kernels.find(key);
    if (found != kernels.end()) {
        return found->second;
    }
    auto kernel = make_direct_kernel(
        layouts,
        batch_rows,
        blockwise,
        vectorized_fp16,
        group64_outputs_per_simd,
        group64_simd_groups);
    kernels.emplace(key, kernel);
    return kernel;
}

std::string make_single_row_mxfp8_source(
    const std::vector<DirectProjectionLayout>& layouts) {
    std::string source = R"METAL(
    constexpr uint SIMD_GROUPS = 4u;
    constexpr uint K_LANES = 8u;
    constexpr uint ROWS_PER_SIMD = 32u / K_LANES;
    constexpr uint ROWS_PER_TG = SIMD_GROUPS * ROWS_PER_SIMD;
    constexpr uint MX_BLOCK = 128u;
    constexpr uint MX_BLOCKS = uint(K) / MX_BLOCK;
    constexpr uint Q8_GROUP = 32u;
    constexpr uint Q8_GROUPS_PER_MX = MX_BLOCK / Q8_GROUP;

    threadgroup half fp8_lut[256];
    uint local_thread = thread_index_in_threadgroup;
    fp8_lut[local_thread] = as_type<half>(
        mfq_grouped_mx_fp8_half_lut[local_thread]);
    fp8_lut[local_thread + 128u] = as_type<half>(
        mfq_grouped_mx_fp8_half_lut[local_thread + 128u]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint k_lane = lane & (K_LANES - 1u);
    uint simd_row = lane / K_LANES;
    uint output_index =
        threadgroup_position_in_grid.x * ROWS_PER_TG
        + simd_group * ROWS_PER_SIMD + simd_row;
)METAL";

    for (
        std::size_t projection = 0;
        projection < layouts.size();
        ++projection
    ) {
        const auto suffix = std::to_string(projection);
        source +=
            "    bool active_" + suffix
            + " = output_index < uint(P" + suffix + "_OUT);\n"
            "    uint output_" + suffix + " = min(\n"
            "        output_index, uint(P" + suffix + "_OUT) - 1u);\n"
            "    float accumulator_" + suffix + " = 0.0f;\n";
    }

    source += R"METAL(
    for (
        uint block = k_lane;
        block < MX_BLOCKS;
        block += K_LANES
    ) {
        uint column_base = block * MX_BLOCK;
)METAL";
    for (
        std::size_t projection = 0;
        projection < layouts.size();
        ++projection
    ) {
        if (layouts[projection].family == kFamilyMx) {
            source +=
                "        float mx_dot_" + std::to_string(projection)
                + " = 0.0f;\n";
        }
    }

    source += R"METAL(
        for (
            uint q8_local = 0u;
            q8_local < Q8_GROUPS_PER_MX;
            ++q8_local
        ) {
)METAL";
    for (
        std::size_t projection = 0;
        projection < layouts.size();
        ++projection
    ) {
        if (layouts[projection].family == kFamilyNint8Zero) {
            source +=
                "            float q8_dot_" + std::to_string(projection)
                + " = 0.0f;\n";
        }
    }

    source += R"METAL(
            uint q8_column_base =
                column_base + q8_local * Q8_GROUP;
            for (uint element = 0u; element < Q8_GROUP; element += 4u) {
                uint column = q8_column_base + element;
                half4 activation = *(device const half4*)(x + column);
)METAL";
    for (
        std::size_t projection = 0;
        projection < layouts.size();
        ++projection
    ) {
        const auto suffix = std::to_string(projection);
        if (layouts[projection].family == kFamilyMx) {
            source +=
                "                if (active_" + suffix + ") {\n"
                "                    uint value_offset = output_" + suffix
                + " * uint(K) + column;\n"
                "                    uchar4 code = *(device const uchar4*)(\n"
                "                        mx_values_" + suffix
                + " + value_offset);\n"
                "                    mx_dot_" + suffix + " = fma(\n"
                "                        float(activation.x),\n"
                "                        float(fp8_lut[uint(code.x)]),\n"
                "                        mx_dot_" + suffix + ");\n"
                "                    mx_dot_" + suffix + " = fma(\n"
                "                        float(activation.y),\n"
                "                        float(fp8_lut[uint(code.y)]),\n"
                "                        mx_dot_" + suffix + ");\n"
                "                    mx_dot_" + suffix + " = fma(\n"
                "                        float(activation.z),\n"
                "                        float(fp8_lut[uint(code.z)]),\n"
                "                        mx_dot_" + suffix + ");\n"
                "                    mx_dot_" + suffix + " = fma(\n"
                "                        float(activation.w),\n"
                "                        float(fp8_lut[uint(code.w)]),\n"
                "                        mx_dot_" + suffix + ");\n"
                "                }\n";
        } else {
            source +=
                "                if (active_" + suffix + ") {\n"
                "                    uint value_offset = output_" + suffix
                + " * uint(K) + column;\n"
                "                    char4 quantized = *(device const char4*)(\n"
                "                        q8_q_" + suffix
                + " + value_offset);\n"
                "                    q8_dot_" + suffix + " = fma(\n"
                "                        float(activation.x),\n"
                "                        float(quantized.x),\n"
                "                        q8_dot_" + suffix + ");\n"
                "                    q8_dot_" + suffix + " = fma(\n"
                "                        float(activation.y),\n"
                "                        float(quantized.y),\n"
                "                        q8_dot_" + suffix + ");\n"
                "                    q8_dot_" + suffix + " = fma(\n"
                "                        float(activation.z),\n"
                "                        float(quantized.z),\n"
                "                        q8_dot_" + suffix + ");\n"
                "                    q8_dot_" + suffix + " = fma(\n"
                "                        float(activation.w),\n"
                "                        float(quantized.w),\n"
                "                        q8_dot_" + suffix + ");\n"
                "                }\n";
        }
    }

    source += "            }\n";
    for (
        std::size_t projection = 0;
        projection < layouts.size();
        ++projection
    ) {
        if (layouts[projection].family != kFamilyNint8Zero) {
            continue;
        }
        const auto suffix = std::to_string(projection);
        source +=
            "            if (active_" + suffix + ") {\n"
            "                uint group = block * Q8_GROUPS_PER_MX"
            " + q8_local;\n"
            "                float scale = float(q8_scales_" + suffix
            + "[output_" + suffix + " * uint(P" + suffix
            + "_NG) + group]);\n"
            "                accumulator_" + suffix + " = fma(\n"
            "                    scale, q8_dot_" + suffix + ",\n"
            "                    accumulator_" + suffix + ");\n"
            "            }\n";
    }
    source += "        }\n";
    for (
        std::size_t projection = 0;
        projection < layouts.size();
        ++projection
    ) {
        if (layouts[projection].family != kFamilyMx) {
            continue;
        }
        const auto suffix = std::to_string(projection);
        source +=
            "        if (active_" + suffix + ") {\n"
            "            uint scale_offset =\n"
            "                (output_" + suffix
            + " / MX_BLOCK) * MX_BLOCKS + block;\n"
            "            accumulator_" + suffix + " = fma(\n"
            "                mfq_grouped_mx_e8m0(\n"
            "                    mx_scales_" + suffix
            + "[scale_offset]),\n"
            "                mx_dot_" + suffix + ",\n"
            "                accumulator_" + suffix + ");\n"
            "        }\n";
    }
    source += "    }\n";

    for (
        std::size_t projection = 0;
        projection < layouts.size();
        ++projection
    ) {
        const auto suffix = std::to_string(projection);
        source +=
            "    accumulator_" + suffix
            + " += simd_shuffle_down(accumulator_" + suffix + ", 4);\n"
            "    accumulator_" + suffix
            + " += simd_shuffle_down(accumulator_" + suffix + ", 2);\n"
            "    accumulator_" + suffix
            + " += simd_shuffle_down(accumulator_" + suffix + ", 1);\n"
            "    if (k_lane == 0u && active_" + suffix + ") {\n"
            "        y[uint(P" + suffix + "_OUT_OFFSET) + output_index] =\n"
            "            half(accumulator_" + suffix + ");\n"
            "    }\n";
    }
    return source;
}

mlx::core::fast::CustomKernelFunction single_row_mxfp8_kernel(
    const std::vector<DirectProjectionLayout>& layouts) {
    static std::mutex mutex;
    static std::unordered_map<
        std::string,
        mlx::core::fast::CustomKernelFunction> kernels;
    const auto key = "mxfp8_m1_" + direct_kernel_key(layouts);
    std::lock_guard<std::mutex> lock(mutex);
    const auto found = kernels.find(key);
    if (found != kernels.end()) {
        return found->second;
    }
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    auto kernel = mlx::core::fast::metal_kernel(
        "mfq_cpp_single_row_grouped_" + key,
        direct_input_names(layouts),
        {"y"},
        make_single_row_mxfp8_source(layouts),
        kGroupedHeader,
        true,
        false,
        options);
    kernels.emplace(key, kernel);
    return kernel;
}

constexpr const char* kSingleRowMxfp8PairSwiglu = R"METAL(
    constexpr uint SIMD_GROUPS = 4u;
    constexpr uint K_LANES = 16u;
    constexpr uint ROWS_PER_TG = SIMD_GROUPS;
    constexpr uint BLOCK = 128u;
    constexpr uint BLOCKS = uint(K) / BLOCK;

    threadgroup half fp8_lut[256];
    uint local_thread = thread_index_in_threadgroup;
    fp8_lut[local_thread] = as_type<half>(
        mfq_grouped_mx_fp8_half_lut[local_thread]);
    fp8_lut[local_thread + 128u] = as_type<half>(
        mfq_grouped_mx_fp8_half_lut[local_thread + 128u]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint lane = thread_index_in_simdgroup;
    uint projection = lane >> 4u;
    uint k_lane = lane & (K_LANES - 1u);
    uint output_index =
        threadgroup_position_in_grid.x * ROWS_PER_TG
        + simdgroup_index_in_threadgroup;
    uint output = min(output_index, uint(OUT) - 1u);
    uint value_base = output * uint(K);
    uint scale_base = (output / BLOCK) * BLOCKS;
    device const uchar* values = projection == 0u
        ? mx_values_0
        : mx_values_1;
    float accumulators[M];
    for (uint row = 0u; row < uint(M); ++row) {
        accumulators[row] = 0.0f;
    }

    // Gate and Up occupy separate SIMD half-groups. Both halves visit the
    // same 128-column block at once: the 16 lanes in each half read one
    // contiguous 128-byte MXFP8 row segment instead of maintaining 16
    // strided weight streams. The activation is tiny and remains cache-hot.
    for (uint block = 0u; block < BLOCKS; ++block) {
        uint column_base = block * BLOCK;
        uint column = column_base + k_lane * 8u;
        uchar4 code0 = *(device const uchar4*)(
            values + value_base + column);
        uchar4 code1 = *(device const uchar4*)(
            values + value_base + column + 4u);
        float4 weight0 = float4(
            float(fp8_lut[uint(code0.x)]),
            float(fp8_lut[uint(code0.y)]),
            float(fp8_lut[uint(code0.z)]),
            float(fp8_lut[uint(code0.w)]));
        float4 weight1 = float4(
            float(fp8_lut[uint(code1.x)]),
            float(fp8_lut[uint(code1.y)]),
            float(fp8_lut[uint(code1.z)]),
            float(fp8_lut[uint(code1.w)]));
        uchar scale = projection == 0u
            ? mx_scales_0[scale_base + block]
            : mx_scales_1[scale_base + block];
        float block_scale = mfq_grouped_mx_e8m0(scale);
        for (uint row = 0u; row < uint(M); ++row) {
            uint input_base = row * uint(K);
            activation4_t activation0 =
                *(device const activation4_t*)(x + input_base + column);
            activation4_t activation1 = *(device const activation4_t*)(
                x + input_base + column + 4u);
            float block_dot =
                dot(float4(activation0), weight0)
                + dot(float4(activation1), weight1);
            accumulators[row] = fma(
                block_scale,
                block_dot,
                accumulators[row]);
        }
    }

    // Each 16-lane half is an independent reduction tree. Only lanes 0 and
    // 16 are consumed, so shuffle-down traffic from inactive upper nodes
    // cannot cross-contaminate the two projection sums.
    for (uint row = 0u; row < uint(M); ++row) {
        float accumulator = accumulators[row];
        accumulator += simd_shuffle_down(accumulator, 8);
        accumulator += simd_shuffle_down(accumulator, 4);
        accumulator += simd_shuffle_down(accumulator, 2);
        accumulator += simd_shuffle_down(accumulator, 1);
        float gate = simd_shuffle(accumulator, 0u);
        float up = simd_shuffle(accumulator, 16u);
        if (lane == 0u && output_index < uint(OUT)) {
            // Match the unfused MXFP8 GEMV activation-dtype boundary.
            gate = float(activation_t(gate));
            up = float(activation_t(up));
            if (params[0] > 0.0f) {
                gate = min(gate, params[0]);
                up = clamp(up, -params[0], params[0]);
            }
            float activated = gate / (1.0f + exp(-gate));
            y[row * uint(OUT) + output_index] =
                activation_t(activated * up);
        }
    }
)METAL";

const mlx::core::fast::CustomKernelFunction&
single_row_mxfp8_pair_swiglu_kernel(Dtype dtype) {
    static const auto fp16_kernel = [] {
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        return mlx::core::fast::metal_kernel(
            "mfq_cpp_single_row_mxfp8_pair_swiglu_f16",
            {
                "mx_values_0",
                "mx_scales_0",
                "mx_values_1",
                "mx_scales_1",
                "x",
                "params",
            },
            {"y"},
            std::string(
                "using activation_t = half;\n"
                "using activation4_t = half4;\n") +
                kSingleRowMxfp8PairSwiglu,
            kGroupedHeader,
            true,
            false,
            options);
    }();
    static const auto bf16_kernel = [] {
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        return mlx::core::fast::metal_kernel(
            "mfq_cpp_single_row_mxfp8_pair_swiglu_bf16",
            {
                "mx_values_0",
                "mx_scales_0",
                "mx_values_1",
                "mx_scales_1",
                "x",
                "params",
            },
            {"y"},
            std::string(
                "using activation_t = bfloat;\n"
                "using activation4_t = bfloat4;\n") +
                kSingleRowMxfp8PairSwiglu,
            kGroupedHeader,
            true,
            false,
            options);
    }();
    return dtype == mlx::core::bfloat16
        ? bf16_kernel
        : fp16_kernel;
}

void validate_direct_array(
    const array& value,
    Dtype dtype,
    const char* name) {
    if (value.dtype() != dtype ||
        !value.flags().row_contiguous) {
        throw std::runtime_error(
            std::string("invalid zero-copy grouped linear ") + name);
    }
}

int weight_input_size(const MlxGroupedLinearWeightRef& value) {
    return std::visit(
        [](const auto* weight) -> int {
            if (weight == nullptr) {
                throw std::invalid_argument(
                    "grouped linear weight cannot be null");
            }
            using Weight = std::remove_cv_t<
                std::remove_pointer_t<decltype(weight)>>;
            if constexpr (std::is_same_v<Weight, array>) {
                return weight->ndim() == 2 ? weight->shape(1) : 0;
            } else {
                return weight->input_size();
            }
        },
        value);
}

int weight_output_size(const MlxGroupedLinearWeightRef& value) {
    return std::visit(
        [](const auto* weight) -> int {
            if (weight == nullptr) {
                throw std::invalid_argument(
                    "grouped linear weight cannot be null");
            }
            using Weight = std::remove_cv_t<
                std::remove_pointer_t<decltype(weight)>>;
            if constexpr (std::is_same_v<Weight, array>) {
                return weight->ndim() == 2 ? weight->shape(0) : 0;
            } else {
                return weight->output_size();
            }
        },
        value);
}

std::size_t weight_packed_nbytes(
    const MlxGroupedLinearWeightRef& value) {
    return std::visit(
        [](const auto* weight) -> std::size_t {
            if (weight == nullptr) {
                throw std::invalid_argument(
                    "grouped linear weight cannot be null");
            }
            using Weight = std::remove_cv_t<
                std::remove_pointer_t<decltype(weight)>>;
            if constexpr (std::is_same_v<Weight, array>) {
                return weight->nbytes();
            } else {
                return weight->packed_nbytes();
            }
        },
        value);
}

} // namespace

struct MlxGroupedLinear::Impl {
    std::optional<array> descriptors;
    std::optional<array> projection_tile_offsets;
    std::optional<array> projection_output_offsets;
    std::optional<array> q8_q;
    std::optional<array> q8_scales;
    std::vector<RetainedProjection> common_kernel_weights;
    std::vector<MlxNintWeight> nint_projection_weights;
    std::vector<MlxMxWeight> mxfp8_block32_projection_weights;
    std::vector<MlxMxfp4SqWeight> mxfp4_sq_projection_weights;
    std::vector<MlxFp8SqWeight> fp8_sq_projection_weights;
    std::vector<array> dense_projection_weights;
    std::vector<DirectProjectionLayout> direct_layouts;
    std::vector<array> direct_weight_inputs;
    std::vector<int> output_sizes;
    int input_size = 0;
    int total_output_size = 0;
    int total_tiles = 0;
    int single_row_mxfp8_tiles = 0;
    std::size_t packed_bytes = 0;
    std::size_t copied_packed_bytes = 0;

    Impl(
        std::vector<RetainedProjection> weights,
        std::vector<int> widths,
        int width,
        int total_output,
        std::size_t bytes)
        : common_kernel_weights(std::move(weights)),
          output_sizes(std::move(widths)),
          input_size(width),
          total_output_size(total_output),
          packed_bytes(bytes) {}

    Impl(
        std::vector<MlxNintWeight> weights,
        std::vector<int> widths,
        int width,
        int total_output,
        std::size_t bytes)
        : nint_projection_weights(std::move(weights)),
          output_sizes(std::move(widths)),
          input_size(width),
          total_output_size(total_output),
          packed_bytes(bytes) {}

    Impl(
        std::vector<MlxMxWeight> weights,
        std::vector<int> widths,
        int width,
        int total_output,
        std::size_t bytes)
        : mxfp8_block32_projection_weights(std::move(weights)),
          output_sizes(std::move(widths)),
          input_size(width),
          total_output_size(total_output),
          packed_bytes(bytes) {}

    Impl(
        std::vector<MlxMxfp4SqWeight> weights,
        std::vector<int> widths,
        int width,
        int total_output,
        std::size_t bytes)
        : mxfp4_sq_projection_weights(std::move(weights)),
          output_sizes(std::move(widths)),
          input_size(width),
          total_output_size(total_output),
          packed_bytes(bytes) {}

    Impl(
        std::vector<MlxFp8SqWeight> weights,
        std::vector<int> widths,
        int width,
        int total_output,
        std::size_t bytes)
        : fp8_sq_projection_weights(std::move(weights)),
          output_sizes(std::move(widths)),
          input_size(width),
          total_output_size(total_output),
          packed_bytes(bytes) {}

    Impl(
        std::vector<array> weights,
        std::vector<int> widths,
        int width,
        int total_output,
        std::size_t bytes)
        : dense_projection_weights(std::move(weights)),
          output_sizes(std::move(widths)),
          input_size(width),
          total_output_size(total_output),
          packed_bytes(bytes) {}

    Impl(
        array descriptors_value,
        array tile_offsets_value,
        array output_offsets_value,
        array q8_q_value,
        array q8_scales_value,
        std::vector<int> widths,
        int width,
        int total_output,
        int tiles,
        std::size_t bytes)
        : descriptors(std::move(descriptors_value)),
          projection_tile_offsets(
              std::move(tile_offsets_value)),
          projection_output_offsets(
              std::move(output_offsets_value)),
          q8_q(std::move(q8_q_value)),
          q8_scales(std::move(q8_scales_value)),
          output_sizes(std::move(widths)),
          input_size(width),
          total_output_size(total_output),
          total_tiles(tiles),
          packed_bytes(bytes),
          copied_packed_bytes(bytes) {}

    Impl(
        std::vector<DirectProjectionLayout> layouts,
        std::vector<array> weight_inputs,
        std::vector<int> widths,
        int width,
        int total_output,
        int tiles,
        std::size_t bytes)
        : direct_layouts(std::move(layouts)),
          direct_weight_inputs(std::move(weight_inputs)),
          output_sizes(std::move(widths)),
          input_size(width),
          total_output_size(total_output),
          total_tiles(tiles),
          packed_bytes(bytes) {
        if (supports_single_row_mxfp8_fast_path(
                direct_layouts)) {
            for (const auto& layout : direct_layouts) {
                single_row_mxfp8_tiles = std::max(
                    single_row_mxfp8_tiles,
                    (layout.output_size + 15) / 16);
            }
        }
    }

    bool uses_zero_copy_storage() const noexcept {
        return !common_kernel_weights.empty()
            || !nint_projection_weights.empty()
            || !mxfp8_block32_projection_weights.empty()
            || !mxfp4_sq_projection_weights.empty()
            || !fp8_sq_projection_weights.empty()
            || !dense_projection_weights.empty()
            || !direct_layouts.empty();
    }

    bool has_nint_projection_group() const noexcept {
        return !nint_projection_weights.empty();
    }

    bool has_mxfp4_sq_projection_group() const noexcept {
        return !mxfp4_sq_projection_weights.empty();
    }

    bool has_mxfp8_block32_projection_group() const noexcept {
        return !mxfp8_block32_projection_weights.empty();
    }

    bool has_fp8_sq_projection_group() const noexcept {
        return !fp8_sq_projection_weights.empty();
    }

    bool has_sq_projection_group() const noexcept {
        return has_mxfp4_sq_projection_group() ||
            has_fp8_sq_projection_group();
    }

    bool has_dense_projection_group() const noexcept {
        return !dense_projection_weights.empty();
    }

    bool has_nint_swiglu_pair() const noexcept {
        return nint_projection_weights.size() == 2
            && output_sizes.size() == 2
            && output_sizes[0] == output_sizes[1]
            && nint_projection_weights[0].can_fuse_swiglu(
                nint_projection_weights[1]);
    }

    bool has_single_row_mxfp8_fast_path() const noexcept {
        return single_row_mxfp8_tiles > 0;
    }

    bool supports_single_row_swiglu() const noexcept {
        if (has_nint_swiglu_pair()) {
            return true;
        }
        if (direct_layouts.size() != 2
            || output_sizes.size() != 2
            || output_sizes[0] != output_sizes[1]) {
            return false;
        }
        const bool mxfp8_pair =
            has_single_row_mxfp8_fast_path()
            && direct_layouts[0].family == kFamilyMx
            && direct_layouts[0].bits == 8
            && direct_layouts[1].family == kFamilyMx
            && direct_layouts[1].bits == 8;
        return mxfp8_pair;
    }

    bool supports_bf16_single_row_swiglu() const noexcept {
        return direct_layouts.size() == 2
            && output_sizes.size() == 2
            && output_sizes[0] == output_sizes[1]
            && has_single_row_mxfp8_fast_path()
            && direct_layouts[0].family == kFamilyMx
            && direct_layouts[0].bits == 8
            && direct_layouts[1].family == kFamilyMx
            && direct_layouts[1].bits == 8;
    }

    bool supports_bf16_matmul() const noexcept {
        // Individual MX/NINT8-0 linears preserve a BF16 model boundary by
        // projecting through FP16 and casting their result back to BF16.
        // The zero-copy group can do the same conversion once for the shared
        // input and once for the combined output, while retaining the native
        // FP16 MXFP8/small-M kernels. Keep the broader retained/NINT families
        // on their established dtype contract until each is verified.
        if (has_sq_projection_group()) {
            return true;
        }
        if (has_dense_projection_group()) {
            return dense_projection_weights.front().dtype()
                == mlx::core::bfloat16;
        }
        if (has_mxfp8_block32_projection_group()) {
            return true;
        }
        return !direct_layouts.empty() && std::all_of(
            direct_layouts.begin(),
            direct_layouts.end(),
            [](const DirectProjectionLayout& layout) {
                return layout.family == kFamilyMx ||
                    layout.family == kFamilyNint8Zero;
            });
    }
};

MlxGroupedLinear::MlxGroupedLinear(
    std::vector<MlxGroupedLinearWeightRef> weights) {
    if (weights.size() < 2) {
        throw std::invalid_argument(
            "grouped linear requires at least two projections");
    }
    const int shared_input = weight_input_size(weights.front());
    if (shared_input <= 0) {
        throw std::invalid_argument(
            "grouped linear input width must be positive");
    }

    // Dense control/compressor projections are common beside packed Q/K/V
    // weights. They previously remained separate MLX matmul submissions even
    // when every projection consumed the same activation. Retain the original
    // arrays and bind a homogeneous FP16/BF16 cohort to one small-M dispatch;
    // mixed dense/packed lists are partitioned by MlxProjectionBatch.
    const bool contains_dense = std::any_of(
        weights.begin(),
        weights.end(),
        [](const MlxGroupedLinearWeightRef& weight) {
            return std::holds_alternative<const array*>(weight);
        });
    if (contains_dense) {
        const bool all_dense = std::all_of(
            weights.begin(),
            weights.end(),
            [](const MlxGroupedLinearWeightRef& weight) {
                return std::holds_alternative<const array*>(weight);
            });
        if (!all_dense || weights.size() > 14) {
            throw MlxGroupedLinearUnsupported(
                "dense projection group requires one homogeneous family");
        }
        const auto* first = std::get<const array*>(weights.front());
        if (first == nullptr || first->ndim() != 2 ||
            (first->dtype() != mlx::core::float16 &&
             first->dtype() != mlx::core::bfloat16)) {
            throw MlxGroupedLinearUnsupported(
                "dense projection group requires FP16/BF16 matrices");
        }
        std::vector<array> retained;
        std::vector<int> output_sizes;
        retained.reserve(weights.size());
        output_sizes.reserve(weights.size());
        int total_output = 0;
        std::size_t packed_bytes = 0;
        for (const auto& weight : weights) {
            const auto* value = std::get<const array*>(weight);
            if (value == nullptr || value->ndim() != 2 ||
                value->shape(1) != shared_input ||
                value->dtype() != first->dtype() ||
                !value->flags().row_contiguous) {
                throw MlxGroupedLinearUnsupported(
                    "dense grouped projections must share width, dtype, and row layout");
            }
            const int output = value->shape(0);
            if (output <= 0) {
                throw std::invalid_argument(
                    "dense grouped projection output must be positive");
            }
            total_output = checked_int(
                static_cast<std::size_t>(total_output) +
                    static_cast<std::size_t>(output),
                "dense grouped total output width");
            if (value->nbytes() >
                std::numeric_limits<std::size_t>::max() - packed_bytes) {
                throw MlxGroupedLinearUnsupported(
                    "dense grouped projection size overflows");
            }
            packed_bytes += value->nbytes();
            output_sizes.push_back(output);
            retained.push_back(*value);
        }
        impl_ = std::make_shared<Impl>(
            std::move(retained),
            std::move(output_sizes),
            shared_input,
            total_output,
            packed_bytes);
        return;
    }

    // Native-QAT SQ projections share one metadata-driven dispatch per format
    // family.  Keep their packed blobs separate and bind them zero-copy; this
    // avoids both a concatenated model-sized pool and architecture-local QKV or
    // gate/up fallbacks.  Mixed scale families are partitioned by the generic
    // MlxProjectionBatch coordinator instead of being silently reinterpreted.
    const bool contains_mxfp4_sq = std::any_of(
        weights.begin(),
        weights.end(),
        [](const MlxGroupedLinearWeightRef& weight) {
            return std::holds_alternative<
                const MlxMxfp4SqWeight*>(weight);
        });
    const bool contains_fp8_sq = std::any_of(
        weights.begin(),
        weights.end(),
        [](const MlxGroupedLinearWeightRef& weight) {
            return std::holds_alternative<
                const MlxFp8SqWeight*>(weight);
        });
    if (contains_mxfp4_sq || contains_fp8_sq) {
        const bool all_mxfp4_sq = std::all_of(
            weights.begin(),
            weights.end(),
            [](const MlxGroupedLinearWeightRef& weight) {
                return std::holds_alternative<
                    const MlxMxfp4SqWeight*>(weight);
            });
        const bool all_fp8_sq = std::all_of(
            weights.begin(),
            weights.end(),
            [](const MlxGroupedLinearWeightRef& weight) {
                return std::holds_alternative<
                    const MlxFp8SqWeight*>(weight);
            });
        if ((!all_mxfp4_sq && !all_fp8_sq) ||
            (all_mxfp4_sq && weights.size() > 5) ||
            (all_fp8_sq && weights.size() > 10)) {
            throw MlxGroupedLinearUnsupported(
                "native-QAT SQ projection group requires one format family");
        }

        std::vector<int> output_sizes;
        output_sizes.reserve(weights.size());
        int total_output = 0;
        std::size_t packed_bytes = 0;
        if (all_mxfp4_sq) {
            std::vector<MlxMxfp4SqWeight> retained;
            retained.reserve(weights.size());
            for (const auto& weight : weights) {
                const auto* value =
                    std::get<const MlxMxfp4SqWeight*>(weight);
                if (value == nullptr || value->input_size() != shared_input) {
                    throw std::invalid_argument(
                        "MXFP4-SQ grouped projections must share one input width");
                }
                total_output = checked_int(
                    static_cast<std::size_t>(total_output) +
                        static_cast<std::size_t>(value->output_size()),
                    "total output width");
                if (value->packed_nbytes() >
                    std::numeric_limits<std::size_t>::max() - packed_bytes) {
                    throw MlxGroupedLinearUnsupported(
                        "MXFP4-SQ grouped packed size overflows");
                }
                packed_bytes += value->packed_nbytes();
                output_sizes.push_back(value->output_size());
                retained.push_back(*value);
            }
            impl_ = std::make_shared<Impl>(
                std::move(retained),
                std::move(output_sizes),
                shared_input,
                total_output,
                packed_bytes);
            return;
        }

        std::vector<MlxFp8SqWeight> retained;
        retained.reserve(weights.size());
        const auto* first = std::get<const MlxFp8SqWeight*>(weights.front());
        for (const auto& weight : weights) {
            const auto* value = std::get<const MlxFp8SqWeight*>(weight);
            if (value == nullptr || first == nullptr ||
                value->input_size() != shared_input ||
                value->dtype() != first->dtype()) {
                throw MlxGroupedLinearUnsupported(
                    "FP8-SQ grouped projections must share width and scale family");
            }
            total_output = checked_int(
                static_cast<std::size_t>(total_output) +
                    static_cast<std::size_t>(value->output_size()),
                "total output width");
            if (value->packed_nbytes() >
                std::numeric_limits<std::size_t>::max() - packed_bytes) {
                throw MlxGroupedLinearUnsupported(
                    "FP8-SQ grouped packed size overflows");
            }
            packed_bytes += value->packed_nbytes();
            output_sizes.push_back(value->output_size());
            retained.push_back(*value);
        }
        impl_ = std::make_shared<Impl>(
            std::move(retained),
            std::move(output_sizes),
            shared_input,
            total_output,
            packed_bytes);
        return;
    }

    // QAT attention commonly stores MXFP8 with one scale per 32 columns
    // (either every row or every 32 rows).  The older direct projection kernel
    // only understood 128x128 scale blocks, so merely registering a projection
    // batch silently excluded the real released-model geometry.  Retain each
    // payload and its expanded row/32 sidecar separately and execute the whole
    // compatible cohort in the format-level kernel.
    const bool contains_block32_mxfp8 = std::any_of(
        weights.begin(),
        weights.end(),
        [](const MlxGroupedLinearWeightRef& weight) {
            const auto* mx = std::get_if<const MlxMxWeight*>(&weight);
            return mx != nullptr && *mx != nullptr &&
                (*mx)->bits() == 8 &&
                (*mx)->scale_column_block_size() == 32;
        });
    if (contains_block32_mxfp8) {
        const bool all_block32_mxfp8 = std::all_of(
            weights.begin(),
            weights.end(),
            [](const MlxGroupedLinearWeightRef& weight) {
                const auto* mx = std::get_if<const MlxMxWeight*>(&weight);
                return mx != nullptr && *mx != nullptr &&
                    (*mx)->bits() == 8 &&
                    (*mx)->scale_column_block_size() == 32;
            });
        if (!all_block32_mxfp8 || weights.size() > 14) {
            throw MlxGroupedLinearUnsupported(
                "column-32 MXFP8 projection group requires one format family");
        }
        std::vector<MlxMxWeight> retained;
        std::vector<int> output_sizes;
        retained.reserve(weights.size());
        output_sizes.reserve(weights.size());
        int total_output = 0;
        std::size_t packed_bytes = 0;
        for (const auto& weight : weights) {
            const auto* value = std::get<const MlxMxWeight*>(weight);
            if (value == nullptr || value->input_size() != shared_input) {
                throw std::invalid_argument(
                    "column-32 MXFP8 projections must share one input width");
            }
            total_output = checked_int(
                static_cast<std::size_t>(total_output) +
                    static_cast<std::size_t>(value->output_size()),
                "total output width");
            if (value->packed_nbytes() >
                std::numeric_limits<std::size_t>::max() - packed_bytes) {
                throw MlxGroupedLinearUnsupported(
                    "column-32 MXFP8 grouped packed size overflows");
            }
            packed_bytes += value->packed_nbytes();
            output_sizes.push_back(value->output_size());
            retained.push_back(*value);
        }
        impl_ = std::make_shared<Impl>(
            std::move(retained),
            std::move(output_sizes),
            shared_input,
            total_output,
            packed_bytes);
        return;
    }

    // Two- and three-projection NINT groups use one projection-fused operator
    // whose decoder consumes the same per-row q/k metadata as the standalone
    // NINT kernel. Larger or mixed-format groups remain graph compositions;
    // no uniform-q or precision-specific NINT implementation is retained.
    const bool contains_nint = std::any_of(
        weights.begin(),
        weights.end(),
        [](const MlxGroupedLinearWeightRef& weight) {
            return std::holds_alternative<
                const MlxNintWeight*>(weight);
        });
    if (contains_nint) {
        const bool all_nint = std::all_of(
            weights.begin(),
            weights.end(),
            [](const MlxGroupedLinearWeightRef& weight) {
                return std::holds_alternative<
                    const MlxNintWeight*>(weight);
            });
        std::vector<RetainedProjection> retained;
        std::vector<int> output_sizes;
        retained.reserve(weights.size());
        output_sizes.reserve(weights.size());
        int total_output = 0;
        std::size_t packed_bytes = 0;
        for (const auto& weight : weights) {
            if (const auto* value = std::get_if<
                    const MlxVqWeight*>(&weight);
                value != nullptr && *value != nullptr &&
                ((*value)->output_shape().size() != 1 ||
                 (*value)->rotation_block() != 0)) {
                throw MlxGroupedLinearUnsupported(
                    "ordinary grouped linear rejects "
                    "expert-shaped or rotated NEPQ weights");
            }
            if (weight_input_size(weight) != shared_input) {
                throw std::invalid_argument(
                    "grouped linear projections must share one input width");
            }
            const int output = weight_output_size(weight);
            if (output <= 0) {
                throw std::invalid_argument(
                    "grouped linear output width must be positive");
            }
            total_output = checked_int(
                static_cast<std::size_t>(total_output)
                    + static_cast<std::size_t>(output),
                "total output width");
            const auto bytes = weight_packed_nbytes(weight);
            if (bytes > std::numeric_limits<std::size_t>::max()
                    - packed_bytes) {
                throw MlxGroupedLinearUnsupported(
                    "grouped linear packed stream size overflows");
            }
            packed_bytes += bytes;
            output_sizes.push_back(output);
            retained.push_back(retain_projection(weight));
        }

        bool can_fuse_projections = all_nint && weights.size() <= 3;
        std::vector<MlxNintWeight> nint_weights;
        if (can_fuse_projections) {
            nint_weights.reserve(weights.size());
            const auto* first = std::get<const MlxNintWeight*>(
                weights.front());
            for (const auto& weight : weights) {
                const auto* value = std::get<const MlxNintWeight*>(weight);
                if (value == nullptr || first == nullptr ||
                    value->group_size() != first->group_size() ||
                    value->groups() != first->groups()) {
                    can_fuse_projections = false;
                    break;
                }
                validate_direct_array(
                    value->packed_values(),
                    mlx::core::uint8,
                    "NINT q");
                validate_direct_array(
                    value->row_q_layout(),
                    mlx::core::uint8,
                    "NINT row q layout");
                validate_direct_array(
                    value->row_q_byte_offsets(),
                    mlx::core::uint32,
                    "NINT row q byte offsets");
                validate_direct_array(
                    value->sub_scales(),
                    mlx::core::uint8,
                    "NINT sub scales");
                validate_direct_array(
                    value->sub_mins(),
                    mlx::core::uint8,
                    "NINT sub minima");
                validate_direct_array(
                    value->neuron_scales(),
                    mlx::core::float32,
                    "NINT neuron scales");
                validate_direct_array(
                    value->neuron_mins(),
                    mlx::core::float32,
                    "NINT neuron minima");
                nint_weights.push_back(*value);
            }
        }
        if (can_fuse_projections) {
            impl_ = std::make_shared<Impl>(
                std::move(nint_weights),
                std::move(output_sizes),
                shared_input,
                total_output,
                packed_bytes);
            return;
        }
        impl_ = std::make_shared<Impl>(
            std::move(retained),
            std::move(output_sizes),
            shared_input,
            total_output,
            packed_bytes);
        return;
    }

    // Q/K/V and gate/up are the production hot paths. Binding the source
    // arrays directly avoids materializing a second model-sized packed pool.
    // Three VQ projections are the largest supported binding: 27 weight
    // buffers (29 resources including x/y), below Metal's 31-buffer kernel
    // argument limit.
    const bool direct_mxfp8_group =
        weights.size() <= 14 &&
        std::any_of(
            weights.begin(),
            weights.end(),
            [](const MlxGroupedLinearWeightRef& weight) {
                const auto* mx = std::get_if<
                    const MlxMxWeight*>(&weight);
                return mx != nullptr && *mx != nullptr &&
                    (*mx)->bits() == 8;
            }) &&
        std::all_of(
            weights.begin(),
            weights.end(),
            [](const MlxGroupedLinearWeightRef& weight) {
                if (std::holds_alternative<
                        const MlxNint8ZeroWeight*>(weight)) {
                    return true;
                }
                const auto* mx = std::get_if<
                    const MlxMxWeight*>(&weight);
                return mx != nullptr && *mx != nullptr &&
                    (*mx)->bits() == 8;
            });
    if (weights.size() <= 3 || direct_mxfp8_group) {
        std::vector<DirectProjectionLayout> layouts;
        std::vector<array> direct_inputs;
        std::vector<int> output_sizes;
        layouts.reserve(weights.size());
        direct_inputs.reserve(weights.size() * 9);
        output_sizes.reserve(weights.size());
        int total_output = 0;
        int total_tiles = 0;
        std::size_t packed_bytes = 0;

        for (
            std::size_t projection = 0;
            projection < weights.size();
            ++projection
        ) {
            const auto& source = weights[projection];
            if (weight_input_size(source) != shared_input) {
                throw std::invalid_argument(
                    "grouped linear projections must share one input width");
            }
            const int output = weight_output_size(source);
            if (output <= 0) {
                throw std::invalid_argument(
                    "grouped linear output width must be positive");
            }

            DirectProjectionLayout layout;
            layout.output_size = output;
            layout.tile_begin = total_tiles;
            layout.output_offset = total_output;
            total_output = checked_int(
                static_cast<std::size_t>(total_output)
                    + static_cast<std::size_t>(output),
                "total output width");
            total_tiles = checked_int(
                static_cast<std::size_t>(total_tiles)
                    + static_cast<std::size_t>((output + 7) / 8),
                "tile count");
            layout.tile_end = total_tiles;

            std::size_t weight_bytes = 0;
            std::visit(
                [&](const auto* weight) {
                    using Weight = std::remove_cv_t<
                        std::remove_pointer_t<decltype(weight)>>;
                    if constexpr (
                        std::is_same_v<Weight, MlxNintWeight>
                    ) {
                        throw MlxGroupedLinearUnsupported(
                            "NINT cannot enter the heterogeneous direct kernel");
                    } else if constexpr (
                        std::is_same_v<Weight, MlxFp8SqWeight> ||
                        std::is_same_v<Weight, MlxMxfp4SqWeight>
                    ) {
                        throw MlxGroupedLinearUnsupported(
                            "native-QAT SQ uses its format-level projection group");
                    } else if constexpr (
                        std::is_same_v<
                            Weight,
                            MlxNint8ZeroWeight>
                    ) {
                        layout.family = kFamilyNint8Zero;
                        layout.groups = weight->groups();
                        validate_direct_array(
                            weight->quantized_values(),
                            mlx::core::int8,
                            "NINT8-0 q");
                        validate_direct_array(
                            weight->scales(),
                            mlx::core::float16,
                            "NINT8-0 scales");
                        direct_inputs.push_back(
                            weight->quantized_values());
                        direct_inputs.push_back(
                            weight->scales());
                    } else if constexpr (
                        std::is_same_v<
                            Weight,
                            MlxVqWeight>
                    ) {
                        if (weight->output_shape().size() != 1 ||
                            weight->rotation_block() != 0) {
                            throw MlxGroupedLinearUnsupported(
                                "ordinary grouped linear rejects "
                                "expert-shaped or rotated NEPQ weights");
                        }
                        layout.family = kFamilyVq;
                        layout.group_size =
                            weight->group_size();
                        layout.groups = weight->groups();
                        layout.vector_size =
                            weight->vector_size();
                        layout.vectors = weight->vectors();
                        layout.index_bits =
                            weight->index_bits();
                        layout.state_bits =
                            weight->state_bits();
                        layout.states = weight->states();
                        layout.entries = weight->entries();
                        layout.code_banks =
                            weight->code_banks();
                        layout.aux_mode =
                            weight->aux_mode();
                        layout.code_bank_mode =
                            weight->code_bank_mode();
                        layout.execution_layout =
                            weight->execution_layout();
                        layout.table_banks =
                            weight->table_banks();
                        layout.groups_per_supergroup =
                            weight->groups_per_supergroup();
                        layout.supergroups =
                            weight->supergroups();
                        validate_direct_array(
                            weight->packed_indices(),
                            mlx::core::uint8,
                            "VQ indices");
                        validate_direct_array(
                            weight->packed_states(),
                            mlx::core::uint8,
                            "VQ states");
                        validate_direct_array(
                            weight->packed_auxiliary(),
                            mlx::core::uint8,
                            "VQ auxiliary stream");
                        validate_direct_array(
                            weight->anchors(),
                            mlx::core::float32,
                            "VQ anchors");
                        validate_direct_array(
                            weight->codebooks(),
                            mlx::core::int8,
                            "VQ codebooks");
                        validate_direct_array(
                            weight->scale_lut(),
                            mlx::core::float32,
                            "VQ scale LUT");
                        validate_direct_array(
                            weight->state_to_codebank(),
                            mlx::core::uint8,
                            "VQ state-to-bank map");
                        validate_direct_array(
                            weight->bank_ids(),
                            mlx::core::uint8,
                            "VQ table-bank selectors");
                        validate_direct_array(
                            weight->parameters(),
                            mlx::core::float32,
                            "VQ parameters");
                        direct_inputs.push_back(
                            weight->packed_indices());
                        direct_inputs.push_back(
                            weight->packed_states());
                        direct_inputs.push_back(
                            weight->packed_auxiliary());
                        direct_inputs.push_back(
                            weight->anchors());
                        direct_inputs.push_back(
                            weight->codebooks());
                        direct_inputs.push_back(
                            weight->scale_lut());
                        direct_inputs.push_back(
                            weight->state_to_codebank());
                        direct_inputs.push_back(
                            weight->bank_ids());
                        direct_inputs.push_back(
                            weight->parameters());
                    } else if constexpr (
                        std::is_same_v<Weight, MlxMxWeight>
                    ) {
                        layout.family = kFamilyMx;
                        layout.bits = weight->bits();
                        validate_direct_array(
                            weight->packed_values(),
                            mlx::core::uint8,
                            "MX packed values");
                        validate_direct_array(
                            weight->block_scales(),
                            mlx::core::uint8,
                            "MX block scales");
                        direct_inputs.push_back(
                            weight->packed_values());
                        direct_inputs.push_back(
                            weight->block_scales());
                    } else if constexpr (
                        std::is_same_v<Weight, array>
                    ) {
                        throw MlxGroupedLinearUnsupported(
                            "dense projections require the dense grouped kernel");
                    }
                    if constexpr (std::is_same_v<Weight, array>) {
                        weight_bytes = weight->nbytes();
                    } else {
                        weight_bytes = weight->packed_nbytes();
                    }
                },
                source);
            if (weight_bytes >
                std::numeric_limits<std::size_t>::max()
                    - packed_bytes) {
                throw MlxGroupedLinearUnsupported(
                    "grouped linear packed stream size overflows");
            }
            packed_bytes += weight_bytes;
            layouts.push_back(layout);
            output_sizes.push_back(output);
        }

        // Metal exposes at most 31 buffer arguments. Account for x and y in
        // addition to the retained source arrays and reject safely before
        // constructing a custom kernel which the driver cannot compile.
        if (direct_inputs.size() + 2 > 31) {
            throw MlxGroupedLinearUnsupported(
                "grouped linear direct binding exceeds "
                "Metal's 31-buffer argument limit");
        }

        impl_ = std::make_shared<Impl>(
            std::move(layouts),
            std::move(direct_inputs),
            std::move(output_sizes),
            shared_input,
            total_output,
            total_tiles,
            packed_bytes);
        return;
    }

    if (std::any_of(
            weights.begin(),
            weights.end(),
            [](const MlxGroupedLinearWeightRef& weight) {
                return std::holds_alternative<
                           const MlxVqWeight*>(weight) ||
                    std::holds_alternative<
                        const MlxMxWeight*>(weight);
            })) {
        throw MlxGroupedLinearUnsupported(
            "VQ/MX grouped linear requires the direct-binding "
            "zero-copy path");
    }

    std::vector<std::int32_t> descriptors(
        weights.size() * kDescriptorSize,
        0);
    std::vector<std::int32_t> tile_offsets(
        weights.size() + 1,
        0);
    std::vector<std::int32_t> output_offsets(
        weights.size() + 1,
        0);
    std::vector<int> output_sizes;
    output_sizes.reserve(weights.size());

    std::vector<std::uint8_t> q8_q;
    std::vector<std::uint8_t> q8_scales;

    for (
        std::size_t projection = 0;
        projection < weights.size();
        ++projection
    ) {
        const auto& source = weights[projection];
        if (weight_input_size(source) != shared_input) {
            throw std::invalid_argument(
                "grouped linear projections must share one input width");
        }
        const int output = weight_output_size(source);
        if (output <= 0) {
            throw std::invalid_argument(
                "grouped linear output width must be positive");
        }
        output_sizes.push_back(output);

        const auto base = projection * kDescriptorSize;
        descriptors[base + kOutput] = output;
        output_offsets[projection + 1] = checked_int(
            static_cast<std::size_t>(output_offsets[projection])
                + static_cast<std::size_t>(output),
            "total output width");
        tile_offsets[projection + 1] = checked_int(
            static_cast<std::size_t>(tile_offsets[projection])
                + static_cast<std::size_t>((output + 7) / 8),
            "tile count");

        std::visit(
            [&](const auto* weight) {
                using Weight = std::remove_cv_t<
                    std::remove_pointer_t<decltype(weight)>>;
                if constexpr (
                    std::is_same_v<Weight, MlxNintWeight>
                ) {
                    throw MlxGroupedLinearUnsupported(
                        "NINT cannot enter the heterogeneous pooled kernel");
                } else if constexpr (
                    std::is_same_v<
                        Weight,
                        MlxNint8ZeroWeight>
                ) {
                    descriptors[base + kFamily] =
                        kFamilyNint8Zero;
                    descriptors[base + kQ8Groups] =
                        weight->groups();
                    descriptors[base + kQ8QOffset] =
                        checked_int(q8_q.size(), "NINT8-0 q offset");
                    descriptors[base + kQ8ScaleOffset] =
                        checked_int(
                            q8_scales.size() / sizeof(std::uint16_t),
                            "NINT8-0 scale offset");
                    append_raw(
                        q8_q,
                        weight->quantized_values(),
                        mlx::core::int8,
                        "NINT8-0 q");
                    append_raw(
                        q8_scales,
                        weight->scales(),
                        mlx::core::float16,
                        "NINT8-0 scales");
                } else {
                    throw MlxGroupedLinearUnsupported(
                        "VQ grouped linear cannot use "
                        "the copied pooled fallback");
                }
            },
            source);
    }

    const auto descriptor_shape = Shape{
        checked_int(weights.size(), "projection count"),
        kDescriptorSize,
    };
    const auto offsets_shape = Shape{
        checked_int(weights.size() + 1, "offset count"),
    };
    const std::size_t packed_bytes = q8_q.size() + q8_scales.size();

    impl_ = std::make_shared<Impl>(
        make_int32_array(descriptors, descriptor_shape),
        make_int32_array(tile_offsets, offsets_shape),
        make_int32_array(output_offsets, offsets_shape),
        make_raw_array(std::move(q8_q), mlx::core::int8),
        make_raw_array(
            std::move(q8_scales),
            mlx::core::float16),
        std::move(output_sizes),
        shared_input,
        output_offsets.back(),
        tile_offsets.back(),
        packed_bytes);
}

array MlxGroupedLinear::run_nint_projection_group(
    const array& source,
    std::size_t rows,
    bool swiglu,
    float limit) const {
    if (!impl_->has_nint_projection_group() || rows < 1 || rows > 6) {
        throw MlxGroupedLinearUnsupported(
            "grouped NINT metadata operator supports one through six rows");
    }
    if (swiglu && !impl_->has_nint_swiglu_pair()) {
        throw MlxGroupedLinearUnsupported(
            "grouped NINT SwiGLU requires two equal-width projections");
    }
    const auto projection_count =
        impl_->nint_projection_weights.size();
    const int max_output = *std::max_element(
        impl_->output_sizes.begin(),
        impl_->output_sizes.end());
    constexpr std::size_t outputs_per_threadgroup = 16;
    constexpr std::size_t threads_per_threadgroup = 256;
    const auto output_threadgroups =
        (static_cast<std::size_t>(max_output)
            + outputs_per_threadgroup - 1) /
        outputs_per_threadgroup;
    const auto grid_x = output_threadgroups * threads_per_threadgroup;
    if (grid_x > static_cast<std::size_t>(
            std::numeric_limits<int>::max())) {
        throw MlxGroupedLinearUnsupported(
            "grouped NINT Metal grid exceeds MLX limits");
    }

    std::vector<array> inputs;
    inputs.reserve(projection_count * 7 + 2);
    for (const auto& weight : impl_->nint_projection_weights) {
        inputs.push_back(weight.packed_values());
        inputs.push_back(weight.row_q_layout());
        inputs.push_back(weight.row_q_byte_offsets());
        inputs.push_back(weight.sub_scales());
        inputs.push_back(weight.sub_mins());
        inputs.push_back(weight.neuron_scales());
        inputs.push_back(weight.neuron_mins());
    }
    inputs.push_back(source);
    inputs.push_back(array({limit}, mlx::core::float32));

    std::vector<
        std::pair<std::string, mlx::core::fast::TemplateArg>>
        templates{
            {"T", source.dtype()},
            {"GS", impl_->nint_projection_weights.front().group_size()},
            {"NG", impl_->nint_projection_weights.front().groups()},
            {"K", impl_->input_size},
            {"M", static_cast<int>(rows)},
            {"TILE_M", static_cast<int>(rows)},
            {"MAX_OUT", max_output},
            {"TOTAL_OUT", impl_->total_output_size},
            {"SWIGLU", static_cast<int>(swiglu)},
        };
    int output_offset = 0;
    for (std::size_t projection = 0;
         projection < projection_count;
         ++projection) {
        const auto prefix = "P" + std::to_string(projection) + "_";
        templates.emplace_back(
            prefix + "OUT",
            impl_->output_sizes[projection]);
        templates.emplace_back(prefix + "OFFSET", output_offset);
        output_offset += impl_->output_sizes[projection];
    }
    const int output_width = swiglu
        ? impl_->output_sizes[0]
        : impl_->total_output_size;
    return nint_projection_group_kernel(projection_count)(
        std::move(inputs),
        {Shape{static_cast<std::int32_t>(rows), output_width}},
        {source.dtype()},
        {static_cast<int>(grid_x), 1, 1},
        {static_cast<int>(threads_per_threadgroup), 1, 1},
        std::move(templates),
        std::nullopt,
        false,
        {}).front();
}

bool MlxGroupedLinear::supports(
    const array& input) const noexcept {
    if (input.ndim() == 0 ||
        input.shape(-1) != impl_->input_size ||
        (input.dtype() != mlx::core::float16 &&
         input.dtype() != mlx::core::float32 &&
         (input.dtype() != mlx::core::bfloat16 ||
          !impl_->supports_bf16_matmul()))) {
        return false;
    }
    if (impl_->has_mxfp4_sq_projection_group() &&
        input.dtype() == mlx::core::float32) {
        // The native MXFP4-SQ packed projection kernel consumes FP16 vectors.
        // Preserve the FP32 contract through each standalone weight instead of
        // claiming that the grouped launch can fuse this uncommon debug path.
        return false;
    }
    if (impl_->has_mxfp8_block32_projection_group() &&
        input.dtype() == mlx::core::float32) {
        // Native row/32 MXFP8 projection groups intentionally preserve the
        // production FP16/BF16 boundary. Debug FP32 remains an exact sequence
        // of the standalone projections.
        return false;
    }
    std::size_t rows = 1;
    for (
        std::size_t dimension = 0;
        dimension + 1 < input.ndim();
        ++dimension
    ) {
        const int value =
            input.shape(static_cast<int>(dimension));
        if (value <= 0 ||
            rows > static_cast<std::size_t>(max_rows()) /
                static_cast<std::size_t>(value)) {
            return false;
        }
        rows *= static_cast<std::size_t>(value);
    }
    return rows >= 1 &&
        rows <= static_cast<std::size_t>(max_rows()) &&
        (!impl_->has_mxfp8_block32_projection_group() || rows <= 6);
}

bool MlxGroupedLinear::supports_single_row_swiglu(
    const array& input) const noexcept {
    const bool nint_pair = impl_->has_nint_swiglu_pair();
    return impl_->supports_single_row_swiglu()
        && input.ndim() > 0
        && input.shape(-1) == impl_->input_size
        && (nint_pair
            ? (input.dtype() == mlx::core::float16 ||
               input.dtype() == mlx::core::float32)
            : (input.dtype() == mlx::core::float16 ||
               (input.dtype() == mlx::core::bfloat16 &&
                impl_->supports_bf16_single_row_swiglu())))
        && input.size() == static_cast<std::size_t>(
            impl_->input_size);
}

bool MlxGroupedLinear::supports_small_m_swiglu(
    const array& input) const noexcept {
    if (impl_->has_nint_swiglu_pair() &&
        input.ndim() > 0 &&
        input.shape(-1) == impl_->input_size &&
        (input.dtype() == mlx::core::float16 ||
         input.dtype() == mlx::core::float32)) {
        const auto rows = input.size() /
            static_cast<std::size_t>(impl_->input_size);
        return rows >= 2 && rows <= 6;
    }
    if (!impl_->has_single_row_mxfp8_fast_path()
        || impl_->direct_layouts.size() != 2
        || impl_->direct_layouts[0].family != kFamilyMx
        || impl_->direct_layouts[0].bits != 8
        || impl_->direct_layouts[1].family != kFamilyMx
        || impl_->direct_layouts[1].bits != 8
        || input.ndim() == 0
        || input.shape(-1) != impl_->input_size
        || (input.dtype() != mlx::core::float16
            && input.dtype() != mlx::core::bfloat16)) {
        return false;
    }
    const auto rows = input.size() /
        static_cast<std::size_t>(impl_->input_size);
    return rows >= 2 && rows <= 6;
}

array MlxGroupedLinear::single_row_swiglu(
    const array& input,
    float limit) const {
    if (!supports_single_row_swiglu(input)) {
        throw MlxGroupedLinearUnsupported(
            "grouped SwiGLU requires one FP16/BF16 row and two "
            "equal-width MXFP8 projections");
    }
    if (!std::isfinite(limit) || limit < 0.0f) {
        throw std::invalid_argument(
            "grouped SwiGLU limit must be finite and non-negative");
    }
    if (impl_->has_nint_swiglu_pair()) {
        if (limit == 0.0f) {
            return impl_->nint_projection_weights[0].swiglu(
                impl_->nint_projection_weights[1],
                input);
        }
        Shape output_shape(
            input.shape().begin(),
            input.shape().end() - 1);
        output_shape.push_back(impl_->output_sizes[0]);
        auto source = mlx::core::contiguous(mlx::core::reshape(
            input,
            Shape{1, impl_->input_size}));
        return mlx::core::reshape(
            run_nint_projection_group(source, 1, true, limit),
            std::move(output_shape));
    }

    Shape prefix(
        input.shape().begin(),
        input.shape().end() - 1);
    auto output_shape = prefix;
    output_shape.push_back(impl_->output_sizes[0]);
    auto source = mlx::core::contiguous(
        mlx::core::reshape(
            input,
            Shape{1, impl_->input_size}));
    const array params({limit}, mlx::core::float32);
    auto inputs = impl_->direct_weight_inputs;
    inputs.push_back(source);
    inputs.push_back(params);
    const auto workgroups =
        (static_cast<std::size_t>(impl_->output_sizes[0]) + 3) / 4;
    const auto grid = workgroups * 128;
    auto result =
        single_row_mxfp8_pair_swiglu_kernel(source.dtype())(
        inputs,
        {Shape{1, impl_->output_sizes[0]}},
        {source.dtype()},
        {
            checked_int(grid, "MXFP8 SwiGLU Metal grid"),
            1,
            1,
        },
        {128, 1, 1},
        {
            {"K", impl_->input_size},
            {"OUT", impl_->output_sizes[0]},
            {"M", 1},
        },
        std::nullopt,
        false,
        {}).front();
    return mlx::core::reshape(
        std::move(result),
        std::move(output_shape));
}

array MlxGroupedLinear::small_m_swiglu(
    const array& input,
    float limit) const {
    if (!supports_small_m_swiglu(input)) {
        throw MlxGroupedLinearUnsupported(
            "grouped MXFP8 SwiGLU requires two through six FP16/BF16 rows");
    }
    if (!std::isfinite(limit) || limit < 0.0f) {
        throw std::invalid_argument(
            "grouped SwiGLU limit must be finite and non-negative");
    }
    if (impl_->has_nint_swiglu_pair()) {
        const auto rows = input.size() /
            static_cast<std::size_t>(impl_->input_size);
        Shape output_shape(
            input.shape().begin(),
            input.shape().end() - 1);
        output_shape.push_back(impl_->output_sizes[0]);
        auto source = mlx::core::contiguous(mlx::core::reshape(
            input,
            Shape{static_cast<std::int32_t>(rows), impl_->input_size}));
        return mlx::core::reshape(
            run_nint_projection_group(source, rows, true, limit),
            std::move(output_shape));
    }
    const int rows = checked_int(
        input.size() / static_cast<std::size_t>(impl_->input_size),
        "MXFP8 SwiGLU row count");
    Shape output_shape(
        input.shape().begin(),
        input.shape().end() - 1);
    output_shape.push_back(impl_->output_sizes[0]);
    auto source = mlx::core::contiguous(mlx::core::reshape(
        input,
        Shape{rows, impl_->input_size}));
    const array params({limit}, mlx::core::float32);
    auto inputs = impl_->direct_weight_inputs;
    inputs.push_back(source);
    inputs.push_back(params);
    const auto workgroups =
        (static_cast<std::size_t>(impl_->output_sizes[0]) + 3) / 4;
    const auto grid = workgroups * 128;
    auto result = single_row_mxfp8_pair_swiglu_kernel(source.dtype())(
        inputs,
        {Shape{rows, impl_->output_sizes[0]}},
        {source.dtype()},
        {checked_int(grid, "MXFP8 SwiGLU Metal grid"), 1, 1},
        {128, 1, 1},
        {
            {"K", impl_->input_size},
            {"OUT", impl_->output_sizes[0]},
            {"M", rows},
        },
        std::nullopt,
        false,
        {}).front();
    return mlx::core::reshape(
        std::move(result),
        std::move(output_shape));
}

std::vector<array> MlxGroupedLinear::matmul(
    const array& input) const {
    if (input.ndim() == 0 ||
        input.shape(-1) != impl_->input_size) {
        throw std::invalid_argument(
            "grouped linear input must end in the shared weight width");
    }
    if (input.dtype() != mlx::core::float16 &&
        input.dtype() != mlx::core::float32 &&
        (input.dtype() != mlx::core::bfloat16 ||
         !impl_->supports_bf16_matmul())) {
        throw MlxGroupedLinearUnsupported(
            "grouped linear input dtype is unsupported by this format group");
    }

    std::size_t rows = 1;
    Shape prefix(
        input.shape().begin(),
        input.shape().end() - 1);
    for (const int value : prefix) {
        if (value <= 0 ||
            rows > static_cast<std::size_t>(max_rows()) /
                static_cast<std::size_t>(value)) {
            throw MlxGroupedLinearUnsupported(
                "grouped linear supports one through 16 input rows");
        }
        rows *= static_cast<std::size_t>(value);
    }
    if (rows == 0 ||
        rows > static_cast<std::size_t>(max_rows())) {
        throw MlxGroupedLinearUnsupported(
            "grouped linear supports one through 16 input rows");
    }

    if (impl_->has_dense_projection_group()) {
        const auto weight_dtype =
            impl_->dense_projection_weights.front().dtype();
        auto source = input.dtype() == weight_dtype
            ? input
            : mlx::core::astype(input, weight_dtype);
        source = mlx::core::contiguous(mlx::core::reshape(
            std::move(source),
            Shape{
                static_cast<std::int32_t>(rows),
                impl_->input_size,
            }));

        std::vector<array> inputs = impl_->dense_projection_weights;
        inputs.push_back(source);
        std::vector<
            std::pair<std::string, mlx::core::fast::TemplateArg>>
            templates{
                {"T", weight_dtype},
                {"M", static_cast<int>(rows)},
                {"K", impl_->input_size},
                {"TOTAL_OUT", impl_->total_output_size},
            };
        int output_offset = 0;
        int tile_offset = 0;
        for (std::size_t projection = 0;
             projection < impl_->output_sizes.size();
             ++projection) {
            const auto prefix_name =
                "P" + std::to_string(projection) + "_";
            const int output = impl_->output_sizes[projection];
            const int tiles = (output + 3) / 4;
            templates.emplace_back(prefix_name + "OUT", output);
            templates.emplace_back(
                prefix_name + "OFFSET", output_offset);
            templates.emplace_back(
                prefix_name + "TILE_BEGIN", tile_offset);
            templates.emplace_back(
                prefix_name + "TILE_END", tile_offset + tiles);
            output_offset += output;
            tile_offset += tiles;
        }
        const auto grid = static_cast<std::size_t>(tile_offset) * 128;
        if (grid > static_cast<std::size_t>(
                std::numeric_limits<int>::max())) {
            throw MlxGroupedLinearUnsupported(
                "dense grouped projection Metal grid exceeds MLX limits");
        }
        auto combined = dense_projection_group_kernel(
            impl_->dense_projection_weights.size())(
            std::move(inputs),
            {Shape{
                static_cast<std::int32_t>(rows),
                impl_->total_output_size,
            }},
            {weight_dtype},
            {static_cast<int>(grid), 1, 1},
            {128, 1, 1},
            std::move(templates),
            std::nullopt,
            false,
            {}).front();

        std::vector<array> outputs;
        outputs.reserve(impl_->output_sizes.size());
        int offset = 0;
        for (const int width : impl_->output_sizes) {
            auto shape = prefix;
            shape.push_back(width);
            outputs.push_back(mlx::core::reshape(
                mlx::core::slice(
                    combined,
                    Shape{0, offset},
                    Shape{
                        static_cast<std::int32_t>(rows),
                        offset + width,
                    }),
                std::move(shape)));
            offset += width;
        }
        return outputs;
    }

    if (impl_->has_sq_projection_group()) {
        const bool preserve_bf16 = input.dtype() == mlx::core::bfloat16;
        const auto source = preserve_bf16
            ? mlx::core::astype(input, mlx::core::float16)
            : input;
        auto outputs = impl_->has_mxfp4_sq_projection_group()
            ? MlxMxfp4SqWeight::projection_group_matmul(
                impl_->mxfp4_sq_projection_weights,
                source)
            : MlxFp8SqWeight::projection_group_matmul(
                impl_->fp8_sq_projection_weights,
                source);
        if (preserve_bf16) {
            for (auto& output : outputs) {
                output = mlx::core::astype(output, mlx::core::bfloat16);
            }
        }
        return outputs;
    }

    if (impl_->has_mxfp8_block32_projection_group()) {
        if (input.dtype() == mlx::core::float32) {
            throw MlxGroupedLinearUnsupported(
                "column-32 MXFP8 projection group requires FP16/BF16 input");
        }
        return MlxMxWeight::projection_group_matmul(
            impl_->mxfp8_block32_projection_weights,
            input);
    }

    if (!impl_->common_kernel_weights.empty()) {
        std::vector<array> outputs;
        outputs.reserve(impl_->common_kernel_weights.size());
        for (const auto& weight : impl_->common_kernel_weights) {
            outputs.push_back(std::visit(
                [&input](const auto& retained) {
                    return retained.matmul(input);
                },
                weight));
        }
        return outputs;
    }
    if (impl_->has_nint_projection_group() && rows > 6) {
        std::vector<array> outputs;
        outputs.reserve(impl_->nint_projection_weights.size());
        for (const auto& weight : impl_->nint_projection_weights) {
            outputs.push_back(weight.matmul(input));
        }
        return outputs;
    }

    const bool preserve_bf16 =
        input.dtype() == mlx::core::bfloat16;
    auto source = preserve_bf16
        ? mlx::core::astype(input, mlx::core::float16)
        : input;
    source = mlx::core::contiguous(
        mlx::core::reshape(
            std::move(source),
            Shape{
                static_cast<std::int32_t>(rows),
                impl_->input_size,
            }));
    if (impl_->has_nint_projection_group()) {
        auto combined = run_nint_projection_group(
            source,
            rows,
            false,
            0.0f);

        std::vector<array> outputs;
        outputs.reserve(impl_->output_sizes.size());
        int offset = 0;
        for (const int width : impl_->output_sizes) {
            auto shape = prefix;
            shape.push_back(width);
            outputs.push_back(
                mlx::core::reshape(
                    mlx::core::slice(
                        combined,
                        Shape{0, offset},
                        Shape{
                            static_cast<std::int32_t>(rows),
                            offset + width,
                        }),
                    std::move(shape)));
            offset += width;
        }
        return outputs;
    }
    const bool use_single_row_mxfp8_fast_path =
        rows == 1 &&
        source.dtype() == mlx::core::float16 &&
        impl_->has_single_row_mxfp8_fast_path();
    // MTP verification and small continuous batches use two through six rows.
    // Keep all rows in one threadgroup tile so every decoded packed group is
    // reused across M instead of replaying the same GEMV M times. Eight lanes
    // reduce one output; this changes only the floating-point reduction order.
    const bool use_small_m_batched_path =
        rows >= 2 && rows <= 6 &&
        impl_->uses_zero_copy_storage();
    const bool supports_small_m_blockwise =
        use_small_m_batched_path &&
        std::all_of(
            impl_->direct_layouts.begin(),
            impl_->direct_layouts.end(),
            [](const DirectProjectionLayout& layout) {
                return layout.family == kFamilyNint8Zero
                    || layout.family == kFamilyVq
                    || layout.family == kFamilyMx;
            });
    const bool small_m_has_mx = std::any_of(
        impl_->direct_layouts.begin(),
        impl_->direct_layouts.end(),
        [](const DirectProjectionLayout& layout) {
            return layout.family == kFamilyMx;
        });
    const auto* grouped_small_m_layout =
        std::getenv("MFQ_METAL_GROUPED_SMALL_M_LAYOUT");
    const bool use_small_m_blockwise =
        supports_small_m_blockwise &&
        (!small_m_has_mx || source.dtype() == mlx::core::float16) &&
        (grouped_small_m_layout == nullptr ||
         std::strcmp(grouped_small_m_layout, "scalar") != 0);
    const bool use_vectorized_fp16 =
        use_small_m_blockwise &&
        source.dtype() == mlx::core::float16;
    int group64_outputs_per_simd = 0;
    int group64_simd_groups = 8;
    if (use_vectorized_fp16 &&
        (impl_->input_size % 8) == 0 &&
        supports_small_m_group64_output_tile(impl_->direct_layouts)) {
        group64_outputs_per_simd = rows >= 5
            ? 2
            : (rows == 4 ? 5 : 8);
        if (const auto* value = std::getenv(
                "MFQ_METAL_GROUPED_GROUP64_OUTPUT_TILE")) {
            group64_outputs_per_simd = 0;
            if (value[1] == '\0' && value[0] >= '1' && value[0] <= '8') {
                group64_outputs_per_simd = value[0] - '0';
            }
        }
        if (const auto* value = std::getenv(
                "MFQ_METAL_GROUPED_GROUP64_SIMD_GROUPS")) {
            if (std::strcmp(value, "4") == 0) {
                group64_simd_groups = 4;
            } else if (std::strcmp(value, "8") == 0) {
                group64_simd_groups = 8;
            }
        }
    }
    const int specialized_outputs_per_simd = group64_outputs_per_simd;
    const int specialized_simd_groups = group64_simd_groups;
    const int small_m_outputs_per_tile =
        specialized_outputs_per_simd > 0
        ? specialized_simd_groups * specialized_outputs_per_simd
        : 8;
    int small_m_blockwise_tiles = 0;
    if (use_small_m_blockwise) {
        for (const auto& layout : impl_->direct_layouts) {
            small_m_blockwise_tiles = checked_int(
                static_cast<std::size_t>(small_m_blockwise_tiles)
                    + static_cast<std::size_t>(
                        (layout.output_size
                            + small_m_outputs_per_tile - 1)
                        / small_m_outputs_per_tile),
                "small-M blockwise tile count");
        }
    }
    const int work_tiles = use_single_row_mxfp8_fast_path
        ? impl_->single_row_mxfp8_tiles
        : (use_small_m_blockwise
            ? small_m_blockwise_tiles
            : impl_->total_tiles);
    const int grouped_threads =
        specialized_outputs_per_simd > 0
        ? specialized_simd_groups * 32
        : (use_single_row_mxfp8_fast_path
            ? 128
            : 64);
    const auto grid = (use_small_m_batched_path ? 1 : rows)
        * static_cast<std::size_t>(work_tiles)
        * static_cast<std::size_t>(grouped_threads);
    if (grid >
        static_cast<std::size_t>(
            std::numeric_limits<int>::max())) {
        throw MlxGroupedLinearUnsupported(
            "grouped linear Metal grid exceeds MLX limits");
    }

    const auto output_shapes = std::vector<Shape>{
        Shape{
            static_cast<std::int32_t>(rows),
            impl_->total_output_size,
        },
    };
    const auto output_dtypes =
        std::vector<Dtype>{source.dtype()};
    const auto grid_shape = std::tuple<int, int, int>{
        static_cast<int>(grid),
        1,
        1,
    };
    const auto threadgroup = std::tuple<int, int, int>{
        grouped_threads,
        1,
        1,
    };

    array combined = [&]() {
        if (impl_->uses_zero_copy_storage()) {
            auto inputs = impl_->direct_weight_inputs;
            inputs.push_back(source);
            if (use_single_row_mxfp8_fast_path) {
                std::vector<
                    std::pair<
                        std::string,
                        mlx::core::fast::TemplateArg>>
                    templates{
                        {"K", impl_->input_size},
                    };
                for (
                    std::size_t projection = 0;
                    projection < impl_->direct_layouts.size();
                    ++projection
                ) {
                    const auto& layout =
                        impl_->direct_layouts[projection];
                    const auto prefix =
                        "P" + std::to_string(projection) + "_";
                    templates.emplace_back(
                        prefix + "OUT",
                        layout.output_size);
                    templates.emplace_back(
                        prefix + "OUT_OFFSET",
                        layout.output_offset);
                    if (layout.family == kFamilyNint8Zero) {
                        templates.emplace_back(
                            prefix + "NG",
                            layout.groups);
                    }
                }
                return single_row_mxfp8_kernel(
                    impl_->direct_layouts)(
                    inputs,
                    output_shapes,
                    output_dtypes,
                    grid_shape,
                    threadgroup,
                    std::move(templates),
                    std::nullopt,
                    false,
                    {}).front();
            }
            std::vector<
                std::pair<
                    std::string,
                    mlx::core::fast::TemplateArg>>
                templates{
                    {"T", source.dtype()},
                    {"ROWS", static_cast<int>(rows)},
                    {"K", impl_->input_size},
                    {"TOTAL_OUT", impl_->total_output_size},
                    {"TOTAL_TILES", impl_->total_tiles},
                };
            int execution_tile_begin = 0;
            for (
                std::size_t projection = 0;
                projection < impl_->direct_layouts.size();
                ++projection
            ) {
                const auto& layout =
                    impl_->direct_layouts[projection];
                const auto prefix =
                    "P" + std::to_string(projection) + "_";
                const int execution_tile_end =
                    use_small_m_blockwise
                    ? execution_tile_begin
                        + (layout.output_size
                            + small_m_outputs_per_tile - 1)
                            / small_m_outputs_per_tile
                    : layout.tile_end;
                templates.emplace_back(
                    prefix + "OUT",
                    layout.output_size);
                templates.emplace_back(
                    prefix + "TILE_BEGIN",
                    use_small_m_blockwise
                        ? execution_tile_begin
                        : layout.tile_begin);
                templates.emplace_back(
                    prefix + "TILE_END",
                    execution_tile_end);
                execution_tile_begin = execution_tile_end;
                templates.emplace_back(
                    prefix + "OUT_OFFSET",
                    layout.output_offset);
                templates.emplace_back(
                    prefix + "NG",
                    layout.groups);
                if (layout.family == kFamilyVq) {
                    templates.emplace_back(
                        prefix + "GS",
                        layout.group_size);
                    templates.emplace_back(
                        prefix + "VECTOR_SIZE",
                        layout.vector_size);
                    templates.emplace_back(
                        prefix + "NVEC",
                        layout.vectors);
                    templates.emplace_back(
                        prefix + "INDEX_BITS",
                        layout.index_bits);
                    templates.emplace_back(
                        prefix + "STATE_BITS",
                        layout.state_bits);
                    templates.emplace_back(
                        prefix + "STATES",
                        layout.states);
                    templates.emplace_back(
                        prefix + "ENTRIES",
                        layout.entries);
                    templates.emplace_back(
                        prefix + "CODE_BANKS",
                        layout.code_banks);
                    templates.emplace_back(
                        prefix + "AUX_MODE",
                        layout.aux_mode);
                    templates.emplace_back(
                        prefix + "CODE_BANK_MODE",
                        layout.code_bank_mode);
                    templates.emplace_back(
                        prefix + "EXECUTION_LAYOUT",
                        layout.execution_layout);
                    templates.emplace_back(
                        prefix + "HAS_TABLE_BANKS",
                        static_cast<int>(
                            layout.table_banks > 1));
                    templates.emplace_back(
                        prefix + "GROUPS_PER_SUPER",
                        layout.groups_per_supergroup);
                    templates.emplace_back(
                        prefix + "NSUPER",
                        layout.supergroups);
                } else if (layout.family == kFamilyMx) {
                    templates.emplace_back(
                        prefix + "BITS",
                        layout.bits);
                }
            }
            auto kernel = direct_kernel(
                impl_->direct_layouts,
                use_small_m_batched_path,
                use_small_m_blockwise,
                use_vectorized_fp16,
                group64_outputs_per_simd,
                group64_simd_groups);
            return kernel(
                inputs,
                output_shapes,
                output_dtypes,
                grid_shape,
                threadgroup,
                std::move(templates),
                std::nullopt,
                false,
                {}).front();
        }

        return grouped_kernel()(
            {
                *impl_->descriptors,
                *impl_->projection_tile_offsets,
                *impl_->projection_output_offsets,
                *impl_->q8_q,
                *impl_->q8_scales,
                source,
            },
            output_shapes,
            output_dtypes,
            grid_shape,
            threadgroup,
            {
                {"T", source.dtype()},
                {"ROWS", static_cast<int>(rows)},
                {
                    "PROJECTIONS",
                    static_cast<int>(
                        impl_->output_sizes.size()),
                },
                {"K", impl_->input_size},
                {"TOTAL_OUT", impl_->total_output_size},
                {"TOTAL_TILES", impl_->total_tiles},
                {"DESCRIPTOR_SIZE", kDescriptorSize},
            },
            std::nullopt,
            false,
            {}).front();
    }();

    if (preserve_bf16) {
        combined = mlx::core::astype(
            std::move(combined),
            mlx::core::bfloat16);
    }

    std::vector<array> outputs;
    outputs.reserve(impl_->output_sizes.size());
    int offset = 0;
    for (const int width : impl_->output_sizes) {
        auto shape = prefix;
        shape.push_back(width);
        outputs.push_back(
            mlx::core::reshape(
                mlx::core::slice(
                    combined,
                    Shape{0, offset},
                    Shape{
                        static_cast<std::int32_t>(rows),
                        offset + width,
                    }),
                std::move(shape)));
        offset += width;
    }
    return outputs;
}

int MlxGroupedLinear::input_size() const noexcept {
    return impl_->input_size;
}

int MlxGroupedLinear::total_output_size() const noexcept {
    return impl_->total_output_size;
}

std::size_t MlxGroupedLinear::projection_count() const noexcept {
    return impl_->output_sizes.size();
}

const std::vector<int>&
MlxGroupedLinear::output_sizes() const noexcept {
    return impl_->output_sizes;
}

std::size_t MlxGroupedLinear::packed_nbytes() const noexcept {
    return impl_->packed_bytes;
}

bool MlxGroupedLinear::uses_zero_copy_storage() const noexcept {
    return impl_->uses_zero_copy_storage();
}

std::size_t
MlxGroupedLinear::copied_packed_nbytes() const noexcept {
    return impl_->copied_packed_bytes;
}

bool MlxGroupedLinear::supports_single_row_projection_fusion()
    const noexcept {
    return impl_->has_nint_projection_group() ||
        impl_->has_dense_projection_group() ||
        impl_->has_mxfp8_block32_projection_group() ||
        impl_->has_sq_projection_group() ||
        impl_->has_single_row_mxfp8_fast_path() ||
        impl_->supports_bf16_matmul();
}

bool MlxGroupedLinear::has_projection_fusion() const noexcept {
    // RetainedProjection is an exact graph composition: matmul() visits every
    // member independently. It remains useful to direct callers, but accepting
    // it as one projection-batch segment would prevent a longer list (for
    // example seven NINT projections) from being partitioned into real 3/2/2
    // fused dispatches.
    return impl_->common_kernel_weights.empty();
}

bool MlxGroupedLinear::has_single_row_mxfp8_fast_path()
    const noexcept {
    return impl_->has_single_row_mxfp8_fast_path();
}

} // namespace mfq::metal
