#include "mlx_nint.h"
#include "mlx_kernel_prepare.h"
#include "mlx_weight_residency.h"

#include "mfq/nint_blob.h"
#include "mfq_mfe_prefill_embedded.h"
#include "mlx_platform.h"
#include "mlx_staging_allocator.h"

#include <mlx/allocator.h>
#include <mlx/backend/metal/device.h>
#include <mlx/backend/metal/utils.h>
#include <mlx/primitives.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mfq::metal {
namespace {

using mlx::core::CompileOptions;
using mlx::core::Dtype;
using mlx::core::MathMode;
using mlx::core::Shape;
using mlx::core::array;

constexpr int kSubSelectorBits = 2;
constexpr int kQSelectorBits = 3;

struct DenseNintMmqParameters {
    int rows = 0;
    int output_width = 0;
    int input_width = 0;
    int group_size = 0;
    int groups = 0;
};

static_assert(sizeof(DenseNintMmqParameters) == 20);

struct DenseNintMmqConfig {
    int rows = 0;
    int output_width = 0;
    int input_width = 0;
    int group_size = 0;
    int groups = 0;
    bool use_nax = false;
};

bool dense_nint_nax_enabled() noexcept {
    return mlx_apple_chip_starts_with("Apple M5");
}

class DenseNintMmqPrimitive final : public mlx::core::UnaryPrimitive, public MlxPreparableKernel {
public:
    DenseNintMmqPrimitive(
        mlx::core::Stream stream,
        DenseNintMmqConfig config)
        : UnaryPrimitive(stream), config_(std::move(config)) {}

    std::string preparation_key() const override {
        return std::string(config_.use_nax ? "dense_nint_nax_" : "dense_nint_mmq_")
            + std::to_string(config_.group_size);
    }
    void prepare_gpu() override { (void)prepared_kernel(); }

    MTL::ComputePipelineState* prepared_kernel() {
        auto& selected_stream = stream();
        auto& device = mlx::core::metal::device(selected_stream.device);
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        auto* library = device.get_library(
            std::string(config_.use_nax
                ? "mfq_dense_metadata_nint_nax_v3_gs"
                : "mfq_dense_metadata_nint_mmq_v3_gs")
                + std::to_string(config_.group_size),
            options,
            [use_nax = config_.use_nax, group_size = config_.group_size] {
                std::string source;
                source.reserve(
                    (use_nax
                        ? sizeof(detail::kSteelNaxSource)
                        : sizeof(detail::kSteelMmaSource))
                    + sizeof(detail::kNintPrefillSource)
                    + 256);
                source += "#include <metal_stdlib>\n";
                source += "#include <metal_simdgroup>\n";
                source += "#include <metal_simdgroup_matrix>\n";
                source += "#define MFQ_NINT_PREFILL_GROUP_SIZE "
                    + std::to_string(group_size) + "\n";
                if (use_nax) {
                    source += "#include <MetalPerformancePrimitives/"
                        "MetalPerformancePrimitives.h>\n";
                    source += "#define MFQ_ENABLE_NAX 1\n";
                }
                source += "using namespace metal;\n";
                source += "using bfloat16_t = bfloat;\n";
                source += use_nax
                    ? detail::kSteelNaxSource
                    : detail::kSteelMmaSource;
                source += detail::kNintPrefillSource;
                return source;
            });
        return device.get_kernel(
            config_.use_nax
                ? "mfq_nint_prefill_nax_f16_bm128_bn64_bk96"
                : "mfq_nint_prefill_mmq_f16_bm128_bn64_bk48",
            library);
    }

    void eval_cpu(const std::vector<array>&, array&) override {
        throw std::runtime_error("dense NINT MMQ has no CPU path");
    }

    void eval_gpu(
        const std::vector<array>& inputs,
        array& output) override {
        if (inputs.size() != 5) {
            throw std::logic_error("dense NINT MMQ input count mismatch");
        }
        output.set_data(mlx::core::allocator::malloc(output.nbytes()));
        auto& selected_stream = stream();
        auto* kernel = prepared_kernel();
        auto& encoder =
            mlx::core::metal::get_command_encoder(selected_stream);
        for (int index = 0; index < 5; ++index) {
            encoder.set_input_array(inputs[static_cast<std::size_t>(index)], index);
        }
        encoder.set_output_array(output, 5);
        const DenseNintMmqParameters parameters{
            .rows = config_.rows,
            .output_width = config_.output_width,
            .input_width = config_.input_width,
            .group_size = config_.group_size,
            .groups = config_.groups,
        };
        encoder.set_bytes(parameters, 6);
        encoder.set_compute_pipeline_state(kernel);
        encoder.dispatch_threadgroups(
            MTL::Size(
                (config_.output_width + 63) / 64,
                config_.use_nax
                    ? (config_.rows + 127) / 128
                    : (config_.rows + 127) / 128,
                1),
            MTL::Size(256, 1, 1));
    }

    const char* name() const override {
        return "DenseNintMmqPrimitive";
    }

    bool is_equivalent(const mlx::core::Primitive& other) const override {
        const auto* primitive =
            dynamic_cast<const DenseNintMmqPrimitive*>(&other);
        return primitive != nullptr &&
            primitive->config_.rows == config_.rows &&
            primitive->config_.output_width == config_.output_width &&
            primitive->config_.input_width == config_.input_width &&
            primitive->config_.group_size == config_.group_size &&
            primitive->config_.groups == config_.groups &&
            primitive->config_.use_nax == config_.use_nax;
    }

    std::vector<Shape> output_shapes(const std::vector<array>&) override {
        return {Shape{config_.rows, config_.output_width}};
    }

private:
    DenseNintMmqConfig config_;
};

array dense_nint_mmq(
    const array& q,
    const array& row_metadata,
    const array& sub_scale,
    const array& sub_min,
    const array& x,
    DenseNintMmqConfig config) {
    auto stream = mlx::core::default_stream(mlx::core::default_device());
    if (stream.device != mlx::core::Device::gpu) {
        throw std::invalid_argument("dense NINT MMQ requires the Metal device");
    }
    const Shape shape{config.rows, config.output_width};
    return array(
        shape,
        mlx::core::float16,
        std::make_shared<DenseNintMmqPrimitive>(stream, std::move(config)),
        {q, row_metadata, sub_scale, sub_min, x});
}

constexpr const char* kNintHeader = R"METAL(
template <typename InputScalar>
inline float4 mfq_nint_load_input4(device const InputScalar* input, uint offset) {
    if ((offset & 3u) == 0u)
        return float4(*reinterpret_cast<device const vec<InputScalar, 4>*>(input + offset));
    return float4(input[offset], input[offset + 1u], input[offset + 2u], input[offset + 3u]);
}

template <typename InputScalar>
inline float4 mfq_nint_load_input4(constant const InputScalar* input, uint offset) {
    return float4(input[offset], input[offset + 1u], input[offset + 2u], input[offset + 3u]);
}

inline uint mfq_nint_load_u32(
    device const uchar* stream,
    uint byte_index
) {
    const packed_uchar4 bytes =
        *reinterpret_cast<device const packed_uchar4*>(stream + byte_index);
    return as_type<uint>(bytes);
}

inline uint mfq_nint_load_u32(
    constant const uchar* stream,
    uint byte_index
) {
    return uint(stream[byte_index])
        | (uint(stream[byte_index + 1u]) << 8u)
        | (uint(stream[byte_index + 2u]) << 16u)
        | (uint(stream[byte_index + 3u]) << 24u);
}

template <typename Stream>
inline uint mfq_nint_read_row_value(
    Stream stream,
    uint row_byte_offset,
    uint row_bit_shift,
    uint value_index,
    uint bits
) {
    uint row_relative_bits = row_bit_shift + value_index * bits;
    uint byte_index = row_byte_offset + (row_relative_bits >> 3u);
    uint shift = row_relative_bits & 7u;
    uint packed = uint(stream[byte_index]);
    if (shift + bits > 8u) {
        packed |= uint(stream[byte_index + 1ul]) << 8u;
    }
    return (packed >> shift) & ((1u << bits) - 1u);
}

template <typename Stream>
inline uint4 mfq_nint_read_row_value4(
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
    uint packed = mfq_nint_load_u32(stream, byte_index);
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

template <typename Stream>
inline uint4 mfq_nint_decode_value4_at(
    Stream stream,
    uint byte_index,
    uint shift,
    uint bits
) {
    uint required_bits = shift + 4u * bits;
    uint packed = mfq_nint_load_u32(stream, byte_index);
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

struct MfqNintValue8 {
    uint4 low;
    uint4 high;
};

template <uint GROUP_SIZE, typename Stream>
inline MfqNintValue8 mfq_nint_decode_value8_at(
    Stream stream,
    uint byte_index,
    uint shift,
    uint bits
) {
    const uint word0 = mfq_nint_load_u32(stream, byte_index);
    uint packed0, packed1;
    if constexpr (GROUP_SIZE % 4u == 0u) {
        if constexpr (GROUP_SIZE % 8u == 0u) shift = 0u;
        const uint word1 = bits > 4u
            ? mfq_nint_load_u32(stream, byte_index + 4u) : 0u;
        packed0 = shift == 0u ? word0 : (word0 >> 4u) | (word1 << 28u);
        const uint bit = 4u * bits + shift;
        packed1 = bit == 32u ? word1 : (word0 >> bit) | (word1 << (32u - bit));
    } else {
        const uint word1 = mfq_nint_load_u32(stream, byte_index + 4u);
        const uint word2 = shift + 8u * bits > 64u
            ? mfq_nint_load_u32(stream, byte_index + 8u) : 0u;
        packed0 = shift == 0u ? word0
            : (word0 >> shift) | (word1 << (32u - shift));
        const uint raw_second_cursor = shift + 4u * bits;
        const bool second_word = raw_second_cursor >= 32u;
        const uint second_cursor = raw_second_cursor & 31u;
        const uint second_low = second_word ? word1 : word0;
        const uint second_high = second_word ? word2 : word1;
        packed1 = second_cursor == 0u ? second_low
            : (second_low >> second_cursor) | (second_high << (32u - second_cursor));
    }
    const uint mask = (1u << bits) - 1u;
    return {
        uint4(
            packed0 & mask,
            (packed0 >> bits) & mask,
            (packed0 >> (2u * bits)) & mask,
            (packed0 >> (3u * bits)) & mask),
        uint4(
            packed1 & mask,
            (packed1 >> bits) & mask,
            (packed1 >> (2u * bits)) & mask,
            (packed1 >> (3u * bits)) & mask),
    };
}

template <typename Stream>
inline uint mfq_nint_read_bits32(
    Stream stream,
    uint row_byte_offset,
    uint row_bit_index
) {
    const uint byte_index = row_byte_offset + (row_bit_index >> 3u);
    const uint shift = row_bit_index & 7u;
    uint packed = mfq_nint_load_u32(stream, byte_index);
    if (shift != 0u) {
        packed = (packed >> shift)
            | (uint(stream[byte_index + 4u]) << (32u - shift));
    }
    return packed;
}

)METAL";

constexpr const char* kNintStoreOutput = R"METAL(
#define MFQ_NINT_STORE_OUTPUT(index, value) y[(index)] = T(value)
)METAL";

constexpr const char* kNintStoreOutputAdd = R"METAL(
#define MFQ_NINT_STORE_OUTPUT(index, value) \
    y[(index)] = T((value) + float(residual[(index)]))
)METAL";

constexpr const char* kNintStoreOutputMoeShared = R"METAL(
#define MFQ_NINT_STORE_OUTPUT(index, projected_value) do { \
    float routed_accumulator = 0.0f; \
    for (uint selected_route = 0u; \
         selected_route < uint(MOE_ROUTES); \
         ++selected_route) { \
        const uint routed_index = \
            (row * uint(MOE_ROUTES) + selected_route) \
                * uint(LOGICAL_OUT) + output; \
        routed_accumulator += float(pair_output[routed_index]) \
            * route_weights[row * uint(MOE_ROUTES) + selected_route]; \
    } \
    const T rounded_routed = T(routed_accumulator); \
    const T rounded_shared = T(projected_value); \
    const float shared_scale = \
        1.0f / (1.0f + exp(-float(gate_logits[row]))); \
    y[(index)] = T( \
        float(rounded_routed) \
        + shared_scale * float(rounded_shared)); \
} while (false)
)METAL";

constexpr const char* kNintMatmul = R"METAL(
#ifndef SMALL_M
#define SMALL_M (M <= 5)
#endif
    constexpr uint SIMD_GROUPS = 8u;
    constexpr uint OUTPUTS_PER_SIMD = uint(OPS_PER_SIMD);
    constexpr uint OUTPUTS_PER_TG = SIMD_GROUPS * OUTPUTS_PER_SIMD;
    constexpr uint BLOCKS8 = SMALL_M != 0 ? uint(GS) / 8u : 0u;
    constexpr uint TAIL_START = BLOCKS8 * 8u;
    constexpr uint TAIL_CHUNKS =
        (uint(GS) - TAIL_START + 3u) / 4u;

    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint output_base =
        threadgroup_position_in_grid.x * OUTPUTS_PER_TG
        + simd_group * OUTPUTS_PER_SIMD;
    uint first_row = threadgroup_position_in_grid.y * uint(TILE_M);
    if (output_base >= uint(LOGICAL_OUT) || first_row >= uint(M)) {
        return;
    }

    int local_expert = 0;
    bool route_valid = true;
    if (uint(ROUTED) != 0u) {
        int expert = expert_ids[first_row];
        route_valid = expert >= 0 && expert < int(EXPERT_MAP_SIZE);
        local_expert = route_valid ? expert_map[expert] : -1;
        route_valid = route_valid && local_expert >= 0
            && local_expert < int(LOCAL_EXPERTS);
    }

    uint outputs[OUTPUTS_PER_SIMD];
    uint metadata_bases[OUTPUTS_PER_SIMD];
    uint q_widths[OUTPUTS_PER_SIMD];
    uint q_row_byte_offsets[OUTPUTS_PER_SIMD];
    uint q_row_bit_shifts[OUTPUTS_PER_SIMD];
    float neuron_scales[OUTPUTS_PER_SIMD];
    float neuron_minimums[OUTPUTS_PER_SIMD];
    float accumulators[OUTPUTS_PER_SIMD][TILE_M];
#pragma unroll
    for (uint output_row = 0u;
         output_row < OUTPUTS_PER_SIMD;
         ++output_row) {
        uint logical_output = min(
            output_base + output_row,
            uint(LOGICAL_OUT) - 1u);
        uint output = uint(ROUTED) != 0u
            ? (route_valid
                ? uint(local_expert) * uint(OUT_PER_EXPERT)
                    + logical_output
                : 0u)
            : logical_output;
        output = min(output, uint(OUT) - 1u);
        outputs[output_row] = output;
        metadata_bases[output_row] = output * uint(NG);
        const uint row_metadata_base = output * 4u;
        uint row_layout = row_metadata[row_metadata_base];
        q_widths[output_row] = row_layout & 15u;
        q_row_byte_offsets[output_row] =
            row_metadata[row_metadata_base + 1u];
        q_row_bit_shifts[output_row] = row_layout >> 4u;
        neuron_scales[output_row] =
            as_type<float>(row_metadata[row_metadata_base + 2u]);
        neuron_minimums[output_row] =
            as_type<float>(row_metadata[row_metadata_base + 3u]);
#pragma unroll
        for (uint local_row = 0u; local_row < uint(TILE_M); ++local_row) {
            accumulators[output_row][local_row] = 0.0f;
        }
    }

    if (route_valid) {
        for (uint group = lane; group < uint(NG); group += 32u) {
        float activation_sums[TILE_M];
        float quantized_dots[OUTPUTS_PER_SIMD][TILE_M];
        uint byte_cursors[OUTPUTS_PER_SIMD];
        uint bit_cursors[OUTPUTS_PER_SIMD];
#pragma unroll
        for (uint local_row = 0u; local_row < uint(TILE_M); ++local_row) {
            activation_sums[local_row] = 0.0f;
#pragma unroll
            for (uint output_row = 0u;
                 output_row < OUTPUTS_PER_SIMD;
                 ++output_row) {
                quantized_dots[output_row][local_row] = 0.0f;
            }
        }
#pragma unroll
        for (uint output_row = 0u;
             output_row < OUTPUTS_PER_SIMD;
             ++output_row) {
            const uint group_bit = q_row_bit_shifts[output_row]
                + group * uint(GS) * q_widths[output_row];
            byte_cursors[output_row] = q_row_byte_offsets[output_row]
                + (group_bit >> 3u);
            bit_cursors[output_row] = group_bit & 7u;
        }
#pragma unroll
        for (uint block = 0u; block < BLOCKS8; ++block) {
            const uint group_element = block * 8u;
            const uint column = group * uint(GS) + group_element;
            MfqNintValue8 codes[OUTPUTS_PER_SIMD];
#pragma unroll
            for (uint output_row = 0u;
                 output_row < OUTPUTS_PER_SIMD;
                 ++output_row) {
                const uint bits = q_widths[output_row];
                codes[output_row] = mfq_nint_decode_value8_at<uint(GS)>(
                    q_packed,
                    byte_cursors[output_row],
                    bit_cursors[output_row],
                    bits);
                byte_cursors[output_row] += bits;
            }
#pragma unroll
            for (uint local_row = 0u;
                 local_row < uint(TILE_M);
                 ++local_row) {
                const uint row = first_row + local_row;
                const uint input_row = uint(ROUTED) != 0u
                    && uint(SHARED_INPUT) != 0u
                    ? row / uint(ROUTES)
                    : row;
                float4 activation0 = float4(0.0f);
                float4 activation1 = float4(0.0f);
                if (row < uint(M)) {
                    const uint input_base = input_row * uint(K) + column;
                    if (column + 7u < uint(K)) {
                        activation0 = mfq_nint_load_input4(x, input_base);
                        activation1 = mfq_nint_load_input4(x, input_base + 4u);
                    } else {
#pragma unroll
                        for (uint element = 0u; element < 4u; ++element) {
                            activation0[element] = column + element < uint(K)
                                ? float(x[input_base + element]) : 0.0f;
                            activation1[element] = column + 4u + element < uint(K)
                                ? float(x[input_base + 4u + element]) : 0.0f;
                        }
                    }
                }
                activation_sums[local_row] +=
                    activation0.x + activation0.y
                    + activation0.z + activation0.w
                    + activation1.x + activation1.y
                    + activation1.z + activation1.w;
#pragma unroll
                for (uint output_row = 0u;
                     output_row < OUTPUTS_PER_SIMD;
                     ++output_row) {
                    quantized_dots[output_row][local_row] +=
                        dot(activation0, float4(codes[output_row].low))
                        + dot(activation1, float4(codes[output_row].high));
                }
            }
        }
#pragma unroll
        for (uint tail = 0u; tail < TAIL_CHUNKS; ++tail) {
            const uint group_element = TAIL_START + tail * 4u;
            const uint column = group * uint(GS) + group_element;
            uint4 codes[OUTPUTS_PER_SIMD];
#pragma unroll
            for (uint output_row = 0u;
                 output_row < OUTPUTS_PER_SIMD;
                 ++output_row) {
                const uint bits = q_widths[output_row];
                codes[output_row] = mfq_nint_decode_value4_at(
                    q_packed,
                    byte_cursors[output_row],
                    bit_cursors[output_row],
                    bits);
                const uint next_bit_cursor =
                    bit_cursors[output_row] + 4u * bits;
                byte_cursors[output_row] += next_bit_cursor >> 3u;
                bit_cursors[output_row] = next_bit_cursor & 7u;
            }
#pragma unroll
            for (uint local_row = 0u;
                 local_row < uint(TILE_M);
                 ++local_row) {
                const uint row = first_row + local_row;
                const uint input_row = uint(ROUTED) != 0u
                    && uint(SHARED_INPUT) != 0u
                    ? row / uint(ROUTES)
                    : row;
                float4 activation = float4(0.0f);
                if (row < uint(M)) {
                    const uint input_base = input_row * uint(K) + column;
                    if (group_element + 3u < uint(GS) &&
                        column + 3u < uint(K)) {
                        activation = mfq_nint_load_input4(x, input_base);
                    } else {
#pragma unroll
                        for (uint element = 0u; element < 4u; ++element) {
                            activation[element] =
                                group_element + element < uint(GS) &&
                                    column + element < uint(K)
                                ? float(x[input_base + element]) : 0.0f;
                        }
                    }
                }
                activation_sums[local_row] +=
                    activation.x + activation.y
                    + activation.z + activation.w;
#pragma unroll
                for (uint output_row = 0u;
                     output_row < OUTPUTS_PER_SIMD;
                     ++output_row) {
                    quantized_dots[output_row][local_row] +=
                        dot(activation, float4(codes[output_row]));
                }
            }
        }
#pragma unroll
        for (uint output_row = 0u;
             output_row < OUTPUTS_PER_SIMD;
             ++output_row) {
            const uint metadata_index =
                metadata_bases[output_row] + group;
            const float scale = neuron_scales[output_row]
                * float(sub_scale[metadata_index]);
            const float minimum = neuron_minimums[output_row]
                * float(sub_min[metadata_index]);
#pragma unroll
            for (uint local_row = 0u;
                 local_row < uint(TILE_M);
                 ++local_row) {
                accumulators[output_row][local_row] = fma(
                    scale,
                    quantized_dots[output_row][local_row],
                    fma(
                        -minimum,
                        activation_sums[local_row],
                        accumulators[output_row][local_row]));
            }
        }
        }
    }

#pragma unroll
    for (uint output_row = 0u;
         output_row < OUTPUTS_PER_SIMD;
         ++output_row) {
        uint output = output_base + output_row;
#pragma unroll
        for (uint local_row = 0u; local_row < uint(TILE_M); ++local_row) {
            float total = simd_sum(accumulators[output_row][local_row]);
            uint row = first_row + local_row;
            if (lane == 0u && output < uint(LOGICAL_OUT) && row < uint(M)) {
                const uint output_index =
                    row * uint(LOGICAL_OUT) + output;
                MFQ_NINT_STORE_OUTPUT(
                    output_index,
                    route_valid ? total : 0.0f);
            }
        }
    }
)METAL";


// The fused epilogue is a second operator over the same metadata-driven
// decoder, not a q-width, shape, or model-specific matmul path. Gate and up
// may use different per-neuron q/k assignments as long as their logical
// matrix geometry agrees.
constexpr const char* kNintSwiGlu = R"METAL(
    constexpr uint SIMD_GROUPS = 8u;
    constexpr uint OUTPUTS_PER_SIMD = 2u;
    constexpr uint OUTPUTS_PER_TG = SIMD_GROUPS * OUTPUTS_PER_SIMD;

    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint output_base =
        threadgroup_position_in_grid.x * OUTPUTS_PER_TG
        + simd_group * OUTPUTS_PER_SIMD;
    if (output_base >= uint(OUT)) {
        return;
    }

    uint outputs[OUTPUTS_PER_SIMD];
    uint metadata_bases[OUTPUTS_PER_SIMD];
    uint gate_widths[OUTPUTS_PER_SIMD];
    uint gate_byte_offsets[OUTPUTS_PER_SIMD];
    uint gate_bit_shifts[OUTPUTS_PER_SIMD];
    uint up_widths[OUTPUTS_PER_SIMD];
    uint up_byte_offsets[OUTPUTS_PER_SIMD];
    uint up_bit_shifts[OUTPUTS_PER_SIMD];
    float gate_neuron_scales[OUTPUTS_PER_SIMD];
    float gate_neuron_minimums[OUTPUTS_PER_SIMD];
    float up_neuron_scales[OUTPUTS_PER_SIMD];
    float up_neuron_minimums[OUTPUTS_PER_SIMD];
    float gate_accumulators[OUTPUTS_PER_SIMD] = {0.0f};
    float up_accumulators[OUTPUTS_PER_SIMD] = {0.0f};

#pragma unroll
    for (uint output_row = 0u;
         output_row < OUTPUTS_PER_SIMD;
         ++output_row) {
        uint output = min(output_base + output_row, uint(OUT) - 1u);
        outputs[output_row] = output;
        metadata_bases[output_row] = output * uint(NG);

        const uint row_metadata_base = output * 4u;
        uint gate_layout = gate_row_metadata[row_metadata_base];
        gate_widths[output_row] = gate_layout & 15u;
        gate_byte_offsets[output_row] =
            gate_row_metadata[row_metadata_base + 1u];
        gate_bit_shifts[output_row] = gate_layout >> 4u;
        gate_neuron_scales[output_row] =
            as_type<float>(gate_row_metadata[row_metadata_base + 2u]);
        gate_neuron_minimums[output_row] =
            as_type<float>(gate_row_metadata[row_metadata_base + 3u]);

        uint up_layout = up_row_metadata[row_metadata_base];
        up_widths[output_row] = up_layout & 15u;
        up_byte_offsets[output_row] =
            up_row_metadata[row_metadata_base + 1u];
        up_bit_shifts[output_row] = up_layout >> 4u;
        up_neuron_scales[output_row] =
            as_type<float>(up_row_metadata[row_metadata_base + 2u]);
        up_neuron_minimums[output_row] =
            as_type<float>(up_row_metadata[row_metadata_base + 3u]);
    }

    constexpr uint BLOCKS8 = uint(GS) / 8u;
    constexpr uint TAIL_START = BLOCKS8 * 8u;
    constexpr uint TAIL_CHUNKS =
        (uint(GS) - TAIL_START + 3u) / 4u;
    for (uint group = lane; group < uint(NG); group += 32u) {
        float activation_sum = 0.0f;
        float gate_quantized_dots[OUTPUTS_PER_SIMD] = {0.0f};
        float up_quantized_dots[OUTPUTS_PER_SIMD] = {0.0f};
        uint gate_byte_cursors[OUTPUTS_PER_SIMD];
        uint gate_bit_cursors[OUTPUTS_PER_SIMD];
        uint up_byte_cursors[OUTPUTS_PER_SIMD];
        uint up_bit_cursors[OUTPUTS_PER_SIMD];
#pragma unroll
        for (uint output_row = 0u;
             output_row < OUTPUTS_PER_SIMD;
             ++output_row) {
            const uint gate_group_bit = gate_bit_shifts[output_row]
                + group * uint(GS) * gate_widths[output_row];
            const uint up_group_bit = up_bit_shifts[output_row]
                + group * uint(GS) * up_widths[output_row];
            gate_byte_cursors[output_row] = gate_byte_offsets[output_row]
                + (gate_group_bit >> 3u);
            gate_bit_cursors[output_row] = gate_group_bit & 7u;
            up_byte_cursors[output_row] = up_byte_offsets[output_row]
                + (up_group_bit >> 3u);
            up_bit_cursors[output_row] = up_group_bit & 7u;
        }

#pragma unroll
        for (uint block = 0u; block < BLOCKS8; ++block) {
            uint group_element = block * 8u;
            uint column = group * uint(GS) + group_element;
            float4 activation0 = float4(0.0f);
            float4 activation1 = float4(0.0f);
            if (column + 7u < uint(K)) {
                activation0 = float4(
                    *reinterpret_cast<device const vec<T, 4>*>(x + column));
                activation1 = float4(
                    *reinterpret_cast<device const vec<T, 4>*>(x + column + 4u));
            } else {
#pragma unroll
                for (uint element = 0u; element < 4u; ++element) {
                    activation0[element] = column + element < uint(K)
                        ? float(x[column + element]) : 0.0f;
                    activation1[element] = column + 4u + element < uint(K)
                        ? float(x[column + 4u + element]) : 0.0f;
                }
            }
            activation_sum +=
                activation0.x + activation0.y
                + activation0.z + activation0.w
                + activation1.x + activation1.y
                + activation1.z + activation1.w;
#pragma unroll
            for (uint output_row = 0u;
                 output_row < OUTPUTS_PER_SIMD;
                 ++output_row) {
                const MfqNintValue8 gate_codes = mfq_nint_decode_value8_at<uint(GS)>(
                    gate_q,
                    gate_byte_cursors[output_row],
                    gate_bit_cursors[output_row],
                    gate_widths[output_row]);
                gate_quantized_dots[output_row] +=
                    dot(activation0, float4(gate_codes.low))
                    + dot(activation1, float4(gate_codes.high));
                const MfqNintValue8 up_codes = mfq_nint_decode_value8_at<uint(GS)>(
                    up_q,
                    up_byte_cursors[output_row],
                    up_bit_cursors[output_row],
                    up_widths[output_row]);
                up_quantized_dots[output_row] +=
                    dot(activation0, float4(up_codes.low))
                    + dot(activation1, float4(up_codes.high));
                gate_byte_cursors[output_row] += gate_widths[output_row];
                up_byte_cursors[output_row] += up_widths[output_row];
            }
        }
#pragma unroll
        for (uint tail = 0u; tail < TAIL_CHUNKS; ++tail) {
            const uint group_element = TAIL_START + tail * 4u;
            const uint column = group * uint(GS) + group_element;
            float4 activation = float4(0.0f);
            if (group_element + 3u < uint(GS) && column + 3u < uint(K)) {
                activation = float4(
                    *reinterpret_cast<device const vec<T, 4>*>(x + column));
            } else {
#pragma unroll
                for (uint element = 0u; element < 4u; ++element) {
                    activation[element] =
                        group_element + element < uint(GS) &&
                            column + element < uint(K)
                        ? float(x[column + element]) : 0.0f;
                }
            }
            activation_sum +=
                activation.x + activation.y + activation.z + activation.w;
#pragma unroll
            for (uint output_row = 0u;
                 output_row < OUTPUTS_PER_SIMD;
                 ++output_row) {
                const uint4 gate_codes = mfq_nint_decode_value4_at(
                    gate_q,
                    gate_byte_cursors[output_row],
                    gate_bit_cursors[output_row],
                    gate_widths[output_row]);
                gate_quantized_dots[output_row] +=
                    dot(activation, float4(gate_codes));
                const uint4 up_codes = mfq_nint_decode_value4_at(
                    up_q,
                    up_byte_cursors[output_row],
                    up_bit_cursors[output_row],
                    up_widths[output_row]);
                up_quantized_dots[output_row] +=
                    dot(activation, float4(up_codes));
                const uint next_gate_bit = gate_bit_cursors[output_row]
                    + 4u * gate_widths[output_row];
                const uint next_up_bit = up_bit_cursors[output_row]
                    + 4u * up_widths[output_row];
                gate_byte_cursors[output_row] += next_gate_bit >> 3u;
                gate_bit_cursors[output_row] = next_gate_bit & 7u;
                up_byte_cursors[output_row] += next_up_bit >> 3u;
                up_bit_cursors[output_row] = next_up_bit & 7u;
            }
        }

#pragma unroll
        for (uint output_row = 0u;
             output_row < OUTPUTS_PER_SIMD;
             ++output_row) {
            uint metadata_index = metadata_bases[output_row] + group;
            float gate_scale = gate_neuron_scales[output_row]
                * float(gate_sub_scale[metadata_index]);
            float gate_minimum = gate_neuron_minimums[output_row]
                * float(gate_sub_min[metadata_index]);
            float up_scale = up_neuron_scales[output_row]
                * float(up_sub_scale[metadata_index]);
            float up_minimum = up_neuron_minimums[output_row]
                * float(up_sub_min[metadata_index]);
            gate_accumulators[output_row] = fma(
                gate_scale,
                gate_quantized_dots[output_row],
                fma(
                    -gate_minimum,
                    activation_sum,
                    gate_accumulators[output_row]));
            up_accumulators[output_row] = fma(
                up_scale,
                up_quantized_dots[output_row],
                fma(
                    -up_minimum,
                    activation_sum,
                    up_accumulators[output_row]));
        }
    }

#pragma unroll
    for (uint output_row = 0u;
         output_row < OUTPUTS_PER_SIMD;
         ++output_row) {
        float gate = simd_sum(gate_accumulators[output_row]);
        float up = simd_sum(up_accumulators[output_row]);
        uint output = output_base + output_row;
        if (lane == 0u && output < uint(OUT)) {
            y[output] = T((gate / (1.0f + exp(-gate))) * up);
        }
    }
)METAL";

constexpr const char* kNintDequantize = R"METAL(
    constexpr uint ITEMS_PER_ROW = (uint(K) + uint(VEC) - 1u) / uint(VEC);
    uint item = thread_position_in_grid.x;
    if (item >= uint(OUT) * ITEMS_PER_ROW) {
        return;
    }
    uint output = item / ITEMS_PER_ROW;
    uint input_index = (item - output * ITEMS_PER_ROW) * uint(VEC);
    uint group = input_index / uint(GS);
    uint metadata_index = output * uint(NG) + group;
    float scale = neuron_scale[output] * float(sub_scale[metadata_index]);
    float minimum = neuron_min[output] * float(sub_min[metadata_index]);
    uint layout = uint(row_q_layout[output]);
    if constexpr (VEC == 4) {
        uint4 quantized = mfq_nint_read_row_value4(
            q_packed, row_q_byte_offsets[output], layout >> 4u,
            input_index, layout & 15u);
        float4 decoded = scale * float4(quantized) - minimum;
#pragma clang loop unroll(full)
        for (uint value = 0u; value < 4u; ++value) {
            if (input_index + value < uint(K))
                y[output * uint(K) + input_index + value] = T(decoded[value]);
        }
    } else {
        uint quantized = mfq_nint_read_row_value(
            q_packed, row_q_byte_offsets[output], layout >> 4u,
            input_index, layout & 15u);
        y[output * uint(K) + input_index] = T(scale * float(quantized) - minimum);
    }
)METAL";

constexpr const char* kNintEmbedding = R"METAL(
    const uint COUNT = uint(token_ids_shape[0]);
    uint output_index = thread_position_in_grid.x;
    if (output_index >= uint(COUNT) * uint(K)) {
        return;
    }
    uint token_position = output_index / uint(K);
    uint input_index = output_index - token_position * uint(K);
    uint output = uint(token_ids[token_position]);
    uint group = input_index / uint(GS);
    uint element = input_index - group * uint(GS);
    uint metadata_index = output * uint(NG) + group;
    uint quantized = mfq_nint_read_row_value(
        q_packed,
        row_q_byte_offsets[output],
        uint(row_q_layout[output]) >> 4u,
        group * uint(GS) + element,
        uint(row_q_layout[output]) & 15u);
    float scale = neuron_scale[output] * float(sub_scale[metadata_index]);
    float minimum = neuron_min[output] * float(sub_min[metadata_index]);
    y[output_index] = T(scale * float(quantized) - minimum);
)METAL";

class BlobCursor {
public:
    explicit BlobCursor(std::span<const std::uint8_t> blob)
        : blob_(blob) {}

    template <typename T>
    T scalar(const char* name) {
        require(sizeof(T), name);
        T value{};
        std::memcpy(&value, blob_.data() + offset_, sizeof(T));
        offset_ += sizeof(T);
        return value;
    }

    detail::StagingVector<std::uint8_t> bytes(
        std::size_t count,
        const char* name,
        std::size_t extra_capacity = 0) {
        require(count, name);
        if (extra_capacity > std::numeric_limits<std::size_t>::max() - count) {
            throw std::runtime_error("NINT stream capacity overflows");
        }
        detail::StagingVector<std::uint8_t> result;
        result.reserve(count + extra_capacity);
        result.insert(result.end(),
            blob_.begin() + static_cast<std::ptrdiff_t>(offset_),
            blob_.begin() + static_cast<std::ptrdiff_t>(offset_ + count));
        offset_ += count;
        return result;
    }

    std::size_t remaining() const noexcept {
        return blob_.size() - offset_;
    }

private:
    void require(std::size_t count, const char* name) {
        if (count > blob_.size() - offset_) {
            throw std::runtime_error(std::string("truncated NINT ") + name);
        }
    }

    std::span<const std::uint8_t> blob_;
    std::size_t offset_ = 0;
};

std::size_t packed_size(std::size_t count, int bits) {
    if (bits <= 0 || bits > 8 ||
        count > (std::numeric_limits<std::size_t>::max() - 7) /
            static_cast<std::size_t>(bits)) {
        throw std::runtime_error("invalid NINT packed bit count");
    }
    return (count * static_cast<std::size_t>(bits) + 7) / 8;
}

std::uint8_t packed_value(
    std::span<const std::uint8_t> data,
    std::size_t index,
    int bits) {
    const auto bit_index = index * static_cast<std::size_t>(bits);
    const auto byte_index = bit_index / 8;
    const auto shift = static_cast<unsigned>(bit_index & 7);
    std::uint32_t value = data[byte_index];
    if (shift + static_cast<unsigned>(bits) > 8) {
        value |= static_cast<std::uint32_t>(data[byte_index + 1]) << 8;
    }
    return static_cast<std::uint8_t>(
        (value >> shift) & ((1u << bits) - 1u));
}

detail::StagingVector<std::uint8_t> unpack_values(
    std::span<const std::uint8_t> data,
    std::size_t count,
    int bits) {
    detail::StagingVector<std::uint8_t> result(count);
    for (std::size_t index = 0; index < count; ++index) {
        result[index] = packed_value(data, index, bits);
    }
    return result;
}

detail::StagingVector<std::uint8_t> pack_values(
    std::span<const std::uint8_t> values,
    int bits,
    std::size_t extra_capacity = 0) {
    const auto bytes = packed_size(values.size(), bits);
    detail::StagingVector<std::uint8_t> result;
    result.reserve(bytes + extra_capacity);
    result.resize(bytes, 0);
    for (std::size_t index = 0; index < values.size(); ++index) {
        const auto bit_index = index * static_cast<std::size_t>(bits);
        for (int bit = 0; bit < bits; ++bit) {
            if ((values[index] >> bit) & 1u) {
                const auto target = bit_index + static_cast<std::size_t>(bit);
                result[target / 8] |=
                    static_cast<std::uint8_t>(1u << (target & 7));
            }
        }
    }
    return result;
}

void copy_packed_bit_range(
    std::span<const std::uint8_t> source,
    std::size_t source_bit,
    std::size_t bit_count,
    std::span<std::uint8_t> destination) {
    const auto byte_count = packed_size(bit_count, 1);
    if (destination.size() != byte_count) {
        throw std::logic_error("NINT aligned row destination size mismatch");
    }
    const auto source_byte = source_bit / 8;
    const auto shift = static_cast<unsigned>(source_bit & 7u);
    for (std::size_t byte = 0; byte < byte_count; ++byte) {
        const auto index = source_byte + byte;
        std::uint16_t packed = source[index];
        if (shift != 0 && index + 1 < source.size()) {
            packed |= static_cast<std::uint16_t>(source[index + 1]) << 8u;
        }
        destination[byte] = static_cast<std::uint8_t>(packed >> shift);
    }
    if ((bit_count & 7u) != 0) {
        destination.back() &= static_cast<std::uint8_t>(
            (1u << (bit_count & 7u)) - 1u);
    }
}

detail::StagingVector<std::uint8_t> read_old_values(
    BlobCursor& cursor,
    std::size_t count,
    int storage_bytes,
    const char* name) {
    detail::StagingVector<std::uint8_t> result(count);
    for (std::size_t index = 0; index < count; ++index) {
        const std::uint32_t value = storage_bytes == 1
            ? cursor.scalar<std::uint8_t>(name)
            : cursor.scalar<std::uint16_t>(name);
        if (value > 255) {
            throw std::runtime_error(
                std::string("NINT ") + name + " exceeds uint8 storage");
        }
        result[index] = static_cast<std::uint8_t>(value);
    }
    return result;
}

float half_to_float(std::uint16_t bits) {
    const bool negative = (bits & 0x8000u) != 0;
    const auto exponent = static_cast<unsigned>((bits >> 10) & 0x1fu);
    const auto mantissa = static_cast<unsigned>(bits & 0x03ffu);
    float value = 0.0f;
    if (exponent == 0) {
        value = std::ldexp(static_cast<float>(mantissa), -24);
    } else if (exponent == 31) {
        value = mantissa == 0
            ? std::numeric_limits<float>::infinity()
            : std::numeric_limits<float>::quiet_NaN();
    } else {
        value = std::ldexp(
            1.0f + static_cast<float>(mantissa) / 1024.0f,
            static_cast<int>(exponent) - 15);
    }
    return negative ? -value : value;
}

template <typename T, typename Allocator>
array make_array(
    const std::vector<T, Allocator>& values,
    Shape shape) {
    auto result = array(values.begin(), std::move(shape));
    MlxWeightResidency::track(result);
    return result;
}

std::int32_t checked_shape(std::int64_t value, const char* name) {
    if (value <= 0 ||
        value > std::numeric_limits<std::int32_t>::max()) {
        throw std::runtime_error(std::string("invalid NINT ") + name);
    }
    return static_cast<std::int32_t>(value);
}

struct NintMatmulKernelConfig {
    bool float32 = false;
    bool add_residual = false;
    bool moe_shared = false;
    int moe_routes = 0;
    int group_size = 0;
    int groups = 0;
    int input_size = 0;
    int output_size = 0;
    int rows = 0;
    int tile_rows = 0;
    int outputs_per_simd = 0;
    int routed = 0;
    int routes = 0;
    int shared_input = 0;
    int expert_map_size = 0;
    int local_experts = 0;
    int out_per_expert = 0;
    int logical_output = 0;
    std::array<Dtype, 3> epilogue_types{
        mlx::core::float16, mlx::core::float32, mlx::core::float16};

    bool operator==(const NintMatmulKernelConfig&) const = default;
};

std::string nint_matmul_kernel_key(const NintMatmulKernelConfig& config) {
    return std::string(config.float32 ? "f32" : "f16")
        + (config.add_residual ? "_add" : "_plain")
        + (config.moe_shared
               ? "_moe" + std::to_string(config.moe_routes)
               : "")
        + "_" + std::to_string(config.group_size)
        + "_" + std::to_string(config.groups)
        + "_" + std::to_string(config.input_size)
        + "_" + std::to_string(config.output_size)
        + "_" + std::to_string(config.rows)
        + "_" + std::to_string(config.tile_rows)
        + "_" + std::to_string(config.outputs_per_simd)
        + "_" + std::to_string(config.routed)
        + "_" + std::to_string(config.routes)
        + "_" + std::to_string(config.shared_input)
        + "_" + std::to_string(config.expert_map_size)
        + "_" + std::to_string(config.local_experts)
        + "_" + std::to_string(config.out_per_expert)
        + "_" + std::to_string(config.logical_output);
}

std::string nint_matmul_kernel_header(
    const NintMatmulKernelConfig& config) {
    return std::string("#define T ")
        + (config.float32 ? "float\n" : "half\n")
        + "#define GS " + std::to_string(config.group_size) + "\n"
        + "#define NG " + std::to_string(config.groups) + "\n"
        + "#define K " + std::to_string(config.input_size) + "\n"
        + "#define OUT " + std::to_string(config.output_size) + "\n"
        + "#define M " + (config.rows == 0 ? "row_count" : std::to_string(config.rows)) + "\n"
        + "#define SMALL_M " + std::to_string(config.rows > 0 && config.rows <= 5) + "\n"
        + "#define TILE_M " + std::to_string(config.tile_rows) + "\n"
        + "#define OPS_PER_SIMD "
        + std::to_string(config.outputs_per_simd) + "\n"
        + "#define ROUTED " + std::to_string(config.routed) + "\n"
        + "#define ROUTES " + std::to_string(config.routes) + "\n"
        + "#define SHARED_INPUT " + std::to_string(config.shared_input) + "\n"
        + "#define EXPERT_MAP_SIZE "
        + std::to_string(config.expert_map_size) + "\n"
        + "#define LOCAL_EXPERTS "
        + std::to_string(config.local_experts) + "\n"
        + "#define OUT_PER_EXPERT "
        + std::to_string(config.out_per_expert) + "\n"
        + "#define LOGICAL_OUT "
        + std::to_string(config.logical_output) + "\n"
        + "#define MOE_ROUTES "
        + std::to_string(config.moe_routes) + "\n";
}

std::string nint_epilogue_metal_type(Dtype dtype) {
    if (dtype == mlx::core::float16) return "half";
    if (dtype == mlx::core::float32) return "float";
    if (dtype == mlx::core::bfloat16) return "bfloat";
    if (dtype == mlx::core::bool_) return "bool";
    if (dtype == mlx::core::int8) return "char";
    if (dtype == mlx::core::uint8) return "uchar";
    if (dtype == mlx::core::int16) return "short";
    if (dtype == mlx::core::uint16) return "ushort";
    if (dtype == mlx::core::int32) return "int";
    if (dtype == mlx::core::uint32) return "uint";
    if (dtype == mlx::core::int64) return "long";
    if (dtype == mlx::core::uint64) return "ulong";
    throw std::invalid_argument("unsupported NINT epilogue dtype");
}

struct NintMatmulPlan {
    NintMatmulKernelConfig config;
    std::string kernel_name;

    std::string source() const {
        std::string result = "#include <metal_stdlib>\nusing namespace metal;\n"
            + nint_matmul_kernel_header(config) + kNintHeader
            + (config.add_residual ? kNintStoreOutputAdd
                : config.moe_shared ? kNintStoreOutputMoeShared : kNintStoreOutput);
        result += "\nkernel void " + kernel_name + "(\n";
        const std::array<const char*, 10> names{
            "q_packed", "row_metadata", "sub_scale", "sub_min", "x",
            "expert_ids", "expert_map", "residual", "route_weights", "gate_logits"};
        const std::array<const char*, 7> types{
            "uchar", "uint", "uchar", "uchar", "T", "int", "int"};
        const int count = config.add_residual ? 8 : config.moe_shared ? 10 : 7;
        for (int index = 0; index < count; ++index) {
            const std::string type = index < 7 ? types[index]
                : nint_epilogue_metal_type(config.epilogue_types[index - 7]);
            const char* name = config.moe_shared && index == 7
                ? "pair_output" : names[index];
            result += "device const " + type + "* " + name + " [[buffer("
                + std::to_string(index) + ")]],\n";
        }
        result += "device T* y [[buffer(" + std::to_string(count) + ")]],\n"
            "constant int& row_count [[buffer(" + std::to_string(count + 1) + ")]],\n"
            "uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],\n"
            "uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],\n"
            "uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {\n";
        result += kNintMatmul;
        result += "\n}\n";
        return result;
    }
};

const std::shared_ptr<const NintMatmulPlan>& compiled_nint_matmul_plan(
    NintMatmulKernelConfig config) {
    if (config.rows > 6) config.rows = 0;
    struct LocalEntry {
        NintMatmulKernelConfig config;
        std::shared_ptr<const NintMatmulPlan> plan;
    };
    thread_local std::vector<LocalEntry> local_cache;
    for (const auto& entry : local_cache) {
        if (entry.config == config) return entry.plan;
    }
    auto key = "mfq_native_nint_matmul_v2_" + nint_matmul_kernel_key(config);
    for (const auto dtype : config.epilogue_types)
        key += "_" + mlx::core::type_to_name(dtype);
    static std::mutex mutex;
    static std::unordered_map<std::string, std::shared_ptr<const NintMatmulPlan>> plans;
    std::lock_guard<std::mutex> lock(mutex);
    const auto [entry, inserted] = plans.try_emplace(key);
    if (inserted) entry->second = std::make_shared<NintMatmulPlan>(NintMatmulPlan{config, key});
    local_cache.push_back({config, entry->second});
    return local_cache.back().plan;
}

class NintMatmulPrimitive final : public mlx::core::UnaryPrimitive, public MlxPreparableKernel {
public:
    NintMatmulPrimitive(
        mlx::core::Stream stream, std::shared_ptr<const NintMatmulPlan> plan, int rows)
        : UnaryPrimitive(stream), plan_(std::move(plan)), rows_(rows) {}

    std::string preparation_key() const override { return plan_->kernel_name; }
    void prepare_gpu() override { (void)prepared_kernel(); }

    MTL::ComputePipelineState* prepared_kernel() {
        auto& selected_stream = stream();
        auto& device = mlx::core::metal::device(selected_stream.device);
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        auto* library = device.get_library(plan_->kernel_name, options,
            [plan = plan_] { return plan->source(); });
        return device.get_kernel(plan_->kernel_name, library);
    }

    void eval_cpu(const std::vector<array>&, array&) override {
        throw std::runtime_error("packed NINT matmul has no CPU path");
    }

    void eval_gpu(const std::vector<array>& inputs, array& output) override {
        const auto& config = plan_->config;
        const int count = config.add_residual ? 8 : config.moe_shared ? 10 : 7;
        if (inputs.size() != static_cast<std::size_t>(count))
            throw std::logic_error("packed NINT matmul input count mismatch");
        output.set_data(mlx::core::allocator::malloc(output.nbytes()));
        auto& selected_stream = stream();
        auto* kernel = prepared_kernel();
        auto& encoder = mlx::core::metal::get_command_encoder(selected_stream);
        encoder.set_compute_pipeline_state(kernel);
        for (int index = 0; index < count; ++index)
            encoder.set_input_array(inputs[index], index);
        encoder.set_output_array(output, count);
        encoder.set_bytes(rows_, count + 1);
        const int output_groups = (config.logical_output + 8 * config.outputs_per_simd - 1)
            / (8 * config.outputs_per_simd);
        const int row_groups = (rows_ + config.tile_rows - 1) / config.tile_rows;
        encoder.dispatch_threadgroups(
            MTL::Size(output_groups, row_groups, 1), MTL::Size(256, 1, 1));
    }

    const char* name() const override { return "NintMatmulPrimitive"; }

    bool is_equivalent(const mlx::core::Primitive& other) const override {
        const auto* primitive = dynamic_cast<const NintMatmulPrimitive*>(&other);
        return primitive && primitive->plan_ == plan_ && primitive->rows_ == rows_;
    }

    std::vector<Shape> output_shapes(const std::vector<array>&) override {
        return {Shape{rows_, plan_->config.logical_output}};
    }

private:
    std::shared_ptr<const NintMatmulPlan> plan_;
    int rows_;
};

array native_nint_matmul(
    std::vector<array> inputs, NintMatmulKernelConfig config) {
    for (auto& input : inputs)
        if (input.status() == array::Status::unscheduled || !input.flags().row_contiguous)
            input = mlx::core::contiguous(input);
    for (std::size_t index = 7; index < inputs.size(); ++index)
        config.epilogue_types[index - 7] = inputs[index].dtype();
    const auto& plan = compiled_nint_matmul_plan(config);
    const auto stream = mlx::core::default_stream(mlx::core::default_device());
    if (stream.device != mlx::core::Device::gpu)
        throw std::invalid_argument("packed NINT matmul requires the Metal device");
    return array(Shape{config.rows, config.logical_output},
        config.float32 ? mlx::core::float32 : mlx::core::float16,
        std::make_shared<NintMatmulPrimitive>(stream, plan, config.rows), std::move(inputs));
}

mlx::core::fast::CustomKernelFunction make_nint_swiglu_kernel() {
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    return mlx::core::fast::metal_kernel(
        "mfq_cpp_nint_metadata_swiglu",
        {
            "gate_q",
            "gate_row_metadata",
            "gate_sub_scale",
            "gate_sub_min",
            "up_q",
            "up_row_metadata",
            "up_sub_scale",
            "up_sub_min",
            "x",
        },
        {"y"},
        kNintSwiGlu,
        kNintHeader,
        true,
        false,
        options);
}

const mlx::core::fast::CustomKernelFunction& nint_swiglu_kernel() {
    static const auto kernel = make_nint_swiglu_kernel();
    return kernel;
}

const mlx::core::fast::CustomKernelFunction& compiled_nint_swiglu_kernel(
    bool float32,
    int group_size,
    int groups,
    int input_size,
    int output_size) {
    using Kernel = mlx::core::fast::CustomKernelFunction;
    struct LocalEntry {
        bool float32;
        int group_size;
        int groups;
        int input_size;
        int output_size;
        const Kernel* kernel;
    };
    thread_local std::vector<LocalEntry> local_cache;
    for (const auto& entry : local_cache) {
        if (entry.float32 == float32 &&
            entry.group_size == group_size &&
            entry.groups == groups &&
            entry.input_size == input_size &&
            entry.output_size == output_size) {
            return *entry.kernel;
        }
    }
    const auto key = std::string(float32 ? "f32" : "f16")
        + "_" + std::to_string(group_size)
        + "_" + std::to_string(groups)
        + "_" + std::to_string(input_size)
        + "_" + std::to_string(output_size);
    static std::mutex mutex;
    static std::unordered_map<std::string, Kernel> kernels;
    std::lock_guard<std::mutex> lock(mutex);
    if (const auto found = kernels.find(key); found != kernels.end()) {
        local_cache.push_back({
            float32, group_size, groups, input_size, output_size,
            &found->second});
        return found->second;
    }
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    auto header = std::string("#define T ")
        + (float32 ? "float\n" : "half\n")
        + "#define GS " + std::to_string(group_size) + "\n"
        + "#define NG " + std::to_string(groups) + "\n"
        + "#define K " + std::to_string(input_size) + "\n"
        + "#define OUT " + std::to_string(output_size) + "\n"
        + kNintHeader;
    auto kernel = mlx::core::fast::metal_kernel(
        "mfq_cpp_nint_runtime_swiglu_" + key,
        {
            "gate_q",
            "gate_row_metadata",
            "gate_sub_scale",
            "gate_sub_min",
            "up_q",
            "up_row_metadata",
            "up_sub_scale",
            "up_sub_min",
            "x",
        },
        {"y"},
        kNintSwiGlu,
        std::move(header),
        true,
        false,
        options);
    const auto [inserted, unused] = kernels.emplace(key, std::move(kernel));
    (void)unused;
    local_cache.push_back({
        float32, group_size, groups, input_size, output_size,
        &inserted->second});
    return inserted->second;
}

const array& nint_unrouted_sentinel() {
    static const auto sentinel = mlx::core::zeros(
        Shape{1}, mlx::core::int32);
    return sentinel;
}

mlx::core::fast::CustomKernelFunction make_nint_dequantize_kernel() {
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    return mlx::core::fast::metal_kernel(
        "mfq_cpp_nint_dequantize",
        {
            "q_packed",
            "row_q_layout",
            "row_q_byte_offsets",
            "sub_scale",
            "sub_min",
            "neuron_scale",
            "neuron_min",
        },
        {"y"},
        kNintDequantize,
        kNintHeader,
        true,
        false,
        options);
}

const mlx::core::fast::CustomKernelFunction& nint_dequantize_kernel() {
    static const auto kernel = make_nint_dequantize_kernel();
    return kernel;
}

mlx::core::fast::CustomKernelFunction make_nint_embedding_kernel() {
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    return mlx::core::fast::metal_kernel(
        "mfq_cpp_nint_embedding",
        {
            "q_packed",
            "row_q_layout",
            "row_q_byte_offsets",
            "sub_scale",
            "sub_min",
            "neuron_scale",
            "neuron_min",
            "token_ids",
        },
        {"y"},
        kNintEmbedding,
        kNintHeader,
        true,
        false,
        options);
}

const mlx::core::fast::CustomKernelFunction& nint_embedding_kernel() {
    static const auto kernel = make_nint_embedding_kernel();
    return kernel;
}

int nint_decode_outputs_per_simd(int input_size, int output_size) noexcept {
    const char* setting = std::getenv("MFQ_METAL_NINT_OUTPUTS_PER_SIMD");
    if (setting != nullptr && setting[1] == '\0' &&
        (setting[0] == '1' || setting[0] == '2' || setting[0] == '4')) {
        return setting[0] - '0';
    }
    return input_size > output_size ? 1 : 2;
}

} // namespace

std::string_view detail::nint_matmul_metal_header() noexcept {
    return kNintHeader;
}

std::string_view detail::nint_matmul_metal_body() noexcept {
    return kNintMatmul;
}

bool is_nint_dtype(std::string_view dtype) noexcept {
    return dtype == "NINT";
}

MlxNintWeight::MlxNintWeight(
    array q_packed,
    array sub_scale,
    array sub_min,
    array neuron_scale,
    array neuron_min,
    array row_q_layout,
    array row_q_byte_offsets,
    array row_metadata,
    int bits,
    int group_size,
    int groups,
    int input_size,
    int output_size,
    NintDescriptor descriptor,
    bool uniform_q_bits)
    : q_packed_(std::move(q_packed)),
      sub_scale_(std::move(sub_scale)),
      sub_min_(std::move(sub_min)),
      neuron_scale_(std::move(neuron_scale)),
      neuron_min_(std::move(neuron_min)),
      row_q_layout_(std::move(row_q_layout)),
      row_q_byte_offsets_(std::move(row_q_byte_offsets)),
      row_metadata_(std::move(row_metadata)),
      bits_(bits),
      group_size_(group_size),
      groups_(groups),
      input_size_(input_size),
      output_size_(output_size),
      descriptor_(descriptor),
      uniform_q_bits_(uniform_q_bits) {}

MlxNintWeight MlxNintWeight::from_blob(
    std::span<const std::uint8_t> blob) {
    BlobCursor cursor(blob);
    const auto raw_bits = cursor.scalar<std::uint8_t>("bits");
    const bool adaptive_storage = mfq::nint_has_adaptive_storage(raw_bits);
    const int bits = mfq::nint_logical_bits(raw_bits);
    const int sub_bits = cursor.scalar<std::uint8_t>("sub bits");
    const int group_size = cursor.scalar<std::int32_t>("group size");
    const int axis = cursor.scalar<std::int32_t>("axis");
    const int input_size = cursor.scalar<std::int32_t>("input size");
    const auto dimensions = cursor.scalar<std::uint32_t>("dimension count");
    if (bits <= 0 || bits > 8 || sub_bits <= 0 || sub_bits > 8 ||
        group_size <= 0 || input_size <= 0 || dimensions != 2 || axis != 0) {
        throw std::runtime_error("unsupported NINT Metal dimensions");
    }

    std::vector<std::int64_t> shape(dimensions);
    for (auto& value : shape) {
        value = cursor.scalar<std::int64_t>("shape");
    }
    const auto output_size = cursor.scalar<std::uint32_t>("output size");
    const auto groups = cursor.scalar<std::uint32_t>("group count");
    if (shape[0] != output_size || shape[1] != input_size || groups == 0 ||
        static_cast<std::uint64_t>(input_size) >
            static_cast<std::uint64_t>(groups) * group_size) {
        throw std::runtime_error("inconsistent NINT Metal dimensions");
    }

    detail::StagingVector<float> neuron_scale(output_size);
    detail::StagingVector<float> neuron_min(output_size);
    for (auto& value : neuron_scale) {
        value = half_to_float(cursor.scalar<std::uint16_t>("neuron scale"));
    }
    for (auto& value : neuron_min) {
        value = half_to_float(cursor.scalar<std::uint16_t>("neuron minimum"));
    }
    if (!std::all_of(
            neuron_scale.begin(), neuron_scale.end(),
            [](float value) { return std::isfinite(value); }) ||
        !std::all_of(
            neuron_min.begin(), neuron_min.end(),
            [](float value) { return std::isfinite(value); })) {
        throw std::runtime_error("NINT neuron metadata must be finite");
    }

    const auto metadata_count =
        static_cast<std::size_t>(output_size) * groups;
    const auto values_per_row =
        static_cast<std::size_t>(groups) * group_size;
    const auto q_count =
        static_cast<std::size_t>(output_size) * values_per_row;
    const auto packed_metadata_bytes = packed_size(metadata_count, sub_bits);
    const auto packed_q_bytes = packed_size(q_count, bits);
    const auto packed_tail = 2 * packed_metadata_bytes + packed_q_bytes;
    const auto old_tail = 2 * metadata_count + q_count;
    const bool old_unpacked_storage =
        !adaptive_storage && cursor.remaining() == old_tail;

    detail::StagingVector<std::uint8_t> sub_scale;
    detail::StagingVector<std::uint8_t> sub_min;
    detail::StagingVector<std::uint8_t> row_sub_bits(
        output_size,
        static_cast<std::uint8_t>(sub_bits));
    if (adaptive_storage) {
        const auto selector_bytes = packed_size(output_size, kSubSelectorBits);
        const auto selectors = unpack_values(
            cursor.bytes(selector_bytes, "sub-bit selectors"),
            output_size,
            kSubSelectorBits);
        sub_scale.resize(metadata_count);
        sub_min.resize(metadata_count);
        for (int selector = 0; selector < (1 << kSubSelectorBits); ++selector) {
            const int row_bits = sub_bits - 1 + selector;
            const auto selected_rows = static_cast<std::size_t>(std::count(
                selectors.begin(), selectors.end(),
                static_cast<std::uint8_t>(selector)));
            if (selected_rows == 0) {
                continue;
            }
            if (row_bits < 1 || row_bits > 8) {
                throw std::runtime_error("invalid NINT subgroup width");
            }
            const auto selected_values = selected_rows * groups;
            const auto stream_bytes = packed_size(selected_values, row_bits);
            const auto scales = unpack_values(
                cursor.bytes(stream_bytes, "sub scale"),
                selected_values,
                row_bits);
            const auto minima = unpack_values(
                cursor.bytes(stream_bytes, "sub minimum"),
                selected_values,
                row_bits);
            std::size_t local_row = 0;
            for (std::size_t row = 0; row < output_size; ++row) {
                if (selectors[row] != selector) {
                    continue;
                }
                const auto source = local_row * groups;
                const auto destination = row * groups;
                std::copy_n(scales.begin() + source, groups, sub_scale.begin() + destination);
                std::copy_n(minima.begin() + source, groups, sub_min.begin() + destination);
                row_sub_bits[row] = static_cast<std::uint8_t>(row_bits);
                ++local_row;
            }
        }
    } else if (old_unpacked_storage) {
        sub_scale = read_old_values(cursor, metadata_count, 1, "sub scale");
        sub_min = read_old_values(cursor, metadata_count, 1, "sub minimum");
    } else {
        if (cursor.remaining() != packed_tail) {
            throw std::runtime_error("invalid NINT packed payload length");
        }
        sub_scale = unpack_values(
            cursor.bytes(packed_metadata_bytes, "sub scale"),
            metadata_count,
            sub_bits);
        sub_min = unpack_values(
            cursor.bytes(packed_metadata_bytes, "sub minimum"),
            metadata_count,
            sub_bits);
    }

    detail::StagingVector<std::uint8_t> q_packed;
    detail::StagingVector<std::uint8_t> row_q_layout(output_size);
    detail::StagingVector<std::uint32_t> row_q_byte_offsets(output_size);
    if (adaptive_storage) {
        const auto selector_bytes = packed_size(output_size, kQSelectorBits);
        const auto selectors = unpack_values(
            cursor.bytes(selector_bytes, "q-bit selectors"),
            output_size,
            kQSelectorBits);
        std::size_t aligned_bytes = 0;
        for (std::size_t row = 0; row < output_size; ++row) {
            const int row_bits = static_cast<int>(selectors[row]) + 1;
            if (aligned_bytes > std::numeric_limits<std::uint32_t>::max()) {
                throw std::runtime_error(
                    "NINT packed stream exceeds Metal uint range");
            }
            row_q_layout[row] = static_cast<std::uint8_t>(row_bits);
            row_q_byte_offsets[row] =
                static_cast<std::uint32_t>(aligned_bytes);
            const auto row_bytes = packed_size(values_per_row, row_bits);
            if (row_bytes > std::numeric_limits<std::size_t>::max() - aligned_bytes) {
                throw std::runtime_error("NINT aligned q stream is too large");
            }
            aligned_bytes += row_bytes;
        }
        q_packed.reserve(aligned_bytes + 4);
        q_packed.resize(aligned_bytes, 0);
        for (int selector = 0; selector < (1 << kQSelectorBits); ++selector) {
            const int row_bits = selector + 1;
            const auto selected_rows = static_cast<std::size_t>(std::count(
                selectors.begin(), selectors.end(),
                static_cast<std::uint8_t>(selector)));
            const auto row_bit_count = values_per_row * row_bits;
            const auto stream_bytes = packed_size(
                selected_rows * values_per_row,
                row_bits);
            const auto stream = cursor.bytes(stream_bytes, "q values");
            std::size_t local_row = 0;
            for (std::size_t row = 0; row < output_size; ++row) {
                if (selectors[row] != selector) {
                    continue;
                }
                const auto destination = static_cast<std::size_t>(
                    row_q_byte_offsets[row]);
                const auto row_bytes = packed_size(values_per_row, row_bits);
                copy_packed_bit_range(
                    stream,
                    local_row * row_bit_count,
                    row_bit_count,
                    std::span<std::uint8_t>(
                        q_packed.data() + destination,
                        row_bytes));
                ++local_row;
            }
        }
    } else {
        q_packed = old_unpacked_storage
            ? pack_values(
                  read_old_values(cursor, q_count, 1, "quantized value"),
                  bits, 4)
            : cursor.bytes(packed_q_bytes, "q values", 4);
        const auto row_bit_count = values_per_row * bits;
        for (std::size_t row = 0; row < output_size; ++row) {
            const auto bit_offset = row * row_bit_count;
            const auto byte_offset = bit_offset / 8;
            if (byte_offset > std::numeric_limits<std::uint32_t>::max()) {
                throw std::runtime_error(
                    "NINT packed stream exceeds Metal uint range");
            }
            row_q_layout[row] = static_cast<std::uint8_t>(
                bits | ((bit_offset & 7u) << 4u));
            row_q_byte_offsets[row] = static_cast<std::uint32_t>(byte_offset);
        }
    }
    if (cursor.remaining() != 0) {
        throw std::runtime_error("trailing bytes in NINT tensor");
    }
    q_packed.insert(q_packed.end(), 4, 0);

    const bool uniform_q_bits = std::all_of(
        row_q_layout.begin(), row_q_layout.end(),
        [bits](std::uint8_t value) { return (value & 15u) == bits; });
    NintDescriptor descriptor;
    double encoded_bits = static_cast<double>(output_size) *
        static_cast<double>(32 + kQSelectorBits + kSubSelectorBits);
    std::array<std::size_t, 64> joint_counts{};
    for (std::size_t row = 0; row < output_size; ++row) {
        const int row_q_bits = static_cast<int>(row_q_layout[row] & 15u);
        const int row_k_bits = static_cast<int>(row_sub_bits[row]);
        encoded_bits +=
            static_cast<double>(values_per_row) * row_q_bits +
            2.0 * static_cast<double>(groups) * row_k_bits;
        ++joint_counts[static_cast<std::size_t>(
            (row_q_bits - 1) * 8 + (row_k_bits - 1))];
    }
    descriptor.aggregate_bpw = encoded_bits /
        (static_cast<double>(output_size) * input_size);
    for (const auto count : joint_counts) {
        if (count == 0) {
            continue;
        }
        const double probability =
            static_cast<double>(count) / output_size;
        descriptor.distribution_entropy -=
            probability * std::log2(probability);
    }
    detail::StagingVector<std::uint32_t> row_metadata(
        static_cast<std::size_t>(output_size) * 4u);
    for (std::size_t row = 0; row < output_size; ++row) {
        const auto base = row * 4u;
        row_metadata[base] = row_q_layout[row];
        row_metadata[base + 1u] = row_q_byte_offsets[row];
        row_metadata[base + 2u] = std::bit_cast<std::uint32_t>(
            neuron_scale[row]);
        row_metadata[base + 3u] = std::bit_cast<std::uint32_t>(
            neuron_min[row]);
    }
    return MlxNintWeight(
        make_array(q_packed, Shape{checked_shape(q_packed.size(), "q bytes")}),
        make_array(
            sub_scale,
            Shape{checked_shape(output_size, "output size"), checked_shape(groups, "groups")}),
        make_array(
            sub_min,
            Shape{checked_shape(output_size, "output size"), checked_shape(groups, "groups")}),
        make_array(neuron_scale, Shape{checked_shape(output_size, "output size")}),
        make_array(neuron_min, Shape{checked_shape(output_size, "output size")}),
        make_array(row_q_layout, Shape{checked_shape(output_size, "output size")}),
        make_array(row_q_byte_offsets, Shape{checked_shape(output_size, "output size")}),
        make_array(
            row_metadata,
            Shape{checked_shape(output_size, "output size"), 4}),
        bits,
        group_size,
        static_cast<int>(groups),
        input_size,
        static_cast<int>(output_size),
        descriptor,
        uniform_q_bits);
}

array MlxNintWeight::matmul(const array& input) const {
    return matmul_impl(input, nullptr);
}

array MlxNintWeight::matmul_packed(const array& input) const {
    return matmul_impl(input, nullptr, nullptr, nullptr, nullptr, false);
}

array MlxNintWeight::matmul_add(
    const array& input,
    const array& residual) const {
    return matmul_impl(input, &residual);
}

array MlxNintWeight::matmul_moe_shared(
    const array& input,
    const array& routed_pairs,
    const array& route_weights,
    const array& gate_logits) const {
    return matmul_impl(
        input,
        nullptr,
        &routed_pairs,
        &route_weights,
        &gate_logits);
}

array MlxNintWeight::matmul_impl(
    const array& input,
    const array* residual,
    const array* routed_pairs,
    const array* route_weights,
    const array* gate_logits,
    bool allow_dequantize) const {
    if (input.ndim() == 0 || input.shape(-1) != input_size_) {
        throw std::runtime_error("NINT input width does not match packed weight");
    }
    std::int64_t rows = 1;
    Shape output_shape = input.shape();
    for (std::size_t index = 0; index + 1 < input.ndim(); ++index) {
        rows *= input.shape(static_cast<int>(index));
    }
    if (rows <= 0 || rows > std::numeric_limits<std::int32_t>::max()) {
        throw std::runtime_error("unsupported NINT input row count");
    }
    output_shape.back() = output_size_;
    if (residual != nullptr && residual->shape() != output_shape) {
        throw std::runtime_error("NINT residual shape does not match output");
    }
    const bool has_moe_shared = routed_pairs != nullptr ||
        route_weights != nullptr || gate_logits != nullptr;
    if (has_moe_shared &&
        (routed_pairs == nullptr || route_weights == nullptr ||
         gate_logits == nullptr)) {
        throw std::invalid_argument(
            "NINT MoE shared epilogue requires pairs, weights, and gate");
    }
    if (has_moe_shared && residual != nullptr) {
        throw std::invalid_argument(
            "NINT residual and MoE shared epilogues are mutually exclusive");
    }
    int moe_routes = 0;
    if (has_moe_shared) {
        if (input.ndim() != 2 || routed_pairs->ndim() != 3 ||
            routed_pairs->shape(0) != rows ||
            routed_pairs->shape(2) != output_size_) {
            throw std::invalid_argument(
                "NINT MoE shared pairs must use [rows,routes,output]");
        }
        moe_routes = routed_pairs->shape(1);
        if (moe_routes <= 0 || moe_routes > 16 ||
            route_weights->shape() !=
                Shape{static_cast<int>(rows), moe_routes} ||
            gate_logits->size() != static_cast<std::size_t>(rows)) {
            throw std::invalid_argument(
                "NINT MoE shared weights or gate shape mismatch");
        }
    }

    auto source = input;
    if (rows >= 64 && source.dtype() == mlx::core::float32) {
        source = mlx::core::astype(source, mlx::core::float16);
    } else if (source.dtype() != mlx::core::float16 &&
               source.dtype() != mlx::core::float32) {
        source = mlx::core::astype(source, mlx::core::float16);
    }
    source = mlx::core::reshape(
        source,
        Shape{static_cast<std::int32_t>(rows), input_size_});
    if (rows >= 64) {
        // Decode packed rows once per matrix tile.  For very long prefills a
        // transient full dequantization wins because each weight tile would
        // otherwise be revisited many times; it is never retained as a
        // resident decoded copy.
        constexpr std::int64_t packed_prefill_rows = 2048;
        auto result = !allow_dequantize || rows < packed_prefill_rows
            ? dense_nint_mmq(
                  q_packed_,
                  row_metadata_,
                  sub_scale_,
                  sub_min_,
                  source,
                  DenseNintMmqConfig{
                      .rows = static_cast<int>(rows),
                      .output_width = output_size_,
                      .input_width = input_size_,
                      .group_size = group_size_,
                      .groups = groups_,
                      .use_nax = dense_nint_nax_enabled(),
                  })
            : mlx::core::matmul(
                  source,
                  mlx::core::transpose(dequantize(mlx::core::float16)));
        result = mlx::core::reshape(std::move(result), output_shape);
        if (residual != nullptr) return result + *residual;
        if (!has_moe_shared) return result;
        auto pairs = *routed_pairs;
        if (pairs.dtype() != result.dtype()) {
            pairs = mlx::core::astype(pairs, result.dtype());
        }
        auto weighted = mlx::core::sum(
            pairs * mlx::core::expand_dims(
                mlx::core::astype(*route_weights, result.dtype()),
                -1),
            1);
        auto gates = mlx::core::reshape(
            *gate_logits,
            Shape{static_cast<int>(rows), 1});
        return mlx::core::astype(
            weighted + mlx::core::sigmoid(gates) * result,
            result.dtype());
    }

    const int tile_rows = rows == 1
        ? 1
        : (rows <= 6 ? static_cast<int>(rows) : 8);
    const std::int64_t outputs_per_simd =
        rows == 1 ? nint_decode_outputs_per_simd(input_size_, output_size_) : 2;
    const std::int64_t outputs_per_threadgroup = 8 * outputs_per_simd;
    constexpr std::int64_t threads_per_threadgroup = 256;
    const auto output_threadgroups =
        (static_cast<std::int64_t>(output_size_) + outputs_per_threadgroup - 1) /
        outputs_per_threadgroup;
    const auto grid_x = output_threadgroups * threads_per_threadgroup;
    if (grid_x > std::numeric_limits<int>::max()) {
        throw std::runtime_error("NINT Metal grid exceeds MLX limits");
    }
    std::vector<array> inputs{
        q_packed_,
        row_metadata_,
        sub_scale_,
        sub_min_,
        source,
        nint_unrouted_sentinel(),
        nint_unrouted_sentinel(),
    };
    if (residual != nullptr) {
        inputs.push_back(mlx::core::contiguous(mlx::core::reshape(
            *residual,
            Shape{static_cast<std::int32_t>(rows), output_size_})));
    } else if (has_moe_shared) {
        auto pairs = *routed_pairs;
        if (pairs.dtype() != source.dtype()) {
            pairs = mlx::core::astype(pairs, source.dtype());
        }
        inputs.push_back(mlx::core::contiguous(std::move(pairs)));
        inputs.push_back(mlx::core::contiguous(
            mlx::core::astype(*route_weights, mlx::core::float32)));
        inputs.push_back(mlx::core::contiguous(mlx::core::reshape(
            *gate_logits,
            Shape{static_cast<std::int32_t>(rows)})));
    }
    auto result = native_nint_matmul(std::move(inputs), {
        .float32 = source.dtype() == mlx::core::float32,
        .add_residual = residual != nullptr,
        .moe_shared = has_moe_shared,
        .moe_routes = moe_routes,
        .group_size = group_size_,
        .groups = groups_,
        .input_size = input_size_,
        .output_size = output_size_,
        .rows = static_cast<int>(rows),
        .tile_rows = tile_rows,
        .outputs_per_simd = static_cast<int>(outputs_per_simd),
        .routed = 0,
        .routes = 1,
        .shared_input = 0,
        .expert_map_size = 1,
        .local_experts = 1,
        .out_per_expert = output_size_,
        .logical_output = output_size_,
    });
    return mlx::core::reshape(std::move(result), std::move(output_shape));
}

array MlxNintWeight::routed_matmul(
    const array& input,
    const array& expert_ids,
    const array& expert_map,
    int out_per_expert) const {
    if (out_per_expert <= 0 || output_size_ % out_per_expert != 0) {
        throw std::invalid_argument(
            "NINT routed output width is inconsistent");
    }
    if (expert_ids.ndim() != 2 || expert_map.ndim() != 1 ||
        expert_ids.dtype() != mlx::core::int32 ||
        expert_map.dtype() != mlx::core::int32) {
        throw std::invalid_argument(
            "NINT routing metadata must be contiguous int32 IDs");
    }
    const int tokens = expert_ids.shape(0);
    const int routes = expert_ids.shape(1);
    const bool shared_input = input.ndim() == 2 &&
        input.shape(0) == tokens && input.shape(1) == input_size_;
    if (!shared_input && (
        input.ndim() != 3 || input.shape(0) != tokens ||
        input.shape(1) != routes || input.shape(2) != input_size_)) {
        throw std::invalid_argument(
            "NINT routed input must be [tokens,K] or [tokens,routes,K]");
    }
    const auto route_count =
        static_cast<std::size_t>(tokens) * static_cast<std::size_t>(routes);
    if (route_count > static_cast<std::size_t>(
            std::numeric_limits<int>::max())) {
        throw std::runtime_error("NINT route count exceeds MLX limits");
    }
    const Shape output_shape{tokens, routes, out_per_expert};
    if (route_count == 0) {
        const auto output_dtype = input.dtype() == mlx::core::float32
            ? mlx::core::float32
            : mlx::core::float16;
        return mlx::core::zeros(output_shape, output_dtype);
    }

    auto source = input;
    if (source.dtype() != mlx::core::float16 &&
        source.dtype() != mlx::core::float32) {
        source = mlx::core::astype(source, mlx::core::float16);
    }
    source = mlx::core::contiguous(std::move(source));
    auto ids = mlx::core::contiguous(expert_ids);
    auto map = mlx::core::contiguous(expert_map);

    constexpr int threads_per_threadgroup = 256;
    constexpr int outputs_per_threadgroup = 16;
    const auto output_threadgroups =
        (static_cast<std::int64_t>(out_per_expert)
            + outputs_per_threadgroup - 1) / outputs_per_threadgroup;
    const auto grid_x = output_threadgroups * threads_per_threadgroup;
    if (grid_x > std::numeric_limits<int>::max()) {
        throw std::runtime_error("NINT routed Metal grid exceeds MLX limits");
    }
    auto result = native_nint_matmul(
        {q_packed_, row_metadata_, sub_scale_, sub_min_, source, ids, map}, {
        .float32 = source.dtype() == mlx::core::float32,
        .add_residual = false,
        .group_size = group_size_,
        .groups = groups_,
        .input_size = input_size_,
        .output_size = output_size_,
        .rows = static_cast<int>(route_count),
        .tile_rows = 1,
        .outputs_per_simd = 2,
        .routed = 1,
        .routes = routes,
        .shared_input = static_cast<int>(shared_input),
        .expert_map_size = expert_map.shape(0),
        .local_experts = output_size_ / out_per_expert,
        .out_per_expert = out_per_expert,
        .logical_output = out_per_expert,
    });
    return mlx::core::reshape(
        std::move(result),
        output_shape);
}

std::optional<array> MlxNintWeight::grouped_row_matmul(
    const array& input,
    int group_count) const {
    if (group_count <= 0 || input.ndim() < 2 ||
        input.shape(-2) != group_count || input.shape(-1) != input_size_ ||
        output_size_ % group_count != 0) {
        throw std::runtime_error("NINT grouped-row input geometry mismatch");
    }

    std::int64_t rows = 1;
    for (std::size_t dimension = 0;
         dimension + 2 < input.ndim();
         ++dimension) {
        const auto dimension_size =
            input.shape(static_cast<int>(dimension));
        if (dimension_size != 0 &&
            rows > std::numeric_limits<int>::max() / dimension_size) {
            throw std::runtime_error(
                "NINT grouped-row prefix exceeds MLX limits");
        }
        rows *= dimension_size;
    }

    Shape output_shape = input.shape();
    output_shape.back() = output_size_ / group_count;
    if (rows == 0) {
        const auto output_dtype = input.dtype() == mlx::core::float32
            ? mlx::core::float32
            : mlx::core::float16;
        return mlx::core::zeros(output_shape, output_dtype);
    }

    std::vector<std::int32_t> ids(
        static_cast<std::size_t>(rows) * group_count);
    for (std::int64_t row = 0; row < rows; ++row) {
        for (int group = 0; group < group_count; ++group) {
            ids[static_cast<std::size_t>(row) * group_count + group] =
                group;
        }
    }
    std::vector<std::int32_t> identity_map(
        static_cast<std::size_t>(group_count));
    for (int group = 0; group < group_count; ++group) {
        identity_map[static_cast<std::size_t>(group)] = group;
    }

    const auto source = mlx::core::reshape(
        input,
        Shape{
            static_cast<int>(rows),
            group_count,
            input_size_,
        });
    const array expert_ids(
        ids.begin(),
        Shape{static_cast<int>(rows), group_count});
    const array expert_map(
        identity_map.begin(),
        Shape{group_count});
    auto output = routed_matmul(
        source,
        expert_ids,
        expert_map,
        output_size_ / group_count);
    return mlx::core::reshape(std::move(output), std::move(output_shape));
}

std::optional<array> MlxNintWeight::greedy_argmax(const array& input) const {
    if (input.ndim() == 0 || input.shape(-1) != input_size_) {
        throw std::runtime_error("NINT greedy input width mismatch");
    }
    return std::nullopt;
}

std::optional<MlxNintSwiGluPair>
MlxNintSwiGluPair::from_weights(
    const MlxNintWeight& gate,
    const MlxNintWeight& up) {
    if (!gate.can_fuse_swiglu(up)) {
        return std::nullopt;
    }
    const auto gate_q_bytes = gate.packed_values().size();
    if (gate_q_bytes > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error(
            "combined NINT Gate/Up q offset exceeds Metal uint range");
    }

    const auto& gate_metadata = gate.row_metadata();
    const auto& up_metadata = up.row_metadata();
    detail::StagingVector<std::uint32_t> metadata(
        gate_metadata.size() + up_metadata.size());
    std::copy_n(
        gate_metadata.data<std::uint32_t>(),
        gate_metadata.size(),
        metadata.begin());
    std::copy_n(
        up_metadata.data<std::uint32_t>(),
        up_metadata.size(),
        metadata.begin() +
            static_cast<std::ptrdiff_t>(gate_metadata.size()));
    for (int row = 0; row < up.output_size(); ++row) {
        const auto offset = gate_metadata.size() +
            static_cast<std::size_t>(row) * 4u + 1u;
        const auto original = metadata[offset];
        if (original > std::numeric_limits<std::uint32_t>::max() -
                static_cast<std::uint32_t>(gate_q_bytes)) {
            throw std::runtime_error(
                "combined NINT Gate/Up row offset exceeds Metal uint range");
        }
        metadata[offset] = original +
            static_cast<std::uint32_t>(gate_q_bytes);
    }

    detail::StagingVector<std::uint8_t> q_values(
        gate.packed_values().size() + up.packed_values().size());
    std::copy_n(
        gate.packed_values().data<std::uint8_t>(),
        gate.packed_values().size(),
        q_values.begin());
    std::copy_n(
        up.packed_values().data<std::uint8_t>(),
        up.packed_values().size(),
        q_values.begin() + static_cast<std::ptrdiff_t>(
            gate.packed_values().size()));
    detail::StagingVector<std::uint8_t> sub_scale_values(
        gate.sub_scales().size() + up.sub_scales().size());
    detail::StagingVector<std::uint8_t> sub_min_values(
        gate.sub_mins().size() + up.sub_mins().size());
    std::copy_n(
        gate.sub_scales().data<std::uint8_t>(),
        gate.sub_scales().size(),
        sub_scale_values.begin());
    std::copy_n(
        up.sub_scales().data<std::uint8_t>(),
        up.sub_scales().size(),
        sub_scale_values.begin() + static_cast<std::ptrdiff_t>(
            gate.sub_scales().size()));
    std::copy_n(
        gate.sub_mins().data<std::uint8_t>(),
        gate.sub_mins().size(),
        sub_min_values.begin());
    std::copy_n(
        up.sub_mins().data<std::uint8_t>(),
        up.sub_mins().size(),
        sub_min_values.begin() + static_cast<std::ptrdiff_t>(
            gate.sub_mins().size()));
    auto q_packed = make_array(
        q_values,
        Shape{static_cast<int>(q_values.size())});
    auto sub_scale = make_array(
        sub_scale_values,
        Shape{2 * gate.output_size(), gate.groups()});
    auto sub_min = make_array(
        sub_min_values,
        Shape{2 * gate.output_size(), gate.groups()});
    auto row_metadata = make_array(
        metadata,
        Shape{2 * gate.output_size(), 4});
    return MlxNintSwiGluPair(
        std::move(q_packed),
        std::move(row_metadata),
        std::move(sub_scale),
        std::move(sub_min),
        gate.group_size(),
        gate.groups(),
        gate.input_size(),
        gate.output_size());
}

bool MlxNintWeight::can_fuse_swiglu(
    const MlxNintWeight& up) const noexcept {
    return group_size_ > 0 &&
        group_size_ == up.group_size_ &&
        groups_ == up.groups_ &&
        input_size_ == up.input_size_ &&
        output_size_ == up.output_size_;
}

array MlxNintWeight::swiglu(
    const MlxNintWeight& up,
    const array& input) const {
    if (!can_fuse_swiglu(up)) {
        throw std::runtime_error(
            "fused NINT SwiGLU requires matching matrix geometry");
    }
    if (input.ndim() == 0 || input.shape(-1) != input_size_) {
        throw std::runtime_error("fused NINT SwiGLU input width mismatch");
    }
    const auto rows =
        input.size() / static_cast<std::size_t>(input_size_);
    if (rows != 1) {
        throw std::runtime_error(
            "fused NINT SwiGLU is restricted to one decode row");
    }

    Shape output_shape = input.shape();
    output_shape.back() = output_size_;
    auto source = input;
    if (source.dtype() != mlx::core::float16 &&
        source.dtype() != mlx::core::float32) {
        source = mlx::core::astype(source, mlx::core::float16);
    }
    source = mlx::core::reshape(source, Shape{1, input_size_});

    constexpr std::int64_t outputs_per_threadgroup = 16;
    constexpr std::int64_t threads_per_threadgroup = 256;
    const auto output_threadgroups =
        (static_cast<std::int64_t>(output_size_) +
         outputs_per_threadgroup - 1) /
        outputs_per_threadgroup;
    const auto grid_x = output_threadgroups * threads_per_threadgroup;
    if (grid_x > std::numeric_limits<int>::max()) {
        throw std::runtime_error(
            "fused NINT SwiGLU Metal grid exceeds MLX limits");
    }

    const auto& kernel = compiled_nint_swiglu_kernel(
        source.dtype() == mlx::core::float32,
        group_size_,
        groups_,
        input_size_,
        output_size_);
    auto outputs = kernel(
        {
            q_packed_,
            row_metadata_,
            sub_scale_,
            sub_min_,
            up.q_packed_,
            up.row_metadata_,
            up.sub_scale_,
            up.sub_min_,
            source,
        },
        {Shape{1, output_size_}},
        {source.dtype()},
        {static_cast<int>(grid_x), 1, 1},
        {static_cast<int>(threads_per_threadgroup), 1, 1},
        {},
        std::nullopt,
        false,
        {});
    return mlx::core::reshape(outputs.front(), std::move(output_shape));
}

array MlxNintWeight::dequantize(Dtype dtype) const {
    if (dtype != mlx::core::float16 && dtype != mlx::core::float32) {
        throw std::runtime_error(
            "NINT dequantization output must be float16 or float32");
    }
    const auto element_count =
        static_cast<std::uint64_t>(output_size_) * input_size_;
    if (element_count > static_cast<std::uint64_t>(
                            std::numeric_limits<int>::max())) {
        throw std::runtime_error("NINT dequantization grid exceeds MLX limits");
    }
    const int values_per_thread = group_size_ % 4 == 0 ? 4 : 1;
    const int grid = static_cast<int>(static_cast<std::uint64_t>(output_size_) *
        ((static_cast<std::uint64_t>(input_size_) + values_per_thread - 1) / values_per_thread));
    auto outputs = nint_dequantize_kernel()(
        {
            q_packed_,
            row_q_layout_,
            row_q_byte_offsets_,
            sub_scale_,
            sub_min_,
            neuron_scale_,
            neuron_min_,
        },
        {Shape{output_size_, input_size_}},
        {dtype},
        {grid, 1, 1},
        {std::min(256, std::max(1, grid)), 1, 1},
        {
            {"T", dtype},
            {"GS", group_size_},
            {"NG", groups_},
            {"K", input_size_},
            {"OUT", output_size_},
            {"VEC", values_per_thread},
        },
        std::nullopt,
        false,
        {});
    return std::move(outputs.front());
}

array MlxNintWeight::embedding(
    const array& token_ids,
    Dtype dtype) const {
    if (dtype != mlx::core::float16 && dtype != mlx::core::float32) {
        throw std::runtime_error(
            "NINT embedding output must be float16 or float32");
    }
    auto ids = token_ids;
    if (ids.dtype() != mlx::core::int32 && ids.dtype() != mlx::core::uint32) {
        ids = mlx::core::astype(ids, mlx::core::int32);
    }
    const auto count = ids.size();
    Shape output_shape = ids.shape();
    output_shape.push_back(input_size_);
    if (count == 0) {
        return mlx::core::zeros(output_shape, dtype);
    }
    if (count > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        count > static_cast<std::size_t>(
                    std::numeric_limits<int>::max() / input_size_)) {
        throw std::runtime_error("NINT embedding input is too large");
    }
    ids = mlx::core::reshape(ids, Shape{static_cast<std::int32_t>(count)});
    const int grid = static_cast<int>(count) * input_size_;
    auto outputs = nint_embedding_kernel()(
        {
            q_packed_,
            row_q_layout_,
            row_q_byte_offsets_,
            sub_scale_,
            sub_min_,
            neuron_scale_,
            neuron_min_,
            ids,
        },
        {Shape{static_cast<std::int32_t>(count), input_size_}},
        {dtype},
        {grid, 1, 1},
        {std::min(256, std::max(1, grid)), 1, 1},
        {
            {"T", dtype},
            {"GS", group_size_},
            {"NG", groups_},
            {"K", input_size_},
        },
        std::nullopt,
        false,
        {});
    return mlx::core::reshape(outputs.front(), std::move(output_shape));
}

std::size_t MlxNintWeight::packed_nbytes() const noexcept {
    return q_packed_.nbytes() +
        sub_scale_.nbytes() +
        sub_min_.nbytes() +
        neuron_scale_.nbytes() +
        neuron_min_.nbytes() +
        row_q_layout_.nbytes() +
        row_q_byte_offsets_.nbytes();
}

} // namespace mfq::metal
