#include "qwen4_ops.h"

#include "mlx_transformer.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mfq::metal {
namespace {

using mlx::core::Shape;
using mlx::core::array;
using mlx::core::CompileOptions;
using mlx::core::MathMode;

constexpr int kGatedHcProjectionThreads = 256;
constexpr int kGatedHcUpThreads = 320;

constexpr const char* kQsaDecodePrologueSource = R"METAL(
    constexpr uint VALUES_PER_THREAD = 4u;
    constexpr uint THREADS = 64u;
    constexpr uint SIMD_GROUPS = THREADS / 32u;
    constexpr uint TOTAL_HEADS =
        uint(QUERY_HEADS + KEY_HEADS + INDEX_HEADS);

    uint head_group = threadgroup_position_in_grid.x;
    uint local_thread = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    bool is_query = head_group < uint(QUERY_HEADS);
    bool is_key = !is_query &&
        head_group < uint(QUERY_HEADS + KEY_HEADS);
    uint head = is_query
        ? head_group
        : (is_key
            ? head_group - uint(QUERY_HEADS)
            : head_group - uint(QUERY_HEADS + KEY_HEADS));
    uint width = is_query || is_key
        ? uint(HEAD_DIM)
        : uint(INDEX_DIM);
    uint source_base = is_query
        ? head * uint(2 * HEAD_DIM)
        : (is_key
            ? head * uint(HEAD_DIM)
            : head * uint(INDEX_DIM));
    device const T* source = is_query
        ? query_gate_input
        : (is_key ? key_input : index_query_key_input);
    device const float* norm_weight = is_query
        ? query_weight
        : (is_key ? key_weight : index_weight);

    threadgroup float inverse_rms[1];
    threadgroup float local_sums[32];
    uint column_base = local_thread * VALUES_PER_THREAD;
    float sum_squares = 0.0f;
    for (uint item = 0u; item < VALUES_PER_THREAD; ++item) {
        uint column = column_base + item;
        if (column < width) {
            float value = float(source[source_base + column]);
            sum_squares += value * value;
        }
    }
    sum_squares = simd_sum(sum_squares);
    if (simd_group == 0u) local_sums[lane] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) local_sums[simd_group] = sum_squares;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0u) {
        float total = simd_sum(local_sums[lane]);
        if (lane == 0u) {
            inverse_rms[0] = metal::precise::rsqrt(
                total / float(width) + params[0]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    constexpr uint ROTARY_HALF = uint(ROTARY_DIM / 2);
    int position = positions[0];
    for (uint item = 0u; item < VALUES_PER_THREAD; ++item) {
        uint column = column_base + item;
        if (column >= width) continue;
        float normalized;
        if (column < uint(ROTARY_DIM)) {
            uint pair = column < ROTARY_HALF
                ? column
                : column - ROTARY_HALF;
            T first_rounded = T(
                float(source[source_base + pair])
                * inverse_rms[0] * norm_weight[pair]);
            T second_rounded = T(
                float(source[source_base + pair + ROTARY_HALF])
                * inverse_rms[0]
                * norm_weight[pair + ROTARY_HALF]);
            float exponent =
                -2.0f * float(pair) / float(ROTARY_DIM);
            float angle = float(position) * pow(params[1], exponent);
            float cosine = cos(angle);
            float sine = sin(angle);
            normalized = column < ROTARY_HALF
                ? float(first_rounded) * cosine
                    - float(second_rounded) * sine
                : float(second_rounded) * cosine
                    + float(first_rounded) * sine;
        } else {
            normalized = float(T(
                float(source[source_base + column])
                * inverse_rms[0] * norm_weight[column]));
        }
        uint output_index = head * width + column;
        if (is_query) {
            query_output[output_index] = T(normalized);
            output_gate[output_index] = query_gate_input[
                source_base + uint(HEAD_DIM) + column];
        } else if (is_key) {
            key_output[output_index] = T(normalized);
        } else {
            index_query_output[output_index] = T(normalized);
        }
    }
)METAL";

constexpr const char* kGroupedRmsAffineSource = R"METAL(
    uint row = threadgroup_position_in_grid.x;
    uint local_thread = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    constexpr uint VALUES_PER_THREAD = 4u;
    constexpr uint SIMD_SIZE = 32u;
    threadgroup float inverse_rms[1];
    threadgroup float local_sums[SIMD_SIZE];

    uint input_offset = row * uint(GROUP_SIZE);
    uint weight_offset =
        (row % uint(GROUP_COUNT)) * uint(GROUP_SIZE);
    uint column = local_thread * VALUES_PER_THREAD;
    float values[VALUES_PER_THREAD] = {
        0.0f, 0.0f, 0.0f, 0.0f};
    float sum_squares = 0.0f;
    for (uint item = 0u; item < VALUES_PER_THREAD; ++item) {
        uint current = column + item;
        if (current < uint(GROUP_SIZE)) {
            float value = float(input[input_offset + current]);
            values[item] = value;
            sum_squares += value * value;
        }
    }

    // Mirror MLX's non-looped FP32 RMS kernel exactly: four adjacent reads
    // per thread, SIMD reduction, then a 32-slot cross-SIMD reduction.
    sum_squares = simd_sum(sum_squares);
    if (simd_group == 0u)
        local_sums[lane] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u)
        local_sums[simd_group] = sum_squares;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0u) {
        float total = simd_sum(local_sums[lane]);
        if (lane == 0u) {
            inverse_rms[0] = metal::precise::rsqrt(
                total / float(GROUP_SIZE) + epsilon[0]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint item = 0u; item < VALUES_PER_THREAD; ++item) {
        uint current = column + item;
        if (current < uint(GROUP_SIZE)) {
            float scaled = (values[item] * inverse_rms[0]) *
                (1.0f + float(weight[weight_offset + current]));
            output[input_offset + current] = T(scaled);
        }
    }
)METAL";

constexpr const char* kGroupedRmsAffineWriteSource = R"METAL(
    uint row = threadgroup_position_in_grid.x;
    uint local_thread = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    constexpr uint VALUES_PER_THREAD = 4u;
    constexpr uint SIMD_SIZE = 32u;
    threadgroup float inverse_rms[1];
    threadgroup float local_sums[SIMD_SIZE];

    uint token = row / uint(HC_COUNT);
    uint stream = row % uint(HC_COUNT);
    uint input_offset = row * uint(GROUP_SIZE);
    uint branch_offset = token * uint(GROUP_SIZE);
    uint weight_offset = stream * uint(GROUP_SIZE);
    G gate = injection[token * uint(HC_COUNT) + stream];
    uint column = local_thread * VALUES_PER_THREAD;
    float values[VALUES_PER_THREAD] = {
        0.0f, 0.0f, 0.0f, 0.0f};
    float sum_squares = 0.0f;
    for (uint item = 0u; item < VALUES_PER_THREAD; ++item) {
        uint current = column + item;
        if (current < uint(GROUP_SIZE)) {
            // Preserve the standalone residual kernel's promoted multiply,
            // explicit rounding, and add order before normalizing the result.
            U update = U(U(branch[branch_offset + current]) * U(gate));
            O value = O(O(residual[input_offset + current]) + O(update));
            updated[input_offset + current] = value;
            values[item] = float(value);
            sum_squares += values[item] * values[item];
        }
    }

    sum_squares = simd_sum(sum_squares);
    if (simd_group == 0u)
        local_sums[lane] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u)
        local_sums[simd_group] = sum_squares;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0u) {
        float total = simd_sum(local_sums[lane]);
        if (lane == 0u) {
            inverse_rms[0] = metal::precise::rsqrt(
                total / float(GROUP_SIZE) + epsilon[0]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint item = 0u; item < VALUES_PER_THREAD; ++item) {
        uint current = column + item;
        if (current < uint(GROUP_SIZE)) {
            float scaled = (values[item] * inverse_rms[0]) *
                (1.0f + float(weight[weight_offset + current]));
            normalized[input_offset + current] = O(scaled);
        }
    }
)METAL";

constexpr const char* kGatedHcProjectionSource = R"METAL(
    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint workgroup = threadgroup_position_in_grid.x;
    constexpr uint ROWS_PER_WORKGROUP = 4u;
    constexpr uint DOWN_WORKGROUPS = uint(LOW_RANK) / ROWS_PER_WORKGROUP;
    constexpr uint SIMD_GROUPS = 8u;
    threadgroup float partials[SIMD_GROUPS * ROWS_PER_WORKGROUP];

    bool is_injection = workgroup >= DOWN_WORKGROUPS;
    uint matrix_workgroup = is_injection
        ? workgroup - DOWN_WORKGROUPS
        : workgroup;
    uint row_base = matrix_workgroup * ROWS_PER_WORKGROUP;

    // This is the same 8-way K split and four-adjacent-value traversal used
    // by MLX's BF16/F16 GEMV for the decode geometry. Both projections share
    // normalized, so one launch can produce the 320 low-rank values and the
    // four residual-injection values without a second tiny GEMV dispatch.
    float accumulators[ROWS_PER_WORKGROUP] = {
        0.0f, 0.0f, 0.0f, 0.0f};
    uint column = simd_group * 128u + lane * 4u;
    for (; column < uint(WIDTH); column += 1024u) {
        for (uint item = 0u; item < 4u; ++item) {
            float value = float(normalized[column + item]);
            for (uint row = 0u; row < ROWS_PER_WORKGROUP; ++row) {
                uint offset = (row_base + row) * uint(WIDTH) + column + item;
                float weight = is_injection
                    ? float(injection_weight[offset])
                    : float(down_weight[offset]);
                accumulators[row] += weight * value;
            }
        }
    }

    for (uint row = 0u; row < ROWS_PER_WORKGROUP; ++row) {
        float value = accumulators[row];
        for (ushort delta = 16; delta >= 1; delta >>= 1)
            value += simd_shuffle_down(value, delta);
        if (lane == 0u)
            partials[simd_group * ROWS_PER_WORKGROUP + row] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0u && lane < ROWS_PER_WORKGROUP) {
        float value = partials[lane];
        for (uint group = 1u; group < SIMD_GROUPS; ++group)
            value += partials[group * ROWS_PER_WORKGROUP + lane];
        uint output_row = row_base + lane;
        if (is_injection) {
            I projected = I(value);
            I scaled = I(projected / I(HC_COUNT));
            I tail = I(1) /
                (I(1) + metal::exp(metal::abs(scaled)));
            I sigmoid = scaled < I(0) ? tail : I(1) - tail;
            injection[output_row] = I(I(2) * sigmoid);
        } else {
            down[output_row] = D(value);
        }
    }
)METAL";

constexpr const char* kGatedHcUpCollapseSource = R"METAL(
    constexpr uint FEATURES_PER_TG = 8u;
    constexpr uint CHUNK = 32u;
    constexpr uint CHUNKS = (uint(LOW_RANK) + CHUNK - 1u) / CHUNK;
    constexpr uint OUTPUT_ROWS = FEATURES_PER_TG * uint(HC_COUNT);
    constexpr uint PART_STRIDE = CHUNKS + 1u;
    static_assert(HC_COUNT == 4, "coalesced HC up expects four streams");
    static_assert(LOW_RANK == 320, "coalesced HC up is tuned for Qwen4");
    static_assert(CHUNKS * OUTPUT_ROWS == 320,
                  "one thread per up-projection chunk");

    const uint tid = thread_index_in_threadgroup;
    const uint chunk = tid % CHUNKS;
    const uint output_row = tid / CHUNKS;
    const uint stream = output_row & 3u;
    const uint local_feature = output_row >> 2u;
    const uint feature =
        threadgroup_position_in_grid.x * FEATURES_PER_TG + local_feature;

    // All 320 threads first produce one activated low-rank value.  The old
    // kernel recomputed these values independently in every SIMD group.
    threadgroup L activated[LOW_RANK];
    threadgroup float partials[OUTPUT_ROWS * PART_STRIDE];
    if (tid < uint(LOW_RANK)) {
        L value = L(down_projection[tid] / L(HC_COUNT));
        L tail = L(1) /
            (L(1) + metal::exp(metal::abs(value)));
        L sigmoid = value < L(0) ? tail : L(1) - tail;
        activated[tid] = L(value * sigmoid);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float accumulator = 0.0f;
    const uint low_base = chunk * CHUNK;
    if (feature < uint(HIDDEN)) {
        const uint row = stream * uint(HIDDEN) + feature;
        const uint weight_base = row * uint(LOW_RANK) + low_base;
        for (uint item = 0u; item < CHUNK; ++item) {
            const uint low = low_base + item;
            if (low < uint(LOW_RANK)) {
                accumulator += float(up_weight[weight_base + item]) *
                    float(activated[low]);
            }
        }
    }
    partials[output_row * PART_STRIDE + chunk] = accumulator;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // The first SIMD group reduces ten contiguous chunks for 32 projection
    // rows (four streams by eight features), then performs the stream mix.
    if (tid < OUTPUT_ROWS) {
        const uint result_stream = tid & 3u;
        const uint result_feature =
            threadgroup_position_in_grid.x * FEATURES_PER_TG + (tid >> 2u);
        float projected = 0.0f;
        const uint base = tid * PART_STRIDE;
        for (uint current = 0u; current < CHUNKS; ++current)
            projected += partials[base + current];
        T rounded = T(projected);
        T tail = T(1) /
            (T(1) + metal::exp(metal::abs(rounded)));
        T gate = rounded < T(0) ? tail : T(1) - tail;
        T product = T(gate * T(normalized[
            result_stream * uint(HIDDEN) + result_feature]));
        float mixed = float(product);
        mixed += simd_shuffle_down(mixed, 1);
        mixed += simd_shuffle_down(mixed, 2);
        if (result_stream == 0u && result_feature < uint(HIDDEN))
            branch[result_feature] = T(mixed / float(HC_COUNT));
    }
)METAL";

// Decode-only two-stage hyper-connection path.  The first launch repeats the
// tiny per-stream norm in each projection row group so the 10240x320 down
// projection can consume threadgroup-resident values without materializing a
// separate norm launch.  Four stream-slice sums are joined by the up launch.
constexpr const char* kGatedHcNormDownSource = R"METAL(
    constexpr uint STREAMS = 4u;
    constexpr uint ROWS_PER_SIMD = 2u;
    constexpr uint ROWS_PER_TG = 8u * ROWS_PER_SIMD;
    constexpr uint DOWN_GROUPS = uint(LOW_RANK) / ROWS_PER_TG;
    constexpr uint GROUPS = DOWN_GROUPS + uint(HAS_INJECTION);
    constexpr uint PER = (uint(HIDDEN) + 255u) / 256u;
    constexpr uint PART_STRIDE = uint(LOW_RANK + HC_COUNT);
    static_assert(HC_COUNT == 4 && HIDDEN == 2560 && LOW_RANK == 320,
                  "Qwen4 HC decode geometry changed");

    const uint tid = thread_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint simd_group = simdgroup_index_in_threadgroup;
    const uint workgroup = threadgroup_position_in_grid.x;
    const uint stream = workgroup & 3u;
    const uint group = workgroup >> 2u;
    const uint input_base = stream * uint(HIDDEN);

    threadgroup N normalized_stream[HIDDEN];
    threadgroup float norm_partials[8];
    float values[PER];
    float scales[PER];
    float sum_squares = 0.0f;
    for (uint item = 0u; item < PER; ++item) {
        const uint feature = tid + item * 256u;
        float value = 0.0f;
        float scale = 0.0f;
        if (feature < uint(HIDDEN)) {
            value = float(input[input_base + feature]);
            scale = 1.0f + float(norm_weight[input_base + feature]);
        }
        values[item] = value;
        scales[item] = scale;
        sum_squares += value * value;
    }
    sum_squares = simd_sum(sum_squares);
    if (lane == 0u) norm_partials[simd_group] = sum_squares;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint part = 0u; part < 8u; ++part)
        total += norm_partials[part];
    const float inverse_rms = metal::rsqrt(
        total / float(HIDDEN) + epsilon[0]);
    N stored = N(0);
    for (uint item = 0u; item < PER; ++item) {
        const uint feature = tid + item * 256u;
        if (feature < uint(HIDDEN)) {
            const N value = N(values[item] * inverse_rms * scales[item]);
            normalized_stream[feature] = value;
            if (item == group) stored = value;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float* output_parts =
        parts + stream * PART_STRIDE;
    if (group < DOWN_GROUPS) {
        const uint first_row =
            group * ROWS_PER_TG + simd_group * ROWS_PER_SIMD;
        float accumulators[ROWS_PER_SIMD] = {0.0f, 0.0f};
        for (uint feature = lane * 4u;
             feature < uint(HIDDEN);
             feature += 128u) {
            float inputs[4];
            for (uint item = 0u; item < 4u; ++item)
                inputs[item] = float(normalized_stream[feature + item]);
            for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {
                const uint weight_base =
                    (first_row + row) * uint(HIDDEN * HC_COUNT) +
                    input_base + feature;
                for (uint item = 0u; item < 4u; ++item)
                    accumulators[row] +=
                        float(down_weight[weight_base + item]) * inputs[item];
            }
        }
        for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {
            const float value = simd_sum(accumulators[row]);
            if (lane == 0u) output_parts[first_row + row] = value;
        }
    } else if (HAS_INJECTION != 0 && simd_group < uint(HC_COUNT)) {
        float accumulator = 0.0f;
        const uint weight_row = simd_group;
        for (uint feature = lane * 4u;
             feature < uint(HIDDEN);
             feature += 128u) {
            const uint weight_base =
                weight_row * uint(HIDDEN * HC_COUNT) + input_base + feature;
            for (uint item = 0u; item < 4u; ++item) {
                accumulator += float(injection_weight[weight_base + item]) *
                    float(normalized_stream[feature + item]);
            }
        }
        accumulator = simd_sum(accumulator);
        if (lane == 0u)
            output_parts[uint(LOW_RANK) + weight_row] = accumulator;
    }

    // Store after all device weight reads.  This avoids a conservative device
    // read-after-write fence in Apple's Metal compiler.
    if (group < PER) {
        const uint feature = tid + group * 256u;
        if (feature < uint(HIDDEN))
            normalized[input_base + feature] = stored;
    }
)METAL";

constexpr const char* kGatedHcWriteNormDownSource = R"METAL(
    constexpr uint STREAMS = 4u;
    constexpr uint ROWS_PER_SIMD = 2u;
    constexpr uint ROWS_PER_TG = 8u * ROWS_PER_SIMD;
    constexpr uint DOWN_GROUPS = uint(LOW_RANK) / ROWS_PER_TG;
    constexpr uint GROUPS = DOWN_GROUPS + uint(HAS_INJECTION);
    constexpr uint PER = (uint(HIDDEN) + 255u) / 256u;
    constexpr uint PART_STRIDE = uint(LOW_RANK + HC_COUNT);
    static_assert(HC_COUNT == 4 && HIDDEN == 2560 && LOW_RANK == 320,
                  "Qwen4 HC decode geometry changed");

    const uint tid = thread_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint simd_group = simdgroup_index_in_threadgroup;
    const uint workgroup = threadgroup_position_in_grid.x;
    const uint stream = workgroup & 3u;
    const uint group = workgroup >> 2u;
    const uint input_base = stream * uint(HIDDEN);

    threadgroup O normalized_stream[HIDDEN];
    threadgroup float norm_partials[8];
    float values[PER];
    float scales[PER];
    const G gate_value = previous_injection[stream];
    float sum_squares = 0.0f;
    O written = O(0);
    for (uint item = 0u; item < PER; ++item) {
        const uint feature = tid + item * 256u;
        float value = 0.0f;
        float scale = 0.0f;
        if (feature < uint(HIDDEN)) {
            U update = U(U(previous_branch[feature]) * U(gate_value));
            written = O(O(previous_residual[input_base + feature]) + O(update));
            value = float(written);
            scale = 1.0f + float(norm_weight[input_base + feature]);
        }
        values[item] = value;
        scales[item] = scale;
        sum_squares += value * value;
    }
    sum_squares = simd_sum(sum_squares);
    if (lane == 0u) norm_partials[simd_group] = sum_squares;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint part = 0u; part < 8u; ++part)
        total += norm_partials[part];
    const float inverse_rms = metal::rsqrt(
        total / float(HIDDEN) + epsilon[0]);
    O stored_norm = O(0);
    O stored_residual = O(0);
    for (uint item = 0u; item < PER; ++item) {
        const uint feature = tid + item * 256u;
        if (feature < uint(HIDDEN)) {
            U update = U(U(previous_branch[feature]) * U(gate_value));
            const O value = O(
                O(previous_residual[input_base + feature]) + O(update));
            const O normed = O(float(value) * inverse_rms * scales[item]);
            normalized_stream[feature] = normed;
            if (item == group) {
                stored_norm = normed;
                stored_residual = value;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float* output_parts = parts + stream * PART_STRIDE;
    if (group < DOWN_GROUPS) {
        const uint first_row =
            group * ROWS_PER_TG + simd_group * ROWS_PER_SIMD;
        float accumulators[ROWS_PER_SIMD] = {0.0f, 0.0f};
        for (uint feature = lane * 4u;
             feature < uint(HIDDEN);
             feature += 128u) {
            float inputs[4];
            for (uint item = 0u; item < 4u; ++item)
                inputs[item] = float(normalized_stream[feature + item]);
            for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {
                const uint weight_base =
                    (first_row + row) * uint(HIDDEN * HC_COUNT) +
                    input_base + feature;
                for (uint item = 0u; item < 4u; ++item)
                    accumulators[row] +=
                        float(down_weight[weight_base + item]) * inputs[item];
            }
        }
        for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {
            const float value = simd_sum(accumulators[row]);
            if (lane == 0u) output_parts[first_row + row] = value;
        }
    } else if (HAS_INJECTION != 0 && simd_group < uint(HC_COUNT)) {
        float accumulator = 0.0f;
        const uint weight_row = simd_group;
        for (uint feature = lane * 4u;
             feature < uint(HIDDEN);
             feature += 128u) {
            const uint weight_base =
                weight_row * uint(HIDDEN * HC_COUNT) + input_base + feature;
            for (uint item = 0u; item < 4u; ++item) {
                accumulator += float(injection_weight[weight_base + item]) *
                    float(normalized_stream[feature + item]);
            }
        }
        accumulator = simd_sum(accumulator);
        if (lane == 0u)
            output_parts[uint(LOW_RANK) + weight_row] = accumulator;
    }

    if (group < PER) {
        const uint feature = tid + group * 256u;
        if (feature < uint(HIDDEN)) {
            const uint index = input_base + feature;
            normalized[index] = stored_norm;
            residual[index] = stored_residual;
        }
    }
)METAL";

constexpr const char* kGatedHcPartsUpSource = R"METAL(
    constexpr uint FEATURES_PER_TG = 8u;
    constexpr uint CHUNK = 32u;
    constexpr uint CHUNKS = uint(LOW_RANK) / CHUNK;
    constexpr uint OUTPUT_ROWS = FEATURES_PER_TG * uint(HC_COUNT);
    constexpr uint PART_STRIDE = uint(LOW_RANK + HC_COUNT);
    constexpr uint REDUCE_STRIDE = CHUNKS + 1u;
    static_assert(HC_COUNT == 4 && LOW_RANK == 320,
                  "Qwen4 HC up geometry changed");

    const uint tid = thread_index_in_threadgroup;
    const uint chunk = tid % CHUNKS;
    const uint output_row = tid / CHUNKS;
    const uint stream = output_row & 3u;
    const uint local_feature = output_row >> 2u;
    const uint feature =
        threadgroup_position_in_grid.x * FEATURES_PER_TG + local_feature;

    threadgroup L activated[LOW_RANK];
    threadgroup float projection_partials[OUTPUT_ROWS * REDUCE_STRIDE];
    if (tid < uint(LOW_RANK)) {
        float value = parts[tid] + parts[PART_STRIDE + tid] +
            parts[2u * PART_STRIDE + tid] +
            parts[3u * PART_STRIDE + tid];
        L rounded = L(value / float(HC_COUNT));
        L tail = L(1) /
            (L(1) + metal::exp(metal::abs(rounded)));
        L sigmoid = rounded < L(0) ? tail : L(1) - tail;
        activated[tid] = L(rounded * sigmoid);
    }
    if (HAS_INJECTION != 0 && tid < uint(HC_COUNT)) {
        const uint index = uint(LOW_RANK) + tid;
        float value = parts[index] + parts[PART_STRIDE + index] +
            parts[2u * PART_STRIDE + index] +
            parts[3u * PART_STRIDE + index];
        I rounded = I(value / float(HC_COUNT));
        I tail = I(1) /
            (I(1) + metal::exp(metal::abs(rounded)));
        I sigmoid = rounded < I(0) ? tail : I(1) - tail;
        injection[tid] = I(I(2) * sigmoid);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float accumulator = 0.0f;
    const uint low_base = chunk * CHUNK;
    if (feature < uint(HIDDEN)) {
        const uint row = stream * uint(HIDDEN) + feature;
        const uint weight_base = row * uint(LOW_RANK) + low_base;
        for (uint item = 0u; item < CHUNK; ++item) {
            accumulator += float(up_weight[weight_base + item]) *
                float(activated[low_base + item]);
        }
    }
    projection_partials[output_row * REDUCE_STRIDE + chunk] = accumulator;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < OUTPUT_ROWS) {
        const uint result_stream = tid & 3u;
        const uint result_feature =
            threadgroup_position_in_grid.x * FEATURES_PER_TG + (tid >> 2u);
        float projected = 0.0f;
        const uint base = tid * REDUCE_STRIDE;
        for (uint current = 0u; current < CHUNKS; ++current)
            projected += projection_partials[base + current];
        T rounded = T(projected);
        T tail = T(1) /
            (T(1) + metal::exp(metal::abs(rounded)));
        T gate = rounded < T(0) ? tail : T(1) - tail;
        T product = T(gate * T(normalized[
            result_stream * uint(HIDDEN) + result_feature]));
        float mixed = float(product);
        mixed += simd_shuffle_down(mixed, 1);
        mixed += simd_shuffle_down(mixed, 2);
        if (result_stream == 0u && result_feature < uint(HIDDEN))
            branch[result_feature] = T(mixed / float(HC_COUNT));
    }
)METAL";

const mlx::core::fast::CustomKernelFunction& grouped_rms_affine_kernel() {
    static const auto kernel = [] {
        CompileOptions options;
        options.math_mode = MathMode::Safe;
        return mlx::core::fast::metal_kernel(
            "mfq_cpp_qwen_grouped_rms_affine",
            {"input", "weight", "epsilon"},
            {"output"},
            kGroupedRmsAffineSource,
            "",
            true,
            false,
            options);
    }();
    return kernel;
}

const mlx::core::fast::CustomKernelFunction&
grouped_rms_affine_write_kernel() {
    static const auto kernel = [] {
        CompileOptions options;
        options.math_mode = MathMode::Safe;
        return mlx::core::fast::metal_kernel(
            "mfq_cpp_qwen_grouped_rms_affine_write",
            {
                "residual",
                "branch",
                "injection",
                "weight",
                "epsilon",
            },
            {"updated", "normalized"},
            kGroupedRmsAffineWriteSource,
            "",
            true,
            false,
            options);
    }();
    return kernel;
}

const mlx::core::fast::CustomKernelFunction& qsa_decode_prologue_kernel() {
    static const auto kernel = [] {
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        return mlx::core::fast::metal_kernel(
            "mfq_cpp_qwen4_qsa_decode_prologue",
            {
                "query_gate_input",
                "key_input",
                "index_query_key_input",
                "query_weight",
                "key_weight",
                "index_weight",
                "positions",
                "params",
            },
            {
                "query_output",
                "output_gate",
                "key_output",
                "index_query_output",
            },
            kQsaDecodePrologueSource,
            "",
            true,
            false,
            options);
    }();
    return kernel;
}

const mlx::core::fast::CustomKernelFunction& gated_hc_post_kernel() {
    static const auto kernel = [] {
        CompileOptions options;
        options.math_mode = MathMode::Safe;
        return mlx::core::fast::metal_kernel(
            "mfq_cpp_qwen_gated_hc_post",
            {"branch", "residual", "injection"},
            {"output"},
            R"METAL(
                uint index = thread_position_in_grid.x;
                if (index >= uint(SIZE)) return;
                uint row = index / uint(HIDDEN * HC_COUNT);
                uint stream = (index / uint(HIDDEN)) % uint(HC_COUNT);
                uint feature = index % uint(HIDDEN);
                // Preserve the multiply's promoted rounding before the add;
                // do not replace this with a fused multiply-add.
                U update = U(U(branch[row * uint(HIDDEN) + feature]) *
                    U(injection[row * uint(HC_COUNT) + stream]));
                output[index] = T(T(residual[index]) + T(update));
            )METAL",
            "",
            true,
            false,
            options);
    }();
    return kernel;
}

const mlx::core::fast::CustomKernelFunction& gated_hc_projection_kernel() {
    static const auto kernel = [] {
        CompileOptions options;
        options.math_mode = MathMode::Safe;
        return mlx::core::fast::metal_kernel(
            "mfq_cpp_qwen_gated_hc_projection",
            {"normalized", "down_weight", "injection_weight"},
            {"down", "injection"},
            kGatedHcProjectionSource,
            "",
            true,
            false,
            options);
    }();
    return kernel;
}

const mlx::core::fast::CustomKernelFunction& gated_hc_up_collapse_kernel() {
    static const auto kernel = [] {
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        return mlx::core::fast::metal_kernel(
            "mfq_cpp_qwen_gated_hc_up_collapse",
            {"down_projection", "up_weight", "normalized"},
            {"branch"},
            kGatedHcUpCollapseSource,
            "",
            true,
            false,
            options);
    }();
    return kernel;
}

const mlx::core::fast::CustomKernelFunction& gated_hc_norm_down_kernel() {
    static const auto kernel = [] {
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        return mlx::core::fast::metal_kernel(
            "mfq_cpp_qwen_gated_hc_norm_down",
            {
                "input",
                "norm_weight",
                "down_weight",
                "injection_weight",
                "epsilon",
            },
            {"normalized", "parts"},
            kGatedHcNormDownSource,
            "",
            true,
            false,
            options);
    }();
    return kernel;
}

const mlx::core::fast::CustomKernelFunction&
gated_hc_write_norm_down_kernel() {
    static const auto kernel = [] {
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        return mlx::core::fast::metal_kernel(
            "mfq_cpp_qwen_gated_hc_write_norm_down",
            {
                "previous_residual",
                "previous_branch",
                "previous_injection",
                "norm_weight",
                "down_weight",
                "injection_weight",
                "epsilon",
            },
            {"residual", "normalized", "parts"},
            kGatedHcWriteNormDownSource,
            "",
            true,
            false,
            options);
    }();
    return kernel;
}

const mlx::core::fast::CustomKernelFunction& gated_hc_parts_up_kernel() {
    static const auto kernel = [] {
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        return mlx::core::fast::metal_kernel(
            "mfq_cpp_qwen_gated_hc_parts_up",
            {"normalized", "parts", "up_weight"},
            {"branch", "injection"},
            kGatedHcPartsUpSource,
            "",
            true,
            false,
            options);
    }();
    return kernel;
}

bool gated_hc_fast_path_enabled() noexcept {
    static const bool enabled = [] {
        const char* setting =
            std::getenv("MFQ_METAL_QWEN_GATED_HC_FAST");
        return setting == nullptr || std::atoi(setting) != 0;
    }();
    return enabled;
}

bool gated_hc_float_dtype(mlx::core::Dtype dtype) noexcept {
    return dtype == mlx::core::float16 ||
        dtype == mlx::core::bfloat16 || dtype == mlx::core::float32;
}

bool can_fuse_grouped_rms_affine(
    const array& value,
    const array& weight,
    int group_size) {
    return gated_hc_fast_path_enabled() &&
        gated_hc_float_dtype(value.dtype()) &&
        gated_hc_float_dtype(weight.dtype()) &&
        value.ndim() >= 1 &&
        group_size > 0 && group_size <= 4096 &&
        value.shape(-1) % group_size == 0 &&
        weight.ndim() == 1 &&
        weight.shape(0) == value.shape(-1);
}

array fused_grouped_rms_affine(
    const array& value,
    const array& weight,
    int group_size,
    float eps) {
    constexpr int values_per_thread = 4;
    constexpr int simd_size = 32;
    const int threads_needed =
        (group_size + values_per_thread - 1) / values_per_thread;
    const int simdgroups =
        (threads_needed + simd_size - 1) / simd_size;
    const int threads = simdgroups * simd_size;
    const int rows = static_cast<int>(value.size() / group_size);
    const int group_count = value.shape(-1) / group_size;
    const array epsilon({eps}, Shape{1});
    auto outputs = grouped_rms_affine_kernel()(
        {value, weight, epsilon},
        {value.shape()},
        {value.dtype()},
        {rows * threads, 1, 1},
        {threads, 1, 1},
        {
            {"T", value.dtype()},
            {"GROUP_SIZE", group_size},
            {"GROUP_COUNT", group_count},
        },
        std::nullopt,
        false,
        {});
    return std::move(outputs.front());
}

struct GatedHcWriteNorm {
    array residual;
    array normalized;
};

bool can_fuse_grouped_rms_affine_write(
    const array& branch,
    const array& residual,
    const array& injection,
    const array& weight,
    int group_size,
    int hc_count) {
    if (!gated_hc_fast_path_enabled() ||
        !gated_hc_float_dtype(branch.dtype()) ||
        !gated_hc_float_dtype(residual.dtype()) ||
        !gated_hc_float_dtype(injection.dtype()) ||
        !gated_hc_float_dtype(weight.dtype()) ||
        branch.ndim() < 2 || residual.ndim() != branch.ndim() ||
        injection.ndim() != branch.ndim() ||
        group_size <= 0 || group_size > 4096 || hc_count <= 1 ||
        residual.shape(-1) != group_size * hc_count ||
        weight.shape() != Shape{group_size * hc_count}) {
        return false;
    }
    auto branch_shape = residual.shape();
    branch_shape.back() = group_size;
    auto injection_shape = residual.shape();
    injection_shape.back() = hc_count;
    return branch.shape() == branch_shape &&
        injection.shape() == injection_shape &&
        residual.size() > 0 &&
        residual.size() <= static_cast<std::size_t>(
            std::numeric_limits<int>::max());
}

GatedHcWriteNorm fused_grouped_rms_affine_write(
    const array& branch,
    const array& residual,
    const array& injection,
    const array& weight,
    int group_size,
    int hc_count,
    float eps) {
    constexpr int values_per_thread = 4;
    constexpr int simd_size = 32;
    const int threads_needed =
        (group_size + values_per_thread - 1) / values_per_thread;
    const int simdgroups =
        (threads_needed + simd_size - 1) / simd_size;
    const int threads = simdgroups * simd_size;
    const int rows = static_cast<int>(residual.size() / group_size);
    const auto update_dtype = mlx::core::promote_types(
        branch.dtype(), injection.dtype());
    const auto output_dtype = mlx::core::promote_types(
        residual.dtype(), update_dtype);
    const array epsilon({eps}, Shape{1});
    auto outputs = grouped_rms_affine_write_kernel()(
        {residual, branch, injection, weight, epsilon},
        {residual.shape(), residual.shape()},
        {output_dtype, output_dtype},
        {rows * threads, 1, 1},
        {threads, 1, 1},
        {
            {"R", residual.dtype()},
            {"B", branch.dtype()},
            {"G", injection.dtype()},
            {"W", weight.dtype()},
            {"U", update_dtype},
            {"O", output_dtype},
            {"GROUP_SIZE", group_size},
            {"HC_COUNT", hc_count},
        },
        std::nullopt,
        false,
        {});
    return {
        std::move(outputs[0]),
        std::move(outputs[1]),
    };
}

bool can_fuse_gated_hc_projections(
    const array& normalized,
    const array& down_weight,
    const array& injection_weight,
    int hc_count) {
    if (!gated_hc_fast_path_enabled() ||
        !gated_hc_float_dtype(normalized.dtype()) ||
        normalized.ndim() != 3 ||
        normalized.shape(0) != 1 ||
        normalized.shape(1) != 1 ||
        down_weight.ndim() != 2 ||
        injection_weight.ndim() != 2 ||
        !gated_hc_float_dtype(down_weight.dtype()) ||
        !gated_hc_float_dtype(injection_weight.dtype()) ||
        hc_count != 4) {
        return false;
    }
    const int width = normalized.shape(2);
    const int low_rank = down_weight.shape(0);
    return width > 0 && width % 1024 == 0 &&
        low_rank > 0 && low_rank % 4 == 0 &&
        down_weight.shape(1) == width &&
        injection_weight.shape() == Shape{hc_count, width} &&
        width >= 16 * low_rank;
}

struct GatedHcProjections {
    array down;
    array injection;
};

GatedHcProjections fused_gated_hc_projections(
    const array& normalized,
    const array& down_weight,
    const array& injection_weight,
    int hc_count) {
    const int width = normalized.shape(2);
    const int low_rank = down_weight.shape(0);
    const int workgroups = low_rank / 4 + hc_count / 4;
    // Keep storage types independent and match matmul's promotion without
    // materializing full FP32 copies of the BF16/FP16 projection matrices.
    const auto down_dtype = mlx::core::promote_types(
        normalized.dtype(), down_weight.dtype());
    const auto injection_dtype = mlx::core::promote_types(
        normalized.dtype(), injection_weight.dtype());
    auto outputs = gated_hc_projection_kernel()(
        {normalized, down_weight, injection_weight},
        {Shape{1, 1, low_rank}, Shape{1, 1, hc_count}},
        {down_dtype, injection_dtype},
        {workgroups * kGatedHcProjectionThreads, 1, 1},
        {kGatedHcProjectionThreads, 1, 1},
        {
            {"D", down_dtype},
            {"I", injection_dtype},
            {"WIDTH", width},
            {"LOW_RANK", low_rank},
            {"HC_COUNT", hc_count},
        },
        std::nullopt,
        false,
        {});
    return {
        std::move(outputs[0]),
        std::move(outputs[1]),
    };
}

bool can_fuse_gated_hc_up_collapse(
    const array& down_projection,
    const array& up_weight,
    const array& normalized,
    int hidden_size,
    int hc_count) {
    return gated_hc_fast_path_enabled() &&
        gated_hc_float_dtype(normalized.dtype()) &&
        gated_hc_float_dtype(down_projection.dtype()) &&
        gated_hc_float_dtype(up_weight.dtype()) &&
        normalized.ndim() == 3 &&
        normalized.shape(0) == 1 &&
        normalized.shape(1) == 1 &&
        normalized.shape(2) == hidden_size * hc_count &&
        down_projection.ndim() == 3 &&
        down_projection.shape(0) == 1 &&
        down_projection.shape(1) == 1 &&
        down_projection.shape(2) == 320 &&
        up_weight.ndim() == 2 &&
        up_weight.shape(0) == hidden_size * hc_count &&
        up_weight.shape(1) == down_projection.shape(2) &&
        hc_count == 4;
}

array fused_gated_hc_up_collapse(
    const array& down_projection,
    const array& up_weight,
    const array& normalized,
    int hidden_size,
    int hc_count) {
    const int workgroups = (hidden_size + 7) / 8;
    const auto dtype = mlx::core::promote_types(
        down_projection.dtype(), up_weight.dtype());
    auto outputs = gated_hc_up_collapse_kernel()(
        {down_projection, up_weight, normalized},
        {Shape{1, 1, hidden_size}},
        {dtype},
        {workgroups * kGatedHcUpThreads, 1, 1},
        {kGatedHcUpThreads, 1, 1},
        {
            {"T", dtype},
            {"L", down_projection.dtype()},
            {"HIDDEN", hidden_size},
            {"LOW_RANK", down_projection.shape(2)},
            {"HC_COUNT", hc_count},
        },
        std::nullopt,
        false,
        {});
    return std::move(outputs.front());
}

bool can_fuse_gated_hc_two_stage(
    const array& input,
    const array& norm_weight,
    const array& down_weight,
    const array& up_weight,
    const std::optional<array>& injection_weight,
    int hidden_size,
    int hc_count) {
    if (!gated_hc_fast_path_enabled() ||
        hidden_size != 2560 || hc_count != 4 ||
        input.ndim() != 3 || input.shape() != Shape{1, 1, 10240} ||
        norm_weight.shape() != Shape{10240} ||
        down_weight.shape() != Shape{320, 10240} ||
        up_weight.shape() != Shape{10240, 320} ||
        !gated_hc_float_dtype(input.dtype()) ||
        !gated_hc_float_dtype(norm_weight.dtype()) ||
        !gated_hc_float_dtype(down_weight.dtype()) ||
        !gated_hc_float_dtype(up_weight.dtype())) {
        return false;
    }
    return !injection_weight ||
        (injection_weight->shape() == Shape{4, 10240} &&
         gated_hc_float_dtype(injection_weight->dtype()));
}

struct GatedHcNormDown {
    array normalized;
    array parts;
};

GatedHcNormDown fused_gated_hc_norm_down(
    const array& input,
    const array& norm_weight,
    const array& down_weight,
    const std::optional<array>& injection_weight,
    float eps) {
    constexpr int hidden_size = 2560;
    constexpr int hc_count = 4;
    constexpr int low_rank = 320;
    constexpr int rows_per_group = 16;
    const bool has_injection = injection_weight.has_value();
    const int groups = low_rank / rows_per_group +
        static_cast<int>(has_injection);
    const array epsilon({eps}, Shape{1});
    const array& injection = has_injection
        ? *injection_weight : down_weight;
    auto outputs = gated_hc_norm_down_kernel()(
        {input, norm_weight, down_weight, injection, epsilon},
        {input.shape(), Shape{1, hc_count, low_rank + hc_count}},
        {input.dtype(), mlx::core::float32},
        {groups * hc_count * kGatedHcProjectionThreads, 1, 1},
        {kGatedHcProjectionThreads, 1, 1},
        {
            {"N", input.dtype()},
            {"HIDDEN", hidden_size},
            {"LOW_RANK", low_rank},
            {"HC_COUNT", hc_count},
            {"HAS_INJECTION", static_cast<int>(has_injection)},
        },
        std::nullopt,
        false,
        {});
    return {std::move(outputs[0]), std::move(outputs[1])};
}

struct GatedHcWriteNormDown {
    array residual;
    array normalized;
    array parts;
};

GatedHcWriteNormDown fused_gated_hc_write_norm_down(
    const array& previous_branch,
    const array& previous_residual,
    const array& previous_injection,
    const array& norm_weight,
    const array& down_weight,
    const std::optional<array>& injection_weight,
    float eps) {
    constexpr int hidden_size = 2560;
    constexpr int hc_count = 4;
    constexpr int low_rank = 320;
    constexpr int rows_per_group = 16;
    const bool has_injection = injection_weight.has_value();
    const int groups = low_rank / rows_per_group +
        static_cast<int>(has_injection);
    const auto update_dtype = mlx::core::promote_types(
        previous_branch.dtype(), previous_injection.dtype());
    const auto output_dtype = mlx::core::promote_types(
        previous_residual.dtype(), update_dtype);
    const array epsilon({eps}, Shape{1});
    const array& injection = has_injection
        ? *injection_weight : down_weight;
    auto outputs = gated_hc_write_norm_down_kernel()(
        {
            previous_residual,
            previous_branch,
            previous_injection,
            norm_weight,
            down_weight,
            injection,
            epsilon,
        },
        {
            previous_residual.shape(),
            previous_residual.shape(),
            Shape{1, hc_count, low_rank + hc_count},
        },
        {output_dtype, output_dtype, mlx::core::float32},
        {groups * hc_count * kGatedHcProjectionThreads, 1, 1},
        {kGatedHcProjectionThreads, 1, 1},
        {
            {"G", previous_injection.dtype()},
            {"U", update_dtype},
            {"O", output_dtype},
            {"HIDDEN", hidden_size},
            {"LOW_RANK", low_rank},
            {"HC_COUNT", hc_count},
            {"HAS_INJECTION", static_cast<int>(has_injection)},
        },
        std::nullopt,
        false,
        {});
    return {
        std::move(outputs[0]),
        std::move(outputs[1]),
        std::move(outputs[2]),
    };
}

struct GatedHcPartsUp {
    array branch;
    array injection;
};

GatedHcPartsUp fused_gated_hc_parts_up(
    const array& normalized,
    const array& parts,
    const array& down_weight,
    const array& up_weight,
    const std::optional<array>& injection_weight) {
    constexpr int hidden_size = 2560;
    constexpr int hc_count = 4;
    constexpr int low_rank = 320;
    const bool has_injection = injection_weight.has_value();
    const auto down_dtype = mlx::core::promote_types(
        normalized.dtype(), down_weight.dtype());
    const auto branch_dtype = mlx::core::promote_types(
        down_dtype, up_weight.dtype());
    const auto injection_dtype = has_injection
        ? mlx::core::promote_types(
              normalized.dtype(), injection_weight->dtype())
        : normalized.dtype();
    const int workgroups = hidden_size / 8;
    auto outputs = gated_hc_parts_up_kernel()(
        {normalized, parts, up_weight},
        {Shape{1, 1, hidden_size}, Shape{1, 1, hc_count}},
        {branch_dtype, injection_dtype},
        {workgroups * kGatedHcUpThreads, 1, 1},
        {kGatedHcUpThreads, 1, 1},
        {
            {"T", branch_dtype},
            {"L", down_dtype},
            {"I", injection_dtype},
            {"HIDDEN", hidden_size},
            {"LOW_RANK", low_rank},
            {"HC_COUNT", hc_count},
            {"HAS_INJECTION", static_cast<int>(has_injection)},
        },
        std::nullopt,
        false,
        {});
    return {std::move(outputs[0]), std::move(outputs[1])};
}

MlxQwen4GatedResidualPre fused_gated_hc_two_stage(
    const array& input,
    const array& norm_weight,
    const array& down_weight,
    const array& up_weight,
    const std::optional<array>& injection_weight,
    float eps) {
    auto first = fused_gated_hc_norm_down(
        input, norm_weight, down_weight, injection_weight, eps);
    auto second = fused_gated_hc_parts_up(
        first.normalized,
        first.parts,
        down_weight,
        up_weight,
        injection_weight);
    std::optional<array> injection;
    if (injection_weight) injection = std::move(second.injection);
    return {
        std::move(second.branch),
        input,
        std::move(injection),
    };
}

MlxQwen4GatedResidualPre fused_gated_hc_two_stage_after(
    const array& previous_branch,
    const array& previous_residual,
    const array& previous_injection,
    const array& norm_weight,
    const array& down_weight,
    const array& up_weight,
    const std::optional<array>& injection_weight,
    float eps) {
    auto first = fused_gated_hc_write_norm_down(
        previous_branch,
        previous_residual,
        previous_injection,
        norm_weight,
        down_weight,
        injection_weight,
        eps);
    auto second = fused_gated_hc_parts_up(
        first.normalized,
        first.parts,
        down_weight,
        up_weight,
        injection_weight);
    std::optional<array> injection;
    if (injection_weight) injection = std::move(second.injection);
    return {
        std::move(second.branch),
        std::move(first.residual),
        std::move(injection),
    };
}

MlxQwen4GatedResidualPre gated_hc_from_normalized(
    array normalized,
    array residual,
    const array& down_weight,
    const array& up_weight,
    const std::optional<array>& inject_weight,
    int hidden_size,
    int hc_count) {
    const int width = hidden_size * hc_count;
    std::optional<array> fused_injection;
    array down_projection = [&] {
        if (inject_weight && can_fuse_gated_hc_projections(
                normalized,
                down_weight,
                *inject_weight,
                hc_count)) {
            auto projections = fused_gated_hc_projections(
                normalized,
                down_weight,
                *inject_weight,
                hc_count);
            fused_injection = std::move(projections.injection);
            return std::move(projections.down);
        }
        return mlx::core::matmul(
            normalized, mlx::core::transpose(down_weight));
    }();
    const array connection_count(
        static_cast<float>(hc_count), normalized.dtype());
    array mixed = [&] {
        if (can_fuse_gated_hc_up_collapse(
                down_projection,
                up_weight,
                normalized,
                hidden_size,
                hc_count)) {
            return fused_gated_hc_up_collapse(
                down_projection,
                up_weight,
                normalized,
                hidden_size,
                hc_count);
        }
        auto low = down_projection / connection_count;
        low = low * mlx::core::sigmoid(low);
        auto mixing = mlx::core::sigmoid(
            mlx::core::matmul(low, mlx::core::transpose(up_weight)));
        auto stream_shape = residual.shape();
        stream_shape.back() = hc_count;
        stream_shape.push_back(hidden_size);
        return mlx::core::mean(
            mlx::core::reshape(mixing, stream_shape) *
                mlx::core::reshape(normalized, stream_shape),
            -2);
    }();
    std::optional<array> injection;
    if (inject_weight) {
        if (inject_weight->ndim() != 2 ||
            inject_weight->shape() != Shape{hc_count, width}) {
            throw std::invalid_argument("Qwen4 residual injection mismatch");
        }
        if (fused_injection) {
            injection = std::move(*fused_injection);
        } else {
            auto projected = mlx::core::matmul(
                normalized, mlx::core::transpose(*inject_weight));
            injection = array(2.0f, normalized.dtype()) *
                mlx::core::sigmoid(projected / connection_count);
        }
    }
    return {
        std::move(mixed),
        std::move(residual),
        std::move(injection),
    };
}

array floating(const array& value, mlx::core::Dtype dtype) {
    auto result = value.dtype() == dtype ? value : mlx::core::astype(value, dtype);
    return mlx::core::contiguous(result);
}

void require_rank(const array& value, int rank, const char* name) {
    if (value.ndim() != rank) {
        throw std::invalid_argument(std::string(name) + " has an invalid rank");
    }
}

} // namespace

array qwen4_grouped_rms_norm(
    const array& value,
    const array& weight,
    int group_size,
    float eps) {
    if (value.ndim() == 0 || group_size <= 0 ||
        value.shape(-1) % group_size != 0 ||
        weight.ndim() != 1 || weight.shape(0) != value.shape(-1) ||
        !std::isfinite(eps) || eps <= 0.0f) {
        throw std::invalid_argument("Qwen4 grouped RMSNorm dimensions disagree");
    }
    if (can_fuse_grouped_rms_affine(value, weight, group_size)) {
        return fused_grouped_rms_affine(
            value, weight, group_size, eps);
    }
    const auto source_dtype = value.dtype();
    auto shape = value.shape();
    const int width = shape.back();
    shape.pop_back();
    shape.push_back(width / group_size);
    shape.push_back(group_size);
    auto source = mlx::core::reshape(
        mlx::core::astype(value, mlx::core::float32), shape);
    // Treat every hyper-connection stream as an independent RMSNorm row.
    // MLX's fast primitive keeps the reduction and normalization in one
    // dispatch; the per-stream centred weights remain a separate affine so
    // all groups can retain their own parameters.
    auto normalized = mlx::core::fast::rms_norm(
        source, std::nullopt, eps);
    normalized = mlx::core::reshape(normalized, value.shape()) *
        (array(1.0f) + mlx::core::astype(weight, mlx::core::float32));
    return normalized.dtype() == source_dtype
        ? normalized : mlx::core::astype(normalized, source_dtype);
}

MlxQwen4GatedResidualPre qwen4_gated_residual_pre(
    const array& hyper_input,
    const array& norm_weight,
    const array& down_weight,
    const array& up_weight,
    const std::optional<array>& inject_weight,
    int hidden_size,
    int hc_count,
    float eps) {
    const int width = hidden_size * hc_count;
    if (hidden_size <= 0 || hc_count <= 1 || hyper_input.ndim() < 2 ||
        hyper_input.shape(-1) != width ||
        down_weight.ndim() != 2 || down_weight.shape(1) != width ||
        up_weight.ndim() != 2 || up_weight.shape(0) != width ||
        up_weight.shape(1) != down_weight.shape(0)) {
        throw std::invalid_argument("Qwen4 gated-residual projection mismatch");
    }
    if (std::isfinite(eps) && eps > 0.0f &&
        can_fuse_gated_hc_two_stage(
            hyper_input,
            norm_weight,
            down_weight,
            up_weight,
            inject_weight,
            hidden_size,
            hc_count)) {
        return fused_gated_hc_two_stage(
            hyper_input,
            norm_weight,
            down_weight,
            up_weight,
            inject_weight,
            eps);
    }
    auto normalized = qwen4_grouped_rms_norm(
        hyper_input, norm_weight, hidden_size, eps);
    return gated_hc_from_normalized(
        std::move(normalized),
        hyper_input,
        down_weight,
        up_weight,
        inject_weight,
        hidden_size,
        hc_count);
}

MlxQwen4GatedResidualPre qwen4_gated_residual_pre_after(
    const array& previous_branch,
    const array& previous_residual,
    const array& previous_injection,
    const array& norm_weight,
    const array& down_weight,
    const array& up_weight,
    const std::optional<array>& inject_weight,
    int hidden_size,
    int hc_count,
    float eps) {
    const int width = hidden_size * hc_count;
    if (hidden_size <= 0 || hc_count <= 1 ||
        previous_residual.ndim() < 2 ||
        previous_residual.shape(-1) != width ||
        norm_weight.shape() != Shape{width} ||
        down_weight.ndim() != 2 || down_weight.shape(1) != width ||
        up_weight.ndim() != 2 || up_weight.shape(0) != width ||
        up_weight.shape(1) != down_weight.shape(0) ||
        !std::isfinite(eps) || eps <= 0.0f) {
        throw std::invalid_argument(
            "Qwen4 chained gated-residual projection mismatch");
    }

    if (can_fuse_gated_hc_two_stage(
            previous_residual,
            norm_weight,
            down_weight,
            up_weight,
            inject_weight,
            hidden_size,
            hc_count) &&
        previous_branch.shape() == Shape{1, 1, hidden_size} &&
        previous_injection.shape() == Shape{1, 1, hc_count} &&
        gated_hc_float_dtype(previous_branch.dtype()) &&
        gated_hc_float_dtype(previous_injection.dtype())) {
        return fused_gated_hc_two_stage_after(
            previous_branch,
            previous_residual,
            previous_injection,
            norm_weight,
            down_weight,
            up_weight,
            inject_weight,
            eps);
    }

    array updated = previous_residual;
    array normalized = previous_residual;
    if (can_fuse_grouped_rms_affine_write(
            previous_branch,
            previous_residual,
            previous_injection,
            norm_weight,
            hidden_size,
            hc_count)) {
        auto result = fused_grouped_rms_affine_write(
            previous_branch,
            previous_residual,
            previous_injection,
            norm_weight,
            hidden_size,
            hc_count,
            eps);
        updated = std::move(result.residual);
        normalized = std::move(result.normalized);
    } else {
        updated = qwen4_gated_residual_post(
            previous_branch,
            previous_residual,
            previous_injection,
            hc_count);
        normalized = qwen4_grouped_rms_norm(
            updated,
            norm_weight,
            hidden_size,
            eps);
    }
    return gated_hc_from_normalized(
        std::move(normalized),
        std::move(updated),
        down_weight,
        up_weight,
        inject_weight,
        hidden_size,
        hc_count);
}

array qwen4_gated_residual_post(
    const array& branch,
    const array& residual,
    const array& injection,
    int hc_count) {
    if (branch.ndim() < 2 || hc_count <= 1 ||
        residual.shape(-1) != hc_count * branch.shape(-1)) {
        throw std::invalid_argument("Qwen4 gated-residual post mismatch");
    }
    auto expected = branch.shape();
    expected.back() = hc_count;
    if (injection.shape() != expected) {
        throw std::invalid_argument("Qwen4 gated-residual gate mismatch");
    }
    if (gated_hc_fast_path_enabled() &&
        gated_hc_float_dtype(branch.dtype()) &&
        gated_hc_float_dtype(residual.dtype()) &&
        gated_hc_float_dtype(injection.dtype()) &&
        residual.size() == branch.size() * static_cast<std::size_t>(hc_count) &&
        residual.size() > 0 &&
        residual.size() <= static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        const auto update_dtype = mlx::core::promote_types(
            branch.dtype(), injection.dtype());
        const auto dtype = mlx::core::promote_types(residual.dtype(), update_dtype);
        auto outputs = gated_hc_post_kernel()(
            {branch, residual, injection},
            {residual.shape()},
            {dtype},
            {static_cast<int>(residual.size()), 1, 1},
            {256, 1, 1},
            {
                {"T", dtype},
                {"U", update_dtype},
                {"SIZE", static_cast<int>(residual.size())},
                {"HIDDEN", branch.shape(-1)},
                {"HC_COUNT", hc_count},
            },
            std::nullopt,
            false,
            {});
        return std::move(outputs.front());
    }
    auto update = mlx::core::expand_dims(branch, -2) *
        mlx::core::expand_dims(injection, -1);
    return residual + mlx::core::reshape(update, residual.shape());
}

MlxQwen4QsaDecodePrologue qwen4_qsa_decode_prologue(
    const array& query_gate,
    const array& key,
    const array& index_query_key,
    const array& query_norm_weight,
    const array& key_norm_weight,
    const array& index_query_norm_weight,
    const array& positions,
    int query_heads,
    int key_heads,
    int index_heads,
    int head_dimension,
    int index_dimension,
    int rotary_dimension,
    float rope_theta,
    float eps) {
    const auto dtype = query_gate.dtype();
    if ((dtype != mlx::core::float16 &&
         dtype != mlx::core::bfloat16 &&
         dtype != mlx::core::float32) ||
        key.dtype() != dtype || index_query_key.dtype() != dtype ||
        query_heads <= 0 || key_heads <= 0 || index_heads <= 0 ||
        head_dimension <= 0 || index_dimension <= 0 ||
        head_dimension > 256 || index_dimension > 256 ||
        rotary_dimension <= 0 || rotary_dimension % 2 != 0 ||
        rotary_dimension > std::min(head_dimension, index_dimension) ||
        !std::isfinite(rope_theta) || rope_theta <= 0.0f ||
        !std::isfinite(eps) || eps <= 0.0f ||
        query_gate.shape() != Shape{1, 1, query_heads * 2 * head_dimension} ||
        key.shape() != Shape{1, 1, key_heads * head_dimension} ||
        index_query_key.shape() !=
            Shape{1, 1, (index_heads + 1) * index_dimension} ||
        query_norm_weight.dtype() != mlx::core::float32 ||
        query_norm_weight.shape() != Shape{head_dimension} ||
        key_norm_weight.dtype() != mlx::core::float32 ||
        key_norm_weight.shape() != Shape{head_dimension} ||
        index_query_norm_weight.dtype() != mlx::core::float32 ||
        index_query_norm_weight.shape() != Shape{index_dimension} ||
        positions.dtype() != mlx::core::int32 ||
        positions.shape() != Shape{1}) {
        throw std::invalid_argument(
            "Qwen4 QSA decode prologue geometry disagrees");
    }
    const array params({eps, rope_theta}, Shape{2});
    auto outputs = qsa_decode_prologue_kernel()(
        {
            mlx::core::contiguous(query_gate),
            mlx::core::contiguous(key),
            mlx::core::contiguous(index_query_key),
            query_norm_weight,
            key_norm_weight,
            index_query_norm_weight,
            positions,
            params,
        },
        {
            Shape{1, query_heads, 1, head_dimension},
            Shape{1, 1, query_heads * head_dimension},
            Shape{1, key_heads, 1, head_dimension},
            Shape{1, 1, index_heads, index_dimension},
        },
        {dtype, dtype, dtype, dtype},
        {(query_heads + key_heads + index_heads) * 64, 1, 1},
        {64, 1, 1},
        {
            {"T", dtype},
            {"QUERY_HEADS", query_heads},
            {"KEY_HEADS", key_heads},
            {"INDEX_HEADS", index_heads},
            {"HEAD_DIM", head_dimension},
            {"INDEX_DIM", index_dimension},
            {"ROTARY_DIM", rotary_dimension},
        },
        std::nullopt,
        false,
        {});
    return {
        std::move(outputs.at(0)),
        std::move(outputs.at(1)),
        std::move(outputs.at(2)),
        std::move(outputs.at(3)),
    };
}

array qwen4_qsa_block_scores(
    const array& query,
    const array& pooled_keys) {
    require_rank(query, 4, "Qwen4 index query");
    require_rank(pooled_keys, 3, "Qwen4 pooled key");
    if (query.shape(0) != pooled_keys.shape(0) ||
        query.shape(3) != pooled_keys.shape(2)) {
        throw std::invalid_argument("Qwen4 QSA score dimensions disagree");
    }
    auto keys = mlx::core::expand_dims(
        mlx::core::transpose(
            mlx::core::astype(pooled_keys, mlx::core::float32), {0, 2, 1}),
        1);
    auto scores = mlx::core::matmul(
        mlx::core::astype(query, mlx::core::float32), keys);
    scores = mlx::core::maximum(scores, array(0.0f));
    return mlx::core::sum(scores, -2) /
        std::sqrt(static_cast<float>(query.shape(3)));
}

array qwen4_dense_gqa_attention(
    const array& query,
    const array& key,
    const array& value,
    int query_offset) {
    require_rank(query, 4, "Qwen4 attention query");
    require_rank(key, 4, "Qwen4 attention key");
    if (value.shape() != key.shape() || query.shape(0) != key.shape(0) ||
        query.shape(1) % key.shape(1) != 0 ||
        query.shape(3) != key.shape(3) || query_offset < 0 ||
        query_offset + query.shape(2) > key.shape(2)) {
        throw std::invalid_argument("Qwen4 dense GQA dimensions disagree");
    }
    const int tokens = query.shape(2);
    const int keys = key.shape(2);
    if (tokens == 1 && query_offset + 1 == keys) {
        auto output = scaled_dot_product_attention(
            query,
            key,
            value,
            false,
            1.0f / std::sqrt(static_cast<float>(query.shape(3))));
        return mlx::core::transpose(output, {0, 2, 1, 3});
    }
    auto key_positions = mlx::core::reshape(
        mlx::core::arange(0, keys, 1, mlx::core::int32), Shape{1, keys});
    auto query_positions = mlx::core::reshape(
        mlx::core::arange(
            query_offset, query_offset + tokens, 1, mlx::core::int32),
        Shape{tokens, 1});
    auto mask = mlx::core::expand_dims(
        mlx::core::expand_dims(key_positions <= query_positions, 0), 0);
    auto output = scaled_dot_product_attention(
        query, key, value, false,
        1.0f / std::sqrt(static_cast<float>(query.shape(3))), mask);
    return mlx::core::transpose(output, {0, 2, 1, 3});
}

} // namespace mfq::metal
