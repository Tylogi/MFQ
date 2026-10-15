#include "quant.h"

#include "ops/quant/cpu_projection_rows.h"
#include "kernels/activation_ops.h"
#include "kernels/quant_ops.h"
#include "ops/format.h"
#include "ops/quant/nint_prefill_cache.h"
#include "packed_nint.h"

#include <array>
#include <cstring>
#include <limits>
#include <map>

using mfq_tensor_backend::indexing::Slice;
using namespace mfq::cuda::quant_format;


static void refresh_nint_descriptor(NintCpu & t) {
    if (t.out <= 0 || t.neuron_len <= 0 || t.ng <= 0 || t.gs <= 0 ||
            t.row_q_bits.size() != static_cast<size_t>(t.out) ||
            t.row_sub_bits.size() != static_cast<size_t>(t.out)) {
        throw std::runtime_error("cannot describe incomplete NINT row metadata");
    }
    constexpr int q_selector_bits = 3;
    constexpr int k_selector_bits = 2;
    double encoded_bits = static_cast<double>(t.out) *
        (32.0 + q_selector_bits + k_selector_bits);
    std::array<size_t, 64> joint_counts{};
    const double padded_values_per_row =
        static_cast<double>(t.ng) * t.gs;
    for (int row = 0; row < t.out; ++row) {
        const int q_bits = t.row_q_bits[static_cast<size_t>(row)];
        const int k_bits = t.row_sub_bits[static_cast<size_t>(row)];
        if (q_bits < 1 || q_bits > 8 || k_bits < 1 || k_bits > 8) {
            throw std::runtime_error("invalid NINT descriptor row metadata");
        }
        encoded_bits += padded_values_per_row * q_bits +
            2.0 * t.ng * k_bits;
        ++joint_counts[static_cast<size_t>((q_bits - 1) * 8 + (k_bits - 1))];
    }
    t.aggregate_bpw = encoded_bits /
        (static_cast<double>(t.out) * t.neuron_len);
    t.distribution_entropy = 0.0;
    for (const size_t count : joint_counts) {
        if (count == 0) continue;
        const double probability = static_cast<double>(count) / t.out;
        t.distribution_entropy -= probability * std::log2(probability);
    }
}

NintCpu unpack_nint(const std::vector<uint8_t> & blob) {
    constexpr size_t fixed_header_nbytes = 2 + 3 * sizeof(int32_t) + sizeof(uint32_t);
    if (blob.size() < fixed_header_nbytes) {
        throw std::runtime_error("truncated NINT header");
    }
    NintCpu t;
    size_t off = 0;
    const int raw_bits = blob[off++];
    const bool is_nint_v2 = (raw_bits & 0x80) != 0;
    t.bits = raw_bits & 0x7f;
    t.sub_bits = blob[off++];
    t.gs = read_i32_from(blob, off);
    t.axis = read_i32_from(blob, off);
    t.neuron_len = read_i32_from(blob, off);
    uint32_t ndim = read_u32_from(blob, off);
    if (t.bits < 1 || t.bits > 8 || t.sub_bits < 1 || t.sub_bits > 8 ||
            t.gs <= 0 || t.axis < 0 || t.neuron_len <= 0 ||
            ndim == 0 || t.axis >= static_cast<int>(ndim) ||
            ndim > (blob.size() - off) / sizeof(int64_t)) {
        throw std::runtime_error("invalid NINT header");
    }
    const size_t shape_and_counts =
        static_cast<size_t>(ndim) * sizeof(int64_t) + 2 * sizeof(uint32_t);
    if (shape_and_counts > blob.size() - off) {
        throw std::runtime_error("truncated NINT dimensions");
    }
    t.shape.resize(ndim);
    for (uint32_t i = 0; i < ndim; ++i) t.shape[i] = read_i64_from(blob, off);
    t.out = (int)read_u32_from(blob, off);
    t.ng = (int)read_u32_from(blob, off);
    int64_t flattened_neuron_len = 1;
    for (uint32_t index = 0; index < ndim; ++index) {
        if (t.shape[index] <= 0 ||
                (static_cast<int>(index) != t.axis &&
                 flattened_neuron_len >
                    std::numeric_limits<int64_t>::max() / t.shape[index])) {
            throw std::runtime_error("invalid NINT tensor shape");
        }
        if (static_cast<int>(index) != t.axis) {
            flattened_neuron_len *= t.shape[index];
        }
    }
    const int64_t expected_groups =
        (static_cast<int64_t>(t.neuron_len) + t.gs - 1) / t.gs;
    const bool routed_shape = ndim == 3 && t.axis == 0 &&
        t.shape[0] <= std::numeric_limits<int>::max() &&
        t.shape[1] <= std::numeric_limits<int>::max() &&
        t.shape[0] * t.shape[1] == t.out && t.shape[2] == t.neuron_len;
    if (t.out <= 0 || t.ng <= 0 ||
            (!routed_shape && (t.shape[static_cast<size_t>(t.axis)] != t.out ||
             flattened_neuron_len != t.neuron_len)) ||
            expected_groups != t.ng) {
        throw std::runtime_error("inconsistent NINT tensor dimensions");
    }
    const size_t anchor_nbytes = static_cast<size_t>(t.out) * 2;
    if (anchor_nbytes > (blob.size() - off) / 2) {
        throw std::runtime_error("truncated NINT neuron metadata");
    }
    t.neuron_scale_h.resize(t.out);
    std::memcpy(t.neuron_scale_h.data(), blob.data() + off, (size_t)t.out * 2);
    off += (size_t)t.out * 2;
    t.neuron_min_h.resize(t.out);
    std::memcpy(t.neuron_min_h.data(), blob.data() + off, (size_t)t.out * 2);
    off += (size_t)t.out * 2;
    for (size_t row = 0; row < static_cast<size_t>(t.out); ++row) {
        if ((t.neuron_scale_h[row] & 0x7c00u) == 0x7c00u ||
                (t.neuron_min_h[row] & 0x7c00u) == 0x7c00u) {
            throw std::runtime_error("NINT neuron metadata must be finite");
        }
    }
    if (static_cast<size_t>(t.out) >
            std::numeric_limits<size_t>::max() / static_cast<size_t>(t.ng)) {
        throw std::runtime_error("NINT metadata size overflow");
    }
    size_t sub_count = static_cast<size_t>(t.out) * static_cast<size_t>(t.ng);
    if (sub_count >
            std::numeric_limits<size_t>::max() / static_cast<size_t>(t.gs)) {
        throw std::runtime_error("NINT value size overflow");
    }
    size_t q_count = sub_count * static_cast<size_t>(t.gs);
    t.row_sub_bits.assign(
        static_cast<size_t>(t.out),
        static_cast<uint8_t>(t.sub_bits));
    if (is_nint_v2) {
        constexpr int selector_bits = 2;
        const auto selectors = unpack_bits(
            blob, off, static_cast<size_t>(t.out), selector_bits);
        for (size_t row = 0; row < static_cast<size_t>(t.out); ++row) {
            const int row_bits = t.sub_bits - 1 + selectors[row];
            if (row_bits < 1 || row_bits > 8) {
                throw std::runtime_error("invalid NINT v2 subgroup width");
            }
            t.row_sub_bits[row] = static_cast<uint8_t>(row_bits);
        }
        t.sub_scale.resize(sub_count);
        t.sub_min.resize(sub_count);
        for (int selector = 0; selector < (1 << selector_bits); ++selector) {
            const int row_bits = t.sub_bits - 1 + selector;
            const size_t selected_rows = static_cast<size_t>(std::count(
                selectors.begin(), selectors.end(), static_cast<uint8_t>(selector)));
            if (selected_rows == 0) continue;
            if (row_bits < 1 || row_bits > 8) {
                throw std::runtime_error("invalid NINT v2 subgroup width");
            }
            const size_t selected_values = selected_rows * static_cast<size_t>(t.ng);
            const auto scales = unpack_bits(blob, off, selected_values, row_bits);
            const auto minima = unpack_bits(blob, off, selected_values, row_bits);
            size_t local_row = 0;
            for (size_t row = 0; row < static_cast<size_t>(t.out); ++row) {
                if (selectors[row] != static_cast<uint8_t>(selector)) continue;
                const size_t source = local_row * static_cast<size_t>(t.ng);
                const size_t destination = row * static_cast<size_t>(t.ng);
                std::copy_n(
                    scales.begin() + static_cast<ptrdiff_t>(source),
                    t.ng,
                    t.sub_scale.begin() + static_cast<ptrdiff_t>(destination));
                std::copy_n(
                    minima.begin() + static_cast<ptrdiff_t>(source),
                    t.ng,
                    t.sub_min.begin() + static_cast<ptrdiff_t>(destination));
                ++local_row;
            }
        }
    } else {
        t.sub_scale = unpack_bits(blob, off, sub_count, t.sub_bits);
        t.sub_min = unpack_bits(blob, off, sub_count, t.sub_bits);
    }
    if (is_nint_v2) {
        constexpr int selector_bits = 3;
        const auto selectors = unpack_bits(
            blob, off, static_cast<size_t>(t.out), selector_bits);
        t.row_q_bits.resize(static_cast<size_t>(t.out));
        t.row_q_bit_offsets.resize(static_cast<size_t>(t.out));
        const size_t values_per_row =
            static_cast<size_t>(t.ng) * static_cast<size_t>(t.gs);
        for (int selector = 0; selector < (1 << selector_bits); ++selector) {
            const int row_bits = selector + 1;
            const size_t selected_rows = static_cast<size_t>(std::count(
                selectors.begin(), selectors.end(), static_cast<uint8_t>(selector)));
            if (selected_rows == 0) continue;
            if (values_per_row > std::numeric_limits<size_t>::max() / selected_rows) {
                throw std::runtime_error("NINT mixed-q value count overflow");
            }
            const size_t selected_values = selected_rows * values_per_row;
            if (selected_values >
                    (std::numeric_limits<size_t>::max() - 7) /
                        static_cast<size_t>(row_bits)) {
                throw std::runtime_error("NINT mixed-q packed size overflow");
            }
            const size_t stream_nbytes =
                (selected_values * static_cast<size_t>(row_bits) + 7) / 8;
            if (off > blob.size() || stream_nbytes > blob.size() - off) {
                throw std::runtime_error("truncated NINT mixed-q value stream");
            }
            const uint64_t cohort_start_bits =
                static_cast<uint64_t>(t.q_packed.size()) * 8u;
            t.q_packed.insert(
                t.q_packed.end(),
                blob.begin() + static_cast<ptrdiff_t>(off),
                blob.begin() + static_cast<ptrdiff_t>(off + stream_nbytes));
            off += stream_nbytes;
            size_t local_row = 0;
            const uint64_t packed_row_bits =
                static_cast<uint64_t>(values_per_row) *
                static_cast<uint64_t>(row_bits);
            for (size_t row = 0; row < static_cast<size_t>(t.out); ++row) {
                if (selectors[row] != static_cast<uint8_t>(selector)) continue;
                const uint64_t bit_offset = cohort_start_bits +
                    static_cast<uint64_t>(local_row) * packed_row_bits;
                if (bit_offset > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
                    throw std::runtime_error("NINT mixed-q row offset overflow");
                }
                t.row_q_bits[row] = static_cast<uint8_t>(row_bits);
                t.row_q_bit_offsets[row] = static_cast<int64_t>(bit_offset);
                ++local_row;
            }
        }
        t.q_packed.insert(t.q_packed.end(), 8, 0);
    } else {
        if (q_count >
                (std::numeric_limits<size_t>::max() - 7) /
                    static_cast<size_t>(t.bits)) {
            throw std::runtime_error("NINT packed value size overflow");
        }
        const size_t compact_q_nbytes =
            (q_count * static_cast<size_t>(t.bits) + 7) / 8;
        if (off > blob.size() || compact_q_nbytes > blob.size() - off) {
            throw std::runtime_error("truncated NINT packed values");
        }
        t.q_packed.assign(
            blob.begin() + static_cast<ptrdiff_t>(off),
            blob.begin() + static_cast<ptrdiff_t>(off + compact_q_nbytes));
        off += compact_q_nbytes;
        t.row_q_bits.assign(static_cast<size_t>(t.out),
                            static_cast<uint8_t>(t.bits));
        t.row_q_bit_offsets.resize(static_cast<size_t>(t.out));
        const uint64_t row_bits =
            static_cast<uint64_t>(t.ng) * static_cast<uint64_t>(t.gs) *
            static_cast<uint64_t>(t.bits);
        for (size_t row = 0; row < static_cast<size_t>(t.out); ++row) {
            const uint64_t bit_offset = static_cast<uint64_t>(row) * row_bits;
            if (bit_offset >
                    static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
                throw std::runtime_error("NINT row offset overflow");
            }
            t.row_q_bit_offsets[row] = static_cast<int64_t>(bit_offset);
        }
        t.q_packed.insert(t.q_packed.end(), 8, 0);
    }
    if (off != blob.size()) {
        throw std::runtime_error("invalid NINT trailing bytes");
    }
    refresh_nint_descriptor(t);
    return t;
}

Nint8ZeroCpu unpack_nint8_zero(const std::vector<uint8_t> & blob) {
    constexpr size_t block_bytes = 34;
    if (blob.size() < 24 || std::memcmp(blob.data(), "NI80", 4) != 0) {
        throw std::runtime_error("invalid NINT8-0 header");
    }
    Nint8ZeroCpu t;
    size_t off = 4;
    t.axis = read_i32_from(blob, off);
    t.neuron_len = read_i32_from(blob, off);
    uint32_t ndim = read_u32_from(blob, off);
    if (ndim == 0 || t.axis < 0 || t.axis >= static_cast<int>(ndim) ||
        t.neuron_len <= 0 || t.neuron_len % 32 != 0) {
        throw std::runtime_error("invalid NINT8-0 dimensions");
    }
    t.shape.resize(ndim);
    for (uint32_t index = 0; index < ndim; ++index) {
        t.shape[index] = read_i64_from(blob, off);
    }
    t.out = static_cast<int>(read_u32_from(blob, off));
    t.ng = static_cast<int>(read_u32_from(blob, off));
    if (t.out <= 0 || t.ng != t.neuron_len / 32 ||
        t.shape[static_cast<size_t>(t.axis)] != t.out) {
        throw std::runtime_error("NINT8-0 shape/header mismatch");
    }
    int64_t elements = 1;
    for (int64_t value : t.shape) {
        if (value <= 0 || elements > INT64_MAX / value) {
            throw std::runtime_error("invalid NINT8-0 logical shape");
        }
        elements *= value;
    }
    if (elements != static_cast<int64_t>(t.out) * t.neuron_len) {
        throw std::runtime_error("NINT8-0 logical element count mismatch");
    }
    const size_t blocks = static_cast<size_t>(t.out) * t.ng;
    if (off > blob.size() ||
            blocks > (blob.size() - off) / block_bytes ||
            blob.size() - off != blocks * block_bytes) {
        throw std::runtime_error("invalid NINT8-0 block payload length");
    }
    t.q.resize(blocks * 32);
    t.scale_h.resize(blocks);
    for (size_t block = 0; block < blocks; ++block) {
        const uint8_t * source = blob.data() + off + block * block_bytes;
        std::memcpy(&t.scale_h[block], source, sizeof(uint16_t));
        const uint16_t scale = t.scale_h[block];
        if ((scale & 0x7c00u) == 0x7c00u || (scale & 0x8000u) != 0u) {
            throw std::runtime_error(
                "NINT8-0 scales must be finite and non-negative");
        }
        std::memcpy(t.q.data() + block * 32, source + 2, 32);
    }
    return t;
}

void require_tp_row_major_weight(
        const std::vector<int64_t> & shape,
        int axis,
        int out,
        int neuron_len,
        const char * format) {
    if (shape.size() != 2 || axis != 0 ||
        shape[0] != out || shape[1] != neuron_len) {
        throw std::runtime_error(
            std::string(format) +
            " tensor parallelism requires a row-major rank-2 weight");
    }
}

static uint64_t nint_row_payload_bits(
        int ng, int gs, int bits) {
    if (ng <= 0 || gs <= 0 || bits < 1 || bits > 8 ||
            static_cast<uint64_t>(ng) >
                std::numeric_limits<uint64_t>::max() /
                    static_cast<uint64_t>(gs) /
                    static_cast<uint64_t>(bits)) {
        throw std::runtime_error("invalid NINT row bit geometry");
    }
    return static_cast<uint64_t>(ng) *
        static_cast<uint64_t>(gs) * static_cast<uint64_t>(bits);
}

static void copy_nint_packed_bits(
        const std::vector<uint8_t> & source,
        uint64_t source_bit,
        std::vector<uint8_t> & destination,
        uint64_t destination_bit,
        uint64_t bit_count) {
    const uint64_t source_capacity =
        static_cast<uint64_t>(source.size()) * 8u;
    const uint64_t destination_capacity =
        static_cast<uint64_t>(destination.size()) * 8u;
    if (source_bit > source_capacity ||
            bit_count > source_capacity - source_bit ||
            destination_bit > destination_capacity ||
            bit_count > destination_capacity - destination_bit) {
        throw std::runtime_error("NINT packed bit range is out of bounds");
    }
    if (((source_bit | destination_bit | bit_count) & 7u) == 0u) {
        std::memcpy(
            destination.data() + static_cast<size_t>(destination_bit >> 3),
            source.data() + static_cast<size_t>(source_bit >> 3),
            static_cast<size_t>(bit_count >> 3));
        return;
    }
    while (bit_count != 0) {
        const int source_shift = static_cast<int>(source_bit & 7u);
        const int destination_shift = static_cast<int>(destination_bit & 7u);
        const int chunk = static_cast<int>(std::min<uint64_t>(
            bit_count,
            static_cast<uint64_t>(std::min(
                8 - source_shift, 8 - destination_shift))));
        const uint8_t mask = static_cast<uint8_t>((1u << chunk) - 1u);
        const uint8_t value = static_cast<uint8_t>(
            (source[static_cast<size_t>(source_bit >> 3)] >> source_shift) &
            mask);
        destination[static_cast<size_t>(destination_bit >> 3)] |=
            static_cast<uint8_t>(value << destination_shift);
        source_bit += static_cast<uint64_t>(chunk);
        destination_bit += static_cast<uint64_t>(chunk);
        bit_count -= static_cast<uint64_t>(chunk);
    }
}

static void require_canonical_nint_cpu(const NintCpu & source) {
    if (source.out <= 0 ||
            source.row_q_bits.size() != static_cast<size_t>(source.out) ||
            source.row_q_bit_offsets.size() != static_cast<size_t>(source.out) ||
            source.row_sub_bits.size() != static_cast<size_t>(source.out) ||
            source.q_packed.size() < 8) {
        throw std::runtime_error("NINT storage is missing canonical row metadata");
    }
    const uint64_t payload_capacity =
        static_cast<uint64_t>(source.q_packed.size() - 8) * 8u;
    for (int row = 0; row < source.out; ++row) {
        const int bits = source.row_q_bits[static_cast<size_t>(row)];
        const int64_t signed_offset =
            source.row_q_bit_offsets[static_cast<size_t>(row)];
        const uint64_t row_bits =
            nint_row_payload_bits(source.ng, source.gs, bits);
        if (signed_offset < 0 ||
                static_cast<uint64_t>(signed_offset) > payload_capacity ||
                row_bits > payload_capacity -
                    static_cast<uint64_t>(signed_offset)) {
            throw std::runtime_error("invalid canonical NINT row metadata");
        }
    }
}

static NintCpu repack_nint_cpu_rows(
        const NintCpu & source,
        const std::vector<int64_t> & rows) {
    require_canonical_nint_cpu(source);
    if (rows.empty()) {
        throw std::runtime_error("cannot create an empty NINT row selection");
    }
    uint64_t total_bits = 0;
    for (int64_t row : rows) {
        if (row < 0 || row >= source.out) {
            throw std::runtime_error("NINT selected row is out of range");
        }
        const uint64_t row_bits = nint_row_payload_bits(
            source.ng, source.gs,
            source.row_q_bits[static_cast<size_t>(row)]);
        if (row_bits > std::numeric_limits<uint64_t>::max() - total_bits) {
            throw std::runtime_error("NINT selected row payload is too large");
        }
        total_bits += row_bits;
    }
    if (total_bits > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
            (total_bits + 7u) / 8u >
            static_cast<uint64_t>(std::numeric_limits<size_t>::max() - 8)) {
        throw std::runtime_error("NINT selected row payload is too large");
    }
    NintCpu result = source;
    result.out = static_cast<int>(rows.size());
    result.shape[0] = result.out;
    result.q_packed.assign(static_cast<size_t>((total_bits + 7u) / 8u) + 8, 0);
    result.row_q_bits.resize(rows.size());
    result.row_q_bit_offsets.resize(rows.size());
    result.row_sub_bits.resize(rows.size());
    result.sub_scale.resize(rows.size() * static_cast<size_t>(source.ng));
    result.sub_min.resize(rows.size() * static_cast<size_t>(source.ng));
    result.neuron_scale_h.resize(rows.size());
    result.neuron_min_h.resize(rows.size());
    uint64_t destination_bit = 0;
    for (size_t destination = 0; destination < rows.size(); ++destination) {
        const size_t source_row = static_cast<size_t>(rows[destination]);
        const int bits = source.row_q_bits[source_row];
        const uint64_t row_bits =
            nint_row_payload_bits(source.ng, source.gs, bits);
        result.row_q_bits[destination] = static_cast<uint8_t>(bits);
        result.row_sub_bits[destination] = source.row_sub_bits[source_row];
        result.row_q_bit_offsets[destination] =
            static_cast<int64_t>(destination_bit);
        copy_nint_packed_bits(
            source.q_packed,
            static_cast<uint64_t>(source.row_q_bit_offsets[source_row]),
            result.q_packed, destination_bit, row_bits);
        const size_t source_group = source_row * static_cast<size_t>(source.ng);
        const size_t destination_group =
            destination * static_cast<size_t>(source.ng);
        std::copy_n(
            source.sub_scale.begin() + static_cast<ptrdiff_t>(source_group),
            source.ng,
            result.sub_scale.begin() + static_cast<ptrdiff_t>(destination_group));
        std::copy_n(
            source.sub_min.begin() + static_cast<ptrdiff_t>(source_group),
            source.ng,
            result.sub_min.begin() + static_cast<ptrdiff_t>(destination_group));
        result.neuron_scale_h[destination] = source.neuron_scale_h[source_row];
        result.neuron_min_h[destination] = source.neuron_min_h[source_row];
        destination_bit += row_bits;
    }
    refresh_nint_descriptor(result);
    return result;
}

NintCpu slice_nint_cpu_output(
        const NintCpu & source, int64_t begin, int64_t end) {
    require_tp_row_major_weight(
        source.shape, source.axis, source.out,
        source.neuron_len, "NINT");
    if (begin < 0 || begin >= end || end > source.out) {
        throw std::runtime_error("invalid NINT output shard");
    }
    std::vector<int64_t> rows(static_cast<size_t>(end - begin));
    std::iota(rows.begin(), rows.end(), begin);
    return repack_nint_cpu_rows(source, rows);
}

NintCpu slice_nint_cpu_input_groups(
        const NintCpu & source, int64_t begin, int64_t end) {
    require_tp_row_major_weight(
        source.shape, source.axis, source.out,
        source.neuron_len, "NINT");
    if (begin < 0 || begin >= end || end > source.ng) {
        throw std::runtime_error("invalid NINT input shard");
    }
    require_canonical_nint_cpu(source);
    NintCpu result = source;
    result.ng = static_cast<int>(end - begin);
    const int64_t element_begin = begin * source.gs;
    const int64_t element_end =
        std::min<int64_t>(end * source.gs, source.neuron_len);
    result.neuron_len = static_cast<int>(element_end - element_begin);
    result.shape[1] = result.neuron_len;
    uint64_t total_bits = 0;
    for (int row = 0; row < source.out; ++row) {
        const uint64_t row_bits = nint_row_payload_bits(
            result.ng, source.gs,
            source.row_q_bits[static_cast<size_t>(row)]);
        if (row_bits > std::numeric_limits<uint64_t>::max() - total_bits) {
            throw std::runtime_error("NINT input shard payload is too large");
        }
        total_bits += row_bits;
    }
    if (total_bits > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
            (total_bits + 7u) / 8u >
            static_cast<uint64_t>(std::numeric_limits<size_t>::max() - 8)) {
        throw std::runtime_error("NINT input shard payload is too large");
    }
    result.q_packed.assign(static_cast<size_t>((total_bits + 7u) / 8u) + 8, 0);
    result.row_q_bit_offsets.resize(static_cast<size_t>(source.out));
    result.sub_scale.resize(
        static_cast<size_t>(source.out) * result.ng);
    result.sub_min.resize(
        static_cast<size_t>(source.out) * result.ng);
    uint64_t destination_bit = 0;
    for (int row = 0; row < source.out; ++row) {
        const int bits = source.row_q_bits[static_cast<size_t>(row)];
        const uint64_t row_bits =
            nint_row_payload_bits(result.ng, source.gs, bits);
        const uint64_t source_bit = static_cast<uint64_t>(
            source.row_q_bit_offsets[static_cast<size_t>(row)]) +
            static_cast<uint64_t>(begin) *
                static_cast<uint64_t>(source.gs) *
                static_cast<uint64_t>(bits);
        result.row_q_bit_offsets[static_cast<size_t>(row)] =
            static_cast<int64_t>(destination_bit);
        copy_nint_packed_bits(
            source.q_packed, source_bit,
            result.q_packed, destination_bit, row_bits);
        destination_bit += row_bits;
        const size_t source_group =
            static_cast<size_t>(row) * source.ng +
            static_cast<size_t>(begin);
        const size_t destination_group =
            static_cast<size_t>(row) * result.ng;
        std::memcpy(
            result.sub_scale.data() + destination_group,
            source.sub_scale.data() + source_group,
            static_cast<size_t>(result.ng));
        std::memcpy(
            result.sub_min.data() + destination_group,
            source.sub_min.data() + source_group,
            static_cast<size_t>(result.ng));
    }
    refresh_nint_descriptor(result);
    return result;
}

Nint8ZeroCpu slice_nint8_zero_cpu_output(
        const Nint8ZeroCpu & source, int64_t begin, int64_t end) {
    require_tp_row_major_weight(
        source.shape, source.axis, source.out,
        source.neuron_len, "NINT8-0");
    if (begin < 0 || begin >= end || end > source.out) {
        throw std::runtime_error("invalid NINT8-0 output shard");
    }
    Nint8ZeroCpu result = source;
    result.out = static_cast<int>(end - begin);
    result.shape[0] = result.out;
    const size_t first_group =
        static_cast<size_t>(begin) * source.ng;
    const size_t group_count =
        static_cast<size_t>(result.out) * source.ng;
    result.q.assign(
        source.q.begin() +
            static_cast<ptrdiff_t>(first_group * 32),
        source.q.begin() +
            static_cast<ptrdiff_t>((first_group + group_count) * 32));
    result.scale_h.assign(
        source.scale_h.begin() + static_cast<ptrdiff_t>(first_group),
        source.scale_h.begin() +
            static_cast<ptrdiff_t>(first_group + group_count));
    return result;
}

Nint8ZeroCpu slice_nint8_zero_cpu_input_groups(
        const Nint8ZeroCpu & source, int64_t begin, int64_t end) {
    require_tp_row_major_weight(
        source.shape, source.axis, source.out,
        source.neuron_len, "NINT8-0");
    if (begin < 0 || begin >= end || end > source.ng) {
        throw std::runtime_error("invalid NINT8-0 input shard");
    }
    Nint8ZeroCpu result = source;
    result.ng = static_cast<int>(end - begin);
    result.neuron_len = result.ng * 32;
    result.shape[1] = result.neuron_len;
    result.q.resize(
        static_cast<size_t>(source.out) * result.ng * 32);
    result.scale_h.resize(
        static_cast<size_t>(source.out) * result.ng);
    for (int row = 0; row < source.out; ++row) {
        const size_t source_group =
            static_cast<size_t>(row) * source.ng +
            static_cast<size_t>(begin);
        const size_t destination_group =
            static_cast<size_t>(row) * result.ng;
        std::memcpy(
            result.q.data() + destination_group * 32,
            source.q.data() + source_group * 32,
            static_cast<size_t>(result.ng) * 32);
        std::memcpy(
            result.scale_h.data() + destination_group,
            source.scale_h.data() + source_group,
            static_cast<size_t>(result.ng) * sizeof(uint16_t));
    }
    return result;
}
NintWeight to_device_nint(const NintCpu & c, bool cuda) {
    if (c.bits < 1 || c.bits > 8) throw std::runtime_error("unsupported NINT bits");
    require_canonical_nint_cpu(c);
    NintWeight w;
    w.out = c.out;
    w.ng = c.ng;
    w.gs = c.gs;
    w.bits = c.bits;
    w.neuron_len = c.neuron_len;
    w.format_version = c.format_version;
    w.aggregate_bpw = c.aggregate_bpw;
    w.distribution_entropy = c.distribution_entropy;
    w.shape = c.shape;
    w.aligned_q8 = c.gs % 4 == 0;
    for (size_t row = 0; row < c.row_q_bits.size(); ++row) {
        w.aligned_q8 = w.aligned_q8 && c.row_q_bits[row] == 8 &&
            (c.row_q_bit_offsets[row] & 31) == 0;
    }
    w.q_packed = cpu_u8_tensor(
        c.q_packed, {static_cast<int64_t>(c.q_packed.size())});
    w.row_q_bits = cpu_u8_tensor(c.row_q_bits, {c.out});
    w.row_q_bit_offsets = cpu_i64_tensor(c.row_q_bit_offsets, {c.out});
    w.sub_scale = cpu_u8_tensor(c.sub_scale, {c.out, c.ng});
    w.sub_min = cpu_u8_tensor(c.sub_min, {c.out, c.ng});
    w.neuron_scale = cpu_f16_to_f32_tensor(c.neuron_scale_h, c.out);
    w.neuron_min = cpu_f16_to_f32_tensor(c.neuron_min_h, c.out);
    if (cuda) {
        const auto target = mfq_tensor_backend::Device(
            mfq_tensor_backend::kCUDA, mfq_current_cuda_device());
        w.q_packed = w.q_packed.to(target).contiguous();
        w.row_q_bits = w.row_q_bits.to(target).contiguous();
        w.row_q_bit_offsets = w.row_q_bit_offsets.to(target).contiguous();
        w.sub_scale = w.sub_scale.to(target).contiguous();
        w.sub_min = w.sub_min.to(target).contiguous();
        w.neuron_scale = w.neuron_scale.to(target).contiguous();
        w.neuron_min = w.neuron_min.to(target).contiguous();
    }
    return w;
}

NintWeight to_gpu_nint(const NintCpu & c) {
    return to_device_nint(c, true);
}

NintWeight to_cuda_device_nint(
        const NintCpu & c, int device) {
    MfqCudaGuard guard(device);
    return to_device_nint(c, true);
}

NintWeight to_cpu_nint(const NintCpu & c) {
    return to_device_nint(c, false);
}

NintWeight to_device_mfe_nint(
        const NintCpu & source,
        int local_experts,
        int out_per_expert,
        bool cuda) {
    require_canonical_nint_cpu(source);
    if (local_experts <= 0 || out_per_expert <= 0 ||
            source.out != local_experts * out_per_expert) {
        throw std::runtime_error("invalid MFE NINT expert geometry");
    }
    uint64_t maximum_payload_bits = 0;
    for (int expert = 0; expert < local_experts; ++expert) {
        uint64_t payload_bits = 0;
        for (int local_row = 0; local_row < out_per_expert; ++local_row) {
            const size_t row = static_cast<size_t>(expert) * out_per_expert +
                static_cast<size_t>(local_row);
            const uint64_t row_bits = nint_row_payload_bits(
                source.ng, source.gs, source.row_q_bits[row]);
            if (row_bits > std::numeric_limits<uint64_t>::max() - payload_bits) {
                throw std::runtime_error("MFE NINT expert payload is too large");
            }
            payload_bits += row_bits;
        }
        maximum_payload_bits = std::max(maximum_payload_bits, payload_bits);
    }
    const uint64_t stride_u64 = (maximum_payload_bits + 7u) / 8u + 8u;
    if (stride_u64 > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
            static_cast<uint64_t>(local_experts) >
                static_cast<uint64_t>(std::numeric_limits<size_t>::max()) /
                    stride_u64) {
        throw std::runtime_error("MFE NINT expert payload is too large");
    }
    const size_t stride = static_cast<size_t>(stride_u64);
    std::vector<uint8_t> expert_stream(
        static_cast<size_t>(local_experts) * stride, 0);
    std::vector<int64_t> relative_offsets(static_cast<size_t>(source.out));
    for (int expert = 0; expert < local_experts; ++expert) {
        uint64_t destination_bit = 0;
        for (int local_row = 0; local_row < out_per_expert; ++local_row) {
            const size_t row = static_cast<size_t>(expert) * out_per_expert +
                static_cast<size_t>(local_row);
            const int bits = source.row_q_bits[row];
            const uint64_t row_bits =
                nint_row_payload_bits(source.ng, source.gs, bits);
            relative_offsets[row] = static_cast<int64_t>(destination_bit);
            copy_nint_packed_bits(
                source.q_packed,
                static_cast<uint64_t>(source.row_q_bit_offsets[row]),
                expert_stream,
                static_cast<uint64_t>(expert) * stride_u64 * 8u +
                    destination_bit,
                row_bits);
            destination_bit += row_bits;
        }
    }
    NintWeight result = to_device_nint(source, false);
    result.aligned_q8 = false;
    result.q_expert_stride = static_cast<int64_t>(stride);
    result.q_packed = cpu_u8_tensor(
        expert_stream, {local_experts, static_cast<int64_t>(stride)});
    result.row_q_bit_offsets = cpu_i64_tensor(
        relative_offsets, {source.out});
    if (cuda) {
        const auto target = mfq_tensor_backend::Device(
            mfq_tensor_backend::kCUDA, mfq_current_cuda_device());
        result.q_packed = result.q_packed.to(target).contiguous();
        result.row_q_bits = result.row_q_bits.to(target).contiguous();
        result.row_q_bit_offsets =
            result.row_q_bit_offsets.to(target).contiguous();
        result.sub_scale = result.sub_scale.to(target).contiguous();
        result.sub_min = result.sub_min.to(target).contiguous();
        result.neuron_scale = result.neuron_scale.to(target).contiguous();
        result.neuron_min = result.neuron_min.to(target).contiguous();
    }
    return result;
}

NintWeight to_device_nint8_zero(
        const Nint8ZeroCpu & c, bool cuda) {
    NintWeight w;
    w.out = c.out;
    w.ng = c.ng;
    w.gs = 32;
    w.bits = 8;
    w.neuron_len = c.neuron_len;
    w.q8_zero = true;
    w.shape = c.shape;
    w.q_packed = cpu_u8_tensor(c.q, {c.out, c.ng, 32});
    w.q8_zero_scale = cpu_f16_tensor(c.scale_h, {c.out, c.ng});
    if (cuda) {
        const auto target = mfq_tensor_backend::Device(
            mfq_tensor_backend::kCUDA, mfq_current_cuda_device());
        w.q_packed = w.q_packed.to(target).contiguous();
        w.q8_zero_scale = w.q8_zero_scale.to(target).contiguous();
    }
    return w;
}

NintWeight to_gpu_nint8_zero(const Nint8ZeroCpu & c) {
    return to_device_nint8_zero(c, true);
}

NintWeight to_cuda_device_nint8_zero(
        const Nint8ZeroCpu & c, int device) {
    MfqCudaGuard guard(device);
    return to_device_nint8_zero(c, true);
}

NintWeight to_cpu_nint8_zero(const Nint8ZeroCpu & c) {
    return to_device_nint8_zero(c, false);
}

NintCpu select_nint_cpu_rows(
        const NintCpu & source,
        const std::vector<int64_t> & rows) {
    require_tp_row_major_weight(
        source.shape, source.axis, source.out,
        source.neuron_len, "NINT");
    return repack_nint_cpu_rows(source, rows);
}

Nint8ZeroCpu select_nint8_zero_cpu_rows(
        const Nint8ZeroCpu & source,
        const std::vector<int64_t> & rows) {
    require_tp_row_major_weight(
        source.shape, source.axis, source.out,
        source.neuron_len, "NINT8-0");
    if (rows.empty()) {
        throw std::runtime_error(
            "cannot create an empty NINT8-0 MoE shard");
    }
    Nint8ZeroCpu result = source;
    result.out = static_cast<int>(rows.size());
    result.shape[0] = result.out;
    const size_t q_row_bytes =
        static_cast<size_t>(source.ng) * 32;
    result.q.resize(rows.size() * q_row_bytes);
    result.scale_h.resize(
        rows.size() * source.ng);
    for (size_t destination = 0;
         destination < rows.size(); ++destination) {
        const int64_t source_row = rows[destination];
        if (source_row < 0 ||
            source_row >= source.out) {
            throw std::runtime_error(
                "NINT8-0 MoE shard row is out of range");
        }
        std::memcpy(
            result.q.data() +
                destination * q_row_bytes,
            source.q.data() +
                static_cast<size_t>(source_row) *
                    q_row_bytes,
            q_row_bytes);
        std::memcpy(
            result.scale_h.data() +
                destination * source.ng,
            source.scale_h.data() +
                static_cast<size_t>(source_row) *
                    source.ng,
            static_cast<size_t>(source.ng) *
                sizeof(uint16_t));
    }
    return result;
}

mfq_tensor_backend::Tensor dequant_nint8_zero_cpu(const Nint8ZeroCpu & source) {
    MFQ_RUNTIME_CHECK(source.shape.size() == 2,
        "CPU NINT8-0 dense tensor must be rank 2");
    auto packed = to_device_nint8_zero(source, false);
    auto result = mfq_tensor_backend::empty(
        source.shape,
        mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCPU).dtype(mfq_tensor_backend::kFloat32));
    const auto * quantized =
        reinterpret_cast<const int8_t *>(packed.q_packed.data_ptr<uint8_t>());
    const mfq_half * scales = packed.q8_zero_scale.data_ptr<mfq_half>();
    float * output = result.data_ptr<float>();
    mfq_parallel_for(0, source.out, 1, [&](int64_t begin, int64_t end) {
        for (int64_t row = begin; row < end; ++row) {
            for (int64_t group = 0; group < source.ng; ++group) {
                const int64_t meta = row * source.ng + group;
                const float scale = static_cast<float>(scales[meta]);
                const int64_t k0 = group * 32;
                const int64_t valid = std::min<int64_t>(
                    32, source.neuron_len - k0);
                for (int64_t index = 0; index < valid; ++index) {
                    output[row * source.neuron_len + k0 + index] =
                        scale * static_cast<float>(quantized[meta * 32 + index]);
                }
            }
        }
    });
    return result.contiguous();
}

