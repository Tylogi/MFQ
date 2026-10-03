#include "cuda_quantized_ops.h"
#include "mfq/mxfp4_sq_decode.h"

#include "mfq_format_compat.h"
#include "mfe_expert_store.h"
#include "moe_cache_policy.h"
#include "moe_cache_transfer.h"
#include "nvq_codebooks.generated.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <iterator>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(__x86_64__) && defined(__GNUC__)
#include <immintrin.h>
#define MFQ_CPU_X86_GNU 1
#endif

using mfq_tensor_backend::indexing::Slice;

std::shared_ptr<MoeExpertCache> g_moe_expert_cache;

static uint32_t read_u32(std::istream & is) {
    uint32_t v = 0;
    is.read(reinterpret_cast<char *>(&v), sizeof(v));
    if (!is) throw std::runtime_error("unexpected EOF reading u32");
    return v;
}

static uint64_t read_u64(std::istream & is) {
    uint64_t v = 0;
    is.read(reinterpret_cast<char *>(&v), sizeof(v));
    if (!is) throw std::runtime_error("unexpected EOF reading u64");
    return v;
}

static int32_t read_i32_from(const std::vector<uint8_t> & b, size_t & off) {
    int32_t v = 0;
    std::memcpy(&v, b.data() + off, sizeof(v));
    off += sizeof(v);
    return v;
}

static uint16_t read_u16_from(const std::vector<uint8_t> & b, size_t & off) {
    uint16_t v = 0;
    std::memcpy(&v, b.data() + off, sizeof(v));
    off += sizeof(v);
    return v;
}

int64_t read_i64_from(const std::vector<uint8_t> & b, size_t & off) {
    int64_t v = 0;
    std::memcpy(&v, b.data() + off, sizeof(v));
    off += sizeof(v);
    return v;
}

uint32_t read_u32_from(const std::vector<uint8_t> & b, size_t & off) {
    uint32_t v = 0;
    std::memcpy(&v, b.data() + off, sizeof(v));
    off += sizeof(v);
    return v;
}

static uint64_t read_u64_from(const std::vector<uint8_t> & b, size_t & off) {
    uint64_t v = 0;
    std::memcpy(&v, b.data() + off, sizeof(v));
    off += sizeof(v);
    return v;
}

static std::string read_str(std::istream & is) {
    uint32_t n = read_u32(is);
    std::string s(n, '\0');
    is.read(s.data(), n);
    if (!is) throw std::runtime_error("unexpected EOF reading string");
    return s;
}

static std::vector<uint8_t> unpack_bits(const std::vector<uint8_t> & blob, size_t & off, size_t count, int bits) {
    if (bits < 1 || bits > 8 ||
            count > (std::numeric_limits<size_t>::max() - 7) /
                static_cast<size_t>(bits)) {
        throw std::runtime_error("invalid packed-bit stream dimensions");
    }
    const size_t required =
        (count * static_cast<size_t>(bits) + 7) / 8;
    if (off > blob.size() || required > blob.size() - off) {
        throw std::runtime_error("truncated packed-bit stream");
    }
    std::vector<uint8_t> out(count);
    if (bits == 8) {
        std::copy(blob.begin() + (ptrdiff_t)off, blob.begin() + (ptrdiff_t)(off + count), out.begin());
        off += count;
        return out;
    }
    if (bits == 4) {
        size_t nbytes = (count + 1) / 2;
        for (size_t i = 0; i < count; ++i) {
            uint8_t p = blob[off + i / 2];
            out[i] = (i & 1) ? (p >> 4) : (p & 0x0f);
        }
        off += nbytes;
        return out;
    }
    size_t nbytes = (count * (size_t)bits + 7) / 8;
    for (size_t i = 0; i < count; ++i) {
        uint32_t v = 0;
        size_t bit0 = i * (size_t)bits;
        for (int j = 0; j < bits; ++j) {
            size_t bit = bit0 + (size_t)j;
            uint8_t by = blob[off + bit / 8];
            v |= ((by >> (bit & 7)) & 1u) << j;
        }
        out[i] = (uint8_t)v;
    }
    off += nbytes;
    return out;
}

bool g_mfq_drop_file_cache = false;

const mfq::TensorMetadata& require_tensor(
        const mfq::ModelSource& source,
        std::string_view name) {
    const auto* tensor = source.find_tensor(name);
    if (tensor == nullptr) {
        throw std::runtime_error("missing tensor: " + std::string(name));
    }
    return *tensor;
}

bool has_tensor(
        const mfq::ModelSource& source,
        std::string_view name) noexcept {
    return source.find_tensor(name) != nullptr;
}

bool has_tensor_prefix(
        const mfq::ModelSource& source,
        std::string_view prefix) {
    for (const auto& tensor : source.tensors()) {
        if (tensor.name == prefix ||
            (tensor.name.size() > prefix.size() &&
             tensor.name.compare(0, prefix.size(), prefix) == 0 &&
             tensor.name[prefix.size()] == '.')) {
            return true;
        }
    }
    return false;
}

std::vector<std::uint8_t> read_tensor(
        const mfq::ModelSource& source,
        std::string_view name) {
    const auto& tensor = require_tensor(source, name);
    if (tensor.nbytes > std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error("model tensor is too large to read");
    }
    std::vector<std::uint8_t> result(
        static_cast<std::size_t>(tensor.nbytes));
    source.read_range_into(
        name, 0, reinterpret_cast<std::byte*>(result.data()), result.size());
    if (g_mfq_drop_file_cache) source.drop_file_cache();
    return result;
}

std::vector<std::uint8_t> read_asset(
        const mfq::ModelSource& source,
        std::string_view name) {
    const auto bytes = source.read_asset(name);
    std::vector<std::uint8_t> result(bytes.size());
    if (!bytes.empty()) {
        std::memcpy(result.data(), bytes.data(), bytes.size());
    }
    return result;
}

std::string read_asset_text(
        const mfq::ModelSource& source,
        std::string_view name) {
    const auto bytes = source.read_asset(name);
    return {
        reinterpret_cast<const char*>(bytes.data()), bytes.size()};
}

struct NintCpu {
    int format_version = 2;
    int bits = 0;
    int sub_bits = 0;
    int gs = 0;
    int axis = 0;
    int neuron_len = 0;
    std::vector<int64_t> shape;
    int out = 0;
    int ng = 0;
    std::vector<uint8_t> q_packed;
    std::vector<uint8_t> row_q_bits;
    std::vector<int64_t> row_q_bit_offsets;
    std::vector<uint8_t> row_sub_bits;
    std::vector<uint8_t> sub_scale;
    std::vector<uint8_t> sub_min;
    std::vector<uint16_t> neuron_scale_h;
    std::vector<uint16_t> neuron_min_h;
    double aggregate_bpw = 0.0;
    double distribution_entropy = 0.0;
};

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

static NintCpu unpack_nint(const std::vector<uint8_t> & blob) {
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
    if (t.out <= 0 || t.ng <= 0 ||
            t.shape[static_cast<size_t>(t.axis)] != t.out ||
            flattened_neuron_len != t.neuron_len ||
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

struct Nint8ZeroCpu {
    int axis = 0;
    int neuron_len = 0;
    std::vector<int64_t> shape;
    int out = 0;
    int ng = 0;
    std::vector<uint8_t> q;
    std::vector<uint16_t> scale_h;
};

static Nint8ZeroCpu unpack_nint8_zero(const std::vector<uint8_t> & blob) {
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

static void require_tp_row_major_weight(
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

static NintCpu slice_nint_cpu_output(
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

static NintCpu slice_nint_cpu_input_groups(
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

static Nint8ZeroCpu slice_nint8_zero_cpu_output(
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

static Nint8ZeroCpu slice_nint8_zero_cpu_input_groups(
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

struct Mxfp8Cpu {
    int64_t out = 0;
    int64_t neuron_len = 0;
    std::vector<uint8_t> values;
    std::vector<uint8_t> scales;
};

struct Mxfp4Cpu {
    int64_t out = 0;
    int64_t neuron_len = 0;
    std::vector<uint8_t> values;
    std::vector<uint8_t> scales;
};

static size_t checked_mxfp8_size(
        uint64_t left, uint64_t right,
        const char * label) {
    if (left == 0 || right == 0 ||
            left > std::numeric_limits<size_t>::max() / right) {
        throw std::runtime_error(
            std::string("invalid MXFP8 ") + label);
    }
    return static_cast<size_t>(left * right);
}

static Mxfp8Cpu unpack_mxfp8(
        const std::vector<uint8_t> & blob) {
    constexpr size_t kHeaderBytes = 56;
    if (blob.size() < kHeaderBytes ||
            std::memcmp(blob.data(), "MXT1", 4) != 0 ||
            blob[4] != 1 || blob[5] != 8 ||
            blob[6] != 0 || blob[7] != 0) {
        throw std::runtime_error("invalid MXFP8 payload header");
    }
    size_t offset = 8;
    const uint64_t rows = read_u64_from(blob, offset);
    const uint64_t columns = read_u64_from(blob, offset);
    const uint64_t storage_rows = read_u64_from(blob, offset);
    const uint64_t storage_columns = read_u64_from(blob, offset);
    const uint64_t scale_rows = read_u64_from(blob, offset);
    const uint64_t scale_columns = read_u64_from(blob, offset);
    if (rows > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
            columns > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
            columns % 128 != 0 ||
            storage_rows != rows || storage_columns != columns ||
            scale_rows != (rows + 127) / 128 ||
            scale_columns != columns / 128) {
        throw std::runtime_error("invalid MXFP8 payload geometry");
    }
    const size_t value_bytes = checked_mxfp8_size(
        storage_rows, storage_columns, "value size");
    const size_t scale_bytes = checked_mxfp8_size(
        scale_rows, scale_columns, "scale size");
    if (value_bytes > blob.size() - offset ||
            scale_bytes != blob.size() - offset - value_bytes) {
        throw std::runtime_error("invalid MXFP8 payload length");
    }
    Mxfp8Cpu result;
    result.out = static_cast<int64_t>(rows);
    result.neuron_len = static_cast<int64_t>(columns);
    result.values.assign(
        blob.begin() + static_cast<ptrdiff_t>(offset),
        blob.begin() + static_cast<ptrdiff_t>(offset + value_bytes));
    offset += value_bytes;
    result.scales.assign(
        blob.begin() + static_cast<ptrdiff_t>(offset),
        blob.end());
    if (std::any_of(
            result.values.begin(), result.values.end(),
            [](uint8_t value) { return (value & 0x7fu) == 0x7fu; })) {
        throw std::runtime_error("MXFP8 payload contains an E4M3 NaN code");
    }
    if (std::find(result.scales.begin(), result.scales.end(), 255u) !=
            result.scales.end()) {
        throw std::runtime_error("MXFP8 payload contains an E8M0 NaN scale");
    }
    return result;
}

static Mxfp4Cpu unpack_mxfp4(
        const std::vector<uint8_t> & blob) {
    constexpr size_t kHeaderBytes = 56;
    if (blob.size() < kHeaderBytes ||
            std::memcmp(blob.data(), "MXT1", 4) != 0 ||
            blob[4] != 1 || blob[5] != 4 ||
            blob[6] != 0 || blob[7] != 0) {
        throw std::runtime_error("invalid MXFP4 payload header");
    }
    size_t offset = 8;
    const uint64_t rows = read_u64_from(blob, offset);
    const uint64_t columns = read_u64_from(blob, offset);
    const uint64_t storage_rows = read_u64_from(blob, offset);
    const uint64_t storage_columns = read_u64_from(blob, offset);
    const uint64_t scale_rows = read_u64_from(blob, offset);
    const uint64_t scale_columns = read_u64_from(blob, offset);
    if (rows > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
            columns > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
            columns % 32 != 0 || storage_rows != rows ||
            storage_columns != columns / 2 || scale_rows != rows ||
            scale_columns != columns / 32) {
        throw std::runtime_error("invalid MXFP4 payload geometry");
    }
    const size_t value_bytes = checked_mxfp8_size(
        storage_rows, storage_columns, "MXFP4 value size");
    const size_t scale_bytes = checked_mxfp8_size(
        scale_rows, scale_columns, "MXFP4 scale size");
    if (value_bytes > blob.size() - offset ||
            scale_bytes != blob.size() - offset - value_bytes) {
        throw std::runtime_error("invalid MXFP4 payload length");
    }
    Mxfp4Cpu result;
    result.out = static_cast<int64_t>(rows);
    result.neuron_len = static_cast<int64_t>(columns);
    result.values.assign(
        blob.begin() + static_cast<ptrdiff_t>(offset),
        blob.begin() + static_cast<ptrdiff_t>(offset + value_bytes));
    offset += value_bytes;
    result.scales.assign(
        blob.begin() + static_cast<ptrdiff_t>(offset), blob.end());
    if (std::find(result.scales.begin(), result.scales.end(), 255u) !=
            result.scales.end()) {
        throw std::runtime_error("MXFP4 payload contains an E8M0 NaN scale");
    }
    return result;
}

static mfq_tensor_backend::Tensor dequant_mxfp4_cpu(const Mxfp4Cpu & source) {
    auto dense = mfq_tensor_backend::empty(
        {source.out, source.neuron_len},
        mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCPU).dtype(mfq_tensor_backend::kFloat32));
    float * destination = dense.data_ptr<float>();
    static constexpr float magnitude[8] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    };
    const int64_t packed_columns = source.neuron_len / 2;
    const int64_t scale_columns = source.neuron_len / 32;
    for (int64_t row = 0; row < source.out; ++row) {
        for (int64_t column = 0; column < source.neuron_len; ++column) {
            const uint8_t packed = source.values[
                static_cast<size_t>(row * packed_columns + column / 2)];
            const uint8_t code = static_cast<uint8_t>(
                (packed >> ((column & 1) * 4)) & 15u);
            const uint8_t raw_scale = source.scales[
                static_cast<size_t>(row * scale_columns + column / 32)];
            const float scale = raw_scale == 255u
                ? std::numeric_limits<float>::quiet_NaN()
                : std::ldexp(1.0f, static_cast<int>(raw_scale) - 127);
            const float value = magnitude[code & 7u] * scale;
            destination[row * source.neuron_len + column] =
                (code & 8u) == 0u ? value : -value;
        }
    }
    return dense.to(mfq_tensor_backend::kFloat16);
}

static Mxfp4Cpu select_mxfp4_cpu_rows(
        const Mxfp4Cpu & source,
        const std::vector<int64_t> & rows) {
    Mxfp4Cpu result;
    result.out = static_cast<int64_t>(rows.size());
    result.neuron_len = source.neuron_len;
    const size_t value_row = static_cast<size_t>(source.neuron_len / 2);
    const size_t scale_row = static_cast<size_t>(source.neuron_len / 32);
    result.values.resize(rows.size() * value_row);
    result.scales.resize(rows.size() * scale_row);
    for (size_t destination = 0; destination < rows.size(); ++destination) {
        const int64_t source_row = rows[destination];
        if (source_row < 0 || source_row >= source.out) {
            throw std::runtime_error("MXFP4 selected row is out of range");
        }
        std::memcpy(
            result.values.data() + destination * value_row,
            source.values.data() + static_cast<size_t>(source_row) * value_row,
            value_row);
        std::memcpy(
            result.scales.data() + destination * scale_row,
            source.scales.data() + static_cast<size_t>(source_row) * scale_row,
            scale_row);
    }
    return result;
}

static Mxfp4Cpu slice_mxfp4_cpu(
        const Mxfp4Cpu & source,
        TensorParallelAxis axis,
        int64_t begin,
        int64_t end) {
    if (axis == TensorParallelAxis::Output) {
        if (begin < 0 || begin >= end || end > source.out) {
            throw std::runtime_error("invalid MXFP4 output shard");
        }
        std::vector<int64_t> rows(static_cast<size_t>(end - begin));
        std::iota(rows.begin(), rows.end(), begin);
        return select_mxfp4_cpu_rows(source, rows);
    }
    if (axis != TensorParallelAxis::Input || begin < 0 || begin >= end ||
            end > source.neuron_len || begin % 32 != 0 || end % 32 != 0) {
        throw std::runtime_error(
            "MXFP4 input shards must preserve 32-value scale blocks");
    }
    Mxfp4Cpu result;
    result.out = source.out;
    result.neuron_len = end - begin;
    const size_t source_value_row =
        static_cast<size_t>(source.neuron_len / 2);
    const size_t destination_value_row =
        static_cast<size_t>(result.neuron_len / 2);
    const size_t source_scale_row =
        static_cast<size_t>(source.neuron_len / 32);
    const size_t destination_scale_row =
        static_cast<size_t>(result.neuron_len / 32);
    result.values.resize(static_cast<size_t>(result.out) * destination_value_row);
    result.scales.resize(static_cast<size_t>(result.out) * destination_scale_row);
    for (int64_t row = 0; row < result.out; ++row) {
        std::memcpy(
            result.values.data() + static_cast<size_t>(row) * destination_value_row,
            source.values.data() + static_cast<size_t>(row) * source_value_row +
                static_cast<size_t>(begin / 2),
            destination_value_row);
        std::memcpy(
            result.scales.data() + static_cast<size_t>(row) * destination_scale_row,
            source.scales.data() + static_cast<size_t>(row) * source_scale_row +
                static_cast<size_t>(begin / 32),
            destination_scale_row);
    }
    return result;
}

struct TpqCpu {
    bool int4 = false;
    int64_t out = 0;
    int64_t neuron_len = 0;
    int group_size = 0;
    int vector_size = 0;
    int index_bits = 0;
    int codebook_entries = 0;
    std::vector<uint8_t> packed;
    std::vector<uint16_t> scales_h;
    std::vector<float> codebook;
};

static bool is_tpq_pq_dtype(const std::string & dtype) {
    return dtype == "TPQ-X" || dtype == "TPQ-W" ||
        dtype == "TPQ-V" || dtype == "TPQ-VV" ||
        dtype == "TPQ-P" || dtype == "TPQ-PVQ";
}

static uint32_t read_tpq_index_cpu(
        const std::vector<uint8_t> & packed,
        size_t linear,
        int bits) {
    const size_t bit = linear * static_cast<size_t>(bits);
    const size_t byte = bit >> 3;
    const int shift = static_cast<int>(bit & 7);
    uint32_t value = byte < packed.size() ? packed[byte] : 0u;
    if (byte + 1 < packed.size()) {
        value |= static_cast<uint32_t>(packed[byte + 1]) << 8;
    }
    if (byte + 2 < packed.size()) {
        value |= static_cast<uint32_t>(packed[byte + 2]) << 16;
    }
    return (value >> shift) & ((1u << bits) - 1u);
}

static TpqCpu unpack_tpq_pq(
        const std::vector<uint8_t> & blob,
        const std::string & dtype) {
    constexpr size_t kHeaderBytes = 24;
    if (!is_tpq_pq_dtype(dtype) || blob.size() < kHeaderBytes ||
            std::memcmp(blob.data(), "CPQ1", 4) != 0 || blob[4] != 1) {
        throw std::runtime_error("invalid TPQ-PQ payload header");
    }
    const int tier = blob[5];
    const int vector_size = blob[6];
    const int index_bits = blob[7];
    const int expected_tier = dtype == "TPQ-X" ? 1 :
        (dtype == "TPQ-W" ? 2 :
         (dtype == "TPQ-V" ? 3 :
          (dtype == "TPQ-VV" ? 4 : 5)));
    size_t offset = 8;
    const int32_t axis = read_i32_from(blob, offset);
    const int32_t neuron_len = read_i32_from(blob, offset);
    const uint32_t ndim = read_u32_from(blob, offset);
    const uint32_t codebook_entries = read_u32_from(blob, offset);
    if (tier != expected_tier || axis != 0 || ndim != 2 ||
            neuron_len <= 0 || vector_size <= 0 ||
            neuron_len % vector_size != 0 ||
            index_bits < 8 || index_bits > 16 ||
            codebook_entries <= 1 ||
            codebook_entries > (1u << index_bits)) {
        throw std::runtime_error("invalid TPQ-PQ payload geometry");
    }
    const uint64_t rows_u64 = read_u64_from(blob, offset);
    const uint64_t columns_u64 = read_u64_from(blob, offset);
    const uint32_t rows_tail = read_u32_from(blob, offset);
    if (rows_u64 == 0 ||
            rows_u64 > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
            columns_u64 != static_cast<uint64_t>(neuron_len) ||
            rows_tail != rows_u64) {
        throw std::runtime_error("inconsistent TPQ-PQ matrix dimensions");
    }
    const size_t codebook_values = checked_mxfp8_size(
        codebook_entries, static_cast<uint64_t>(vector_size),
        "TPQ-PQ codebook size");
    const size_t codebook_bytes = checked_mxfp8_size(
        codebook_values, sizeof(float), "TPQ-PQ codebook bytes");
    if (codebook_bytes > blob.size() - offset) {
        throw std::runtime_error("truncated TPQ-PQ codebook");
    }
    TpqCpu result;
    result.out = static_cast<int64_t>(rows_u64);
    result.neuron_len = neuron_len;
    result.vector_size = vector_size;
    result.index_bits = index_bits;
    result.codebook_entries = static_cast<int>(codebook_entries);
    result.codebook.resize(codebook_values);
    std::memcpy(result.codebook.data(), blob.data() + offset, codebook_bytes);
    offset += codebook_bytes;
    if (std::any_of(
            result.codebook.begin(), result.codebook.end(),
            [](float value) { return !std::isfinite(value); })) {
        throw std::runtime_error("TPQ-PQ codebook must be finite");
    }
    const size_t vectors = static_cast<size_t>(neuron_len / vector_size);
    const size_t index_count = checked_mxfp8_size(
        rows_u64, vectors, "TPQ-PQ index count");
    if (index_count > (std::numeric_limits<size_t>::max() - 7) /
            static_cast<size_t>(index_bits)) {
        throw std::runtime_error("TPQ-PQ index stream is too large");
    }
    const size_t index_bytes =
        (index_count * static_cast<size_t>(index_bits) + 7) / 8;
    if (index_bytes != blob.size() - offset) {
        throw std::runtime_error("invalid TPQ-PQ index stream length");
    }
    result.packed.assign(
        blob.begin() + static_cast<ptrdiff_t>(offset), blob.end());
    for (size_t linear = 0; linear < index_count; ++linear) {
        if (read_tpq_index_cpu(result.packed, linear, index_bits) >=
                codebook_entries) {
            throw std::runtime_error(
                "TPQ-PQ index references a missing codeword");
        }
    }
    return result;
}

static TpqCpu unpack_tpq_int4(const std::vector<uint8_t> & blob) {
    constexpr size_t kHeaderBytes = 48;
    if (blob.size() < kHeaderBytes ||
            std::memcmp(blob.data(), "CI41", 4) != 0 || blob[4] != 1 ||
            blob[5] != 0 || blob[6] != 0 || blob[7] != 0) {
        throw std::runtime_error("invalid TPQ-I4 payload header");
    }
    size_t offset = 8;
    const uint32_t group_size = read_u32_from(blob, offset);
    const int32_t axis = read_i32_from(blob, offset);
    const int32_t neuron_len = read_i32_from(blob, offset);
    const uint32_t ndim = read_u32_from(blob, offset);
    const uint64_t rows_u64 = read_u64_from(blob, offset);
    const uint64_t columns_u64 = read_u64_from(blob, offset);
    const uint32_t rows_tail = read_u32_from(blob, offset);
    const uint32_t groups_tail = read_u32_from(blob, offset);
    if (group_size != 64 || axis != 0 || ndim != 2 || neuron_len <= 0 ||
            neuron_len % static_cast<int>(group_size) != 0 ||
            columns_u64 != static_cast<uint64_t>(neuron_len) ||
            rows_u64 == 0 || rows_tail != rows_u64 ||
            groups_tail != static_cast<uint32_t>(neuron_len) / group_size ||
            rows_u64 > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        throw std::runtime_error("invalid TPQ-I4 payload geometry");
    }
    const size_t packed_bytes = checked_mxfp8_size(
        rows_u64, static_cast<uint64_t>(neuron_len / 2),
        "TPQ-I4 packed size");
    const size_t scale_count = checked_mxfp8_size(
        rows_u64, groups_tail, "TPQ-I4 scale count");
    const size_t scale_bytes = checked_mxfp8_size(
        scale_count, sizeof(uint16_t), "TPQ-I4 scale bytes");
    if (packed_bytes > blob.size() - offset ||
            scale_bytes != blob.size() - offset - packed_bytes) {
        throw std::runtime_error("invalid TPQ-I4 payload length");
    }
    TpqCpu result;
    result.int4 = true;
    result.out = static_cast<int64_t>(rows_u64);
    result.neuron_len = neuron_len;
    result.group_size = static_cast<int>(group_size);
    result.packed.assign(
        blob.begin() + static_cast<ptrdiff_t>(offset),
        blob.begin() + static_cast<ptrdiff_t>(offset + packed_bytes));
    offset += packed_bytes;
    result.scales_h.resize(scale_count);
    std::memcpy(result.scales_h.data(), blob.data() + offset, scale_bytes);
    for (uint16_t raw : result.scales_h) {
        const bool nonzero_negative =
            (raw & 0x8000u) != 0 && (raw & 0x7fffu) != 0;
        if ((raw & 0x7c00u) == 0x7c00u || nonzero_negative) {
            throw std::runtime_error(
                "TPQ-I4 scales must be finite and non-negative");
        }
    }
    return result;
}

static void write_tpq_index_cpu(
        std::vector<uint8_t> & packed,
        size_t linear,
        int bits,
        uint32_t value) {
    const size_t bit = linear * static_cast<size_t>(bits);
    for (int index = 0; index < bits; ++index) {
        if (((value >> index) & 1u) != 0) {
            packed[(bit + static_cast<size_t>(index)) >> 3] |=
                static_cast<uint8_t>(1u << ((bit + index) & 7));
        }
    }
}

static TpqCpu slice_tpq_cpu(
        const TpqCpu & source,
        TensorParallelAxis axis,
        int64_t begin,
        int64_t end) {
    if (begin < 0 || begin >= end) {
        throw std::runtime_error("invalid TPQ tensor-parallel shard");
    }
    if (axis == TensorParallelAxis::Output) {
        if (end > source.out) {
            throw std::runtime_error("invalid TPQ output shard");
        }
        TpqCpu result = source;
        result.out = end - begin;
        if (source.int4) {
            const size_t packed_row = static_cast<size_t>(source.neuron_len / 2);
            const size_t scale_row = static_cast<size_t>(
                source.neuron_len / source.group_size);
            result.packed.assign(
                source.packed.begin() + static_cast<ptrdiff_t>(begin * packed_row),
                source.packed.begin() + static_cast<ptrdiff_t>(end * packed_row));
            result.scales_h.assign(
                source.scales_h.begin() + static_cast<ptrdiff_t>(begin * scale_row),
                source.scales_h.begin() + static_cast<ptrdiff_t>(end * scale_row));
        } else {
            const size_t vectors = static_cast<size_t>(
                source.neuron_len / source.vector_size);
            const size_t count = static_cast<size_t>(result.out) * vectors;
            result.packed.assign(
                (count * source.index_bits + 7) / 8, 0);
            for (size_t row = 0; row < static_cast<size_t>(result.out); ++row) {
                for (size_t vector = 0; vector < vectors; ++vector) {
                    const uint32_t code = read_tpq_index_cpu(
                        source.packed,
                        (static_cast<size_t>(begin) + row) * vectors + vector,
                        source.index_bits);
                    write_tpq_index_cpu(
                        result.packed, row * vectors + vector,
                        source.index_bits, code);
                }
            }
        }
        return result;
    }
    const int granularity = source.int4
        ? source.group_size : source.vector_size;
    if (axis != TensorParallelAxis::Input || end > source.neuron_len ||
            begin % granularity != 0 || end % granularity != 0) {
        throw std::runtime_error(
            "TPQ input shard does not preserve packed vector boundaries");
    }
    TpqCpu result = source;
    result.neuron_len = end - begin;
    if (source.int4) {
        const size_t source_packed_row =
            static_cast<size_t>(source.neuron_len / 2);
        const size_t result_packed_row =
            static_cast<size_t>(result.neuron_len / 2);
        const size_t source_scale_row = static_cast<size_t>(
            source.neuron_len / source.group_size);
        const size_t result_scale_row = static_cast<size_t>(
            result.neuron_len / result.group_size);
        result.packed.resize(static_cast<size_t>(result.out) * result_packed_row);
        result.scales_h.resize(static_cast<size_t>(result.out) * result_scale_row);
        for (int64_t row = 0; row < result.out; ++row) {
            std::memcpy(
                result.packed.data() + static_cast<size_t>(row) * result_packed_row,
                source.packed.data() + static_cast<size_t>(row) * source_packed_row +
                    static_cast<size_t>(begin / 2),
                result_packed_row);
            std::memcpy(
                result.scales_h.data() + static_cast<size_t>(row) * result_scale_row,
                source.scales_h.data() + static_cast<size_t>(row) * source_scale_row +
                    static_cast<size_t>(begin / source.group_size),
                result_scale_row * sizeof(uint16_t));
        }
    } else {
        const size_t source_vectors = static_cast<size_t>(
            source.neuron_len / source.vector_size);
        const size_t first_vector = static_cast<size_t>(begin / source.vector_size);
        const size_t result_vectors = static_cast<size_t>(
            result.neuron_len / result.vector_size);
        const size_t count = static_cast<size_t>(result.out) * result_vectors;
        result.packed.assign((count * source.index_bits + 7) / 8, 0);
        for (size_t row = 0; row < static_cast<size_t>(result.out); ++row) {
            for (size_t vector = 0; vector < result_vectors; ++vector) {
                const uint32_t code = read_tpq_index_cpu(
                    source.packed,
                    row * source_vectors + first_vector + vector,
                    source.index_bits);
                write_tpq_index_cpu(
                    result.packed, row * result_vectors + vector,
                    source.index_bits, code);
            }
        }
    }
    return result;
}

static Mxfp8Cpu slice_mxfp8_cpu(
        const Mxfp8Cpu & source,
        TensorParallelAxis axis,
        int64_t begin,
        int64_t end) {
    if (axis != TensorParallelAxis::Output &&
            axis != TensorParallelAxis::Input) {
        throw std::runtime_error("MXFP8 slicing requires output or input axis");
    }
    const int64_t extent = axis == TensorParallelAxis::Output
        ? source.out : source.neuron_len;
    if (begin < 0 || begin >= end || end > extent ||
            begin % 128 != 0 ||
            (end != extent && end % 128 != 0)) {
        throw std::runtime_error(
            "MXFP8 tensor-parallel shards must preserve 128-element blocks");
    }
    Mxfp8Cpu result;
    if (axis == TensorParallelAxis::Output) {
        result.out = end - begin;
        result.neuron_len = source.neuron_len;
        const size_t value_begin =
            static_cast<size_t>(begin * source.neuron_len);
        const size_t value_end =
            static_cast<size_t>(end * source.neuron_len);
        result.values.assign(
            source.values.begin() + static_cast<ptrdiff_t>(value_begin),
            source.values.begin() + static_cast<ptrdiff_t>(value_end));
        const int64_t scale_columns = source.neuron_len / 128;
        const size_t scale_begin =
            static_cast<size_t>((begin / 128) * scale_columns);
        const size_t scale_end = static_cast<size_t>(
            ((end + 127) / 128) * scale_columns);
        result.scales.assign(
            source.scales.begin() + static_cast<ptrdiff_t>(scale_begin),
            source.scales.begin() + static_cast<ptrdiff_t>(scale_end));
        return result;
    }

    result.out = source.out;
    result.neuron_len = end - begin;
    result.values.resize(
        static_cast<size_t>(result.out * result.neuron_len));
    for (int64_t row = 0; row < source.out; ++row) {
        std::memcpy(
            result.values.data() +
                static_cast<size_t>(row * result.neuron_len),
            source.values.data() +
                static_cast<size_t>(row * source.neuron_len + begin),
            static_cast<size_t>(result.neuron_len));
    }
    const int64_t source_scale_columns = source.neuron_len / 128;
    const int64_t result_scale_columns = result.neuron_len / 128;
    const int64_t scale_rows = (source.out + 127) / 128;
    result.scales.resize(
        static_cast<size_t>(scale_rows * result_scale_columns));
    for (int64_t row = 0; row < scale_rows; ++row) {
        std::memcpy(
            result.scales.data() +
                static_cast<size_t>(row * result_scale_columns),
            source.scales.data() + static_cast<size_t>(
                row * source_scale_columns + begin / 128),
            static_cast<size_t>(result_scale_columns));
    }
    return result;
}

static mfq_tensor_backend::Tensor cpu_u8_tensor(const std::vector<uint8_t> & v, std::initializer_list<int64_t> shape) {
    return mfq_tensor_backend::from_blob((void *)v.data(), shape, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kUInt8)).clone();
}

static mfq_tensor_backend::Tensor cpu_i64_tensor(
        const std::vector<int64_t> & v,
        std::initializer_list<int64_t> shape) {
    return mfq_tensor_backend::from_blob(
        (void *)v.data(), shape,
        mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64)).clone();
}

static mfq_tensor_backend::Tensor cpu_i32_tensor(
        const std::vector<int32_t> & v,
        std::initializer_list<int64_t> shape) {
    return mfq_tensor_backend::from_blob(
        (void *)v.data(), shape,
        mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt32)).clone();
}

static mfq_tensor_backend::Tensor cpu_f16_tensor(
        const std::vector<uint16_t> & v,
        std::initializer_list<int64_t> shape) {
    return mfq_tensor_backend::from_blob(
        (void *)v.data(), shape,
        mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kFloat16)).clone();
}

static mfq_tensor_backend::Tensor cpu_f16_to_f32_tensor(const std::vector<uint16_t> & v, int64_t n) {
    auto h = mfq_tensor_backend::from_blob((void *)v.data(), {n}, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kFloat16)).clone();
    return h.to(mfq_tensor_backend::kFloat32).contiguous();
}



static Mxfp8Weight to_device_mxfp8(
        const Mxfp8Cpu & source,
        bool cuda,
        int device = -1) {
    Mxfp8Weight result;
    result.out = source.out;
    result.neuron_len = source.neuron_len;
    result.values = cpu_u8_tensor(
        source.values,
        {source.out, source.neuron_len});
    result.scales = cpu_u8_tensor(
        source.scales,
        {(source.out + 127) / 128, source.neuron_len / 128});
    if (cuda) {
        const int target_device = device >= 0
            ? device : mfq_current_cuda_device();
        MfqCudaGuard guard(target_device);
        const auto target = mfq_tensor_backend::Device(mfq_tensor_backend::kCUDA, target_device);
        result.values = result.values.to(target, false, false).contiguous();
        result.scales = result.scales.to(target, false, false).contiguous();
    }
    return result;
}

static Mxfp8Weight to_cuda_device_mxfp8(
        const Mxfp8Cpu & source,
        int device) {
    return to_device_mxfp8(source, true, device);
}





static NintWeight to_device_nint(const NintCpu & c, bool cuda) {
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
    w.uniform_q46 = c.gs % 4 == 0 && (c.bits == 4 || c.bits == 6)
        ? static_cast<int>(c.bits) : 0;
    for (size_t row = 0; row < c.row_q_bits.size(); ++row) {
        w.aligned_q8 = w.aligned_q8 && c.row_q_bits[row] == 8 &&
            (c.row_q_bit_offsets[row] & 31) == 0;
        if (c.row_q_bits[row] != w.uniform_q46 ||
            (c.row_q_bit_offsets[row] & 7) != 0) {
            w.uniform_q46 = 0;
        }
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

static NintWeight to_gpu_nint(const NintCpu & c) {
    return to_device_nint(c, true);
}

static NintWeight to_cuda_device_nint(
        const NintCpu & c, int device) {
    MfqCudaGuard guard(device);
    return to_device_nint(c, true);
}

static NintWeight to_cpu_nint(const NintCpu & c) {
    return to_device_nint(c, false);
}

static NintWeight to_device_mfe_nint(
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
    result.uniform_q46 = 0;
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

static NintWeight to_device_nint8_zero(
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

static NintWeight to_gpu_nint8_zero(const Nint8ZeroCpu & c) {
    return to_device_nint8_zero(c, true);
}

static NintWeight to_cuda_device_nint8_zero(
        const Nint8ZeroCpu & c, int device) {
    MfqCudaGuard guard(device);
    return to_device_nint8_zero(c, true);
}

static NintWeight to_cpu_nint8_zero(const Nint8ZeroCpu & c) {
    return to_device_nint8_zero(c, false);
}

static NintWeight load_nint_gpu(const mfq::ModelSource & mfq, const std::string & name) {
    if (require_tensor(mfq, name).dtype == "NINT8-0") {
        return to_gpu_nint8_zero(unpack_nint8_zero(read_tensor(mfq, name)));
    }
    return to_gpu_nint(unpack_nint(read_tensor(mfq, name)));
}

struct MfeCpuPool {
    std::vector<int32_t> expert_ids;
    std::string dtype;
    std::vector<uint8_t> payload;
    std::vector<uint8_t> runtime_payload;
    NintCpu weight;
    Nint8ZeroCpu q8_zero;
    Mxfp4Cpu mxfp4;
    TpqCpu tpq;
};

struct MfeCpu {
    int n_experts = 0;
    int out_per_expert = 0;
    int neuron_len = 0;
    std::vector<MfeCpuPool> pools;
};

static MfeCpu unpack_mfe_impl(
        const std::vector<uint8_t> & blob,
        bool allow_partial) {
    if (blob.size() < 20) {
        throw std::runtime_error("invalid MFE header");
    }
    const bool version1 = std::memcmp(blob.data(), "NIM1", 4) == 0;
    const bool delta =
        std::memcmp(blob.data(), "MFD1", 4) == 0 ||
        std::memcmp(blob.data(), "NID2", 4) == 0;
    const bool version2 =
        std::memcmp(blob.data(), "MFE1", 4) == 0 ||
        std::memcmp(blob.data(), "NIM2", 4) == 0 || delta;
    if (!version1 && !version2) throw std::runtime_error("invalid MFE header");
    if (delta && !allow_partial) {
        throw std::runtime_error("MFE delta cannot be loaded as a full tensor");
    }
    size_t off = 4;
    MfeCpu result;
    result.n_experts = (int)read_u32_from(blob, off);
    result.out_per_expert = (int)read_u32_from(blob, off);
    result.neuron_len = (int)read_u32_from(blob, off);
    int pool_count = (int)read_u32_from(blob, off);
    if (result.n_experts <= 0 || result.out_per_expert <= 0 || result.neuron_len <= 0 ||
        pool_count <= 0 || pool_count > result.n_experts) {
        throw std::runtime_error("invalid MFE dimensions");
    }
    std::vector<int> owners((size_t)result.n_experts, -1);
    result.pools.reserve((size_t)pool_count);
    for (int pool_index = 0; pool_index < pool_count; ++pool_index) {
        const size_t pool_header_bytes = version1 ? 12 : 24;
        if (off + pool_header_bytes > blob.size()) {
            throw std::runtime_error("truncated MFE pool header");
        }
        uint32_t expert_count = read_u32_from(blob, off);
        uint32_t dtype_nbytes = version1 ? 0 : read_u32_from(blob, off);
        uint64_t payload_nbytes = read_u64_from(blob, off);
        uint64_t runtime_nbytes = version1 ? 0 : read_u64_from(blob, off);
        if (expert_count == 0 || expert_count > static_cast<uint32_t>(result.n_experts) ||
                expert_count > static_cast<uint32_t>(
                    std::numeric_limits<int>::max() / result.out_per_expert) ||
                dtype_nbytes > 32) {
            throw std::runtime_error("invalid MFE pool dimensions");
        }
        size_t ids_nbytes = (size_t)expert_count * sizeof(int32_t);
        size_t remaining = blob.size() - off;
        if (ids_nbytes > remaining) {
            throw std::runtime_error("truncated MFE pool payload");
        }
        remaining -= ids_nbytes;
        if (dtype_nbytes > remaining) {
            throw std::runtime_error("truncated MFE pool payload");
        }
        remaining -= dtype_nbytes;
        if (runtime_nbytes > remaining) {
            throw std::runtime_error("truncated MFE pool payload");
        }
        remaining -= static_cast<size_t>(runtime_nbytes);
        if (payload_nbytes > remaining) {
            throw std::runtime_error("truncated MFE pool payload");
        }
        MfeCpuPool pool;
        pool.expert_ids.resize(expert_count);
        std::memcpy(pool.expert_ids.data(), blob.data() + off, ids_nbytes);
        off += ids_nbytes;
        for (int32_t expert : pool.expert_ids) {
            if (expert < 0 || expert >= result.n_experts || owners[(size_t)expert] >= 0) {
                throw std::runtime_error("invalid or duplicate MFE expert id");
            }
            owners[(size_t)expert] = pool_index;
        }
        if (version2) {
            if (dtype_nbytes == 0) throw std::runtime_error("empty MFE pool dtype");
            pool.dtype.assign(
                reinterpret_cast<const char *>(blob.data() + off), dtype_nbytes);
            pool.dtype = std::string(
                mfq::canonical_format_dtype(pool.dtype));
            off += dtype_nbytes;
            pool.runtime_payload.assign(
                blob.begin() + (ptrdiff_t)off,
                blob.begin() + (ptrdiff_t)(off + (size_t)runtime_nbytes));
            off += (size_t)runtime_nbytes;
        }
        size_t payload_end = off + (size_t)payload_nbytes;
        pool.payload.assign(
            blob.begin() + (ptrdiff_t)off, blob.begin() + (ptrdiff_t)payload_end);
        off = payload_end;
        if (!version1 && mfq::fp8sq::is_dtype(pool.dtype)) {
            if (!pool.runtime_payload.empty()) {
                throw std::runtime_error(
                    "unexpected MFE FP8-SQ runtime metadata");
            }
            const auto layout = mfq::fp8sq::parse(
                pool.dtype, pool.payload.data(), pool.payload.size());
            const int expected_rows =
                static_cast<int>(expert_count) * result.out_per_expert;
            if (layout.outputs != expected_rows ||
                    layout.width != result.neuron_len) {
                throw std::runtime_error(
                    "MFE FP8-SQ pool weight shape mismatch");
            }
        } else if (!version1 && pool.dtype == "MXFP4-SQ") {
            if (!pool.runtime_payload.empty()) {
                throw std::runtime_error(
                    "unexpected MFE MXFP4-SQ runtime metadata");
            }
            const auto layout = mfq::sq::parse(
                pool.payload.data(), pool.payload.size());
            const int expected_rows =
                static_cast<int>(expert_count) * result.out_per_expert;
            if (layout.outputs != expected_rows ||
                    layout.width != result.neuron_len) {
                throw std::runtime_error(
                    "MFE MXFP4-SQ pool weight shape mismatch");
            }
        } else if (!version1 && pool.dtype == "MXFP4") {
            pool.mxfp4 = unpack_mxfp4(pool.payload);
            const int expected_rows =
                static_cast<int>(expert_count) * result.out_per_expert;
            if (pool.mxfp4.out != expected_rows ||
                    pool.mxfp4.neuron_len != result.neuron_len) {
                throw std::runtime_error(
                    "MFE MXFP4 pool weight shape mismatch");
            }
        } else if (!version1 && is_tpq_pq_dtype(pool.dtype)) {
            pool.tpq = unpack_tpq_pq(pool.payload, pool.dtype);
            const int expected_rows =
                static_cast<int>(expert_count) * result.out_per_expert;
            if (pool.tpq.out != expected_rows ||
                    pool.tpq.neuron_len != result.neuron_len) {
                throw std::runtime_error(
                    "MFE TPQ-PQ pool weight shape mismatch");
            }
        } else if (!version1 && pool.dtype == "NINT8-0") {
            pool.q8_zero = unpack_nint8_zero(pool.payload);
            const int expected_rows =
                static_cast<int>(expert_count) * result.out_per_expert;
            if (pool.q8_zero.axis != 0 ||
                pool.q8_zero.out != expected_rows ||
                pool.q8_zero.neuron_len != result.neuron_len ||
                pool.q8_zero.shape.size() != 2 ||
                pool.q8_zero.shape[0] != expected_rows ||
                pool.q8_zero.shape[1] != result.neuron_len) {
                throw std::runtime_error(
                    "MFE NINT8-0 pool weight shape mismatch");
            }
        } else if (version1 || pool.dtype == "NINT") {
            pool.weight = unpack_nint(pool.payload);
            if (version1) pool.dtype = "NINT";
            int expected_rows = (int)expert_count * result.out_per_expert;
            if (pool.weight.axis != 0 || pool.weight.out != expected_rows ||
                pool.weight.neuron_len != result.neuron_len ||
                pool.weight.shape.size() != 2 ||
                pool.weight.shape[0] != expected_rows ||
                pool.weight.shape[1] != result.neuron_len) {
                throw std::runtime_error("MFE pool weight shape mismatch");
            }
        }
        result.pools.push_back(std::move(pool));
    }
    if (off != blob.size() ||
        (!allow_partial &&
         std::find(owners.begin(), owners.end(), -1) != owners.end())) {
        throw std::runtime_error("MFE expert coverage or tail mismatch");
    }
    return result;
}

static MfeCpu unpack_mfe(const std::vector<uint8_t> & blob) {
    return unpack_mfe_impl(blob, false);
}

static MfeCpu unpack_mfe_delta(
        const std::vector<uint8_t> & blob) {
    return unpack_mfe_impl(blob, true);
}















static Fp8SqWeight to_device_fp8_sq(
        std::string_view dtype,
        const std::vector<uint8_t> & payload,
        bool cuda,
        int device = -1) {
    const auto layout = mfq::fp8sq::parse(
        dtype, payload.data(), payload.size());
    const auto rows = mfq::fp8sq::row_metadata(payload.data(), layout);
    Fp8SqWeight result;
    result.dtype = std::string(dtype);
    result.out = layout.outputs;
    result.neuron_len = layout.width;
    result.block_rows = layout.block_rows;
    result.block_columns = layout.block_columns;
    result.scale_rows = layout.scale_rows;
    result.scale_columns = layout.scale_columns;
    result.scale_kind = static_cast<int64_t>(layout.scale_kind);
    result.palettes_offset = static_cast<int64_t>(layout.palettes);
    result.symbols_offset = static_cast<int64_t>(layout.symbols);
    result.scales_offset = static_cast<int64_t>(layout.scales);
    result.format_version = layout.version;
    result.aggregate_bpw = static_cast<double>(
        layout.bytes - mfq::fp8sq::kHeaderBytes) * 8.0 /
        (static_cast<double>(layout.outputs) * layout.width);
    std::array<size_t, 8> q_counts{};
    std::vector<int32_t> symbol_offsets(rows.symbol_byte_offsets.size());
    for (size_t row = 0; row < rows.q.size(); ++row) {
        const int q = rows.q[row];
        ++q_counts[static_cast<size_t>(q - 1)];
        if (rows.symbol_byte_offsets[row] >
                static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
            throw std::runtime_error(
                "FP8-SQ runtime row metadata exceeds int32 limits");
        }
        symbol_offsets[row] =
            static_cast<int32_t>(rows.symbol_byte_offsets[row]);
    }
    for (const auto count : q_counts) {
        if (count == 0) continue;
        const double probability =
            static_cast<double>(count) / layout.outputs;
        result.distribution_entropy -= probability * std::log2(probability);
    }
    result.blob = cpu_u8_tensor(
        payload, {static_cast<int64_t>(payload.size())});
    result.row_q = cpu_u8_tensor(rows.q, {layout.outputs});
    result.row_symbol_byte_offsets = cpu_i32_tensor(
        symbol_offsets, {layout.outputs});
    if (cuda) {
        const int target_device = device >= 0
            ? device : mfq_current_cuda_device();
        MfqCudaGuard guard(target_device);
        const auto target = mfq_tensor_backend::Device(
            mfq_tensor_backend::kCUDA, target_device);
        result.blob = result.blob.to(target, false, false).contiguous();
        result.row_q = result.row_q.to(target, false, false).contiguous();
        result.row_symbol_byte_offsets =
            result.row_symbol_byte_offsets.to(
                target, false, false).contiguous();
    }
    return result;
}

static Mxfp4SqWeight to_device_mxfp4_sq(
        const std::vector<uint8_t> & payload,
        bool cuda,
        int device = -1) {
    const auto layout = mfq::sq::parse(payload.data(), payload.size());
    const auto rows = mfq::sq::row_metadata(payload.data(), layout);
    Mxfp4SqWeight result;
    result.bits = layout.bits;
    result.out = layout.outputs;
    result.neuron_len = layout.width;
    result.matrix_scale_base = layout.base;
    result.sq4_rows = layout.sq4_rows;
    result.format_version = layout.version;
    result.aggregate_bpw = static_cast<double>(layout.bytes - 24) * 8.0 /
        (static_cast<double>(layout.outputs) * layout.width);
    std::array<size_t, 4> q_counts{};
    std::vector<int32_t> symbol_offsets(rows.symbol_byte_offsets.size());
    std::vector<int32_t> auxiliary_rows(rows.auxiliary_rows.size());
    for (size_t row = 0; row < rows.q.size(); ++row) {
        const int q = rows.q[row];
        result.q_sum += q;
        ++q_counts[static_cast<size_t>(q - 1)];
        if (rows.symbol_byte_offsets[row] >
                static_cast<uint32_t>(std::numeric_limits<int32_t>::max()) ||
                rows.auxiliary_rows[row] >
                static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
            throw std::runtime_error(
                "MXFP4-SQ runtime row metadata exceeds int32 limits");
        }
        symbol_offsets[row] =
            static_cast<int32_t>(rows.symbol_byte_offsets[row]);
        auxiliary_rows[row] =
            static_cast<int32_t>(rows.auxiliary_rows[row]);
    }
    for (const auto count : q_counts) {
        if (count == 0) continue;
        const double probability =
            static_cast<double>(count) / layout.outputs;
        result.distribution_entropy -= probability * std::log2(probability);
    }
    result.blob = cpu_u8_tensor(
        payload,
        {static_cast<int64_t>(payload.size())});
    result.row_q = cpu_u8_tensor(rows.q, {layout.outputs});
    result.row_symbol_byte_offsets = cpu_i32_tensor(
        symbol_offsets, {layout.outputs});
    result.row_auxiliary = cpu_i32_tensor(
        auxiliary_rows, {layout.outputs});
    if (cuda) {
        const int target_device = device >= 0
            ? device : mfq_current_cuda_device();
        MfqCudaGuard guard(target_device);
        result.blob = result.blob.to(
            mfq_tensor_backend::Device(
                mfq_tensor_backend::kCUDA,
                target_device),
            false,
            false).contiguous();
        result.row_q = result.row_q.to(
            mfq_tensor_backend::Device(
                mfq_tensor_backend::kCUDA,
                target_device),
            false,
            false).contiguous();
        result.row_symbol_byte_offsets =
            result.row_symbol_byte_offsets.to(
                mfq_tensor_backend::Device(
                    mfq_tensor_backend::kCUDA,
                    target_device),
                false,
                false).contiguous();
        result.row_auxiliary = result.row_auxiliary.to(
            mfq_tensor_backend::Device(
                mfq_tensor_backend::kCUDA,
                target_device),
            false,
            false).contiguous();
    }
    return result;
}

static Mxfp4Weight to_device_mxfp4(
        const Mxfp4Cpu & source, bool cuda, int device = -1) {
    Mxfp4Weight result;
    result.out = source.out;
    result.neuron_len = source.neuron_len;
    result.values = cpu_u8_tensor(
        source.values, {source.out, source.neuron_len / 2});
    result.scales = cpu_u8_tensor(
        source.scales, {source.out, source.neuron_len / 32});
    if (cuda) {
        const int target_device = device >= 0
            ? device : mfq_current_cuda_device();
        MfqCudaGuard guard(target_device);
        const auto target = mfq_tensor_backend::Device(mfq_tensor_backend::kCUDA, target_device);
        result.values = result.values.to(target, false, false).contiguous();
        result.scales = result.scales.to(target, false, false).contiguous();
    }
    return result;
}





static std::atomic<uint64_t> g_moe_route_generation{1};

mfq_tensor_backend::Tensor tensor_to_cuda_device(
    mfq_tensor_backend::Tensor value,
    int device,
    mfq_tensor_backend::Tensor reusable);

mfq_tensor_backend::Tensor moe_tensor_to_device(
        mfq_tensor_backend::Tensor value, int device) {
    return value.defined()
        ? tensor_to_cuda_device(std::move(value), device)
        : value;
}

struct MoeRouteReplicaEntry {
    uint64_t generation = 0;
    MoeRoutePlan plan;
};

class MoeRouteReplicaCache {
public:
    const MoeRoutePlan & get(
            const MoeRoutePlan & source, int device) {
        auto & entry = entries_[device];
        if (entry.generation == source.generation &&
                entry.plan.generation == source.generation) {
            return entry.plan;
        }
        copy_plan(entry.plan, source, device);
        entry.generation = source.generation;
        return entry.plan;
    }

private:
    static void copy_tensor(
            mfq_tensor_backend::Tensor & destination,
            const mfq_tensor_backend::Tensor & source,
            int device) {
        if (!source.defined()) {
            destination = mfq_tensor_backend::Tensor();
            return;
        }
        if (source.is_cuda() && source.get_device() == device) {
            destination = source.contiguous();
            return;
        }
        destination = tensor_to_cuda_device(
            source, device, std::move(destination));
    }

    static void copy_plan(
            MoeRoutePlan & destination,
            const MoeRoutePlan & source,
            int device) {
        copy_tensor(destination.ids, source.ids, device);
        copy_tensor(destination.ids_dst, source.ids_dst, device);
        copy_tensor(destination.expert_bounds, source.expert_bounds, device);
        copy_tensor(destination.tile_bounds, source.tile_bounds, device);
        copy_tensor(destination.tile_experts, source.tile_experts, device);
        if (source.mma_tile_m == 8) {
            destination.mma_tile_bounds = destination.tile_bounds;
            destination.mma_tile_experts = destination.tile_experts;
        } else {
            copy_tensor(destination.mma_tile_bounds, source.mma_tile_bounds, device);
            copy_tensor(destination.mma_tile_experts, source.mma_tile_experts, device);
        }
        if (source.wide_tile_m == 8) {
            destination.wide_tile_bounds = destination.tile_bounds;
            destination.wide_tile_experts = destination.tile_experts;
        } else if (source.wide_tile_m == source.mma_tile_m) {
            destination.wide_tile_bounds = destination.mma_tile_bounds;
            destination.wide_tile_experts = destination.mma_tile_experts;
        } else {
            copy_tensor(destination.wide_tile_bounds, source.wide_tile_bounds, device);
            copy_tensor(destination.wide_tile_experts, source.wide_tile_experts, device);
        }
        copy_tensor(destination.counts, source.counts, device);
        copy_tensor(destination.cursors, source.cursors, device);
        destination.n_experts = source.n_experts;
        destination.mma_tile_m = source.mma_tile_m;
        destination.wide_tile_m = source.wide_tile_m;
        destination.map_ready = source.map_ready;
        destination.generation = source.generation;
        destination.host_unique_experts = source.host_unique_experts;
    }

    std::unordered_map<int, MoeRouteReplicaEntry> entries_;
};

static thread_local MoeRouteReplicaCache g_moe_route_replica_cache;

MoeRoutePlan moe_route_to_device(
        const MoeRoutePlan & source, int device) {
    return g_moe_route_replica_cache.get(source, device);
}

MoeRoutePlan build_moe_route_plan(mfq_tensor_backend::Tensor ids, int n_experts) {
    if (!ids.is_cuda() || ids.scalar_type() != mfq_tensor_backend::kInt32 || ids.dim() != 2) {
        throw std::runtime_error("MoE ids must be CUDA int32 [tokens, routes]");
    }
    MoeRoutePlan result;
    result.generation = g_moe_route_generation.fetch_add(
        1, std::memory_order_relaxed);
    result.ids = ids.contiguous();
    result.n_experts = n_experts;
    auto empty = mfq_tensor_backend::empty({0}, result.ids.options());
    result.ids_dst = empty;
    result.expert_bounds = empty;
    result.tile_bounds = empty;
    result.tile_experts = empty;
    result.mma_tile_bounds = empty;
    result.mma_tile_experts = empty;
    result.wide_tile_bounds = empty;
    result.wide_tile_experts = empty;
    result.counts = empty;
    result.cursors = empty;
    if (result.ids.size(0) > 8) {
        const int64_t rows_per_expert = std::max<int64_t>(
            1, (result.ids.numel() + n_experts - 1) / n_experts);
        const bool use_coarse_mma = rows_per_expert >= 4;
        const int mma_tile_m = rows_per_expert <= 16
            ? 16 : (rows_per_expert <= 32 ? 32 : 64);
        auto mapped = !use_coarse_mma
            ? moe_build_expert_map_cuda(result.ids, n_experts, 8)
            : (rows_per_expert > 64
                ? moe_build_expert_maps_cuda(
                    result.ids, n_experts, 8, mma_tile_m, 128)
                : moe_build_expert_maps_cuda(
                    result.ids, n_experts, 8, mma_tile_m, 0));
        result.ids_dst = mapped.at(0);
        result.expert_bounds = mapped.at(1);
        result.tile_bounds = mapped.at(2);
        result.tile_experts = mapped.at(3);
        result.counts = mapped.at(4);
        if (use_coarse_mma) {
            result.mma_tile_bounds = mapped.at(5);
            result.mma_tile_experts = mapped.at(6);
            result.mma_tile_m = mma_tile_m;
            if (rows_per_expert > 64) {
                result.wide_tile_bounds = mapped.at(7);
                result.wide_tile_experts = mapped.at(8);
                result.wide_tile_m = 128;
            } else {
                result.wide_tile_bounds = result.mma_tile_bounds;
                result.wide_tile_experts = result.mma_tile_experts;
                result.wide_tile_m = result.mma_tile_m;
            }
        } else {
            result.mma_tile_bounds = result.tile_bounds;
            result.mma_tile_experts = result.tile_experts;
            result.mma_tile_m = 8;
            result.wide_tile_bounds = result.tile_bounds;
            result.wide_tile_experts = result.tile_experts;
            result.wide_tile_m = 8;
        }
    }
    result.map_ready = result.ids.size(0) <= 8 ||
        result.ids_dst.numel() == result.ids.numel();
    return result;
}

int select_nint_prefill_route_tile(
        const MoeRoutePlan & route,
        int tokens,
        int routes,
        int experts) {
    const int64_t pairs = static_cast<int64_t>(tokens) * routes;
    const int64_t rows_per_expert = std::max<int64_t>(
        1, (pairs + experts - 1) / experts);
    if (rows_per_expert < 4) {
        return 8;
    }
    if (rows_per_expert > 64 && route.wide_tile_m == 128) {
        return 128;
    }
    return route.mma_tile_m == 16 || route.mma_tile_m == 32 ||
            route.mma_tile_m == 64
        ? route.mma_tile_m
        : 8;
}

const mfq_tensor_backend::Tensor & nint_route_tile_bounds(
        const MoeRoutePlan & route, int tile_m) {
    return tile_m == 128
        ? route.wide_tile_bounds
        : (tile_m == 8 ? route.tile_bounds : route.mma_tile_bounds);
}

const mfq_tensor_backend::Tensor & nint_route_tile_experts(
        const MoeRoutePlan & route, int tile_m) {
    return tile_m == 128
        ? route.wide_tile_experts
        : (tile_m == 8 ? route.tile_experts : route.mma_tile_experts);
}

mfq_tensor_backend::Tensor reduce_model_parallel_outputs(
    std::vector<mfq_tensor_backend::Tensor> outputs);



static void initialize_mfe_nint_runtime(MfeWeight & result) {
    if (result.pools.empty()) {
        throw std::runtime_error("MFE NINT runtime requires at least one pool");
    }
    result.activation_geometries.clear();
    result.activation_workspace_domain = 1;
    std::unordered_set<int64_t> quantized_shapes;
    const auto target = result.pools.front().weight.q_packed.device();
    for (const auto & pool : result.pools) {
        MFQ_RUNTIME_CHECK(
            pool.weight.q_packed.is_cuda() &&
            pool.weight.q_packed.device() == target,
            "MFE NINT pools must share one CUDA device");
        const int64_t activation_key =
            (pool.weight.gs << 32) ^ pool.weight.ng;
        if (quantized_shapes.insert(activation_key).second) {
            result.activation_geometries.push_back({
                static_cast<int>(pool.weight.ng),
                static_cast<int>(pool.weight.gs),
                0,
                0,
            });
        }
    }
}

static MfeWeight to_gpu_mfe(const MfeCpu & cpu) {
    MfeWeight result;
    result.unified_nint_projection = true;
    result.n_experts = cpu.n_experts;
    result.out_per_expert = cpu.out_per_expert;
    result.neuron_len = cpu.neuron_len;
    result.pools.reserve(cpu.pools.size());
    for (const auto & source_pool : cpu.pools) {
        if (source_pool.dtype != "NINT") {
            throw std::runtime_error("non-NINT cohort reached the NINT-only loader");
        }
        MfePoolWeight pool;
        pool.local_experts = (int)source_pool.expert_ids.size();
        pool.weight = to_device_mfe_nint(
            source_pool.weight, pool.local_experts,
            cpu.out_per_expert, true);
        std::vector<int32_t> local((size_t)cpu.n_experts, -1);
        for (int index = 0; index < pool.local_experts; ++index) {
            const int expert = source_pool.expert_ids.at(static_cast<size_t>(index));
            local[static_cast<size_t>(expert)] = index;
        }
        pool.expert_local = mfq_tensor_backend::from_blob(
            local.data(), {(int64_t)local.size()}, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt32))
            .clone().to(pool.weight.q_packed.device()).contiguous();
        result.pools.push_back(std::move(pool));
    }
    initialize_mfe_nint_runtime(result);
    return result;
}

static std::vector<mfq::TensorParallelSlice>
plan_moe_expert_parallel_slices(
    int64_t extent,
    const std::string & name);

MfeWeight load_mfe_gpu(
    const mfq::ModelSource & mfq, const std::string & name,
    bool cacheable,
    int layer_id,
    const std::string & projection_role);

static std::vector<mfq_tensor_backend::Tensor> mfe_ffn_forward(
        const MfeWeight & gate_up, const MfeWeight & down,
        mfq_tensor_backend::Tensor x, mfq_tensor_backend::Tensor router_logits,
        int top_k, bool use_sigmoid, bool use_sqrt_softplus,
        bool normalize, bool delayed_softmax,
        MfqOptional<mfq_tensor_backend::Tensor> router_bias, double router_scale) {
    if (gate_up.n_experts != down.n_experts ||
        gate_up.out_per_expert != 2 * down.neuron_len ||
        down.out_per_expert != gate_up.neuron_len) {
        throw std::runtime_error("incompatible fused gate_up/down MFE tensors");
    }
    auto selected = moe_topk_cuda(
        router_logits.contiguous(), top_k, use_sigmoid, use_sqrt_softplus,
        normalize, delayed_softmax,
        router_bias, 1e-20, router_scale);
    MoeRoutePlan route = build_moe_route_plan(selected.at(0), gate_up.n_experts);
    down.prefetch(route);
    auto hidden = gate_up.forward_glu_output(x, route, false);
    auto down_pair = down.forward(hidden, route);
    auto output = moe_weighted_reduce_cuda(down_pair, selected.at(1));
    return {output, selected.at(0), selected.at(1)};
}

struct NvqCpu {
    int format = 0;
    int sign_mode = 0;
    int sub_bits = 0;
    int gs = 0;
    int axis = 0;
    int neuron_len = 0;
    int out = 0;
    int ng = 0;
    int nvec = 0;
    int nsign = 0;
    std::vector<int64_t> shape;
    std::vector<uint8_t> indices_packed;
    std::vector<uint8_t> aux_packed;
    std::vector<uint8_t> sub_scale_packed;
    std::vector<uint16_t> neuron_scale_h;
    std::vector<int8_t> codebook;
};

static int nvq_vector_size(int format) {
    return (format == 3 || format == 10 || format == 12 || format == 15) ? 4 : 8;
}

static int nvq_index_bits(int format) {
    return format == 1 ? 11 :
        (format == 8 ? 9 :
         (format == 7 ? 7 :
          (format == 9 ? 6 :
           (format == 12 ? 9 :
            ((format == 13 || format == 15) ? 10 :
             (format == 14 ? 12 : 8))))));
}

static bool nvq_delta_format(int format) {
    return format == 1 || format == 8;
}

static bool nvq_no_aux_format(int format) {
    return format == 7 || format == 9;
}

static void copy_packed_bits(
        const std::vector<uint8_t> & source,
        size_t source_bit,
        std::vector<uint8_t> & destination,
        size_t destination_bit,
        size_t bit_count) {
    if (bit_count == 0) return;
    if (source_bit + bit_count >
            source.size() * 8 ||
        destination_bit + bit_count >
            destination.size() * 8) {
        throw std::runtime_error(
            "packed tensor-parallel bit copy is out of bounds");
    }
    if ((source_bit & 7u) == 0 &&
        (destination_bit & 7u) == 0) {
        const size_t full_bytes =
            bit_count / 8;
        if (full_bytes != 0) {
            std::memcpy(
                destination.data() +
                    destination_bit / 8,
                source.data() +
                    source_bit / 8,
                full_bytes);
            source_bit += full_bytes * 8;
            destination_bit += full_bytes * 8;
            bit_count -= full_bytes * 8;
        }
    }
    while (bit_count >= 8) {
        const size_t source_byte =
            source_bit >> 3;
        const int source_shift =
            static_cast<int>(source_bit & 7u);
        uint16_t source_word =
            source[source_byte];
        if (source_shift != 0 &&
            source_byte + 1 < source.size()) {
            source_word |=
                static_cast<uint16_t>(
                    source[source_byte + 1])
                << 8;
        }
        const uint8_t value =
            static_cast<uint8_t>(
                source_word >> source_shift);
        const size_t destination_byte =
            destination_bit >> 3;
        const int destination_shift =
            static_cast<int>(
                destination_bit & 7u);
        destination[destination_byte] |=
            static_cast<uint8_t>(
                value << destination_shift);
        if (destination_shift != 0 &&
            destination_byte + 1 <
                destination.size()) {
            destination[destination_byte + 1] |=
                static_cast<uint8_t>(
                    value >>
                    (8 - destination_shift));
        }
        source_bit += 8;
        destination_bit += 8;
        bit_count -= 8;
    }
    for (size_t bit = 0; bit < bit_count; ++bit) {
        if ((source[source_bit >> 3] >>
             (source_bit & 7u)) & 1u) {
            destination[destination_bit >> 3] |=
                static_cast<uint8_t>(
                    1u << (destination_bit & 7u));
        }
        ++source_bit;
        ++destination_bit;
    }
}

static std::vector<uint8_t> slice_packed_rows(
        const std::vector<uint8_t> & source,
        int64_t source_rows,
        int64_t source_items_per_row,
        int64_t row_begin,
        int64_t row_end,
        int64_t item_begin,
        int64_t item_count,
        int bits) {
    if (bits == 0) return {};
    if (source_rows <= 0 || source_items_per_row <= 0 ||
        row_begin < 0 || row_begin >= row_end ||
        row_end > source_rows || item_begin < 0 ||
        item_count <= 0 ||
        item_begin + item_count > source_items_per_row) {
        throw std::runtime_error("invalid packed tensor-parallel slice");
    }
    const int64_t destination_rows = row_end - row_begin;
    const size_t destination_items =
        static_cast<size_t>(destination_rows) * item_count;
    std::vector<uint8_t> destination(
        (destination_items * static_cast<size_t>(bits) + 7) / 8,
        0);
    for (int64_t row = 0; row < destination_rows; ++row) {
        const size_t source_item =
            static_cast<size_t>(row + row_begin) *
                source_items_per_row +
            static_cast<size_t>(item_begin);
        const size_t destination_item =
            static_cast<size_t>(row) * item_count;
        copy_packed_bits(
            source,
            source_item * static_cast<size_t>(bits),
            destination,
            destination_item * static_cast<size_t>(bits),
            static_cast<size_t>(item_count) *
                static_cast<size_t>(bits));
    }
    return destination;
}

static NvqCpu slice_nvq_cpu(
        const NvqCpu & source,
        TensorParallelAxis axis,
        int64_t begin,
        int64_t end) {
    require_tp_row_major_weight(
        source.shape, source.axis, source.out,
        source.neuron_len, "NVQ");
    if (axis != TensorParallelAxis::Output &&
        axis != TensorParallelAxis::Input) {
        throw std::runtime_error("NVQ shard requires an output or input axis");
    }
    NvqCpu result = source;
    int64_t row_begin = 0;
    int64_t row_end = source.out;
    int64_t group_begin = 0;
    int64_t group_count = source.ng;
    if (axis == TensorParallelAxis::Output) {
        if (begin < 0 || begin >= end || end > source.out) {
            throw std::runtime_error("invalid NVQ output shard");
        }
        row_begin = begin;
        row_end = end;
        result.out = static_cast<int>(end - begin);
        result.shape[0] = result.out;
    } else {
        if (begin < 0 || begin >= end || end > source.ng) {
            throw std::runtime_error("invalid NVQ input shard");
        }
        group_begin = begin;
        group_count = end - begin;
        result.ng = static_cast<int>(group_count);
        const int64_t element_begin = begin * source.gs;
        const int64_t element_end =
            std::min<int64_t>(end * source.gs, source.neuron_len);
        result.neuron_len =
            static_cast<int>(element_end - element_begin);
        result.shape[1] = result.neuron_len;
        const int vector_size = nvq_vector_size(source.format);
        if (element_begin % vector_size != 0 ||
            element_begin % 8 != 0) {
            throw std::runtime_error(
                "NVQ input shard does not preserve vector boundaries");
        }
        result.nvec =
            (result.neuron_len + vector_size - 1) / vector_size;
        result.nsign = (result.neuron_len + 7) / 8;
    }

    result.neuron_scale_h.assign(
        source.neuron_scale_h.begin() +
            static_cast<ptrdiff_t>(row_begin),
        source.neuron_scale_h.begin() +
            static_cast<ptrdiff_t>(row_end));
    result.sub_scale_packed = slice_packed_rows(
        source.sub_scale_packed,
        source.out, source.ng,
        row_begin, row_end,
        group_begin, group_count,
        source.sub_bits);

    const int vector_size = nvq_vector_size(source.format);
    const int64_t element_begin = group_begin * source.gs;
    const int64_t vector_begin = element_begin / vector_size;
    const int64_t vector_count =
        axis == TensorParallelAxis::Output
        ? source.nvec
        : result.nvec;
    result.indices_packed = slice_packed_rows(
        source.indices_packed,
        source.out, source.nvec,
        row_begin, row_end,
        vector_begin, vector_count,
        nvq_index_bits(source.format));

    if (nvq_delta_format(source.format)) {
        result.aux_packed = slice_packed_rows(
            source.aux_packed,
            source.out, source.ng,
            row_begin, row_end,
            group_begin, group_count,
            1);
    } else if (nvq_no_aux_format(source.format)) {
        result.aux_packed.clear();
    } else {
        const int64_t sign_begin = element_begin / 8;
        const int64_t sign_count =
            axis == TensorParallelAxis::Output
            ? source.nsign
            : result.nsign;
        result.aux_packed = slice_packed_rows(
            source.aux_packed,
            source.out, source.nsign,
            row_begin, row_end,
            sign_begin, sign_count,
            7);
    }
    return result;
}

static std::vector<int8_t> expand_nvq_codebook(
    int format, const uint16_t * packed, size_t count) {
    const int dims = format == 3 ? 4 : 8;
    const int expected = format == 1 ? 2048 : (format == 8 ? 1024 : 256);
    if ((int)count != expected) throw std::runtime_error("NVQ codebook entry count mismatch");
    const int digit_bits = format == 3 ? 3 : 2;
    std::vector<int8_t> codebook(count * (size_t)dims);
    for (size_t row = 0; row < count; ++row) {
        uint16_t word = packed[row];
        for (int i = 0; i < dims; ++i) {
            int digit = (word >> (digit_bits * i)) & ((1 << digit_bits) - 1);
            const bool ternary = format == 1 || format == 8;
            int value = ternary ? digit - 1 : 2 * digit + 1;
            if ((ternary && (value < -1 || value > 1)) ||
                (!ternary && (value < 1 || value > (format == 2 ? 7 : 15)))) {
                throw std::runtime_error("invalid NVQ codebook digit");
            }
            codebook[row * (size_t)dims + i] = (int8_t)value;
        }
    }
    return codebook;
}

static std::vector<int8_t> expand_npq0_s_runtime_lut(
    const std::vector<uint8_t> & packed) {
    constexpr size_t kMetadataBytes = 64;
    constexpr size_t kStates = 4;
    constexpr size_t kEntries = 8;
    constexpr size_t kSubvector = 4;
    constexpr size_t kPackedBytes = kMetadataBytes + 2 * kStates * kEntries * kSubvector;
    constexpr size_t kRuntimeBytes = kMetadataBytes + kStates * 64 * 8;
    if (packed.size() != kPackedBytes) {
        throw std::runtime_error("NPQ0-S packed table size mismatch");
    }
    std::vector<int8_t> runtime(kRuntimeBytes);
    std::memcpy(runtime.data(), packed.data(), kMetadataBytes);
    const int8_t * first = reinterpret_cast<const int8_t *>(packed.data() + kMetadataBytes);
    const int8_t * second = first + kStates * kEntries * kSubvector;
    for (size_t state = 0; state < kStates; ++state) {
        for (size_t second_index = 0; second_index < kEntries; ++second_index) {
            for (size_t first_index = 0; first_index < kEntries; ++first_index) {
                const size_t index = first_index | (second_index << 3);
                int8_t * destination = runtime.data() + kMetadataBytes
                    + (state * 64 + index) * 8;
                std::memcpy(
                    destination,
                    first + (state * kEntries + first_index) * kSubvector,
                    kSubvector);
                std::memcpy(
                    destination + kSubvector,
                    second + (state * kEntries + second_index) * kSubvector,
                    kSubvector);
            }
        }
    }
    return runtime;
}

static std::vector<uint16_t> read_codebook_words(
    const std::vector<uint8_t> & blob, size_t & off, size_t count) {
    size_t nbytes = count * sizeof(uint16_t);
    if (off + nbytes > blob.size()) throw std::runtime_error("truncated NVQ custom codebook");
    std::vector<uint16_t> words(count);
    std::memcpy(words.data(), blob.data() + off, nbytes);
    off += nbytes;
    return words;
}

static std::vector<uint8_t> take_bytes(
    const std::vector<uint8_t> & blob, size_t & off, size_t count, const char * label) {
    if (off + count > blob.size()) throw std::runtime_error(std::string("truncated NVQ ") + label);
    std::vector<uint8_t> result(count);
    std::memcpy(result.data(), blob.data() + off, count);
    off += count;
    return result;
}

static void store_compact_bits_cpu(
        std::vector<uint8_t> & destination,
        size_t bit,
        int bits,
        uint32_t value) {
    const size_t byte = bit >> 3;
    const int shift = static_cast<int>(bit & 7);
    const int bytes = (shift + bits + 7) >> 3;
    if (bits <= 0 || bits > 24 || byte + bytes > destination.size() ||
            value >= (1u << bits)) {
        throw std::runtime_error("invalid compact bitstream write");
    }
    const uint32_t shifted = value << shift;
    for (int index = 0; index < bytes; ++index) {
        destination[byte + index] |= static_cast<uint8_t>(
            shifted >> (index * 8));
    }
}

static NvqCpu unpack_nvq(const std::vector<uint8_t> & blob, const std::string & dtype) {
    if (blob.size() < 20) throw std::runtime_error("truncated NVQ header");
    NvqCpu t;
    const bool nvq_family = dtype == "NVQ";
    const bool npq_family = dtype == "NPQ";
    const bool nvq_magic =
        (std::memcmp(blob.data(), "NVQ1", 4) == 0) ||
        (std::memcmp(blob.data(), "NIQ1", 4) == 0);
    const bool nvq1_l_magic = std::memcmp(blob.data(), "NQ1L", 4) == 0;
    const bool nvq1_s_magic = std::memcmp(blob.data(), "NQ1S", 4) == 0;
    const bool npq0_l_magic = std::memcmp(blob.data(), "NPQL", 4) == 0;
    const bool npq0_s_magic = std::memcmp(blob.data(), "NPQS", 4) == 0;
    if ((nvq_family && !(nvq_magic || nvq1_l_magic || nvq1_s_magic)) ||
        (npq_family && !(npq0_l_magic || npq0_s_magic)) ||
        (!nvq_family && !npq_family)) {
        throw std::runtime_error("compact VQ family/payload mismatch: " + dtype);
    }
    size_t off = 4;
    const uint8_t profile = blob[off++];
    if (nvq1_l_magic) {
        t.format = 1;
    } else if (nvq1_s_magic) {
        t.format = 8;
    } else if (npq0_l_magic) {
        t.format = 7;
    } else if (npq0_s_magic) {
        t.format = 9;
    } else {
        constexpr uint8_t kIndexParity = 0x80;
        constexpr uint8_t kCustom = 0x40;
        constexpr uint8_t kJsc = 0x20;
        const bool jsc = (profile & kJsc) != 0;
        const int codebook_id = profile & ~(kIndexParity | kCustom | kJsc);
        if (jsc) {
            t.format = codebook_id == 1 ? 5 :
                (codebook_id == 2 ? 10 :
                 (codebook_id == 3 ? 12 :
                  (codebook_id == 4 ? 13 :
                   (codebook_id == 5 ? 14 :
                    (codebook_id == 6 ? 15 : 0)))));
        } else {
            t.format = codebook_id == 1 ? 2 :
                (codebook_id == 2 ? 3 : 0);
        }
        if (t.format == 0) {
            throw std::runtime_error("unsupported NVQ payload profile");
        }
    }
    t.sub_bits = blob[off++];
    t.gs = (int)read_u16_from(blob, off);
    t.axis = read_i32_from(blob, off);
    t.neuron_len = read_i32_from(blob, off);
    uint32_t ndim = read_u32_from(blob, off);
    if (ndim == 0 || ndim > 8) throw std::runtime_error("invalid NVQ ndim");
    t.shape.resize(ndim);
    for (uint32_t i = 0; i < ndim; ++i) t.shape[i] = read_i64_from(blob, off);
    t.out = (int)read_u32_from(blob, off);
    if (t.gs != 24 || t.sub_bits < 1 || t.sub_bits > 8) {
        throw std::runtime_error("C++ NVQ runtime requires gs24 and sub_bits in [1,8]");
    }
    if (t.axis != 0 || t.shape.size() != 2 || t.shape[0] != t.out || t.shape[1] != t.neuron_len) {
        throw std::runtime_error("C++ NVQ runtime requires row-major rank-2 axis=0 weights");
    }

    bool custom = false;
    bool group64_storage = false;
    if (t.format == 7) {
        if (profile != 1 || t.sub_bits != 3) {
            throw std::runtime_error("unsupported NPQ0-L profile");
        }
    } else if (t.format == 1) {
        if (profile != 1 && profile != 2) throw std::runtime_error("unsupported NVQ1-L profile");
        custom = profile == 2;
    } else if (t.format == 8) {
        if (profile != 1 || t.sub_bits != 4) {
            throw std::runtime_error("unsupported NVQ1-S profile");
        }
        custom = true;
    } else if (t.format == 9) {
        if (profile != 2 || t.sub_bits != 2) {
            throw std::runtime_error("unsupported NPQ0-S profile");
        }
    } else {
        constexpr uint8_t kIndexParity = 0x80;
        constexpr uint8_t kCustom = 0x40;
        constexpr uint8_t kJsc = 0x20;
        const bool jsc = (profile & kJsc) != 0;
        int codebook_id = profile & ~(kIndexParity | kCustom | kJsc);
        int expected_id =
            t.format == 13 ? 4 :
            (t.format == 14 ? 5 :
             (t.format == 15 ? 6 :
              ((t.format == 2 || t.format == 5)
               ? 1
               : (t.format == 12 ? 3 : 2))));
        if (codebook_id != expected_id) throw std::runtime_error("NVQ payload profile/codebook mismatch");
        t.sign_mode = (profile & kIndexParity) ? 1 : 0;
        if (t.sign_mode && t.format != 2) throw std::runtime_error("NVQ index parity requires NVQ2");
        custom = (profile & kCustom) != 0;
        if (t.format == 5 || t.format == 10 || t.format == 12 ||
            t.format == 13 || t.format == 14 || t.format == 15) {
            if (!jsc || custom || t.sign_mode || t.sub_bits != 4) {
                throw std::runtime_error("invalid NVQ-JSC profile flags");
            }
        } else if (jsc) {
            throw std::runtime_error("NVQ-JSC flag requires a JSC payload profile");
        }
    }

    const size_t codebook_count = t.format == 1 ? 2048 : (t.format == 8 ? 1024 : 256);
    if (t.format == 7) {
        constexpr size_t kMetadataBytes = 64;
        constexpr size_t kTableBytes = 832;
        if (off + kTableBytes > blob.size()) {
            throw std::runtime_error("truncated NPQ0-L product tables");
        }
        const uint8_t * header = blob.data() + off;
        const uint8_t expected[6] = {1, 8, 3, 4, 24, 8};
        for (int i = 0; i < 6; ++i) {
            if (header[i] != expected[i]) {
                throw std::runtime_error("unsupported NPQ0-L table profile");
            }
        }
        if (header[6] != 0 || header[7] != 0) {
            throw std::runtime_error("invalid NPQ0-L reserved table bytes");
        }
        for (int state = 0; state < 8; ++state) {
            uint16_t alpha_h;
            std::memcpy(&alpha_h, header + 8 + state * 2, sizeof(alpha_h));
            if ((alpha_h & 0x8000u) || (alpha_h & 0x7c00u) == 0x7c00u) {
                throw std::runtime_error("NPQ0-L scale LUT must be finite and non-negative");
            }
        }
        for (size_t i = 24; i < kMetadataBytes; ++i) {
            if (header[i] != 0) {
                throw std::runtime_error("invalid NPQ0-L reserved table bytes");
            }
        }
        auto metadata = take_bytes(blob, off, kTableBytes, "NPQ0-L product tables");
        t.codebook.resize(metadata.size());
        std::memcpy(t.codebook.data(), metadata.data(), metadata.size());
    } else if (t.format == 9) {
        constexpr size_t kMetadataBytes = 64;
        constexpr size_t kTableBytes = 320;
        if (off + kTableBytes > blob.size()) {
            throw std::runtime_error("truncated NPQ0-S product tables");
        }
        const uint8_t * header = blob.data() + off;
        const uint8_t expected[6] = {2, 4, 3, 3, 24, 8};
        for (int i = 0; i < 6; ++i) {
            if (header[i] != expected[i]) {
                throw std::runtime_error("unsupported NPQ0-S table profile");
            }
        }
        if (header[6] != 0 || header[7] != 0) {
            throw std::runtime_error("invalid NPQ0-S reserved table bytes");
        }
        for (int state = 0; state < 4; ++state) {
            uint16_t alpha_h;
            std::memcpy(&alpha_h, header + 8 + state * 2, sizeof(alpha_h));
            if ((alpha_h & 0x8000u) || (alpha_h & 0x7c00u) == 0x7c00u) {
                throw std::runtime_error("NPQ0-S scale LUT must be finite and non-negative");
            }
        }
        for (size_t i = 16; i < kMetadataBytes; ++i) {
            if (header[i] != 0) {
                throw std::runtime_error("invalid NPQ0-S reserved table bytes");
            }
        }
        auto metadata = take_bytes(blob, off, kTableBytes, "NPQ0-S product tables");
        t.codebook = expand_npq0_s_runtime_lut(metadata);
    } else if (t.format == 5 || t.format == 10 || t.format == 12 ||
               t.format == 13 || t.format == 14 || t.format == 15) {
        constexpr size_t kHeaderBytes = 64;
        const int vector_size = nvq_vector_size(t.format);
        const size_t entries =
            t.format == 12 ? 512 :
            ((t.format == 13 || t.format == 15) ? 1024 :
             (t.format == 14 ? 4096 : 256));
        const size_t kBankBytes = entries * (size_t)vector_size;
        if (off + kHeaderBytes > blob.size()) {
            throw std::runtime_error("truncated NVQ2J metadata header");
        }
        const uint8_t * header = blob.data() + off;
        const int banks = header[1];
        const int metadata_version = header[0];
        if ((metadata_version != 1 && metadata_version != 2) ||
                (banks != 1 && banks != 2 && banks != 4) || header[2] != 16) {
            throw std::runtime_error("unsupported NVQ2J metadata dimensions");
        }
        if (metadata_version == 1) {
            if (header[52] != 0) {
                throw std::runtime_error("invalid NVQ-JSC v1 storage layout");
            }
        } else {
            if (t.format != 14 || header[52] != 1) {
                throw std::runtime_error(
                    "NVQ-JSC group64 storage requires NVQ2J-XL");
            }
            group64_storage = true;
        }
        if (header[3] != 0 && header[3] != 1) {
            throw std::runtime_error("invalid NVQ-JSC state mode");
        }
        for (int state = 0; state < 16; ++state) {
            uint16_t alpha_h;
            std::memcpy(&alpha_h, header + 4 + state * 2, sizeof(alpha_h));
            if ((alpha_h & 0x8000u) || (alpha_h & 0x7c00u) == 0x7c00u) {
                throw std::runtime_error("NVQ-JSC scale LUT must be finite and non-negative");
            }
            if (header[36 + state] >= banks) {
                throw std::runtime_error("NVQ-JSC state references a missing bank");
            }
            if (header[3] == 1) {
                const uint8_t expected_bank = (uint8_t)(state % banks);
                const int rank = state / banks;
                const float expected_scale = banks == 1
                    ? (float)state
                    : (vector_size == 4
                        ? (float)(rank + 1)
                        : 15.0f * (float)(rank + 1) / (float)(16 / banks));
                const mfq_half expected_half(expected_scale);
                uint16_t expected_alpha_h;
                std::memcpy(&expected_alpha_h, &expected_half, sizeof(expected_alpha_h));
                if (header[36 + state] != expected_bank || alpha_h != expected_alpha_h) {
                    throw std::runtime_error("invalid analytic NVQ-JSC state tables");
                }
            }
        }
        const size_t reserved_begin = metadata_version == 1 ? 52 : 53;
        for (size_t i = reserved_begin; i < kHeaderBytes; ++i) {
            if (header[i] != 0) throw std::runtime_error("invalid NVQ-JSC reserved metadata bytes");
        }
        const size_t metadata_bytes = kHeaderBytes + (size_t)banks * kBankBytes;
        auto metadata = take_bytes(blob, off, metadata_bytes, "JSC metadata");
        for (int bank = 0; bank < banks; ++bank) {
            const size_t base = kHeaderBytes + (size_t)bank * kBankBytes;
            for (size_t entry = 0; entry < entries; ++entry) {
                bool nonzero = false;
                for (int dim = 0; dim < vector_size; ++dim) {
                    const uint8_t value =
                        metadata[base + (size_t)entry * vector_size + dim];
                    if (value > 127) throw std::runtime_error("NVQ-JSC codebook value exceeds int8");
                    nonzero = nonzero || value != 0;
                }
                if (!nonzero) throw std::runtime_error("NVQ-JSC codeword must not be all zero");
            }
        }
        t.codebook.resize(metadata.size());
        std::memcpy(t.codebook.data(), metadata.data(), metadata.size());
    } else if (custom) {
        auto words = read_codebook_words(blob, off, codebook_count);
        t.codebook = expand_nvq_codebook(t.format, words.data(), words.size());
    } else if (t.format == 1) {
        t.codebook = expand_nvq_codebook(
            1, mfq::nvq_codebooks::kNvq1LCodebookPacked,
            sizeof(mfq::nvq_codebooks::kNvq1LCodebookPacked) / sizeof(uint16_t));
    } else if (t.format == 2) {
        t.codebook = expand_nvq_codebook(
            2, mfq::nvq_codebooks::kNvq2CodebookPacked,
            sizeof(mfq::nvq_codebooks::kNvq2CodebookPacked) / sizeof(uint16_t));
    } else {
        t.codebook = expand_nvq_codebook(
            3, mfq::nvq_codebooks::kNvq3CodebookPacked,
            sizeof(mfq::nvq_codebooks::kNvq3CodebookPacked) / sizeof(uint16_t));
    }

    t.ng = (t.neuron_len + t.gs - 1) / t.gs;
    const int vector_size = nvq_vector_size(t.format);
    t.nvec = (t.neuron_len + vector_size - 1) / vector_size;
    t.nsign = (t.neuron_len + 7) / 8;
    size_t anchor_bytes = (size_t)t.out * 2;
    if (off + anchor_bytes > blob.size()) throw std::runtime_error("truncated NVQ neuron anchors");
    t.neuron_scale_h.resize(t.out);
    std::memcpy(t.neuron_scale_h.data(), blob.data() + off, anchor_bytes);
    for (uint16_t anchor : t.neuron_scale_h) {
        if ((anchor & 0x7c00u) == 0x7c00u || (anchor & 0x8000u) != 0u) {
            throw std::runtime_error(
                "NVQ neuron anchors must be finite and non-negative");
        }
    }
    off += anchor_bytes;
    size_t sub_bytes = ((size_t)t.out * t.ng * t.sub_bits + 7) / 8;
    const size_t index_bits = (size_t)nvq_index_bits(t.format);
    size_t index_bytes = ((size_t)t.out * t.nvec * index_bits + 7) / 8;
    const bool delta_format = t.format == 1 || t.format == 8;
    const bool no_aux_format = t.format == 7 || t.format == 9;
    size_t aux_count = delta_format ? (size_t)t.out * t.ng :
        (no_aux_format ? 0 : (size_t)t.out * t.nsign);
    size_t aux_bits = delta_format ? 1 : (no_aux_format ? 0 : 7);
    size_t aux_bytes = (aux_count * aux_bits + 7) / 8;
    if (group64_storage) {
        const size_t record_count = static_cast<size_t>(t.out) * t.ng;
        const auto records = take_bytes(
            blob, off, record_count * 8, "group64 stream");
        t.sub_scale_packed.assign(sub_bytes, 0);
        t.indices_packed.assign(index_bytes, 0);
        t.aux_packed.assign(aux_bytes, 0);
        for (size_t linear = 0; linear < record_count; ++linear) {
            uint64_t packed = 0;
            std::memcpy(&packed, records.data() + linear * 8, sizeof(packed));
            const uint32_t state = static_cast<uint32_t>(packed >> 60);
            store_compact_bits_cpu(
                t.sub_scale_packed, linear * t.sub_bits, t.sub_bits, state);
            const size_t row = linear / static_cast<size_t>(t.ng);
            const size_t group = linear - row * static_cast<size_t>(t.ng);
            for (int local = 0; local < 3; ++local) {
                const uint32_t segment = static_cast<uint32_t>(
                    (packed >> (local * 20)) & 0xfffffu);
                const uint32_t index = segment & 0xfffu;
                const uint32_t mask8 = segment >> 12;
                uint32_t parity = mask8 & 0x7fu;
                parity ^= parity >> 4;
                parity ^= parity >> 2;
                parity ^= parity >> 1;
                if ((mask8 >> 7) != (parity & 1u)) {
                    throw std::runtime_error(
                        "invalid NVQ-JSC group64 parity bit");
                }
                const size_t local_vector = group * 3 + local;
                if (local_vector >= static_cast<size_t>(t.nvec)) {
                    if (segment != 0) {
                        throw std::runtime_error(
                            "NVQ-JSC group64 padding must be zero");
                    }
                    continue;
                }
                const size_t vector = row * static_cast<size_t>(t.nvec) +
                    local_vector;
                store_compact_bits_cpu(
                    t.indices_packed, vector * index_bits,
                    static_cast<int>(index_bits), index);
                store_compact_bits_cpu(
                    t.aux_packed, vector * 7, 7, mask8 & 0x7fu);
            }
        }
    } else {
        t.sub_scale_packed = take_bytes(blob, off, sub_bytes, "sub-scale stream");
        t.indices_packed = take_bytes(blob, off, index_bytes, "index stream");
        t.aux_packed = take_bytes(blob, off, aux_bytes, "aux stream");
    }
    if (off != blob.size()) throw std::runtime_error("invalid NVQ blob tail");
    return t;
}

static mfq_tensor_backend::Tensor cpu_i8_tensor(const std::vector<int8_t> & v, std::initializer_list<int64_t> shape) {
    return mfq_tensor_backend::from_blob((void *)v.data(), shape, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt8)).clone();
}



constexpr int64_t kNvq2JscXlGroupExecKernelFormat = 16;
constexpr int64_t kNvq3JscLGroupExecKernelFormat = 17;



static std::vector<uint8_t> repack_nvq2_exec_metadata(const NvqCpu & c) {
    if ((c.format != 2 && c.format != 5) || c.nvec != c.nsign) {
        throw std::runtime_error("NVQ2 execution metadata requires one sign record per vector");
    }
    const size_t count = (size_t)c.out * c.nvec;
    if (c.indices_packed.size() != count) {
        throw std::runtime_error("NVQ2 index stream length mismatch during execution repack");
    }
    std::vector<uint8_t> metadata(count * 2);
    for (size_t linear = 0; linear < count; ++linear) {
        const size_t bit = linear * 7;
        const size_t byte = bit >> 3;
        const int shift = (int)(bit & 7);
        uint16_t word = c.aux_packed[byte];
        if (byte + 1 < c.aux_packed.size()) {
            word |= (uint16_t)c.aux_packed[byte + 1] << 8;
        }
        const uint8_t index = c.indices_packed[linear];
        const uint8_t mask7 = (uint8_t)((word >> shift) & 0x7f);
        uint8_t parity = mask7;
        parity ^= parity >> 4;
        parity ^= parity >> 2;
        parity ^= parity >> 1;
        const uint8_t last = (parity & 1u) ^
            (c.sign_mode ? ((index >> 7) & 1u) : 0u);
        metadata[2 * linear] = c.indices_packed[linear];
        metadata[2 * linear + 1] = mask7 | (last << 7);
    }
    return metadata;
}

static uint32_t load_compact_bits_cpu(
        const std::vector<uint8_t> & data, size_t bit, int bits) {
    const size_t byte = bit >> 3;
    const int shift = static_cast<int>(bit & 7);
    uint32_t word = byte < data.size() ? data[byte] : 0;
    if (byte + 1 < data.size()) word |= static_cast<uint32_t>(data[byte + 1]) << 8;
    if (byte + 2 < data.size()) word |= static_cast<uint32_t>(data[byte + 2]) << 16;
    return (word >> shift) & ((1u << bits) - 1u);
}

static void put_compact_bits96(
        uint32_t (&words)[3], int bit, int bits, uint32_t value) {
    const int word = bit >> 5;
    const int shift = bit & 31;
    words[word] |= value << shift;
    if (shift + bits > 32) words[word + 1] |= value >> (32 - shift);
}

static uint32_t compact_parity7(uint32_t value) {
    value &= 0x7fu;
    value ^= value >> 4;
    value ^= value >> 2;
    value ^= value >> 1;
    return value & 1u;
}

// NVQ2J-XL needs exactly 64 bits per gs24 group (three 12-bit indices, three
// 8-bit sign masks, and one 4-bit state). NVQ3J-L stores three indices in
// a 96-bit record per gs24 group (six 10-bit indices, three 8-bit sign masks,
// and one 4-bit state).
// This replaces the two independently bit-packed streams on CUDA; it does not
// keep a second execution copy of the model weights.
static std::vector<uint8_t> repack_extended_jsc_group_metadata(
        const NvqCpu & c,
        const std::array<uint8_t, 16> * state_remap = nullptr,
        const std::array<std::array<uint16_t, 4096>, 4> * index_remap = nullptr) {
    if (c.format != 14 && c.format != 15) {
        throw std::runtime_error("extended JSC group metadata requires NVQ2J-XL or NVQ3J-L");
    }
    const int index_bits = c.format == 14 ? 12 : 10;
    const int vectors_per_group = c.format == 14 ? 3 : 6;
    const size_t record_bytes = c.format == 14 ? 8 : 12;
    const size_t group_count = static_cast<size_t>(c.out) * c.ng;
    std::vector<uint8_t> metadata(group_count * record_bytes, 0);
    mfq_parallel_for(0, static_cast<int64_t>(group_count), 4096, [&](int64_t begin, int64_t end) {
        for (int64_t linear = begin; linear < end; ++linear) {
            const int64_t row = linear / c.ng;
            const int group = static_cast<int>(linear - row * c.ng);
            const size_t vector_base = static_cast<size_t>(row) * c.nvec +
                static_cast<size_t>(group) * vectors_per_group;
            const size_t sign_base = static_cast<size_t>(row) * c.nsign +
                static_cast<size_t>(group) * 3;
            const uint32_t source_state = load_compact_bits_cpu(
                c.sub_scale_packed, static_cast<size_t>(linear) * c.sub_bits,
                c.sub_bits);
            const uint32_t state = state_remap == nullptr
                ? source_state
                : (*state_remap)[source_state];
            uint8_t * destination = metadata.data() +
                static_cast<size_t>(linear) * record_bytes;
            if (c.format == 14) {
                uint64_t packed = 0;
                for (int local = 0; local < 3; ++local) {
                    const size_t vector = vector_base + local;
                    const uint32_t source_index = vector < static_cast<size_t>((row + 1) * c.nvec)
                        ? load_compact_bits_cpu(c.indices_packed, vector * index_bits, index_bits)
                        : 0;
                    const uint32_t source_bank = static_cast<uint8_t>(
                        c.codebook[36 + source_state]);
                    const uint32_t index = index_remap == nullptr
                        ? source_index
                        : (*index_remap)[source_bank][source_index];
                    const size_t sign = sign_base + local;
                    const uint32_t mask7 = sign < static_cast<size_t>((row + 1) * c.nsign)
                        ? load_compact_bits_cpu(c.aux_packed, sign * 7, 7)
                        : 0;
                    const uint32_t mask8 = mask7 | (compact_parity7(mask7) << 7);
                    const uint32_t segment = index | (mask8 << index_bits);
                    packed |= static_cast<uint64_t>(segment) << (local * 20);
                }
                packed |= static_cast<uint64_t>(state) << 60;
                std::memcpy(destination, &packed, sizeof(packed));
            } else {
                uint32_t words[3] = {0, 0, 0};
                for (int local = 0; local < 6; ++local) {
                    const size_t vector = vector_base + local;
                    const uint32_t index = vector < static_cast<size_t>((row + 1) * c.nvec)
                        ? load_compact_bits_cpu(c.indices_packed, vector * index_bits, index_bits)
                        : 0;
                    put_compact_bits96(words, local * index_bits, index_bits, index);
                }
                for (int local = 0; local < 3; ++local) {
                    const size_t sign = sign_base + local;
                    const uint32_t mask7 = sign < static_cast<size_t>((row + 1) * c.nsign)
                        ? load_compact_bits_cpu(c.aux_packed, sign * 7, 7)
                        : 0;
                    const uint32_t mask8 = mask7 | (compact_parity7(mask7) << 7);
                    put_compact_bits96(words, 60 + local * 8, 8, mask8);
                }
                put_compact_bits96(words, 84, 4, state);
                std::memcpy(destination, words, sizeof(words));
            }
        }
    });
    return metadata;
}

static bool nvq2_exec_layout_enabled() {
    const char * disable = std::getenv("MFQ_DISABLE_NVQ2_EXEC");
    if (disable == nullptr) disable = std::getenv("MFQ_DISABLE_NIQ2_EXEC");
    return disable == nullptr || disable[0] != '1';
}

static bool extended_jsc_group_layout_enabled() {
    const char * enable = std::getenv("MFQ_NVQ_EXTENDED_GROUP_EXEC");
    return enable != nullptr && enable[0] == '1';
}

static std::vector<int8_t> reorder_extended_e8_codebook_for_stage3(
        const NvqCpu & c,
        std::array<uint8_t, 16> & state_remap,
        std::array<std::array<uint16_t, 4096>, 4> & index_remap) {
    constexpr size_t metadata_bytes = 64;
    constexpr size_t bank_bytes = 4096 * 8;
    constexpr int bank_count = 4;
    if (c.format != 14 ||
        c.codebook.size() != metadata_bytes + bank_count * bank_bytes ||
        static_cast<uint8_t>(c.codebook[1]) != bank_count) {
        throw std::runtime_error("extended E8 runtime layout requires NVQ2J-XL");
    }

    uint64_t counts[bank_count] = {0, 0, 0, 0};
    std::vector<uint64_t> index_counts(bank_count * 4096, 0);
    const size_t groups = static_cast<size_t>(c.out) * c.ng;
    for (size_t linear = 0; linear < groups; ++linear) {
        const uint32_t state = load_compact_bits_cpu(
            c.sub_scale_packed, linear * c.sub_bits, c.sub_bits);
        const uint32_t bank = static_cast<uint8_t>(c.codebook[36 + state]);
        if (bank >= bank_count) {
            throw std::runtime_error("NVQ2J-XL state references an invalid bank");
        }
        ++counts[bank];
        const size_t row = linear / c.ng;
        const size_t group = linear - row * c.ng;
        const size_t vector_base = row * c.nvec + group * 3;
        const size_t vector_end = (row + 1) * c.nvec;
        for (int local = 0; local < 3; ++local) {
            const size_t vector = vector_base + local;
            if (vector >= vector_end) break;
            const uint32_t index = load_compact_bits_cpu(
                c.indices_packed, vector * 12, 12);
            ++index_counts[bank * 4096 + index];
        }
    }
    std::array<int, bank_count> order = {0, 1, 2, 3};
    std::stable_sort(order.begin(), order.end(), [&](int left, int right) {
        return counts[left] > counts[right];
    });
    std::vector<int8_t> reordered = c.codebook;
    int runtime_bank[bank_count] = {0, 0, 0, 0};
    for (int destination = 0; destination < bank_count; ++destination) {
        const int source = order[destination];
        runtime_bank[source] = destination;
        std::array<int, 4096> index_order;
        std::iota(index_order.begin(), index_order.end(), 0);
        std::stable_sort(index_order.begin(), index_order.end(),
            [&](int left, int right) {
                return index_counts[source * 4096 + left] >
                    index_counts[source * 4096 + right];
            });
        for (int destination_index = 0;
             destination_index < 4096;
             ++destination_index) {
            const int source_index = index_order[destination_index];
            index_remap[source][source_index] =
                static_cast<uint16_t>(destination_index);
            std::memcpy(
                reordered.data() + metadata_bytes + destination * bank_bytes +
                    destination_index * 8,
                c.codebook.data() + metadata_bytes + source * bank_bytes +
                    source_index * 8,
                8);
        }
    }
    int bank_state_count[bank_count] = {0, 0, 0, 0};
    for (int source_state = 0; source_state < 16; ++source_state) {
        const uint8_t source_bank = static_cast<uint8_t>(
            c.codebook[36 + source_state]);
        const int rank_in_bank = bank_state_count[source_bank]++;
        if (rank_in_bank >= 4) {
            throw std::runtime_error(
                "NVQ2J-XL runtime bank has more than four scale states");
        }
        const int destination_bank = runtime_bank[source_bank];
        const int destination_state = rank_in_bank * bank_count + destination_bank;
        state_remap[source_state] = static_cast<uint8_t>(destination_state);
        std::memcpy(
            reordered.data() + 4 + destination_state * sizeof(uint16_t),
            c.codebook.data() + 4 + source_state * sizeof(uint16_t),
            sizeof(uint16_t));
        reordered[36 + destination_state] = static_cast<int8_t>(destination_bank);
    }
    for (int bank = 0; bank < bank_count; ++bank) {
        if (bank_state_count[bank] != 4) {
            throw std::runtime_error(
                "NVQ2J-XL runtime bank must have four scale states");
        }
    }
    return reordered;
}

static std::vector<uint8_t> remap_extended_e8_sub_scale_states(
        const NvqCpu & c,
        const std::array<uint8_t, 16> & state_remap) {
    if (c.format != 14 || c.sub_bits != 4) {
        throw std::runtime_error(
            "extended E8 scale-state remap requires 4-bit NVQ2J-XL states");
    }
    const size_t state_count = static_cast<size_t>(c.out) * c.ng;
    const size_t expected_bytes = (state_count + 1) / 2;
    if (c.sub_scale_packed.size() != expected_bytes) {
        throw std::runtime_error("NVQ2J-XL scale-state stream length mismatch");
    }
    std::vector<uint8_t> remapped(expected_bytes, 0);
    mfq_parallel_for(0, static_cast<int64_t>(expected_bytes), 4096,
        [&](int64_t begin, int64_t end) {
            for (int64_t byte = begin; byte < end; ++byte) {
                const uint8_t source = c.sub_scale_packed[byte];
                const size_t low_state = static_cast<size_t>(byte) * 2;
                const uint8_t low = state_remap[source & 0x0fu];
                const uint8_t high = low_state + 1 < state_count
                    ? state_remap[source >> 4]
                    : 0;
                remapped[byte] = low | static_cast<uint8_t>(high << 4);
            }
        });
    return remapped;
}

static NvqWeight to_device_nvq(const NvqCpu & c, bool cuda) {
    NvqWeight w;
    std::array<uint8_t, 16> e8_state_remap = {};
    auto e8_index_remap =
        std::make_unique<std::array<std::array<uint16_t, 4096>, 4>>();
    std::vector<int8_t> e8_runtime_codebook;
    std::vector<uint8_t> e8_runtime_sub_scale;
    const bool use_extended_e8 =
        cuda && c.format == 14 && extended_jsc_group_layout_enabled();
    if (use_extended_e8) {
        e8_runtime_codebook = reorder_extended_e8_codebook_for_stage3(
            c, e8_state_remap, *e8_index_remap);
        e8_runtime_sub_scale = remap_extended_e8_sub_scale_states(
            c, e8_state_remap);
    }
    w.format = c.format;
    w.kernel_format = c.format;
    w.sign_mode = c.sign_mode;
    w.sub_bits = c.sub_bits;
    w.gs = c.gs;
    w.out = c.out;
    w.ng = c.ng;
    w.neuron_len = c.neuron_len;
    w.shape = c.shape;
    if (cuda && (c.format == 14 || c.format == 15) &&
        extended_jsc_group_layout_enabled()) {
        auto metadata = repack_extended_jsc_group_metadata(
            c,
            use_extended_e8 ? &e8_state_remap : nullptr,
            use_extended_e8 ? e8_index_remap.get() : nullptr);
        w.indices_packed = cpu_u8_tensor(
            metadata, {(int64_t)metadata.size()});
        w.aux_packed = mfq_tensor_backend::empty(
            {0}, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kUInt8));
        w.kernel_format = c.format == 14
            ? kNvq2JscXlGroupExecKernelFormat
            : kNvq3JscLGroupExecKernelFormat;
    } else if (cuda && (c.format == 2 || c.format == 5) &&
               nvq2_exec_layout_enabled()) {
        auto metadata = repack_nvq2_exec_metadata(c);
        w.indices_packed = cpu_u8_tensor(
            metadata, {(int64_t)metadata.size()});
        w.aux_packed = mfq_tensor_backend::empty(
            {0}, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kUInt8));
        w.kernel_format = c.format == 5 ? 6 : 4;
    } else if (
        c.format == 10 && c.codebook.size() >= 64 &&
        c.codebook[3] == 1 && c.codebook[1] == 2) {
        w.indices_packed = cpu_u8_tensor(
            c.indices_packed, {(int64_t)c.indices_packed.size()});
        w.aux_packed = cpu_u8_tensor(
            c.aux_packed, {(int64_t)c.aux_packed.size()});
        w.kernel_format = 11;
    } else {
        w.indices_packed = cpu_u8_tensor(
            c.indices_packed, {(int64_t)c.indices_packed.size()});
        w.aux_packed = cpu_u8_tensor(
            c.aux_packed, {(int64_t)c.aux_packed.size()});
    }
    w.sub_scale_packed = use_extended_e8
        ? cpu_u8_tensor(
            e8_runtime_sub_scale, {(int64_t)e8_runtime_sub_scale.size()})
        : cpu_u8_tensor(
            c.sub_scale_packed, {(int64_t)c.sub_scale_packed.size()});
    w.neuron_scale = cpu_f16_to_f32_tensor(c.neuron_scale_h, c.out);
    if (
        c.format == 5 || c.format == 7 || c.format == 9 ||
        c.format == 10 || c.format == 12 ||
        c.format == 13 || c.format == 14 || c.format == 15) {
        if (use_extended_e8) {
            w.codebook = cpu_i8_tensor(
                e8_runtime_codebook, {(int64_t)e8_runtime_codebook.size()});
        } else {
            w.codebook = cpu_i8_tensor(
                c.codebook, {(int64_t)c.codebook.size()});
        }
    } else {
        const int dims = nvq_vector_size(c.format);
        const int entries = c.format == 1 ? 2048 : (c.format == 8 ? 1024 : 256);
        w.codebook = cpu_i8_tensor(c.codebook, {entries, dims});
    }
    if (cuda) {
        const auto target = mfq_tensor_backend::Device(
            mfq_tensor_backend::kCUDA, mfq_current_cuda_device());
        w.indices_packed = w.indices_packed.to(target).contiguous();
        w.aux_packed = w.aux_packed.to(target).contiguous();
        w.sub_scale_packed =
            w.sub_scale_packed.to(target).contiguous();
        w.neuron_scale = w.neuron_scale.to(target).contiguous();
        w.codebook = w.codebook.to(target).contiguous();
        (void)w.workspace(1);
    }
    return w;
}

static NvqWeight to_gpu_nvq(const NvqCpu & c) {
    return to_device_nvq(c, true);
}

static NvqWeight to_cuda_device_nvq(
        const NvqCpu & c, int device) {
    MfqCudaGuard guard(device);
    return to_device_nvq(c, true);
}

static NvqWeight to_cpu_nvq(const NvqCpu & c) {
    return to_device_nvq(c, false);
}

struct NepqCpu {
    int profile = -1;
    int format = 0;
    int state_bits = 0;
    int index_bits = 0;
    int aux_bits = 0;
    int table_bytes = 0;
    int runtime_table_bytes = 0;
    int n_experts = 0;
    int out_per_expert = 0;
    int neuron_len = 0;
    int bank_count = 0;
    int rotation_block = 0;
    uint64_t rotation_seed = 0;
    int ng = 0;
    int nvec = 0;
    int nsuper = 0;
    std::vector<uint8_t> indices_packed;
    std::vector<uint8_t> aux_packed;
    std::vector<uint8_t> state_packed;
    std::vector<uint16_t> neuron_scale_h;
    std::vector<int8_t> table_pool;
    std::vector<int8_t> grouped_table_pool;
    std::vector<uint8_t> bank_ids;
    std::vector<int8_t> rotation_signs;
    bool residual = false;
    bool residual_second = false;
    int residual_position_bits = 0;
    int residual_record_bits = 0;
    int residual_block_vectors = 0;
    int residual_blocks_per_row = 0;
    std::vector<uint16_t> residual_codebook_h;
    std::vector<int16_t> residual_first;
    std::vector<int16_t> residual_second_dense;
};

static size_t packed_nbytes(size_t count, int bits) {
    return bits == 0 ? 0 : (count * (size_t)bits + 7) / 8;
}

static std::vector<uint16_t> unpack_nepq_u16_bits(
        const std::vector<uint8_t> & blob,
        size_t & off,
        size_t count,
        int bits,
        const char * label) {
    const size_t nbytes = packed_nbytes(count, bits);
    if (off + nbytes > blob.size()) {
        throw std::runtime_error(std::string("truncated ") + label);
    }
    std::vector<uint16_t> result(count, 0);
    for (size_t index = 0; index < count; ++index) {
        const size_t first_bit = index * static_cast<size_t>(bits);
        uint16_t value = 0;
        for (int bit = 0; bit < bits; ++bit) {
            const size_t source_bit = first_bit + static_cast<size_t>(bit);
            value |= static_cast<uint16_t>(
                ((blob[off + source_bit / 8] >> (source_bit & 7)) & 1u)
                << bit);
        }
        result[index] = value;
    }
    off += nbytes;
    return result;
}

static void configure_nepq_profile(NepqCpu & value, int profile) {
    if (profile == 0) {
        value.profile = 0;
        value.format = 9;
        value.state_bits = 2;
        value.index_bits = 6;
        value.aux_bits = 0;
        value.table_bytes = 320;
        value.runtime_table_bytes = 320;
    } else if (profile == 1) {
        value.profile = 1;
        value.format = 7;
        value.state_bits = 3;
        value.index_bits = 7;
        value.aux_bits = 0;
        value.table_bytes = 832;
        value.runtime_table_bytes = 832;
    } else if (profile == 2) {
        value.profile = 2;
        value.format = 8;
        value.state_bits = 4;
        value.index_bits = 9;
        value.aux_bits = 1;
        value.table_bytes = 2048;
        value.runtime_table_bytes = 1024 * 8;
    } else if (profile == 3) {
        value.profile = 3;
        value.format = 1;
        value.state_bits = 3;
        value.index_bits = 11;
        value.aux_bits = 1;
        value.table_bytes = 4096;
        value.runtime_table_bytes = 2048 * 8;
    } else if (profile == 4) {
        value.profile = 4;
        value.format = 9;
        value.state_bits = 2;
        value.index_bits = 6;
        value.aux_bits = 0;
        value.table_bytes = 320;
        value.runtime_table_bytes = 320;
        value.residual = true;
        value.residual_position_bits = 5;
        value.residual_record_bits = 15;
        value.residual_block_vectors = 24;
    } else if (profile == 5) {
        value.profile = 5;
        value.format = 8;
        value.state_bits = 4;
        value.index_bits = 9;
        value.aux_bits = 1;
        value.table_bytes = 2048;
        value.runtime_table_bytes = 1024 * 8;
        value.residual = true;
        value.residual_second = true;
        value.residual_position_bits = 4;
        value.residual_record_bits = 14;
        value.residual_block_vectors = 16;
    } else {
        throw std::runtime_error("unsupported NEPQ cohort profile");
    }
}

static std::vector<int8_t> expand_nepq_table(
        const uint8_t * source, const NepqCpu & value) {
    std::vector<uint8_t> packed(
        source, source + static_cast<ptrdiff_t>(value.table_bytes));
    if (value.profile == 0 || value.profile == 4) {
        const uint8_t expected[6] = {2, 4, 3, 3, 24, 8};
        for (int i = 0; i < 6; ++i) {
            if (packed[(size_t)i] != expected[i]) {
                throw std::runtime_error("unsupported NEPQ0-S table profile");
            }
        }
        std::vector<int8_t> result(value.table_bytes);
        std::memcpy(result.data(), packed.data(), packed.size());
        return result;
    }
    if (value.profile == 1) {
        const uint8_t expected[6] = {1, 8, 3, 4, 24, 8};
        for (int i = 0; i < 6; ++i) {
            if (packed[(size_t)i] != expected[i]) {
                throw std::runtime_error("unsupported NEPQ0-L table profile");
            }
        }
        std::vector<int8_t> result(value.table_bytes);
        std::memcpy(result.data(), packed.data(), packed.size());
        return result;
    }
    std::vector<uint16_t> words((size_t)value.table_bytes / 2);
    std::memcpy(words.data(), packed.data(), packed.size());
    return expand_nvq_codebook(
        (value.profile == 2 || value.profile == 5) ? 8 : 1,
        words.data(), words.size());
}

static NepqCpu unpack_nepq(
        const std::vector<uint8_t> & blob,
        const std::string & dtype,
        const std::vector<uint8_t> & runtime_payload) {
    if (blob.size() < 36 || std::memcmp(blob.data(), "NEP1", 4) != 0) {
        throw std::runtime_error("invalid NEPQ cohort header");
    }
    if (dtype != "NEPQ") {
        throw std::runtime_error("unsupported NEPQ cohort dtype: " + dtype);
    }
    NepqCpu value;
    size_t off = 4;
    const uint8_t version = blob[off++];
    const uint8_t profile = blob[off++];
    configure_nepq_profile(value, profile);
    const uint8_t groups_per_supergroup = blob[off++];
    const uint8_t flags = blob[off++];
    value.n_experts = (int)read_u32_from(blob, off);
    value.out_per_expert = (int)read_u32_from(blob, off);
    value.neuron_len = (int)read_u32_from(blob, off);
    value.bank_count = (int)read_u32_from(blob, off);
    value.rotation_block = (int)read_u32_from(blob, off);
    value.rotation_seed = read_u64_from(blob, off);
    if (version != 1 || profile != value.profile ||
        groups_per_supergroup != 4 || (flags & ~1u) != 0 ||
        ((flags & 1u) != 0) != (value.rotation_block != 0)) {
        throw std::runtime_error("unsupported NEPQ cohort profile");
    }
    if (value.n_experts <= 0 || value.out_per_expert <= 0 ||
        value.neuron_len <= 0 || value.neuron_len % 8 != 0 ||
        value.bank_count <= 0 || value.bank_count > 256 ||
        value.n_experts >
            std::numeric_limits<int>::max() / value.out_per_expert) {
        throw std::runtime_error("invalid NEPQ cohort dimensions");
    }
    if (value.rotation_block != 0 &&
        ((value.rotation_block & (value.rotation_block - 1)) != 0 ||
         value.neuron_len % value.rotation_block != 0)) {
        throw std::runtime_error("invalid NEPQ Hadamard block");
    }
    if (value.residual && value.rotation_block == 0) {
        throw std::runtime_error("NEPQ-A requires a Hadamard rotation");
    }
    value.ng = (value.neuron_len + 23) / 24;
    value.nvec = value.neuron_len / 8;
    value.nsuper = (value.ng + 3) / 4;
    const int rows = value.n_experts * value.out_per_expert;

    const size_t all_table_bytes =
        (size_t)value.bank_count * value.table_bytes;
    if (off + all_table_bytes > blob.size()) {
        throw std::runtime_error("truncated NEPQ table pool");
    }
    value.table_pool.reserve(
        (size_t)value.bank_count * value.runtime_table_bytes);
    value.grouped_table_pool.reserve(
        (size_t)value.bank_count *
        ((value.profile == 0 || value.profile == 4)
            ? 2112 : value.runtime_table_bytes));
    for (int bank = 0; bank < value.bank_count; ++bank) {
        const uint8_t * table = blob.data() + off + (size_t)bank * value.table_bytes;
        auto runtime = expand_nepq_table(table, value);
        value.table_pool.insert(
            value.table_pool.end(), runtime.begin(), runtime.end());
        if (value.profile == 0 || value.profile == 4) {
            std::vector<uint8_t> compact(
                table, table + static_cast<ptrdiff_t>(value.table_bytes));
            auto grouped = expand_npq0_s_runtime_lut(compact);
            value.grouped_table_pool.insert(
                value.grouped_table_pool.end(), grouped.begin(), grouped.end());
        } else {
            value.grouped_table_pool.insert(
                value.grouped_table_pool.end(), runtime.begin(), runtime.end());
        }
    }
    off += all_table_bytes;
    const size_t anchor_bytes = (size_t)rows * 2;
    if (off + anchor_bytes > blob.size()) {
        throw std::runtime_error("truncated NEPQ neuron anchors");
    }
    value.neuron_scale_h.resize((size_t)rows);
    std::memcpy(value.neuron_scale_h.data(), blob.data() + off, anchor_bytes);
    for (uint16_t raw : value.neuron_scale_h) {
        if ((raw & 0x7c00u) == 0x7c00u ||
                ((raw & 0x8000u) != 0 && (raw & 0x7fffu) != 0)) {
            throw std::runtime_error(
                "NEPQ neuron anchors must be finite and non-negative");
        }
    }
    off += anchor_bytes;
    value.state_packed = take_bytes(
        blob, off, packed_nbytes((size_t)rows * value.ng, value.state_bits),
        "NEPQ state stream");
    value.indices_packed = take_bytes(
        blob, off, packed_nbytes((size_t)rows * value.nvec, value.index_bits),
        "NEPQ index stream");
    value.aux_packed = take_bytes(
        blob, off, packed_nbytes((size_t)rows * value.ng, value.aux_bits),
        "NEPQ aux stream");
    value.bank_ids = take_bytes(
        blob, off, (size_t)rows * value.nsuper, "NEPQ bank selectors");
    if (value.residual) {
        const size_t header = off;
        if (header + 64 > blob.size() ||
            std::memcmp(blob.data() + header, "NRA1", 4) != 0) {
            throw std::runtime_error("invalid NEPQ-A residual header");
        }
        off += 4;
        const uint8_t residual_version = blob[off++];
        const uint8_t record_bits = blob[off++];
        const uint8_t position_bits = blob[off++];
        const uint8_t residual_flags = blob[off++];
        const uint32_t dictionary_entries = read_u32_from(blob, off);
        const uint32_t block_count = read_u32_from(blob, off);
        const uint32_t second_count = read_u32_from(blob, off);
        const uint32_t padding_nbytes = read_u32_from(blob, off);
        const uint64_t reserved = read_u64_from(blob, off);
        value.residual_blocks_per_row =
            (value.nvec + value.residual_block_vectors - 1) /
            value.residual_block_vectors;
        const size_t expected_blocks =
            static_cast<size_t>(rows) * value.residual_blocks_per_row;
        const uint8_t expected_flags = value.residual_second ? 1 : 0;
        if (residual_version != 1 ||
            record_bits != value.residual_record_bits ||
            position_bits != value.residual_position_bits ||
            residual_flags != expected_flags ||
            dictionary_entries != 1024 ||
            block_count != expected_blocks ||
            reserved != 0 ||
            (!value.residual_second && second_count != 0)) {
            throw std::runtime_error("unsupported NEPQ-A residual profile");
        }
        off = header + 64;
        const size_t dictionary_values = 1024 * 8;
        const size_t dictionary_nbytes = dictionary_values * sizeof(uint16_t);
        if (off + dictionary_nbytes > blob.size()) {
            throw std::runtime_error("truncated NEPQ-A residual dictionary");
        }
        value.residual_codebook_h.resize(dictionary_values);
        std::memcpy(
            value.residual_codebook_h.data(), blob.data() + off,
            dictionary_nbytes);
        for (size_t row = 0; row < 1024; ++row) {
            for (size_t column = 0; column < 8; ++column) {
                const uint16_t raw = value.residual_codebook_h[row * 8 + column];
                if ((raw & 0x7c00u) == 0x7c00u) {
                    throw std::runtime_error(
                        "NEPQ-A residual dictionary must be finite");
                }
            }
        }
        off += dictionary_nbytes;
        auto first = unpack_nepq_u16_bits(
            blob, off, expected_blocks, value.residual_record_bits,
            "NEPQ-A first residual stream");
        value.residual_first.assign(first.begin(), first.end());
        value.residual_second_dense.assign(expected_blocks, -1);
        if (value.residual_second) {
            auto mask = unpack_nepq_u16_bits(
                blob, off, expected_blocks, 1,
                "NEPQ-A second residual bitmap");
            auto second = unpack_nepq_u16_bits(
                blob, off, second_count, value.residual_record_bits,
                "NEPQ-A second residual stream");
            size_t compact = 0;
            for (size_t block = 0; block < expected_blocks; ++block) {
                if (mask[block] != 0) {
                    if (compact >= second.size()) {
                        throw std::runtime_error(
                            "NEPQ-A second residual count mismatch");
                    }
                    value.residual_second_dense[block] = second[compact++];
                }
            }
            if (compact != second.size()) {
                throw std::runtime_error(
                    "NEPQ-A second residual count mismatch");
            }
        }
        auto validate_record = [&](uint16_t record, size_t block) {
            const uint32_t position_mask =
                (1u << value.residual_position_bits) - 1u;
            const uint32_t position = record & position_mask;
            const uint32_t dictionary_id =
                record >> value.residual_position_bits;
            const size_t block_in_row =
                block % value.residual_blocks_per_row;
            const int available = std::min(
                value.residual_block_vectors,
                value.nvec - static_cast<int>(block_in_row) *
                    value.residual_block_vectors);
            if (dictionary_id >= 1024 || position >=
                    static_cast<uint32_t>(available)) {
                throw std::runtime_error(
                    "NEPQ-A residual record is out of range");
            }
        };
        for (size_t block = 0; block < expected_blocks; ++block) {
            validate_record(value.residual_first[block], block);
            if (value.residual_second_dense[block] >= 0) {
                validate_record(
                    static_cast<uint16_t>(
                        value.residual_second_dense[block]),
                    block);
            }
        }
        if (padding_nbytes > blob.size() - off) {
            throw std::runtime_error("truncated NEPQ-A residual padding");
        }
        const size_t padding_end = off + padding_nbytes;
        for (; off < padding_end; ++off) {
            if (blob[off] != 0) {
                throw std::runtime_error(
                    "NEPQ-A residual padding must be zero");
            }
        }
        const uint32_t position_mask =
            (1u << value.residual_position_bits) - 1u;
        for (size_t block = 0; block < expected_blocks; ++block) {
            const int block_in_row =
                static_cast<int>(block % value.residual_blocks_per_row);
            const int available = std::min(
                value.residual_block_vectors,
                value.nvec - block_in_row * value.residual_block_vectors);
            for (int record : {
                     value.residual_first[block],
                     value.residual_second_dense[block]}) {
                if (record < 0) continue;
                const int position = record & position_mask;
                const int dictionary_id =
                    record >> value.residual_position_bits;
                if (position >= available || dictionary_id >= 1024) {
                    throw std::runtime_error(
                        "invalid NEPQ-A residual record");
                }
            }
        }
    }
    if (off != blob.size()) throw std::runtime_error("invalid NEPQ cohort tail");
    for (uint8_t bank : value.bank_ids) {
        if ((int)bank >= value.bank_count) {
            throw std::runtime_error("NEPQ selector references a missing bank");
        }
    }

    if (value.rotation_block == 0) {
        if (!runtime_payload.empty()) {
            throw std::runtime_error("unexpected NEPQ rotation metadata");
        }
    } else {
        if (runtime_payload.size() != 20 + (size_t)value.neuron_len ||
            std::memcmp(runtime_payload.data(), "HSG1", 4) != 0) {
            throw std::runtime_error("missing NEPQ rotation sign vector");
        }
        size_t runtime_off = 4;
        uint32_t width = read_u32_from(runtime_payload, runtime_off);
        uint32_t block = read_u32_from(runtime_payload, runtime_off);
        uint64_t seed = read_u64_from(runtime_payload, runtime_off);
        if ((int)width != value.neuron_len ||
            (int)block != value.rotation_block || seed != value.rotation_seed) {
            throw std::runtime_error("NEPQ rotation metadata mismatch");
        }
        value.rotation_signs.resize((size_t)value.neuron_len);
        std::memcpy(
            value.rotation_signs.data(), runtime_payload.data() + runtime_off,
            (size_t)value.neuron_len);
        for (int8_t sign : value.rotation_signs) {
            if (sign != -1 && sign != 1) {
                throw std::runtime_error("invalid NEPQ rotation sign");
            }
        }
    }
    return value;
}

struct NepqWeight {
    mfq_tensor_backend::Tensor indices_packed;
    mfq_tensor_backend::Tensor aux_packed;
    mfq_tensor_backend::Tensor state_packed;
    mfq_tensor_backend::Tensor neuron_scale;
    mfq_tensor_backend::Tensor table_pool;
    mfq_tensor_backend::Tensor grouped_table_pool;
    mfq_tensor_backend::Tensor bank_ids;
    mfq_tensor_backend::Tensor rotation_signs;
    int format = 0;
    int state_bits = 0;
    int n_experts = 0;
    int out_per_expert = 0;
    int neuron_len = 0;
    int ng = 0;
    int rotation_block = 0;
    uint64_t rotation_seed = 0;
    bool residual = false;
    int residual_position_bits = 0;
    int residual_block_vectors = 0;
    mfq_tensor_backend::Tensor residual_codebook;
    mfq_tensor_backend::Tensor residual_first;
    mfq_tensor_backend::Tensor residual_second;
};

static NepqWeight to_device_nepq(const NepqCpu & cpu, bool cuda) {
    NepqWeight value;
    value.format = cpu.format;
    value.state_bits = cpu.state_bits;
    value.n_experts = cpu.n_experts;
    value.out_per_expert = cpu.out_per_expert;
    value.neuron_len = cpu.neuron_len;
    value.ng = cpu.ng;
    value.rotation_block = cpu.rotation_block;
    value.rotation_seed = cpu.rotation_seed;
    value.residual = cpu.residual;
    value.residual_position_bits = cpu.residual_position_bits;
    value.residual_block_vectors = cpu.residual_block_vectors;
    value.indices_packed = cpu_u8_tensor(
        cpu.indices_packed, {(int64_t)cpu.indices_packed.size()});
    value.aux_packed = cpu_u8_tensor(
        cpu.aux_packed, {(int64_t)cpu.aux_packed.size()});
    value.state_packed = cpu_u8_tensor(
        cpu.state_packed, {(int64_t)cpu.state_packed.size()});
    value.neuron_scale = cpu_f16_to_f32_tensor(
        cpu.neuron_scale_h, cpu.n_experts * cpu.out_per_expert);
    value.table_pool = cpu_i8_tensor(
        cpu.table_pool, {cpu.bank_count, cpu.runtime_table_bytes});
    const int grouped_stride =
        (cpu.profile == 0 || cpu.profile == 4)
        ? 2112 : cpu.runtime_table_bytes;
    value.grouped_table_pool = cpu_i8_tensor(
        cpu.grouped_table_pool, {cpu.bank_count, grouped_stride});
    value.bank_ids = cpu_u8_tensor(
        cpu.bank_ids, {
            (int64_t)cpu.n_experts * cpu.out_per_expert, cpu.nsuper});
    value.rotation_signs = cpu_i8_tensor(
        cpu.rotation_signs, {(int64_t)cpu.rotation_signs.size()});
    if (cpu.residual) {
        value.residual_codebook = cpu_f16_tensor(
            cpu.residual_codebook_h, {1024, 8});
        value.residual_first = mfq_tensor_backend::from_blob(
            (void *)cpu.residual_first.data(),
            {cpu.n_experts * cpu.out_per_expert,
             cpu.residual_blocks_per_row},
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt16)).clone();
        value.residual_second = mfq_tensor_backend::from_blob(
            (void *)cpu.residual_second_dense.data(),
            {cpu.n_experts * cpu.out_per_expert,
             cpu.residual_blocks_per_row},
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt16)).clone();
    }
    if (cuda) {
        const auto target = mfq_tensor_backend::Device(
            mfq_tensor_backend::kCUDA,
            mfq_current_cuda_device());
        value.indices_packed =
            value.indices_packed.to(target).contiguous();
        value.aux_packed =
            value.aux_packed.to(target).contiguous();
        value.state_packed =
            value.state_packed.to(target).contiguous();
        value.neuron_scale =
            value.neuron_scale.to(target).contiguous();
        value.table_pool =
            value.table_pool.to(target).contiguous();
        value.grouped_table_pool =
            value.grouped_table_pool.to(target).contiguous();
        value.bank_ids =
            value.bank_ids.to(target).contiguous();
        value.rotation_signs =
            value.rotation_signs.to(target).contiguous();
        if (cpu.residual) {
            value.residual_codebook =
                value.residual_codebook.to(target).contiguous();
            value.residual_first =
                value.residual_first.to(target).contiguous();
            value.residual_second =
                value.residual_second.to(target).contiguous();
        }
    }
    return value;
}

static NepqWeight to_gpu_nepq(const NepqCpu & cpu) {
    return to_device_nepq(cpu, true);
}

static NepqWeight to_cpu_nepq(const NepqCpu & cpu) {
    return to_device_nepq(cpu, false);
}

static NintCpu select_nint_cpu_rows(
        const NintCpu & source,
        const std::vector<int64_t> & rows) {
    require_tp_row_major_weight(
        source.shape, source.axis, source.out,
        source.neuron_len, "NINT");
    return repack_nint_cpu_rows(source, rows);
}

static Nint8ZeroCpu select_nint8_zero_cpu_rows(
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



static TpqWeight to_device_tpq(
        const TpqCpu & source, bool cuda, int device = -1) {
    TpqWeight result;
    result.int4 = source.int4;
    result.out = source.out;
    result.neuron_len = source.neuron_len;
    result.group_size = source.group_size;
    result.vector_size = source.vector_size;
    result.index_bits = source.index_bits;
    result.packed = cpu_u8_tensor(
        source.packed,
        source.int4
            ? std::initializer_list<int64_t>{
                source.out, source.neuron_len / 2}
            : std::initializer_list<int64_t>{
                static_cast<int64_t>(source.packed.size())});
    if (source.int4) {
        result.scales = cpu_f16_tensor(
            source.scales_h,
            {source.out, source.neuron_len / source.group_size});
    } else {
        result.codebook = mfq_tensor_backend::from_blob(
            const_cast<float *>(source.codebook.data()),
            {source.codebook_entries, source.vector_size},
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kFloat32)).clone();
    }
    if (cuda) {
        const int target_device = device >= 0
            ? device : mfq_current_cuda_device();
        MfqCudaGuard guard(target_device);
        const auto target = mfq_tensor_backend::Device(mfq_tensor_backend::kCUDA, target_device);
        result.packed = result.packed.to(target, false, false).contiguous();
        if (source.int4) {
            result.scales = result.scales.to(target, false, false).contiguous();
        } else {
            result.codebook = result.codebook.to(target, false, false).contiguous();
        }
    }
    return result;
}

static TpqCpu select_tpq_cpu_rows(
        const TpqCpu & source,
        const std::vector<int64_t> & rows) {
    if (rows.empty()) {
        throw std::runtime_error("cannot create an empty TPQ MoE shard");
    }
    TpqCpu result = source;
    result.out = static_cast<int64_t>(rows.size());
    if (source.int4) {
        const size_t packed_row = static_cast<size_t>(source.neuron_len / 2);
        const size_t scale_row = static_cast<size_t>(
            source.neuron_len / source.group_size);
        result.packed.resize(rows.size() * packed_row);
        result.scales_h.resize(rows.size() * scale_row);
        for (size_t destination = 0; destination < rows.size(); ++destination) {
            const int64_t source_row = rows[destination];
            if (source_row < 0 || source_row >= source.out) {
                throw std::runtime_error("TPQ MoE shard row is out of range");
            }
            std::memcpy(
                result.packed.data() + destination * packed_row,
                source.packed.data() +
                    static_cast<size_t>(source_row) * packed_row,
                packed_row);
            std::memcpy(
                result.scales_h.data() + destination * scale_row,
                source.scales_h.data() +
                    static_cast<size_t>(source_row) * scale_row,
                scale_row * sizeof(uint16_t));
        }
        return result;
    }
    const size_t vectors = static_cast<size_t>(
        source.neuron_len / source.vector_size);
    const size_t count = rows.size() * vectors;
    result.packed.assign((count * source.index_bits + 7) / 8, 0);
    for (size_t destination = 0; destination < rows.size(); ++destination) {
        const int64_t source_row = rows[destination];
        if (source_row < 0 || source_row >= source.out) {
            throw std::runtime_error("TPQ MoE shard row is out of range");
        }
        for (size_t vector = 0; vector < vectors; ++vector) {
            const uint32_t code = read_tpq_index_cpu(
                source.packed,
                static_cast<size_t>(source_row) * vectors + vector,
                source.index_bits);
            write_tpq_index_cpu(
                result.packed, destination * vectors + vector,
                source.index_bits, code);
        }
    }
    return result;
}

static NvqCpu select_nvq_cpu_rows(
        const NvqCpu & source,
        const std::vector<int64_t> & rows) {
    if (source.shape.size() != 2 ||
        source.axis != 0 ||
        source.shape[0] != source.out ||
        rows.empty()) {
        throw std::runtime_error(
            "invalid NVQ MoE output shard source");
    }
    NvqCpu result = source;
    result.out = static_cast<int>(rows.size());
    result.shape[0] = result.out;
    result.neuron_scale_h.resize(rows.size());
    const int index_bits =
        nvq_index_bits(source.format);
    const bool delta =
        nvq_delta_format(source.format);
    const bool no_aux =
        nvq_no_aux_format(source.format);
    const int aux_items =
        delta ? source.ng :
        (no_aux ? 0 : source.nsign);
    const int aux_bits =
        delta ? 1 : (no_aux ? 0 : 7);
    result.sub_scale_packed.assign(
        packed_nbytes(
            rows.size() *
                static_cast<size_t>(source.ng),
            source.sub_bits),
        0);
    result.indices_packed.assign(
        packed_nbytes(
            rows.size() *
                static_cast<size_t>(source.nvec),
            index_bits),
        0);
    result.aux_packed.assign(
        packed_nbytes(
            rows.size() *
                static_cast<size_t>(aux_items),
            aux_bits),
        0);
    for (size_t destination = 0;
         destination < rows.size(); ++destination) {
        const int64_t source_row = rows[destination];
        if (source_row < 0 ||
            source_row >= source.out) {
            throw std::runtime_error(
                "NVQ MoE shard row is out of range");
        }
        copy_packed_bits(
            source.sub_scale_packed,
            static_cast<size_t>(source_row) *
                source.ng * source.sub_bits,
            result.sub_scale_packed,
            destination * source.ng *
                source.sub_bits,
            static_cast<size_t>(source.ng) *
                source.sub_bits);
        copy_packed_bits(
            source.indices_packed,
            static_cast<size_t>(source_row) *
                source.nvec * index_bits,
            result.indices_packed,
            destination * source.nvec *
                index_bits,
            static_cast<size_t>(source.nvec) *
                index_bits);
        if (aux_bits != 0) {
            copy_packed_bits(
                source.aux_packed,
                static_cast<size_t>(source_row) *
                    aux_items * aux_bits,
                result.aux_packed,
                destination * aux_items *
                    aux_bits,
                static_cast<size_t>(aux_items) *
                    aux_bits);
        }
        result.neuron_scale_h[destination] =
            source.neuron_scale_h[
                static_cast<size_t>(source_row)];
    }
    return result;
}

static NepqCpu select_nepq_cpu_rows(
        const NepqCpu & source,
        const std::vector<int64_t> & rows,
        int output_per_expert,
        int selected_experts = -1) {
    const int destination_experts = selected_experts < 0
        ? source.n_experts
        : selected_experts;
    const int source_rows =
        source.n_experts *
        source.out_per_expert;
    if (rows.empty() ||
        output_per_expert <= 0 ||
        destination_experts <= 0 ||
        static_cast<int>(rows.size()) !=
            destination_experts *
                output_per_expert) {
        throw std::runtime_error(
            "invalid NEPQ MoE output shard");
    }
    NepqCpu result = source;
    result.n_experts = destination_experts;
    result.out_per_expert = output_per_expert;
    result.neuron_scale_h.resize(rows.size());
    result.state_packed.assign(
        packed_nbytes(
            rows.size() *
                static_cast<size_t>(source.ng),
            source.state_bits),
        0);
    result.indices_packed.assign(
        packed_nbytes(
            rows.size() *
                static_cast<size_t>(source.nvec),
            source.index_bits),
        0);
    result.aux_packed.assign(
        packed_nbytes(
            rows.size() *
                static_cast<size_t>(source.ng),
            source.aux_bits),
        0);
    result.bank_ids.resize(
        rows.size() * source.nsuper);
    if (source.residual) {
        result.residual_first.resize(
            rows.size() * source.residual_blocks_per_row);
        result.residual_second_dense.resize(
            rows.size() * source.residual_blocks_per_row);
    }
    for (size_t destination = 0;
         destination < rows.size(); ++destination) {
        const int64_t source_row = rows[destination];
        if (source_row < 0 ||
            source_row >= source_rows) {
            throw std::runtime_error(
                "NEPQ MoE shard row is out of range");
        }
        copy_packed_bits(
            source.state_packed,
            static_cast<size_t>(source_row) *
                source.ng * source.state_bits,
            result.state_packed,
            destination * source.ng *
                source.state_bits,
            static_cast<size_t>(source.ng) *
                source.state_bits);
        copy_packed_bits(
            source.indices_packed,
            static_cast<size_t>(source_row) *
                source.nvec * source.index_bits,
            result.indices_packed,
            destination * source.nvec *
                source.index_bits,
            static_cast<size_t>(source.nvec) *
                source.index_bits);
        if (source.aux_bits != 0) {
            copy_packed_bits(
                source.aux_packed,
                static_cast<size_t>(source_row) *
                    source.ng * source.aux_bits,
                result.aux_packed,
                destination * source.ng *
                    source.aux_bits,
                static_cast<size_t>(source.ng) *
                    source.aux_bits);
        }
        std::memcpy(
            result.bank_ids.data() +
                destination * source.nsuper,
            source.bank_ids.data() +
                static_cast<size_t>(source_row) *
                    source.nsuper,
            static_cast<size_t>(source.nsuper));
        result.neuron_scale_h[destination] =
            source.neuron_scale_h[
                static_cast<size_t>(source_row)];
        if (source.residual) {
            std::memcpy(
                result.residual_first.data()
                    + destination * source.residual_blocks_per_row,
                source.residual_first.data()
                    + static_cast<size_t>(source_row)
                        * source.residual_blocks_per_row,
                static_cast<size_t>(source.residual_blocks_per_row)
                    * sizeof(int16_t));
            std::memcpy(
                result.residual_second_dense.data()
                    + destination * source.residual_blocks_per_row,
                source.residual_second_dense.data()
                    + static_cast<size_t>(source_row)
                        * source.residual_blocks_per_row,
                static_cast<size_t>(source.residual_blocks_per_row)
                    * sizeof(int16_t));
        }
    }
    return result;
}

enum class MixedMoeFamily {
    Nint,
    Nint8Zero,
    Mxfp4,
    Mxfp4Sq,
    Fp8Sq,
    Tpq,
    Nvq,
    Nepq,
};

struct MixedMoePool {
    MixedMoeFamily family = MixedMoeFamily::Nint;
    NintWeight nint;
    NintWeight q8_zero;
    Mxfp4Weight mxfp4;
    Mxfp4SqWeight mxfp4_sq;
    Fp8SqWeight fp8_sq;
    TpqWeight tpq;
    NvqWeight nvq;
    NepqWeight nepq;
    mfq_tensor_backend::Tensor expert_local;
    int local_experts = 0;
};

static void fp8_sq_moe_matmul(
        const Fp8SqWeight & weight,
        mfq_tensor_backend::Tensor input,
        mfq_tensor_backend::Tensor expert_ids,
        mfq_tensor_backend::Tensor expert_local,
        int n_experts,
        int local_experts,
        int out_per_expert,
        int neuron_len,
        mfq_tensor_backend::Tensor output) {
    if (weight.dtype == "MXFP8-SQ") {
        mxfp8_sq_moe_matmul_cuda(
            weight.blob, weight.row_q, weight.row_symbol_byte_offsets,
            std::move(input), std::move(expert_ids),
            std::move(expert_local), n_experts, local_experts,
            out_per_expert, neuron_len, weight.block_rows,
            weight.block_columns, weight.scale_rows, weight.scale_columns,
            weight.palettes_offset, weight.symbols_offset,
            weight.scales_offset, std::move(output));
        return;
    }
    if (weight.dtype == "FP8-128SQ") {
        fp8_128_sq_moe_matmul_cuda(
            weight.blob, weight.row_q, weight.row_symbol_byte_offsets,
            std::move(input), std::move(expert_ids),
            std::move(expert_local), n_experts, local_experts,
            out_per_expert, neuron_len, weight.scale_kind,
            weight.palettes_offset, weight.symbols_offset,
            weight.scales_offset, std::move(output));
        return;
    }
    throw std::runtime_error("unsupported FP8-SQ MoE dtype");
}

struct MixedMoeTransformKey {
    int block = 0;
    uint64_t seed = 0;

    bool operator==(const MixedMoeTransformKey & other) const {
        return block == other.block && seed == other.seed;
    }
};

struct MixedMoeTransformKeyHash {
    size_t operator()(const MixedMoeTransformKey & key) const {
        return ((size_t)key.block * 1315423911u) ^
            (size_t)(key.seed ^ (key.seed >> 32));
    }
};

struct MixedMoeActivationKey {
    int input_rows = 0;
    int groups = 0;
    int gs = 0;
    int device = 0;
    MixedMoeTransformKey transform;

    bool operator==(const MixedMoeActivationKey & other) const {
        return input_rows == other.input_rows && groups == other.groups &&
            gs == other.gs && device == other.device &&
            transform == other.transform;
    }
};

struct MixedMoeActivationKeyHash {
    size_t operator()(const MixedMoeActivationKey & key) const {
        size_t value = (size_t)key.input_rows;
        value = value * 1315423911u + (size_t)key.groups;
        value = value * 1315423911u + (size_t)key.gs;
        value = value * 1315423911u + (size_t)key.device;
        return value * 1315423911u +
            MixedMoeTransformKeyHash{}(key.transform);
    }
};

enum class MixedNvqF16FormatGroup : int {
    All = 0,
    Standard = 1,
    Extended = 2,
    Legacy = 3,
};

static MixedNvqF16FormatGroup mixed_nvq_f16_format_group(int format) {
    switch (format) {
        case 2:
        case 3:
        case 4:
        case 5:
        case 6:
        case 10:
        case 11:
            return MixedNvqF16FormatGroup::Standard;
        case 12:
        case 13:
        case 14:
        case 15:
        case 16:
        case 17:
            return MixedNvqF16FormatGroup::Extended;
        case 1:
        case 7:
        case 8:
        case 9:
            return MixedNvqF16FormatGroup::Legacy;
        default:
            return MixedNvqF16FormatGroup::All;
    }
}

struct MixedNvqDispatch {
    mfq_tensor_backend::Tensor weight_ptrs;
    mfq_tensor_backend::Tensor weight_sizes;
    mfq_tensor_backend::Tensor pool_params;
    mfq_tensor_backend::Tensor expert_pool;
    mfq_tensor_backend::Tensor expert_local;
    int pool_count = 0;
    MixedNvqF16FormatGroup f16_format_group =
        MixedNvqF16FormatGroup::All;
    bool masked_experts = false;
};

struct MixedMoeRuntime {
    int n_experts = 0;
    int out_per_expert = 0;
    int neuron_len = 0;
    bool partial_experts = false;
    std::vector<MixedMoePool> pools;
    std::shared_ptr<MixedNvqDispatch> nvq_dispatch;

    bool nint_only() const {
        return !pools.empty() && std::all_of(
            pools.begin(), pools.end(), [](const MixedMoePool & pool) {
                return pool.family == MixedMoeFamily::Nint;
            });
    }
    MoeActivationWorkspace & activation_workspace(
            mfq_tensor_backend::Tensor x, int input_rows, int groups, int gs,
            MixedMoeTransformKey transform) const {
        static thread_local std::unordered_map<
            MixedMoeActivationKey, MoeActivationWorkspace,
            MixedMoeActivationKeyHash> shared_activation_workspaces;
        MixedMoeActivationKey key{
            input_rows, groups, gs, x.get_device(), transform};
        auto found = shared_activation_workspaces.find(key);
        if (found != shared_activation_workspaces.end()) return found->second;
        MoeActivationWorkspace value;
        value.qx = mfq_tensor_backend::empty(
            {input_rows, groups * gs}, x.options().dtype(mfq_tensor_backend::kInt8));
        value.xscale = mfq_tensor_backend::empty(
            {input_rows, groups}, x.options().dtype(mfq_tensor_backend::kFloat32));
        return shared_activation_workspaces.emplace(
            key, std::move(value)).first->second;
    }

    std::vector<MoeActivationGeometry> activation_geometry() const {
        std::vector<MoeActivationGeometry> result;
        for (const auto & pool : pools) {
            int groups = 0;
            int gs = 24;
            int transform_block = 0;
            uint64_t transform_seed = 0;
            if (pool.family == MixedMoeFamily::Nint) {
                groups = static_cast<int>(pool.nint.ng);
                gs = static_cast<int>(pool.nint.gs);
            } else if (pool.family == MixedMoeFamily::Nint8Zero) {
                groups = static_cast<int>(pool.q8_zero.ng);
                gs = 32;
            } else if (pool.family == MixedMoeFamily::Nvq) {
                groups = static_cast<int>(pool.nvq.ng);
                gs = static_cast<int>(pool.nvq.gs);
            } else if (pool.family == MixedMoeFamily::Mxfp4 ||
                    pool.family == MixedMoeFamily::Mxfp4Sq ||
                    pool.family == MixedMoeFamily::Fp8Sq ||
                    pool.family == MixedMoeFamily::Tpq) {
                continue;
            } else {
                groups = pool.nepq.ng;
                transform_block = pool.nepq.rotation_block;
                transform_seed = pool.nepq.rotation_seed;
            }
            const MoeActivationGeometry geometry{
                groups, gs, transform_block, transform_seed};
            if (std::find(result.begin(), result.end(), geometry) == result.end()) {
                result.push_back(geometry);
            }
        }
        return result;
    }

    mfq_tensor_backend::Tensor forward(
            mfq_tensor_backend::Tensor x,
            const MoeRoutePlan & route,
            bool input_prequantized = false,
            int epilogue_mode = 0) const {
        if (epilogue_mode < 0 || epilogue_mode > 2 ||
                (epilogue_mode != 0 &&
                 (!nint_only() || (out_per_expert % 2) != 0))) {
            throw std::runtime_error(
                "mixed MFE projection epilogue requires only canonical NINT");
        }
        if (!x.is_cuda() || !x.is_contiguous() ||
            x.scalar_type() != mfq_tensor_backend::kFloat16 ||
            (x.dim() != 2 && x.dim() != 3) || x.size(-1) != neuron_len) {
            throw std::runtime_error(
                "mixed MFE input must be contiguous CUDA f16 with exact K");
        }
        const int tokens = (int)route.ids.size(0);
        const int routes = (int)route.ids.size(1);
        if (route.n_experts != n_experts || x.size(0) != tokens ||
            (x.dim() == 3 && x.size(1) != routes)) {
            throw std::runtime_error("mixed MFE input and route shape mismatch");
        }
        const int input_rows = x.dim() == 3 ? tokens * routes : tokens;
        const int result_width = epilogue_mode == 0
            ? out_per_expert
            : out_per_expert / 2;
        auto output = partial_experts
            ? mfq_tensor_backend::zeros(
                {tokens, routes, result_width},
                x.options().dtype(mfq_tensor_backend::kFloat16))
            : mfq_tensor_backend::empty(
                {tokens, routes, result_width},
                x.options().dtype(mfq_tensor_backend::kFloat16));
        std::unordered_map<
            MixedMoeTransformKey, mfq_tensor_backend::Tensor,
            MixedMoeTransformKeyHash> prepared_inputs;
        const MixedMoeTransformKey identity{};
        prepared_inputs.emplace(identity, x);
        std::unordered_set<
            MixedMoeActivationKey, MixedMoeActivationKeyHash> quantized;
        static const bool disable_prefill_mma = [] {
            const char * value = std::getenv("MFQ_DISABLE_MOE_PREFILL_MMA");
            return value != nullptr && std::atoi(value) != 0;
        }();
        static const int prefill_mma_min_tokens = [] {
            const char * value = std::getenv("MFQ_MOE_PREFILL_MMA_MIN_TOKENS");
            return value == nullptr ? 9 : std::max(9, std::atoi(value));
        }();
        static const bool disable_nvq_hetero_decode = [] {
            const char * disabled =
                std::getenv("MFQ_DISABLE_MOE_NVQ_HETERO_DECODE");
            const char * exact =
                std::getenv("MFQ_NVQ_MOE_EXACT_REDUCTION");
            const char * rows =
                std::getenv("MFQ_NVQ_MOE_ROWS_PER_BLOCK");
            const char * warps =
                std::getenv("MFQ_NVQ_MOE_WARPS");
            const char * shared =
                std::getenv("MFQ_NVQ_MOE_SHARE_GROUP_STATE");
            return (disabled != nullptr && std::atoi(disabled) != 0) ||
                (exact != nullptr && std::atoi(exact) != 0) ||
                rows != nullptr || warps != nullptr || shared != nullptr;
        }();
        const bool use_f16_mma =
            !disable_prefill_mma && !g_force_moe_prefill_mma_off &&
            tokens >= prefill_mma_min_tokens && route.map_ready &&
            route.ids_dst.numel() == route.ids.numel();
        const bool use_kl_mmq = g_kl_mmq_mode != KlMmqMode::Default;
        const bool nvq_hetero_prefill_ready = nvq_dispatch &&
            (nvq_dispatch->pool_count > 1 ||
             (out_per_expert >= 128 &&
              static_cast<int64_t>(tokens) * routes >
                  static_cast<int64_t>(n_experts) * 16));
        const bool use_nvq_prefill =
            use_f16_mma && !use_kl_mmq && nvq_hetero_prefill_ready;
        const bool use_nvq_decode =
            !use_f16_mma && !use_kl_mmq && nvq_dispatch &&
            nvq_dispatch->pool_count > 1 &&
            tokens <= 8 && !g_force_moe_pool_path &&
            !disable_nvq_hetero_decode;
        int nint_pool_phase = 0;
        if (input_prequantized && use_kl_mmq) {
            throw std::runtime_error(
                "mixed prequantized activation reuse is unavailable in KLD MMQ mode");
        }
        if (use_kl_mmq) {
            MFQ_RUNTIME_CHECK(
                route.map_ready && route.ids_dst.numel() == route.ids.numel(),
                "KLD mixed routed FP16 requires the compact route map");
        }

        if (use_nvq_prefill) {
            const int routed_rows_per_expert = std::max(
                1, (tokens * routes + n_experts - 1) / n_experts);
            const bool use_coarse_nvq_tiles =
                routed_rows_per_expert > 64 && route.wide_tile_m > 8;
            const int nvq_tile_m = use_coarse_nvq_tiles
                ? route.wide_tile_m : 8;
            const auto & nvq_tile_bounds = use_coarse_nvq_tiles
                ? route.wide_tile_bounds : route.tile_bounds;
            const auto & nvq_tile_experts = use_coarse_nvq_tiles
                ? route.wide_tile_experts : route.tile_experts;
            nvq_moe_grouped_matmul_hetero_f16_cuda(
                nvq_dispatch->weight_ptrs,
                nvq_dispatch->weight_sizes,
                nvq_dispatch->pool_params,
                nvq_dispatch->expert_pool,
                nvq_dispatch->expert_local,
                x, n_experts, out_per_expert, neuron_len,
                nvq_tile_m, output,
                route.ids_dst, route.expert_bounds,
                nvq_tile_bounds, nvq_tile_experts,
                static_cast<int>(nvq_dispatch->f16_format_group),
                nvq_dispatch->masked_experts);
        }

        if (use_nvq_decode) {
            const int groups = (neuron_len + 23) / 24;
            const MixedMoeActivationKey activation_key{
                input_rows, groups, 24, x.get_device(), identity};
            auto & workspace = activation_workspace(
                x, input_rows, groups, 24, identity);
            auto qx = workspace.qx;
            auto xscale = workspace.xscale;
            bool input_quantized = input_prequantized;
            nvq_moe_grouped_matmul_hetero_ws_cuda(
                nvq_dispatch->weight_ptrs,
                nvq_dispatch->weight_sizes,
                nvq_dispatch->pool_params,
                nvq_dispatch->expert_pool,
                nvq_dispatch->expert_local,
                x, route.ids, n_experts, out_per_expert, neuron_len,
                input_quantized, output, qx, xscale);
            quantized.insert(activation_key);
        }

        for (const auto & pool : pools) {
            if (pool.family == MixedMoeFamily::Nvq &&
                    (use_nvq_prefill || use_nvq_decode)) {
                continue;
            }
            int gs = 24;
            int groups = 0;
            MixedMoeTransformKey transform{};
            mfq_tensor_backend::Tensor value = x;
            if (pool.family == MixedMoeFamily::Nint) {
                gs = (int)pool.nint.gs;
                groups = (int)pool.nint.ng;
            } else if (pool.family == MixedMoeFamily::Nint8Zero) {
                gs = 32;
                groups = (int)pool.q8_zero.ng;
            } else if (pool.family == MixedMoeFamily::Nvq) {
                gs = (int)pool.nvq.gs;
                groups = (int)pool.nvq.ng;
            } else if (pool.family == MixedMoeFamily::Mxfp4 ||
                    pool.family == MixedMoeFamily::Mxfp4Sq ||
                    pool.family == MixedMoeFamily::Fp8Sq ||
                    pool.family == MixedMoeFamily::Tpq) {
                groups = 0;
            } else {
                groups = pool.nepq.ng;
                transform = {
                    pool.nepq.rotation_block,
                    pool.nepq.rotation_seed,
                };
                if (transform.block != 0 && !input_prequantized) {
                    auto found = prepared_inputs.find(transform);
                    if (found == prepared_inputs.end()) {
                        auto flat = x.reshape({input_rows, neuron_len}).contiguous();
                        auto rotated = nepq_hadamard_input_cuda(
                            flat, pool.nepq.rotation_signs, transform.block);
                        value = x.dim() == 3
                            ? rotated.reshape({tokens, routes, neuron_len})
                            : rotated;
                        prepared_inputs.emplace(transform, value);
                    } else {
                        value = found->second;
                    }
                }
            }
            if (use_kl_mmq) {
                value = kl_mmq_prepare_activation(value);
                ++g_kl_mmq_moe_calls;
                if (pool.family == MixedMoeFamily::Mxfp4) {
                    mxfp4_moe_grouped_matmul_pool_f16_cuda(
                        pool.mxfp4.values, pool.mxfp4.scales, value,
                        route.ids, pool.expert_local, n_experts,
                        pool.local_experts, out_per_expert, neuron_len,
                        output, route.ids_dst, route.expert_bounds,
                        route.tile_bounds, route.tile_experts);
                    continue;
                }
                if (pool.family == MixedMoeFamily::Mxfp4Sq) {
                    mxfp4_sq_moe_matmul_cuda(
                        pool.mxfp4_sq.blob,
                        pool.mxfp4_sq.row_q,
                        pool.mxfp4_sq.row_symbol_byte_offsets,
                        pool.mxfp4_sq.row_auxiliary,
                        value, route.ids,
                        pool.expert_local, pool.mxfp4_sq.bits,
                        n_experts, pool.local_experts, out_per_expert,
                        neuron_len, pool.mxfp4_sq.matrix_scale_base,
                        pool.mxfp4_sq.q_sum,
                        pool.mxfp4_sq.sq4_rows,
                        output);
                    continue;
                }
                if (pool.family == MixedMoeFamily::Fp8Sq) {
                    fp8_sq_moe_matmul(
                        pool.fp8_sq, value, route.ids, pool.expert_local,
                        n_experts, pool.local_experts, out_per_expert,
                        neuron_len, output);
                    continue;
                }
                if (pool.family == MixedMoeFamily::Tpq) {
                    tpq_pq_moe_grouped_matmul_pool_f16_cuda(
                        pool.tpq.packed, pool.tpq.codebook, value,
                        route.ids, pool.expert_local, n_experts,
                        pool.local_experts, out_per_expert, neuron_len,
                        pool.tpq.vector_size, pool.tpq.index_bits, output,
                        route.ids_dst, route.expert_bounds,
                        route.tile_bounds, route.tile_experts);
                    continue;
                }
                if (pool.family == MixedMoeFamily::Nvq) {
                    nvq_moe_grouped_matmul_pool_f16_cuda(
                        pool.nvq.indices_packed, pool.nvq.aux_packed,
                        pool.nvq.sub_scale_packed, pool.nvq.neuron_scale,
                        pool.nvq.codebook, value, pool.expert_local,
                        n_experts, pool.local_experts, out_per_expert,
                        neuron_len, pool.nvq.gs, pool.nvq.sub_bits,
                        pool.nvq.kernel_format, pool.nvq.sign_mode, output,
                        route.ids_dst, route.expert_bounds,
                        route.tile_bounds, route.tile_experts);
                    continue;
                }
                if (pool.family == MixedMoeFamily::Nepq) {
                    nepq_moe_grouped_matmul_pool_f16_cuda(
                        pool.nepq.indices_packed, pool.nepq.aux_packed,
                        pool.nepq.state_packed, pool.nepq.neuron_scale,
                        pool.nepq.table_pool, pool.nepq.bank_ids, value,
                        pool.expert_local, n_experts, pool.local_experts,
                        out_per_expert, neuron_len, pool.nepq.state_bits,
                        pool.nepq.format, output, route.ids_dst,
                        route.expert_bounds, route.tile_bounds,
                        route.tile_experts);
                    if (pool.nepq.residual) {
                        nepq_sparse_residual_grouped_cuda(
                            pool.nepq.residual_codebook,
                            pool.nepq.residual_first,
                            pool.nepq.residual_second,
                            value, route.ids, pool.expert_local,
                            out_per_expert,
                            pool.nepq.residual_position_bits,
                            pool.nepq.residual_block_vectors,
                            output);
                    }
                    continue;
                }
                ++g_kl_mmq_fallback_calls;
                throw std::runtime_error(
                    "KLD mixed routed FP16 encountered a non-VQ pool");
            }
            if (pool.family == MixedMoeFamily::Mxfp4) {
                mxfp4_moe_grouped_matmul_pool_f16_cuda(
                    pool.mxfp4.values, pool.mxfp4.scales, value,
                    route.ids, pool.expert_local, n_experts,
                    pool.local_experts, out_per_expert, neuron_len,
                    output, route.ids_dst, route.expert_bounds,
                    route.tile_bounds, route.tile_experts);
                continue;
            }
            if (pool.family == MixedMoeFamily::Mxfp4Sq) {
                mxfp4_sq_moe_matmul_cuda(
                    pool.mxfp4_sq.blob,
                    pool.mxfp4_sq.row_q,
                    pool.mxfp4_sq.row_symbol_byte_offsets,
                    pool.mxfp4_sq.row_auxiliary,
                    value, route.ids,
                    pool.expert_local, pool.mxfp4_sq.bits,
                    n_experts, pool.local_experts, out_per_expert,
                    neuron_len, pool.mxfp4_sq.matrix_scale_base,
                    pool.mxfp4_sq.q_sum,
                    pool.mxfp4_sq.sq4_rows,
                    output);
                continue;
            }
            if (pool.family == MixedMoeFamily::Fp8Sq) {
                fp8_sq_moe_matmul(
                    pool.fp8_sq, value, route.ids, pool.expert_local,
                    n_experts, pool.local_experts, out_per_expert,
                    neuron_len, output);
                continue;
            }
            if (pool.family == MixedMoeFamily::Tpq) {
                tpq_pq_moe_grouped_matmul_pool_f16_cuda(
                    pool.tpq.packed, pool.tpq.codebook, value,
                    route.ids, pool.expert_local, n_experts,
                    pool.local_experts, out_per_expert, neuron_len,
                    pool.tpq.vector_size, pool.tpq.index_bits, output,
                    route.ids_dst, route.expert_bounds,
                    route.tile_bounds, route.tile_experts);
                continue;
            }
            if (pool.family == MixedMoeFamily::Nvq && use_f16_mma) {
                nvq_moe_grouped_matmul_pool_f16_cuda(
                    pool.nvq.indices_packed, pool.nvq.aux_packed,
                    pool.nvq.sub_scale_packed, pool.nvq.neuron_scale,
                    pool.nvq.codebook, value, pool.expert_local,
                    n_experts, pool.local_experts, out_per_expert,
                    neuron_len, pool.nvq.gs, pool.nvq.sub_bits,
                    pool.nvq.kernel_format, pool.nvq.sign_mode, output,
                    route.ids_dst, route.expert_bounds,
                    route.tile_bounds, route.tile_experts);
                continue;
            }
            MixedMoeActivationKey activation_key{
                input_rows, groups, gs, x.get_device(), transform};
            auto & workspace = activation_workspace(
                value, input_rows, groups, gs, transform);
            bool input_quantized = input_prequantized ||
                quantized.find(activation_key) != quantized.end();
            bool activation_quantized_after_call = true;
            auto qx = workspace.qx;
            auto xscale = workspace.xscale;
            if (pool.family == MixedMoeFamily::Nint) {
                const int route_tile_m = select_nint_prefill_route_tile(
                    route, tokens, routes, n_experts);
                mfe_nint_matmul_ws_cuda(
                    pool.nint.q_packed, pool.nint.row_q_bits,
                    pool.nint.row_q_bit_offsets, pool.nint.sub_scale,
                    pool.nint.sub_min, pool.nint.neuron_scale,
                    pool.nint.neuron_min, value, route.ids,
                    pool.expert_local, n_experts, pool.local_experts,
                    out_per_expert, gs, epilogue_mode, route.map_ready,
                    input_quantized,
                    output, qx, xscale, route.ids_dst,
                    route.expert_bounds,
                    nint_route_tile_bounds(route, route_tile_m),
                    nint_route_tile_experts(route, route_tile_m),
                    route_tile_m, nint_pool_phase++);
                activation_quantized_after_call =
                    input_quantized || route_tile_m == 8;
            } else if (pool.family == MixedMoeFamily::Nint8Zero) {
                const int routed_rows_per_expert = std::max(
                    1, (tokens * routes + n_experts - 1) / n_experts);
                const bool use_q8_f16_mma = use_f16_mma &&
                    (routed_rows_per_expert >= 5 ||
                     tokens >= n_experts * 4);
                const bool use_coarse_q8_tiles = use_q8_f16_mma &&
                    route.mma_tile_m == 64 && routed_rows_per_expert > 32;
                nint8_zero_moe_grouped_matmul_pool_ws_cuda(
                    pool.q8_zero.q_packed, pool.q8_zero.q8_zero_scale, value,
                    route.ids, pool.expert_local, n_experts,
                    pool.local_experts, out_per_expert, route.map_ready,
                    input_quantized, use_q8_f16_mma, output,
                    qx, xscale,
                    route.counts, route.cursors, route.ids_dst,
                    route.expert_bounds,
                    use_coarse_q8_tiles
                        ? route.mma_tile_bounds : route.tile_bounds,
                    use_coarse_q8_tiles
                        ? route.mma_tile_experts : route.tile_experts,
                    use_coarse_q8_tiles ? route.mma_tile_m : 8);
            } else if (pool.family == MixedMoeFamily::Nvq) {
                nvq_moe_grouped_matmul_pool_ws_cuda(
                    pool.nvq.indices_packed, pool.nvq.aux_packed,
                    pool.nvq.sub_scale_packed, pool.nvq.neuron_scale,
                    pool.nvq.codebook, value, route.ids, pool.expert_local,
                    n_experts, pool.local_experts, out_per_expert, neuron_len,
                    pool.nvq.gs, pool.nvq.sub_bits, pool.nvq.kernel_format,
                    pool.nvq.sign_mode, input_quantized, output,
                    qx, xscale, route.ids_dst,
                    route.expert_bounds, route.tile_bounds, route.tile_experts);
            } else {
                nepq_moe_grouped_matmul_pool_ws_cuda(
                    pool.nepq.indices_packed, pool.nepq.aux_packed,
                    pool.nepq.state_packed, pool.nepq.neuron_scale,
                    pool.nepq.table_pool, pool.nepq.bank_ids,
                    pool.nepq.grouped_table_pool, value, route.ids,
                    pool.expert_local, n_experts, pool.local_experts,
                    out_per_expert, neuron_len, pool.nepq.state_bits,
                    pool.nepq.format, input_quantized, output,
                    qx, xscale, route.ids_dst,
                    route.expert_bounds, route.tile_bounds, route.tile_experts);
                if (pool.nepq.residual) {
                    nepq_sparse_residual_grouped_cuda(
                        pool.nepq.residual_codebook,
                        pool.nepq.residual_first,
                        pool.nepq.residual_second,
                        value, route.ids, pool.expert_local,
                        out_per_expert,
                        pool.nepq.residual_position_bits,
                        pool.nepq.residual_block_vectors,
                        output);
                }
            }
            if (activation_quantized_after_call) {
                quantized.insert(activation_key);
            }
        }
        return output;
    }

    mfq_tensor_backend::Tensor forward_glu_output(
            mfq_tensor_backend::Tensor x,
            const MoeRoutePlan & route,
            bool gelu) const {
        if (nint_only() && g_kl_mmq_mode == KlMmqMode::Default) {
            return forward(x, route, false, gelu ? 2 : 1);
        }
        auto gate_up = forward(x, route);
        return gelu
            ? moe_geglu_split_cuda(gate_up)
            : moe_swiglu_split_cuda(gate_up);
    }

    bool supports_clamped_swiglu() const {
        return !pools.empty();
    }

    mfq_tensor_backend::Tensor forward_clamped_swiglu(
            mfq_tensor_backend::Tensor gate_up,
            const MoeRoutePlan & route,
            double limit) const {
        if (!supports_clamped_swiglu() ||
            !gate_up.is_cuda() || !gate_up.is_contiguous() ||
            gate_up.scalar_type() != mfq_tensor_backend::kFloat16 ||
            gate_up.dim() != 3 || gate_up.size(2) != 2 * neuron_len) {
            throw std::runtime_error(
                "mixed MFE clamped SwiGLU input is unsupported");
        }
        const int tokens = static_cast<int>(route.ids.size(0));
        const int routes = static_cast<int>(route.ids.size(1));
        if (route.n_experts != n_experts ||
            gate_up.size(0) != tokens || gate_up.size(1) != routes) {
            throw std::runtime_error(
                "mixed MFE clamped SwiGLU route shape mismatch");
        }
        auto parts = gate_up.split_with_sizes(
            {neuron_len, neuron_len}, -1);
        auto gate = mfq_tensor_backend::clamp_max(parts[0], limit);
        auto up = mfq_tensor_backend::clamp(parts[1], -limit, limit);
        auto activation =
            (mfq_tensor_backend::silu(gate) * up).contiguous();
        return forward(activation, route);
    }
};

static void initialize_mixed_nvq_dispatch(
        MixedMoeRuntime & runtime) {
    runtime.nvq_dispatch.reset();
    const char * disabled =
        std::getenv("MFQ_DISABLE_MOE_NVQ_HETERO");
    if (disabled != nullptr && std::atoi(disabled) != 0) return;

    int nvq_pools = 0;
    for (const auto & pool : runtime.pools) {
        if (pool.family == MixedMoeFamily::Nvq) ++nvq_pools;
    }
    if (nvq_pools == 0) return;

    std::vector<int64_t> weight_ptrs;
    std::vector<int64_t> weight_sizes;
    std::vector<int32_t> pool_params;
    std::vector<int32_t> expert_pool(
        static_cast<size_t>(runtime.n_experts), -1);
    std::vector<int32_t> expert_local(
        static_cast<size_t>(runtime.n_experts), -1);
    weight_ptrs.reserve(static_cast<size_t>(nvq_pools) * 5);
    weight_sizes.reserve(static_cast<size_t>(nvq_pools) * 3);
    pool_params.reserve(static_cast<size_t>(nvq_pools) * 7);

    mfq_tensor_backend::Device target = mfq_tensor_backend::Device(
        mfq_tensor_backend::kCUDA, mfq_current_cuda_device());
    int dispatch_pool = 0;
    int owned_experts = 0;
    MixedNvqF16FormatGroup f16_format_group =
        MixedNvqF16FormatGroup::All;
    bool first_nvq_format = true;
    for (const auto & pool : runtime.pools) {
        if (pool.family != MixedMoeFamily::Nvq) continue;
        const auto & weight = pool.nvq;
        if (weight.gs != 24 || weight.ng <= 0 ||
                weight.ng > std::numeric_limits<int32_t>::max() ||
                weight.sub_bits < 1 || weight.sub_bits > 8 ||
                weight.kernel_format < 1 || weight.kernel_format > 17 ||
                weight.sign_mode < 0 || weight.sign_mode > 1) {
            return;
        }
        if (dispatch_pool == 0) target = weight.indices_packed.device();
        if (weight.indices_packed.device() != target ||
                weight.aux_packed.device() != target ||
                weight.sub_scale_packed.device() != target ||
                weight.neuron_scale.device() != target ||
                weight.codebook.device() != target) {
            return;
        }
        weight_ptrs.push_back(static_cast<int64_t>(
            reinterpret_cast<uintptr_t>(
                weight.indices_packed.data_ptr<uint8_t>())));
        weight_ptrs.push_back(static_cast<int64_t>(
            reinterpret_cast<uintptr_t>(
                weight.aux_packed.data_ptr<uint8_t>())));
        weight_ptrs.push_back(static_cast<int64_t>(
            reinterpret_cast<uintptr_t>(
                weight.sub_scale_packed.data_ptr<uint8_t>())));
        weight_ptrs.push_back(static_cast<int64_t>(
            reinterpret_cast<uintptr_t>(
                weight.neuron_scale.data_ptr<float>())));
        weight_ptrs.push_back(static_cast<int64_t>(
            reinterpret_cast<uintptr_t>(
                weight.codebook.data_ptr<int8_t>())));
        weight_sizes.push_back(weight.indices_packed.numel());
        weight_sizes.push_back(weight.aux_packed.numel());
        weight_sizes.push_back(weight.sub_scale_packed.numel());
        const int format = static_cast<int>(weight.kernel_format);
        const auto pool_format_group = mixed_nvq_f16_format_group(format);
        if (first_nvq_format) {
            f16_format_group = pool_format_group;
            first_nvq_format = false;
        } else if (f16_format_group != pool_format_group) {
            f16_format_group = MixedNvqF16FormatGroup::All;
        }
        const bool d4 =
            format == 3 || format == 10 || format == 11 ||
            format == 12 || format == 15 || format == 17;
        const int nvec =
            (runtime.neuron_len + (d4 ? 3 : 7)) / (d4 ? 4 : 8);
        const int nsign = (runtime.neuron_len + 7) / 8;
        pool_params.push_back(pool.local_experts);
        pool_params.push_back(static_cast<int32_t>(weight.ng));
        pool_params.push_back(nvec);
        pool_params.push_back(nsign);
        pool_params.push_back(static_cast<int32_t>(weight.sub_bits));
        pool_params.push_back(static_cast<int32_t>(weight.sign_mode));
        pool_params.push_back(format);

        auto local_host = pool.expert_local
            .to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt32)
            .contiguous();
        const int32_t * local = local_host.data_ptr<int32_t>();
        for (int expert = 0; expert < runtime.n_experts; ++expert) {
            if (local[expert] < 0) continue;
            if (local[expert] >= pool.local_experts ||
                    expert_pool[static_cast<size_t>(expert)] >= 0) {
                throw std::runtime_error(
                    "mixed NVQ prefill has invalid expert ownership");
            }
            expert_pool[static_cast<size_t>(expert)] = dispatch_pool;
            expert_local[static_cast<size_t>(expert)] = local[expert];
            ++owned_experts;
        }
        ++dispatch_pool;
    }

    auto dispatch = std::make_shared<MixedNvqDispatch>();
    dispatch->pool_count = dispatch_pool;
    dispatch->f16_format_group = f16_format_group;
    dispatch->masked_experts = owned_experts < runtime.n_experts;
    dispatch->weight_ptrs = mfq_tensor_backend::from_blob(
        weight_ptrs.data(), {dispatch_pool, 5},
        mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64))
        .clone().to(target).contiguous();
    dispatch->weight_sizes = mfq_tensor_backend::from_blob(
        weight_sizes.data(), {dispatch_pool, 3},
        mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64))
        .clone().to(target).contiguous();
    dispatch->pool_params = mfq_tensor_backend::from_blob(
        pool_params.data(), {dispatch_pool, 7},
        mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt32))
        .clone().to(target).contiguous();
    dispatch->expert_pool = mfq_tensor_backend::from_blob(
        expert_pool.data(), {runtime.n_experts},
        mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt32))
        .clone().to(target).contiguous();
    dispatch->expert_local = mfq_tensor_backend::from_blob(
        expert_local.data(), {runtime.n_experts},
        mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt32))
        .clone().to(target).contiguous();
    runtime.nvq_dispatch = std::move(dispatch);
}

static int64_t tensor_storage_bytes(const mfq_tensor_backend::Tensor & value) {
    return value.defined()
        ? value.numel() * (int64_t)value.element_size()
        : 0;
}

static int64_t mixed_moe_storage_bytes(const MixedMoeRuntime & runtime) {
    int64_t bytes = 0;
    if (runtime.nvq_dispatch) {
        bytes += tensor_storage_bytes(runtime.nvq_dispatch->weight_ptrs);
        bytes += tensor_storage_bytes(runtime.nvq_dispatch->weight_sizes);
        bytes += tensor_storage_bytes(runtime.nvq_dispatch->pool_params);
        bytes += tensor_storage_bytes(runtime.nvq_dispatch->expert_pool);
        bytes += tensor_storage_bytes(runtime.nvq_dispatch->expert_local);
    }
    for (const auto & pool : runtime.pools) {
        bytes += tensor_storage_bytes(pool.expert_local);
        if (pool.family == MixedMoeFamily::Nint) {
            bytes += tensor_storage_bytes(pool.nint.q_packed);
            bytes += tensor_storage_bytes(pool.nint.row_q_bits);
            bytes += tensor_storage_bytes(pool.nint.row_q_bit_offsets);
            bytes += tensor_storage_bytes(pool.nint.sub_scale);
            bytes += tensor_storage_bytes(pool.nint.sub_min);
            bytes += tensor_storage_bytes(pool.nint.neuron_scale);
            bytes += tensor_storage_bytes(pool.nint.neuron_min);
        } else if (pool.family == MixedMoeFamily::Nint8Zero) {
            bytes += tensor_storage_bytes(pool.q8_zero.q_packed);
            bytes += tensor_storage_bytes(pool.q8_zero.q8_zero_scale);
        } else if (pool.family == MixedMoeFamily::Mxfp4) {
            bytes += tensor_storage_bytes(pool.mxfp4.values);
            bytes += tensor_storage_bytes(pool.mxfp4.scales);
        } else if (pool.family == MixedMoeFamily::Mxfp4Sq) {
            bytes += tensor_storage_bytes(pool.mxfp4_sq.blob);
            bytes += tensor_storage_bytes(pool.mxfp4_sq.row_q);
            bytes += tensor_storage_bytes(
                pool.mxfp4_sq.row_symbol_byte_offsets);
            bytes += tensor_storage_bytes(pool.mxfp4_sq.row_auxiliary);
        } else if (pool.family == MixedMoeFamily::Fp8Sq) {
            bytes += tensor_storage_bytes(pool.fp8_sq.blob);
            bytes += tensor_storage_bytes(pool.fp8_sq.row_q);
            bytes += tensor_storage_bytes(
                pool.fp8_sq.row_symbol_byte_offsets);
        } else if (pool.family == MixedMoeFamily::Tpq) {
            bytes += tensor_storage_bytes(pool.tpq.packed);
            bytes += tensor_storage_bytes(pool.tpq.codebook);
        } else if (pool.family == MixedMoeFamily::Nvq) {
            bytes += tensor_storage_bytes(pool.nvq.indices_packed);
            bytes += tensor_storage_bytes(pool.nvq.aux_packed);
            bytes += tensor_storage_bytes(pool.nvq.sub_scale_packed);
            bytes += tensor_storage_bytes(pool.nvq.neuron_scale);
            bytes += tensor_storage_bytes(pool.nvq.codebook);
        } else {
            bytes += tensor_storage_bytes(pool.nepq.indices_packed);
            bytes += tensor_storage_bytes(pool.nepq.aux_packed);
            bytes += tensor_storage_bytes(pool.nepq.state_packed);
            bytes += tensor_storage_bytes(pool.nepq.neuron_scale);
            bytes += tensor_storage_bytes(pool.nepq.table_pool);
            bytes += tensor_storage_bytes(pool.nepq.grouped_table_pool);
            bytes += tensor_storage_bytes(pool.nepq.bank_ids);
            bytes += tensor_storage_bytes(pool.nepq.rotation_signs);
            bytes += tensor_storage_bytes(pool.nepq.residual_codebook);
            bytes += tensor_storage_bytes(pool.nepq.residual_first);
            bytes += tensor_storage_bytes(pool.nepq.residual_second);
        }
    }
    return bytes;
}

static std::shared_ptr<MixedMoeRuntime> make_mixed_moe_runtime(
        const MfeCpu & cpu, bool cuda) {
    auto runtime = std::make_shared<MixedMoeRuntime>();
    runtime->n_experts = cpu.n_experts;
    runtime->out_per_expert = cpu.out_per_expert;
    runtime->neuron_len = cpu.neuron_len;
    runtime->pools.reserve(cpu.pools.size());
    for (const auto & source : cpu.pools) {
        MixedMoePool pool;
        pool.local_experts = (int)source.expert_ids.size();
        std::vector<int32_t> local((size_t)cpu.n_experts, -1);
        for (int index = 0; index < pool.local_experts; ++index) {
            local[(size_t)source.expert_ids[(size_t)index]] = index;
        }
        pool.expert_local = mfq_tensor_backend::from_blob(
            local.data(), {(int64_t)local.size()},
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt32))
            .clone();
        if (cuda) {
            const auto target = mfq_tensor_backend::Device(
                mfq_tensor_backend::kCUDA,
                mfq_current_cuda_device());
            pool.expert_local =
                pool.expert_local.to(target).contiguous();
        }
        const int expected_rows = pool.local_experts * cpu.out_per_expert;
        if (source.dtype == "NINT8-0") {
            pool.family = MixedMoeFamily::Nint8Zero;
            pool.q8_zero = cuda
                ? to_gpu_nint8_zero(source.q8_zero)
                : to_cpu_nint8_zero(source.q8_zero);
            if (pool.q8_zero.out != expected_rows ||
                pool.q8_zero.neuron_len != cpu.neuron_len) {
                throw std::runtime_error(
                    "mixed NINT8-0 cohort shape mismatch");
            }
        } else if (source.dtype == "MXFP4") {
            pool.family = MixedMoeFamily::Mxfp4;
            pool.mxfp4 = to_device_mxfp4(source.mxfp4, cuda);
            if (pool.mxfp4.out != expected_rows ||
                    pool.mxfp4.neuron_len != cpu.neuron_len) {
                throw std::runtime_error(
                    "mixed MXFP4 cohort shape mismatch");
            }
        } else if (source.dtype == "MXFP4-SQ") {
            pool.family = MixedMoeFamily::Mxfp4Sq;
            pool.mxfp4_sq = to_device_mxfp4_sq(
                source.payload, cuda);
            if (pool.mxfp4_sq.out != expected_rows ||
                    pool.mxfp4_sq.neuron_len != cpu.neuron_len) {
                throw std::runtime_error(
                    "mixed MXFP4-SQ cohort shape mismatch");
            }
        } else if (mfq::fp8sq::is_dtype(source.dtype)) {
            pool.family = MixedMoeFamily::Fp8Sq;
            pool.fp8_sq = to_device_fp8_sq(
                source.dtype, source.payload, cuda);
            if (pool.fp8_sq.out != expected_rows ||
                    pool.fp8_sq.neuron_len != cpu.neuron_len) {
                throw std::runtime_error(
                    "mixed FP8-SQ cohort shape mismatch");
            }
        } else if (is_tpq_pq_dtype(source.dtype)) {
            pool.family = MixedMoeFamily::Tpq;
            pool.tpq = to_device_tpq(source.tpq, cuda);
            if (pool.tpq.int4 || pool.tpq.out != expected_rows ||
                    pool.tpq.neuron_len != cpu.neuron_len) {
                throw std::runtime_error(
                    "mixed TPQ-PQ cohort shape mismatch");
            }
        } else if (source.dtype == "NINT") {
            pool.family = MixedMoeFamily::Nint;
            pool.nint = to_device_mfe_nint(
                source.weight, pool.local_experts,
                cpu.out_per_expert, cuda);
            if (pool.nint.out != expected_rows ||
                pool.nint.neuron_len != cpu.neuron_len) {
                throw std::runtime_error("mixed NINT cohort shape mismatch");
            }
        } else if (source.dtype == "NEPQ") {
            pool.family = MixedMoeFamily::Nepq;
            auto parsed = unpack_nepq(
                source.payload, source.dtype, source.runtime_payload);
            if (parsed.n_experts != pool.local_experts ||
                parsed.out_per_expert != cpu.out_per_expert ||
                parsed.neuron_len != cpu.neuron_len) {
                throw std::runtime_error("mixed NEPQ cohort shape mismatch");
            }
            pool.nepq = cuda
                ? to_gpu_nepq(parsed)
                : to_cpu_nepq(parsed);
        } else {
            pool.family = MixedMoeFamily::Nvq;
            auto parsed = unpack_nvq(source.payload, source.dtype);
            if (parsed.out != expected_rows || parsed.neuron_len != cpu.neuron_len) {
                throw std::runtime_error("mixed NVQ/NPQ cohort shape mismatch");
            }
            pool.nvq = cuda
                ? to_gpu_nvq(parsed)
                : to_cpu_nvq(parsed);
        }
        runtime->pools.push_back(std::move(pool));
    }
    if (cuda) {
        initialize_mixed_nvq_dispatch(*runtime);
    }
    return runtime;
}

static MfeWeight wrap_mixed_moe_runtime(
        const std::shared_ptr<MixedMoeRuntime> & runtime) {
    MfeWeight result;
    result.unified_nint_projection = runtime->nint_only();
    result.n_experts = runtime->n_experts;
    result.out_per_expert = runtime->out_per_expert;
    result.neuron_len = runtime->neuron_len;
    result.partial_experts = runtime->partial_experts;
    result.mixed_weight_bytes = mixed_moe_storage_bytes(*runtime);
    result.activation_geometries = runtime->activation_geometry();
    result.activation_workspace_domain = 2;
    result.mixed_forward = [runtime](
            mfq_tensor_backend::Tensor x, const MoeRoutePlan & route) {
        return runtime->forward(x, route);
    };
    result.mixed_prequantized_forward = [runtime](
            mfq_tensor_backend::Tensor x, const MoeRoutePlan & route) {
        return runtime->forward(x, route, true);
    };
    result.mixed_glu_output_forward = [runtime](
            mfq_tensor_backend::Tensor x,
            const MoeRoutePlan & route,
            bool gelu) {
        return runtime->forward_glu_output(x, route, gelu);
    };
    if (runtime->supports_clamped_swiglu()) {
        result.mixed_clamped_swiglu_forward = [runtime](
                mfq_tensor_backend::Tensor gate_up,
                const MoeRoutePlan & route,
                double limit) {
            return runtime->forward_clamped_swiglu(gate_up, route, limit);
        };
    }
    return result;
}

static MfeWeight to_gpu_mixed_moe(const MfeCpu & cpu) {
    return wrap_mixed_moe_runtime(make_mixed_moe_runtime(cpu, true));
}

static MfeWeight to_cuda_device_moe_expert_slice(
        const MfeCpu & cpu,
        int64_t expert_begin,
        int64_t expert_end,
        int device) {
    if (expert_begin < 0 || expert_begin >= expert_end ||
            expert_end > cpu.n_experts) {
        throw std::runtime_error(
            "invalid expert-parallel MoE shard");
    }
    const bool all_nint = std::all_of(
        cpu.pools.begin(), cpu.pools.end(),
        [](const MfeCpuPool & pool) {
            return pool.dtype == "NINT";
        });
    MfqCudaGuard guard(device);
    if (all_nint) {
        MfeCpu sliced = cpu;
        sliced.pools.clear();
        for (const auto & source : cpu.pools) {
            MfeCpuPool destination = source;
            destination.expert_ids.clear();
            std::vector<int64_t> rows;
            for (size_t local = 0;
                 local < source.expert_ids.size(); ++local) {
                const int expert = source.expert_ids[local];
                if (expert < expert_begin || expert >= expert_end) continue;
                destination.expert_ids.push_back(expert);
                for (int row = 0; row < cpu.out_per_expert; ++row) {
                    rows.push_back(
                        static_cast<int64_t>(local) * cpu.out_per_expert + row);
                }
            }
            if (rows.empty()) continue;
            destination.weight = select_nint_cpu_rows(
                source.weight, rows);
            sliced.pools.push_back(std::move(destination));
        }
        auto result = to_gpu_mfe(sliced);
        result.partial_experts = true;
        return result;
    }

    auto runtime = std::make_shared<MixedMoeRuntime>();
    runtime->n_experts = cpu.n_experts;
    runtime->out_per_expert = cpu.out_per_expert;
    runtime->neuron_len = cpu.neuron_len;
    runtime->partial_experts = true;
    for (const auto & source : cpu.pools) {
        std::vector<int> expert_ids;
        std::vector<int64_t> rows;
        for (size_t local = 0;
             local < source.expert_ids.size(); ++local) {
            const int expert = source.expert_ids[local];
            if (expert < expert_begin || expert >= expert_end) continue;
            expert_ids.push_back(expert);
            for (int row = 0; row < cpu.out_per_expert; ++row) {
                rows.push_back(
                    static_cast<int64_t>(local) * cpu.out_per_expert + row);
            }
        }
        if (rows.empty()) continue;

        MixedMoePool pool;
        pool.local_experts = static_cast<int>(expert_ids.size());
        std::vector<int32_t> local_map(
            static_cast<size_t>(cpu.n_experts), -1);
        for (size_t local = 0; local < expert_ids.size(); ++local) {
            local_map[static_cast<size_t>(expert_ids[local])] =
                static_cast<int32_t>(local);
        }
        pool.expert_local = mfq_tensor_backend::from_blob(
            local_map.data(),
            {static_cast<int64_t>(local_map.size())},
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt32))
            .clone().to(mfq_tensor_backend::Device(
                mfq_tensor_backend::kCUDA,
                mfq_current_cuda_device())).contiguous();
        if (source.dtype == "NINT8-0") {
            pool.family = MixedMoeFamily::Nint8Zero;
            pool.q8_zero = to_gpu_nint8_zero(
                select_nint8_zero_cpu_rows(source.q8_zero, rows));
        } else if (source.dtype == "NINT") {
            pool.family = MixedMoeFamily::Nint;
            const auto selected = select_nint_cpu_rows(source.weight, rows);
            pool.nint = to_device_mfe_nint(
                selected, pool.local_experts,
                cpu.out_per_expert, true);
        } else if (source.dtype == "MXFP4") {
            pool.family = MixedMoeFamily::Mxfp4;
            pool.mxfp4 = to_device_mxfp4(
                select_mxfp4_cpu_rows(source.mxfp4, rows), true);
        } else if (source.dtype == "MXFP4-SQ") {
            pool.family = MixedMoeFamily::Mxfp4Sq;
            pool.mxfp4_sq = to_device_mxfp4_sq(
                mfq::sq::select_rows(source.payload, rows), true);
        } else if (mfq::fp8sq::is_dtype(source.dtype)) {
            if (rows.size() != source.expert_ids.size() *
                    static_cast<size_t>(cpu.out_per_expert)) {
                throw std::runtime_error(
                    "FP8-SQ MFE pools must remain matrix-local across devices");
            }
            pool.family = MixedMoeFamily::Fp8Sq;
            pool.fp8_sq = to_device_fp8_sq(
                source.dtype, source.payload, true);
        } else if (is_tpq_pq_dtype(source.dtype)) {
            pool.family = MixedMoeFamily::Tpq;
            pool.tpq = to_device_tpq(
                select_tpq_cpu_rows(source.tpq, rows), true);
        } else if (source.dtype == "NEPQ") {
            pool.family = MixedMoeFamily::Nepq;
            auto parsed = unpack_nepq(
                source.payload, source.dtype, source.runtime_payload);
            pool.nepq = to_gpu_nepq(select_nepq_cpu_rows(
                parsed, rows, cpu.out_per_expert,
                pool.local_experts));
        } else {
            pool.family = MixedMoeFamily::Nvq;
            auto parsed = unpack_nvq(source.payload, source.dtype);
            pool.nvq = to_gpu_nvq(
                select_nvq_cpu_rows(parsed, rows));
        }
        runtime->pools.push_back(std::move(pool));
    }
    if (runtime->pools.empty()) {
        throw std::runtime_error(
            "expert-parallel MoE shard has no owned experts");
    }
    initialize_mixed_nvq_dispatch(*runtime);
    return wrap_mixed_moe_runtime(runtime);
}

static mfq_tensor_backend::Tensor copy_cpu_weight_to_cuda(
        const mfq_tensor_backend::Tensor & source) {
    return source.defined()
        ? source.to(mfq_tensor_backend::Device(
            mfq_tensor_backend::kCUDA,
            mfq_current_cuda_device())).contiguous()
        : mfq_tensor_backend::Tensor();
}

static NintWeight copy_cpu_nint_to_cuda(const NintWeight & source) {
    NintWeight result = source;
    result.workspaces.clear();
    result.q_packed = copy_cpu_weight_to_cuda(source.q_packed);
    result.row_q_bits = copy_cpu_weight_to_cuda(source.row_q_bits);
    result.row_q_bit_offsets =
        copy_cpu_weight_to_cuda(source.row_q_bit_offsets);
    result.q8_zero_scale =
        copy_cpu_weight_to_cuda(source.q8_zero_scale);
    result.sub_scale = copy_cpu_weight_to_cuda(source.sub_scale);
    result.sub_min = copy_cpu_weight_to_cuda(source.sub_min);
    result.neuron_scale =
        copy_cpu_weight_to_cuda(source.neuron_scale);
    result.neuron_min = copy_cpu_weight_to_cuda(source.neuron_min);
    return result;
}

static NvqWeight copy_cpu_nvq_to_cuda(const NvqWeight & source) {
    NvqWeight result = source;
    result.workspaces.clear();
    result.indices_packed =
        copy_cpu_weight_to_cuda(source.indices_packed);
    result.aux_packed = copy_cpu_weight_to_cuda(source.aux_packed);
    result.sub_scale_packed =
        copy_cpu_weight_to_cuda(source.sub_scale_packed);
    result.neuron_scale =
        copy_cpu_weight_to_cuda(source.neuron_scale);
    result.codebook = copy_cpu_weight_to_cuda(source.codebook);
    return result;
}

static Mxfp4Weight copy_cpu_mxfp4_to_cuda(const Mxfp4Weight & source) {
    Mxfp4Weight result = source;
    result.values = copy_cpu_weight_to_cuda(source.values);
    result.scales = copy_cpu_weight_to_cuda(source.scales);
    return result;
}

static Mxfp4SqWeight copy_cpu_mxfp4_sq_to_cuda(
        const Mxfp4SqWeight & source) {
    Mxfp4SqWeight result = source;
    result.blob = copy_cpu_weight_to_cuda(source.blob);
    result.row_q = copy_cpu_weight_to_cuda(source.row_q);
    result.row_symbol_byte_offsets =
        copy_cpu_weight_to_cuda(source.row_symbol_byte_offsets);
    result.row_auxiliary =
        copy_cpu_weight_to_cuda(source.row_auxiliary);
    return result;
}

static Fp8SqWeight copy_cpu_fp8_sq_to_cuda(
        const Fp8SqWeight & source) {
    Fp8SqWeight result = source;
    result.blob = copy_cpu_weight_to_cuda(source.blob);
    result.row_q = copy_cpu_weight_to_cuda(source.row_q);
    result.row_symbol_byte_offsets =
        copy_cpu_weight_to_cuda(source.row_symbol_byte_offsets);
    return result;
}

static TpqWeight copy_cpu_tpq_to_cuda(const TpqWeight & source) {
    TpqWeight result = source;
    result.packed = copy_cpu_weight_to_cuda(source.packed);
    result.scales = copy_cpu_weight_to_cuda(source.scales);
    result.codebook = copy_cpu_weight_to_cuda(source.codebook);
    return result;
}

static NepqWeight copy_cpu_nepq_to_cuda(const NepqWeight & source) {
    NepqWeight result = source;
    result.indices_packed =
        copy_cpu_weight_to_cuda(source.indices_packed);
    result.aux_packed = copy_cpu_weight_to_cuda(source.aux_packed);
    result.state_packed =
        copy_cpu_weight_to_cuda(source.state_packed);
    result.neuron_scale =
        copy_cpu_weight_to_cuda(source.neuron_scale);
    result.table_pool = copy_cpu_weight_to_cuda(source.table_pool);
    result.grouped_table_pool =
        copy_cpu_weight_to_cuda(source.grouped_table_pool);
    result.bank_ids = copy_cpu_weight_to_cuda(source.bank_ids);
    result.rotation_signs =
        copy_cpu_weight_to_cuda(source.rotation_signs);
    if (source.residual) {
        result.residual_codebook =
            copy_cpu_weight_to_cuda(source.residual_codebook);
        result.residual_first =
            copy_cpu_weight_to_cuda(source.residual_first);
        result.residual_second =
            copy_cpu_weight_to_cuda(source.residual_second);
    }
    return result;
}

MfeWeight stage_cpu_mixed_moe(
        const std::shared_ptr<MixedMoeRuntime> & cpu) {
    if (!cpu) throw std::runtime_error("missing CPU-offloaded MoE state");
    auto runtime = std::make_shared<MixedMoeRuntime>();
    runtime->n_experts = cpu->n_experts;
    runtime->out_per_expert = cpu->out_per_expert;
    runtime->neuron_len = cpu->neuron_len;
    runtime->pools.reserve(cpu->pools.size());
    for (const auto & source : cpu->pools) {
        MixedMoePool pool;
        pool.family = source.family;
        pool.local_experts = source.local_experts;
        pool.expert_local =
            copy_cpu_weight_to_cuda(source.expert_local);
        if (pool.family == MixedMoeFamily::Nint) {
            pool.nint = copy_cpu_nint_to_cuda(source.nint);
        } else if (pool.family == MixedMoeFamily::Nint8Zero) {
            pool.q8_zero = copy_cpu_nint_to_cuda(source.q8_zero);
        } else if (pool.family == MixedMoeFamily::Mxfp4) {
            pool.mxfp4 = copy_cpu_mxfp4_to_cuda(source.mxfp4);
        } else if (pool.family == MixedMoeFamily::Mxfp4Sq) {
            pool.mxfp4_sq = copy_cpu_mxfp4_sq_to_cuda(
                source.mxfp4_sq);
        } else if (pool.family == MixedMoeFamily::Fp8Sq) {
            pool.fp8_sq = copy_cpu_fp8_sq_to_cuda(source.fp8_sq);
        } else if (pool.family == MixedMoeFamily::Tpq) {
            pool.tpq = copy_cpu_tpq_to_cuda(source.tpq);
        } else if (pool.family == MixedMoeFamily::Nvq) {
            pool.nvq = copy_cpu_nvq_to_cuda(source.nvq);
        } else {
            pool.nepq = copy_cpu_nepq_to_cuda(source.nepq);
        }
        runtime->pools.push_back(std::move(pool));
    }
    initialize_mixed_nvq_dispatch(*runtime);
    return wrap_mixed_moe_runtime(runtime);
}

MfeWeight cpu_mixed_moe_metadata(
        const std::shared_ptr<MixedMoeRuntime> & runtime) {
    MfeWeight result;
    result.unified_nint_projection = runtime->nint_only();
    result.n_experts = runtime->n_experts;
    result.out_per_expert = runtime->out_per_expert;
    result.neuron_len = runtime->neuron_len;
    result.mixed_weight_bytes = mixed_moe_storage_bytes(*runtime);
    result.activation_geometries = runtime->activation_geometry();
    result.activation_workspace_domain = 2;
    return result;
}

static MfeCpu load_mfe_cpu(
        const mfq::ModelSource & mfq, const std::string & name) {
    if (require_tensor(mfq, name).dtype != "MFE") {
        throw std::runtime_error("expert tensor must use MFE: " + name);
    }
    return unpack_mfe(read_tensor(mfq, name));
}

static std::shared_ptr<MixedMoeRuntime> make_mxfp4_range_runtime(
        const mfq::cuda::MfeMxfp4ExpertStore & store) {
    auto runtime = std::make_shared<MixedMoeRuntime>();
    runtime->n_experts = store.num_experts();
    runtime->out_per_expert = store.out_per_expert();
    runtime->neuron_len = store.neuron_len();
    MixedMoePool pool;
    pool.family = MixedMoeFamily::Mxfp4;
    pool.local_experts = store.num_experts();
    pool.mxfp4.out =
        static_cast<int64_t>(store.num_experts()) * store.out_per_expert();
    pool.mxfp4.neuron_len = store.neuron_len();
    std::vector<int32_t> local(static_cast<size_t>(store.num_experts()));
    std::iota(local.begin(), local.end(), int32_t{0});
    pool.expert_local = mfq_tensor_backend::from_blob(
        local.data(),
        {static_cast<int64_t>(local.size())},
        mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt32))
        .clone();
    runtime->pools.push_back(std::move(pool));
    return runtime;
}

struct MoeCacheTransfer {
    const uint8_t * source = nullptr;
    uint8_t * destination = nullptr;
    int64_t nbytes = 0;
    bool packed_weight = false;
    const uint8_t * mapped_source = nullptr;
    const mfq::cuda::MfeMxfp4ExpertStore * range_store = nullptr;
    const mfq::cuda::MfeMxfp4ExpertPart * range_part = nullptr;
};

struct MoeCacheNewLease {
    mfq::MoeCacheSlotBook * book = nullptr;
    mfq::MoeCacheKey key;
    int slot = -1;
    uint64_t generation = 0;
};

struct MoeCachedCohort;
class MoeCachedSource;

struct MoeCacheFieldLayout {
    mfq_tensor_backend::ScalarType scalar_type =
        mfq_tensor_backend::kUInt8;
    std::vector<int64_t> slot_shape;
    int64_t elements = 0;
    int64_t element_size = 0;
};

static int64_t tensor_nbytes(const mfq_tensor_backend::Tensor & value) {
    return value.defined()
        ? value.numel() * static_cast<int64_t>(value.element_size())
        : 0;
}

static std::vector<mfq_tensor_backend::Tensor> moe_cache_fields(
        const MixedMoePool & pool) {
    if (pool.family == MixedMoeFamily::Nint) {
        return {
            pool.nint.q_packed,
            pool.nint.row_q_bits,
            pool.nint.row_q_bit_offsets,
            pool.nint.sub_scale,
            pool.nint.sub_min,
            pool.nint.neuron_scale,
            pool.nint.neuron_min,
        };
    }
    if (pool.family == MixedMoeFamily::Nint8Zero) {
        return {
            pool.q8_zero.q_packed,
            pool.q8_zero.q8_zero_scale,
        };
    }
    if (pool.family == MixedMoeFamily::Mxfp4) {
        return {pool.mxfp4.values, pool.mxfp4.scales};
    }
    if (pool.family == MixedMoeFamily::Tpq) {
        return {pool.tpq.packed};
    }
    if (pool.family == MixedMoeFamily::Nvq) {
        return {
            pool.nvq.indices_packed,
            pool.nvq.aux_packed,
            pool.nvq.sub_scale_packed,
            pool.nvq.neuron_scale,
        };
    }
    std::vector<mfq_tensor_backend::Tensor> fields{
        pool.nepq.indices_packed,
        pool.nepq.aux_packed,
        pool.nepq.state_packed,
        pool.nepq.neuron_scale,
        pool.nepq.bank_ids,
    };
    if (pool.nepq.residual) {
        fields.push_back(pool.nepq.residual_first);
        fields.push_back(pool.nepq.residual_second);
    }
    return fields;
}

static std::vector<MoeCacheFieldLayout> moe_cache_field_layouts(
        const MixedMoePool & pool) {
    const auto fields = moe_cache_fields(pool);
    std::vector<MoeCacheFieldLayout> result;
    result.reserve(fields.size());
    for (const auto & field : fields) {
        if (!field.defined() || !field.is_cpu() || !field.is_contiguous() ||
                field.dim() < 1 || pool.local_experts <= 0 ||
                field.size(0) % pool.local_experts != 0 ||
                field.numel() % pool.local_experts != 0) {
            throw std::runtime_error(
                "MoE cache source fields must be contiguous, expert-major CPU tensors");
        }
        auto shape = field.sizes().vec();
        shape[0] /= pool.local_experts;
        result.push_back({
            field.scalar_type(),
            std::move(shape),
            field.numel() / pool.local_experts,
            static_cast<int64_t>(field.element_size()),
        });
    }
    return result;
}

static void validate_nepq_expert_boundaries(
        const MixedMoePool & pool,
        int out_per_expert,
        int neuron_len) {
    if (pool.family != MixedMoeFamily::Nepq) return;
    const int index_bits =
        pool.nepq.format == 9 ? 6 :
        pool.nepq.format == 7 ? 7 :
        pool.nepq.format == 8 ? 9 :
        pool.nepq.format == 1 ? 11 : 0;
    const int aux_bits =
        pool.nepq.format == 8 || pool.nepq.format == 1 ? 1 : 0;
    const int nvec = neuron_len / 8;
    const int ng = (neuron_len + 23) / 24;
    const int nsuper = (ng + 3) / 4;
    const int rows = pool.local_experts * out_per_expert;
    const std::vector<int64_t> bits{
        static_cast<int64_t>(out_per_expert) * nvec * index_bits,
        static_cast<int64_t>(out_per_expert) * ng * aux_bits,
        static_cast<int64_t>(out_per_expert) * ng *
            pool.nepq.state_bits,
        static_cast<int64_t>(out_per_expert) * 32,
        static_cast<int64_t>(out_per_expert) * nsuper * 8,
    };
    if (index_bits == 0 ||
        std::any_of(bits.begin(), bits.end(), [](int64_t value) {
            return value % 8 != 0;
        })) {
        throw std::runtime_error(
            "NEPQ expert payload is not byte aligned for GPU caching");
    }
    if (pool.nepq.residual) {
        const int blocks =
            (nvec + pool.nepq.residual_block_vectors - 1) /
            pool.nepq.residual_block_vectors;
        if (!pool.nepq.residual_codebook.defined() ||
                !pool.nepq.residual_codebook.is_cpu() ||
                !pool.nepq.residual_codebook.is_contiguous() ||
                pool.nepq.residual_codebook.scalar_type() != mfq_tensor_backend::kFloat16 ||
                pool.nepq.residual_codebook.dim() != 2 ||
                pool.nepq.residual_codebook.size(0) != 1024 ||
                pool.nepq.residual_codebook.size(1) != 8 ||
                !pool.nepq.residual_first.defined() ||
                !pool.nepq.residual_first.is_cpu() ||
                !pool.nepq.residual_first.is_contiguous() ||
                pool.nepq.residual_first.scalar_type() != mfq_tensor_backend::kInt16 ||
                pool.nepq.residual_first.dim() != 2 ||
                pool.nepq.residual_first.size(0) != rows ||
                pool.nepq.residual_first.size(1) != blocks ||
                !pool.nepq.residual_second.defined() ||
                !pool.nepq.residual_second.is_cpu() ||
                !pool.nepq.residual_second.is_contiguous() ||
                pool.nepq.residual_second.scalar_type() != mfq_tensor_backend::kInt16 ||
                pool.nepq.residual_second.dim() != 2 ||
                pool.nepq.residual_second.size(0) != rows ||
                pool.nepq.residual_second.size(1) != blocks) {
            throw std::runtime_error(
                "NEPQ-A residual fields are incompatible with GPU caching");
        }
    }
}

static void validate_tpq_expert_boundaries(
        const MixedMoePool & pool,
        int out_per_expert,
        int neuron_len) {
    if (pool.family != MixedMoeFamily::Tpq) return;
    if (pool.tpq.vector_size <= 0) {
        throw std::runtime_error("TPQ-PQ cache vector size must be positive");
    }
    const int64_t vectors = neuron_len / pool.tpq.vector_size;
    const int64_t bits_per_expert =
        static_cast<int64_t>(out_per_expert) * vectors *
        pool.tpq.index_bits;
    if (pool.tpq.int4 ||
            neuron_len % pool.tpq.vector_size != 0 ||
            pool.tpq.index_bits < 8 || pool.tpq.index_bits > 16 ||
            bits_per_expert % 8 != 0 ||
            !pool.tpq.packed.defined() || !pool.tpq.packed.is_cpu() ||
            !pool.tpq.packed.is_contiguous() ||
            pool.tpq.packed.scalar_type() != mfq_tensor_backend::kUInt8 ||
            pool.tpq.packed.numel() !=
                pool.local_experts * bits_per_expert / 8 ||
            !pool.tpq.codebook.defined() || !pool.tpq.codebook.is_cpu() ||
            !pool.tpq.codebook.is_contiguous() ||
            pool.tpq.codebook.scalar_type() != mfq_tensor_backend::kFloat32 ||
            pool.tpq.codebook.dim() != 2 ||
            pool.tpq.codebook.size(0) <= 1 ||
            pool.tpq.codebook.size(1) != pool.tpq.vector_size) {
        throw std::runtime_error(
            "TPQ-PQ expert fields are incompatible with GPU caching");
    }
}

static std::string moe_cache_signature(
        const MixedMoePool & pool,
        int out_per_expert,
        int neuron_len,
        const std::vector<MoeCacheFieldLayout> & layouts) {
    std::ostringstream stream;
    stream << static_cast<int>(pool.family)
           << ":o" << out_per_expert
           << ":k" << neuron_len;
    if (pool.family == MixedMoeFamily::Nint) {
        stream << ":b" << pool.nint.bits
               << ":g" << pool.nint.gs
               << ":n" << pool.nint.ng
               << ":s" << pool.nint.q_expert_stride
               << ":v2";
    } else if (pool.family == MixedMoeFamily::Nint8Zero) {
        stream << ":g32:n" << pool.q8_zero.ng;
    } else if (pool.family == MixedMoeFamily::Mxfp4) {
        stream << ":mx4";
    } else if (pool.family == MixedMoeFamily::Tpq) {
        stream << ":tpq:v" << pool.tpq.vector_size
               << ":b" << pool.tpq.index_bits;
    } else if (pool.family == MixedMoeFamily::Nvq) {
        stream << ":f" << pool.nvq.format
               << ":kf" << pool.nvq.kernel_format
               << ":s" << pool.nvq.sub_bits
               << ":g" << pool.nvq.gs
               << ":n" << pool.nvq.ng
               << ":sm" << pool.nvq.sign_mode;
    } else {
        stream << ":f" << pool.nepq.format
               << ":s" << pool.nepq.state_bits
               << ":g" << pool.nepq.ng
               << ":r" << pool.nepq.rotation_block
               << ":a" << (pool.nepq.residual ? 1 : 0);
        if (pool.nepq.residual) {
            stream << ":p" << pool.nepq.residual_position_bits
                   << ":v" << pool.nepq.residual_block_vectors;
        }
    }
    for (const auto & layout : layouts) {
        stream << ":" << static_cast<int>(layout.scalar_type)
               << "x" << layout.elements;
    }
    return stream.str();
}

struct MoeGpuArena {
    std::string signature;
    int64_t slot_bytes = 0;
    int minimum_slots = 0;
    int registered_experts = 0;
    int slots = 0;
    std::vector<MoeCacheFieldLayout> layouts;
    std::vector<mfq_tensor_backend::Tensor> fields;
    std::unique_ptr<mfq::MoeCacheSlotBook> book;
};

struct MoePinnedStage {
    mfq_tensor_backend::Tensor host;
    mfq_tensor_backend::Tensor device;
    cudaEvent_t done = nullptr;
    bool pending = false;
};

struct MoePendingRangeRead {
    mfq_tensor_backend::Tensor host;
    std::vector<MoeCacheTransfer> transfers;
    std::vector<std::pair<mfq::MoeCacheSlotBook *, int>> held_slots;
    std::vector<MoeCacheNewLease> new_leases;
    mfq::cuda::MfeMxfp4ReadTicket ticket;
    bool replaced_occupied = false;
};

struct MoeCacheStats {
    int64_t demand_hits = 0;
    int64_t demand_misses = 0;
    int64_t prefetch_hits = 0;
    int64_t prefetch_misses = 0;
    int64_t evictions = 0;
    int64_t h2d_bytes = 0;
    int64_t route_d2h_bytes = 0;
    int64_t full_projection_fallbacks = 0;
    int64_t h2d_submissions = 0;
    int64_t h2d_descriptors = 0;
    int64_t mapped_gather_bytes = 0;
    int64_t mapped_gather_submissions = 0;
    int64_t mapped_gather_descriptors = 0;
    int64_t range_read_bytes = 0;
    int64_t range_read_calls = 0;
    int64_t range_file_opens = 0;
    int64_t range_read_nanoseconds = 0;
    int64_t range_overlap_batches = 0;
    int64_t range_overlap_wait_nanoseconds = 0;
};

class MoeExpertCache : public std::enable_shared_from_this<MoeExpertCache> {
public:
    explicit MoeExpertCache(int64_t budget_bytes)
        : budget_bytes_(budget_bytes) {
        if (budget_bytes_ <= 0) {
            throw std::invalid_argument(
                "MoE GPU cache budget must be positive");
        }
        MFQ_CUDA_CHECK(cudaStreamCreateWithFlags(
            &weight_stream_, cudaStreamNonBlocking));
        MFQ_CUDA_CHECK(cudaStreamCreateWithFlags(
            &route_stream_, cudaStreamNonBlocking));
        MFQ_CUDA_CHECK(cudaEventCreateWithFlags(
            &compute_done_, cudaEventDisableTiming));
        MFQ_CUDA_CHECK(cudaEventCreateWithFlags(
            &transfer_ready_, cudaEventDisableTiming));
        MFQ_CUDA_CHECK(cudaEventCreateWithFlags(
            &route_input_ready_, cudaEventDisableTiming));
        MFQ_CUDA_CHECK(cudaEventCreateWithFlags(
            &route_done_, cudaEventDisableTiming));
        stages_.resize(4);
        for (auto & stage : stages_) {
            MFQ_CUDA_CHECK(cudaEventCreateWithFlags(
                &stage.done, cudaEventDisableTiming));
        }
        const char * mapped = std::getenv("MFQ_MOE_MAPPED_GATHER");
        mapped_gather_enabled_ =
            mapped != nullptr && std::atoi(mapped) != 0;
        const char * blocks =
            std::getenv("MFQ_MOE_MAPPED_COPY_BLOCKS");
        if (blocks != nullptr) {
            mapped_copy_blocks_ = std::max(
                4, std::min(128, std::atoi(blocks)));
        }
        int range_workers = 8;
        const char * workers =
            std::getenv("MFQ_MOE_SSD_IO_WORKERS");
        if (workers != nullptr) {
            range_workers = std::max(
                1, std::min(64, std::atoi(workers)));
        }
        range_read_pool_ =
            std::make_unique<mfq::cuda::MfeMxfp4ReadPool>(range_workers);
        const char * disable_overlap =
            std::getenv("MFQ_DISABLE_MOE_SSD_OVERLAP");
        range_overlap_enabled_ =
            disable_overlap == nullptr || std::atoi(disable_overlap) == 0;
    }

    ~MoeExpertCache() {
        if (!registered_host_fields_.empty() &&
                weight_stream_ != nullptr) {
            (void)cudaStreamSynchronize(weight_stream_);
        }
        for (auto it = registered_host_fields_.rbegin();
             it != registered_host_fields_.rend();
             ++it) {
            if (it->owned) {
                (void)cudaHostUnregister(it->host);
            }
        }
        for (auto & stage : stages_) {
            if (stage.done != nullptr) cudaEventDestroy(stage.done);
        }
        if (compute_done_ != nullptr) cudaEventDestroy(compute_done_);
        if (transfer_ready_ != nullptr) {
            cudaEventDestroy(transfer_ready_);
        }
        if (route_input_ready_ != nullptr) {
            cudaEventDestroy(route_input_ready_);
        }
        if (route_done_ != nullptr) {
            cudaEventDestroy(route_done_);
        }
        if (route_stream_ != nullptr) cudaStreamDestroy(route_stream_);
        if (weight_stream_ != nullptr) cudaStreamDestroy(weight_stream_);
    }

    std::shared_ptr<MoeCachedSource> register_source(
        const std::string & name,
        std::shared_ptr<MixedMoeRuntime> cpu,
        int minimum_slots,
        int layer_id,
        std::string projection_role);

    std::shared_ptr<MoeCachedSource> register_range_source(
        const std::string & name,
        std::shared_ptr<MixedMoeRuntime> metadata,
        std::shared_ptr<mfq::cuda::MfeMxfp4ExpertStore> store,
        int minimum_slots,
        int layer_id,
        std::string projection_role);

    MoeGpuArena * register_cohort(
            const MixedMoePool & pool,
            int out_per_expert,
            int neuron_len,
            int minimum_slots) {
        if (finalized_) {
            throw std::runtime_error(
                "cannot register a MoE source after cache finalization");
        }
        validate_nepq_expert_boundaries(
            pool, out_per_expert, neuron_len);
        validate_tpq_expert_boundaries(
            pool, out_per_expert, neuron_len);
        return register_cohort_layout(
            pool,
            out_per_expert,
            neuron_len,
            minimum_slots,
            pool.local_experts,
            moe_cache_field_layouts(pool));
    }

    MoeGpuArena * register_cohort_layout(
            const MixedMoePool & pool,
            int out_per_expert,
            int neuron_len,
            int minimum_slots,
            int registered_experts,
            std::vector<MoeCacheFieldLayout> layouts) {
        if (finalized_) {
            throw std::runtime_error(
                "cannot register a MoE source after cache finalization");
        }
        if (registered_experts <= 0 || layouts.empty()) {
            throw std::runtime_error("invalid MoE cache cohort layout");
        }
        const std::string signature =
            moe_cache_signature(pool, out_per_expert, neuron_len, layouts);
        auto found = arenas_.find(signature);
        if (found == arenas_.end()) {
            auto arena = std::make_unique<MoeGpuArena>();
            arena->signature = signature;
            arena->minimum_slots =
                std::min(minimum_slots, registered_experts);
            arena->registered_experts = registered_experts;
            for (const auto & layout : layouts) {
                if (layout.slot_shape.empty() || layout.elements < 0 ||
                        layout.element_size <= 0) {
                    throw std::runtime_error(
                        "invalid MoE cache field layout");
                }
                arena->slot_bytes +=
                    layout.elements * layout.element_size;
            }
            arena->layouts = std::move(layouts);
            MoeGpuArena * result = arena.get();
            arenas_.emplace(signature, std::move(arena));
            return result;
        }
        MoeGpuArena * arena = found->second.get();
        arena->minimum_slots = std::max(
            arena->minimum_slots,
            std::min(minimum_slots, registered_experts));
        arena->registered_experts += registered_experts;
        if (arena->layouts.size() != layouts.size()) {
            throw std::runtime_error(
                "MoE cache signature merged incompatible field counts");
        }
        for (size_t index = 0; index < layouts.size(); ++index) {
            const auto & left = arena->layouts[index];
            const auto & right = layouts[index];
            if (left.scalar_type != right.scalar_type ||
                    left.slot_shape != right.slot_shape ||
                    left.elements != right.elements ||
                    left.element_size != right.element_size) {
                throw std::runtime_error(
                    "MoE cache signature merged incompatible field layouts");
            }
        }
        return arena;
    }

    void finalize();

    void set_profile(mfq::MoeCacheProfile profile) {
        if (finalized_ || !sources_.empty()) {
            throw std::runtime_error(
                "MoE cache profile must be set before source registration");
        }
        profile_ = std::move(profile);
    }

    bool finalized() const noexcept {
        return finalized_;
    }

    bool has_sources() const noexcept {
        return !sources_.empty();
    }

    int64_t budget_bytes() const noexcept {
        return budget_bytes_;
    }

    int64_t allocated_bytes() const noexcept {
        return allocated_bytes_;
    }

    const uint8_t * register_mapped_field(
            const mfq_tensor_backend::Tensor & field) {
        if (!mapped_gather_enabled_ || !field.defined() ||
                field.numel() == 0) {
            return nullptr;
        }
        if (!field.is_cpu() || !field.is_contiguous()) {
            throw std::runtime_error(
                "mapped MoE cache fields must be contiguous CPU tensors");
        }
        auto * host = field.data_ptr();
        const auto existing = mapped_host_lookup_.find(host);
        if (existing != mapped_host_lookup_.end()) {
            return existing->second;
        }
        const int64_t bytes = tensor_nbytes(field);
        if (bytes <= 0 ||
                static_cast<uint64_t>(bytes) >
                    std::numeric_limits<size_t>::max()) {
            throw std::runtime_error(
                "mapped MoE cache field has an invalid byte count");
        }
        cudaError_t status = cudaHostRegister(
            host,
            static_cast<size_t>(bytes),
            cudaHostRegisterMapped);
        bool owned = true;
        if (status == cudaErrorHostMemoryAlreadyRegistered) {
            (void)cudaGetLastError();
            owned = false;
        } else {
            MFQ_CUDA_CHECK(status);
        }
        void * device = nullptr;
        status = cudaHostGetDevicePointer(&device, host, 0);
        if (status != cudaSuccess) {
            if (owned) (void)cudaHostUnregister(host);
            MFQ_CUDA_CHECK(status);
        }
        auto * mapped = reinterpret_cast<const uint8_t *>(device);
        mapped_host_lookup_.emplace(host, mapped);
        registered_host_fields_.push_back({host, bytes, owned});
        mapped_registered_bytes_ += bytes;
        return mapped;
    }

    const MoeCacheStats & stats() const noexcept {
        return stats_;
    }

    bool prepare(
        MoeCachedSource & source,
        const std::vector<int32_t> & experts,
        bool prefetch);

    bool prepare_bundle(
        const std::vector<MoeCachedSource *> & sources,
        const std::vector<int32_t> & experts);

    bool prepare_bundle_deferred(
        const std::vector<MoeCachedSource *> & ready_sources,
        MoeCachedSource & deferred_source,
        const std::vector<int32_t> & experts);

    void prewarm();

    void begin_route_experts(
            const MoeRoutePlan & route,
            int n_experts) {
        if (route.host_unique_experts) return;
        const auto & ids = route.ids;
        if (!ids.is_cuda() || !ids.is_contiguous() ||
                ids.scalar_type() != mfq_tensor_backend::kInt32 ||
                ids.dim() != 2) {
            throw std::runtime_error(
                "cached MoE routes must be contiguous CUDA int32");
        }
        const int64_t count = ids.numel();
        if (count <= 0) {
            throw std::runtime_error(
                "cached MoE route list is empty");
        }
        if (n_experts <= 0) {
            throw std::runtime_error(
                "cached MoE expert count must be positive");
        }
        if (route_readback_pending_) {
            if (pending_route_generation_ == route.generation) return;
            throw std::runtime_error(
                "overlapping cached MoE route readbacks are unsupported");
        }
        if (!route_host_.defined() ||
                route_host_.numel() < count) {
            route_host_ = mfq_tensor_backend::empty(
                {count},
                mfq_tensor_backend::TensorOptions()
                    .device(mfq_tensor_backend::kCPU)
                    .dtype(mfq_tensor_backend::kInt32)
                    .pinned_memory(true));
        }
        auto current =
            mfq_get_current_cuda_stream().stream();
        MFQ_CUDA_CHECK(cudaEventRecord(
            route_input_ready_, current));
        MFQ_CUDA_CHECK(cudaStreamWaitEvent(
            route_stream_, route_input_ready_, 0));
        const int64_t nbytes =
            count * static_cast<int64_t>(sizeof(int32_t));
        MFQ_CUDA_CHECK(cudaMemcpyAsync(
            route_host_.data_ptr<int32_t>(),
            ids.data_ptr<int32_t>(),
            static_cast<size_t>(nbytes),
            cudaMemcpyDeviceToHost,
            route_stream_));
        MFQ_CUDA_CHECK(cudaEventRecord(
            route_done_, route_stream_));
        stats_.route_d2h_bytes += nbytes;
        pending_route_generation_ = route.generation;
        pending_route_count_ = count;
        pending_route_n_experts_ = n_experts;
        route_readback_pending_ = true;
    }

    std::vector<int32_t> read_route_experts(
            const MoeRoutePlan & route,
            int n_experts) {
        if (!route.host_unique_experts &&
                (!route_readback_pending_ ||
                 pending_route_generation_ != route.generation)) {
            begin_route_experts(route, n_experts);
        }
        if (!route_readback_pending_ ||
                pending_route_generation_ != route.generation ||
                pending_route_n_experts_ != n_experts) {
            throw std::runtime_error(
                "cached MoE route readback state does not match the route");
        }
        MFQ_CUDA_CHECK(cudaEventSynchronize(route_done_));
        const int64_t count = pending_route_count_;
        route_readback_pending_ = false;
        pending_route_generation_ = 0;
        pending_route_count_ = 0;
        pending_route_n_experts_ = 0;
        const auto * values =
            route_host_.data_ptr<int32_t>();
        std::vector<int32_t> result(
            values, values + count);
        std::sort(result.begin(), result.end());
        result.erase(
            std::unique(result.begin(), result.end()),
            result.end());
        for (int expert : result) {
            if (expert < 0 || expert >= n_experts) {
                throw std::runtime_error(
                    "MoE route selected an out-of-range expert");
            }
        }
        return result;
    }

    void record_compute_use() {
        MFQ_CUDA_CHECK(cudaEventRecord(
            compute_done_,
            mfq_get_current_cuda_stream().stream()));
        compute_done_recorded_ = true;
    }

    void count_full_projection_fallback() {
        ++stats_.full_projection_fallbacks;
    }

    void print_stats(std::ostream & stream) const {
        stream << "moe_cache_stats"
               << " budget_bytes=" << budget_bytes_
               << " allocated_bytes=" << allocated_bytes_
               << " host_bytes=" << host_bytes_
               << " demand_hits=" << stats_.demand_hits
               << " demand_misses=" << stats_.demand_misses
               << " prefetch_hits=" << stats_.prefetch_hits
               << " prefetch_misses=" << stats_.prefetch_misses
               << " evictions=" << stats_.evictions
               << " h2d_bytes=" << stats_.h2d_bytes
               << " route_d2h_bytes="
               << stats_.route_d2h_bytes
               << " full_projection_fallbacks="
               << stats_.full_projection_fallbacks
               << " h2d_submissions="
               << stats_.h2d_submissions
               << " h2d_descriptors="
               << stats_.h2d_descriptors
               << " mapped_registered_bytes="
               << mapped_registered_bytes_
               << " mapped_gather_bytes="
               << stats_.mapped_gather_bytes
               << " mapped_gather_submissions="
               << stats_.mapped_gather_submissions
               << " mapped_gather_descriptors="
               << stats_.mapped_gather_descriptors
               << " range_read_bytes="
               << stats_.range_read_bytes
               << " range_read_calls="
               << stats_.range_read_calls
               << " range_file_opens="
               << stats_.range_file_opens
               << " range_io_workers="
               << range_read_pool_->workers()
               << " range_read_ms="
               << static_cast<double>(stats_.range_read_nanoseconds) /
                    1.0e6
               << " range_overlap_batches="
               << stats_.range_overlap_batches
               << " range_overlap_wait_ms="
               << static_cast<double>(
                    stats_.range_overlap_wait_nanoseconds) / 1.0e6
               << "\n";
    }

private:
    friend class MoeCachedSource;

    void append_source_transfers(
        MoeCachedSource & source,
        const std::vector<int32_t> & experts,
        bool prefetch,
        std::vector<MoeCacheTransfer> & transfers,
        bool & replaced_occupied,
        std::vector<std::pair<mfq::MoeCacheSlotBook *, int>> * held_slots,
        std::vector<MoeCacheNewLease> * new_leases);

    void rollback_preparation(
        const std::vector<MoeCacheNewLease> & new_leases,
        const std::vector<
            std::pair<mfq::MoeCacheSlotBook *, int>> & held_slots) noexcept;

    bool begin_deferred_range_read(
        MoeCachedSource & source,
        const std::vector<int32_t> & experts);

    void finish_deferred_range_read(MoeCachedSource & source);

    void record_range_read(
        const mfq::cuda::MfeMxfp4ReadBatchStats & range_stats) {
        stats_.range_read_bytes += static_cast<int64_t>(range_stats.bytes);
        stats_.range_read_calls += static_cast<int64_t>(range_stats.calls);
        stats_.range_file_opens +=
            static_cast<int64_t>(range_stats.file_opens);
        stats_.range_read_nanoseconds +=
            static_cast<int64_t>(range_stats.wall_nanoseconds);
    }

    MoePinnedStage & acquire_stage(
            int64_t required_bytes,
            bool require_device) {
        for (size_t offset = 0; offset < stages_.size(); ++offset) {
            const size_t index =
                (next_stage_ + offset) % stages_.size();
            auto & stage = stages_[index];
            if (!stage.pending ||
                cudaEventQuery(stage.done) == cudaSuccess) {
                stage.pending = false;
                next_stage_ = (index + 1) % stages_.size();
                if (!stage.host.defined() ||
                        stage.host.numel() < required_bytes) {
                    int64_t capacity = 1;
                    while (capacity < required_bytes) capacity *= 2;
                    stage.host = mfq_tensor_backend::empty(
                        {capacity},
                        mfq_tensor_backend::TensorOptions()
                            .device(mfq_tensor_backend::kCPU)
                            .dtype(mfq_tensor_backend::kUInt8)
                            .pinned_memory(true));
                }
                if (require_device &&
                        (!stage.device.defined() ||
                         stage.device.numel() < required_bytes)) {
                    int64_t capacity = 1;
                    while (capacity < required_bytes) capacity *= 2;
                    stage.device = mfq_tensor_backend::empty(
                        {capacity},
                        mfq_tensor_backend::TensorOptions()
                            .device(mfq_tensor_backend::kCUDA)
                            .dtype(mfq_tensor_backend::kUInt8));
                }
                return stage;
            }
            (void)cudaGetLastError();
        }
        auto & stage = stages_[next_stage_];
        MFQ_CUDA_CHECK(cudaEventSynchronize(stage.done));
        stage.pending = false;
        next_stage_ = (next_stage_ + 1) % stages_.size();
        if (!stage.host.defined() ||
                stage.host.numel() < required_bytes) {
            int64_t capacity = 1;
            while (capacity < required_bytes) capacity *= 2;
            stage.host = mfq_tensor_backend::empty(
                {capacity},
                mfq_tensor_backend::TensorOptions()
                    .device(mfq_tensor_backend::kCPU)
                    .dtype(mfq_tensor_backend::kUInt8)
                    .pinned_memory(true));
        }
        if (require_device &&
                (!stage.device.defined() ||
                 stage.device.numel() < required_bytes)) {
            int64_t capacity = 1;
            while (capacity < required_bytes) capacity *= 2;
            stage.device = mfq_tensor_backend::empty(
                {capacity},
                mfq_tensor_backend::TensorOptions()
                    .device(mfq_tensor_backend::kCUDA)
                    .dtype(mfq_tensor_backend::kUInt8));
        }
        return stage;
    }

    void submit_transfers(
            const std::vector<MoeCacheTransfer> & transfers,
            bool waits_for_compute,
            bool wait_on_compute_stream) {
        std::vector<mfq::cuda::MfeMxfp4ReadRequest> range_requests;
        auto materialize_source = [&range_requests](
                const MoeCacheTransfer & transfer,
                uint8_t * destination) {
            if (transfer.range_store != nullptr) {
                range_requests.push_back({
                    transfer.range_store,
                    transfer.range_part,
                    std::span<uint8_t>(
                        destination,
                        static_cast<size_t>(transfer.nbytes)),
                });
            } else {
                std::memcpy(
                    destination,
                    transfer.source,
                    static_cast<size_t>(transfer.nbytes));
            }
        };
        auto finish_range_reads = [this, &range_requests]() {
            if (range_requests.empty()) return;
            const auto range_stats = range_read_pool_->read(range_requests);
            record_range_read(range_stats);
            range_requests.clear();
        };
        int64_t staged_payload_bytes = 0;
        int transfer_count = 0;
        int staged_count = 0;
        int mapped_count = 0;
        for (const auto & transfer : transfers) {
            const bool range_source =
                transfer.range_store != nullptr &&
                transfer.range_part != nullptr;
            if (transfer.nbytes < 0 ||
                (transfer.nbytes > 0 &&
                 ((!range_source && transfer.source == nullptr) ||
                  transfer.destination == nullptr)) ||
                ((transfer.range_store == nullptr) !=
                 (transfer.range_part == nullptr)) ||
                (range_source &&
                 (transfer.mapped_source != nullptr ||
                  transfer.range_part->nbytes !=
                    static_cast<uint64_t>(transfer.nbytes)))) {
                throw std::runtime_error(
                    "invalid MoE cache transfer");
            }
            if (transfer.nbytes == 0) continue;
            const bool direct_mapped =
                !prewarming_ && transfer.mapped_source != nullptr;
            if (direct_mapped) {
                ++mapped_count;
            } else {
                staged_payload_bytes =
                    (staged_payload_bytes + 15) & ~int64_t{15};
                if (transfer.nbytes >
                        std::numeric_limits<int64_t>::max() -
                            staged_payload_bytes) {
                    throw std::overflow_error(
                        "MoE cache transfer byte count overflows int64");
                }
                staged_payload_bytes += transfer.nbytes;
                ++staged_count;
            }
            ++transfer_count;
        }
        if (transfer_count == 0) {
            if (wait_on_compute_stream &&
                    transfer_ready_recorded_) {
                MFQ_CUDA_CHECK(cudaStreamWaitEvent(
                    mfq_get_current_cuda_stream().stream(),
                    transfer_ready_, 0));
            }
            return;
        }
        ++stats_.h2d_submissions;
        stats_.h2d_descriptors += transfer_count;
        if (prewarming_) {
            auto & stage = acquire_stage(staged_payload_bytes, false);
            auto * staging = stage.host.data_ptr<uint8_t>();
            int64_t offset = 0;
            for (const auto & transfer : transfers) {
                if (transfer.nbytes == 0) continue;
                offset = (offset + 15) & ~int64_t{15};
                materialize_source(transfer, staging + offset);
                offset += transfer.nbytes;
            }
            finish_range_reads();
            offset = 0;
            for (const auto & transfer : transfers) {
                if (transfer.nbytes == 0) continue;
                offset = (offset + 15) & ~int64_t{15};
                MFQ_CUDA_CHECK(cudaMemcpyAsync(
                    transfer.destination,
                    staging + offset,
                    static_cast<size_t>(transfer.nbytes),
                    cudaMemcpyHostToDevice,
                    weight_stream_));
                if (transfer.packed_weight) {
                    stats_.h2d_bytes += transfer.nbytes;
                }
                offset += transfer.nbytes;
            }
            MFQ_CUDA_CHECK(cudaEventRecord(
                stage.done, weight_stream_));
            MFQ_CUDA_CHECK(cudaEventRecord(
                transfer_ready_, weight_stream_));
            transfer_ready_recorded_ = true;
            stage.pending = true;
            if (wait_on_compute_stream) {
                MFQ_CUDA_CHECK(cudaStreamWaitEvent(
                    mfq_get_current_cuda_stream().stream(),
                    transfer_ready_, 0));
            }
            return;
        }
        const int64_t scatter_descriptor_offset =
            (staged_payload_bytes + 15) & ~int64_t{15};
        const int64_t scatter_descriptor_bytes =
            static_cast<int64_t>(staged_count) *
            static_cast<int64_t>(
                sizeof(mfq::MoeCacheScatterDescriptor));
        if (scatter_descriptor_bytes >
                std::numeric_limits<int64_t>::max() -
                    scatter_descriptor_offset) {
            throw std::overflow_error(
                "MoE cache descriptor byte count overflows int64");
        }
        const int64_t mapped_descriptor_offset =
            (scatter_descriptor_offset +
             scatter_descriptor_bytes + 15) & ~int64_t{15};
        const int64_t mapped_descriptor_bytes =
            static_cast<int64_t>(mapped_count) *
            static_cast<int64_t>(
                sizeof(mfq::MoeCacheMappedCopyDescriptor));
        if (mapped_descriptor_bytes >
                std::numeric_limits<int64_t>::max() -
                    mapped_descriptor_offset) {
            throw std::overflow_error(
                "MoE cache mapped descriptor byte count overflows int64");
        }
        const int64_t total_bytes =
            mapped_descriptor_offset + mapped_descriptor_bytes;
        auto & stage = acquire_stage(total_bytes, true);
        auto * staging = stage.host.data_ptr<uint8_t>();
        auto * scatter_descriptors =
            reinterpret_cast<mfq::MoeCacheScatterDescriptor *>(
                staging + scatter_descriptor_offset);
        auto * mapped_descriptors =
            reinterpret_cast<mfq::MoeCacheMappedCopyDescriptor *>(
                staging + mapped_descriptor_offset);
        int64_t offset = 0;
        int scatter_descriptor = 0;
        int mapped_descriptor = 0;
        for (const auto & transfer : transfers) {
            if (transfer.nbytes == 0) continue;
            if (transfer.mapped_source != nullptr) {
                mapped_descriptors[mapped_descriptor++] = {
                    static_cast<uint64_t>(
                        reinterpret_cast<uintptr_t>(
                            transfer.destination)),
                    static_cast<uint64_t>(
                        reinterpret_cast<uintptr_t>(
                            transfer.mapped_source)),
                    static_cast<uint64_t>(transfer.nbytes),
                };
                stats_.mapped_gather_bytes += transfer.nbytes;
            } else {
                offset = (offset + 15) & ~int64_t{15};
                materialize_source(transfer, staging + offset);
                scatter_descriptors[scatter_descriptor++] = {
                    static_cast<uint64_t>(
                        reinterpret_cast<uintptr_t>(
                            transfer.destination)),
                    static_cast<uint64_t>(offset),
                    static_cast<uint64_t>(transfer.nbytes),
                };
                offset += transfer.nbytes;
            }
            if (transfer.packed_weight) {
                stats_.h2d_bytes += transfer.nbytes;
            }
        }
        finish_range_reads();
        if (scatter_descriptor != staged_count ||
                mapped_descriptor != mapped_count) {
            throw std::runtime_error(
                "MoE cache transfer descriptor count mismatch");
        }
        if (waits_for_compute && compute_done_recorded_) {
            MFQ_CUDA_CHECK(cudaStreamWaitEvent(
                weight_stream_, compute_done_, 0));
        }
        MFQ_CUDA_CHECK(cudaMemcpyAsync(
            stage.device.data_ptr<uint8_t>(),
            staging,
            static_cast<size_t>(total_bytes),
            cudaMemcpyHostToDevice,
            weight_stream_));
        if (staged_count > 0) {
            mfq::moe_cache_scatter_cuda(
                stage.device.data_ptr<uint8_t>(),
                scatter_descriptor_offset,
                staged_count,
                weight_stream_);
        }
        if (mapped_count > 0) {
            auto * device_descriptors =
                reinterpret_cast<const mfq::MoeCacheMappedCopyDescriptor *>(
                    stage.device.data_ptr<uint8_t>() +
                    mapped_descriptor_offset);
            mfq::moe_cache_mapped_gather_cuda(
                device_descriptors,
                mapped_count,
                mapped_copy_blocks_,
                weight_stream_);
            ++stats_.mapped_gather_submissions;
            stats_.mapped_gather_descriptors += mapped_count;
        }
        MFQ_CUDA_CHECK(cudaEventRecord(stage.done, weight_stream_));
        MFQ_CUDA_CHECK(cudaEventRecord(
            transfer_ready_, weight_stream_));
        transfer_ready_recorded_ = true;
        stage.pending = true;
        if (wait_on_compute_stream) {
            MFQ_CUDA_CHECK(cudaStreamWaitEvent(
                mfq_get_current_cuda_stream().stream(),
                transfer_ready_, 0));
        }
    }

    void invalidate(const mfq::MoeCacheKey & key, int slot);

    int64_t budget_bytes_ = 0;
    int64_t allocated_bytes_ = 0;
    int64_t host_bytes_ = 0;
    bool finalized_ = false;
    bool prewarming_ = false;
    cudaStream_t weight_stream_ = nullptr;
    cudaStream_t route_stream_ = nullptr;
    cudaEvent_t compute_done_ = nullptr;
    bool compute_done_recorded_ = false;
    cudaEvent_t transfer_ready_ = nullptr;
    bool transfer_ready_recorded_ = false;
    cudaEvent_t route_input_ready_ = nullptr;
    cudaEvent_t route_done_ = nullptr;
    mfq_tensor_backend::Tensor route_host_;
    uint64_t pending_route_generation_ = 0;
    int64_t pending_route_count_ = 0;
    int pending_route_n_experts_ = 0;
    bool route_readback_pending_ = false;
    std::vector<MoePinnedStage> stages_;
    size_t next_stage_ = 0;
    std::unordered_map<std::string, std::unique_ptr<MoeGpuArena>> arenas_;
    std::vector<std::shared_ptr<MoeCachedSource>> sources_;
    std::optional<mfq::MoeCacheProfile> profile_;
    std::vector<mfq::MoeProfileCandidate> prewarm_selected_;
    MoeCacheStats stats_;
    struct RegisteredHostField {
        void * host = nullptr;
        int64_t bytes = 0;
        bool owned = false;
    };
    bool mapped_gather_enabled_ = false;
    int mapped_copy_blocks_ = 64;
    int64_t mapped_registered_bytes_ = 0;
    std::vector<RegisteredHostField> registered_host_fields_;
    std::unordered_map<void *, const uint8_t *> mapped_host_lookup_;
    std::unique_ptr<mfq::cuda::MfeMxfp4ReadPool> range_read_pool_;
    std::unordered_map<int, std::unique_ptr<MoePendingRangeRead>>
        pending_range_reads_;
    bool range_overlap_enabled_ = true;
};

struct MoeCachedCohort {
    int index = -1;
    const MixedMoePool * cpu = nullptr;
    MoeGpuArena * arena = nullptr;
    std::vector<mfq_tensor_backend::Tensor> cpu_fields;
    std::vector<const uint8_t *> mapped_fields;
    std::vector<int64_t> bytes_per_expert;
    std::vector<int32_t> expert_to_local;
    std::vector<int32_t> host_map;
    std::shared_ptr<mfq::cuda::MfeMxfp4ExpertStore> range_store;
    bool map_dirty = false;
    MixedMoePool active;
};

class MoeCachedSource : public std::enable_shared_from_this<MoeCachedSource> {
public:
    MoeCachedSource(
            MoeExpertCache * cache,
            int id,
            std::string name,
            std::shared_ptr<MixedMoeRuntime> cpu,
            int minimum_slots,
            int layer_id,
            std::string projection_role,
            std::shared_ptr<mfq::cuda::MfeMxfp4ExpertStore> range_store)
        : cache_(cache),
          id_(id),
          name_(std::move(name)),
          layer_id_(layer_id),
          projection_role_(std::move(projection_role)),
          cpu_(std::move(cpu)),
          range_store_(std::move(range_store)),
          expert_to_cohort_(
              static_cast<size_t>(cpu_->n_experts), -1),
          expert_to_local_(
              static_cast<size_t>(cpu_->n_experts), -1) {
        if (minimum_slots <= 0) {
            throw std::runtime_error(
                "MoE cache source minimum slots must be positive");
        }
        cohorts_.reserve(cpu_->pools.size());
        if (range_store_) {
            if (cpu_->pools.size() != 1 ||
                    cpu_->pools.front().family != MixedMoeFamily::Mxfp4 ||
                    cpu_->n_experts != range_store_->num_experts() ||
                    cpu_->out_per_expert != range_store_->out_per_expert() ||
                    cpu_->neuron_len != range_store_->neuron_len() ||
                    range_store_->values_bytes_per_expert() >
                        static_cast<uint64_t>(
                            std::numeric_limits<int64_t>::max()) ||
                    range_store_->scales_bytes_per_expert() >
                        static_cast<uint64_t>(
                            std::numeric_limits<int64_t>::max())) {
                throw std::runtime_error(
                    "invalid exact-range MXFP4 cache metadata");
            }
            const auto & pool = cpu_->pools.front();
            MoeCachedCohort cohort;
            cohort.index = 0;
            cohort.cpu = &pool;
            cohort.range_store = range_store_;
            const int64_t values = static_cast<int64_t>(
                range_store_->values_bytes_per_expert());
            const int64_t scales = static_cast<int64_t>(
                range_store_->scales_bytes_per_expert());
            cohort.arena = cache_->register_cohort_layout(
                pool,
                cpu_->out_per_expert,
                cpu_->neuron_len,
                minimum_slots,
                cpu_->n_experts,
                {
                    {
                        mfq_tensor_backend::kUInt8,
                        {cpu_->out_per_expert, cpu_->neuron_len / 2},
                        values,
                        1,
                    },
                    {
                        mfq_tensor_backend::kUInt8,
                        {cpu_->out_per_expert, cpu_->neuron_len / 32},
                        scales,
                        1,
                    },
                });
            cohort.bytes_per_expert = {values, scales};
            cohort.mapped_fields = {nullptr, nullptr};
            cohort.expert_to_local.resize(
                static_cast<size_t>(cpu_->n_experts));
            cohort.host_map.assign(
                static_cast<size_t>(cpu_->n_experts), -1);
            for (int expert = 0; expert < cpu_->n_experts; ++expert) {
                cohort.expert_to_local[static_cast<size_t>(expert)] = expert;
                expert_to_cohort_[static_cast<size_t>(expert)] = 0;
                expert_to_local_[static_cast<size_t>(expert)] = expert;
            }
            cohorts_.push_back(std::move(cohort));
        } else {
            for (int cohort_index = 0;
                 cohort_index < static_cast<int>(cpu_->pools.size());
                 ++cohort_index) {
                const auto & pool =
                    cpu_->pools.at(static_cast<size_t>(cohort_index));
                MoeCachedCohort cohort;
                cohort.index = cohort_index;
                cohort.cpu = &pool;
                cohort.arena = cache_->register_cohort(
                    pool, cpu_->out_per_expert, cpu_->neuron_len,
                    minimum_slots);
                cohort.cpu_fields = moe_cache_fields(pool);
                cohort.expert_to_local.assign(
                    static_cast<size_t>(cpu_->n_experts), -1);
                cohort.host_map.assign(
                    static_cast<size_t>(cpu_->n_experts), -1);
                const auto * local =
                    pool.expert_local.data_ptr<int32_t>();
                for (int expert = 0; expert < cpu_->n_experts; ++expert) {
                    const int local_index = local[expert];
                    cohort.expert_to_local[
                        static_cast<size_t>(expert)] = local_index;
                    if (local_index < 0) continue;
                    if (expert_to_cohort_[
                            static_cast<size_t>(expert)] >= 0) {
                        throw std::runtime_error(
                            "MoE cache source has duplicate expert ownership");
                    }
                    expert_to_cohort_[
                        static_cast<size_t>(expert)] = cohort_index;
                    expert_to_local_[
                        static_cast<size_t>(expert)] = local_index;
                }
                for (const auto & field : cohort.cpu_fields) {
                    const int64_t nbytes = tensor_nbytes(field);
                    if (nbytes % pool.local_experts != 0) {
                        throw std::runtime_error(
                            "MoE cache field byte count is not expert aligned");
                    }
                    cohort.bytes_per_expert.push_back(
                        nbytes / pool.local_experts);
                    cohort.mapped_fields.push_back(
                        cache_->register_mapped_field(field));
                }
                cohorts_.push_back(std::move(cohort));
            }
        }
        if (std::any_of(
                expert_to_cohort_.begin(),
                expert_to_cohort_.end(),
                [](int value) { return value < 0; })) {
            throw std::runtime_error(
                "MoE cache source does not cover every expert");
        }
    }

    int id() const noexcept {
        return id_;
    }

    const std::string & name() const noexcept {
        return name_;
    }

    int layer_id() const noexcept {
        return layer_id_;
    }

    const std::string & projection_role() const noexcept {
        return projection_role_;
    }

    int n_experts() const noexcept {
        return cpu_->n_experts;
    }

    MoeGpuArena * arena_for_expert(int expert) const {
        if (expert < 0 || expert >= cpu_->n_experts) {
            throw std::out_of_range(
                "MoE cache profile expert is out of range");
        }
        const int cohort = expert_to_cohort_.at(
            static_cast<size_t>(expert));
        return cohorts_.at(static_cast<size_t>(cohort)).arena;
    }

    int64_t bytes_for_expert(int expert) const {
        return arena_for_expert(expert)->slot_bytes;
    }

    void touch_expert(int expert) {
        const int cohort = expert_to_cohort_.at(
            static_cast<size_t>(expert));
        auto & arena = *cohorts_.at(
            static_cast<size_t>(cohort)).arena;
        const int slot = arena.book->slot_for(
            {id_, cohort, expert});
        if (slot < 0) {
            throw std::runtime_error(
                "prewarmed MoE expert is absent from its arena");
        }
        arena.book->touch(slot);
    }

    int64_t host_bytes() const {
        return mixed_moe_storage_bytes(*cpu_);
    }

    int64_t logical_weight_bytes() const {
        if (!range_store_) return host_bytes();
        return range_store_->record().nbytes >
                static_cast<uint64_t>(std::numeric_limits<int64_t>::max())
            ? std::numeric_limits<int64_t>::max()
            : static_cast<int64_t>(range_store_->record().nbytes);
    }

    void finalize() {
        if (active_) return;
        active_ = std::make_shared<MixedMoeRuntime>();
        active_->n_experts = cpu_->n_experts;
        active_->out_per_expert = cpu_->out_per_expert;
        active_->neuron_len = cpu_->neuron_len;
        active_->pools.reserve(cohorts_.size());
        for (auto & cohort : cohorts_) {
            auto & source = *cohort.cpu;
            auto & arena = *cohort.arena;
            MixedMoePool pool = source;
            pool.local_experts = arena.slots;
            pool.expert_local = mfq_tensor_backend::full(
                {cpu_->n_experts}, -1,
                mfq_tensor_backend::TensorOptions()
                    .device(mfq_tensor_backend::kCUDA)
                    .dtype(mfq_tensor_backend::kInt32));
            size_t field = 0;
            if (pool.family == MixedMoeFamily::Nint) {
                pool.nint.aligned_q8 = false;
                pool.nint.uniform_q46 = 0;
                pool.nint.workspaces.clear();
                pool.nint.q_packed = arena.fields.at(field++);
                pool.nint.row_q_bits = arena.fields.at(field++);
                pool.nint.row_q_bit_offsets = arena.fields.at(field++);
                pool.nint.sub_scale = arena.fields.at(field++);
                pool.nint.sub_min = arena.fields.at(field++);
                pool.nint.neuron_scale = arena.fields.at(field++);
                pool.nint.neuron_min = arena.fields.at(field++);
                pool.nint.out =
                    static_cast<int64_t>(arena.slots) *
                    cpu_->out_per_expert;
                if (!pool.nint.shape.empty()) {
                    pool.nint.shape[0] = pool.nint.out;
                }
            } else if (pool.family == MixedMoeFamily::Nint8Zero) {
                pool.q8_zero.workspaces.clear();
                pool.q8_zero.q_packed = arena.fields.at(field++);
                pool.q8_zero.q8_zero_scale = arena.fields.at(field++);
                pool.q8_zero.out =
                    static_cast<int64_t>(arena.slots) *
                    cpu_->out_per_expert;
                if (!pool.q8_zero.shape.empty()) {
                    pool.q8_zero.shape[0] = pool.q8_zero.out;
                }
            } else if (pool.family == MixedMoeFamily::Mxfp4) {
                pool.mxfp4.values = arena.fields.at(field++);
                pool.mxfp4.scales = arena.fields.at(field++);
                pool.mxfp4.out =
                    static_cast<int64_t>(arena.slots) *
                    cpu_->out_per_expert;
            } else if (pool.family == MixedMoeFamily::Tpq) {
                pool.tpq.packed = arena.fields.at(field++);
                pool.tpq.codebook =
                    copy_cpu_weight_to_cuda(source.tpq.codebook);
                pool.tpq.out =
                    static_cast<int64_t>(arena.slots) *
                    cpu_->out_per_expert;
            } else if (pool.family == MixedMoeFamily::Nvq) {
                pool.nvq.workspaces.clear();
                pool.nvq.indices_packed = arena.fields.at(field++);
                pool.nvq.aux_packed = arena.fields.at(field++);
                pool.nvq.sub_scale_packed = arena.fields.at(field++);
                pool.nvq.neuron_scale = arena.fields.at(field++);
                pool.nvq.codebook =
                    copy_cpu_weight_to_cuda(source.nvq.codebook);
                pool.nvq.out =
                    static_cast<int64_t>(arena.slots) *
                    cpu_->out_per_expert;
                if (!pool.nvq.shape.empty()) {
                    pool.nvq.shape[0] = pool.nvq.out;
                }
            } else {
                pool.nepq.indices_packed = arena.fields.at(field++);
                pool.nepq.aux_packed = arena.fields.at(field++);
                pool.nepq.state_packed = arena.fields.at(field++);
                pool.nepq.neuron_scale = arena.fields.at(field++);
                pool.nepq.bank_ids = arena.fields.at(field++);
                pool.nepq.table_pool =
                    copy_cpu_weight_to_cuda(source.nepq.table_pool);
                pool.nepq.grouped_table_pool =
                    copy_cpu_weight_to_cuda(
                        source.nepq.grouped_table_pool);
                pool.nepq.rotation_signs =
                    copy_cpu_weight_to_cuda(
                        source.nepq.rotation_signs);
                if (pool.nepq.residual) {
                    pool.nepq.residual_codebook =
                        copy_cpu_weight_to_cuda(
                            source.nepq.residual_codebook);
                    pool.nepq.residual_first =
                        arena.fields.at(field++);
                    pool.nepq.residual_second =
                        arena.fields.at(field++);
                }
                pool.nepq.n_experts = arena.slots;
            }
            if (field != arena.fields.size()) {
                throw std::runtime_error(
                    "MoE cache active field count does not match its arena");
            }
            cohort.active = std::move(pool);
            active_->pools.push_back(cohort.active);
        }
    }

    void invalidate(int cohort_index, int expert, int slot) {
        if (cohort_index < 0 ||
            cohort_index >= static_cast<int>(cohorts_.size()) ||
            expert < 0 || expert >= cpu_->n_experts) {
            throw std::runtime_error(
                "invalid MoE cache eviction key");
        }
        auto & cohort =
            cohorts_.at(static_cast<size_t>(cohort_index));
        if (cohort.host_map[static_cast<size_t>(expert)] == slot) {
            cohort.host_map[static_cast<size_t>(expert)] = -1;
            cohort.map_dirty = true;
        }
    }

    std::vector<int32_t> route_experts(
            const MoeRoutePlan & route) const {
        if (!route.host_unique_experts) {
            route.host_unique_experts =
                std::make_shared<std::vector<int32_t>>(
                    cache_->read_route_experts(
                        route, cpu_->n_experts));
        }
        return *route.host_unique_experts;
    }

    void begin_prefetch(const MoeRoutePlan & route) {
        if (use_full_projection(route)) return;
        cache_->begin_route_experts(route, cpu_->n_experts);
    }

    bool use_full_projection(const MoeRoutePlan & route) const {
        return !range_store_ && route.ids.size(0) > 8;
    }

    mfq_tensor_backend::Tensor forward(
            mfq_tensor_backend::Tensor x,
            const MoeRoutePlan & route) {
        if (use_full_projection(route)) {
            cache_->count_full_projection_fallback();
            auto staged = stage_fallback_runtime();
            return staged.forward(x, route);
        }
        if (!cache_->prepare(
                *this, route_experts(route), false)) {
            cache_->count_full_projection_fallback();
            auto staged = stage_fallback_runtime();
            return staged.forward(x, route);
        }
        auto output = active_->forward(x, route);
        cache_->record_compute_use();
        return output;
    }

    mfq_tensor_backend::Tensor forward_prequantized(
            mfq_tensor_backend::Tensor x,
            const MoeRoutePlan & route) {
        if (use_full_projection(route)) {
            throw std::runtime_error(
                "cached prequantized activation reuse only supports decode-sized routes");
        }
        if (!cache_->prepare(
                *this, route_experts(route), false)) {
            cache_->count_full_projection_fallback();
            auto staged = stage_fallback_runtime();
            return staged.forward(x, route);
        }
        auto output = active_->forward(x, route, true);
        cache_->record_compute_use();
        return output;
    }

    void prefetch(const MoeRoutePlan & route) {
        if (use_full_projection(route)) return;
        (void)cache_->prepare(
            *this, route_experts(route), true);
    }

    mfq_tensor_backend::Tensor forward_glu_output(
            mfq_tensor_backend::Tensor x,
            const MoeRoutePlan & route,
            bool gelu) {
        if (use_full_projection(route)) {
            cache_->count_full_projection_fallback();
            auto staged = stage_fallback_runtime();
            return staged.forward_glu_output(x, route, gelu);
        }
        if (!cache_->prepare(
                *this, route_experts(route), false)) {
            cache_->count_full_projection_fallback();
            auto staged = stage_fallback_runtime();
            return staged.forward_glu_output(x, route, gelu);
        }
        auto output = active_->forward_glu_output(x, route, gelu);
        cache_->record_compute_use();
        return output;
    }

    mfq_tensor_backend::Tensor forward_glu(
            mfq_tensor_backend::Tensor gate_up,
            const MoeRoutePlan & route,
            bool gelu) {
        auto activation = gelu
            ? moe_geglu_split_cuda(gate_up)
            : moe_swiglu_split_cuda(gate_up);
        return forward(activation, route);
    }

    mfq_tensor_backend::Tensor forward_clamped_swiglu(
            mfq_tensor_backend::Tensor gate_up,
            const MoeRoutePlan & route,
            double limit) {
        if (use_full_projection(route)) {
            cache_->count_full_projection_fallback();
            auto staged = stage_cpu_mixed_moe(cpu_);
            return staged.forward_clamped_swiglu(
                gate_up, route, limit);
        }
        if (!cache_->prepare(
                *this, route_experts(route), false)) {
            cache_->count_full_projection_fallback();
            auto staged = stage_cpu_mixed_moe(cpu_);
            return staged.forward_clamped_swiglu(
                gate_up, route, limit);
        }
        auto output =
            active_->forward_clamped_swiglu(
                gate_up, route, limit);
        cache_->record_compute_use();
        return output;
    }

    bool supports_clamped_swiglu() const {
        return cpu_->supports_clamped_swiglu();
    }

    static bool prefetch_bundle(
            const std::vector<std::shared_ptr<MoeCachedSource>> & sources,
            const MoeRoutePlan & route) {
        if (sources.empty()) return false;
        auto & first = *sources.front();
        if (first.use_full_projection(route)) return false;
        std::vector<MoeCachedSource *> raw_sources;
        raw_sources.reserve(sources.size());
        for (const auto & source : sources) {
            if (!source || source->cache_ != first.cache_ ||
                    source->n_experts() != first.n_experts() ||
                    source->layer_id() != first.layer_id() ||
                    source->use_full_projection(route)) {
                return false;
            }
            raw_sources.push_back(source.get());
        }
        if (raw_sources.size() == 3 &&
                raw_sources[0]->projection_role_ == "gate" &&
                raw_sources[1]->projection_role_ == "up" &&
                raw_sources[2]->projection_role_ == "down" &&
                raw_sources[2]->range_store_) {
            return first.cache_->prepare_bundle_deferred(
                {raw_sources[0], raw_sources[1]},
                *raw_sources[2],
                first.route_experts(route));
        }
        if (raw_sources.size() == 2 &&
                raw_sources[0]->projection_role_ == "gate_up" &&
                raw_sources[1]->projection_role_ == "down" &&
                raw_sources[1]->range_store_) {
            return first.cache_->prepare_bundle_deferred(
                {raw_sources[0]},
                *raw_sources[1],
                first.route_experts(route));
        }
        return first.cache_->prepare_bundle(
            raw_sources, first.route_experts(route));
    }

private:
    friend class MoeExpertCache;

    std::shared_ptr<MixedMoeRuntime> fallback_runtime() {
        if (!range_store_) return cpu_;
        std::lock_guard<std::mutex> guard(fallback_mutex_);
        if (!fallback_cpu_) {
            fallback_cpu_ = make_mixed_moe_runtime(
                unpack_mfe(range_store_->read_blob()), false);
        }
        return fallback_cpu_;
    }

    MfeWeight stage_fallback_runtime() {
        auto runtime = fallback_runtime();
        return stage_cpu_mixed_moe(runtime);
    }

    MoeExpertCache * cache_ = nullptr;
    int id_ = -1;
    std::string name_;
    int layer_id_ = -1;
    std::string projection_role_;
    std::shared_ptr<MixedMoeRuntime> cpu_;
    std::shared_ptr<mfq::cuda::MfeMxfp4ExpertStore> range_store_;
    std::mutex fallback_mutex_;
    std::shared_ptr<MixedMoeRuntime> fallback_cpu_;
    std::vector<MoeCachedCohort> cohorts_;
    std::vector<int> expert_to_cohort_;
    std::vector<int> expert_to_local_;
    std::shared_ptr<MixedMoeRuntime> active_;
};

std::shared_ptr<MoeCachedSource> MoeExpertCache::register_source(
        const std::string & name,
        std::shared_ptr<MixedMoeRuntime> cpu,
        int minimum_slots,
        int layer_id,
        std::string projection_role) {
    const int id = static_cast<int>(sources_.size());
    auto source = std::make_shared<MoeCachedSource>(
        this, id, name, std::move(cpu), minimum_slots,
        layer_id, std::move(projection_role), nullptr);
    host_bytes_ += source->host_bytes();
    sources_.push_back(source);
    return source;
}

std::shared_ptr<MoeCachedSource> MoeExpertCache::register_range_source(
        const std::string & name,
        std::shared_ptr<MixedMoeRuntime> metadata,
        std::shared_ptr<mfq::cuda::MfeMxfp4ExpertStore> store,
        int minimum_slots,
        int layer_id,
        std::string projection_role) {
    const int id = static_cast<int>(sources_.size());
    auto source = std::make_shared<MoeCachedSource>(
        this, id, name, std::move(metadata), minimum_slots,
        layer_id, std::move(projection_role), std::move(store));
    host_bytes_ += source->host_bytes();
    sources_.push_back(source);
    return source;
}

void MoeExpertCache::finalize() {
    if (finalized_) return;
    if (sources_.empty()) {
        throw std::runtime_error(
            "MoE cache has no registered expert sources");
    }
    std::vector<mfq::MoeArenaDemand> demands;
    demands.reserve(arenas_.size());
    for (const auto & item : arenas_) {
        const auto & arena = *item.second;
        demands.push_back({
            arena.signature,
            arena.slot_bytes,
            arena.minimum_slots,
            arena.registered_experts,
        });
    }

    if (profile_.has_value()) {
        std::unordered_map<
            int, std::vector<std::shared_ptr<MoeCachedSource>>>
            layer_sources;
        std::unordered_map<int, int> layer_experts;
        std::unordered_map<int, std::unordered_set<std::string>>
            layer_roles;
        for (const auto & source : sources_) {
            if (source->layer_id() < 0) continue;
            if (source->projection_role().empty()) {
                throw std::runtime_error(
                    "profiled MoE cache source has no projection role");
            }
            if (!layer_roles[source->layer_id()]
                     .insert(source->projection_role()).second) {
                throw std::runtime_error(
                    "profiled MoE cache layer has a duplicate projection role");
            }
            const auto inserted = layer_experts.emplace(
                source->layer_id(), source->n_experts());
            if (!inserted.second &&
                    inserted.first->second != source->n_experts()) {
                throw std::runtime_error(
                    "profiled MoE cache layer sources disagree on expert count");
            }
            layer_sources[source->layer_id()].push_back(source);
        }
        mfq::validate_moe_cache_profile(
            *profile_, layer_experts);

        std::unordered_map<std::string, int> required_slots;
        std::unordered_map<std::string, int> selected_slots;
        int64_t required_bytes = 0;
        for (const auto & demand : demands) {
            required_slots[demand.signature] =
                demand.minimum_slots;
            required_bytes +=
                static_cast<int64_t>(demand.minimum_slots) *
                demand.slot_bytes;
        }
        if (required_bytes > budget_bytes_) {
            throw std::runtime_error(
                "MoE cache budget is below the minimum decode working set");
        }

        auto bundle_bytes = [&](int layer, int expert) {
            int64_t bytes = 0;
            for (const auto & source : layer_sources.at(layer)) {
                const int64_t source_bytes =
                    source->bytes_for_expert(expert);
                if (source_bytes >
                        std::numeric_limits<int64_t>::max() - bytes) {
                    throw std::overflow_error(
                        "MoE cache prewarm bundle byte count overflows int64");
                }
                bytes += source_bytes;
            }
            return bytes;
        };

        auto select_candidate = [&](
                const mfq::MoeProfileCandidate & candidate,
                int64_t layer_limit,
                std::unordered_map<int, int64_t> * layer_used) {
            const auto source_it =
                layer_sources.find(candidate.layer);
            if (source_it == layer_sources.end() ||
                    source_it->second.empty()) {
                return false;
            }
            std::unordered_map<std::string, int> needed;
            for (const auto & source : source_it->second) {
                ++needed[
                    source->arena_for_expert(
                        candidate.expert)->signature];
            }
            int64_t incremental_bytes = 0;
            for (const auto & item : needed) {
                const auto arena_it = arenas_.find(item.first);
                if (arena_it == arenas_.end()) {
                    throw std::runtime_error(
                        "MoE cache profile references an unknown arena");
                }
                const auto & arena = *arena_it->second;
                const int selected =
                    selected_slots[item.first] + item.second;
                if (selected > arena.registered_experts) {
                    return false;
                }
                const int new_required = std::max(
                    required_slots[item.first], selected);
                incremental_bytes +=
                    static_cast<int64_t>(
                        new_required -
                        required_slots[item.first]) *
                    arena.slot_bytes;
            }
            const int64_t bytes =
                bundle_bytes(candidate.layer, candidate.expert);
            if (layer_used != nullptr &&
                    bytes >
                        layer_limit -
                        (*layer_used)[candidate.layer]) {
                return false;
            }
            if (incremental_bytes >
                    budget_bytes_ - required_bytes) {
                return false;
            }
            for (const auto & item : needed) {
                selected_slots[item.first] += item.second;
                required_slots[item.first] = std::max(
                    required_slots[item.first],
                    selected_slots[item.first]);
            }
            required_bytes += incremental_bytes;
            if (layer_used != nullptr) {
                (*layer_used)[candidate.layer] += bytes;
            }
            prewarm_selected_.push_back(candidate);
            return true;
        };

        const auto candidates =
            mfq::order_moe_profile_candidates(
                *profile_, false);
        int64_t frequency_bytes = 0;
        std::unordered_set<int> ranking_layers;
        for (const auto & candidate : candidates) {
            if (candidate.has_frequency) {
                if (select_candidate(
                        candidate,
                        std::numeric_limits<int64_t>::max(),
                        nullptr)) {
                    frequency_bytes +=
                        bundle_bytes(
                            candidate.layer,
                            candidate.expert);
                }
            } else {
                ranking_layers.insert(candidate.layer);
            }
        }

        std::unordered_map<int, int64_t> rank_layer_total;
        int64_t rank_model_total = 0;
        for (int layer : ranking_layers) {
            int64_t total = 0;
            for (int expert = 0;
                 expert < layer_experts.at(layer);
                 ++expert) {
                total += bundle_bytes(layer, expert);
            }
            rank_layer_total[layer] = total;
            rank_model_total += total;
        }
        const int64_t rank_budget =
            std::max<int64_t>(
                0, budget_bytes_ - frequency_bytes);
        std::unordered_map<int, int64_t> rank_layer_limit;
        for (const auto & item : rank_layer_total) {
            const long double fraction =
                rank_model_total > 0
                ? static_cast<long double>(item.second) /
                    static_cast<long double>(rank_model_total)
                : 0.0L;
            rank_layer_limit[item.first] =
                static_cast<int64_t>(
                    static_cast<long double>(rank_budget) *
                    fraction);
        }
        std::unordered_map<int, int64_t> rank_layer_used;
        for (const auto & candidate : candidates) {
            if (candidate.has_frequency) continue;
            (void)select_candidate(
                candidate,
                rank_layer_limit.at(candidate.layer),
                &rank_layer_used);
        }

        for (auto & demand : demands) {
            demand.minimum_slots = std::max(
                demand.minimum_slots,
                required_slots.at(demand.signature));
        }
    }
    const auto plan =
        mfq::plan_moe_arena_slots(budget_bytes_, demands);
    for (auto & item : arenas_) {
        auto & arena = *item.second;
        arena.slots = plan.at(arena.signature);
        arena.fields.reserve(arena.layouts.size());
        for (size_t index = 0;
             index < arena.layouts.size();
             ++index) {
            const auto & layout = arena.layouts[index];
            if (layout.slot_shape.empty()) {
                throw std::runtime_error(
                    "MoE cache field has no expert-major leading dimension");
            }
            auto shape = layout.slot_shape;
            shape[0] *= arena.slots;
            arena.fields.push_back(mfq_tensor_backend::empty(
                shape,
                mfq_tensor_backend::TensorOptions()
                    .device(mfq_tensor_backend::kCUDA)
                    .dtype(layout.scalar_type)));
        }
        arena.book =
            std::make_unique<mfq::MoeCacheSlotBook>(
                arena.slots);
        allocated_bytes_ +=
            static_cast<int64_t>(arena.slots) *
            arena.slot_bytes;
        std::cerr
            << "moe_cache_arena"
            << " signature=" << arena.signature
            << " slots=" << arena.slots
            << " slot_bytes=" << arena.slot_bytes
            << " bytes="
            << static_cast<int64_t>(arena.slots) *
                arena.slot_bytes
            << std::endl;
    }
    if (allocated_bytes_ > budget_bytes_) {
        throw std::runtime_error(
            "MoE cache arena allocation exceeded its budget");
    }
    for (auto & source : sources_) source->finalize();
    finalized_ = true;
    std::cerr
        << "moe_cache_ready"
        << " sources=" << sources_.size()
        << " host_bytes=" << host_bytes_
        << " budget_bytes=" << budget_bytes_
        << " allocated_bytes=" << allocated_bytes_
        << " mapped_gather=" << (mapped_gather_enabled_ ? 1 : 0)
        << " mapped_registered_bytes=" << mapped_registered_bytes_
        << " mapped_copy_blocks=" << mapped_copy_blocks_
        << std::endl;
    if (profile_.has_value()) prewarm();
}

void MoeExpertCache::prewarm() {
    if (!finalized_ || !profile_.has_value()) return;

    std::unordered_map<
        int, std::vector<std::shared_ptr<MoeCachedSource>>>
        layer_sources;
    for (const auto & source : sources_) {
        if (source->layer_id() < 0) continue;
        layer_sources[source->layer_id()].push_back(source);
    }

    const auto started = std::chrono::steady_clock::now();
    std::unordered_map<
        MoeCachedSource *, std::vector<int32_t>>
        source_experts;
    for (auto selected_it = prewarm_selected_.rbegin();
         selected_it != prewarm_selected_.rend();
         ++selected_it) {
        for (const auto & source :
             layer_sources.at(selected_it->layer)) {
            source_experts[source.get()].push_back(
                selected_it->expert);
        }
    }
    prewarming_ = true;
    try {
        for (const auto & source : sources_) {
            const auto found = source_experts.find(source.get());
            if (found == source_experts.end()) continue;
            prepare(*source, found->second, true);
        }
        if (transfer_ready_recorded_) {
            MFQ_CUDA_CHECK(
                cudaEventSynchronize(transfer_ready_));
        }
        prewarming_ = false;
    } catch (...) {
        prewarming_ = false;
        throw;
    }
    for (auto selected_it = prewarm_selected_.rbegin();
         selected_it != prewarm_selected_.rend();
         ++selected_it) {
        for (const auto & source :
             layer_sources.at(selected_it->layer)) {
            source->touch_expert(
                selected_it->expert);
        }
    }
    const auto stopped = std::chrono::steady_clock::now();
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(
            stopped - started).count();
    const int64_t prewarm_h2d_bytes = stats_.h2d_bytes;
    const int64_t prewarm_range_read_bytes = stats_.range_read_bytes;
    const int64_t prewarm_range_read_calls = stats_.range_read_calls;
    const int64_t prewarm_range_file_opens = stats_.range_file_opens;
    const double prewarm_range_read_ms =
        static_cast<double>(stats_.range_read_nanoseconds) / 1.0e6;
    const int64_t projection_entries =
        stats_.prefetch_misses;
    const int64_t unfilled_bytes =
        std::max<int64_t>(
            0, allocated_bytes_ - prewarm_h2d_bytes);
    std::cout
        << "moe_cache_prewarm"
        << " profile=" << profile_->path
        << " expert_bundles=" << prewarm_selected_.size()
        << " projection_entries=" << projection_entries
        << " h2d_bytes=" << prewarm_h2d_bytes
        << " range_read_bytes=" << prewarm_range_read_bytes
        << " range_read_calls=" << prewarm_range_read_calls
        << " range_file_opens=" << prewarm_range_file_opens
        << " range_read_ms=" << prewarm_range_read_ms
        << " time_ms=" << elapsed_ms
        << " unfilled_bytes=" << unfilled_bytes
        << "\n";
    stats_ = {};
}

void MoeExpertCache::invalidate(
        const mfq::MoeCacheKey & key,
        int slot) {
    if (key.source < 0 ||
        key.source >= static_cast<int>(sources_.size())) {
        throw std::runtime_error(
            "MoE cache eviction references an invalid source");
    }
    sources_.at(static_cast<size_t>(key.source))
        ->invalidate(key.cohort, key.expert, slot);
}

void MoeExpertCache::append_source_transfers(
        MoeCachedSource & source,
        const std::vector<int32_t> & experts,
        bool prefetch,
        std::vector<MoeCacheTransfer> & transfers,
        bool & replaced_occupied,
        std::vector<std::pair<mfq::MoeCacheSlotBook *, int>> * held_slots,
        std::vector<MoeCacheNewLease> * new_leases) {
    for (int expert : experts) {
        const int cohort_index =
            source.expert_to_cohort_.at(
                static_cast<size_t>(expert));
        const int local_index =
            source.expert_to_local_.at(
                static_cast<size_t>(expert));
        auto & cohort =
            source.cohorts_.at(
                static_cast<size_t>(cohort_index));
        auto & arena = *cohort.arena;
        const mfq::MoeCacheKey key{
            source.id_, cohort_index, expert};
        auto lease = arena.book->acquire(key);
        if (held_slots != nullptr &&
                !arena.book->inflight(lease.slot)) {
            arena.book->mark_inflight(lease.slot);
            held_slots->emplace_back(arena.book.get(), lease.slot);
        }
        if (lease.hit) {
            if (prefetch) {
                ++stats_.prefetch_hits;
            } else {
                ++stats_.demand_hits;
            }
        } else {
            if (new_leases != nullptr) {
                new_leases->push_back({
                    arena.book.get(),
                    key,
                    lease.slot,
                    lease.generation,
                });
            }
            if (prefetch) {
                ++stats_.prefetch_misses;
            } else {
                ++stats_.demand_misses;
            }
            if (lease.replaced.has_value()) {
                ++stats_.evictions;
                replaced_occupied = true;
                invalidate(*lease.replaced, lease.slot);
            }
            for (size_t field = 0;
                 field < cohort.bytes_per_expert.size();
                 ++field) {
                const int64_t nbytes =
                    cohort.bytes_per_expert[field];
                if (nbytes == 0) continue;
                auto & gpu_field =
                    arena.fields[field];
                if (cohort.range_store) {
                    const auto & part =
                        cohort.range_store->part(expert, field);
                    transfers.push_back({
                        nullptr,
                        reinterpret_cast<uint8_t *>(
                            gpu_field.data_ptr()) +
                            static_cast<int64_t>(lease.slot) * nbytes,
                        nbytes,
                        true,
                        nullptr,
                        cohort.range_store.get(),
                        &part,
                    });
                    continue;
                }
                const auto & cpu_field =
                    cohort.cpu_fields[field];
                transfers.push_back({
                    reinterpret_cast<const uint8_t *>(
                        cpu_field.data_ptr()) +
                        static_cast<int64_t>(local_index) *
                        nbytes,
                    reinterpret_cast<uint8_t *>(
                        gpu_field.data_ptr()) +
                        static_cast<int64_t>(lease.slot) *
                        nbytes,
                    nbytes,
                    true,
                    cohort.mapped_fields.at(field) == nullptr
                        ? nullptr
                        : cohort.mapped_fields.at(field) +
                            static_cast<int64_t>(local_index) * nbytes,
                });
            }
        }
        if (cohort.host_map[
                static_cast<size_t>(expert)] != lease.slot) {
            cohort.host_map[
                static_cast<size_t>(expert)] = lease.slot;
            cohort.map_dirty = true;
        }
    }

    for (auto & cohort : source.cohorts_) {
        if (!cohort.map_dirty) continue;
        auto & active =
            source.active_->pools.at(
                static_cast<size_t>(cohort.index));
        transfers.push_back({
            reinterpret_cast<const uint8_t *>(
                cohort.host_map.data()),
            reinterpret_cast<uint8_t *>(
                active.expert_local.data_ptr<int32_t>()),
            static_cast<int64_t>(
                cohort.host_map.size() *
                sizeof(int32_t)),
            false,
        });
        cohort.map_dirty = false;
    }
}

void MoeExpertCache::rollback_preparation(
        const std::vector<MoeCacheNewLease> & new_leases,
        const std::vector<
            std::pair<mfq::MoeCacheSlotBook *, int>> & held_slots) noexcept {
    (void)cudaStreamSynchronize(weight_stream_);
    for (auto lease = new_leases.rbegin();
         lease != new_leases.rend();
         ++lease) {
        try {
            invalidate(lease->key, lease->slot);
            (void)lease->book->discard(
                lease->key, lease->slot, lease->generation);
        } catch (...) {
        }
    }
    for (const auto & held : held_slots) {
        try {
            held.first->clear_inflight(held.second);
        } catch (...) {
        }
    }
}

bool MoeExpertCache::begin_deferred_range_read(
        MoeCachedSource & source,
        const std::vector<int32_t> & experts) {
    if (!range_overlap_enabled_ || prewarming_ || !source.range_store_ ||
            experts.empty()) {
        return false;
    }
    finish_deferred_range_read(source);

    std::unordered_map<
        MoeGpuArena *,
        std::unordered_set<mfq::MoeCacheKey, mfq::MoeCacheKeyHash>>
        arena_demands;
    for (int expert : experts) {
        if (expert < 0 || expert >= source.n_experts()) return false;
        const int cohort_index = source.expert_to_cohort_.at(
            static_cast<size_t>(expert));
        auto & cohort = source.cohorts_.at(
            static_cast<size_t>(cohort_index));
        arena_demands[cohort.arena].insert(
            {source.id_, cohort_index, expert});
    }
    for (const auto & item : arena_demands) {
        if (item.second.size() >
                static_cast<size_t>(item.first->book->capacity())) {
            return false;
        }
    }

    auto pending = std::make_unique<MoePendingRangeRead>();
    for (const auto & item : arena_demands) {
        auto * book = item.first->book.get();
        for (const auto & key : item.second) {
            const int slot = book->slot_for(key);
            if (slot >= 0 && !book->inflight(slot)) {
                book->mark_inflight(slot);
                pending->held_slots.emplace_back(book, slot);
            }
        }
    }

    try {
        append_source_transfers(
            source,
            experts,
            true,
            pending->transfers,
            pending->replaced_occupied,
            &pending->held_slots,
            &pending->new_leases);

        int64_t range_bytes = 0;
        size_t range_count = 0;
        for (const auto & transfer : pending->transfers) {
            if (transfer.range_store == nullptr) continue;
            range_bytes = (range_bytes + 15) & ~int64_t{15};
            if (transfer.nbytes >
                    std::numeric_limits<int64_t>::max() - range_bytes) {
                throw std::overflow_error(
                    "deferred MoE range byte count overflows int64");
            }
            range_bytes += transfer.nbytes;
            ++range_count;
        }
        if (range_count == 0) {
            submit_transfers(
                pending->transfers,
                pending->replaced_occupied,
                false);
            for (const auto & held : pending->held_slots) {
                held.first->clear_inflight(held.second);
            }
            return true;
        }

        pending->host = mfq_tensor_backend::empty(
            {range_bytes},
            mfq_tensor_backend::TensorOptions()
                .device(mfq_tensor_backend::kCPU)
                .dtype(mfq_tensor_backend::kUInt8)
                .pinned_memory(true));
        auto * staging = pending->host.data_ptr<uint8_t>();
        std::vector<mfq::cuda::MfeMxfp4ReadRequest> requests;
        requests.reserve(range_count);
        int64_t offset = 0;
        for (auto & transfer : pending->transfers) {
            if (transfer.range_store == nullptr) continue;
            offset = (offset + 15) & ~int64_t{15};
            requests.push_back({
                transfer.range_store,
                transfer.range_part,
                std::span<uint8_t>(
                    staging + offset,
                    static_cast<size_t>(transfer.nbytes)),
            });
            transfer.source = staging + offset;
            transfer.range_store = nullptr;
            transfer.range_part = nullptr;
            offset += transfer.nbytes;
        }
        pending->ticket = range_read_pool_->submit(requests);
        const auto inserted = pending_range_reads_.try_emplace(
            source.id_, std::move(pending));
        if (!inserted.second) {
            throw std::runtime_error(
                "MoE source already has a deferred range read");
        }
        ++stats_.range_overlap_batches;
        return true;
    } catch (...) {
        if (pending) {
            rollback_preparation(
                pending->new_leases, pending->held_slots);
        }
        throw;
    }
}

void MoeExpertCache::finish_deferred_range_read(
        MoeCachedSource & source) {
    const auto found = pending_range_reads_.find(source.id_);
    if (found == pending_range_reads_.end()) return;
    auto pending = std::move(found->second);
    pending_range_reads_.erase(found);
    try {
        const auto wait_begin = std::chrono::steady_clock::now();
        const auto range_stats = pending->ticket.wait();
        stats_.range_overlap_wait_nanoseconds +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - wait_begin).count();
        record_range_read(range_stats);
        submit_transfers(
            pending->transfers,
            pending->replaced_occupied,
            true);
    } catch (...) {
        rollback_preparation(
            pending->new_leases, pending->held_slots);
        throw;
    }
    for (const auto & held : pending->held_slots) {
        held.first->clear_inflight(held.second);
    }
}

bool MoeExpertCache::prepare(
        MoeCachedSource & source,
        const std::vector<int32_t> & experts,
        bool prefetch) {
    if (!finalized_) {
        throw std::runtime_error(
            "MoE cache must be finalized before inference");
    }
    finish_deferred_range_read(source);
    if (experts.empty()) return false;

    std::unordered_map<
        MoeGpuArena *,
        std::unordered_set<mfq::MoeCacheKey, mfq::MoeCacheKeyHash>>
        arena_demands;
    for (int expert : experts) {
        if (expert < 0 || expert >= source.n_experts()) return false;
        const int cohort_index = source.expert_to_cohort_.at(
            static_cast<size_t>(expert));
        auto & cohort = source.cohorts_.at(
            static_cast<size_t>(cohort_index));
        arena_demands[cohort.arena].insert(
            {source.id_, cohort_index, expert});
    }
    for (const auto & item : arena_demands) {
        if (item.second.size() >
                static_cast<size_t>(item.first->book->capacity())) {
            return false;
        }
    }

    std::vector<std::pair<mfq::MoeCacheSlotBook *, int>> held_slots;
    for (const auto & item : arena_demands) {
        auto * book = item.first->book.get();
        for (const auto & key : item.second) {
            const int slot = book->slot_for(key);
            if (slot >= 0 && !book->inflight(slot)) {
                book->mark_inflight(slot);
                held_slots.emplace_back(book, slot);
            }
        }
    }
    std::vector<MoeCacheTransfer> transfers;
    std::vector<MoeCacheNewLease> new_leases;
    bool replaced_occupied = false;
    try {
        append_source_transfers(
            source, experts, prefetch, transfers,
            replaced_occupied, &held_slots, &new_leases);
        submit_transfers(
            transfers,
            replaced_occupied,
            !prefetch);
    } catch (...) {
        rollback_preparation(new_leases, held_slots);
        throw;
    }
    for (const auto & held : held_slots) {
        held.first->clear_inflight(held.second);
    }
    return true;
}

bool MoeExpertCache::prepare_bundle(
        const std::vector<MoeCachedSource *> & sources,
        const std::vector<int32_t> & experts) {
    if (!finalized_) {
        throw std::runtime_error(
            "MoE cache must be finalized before inference");
    }
    for (auto * source : sources) {
        if (source != nullptr && source->cache_ == this) {
            finish_deferred_range_read(*source);
        }
    }
    if (sources.empty() || experts.empty()) return false;

    std::unordered_map<
        MoeGpuArena *,
        std::unordered_set<mfq::MoeCacheKey, mfq::MoeCacheKeyHash>>
        arena_demands;
    for (auto * source : sources) {
        if (source == nullptr || source->cache_ != this) return false;
        for (int expert : experts) {
            if (expert < 0 || expert >= source->n_experts()) return false;
            const int cohort_index = source->expert_to_cohort_.at(
                static_cast<size_t>(expert));
            auto & cohort = source->cohorts_.at(
                static_cast<size_t>(cohort_index));
            arena_demands[cohort.arena].insert(
                {source->id_, cohort_index, expert});
        }
    }
    for (const auto & item : arena_demands) {
        if (item.second.size() >
                static_cast<size_t>(item.first->book->capacity())) {
            return false;
        }
    }

    std::vector<std::pair<mfq::MoeCacheSlotBook *, int>> held_slots;
    for (const auto & item : arena_demands) {
        auto * book = item.first->book.get();
        for (const auto & key : item.second) {
            const int slot = book->slot_for(key);
            if (slot >= 0 && !book->inflight(slot)) {
                book->mark_inflight(slot);
                held_slots.emplace_back(book, slot);
            }
        }
    }
    std::vector<MoeCacheTransfer> transfers;
    std::vector<MoeCacheNewLease> new_leases;
    bool replaced_occupied = false;
    try {
        for (auto * source : sources) {
            append_source_transfers(
                *source, experts, true, transfers,
                replaced_occupied, &held_slots, &new_leases);
        }
        submit_transfers(
            transfers, replaced_occupied, false);
    } catch (...) {
        rollback_preparation(new_leases, held_slots);
        throw;
    }
    for (const auto & held : held_slots) {
        held.first->clear_inflight(held.second);
    }
    return true;
}

bool MoeExpertCache::prepare_bundle_deferred(
        const std::vector<MoeCachedSource *> & ready_sources,
        MoeCachedSource & deferred_source,
        const std::vector<int32_t> & experts) {
    if (!range_overlap_enabled_ || !deferred_source.range_store_) {
        auto sources = ready_sources;
        sources.push_back(&deferred_source);
        return prepare_bundle(sources, experts);
    }
    if (!prepare_bundle(ready_sources, experts)) return false;
    return begin_deferred_range_read(deferred_source, experts);
}

static MfeWeight wrap_cached_moe_source(
        const std::shared_ptr<MoeCachedSource> & source,
        const std::shared_ptr<MixedMoeRuntime> & cpu) {
    MfeWeight result =
        cpu_mixed_moe_metadata(cpu);
    result.mixed_weight_bytes = source->logical_weight_bytes();
    result.cached_source = source;
    result.cache_prefetch = [source](
            const MoeRoutePlan & route) {
        source->prefetch(route);
    };
    result.cache_prefetch_begin = [source](
            const MoeRoutePlan & route) {
        source->begin_prefetch(route);
    };
    result.mixed_forward = [source](
            mfq_tensor_backend::Tensor x,
            const MoeRoutePlan & route) {
        return source->forward(x, route);
    };
    result.mixed_prequantized_forward = [source](
            mfq_tensor_backend::Tensor x,
            const MoeRoutePlan & route) {
        return source->forward_prequantized(x, route);
    };
    result.mixed_glu_output_forward = [source](
            mfq_tensor_backend::Tensor x,
            const MoeRoutePlan & route,
            bool gelu) {
        return source->forward_glu_output(
            x, route, gelu);
    };
    result.mixed_glu_forward = [source](
            mfq_tensor_backend::Tensor gate_up,
            const MoeRoutePlan & route,
            bool gelu) {
        return source->forward_glu(
            gate_up, route, gelu);
    };
    if (source->supports_clamped_swiglu()) {
        result.mixed_clamped_swiglu_forward = [source](
                mfq_tensor_backend::Tensor gate_up,
                const MoeRoutePlan & route,
                double limit) {
            return source->forward_clamped_swiglu(
                gate_up, route, limit);
        };
    }
    return result;
}

bool prefetch_cached_moe_projection_bundle(
        const MfeWeight & gate,
        const MfeWeight & up,
        const MfeWeight & down,
        const MoeRoutePlan & route) {
    if (!gate.cached_source ||
            !up.cached_source ||
            !down.cached_source) {
        return false;
    }
    return MoeCachedSource::prefetch_bundle(
        {gate.cached_source, up.cached_source, down.cached_source},
        route);
}

bool prefetch_cached_moe_projection_bundle(
        const MfeWeight & gate_up,
        const MfeWeight & down,
        const MoeRoutePlan & route) {
    if (!gate_up.cached_source || !down.cached_source) {
        return false;
    }
    return MoeCachedSource::prefetch_bundle(
        {gate_up.cached_source, down.cached_source}, route);
}

MfeWeight load_mfe_gpu(
        const mfq::ModelSource & mfq, const std::string & name,
        bool cacheable,
        int layer_id,
        const std::string & projection_role) {
    const char * disable_ranges =
        std::getenv("MFQ_DISABLE_MOE_SSD_RANGES");
    if (g_moe_expert_cache && cacheable &&
            !moe_parallel_config().enabled() &&
            (disable_ranges == nullptr || std::atoi(disable_ranges) == 0)) {
        const auto & record = require_tensor(mfq, name);
        try {
            auto store =
                std::make_shared<mfq::cuda::MfeMxfp4ExpertStore>(
                    mfq::cuda::MfqRecordRange{
                        name,
                        record.dtype,
                        {},
                        0,
                        record.nbytes,
                        [&mfq, name](
                                std::uint64_t offset,
                                std::span<std::uint8_t> destination) {
                            mfq.read_range_into(
                                name,
                                offset,
                                reinterpret_cast<std::byte*>(destination.data()),
                                destination.size());
                        },
                    });
            auto runtime = make_mxfp4_range_runtime(*store);
            auto source = g_moe_expert_cache->register_range_source(
                name,
                runtime,
                std::move(store),
                std::min(
                    g_moe_cache_registration_min_slots,
                    runtime->n_experts),
                layer_id,
                projection_role);
            return wrap_cached_moe_source(source, runtime);
        } catch (const mfq::cuda::MfeMxfp4Unsupported &) {
        }
    }
    auto cpu = load_mfe_cpu(mfq, name);
    const bool has_matrix_local_sq = std::any_of(
        cpu.pools.begin(), cpu.pools.end(), [](const MfeCpuPool & pool) {
            return pool.dtype == "MXFP4-SQ" ||
                mfq::fp8sq::is_dtype(pool.dtype);
        });
    if (moe_parallel_config().enabled()) {
        auto slices = plan_moe_expert_parallel_slices(
            cpu.n_experts, name);
        MfeWeight result;
        result.n_experts = cpu.n_experts;
        result.out_per_expert =
            cpu.out_per_expert;
        result.neuron_len =
            cpu.neuron_len;
        for (const auto & slice : slices) {
            auto shard =
                std::make_shared<MfeWeight>(
                    to_cuda_device_moe_expert_slice(
                        cpu, slice.begin,
                        slice.end,
                        slice.device));
            result.expert_parallel_shards.push_back({
                slice.device,
                slice.begin,
                slice.end,
                std::move(shard),
            });
        }
        return result;
    }
    if (g_moe_expert_cache && cacheable && !has_matrix_local_sq) {
        auto runtime =
            make_mixed_moe_runtime(cpu, false);
        auto source = g_moe_expert_cache->register_source(
            name, runtime,
            std::min(
                g_moe_cache_registration_min_slots,
                runtime->n_experts),
            layer_id,
            projection_role);
        return wrap_cached_moe_source(source, runtime);
    }
    const bool all_nint = std::all_of(
        cpu.pools.begin(), cpu.pools.end(), [](const MfeCpuPool & pool) {
            return pool.dtype == "NINT";
        });
    return all_nint ? to_gpu_mfe(cpu) : to_gpu_mixed_moe(cpu);
}

std::shared_ptr<MixedMoeRuntime> load_mfe_cpu_offloaded(
        const mfq::ModelSource & mfq, const std::string & name) {
    return make_mixed_moe_runtime(load_mfe_cpu(mfq, name), false);
}

static NvqWeight load_nvq_gpu(const mfq::ModelSource & mfq, const std::string & name) {
    const auto & rec = require_tensor(mfq, name);
    return to_gpu_nvq(unpack_nvq(read_tensor(mfq, name), rec.dtype));
}

static mfq_tensor_backend::Tensor nvq_dequant(const NvqWeight & w) {
    return nvq_dequant_cuda(
        w.indices_packed, w.aux_packed, w.sub_scale_packed,
        w.neuron_scale, w.codebook, w.neuron_len, w.gs,
        w.sub_bits, w.kernel_format, w.sign_mode);
}

static mfq_tensor_backend::Tensor dequant_nint8_zero_cpu(const Nint8ZeroCpu & source) {
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

static mfq_tensor_backend::Tensor dequant_nint_dense_f32(
    const NintWeight & weight);

mfq_tensor_backend::Tensor load_dense_gpu(const mfq::ModelSource & mfq, const std::string & name) {
    MfqCudaGuard guard(
        active_weight_load_device());
    const auto & rec = require_tensor(mfq, name);
    auto blob = read_tensor(mfq, name);
    if (rec.dtype == "NINT8-0") {
        const auto source = unpack_nint8_zero(blob);
        if (g_loading_cpu_layer) {
            return dequant_nint8_zero_cpu(source);
        }
        const auto packed = to_gpu_nint8_zero(source);
        auto dense = nint8_zero_dequant_cuda(
            packed.q_packed,
            packed.q8_zero_scale,
            packed.neuron_len)
            .to(mfq_tensor_backend::kFloat32)
            .contiguous();
        if (packed.shape.size() != 2 ||
            dense.size(0) != packed.shape[0] ||
            dense.size(1) != packed.shape[1]) {
            throw std::runtime_error(
                "NINT8-0 dense tensor shape mismatch: " + name);
        }
        return dense;
    }
    if (rec.dtype == "NINT") {
        auto dense = dequant_nint_dense_f32(load_nint_gpu(mfq, name));
        return g_loading_cpu_layer
            ? dense.cpu().contiguous()
            : dense;
    }
    if (rec.dtype != "F32" && rec.dtype != "BF16" &&
            rec.dtype != "F16" && rec.dtype != "I64" &&
            rec.dtype != "I32") {
        throw std::runtime_error(
            "unsupported dense dtype for C++ runtime: " + rec.dtype +
            " tensor " + name);
    }
    size_t off = 0;
    uint32_t ndim = read_u32_from(blob, off);
    std::vector<int64_t> shape(ndim);
    int64_t numel = 1;
    for (uint32_t i = 0; i < ndim; ++i) {
        shape[i] = read_i64_from(blob, off);
        numel *= shape[i];
    }
    mfq_tensor_backend::Tensor t;
    if (rec.dtype == "F32") {
        t = mfq_tensor_backend::from_blob(blob.data() + off, shape, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kFloat32)).clone();
    } else if (rec.dtype == "BF16") {
        t = mfq_tensor_backend::from_blob(blob.data() + off, shape, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kBFloat16)).clone().to(mfq_tensor_backend::kFloat32);
    } else if (rec.dtype == "F16") {
        t = mfq_tensor_backend::from_blob(blob.data() + off, shape, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kFloat16)).clone().to(mfq_tensor_backend::kFloat32);
    } else if (rec.dtype == "I64") {
        t = mfq_tensor_backend::from_blob(blob.data() + off, shape, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64)).clone();
    } else if (rec.dtype == "I32") {
        t = mfq_tensor_backend::from_blob(blob.data() + off, shape, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt32)).clone();
    } else {
        throw std::runtime_error("unsupported dense dtype for C++ runtime: " + rec.dtype + " tensor " + name);
    }
    (void)numel;
    return g_loading_cpu_layer
        ? t.contiguous()
        : t.to(mfq_tensor_backend::kCUDA).contiguous();
}

static mfq_tensor_backend::Tensor load_dense_linear_cpu(
        const mfq::ModelSource & mfq,
        const std::string & name) {
    const auto & rec = require_tensor(mfq, name);
    auto blob = read_tensor(mfq, name);
    size_t off = 0;
    const uint32_t ndim = read_u32_from(blob, off);
    std::vector<int64_t> shape(ndim);
    for (uint32_t index = 0; index < ndim; ++index) {
        shape[index] = read_i64_from(blob, off);
    }
    if (shape.size() != 2) {
        throw std::runtime_error(
            "dense linear tensor must be rank 2: " + name);
    }
    mfq_tensor_backend::Tensor value;
    if (rec.dtype == "BF16") {
        value = mfq_tensor_backend::from_blob(
            blob.data() + off, shape,
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kBFloat16)).clone();
    } else if (rec.dtype == "F16") {
        value = mfq_tensor_backend::from_blob(
            blob.data() + off, shape,
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kFloat16)).clone();
    } else if (rec.dtype == "F32") {
        value = mfq_tensor_backend::from_blob(
            blob.data() + off, shape,
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kFloat32)).clone();
    } else {
        throw std::runtime_error(
            "unsupported dense linear dtype: " + rec.dtype +
            " tensor " + name);
    }
    return value.contiguous();
}

static mfq_tensor_backend::Tensor load_dense_linear_gpu(
        const mfq::ModelSource & mfq,
        const std::string & name) {
    MfqCudaGuard guard(active_weight_load_device());
    return load_dense_linear_cpu(mfq, name)
        .to(mfq_tensor_backend::kCUDA).contiguous();
}

static NintWeight cat_weights(const std::vector<NintWeight> & ws) {
    if (ws.empty()) throw std::runtime_error("empty NINT group");
    const auto & a = ws[0];
    std::vector<mfq_tensor_backend::Tensor> qp, rqb, rqoff, q8s, ss, sm, ns, nm;
    int64_t out = 0;
    int64_t q_bit_base = 0;
    bool aligned_q8 = !a.q8_zero;
    int uniform_q46 = a.q8_zero ? 0 : a.uniform_q46;
    for (const auto & w : ws) {
        if (w.ng != a.ng || w.gs != a.gs ||
            w.neuron_len != a.neuron_len || w.q8_zero != a.q8_zero) {
            throw std::runtime_error("cannot group NINT tensors with different input layout");
        }
        qp.push_back(w.q_packed);
        if (!w.q8_zero) {
            aligned_q8 = aligned_q8 && w.aligned_q8 && (q_bit_base & 31) == 0;
            if (w.uniform_q46 != uniform_q46 || (q_bit_base & 7) != 0) {
                uniform_q46 = 0;
            }
            rqb.push_back(w.row_q_bits);
            rqoff.push_back(w.row_q_bit_offsets + q_bit_base);
            q_bit_base += w.q_packed.numel() * 8;
        }
        if (w.q8_zero) {
            q8s.push_back(w.q8_zero_scale);
            out += w.out;
            continue;
        }
        ss.push_back(w.sub_scale);
        sm.push_back(w.sub_min);
        ns.push_back(w.neuron_scale);
        nm.push_back(w.neuron_min);
        out += w.out;
    }
    NintWeight g;
    g.out = out;
    g.ng = a.ng;
    g.gs = a.gs;
    g.bits = a.bits;
    g.neuron_len = a.neuron_len;
    g.q8_zero = a.q8_zero;
    g.aligned_q8 = aligned_q8;
    g.uniform_q46 = uniform_q46;
    g.shape = a.shape;
    g.shape[0] = out;
    g.q_packed = mfq_tensor_backend::cat(qp, 0).contiguous();
    if (!a.q8_zero) {
        g.row_q_bits = mfq_tensor_backend::cat(rqb, 0).contiguous();
        g.row_q_bit_offsets = mfq_tensor_backend::cat(rqoff, 0).contiguous();
    }
    if (a.q8_zero) {
        g.q8_zero_scale = mfq_tensor_backend::cat(q8s, 0).contiguous();
        return g;
    }
    g.sub_scale = mfq_tensor_backend::cat(ss, 0).contiguous();
    g.sub_min = mfq_tensor_backend::cat(sm, 0).contiguous();
    g.neuron_scale = mfq_tensor_backend::cat(ns, 0).contiguous();
    g.neuron_min = mfq_tensor_backend::cat(nm, 0).contiguous();
    return g;
}

static mfq_tensor_backend::Tensor pad_last(mfq_tensor_backend::Tensor x, int64_t target) {
    if (x.size(1) == target) return x;
    if (x.size(1) > target) throw std::runtime_error("activation width exceeds neuron_len");
    return mfq_tensor_backend::constant_pad_nd(x, {0, target - x.size(1)}, 0);
}

enum class NvqMatmulPath {
    Gemv,
    Mmq,
    OnlineF16,
    DequantGemm,
};

static NvqMatmulPath select_nvq_matmul_path(const NvqWeight & w, int M) {
    const bool e8_family =
        w.format == 2 || w.format == 5 || w.format == 7 ||
        w.format == 8 || w.format == 9 ||
        w.format == 13 || w.format == 14;

    if (w.format == 8 && M >= 14 && M <= 16 && w.neuron_len >= 2 * w.out) {
        return NvqMatmulPath::DequantGemm;
    }

    if (w.format == 9 && M >= 13) {
        const bool wide_output = w.out * 8 >= w.neuron_len * 21;
        const bool wide_mmq =
            w.out >= 4096 && (w.out >= 2 * w.neuron_len || w.neuron_len >= 2 * w.out);
        if (M <= 15) {
            if (wide_mmq) return NvqMatmulPath::Mmq;
            return w.out >= 1024 ? NvqMatmulPath::Gemv : NvqMatmulPath::DequantGemm;
        }
        if (M == 16) {
            if (w.out >= 4096) return NvqMatmulPath::Mmq;
            return w.out >= 1024 ? NvqMatmulPath::Gemv : NvqMatmulPath::DequantGemm;
        }
        if (M <= 31) {
            if (wide_output) return NvqMatmulPath::OnlineF16;
            if (w.out >= 4096 && w.out >= w.neuron_len) return NvqMatmulPath::Mmq;
            return NvqMatmulPath::DequantGemm;
        }
        if (M == 32) {
            return w.out >= 4096 ? NvqMatmulPath::Mmq : NvqMatmulPath::DequantGemm;
        }
        if (M <= 47) {
            return wide_output ? NvqMatmulPath::OnlineF16 : NvqMatmulPath::DequantGemm;
        }
        if (M == 48) {
            return wide_output ? NvqMatmulPath::Mmq : NvqMatmulPath::DequantGemm;
        }
        if (M <= 63) {
            return wide_output ? NvqMatmulPath::OnlineF16 : NvqMatmulPath::DequantGemm;
        }
        if (M == 64) {
            if (wide_output) return NvqMatmulPath::OnlineF16;
            return w.out >= 8192 ? NvqMatmulPath::Mmq : NvqMatmulPath::DequantGemm;
        }
        return NvqMatmulPath::DequantGemm;
    }

    if (M <= 13) return NvqMatmulPath::Gemv;
    if (M == 14) {
        return w.out >= 2048 ? NvqMatmulPath::Gemv : NvqMatmulPath::DequantGemm;
    }
    if (M == 15) {
        if (e8_family && w.out >= 8192) return NvqMatmulPath::Mmq;
        if (e8_family && w.out >= 2048) return NvqMatmulPath::Gemv;
        return NvqMatmulPath::DequantGemm;
    }
    if (M == 16) {
        if (e8_family && w.out >= 6144) return NvqMatmulPath::Mmq;
        if (e8_family && w.out >= 2048) return NvqMatmulPath::Gemv;
        if ((w.format == 3 || w.format == 10 ||
             w.format == 12 || w.format == 15) &&
            w.out >= 4096 && w.neuron_len >= 8192) {
            return NvqMatmulPath::Mmq;
        }
        return NvqMatmulPath::DequantGemm;
    }

    const bool wide_expansion = e8_family && w.out >= 3 * w.neuron_len;
    if (wide_expansion && ((M >= 17 && M <= 31) || (M >= 33 && M <= 47))) {
        return NvqMatmulPath::OnlineF16;
    }
    if (M == 32 && e8_family && w.out >= 8192) return NvqMatmulPath::Mmq;
    if (M == 48 && e8_family && w.out >= 12288) return NvqMatmulPath::Mmq;
    if (M == 64 && (w.format == 7 || w.format == 9) && w.out >= 8192) {
        return NvqMatmulPath::Mmq;
    }
    return NvqMatmulPath::DequantGemm;
}

static inline uint32_t cpu_load_packed_bits(
        const uint8_t * data,
        int64_t nbytes,
        int64_t bit,
        int bits) {
    const int64_t byte = bit >> 3;
    const int shift = static_cast<int>(bit & 7);
    uint32_t word = data[byte];
    if (byte + 1 < nbytes) word |= static_cast<uint32_t>(data[byte + 1]) << 8;
    if (byte + 2 < nbytes) word |= static_cast<uint32_t>(data[byte + 2]) << 16;
    return (word >> shift) & ((1u << bits) - 1u);
}

#ifdef MFQ_CPU_X86_GNU
__attribute__((target("avx512f,avx512bw,avx512vnni")))
static int32_t cpu_dot_s8_s8_vnni(
        const int8_t * left,
        const int8_t * right,
        int32_t right_sum) {
    const __m512i signed_a = _mm512_loadu_si512(
        reinterpret_cast<const __m512i *>(left));
    const __m512i a = _mm512_xor_si512(
        signed_a, _mm512_set1_epi8(static_cast<char>(-128)));
    const __m512i b = _mm512_loadu_si512(
        reinterpret_cast<const __m512i *>(right));
    const __m512i dot = _mm512_dpbusd_epi32(_mm512_setzero_si512(), a, b);
    return _mm512_reduce_add_epi32(dot) - 128 * right_sum;
}

__attribute__((target("avx512f,avx512bw,avx512vnni")))
static int32_t cpu_dot_u8_s8_vnni(
        const uint8_t * left,
        const int8_t * right) {
    const __m512i a = _mm512_loadu_si512(
        reinterpret_cast<const __m512i *>(left));
    const __m512i b = _mm512_loadu_si512(
        reinterpret_cast<const __m512i *>(right));
    const __m512i dot = _mm512_dpbusd_epi32(_mm512_setzero_si512(), a, b);
    return _mm512_reduce_add_epi32(dot);
}

static bool cpu_has_avx512_vnni() {
    static const bool supported = []() {
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx512f") &&
            __builtin_cpu_supports("avx512bw") &&
            __builtin_cpu_supports("avx512vnni");
    }();
    return supported;
}
#endif

static inline int32_t cpu_dot_s8_s8_64(
        const int8_t * left,
        const int8_t * right,
        int32_t right_sum) {
#ifdef MFQ_CPU_X86_GNU
    if (cpu_has_avx512_vnni()) {
        return cpu_dot_s8_s8_vnni(left, right, right_sum);
    }
#endif
    int32_t result = 0;
    for (int index = 0; index < 64; ++index) {
        result += static_cast<int32_t>(left[index]) *
            static_cast<int32_t>(right[index]);
    }
    return result;
}

static inline int32_t cpu_dot_u8_s8_64(
        const uint8_t * left,
        const int8_t * right) {
#ifdef MFQ_CPU_X86_GNU
    if (cpu_has_avx512_vnni()) return cpu_dot_u8_s8_vnni(left, right);
#endif
    int32_t result = 0;
    for (int index = 0; index < 64; ++index) {
        result += static_cast<int32_t>(left[index]) *
            static_cast<int32_t>(right[index]);
    }
    return result;
}

static inline void cpu_unpack_nint_group(
        const uint8_t * packed,
        int bits,
        int valid,
        uint8_t * values) {
    std::fill(values, values + 64, static_cast<uint8_t>(0));
    if (bits == 8) {
        std::memcpy(values, packed, static_cast<size_t>(valid));
        return;
    }
    const uint32_t mask = (1u << bits) - 1u;
    uint32_t reservoir = 0;
    int available = 0;
    int source = 0;
    for (int index = 0; index < valid; ++index) {
        while (available < bits) {
            reservoir |= static_cast<uint32_t>(packed[source++]) << available;
            available += 8;
        }
        values[index] = static_cast<uint8_t>(reservoir & mask);
        reservoir >>= bits;
        available -= bits;
    }
}

static inline void cpu_unpack_nint_group_at_bit_offset(
        const uint8_t * packed,
        uint64_t bit_offset,
        int bits,
        int valid,
        uint8_t * values) {
    std::fill(values, values + 64, static_cast<uint8_t>(0));
    const uint32_t mask = bits == 8 ? 255u : ((1u << bits) - 1u);
    for (int index = 0; index < valid; ++index) {
        const uint64_t value_bit =
            bit_offset + static_cast<uint64_t>(index) * static_cast<uint64_t>(bits);
        const size_t byte = static_cast<size_t>(value_bit >> 3);
        const int shift = static_cast<int>(value_bit & 7u);
        const uint32_t pair = static_cast<uint32_t>(packed[byte]) |
            (static_cast<uint32_t>(packed[byte + 1]) << 8);
        values[index] = static_cast<uint8_t>((pair >> shift) & mask);
    }
}

struct CpuQuantizedActivation {
    int64_t rows = 0;
    int64_t groups = 0;
    int64_t group_size = 0;
    std::vector<int8_t> values;
    std::vector<float> scales;
    std::vector<int32_t> sums;
};

static CpuQuantizedActivation cpu_quantize_activation(
        mfq_tensor_backend::Tensor x,
        int64_t width,
        int64_t group_size,
        bool with_sums) {
    MFQ_RUNTIME_CHECK(!x.is_cuda(), "CPU activation quantization requires a CPU tensor");
    x = pad_last(x.contiguous().to(mfq_tensor_backend::kFloat16), width).contiguous();
    const int64_t rows = x.size(0);
    const int64_t groups = (width + group_size - 1) / group_size;
    MFQ_RUNTIME_CHECK(group_size > 0 && group_size <= 64,
        "CPU compact GEMV group size must be in [1, 64]");
    CpuQuantizedActivation result;
    result.rows = rows;
    result.groups = groups;
    result.group_size = group_size;
    result.values.resize(static_cast<size_t>(rows * groups * 64));
    result.scales.resize(static_cast<size_t>(rows * groups));
    if (with_sums) {
        result.sums.resize(static_cast<size_t>(rows * groups));
    }
    const mfq_half * input = x.data_ptr<mfq_half>();
    mfq_parallel_for(0, rows * groups, 1, [&](int64_t begin, int64_t end) {
        for (int64_t linear = begin; linear < end; ++linear) {
            const int64_t row = linear / groups;
            const int64_t group = linear - row * groups;
            const int64_t k0 = group * group_size;
            const int64_t valid = std::min(group_size, width - k0);
            float maximum = 0.0f;
            for (int64_t index = 0; index < valid; ++index) {
                maximum = std::max(
                    maximum,
                    std::fabs(static_cast<float>(input[row * width + k0 + index])));
            }
            const float scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
            result.scales[static_cast<size_t>(linear)] = scale;
            int32_t sum = 0;
            int8_t * quantized = result.values.data() + linear * 64;
            for (int64_t index = 0; index < valid; ++index) {
                int value = static_cast<int>(std::round(
                    static_cast<float>(input[row * width + k0 + index]) / scale));
                value = std::max(-127, std::min(127, value));
                quantized[index] = static_cast<int8_t>(value);
                sum += value;
            }
            std::fill(
                quantized + valid,
                quantized + 64,
                static_cast<int8_t>(0));
            if (with_sums) {
                result.sums[static_cast<size_t>(linear)] = sum;
            }
        }
    });
    return result;
}

static float cpu_half_from_bytes(const int8_t * bytes, int64_t offset) {
    uint16_t raw = 0;
    std::memcpy(&raw, bytes + offset, sizeof(raw));
    mfq_half value;
    std::memcpy(&value, &raw, sizeof(raw));
    return static_cast<float>(value);
}

static int cpu_nvq_index_bits(int format) {
    switch (format) {
        case 1: return 11;
        case 7: return 7;
        case 8: case 12: return 9;
        case 9: return 6;
        case 13: case 15: return 10;
        case 14: return 12;
        case 2: case 3: case 5: case 10: case 11: return 8;
        default:
            throw std::runtime_error(
                "CPU dense offload does not support NVQ kernel format " +
                std::to_string(format));
    }
}

static bool cpu_nvq_d4(int format) {
    return format == 3 || format == 10 || format == 11 ||
        format == 12 || format == 15;
}

static const int8_t * cpu_nvq_codebook(
        const int8_t * metadata,
        int format,
        uint32_t state) {
    constexpr int64_t header = 64;
    if (format == 1 || format == 2 || format == 3 || format == 8) {
        return metadata;
    }
    const uint32_t bank = format == 11
        ? state & 1u
        : static_cast<uint8_t>(metadata[36 + state]);
    if (format == 5) return metadata + header + bank * 256 * 8;
    if (format == 10 || format == 11) {
        return metadata + header + bank * 256 * 4;
    }
    if (format == 12) return metadata + header + bank * 512 * 4;
    if (format == 13) return metadata + header + bank * 1024 * 8;
    if (format == 14) return metadata + header + bank * 4096 * 8;
    if (format == 15) return metadata + header + bank * 1024 * 4;
    throw std::runtime_error("unsupported CPU NVQ codebook format");
}

static float cpu_nvq_scale(
        const int8_t * metadata,
        int format,
        float anchor,
        uint32_t state) {
    if (format == 1) {
        return anchor * static_cast<float>(state) * 0.125f;
    }
    if (format == 8) {
        return anchor * static_cast<float>(state) * 0.03125f;
    }
    if (format == 2 || format == 3) {
        return anchor * static_cast<float>(state);
    }
    if (format == 11) {
        return anchor * static_cast<float>((state >> 1) + 1);
    }
    const int64_t lut_offset = (format == 7 || format == 9) ? 8 : 4;
    return anchor * cpu_half_from_bytes(
        metadata, lut_offset + static_cast<int64_t>(state) * 2);
}

static int cpu_nvq_parity7(uint32_t value) {
    value &= 0x7fu;
    value ^= value >> 4;
    value ^= value >> 2;
    value ^= value >> 1;
    return static_cast<int>(value & 1u);
}

static void cpu_decode_nvq_group(
        const NvqWeight & w,
        int row,
        int group,
        uint32_t state,
        int8_t values[64]) {
    std::fill(values, values + 64, static_cast<int8_t>(0));
    const int format = static_cast<int>(w.kernel_format);
    const bool d4 = cpu_nvq_d4(format);
    const int vector_size = d4 ? 4 : 8;
    const int nvec = static_cast<int>(
        (w.neuron_len + vector_size - 1) / vector_size);
    const int nsign = static_cast<int>((w.neuron_len + 7) / 8);
    const auto * indices = w.indices_packed.data_ptr<uint8_t>();
    const auto * aux = w.aux_packed.data_ptr<uint8_t>();
    const int bits = cpu_nvq_index_bits(format);
    const int8_t * metadata = w.codebook.data_ptr<int8_t>();
    const int8_t * bank = (format == 7 || format == 9)
        ? nullptr
        : cpu_nvq_codebook(metadata, format, state);
    for (int chunk = 0; chunk < 6; ++chunk) {
        const int vector8 = group * 3 + (chunk >> 1);
        const int vector = d4 ? group * 6 + chunk : vector8;
        if (vector >= nvec) continue;
        const int64_t index_linear =
            static_cast<int64_t>(row) * nvec + vector;
        const uint32_t code = bits == 8
            ? indices[index_linear]
            : cpu_load_packed_bits(
                indices, w.indices_packed.numel(), index_linear * bits, bits);
        int decoded[4] = {0, 0, 0, 0};
        if (format == 7) {
            constexpr int64_t header = 64;
            constexpr int64_t first_state_bytes = 8 * 4;
            constexpr int64_t second_offset = header + 8 * first_state_bytes;
            constexpr int64_t second_state_bytes = 16 * 4;
            const int8_t * source = (chunk & 1) == 0
                ? metadata + header + state * first_state_bytes +
                    (code & 7u) * 4
                : metadata + second_offset + state * second_state_bytes +
                    (code >> 3) * 4;
            for (int index = 0; index < 4; ++index) decoded[index] = source[index];
        } else if (format == 9) {
            constexpr int64_t header = 64;
            const int8_t * source = metadata + header +
                (static_cast<int64_t>(state) * 64 + code) * 8 +
                (chunk & 1) * 4;
            for (int index = 0; index < 4; ++index) decoded[index] = source[index];
        } else {
            const int8_t * source = bank +
                static_cast<int64_t>(code) * vector_size +
                (d4 ? 0 : (chunk & 1) * 4);
            for (int index = 0; index < 4; ++index) decoded[index] = source[index];
            if (format == 1 || format == 8) {
                const int64_t delta_linear =
                    static_cast<int64_t>(row) * w.ng + group;
                const bool negative = cpu_load_packed_bits(
                    aux, w.aux_packed.numel(), delta_linear, 1) != 0;
                const int delta = negative ? -1 : 1;
                if (format == 8) {
                    source = metadata +
                        static_cast<int64_t>(negative) * 512 * 8 +
                        static_cast<int64_t>(code) * 8 + (chunk & 1) * 4;
                    for (int index = 0; index < 4; ++index) {
                        decoded[index] = 32 * source[index] + 5 * delta;
                    }
                } else {
                    for (int index = 0; index < 4; ++index) {
                        decoded[index] = 8 * decoded[index] + delta;
                    }
                }
            } else {
                if (vector8 >= nsign) continue;
                const int64_t sign_linear =
                    static_cast<int64_t>(row) * nsign + vector8;
                const uint32_t mask7 = cpu_load_packed_bits(
                    aux, w.aux_packed.numel(), sign_linear * 7, 7);
                const uint32_t last =
                    static_cast<uint32_t>(cpu_nvq_parity7(mask7)) ^
                    ((format == 2 && w.sign_mode != 0)
                        ? ((code >> 7) & 1u) : 0u);
                const uint32_t mask8 = mask7 | (last << 7);
                const int sign_base = (chunk & 1) * 4;
                for (int index = 0; index < 4; ++index) {
                    if (((mask8 >> (sign_base + index)) & 1u) != 0) {
                        decoded[index] = -decoded[index];
                    }
                }
            }
        }
        const int destination = chunk * 4;
        for (int index = 0; index < 4; ++index) {
            values[destination + index] = static_cast<int8_t>(decoded[index]);
        }
    }
}

static mfq_tensor_backend::Tensor nvq_matmul_cpu(
        const NvqWeight & w,
        mfq_tensor_backend::Tensor x) {
    MFQ_RUNTIME_CHECK(!x.is_cuda(), "CPU NVQ GEMV requires CPU activations");
    MFQ_RUNTIME_CHECK(
        !w.indices_packed.is_cuda() && !w.sub_scale_packed.is_cuda() &&
        !w.neuron_scale.is_cuda() && !w.codebook.is_cuda(),
        "CPU NVQ GEMV requires CPU-resident weights");
    MFQ_RUNTIME_CHECK(w.gs == 24,
        "CPU NVQ GEMV currently requires 24-value groups");
    auto activation = cpu_quantize_activation(
        std::move(x), w.neuron_len, w.gs, true);
    const int64_t rows = activation.rows;
    const int64_t outputs = w.out;
    auto result = mfq_tensor_backend::empty(
        {rows, outputs},
        mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCPU).dtype(mfq_tensor_backend::kFloat16));
    mfq_half * output = result.data_ptr<mfq_half>();
    const uint8_t * scales = w.sub_scale_packed.data_ptr<uint8_t>();
    const float * anchors = w.neuron_scale.data_ptr<float>();
    const int8_t * metadata = w.codebook.data_ptr<int8_t>();
    const int64_t scale_bytes = w.sub_scale_packed.numel();
    mfq_parallel_for(0, outputs, 1, [&](int64_t begin, int64_t end) {
        std::vector<float> accumulators(static_cast<size_t>(rows));
        for (int64_t neuron = begin; neuron < end; ++neuron) {
            std::fill(accumulators.begin(), accumulators.end(), 0.0f);
            for (int group = 0; group < w.ng; ++group) {
                const int64_t scale_linear = neuron * w.ng + group;
                const uint32_t state = cpu_load_packed_bits(
                    scales, scale_bytes,
                    scale_linear * w.sub_bits,
                    static_cast<int>(w.sub_bits));
                const float scale = cpu_nvq_scale(
                    metadata, static_cast<int>(w.kernel_format),
                    anchors[neuron], state);
                alignas(64) int8_t weights[64];
                cpu_decode_nvq_group(
                    w, static_cast<int>(neuron), group, state, weights);
                for (int64_t row = 0; row < rows; ++row) {
                    const int8_t * quantized = activation.values.data() +
                        (row * activation.groups + group) * 64;
                    const int32_t dot = cpu_dot_s8_s8_64(
                        weights, quantized,
                        activation.sums[static_cast<size_t>(
                            row * activation.groups + group)]);
                    const float activation_scale = activation.scales[
                        static_cast<size_t>(row * activation.groups + group)];
                    accumulators[static_cast<size_t>(row)] = std::fma(
                        scale * activation_scale,
                        static_cast<float>(dot),
                        accumulators[static_cast<size_t>(row)]);
                }
            }
            for (int64_t row = 0; row < rows; ++row) {
                output[row * outputs + neuron] =
                    mfq_half(accumulators[static_cast<size_t>(row)]);
            }
        }
    });
    return result;
}

static mfq_tensor_backend::Tensor nint_matmul_cpu(
        const NintWeight & w,
        mfq_tensor_backend::Tensor x) {
    MFQ_RUNTIME_CHECK(!x.is_cuda(), "CPU NINT GEMV requires CPU activations");
    MFQ_RUNTIME_CHECK(!w.q_packed.is_cuda(), "CPU NINT GEMV requires CPU-resident weights");
    auto activation = cpu_quantize_activation(
        std::move(x), w.neuron_len, w.gs, true);
    const int64_t rows = activation.rows;
    const int64_t width = w.neuron_len;
    const int64_t outputs = w.out;
    auto result = mfq_tensor_backend::empty(
        {rows, outputs},
        mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCPU).dtype(mfq_tensor_backend::kFloat16));
    mfq_half * output = result.data_ptr<mfq_half>();
    const uint8_t * quant = w.q_packed.data_ptr<uint8_t>();
    const uint8_t * row_q_bits = w.q8_zero
        ? nullptr : w.row_q_bits.data_ptr<uint8_t>();
    const int64_t * row_q_bit_offsets = w.q8_zero
        ? nullptr : w.row_q_bit_offsets.data_ptr<int64_t>();
    mfq_parallel_for(0, outputs, 1, [&](int64_t begin, int64_t end) {
        std::vector<float> accumulators(static_cast<size_t>(rows));
        for (int64_t neuron = begin; neuron < end; ++neuron) {
            std::fill(accumulators.begin(), accumulators.end(), 0.0f);
            const int neuron_bits = w.q8_zero
                ? 8 : static_cast<int>(row_q_bits[neuron]);
            for (int64_t group = 0; group < w.ng; ++group) {
                const int64_t meta = neuron * w.ng + group;
                const uint8_t * packed = w.q8_zero
                    ? quant + meta * 32
                    : quant;
                float scale = 0.0f;
                float minimum = 0.0f;
                if (w.q8_zero) {
                    scale = static_cast<float>(
                        w.q8_zero_scale.data_ptr<mfq_half>()[meta]);
                } else {
                    scale = w.neuron_scale.data_ptr<float>()[neuron] *
                        static_cast<float>(w.sub_scale.data_ptr<uint8_t>()[meta]);
                    minimum = w.neuron_min.data_ptr<float>()[neuron] *
                        static_cast<float>(w.sub_min.data_ptr<uint8_t>()[meta]);
                }
                const int64_t valid = std::min<int64_t>(
                    w.gs, width - group * w.gs);
                alignas(64) uint8_t weights[64];
                if (w.q8_zero) {
                    std::fill(weights, weights + 64, static_cast<uint8_t>(0));
                    std::memcpy(weights, packed, static_cast<size_t>(valid));
                } else {
                    const uint64_t bit_offset =
                        static_cast<uint64_t>(row_q_bit_offsets[neuron]) +
                        static_cast<uint64_t>(group * w.gs) *
                            static_cast<uint64_t>(neuron_bits);
                    cpu_unpack_nint_group_at_bit_offset(
                        packed, bit_offset, neuron_bits,
                        static_cast<int>(valid), weights);
                }
                for (int64_t row = 0; row < rows; ++row) {
                    const int64_t activation_group =
                        row * activation.groups + group;
                    const int8_t * quantized = activation.values.data() +
                        activation_group * 64;
                    int32_t dot = 0;
                    if (w.q8_zero) {
                        dot = cpu_dot_s8_s8_64(
                            reinterpret_cast<const int8_t *>(weights), quantized,
                            activation.sums[
                                static_cast<size_t>(activation_group)]);
                        accumulators[static_cast<size_t>(row)] = std::fma(
                            scale * activation.scales[
                                static_cast<size_t>(activation_group)],
                            static_cast<float>(dot),
                            accumulators[static_cast<size_t>(row)]);
                    } else {
                        dot = cpu_dot_u8_s8_64(weights, quantized);
                        const float activation_scale = activation.scales[
                            static_cast<size_t>(activation_group)];
                        const float dot_term = scale * activation_scale *
                            static_cast<float>(dot);
                        const float min_term = minimum * activation_scale *
                            static_cast<float>(activation.sums[
                                static_cast<size_t>(activation_group)]);
                        accumulators[static_cast<size_t>(row)] +=
                            dot_term - min_term;
                    }
                }
            }
            for (int64_t row = 0; row < rows; ++row) {
                output[row * outputs + neuron] =
                    mfq_half(accumulators[static_cast<size_t>(row)]);
            }
        }
    });
    return result;
}

mfq_tensor_backend::Tensor nvq_matmul(const NvqWeight & w, mfq_tensor_backend::Tensor x) {
    if (!x.is_cuda()) return nvq_matmul_cpu(w, std::move(x));
    x = x.contiguous().to(mfq_tensor_backend::kFloat16);
    x = pad_last(x, w.neuron_len);
    int M = (int)x.size(0);
    if (g_kl_mmq_mode != KlMmqMode::Default) {
        MFQ_RUNTIME_CHECK(
            M >= 16,
            "KLD common NVQ MMQ requires at least 16 activation rows");
        x = kl_mmq_prepare_activation(x);
        ++g_kl_mmq_dense_calls;
        return g_profiler.measure("kld_mmq.nvq.fp16", [&]() {
            return nvq_gemm_f16_cuda(
                w.indices_packed, w.aux_packed, w.sub_scale_packed,
                w.neuron_scale, w.codebook, x, w.neuron_len, w.gs,
                w.sub_bits, w.kernel_format, w.sign_mode);
        });
    }
    const NvqMatmulPath path = select_nvq_matmul_path(w, M);
    if (path == NvqMatmulPath::Gemv) {
        NvqWorkspace & ws = w.workspace(M);
        return g_profiler.measure("nvq.gemv", [&]() {
            return nvq_gemv_ws_cuda(
                w.indices_packed, w.aux_packed, w.sub_scale_packed,
                w.neuron_scale, w.codebook, x, w.neuron_len, w.gs,
                w.sub_bits, w.kernel_format, w.sign_mode, ws.qx, ws.xscale);
        });
    }
    if (path == NvqMatmulPath::Mmq) {
        NvqWorkspace & ws = w.workspace(M);
        return g_profiler.measure("nvq.mma24", [&]() {
            return nvq_mmq_ws_cuda(
                w.indices_packed, w.aux_packed, w.sub_scale_packed,
                w.neuron_scale, w.codebook, x, w.neuron_len, w.gs,
                w.sub_bits, w.kernel_format, w.sign_mode, ws.qx, ws.xscale);
        });
    }
    if (path == NvqMatmulPath::OnlineF16) {
        return g_profiler.measure("nvq.gemm_online_f16", [&]() {
            return nvq_gemm_f16_cuda(
                w.indices_packed, w.aux_packed, w.sub_scale_packed,
                w.neuron_scale, w.codebook, x, w.neuron_len, w.gs,
                w.sub_bits, w.kernel_format, w.sign_mode);
        });
    }
    auto weight = g_profiler.measure("nvq.dequant", [&]() { return nvq_dequant(w); });
    return g_profiler.measure("nvq.gemm", [&]() {
        return nint_cublas_gemm_nt_f32acc_cuda(x, weight);
    });
}

mfq_tensor_backend::Tensor nvq_matmul_input_mul(
    const NvqWeight & w,
    mfq_tensor_backend::Tensor x,
    mfq_tensor_backend::Tensor gate,
    int mode) {
    MFQ_RUNTIME_CHECK(mode == 1 || mode == 2, "NVQ input gate mode must be 1(sigmoid) or 2(silu)");
    if (!x.is_cuda()) {
        x = x.contiguous().to(mfq_tensor_backend::kFloat16);
        gate = gate.contiguous().to(mfq_tensor_backend::kFloat16);
        MFQ_RUNTIME_CHECK(x.sizes() == gate.sizes(), "NVQ x and gate shapes must match");
        auto value = mode == 1
            ? x * mfq_tensor_backend::sigmoid(gate)
            : x * mfq_tensor_backend::silu(gate);
        return nvq_matmul_cpu(w, value);
    }
    x = x.contiguous().to(mfq_tensor_backend::kFloat16);
    gate = gate.contiguous().to(mfq_tensor_backend::kFloat16);
    MFQ_RUNTIME_CHECK(x.sizes() == gate.sizes(), "NVQ x and gate shapes must match");
    x = pad_last(x, w.neuron_len);
    gate = pad_last(gate, w.neuron_len);
    int M = (int)x.size(0);
    const NvqMatmulPath path = select_nvq_matmul_path(w, M);
    if (path == NvqMatmulPath::Gemv) {
        NvqWorkspace & ws = w.workspace(M);
        return g_profiler.measure("nvq.gemv_gate", [&]() {
            return nvq_gemv_gate_ws_cuda(
                w.indices_packed, w.aux_packed, w.sub_scale_packed,
                w.neuron_scale, w.codebook, x, gate, w.neuron_len, w.gs,
                w.sub_bits, w.kernel_format, w.sign_mode, mode, ws.qx, ws.xscale);
        });
    }
    if (path == NvqMatmulPath::Mmq) {
        NvqWorkspace & ws = w.workspace(M);
        return g_profiler.measure("nvq.mma24_gate", [&]() {
            return nvq_mmq_gate_ws_cuda(
                w.indices_packed, w.aux_packed, w.sub_scale_packed,
                w.neuron_scale, w.codebook, x, gate, w.neuron_len, w.gs,
                w.sub_bits, w.kernel_format, w.sign_mode, mode, ws.qx, ws.xscale);
        });
    }
    mfq_tensor_backend::Tensor value = mode == 1 ? x * mfq_tensor_backend::sigmoid(gate) : x * mfq_tensor_backend::silu(gate);
    return nvq_matmul(w, value);
}

static bool nvq_pair_compatible(const NvqWeight & first, const NvqWeight & second) {
    return first.format == second.format && first.kernel_format == second.kernel_format &&
           first.gs == second.gs &&
           first.neuron_len == second.neuron_len && first.ng == second.ng;
}

bool nvq_fusion_enabled() {
    const char * disable = std::getenv("MFQ_DISABLE_NVQ_FUSION");
    if (disable == nullptr) disable = std::getenv("MFQ_DISABLE_NIQ_FUSION");
    return disable == nullptr || disable[0] != '1';
}

mfq_tensor_backend::Tensor nvq_matmul_multi2(
    const NvqWeight & first,
    const NvqWeight & second,
    mfq_tensor_backend::Tensor x) {
    if (!nvq_pair_compatible(first, second)) {
        throw std::runtime_error("NVQ multi-projection requires compatible formats and input layouts");
    }
    x = pad_last(x.contiguous().to(mfq_tensor_backend::kFloat16), first.neuron_len);
    const int M = (int)x.size(0);
    if (M > 8) return mfq_tensor_backend::cat({nvq_matmul(first, x), nvq_matmul(second, x)}, -1);
    NvqWorkspace & ws = first.workspace(M);
    return g_profiler.measure("nvq.gemv_multi2", [&]() {
        return nvq_gemv_multi2_ws_cuda(
            first.indices_packed, first.aux_packed, first.sub_scale_packed,
            first.neuron_scale, first.codebook,
            second.indices_packed, second.aux_packed, second.sub_scale_packed,
            second.neuron_scale, second.codebook,
            x, first.neuron_len, first.gs,
            first.sub_bits, first.kernel_format, first.sign_mode,
            second.sub_bits, second.kernel_format, second.sign_mode,
            ws.qx, ws.xscale);
    });
}

mfq_tensor_backend::Tensor nvq_matmul_swiglu(
    const NvqWeight & gate,
    const NvqWeight & up,
    mfq_tensor_backend::Tensor x) {
    if (!nvq_pair_compatible(gate, up) || gate.out != up.out) {
        throw std::runtime_error("NVQ SwiGLU requires compatible equal-width gate/up weights");
    }
    x = pad_last(x.contiguous().to(mfq_tensor_backend::kFloat16), gate.neuron_len);
    if (x.size(0) != 1) {
        auto pair = nvq_matmul_multi2(gate, up, x);
        auto parts = pair.split_with_sizes({gate.out, up.out}, -1);
        return mfq_tensor_backend::silu(parts[0]) * parts[1];
    }
    NvqWorkspace & ws = gate.workspace(1);
    return g_profiler.measure("nvq.gemv_swiglu", [&]() {
        return nvq_gemv_swiglu_ws_cuda(
            gate.indices_packed, gate.aux_packed, gate.sub_scale_packed,
            gate.neuron_scale, gate.codebook,
            up.indices_packed, up.aux_packed, up.sub_scale_packed,
            up.neuron_scale, up.codebook,
            x, gate.neuron_len, gate.gs,
            gate.sub_bits, gate.kernel_format, gate.sign_mode,
            up.sub_bits, up.kernel_format, up.sign_mode,
            ws.qx, ws.xscale);
    });
}

mfq_tensor_backend::Tensor nvq_ffn_swiglu_down(
    const NvqWeight & gate,
    const NvqWeight & up,
    const NvqWeight & down,
    mfq_tensor_backend::Tensor x,
    MfqOptional<mfq_tensor_backend::Tensor> residual) {
    if (!nvq_pair_compatible(gate, up) || gate.out != up.out || gate.out != down.neuron_len) {
        throw std::runtime_error("NVQ fused FFN weight layouts are incompatible");
    }
    x = pad_last(x.contiguous().to(mfq_tensor_backend::kFloat16), gate.neuron_len);
    if (x.size(0) != 1 || (down.gs != 24 && down.gs != 28 && down.gs != 32)) {
        auto output = nvq_matmul(down, nvq_matmul_swiglu(gate, up, x));
        return residual.has_value()
            ? acc_cuda(residual.value(), output)
            : output;
    }
    NvqWorkspace & input_ws = gate.workspace(1);
    NvqWorkspace & output_ws = down.workspace(1);
    if (!input_ws.swiglu_scratch.defined() || input_ws.swiglu_scratch.numel() < gate.out) {
        input_ws.swiglu_scratch = mfq_tensor_backend::empty(
            {gate.out}, gate.indices_packed.options().dtype(
                mfq_tensor_backend::kFloat32));
    }
    g_profiler.measure("nvq.ffn_swiglu_quant", [&]() {
        nvq_ffn_swiglu_quant_ws_cuda(
            gate.indices_packed, gate.aux_packed, gate.sub_scale_packed,
            gate.neuron_scale, gate.codebook,
            up.indices_packed, up.aux_packed, up.sub_scale_packed,
            up.neuron_scale, up.codebook,
            x, gate.neuron_len, gate.gs,
            gate.sub_bits, gate.kernel_format, gate.sign_mode,
            up.sub_bits, up.kernel_format, up.sign_mode, down.gs,
            input_ws.qx, input_ws.xscale,
            output_ws.qx, output_ws.xscale, input_ws.swiglu_scratch);
        return 0;
    });
    return g_profiler.measure(
        residual.has_value() ? "nvq.gemv_qx_residual" : "nvq.gemv_qx",
        [&]() {
        if (residual.has_value()) {
            return nvq_gemv_qx_residual_ws_cuda(
                down.indices_packed, down.aux_packed, down.sub_scale_packed,
                down.neuron_scale, down.codebook,
                down.neuron_len, down.gs, down.sub_bits,
                down.kernel_format, down.sign_mode,
                output_ws.qx, output_ws.xscale, residual.value());
        }
        return nvq_gemv_qx_ws_cuda(
            down.indices_packed, down.aux_packed, down.sub_scale_packed,
            down.neuron_scale, down.codebook,
            down.neuron_len, down.gs, down.sub_bits, down.kernel_format, down.sign_mode,
            output_ws.qx, output_ws.xscale);
    });
}

static mfq_tensor_backend::Tensor nvq_embedding(const NvqWeight & w, mfq_tensor_backend::Tensor token_ids) {
    return nvq_embedding_lookup_cuda(
        w.indices_packed, w.aux_packed, w.sub_scale_packed,
        w.neuron_scale, w.codebook, token_ids, w.neuron_len, w.gs,
        w.sub_bits, w.kernel_format, w.sign_mode);
}

mfq_tensor_backend::Tensor nint_matmul(const NintWeight & w, mfq_tensor_backend::Tensor x) {
    if (!x.is_cuda()) return nint_matmul_cpu(w, std::move(x));
    x = x.contiguous().to(mfq_tensor_backend::kFloat16);
    x = pad_last(x, w.neuron_len);
    int M = (int)x.size(0);
    if (!w.q8_zero) {
        if (M <= 8) {
            Workspace & ws = w.workspace(M);
            return g_profiler.measure("nint.matmul", [&]() {
                if (w.aligned_q8) {
                    return nint_matmul_q8_ws_cuda(
                        w.q_packed, w.row_q_bits, w.row_q_bit_offsets,
                        w.sub_scale, w.sub_min, w.neuron_scale, w.neuron_min,
                        x, w.gs, ws.qx, ws.xscale);
                }
                if (M == 1 && w.uniform_q46 != 0) {
                    return nint_matmul_q46_ws_cuda(
                        w.q_packed, w.row_q_bits, w.row_q_bit_offsets,
                        w.sub_scale, w.sub_min, w.neuron_scale, w.neuron_min,
                        x, w.gs, ws.qx, ws.xscale, w.uniform_q46);
                }
                return nint_matmul_ws_cuda(
                    w.q_packed, w.row_q_bits, w.row_q_bit_offsets,
                    w.sub_scale, w.sub_min, w.neuron_scale, w.neuron_min,
                    x, w.gs, ws.qx, ws.xscale);
            });
        }
        auto dense = g_profiler.measure("nint.dequant", [&]() {
            return nint_decode_cuda(
                w.q_packed, w.row_q_bits, w.row_q_bit_offsets,
                w.sub_scale, w.sub_min, w.neuron_scale, w.neuron_min,
                w.neuron_len, w.gs);
        });
        return g_profiler.measure("nint.gemm", [&]() {
            return mfq_tensor_backend::matmul(x, dense.transpose(0, 1));
        });
    }
    if (g_kl_mmq_mode != KlMmqMode::Default) {
        const int original_m = M;
        MFQ_RUNTIME_CHECK(original_m > 0, "KLD NINT8-0 MMQ requires activation rows");
        if (M < 16) {
            auto padding = mfq_tensor_backend::zeros(
                {16 - M, x.size(1)}, x.options());
            x = mfq_tensor_backend::cat({x, padding}, 0).contiguous();
            M = 16;
        }
        x = kl_mmq_prepare_activation(x);
        ++g_kl_mmq_dense_calls;
        auto result = g_profiler.measure("kld_mmq.nint8_zero.fp16", [&]() {
            return nint8_zero_mmq_f16_packed_cuda(
                w.q_packed, w.q8_zero_scale, x, w.neuron_len);
        });
        return original_m < 16
            ? result.index({Slice(0, original_m)}).contiguous()
            : result;
    }
    if (M <= 8) {
        Workspace & ws = w.workspace(M);
        return g_profiler.measure("nint8_zero.gemv", [&]() {
            return nint8_zero_gemv_ws_cuda(
                w.q_packed, w.q8_zero_scale, x, ws.qx, ws.xscale);
        });
    }
    return g_profiler.measure("nint8_zero.packed_mmq", [&]() {
        return nint8_zero_mmq_f16_packed_cuda(
            w.q_packed, w.q8_zero_scale, x, w.neuron_len);
    });
}

mfq_tensor_backend::Tensor nint_matmul_bf16_output(
        const NintWeight & w, mfq_tensor_backend::Tensor x) {
    auto shape = x.sizes().vec();
    auto flat = x.reshape({-1, x.size(-1)});
    auto output = nint_matmul(w, flat)
        .to(mfq_tensor_backend::kBFloat16).contiguous();
    shape.back() = output.size(-1);
    return output.reshape(shape);
}

static mfq_tensor_backend::Tensor nint_matmul_f32_kld(
        const NintWeight & w, mfq_tensor_backend::Tensor x) {
    MFQ_RUNTIME_CHECK(
        g_kl_mmq_mode == KlMmqMode::Fp16,
        "FP32-output NINT MMQ is restricted to the FP16 KLD path");
    x = pad_last(
        x.contiguous().to(mfq_tensor_backend::kFloat16),
        w.neuron_len);
    if (!w.q8_zero) {
        return nint_matmul(w, x).to(mfq_tensor_backend::kFloat32);
    }
    MFQ_RUNTIME_CHECK(
        x.size(0) >= 16,
        "FP32-output NINT MMQ requires at least 16 activation rows");
    ++g_kl_mmq_dense_calls;
    return g_profiler.measure(
        "kld_mmq.nint8_zero.fp32_output", [&]() {
            return nint8_zero_mmq_f32_packed_cuda(
                w.q_packed, w.q8_zero_scale,
                x, w.neuron_len);
        });
}

mfq_tensor_backend::Tensor nint_matmul_input_mul_f32_kld(
        const NintWeight & w,
        mfq_tensor_backend::Tensor x,
        mfq_tensor_backend::Tensor gate,
        int mode) {
    MFQ_RUNTIME_CHECK(
        mode == 1 || mode == 2,
        "FP32-output NINT input gate mode must be sigmoid or SiLU");
    x = pad_last(
        x.contiguous().to(mfq_tensor_backend::kFloat16),
        w.neuron_len);
    gate = pad_last(
        gate.contiguous().to(mfq_tensor_backend::kFloat16),
        w.neuron_len);
    MFQ_RUNTIME_CHECK(
        x.sizes() == gate.sizes(),
        "FP32-output NINT x and gate shapes must match");
    auto activation = mode == 1
        ? x * mfq_tensor_backend::sigmoid(gate)
        : x * mfq_tensor_backend::silu(gate);
    return nint_matmul_f32_kld(
        w, activation.contiguous());
}

mfq_tensor_backend::Tensor nint_matmul_groupwise_u8(
        const NintWeight & w, mfq_tensor_backend::Tensor x, int64_t groups) {
    MFQ_RUNTIME_CHECK(
        w.bits == 8 && w.gs == 48,
        "groupwise NINT projection requires NINT8 gs48");
    MFQ_RUNTIME_CHECK(
        x.dim() == 3 && x.size(1) == groups && x.size(2) == w.neuron_len,
        "groupwise NINT projection expects [B, groups, K]");
    MFQ_RUNTIME_CHECK(
        w.out % groups == 0,
        "groupwise NINT projection output rows must divide groups");
    x = x.contiguous().to(mfq_tensor_backend::kFloat16);
    auto dense = g_profiler.measure("nint.groupwise_dequant", [&]() {
        return nint_decode_cuda(
            w.q_packed, w.row_q_bits, w.row_q_bit_offsets,
            w.sub_scale, w.sub_min, w.neuron_scale, w.neuron_min,
            w.neuron_len, w.gs);
    });
    const int64_t rows_per_group = w.out / groups;
    return g_profiler.measure("nint.groupwise_gemm", [&]() {
        return mfq_tensor_backend::bmm(
            x.transpose(0, 1),
            dense.reshape({groups, rows_per_group, w.neuron_len})
                .transpose(1, 2))
            .transpose(0, 1)
            .contiguous()
            .reshape({x.size(0), w.out});
    });
}

mfq_tensor_backend::Tensor nint_matmul_input_mul(const NintWeight & w, mfq_tensor_backend::Tensor x, mfq_tensor_backend::Tensor gate, int mode) {
    MFQ_RUNTIME_CHECK(
        mode == 1 || mode == 2,
        "NINT input gate mode must be sigmoid or SiLU");
    if (!x.is_cuda()) {
        x = x.contiguous().to(mfq_tensor_backend::kFloat16);
        gate = gate.contiguous().to(mfq_tensor_backend::kFloat16);
        MFQ_RUNTIME_CHECK(x.sizes() == gate.sizes(), "NINT x and gate shapes must match");
        auto value = mode == 1
            ? x * mfq_tensor_backend::sigmoid(gate)
            : x * mfq_tensor_backend::silu(gate);
        return nint_matmul_cpu(w, value);
    }
    x = x.contiguous().to(mfq_tensor_backend::kFloat16);
    gate = gate.contiguous().to(mfq_tensor_backend::kFloat16);
    MFQ_RUNTIME_CHECK(
        x.sizes() == gate.sizes(),
        "NINT x and gate shapes must match");
    x = pad_last(x, w.neuron_len);
    gate = pad_last(gate, w.neuron_len);
    if (!w.q8_zero && x.size(0) <= 8) {
        Workspace & ws = w.workspace(static_cast<int>(x.size(0)));
        return g_profiler.measure("nint.matmul.input_mul", [&]() {
            if (w.aligned_q8) {
                return nint_matmul_input_mul_q8_ws_cuda(
                    w.q_packed, w.row_q_bits, w.row_q_bit_offsets,
                    w.sub_scale, w.sub_min, w.neuron_scale, w.neuron_min,
                    x, gate, mode, w.gs, ws.qx, ws.xscale);
            }
            if (x.size(0) == 1 && w.uniform_q46 != 0) {
                return nint_matmul_input_mul_q46_ws_cuda(
                    w.q_packed, w.row_q_bits, w.row_q_bit_offsets,
                    w.sub_scale, w.sub_min, w.neuron_scale, w.neuron_min,
                    x, gate, mode, w.gs, ws.qx, ws.xscale, w.uniform_q46);
            }
            return nint_matmul_input_mul_ws_cuda(
                w.q_packed, w.row_q_bits, w.row_q_bit_offsets,
                w.sub_scale, w.sub_min, w.neuron_scale, w.neuron_min,
                x, gate, mode, w.gs, ws.qx, ws.xscale);
        });
    }
    if (mode == 1) return nint_matmul(w, x * mfq_tensor_backend::sigmoid(gate));
    return nint_matmul(w, x * mfq_tensor_backend::silu(gate));
}

mfq_tensor_backend::Tensor nint_matmul_swiglu(const NintWeight & w, mfq_tensor_backend::Tensor x) {
    x = x.contiguous().to(mfq_tensor_backend::kFloat16);
    x = pad_last(x, w.neuron_len);
    auto parts = nint_matmul(w, x).chunk(2, -1);
    return mfq_tensor_backend::silu(parts[0]) * parts[1];
}

mfq_tensor_backend::Tensor nint_matmul_geglu(const NintWeight & w, mfq_tensor_backend::Tensor x) {
    x = x.contiguous().to(mfq_tensor_backend::kFloat16);
    x = pad_last(x, w.neuron_len);
    auto parts = nint_matmul(w, x).chunk(2, -1);
    return gelu_mul_cuda(parts[0].contiguous(), parts[1].contiguous());
}



thread_local bool g_decode_graph_serial_branches = false;
thread_local bool g_decode_graph_tp_projection_major = false;





bool decode_branch_parallel_enabled(int64_t rows) {
    const char * disabled =
        std::getenv("MFQ_DISABLE_DECODE_BRANCH_PARALLEL");
    // Branch output storage belongs to its allocating stream. Keep graph
    // warmup/capture on the graph pool's stream until cross-stream allocation
    // lifetime tracking supports a fully rejoined capture. Eager is unchanged.
    return rows == 1 && !g_decode_graph_serial_branches &&
        (disabled == nullptr || disabled[0] != '1');
}





static NintLinearGroup make_linear_group(const std::vector<NintWeight> & ws) {
    NintLinearGroup g;
    for (const auto & w : ws) g.outs.push_back(w.out);
    bool same = !ws.empty();
    for (size_t i = 1; i < ws.size(); ++i) {
        if (ws[i].ng != ws[0].ng || ws[i].gs != ws[0].gs ||
            ws[i].neuron_len != ws[0].neuron_len ||
            ws[i].q8_zero != ws[0].q8_zero) {
            same = false;
            break;
        }
    }
    if (same) {
        g.w = cat_weights(ws);
        if (g.w.q8_zero) {
            g.projection_w.reserve(ws.size());
            int64_t offset = 0;
            for (const auto & source : ws) {
                NintWeight projection = g.w;
                projection.out = source.out;
                projection.shape = source.shape;
                projection.q_packed =
                    g.w.q_packed.narrow(0, offset, source.out);
                projection.q8_zero_scale =
                    g.w.q8_zero_scale.narrow(0, offset, source.out);
                projection.workspaces.clear();
                g.projection_w.push_back(std::move(projection));
                offset += source.out;
            }
            MFQ_RUNTIME_CHECK(offset == g.w.out,
                        "Q8 projection views do not cover grouped output");
        }
    } else {
        std::vector<NintWeight> cur;
        std::vector<int64_t> cur_outs;
        auto flush = [&]() {
            if (cur.empty()) return;
            g.split_w.push_back(cur.size() == 1 ? cur[0] : cat_weights(cur));
            g.split_outs.push_back(cur_outs);
            cur.clear();
            cur_outs.clear();
        };
        for (const auto & w : ws) {
            bool append = !cur.empty() &&
                w.ng == cur[0].ng && w.gs == cur[0].gs &&
                w.neuron_len == cur[0].neuron_len &&
                w.q8_zero == cur[0].q8_zero;
            if (!append) flush();
            cur.push_back(w);
            cur_outs.push_back(w.out);
        }
        flush();
    }
    return g;
}



mfq_tensor_backend::Tensor mxfp8_matmul(
        const Mxfp8Weight & weight,
        mfq_tensor_backend::Tensor x) {
    x = x.contiguous().to(mfq_tensor_backend::kFloat16);
    MFQ_RUNTIME_CHECK(
        x.dim() == 2 && x.size(1) == weight.neuron_len,
        "MXFP8 activation width mismatch");
    if (!x.is_cuda()) {
        MFQ_RUNTIME_CHECK(
            !weight.values.is_cuda() && !weight.scales.is_cuda(),
            "CPU MXFP8 matmul requires CPU-resident weights");
        const int64_t rows = x.size(0);
        const int64_t outputs = weight.out;
        const int64_t width = weight.neuron_len;
        const int64_t scale_columns = width / 128;
        const auto * input = x.data_ptr<mfq_half>();
        const auto * values = weight.values.data_ptr<uint8_t>();
        const auto * scales = weight.scales.data_ptr<uint8_t>();
        auto result = mfq_tensor_backend::empty(
            {rows, outputs},
            mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCPU).dtype(mfq_tensor_backend::kFloat16));
        auto * output = result.data_ptr<mfq_half>();
        mfq_parallel_for(0, outputs, 1, [&](int64_t begin, int64_t end) {
            std::vector<float> accumulators(static_cast<size_t>(rows));
            for (int64_t neuron = begin; neuron < end; ++neuron) {
                std::fill(accumulators.begin(), accumulators.end(), 0.0f);
                for (int64_t column = 0; column < width; ++column) {
                    const uint8_t raw = values[neuron * width + column];
                    const unsigned exponent =
                        (static_cast<unsigned>(raw) >> 3u) & 15u;
                    const unsigned mantissa = static_cast<unsigned>(raw) & 7u;
                    float decoded = exponent == 0u
                        ? std::ldexp(float(mantissa) * 0.125f, -6)
                        : std::ldexp(
                            1.0f + float(mantissa) * 0.125f,
                            static_cast<int>(exponent) - 7);
                    if ((raw & 128u) != 0u) decoded = -decoded;
                    const uint8_t raw_scale = scales[
                        (neuron / 128) * scale_columns + column / 128];
                    const float weight_value = decoded * std::ldexp(
                        1.0f, static_cast<int>(raw_scale) - 127);
                    for (int64_t row = 0; row < rows; ++row) {
                        accumulators[static_cast<size_t>(row)] = std::fma(
                            static_cast<float>(input[row * width + column]),
                            weight_value,
                            accumulators[static_cast<size_t>(row)]);
                    }
                }
                for (int64_t row = 0; row < rows; ++row) {
                    output[row * outputs + neuron] = mfq_half(
                        accumulators[static_cast<size_t>(row)]);
                }
            }
        });
        return result;
    }
    if (x.size(0) <= 8) {
        return g_profiler.measure("mxfp8.small_m", [&]() {
            return mxfp8_small_m_cuda(
                weight.values, weight.scales, x);
        });
    }
    return g_profiler.measure("mxfp8.packed_matmul", [&]() {
        return mxfp8_matmul_f16_cuda(
            weight.values, weight.scales, x);
    });
}

mfq_tensor_backend::Tensor mxfp8_matmul_f32(
        const Mxfp8Weight & weight,
        mfq_tensor_backend::Tensor x) {
    x = x.contiguous().to(mfq_tensor_backend::kFloat16);
    MFQ_RUNTIME_CHECK(
        x.dim() == 2 && x.size(1) == weight.neuron_len,
        "MXFP8 FP32-output activation width mismatch");
    if (x.size(0) <= 8) {
        return g_profiler.measure("mxfp8.small_m_f32", [&]() {
            return mxfp8_small_m_f32_cuda(
                weight.values, weight.scales, x);
        });
    }
    return g_profiler.measure("mxfp8.gemm_f32", [&]() {
        return mxfp8_gemm_f32_cuda(
            weight.values, weight.scales, x);
    });
}

static mfq_tensor_backend::Tensor mxfp8_cpu_reference(
        const Mxfp8Weight & weight) {
    auto values = weight.values.to(mfq_tensor_backend::kCPU).contiguous();
    auto scales = weight.scales.to(mfq_tensor_backend::kCPU).contiguous();
    const auto * value_bytes = values.data_ptr<uint8_t>();
    const auto * scale_bytes = scales.data_ptr<uint8_t>();
    const int64_t scale_columns = weight.neuron_len / 128;
    std::vector<float> dense(
        static_cast<size_t>(weight.out * weight.neuron_len));
    for (int64_t row = 0; row < weight.out; ++row) {
        for (int64_t column = 0;
             column < weight.neuron_len; ++column) {
            const uint8_t raw = value_bytes[
                static_cast<size_t>(row * weight.neuron_len + column)];
            const unsigned exponent =
                (static_cast<unsigned>(raw) >> 3u) & 15u;
            const unsigned mantissa =
                static_cast<unsigned>(raw) & 7u;
            float value;
            if (exponent == 15u && mantissa == 7u) {
                value = std::numeric_limits<float>::quiet_NaN();
            } else {
                value = exponent == 0u
                    ? std::ldexp(float(mantissa) * 0.125f, -6)
                    : std::ldexp(
                        1.0f + float(mantissa) * 0.125f,
                        static_cast<int>(exponent) - 7);
                if ((raw & 128u) != 0u) value = -value;
            }
            const uint8_t raw_scale = scale_bytes[
                static_cast<size_t>(
                    (row / 128) * scale_columns + column / 128)];
            const float scale = raw_scale == 255u
                ? std::numeric_limits<float>::quiet_NaN()
                : std::ldexp(1.0f, int(raw_scale) - 127);
            dense[static_cast<size_t>(
                row * weight.neuron_len + column)] = value * scale;
        }
    }
    return mfq_tensor_backend::from_blob(
        dense.data(), {weight.out, weight.neuron_len},
        mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kFloat32))
        .clone().to(weight.values.device())
        .to(mfq_tensor_backend::kFloat16).contiguous();
}

mfq_tensor_backend::Tensor mxfp8_groupwise_matmul(
        const Mxfp8Weight & weight,
        mfq_tensor_backend::Tensor grouped,
        int64_t groups) {
    grouped = grouped.contiguous().to(mfq_tensor_backend::kFloat16);
    MFQ_RUNTIME_CHECK(
        grouped.dim() == 3 && grouped.size(1) == groups &&
            grouped.size(2) == weight.neuron_len &&
            weight.out % groups == 0,
        "MXFP8 groupwise projection geometry mismatch");
    const int64_t outputs_per_group = weight.out / groups;
    MFQ_RUNTIME_CHECK(
        outputs_per_group % 128 == 0,
        "MXFP8 groupwise output width must preserve scale blocks");
    if (grouped.size(0) <= 8) {
        return g_profiler.measure("mxfp8.groupwise_small_m", [&]() {
            return mxfp8_groupwise_small_m_cuda(
                weight.values, weight.scales, grouped, groups);
        });
    }
    std::vector<mfq_tensor_backend::Tensor> outputs;
    outputs.reserve(static_cast<size_t>(groups));
    const int64_t scale_rows_per_group = outputs_per_group / 128;
    for (int64_t group = 0; group < groups; ++group) {
        Mxfp8Weight shard;
        shard.out = outputs_per_group;
        shard.neuron_len = weight.neuron_len;
        shard.values = weight.values.narrow(
            0, group * outputs_per_group,
            outputs_per_group).contiguous();
        shard.scales = weight.scales.narrow(
            0, group * scale_rows_per_group,
            scale_rows_per_group).contiguous();
        outputs.push_back(mxfp8_matmul(
            shard, grouped.select(1, group).contiguous()));
    }
    return mfq_tensor_backend::cat(outputs, -1).contiguous();
}

mfq_tensor_backend::Tensor mxfp8_groupwise_matmul_f32(
        const Mxfp8Weight & weight,
        mfq_tensor_backend::Tensor grouped,
        int64_t groups) {
    grouped = grouped.contiguous().to(mfq_tensor_backend::kFloat16);
    MFQ_RUNTIME_CHECK(
        grouped.dim() == 3 && grouped.size(1) == groups &&
            grouped.size(2) == weight.neuron_len &&
            weight.out % groups == 0,
        "MXFP8 groupwise FP32-output projection geometry mismatch");
    const int64_t outputs_per_group = weight.out / groups;
    MFQ_RUNTIME_CHECK(
        outputs_per_group % 128 == 0,
        "MXFP8 groupwise output width must preserve scale blocks");
    if (grouped.size(0) <= 8) {
        return g_profiler.measure(
            "mxfp8.groupwise_small_m_f32", [&]() {
                return mxfp8_groupwise_small_m_f32_cuda(
                    weight.values, weight.scales, grouped, groups);
            });
    }
    std::vector<mfq_tensor_backend::Tensor> outputs;
    outputs.reserve(static_cast<size_t>(groups));
    const int64_t scale_rows_per_group = outputs_per_group / 128;
    for (int64_t group = 0; group < groups; ++group) {
        Mxfp8Weight shard;
        shard.out = outputs_per_group;
        shard.neuron_len = weight.neuron_len;
        shard.values = weight.values.narrow(
            0, group * outputs_per_group,
            outputs_per_group).contiguous();
        shard.scales = weight.scales.narrow(
            0, group * scale_rows_per_group,
            scale_rows_per_group).contiguous();
        outputs.push_back(mxfp8_matmul_f32(
            shard, grouped.select(1, group).contiguous()));
    }
    return mfq_tensor_backend::cat(outputs, -1).contiguous();
}

mfq_tensor_backend::Tensor mxfp4_matmul(
        const Mxfp4Weight & weight,
        mfq_tensor_backend::Tensor x) {
    x = x.contiguous().to(mfq_tensor_backend::kFloat16);
    MFQ_RUNTIME_CHECK(
        x.dim() == 2 && x.size(1) == weight.neuron_len,
        "MXFP4 activation width mismatch");
    if (x.is_cuda()) {
        MFQ_RUNTIME_CHECK(
            weight.values.is_cuda() && weight.scales.is_cuda(),
            "CUDA MXFP4 matmul requires CUDA-resident weights");
        return mxfp4_matmul_f16_cuda(
            weight.values, weight.scales, x);
    }
    MFQ_RUNTIME_CHECK(
        !weight.values.is_cuda() && !weight.scales.is_cuda(),
        "CPU MXFP4 matmul requires CPU-resident weights");
    const int64_t rows = x.size(0);
    const int64_t outputs = weight.out;
    const int64_t width = weight.neuron_len;
    const int64_t packed_columns = width / 2;
    const int64_t scale_columns = width / 32;
    const auto * input = x.data_ptr<mfq_half>();
    const auto * values = weight.values.data_ptr<uint8_t>();
    const auto * scales = weight.scales.data_ptr<uint8_t>();
    auto result = mfq_tensor_backend::empty(
        {rows, outputs},
        mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCPU).dtype(mfq_tensor_backend::kFloat16));
    auto * output = result.data_ptr<mfq_half>();
    static constexpr float magnitude[8] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    };
    mfq_parallel_for(0, outputs, 1, [&](int64_t begin, int64_t end) {
        std::vector<float> accumulators(static_cast<size_t>(rows));
        for (int64_t neuron = begin; neuron < end; ++neuron) {
            std::fill(accumulators.begin(), accumulators.end(), 0.0f);
            for (int64_t column = 0; column < width; ++column) {
                const uint8_t packed = values[
                    neuron * packed_columns + column / 2];
                const uint8_t code = static_cast<uint8_t>(
                    (packed >> ((column & 1) * 4)) & 15u);
                float decoded = magnitude[code & 7u];
                if ((code & 8u) != 0u) decoded = -decoded;
                const uint8_t raw_scale = scales[
                    neuron * scale_columns + column / 32];
                const float weight_value = decoded * std::ldexp(
                    1.0f, static_cast<int>(raw_scale) - 127);
                for (int64_t row = 0; row < rows; ++row) {
                    accumulators[static_cast<size_t>(row)] = std::fma(
                        static_cast<float>(input[row * width + column]),
                        weight_value,
                        accumulators[static_cast<size_t>(row)]);
                }
            }
            for (int64_t row = 0; row < rows; ++row) {
                output[row * outputs + neuron] = mfq_half(
                    accumulators[static_cast<size_t>(row)]);
            }
        }
    });
    return result;
}

mfq_tensor_backend::Tensor tpq_matmul(
        const TpqWeight & weight,
        mfq_tensor_backend::Tensor x) {
    x = x.contiguous().to(mfq_tensor_backend::kFloat16);
    MFQ_RUNTIME_CHECK(
        x.dim() == 2 && x.size(1) == weight.neuron_len,
        "TPQ activation width mismatch");
    if (x.is_cuda()) {
        MFQ_RUNTIME_CHECK(weight.packed.is_cuda(),
            "CUDA TPQ matmul requires CUDA-resident weights");
        return weight.int4
            ? tpq_int4_matmul_f16_cuda(
                weight.packed, weight.scales, x, weight.group_size)
            : tpq_pq_matmul_f16_cuda(
                weight.packed, weight.codebook, x,
                weight.out, weight.neuron_len,
                weight.vector_size, weight.index_bits);
    }
    MFQ_RUNTIME_CHECK(!weight.packed.is_cuda(),
        "CPU TPQ matmul requires CPU-resident weights");
    const int64_t rows = x.size(0);
    const int64_t outputs = weight.out;
    const int64_t width = weight.neuron_len;
    const auto * input = x.data_ptr<mfq_half>();
    const auto * packed = weight.packed.data_ptr<uint8_t>();
    auto result = mfq_tensor_backend::empty(
        {rows, outputs},
        mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCPU).dtype(mfq_tensor_backend::kFloat16));
    auto * output = result.data_ptr<mfq_half>();
    const auto * scales = weight.int4
        ? weight.scales.data_ptr<mfq_half>() : nullptr;
    const auto * codebook = weight.int4
        ? nullptr : weight.codebook.data_ptr<float>();
    const size_t packed_size = static_cast<size_t>(weight.packed.numel());
    mfq_parallel_for(0, outputs, 1, [&](int64_t begin, int64_t end) {
        std::vector<float> accumulators(static_cast<size_t>(rows));
        for (int64_t neuron = begin; neuron < end; ++neuron) {
            std::fill(accumulators.begin(), accumulators.end(), 0.0f);
            if (weight.int4) {
                const int64_t packed_columns = width / 2;
                const int64_t scale_columns = width / weight.group_size;
                for (int64_t column = 0; column < width; ++column) {
                    const uint8_t byte = packed[
                        neuron * packed_columns + column / 2];
                    const int code = static_cast<int>(
                        (byte >> ((column & 1) * 4)) & 15u) - 8;
                    const float value = static_cast<float>(code) *
                        static_cast<float>(scales[
                            neuron * scale_columns +
                            column / weight.group_size]);
                    for (int64_t row = 0; row < rows; ++row) {
                        accumulators[static_cast<size_t>(row)] = std::fma(
                            static_cast<float>(input[row * width + column]),
                            value, accumulators[static_cast<size_t>(row)]);
                    }
                }
            } else {
                const int64_t vectors = width / weight.vector_size;
                for (int64_t vector = 0; vector < vectors; ++vector) {
                    const size_t linear = static_cast<size_t>(
                        neuron * vectors + vector);
                    const size_t bit = linear *
                        static_cast<size_t>(weight.index_bits);
                    const size_t byte = bit >> 3;
                    const int shift = static_cast<int>(bit & 7);
                    uint32_t code = byte < packed_size ? packed[byte] : 0u;
                    if (byte + 1 < packed_size) {
                        code |= static_cast<uint32_t>(packed[byte + 1]) << 8;
                    }
                    if (byte + 2 < packed_size) {
                        code |= static_cast<uint32_t>(packed[byte + 2]) << 16;
                    }
                    code = (code >> shift) &
                        ((1u << weight.index_bits) - 1u);
                    const float * values = codebook +
                        static_cast<size_t>(code) * weight.vector_size;
                    for (int element = 0;
                         element < weight.vector_size; ++element) {
                        const int64_t column =
                            vector * weight.vector_size + element;
                        for (int64_t row = 0; row < rows; ++row) {
                            accumulators[static_cast<size_t>(row)] = std::fma(
                                static_cast<float>(input[row * width + column]),
                                values[element],
                                accumulators[static_cast<size_t>(row)]);
                        }
                    }
                }
            }
            for (int64_t row = 0; row < rows; ++row) {
                output[row * outputs + neuron] = mfq_half(
                    accumulators[static_cast<size_t>(row)]);
            }
        }
    });
    return result;
}















template <typename Decode>
static mfq_tensor_backend::Tensor sq_matmul_cpu_impl(
        mfq_tensor_backend::Tensor x, int64_t outputs, int64_t width, Decode decode) {
    MFQ_RUNTIME_CHECK(!x.is_cuda() && x.size(-1) == width, "SQ CPU activation device or width mismatch");
    const auto dtype = x.scalar_type();
    auto shape = x.sizes().vec();
    x = x.to(mfq_tensor_backend::kFloat32).reshape({-1, width}).contiguous();
    const auto rows = x.size(0);
    auto result = mfq_tensor_backend::empty({rows, outputs}, x.options());
    const auto* input = x.data_ptr<float>();
    auto* output = result.data_ptr<float>();
    mfq_parallel_for(0, outputs, 1, [&](int64_t begin, int64_t end) {
        std::vector<float> weights(width);
        for (auto n = begin; n < end; ++n) {
            decode(n, weights.data());
            for (int64_t m = 0; m < rows; ++m) {
                float sum = 0;
                for (int64_t k = 0; k < width; ++k) sum = std::fma(input[m * width + k], weights[k], sum);
                output[m * outputs + n] = sum;
            }
        }
    });
    shape.back() = outputs;
    return result.to(dtype).reshape(shape);
}

mfq_tensor_backend::Tensor sq_matmul_cpu(const Mxfp4SqWeight& weight, mfq_tensor_backend::Tensor x) {
    MFQ_RUNTIME_CHECK(!weight.blob.is_cuda(), "SQ CPU weights must be on CPU");
    const auto* blob = weight.blob.data_ptr<uint8_t>();
    const auto layout = mfq::sq::parse(blob, weight.blob.numel());
    return sq_matmul_cpu_impl(x, weight.out, weight.neuron_len, [&](int64_t n, float* row) {
        mfq::sq::decode_cpu_row(blob, layout, weight.row_q.data_ptr<uint8_t>()[n],
            weight.row_symbol_byte_offsets.data_ptr<int32_t>()[n], weight.row_auxiliary.data_ptr<int32_t>()[n], row);
    });
}

mfq_tensor_backend::Tensor sq_matmul_cpu(const Fp8SqWeight& weight, mfq_tensor_backend::Tensor x) {
    MFQ_RUNTIME_CHECK(!weight.blob.is_cuda(), "SQ CPU weights must be on CPU");
    const auto* blob = weight.blob.data_ptr<uint8_t>();
    const auto layout = mfq::fp8sq::parse(weight.dtype, blob, weight.blob.numel());
    return sq_matmul_cpu_impl(x, weight.out, weight.neuron_len, [&](int64_t n, float* row) {
        for (int64_t k = 0; k < weight.neuron_len; ++k) row[k] = mfq::fp8sq::decode_cpu(
            blob, layout, weight.row_q.data_ptr<uint8_t>()[n],
            weight.row_symbol_byte_offsets.data_ptr<int32_t>()[n], n, k);
    });
}

mfq_tensor_backend::Tensor run_quant_linear_shard(
        const QuantLinearShard & shard,
        mfq_tensor_backend::Tensor x,
        MfqOptional<mfq_tensor_backend::Tensor> gate,
        int gate_mode) {
    MfqCudaGuard guard(shard.device);
    if (shard.kind == QuantLinearKind::Mxfp4Sq || shard.kind == QuantLinearKind::Fp8Sq) {
        if (gate.has_value()) {
            MFQ_RUNTIME_CHECK(gate_mode == 1 || gate_mode == 2, "SQ input gate mode must be sigmoid or SiLU");
            x = x * (gate_mode == 1 ? mfq_tensor_backend::sigmoid(*gate) : mfq_tensor_backend::silu(*gate));
        }
        return shard.kind == QuantLinearKind::Mxfp4Sq
            ? shard.mxfp4_sq.forward(x) : shard.fp8_sq.forward(x);
    }
    if (shard.kind == QuantLinearKind::Nint) {
        return gate.has_value()
            ? nint_matmul_input_mul(
                shard.nint, x, gate.value(), gate_mode)
            : nint_matmul(shard.nint, x);
    }
    if (shard.kind == QuantLinearKind::Nvq) {
        return gate.has_value()
            ? nvq_matmul_input_mul(
                shard.nvq, x, gate.value(), gate_mode)
            : nvq_matmul(shard.nvq, x);
    }
    if (shard.kind == QuantLinearKind::Mxfp4) {
        MFQ_RUNTIME_CHECK(
            !gate.has_value(),
            "MXFP4 tensor-parallel linear does not support input gating");
        return mxfp4_matmul(shard.mxfp4, x);
    }
    if (shard.kind == QuantLinearKind::Tpq) {
        MFQ_RUNTIME_CHECK(
            !gate.has_value(),
            "TPQ tensor-parallel linear does not support input gating");
        return tpq_matmul(shard.tpq, x);
    }
    if (shard.kind == QuantLinearKind::Dense) {
        auto local = x.to(shard.dense.scalar_type());
        if (gate.has_value()) {
            MFQ_RUNTIME_CHECK(
                gate_mode == 1 || gate_mode == 2,
                "dense input gate mode must be sigmoid or SiLU");
            auto local_gate = gate.value().to(local.scalar_type());
            local = gate_mode == 1
                ? local * mfq_tensor_backend::sigmoid(local_gate)
                : local * mfq_tensor_backend::silu(local_gate);
        }
        return mfq_tensor_backend::matmul(local, shard.dense.transpose(0, 1));
    }
    MFQ_RUNTIME_CHECK(
        !gate.has_value(),
        "MXFP8 tensor-parallel linear does not support input gating");
    return mxfp8_matmul(shard.mxfp8, x);
}

mfq_tensor_backend::Tensor tensor_to_cuda_device(
        mfq_tensor_backend::Tensor value,
        int device,
        mfq_tensor_backend::Tensor reusable) {
    if (value.is_cuda() && value.get_device() == device) {
        return value.contiguous();
    }
    const auto reusable_matches = [&]() {
        return reusable.defined() && reusable.is_cuda() &&
            reusable.get_device() == device &&
            reusable.sizes() == value.sizes() &&
            reusable.scalar_type() == value.scalar_type() &&
            reusable.is_contiguous();
    };
    const auto destination_tensor = [&]() {
        if (reusable_matches()) return reusable;
        MfqCudaGuard destination_guard(device);
        return mfq_tensor_backend::empty(
            value.sizes(),
            value.options().device(
                mfq_tensor_backend::Device(
                    mfq_tensor_backend::kCUDA, device)));
    };
    if (value.numel() == 0) {
        return destination_tensor();
    }
#if defined(MFQ_NATIVE_CUDA_RUNTIME) && defined(MFQ_HAVE_NCCL)
    if (value.is_cuda() &&
            g_model_parallel_collectives.collectives_enabled) {
        const int source_device = value.get_device();
        const auto source_rank_it = std::find(
            g_model_parallel_collectives.devices.begin(),
            g_model_parallel_collectives.devices.end(),
            source_device);
        const auto destination_rank_it = std::find(
            g_model_parallel_collectives.devices.begin(),
            g_model_parallel_collectives.devices.end(),
            device);
        if (source_rank_it !=
                g_model_parallel_collectives.devices.end() &&
                destination_rank_it !=
                g_model_parallel_collectives.devices.end()) {
            const auto source_stream =
                mfq_get_current_cuda_stream(source_device);
            cudaStreamCaptureStatus capture_status =
                cudaStreamCaptureStatusNone;
            {
                MfqCudaGuard source_guard(source_device);
                MFQ_CUDA_CHECK(cudaStreamIsCapturing(
                    source_stream.stream(), &capture_status));
            }
            if (capture_status != cudaStreamCaptureStatusNone) {
                auto source = value.contiguous();
                auto destination = destination_tensor();
                auto& runtime = g_model_parallel_collectives;
                const auto source_rank = static_cast<size_t>(
                    source_rank_it - runtime.devices.begin());
                const auto destination_rank = static_cast<size_t>(
                    destination_rank_it - runtime.devices.begin());
                {
                    MfqCudaGuard source_guard(source_device);
                    MFQ_CUDA_CHECK(cudaEventRecord(
                        runtime.ready[source_rank],
                        source_stream.stream()));
                    MFQ_CUDA_CHECK(cudaStreamWaitEvent(
                        runtime.streams[source_rank].stream(),
                        runtime.ready[source_rank], 0));
                }
                {
                    MfqCudaGuard destination_guard(device);
                    // Cross-device event waits make the receiving NCCL stream
                    // participate in the same capture before ncclRecv runs.
                    MFQ_CUDA_CHECK(cudaStreamWaitEvent(
                        runtime.streams[destination_rank].stream(),
                        runtime.ready[source_rank], 0));
                }
                MFQ_NCCL_CHECK(ncclGroupStart());
                MFQ_NCCL_CHECK(ncclSend(
                    source.data_ptr(), source.nbytes(), ncclUint8,
                    static_cast<int>(destination_rank),
                    runtime.communicators[source_rank],
                    runtime.streams[source_rank].stream()));
                MFQ_NCCL_CHECK(ncclRecv(
                    destination.data_ptr(), destination.nbytes(), ncclUint8,
                    static_cast<int>(source_rank),
                    runtime.communicators[destination_rank],
                    runtime.streams[destination_rank].stream()));
                MFQ_NCCL_CHECK(ncclGroupEnd());
                {
                    MfqCudaGuard source_guard(source_device);
                    MFQ_CUDA_CHECK(cudaEventRecord(
                        runtime.completed[source_rank],
                        runtime.streams[source_rank].stream()));
                    MFQ_CUDA_CHECK(cudaStreamWaitEvent(
                        source_stream.stream(),
                        runtime.completed[source_rank], 0));
                }
                {
                    MfqCudaGuard destination_guard(device);
                    const auto destination_stream =
                        mfq_get_current_cuda_stream(device);
                    MFQ_CUDA_CHECK(cudaEventRecord(
                        runtime.completed[destination_rank],
                        runtime.streams[destination_rank].stream()));
                    MFQ_CUDA_CHECK(cudaStreamWaitEvent(
                        destination_stream.stream(),
                        runtime.completed[destination_rank], 0));
                }
                return destination;
            }
        }
    }
#endif
    MfqCudaGuard guard(device);
    if (reusable_matches()) {
        reusable.copy_(value, true);
        return reusable;
    }
    return value.to(
        value.options().device(mfq_tensor_backend::Device(mfq_tensor_backend::kCUDA, device)),
        true, false).contiguous();
}

mfq_tensor_backend::Tensor reduce_model_parallel_outputs(
        std::vector<mfq_tensor_backend::Tensor> outputs) {
    if (outputs.empty()) {
        throw std::runtime_error(
            "cannot reduce an empty model-parallel output");
    }
#ifdef MFQ_HAVE_NCCL
    if (g_model_parallel_collectives.collectives_enabled &&
            outputs.size() ==
                g_model_parallel_collectives.devices.size()) {
        auto & runtime = g_model_parallel_collectives;
        const auto shape = outputs.front().sizes().vec();
        const auto output_dtype = outputs.front().scalar_type();
        const int64_t elements = outputs.front().numel();
        const bool reduce_to_primary =
            model_parallel_reduce_to_primary_enabled();
        // A two-input FP16 sum has the same final FP16 rounding as the former
        // FP32 reduction, while avoiding both conversion passes.
        const bool fp16_reduce =
            model_parallel_fp16_reduce_enabled() &&
            reduce_to_primary &&
            outputs.size() == 2 &&
            output_dtype == mfq_tensor_backend::kFloat16;
        const int primary = model_parallel_primary_device();
        const auto primary_rank_it = std::find(
            runtime.devices.begin(), runtime.devices.end(), primary);
        if (primary_rank_it == runtime.devices.end()) {
            throw std::runtime_error(
                "model-parallel primary device is absent from NCCL ranks");
        }
        const auto primary_rank = static_cast<size_t>(
            primary_rank_it - runtime.devices.begin());
        for (size_t index = 0; index < outputs.size(); ++index) {
            const int device = runtime.devices[index];
            if (!outputs[index].defined() || !outputs[index].is_cuda() ||
                    outputs[index].get_device() != device ||
                    outputs[index].sizes().vec() != shape ||
                    outputs[index].scalar_type() != output_dtype) {
                throw std::runtime_error(
                    "NCCL model-parallel reduction received mismatched shards");
            }
            MfqCudaGuard guard(device);
            if (fp16_reduce) {
                outputs[index] = outputs[index].contiguous();
            }
            const auto producer =
                mfq_get_current_cuda_stream(device);
            MFQ_CUDA_CHECK(cudaEventRecord(
                runtime.ready[index], producer.stream()));
            const auto communication = runtime.streams[index];
            MFQ_CUDA_CHECK(cudaStreamWaitEvent(
                communication.stream(), runtime.ready[index], 0));
            if (fp16_reduce) {
                mfq_cuda_record_stream(outputs[index], communication);
            } else {
                MfqCudaStreamGuard stream_guard(communication);
                auto & buffer = runtime.reduction_buffers[index];
                if (!buffer.defined() || buffer.sizes().vec() != shape ||
                        buffer.get_device() != device ||
                        buffer.scalar_type() != mfq_tensor_backend::kFloat32) {
                    buffer = mfq_tensor_backend::empty(
                        shape,
                        outputs[index].options().dtype(mfq_tensor_backend::kFloat32));
                }
                buffer.copy_(outputs[index], true);
                mfq_cuda_record_stream(outputs[index], communication);
            }
        }

        MFQ_NCCL_CHECK(ncclGroupStart());
        for (size_t index = 0; index < outputs.size(); ++index) {
            auto & buffer = runtime.reduction_buffers[index];
            if (reduce_to_primary) {
                void * reduction_data = fp16_reduce
                    ? outputs[index].data_ptr()
                    : buffer.data_ptr<float>();
                MFQ_NCCL_CHECK(ncclReduce(
                    reduction_data,
                    reduction_data,
                    static_cast<size_t>(elements),
                    fp16_reduce ? ncclFloat16 : ncclFloat32,
                    ncclSum,
                    static_cast<int>(primary_rank),
                    runtime.communicators[index],
                    runtime.streams[index].stream()));
            } else {
                MFQ_NCCL_CHECK(ncclAllReduce(
                    buffer.data_ptr<float>(),
                    buffer.data_ptr<float>(),
                    static_cast<size_t>(elements),
                    ncclFloat32,
                    ncclSum,
                    runtime.communicators[index],
                    runtime.streams[index].stream()));
            }
        }
        MFQ_NCCL_CHECK(ncclGroupEnd());

        mfq_tensor_backend::Tensor result;
        for (size_t index = 0; index < outputs.size(); ++index) {
            MfqCudaGuard guard(runtime.devices[index]);
            const auto communication = runtime.streams[index];
            if (runtime.devices[index] == primary) {
                if (fp16_reduce) {
                    result = outputs[index];
                } else {
                    MfqCudaStreamGuard stream_guard(communication);
                    result = runtime.reduction_buffers[index]
                        .to(output_dtype).contiguous();
                }
            }
            MFQ_CUDA_CHECK(cudaEventRecord(
                runtime.completed[index], communication.stream()));
        }
        MfqCudaGuard primary_guard(primary);
        const auto parent = mfq_get_current_cuda_stream(primary);
        MFQ_CUDA_CHECK(cudaStreamWaitEvent(
            parent.stream(), runtime.completed[primary_rank], 0));
        mfq_cuda_record_stream(result, parent);
        return result;
    }
#endif
    const int primary =
        model_parallel_primary_device();
    MfqCudaGuard primary_guard(primary);
    const auto output_dtype =
        outputs.front().scalar_type();
    mfq_tensor_backend::Tensor reduced;
    for (auto & output : outputs) {
        auto partial =
            tensor_to_cuda_device(output, primary)
                .to(mfq_tensor_backend::kFloat32);
        if (!reduced.defined()) {
            reduced = std::move(partial);
        } else {
            reduced.add_(partial);
        }
    }
    return reduced.to(output_dtype).contiguous();
}



std::shared_ptr<mfq::NintRows> load_nint_row_table(
        const mfq::ModelSource& source, const std::string& name) {
    const auto& metadata = require_tensor(source, name);
    MFQ_RUNTIME_CHECK(metadata.dtype == "NINT", "NINT row table requires packed NINT");
    MFQ_RUNTIME_CHECK(metadata.nbytes <= std::numeric_limits<std::size_t>::max(),
        "NINT row table size overflow");
    auto read = source.tensor_reader(name);
    return std::make_shared<mfq::NintRows>(static_cast<std::size_t>(metadata.nbytes),
        [read = std::move(read)](std::size_t offset, std::uint8_t* out, std::size_t count) {
            read(offset, reinterpret_cast<std::byte*>(out), count);
        });
}

mfq_tensor_backend::Tensor nint_row_embedding_lookup(
        const mfq::NintRows& table, const mfq_tensor_backend::Tensor& ids) {
    MFQ_RUNTIME_CHECK(ids.is_cuda() && ids.scalar_type() == mfq_tensor_backend::kInt64,
        "NINT row IDs must be CUDA int64");
    const MfqCudaGuard guard(ids.device());
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    MFQ_CUDA_CHECK(cudaStreamIsCapturing(mfq_current_cuda_stream(), &capture));
    MFQ_RUNTIME_CHECK(capture == cudaStreamCaptureStatusNone,
        "range-backed NINT row lookup cannot run inside CUDA graph capture");
    auto shape = ids.sizes().vec();
    shape.push_back(table.width());
    if (ids.numel() == 0)
        return mfq_tensor_backend::empty(shape, ids.options().dtype(mfq_tensor_backend::kFloat16));
    MFQ_RUNTIME_CHECK(ids.numel() <= std::numeric_limits<int>::max() / 6 &&
        static_cast<uint64_t>(ids.numel()) * table.width() <= std::numeric_limits<uint32_t>::max(),
        "NINT row lookup batch exceeds bounds");
    auto host = ids.contiguous().cpu();
    mfq::NintRowBatch selected;
    for (int64_t i = 0; i < host.numel(); ++i)
        table.append_row(host.data_ptr<int64_t>()[i], selected);
    selected.validate();
    auto packed = cpu_u8_tensor(selected.packed(), {static_cast<int64_t>(selected.packed_nbytes())}).to(ids.device());
    std::vector<int32_t> words(selected.descriptors().size());
    std::memcpy(words.data(), selected.descriptors().data(), words.size() * sizeof(uint32_t));
    auto descriptors = cpu_i32_tensor(words, {static_cast<int64_t>(selected.rows()), 6}).to(ids.device());
    return nint_selected_rows_cuda(packed, descriptors, table.width()).reshape(shape);
}

mfq_tensor_backend::Tensor quant_embedding_lookup(
        const QuantLinear & embedding,
        mfq_tensor_backend::Tensor token_ids) {
    MFQ_RUNTIME_CHECK(
        !embedding.tensor_parallel(),
        "quantized embedding does not support tensor-parallel shards");
    if (embedding.is_dense()) {
        auto output_shape = token_ids.sizes().vec();
        output_shape.push_back(embedding.dense.size(1));
        return embedding.dense.index_select(
            0, token_ids.reshape({-1})).reshape(output_shape);
    }
    if (embedding.is_nvq()) {
        return nvq_embedding(embedding.nvq.w, token_ids);
    }
    if (embedding.is_nint()) {
        if (embedding.nint.w.q8_zero) {
            return nint8_zero_embedding_lookup_cuda(
                embedding.nint.w.q_packed,
                embedding.nint.w.q8_zero_scale,
                token_ids, embedding.nint.w.neuron_len);
        }
        return nint_embedding_cuda(
            embedding.nint.w.q_packed,
            embedding.nint.w.row_q_bits,
            embedding.nint.w.row_q_bit_offsets,
            embedding.nint.w.sub_scale,
            embedding.nint.w.sub_min,
            embedding.nint.w.neuron_scale,
            embedding.nint.w.neuron_min,
            token_ids, embedding.nint.w.neuron_len,
            embedding.nint.w.gs);
    }
    if (embedding.is_mxfp4()) {
        return mxfp4_embedding_lookup_cuda(
            embedding.mxfp4.weight.values,
            embedding.mxfp4.weight.scales, token_ids);
    }
    if (embedding.is_mxfp8()) {
        return mxfp8_embedding_lookup_cuda(
            embedding.mxfp8.weight.values,
            embedding.mxfp8.weight.scales, token_ids);
    }
    MFQ_RUNTIME_CHECK(
        !embedding.is_mxfp4_sq() && !embedding.is_fp8_sq(),
        "SQ tensors do not support embedding lookup");
    MFQ_RUNTIME_CHECK(embedding.is_tpq(), "unsupported quantized embedding kind");
    if (embedding.tpq.weight.int4) {
        return tpq_int4_embedding_lookup_cuda(
            embedding.tpq.weight.packed,
            embedding.tpq.weight.scales, token_ids,
            embedding.tpq.weight.group_size);
    }
    return tpq_pq_embedding_lookup_cuda(
        embedding.tpq.weight.packed,
        embedding.tpq.weight.codebook, token_ids,
        embedding.tpq.weight.out,
        embedding.tpq.weight.neuron_len,
        embedding.tpq.weight.vector_size,
        embedding.tpq.weight.index_bits);
}

bool tensor_parallel_output_projections_compatible(
        const QuantLinearProjectionRefs & projections) {
    if (projections.size() < 2 ||
            !std::all_of(
                projections.begin(), projections.end(),
                [](const QuantLinear * layer) {
                    return layer != nullptr &&
                        layer->tensor_parallel() &&
                        layer->tensor_parallel_axis ==
                            TensorParallelAxis::Output;
                })) {
        return false;
    }
    const size_t shard_count =
        projections.front()->tensor_parallel_shards.size();
    if (shard_count < 2) return false;
    for (const auto * layer : projections) {
        if (layer->tensor_parallel_shards.size() != shard_count) {
            return false;
        }
        for (size_t shard = 0; shard < shard_count; ++shard) {
            const auto & candidate =
                layer->tensor_parallel_shards[shard];
            const auto & reference =
                projections.front()->tensor_parallel_shards[shard];
            if (candidate.device != reference.device ||
                    candidate.input_begin != reference.input_begin ||
                    candidate.input_end != reference.input_end) {
                return false;
            }
        }
    }
    return true;
}

std::vector<mfq_tensor_backend::Tensor>
forward_tensor_parallel_output_projections(
        mfq_tensor_backend::Tensor x,
        const QuantLinearProjectionRefs & projections) {
    MFQ_RUNTIME_CHECK(
        tensor_parallel_output_projections_compatible(projections),
        "tensor-parallel output projections are incompatible");
    auto output_shape = x.sizes().vec();
    auto flat = x.reshape({-1, x.size(-1)});
    const size_t shard_count =
        projections.front()->tensor_parallel_shards.size();
    if (g_decode_graph_tp_projection_major) {
        std::vector<mfq_tensor_backend::Tensor> local_inputs(shard_count);
        for (size_t launch_position = 0;
             launch_position < shard_count; ++launch_position) {
            const size_t shard = model_parallel_launch_index(
                launch_position, shard_count);
            const int device =
                projections.front()->tensor_parallel_shards[shard].device;
            MfqCudaGuard guard(device);
            local_inputs[shard] = tensor_to_cuda_device(flat, device);
        }

        const int primary = model_parallel_primary_device();
        std::vector<mfq_tensor_backend::Tensor> result;
        result.reserve(projections.size());
        for (const auto * projection : projections) {
            std::vector<mfq_tensor_backend::Tensor> local_outputs(shard_count);
            for (size_t launch_position = 0;
                 launch_position < shard_count; ++launch_position) {
                const size_t shard = model_parallel_launch_index(
                    launch_position, shard_count);
                const auto & weight =
                    projection->tensor_parallel_shards[shard];
                MfqCudaGuard guard(weight.device);
                local_outputs[shard] = run_quant_linear_shard(
                    weight, local_inputs[shard]);
            }
            MfqCudaGuard primary_guard(primary);
            std::vector<mfq_tensor_backend::Tensor> gathered;
            gathered.reserve(shard_count);
            for (auto & output : local_outputs) {
                gathered.push_back(tensor_to_cuda_device(output, primary));
            }
            auto combined = mfq_tensor_backend::cat(
                gathered, -1).contiguous();
            auto shape = output_shape;
            shape.back() = combined.size(-1);
            result.push_back(combined.reshape(shape));
        }
        return result;
    }
    std::vector<std::vector<mfq_tensor_backend::Tensor>>
        local_outputs(projections.size());
    for (auto & outputs : local_outputs) {
        outputs.resize(shard_count);
    }
    for (size_t launch_position = 0;
         launch_position < shard_count; ++launch_position) {
        const size_t shard = model_parallel_launch_index(
            launch_position, shard_count);
        const int device =
            projections.front()->tensor_parallel_shards[shard].device;
        MfqCudaGuard guard(device);
        // All projections consume the same immutable activation. Transfer it
        // once per rank, then keep the independent local projection work on
        // that rank before gathering each output.
        auto local_x = tensor_to_cuda_device(flat, device);
        for (size_t projection = 0;
                projection < projections.size(); ++projection) {
            local_outputs[projection][shard] =
                run_quant_linear_shard(
                    projections[projection]
                        ->tensor_parallel_shards[shard],
                    local_x);
        }
    }

    const int primary = model_parallel_primary_device();
    MfqCudaGuard primary_guard(primary);
    std::vector<mfq_tensor_backend::Tensor> result;
    result.reserve(projections.size());
    for (auto & projection_outputs : local_outputs) {
        std::vector<mfq_tensor_backend::Tensor> gathered;
        gathered.reserve(shard_count);
        for (auto & output : projection_outputs) {
            gathered.push_back(
                tensor_to_cuda_device(output, primary));
        }
        auto combined = mfq_tensor_backend::cat(
            gathered, -1).contiguous();
        auto shape = output_shape;
        shape.back() = combined.size(-1);
        result.push_back(combined.reshape(shape));
    }
    return result;
}



static bool is_nint_linear_dtype(const std::string & dtype) {
    return dtype == "NINT";
}

static bool is_nvq_linear_dtype(const std::string & dtype) {
    return dtype == "NVQ" || dtype == "NPQ";
}

static TensorParallelAxis infer_tensor_parallel_axis(
        const std::string & name) {
    const auto ends_with = [&name](std::string_view suffix) {
        return name.size() >= suffix.size() &&
            std::string_view(name).substr(name.size() - suffix.size()) == suffix;
    };
    if (name == "model.token_embedding.weight" ||
        ends_with(".token_embedding.weight") ||
        ends_with(".text_embedding.weight") ||
        (name.find(".code_embedding.") != std::string::npos &&
         ends_with(".weight"))) {
        return TensorParallelAxis::Mirrored;
    }
    if (name == "model.output.weight" ||
        name.find(".code_output.") != std::string::npos) {
        return TensorParallelAxis::Output;
    }
    if (ends_with(".mlp.down.weight") ||
        ends_with(".mlp.shared_expert.down.weight") ||
        ends_with(".attention.output.weight") ||
        ends_with(".linear_attention.output.weight") ||
        ends_with(".attention.output_a.weight") ||
        ends_with(".attention.output_b.weight") ||
        ends_with(".projector.output.weight")) {
        return TensorParallelAxis::Input;
    }
    return TensorParallelAxis::Output;
}

static std::vector<mfq::TensorParallelSlice>
plan_parallel_slices(
        int64_t extent,
        int64_t preferred_granularity,
        const ParallelConfig & config,
        const std::string & name) {
    (void)name;
    if (!config.enabled()) {
        throw std::runtime_error(
            "parallel slice planning requires at least two ranks");
    }
    int64_t granularity = std::max<int64_t>(
        1, std::min<int64_t>(
            preferred_granularity,
            extent / static_cast<int64_t>(config.devices.size())));
    while (granularity > 1 &&
           extent < static_cast<int64_t>(config.devices.size()) *
               granularity) {
        granularity /= 2;
    }
    auto slices = mfq::plan_tensor_parallel_slices(
        extent, granularity, config.devices, config.split);
    mfq::validate_tensor_parallel_slices(
        slices, extent, granularity);
    return slices;
}

static std::vector<mfq::TensorParallelSlice>
plan_quant_tensor_parallel_slices(
        int64_t extent,
        int64_t preferred_granularity,
        const std::string & name) {
    return plan_parallel_slices(
        extent, preferred_granularity,
        g_tensor_parallel, name);
}

static std::vector<mfq::TensorParallelSlice>
plan_moe_expert_parallel_slices(
        int64_t extent,
        const std::string & name) {
    return plan_parallel_slices(
        extent, 1, moe_parallel_config(), name);
}

QuantLinear load_quant_linear(
        const mfq::ModelSource & mfq,
        const std::string & name,
        std::optional<TensorParallelAxis> axis_override,
        const std::vector<mfq::TensorParallelSlice> *
            slices_override) {
    const auto & dtype = require_tensor(mfq, name).dtype;
    QuantLinear result;
    const TensorParallelAxis axis =
        axis_override.value_or(
            infer_tensor_parallel_axis(name));
    if (slices_override != nullptr &&
        axis != TensorParallelAxis::Output) {
        throw std::runtime_error(
            "explicit tensor-parallel slices require an output-axis weight");
    }
    auto select_slices = [&](
            int64_t extent,
            int64_t preferred) {
        if (slices_override == nullptr) {
            return plan_quant_tensor_parallel_slices(
                extent, preferred, name);
        }
        auto slices = *slices_override;
        mfq::validate_tensor_parallel_slices(
            slices, extent, 1);
        if (slices.size() !=
            g_tensor_parallel.devices.size()) {
            throw std::runtime_error(
                "explicit tensor-parallel slice count mismatch");
        }
        for (size_t index = 0;
             index < slices.size(); ++index) {
            if (slices[index].device !=
                g_tensor_parallel.devices[index]) {
                throw std::runtime_error(
                    "explicit tensor-parallel device order mismatch");
            }
        }
        return slices;
    };
    result.tensor_parallel_axis = axis;
    if (is_nint_linear_dtype(dtype)) {
        result.kind = QuantLinearKind::Nint;
        const auto blob = read_tensor(mfq, name);
        if (dtype == "NINT8-0") {
            const auto cpu = unpack_nint8_zero(blob);
            result.logical_out = cpu.out;
            result.logical_neuron_len = cpu.neuron_len;
            if (g_tensor_parallel.enabled() &&
                axis != TensorParallelAxis::Mirrored) {
                const int64_t extent =
                    axis == TensorParallelAxis::Output
                    ? cpu.out : cpu.ng;
                const int64_t preferred =
                    axis == TensorParallelAxis::Output
                    ? 128
                    : 4;
                for (const auto & slice :
                     select_slices(
                         extent, preferred)) {
                    auto shard_cpu =
                        axis == TensorParallelAxis::Output
                        ? slice_nint8_zero_cpu_output(
                            cpu, slice.begin, slice.end)
                        : slice_nint8_zero_cpu_input_groups(
                            cpu, slice.begin, slice.end);
                    QuantLinearShard shard;
                    shard.device = slice.device;
                    shard.kind = QuantLinearKind::Nint;
                    shard.output_begin =
                        axis == TensorParallelAxis::Output
                        ? slice.begin : 0;
                    shard.output_end =
                        axis == TensorParallelAxis::Output
                        ? slice.end : cpu.out;
                    shard.input_begin =
                        axis == TensorParallelAxis::Input
                        ? slice.begin * 32 : 0;
                    shard.input_end =
                        axis == TensorParallelAxis::Input
                        ? std::min<int64_t>(
                            slice.end * 32,
                            cpu.neuron_len)
                        : cpu.neuron_len;
                    shard.nint =
                        to_cuda_device_nint8_zero(
                            shard_cpu, slice.device);
                    result.tensor_parallel_shards.push_back(
                        std::move(shard));
                }
            } else {
                if (g_loading_cpu_layer) {
                    result.nint.w = to_device_nint8_zero(cpu, false);
                } else {
                    MfqCudaGuard guard(
                        active_weight_load_device());
                    result.nint.w =
                        to_device_nint8_zero(cpu, true);
                }
            }
        } else {
            const auto cpu = unpack_nint(blob);
            result.logical_out = cpu.out;
            result.logical_neuron_len = cpu.neuron_len;
            if (g_tensor_parallel.enabled() &&
                axis != TensorParallelAxis::Mirrored) {
                const int64_t extent =
                    axis == TensorParallelAxis::Output
                    ? cpu.out : cpu.ng;
                const int64_t preferred =
                    axis == TensorParallelAxis::Output
                    ? 128
                    : std::lcm<int64_t>(cpu.gs, 128) / cpu.gs;
                for (const auto & slice :
                     select_slices(
                         extent, preferred)) {
                    auto shard_cpu =
                        axis == TensorParallelAxis::Output
                        ? slice_nint_cpu_output(
                            cpu, slice.begin, slice.end)
                        : slice_nint_cpu_input_groups(
                            cpu, slice.begin, slice.end);
                    QuantLinearShard shard;
                    shard.device = slice.device;
                    shard.kind = QuantLinearKind::Nint;
                    shard.output_begin =
                        axis == TensorParallelAxis::Output
                        ? slice.begin : 0;
                    shard.output_end =
                        axis == TensorParallelAxis::Output
                        ? slice.end : cpu.out;
                    shard.input_begin =
                        axis == TensorParallelAxis::Input
                        ? slice.begin * cpu.gs : 0;
                    shard.input_end =
                        axis == TensorParallelAxis::Input
                        ? std::min<int64_t>(
                            slice.end * cpu.gs,
                            cpu.neuron_len)
                        : cpu.neuron_len;
                    shard.nint = to_cuda_device_nint(
                        shard_cpu, slice.device);
                    result.tensor_parallel_shards.push_back(
                        std::move(shard));
                }
            } else {
                if (g_loading_cpu_layer) {
                    result.nint.w = to_device_nint(cpu, false);
                } else {
                    MfqCudaGuard guard(
                        active_weight_load_device());
                    result.nint.w =
                        to_device_nint(cpu, true);
                }
            }
        }
    } else if (is_nvq_linear_dtype(dtype)) {
        result.kind = QuantLinearKind::Nvq;
        const auto cpu =
            unpack_nvq(read_tensor(mfq, name), dtype);
        result.logical_out = cpu.out;
        result.logical_neuron_len = cpu.neuron_len;
        if (g_tensor_parallel.enabled() &&
            axis != TensorParallelAxis::Mirrored) {
            const int64_t extent =
                axis == TensorParallelAxis::Output
                ? cpu.out : cpu.ng;
            const int64_t preferred =
                axis == TensorParallelAxis::Output
                ? 128
                : std::lcm<int64_t>(cpu.gs, 128) / cpu.gs;
            for (const auto & slice :
                 select_slices(
                     extent, preferred)) {
                auto shard_cpu = slice_nvq_cpu(
                    cpu, axis, slice.begin, slice.end);
                QuantLinearShard shard;
                shard.device = slice.device;
                shard.kind = QuantLinearKind::Nvq;
                shard.output_begin =
                    axis == TensorParallelAxis::Output
                    ? slice.begin : 0;
                shard.output_end =
                    axis == TensorParallelAxis::Output
                    ? slice.end : cpu.out;
                shard.input_begin =
                    axis == TensorParallelAxis::Input
                    ? slice.begin * cpu.gs : 0;
                shard.input_end =
                    axis == TensorParallelAxis::Input
                    ? std::min<int64_t>(
                        slice.end * cpu.gs,
                        cpu.neuron_len)
                    : cpu.neuron_len;
                shard.nvq = to_cuda_device_nvq(
                    shard_cpu, slice.device);
                result.tensor_parallel_shards.push_back(
                    std::move(shard));
            }
        } else {
            if (g_loading_cpu_layer) {
                result.nvq.w = to_device_nvq(cpu, false);
            } else {
                MfqCudaGuard guard(
                    active_weight_load_device());
                result.nvq.w = to_device_nvq(cpu, true);
            }
        }
    } else if (dtype == "MXFP4-SQ") {
        result.kind = QuantLinearKind::Mxfp4Sq;
        const auto payload = read_tensor(mfq, name);
        const auto layout = mfq::sq::parse(payload.data(), payload.size());
        result.logical_out = layout.outputs;
        result.logical_neuron_len = layout.width;
        if (!g_loading_cpu_layer && g_tensor_parallel.enabled() && axis != TensorParallelAxis::Mirrored) {
            const bool output_axis = axis == TensorParallelAxis::Output;
            for (const auto& slice : select_slices(output_axis ? layout.outputs : layout.width, output_axis ? 128 : 32)) {
                QuantLinearShard shard;
                shard.kind = result.kind;
                shard.device = slice.device;
                shard.output_begin = output_axis ? slice.begin : 0;
                shard.output_end = output_axis ? slice.end : layout.outputs;
                shard.input_begin = output_axis ? 0 : slice.begin;
                shard.input_end = output_axis ? layout.width : slice.end;
                std::vector<int64_t> rows(shard.output_end - shard.output_begin);
                std::iota(rows.begin(), rows.end(), shard.output_begin);
                shard.mxfp4_sq.weight = to_device_mxfp4_sq(
                    mfq::sq::select_rows(payload, rows, shard.input_begin, shard.input_end), true, slice.device);
                result.tensor_parallel_shards.push_back(std::move(shard));
            }
        } else {
            result.mxfp4_sq.weight = to_device_mxfp4_sq(payload, !g_loading_cpu_layer, active_weight_load_device());
        }
    } else if (mfq::fp8sq::is_dtype(dtype)) {
        result.kind = QuantLinearKind::Fp8Sq;
        const auto payload = read_tensor(mfq, name);
        const auto layout = mfq::fp8sq::parse(dtype, payload.data(), payload.size());
        result.logical_out = layout.outputs;
        result.logical_neuron_len = layout.width;
        if (!g_loading_cpu_layer && g_tensor_parallel.enabled() && axis != TensorParallelAxis::Mirrored) {
            const bool output_axis = axis == TensorParallelAxis::Output;
            const auto alignment = output_axis ? layout.block_rows : layout.block_columns;
            for (const auto& slice : select_slices(output_axis ? layout.outputs : layout.width, alignment)) {
                QuantLinearShard shard;
                shard.kind = result.kind;
                shard.device = slice.device;
                shard.output_begin = output_axis ? slice.begin : 0;
                shard.output_end = output_axis ? slice.end : layout.outputs;
                shard.input_begin = output_axis ? 0 : slice.begin;
                shard.input_end = output_axis ? layout.width : slice.end;
                shard.fp8_sq.weight = to_device_fp8_sq(dtype,
                    mfq::fp8sq::slice(dtype, payload, shard.output_begin, shard.output_end,
                        shard.input_begin, shard.input_end), true, slice.device);
                result.tensor_parallel_shards.push_back(std::move(shard));
            }
        } else {
            result.fp8_sq.weight = to_device_fp8_sq(dtype, payload, !g_loading_cpu_layer, active_weight_load_device());
        }
    } else if (dtype == "MXFP4") {
        result.kind = QuantLinearKind::Mxfp4;
        const auto cpu = unpack_mxfp4(read_tensor(mfq, name));
        result.logical_out = cpu.out;
        result.logical_neuron_len = cpu.neuron_len;
        if (g_tensor_parallel.enabled() &&
                axis != TensorParallelAxis::Mirrored) {
            const int64_t extent = axis == TensorParallelAxis::Output
                ? cpu.out : cpu.neuron_len;
            const int64_t preferred = axis == TensorParallelAxis::Output
                ? 128 : 32;
            for (const auto & slice : select_slices(extent, preferred)) {
                auto shard_cpu = slice_mxfp4_cpu(
                    cpu, axis, slice.begin, slice.end);
                QuantLinearShard shard;
                shard.device = slice.device;
                shard.kind = QuantLinearKind::Mxfp4;
                shard.output_begin = axis == TensorParallelAxis::Output
                    ? slice.begin : 0;
                shard.output_end = axis == TensorParallelAxis::Output
                    ? slice.end : cpu.out;
                shard.input_begin = axis == TensorParallelAxis::Input
                    ? slice.begin : 0;
                shard.input_end = axis == TensorParallelAxis::Input
                    ? slice.end : cpu.neuron_len;
                shard.mxfp4 = to_device_mxfp4(
                    shard_cpu, true, slice.device);
                result.tensor_parallel_shards.push_back(std::move(shard));
            }
        } else {
            result.mxfp4.weight = to_device_mxfp4(
                cpu, !g_loading_cpu_layer,
                g_loading_cpu_layer ? -1 : active_weight_load_device());
        }
    } else if (dtype == "MXFP8") {
        result.kind = QuantLinearKind::Mxfp8;
        const auto cpu = unpack_mxfp8(read_tensor(mfq, name));
        result.logical_out = cpu.out;
        result.logical_neuron_len = cpu.neuron_len;
        if (g_tensor_parallel.enabled() &&
                axis != TensorParallelAxis::Mirrored) {
            const int64_t extent = axis == TensorParallelAxis::Output
                ? cpu.out : cpu.neuron_len;
            for (const auto & slice : select_slices(extent, 128)) {
                auto shard_cpu = slice_mxfp8_cpu(
                    cpu, axis, slice.begin, slice.end);
                QuantLinearShard shard;
                shard.device = slice.device;
                shard.kind = QuantLinearKind::Mxfp8;
                shard.output_begin = axis == TensorParallelAxis::Output
                    ? slice.begin : 0;
                shard.output_end = axis == TensorParallelAxis::Output
                    ? slice.end : cpu.out;
                shard.input_begin = axis == TensorParallelAxis::Input
                    ? slice.begin : 0;
                shard.input_end = axis == TensorParallelAxis::Input
                    ? slice.end : cpu.neuron_len;
                shard.mxfp8 = to_cuda_device_mxfp8(
                    shard_cpu, slice.device);
                result.tensor_parallel_shards.push_back(
                    std::move(shard));
            }
        } else if (g_loading_cpu_layer) {
            result.mxfp8.weight = to_device_mxfp8(cpu, false);
        } else {
            result.mxfp8.weight = to_cuda_device_mxfp8(
                cpu, active_weight_load_device());
        }
    } else if (dtype == "TPQ-I4G64" || is_tpq_pq_dtype(dtype)) {
        result.kind = QuantLinearKind::Tpq;
        const auto cpu = dtype == "TPQ-I4G64"
            ? unpack_tpq_int4(read_tensor(mfq, name))
            : unpack_tpq_pq(read_tensor(mfq, name), dtype);
        result.logical_out = cpu.out;
        result.logical_neuron_len = cpu.neuron_len;
        if (g_tensor_parallel.enabled() &&
                axis != TensorParallelAxis::Mirrored) {
            const int64_t extent = axis == TensorParallelAxis::Output
                ? cpu.out : cpu.neuron_len;
            const int64_t preferred = axis == TensorParallelAxis::Output
                ? 8 : (cpu.int4 ? cpu.group_size : cpu.vector_size);
            for (const auto & slice : select_slices(extent, preferred)) {
                auto shard_cpu = slice_tpq_cpu(
                    cpu, axis, slice.begin, slice.end);
                QuantLinearShard shard;
                shard.device = slice.device;
                shard.kind = QuantLinearKind::Tpq;
                shard.output_begin = axis == TensorParallelAxis::Output
                    ? slice.begin : 0;
                shard.output_end = axis == TensorParallelAxis::Output
                    ? slice.end : cpu.out;
                shard.input_begin = axis == TensorParallelAxis::Input
                    ? slice.begin : 0;
                shard.input_end = axis == TensorParallelAxis::Input
                    ? slice.end : cpu.neuron_len;
                shard.tpq = to_device_tpq(
                    shard_cpu, true, slice.device);
                result.tensor_parallel_shards.push_back(
                    std::move(shard));
            }
        } else {
            result.tpq.weight = to_device_tpq(
                cpu, !g_loading_cpu_layer,
                g_loading_cpu_layer ? -1 : active_weight_load_device());
        }
    } else if (dtype == "BF16" || dtype == "F16" || dtype == "F32") {
        result.kind = QuantLinearKind::Dense;
        result.dense_small_m_rowwise = name.rfind("predictor.", 0) == 0;
        auto cpu = load_dense_linear_cpu(mfq, name);
        result.logical_out = cpu.size(0);
        result.logical_neuron_len = cpu.size(1);
        // Keep native floating-point linears whole on the primary TP rank.
        // Splitting these matrices changes the cuBLAS GEMM geometry and causes
        // materially larger drift than the weight formats under test. They
        // are a small fraction of the model; routed experts and MXFP8/NINT/NVQ
        // weights remain sharded.
        const char * shard_native_float_env =
            std::getenv("MFQ_TP_SHARD_NATIVE_FLOAT");
        const bool shard_native_float =
            shard_native_float_env != nullptr &&
            std::atoi(shard_native_float_env) != 0;
        if (shard_native_float && g_tensor_parallel.enabled() &&
                axis != TensorParallelAxis::Mirrored) {
            const int64_t extent = axis == TensorParallelAxis::Output
                ? cpu.size(0) : cpu.size(1);
            for (const auto & slice : select_slices(extent, 128)) {
                QuantLinearShard shard;
                shard.device = slice.device;
                shard.kind = QuantLinearKind::Dense;
                shard.output_begin = axis == TensorParallelAxis::Output
                    ? slice.begin : 0;
                shard.output_end = axis == TensorParallelAxis::Output
                    ? slice.end : cpu.size(0);
                shard.input_begin = axis == TensorParallelAxis::Input
                    ? slice.begin : 0;
                shard.input_end = axis == TensorParallelAxis::Input
                    ? slice.end : cpu.size(1);
                const int64_t dimension =
                    axis == TensorParallelAxis::Output ? 0 : 1;
                MfqCudaGuard guard(slice.device);
                shard.dense = cpu.narrow(
                        dimension, slice.begin,
                        slice.end - slice.begin)
                    .to(mfq_tensor_backend::Device(mfq_tensor_backend::kCUDA, slice.device))
                    .contiguous();
                result.tensor_parallel_shards.push_back(
                    std::move(shard));
            }
        } else if (g_loading_cpu_layer) {
            result.dense = cpu.contiguous();
        } else {
            MfqCudaGuard guard(active_weight_load_device());
            result.dense = cpu.to(mfq_tensor_backend::kCUDA).contiguous();
        }
    } else {
        throw std::runtime_error(
            "linear tensor must be NINT/NVQ/MXFP4-SQ/MXFP8-SQ/FP8-128SQ/MXFP4/MXFP8/TPQ/BF16/F16/F32: " +
            name + " dtype=" + dtype);
    }
    return result;
}

bool is_quant_dtype(const std::string & dtype) {
    return is_nint_linear_dtype(dtype) ||
        is_nvq_linear_dtype(dtype) || dtype == "MXFP4-SQ" ||
        mfq::fp8sq::is_dtype(dtype) ||
        dtype == "MXFP4" || dtype == "MXFP8" ||
        dtype == "TPQ-I4G64" || is_tpq_pq_dtype(dtype);
}

QuantLinearGroup make_quant_group(
        std::vector<QuantLinear> layers,
        bool preserve_projection_boundaries) {
    if (layers.empty()) throw std::runtime_error("empty quantized linear group");
    QuantLinearGroup result;
    result.decode_branch_parallel = !preserve_projection_boundaries;
    result.outs.reserve(layers.size());
    bool all_nint = true;
    std::vector<NintWeight> nint_weights;
    nint_weights.reserve(layers.size());
    for (const auto & layer : layers) {
        result.outs.push_back(layer.out());
        all_nint = all_nint && layer.is_nint();
        if (layer.is_nint() && !layer.tensor_parallel()) {
            nint_weights.push_back(layer.nint.w);
        }
    }
    const char * disable_nint_group =
        std::getenv("MFQ_DIAGNOSTIC_DISABLE_NINT_GROUP");
    const bool diagnostic_keep_nint_separate =
        disable_nint_group != nullptr && disable_nint_group[0] == '1';
    const bool any_tensor_parallel = std::any_of(
        layers.begin(), layers.end(),
        [](const QuantLinear & layer) {
            return layer.tensor_parallel();
        });
    if (all_nint && !preserve_projection_boundaries &&
        !diagnostic_keep_nint_separate && !g_loading_cpu_layer &&
        !any_tensor_parallel) {
        result.nint_grouped = true;
        result.nint = make_linear_group(nint_weights);
    } else {
        result.layers = std::move(layers);
        result.nvq_prefix2 = !g_loading_cpu_layer && result.layers.size() >= 2 &&
            !any_tensor_parallel &&
            result.layers[0].is_nvq() && result.layers[1].is_nvq() &&
            nvq_pair_compatible(result.layers[0].nvq.w, result.layers[1].nvq.w);
    }
    return result;
}

static bool quant_linear_pair_compatible(const QuantLinear & a, const QuantLinear & b) {
    if (a.tensor_parallel() || b.tensor_parallel()) {
        return a.tensor_parallel() && b.tensor_parallel() &&
            a.kind == b.kind &&
            a.tensor_parallel_axis == b.tensor_parallel_axis &&
            a.tensor_parallel_shards.size() ==
                b.tensor_parallel_shards.size();
    }
    if (a.kind != b.kind) return false;
    if (a.is_nvq()) return nvq_pair_compatible(a.nvq.w, b.nvq.w);
    if (a.is_mxfp8()) {
        return a.mxfp8.weight.neuron_len == b.mxfp8.weight.neuron_len;
    }
    if (a.is_mxfp4()) {
        return a.mxfp4.weight.neuron_len == b.mxfp4.weight.neuron_len;
    }
    if (a.is_mxfp4_sq()) {
        return a.mxfp4_sq.weight.neuron_len ==
            b.mxfp4_sq.weight.neuron_len;
    }
    if (a.is_fp8_sq()) {
        return a.fp8_sq.weight.dtype == b.fp8_sq.weight.dtype &&
            a.fp8_sq.weight.neuron_len == b.fp8_sq.weight.neuron_len &&
            a.fp8_sq.weight.block_rows == b.fp8_sq.weight.block_rows &&
            a.fp8_sq.weight.block_columns == b.fp8_sq.weight.block_columns &&
            a.fp8_sq.weight.scale_kind == b.fp8_sq.weight.scale_kind;
    }
    if (a.is_tpq()) {
        return a.tpq.weight.int4 == b.tpq.weight.int4 &&
            a.tpq.weight.neuron_len == b.tpq.weight.neuron_len &&
            a.tpq.weight.group_size == b.tpq.weight.group_size &&
            a.tpq.weight.vector_size == b.tpq.weight.vector_size &&
            a.tpq.weight.index_bits == b.tpq.weight.index_bits;
    }
    if (a.is_dense()) {
        return a.dense.size(1) == b.dense.size(1);
    }
    const auto & x = a.nint.w;
    const auto & y = b.nint.w;
    return x.ng == y.ng && x.gs == y.gs &&
        x.neuron_len == y.neuron_len && x.q8_zero == y.q8_zero;
}

QuantLinearGroup load_quant_group(
    const mfq::ModelSource & mfq, const std::vector<std::string> & names,
    size_t required_compatible_prefix,
    const std::vector<mfq::TensorParallelSlice> *
        slices_override,
    bool preserve_projection_boundaries) {
    std::vector<QuantLinear> layers;
    layers.reserve(names.size());
    for (const auto & name : names) {
        layers.push_back(
            load_quant_linear(
                mfq, name,
                slices_override != nullptr
                    ? std::optional<TensorParallelAxis>(
                        TensorParallelAxis::Output)
                    : std::nullopt,
                slices_override));
    }
    if (required_compatible_prefix > layers.size()) {
        throw std::runtime_error("invalid required quantized-group prefix length");
    }
    // A compatible prefix is fused opportunistically by make_quant_group().
    // Mixed-precision Q/K and gate/up pairs remain separate QuantLinear
    // branches and preserve the recipe-selected layouts.
    return make_quant_group(
        std::move(layers), preserve_projection_boundaries);
}

static std::vector<mfq::TensorParallelSlice>
tensor_parallel_output_slices_for_input(
        const QuantLinear & input_parallel) {
    if (!input_parallel.tensor_parallel() ||
        input_parallel.tensor_parallel_axis !=
            TensorParallelAxis::Input) {
        return {};
    }
    std::vector<mfq::TensorParallelSlice> result;
    result.reserve(
        input_parallel.tensor_parallel_shards.size());
    for (const auto & shard :
         input_parallel.tensor_parallel_shards) {
        result.push_back({
            shard.device,
            shard.input_begin,
            shard.input_end,
        });
    }
    mfq::validate_tensor_parallel_slices(
        result,
        input_parallel.neuron_len(),
        1);
    return result;
}

QuantLinearGroup load_paired_gate_up(
        const mfq::ModelSource & mfq,
        const std::vector<std::string> & names,
        const QuantLinear & down,
        size_t required_compatible_prefix,
        bool preserve_projection_boundaries) {
    auto slices =
        tensor_parallel_output_slices_for_input(
            down);
    return slices.empty()
        ? load_quant_group(
            mfq, names,
            required_compatible_prefix,
            nullptr,
            preserve_projection_boundaries)
        : load_quant_group(
            mfq, names,
            required_compatible_prefix,
            &slices,
            preserve_projection_boundaries);
}



DenseLinearGroup make_dense_group(const std::vector<mfq_tensor_backend::Tensor> & ws) {
    if (ws.empty()) throw std::runtime_error("empty dense group");
    DenseLinearGroup g;
    std::vector<mfq_tensor_backend::Tensor> parts;
    parts.reserve(ws.size());
    for (const auto & w : ws) {
        if (w.dim() != 2) throw std::runtime_error("dense linear group expects 2D weights");
        if (!parts.empty() && w.size(1) != parts[0].size(1)) {
            throw std::runtime_error("cannot group dense tensors with different input width");
        }
        g.outs.push_back(w.size(0));
        parts.push_back(w.to(mfq_tensor_backend::kFloat32));
    }
    g.w = mfq_tensor_backend::cat(parts, 0).contiguous();
    return g;
}

static mfq_tensor_backend::Tensor dequant_sq_cpu(const Mxfp4SqWeight& weight) {
    auto output = mfq_tensor_backend::empty({weight.out, weight.neuron_len},
        weight.blob.options().dtype(mfq_tensor_backend::kFloat32));
    const auto* blob = weight.blob.data_ptr<uint8_t>();
    const auto layout = mfq::sq::parse(blob, weight.blob.numel());
    mfq_parallel_for(0, weight.out, 1, [&](int64_t begin, int64_t end) {
        for (auto n = begin; n < end; ++n) mfq::sq::decode_cpu_row(blob, layout,
            weight.row_q.data_ptr<uint8_t>()[n], weight.row_symbol_byte_offsets.data_ptr<int32_t>()[n],
            weight.row_auxiliary.data_ptr<int32_t>()[n], output.data_ptr<float>() + n * weight.neuron_len);
    });
    return output;
}

static mfq_tensor_backend::Tensor dequant_sq_cpu(const Fp8SqWeight& weight) {
    auto output = mfq_tensor_backend::empty({weight.out, weight.neuron_len},
        weight.blob.options().dtype(mfq_tensor_backend::kFloat32));
    const auto* blob = weight.blob.data_ptr<uint8_t>();
    const auto layout = mfq::fp8sq::parse(weight.dtype, blob, weight.blob.numel());
    mfq_parallel_for(0, weight.out, 1, [&](int64_t begin, int64_t end) {
        for (auto n = begin; n < end; ++n) for (int64_t k = 0; k < weight.neuron_len; ++k)
            output.data_ptr<float>()[n * weight.neuron_len + k] = mfq::fp8sq::decode_cpu(blob, layout,
                weight.row_q.data_ptr<uint8_t>()[n], weight.row_symbol_byte_offsets.data_ptr<int32_t>()[n], n, k);
    });
    return output;
}

static mfq_tensor_backend::Tensor dequant_fp8_sq(
        const Fp8SqWeight & weight,
        bool fp32) {
    if (!weight.blob.is_cuda()) return dequant_sq_cpu(weight).to(
        fp32 ? mfq_tensor_backend::kFloat32 : mfq_tensor_backend::kFloat16);
    if (weight.dtype == "MXFP8-SQ") {
        return mxfp8_sq_dequant_cuda(
            weight.blob, weight.row_q, weight.row_symbol_byte_offsets,
            weight.out, weight.neuron_len,
            weight.block_rows, weight.block_columns,
            weight.scale_rows, weight.scale_columns,
            weight.palettes_offset, weight.symbols_offset,
            weight.scales_offset, fp32);
    }
    if (weight.dtype == "FP8-128SQ") {
        return fp8_128_sq_dequant_cuda(
            weight.blob, weight.row_q, weight.row_symbol_byte_offsets,
            weight.out, weight.neuron_len, weight.scale_kind,
            weight.palettes_offset, weight.symbols_offset,
            weight.scales_offset, fp32);
    }
    throw std::runtime_error("unsupported FP8-SQ dequant dtype");
}

static mfq_tensor_backend::Tensor dequant_nint_dense_f32(const NintWeight & w) {
    mfq_tensor_backend::Tensor dense;
    if (w.q8_zero) {
        dense = nint8_zero_dequant_cuda(
            w.q_packed, w.q8_zero_scale, w.neuron_len);
    } else {
        dense = nint_decode_cuda(
            w.q_packed, w.row_q_bits, w.row_q_bit_offsets,
            w.sub_scale, w.sub_min, w.neuron_scale, w.neuron_min,
            w.neuron_len, w.gs);
    }
    return dense.to(mfq_tensor_backend::kFloat32).contiguous();
}

static mfq_tensor_backend::Tensor dequant_quant_linear_f32(const QuantLinear & linear) {
    if (!linear.tensor_parallel()) {
        if (linear.is_nint()) {
            return dequant_nint_dense_f32(linear.nint.w);
        }
        if (linear.is_nvq()) {
            return nvq_dequant(linear.nvq.w)
                .to(mfq_tensor_backend::kFloat32).contiguous();
        }
        if (linear.is_mxfp8()) {
            return mxfp8_dequant_cuda(
                linear.mxfp8.weight.values,
                linear.mxfp8.weight.scales)
                .to(mfq_tensor_backend::kFloat32).contiguous();
        }
        if (linear.is_mxfp4()) {
            return mxfp4_dequant_cuda(
                linear.mxfp4.weight.values,
                linear.mxfp4.weight.scales)
                .to(mfq_tensor_backend::kFloat32).contiguous();
        }
        if (linear.is_mxfp4_sq()) {
            const auto & weight = linear.mxfp4_sq.weight;
            if (!weight.blob.is_cuda()) return dequant_sq_cpu(weight);
            return mxfp4_sq_dequant_cuda(
                weight.blob,
                weight.row_q,
                weight.row_symbol_byte_offsets,
                weight.row_auxiliary,
                weight.bits,
                weight.out,
                weight.neuron_len,
                weight.matrix_scale_base,
                weight.q_sum,
                weight.sq4_rows,
                true).contiguous();
        }
        if (linear.is_fp8_sq()) {
            return dequant_fp8_sq(linear.fp8_sq.weight, true).contiguous();
        }
        if (linear.is_tpq()) {
            const auto & weight = linear.tpq.weight;
            auto dense = weight.int4
                ? tpq_int4_dequant_cuda(
                    weight.packed, weight.scales, weight.group_size)
                : tpq_pq_dequant_cuda(
                    weight.packed, weight.codebook,
                    weight.out, weight.neuron_len,
                    weight.vector_size, weight.index_bits);
            return dense.to(mfq_tensor_backend::kFloat32).contiguous();
        }
        if (linear.is_dense()) {
            return linear.dense.to(mfq_tensor_backend::kFloat32).contiguous();
        }
        throw std::runtime_error(
            "unsupported linear kind for FP32 reconstruction");
    }
    if (linear.tensor_parallel_axis != TensorParallelAxis::Output &&
        linear.tensor_parallel_axis != TensorParallelAxis::Input) {
        throw std::runtime_error(
            "cannot reconstruct a mirrored tensor-parallel linear");
    }
    const int primary = model_parallel_primary_device();
    std::vector<mfq_tensor_backend::Tensor> parts;
    parts.reserve(linear.tensor_parallel_shards.size());
    for (const auto & shard : linear.tensor_parallel_shards) {
        MfqCudaGuard shard_guard(shard.device);
        mfq_tensor_backend::Tensor part;
        if (shard.kind == QuantLinearKind::Nint) {
            part = dequant_nint_dense_f32(shard.nint);
        } else if (shard.kind == QuantLinearKind::Nvq) {
            part = nvq_dequant(shard.nvq)
                .to(mfq_tensor_backend::kFloat32).contiguous();
        } else if (shard.kind == QuantLinearKind::Mxfp8) {
            part = mxfp8_dequant_cuda(
                shard.mxfp8.values,
                shard.mxfp8.scales)
                .to(mfq_tensor_backend::kFloat32).contiguous();
        } else if (shard.kind == QuantLinearKind::Mxfp4) {
            part = mxfp4_dequant_cuda(
                shard.mxfp4.values, shard.mxfp4.scales)
                .to(mfq_tensor_backend::kFloat32).contiguous();
        } else if (shard.kind == QuantLinearKind::Mxfp4Sq || shard.kind == QuantLinearKind::Fp8Sq) {
            QuantLinear local;
            local.kind = shard.kind;
            local.mxfp4_sq = shard.mxfp4_sq;
            local.fp8_sq = shard.fp8_sq;
            part = dequant_quant_linear_f32(local);
        } else if (shard.kind == QuantLinearKind::Tpq) {
            part = shard.tpq.int4
                ? tpq_int4_dequant_cuda(
                    shard.tpq.packed, shard.tpq.scales,
                    shard.tpq.group_size)
                : tpq_pq_dequant_cuda(
                    shard.tpq.packed, shard.tpq.codebook,
                    shard.tpq.out, shard.tpq.neuron_len,
                    shard.tpq.vector_size, shard.tpq.index_bits);
            part = part.to(mfq_tensor_backend::kFloat32).contiguous();
        } else if (shard.kind == QuantLinearKind::Dense) {
            part = shard.dense.to(mfq_tensor_backend::kFloat32).contiguous();
        } else {
            throw std::runtime_error(
                "unsupported tensor-parallel shard kind for reconstruction");
        }
        parts.push_back(
            tensor_to_cuda_device(part, primary)
                .to(mfq_tensor_backend::kFloat32).contiguous());
    }
    MfqCudaGuard primary_guard(primary);
    auto dense = mfq_tensor_backend::cat(
        parts,
        linear.tensor_parallel_axis == TensorParallelAxis::Output
            ? 0 : 1).contiguous();
    if (dense.size(0) != linear.out() ||
        dense.size(1) != linear.neuron_len()) {
        throw std::runtime_error(
            "reconstructed tensor-parallel linear shape mismatch");
    }
    return dense;
}

DenseLinearGroup make_fp32_quant_group(QuantLinearGroup group) {
    DenseLinearGroup dense;
    dense.outs = group.outs;
    if (group.nint_grouped) {
        if (group.nint.split_w.empty()) {
            dense.w = dequant_nint_dense_f32(group.nint.w);
        } else {
            std::vector<mfq_tensor_backend::Tensor> parts;
            parts.reserve(group.nint.split_w.size());
            for (const auto & weight : group.nint.split_w) {
                parts.push_back(dequant_nint_dense_f32(weight));
            }
            dense.w = mfq_tensor_backend::cat(parts, 0).contiguous();
        }
    } else {
        std::vector<mfq_tensor_backend::Tensor> parts;
        parts.reserve(group.layers.size());
        for (const auto & linear : group.layers) {
            parts.push_back(dequant_quant_linear_f32(linear));
        }
        dense.w = mfq_tensor_backend::cat(parts, 0).contiguous();
    }
    MFQ_RUNTIME_CHECK(
        dense.w.dim() == 2 &&
            dense.w.size(0) ==
                std::accumulate(
                    dense.outs.begin(), dense.outs.end(), int64_t{0}),
        "FP32 compressor projection shape mismatch");
    return dense;
}

bool nvq_fused_residual_format(std::int64_t kernel_format) {
    return kernel_format == kNvq2JscXlGroupExecKernelFormat ||
        kernel_format == kNvq3JscLGroupExecKernelFormat;
}

mfq_tensor_backend::Tensor quant_linear_reference_weight(
        const QuantLinear& linear) {
    if (linear.is_nint()) {
        const auto& weight = linear.nint.w;
        return weight.q8_zero
            ? nint8_zero_dequant_cuda(
                  weight.q_packed, weight.q8_zero_scale,
                  weight.neuron_len)
            : nint_decode_cuda(
                  weight.q_packed, weight.row_q_bits,
                  weight.row_q_bit_offsets, weight.sub_scale,
                  weight.sub_min, weight.neuron_scale,
                  weight.neuron_min, weight.neuron_len, weight.gs);
    }
    if (linear.is_nvq()) return nvq_dequant(linear.nvq.w);
    if (linear.is_mxfp4()) {
        return mxfp4_dequant_cuda(
            linear.mxfp4.weight.values, linear.mxfp4.weight.scales);
    }
    if (linear.is_mxfp4_sq()) {
        const auto& weight = linear.mxfp4_sq.weight;
        if (!weight.blob.is_cuda()) return dequant_sq_cpu(weight).to(mfq_tensor_backend::kFloat16);
        return mxfp4_sq_dequant_cuda(
            weight.blob, weight.row_q, weight.row_symbol_byte_offsets,
            weight.row_auxiliary, weight.bits, weight.out,
            weight.neuron_len, weight.matrix_scale_base,
            weight.q_sum, weight.sq4_rows, false);
    }
    if (linear.is_fp8_sq()) {
        return dequant_fp8_sq(linear.fp8_sq.weight, false);
    }
    if (linear.is_tpq()) {
        const auto& weight = linear.tpq.weight;
        return weight.int4
            ? tpq_int4_dequant_cuda(
                  weight.packed, weight.scales, weight.group_size)
            : tpq_pq_dequant_cuda(
                  weight.packed, weight.codebook, weight.out,
                  weight.neuron_len, weight.vector_size,
                  weight.index_bits);
    }
    if (linear.is_mxfp8()) {
        return mxfp8_cpu_reference(linear.mxfp8.weight);
    }
    if (linear.is_dense()) return linear.dense;
    throw std::runtime_error("unsupported linear reference format");
}

mfq_tensor_backend::Tensor mfe_dense_reference(
        const mfq::ModelSource& source,
        const std::string& name,
        mfq_tensor_backend::Tensor input,
        const std::vector<std::int32_t>& expert_ids,
        int tokens,
        int routes,
        bool routed_input) {
    auto cpu = unpack_mfe(read_tensor(source, name));
    auto reference = mfq_tensor_backend::empty(
        {tokens * routes, cpu.out_per_expert},
        input.options().dtype(mfq_tensor_backend::kFloat16));
    for (const auto& pool : cpu.pools) {
        mfq_tensor_backend::Tensor dense_flat;
        int rotation_block = 0;
        mfq_tensor_backend::Tensor rotation_signs;
        if (pool.dtype == "NINT8-0") {
            auto packed = to_gpu_nint8_zero(pool.q8_zero);
            dense_flat = nint8_zero_dequant_cuda(
                packed.q_packed, packed.q8_zero_scale,
                packed.neuron_len);
        } else if (pool.dtype == "NINT") {
            auto packed = to_gpu_nint(pool.weight);
            dense_flat = nint_decode_cuda(
                packed.q_packed, packed.row_q_bits,
                packed.row_q_bit_offsets, packed.sub_scale,
                packed.sub_min, packed.neuron_scale,
                packed.neuron_min, packed.neuron_len, packed.gs);
        } else if (pool.dtype == "MXFP4") {
            dense_flat = dequant_mxfp4_cpu(pool.mxfp4)
                .to(mfq_tensor_backend::kCUDA).contiguous();
        } else if (is_tpq_pq_dtype(pool.dtype)) {
            auto packed = to_device_tpq(pool.tpq, true);
            dense_flat = tpq_pq_dequant_cuda(
                packed.packed, packed.codebook,
                packed.out, packed.neuron_len,
                packed.vector_size, packed.index_bits);
        } else if (pool.dtype == "NEPQ") {
            auto packed = to_gpu_nepq(unpack_nepq(
                pool.payload, pool.dtype, pool.runtime_payload));
            dense_flat = nepq_dequant_cuda(
                packed.indices_packed, packed.aux_packed,
                packed.state_packed, packed.neuron_scale,
                packed.table_pool, packed.bank_ids,
                packed.neuron_len, packed.state_bits,
                packed.format);
            if (packed.residual) {
                dense_flat = nepq_sparse_residual_dequant_cuda(
                    packed.residual_codebook,
                    packed.residual_first,
                    packed.residual_second,
                    packed.residual_position_bits,
                    packed.residual_block_vectors,
                    dense_flat.reshape({
                        packed.n_experts * packed.out_per_expert,
                        packed.neuron_len}));
            }
            rotation_block = packed.rotation_block;
            rotation_signs = packed.rotation_signs;
        } else {
            dense_flat = nvq_dequant(to_gpu_nvq(
                unpack_nvq(pool.payload, pool.dtype)));
        }
        auto dense = dense_flat.reshape({
            static_cast<int64_t>(pool.expert_ids.size()),
            cpu.out_per_expert, cpu.neuron_len});
        for (size_t local = 0; local < pool.expert_ids.size(); ++local) {
            const int expert = pool.expert_ids[local];
            std::vector<int64_t> pair_indices;
            std::vector<int64_t> token_indices;
            for (int pair = 0; pair < tokens * routes; ++pair) {
                if (expert_ids[static_cast<size_t>(pair)] == expert) {
                    pair_indices.push_back(pair);
                    token_indices.push_back(pair / routes);
                }
            }
            if (pair_indices.empty()) continue;
            auto pair_index = mfq_tensor_backend::from_blob(
                pair_indices.data(),
                {static_cast<int64_t>(pair_indices.size())},
                mfq_tensor_backend::TensorOptions()
                    .dtype(mfq_tensor_backend::kInt64))
                .clone().to(mfq_tensor_backend::kCUDA);
            auto token_index = mfq_tensor_backend::from_blob(
                token_indices.data(),
                {static_cast<int64_t>(token_indices.size())},
                mfq_tensor_backend::TensorOptions()
                    .dtype(mfq_tensor_backend::kInt64))
                .clone().to(mfq_tensor_backend::kCUDA);
            auto selected = routed_input
                ? input.reshape({tokens * routes, cpu.neuron_len})
                      .index_select(0, pair_index)
                : input.index_select(0, token_index);
            if (rotation_block != 0) {
                selected = nepq_hadamard_input_cuda(
                    selected.contiguous(), rotation_signs,
                    rotation_block);
            }
            auto expected = mfq_tensor_backend::matmul(
                selected,
                dense.index({static_cast<int64_t>(local)})
                    .transpose(0, 1));
            reference.index_copy_(0, pair_index, expected);
        }
    }
    return reference;
}

mfq_tensor_backend::Tensor materialize_mfe_dense(
        const mfq::ModelSource& source,
        const std::string& name) {
    auto cpu = unpack_mfe(read_tensor(source, name));
    auto dense = mfq_tensor_backend::empty(
        {cpu.n_experts, cpu.out_per_expert, cpu.neuron_len},
        mfq_tensor_backend::TensorOptions()
            .device(mfq_tensor_backend::kCUDA)
            .dtype(mfq_tensor_backend::kFloat16));
    for (const auto& pool : cpu.pools) {
        mfq_tensor_backend::Tensor local_flat;
        if (pool.dtype == "NINT8-0") {
            auto packed = to_gpu_nint8_zero(pool.q8_zero);
            local_flat = nint8_zero_dequant_cuda(
                packed.q_packed, packed.q8_zero_scale,
                packed.neuron_len);
        } else if (pool.dtype == "NINT") {
            auto packed = to_gpu_nint(pool.weight);
            local_flat = nint_decode_cuda(
                packed.q_packed, packed.row_q_bits,
                packed.row_q_bit_offsets, packed.sub_scale,
                packed.sub_min, packed.neuron_scale,
                packed.neuron_min, packed.neuron_len, packed.gs);
        } else if (pool.dtype == "MXFP4") {
            local_flat = dequant_mxfp4_cpu(pool.mxfp4)
                .to(mfq_tensor_backend::kCUDA).contiguous();
        } else {
            throw std::runtime_error(
                "dense MoE reference requires NINT or MXFP4 cohorts");
        }
        auto local = local_flat.reshape({
            static_cast<int64_t>(pool.expert_ids.size()),
            cpu.out_per_expert, cpu.neuron_len});
        auto expert_index = mfq_tensor_backend::from_blob(
            const_cast<int32_t*>(pool.expert_ids.data()),
            {static_cast<int64_t>(pool.expert_ids.size())},
            mfq_tensor_backend::TensorOptions()
                .dtype(mfq_tensor_backend::kInt32))
            .clone().to(mfq_tensor_backend::kCUDA)
            .to(mfq_tensor_backend::kInt64);
        dense.index_copy_(0, expert_index, local);
    }
    return dense;
}

std::shared_ptr<MoeExpertCache> make_moe_expert_cache(std::int64_t bytes) {
    return std::make_shared<MoeExpertCache>(bytes);
}

bool moe_expert_cache_has_sources() {
    return g_moe_expert_cache && g_moe_expert_cache->has_sources();
}

bool moe_expert_cache_finalized() {
    return g_moe_expert_cache && g_moe_expert_cache->finalized();
}

void finalize_moe_expert_cache() {
    if (g_moe_expert_cache && !g_moe_expert_cache->finalized()) {
        g_moe_expert_cache->finalize();
    }
}

void print_moe_expert_cache_stats(std::ostream& output) {
    if (g_moe_expert_cache) {
        g_moe_expert_cache->print_stats(output);
    }
}

void set_moe_expert_cache_profile(mfq::MoeCacheProfile profile) {
    if (!g_moe_expert_cache) {
        throw std::runtime_error("MoE expert cache is not configured");
    }
    g_moe_expert_cache->set_profile(std::move(profile));
}
