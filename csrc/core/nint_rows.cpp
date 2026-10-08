#include "mfq/nint_rows.h"

#include "mfq/nint_blob.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace mfq {
namespace {

constexpr std::size_t kRankStride = 256;

std::size_t packed_size(std::uint64_t count, int bits) {
    if (count > (std::numeric_limits<std::uint64_t>::max() - 7) / bits) {
        throw std::overflow_error("NINT row stream size overflow");
    }
    return static_cast<std::size_t>((count * bits + 7) / 8);
}

class Cursor {
public:
    Cursor(std::size_t size, const NintRows::Read& read) : size_(size), read_(read) {}
    std::size_t skip(std::size_t count) {
        if (offset_ > size_ || count > size_ - offset_) {
            throw std::runtime_error("truncated mapped NINT row tensor");
        }
        auto start = offset_;
        offset_ += count;
        return start;
    }
    template <typename T> T scalar() {
        const auto start = skip(sizeof(T));
        T value;
        read_(start, reinterpret_cast<std::uint8_t*>(&value), sizeof(T));
        return value;
    }
    bool finished() const noexcept { return offset_ == size_; }
private:
    std::size_t size_;
    const NintRows::Read& read_;
    std::size_t offset_ = 0;
};

std::uint32_t selector(
    const std::uint8_t* stream, std::size_t row, int bits) {
    const auto bit = row * static_cast<std::size_t>(bits);
    const auto byte = bit / 8;
    std::uint32_t value = stream[byte];
    if ((bit & 7) + bits > 8) value |= std::uint32_t(stream[byte + 1]) << 8;
    return (value >> (bit & 7)) & ((1u << bits) - 1);
}

template <std::size_t N>
std::array<std::uint32_t, N> build_ranks(
    const std::uint8_t* selectors, int rows, int bits,
    std::vector<std::array<std::uint32_t, N>>& checkpoints) {
    std::array<std::uint32_t, N> counts{};
    checkpoints.reserve((static_cast<std::size_t>(rows) + kRankStride - 1) / kRankStride);
    for (int row = 0; row < rows; ++row) {
        if (row % kRankStride == 0) checkpoints.push_back(counts);
        ++counts[selector(selectors, row, bits)];
    }
    return counts;
}

template <std::size_t N>
std::uint32_t cohort_rank(
    const std::uint8_t* selectors, std::size_t row, int bits,
    std::uint32_t cohort,
    const std::vector<std::array<std::uint32_t, N>>& checkpoints) {
    auto rank = checkpoints[row / kRankStride][cohort];
    for (auto index = row - row % kRankStride; index < row; ++index) {
        rank += selector(selectors, index, bits) == cohort;
    }
    return rank;
}

} // namespace

NintRows::NintRows(const std::uint8_t* data, std::size_t size)
    : read_([data](std::size_t offset, std::uint8_t* out, std::size_t count) {
        if (count) std::memcpy(out, data + offset, count);
    }), borrowed_(data), nbytes_(size) {
    if (!data && size) throw std::invalid_argument("null NINT row storage");
    initialize();
}

NintRows::NintRows(std::size_t size, Read read)
    : read_(std::move(read)), nbytes_(size) {
    if (!read_) throw std::invalid_argument("missing NINT row range reader");
    initialize();
}

const std::uint8_t* NintRows::selectors(
    std::size_t offset, std::size_t size, std::vector<std::uint8_t>& owned) {
    if (borrowed_) return borrowed_ + offset;
    owned.resize(size);
    read_(offset, owned.data(), size);
    return owned.data();
}

void NintRows::initialize() {
    Cursor cursor(nbytes_, read_);
    const auto raw_bits = cursor.scalar<std::uint8_t>();
    adaptive_ = mfq::nint_has_adaptive_storage(raw_bits);
    bits_ = mfq::nint_logical_bits(raw_bits);
    sub_bits_ = cursor.scalar<std::uint8_t>();
    group_size_ = cursor.scalar<std::int32_t>();
    const auto axis = cursor.scalar<std::int32_t>();
    width_ = cursor.scalar<std::int32_t>();
    const auto dimensions = cursor.scalar<std::uint32_t>();
    if (bits_ < 1 || bits_ > 8 || sub_bits_ < 1 || sub_bits_ > 8 ||
        group_size_ <= 0 || axis != 0 || width_ <= 0 || dimensions != 2) {
        throw std::runtime_error("invalid mapped NINT row geometry");
    }
    const auto shape_rows = cursor.scalar<std::int64_t>();
    const auto shape_width = cursor.scalar<std::int64_t>();
    const auto rows = cursor.scalar<std::uint32_t>();
    const auto groups = cursor.scalar<std::uint32_t>();
    if (rows == 0 || rows > std::numeric_limits<int>::max() ||
        shape_rows != rows || shape_width != width_ ||
        groups != (static_cast<std::uint64_t>(width_) + group_size_ - 1) / group_size_) {
        throw std::runtime_error("inconsistent mapped NINT row shape");
    }
    rows_ = static_cast<int>(rows);
    groups_ = static_cast<int>(groups);
    neuron_scale_offset_ = cursor.skip(static_cast<std::size_t>(rows) * 2);
    neuron_min_offset_ = cursor.skip(static_cast<std::size_t>(rows) * 2);
    const auto values_per_row = static_cast<std::uint64_t>(groups) * group_size_;
    if (adaptive_) {
        const auto size = packed_size(rows, 2);
        k_selectors_ = selectors(cursor.skip(size), size, owned_k_);
        const auto counts = build_ranks(k_selectors_, rows_, 2, k_ranks_);
        for (int cohort = 0; cohort < 4; ++cohort) {
            const int width = sub_bits_ - 1 + cohort;
            if (counts[cohort] && (width < 1 || width > 8)) {
                throw std::runtime_error("invalid mapped NINT k selector");
            }
            const auto bytes = counts[cohort]
                ? packed_size(std::uint64_t(counts[cohort]) * groups, width) : 0;
            scale_offsets_[cohort] = cursor.skip(bytes);
            min_offsets_[cohort] = cursor.skip(bytes);
        }
        const auto qsize = packed_size(rows, 3);
        q_selectors_ = selectors(cursor.skip(qsize), qsize, owned_q_);
        const auto qcounts = build_ranks(q_selectors_, rows_, 3, q_ranks_);
        for (int cohort = 0; cohort < 8; ++cohort) {
            q_offsets_[cohort] = cursor.skip(packed_size(
                std::uint64_t(qcounts[cohort]) * values_per_row, cohort + 1));
        }
    } else {
        const auto metadata_bytes = packed_size(std::uint64_t(rows) * groups, sub_bits_);
        scale_offsets_[0] = cursor.skip(metadata_bytes);
        min_offsets_[0] = cursor.skip(metadata_bytes);
        q_offsets_[0] = cursor.skip(packed_size(std::uint64_t(rows) * values_per_row, bits_));
    }
    if (!cursor.finished()) throw std::runtime_error("trailing mapped NINT row bytes");
}

std::size_t NintRows::index_nbytes() const noexcept {
    return k_ranks_.size() * sizeof(k_ranks_[0]) + q_ranks_.size() * sizeof(q_ranks_[0]) +
        owned_k_.size() + owned_q_.size();
}

void NintRows::append_row(std::int64_t row, NintRowBatch& batch) const {
    if (row < 0 || row >= rows_) throw std::out_of_range("mapped NINT row ID");
    if (batch.width_ && batch.width_ != width_) {
        throw std::runtime_error("NINT row batch widths disagree");
    }
    const auto local = static_cast<std::size_t>(row);
    const auto kc = adaptive_ ? selector(k_selectors_, local, 2) : 0;
    const auto qc = adaptive_ ? selector(q_selectors_, local, 3) : 0;
    const int k = adaptive_ ? sub_bits_ - 1 + static_cast<int>(kc) : sub_bits_;
    const int q = adaptive_ ? static_cast<int>(qc) + 1 : bits_;
    const auto kr = adaptive_ ? cohort_rank(k_selectors_, local, 2, kc, k_ranks_) : local;
    const auto qr = adaptive_ ? cohort_rank(q_selectors_, local, 3, qc, q_ranks_) : local;
    const auto kbit = std::uint64_t(kr) * groups_ * k;
    const auto qbit = std::uint64_t(qr) * groups_ * group_size_ * q;
    std::uint16_t scale, minimum;
    read_(neuron_scale_offset_ + local * 2, reinterpret_cast<std::uint8_t*>(&scale), 2);
    read_(neuron_min_offset_ + local * 2, reinterpret_cast<std::uint8_t*>(&minimum), 2);
    if ((scale & 0x7c00u) == 0x7c00u || (minimum & 0x7c00u) == 0x7c00u) {
        throw std::runtime_error("NINT row anchors must be finite");
    }
    const auto copy_stream = [&](std::size_t offset, std::uint64_t bit, std::uint64_t count, int bits) {
        const auto start = offset + static_cast<std::size_t>(bit / 8);
        const auto exact_bytes = static_cast<std::size_t>(((bit & 7) + count * bits + 7) / 8);
        if (start > nbytes_ || exact_bytes > nbytes_ - start ||
            exact_bytes >= std::numeric_limits<int>::max() ||
            batch.packed_.size() > std::numeric_limits<int>::max() - exact_bytes - 1) {
            throw std::runtime_error("NINT selected row stream exceeds bounds");
        }
        const auto destination = static_cast<std::uint32_t>(batch.packed_.size());
        batch.packed_.resize(batch.packed_.size() + exact_bytes);
        read_(start, batch.packed_.data() + destination, exact_bytes);
        batch.packed_.push_back(0); // The kernel may safely read a two-byte window.
        batch.source_bytes_read_ += exact_bytes;
        return destination;
    };
    const auto qoffset = copy_stream(q_offsets_[qc], qbit, std::uint64_t(groups_) * group_size_, q);
    const auto soffset = copy_stream(scale_offsets_[kc], kbit, groups_, k);
    const auto moffset = copy_stream(min_offsets_[kc], kbit, groups_, k);
    const std::uint32_t layout = q | (k << 4) | ((qbit & 7) << 8) |
        ((kbit & 7) << 12) | ((kbit & 7) << 16);
    batch.descriptors_.insert(batch.descriptors_.end(), {
        qoffset, soffset, moffset, layout, static_cast<std::uint32_t>(group_size_),
        std::uint32_t(scale) | (std::uint32_t(minimum) << 16)});
    batch.source_bytes_read_ += 4;
    batch.width_ = width_;
}

void NintRowBatch::append_batch(const NintRowBatch& other) {
    other.validate();
    if (this == &other || (width_ && width_ != other.width_)) {
        throw std::invalid_argument("NINT row batch merge disagrees");
    }
    if (other.packed_.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        packed_.size() > std::numeric_limits<int>::max() - other.packed_.size() ||
        rows() + other.rows() > std::numeric_limits<int>::max() / 6 ||
        (rows() + other.rows()) * static_cast<std::uint64_t>(other.width_) >
            std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("NINT merged row batch exceeds bounds");
    }
    const auto offset = static_cast<std::uint32_t>(packed_.size());
    packed_.insert(packed_.end(), other.packed_.begin(), other.packed_.end());
    for (std::size_t row = 0; row < other.rows(); ++row) {
        const auto* descriptor = other.descriptors_.data() + row * 6;
        descriptors_.insert(descriptors_.end(), {
            descriptor[0] + offset, descriptor[1] + offset, descriptor[2] + offset,
            descriptor[3], descriptor[4], descriptor[5]});
    }
    source_bytes_read_ += other.source_bytes_read_;
    width_ = other.width_;
}

void NintRowBatch::validate() const {
    if (rows() == 0 || rows() > std::numeric_limits<int>::max() / 6 ||
        rows() * static_cast<std::uint64_t>(width_) > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("invalid NINT row decode batch size");
    }
}
} // namespace mfq
