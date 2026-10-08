#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>
#include "mfq/read_span.h"

namespace mfq {

// Portable selected-row wire protocol shared by Metal and CUDA. Six uint32
// words per row: q/scale/min byte offsets, q/k widths and bit shifts, group
// size, and two raw FP16 neuron anchors. Each copied stream has a guard byte.
class NintRowBatch {
public:
    std::size_t rows() const noexcept { return descriptors_.size() / 6; }
    int width() const noexcept { return width_; }
    std::size_t source_bytes_read() const noexcept { return source_bytes_read_; }
    std::size_t packed_nbytes() const noexcept { return packed_.size(); }
    const std::vector<std::uint8_t>& packed() const noexcept { return packed_; }
    const std::vector<std::uint32_t>& descriptors() const noexcept { return descriptors_; }
    void append_batch(const NintRowBatch& other);
    void validate() const;
    void copy_row(std::size_t row,NintRowBatch&) const;
private:
    friend class NintRows;
    std::vector<std::uint8_t> packed_;
    std::vector<std::uint32_t> descriptors_;
    std::size_t source_bytes_read_ = 0;
    int width_ = 0;
};

// Memory mode borrows immutable storage. Range mode owns its callback and only
// retains selectors/rank checkpoints, never the complete q/k payload.
class NintRows {
public:
    using Read = std::function<void(std::size_t, std::uint8_t*, std::size_t)>;
    using ReadBatch=std::function<void(const std::vector<ReadSpan>&)>;
    NintRows(const std::uint8_t* data, std::size_t size);
    NintRows(std::size_t size, Read read, bool parallel_reads = false,ReadBatch batch = {});
    NintRows(const NintRows&) = delete;
    NintRows& operator=(const NintRows&) = delete;
    NintRows(NintRows&&) noexcept = default;
    NintRows& operator=(NintRows&&) noexcept = default;
    int rows() const noexcept { return rows_; }
    int width() const noexcept { return width_; }
    std::size_t index_nbytes() const noexcept;
    std::size_t selected_row_nbytes_bound() const noexcept {
        return std::size_t(groups_) * group_size_ + std::size_t(groups_) * 2 + 6;
    }
    void append_row(std::int64_t row, NintRowBatch& batch) const;
    // Preserve input order and the selected-row wire protocol while issuing
    // independent reads on persistent host workers. For threads > 1 the Read
    // callback must opt into concurrent positional reads at construction.
    // Otherwise the calling thread performs every read. Payload stays local
    // to this batch; no table or filesystem cache is introduced.
    void append_rows(const std::int64_t* rows, std::size_t count,
        NintRowBatch& batch, int threads = 1) const;
    // Canonical compact tensor containing [begin,end). Reads each selected
    // q/k cohort as a contiguous range, rather than issuing I/O per neuron.
    // Adaptive selectors and the original quantization profile are preserved.
    std::vector<std::uint8_t> slice_rows_blob(std::int64_t begin,std::int64_t end) const;
    std::size_t row_range_nbytes(std::int64_t begin,std::int64_t end) const;
    std::uint64_t row_values_bits(std::int64_t begin,std::int64_t end) const;
private:
    void initialize();
    const std::uint8_t* selectors(std::size_t offset, std::size_t size,
        std::vector<std::uint8_t>& owned);
    Read read_;
    ReadBatch read_batch_;
    void append_rows_batched(const std::int64_t*,std::size_t,NintRowBatch&) const;
    const std::uint8_t* borrowed_ = nullptr;
    std::size_t nbytes_ = 0;
    const std::uint8_t* k_selectors_ = nullptr;
    const std::uint8_t* q_selectors_ = nullptr;
    std::vector<std::uint8_t> owned_k_, owned_q_;
    std::vector<std::array<std::uint32_t, 4>> k_ranks_;
    std::vector<std::array<std::uint32_t, 8>> q_ranks_;
    std::array<std::uint32_t, 4> k_counts_{};
    std::array<std::uint32_t, 8> q_counts_{};
    std::array<std::size_t, 4> scale_offsets_{};
    std::array<std::size_t, 4> min_offsets_{};
    std::array<std::size_t, 8> q_offsets_{};
    std::size_t neuron_scale_offset_ = 0, neuron_min_offset_ = 0;
    int rows_ = 0, width_ = 0, groups_ = 0, group_size_ = 0;
    int bits_ = 0, sub_bits_ = 0;
    bool adaptive_ = false;
    bool parallel_reads_ = false;
};
} // namespace mfq
