#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

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
    void validate() const;
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
    NintRows(const std::uint8_t* data, std::size_t size);
    NintRows(std::size_t size, Read read);
    NintRows(const NintRows&) = delete;
    NintRows& operator=(const NintRows&) = delete;
    NintRows(NintRows&&) noexcept = default;
    NintRows& operator=(NintRows&&) noexcept = default;
    int rows() const noexcept { return rows_; }
    int width() const noexcept { return width_; }
    std::size_t index_nbytes() const noexcept;
    void append_row(std::int64_t row, NintRowBatch& batch) const;
private:
    void initialize();
    const std::uint8_t* selectors(std::size_t offset, std::size_t size,
        std::vector<std::uint8_t>& owned);
    Read read_;
    const std::uint8_t* borrowed_ = nullptr;
    std::size_t nbytes_ = 0;
    const std::uint8_t* k_selectors_ = nullptr;
    const std::uint8_t* q_selectors_ = nullptr;
    std::vector<std::uint8_t> owned_k_, owned_q_;
    std::vector<std::array<std::uint32_t, 4>> k_ranks_;
    std::vector<std::array<std::uint32_t, 8>> q_ranks_;
    std::array<std::size_t, 4> scale_offsets_{};
    std::array<std::size_t, 4> min_offsets_{};
    std::array<std::size_t, 8> q_offsets_{};
    std::size_t neuron_scale_offset_ = 0, neuron_min_offset_ = 0;
    int rows_ = 0, width_ = 0, groups_ = 0, group_size_ = 0;
    int bits_ = 0, sub_bits_ = 0;
    bool adaptive_ = false;
};
} // namespace mfq
