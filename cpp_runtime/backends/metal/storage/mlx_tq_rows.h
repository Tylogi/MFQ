#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>
#include <mlx/mlx.h>

namespace mfq::metal {

class MlxTqRowBatch {
public:
    std::size_t rows() const noexcept { return descriptors_.size(); }
    std::size_t packed_nbytes() const noexcept;
    std::size_t source_bytes_read() const noexcept { return source_bytes_; }
    void append_batch(const MlxTqRowBatch& other);
    mlx::core::array decode() const;

private:
    friend class MlxMappedTqRows;
    int width_ = 0;
    std::size_t source_bytes_ = 0;
    std::vector<std::uint64_t> descriptors_;
    std::vector<std::uint32_t> layouts_, packed_;
    std::vector<float> scales_;
};

class MlxMappedTqRows {
public:
    struct PackedRows { std::size_t offset, bytes; };
    explicit MlxMappedTqRows(std::span<const std::uint8_t> blob, int width = 0);
    std::int64_t rows() const noexcept { return rows_; }
    int width() const noexcept { return width_; }
    std::size_t selected_row_nbytes_bound() const noexcept;
    PackedRows packed_rows(std::int64_t first, std::size_t count) const;
    void write_descriptors(std::int64_t first, std::uint32_t word_base,
        std::span<std::uint64_t> destination) const;
    void append_row(std::int64_t row, MlxTqRowBatch& batch) const;

private:
    std::uint64_t descriptor(std::int64_t row) const;
    std::span<const std::uint8_t> blob_;
    int rows_, width_, stored_width_, group_size_;
    bool columns_;
    std::size_t groups_, scales_offset_, packed_offset_;
};

class MlxMappedTqTable {
public:
    explicit MlxMappedTqTable(std::vector<MlxMappedTqRows> shards);
    std::int64_t rows() const noexcept { return rows_; }
    int width() const noexcept { return shards_.front().width(); }
    std::size_t selected_row_nbytes_bound() const noexcept { return row_bound_; }
    void append_row(std::int64_t row, MlxTqRowBatch& batch) const;

private:
    std::vector<MlxMappedTqRows> shards_;
    std::vector<std::int64_t> ends_;
    std::int64_t rows_ = 0;
    std::size_t row_bound_ = 0;
};

} // namespace mfq::metal
