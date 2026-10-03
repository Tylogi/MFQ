#pragma once

#include "mfq_container.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>

#include <mlx/mlx.h>

namespace mfq::metal {

struct MlxMxfp8RowStoreStats {
    std::uint64_t row_requests = 0;
    std::uint64_t cache_hits = 0;
    std::uint64_t cache_misses = 0;
    std::uint64_t rows_loaded = 0;
    std::uint64_t bytes_read = 0;
    std::uint64_t read_calls = 0;
    double io_seconds = 0.0;
    std::size_t resident_rows = 0;
    std::size_t resident_payload_bytes = 0;
    std::size_t cache_limit_bytes = 0;

    double hit_rate() const noexcept {
        return row_requests == 0
            ? 0.0
            : static_cast<double>(cache_hits) /
                static_cast<double>(row_requests);
    }
};

// Architecture-neutral bounded row cache for row-scaled native MXFP8 tables.
// Cold rows are read through MfqContainer, so the same implementation serves
// regular MFQ artifacts and canonical virtual records over untouched HF
// checkpoints. Persistent workers expose SSD queue depth without holding the
// LRU lock across I/O.
class MlxMxfp8RowStore {
public:
    MlxMxfp8RowStore(
        const MfqContainer& model,
        std::string record,
        std::int64_t expected_rows,
        int expected_width,
        std::size_t cache_rows,
        std::size_t io_workers = 8);
    ~MlxMxfp8RowStore();

    MlxMxfp8RowStore(const MlxMxfp8RowStore&) = delete;
    MlxMxfp8RowStore& operator=(const MlxMxfp8RowStore&) = delete;

    // Returns decoded F16 rows in request order with shape [N,width].
    mlx::core::array gather(std::span<const std::int64_t> row_ids) const;

    std::int64_t rows() const noexcept;
    int width() const noexcept;
    std::size_t cached_rows() const noexcept;
    MlxMxfp8RowStoreStats stats() const;
    void clear();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mfq::metal
