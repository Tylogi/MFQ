#include "mlx_mxfp8_row_store.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <exception>
#include <functional>
#include <future>
#include <limits>
#include <list>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mfq::metal {
namespace {

constexpr std::size_t kMxHeaderBytes = 56;

std::uint64_t read_u64_at(
    std::span<const std::uint8_t> bytes,
    std::size_t offset) {
    if (offset > bytes.size() || 8 > bytes.size() - offset) {
        throw std::runtime_error("truncated native MXFP8 header");
    }
    std::uint64_t value = 0;
    for (int index = 0; index < 8; ++index) {
        value |= static_cast<std::uint64_t>(bytes[offset + index])
            << (8 * index);
    }
    return value;
}

std::uint64_t checked_product(
    std::uint64_t left,
    std::uint64_t right,
    const char* description) {
    if (left != 0 &&
        right > std::numeric_limits<std::uint64_t>::max() / left) {
        throw std::overflow_error(
            std::string("native MXFP8 ") + description + " size overflow");
    }
    return left * right;
}

float decode_e4m3(std::uint8_t raw) {
    if ((raw & 0x7fu) == 0x7fu) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    const float sign = (raw & 0x80u) ? -1.0f : 1.0f;
    const int exponent = (raw >> 3u) & 0x0fu;
    const int mantissa = raw & 0x07u;
    return sign * (exponent == 0
        ? std::ldexp(static_cast<float>(mantissa), -9)
        : std::ldexp(
              1.0f + static_cast<float>(mantissa) / 8.0f,
              exponent - 7));
}

float decode_e8m0(std::uint8_t raw) {
    return raw == 255
        ? std::numeric_limits<float>::quiet_NaN()
        : std::ldexp(1.0f, static_cast<int>(raw) - 127);
}

class WorkerPool {
public:
    explicit WorkerPool(std::size_t count) {
        if (count == 0) {
            throw std::invalid_argument(
                "native MXFP8 row store requires at least one I/O worker");
        }
        workers_.reserve(count);
        for (std::size_t index = 0; index < count; ++index) {
            workers_.emplace_back([this] { run(); });
        }
    }

    ~WorkerPool() {
        {
            std::scoped_lock lock(mutex_);
            stopping_ = true;
        }
        condition_.notify_all();
        for (auto& worker : workers_) {
            if (worker.joinable()) worker.join();
        }
    }

    std::future<void> submit(std::function<void()> function) {
        auto task = std::make_shared<std::packaged_task<void()>>(
            std::move(function));
        auto result = task->get_future();
        {
            std::scoped_lock lock(mutex_);
            if (stopping_) {
                throw std::runtime_error(
                    "native MXFP8 row-store worker pool is stopping");
            }
            tasks_.push_back([task] { (*task)(); });
        }
        condition_.notify_one();
        return result;
    }

private:
    void run() {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock lock(mutex_);
                condition_.wait(lock, [this] {
                    return stopping_ || !tasks_.empty();
                });
                if (stopping_ && tasks_.empty()) return;
                task = std::move(tasks_.front());
                tasks_.pop_front();
            }
            task();
        }
    }

    std::mutex mutex_;
    std::condition_variable condition_;
    std::deque<std::function<void()>> tasks_;
    std::vector<std::thread> workers_;
    bool stopping_ = false;
};

} // namespace

struct MlxMxfp8RowStore::Impl {
    struct CacheEntry {
        std::vector<mlx::core::float16_t> values;
        std::list<std::int64_t>::iterator recency;
    };

    struct RequestedRow {
        std::int64_t row = 0;
        std::vector<mlx::core::float16_t> values;
        bool cold = false;
    };

    Impl(
        const MfqContainer& selected_model,
        std::string selected_record,
        std::int64_t expected_rows,
        int expected_width,
        std::size_t selected_cache_rows,
        std::size_t io_workers)
        : model(selected_model),
          record(std::move(selected_record)),
          cache_capacity(selected_cache_rows),
          workers(io_workers) {
        const auto& entry = model.record(record);
        if (entry.dtype != "MXFP8") {
            throw std::runtime_error(
                "native MXFP8 row store received non-MXFP8 record: " +
                record);
        }
        const auto header = model.read_range(record, 0, kMxHeaderBytes);
        if (header.size() != kMxHeaderBytes ||
            std::memcmp(header.data(), "MXT1", 4) != 0 ||
            header[4] != 1 || header[5] != 8 ||
            header[6] != 0 || header[7] != 0) {
            throw std::runtime_error(
                "invalid native MXFP8 row-store header: " + record);
        }
        const auto logical_rows = read_u64_at(header, 8);
        const auto logical_width = read_u64_at(header, 16);
        const auto storage_rows = read_u64_at(header, 24);
        const auto storage_width = read_u64_at(header, 32);
        const auto scale_rows = read_u64_at(header, 40);
        const auto scale_width = read_u64_at(header, 48);
        if (expected_rows <= 0 || expected_width <= 0 ||
            expected_width % 32 != 0 ||
            logical_rows != static_cast<std::uint64_t>(expected_rows) ||
            logical_width != static_cast<std::uint64_t>(expected_width) ||
            storage_rows != logical_rows || storage_width != logical_width ||
            scale_rows != logical_rows || scale_width != logical_width / 32) {
            throw std::runtime_error(
                "native MXFP8 record is not the expected row-scaled table: " +
                record);
        }
        const auto value_bytes = checked_product(
            logical_rows, logical_width, "value");
        const auto scale_bytes = checked_product(
            scale_rows, scale_width, "scale");
        if (value_bytes > std::numeric_limits<std::uint64_t>::max() -
                kMxHeaderBytes ||
            scale_bytes > std::numeric_limits<std::uint64_t>::max() -
                kMxHeaderBytes - value_bytes ||
            kMxHeaderBytes + value_bytes + scale_bytes != entry.nbytes) {
            throw std::runtime_error(
                "native MXFP8 row-store payload size disagrees: " + record);
        }
        rows = expected_rows;
        width = expected_width;
        scale_columns = expected_width / 32;
        values_offset = kMxHeaderBytes;
        scales_offset = kMxHeaderBytes + value_bytes;
        source_row_bytes = static_cast<std::size_t>(width + scale_columns);
        decoded_row_bytes = static_cast<std::size_t>(width) *
            sizeof(mlx::core::float16_t);
        if (cache_capacity != 0 &&
            decoded_row_bytes > std::numeric_limits<std::size_t>::max() /
                cache_capacity) {
            throw std::overflow_error(
                "native MXFP8 row-store cache size overflow");
        }
        for (std::size_t scale = 0; scale < 256; ++scale) {
            const auto scale_value = decode_e8m0(
                static_cast<std::uint8_t>(scale));
            for (std::size_t value = 0; value < 256; ++value) {
                decode_lut[(scale << 8u) | value] =
                    static_cast<mlx::core::float16_t>(
                        scale_value * decode_e4m3(
                            static_cast<std::uint8_t>(value)));
            }
        }
    }

    void read_decode(
        std::int64_t row,
        std::vector<mlx::core::float16_t>& output) const {
        std::vector<std::uint8_t> raw(source_row_bytes);
        model.read_range_into(
            record,
            values_offset + static_cast<std::uint64_t>(row) *
                static_cast<std::uint64_t>(width),
            std::as_writable_bytes(std::span<std::uint8_t>(
                raw.data(), static_cast<std::size_t>(width))));
        model.read_range_into(
            record,
            scales_offset + static_cast<std::uint64_t>(row) *
                static_cast<std::uint64_t>(scale_columns),
            std::as_writable_bytes(std::span<std::uint8_t>(
                raw.data() + width,
                static_cast<std::size_t>(scale_columns))));
        output.resize(static_cast<std::size_t>(width));
        for (int group = 0; group < scale_columns; ++group) {
            const auto scale = raw[static_cast<std::size_t>(width + group)];
            if (scale == 255) {
                throw std::runtime_error(
                    "native MXFP8 row contains an invalid E8M0 scale");
            }
            const auto lookup_base = static_cast<std::size_t>(scale) << 8u;
            const int begin = group * 32;
            for (int column = begin; column < begin + 32; ++column) {
                const auto value = raw[static_cast<std::size_t>(column)];
                if ((value & 0x7fu) == 0x7fu) {
                    throw std::runtime_error(
                        "native MXFP8 row contains an E4M3 NaN");
                }
                output[static_cast<std::size_t>(column)] =
                    decode_lut[lookup_base | value];
            }
        }
    }

    void touch(
        typename std::unordered_map<std::int64_t, CacheEntry>::iterator found)
        const {
        recency.splice(recency.begin(), recency, found->second.recency);
        found->second.recency = recency.begin();
    }

    void admit(RequestedRow row) const {
        if (cache_capacity == 0) return;
        auto found = cache.find(row.row);
        if (found != cache.end()) {
            touch(found);
            return;
        }
        while (cache.size() >= cache_capacity) {
            const auto victim = recency.back();
            recency.pop_back();
            cache.erase(victim);
        }
        recency.push_front(row.row);
        cache.emplace(
            row.row,
            CacheEntry{std::move(row.values), recency.begin()});
    }

    mlx::core::array gather(std::span<const std::int64_t> row_ids) const {
        if (row_ids.empty()) {
            std::vector<mlx::core::float16_t> empty;
            return mlx::core::array(
                empty.begin(), mlx::core::Shape{0, width});
        }
        std::vector<std::size_t> request_sources;
        request_sources.reserve(row_ids.size());
        std::unordered_map<std::int64_t, std::size_t> unique_index;
        unique_index.reserve(row_ids.size());
        std::vector<RequestedRow> requested;
        requested.reserve(row_ids.size());
        std::vector<std::size_t> misses;
        misses.reserve(row_ids.size());
        {
            std::scoped_lock lock(mutex);
            for (const auto row : row_ids) {
                if (row < 0 || row >= rows) {
                    throw std::out_of_range(
                        "native MXFP8 row index is outside the table");
                }
                const auto known = unique_index.find(row);
                if (known != unique_index.end()) {
                    request_sources.push_back(known->second);
                    continue;
                }
                const auto index = requested.size();
                unique_index.emplace(row, index);
                request_sources.push_back(index);
                auto found = cache.find(row);
                if (found == cache.end()) {
                    requested.push_back({row, {}, true});
                    misses.push_back(index);
                } else {
                    touch(found);
                    requested.push_back({row, found->second.values, false});
                }
            }
        }

        const auto io_begin = std::chrono::steady_clock::now();
        std::vector<std::future<void>> futures;
        futures.reserve(misses.size());
        for (const auto index : misses) {
            futures.push_back(workers.submit([this, &requested, index] {
                read_decode(requested[index].row, requested[index].values);
            }));
        }
        std::exception_ptr failure;
        for (auto& future : futures) {
            try {
                future.get();
            } catch (...) {
                if (!failure) failure = std::current_exception();
            }
        }
        if (failure) std::rethrow_exception(failure);
        const auto io_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - io_begin).count();

        std::vector<mlx::core::float16_t> output(
            row_ids.size() * static_cast<std::size_t>(width));
        for (std::size_t request = 0; request < request_sources.size();
             ++request) {
            const auto& source = requested[request_sources[request]].values;
            std::copy_n(
                source.data(),
                width,
                output.data() + request * static_cast<std::size_t>(width));
        }
        {
            std::scoped_lock lock(mutex);
            for (const auto index : misses) {
                admit(std::move(requested[index]));
            }
            counters.row_requests += row_ids.size();
            counters.cache_misses += misses.size();
            counters.cache_hits += row_ids.size() - misses.size();
            counters.rows_loaded += misses.size();
            counters.bytes_read += misses.size() * source_row_bytes;
            counters.read_calls += misses.size() * 2;
            counters.io_seconds += io_seconds;
        }
        return mlx::core::array(
            output.begin(),
            mlx::core::Shape{
                static_cast<int>(row_ids.size()), width});
    }

    MfqContainer model;
    std::string record;
    std::int64_t rows = 0;
    int width = 0;
    int scale_columns = 0;
    std::uint64_t values_offset = 0;
    std::uint64_t scales_offset = 0;
    std::size_t source_row_bytes = 0;
    std::size_t decoded_row_bytes = 0;
    std::size_t cache_capacity = 0;
    std::array<mlx::core::float16_t, 1u << 16u> decode_lut{};
    mutable WorkerPool workers;
    mutable std::mutex mutex;
    mutable std::list<std::int64_t> recency;
    mutable std::unordered_map<std::int64_t, CacheEntry> cache;
    mutable MlxMxfp8RowStoreStats counters;
};

MlxMxfp8RowStore::MlxMxfp8RowStore(
    const MfqContainer& model,
    std::string record,
    std::int64_t expected_rows,
    int expected_width,
    std::size_t cache_rows,
    std::size_t io_workers)
    : impl_(std::make_unique<Impl>(
          model,
          std::move(record),
          expected_rows,
          expected_width,
          cache_rows,
          io_workers)) {}

MlxMxfp8RowStore::~MlxMxfp8RowStore() = default;

mlx::core::array MlxMxfp8RowStore::gather(
    std::span<const std::int64_t> row_ids) const {
    return impl_->gather(row_ids);
}

std::int64_t MlxMxfp8RowStore::rows() const noexcept {
    return impl_->rows;
}

int MlxMxfp8RowStore::width() const noexcept {
    return impl_->width;
}

std::size_t MlxMxfp8RowStore::cached_rows() const noexcept {
    std::scoped_lock lock(impl_->mutex);
    return impl_->cache.size();
}

MlxMxfp8RowStoreStats MlxMxfp8RowStore::stats() const {
    std::scoped_lock lock(impl_->mutex);
    auto result = impl_->counters;
    result.resident_rows = impl_->cache.size();
    result.resident_payload_bytes =
        impl_->cache.size() * impl_->decoded_row_bytes;
    result.cache_limit_bytes =
        impl_->cache_capacity * impl_->decoded_row_bytes;
    return result;
}

void MlxMxfp8RowStore::clear() {
    std::scoped_lock lock(impl_->mutex);
    impl_->cache.clear();
    impl_->recency.clear();
}

} // namespace mfq::metal
