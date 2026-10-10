#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace mfq::metal {

struct QsaKvStoreConfig {
    std::size_t budget_bytes = 0;
    std::size_t target_index_bytes = 0;
    std::size_t buffer_bytes = 64 * 1024 * 1024;
    std::size_t microblock_rows = 4;
    std::string directory;
    std::function<void(std::size_t)> reserve;
};

struct QsaKvStoreStats {
    std::size_t index_bytes = 0;
    std::size_t hot_bytes = 0;
    std::size_t hot_limit = 0;
    std::size_t pending_bytes = 0;
    std::size_t disk_bytes = 0;
    std::uint64_t reads = 0;
    std::uint64_t hits = 0;
    std::uint64_t read_bytes = 0;
    std::uint64_t written_bytes = 0;
    std::size_t budget_bytes = 0;
};

class QsaKvStore {
public:
    struct Block;
    using BlockPtr = std::shared_ptr<const Block>;
    explicit QsaKvStore(QsaKvStoreConfig config);
    ~QsaKvStore();
    QsaKvStore(const QsaKvStore&) = delete;
    QsaKvStore& operator=(const QsaKvStore&) = delete;
    BlockPtr write(std::vector<std::uint8_t> bytes, int priority = 0);
    std::shared_ptr<const std::vector<std::uint8_t>> read(const BlockPtr& block);
    void flush();
    QsaKvStoreStats stats() const;
    const QsaKvStoreConfig& config() const noexcept;
private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

struct QsaKvSequenceSnapshot {
    std::shared_ptr<QsaKvStore> store;
    std::size_t row_bytes = 0;
    std::size_t block_rows = 0;
    std::size_t position = 0;
    int priority = 0;
    std::vector<QsaKvStore::BlockPtr> blocks;
    std::vector<std::uint8_t> tail;
    QsaKvStore::BlockPtr tail_block;
    QsaKvSequenceSnapshot prefix(std::size_t rows) const;
    void read_rows(std::size_t start, std::size_t rows, std::uint8_t* output) const;
};

class QsaKvSequence {
public:
    QsaKvSequence(std::shared_ptr<QsaKvStore> store, std::size_t row_bytes,
        std::size_t block_rows, int priority = 0);
    void append(const std::uint8_t* rows, std::size_t count);
    void seal_tail();
    void trim(std::size_t position);
    void restore(const QsaKvSequenceSnapshot& state);
    QsaKvSequenceSnapshot snapshot() const;
    const QsaKvSequenceSnapshot& view() const noexcept { return state_; }
    std::size_t position() const noexcept { return state_.position; }
private:
    QsaKvSequenceSnapshot state_;
};

}
