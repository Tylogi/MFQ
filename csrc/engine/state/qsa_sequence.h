#pragma once
#include "qsa_blocks.h"

namespace mfq::cache {
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
