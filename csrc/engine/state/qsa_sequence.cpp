#include "qsa_sequence.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace mfq::cache {
QsaKvSequence::QsaKvSequence(std::shared_ptr<QsaKvStore> store,
    std::size_t row_bytes, std::size_t block_rows, int priority) {
    if (!store || !row_bytes || !block_rows ||
        row_bytes > store->config().buffer_bytes / 2 / block_rows)
        throw std::invalid_argument("invalid QSA KV microblock geometry");
    state_.store = std::move(store);
    state_.row_bytes = row_bytes;
    state_.block_rows = block_rows;
    state_.priority = priority;
}

void QsaKvSequence::append(const std::uint8_t* rows, std::size_t count) {
    if (count && !rows) throw std::invalid_argument("missing QSA KV rows");
    const auto position = state_.position;
    const auto blocks = state_.blocks.size();
    auto tail = state_.tail;
    const auto tail_block = state_.tail_block;
    try {
        if (count && state_.tail_block) {
            const auto bytes = state_.store->read(state_.tail_block);
            state_.tail.assign(bytes->begin(), bytes->end());
            state_.tail_block.reset();
        }
        while (count) {
            const auto tail_rows = state_.tail.size() / state_.row_bytes;
            const auto take = std::min(count, state_.block_rows - tail_rows);
            const auto bytes = take * state_.row_bytes;
            state_.tail.insert(state_.tail.end(), rows, rows + bytes);
            state_.position += take;
            rows += bytes;
            count -= take;
            if (state_.tail.size() == state_.row_bytes * state_.block_rows) {
                state_.blocks.push_back(state_.store->write(std::move(state_.tail), state_.priority));
                state_.tail.clear();
            }
        }
    } catch (...) {
        state_.blocks.resize(blocks);
        state_.tail = std::move(tail);
        state_.tail_block = tail_block;
        state_.position = position;
        throw;
    }
}

void QsaKvSequence::seal_tail() {
    if (!state_.tail.empty()) {
        state_.tail_block = state_.store->write(state_.tail, state_.priority);
        state_.tail.clear();
    }
}

QsaKvSequenceSnapshot QsaKvSequence::snapshot() const { return state_; }
void QsaKvSequence::trim(std::size_t position) {
    if (position > state_.position) throw std::out_of_range("QSA KV trim exceeds its position");
    std::vector<std::uint8_t> tail((position % state_.block_rows) * state_.row_bytes);
    if (!tail.empty()) state_.read_rows(position / state_.block_rows * state_.block_rows,
        position % state_.block_rows, tail.data());
    state_.blocks.resize(position / state_.block_rows);
    state_.tail_block.reset();
    state_.tail = std::move(tail);
    state_.position = position;
    seal_tail();
}
void QsaKvSequence::restore(const QsaKvSequenceSnapshot& state) {
    if (state.store != state_.store || state.row_bytes != state_.row_bytes ||
        state.block_rows != state_.block_rows || state.priority != state_.priority ||
        state.blocks.size() != state.position / state.block_rows ||
        (state.tail_block ? QsaKvStore::block_bytes(state.tail_block) : state.tail.size()) !=
            (state.position % state.block_rows) * state.row_bytes ||
        std::any_of(state.blocks.begin(), state.blocks.end(), [](const auto& block) { return !block; }))
        throw std::invalid_argument("QSA KV snapshot geometry mismatch");
    state_ = state;
}

QsaKvSequenceSnapshot QsaKvSequenceSnapshot::prefix(std::size_t rows) const {
    if (rows > position) throw std::out_of_range("QSA KV snapshot prefix exceeds its position");
    auto result = *this;
    result.position = rows;
    const auto complete = rows / block_rows;
    result.tail.resize((rows % block_rows) * row_bytes);
    if (!result.tail.empty()) read_rows(complete * block_rows, rows % block_rows, result.tail.data());
    result.blocks.resize(complete);
    result.tail_block.reset();
    if (!result.tail.empty()) {
        result.tail_block = result.store->write(std::move(result.tail), result.priority);
        result.tail.clear();
    }
    return result;
}

void QsaKvSequenceSnapshot::read_rows(std::size_t start, std::size_t rows, std::uint8_t* output) const {
    if (start > position || rows > position - start || (rows && !output))
        throw std::out_of_range("QSA KV read exceeds its snapshot");
    while (rows) {
        const auto block = start / block_rows;
        const auto within = start % block_rows;
        const auto take = std::min(rows, block_rows - within);
        const auto size = take * row_bytes;
        if (block == blocks.size()) {
            if (tail_block) {
                const auto bytes = store->read(tail_block);
                std::memcpy(output, bytes->data() + within * row_bytes, size);
            } else std::memcpy(output, tail.data() + within * row_bytes, size);
        } else {
            const auto bytes = store->read(blocks.at(block));
            std::memcpy(output, bytes->data() + within * row_bytes, size);
        }
        output += size;
        start += take;
        rows -= take;
    }
}

}
