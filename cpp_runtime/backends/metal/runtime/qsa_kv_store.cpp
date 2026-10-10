#include "qsa_kv_store.h"

#include <algorithm>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <exception>
#include <filesystem>
#include <list>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <unistd.h>
#include <fcntl.h>

namespace mfq::metal {

struct QsaKvStore::Block {
    std::uint64_t id = 0;
    std::size_t offset = 0;
    std::size_t bytes = 0;
    bool ready = false;
    bool queued = false;
    int priority = 0;
    std::shared_ptr<const std::vector<std::uint8_t>> pending;
};

struct QsaKvStore::Impl {
    explicit Impl(QsaKvStoreConfig value) : config(std::move(value)) {
        if (config.buffer_bytes < 4096 || config.budget_bytes < config.buffer_bytes)
            throw std::invalid_argument("QSA KV budget is smaller than its required I/O buffer");
        auto directory = config.directory.empty() ? std::filesystem::temp_directory_path()
            : std::filesystem::path(config.directory);
        std::filesystem::create_directories(directory);
        auto name = (directory / "mfq-qsa-kv-XXXXXX").string();
        std::vector<char> path(name.begin(), name.end());
        path.push_back(0);
        fd = ::mkstemp(path.data());
        if (fd < 0) throw std::runtime_error("cannot create QSA KV offload file");
        if (::unlink(path.data()) != 0) {
            ::close(fd);
            throw std::runtime_error("cannot make QSA KV offload file session-owned");
        }
        worker = std::thread([this] { run(); });
    }

    void stop() {
        {
            std::lock_guard<std::recursive_mutex> guard(mutex);
            stopping = true;
            changed.notify_all();
        }
        if (worker.joinable()) worker.join();
    }
    ~Impl() {
        stop();
        ::close(fd);
    }

    void check() const { if (error) std::rethrow_exception(error); }

    std::size_t hot_limit() const {
        const auto available = config.budget_bytes - config.buffer_bytes;
        return available;
    }

    void evict(std::unique_lock<std::recursive_mutex>& guard) {
        while (hot_bytes > hot_limit() && !hot.empty()) {
            int priority = 0;
            while (priority < 2 && lru[priority].empty()) ++priority;
            auto found = hot.find(lru[priority].back());
            if (auto block = found->second.block.lock(); block && !block->ready && !block->queued) {
                const auto size = block->bytes;
                const auto id = block->id;
                changed.wait(guard, [&] { return error || pending_bytes <= config.buffer_bytes / 2 - size; });
                check();
                found = hot.find(id);
                if (found == hot.end()) continue;
                if (block->queued || block->ready) continue;
                block->pending = found->second.bytes;
                block->queued = true;
                pending_bytes += size;
                disk_bytes += size;
                queue.push_back(std::move(block));
                changed.notify_all();
            }
            hot_bytes -= found->second.bytes->size();
            lru[priority].erase(found->second.order);
            hot.erase(found);
        }
    }

    void release(const Block* block) {
        std::lock_guard<std::recursive_mutex> guard(mutex);
        auto found = hot.find(block->id);
        if (found != hot.end()) {
            hot_bytes -= found->second.bytes->size();
            lru[block->priority].erase(found->second.order);
            hot.erase(found);
        }
        free_offsets[block->bytes].push_back(block->offset);
        if (block->queued) disk_bytes -= block->bytes;
        if (--live_blocks == 0 && ::ftruncate(fd, 0) == 0) {
            free_offsets.clear();
            next_offset = 0;
        }
    }

    void run() {
        while (true) {
            std::shared_ptr<Block> block;
            {
                std::unique_lock<std::recursive_mutex> guard(mutex);
                changed.wait(guard, [&] { return stopping || !queue.empty(); });
                if (queue.empty()) return;
                block = std::move(queue.front());
                queue.pop_front();
            }
            std::exception_ptr failure;
            try {
                std::size_t done = 0;
                while (done < block->bytes) {
                    const auto written = ::pwrite(fd, block->pending->data() + done,
                        block->bytes - done, static_cast<off_t>(block->offset + done));
                    if (written < 0 && errno == EINTR) continue;
                    if (written <= 0) throw std::runtime_error("QSA KV SSD write failed");
                    done += static_cast<std::size_t>(written);
                }
            } catch (...) { failure = std::current_exception(); }
            {
                std::lock_guard<std::recursive_mutex> guard(mutex);
                if (failure && !error) error = failure;
                block->ready = !failure;
                if (!failure) written_bytes += block->bytes;
                pending_bytes -= block->bytes;
                block->pending.reset();
                changed.notify_all();
            }
        }
    }

    struct Hot {
        std::shared_ptr<const std::vector<std::uint8_t>> bytes;
        std::list<std::uint64_t>::iterator order;
        std::weak_ptr<Block> block;
    };
    QsaKvStoreConfig config;
    int fd = -1;
    mutable std::recursive_mutex mutex;
    std::condition_variable_any changed;
    std::thread worker;
    bool stopping = false;
    std::exception_ptr error;
    std::size_t hot_bytes = 0, pending_bytes = 0, disk_bytes = 0;
    std::size_t next_offset = 0;
    std::uint64_t next_id = 0, reads = 0, hits = 0, live_blocks = 0;
    std::uint64_t read_bytes = 0, written_bytes = 0;
    std::deque<std::shared_ptr<Block>> queue;
    std::unordered_map<std::uint64_t, Hot> hot;
    std::list<std::uint64_t> lru[3];
    std::map<std::size_t, std::vector<std::size_t>> free_offsets;
};

QsaKvStore::QsaKvStore(QsaKvStoreConfig config) : impl_(std::make_shared<Impl>(std::move(config))) {}
QsaKvStore::~QsaKvStore() {
    try { flush(); } catch (...) {}
    impl_->stop();
}
const QsaKvStoreConfig& QsaKvStore::config() const noexcept { return impl_->config; }

QsaKvStore::BlockPtr QsaKvStore::write(std::vector<std::uint8_t> bytes, int priority) {
    if (bytes.empty() || bytes.size() > impl_->config.buffer_bytes / 2 || priority < 0 || priority > 2)
        throw std::invalid_argument("QSA KV microblock exceeds the bounded write buffer");
    const auto size = bytes.size();
    if (impl_->config.reserve) impl_->config.reserve(size);
    std::unique_lock<std::recursive_mutex> guard(impl_->mutex);
    impl_->check();
    auto* raw = new Block;
    std::weak_ptr<Impl> weak = impl_;
    auto block = std::shared_ptr<Block>(raw, [weak](Block* value) {
        if (auto impl = weak.lock()) impl->release(value);
        delete value;
    });
    block->id = ++impl_->next_id;
    ++impl_->live_blocks;
    block->priority = priority;
    auto& free = impl_->free_offsets[size];
    if (free.empty()) {
        block->offset = impl_->next_offset;
        impl_->next_offset += size;
    } else {
        block->offset = free.back();
        free.pop_back();
    }
    block->bytes = size;
    auto payload = std::make_shared<const std::vector<std::uint8_t>>(std::move(bytes));
    impl_->lru[priority].push_front(block->id);
    impl_->hot.emplace(block->id, Impl::Hot{payload, impl_->lru[priority].begin(), block});
    impl_->hot_bytes += size;
    impl_->evict(guard);
    return block;
}

std::shared_ptr<const std::vector<std::uint8_t>> QsaKvStore::read(const BlockPtr& block) {
    if (!block) throw std::invalid_argument("missing QSA KV microblock");
    std::unique_lock<std::recursive_mutex> guard(impl_->mutex);
    impl_->check();
    if (block->pending) {
        ++impl_->hits;
        return block->pending;
    }
    auto found = impl_->hot.find(block->id);
    if (found != impl_->hot.end()) {
        ++impl_->hits;
        auto& order = impl_->lru[block->priority];
        order.splice(order.begin(), order, found->second.order);
        return found->second.bytes;
    }
    if (!block->ready) throw std::runtime_error("QSA KV microblock was not written");
    auto bytes = std::make_shared<std::vector<std::uint8_t>>(block->bytes);
    std::size_t done = 0;
    while (done < bytes->size()) {
        const auto read = ::pread(impl_->fd, bytes->data() + done, bytes->size() - done,
            static_cast<off_t>(block->offset + done));
        if (read < 0 && errno == EINTR) continue;
        if (read <= 0) throw std::runtime_error("QSA KV SSD read failed");
        done += static_cast<std::size_t>(read);
    }
    ++impl_->reads;
    impl_->read_bytes += bytes->size();
    if (bytes->size() <= impl_->hot_limit()) {
        auto& order = impl_->lru[block->priority];
        order.push_front(block->id);
        impl_->hot.emplace(block->id, Impl::Hot{bytes, order.begin(), std::const_pointer_cast<Block>(block)});
        impl_->hot_bytes += bytes->size();
        impl_->evict(guard);
    }
    return bytes;
}

void QsaKvStore::flush() {
    std::unique_lock<std::recursive_mutex> guard(impl_->mutex);
    impl_->changed.wait(guard, [&] { return impl_->error || impl_->pending_bytes == 0; });
    impl_->check();
}

QsaKvStoreStats QsaKvStore::stats() const {
    std::lock_guard<std::recursive_mutex> guard(impl_->mutex);
    std::size_t index = 0;
    for (const auto& id : impl_->lru[2]) index += impl_->hot.at(id).bytes->size();
    return {index, impl_->hot_bytes, impl_->hot_limit(), impl_->pending_bytes,
        impl_->disk_bytes, impl_->reads, impl_->hits, impl_->read_bytes, impl_->written_bytes, impl_->config.budget_bytes};
}

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
        (state.tail_block ? state.tail_block->bytes : state.tail.size()) !=
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
