#include "kv_offload.h"

#include <algorithm>
#include <cstring>
#include <list>
#include <map>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

namespace mfq::cuda {
namespace tb = mfq_tensor_backend;
namespace {
tb::Tensor indices(const std::vector<int64_t>& values, const tb::TensorOptions& options) {
    auto cpu = tb::empty({static_cast<int64_t>(values.size())}, options.device(tb::kCPU).dtype(tb::kInt64));
    if (!values.empty()) std::memcpy(cpu.data_ptr(), values.data(), values.size() * sizeof(int64_t));
    return cpu.to(options.dtype(tb::kInt64));
}
}

struct KvOffloadStore::Block {
    std::uint64_t id = 0;
    std::size_t bytes = 0;
    tb::Tensor gpu;
    tb::TensorOptions options;
    int64_t rows = 0, width = 0;
    cache::QsaKvStore::BlockPtr cold;
};

struct KvOffloadStore::Impl {
    explicit Impl(KvOffloadConfig value) : config(std::move(value)) {
        if (config.buffer_bytes < 4096 || config.gpu_budget_bytes < config.buffer_bytes)
            throw std::invalid_argument("CUDA KV budget cannot hold its I/O staging buffer");
        cache::QsaKvStoreConfig cold;
        cold.buffer_bytes = config.buffer_bytes;
        cold.budget_bytes = config.ram_budget_bytes + config.buffer_bytes;
        cold.directory = config.directory;
        ram = std::make_shared<cache::QsaKvStore>(std::move(cold));
    }
    std::size_t hot_limit() const { return config.gpu_budget_bytes - config.buffer_bytes; }
    void touch(const BlockPtr& block) {
        auto found = hot.find(block->id);
        if (found != hot.end()) lru.splice(lru.begin(), lru, found->second.order);
        else {
            lru.push_front(block->id);
            hot.emplace(block->id, Hot{block, lru.begin()});
            gpu_bytes += block->bytes;
        }
    }
    void evict() {
        while (gpu_bytes > hot_limit() && !lru.empty()) {
            const auto id = lru.back();
            auto found = hot.find(id);
            auto block = found->second.block.lock();
            if (block) {
                if (!block->cold) {
                    auto cpu = block->gpu.to(tb::kCPU).contiguous();
                    std::vector<std::uint8_t> payload(block->bytes);
                    std::memcpy(payload.data(), cpu.data_ptr(), payload.size());
                    block->cold = ram->write(std::move(payload));
                    d2h_bytes += block->bytes;
                }
                block->gpu = {};
                gpu_bytes -= block->bytes;
                ++evictions;
            }
            lru.pop_back(); hot.erase(found);
        }
    }
    void release(Block* block) {
        std::lock_guard<std::recursive_mutex> lock(mutex);
        auto found = hot.find(block->id);
        if (found != hot.end()) {
            gpu_bytes -= block->bytes;
            lru.erase(found->second.order); hot.erase(found);
        }
    }
    struct Hot { std::weak_ptr<Block> block; std::list<std::uint64_t>::iterator order; };
    KvOffloadConfig config;
    std::shared_ptr<cache::QsaKvStore> ram;
    mutable std::recursive_mutex mutex;
    std::list<std::uint64_t> lru;
    std::unordered_map<std::uint64_t, Hot> hot;
    std::size_t gpu_bytes = 0;
    std::uint64_t next_id = 0, hits = 0, h2d_bytes = 0, d2h_bytes = 0, evictions = 0;
};

KvOffloadStore::KvOffloadStore(KvOffloadConfig config) : impl_(std::make_shared<Impl>(std::move(config))) {}
KvOffloadStore::~KvOffloadStore() = default;
const KvOffloadConfig& KvOffloadStore::config() const { return impl_->config; }

KvOffloadStore::BlockPtr KvOffloadStore::write(const tb::Tensor& rows) {
    if (!rows.is_cuda() || rows.dim() != 2 || !rows.numel() || rows.scalar_type() != tb::kFloat16)
        throw std::invalid_argument("CUDA KV microblock must contain nonempty FP16 GPU rows");
    const auto bytes = static_cast<std::size_t>(rows.numel()) * rows.element_size();
    if (bytes > impl_->config.buffer_bytes / 2)
        throw std::invalid_argument("CUDA KV microblock exceeds its write staging buffer");
    std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
    std::weak_ptr<Impl> owner = impl_;
    auto block = BlockPtr(new Block, [owner](Block* value) {
        if (auto impl = owner.lock()) impl->release(value);
        delete value;
    });
    block->id = ++impl_->next_id;
    block->bytes = bytes; block->rows = rows.size(0); block->width = rows.size(1);
    block->options = rows.options(); block->gpu = rows.contiguous().clone();
    impl_->touch(block); impl_->evict();
    return block;
}

tb::Tensor KvOffloadStore::read(const BlockPtr& block) {
    if (!block) throw std::invalid_argument("missing CUDA KV block");
    std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
    if (block->gpu.defined()) { ++impl_->hits; impl_->touch(block); return block->gpu; }
    auto payload = impl_->ram->read(block->cold);
    const auto options = block->options;
    auto cpu = tb::empty({block->rows, block->width}, options.device(tb::kCPU));
    std::memcpy(cpu.data_ptr(), payload->data(), block->bytes);
    auto gpu = cpu.to(block->options);
    impl_->h2d_bytes += block->bytes;
    if (block->bytes <= impl_->hot_limit()) {
        block->gpu = gpu; impl_->touch(block); impl_->evict();
    }
    return gpu;
}

void KvOffloadStore::flush() { impl_->ram->flush(); }
std::vector<std::pair<std::string, double>> KvOffloadStore::metrics() const {
    std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
    const auto cold = impl_->ram->stats();
    return {{"qsa_kv_offload_enabled", 1.0}, {"qsa_kv_resident_bytes", double(impl_->gpu_bytes)},
        {"qsa_kv_budget_bytes", double(impl_->config.gpu_budget_bytes)},
        {"qsa_kv_ram_enabled", impl_->config.ram_budget_bytes ? 1.0 : 0.0},
        {"qsa_kv_ram_bytes", double(cold.hot_bytes)}, {"qsa_kv_ram_limit_bytes", double(impl_->config.ram_budget_bytes)},
        {"qsa_kv_ssd_bytes", double(cold.disk_bytes)}, {"qsa_kv_pending_bytes", double(cold.pending_bytes)},
        {"qsa_kv_ssd_reads", double(cold.reads)}, {"qsa_kv_ram_hits", double(cold.hits)},
        {"qsa_kv_gpu_hits", double(impl_->hits)}, {"qsa_kv_ssd_read_bytes", double(cold.read_bytes)},
        {"qsa_kv_ssd_written_bytes", double(cold.written_bytes)},
        {"qsa_kv_h2d_bytes", double(impl_->h2d_bytes)}, {"qsa_kv_d2h_bytes", double(impl_->d2h_bytes)},
        {"qsa_kv_gpu_evictions", double(impl_->evictions)}};
}

KvOffloadSequence::KvOffloadSequence(std::shared_ptr<KvOffloadStore> store, int64_t width, int64_t block_rows)
    : store_(std::move(store)), width_(width), block_rows_(block_rows) {
    if (!store_ || width <= 0 || block_rows <= 0 || std::size_t(width) * block_rows * 2 > store_->config().buffer_bytes / 2)
        throw std::invalid_argument("invalid CUDA KV sequence geometry");
}
void KvOffloadSequence::reset() { blocks_.clear(); position_ = 0; }
void KvOffloadSequence::append(const tb::Tensor& source) {
    if (!source.is_cuda() || source.dim() != 2 || source.size(1) != width_ || source.size(0) <= 0)
        throw std::invalid_argument("invalid CUDA KV rows");
    const auto offset = position_;
    const auto previous_count = blocks_.size();
    const auto previous_tail = position_ % block_rows_ ? blocks_.back() : KvOffloadStore::BlockPtr{};
    try {
        auto rows = source.to(tb::kFloat16);
        if (position_ % block_rows_) {
            rows = tb::cat({store_->read(blocks_.back()), rows}, 0);
            blocks_.pop_back();
        }
        for (int64_t begin = 0; begin < rows.size(0); begin += block_rows_)
            blocks_.push_back(store_->write(rows.narrow(0, begin, std::min(block_rows_, rows.size(0) - begin))));
        position_ += source.size(0);
    } catch (...) {
        blocks_.resize(previous_count - (previous_tail ? 1 : 0));
        if (previous_tail) blocks_.push_back(previous_tail);
        position_ = offset; throw;
    }
}
void KvOffloadSequence::truncate(int64_t position) {
    if (position < 0 || position > position_) throw std::out_of_range("CUDA KV trim exceeds its sequence");
    if (position == position_) return;
    const auto full = position / block_rows_, tail = position % block_rows_;
    KvOffloadStore::BlockPtr remainder;
    if (tail) remainder = store_->write(store_->read(blocks_.at(full)).narrow(0, 0, tail));
    blocks_.resize(full);
    if (remainder) blocks_.push_back(std::move(remainder));
    position_ = position;
}
tb::Tensor KvOffloadSequence::gather(const std::vector<int64_t>& rows) const {
    if (rows.empty() || blocks_.empty()) throw std::invalid_argument("empty CUDA KV selection");
    std::map<int64_t, std::pair<std::vector<int64_t>, std::vector<int64_t>>> groups;
    for (std::size_t output = 0; output < rows.size(); ++output) {
        if (rows[output] < 0 || rows[output] >= position_) throw std::out_of_range("CUDA KV selected row exceeds its sequence");
        auto& group = groups[rows[output] / block_rows_];
        group.first.push_back(rows[output] % block_rows_); group.second.push_back(output);
    }
    const auto options = blocks_.front()->options;
    auto result = tb::empty({static_cast<int64_t>(rows.size()), width_}, options);
    for (const auto& [block, group] : groups) {
        auto payload = store_->read(blocks_.at(block));
        auto selected = payload.index_select(0, indices(group.first, options));
        result.index_copy_(0, indices(group.second, options), selected);
    }
    return result;
}
tb::Tensor KvOffloadSequence::range(int64_t start, int64_t count) const {
    if (start < 0 || count <= 0 || start > position_ || count > position_ - start)
        throw std::out_of_range("CUDA KV range exceeds its sequence");
    std::vector<int64_t> rows(count); std::iota(rows.begin(), rows.end(), start);
    return gather(rows);
}
}
