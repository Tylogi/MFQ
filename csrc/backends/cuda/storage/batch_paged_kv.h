#pragma once

#include "kv_cache.h"
#include "paged_kv_allocator.h"

#include <algorithm>
#include <limits>
#include <map>
#include <optional>
#include <span>
#include <vector>

namespace mfq::cuda {

struct PagedKvSequence { std::vector<int32_t> physical_pages; };
struct PagedKvLayerGeometry {
    int device;
    int64_t heads, head_dim;
    mfq_tensor_backend::ScalarType dtype = mfq_tensor_backend::kFloat16;
};
struct PagedKvStats {
    int64_t page_size = 0, live_pages = 0, peak_live_pages = 0, capacity_pages = 0;
    int64_t reserved_bytes = 0, allocations = 0, reuses = 0, releases = 0, table_updates = 0;
};

struct BatchStorageStats {
    int64_t active_rows = 0, slot_releases = 0;
    std::optional<PagedKvStats> paged;
};

// Physical storage only: callers provide geometry and attach the returned views
// to their native layers. No model selection, block pointers or Engine lifecycle.
class BatchPagedKvPool {
  public:
    static constexpr int64_t kPageSize = 16, kPagesPerChunk = 64;
    BatchPagedKvPool(std::span<const PagedKvLayerGeometry> geometry,
                     int32_t maximum_sequences, int64_t context)
        : maximum_sequences_(maximum_sequences), logical_pages_(context_pages(context)),
          maximum_pages_(maximum_pages(maximum_sequences, logical_pages_)),
          allocator_(maximum_pages_), host_table_(maximum_sequences * logical_pages_, -1) {
        MFQ_RUNTIME_CHECK(!geometry.empty(), "Paged KV requires at least one layer");
        const auto chunks = (maximum_pages_ + kPagesPerChunk - 1) / kPagesPerChunk;
        for (const auto& layer : geometry) {
            MFQ_RUNTIME_CHECK(layer.heads > 0 && layer.head_dim > 0, "invalid paged KV geometry");
            MfqCudaGuard guard(layer.device);
            if (!tables_.contains(layer.device)) {
                auto table = mfq_tensor_backend::empty({maximum_sequences_, logical_pages_},
                    mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::Device(
                        mfq_tensor_backend::kCUDA, layer.device)).dtype(mfq_tensor_backend::kInt32));
                table.fill_(-1);
                tables_.emplace(layer.device, std::move(table));
            }
            auto options = mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::Device(
                mfq_tensor_backend::kCUDA, layer.device)).dtype(mfq_tensor_backend::kInt64);
            layers_.push_back({layer, mfq_tensor_backend::zeros({int64_t(chunks)}, options),
                                    mfq_tensor_backend::zeros({int64_t(chunks)}, options), {}, {}});
        }
    }

    bool ensure_tokens(PagedKvSequence& sequence, int64_t tokens) {
        MFQ_RUNTIME_CHECK(tokens >= 0 && tokens <= logical_pages_ * kPageSize,
                          "Paged KV sequence exceeds configured context");
        const size_t required = (tokens + kPageSize - 1) / kPageSize;
        if (required <= sequence.physical_pages.size()) return false;
        const auto previous = sequence.physical_pages.size();
        auto pages = allocator_.allocate(required - previous);
        try {
            sequence.physical_pages.insert(sequence.physical_pages.end(), pages.begin(), pages.end());
            ensure_storage(allocator_.high_watermark());
        } catch (...) {
            sequence.physical_pages.resize(previous);
            allocator_.release(pages);
            throw;
        }
        return true;
    }
    void release(PagedKvSequence& sequence) {
        if (sequence.physical_pages.empty()) return;
        allocator_.release(sequence.physical_pages);
        sequence.physical_pages.clear();
    }
    void bind(std::span<const PagedKvSequence* const> sequences) {
        MFQ_RUNTIME_CHECK(!sequences.empty() && sequences.size() <= maximum_sequences_,
                          "invalid paged KV batch rows");
        std::fill_n(host_table_.begin(), sequences.size() * logical_pages_, -1);
        for (size_t row = 0; row < sequences.size(); ++row) {
            const auto* sequence = sequences[row];
            MFQ_RUNTIME_CHECK(sequence && sequence->physical_pages.size() <= logical_pages_,
                              "invalid paged KV sequence");
            for (size_t page = 0; page < sequence->physical_pages.size(); ++page) {
                const auto physical = sequence->physical_pages[page];
                MFQ_RUNTIME_CHECK(allocator_.owns(physical), "paged KV references an unowned page");
                host_table_[row * logical_pages_ + page] = physical;
            }
        }
        for (auto& [device, table] : tables_) {
            MfqCudaGuard guard(device);
            const auto stream = mfq_get_current_cuda_stream();
            MFQ_CUDA_CHECK(cudaMemcpyAsync(table.data_ptr<int32_t>(), host_table_.data(),
                sequences.size() * logical_pages_ * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
            // The next bind may reuse both buffers; finish their previous users.
            MFQ_CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        bound_rows_ = sequences.size();
        ++table_updates_;
    }
    KVCache view(size_t layer) const {
        MFQ_RUNTIME_CHECK(bound_rows_ > 0, "paged KV views require bound rows");
        const auto& pool = layers_.at(layer);
        const auto& geometry = pool.geometry;
        return KVCache::paged_view(pool.k_pointers, pool.v_pointers, tables_.at(geometry.device),
            bound_rows_, geometry.heads, geometry.head_dim, kPageSize, kPagesPerChunk, geometry.dtype);
    }
    PagedKvStats stats() const noexcept {
        return {kPageSize, int64_t(allocator_.live_pages()), int64_t(allocator_.peak_live_pages()),
                int64_t(storage_pages_), int64_t(reserved_bytes_), int64_t(allocator_.allocation_count()),
                int64_t(allocator_.reuse_count()), int64_t(allocator_.release_count()), table_updates_};
    }

  private:
    using Tensor = mfq_tensor_backend::Tensor;
    struct LayerPool {
        PagedKvLayerGeometry geometry;
        Tensor k_pointers, v_pointers;
        std::vector<Tensor> k_chunks, v_chunks;
    };
    static int64_t context_pages(int64_t context) {
        MFQ_RUNTIME_CHECK(context > 0 && context <= INT64_MAX - kPageSize + 1,
                          "invalid paged KV context");
        return (context + kPageSize - 1) / kPageSize;
    }
    static size_t maximum_pages(int32_t sequences, int64_t pages) {
        MFQ_RUNTIME_CHECK(sequences > 0 && pages > 0 &&
            uint64_t(sequences) <= uint64_t(INT32_MAX) / (uint64_t(pages) + 1),
            "Paged KV physical page count exceeds int32");
        // Each padded physical row owns a scratch page, never a paused prefix.
        return size_t(sequences) * (size_t(pages) + 1);
    }
    void ensure_storage(size_t required) {
        while (storage_pages_ < required) {
            const auto pages = std::min<size_t>(kPagesPerChunk, maximum_pages_ - storage_pages_);
            MFQ_RUNTIME_CHECK(pages > 0, "Paged KV physical storage exhausted");
            std::vector<std::pair<Tensor, Tensor>> pending;
            for (const auto& layer : layers_) {
                const auto& geometry = layer.geometry;
                MfqCudaGuard guard(geometry.device);
                auto options = mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::Device(
                    mfq_tensor_backend::kCUDA, geometry.device)).dtype(geometry.dtype);
                pending.emplace_back(
                    mfq_tensor_backend::empty({int64_t(pages), geometry.heads, kPageSize, geometry.head_dim}, options),
                    mfq_tensor_backend::empty({int64_t(pages), geometry.heads, kPageSize, geometry.head_dim}, options));
            }
            const auto index = storage_pages_ / kPagesPerChunk;
            for (size_t layer = 0; layer < layers_.size(); ++layer) {
                auto& pool = layers_[layer];
                MfqCudaGuard guard(pool.geometry.device);
                const auto k = reinterpret_cast<int64_t>(pending[layer].first.data_ptr());
                const auto v = reinterpret_cast<int64_t>(pending[layer].second.data_ptr());
                const auto stream = mfq_get_current_cuda_stream();
                MFQ_CUDA_CHECK(cudaMemcpyAsync(pool.k_pointers.data_ptr<int64_t>() + index,
                    &k, sizeof(k), cudaMemcpyHostToDevice, stream));
                MFQ_CUDA_CHECK(cudaMemcpyAsync(pool.v_pointers.data_ptr<int64_t>() + index,
                    &v, sizeof(v), cudaMemcpyHostToDevice, stream));
                MFQ_CUDA_CHECK(cudaStreamSynchronize(stream));
            }
            for (size_t layer = 0; layer < layers_.size(); ++layer) {
                reserved_bytes_ += pending[layer].first.nbytes() + pending[layer].second.nbytes();
                layers_[layer].k_chunks.push_back(std::move(pending[layer].first));
                layers_[layer].v_chunks.push_back(std::move(pending[layer].second));
            }
            storage_pages_ += pages;
        }
    }
    int32_t maximum_sequences_;
    int64_t logical_pages_, bound_rows_ = 0, table_updates_ = 0;
    size_t maximum_pages_, storage_pages_ = 0, reserved_bytes_ = 0;
    mfq::engine::PagedKvPageAllocator allocator_;
    std::vector<int32_t> host_table_;
    std::map<int, Tensor> tables_;
    std::vector<LayerPool> layers_;
};
} // namespace mfq::cuda
