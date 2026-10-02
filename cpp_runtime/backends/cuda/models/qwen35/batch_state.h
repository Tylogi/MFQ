#pragma once

#include "mfq_tensor_backend.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

struct CudaExecutionContext;

namespace mfq::models { template <class Backend> struct CausalLm; }

namespace mfq::cuda {
struct Qwen35Model;
template <typename Model> struct CudaCausalOps;
template <typename Model> using CausalLm = mfq::models::CausalLm<CudaCausalOps<Model>>;
using Qwen35CausalLm = CausalLm<Qwen35Model>;
} // namespace mfq::cuda

namespace mfq::cuda::continuous {
class QwenPagedKvArena;
struct QwenPagedKvSequence {
    std::vector<std::int32_t> physical_pages;
};
} // namespace mfq::cuda::continuous

namespace mfq::cuda::qwen35 {

struct QwenBatchLayerState {
    enum class Kind { FullAttention, Recurrent };
    Kind kind = Kind::FullAttention;
    mfq_tensor_backend::Tensor first;
    mfq_tensor_backend::Tensor second;
    bool ring = false;
    bool paged = false;
};

struct QwenBatchState {
    std::int64_t batch = 0;
    std::vector<QwenBatchLayerState> layers;
};

struct QwenPagedKvStats {
    std::int64_t page_size = 0;
    std::int64_t live_pages = 0;
    std::int64_t peak_live_pages = 0;
    std::int64_t capacity_pages = 0;
    std::int64_t reserved_bytes = 0;
    std::int64_t allocations = 0;
    std::int64_t reuses = 0;
    std::int64_t releases = 0;
    std::int64_t table_updates = 0;
};

// A request owns its native cache state; the adapter alone binds it to slots.
// Requests have stable addresses while admitted and release before retirement.
class QwenBatchRequestState {
  public:
    QwenBatchRequestState() = default;
    QwenBatchRequestState(const QwenBatchRequestState &) = delete;
    QwenBatchRequestState &operator=(const QwenBatchRequestState &) = delete;
    std::int32_t slot() const noexcept { return slot_; }

  private:
    friend class Qwen35BatchStateAdapter;
    std::optional<QwenBatchState> prefill_;
    continuous::QwenPagedKvSequence pages_;
    std::int32_t slot_ = -1;
};

class Qwen35BatchStateAdapter {
  public:
    Qwen35BatchStateAdapter(Qwen35CausalLm &model, std::size_t slots, bool paged);
    Qwen35BatchStateAdapter(const Qwen35BatchStateAdapter &) = delete;
    Qwen35BatchStateAdapter &operator=(const Qwen35BatchStateAdapter &) = delete;
    ~Qwen35BatchStateAdapter();

    bool has_moe() const;
    bool has_cached_moe() const;
    static bool has_cached_moe(const Qwen35CausalLm &model);
    std::string incompatibility(const CudaExecutionContext &execution) const;
    bool paged_kv_enabled() const noexcept;
    QwenPagedKvStats paged_kv_stats() const noexcept;
    std::int64_t slot_releases() const noexcept { return slot_releases_; }

    void suspend_decode();
    void prepare_prefill(QwenBatchRequestState &request, std::int64_t offset, std::int64_t tokens);
    void pause_prefill(QwenBatchRequestState &request);
    void activate(QwenBatchRequestState &request);
    void resume_decode(std::int64_t cache_position);
    void discard_prefill(QwenBatchRequestState &request);
    void release(QwenBatchRequestState &request);
    void finish_retire(std::int64_t cache_position);
    void ensure_decode_tokens(const QwenBatchRequestState &request, std::int64_t tokens);
    void prepare_decode(std::int64_t cache_position);
    void finish_decode(std::int64_t cache_position);
    void recover(const std::vector<QwenBatchRequestState *> &requests);

    mfq_tensor_backend::Tensor logits_from_last_hidden(mfq_tensor_backend::Tensor hidden);
    std::vector<const void *> decode_state_addresses();
    QwenBatchState capture_recurrent_slots(const std::vector<std::int32_t> &slots) const;
    void restore_recurrent_slots(const std::vector<std::int32_t> &slots,
                                 const QwenBatchState &state);

  private:
    QwenBatchState take(std::int64_t batch);
    void restore(const QwenBatchState &state, std::int64_t cache_position);
    QwenBatchState make_slot_state(const QwenBatchState &source, std::int64_t slots) const;
    void copy_to_slot(QwenBatchState &slots, const QwenBatchState &source, std::int64_t slot) const;
    void bind_paged_slots();
    void clear_model();

    Qwen35CausalLm &model_;
    std::vector<QwenBatchRequestState *> slots_;
    std::optional<QwenBatchState> suspended_;
    std::int64_t slot_releases_ = 0;
    bool page_table_dirty_ = false;
    std::unique_ptr<continuous::QwenPagedKvArena> paged_kv_;
    std::vector<continuous::QwenPagedKvSequence> paged_slots_;
};

} // namespace mfq::cuda::qwen35
