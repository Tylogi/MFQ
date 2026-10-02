#pragma once

#include "mfq_tensor_backend.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct CudaExecutionContext;

namespace mfq::cuda {
struct Qwen35Model;
template <typename Model>
struct CausalLm;
using Qwen35CausalLm = CausalLm<Qwen35Model>;
}

namespace mfq::cuda::continuous {
class QwenPagedKvArena;
struct QwenPagedKvSequence;
}

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

class Qwen35BatchStateAdapter {
public:
    explicit Qwen35BatchStateAdapter(Qwen35CausalLm& model) : model_(model) {}
    ~Qwen35BatchStateAdapter();

    bool has_moe() const;
    bool has_cached_moe() const;
    static bool has_cached_moe(const Qwen35CausalLm& model);
    std::string incompatibility(const CudaExecutionContext& execution) const;
    void enable_paged_kv(std::int32_t slots);
    bool paged_kv_enabled() const noexcept;
    QwenPagedKvStats paged_kv_stats() const noexcept;
    void ensure_request_tokens(
        continuous::QwenPagedKvSequence& sequence,
        std::int64_t tokens);
    bool ensure_slot_tokens(std::int32_t slot, std::int64_t tokens);
    void bind_paged_requests(
        const std::vector<const continuous::QwenPagedKvSequence*>& sequences);
    void bind_paged_slots();
    void move_paged_to_slot(
        std::int32_t slot,
        continuous::QwenPagedKvSequence& sequence);
    void release_paged(continuous::QwenPagedKvSequence& sequence);
    void release_paged_slot(std::int32_t slot);
    void release_idle_paged_slots();
    void detach_paged_kv();

    QwenBatchState take(std::int64_t batch);
    void restore(
        const std::vector<QwenBatchState>& states,
        std::int64_t cache_position);
    QwenBatchState make_slot_state(
        const QwenBatchState& source,
        std::int64_t slots) const;
    void copy_to_slot(
        QwenBatchState& slots,
        const QwenBatchState& source,
        std::int64_t slot) const;
    mfq_tensor_backend::Tensor logits_from_last_hidden(
        mfq_tensor_backend::Tensor hidden);
    std::vector<const void*> decode_state_addresses();

private:
    Qwen35CausalLm& model_;
    std::unique_ptr<continuous::QwenPagedKvArena> paged_kv_;
    std::vector<continuous::QwenPagedKvSequence> paged_slots_;
};

} // namespace mfq::cuda::qwen35
