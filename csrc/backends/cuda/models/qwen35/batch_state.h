#pragma once

#include "native/tensor/backend.h"
#include "storage/state/batch_paged.h"
#include "common/causal_metadata.h"

#include <exception>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <vector>

struct FullAttentionState;
struct FullBlock;
struct MtpModule;

namespace mfq::models::qwen35 {
template <class Backend> struct CausalLm;
}
namespace mfq::cuda {
struct Qwen35Model;
template <typename Model> struct CudaCausalOps;
using Qwen35CausalLm = mfq::models::qwen35::CausalLm<CudaCausalOps<Qwen35Model>>;
}

namespace mfq::cuda::qwen35 {

class Qwen35GenerationState {
  public:
    Qwen35GenerationState(Qwen35CausalLm&, MtpModule*);
    ~Qwen35GenerationState();
    class Binding {
      public:
        explicit Binding(Qwen35GenerationState& owner) noexcept : owner_(owner) { owner_.swap(); }
        ~Binding() { owner_.swap(); }
        Binding(const Binding&) = delete;
        Binding& operator=(const Binding&) = delete;
      private:
        Qwen35GenerationState& owner_;
    };
    Binding bind() noexcept { return Binding(*this); }
  private:
    void swap() noexcept;
    struct Impl;
    std::unique_ptr<Impl> state_;
};

struct LinearAttentionState;
struct QwenBatchLayerState {
    std::shared_ptr<FullAttentionState> full;
    std::shared_ptr<LinearAttentionState> recurrent;
};
struct QwenBatchState {
    int64_t batch = 0;
    std::vector<QwenBatchLayerState> layers;
    mfq::models::CausalState causal;
    static QwenBatchState empty(const Qwen35CausalLm& model);
    void swap(Qwen35CausalLm& model) noexcept;
};

class QwenBatchRequestState {
  public:
    QwenBatchRequestState() = default;
    QwenBatchRequestState(const QwenBatchRequestState&) = delete;
    QwenBatchRequestState& operator=(const QwenBatchRequestState&) = delete;
  private:
    friend class Qwen35BatchStorage;
    std::optional<QwenBatchState> prefill_;
    PagedKvSequence pages_;
    int32_t slot_ = -1;
    int64_t position_ = 0;
};

// Native Qwen ownership only. Scoped bindings hide suspend/restore, packing and
// global cache-position bookkeeping. No sampler, logits, Graph policy or events.
class Qwen35BatchStorage {
  public:
    using DecodeRow = std::pair<QwenBatchRequestState*, int64_t>;
    Qwen35BatchStorage(Qwen35CausalLm& model, size_t slots, bool paged);
    Qwen35BatchStorage(const Qwen35BatchStorage&) = delete;
    Qwen35BatchStorage& operator=(const Qwen35BatchStorage&) = delete;
    ~Qwen35BatchStorage();

    class PrefillBinding {
      public:
        PrefillBinding(Qwen35BatchStorage&, QwenBatchRequestState&, int64_t offset,
                       int64_t count, int64_t tokens);
        ~PrefillBinding();
        PrefillBinding(const PrefillBinding&) = delete;
        PrefillBinding& operator=(const PrefillBinding&) = delete;
        void commit();
      private:
        Qwen35BatchStorage& owner_;
        QwenBatchRequestState& request_;
        int64_t end_, tokens_;
        bool finished_ = false;
    };
    class DecodeBinding {
      public:
        DecodeBinding(Qwen35BatchStorage&, std::span<const DecodeRow>, int64_t batch);
        ~DecodeBinding();
        DecodeBinding(const DecodeBinding&) = delete;
        DecodeBinding& operator=(const DecodeBinding&) = delete;
        void commit();
        void abort(); // Preparation-only: restore binding without advancing rows.
      private:
        Qwen35BatchStorage& owner_;
        std::span<const DecodeRow> rows_;
        bool finished_ = false;
    };
    PrefillBinding prefill(QwenBatchRequestState& request, int64_t offset, int64_t count, int64_t tokens) {
        return PrefillBinding(*this, request, offset, count, tokens);
    }
    DecodeBinding decode(std::span<const DecodeRow> rows, int64_t batch) {
        return DecodeBinding(*this, rows, batch);
    }
    void release(QwenBatchRequestState&);
    void recover(std::span<QwenBatchRequestState* const>);
    BatchStorageStats stats() const noexcept;

  private:
    void check_binding_failure() const;
    int64_t cache_position() const;
    void suspend_decode();
    void prepare_prefill(QwenBatchRequestState&, int64_t offset, int64_t tokens);
    void pause_prefill(QwenBatchRequestState&);
    void activate(QwenBatchRequestState&);
    void resume_decode();
    void discard_prefill(QwenBatchRequestState&);
    void release_state(QwenBatchRequestState&);
    void finish_retire();
    void ensure_decode_tokens(const QwenBatchRequestState&, int64_t tokens);
    void prepare_decode(const std::vector<int32_t>& slots, int64_t batch, int64_t position);
    void finish_decode();
    QwenBatchState take(int64_t batch);
    void restore(const QwenBatchState&, int64_t position);
    QwenBatchState make_slot_state(const QwenBatchState&, int64_t slots) const;
    void copy_to_slot(QwenBatchState& slots, const QwenBatchState& source, int64_t slot) const;
    void bind_pages(std::span<const PagedKvSequence* const>);
    void bind_paged_slots();
    void clear_model();

    Qwen35CausalLm& model_;
    std::vector<QwenBatchRequestState*> slots_;
    std::optional<QwenBatchState> suspended_, unpacked_;
    std::map<int64_t, QwenBatchState> decode_buckets_;
    int64_t decode_batch_ = 0;
    std::vector<int32_t> decode_slots_;
    std::vector<PagedKvSequence> padding_pages_;
    int64_t slot_releases_ = 0;
    bool page_table_dirty_ = false;
    std::unique_ptr<BatchPagedKvPool> paged_kv_;
    std::vector<PagedKvSequence> paged_slots_;
    std::vector<FullBlock*> paged_layers_;
    std::exception_ptr binding_failure_;
};
} // namespace mfq::cuda::qwen35
