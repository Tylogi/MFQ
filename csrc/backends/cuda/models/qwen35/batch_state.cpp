#include "batch_state.h"

#include "../../ops/cuda_execution.h"
#include "linear_attention.h"
#include "ops.h"
#include "paged_kv.h"

#include <limits>
#include <stdexcept>
#include <utility>

namespace mfq::cuda::qwen35 {

using Tensor = mfq_tensor_backend::Tensor;
using LinearBlock = LinearAttentionBlock;
using mfq::cuda::continuous::QwenPagedKvArena;
using mfq::cuda::continuous::QwenPagedKvSequence;

Qwen35BatchStateAdapter::~Qwen35BatchStateAdapter() = default;

Qwen35BatchStateAdapter::Qwen35BatchStateAdapter(Qwen35CausalLm &model, std::size_t slots,
                                                 bool paged)
    : model_(model) {
    if (slots == 0 || slots > static_cast<std::size_t>(std::numeric_limits<int32_t>::max()))
        throw std::invalid_argument("continuous batching max sequences must be positive int32");
    const auto reason = incompatibility(*model.execution);
    if (!reason.empty())
        throw std::runtime_error(reason);
    slots_.resize(slots);
    if (paged) {
        paged_slots_.resize(slots);
        paged_kv_ = std::make_unique<QwenPagedKvArena>(model_, static_cast<int32_t>(slots));
    }
}

bool Qwen35BatchStateAdapter::paged_kv_enabled() const noexcept {
    return static_cast<bool>(paged_kv_);
}

QwenPagedKvStats Qwen35BatchStateAdapter::paged_kv_stats() const noexcept {
    if (!paged_kv_)
        return {};
    return {
        paged_kv_->page_size(),
        static_cast<std::int64_t>(paged_kv_->live_pages()),
        static_cast<std::int64_t>(paged_kv_->peak_live_pages()),
        static_cast<std::int64_t>(paged_kv_->capacity_pages()),
        static_cast<std::int64_t>(paged_kv_->reserved_bytes()),
        static_cast<std::int64_t>(paged_kv_->allocation_count()),
        static_cast<std::int64_t>(paged_kv_->reuse_count()),
        static_cast<std::int64_t>(paged_kv_->release_count()),
        static_cast<std::int64_t>(paged_kv_->page_table_updates()),
    };
}

void Qwen35BatchStateAdapter::suspend_decode() {
    MFQ_RUNTIME_CHECK(!suspended_, "continuous batching decode state is already suspended");
    if (std::any_of(slots_.begin(), slots_.end(), [](auto *owner) { return owner != nullptr; }))
        suspended_ = take(static_cast<int64_t>(slots_.size()));
}

void Qwen35BatchStateAdapter::prepare_prefill(QwenBatchRequestState &request, int64_t offset,
                                              int64_t tokens) {
    MFQ_RUNTIME_CHECK(request.slot_ < 0 && offset >= 0 && offset < tokens,
                      "continuous batching prefill state is invalid");
    if (request.prefill_) {
        restore(*request.prefill_, offset);
        request.prefill_.reset();
    } else {
        MFQ_RUNTIME_CHECK(offset == 0, "continuous batching lost prefill state");
        model_.reset(1);
        if (paged_kv_)
            paged_kv_->ensure_tokens(request.pages_, tokens);
    }
    if (paged_kv_)
        paged_kv_->bind({&request.pages_});
}

void Qwen35BatchStateAdapter::pause_prefill(QwenBatchRequestState &request) {
    request.prefill_ = take(1);
    if (paged_kv_)
        paged_kv_->detach();
}

void Qwen35BatchStateAdapter::activate(QwenBatchRequestState &request) {
    MFQ_RUNTIME_CHECK(request.slot_ < 0, "continuous batching request already has a slot");
    const auto free = std::find(slots_.begin(), slots_.end(), nullptr);
    MFQ_RUNTIME_CHECK(free != slots_.end(), "continuous batching has no free stable slot");
    const auto slot = static_cast<int32_t>(free - slots_.begin());
    auto admitted = take(1);
    if (!suspended_)
        suspended_ = make_slot_state(admitted, slots_.size());
    copy_to_slot(*suspended_, admitted, slot);
    if (paged_kv_) {
        auto &target = paged_slots_[slot];
        paged_kv_->release(target);
        std::swap(target, request.pages_);
    }
    *free = &request;
    request.slot_ = slot;
}

void Qwen35BatchStateAdapter::resume_decode(int64_t cache_position) {
    if (suspended_) {
        restore(*suspended_, cache_position);
        bind_paged_slots();
        suspended_.reset();
    } else {
        clear_model();
    }
}

void Qwen35BatchStateAdapter::release(QwenBatchRequestState &request) {
    if (request.slot_ >= 0) {
        const auto slot = static_cast<std::size_t>(request.slot_);
        MFQ_RUNTIME_CHECK(slot < slots_.size() && slots_[slot] == &request,
                          "continuous batching stable slot ownership changed");
        if (paged_kv_)
            paged_kv_->release(paged_slots_[slot]);
        slots_[slot] = nullptr;
        request.slot_ = -1;
        ++slot_releases_;
    }
    if (paged_kv_)
        paged_kv_->release(request.pages_);
    request.prefill_.reset();
}

void Qwen35BatchStateAdapter::discard_prefill(QwenBatchRequestState &request) {
    mfq_cuda_synchronize();
    release(request);
    clear_model();
}

void Qwen35BatchStateAdapter::clear_model() {
    model_.reset(1);
    if (paged_kv_)
        paged_kv_->detach();
}

void Qwen35BatchStateAdapter::finish_retire(int64_t cache_position) {
    if (std::none_of(slots_.begin(), slots_.end(), [](auto *owner) { return owner != nullptr; })) {
        clear_model();
        if (paged_kv_)
            for (auto &sequence : paged_slots_)
                paged_kv_->release(sequence);
    } else {
        model_.cache_pos = cache_position;
        bind_paged_slots();
    }
}

void Qwen35BatchStateAdapter::ensure_decode_tokens(const QwenBatchRequestState &request,
                                                   int64_t tokens) {
    MFQ_RUNTIME_CHECK(request.slot_ >= 0 &&
                          static_cast<std::size_t>(request.slot_) < slots_.size() &&
                          slots_[request.slot_] == &request,
                      "continuous batching request lost its stable slot");
    if (paged_kv_)
        page_table_dirty_ =
            paged_kv_->ensure_tokens(paged_slots_[request.slot_], tokens) || page_table_dirty_;
}

void Qwen35BatchStateAdapter::prepare_decode(int64_t cache_position) {
    if (page_table_dirty_)
        bind_paged_slots();
    model_.cache_pos = cache_position;
}

void Qwen35BatchStateAdapter::finish_decode(int64_t cache_position) {
    model_.cache_pos = cache_position;
}

void Qwen35BatchStateAdapter::recover(const std::vector<QwenBatchRequestState *> &requests) {
    try {
        mfq_cuda_synchronize();
    } catch (...) {
    }
    for (auto *request : requests)
        release(*request);
    suspended_.reset();
    finish_retire(0);
}

void Qwen35BatchStateAdapter::bind_paged_slots() {
    if (!paged_kv_)
        return;
    std::vector<const QwenPagedKvSequence *> sequences;
    sequences.reserve(paged_slots_.size());
    for (auto &sequence : paged_slots_) {
        paged_kv_->ensure_tokens(sequence, 1);
        sequences.push_back(&sequence);
    }
    paged_kv_->bind(sequences);
    page_table_dirty_ = false;
}

bool Qwen35BatchStateAdapter::has_moe() const {
    for (const auto &block : model_.blocks) {
        if (const auto *full = dynamic_cast<const FullBlock *>(block.get())) {
            if (full->ffn.is_moe)
                return true;
        } else if (const auto *linear = dynamic_cast<const LinearBlock *>(block.get())) {
            if (linear->ffn.is_moe)
                return true;
        }
    }
    return false;
}

bool Qwen35BatchStateAdapter::has_cached_moe() const { return has_cached_moe(model_); }

bool Qwen35BatchStateAdapter::has_cached_moe(const Qwen35CausalLm &model) {
    for (const auto &block : model.blocks) {
        if (const auto *full = dynamic_cast<const FullBlock *>(block.get())) {
            if (full->ffn.uses_moe_expert_cache())
                return true;
        } else if (const auto *linear = dynamic_cast<const LinearBlock *>(block.get())) {
            if (linear->ffn.uses_moe_expert_cache())
                return true;
        }
    }
    return false;
}

std::string Qwen35BatchStateAdapter::incompatibility(const CudaExecutionContext &execution) const {
    if (model_.blocks.empty()) {
        return "continuous batching requires at least one model block";
    }
    if (execution.dense_cpu_layer_count != 0 || !execution.dsv4_cpu_offload_layers.empty()) {
        return "continuous batching requires GPU-resident model blocks";
    }
    for (const auto &block : model_.blocks) {
        if (block->cpu_offloaded) {
            return "continuous batching cannot use CPU-offloaded blocks";
        }
        if (const auto *full = dynamic_cast<const FullBlock *>(block.get())) {
            if (full->sliding) {
                return "continuous batching requires non-sliding Qwen attention";
            }
            if (full->ffn.uses_moe_expert_cache() &&
                full->ffn.moe_top_k > execution.moe_cache_registration_min_slots) {
                return "continuous batching requires one cached slot per routed expert";
            }
            continue;
        }
        if (const auto *linear = dynamic_cast<const LinearBlock *>(block.get())) {
            if (linear->ffn.uses_moe_expert_cache() &&
                linear->ffn.moe_top_k > execution.moe_cache_registration_min_slots) {
                return "continuous batching requires one cached slot per routed expert";
            }
            continue;
        }
        return "continuous batching encountered an unsupported Qwen block";
    }
    return {};
}

QwenBatchState Qwen35BatchStateAdapter::take(int64_t batch) {
    MFQ_RUNTIME_CHECK(model_.speculative_start < 0, "cannot unbind speculative state");
    QwenBatchState result;
    result.batch = batch;
    result.layers.reserve(model_.blocks.size());
    for (auto& block : model_.blocks) {
        QwenBatchLayerState layer;
        if (auto* full = dynamic_cast<FullBlock*>(block.get())) {
            layer.full = std::exchange(full->state, std::make_shared<FullAttentionState>());
        } else if (auto* linear = dynamic_cast<LinearBlock*>(block.get())) {
            MFQ_RUNTIME_CHECK(!linear->state->speculative_pending, "cannot unbind speculative state");
            layer.recurrent = std::exchange(linear->state, std::make_shared<LinearAttentionState>());
        } else throw std::runtime_error("unsupported Qwen block state");
        result.layers.push_back(std::move(layer));
    }
    model_.cache_pos = 0;
    return result;
}

void Qwen35BatchStateAdapter::restore(const QwenBatchState& state, int64_t cache_position) {
    MFQ_RUNTIME_CHECK(state.batch > 0 && state.layers.size() == model_.blocks.size(),
        "Qwen state layout changed");
    for (size_t i = 0; i < model_.blocks.size(); ++i) {
        auto& block = model_.blocks[i];
        const auto& saved = state.layers[i];
        if (auto* full = dynamic_cast<FullBlock*>(block.get())) {
            MFQ_RUNTIME_CHECK(saved.full && !saved.recurrent, "full attention state changed");
            full->state = saved.full;
        } else if (auto* linear = dynamic_cast<LinearBlock*>(block.get())) {
            MFQ_RUNTIME_CHECK(saved.recurrent && !saved.full, "recurrent state changed");
            linear->state = saved.recurrent;
        } else throw std::runtime_error("unsupported Qwen block state");
    }
    model_.cache_pos = cache_position;
    model_.speculative_start = -1;
    model_.speculative_confirmed = 0;
}

static std::pair<Tensor, Tensor> tensors(const QwenBatchLayerState& layer) {
    if (layer.full) return {layer.full->cache.k, layer.full->cache.v};
    if (layer.recurrent) return {layer.recurrent->conv_state, layer.recurrent->gdn_state};
    return {};
}

QwenBatchState Qwen35BatchStateAdapter::make_slot_state(const QwenBatchState& source,
                                                       int64_t slots) const {
    MFQ_RUNTIME_CHECK(source.batch == 1 && slots > 0, "slot state requires one source row");
    QwenBatchState result;
    result.batch = slots;
    for (const auto& saved : source.layers) {
        QwenBatchLayerState layer;
        const auto expand = [slots](const Tensor& tensor) {
            if (!tensor.defined()) return Tensor{};
            auto shape = tensor.sizes().vec(); shape[0] = slots;
            return mfq_tensor_backend::zeros(shape, tensor.options());
        };
        if (saved.full) {
            layer.full = std::make_shared<FullAttentionState>();
            layer.full->cache.k = expand(saved.full->cache.k);
            layer.full->cache.v = expand(saved.full->cache.v);
        } else {
            layer.recurrent = std::make_shared<LinearAttentionState>();
            layer.recurrent->conv_state = expand(saved.recurrent->conv_state);
            layer.recurrent->gdn_state = expand(saved.recurrent->gdn_state);
        }
        result.layers.push_back(std::move(layer));
    }
    return result;
}

void Qwen35BatchStateAdapter::copy_to_slot(QwenBatchState& slots, const QwenBatchState& source,
                                          int64_t slot) const {
    MFQ_RUNTIME_CHECK(source.batch == 1 && slot >= 0 && slot < slots.batch &&
        slots.layers.size() == source.layers.size(), "invalid stable slot");
    for (size_t i = 0; i < slots.layers.size(); ++i) {
        MFQ_RUNTIME_CHECK(bool(slots.layers[i].full) == bool(source.layers[i].full), "slot layout changed");
        auto [first, second] = tensors(slots.layers[i]);
        auto [saved_first, saved_second] = tensors(source.layers[i]);
        if (!first.defined()) continue;
        first.narrow(0, slot, 1).copy_(saved_first);
        second.narrow(0, slot, 1).copy_(saved_second);
    }
}

Tensor Qwen35BatchStateAdapter::logits_from_last_hidden(Tensor hidden) {
    auto last = hidden.index({Slice(), -1, Slice()}).to(mfq_tensor_backend::kFloat16).contiguous();
    return model_.apply_final_logit_softcap(model_.lm_head.forward(*model_.execution, last));
}

std::vector<const void *> Qwen35BatchStateAdapter::decode_state_addresses() {
    std::vector<const void *> addresses;
    addresses.reserve(2 * model_.blocks.size());
    for (auto &block : model_.blocks) {
        if (auto *full = dynamic_cast<FullBlock *>(block.get())) {
            if (full->state->cache.is_paged()) {
                addresses.push_back(full->state->cache.k_chunk_ptrs.data_ptr());
                addresses.push_back(full->state->cache.v_chunk_ptrs.data_ptr());
                addresses.push_back(full->state->cache.page_table.data_ptr());
            } else {
                addresses.push_back(full->state->cache.k.data_ptr());
                addresses.push_back(full->state->cache.v.data_ptr());
            }
            for (const auto* workspace : {&full->state->decode_partial_o, &full->state->decode_partial_m,
                    &full->state->decode_partial_l, &full->state->decode_mma_mask,
                    &full->state->decode_mma_kv_max, &full->state->decode_mma_meta})
                addresses.push_back(workspace->defined() ? workspace->data_ptr() : nullptr);
        } else if (auto *linear = dynamic_cast<LinearBlock *>(block.get())) {
            addresses.push_back(linear->state->conv_state.data_ptr());
            addresses.push_back(linear->state->gdn_state.data_ptr());
        }
    }
    return addresses;
}

QwenBatchState
Qwen35BatchStateAdapter::capture_recurrent_slots(const std::vector<std::int32_t> &slots) const {
    QwenBatchState state;
    if (slots.empty())
        return state;
    state.layers.resize(model_.blocks.size());
    for (std::size_t i = 0; i < model_.blocks.size(); ++i) {
        auto *linear = dynamic_cast<LinearBlock *>(model_.blocks[i].get());
        if (!linear)
            continue;
        MfqCudaGuard guard(linear->cuda_device);
        std::vector<Tensor> conv, gdn;
        for (auto slot : slots) {
            conv.push_back(linear->state->conv_state.narrow(0, slot, 1));
            gdn.push_back(linear->state->gdn_state.narrow(0, slot, 1));
        }
        state.layers[i].recurrent = std::make_shared<LinearAttentionState>();
        state.layers[i].recurrent->conv_state = mfq_tensor_backend::cat(conv, 0).clone();
        state.layers[i].recurrent->gdn_state = mfq_tensor_backend::cat(gdn, 0).clone();
    }
    return state;
}

void Qwen35BatchStateAdapter::restore_recurrent_slots(const std::vector<std::int32_t> &slots,
                                                      const QwenBatchState &state) {
    if (slots.empty())
        return;
    for (std::size_t i = 0; i < model_.blocks.size(); ++i) {
        auto *linear = dynamic_cast<LinearBlock *>(model_.blocks[i].get());
        if (!linear)
            continue;
        MfqCudaGuard guard(linear->cuda_device);
        for (std::size_t row = 0; row < slots.size(); ++row) {
            linear->state->conv_state.narrow(0, slots[row], 1)
                .copy_(state.layers[i].recurrent->conv_state.narrow(0, row, 1));
            linear->state->gdn_state.narrow(0, slots[row], 1)
                .copy_(state.layers[i].recurrent->gdn_state.narrow(0, row, 1));
        }
    }
}

} // namespace mfq::cuda::qwen35
