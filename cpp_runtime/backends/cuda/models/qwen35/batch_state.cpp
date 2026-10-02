#include "batch_state.h"

#include "causal_lm.h"
#include "linear_attention.h"
#include "paged_kv.h"
#include "cuda_execution.h"

#include <stdexcept>
#include <utility>

namespace mfq::cuda::qwen35 {

using Tensor = mfq_tensor_backend::Tensor;
using LinearBlock = LinearAttentionBlock;
using mfq::cuda::continuous::QwenPagedKvArena;
using mfq::cuda::continuous::QwenPagedKvSequence;

Qwen35BatchStateAdapter::~Qwen35BatchStateAdapter() = default;

void Qwen35BatchStateAdapter::enable_paged_kv(std::int32_t slots) {
    paged_slots_.resize(static_cast<std::size_t>(slots));
    paged_kv_ = std::make_unique<QwenPagedKvArena>(model_, slots);
}

bool Qwen35BatchStateAdapter::paged_kv_enabled() const noexcept {
    return static_cast<bool>(paged_kv_);
}

QwenPagedKvStats Qwen35BatchStateAdapter::paged_kv_stats() const noexcept {
    if (!paged_kv_) return {};
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

void Qwen35BatchStateAdapter::ensure_request_tokens(
        QwenPagedKvSequence& sequence, std::int64_t tokens) {
    if (paged_kv_) paged_kv_->ensure_tokens(sequence, tokens);
}

bool Qwen35BatchStateAdapter::ensure_slot_tokens(
        std::int32_t slot, std::int64_t tokens) {
    if (!paged_kv_) return false;
    MFQ_RUNTIME_CHECK(slot >= 0 &&
        static_cast<std::size_t>(slot) < paged_slots_.size(),
        "continuous batching received an invalid paged-KV slot");
    return paged_kv_->ensure_tokens(
        paged_slots_[static_cast<std::size_t>(slot)], tokens);
}

void Qwen35BatchStateAdapter::bind_paged_requests(
        const std::vector<const QwenPagedKvSequence*>& sequences) {
    if (paged_kv_) paged_kv_->bind(sequences);
}

void Qwen35BatchStateAdapter::bind_paged_slots() {
    if (!paged_kv_) return;
    std::vector<const QwenPagedKvSequence*> sequences;
    sequences.reserve(paged_slots_.size());
    for (auto& sequence : paged_slots_) {
        paged_kv_->ensure_tokens(sequence, 1);
        sequences.push_back(&sequence);
    }
    paged_kv_->bind(sequences);
}

void Qwen35BatchStateAdapter::move_paged_to_slot(
        std::int32_t slot, QwenPagedKvSequence& sequence) {
    if (!paged_kv_) return;
    MFQ_RUNTIME_CHECK(slot >= 0 &&
        static_cast<std::size_t>(slot) < paged_slots_.size(),
        "continuous batching received an invalid paged-KV slot");
    auto& target = paged_slots_[static_cast<std::size_t>(slot)];
    paged_kv_->release(target);
    std::swap(target, sequence);
}

void Qwen35BatchStateAdapter::release_paged(
        QwenPagedKvSequence& sequence) {
    if (paged_kv_) paged_kv_->release(sequence);
}

void Qwen35BatchStateAdapter::release_paged_slot(std::int32_t slot) {
    if (!paged_kv_) return;
    MFQ_RUNTIME_CHECK(slot >= 0 &&
        static_cast<std::size_t>(slot) < paged_slots_.size(),
        "continuous batching received an invalid paged-KV slot");
    paged_kv_->release(paged_slots_[static_cast<std::size_t>(slot)]);
}

void Qwen35BatchStateAdapter::release_idle_paged_slots() {
    if (!paged_kv_) return;
    for (auto& sequence : paged_slots_) paged_kv_->release(sequence);
}

void Qwen35BatchStateAdapter::detach_paged_kv() {
    if (paged_kv_) paged_kv_->detach();
}

static void clear_full_attention_decode_workspaces(FullBlock & block) {
    block.decode_partial_o = Tensor();
    block.decode_partial_m = Tensor();
    block.decode_partial_l = Tensor();
    block.decode_mma_mask = Tensor();
    block.decode_mma_kv_max = Tensor();
    block.decode_mma_meta = Tensor();
}

bool Qwen35BatchStateAdapter::has_moe() const {
    for (const auto & block : model_.blocks) {
        if (const auto * full = dynamic_cast<const FullBlock *>(block.get())) {
            if (full->ffn.is_moe) return true;
        } else if (const auto * linear =
                dynamic_cast<const LinearBlock *>(block.get())) {
            if (linear->ffn.is_moe) return true;
        }
    }
    return false;
}

bool Qwen35BatchStateAdapter::has_cached_moe() const {
    return has_cached_moe(model_);
}

bool Qwen35BatchStateAdapter::has_cached_moe(
        const Qwen35CausalLm& model) {
    for (const auto & block : model.blocks) {
        if (const auto * full = dynamic_cast<const FullBlock *>(block.get())) {
            if (full->ffn.uses_moe_expert_cache()) return true;
        } else if (const auto * linear =
                dynamic_cast<const LinearBlock *>(block.get())) {
            if (linear->ffn.uses_moe_expert_cache()) return true;
        }
    }
    return false;
}

std::string Qwen35BatchStateAdapter::incompatibility(
        const CudaExecutionContext& execution) const {
    if (model_.blocks.empty()) {
        return "continuous batching requires at least one model block";
    }
    if (execution.dense_cpu_layer_count != 0 ||
            !execution.dsv4_cpu_offload_layers.empty()) {
        return "continuous batching requires GPU-resident model blocks";
    }
    for (const auto & block : model_.blocks) {
        if (block->cpu_offloaded) {
            return "continuous batching cannot use CPU-offloaded blocks";
        }
        if (const auto * full = dynamic_cast<const FullBlock *>(block.get())) {
            if (full->sliding) {
                return "continuous batching requires non-sliding Qwen attention";
            }
            if (full->ffn.uses_moe_expert_cache() &&
                    full->ffn.moe_top_k >
                        execution.moe_cache_registration_min_slots) {
                return "continuous batching requires one cached slot per routed expert";
            }
            continue;
        }
        if (const auto * linear = dynamic_cast<const LinearBlock *>(block.get())) {
            if (linear->ffn.uses_moe_expert_cache() &&
                    linear->ffn.moe_top_k >
                        execution.moe_cache_registration_min_slots) {
                return "continuous batching requires one cached slot per routed expert";
            }
            continue;
        }
        return "continuous batching encountered an unsupported Qwen block";
    }
    return {};
}

QwenBatchState Qwen35BatchStateAdapter::take(int64_t batch) {
    auto* paged_kv = paged_kv_.get();
    MFQ_RUNTIME_CHECK(model_.speculative_start < 0,
        "continuous batching cannot detach speculative state");
    QwenBatchState state;
    state.batch = batch;
    state.layers.reserve(model_.blocks.size());
    for (auto & block : model_.blocks) {
        MfqCudaGuard guard(block->cuda_device);
        QwenBatchLayerState layer;
        if (auto * full = dynamic_cast<FullBlock *>(block.get())) {
            layer.kind = QwenBatchLayerState::Kind::FullAttention;
            layer.paged = paged_kv != nullptr;
            if (layer.paged) {
                MFQ_RUNTIME_CHECK(full->cache.is_paged() &&
                    full->cache.batch_size() == batch,
                    "continuous batching paged KV state is unavailable");
            } else {
                MFQ_RUNTIME_CHECK(
                    full->cache.k.defined() && full->cache.v.defined() &&
                    full->cache.k.dim() == 4 &&
                    full->cache.k.size(0) == batch &&
                    full->cache.v.sizes() == full->cache.k.sizes(),
                    "continuous batching full-attention state is unavailable");
                layer.first = full->cache.k;
                layer.second = full->cache.v;
                layer.ring = full->cache.ring;
            }
            full->cache = KVCache();
            clear_full_attention_decode_workspaces(*full);
        } else if (auto * linear = dynamic_cast<LinearBlock *>(block.get())) {
            MFQ_RUNTIME_CHECK(linear->conv_state.defined() &&
                linear->gdn_state.defined() &&
                linear->conv_state.size(0) == batch &&
                linear->gdn_state.size(0) == batch &&
                !linear->speculative_pending,
                "continuous batching recurrent state is unavailable");
            layer.kind = QwenBatchLayerState::Kind::Recurrent;
            layer.first = linear->conv_state;
            layer.second = linear->gdn_state;
            linear->conv_state = Tensor();
            linear->gdn_state = Tensor();
            linear->speculative_conv = Tensor();
            linear->speculative_gdn = Tensor();
            linear->speculative_pending = false;
        } else {
            throw std::runtime_error(
                "continuous batching encountered an unsupported block state");
        }
        state.layers.push_back(std::move(layer));
    }
    model_.cache_pos = 0;
    return state;
}

static Tensor merge_batch_tensors(
        const std::vector<QwenBatchState> & states, size_t layer,
        bool second) {
    std::vector<Tensor> values;
    values.reserve(states.size());
    for (const auto & state : states) {
        values.push_back(second
            ? state.layers[layer].second
            : state.layers[layer].first);
    }
    return values.size() == 1 ? values.front() :
        mfq_tensor_backend::cat(values, 0).contiguous();
}

void Qwen35BatchStateAdapter::restore(
        const std::vector<QwenBatchState>& states,
        int64_t cache_position) {
    auto* paged_kv = paged_kv_.get();
    MFQ_RUNTIME_CHECK(!states.empty(),
        "continuous batching cannot restore an empty state list");
    int64_t batch = 0;
    for (const auto & state : states) {
        MFQ_RUNTIME_CHECK(state.batch > 0 &&
            state.layers.size() == model_.blocks.size(),
            "continuous batching state layout changed");
        batch += state.batch;
    }
    for (size_t layer_index = 0;
            layer_index < model_.blocks.size(); ++layer_index) {
        auto & block = model_.blocks[layer_index];
        MfqCudaGuard guard(block->cuda_device);
        const auto kind = states.front().layers[layer_index].kind;
        for (const auto & state : states) {
            MFQ_RUNTIME_CHECK(state.layers[layer_index].kind == kind,
                "continuous batching mixed incompatible layer states");
        }
        if (auto * full = dynamic_cast<FullBlock *>(block.get())) {
            MFQ_RUNTIME_CHECK(kind ==
                QwenBatchLayerState::Kind::FullAttention,
                "continuous batching full-attention state kind changed");
            if (paged_kv != nullptr) {
                for (const auto & state : states) {
                    MFQ_RUNTIME_CHECK(state.layers[layer_index].paged,
                        "continuous batching lost paged KV state");
                }
                full->cache = KVCache();
                clear_full_attention_decode_workspaces(*full);
                continue;
            }
            const auto shape = states.front().layers[layer_index].first.sizes();
            for (const auto & state : states) {
                const auto & saved = state.layers[layer_index];
                MFQ_RUNTIME_CHECK(!saved.paged && !saved.ring &&
                    saved.first.dim() == 4 &&
                    saved.second.sizes() == saved.first.sizes() &&
                    saved.first.size(1) == shape[1] &&
                    saved.first.size(2) == shape[2] &&
                    saved.first.size(3) == shape[3],
                    "continuous batching KV cache geometry changed");
            }
            full->cache.k = merge_batch_tensors(
                states, layer_index, false);
            full->cache.v = merge_batch_tensors(
                states, layer_index, true);
            full->cache.ring = false;
            MFQ_RUNTIME_CHECK(full->cache.k.size(0) == batch,
                "continuous batching KV merge produced the wrong batch");
            clear_full_attention_decode_workspaces(*full);
        } else if (auto * linear = dynamic_cast<LinearBlock *>(block.get())) {
            MFQ_RUNTIME_CHECK(kind == QwenBatchLayerState::Kind::Recurrent,
                "continuous batching recurrent state kind changed");
            linear->conv_state = merge_batch_tensors(
                states, layer_index, false);
            linear->gdn_state = merge_batch_tensors(
                states, layer_index, true);
            linear->speculative_conv = Tensor();
            linear->speculative_gdn = Tensor();
            linear->speculative_pending = false;
            MFQ_RUNTIME_CHECK(linear->conv_state.size(0) == batch &&
                linear->gdn_state.size(0) == batch,
                "continuous batching recurrent merge produced the wrong batch");
        } else {
            throw std::runtime_error(
                "continuous batching restore encountered an unsupported block");
        }
    }
    model_.cache_pos = cache_position;
    model_.speculative_start = -1;
    model_.speculative_confirmed = 0;
}

QwenBatchState Qwen35BatchStateAdapter::make_slot_state(
        const QwenBatchState& source, int64_t slots) const {
    MFQ_RUNTIME_CHECK(source.batch == 1 && slots > 0,
        "continuous batching slot state requires one source row");
    QwenBatchState result;
    result.batch = slots;
    result.layers.reserve(source.layers.size());
    for (const auto& saved : source.layers) {
        QwenBatchLayerState layer;
        layer.kind = saved.kind;
        layer.ring = saved.ring;
        layer.paged = saved.paged;
        if (saved.first.defined()) {
            auto first_shape = saved.first.sizes().vec();
            auto second_shape = saved.second.sizes().vec();
            first_shape[0] = slots;
            second_shape[0] = slots;
            layer.first = mfq_tensor_backend::zeros(
                first_shape, saved.first.options());
            layer.second = mfq_tensor_backend::zeros(
                second_shape, saved.second.options());
        }
        result.layers.push_back(std::move(layer));
    }
    return result;
}

void Qwen35BatchStateAdapter::copy_to_slot(
        QwenBatchState& slots,
        const QwenBatchState& source,
        int64_t slot) const {
    MFQ_RUNTIME_CHECK(source.batch == 1 && slot >= 0 && slot < slots.batch &&
        slots.layers.size() == source.layers.size(),
        "continuous batching received an invalid stable slot");
    for (size_t layer = 0; layer < slots.layers.size(); ++layer) {
        auto& target = slots.layers[layer];
        const auto& saved = source.layers[layer];
        MFQ_RUNTIME_CHECK(target.kind == saved.kind &&
            target.paged == saved.paged,
            "continuous batching stable slot layout changed");
        if (!target.first.defined()) continue;
        target.first.narrow(0, slot, 1).copy_(saved.first);
        target.second.narrow(0, slot, 1).copy_(saved.second);
    }
}

Tensor Qwen35BatchStateAdapter::logits_from_last_hidden(Tensor hidden) {
    auto last = hidden.index({Slice(), -1, Slice()})
        .to(mfq_tensor_backend::kFloat16).contiguous();
    return model_.apply_final_logit_softcap(
        model_.lm_head.forward(*model_.execution, last));
}

std::vector<const void*> Qwen35BatchStateAdapter::decode_state_addresses() {
    std::vector<const void *> addresses;
    addresses.reserve(2 * model_.blocks.size());
    for (auto & block : model_.blocks) {
        if (auto * full = dynamic_cast<FullBlock *>(block.get())) {
            if (full->cache.is_paged()) {
                addresses.push_back(full->cache.k_chunk_ptrs.data_ptr());
                addresses.push_back(full->cache.v_chunk_ptrs.data_ptr());
                addresses.push_back(full->cache.page_table.data_ptr());
            } else {
                addresses.push_back(full->cache.k.data_ptr());
                addresses.push_back(full->cache.v.data_ptr());
            }
        } else if (auto * linear = dynamic_cast<LinearBlock *>(block.get())) {
            addresses.push_back(linear->conv_state.data_ptr());
            addresses.push_back(linear->gdn_state.data_ptr());
        }
    }
    return addresses;
}

QwenBatchState Qwen35BatchStateAdapter::capture_recurrent_slots(
        const std::vector<std::int32_t>& slots) const {
    QwenBatchState state;
    if (slots.empty()) return state;
    state.layers.resize(model_.blocks.size());
    for (std::size_t i = 0; i < model_.blocks.size(); ++i) {
        auto* linear = dynamic_cast<LinearBlock*>(model_.blocks[i].get());
        if (!linear) continue;
        MfqCudaGuard guard(linear->cuda_device);
        std::vector<Tensor> conv, gdn;
        for (auto slot : slots) {
            conv.push_back(linear->conv_state.narrow(0, slot, 1));
            gdn.push_back(linear->gdn_state.narrow(0, slot, 1));
        }
        state.layers[i].first = mfq_tensor_backend::cat(conv, 0).clone();
        state.layers[i].second = mfq_tensor_backend::cat(gdn, 0).clone();
    }
    return state;
}

void Qwen35BatchStateAdapter::restore_recurrent_slots(
        const std::vector<std::int32_t>& slots, const QwenBatchState& state) {
    if (slots.empty()) return;
    for (std::size_t i = 0; i < model_.blocks.size(); ++i) {
        auto* linear = dynamic_cast<LinearBlock*>(model_.blocks[i].get());
        if (!linear) continue;
        MfqCudaGuard guard(linear->cuda_device);
        for (std::size_t row = 0; row < slots.size(); ++row) {
            linear->conv_state.narrow(0, slots[row], 1).copy_(state.layers[i].first.narrow(0, row, 1));
            linear->gdn_state.narrow(0, slots[row], 1).copy_(state.layers[i].second.narrow(0, row, 1));
        }
    }
}

} // namespace mfq::cuda::qwen35
