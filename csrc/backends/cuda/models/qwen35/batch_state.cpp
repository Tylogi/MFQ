#include "batch_state.h"
#include "ops/execution/execution_options.h"
#include "ops/execution/cuda_execution.h"
#include "linear_attention.h"
#include "ops.h"
#include "mtp.h"

#include <limits>
#include <stdexcept>
#include <utility>

namespace mfq::cuda::qwen35 {
using Tensor = mfq_tensor_backend::Tensor;
using LinearBlock = LinearAttentionBlock;

struct Qwen35GenerationState::Impl {
    Qwen35CausalLm& model;
    QwenBatchState native;
    Qwen35Mtp* predictor;
    std::optional<Qwen35Mtp::RequestState> predictor_state;
    Impl(Qwen35CausalLm& model, MtpModule* mtp)
        : model(model), native(QwenBatchState::empty(model)),
          predictor(dynamic_cast<Qwen35Mtp*>(mtp)) {
        if (mtp && !predictor) throw std::invalid_argument("Qwen generation requires a Qwen predictor");
        if (predictor) predictor_state.emplace(*predictor);
    }
};
Qwen35GenerationState::Qwen35GenerationState(Qwen35CausalLm& model, MtpModule* mtp)
    : state_(std::make_unique<Impl>(model, mtp)) {}
Qwen35GenerationState::~Qwen35GenerationState() = default;
void Qwen35GenerationState::swap() noexcept {
    state_->native.swap(state_->model);
    if (state_->predictor) state_->predictor_state->swap(*state_->predictor);
}

QwenBatchState QwenBatchState::empty(const Qwen35CausalLm& model) {
    QwenBatchState result;
    result.batch = 1;
    for (const auto& block : model.blocks) {
        QwenBatchLayerState layer;
        if (dynamic_cast<const FullBlock*>(block.get())) layer.full = std::make_shared<FullAttentionState>();
        else if (dynamic_cast<const LinearBlock*>(block.get())) layer.recurrent = std::make_shared<LinearAttentionState>();
        else throw std::runtime_error("unsupported Qwen request state");
        result.layers.push_back(std::move(layer));
    }
    return result;
}
void QwenBatchState::swap(Qwen35CausalLm& model) noexcept {
    for (size_t i = 0; i < layers.size(); ++i) {
        if (layers[i].full) layers[i].full.swap(static_cast<FullBlock*>(model.blocks[i].get())->state);
        else layers[i].recurrent.swap(static_cast<LinearBlock*>(model.blocks[i].get())->state);
    }
    std::swap(causal, static_cast<mfq::models::CausalState&>(model));
}

Qwen35BatchStorage::~Qwen35BatchStorage() = default;
Qwen35BatchStorage::Qwen35BatchStorage(Qwen35CausalLm& model, size_t slots, bool paged)
    : model_(model) {
    if (!slots || slots > size_t(INT32_MAX)) throw std::invalid_argument("invalid native batch capacity");
    MFQ_RUNTIME_CHECK(!model.blocks.empty(), "continuous batching requires model blocks");
    MFQ_RUNTIME_CHECK(!model.execution->dense_cpu_layer_count && model.execution->dsv4_cpu_offload_layers.empty(),
                      "continuous batching requires GPU-resident model blocks");
    for (const auto& block : model.blocks) {
        MFQ_RUNTIME_CHECK(!block->cpu_offloaded, "continuous batching cannot use CPU-offloaded blocks");
        const FFN* ffn = nullptr;
        if (auto* full = dynamic_cast<FullBlock*>(block.get())) {
            MFQ_RUNTIME_CHECK(!full->sliding, "continuous batching requires non-sliding Qwen attention");
            ffn = &full->ffn;
        } else if (auto* linear = dynamic_cast<LinearBlock*>(block.get())) ffn = &linear->ffn;
        else throw std::runtime_error("continuous batching encountered an unsupported Qwen block");
        MFQ_RUNTIME_CHECK(!ffn->uses_moe_expert_cache() ||
            ffn->moe_top_k <= model.execution->moe_cache_registration_min_slots,
            "continuous batching requires one cached slot per routed expert");
    }
    slots_.resize(slots);
    if (paged) {
        paged_slots_.resize(slots);
        std::vector<PagedKvLayerGeometry> geometry;
        for (const auto& block : model.blocks) if (auto* full = dynamic_cast<FullBlock*>(block.get())) {
            paged_layers_.push_back(full);
            geometry.push_back({full->cuda_device,
                full->kv_heads > 0 ? full->kv_heads : model.num_key_value_heads(),
                full->attention_head_dim > 0 ? full->attention_head_dim : model.head_dim(),
                mfq_tensor_backend::kFloat16,full->kv_quantization});
        }
        paged_kv_ = std::make_unique<BatchPagedKvPool>(geometry, slots, model.max_position_embeddings());
        if(runtime_options::kv_vram_preallocated()) {
            for(auto* layer:paged_layers_)layer->state->cache={};
            paged_kv_->preallocate_context();
        }
    }
}
BatchStorageStats Qwen35BatchStorage::stats() const noexcept {
    return {int64_t(std::count_if(slots_.begin(), slots_.end(), [](auto* request) { return request != nullptr; })),
            slot_releases_, paged_kv_ ? std::optional{paged_kv_->stats()} : std::nullopt};
}
int64_t Qwen35BatchStorage::cache_position() const {
    int64_t position = 0;
    for (auto* request : slots_) if (request) position = std::max(position, request->position_);
    return position;
}
void Qwen35BatchStorage::check_binding_failure() const {
    if (binding_failure_) std::rethrow_exception(binding_failure_);
}

Qwen35BatchStorage::PrefillBinding::PrefillBinding(Qwen35BatchStorage& owner,
        QwenBatchRequestState& request, int64_t offset, int64_t count, int64_t tokens)
    : owner_(owner), request_(request), end_(offset), tokens_(tokens) {
    owner.check_binding_failure();
    MFQ_RUNTIME_CHECK(count > 0 && offset >= 0 && offset <= tokens - count && request.position_ == offset,
                      "invalid native prefill chunk");
    end_ += count;
    owner.suspend_decode();
    owner.prepare_prefill(request, offset, tokens);
}
void Qwen35BatchStorage::PrefillBinding::commit() {
    MFQ_RUNTIME_CHECK(!std::exchange(finished_, true), "prefill binding already finished");
    request_.position_ = end_;
    if (end_ == tokens_) owner_.activate(request_);
    else owner_.pause_prefill(request_);
    owner_.resume_decode();
}
Qwen35BatchStorage::PrefillBinding::~PrefillBinding() {
    if (std::exchange(finished_, true)) return;
    try { owner_.discard_prefill(request_); owner_.resume_decode(); }
    catch (...) { owner_.binding_failure_ = std::current_exception(); }
}

Qwen35BatchStorage::DecodeBinding::DecodeBinding(Qwen35BatchStorage& owner,
        std::span<const DecodeRow> rows, int64_t batch) : owner_(owner), rows_(rows) {
    owner.check_binding_failure();
    std::vector<int32_t> slots;
    int64_t position = 0;
    for (const auto& [request, length] : rows) {
        MFQ_RUNTIME_CHECK(request && request->position_ == length, "native decode position disagrees with request");
        owner.ensure_decode_tokens(*request, length + 1);
        slots.push_back(request->slot_);
        position = std::max(position, length);
    }
    owner.prepare_decode(slots, batch, position);
}
void Qwen35BatchStorage::DecodeBinding::commit() {
    MFQ_RUNTIME_CHECK(!std::exchange(finished_, true), "decode binding already finished");
    for (const auto& [request, length] : rows_) request->position_ = length + 1;
    owner_.finish_decode();
}
void Qwen35BatchStorage::DecodeBinding::abort() {
    if (!std::exchange(finished_, true)) owner_.finish_decode();
}
Qwen35BatchStorage::DecodeBinding::~DecodeBinding() {
    try { abort(); } catch (...) { owner_.binding_failure_ = std::current_exception(); }
}

void Qwen35BatchStorage::suspend_decode() {
    MFQ_RUNTIME_CHECK(!suspended_, "continuous batching decode state is already suspended");
    if (std::any_of(slots_.begin(), slots_.end(), [](auto* owner) { return owner != nullptr; }))
        suspended_ = take(slots_.size());
}
void Qwen35BatchStorage::prepare_prefill(QwenBatchRequestState& request, int64_t offset, int64_t tokens) {
    MFQ_RUNTIME_CHECK(request.slot_ < 0 && offset >= 0 && offset < tokens, "invalid native prefill state");
    if (request.prefill_) {
        restore(*request.prefill_, offset);
        request.prefill_.reset();
    } else {
        MFQ_RUNTIME_CHECK(offset == 0, "continuous batching lost prefill state");
        model_.reset(1);
        if (paged_kv_) paged_kv_->ensure_tokens(request.pages_, tokens);
    }
    if (paged_kv_) {
        const PagedKvSequence* sequence = &request.pages_;
        bind_pages({&sequence, 1});
    }
}
void Qwen35BatchStorage::pause_prefill(QwenBatchRequestState& request) {
    request.prefill_ = take(1);
    if (paged_kv_) for (auto* layer : paged_layers_) layer->state->cache = KVCache();
}
void Qwen35BatchStorage::activate(QwenBatchRequestState& request) {
    MFQ_RUNTIME_CHECK(request.slot_ < 0, "continuous batching request already has a slot");
    const auto free = std::find(slots_.begin(), slots_.end(), nullptr);
    MFQ_RUNTIME_CHECK(free != slots_.end(), "continuous batching has no free stable slot");
    const auto slot = int32_t(free - slots_.begin());
    auto admitted = take(1);
    if (!suspended_) suspended_ = make_slot_state(admitted, slots_.size());
    copy_to_slot(*suspended_, admitted, slot);
    if (paged_kv_) {
        paged_kv_->release(paged_slots_[slot]);
        std::swap(paged_slots_[slot], request.pages_);
    }
    *free = &request;
    request.slot_ = slot;
}
void Qwen35BatchStorage::resume_decode() {
    if (suspended_) {
        restore(*suspended_, cache_position());
        bind_paged_slots();
        suspended_.reset();
    } else clear_model();
}
void Qwen35BatchStorage::release_state(QwenBatchRequestState& request) {
    if (request.slot_ >= 0) {
        const auto slot = size_t(request.slot_);
        MFQ_RUNTIME_CHECK(slot < slots_.size() && slots_[slot] == &request, "native slot ownership changed");
        if (paged_kv_) paged_kv_->release(paged_slots_[slot]);
        slots_[slot] = nullptr;
        request.slot_ = -1;
        ++slot_releases_;
    }
    if (paged_kv_) paged_kv_->release(request.pages_);
    request.prefill_.reset();
    request.position_ = 0;
}
void Qwen35BatchStorage::release(QwenBatchRequestState& request) {
    check_binding_failure();
    if (request.slot_ >= 0) { release_state(request); finish_retire(); }
    else { suspend_decode(); discard_prefill(request); resume_decode(); }
}
void Qwen35BatchStorage::discard_prefill(QwenBatchRequestState& request) {
    mfq_cuda_synchronize();
    release_state(request);
    clear_model();
}
void Qwen35BatchStorage::clear_model() {
    model_.reset(1);
    if (paged_kv_) for (auto* layer : paged_layers_) layer->state->cache = KVCache();
}
void Qwen35BatchStorage::finish_retire() {
    if (std::none_of(slots_.begin(), slots_.end(), [](auto* owner) { return owner != nullptr; })) {
        decode_buckets_.clear();
        clear_model();
        if (paged_kv_) for (auto& sequence : paged_slots_) paged_kv_->release(sequence);
    } else {
        model_.cache_pos = cache_position();
        bind_paged_slots();
    }
}
void Qwen35BatchStorage::ensure_decode_tokens(const QwenBatchRequestState& request, int64_t tokens) {
    MFQ_RUNTIME_CHECK(request.slot_ >= 0 && size_t(request.slot_) < slots_.size() &&
                      slots_[request.slot_] == &request, "request lost its native slot");
    if (paged_kv_) page_table_dirty_ = paged_kv_->ensure_tokens(paged_slots_[request.slot_], tokens) || page_table_dirty_;
}
static std::pair<Tensor, Tensor> tensors(const QwenBatchLayerState& layer) {
    if (layer.full) return {layer.full->cache.k, layer.full->cache.v};
    if (layer.recurrent) return {layer.recurrent->conv_state, layer.recurrent->gdn_state};
    return {};
}
void Qwen35BatchStorage::prepare_decode(const std::vector<int32_t>& slots, int64_t batch, int64_t position) {
    MFQ_RUNTIME_CHECK(!slots.empty() && slots.size() <= batch && batch <= slots_.size() && !unpacked_,
                      "invalid physical decode rows");
    bool direct = slots.size() == slots_.size();
    for (size_t row = 0; row < slots.size(); ++row) direct &= slots[row] == row;
    if (direct) {
        if (page_table_dirty_) bind_paged_slots();
        model_.cache_pos = position;
        return;
    }
    unpacked_ = take(slots_.size());
    decode_batch_ = batch;
    decode_slots_ = slots;
    auto [found, inserted] = decode_buckets_.try_emplace(batch);
    if (inserted) found->second = make_slot_state(*unpacked_, batch);
    auto& packed = found->second;
    for (size_t layer = 0; layer < packed.layers.size(); ++layer) {
        auto [source1, source2] = tensors(unpacked_->layers[layer]);
        auto [target1, target2] = tensors(packed.layers[layer]);
        if (!target1.defined()) continue;
        for (size_t row = 0; row < slots.size(); ++row) {
            target1.narrow(0, row, 1).copy_(source1.narrow(0, slots[row], 1));
            target2.narrow(0, row, 1).copy_(source2.narrow(0, slots[row], 1));
        }
    }
    restore(packed, position);
    if (paged_kv_) {
        std::vector<const PagedKvSequence*> pages;
        for (auto slot : slots) pages.push_back(&paged_slots_[slot]);
        padding_pages_.resize(batch - slots.size());
        for (auto& padding : padding_pages_) { paged_kv_->ensure_tokens(padding, 1); pages.push_back(&padding); }
        bind_pages(pages);
        page_table_dirty_ = true;
    }
}
void Qwen35BatchStorage::finish_decode() {
    if (unpacked_) {
        const auto& packed = decode_buckets_.at(decode_batch_);
        for (size_t layer = 0; layer < packed.layers.size(); ++layer) {
            auto [source1, source2] = tensors(packed.layers[layer]);
            auto [target1, target2] = tensors(unpacked_->layers[layer]);
            if (!target1.defined()) continue;
            for (size_t row = 0; row < decode_slots_.size(); ++row) {
                target1.narrow(0, decode_slots_[row], 1).copy_(source1.narrow(0, row, 1));
                target2.narrow(0, decode_slots_[row], 1).copy_(source2.narrow(0, row, 1));
            }
        }
        restore(*unpacked_, cache_position());
        unpacked_.reset();
        decode_slots_.clear();
        if (paged_kv_) for (auto& padding : padding_pages_) paged_kv_->release(padding);
    }
    model_.cache_pos = cache_position();
}
void Qwen35BatchStorage::recover(std::span<QwenBatchRequestState* const> requests) {
    auto failure = std::exchange(binding_failure_, {});
    try { mfq_cuda_synchronize(); } catch (...) {}
    for (auto* request : requests) release_state(*request);
    suspended_.reset();
    unpacked_.reset();
    decode_buckets_.clear();
    decode_slots_.clear();
    if (paged_kv_) for (auto& padding : padding_pages_) paged_kv_->release(padding);
    finish_retire();
    if (failure) std::rethrow_exception(failure);
}
void Qwen35BatchStorage::bind_pages(std::span<const PagedKvSequence* const> pages) {
    paged_kv_->bind(pages);
    for (size_t layer = 0; layer < paged_layers_.size(); ++layer)
        paged_layers_[layer]->state->cache = paged_kv_->view(layer);
}
void Qwen35BatchStorage::bind_paged_slots() {
    if (!paged_kv_) return;
    std::vector<const PagedKvSequence*> pages;
    for (auto& sequence : paged_slots_) { paged_kv_->ensure_tokens(sequence, 1); pages.push_back(&sequence); }
    bind_pages(pages);
    page_table_dirty_ = false;
}
QwenBatchState Qwen35BatchStorage::take(int64_t batch) {
    MFQ_RUNTIME_CHECK(model_.speculative_start < 0, "cannot unbind speculative state");
    QwenBatchState result;
    result.batch = batch;
    for (auto& block : model_.blocks) {
        QwenBatchLayerState layer;
        if (auto* full = dynamic_cast<FullBlock*>(block.get()))
            layer.full = std::exchange(full->state, std::make_shared<FullAttentionState>());
        else if (auto* linear = dynamic_cast<LinearBlock*>(block.get())) {
            MFQ_RUNTIME_CHECK(!linear->state->speculative_pending, "cannot unbind speculative state");
            layer.recurrent = std::exchange(linear->state, std::make_shared<LinearAttentionState>());
        } else throw std::runtime_error("unsupported Qwen block state");
        result.layers.push_back(std::move(layer));
    }
    model_.cache_pos = 0;
    return result;
}
void Qwen35BatchStorage::restore(const QwenBatchState& state, int64_t position) {
    MFQ_RUNTIME_CHECK(state.batch > 0 && state.layers.size() == model_.blocks.size(), "Qwen state layout changed");
    for (size_t i = 0; i < model_.blocks.size(); ++i) {
        const auto& saved = state.layers[i];
        if (auto* full = dynamic_cast<FullBlock*>(model_.blocks[i].get())) {
            MFQ_RUNTIME_CHECK(saved.full && !saved.recurrent, "full attention state changed");
            full->state = saved.full;
        } else if (auto* linear = dynamic_cast<LinearBlock*>(model_.blocks[i].get())) {
            MFQ_RUNTIME_CHECK(saved.recurrent && !saved.full, "recurrent state changed");
            linear->state = saved.recurrent;
        } else throw std::runtime_error("unsupported Qwen block state");
    }
    model_.cache_pos = position;
    model_.speculative_start = -1;
    model_.speculative_confirmed = 0;
}
QwenBatchState Qwen35BatchStorage::make_slot_state(const QwenBatchState& source, int64_t slots) const {
    MFQ_RUNTIME_CHECK(source.batch > 0 && slots > 0, "slot state requires source rows");
    QwenBatchState result;
    result.batch = slots;
    for (const auto& saved : source.layers) {
        QwenBatchLayerState layer;
        const auto expand = [slots](const Tensor& value) {
            if (!value.defined()) return Tensor{};
            auto shape = value.sizes().vec(); shape[0] = slots;
            return mfq_tensor_backend::zeros(shape, value.options());
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
void Qwen35BatchStorage::copy_to_slot(QwenBatchState& slots, const QwenBatchState& source, int64_t slot) const {
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
} // namespace mfq::cuda::qwen35
