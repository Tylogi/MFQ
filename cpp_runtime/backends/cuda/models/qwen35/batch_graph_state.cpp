#include "batch_state.h"

#include "linear_attention.h"
#include "ops.h"
#include "core/full_block.h"
#include "runtime/batch_decode_graph.h"

#include <set>

namespace mfq::cuda::qwen35 {
namespace tb = mfq_tensor_backend;
using Tensor = tb::Tensor;

std::vector<int64_t> Qwen35BatchStateAdapter::decode_capacities() const {
    std::vector<int64_t> capacities;
    for (const auto& block : model_.blocks) {
        if (const auto* full = dynamic_cast<const FullBlock*>(block.get()))
            capacities.push_back(full->cache.is_paged()
                ? full->cache.page_table.size(1) * full->cache.page_size : full->cache.k.size(2));
    }
    return capacities;
}

void Qwen35BatchStateAdapter::bind_decode_rows(BatchDecodeInputs& inputs, const std::vector<int32_t>& slots,
                                             bool graph) {
    MFQ_RUNTIME_CHECK(!decode_bound_ && slots.size() <= std::size_t(inputs.rows()), "decode rows already bound");
    if (decode_layers_.empty())
        decode_layers_.resize(model_.blocks.size());
    decode_bound_ = true;
    decode_graph_binding_ = graph;
    decode_bound_layers_ = 0;
    try {
        for (std::size_t i = 0; i < model_.blocks.size(); ++i) {
            auto& block = model_.blocks[i];
            auto& saved = decode_layers_[i];
            MfqCudaGuard guard(block->cuda_device);
            if (auto* full = dynamic_cast<FullBlock*>(block.get())) {
                saved.cache = full->cache;
                decode_bound_layers_ = i + 1;
                for (int64_t row = 0; row < inputs.rows(); ++row) {
                    const auto slot = row < int64_t(slots.size()) ? slots[row] : -1;
                    inputs.state(row, 2 * i, slot >= 0 && !saved.cache.is_paged()
                        ? saved.cache.k.narrow(0, slot, 1) : Tensor{}, 2);
                    inputs.state(row, 2 * i + 1, slot >= 0 && !saved.cache.is_paged()
                        ? saved.cache.v.narrow(0, slot, 1) : Tensor{}, 2);
                }
            } else if (auto* linear = dynamic_cast<LinearAttentionBlock*>(block.get())) {
                saved.first = linear->mixer.convolution;
                saved.second = linear->mixer.recurrent;
                decode_bound_layers_ = i + 1;
                for (int64_t row = 0; row < inputs.rows(); ++row) {
                    const auto slot = row < int64_t(slots.size()) ? slots[row] : -1;
                    inputs.state(row, 2 * i, slot >= 0 ? saved.first.narrow(0, slot, 1) : Tensor{});
                    inputs.state(row, 2 * i + 1, slot >= 0 ? saved.second.narrow(0, slot, 1) : Tensor{});
                }
            }
        }
        inputs.upload();
        std::set<int> uploaded;
        for (const auto& block : model_.blocks) {
            const auto device = block->cuda_device;
            if (device == inputs.ids().get_device() || !uploaded.insert(device).second)
                continue;
            auto& mirror = decode_inputs_[{inputs.rows(), device}];
            if (!mirror)
                mirror = std::make_unique<BatchDecodeInputs>(inputs.rows(), 2 * model_.blocks.size(), device);
            MfqCudaGuard guard(device);
            mirror->copy_from(inputs);
        }
    } catch (...) {
        unbind_decode_rows();
        throw;
    }
}

void Qwen35BatchStateAdapter::load_decode_rows(const BatchDecodeInputs& primary) {
    for (std::size_t i = 0; i < model_.blocks.size(); ++i) {
        auto& block = model_.blocks[i];
        auto& saved = decode_layers_[i];
        MfqCudaGuard guard(block->cuda_device);
        const auto& inputs = block->cuda_device == primary.ids().get_device() ? primary
            : *decode_inputs_.at({primary.rows(), block->cuda_device});
        const auto batch = inputs.rows();
        if (auto* full = dynamic_cast<FullBlock*>(block.get())) {
            const auto& original = saved.cache;
            if (original.is_paged()) {
                auto selected = original.page_table.index_select(0, inputs.slots().clamp_min(0));
                auto pages = tb::where((inputs.enabled() != 0).unsqueeze(1), selected, tb::full_like(selected, -1));
                full->cache = KVCache::paged_view(original.k_chunk_ptrs, original.v_chunk_ptrs, pages,
                    batch, original.paged_heads, original.paged_head_dim, original.page_size,
                    original.pages_per_chunk, original.paged_dtype);
            } else {
                auto pointers = inputs.states().narrow(0, 2 * i, 2).select(2, 0).contiguous();
                auto pages = tb::arange(batch, inputs.ids().options().dtype(tb::kInt32)).reshape({batch, 1});
                pages = tb::where((inputs.enabled() != 0).unsqueeze(1), pages, tb::full_like(pages, -1));
                const auto heads = original.k.size(1);
                const auto width = full->attention_head_dim > 0 ? full->attention_head_dim : model_.head_dim();
                full->cache = KVCache::paged_view(pointers.select(0, 0), pointers.select(0, 1), pages,
                    batch, heads, width, original.k.size(2), 1, original.scalar_type());
            }
            full->cache.quantization = original.quantization;
        } else if (auto* linear = dynamic_cast<LinearAttentionBlock*>(block.get())) {
            if (!saved.working_first.defined() || saved.working_first.size(0) != batch ||
                saved.working_graph != decode_graph_binding_) {
                auto conv_shape = saved.first.sizes().vec();
                auto state_shape = saved.second.sizes().vec();
                conv_shape[0] = state_shape[0] = batch;
                auto convolution = tb::empty(conv_shape, saved.first.options());
                auto recurrent = tb::empty(state_shape, saved.second.options());
                saved.working_first = std::move(convolution);
                saved.working_second = std::move(recurrent);
                saved.working_graph = decode_graph_binding_;
            }
            linear->mixer.convolution = saved.working_first;
            linear->mixer.recurrent = saved.working_second;
            for (int64_t row = 0; row < batch; ++row) {
                batch_state_load(inputs, 2 * i, row, linear->mixer.convolution.narrow(0, row, 1));
                batch_state_load(inputs, 2 * i + 1, row, linear->mixer.recurrent.narrow(0, row, 1));
            }
        }
    }
}

void Qwen35BatchStateAdapter::store_decode_rows(const BatchDecodeInputs& primary) {
    for (std::size_t i = 0; i < model_.blocks.size(); ++i) {
        auto& block = model_.blocks[i];
        auto* linear = dynamic_cast<LinearAttentionBlock*>(block.get());
        if (!linear)
            continue;
        MfqCudaGuard guard(block->cuda_device);
        const auto& inputs = block->cuda_device == primary.ids().get_device() ? primary
            : *decode_inputs_.at({primary.rows(), block->cuda_device});
        for (int64_t row = 0; row < inputs.rows(); ++row) {
            batch_state_store(inputs, 2 * i, row, linear->mixer.convolution.narrow(0, row, 1));
            batch_state_store(inputs, 2 * i + 1, row, linear->mixer.recurrent.narrow(0, row, 1));
        }
        if (block->cuda_device != primary.ids().get_device()) {
            join_batch_decode_stream(mfq_get_current_cuda_stream(block->cuda_device),
                                     mfq_get_current_cuda_stream(primary.ids().get_device()));
        }
    }
}

void Qwen35BatchStateAdapter::unbind_decode_rows() {
    if (!decode_bound_)
        return;
    for (std::size_t i = 0; i < decode_bound_layers_; ++i) {
        auto& block = model_.blocks[i];
        auto& saved = decode_layers_[i];
        if (auto* full = dynamic_cast<FullBlock*>(block.get())) {
            full->cache = std::move(saved.cache);
        } else if (auto* linear = dynamic_cast<LinearAttentionBlock*>(block.get())) {
            linear->mixer.convolution = std::move(saved.first);
            linear->mixer.recurrent = std::move(saved.second);
            // Graph temporaries stay in the shared pool after their tensor
            // handles are released, so every bucket can reuse these pages.
            if (saved.working_graph) {
                saved.working_first = {};
                saved.working_second = {};
                saved.working_graph = false;
            }
        }
    }
    decode_bound_ = false;
    decode_bound_layers_ = 0;
}

std::vector<Tensor> Qwen35BatchStateAdapter::recurrent_slot_tensors(const std::vector<int32_t>& slots) const {
    std::vector<Tensor> result;
    for (const auto& block : model_.blocks) {
        if (auto* linear = dynamic_cast<LinearAttentionBlock*>(block.get())) {
            for (const auto slot : slots) {
                result.push_back(linear->mixer.convolution.narrow(0, slot, 1));
                result.push_back(linear->mixer.recurrent.narrow(0, slot, 1));
            }
        }
    }
    return result;
}

} // namespace mfq::cuda::qwen35
