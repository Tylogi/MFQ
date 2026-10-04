#pragma once

#include "models/common/causal_model_ops.h"
#include "models/common/full_block.h"
#include "storage/session_state.h"

namespace mfq::cuda {

template <typename Model>
TextSessionState
CudaSessionCodec<Model>::capture(const CausalLm<Model> &model,
                                 const std::vector<int64_t> &tokens) {
  return mfq::finish_steps(capture_steps(model, tokens));
}
template <typename Model>
void CudaSessionCodec<Model>::restore(CausalLm<Model> &model,
                                      const TextSessionState &state) {
  (void)mfq::finish_steps(restore_steps(model, state));
}

template <typename Model>
TextSessionStateKind CudaSessionCodec<Model>::kind(const CausalLm<Model> &) {
    return TextSessionStateKind::Unsupported;
}

template <typename Model> bool CudaSessionCodec<Model>::supports_paged(const CausalLm<Model> &) {
    return false;
}

template <typename Model>
mfq::StepSequence<TextSessionState>
CudaSessionCodec<Model>::capture_steps(const CausalLm<Model> &,
                                       const std::vector<int64_t> &) {
  throw std::runtime_error("text session state is unsupported by this model");
  co_return;
}

template <typename Model>
mfq::StepSequence<std::monostate>
CudaSessionCodec<Model>::restore_steps(CausalLm<Model> &,
                                       const TextSessionState &) {
  throw CudaSessionStateError(
      "text session state is unsupported by this model");
  co_yield std::monostate{};
}

template <typename Model>
TextSessionStateKind FullAttentionSessionCodec<Model>::kind(const CausalLm<Model> &model) {
    return !model.blocks.empty() && std::all_of(model.blocks.begin(),
                                        model.blocks.end(),
                                        [](const std::unique_ptr<Block> &block) {
        return dynamic_cast<const FullBlock *>(block.get()) != nullptr;
    })
               ? TextSessionStateKind::FullAttention
               : TextSessionStateKind::Unsupported;
}

template <typename Model>
bool FullAttentionSessionCodec<Model>::supports_paged(const CausalLm<Model> &model) {
    return kind(model) == TextSessionStateKind::FullAttention &&
           std::all_of(
               model.blocks.begin(), model.blocks.end(), [](const std::unique_ptr<Block> &block) {
        const auto *full = dynamic_cast<const FullBlock *>(block.get());
        return full != nullptr && !full->sliding;
    });
}

template <typename Model>
mfq::StepSequence<TextSessionState>
FullAttentionSessionCodec<Model>::capture_steps(
    const CausalLm<Model> &model, const std::vector<int64_t> &tokens) {
  if (kind(model) != TextSessionStateKind::FullAttention) {
    throw std::runtime_error(
        "text session state is unsupported by this block layout");
  }
  if (model.cache_pos <= 0 ||
      static_cast<size_t>(model.cache_pos) != tokens.size()) {
    throw std::runtime_error(
        "text session token count does not match the model cache");
  }
  TextSessionState state;
  state.tokens = tokens;
  state.cache_pos = model.cache_pos;
  state.payload = std::vector<FullBlockSessionState>{};
  auto &layers = std::get<std::vector<FullBlockSessionState>>(state.payload);
  layers.reserve(model.blocks.size());
  for (const auto &block : model.blocks) {
    {
      MfqCudaGuard guard(block->cuda_device);
      const auto *full = dynamic_cast<const FullBlock *>(block.get());
      if (full == nullptr) {
        throw std::runtime_error("full-attention session layer changed");
      }
      layers.push_back(capture_full_attention_session_state(
          *full, model.cache_pos, state.bytes));
    }
    co_yield mfq::StepState::advanced;
  }
  co_yield std::move(state);
}

template <typename Model>
mfq::StepSequence<std::monostate>
FullAttentionSessionCodec<Model>::restore_steps(CausalLm<Model> &model,
                                                const TextSessionState &state) {
  const auto *layers =
      std::get_if<std::vector<FullBlockSessionState>>(&state.payload);
  if (kind(model) != TextSessionStateKind::FullAttention ||
      state.kind() != TextSessionStateKind::FullAttention ||
      state.cache_pos <= 0 ||
      static_cast<size_t>(state.cache_pos) != state.tokens.size() ||
      layers == nullptr || layers->size() != model.blocks.size()) {
    throw CudaSessionStateError(
        "full-attention text session state is incompatible");
  }
  for (size_t index = 0; index < model.blocks.size(); ++index) {
    {
      auto &block = model.blocks[index];
      MfqCudaGuard guard(block->cuda_device);
      auto *full = dynamic_cast<FullBlock *>(block.get());
      if (full == nullptr) {
        throw CudaSessionStateError("full-attention session layer changed");
      }
      restore_full_attention_session_state(*full, (*layers)[index]);
    }
    co_yield mfq::StepState::advanced;
  }
  model.cache_pos = state.cache_pos;
  co_yield std::monostate{};
}

} // namespace mfq::cuda
