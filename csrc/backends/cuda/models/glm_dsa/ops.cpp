#include "storage/transformer_loader.h"
#include "storage/weight_loader.h"
#include "ops.h"
#include "storage/session_codec.h"

namespace mfq::cuda::glm_dsa {

void load_ffn(CudaExecutionContext& execution, const mfq::ModelSource& source,
              const Config& config, int layer, FFN& ffn) {
    TransformerWeightLoader loader{execution, source};
    mfq::models::glm_dsa::load_ffn(loader, config, layer, ffn);
}

std::unique_ptr<::Block> load_block(CudaExecutionContext& execution, const mfq::ModelSource& source,
                                    const Config& config, int layer, const std::string& type,
                                    const std::shared_ptr<::GlmDsaSharedState>& state) {
    if (!state) throw std::runtime_error("invalid GLM DSA block loader state");
    auto block = std::make_unique<GlmDsaBlock>();
    block->shared_state = state;
    TransformerWeightLoader loader{execution, source};
    mfq::models::glm_dsa::load_block(*block, loader, config, layer, type);
    return block;
}

} // namespace mfq::cuda::glm_dsa

namespace mfq::cuda {
void GlmDsaModel::adapter_validate_model_geometry() const {
    if (config.q_lora_rank <= 0 || config.kv_lora_rank <= 0 || config.qk_nope_head_dim <= 0 ||
        config.qk_rope_head_dim <= 0 || config.v_head_dim <= 0 || config.index_head_dim <= 0 ||
        config.index_n_heads <= 0 || config.index_topk <= 0 || config.num_experts <= 0 ||
        config.num_experts_per_tok <= 0 || config.num_attention_heads != 64 ||
        config.num_key_value_heads != 64 || config.kv_lora_rank != 512 ||
        config.qk_nope_head_dim != 192 || config.qk_rope_head_dim != 64 ||
        config.v_head_dim != 256 || config.index_head_dim != 128 || config.index_n_heads != 32 ||
        config.index_topk != 2048 ||
        config.qk_head_dim != config.qk_nope_head_dim + config.qk_rope_head_dim ||
        config.attention_bias || !config.rope_interleave || !config.indexer_rope_interleave ||
        config.hidden_act != "silu" || config.expert_group_count != 1 ||
        config.selected_group_count != 1 || config.shared_expert_count != 1 ||
        config.scoring_func != "sigmoid" || config.topk_method != "noaux_tc") {
        throw std::runtime_error("unsupported GLM DSA CUDA configuration");
    }
}

bool GlmDsaModel::adapter_uses_common_rope() const noexcept { return true; }

std::unique_ptr<Block> GlmDsaModel::adapter_load_block(const mfq::ModelSource &source, int layer,
                                                       int device, const std::string &type) {
    auto &state = block_states[device];
    if (!state)
        state = std::make_shared<GlmDsaSharedState>();
    return glm_dsa::load_block(*execution, source, config, layer, type, state);
}

TextSessionStateKind CudaSessionCodec<GlmDsaModel>::kind(const Model &model) {
    return !model.blocks.empty() &&
                   std::all_of(model.blocks.begin(), model.blocks.end(),
                               [](const std::unique_ptr<::Block> &block) {
                                   return dynamic_cast<const GlmDsaBlock *>(block.get()) != nullptr;
                               })
               ? TextSessionStateKind::GlmDsa
               : TextSessionStateKind::Unsupported;
}

bool CudaSessionCodec<GlmDsaModel>::supports_paged(const Model &) { return false; }

mfq::StepSequence<TextSessionState>
CudaSessionCodec<GlmDsaModel>::capture_steps(
    const Model &model, const std::vector<int64_t> &tokens) {
  if (kind(model) != TextSessionStateKind::GlmDsa || model.cache_pos <= 0 ||
      static_cast<size_t>(model.cache_pos) != tokens.size()) {
    throw std::runtime_error("GLM DSA text session state is unavailable");
  }
  TextSessionState state;
  state.tokens = tokens;
  state.cache_pos = model.cache_pos;
  state.payload = std::vector<GlmDsaBlockSessionState>{};
  auto &layers = std::get<std::vector<GlmDsaBlockSessionState>>(state.payload);
  layers.reserve(model.blocks.size());
  for (const auto &block : model.blocks) {
    {
      MfqCudaGuard guard(block->cuda_device);
      const auto *glm = dynamic_cast<const GlmDsaBlock *>(block.get());
      if (glm == nullptr || !glm->kv_cache.defined() ||
          glm->kv_cache.dim() != 4 || glm->kv_cache.size(0) != 1 ||
          model.cache_pos > glm->kv_cache.size(2)) {
        throw std::runtime_error("GLM DSA session MLA cache is unavailable");
      }
      GlmDsaBlockSessionState saved;
      saved.full_indexer = glm->full_indexer;
      saved.kv_capacity = glm->kv_cache.size(2);
      saved.kv_cache = glm->kv_cache.narrow(2, 0, model.cache_pos).clone();
      state.bytes += session_tensor_bytes(saved.kv_cache);
      if (glm->full_indexer) {
        if (!glm->index_cache.defined() || glm->index_cache.dim() != 3 ||
            glm->index_cache.size(0) != 1 ||
            model.cache_pos > glm->index_cache.size(1)) {
          throw std::runtime_error(
              "GLM DSA session index cache is unavailable");
        }
        saved.index_capacity = glm->index_cache.size(1);
        saved.index_cache =
            glm->index_cache.narrow(1, 0, model.cache_pos).clone();
        state.bytes += session_tensor_bytes(saved.index_cache);
      }
      layers.push_back(std::move(saved));
    }
    co_yield mfq::StepState::advanced;
  }
  co_yield std::move(state);
}

mfq::StepSequence<std::monostate>
CudaSessionCodec<GlmDsaModel>::restore_steps(Model &model,
                                             const TextSessionState &state) {
  const auto *layers =
      std::get_if<std::vector<GlmDsaBlockSessionState>>(&state.payload);
  if (kind(model) != TextSessionStateKind::GlmDsa ||
      state.kind() != TextSessionStateKind::GlmDsa || state.cache_pos <= 0 ||
      static_cast<size_t>(state.cache_pos) != state.tokens.size() ||
      layers == nullptr || layers->size() != model.blocks.size()) {
    throw CudaSessionStateError("GLM DSA text session state is incompatible");
  }
  for (size_t index = 0; index < model.blocks.size(); ++index) {
    {
      auto &block = model.blocks[index];
      MfqCudaGuard guard(block->cuda_device);
      auto *glm = dynamic_cast<GlmDsaBlock *>(block.get());
      const auto &saved = (*layers)[index];
      if (glm == nullptr || glm->full_indexer != saved.full_indexer ||
          !saved.kv_cache.defined() || saved.kv_cache.dim() != 4 ||
          saved.kv_cache.size(0) != 1 ||
          saved.kv_cache.size(2) != state.cache_pos) {
        throw CudaSessionStateError("GLM DSA saved MLA cache is invalid");
      }
      restore_session_prefix_tensor(glm->kv_cache, saved.kv_cache, 2,
                                    saved.kv_capacity);
      if (saved.full_indexer) {
        if (!saved.index_cache.defined() || saved.index_cache.dim() != 3 ||
            saved.index_cache.size(0) != 1 ||
            saved.index_cache.size(1) != state.cache_pos) {
          throw CudaSessionStateError("GLM DSA saved index cache is invalid");
        }
        restore_session_prefix_tensor(glm->index_cache, saved.index_cache, 1,
                                      saved.index_capacity);
      } else {
        glm->index_cache = mfq_tensor_backend::Tensor();
      }
      glm->shared_state->reset();
    }
    co_yield mfq::StepState::advanced;
  }
  model.cache_pos = state.cache_pos;
  co_yield std::monostate{};
}

} // namespace mfq::cuda

namespace mfq::cuda {} // namespace mfq::cuda

namespace mfq::models {
template struct glm_dsa::CausalLm<cuda::CudaCausalOps<cuda::GlmDsaModel>>;
} // namespace mfq::models
