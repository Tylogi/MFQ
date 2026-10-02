#include "storage/weight_loader.h"
#include "ops.h"
#include "../session_codec_impl.h"

namespace mfq::cuda::glm_dsa {

void load_ffn(CudaExecutionContext &execution, const mfq::ModelSource &mfq, const Config &c, int i,
              FFN &f) {
    const auto &config = c;
    const std::string p = "model.block." + std::to_string(i) + ".mlp.";
    if (config.mlp_layer_types.at(static_cast<size_t>(i)) == "sparse") {
        f = load_moe_weights(execution, mfq, p, {.layer = i, .router_bias_required = true});
        f.moe_top_k = static_cast<int>(c.num_experts_per_tok);
        f.moe_use_sigmoid = true;
        f.moe_use_sqrt_softplus = false;
        f.moe_normalize = c.norm_topk_prob;
        f.moe_delayed_softmax = false;
        f.moe_shared_ungated = true;
        f.moe_router_scale = c.routed_scaling_factor;
        if (f.moe_gate_up.n_experts != c.num_experts || f.moe_down.n_experts != c.num_experts ||
            f.moe_gate_up.neuron_len != c.hidden_size ||
            f.moe_gate_up.out_per_expert != 2 * c.moe_intermediate_size ||
            f.moe_down.neuron_len != c.moe_intermediate_size ||
            f.moe_down.out_per_expert != c.hidden_size || f.moe_router.dim() != 2 ||
            f.moe_router.size(0) != c.num_experts || f.moe_router.size(1) != c.hidden_size ||
            f.moe_router_bias.numel() != c.num_experts) {
            throw std::runtime_error("GLM DSA MoE tensor shapes disagree with config at layer " +
                                     std::to_string(i));
        }
        return;
    }
    const std::string down_name = p + "down.weight";
    const std::string gate_name = p + "gate.weight";
    const std::string up_name = p + "up.weight";
    f.down = load_quant_linear(execution, mfq, down_name);
    f.gate_up = load_paired_gate_up(execution, mfq, {gate_name, up_name}, f.down);
    load_important_neuron_branch(execution, mfq, c.hidden_size, c.intermediate_size, f, down_name,
                                 gate_name, up_name);
    prepare_ffn_workspaces(execution, f);
}

std::unique_ptr<::Block> load_block(CudaExecutionContext &execution, const mfq::ModelSource &mfq,
                                    const Config &c, int i, const std::string &type,
                                    const std::shared_ptr<::GlmDsaSharedState> &state) {
    const auto &config = c;
    if (type != "glm_dsa" || !state) {
        throw std::runtime_error("invalid GLM DSA block loader state");
    }
    const std::string lp = "model.block." + std::to_string(i) + ".";
    const std::string ap = lp + "attention.";
    auto b = std::make_unique<GlmDsaBlock>();
    b->config = c;
    b->layer = i;
    b->full_indexer = config.indexer_types.at(static_cast<size_t>(i)) == "full";
    b->shared_state = state;
    b->attn_norm = load_dense_gpu(execution, mfq, ap + "norm.weight");
    b->ffn_norm = load_dense_gpu(execution, mfq, lp + "mlp.norm.weight");
    b->q_a_norm = load_dense_gpu(execution, mfq, ap + "query_a_norm.weight");
    b->kv_a_norm = load_dense_gpu(execution, mfq, ap + "key_value_a_norm.weight");
    std::vector<std::string> first_names = {
        ap + "query_a.weight",
        ap + "key_value_a.weight",
    };
    std::vector<std::string> second_names = {
        ap + "query_b.weight",
    };
    if (b->full_indexer) {
        first_names.push_back(ap + "indexer.key.weight");
        first_names.push_back(ap + "indexer.score.weight");
        second_names.push_back(ap + "indexer.query.weight");
        b->index_k_norm = load_dense_gpu(execution, mfq, ap + "indexer.key_norm.weight");
        b->index_k_bias = load_dense_gpu(execution, mfq, ap + "indexer.key_norm.bias");
    }
    b->input_proj = load_quant_group(execution, mfq, first_names);
    b->q_proj = load_quant_group(execution, mfq, second_names);
    b->embed_q = load_mfe_gpu(execution, mfq, ap + "latent.query_embedding.weight");
    b->unembed_out = load_mfe_gpu(execution, mfq, ap + "latent.output_unembedding.weight");
    b->o_proj = load_quant_linear(execution, mfq, ap + "output.weight");
    load_ffn(execution, mfq, c, i, b->ffn);

    const bool input_shape_ok = b->input_proj.outs.size() == (b->full_indexer ? 4u : 2u) &&
                                b->input_proj.outs[0] == c.q_lora_rank &&
                                b->input_proj.outs[1] == c.kv_lora_rank + c.qk_rope_head_dim &&
                                (!b->full_indexer || (b->input_proj.outs[2] == c.index_head_dim &&
                                                      b->input_proj.outs[3] == c.index_n_heads));
    const bool q_shape_ok =
        b->q_proj.outs.size() == (b->full_indexer ? 2u : 1u) &&
        b->q_proj.outs[0] == c.num_attention_heads * (c.qk_nope_head_dim + c.qk_rope_head_dim) &&
        (!b->full_indexer || b->q_proj.outs[1] == c.index_n_heads * c.index_head_dim);
    const bool head_shape_ok = b->embed_q.n_experts == c.num_attention_heads &&
                               b->embed_q.neuron_len == c.qk_nope_head_dim &&
                               b->embed_q.out_per_expert == c.kv_lora_rank &&
                               b->unembed_out.n_experts == c.num_attention_heads &&
                               b->unembed_out.neuron_len == c.kv_lora_rank &&
                               b->unembed_out.out_per_expert == c.v_head_dim &&
                               b->o_proj.neuron_len() == c.num_attention_heads * c.v_head_dim &&
                               b->o_proj.out() == c.hidden_size;
    if (!input_shape_ok || !q_shape_ok || !head_shape_ok || b->attn_norm.numel() != c.hidden_size ||
        b->ffn_norm.numel() != c.hidden_size || b->q_a_norm.numel() != c.q_lora_rank ||
        b->kv_a_norm.numel() != c.kv_lora_rank ||
        (b->full_indexer && (b->index_k_norm.numel() != c.index_head_dim ||
                             b->index_k_bias.numel() != c.index_head_dim))) {
        throw std::runtime_error("GLM DSA tensor shapes disagree with config at layer " +
                                 std::to_string(i));
    }
    return b;
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

TextSessionState CudaSessionCodec<GlmDsaModel>::capture(const Model &model,
                                                        const std::vector<int64_t> &tokens) {
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
        MfqCudaGuard guard(block->cuda_device);
        const auto *glm = dynamic_cast<const GlmDsaBlock *>(block.get());
        if (glm == nullptr || !glm->kv_cache.defined() || glm->kv_cache.dim() != 4 ||
            glm->kv_cache.size(0) != 1 || model.cache_pos > glm->kv_cache.size(2)) {
            throw std::runtime_error("GLM DSA session MLA cache is unavailable");
        }
        GlmDsaBlockSessionState saved;
        saved.full_indexer = glm->full_indexer;
        saved.kv_capacity = glm->kv_cache.size(2);
        saved.kv_cache = glm->kv_cache.narrow(2, 0, model.cache_pos).clone();
        state.bytes += session_tensor_bytes(saved.kv_cache);
        if (glm->full_indexer) {
            if (!glm->index_cache.defined() || glm->index_cache.dim() != 3 ||
                glm->index_cache.size(0) != 1 || model.cache_pos > glm->index_cache.size(1)) {
                throw std::runtime_error("GLM DSA session index cache is unavailable");
            }
            saved.index_capacity = glm->index_cache.size(1);
            saved.index_cache = glm->index_cache.narrow(1, 0, model.cache_pos).clone();
            state.bytes += session_tensor_bytes(saved.index_cache);
        }
        layers.push_back(std::move(saved));
    }
    return state;
}

void CudaSessionCodec<GlmDsaModel>::restore(Model &model, const TextSessionState &state) {
    const auto *layers = std::get_if<std::vector<GlmDsaBlockSessionState>>(&state.payload);
    if (kind(model) != TextSessionStateKind::GlmDsa ||
        state.kind() != TextSessionStateKind::GlmDsa || state.cache_pos <= 0 ||
        static_cast<size_t>(state.cache_pos) != state.tokens.size() || layers == nullptr ||
        layers->size() != model.blocks.size()) {
        throw CudaSessionStateError("GLM DSA text session state is incompatible");
    }
    for (size_t index = 0; index < model.blocks.size(); ++index) {
        auto &block = model.blocks[index];
        MfqCudaGuard guard(block->cuda_device);
        auto *glm = dynamic_cast<GlmDsaBlock *>(block.get());
        const auto &saved = (*layers)[index];
        if (glm == nullptr || glm->full_indexer != saved.full_indexer ||
            !saved.kv_cache.defined() || saved.kv_cache.dim() != 4 || saved.kv_cache.size(0) != 1 ||
            saved.kv_cache.size(2) != state.cache_pos) {
            throw CudaSessionStateError("GLM DSA saved MLA cache is invalid");
        }
        restore_session_prefix_tensor(glm->kv_cache, saved.kv_cache, 2, saved.kv_capacity);
        if (saved.full_indexer) {
            if (!saved.index_cache.defined() || saved.index_cache.dim() != 3 ||
                saved.index_cache.size(0) != 1 || saved.index_cache.size(1) != state.cache_pos) {
                throw CudaSessionStateError("GLM DSA saved index cache is invalid");
            }
            restore_session_prefix_tensor(glm->index_cache, saved.index_cache, 1,
                                          saved.index_capacity);
        } else {
            glm->index_cache = mfq_tensor_backend::Tensor();
        }
        glm->shared_state->reset();
    }
    model.cache_pos = state.cache_pos;
}

} // namespace mfq::cuda

namespace mfq::cuda {} // namespace mfq::cuda

namespace mfq::models {
template struct glm_dsa::CausalLm<cuda::CudaCausalOps<cuda::GlmDsaModel>>;
} // namespace mfq::models
