#include "qwen35_causal_lm.h"
#include "qwen35_linear_attention.h"

#include "../../runtime/causal_lm.h"
#include "../../runtime/cuda_transformer.h"
#include "../../runtime/cuda_transformer_loader.h"

namespace mfq::cuda::qwen35 {
namespace {

FFN load_qwen_ffn(
        const mfq::ModelSource& source,
        const Config& config,
        int layer,
        std::string_view tensor_root) {
    const std::string prefix =
        std::string(tensor_root) + ".block." +
        std::to_string(layer) + ".mlp.";
    const std::string expert_gate_up =
        prefix + "experts.gate_up.weight";
    const std::string expert_gate = prefix + "experts.gate.weight";
    const std::string expert_up = prefix + "experts.up.weight";
    const std::string expert_down = prefix + "experts.down.weight";
    const bool has_expert_gate_up = has_tensor(source, expert_gate_up);
    const bool has_expert_gate = has_tensor(source, expert_gate);
    const bool has_expert_up = has_tensor(source, expert_up);
    const bool has_expert_down = has_tensor(source, expert_down);

    if (!has_expert_gate_up && !has_expert_gate &&
            !has_expert_up && !has_expert_down) {
        return load_ffn(source, config, layer, false, tensor_root);
    }
    if (has_expert_gate != has_expert_up) {
        throw std::runtime_error(
            "Qwen MoE split Gate/Up records are incomplete at layer " +
            std::to_string(layer));
    }
    if (has_expert_gate_up == has_expert_gate || !has_expert_down) {
        throw std::runtime_error(
            "Qwen MoE layer requires exactly one fused or split Gate/Up representation at layer " +
            std::to_string(layer));
    }
    if (config.num_experts <= 0 || config.num_experts_per_tok <= 0 ||
            config.moe_intermediate_size <= 0 ||
            config.shared_expert_intermediate_size <= 0) {
        throw std::runtime_error("Qwen MoE config fields are missing");
    }

    FFN ffn;
    ffn.is_moe = true;
    ffn.moe_split_gate_up = has_expert_gate;
    if (ffn.moe_split_gate_up) {
        ffn.moe_gate = load_mfe_gpu(
            source, expert_gate, true, layer, "gate");
        ffn.moe_up = load_mfe_gpu(
            source, expert_up, true, layer, "up");
    } else {
        ffn.moe_gate_up = load_mfe_gpu(
            source, expert_gate_up, true, layer, "gate_up");
    }
    ffn.moe_down = load_mfe_gpu(
        source, expert_down, true, layer, "down");
    ffn.moe_router = load_dense_gpu(source, prefix + "router.weight")
        .to(mfq_tensor_backend::kFloat32).contiguous();
    ffn.moe_shared_gate = load_dense_gpu(
        source, prefix + "shared_expert.router.weight")
        .to(mfq_tensor_backend::kFloat32).contiguous();
    ffn.moe_top_k = static_cast<int>(config.num_experts_per_tok);
    ffn.moe_use_sqrt_softplus =
        config.expert_gating_func == "sqrtsoftplus";
    if (ffn.moe_use_sqrt_softplus) {
        ffn.moe_normalize = config.norm_topk_prob;
        ffn.moe_delayed_softmax = false;
        ffn.moe_router_scale = config.routed_scaling_factor;
    } else {
        // Qwen's full-softmax -> top-k -> renormalize is exactly a
        // softmax over the selected logits; keep that fused delayed form.
        ffn.moe_delayed_softmax = true;
    }
    ffn.moe_layer = layer;
    ffn.shared = std::make_unique<FFN>();
    ffn.shared->down = load_quant_linear(
        source, prefix + "shared_expert.down.weight");
    ffn.shared->gate_up = load_paired_gate_up(source, {
        prefix + "shared_expert.gate.weight",
        prefix + "shared_expert.up.weight"},
        ffn.shared->down);
    prepare_ffn_workspaces(*ffn.shared);

    const bool routed_gate_shapes = ffn.moe_split_gate_up
        ? ffn.moe_gate.n_experts == config.num_experts &&
            ffn.moe_up.n_experts == config.num_experts &&
            ffn.moe_gate.neuron_len == config.hidden_size &&
            ffn.moe_up.neuron_len == config.hidden_size &&
            ffn.moe_gate.out_per_expert == config.moe_intermediate_size &&
            ffn.moe_up.out_per_expert == config.moe_intermediate_size
        : ffn.moe_gate_up.n_experts == config.num_experts &&
            ffn.moe_gate_up.neuron_len == config.hidden_size &&
            ffn.moe_gate_up.out_per_expert ==
                2 * config.moe_intermediate_size;
    if (!routed_gate_shapes ||
            ffn.moe_down.n_experts != config.num_experts ||
            ffn.moe_down.neuron_len != config.moe_intermediate_size ||
            ffn.moe_down.out_per_expert != config.hidden_size ||
            ffn.moe_router.dim() != 2 ||
            ffn.moe_router.size(0) != config.num_experts ||
            ffn.moe_router.size(1) != config.hidden_size ||
            ffn.moe_shared_gate.dim() != 2 ||
            ffn.moe_shared_gate.size(0) != 1 ||
            ffn.moe_shared_gate.size(1) != config.hidden_size) {
        throw std::runtime_error(
            "Qwen MoE tensor shapes disagree with config at layer " +
            std::to_string(layer));
    }
    return ffn;
}

} // namespace

std::unique_ptr<::Block> load_block(
        const mfq::ModelSource& source,
        const Config& config,
        int layer,
        const std::string& type,
        std::string_view tensor_root) {
    const int i = layer;
    const std::string lp =
        std::string(tensor_root) + ".block." +
        std::to_string(i) + ".";
    if (type == "full_attention") {
        auto b = std::make_unique<FullBlock>();
        b->layer = i;
        b->attention_heads = config.num_attention_heads;
        b->kv_heads = config.num_key_value_heads;
        b->attention_head_dim = config.head_dim;
        b->max_position_embeddings = config.max_position_embeddings;
        b->rms_norm_eps = config.rms_norm_eps;
        b->norm_weight_offset =
            config.legacy_tensor_layout.norm_weight_offset;
        b->attention_output_gate = config.attention_output_gate;
        b->attn_norm = load_dense_gpu(
            source, lp + "attention.norm.weight");
        b->ffn_norm = load_dense_gpu(source, lp + "mlp.norm.weight");
        const std::string ap = lp + "attention.";
        const std::string query_name = ap + "query.weight";
        const std::string key_name = ap + "key.weight";
        const std::string value_name = ap + "value.weight";
        const bool mirror_kv =
            config.attention_output_gate &&
            tensor_parallel_mirror_qwen35_attention_kv_enabled() &&
            is_quant_dtype(require_tensor(source, key_name).dtype) &&
            is_quant_dtype(require_tensor(source, value_name).dtype);
        if (mirror_kv) {
            b->split_q_kv_projections = true;
            b->q_projection = load_quant_linear(source, query_name);
            const auto mirrored = std::optional<TensorParallelAxis>(
                TensorParallelAxis::Mirrored);
            b->k_projection = load_quant_linear(
                source, key_name, mirrored);
            b->v_projection = load_quant_linear(
                source, value_name, mirrored);
        } else {
            b->qkv = load_quant_group(
                source, {query_name, key_name, value_name}, 2);
        }
        b->o = load_quant_linear(source, ap + "output.weight");
        if (has_tensor(source, ap + "query_norm.weight")) {
            b->q_norm = load_dense_gpu(
                source, ap + "query_norm.weight");
        }
        if (has_tensor(source, ap + "key_norm.weight")) {
            b->k_norm = load_dense_gpu(
                source, ap + "key_norm.weight");
        }
        b->ffn = load_qwen_ffn(
            source, config, layer, tensor_root);
        return b;
    }
    if (type == "linear_attention") {
        auto b = std::make_unique<LinearAttentionBlock>();
        b->qwen_config = config;
        b->tiled_v_heads =
            config.legacy_tensor_layout.qwen_gdn_gguf_layout;
        b->attn_norm = load_dense_gpu(source, lp + "attention.norm.weight");
        b->ffn_norm = load_dense_gpu(source, lp + "mlp.norm.weight");
        const std::string sp = lp + "linear_attention.";
        const std::string alpha_name = sp + "alpha.weight";
        const std::string beta_name = sp + "beta.weight";
        if (has_tensor(source, sp + "qk.weight") &&
                has_tensor(source, sp + "value.weight")) {
            b->split_in_proj = true;
            b->qkv_proj = load_quant_group(
                source, {sp + "qk.weight", sp + "value.weight"});
            if (is_quant_dtype(require_tensor(source, sp + "gate.weight").dtype)) {
                b->z_proj = load_quant_linear(source, sp + "gate.weight");
                const bool a_nint = is_quant_dtype(require_tensor(source, alpha_name).dtype);
                const bool b_nint = is_quant_dtype(require_tensor(source, beta_name).dtype);
                if (a_nint != b_nint) throw std::runtime_error("linear_attn a/b must use the same storage kind");
                b->ab_is_nint = a_nint;
                if (b->ab_is_nint) {
                    const auto scalar_axis =
                        tensor_parallel_mirror_linear_attention_scalars_enabled()
                            ? std::optional<TensorParallelAxis>(
                                TensorParallelAxis::Mirrored)
                            : std::nullopt;
                    b->ab_nint_proj = make_quant_group({
                        load_quant_linear(source, alpha_name, scalar_axis),
                        load_quant_linear(source, beta_name, scalar_axis),
                    });
                } else {
                    b->ab_proj = make_dense_group({
                        load_dense_gpu(source, alpha_name),
                        load_dense_gpu(source, beta_name),
                    });
                }
            } else {
                b->split_dense_zab = true;
                b->zab_proj = make_dense_group({
                    load_dense_gpu(source, sp + "gate.weight"),
                    load_dense_gpu(source, alpha_name),
                    load_dense_gpu(source, beta_name),
                });
            }
        } else {
            const bool alpha_quant =
                is_quant_dtype(require_tensor(source, alpha_name).dtype);
            const bool beta_quant =
                is_quant_dtype(require_tensor(source, beta_name).dtype);
            if (alpha_quant != beta_quant) {
                throw std::runtime_error(
                    "linear_attention alpha/beta must use the same storage kind");
            }
            if (alpha_quant) {
                b->in_proj = load_quant_group(source, {
                    sp + "qkv.weight", sp + "gate.weight",
                    alpha_name, beta_name});
            } else {
                b->dense_ab_tail = true;
                b->in_proj = load_quant_group(
                    source, {sp + "qkv.weight", sp + "gate.weight"});
                b->ab_proj = make_dense_group({
                    load_dense_gpu(source, alpha_name),
                    load_dense_gpu(source, beta_name),
                });
            }
        }
        b->conv_weight = load_dense_gpu(source, sp + "conv.weight");
        if (has_tensor(source, sp + "conv.bias")) {
            b->conv_bias = load_dense_gpu(source, sp + "conv.bias");
        }
        b->dt_bias = load_dense_gpu(source, sp + "dt_bias");
        const auto a_parameter = load_dense_gpu(source, sp + "a");
        b->a_log = config.legacy_tensor_layout.linear_attention_a_is_log
            ? a_parameter
            : mfq_tensor_backend::log(-a_parameter);
        b->linear_norm = load_dense_gpu(source, sp + "norm.weight");
        const std::string out_name = sp + "output.weight";
        if (is_quant_dtype(require_tensor(source, out_name).dtype)) {
            b->out_proj = load_quant_linear(source, out_name);
        } else {
            b->dense_out_proj = true;
            b->out_proj_dense = load_dense_gpu(source, out_name);
            if (require_tensor(source, out_name).dtype == "F16") {
                b->out_proj_dense = b->out_proj_dense
                    .to(mfq_tensor_backend::kFloat16).contiguous();
            }
            if (b->out_proj_dense.dim() != 2) {
                throw std::runtime_error(
                    "dense linear_attention output projection must be 2D");
            }
        }
        b->ffn = load_qwen_ffn(
            source, config, layer, tensor_root);
        return b;
    }
    return load_transformer_block(
        source, config, layer, type, false, tensor_root);
}

void LinearAttentionBlock::clear_speculative() noexcept {
    speculative_recurrent = {};
    speculative_start = -1;
    speculative_confirmed = 0;
    speculative_tokens = 0;
    speculative_pending = false;
}

mfq_tensor_backend::Tensor LinearAttentionBlock::forward_context(
        mfq_tensor_backend::Tensor input,
        const Block::Context& context,
        const RopeCache& rope) {
    if (context.confirmed_prefix == 0) {
        return Block::forward_context(
            std::move(input), context, rope);
    }
    MFQ_RUNTIME_CHECK(
        input.is_cuda() && !speculative_pending &&
            context.confirmed_prefix > 0 &&
            context.confirmed_prefix < input.size(1) &&
            conv_state.defined() && gdn_state.defined(),
        "invalid Qwen3.5 speculative linear-attention transaction");

    if (!speculative_conv.defined() ||
            speculative_conv.sizes() != conv_state.sizes() ||
            !speculative_gdn.defined() ||
            speculative_gdn.sizes() != gdn_state.sizes()) {
        speculative_conv = conv_state.clone();
        speculative_gdn = gdn_state.clone();
    } else {
        speculative_conv.copy_(conv_state);
        speculative_gdn.copy_(gdn_state);
    }
    speculative_start = context.cache_position;
    speculative_confirmed = context.confirmed_prefix;
    speculative_tokens = input.size(1);
    speculative_pending = true;

    try {
        // Match Metal: evaluate the complete [confirmed, drafts...] window
        // once so projections and FFN stay batched. Rollback replays only the
        // recurrent state prefix from this layer's retained projections.
        auto attention = forward_attention_cuda(
            std::move(input), &speculative_recurrent);
        auto result = forward_ffn_cuda(
            std::move(attention[0]), std::move(attention[1]));
        ++speculative_projection_batches;
        ++speculative_ffn_batches;
        return result;
    } catch (...) {
        conv_state.copy_(speculative_conv);
        gdn_state.copy_(speculative_gdn);
        clear_speculative();
        throw;
    }
}

void LinearAttentionBlock::commit_speculative() noexcept {
    clear_speculative();
}

void LinearAttentionBlock::rollback_speculative(int64_t keep_position) {
    MFQ_RUNTIME_CHECK(
        speculative_pending &&
            keep_position >= speculative_start + speculative_confirmed &&
            keep_position < speculative_start + speculative_tokens,
        "invalid Qwen3.5 speculative rollback position");
    const int64_t retained = keep_position - speculative_start;
    conv_state.copy_(speculative_conv);
    gdn_state.copy_(speculative_gdn);
    try {
        // Restore only convolution/GDN state from the retained projected
        // rows; target projections, output projection and FFN are not rerun.
        replay_recurrent_cuda(speculative_recurrent, retained);
        clear_speculative();
    } catch (...) {
        clear_speculative();
        throw;
    }
}

bool supports_text_session_state(
        const std::vector<std::unique_ptr<::Block>>& blocks) {
    return !blocks.empty() && std::all_of(
        blocks.begin(), blocks.end(),
        [](const std::unique_ptr<::Block>& block) {
            return dynamic_cast<const FullBlock*>(block.get()) != nullptr ||
                dynamic_cast<const LinearAttentionBlock*>(block.get()) !=
                    nullptr;
        });
}

TextSessionState capture_text_session_state(
        const std::vector<std::unique_ptr<::Block>>& blocks,
        const std::vector<std::int64_t>& tokens,
        std::int64_t cache_position) {
    if (!supports_text_session_state(blocks) || cache_position <= 0 ||
            static_cast<std::size_t>(cache_position) != tokens.size()) {
        throw std::runtime_error(
            "Qwen hybrid session token count does not match the cache");
    }
    TextSessionState state;
    state.tokens = tokens;
    state.kind = TextSessionStateKind::HybridAttention;
    state.cache_pos = cache_position;
    state.hybrid_blocks.reserve(blocks.size());
    for (const auto& block : blocks) {
        MfqCudaGuard guard(block->cuda_device);
        HybridBlockSessionState saved;
        if (const auto* full = dynamic_cast<const FullBlock*>(block.get())) {
            saved.kind = HybridBlockSessionStateKind::FullAttention;
            saved.full_attention = capture_full_attention_session_state(
                *full, cache_position, state.bytes);
        } else if (const auto* linear =
                       dynamic_cast<const LinearAttentionBlock*>(block.get())) {
            if (linear->speculative_pending ||
                    !linear->conv_state.defined() ||
                    !linear->gdn_state.defined()) {
                throw std::runtime_error(
                    "Qwen recurrent session state is unavailable");
            }
            saved.kind = HybridBlockSessionStateKind::Recurrent;
            saved.convolution_state = linear->conv_state.clone();
            saved.recurrent_state = linear->gdn_state.clone();
            state.bytes += session_tensor_bytes(saved.convolution_state);
            state.bytes += session_tensor_bytes(saved.recurrent_state);
        } else {
            throw std::runtime_error(
                "Qwen hybrid session layer type changed");
        }
        state.hybrid_blocks.push_back(std::move(saved));
    }
    return state;
}

void restore_text_session_state(
        std::vector<std::unique_ptr<::Block>>& blocks,
        const TextSessionState& state) {
    if (state.kind != TextSessionStateKind::HybridAttention ||
            state.cache_pos <= 0 ||
            state.tokens.size() != static_cast<std::size_t>(state.cache_pos) ||
            state.hybrid_blocks.size() != blocks.size()) {
        throw std::runtime_error("Qwen hybrid session state is incompatible");
    }
    for (std::size_t index = 0; index < blocks.size(); ++index) {
        auto& block = blocks[index];
        const auto& saved = state.hybrid_blocks[index];
        MfqCudaGuard guard(block->cuda_device);
        if (saved.kind == HybridBlockSessionStateKind::FullAttention) {
            auto* full = dynamic_cast<FullBlock*>(block.get());
            if (full == nullptr) {
                throw std::runtime_error(
                    "Qwen hybrid full-attention layer changed");
            }
            restore_full_attention_session_state(
                *full, saved.full_attention);
            continue;
        }
        auto* linear = dynamic_cast<LinearAttentionBlock*>(block.get());
        const auto convolution_width = linear != nullptr
            ? 2 * linear->qwen_config.linear_k_size() +
                  linear->qwen_config.linear_v_size()
            : 0;
        if (linear == nullptr || !saved.convolution_state.defined() ||
                !saved.recurrent_state.defined() ||
                saved.convolution_state.scalar_type() !=
                    mfq_tensor_backend::kFloat32 ||
                saved.recurrent_state.scalar_type() !=
                    mfq_tensor_backend::kFloat32 ||
                saved.convolution_state.dim() != 3 ||
                saved.convolution_state.size(0) != 1 ||
                saved.convolution_state.size(1) !=
                    linear->qwen_config.linear_conv_kernel_dim - 1 ||
                saved.convolution_state.size(2) != convolution_width ||
                saved.recurrent_state.dim() != 4 ||
                saved.recurrent_state.size(0) != 1 ||
                saved.recurrent_state.size(1) !=
                    linear->qwen_config.linear_num_value_heads ||
                saved.recurrent_state.size(2) !=
                    linear->qwen_config.linear_value_head_dim ||
                saved.recurrent_state.size(3) !=
                    linear->qwen_config.linear_value_head_dim ||
                !saved.convolution_state.is_cuda() ||
                !saved.recurrent_state.is_cuda() ||
                saved.convolution_state.get_device() != block->cuda_device ||
                saved.recurrent_state.get_device() != block->cuda_device) {
            throw std::runtime_error(
                "Qwen recurrent session topology changed");
        }
        restore_session_tensor(
            linear->conv_state, saved.convolution_state);
        restore_session_tensor(
            linear->gdn_state, saved.recurrent_state);
        linear->speculative_conv = {};
        linear->speculative_gdn = {};
        linear->clear_speculative();
    }
}

} // namespace mfq::cuda::qwen35
