#include "causal_lm.h"

#include "models/transformer.h"

namespace mfq::cuda::glm_dsa {

void load_ffn(
        const mfq::ModelSource& mfq,
        const Config& c,
        int i,
        FFN& f) {
        const auto& config = c;
        const std::string p =
            "model.block." + std::to_string(i) + ".mlp.";
        if (config.mlp_layer_types.at(static_cast<size_t>(i)) == "sparse") {
            const std::string expert_gate_up = p + "experts.gate_up.weight";
            const std::string expert_down = p + "experts.down.weight";
            f.is_moe = true;
            f.moe_gate_up = load_mfe_gpu(
                mfq, expert_gate_up, true, i, "gate_up");
            f.moe_down = load_mfe_gpu(
                mfq, expert_down, true, i, "down");
            f.moe_router = load_dense_gpu(
                mfq, p + "router.weight").to(mfq_tensor_backend::kFloat32).contiguous();
            f.moe_router_bias = load_dense_gpu(
                mfq, p + "router.bias")
                .to(mfq_tensor_backend::kFloat32).contiguous();
            f.moe_top_k = static_cast<int>(c.num_experts_per_tok);
            f.moe_use_sigmoid = true;
            f.moe_use_sqrt_softplus = false;
            f.moe_normalize = c.norm_topk_prob;
            f.moe_delayed_softmax = false;
            f.moe_shared_ungated = true;
            f.moe_router_scale = c.routed_scaling_factor;
            f.shared = std::make_unique<FFN>();
            f.shared->down = load_quant_linear(
                mfq, p + "shared_expert.down.weight");
            f.shared->gate_up = load_paired_gate_up(mfq, {
                p + "shared_expert.gate.weight",
                p + "shared_expert.up.weight"},
                f.shared->down);
            prepare_ffn_workspaces(*f.shared);
            if (f.moe_gate_up.n_experts != c.num_experts ||
                f.moe_down.n_experts != c.num_experts ||
                f.moe_gate_up.neuron_len != c.hidden_size ||
                f.moe_gate_up.out_per_expert != 2 * c.moe_intermediate_size ||
                f.moe_down.neuron_len != c.moe_intermediate_size ||
                f.moe_down.out_per_expert != c.hidden_size ||
                f.moe_router.dim() != 2 ||
                f.moe_router.size(0) != c.num_experts ||
                f.moe_router.size(1) != c.hidden_size ||
                f.moe_router_bias.numel() != c.num_experts) {
                throw std::runtime_error(
                    "GLM DSA MoE tensor shapes disagree with config at layer " +
                    std::to_string(i));
            }
            return;
        }
        const std::string down_name = p + "down.weight";
        const std::string gate_name = p + "gate.weight";
        const std::string up_name = p + "up.weight";
        f.down = load_quant_linear(mfq, down_name);
        f.gate_up = load_paired_gate_up(mfq, {
            gate_name, up_name},
            f.down);
        load_important_neuron_branch(
            mfq, c.hidden_size, c.intermediate_size,
            f, down_name, gate_name, up_name);
        prepare_ffn_workspaces(f);
}


std::unique_ptr<::Block> load_block(
        const mfq::ModelSource& mfq,
        const Config& c,
        int i,
        const std::string& type,
        const std::shared_ptr<::GlmDsaSharedState>& state) {
        const auto& config = c;
        if (type != "glm_dsa" || !state) {
            throw std::runtime_error("invalid GLM DSA block loader state");
        }
        const std::string lp =
            "model.block." + std::to_string(i) + ".";
        const std::string ap = lp + "attention.";
        auto b = std::make_unique<GlmDsaBlock>();
        b->config = c;
        b->layer = i;
        b->full_indexer =
            config.indexer_types.at(static_cast<size_t>(i)) == "full";
        b->shared_state = state;
        b->attn_norm = load_dense_gpu(mfq, ap + "norm.weight");
        b->ffn_norm = load_dense_gpu(
            mfq, lp + "mlp.norm.weight");
        b->q_a_norm = load_dense_gpu(
            mfq, ap + "query_a_norm.weight");
        b->kv_a_norm = load_dense_gpu(
            mfq, ap + "key_value_a_norm.weight");
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
            b->index_k_norm = load_dense_gpu(
                mfq, ap + "indexer.key_norm.weight");
            b->index_k_bias = load_dense_gpu(
                mfq, ap + "indexer.key_norm.bias");
        }
        b->input_proj = load_quant_group(mfq, first_names);
        b->q_proj = load_quant_group(mfq, second_names);
        b->embed_q = load_mfe_gpu(
            mfq, ap + "latent.query_embedding.weight");
        b->unembed_out = load_mfe_gpu(
            mfq, ap + "latent.output_unembedding.weight");
        b->o_proj = load_quant_linear(mfq, ap + "output.weight");
        load_ffn(mfq, c, i, b->ffn);

        const bool input_shape_ok =
            b->input_proj.outs.size() == (b->full_indexer ? 4u : 2u) &&
            b->input_proj.outs[0] == c.q_lora_rank &&
            b->input_proj.outs[1] == c.kv_lora_rank + c.qk_rope_head_dim &&
            (!b->full_indexer ||
             (b->input_proj.outs[2] == c.index_head_dim &&
              b->input_proj.outs[3] == c.index_n_heads));
        const bool q_shape_ok =
            b->q_proj.outs.size() == (b->full_indexer ? 2u : 1u) &&
            b->q_proj.outs[0] ==
                c.num_attention_heads *
                    (c.qk_nope_head_dim + c.qk_rope_head_dim) &&
            (!b->full_indexer ||
             b->q_proj.outs[1] == c.index_n_heads * c.index_head_dim);
        const bool head_shape_ok =
            b->embed_q.n_experts == c.num_attention_heads &&
            b->embed_q.neuron_len == c.qk_nope_head_dim &&
            b->embed_q.out_per_expert == c.kv_lora_rank &&
            b->unembed_out.n_experts == c.num_attention_heads &&
            b->unembed_out.neuron_len == c.kv_lora_rank &&
            b->unembed_out.out_per_expert == c.v_head_dim &&
            b->o_proj.neuron_len() ==
                c.num_attention_heads * c.v_head_dim &&
            b->o_proj.out() == c.hidden_size;
        if (!input_shape_ok || !q_shape_ok || !head_shape_ok ||
            b->attn_norm.numel() != c.hidden_size ||
            b->ffn_norm.numel() != c.hidden_size ||
            b->q_a_norm.numel() != c.q_lora_rank ||
            b->kv_a_norm.numel() != c.kv_lora_rank ||
            (b->full_indexer &&
             (b->index_k_norm.numel() != c.index_head_dim ||
              b->index_k_bias.numel() != c.index_head_dim))) {
            throw std::runtime_error(
                "GLM DSA tensor shapes disagree with config at layer " +
                std::to_string(i));
        }
        return b;
}


} // namespace mfq::cuda::glm_dsa
