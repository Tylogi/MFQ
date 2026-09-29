#include "causal_lm.h"
#include "../causal_lm_impl.h"

#include "models/transformer.h"

#include <bit>
#include <cmath>

namespace mfq::cuda::gemma4 {

std::unique_ptr<::Block> load_block(
        const mfq::ModelSource& mfq,
        const Config& config,
        int i,
        const std::string& type) {
        const auto& c = config;
        if (type != "full_attention" && type != "sliding_attention") {
            throw std::runtime_error("unsupported Gemma4 layer type: " + type);
        }
        const std::string lp =
            "model.block." + std::to_string(i) + ".";
        const std::string ap = lp + "attention.";
        auto b = std::make_unique<FullBlock>();
        b->layer = i;
        b->gemma4 = true;
        b->gemma4_moe = c.num_experts > 0;
        b->max_position_embeddings = c.max_position_embeddings;
        b->rms_norm_eps = c.rms_norm_eps;
        b->norm_weight_offset = 0.0;
        b->sliding = type == "sliding_attention";
        b->value_equals_key =
            !b->sliding && config.attention_key_equals_value;
        b->attention_heads = c.num_attention_heads;
        b->kv_heads = b->sliding
            ? c.num_key_value_heads : config.num_global_key_value_heads;
        b->attention_head_dim = b->sliding
            ? c.head_dim : config.global_head_dim;
        b->attention_rotary_dim = b->attention_head_dim;
        b->attention_window = b->sliding ? config.sliding_window : 0;
        b->attention_scale = 1.0;
        b->attention_rope = RopeCache(
            c.max_position_embeddings, b->attention_rotary_dim,
            b->sliding ? config.sliding_rope_base : c.rope_base,
            b->attention_head_dim,
            b->sliding ? -1 : (int64_t)std::llround(
                c.full_rotary_factor * (double)b->attention_head_dim / 2.0));

        b->attn_norm = load_dense_gpu(mfq, ap + "norm.weight");
        b->attn_post_norm = load_dense_gpu(
            mfq, ap + "output_norm.weight");
        b->q_norm = load_dense_gpu(mfq, ap + "query_norm.weight");
        b->k_norm = load_dense_gpu(mfq, ap + "key_norm.weight");
        b->v_norm = mfq_tensor_backend::ones(
            {b->attention_head_dim},
            mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA).dtype(mfq_tensor_backend::kFloat32));
        std::vector<std::string> projections = {
            ap + "query.weight", ap + "key.weight"};
        if (!b->value_equals_key) projections.push_back(ap + "value.weight");
        b->qkv = load_quant_group(mfq, projections, 2);
        b->o = load_quant_linear(mfq, ap + "output.weight");

        b->ffn_norm = load_dense_gpu(
            mfq, lp + "mlp.dense.input_norm.weight");
        b->ffn_post_norm = load_dense_gpu(
            mfq, lp + "mlp.output_norm.weight");
        b->layer_scale = load_dense_gpu(mfq, lp + "output_scale")
            .to(mfq_tensor_backend::kFloat16).contiguous();
        if (b->gemma4_moe) {
            b->ffn_post_norm_1 = load_dense_gpu(
                mfq, lp + "mlp.dense.output_norm.weight");
            b->ffn_pre_norm_2 = load_dense_gpu(
                mfq, lp + "mlp.experts.input_norm.weight");
            b->ffn_post_norm_2 = load_dense_gpu(
                mfq, lp + "mlp.experts.output_norm.weight");
        }

        const std::string mp = lp + "mlp.";
        b->ffn.geglu = true;
        b->ffn.down = load_quant_linear(mfq, mp + "down.weight");
        b->ffn.gate_up = load_paired_gate_up(mfq, {
            mp + "gate.weight", mp + "up.weight"},
            b->ffn.down);
        prepare_ffn_workspaces(b->ffn);

        if (!b->gemma4_moe) return b;

        b->gemma_moe_gate_up = load_mfe_gpu(
            mfq, mp + "experts.gate_up.weight",
            true, i, "gate_up");
        b->gemma_moe_down = load_mfe_gpu(
            mfq, mp + "experts.down.weight",
            true, i, "down");
        b->gemma_router = load_dense_gpu(
            mfq, mp + "router.weight").to(mfq_tensor_backend::kFloat32).contiguous();
        b->gemma_router_norm_scale = (
            load_dense_gpu(mfq, mp + "router.norm.weight").to(mfq_tensor_backend::kFloat32) /
            std::sqrt((double)c.hidden_size)).contiguous();
        b->gemma_expert_scale = load_dense_gpu(
            mfq, mp + "router.expert_scale").to(mfq_tensor_backend::kFloat32).contiguous();
        b->gemma_top_k = static_cast<int>(c.num_experts_per_tok);

        if (b->gemma_moe_gate_up.n_experts != c.num_experts ||
            b->gemma_moe_down.n_experts != c.num_experts ||
            b->gemma_moe_gate_up.neuron_len != c.hidden_size ||
            b->gemma_moe_gate_up.out_per_expert != 2 * c.moe_intermediate_size ||
            b->gemma_moe_down.neuron_len != c.moe_intermediate_size ||
            b->gemma_moe_down.out_per_expert != c.hidden_size ||
            b->gemma_router.dim() != 2 ||
            b->gemma_router.size(0) != c.num_experts ||
            b->gemma_router.size(1) != c.hidden_size ||
            b->gemma_router_norm_scale.numel() != c.hidden_size ||
            b->gemma_expert_scale.numel() != c.num_experts) {
            throw std::runtime_error(
                "Gemma4 MoE tensor shapes disagree with config at layer " +
                std::to_string(i));
        }
        return b;
}


} // namespace mfq::cuda::gemma4

namespace mfq::cuda {

void Gemma4Model::adapter_load_config(
        std::string_view payload,
        const mfq::ModelGraph&,
        const mfq::ModelSource&) {
    config = gemma4::Config::from_json(payload);
    auto bits = std::bit_cast<std::uint32_t>(static_cast<float>(
        std::sqrt(static_cast<double>(config.hidden_size))));
    bits += 0x7fffU + ((bits >> 16U) & 1U);
    embed_scale = std::bit_cast<float>(bits & 0xffff0000U);
    metadata.vocab_size = config.vocab_size;
    metadata.hidden_size = config.hidden_size;
    metadata.num_hidden_layers = config.num_hidden_layers;
    metadata.num_attention_heads = config.num_attention_heads;
    metadata.num_key_value_heads = config.num_key_value_heads;
    metadata.head_dim = config.head_dim;
    metadata.max_position_embeddings = config.max_position_embeddings;
    metadata.rotary_dim = config.rotary_dim;
    metadata.num_experts = config.num_experts;
    metadata.rope_base = config.rope_base;
    metadata.rms_norm_eps = config.rms_norm_eps;
    metadata.final_logit_softcapping = config.final_logit_softcapping;
    metadata.embedding_scale = embed_scale;
    metadata.tie_word_embeddings = config.tie_word_embeddings;
    metadata.gemma4 = true;
    metadata.decode_graph_double_warmup = true;
    metadata.model_type = config.model_type;
    metadata.layer_types = config.layer_types;
}

std::unique_ptr<Block>
Gemma4Model::adapter_load_block(
        const mfq::ModelSource& source,
        int layer,
        int,
        const std::string& type) {
    return gemma4::load_block(source, config, layer, type);
}

mfq_tensor_backend::Tensor
Gemma4Model::adapter_prepare_hidden(
        mfq_tensor_backend::Tensor hidden,
        int64_t,
        int64_t) const {
    return execution->profiler.measure("model.embed_scale", [&]() {
        return hidden * embed_scale;
    });
}

} // namespace mfq::cuda

namespace mfq::cuda {

template struct CudaSessionCodec<Gemma4Model>;
template struct CausalLm<Gemma4Model>;

} // namespace mfq::cuda
