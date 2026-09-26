#include "cuda_transformer_loader.h"

#include "cuda_transformer.h"

std::unique_ptr<Block> load_transformer_block(
        const mfq::ModelSource& source,
        const mfq::models::ModelConfig& config,
        int layer,
        const std::string& type,
        bool minicpmo45,
        std::string_view tensor_root) {
    if (type != "full_attention") {
        throw std::runtime_error("unsupported layer type: " + type);
    }

    const std::string prefix =
        std::string(tensor_root) + ".block." +
        std::to_string(layer) + ".";
    auto block = std::make_unique<FullBlock>();
    block->layer = layer;
    block->attention_heads = config.num_attention_heads;
    block->kv_heads = config.num_key_value_heads;
    block->attention_head_dim = config.head_dim;
    block->max_position_embeddings = config.max_position_embeddings;
    block->rms_norm_eps = config.rms_norm_eps;
    block->norm_weight_offset = minicpmo45 ? 0.0 : 1.0;
    block->official_bf16 = minicpmo45;
    block->attn_norm = load_dense_gpu(
        source, prefix + "attention.norm.weight");
    block->ffn_norm = load_dense_gpu(
        source, prefix + "mlp.norm.weight");
    const std::string attention = prefix + "attention.";
    block->qkv = load_quant_group(source, {
        attention + "query.weight",
        attention + "key.weight",
        attention + "value.weight"}, 2, nullptr, minicpmo45);
    block->o = load_quant_linear(source, attention + "output.weight");
    if (has_tensor(source, attention + "query_norm.weight")) {
        block->q_norm = load_dense_gpu(
            source, attention + "query_norm.weight");
    }
    if (has_tensor(source, attention + "key_norm.weight")) {
        block->k_norm = load_dense_gpu(
            source, attention + "key_norm.weight");
    }
    block->ffn = load_ffn(
        source, config, layer, minicpmo45, tensor_root);
    return block;
}

FFN load_ffn(
        const mfq::ModelSource& source,
        const mfq::models::ModelConfig& config,
        int layer,
        bool minicpmo45,
        std::string_view tensor_root) {
    FFN ffn;
    const std::string prefix =
        std::string(tensor_root) + ".block." +
        std::to_string(layer) + ".mlp.";
    const std::string down = prefix + "down.weight";
    const std::string gate = prefix + "gate.weight";
    const std::string up = prefix + "up.weight";
    ffn.down = load_quant_linear(source, down);
    ffn.gate_up = load_paired_gate_up(
        source, {gate, up}, ffn.down, 2, minicpmo45);
    load_important_neuron_branch(
        source, config.hidden_size, config.intermediate_size,
        ffn, down, gate, up);
    prepare_ffn_workspaces(ffn);
    return ffn;
}
