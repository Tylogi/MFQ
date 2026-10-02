#include "storage/weight_loader.h"
#include "transformer.h"

#include "full_block.h"

#include <stdexcept>
#include <string>

std::unique_ptr<Block> load_transformer_block(
        CudaExecutionContext& execution,
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
    block->attn_norm = load_dense_gpu(execution, source, prefix + "attention.norm.weight");
    block->ffn_norm = load_dense_gpu(execution, source, prefix + "mlp.norm.weight");
    const std::string attention = prefix + "attention.";
    block->qkv = load_quant_group(execution, source, {
        attention + "query.weight",
        attention + "key.weight",
        attention + "value.weight"}, 2, nullptr, minicpmo45);
    block->o = load_quant_linear(execution, source, attention + "output.weight");
    if (has_tensor(source, attention + "query_norm.weight")) {
        block->q_norm = load_dense_gpu(execution, source, attention + "query_norm.weight");
    }
    if (has_tensor(source, attention + "key_norm.weight")) {
        block->k_norm = load_dense_gpu(execution, source, attention + "key_norm.weight");
    }
    block->ffn = load_ffn(
        execution, source, config, layer, minicpmo45, tensor_root);
    return block;
}
