#include "causal_lm.h"

namespace mfq::cuda::deepseek_v41_runtime {

std::unique_ptr<::Block> load_block(
    const mfq::ModelSource& model,
    std::int64_t layer,
    const std::shared_ptr<SharedState>& shared) {
    MFQ_RUNTIME_CHECK(shared, "missing DeepSeek-V4.1 CUDA state");
    const auto& config = shared->config;
    const auto prefix = "model.block." + std::to_string(layer) + ".";
    auto result = std::make_unique<Block>();
    result->config = config;
    result->layer = layer;
    result->ratio = config.compress_ratios.at(
        static_cast<std::size_t>(layer));
    result->max_context = config.max_position_embeddings;
    result->shared = shared;
    if (config.has_engram(layer)) {
        result->engram = Engram::load(
            model, config, static_cast<int>(layer));
    }
    result->attention_norm = load_dense_gpu(
        model, prefix + "attention.norm.weight");
    result->mlp_norm = load_dense_gpu(
        model, prefix + "mlp.norm.weight");
    result->query_a_norm = load_dense_gpu(
        model, prefix + "attention.query_a_norm.weight");
    result->key_value_norm = load_dense_gpu(
        model, prefix + "attention.key_value_norm.weight");
    result->sinks = load_dense_gpu(
        model, prefix + "attention.sink")
                            .to(mfq_tensor_backend::kFloat32)
                            .contiguous();
    result->attention_mhc_function = load_dense_gpu(
        model, prefix + "attention.mhc.pre.function")
                                         .to(mfq_tensor_backend::kFloat32)
                                         .contiguous();
    result->attention_mhc_scale = load_dense_gpu(
        model, prefix + "attention.mhc.pre.scale")
                                      .to(mfq_tensor_backend::kFloat32)
                                      .contiguous();
    result->attention_mhc_base = load_dense_gpu(
        model, prefix + "attention.mhc.pre.base")
                                     .to(mfq_tensor_backend::kFloat32)
                                     .contiguous();
    result->mlp_mhc_function = load_dense_gpu(
        model, prefix + "mlp.mhc.pre.function")
                                   .to(mfq_tensor_backend::kFloat32)
                                   .contiguous();
    result->mlp_mhc_scale = load_dense_gpu(
        model, prefix + "mlp.mhc.pre.scale")
                                .to(mfq_tensor_backend::kFloat32)
                                .contiguous();
    result->mlp_mhc_base = load_dense_gpu(
        model, prefix + "mlp.mhc.pre.base")
                               .to(mfq_tensor_backend::kFloat32)
                               .contiguous();
    result->query_a = load_quant_linear(
        model, prefix + "attention.query_a.weight");
    result->query_b = load_quant_linear(
        model, prefix + "attention.query_b.weight");
    result->key_value = load_quant_linear(
        model, prefix + "attention.key_value.weight");
    result->output_a = load_quant_linear(
        model, prefix + "attention.output_a.weight");
    result->output_b = load_quant_linear(
        model, prefix + "attention.output_b.weight");
    if (result->kv_source()) {
        result->compressor_key_value = load_quant_linear(
            model, prefix + "attention.compressor.key_value.weight");
        result->compressor_norm = load_dense_gpu(
            model, prefix + "attention.compressor.norm.weight");
        result->index_key = load_quant_linear(
            model, prefix + "attention.indexer.key.weight");
        result->index_key_norm = load_dense_gpu(
            model, prefix + "attention.indexer.key_norm.weight");
        if (result->ratio > 1) {
            result->compressor_gate = load_quant_linear(
                model, prefix + "attention.compressor.gate.weight");
        }
    }
    if (result->index_source()) {
        result->index_query = load_quant_linear(
            model, prefix + "attention.indexer.query.weight");
        result->index_score = load_quant_linear(
            model, prefix + "attention.indexer.score.weight");
    }
    result->mlp = load_moe(model, config, layer);
    result->rope = Dsv4RopeTable(
        config.max_position_embeddings,
        config.rope_theta,
        result->ratio,
        config.compress_rope_theta,
        config.rope_scaling.original_max_position_embeddings,
        config.rope_scaling.factor,
        config.rope_scaling.beta_fast,
        config.rope_scaling.beta_slow);

    const auto function_width = config.hc_mult * config.hidden;
    const auto valid_mhc = [&](const Tensor& function,
                               const Tensor& scale,
                               const Tensor& base,
                               const Tensor& norm) {
        return function.dim() == 2 && function.size(0) == 24 &&
            function.size(1) == function_width &&
            scale.numel() == 3 && base.numel() == 24 &&
            norm.numel() == config.hidden;
    };
    const bool base_shapes =
        valid_mhc(
            result->attention_mhc_function,
            result->attention_mhc_scale,
            result->attention_mhc_base,
            result->attention_norm) &&
        valid_mhc(
            result->mlp_mhc_function,
            result->mlp_mhc_scale,
            result->mlp_mhc_base,
            result->mlp_norm) &&
        result->query_a_norm.numel() == config.q_lora_rank &&
        result->key_value_norm.numel() == config.head_dim &&
        result->sinks.numel() == config.n_heads &&
        result->query_a.neuron_len() == config.hidden &&
        result->query_a.out() == config.q_lora_rank &&
        result->query_b.neuron_len() == config.q_lora_rank &&
        result->query_b.out() == config.n_heads * config.head_dim &&
        result->key_value.neuron_len() == config.hidden &&
        result->key_value.out() == config.head_dim &&
        result->output_a.neuron_len() ==
            config.n_heads * config.head_dim / config.o_groups &&
        result->output_a.out() == config.o_groups * config.o_lora_rank &&
        result->output_b.neuron_len() ==
            config.o_groups * config.o_lora_rank &&
        result->output_b.out() == config.hidden;
    const bool source_shapes = !result->kv_source() ||
        (result->compressor_key_value.neuron_len() == config.hidden &&
         result->compressor_key_value.out() == config.head_dim &&
         result->compressor_norm.numel() == config.head_dim &&
         result->index_key.neuron_len() == config.head_dim &&
         result->index_key.out() == config.index_head_dim &&
         result->index_key_norm.numel() == config.index_head_dim &&
         (result->ratio == 1 ||
          (result->compressor_gate.neuron_len() == config.hidden &&
           result->compressor_gate.out() == config.head_dim)));
    const bool index_shapes = !result->index_source() ||
        (result->index_query.neuron_len() == config.q_lora_rank &&
         result->index_query.out() ==
             config.index_n_heads * config.index_head_dim &&
         result->index_score.neuron_len() == config.hidden &&
         result->index_score.out() == config.index_n_heads);
    MFQ_RUNTIME_CHECK(
        config.hc_mult == 4 && config.n_heads == 64 &&
            config.head_dim == 512 && config.rope_head_dim == 64 &&
            config.index_n_heads == 32 && config.index_head_dim == 128 &&
            base_shapes && source_shapes && index_shapes,
        "DeepSeek-V4.1 CUDA tensor geometry disagrees at layer ",
        layer);
    return result;
}

void validate_load_options() {
    const auto& execution = cuda_execution_context();
    if (execution.tensor_parallel.enabled() ||
            execution.layer_placement.enabled() ||
            execution.n_gpu_layers >= 0) {
        throw std::runtime_error(
            "DeepSeek-V4.1 native CUDA currently supports single-device dense "
            "placement or expert parallelism");
    }
}

std::shared_ptr<SharedState> load_shared_state(
        const mfq::ModelSource& source,
        const CommonConfig& config) {
    auto state = std::make_shared<SharedState>();
    state->config = config;
    state->engram_hash = EngramHashState::load(source, config);
    return state;
}

} // namespace mfq::cuda::deepseek_v41_runtime
