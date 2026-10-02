#include "storage/weight_loader.h"
#include "transformer_loader.h"

#include "core/full_block.h"
#include "moe.h"

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

void load_important_neuron_branch(CudaExecutionContext &execution, const mfq::ModelSource &mfq,
                                  int64_t hidden_size, int64_t intermediate_size, FFN &f,
                                  const std::string &down_name, const std::string &gate_name,
                                  const std::string &up_name) {
    const std::string down_high = down_name + ".in_high";
    const std::string gate_high = gate_name + ".in_high";
    const std::string up_high = up_name + ".in_high";
    const bool has_down = has_tensor(mfq, down_high);
    const bool has_gate = has_tensor(mfq, gate_high);
    const bool has_up = has_tensor(mfq, up_high);
    if (!has_down && !has_gate && !has_up) {
        return;
    }
    if (!has_down || !has_gate || !has_up) {
        throw std::runtime_error(
            "important-neuron FFN requires matching gate/up/down .in_high records");
    }
    if (f.is_moe) {
        throw std::runtime_error("important-neuron records are unsupported on routed MoE FFNs");
    }

    auto high = std::make_unique<FFN>();
    high->down = load_quant_linear(execution, mfq, down_high, TensorParallelAxis::Input);
    high->gate_up = load_paired_gate_up(execution, mfq, {gate_high, up_high}, high->down);
    high->geglu = f.geglu;
    high->swiglu_limit = f.swiglu_limit;

    if (f.gate_up.outs.size() != 2 || high->gate_up.outs.size() != 2 ||
        f.gate_up.outs[0] != f.gate_up.outs[1] || high->gate_up.outs[0] != high->gate_up.outs[1] ||
        f.down.out() != hidden_size || high->down.out() != hidden_size ||
        f.down.neuron_len() != f.gate_up.outs[0] ||
        high->down.neuron_len() != high->gate_up.outs[0] ||
        f.down.neuron_len() + high->down.neuron_len() != intermediate_size) {
        throw std::runtime_error("important-neuron FFN tensor shapes disagree with model config");
    }
    f.important_neurons = std::move(high);
}

FFN load_moe_weights(CudaExecutionContext &execution, const mfq::ModelSource &source,
                     std::string_view prefix, const MoeWeightLoadOptions &options) {
    const std::string base(prefix);
    const std::string gate_up = base + "experts.gate_up.weight";
    const std::string gate = base + "experts.gate.weight";
    const std::string up = base + "experts.up.weight";
    const std::string down = base + "experts.down.weight";
    const bool fused = has_tensor(source, gate_up);
    const bool split_gate = has_tensor(source, gate);
    const bool split_up = has_tensor(source, up);
    if (split_gate != split_up || fused == split_gate || !has_tensor(source, down)) {
        throw std::runtime_error("routed MoE requires down and exactly one fused or split "
                                 "Gate/Up representation at layer " +
                                 std::to_string(options.layer));
    }

    FFN result;
    result.is_moe = true;
    result.moe_split_gate_up = split_gate;
    result.moe_layer = options.layer;
    if (options.cpu_offloaded) {
        if (split_gate) {
            result.cpu_moe_gate = load_mfe_cpu_offloaded(source, gate);
            result.cpu_moe_up = load_mfe_cpu_offloaded(source, up);
            result.moe_gate = cpu_mixed_moe_metadata(result.cpu_moe_gate);
            result.moe_up = cpu_mixed_moe_metadata(result.cpu_moe_up);
        } else {
            result.cpu_moe_gate_up = load_mfe_cpu_offloaded(source, gate_up);
            result.moe_gate_up = cpu_mixed_moe_metadata(result.cpu_moe_gate_up);
        }
        result.cpu_moe_down = load_mfe_cpu_offloaded(source, down);
        result.moe_down = cpu_mixed_moe_metadata(result.cpu_moe_down);
    } else {
        if (split_gate) {
            result.moe_gate = load_mfe_gpu(execution, source, gate, true, options.layer, "gate");
            result.moe_up = load_mfe_gpu(execution, source, up, true, options.layer, "up");
        } else {
            result.moe_gate_up =
                load_mfe_gpu(execution, source, gate_up, true, options.layer, "gate_up");
        }
        result.moe_down = load_mfe_gpu(execution, source, down, true, options.layer, "down");
    }
    result.moe_router = load_dense_gpu(execution, source, base + "router.weight")
                            .to(mfq_tensor_backend::kFloat32)
                            .contiguous();
    const std::string router_bias = base + "router.bias";
    if (options.router_bias_required || has_tensor(source, router_bias)) {
        result.moe_router_bias = load_dense_gpu(execution, source, router_bias)
                                     .to(mfq_tensor_backend::kFloat32)
                                     .contiguous();
    }
    result.shared = std::make_unique<FFN>();
    result.shared->down = load_quant_linear(execution, source, base + "shared_expert.down.weight");
    result.shared->gate_up = load_paired_gate_up(
        execution, source, {base + "shared_expert.gate.weight", base + "shared_expert.up.weight"},
        result.shared->down, options.shared_gate_up_compatible_prefix);
    prepare_ffn_workspaces(execution, *result.shared);
    return result;
}

FFN load_ffn(CudaExecutionContext &execution, const mfq::ModelSource &source,
             const mfq::models::ModelConfig &config, int layer, bool minicpmo45,
             std::string_view tensor_root) {
    FFN ffn;
    const std::string prefix =
        std::string(tensor_root) + ".block." + std::to_string(layer) + ".mlp.";
    const std::string down = prefix + "down.weight";
    const std::string gate = prefix + "gate.weight";
    const std::string up = prefix + "up.weight";
    ffn.down = load_quant_linear(execution, source, down);
    ffn.gate_up = load_paired_gate_up(execution, source, {gate, up}, ffn.down, 2, minicpmo45);
    load_important_neuron_branch(execution, source, config.hidden_size, config.intermediate_size,
                                 ffn, down, gate, up);
    prepare_ffn_workspaces(execution, ffn);
    return ffn;
}
