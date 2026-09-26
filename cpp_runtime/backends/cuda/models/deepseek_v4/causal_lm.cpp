#include "causal_lm.h"

#include "../../engine/cuda_transformer.h"

namespace mfq::cuda::deepseek_v4 {

std::unique_ptr<::Block> load_block(
        const mfq::ModelSource& mfq,
        const Config& c,
        int i,
        const std::string& type,
        const std::shared_ptr<::Dsv4SharedState>& state) {
        const auto& config = c;
        if (type != "deepseek_v4" || !state) {
            throw std::runtime_error(
                "invalid DeepSeek V4 block loader state");
        }
        const std::string p =
            "model.block." + std::to_string(i) + ".";
        auto b = std::make_unique<Dsv4Block>();
        b->layer = i;
        b->max_positions = c.max_position_embeddings;
        b->compress_ratio =
            config.compress_ratios.at(static_cast<size_t>(i));
        b->hidden_size = c.hidden_size;
        b->heads = c.num_attention_heads;
        b->head_dim = c.head_dim;
        b->groups = c.o_groups;
        b->o_rank = c.o_lora_rank;
        b->hc_mult = c.hc_mult;
        b->hc_iterations = c.hc_sinkhorn_iters;
        b->eps = c.rms_norm_eps;
        b->hc_eps = c.hc_eps;
        b->shared_state = state;

        b->attn_norm = load_dense_gpu(
            mfq, p + "attention.norm.weight");
        b->ffn_norm = load_dense_gpu(
            mfq, p + "mlp.norm.weight");
        b->q_a_norm = load_dense_gpu(
            mfq, p + "attention.query_a_norm.weight");
        b->kv_norm = load_dense_gpu(
            mfq, p + "attention.key_value_a_norm.weight");
        b->sinks = load_dense_gpu(
            mfq, p + "attention.sink")
            .to(mfq_tensor_backend::kFloat32).contiguous();
        b->hc_attn_fn = load_dense_gpu(
            mfq, p + "attention.mhc.pre.function")
            .to(mfq_tensor_backend::kFloat32).contiguous();
        b->hc_attn_scale = load_dense_gpu(
            mfq, p + "attention.mhc.pre.scale")
            .to(mfq_tensor_backend::kFloat32).contiguous();
        b->hc_attn_base = load_dense_gpu(
            mfq, p + "attention.mhc.pre.base")
            .to(mfq_tensor_backend::kFloat32).contiguous();
        b->hc_ffn_fn = load_dense_gpu(
            mfq, p + "mlp.mhc.pre.function")
            .to(mfq_tensor_backend::kFloat32).contiguous();
        b->hc_ffn_scale = load_dense_gpu(
            mfq, p + "mlp.mhc.pre.scale")
            .to(mfq_tensor_backend::kFloat32).contiguous();
        b->hc_ffn_base = load_dense_gpu(
            mfq, p + "mlp.mhc.pre.base")
            .to(mfq_tensor_backend::kFloat32).contiguous();
        b->q_a = load_quant_linear(
            mfq, p + "attention.query_a.weight");
        b->q_b = load_quant_linear(
            mfq, p + "attention.query_b.weight");
        b->kv = load_quant_linear(
            mfq, p + "attention.key_value_a.weight");
        b->output_a = load_quant_linear(
            mfq, p + "attention.output_a.weight");
        b->output_b = load_quant_linear(
            mfq, p + "attention.output_b.weight");
        b->attention_rope = Dsv4RopeTable(
            c.max_position_embeddings, c.rope_base,
            b->compress_ratio,
            config.compress_rope_base,
            config.rope_original_positions,
            config.rope_factor,
            config.rope_beta_fast,
            config.rope_beta_slow);

        if (b->compress_ratio > 0) {
            b->compressor.ratio = b->compress_ratio;
            b->compressor.head_dim = c.head_dim;
            b->compressor.overlap =
                b->compress_ratio == 4;
            b->compressor.cache_quant_mode = 1;
            b->compressor.projection = make_fp32_quant_group(
                load_quant_group(mfq, {
                    p + "attention.compressor.key_value.weight",
                    p + "attention.compressor.gate.weight"}));
            b->compressor.ape = load_dense_gpu(
                mfq, p + "attention.compressor.position")
                .to(mfq_tensor_backend::kFloat32).contiguous();
            b->compressor.norm = load_dense_gpu(
                mfq, p + "attention.compressor.norm.weight")
                .to(mfq_tensor_backend::kFloat32).contiguous();
        }
        if (b->compress_ratio == 4) {
            b->indexer_compressor.ratio = 4;
            b->indexer_compressor.head_dim = c.index_head_dim;
            b->indexer_compressor.overlap = true;
            b->indexer_compressor.cache_quant_mode = 2;
            b->indexer_compressor.projection =
                make_fp32_quant_group(load_quant_group(mfq, {
                    p + "attention.indexer.compressor.key_value.weight",
                    p + "attention.indexer.compressor.gate.weight"}));
            b->indexer_compressor.ape = load_dense_gpu(
                mfq, p + "attention.indexer.compressor.position")
                .to(mfq_tensor_backend::kFloat32).contiguous();
            b->indexer_compressor.norm = load_dense_gpu(
                mfq, p + "attention.indexer.compressor.norm.weight")
                .to(mfq_tensor_backend::kFloat32).contiguous();
            b->indexer_q = load_quant_linear(
                mfq, p + "attention.indexer.query.weight");
            b->indexer_weight = load_dense_gpu(
                mfq, p + "attention.indexer.score.weight")
                .to(mfq_tensor_backend::kFloat32).contiguous();
        }

        b->ffn.is_moe = true;
        const bool has_split_gate =
            has_tensor(mfq, p + "mlp.experts.gate.weight");
        const bool has_split_up =
            has_tensor(mfq, p + "mlp.experts.up.weight");
        if (has_split_gate != has_split_up) {
            throw std::runtime_error(
                "DeepSeek V4 split routed Gate/Up records are incomplete at layer " +
                std::to_string(i));
        }
        b->ffn.moe_split_gate_up = has_split_gate;
        const bool cpu_offload =
            g_dsv4_cpu_offload_layers.count(i) != 0;
        if (cpu_offload) {
            if (b->ffn.moe_split_gate_up) {
                b->ffn.cpu_moe_gate = load_mfe_cpu_offloaded(
                    mfq, p + "mlp.experts.gate.weight");
                b->ffn.cpu_moe_up = load_mfe_cpu_offloaded(
                    mfq, p + "mlp.experts.up.weight");
                b->ffn.moe_gate =
                    cpu_mixed_moe_metadata(b->ffn.cpu_moe_gate);
                b->ffn.moe_up =
                    cpu_mixed_moe_metadata(b->ffn.cpu_moe_up);
            } else {
                b->ffn.cpu_moe_gate_up = load_mfe_cpu_offloaded(
                    mfq, p + "mlp.experts.gate_up.weight");
                b->ffn.moe_gate_up =
                    cpu_mixed_moe_metadata(b->ffn.cpu_moe_gate_up);
            }
            b->ffn.cpu_moe_down = load_mfe_cpu_offloaded(
                mfq, p + "mlp.experts.down.weight");
            b->ffn.moe_down =
                cpu_mixed_moe_metadata(b->ffn.cpu_moe_down);
            const int64_t gate_up_bytes = b->ffn.moe_split_gate_up
                ? b->ffn.moe_gate.mixed_weight_bytes +
                    b->ffn.moe_up.mixed_weight_bytes
                : b->ffn.moe_gate_up.mixed_weight_bytes;
            const int64_t down_bytes =
                b->ffn.moe_down.mixed_weight_bytes;
            g_dsv4_cpu_offload_host_bytes +=
                gate_up_bytes + down_bytes;
            std::cerr
                << "cpu_offload layer=" << i
                << " gate_up_bytes=" << gate_up_bytes
                << " down_bytes=" << down_bytes
                << " total_host_bytes="
                << g_dsv4_cpu_offload_host_bytes
                << std::endl;
        } else {
            if (b->ffn.moe_split_gate_up) {
                b->ffn.moe_gate = load_mfe_gpu(
                    mfq, p + "mlp.experts.gate.weight",
                    true, i, "gate");
                b->ffn.moe_up = load_mfe_gpu(
                    mfq, p + "mlp.experts.up.weight",
                    true, i, "up");
            } else {
                b->ffn.moe_gate_up = load_mfe_gpu(
                    mfq, p + "mlp.experts.gate_up.weight",
                    true, i, "gate_up");
            }
            b->ffn.moe_down = load_mfe_gpu(
                mfq, p + "mlp.experts.down.weight",
                true, i, "down");
        }
        b->ffn.moe_router = load_dense_gpu(
            mfq, p + "mlp.router.weight")
            .to(mfq_tensor_backend::kFloat32).contiguous();
        if (has_tensor(mfq, p + "mlp.router.bias")) {
            b->ffn.moe_router_bias = load_dense_gpu(
                mfq, p + "mlp.router.bias")
                .to(mfq_tensor_backend::kFloat32).contiguous();
        }
        if (i < config.hash_layer_count) {
            b->ffn.moe_hash_ids = load_dense_gpu(
                mfq, p + "mlp.router.token_to_expert")
                .to(mfq_tensor_backend::kInt32).contiguous();
        }
        b->ffn.moe_top_k =
            static_cast<int>(c.num_experts_per_tok);
        b->ffn.moe_use_sqrt_softplus = true;
        b->ffn.moe_normalize = c.norm_topk_prob;
        b->ffn.moe_delayed_softmax = false;
        b->ffn.moe_shared_ungated = true;
        b->ffn.moe_router_scale =
            c.routed_scaling_factor;
        b->ffn.swiglu_limit = c.swiglu_limit;
        b->ffn.moe_layer = i;
        b->ffn.shared = std::make_unique<FFN>();
        b->ffn.shared->down = load_quant_linear(
            mfq, p + "mlp.shared_expert.down.weight");
        b->ffn.shared->gate_up = load_paired_gate_up(mfq, {
            p + "mlp.shared_expert.gate.weight",
            p + "mlp.shared_expert.up.weight"},
            b->ffn.shared->down, 0);
        b->ffn.shared->swiglu_limit = c.swiglu_limit;
        prepare_ffn_workspaces(*b->ffn.shared);

        const bool base_shapes =
            b->attn_norm.numel() == c.hidden_size &&
            b->ffn_norm.numel() == c.hidden_size &&
            b->q_a_norm.numel() == c.q_lora_rank &&
            b->kv_norm.numel() == c.head_dim &&
            b->sinks.numel() == c.num_attention_heads &&
            b->q_a.neuron_len() == c.hidden_size &&
            b->q_a.out() == c.q_lora_rank &&
            b->q_b.neuron_len() == c.q_lora_rank &&
            b->q_b.out() == c.num_attention_heads * c.head_dim &&
            b->kv.neuron_len() == c.hidden_size &&
            b->kv.out() == c.head_dim &&
            b->output_a.neuron_len() ==
                c.num_attention_heads * c.head_dim / c.o_groups &&
            b->output_a.out() == c.o_groups * c.o_lora_rank &&
            b->output_b.neuron_len() == c.o_groups * c.o_lora_rank &&
            b->output_b.out() == c.hidden_size;
        const bool gate_up_shapes = b->ffn.moe_split_gate_up
            ? b->ffn.moe_gate.n_experts == c.num_experts &&
                b->ffn.moe_up.n_experts == c.num_experts &&
                b->ffn.moe_gate.neuron_len == c.hidden_size &&
                b->ffn.moe_up.neuron_len == c.hidden_size &&
                b->ffn.moe_gate.out_per_expert == c.moe_intermediate_size &&
                b->ffn.moe_up.out_per_expert == c.moe_intermediate_size
            : b->ffn.moe_gate_up.n_experts == c.num_experts &&
                b->ffn.moe_gate_up.neuron_len == c.hidden_size &&
                b->ffn.moe_gate_up.out_per_expert ==
                    2 * c.moe_intermediate_size;
        const bool moe_shapes =
            gate_up_shapes &&
            b->ffn.moe_down.n_experts == c.num_experts &&
            b->ffn.moe_down.neuron_len == c.moe_intermediate_size &&
            b->ffn.moe_down.out_per_expert == c.hidden_size &&
            b->ffn.moe_router.size(0) == c.num_experts &&
            b->ffn.moe_router.size(1) == c.hidden_size;
        if (!base_shapes || !moe_shapes) {
            throw std::runtime_error(
                "DeepSeek V4 tensor shapes disagree with config at layer " +
                std::to_string(i));
        }
        return b;
}

void validate_load_options(const Config& config) {
    if (config.hidden_size != 4096 || config.num_attention_heads != 64 ||
            config.head_dim != 512 || config.q_lora_rank != 1024 ||
            config.qk_rope_head_dim != 64 || config.index_head_dim != 128 ||
            config.index_n_heads != 64 || config.index_topk != 512 ||
            config.o_groups != 8 || config.o_lora_rank != 1024 ||
            config.hc_mult != 4 || config.hc_sinkhorn_iters != 20 ||
            config.num_experts != 256 || config.num_experts_per_tok != 6 ||
            config.moe_intermediate_size != 2048 ||
            config.shared_expert_count != 1 ||
            config.scoring_func != "sqrtsoftplus") {
        throw std::runtime_error(
            "unsupported DeepSeek V4 CUDA configuration");
    }
    if (g_dsv4_cpu_offload_layers.empty()) return;
    for (int layer : g_dsv4_cpu_offload_layers) {
        if (layer < 0 || layer >= config.num_hidden_layers) {
            throw std::runtime_error(
                "CPU-offload layer is outside the model: " +
                std::to_string(layer));
        }
    }
    g_dsv4_cpu_offload_host_bytes = 0;
    g_mfq_drop_file_cache = true;
}

OutputHeadWeights load_output_head(
        const mfq::ModelSource& source) {
    return {
        load_dense_gpu(source, "model.mhc.output.function")
            .to(mfq_tensor_backend::kFloat32).contiguous(),
        load_dense_gpu(source, "model.mhc.output.scale")
            .to(mfq_tensor_backend::kFloat32).contiguous(),
        load_dense_gpu(source, "model.mhc.output.base")
            .to(mfq_tensor_backend::kFloat32).contiguous(),
    };
}

} // namespace mfq::cuda::deepseek_v4
