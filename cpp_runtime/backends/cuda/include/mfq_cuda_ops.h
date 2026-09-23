#pragma once

#include "mfq_tensor_backend.h"

#include <cstdint>
#include <vector>

mfq_tensor_backend::Tensor rms_norm_cuda(mfq_tensor_backend::Tensor x, mfq_tensor_backend::Tensor weight, double eps);
mfq_tensor_backend::Tensor rms_norm_offset_cuda(mfq_tensor_backend::Tensor x, mfq_tensor_backend::Tensor weight, double eps, double weight_offset);
mfq_tensor_backend::Tensor rms_norm_f16_cuda(mfq_tensor_backend::Tensor x, mfq_tensor_backend::Tensor weight, double eps,
                                double weight_offset);
mfq_tensor_backend::Tensor qwen_rms_norm_bf16_cuda(
    mfq_tensor_backend::Tensor input, mfq_tensor_backend::Tensor weight, double eps,
    double weight_offset);
std::vector<mfq_tensor_backend::Tensor> qwen_rms_norm_pair_bf16_cuda(
    mfq_tensor_backend::Tensor first, mfq_tensor_backend::Tensor second,
    mfq_tensor_backend::Tensor first_weight, mfq_tensor_backend::Tensor second_weight,
    double eps, double weight_offset);
mfq_tensor_backend::Tensor minicpm_qk_norm_rope_cache_write_bf16_cuda(
    mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor k, mfq_tensor_backend::Tensor v,
    mfq_tensor_backend::Tensor q_weight, mfq_tensor_backend::Tensor k_weight,
    mfq_tensor_backend::Tensor rope_pos, mfq_tensor_backend::Tensor write_pos,
    mfq_tensor_backend::Tensor cos, mfq_tensor_backend::Tensor sin,
    mfq_tensor_backend::Tensor k_cache, mfq_tensor_backend::Tensor v_cache,
    double eps, double weight_offset);
std::vector<mfq_tensor_backend::Tensor> rms_norm_pair_f16_f32_offset_cuda(
    mfq_tensor_backend::Tensor first, mfq_tensor_backend::Tensor second,
    mfq_tensor_backend::Tensor first_weight, mfq_tensor_backend::Tensor second_weight,
    double eps, double weight_offset);
mfq_tensor_backend::Tensor l2_norm_cuda(mfq_tensor_backend::Tensor x, double eps);
mfq_tensor_backend::Tensor acc_cuda(mfq_tensor_backend::Tensor a, mfq_tensor_backend::Tensor b);
std::vector<mfq_tensor_backend::Tensor> acc_rms_norm_cuda(mfq_tensor_backend::Tensor a, mfq_tensor_backend::Tensor b,
                                             mfq_tensor_backend::Tensor weight, double eps,
                                             double weight_offset);
std::vector<mfq_tensor_backend::Tensor> acc_rms_norm_f16_cuda(mfq_tensor_backend::Tensor a, mfq_tensor_backend::Tensor b,
                                                 mfq_tensor_backend::Tensor weight, double eps,
                                                 double weight_offset);
std::vector<mfq_tensor_backend::Tensor> acc_rms_norm_bf16_cuda(
    mfq_tensor_backend::Tensor a, mfq_tensor_backend::Tensor b,
    mfq_tensor_backend::Tensor weight, double eps, double weight_offset);
std::vector<mfq_tensor_backend::Tensor> gemma4_attn_residual_pre_norms_f16_cuda(
    mfq_tensor_backend::Tensor residual, mfq_tensor_backend::Tensor attn,
    mfq_tensor_backend::Tensor attn_post_weight, mfq_tensor_backend::Tensor dense_pre_weight,
    mfq_tensor_backend::Tensor router_weight, mfq_tensor_backend::Tensor moe_pre_weight, double eps);
mfq_tensor_backend::Tensor gemma4_ffn_merge_f16_cuda(
    mfq_tensor_backend::Tensor dense, mfq_tensor_backend::Tensor moe, mfq_tensor_backend::Tensor residual,
    mfq_tensor_backend::Tensor dense_post_weight, mfq_tensor_backend::Tensor moe_post_weight,
    mfq_tensor_backend::Tensor final_post_weight, mfq_tensor_backend::Tensor layer_scale, double eps);
void decode_graph_commit_cuda(mfq_tensor_backend::Tensor next, mfq_tensor_backend::Tensor generated, mfq_tensor_backend::Tensor step,
                              mfq_tensor_backend::Tensor input, mfq_tensor_backend::Tensor pos, mfq_tensor_backend::Tensor len);
std::vector<mfq_tensor_backend::Tensor> linear_gate_beta_cuda(
    mfq_tensor_backend::Tensor alpha, mfq_tensor_backend::Tensor beta, mfq_tensor_backend::Tensor dt_bias, mfq_tensor_backend::Tensor a_log);
mfq_tensor_backend::Tensor rope_table_cuda(mfq_tensor_backend::Tensor x, mfq_tensor_backend::Tensor pos, mfq_tensor_backend::Tensor cos, mfq_tensor_backend::Tensor sin,
                              int64_t rotary_dim, mfq_tensor_backend::Tensor sections);
mfq_tensor_backend::Tensor rope_table_bf16_cuda(mfq_tensor_backend::Tensor x, mfq_tensor_backend::Tensor pos, mfq_tensor_backend::Tensor cos, mfq_tensor_backend::Tensor sin,
                                   int64_t rotary_dim);
mfq_tensor_backend::Tensor minicpm_bf16_rope_cache_write_cuda(
    mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor k, mfq_tensor_backend::Tensor v,
    mfq_tensor_backend::Tensor rope_pos, mfq_tensor_backend::Tensor write_pos,
    mfq_tensor_backend::Tensor cos, mfq_tensor_backend::Tensor sin,
    mfq_tensor_backend::Tensor k_cache, mfq_tensor_backend::Tensor v_cache,
    int64_t rotary_dim);
mfq_tensor_backend::Tensor attention_cuda(mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor k, mfq_tensor_backend::Tensor v, double scale, bool causal);
mfq_tensor_backend::Tensor attention_swa_cuda(mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor k, mfq_tensor_backend::Tensor v,
                                 double scale, int64_t window);
mfq_tensor_backend::Tensor attention_cache_swa_cuda(
    mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor k_cache, mfq_tensor_backend::Tensor v_cache,
    mfq_tensor_backend::Tensor seq_len, double scale, int64_t window);
mfq_tensor_backend::Tensor attention_cache_swa_planned_cuda(
    mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor k_cache, mfq_tensor_backend::Tensor v_cache,
    mfq_tensor_backend::Tensor seq_len, double scale, int64_t window, int64_t planned_length);
mfq_tensor_backend::Tensor mfq_attention_mma256_cuda(mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor k, mfq_tensor_backend::Tensor v, double scale);
mfq_tensor_backend::Tensor mfq_attention_mma128_cuda(mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor k, mfq_tensor_backend::Tensor v, double scale);
mfq_tensor_backend::Tensor minicpm_flash128_q_cast_cuda(
    mfq_tensor_backend::Tensor q);
std::vector<mfq_tensor_backend::Tensor> minicpm_flash128_kv_cast_cuda(
    mfq_tensor_backend::Tensor k, mfq_tensor_backend::Tensor v);
mfq_tensor_backend::Tensor minicpm_flash128_output_cast_cuda(
    mfq_tensor_backend::Tensor output);
mfq_tensor_backend::Tensor mfq_attention_mma512_cuda(mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor k, mfq_tensor_backend::Tensor v, double scale);
mfq_tensor_backend::Tensor mfq_attention_mma256_swa_cuda(
    mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor k, mfq_tensor_backend::Tensor v, double scale, int64_t window);
mfq_tensor_backend::Tensor mfq_attention_mma256_decode_cuda(
    mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor k_cache, mfq_tensor_backend::Tensor v_cache,
    mfq_tensor_backend::Tensor seq_len, double scale, int64_t planned_len,
    mfq_tensor_backend::Tensor mask, mfq_tensor_backend::Tensor kv_max, mfq_tensor_backend::Tensor meta);
mfq_tensor_backend::Tensor mfq_attention_mma512_decode_cuda(
    mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor k_cache, mfq_tensor_backend::Tensor v_cache,
    mfq_tensor_backend::Tensor seq_len, double scale, int64_t planned_len,
    mfq_tensor_backend::Tensor mask, mfq_tensor_backend::Tensor kv_max, mfq_tensor_backend::Tensor meta);
mfq_tensor_backend::Tensor attention_glm_mla576_cuda(
    mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor kv, double scale);
mfq_tensor_backend::Tensor attention_glm_mla576_cached_cuda(
    mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor kv_cache, int64_t logical_len,
    mfq_tensor_backend::Tensor mask, mfq_tensor_backend::Tensor kv_max, mfq_tensor_backend::Tensor meta,
    double scale);
mfq_tensor_backend::Tensor attention_glm_mla576_decode_cuda(
    mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor kv_cache, mfq_tensor_backend::Tensor seq_len,
    double scale, int64_t planned_len, mfq_tensor_backend::Tensor mask,
    mfq_tensor_backend::Tensor kv_max, mfq_tensor_backend::Tensor meta);
mfq_tensor_backend::Tensor glm_interleaved_rope_cuda(
    mfq_tensor_backend::Tensor x, mfq_tensor_backend::Tensor positions,
    mfq_tensor_backend::Tensor cos, mfq_tensor_backend::Tensor sin, int64_t rotary_dim);
mfq_tensor_backend::Tensor glm_dsa_indexer_layer_norm_cuda(
    mfq_tensor_backend::Tensor x, mfq_tensor_backend::Tensor weight, mfq_tensor_backend::Tensor bias, double eps);
mfq_tensor_backend::Tensor glm_dsa_cache_write_cuda(
    mfq_tensor_backend::Tensor cache, mfq_tensor_backend::Tensor values, mfq_tensor_backend::Tensor positions);
mfq_tensor_backend::Tensor glm_dsa_indexer_scores_cuda(
    mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor k, mfq_tensor_backend::Tensor weights,
    int64_t query_offset, int64_t logical_k);
mfq_tensor_backend::Tensor glm_dsa_indexer_scores_decode_cuda(
    mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor k, mfq_tensor_backend::Tensor weights,
    mfq_tensor_backend::Tensor seq_len, int64_t planned_k);
mfq_tensor_backend::Tensor attention_glm_mla_sparse_cuda(
    mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor kv, mfq_tensor_backend::Tensor indices,
    mfq_tensor_backend::Tensor meta, double scale);
mfq_tensor_backend::Tensor mfq_attention_mma256_swa_decode_cuda(
    mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor k_cache, mfq_tensor_backend::Tensor v_cache,
    mfq_tensor_backend::Tensor seq_len, double scale, int64_t planned_len,
    mfq_tensor_backend::Tensor mask, mfq_tensor_backend::Tensor kv_max, mfq_tensor_backend::Tensor meta);
mfq_tensor_backend::Tensor attention_cache_decode_cuda(mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor k_cache, mfq_tensor_backend::Tensor v_cache,
                                          mfq_tensor_backend::Tensor seq_len, double scale);
mfq_tensor_backend::Tensor attention_cache_decode_split_cuda(
    mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor k_cache, mfq_tensor_backend::Tensor v_cache,
    mfq_tensor_backend::Tensor seq_len, double scale,
    mfq_tensor_backend::Tensor partial_o, mfq_tensor_backend::Tensor partial_m, mfq_tensor_backend::Tensor partial_l,
    int64_t parts);
mfq_tensor_backend::Tensor attention_cache_decode_dynamic_cuda(
    mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor k_cache,
    mfq_tensor_backend::Tensor v_cache, mfq_tensor_backend::Tensor seq_len,
    double scale, mfq_tensor_backend::Tensor partial_o,
    mfq_tensor_backend::Tensor partial_m,
    mfq_tensor_backend::Tensor partial_l, int64_t max_parts);
mfq_tensor_backend::Tensor silu_mul_cuda(mfq_tensor_backend::Tensor gate, mfq_tensor_backend::Tensor up);
mfq_tensor_backend::Tensor gelu_mul_cuda(mfq_tensor_backend::Tensor gate, mfq_tensor_backend::Tensor up);
std::vector<mfq_tensor_backend::Tensor> moe_topk_cuda(
    mfq_tensor_backend::Tensor logits, int64_t top_k, bool use_sigmoid, bool use_sqrt_softplus,
    bool normalize,
    bool delayed_softmax, MfqOptional<mfq_tensor_backend::Tensor> bias, double norm_floor,
    double scale);
mfq_tensor_backend::Tensor moe_sqrtsoftplus_weights_cuda(
    mfq_tensor_backend::Tensor logits, mfq_tensor_backend::Tensor ids, double norm_floor, double scale);
std::vector<mfq_tensor_backend::Tensor> moe_build_expert_map_cuda(
    mfq_tensor_backend::Tensor ids, int64_t n_experts, int64_t tile_m);
std::vector<mfq_tensor_backend::Tensor> moe_build_expert_maps_cuda(
    mfq_tensor_backend::Tensor ids, int64_t n_experts, int64_t tile_m,
    int64_t secondary_tile_m, int64_t tertiary_tile_m);
mfq_tensor_backend::Tensor mfe_nint_matmul_ws_cuda(
    mfq_tensor_backend::Tensor q_packed, mfq_tensor_backend::Tensor row_q_bits,
    mfq_tensor_backend::Tensor row_q_bit_offsets,
    mfq_tensor_backend::Tensor sub_scale, mfq_tensor_backend::Tensor sub_min,
    mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor neuron_min,
    mfq_tensor_backend::Tensor x, mfq_tensor_backend::Tensor ids,
    mfq_tensor_backend::Tensor expert_local, int64_t n_experts,
    int64_t n_local_experts, int64_t out_per_expert, int64_t gs,
    int64_t epilogue_mode, bool route_map_ready, bool input_quantized,
    mfq_tensor_backend::Tensor out, mfq_tensor_backend::Tensor qx,
    mfq_tensor_backend::Tensor xscale,
    mfq_tensor_backend::Tensor ids_dst,
    mfq_tensor_backend::Tensor expert_bounds,
    mfq_tensor_backend::Tensor tile_bounds,
    mfq_tensor_backend::Tensor tile_experts,
    int64_t route_tile_m,
    int64_t pool_phase);
mfq_tensor_backend::Tensor nint8_zero_moe_grouped_matmul_pool_ws_cuda(
    mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor scale, mfq_tensor_backend::Tensor x,
    mfq_tensor_backend::Tensor ids, mfq_tensor_backend::Tensor expert_local, int64_t n_experts,
    int64_t n_local_experts, int64_t out_per_expert, bool route_map_ready,
    bool input_quantized, bool use_f16_mma, mfq_tensor_backend::Tensor out, mfq_tensor_backend::Tensor qx,
    mfq_tensor_backend::Tensor xscale, mfq_tensor_backend::Tensor counts, mfq_tensor_backend::Tensor cursors,
    mfq_tensor_backend::Tensor ids_dst, mfq_tensor_backend::Tensor expert_bounds,
    mfq_tensor_backend::Tensor tile_bounds, mfq_tensor_backend::Tensor tile_experts,
    int64_t route_tile_m);
mfq_tensor_backend::Tensor nvq_moe_grouped_matmul_pool_ws_cuda(
    mfq_tensor_backend::Tensor indices, mfq_tensor_backend::Tensor aux, mfq_tensor_backend::Tensor sub_scale,
    mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor codebook, mfq_tensor_backend::Tensor x,
    mfq_tensor_backend::Tensor ids, mfq_tensor_backend::Tensor expert_local, int64_t n_experts,
    int64_t pool_experts, int64_t out_per_expert, int64_t neuron_len,
    int64_t gs, int64_t sub_bits, int64_t format, int64_t sign_mode,
    bool input_quantized, mfq_tensor_backend::Tensor out, mfq_tensor_backend::Tensor qx,
    mfq_tensor_backend::Tensor xscale, mfq_tensor_backend::Tensor ids_dst, mfq_tensor_backend::Tensor expert_bounds,
    mfq_tensor_backend::Tensor tile_bounds, mfq_tensor_backend::Tensor tile_experts);
mfq_tensor_backend::Tensor nvq_moe_grouped_matmul_pool_f16_cuda(
    mfq_tensor_backend::Tensor indices, mfq_tensor_backend::Tensor aux, mfq_tensor_backend::Tensor sub_scale,
    mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor codebook, mfq_tensor_backend::Tensor x,
    mfq_tensor_backend::Tensor expert_local, int64_t n_experts, int64_t pool_experts,
    int64_t out_per_expert, int64_t neuron_len, int64_t gs,
    int64_t sub_bits, int64_t format, int64_t sign_mode,
    mfq_tensor_backend::Tensor out, mfq_tensor_backend::Tensor ids_dst, mfq_tensor_backend::Tensor expert_bounds,
    mfq_tensor_backend::Tensor tile_bounds, mfq_tensor_backend::Tensor tile_experts);
mfq_tensor_backend::Tensor nvq_moe_grouped_matmul_hetero_f16_cuda(
    mfq_tensor_backend::Tensor weight_ptrs, mfq_tensor_backend::Tensor weight_sizes,
    mfq_tensor_backend::Tensor pool_params, mfq_tensor_backend::Tensor expert_pool,
    mfq_tensor_backend::Tensor expert_local, mfq_tensor_backend::Tensor x,
    int64_t n_experts, int64_t out_per_expert, int64_t neuron_len,
    int64_t route_tile_m, mfq_tensor_backend::Tensor out,
    mfq_tensor_backend::Tensor ids_dst,
    mfq_tensor_backend::Tensor expert_bounds, mfq_tensor_backend::Tensor tile_bounds,
    mfq_tensor_backend::Tensor tile_experts, int64_t format_group,
    bool masked_experts);
mfq_tensor_backend::Tensor nvq_moe_grouped_matmul_hetero_ws_cuda(
    mfq_tensor_backend::Tensor weight_ptrs, mfq_tensor_backend::Tensor weight_sizes,
    mfq_tensor_backend::Tensor pool_params, mfq_tensor_backend::Tensor expert_pool,
    mfq_tensor_backend::Tensor expert_local, mfq_tensor_backend::Tensor x,
    mfq_tensor_backend::Tensor ids, int64_t n_experts, int64_t out_per_expert,
    int64_t neuron_len, bool input_quantized, mfq_tensor_backend::Tensor out,
    mfq_tensor_backend::Tensor qx, mfq_tensor_backend::Tensor xscale);
mfq_tensor_backend::Tensor nepq_moe_grouped_matmul_pool_ws_cuda(
    mfq_tensor_backend::Tensor indices, mfq_tensor_backend::Tensor aux, mfq_tensor_backend::Tensor sub_scale,
    mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor table_pool, mfq_tensor_backend::Tensor bank_ids,
    mfq_tensor_backend::Tensor grouped_table_pool, mfq_tensor_backend::Tensor x, mfq_tensor_backend::Tensor ids,
    mfq_tensor_backend::Tensor expert_local, int64_t n_experts, int64_t pool_experts,
    int64_t out_per_expert, int64_t neuron_len, int64_t sub_bits, int64_t format,
    bool input_quantized, mfq_tensor_backend::Tensor out, mfq_tensor_backend::Tensor qx,
    mfq_tensor_backend::Tensor xscale, mfq_tensor_backend::Tensor ids_dst, mfq_tensor_backend::Tensor expert_bounds,
    mfq_tensor_backend::Tensor tile_bounds, mfq_tensor_backend::Tensor tile_experts);
mfq_tensor_backend::Tensor nepq_moe_grouped_matmul_pool_f16_cuda(
    mfq_tensor_backend::Tensor indices, mfq_tensor_backend::Tensor aux, mfq_tensor_backend::Tensor sub_scale,
    mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor table_pool,
    mfq_tensor_backend::Tensor bank_ids, mfq_tensor_backend::Tensor x, mfq_tensor_backend::Tensor expert_local,
    int64_t n_experts, int64_t pool_experts, int64_t out_per_expert,
    int64_t neuron_len, int64_t sub_bits, int64_t format,
    mfq_tensor_backend::Tensor out, mfq_tensor_backend::Tensor ids_dst, mfq_tensor_backend::Tensor expert_bounds,
    mfq_tensor_backend::Tensor tile_bounds, mfq_tensor_backend::Tensor tile_experts);
mfq_tensor_backend::Tensor nepq_hadamard_input_cuda(
    mfq_tensor_backend::Tensor input, mfq_tensor_backend::Tensor signs, int64_t block_size);
mfq_tensor_backend::Tensor nepq_sparse_residual_grouped_cuda(
    mfq_tensor_backend::Tensor dictionary, mfq_tensor_backend::Tensor first, mfq_tensor_backend::Tensor second,
    mfq_tensor_backend::Tensor input, mfq_tensor_backend::Tensor route_ids, mfq_tensor_backend::Tensor expert_local,
    int64_t out_per_expert, int64_t position_bits, int64_t block_vectors,
    mfq_tensor_backend::Tensor output);
mfq_tensor_backend::Tensor moe_weighted_reduce_cuda(mfq_tensor_backend::Tensor pair_output, mfq_tensor_backend::Tensor weights);
mfq_tensor_backend::Tensor moe_swiglu_split_cuda(mfq_tensor_backend::Tensor gate_up);
mfq_tensor_backend::Tensor moe_geglu_split_cuda(mfq_tensor_backend::Tensor gate_up);
mfq_tensor_backend::Tensor moe_apply_expert_scale_cuda(
    mfq_tensor_backend::Tensor weights, mfq_tensor_backend::Tensor ids, mfq_tensor_backend::Tensor scales);
mfq_tensor_backend::Tensor moe_add_shared_gate_cuda(
    mfq_tensor_backend::Tensor routed, mfq_tensor_backend::Tensor shared, mfq_tensor_backend::Tensor gate_logits);
mfq_tensor_backend::Tensor moe_weighted_reduce_shared_gate_cuda(
    mfq_tensor_backend::Tensor pair_output, mfq_tensor_backend::Tensor weights,
    mfq_tensor_backend::Tensor shared, mfq_tensor_backend::Tensor gate_logits);
std::vector<mfq_tensor_backend::Tensor> gdn_cuda(mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor k, mfq_tensor_backend::Tensor v,
                                    mfq_tensor_backend::Tensor g, mfq_tensor_backend::Tensor beta, MfqOptional<mfq_tensor_backend::Tensor> state);
std::vector<mfq_tensor_backend::Tensor> gdn_inplace_cuda(mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor k, mfq_tensor_backend::Tensor v,
                                             mfq_tensor_backend::Tensor g, mfq_tensor_backend::Tensor beta, mfq_tensor_backend::Tensor state);
std::vector<mfq_tensor_backend::Tensor> gdn_inplace_transposed_cuda(
    mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor k, mfq_tensor_backend::Tensor v,
    mfq_tensor_backend::Tensor g, mfq_tensor_backend::Tensor beta, mfq_tensor_backend::Tensor state);
std::vector<mfq_tensor_backend::Tensor> gdn_inplace_tiled_cuda(
    mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor k, mfq_tensor_backend::Tensor v,
    mfq_tensor_backend::Tensor g, mfq_tensor_backend::Tensor beta, mfq_tensor_backend::Tensor state);
std::vector<mfq_tensor_backend::Tensor> gdn_inplace_transposed_tiled_cuda(
    mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor k, mfq_tensor_backend::Tensor v,
    mfq_tensor_backend::Tensor g, mfq_tensor_backend::Tensor beta, mfq_tensor_backend::Tensor state);
std::vector<mfq_tensor_backend::Tensor> kv_cache_write_cuda(mfq_tensor_backend::Tensor k_cache, mfq_tensor_backend::Tensor v_cache,
                                               mfq_tensor_backend::Tensor k, mfq_tensor_backend::Tensor v, mfq_tensor_backend::Tensor positions);
std::vector<mfq_tensor_backend::Tensor> kv_cache_write_ring_cuda(
    mfq_tensor_backend::Tensor k_cache, mfq_tensor_backend::Tensor v_cache,
    mfq_tensor_backend::Tensor k, mfq_tensor_backend::Tensor v, int64_t position_start);
std::vector<mfq_tensor_backend::Tensor> kv_cache_write_ring_positions_cuda(
    mfq_tensor_backend::Tensor k_cache, mfq_tensor_backend::Tensor v_cache,
    mfq_tensor_backend::Tensor k, mfq_tensor_backend::Tensor v, mfq_tensor_backend::Tensor positions);
mfq_tensor_backend::Tensor ssm_conv_silu_cuda(mfq_tensor_backend::Tensor conv_input, mfq_tensor_backend::Tensor weight, mfq_tensor_backend::Tensor bias, int64_t n_tokens);
mfq_tensor_backend::Tensor ssm_conv_silu_decode_cuda(mfq_tensor_backend::Tensor state, mfq_tensor_backend::Tensor x, mfq_tensor_backend::Tensor weight, mfq_tensor_backend::Tensor bias);
std::vector<mfq_tensor_backend::Tensor> linear_conv_qkv_decode_cuda(
    mfq_tensor_backend::Tensor state, mfq_tensor_backend::Tensor qk, mfq_tensor_backend::Tensor v, mfq_tensor_backend::Tensor weight, mfq_tensor_backend::Tensor bias,
    int64_t nk, int64_t nv, int64_t dk, int64_t dv, double eps);
std::vector<mfq_tensor_backend::Tensor> linear_conv_qkv_prefill_cuda(
    mfq_tensor_backend::Tensor state, mfq_tensor_backend::Tensor qk, mfq_tensor_backend::Tensor v, mfq_tensor_backend::Tensor weight, mfq_tensor_backend::Tensor bias,
    int64_t nk, int64_t nv, int64_t dk, int64_t dv, double eps);
mfq_tensor_backend::Tensor nint_embedding_cuda(
    mfq_tensor_backend::Tensor q_packed, mfq_tensor_backend::Tensor row_q_bits,
    mfq_tensor_backend::Tensor row_q_bit_offsets,
    mfq_tensor_backend::Tensor sub_scale, mfq_tensor_backend::Tensor sub_min,
    mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor neuron_min,
    mfq_tensor_backend::Tensor token_ids, int64_t neuron_len, int64_t gs);
mfq_tensor_backend::Tensor nint8_zero_embedding_lookup_cuda(
    mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor scale, mfq_tensor_backend::Tensor token_ids,
    int64_t neuron_len);
mfq_tensor_backend::Tensor sample_greedy_cuda(mfq_tensor_backend::Tensor logits);
mfq_tensor_backend::Tensor sample_softmax_cuda(mfq_tensor_backend::Tensor logits, mfq_tensor_backend::Tensor random, double temperature);
mfq_tensor_backend::Tensor sample_top_k_top_p_cuda(
    mfq_tensor_backend::Tensor logits, mfq_tensor_backend::Tensor random, double temperature, int64_t top_k, double top_p);
void sample_token_counts_add_cuda(mfq_tensor_backend::Tensor counts, mfq_tensor_backend::Tensor tokens);
mfq_tensor_backend::Tensor sample_apply_penalties_cuda(
    mfq_tensor_backend::Tensor logits, mfq_tensor_backend::Tensor counts,
    double presence_penalty, double frequency_penalty, double repetition_penalty);
std::vector<mfq_tensor_backend::Tensor> nint8_one_quantize_reconstruct_cuda(
    mfq_tensor_backend::Tensor x);
mfq_tensor_backend::Tensor nint8_zero_mmq_f16_packed_cuda(
    mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor scale, mfq_tensor_backend::Tensor x,
    int64_t neuron_len);
mfq_tensor_backend::Tensor nint8_zero_mmq_f32_packed_cuda(
    mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor scale, mfq_tensor_backend::Tensor x,
    int64_t neuron_len);
mfq_tensor_backend::Tensor nint_matmul_ws_cuda(
    mfq_tensor_backend::Tensor q_packed, mfq_tensor_backend::Tensor row_q_bits,
    mfq_tensor_backend::Tensor row_q_bit_offsets,
    mfq_tensor_backend::Tensor sub_scale, mfq_tensor_backend::Tensor sub_min,
    mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor neuron_min,
    mfq_tensor_backend::Tensor x, int64_t gs,
    mfq_tensor_backend::Tensor qx, mfq_tensor_backend::Tensor xscale);
mfq_tensor_backend::Tensor nint_matmul_input_mul_ws_cuda(
    mfq_tensor_backend::Tensor q_packed, mfq_tensor_backend::Tensor row_q_bits,
    mfq_tensor_backend::Tensor row_q_bit_offsets,
    mfq_tensor_backend::Tensor sub_scale, mfq_tensor_backend::Tensor sub_min,
    mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor neuron_min,
    mfq_tensor_backend::Tensor x, mfq_tensor_backend::Tensor gate,
    int64_t activation_mode, int64_t gs,
    mfq_tensor_backend::Tensor qx, mfq_tensor_backend::Tensor xscale);
mfq_tensor_backend::Tensor nint8_zero_gemv_ws_cuda(
    mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor scale, mfq_tensor_backend::Tensor x,
    mfq_tensor_backend::Tensor qx, mfq_tensor_backend::Tensor xscale);
mfq_tensor_backend::Tensor nint8_zero_dequant_cuda(
    mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor scale, int64_t neuron_len);
mfq_tensor_backend::Tensor nint_decode_cuda(
    mfq_tensor_backend::Tensor q_packed, mfq_tensor_backend::Tensor row_q_bits,
    mfq_tensor_backend::Tensor row_q_bit_offsets,
    mfq_tensor_backend::Tensor sub_scale, mfq_tensor_backend::Tensor sub_min,
    mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor neuron_min,
    int64_t neuron_len, int64_t gs);
mfq_tensor_backend::Tensor nint_cublas_gemm_nt_f32acc_cuda(mfq_tensor_backend::Tensor x, mfq_tensor_backend::Tensor w);
mfq_tensor_backend::Tensor mxfp8_dequant_cuda(
    mfq_tensor_backend::Tensor values, mfq_tensor_backend::Tensor scales);
mfq_tensor_backend::Tensor mxfp8_embedding_lookup_cuda(
    mfq_tensor_backend::Tensor values, mfq_tensor_backend::Tensor scales, mfq_tensor_backend::Tensor token_ids);
mfq_tensor_backend::Tensor mxfp8_small_m_cuda(
    mfq_tensor_backend::Tensor values, mfq_tensor_backend::Tensor scales, mfq_tensor_backend::Tensor x);
mfq_tensor_backend::Tensor mxfp8_matmul_f16_cuda(
    mfq_tensor_backend::Tensor values, mfq_tensor_backend::Tensor scales, mfq_tensor_backend::Tensor input);
mfq_tensor_backend::Tensor mxfp8_small_m_f32_cuda(
    mfq_tensor_backend::Tensor values, mfq_tensor_backend::Tensor scales, mfq_tensor_backend::Tensor x);
mfq_tensor_backend::Tensor mxfp8_gemm_f32_cuda(
    mfq_tensor_backend::Tensor values, mfq_tensor_backend::Tensor scales, mfq_tensor_backend::Tensor x);
mfq_tensor_backend::Tensor mxfp8_groupwise_small_m_cuda(
    mfq_tensor_backend::Tensor values, mfq_tensor_backend::Tensor scales,
    mfq_tensor_backend::Tensor x, int64_t groups);
mfq_tensor_backend::Tensor mxfp8_groupwise_small_m_f32_cuda(
    mfq_tensor_backend::Tensor values, mfq_tensor_backend::Tensor scales,
    mfq_tensor_backend::Tensor x, int64_t groups);
mfq_tensor_backend::Tensor mxfp4_dequant_cuda(
    mfq_tensor_backend::Tensor values, mfq_tensor_backend::Tensor scales);
mfq_tensor_backend::Tensor mxfp4_embedding_lookup_cuda(
    mfq_tensor_backend::Tensor values, mfq_tensor_backend::Tensor scales, mfq_tensor_backend::Tensor token_ids);
mfq_tensor_backend::Tensor mxfp4_matmul_f16_cuda(
    mfq_tensor_backend::Tensor values, mfq_tensor_backend::Tensor scales, mfq_tensor_backend::Tensor input);
mfq_tensor_backend::Tensor mxfp4_moe_grouped_matmul_pool_f16_cuda(
    mfq_tensor_backend::Tensor values, mfq_tensor_backend::Tensor scales, mfq_tensor_backend::Tensor input,
    mfq_tensor_backend::Tensor ids, mfq_tensor_backend::Tensor expert_local,
    int64_t global_experts, int64_t pool_experts,
    int64_t out_per_expert, int64_t neuron_len,
    mfq_tensor_backend::Tensor output, mfq_tensor_backend::Tensor ids_dst,
    mfq_tensor_backend::Tensor expert_bounds, mfq_tensor_backend::Tensor tile_bounds,
    mfq_tensor_backend::Tensor tile_experts);
mfq_tensor_backend::Tensor nvq_dequant_cuda(
    mfq_tensor_backend::Tensor indices, mfq_tensor_backend::Tensor aux, mfq_tensor_backend::Tensor sub_scale,
    mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor codebook,
    int64_t neuron_len, int64_t gs, int64_t sub_bits, int64_t format, int64_t sign_mode);
mfq_tensor_backend::Tensor nepq_dequant_cuda(
    mfq_tensor_backend::Tensor indices, mfq_tensor_backend::Tensor aux, mfq_tensor_backend::Tensor sub_scale,
    mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor table_pool,
    mfq_tensor_backend::Tensor bank_ids, int64_t neuron_len,
    int64_t sub_bits, int64_t format);
mfq_tensor_backend::Tensor nepq_sparse_residual_dequant_cuda(
    mfq_tensor_backend::Tensor dictionary, mfq_tensor_backend::Tensor first, mfq_tensor_backend::Tensor second,
    int64_t position_bits, int64_t block_vectors, mfq_tensor_backend::Tensor weight);
mfq_tensor_backend::Tensor nvq_gemm_f16_cuda(
    mfq_tensor_backend::Tensor indices, mfq_tensor_backend::Tensor aux, mfq_tensor_backend::Tensor sub_scale,
    mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor codebook, mfq_tensor_backend::Tensor x,
    int64_t neuron_len, int64_t gs, int64_t sub_bits, int64_t format, int64_t sign_mode);
mfq_tensor_backend::Tensor nvq_gemv_ws_cuda(
    mfq_tensor_backend::Tensor indices, mfq_tensor_backend::Tensor aux, mfq_tensor_backend::Tensor sub_scale,
    mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor codebook, mfq_tensor_backend::Tensor x,
    int64_t neuron_len, int64_t gs, int64_t sub_bits, int64_t format, int64_t sign_mode,
    mfq_tensor_backend::Tensor qx, mfq_tensor_backend::Tensor xscale);
mfq_tensor_backend::Tensor nvq_gemv_qx_ws_cuda(
    mfq_tensor_backend::Tensor indices, mfq_tensor_backend::Tensor aux, mfq_tensor_backend::Tensor sub_scale,
    mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor codebook,
    int64_t neuron_len, int64_t gs, int64_t sub_bits, int64_t format, int64_t sign_mode,
    mfq_tensor_backend::Tensor qx, mfq_tensor_backend::Tensor xscale);
mfq_tensor_backend::Tensor nvq_gemv_qx_residual_ws_cuda(
    mfq_tensor_backend::Tensor indices, mfq_tensor_backend::Tensor aux, mfq_tensor_backend::Tensor sub_scale,
    mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor codebook,
    int64_t neuron_len, int64_t gs, int64_t sub_bits, int64_t format,
    int64_t sign_mode, mfq_tensor_backend::Tensor qx, mfq_tensor_backend::Tensor xscale,
    mfq_tensor_backend::Tensor residual);
mfq_tensor_backend::Tensor nvq_gemv_multi2_ws_cuda(
    mfq_tensor_backend::Tensor first_indices, mfq_tensor_backend::Tensor first_aux, mfq_tensor_backend::Tensor first_sub_scale,
    mfq_tensor_backend::Tensor first_neuron_scale, mfq_tensor_backend::Tensor first_codebook,
    mfq_tensor_backend::Tensor second_indices, mfq_tensor_backend::Tensor second_aux, mfq_tensor_backend::Tensor second_sub_scale,
    mfq_tensor_backend::Tensor second_neuron_scale, mfq_tensor_backend::Tensor second_codebook,
    mfq_tensor_backend::Tensor x, int64_t neuron_len, int64_t gs,
    int64_t first_sub_bits, int64_t first_format, int64_t first_sign_mode,
    int64_t second_sub_bits, int64_t second_format, int64_t second_sign_mode,
    mfq_tensor_backend::Tensor qx, mfq_tensor_backend::Tensor xscale);
mfq_tensor_backend::Tensor nvq_gemv_swiglu_ws_cuda(
    mfq_tensor_backend::Tensor gate_indices, mfq_tensor_backend::Tensor gate_aux, mfq_tensor_backend::Tensor gate_sub_scale,
    mfq_tensor_backend::Tensor gate_neuron_scale, mfq_tensor_backend::Tensor gate_codebook,
    mfq_tensor_backend::Tensor up_indices, mfq_tensor_backend::Tensor up_aux, mfq_tensor_backend::Tensor up_sub_scale,
    mfq_tensor_backend::Tensor up_neuron_scale, mfq_tensor_backend::Tensor up_codebook,
    mfq_tensor_backend::Tensor x, int64_t neuron_len, int64_t gs,
    int64_t gate_sub_bits, int64_t gate_format, int64_t gate_sign_mode,
    int64_t up_sub_bits, int64_t up_format, int64_t up_sign_mode,
    mfq_tensor_backend::Tensor qx, mfq_tensor_backend::Tensor xscale);
void nvq_ffn_swiglu_quant_ws_cuda(
    mfq_tensor_backend::Tensor gate_indices, mfq_tensor_backend::Tensor gate_aux, mfq_tensor_backend::Tensor gate_sub_scale,
    mfq_tensor_backend::Tensor gate_neuron_scale, mfq_tensor_backend::Tensor gate_codebook,
    mfq_tensor_backend::Tensor up_indices, mfq_tensor_backend::Tensor up_aux, mfq_tensor_backend::Tensor up_sub_scale,
    mfq_tensor_backend::Tensor up_neuron_scale, mfq_tensor_backend::Tensor up_codebook,
    mfq_tensor_backend::Tensor x, int64_t neuron_len, int64_t gs,
    int64_t gate_sub_bits, int64_t gate_format, int64_t gate_sign_mode,
    int64_t up_sub_bits, int64_t up_format, int64_t up_sign_mode, int64_t down_gs,
    mfq_tensor_backend::Tensor input_qx, mfq_tensor_backend::Tensor input_xscale,
    mfq_tensor_backend::Tensor output_qx, mfq_tensor_backend::Tensor output_xscale, mfq_tensor_backend::Tensor swiglu_scratch);
mfq_tensor_backend::Tensor nvq_gemv_gate_ws_cuda(
    mfq_tensor_backend::Tensor indices, mfq_tensor_backend::Tensor aux, mfq_tensor_backend::Tensor sub_scale,
    mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor codebook, mfq_tensor_backend::Tensor x, mfq_tensor_backend::Tensor gate,
    int64_t neuron_len, int64_t gs, int64_t sub_bits, int64_t format, int64_t sign_mode,
    int64_t mode, mfq_tensor_backend::Tensor qx, mfq_tensor_backend::Tensor xscale);
mfq_tensor_backend::Tensor nvq_mmq_ws_cuda(
    mfq_tensor_backend::Tensor indices, mfq_tensor_backend::Tensor aux, mfq_tensor_backend::Tensor sub_scale,
    mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor codebook, mfq_tensor_backend::Tensor x,
    int64_t neuron_len, int64_t gs, int64_t sub_bits, int64_t format, int64_t sign_mode,
    mfq_tensor_backend::Tensor qx, mfq_tensor_backend::Tensor xscale);
mfq_tensor_backend::Tensor nvq_mmq_gate_ws_cuda(
    mfq_tensor_backend::Tensor indices, mfq_tensor_backend::Tensor aux, mfq_tensor_backend::Tensor sub_scale,
    mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor codebook, mfq_tensor_backend::Tensor x, mfq_tensor_backend::Tensor gate,
    int64_t neuron_len, int64_t gs, int64_t sub_bits, int64_t format, int64_t sign_mode,
    int64_t mode, mfq_tensor_backend::Tensor qx, mfq_tensor_backend::Tensor xscale);
mfq_tensor_backend::Tensor nvq_embedding_lookup_cuda(
    mfq_tensor_backend::Tensor indices, mfq_tensor_backend::Tensor aux, mfq_tensor_backend::Tensor sub_scale,
    mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor codebook, mfq_tensor_backend::Tensor token_ids,
    int64_t neuron_len, int64_t gs, int64_t sub_bits, int64_t format, int64_t sign_mode);
