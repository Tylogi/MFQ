// Stub definitions for kernel entry points whose NVIDIA tensor-core tile
// implementations are not yet ported to HIP. Each stub fails loudly if a
// model graph selects the corresponding path; they exist so the runtime
// links while those kernels gain MFMA or direct implementations.
#include "hip/hip_runtime.h"
#include "mfq_tensor_backend.h"

#include <stdexcept>
// The kernel entry points are declared at global scope in their headers.

mfq_tensor_backend::Tensor deepseek_v41_mxfp4_e4m3_scale_sim_cuda( mfq_tensor_backend::Tensor input) {
    throw std::runtime_error("deepseek_v41_mxfp4_e4m3_scale_sim_cuda is not yet available on HIP");
    return {};
}

mfq_tensor_backend::Tensor deepseek_v41_mxfp8_e4m3_sim_cuda( mfq_tensor_backend::Tensor input) {
    throw std::runtime_error("deepseek_v41_mxfp8_e4m3_sim_cuda is not yet available on HIP");
    return {};
}

mfq_tensor_backend::Tensor glm_dsa_cache_write_cuda( mfq_tensor_backend::Tensor cache, mfq_tensor_backend::Tensor values, mfq_tensor_backend::Tensor positions) {
    throw std::runtime_error("glm_dsa_cache_write_cuda is not yet available on HIP");
    return {};
}

mfq_tensor_backend::Tensor glm_interleaved_rope_cuda( mfq_tensor_backend::Tensor x, mfq_tensor_backend::Tensor positions, mfq_tensor_backend::Tensor cos, mfq_tensor_backend::Tensor sin, int64_t rotary_dim) {
    throw std::runtime_error("glm_interleaved_rope_cuda is not yet available on HIP");
    return {};
}

std::vector<mfq_tensor_backend::Tensor> dsv4_build_decode_plan_cuda( mfq_tensor_backend::Tensor topk, mfq_tensor_backend::Tensor seq_len, std::int64_t pool_len, std::int64_t ratio, std::int64_t window) {
    throw std::runtime_error("dsv4_build_decode_plan_cuda is not yet available on HIP");
    return {};
}

std::vector<mfq_tensor_backend::Tensor> dsv4_build_prefill_plan_cuda( mfq_tensor_backend::Tensor topk, std::int64_t query_offset, std::int64_t local_history, std::int64_t pool_len, std::int64_t ratio, std::int64_t window) {
    throw std::runtime_error("dsv4_build_prefill_plan_cuda is not yet available on HIP");
    return {};
}

mfq_tensor_backend::Tensor dsv4_compress_cuda( mfq_tensor_backend::Tensor kv, mfq_tensor_backend::Tensor gate, mfq_tensor_backend::Tensor ape, mfq_tensor_backend::Tensor norm, mfq_tensor_backend::Tensor prev_kv, mfq_tensor_backend::Tensor prev_gate, mfq_tensor_backend::Tensor positions, mfq_tensor_backend::Tensor cos, mfq_tensor_backend::Tensor sin, std::int64_t ratio, bool overlap, std::int64_t quant_mode, double eps) {
    throw std::runtime_error("dsv4_compress_cuda is not yet available on HIP");
    return {};
}

mfq_tensor_backend::Tensor dsv4_decode_pool_update_cuda( mfq_tensor_backend::Tensor kv_token, mfq_tensor_backend::Tensor gate_token, mfq_tensor_backend::Tensor ape, mfq_tensor_backend::Tensor norm, mfq_tensor_backend::Tensor state_kv, mfq_tensor_backend::Tensor state_gate, mfq_tensor_backend::Tensor prev_kv, mfq_tensor_backend::Tensor prev_gate, mfq_tensor_backend::Tensor pool, mfq_tensor_backend::Tensor seq_len, mfq_tensor_backend::Tensor cos, mfq_tensor_backend::Tensor sin, std::int64_t ratio, bool overlap, std::int64_t quant_mode, double eps) {
    throw std::runtime_error("dsv4_decode_pool_update_cuda is not yet available on HIP");
    return {};
}

mfq_tensor_backend::Tensor dsv4_fp4_sim_cuda( mfq_tensor_backend::Tensor input) {
    throw std::runtime_error("dsv4_fp4_sim_cuda is not yet available on HIP");
    return {};
}

mfq_tensor_backend::Tensor dsv4_hc_post_cuda( mfq_tensor_backend::Tensor x, mfq_tensor_backend::Tensor residual, mfq_tensor_backend::Tensor post, mfq_tensor_backend::Tensor combination) {
    throw std::runtime_error("dsv4_hc_post_cuda is not yet available on HIP");
    return {};
}

std::vector<mfq_tensor_backend::Tensor> dsv4_hc_pre_cuda( mfq_tensor_backend::Tensor x, mfq_tensor_backend::Tensor mixes, mfq_tensor_backend::Tensor scale, mfq_tensor_backend::Tensor base, std::int64_t iterations, double eps) {
    throw std::runtime_error("dsv4_hc_pre_cuda is not yet available on HIP");
    return {};
}

mfq_tensor_backend::Tensor dsv4_indexer_scores_cuda( mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor k, mfq_tensor_backend::Tensor weights, std::int64_t query_offset, std::int64_t ratio) {
    throw std::runtime_error("dsv4_indexer_scores_cuda is not yet available on HIP");
    return {};
}

mfq_tensor_backend::Tensor dsv4_topk512_cuda( mfq_tensor_backend::Tensor scores) {
    throw std::runtime_error("dsv4_topk512_cuda is not yet available on HIP");
    return {};
}

mfq_tensor_backend::Tensor attention_dsv4_sparse_cuda( mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor kv, mfq_tensor_backend::Tensor indices, mfq_tensor_backend::Tensor mask, mfq_tensor_backend::Tensor sinks, mfq_tensor_backend::Tensor meta, double scale) {
    throw std::runtime_error("attention_dsv4_sparse_cuda is not yet available on HIP");
    return {};
}

mfq_tensor_backend::Tensor attention_glm_mla_sparse_cuda( mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor kv, mfq_tensor_backend::Tensor indices, mfq_tensor_backend::Tensor meta, double scale) {
    throw std::runtime_error("attention_glm_mla_sparse_cuda is not yet available on HIP");
    return {};
}

mfq_tensor_backend::Tensor glm_dsa_indexer_layer_norm_cuda( mfq_tensor_backend::Tensor x, mfq_tensor_backend::Tensor weight, mfq_tensor_backend::Tensor bias, double eps) {
    throw std::runtime_error("glm_dsa_indexer_layer_norm_cuda is not yet available on HIP");
    return {};
}

mfq_tensor_backend::Tensor glm_dsa_indexer_scores_cuda( mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor k, mfq_tensor_backend::Tensor weights, std::int64_t query_offset, std::int64_t ratio) {
    throw std::runtime_error("glm_dsa_indexer_scores_cuda is not yet available on HIP");
    return {};
}

mfq_tensor_backend::Tensor glm_dsa_indexer_scores_decode_cuda( mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor k, mfq_tensor_backend::Tensor weights, mfq_tensor_backend::Tensor scales, std::int64_t ratio) {
    throw std::runtime_error("glm_dsa_indexer_scores_decode_cuda is not yet available on HIP");
    return {};
}
