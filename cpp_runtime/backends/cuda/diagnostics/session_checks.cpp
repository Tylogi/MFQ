#include "model_checks.h"

#include "models/deepseek_v4/ops.h"
#include "models/glm_dsa/ops.h"
#include "storage/session_state.h"

#include <cstdint>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

int run_text_session_state_check() {
    MfqCudaGuard guard(0);
    auto cuda_float = mfq_tensor_backend::TensorOptions()
        .device(mfq_tensor_backend::kCUDA).dtype(mfq_tensor_backend::kFloat32);
    const auto values = [&](std::vector<int64_t> shape,
                            mfq_tensor_backend::ScalarType dtype,
                            float offset) {
        int64_t elements = 1;
        for (int64_t extent : shape) elements *= extent;
        return (mfq_tensor_backend::arange(elements, cuda_float) + offset)
            .reshape(shape).to(dtype).contiguous();
    };
    const auto require_equal = [](bool condition, const char * message) {
        if (!condition) throw std::runtime_error(message);
    };

    mfq::cuda::DeepseekV4CausalLm dsv4_model;
    auto dsv4 = std::make_unique<Dsv4Block>();
    auto * dsv4_block = dsv4.get();
    dsv4_block->cuda_device = 0;
    dsv4_block->shared_state = std::make_shared<Dsv4SharedState>();
    dsv4_block->shared_state->attention_meta = mfq_tensor_backend::empty({1}, cuda_float);
    dsv4_block->shared_state->hadamard_signs = mfq_tensor_backend::ones(
        {128}, cuda_float.dtype(mfq_tensor_backend::kInt8));
    dsv4_block->local_cache = values(
        {1, 128, 8}, mfq_tensor_backend::kFloat16, 1.0f);
    const auto initialize_dsv4_pool = [&](Dsv4PoolState & pool,
                                          int64_t head_dim,
                                          bool overlap,
                                          float offset) {
        pool.ratio = 4;
        pool.head_dim = head_dim;
        pool.overlap = overlap;
        pool.cache_quant_mode = overlap ? 2 : 0;
        pool.capacity = 16;
        const int64_t state_width = overlap ? 2 * head_dim : head_dim;
        pool.state_kv = values(
            {1, pool.ratio, state_width}, mfq_tensor_backend::kFloat32, offset);
        pool.state_gate = values(
            {1, pool.ratio, state_width}, mfq_tensor_backend::kFloat32, offset + 100.0f);
        if (overlap) {
            pool.previous_kv = values(
                {1, pool.ratio, head_dim}, mfq_tensor_backend::kFloat32, offset + 200.0f);
            pool.previous_gate = values(
                {1, pool.ratio, head_dim}, mfq_tensor_backend::kFloat32, offset + 300.0f);
        }
        pool.pool = values(
            {1, pool.capacity, head_dim}, mfq_tensor_backend::kFloat16, offset + 400.0f);
    };
    initialize_dsv4_pool(
        dsv4_block->compressor, 8, false, 10.0f);
    initialize_dsv4_pool(
        dsv4_block->indexer_compressor, 4, true, 20.0f);
    dsv4_model.blocks.push_back(std::move(dsv4));
    dsv4_model.cache_pos = 10;
    std::vector<int64_t> tokens(10);
    std::iota(tokens.begin(), tokens.end(), 0);
    const TextSessionState dsv4_state =
        dsv4_model.capture_text_session_state(tokens);
    dsv4_block->local_cache.zero_();
    dsv4_block->compressor.state_kv.zero_();
    dsv4_block->compressor.state_gate.zero_();
    dsv4_block->compressor.pool.zero_();
    dsv4_block->indexer_compressor.state_kv.zero_();
    dsv4_block->indexer_compressor.state_gate.zero_();
    dsv4_block->indexer_compressor.previous_kv.zero_();
    dsv4_block->indexer_compressor.previous_gate.zero_();
    dsv4_block->indexer_compressor.pool.zero_();
    dsv4_model.cache_pos = 0;
    dsv4_model.restore_text_session_state(dsv4_state);
    const auto & saved_dsv4 = std::get<
        std::vector<Dsv4BlockSessionState>>(dsv4_state.payload).at(0);
    require_equal(
        mfq_tensor_backend::equal(dsv4_block->local_cache, saved_dsv4.local_cache),
        "DeepSeek V4 local session cache restore failed");
    const auto check_dsv4_pool = [&](const Dsv4PoolState & restored,
                                     const Dsv4PoolSessionState & saved) {
        require_equal(
            mfq_tensor_backend::equal(restored.state_kv, saved.state_kv) &&
            mfq_tensor_backend::equal(restored.state_gate, saved.state_gate),
            "DeepSeek V4 compressor recurrent state restore failed");
        if (saved.overlap) {
            require_equal(
                mfq_tensor_backend::equal(restored.previous_kv, saved.previous_kv) &&
                mfq_tensor_backend::equal(restored.previous_gate, saved.previous_gate),
                "DeepSeek V4 overlap compressor state restore failed");
        }
        require_equal(
            restored.capacity == saved.capacity &&
            mfq_tensor_backend::equal(
                restored.pool.narrow(1, 0, saved.pool.size(1)),
                saved.pool),
            "DeepSeek V4 compressed pool restore failed");
    };
    check_dsv4_pool(
        dsv4_block->compressor, saved_dsv4.compressor);
    check_dsv4_pool(
        dsv4_block->indexer_compressor,
        saved_dsv4.indexer_compressor);
    require_equal(
        dsv4_model.cache_pos == 10 && dsv4_state.bytes > 0,
        "DeepSeek V4 session position restore failed");

    mfq::cuda::GlmDsaCausalLm glm_model;
    auto glm_shared = std::make_shared<GlmDsaSharedState>();
    std::vector<GlmDsaBlock *> glm_blocks;
    for (int index = 0; index < 2; ++index) {
        auto glm = std::make_unique<GlmDsaBlock>();
        glm->cuda_device = 0;
        glm->full_indexer = index == 0;
        glm->shared_state = glm_shared;
        glm->kv_cache = values(
            {1, 1, 16, 6}, mfq_tensor_backend::kFloat16,
            1000.0f + index * 100.0f);
        if (glm->full_indexer) {
            glm->index_cache = values(
                {1, 16, 4}, mfq_tensor_backend::kFloat16, 2000.0f);
        }
        glm_blocks.push_back(glm.get());
        glm_model.blocks.push_back(std::move(glm));
    }
    glm_model.cache_pos = 10;
    const TextSessionState glm_state =
        glm_model.capture_text_session_state(tokens);
    for (auto * glm : glm_blocks) {
        glm->kv_cache.zero_();
        if (glm->index_cache.defined()) glm->index_cache.zero_();
    }
    glm_shared->topk_indices = mfq_tensor_backend::ones(
        {1, 1, 1}, cuda_float.dtype(mfq_tensor_backend::kInt32));
    glm_model.cache_pos = 0;
    glm_model.restore_text_session_state(glm_state);
    const auto& saved_glm = std::get<
        std::vector<GlmDsaBlockSessionState>>(glm_state.payload);
    for (size_t index = 0; index < glm_blocks.size(); ++index) {
        const auto & saved = saved_glm.at(index);
        require_equal(
            glm_blocks[index]->kv_cache.size(2) == saved.kv_capacity &&
            mfq_tensor_backend::equal(
                glm_blocks[index]->kv_cache.narrow(
                    2, 0, saved.kv_cache.size(2)),
                saved.kv_cache),
            "GLM DSA MLA session cache restore failed");
        if (saved.full_indexer) {
            require_equal(
                glm_blocks[index]->index_cache.size(1) ==
                    saved.index_capacity &&
                mfq_tensor_backend::equal(
                    glm_blocks[index]->index_cache.narrow(
                        1, 0, saved.index_cache.size(1)),
                    saved.index_cache),
                "GLM DSA index session cache restore failed");
        }
    }
    require_equal(
        glm_model.cache_pos == 10 && glm_state.bytes > 0 &&
        !glm_shared->topk_indices.defined(),
        "GLM DSA session metadata restore failed");

    std::cout << "text_session_state_check dsv4=1 glm_dsa=1\n";
    return 0;
}
