#pragma once

#include "mfq_tensor_backend.h"
#include "cuda_model_plan.h"
#include "mfq/runtime.h"
#include "../models/qwen35/qwen35_linear_attention.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

std::vector<MfqCudaStream> make_cuda_graph_compute_streams(
    const MfqCudaStream& primary_stream);
std::vector<MfqCudaStream> cuda_graph_participant_streams(
    const std::vector<MfqCudaStream>& compute_streams);
std::vector<std::unique_ptr<MfqCudaStreamGuard>>
activate_cuda_graph_compute_streams(
    const std::vector<MfqCudaStream>& compute_streams);

struct DecodeGraphCache {
    decltype(mfq_get_stream_from_pool(false)) stream;
    std::vector<MfqCudaStream> compute_streams;
    std::unique_ptr<MfqCudaGraph> graph;
    mfq_tensor_backend::Tensor static_input;
    mfq_tensor_backend::Tensor static_pos;
    mfq_tensor_backend::Tensor static_len;
    mfq_tensor_backend::Tensor static_step;
    mfq_tensor_backend::Tensor generated;
    mfq_tensor_backend::Tensor random;
    mfq_tensor_backend::Tensor counts;
    mfq_tensor_backend::Tensor static_next;
    int64_t generated_capacity = 0;
    int64_t planned_len = 0;
    bool greedy = false;
    double temperature = 0.0;
    int32_t top_k = 0;
    double top_p = 1.0;
    double presence_penalty = 0.0;
    double frequency_penalty = 0.0;
    double repetition_penalty = 1.0;
    uint64_t captures = 0;
    uint64_t reuses = 0;
    bool valid = false;

    explicit DecodeGraphCache(int64_t context_capacity);
    void ensure_compute_streams();
    std::vector<MfqCudaStream> graph_participant_streams() const;
    bool matches(int64_t candidate_len, const MfqSamplingParams& sampling,
                 bool candidate_greedy) const;
    void ensure_storage(int64_t vocab_size);
    void invalidate();
    void set_key(int64_t candidate_len, const MfqSamplingParams& sampling,
                 bool candidate_greedy);

    template <typename Model, typename Sample, typename Commit>
    bool ensure_captured(
        Model& model,
        int64_t candidate_len,
        const MfqSamplingParams& sampling,
        bool candidate_greedy,
        Sample&& sample,
        Commit&& commit);
};

template <typename Model>
void prepare_decode_graph_memory(Model& model, MfqCudaGraph& graph,
        const std::function<void()>& warmup,
        const std::vector<MfqCudaStream>& participant_streams = {}) {
    using Tensor = mfq_tensor_backend::Tensor;
    struct SavedRecurrentState {
        mfq::cuda::qwen35::LinearAttentionBlock* block;
        Tensor conv, gdn;
        const void* conv_address;
        const void* gdn_address;
    };
    // Snapshot before entering the private graph allocator. These copies are
    // temporary and must not become retained allocations in the captured pool.
    std::vector<SavedRecurrentState> saved;
    for (const auto& block : model.blocks) {
        if (auto* linear = dynamic_cast<
                mfq::cuda::qwen35::LinearAttentionBlock*>(block.get())) {
            MFQ_RUNTIME_CHECK(!linear->speculative_pending && linear->conv_state.defined() &&
                linear->gdn_state.defined(), "decode warmup requires confirmed recurrent state");
            saved.push_back({linear, linear->conv_state.clone(), linear->gdn_state.clone(),
                linear->conv_state.data_ptr(), linear->gdn_state.data_ptr()});
        }
    }
    const int64_t saved_position = model.cache_pos;
    auto restore = [&]() {
        for (const auto& state : saved) {
            MFQ_RUNTIME_CHECK(state.block->conv_state.data_ptr() == state.conv_address &&
                state.block->gdn_state.data_ptr() == state.gdn_address,
                "decode warmup changed recurrent storage addresses");
            state.block->conv_state.copy_(state.conv);
            state.block->gdn_state.copy_(state.gdn);
        }
        MFQ_RUNTIME_CHECK(model.cache_pos == saved_position, "static decode warmup changed cache position");
        if (participant_streams.empty()) {
            MFQ_CUDA_CHECK(cudaStreamSynchronize(
                mfq_get_current_cuda_stream().stream()));
            return;
        }
        for (const auto& participant : participant_streams) {
            MfqCudaGuard guard(participant.device_index());
            MFQ_CUDA_CHECK(cudaStreamSynchronize(
                participant.stream()));
        }
    };
    mfq_prepare_cuda_graph_memory(graph, participant_streams);
    if (!saved.empty() || Model::is_gemma4 || (Model::backbone == mfq::cuda::CudaBackbone::glm_dsa) || Model::is_minicpmo45) {
        // The first pass may initialize persistent CUDA/NCCL workspace state.
        // A second pass then records the complete set of reusable temporaries
        // needed by capture after those persistent allocations exist.
        for (int pass = 0; pass < 2; ++pass) {
            try {
                // Replay overwrites this same, still-unconfirmed KV position.
                warmup();
                restore();
            } catch (...) {
                restore();
                throw;
            }
        }
    }
}

template <typename Model, typename Sample, typename Commit>
bool DecodeGraphCache::ensure_captured(
        Model& model,
        int64_t candidate_len,
        const MfqSamplingParams& sampling,
        bool candidate_greedy,
        Sample&& sample,
        Commit&& commit) {
    if (matches(candidate_len, sampling, candidate_greedy)) {
        ++reuses;
        return true;
    }

    invalidate();
    mfq_cuda_empty_cache();
    DecodeGraphBranchScope branch_scope;
    graph = std::make_unique<MfqCudaGraph>();
    try {
        prepare_decode_graph_memory(
            model, *graph, sample, graph_participant_streams());
        graph->capture_begin();
        static_next = sample();
        commit(static_next);
        graph->capture_end();
        set_key(candidate_len, sampling, candidate_greedy);
        ++captures;
    } catch (...) {
        invalidate();
        throw;
    }
    return false;
}

int64_t decode_graph_bucket(int64_t planned_len, int64_t context_capacity);
int64_t decode_graph_attention_parts(int64_t planned_len, int64_t max_parts);
bool trace_cuda_graph();
