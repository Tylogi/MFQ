#include "batch_execution.h"
#include "storage/text_session_cache.h"
#include <algorithm>

namespace mfq::cuda {
using Tensor = mfq_tensor_backend::Tensor;

bool batch_requires_owned_generation(const mfq::engine::InferenceRequest& input,
        const internal::TextSessionCache& cache, bool predictor) {
    return input.vision || !input.cache_plan.session_id.empty() ||
        (cache.persistent_prefix_enabled() && input.cache_plan.stable_prefix_tokens) ||
        (input.sampling.enable_mtp && predictor);
}

bool batch_cuda_graph_enabled(const CudaExecutionContext& execution,
        const CudaContinuousBatchConfig& config, bool cached_moe) {
    return config.cuda_graph && !cached_moe && mfq_cuda_graph_capture_supported() &&
        model_parallel_cuda_graph_enabled(execution);
}

void wait_for_batch_stream(const MfqCudaStream& destination, const MfqCudaStream& source) {
    if (destination.stream() == source.stream()) return;
    MfqCudaGuard guard(source.device_index());
    cudaEvent_t ready = nullptr;
    MFQ_CUDA_CHECK(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
    try {
        MFQ_CUDA_CHECK(cudaEventRecord(ready, source.stream()));
        MFQ_CUDA_CHECK(cudaStreamWaitEvent(destination.stream(), ready, 0));
    } catch (...) { cudaEventDestroy(ready); throw; }
    MFQ_CUDA_CHECK(cudaEventDestroy(ready));
}

void BatchDecodeGraph::ensure_compute_streams(const ParallelConfig& parallel) {
    if (compute_streams.empty()) compute_streams = make_cuda_graph_compute_streams(stream, parallel);
}
std::vector<MfqCudaStream> BatchDecodeGraph::participant_streams(
        const ModelParallelCollectiveRuntime& collectives) const {
    return cuda_graph_participant_streams(compute_streams, collectives);
}
bool BatchDecodeGraph::matches(int64_t candidate_batch, int64_t candidate_len,
        const std::vector<const void*>& addresses) const {
    return valid && batch == candidate_batch && planned_len == candidate_len && state_addresses == addresses;
}
void BatchDecodeGraph::set_key(int64_t candidate_batch, int64_t candidate_len,
        std::vector<const void*> addresses) {
    batch = candidate_batch;
    planned_len = candidate_len;
    state_addresses = std::move(addresses);
    valid = true;
}
void BatchDecodeGraph::invalidate() {
    if (graph) graph->reset();
    graph.reset();
    static_next = {};
    state_addresses.clear();
    valid = false;
}

BatchDecodeInputs BatchDecodeMetadata::pack(std::span<const BatchDecodeRow> rows,
        int64_t batch, int device) {
    MFQ_RUNTIME_CHECK(batch > 0 && !rows.empty() && rows.size() <= batch && batch <= capacity_, "invalid batch metadata rows");
    if (!host_.defined()) {
        host_ = mfq_tensor_backend::empty({3 * capacity_}, mfq_tensor_backend::TensorOptions()
            .dtype(mfq_tensor_backend::kInt64).device(mfq_tensor_backend::kCPU).pinned_memory(true));
        device_ = mfq_tensor_backend::empty({3 * capacity_}, mfq_tensor_backend::TensorOptions()
            .dtype(mfq_tensor_backend::kInt64).device(mfq_tensor_backend::Device(mfq_tensor_backend::kCUDA, device)));
    }
    auto* data = host_.data_ptr<int64_t>();
    std::fill_n(data, 2 * batch, int64_t{0});
    std::fill_n(data + 2 * batch, batch, int64_t{1});
    int64_t max_length = 0;
    for (size_t row = 0; row < rows.size(); ++row) {
        data[row] = rows[row].token;
        data[batch + row] = rows[row].position;
        data[2 * batch + row] = rows[row].position + 1;
        max_length = std::max(max_length, rows[row].position + 1);
    }
    auto metadata = device_.narrow(0, 0, 3 * batch);
    metadata.copy_(host_.narrow(0, 0, 3 * batch));
    auto matrix = metadata.reshape({3, batch});
    return {matrix.narrow(0, 0, 1).reshape({batch, 1}), matrix.narrow(0, 1, 1).reshape({batch, 1}),
            matrix.narrow(0, 2, 1).reshape({batch}), max_length};
}

bool batch_sampling_is_greedy(std::span<const BatchSamplingRow> rows) {
    return std::all_of(rows.begin(), rows.end(), [](const auto& row) {
        return row.sampler.greedy() && !row.sampler.has_penalties() && !row.constraint;
    });
}

BatchTokens sample_cuda_batch(Tensor logits, Tensor greedy_tokens,
        std::span<const BatchSamplingRow> rows, bool greedy, int64_t& readbacks) {
    MFQ_RUNTIME_CHECK(!rows.empty(), "batch sampling requires selected rows");
    MFQ_RUNTIME_CHECK(greedy_tokens.defined() ? greedy_tokens.numel() >= rows.size() :
        (logits.defined() && logits.dim() == 2 && logits.size(0) >= rows.size()), "invalid batch sampling rows");
    Tensor sampled;
    if (greedy_tokens.defined()) sampled = std::move(greedy_tokens);
    else if (greedy) sampled = sample_greedy_cuda(logits.contiguous().view({logits.size(0), -1}));
    else {
        sampled = mfq_tensor_backend::empty({logits.size(0)}, logits.options().dtype(mfq_tensor_backend::kInt64));
        for (size_t row = 0; row < rows.size(); ++row) {
            auto scores = logits.narrow(0, row, 1);
            if (rows[row].sampler.has_penalties())
                scores = rows[row].sampler.apply_penalties(std::move(scores), rows[row].counts);
            // Preserve penalized scores for the constraint fallback below.
            logits.narrow(0, row, 1).copy_(scores);
            sampled.narrow(0, row, 1).copy_(rows[row].sampler.sample(scores));
        }
    }
    auto host = sampled.to(mfq_tensor_backend::kCPU).contiguous();
    ++readbacks;
    std::vector<size_t> rejected;
    for (size_t row = 0; row < rows.size(); ++row)
        if (rows[row].constraint && !rows[row].constraint->allows(host.data_ptr<int64_t>()[row]))
            rejected.push_back(row);
    if (!rejected.empty()) {
        // Transfer logits only for rejected grammar candidates, in one batch.
        std::vector<Tensor> scores;
        for (auto row : rejected) scores.push_back(logits.narrow(0, row, 1));
        auto masked = mfq_tensor_backend::cat(scores, 0).to(mfq_tensor_backend::kCPU,
            mfq_tensor_backend::kFloat32).contiguous();
        ++readbacks;
        const auto vocabulary = logits.size(-1);
        for (size_t row = 0; row < rejected.size(); ++row)
            rows[rejected[row]].constraint->apply(masked.data_ptr<float>() + row * vocabulary, vocabulary);
        auto device = masked.to(logits.device());
        for (size_t row = 0; row < rejected.size(); ++row)
            sampled.narrow(0, rejected[row], 1).copy_(rows[rejected[row]].sampler.sample(device.narrow(0, row, 1)));
        host = sampled.to(mfq_tensor_backend::kCPU).contiguous();
        ++readbacks;
    }
    for (size_t row = 0; row < rows.size(); ++row) if (rows[row].constraint) {
        const auto token = host.data_ptr<int64_t>()[row];
        MFQ_RUNTIME_CHECK(rows[row].constraint->allows(token), "CUDA constrained sampler returned an invalid token");
        rows[row].constraint->accept(token);
    }
    return {std::move(sampled), std::move(host)};
}
} // namespace mfq::cuda
