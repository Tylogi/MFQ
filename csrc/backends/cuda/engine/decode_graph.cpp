#include "engine/decode_graph.h"
#include "../ops/cuda_execution.h"

#include <algorithm>

std::vector<MfqCudaStream> make_cuda_graph_compute_streams(
        const MfqCudaStream& primary_stream,
        const ParallelConfig& parallel) {
    if (!parallel.enabled()) return {primary_stream};
    std::vector<MfqCudaStream> streams;
    streams.reserve(parallel.devices.size());
    for (const int device : parallel.devices) {
        streams.push_back(
            device == primary_stream.device_index()
                ? primary_stream
                : mfq_get_stream_from_pool(false, device));
    }
    return streams;
}

std::vector<MfqCudaStream> cuda_graph_participant_streams(
        const std::vector<MfqCudaStream>& compute_streams,
        const ModelParallelCollectiveRuntime& collectives) {
    auto participants = compute_streams;
    participants.insert(
        participants.end(),
        collectives.streams.begin(), collectives.streams.end());
    return participants;
}

std::vector<std::unique_ptr<MfqCudaStreamGuard>>
activate_cuda_graph_compute_streams(
        const std::vector<MfqCudaStream>& compute_streams) {
    std::vector<std::unique_ptr<MfqCudaStreamGuard>> guards;
    guards.reserve(compute_streams.size());
    for (const auto& stream : compute_streams) {
        guards.push_back(
            std::make_unique<MfqCudaStreamGuard>(stream));
    }
    return guards;
}

DecodeGraphCache::DecodeGraphCache(int64_t context_capacity)
    : stream(mfq_get_stream_from_pool(false)),
      generated_capacity(std::max<int64_t>(context_capacity, 2048)) {}

void DecodeGraphCache::ensure_compute_streams(
        const ParallelConfig& parallel) {
    if (!compute_streams.empty()) return;
    compute_streams = make_cuda_graph_compute_streams(stream, parallel);
}

std::vector<MfqCudaStream> DecodeGraphCache::graph_participant_streams(
        const ModelParallelCollectiveRuntime& collectives) const {
    return cuda_graph_participant_streams(compute_streams, collectives);
}

bool DecodeGraphCache::matches(
        int64_t candidate_len, const MfqSamplingParams & sampling,
        bool candidate_greedy) const {
    return valid && planned_len == candidate_len && greedy == candidate_greedy &&
           (candidate_greedy ||
            (temperature == sampling.temperature && top_k == sampling.top_k &&
             top_p == sampling.top_p)) &&
           presence_penalty == sampling.presence_penalty &&
           frequency_penalty == sampling.frequency_penalty &&
           repetition_penalty == sampling.repetition_penalty;
}

void DecodeGraphCache::ensure_storage(int64_t vocab_size) {
    auto i64 = mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64).device(mfq_tensor_backend::kCUDA);
    if (!static_input.defined()) static_input = mfq_tensor_backend::empty({1, 1}, i64);
    if (!static_pos.defined()) static_pos = mfq_tensor_backend::empty({1}, i64);
    if (!static_len.defined()) static_len = mfq_tensor_backend::empty({1}, i64);
    if (!static_step.defined()) static_step = mfq_tensor_backend::empty({1}, i64);
    if (!generated.defined()) generated = mfq_tensor_backend::empty({generated_capacity}, i64);
    if (!random.defined()) {
        random = mfq_tensor_backend::empty(
            {1}, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kFloat32).device(mfq_tensor_backend::kCUDA));
    }
    if (!counts.defined()) {
        counts = mfq_tensor_backend::empty(
            {vocab_size}, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt32).device(mfq_tensor_backend::kCUDA));
    }
}

void DecodeGraphCache::invalidate() {
    if (graph) graph->reset();
    graph.reset();
    static_next = mfq_tensor_backend::Tensor();
    valid = false;
}

void DecodeGraphCache::set_key(
        int64_t candidate_len, const MfqSamplingParams & sampling,
        bool candidate_greedy) {
    planned_len = candidate_len;
    greedy = candidate_greedy;
    temperature = sampling.temperature;
    top_k = sampling.top_k;
    top_p = sampling.top_p;
    presence_penalty = sampling.presence_penalty;
    frequency_penalty = sampling.frequency_penalty;
    repetition_penalty = sampling.repetition_penalty;
    valid = true;
}

int64_t decode_graph_bucket(int64_t planned_len, int64_t context_capacity) {
    const int64_t quantum = planned_len <= 4096 ? 512 :
        (planned_len <= 16384 ? 1024 : 2048);
    const int64_t rounded = ((planned_len + quantum - 1) / quantum) * quantum;
    return std::min<int64_t>(rounded, context_capacity);
}

int64_t decode_graph_attention_parts(
        int64_t planned_len, int64_t max_parts) {
    const int64_t parts = planned_len >= 192
        ? (planned_len + 127) / 128 : 1;
    return std::min(parts, max_parts);
}
