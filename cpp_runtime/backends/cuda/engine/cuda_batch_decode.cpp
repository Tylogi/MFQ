#include "cuda_batching.h"

#include "core/full_block.h"
#include "models/qwen35/ops.h"
#include "runtime/batch_decode_graph.h"

#include <limits>

namespace mfq::cuda {
namespace tb = mfq_tensor_backend;

QwenBatchOperations::Decoded QwenBatchOperations::decode(State& state) {
    const int primary = execution_.layer_placement.primary_device();
    MfqCudaGuard primary_guard(primary);
    std::vector<std::shared_ptr<Request>> requests;
    std::vector<int32_t> slots;
    int64_t max_position = 0, requested_len = 0;
    int64_t minimum_remaining = std::numeric_limits<int64_t>::max();
    bool batch_greedy = true;
    for (const auto& request : state.active) {
        request->decode_row = -1;
        if (!request->eligible)
            continue;
        request->decode_row = requests.size();
        requests.push_back(request);
        slots.push_back(request->cache.slot());
        state_adapter_.ensure_decode_tokens(request->cache, request->cache_length + 1);
        max_position = std::max(max_position, request->cache_length);
        const auto remaining = request->generation_limit - request->produced;
        minimum_remaining = std::min<int64_t>(minimum_remaining, remaining);
        requested_len = std::max(requested_len, request->cache_length + remaining);
        batch_greedy = batch_greedy && request->sampler->greedy() &&
            !request->sampler->has_penalties() && !request->token_constraint;
    }
    state_adapter_.prepare_decode(max_position);
    const auto planned_len = decode_graph_bucket(requested_len, model_.max_position_embeddings());
    const auto parts = decode_graph_attention_parts(planned_len, FullBlock::kDecodeAttentionMaxParts);
    const bool graph_supported = config_.greedy && batch_greedy &&
        qwen_continuous_batch_cuda_graph_enabled(model_, config_);
    const auto batch = graph_supported ? batch_decode_row_bucket(requests.size(), max_sequences_)
                                       : static_cast<int64_t>(requests.size());
    auto& graph = decode_graphs_->get({batch, planned_len, 0, state_adapter_.decode_capacities()},
                                      2 * model_.blocks.size(), primary);
    const bool graph_decode = graph_supported &&
        (graph.valid || minimum_remaining >= config_.cuda_graph_minimum_tokens);
    Event ready;
    std::unique_ptr<MfqCudaStreamGuard> stream_guard;
    std::vector<std::unique_ptr<MfqCudaStreamGuard>> compute_guards;
    if (graph_decode) {
        ready.record(mfq_get_current_cuda_stream());
        stream_guard = std::make_unique<MfqCudaStreamGuard>(decode_graphs_->stream());
        MFQ_CUDA_CHECK(cudaStreamWaitEvent(decode_graphs_->stream().stream(), ready.get(), 0));
        const auto& parallel = execution_.tensor_parallel.enabled() ? execution_.tensor_parallel
                                                                   : execution_.expert_parallel;
        decode_graphs_->prepare_pool(parallel, execution_.model_parallel_collectives);
        compute_guards = activate_cuda_graph_compute_streams(decode_graphs_->compute_streams());
    }
    auto& input = *graph.inputs;
    std::vector<BatchDecodeRow> rows;
    for (std::size_t row = 0; row < requests.size(); ++row)
        rows.push_back({requests[row]->pending_token, requests[row]->cache_length, slots[row]});
    input.update_rows(rows);
    BatchDecodeMaskScope mask_scope(execution_, graph_supported ? input.enabled() : Tensor{});
    MoeContinuousBatchCacheScope moe_cache_scope(execution_, cached_moe_enabled_ && batch > 1);
    std::vector<Tensor> saved_targets, snapshots;
    if (graph_decode && !graph.valid) {
        saved_targets = state_adapter_.recurrent_slot_tensors(slots);
        for (const auto& tensor : saved_targets)
            snapshots.push_back(tensor.clone());
    }
    state_adapter_.bind_decode_rows(input, slots, graph_decode);
    ++packed_metadata_batches_;
    const auto restore = [&] {
        for (std::size_t i = 0; i < snapshots.size(); ++i)
            saved_targets[i].copy_(snapshots[i]);
        if (!snapshots.empty()) {
            for (const auto& stream : decode_graphs_->participants(execution_.model_parallel_collectives)) {
                MfqCudaGuard guard(stream.device_index());
                MFQ_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
            }
        }
    };
    Tensor logits, graph_tokens;
    const auto invoke = [&](bool captured) {
        state_adapter_.load_decode_rows(input);
        auto ids = input.ids();
        auto positions = input.positions().reshape({batch, 1});
        auto hidden = captured
            ? model_.hidden_forward_static(ids, positions, input.lengths(), {planned_len, parts})
            : model_.hidden_forward(ids, positions, input.lengths(), nullptr, positions);
        auto output = state_adapter_.logits_from_last_hidden(std::move(hidden));
        state_adapter_.store_decode_rows(input);
        return captured ? sample_greedy_cuda(output.contiguous().view({batch, -1})) : output;
    };
    try {
        if (graph_decode) {
            if (!graph.valid) {
                DecodeGraphBranchScope branch_scope(execution_.decode_graph_serial_branches);
                DecodeGraphTpProjectionScope tp_projection_scope(execution_.decode_graph_tp_projection_major);
                const auto participants = decode_graphs_->participants(execution_.model_parallel_collectives);
                const auto started = std::chrono::steady_clock::now();
                for (int pass = 0; pass < 2; ++pass) {
                    std::vector<std::unique_ptr<GraphWarmupScope>> warmup;
                    for (const auto& stream : participants)
                        warmup.push_back(std::make_unique<GraphWarmupScope>(
                            default_context(stream.device_index()), stream.stream()));
                    (void)invoke(true);
                    for (const auto& stream : participants) {
                        MfqCudaGuard guard(stream.device_index());
                        MFQ_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
                    }
                    warmup.clear();
                    restore();
                }
                graph.graph = std::make_unique<MfqCudaGraph>();
                BatchDecodeCaptureScope capture(*graph.graph, participants);
                graph.output = invoke(true);
                capture.finish();
                decode_graphs_->captured(graph, std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - started).count());
            }
            graph.graph->replay();
            graph_tokens = graph.output;
            decode_graphs_->replayed(graph);
        } else {
            logits = invoke(false);
        }
        state_adapter_.unbind_decode_rows();
        state_adapter_.finish_decode(max_position + 1);
    } catch (...) {
        restore();
        state_adapter_.unbind_decode_rows();
        invalidate_decode_graph();
        throw;
    }
    ++decode_batches_;
    decode_tokens_ += requests.size();
    Tensor greedy_tokens;
    if (graph_tokens.defined()) {
        greedy_tokens = graph_tokens.to(tb::kCPU).contiguous();
        ++batched_greedy_batches_;
    } else if (config_.greedy && batch_greedy) {
        greedy_tokens = sample_greedy_cuda(logits.contiguous().view({batch, -1})).to(tb::kCPU).contiguous();
        ++batched_greedy_batches_;
    }
    return {std::move(logits), std::move(greedy_tokens)};
}

} // namespace mfq::cuda
