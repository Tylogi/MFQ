#pragma once

#include <chrono>
#include "token_constraint.h"
#include "tokenizer.h"
#include "chat.h"

namespace mfq::cuda::internal {

// Exercise the actual shared coroutines, including paused output and teardown
// of verified work. Every oracle is an ordinary single-request generation.
template<class Model>
void check_batch_mtp(Model& model, MtpModule& predictor, CudaRuntimeConfig config, int rows) {
    using namespace mfq::engine;
    using Clock = std::chrono::steady_clock;
    config.decode_graph.enabled = false;
    config.continuous_batch.cuda_graph = false;
    std::unique_ptr<MfqTokenizer> tokenizer;
    if (const auto* path = std::getenv("MFQ_BATCH_MTP_TEST_TOKENIZER"))
        tokenizer = std::make_unique<MfqTokenizer>(path);
    const auto execute = [&](int batch_size, bool enabled, bool membership, bool report) {
        std::vector<std::unique_ptr<InferenceRequest>> requests;
        std::vector<std::unique_ptr<InferenceOutput>> outputs;
        std::vector<std::vector<int64_t>> reference(batch_size), actual(batch_size);
        for (int row = 0; row < batch_size; ++row) {
            auto input = std::make_unique<InferenceRequest>();
            const int length = membership && row == 0 ? 253
                : membership && row == 1 ? 3 : 17 + (membership ? 3 * row : 0);
            for (int token = 0; token < length; ++token)
                input->prompt.push_back((token * 37 + 13 + row * 17) % model.vocab_size());
            input->sampling.max_tokens = 48;
            input->sampling.temperature = 0;
            input->sampling.top_k = 1;
            input->sampling.mtp_max_draft_tokens = row % 3 + 1;
            input->sampling.seed = 919 + row;
            if (membership && row % 2) {
                input->sampling.presence_penalty = .2;
                input->sampling.frequency_penalty = -.01;
                input->sampling.repetition_penalty = 1.1;
            }
            input->sampling.enable_mtp = false;
            if (membership && tokenizer && row % 4 == 3) {
                common_chat_params grammar;
                grammar.grammar = "root ::= (\"a\" | \"b\")+";
                input->token_constraint = make_chat_token_constraint(*tokenizer, grammar);
                input->stop_token_ids = {tokenizer->eos_token(), tokenizer->eot_token()};
            }
            auto constraint = input->token_constraint ? input->token_constraint->clone() : MfqTokenConstraintPtr{};
            DecodeGraphCache graph(model.max_position_embeddings());
            TextSessionCache cache(config.session_cache, config.prefix_cache);
            InferenceOutput oracle(*input, nullptr, "ordinary-oracle");
            reference[row] = diagnostics::collect_generation(
                generate(model, graph, cache, config, *input, oracle)).tokens;
            graph.invalidate();
            if constexpr (std::is_same_v<Model, Qwen4CausalLm>) qwen4_release_decode_graphs(model, nullptr);
            input->sampling.enable_mtp = enabled && (!membership || row % 8 != 4);
            input->token_constraint = std::move(constraint);
            requests.push_back(std::move(input));
            outputs.push_back(std::make_unique<InferenceOutput>(*requests.back(), nullptr, "batch-mtp"));
        }
        auto batch_config = config.continuous_batch;
        batch_config.max_sequences = batch_size;
        ContinuousBatch<CudaBatchOperations<Model>> batcher(64, batch_config.prefill_token_budget,
            model, *model.execution, batch_config, &predictor);
        std::vector<Generation> generations;
        std::vector<bool> done(batch_size, false);
        std::vector<Clock::time_point> first(batch_size), last(batch_size);
        for (int row = 0; row < batch_size; ++row)
            generations.push_back(batcher.generate("mtp-" + std::to_string(row), *requests[row], *outputs[row]));
        mfq_cuda_synchronize();
        const auto started = Clock::now();
        bool cancelled = false, cancel_scheduled = false;
        const auto mtp_cycles = [&] {
            for (const auto& [name, value] : batcher.metrics())
                if (name == "continuous_batching_mtp_batches") return value;
            return 0.0;
        };
        for (int tick = 0; tick == 0 || !batcher.empty(); ++tick) {
            MFQ_RUNTIME_CHECK(tick < 2000, "batched MTP scheduler failed to drain");
            std::vector<std::string> eligible;
            for (int row = 0; row < batch_size; ++row) {
                if (done[row]) continue;
                if (membership && row == 1 && tick >= 12 && tick < 20) continue;
                auto step = generations[row].next();
                if (!step) {
                    done[row] = true;
                    last[row] = Clock::now();
                    continue;
                }
                eligible.push_back("mtp-" + std::to_string(row));
                if (membership && row == 2 && actual[row].size() >= 5 && !cancelled &&
                        step.state == mfq::StepState::advanced && !step.value)
                    cancel_scheduled = true;
                if (step.value) if (const auto* delta = std::get_if<TokenOutput>(&*step.value)) {
                    if (actual[row].empty()) first[row] = Clock::now();
                    actual[row].insert(actual[row].end(), delta->token_ids.begin(), delta->token_ids.end());
                    if (actual[row].size() > reference[row].size())
                        throw std::runtime_error("batched MTP exceeded the greedy oracle length");
                    for (std::size_t position = 0; position < actual[row].size(); ++position)
                        if (actual[row][position] != reference[row][position])
                            throw std::runtime_error("batched MTP greedy mismatch row=" + std::to_string(row) +
                                " position=" + std::to_string(position) + " actual=" +
                                std::to_string(actual[row][position]) + " expected=" +
                                std::to_string(reference[row][position]) + " accepted=" +
                                std::to_string(outputs[row]->metrics.mtp.accepted_tokens));
                }
            }
            const auto before = cancel_scheduled ? mtp_cycles() : 0;
            batcher.step(eligible);
            if (cancel_scheduled && mtp_cycles() > before) {
                outputs[2]->result.cancelled = true;
                cancelled = true;
                cancel_scheduled = false;
            }
        }
        mfq_cuda_synchronize();
        int total = 0;
        double stream_rate = 0, drafted = 0, accepted = 0, cycles = 0;
        for (int row = 0; row < batch_size; ++row) {
            if (!outputs[row]->result.cancelled)
                MFQ_RUNTIME_CHECK(actual[row] == reference[row], "batched MTP lost completion tokens");
            MFQ_RUNTIME_CHECK(!outputs[row]->cleanup_failure, "batched MTP failed request cleanup");
            total += actual[row].size();
            stream_rate += (actual[row].size() - 1) /
                std::chrono::duration<double>(last[row] - first[row]).count();
        }
        for (const auto& [name, value] : batcher.metrics()) {
            if (name == "continuous_batching_mtp_drafted_tokens") drafted = value;
            if (name == "continuous_batching_mtp_accepted_tokens") accepted = value;
            if (name == "continuous_batching_mtp_batches") cycles = value;
        }
        MFQ_RUNTIME_CHECK(!enabled || cycles > 0, "batch check did not execute MTP");
        const auto elapsed = std::chrono::duration<double>(Clock::now() - started).count();
        std::cout << (report ? "batch_mtp_bench" : "continuous_batching_mtp_check")
            << " rows=" << batch_size << " mtp=" << enabled << " tokens=" << total
            << " stream_tok_s=" << stream_rate / batch_size << " total_tok_s=" << total / elapsed
            << " accepted=" << accepted << " drafted=" << drafted
            << " acceptance=" << (drafted ? accepted / drafted : 0) << " greedy=PASS"
            << " paused=" << membership << " cancelled=" << cancelled
            << " grammar=" << bool(tokenizer) << '\n';
    };
    execute(rows, true, !std::getenv("MFQ_BATCH_MTP_HOMOGENEOUS"), false);
    if (std::getenv("MFQ_BATCH_MTP_BENCH"))
        for (const int batch : {1, 4, 8, 16, 32})
            for (bool enabled : {false, true}) execute(batch, enabled, false, true);
}

} // namespace mfq::cuda::internal
