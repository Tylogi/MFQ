#include "runtime/mlx_qwen4_batching.h"
#include "table.h"

#include <cmath>
#include <iostream>

using namespace mfq::metal;
using namespace mfq::engine;
using namespace mlx::core;

static void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
static void near(const array& actual, const array& expected) {
    const auto error = max(abs(astype(actual, float32) - astype(expected, float32))).item<float>();
    if (!std::isfinite(error) || error >= .005f)
        std::cerr << "batch parity error=" << error << " shape=" << actual.shape() << '\n';
    require(std::isfinite(error) && error < .005f, "batched logits differ from independent serial reference");
}
static std::vector<std::int64_t> prompt(int count, int offset) {
    std::vector<std::int64_t> result(count);
    for (int index = 0; index < count; ++index) result[index] = (index * 7 + offset) % 23;
    return result;
}
static void check_ready_decode(MlxQwen4CausalLm& runtime) {
    MlxQwen4BatchOperations operations(runtime, {});
    InferenceRequest input;
    input.prompt = {1};
    input.sampling.max_tokens = 12;
    auto first = std::make_shared<MlxQwen4BatchRequest>("first", input);
    auto second = std::make_shared<MlxQwen4BatchRequest>("second", input);
    first->mtp = second->mtp = true;
    first->eligible = true;
    MlxQwen4BatchOperations::State state;
    state.active = {first, second};
    require(!operations.ready_decode(state, {"first", "second"}), "MTP stepped before its runnable peer drained output");
    require(operations.ready_decode(state, {"first"}), "a backpressured MTP peer blocked progress");
    second->eligible = true;
    require(operations.ready_decode(state, {"first", "second"}), "ready MTP batch was blocked");
    second->eligible = false;
    second->produced = second->generation_limit;
    require(operations.ready_decode(state, {"first", "second"}), "finished MTP peer blocked progress");
    first->mtp = false;
    require(operations.ready_decode(state, {"first", "second"}), "ordinary decode inherited MTP synchronization");
    first->observe_mtp_depth(0, true);
    first->observe_mtp_depth(0, true);
    first->observe_mtp_depth(0, false);
    require(first->mtp_parks == 1 && first->mtp_reentries == 0,
        "MTP membership change erased a park or counted warmup as reentry");
    first->observe_mtp_depth(3, false);
    first->observe_mtp_depth(3, false);
    first->observe_mtp_depth(0, true);
    require(first->mtp_parks == 2 && first->mtp_reentries == 1 && second->mtp_parks == 0,
        "MTP park/reentry counts crossed requests or duplicated depth observations");
}
static void check_draft_heads(MlxQwen4CausalLm& runtime) {
    constexpr int batch = 3, depth = 5;
    std::vector<std::shared_ptr<MlxQwen4RequestCache>> caches;
    std::vector<MlxMtpEngineCallbacks> callbacks;
    std::vector<std::vector<array>> expected(batch);
    for (int row = 0; row < batch; ++row) {
        const auto tokens = prompt(33 + row * 31, row);
        auto reference = runtime.make_request_cache();
        runtime.prefill_request(*reference, tokens, true);
        const MlxMtpDraftContext context{true, row + 1, depth, runtime.request_cache_position(*reference),
            0, nullptr, {}, nullptr};
        auto scalar = runtime.request_mtp_callbacks(reference);
        scalar.prepare_draft(context, [&](const array& logits) {
            expected[row].push_back(logits);
            return mlx::core::reshape(sample_greedy(logits), {1});
        });
        auto cache = runtime.make_request_cache();
        runtime.prefill_request(*cache, tokens, true);
        auto adapter = runtime.request_mtp_callbacks(cache, true);
        adapter.prepare_draft_history(context);
        caches.push_back(std::move(cache));
        callbacks.push_back(std::move(adapter));
    }
    std::vector<std::optional<array>> previous(batch);
    for (int step = 0; step < depth; ++step) {
        runtime.draft_request_heads(caches, previous);
        for (int row = 0; row < batch; ++row) {
            auto actual = callbacks[row].draft_logits();
            near(actual, expected[row][step]);
            previous[row] = mlx::core::reshape(sample_greedy(actual), {1});
        }
    }
    std::cout << "Batched predictor and LM Head match independent drafting\n";
}
static void check_packed_prefill(MlxQwen4CausalLm& runtime) {
    const std::vector<bool> mtp{true, false, true};
    std::vector<std::shared_ptr<MlxQwen4RequestCache>> caches, reference;
    std::vector<std::vector<std::int64_t>> history(3);
    for (int row = 0; row < 3; ++row) {
        caches.push_back(runtime.make_request_cache());
        reference.push_back(runtime.make_request_cache());
    }
    const std::array<std::array<int, 3>, 3> lengths{{{1, 17, 43}, {7, 1, 19}, {13, 8, 1}}};
    for (int step = 0; step < 3; ++step) {
        std::vector<std::vector<std::int64_t>> tokens;
        std::vector<array> expected;
        for (int row = 0; row < 3; ++row) {
            tokens.push_back(prompt(lengths[step][row], row + step));
            expected.push_back(runtime.prefill_request(*reference[row], tokens.back(), mtp[row]));
            history[row].insert(history[row].end(), tokens.back().begin(), tokens.back().end());
        }
        auto actual = runtime.prefill_requests(caches, tokens, mtp);
        for (int row = 0; row < 3; ++row) {
            near(actual[row], expected[row]);
            const auto state = runtime.capture_request_cache(*caches[row], history[row]);
            const auto oracle = runtime.capture_request_cache(*reference[row], history[row]);
            require(state.cache_position == oracle.cache_position, "packed prefill position crossed requests");
            for (std::size_t layer = 0; layer < state.layers.size(); ++layer) {
                const auto& left = state.layers[layer];
                const auto& right = oracle.layers[layer];
                if (left.convolution) near(*left.convolution, *right.convolution);
                if (left.recurrent) near(*left.recurrent, *right.recurrent);
                if (left.ple_convolution) near(*left.ple_convolution, *right.ple_convolution);
                require(left.ple_context == right.ple_context, "packed prefill PLE history crossed requests");
            }
            if (step == 1) {
                auto restored = runtime.make_request_cache();
                runtime.restore_request_cache(*restored, state);
                caches[row] = std::move(restored);
            }
        }
    }
    for (int row : {0, 2}) {
        MlxMtpDraftContext context{true, row + 1, 2, runtime.request_cache_position(*reference[row]),
            0, nullptr, {}, nullptr};
        std::vector<array> expected, actual;
        runtime.request_mtp_callbacks(reference[row]).prepare_draft(context, [&](const array& logits) {
            expected.push_back(logits);
            return reshape(sample_greedy(logits), {1});
        });
        runtime.request_mtp_callbacks(caches[row]).prepare_draft(context, [&](const array& logits) {
            actual.push_back(logits);
            return reshape(sample_greedy(logits), {1});
        });
        for (int step = 0; step < 2; ++step) near(actual[step], expected[step]);
    }
    bool rejected = false;
    const int position = runtime.request_cache_position(*caches[0]);
    try { runtime.prefill_requests({caches[0], caches[0]}, {{1}, {2}}, {false, false}); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected && runtime.request_cache_position(*caches[0]) == position,
        "duplicate packed prefill cache mutated state");
    runtime.clear_cache();
    std::cout << "Ragged prefill, predictor priming and restored prefix match independent execution\n";
}
static void check_state(MlxQwen4CausalLm& runtime, int rows = 3) {
    std::vector<std::shared_ptr<MlxQwen4RequestCache>> caches;
    std::vector<MlxQwen4TextSessionState> reference;
    std::vector<std::vector<std::int64_t>> history;
    for (int row = 0; row < rows; ++row) {
        auto tokens = prompt(rows == 8 ? 17 + row : row == 2 ? 769 : 65 + row * 64, row + 1);
        runtime.clear_cache();
        auto expected = runtime.forward(array(tokens.begin(), {1, static_cast<int>(tokens.size())}, int32));
        expected.eval();
        reference.push_back(runtime.capture_text_session_state(tokens));
        auto cache = runtime.make_request_cache();
        auto actual = runtime.prefill_request(*cache, tokens);
        near(actual, slice(expected, {0, static_cast<int>(tokens.size()) - 1, 0},
            {1, static_cast<int>(tokens.size()), expected.shape(2)}));
        history.push_back(tokens);
        caches.push_back(std::move(cache));
    }
    runtime.clear_cache();
    require(runtime.kv_cache_contexts() == static_cast<std::size_t>(rows) && runtime.kv_cache_bytes() > 0,
        "live request KV missing from telemetry");
    for (int step = 0; step < 4; ++step) {
        std::vector<std::int64_t> tokens;
        for (int row = 0; row < rows; ++row) tokens.push_back((7 + row * 5 + step) % 23);
        std::vector<array> expected;
        for (int row = 0; row < rows; ++row) {
            runtime.restore_text_session_state(reference[row]);
            auto logits = runtime.forward(array({tokens[row]}, {1, 1}, int32));
            logits.eval();
            expected.push_back(logits);
            history[row].push_back(tokens[row]);
            reference[row] = runtime.capture_text_session_state(history[row]);
        }
        runtime.clear_cache();
        const auto before = runtime.ssd_expert_cache_stats();
        auto actual = runtime.decode_requests(caches, tokens);
        if (before) {
            const auto after = runtime.ssd_expert_cache_stats();
            const auto limit = runtime.config().num_hidden_layers * 3 * caches.size() *
                runtime.config().num_experts_per_tok;
            require(after && after->requests - before->requests <= limit,
                "batch decode materialized unselected expert banks");
        }
        for (int row = 0; row < rows; ++row) {
            near(slice(actual, {row, 0, 0}, {row + 1, 1, actual.shape(2)}), expected[row]);
            const auto state = runtime.capture_request_cache(*caches[row], history[row]);
            require(state.cache_position == static_cast<int>(history[row].size()), "request position crossed rows");
            for (std::size_t layer = 0; layer < state.layers.size(); ++layer) {
                const auto& left = state.layers[layer];
                const auto& right = reference[row].layers[layer];
                if (left.convolution) near(*left.convolution, *right.convolution);
                if (left.recurrent) near(*left.recurrent, *right.recurrent);
                if (left.ple_convolution) near(*left.ple_convolution, *right.ple_convolution);
                require(left.ple_context == right.ple_context, "PLE history crossed requests");
            }
        }
    }
    bool rejected = false;
    try { runtime.decode_requests({caches[0], caches[0]}, {1, 2}); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "duplicate mutable request cache was accepted");
    caches.erase(caches.begin() + 1);
    auto logits = runtime.decode_requests(caches, std::vector<std::int64_t>(caches.size(), 3));
    require(logits.shape(0) == rows - 1, "retiring the middle row corrupted batch shape");
    caches.clear();
    reference.clear();
    runtime.clear_cache();
    if (runtime.kv_cache_bytes()) {
        const auto stats = runtime.qsa_kv_offload_stats();
        std::cerr << "retired KV bytes=" << runtime.kv_cache_bytes() << " contexts=" << runtime.kv_cache_contexts()
                  << " hot=" << (stats ? stats->hot_bytes : 0) << " pending=" << (stats ? stats->pending_bytes : 0) << '\n';
    }
    require(runtime.kv_cache_bytes() == 0 && runtime.kv_cache_contexts() == 0,
        "retired requests retained active KV storage");
}
static void check_greedy_sequence(MlxQwen4CausalLm& runtime, const InferenceRequest& input,
    const std::vector<std::int64_t>& tokens) {
    runtime.clear_cache();
    auto logits = runtime.forward(array(input.prompt.begin(), {1, static_cast<int>(input.prompt.size())}, int32));
    MlxSampler sampler(mlx_batch_sampling(input.sampling));
    std::optional<array> counts;
    if (sampler.params().has_penalties()) counts = sample_token_counts_add(zeros({1, 32}, float32),
        array(input.prompt.begin(), {1, static_cast<int>(input.prompt.size())}, int32));
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        require(tokens[index] >= 0 && tokens[index] < 32, "batched token is outside the vocabulary");
        auto row = mlx_last_token_logits(logits, 32);
        if (counts) row = sampler.apply_penalties(row, *counts);
        row = contiguous(astype(row, float32));
        row.eval();
        const auto* values = row.data<float>();
        const auto best = *std::max_element(values, values + 32);
        const auto gap = best - values[tokens[index]];
        if (!std::isfinite(gap) || gap > .0005f) {
            std::cerr << "greedy teacher-forced token=" << index << " gap=" << gap << '\n';
            throw std::runtime_error("batched token is not a near-tied serial argmax");
        }
        if (counts) *counts = sample_token_counts_add(*counts, array({tokens[index]}, {1, 1}, int32));
        if (index + 1 < tokens.size()) logits = runtime.forward(array({tokens[index]}, {1, 1}, int32));
    }
    runtime.clear_cache();
}
static void check_scheduler(MlxQwen4CausalLm& runtime, bool prefix_cache, bool mtp = false,
    bool packed_prefill = false) {
    std::vector<MlxQwen4TextSessionState> checkpoints;
    std::size_t restored_tokens = 0;
    MlxQwen4BatchOperations::Hooks hooks;
    if (prefix_cache) {
        hooks.block_size = 16;
        hooks.restore = [&](auto& request) {
            request.prefix = request.cache_plan.stable_prefix_tokens;
            const MlxQwen4TextSessionState* best = nullptr;
            for (const auto& state : checkpoints) if (state.tokens.size() < request.prompt.size() &&
                std::equal(state.tokens.begin(), state.tokens.end(), request.prompt.begin()) &&
                (!best || state.tokens.size() > best->tokens.size())) best = &state;
            if (!best) return std::size_t{0};
            runtime.restore_text_session_state(*best);
            restored_tokens += best->tokens.size();
            return best->tokens.size();
        };
        hooks.store = [&](auto&, auto state) { checkpoints.push_back(std::move(state)); };
    }
    const int chunk = packed_prefill ? 256 : 32;
    ContinuousBatch<MlxQwen4BatchOperations> batch(chunk, chunk, runtime, std::move(hooks));
    std::vector<InferenceRequest> inputs(4);
    std::vector<std::unique_ptr<InferenceOutput>> outputs;
    std::vector<Generation> generations;
    std::vector<std::vector<std::int64_t>> actual(4), expected(4);
    for (int row = 0; row < 4; ++row) {
        inputs[row].chat = false;
        inputs[row].prompt = prompt(33 + row * 13, row);
        inputs[row].sampling.enable_mtp = mtp && row != 1;
        inputs[row].sampling.mtp_max_draft_tokens = 2;
        inputs[row].sampling.top_k = 20;
        inputs[row].sampling.temperature = !mtp && row == 2 ? .35 : 0.0;
        inputs[row].sampling.seed = 17 + row;
        inputs[row].sampling.presence_penalty = row == 3 ? .2 : 0.0;
        inputs[row].sampling.frequency_penalty = row == 3 ? .05 : 0.0;
        inputs[row].sampling.repetition_penalty = row == 3 ? 1.1 : 1.0;
        inputs[row].sampling.max_tokens = (mtp ? 24 : 6) + row;
        if (prefix_cache) {
            inputs[row].cache_plan.stable_prefix_tokens = inputs[row].prompt.size();
            inputs[row].cache_plan.cache_output_tokens = true;
        }
        auto sampling = mlx_batch_sampling(inputs[row].sampling);
        runtime.generate(inputs[row].prompt, sampling, inputs[row].sampling.max_tokens,
            [&](std::int64_t token) { expected[row].push_back(token); return true; }, {}, {}, {}, 32);
        outputs.push_back(std::make_unique<InferenceOutput>(inputs[row], nullptr, std::to_string(row)));
        generations.push_back(batch.generate(std::to_string(row), inputs[row], *outputs.back()));
    }
    std::vector<bool> done(4, false);
    bool draft_progress = false;
    for (int iteration = 0; iteration < 200; ++iteration) {
        std::vector<std::string> eligible;
        const int admitted = iteration < 4 ? 2 : 4;
        for (int row = 0; row < admitted; ++row) if (!done[row]) eligible.push_back(std::to_string(row));
        batch.step(eligible);
        for (int row = 0; row < admitted; ++row) if (!done[row]) {
            auto step = generations[row].next();
            if (mtp && !actual[row].empty() && step.state == mfq::StepState::advanced && !step.value)
                draft_progress = true;
            if (!step) { done[row] = true; continue; }
            if (step.value) if (auto* delta = std::get_if<OutputDelta>(&*step.value))
                actual[row].insert(actual[row].end(), delta->token_ids.begin(), delta->token_ids.end());
        }
        if (std::all_of(done.begin(), done.end(), [](bool value) { return value; })) break;
    }
    require(batch.empty(), "continuous scheduler failed to retire requests");
    if (actual != expected) for (int row = 0; row < 4; ++row) if (actual[row] != expected[row]) {
        std::cerr << "row=" << row << " actual=";
        for (auto token : actual[row]) std::cerr << token << ',';
        std::cerr << " expected=";
        for (auto token : expected[row]) std::cerr << token << ',';
        std::cerr << '\n';
    }
    for (int row = 0; row < 4; ++row) if (actual[row] != expected[row]) {
        require(actual[row].size() == expected[row].size() && inputs[row].sampling.temperature == 0,
            "continuous scheduler changed stochastic sampling or termination");
        check_greedy_sequence(runtime, inputs[row], actual[row]);
    }
    if (mtp) require(draft_progress, "MTP draft work never advanced the executor");
    bool merged = false;
    bool prefill_merged = false;
    double mtp_cycles = 0, mtp_batches = 0;
    for (const auto& [name, value] : batch.metrics()) {
        std::cout << name << '=' << value << ' ';
        if (name == "continuous_batching_maximum_batch") merged = value >= 2;
        if (name == "continuous_batching_prefill_maximum_batch") prefill_merged = value >= 2;
        if (name == "continuous_batching_mtp_batches") mtp_batches = value;
        for (int depth = 0; depth <= 5; ++depth)
            if (name == "continuous_batching_mtp_depth_" + std::to_string(depth) + "_cycles") mtp_cycles += value;
        require(std::isfinite(value) && value >= 0, "batch metrics reported invalid counts or timings");
    }
    require(merged, "requests were serialized instead of batched");
    if (packed_prefill) require(prefill_merged, "short prefills were serialized instead of packed");
    require(mtp_cycles == mtp_batches && (!mtp || mtp_cycles > 0),
        "MTP cycle wall timing lost a membership transition or the final cycle");
    std::cout << '\n';
    if (prefix_cache) {
        require(!checkpoints.empty(), "batched requests did not publish prefix checkpoints");
        InferenceOutput output(inputs[0], nullptr, "reused");
        auto generation = batch.generate("reused", inputs[0], output);
        std::vector<std::int64_t> reused;
        for (int tick = 0; tick < 100; ++tick) {
            batch.step({"reused"});
            auto step = generation.next();
            if (!step) break;
            if (step.value) if (auto* delta = std::get_if<OutputDelta>(&*step.value))
                reused.insert(reused.end(), delta->token_ids.begin(), delta->token_ids.end());
        }
        require(restored_tokens > 0 && reused == expected[0], "batched prefix restore changed sampling output");
        require(batch.empty(), "prefix-restored request did not retire");
        std::cout << "Prefix checkpoint and restore passed, reused=" << restored_tokens << '\n';
    }
}
static void check_speculative(MlxQwen4CausalLm& runtime, int batch) {
    for (int depth = 1; depth <= 5; ++depth) {
        std::vector<std::shared_ptr<MlxQwen4RequestCache>> caches;
        std::vector<std::vector<std::int64_t>> histories, windows;
        std::vector<MlxQwen4TextSessionState> committed;
        std::vector<array> expected;
        for (int row = 0; row < batch; ++row) {
            auto history = prompt(batch == 8 ? 17 + row : 33 + row * 64, row + depth);
            auto window = prompt(depth + 1, row + 3);
            history[history.size() - 2] = runtime.config().eos_token_id;
            window[depth / 2] = runtime.config().eos_token_id;
            auto cache = runtime.make_request_cache();
            runtime.prefill_request(*cache, history);
            runtime.clear_cache();
            runtime.forward(array(history.begin(), {1, static_cast<int>(history.size())}, int32)).eval();
            auto checkpoint = runtime.capture_text_session_state(history);
            expected.push_back(runtime.forward(array(window.begin(), {1, depth + 1}, int32)));
            expected.back().eval();
            runtime.restore_text_session_state(checkpoint);
            const int accepted = batch == 1 ? depth / 2 : row == 0 ? 0 : row == 1 ? depth / 2 : depth;
            const std::vector<std::int64_t> prefix(window.begin(), window.begin() + accepted + 1);
            runtime.forward(array(prefix.begin(), {1, accepted + 1}, int32)).eval();
            history.insert(history.end(), prefix.begin(), prefix.end());
            committed.push_back(runtime.capture_text_session_state(history));
            histories.push_back(std::move(history));
            windows.push_back(std::move(window));
            caches.push_back(std::move(cache));
        }
        runtime.clear_cache();
        auto verified = runtime.verify_requests(caches, windows);
        eval(verified.first, verified.second);
        for (int row = 0; row < batch; ++row)
            near(slice(verified.first, {row, 0, 0}, {row + 1, depth + 1, verified.first.shape(2)}), expected[row]);
        bool rejected = false;
        try { runtime.decode_requests(caches, std::vector<std::int64_t>(batch, 3)); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "unfinished speculative transaction allowed another decode");
        for (int row = batch - 1; row >= 0; --row) {
            const int accepted = batch == 1 ? depth / 2 : row == 0 ? 0 : row == 1 ? depth / 2 : depth;
            runtime.resolve_request(*caches[row], accepted);
            const auto actual = runtime.capture_request_cache(*caches[row], histories[row]);
            require(actual.cache_position == committed[row].cache_position, "speculative positions crossed requests");
            for (std::size_t layer = 0; layer < actual.layers.size(); ++layer) {
                const auto& left = actual.layers[layer];
                const auto& right = committed[row].layers[layer];
                if (left.convolution) near(*left.convolution, *right.convolution);
                if (left.recurrent) near(*left.recurrent, *right.recurrent);
                if (left.ple_convolution) near(*left.ple_convolution, *right.ple_convolution);
                require(left.ple_context == right.ple_context, "speculative PLE history crossed requests");
            }
        }
        std::vector<array> next;
        for (int row = 0; row < batch; ++row) {
            runtime.restore_text_session_state(committed[row]);
            next.push_back(runtime.forward(array({7 + row}, {1, 1}, int32)));
            next.back().eval();
        }
        runtime.clear_cache();
        std::vector<std::int64_t> tokens;
        for (int row = 0; row < batch; ++row) tokens.push_back(7 + row);
        auto decoded = runtime.decode_requests(caches, tokens);
        for (int row = 0; row < batch; ++row)
            near(slice(decoded, {row, 0, 0}, {row + 1, 1, decoded.shape(2)}), next[row]);
    }
    runtime.clear_cache();
    require(runtime.kv_cache_bytes() == 0, "speculative request retirement leaked KV");
}
static void check_mtp_depth_metrics(MlxQwen4CausalLm& runtime) {
    ContinuousBatch<MlxQwen4BatchOperations> batch(64, 64, runtime, MlxQwen4BatchOperations::Hooks{});
    InferenceRequest input;
    input.chat = false;
    input.prompt = prompt(33, 1);
    input.sampling.enable_mtp = true;
    input.sampling.mtp_max_draft_tokens = 3;
    input.sampling.max_tokens = 96;
    input.sampling.temperature = 0;
    input.sampling.seed = 42;
    InferenceOutput output(input, nullptr, "depth-metrics");
    auto generation = batch.generate("depth-metrics", input, output);
    bool done = false;
    for (int iteration = 0; iteration < 1024; ++iteration) {
        batch.step({"depth-metrics"});
        if (!generation.next()) { done = true; break; }
        if (iteration == 12) {
            const auto bytes = runtime.kv_cache_bytes();
            batch.invalidate_runtime_policy();
            require(runtime.kv_cache_bytes() == bytes, "policy invalidation discarded active KV");
        }
    }
    require(done && batch.empty() && output.metrics.mtp.cycles > 0,
        "single-request MTP depth metric regression did not finish verification");
    std::array<std::uint64_t, 6> actual{};
    for (const auto& [name, value] : batch.metrics()) for (int depth = 0; depth <= 5; ++depth)
        if (name == "continuous_batching_mtp_depth_" + std::to_string(depth) + "_cycles")
            actual[depth] = static_cast<std::uint64_t>(value);
    require(actual[0] == output.metrics.mtp.standard_tokens,
        "ordinary MTP rounds were attributed to a selected speculative depth");
    for (int depth = 1; depth <= 5; ++depth)
        require(actual[depth] == output.metrics.mtp.depth_cycles[depth],
            "batch depth metrics disagree with the actually verified draft windows");
    std::cout << "MTP batch depth metrics match actual verified windows\n";
}
static void check_ragged_speculative(MlxQwen4CausalLm& runtime) {
    for (const auto lengths : {std::array{1, 2, 2}, std::array{2, 1, 2}, std::array{1, 2, 6},
        std::array{5, 1, 3}, std::array{3, 6, 1}}) {
        std::vector<std::shared_ptr<MlxQwen4RequestCache>> caches;
        std::vector<std::vector<std::int64_t>> histories, windows;
        std::vector<MlxQwen4TextSessionState> committed;
        std::vector<array> expected;
        for (int row = 0; row < 3; ++row) {
            auto history = prompt(33 + row * 32, row + 1);
            auto window = prompt(lengths[row], row + 3);
            window[lengths[row] / 2] = runtime.config().eos_token_id;
            auto cache = runtime.make_request_cache();
            runtime.prefill_request(*cache, history);
            runtime.clear_cache();
            runtime.forward(array(history.begin(), {1, static_cast<int>(history.size())}, int32)).eval();
            const auto state = runtime.capture_text_session_state(history);
            expected.push_back(runtime.forward(array(window.begin(), {1, lengths[row]}, int32)));
            expected.back().eval();
            runtime.restore_text_session_state(state);
            const int keep = 1 + (lengths[row] - 1) * row / 2;
            runtime.forward(array(window.begin(), {1, keep}, int32)).eval();
            history.insert(history.end(), window.begin(), window.begin() + keep);
            committed.push_back(runtime.capture_text_session_state(history));
            histories.push_back(std::move(history));
            windows.push_back(std::move(window));
            caches.push_back(std::move(cache));
        }
        runtime.clear_cache();
        std::optional<array> packed_logits;
        auto actual = runtime.forward_request_windows(caches, windows, &packed_logits);
        const int packed_rows = lengths[0] + lengths[1] + lengths[2];
        require(packed_logits && packed_logits->shape() == Shape{1, packed_rows, 32},
            "packed target logits lost ragged request offsets");
        auto selected = sample_greedy(reshape(*packed_logits, {packed_rows, 32}));
        selected.eval();
        int offset = 0;
        for (int row = 0; row < 3; ++row) {
            near(actual[row].first, expected[row]);
            auto independent = sample_greedy(reshape(actual[row].first, {lengths[row], 32}));
            require(all(equal(independent, slice(selected, {offset}, {offset + lengths[row]}))).item<bool>(),
                "packed greedy sampling mixed independent request windows");
            offset += lengths[row];
            if (lengths[row] > 1) runtime.resolve_request(*caches[row], (lengths[row] - 1) * row / 2);
            auto state = runtime.capture_request_cache(*caches[row], histories[row]);
            require(state.cache_position == committed[row].cache_position, "ragged verify advanced the wrong request length");
            for (std::size_t layer = 0; layer < state.layers.size(); ++layer) {
                const auto& left = state.layers[layer];
                const auto& right = committed[row].layers[layer];
                if (left.convolution) near(*left.convolution, *right.convolution);
                if (left.recurrent) near(*left.recurrent, *right.recurrent);
                if (left.ple_convolution) near(*left.ple_convolution, *right.ple_convolution);
                require(left.ple_context == right.ple_context, "ragged verify mixed PLE histories");
            }
        }
        const int position = runtime.request_cache_position(*caches[0]);
        bool rejected = false;
        try { runtime.forward_request_windows({caches[0], caches[0]}, {{1}, {2, 3}}); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected && position == runtime.request_cache_position(*caches[0]), "duplicate ragged cache mutated state");
        rejected = false;
        try { runtime.forward_request_windows(caches, {{1}, {}, {3}}); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected && position == runtime.request_cache_position(*caches[0]), "empty ragged window mutated state");
    }
    runtime.clear_cache();
    std::cout << "Ragged ordinary/speculative windows and independent rollback passed\n";
}
static void check_penalties(MlxQwen4CausalLm& runtime) {
    InferenceRequest input;
    input.sampling.enable_mtp = false;
    input.prompt = prompt(72, 3);
    input.sampling.temperature = .35;
    input.sampling.top_k = 20;
    input.sampling.seed = 20;
    input.sampling.presence_penalty = .2;
    input.sampling.frequency_penalty = .05;
    input.sampling.repetition_penalty = 1.1;
    MlxSampler sampler(mlx_batch_sampling(input.sampling));
    auto counts = sample_token_counts_add(zeros({32}, int32),
        array(input.prompt.begin(), {1, 72}, int32));
    runtime.clear_cache();
    auto logits = runtime.forward(array(input.prompt.begin(), {1, 72}, int32));
    auto selected = sampler.sample(mlx_last_token_logits(logits, 32), counts);
    selected.eval();
    const auto first = selected.item<std::int32_t>();
    counts = sample_token_counts_add(counts, array({first}, {1, 1}, int32));
    auto expected = runtime.forward(array({first}, {1, 1}, int32));
    expected.eval();
    auto request = std::make_shared<MlxQwen4BatchRequest>("penalties", input);
    MlxQwen4BatchOperations ops(runtime, {});
    ops.prepare(request);
    auto initial = ops.prefill(request, {0, 72, true});
    require(initial.token == first, "first penalized sample differs");
    ops.activate(request, initial);
    near(*request->counts, counts);
    request->pending_token = first;
    request->eligible = true;
    MlxQwen4BatchOperations::State state;
    state.active.push_back(request);
    InferenceRequest other;
    other.sampling.enable_mtp = false;
    other.prompt = prompt(59, 2);
    other.sampling.top_k = 20;
    auto companion = std::make_shared<MlxQwen4BatchRequest>("companion", other);
    ops.prepare(companion);
    auto companion_initial = ops.prefill(companion, {0, 59, true});
    ops.activate(companion, companion_initial);
    companion->pending_token = companion_initial.token;
    companion->eligible = true;
    state.active.push_back(companion);
    auto decoded = ops.decode(state);
    auto actual_logits = slice(decoded.logits, {0, 0, 0}, {1, 1, 32});
    near(actual_logits, expected);
    auto a = host_sampling_distribution(request->sampler.apply_penalties(
        mlx_last_token_logits(actual_logits, 32), *request->counts), sampler.params());
    auto b = host_sampling_distribution(sampler.apply_penalties(
        mlx_last_token_logits(expected, 32), counts), sampler.params());
    float difference = 0;
    for (int token = 0; token < 32; ++token) difference += std::abs(a[token] - b[token]);
    require(difference < .001f, "batched penalties changed the sampling distribution");
    auto actual = ops.sample(request, decoded);
    auto reference = sampler.sample(mlx_last_token_logits(actual_logits, 32), counts);
    require(actual.token == reference.item<std::int32_t>(), "batched sampling RNG or penalty counts differ");
    ops.retire({request, companion}, 0);
    runtime.clear_cache();
    std::cout << "Independent penalty counts, distribution and sampling RNG passed\n";
}
static void check_cancellation(MlxQwen4CausalLm& runtime, bool mtp = false) {
    struct Execution {
        ContinuousBatch<MlxQwen4BatchOperations> batch;
        explicit Execution(MlxQwen4CausalLm& runtime)
            : batch(16, 16, runtime, MlxQwen4BatchOperations::Hooks{}) {}
        bool can_batch(const EngineRequest&) const { return true; }
        bool exclusive() const { return false; }
        bool mtp_available() const { return false; }
        void execute(const std::vector<RequestId>& eligible) { batch.step(eligible); }
        Generation generate(const RequestId& id, ExecutionRequest& request) {
            return batch.generate(id, request.input, request.output);
        }
    } ops(runtime);
    RequestExecutor executor(2);
    EngineInfo info{2, 1024, 32};
    auto input = [mtp](std::string id, int length) {
        EngineRequest result;
        result.id = std::move(id);
        result.token_ids = prompt(length, 2);
        result.input.sampling.temperature = 0;
        result.input.sampling.top_k = 20;
        result.input.sampling.enable_mtp = mtp;
        result.input.sampling.max_tokens = 12;
        return result;
    };
    require(executor.admit(input("active", 17), nullptr, info, ops) == Admission::accepted,
        "first request was not admitted");
    require(executor.admit(input("prefill", 401), nullptr, info, ops) == Admission::accepted,
        "second request was not admitted");
    bool produced = false;
    for (int tick = 0; tick < 40 && !produced; ++tick) {
        auto step = executor.step({"active"}, ops);
        for (const auto& event : step.events) if (std::holds_alternative<OutputDelta>(event.data)) produced = true;
    }
    require(produced, "active request failed to produce before cancellation");
    for (int tick = 0; tick < 3; ++tick) executor.step({"prefill"}, ops);
    executor.cancel("active");
    executor.cancel("prefill");
    int cancelled = 0;
    for (int tick = 0; tick < 10 && !executor.empty(); ++tick) {
        auto step = executor.step({}, ops);
        require(step.status.healthy, "cancellation poisoned the engine");
        for (const auto& event : step.events) if (std::holds_alternative<Cancelled>(event.data)) ++cancelled;
    }
    require(executor.empty() && ops.batch.empty() && cancelled == 2 && runtime.kv_cache_bytes() == 0,
        "active or partially-prefilled cancellation leaked state");
    require(executor.admit(input("replacement", 7), nullptr, info, ops) == Admission::accepted,
        "cancelled slot could not be reused");
    bool completed = false;
    for (int tick = 0; tick < 100 && !executor.empty(); ++tick) {
        auto step = executor.step({"replacement"}, ops);
        for (const auto& event : step.events) if (std::holds_alternative<Completed>(event.data)) completed = true;
    }
    require(completed && executor.empty() && runtime.kv_cache_bytes() == 0, "replacement request retained KV");
    std::cout << "Cancellation, paused output and slot reuse passed\n";
}
static void check_expert_budget_transition(MlxQwen4CausalLm& runtime) {
    std::vector<std::shared_ptr<MlxQwen4RequestCache>> caches;
    std::vector<MlxQwen4TextSessionState> committed;
    std::vector<std::vector<std::int64_t>> histories, windows;
    std::vector<array> next;
    for (int row = 0; row < 2; ++row) {
        auto history = prompt(33 + row * 16, row);
        auto window = prompt(3, row + 4);
        auto cache = runtime.make_request_cache();
        runtime.prefill_request(*cache, history, true);
        runtime.clear_cache();
        runtime.forward(array(history.begin(), {1, static_cast<int>(history.size())}, int32)).eval();
        const std::vector<std::int64_t> accepted(window.begin(), window.begin() + (row ? 3 : 1));
        runtime.forward(array(accepted.begin(), {1, static_cast<int>(accepted.size())}, int32)).eval();
        history.insert(history.end(), accepted.begin(), accepted.end());
        committed.push_back(runtime.capture_text_session_state(history));
        next.push_back(runtime.forward(array({7 + row}, {1, 1}, int32)));
        next.back().eval();
        histories.push_back(std::move(history));
        windows.push_back(std::move(window));
        caches.push_back(std::move(cache));
    }
    runtime.clear_cache();
    auto verified = runtime.verify_requests(caches, windows);
    eval(verified.first, verified.second);
    const auto resident = runtime.resident_full_expert_bytes();
    const auto budget = std::max<std::size_t>(1, (resident ? resident : runtime.expert_cache_limit_bytes()) / 2);
    runtime.set_expert_cache_limit(budget);
    require(runtime.resident_full_expert_bytes() <= budget &&
        (!resident || runtime.resident_full_expert_bytes() < resident),
        "budget pressure failed to reduce resident expert layers");
    require(runtime.expert_cache_limit_bytes() <= budget &&
        runtime.resident_full_expert_bytes() + runtime.dynamic_weight_bytes() <= budget,
        "expert cache transition exceeded its budget");
    for (int row = 0; row < 2; ++row) {
        runtime.resolve_request(*caches[row], row ? 2 : 0);
        const auto actual = runtime.capture_request_cache(*caches[row], histories[row]);
        require(actual.cache_position == committed[row].cache_position,
            "expert budget transition changed a speculative position");
        for (std::size_t layer = 0; layer < actual.layers.size(); ++layer) {
            const auto& left = actual.layers[layer];
            const auto& right = committed[row].layers[layer];
            if (left.convolution) near(*left.convolution, *right.convolution);
            if (left.recurrent) near(*left.recurrent, *right.recurrent);
            if (left.ple_convolution) near(*left.ple_convolution, *right.ple_convolution);
            require(left.ple_context == right.ple_context, "expert budget transition changed PLE history");
        }
    }
    auto decoded = runtime.decode_requests(caches, {7, 8});
    for (int row = 0; row < 2; ++row)
        near(slice(decoded, {row, 0, 0}, {row + 1, 1, decoded.shape(2)}), next[row]);
    require(runtime.resident_full_expert_bytes() + runtime.dynamic_weight_bytes() <= budget,
        "streamed decode grew beyond its expert budget");
    caches.clear();
    committed.clear();
    runtime.clear_cache();
    check_draft_heads(runtime);
    require(runtime.kv_cache_bytes() == 0, "budget transition retained retired request KV");
    std::cout << "Expert budget transition preserves active KV, speculative rollback and draft heads\n";
}
int main(int argc, char** argv) {
    try {
        if (argc != 2 && argc != 3) throw std::invalid_argument("usage: mfq-metal-qwen4-batching-test MODEL [ssd]");
        set_default_device(Device::gpu);
        MfqContainer source(argv[1]);
        auto runtime = MlxQwen4CausalLm::load(source, 1024,
            argc == 3 ? std::optional<std::size_t>(1024) : std::nullopt);
        check_ready_decode(runtime);
        check_draft_heads(runtime);
        check_packed_prefill(runtime);
        check_state(runtime);
        check_state(runtime, 8);
        std::cout << "Independent request state comparison passed\n";
        check_speculative(runtime, 1);
        check_speculative(runtime, 3);
        check_speculative(runtime, 8);
        check_mtp_depth_metrics(runtime);
        check_ragged_speculative(runtime);
        std::cout << "Independent speculative verification and rollback passed\n";
        check_penalties(runtime);
        check_scheduler(runtime, false);
        check_scheduler(runtime, true);
        check_scheduler(runtime, false, true);
        check_scheduler(runtime, true, true);
        check_scheduler(runtime, false, false, true);
        check_scheduler(runtime, false, true, true);
        check_cancellation(runtime);
        check_cancellation(runtime, true);
        check_expert_budget_transition(runtime);
        std::cout << "Metal continuous batching state and scheduling passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
