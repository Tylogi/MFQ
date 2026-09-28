#include "continuous_batching.h"
#include "generation_policy.h"
#include "inference.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {
void require(bool ok) {
    if (!ok) throw std::runtime_error("inference test failed");
}

struct FakeModel {
    std::vector<std::string> calls;
    std::vector<std::vector<std::int64_t>> snapshots;
    std::int64_t position = 0;
    bool cache = true;
    bool transformed = false;
    bool fail_decode = false;
    std::function<void()> after_prefill;
    std::string input_key;
    std::vector<std::string> snapshot_keys;

    bool supports_cache() const {
        return cache && (!transformed || !input_key.empty());
    }
    bool persistent_prefix_enabled() const { return false; }
    std::size_t restore(std::size_t stable) {
        require(stable == 3);
        calls.push_back("restore");
        position = 2;
        return 2;
    }
    void reset() {
        calls.push_back("reset");
        position = 0;
    }
    std::int64_t cache_position() const { return position; }
    std::size_t prompt_size() const { return 4; }
    void snapshot(std::vector<std::int64_t> tokens) {
        calls.push_back("snapshot");
        snapshots.push_back(std::move(tokens));
        snapshot_keys.push_back(input_key);
    }
    mfq::engine::PrefillResult prefill(
            std::size_t reused, std::size_t stable,
            const std::function<void(std::size_t)>& checkpoint) {
        calls.push_back("prefill");
        require(reused == (supports_cache() ? 2u : 0u));
        if (stable) {
            position = static_cast<std::int64_t>(stable);
            checkpoint(stable);
        }
        position = 4;
        if (after_prefill) after_prefill();
        return {41, {4 - reused, 1.0, transformed ? 2.0 : 0.0,
                     transformed ? 3.0 : 1.0}};
    }
    std::int64_t advance() {
        calls.push_back("advance");
        if (fail_decode) throw std::runtime_error("decode failed");
        ++position; // The previous emitted token has now entered the cache.
        return 42;
    }
    void accept(std::int64_t token) {
        calls.push_back("accept");
        require(token == 41 || token == 42);
    }
    std::int32_t generate(
            std::size_t reused, std::size_t stable,
            const std::function<void(std::size_t)>& checkpoint,
            const MfqTokenCallback& emit,
            const MfqPrefillCallback& on_prefill, std::int32_t limit,
            const MfqCancellationCheck& cancelled) {
        return mfq::engine::generate_target(
            *this, reused, stable, checkpoint, emit, on_prefill, limit,
            cancelled);
    }
};

struct FakeMtp : FakeModel {
    std::int32_t generate(
            std::size_t reused, std::size_t,
            const std::function<void(std::size_t)>&,
            const MfqTokenCallback& emit,
            const MfqPrefillCallback& on_prefill, std::int32_t,
            const MfqCancellationCheck&) {
        calls.push_back("mtp");
        require(reused == (supports_cache() ? 2u : 0u));
        position = 4; // The prompt is evaluated; the first sample is pending.
        if (fail_decode) throw std::runtime_error("MTP failed after prefill");
        if (on_prefill) on_prefill({2, 1.0, 0.0, 1.0});
        if (!emit(41)) return 1;
        position = 5; // Target verification commits the pending token.
        (void)emit(42); // The replacement/bonus is not evaluated yet.
        return 2;
    }
};
}

int main() {
    const std::vector<std::int64_t> planned_prompt{1, 2};
    const auto generation_plan = mfq::engine::plan_generation(
        planned_prompt, 4, 3, 5, 9);
    require(generation_plan.prompt_tokens == 2 &&
        generation_plan.stable_prefix_tokens == 2 &&
        generation_plan.generation_tokens == 1);
    const auto first_chunk = mfq::engine::next_prefill_chunk(5, 0, 2);
    const auto last_chunk = mfq::engine::next_prefill_chunk(5, 4, 2);
    require(first_chunk.offset == 0 && first_chunk.count == 2 &&
        !first_chunk.last && last_chunk.offset == 4 &&
        last_chunk.count == 1 && last_chunk.last);
    bool invalid_plan = false;
    try {
        (void)mfq::engine::plan_generation(
            std::span<const std::int64_t>{planned_prompt.data(), 1},
            1, 3, 1);
    } catch (const std::invalid_argument&) {
        invalid_plan = true;
    }
    require(invalid_plan);

    mfq::engine::ContinuousBatchRequest batch_request({1, 2}, {}, {}, {});
    std::thread producer([&] {
        batch_request.publish_prefill({2, 1.0, 0.0, 1.0});
        batch_request.publish_token(3);
        batch_request.complete();
    });
    bool batch_prefilled = false;
    const auto batch_tokens = batch_request.consume(
        [](int64_t token) { return token == 3; },
        [&](const MfqPrefillTiming& timing) {
            batch_prefilled = timing.prompt_tokens == 2;
        });
    producer.join();
    require(batch_tokens == 1 && batch_prefilled);

    using BatchRequest = mfq::engine::ContinuousBatchRequest;
    mfq::engine::ContinuousBatchQueue<BatchRequest> queue(2);
    auto first = std::make_shared<BatchRequest>(
        std::vector<int64_t>{1}, MfqSamplingParams{}, nullptr);
    auto second = std::make_shared<BatchRequest>(
        std::vector<int64_t>{2}, MfqSamplingParams{}, nullptr);
    auto third = std::make_shared<BatchRequest>(
        std::vector<int64_t>{3}, MfqSamplingParams{}, nullptr);
    queue.submit(first);
    queue.submit(second);
    queue.submit(third);
    require(queue.wait_for_work(false, std::chrono::microseconds(0)));
    const auto admitted = queue.take(1, 2);
    require(admitted.size() == 1 && admitted.front() == first);
    require(queue.size() == 2);
    const auto drained = queue.stop_and_drain();
    require(drained.size() == 2 && queue.stopping());
    bool rejected_submit = false;
    try {
        queue.submit(first);
    } catch (const std::runtime_error&) {
        rejected_submit = true;
    }
    require(rejected_submit);

    const std::vector<std::int64_t> prompt{1, 2, 3, 4};
    MfqSamplingParams sampling;
    sampling.max_tokens = 2;
    MfqPromptCachePlan plan{"session", 3};
    FakeModel model;
    MfqPrefillTiming timing;
    std::vector<std::int64_t> output;
    const auto generated = mfq::engine::generate(
        model, prompt, sampling,
        [&](std::int64_t token) { output.push_back(token); return true; },
        [&](const MfqPrefillTiming& value) { timing = value; }, plan);
    require(generated == 2 && (output == std::vector<std::int64_t>{41, 42}));
    require(timing.prompt_tokens == 2);
    require((model.snapshots == std::vector<std::vector<std::int64_t>>{
        {1, 2, 3}, {1, 2, 3, 4, 41}}));
    require((model.calls == std::vector<std::string>{
        "restore", "prefill", "snapshot", "accept", "advance",
        "accept", "snapshot"}));

    // A stopped token is counted, but never claimed as evaluated KV history.
    FakeModel stopped;
    const auto stopped_count = mfq::engine::generate(
        stopped, prompt, sampling, [](std::int64_t) { return false; },
        {}, plan);
    require(stopped_count == 1 && stopped.snapshots.back() == prompt);
    require(stopped.calls.end() == std::find(
        stopped.calls.begin(), stopped.calls.end(), "advance"));

    FakeModel stopped_after_decode;
    require(mfq::engine::generate(
        stopped_after_decode, prompt, sampling,
        [](std::int64_t token) { return token == 41; }, {}, plan) == 2);
    require(stopped_after_decode.snapshots.back() ==
        (std::vector<std::int64_t>{1, 2, 3, 4, 41}));

    FakeMtp speculative;
    std::vector<std::int64_t> speculative_output;
    require(mfq::engine::generate(
        speculative, prompt, sampling,
        [&](std::int64_t token) {
            speculative_output.push_back(token);
            return true;
        }, {}, plan) == 2);
    require((speculative_output == std::vector<std::int64_t>{41, 42}));
    require((speculative.snapshots == std::vector<std::vector<std::int64_t>>{
        {1, 2, 3, 4, 41}}));
    require((speculative.calls == std::vector<std::string>{
        "restore", "mtp", "snapshot"}));

    FakeMtp speculative_stop;
    require(mfq::engine::generate(
        speculative_stop, prompt, sampling,
        [](std::int64_t token) { return token == 41; }, {}, plan) == 2);
    require(speculative_stop.snapshots.back() ==
        (std::vector<std::int64_t>{1, 2, 3, 4, 41}));

    FakeMtp visual_mtp;
    visual_mtp.transformed = true;
    visual_mtp.input_key = "image|positions";
    require(mfq::engine::generate(
        visual_mtp, prompt, sampling, {}, {}, plan) == 2);
    require(visual_mtp.snapshot_keys.back() == visual_mtp.input_key);

    FakeMtp unknown_mtp_media;
    unknown_mtp_media.transformed = true;
    require(mfq::engine::generate(
        unknown_mtp_media, prompt, sampling, {}, {}, plan) == 2);
    require(unknown_mtp_media.snapshots.empty() &&
            unknown_mtp_media.calls.front() == "reset");

    FakeMtp failed_mtp;
    failed_mtp.fail_decode = true;
    bool mtp_caught = false;
    try {
        (void)mfq::engine::generate(
            failed_mtp, prompt, sampling, {}, {}, plan);
    } catch (const std::runtime_error&) {
        mtp_caught = true;
    }
    require(mtp_caught && failed_mtp.position == 0 &&
        failed_mtp.calls.back() == "reset");

    // Media and text use the same lifecycle, but transformed KV requires
    // a stable identity (pixels and position policy) before restoration.
    FakeModel visual;
    visual.transformed = true;
    visual.input_key = "image|positions";
    MfqPrefillTiming visual_timing;
    require(mfq::engine::generate(
        visual, prompt, sampling, {},
        [&](const MfqPrefillTiming& value) { visual_timing = value; },
        plan) == 2);
    require(visual_timing.multimodal_ms == 2.0 &&
            visual_timing.model_ms == 3.0 &&
            visual.snapshot_keys.back() == visual.input_key);

    FakeModel unknown_media;
    unknown_media.transformed = true;
    require(mfq::engine::generate(
        unknown_media, prompt, sampling, {}, {}, plan) == 2);
    require(unknown_media.snapshots.empty() &&
            unknown_media.calls.front() == "reset");

    FakeModel uncached;
    uncached.cache = false;
    require(mfq::engine::generate(
        uncached, prompt, sampling, {}, {}, plan) == 2);
    require(uncached.snapshots.empty() && uncached.calls.front() == "reset");

    FakeModel cancelled_before;
    require(mfq::engine::generate(
        cancelled_before, prompt, sampling, {}, {}, plan,
        [] { return true; }) == 0);
    require(cancelled_before.calls.empty());

    FakeModel cancelled_during_prefill;
    cancelled_during_prefill.after_prefill = [] {
        throw mfq::engine::InferenceCancelled{};
    };
    require(mfq::engine::generate(
        cancelled_during_prefill, prompt, sampling, {}, {}, plan) == 0);
    require(cancelled_during_prefill.calls.back() == "reset");

    FakeModel cancelled_after_prefill;
    bool cancel_requested = false;
    cancelled_after_prefill.after_prefill = [&] { cancel_requested = true; };
    require(mfq::engine::generate(
        cancelled_after_prefill, prompt, sampling, {}, {}, plan,
        [&] { return cancel_requested; }) == 0);
    require(!cancelled_after_prefill.snapshots.empty());
    require(cancelled_after_prefill.snapshots.back() == prompt);
    require(std::find(
        cancelled_after_prefill.calls.begin(),
        cancelled_after_prefill.calls.end(), "advance") ==
        cancelled_after_prefill.calls.end());

    FakeModel failed;
    failed.fail_decode = true;
    bool caught = false;
    try {
        (void)mfq::engine::generate(
            failed, prompt, sampling, {}, {}, plan);
    } catch (const std::runtime_error&) {
        caught = true;
    }
    require(caught && failed.calls.back() == "reset" && failed.position == 0);
}
