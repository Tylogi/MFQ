#include "inference.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
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
            const MfqPrefillCallback& on_prefill, std::int32_t limit) {
        return mfq::engine::generate_target(
            *this, reused, stable, checkpoint, emit, on_prefill, limit);
    }
};

struct FakeMtp : FakeModel {
    std::int32_t generate(
            std::size_t reused, std::size_t,
            const std::function<void(std::size_t)>&,
            const MfqTokenCallback& emit,
            const MfqPrefillCallback& on_prefill, std::int32_t) {
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
