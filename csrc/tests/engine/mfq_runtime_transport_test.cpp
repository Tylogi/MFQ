#include "scheduler.h"
#include "transport.h"
#include "common.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <thread>
#include <tuple>
#ifndef _WIN32
#include <unistd.h>
#endif
using namespace mfq::engine;
using namespace std::chrono_literals;

struct FakeEngine final : Engine {
    struct Work { EngineRequest request; int steps = 0; bool cancelled = false; };
    std::map<std::string, Work> active;
    std::array<std::atomic<int>, 16> advances{};
    std::atomic<int> released{0}, reloads{0}, control_steps{0}, control_terminals{0}, stops{0};
    bool duplex_active = false, control_pending = false, control_stop = false, initializing = false;
    int control_limit = 0;
    std::atomic<bool> blocked{false}, resume{false};
    std::thread::id owner;
    bool fail = false, supports_duplex = false;
    MfqDuplexSessionParams last_parameters;
    std::atomic<bool> defer{false};
    std::atomic<const float*> media{nullptr};
    EngineInfo info() const override {
        EngineInfo info{2, 128, 64, false, supports_duplex, true, {}};
        info.backend = "fake";
        info.chat = {true, {"low", "high"}};
        return info;
    }
    void check_thread() {
        if (owner == std::thread::id{}) owner = std::this_thread::get_id();
        assert(owner == std::this_thread::get_id());
    }
    Admission admit(EngineRequest&& request) override {
        check_thread();
        if (defer || active.size() == 2) return Admission::deferred;
        if (request.input.media) media = request.input.media->pixel_values.data();
        auto id = request.id; active.emplace(id, Work{std::move(request)});
        return Admission::accepted;
    }
    void cancel(const RequestId& id) override {
        check_thread(); if (active.contains(id)) active.at(id).cancelled = true;
    }
    EngineStatus status() const override { return {2 - active.size(), true}; }
    EngineStepResult step(const std::vector<RequestId>& eligible) override {
        check_thread();
        EngineStepResult result;
        for (auto it = active.begin(); it != active.end();) {
            auto& work = it->second;
            const auto id = it->first;
            if (work.cancelled) {
                it = active.erase(it); ++released;
                result.events.push_back({id, Cancelled{}}); continue;
            }
            if (std::find(eligible.begin(), eligible.end(), id) == eligible.end()) { ++it; continue; }
            if (work.request.diagnostic_input && work.request.diagnostic_input->tokens == std::vector<int64_t>{59} && !resume) {
                blocked = true;
                while (!resume) std::this_thread::sleep_for(1ms);
            }
            if(work.request.diagnostic_input)++advances.at(std::stoi(id));
            ++work.steps;
            if (!work.request.diagnostic_input) {
                const auto& content = work.request.input.chat_input->messages.front().content;
                if (content != "hello") {
                    result.events.push_back({id, Failed{content, "fake request failure"}});
                } else if (work.steps <= 2) {
                    OutputDiff diff;
                    diff.content_delta = work.steps == 1 ? "hel" : "lo";
                    diff.reasoning_content_delta = work.steps == 1 ? "think" : "ing";
                    diff.tool_call_index = 0;
                    diff.tool_call_delta = work.steps == 1
                        ? ToolCall{"lookup", "{\"ok\":", "call_test"}
                        : ToolCall{"", "true}", ""};
                    result.events.push_back({id, OutputDelta{{1,2,3}, {std::move(diff)}}});
                    result.advanced.push_back(id);
                    ++it;
                    continue;
                } else {
                    InferenceMetrics metrics;
                    metrics.mtp.available = true;
                    metrics.mtp.cycles = 2;
                    result.events.push_back({id, Completed{"tool_calls", {5, 2}, metrics}});
                }
                it = active.erase(it);
                ++released;
                continue;
            }
            const auto& tokens = work.request.diagnostic_input->tokens;
            if (tokens == std::vector<int64_t>{63}) throw std::runtime_error("device failure");
            result.advanced.push_back(id);
            if (work.steps <= int(tokens.size()))
                result.events.push_back({id, PrefillProgress{{std::size_t(work.steps), 1, 0, 1}}});
            else if (work.steps <= int(tokens.size()) + work.request.input.sampling.max_tokens) {
                OutputDelta delta{{1, 2, 3}, {}};
                if (tokens == std::vector<int64_t>{60} ||
                        tokens == std::vector<int64_t>{61} ||
                        tokens == std::vector<int64_t>{62}) {
                    OutputDiff diff;
                    const auto marker = tokens.front();
                    diff.content_delta.assign(marker == 60 ? (work.steps == 2 ? 2000 : 6000)
                        : (marker == 61 ? 1024 : 8192), 'x');
                    delta.diffs.push_back(std::move(diff));
                }
                result.events.push_back({id, std::move(delta)});
            }
            if (work.steps > int(tokens.size()) + work.request.input.sampling.max_tokens ||
                    (tokens == std::vector<int64_t>{60} &&
                     work.steps == int(tokens.size()) + work.request.input.sampling.max_tokens)) {
                it = active.erase(it); ++released;
                InferenceMetrics metrics;
                metrics.mtp.available = true;
                metrics.mtp.cycles = std::stoi(id);
                result.events.push_back({id, Completed{{}, {}, metrics}}); continue;
            }
            ++it;
        }
        if (control_stop) {
            if (control_pending) { result.control = Cancelled{}; ++control_terminals; }
            control_pending = duplex_active = control_stop = false;
        } else if (control_pending) {
            ++control_steps;
            result.control_advanced = true;
            if (control_steps >= control_limit) {
                MfqDuplexStepResult output;
                output.generated_tokens = {1, 2, 3};
                output.audio_tokens = {4, 5};
                output.end_of_turn = output.tts_force_flush = true;
                result.control = initializing ? ControlCompletion{std::monostate{}} : ControlCompletion{std::move(output)};
                control_pending = false;
                ++control_terminals;
            }
            std::this_thread::sleep_for(1ms); // One finite device quantum in the fake backend.
        }
        result.has_control_work = control_pending || control_stop;
        result.status = status(); result.has_work = !active.empty() || result.has_control_work; return result;
    }
    SessionResult session(const SessionCommand&) override { check_thread(); return {7, {{"test_session_only", 99}}}; }
    int64_t reload(int64_t context) override {
        check_thread(); assert(active.empty());
        if (context == 13) throw std::runtime_error("load failed");
        ++reloads; return context;
    }
    void shutdown() override { check_thread(); assert(active.empty()); }
    ControlResult control(ControlRequest request) override {
        check_thread();
        if (std::holds_alternative<RuntimeMetrics>(request)) return Metrics{{"test_health_only", 99}};
        if (auto* prepare = std::get_if<PrepareDuplex>(&request)) {
            last_parameters = prepare->parameters;
            prepare->parameters.special_ids.assign(15, 1);
            return std::move(prepare->parameters);
        }
        if (auto* prepare = std::get_if<PrepareDuplexStep>(&request)) {
            if (prepare->text == "invalid") throw InferenceInputError(InferenceInputErrorCode::Invalid, "invalid text");
            prepare->input.text_tokens = {1};
            return std::move(prepare->input);
        }
        if (auto* decode = std::get_if<DecodeTokens>(&request))
            return decode->tokens.empty() ? std::string{} : std::string{"hello"};
        if (std::holds_alternative<StopDuplex>(request)) { ++stops; control_stop = true; return std::monostate{}; }
        assert(!control_pending);
        initializing = std::holds_alternative<MfqDuplexSessionParams>(request);
        control_limit = initializing ? 3 : std::get<MfqDuplexStepInput>(request).max_new_speak_tokens;
        control_steps = 0;
        control_pending = duplex_active = true;
        return ControlPending{};
    }
};
EngineRequest request(int id, int tokens = 12) {
    EngineRequest result;
    result.id = std::to_string(id); result.diagnostic_input = DiagnosticTokenInput{{1, 2}};
    result.input.sampling.max_tokens = tokens; return result;
}
int drain(const std::shared_ptr<MfqScheduledRequest>& handle, bool cancelled = false, bool failed = false) {
    int terminals = 0;
    while (!handle->done()) for (const auto& event : handle->wait()) {
        if (const auto* delta = std::get_if<OutputDelta>(&event.data))
            assert(delta->token_ids == std::vector<int64_t>({1, 2, 3}));
        if (terminal(event.data)) {
            ++terminals;
            assert(std::holds_alternative<Cancelled>(event.data) == cancelled);
            assert(std::holds_alternative<Failed>(event.data) == failed);
            if (const auto* completed = std::get_if<Completed>(&event.data)) {
                assert(completed->metrics.mtp.available);
                assert(completed->metrics.mtp.cycles == std::stoul(event.id));
            }
        }
    }
    assert(terminals == 1);
    assert(handle->wait().empty());
    return terminals;
}
void test_model_capabilities() {
    ModelCapabilities capabilities;
    capabilities.family = "test-family";
    capabilities.image_input = true;
    capabilities.audio_output = true;
    capabilities.mtp = true;
    capabilities.source = "engine-test";
    const auto value =
        mfq::transport::model_capability_profile_json(capabilities);
    assert(value["architecture_family"] == "test-family");
    assert(value["source"] == "engine-test");
    assert(value["features"]["text"] == true);
    assert(value["features"]["image_input"] == true);
    assert(value["features"]["audio_output"] == true);
    assert(value["features"]["mtp"] == true);
    assert(value["features"]["video_input"] == false);
}

void test_decode_timing() {
    const auto now = InferenceMetrics::Clock::now();
    InferenceMetrics metrics;
    metrics.started = now - 80ms;
    metrics.first_token = now - 70ms;
    metrics.last_token = now - 20ms;
    metrics.saw_token = true;
    mfq::transport::CompletionResult result;
    result.completion_tokens = 6;
    const auto values = mfq::transport::request_metric_values(result, metrics);
    assert(std::abs(values.ttft_ms - 10.0) < 1e-12);
    assert(std::abs(values.decode_ms - 50.0) < 1e-12);
    assert(std::abs(values.decode_tps - 100.0) < 1e-12);
    assert(values.generation_ms >= 80.0);
    assert(values.generation_ms - values.ttft_ms - values.decode_ms >= 20.0);
    result.cancelled = true;
    assert(mfq::transport::request_metric_values(result, metrics).decode_ms == values.decode_ms);
    InferenceMetrics single;
    single.mark_token();
    result.completion_tokens = 1;
    const auto one = mfq::transport::request_metric_values(result, single);
    assert(one.decode_ms == 0.0 && one.decode_tps == 0.0);
    result.completion_tokens = 0;
    const auto empty = mfq::transport::request_metric_values(result, InferenceMetrics{});
    assert(empty.decode_ms == 0.0 && empty.decode_tps == 0.0);
}

void test_benchmark_input_contract() {
    using namespace mfq::transport;
    json params = {{"input", {{"messages", json::array({{{"role", "user"}, {"content", "test input"}}})},
        {"preformatted_prompt", "test input"}, {"benchmark_prompt_tokens", 32}}}, {"cache", {{"enabled", false}}}};
    auto body = runtime_generate_body(params);
    const auto input = parse_input(body, true, {});
    assert(!input.prefix_cache_enabled && input.benchmark_prompt_tokens == 32);
    body.erase("mfq_benchmark_prompt_tokens");
    body.erase("mfq_prefix_cache_enabled");
    const auto normal = parse_input(body, true, {});
    assert(normal.prefix_cache_enabled && !normal.benchmark_prompt_tokens);
    for (const auto& invalid : std::vector<json>{
            {{"mfq_benchmark_prompt_tokens", 0}, {"mfq_prefix_cache_enabled", false}, {"mfq_preformatted_prompt", "test"}},
            {{"mfq_benchmark_prompt_tokens", 32}, {"mfq_prefix_cache_enabled", true}, {"mfq_preformatted_prompt", "test"}},
            {{"mfq_benchmark_prompt_tokens", 32}, {"mfq_prefix_cache_enabled", false}, {"messages", json::array()}}}) {
        bool rejected = false;
        try { parse_input(invalid, true, {}); } catch (const ApiError&) { rejected = true; }
        assert(rejected);
    }
}

void test_probability_contract() {
    using namespace mfq::transport;
    const auto request = parse_score_request({{"prompt", "Q:\nA:"}, {"continuations", {"A", "B"}}, {"mode", "next_token"}});
    assert(request.next_token && request.prompt == "Q:\nA:" && request.continuations.size() == 2);
    for (const auto& invalid : std::vector<json>{
            {{"prompt", ""}, {"continuations", {"A"}}},
            {{"prompt", "x"}, {"continuations", json::array()}},
            {{"prompt", "x"}, {"continuations", {1}}},
            {{"prompt", "x"}, {"continuations", {"A"}}, {"mode", 1}},
            {{"prompt", "x"}, {"continuations", {"A"}}, {"mode", "generate"}},
            {{"prompt", "x"}, {"continuations", {"A"}}, {"temperature", 1}}}) {
        bool rejected = false;
        try { (void)parse_score_request(invalid); } catch (const ApiError& error) { rejected = error.status == 400; }
        assert(rejected);
    }
    const auto result = likelihood_result_json({3, {{{1, 2}, {-1., -2.}, -3.}}});
    assert(result["log_base"] == "e" && result["scores"][0]["log_likelihood"] == -3);
}

void test_chat_input_contract() {
    using mfq::transport::ApiError;
    using mfq::transport::json;
    json body = {
        {"messages", json::array({{{"role", "user"}, {"content", "hello"}}})},
        {"tools", json::array({
            {{"type", "function"}, {"function", {{"name", "lookup"}, {"description", "lookup"}, {"parameters", {{"type", "object"}}}}}},
            {{"type", "function"}, {"function", {{"name", "other"}, {"parameters", {{"type", "object"}}}}}},
        })},
        {"tool_choice", {{"type", "function"}, {"function", {{"name", "lookup"}}}}},
        {"response_format", {{"type", "json_schema"}, {"json_schema", {{"schema", {{"type", "object"}}}}}}},
        {"reasoning_format", "deepseek"},
        {"chat_template_kwargs", {{"mode", "strict"}}},
    };
    auto input = mfq::transport::parse_input(
        body, true, mfq::SamplingParams{});
    assert(input.chat_input);
    const auto& chat = *input.chat_input;
    assert(chat.messages.size() == 1);
    assert(chat.messages[0].role == "user");
    assert(chat.messages[0].content == "hello");
    assert(chat.tools.size() == 1);
    assert(chat.tools[0].name == "lookup");
    assert(chat.tool_choice == ChatToolChoice::required);
    assert(json::parse(chat.json_schema)["type"] == "object");
    assert(chat.reasoning_format == ReasoningFormat::deepseek);
    assert(chat.template_kwargs.at("mode") == "\"strict\"");

    body["tool_choice"]["function"]["name"] = "missing";
    bool rejected = false;
    try {
        (void) mfq::transport::parse_input(
            body, true, mfq::SamplingParams{});
    } catch (const ApiError& error) {
        rejected = error.status == 400 && error.param == "tool_choice";
    }
    assert(rejected);
}

static void check_generation_protocol() {
    using namespace mfq::transport;
    FakeEngine engine;
    MfqScheduler scheduler(engine);
    MfqRuntimeTransportConfig config;
    RuntimeRequestMetrics metrics;
    json params = {
        {"input", {{"messages", json::array({{{"role", "user"}, {"content", "hello"}}})}}},
        {"sampling", {{"max_new_tokens", 2}, {"seed", 1}}},
        {"stream", false}, {"include_usage", true},
    };
    {
        CompletionStream plain(scheduler, config, metrics, params, "plain");
        assert(metrics.active_requests() == 1);
        const auto frames = plain.next();
        assert(!plain.streaming() && frames && frames->size() == 1);
        const auto& result = frames->front();
        assert(result["output"]["text"] == "hello");
        assert(result["output"]["reasoning"] == "thinking");
        const auto& call = result["output"]["tool_calls"][0];
        assert(call["id"] == "call_test");
        assert(call["function"]["name"] == "lookup");
        assert(call["function"]["arguments"] == "{\"ok\":true}");
        assert(result["usage"]["prompt_tokens"] == 5);
        assert(result["usage"]["completion_tokens"] == 2);
        assert(result["finish_reason"] == "tool_calls");
        assert(result["metrics"]["mtp_cycles"] == 2);
        assert(!plain.next() && !plain.next());
    }
    params["stream"] = true;
    for (bool include_usage : {true, false}) {
        params["include_usage"] = include_usage;
        CompletionStream stream(scheduler, config, metrics, params, "stream");
        assert(stream.streaming());
        std::vector<json> frames;
        while (auto batch = stream.next())
            for (auto& frame : *batch) frames.push_back(std::move(frame));
        assert(frames.size() == (include_usage ? 4 : 3));
        assert(frames[0]["event"] == "delta" && frames[1]["event"] == "delta");
        assert(frames[0]["delta"]["content"].get<std::string>() +
               frames[1]["delta"]["content"].get<std::string>() == "hello");
        assert(frames[2]["event"] == "complete");
        assert(frames[2]["finish_reason"] == "tool_calls");
        assert(frames[2]["metrics"]["mtp_cycles"] == 2);
        if (include_usage) assert(frames[3]["usage"]["total_tokens"] == 7);
        for (const auto& frame : frames) {
            assert(frame["request_id"] == frames[0]["request_id"]);
            assert(frame["created"] == frames[0]["created"]);
        }
        assert(!stream.next());
    }
    for (const auto& [code, status, type] : {
            std::tuple{"invalid_request", 400, "invalid_request_error"},
            std::tuple{"unsupported_input", 501, "unsupported_parameter"},
            std::tuple{"resource_exhausted", 503, "resource_exhausted"},
            std::tuple{"backend_failure", 500, "server_error"}}) {
        params["input"]["messages"][0]["content"] = code;
        CompletionStream stream(scheduler, config, metrics, params, "error");
        bool failed = false;
        try { (void)stream.next(); }
        catch (const ApiError& error) {
            failed = error.status == status && error.type == type;
        }
        assert(failed && !stream.next());
        assert(metrics.active_requests() == 0);
    }
    params["input"]["messages"][0]["content"] = "hello";
    engine.defer = true;
    {
        CompletionStream disconnected(scheduler, config, metrics, params, "disconnected");
        disconnected.cancel();
        disconnected.cancel();
        assert(!disconnected.next());
    }
    { CompletionStream abandoned(scheduler, config, metrics, params, "abandoned"); }
    auto counts = metrics.snapshot(config, 128, false);
    assert(counts["active_requests"] == 0);
    assert(counts["total_requests"] == 9 && counts["failed_requests"] == 6);
    assert(counts["total_prompt_tokens"] == 15 && counts["total_completion_tokens"] == 6);
    (void)scheduler.status();

    FakeEngine limited_engine;
    limited_engine.defer = true;
    MfqScheduler limited(limited_engine, {64, 4096, 1, 65536});
    RuntimeRequestMetrics rejected_metrics;
    {
        CompletionStream pending(limited, config, rejected_metrics, params, "pending");
        bool rejected = false;
        try { CompletionStream rejected_request(limited, config, rejected_metrics, params, "full"); }
        catch (const ApiError& error) { rejected = error.status == 429; }
        assert(rejected && rejected_metrics.active_requests() == 1);
    }
    counts = rejected_metrics.snapshot(config, 128, false);
    assert(counts["active_requests"] == 0);
    assert(counts["total_requests"] == 2 && counts["failed_requests"] == 2);
}

static void check_media_file_codec() {
#ifndef _WIN32
    using namespace mfq::transport;
    auto path = (std::filesystem::temp_directory_path() / "mfq-multimodal-XXXXXX.bin").string();
    const int descriptor = mkstemps(path.data(), 4);
    assert(descriptor >= 0);
    struct Cleanup {
        std::string path;
        ~Cleanup() { std::filesystem::remove(path); }
    } cleanup{path};
    std::array<uint8_t, 68> contents{};
    const std::string magic = "MFQMM01";
    std::copy(magic.begin(), magic.end(), contents.begin());
    assert(::write(descriptor, contents.data(), contents.size()) == ssize_t(contents.size()));
    assert(::close(descriptor) == 0);
    json payload{{"version", 1}, {"binary_file", {{"path", path}, {"token", std::string(64, '0')}, {"size", 68}}},
        {"pixel_values", {{"dtype", "float32"}, {"shape", {1}}, {"data_offset", 64}, {"data_length", 4}}}};
    assert(parse_mfq_vision(payload).pixel_values == std::vector<float>{0});
    for (int invalid = 0; invalid < 4; ++invalid) {
        auto bad = payload;
        if (invalid == 0) { bad["pixel_values"]["shape"] = {1000000000}; bad["pixel_values"]["data_length"] = 4000000000LL; }
        if (invalid == 1) bad["pixel_values"]["data_offset"] = 65;
        if (invalid == 2) bad["binary_file"]["token"] = std::string(64, '1');
        if (invalid == 3) bad["binary_file"]["size"] = 69;
        bool rejected = false;
        try { parse_mfq_vision(bad); }
        catch (const ApiError& error) { rejected = error.status == 400; }
        assert(rejected);
    }
#endif
}

static void check_media_codec() {
    using namespace mfq::transport;
    json payload{{"version", 1}, {"pixel_values", {{"dtype", "float32"}, {"shape", {1}}, {"data_base64", "AAAAAA=="}}}};
    for (const auto processor : {"minicpmo", "deepseek_v4", "deepseek_v41", "qwen3_vision"}) {
        payload["version"] = processor == std::string("minicpmo") ? 1 : processor == std::string("qwen3_vision") ? 3 : 2;
        payload["processor"] = processor;
        const auto media = mfq::transport::parse_mfq_vision(payload);
        assert(media.pixel_shape == std::vector<int64_t>{1} && media.pixel_values == std::vector<float>{0});
    }
    for (int invalid = 0; invalid < 5; ++invalid) {
        auto bad = payload;
        if (invalid == 0) bad["pixel_values"]["shape"] = {1000000000}; // Must reject bytes before allocating declared tensor size.
        if (invalid == 1) bad["pixel_values"]["dtype"] = "uint8";
        if (invalid == 2) bad["version"] = (uint64_t(1) << 32) | 1;
        if (invalid == 3) bad["processor"] = "invalid/processor";
        if (invalid == 4) bad["vision_types"] = {{"dtype", "int32"}, {"shape", {1, 1}}, {"data_base64", "AQAAAA=="}};
        bool rejected = false;
        try { mfq::transport::parse_mfq_vision(bad); }
        catch (const mfq::transport::ApiError& error) { rejected = error.status == 400; }
        assert(rejected);
    }
}

static void check_control_protocol() {
    using namespace mfq::transport;
    FakeEngine engine;
    engine.supports_duplex = true;
    MfqScheduler scheduler(engine);
    MfqRuntimeTransportConfig config;
    config.max_context = 128;
    RuntimeProtocol protocol(scheduler, config);
    const auto health = protocol.call("health");
    assert(health["status"] == "ok");
    assert(health["chat_template_capabilities"]["thinking"]["supported"] == true);
    assert(health["chat_template_capabilities"]["reasoning_effort"]["values"] == json::array({"low", "high"}));
    assert(protocol.call("status")["test_session_only"] == 99);
    assert(protocol.call("health")["test_health_only"] == 99);
    assert(protocol.call("session.fork", {{"source_session_id", "a"}, {"target_session_id", "b"}})["copied_snapshots"] == 7);
    assert(protocol.call("session.close", {{"session_id", "a"}})["released_snapshots"] == 7);
    assert(protocol.call("cache.clear")["released_snapshots"] == 7);
    assert(protocol.call("cache.trim", {{"target_bytes", uint64_t(-1)}})["target_bytes"] == uint64_t(-1));
    assert(protocol.call("reload", {{"context_size", 64}})["max_context"] == 64);
    assert(protocol.call("status")["max_context"] == 64);
    const auto invalid = [&](auto work, int status = 400) {
        bool rejected = false;
        try { work(); }
        catch (const ApiError& error) { rejected = error.status == status; }
        assert(rejected);
    };
    invalid([&] { protocol.call("cache.trim", {{"target_bytes", -1}}); });
    invalid([&] { protocol.call("session.fork", {{"source_session_id", "a"}, {"target_session_id", "a"}}); });
    invalid([&] { protocol.call("session.close", {{"session_id", "../unsafe"}}); });
    invalid([&] { protocol.call("reload", {{"context_size", uint64_t(-1)}}); });
    invalid([&] { protocol.call("reload", {{"context_size", 129}}); });
    protocol.open_realtime("channel");
    invalid([&] { protocol.open_realtime("peer"); }, 409);
    invalid([&] { protocol.realtime_message("channel", {{"type", "input.append"}, {"input", {{"text", "hello"}}}}); }, 409);
    invalid([&] { protocol.realtime_message("channel", {{"type", "session.init"}, {"payload", {{"config", {{"top_k", int64_t(1) << 32}}}}}}); });
    auto events = protocol.realtime_message("channel", {{"type", "session.init"}});
    assert(events.size() == 1 && events[0]["type"] == "session.created");
    const auto session = events[0]["session_id"].get<std::string>();
    assert(events[0]["metrics"]["backend"] == "fake");
    assert(engine.last_parameters.top_k == 100 && engine.last_parameters.length_penalty == 1.0);
    assert(engine.last_parameters.tts_temperature == 0.8 && engine.last_parameters.tts_repetition_penalty == 1.05);
    assert(protocol.call("health")["duplex_active"] == true);
    invalid([&] { protocol.realtime_message("channel", {{"type", "input.append"}, {"input", {{"text", "hello"}, {"force_listen", 1}}}}); });
    events = protocol.realtime_message("channel", {{"type", "input.append"}, {"input", {{"text", "hello"}, {"max_new_speak_tokens", 2}}}});
    assert(events.size() == 3);
    assert(events[0]["kind"] == "text" && events[0]["text"] == "hello");
    assert(events[1]["kind"] == "audio_tokens" && events[1]["force_flush"] == true);
    assert(events[2]["type"] == "response.step.done");
    for (const auto& event : events) {
        assert(event["session_id"] == session && event.contains("server_send_ts"));
        assert(event["response_id"] == events[0]["response_id"]);
    }
    assert(!protocol.close_realtime({}, "not-this-session"));
    assert(protocol.close_realtime({}, session) && !protocol.close_realtime("channel"));
    protocol.open_realtime("channel");
    protocol.realtime_message("channel", {{"type", "session.init"}});
    bool failed = false;
    try { protocol.realtime_message("channel", {{"type", "input.append"}, {"input", {{"text", "invalid"}}}}); }
    catch (const InferenceInputError&) { failed = true; }
    assert(failed && protocol.call("health")["duplex_active"] == false);
    protocol.open_realtime("after-error");
    assert(protocol.close_realtime("after-error"));
    config.runtime_profile.duplex.top_k = 17;
    config.runtime_profile.duplex.length_penalty = 1.25;
    config.runtime_profile.tts.temperature = 0.5;
    RuntimeProtocol profiled(scheduler, config);
    profiled.open_realtime("profiled");
    profiled.realtime_message("profiled", {{"type", "session.init"}, {"payload", {{"config", {{"tts_repetition_penalty", 1.2}, {"seed", uint64_t(-1)}}}}}});
    assert(engine.last_parameters.top_k == 17 && engine.last_parameters.length_penalty == 1.25);
    assert(engine.last_parameters.tts_temperature == 0.5 && engine.last_parameters.tts_repetition_penalty == 1.2);
    assert(engine.last_parameters.seed == uint64_t(-1));
    assert(profiled.close_realtime("profiled"));
    const auto stops = engine.stops.load();
    {
        RuntimeProtocol abandoned(scheduler, config);
        abandoned.open_realtime("abandoned");
        abandoned.realtime_message("abandoned", {{"type", "session.init"}});
    }
    assert(engine.stops == stops + 1);
}

static void check_input_ownership_and_budget() {
    FakeEngine engine;
    engine.defer = true;
    MfqScheduler scheduler(engine, {8, 4096, 2, 65536});
    auto input = request(0, 1000);
    input.input.cache_plan.session_id = "media-session";
    input.input.media.emplace().pixel_values.resize(10000, 0.5f);
    const auto* pixels = input.input.media->pixel_values.data();
    auto pending = scheduler.submit(std::move(input));
    auto too_large = request(1);
    too_large.input.media.emplace().audio_features.resize(10000);
    bool rejected = false;
    try { scheduler.submit(std::move(too_large)); }
    catch (const MfqSchedulerOverloaded&) { rejected = true; }
    assert(rejected);
    auto second = scheduler.submit(request(1, 1000));
    rejected = false;
    try { scheduler.submit(request(2)); }
    catch (const MfqSchedulerOverloaded&) { rejected = true; }
    assert(rejected);
    engine.defer = false;
    while (!engine.media.load()) std::this_thread::yield();
    assert(engine.media == pixels); // Deferred retries and acceptance never clone the media buffer.
    assert(scheduler.cancel_session("media-session")); // Logical identity survives moving the payload.
    drain(pending, true);
    scheduler.cancel_request("1");
    drain(second, true);
    (void)scheduler.status();
    drain(scheduler.submit(request(2, 1))); // Both count and byte reservations were returned.
    auto invalid = request(3); invalid.input.media.emplace().pixel_values.reserve(65536);
    rejected = false;
    try { scheduler.submit(std::move(invalid)); }
    catch (const MfqSchedulerOverloaded&) { rejected = true; }
    assert(rejected); // Account reserved capacity, even when the vector is empty.
}

static void check_duplex_preemption() {
    FakeEngine engine;
    MfqScheduler scheduler(engine);
    scheduler.control(MfqDuplexSessionParams{});
    const auto wait_phase = [&](int phase) {
        const auto deadline = Clock::now() + 2s;
        while (engine.control_steps < phase) {
            assert(Clock::now() < deadline);
            std::this_thread::sleep_for(1ms);
        }
    };
    for (int phase : {3, 20}) {
        MfqDuplexStepInput input; input.max_new_speak_tokens = 1000;
        engine.control_steps = 0;
        auto result = scheduler.control_async(input);
        wait_phase(phase);
        const auto start = Clock::now();
        (void)scheduler.control(RuntimeMetrics{});
        scheduler.control(StopDuplex{});
        assert(result.wait_for(100ms) == std::future_status::ready);
        bool cancelled = false;
        try { (void)result.get(); } catch (const std::runtime_error&) { cancelled = true; }
        assert(cancelled && Clock::now() - start < 100ms);
        scheduler.control(MfqDuplexSessionParams{});
    }
    MfqDuplexStepInput input; input.max_new_speak_tokens = 4;
    (void)scheduler.control(input);
    assert(engine.control_terminals == 6); // Three starts, two cancels, one completed chunk.
    input.max_new_speak_tokens = 1000;
    engine.control_steps = 0;
    auto reloaded = scheduler.control_async(input);
    wait_phase(3);
    scheduler.reload(64);
    assert(reloaded.wait_for(100ms) == std::future_status::ready);
    bool cancelled = false;
    try { reloaded.get(); } catch (...) { cancelled = true; }
    assert(cancelled);
    scheduler.control(MfqDuplexSessionParams{});
    engine.control_steps = 0;
    auto stopped = scheduler.control_async(input);
    wait_phase(20);
    const auto start = Clock::now();
    scheduler.shutdown();
    assert(Clock::now() - start < 100ms && stopped.wait_for(0ms) == std::future_status::ready);
    cancelled = false;
    try { stopped.get(); } catch (...) { cancelled = true; }
    assert(cancelled && engine.control_terminals == 9);
}

void test_live_request_metrics() {
    using namespace mfq::transport;
    RuntimeRequestMetrics store;
    {
        ActiveRequest first(store), second(store);
        first.progress("one", {{"generated_tokens", 6}});
        second.progress("two", {{"processed_tokens", 2048}});
        auto value = store.snapshot({}, 512, false);
        assert(value["live_requests"].size() == 2);
        assert(value["live_requests"]["one"]["generated_tokens"] == 6);
        first.complete("one", true, true, 10, {}, {});
        value = store.snapshot({}, 512, false);
        assert(value["live_requests"].size() == 1);
        assert(value["live_requests"].contains("two"));
    }
    assert(store.snapshot({}, 512, false)["live_requests"].empty());
    FakeEngine engine;
    MfqScheduler scheduler(engine);
    MfqRuntimeTransportConfig config;
    CompletionStream completion(scheduler, config, store,
        {{"input",{{"messages",json::array({{{"role","user"},{"content","hello"}}})}}},{"sampling",{{"max_new_tokens",2}}}}, "0");
    while (completion.next()) {}
    assert(completion.progress()["generated_tokens"] == 6);
}

int main(int argc, char** argv) {
    if (argc >= 2) {
        const bool stdio = std::string(argv[1]) == "--stdio";
        if (stdio) prepare_mfq_stdio_transport();
        FakeEngine engine;
        engine.supports_duplex = true;
        MfqScheduler scheduler(engine);
        MfqHttpRuntimeTransportConfig config;
        config.model_type = "minicpmo45";
        config.max_context = 128;
        config.vocab_size = 64;
        if (stdio) return make_mfq_stdio_transport(config)->run(scheduler);
        assert(argc == 3 && std::string(argv[1]) == "--http");
        config.port = std::stoi(argv[2]);
        return make_mfq_http_transport(config)->run(scheduler);
    }
    check_duplex_preemption();
    check_media_file_codec();
    check_media_codec();
    check_control_protocol();
    check_input_ownership_and_budget();
    test_model_capabilities();
    test_live_request_metrics();
    test_decode_timing();
    test_benchmark_input_contract();
    test_probability_contract();
    test_chat_input_contract();
    check_generation_protocol();
    {
        FakeEngine engine;
        MfqScheduler scheduler(engine);
        auto input = request(0, 1); input.diagnostic_input->tokens = {59};
        auto handle = scheduler.submit(input);
        while (!engine.blocked) std::this_thread::yield();
        const auto started = Clock::now();
        for (int index = 0; index < 1000; ++index) {
            const auto snapshot = scheduler.snapshot();
            assert(snapshot.status.healthy && snapshot.runtime == Metrics({{"test_health_only", 99}}));
        }
        assert(Clock::now() - started < 100ms);
        assert(engine.advances[0] == 0);
        engine.resume = true;
        drain(handle);
        scheduler.control(RuntimeMetrics{});
        assert(scheduler.snapshot().runtime == Metrics({{"test_health_only", 99}}));
    }
    {
        FakeEngine engine;
        MfqScheduler scheduler(engine, {8, 4096});
        auto slow = scheduler.submit(request(0));
        auto fast = scheduler.submit(request(1));
        drain(fast);
        const auto paused = engine.advances[0].load();
        assert(paused > 0 && paused < 12);
        auto another = scheduler.submit(request(2, 3));
        drain(another);
        assert(engine.advances[0] == paused);
        drain(slow);
        auto blocked = scheduler.submit(request(3, 100));
        while (engine.advances[3] == 0) std::this_thread::yield();
        assert(scheduler.cancel_request("3"));
        assert(!scheduler.cancel_request("unknown"));
        drain(blocked, true);
        auto session = request(4, 100); session.input.cache_plan.session_id = "session";
        auto handle = scheduler.submit(session);
        assert(scheduler.session({SessionCommand::Kind::trim}).count == 7);
        assert(scheduler.session({SessionCommand::Kind::budget}).count == 7);
        for (const auto kind : {SessionCommand::Kind::clear, SessionCommand::Kind::memory_budget,
                SessionCommand::Kind::refresh}) {
            bool rejected = false;
            try { scheduler.session({kind}); } catch (const std::runtime_error&) { rejected = true; }
            assert(rejected);
        }
        session.id = "5";
        bool conflict = false;
        try { scheduler.submit(session); } catch (const std::invalid_argument&) { conflict = true; }
        assert(conflict);
        conflict = false;
        try { scheduler.session({SessionCommand::Kind::close, "session"}); } catch (const std::runtime_error&) { conflict = true; }
        assert(conflict);
        assert(scheduler.cancel_session("session"));
        drain(handle, true);
        assert(scheduler.session({SessionCommand::Kind::close, "session"}).count == 7);
        auto deadline = request(6, 100); deadline.deadline = Clock::now() + 15ms;
        auto expiring = scheduler.submit(deadline);
        std::this_thread::sleep_for(30ms);
        drain(expiring, true);
        auto active = scheduler.submit(request(7, 100));
        auto queued = scheduler.submit(request(8, 100));
        auto waiting = scheduler.submit(request(9, 100));
        assert(scheduler.cancel_request("9"));
        drain(waiting, true);
        assert(scheduler.reload(64) == 64 && engine.reloads == 1);
        drain(active, true); drain(queued, true);
        auto shutdown = scheduler.submit(request(10, 100));
        scheduler.shutdown();
        drain(shutdown, true);
    }
    {
        FakeEngine engine;
        MfqScheduler scheduler(engine);
        auto first = scheduler.submit(request(0, 100));
        auto bad = request(1); bad.diagnostic_input = DiagnosticTokenInput{{63}};
        auto second = scheduler.submit(bad);
        drain(first, false, true); drain(second, false, true);
        assert(!scheduler.status().healthy);
        assert(engine.active.empty());
    }
    {
        FakeEngine engine;
        MfqScheduler scheduler(engine, {64, 4096}); // bytes, not event count, limits progress
        auto large = request(0, 100); large.diagnostic_input = DiagnosticTokenInput{{61}};
        auto slow = scheduler.submit(large);
        drain(scheduler.submit(request(1, 4)));
        const auto paused = engine.advances[0].load();
        assert(paused >= 2 && paused <= 3);
        drain(scheduler.submit(request(2, 4)));
        assert(engine.advances[0] == paused);
        scheduler.cancel_request("0");
        drain(slow, true); // reserved terminal space remains available
        auto oversized = request(3); oversized.diagnostic_input = DiagnosticTokenInput{{62}};
        auto bad = scheduler.submit(oversized);
        drain(scheduler.submit(request(4, 4)));
        drain(bad, false, true);
        assert(engine.released == 5);
        bool failed = false;
        try { scheduler.reload(13); } catch (const std::runtime_error&) { failed = true; }
        assert(failed);
        failed = false;
        try { scheduler.submit(request(5)); } catch (const std::runtime_error&) { failed = true; }
        assert(failed);
        assert(scheduler.reload(64) == 64);
        assert(scheduler.status().healthy);
        drain(scheduler.submit(request(6, 4)));
    }
    {
        FakeEngine engine;
        MfqScheduler scheduler(engine, {64, 8192});
        auto variable = request(0, 4); variable.diagnostic_input = DiagnosticTokenInput{{60}};
        auto slow = scheduler.submit(variable);
        drain(scheduler.submit(request(1, 4)));
        // Both deltas fit individually. Their sum pauses the producer instead
        // of reporting output_limit or losing the second delta.
        int outputs = 0, terminals = 0;
        while (!slow->done()) for (const auto& event : slow->wait()) {
            outputs += std::holds_alternative<OutputDelta>(event.data);
            if (terminal(event.data)) {
                assert(std::holds_alternative<Completed>(event.data));
                ++terminals;
            }
        }
        assert(outputs == 4 && terminals == 1);
    }
    {
        FakeEngine engine;
        MfqScheduler scheduler(engine, {64, 8192});
        auto variable = request(0, 2); variable.diagnostic_input = DiagnosticTokenInput{{60}};
        auto slow = scheduler.submit(variable);
        // The final delta and terminal arrive together, with the delta blocked.
        while (engine.released == 0) std::this_thread::yield();
        scheduler.shutdown();
        drain(slow); // shutdown must not wait for the consumer to drain
    }
}
