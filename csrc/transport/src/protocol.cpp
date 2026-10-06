#include "common.h"

#include <limits>
#include <random>

namespace mfq::transport {
using namespace mfq::engine;

namespace {
int32_t int32_field(const json& body, const char* name, int32_t fallback) {
    const auto value = integer_field(body, name, fallback);
    if (value < std::numeric_limits<int32_t>::min() || value > std::numeric_limits<int32_t>::max())
        throw ApiError(400, "invalid_request_error", std::string(name) + " must fit a signed 32-bit integer", name);
    return static_cast<int32_t>(value);
}
} // namespace

RuntimeProtocol::RuntimeProtocol(const MfqScheduler& scheduler, MfqRuntimeTransportConfig config)
    : scheduler_(scheduler), config_(std::move(config)), context_(config_.max_context) {}

void RuntimeProtocol::add_session_metrics(json& value) const {
    for (const auto& [name, number] : scheduler_.session({SessionCommand::Kind::metrics}).metrics)
        value[name] = number;
}

json RuntimeProtocol::snapshot(bool health) const {
    const auto defaults = default_sampling_params(config_);
    const auto& capabilities = config_.model_capabilities;
    json value = health ? json{{"status", reloading_ ? "loading" :
        (scheduler_.status().healthy ? "ok" : "unhealthy")},
        {"model", config_.model_name}, {"model_type", config_.model_type},
        {"max_context", context_.load()}} : metrics_.snapshot(config_, context_, reloading_);
    value["sampling_defaults"] = sampling_params_json(defaults);
    value["duplex_sampling_defaults"] = duplex_profile_json(config_.runtime_profile.duplex);
    value["tts_sampling_defaults"] = tts_profile_json(config_.runtime_profile.tts);
    value["runtime_profile_source"] = config_.runtime_profile.source;
    value["chat_template_capabilities"] = chat_template_capabilities_json(scheduler_.chat_template_capabilities());
    value["model_capabilities"] = model_capability_profile_json(capabilities);
    value["vision_supported"] = capabilities.image_input || capabilities.video_input;
    value["vision_available"] = scheduler_.supports_multimodal_generation() && capabilities.image_input;
    value["video_available"] = scheduler_.supports_multimodal_generation() && capabilities.video_input;
    value["mtp_supported"] = capabilities.mtp;
    value["mtp_available"] = false;
    value["vision_enabled_default"] = defaults.enable_vision;
    value["mtp_enabled_default"] = defaults.enable_mtp;
    value["duplex_available"] = scheduler_.info().duplex;
    {
        std::lock_guard lock(realtime_gate_);
        value["duplex_active"] = !session_.empty();
    }
    const auto runtime_metrics = std::get<Metrics>(scheduler_.control(RuntimeMetrics{}));
    for (const auto& [name, number] : runtime_metrics) value[name] = number;
    add_session_metrics(value);
    return value;
}

json RuntimeProtocol::call(const std::string& operation, const json& params) {
    if (!params.is_object()) throw ApiError(400, "invalid_request_error", "parameters must be an object");
    if (operation == "health" || operation == "status") return snapshot(operation == "health");
    if (operation == "models") return {{"models", json::array({{
        {"name", config_.model_name}, {"type", config_.model_type},
        {"capabilities", model_capability_profile_json(config_.model_capabilities)}}})}};
    if (operation == "realtime.capabilities") return {{"available", scheduler_.info().duplex},
        {"modes", scheduler_.info().duplex ? json::array({"audio"}) : json::array()}};
    if (operation == "request.cancel") return {{"status", "ok"},
        {"cancelled", scheduler_.cancel_request(params.value("target_id", std::string{}))}};
    const auto session_id = [&](const char* field) {
        if (!params.contains(field) || !params[field].is_string() ||
                !valid_mfq_session_id(params[field].get<std::string>()))
            throw ApiError(400, "invalid_request_error",
                std::string(field) + " must contain 1 to 128 safe identifier bytes", field);
        return params[field].get<std::string>();
    };
    if (operation == "session.cancel") return {{"status", "ok"},
        {"cancelled", scheduler_.cancel_session(session_id("session_id"))}};
    if (operation == "session.close") return {{"status", "ok"},
        {"released_snapshots", scheduler_.session({SessionCommand::Kind::close, session_id("session_id")}).count}};
    if (operation == "session.fork") {
        auto source = session_id("source_session_id"), target = session_id("target_session_id");
        if (source == target) throw ApiError(400, "invalid_request_error",
            "source and target sessions must differ", "target_session_id");
        return {{"status", "ok"}, {"copied_snapshots",
            scheduler_.session({SessionCommand::Kind::fork, source, target}).count}};
    }
    if (operation == "cache.clear" || operation == "cache.trim") {
        uint64_t target = 0;
        if (params.contains("target_bytes")) {
            const auto& number = params["target_bytes"];
            if (!number.is_number_unsigned() &&
                    !(number.is_number_integer() && number.get<int64_t>() >= 0))
                throw ApiError(400, "invalid_request_error", "target_bytes must be a non-negative integer", "target_bytes");
            target = number.get<uint64_t>();
        }
        const bool clear = operation == "cache.clear";
        json result{{"status", "ok"}, {clear ? "released_snapshots" : "released_bytes",
            scheduler_.session({clear ? SessionCommand::Kind::clear : SessionCommand::Kind::trim,
                {}, {}, target}).count}};
        if (!clear) result["target_bytes"] = target;
        add_session_metrics(result);
        return result;
    }
    if (operation == "reload") {
        if (!scheduler_.supports_reload()) throw ApiError(501, "unsupported_operation", "this runtime does not support model reload");
        const auto requested = integer_field(params, "context_size", context_);
        const auto capacity = config_.context_capacity > 0 ? config_.context_capacity : config_.max_context;
        if (requested < 1 || (capacity > 0 && requested > capacity))
            throw ApiError(400, "invalid_request_error", "context_size exceeds model capacity", "context_size");
        bool idle = false;
        if (!reloading_.compare_exchange_strong(idle, true))
            throw ApiError(409, "conflict", "model reload is already pending");
        try {
            const auto loaded = scheduler_.reload(requested);
            if (loaded < 1 || (capacity > 0 && loaded > capacity))
                throw std::runtime_error("runtime reload returned an invalid context size");
            context_ = loaded;
            reloading_ = false;
            return {{"status", "ok"}, {"model", config_.model_name},
                {"max_context", loaded}, {"context_capacity", capacity}};
        } catch (...) { reloading_ = false; throw; }
    }
    throw ApiError(404, "unsupported_operation", "unsupported runtime operation: " + operation);
}

void RuntimeProtocol::open_realtime(std::string channel, const std::string& mode) {
    if (!scheduler_.info().duplex) throw ApiError(501, "unsupported_operation", "this runtime has no realtime backend");
    if (mode != "audio") throw ApiError(400, "unsupported_operation", "realtime supports audio mode only");
    if (channel.empty()) throw ApiError(400, "invalid_request_error", "realtime channel requires an identity");
    std::lock_guard lock(realtime_gate_);
    if (!channel_.empty()) throw ApiError(409, "conflict", "a realtime channel is already open");
    channel_ = std::move(channel);
}

bool RuntimeProtocol::close_realtime(const std::string& channel, const std::string& session) {
    std::future<ControlResult> stop;
    {
        std::lock_guard lock(realtime_gate_);
        if (channel_.empty() || (!channel.empty() && channel_ != channel) ||
                (!session.empty() && session_ != session)) return false;
        // Queue stop before another channel may start. Preparation cannot enqueue
        // execution after this identity has been invalidated.
        if (backend_started_) stop = scheduler_.control_async(StopDuplex{});
        channel_.clear(); session_.clear(); control_tokens_.clear();
        pending_ = backend_started_ = false;
    }
    if (stop.valid()) stop.get();
    return true;
}

json RuntimeProtocol::realtime_closed(const std::string& channel, const std::string& reason,
                                      const std::string& message) const {
    std::lock_guard lock(realtime_gate_);
    return {{"type", "session.closed"}, {"session_id", channel_ == channel ? session_ : ""},
        {"reason", reason}, {"diagnostic", {{"message", message}}}};
}

std::vector<json> RuntimeProtocol::realtime_message(const std::string& channel, const json& body) {
    if (!body.is_object() || !body.contains("type") || !body["type"].is_string())
        throw ApiError(400, "invalid_request_error", "realtime message requires a string type");
    const auto type = body["type"].get<std::string>();
    const bool initialize = type == "session.init";
    ControlRequest request;
    if (initialize) {
        const auto payload = body.value("payload", json::object());
        if (!payload.is_object()) throw ApiError(400, "invalid_request_error", "session.init payload must be an object");
        if (payload.value("mode", std::string("full_duplex")) != "full_duplex")
            throw ApiError(400, "unsupported_operation", "realtime backend supports full_duplex only");
        MfqDuplexSessionParams parameters;
        const auto generation = payload.value("config", json::object());
        if (!generation.is_object()) throw ApiError(400, "invalid_request_error", "session config must be an object", "config");
        const auto& defaults = config_.runtime_profile.duplex;
        parameters.greedy = generation.value("decode_mode", defaults.decode_mode.value_or("sampling")) == "greedy";
        parameters.temperature = number_field(generation, "temperature", defaults.temperature.value_or(0.7));
        parameters.top_k = int32_field(generation, "top_k", defaults.top_k.value_or(100));
        parameters.top_p = number_field(generation, "top_p", defaults.top_p.value_or(0.8));
        parameters.listen_probability_scale = number_field(generation, "listen_prob_scale", defaults.listen_prob_scale.value_or(1.0));
        parameters.repetition_penalty = number_field(generation, "text_repetition_penalty", defaults.text_repetition_penalty.value_or(1.05));
        parameters.repetition_window = int32_field(generation, "text_repetition_window_size", defaults.text_repetition_window_size.value_or(512));
        parameters.length_penalty = number_field(generation, "length_penalty", defaults.length_penalty.value_or(1.0));
        parameters.tts_temperature = number_field(generation, "tts_temperature", config_.runtime_profile.tts.temperature.value_or(0.8));
        parameters.tts_repetition_penalty = number_field(generation, "tts_repetition_penalty", config_.runtime_profile.tts.repetition_penalty.value_or(1.05));
        if (generation.contains("seed")) {
            const auto& seed = generation["seed"];
            if (!seed.is_number_unsigned() && !(seed.is_number_integer() && seed.get<int64_t>() >= 0))
                throw ApiError(400, "invalid_request_error", "seed must be a non-negative integer", "seed");
            parameters.seed = seed.get<uint64_t>();
        } else { std::random_device random; parameters.seed = (uint64_t(random()) << 32) ^ random(); }
        if (payload.contains("reference_audio_features")) {
            if (!payload["reference_audio_features"].is_string())
                throw ApiError(400, "invalid_request_error", "reference_audio_features must be base64 float32 Mel data", "reference_audio_features");
            parameters.reference_audio_frames = int32_field(payload, "reference_audio_frames", 0);
            parameters.reference_audio_features = decode_audio_features(payload["reference_audio_features"].get<std::string>(), parameters.reference_audio_frames);
        }
        request = PrepareDuplex{payload.value("system_prompt",
            defaults.system_prompt.value_or("Streaming Omni Conversation.")), std::move(parameters)};
    } else {
        if (type != "input.append") throw ApiError(400, "invalid_request_error", "unsupported realtime message type: " + type);
        const auto input = body.value("input", json::object());
        if (!input.is_object()) throw ApiError(400, "invalid_request_error", "input must be an object", "input");
        if (!input.contains("text") && !input.contains("audio_features"))
            throw ApiError(400, "invalid_request_error", "input requires audio_features or text", "input");
        MfqDuplexStepInput step;
        std::string text;
        if (input.contains("audio_features")) {
            if (!input["audio_features"].is_string()) throw ApiError(400, "invalid_request_error", "audio_features must be base64 float32 Mel data", "audio_features");
            step.audio_frames = int32_field(input, "audio_frames", 0);
            step.audio_features = decode_audio_features(input["audio_features"].get<std::string>(), step.audio_frames);
            step.audio_prefix_extra_frames = integer_field(input, "audio_prefix_extra_frames", 0);
            step.audio_suffix_extra_frames = integer_field(input, "audio_suffix_extra_frames", 0);
        }
        if (input.contains("text")) {
            if (!input["text"].is_string() || input["text"].get_ref<const std::string&>().empty())
                throw ApiError(400, "invalid_request_error", "input.text must be a non-empty string", "text");
            text = input["text"].get<std::string>();
        }
        step.max_new_speak_tokens = int32_field(input, "max_new_speak_tokens", config_.runtime_profile.duplex.max_new_speak_tokens_per_chunk.value_or(20));
        for (const auto* flag : {"force_listen", "force_speak"})
            if (input.contains(flag) && !input[flag].is_boolean())
                throw ApiError(400, "invalid_request_error", std::string(flag) + " must be boolean", flag);
        step.force_listen = input.value("force_listen", false);
        step.force_speak = input.value("force_speak", false);
        if (step.force_listen && step.force_speak) throw ApiError(400, "invalid_request_error", "force_listen and force_speak are mutually exclusive");
        request = text.empty() ? ControlRequest{std::move(step)} :
            ControlRequest{PrepareDuplexStep{std::move(text), std::move(step)}};
    }
    std::string session;
    std::future<ControlResult> result;
    const bool prepare = std::holds_alternative<PrepareDuplex>(request) || std::holds_alternative<PrepareDuplexStep>(request);
    {
        std::lock_guard lock(realtime_gate_);
        if (channel.empty() || channel_ != channel) throw ApiError(404, "not_found", "realtime channel was not found");
        if (pending_) throw ApiError(409, "conflict", "realtime operation is already pending");
        if (initialize && !session_.empty()) throw ApiError(409, "conflict", "realtime session is already initialized");
        if (!initialize && session_.empty()) throw ApiError(409, "conflict", "session.init must precede input.append");
        if (initialize) session_ = request_id("sess-");
        session = session_;
        result = scheduler_.control_async(std::move(request));
        pending_ = backend_started_ = true;
    }
    try {
        auto value = result.get();
        if (prepare) {
            std::future<ControlResult> execution;
            {
                std::lock_guard lock(realtime_gate_);
                if (channel_ != channel || session_ != session) throw ApiError(409, "conflict", "realtime session closed");
                if (auto* parameters = std::get_if<MfqDuplexSessionParams>(&value)) {
                    control_tokens_ = {parameters->special_ids.begin(), parameters->special_ids.end()};
                    execution = scheduler_.control_async(std::move(*parameters));
                } else execution = scheduler_.control_async(std::move(std::get<MfqDuplexStepInput>(value)));
            }
            value = execution.get();
        }
        std::unordered_set<int64_t> controls;
        {
            std::lock_guard lock(realtime_gate_);
            if (channel_ != channel || session_ != session) throw ApiError(409, "conflict", "realtime session closed");
            controls = control_tokens_;
        }
        std::vector<json> events;
        if (initialize) events.push_back({{"type", "session.created"}, {"session_id", session},
            {"mode", "full_duplex"}, {"metrics", {{"backend", scheduler_.info().backend}}}});
        else {
            const auto output = std::get<MfqDuplexStepResult>(std::move(value));
            const auto response = request_id("resp-");
            const json metrics{{"backend", scheduler_.info().backend}, {"wall_clock_ms", output.inference_ms},
                {"kv_cache_length", output.language_cache_position}, {"audio_cache_length", output.audio_cache_position},
                {"tts_cache_length", output.tts_cache_position}, {"audio_chunk_index", output.audio_chunk_index}};
            const auto event = [&](const char* type) { return json{{"type", type}, {"session_id", session},
                {"response_id", response}, {"end_of_turn", output.end_of_turn}, {"metrics", metrics}}; };
            const auto text = scheduler_.decode_tokens(output.generated_tokens, controls);
            if (!text.empty()) { auto delta = event("response.output.delta"); delta["kind"] = "text"; delta["text"] = text; events.push_back(std::move(delta)); }
            if (!output.audio_tokens.empty() || (output.end_of_turn && !output.is_listen)) {
                auto delta = event("response.output.delta"); delta["kind"] = "audio_tokens";
                delta["audio_tokens"] = output.audio_tokens; delta["force_flush"] = output.tts_force_flush;
                events.push_back(std::move(delta));
            }
            if (output.is_listen) { auto delta = event("response.output.delta"); delta["kind"] = "listen"; events.push_back(std::move(delta)); }
            events.push_back(event("response.step.done"));
        }
        for (auto& event : events) event["server_send_ts"] =
            std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
        {
            std::lock_guard lock(realtime_gate_);
            if (channel_ != channel || session_ != session) throw ApiError(409, "conflict", "realtime session closed");
            pending_ = false;
        }
        return events;
    } catch (...) {
        close_realtime(channel, session);
        throw;
    }
}

} // namespace mfq::transport
