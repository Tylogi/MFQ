#include "common.h"

#include "httplib.h"

#include <atomic>
#include <chrono>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

namespace {

using namespace mfq::transport_detail;

static json error_body(const std::string & message, const std::string & type, const std::string & param = {}) {
    return {
        {"error", {
            {"message", message},
            {"type", type},
            {"param", param.empty() ? json(nullptr) : json(param)},
            {"code", json(nullptr)},
        }},
    };
}

static void set_json(httplib::Response & res, const json & body, int status = 200) {
    res.status = status;
    res.set_content(body.dump(), "application/json; charset=utf-8");
}

static bool write_sse(httplib::DataSink & sink, const json & value) {
    const std::string event = "data: " + value.dump() + "\n\n";
    return sink.write(event.data(), event.size());
}

static bool authorized(const httplib::Request & req, httplib::Response & res, const std::string & api_key) {
    if (api_key.empty()) return true;
    const std::string expected = "Bearer " + api_key;
    if (req.get_header_value("Authorization") == expected) return true;
    res.set_header("WWW-Authenticate", "Bearer");
    set_json(res, error_body("invalid API key", "authentication_error"), 401);
    return false;
}

static json parse_body(const httplib::Request & req) {
    try {
        return json::parse(req.body);
    } catch (const json::parse_error & error) {
        throw ApiError(400, "invalid_request_error", std::string("invalid JSON: ") + error.what());
    }
}

static void handle_api_error(httplib::Response & res, const ApiError & error) {
    set_json(res, error_body(error.what(), error.type, error.param), error.status);
}

int run_mfq_http_transport(
        const MfqHttpRuntimeTransportConfig & config,
        const MfqScheduler & scheduler) {
    const auto & duplex = scheduler.duplex();
    const auto & session_control = scheduler.session_control();
    const auto & runtime_metrics = scheduler.runtime_metrics();
    if (!scheduler.supports_generation()) {
        throw std::runtime_error(
            "MFQ runtime transport requires a generation engine");
    }
    if (config.tokenizer_gguf.empty() &&
        config.tokenizer_model.empty()) {
        throw std::runtime_error(
            "MFQ runtime transport requires an embedded or external tokenizer GGUF");
    }
    if (!config.tokenizer_gguf.empty() &&
        !config.tokenizer_model.empty()) {
        throw std::runtime_error(
            "MFQ runtime transport tokenizer source is ambiguous");
    }
    if (config.port < 1 || config.port > 65535) {
        throw std::runtime_error("runtime transport port must be in [1, 65535]");
    }

    std::unique_ptr<MfqTokenizer> tokenizer =
        config.tokenizer_gguf.empty()
        ? std::make_unique<MfqTokenizer>(
              config.tokenizer_model)
        : std::make_unique<MfqTokenizer>(
              config.tokenizer_gguf);
    if (config.vocab_size > 0 && tokenizer->vocab_size() != config.vocab_size) {
        throw std::runtime_error("tokenizer/model vocabulary mismatch: tokenizer=" +
                                 std::to_string(tokenizer->vocab_size()) + " model=" +
                                 std::to_string(config.vocab_size));
    }
    common_chat_templates_ptr chat_templates = nullptr;
    if (!tokenizer->chat_template().empty()) {
        chat_templates = common_chat_templates_init(
            tokenizer->context(), "");
        if (!chat_templates) {
            throw std::runtime_error(
                "cannot initialize tokenizer.chat_template");
        }
    }
    const MfqSamplingParams sampling_defaults =
        default_sampling_params(config);
    const json duplex_sampling_defaults =
        duplex_profile_json(config.runtime_profile.duplex);
    const json tts_sampling_defaults =
        tts_profile_json(config.runtime_profile.tts);
    const std::string duplex_backend_name =
        duplex.name.empty() ? "native" : duplex.name;
    const json chat_template_capabilities =
        chat_template_capabilities_json(
            tokenizer->chat_template());
    const auto model_capability_profile = config.model_capabilities
        ? *config.model_capabilities
        : architecture_capability_profile(config.model_type);
    const json model_capabilities =
        model_capability_profile_json(model_capability_profile);
    const bool vision_supported =
        model_capability_profile.image_input ||
        model_capability_profile.video_input;
    const bool vision_available =
        scheduler.supports_multimodal_generation() &&
        model_capability_profile.image_input;
    const bool video_available =
        scheduler.supports_multimodal_generation() &&
        model_capability_profile.video_input;

    httplib::Server server;
    RuntimeRequestMetrics request_metrics_store;
    std::atomic<int64_t> active_context{config.max_context};
    std::atomic<bool> reloading{false};
    std::mutex reload_gate;
    std::mutex duplex_gate;
    std::string duplex_session_id;
    httplib::ws::WebSocket * duplex_socket = nullptr;
    bool duplex_backend_started = false;

    const auto duplex_is_active = [&]() {
        std::lock_guard<std::mutex> lock(duplex_gate);
        return !duplex_session_id.empty();
    };
    const auto stop_duplex_session = [&](const std::string & session_id,
                                         bool close_socket) {
        httplib::ws::WebSocket * socket = nullptr;
        bool stop_backend = false;
        {
            std::lock_guard<std::mutex> lock(duplex_gate);
            if (duplex_session_id.empty() ||
                duplex_session_id != session_id) {
                return false;
            }
            socket = duplex_socket;
            stop_backend = duplex_backend_started;
            duplex_session_id.clear();
            duplex_socket = nullptr;
            duplex_backend_started = false;
        }
        if (stop_backend) duplex.stop();
        if (close_socket && socket != nullptr && socket->is_open()) {
            socket->close(
                httplib::ws::CloseStatus::Normal, "session closed");
        }
        return true;
    };
    server.set_payload_max_length(
        (scheduler.supports_multimodal_generation() ? 512ULL : 16ULL) *
        1024ULL * 1024ULL);
    server.set_read_timeout(300, 0);
    server.set_write_timeout(300, 0);
    server.set_keep_alive_max_count(100);
    server.set_default_headers({
        {"Access-Control-Allow-Origin", "*"},
        {"Access-Control-Allow-Headers", "Authorization, Content-Type"},
        {"Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS"},
        {"X-Content-Type-Options", "nosniff"},
    });

    server.Options(R"(.*)", [](const httplib::Request &, httplib::Response & res) {
        res.status = 204;
    });

    if (duplex) {
        server.WebSocket("/runtime/realtime", [&](const httplib::Request & req,
                                          httplib::ws::WebSocket & ws) {
            const std::string expected = "Bearer " + config.api_key;
            if (!config.api_key.empty() &&
                req.get_header_value("Authorization") != expected) {
                ws.close(
                    httplib::ws::CloseStatus::PolicyViolation,
                    "authentication failed");
                return;
            }

            std::string owned_session;
            std::unordered_set<int64_t> session_controls;
            const auto send_event = [&](json event) {
                event["server_send_ts"] =
                    std::chrono::duration<double>(
                        std::chrono::system_clock::now()
                            .time_since_epoch()).count();
                return ws.send(event.dump());
            };
            try {
                std::string message;
                while (ws.is_open()) {
                    const auto read_result = ws.read(message);
                    if (read_result == httplib::ws::ReadResult::Fail) break;
                    if (read_result != httplib::ws::ReadResult::Text) {
                        throw ApiError(
                            400, "invalid_request_error",
                            "duplex backend accepts JSON text frames only");
                    }
                    json body;
                    try {
                        body = json::parse(message);
                    } catch (const json::parse_error & error) {
                        throw ApiError(
                            400, "invalid_request_error",
                            std::string("invalid duplex JSON: ") +
                                error.what());
                    }
                    if (!body.is_object() || !body.contains("type") ||
                        !body["type"].is_string()) {
                        throw ApiError(
                            400, "invalid_request_error",
                            "duplex message requires a string type");
                    }
                    const std::string type = body["type"].get<std::string>();

                    if (type == "session.init") {
                        if (!owned_session.empty()) {
                            throw ApiError(
                                409, "conflict",
                                "duplex session is already initialized");
                        }
                        const json payload = body.value(
                            "payload", json::object());
                        if (!payload.is_object()) {
                            throw ApiError(
                                400, "invalid_request_error",
                                "session.init payload must be an object");
                        }
                        const std::string mode = payload.value(
                            "mode", std::string("full_duplex"));
                        if (mode != "full_duplex") {
                            throw ApiError(
                                400, "unsupported_operation",
                                duplex_backend_name +
                                    " duplex backend supports full_duplex only");
                        }

                        owned_session = request_id("sess-");
                        {
                            std::lock_guard<std::mutex> lock(duplex_gate);
                            if (!duplex_session_id.empty()) {
                                owned_session.clear();
                                throw ApiError(
                                    409, "conflict",
                                    "the " + duplex_backend_name +
                                        " worker already owns a duplex session");
                            }
                            duplex_session_id = owned_session;
                            duplex_socket = &ws;
                        }

                        MfqDuplexSessionParams parameters;
                        const std::string system_prompt = payload.value(
                            "system_prompt",
                            config.runtime_profile.duplex.system_prompt.value_or(
                                "Streaming Omni Conversation."));
                        const std::string rendered_prefix =
                            "<|im_start|>system\n" + system_prompt +
                            "\n<|audio_start|>";
                        parameters.system_prefix = tokenizer->tokenize(
                            rendered_prefix, true, false);
                        parameters.system_suffix = tokenizer->tokenize(
                            "<|audio_end|><|im_end|>", true, false);
                        if (payload.contains("reference_audio_features")) {
                            if (!payload["reference_audio_features"].is_string()) {
                                throw ApiError(
                                    400, "invalid_request_error",
                                    "reference_audio_features must be base64 float32 Mel data",
                                    "reference_audio_features");
                            }
                            parameters.reference_audio_frames =
                                static_cast<int32_t>(integer_field(
                                    payload, "reference_audio_frames", 0));
                            parameters.reference_audio_features =
                                decode_audio_features(
                                    payload["reference_audio_features"].get<std::string>(),
                                    parameters.reference_audio_frames);
                        }
                        parameters.special_ids = {
                            tokenizer->special_token_id("<unit>"),
                            tokenizer->special_token_id("</unit>"),
                            tokenizer->special_token_id("<image>"),
                            tokenizer->special_token_id("</image>"),
                            tokenizer->special_token_id("<slice>"),
                            tokenizer->special_token_id("</slice>"),
                            tokenizer->special_token_id("<|listen|>"),
                            tokenizer->special_token_id("<|speak|>"),
                            tokenizer->special_token_id("<|tts_bos|>"),
                            tokenizer->special_token_id("<|tts_eos|>"),
                            tokenizer->special_token_id("<|chunk_eos|>"),
                            tokenizer->special_token_id("<|chunk_tts_eos|>"),
                            tokenizer->special_token_id("<|turn_eos|>"),
                            tokenizer->special_token_id("<|tts_pad|>"),
                            151687,
                        };
                        parameters.forbidden_ids = {
                            tokenizer->special_token_id("<|tts_pad|>"),
                        };
                        session_controls = std::unordered_set<int64_t>(
                            parameters.special_ids.begin(),
                            parameters.special_ids.end());
                        const json generation = payload.value(
                            "config", json::object());
                        if (!generation.is_object()) {
                            throw ApiError(
                                400, "invalid_request_error",
                                "session config must be an object", "config");
                        }
                        const auto & duplex_defaults = config.runtime_profile.duplex;
                        parameters.greedy = generation.value(
                            "decode_mode", duplex_defaults.decode_mode.value_or("sampling")) ==
                            "greedy";
                        parameters.temperature = number_field(
                            generation, "temperature", duplex_defaults.temperature.value_or(0.7));
                        parameters.top_k = static_cast<int32_t>(integer_field(
                            generation, "top_k", duplex_defaults.top_k.value_or(100)));
                        parameters.top_p = number_field(
                            generation, "top_p", duplex_defaults.top_p.value_or(0.8));
                        parameters.listen_probability_scale = number_field(
                            generation, "listen_prob_scale", duplex_defaults.listen_prob_scale.value_or(1.0));
                        parameters.repetition_penalty = number_field(
                            generation, "text_repetition_penalty", duplex_defaults.text_repetition_penalty.value_or(1.05));
                        parameters.repetition_window =
                            static_cast<int32_t>(integer_field(
                                generation,
                                "text_repetition_window_size",
                                duplex_defaults.text_repetition_window_size.value_or(512)));
                        parameters.length_penalty = number_field(
                            generation, "length_penalty", duplex_defaults.length_penalty.value_or(1.0));
                        parameters.tts_temperature = number_field(
                            generation, "tts_temperature",
                            config.runtime_profile.tts.temperature.value_or(0.8));
                        parameters.tts_repetition_penalty = number_field(
                            generation, "tts_repetition_penalty",
                            config.runtime_profile.tts.repetition_penalty.value_or(1.05));
                        if (generation.contains("seed")) {
                            parameters.seed = static_cast<uint64_t>(
                                integer_field(generation, "seed", 0));
                        } else {
                            std::random_device random;
                            parameters.seed =
                                (static_cast<uint64_t>(random()) << 32) ^
                                static_cast<uint64_t>(random());
                        }

                        try {
                            duplex.start(parameters);
                            std::lock_guard<std::mutex> lock(duplex_gate);
                            if (duplex_session_id == owned_session) {
                                duplex_backend_started = true;
                            }
                        } catch (...) {
                            std::lock_guard<std::mutex> lock(duplex_gate);
                            if (duplex_session_id == owned_session) {
                                duplex_session_id.clear();
                                duplex_socket = nullptr;
                            }
                            owned_session.clear();
                            throw;
                        }
                        send_event({
                            {"type", "session.created"},
                            {"session_id", owned_session},
                            {"mode", "full_duplex"},
                            {"metrics", {{"backend", duplex_backend_name}}},
                        });
                        continue;
                    }

                    if (type != "input.append") {
                        throw ApiError(
                            400, "invalid_request_error",
                            "unsupported duplex message type: " + type);
                    }
                    if (owned_session.empty()) {
                        throw ApiError(
                            409, "conflict",
                            "session.init must precede input.append");
                    }
                    const json input = body.value("input", json::object());
                    if (!input.is_object()) {
                        throw ApiError(
                            400, "invalid_request_error",
                            "input must be an object", "input");
                    }
                    const bool has_audio = input.contains("audio_features");
                    const bool has_text = input.contains("text");
                    if (!has_audio && !has_text) {
                        throw ApiError(
                            400, "invalid_request_error",
                            "input requires audio_features or text", "input");
                    }
                    MfqDuplexStepInput step;
                    if (has_audio) {
                        if (!input["audio_features"].is_string()) {
                            throw ApiError(
                                400, "invalid_request_error",
                                "input.audio_features must be base64 float32 Mel data",
                                "audio_features");
                        }
                        step.audio_frames = static_cast<int32_t>(integer_field(
                            input, "audio_frames", 0));
                        step.audio_features = decode_audio_features(
                            input["audio_features"].get<std::string>(),
                            step.audio_frames);
                        step.audio_prefix_extra_frames = integer_field(
                            input, "audio_prefix_extra_frames", 0);
                        step.audio_suffix_extra_frames = integer_field(
                            input, "audio_suffix_extra_frames", 0);
                    }
                    if (has_text) {
                        if (!input["text"].is_string() ||
                            input["text"].get_ref<const std::string&>().empty()) {
                            throw ApiError(
                                400, "invalid_request_error",
                                "input.text must be a non-empty string", "text");
                        }
                        step.text_tokens = tokenizer->tokenize(
                            input["text"].get<std::string>(), false, false);
                        if (step.text_tokens.empty()) {
                            throw ApiError(
                                400, "invalid_request_error",
                                "input.text produced no tokens", "text");
                        }
                    }
                    step.max_new_speak_tokens = static_cast<int32_t>(
                        integer_field(
                            input,
                            "max_new_speak_tokens",
                            config.runtime_profile.duplex
                                .max_new_speak_tokens_per_chunk
                                .value_or(20)));
                    if (input.contains("force_listen") &&
                        !input["force_listen"].is_boolean()) {
                        throw ApiError(
                            400, "invalid_request_error",
                            "force_listen must be boolean", "force_listen");
                    }
                    step.force_listen = input.value("force_listen", false);
                    if (input.contains("force_speak") &&
                        !input["force_speak"].is_boolean()) {
                        throw ApiError(
                            400, "invalid_request_error",
                            "force_speak must be boolean", "force_speak");
                    }
                    step.force_speak = input.value("force_speak", false);
                    if (step.force_listen && step.force_speak) {
                        throw ApiError(
                            400, "invalid_request_error",
                            "force_listen and force_speak are mutually exclusive");
                    }

                    const auto result = duplex.step(step);
                    const std::string response_id = request_id("resp-");
                    json metrics = {
                        {"backend", duplex_backend_name},
                        {"wall_clock_ms", result.inference_ms},
                        {"kv_cache_length", result.language_cache_position},
                        {"audio_cache_length", result.audio_cache_position},
                        {"tts_cache_length", result.tts_cache_position},
                        {"audio_chunk_index", result.audio_chunk_index},
                    };

                    std::string text_delta;
                    for (const int64_t token : result.generated_tokens) {
                        if (session_controls.count(token) == 0) {
                            text_delta += tokenizer->piece(token, false);
                        }
                    }
                    if (!text_delta.empty()) {
                        send_event({
                            {"type", "response.output.delta"},
                            {"kind", "text"},
                            {"text", text_delta},
                            {"session_id", owned_session},
                            {"response_id", response_id},
                            {"end_of_turn", result.end_of_turn},
                            {"metrics", metrics},
                        });
                    }
                    if (!result.audio_tokens.empty() ||
                        (result.end_of_turn && !result.is_listen)) {
                        send_event({
                            {"type", "response.output.delta"},
                            {"kind", "audio_tokens"},
                            {"audio_tokens", result.audio_tokens},
                            {"session_id", owned_session},
                            {"response_id", response_id},
                            {"end_of_turn", result.end_of_turn},
                            {"force_flush", result.tts_force_flush},
                            {"metrics", metrics},
                        });
                    }
                    if (result.is_listen) {
                        send_event({
                            {"type", "response.output.delta"},
                            {"kind", "listen"},
                            {"session_id", owned_session},
                            {"response_id", response_id},
                            {"metrics", metrics},
                        });
                    }
                    send_event({
                        {"type", "response.step.done"},
                        {"session_id", owned_session},
                        {"response_id", response_id},
                        {"end_of_turn", result.end_of_turn},
                        {"metrics", metrics},
                    });
                }
            } catch (const std::exception & error) {
                send_event({
                    {"type", "session.closed"},
                    {"session_id", owned_session},
                    {"reason", "backend_error"},
                    {"diagnostic", {{"message", error.what()}}},
                });
                if (ws.is_open()) {
                    ws.close(
                        httplib::ws::CloseStatus::InternalError,
                        "duplex backend error");
                }
            }
            if (!owned_session.empty()) {
                stop_duplex_session(owned_session, false);
            }
        });

        server.Post(R"(/runtime/realtime/sessions/([A-Za-z0-9_-]+)/close)",
            [&](const httplib::Request & req, httplib::Response & res) {
                if (!authorized(req, res, config.api_key)) return;
                const std::string session_id = req.matches[1].str();
                if (!stop_duplex_session(session_id, true)) {
                    set_json(res, error_body(
                        "duplex session was not found", "not_found"), 404);
                    return;
                }
                set_json(res, {
                    {"ok", true},
                    {"session_id", session_id},
                    {"closed", true},
                });
            });
    }

    server.Get("/", [&](const httplib::Request & req, httplib::Response & res) {
        if (!authorized(req, res, config.api_key)) return;
        set_json(res, {
            {"name", "MFQ C++ HTTP runtime transport"},
            {"model", config.model_name},
            {"endpoints", {
                "/runtime/generate", "/runtime/models",
                "/runtime/health", "/runtime/status", "/runtime/reload",
                "/runtime/realtime", "/runtime/cache/clear",
                "/runtime/cache/trim", "/runtime/sessions/fork",
                "/runtime/sessions/{id}",
                "/runtime/sessions/{id}/cancel",
            }},
        });
    });

    const auto add_runtime_metrics = [&](json & value) {
        if (!runtime_metrics) return;
        for (const auto & item : runtime_metrics()) {
            value[item.first] = item.second;
        }
    };
    const auto add_request_runtime_metrics = [&](json & value) {
        if (!runtime_metrics) return;
        static const std::unordered_set<std::string> request_metric_names{
            "mtp_available",
            "mtp_used",
            "mtp_cycles",
            "mtp_drafted_tokens",
            "mtp_accepted_tokens",
            "mtp_acceptance_rate",
            "mtp_selected_depth",
            "mtp_depth_0_cycles",
            "mtp_depth_1_cycles",
            "mtp_depth_2_cycles",
            "mtp_depth_3_cycles",
            "mtp_depth_4_cycles",
            "mtp_depth_5_cycles",
            "mtp_position_1_acceptance_rate",
            "mtp_position_2_acceptance_rate",
            "mtp_position_3_acceptance_rate",
            "mtp_position_4_acceptance_rate",
            "mtp_position_5_acceptance_rate",
            "mtp_depth_0_cycle_ms",
            "mtp_depth_1_cycle_ms",
            "mtp_depth_2_cycle_ms",
            "mtp_depth_3_cycle_ms",
            "mtp_depth_4_cycle_ms",
            "mtp_depth_5_cycle_ms",
            "mtp_target_ms",
            "mtp_head_ms",
            "mtp_rollback_ms",
        };
        for (const auto & item : runtime_metrics()) {
            if (request_metric_names.find(item.first) !=
                request_metric_names.end()) {
                value[item.first] = item.second;
            }
        }
    };
    const auto add_session_metrics = [&](json & value) {
        if (!session_control.metrics) return;
        for (const auto & item : session_control.metrics()) {
            value[item.first] = item.second;
        }
    };

    server.Get("/runtime/health", [&](const httplib::Request &, httplib::Response & res) {
        json health = {
            {"status", reloading.load() ? "loading" : "ok"},
            {"model", config.model_name},
            {"model_type", config.model_type},
            {"model_capabilities", model_capabilities},
            {"vision_supported", vision_supported},
            {"vision_available", vision_available},
            {"video_available", video_available},
            {"mtp_supported", model_capability_profile.mtp},
            {"mtp_available", false},
            {"vision_enabled_default", sampling_defaults.enable_vision},
            {"mtp_enabled_default", sampling_defaults.enable_mtp},
            {"max_context", active_context.load()},
            {"duplex_available", static_cast<bool>(duplex)},
            {"duplex_active", duplex_is_active()},
            {"sampling_defaults", sampling_params_json(sampling_defaults)},
            {"duplex_sampling_defaults", duplex_sampling_defaults},
            {"tts_sampling_defaults", tts_sampling_defaults},
            {"runtime_profile_source", config.runtime_profile.source},
            {"chat_template_capabilities", chat_template_capabilities},
        };
        add_runtime_metrics(health);
        add_session_metrics(health);
        set_json(res, health);
    });

    server.Get("/runtime/realtime/capabilities", [&] (
            const httplib::Request & req, httplib::Response & res) {
        if (!authorized(req, res, config.api_key)) return;
        set_json(res, {
            {"available", static_cast<bool>(duplex)},
            {"modes", duplex ? json::array({"audio"}) : json::array()},
        });
    });

    server.Get("/runtime/status", [&](const httplib::Request & req, httplib::Response & res) {
        if (!authorized(req, res, config.api_key)) return;
        json status = request_metrics_store.snapshot(
            config, active_context.load(), reloading.load());
        status["sampling_defaults"] = sampling_params_json(
            sampling_defaults);
        status["duplex_sampling_defaults"] = duplex_sampling_defaults;
        status["tts_sampling_defaults"] = tts_sampling_defaults;
        status["runtime_profile_source"] = config.runtime_profile.source;
        status["chat_template_capabilities"] =
            chat_template_capabilities;
        status["model_capabilities"] = model_capabilities;
        status["vision_supported"] = vision_supported;
        status["vision_available"] = vision_available;
        status["video_available"] = video_available;
        status["mtp_supported"] = model_capability_profile.mtp;
        status["mtp_available"] = false;
        status["vision_enabled_default"] = sampling_defaults.enable_vision;
        status["mtp_enabled_default"] = sampling_defaults.enable_mtp;
        status["duplex_available"] = static_cast<bool>(duplex);
        status["duplex_active"] = duplex_is_active();
        add_runtime_metrics(status);
        add_session_metrics(status);
        set_json(res, status);
    });

    server.Post("/runtime/cache/clear", [&] (
            const httplib::Request & req, httplib::Response & res) {
        if (!authorized(req, res, config.api_key)) return;
        if (!session_control.clear) {
            set_json(res, error_body(
                "this runtime does not expose a prefix cache",
                "unsupported_operation"), 501);
            return;
        }
        {
            std::lock_guard<std::mutex> gate(reload_gate);
            bool expected = false;
            if (!reloading.compare_exchange_strong(expected, true)) {
                set_json(res, error_body(
                    "a runtime control operation is already in progress",
                    "conflict"), 409);
                return;
            }
            if (request_metrics_store.active_requests() != 0 ||
                duplex_is_active()) {
                reloading.store(false);
                set_json(res, error_body(
                    "cannot clear the prefix cache while a generation or "
                    "duplex session is active",
                    "conflict"), 409);
                return;
            }
        }
        try {
            const size_t released = session_control.clear();
            json result = {
                {"status", "ok"},
                {"released_snapshots", released},
            };
            add_session_metrics(result);
            reloading.store(false);
            set_json(res, result);
        } catch (const std::exception & error) {
            reloading.store(false);
            set_json(res, error_body(error.what(), "server_error"), 500);
        }
    });

    server.Post("/runtime/cache/trim", [&] (
            const httplib::Request & req, httplib::Response & res) {
        if (!authorized(req, res, config.api_key)) return;
        if (!session_control.trim_hot) {
            set_json(res, error_body(
                "this runtime does not expose a tiered prefix cache",
                "unsupported_operation"), 501);
            return;
        }
        try {
            const json body = parse_body(req);
            if (!body.is_object()) {
                throw ApiError(
                    400, "invalid_request_error",
                    "request body must be a JSON object");
            }
            std::uint64_t target_bytes = 0;
            if (body.contains("target_bytes")) {
                if (!body["target_bytes"].is_number_unsigned() &&
                    !(body["target_bytes"].is_number_integer() &&
                      body["target_bytes"].get<std::int64_t>() >= 0)) {
                    throw ApiError(
                        400, "invalid_request_error",
                        "target_bytes must be a non-negative integer",
                        "target_bytes");
                }
                target_bytes = body["target_bytes"].get<std::uint64_t>();
            }
            const auto released = session_control.trim_hot(target_bytes);
            json result = {
                {"status", "ok"},
                {"released_bytes", released},
                {"target_bytes", target_bytes},
            };
            add_session_metrics(result);
            set_json(res, result);
        } catch (const ApiError & error) {
            handle_api_error(res, error);
        } catch (const std::exception & error) {
            set_json(res, error_body(error.what(), "server_error"), 500);
        }
    });

    server.Post(
        R"(/runtime/sessions/([A-Za-z0-9._:-]{1,128})/cancel)",
        [&] (const httplib::Request & req, httplib::Response & res) {
            if (!authorized(req, res, config.api_key)) return;
            const std::string session_id = req.matches[1].str();
            set_json(res, {
                {"status", "ok"},
                {"cancelled", scheduler.cancel_request(session_id)},
            });
        });

    server.Post("/runtime/sessions/fork", [&] (
            const httplib::Request & req, httplib::Response & res) {
        if (!authorized(req, res, config.api_key)) return;
        if (!session_control.fork) {
            set_json(res, error_body(
                "this runtime does not support session forks",
                "unsupported_operation"), 501);
            return;
        }
        try {
            const json body = parse_body(req);
            const auto read_session_id = [&](const char * field) {
                if (!body.contains(field) || !body[field].is_string()) {
                    throw ApiError(
                        400, "invalid_request_error",
                        std::string(field) + " must be a string", field);
                }
                auto session_id = body[field].get<std::string>();
                if (!valid_mfq_session_id(session_id)) {
                    throw ApiError(
                        400, "invalid_request_error",
                        std::string(field) +
                            " must contain 1 to 128 safe identifier bytes",
                        field);
                }
                return session_id;
            };
            const std::string source_session_id =
                read_session_id("source_session_id");
            const std::string target_session_id =
                read_session_id("target_session_id");
            if (source_session_id == target_session_id) {
                throw ApiError(
                    400, "invalid_request_error",
                    "source and target sessions must differ",
                    "target_session_id");
            }
            const size_t copied = session_control.fork(
                source_session_id, target_session_id);
            set_json(res, {
                {"status", "ok"},
                {"copied_snapshots", copied},
            });
        } catch (const ApiError & error) {
            handle_api_error(res, error);
        } catch (const std::exception & error) {
            set_json(res, error_body(error.what(), "server_error"), 500);
        }
    });

    server.Delete(
        R"(/runtime/sessions/([A-Za-z0-9._:-]{1,128}))",
        [&] (const httplib::Request & req, httplib::Response & res) {
            if (!authorized(req, res, config.api_key)) return;
            if (!session_control.close) {
                set_json(res, error_body(
                    "this runtime does not support session close",
                    "unsupported_operation"), 501);
                return;
            }
            try {
                const std::string session_id = req.matches[1].str();
                const size_t released = session_control.close(session_id);
                set_json(res, {
                    {"status", "ok"},
                    {"released_snapshots", released},
                });
            } catch (const std::exception & error) {
                set_json(res, error_body(error.what(), "server_error"), 500);
            }
        });

    server.Post("/runtime/reload", [&](const httplib::Request & req, httplib::Response & res) {
        if (!authorized(req, res, config.api_key)) return;
        if (!scheduler.supports_reload()) {
            set_json(res, error_body(
                "this runtime does not support model reload",
                "unsupported_operation"), 501);
            return;
        }
        {
            std::lock_guard<std::mutex> gate(reload_gate);
            bool expected = false;
            if (!reloading.compare_exchange_strong(expected, true)) {
                set_json(res, error_body(
                    "model reload is already in progress", "conflict"), 409);
                return;
            }
            if (request_metrics_store.active_requests() != 0 ||
                duplex_is_active()) {
                reloading.store(false);
                set_json(res, error_body(
                    "cannot reload while a generation or duplex session is active",
                    "conflict"), 409);
                return;
            }
        }
        const auto finish_reload = [&] {
            reloading.store(false);
        };
        try {
            const json body = parse_body(req);
            const int64_t context_size = integer_field(
                body, "context_size", active_context.load());
            const int64_t capacity = config.context_capacity > 0
                ? config.context_capacity
                : config.max_context;
            if (context_size < 1 ||
                (capacity > 0 && context_size > capacity)) {
                throw ApiError(
                    400, "invalid_request_error",
                    "context_size must be within the model context capacity",
                    "context_size");
            }
            const int64_t loaded_context = scheduler.reload(context_size);
            if (loaded_context < 1 ||
                (capacity > 0 && loaded_context > capacity)) {
                throw std::runtime_error(
                    "runtime reload returned an invalid context size");
            }
            active_context.store(loaded_context);
            finish_reload();
            set_json(res, {
                {"status", "ok"},
                {"model", config.model_name},
                {"max_context", loaded_context},
                {"context_capacity", capacity},
            });
        } catch (const ApiError & error) {
            finish_reload();
            handle_api_error(res, error);
        } catch (const std::exception & error) {
            finish_reload();
            set_json(res, error_body(error.what(), "server_error"), 500);
        }
    });

    server.Get("/runtime/models", [&](const httplib::Request & req, httplib::Response & res) {
        if (!authorized(req, res, config.api_key)) return;
        set_json(res, {
            {"models", json::array({{
                {"name", config.model_name},
                {"type", config.model_type},
                {"capabilities", model_capabilities},
            }})},
        });
    });

    auto runtime_generate_handler = [&](const httplib::Request & req, httplib::Response & res) {
        if (!authorized(req, res, config.api_key)) return;
        if (duplex_is_active()) {
            set_json(res, error_body(
                "the model is reserved by an active duplex session",
                "conflict"), 409);
            return;
        }
        if (reloading.load()) {
            set_json(res, error_body(
                "model reload is in progress", "service_unavailable"), 503);
            return;
        }
        try {
            const json body = runtime_generate_body(parse_body(req));
            RequestWork work = parse_work(
                body, true, *tokenizer, chat_templates.get(),
                active_context.load(), config.model_type,
                sampling_defaults);
            if (body.contains("mfq_multimodal")) {
                if (!work.sampling.enable_vision) {
                    throw ApiError(
                        400, "invalid_request_error",
                        "vision is disabled for this request; set enable_vision=true",
                        "enable_vision");
                }
                if (!scheduler.supports_multimodal_generation()) {
                    throw ApiError(
                        501, "unsupported_parameter",
                        "the loaded model has no native vision runtime",
                        "mfq_multimodal");
                }
                work.vision = parse_mfq_vision(
                    body["mfq_multimodal"], work.prompt, *tokenizer,
                    config.vocab_size);
                if (active_context.load() > 0 &&
                    static_cast<int64_t>(work.prompt.size()) +
                        work.sampling.max_tokens > active_context.load()) {
                    throw ApiError(
                        400, "context_length_exceeded",
                        "expanded multimodal prompt plus max_tokens exceed "
                        "the model context window",
                        "max_tokens");
                }
            }
            const std::string id = request_id("run-");
            const int64_t created = unix_time_seconds();
            std::shared_ptr<ActiveRequest> active_request;
            {
                std::lock_guard<std::mutex> gate(reload_gate);
                if (reloading.load()) {
                    set_json(res, error_body(
                        "model reload is in progress",
                        "service_unavailable"), 503);
                    return;
                }
                active_request =
                    std::make_shared<ActiveRequest>(request_metrics_store);
            }
            auto cancellation = scheduler.activate_request(
                work.cache_plan.session_id, true);

            if (!work.stream) {
                RequestMetrics metrics;
                CompletionResult result = generate_text(
                    work, *tokenizer, scheduler,
                    cancellation->cancel_flag(),
                    [](const common_chat_msg_diff &) {
                        return true;
                    },
                    &metrics,
                    true);
                const RequestMetricValues metric_values =
                    request_metric_values(result, metrics);
                log_request_metrics(
                    id, true, false, work.prompt.size(), work.sampling,
                    result, metric_values);
                active_request->complete(
                    id, true, false, work.prompt.size(), result, metric_values);
                auto performance =
                    request_metric_values_json(metric_values, work.sampling);
                add_request_runtime_metrics(performance);
                set_json(res, runtime_generation_result(
                    id, created, config.model_name, result,
                    usage_json(work.prompt.size(), result.completion_tokens),
                    std::move(performance)));
                return;
            }

            res.set_header("Cache-Control", "no-cache");
            res.set_header("X-Accel-Buffering", "no");
            res.set_chunked_content_provider(
                "text/event-stream; charset=utf-8",
                [work = std::move(work), id, created, &tokenizer, &scheduler,
                 &config, active_request, cancellation,
                 &add_request_runtime_metrics]
                (size_t offset, httplib::DataSink & sink) mutable -> bool {
                    if (offset != 0) {
                        sink.done();
                        return false;
                    }
                    try {
                        RequestMetrics metrics;
                        CompletionResult result = generate_text(
                            work, *tokenizer, scheduler,
                            cancellation->cancel_flag(),
                            [&](const common_chat_msg_diff & diff) {
                                json delta = chat_diff_json(diff);
                                if (delta.empty()) return true;
                                auto event = runtime_generation_event(
                                    "delta", id, created, config.model_name);
                                event["delta"] = std::move(delta);
                                return write_sse(sink, event);
                        }, &metrics, false);
                        const RequestMetricValues metric_values =
                            request_metric_values(result, metrics);
                        log_request_metrics(
                            id, true, true, work.prompt.size(), work.sampling,
                            result, metric_values);
                        active_request->complete(
                            id, true, true, work.prompt.size(), result,
                            metric_values);
                        if (!result.client_connected) return false;
                        auto complete = runtime_generation_event(
                            "complete", id, created, config.model_name);
                        complete["finish_reason"] = result.finish_reason;
                        auto performance = request_metric_values_json(
                            metric_values, work.sampling);
                        add_request_runtime_metrics(performance);
                        complete["metrics"] = std::move(performance);
                        if (!write_sse(sink, complete)) return false;
                        if (work.include_usage) {
                            auto usage = runtime_generation_event(
                                "usage", id, created, config.model_name);
                            usage["usage"] = usage_json(
                                work.prompt.size(), result.completion_tokens);
                            if (!write_sse(sink, usage)) return false;
                        }
                        static constexpr char done[] = "data: [DONE]\n\n";
                        if (!sink.write(done, sizeof(done) - 1)) return false;
                    } catch (const std::exception & error) {
                        write_sse(sink, error_body(error.what(), "runtime_error"));
                    }
                    sink.done();
                    return false;
                });
        } catch (const ApiError & error) {
            handle_api_error(res, error);
        } catch (const std::exception & error) {
            set_json(res, error_body(error.what(), "server_error"), 500);
        }
    };

    server.Post("/runtime/generate", runtime_generate_handler);

    server.set_exception_handler([](const httplib::Request &, httplib::Response & res, std::exception_ptr ep) {
        std::string message = "unhandled server exception";
        try {
            if (ep) std::rethrow_exception(ep);
        } catch (const std::exception & error) {
            message = error.what();
        }
        set_json(res, error_body(message, "server_error"), 500);
    });

    if (!server.bind_to_port(config.host, config.port)) {
        throw std::runtime_error(
            "failed to bind " + config.host + ":" +
            std::to_string(config.port));
    }
    const std::string endpoint = "http://" + config.host + ":" +
        std::to_string(config.port);
    std::cout << "MFQ HTTP runtime transport ready: " << endpoint
              << " model=" << config.model_name
              << " context=" << config.max_context
              << " vocab=" << tokenizer->vocab_size() << std::endl;
    if (!server.listen_after_bind()) {
        throw std::runtime_error(
            "runtime transport stopped after binding " + config.host + ":" +
            std::to_string(config.port));
    }
    return 0;
}

class MfqHttpTransport final : public MfqTransport {
public:
    explicit MfqHttpTransport(MfqHttpRuntimeTransportConfig config)
        : config_(std::move(config)) {}

    int run(const MfqScheduler & scheduler) override {
        return run_mfq_http_transport(config_, scheduler);
    }

private:
    MfqHttpRuntimeTransportConfig config_;
};

} // namespace

std::unique_ptr<MfqTransport> make_mfq_http_transport(
        MfqHttpRuntimeTransportConfig config) {
    return std::make_unique<MfqHttpTransport>(std::move(config));
}
