#include "common.h"
#include "httplib.h"

#include <future>
#include <iostream>
#include <mutex>

namespace {
using namespace mfq::transport;

json error_body(const std::string& message, const std::string& type, const std::string& param = {}) {
    return {{"error", {{"message", message}, {"type", type},
        {"param", param.empty() ? json(nullptr) : json(param)}, {"code", nullptr}}}};
}
void set_json(httplib::Response& response, const json& body, int status = 200) {
    response.status = status;
    response.set_content(body.dump(), "application/json; charset=utf-8");
}
bool authorized(const httplib::Request& request, httplib::Response& response, const std::string& key) {
    if (key.empty() || request.get_header_value("Authorization") == "Bearer " + key) return true;
    response.set_header("WWW-Authenticate", "Bearer");
    set_json(response, error_body("invalid API key", "authentication_error"), 401);
    return false;
}
json parse_body(const httplib::Request& request) {
    try { return request.body.empty() ? json::object() : json::parse(request.body); }
    catch (const json::parse_error& error) {
        throw ApiError(400, "invalid_request_error", std::string("invalid JSON: ") + error.what());
    }
}
bool write_sse(httplib::DataSink& sink, const json& value) {
    const auto event = "data: " + value.dump() + "\n\n";
    return sink.write(event.data(), event.size());
}

class MfqHttpTransport final : public MfqTransport {
public:
    explicit MfqHttpTransport(MfqHttpRuntimeTransportConfig value) : config(std::move(value)) {}
    int run(const MfqScheduler& scheduler) override;
private:
    MfqHttpRuntimeTransportConfig config;
};

int MfqHttpTransport::run(const MfqScheduler& scheduler) {
    if (config.port < 1 || config.port > 65535) throw std::runtime_error("runtime transport port must be in [1, 65535]");
    RuntimeProtocol protocol(scheduler, config);
    httplib::Server server;
    server.set_payload_max_length((scheduler.supports_multimodal_generation() ? 512ULL : 16ULL) * 1024 * 1024);
    server.set_read_timeout(300, 0);
    server.set_write_timeout(300, 0);
    server.set_keep_alive_max_count(100);
    server.set_default_headers({{"Access-Control-Allow-Origin", "*"},
        {"Access-Control-Allow-Headers", "Authorization, Content-Type"},
        {"Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS"},
        {"X-Content-Type-Options", "nosniff"}});
    server.Options(R"(.*)", [](const httplib::Request&, httplib::Response& response) { response.status = 204; });

    // These handlers translate route framing to protocol operations only.
    const auto handle = [&](std::string operation, bool public_route = false) {
        return [&, operation = std::move(operation), public_route](const httplib::Request& request, httplib::Response& response) {
            if (!public_route && !authorized(request, response, config.api_key)) return;
            try {
                json params = parse_body(request);
                if (operation == "session.cancel" || operation == "session.close") params["session_id"] = request.matches[1].str();
                set_json(response, protocol.call(operation, params));
            } catch (const ApiError& error) { set_json(response, error_body(error.what(), error.type, error.param), error.status); }
            catch (const json::exception& error) { set_json(response, error_body(error.what(), "invalid_request_error"), 400); }
            catch (const std::exception& error) { set_json(response, error_body(error.what(), "server_error"), 500); }
        };
    };
    server.Get("/runtime/health", handle("health", true));
    server.Get("/runtime/status", handle("status"));
    server.Get("/runtime/models", handle("models"));
    server.Get("/runtime/realtime/capabilities", handle("realtime.capabilities"));
    server.Post("/runtime/cache/clear", handle("cache.clear"));
    server.Post("/runtime/cache/trim", handle("cache.trim"));
    server.Post("/runtime/sessions/fork", handle("session.fork"));
    server.Post(R"(/runtime/sessions/([A-Za-z0-9._:-]{1,128})/cancel)", handle("session.cancel"));
    server.Delete(R"(/runtime/sessions/([A-Za-z0-9._:-]{1,128}))", handle("session.close"));
    server.Post("/runtime/reload", handle("reload"));
    server.Get("/", [&](const httplib::Request& request, httplib::Response& response) {
        if (!authorized(request, response, config.api_key)) return;
        set_json(response, {{"name", "MFQ C++ HTTP runtime transport"}, {"model", config.model_name},
            {"endpoints", {"/runtime/generate", "/runtime/models", "/runtime/health", "/runtime/status",
                "/runtime/reload", "/runtime/realtime", "/runtime/cache/clear", "/runtime/cache/trim",
                "/runtime/sessions/fork", "/runtime/sessions/{id}", "/runtime/sessions/{id}/cancel"}}});
    });

    std::mutex socket_gate;
    httplib::ws::WebSocket* active_socket = nullptr;
    if (scheduler.info().duplex) {
        server.WebSocket("/runtime/realtime", [&](const httplib::Request& request, httplib::ws::WebSocket& socket) {
            if (!config.api_key.empty() && request.get_header_value("Authorization") != "Bearer " + config.api_key) {
                socket.close(httplib::ws::CloseStatus::PolicyViolation, "authentication failed");
                return;
            }
            const auto channel = request_id("ws-");
            std::future<void> pending;
            const auto fail = [&](const std::exception& error, bool invalid) {
                if (!socket.is_open()) return;
                socket.send(protocol.realtime_closed(channel, invalid ? "invalid_request" : "backend_error", error.what()).dump());
                socket.close(invalid ? httplib::ws::CloseStatus::PolicyViolation : httplib::ws::CloseStatus::InternalError,
                    invalid ? "invalid realtime input" : "realtime backend error");
            };
            try {
                protocol.open_realtime(channel);
                { std::lock_guard lock(socket_gate); active_socket = &socket; }
                std::string message;
                while (socket.is_open()) {
                    const auto read = socket.read(message);
                    if (read == httplib::ws::ReadResult::Fail) break;
                    if (read != httplib::ws::ReadResult::Text)
                        throw ApiError(400, "invalid_request_error", "realtime accepts JSON text frames only");
                    if (pending.valid()) {
                        if (pending.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
                            throw ApiError(409, "conflict", "realtime operation is already pending");
                        pending.get();
                    }
                    auto body = json::parse(message);
                    pending = std::async(std::launch::async, [&, body = std::move(body)] {
                        try {
                            for (const auto& event : protocol.realtime_message(channel, body))
                                if (!socket.send(event.dump())) break;
                        } catch (const ApiError& error) { fail(error, error.status < 500); }
                        catch (const mfq::engine::InferenceInputError& error) { fail(error, true); }
                        catch (const json::exception& error) { fail(error, true); }
                        catch (const std::exception& error) { fail(error, false); }
                    });
                }
            } catch (const ApiError& error) { fail(error, error.status < 500); }
            catch (const json::exception& error) { fail(error, true); }
            catch (const std::exception& error) { fail(error, false); }
            protocol.close_realtime(channel);
            if (pending.valid()) pending.get();
            std::lock_guard lock(socket_gate);
            if (active_socket == &socket) active_socket = nullptr;
        });
        server.Post(R"(/runtime/realtime/sessions/([A-Za-z0-9_-]+)/close)",
            [&](const httplib::Request& request, httplib::Response& response) {
                if (!authorized(request, response, config.api_key)) return;
                const auto session = request.matches[1].str();
                std::lock_guard lock(socket_gate);
                if (!protocol.close_realtime({}, session)) {
                    set_json(response, error_body("realtime session was not found", "not_found"), 404);
                    return;
                }
                if (active_socket && active_socket->is_open()) active_socket->close(httplib::ws::CloseStatus::Normal, "session closed");
                set_json(response, {{"ok", true}, {"session_id", session}, {"closed", true}});
            });
    }

    server.Post("/runtime/generate", [&](const httplib::Request& request, httplib::Response& response) {
        if (!authorized(request, response, config.api_key)) return;
        try {
            auto completion = std::make_shared<CompletionStream>(scheduler, config, protocol.request_metrics(), parse_body(request));
            if (!completion->streaming()) { set_json(response, completion->next()->front()); return; }
            response.set_header("Cache-Control", "no-cache");
            response.set_header("X-Accel-Buffering", "no");
            response.set_chunked_content_provider("text/event-stream; charset=utf-8",
                [completion](size_t offset, httplib::DataSink& sink) {
                    if (offset != 0) { sink.done(); return false; }
                    try {
                        while (auto frames = completion->next()) for (const auto& frame : *frames)
                            if (!write_sse(sink, frame)) { completion->cancel(); return false; }
                        static constexpr char done[] = "data: [DONE]\n\n";
                        if (!sink.write(done, sizeof(done) - 1)) return false;
                    } catch (const ApiError& error) { write_sse(sink, error_body(error.what(), error.type, error.param)); }
                    catch (const std::exception& error) { write_sse(sink, error_body(error.what(), "runtime_error")); }
                    sink.done(); return false;
                });
        } catch (const ApiError& error) { set_json(response, error_body(error.what(), error.type, error.param), error.status); }
        catch (const json::exception& error) { set_json(response, error_body(error.what(), "invalid_request_error"), 400); }
        catch (const std::exception& error) { set_json(response, error_body(error.what(), "server_error"), 500); }
    });
    server.set_exception_handler([](const httplib::Request&, httplib::Response& response, std::exception_ptr failure) {
        try { if (failure) std::rethrow_exception(failure); }
        catch (const std::exception& error) { set_json(response, error_body(error.what(), "server_error"), 500); }
    });
    if (!server.bind_to_port(config.host, config.port)) throw std::runtime_error("failed to bind " + config.host + ":" + std::to_string(config.port));
    std::cout << "MFQ HTTP runtime transport ready: http://" << config.host << ':' << config.port
        << " model=" << config.model_name << " context=" << config.max_context << " vocab=" << scheduler.vocab_size() << std::endl;
    if (!server.listen_after_bind()) throw std::runtime_error("runtime transport stopped after binding " + config.host + ":" + std::to_string(config.port));
    return 0;
}
} // namespace

std::unique_ptr<MfqTransport> make_mfq_http_transport(MfqHttpRuntimeTransportConfig config) {
    return std::make_unique<MfqHttpTransport>(std::move(config));
}
