#include "common.h"

#include <algorithm>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <iostream>
#include <limits>
#include <thread>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace {
using namespace mfq::transport;
std::mutex stdio_protocol_mutex;
int stdio_protocol_fd = -1;

bool write_stdio_protocol(const json& frame) {
    const auto line = frame.dump() + "\n";
    std::lock_guard lock(stdio_protocol_mutex);
    if (stdio_protocol_fd < 0) return false;
    size_t offset = 0;
    while (offset < line.size()) {
#ifdef _WIN32
        const auto remaining = std::min<size_t>(line.size() - offset, std::numeric_limits<int>::max());
        const int written = _write(stdio_protocol_fd, line.data() + offset, static_cast<unsigned int>(remaining));
#else
        const ssize_t written = ::write(stdio_protocol_fd, line.data() + offset, line.size() - offset);
#endif
        if (written < 0 && errno == EINTR) continue;
        if (written <= 0) return false;
        offset += static_cast<size_t>(written);
    }
    return true;
}

class MfqStdioTransport final : public MfqTransport {
public:
    explicit MfqStdioTransport(MfqRuntimeTransportConfig value) : config(std::move(value)) {}
    int run(const MfqScheduler& scheduler) override {
        if (stdio_protocol_fd < 0) throw std::runtime_error("prepare_mfq_stdio_transport must be called before model loading");
        RuntimeProtocol protocol(scheduler, config);
        struct Task { std::thread thread; std::shared_ptr<std::atomic<bool>> done; };
        std::vector<Task> tasks;
        const auto send = [](json frame) { frame["v"] = 1; return write_stdio_protocol(frame); };
        const auto send_result = [&](const std::string& id, json data) {
            return send({{"id", id}, {"type", "result"}, {"data", std::move(data)}});
        };
        const auto send_event = [&](const std::string& id, json data) {
            return send({{"id", id}, {"type", "event"}, {"data", std::move(data)}});
        };
        const auto send_done = [&](const std::string& id) { return send({{"id", id}, {"type", "done"}}); };
        const auto send_error = [&](const std::string& id, int status, const std::string& code, const std::string& message) {
            return send({{"id", id}, {"type", "error"}, {"error", {{"code", code}, {"message", message},
                {"status_code", status}, {"retryable", status == 429 || status == 502 || status == 503 || status == 504}}}});
        };
        const auto input_error = [&](const std::string& id, const mfq::engine::InferenceInputError& error) {
            using Code = mfq::engine::InferenceInputErrorCode;
            send_error(id, 400, error.code == Code::Unsupported ? "unsupported_parameter" :
                (error.code == Code::ContextLength ? "context_length_exceeded" : "invalid_request_error"), error.what());
        };
        const auto reap = [&](bool all) {
            for (auto item = tasks.begin(); item != tasks.end();) {
                if (all || item->done->load(std::memory_order_acquire)) {
                    item->thread.join(); item = tasks.erase(item);
                } else ++item;
            }
        };
        const auto launch = [&](const std::string& id, auto work) {
            auto done = std::make_shared<std::atomic<bool>>(false);
            tasks.push_back({std::thread([&, id, done, work = std::move(work)] {
                try { work(); }
                catch (const ApiError& error) { send_error(id, error.status, error.type, error.what()); }
                catch (const mfq::engine::InferenceInputError& error) { input_error(id, error); }
                catch (const json::exception& error) { send_error(id, 400, "invalid_request_error", error.what()); }
                catch (const std::exception& error) { send_error(id, 500, "server_error", error.what()); }
                done->store(true, std::memory_order_release);
            }), done});
        };
        send({{"type", "ready"}});
        std::string line;
        while (std::getline(std::cin, line)) {
            reap(false);
            std::string id;
            try {
                if (line.size() > 64ULL * 1024 * 1024) throw ApiError(413, "request_too_large", "stdio frame exceeds 64 MiB");
                const auto request = json::parse(line);
                if (!request.is_object() || request.value("v", 0) != 1)
                    throw ApiError(400, "backend_protocol_error", "stdio request requires protocol version 1");
                if (request.contains("id")) {
                    if (!request["id"].is_string()) throw ApiError(400, "backend_protocol_error", "stdio request id must be a string");
                    id = request["id"].get<std::string>();
                }
                if (!request.contains("op") || !request["op"].is_string())
                    throw ApiError(400, "backend_protocol_error", "stdio request requires a string op");
                const auto op = request["op"].get<std::string>();
                const auto params = request.value("params", json::object());
                if (!params.is_object()) throw ApiError(400, "backend_protocol_error", "stdio params must be an object");
                if (op == "shutdown") { if (!id.empty()) send_result(id, {{"status", "ok"}}); break; }
                if (op == "request.cancel") { const auto result = protocol.call(op, params); if (!id.empty()) send_result(id, result); continue; }
                if (id.empty()) throw ApiError(400, "backend_protocol_error", "stdio operation requires an id");
                if (op == "realtime.open") {
                    protocol.open_realtime(id, params.value("mode", std::string("audio")));
                    send_result(id, {{"status", "ok"}}); continue;
                }
                if (op == "realtime.close") {
                    const auto target = params.value("target_id", std::string{});
                    if (target.empty() || !protocol.close_realtime(target)) throw ApiError(404, "not_found", "realtime channel was not found");
                    send_done(target); send_result(id, {{"status", "ok"}}); continue;
                }
                if (op == "realtime.send") {
                    const auto target = params.value("target_id", std::string{});
                    auto body = json::parse(params.value("data", std::string{}));
                    launch(id, [&, id, target, body = std::move(body)] {
                        for (const auto& event : protocol.realtime_message(target, body))
                            if (!send_event(target, event.dump())) { protocol.close_realtime(target); return; }
                        send_result(id, {{"status", "ok"}});
                    });
                    continue;
                }
                if (op != "generate") { send_result(id, protocol.call(op, params)); continue; }
                auto completion = std::make_shared<CompletionStream>(scheduler, config, protocol.request_metrics(), params, id);
                launch(id, [&, id, completion] {
                    while (auto frames = completion->next()) for (auto& frame : *frames)
                        if (!send_event(id, std::move(frame))) { completion->cancel(); return; }
                    send_done(id);
                });
            } catch (const ApiError& error) { send_error(id, error.status, error.type, error.what()); }
            catch (const mfq::engine::InferenceInputError& error) { input_error(id, error); }
            catch (const json::exception& error) { send_error(id, 400, "backend_protocol_error", error.what()); }
            catch (const std::exception& error) { send_error(id, 500, "server_error", error.what()); }
        }
        scheduler.cancel_all();
        protocol.close_realtime();
        reap(true);
        return 0;
    }
private:
    MfqRuntimeTransportConfig config;
};
} // namespace

void prepare_mfq_stdio_transport() {
    std::lock_guard lock(stdio_protocol_mutex);
    if (stdio_protocol_fd >= 0) return;
    std::cout.flush(); std::fflush(stdout);
#ifdef _WIN32
    const HANDLE stdout_handle = reinterpret_cast<HANDLE>(_get_osfhandle(_fileno(stdout)));
    HANDLE protocol_handle = nullptr;
    if (stdout_handle == INVALID_HANDLE_VALUE || !DuplicateHandle(GetCurrentProcess(), stdout_handle,
            GetCurrentProcess(), &protocol_handle, 0, FALSE, DUPLICATE_SAME_ACCESS))
        throw std::runtime_error("cannot reserve stdout for MFQ stdio protocol");
    stdio_protocol_fd = _open_osfhandle(reinterpret_cast<intptr_t>(protocol_handle), _O_WRONLY | _O_BINARY);
    if (stdio_protocol_fd < 0) { CloseHandle(protocol_handle); throw std::runtime_error("cannot reserve stdout for MFQ stdio protocol"); }
    if (_dup2(_fileno(stderr), _fileno(stdout)) != 0) {
        _close(stdio_protocol_fd); stdio_protocol_fd = -1;
        throw std::runtime_error("cannot redirect runtime stdout to stderr");
    }
#else
    stdio_protocol_fd = ::dup(STDOUT_FILENO);
    if (stdio_protocol_fd < 0 || ::fcntl(stdio_protocol_fd, F_SETFD, FD_CLOEXEC) < 0 || ::dup2(STDERR_FILENO, STDOUT_FILENO) < 0) {
        if (stdio_protocol_fd >= 0) ::close(stdio_protocol_fd);
        stdio_protocol_fd = -1;
        throw std::runtime_error("cannot reserve stdout for MFQ stdio protocol");
    }
    std::signal(SIGPIPE, SIG_IGN);
#endif
    std::cout.clear();
}
std::unique_ptr<MfqTransport> make_mfq_stdio_transport(MfqRuntimeTransportConfig config) {
    return std::make_unique<MfqStdioTransport>(std::move(config));
}
