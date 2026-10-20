#include "server.h"
#include "instance.h"

#include "httplib.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#else
#include <cerrno>
#include <poll.h>
#include <unistd.h>
#endif

namespace mfq::server {
namespace {

using json = nlohmann::json;
namespace fs = std::filesystem;

void set_json(httplib::Response& response, const json& body, int status = 200) {
    response.status = status;
    response.set_content(body.dump(), "application/json; charset=utf-8");
}

void error(httplib::Response& response, int status, const char* code, const char* message) {
    set_json(response, {{"error", {{"code", code}, {"message", message}}}}, status);
}

bool within(const fs::path& root, const fs::path& path) {
    return std::mismatch(root.begin(), root.end(), path.begin(), path.end()).first == root.end();
}

bool reserved_path(const std::string& path) {
    constexpr std::array<std::string_view, 8> reserved{
        "api", "v1", "realtime", "health", "docs", "redoc", "assets", "static"};
    const auto first = std::string_view(path).substr(1, path.find('/', 1) - 1);
    return std::find(reserved.begin(), reserved.end(), first) != reserved.end();
}

bool invalid_path(const std::string& path) {
    if (path.find('\0') != std::string::npos || path.find('\\') != std::string::npos) return true;
    for (const auto& part : fs::u8path(path)) if (part == "..") return true;
    return false;
}

void static_response(const fs::path& root, const httplib::Request& request, httplib::Response& response) {
    if (invalid_path(request.path)) {
        error(response, 404, "not_found", "resource not found");
        return;
    }
    const auto first = std::string_view(request.path).substr(1, request.path.find('/', 1) - 1);
    if (reserved_path(request.path) && first != "assets" && first != "static") {
        error(response, 404, "not_found", "resource not found");
        return;
    }
    auto path = fs::weakly_canonical(root / fs::u8path(request.path).relative_path());
    if (fs::is_directory(path)) path = fs::weakly_canonical(path / "index.html");
    if (!within(root, path)) {
        error(response, 403, "forbidden", "resource is outside the web root");
        return;
    }
    if (!fs::is_regular_file(path)) {
        auto accept = request.get_header_value("Accept");
        std::transform(accept.begin(), accept.end(), accept.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        const auto destination = request.get_header_value("Sec-Fetch-Dest");
        auto normalized = request.path;
        while (!normalized.empty() && normalized.back() == '/') normalized.pop_back();
        if (reserved_path(request.path) || accept.find("text/html") == std::string::npos ||
            (destination != "" && destination != "document" && destination != "iframe") ||
            fs::u8path(normalized).has_extension()) {
            error(response, 404, "not_found", "resource not found");
            return;
        }
        path = fs::weakly_canonical(root / "index.html");
        if (!within(root, path)) {
            error(response, 403, "forbidden", "resource is outside the web root");
            return;
        }
    }
    response.set_file_content(path.u8string());
}

bool stdin_closed() {
    char buffer[1024];
#ifdef _WIN32
    const auto input = GetStdHandle(STD_INPUT_HANDLE);
    if (!input || input == INVALID_HANDLE_VALUE) return true;
    const auto kind = GetFileType(input);
    if (kind == FILE_TYPE_CHAR) return false;
    if (kind == FILE_TYPE_PIPE) {
        DWORD available = 0;
        if (!PeekNamedPipe(input, nullptr, 0, nullptr, &available, nullptr)) return true;
        if (!available) return false;
    }
    DWORD count = 0;
    return !ReadFile(input, buffer, sizeof(buffer), &count, nullptr) || count == 0;
#else
    pollfd input{STDIN_FILENO, POLLIN, 0};
    const auto ready = ::poll(&input, 1, 0);
    if (ready < 0) return errno != EINTR;
    if (!ready) return false;
    if (input.revents & (POLLERR | POLLNVAL)) return true;
    const auto count = ::read(STDIN_FILENO, buffer, sizeof(buffer));
    return count == 0 || (count < 0 && errno != EINTR && errno != EAGAIN);
#endif
}

} // namespace

int run(const Config& config, const std::atomic<bool>& stop_requested) {
    fs::path web_root;
    if (!config.web_root.empty()) {
        web_root = resolved_path(config.web_root);
        const auto index = fs::weakly_canonical(web_root / "index.html");
        if (!fs::is_directory(web_root) || !fs::is_regular_file(index) || !within(web_root, index))
            throw std::invalid_argument("--web-root must contain an index.html inside that directory");
    }
    InstanceLock instance(config.data_dir, config.instance_id, config.exit_on_stdin_close);
    httplib::Server http;
    http.set_socket_options([](socket_t socket) {
        // A public listener must fail on conflicts, not share traffic via SO_REUSEPORT.
#ifdef _WIN32
        constexpr int option = SO_EXCLUSIVEADDRUSE;
#else
        constexpr int option = SO_REUSEADDR;
#endif
        if (!httplib::set_socket_opt(socket, SOL_SOCKET, option, 1))
            throw std::runtime_error("cannot configure exclusive HTTP listener");
    });
    http.new_task_queue = [] { return new httplib::ThreadPool(4, 0, 32); };
    http.set_payload_max_length(1024 * 1024);
    http.set_read_timeout(5);
    http.set_write_timeout(5);
    http.set_keep_alive_timeout(1);
    http.set_pre_routing_handler([&](const httplib::Request& request, httplib::Response& response) {
        const auto expected = request.get_header_value("X-MFQ-Instance-ID");
        if (!expected.empty() && expected != instance.identity()["instance_id"].get<std::string>()) {
            error(response, 409, "server_instance_changed", "The server instance changed. Reconnect and confirm the new server.");
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });
    http.set_post_routing_handler([](const httplib::Request&, httplib::Response& response) {
        response.set_header("X-Content-Type-Options", "nosniff");
        response.set_header("X-Frame-Options", "DENY");
        response.set_header("Referrer-Policy", "no-referrer");
        response.set_header("Content-Security-Policy", "default-src 'self'; connect-src 'self' ws: wss:; "
            "img-src 'self' data: blob:; media-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; "
            "script-src 'self'; font-src 'self' data:");
    });
    http.set_error_handler([](const httplib::Request&, httplib::Response& response) {
        if (response.body.empty()) error(response, response.status, "http_error", "request failed");
    });
    http.set_exception_handler([](const httplib::Request&, httplib::Response& response, std::exception_ptr) {
        error(response, 500, "internal_error", "internal server error");
    });
    http.Get("/health", [](const httplib::Request&, httplib::Response& response) {
        set_json(response, {{"status", "ok"}, {"service", "mfq-server"}, {"protocol_version", "1.0"}});
    });
    http.Get("/health/identity", [&](const httplib::Request&, httplib::Response& response) {
        set_json(response, instance.identity());
    });
    if (!web_root.empty()) http.Get("/.*", [&](const httplib::Request& request, httplib::Response& response) {
        static_response(web_root, request, response);
    });
    const int port = config.port == 0 ? http.bind_to_any_port(config.host) :
        (http.bind_to_port(config.host, config.port) ? config.port : -1);
    if (port < 0) throw std::runtime_error("cannot bind listener on " + config.host + ":" + std::to_string(config.port));
    http.set_start_handler([&] {
        std::cout << json{{"event", "listening"}, {"host", config.host}, {"port", port},
            {"instance_id", instance.identity()["instance_id"]}}.dump() << std::endl;
    });
    std::atomic<bool> finished{false};
    std::thread lifecycle([&] {
        while (!finished.load(std::memory_order_relaxed)) {
            if (http.is_running() && (stop_requested.load(std::memory_order_relaxed) ||
                    (config.exit_on_stdin_close && stdin_closed()))) {
                http.stop();
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
        }
    });
    bool successful;
    try {
        successful = http.listen_after_bind();
    } catch (...) {
        finished.store(true, std::memory_order_relaxed);
        lifecycle.join();
        throw;
    }
    finished.store(true, std::memory_order_relaxed);
    lifecycle.join();
    if (!successful) throw std::runtime_error("HTTP listener failed");
    std::cerr << "MFQ Server stopped\n";
    return 0;
}

} // namespace mfq::server
