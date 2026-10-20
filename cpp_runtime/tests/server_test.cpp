#include "httplib.h"
#include "nlohmann/json.hpp"
#include "schema.h"
#include "server_test_process.h"

#include <sqlite3.h>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <random>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <share.h>
#include <sys/locking.h>
#include <sys/stat.h>
#else
#include <sys/file.h>
#endif

namespace {

using json = nlohmann::json;
namespace fs = std::filesystem;
using server_test::Process;
using server_test::read_file;
int checks = 0;
#define CHECK(value) do { ++checks; if (!(value)) throw std::runtime_error( \
    std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": " #value); } while (false)

fs::path executable;

void write_file(const fs::path& path, const std::string& content) {
    std::ofstream stream(path, std::ios::binary);
    stream << content;
    CHECK(stream.good());
}

template <typename T> void write_scalar(std::ofstream& stream, T value) {
    stream.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

void write_model_string(std::ofstream& stream, const std::string& value) {
    write_scalar(stream, static_cast<std::uint32_t>(value.size()));
    stream.write(value.data(), static_cast<std::streamsize>(value.size()));
}

void write_mfq(const fs::path& path) {
    const std::string config = R"({"model_type":"qwen3_5","max_position_embeddings":8192})";
    std::ofstream stream(path, std::ios::binary);
    stream.write("MFQ1", 4);
    write_scalar<std::uint32_t>(stream, 2);
    write_model_string(stream, "qwen3_5");
    write_scalar<std::uint32_t>(stream, 0);
    write_scalar<std::uint32_t>(stream, 2);
    write_model_string(stream, "model.token_embedding.weight");
    write_model_string(stream, "BF16");
    write_scalar<std::uint64_t>(stream, 14);
    write_model_string(stream, "__mfq_asset__/model_config.json");
    write_model_string(stream, "BLOB");
    write_scalar<std::uint64_t>(stream, config.size());
    write_scalar<std::uint32_t>(stream, 1);
    write_scalar<std::int64_t>(stream, 1);
    write_scalar<std::uint16_t>(stream, 0);
    stream << config;
    CHECK(stream.good());
}

void write_hf(const fs::path& path) {
    fs::create_directories(path);
    write_file(path / "config.json", R"({"model_type":"qwen3_5","text_config":{"max_position_embeddings":4096}})");
    const std::string header = R"({"model.embed_tokens.weight":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}})";
    std::ofstream stream(path / "model.safetensors", std::ios::binary);
    write_scalar<std::uint64_t>(stream, header.size());
    stream << header;
    write_scalar<std::uint16_t>(stream, 0);
    CHECK(stream.good());
}

class Database {
public:
    explicit Database(const fs::path& path) {
        if (sqlite3_open(path.u8string().c_str(), &database_) != SQLITE_OK) {
            sqlite3_close(database_);
            throw std::runtime_error("cannot open test database");
        }
        sqlite3_busy_timeout(database_, 5000);
    }
    ~Database() { sqlite3_close(database_); }
    void execute(const std::string& sql) {
        if (sqlite3_exec(database_, sql.c_str(), nullptr, nullptr, nullptr) != SQLITE_OK)
            throw std::runtime_error(sqlite3_errmsg(database_));
    }
    std::string scalar(const std::string& sql) {
        sqlite3_stmt* statement = nullptr;
        CHECK(sqlite3_prepare_v2(database_, sql.c_str(), -1, &statement, nullptr) == SQLITE_OK);
        const auto result = sqlite3_step(statement);
        std::string value;
        if (result == SQLITE_ROW && sqlite3_column_text(statement, 0))
            value = reinterpret_cast<const char*>(sqlite3_column_text(statement, 0));
        sqlite3_finalize(statement);
        CHECK(result == SQLITE_ROW);
        return value;
    }
private:
    sqlite3* database_ = nullptr;
};

struct Response {
    int status;
    httplib::Headers headers;
    std::string body;
    json document() const { return json::parse(body); }
    std::string header(const char* name) const {
        const auto found = headers.find(name);
        return found == headers.end() ? "" : found->second;
    }
};

struct Server {
    Process& process;
    std::string host;
    int port;
    Response raw(const std::string& method, const std::string& path, httplib::Headers headers = {}, const std::string& body = {}) {
        httplib::Client client(host == "0.0.0.0" ? "127.0.0.1" : host == "::" ? "::1" : host, port);
        client.set_connection_timeout(2);
        client.set_read_timeout(2);
        client.set_path_encode(false);
        httplib::Request request;
        request.method = method;
        request.path = path;
        request.headers = std::move(headers);
        request.body = body;
        const auto response = client.send(request);
        if (!response) throw std::runtime_error("HTTP request failed: " + process.output());
        return {response->status, response->headers, response->body};
    }
    Response get(const std::string& path, httplib::Headers headers = {}) { return raw("GET", path, std::move(headers)); }
    Response send(const std::string& method, const std::string& path, const json& body, httplib::Headers headers = {}) {
        headers.emplace("Content-Type", "application/json");
        return raw(method, path, std::move(headers), body.dump());
    }
    json identity() { return get("/health/identity").document(); }
    void stop() { process.close_input(); CHECK(process.wait() == 0); }
};

class Fixture {
public:
    Fixture() {
        static unsigned sequence = 0;
        root = fs::temp_directory_path() / ("mfq-server-native-test-" + std::to_string(std::random_device{}()) + "-" + std::to_string(++sequence));
        CHECK(fs::create_directory(root));
        data = root / "data";
    }
    ~Fixture() {
        for (auto& child : children_) {
            child.first->close_input();
            if (!child.second && child.first->running()) {
#ifndef _WIN32
                child.first->signal(SIGTERM);
#endif
            }
            if (child.second) try { child.first->wait(); } catch (...) {}
        }
        children_.clear();
        std::error_code ignored;
        fs::permissions(data, fs::perms::owner_all, fs::perm_options::add, ignored);
        fs::remove_all(root, ignored);
    }
    Process& launch(const fs::path& directory, std::vector<std::string> options = {},
            std::map<std::string, std::string> environment = {}, bool owned = false) {
        std::vector<std::string> arguments{"--port", "0", "--data-dir", directory.u8string()};
        if (owned) arguments.push_back("--exit-on-stdin-close");
        arguments.insert(arguments.end(), options.begin(), options.end());
        auto child = std::make_unique<Process>(executable, arguments, root / ("child-" + std::to_string(children_.size()) + ".log"), environment);
        auto& result = *child;
        children_.emplace_back(std::move(child), owned);
        return result;
    }
    Server start_at(const fs::path& directory, std::vector<std::string> options = {},
            std::map<std::string, std::string> environment = {}, bool owned = true) {
        auto& child = launch(directory, std::move(options), std::move(environment), owned);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (std::chrono::steady_clock::now() < deadline) {
            if (!child.running()) throw std::runtime_error("server exited during startup: " + child.output());
            const auto log = child.output();
            const auto end = log.find('\n');
            if (end != std::string::npos) {
                const auto event = json::parse(log.substr(0, end));
                Server server{child, event.at("host"), event.at("port")};
                CHECK(server.get("/health").status == 200);
                return server;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        throw std::runtime_error("server startup timed out: " + child.output());
    }
    Server start(std::vector<std::string> options = {}, std::map<std::string, std::string> environment = {}, bool owned = true) {
        return start_at(data, std::move(options), std::move(environment), owned);
    }
    fs::path root, data;
private:
    std::vector<std::pair<std::unique_ptr<Process>, bool>> children_;
};

httplib::Headers bearer(const std::string& token) { return {{"Authorization", "Bearer " + token}}; }
const httplib::Headers admin = bearer("example-root");
const std::map<std::string, std::string> root_environment{{"MFQ_SERVER_API_KEY", "example-root"}};

class PartialRequest {
public:
    explicit PartialRequest(int port) {
        socket_ = ::socket(AF_INET, SOCK_STREAM, 0);
        CHECK(socket_ != INVALID_SOCKET);
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_port = htons(static_cast<std::uint16_t>(port));
        inet_pton(AF_INET, "127.0.0.1", &address.sin_addr);
        CHECK(::connect(socket_, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0);
        const std::string request = "GET /health HTTP/1.1\r\n";
        CHECK(::send(socket_, request.data(), static_cast<int>(request.size()), 0) == static_cast<int>(request.size()));
    }
    ~PartialRequest() {
#ifdef _WIN32
        ::closesocket(socket_);
#else
        ::close(socket_);
#endif
    }
    PartialRequest(const PartialRequest&) = delete;
    PartialRequest& operator=(const PartialRequest&) = delete;
private:
    socket_t socket_ = INVALID_SOCKET;
};

void cli_and_lifetime() {
    Fixture f;
    write_file(f.root / "file", "not a directory");
    for (const auto& arguments : std::vector<std::vector<std::string>>{
        {"--unknown"}, {"--port"}, {"--port", "-1"}, {"--port", "+80"}, {"--port", "65536"},
        {"--port", "80junk"}, {"--port", "999999999999999999999999999999"},
        {"--host", "0.0.0.0"}, {"--host", "localhost"}, {"--host", "::"},
        {"--api-key-env", "9INVALID"}, {"--studio-instance-id", "00000000-0000-4000-8000-000000000001"},
        {"--studio-instance-id", "invalid", "--exit-on-stdin-close"},
        {"--web-root", (f.root / "missing").u8string()}, {"--data-dir", (f.root / "file").u8string()},
        {"--model-dir", (f.root / "file").u8string()}}) {
        auto& child = f.launch(f.data, arguments);
        CHECK(child.wait() != 0);
        CHECK(child.output().find("MFQ Server not started:") != std::string::npos);
    }
    CHECK(!fs::exists(f.data));
    auto& version = f.launch(f.data, {"--version"});
    CHECK(version.wait() == 0);
    auto& help = f.launch(f.data, {"--help"});
    CHECK(help.wait() == 0);
    CHECK(help.output().find("Usage: mfq-server") != std::string::npos);
    auto server = f.start({"--studio-instance-id", "ABCDEF01-0000-4000-8000-000000000001"});
    const auto identity = server.identity();
    CHECK(identity == server.identity());
    CHECK(identity.at("service") == "mfq-server");
    CHECK(identity.at("protocol_version") == "1.0");
    CHECK(identity.at("studio_protocol_version") == "1.0");
    CHECK(identity.at("instance_id") == "abcdef01-0000-4000-8000-000000000001");
    CHECK(identity.at("pid") == server.process.pid());
    CHECK(identity.at("launch_mode") == "studio");
    const auto version_output = version.output();
    CHECK(identity.at("version") == version_output.substr(0, version_output.find_first_of("\r\n")));
    CHECK(identity.at("workspace_id").get<std::string>().size() == 64);
    CHECK(identity.dump().find(f.data.u8string()) == std::string::npos);
    CHECK(server.get("/health").document() == json({{"status", "ok"}, {"service", "mfq-server"}, {"protocol_version", "1.0"}}));
    CHECK(server.get("/health").header("Content-Type").find("application/json") != std::string::npos);
    CHECK(server.get("/health").header("X-Content-Type-Options") == "nosniff");
    CHECK(server.get("/health", {{"X-MFQ-Instance-ID", "old-instance"}}).status == 409);
    CHECK(server.get("/health", {{"X-MFQ-Instance-ID", identity.at("instance_id")}}).status == 200);
    server.process.write_input("parent still alive\n");
    CHECK(server.get("/health").status == 200);
    server.stop();
    auto replacement = f.start();
    CHECK(replacement.identity().at("workspace_id") == identity.at("workspace_id"));
    CHECK(replacement.identity().at("instance_id") != identity.at("instance_id"));
    CHECK(replacement.identity().at("launch_mode") == "studio");
    replacement.stop();
    auto unicode = f.start_at(f.root / fs::u8path("数据 space"));
    CHECK(unicode.identity().at("workspace_id") != identity.at("workspace_id"));
    unicode.stop();
    {
        Fixture cli_fixture;
        auto cli = cli_fixture.start({}, {}, false);
        CHECK(cli.identity().at("launch_mode") == "cli");
        cli.process.close_input();
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        CHECK(cli.get("/health").status == 200);
    }
    const auto probe = ::socket(AF_INET6, SOCK_STREAM, 0);
    sockaddr_in6 address{};
    address.sin6_family = AF_INET6;
    inet_pton(AF_INET6, "::1", &address.sin6_addr);
    const bool ipv6_available = probe != INVALID_SOCKET && ::bind(probe, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0;
#ifdef _WIN32
    if (probe != INVALID_SOCKET) ::closesocket(probe);
#else
    if (probe != INVALID_SOCKET) ::close(probe);
#endif
    if (ipv6_available) {
        auto ipv6 = f.start({"--host", "::1"});
        ipv6.stop();
    } else std::cout << "SKIP IPv6 loopback unavailable\n";
#ifndef _WIN32
    for (const auto signal : {SIGTERM, SIGINT, SIGKILL}) {
        auto child = f.start();
        child.process.signal(signal);
        CHECK(child.process.wait() == (signal == SIGKILL ? -SIGKILL : 0));
        auto next = f.start();
        next.stop();
    }
#endif
}

void locks_and_ports() {
    Fixture f;
    auto server = f.start();
    auto& duplicate = f.launch(f.data);
    CHECK(duplicate.wait() != 0);
    CHECK(duplicate.output().find("already managed") != std::string::npos);
    auto& conflict = f.launch(f.root / "other", {"--port", std::to_string(server.port)});
    CHECK(conflict.wait() != 0);
    CHECK(conflict.output().find("cannot bind listener") != std::string::npos);
    auto other = f.start_at(f.root / "other");
    other.stop();
#ifdef _WIN32
    const auto* local = _wgetenv(L"LOCALAPPDATA");
    const auto lock_path = (local ? fs::path(local) : fs::path(_wgetenv(L"USERPROFILE")) / "AppData" / "Local") /
        "MFQ" / "run" / (server.identity().at("workspace_id").get<std::string>() + ".lock");
    int lock = -1;
    CHECK(_wsopen_s(&lock, lock_path.c_str(), _O_RDWR | _O_BINARY | _O_NOINHERIT, _SH_DENYNO, _S_IREAD | _S_IWRITE) == 0);
    CHECK(_locking(lock, _LK_NBLCK, 1) != 0);
    server.stop();
    CHECK(_locking(lock, _LK_NBLCK, 1) == 0);
    auto& externally_locked = f.launch(f.data);
    CHECK(externally_locked.wait() != 0);
    CHECK(externally_locked.output().find("already managed") != std::string::npos);
    CHECK(_locking(lock, _LK_UNLCK, 1) == 0);
    _close(lock);
#else
    fs::create_directory_symlink(f.data, f.root / "alias");
    auto& alias = f.launch(f.root / "alias");
    CHECK(alias.wait() != 0);
    CHECK(alias.output().find("already managed") != std::string::npos);
    const auto lock_path = fs::path("/tmp") / ("mfq-server-locks-" + std::to_string(getuid())) /
        (server.identity().at("workspace_id").get<std::string>() + ".lock");
    auto record = read_file(lock_path);
    CHECK(!record.empty());
    CHECK(json::parse(record.substr(1)) == server.identity());
    const int lock = ::open(lock_path.c_str(), O_RDWR | O_CLOEXEC);
    CHECK(lock >= 0);
    CHECK(::flock(lock, LOCK_EX | LOCK_NB) != 0);
    server.stop();
    CHECK(::flock(lock, LOCK_EX | LOCK_NB) == 0);
    auto& externally_locked = f.launch(f.data);
    CHECK(externally_locked.wait() != 0);
    CHECK(externally_locked.output().find("already managed") != std::string::npos);
    ::close(lock);
    auto removed = f.start();
    fs::remove_all(f.data);
    fs::create_directory(f.data);
    auto& deleted = f.launch(f.data);
    CHECK(deleted.wait() != 0);
    CHECK(deleted.output().find("already managed") != std::string::npos);
    removed.stop();
#endif
    auto retry = f.start();
    retry.stop();
}

void shutdown_with_pending_work() {
    {
        Fixture f;
        auto server = f.start({}, root_environment);
        Database database(f.data / "server.sqlite3");
        database.execute("BEGIN IMMEDIATE");
        auto pending = std::async(std::launch::async, [&] {
            return server.send("POST", "/api/v1/auth/keys", {{"name", "during-shutdown"}, {"scopes", {"jobs"}}}, admin);
        });
        CHECK(pending.wait_for(std::chrono::milliseconds(100)) == std::future_status::timeout);
        CHECK(server.get("/health").status == 200);
        server.process.close_input();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        CHECK(server.process.running());
        database.execute("COMMIT");
        const auto created = pending.get();
        CHECK(created.status == 201);
        CHECK(server.process.wait() == 0);
        CHECK(database.scalar("PRAGMA integrity_check") == "ok");
        auto restarted = f.start({}, root_environment);
        CHECK(restarted.get("/api/v1/jobs", bearer(created.document().at("token"))).status == 404);
    }
    {
        Fixture f;
        auto server = f.start();
        std::vector<std::unique_ptr<PartialRequest>> connections;
        for (int i = 0; i < 12; ++i) connections.push_back(std::make_unique<PartialRequest>(server.port));
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        server.process.close_input();
        CHECK(server.process.wait() == 0);
        auto restarted = f.start();
        CHECK(restarted.get("/health").status == 200);
    }
}

void static_files() {
    Fixture f;
    const auto web = f.root / "web";
    fs::create_directories(web / "assets");
    fs::create_directory(web / "guide");
    fs::create_directory(web / "api");
    const std::string index = "<!doctype html><title>MFQ Studio</title>";
    write_file(web / "index.html", index);
    write_file(web / "assets" / "app.js", "console.log('ok');");
    write_file(web / "guide" / "index.html", "guide");
    write_file(web / "api" / "secret", "not an API");
    write_file(web / "health", "not health");
    auto server = f.start({"--web-root", web.u8string() + "/"});
    for (const auto* path : {"/", "/chat", "/settings/profile", "/future/page?session=1"}) {
        const auto response = server.get(path, {{"Accept", "text/html"}});
        CHECK(response.status == 200);
        CHECK(response.body == index);
        CHECK(response.header("Content-Type").find("text/html") != std::string::npos);
        const auto head = server.raw("HEAD", path, {{"Accept", "text/html"}});
        CHECK(head.status == 200);
        CHECK(head.body.empty());
        CHECK(std::stoul(head.header("Content-Length")) == index.size());
    }
    CHECK(server.get("/assets/app.js").body == "console.log('ok');");
    CHECK(server.get("/guide/").body == "guide");
    CHECK(server.get("/health").document().at("service") == "mfq-server");
    for (const auto* path : {"/api/secret", "/v1/missing", "/health/missing", "/docs/missing",
            "/assets/missing", "/missing.js/", "/%2e%2e/secret.txt", "/%5csecret", "/%00secret"})
        CHECK(server.get(path, {{"Accept", "text/html"}}).status == 404);
    CHECK(server.get("/chat", {{"Accept", "*/*"}}).status == 404);
    CHECK(server.get("/chat", {{"Accept", "application/json"}}).status == 404);
    CHECK(server.get("/chat", {{"Accept", "text/html"}, {"Sec-Fetch-Dest", "script"}}).status == 404);
    CHECK(server.raw("POST", "/chat", {{"Accept", "text/html"}}).status != 200);
#ifndef _WIN32
    write_file(f.root / "secret.txt", "private content");
    fs::create_symlink(f.root / "secret.txt", web / "outside.txt");
    CHECK(server.get("/outside.txt").status == 403);
    fs::remove(web / "index.html");
    fs::create_symlink(f.root / "secret.txt", web / "index.html");
    CHECK(server.get("/chat", {{"Accept", "text/html"}}).status == 403);
    server.stop();
    auto& invalid = f.launch(f.data, {"--web-root", web.u8string()});
    CHECK(invalid.wait() != 0);
#else
    server.stop();
#endif
}

void policy_and_protocols() {
    Fixture f;
    auto server = f.start();
    CHECK(server.get("/api/v1/auth/policy").document() == json({{"enabled", false}, {"has_key", false}}));
    CHECK(server.send("PUT", "/api/v1/auth/policy", {{"enabled", true}}).status == 400);
    CHECK(server.send("PUT", "/api/v1/auth/policy", {{"enabled", true}, {"api_key", "example-root"}}).status == 200);
    CHECK(server.get("/api/v1/auth/policy").status == 401);
    for (const auto& key : std::vector<json>{"", "has space", "中文", std::string(513, 'x'), 123})
        CHECK(server.send("PUT", "/api/v1/auth/policy", {{"enabled", true}, {"api_key", key}}, admin).status == 400);
    CHECK(server.send("PUT", "/api/v1/auth/policy", {{"enabled", 1}}, admin).status == 400);
    CHECK(server.send("PUT", "/api/v1/auth/policy", {{"enabled", true}, {"unknown", 1}}, admin).status == 400);
    CHECK(server.raw("PUT", "/api/v1/auth/policy", admin, "{}").status == 415);
    auto content_headers = admin;
    content_headers.emplace("Content-Type", "application/json");
    CHECK(server.raw("PUT", "/api/v1/auth/policy", content_headers, "{").status == 400);
    for (const auto* origin : {"tauri://localhost", "http://tauri.localhost", "https://tauri.localhost", "http://127.0.0.1:5174"}) {
        const auto denied = server.get("/api/v1/auth/policy", {{"Origin", origin}});
        CHECK(denied.status == 401);
        CHECK(denied.header("Access-Control-Allow-Origin") == origin);
        CHECK(denied.header("WWW-Authenticate") == "Bearer");
        const auto preflight = server.raw("OPTIONS", "/api/v1/auth/policy", {{"Origin", origin},
            {"Access-Control-Request-Method", "PUT"}, {"Access-Control-Request-Headers", "Authorization, Content-Type, X-MFQ-Instance-ID"}});
        CHECK(preflight.status == 204);
        CHECK(preflight.header("Access-Control-Allow-Origin") == origin);
    }
    CHECK(server.get("/api/v1/auth/policy", {{"Origin", "https://untrusted.example"}}).header("Access-Control-Allow-Origin").empty());
    CHECK(server.raw("OPTIONS", "/api/v1/auth/policy", {{"Origin", "tauri://localhost"},
        {"Access-Control-Request-Method", "PUT"}, {"Access-Control-Request-Headers", "X-Untrusted"}}).status == 403);
    CHECK(server.raw("POST", "/v1/messages", {{"X-Api-Key", "wrong"}}).document().at("error").at("type") == "authentication_error");
    CHECK(server.raw("POST", "/v1/messages", {{"X-Api-Key", "example-root"}}).status == 404);
    const httplib::Headers upgrade{{"Upgrade", "websocket"}, {"Connection", "Upgrade"},
        {"Sec-WebSocket-Version", "13"}, {"Sec-WebSocket-Key", "dGhlIHNhbXBsZSBub25jZQ=="}};
    CHECK(server.get("/api/v1/realtime", upgrade).status == 401);
    CHECK(server.get("/api/v1/realtime?access_token=example-root", upgrade).status == 404);
    CHECK(server.get("/api/v1/realtime?access_token=example-root&server_instance=old", upgrade).status == 409);
#ifndef _WIN32
    const auto previous = read_file(f.data / "auth.json");
    fs::rename(f.data / "auth.json", f.data / "auth.saved");
    fs::create_directory(f.data / "auth.json");
    const auto failed = server.send("PUT", "/api/v1/auth/policy", {{"enabled", true}, {"api_key", "failed-key"}}, admin);
    fs::remove(f.data / "auth.json");
    fs::rename(f.data / "auth.saved", f.data / "auth.json");
    CHECK(failed.status == 500);
    CHECK(read_file(f.data / "auth.json") == previous);
    CHECK(server.get("/api/v1/auth/policy", admin).status == 200);
    CHECK(server.get("/api/v1/auth/policy", bearer("failed-key")).status == 401);
    CHECK((fs::status(f.data / "auth.json").permissions() & fs::perms::mask) == (fs::perms::owner_read | fs::perms::owner_write));
#endif
    CHECK(server.send("PUT", "/api/v1/auth/policy", {{"enabled", true}, {"api_key", "replacement"}}, admin).status == 200);
    CHECK(server.get("/api/v1/auth/policy", admin).status == 401);
    server.stop();
    const auto saved = read_file(f.data / "auth.json");
    CHECK(saved.find("example-root") == std::string::npos);
    CHECK(saved.find("replacement") == std::string::npos);
    CHECK(json::parse(saved).at("key_hash").get<std::string>().size() == 64);
    auto network_saved = f.start({"--host", "0.0.0.0"});
    CHECK(network_saved.get("/api/v1/auth/policy").status == 401);
    CHECK(network_saved.get("/api/v1/auth/policy", bearer("replacement")).status == 200);
    CHECK(network_saved.send("PUT", "/api/v1/auth/policy", {{"enabled", false}}, bearer("replacement")).status == 400);
    network_saved.stop();
    auto restarted = f.start({}, root_environment);
    CHECK(restarted.get("/api/v1/auth/policy", admin).status == 401);
    CHECK(restarted.send("PUT", "/api/v1/auth/policy", {{"enabled", false}}, bearer("replacement")).status == 200);
    restarted.stop();
    auto disabled = f.start();
    CHECK(disabled.get("/api/v1/auth/policy").document() == json({{"enabled", false}, {"has_key", true}}));
    disabled.stop();
    auto& network = f.launch(f.data, {"--host", "0.0.0.0"}, root_environment);
    CHECK(network.wait() != 0);
    CHECK(network.output().find("authentication is required") != std::string::npos);
}

void keys_and_network() {
    Fixture f;
    auto server = f.start({"--host", "0.0.0.0", "--api-key-env", "MFQ_TEST_ROOT"}, {{"MFQ_TEST_ROOT", "example-root"}});
    CHECK(server.get("/api/v1/auth/policy").status == 401);
    CHECK(server.send("PUT", "/api/v1/auth/policy", {{"enabled", false}}, admin).status == 400);
    auto issue = [&](const json& input) {
        const auto response = server.send("POST", "/api/v1/auth/keys", input, admin);
        CHECK(response.status == 201);
        return response.document();
    };
    auto key = issue({{"name", "jobs"}, {"scopes", {"jobs"}}});
    const auto token = key.at("token").get<std::string>(), id = key.at("key").at("id").get<std::string>();
    CHECK(token.rfind("mfq_", 0) == 0);
    CHECK(server.get("/api/v1/jobs", bearer(token)).status == 404);
    CHECK(server.get("/api/v1/sessions", bearer(token)).status == 403);
    CHECK(server.get("/api/v1/auth/keys", bearer(token)).status == 403);
    CHECK(server.send("POST", "/api/v1/auth/keys", {{"name", "jobs"}, {"scopes", {"jobs"}}}, admin).status == 409);
    for (const auto* role : {"viewer", "operator", "administrator"}) {
        const auto issued = issue({{"name", role}, {"role", role}});
        const auto credentials = bearer(issued.at("token"));
        CHECK(server.get("/api/v1/runtime/status", credentials).status == 404);
        CHECK(server.get("/api/v1/models/directories", credentials).status == (std::string(role) == "viewer" ? 403 : 200));
        CHECK(server.get("/api/v1/models/00000000000000000000000000000000/directory", credentials).status == (std::string(role) == "viewer" ? 403 : 404));
        CHECK(server.get("/api/v1/models", credentials).status == 200);
        CHECK(server.raw("HEAD", "/api/v1/models", credentials).status == 200);
        CHECK(server.raw("HEAD", "/api/v1/runtime/listener", credentials).status == 404);
        CHECK(server.get("/api/v1/auth/keys", credentials).status == (std::string(role) == "administrator" ? 200 : 403));
    }
    for (const auto& input : std::vector<json>{json::object(), {{"name", "x"}}, {{"name", "x"}, {"scopes", {"root"}}},
        {{"name", "x"}, {"scopes", {"jobs", "jobs"}}}, {{"name", "x"}, {"role", "root"}},
        {{"name", "x"}, {"role", "viewer"}, {"unknown", 1}},
        {{"name", "x"}, {"role", "viewer"}, {"expires_at", "2025-02-29T00:00:00Z"}},
        {{"name", "x"}, {"role", "viewer"}, {"expires_at", "2099-01-01"}},
        {{"name", "x"}, {"role", "viewer"}, {"expires_at", "2099-01-01T00:00:00+25:00"}}})
        CHECK(server.send("POST", "/api/v1/auth/keys", input, admin).status == 400);
    for (const auto expired : {true, false}) {
        const auto issued = issue({{"name", expired ? "expired" : "future"}, {"role", "viewer"},
            {"expires_at", expired ? "2024-02-29T00:00:00+00:00" : "2099-02-28T23:59:59.999999-08:00"}});
        CHECK(server.get("/v1/models", bearer(issued.at("token"))).status == (expired ? 401 : 404));
    }
    issue({{"name", "'; DROP TABLE sessions; --"}, {"scopes", {"inference"}}});
    const auto rotated = server.raw("POST", "/api/v1/auth/keys/" + id + "/rotate", admin);
    CHECK(rotated.status == 200);
    const auto replacement = rotated.document().at("token").get<std::string>();
    CHECK(replacement != token);
    CHECK(server.get("/api/v1/jobs", bearer(token)).status == 401);
    CHECK(server.get("/api/v1/jobs", bearer(replacement)).status == 404);
    const auto listed = server.get("/api/v1/auth/keys", admin).body;
    CHECK(listed.find("key_hash") == std::string::npos);
    CHECK(listed.find(replacement) == std::string::npos);
    for (const auto& file : fs::directory_iterator(f.data)) if (file.path().filename().u8string().rfind("server.sqlite3", 0) == 0) {
        CHECK(read_file(file.path()).find(token) == std::string::npos);
        CHECK(read_file(file.path()).find(replacement) == std::string::npos);
    }
    server.stop();
    auto restarted = f.start({}, root_environment);
    CHECK(restarted.get("/api/v1/jobs", bearer(replacement)).status == 404);
    CHECK(restarted.raw("POST", "/api/v1/auth/keys/" + id + "/revoke", admin).status == 200);
    CHECK(restarted.get("/api/v1/jobs", bearer(replacement)).status == 401);
    CHECK(restarted.raw("POST", "/api/v1/auth/keys/00000000-0000-4000-8000-000000000000/revoke", admin).status == 404);
    CHECK(restarted.raw("POST", "/api/v1/auth/keys/invalid/rotate", admin).status == 400);
    Database database(f.data / "server.sqlite3");
    CHECK(database.scalar("SELECT count(*) FROM sessions") == "0");
}

void model_catalog() {
    Fixture f;
    auto server = f.start({}, root_environment);
    CHECK(server.get("/api/v1/models").status == 401);
    CHECK(server.get("/api/v1/models", admin).document().at("data").empty());
    const auto roots = server.get("/api/v1/models/directories", admin).document();
    CHECK(roots.at("current_path").is_null());
    CHECK(roots.at("data").size() == 1);
    const auto root_id = roots.at("data")[0].at("id").get<std::string>();
    CHECK(root_id.size() == 32);
    const auto root = f.data / "models";
    const auto browse = server.get("/api/v1/models/directories?directory_id=" + root_id, admin).document();
    CHECK(browse.at("current_path") == fs::canonical(root).u8string());
    CHECK(!browse.at("can_open_directory").get<bool>());
    CHECK(!browse.at("parent_id").is_null());
    CHECK(server.get("/api/v1/models?refresh=invalid", admin).status == 400);
    for (const auto* query : {"directory_id=invalid", "directory_id=", "path=", "path=%00bad", "directory_id=00000000000000000000000000000000&path=/"})
        CHECK(server.get(std::string("/api/v1/models/directories?") + query, admin).status == 400);
    CHECK(server.get("/api/v1/models/directories?directory_id=00000000000000000000000000000000", admin).status == 404);
    write_hf(root / "alpha");
    write_mfq(root / "beta.MFQ");
    write_file(root / "broken.mfq", "broken");
    write_file(root / "split-00002-of-00003.mfq", "missing shards");
    CHECK(server.get("/api/v1/models", admin).document().at("data").empty());
    const auto artifacts = server.get("/api/v1/models?refresh=true", admin).document().at("data");
    CHECK(artifacts.size() == 4);
    CHECK(artifacts[0].at("name") == "alpha");
    CHECK(artifacts[0].at("format") == "hf");
    CHECK(artifacts[0].at("complete") == true);
    CHECK(artifacts[0].at("loadable") == false);
    CHECK(artifacts[0].at("architecture") == "qwen3_5-hf-full-mfq");
    CHECK(artifacts[0].at("tensor_count") == 1);
    CHECK(artifacts[0].at("dtypes") == json({"BF16"}));
    CHECK(artifacts[0].at("context_capacity") == 4096);
    CHECK(artifacts[0].at("total_bytes") == fs::file_size(root / "alpha" / "model.safetensors"));
    CHECK(artifacts[1].at("name") == "beta");
    CHECK(artifacts[1].at("complete") == true);
    CHECK(artifacts[1].at("tensor_count") == 1);
    CHECK(artifacts[1].at("record_count") == 2);
    CHECK(artifacts[1].at("context_capacity") == 8192);
    CHECK(artifacts[2].at("complete") == false);
    CHECK(artifacts[3].at("shard_count") == 3);
    CHECK(artifacts[3].at("missing_shards") == 2);
    CHECK(artifacts.dump().find(f.root.u8string()) == std::string::npos);
    const auto hf_id = artifacts[0].at("id").get<std::string>();
    const auto mfq_id = artifacts[1].at("id").get<std::string>();
    CHECK(server.get("/api/v1/models/" + hf_id + "/directory", admin).document().at("current_path") == (root / "alpha").u8string());
    CHECK(server.get("/api/v1/models/" + mfq_id + "/directory", admin).document().at("current_path") == root.u8string());
    CHECK(server.get("/api/v1/models/invalid/directory", admin).status == 400);
    const auto files = server.get("/api/v1/models/directories?directory_id=" + root_id, admin).document();
    CHECK(files.at("model_file_count") == 4);
    CHECK(files.at("files").size() == 3);
    CHECK(files.at("data").size() == 1);

    const auto external = f.root / fs::u8path("外部 models");
    write_hf(external / "gamma");
    const auto selected = server.get("/api/v1/models/directories?path=" + httplib::encode_uri_component(external.u8string()), admin).document();
    const auto source = json{{"directory_id", selected.at("current_id")}};
    const auto registered = server.send("POST", "/api/v1/models/directories/register", source, admin);
    CHECK(registered.status == 200);
    CHECK(registered.document().at("data").size() == 1);
    const auto gamma_id = registered.document().at("data")[0].at("id");
    const auto index = root / ".mfq-files.json";
    const auto saved = read_file(index);
    CHECK(json::parse(saved).at("version") == 1);
    CHECK(json::parse(saved).at("files") == json({(external / "gamma").u8string()}));
    CHECK(server.send("POST", "/api/v1/models/directories/register", source, admin).document().at("data").size() == 1);
    CHECK(server.get("/api/v1/models?refresh=1", admin).document().at("data").size() == 5);
    for (const auto& input : std::vector<json>{json::object(), json::array(), {{"path", ""}}, {{"path", 42}},
            {{"path", external.u8string()}, {"directory_id", root_id}}, {{"path", external.u8string()}, {"unexpected", true}},
            {{"path", external.u8string()}, {"directory_id", ""}}})
        CHECK(server.send("POST", "/api/v1/models/directories/register", input, admin).status == 400);
    CHECK(server.send("POST", "/api/v1/models/directories/register", {{"directory_id", "00000000000000000000000000000000"}}, admin).status == 404);
    fs::create_directory(f.root / "empty");
    CHECK(server.send("POST", "/api/v1/models/directories/register", {{"path", (f.root / "empty").u8string()}}, admin).status == 400);
    write_hf(f.root / "duplicate" / "alpha");
    CHECK(server.send("POST", "/api/v1/models/directories/register", {{"path", (f.root / "duplicate").u8string()}}, admin).status == 409);
    CHECK(read_file(index) == saved);
    CHECK(server.get("/api/v1/models", admin).document().at("data").size() == 5);
    auto deep = f.root / "deep";
    for (int i = 0; i < 34; ++i) deep /= "child";
    fs::create_directories(deep);
    write_mfq(deep / "deep.mfq");
    CHECK(server.send("POST", "/api/v1/models/directories/register", {{"path", (f.root / "deep").u8string()}}, admin).status == 413);
    CHECK(read_file(index) == saved);
    json large{{"version", 1}, {"files", json::array()}};
    constexpr std::size_t limit = 4 * 1024 * 1024;
    std::size_t size = large.dump().size();
    const auto prefix = f.root.generic_u8string() + "/missing-";
    for (unsigned i = 0; ; ++i) {
        const auto value = prefix + std::to_string(i) + std::string(512, 'x');
        const auto growth = json(value).dump().size() + (large["files"].empty() ? 0 : 1);
        if (size + growth > limit - 1024) break;
        large["files"].push_back(value);
        size += growth;
    }
    large["files"].push_back(prefix + std::string(limit - 1 - size - json(prefix).dump().size() - 1, 'x'));
    const auto large_saved = large.dump();
    CHECK(large_saved.size() == limit - 1);
    CHECK(large["files"].size() < 10000);
    write_file(index, large_saved);
    CHECK(server.send("POST", "/api/v1/models/directories/register", source, admin).status == 413);
    CHECK(read_file(index) == large_saved);
    write_file(index, saved);
    write_file(index, "{invalid");
    CHECK(server.get("/api/v1/models?refresh=true", admin).status == 500);
    CHECK(read_file(index) == "{invalid");
    write_file(index, saved);
    server.stop();
    auto restarted = f.start({}, root_environment);
    const auto recovered = restarted.get("/api/v1/models", admin).document().at("data");
    CHECK(recovered.size() == 5);
    CHECK(recovered[0].at("id") == hf_id);
    CHECK(recovered[1].at("id") == mfq_id);
    CHECK(recovered[3].at("id") == gamma_id);
    CHECK(restarted.get("/api/v1/models/directories?directory_id=" + root_id, admin).status == 404);
    fs::remove(external / "gamma" / "model.safetensors");
    CHECK(restarted.get("/api/v1/models?refresh=true", admin).document().at("data").size() == 4);
}

void model_roots() {
    Fixture f;
    const auto first = f.root / "first", second = f.root / "second", environment = f.root / "environment";
    write_hf(first / "one");
    write_hf(second / "two");
    write_hf(environment / "ignored");
    auto explicit_roots = f.start({"--model-dir", first.u8string(), "--model-dir", second.u8string()}, {{"MFQ_SERVER_MODEL_DIRS", environment.u8string()}});
    CHECK(explicit_roots.get("/api/v1/models").document().at("data").size() == 2);
    CHECK(explicit_roots.get("/api/v1/models/directories").document().at("data").size() == 2);
    explicit_roots.stop();
    auto from_environment = f.start({}, {{"MFQ_SERVER_MODEL_DIRS", environment.u8string()}});
    CHECK(from_environment.get("/api/v1/models").document().at("data")[0].at("name") == "ignored");
    from_environment.stop();
#ifndef _WIN32
    fs::create_directory_symlink(first, f.root / "alias");
    auto alias = f.start({"--model-dir", first.u8string(), "--model-dir", (f.root / "alias").u8string()});
    CHECK(alias.get("/api/v1/models/directories").document().at("data").size() == 1);
    alias.stop();
#endif
    write_mfq(environment / "ignored.mfq");
    auto dual_format = f.start({"--model-dir", environment.u8string()});
    const auto artifacts = dual_format.get("/api/v1/models").document().at("data");
    CHECK(artifacts.size() == 1);
    CHECK(artifacts[0].at("format") == "mfq");
}

void databases_and_migration() {
    Fixture f;
    fs::create_directory(f.data);
    const auto database = f.data / "server.sqlite3";
    {
        Database db(database);
        db.execute("CREATE TABLE schema_meta(key TEXT PRIMARY KEY,value TEXT NOT NULL); INSERT INTO schema_meta VALUES('schema_version','99');");
    }
    auto before = read_file(database);
    auto& unsupported = f.launch(f.data);
    CHECK(unsupported.wait() != 0);
    CHECK(unsupported.output().find("unsupported database schema") != std::string::npos);
    CHECK(read_file(database) == before);
    write_file(database, "corrupt data must not be overwritten");
    before = read_file(database);
    CHECK(f.launch(f.data).wait() != 0);
    CHECK(read_file(database) == before);
    fs::remove(database);
    {
        Database db(database);
        db.execute("CREATE TABLE existing_data(value TEXT); INSERT INTO existing_data VALUES('must survive');");
    }
    before = read_file(database);
    CHECK(f.launch(f.data).wait() != 0);
    CHECK(read_file(database) == before);
    fs::remove(database);
    for (const auto& policy : std::vector<json>{{{"enabled", true}, {"key_hash", ""}},
            {{"enabled", "yes"}, {"key_hash", std::string(64, 'a')}}, {{"enabled", true}, {"key_hash", std::string(64, 'A')}}}) {
        write_file(f.data / "auth.json", policy.dump());
        CHECK(f.launch(f.data, {}, root_environment).wait() != 0);
        CHECK(read_file(f.data / "auth.json") == policy.dump());
        CHECK(!fs::exists(database));
    }
    fs::remove(f.data / "auth.json");
    const auto explicit_database = f.root / "custom.sqlite3";
    auto explicit_server = f.start({"--db", explicit_database.u8string()});
    CHECK(fs::exists(explicit_database));
    CHECK(!fs::exists(database));
    explicit_server.stop();
    {
        Database db(explicit_database);
        db.execute("CREATE TRIGGER reject_version_rewrite BEFORE INSERT ON schema_meta "
            "WHEN NEW.key='schema_version' BEGIN SELECT RAISE(ABORT,'current version must not be rewritten'); END;");
        auto current = f.start({"--db", explicit_database.u8string()});
        CHECK(current.get("/api/v1/auth/keys").status == 200);
        current.stop();
        db.execute("DROP TRIGGER reject_version_rewrite;");
        db.execute("ALTER TABLE api_keys DROP COLUMN role;");
        auto& incomplete = f.launch(f.data, {"--db", explicit_database.u8string()});
        CHECK(incomplete.wait() != 0);
        CHECK(db.scalar("SELECT value FROM schema_meta WHERE key='schema_version'") == "19");
        CHECK(db.scalar("SELECT count(*) FROM pragma_table_info('api_keys') WHERE name='role'") == "0");
    }
    for (const auto version : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18}) {
        const auto data = f.root / ("version-" + std::to_string(version));
        fs::create_directory(data);
        const auto legacy = data / "mfq-server.sqlite3";
        {
            Database db(legacy);
            db.execute(mfq::server::schema_sql);
            db.execute("INSERT INTO schema_meta VALUES('schema_version','" + std::to_string(version) + "');"
                "ALTER TABLE api_keys DROP COLUMN role; ALTER TABLE responses DROP COLUMN settings_json;"
                "ALTER TABLE responses DROP COLUMN performance_json; ALTER TABLE responses DROP COLUMN output_message_id;"
                "DROP TABLE generation_presets;"
                "CREATE TABLE generation_presets(id TEXT PRIMARY KEY,name TEXT NOT NULL UNIQUE,model TEXT NOT NULL,mode TEXT NOT NULL,"
                    "settings_json TEXT NOT NULL,context_size INTEGER CHECK(context_size>=512),created_at TEXT NOT NULL,updated_at TEXT NOT NULL);"
                "ALTER TABLE jobs DROP COLUMN progress_data_json;"
                "ALTER TABLE jobs DROP COLUMN archived_at;"
                "INSERT INTO generation_presets(id,name,model,mode,settings_json,created_at,updated_at) VALUES('preset','old','model','chat','{}','2020','2020');"
                "INSERT INTO sessions(id,model,mode,state,revision,created_at,updated_at,metadata_json) VALUES('session','model','chat','open',7,'2020','2020','{\"keep\":1}');"
                "INSERT INTO messages(id,role,parts_json,created_at) VALUES('message','user','[]','2020');"
                "INSERT INTO session_messages VALUES('session',1,'message');"
                "INSERT INTO responses(id,session_id,request_id,request_fingerprint,status,input_message_id,output_json,created_at) "
                    "VALUES('response','session','request','fingerprint','completed','message','{\"text\":\"history\"}','2020');"
                "INSERT INTO jobs(id,kind,status,payload_json,progress,cancel_requested,created_at,updated_at) VALUES('job','download','completed','{}',1,0,'2020','2020');"
                "INSERT INTO api_keys VALUES('00000000-0000-4000-8000-000000000001','legacy',"
                "'762c08fc17a1cc5f00d248f8b50f2f2f4d17ff2934ac31e64deacb3f5bb3f2ec','old','[\"admin\"]',NULL,NULL,'2020-01-01T00:00:00+00:00',NULL);"
                "CREATE TABLE preserve_me(value TEXT); INSERT INTO preserve_me VALUES('existing data');");
        }
        auto server = f.start_at(data, {}, root_environment);
        CHECK(!fs::exists(data / "server.sqlite3"));
        CHECK(server.get("/api/v1/auth/keys", bearer("old-key")).status == 200);
        server.stop();
        Database db(legacy);
        CHECK(db.scalar("SELECT value FROM schema_meta WHERE key='schema_version'") == "19");
        CHECK(db.scalar("SELECT name||'|'||metadata_json FROM generation_presets WHERE id='preset'") == "old|{}");
        db.execute("INSERT INTO generation_presets(id,name,settings_json,metadata_json,created_at,updated_at) VALUES('nullable','new','{}','{}','2020','2020');");
        CHECK(db.scalar("SELECT model IS NULL AND mode IS NULL FROM generation_presets WHERE id='nullable'") == "1");
        CHECK(db.scalar("SELECT revision||'|'||metadata_json FROM sessions WHERE id='session'") == "7|{\"keep\":1}");
        CHECK(db.scalar("SELECT message_id FROM session_messages WHERE session_id='session' AND ordinal=1") == "message");
        CHECK(db.scalar("SELECT output_json FROM responses WHERE id='response'") == "{\"text\":\"history\"}");
        CHECK(db.scalar("SELECT performance_json IS NULL AND output_message_id IS NULL AND settings_json IS NULL FROM responses WHERE id='response'") == "1");
        CHECK(db.scalar("SELECT status||'|'||progress_data_json FROM jobs WHERE id='job'") == "completed|{}");
        CHECK(db.scalar("SELECT value FROM preserve_me") == "existing data");
        if (version == 18) {
            db.execute("UPDATE schema_meta SET value='18' WHERE key='schema_version'; ALTER TABLE api_keys DROP COLUMN role;"
                "CREATE TABLE generation_presets_nullable(dummy TEXT);");
            CHECK(f.launch(data).wait() != 0);
            CHECK(db.scalar("SELECT value FROM schema_meta WHERE key='schema_version'") == "18");
            CHECK(db.scalar("SELECT count(*) FROM pragma_table_info('api_keys') WHERE name='role'") == "0");
            CHECK(db.scalar("SELECT name FROM generation_presets WHERE id='preset'") == "old");
        }
    }
}

} // namespace

int run_tests(const fs::path& server, bool models_only) {
    executable = fs::absolute(server);
    using Tests = std::vector<std::pair<const char*, std::function<void()>>>;
    const Tests tests = models_only ? Tests{{"model catalog", model_catalog}, {"model roots", model_roots}} : Tests{
        {"CLI and lifetime", cli_and_lifetime}, {"locks and ports", locks_and_ports}, {"static files", static_files},
        {"policy and protocols", policy_and_protocols}, {"keys and network", keys_and_network},
        {"shutdown with pending work", shutdown_with_pending_work}, {"database migrations", databases_and_migration}};
    int failures = 0;
    for (const auto& test : tests) {
        try { test.second(); std::cout << "PASS " << test.first << '\n'; }
        catch (const std::exception& error) { ++failures; std::cerr << "FAIL " << test.first << ": " << error.what() << '\n'; }
    }
    std::cout << checks << " checks, " << failures << " failed scenarios\n";
    return failures ? 1 : 0;
}

#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
    if (argc != 2 && !(argc == 3 && std::wstring(argv[2]) == L"--models")) return 2;
    return run_tests(argv[1], argc == 3);
}
#else
int main(int argc, char** argv) {
    if (argc != 2 && !(argc == 3 && std::string(argv[2]) == "--models")) return 2;
    return run_tests(fs::u8path(argv[1]), argc == 3);
}
#endif
