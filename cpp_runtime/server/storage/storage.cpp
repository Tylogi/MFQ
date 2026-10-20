#include "storage.h"
#include "api/errors.h"
#include "schema.h"

#include <sqlite3.h>
#include <stdexcept>
#include <utility>

namespace mfq::server {
namespace {

using json = nlohmann::json;
constexpr int schema_version = 19;
constexpr const char* key_columns = "id,name,prefix,scopes_json,role,expires_at,revoked_at,created_at,last_used_at";

void check(sqlite3* database, int result) {
    if (result == SQLITE_OK || result == SQLITE_ROW || result == SQLITE_DONE) return;
    if ((result & 255) == SQLITE_CONSTRAINT)
        throw ApiError(409, "api_key_conflict", "API key name or credential already exists");
    throw std::runtime_error(std::string("SQLite: ") + sqlite3_errmsg(database));
}

void execute(sqlite3* database, const char* sql) {
    check(database, sqlite3_exec(database, sql, nullptr, nullptr, nullptr));
}

class Statement {
public:
    Statement(sqlite3* database, const char* sql) : database_(database) {
        check(database_, sqlite3_prepare_v2(database_, sql, -1, &statement_, nullptr));
    }
    ~Statement() { sqlite3_finalize(statement_); }
    Statement(const Statement&) = delete;
    Statement& operator=(const Statement&) = delete;
    void bind(int index, const json& value) {
        if (value.is_null()) check(database_, sqlite3_bind_null(statement_, index));
        else {
            const auto text = value.get<std::string>();
            check(database_, sqlite3_bind_text(statement_, index, text.data(), static_cast<int>(text.size()), SQLITE_TRANSIENT));
        }
    }
    bool step() {
        const auto result = sqlite3_step(statement_);
        check(database_, result);
        return result == SQLITE_ROW;
    }
    json column(int index) const {
        const auto* text = sqlite3_column_text(statement_, index);
        return text ? json(std::string(reinterpret_cast<const char*>(text), sqlite3_column_bytes(statement_, index))) : json(nullptr);
    }

private:
    sqlite3* database_;
    sqlite3_stmt* statement_ = nullptr;
};

bool table_exists(sqlite3* database, const char* name) {
    Statement query(database, "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?");
    query.bind(1, name);
    return query.step();
}

bool column_exists(sqlite3* database, const char* table, const char* column) {
    const auto sql = "PRAGMA table_info(" + std::string(table) + ")";
    Statement query(database, sql.c_str());
    while (query.step()) if (query.column(1) == column) return true;
    return false;
}

void validate_key_schema(sqlite3* database) {
    const auto sql = "SELECT " + std::string(key_columns) + ",key_hash FROM api_keys LIMIT 0";
    Statement keys(database, sql.c_str());
    keys.step();
}

void initialize(sqlite3* database) {
    int previous = 0;
    if (table_exists(database, "schema_meta")) {
        Statement query(database, "SELECT value FROM schema_meta WHERE key='schema_version'");
        if (!query.step()) throw std::runtime_error("database has no schema version");
        const auto value = query.column(0).get<std::string>();
        for (int version = 1; version <= schema_version; ++version)
            if (value == std::to_string(version)) previous = version;
        if (!previous) throw std::runtime_error("unsupported database schema " + value);
    } else {
        Statement query(database, "SELECT 1 FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' LIMIT 1");
        if (query.step()) throw std::runtime_error("unrecognized database without schema_meta");
    }
    if (previous == schema_version) validate_key_schema(database);
    {
        Statement journal(database, "PRAGMA journal_mode=WAL");
        if (!journal.step() || journal.column(0) != "wal") throw std::runtime_error("SQLite refused WAL mode");
    }
    if (previous == schema_version) return;
    execute(database, "BEGIN IMMEDIATE");
    try {
        execute(database, schema_sql);
        if (previous && previous < schema_version) {
            for (const auto* column : {"performance_json", "output_message_id", "settings_json"}) {
                if (!column_exists(database, "responses", column))
                    execute(database, ("ALTER TABLE responses ADD COLUMN " + std::string(column) + " TEXT").c_str());
            }
            if (!column_exists(database, "api_keys", "role"))
                execute(database, "ALTER TABLE api_keys ADD COLUMN role TEXT");
            if (!column_exists(database, "generation_presets", "metadata_json"))
                execute(database, "ALTER TABLE generation_presets ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}'");
            if (!column_exists(database, "jobs", "archived_at"))
                execute(database, "ALTER TABLE jobs ADD COLUMN archived_at TEXT");
            if (!column_exists(database, "jobs", "progress_data_json"))
                execute(database, "ALTER TABLE jobs ADD COLUMN progress_data_json TEXT NOT NULL DEFAULT '{}'");
            execute(database, R"SQL(
                CREATE TABLE generation_presets_nullable (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE, model TEXT, mode TEXT,
                    settings_json TEXT NOT NULL, context_size INTEGER CHECK(context_size >= 512),
                    metadata_json TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                INSERT INTO generation_presets_nullable
                    SELECT id,name,model,mode,settings_json,context_size,metadata_json,created_at,updated_at
                    FROM generation_presets;
                DROP TABLE generation_presets;
                ALTER TABLE generation_presets_nullable RENAME TO generation_presets;
            )SQL");
        }
        validate_key_schema(database);
        Statement version(database, "INSERT OR REPLACE INTO schema_meta(key,value) VALUES('schema_version',?)");
        version.bind(1, std::to_string(schema_version));
        version.step();
        execute(database, "COMMIT");
    } catch (...) {
        sqlite3_exec(database, "ROLLBACK", nullptr, nullptr, nullptr);
        throw;
    }
}

json key_row(const Statement& query) {
    return {{"id", query.column(0)}, {"name", query.column(1)}, {"prefix", query.column(2)},
        {"scopes", json::parse(query.column(3).get<std::string>())}, {"role", query.column(4)},
        {"expires_at", query.column(5)}, {"revoked_at", query.column(6)},
        {"created_at", query.column(7)}, {"last_used_at", query.column(8)}};
}

json get_key(sqlite3* database, const std::string& id) {
    const auto sql = "SELECT " + std::string(key_columns) + " FROM api_keys WHERE id=?";
    Statement query(database, sql.c_str());
    query.bind(1, id);
    if (!query.step()) throw ApiError(404, "api_key_not_found", "API key not found");
    return key_row(query);
}

} // namespace

Storage::Storage(const std::filesystem::path& path) {
    std::filesystem::create_directories(path.parent_path());
    const auto result = sqlite3_open_v2(path.u8string().c_str(), &database_,
        SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX, nullptr);
    try {
        check(database_, result);
        check(database_, sqlite3_busy_timeout(database_, 5000));
        execute(database_, "PRAGMA foreign_keys=ON");
        initialize(database_);
    } catch (...) {
        sqlite3_close(database_);
        throw;
    }
}

Storage::~Storage() { sqlite3_close(database_); }

json Storage::create_key(const json& request, const std::string& id, const std::string& hash, const std::string& prefix) {
    std::lock_guard<std::mutex> guard(mutex_);
    {
        Statement insert(database_, "INSERT INTO api_keys(id,name,key_hash,prefix,scopes_json,role,expires_at,created_at) "
            "VALUES(?,?,?,?,?,?,?,strftime('%Y-%m-%dT%H:%M:%f+00:00','now'))");
        insert.bind(1, id);
        insert.bind(2, request.at("name"));
        insert.bind(3, hash);
        insert.bind(4, prefix);
        insert.bind(5, request.at("scopes").dump());
        insert.bind(6, request.at("role"));
        insert.bind(7, request.at("expires_at"));
        insert.step();
    }
    return get_key(database_, id);
}

json Storage::list_keys() {
    std::lock_guard<std::mutex> guard(mutex_);
    const auto sql = "SELECT " + std::string(key_columns) + " FROM api_keys ORDER BY created_at DESC,id";
    Statement query(database_, sql.c_str());
    json keys = json::array();
    while (query.step()) keys.push_back(key_row(query));
    return {{"data", std::move(keys)}};
}

std::optional<json> Storage::authenticate_key(const std::string& hash) {
    std::lock_guard<std::mutex> guard(mutex_);
    Statement update(database_, "UPDATE api_keys SET last_used_at=strftime('%Y-%m-%dT%H:%M:%f+00:00','now') "
        "WHERE key_hash=? AND revoked_at IS NULL AND (expires_at IS NULL OR julianday(expires_at)>julianday('now'))");
    update.bind(1, hash);
    update.step();
    if (!sqlite3_changes(database_)) return std::nullopt;
    const auto sql = "SELECT " + std::string(key_columns) + " FROM api_keys WHERE key_hash=?";
    Statement query(database_, sql.c_str());
    query.bind(1, hash);
    if (!query.step()) return std::nullopt;
    return key_row(query);
}

json Storage::revoke_key(const std::string& id) {
    std::lock_guard<std::mutex> guard(mutex_);
    Statement update(database_, "UPDATE api_keys SET revoked_at=strftime('%Y-%m-%dT%H:%M:%f+00:00','now') WHERE id=?");
    update.bind(1, id);
    update.step();
    if (!sqlite3_changes(database_)) throw ApiError(404, "api_key_not_found", "API key not found");
    return get_key(database_, id);
}

json Storage::rotate_key(const std::string& id, const std::string& hash, const std::string& prefix) {
    std::lock_guard<std::mutex> guard(mutex_);
    Statement update(database_, "UPDATE api_keys SET key_hash=?,prefix=?,revoked_at=NULL,last_used_at=NULL WHERE id=?");
    update.bind(1, hash);
    update.bind(2, prefix);
    update.bind(3, id);
    update.step();
    if (!sqlite3_changes(database_)) throw ApiError(404, "api_key_not_found", "API key not found");
    return get_key(database_, id);
}

} // namespace mfq::server
