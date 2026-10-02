#include "runtime_config.h"

#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

namespace mfq::engine {

std::uint64_t environment_uint64(
        const char* name, std::uint64_t fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') return fallback;
    std::uint64_t result = 0;
    const auto length = std::char_traits<char>::length(value);
    const auto [end, error] = std::from_chars(value, value + length, result);
    if (error != std::errc{} || end != value + length) {
        throw std::runtime_error(std::string("invalid ") + name);
    }
    return result;
}

bool environment_enabled(const char* name, bool fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') return fallback;
    int parsed = 0;
    const auto end = value + std::char_traits<char>::length(value);
    const auto [last, error] = std::from_chars(value, end, parsed);
    if (error != std::errc{} || last != end)
        throw std::runtime_error(std::string("invalid ") + name);
    return parsed != 0;
}

static std::filesystem::path default_prefix_cache_directory() {
    if (const char* configured =
            std::getenv("MFQ_RUNTIME_PREFIX_CACHE_DIR")) {
        if (configured[0] != '\0') return configured;
    }
#ifdef _WIN32
    if (const char* local = std::getenv("LOCALAPPDATA")) {
        if (local[0] != '\0') {
            return std::filesystem::path(local) /
                "TyloQuant" / "MFQ" / "prefix-cache";
        }
    }
#else
    if (const char* xdg = std::getenv("XDG_CACHE_HOME")) {
        if (xdg[0] != '\0') {
            return std::filesystem::path(xdg) /
                "tyloquant" / "mfq" / "prefix-cache";
        }
    }
    if (const char* home = std::getenv("HOME")) {
        if (home[0] != '\0') {
            return std::filesystem::path(home) / ".cache" /
                "tyloquant" / "mfq" / "prefix-cache";
        }
    }
#endif
    return std::filesystem::temp_directory_path() /
        "tyloquant-mfq-prefix-cache";
}

static std::size_t checked_size(std::uint64_t value, const char* name) {
    if (value > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error(std::string(name) + " exceeds size_t");
    }
    return static_cast<std::size_t>(value);
}

RuntimeConfig resolve_runtime_config(std::int64_t prefill_chunk_size) {
    if (prefill_chunk_size <= 0)
        throw std::invalid_argument("prefill chunk size must be positive");
    RuntimeConfig config;
    config.generation.prefill_chunk_size = prefill_chunk_size;
    auto& sessions = config.session_cache;
    sessions.snapshots.max_sessions = checked_size(environment_uint64(
        "MFQ_RUNTIME_MAX_KV_SESSIONS", sessions.snapshots.max_sessions),
        "MFQ_RUNTIME_MAX_KV_SESSIONS");
    sessions.snapshots.max_snapshots_per_session = checked_size(
        environment_uint64(
            "MFQ_RUNTIME_MAX_KV_SNAPSHOTS_PER_SESSION",
            sessions.snapshots.max_snapshots_per_session),
        "MFQ_RUNTIME_MAX_KV_SNAPSHOTS_PER_SESSION");
    sessions.snapshots.max_bytes = checked_size(environment_uint64(
        "MFQ_RUNTIME_KV_SESSION_BYTES", sessions.snapshots.max_bytes),
        "MFQ_RUNTIME_KV_SESSION_BYTES");
    sessions.trace = environment_enabled(
        "MFQ_RUNTIME_TRACE_SESSION_CACHE", false);

    auto& prefix = config.prefix_cache;
    prefix.directory = default_prefix_cache_directory();
    prefix.enabled = !environment_enabled(
        "MFQ_RUNTIME_DISABLE_PREFIX_CACHE", false);
    prefix.block_tokens = environment_uint64(
        "MFQ_RUNTIME_PREFIX_CACHE_BLOCK_TOKENS", prefix.block_tokens);
    if (prefix.block_tokens == 0 || prefix.block_tokens > 65536) {
        throw std::runtime_error(
            "MFQ_RUNTIME_PREFIX_CACHE_BLOCK_TOKENS must be in [1, 65536]");
    }
    prefix.disk_bytes = environment_uint64(
        "MFQ_RUNTIME_PREFIX_CACHE_DISK_BYTES", prefix.disk_bytes);
    prefix.hot_bytes = environment_uint64(
        "MFQ_RUNTIME_PREFIX_CACHE_HOT_BYTES", prefix.hot_bytes);
    prefix.pending_writes = checked_size(environment_uint64(
        "MFQ_RUNTIME_PREFIX_CACHE_PENDING_WRITES", prefix.pending_writes),
        "MFQ_RUNTIME_PREFIX_CACHE_PENDING_WRITES");
    prefix.pending_bytes = environment_uint64(
        "MFQ_RUNTIME_PREFIX_CACHE_PENDING_BYTES", prefix.pending_bytes);
    return config;
}

ContinuousBatchConfig resolve_batch_config(int capacity, std::int64_t prefill_chunk_size) {
    if (capacity < 0)
        throw std::invalid_argument("continuous batching capacity must be non-negative");
    if (prefill_chunk_size <= 0)
        throw std::invalid_argument("prefill chunk size must be positive");
    ContinuousBatchConfig config;
    config.max_sequences = static_cast<std::size_t>(capacity);
    config.prefill_token_budget = static_cast<std::int64_t>(std::min<std::uint64_t>(
        environment_uint64("MFQ_CONTINUOUS_BATCH_PREFILL_TOKEN_BUDGET", prefill_chunk_size),
        std::numeric_limits<std::int64_t>::max()));
    if (config.prefill_token_budget < 1)
        throw std::invalid_argument("continuous batching prefill token budget must be positive");
    return config;
}

} // namespace mfq::engine
