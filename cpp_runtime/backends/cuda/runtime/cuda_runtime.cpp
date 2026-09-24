#include "cuda_runtime.h"
#include "moe_expert_cache.h"
#include "cuda_execution.h"
#include "cuda_sampling.h"
#include "decode_graph.h"
#include "quant_linear.h"
#include "../models/qwen35/qwen35_linear_attention.h"
#include "../models/registry.h"
#include "mfq_tensor_backend.h"
#include "prepared_prompt.h"
#include "mfq/kernels/cuda/deepseek_v4_attention.h"
#include "mfq/kernels/cuda/deepseek_v4_hc.h"
#include "mfq/kernels/cuda/deepseek_v41.h"
#include "mfq/kernels/cuda/fp8_sq.h"
#include "mfq/kernels/cuda/mxfp4_sq.h"
#include "cuda_model_plan.h"
#include "mfq_cuda_mtp.h"
#include "mfq_cuda_paged_kv.h"
#include "mfq_cuda_ops.h"
#include "glm5_next/model.h"
#include "models/deepseek_v41.h"
#include "qwen4_exp/model.h"
#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>

#ifdef MFQ_HAVE_NCCL
#include <nccl.h>
#endif

#include "transport.h"
#include "mfq_format_compat.h"
#include "grid_vision.h"
#include "mfq_paged_prefix_cache.h"
#include "moe_cache_transfer.h"
#include "moe_cache_policy.h"
#include "moe_cache_profile.h"
#include "mfe_expert_store.h"
#include "tensor_parallel.h"
#include "nvq_codebooks.generated.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cctype>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <list>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(__GLIBC__)
#include <malloc.h>
#endif

#if defined(__x86_64__) && defined(__GNUC__)
#include <immintrin.h>
#define MFQ_CPU_X86_GNU 1
#endif

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif

#ifndef MFQ_CUDA_CHECK
#define MFQ_CUDA_CHECK(expr) do { \
    cudaError_t err__ = (expr); \
    if (err__ != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err__)); \
    } \
} while (0)
#endif

#ifdef MFQ_HAVE_NCCL
#define MFQ_NCCL_CHECK(expr) do { \
    ncclResult_t err__ = (expr); \
    if (err__ != ncclSuccess) { \
        throw std::runtime_error(std::string("NCCL error: ") + ncclGetErrorString(err__)); \
    } \
} while (0)
#endif

#include "causal_lm.h"
#include "runtime_components.h"

#include "runner.h"
#include "generation.h"
using namespace mfq::cuda::internal;
template <typename Model>
static mfq_tensor_backend::Tensor sample_token(
    Model& model,
    mfq_tensor_backend::Tensor ids,
    mfq::cuda::Sampler& sampler,
    mfq_tensor_backend::Tensor counts,
    const MfqTokenConstraintPtr & token_constraint,
    cudaEvent_t prefill_finished = nullptr)
{
    if (sampler.greedy() && !sampler.has_penalties() && !token_constraint) {
        auto next = model.next_token(ids);
        if (prefill_finished != nullptr) {
            MFQ_CUDA_CHECK(cudaEventRecord(
                prefill_finished, mfq_get_current_cuda_stream()));
        }
        return next;
    }

    auto logits = model.last_logits(ids).contiguous().view({1, -1});
    if (prefill_finished != nullptr) {
        MFQ_CUDA_CHECK(cudaEventRecord(
            prefill_finished, mfq_get_current_cuda_stream()));
    }
    return mfq::cuda::sample_logits(
        sampler, std::move(logits), counts, token_constraint);
}

template <typename Model>
static mfq_tensor_backend::Tensor prefill_tail(
    Model& model,
    mfq_tensor_backend::Tensor ids,
    int64_t chunk_size) {
    MFQ_RUNTIME_CHECK(
        chunk_size > 0,
        "runtime prefill chunk size must be positive");
    MFQ_RUNTIME_CHECK(
        ids.dim() == 2 && ids.size(0) == 1 && ids.size(1) > 0,
        "runtime prefill IDs must have shape [1, tokens]");
    int64_t offset = 0;
    while (ids.size(1) - offset > chunk_size) {
        (void)model.hidden_forward(
            ids.narrow(1, offset, chunk_size).contiguous());
        offset += chunk_size;
    }
    return offset == 0
        ? ids
        : ids.narrow(1, offset, ids.size(1) - offset).contiguous();
}

template <typename Model>
static mfq_tensor_backend::Tensor hidden_forward_chunked(
    Model& model,
    const mfq_tensor_backend::Tensor & ids,
    int64_t chunk_size,
    mfq_tensor_backend::Tensor * raw_hidden = nullptr) {
    MFQ_RUNTIME_CHECK(
        chunk_size > 0,
        "runtime prefill chunk size must be positive");
    MFQ_RUNTIME_CHECK(
        ids.dim() == 2 && ids.size(0) == 1 && ids.size(1) > 0,
        "runtime prefill IDs must have shape [1, tokens]");
    std::vector<mfq_tensor_backend::Tensor> raw_chunks;
    if (raw_hidden != nullptr) {
        raw_chunks.reserve(static_cast<std::size_t>(
            (ids.size(1) + chunk_size - 1) / chunk_size));
    }
    mfq_tensor_backend::Tensor hidden;
    for (int64_t offset = 0; offset < ids.size(1); offset += chunk_size) {
        const int64_t count = std::min(chunk_size, ids.size(1) - offset);
        mfq_tensor_backend::Tensor raw_chunk;
        hidden = model.hidden_forward(
            ids.narrow(1, offset, count).contiguous(),
            mfq_nullopt,
            mfq_nullopt,
            nullptr,
            mfq_nullopt,
            raw_hidden != nullptr ? &raw_chunk : nullptr);
        if (raw_hidden != nullptr) {
            raw_chunks.push_back(std::move(raw_chunk));
        }
    }
    if (raw_hidden != nullptr) {
        *raw_hidden = raw_chunks.size() == 1
            ? std::move(raw_chunks.front())
            : mfq_tensor_backend::cat(raw_chunks, 1).contiguous();
    }
    return hidden;
}

template <typename Model>
static mfq_tensor_backend::Tensor hidden_forward_prepared_chunked(
    Model& model,
    const mfq_tensor_backend::Tensor& ids,
    const CudaPreparedPrompt& prepared,
    int64_t chunk_size,
    mfq_tensor_backend::Tensor* raw_hidden = nullptr,
    int64_t prepared_offset = 0) {
    MFQ_RUNTIME_CHECK(
        chunk_size > 0 && prepared.transformed() && prepared_offset >= 0 &&
            ids.dim() == 2 && ids.size(0) == 1 && ids.size(1) > 0 &&
            prepared.embeddings.defined() && prepared.positions.defined() &&
            prepared.embeddings.dim() == 3 &&
            prepared.embeddings.size(0) == 1 &&
            prepared_offset + ids.size(1) <= prepared.embeddings.size(1) &&
            prepared.embeddings.size(2) == model.hidden_size() &&
            (prepared.positions.dim() == 1 ||
             prepared.positions.dim() == 2) &&
            prepared_offset + ids.size(1) <= prepared.positions.size(-1),
        "prepared CUDA prefill tensors disagree with prompt geometry");
    std::vector<mfq_tensor_backend::Tensor> raw_chunks;
    if (raw_hidden != nullptr) {
        raw_chunks.reserve(static_cast<std::size_t>(
            (ids.size(1) + chunk_size - 1) / chunk_size));
    }
    mfq_tensor_backend::Tensor hidden;
    for (int64_t offset = 0; offset < ids.size(1); offset += chunk_size) {
        const int64_t count = std::min(chunk_size, ids.size(1) - offset);
        mfq_tensor_backend::Tensor raw_chunk;
        hidden = model.hidden_forward_inputs(
            ids.narrow(1, offset, count).contiguous(),
            prepared.embeddings.narrow(
                1, prepared_offset + offset, count).contiguous(),
            prepared.positions.narrow(
                -1, prepared_offset + offset, count).contiguous(),
            mfq_nullopt, nullptr, mfq_nullopt, true, mfq_nullopt,
            raw_hidden != nullptr ? &raw_chunk : nullptr);
        if (raw_hidden != nullptr) raw_chunks.push_back(std::move(raw_chunk));
    }
    model.decode_position_delta = prepared.decode_position_delta;
    if (raw_hidden != nullptr) {
        *raw_hidden = raw_chunks.size() == 1
            ? std::move(raw_chunks.front())
            : mfq_tensor_backend::cat(raw_chunks, 1).contiguous();
    }
    return hidden;
}

class PrefillCudaTimer {
public:
    PrefillCudaTimer()
        : stream_(mfq_get_current_cuda_stream()) {
        MFQ_CUDA_CHECK(cudaEventCreate(&started_));
        try {
            MFQ_CUDA_CHECK(cudaEventCreate(&finished_));
            MFQ_CUDA_CHECK(cudaEventRecord(started_, stream_));
        } catch (...) {
            if (finished_ != nullptr) cudaEventDestroy(finished_);
            cudaEventDestroy(started_);
            finished_ = nullptr;
            started_ = nullptr;
            throw;
        }
    }

    ~PrefillCudaTimer() {
        if (finished_ != nullptr) cudaEventDestroy(finished_);
        if (started_ != nullptr) cudaEventDestroy(started_);
    }

    cudaEvent_t finished_event() const {
        return finished_;
    }

    double elapsed_ms() const {
        MFQ_CUDA_CHECK(cudaEventSynchronize(finished_));
        float elapsed = 0.0f;
        MFQ_CUDA_CHECK(cudaEventElapsedTime(&elapsed, started_, finished_));
        return static_cast<double>(elapsed);
    }

private:
    cudaStream_t stream_ = nullptr;
    cudaEvent_t started_ = nullptr;
    cudaEvent_t finished_ = nullptr;
};

static uint64_t cuda_cache_environment_bytes(
        const char * name, uint64_t fallback) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') return fallback;
    char * end = nullptr;
    const auto parsed = std::strtoull(value, &end, 10);
    if (end == value || *end != '\0') {
        throw std::runtime_error(std::string("invalid ") + name);
    }
    return parsed;
}

static std::filesystem::path default_cuda_prefix_cache_directory() {
    if (const char * configured =
            std::getenv("MFQ_RUNTIME_PREFIX_CACHE_DIR")) {
        if (configured[0] != '\0') return configured;
    }
#ifdef _WIN32
    if (const char * local = std::getenv("LOCALAPPDATA")) {
        if (local[0] != '\0') {
            return std::filesystem::path(local) /
                "TyloQuant" / "MFQ" / "prefix-cache";
        }
    }
#else
    if (const char * xdg = std::getenv("XDG_CACHE_HOME")) {
        if (xdg[0] != '\0') {
            return std::filesystem::path(xdg) /
                "tyloquant" / "mfq" / "prefix-cache";
        }
    }
    if (const char * home = std::getenv("HOME")) {
        if (home[0] != '\0') {
            return std::filesystem::path(home) / ".cache" /
                "tyloquant" / "mfq" / "prefix-cache";
        }
    }
#endif
    return std::filesystem::temp_directory_path() /
        "tyloquant-mfq-prefix-cache";
}

template <typename Model>
static std::string cuda_prefix_cache_compatibility_key(
        const mfq::ModelSource& source, const Model& model) {
    std::ostringstream key;
    key << "mfq-cuda-prefix-v1\n"
        << "codec=cuda-full-attention-kv-v1\n"
        << "context=" << model.max_position_embeddings() << '\n'
        << "architecture=" << source.architecture() << '\n';
    for (std::size_t index = 0; index < source.source_paths().size(); ++index) {
        const auto& path = source.source_paths()[index];
        std::error_code error;
        const auto size = std::filesystem::file_size(path, error);
        if (error) {
            throw std::runtime_error(
                "cannot identify model source for prefix cache: " +
                error.message());
        }
        const auto modified = std::filesystem::last_write_time(path, error);
        if (error) {
            throw std::runtime_error(
                "cannot identify model source timestamp: " +
                error.message());
        }
        const auto modified_ns = std::chrono::duration_cast<
            std::chrono::nanoseconds>(modified.time_since_epoch()).count();
        key << "source=" << index << ':' << path.filename().string() << ':'
            << size << ':' << modified_ns << '\n';
    }
    for (const auto& tensor : source.tensors()) {
        key << "tensor=" << tensor.name << ':' << tensor.dtype << ':'
            << tensor.nbytes << '\n';
    }
    return key.str();
}

template <typename Model>
static std::shared_ptr<mfq::cache::PagedPrefixCache>
make_cuda_paged_prefix_cache(
        const mfq::ModelSource& source, const Model& model) {
    const auto format = source.metadata().find("source.format");
    // ponytail: HF source fingerprints exclude config sidecars for now.
    if ((format != source.metadata().end() &&
         format->second == "hf-safetensors") ||
        !model.supports_paged_text_session_state()) {
        return {};
    }
    if (const char * disabled =
            std::getenv("MFQ_RUNTIME_DISABLE_PREFIX_CACHE")) {
        if (disabled[0] == '1') return {};
    }
    const auto block_size = cuda_cache_environment_bytes(
        "MFQ_RUNTIME_PREFIX_CACHE_BLOCK_TOKENS", 256);
    if (block_size == 0 || block_size > 65536) {
        throw std::runtime_error(
            "MFQ_RUNTIME_PREFIX_CACHE_BLOCK_TOKENS must be in [1, 65536]");
    }
    mfq::cache::PagedPrefixCacheConfig config;
    config.cache_dir = default_cuda_prefix_cache_directory();
    config.compatibility_key =
        cuda_prefix_cache_compatibility_key(source, model);
    config.block_size_tokens = static_cast<size_t>(block_size);
    config.max_disk_bytes = cuda_cache_environment_bytes(
        "MFQ_RUNTIME_PREFIX_CACHE_DISK_BYTES",
        100ULL * 1024ULL * 1024ULL * 1024ULL);
    config.max_hot_bytes = cuda_cache_environment_bytes(
        "MFQ_RUNTIME_PREFIX_CACHE_HOT_BYTES",
        2ULL * 1024ULL * 1024ULL * 1024ULL);
    config.max_pending_writes = static_cast<size_t>(
        cuda_cache_environment_bytes(
            "MFQ_RUNTIME_PREFIX_CACHE_PENDING_WRITES", 64));
    config.max_pending_bytes = cuda_cache_environment_bytes(
        "MFQ_RUNTIME_PREFIX_CACHE_PENDING_BYTES",
        512ULL * 1024ULL * 1024ULL);
    return std::make_shared<mfq::cache::PagedPrefixCache>(
        std::move(config));
}

struct TextSessionRestore {
    size_t tokens = 0;
    mfq_tensor_backend::Tensor mtp_last_target_hidden;
};

class TextSessionCache {
private:
    struct PagedBinding {
        std::vector<mfq::cache::BlockHash> blocks;
        size_t tokens = 0;
        uint64_t last_used = 0;
    };

public:
    explicit TextSessionCache(
            std::shared_ptr<mfq::cache::PagedPrefixCache> paged_cache = {},
            bool supported = true,
            int disabled_reason = 0)
        : paged_cache_(std::move(paged_cache)),
          supported_(supported),
          disabled_reason_(disabled_reason) {
        const char * entries =
            std::getenv("MFQ_RUNTIME_MAX_KV_SESSIONS");
        if (entries != nullptr) {
            max_sessions_ = static_cast<size_t>(std::strtoull(
                entries, nullptr, 10));
        }
        const char * snapshots =
            std::getenv("MFQ_RUNTIME_MAX_KV_SNAPSHOTS_PER_SESSION");
        if (snapshots != nullptr) {
            max_snapshots_per_session_ = static_cast<size_t>(std::strtoull(
                snapshots, nullptr, 10));
        }
        const char * bytes =
            std::getenv("MFQ_RUNTIME_KV_SESSION_BYTES");
        if (bytes != nullptr) {
            max_bytes_ = static_cast<size_t>(std::strtoull(
                bytes, nullptr, 10));
        }
        const char * trace =
            std::getenv("MFQ_RUNTIME_TRACE_SESSION_CACHE");
        trace_ = trace != nullptr && trace[0] == '1';
        if (paged_cache_) {
            paged_disk_budget_ = cuda_cache_environment_bytes(
                "MFQ_RUNTIME_PREFIX_CACHE_DISK_BYTES",
                100ULL * 1024ULL * 1024ULL * 1024ULL);
            paged_hot_budget_ = cuda_cache_environment_bytes(
                "MFQ_RUNTIME_PREFIX_CACHE_HOT_BYTES",
                2ULL * 1024ULL * 1024ULL * 1024ULL);
        }
    }

    bool persistent_prefix_enabled() const noexcept {
        return static_cast<bool>(paged_cache_);
    }

    template <typename Model>
    TextSessionRestore restore_best(
            Model& model,
            MtpModule* mtp,
            const std::string & requested_session,
            const std::vector<int64_t> & prompt,
            size_t maximum_prefix_tokens,
            const std::string& input_key = {}) {
        if (!supported_) return {};
        if (paged_cache_ && mtp == nullptr && input_key.empty()) {
            return {restore_paged(
                model,
                requested_session,
                prompt,
                maximum_prefix_tokens), {}};
        }
        if (requested_session.empty() || max_sessions_ == 0 ||
                max_snapshots_per_session_ == 0 ||
                max_bytes_ == 0 || !model.supports_text_session_state()) {
            return {};
        }
        ++queries_;
        std::string selected_session;
        size_t selected_snapshot = 0;
        size_t selected_tokens = 0;
        for (const auto & [session_id, history] : states_) {
            for (size_t index = 0; index < history.size(); ++index) {
                const auto & state = history[index];
                const auto & tokens = state.tokens;
                if (tokens.empty() || tokens.size() >= prompt.size() ||
                        tokens.size() > maximum_prefix_tokens ||
                        tokens.size() < selected_tokens ||
                        state.input_key != input_key ||
                        (mtp != nullptr &&
                         (!mtp->supports_session_state() ||
                          !state.mtp.has_value())) ||
                        !std::equal(
                            tokens.begin(), tokens.end(), prompt.begin())) {
                    continue;
                }
                const bool requested_tie =
                    tokens.size() == selected_tokens &&
                    session_id == requested_session &&
                    selected_session != requested_session;
                if (tokens.size() > selected_tokens || requested_tie) {
                    selected_session = session_id;
                    selected_snapshot = index;
                    selected_tokens = tokens.size();
                }
            }
        }
        if (selected_session.empty()) return {};
        auto & selected = states_.at(selected_session)[selected_snapshot];
        try {
            model.restore_text_session_state(selected);
            TextSessionRestore restored{selected_tokens, {}};
            if (mtp != nullptr) {
                mtp->restore_session_state(*selected.mtp);
                restored.mtp_last_target_hidden =
                    selected.mtp->last_target_hidden;
            }
            selected.last_used = ++clock_;
            ++hits_;
            hit_tokens_ += selected_tokens;
            if (trace_) {
                std::cerr << "runtime_session_cache action=hit session="
                          << requested_session
                          << " source=" << selected_session
                          << " reused_tokens=" << selected_tokens
                          << " prefill_tokens="
                          << prompt.size() - selected_tokens << std::endl;
            }
            return restored;
        } catch (const std::exception & error) {
            erase_snapshot(selected_session, selected_snapshot, "invalidate");
            model.reset(1);
            if (mtp != nullptr) mtp->reset(1);
            std::cerr << "runtime_session_cache action=invalidate session="
                      << selected_session << " error=" << error.what()
                      << std::endl;
            return {};
        }
    }

    void store(
            const std::string & session_id,
            TextSessionState state) {
        if (!supported_) return;
        if (paged_cache_ && state.input_key.empty() &&
                !state.mtp.has_value()) {
            store_paged(session_id, state);
            return;
        }
        if (session_id.empty() || max_sessions_ == 0 ||
                max_snapshots_per_session_ == 0 || max_bytes_ == 0) {
            return;
        }
        if (state.bytes > max_bytes_) {
            if (trace_) {
                std::cerr << "runtime_session_cache action=skip session="
                          << session_id << " bytes=" << state.bytes
                          << " budget=" << max_bytes_ << std::endl;
            }
            return;
        }
        state.last_used = ++clock_;
        const uint64_t protected_clock = state.last_used;
        auto & history = states_[session_id];
        auto previous = std::find_if(
            history.begin(), history.end(),
            [&](const TextSessionState & saved) {
                return saved.tokens == state.tokens &&
                    saved.input_key == state.input_key;
            });
        if (previous != history.end()) {
            bytes_ -= previous->bytes;
            *previous = std::move(state);
        } else {
            history.push_back(std::move(state));
        }
        const auto stored = std::find_if(
            history.begin(), history.end(),
            [&](const TextSessionState & saved) {
                return saved.last_used == protected_clock;
            });
        if (stored == history.end()) {
            throw std::runtime_error("stored session snapshot is unavailable");
        }
        bytes_ += stored->bytes;
        evict_history_to_limit(session_id, protected_clock);
        evict_to_budget(session_id, protected_clock);
        sync_telemetry();
        if (trace_) {
            const auto & saved_history = states_.at(session_id);
            const auto saved = std::find_if(
                saved_history.begin(), saved_history.end(),
                [&](const TextSessionState & candidate) {
                    return candidate.last_used == protected_clock;
                });
            if (saved == saved_history.end()) {
                throw std::runtime_error(
                    "protected session snapshot was evicted");
            }
            std::cerr << "runtime_session_cache action=store session="
                      << session_id << " tokens=" << saved->tokens.size()
                      << " bytes=" << saved->bytes
                      << " snapshots=" << saved_history.size()
                      << " total_bytes=" << bytes_ << std::endl;
        }
    }

    size_t fork_session(
            const std::string & source_session,
            const std::string & target_session) {
        if (paged_cache_) {
            const auto source = paged_bindings_.find(source_session);
            if (source == paged_bindings_.end() ||
                    source_session.empty() || target_session.empty() ||
                    source_session == target_session) {
                return 0;
            }
            bind_paged_session(
                target_session,
                source->second.blocks,
                source->second.tokens);
            return source->second.blocks.size();
        }
        if (source_session.empty() || target_session.empty() ||
                source_session == target_session || max_sessions_ == 0 ||
                max_snapshots_per_session_ == 0 || max_bytes_ == 0) {
            return 0;
        }
        const auto source = states_.find(source_session);
        if (source == states_.end()) return 0;
        std::vector<TextSessionState> copied = source->second;
        close_session(target_session);
        auto & target = states_[target_session];
        uint64_t protected_clock = 0;
        for (auto & snapshot : copied) {
            snapshot.last_used = ++clock_;
            protected_clock = snapshot.last_used;
            bytes_ += snapshot.bytes;
            target.push_back(std::move(snapshot));
        }
        evict_history_to_limit(target_session, protected_clock);
        evict_to_budget(target_session, protected_clock);
        const auto remaining = states_.find(target_session);
        const size_t copied_snapshots = remaining == states_.end()
            ? 0 : remaining->second.size();
        sync_telemetry();
        if (trace_) {
            std::cerr << "runtime_session_cache action=fork source="
                      << source_session << " target=" << target_session
                      << " snapshots=" << copied_snapshots
                      << " total_bytes=" << bytes_ << std::endl;
        }
        return copied_snapshots;
    }

    size_t close_session(const std::string & session_id) {
        if (paged_cache_) return close_paged_session(session_id);
        auto found = states_.find(session_id);
        if (found == states_.end()) return 0;
        const size_t released = found->second.size();
        size_t released_bytes = 0;
        for (const auto & snapshot : found->second) {
            released_bytes += snapshot.bytes;
        }
        bytes_ -= released_bytes;
        states_.erase(found);
        sync_telemetry();
        if (trace_) {
            std::cerr << "runtime_session_cache action=close session="
                      << session_id << " snapshots=" << released
                      << " bytes=" << released_bytes
                      << " total_bytes=" << bytes_ << std::endl;
        }
        return released;
    }

    std::vector<std::pair<std::string, double>> metrics() const {
        if (paged_cache_) {
            const auto value = paged_cache_->metrics();
            return {
                {"prefix_cache_supported", supported_ ? 1.0 : 0.0},
                {"prefix_cache_disabled_reason",
                    static_cast<double>(disabled_reason_)},
                {"prefix_cache_queries", static_cast<double>(value.queries)},
                {"prefix_cache_hits", static_cast<double>(value.hits)},
                {"prefix_cache_hit_tokens", static_cast<double>(value.hit_tokens)},
                {"prefix_cache_sessions", static_cast<double>(metric_sessions_.load())},
                {"prefix_cache_snapshots", static_cast<double>(value.disk_blocks)},
                {"prefix_cache_tokens", static_cast<double>(metric_tokens_.load())},
                {"prefix_cache_bytes", static_cast<double>(value.hot_bytes)},
                {"prefix_cache_max_sessions", static_cast<double>(max_sessions_)},
                {"prefix_cache_max_snapshots_per_session", 1.0},
                {"prefix_cache_max_bytes", static_cast<double>(paged_hot_budget_)},
                {"prefix_cache_disk_blocks", static_cast<double>(value.disk_blocks)},
                {"prefix_cache_disk_bytes", static_cast<double>(value.disk_bytes)},
                {"prefix_cache_disk_max_bytes", static_cast<double>(paged_disk_budget_)},
                {"prefix_cache_hot_blocks", static_cast<double>(value.hot_blocks)},
                {"prefix_cache_hot_bytes", static_cast<double>(value.hot_bytes)},
                {"prefix_cache_pending_writes", static_cast<double>(value.pending_writes)},
                {"prefix_cache_pending_bytes", static_cast<double>(value.pending_bytes)},
                {"prefix_cache_pending_max_bytes", static_cast<double>(value.pending_max_bytes)},
                {"prefix_cache_writes", static_cast<double>(value.writes)},
                {"prefix_cache_deduplicated_writes", static_cast<double>(value.deduplicated_writes)},
                {"prefix_cache_disk_hits", static_cast<double>(value.disk_hits)},
                {"prefix_cache_hot_hits", static_cast<double>(value.hot_hits)},
                {"prefix_cache_evictions", static_cast<double>(value.evictions)},
                {"prefix_cache_corrupt_blocks", static_cast<double>(value.corrupt_blocks)},
            };
        }
        return {
            {"prefix_cache_supported", supported_ ? 1.0 : 0.0},
            {"prefix_cache_disabled_reason",
                static_cast<double>(disabled_reason_)},
            {"prefix_cache_queries", static_cast<double>(queries_.load())},
            {"prefix_cache_hits", static_cast<double>(hits_.load())},
            {"prefix_cache_hit_tokens", static_cast<double>(hit_tokens_.load())},
            {"prefix_cache_sessions", static_cast<double>(metric_sessions_.load())},
            {"prefix_cache_snapshots", static_cast<double>(metric_snapshots_.load())},
            {"prefix_cache_tokens", static_cast<double>(metric_tokens_.load())},
            {"prefix_cache_bytes", static_cast<double>(metric_bytes_.load())},
            {"prefix_cache_max_sessions", static_cast<double>(max_sessions_)},
            {"prefix_cache_max_snapshots_per_session",
                static_cast<double>(max_snapshots_per_session_)},
            {"prefix_cache_max_bytes", static_cast<double>(max_bytes_)},
        };
    }

    size_t clear_live_sessions() noexcept {
        if (paged_cache_) {
            const auto sessions = paged_bindings_.size();
            for (const auto & [session, binding] : paged_bindings_) {
                (void)session;
                paged_cache_->unpin(binding.blocks);
            }
            paged_bindings_.clear();
            sync_paged_telemetry();
            return sessions;
        }
        size_t snapshots = 0;
        for (const auto & [session_id, history] : states_) {
            (void)session_id;
            snapshots += history.size();
        }
        states_.clear();
        bytes_ = 0;
        sync_telemetry();
        return snapshots;
    }

    size_t clear() {
        if (paged_cache_) {
            clear_live_sessions();
            return paged_cache_->clear();
        }
        return clear_live_sessions();
    }

    uint64_t trim_hot(uint64_t target_bytes) {
        if (!paged_cache_) return 0;
        const auto released = paged_cache_->trim_hot(target_bytes);
        if (released > 0) mfq_release_host_allocator_cache();
        return released;
    }

private:
    template <typename Model>
    size_t restore_paged(
            Model& model,
            const std::string & requested_session,
            const std::vector<int64_t> & prompt,
            size_t maximum_prefix_tokens) {
        if (max_sessions_ == 0 || !model.supports_paged_text_session_state() ||
                prompt.size() < 2) {
            return 0;
        }
        const auto limit = std::min(
            maximum_prefix_tokens, prompt.size() - 1);
        std::vector<int64_t> candidate(
            prompt.begin(),
            prompt.begin() + static_cast<std::ptrdiff_t>(limit));
        auto match = paged_cache_->match(candidate, {}, false);
        if (match.matched_tokens == 0) {
            paged_cache_->record_match(0);
            return 0;
        }

        auto payloads = paged_cache_->load_prefix(match.blocks);
        if (payloads.empty()) {
            paged_cache_->record_match(0);
            return 0;
        }
        if (payloads.size() != match.blocks.size()) {
            match.blocks.resize(payloads.size());
            match.matched_tokens =
                payloads.size() * paged_cache_->block_size_tokens();
        }
        std::vector<int64_t> matched_tokens(
            prompt.begin(),
            prompt.begin() + static_cast<std::ptrdiff_t>(
                match.matched_tokens));
        std::optional<TextSessionState> state;
        try {
            state.emplace(decode_cuda_paged_session(
                payloads,
                matched_tokens,
                paged_cache_->block_size_tokens()));
        } catch (const std::exception & error) {
            if (!match.blocks.empty()) {
                paged_cache_->invalidate(match.blocks.back());
            }
            paged_cache_->record_match(0);
            model.reset(1);
            std::cerr
                << "runtime_session_cache backend=cuda "
                << "action=paged_codec_invalidate "
                << "session=" << requested_session
                << " error=" << error.what() << std::endl;
            return 0;
        }
        try {
            model.restore_text_session_state(*state);
            if (!requested_session.empty()) {
                bind_paged_session(
                    requested_session, match.blocks, match.matched_tokens);
            }
            paged_cache_->record_match(match.matched_tokens);
            if (trace_) {
                std::cerr
                    << "runtime_session_cache backend=cuda action=paged_hit "
                    << "session=" << requested_session
                    << " reused_tokens=" << match.matched_tokens
                    << " prefill_tokens="
                    << prompt.size() - match.matched_tokens << std::endl;
            }
            return match.matched_tokens;
        } catch (const std::exception & error) {
            paged_cache_->record_match(0);
            model.reset(1);
            std::cerr
                << "runtime_session_cache backend=cuda "
                << "action=paged_restore_failed "
                << "session=" << requested_session
                << " error=" << error.what() << std::endl;
            return 0;
        }
    }

    void store_paged(
            const std::string & session_id,
            const TextSessionState & state) {
        if (max_sessions_ == 0 || state.tokens.empty()) {
            return;
        }
        const auto block_size = paged_cache_->block_size_tokens();
        const auto full_blocks = state.tokens.size() / block_size;
        if (full_blocks == 0) return;

        auto existing = paged_cache_->match(state.tokens, {}, false);
        if (existing.blocks.size() > full_blocks) {
            throw std::runtime_error(
                "paged prefix match exceeds the CUDA session state");
        }
        auto blocks = std::move(existing.blocks);
        mfq::cache::BlockHash parent{};
        if (!blocks.empty()) parent = blocks.back();
        for (size_t index = blocks.size(); index < full_blocks; ++index) {
            const auto token_offset = index * block_size;
            auto payload = encode_cuda_paged_block(
                state, block_size, index);
            parent = paged_cache_->store(
                parent,
                state.tokens.data() + token_offset,
                block_size,
                std::move(payload));
            blocks.push_back(parent);
        }
        if (!session_id.empty()) {
            bind_paged_session(
                session_id,
                std::move(blocks),
                full_blocks * block_size);
        }
        if (trace_) {
            std::cerr
                << "runtime_session_cache backend=cuda action=paged_store "
                << "session=" << session_id
                << " tokens=" << full_blocks * block_size
                << " blocks=" << full_blocks << std::endl;
        }
    }

    void bind_paged_session(
            const std::string & session_id,
            std::vector<mfq::cache::BlockHash> blocks,
            size_t tokens) {
        close_paged_session(session_id);
        paged_cache_->pin(blocks);
        paged_bindings_[session_id] = PagedBinding{
            std::move(blocks), tokens, ++clock_};
        while (paged_bindings_.size() > max_sessions_) {
            auto victim = paged_bindings_.end();
            for (auto iterator = paged_bindings_.begin();
                    iterator != paged_bindings_.end(); ++iterator) {
                if (iterator->first == session_id) continue;
                if (victim == paged_bindings_.end() ||
                        iterator->second.last_used <
                            victim->second.last_used) {
                    victim = iterator;
                }
            }
            if (victim == paged_bindings_.end()) break;
            close_paged_session(victim->first);
        }
        sync_paged_telemetry();
    }

    size_t close_paged_session(const std::string & session_id) {
        auto found = paged_bindings_.find(session_id);
        if (found == paged_bindings_.end()) return 0;
        const auto blocks = found->second.blocks.size();
        paged_cache_->unpin(found->second.blocks);
        paged_bindings_.erase(found);
        sync_paged_telemetry();
        return blocks;
    }

    void sync_paged_telemetry() noexcept {
        size_t tokens = 0;
        for (const auto & [session, binding] : paged_bindings_) {
            (void)session;
            tokens += binding.tokens;
        }
        metric_sessions_.store(paged_bindings_.size());
        metric_tokens_.store(tokens);
    }

    void sync_telemetry() noexcept {
        size_t snapshots = 0;
        size_t tokens = 0;
        for (const auto & [session_id, history] : states_) {
            (void)session_id;
            snapshots += history.size();
            for (const auto & snapshot : history) {
                tokens += snapshot.tokens.size();
            }
        }
        metric_sessions_.store(states_.size());
        metric_snapshots_.store(snapshots);
        metric_tokens_.store(tokens);
        metric_bytes_.store(bytes_);
    }

    void evict_history_to_limit(
            const std::string & session_id,
            uint64_t protected_clock) {
        auto found = states_.find(session_id);
        while (found != states_.end() &&
                found->second.size() > max_snapshots_per_session_) {
            size_t victim = found->second.size();
            for (size_t index = 0; index < found->second.size(); ++index) {
                const auto & snapshot = found->second[index];
                if (snapshot.last_used == protected_clock) continue;
                if (victim == found->second.size() ||
                        snapshot.last_used <
                            found->second[victim].last_used) {
                    victim = index;
                }
            }
            if (victim == found->second.size()) break;
            erase_snapshot(session_id, victim, "history_evict");
            found = states_.find(session_id);
        }
    }

    void evict_to_budget(
            const std::string & protected_session,
            uint64_t protected_clock) {
        while (states_.size() > max_sessions_) {
            auto victim = states_.end();
            uint64_t victim_last_used = 0;
            for (auto it = states_.begin(); it != states_.end(); ++it) {
                if (it->first == protected_session) continue;
                uint64_t session_last_used = 0;
                for (const auto & snapshot : it->second) {
                    session_last_used = std::max(
                        session_last_used, snapshot.last_used);
                }
                if (victim == states_.end() ||
                        session_last_used < victim_last_used) {
                    victim = it;
                    victim_last_used = session_last_used;
                }
            }
            if (victim == states_.end()) break;
            close_session(victim->first);
        }
        while (bytes_ > max_bytes_) {
            std::string victim_session;
            size_t victim_snapshot = 0;
            uint64_t victim_last_used = 0;
            bool found_victim = false;
            for (const auto & [session_id, history] : states_) {
                for (size_t index = 0; index < history.size(); ++index) {
                    const auto & snapshot = history[index];
                    if (session_id == protected_session &&
                            snapshot.last_used == protected_clock) {
                        continue;
                    }
                    if (!found_victim ||
                            snapshot.last_used < victim_last_used) {
                        victim_session = session_id;
                        victim_snapshot = index;
                        victim_last_used = snapshot.last_used;
                        found_victim = true;
                    }
                }
            }
            if (!found_victim) break;
            erase_snapshot(victim_session, victim_snapshot, "budget_evict");
        }
    }

    void erase_snapshot(
            const std::string & session_id,
            size_t index,
            const char * action) {
        auto found = states_.find(session_id);
        if (found == states_.end() || index >= found->second.size()) return;
        const size_t removed_bytes = found->second[index].bytes;
        if (trace_) {
            std::cerr << "runtime_session_cache action=" << action
                      << " session=" << session_id
                      << " tokens=" << found->second[index].tokens.size()
                      << " bytes=" << removed_bytes << std::endl;
        }
        bytes_ -= removed_bytes;
        found->second.erase(found->second.begin() +
            static_cast<std::ptrdiff_t>(index));
        if (found->second.empty()) states_.erase(found);
        sync_telemetry();
    }

    std::unordered_map<
        std::string, std::vector<TextSessionState>> states_;
    std::unordered_map<std::string, PagedBinding> paged_bindings_;
    std::shared_ptr<mfq::cache::PagedPrefixCache> paged_cache_;
    uint64_t paged_disk_budget_ = 0;
    uint64_t paged_hot_budget_ = 0;
    size_t max_sessions_ = 4;
    size_t max_snapshots_per_session_ = 4;
    size_t max_bytes_ = 2ULL * 1024ULL * 1024ULL * 1024ULL;
    size_t bytes_ = 0;
    uint64_t clock_ = 0;
    std::atomic<uint64_t> queries_{0};
    std::atomic<uint64_t> hits_{0};
    std::atomic<uint64_t> hit_tokens_{0};
    std::atomic<size_t> metric_sessions_{0};
    std::atomic<size_t> metric_snapshots_{0};
    std::atomic<size_t> metric_tokens_{0};
    std::atomic<size_t> metric_bytes_{0};
    bool trace_ = false;
    bool supported_ = true;
    int disabled_reason_ = 0;
};



template <typename Model>
using PreparedPromptFactory =
    std::function<std::optional<CudaPreparedPrompt>(Model&)>;

template <typename Model>
static int32_t generate_tokens(
    Model& model,
    std::mutex & model_mutex,
    DecodeGraphCache & graph_cache,
    TextSessionCache & session_cache,
    const std::vector<int64_t> & prompt,
    const MfqSamplingParams & sampling,
    const MfqTokenCallback & on_token,
    const MfqPrefillCallback & on_prefill,
    const MfqPromptCachePlan & cache_plan,
    const MfqTokenConstraintPtr & token_constraint,
    MtpModule* mtp = nullptr,
    int64_t prefill_chunk_size = 2048,
    PreparedPromptFactory<Model> prepare_prompt = {})
{
    std::lock_guard<std::mutex> lock(model_mutex);
    std::optional<CudaPreparedPrompt> prepared;
    double multimodal_ms = 0.0;
    if (prepare_prompt) {
        PrefillCudaTimer multimodal_timer;
        prepared = prepare_prompt(model);
        MFQ_CUDA_CHECK(cudaEventRecord(
            multimodal_timer.finished_event(),
            mfq_get_current_cuda_stream()));
        multimodal_ms = multimodal_timer.elapsed_ms();
    }
    if (prepared && prepared->token_ids != prompt) {
        throw std::invalid_argument(
            "prepared prompt token IDs disagree with the rendered prompt");
    }
    const bool transformed_prompt = prepared && prepared->transformed();
    const std::string input_key = prepared ? prepared->cache_key : std::string{};
    if (mtp != nullptr) {
        mtp->last_stats = {};
        mtp->last_stats.available = true;
    }
    const char* mtp_reprefill = std::getenv("MFQ_RUNTIME_REPREFILL");
    const char* mtp_trace = std::getenv("MFQ_RUNTIME_TRACE_INCREMENTAL");
    const bool use_mtp =
        mtp != nullptr && sampling.enable_mtp && sampling.max_tokens > 1 &&
        mfq_token_constraint_supports_speculation(token_constraint) &&
        !(mtp_reprefill != nullptr && mtp_reprefill[0] == '1') &&
        !(mtp_trace != nullptr && mtp_trace[0] == '1');
    const size_t stable_prefix_tokens = std::min(
        cache_plan.stable_prefix_tokens, prompt.size());
    const bool cache_enabled =
        stable_prefix_tokens > 0 &&
        (!transformed_prompt || !input_key.empty()) &&
        (!cache_plan.session_id.empty() ||
         session_cache.persistent_prefix_enabled()) &&
        model.supports_text_session_state();
    const bool can_restore_mtp = use_mtp && mtp->supports_session_state();
    const auto restored = cache_enabled && (!use_mtp || can_restore_mtp)
        ? session_cache.restore_best(
            model, use_mtp ? mtp : nullptr,
            cache_plan.session_id, prompt, stable_prefix_tokens, input_key)
        : TextSessionRestore{};
    const size_t reused_tokens = restored.tokens;
    if (reused_tokens == 0) {
        model.reset(1);
        if (use_mtp) mtp->reset(1);
    }
    if (use_mtp) {
        if constexpr (
                Model::backbone == mfq::cuda::CudaBackbone::generic_qwen ||
                Model::backbone == mfq::cuda::CudaBackbone::qwen4_exp ||
                Model::backbone == mfq::cuda::CudaBackbone::glm5_next ||
                Model::backbone == mfq::cuda::CudaBackbone::deepseek_v41) {
            std::vector<int64_t> history = prompt;
            const auto tracking_callback = [&](int32_t token) {
                const bool keep_going = !on_token || on_token(token);
                if (keep_going) history.push_back(token);
                return keep_going;
            };
            mfq_tensor_backend::Tensor last_target_hidden;
            const int32_t generated = run_mtp_generation<Model::backbone>(
                model, *mtp, prompt, sampling, tracking_callback, on_prefill,
                prefill_chunk_size, token_constraint,
                transformed_prompt ? &*prepared : nullptr,
                reused_tokens, restored.mtp_last_target_hidden,
                &last_target_hidden, multimodal_ms);
            if (cache_enabled && last_target_hidden.defined() &&
                    model.cache_pos > 1 &&
                    model.cache_pos <= static_cast<int64_t>(history.size())) {
                try {
                    history.resize(static_cast<size_t>(model.cache_pos));
                    auto state = model.capture_text_session_state(history);
                    state.input_key = input_key;
                    state.mtp = mtp->capture_session_state(
                        model.cache_pos, last_target_hidden);
                    state.bytes += state.mtp->bytes;
                    session_cache.store(
                        cache_plan.session_id, std::move(state));
                } catch (const std::exception& error) {
                    std::cerr
                        << "runtime_session_cache action=skip session="
                        << cache_plan.session_id
                        << " error=" << error.what() << std::endl;
                }
            }
            return generated;
        }
        throw std::runtime_error(
            "MTP is unavailable for this causal LM type");
    }
    auto options = mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64).device(mfq_tensor_backend::kCUDA);
    auto full_ids = mfq_tensor_backend::tensor(prompt, options)
        .reshape({1, -1}).contiguous();
    auto ids = full_ids.narrow(
        1, static_cast<int64_t>(reused_tokens),
        static_cast<int64_t>(prompt.size() - reused_tokens)).contiguous();
    graph_cache.ensure_storage(model.vocab_size());
    auto random_host = mfq_tensor_backend::empty(
        {1}, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kFloat32).device(mfq_tensor_backend::kCPU).pinned_memory(true));
    auto random_cuda = mfq_tensor_backend::empty(
        {1}, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kFloat32).device(mfq_tensor_backend::kCUDA));
    mfq::cuda::Sampler sampler(
        sampling,
        mfq::cuda::SamplingOps(random_host, std::move(random_cuda)));
    const bool has_penalties = sampler.has_penalties();
    auto counts = has_penalties ? graph_cache.counts : mfq_tensor_backend::Tensor();
    if (has_penalties) {
        counts.zero_();
        sample_token_counts_add_cuda(counts, full_ids);
    }
    const auto store_session_snapshot = [&](
            const std::vector<int64_t>& snapshot_tokens) {
        if (!cache_enabled || model.cache_pos !=
                static_cast<int64_t>(snapshot_tokens.size())) {
            return;
        }
        try {
            auto state = model.capture_text_session_state(snapshot_tokens);
            state.input_key = input_key;
            session_cache.store(
                cache_plan.session_id, std::move(state));
        } catch (const std::exception & error) {
            std::cerr << "runtime_session_cache action=skip session="
                      << cache_plan.session_id
                      << " error=" << error.what() << std::endl;
        }
    };
    auto sample_first_token = [&]() {
        PrefillCudaTimer prefill_timer;
        mfq_tensor_backend::Tensor next;
        if (transformed_prompt) {
            size_t begin = reused_tokens;
            mfq_tensor_backend::Tensor hidden;
            if (cache_enabled && stable_prefix_tokens < prompt.size()) {
                if (begin < stable_prefix_tokens) {
                    auto stable_ids = full_ids.narrow(
                        1, static_cast<int64_t>(begin),
                        static_cast<int64_t>(stable_prefix_tokens - begin))
                        .contiguous();
                    hidden = hidden_forward_prepared_chunked(
                        model, stable_ids, *prepared, prefill_chunk_size,
                        nullptr, static_cast<int64_t>(begin));
                }
                store_session_snapshot(std::vector<int64_t>(
                    prompt.begin(), prompt.begin() +
                        static_cast<std::ptrdiff_t>(stable_prefix_tokens)));
                begin = stable_prefix_tokens;
            }
            if (begin < prompt.size()) {
                auto remaining_ids = full_ids.narrow(
                    1, static_cast<int64_t>(begin),
                    static_cast<int64_t>(prompt.size() - begin)).contiguous();
                hidden = hidden_forward_prepared_chunked(
                    model, remaining_ids, *prepared, prefill_chunk_size,
                    nullptr, static_cast<int64_t>(begin));
            }
            MFQ_RUNTIME_CHECK(
                hidden.defined(),
                "prepared CUDA prefill produced no hidden state");
            auto logits = model.lm_head.forward(
                hidden.index({Slice(), -1, Slice()})
                    .to(mfq_tensor_backend::kFloat16).contiguous())
                .contiguous().view({1, -1});
            MFQ_CUDA_CHECK(cudaEventRecord(
                prefill_timer.finished_event(),
                mfq_get_current_cuda_stream()));
            next = mfq::cuda::sample_logits(
                sampler, std::move(logits), counts, token_constraint);
        } else {
            if (cache_enabled && stable_prefix_tokens < prompt.size()) {
                if (reused_tokens < stable_prefix_tokens) {
                    auto stable_suffix = full_ids.narrow(
                        1, static_cast<int64_t>(reused_tokens),
                        static_cast<int64_t>(
                            stable_prefix_tokens - reused_tokens)).contiguous();
                    stable_suffix = prefill_tail(
                        model, std::move(stable_suffix), prefill_chunk_size);
                    MfqOptional<mfq_tensor_backend::Tensor> stable_seq_len = mfq_nullopt;
                    if (!Model::is_minicpmo45 && model.cache_pos > 0 &&
                            stable_suffix.size(1) == 1) {
                        stable_seq_len = mfq_tensor_backend::full(
                            {1}, model.cache_pos + 1, options);
                    }
                    (void)model.hidden_forward(
                        stable_suffix, mfq_nullopt, stable_seq_len);
                }
                store_session_snapshot(std::vector<int64_t>(
                    prompt.begin(), prompt.begin() +
                        static_cast<std::ptrdiff_t>(stable_prefix_tokens)));
                ids = full_ids.narrow(
                    1, static_cast<int64_t>(stable_prefix_tokens),
                    static_cast<int64_t>(
                        prompt.size() - stable_prefix_tokens)).contiguous();
            }
            ids = prefill_tail(
                model, std::move(ids), prefill_chunk_size);
            next = sample_token(
                model, ids, sampler, counts, token_constraint,
                prefill_timer.finished_event());
        }
        const int64_t token = next.template item<int64_t>();
        const double prefill_ms = prefill_timer.elapsed_ms();
        if (stable_prefix_tokens == prompt.size()) {
            store_session_snapshot(prompt);
        }
        if (on_prefill) {
            on_prefill(MfqPrefillTiming{
                prompt.size() - reused_tokens,
                prefill_ms,
                multimodal_ms,
                prefill_ms + multimodal_ms});
        }
        return std::make_pair(std::move(next), token);
    };
    const char * reprefill_env = std::getenv("MFQ_RUNTIME_REPREFILL");
    const bool reprefill = !transformed_prompt &&
        reprefill_env != nullptr && reprefill_env[0] == '1';
    std::vector<int64_t> history = prompt;
    const auto store_live_history = [&]() {
        if (model.cache_pos <= 0 ||
                model.cache_pos > static_cast<int64_t>(history.size())) {
            return;
        }
        store_session_snapshot(std::vector<int64_t>(
            history.begin(), history.begin() + model.cache_pos));
    };
    const char * trace_incremental_env =
        std::getenv("MFQ_RUNTIME_TRACE_INCREMENTAL");
    const bool trace_incremental =
        trace_incremental_env != nullptr && trace_incremental_env[0] == '1';
    if (!transformed_prompt && trace_incremental && sampling.max_tokens > 0) {
        auto [first, first_token] = sample_first_token();
        if (!on_token(first_token)) return 1;

        std::vector<mfq_tensor_backend::Tensor> incremental_trace;
        std::vector<mfq_tensor_backend::Tensor> full_trace;
        std::vector<std::pair<std::string, mfq_tensor_backend::Tensor>> incremental_gemma_trace;
        std::vector<std::pair<std::string, mfq_tensor_backend::Tensor>> full_gemma_trace;
        const int64_t decode_len = model.cache_pos + 1;
        auto seq_len = mfq_tensor_backend::tensor({decode_len}, options);
        g_gemma_trace_layer = 0;
        g_gemma_stage_trace = &incremental_gemma_trace;
        auto incremental_hidden = model.hidden_forward(
            first.reshape({1, 1}), mfq_nullopt, seq_len, &incremental_trace);
        g_gemma_stage_trace = nullptr;

        history.push_back(first_token);
        model.reset(1);
        auto full_ids = mfq_tensor_backend::tensor(history, options).reshape({1, -1}).contiguous();
        g_gemma_stage_trace = &full_gemma_trace;
        auto full_hidden = model.hidden_forward(
            full_ids, mfq_nullopt, mfq_nullopt, &full_trace);
        g_gemma_stage_trace = nullptr;
        g_gemma_trace_layer = -1;
        mfq_cuda_synchronize();

        if (incremental_trace.size() != full_trace.size()) {
            throw std::runtime_error("incremental trace stage count mismatch");
        }
        for (size_t i = 0; i < incremental_trace.size(); ++i) {
            auto got = incremental_trace[i].reshape({-1}).to(mfq_tensor_backend::kFloat64);
            auto ref = full_trace[i].index({Slice(), -1, Slice()}).reshape({-1}).to(mfq_tensor_backend::kFloat64);
            const double denominator = std::max(ref.norm().template item<double>(), 1.0e-30);
            const double relative_l2 = (got - ref).norm().template item<double>() / denominator;
            const double cosine = mfq_tensor_backend::dot(got, ref).template item<double>() /
                std::max(got.norm().template item<double>() * denominator, 1.0e-30);
            std::cerr << "incremental_trace stage="
                      << (i == 0 ? "embedding" : "block_" + std::to_string(i - 1))
                      << " relative_l2=" << relative_l2
                      << " cosine=" << cosine << std::endl;
        }
        if (incremental_gemma_trace.size() != full_gemma_trace.size()) {
            throw std::runtime_error("incremental Gemma stage count mismatch");
        }
        for (size_t i = 0; i < incremental_gemma_trace.size(); ++i) {
            const auto & got_tensor = incremental_gemma_trace[i].second;
            const auto & full_tensor = full_gemma_trace[i].second;
            if (incremental_gemma_trace[i].first != full_gemma_trace[i].first ||
                full_tensor.numel() < got_tensor.numel()) {
                throw std::runtime_error("incremental Gemma stage layout mismatch");
            }
            auto got = got_tensor.reshape({-1}).to(mfq_tensor_backend::kFloat64);
            auto full_flat = full_tensor.reshape({-1});
            auto ref = full_flat.narrow(
                0, full_flat.numel() - got_tensor.numel(), got_tensor.numel()).to(mfq_tensor_backend::kFloat64);
            const double denominator = std::max(ref.norm().template item<double>(), 1.0e-30);
            const double relative_l2 = (got - ref).norm().template item<double>() / denominator;
            const double cosine = mfq_tensor_backend::dot(got, ref).template item<double>() /
                std::max(got.norm().template item<double>() * denominator, 1.0e-30);
            std::cerr << "incremental_gemma_trace stage="
                      << incremental_gemma_trace[i].first
                      << " relative_l2=" << relative_l2
                      << " cosine=" << cosine << std::endl;
        }
        auto incremental_logits = model.lm_head.forward(
            incremental_hidden.index({Slice(), -1, Slice()}).to(mfq_tensor_backend::kFloat16).contiguous());
        auto full_logits = model.lm_head.forward(
            full_hidden.index({Slice(), -1, Slice()}).to(mfq_tensor_backend::kFloat16).contiguous());
        const double logits_relative_l2 =
            (incremental_logits.to(mfq_tensor_backend::kFloat64) - full_logits.to(mfq_tensor_backend::kFloat64)).norm().template item<double>() /
            std::max(full_logits.to(mfq_tensor_backend::kFloat64).norm().template item<double>(), 1.0e-30);
        std::cerr << "incremental_trace logits_relative_l2=" << logits_relative_l2
                  << " incremental_top=" << incremental_logits.argmax(-1).template item<int64_t>()
                  << " full_top=" << full_logits.argmax(-1).template item<int64_t>() << std::endl;
        return 1;
    }

    const char * graph_env = std::getenv("MFQ_RUNTIME_CUDA_GRAPH");
    const bool graph_enabled =
        (graph_env == nullptr || graph_env[0] != '0') &&
        !Model::is_flash_next &&
        mfq_cuda_graph_capture_supported() &&
        g_dsv4_cpu_offload_layers.empty() &&
        g_dense_cpu_layer_count == 0 &&
        !g_moe_expert_cache &&
        model_parallel_cuda_graph_enabled();
    const char * graph_min_env =
        std::getenv("MFQ_RUNTIME_CUDA_GRAPH_MIN_TOKENS");
    const int32_t graph_min_tokens = graph_min_env != nullptr
        ? std::max<int32_t>(2, std::atoi(graph_min_env))
        : 16;
    // Grammar state advances on the CPU and may require a one-off full-logit
    // mask, so constrained requests cannot be replayed as a fixed CUDA graph.
    // Unconstrained decode keeps the existing graph fast path unchanged.
    const bool graph_eligible = !transformed_prompt && graph_enabled && !reprefill &&
        !token_constraint &&
        sampling.max_tokens >= graph_min_tokens &&
        sampling.max_tokens <= graph_cache.generated_capacity;
    if (graph_eligible) {
        const bool greedy = sampler.greedy();
        auto [first, first_token] = sample_first_token();
        int32_t generated = 1;
        if (!on_token(first_token)) {
            store_live_history();
            return generated;
        }
        history.push_back(first_token);
        if (generated >= sampling.max_tokens) {
            store_live_history();
            return generated;
        }

        graph_cache.ensure_compute_streams();
        MfqCudaGuard graph_device_guard(
            graph_cache.stream.device_index());
        auto graph_stream_guards =
            activate_cuda_graph_compute_streams(
                graph_cache.compute_streams);
        cudaStream_t graph_raw_stream = graph_cache.stream.stream();
        if (has_penalties) {
            sample_token_counts_add_cuda(graph_cache.counts, first.contiguous());
        }

        int64_t pos_h = model.cache_pos;
        int64_t len_h = pos_h + 1;
        int64_t step_h = 1;
        MFQ_CUDA_CHECK(cudaMemcpyAsync(
            graph_cache.static_input.template data_ptr<int64_t>(), first.template data_ptr<int64_t>(),
            sizeof(int64_t), cudaMemcpyDeviceToDevice, graph_raw_stream));
        MFQ_CUDA_CHECK(cudaMemcpyAsync(
            graph_cache.generated.template data_ptr<int64_t>(), first.template data_ptr<int64_t>(),
            sizeof(int64_t), cudaMemcpyDeviceToDevice, graph_raw_stream));
        MFQ_CUDA_CHECK(cudaMemcpyAsync(
            graph_cache.static_pos.template data_ptr<int64_t>(), &pos_h,
            sizeof(int64_t), cudaMemcpyHostToDevice, graph_raw_stream));
        MFQ_CUDA_CHECK(cudaMemcpyAsync(
            graph_cache.static_len.template data_ptr<int64_t>(), &len_h,
            sizeof(int64_t), cudaMemcpyHostToDevice, graph_raw_stream));
        MFQ_CUDA_CHECK(cudaMemcpyAsync(
            graph_cache.static_step.template data_ptr<int64_t>(), &step_h,
            sizeof(int64_t), cudaMemcpyHostToDevice, graph_raw_stream));
        *random_host.template data_ptr<float>() = 0.5f;
        MFQ_CUDA_CHECK(cudaMemcpyAsync(
            graph_cache.random.template data_ptr<float>(), random_host.template data_ptr<float>(),
            sizeof(float), cudaMemcpyHostToDevice, graph_raw_stream));
        MFQ_CUDA_CHECK(cudaStreamSynchronize(graph_raw_stream));

        const int64_t requested_len = model.cache_pos + sampling.max_tokens;
        const int64_t planned_len = decode_graph_bucket(
            requested_len, model.max_position_embeddings());
        const int64_t attention_parts = decode_graph_attention_parts(
            planned_len, FullBlock::kDecodeAttentionMaxParts);
        auto sample_static = [&]() {
            if (greedy && !has_penalties) {
                return model.next_token_static(
                    graph_cache.static_input, graph_cache.static_pos,
                    graph_cache.static_len, planned_len, attention_parts);
            }
            auto logits = model.last_logits_static(
                    graph_cache.static_input, graph_cache.static_pos,
                    graph_cache.static_len, planned_len, attention_parts)
                .contiguous().view({1, -1});
            if (has_penalties) {
                logits = sampler.apply_penalties(
                    std::move(logits), graph_cache.counts);
            }
            if (greedy) {
                return sampler.ops().sample_greedy(std::move(logits));
            }
            return sampler.ops().sample_stochastic(
                std::move(logits), graph_cache.random, sampler.params());
        };

        const bool cache_hit = graph_cache.ensure_captured(
            model, planned_len, sampling, greedy,
            [&]() { return sample_static(); },
            [&](const mfq_tensor_backend::Tensor& next) {
                if (has_penalties) {
                    sample_token_counts_add_cuda(
                        graph_cache.counts, next.contiguous());
                }
                decode_graph_commit_cuda(
                    next, graph_cache.generated, graph_cache.static_step,
                    graph_cache.static_input, graph_cache.static_pos,
                    graph_cache.static_len);
            });
        if (!cache_hit) report_cuda_memory("runtime_graph_capture");
        if (trace_cuda_graph()) {
            std::cerr << "runtime_cuda_graph action=" << (cache_hit ? "reuse" : "capture")
                      << " requested_len=" << requested_len
                      << " planned_len=" << planned_len
                      << " captures=" << graph_cache.captures
                      << " reuses=" << graph_cache.reuses << std::endl;
        }

        while (generated < sampling.max_tokens) {
            if (!greedy) {
                *random_host.template data_ptr<float>() = sampler.next_uniform_float();
                MFQ_CUDA_CHECK(cudaMemcpyAsync(
                    graph_cache.random.template data_ptr<float>(), random_host.template data_ptr<float>(),
                    sizeof(float), cudaMemcpyHostToDevice, graph_raw_stream));
            }
            graph_cache.graph->replay();
            const int64_t token =
                graph_cache.static_next.template item<int64_t>();
            ++generated;
            if (!on_token(token)) break;
            history.push_back(token);
        }
        model.cache_pos += generated - 1;
        store_live_history();
        return generated;
    }

    int32_t generated = 0;
    while (generated < sampling.max_tokens) {
        if (reprefill && generated > 0) {
            model.reset(1);
            ids = mfq_tensor_backend::tensor(history, options).reshape({1, -1}).contiguous();
        }
        mfq_tensor_backend::Tensor next;
        int64_t token = 0;
        if (generated == 0) {
            auto first = sample_first_token();
            next = std::move(first.first);
            token = first.second;
        } else {
            next = sample_token(
                model, ids, sampler, counts, token_constraint);
            token = next.template item<int64_t>();
        }
        ++generated;
        if (!on_token(token)) break;
        history.push_back(token);
        if (has_penalties) sample_token_counts_add_cuda(counts, next.contiguous());
        ids = next.reshape({1, 1});
    }
    store_live_history();
    return generated;
}

#include "qwen_continuous_batching.h"

// Fixed synthetic-ID latency probe through the actual runtime dispatch. Model
// loading and the independent serial oracle are outside every timed request.
int run_qwen35_mtp_bench(
        mfq::cuda::Qwen35CausalLm& model, Qwen35Mtp& mtp,
        bool enable_mtp, int generated_tokens, int repetitions) {
    using Clock = std::chrono::steady_clock;
    using Tensor = mfq_tensor_backend::Tensor;
    MFQ_RUNTIME_CHECK(
        generated_tokens >= 2 && repetitions > 0 && repetitions <= 100 &&
            model.max_position_embeddings() >= generated_tokens + 17,
        "MTP benchmark requires gen>=2, reps1-100 and context>=gen+17");
    DecodeGraphCache graph_cache(model.max_position_embeddings());
    TextSessionCache session_cache;
    std::mutex model_mutex;
    MfqSamplingParams params;
    params.max_tokens = generated_tokens;
    params.temperature = 0.;
    params.top_k = 1;
    params.top_p = 1.;
    params.enable_mtp = enable_mtp;
    params.seed = 20260907;
    const auto options = mfq_tensor_backend::TensorOptions()
        .device(mfq_tensor_backend::kCUDA).dtype(mfq_tensor_backend::kInt64);
    const char* graph_env = std::getenv("MFQ_RUNTIME_CUDA_GRAPH");
    const char* profiler_env = std::getenv("MFQ_CUDA_PROFILER_RANGE");
    const bool profiler_range = profiler_env != nullptr && std::atoi(profiler_env) != 0;
    std::cout << "mtp_bench config mode=" << (enable_mtp ? "mtp" : "ordinary")
        << " full_window_batching=1"
        << " runtime_graph=" << (graph_env == nullptr ? "default" : graph_env)
        << " gen=" << generated_tokens << " reps=" << repetitions
        << " warmup=1 seed=20260907 synthetic_ids=1 eos_stop=0 session_cache=0\n";
    const auto ms_between = [](Clock::time_point first, Clock::time_point last) {
        return std::chrono::duration<double, std::milli>(last - first).count();
    };
    for (const auto& prompt : std::vector<std::vector<int64_t>>{
            {1, 2, 3}, {100, 200, 300, 400, 500, 600, 700}, std::vector<int64_t>(17, 10)}) {
        model.reset(1);
        Tensor input = mfq_tensor_backend::tensor(prompt, options).reshape({1, -1}).contiguous();
        std::vector<int64_t> reference;
        reference.reserve(generated_tokens);
        for (int step = 0; step < generated_tokens; ++step) {
            auto next = model.next_token(input);
            reference.push_back(next.template item<int64_t>());
            input = next.reshape({1, 1});
        }
        mfq_cuda_synchronize();
        for (int repeat = -1; repeat < repetitions; ++repeat) {
            std::vector<int64_t> output;
            output.reserve(generated_tokens);
            MfqPrefillTiming prefill;
            Clock::time_point first_token;
            mfq_cuda_synchronize();
            // Capture only the actual request; its independent oracle, full
            // warmup, model load and context construction remain unprofiled.
            const bool capture_request = profiler_range && repeat >= 0;
            if (capture_request) {
                std::cout << "mtp_bench profiler_begin prompt=" << prompt.size()
                    << " repeat=" << repeat << '\n';
                MFQ_CUDA_CHECK(cudaProfilerStart());
            }
            const auto started = Clock::now();
            const int produced = generate_tokens(model, model_mutex, graph_cache,
                session_cache, prompt, params, [&](int64_t token) {
                    if (output.empty()) first_token = Clock::now();
                    output.push_back(token);
                    return true;
                }, [&](const MfqPrefillTiming& timing) { prefill = timing; }, {}, {}, &mtp);
            mfq_cuda_synchronize();
            const auto finished = Clock::now();
            if (capture_request) {
                MFQ_CUDA_CHECK(cudaProfilerStop());
                std::cout << "mtp_bench profiler_end prompt=" << prompt.size()
                    << " repeat=" << repeat << '\n';
            }
            MFQ_RUNTIME_CHECK(produced == generated_tokens && output == reference,
                "MTP benchmark output differs from ordinary serial oracle");
            MFQ_RUNTIME_CHECK(!enable_mtp || mtp.last_cycles > 0,
                "MTP benchmark did not execute speculative cycles");
            const double total_ms = ms_between(started, finished);
            const double first_ms = ms_between(started, first_token);
            const double decode_ms = ms_between(first_token, finished);
            MFQ_RUNTIME_CHECK(std::isfinite(total_ms) && total_ms > 0. &&
                std::isfinite(decode_ms) && decode_ms > 0., "invalid benchmark clock interval");
            std::cout << "mtp_bench sample mode=" << (enable_mtp ? "mtp" : "ordinary")
                << " prompt=" << prompt.size() << " gen=" << produced
                << " repeat=" << repeat << " warmup=" << (repeat < 0)
                << " total_ms=" << total_ms << " ttft_ms=" << first_ms
                << " decode_ms=" << decode_ms << " prefill_gpu_ms=" << prefill.llm_ms
                << " total_tps=" << produced * 1000. / total_ms
                << " decode_tps=" << (produced - 1) * 1000. / decode_ms
                << " exact=1 cycles=" << (enable_mtp ? mtp.last_cycles : 0)
                << " accepted=" << (enable_mtp ? mtp.last_accepted : 0)
                << " rejected=" << (enable_mtp ? mtp.last_rejected : 0)
                << " graph_captures=" << graph_cache.captures
                << " graph_reuses=" << graph_cache.reuses << '\n';
        }
    }
    std::cout << "mtp_bench PASS samples=" << repetitions * 3 << '\n';
    return 0;
}

static int32_t generate_multimodal_tokens(
    MiniCPMO45Runtime & runtime,
    std::mutex & model_mutex,
    const std::vector<int64_t> & prompt,
    const MfqVisionInput & vision,
    const MfqSamplingParams & sampling,
    const MfqTokenCallback & on_token,
    const MfqPrefillCallback & on_prefill,
    const MfqTokenConstraintPtr & token_constraint)
{
    std::lock_guard<std::mutex> lock(model_mutex);
    if (prompt.empty() || sampling.max_tokens < 0) {
        throw std::invalid_argument(
            "MiniCPM-o multimodal generation input is invalid");
    }
    if (sampling.max_tokens == 0) {
        runtime.language.reset(1);
        return 0;
    }
    const auto generation_limit = std::min<int32_t>(
        sampling.max_tokens,
        static_cast<int32_t>(
            runtime.language.max_position_embeddings() -
            static_cast<int64_t>(prompt.size()) + 1));
    if (generation_limit <= 0) {
        throw std::invalid_argument(
            "MiniCPM-o multimodal prompt exceeds the context capacity");
    }

    const auto cpu_i64 =
        mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64).device(mfq_tensor_backend::kCPU);
    const auto cuda_i64 =
        mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64).device(mfq_tensor_backend::kCUDA);
    auto input_ids = mfq_tensor_backend::tensor(prompt, cuda_i64)
        .reshape({1, -1}).contiguous();
    mfq_tensor_backend::Tensor pixels;
    mfq_tensor_backend::Tensor patch_mask;
    mfq_tensor_backend::Tensor target_sizes;
    mfq_tensor_backend::Tensor image_bounds;
    if (!vision.image_bounds.empty()) {
        pixels = mfq_tensor_backend::from_blob(
            const_cast<float *>(vision.pixel_values.data()),
            vision.pixel_shape,
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kFloat32).device(mfq_tensor_backend::kCPU))
            .clone();
        patch_mask = mfq_tensor_backend::from_blob(
            const_cast<uint8_t *>(vision.patch_mask.data()),
            vision.patch_mask_shape,
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kUInt8).device(mfq_tensor_backend::kCPU))
            .clone().to(mfq_tensor_backend::kBool);
        target_sizes = mfq_tensor_backend::from_blob(
            const_cast<int32_t *>(vision.target_sizes.data()),
            vision.target_sizes_shape,
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt32).device(mfq_tensor_backend::kCPU))
            .clone();
        image_bounds = mfq_tensor_backend::from_blob(
            const_cast<int64_t *>(vision.image_bounds.data()),
            std::vector<int64_t>{
                static_cast<int64_t>(vision.image_bounds.size() / 4), 4},
            cpu_i64).clone();
    }
    mfq_tensor_backend::Tensor audio_features;
    mfq_tensor_backend::Tensor audio_lengths;
    mfq_tensor_backend::Tensor audio_bounds;
    if (!vision.audio_bounds.empty()) {
        audio_features = mfq_tensor_backend::from_blob(
            const_cast<float *>(vision.audio_features.data()),
            vision.audio_features_shape,
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kFloat32).device(mfq_tensor_backend::kCPU))
            .clone();
        audio_lengths = mfq_tensor_backend::tensor(
            vision.audio_lengths, cpu_i64).contiguous();
        audio_bounds = mfq_tensor_backend::from_blob(
            const_cast<int64_t *>(vision.audio_bounds.data()),
            std::vector<int64_t>{
                static_cast<int64_t>(vision.audio_bounds.size() / 4), 4},
            cpu_i64).clone();
    }

    auto random_host = mfq_tensor_backend::empty(
        {1}, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kFloat32)
            .device(mfq_tensor_backend::kCPU).pinned_memory(true));
    auto random_cuda = mfq_tensor_backend::empty(
        {1}, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kFloat32)
            .device(mfq_tensor_backend::kCUDA));
    mfq::cuda::Sampler sampler(
        sampling,
        mfq::cuda::SamplingOps(
            std::move(random_host), std::move(random_cuda)));
    const bool has_penalties = sampler.has_penalties();
    auto counts = has_penalties
        ? mfq_tensor_backend::zeros(
              {runtime.language.vocab_size()},
              mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt32).device(mfq_tensor_backend::kCUDA))
        : mfq_tensor_backend::Tensor();
    if (has_penalties) {
        sample_token_counts_add_cuda(counts, input_ids);
    }

    PrefillCudaTimer prefill_timer;
    auto result = runtime.forward(
        input_ids,
        mfq_tensor_backend::Tensor(),
        mfq_tensor_backend::Tensor(),
        pixels,
        patch_mask,
        target_sizes,
        image_bounds,
        audio_features,
        audio_lengths,
        audio_bounds);
    auto logits = result.logits.index({Slice(), -1, Slice()})
        .contiguous().view({1, -1});
    MFQ_CUDA_CHECK(cudaEventRecord(
        prefill_timer.finished_event(),
        mfq_get_current_cuda_stream()));
    auto next = mfq::cuda::sample_logits(
        sampler, std::move(logits), counts, token_constraint);
    if (on_prefill) {
        const double model_ms = prefill_timer.elapsed_ms();
        // The current CUDA composite timer covers both the multimodal encoder
        // and language prefill. Keep that total explicit instead of falsely
        // presenting it as comparable language-model-only time.
        on_prefill(MfqPrefillTiming{
            prompt.size(),
            0.0,
            0.0,
            model_ms});
    }

    int32_t generated = 0;
    while (generated < generation_limit) {
        const int64_t token = next.template item<int64_t>();
        ++generated;
        if (!on_token(token) || generated >= generation_limit) break;
        if (has_penalties) {
            sample_token_counts_add_cuda(counts, next.contiguous());
        }
        next = sample_token(
            runtime.language, next.reshape({1, 1}), sampler, counts,
            token_constraint);
    }
    return generated;
}

static MfqDuplexBackend make_cuda_minicpmo45_duplex_backend(
        MiniCPMO45Runtime & runtime,
        std::mutex & model_mutex,
        std::optional<MiniCPMO45DuplexSession> & session) {
    MfqDuplexBackend backend;
    backend.name = "cuda";
    backend.start = [&](const MfqDuplexSessionParams & parameters) {
        if (parameters.special_ids.size() != 15) {
            throw std::invalid_argument(
                "MiniCPM-o duplex requires 15 special token IDs");
        }
        if ((!parameters.greedy &&
                (!std::isfinite(parameters.temperature) ||
                 parameters.temperature <= 0.0)) ||
                parameters.top_k < 0 ||
                parameters.top_k >
                    std::min<int64_t>(
                        runtime.language.vocab_size(), 1024) ||
                !std::isfinite(parameters.top_p) ||
                parameters.top_p <= 0.0 || parameters.top_p > 1.0 ||
                !std::isfinite(parameters.listen_probability_scale) ||
                parameters.listen_probability_scale < 0.0 ||
                !std::isfinite(parameters.repetition_penalty) ||
                parameters.repetition_penalty <= 0.0 ||
                parameters.repetition_window <= 0 ||
                !std::isfinite(parameters.length_penalty) ||
                parameters.length_penalty <= 0.0 ||
                !std::isfinite(parameters.tts_temperature) ||
                parameters.tts_temperature <= 0.0 ||
                !std::isfinite(parameters.tts_repetition_penalty) ||
                parameters.tts_repetition_penalty <= 0.0) {
            throw std::invalid_argument(
                "MiniCPM-o duplex sampling configuration is invalid");
        }
        const int64_t audio_bos = parameters.special_ids.back();
        if (audio_bos < 0 || audio_bos >= 152064 ||
                std::any_of(
                    parameters.special_ids.begin(),
                    parameters.special_ids.end() - 1,
                    [&](int64_t token) {
                        return token < 0 ||
                            token >= runtime.language.vocab_size();
                    }) ||
                std::any_of(
                    parameters.forbidden_ids.begin(),
                    parameters.forbidden_ids.end(),
                    [&](int64_t token) {
                        return token < 0 ||
                            token >= runtime.language.vocab_size();
                    })) {
            throw std::invalid_argument(
                "MiniCPM-o duplex token ID is out of range");
        }
        if ((parameters.reference_audio_frames == 0) !=
                parameters.reference_audio_features.empty() ||
                parameters.reference_audio_frames < 0 ||
                (!parameters.reference_audio_features.empty() &&
                 (parameters.reference_audio_frames < 3 ||
                  parameters.reference_audio_features.size() !=
                    static_cast<size_t>(
                        parameters.reference_audio_frames) * 80))) {
            throw std::invalid_argument(
                "MiniCPM-o reference Mel geometry is invalid");
        }

        std::lock_guard<std::mutex> lock(model_mutex);
        MfqCudaGuard guard(
            g_layer_placement.primary_device());
        mfq_tensor_backend::manual_seed(static_cast<int64_t>(parameters.seed));
        mfq_cuda_manual_seed_all(parameters.seed);
        auto special_ids = MiniCPMO45DuplexSpecialIds::from_tensor(
            mfq_tensor_backend::tensor(
                parameters.special_ids,
                mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64)));
        session.reset();
        session.emplace(
            runtime, special_ids, parameters.forbidden_ids,
            parameters.greedy);
        session->temperature = parameters.temperature;
        session->top_k = parameters.top_k;
        session->top_p = parameters.top_p;
        session->listen_probability_scale =
            parameters.listen_probability_scale;
        session->repetition_penalty = parameters.repetition_penalty;
        session->repetition_window = parameters.repetition_window;
        session->length_penalty = parameters.length_penalty;
        session->tts_temperature = parameters.tts_temperature;
        session->tts_repetition_penalty =
            parameters.tts_repetition_penalty;

        const auto ids_tensor = [](const std::vector<int64_t> & values) {
            return values.empty()
                ? mfq_tensor_backend::Tensor()
                : mfq_tensor_backend::tensor(
                    values,
                    mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64));
        };
        mfq_tensor_backend::Tensor reference_features;
        if (!parameters.reference_audio_features.empty()) {
            reference_features = mfq_tensor_backend::tensor(
                parameters.reference_audio_features,
                mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kFloat32))
                .reshape({1, 80, parameters.reference_audio_frames});
        }
        session->prepare(
            ids_tensor(parameters.system_prefix),
            reference_features,
            ids_tensor(parameters.system_suffix));
        mfq_cuda_synchronize();
    };
    backend.step = [&](const MfqDuplexStepInput & input) {
        const bool has_audio = input.audio_frames > 0;
        const bool has_text = !input.text_tokens.empty();
        if (has_audio && input.audio_features.size() !=
                static_cast<size_t>(input.audio_frames) * 80) {
            throw std::invalid_argument(
                "MiniCPM-o duplex Mel geometry is invalid");
        }
        if (!has_audio && !has_text) {
            throw std::invalid_argument(
                "MiniCPM-o duplex step has no input");
        }
        if (input.max_new_speak_tokens < 2) {
            throw std::invalid_argument(
                "MiniCPM-o duplex generation requires at least two token slots");
        }

        std::lock_guard<std::mutex> lock(model_mutex);
        MfqCudaGuard guard(
            g_layer_placement.primary_device());
        if (!session) {
            throw std::runtime_error(
                "MiniCPM-o duplex session is not prepared");
        }
        mfq_tensor_backend::Tensor audio_features;
        if (has_audio) {
            audio_features = mfq_tensor_backend::tensor(
                input.audio_features,
                mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kFloat32))
                .reshape({1, 80, input.audio_frames});
        }
        mfq_tensor_backend::Tensor text_ids;
        if (has_text) {
            text_ids = mfq_tensor_backend::tensor(
                input.text_tokens,
                mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64))
                .reshape({1, static_cast<int64_t>(input.text_tokens.size())});
        }
        const auto started = std::chrono::steady_clock::now();
        auto result = session->run_step(
            {}, {}, {}, {}, audio_features,
            input.audio_prefix_extra_frames,
            input.audio_suffix_extra_frames,
            text_ids,
            input.max_new_speak_tokens,
            input.force_listen,
            input.force_speak);

        MfqDuplexStepResult response;
        auto generated = result.generated_ids
            .to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt64).contiguous().reshape({-1});
        const auto * generated_data = generated.template data_ptr<int64_t>();
        response.generated_tokens.assign(
            generated_data, generated_data + generated.numel());
        auto codes = result.tts_codes
            .to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt32).contiguous().reshape({-1});
        const auto * code_data = codes.template data_ptr<int32_t>();
        response.audio_tokens.assign(
            code_data, code_data + codes.numel());
        response.is_listen = result.is_listen;
        response.end_of_turn = result.end_of_turn;
        response.tts_force_flush = result.tts_force_flush;
        response.audio_chunk_index = session->audio_chunk_index;
        response.language_cache_position = runtime.language.cache_pos;
        response.audio_cache_position = runtime.audio.cache_length();
        response.tts_cache_position = runtime.tts.cache_position;
        response.inference_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count();
        return response;
    };
    backend.stop = [&]() {
        std::lock_guard<std::mutex> lock(model_mutex);
        MfqCudaGuard guard(
            g_layer_placement.primary_device());
        session.reset();
        runtime.language.reset(1);
        runtime.audio.reset();
        runtime.tts.reset(1);
        mfq_cuda_synchronize();
    };
    return backend;
}

namespace {
struct CudaInferenceEngine : mfq::cuda::RuntimeOptions {
    explicit CudaInferenceEngine(mfq::cuda::RuntimeOptions options)
        : mfq::cuda::RuntimeOptions(std::move(options)) {}
    int run() {
        if (transport_mode && context_size == 0) context_size = 32768;
        if (!cpu_offload_layers_arg.empty()) {
            g_dsv4_cpu_offload_layers =
                parse_layer_ranges(cpu_offload_layers_arg);
            std::vector<int> ordered(
                g_dsv4_cpu_offload_layers.begin(),
                g_dsv4_cpu_offload_layers.end());
            std::sort(ordered.begin(), ordered.end());
            std::cerr << "cpu_offload_layers=";
            for (size_t index = 0; index < ordered.size(); ++index) {
                if (index) std::cerr << ',';
                std::cerr << ordered[index];
            }
            std::cerr << std::endl;
        }
        g_profiler.enabled = false;
        mfq_tensor_backend::NoGradGuard no_grad;
        if (transport_mode) {
            auto source = mfq::open_model_source(model_path);
            if (!config_path.empty())
                throw std::runtime_error("model runtime does not accept an external model config");
            if (!source->has_asset(mfq::kModelConfigAsset) ||
                    (!source->has_asset(mfq::cuda::kTokenizerGgufAsset) && tokenizer_model.empty()))
                throw std::runtime_error("model runtime requires model config and tokenizer GGUF");
        }
        return with_loaded_cuda_model(*this, transport_mode,
            [&]<mfq::cuda::CudaBackbone Backbone>(auto& model,
                    auto& runtime_components, auto t0, auto t1) -> int {
        using Model = mfq::cuda::CausalLmFor<Backbone>;
        if (transport_mode) {
            Model& inference_model = runtime_components.language(model);
            if (transport_api_key.empty()) {
                const char * env_key = std::getenv("MFQ_API_KEY");
                if (env_key != nullptr) transport_api_key = env_key;
            }
            MfqHttpRuntimeTransportConfig transport_config;
            transport_config.host = transport_host;
            transport_config.port = transport_port;
            transport_config.model_name = runtime_model_name;
            transport_config.model_type = inference_model.model_type();
            const auto component_state = runtime_components.state();
            MfqModelCapabilities model_capabilities;
            model_capabilities.family = runtime_components.graph.architecture;
            model_capabilities.source = "model-graph+cuda-adapters";
            model_capabilities.text =
                runtime_components.graph.has_component("text") &&
                runtime_components.plan.backbone !=
                    mfq::cuda::CudaBackbone::unsupported;
            model_capabilities.image_input =
                component_state.vision_available;
            model_capabilities.video_input =
                component_state.vision_available &&
                !runtime_components.grid_vision.has_value();
            const bool composite_loaded =
                runtime_components.minicpmo.has_value();
            model_capabilities.audio_input = composite_loaded &&
                runtime_components.graph.has_component("audio_input");
            model_capabilities.audio_output = composite_loaded &&
                runtime_components.graph.has_component("audio_output");
            model_capabilities.full_duplex = composite_loaded &&
                runtime_components.graph.has_component("duplex");
            model_capabilities.mtp = component_state.mtp_available &&
                continuous_batching == 0;
            transport_config.model_capabilities = std::move(model_capabilities);
            const auto& runtime_assets = *inference_model.source;
            if (runtime_assets.has_asset(mfq::cuda::kTokenizerGgufAsset)) {
                transport_config.tokenizer_gguf =
                    read_asset(runtime_assets, mfq::cuda::kTokenizerGgufAsset);
            } else if (!tokenizer_model.empty()) {
                transport_config.tokenizer_model = tokenizer_model;
            } else {
                throw std::runtime_error(
                    "model runtime requires a tokenizer GGUF");
            }
            transport_config.api_key = transport_api_key;
            transport_config.max_context =
                inference_model.max_position_embeddings();
            transport_config.vocab_size = inference_model.vocab_size();
            const auto embedded_profile = runtime_assets.metadata().find(
                "runtime.sampling.v1");
            transport_config.runtime_profile = resolve_mfq_runtime_profile(
                model_path,
                runtime_components.graph.architecture,
                transport_config.model_type,
                transport_config.model_name,
                embedded_profile == runtime_assets.metadata().end()
                    ? std::string()
                    : embedded_profile->second,
                runtime_assets.has_asset(mfq::kModelConfigAsset)
                    ? read_asset_text(runtime_assets, mfq::kModelConfigAsset)
                    : std::string(),
                runtime_sampling_profile);
            std::mutex model_mutex;
            DecodeGraphCache decode_graph_cache(
                inference_model.max_position_embeddings());
            const bool session_cache_supported =
                inference_model.supports_text_session_state() &&
                continuous_batching == 0;
            TextSessionCache text_session_cache(
                make_cuda_paged_prefix_cache(
                    runtime_assets, inference_model),
                session_cache_supported,
                inference_model.supports_text_session_state()
                    ? (continuous_batching == 0 ? 0 : 2)
                    : 1);
            std::unique_ptr<
                mfq::cuda::continuous::CudaContinuousBatcher>
                continuous_batcher;
            if (continuous_batching > 0) {
                if constexpr (
                        Backbone == mfq::cuda::CudaBackbone::generic_qwen) {
                    continuous_batcher = std::make_unique<
                        mfq::cuda::continuous::CudaContinuousBatcher>(
                            inference_model, model_mutex,
                            continuous_batching,
                            prefill_chunk_size);
                    std::cerr
                        << "continuous_batching enabled=1 max_sequences="
                        << continuous_batching
                        << " prefill_chunk_size=" << prefill_chunk_size
                        << " decode=target_only mtp=disabled"
                        << " moe="
                        << (mfq::cuda::continuous::
                            qwen_continuous_batch_has_moe(inference_model) ? 1 : 0)
                        << " moe_expert_cache="
                        << (mfq::cuda::continuous::
                            qwen_continuous_batch_has_cached_moe(inference_model)
                                ? 1 : 0)
                        << " paged_kv="
                        << (continuous_batcher->paged_kv_enabled() ? 1 : 0)
                        << " page_size="
                        << continuous_batcher->paged_kv_page_size()
                        << " prefix_cache=fresh_prefill\n";
                } else {
                    throw std::runtime_error(
                        "continuous batching requires Qwen35CausalLm");
                }
            }
            std::optional<MiniCPMO45DuplexSession> minicpmo_duplex_session;
            MfqDuplexBackend duplex_backend;
            if (runtime_components.minicpmo) {
                duplex_backend = make_cuda_minicpmo45_duplex_backend(
                    *runtime_components.minicpmo,
                    model_mutex,
                    minicpmo_duplex_session);
            }
            MfqMultimodalGenerateFn multimodal_generate;
            if (runtime_components.minicpmo) {
                multimodal_generate =
                    [&](const std::vector<int64_t> & prompt,
                        const MfqVisionInput & vision,
                        const MfqSamplingParams & sampling,
                        const MfqTokenCallback & on_token,
                        const MfqPrefillCallback & on_prefill,
                        const MfqPromptCachePlan &,
                        const MfqTokenConstraintPtr & token_constraint) {
                        return generate_multimodal_tokens(
                            *runtime_components.minicpmo,
                            model_mutex,
                            prompt,
                            vision,
                            sampling,
                            on_token,
                            on_prefill,
                            token_constraint);
                    };
            } else if (runtime_components.grid_vision) {
                multimodal_generate =
                    [&](const std::vector<int64_t>& prompt,
                        const MfqVisionInput& vision,
                        const MfqSamplingParams& sampling,
                        const MfqTokenCallback& on_token,
                        const MfqPrefillCallback& on_prefill,
                        const MfqPromptCachePlan& cache_plan,
                        const MfqTokenConstraintPtr& token_constraint) {
                        return generate_tokens<Model>(
                            inference_model, model_mutex, decode_graph_cache,
                            text_session_cache, prompt, sampling, on_token,
                            on_prefill, cache_plan, token_constraint,
                            continuous_batching == 0
                                ? runtime_components.mtp.get()
                                : nullptr,
                            prefill_chunk_size, [&](Model& language) {
                                return std::optional<CudaPreparedPrompt>{
                                    runtime_components.grid_vision->prepare(
                                        language, prompt, vision)};
                            });
                    };
            }
            MfqInferenceEngine inference_engine;
            inference_engine.generate =
                [&](const std::vector<int64_t> & prompt,
                                   const MfqSamplingParams & sampling,
                                   const MfqTokenCallback & on_token,
                                   const MfqPrefillCallback & on_prefill,
                                   const MfqPromptCachePlan & cache_plan,
                                   const MfqTokenConstraintPtr & token_constraint) {
                if (continuous_batcher) {
                    return continuous_batcher->submit(
                        prompt, sampling, on_token, on_prefill,
                        cache_plan, token_constraint);
                }
                return generate_tokens(
                    inference_model, model_mutex, decode_graph_cache,
                    text_session_cache, prompt, sampling,
                    on_token, on_prefill, cache_plan, token_constraint,
                    runtime_components.mtp.get(), prefill_chunk_size);
            };
            inference_engine.duplex = duplex_backend;
            inference_engine.session_control = {
                [&](const std::string & source_session_id,
                        const std::string & target_session_id) {
                    std::lock_guard<std::mutex> lock(model_mutex);
                    return text_session_cache.fork_session(
                        source_session_id, target_session_id);
                },
                [&](const std::string & session_id) {
                    std::lock_guard<std::mutex> lock(model_mutex);
                    return text_session_cache.close_session(session_id);
                },
                [&] {
                    return text_session_cache.metrics();
                },
                [&] {
                    std::lock_guard<std::mutex> lock(model_mutex);
                    return text_session_cache.clear();
                },
                [&](uint64_t target_bytes) {
                    return text_session_cache.trim_hot(target_bytes);
                },
            };
            inference_engine.multimodal_generate = multimodal_generate;
            inference_engine.runtime_metrics =
            [&runtime_components, &continuous_batcher, &model_mutex] {
                size_t free_bytes = 0;
                size_t total_bytes = 0;
                MFQ_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
                const auto stats = mfq_cuda_memory_stats(
                    mfq_current_cuda_device());
                const auto components = runtime_components.state();
                const bool mtp_available =
                    components.mtp_available && !continuous_batcher;
                std::vector<std::pair<std::string, double>> result{
                    {"device_free_bytes", static_cast<double>(free_bytes)},
                    {"device_total_bytes", static_cast<double>(total_bytes)},
                    {"cuda_allocated_bytes", static_cast<double>(
                        stats.allocated_bytes)},
                    {"cuda_reserved_bytes", static_cast<double>(
                        stats.reserved_bytes)},
                    {"vision_declared", components.vision_declared ? 1.0 : 0.0},
                    {"vision_supported", components.vision_supported ? 1.0 : 0.0},
                    {"vision_available", components.vision_available ? 1.0 : 0.0},
                    {"mtp_declared", components.mtp_declared ? 1.0 : 0.0},
                    {"mtp_supported", components.mtp_supported ? 1.0 : 0.0},
                    {"mtp_available", mtp_available ? 1.0 : 0.0},
                };
                std::unique_lock lock(model_mutex, std::try_to_lock);
                if (mtp_available && lock.owns_lock() &&
                    runtime_components.mtp) {
                    const auto& mtp_stats = runtime_components.mtp->last_stats;
                    result.emplace_back(
                        "mtp_used", mtp_stats.used ? 1.0 : 0.0);
                    result.emplace_back(
                        "mtp_cycles", static_cast<double>(mtp_stats.cycles));
                    result.emplace_back(
                        "mtp_drafted_tokens",
                        static_cast<double>(mtp_stats.drafted_tokens));
                    result.emplace_back(
                        "mtp_accepted_tokens",
                        static_cast<double>(mtp_stats.accepted_tokens));
                    result.emplace_back(
                        "mtp_acceptance_rate",
                        mtp_stats.drafted_tokens == 0
                            ? 0.0
                            : static_cast<double>(
                                  mtp_stats.accepted_tokens) /
                                  mtp_stats.drafted_tokens);
                    result.emplace_back(
                        "mtp_selected_depth",
                        static_cast<double>(mtp_stats.selected_depth));
                    for (std::size_t depth = 0;
                         depth < mtp_stats.depth_cycles.size(); ++depth) {
                        result.emplace_back(
                            "mtp_depth_" + std::to_string(depth) + "_cycles",
                            static_cast<double>(
                                mtp_stats.depth_cycles[depth]));
                    }
                    for (std::size_t position = 0;
                         position < mtp_stats.position_drafted.size();
                         ++position) {
                        result.emplace_back(
                            "mtp_position_" + std::to_string(position + 1) +
                                "_acceptance_rate",
                            mtp_stats.position_drafted[position] == 0
                                ? 0.0
                                : static_cast<double>(
                                      mtp_stats.position_accepted[position]) /
                                      mtp_stats.position_drafted[position]);
                    }
                    for (std::size_t depth = 0;
                         depth < mtp_stats.measured_depth_ms.size(); ++depth) {
                        result.emplace_back(
                            "mtp_depth_" + std::to_string(depth) +
                                "_cycle_ms",
                            mtp_stats.measured_depth_ms[depth]);
                    }
                }
                if (continuous_batcher) {
                    auto batching = continuous_batcher->metrics();
                    result.insert(
                        result.end(), batching.begin(), batching.end());
                }
                return result;
            };
            auto transport = stdio_mode
                ? make_mfq_stdio_transport(transport_config)
                : make_mfq_http_transport(transport_config);
            MfqRuntime runtime(
                std::move(inference_engine), std::move(transport));
            const int status = runtime.run();
            if (g_moe_expert_cache) {
                print_moe_expert_cache_stats(std::cout);
            }
            return status;
        }
        auto ids_vec = ids_file.empty() ? parse_ids(ids_arg) : load_ids_file(ids_file);
        auto ids = mfq_tensor_backend::tensor(ids_vec, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64).device(mfq_tensor_backend::kCUDA)).unsqueeze(0);
        return generate_cli_tokens(model, ids, gen, false, t0, t1);
            });
    }
};

} // namespace

int mfq::cuda::run_cuda_inference(RuntimeOptions options) {
    return CudaInferenceEngine(std::move(options)).run();
}

int mfq::cuda::run_cuda_minicpmo_composite(
        const RuntimeOptions& options) {
    return run_minicpmo45_composite(
        options.model_path, options.config_path,
        options.minicpmo_input_prefix, options.minicpmo_output_prefix,
        options.context_size, options.minicpmo_tts_steps);
}

int mfq::cuda::run_cuda_minicpmo_duplex(
        const RuntimeOptions& options) {
    return run_minicpmo45_duplex(
        options.model_path, options.config_path,
        options.minicpmo_duplex_input_prefix,
        options.minicpmo_duplex_output_prefix,
        options.context_size, options.minicpmo_duplex_steps,
        options.minicpmo_duplex_max_speak_tokens,
        options.minicpmo_duplex_greedy,
        options.minicpmo_duplex_seed);
}

int run_cuda_continuous_batching_check(
        mfq::cuda::Qwen35CausalLm& model) {
    return mfq::cuda::continuous::run_qwen_continuous_batching_check(model);
}
