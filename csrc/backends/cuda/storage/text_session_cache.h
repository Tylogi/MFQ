#pragma once

#include "../native/tensor_backend.h"
#include "step_sequence.h"
#include <variant>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

struct MtpModule;
struct TextSessionState;

namespace mfq {
class ModelSource;
namespace cache {
class PagedPrefixCache;
}
}

namespace mfq::engine {
struct PrefixCacheConfig;
struct SessionCacheConfig;
}

namespace mfq::cuda::internal {

struct TextSessionRestore {
    std::size_t tokens = 0;
    mfq_tensor_backend::Tensor mtp_last_target_hidden;
};

std::shared_ptr<mfq::cache::PagedPrefixCache>
make_cuda_paged_prefix_cache(
    const mfq::ModelSource& source,
    std::int64_t max_position_embeddings,
    bool supports_paged_text_session_state,
    const mfq::engine::PrefixCacheConfig& config);

class TextSessionCache {
public:
    explicit TextSessionCache(
        const mfq::engine::SessionCacheConfig& session_config,
        const mfq::engine::PrefixCacheConfig& prefix_config,
        std::shared_ptr<mfq::cache::PagedPrefixCache> paged_cache = {},
        bool supported = true,
        int disabled_reason = 0);
    ~TextSessionCache();

    TextSessionCache(const TextSessionCache&) = delete;
    TextSessionCache& operator=(const TextSessionCache&) = delete;

    void limit_snapshot_bytes(std::size_t bytes);

    bool persistent_prefix_enabled() const noexcept;

    template <typename Model>
    TextSessionRestore restore_best(
        Model& model,
        MtpModule* mtp,
        const std::string& requested_session,
        const std::vector<std::int64_t>& prompt,
        std::size_t maximum_prefix_tokens,
        const std::string& input_key = {});

    template <typename Model>
    mfq::StepSequence<TextSessionRestore> restore_steps(Model& model, MtpModule* mtp,
        const std::string& requested_session, const std::vector<std::int64_t>& prompt,
        std::size_t maximum_prefix_tokens, const std::string& input_key = {});
    mfq::StepSequence<std::monostate> store_steps(const std::string& session_id, TextSessionState state);

    void store(const std::string& session_id, TextSessionState state);
    std::size_t fork_session(
        const std::string& source_session,
        const std::string& target_session);
    std::size_t close_session(const std::string& session_id);
    std::vector<std::pair<std::string, double>> metrics() const;
    std::size_t clear_live_sessions() noexcept;
    std::size_t clear();
    std::uint64_t trim_hot(std::uint64_t target_bytes);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mfq::cuda::internal
