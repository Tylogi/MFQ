#pragma once

#include "mfq_paged_prefix_cache.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace mfq::engine {

// Logical session ownership for a backend-provided paged cache codec.
class PagedSessionBindings {
public:
    PagedSessionBindings(
        std::shared_ptr<mfq::cache::PagedPrefixCache> cache,
        std::size_t max_sessions);

    void bind(
        const std::string& session_id,
        std::vector<mfq::cache::BlockHash> blocks,
        std::size_t tokens);
    std::size_t fork(
        const std::string& source_session,
        const std::string& target_session);
    std::size_t close(const std::string& session_id);
    std::size_t clear() noexcept;

    std::size_t sessions() const noexcept { return sessions_.load(); }
    std::size_t tokens() const noexcept { return tokens_.load(); }

private:
    struct Binding {
        std::vector<mfq::cache::BlockHash> blocks;
        std::size_t tokens = 0;
        std::uint64_t last_used = 0;
    };

    void sync_metrics() noexcept;

    std::shared_ptr<mfq::cache::PagedPrefixCache> cache_;
    std::size_t max_sessions_ = 0;
    std::unordered_map<std::string, Binding> bindings_;
    std::uint64_t clock_ = 0;
    std::atomic<std::size_t> sessions_{0};
    std::atomic<std::size_t> tokens_{0};
};

} // namespace mfq::engine
