#include "paged_session_bindings.h"

#include <utility>

namespace mfq::engine {

PagedSessionBindings::PagedSessionBindings(
        std::shared_ptr<mfq::cache::PagedPrefixCache> cache,
        std::size_t max_sessions)
    : cache_(std::move(cache)), max_sessions_(max_sessions) {}

void PagedSessionBindings::bind(
        const std::string& session_id,
        std::vector<mfq::cache::BlockHash> blocks,
        std::size_t tokens) {
    if (!cache_ || session_id.empty() || max_sessions_ == 0) return;
    close(session_id);
    bindings_[session_id] = Binding{
        std::move(blocks), tokens, ++clock_};
    while (bindings_.size() > max_sessions_) {
        auto victim = bindings_.end();
        for (auto candidate = bindings_.begin();
                candidate != bindings_.end(); ++candidate) {
            if (candidate->first == session_id) continue;
            if (victim == bindings_.end() ||
                    candidate->second.last_used < victim->second.last_used) {
                victim = candidate;
            }
        }
        if (victim == bindings_.end()) break;
        close(victim->first);
    }
    sync_metrics();
}

std::size_t PagedSessionBindings::fork(
        const std::string& source_session,
        const std::string& target_session) {
    const auto source = bindings_.find(source_session);
    if (source == bindings_.end() || source_session.empty() ||
            target_session.empty() || source_session == target_session) {
        return 0;
    }
    const auto blocks = source->second.blocks;
    const auto tokens = source->second.tokens;
    const auto count = blocks.size();
    bind(target_session, blocks, tokens);
    return count;
}

std::size_t PagedSessionBindings::close(const std::string& session_id) {
    const auto found = bindings_.find(session_id);
    if (found == bindings_.end()) return 0;
    const auto blocks = found->second.blocks.size();
    bindings_.erase(found);
    sync_metrics();
    return blocks;
}

std::size_t PagedSessionBindings::clear() noexcept {
    const auto sessions = bindings_.size();
    bindings_.clear();
    sync_metrics();
    return sessions;
}

void PagedSessionBindings::sync_metrics() noexcept {
    std::size_t tokens = 0;
    for (const auto& [session, binding] : bindings_) {
        (void)session;
        tokens += binding.tokens;
    }
    sessions_.store(bindings_.size());
    tokens_.store(tokens);
}

} // namespace mfq::engine
