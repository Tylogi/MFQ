#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mfq::engine {

struct SessionSnapshotCacheConfig {
    std::size_t max_sessions = 4;
    std::size_t max_snapshots_per_session = 4;
    std::size_t max_bytes = 2ULL * 1024ULL * 1024ULL * 1024ULL;
};

struct SessionSnapshotCacheMetrics {
    std::uint64_t queries = 0;
    std::uint64_t hits = 0;
    std::uint64_t hit_tokens = 0;
    std::size_t sessions = 0;
    std::size_t snapshots = 0;
    std::size_t tokens = 0;
    std::size_t bytes = 0;
};

// State owns backend data and exposes tokens and bytes. Matching and eviction
// are engine policy; restoring and capturing state remain backend operations.
template <typename State>
class SessionSnapshotCache {
    struct Entry {
        State state;
        std::uint64_t last_used = 0;
    };

public:
    struct Match {
        State* state = nullptr;
        std::string session_id;
        std::size_t index = 0;

        std::size_t tokens() const noexcept {
            return state ? state->tokens.size() : 0;
        }

        explicit operator bool() const noexcept { return state != nullptr; }
    };

    struct StoreResult {
        const State* state = nullptr;
        std::size_t session_snapshots = 0;
        std::size_t total_bytes = 0;

        explicit operator bool() const noexcept { return state != nullptr; }
    };

    struct CloseResult {
        std::size_t snapshots = 0;
        std::size_t bytes = 0;
    };

    explicit SessionSnapshotCache(SessionSnapshotCacheConfig config = {})
        : config_(config) {}

    bool enabled() const noexcept {
        return config_.max_sessions > 0 &&
            config_.max_snapshots_per_session > 0 && config_.max_bytes > 0;
    }

    std::size_t max_sessions() const noexcept { return config_.max_sessions; }
    std::size_t max_snapshots_per_session() const noexcept {
        return config_.max_snapshots_per_session;
    }
    std::size_t max_bytes() const noexcept { return config_.max_bytes; }

    template <typename Eligible>
    std::optional<Match> find_best(
            const std::string& requested_session,
            const std::vector<std::int64_t>& prompt,
            std::size_t maximum_prefix_tokens,
            Eligible&& eligible) {
        if (requested_session.empty() || !enabled()) return std::nullopt;
        ++queries_;
        std::string selected_session;
        std::size_t selected_snapshot = 0;
        std::size_t selected_tokens = 0;
        for (auto& [session_id, history] : states_) {
            for (std::size_t index = 0; index < history.size(); ++index) {
                auto& state = history[index].state;
                const auto& tokens = state.tokens;
                if (tokens.empty() || tokens.size() >= prompt.size() ||
                        tokens.size() > maximum_prefix_tokens ||
                        tokens.size() < selected_tokens || !eligible(state) ||
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
        if (selected_session.empty()) return std::nullopt;
        return Match{
            &states_.at(selected_session)[selected_snapshot].state,
            std::move(selected_session),
            selected_snapshot};
    }

    void record_hit(const Match& match) {
        auto found = states_.find(match.session_id);
        if (found == states_.end() || match.index >= found->second.size() ||
                &found->second[match.index].state != match.state) {
            throw std::logic_error("session snapshot match is stale");
        }
        found->second[match.index].last_used = ++clock_;
        ++hits_;
        hit_tokens_ += match.tokens();
    }

    void erase(const Match& match) {
        erase_snapshot(match.session_id, match.index);
    }

    template <typename SameSnapshot>
    StoreResult store(
            const std::string& session_id,
            State state,
            SameSnapshot&& same_snapshot) {
        if (session_id.empty() || !enabled() || state.bytes > config_.max_bytes) {
            return {};
        }
        Entry entry{std::move(state), ++clock_};
        const auto protected_clock = entry.last_used;
        auto& history = states_[session_id];
        auto previous = std::find_if(
            history.begin(), history.end(),
            [&](const Entry& saved) {
                return same_snapshot(saved.state, entry.state);
            });
        if (previous != history.end()) {
            bytes_ -= previous->state.bytes;
            *previous = std::move(entry);
        } else {
            history.push_back(std::move(entry));
        }
        const auto stored = std::find_if(
            history.begin(), history.end(),
            [&](const Entry& saved) {
                return saved.last_used == protected_clock;
            });
        if (stored == history.end()) {
            throw std::logic_error("stored session snapshot is unavailable");
        }
        bytes_ += stored->state.bytes;
        evict_history_to_limit(session_id, protected_clock);
        evict_to_budget(session_id, protected_clock);
        sync_metrics();
        const auto& saved_history = states_.at(session_id);
        const auto saved = std::find_if(
            saved_history.begin(), saved_history.end(),
            [&](const Entry& candidate) {
                return candidate.last_used == protected_clock;
            });
        if (saved == saved_history.end()) {
            throw std::logic_error("protected session snapshot was evicted");
        }
        return {&saved->state, saved_history.size(), bytes_};
    }

    std::size_t fork(
            const std::string& source_session,
            const std::string& target_session) {
        if (source_session.empty() || target_session.empty() ||
                source_session == target_session || !enabled()) {
            return 0;
        }
        const auto source = states_.find(source_session);
        if (source == states_.end()) return 0;
        auto copied = source->second;
        (void)close(target_session);
        auto& target = states_[target_session];
        std::uint64_t protected_clock = 0;
        for (auto& snapshot : copied) {
            snapshot.last_used = ++clock_;
            protected_clock = snapshot.last_used;
            bytes_ += snapshot.state.bytes;
            target.push_back(std::move(snapshot));
        }
        evict_history_to_limit(target_session, protected_clock);
        evict_to_budget(target_session, protected_clock);
        const auto remaining = states_.find(target_session);
        const auto copied_snapshots = remaining == states_.end()
            ? 0 : remaining->second.size();
        sync_metrics();
        return copied_snapshots;
    }

    CloseResult close(const std::string& session_id) {
        auto found = states_.find(session_id);
        if (found == states_.end()) return {};
        CloseResult released{found->second.size(), 0};
        for (const auto& snapshot : found->second) {
            released.bytes += snapshot.state.bytes;
        }
        bytes_ -= released.bytes;
        states_.erase(found);
        sync_metrics();
        return released;
    }

    std::size_t clear() noexcept {
        std::size_t snapshots = 0;
        for (const auto& [session_id, history] : states_) {
            (void)session_id;
            snapshots += history.size();
        }
        states_.clear();
        bytes_ = 0;
        sync_metrics();
        return snapshots;
    }

    SessionSnapshotCacheMetrics metrics() const noexcept {
        return {
            queries_.load(),
            hits_.load(),
            hit_tokens_.load(),
            metric_sessions_.load(),
            metric_snapshots_.load(),
            metric_tokens_.load(),
            metric_bytes_.load(),
        };
    }

private:
    void evict_history_to_limit(
            const std::string& session_id,
            std::uint64_t protected_clock) {
        auto found = states_.find(session_id);
        while (found != states_.end() &&
                found->second.size() > config_.max_snapshots_per_session) {
            std::size_t victim = found->second.size();
            for (std::size_t index = 0; index < found->second.size(); ++index) {
                const auto& snapshot = found->second[index];
                if (snapshot.last_used == protected_clock) continue;
                if (victim == found->second.size() ||
                        snapshot.last_used < found->second[victim].last_used) {
                    victim = index;
                }
            }
            if (victim == found->second.size()) break;
            erase_snapshot(session_id, victim);
            found = states_.find(session_id);
        }
    }

    void evict_to_budget(
            const std::string& protected_session,
            std::uint64_t protected_clock) {
        while (states_.size() > config_.max_sessions) {
            auto victim = states_.end();
            std::uint64_t victim_last_used = 0;
            for (auto iterator = states_.begin();
                    iterator != states_.end(); ++iterator) {
                if (iterator->first == protected_session) continue;
                std::uint64_t session_last_used = 0;
                for (const auto& snapshot : iterator->second) {
                    session_last_used = std::max(
                        session_last_used, snapshot.last_used);
                }
                if (victim == states_.end() ||
                        session_last_used < victim_last_used) {
                    victim = iterator;
                    victim_last_used = session_last_used;
                }
            }
            if (victim == states_.end()) break;
            (void)close(victim->first);
        }
        while (bytes_ > config_.max_bytes) {
            std::string victim_session;
            std::size_t victim_snapshot = 0;
            std::uint64_t victim_last_used = 0;
            bool found_victim = false;
            for (const auto& [session_id, history] : states_) {
                for (std::size_t index = 0; index < history.size(); ++index) {
                    const auto& snapshot = history[index];
                    if (session_id == protected_session &&
                            snapshot.last_used == protected_clock) {
                        continue;
                    }
                    if (!found_victim || snapshot.last_used < victim_last_used) {
                        victim_session = session_id;
                        victim_snapshot = index;
                        victim_last_used = snapshot.last_used;
                        found_victim = true;
                    }
                }
            }
            if (!found_victim) break;
            erase_snapshot(victim_session, victim_snapshot);
        }
    }

    void erase_snapshot(const std::string& session_id, std::size_t index) {
        auto found = states_.find(session_id);
        if (found == states_.end() || index >= found->second.size()) return;
        bytes_ -= found->second[index].state.bytes;
        found->second.erase(found->second.begin() +
            static_cast<std::ptrdiff_t>(index));
        if (found->second.empty()) states_.erase(found);
        sync_metrics();
    }

    void sync_metrics() noexcept {
        std::size_t snapshots = 0;
        std::size_t tokens = 0;
        for (const auto& [session_id, history] : states_) {
            (void)session_id;
            snapshots += history.size();
            for (const auto& snapshot : history) {
                tokens += snapshot.state.tokens.size();
            }
        }
        metric_sessions_.store(states_.size());
        metric_snapshots_.store(snapshots);
        metric_tokens_.store(tokens);
        metric_bytes_.store(bytes_);
    }

    SessionSnapshotCacheConfig config_;
    std::unordered_map<std::string, std::vector<Entry>> states_;
    std::size_t bytes_ = 0;
    std::uint64_t clock_ = 0;
    std::atomic<std::uint64_t> queries_{0};
    std::atomic<std::uint64_t> hits_{0};
    std::atomic<std::uint64_t> hit_tokens_{0};
    std::atomic<std::size_t> metric_sessions_{0};
    std::atomic<std::size_t> metric_snapshots_{0};
    std::atomic<std::size_t> metric_tokens_{0};
    std::atomic<std::size_t> metric_bytes_{0};
};

} // namespace mfq::engine
