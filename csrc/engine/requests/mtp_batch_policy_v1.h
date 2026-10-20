#pragma once

// CUDA keeps the measured pre-7d57bd30 policy until the updated scheduler is validated.

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <deque>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace mfq::engine::mtp::v1 {

class BatchDepthPolicy {
public:
    using Clock = std::chrono::steady_clock;
    BatchDepthPolicy(std::vector<std::string> requests, int maximum, std::size_t ordinary_requests = 0)
        : requests_(std::move(requests)), ordinary_requests_(ordinary_requests),
          maximum_(std::clamp(maximum, 1, 5)), current_(maximum_) {}
    const auto& requests() const noexcept { return requests_; }
    std::size_t ordinary_requests() const noexcept { return ordinary_requests_; }
    int maximum() const noexcept { return maximum_; }
    void interrupt() noexcept { finished_.reset(); last_depth_ = -1; }
    int depth() const noexcept {
        return standard_warmup_ || standard_.size() < 3 || remaining_ ? 0 : current_;
    }
    std::optional<double> cost(int depth) const {
        const auto& values = depth ? costs_.at(depth) : standard_;
        if (values.empty()) return std::nullopt;
        std::vector<double> sorted(values.begin(), values.end());
        std::sort(sorted.begin(), sorted.end());
        const auto middle = sorted.size() / 2;
        return sorted.size() % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) * .5;
    }
    void observe(int depth, std::span<const int> accepted, Clock::time_point started,
        Clock::time_point finished, bool stable, bool interrupted) {
        if (!stable || interrupted) { interrupt(); return; }
        const auto begin = last_depth_ == depth && finished_ ? *finished_ : started;
        const bool same_depth = last_depth_ == depth;
        finished_ = finished;
        last_depth_ = depth;
        const auto ms = std::max(1e-6, std::chrono::duration<double, std::milli>(finished - begin).count());
        if (!depth) {
            if (standard_warmup_) --standard_warmup_;
            else append(standard_, ms);
            if (remaining_ && !--remaining_) {
                costs_ = {};
                last_seen_ = {};
                acceptance_.fill(.6);
                current_ = maximum_;
                losing_ = clear_losing_ = 0;
            }
            return;
        }
        for (int position = 0; position < depth; ++position) {
            int reached = 0, hits = 0;
            for (int count : accepted) {
                reached += count >= position;
                hits += count > position;
            }
            if (!reached) break;
            acceptance_[position] += .08 * (double(hits) / reached - acceptance_[position]);
        }
        elapsed_ms_ += ms;
        if (!same_depth) return;
        append(costs_[depth], ms);
        last_seen_[depth] = elapsed_ms_;
        for (int candidate = maximum_; candidate > 0; --candidate) if (costs_[candidate].size() < 3) {
            current_ = candidate;
            return;
        }
        ++decisions_;
        int best = 1;
        for (int candidate = 2; candidate <= maximum_; ++candidate)
            if (score(candidate) > score(best)) best = candidate;
        if (score(best) > score(current_) * 1.03) current_ = best;
        losing_ = score(best) <= score(0) * 1.03 ? losing_ + 1 : 0;
        clear_losing_ = score(best) < score(0) * .9 ? clear_losing_ + 1 : 0;
        if (elapsed_ms_ - last_probe_ms_ >= 1000) {
            int rival = 1;
            for (int candidate = 2; candidate <= maximum_; ++candidate)
                if (last_seen_[candidate] < last_seen_[rival]) rival = candidate;
            if (rival != current_ && (score(rival) * 1.15 >= score(best) ||
                elapsed_ms_ - last_seen_[rival] >= 5000)) {
                current_ = rival;
                last_probe_ms_ = elapsed_ms_;
            }
        }
        if (losing_ >= 16 || clear_losing_ >= 4) {
            start_parked(cooldown_);
            standard_warmup_ = 2;
            losing_ = clear_losing_ = 0;
            ++parks_;
        }
    }
    void start_parked(std::uint64_t remaining, std::uint64_t next = 0) {
        remaining_ = remaining;
        cooldown_ = next ? next : std::min<std::uint64_t>(4096, remaining * 2);
        decisions_ = 0;
    }
    std::uint64_t remaining() const noexcept { return remaining_; }
    std::uint64_t cooldown() const noexcept { return cooldown_; }
    std::uint64_t parks() const noexcept { return parks_; }
    bool held_up() const noexcept { return decisions_ >= 32; }

private:
    static void append(std::deque<double>& values, double value) {
        if (values.size() == 8) values.pop_front();
        values.push_back(value);
    }
    double score(int depth) const {
        const auto milliseconds = cost(depth);
        if (!milliseconds) return 0;
        double expected = 1, reach = 1;
        for (int position = 0; position < depth; ++position) {
            reach *= acceptance_[position];
            expected += reach;
        }
        return requests_.size() * expected / *milliseconds;
    }
    std::vector<std::string> requests_;
    std::size_t ordinary_requests_;
    int maximum_, current_, standard_warmup_ = 2, losing_ = 0, clear_losing_ = 0, last_depth_ = -1;
    std::array<double, 5> acceptance_{.6, .6, .6, .6, .6};
    std::array<std::deque<double>, 6> costs_;
    std::deque<double> standard_;
    std::array<double, 6> last_seen_{};
    double elapsed_ms_ = 0, last_probe_ms_ = 0;
    std::optional<Clock::time_point> finished_;
    std::uint64_t remaining_ = 0, cooldown_ = 128, decisions_ = 0, parks_ = 0;
};

class BatchParkMemory {
public:
    void tick() noexcept { ++clock_; }
    void seed(BatchDepthPolicy& policy) const {
        const auto found = verdicts_.find(key(policy));
        if (found != verdicts_.end() && found->second.until > clock_)
            policy.start_parked(found->second.until - clock_, found->second.cooldown);
    }
    void parked(const BatchDepthPolicy& policy) {
        verdicts_[key(policy)] = {clock_ + policy.remaining(), policy.cooldown()};
    }
    void retired(const BatchDepthPolicy& policy) {
        if (policy.held_up()) verdicts_.erase(key(policy));
    }
private:
    using Key = std::tuple<std::size_t, std::size_t, int>;
    static Key key(const BatchDepthPolicy& policy) {
        return {policy.requests().size(), policy.ordinary_requests(), policy.maximum()};
    }
    struct Verdict { std::uint64_t until, cooldown; };
    std::uint64_t clock_ = 0;
    std::map<Key, Verdict> verdicts_;
};

} // namespace mfq::engine::mtp::v1
