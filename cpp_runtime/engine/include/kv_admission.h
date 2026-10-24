#pragma once

#include "prefill_budget.h"

#include <cmath>
#include <map>
#include <utility>
#include <vector>

namespace mfq::engine {

struct KvAdmissionConfig {
    double initial_ratio = 0.7;
    double minimum_ratio_factor = 0.14;
    double conservativeness = 1.0;
    std::uint64_t decay_steps = 600;
    std::uint64_t retract_steps = 20;
    std::int64_t decode_token_clip = 4096;
};

// Estimates future tokens only; it never changes a request's generation limit.
class NewTokenRatioTracker {
    KvAdmissionConfig config_;
    double initial_, minimum_, decay_, current_;
  public:
    explicit NewTokenRatioTracker(KvAdmissionConfig config = {}) : config_(config) {
        if (!std::isfinite(config.initial_ratio) || config.initial_ratio <= 0 || config.initial_ratio > 1 ||
            !std::isfinite(config.minimum_ratio_factor) || config.minimum_ratio_factor <= 0 ||
            config.minimum_ratio_factor > 1 || !std::isfinite(config.conservativeness) ||
            config.conservativeness <= 0 || !config.decay_steps || !config.retract_steps ||
            config.decode_token_clip <= 0)
            throw std::invalid_argument("invalid KV admission ratio configuration");
        initial_ = std::min(config.initial_ratio * config.conservativeness, 1.0);
        minimum_ = initial_ * config.minimum_ratio_factor;
        decay_ = (initial_ - minimum_) / config.decay_steps;
        current_ = initial_;
    }
    double current() const noexcept { return current_; }
    void reset() noexcept { current_ = initial_; }
    void decay_step() noexcept { current_ = std::max(minimum_, current_ - decay_); }
    std::int64_t expected(std::int64_t maximum, std::int64_t produced) const {
        const auto remaining = std::min(std::max<std::int64_t>(0, maximum - produced), config_.decode_token_clip);
        const auto estimate = std::ceil(remaining * current_);
        return estimate >= remaining ? remaining : static_cast<std::int64_t>(estimate);
    }
    template<class Requests> void after_retraction(const Requests& running) {
        long double decoded = 0, maximum = 1;
        for (const auto& request : running) {
            decoded += request->produced;
            maximum += request->generation_limit;
        }
        const auto reserve = static_cast<long double>(config_.retract_steps) * running.size();
        current_ = std::min(double((decoded + reserve) / maximum), 1.0);
    }
};

// Arena IDs are backend-owned (for example device IDs). Never spend another
// arena's free bytes, and never partially commit a multi-arena reservation.
using KvAdmissionBytes = std::map<int, std::size_t>;
class KvAdmissionBudget {
    KvAdmissionBytes remaining_;
  public:
    explicit KvAdmissionBudget(KvAdmissionBytes available) : remaining_(std::move(available)) {}
    bool reserve(const KvAdmissionBytes& demand) {
        for (const auto& [arena, bytes] : demand) {
            const auto it = remaining_.find(arena);
            if (it == remaining_.end() || bytes > it->second) return false;
        }
        for (const auto& [arena, bytes] : demand) remaining_.at(arena) -= bytes;
        return true;
    }
};

// Stable ties preserve arrival order. Callers stop retracting at one live row.
template<class Requests> Requests retraction_order(Requests requests) {
    std::stable_sort(requests.begin(), requests.end(), [](const auto& a, const auto& b) {
        if (a->produced != b->produced) return a->produced < b->produced;
        return a->prompt.size() > b->prompt.size();
    });
    return requests;
}

inline std::size_t kv_admission_round_up(std::size_t bytes, std::size_t quantum) {
    if (!quantum) throw std::invalid_argument("KV allocation quantum must be positive");
    const auto remainder = bytes % quantum;
    return remainder ? prefill_budget_add(bytes, quantum - remainder) : bytes;
}

inline std::size_t kv_admission_multiply(std::size_t a, std::size_t b) {
    const auto maximum = std::numeric_limits<std::size_t>::max();
    return a && b > maximum / a ? maximum : a * b;
}

// Count the union of physical pages touched by a prefix in a strided head
// layout. Heads can share a mapping page, so rounding their aggregate is wrong.
inline std::size_t kv_admission_mapping_bytes(std::size_t heads, std::size_t maximum,
        std::size_t tokens, std::size_t row_bytes, std::size_t quantum) {
    if (!quantum || tokens > maximum) throw std::invalid_argument("invalid KV mapping geometry");
    if (!tokens) return 0;
    const auto stride = kv_admission_multiply(maximum, row_bytes);
    const auto extent = kv_admission_multiply(tokens, row_bytes);
    std::size_t pages = 0, previous_end = 0;
    for (std::size_t head = 0; head < heads; ++head) {
        const auto begin = kv_admission_multiply(head, stride);
        const auto end = prefill_budget_add(begin, extent);
        const auto first = std::max(begin / quantum, previous_end);
        const auto last = end / quantum + (end % quantum != 0);
        if (last > first) pages = prefill_budget_add(pages, last - first);
        previous_end = std::max(previous_end, last);
    }
    return kv_admission_multiply(pages, quantum);
}

} // namespace mfq::engine
