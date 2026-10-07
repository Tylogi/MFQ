#pragma once

#include "prefill_activity.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

namespace mfq::engine::mtp {

class DepthController {
public:
    explicit DepthController(int maximum_depth = 3,
            const DepthController* seed = nullptr,
            double marginal_ms = 7.0, double exit_margin = 1.15)
        : maximum_depth_(std::clamp(maximum_depth, 1, 5)),
          current_depth_(maximum_depth_),
          marginal_ms_(marginal_ms > 0.0 ? marginal_ms : 7.0),
          exit_margin_(std::clamp(exit_margin, 1.0, 1.5)),
          acceptance_(static_cast<std::size_t>(maximum_depth_)),
          cycle_ms_(static_cast<std::size_t>(maximum_depth_ + 1)),
          cycle_age_ms_(static_cast<std::size_t>(maximum_depth_ + 1)) {
        acceptance_.front() = 0.6;
        for (int depth = maximum_depth_; depth > 0; --depth)
            warmup_.push_back(depth);
        if (maximum_depth_ > 1)
            warmup_.insert(warmup_.end(), {0, 0, 0});
        if (seed && !seed->measurement_order_.empty()) {
            for (int position = 0; position < std::min(maximum_depth_, seed->maximum_depth_); ++position)
                acceptance_[static_cast<std::size_t>(position)] = seed->acceptance_[static_cast<std::size_t>(position)];
            for (int depth = 0; depth <= std::min(maximum_depth_, seed->maximum_depth_); ++depth) {
                const auto index = static_cast<std::size_t>(depth);
                cycle_ms_[index] = seed->cycle_ms_[index];
                if (cycle_ms_[index])
                    cycle_age_ms_[index] = std::numeric_limits<double>::infinity();
            }
            for (int depth : seed->measurement_order_)
                if (depth <= maximum_depth_) measurement_order_.push_back(depth);
        }
    }

    int depth() const noexcept { return current_depth_; }
    int maximum_depth() const noexcept { return maximum_depth_; }
    bool warming_up() const noexcept { return !warmup_.empty(); }
    bool should_exit() const noexcept { return exit_streak_ >= 16; }
    bool speculation_losing() const {
        if (!warmup_.empty() || !cycle_ms_.front()) return false;
        const double baseline = score(0);
        if (baseline <= 0.0) return false;
        double best = 0.0;
        for (int depth = 1; depth <= maximum_depth_; ++depth)
            best = std::max(best, score(depth));
        return best < baseline * exit_margin_;
    }

    void observe(int used_depth, int accepted_drafts, double cycle_ms,
            bool time_sample = true) {
        used_depth = std::clamp(used_depth, 0, maximum_depth_);
        accepted_drafts = std::clamp(accepted_drafts, 0, used_depth);
        for (int position = 0; position < used_depth; ++position) {
            const auto index = static_cast<std::size_t>(position);
            if (!acceptance_[index]) acceptance_[index] = conditional_acceptance(position);
            const double hit = position < accepted_drafts ? 1.0 : 0.0;
            acceptance_[index] = 0.92 * *acceptance_[index] + 0.08 * hit;
            if (position >= accepted_drafts) break;
        }
        cycle_ms = std::max(0.0, cycle_ms);
        if (time_sample) update_time(used_depth, cycle_ms);
        for (auto& age : cycle_age_ms_) if (age) *age += cycle_ms;
        if (time_sample) cycle_age_ms_[static_cast<std::size_t>(used_depth)] = 0.0;
        milliseconds_since_probe_ += cycle_ms;
        milliseconds_since_explore_ += cycle_ms;
        exit_streak_ = speculation_losing() ? exit_streak_ + 1 : 0;
        if (!warmup_.empty()) {
            if (!time_sample) return;
            warmup_.erase(warmup_.begin());
            if (!warmup_.empty()) {
                current_depth_ = warmup_.front();
                return;
            }
            current_depth_ = best_depth();
            milliseconds_since_probe_ = 0.0;
            return;
        }
        if (probe_left_ > 0) {
            --probe_left_;
            if (probe_left_ == 0) {
                current_depth_ = best_depth();
                milliseconds_since_probe_ = 0.0;
            }
            return;
        }
        current_depth_ = best_depth();
        if (maximum_depth_ <= 1) return;
        const double period = std::max(1000.0, 4.0 * cycle_ms / 0.15);
        if (milliseconds_since_probe_ < period) return;
        const bool explore_due = milliseconds_since_explore_ >= std::max(5000.0, 2.0 * period);
        const auto target = explore_due ? most_stale() : best_rival();
        if (!target) return;
        current_depth_ = *target;
        probe_left_ = 4;
        milliseconds_since_probe_ = 0.0;
        if (explore_due) milliseconds_since_explore_ = 0.0;
    }

    double conditional_acceptance(int position) const {
        if (position < 0 || position >= maximum_depth_)
            throw std::out_of_range("MTP acceptance position is out of range");
        while (!acceptance_[static_cast<std::size_t>(position)]) --position;
        return *acceptance_[static_cast<std::size_t>(position)];
    }

    std::optional<double> measured_cycle_ms(int depth) const {
        if (depth < 0 || depth > maximum_depth_)
            throw std::out_of_range("MTP measured depth is out of range");
        return cycle_ms_[static_cast<std::size_t>(depth)];
    }

private:
    void update_time(int depth, double cycle_ms) {
        const auto index = static_cast<std::size_t>(depth);
        auto& estimate = cycle_ms_[index];
        if (!estimate || cycle_age_ms_[index] == std::numeric_limits<double>::infinity()) {
            if (!estimate) measurement_order_.push_back(depth);
            estimate = cycle_ms;
            return;
        }
        if (!warmup_.empty()) {
            *estimate = std::min(*estimate, cycle_ms);
            return;
        }
        double alpha = 1.0 - std::exp(-cycle_ms / 400.0);
        if (cycle_ms > 2.0 * *estimate) alpha *= 0.25;
        *estimate = (1.0 - alpha) * *estimate + alpha * cycle_ms;
    }

    double marginal_estimate() const {
        int low = -1, high = -1;
        for (int depth = 0; depth <= maximum_depth_; ++depth) {
            if (!cycle_ms_[static_cast<std::size_t>(depth)]) continue;
            if (low < 0) low = depth;
            high = depth;
        }
        if (low >= 0 && high > low) {
            const double slope = (*cycle_ms_[static_cast<std::size_t>(high)] -
                *cycle_ms_[static_cast<std::size_t>(low)]) / (high - low);
            if (slope > 0.0) return slope;
        }
        return marginal_ms_;
    }

    double time_estimate(int depth) const {
        if (cycle_ms_[static_cast<std::size_t>(depth)]) return *cycle_ms_[static_cast<std::size_t>(depth)];
        int reference = -1;
        double minimum = std::numeric_limits<double>::infinity();
        for (int candidate : measurement_order_) {
            const auto& measured = cycle_ms_[static_cast<std::size_t>(candidate)];
            if (!measured) continue;
            minimum = std::min(minimum, *measured);
            if (reference < 0 || std::abs(candidate-depth) < std::abs(reference-depth)) reference = candidate;
        }
        if (reference < 0) return 30.0 + marginal_ms_ * depth;
        if (depth == 0) return minimum;
        return std::max(1.0e-3, *cycle_ms_[static_cast<std::size_t>(reference)] +
            marginal_estimate() * (depth-reference));
    }

    double score(int depth) const {
        double expected = 1.0, run = 1.0;
        for (int position = 0; position < depth; ++position) {
            run *= conditional_acceptance(position);
            expected += run;
        }
        return expected / std::max(1.0e-6, time_estimate(depth));
    }

    int best_depth() const {
        int best = current_depth_;
        double best_score = -1.0;
        for (int depth = cycle_ms_.front() ? 0 : 1; depth <= maximum_depth_; ++depth) {
            const double candidate = score(depth);
            if (candidate > best_score) { best = depth; best_score = candidate; }
        }
        if (best != current_depth_ && best_score < score(current_depth_) * 1.03) return current_depth_;
        return best;
    }

    std::optional<int> best_rival() const {
        const double current_score = score(current_depth_);
        if (current_score <= 0.0) return most_stale();
        std::optional<int> rival;
        double rival_score = 0.0;
        const auto consider = [&](int depth) {
            if (depth == current_depth_) return;
            const double candidate = score(depth);
            if (candidate > rival_score) { rival = depth; rival_score = candidate; }
        };
        for (int depth = 1; depth <= maximum_depth_; ++depth) consider(depth);
        consider(0);
        return rival && rival_score >= current_score / 1.15 ? rival : std::nullopt;
    }

    std::optional<int> most_stale() const {
        std::optional<int> result;
        double oldest = -1.0;
        const auto consider = [&](int depth) {
            if (depth == current_depth_) return;
            const double age = cycle_age_ms_[static_cast<std::size_t>(depth)].value_or(
                std::numeric_limits<double>::infinity());
            if (age > oldest) { result = depth; oldest = age; }
        };
        for (int depth = 1; depth <= maximum_depth_; ++depth) consider(depth);
        consider(0);
        return result;
    }

    int maximum_depth_;
    int current_depth_;
    int probe_left_ = 0;
    int exit_streak_ = 0;
    double marginal_ms_;
    double exit_margin_;
    double milliseconds_since_probe_ = 0.0;
    double milliseconds_since_explore_ = 0.0;
    std::vector<std::optional<double>> acceptance_;
    std::vector<std::optional<double>> cycle_ms_;
    std::vector<std::optional<double>> cycle_age_ms_;
    std::vector<int> warmup_;
    std::vector<int> measurement_order_;
};

class PolicyState {
public:
    DepthController begin(int maximum_depth) const {
        double margin = 1.15;
        if (loop_tax_) {
            const double age = std::chrono::duration<double>(Clock::now()-loop_tax_time_).count();
            margin += (*loop_tax_-margin) * std::exp(-age/600.0);
        }
        return DepthController(maximum_depth, seed_ ? &*seed_ : nullptr, 7.0, margin);
    }
    void remember(const DepthController& controller) { seed_ = controller; }
    void record_loop_tax(double baseline_ms, const std::vector<double>& samples) {
        if (baseline_ms <= 0.0 || samples.size() < 8 || PrefillActivity::recent()) return;
        auto sorted = samples;
        std::sort(sorted.begin(), sorted.end());
        const double normal_ms = sorted[sorted.size()/2];
        if (normal_ms <= 0.0) return;
        double tax = std::clamp(baseline_ms/normal_ms, 1.0, 1.5);
        if (loop_tax_) tax = 0.5 * *loop_tax_ + 0.5 * tax;
        loop_tax_ = tax;
        loop_tax_time_ = Clock::now();
    }
private:
    using Clock = std::chrono::steady_clock;
    std::optional<DepthController> seed_;
    std::optional<double> loop_tax_;
    Clock::time_point loop_tax_time_{};
};

struct ParkState {
    bool parked = false;
    bool reentry = false;
    int cooldown_tokens = 128;
    int tokens_remaining = 128;
    int tax_skip = 2;
    double baseline_ms = 0.0;
    std::vector<double> tax_samples;

    void park(const DepthController& controller) {
        if (reentry) cooldown_tokens = std::min(4096, cooldown_tokens*2);
        tokens_remaining = cooldown_tokens;
        parked = true;
        reentry = false;
        tax_skip = 2;
        baseline_ms = PrefillActivity::recent() ? 0.0 : controller.measured_cycle_ms(0).value_or(0.0);
        tax_samples.clear();
    }
    void observe_plain(double elapsed_ms, PolicyState& policy, bool time_sample = true) {
        tokens_remaining = std::max(0, tokens_remaining-1);
        if (!time_sample) { baseline_ms = 0.0; return; }
        if (tax_skip > 0) { --tax_skip; return; }
        if (tax_samples.size() >= 8) return;
        tax_samples.push_back(elapsed_ms);
        if (tax_samples.size() == 8) policy.record_loop_tax(baseline_ms, tax_samples);
    }
    bool probe_ready() const noexcept {
        return parked && tokens_remaining <= 0 && !PrefillActivity::recent();
    }
    void begin_probe() noexcept { parked = false; reentry = true; }
    void probe_succeeded() noexcept { reentry = false; cooldown_tokens = 128; }
};

template<class DsparkController>
class GenerationPolicy {
public:
    GenerationPolicy(int maximum_depth, bool dspark, PolicyState* state = nullptr)
        : state_(state ? *state : local_state_) {
        if (dspark) dspark_.emplace(maximum_depth);
        else controller_.emplace(state_.begin(maximum_depth));
    }
    ~GenerationPolicy() { if (controller_) state_.remember(*controller_); }
    int depth() const noexcept {
        return dspark_ ? dspark_->depth() : park_.parked ? 0 : controller_->depth();
    }
    int maximum_depth() const noexcept {
        return dspark_ ? dspark_->maximum_depth() : controller_->maximum_depth();
    }
    bool parked() const noexcept { return !dspark_ && park_.parked; }
    bool dspark() const noexcept { return dspark_.has_value(); }
    std::optional<double> measured_cycle_ms(int depth) const {
        return dspark_ ? dspark_->measured_cycle_ms(depth) : controller_->measured_cycle_ms(depth);
    }
    void observe(int used, int accepted, double elapsed_ms, bool time_sample = true) {
        if (dspark_) { dspark_->observe(used, accepted, elapsed_ms, time_sample); return; }
        const bool was_warmup = controller_->warming_up();
        controller_->observe(used, accepted, elapsed_ms, time_sample);
        if (park_.reentry && !was_warmup && !controller_->warming_up() &&
            !controller_->speculation_losing()) park_.probe_succeeded();
        if (controller_->should_exit()) {
            state_.remember(*controller_);
            park_.park(*controller_);
            ++parks_;
        }
    }
    void observe_plain(double elapsed_ms, bool time_sample = true) {
        park_.observe_plain(elapsed_ms, state_, time_sample);
        if (park_.probe_ready()) {
            park_.begin_probe();
            controller_.emplace(state_.begin(controller_->maximum_depth()));
            ++reentries_;
        }
    }
    std::uint64_t parks() const noexcept { return parks_; }
    std::uint64_t reentries() const noexcept { return reentries_; }
private:
    PolicyState local_state_;
    PolicyState& state_;
    std::optional<DepthController> controller_;
    std::optional<DsparkController> dspark_;
    ParkState park_;
    std::uint64_t parks_ = 0;
    std::uint64_t reentries_ = 0;
};

}
