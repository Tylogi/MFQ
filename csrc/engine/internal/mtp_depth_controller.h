#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

namespace mfq::engine::mtp {

// One adaptive-depth model: acceptance estimates per draft position, cycle-time
// estimates per depth, and a probe/explore schedule that tests rival depths.
// Two tunings share it. Park mode (default) seeds from the previous request and
// gives up on speculation once it stops paying. Cycle mode measures the loop tax
// instead and never parks.
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
        if (seed && !seed->measured_.empty()) {
            for (int position = 0; position < std::min(maximum_depth_, seed->maximum_depth_); ++position)
                acceptance_[static_cast<std::size_t>(position)] = seed->acceptance_[static_cast<std::size_t>(position)];
            for (int depth = 0; depth <= std::min(maximum_depth_, seed->maximum_depth_); ++depth) {
                const auto index = static_cast<std::size_t>(depth);
                cycle_ms_[index] = seed->cycle_ms_[index];
                if (cycle_ms_[index]) {
                    cycle_age_ms_[index] = std::numeric_limits<double>::infinity();
                    measured_.push_back(depth);
                }
            }
        }
    }

    // Cycle-timed tuning: every position starts optimistic, warmup counts hits
    // exactly before falling back to the exponential average, and the depth is
    // never parked.
    static DepthController cycle_timed(int maximum_depth) {
        DepthController controller(std::clamp(maximum_depth, 1, 5));
        controller.cycle_timed_ = true;
        controller.exit_margin_ = 1.0;
        controller.acceptance_.assign(static_cast<std::size_t>(controller.maximum_depth_), 0.6);
        controller.warmup_accepts_.assign(static_cast<std::size_t>(controller.maximum_depth_), 0);
        controller.warmup_trials_.assign(static_cast<std::size_t>(controller.maximum_depth_), 0);
        controller.warmup_ = {controller.maximum_depth_, controller.maximum_depth_, 0, 0, 0};
        return controller;
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
            const double hit = position < accepted_drafts ? 1.0 : 0.0;
            if (!cycle_timed_ && !acceptance_[index])
                acceptance_[index] = conditional_acceptance(position);
            if (cycle_timed_ && !warmup_.empty()) {
                warmup_accepts_[index] += hit > 0.0 ? 1 : 0;
                ++warmup_trials_[index];
                acceptance_[index] = static_cast<double>(warmup_accepts_[index]) /
                    static_cast<double>(warmup_trials_[index]);
            } else {
                acceptance_[index] = (1.0 - kAcceptanceAlpha) * acceptance_[index] +
                    kAcceptanceAlpha * hit;
            }
            if (position >= accepted_drafts) break;
        }
        cycle_ms = std::max(0.0, cycle_ms);
        if (time_sample) update_time(used_depth, cycle_ms);
        for (auto& age : cycle_age_ms_) if (age) *age += cycle_ms;
        if (time_sample) cycle_age_ms_[static_cast<std::size_t>(used_depth)] = 0.0;
        milliseconds_since_probe_ += cycle_ms;
        milliseconds_since_explore_ += cycle_ms;
        if (!cycle_timed_) exit_streak_ = speculation_losing() ? exit_streak_ + 1 : 0;
        if (!warmup_.empty()) {
            if (!cycle_timed_ && !time_sample) return;
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
        const double period = std::max(kProbePeriodMs,
            static_cast<double>(kProbeLength) * cycle_ms / kProbeDuty);
        if (milliseconds_since_probe_ < period) return;
        const bool explore_due = milliseconds_since_explore_ >=
            std::max(kExplorePeriodMs, 2.0 * period);
        const auto target = explore_due ? most_stale() : best_rival();
        if (!target) return;
        current_depth_ = *target;
        probe_left_ = kProbeLength;
        milliseconds_since_probe_ = 0.0;
        if (explore_due) milliseconds_since_explore_ = 0.0;
    }

    // Park mode inherits from the deepest position already measured, so a fresh
    // request starts where the last one left off. Cycle mode keeps its own
    // optimistic estimate for every position.
    double conditional_acceptance(int position) const {
        if (position < 0 || position >= maximum_depth_)
            throw std::out_of_range("MTP acceptance position is out of range");
        if (!cycle_timed_)
            while (!acceptance_[static_cast<std::size_t>(position)]) --position;
        return acceptance_[static_cast<std::size_t>(position)];
    }

    std::optional<double> measured_cycle_ms(int depth) const {
        if (depth < 0 || depth > maximum_depth_)
            throw std::out_of_range("MTP measured depth is out of range");
        return cycle_ms_[static_cast<std::size_t>(depth)];
    }

private:
    static constexpr double kAcceptanceAlpha = 0.08;
    static constexpr double kTimeTauMs = 400.0;
    static constexpr double kProbePeriodMs = 1000.0;
    static constexpr double kExplorePeriodMs = 5000.0;
    static constexpr int kProbeLength = 4;
    static constexpr double kProbeDuty = 0.15;
    static constexpr double kProbeMargin = 1.15;
    static constexpr double kSpikeRatio = 2.0;
    static constexpr double kSpikeDamp = 0.25;
    static constexpr double kHysteresis = 1.03;

    void update_time(int depth, double cycle_ms) {
        const auto index = static_cast<std::size_t>(depth);
        auto& estimate = cycle_ms_[index];
        // A seeded estimate is stale by construction; replace it outright.
        if (!estimate || cycle_age_ms_[index] == std::numeric_limits<double>::infinity()) {
            if (!estimate) measured_.push_back(depth);
            estimate = cycle_ms;
            return;
        }
        if (!warmup_.empty()) {
            *estimate = std::min(*estimate, cycle_ms);
            return;
        }
        double alpha = 1.0 - std::exp(-cycle_ms / kTimeTauMs);
        if (cycle_ms > kSpikeRatio * *estimate) alpha *= kSpikeDamp;
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
        for (int candidate = 0; candidate <= maximum_depth_; ++candidate) {
            if (!cycle_ms_[static_cast<std::size_t>(candidate)]) continue;
            if (reference < 0 || std::abs(candidate - depth) < std::abs(reference - depth))
                reference = candidate;
        }
        if (reference < 0) return 30.0 + marginal_ms_ * depth;
        if (depth == 0) {
            double minimum = std::numeric_limits<double>::infinity();
            for (const auto& measured : cycle_ms_)
                if (measured) minimum = std::min(minimum, *measured);
            return minimum;
        }
        return std::max(1.0e-3, *cycle_ms_[static_cast<std::size_t>(reference)] +
            marginal_estimate() * (depth - reference));
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
        if (best != current_depth_ && best_score < score(current_depth_) * kHysteresis)
            return current_depth_;
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
        return rival && rival_score >= current_score / kProbeMargin ? rival : std::nullopt;
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
    bool cycle_timed_ = false;
    std::vector<double> acceptance_;
    std::vector<int> warmup_accepts_;
    std::vector<int> warmup_trials_;
    std::vector<std::optional<double>> cycle_ms_;
    std::vector<std::optional<double>> cycle_age_ms_;
    std::vector<int> warmup_;
    std::vector<int> measured_;
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
    // busy: prefill is in flight somewhere in the process, so wall-clock cycle
            // times say nothing about the loop tax and must not be sampled.
    void record_loop_tax(double baseline_ms, const std::vector<double>& samples, bool busy) {
        if (baseline_ms <= 0.0 || samples.size() < 8 || busy) return;
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

    void park(const DepthController& controller, bool busy) {
        if (reentry) cooldown_tokens = std::min(4096, cooldown_tokens*2);
        tokens_remaining = cooldown_tokens;
        parked = true;
        reentry = false;
        tax_skip = 2;
        baseline_ms = busy ? 0.0 : controller.measured_cycle_ms(0).value_or(0.0);
        tax_samples.clear();
    }
    void observe_plain(double elapsed_ms, PolicyState& policy, bool time_sample, bool busy) {
        tokens_remaining = std::max(0, tokens_remaining-1);
        if (!time_sample) { baseline_ms = 0.0; return; }
        if (tax_skip > 0) { --tax_skip; return; }
        if (tax_samples.size() >= 8) return;
        tax_samples.push_back(elapsed_ms);
        if (tax_samples.size() == 8) policy.record_loop_tax(baseline_ms, tax_samples, busy);
    }
    bool probe_ready(bool busy) const noexcept {
        return parked && tokens_remaining <= 0 && !busy;
    }
    void begin_probe() noexcept { parked = false; reentry = true; }
    void probe_succeeded() noexcept { reentry = false; cooldown_tokens = 128; }
};

class GenerationPolicy {
public:
    GenerationPolicy(int maximum_depth, bool cycle_timed, PolicyState* state = nullptr)
        : state_(state ? *state : local_state_),
          cycle_timed_(cycle_timed),
          controller_(cycle_timed ? DepthController::cycle_timed(maximum_depth)
                                  : state_.begin(maximum_depth)) {}
    ~GenerationPolicy() { if (!cycle_timed_) state_.remember(controller_); }
    int depth() const noexcept {
        return cycle_timed_ || !park_.parked ? controller_.depth() : 0;
    }
    int maximum_depth() const noexcept { return controller_.maximum_depth(); }
    bool parked() const noexcept { return !cycle_timed_ && park_.parked; }
    bool cycle_timed() const noexcept { return cycle_timed_; }
    std::optional<double> measured_cycle_ms(int depth) const {
        return controller_.measured_cycle_ms(depth);
    }
    void observe(int used, int accepted, double elapsed_ms, bool time_sample = true, bool busy = false) {
        const bool was_warmup = controller_.warming_up();
        controller_.observe(used, accepted, elapsed_ms, time_sample);
        if (cycle_timed_) return;
        if (park_.reentry && !was_warmup && !controller_.warming_up() &&
            !controller_.speculation_losing()) park_.probe_succeeded();
        if (controller_.should_exit()) {
            state_.remember(controller_);
            park_.park(controller_, busy);
            ++parks_;
        }
    }
    void observe_plain(double elapsed_ms, bool time_sample = true, bool busy = false) {
        if (cycle_timed_) return;
        park_.observe_plain(elapsed_ms, state_, time_sample, busy);
        if (park_.probe_ready(busy)) {
            park_.begin_probe();
            controller_ = state_.begin(controller_.maximum_depth());
            ++reentries_;
        }
    }
    std::uint64_t parks() const noexcept { return parks_; }
    std::uint64_t reentries() const noexcept { return reentries_; }
private:
    PolicyState local_state_;
    PolicyState& state_;
    bool cycle_timed_;
    DepthController controller_;
    ParkState park_;
    std::uint64_t parks_ = 0;
    std::uint64_t reentries_ = 0;
};

}
