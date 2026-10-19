#pragma once

#include "prefill_activity.h"
#include "mtp_batch_policy.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

namespace mfq::engine::mtp {

class DepthController {
public:
    explicit DepthController(int maximum_depth = 3,
            const DepthController* seed = nullptr,
            double marginal_ms = 7.0, double exit_margin = 1.15,
            bool gradual_warmup = false, double cold_window_ms = 0.0)
        : maximum_depth_(std::clamp(maximum_depth, 1, 5)),
          current_depth_(maximum_depth_),
          gradual_warmup_(gradual_warmup),
          marginal_ms_(marginal_ms > 0.0 ? marginal_ms : 7.0),
          exit_margin_(std::clamp(exit_margin, 1.0, 1.5)),
          acceptance_(static_cast<std::size_t>(maximum_depth_)),
          cycle_ms_(static_cast<std::size_t>(maximum_depth_ + 1)),
          cycle_age_ms_(static_cast<std::size_t>(maximum_depth_ + 1)) {
        if(gradual_warmup && cold_window_ms>0.0 && (!seed || seed->measurement_order_.empty())) {
            cold_window_ms_=cold_window_ms;cold_tokens_left_=48;cold_maximum_depth_=1;
        }
        acceptance_.front() = 0.6;
        if(gradual_warmup) {
            // Establish ordinary-decode cost before wider speculation. A
            // fresh width cap must not postpone the baseline for 48 tokens.
            // Cover the default four-window residency exchange period rather
            // than calibrating from three cheap windows between exchanges.
            warmup_={1,0,0,0,0,1};
            for(int depth=2;depth<=maximum_depth_;++depth)warmup_.push_back(depth);
        } else for (int index = 1; index <= maximum_depth_; ++index)
            warmup_.push_back(maximum_depth_+1-index);
        current_depth_=warmup_.front();
        if (maximum_depth_ > 1) {
            if(!gradual_warmup)warmup_.insert(warmup_.end(), {0, 0, 0});
            // The initial sweep follows prefill and can include several cold
            // transfer/cache cycles even when graphs were captured at load.
            // Revisit every speculative width after the baseline has run;
            // otherwise a cold first measurement can suppress that width until
            // the multi-second stale-depth probe. Speculative warmup uses its
            // minimum so calibration can replace the cold cost immediately.
            for (int depth = maximum_depth_; depth > 0; --depth)
                warmup_.push_back(depth);
            // A fresh CUDA session is still adapting expert residency during
            // the first revisit. Refresh every width once more before those
            // transient costs can park MTP for the rest of a short response.
            // Continuations and re-entry already have an inherited policy.
            if(gradual_warmup && (!seed || seed->measurement_order_.empty()))
                for(int depth=maximum_depth_;depth>0;--depth)warmup_.push_back(depth);
        }
        // The first scalar samples can retain cold residency costs even after
        // a shallow replay becomes faster. Refresh the baseline at the end as
        // well, so that cheap ordinary decode is compared at the same stage.
        if(gradual_warmup)warmup_.insert(warmup_.end(),{0,0,0,0});
        if (seed && !seed->measurement_order_.empty()) {
            width_baseline_ms_=seed->width_baseline_ms_;
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

    int depth() const noexcept { return cold_tokens_left_>0?std::min(current_depth_,cold_maximum_depth_):current_depth_; }
    int maximum_depth() const noexcept { return maximum_depth_; }
    double comparison_margin() const noexcept {return exit_margin_;}
    bool warming_up() const noexcept { return !warmup_.empty(); }
    bool should_exit() const noexcept { return exit_streak_ >= (gradual_warmup_?6:16); }
    bool speculation_losing() const {
        if (!warmup_.empty() || !cycle_ms_.front()) return false;
        // The first speculative width also pays predictor setup and sampling.
        // Its difference from scalar decode is not the marginal cost of every
        // additional draft. With good acceptance, measure wider windows before
        // parking; cold-window limits may defer that measurement until later.
        if (needs_wider_measurement()) return false;
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
        if(cold_tokens_left_>0) {
            cold_tokens_left_=std::max(0,cold_tokens_left_-accepted_drafts-1);
            // Bound the first widening by recently observed latency. A linear
            // row-cost estimate is conservative while expert residency adapts;
            // normal throughput-based selection resumes after 48 output tokens.
            // Maintenance must not authorize widening from an unmeasured cycle.
            if(time_sample && used_depth==cold_maximum_depth_ && used_depth<maximum_depth_ &&
                    cycle_ms*double(used_depth+2)/double(used_depth+1)<=cold_window_ms_)
                ++cold_maximum_depth_;
        }
        for (int position = 0; position < used_depth; ++position) {
            const auto index = static_cast<std::size_t>(position);
            if (!acceptance_[index]) acceptance_[index] = conditional_acceptance(position);
            const double hit = position < accepted_drafts ? 1.0 : 0.0;
            acceptance_[index] = 0.92 * *acceptance_[index] + 0.08 * hit;
            if (position >= accepted_drafts) break;
        }
        cycle_ms = std::max(0.0, cycle_ms);
        if (time_sample) {
            update_time(used_depth, cycle_ms);
            if(gradual_warmup_ && used_depth>0)width_baseline_ms_[used_depth]=cycle_ms_.front().value_or(0.0);
        }
        if(time_sample && used_depth==1 && cycle_ms_.front())post_baseline_shallow_=true;
        for (auto& age : cycle_age_ms_) if (age) *age += cycle_ms;
        if (time_sample) cycle_age_ms_[static_cast<std::size_t>(used_depth)] = 0.0;
        milliseconds_since_probe_ += cycle_ms;
        milliseconds_since_explore_ += cycle_ms;
        if (time_sample) exit_streak_ = speculation_losing() ? exit_streak_ + 1 : 0;
        if (!warmup_.empty()) {
            if (!time_sample || used_depth != warmup_.front()) return;
            warmup_.erase(warmup_.begin());
            // A valid shallow replay after baseline calibration supplies the
            // current marginal cost. Defer predicted losing wider windows to
            // periodic exploration instead of forcing them into every short
            // response. Cold latency limits also skip blocked warmup entries.
            if(gradual_warmup_ && post_baseline_shallow_) {
                while(!warmup_.empty() && warmup_.front()>1) {
                    const int next=warmup_.front();
                    const bool cold_blocked=cold_tokens_left_>0 && next>cold_maximum_depth_ &&
                        time_estimate(next)>cold_window_ms_;
                    if(!cold_blocked && (score(next)>=score(0) ||
                            (width_needs_measurement(next) && needs_wider_measurement())))break;
                    warmup_.erase(warmup_.begin());
                }
            }
            if (!warmup_.empty()) {
                current_depth_ = warmup_.front();
                return;
            }
            current_depth_ = best_depth();
            if (const auto next = unmeasured_width()) {
                current_depth_ = *next;
                warmup_.push_back(*next);
            }
            milliseconds_since_probe_ = 0.0;
            return;
        }
        if (const auto next = unmeasured_width()) {
            current_depth_ = *next;
            warmup_.push_back(*next);
            return;
        }
        if (probe_left_ > 0) {
            if (!time_sample || used_depth != current_depth_) return;
            --probe_left_;
            if (probe_left_ == 0) {
                current_depth_ = best_depth();
                milliseconds_since_probe_ = 0.0;
            }
            return;
        }
        current_depth_ = best_depth();
        if (maximum_depth_ <= 1 && !gradual_warmup_) return;
        const double period = std::max(1000.0, 4.0 * cycle_ms / 0.15);
        if (milliseconds_since_probe_ < period) return;
        const bool explore_due = milliseconds_since_explore_ >= std::max(5000.0, 2.0 * period);
        // A cold scalar estimate can look too far behind to qualify as a
        // rival after expert residency warms. Refresh it independently of
        // the old score; the four probe tokens still produce normal output.
        const bool baseline_due=gradual_warmup_ && current_depth_>0 &&
            cycle_age_ms_.front().value_or(0.0)>=2.0*period;
        const auto target=baseline_due?std::optional<int>(0):maximum_depth_<=1?
            std::optional<int>{}:explore_due?most_stale():best_rival();
        if (!target) return;
        current_depth_ = *target;
        probe_left_ = 4;
        milliseconds_since_probe_ = 0.0;
        if (explore_due && !baseline_due) milliseconds_since_explore_ = 0.0;
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

    // Ordinary decoding continues while speculation is parked. Refresh its
    // measured cost without advancing a speculative warmup or starting probes.
    void observe_plain_time(double elapsed_ms,bool time_sample=true) {
        if(!std::isfinite(elapsed_ms) || elapsed_ms<=0.0)return;
        for(auto& age:cycle_age_ms_)if(age)*age+=elapsed_ms;
        if(time_sample) {
            update_time(0,elapsed_ms);
            cycle_age_ms_.front()=0.0;
        }
    }

private:
    bool baseline_warmed_since_width(int depth) const {
        return gradual_warmup_ && cache_resweep_ && depth>0 && width_baseline_ms_[depth]>0.0 &&
            cycle_ms_.front() && *cycle_ms_.front()>0.0 && *cycle_ms_.front()<0.9*width_baseline_ms_[depth];
    }
    bool width_needs_measurement(int depth) const {
        const auto index = static_cast<std::size_t>(depth);
        return !cycle_ms_[index] || cycle_age_ms_[index] == std::numeric_limits<double>::infinity() ||
            baseline_warmed_since_width(depth);
    }
    bool needs_wider_measurement() const {
        if (!gradual_warmup_ || !post_baseline_shallow_ || conditional_acceptance(0) < 0.65)
            return false;
        for (int depth = 2; depth <= maximum_depth_; ++depth)
            if (width_needs_measurement(depth)) return true;
        return false;
    }
    std::optional<int> unmeasured_width() const {
        if (!needs_wider_measurement()) return std::nullopt;
        const int maximum = cold_tokens_left_ > 0 ? cold_maximum_depth_ : maximum_depth_;
        for (int depth = 2; depth <= maximum; ++depth)
            if (width_needs_measurement(depth)) return depth;
        return std::nullopt;
    }
    void seed_width_cost(int depth,double cycle_ms) {
        if(!gradual_warmup_ || !cost_window_ || depth<=0)return;
        const auto index=std::size_t(depth);
        width_cost_ms_[index][0]=cycle_ms;width_cost_count_[index]=1;width_cost_next_[index]=1;
    }
    void update_time(int depth, double cycle_ms) {
        const auto index = static_cast<std::size_t>(depth);
        auto& estimate = cycle_ms_[index];
        if(gradual_warmup_ && depth==0) {
            if(!estimate)measurement_order_.push_back(0);
            if(cycle_age_ms_[index]==std::numeric_limits<double>::infinity()) {
                baseline_warmup_count_=0;baseline_warmup_next_=0;
            }
            baseline_warmup_ms_[baseline_warmup_next_]=cycle_ms;
            baseline_warmup_next_=(baseline_warmup_next_+1)%baseline_warmup_ms_.size();
            baseline_warmup_count_=std::min(baseline_warmup_count_+1,baseline_warmup_ms_.size());
            double total=0;
            for(std::size_t i=0;i<baseline_warmup_count_;++i)total+=baseline_warmup_ms_[i];
            estimate=total/baseline_warmup_count_;
            return;
        }
        const bool warmed=baseline_warmed_since_width(depth);
        if (!estimate || cycle_age_ms_[index] == std::numeric_limits<double>::infinity() || warmed) {
            if (!estimate) measurement_order_.push_back(depth);
            if(warmed)std::fprintf(stderr,"runtime_mtp_cache_resweep enabled=1 depth=%d previous_base_ms=%.4f current_base_ms=%.4f previous_cycle_ms=%.4f cycle_ms=%.4f\n",
                depth,width_baseline_ms_[depth],*cycle_ms_.front(),estimate.value_or(0.0),cycle_ms);
            estimate = cycle_ms;
            seed_width_cost(depth,cycle_ms);
            return;
        }
        if (!warmup_.empty()) {
            *estimate = std::min(*estimate, cycle_ms);
            seed_width_cost(depth,*estimate);
            return;
        }
        if(gradual_warmup_ && cost_window_ && depth>0) {
            // A width's expert cache can warm while scalar time stays flat.
            // Include complete periodic exchange costs, but do not retain a
            // cold width estimate for the old ~1.6-second smoothing horizon.
            const auto index=std::size_t(depth);
            if(!width_cost_count_[index])seed_width_cost(depth,*estimate);
            auto& samples=width_cost_ms_[index];auto& next=width_cost_next_[index];
            samples[next]=cycle_ms;next=(next+1)%samples.size();
            auto& count=width_cost_count_[index];count=std::min(count+1,samples.size());
            double total=0;for(std::size_t i=0;i<count;++i)total+=samples[i];
            estimate=total/count;
            if(count==samples.size() && !cost_window_reported_) {
                cost_window_reported_=true;
                std::fprintf(stderr,"runtime_mtp_cost_window enabled=1 depth=%d samples=%zu estimate_ms=%.4f\n",
                    depth,count,*estimate);
            }
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
        const int maximum=cold_tokens_left_>0?cold_maximum_depth_:maximum_depth_;
        for (int depth = cycle_ms_.front() ? 0 : 1; depth <= maximum; ++depth) {
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
    bool gradual_warmup_=false,post_baseline_shallow_=false;
    int probe_left_ = 0;
    int exit_streak_ = 0;
    int cold_tokens_left_=0,cold_maximum_depth_=1;
    double cold_window_ms_=0.0;
    double marginal_ms_;
    double exit_margin_;
    double milliseconds_since_probe_ = 0.0;
    double milliseconds_since_explore_ = 0.0;
    std::vector<std::optional<double>> acceptance_;
    std::vector<std::optional<double>> cycle_ms_;
    std::vector<std::optional<double>> cycle_age_ms_;
    std::vector<int> warmup_;
    std::vector<int> measurement_order_;
    std::array<double,4> baseline_warmup_ms_{};
    std::array<double,6> width_baseline_ms_{};
    std::array<std::array<double,4>,6> width_cost_ms_{};
    std::array<std::size_t,6> width_cost_count_{},width_cost_next_{};
    bool cost_window_reported_=false;
    bool cost_window_=[] {
        const auto* option=std::getenv("MFQ_MTP_COST_WINDOW");return !option || option[0]!='0';
    }();
    bool cache_resweep_=[] {
        const auto* option=std::getenv("MFQ_MTP_CACHE_RESWEEP");return !option || option[0]!='0';
    }();
    std::size_t baseline_warmup_count_=0,baseline_warmup_next_=0;
};

class PolicyState {
public:
    DepthController begin(int maximum_depth,bool gradual_warmup=false,double cold_window_ms=0.0) const {
        // CUDA measures complete graph windows, including drafting and exact
        // sampling. Do not discard a measured 3-15% gain by imposing the
        // generic loop's initial tax; observed plain-loop tax still overrides it.
        double margin = gradual_warmup ? 1.03 : 1.15;
        if (loop_tax_) {
            const double age = std::chrono::duration<double>(Clock::now()-loop_tax_time_).count();
            margin += (*loop_tax_-margin) * std::exp(-age/600.0);
        }
        return DepthController(maximum_depth, seed_ ? &*seed_ : nullptr, 7.0, margin,gradual_warmup,cold_window_ms);
    }
    void remember(const DepthController& controller) { seed_ = controller; }
    void seed_batch(BatchDepthPolicy& policy) const { batch_memory_.seed(policy); }
    void tick_batch() { batch_memory_.tick(); }
    void park_batch(const BatchDepthPolicy& policy) { batch_memory_.parked(policy); }
    void resume_batch(const BatchDepthPolicy& policy) { batch_memory_.resumed(policy); }
    void retire_batch(const BatchDepthPolicy& policy) { batch_memory_.retired(policy); }
    void record_loop_tax(double baseline_ms, const std::vector<double>& samples,bool periodic_mean=false) {
        if (baseline_ms <= 0.0 || samples.size() < 8 || PrefillActivity::recent()) return;
        double normal_ms=0.0;
        if(periodic_mean) {
            // CUDA's baseline includes one real expert exchange per four
            // windows. Its plain-loop comparison must include those windows
            // too; the median mistakes that periodic work for removable tax.
            for(double sample:samples) {
                if(!std::isfinite(sample) || sample<=0.0)return;
                normal_ms+=sample;
            }
            normal_ms/=samples.size();
        } else {
            auto sorted = samples;
            std::sort(sorted.begin(), sorted.end());
            normal_ms = sorted[sorted.size()/2];
        }
        if (normal_ms <= 0.0) return;
        double tax = std::clamp(baseline_ms/normal_ms, 1.0, 1.5);
        if (loop_tax_) tax = 0.5 * *loop_tax_ + 0.5 * tax;
        loop_tax_ = tax;
        loop_tax_time_ = Clock::now();
        if(periodic_mean && !periodic_reported_) {
            periodic_reported_=true;
            std::fprintf(stderr,"runtime_mtp_plain_mean enabled=1 samples=%zu baseline_ms=%.4f plain_ms=%.4f tax=%.4f\n",
                samples.size(),baseline_ms,normal_ms,tax);
        }
    }
private:
    using Clock = std::chrono::steady_clock;
    std::optional<DepthController> seed_;
    BatchParkMemory batch_memory_;
    std::optional<double> loop_tax_;
    bool periodic_reported_=false;
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
    void observe_plain(double elapsed_ms, PolicyState& policy, bool time_sample = true,bool periodic_mean=false) {
        tokens_remaining = std::max(0, tokens_remaining-1);
        if (!time_sample) { baseline_ms = 0.0; return; }
        if (tax_skip > 0) { --tax_skip; return; }
        if (tax_samples.size() >= 8) return;
        tax_samples.push_back(elapsed_ms);
        if (tax_samples.size() == 8) policy.record_loop_tax(baseline_ms, tax_samples,periodic_mean);
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
    using Clock = BatchDepthPolicy::Clock;
    GenerationPolicy(int maximum_depth, bool dspark, PolicyState* state = nullptr,bool gradual_warmup=false,
            bool metal_batch_policy=false)
        : state_(state ? *state : local_state_),gradual_warmup_(gradual_warmup) {
        if (dspark) dspark_.emplace(maximum_depth);
        else if(metal_batch_policy) {
            batch_.emplace(std::vector<std::string>{"cuda-mtp"},maximum_depth);
            state_.seed_batch(*batch_);
            seeded_batch_park_=batch_->remaining()!=0;
        }
        else controller_.emplace(state_.begin(maximum_depth,gradual_warmup_,gradual_warmup_?60.0:0.0));
        if(gradual_warmup_ && !dspark)
            std::fprintf(stderr,"runtime_mtp_depth_policy name=%s maximum_depth=%d\n",
                batch_?"metal_batch":"cuda_adaptive",maximum_depth);
    }
    ~GenerationPolicy() {
        if (controller_) state_.remember(*controller_);
        if (batch_) state_.retire_batch(*batch_);
    }
    int depth() const noexcept {
        return batch_ ? batch_->depth() : dspark_ ? dspark_->depth() : park_.parked ? 0 : controller_->depth();
    }
    int maximum_depth() const noexcept {
        return batch_ ? batch_->maximum() : dspark_ ? dspark_->maximum_depth() : controller_->maximum_depth();
    }
    bool parked() const noexcept { return batch_ ? batch_->remaining()!=0 : !dspark_ && park_.parked; }
    bool dspark() const noexcept { return dspark_.has_value(); }
    std::optional<double> measured_cycle_ms(int depth) const {
        return batch_ ? batch_->cost(depth) : dspark_ ? dspark_->measured_cycle_ms(depth) : controller_->measured_cycle_ms(depth);
    }
    void observe_window(int used,int accepted,Clock::time_point started,Clock::time_point finished,
            bool time_sample=true) {
        if(batch_)observe_batch(used,accepted,started,finished,time_sample);
        else observe(used,accepted,std::chrono::duration<double,std::milli>(finished-started).count(),time_sample);
    }
    void observe_plain_window(Clock::time_point started,Clock::time_point finished,bool time_sample=true) {
        if(batch_)observe_batch(0,0,started,finished,time_sample);
        else observe_plain(std::chrono::duration<double,std::milli>(finished-started).count(),time_sample);
    }
    void observe(int used, int accepted, double elapsed_ms, bool time_sample = true) {
        if(batch_) {
            const auto finished=Clock::now();
            observe_window(used,accepted,finished-std::chrono::duration_cast<Clock::duration>(
                std::chrono::duration<double,std::milli>(elapsed_ms)),finished,time_sample);
            return;
        }
        if (dspark_) { dspark_->observe(used, accepted, elapsed_ms, time_sample); return; }
        const bool was_warmup = controller_->warming_up();
        controller_->observe(used, accepted, elapsed_ms, time_sample);
        trace_decision("observe",used,accepted,elapsed_ms,time_sample);
        if (park_.reentry && !was_warmup && !controller_->warming_up() &&
            !controller_->speculation_losing()) park_.probe_succeeded();
        if (controller_->should_exit()) {
            state_.remember(*controller_);
            park_.park(*controller_);
            ++parks_;
        }
    }
    void observe_plain(double elapsed_ms, bool time_sample = true) {
        if(batch_) {
            const auto finished=Clock::now();
            observe_plain_window(finished-std::chrono::duration_cast<Clock::duration>(
                std::chrono::duration<double,std::milli>(elapsed_ms)),finished,time_sample);
            return;
        }
        if(controller_ && park_.parked)controller_->observe_plain_time(elapsed_ms,time_sample);
        park_.observe_plain(elapsed_ms, state_, time_sample,gradual_warmup_);
        if (park_.probe_ready()) {
            // begin() inherits the controller saved in state_. Ordinary
            // samples since parking must replace that older baseline first.
            state_.remember(*controller_);
            park_.begin_probe();
            controller_.emplace(state_.begin(controller_->maximum_depth(),gradual_warmup_));
            ++reentries_;
            trace_decision("reentry",0,0,elapsed_ms,time_sample,true);
        }
    }
    std::uint64_t parks() const noexcept { return batch_ ? batch_->parks()+seeded_batch_park_ : parks_; }
    std::uint64_t reentries() const noexcept { return reentries_; }
private:
    void observe_batch(int used,int accepted,Clock::time_point started,Clock::time_point finished,bool valid) {
        const bool was_parked=batch_->remaining()!=0;
        const auto previous_parks=batch_->parks();
        state_.tick_batch();
        const std::array<BatchAcceptance,1> counts{{{used,accepted}}};
        batch_->observe(counts,started,finished,valid,false);
        if(was_parked && !batch_->remaining()) { ++reentries_;state_.resume_batch(*batch_); }
        if(batch_->parks()!=previous_parks)state_.park_batch(*batch_);
        if(trace_ && (depth()!=trace_depth_ || batch_->parks()!=previous_parks ||
                (was_parked && !batch_->remaining()))) {
            trace_depth_=depth();
            std::fprintf(stderr,"runtime_mtp_batch_decision used=%d accepted=%d valid=%d depth=%d remaining=%llu parks=%llu reentries=%llu ms0=%.4f ms1=%.4f ms2=%.4f ms3=%.4f\n",
                used,accepted,int(valid),depth(),static_cast<unsigned long long>(batch_->remaining()),
                static_cast<unsigned long long>(parks()),static_cast<unsigned long long>(reentries_),
                batch_->cost(0).value_or(0),batch_->cost(1).value_or(0),
                batch_->cost(2).value_or(0),batch_->cost(3).value_or(0));
        }
    }
    void trace_decision(const char* event,int used,int accepted,double elapsed,bool valid,bool force=false) {
        if(!trace_)return;
        const auto& c=*controller_;const int depth=c.depth();const bool warm=c.warming_up();
        ++trace_observations_;
        if(!force && trace_observations_>12 && depth==trace_depth_ && warm==trace_warm_ && !c.should_exit())return;
        trace_depth_=depth;trace_warm_=warm;
        std::array<double,6> costs{};std::array<double,5> acceptance{};
        for(int i=0;i<=c.maximum_depth();++i)costs[i]=c.measured_cycle_ms(i).value_or(0.0);
        for(int i=0;i<c.maximum_depth();++i)acceptance[i]=c.conditional_acceptance(i);
        std::fprintf(stderr,"runtime_mtp_policy_decision event=%s used=%d accepted=%d cycle_ms=%.4f valid=%d depth=%d warm=%d losing=%d exit=%d gain=%.4f ms0=%.4f ms1=%.4f ms2=%.4f ms3=%.4f ms4=%.4f ms5=%.4f a1=%.5f a2=%.5f a3=%.5f a4=%.5f a5=%.5f\n",
            event,used,accepted,elapsed,int(valid),depth,int(warm),int(c.speculation_losing()),int(c.should_exit()),c.comparison_margin(),
            costs[0],costs[1],costs[2],costs[3],costs[4],costs[5],acceptance[0],acceptance[1],acceptance[2],acceptance[3],acceptance[4]);
    }
    bool trace_=[] {const auto* setting=std::getenv("MFQ_PROFILE_MTP");return setting && setting[0]=='1';}();
    uint64_t trace_observations_=0;int trace_depth_=-1;bool trace_warm_=false;
    PolicyState local_state_;
    PolicyState& state_;
    bool gradual_warmup_=false;
    std::optional<DepthController> controller_;
    std::optional<DsparkController> dspark_;
    std::optional<BatchDepthPolicy> batch_;
    bool seeded_batch_park_=false;
    ParkState park_;
    std::uint64_t parks_ = 0;
    std::uint64_t reentries_ = 0;
};

}
