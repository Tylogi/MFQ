#include "mtp_metrics.h"
#include <array>
#include <iostream>

namespace {
void require(bool ok) { if (!ok) throw std::runtime_error("MTP policy test failed"); }
template<class F> void rejects(F&& fn) {
    bool rejected = false;
    try { fn(); } catch (const std::exception&) { rejected = true; }
    require(rejected);
}
}

int main() {
    using namespace mfq::engine::mtp;
    {
        DepthController controller(3);
        const std::array<int, 6> sweep{3, 2, 1, 0, 0, 0};
        for (int depth : sweep) {
            require(controller.depth() == depth);
            controller.observe(depth, depth, depth ? 20.0 + depth : 20.0);
        }
        require(!controller.warming_up());
        require(std::fabs(controller.conditional_acceptance(0)-0.6885248) < 1e-12);
        require(std::fabs(*controller.measured_cycle_ms(3)-23.0) < 1e-12);
        require(std::fabs(*controller.measured_cycle_ms(0)-20.0) < 1e-12);
        const double before = *controller.measured_cycle_ms(3);
        controller.observe(3, 3, 100.0);
        const double alpha = 0.25 * (1.0-std::exp(-100.0/400.0));
        require(std::fabs(*controller.measured_cycle_ms(3)-((1-alpha)*before+alpha*100.0)) < 1e-12);
        DepthController inherited(3, &controller);
        inherited.observe(3, 0, 90.0);
        require(*inherited.measured_cycle_ms(3) == 90.0);
        DepthController fixed(1);
        fixed.observe(1, 0, 100.0);
        require(fixed.depth() == 1 && !fixed.warming_up() && !fixed.should_exit());
        require(!fixed.measured_cycle_ms(0));
        DepthController maintenance(3);
        maintenance.observe(3, 0, 9999.0, false);
        require(maintenance.depth() == 3 && maintenance.warming_up());
        require(!maintenance.measured_cycle_ms(3));
        rejects([&] { controller.conditional_acceptance(-1); });
        rejects([&] { controller.measured_cycle_ms(4); });
    }
    {
        PolicyState state;
        GenerationPolicy<DsparkDepthController> policy(3, false, &state);
        for (int cycle = 0; cycle < 64 && !policy.parked(); ++cycle) {
            const int depth = policy.depth();
            policy.observe(depth, 0, depth ? 100.0 : 20.0);
        }
        require(policy.parked() && policy.parks() == 1 && policy.depth() == 0);
        for (int token = 0; token < 127; ++token) policy.observe_plain(19.0);
        require(policy.parked() && policy.reentries() == 0);
        policy.observe_plain(19.0);
        require(!policy.parked() && policy.reentries() == 1 && policy.depth() == 3);
        for (int cycle = 0; cycle < 64 && !policy.parked(); ++cycle) {
            const int depth = policy.depth();
            policy.observe(depth, 0, depth ? 100.0 : 20.0);
        }
        require(policy.parked() && policy.parks() == 2);
        for (int token = 0; token < 255; ++token) policy.observe_plain(19.0);
        require(policy.parked());
        policy.observe_plain(19.0);
        require(!policy.parked() && policy.reentries() == 2);
        for (int cycle = 0; cycle < 500; ++cycle) {
            if (policy.parked()) policy.observe_plain(20.0);
            else policy.observe(policy.depth(), policy.depth(), policy.depth() ? 21.0 : 20.0);
        }
        require(!policy.parked() && policy.depth() > 0);
        GenerationPolicy<DsparkDepthController> legacy(3, true, &state);
        for (int cycle = 0; cycle < 100; ++cycle)
            legacy.observe(legacy.depth(), 0, legacy.depth() ? 100.0 : 20.0);
        require(!legacy.parked() && legacy.parks() == 0 && legacy.reentries() == 0);
    }
    {
        PolicyState state;
        DepthController controller(2);
        while (controller.warming_up()) controller.observe(controller.depth(), 0, 30.0);
        ParkState park;
        {
            mfq::engine::PrefillActivity activity;
            require(mfq::engine::PrefillActivity::recent());
            park.park(controller);
            require(park.baseline_ms == 0.0);
            park.tokens_remaining = 0;
            require(!park.probe_ready());
        }
        require(mfq::engine::PrefillActivity::recent() && !park.probe_ready());
    }
    {
        DsparkDepthController controller(3);
        require(controller.depth() == 3);
        controller.observe(3, 3, 30.0);
        controller.observe(3, 3, 29.0);
        controller.observe(0, 0, 22.0);
        controller.observe(0, 0, 21.0);
        controller.observe(0, 0, 23.0);
        require(controller.depth() == 3);
        require(controller.measured_cycle_ms(3).has_value());
        require(controller.conditional_acceptance(0) > 0.6);
    }
    {
        DsparkDepthController controller(1);
        controller.observe(1, 0, 80.0);
        controller.observe(1, 0, 75.0);
        controller.observe(0, 0, 40.0);
        controller.observe(0, 0, 39.0);
        controller.observe(0, 0, 41.0);
        require(controller.depth() == 0);
        require(controller.measured_cycle_ms(0).has_value());
    }
    {
        DsparkDepthController controller(3);
        controller.observe(3, 0, 70.0);
        controller.observe(3, 0, 65.0);
        controller.observe(0, 0, 30.0);
        controller.observe(0, 0, 29.0);
        controller.observe(0, 0, 31.0);
        require(controller.depth() == 0);
        // Depth zero is a reversible scheduling choice, not a permanent exit
        // from the common MTP state machine. Periodic exploration must remain
        // able to probe speculation again when runtime conditions change.
        bool probed_speculation = false;
        for (int cycle = 0; cycle < 240; ++cycle) {
            const int used_depth = controller.depth();
            if (used_depth > 0) {
                probed_speculation = true;
                break;
            }
            controller.observe(0, 0, 30.0);
        }
        require(probed_speculation);
    }
    {
        const std::array<int32_t, 4> drafts{11, 12, 13, 14};
        const std::array<int32_t, 5> partial{11, 12, 99, 14, 15};
        const auto result = verify_greedy(drafts, partial);
        require(result.accepted_drafts == 2 && result.next_token == 99 && !result.bonus);
        const std::array<int32_t, 5> complete{11, 12, 13, 14, 15};
        const auto bonus_result = verify_greedy(drafts, complete);
        require(bonus_result.accepted_drafts == 4 && bonus_result.next_token == 15 && bonus_result.bonus);
        rejects([&] { verify_greedy(drafts, std::span<const int32_t>(partial).first(4)); });
        rejects([&] { verify_greedy(std::span<const int32_t>{}, std::span<const int32_t>{}); });
    }
    {
        const std::array<int32_t, 2> drafts{0, 1};
        const std::vector<std::vector<float>> proposal{
            {.5f, .5f}, {.25f, .75f}};
        const std::vector<std::vector<float>> target{
            {.75f, .25f}, {.5f, .5f}, {0.f, 1.f}};
        const std::array<double, 2> accept_all{.5, .5};
        const auto accepted = verify_stochastic_chain(
            drafts, proposal, target, accept_all, .5);
        require(accepted.accepted_drafts == 2 && accepted.next_token == 1 && accepted.bonus);
        const std::array<double, 2> reject_second{.5, .95};
        const auto rejected = verify_stochastic_chain(
            drafts, proposal, target, reject_second, .5);
        require(rejected.accepted_drafts == 1 && rejected.next_token == 0 && !rejected.bonus);
    }
    {
        const std::array<int32_t, 2> drafts{10, 20};
        const std::vector<CompactDistribution> proposal{
            {{10, 20}, {.5f, .5f}}, {{10, 20}, {.25f, .75f}}};
        const std::vector<CompactDistribution> target{
            {{10, 20}, {.75f, .25f}},
            {{10, 20}, {.5f, .5f}},
            {{10, 20}, {0.f, 1.f}}};
        const std::array<double, 2> accept_all{.5, .5};
        const auto accepted = verify_compact_chain(
            drafts, proposal, target, accept_all, .5);
        require(accepted.accepted_drafts == 2 &&
                accepted.next_token == 20 && accepted.bonus);
        const std::array<double, 2> reject_second{.5, .95};
        const auto rejected = verify_compact_chain(
            drafts, proposal, target, reject_second, .5);
        require(rejected.accepted_drafts == 1 &&
                rejected.next_token == 10 && !rejected.bonus);

        const std::array<float, 2> logits{2.f, 1.f};
        const std::array<int64_t, 2> indices{9, 3};
        const auto compact = compact_distribution_from_topk(
            logits.data(), indices.data(), 2, 1.0, 1.0);
        require(compact.tokens == std::vector<int32_t>({3, 9}));
        require(sample_compact(compact, 0.0) == 3);
        rejects([&] {
            compact_distribution_from_topk(
                logits.data(), indices.data(), 2, 0.0, 1.0);
        });
    }
    const std::array<float, 3> q{.5f, .25f, .25f}, p{.25f, .5f, .25f}, bonus{0.f, 0.f, 1.f};
    require(verify(0, q, p, bonus, .49, .2).accepted);
    require(verify(0, q, p, bonus, .49, .2).next_token == 2);
    require(!verify(0, q, p, bonus, .5, .2).accepted);
    require(verify(0, q, p, bonus, .5, .2).next_token == 1);
    require(verify(1, q, p, bonus, .999, .2).accepted);
    require(sample(bonus, 0.) == 2 && sample(bonus, 1.) == 2);
    rejects([&] { verify(-1, q, p, bonus, .2, .3); });
    rejects([&] { verify(0, bonus, p, bonus, .2, .3); });
    rejects([&] { verify(0, q, p, bonus, NAN, .3); });
    rejects([&] { sample(std::array<float, 3>{1.f, 0.f, NAN}, 0.); });
    rejects([&] { sample(std::array<float, 3>{.1f, .1f, .1f}, 0.); });
    const std::array<float, 4> ties{1.f, 1.f, 1.f, 0.f};
    auto top = distribution(ties, 1., 2, 1.);
    require(top[0] == .5f && top[1] == .5f && top[2] == 0.f && top[3] == 0.f);
    auto nucleus = distribution(ties, 1., 2, .4);
    require(nucleus[0] == 1.f && nucleus[1] == 0.f);
    auto all = distribution(ties, 1., 0, .1);
    require(all[0] > 0.f && all[1] > 0.f && all[2] > 0.f && all[3] > 0.f);
    rejects([&] { distribution(ties, 0., 2, 1.); });
    rejects([&] { distribution(ties, 1., 5, 1.); });

    // Enumerate a deterministic uniform grid: accepted drafts plus residual
    // replacements must recover p, despite a deliberately biased proposal q.
    std::array<int, 3> histogram{};
    for (int draw = 0; draw < 400; ++draw) {
        const auto draft = sample(q, (draw + .5) / 400.);
        for (int accept = 0; accept < 100; ++accept) {
            const auto result = verify(draft, q, p, bonus, (accept + .5) / 100., .5);
            ++histogram[result.accepted ? draft : result.next_token];
        }
    }
    require(histogram[0] == 10000 && histogram[1] == 20000 && histogram[2] == 10000);

    GenerationStats stats;
    stats.used = true;
    stats.cycles = 3;
    stats.drafted_tokens = 4;
    stats.accepted_tokens = 2;
    std::vector<std::pair<std::string, double>> metrics;
    append_generation_metrics(metrics, stats);
    const auto acceptance = std::find_if(
        metrics.begin(), metrics.end(), [](const auto& metric) {
            return metric.first == "mtp_acceptance_rate";
        });
    require(acceptance != metrics.end() && acceptance->second == .5);
    std::cout << "MTP sampling/acceptance/correction tests passed; 40000 exact grid checks\n";
}
