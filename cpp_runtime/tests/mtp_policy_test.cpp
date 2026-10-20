#include "mtp_metrics.h"
#include <array>
#include <iostream>
#include <source_location>

namespace {
void require(bool ok, const std::source_location location = std::source_location::current()) {
    if (!ok) {
        std::cerr << "MTP policy test failed at line " << location.line() << std::endl;
        throw std::runtime_error("MTP policy test failed");
    }
}
template<class F> void rejects(F&& fn) {
    bool rejected = false;
    try { fn(); } catch (const std::exception&) { rejected = true; }
    require(rejected);
}
}

int main() {
    using namespace mfq::engine::mtp;
    {
        // Keep CUDA on the previously measured policy, including host gaps,
        // excluded maintenance, parking and retry.
        using Clock=BatchDepthPolicy::Clock;
        GenerationPolicy<DsparkDepthController> policy(3,false,nullptr,true,true);
        v1::BatchDepthPolicy previous({"cuda-mtp"},3);
        Clock::time_point cursor{};
        std::uint64_t reentries=0;
        for(int cycle=0;cycle<1600;++cycle) {
            require(policy.depth()==previous.depth());
            const int depth=policy.depth();
            const bool parked=previous.remaining()!=0;
            const bool valid=cycle%31!=17;
            const double ms=!valid?9000.0:depth==0?20.0:cycle<500?10.0+2.0*depth:35.0+30.0*depth;
            const auto started=cursor+std::chrono::microseconds(400);
            cursor=started+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double,std::milli>(ms));
            const std::array<int,1> accepted{depth};
            previous.observe(depth,accepted,started,cursor,valid,false);
            if(parked)policy.observe_plain_window(started,cursor,valid);
            else policy.observe_window(depth,depth,started,cursor,valid);
            if(parked && !previous.remaining())++reentries;
            require(policy.depth()==previous.depth() && policy.parked()==bool(previous.remaining()));
            require(policy.parks()==previous.parks() && policy.reentries()==reentries);
            for(int width=0;width<=3;++width)require(policy.measured_cycle_ms(width)==previous.cost(width));
            if(cycle==499)require(policy.parks()==0);
        }
        require(policy.parks()>0 && policy.reentries()>0);
        std::cout<<"CUDA v1 policy matches 1600 timed windows, exclusions and retries\n";
    }
    {
        // Wide-window caches can warm without a 10% change in scalar cost.
        // Use complete four-cycle costs, including the periodic expensive
        // exchange, before comparing speculation with the unchanged baseline.
        for(int phase=0;phase<4;++phase)for(bool profitable:{false,true}) {
            DepthController controller(3,nullptr,7.0,1.03,true);
            while(controller.warming_up()) {
                const int depth=controller.depth();
                controller.observe(depth,depth,depth?40.0+30.0*depth:20.0);
            }
            for(int i=0;i<128;++i)controller.observe(3,3,130.0);
            const double normal=profitable?48.0:68.0,expected=normal*1.25;
            for(int i=0;i<4;++i) {
                controller.observe(3,3,(i+phase)%4==3?normal*2:normal);
                const auto measured=*controller.measured_cycle_ms(3);
                controller.observe(3,3,9000.0,false);
                require(*controller.measured_cycle_ms(3)==measured);
            }
            std::cout<<"wide cache-warm cost phase="<<phase<<" measured="<<*controller.measured_cycle_ms(3)
                <<" expected="<<expected<<" losing="<<controller.speculation_losing()<<std::endl;
            require(std::abs(*controller.measured_cycle_ms(3)-expected)<1e-12);
            require(controller.speculation_losing()!=profitable);
        }
    }
    {
        // Scalar decode warms substantially after the first wide replay.
        // Revalidate that old wide cost before parking against a hot baseline.
        for(bool profitable:{false,true}) {
        DepthController controller(3,nullptr,7.0,1.03,true);
        while(controller.warming_up()) {
            const int depth=controller.depth();
            controller.observe(depth,depth,depth==0?30.0:depth==1?40.0:depth==2?55.0:70.0);
        }
        require(*controller.measured_cycle_ms(3)==70.0);
        for(int i=0;i<4;++i)controller.observe(0,0,15.0);
        for(int i=0;i<24;++i) {
            const int depth=controller.depth();
            controller.observe(depth,depth,depth==0?15.0:profitable?(depth==1?24.0:depth==2?34.0:40.0):
                depth==1?40.0:depth==2?55.0:70.0);
        }
        std::cout<<"cache-warm revalidation baseline="<<*controller.measured_cycle_ms(0)
            <<" wide="<<*controller.measured_cycle_ms(3)<<" depth="<<controller.depth()<<std::endl;
        require(*controller.measured_cycle_ms(3)==(profitable?40.0:70.0) &&
            controller.speculation_losing()!=profitable && (controller.depth()>0)==profitable);
        }
        DepthController stable(3,nullptr,7.0,1.03,true);
        while(stable.warming_up()) {
            const int depth=stable.depth();stable.observe(depth,depth,depth==0?30.0:40.0+10.0*depth);
        }
        for(int i=0;i<4;++i)stable.observe(0,0,27.6);
        for(int i=0;i<12;++i) {
            const int depth=stable.depth();stable.observe(depth,depth,depth==0?27.6:40.0+10.0*depth);
        }
        require(*stable.measured_cycle_ms(3)==70.0);
    }
    {
        // The plain loop also pays the four-window expert-exchange cycle.
        // A median-based loop tax must not make profitable MTP park again.
        for(int phase=0;phase<4;++phase)for(double speculative_ms:{38.0,44.0}) {
            PolicyState state;
            // Isolate time comparison from a fresh 60% acceptance prior.
            DepthController seed(1,nullptr,7.0,1.03,true);
            while(seed.warming_up()) {
                const int depth=seed.depth();seed.observe(depth,depth,depth?100.0:20.0);
            }
            for(int i=0;i<128;++i)seed.observe(1,1,100.0);
            state.remember(seed);
            GenerationPolicy<DsparkDepthController> policy(1,false,&state,true);
            for(int cycle=0;cycle<256 && !policy.parked();++cycle) {
                const int depth=policy.depth();policy.observe(depth,depth,depth?100.0:20.0);
            }
            require(policy.parked());
            for(int i=0;i<128;++i)policy.observe_plain((i+phase)%4==3?32.0:16.0);
            require(!policy.parked() && policy.reentries()==1);
            int baseline=0;
            for(int cycle=0;cycle<128 && !policy.parked();++cycle) {
                const int depth=policy.depth();
                policy.observe(depth,depth,depth?speculative_ms:(++baseline%4?16.0:32.0));
            }
            std::cout<<"periodic plain-loop tax speculative_ms="<<speculative_ms
                <<" parked="<<policy.parked()<<std::endl;
            require(policy.parked()==(speculative_ms>40.0));
        }
    }
    {
        GenerationPolicy<DsparkDepthController> policy(3,false,nullptr,true);
        for(int cycle=0;cycle<300 && !policy.parked();++cycle) {
            const auto depth=policy.depth();
            policy.observe(depth,depth,depth?80.0+30.0*depth:20.0);
        }
        require(policy.parked() && *policy.measured_cycle_ms(0)==20.0);
        for(int i=0;i<64;++i)policy.observe_plain(40.0);
        require(policy.parked() && *policy.measured_cycle_ms(0)>39.9);
        const auto baseline=*policy.measured_cycle_ms(0);
        policy.observe_plain(9000.0,false);
        require(*policy.measured_cycle_ms(0)==baseline);
        for(int i=0;i<62;++i)policy.observe_plain(40.0);
        require(policy.parked() && policy.reentries()==0);
        policy.observe_plain(40.0);
        require(!policy.parked() && policy.reentries()==1 && policy.depth()==1);
        require(*policy.measured_cycle_ms(0)>39.9);
    }
    {
        DepthController controller(3);
        const std::array<int, 9> sweep{3, 2, 1, 0, 0, 0, 3, 2, 1};
        for (int depth : sweep) {
            require(controller.depth() == depth);
            controller.observe(depth, depth, depth ? 20.0 + depth : 20.0);
        }
        require(!controller.warming_up());
        require(std::fabs(controller.conditional_acceptance(0)-(1.0-0.4*std::pow(0.92,6))) < 1e-12);
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
        // Reproduce the cold 8K request: 657 ms at depth 3, then progressively
        // faster lower widths. A short startup resweep must replace that cost
        // without waiting five seconds for stale-depth exploration.
        DepthController controller(3);
        for (const auto [depth, ms] : std::array<std::pair<int,double>,6>{
                {{3,657.0},{2,189.0},{1,60.0},{0,60.0},{0,30.0},{0,20.0}}}) {
            require(controller.depth()==depth);
            controller.observe(depth,depth,ms);
        }
        require(controller.warming_up() && controller.depth()==3);
        controller.observe(3,3,9000.0,false);
        require(controller.depth()==3 && *controller.measured_cycle_ms(3)==657.0);
        for (int depth : {3,2,1}) {
            require(controller.depth()==depth);
            controller.observe(depth,depth,20.0+5.0*depth);
        }
        require(!controller.warming_up() && controller.depth()>1);
        require(*controller.measured_cycle_ms(3)==35.0);
        // Consistently slow speculation still loses; calibration must not
        // force MTP when its actual per-token cost exceeds ordinary decode.
        DepthController slow(3);
        while(slow.warming_up()) {
            const auto depth=slow.depth();
            slow.observe(depth,depth,depth?200.0:20.0);
        }
        require(slow.depth()==0 && slow.speculation_losing());
        DepthController limited(3);
        limited.observe(1,1,20.0);
        require(limited.depth()==3 && limited.warming_up());
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
        DepthController latency(3,nullptr,7.0,1.15,true,60.0);
        require(latency.depth()==1);
        latency.observe(1,1,45.0); // Projected three-row window is still cold.
        require(latency.depth()==0);
        latency.observe(0,0,1.0,false); // Maintenance cannot advance calibration.
        require(latency.depth()==0 && !latency.measured_cycle_ms(0));
        for(int i=0;i<4;++i)latency.observe(0,0,30.0);
        require(latency.depth()==1);
        latency.observe(1,1,39.0);
        require(latency.depth()==2);
        latency.observe(2,2,51.0);
        require(latency.depth()==2);
        latency.observe(2,2,44.0);
        latency.observe(1,1,39.0);
        require(latency.depth()==3);
        DepthController sampled(3,nullptr,7.0,1.15,true,60.0);
        int calibration=0;
        while(sampled.warming_up() && calibration<20) {
            const auto depth=sampled.depth();
            sampled.observe(depth,0,depth?38.0:25.0);
            ++calibration;
        }
        require(calibration<14 && !sampled.warming_up() && sampled.depth()==0);
        require(!sampled.measured_cycle_ms(2) && !sampled.measured_cycle_ms(3));
        for(int i=0;i<5;++i) {
            require(!sampled.should_exit());sampled.observe(0,0,25.0);
        }
        require(!sampled.should_exit());sampled.observe(0,0,25.0);
        require(sampled.should_exit());
        // Deferred widths remain eligible for exploration when conditions change.
        bool revisited=false;
        for(int i=0;i<400;++i) {
            const int depth=sampled.depth();
            if(depth>1){revisited=true;break;}
            sampled.observe(depth,depth,depth?28.0:25.0);
        }
        require(revisited);
        DepthController refreshed(3,nullptr,7.0,1.15,true);
        int baselines=0;
        while(refreshed.warming_up()) {
            const int depth=refreshed.depth();
            const double ms=depth?50.0+depth*5.0:(++baselines<=4?60.0:25.0);
            refreshed.observe(depth,0,ms);
        }
        require(baselines==8 && *refreshed.measured_cycle_ms(0)==25.0 && refreshed.depth()==0);
        DepthController gradual(3,nullptr,7.0,1.15,true);
        const std::array<int,18> sweep{1,0,0,0,0,1,2,3,3,2,1,3,2,1,0,0,0,0};
        int step=0;
        for(int depth:sweep) {
            require(gradual.depth()==depth);
            gradual.observe(depth,depth,1000.0,false);
            require(gradual.depth()==depth && gradual.warming_up());
            const double ms=step==0?70.0:20.0+5.0*depth;
            gradual.observe(depth,depth,ms);
            if(++step==9)require(gradual.warming_up());
        }
        require(!gradual.warming_up() && gradual.depth()>1);
        require(*gradual.measured_cycle_ms(3)==35.0);
        DepthController continuation(3,&gradual,7.0,1.15,true);
        for(int depth:std::array<int,15>{1,0,0,0,0,1,2,3,3,2,1,0,0,0,0}) {
            require(continuation.depth()==depth);
            continuation.observe(depth,depth,20.0+5.0*depth);
        }
        require(!continuation.warming_up());
        PolicyState state;
        GenerationPolicy<DsparkDepthController> policy(3,false,&state,true);
        require(policy.depth()==1);
        for(int cycle=0;cycle<96 && !policy.parked();++cycle)
            policy.observe(policy.depth(),0,policy.depth()?100.0:20.0);
        require(policy.parked());
        for(int i=0;i<128;++i)policy.observe_plain(20.0);
        require(!policy.parked() && policy.depth()==1);
    }
    {
        // One maintenance-sized exchange every four windows is real decode
        // work. Its average is 20 ms, so a 19 ms/token speculative path wins.
        // Taking the minimum 16 ms incorrectly parks that profitable path.
        for(const double speculative_ms:{38.0,45.0}) {
            DepthController periodic(1,nullptr,7.0,1.03,true);
            int baseline=0;
            while(periodic.warming_up()) {
                const auto depth=periodic.depth();
                periodic.observe(depth,depth,9000.0,false);
                periodic.observe(depth,depth,depth?speculative_ms:(++baseline%4?16.0:32.0));
            }
            require(*periodic.measured_cycle_ms(0)==20.0);
            for(int i=0;i<128;++i)periodic.observe(1,1,speculative_ms);
            require(periodic.speculation_losing()==(speculative_ms>40.0));
            DepthController inherited(1,&periodic,7.0,1.03,true);
            baseline=0;
            while(inherited.warming_up()) {
                const auto depth=inherited.depth();
                inherited.observe(depth,depth,depth?speculative_ms:(++baseline%4?16.0:32.0));
            }
            require(*inherited.measured_cycle_ms(0)==20.0);
        }
    }
    {
        // Cold calibration says 25 ms; a warm scalar path actually takes
        // 10 ms. It is initially too far behind to be selected as a rival.
        // A mandatory periodic probe must discover that speculation loses.
        DepthController warmed(1,nullptr,7.0,1.03,true);
        while(warmed.warming_up()) {
            const auto depth=warmed.depth();warmed.observe(depth,depth,depth?35.0:25.0);
        }
        int plain_probes=0;
        for(int cycle=0;cycle<100;++cycle) {
            const auto depth=warmed.depth();
            plain_probes+=depth==0;
            warmed.observe(depth,depth,depth?35.0:10.0);
        }
        require(plain_probes>=4 && *warmed.measured_cycle_ms(0)==10.0);
        require(warmed.depth()==0 && warmed.speculation_losing());
    }
    {
        // A shared predictor cost makes depth-one extrapolation pessimistic:
        // 1.875 / 32.4 loses against 1 / 15.9, while wider windows can win.
        DepthController controller(3,nullptr,7.0,1.15,true,60.0);
        bool tried_two=false,tried_three=false;
        for(int cycle=0;cycle<120 && (controller.warming_up() || !tried_three);++cycle) {
            const int depth=controller.depth();
            tried_two=tried_two || depth==2;
            tried_three=tried_three || depth==3;
            const double ms=depth==0?15.9:depth==1?32.4:depth==2?38.0:44.0;
            controller.observe(depth,depth,ms);
            require(!controller.should_exit());
        }
        require(tried_two && tried_three);
        require(controller.measured_cycle_ms(2) && controller.measured_cycle_ms(3));
        while(controller.warming_up()) {
            const int depth=controller.depth();
            controller.observe(depth,depth,depth==0?15.9:depth==1?32.4:depth==2?38.0:44.0);
        }
        require(*controller.measured_cycle_ms(2)==38.0 && *controller.measured_cycle_ms(3)==44.0);
    }
    {
        PolicyState state;
        auto cuda=state.begin(1,true),generic=state.begin(1,false);
        for(int cycle=0;cycle<128;++cycle) {
            for(auto* controller:{&cuda,&generic}) {
                const auto depth=controller->warming_up()?controller->depth():1;
                controller->observe(depth,depth,depth?28.0:15.0);
            }
        }
        cuda.observe(0,0,15.0);generic.observe(0,0,15.0);
        // 2/28 is 7.1% faster than 1/15. CUDA must retain that benefit;
        // the generic backend still requires its original 15% margin.
        require(!cuda.speculation_losing() && !cuda.should_exit());
        require(generic.speculation_losing());
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
        require(controller.depth() == 2);
        controller.observe(2, 2, 30.0);
        controller.observe(2, 2, 29.0);
        controller.observe(0, 0, 22.0);
        controller.observe(0, 0, 21.0);
        controller.observe(0, 0, 23.0);
        controller.observe(2,2,29.0);
        for(int cycle=0;cycle<12;++cycle) {
            const int depth=controller.depth();controller.observe(depth,depth,26.0+depth);
        }
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
        controller.observe(1, 0, 75.0);
        require(controller.depth() == 0);
        require(controller.measured_cycle_ms(0).has_value());
    }
    {
        DsparkDepthController controller(3);
        controller.observe(2, 0, 70.0);
        controller.observe(2, 0, 65.0);
        controller.observe(0, 0, 30.0);
        controller.observe(0, 0, 29.0);
        controller.observe(0, 0, 31.0);
        controller.observe(2,0,62.0);
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
        DsparkDepthController controller(3);
        controller.observe(2,2,9000.0,false);
        require(controller.depth()==2 && !controller.measured_cycle_ms(2));
        controller.observe(1,1,20.0);
        require(controller.depth()==2 && !controller.measured_cycle_ms(2));
        for (const auto [depth,ms] : std::array<std::pair<int,double>,6>{
                {{2,700.0},{2,657.0},{0,60.0},{0,30.0},{0,20.0},{2,35.0}}}) {
            require(controller.depth()==depth);
            controller.observe(depth,depth,ms);
        }
        require(controller.depth()>0 && *controller.measured_cycle_ms(2)==35.0);
    }
    {
        DsparkDepthController controller(5);
        require(controller.depth()==2 && controller.maximum_depth()==5);
        bool explored_maximum=false;
        for(int cycle=0;cycle<1500;++cycle) {
            const int depth=controller.depth();explored_maximum|=depth==5;
            controller.observe(depth,depth,depth?24.0+2*depth:20.0);
        }
        require(explored_maximum && controller.depth()==5);
        require(DsparkDepthController(1).depth()==1);
        require(DsparkDepthController(3,5).depth()==3);
        require(DsparkDepthController(3,0).depth()==1);
    }
    {
        // SSD verification can cost several ordinary decode cycles. Charge
        // exploration against the width being probed, keeping costly losing
        // probes within their time budget while still retrying every width.
        DsparkDepthController controller(3);
        constexpr std::array<double,4> cost{200.0,600.0,900.0,1200.0};
        for(int i=0;i<8;++i) {
            const int depth=controller.depth();controller.observe(depth,0,cost[depth]);
        }
        require(controller.depth()==0);
        double elapsed=0,probing=0;std::array<int,4> visits{};
        while(elapsed<120000.0) {
            const int depth=controller.depth();
            elapsed+=cost[depth];if(depth)probing+=cost[depth];++visits[depth];
            controller.observe(depth,0,cost[depth]);
        }
        std::cout<<"DSpark costly probe time fraction="<<probing/elapsed<<std::endl;
        require(probing/elapsed<=0.18);
        require(visits[1]>=4 && visits[2]>=4 && visits[3]>=4);
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
    stats.target_ms=12.5;stats.head_ms=3.25;stats.rollback_ms=.5;
    std::vector<std::pair<std::string, double>> metrics;
    append_generation_metrics(metrics, stats);
    const auto acceptance = std::find_if(
        metrics.begin(), metrics.end(), [](const auto& metric) {
            return metric.first == "mtp_acceptance_rate";
        });
    require(acceptance != metrics.end() && acceptance->second == .5);
    for(const auto& [name,value]:std::vector<std::pair<std::string,double>>{
            {"mtp_target_ms",12.5},{"mtp_head_ms",3.25},{"mtp_rollback_ms",.5}}) {
        const auto found=std::find_if(metrics.begin(),metrics.end(),[&](const auto& metric){return metric.first==name;});
        require(found!=metrics.end() && found->second==value);
    }
    std::cout << "MTP sampling/acceptance/correction tests passed; 40000 exact grid checks\n";
}
