#include "mfq/moe_dispatch_plan.h"
#include "moe_residency_rank_checks.h"
#include <limits>
#include "moe_cpu_cost_checks.h"
#include "moe_dispatch_reuse_checks.h"
#include <iostream>
#include <stdexcept>

int main() try {
    using K = mfq::MoeDispatchKind;
    const std::vector<std::int32_t> ids{2, 0, 1, 0, 3, 2, 4, 1};
    const auto hot = [](int e) { return e == 2; };
    const auto all = [](int) { return true; };
    const auto p = mfq::plan_moe_dispatch(ids, 5, 2, hot, all);
    if (p.groups.size() != 5 || p.kinds != std::vector<K>{K::GpuResident, K::Cpu, K::Cpu, K::Cpu,
            K::GpuTransfer, K::GpuResident, K::GpuTransfer, K::Cpu})
        throw std::runtime_error("MFQ ordered miss quota or duplicate route decision differs");
    if (p.groups[1].positions != std::vector<std::size_t>{1, 3})
        throw std::runtime_error("duplicate output positions were lost");
    const auto tiered = mfq::plan_moe_dispatch(ids, 5, ids.size(), hot, [](int e) { return e != 0 && e != 1; });
    if (tiered.kinds != p.kinds) throw std::runtime_error("RAM cold eligibility differs");
    for (auto quota : {std::size_t(0), std::size_t(5), std::size_t(1000)}) {
        const auto q = mfq::plan_moe_dispatch(ids, 5, quota, hot, all);
        for (std::size_t i = 0; i < ids.size(); ++i)
            if (q.kinds[i] != (hot(ids[i]) ? K::GpuResident : quota ? K::GpuTransfer : K::Cpu))
                throw std::runtime_error("zero/full transfer quota differs");
    }
    bool failed = false;
    try { mfq::plan_moe_dispatch(std::vector<std::int32_t>{2, -1}, 5, 5, hot, all); }
    catch (const std::out_of_range&) { failed = true; }
    if (!failed) throw std::runtime_error("invalid route accepted");
    if (!mfq::plan_moe_dispatch({}, 5, 0, hot, all).groups.empty()) throw std::runtime_error("empty route differs");
    const auto skipped = mfq::plan_moe_dispatch(std::vector<std::int32_t>{0, 1, 3}, 5, 1,
        [](int) { return false; }, [](int e) { return e == 1; });
    if (skipped.kinds != std::vector<K>{K::Cpu, K::GpuTransfer, K::Cpu})
        throw std::runtime_error("ineligible tail route consumed transfer quota");
    int cost_cases=0;
    for(double bandwidth:{0.01,1.0,26.9})for(double stage:{0.0,0.1,1.0})
        for(double gpu:{0.0,1e5,1e6,1e7})for(double cpu_rate:{1e4,1e6,1e8})
            for(bool budget:{false,true}) {
            std::vector<mfq::MoeDispatchCost> costs={{cpu_rate,500000,true},{cpu_rate,2000000,true},
                {cpu_rate,1000000,false},{cpu_rate,3000000,true},{cpu_rate,750000,true}};
            auto selected=mfq::select_moe_cpu_work(costs,{bandwidth,stage,gpu,budget});
            const auto duration=[&](unsigned mask) {
                double cpu=0,bytes=0;
                for(std::size_t i=0;i<costs.size();++i) {
                    if(mask&(1u<<i)) {
                        if(!costs[i].cpu_eligible || (budget && costs[i].cpu_ns>=
                            costs[i].transfer_bytes*(stage+1.0/bandwidth)))return 1e100;
                        cpu+=costs[i].cpu_ns;
                    } else bytes+=costs[i].transfer_bytes;
                }
                return stage*bytes+std::max(cpu,gpu+bytes/bandwidth);
            };
            unsigned actual=0;for(std::size_t i=0;i<selected.size();++i)if(selected[i])actual|=1u<<i;
            double oracle=1e100;for(unsigned mask=0;mask<(1u<<costs.size());++mask)oracle=std::min(oracle,duration(mask));
            if(std::abs(duration(actual)-oracle)>1e-6*std::max(1.0,oracle))
                throw std::runtime_error("measured dispatch differs from exhaustive equal-work optimum");
            ++cost_cases;
        }
    // A long resident GPU interval cannot make a slower CPU expert cheap.
    if(mfq::select_moe_cpu_work({{200,100,true}},{1,0,10000,true})!=std::vector<bool>{false} ||
       mfq::select_moe_cpu_work({{200,100,true}},{1,0,10000,false})!=std::vector<bool>{true} ||
       mfq::select_moe_cpu_work({{50,100,true},{200,100,true},{1,0,true}},
           {1,0,10000,true})!=std::vector<bool>{true,false,false} ||
       mfq::select_moe_cpu_work({{100,100,true}},{1,0,10000,true})!=std::vector<bool>{false})
        throw std::runtime_error("CPU transfer budget admitted slow or cached work");
    std::cout<<"MFQ CPU transfer budget boundary cases=4 PASS\n";
    std::cout<<"MFQ reused dispatch changing routes/tiers/eligibility, duplicate positions, quotas, invalid guards and allocation reuse cases="<<dispatch_reuse_checks()<<" PASS\n";
    for(auto rates:{mfq::MoeDispatchRates{0,0,0},mfq::MoeDispatchRates{1,-1,0},
                   mfq::MoeDispatchRates{1,0,-1}}) {
        bool rejected=false;
        try{mfq::select_moe_cpu_work({{1,1,true}},rates);}catch(const std::invalid_argument&){rejected=true;}
        if(!rejected)throw std::runtime_error("invalid measured dispatch rates accepted");
    }
    std::cout<<"MFQ measured CPU/PCIe dispatch oracle cases="<<cost_cases<<'\n';
    std::cout<<"MFQ cached residency ordered candidate cases="<<residency_rank_checks()<<'\n';
    std::cout<<"MFQ shared CPU cost history cases="<<cpu_cost_checks()<<'\n';
    std::cout << "MFQ ordered GPU-hot/CPU-cold/PCIe dispatch policy passed\n";
    return 0;
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
