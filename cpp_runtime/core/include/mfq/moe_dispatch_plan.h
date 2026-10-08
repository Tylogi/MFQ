#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace mfq {
enum class MoeDispatchKind { Cpu = -1, GpuResident = 0, GpuTransfer = 1 };
struct MoeDispatchGroup {
    std::int32_t expert = -1;
    MoeDispatchKind kind = MoeDispatchKind::Cpu;
    std::vector<std::size_t> positions;
};
struct MoeDispatchPlan {
    std::vector<MoeDispatchGroup> groups;
    std::vector<MoeDispatchKind> kinds;
};
struct MoeDispatchCost {
    double cpu_ns=0;
    std::uint64_t transfer_bytes=0;
    bool cpu_eligible=true;
};
struct MoeDispatchRates {
    double pcie_gbps=0,stage_ns_per_byte=0,gpu_ns=0;
    bool cpu_transfer_budget=false;
};
// Without a measured per-expert GPU saving, CPU work must repay its own
// avoided host staging and PCIe transfer time before using the GPU window.
inline bool moe_cpu_repays_transfer(const MoeDispatchCost& cost,MoeDispatchRates rates) {
    return cost.cpu_eligible && cost.cpu_ns>0 && cost.transfer_bytes>0 &&
        cost.cpu_ns<static_cast<double>(cost.transfer_bytes)*
            (rates.stage_ns_per_byte+1.0/rates.pcie_gbps);
}
inline std::vector<bool> select_moe_cpu_work(const std::vector<MoeDispatchCost>& costs,
        MoeDispatchRates rates) {
    if(!std::isfinite(rates.pcie_gbps) || rates.pcie_gbps<=0 ||
       !std::isfinite(rates.stage_ns_per_byte) || rates.stage_ns_per_byte<0 ||
       !std::isfinite(rates.gpu_ns) || rates.gpu_ns<0)
        throw std::invalid_argument("MoE dispatch needs finite measured rates");
    std::vector<std::size_t> order;
    double bytes=0;
    for(std::size_t i=0;i<costs.size();++i) {
        bytes+=static_cast<double>(costs[i].transfer_bytes);
        if(costs[i].cpu_eligible) {
            if(!std::isfinite(costs[i].cpu_ns) || costs[i].cpu_ns<=0)
                throw std::invalid_argument("MoE CPU work needs positive measured time");
            if(!rates.cpu_transfer_budget || moe_cpu_repays_transfer(costs[i],rates))order.push_back(i);
        }
    }
    std::stable_sort(order.begin(),order.end(),[&](auto a,auto b) {
        return costs[a].transfer_bytes/costs[a].cpu_ns>costs[b].transfer_bytes/costs[b].cpu_ns;
    });
    const auto duration=[&](double cpu,double transfer) {
        return transfer*rates.stage_ns_per_byte+
            std::max(cpu,rates.gpu_ns+transfer/rates.pcie_gbps);
    };
    double best=duration(0,bytes),cpu=0;std::size_t count=0;
    for(std::size_t i=0;i<order.size();++i) {
        const auto& work=costs[order[i]];cpu+=work.cpu_ns;bytes-=work.transfer_bytes;
        const auto predicted=duration(cpu,std::max(bytes,0.0));
        if(predicted<best){best=predicted;count=i+1;}
    }
    std::vector<bool> selected(costs.size(),false);
    for(std::size_t i=0;i<count;++i)selected[order[i]]=true;
    return selected;
}
inline double measured_pcie_share(double gbps) {
    if (!std::isfinite(gbps) || gbps <= 0)
        throw std::invalid_argument("PCIe split requires finite measured bandwidth");
    const double full_rate_share = 0.55;
    const double reference_gbps = 20.0;
    return full_rate_share * std::min(gbps / reference_gbps, 1.0);
}

template<class Resident, class TransferEligible>
MoeDispatchPlan plan_moe_dispatch(const std::vector<std::int32_t>& ids,
        int experts, std::size_t transfer_quota, Resident resident,
        TransferEligible transfer_eligible) {
    if (experts < 1) throw std::invalid_argument("expert count must be positive");
    // Validate the complete input before invoking placement callbacks.
    for (const auto id : ids)
        if (id < 0 || id >= experts) throw std::out_of_range("MoE expert ID is out of range");
    MoeDispatchPlan plan;
    plan.kinds.resize(ids.size(), MoeDispatchKind::Cpu);
    std::vector<std::size_t> group_for(static_cast<std::size_t>(experts), ids.size());
    for (std::size_t position = 0; position < ids.size(); ++position) {
        auto& group = group_for[static_cast<std::size_t>(ids[position])];
        if (group == ids.size()) {
            group = plan.groups.size();
            plan.groups.push_back({ids[position], MoeDispatchKind::Cpu, {}});
        }
        plan.groups[group].positions.push_back(position);
    }
    for (auto& group : plan.groups)
        if (resident(group.expert)) group.kind = MoeDispatchKind::GpuResident;
    for (auto at = plan.groups.rbegin(); at != plan.groups.rend() && transfer_quota; ++at) {
        if (at->kind != MoeDispatchKind::Cpu || !transfer_eligible(at->expert)) continue;
        at->kind = MoeDispatchKind::GpuTransfer;
        --transfer_quota;
    }
    for (const auto& group : plan.groups)
        for (const auto position : group.positions) plan.kinds[position] = group.kind;
    return plan;
}
}
