#pragma once
#include <cstddef>
#include <cstdint>
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
struct MoeDispatchWorkspace {
    std::vector<std::size_t> group_for;
};

} // namespace mfq
