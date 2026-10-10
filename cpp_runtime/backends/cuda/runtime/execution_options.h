#pragma once

#include <cstddef>
#include <optional>
#include <string>

namespace mfq::cuda::runtime_options {
// Keep dynamic diagnostic/ablation options at the execution boundary. Values
// are read on access so changing an option before graph capture remains valid.
bool resident_plan_overlap();
std::optional<bool> rotary_fusion();
bool mhc_timings();
bool layer_timings();
int route_poll_interval_us();
bool single_window_flush();
bool early_no_cpu_ready();
bool skip_gpu_only_cpu_wait();
bool skip_fixed_gpu_timing();
bool window_input_views();
bool decode_host_pin();
bool dma_graph_batch();
bool warp_multi_sum();
float residency_heat_retention();
int residency_interval();
bool router_lookahead();
std::size_t workspace_reserve_bytes();
bool phased_transfer();
bool mapped_copy_overlap();
bool background_calibration();
bool early_gate_up();
bool parallel_gate_up();
bool cpu_transfer_budget();
bool transfer_cache();
bool mapped_copy();
struct MoeDiagnostics {
    bool cpu_rows = false, cpu_rows_fail_alloc = false;
    bool dma = false, dma_fail_alloc = false, shared_cpu_cost = false;
    std::string dispatch_record, miss_record;
    std::optional<std::string> dispatch_replay;
};
MoeDiagnostics moe_diagnostics();
} // namespace mfq::cuda::runtime_options
