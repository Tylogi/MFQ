#pragma once

#include <cstddef>
#include <cstdint>
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
// Operator and storage controls are parsed only at this execution boundary.
std::optional<int> embedding_mapped();
std::optional<bool> gdn_inplace_state();
std::optional<bool> gdn_prefill_fused();
std::optional<bool> gr_dense_vector();
std::optional<bool> mfe_group_dot();
std::optional<bool> mfe_nint_late_scale();
std::optional<bool> mfe_nvq_word_signs();
std::optional<bool> mfe_shared_dense_math();
std::optional<bool> moe_assert_resident();
std::optional<bool> moe_nvq_dense();
std::optional<bool> moe_plan_reuse();
std::optional<bool> moe_plan_reuse_check();
std::optional<bool> moe_prefill_compact_mma();
std::optional<bool> moe_prefill_direct_ram();
std::optional<bool> moe_prefill_expert_batch();
std::optional<std::int64_t> moe_prefill_expert_batch_tokens();
std::optional<bool> moe_prefill_graph();
std::optional<bool> moe_prefill_groups();
std::optional<bool> moe_prefill_layer();
std::optional<bool> moe_prefill_layer_async();
std::optional<bool> moe_prefill_layer_flush();
std::optional<bool> moe_prefill_layer_major();
std::optional<bool> moe_prefill_layer_overlap();
std::optional<bool> moe_prefill_layer_phased();
std::optional<bool> moe_prefill_layer_reuse_async();
std::optional<bool> moe_prefill_pool_retain();
std::optional<bool> moe_preload_all();
bool nint_dense_reference();
std::optional<bool> nint_direct_pool();
std::optional<bool> nint_group_dot();
std::optional<bool> nint_route_hint();
std::optional<bool> nint_route_row_workspace();
std::optional<int> nint_route_warps();
std::optional<bool> nvq1_group_records();
std::optional<bool> nvq1_integer_delta();
bool nvq_dense_reference();
std::optional<bool> ple_file_map();
std::optional<bool> prefill_expert_batch_audit();
std::optional<bool> prefill_layer_major_trace();
std::optional<bool> qsa_prefill_fused();
std::optional<bool> qsa_select_fused();
std::optional<bool> qsa_sparse_gate_fused();
std::optional<bool> shared_gate_fused();
std::optional<bool> trace_mfe_dispatch();
std::optional<bool> trace_mfe_dispatch_requested();
std::optional<bool> trace_moe_serve();
std::optional<bool> trace_ple_timings();
int moe_prefill_group_size();
} // namespace mfq::cuda::runtime_options
