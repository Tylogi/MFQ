#pragma once

#include "mfq_tensor_backend.h"
#include "tensor_parallel.h"

#include <cuda_runtime_api.h>
#ifdef MFQ_HAVE_NCCL
#include <nccl.h>
#endif

#include <chrono>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifndef MFQ_CUDA_CHECK
#define MFQ_CUDA_CHECK(expr) do { \
    cudaError_t err__ = (expr); \
    if (err__ != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err__)); \
    } \
} while (0)
#endif

#ifdef MFQ_HAVE_NCCL
#ifndef MFQ_NCCL_CHECK
#define MFQ_NCCL_CHECK(expr) do { \
    ncclResult_t err__ = (expr); \
    if (err__ != ncclSuccess) { \
        throw std::runtime_error(std::string("NCCL error: ") + ncclGetErrorString(err__)); \
    } \
} while (0)
#endif
#endif

class MoeExpertCache;

namespace mfq::cuda::internal {

class PrefillCudaTimer {
public:
    PrefillCudaTimer();
    ~PrefillCudaTimer();

    PrefillCudaTimer(const PrefillCudaTimer&) = delete;
    PrefillCudaTimer& operator=(const PrefillCudaTimer&) = delete;

    cudaEvent_t finished_event() const;
    double elapsed_ms() const;

private:
    cudaStream_t stream_ = nullptr;
    cudaEvent_t started_ = nullptr;
    cudaEvent_t finished_ = nullptr;
};

} // namespace mfq::cuda::internal

struct CudaPreparedPrompt {
    std::vector<std::int64_t> token_ids;
    mfq_tensor_backend::Tensor embeddings;
    mfq_tensor_backend::Tensor positions;
    int decode_position_delta = 0;
    std::string cache_key;

    bool transformed() const noexcept {
        return embeddings.defined() || positions.defined() ||
            decode_position_delta != 0;
    }
};

struct ProfileStat {
    double ms = 0.0;
    double wall_ms = 0.0;
    int64_t calls = 0;
};

struct CudaProfiler {
    struct PendingEvent {
        std::string name;
        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
    };

    bool enabled = false;
    bool graph_events = false;
    std::string filter;
    std::unordered_map<std::string, ProfileStat> stats;
    std::vector<std::string> order;
    std::vector<PendingEvent> pending;

    bool selected(const std::string& name) const;
    void reset();

    template <typename Fn>
    auto measure(const std::string & name, Fn && fn) -> decltype(fn()) {
        if (!enabled || !selected(name)) return fn();
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        auto stream = mfq_get_current_cuda_stream().stream();
        if (graph_events) {
            cudaEventRecordWithFlags(start, stream, cudaEventRecordExternal);
        } else {
            cudaEventRecord(start, stream);
        }
        auto t0 = std::chrono::steady_clock::now();
        auto out = fn();
        if (graph_events) {
            cudaEventRecordWithFlags(stop, stream, cudaEventRecordExternal);
        } else {
            cudaEventRecord(stop, stream);
        }
        auto t1 = std::chrono::steady_clock::now();
        auto it = stats.find(name);
        if (it == stats.end()) {
            order.push_back(name);
            it = stats.emplace(name, ProfileStat{}).first;
        }
        it->second.wall_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        it->second.calls += 1;
        pending.push_back(PendingEvent{name, start, stop});
        return out;
    }

    void report(const std::string& title);
};

enum class KlMmqMode {
    Default,
    Nint8One,
    Fp16,
};

enum class TensorParallelAxis {
    Mirrored,
    Output,
    Input,
};

struct ParallelConfig {
    std::vector<int> devices;
    std::vector<double> split;
    bool allow_duplicate_devices = false;

    bool enabled() const {
        return devices.size() > 1;
    }

    int primary_device() const {
        return devices.empty() ? 0 : devices.front();
    }
};

struct ModelParallelCollectiveRuntime {
    using Stream = decltype(mfq_get_stream_from_pool(false));

    std::vector<int> devices;
    std::vector<Stream> streams;
    std::vector<cudaEvent_t> ready;
    std::vector<cudaEvent_t> completed;
    std::vector<mfq_tensor_backend::Tensor> reduction_buffers;
#ifdef MFQ_HAVE_NCCL
    std::vector<ncclComm_t> communicators;
#endif
    bool collectives_enabled = false;

    ~ModelParallelCollectiveRuntime();
    void reset() noexcept;
    void configure(
        const std::vector<int>& requested_devices,
        bool allow_duplicate_devices);
};

struct LayerPlacementConfig {
    std::vector<int> devices;
    std::vector<double> split;
    std::vector<int> layer_devices;
    int load_device = -1;

    bool enabled() const {
        return devices.size() > 1;
    }

    int primary_device() const {
        return devices.empty() ? 0 : devices.front();
    }

    void prepare(int64_t layers);
    int device_for_layer(int64_t layer) const;
};

struct CudaExecutionConfig {
    std::string profile_filter;
    std::string moe_route_stats_path;
    bool moe_route_output_energy = false;
    bool report_cuda_memory = false;
    bool model_parallel_cuda_graph = true;
    bool tensor_parallel_cuda_graph = true;
    bool expert_parallel_cuda_graph = true;
    bool tensor_parallel_grouped_projections = true;
    bool tensor_parallel_shared_linear_attention_input = true;
    bool tensor_parallel_mirror_linear_attention_scalars = true;
    bool tensor_parallel_mirror_qwen35_attention_kv = true;
    bool model_parallel_reduce_to_primary = true;
    bool model_parallel_fp16_reduce = true;
    bool model_parallel_peer_first_launch = true;
    bool decode_branch_parallel = true;
    bool tensor_parallel_shard_native_float = false;
    bool nvq_fusion = true;
    bool nvq2_exec = true;
    bool nvq_extended_group_exec = false;
    bool moe_nvq_heterogeneous = true;
    bool moe_nvq_heterogeneous_decode = true;
    bool moe_prefill_mma = true;
    int moe_prefill_mma_min_tokens = 9;
    bool moe_small_heterogeneous = true;
    bool moe_delayed_route_readback = true;
    bool moe_projection_bundle_prefetch = true;
    bool split_moe_activation_reuse = true;
    bool moe_swiglu_quant_fusion = true;
    bool moe_reduce_gate_fusion = true;
    bool ffn_geglu_fusion = true;
    bool ffn_swiglu_fusion = true;
    bool important_neuron_branch_parallel = true;
    bool gemma4_fused_norms = true;
    bool dsv4_groupwise_output_a = true;
    bool diagnostic_nint_group = true;
    bool diagnostic_fp32_residual = false;
    bool diagnostic_in_f32_down = false;
    bool kv_cache_write_aten = false;
    bool gdn_transposed_state = true;
    bool linear_conv_prefill_fused = true;
    bool minicpm_fused_bf16_rope = true;
    bool minicpm_fused_qk_norm_rope_kv = true;
    bool minicpm_fused_bf16_rmsnorm = true;
    bool minicpm_fused_rope_kv = true;
    bool minicpm_bf16_flash128 = true;
    bool minicpm_flash128_specialized_casts = true;
    bool minicpm_bf16_gqa_decode = true;
    bool minicpm_bf16_swiglu_fusion = true;
    bool minicpm_bf16_residual_acc = true;
    bool mma_attention = true;
    bool mma_attention_decode = true;
    bool attention_decode_aten = false;
    bool attention_decode_split_k = true;
    std::size_t deepseek_v41_engram_cache_rows = 16'384;
    bool moe_mapped_gather = false;
    int moe_mapped_copy_blocks = 64;
    int moe_ssd_io_workers = 8;
    bool moe_ssd_overlap = true;
    bool moe_ssd_ranges = true;
};

CudaExecutionConfig load_cuda_execution_config();

struct MoeRouteLayerStats {
    mfq_tensor_backend::Tensor counts;
    mfq_tensor_backend::Tensor weight_sum;
    mfq_tensor_backend::Tensor weight_sq_sum;
    mfq_tensor_backend::Tensor output_energy;
    mfq_tensor_backend::Tensor weighted_output_energy;
};

struct CudaExecutionContext {
    CudaExecutionContext();

    CudaProfiler profiler;
    CudaExecutionConfig config;
    bool force_moe_pool_path = false;
    bool force_moe_unfused_reduce = false;
    bool force_moe_materialized_swiglu = false;
    bool force_moe_prefill_mma_off = false;
    KlMmqMode kl_mmq_mode = KlMmqMode::Default;
    int64_t kl_mmq_activation_quantize_calls = 0;
    int64_t kl_mmq_dense_calls = 0;
    int64_t kl_mmq_moe_calls = 0;
    int64_t kl_mmq_fallback_calls = 0;
    int64_t kl_kv_cache_capacity = 0;
    std::unordered_set<int> dsv4_cpu_offload_layers;
    int64_t dsv4_cpu_offload_host_bytes = 0;
    int n_gpu_layers = -1;
    int dense_cpu_layer_count = 0;
    bool loading_cpu_layer = false;
    bool drop_file_cache = false;
    bool decode_graph_serial_branches = false;
    bool decode_graph_tp_projection_major = false;
    bool continuous_batch_cache_serial = false;
    int moe_cache_registration_min_slots = 8;
    int gemma_trace_layer = -1;
    std::vector<std::pair<std::string, mfq_tensor_backend::Tensor>>*
        gemma_stage_trace = nullptr;
    ParallelConfig tensor_parallel;
    ParallelConfig expert_parallel;
    ModelParallelCollectiveRuntime model_parallel_collectives;
    LayerPlacementConfig layer_placement;
    std::shared_ptr<MoeExpertCache> moe_expert_cache;
    std::unordered_map<int, MoeRouteLayerStats> moe_route_stats;

    mfq_tensor_backend::Tensor kl_mmq_prepare_activation(mfq_tensor_backend::Tensor x);
    void reset() noexcept;
};

void mfq_release_host_allocator_cache() noexcept;
bool model_parallel_enabled(const CudaExecutionContext& execution);
const ParallelConfig& model_parallel_config(
    const CudaExecutionContext& execution);
const ParallelConfig& moe_parallel_config(
    const CudaExecutionContext& execution);
int model_parallel_primary_device(const CudaExecutionContext& execution);
bool model_parallel_cuda_graph_enabled(const CudaExecutionContext& execution);
size_t model_parallel_launch_index(
    const CudaExecutionConfig& config,
    size_t launch_position,
    size_t shard_count);
int active_weight_load_device(const CudaExecutionContext& execution);
const char* kl_mmq_mode_name(KlMmqMode mode);
void record_moe_route_stats(
    CudaExecutionContext& execution,
    int layer,
    const mfq_tensor_backend::Tensor& ids,
    const mfq_tensor_backend::Tensor& weights,
    const mfq_tensor_backend::Tensor& output,
    int n_experts);
void write_moe_route_stats(
    const CudaExecutionConfig& config,
    const std::unordered_map<int, MoeRouteLayerStats>& stats);
void trace_gemma_stage(
    CudaExecutionContext& execution,
    int layer,
    const char* name,
    const mfq_tensor_backend::Tensor& value);
void report_cuda_memory(
    const CudaExecutionConfig& config,
    const char* stage);

struct KlMmqScope {
    CudaExecutionContext& execution;
    KlMmqMode previous_mode;
    int64_t previous_activation_quantize_calls;
    int64_t previous_dense_calls;
    int64_t previous_moe_calls;
    int64_t previous_fallback_calls;

    KlMmqScope(KlMmqMode mode, CudaExecutionContext& context);
    ~KlMmqScope();
};

struct KlKvCacheCapacityScope {
    CudaExecutionContext& execution;
    int64_t previous_capacity;

    KlKvCacheCapacityScope(
        int64_t capacity, CudaExecutionContext& context);
    ~KlKvCacheCapacityScope();
};
