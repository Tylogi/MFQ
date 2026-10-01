#include "cuda_execution.h"

#include "mfq_cuda_ops.h"

#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

#if defined(__GLIBC__)
#include <malloc.h>
#endif

namespace {

std::optional<std::string> environment(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') return std::nullopt;
    return value;
}

bool environment_flag(const char* name, bool fallback) {
    const auto value = environment(name);
    if (!value) return fallback;
    int parsed = 0;
    const auto [end, error] = std::from_chars(
        value->data(), value->data() + value->size(), parsed);
    if (error != std::errc{} || end != value->data() + value->size()) {
        throw std::runtime_error(std::string("invalid ") + name);
    }
    return parsed != 0;
}

int environment_int(
        const char* name, int fallback, int minimum, int maximum) {
    const auto value = environment(name);
    if (!value) return fallback;
    int parsed = 0;
    const auto [end, error] = std::from_chars(
        value->data(), value->data() + value->size(), parsed);
    if (error != std::errc{} || end != value->data() + value->size() ||
            parsed < minimum || parsed > maximum) {
        throw std::runtime_error(std::string("invalid ") + name);
    }
    return parsed;
}

std::size_t environment_size(
        const char* name, std::size_t fallback) {
    const auto value = environment(name);
    if (!value) return fallback;
    std::size_t parsed = 0;
    const auto [end, error] = std::from_chars(
        value->data(), value->data() + value->size(), parsed);
    if (error != std::errc{} || end != value->data() + value->size()) {
        throw std::runtime_error(std::string("invalid ") + name);
    }
    return parsed;
}

bool enabled_unless_disabled(const char* name) {
    return !environment_flag(name, false);
}

} // namespace

CudaExecutionConfig load_cuda_execution_config() {
    CudaExecutionConfig result;
    result.profile_filter = environment("MFQ_PROFILE_CUDA_FILTER").value_or("");
    result.moe_route_stats_path = environment("MFQ_MOE_ROUTE_STATS").value_or("");
    result.moe_route_output_energy = environment_flag(
        "MFQ_MOE_ROUTE_OUTPUT_ENERGY", false);
    result.report_cuda_memory = environment_flag("MFQ_REPORT_CUDA_MEMORY", false);

    const auto parallel_graph = environment("MFQ_MODEL_PARALLEL_CUDA_GRAPH");
    if (parallel_graph) {
        result.model_parallel_cuda_graph = environment_flag(
            "MFQ_MODEL_PARALLEL_CUDA_GRAPH", true);
        result.tensor_parallel_cuda_graph = result.model_parallel_cuda_graph;
        result.expert_parallel_cuda_graph = result.model_parallel_cuda_graph;
    } else {
        result.tensor_parallel_cuda_graph = environment_flag(
            "MFQ_TP_CUDA_GRAPH", true);
        result.expert_parallel_cuda_graph = environment_flag(
            "MFQ_EP_CUDA_GRAPH", true);
    }
    result.tensor_parallel_grouped_projections = environment_flag(
        "MFQ_TP_GROUPED_PROJECTIONS", true);
    result.tensor_parallel_shared_linear_attention_input = environment_flag(
        "MFQ_TP_SHARED_LINEAR_ATTENTION_INPUT", true);
    result.tensor_parallel_mirror_linear_attention_scalars = environment_flag(
        "MFQ_TP_MIRROR_LINEAR_ATTENTION_SCALARS", true);
    result.tensor_parallel_mirror_qwen35_attention_kv = environment_flag(
        "MFQ_TP_MIRROR_QWEN35_ATTENTION_KV", true);
    result.model_parallel_reduce_to_primary = environment(
        "MFQ_MODEL_PARALLEL_REDUCE_TO_PRIMARY")
        ? environment_flag("MFQ_MODEL_PARALLEL_REDUCE_TO_PRIMARY", true)
        : environment_flag("MFQ_TP_REDUCE_TO_PRIMARY", true);
    result.model_parallel_fp16_reduce = environment(
        "MFQ_MODEL_PARALLEL_FP16_REDUCE")
        ? environment_flag("MFQ_MODEL_PARALLEL_FP16_REDUCE", true)
        : environment_flag("MFQ_TP_FP16_REDUCE", true);
    result.model_parallel_peer_first_launch = environment(
        "MFQ_MODEL_PARALLEL_PEER_FIRST_LAUNCH")
        ? environment_flag("MFQ_MODEL_PARALLEL_PEER_FIRST_LAUNCH", true)
        : environment_flag("MFQ_TP_PEER_FIRST_LAUNCH", true);

    result.decode_branch_parallel = enabled_unless_disabled(
        "MFQ_DISABLE_DECODE_BRANCH_PARALLEL");
    result.tensor_parallel_shard_native_float = environment_flag(
        "MFQ_TP_SHARD_NATIVE_FLOAT", false);
    result.nvq_fusion = environment("MFQ_DISABLE_NVQ_FUSION")
        ? enabled_unless_disabled("MFQ_DISABLE_NVQ_FUSION")
        : enabled_unless_disabled("MFQ_DISABLE_NIQ_FUSION");
    result.nvq2_exec = environment("MFQ_DISABLE_NVQ2_EXEC")
        ? enabled_unless_disabled("MFQ_DISABLE_NVQ2_EXEC")
        : enabled_unless_disabled("MFQ_DISABLE_NIQ2_EXEC");
    result.nvq_extended_group_exec = environment_flag(
        "MFQ_NVQ_EXTENDED_GROUP_EXEC", false);
    result.moe_nvq_heterogeneous = enabled_unless_disabled(
        "MFQ_DISABLE_MOE_NVQ_HETERO");
    result.moe_nvq_heterogeneous_decode =
        enabled_unless_disabled("MFQ_DISABLE_MOE_NVQ_HETERO_DECODE") &&
        !environment_flag("MFQ_NVQ_MOE_EXACT_REDUCTION", false) &&
        !environment("MFQ_NVQ_MOE_ROWS_PER_BLOCK") &&
        !environment("MFQ_NVQ_MOE_WARPS") &&
        !environment("MFQ_NVQ_MOE_SHARE_GROUP_STATE");
    result.moe_prefill_mma = enabled_unless_disabled(
        "MFQ_DISABLE_MOE_PREFILL_MMA");
    result.moe_prefill_mma_min_tokens = environment_int(
        "MFQ_MOE_PREFILL_MMA_MIN_TOKENS", 9, 9,
        std::numeric_limits<int>::max());
    result.moe_small_heterogeneous = enabled_unless_disabled(
        "MFQ_DISABLE_MOE_SMALL_HETERO");
    result.moe_delayed_route_readback = environment_flag(
        "MFQ_MOE_DELAYED_ROUTE_READBACK", true);
    result.moe_projection_bundle_prefetch = enabled_unless_disabled(
        "MFQ_DISABLE_MOE_PROJECTION_BUNDLE_PREFETCH");
    result.split_moe_activation_reuse = enabled_unless_disabled(
        "MFQ_DISABLE_SPLIT_MOE_ACTIVATION_REUSE");
    result.moe_swiglu_quant_fusion = enabled_unless_disabled(
        "MFQ_DISABLE_MOE_SWIGLU_QUANT_FUSION");
    result.moe_reduce_gate_fusion = enabled_unless_disabled(
        "MFQ_DISABLE_MOE_REDUCE_GATE_FUSION");
    result.ffn_geglu_fusion = enabled_unless_disabled(
        "MFQ_DISABLE_FFN_GEGLU_FUSION");
    result.ffn_swiglu_fusion = enabled_unless_disabled(
        "MFQ_DISABLE_FFN_SWIGLU_FUSION");
    result.important_neuron_branch_parallel = enabled_unless_disabled(
        "MFQ_DISABLE_IN_BRANCH_PARALLEL");
    result.gemma4_fused_norms = environment_flag(
        "MFQ_GEMMA4_FUSED_NORMS", true);
    result.dsv4_groupwise_output_a = environment_flag(
        "MFQ_DSV4_GROUPWISE_OUTPUT_A", true);
    result.diagnostic_nint_group = enabled_unless_disabled(
        "MFQ_DIAGNOSTIC_DISABLE_NINT_GROUP");
    result.diagnostic_fp32_residual = environment_flag(
        "MFQ_DIAGNOSTIC_FP32_RESIDUAL", false);
    result.diagnostic_in_f32_down = environment_flag(
        "MFQ_DIAGNOSTIC_IN_F32_DOWN", false);
    result.kv_cache_write_aten = environment_flag(
        "MFQ_KV_CACHE_WRITE_ATEN", false);
    result.gdn_transposed_state = environment_flag(
        "MFQ_GDN_TRANSPOSED_STATE", true);
    result.linear_conv_prefill_fused = environment_flag(
        "MFQ_LINEAR_CONV_PREFILL_FUSED", true);
    result.minicpm_fused_bf16_rope = environment_flag(
        "MFQ_MINICPM_FUSED_BF16_ROPE", true);
    result.minicpm_fused_qk_norm_rope_kv = environment_flag(
        "MFQ_MINICPM_FUSED_QK_NORM_ROPE_KV", true);
    result.minicpm_fused_bf16_rmsnorm = environment_flag(
        "MFQ_MINICPM_FUSED_BF16_RMSNORM", true);
    result.minicpm_fused_rope_kv = environment_flag(
        "MFQ_MINICPM_FUSED_ROPE_KV", true);
    result.minicpm_bf16_flash128 = enabled_unless_disabled(
        "MFQ_DISABLE_MINICPM_BF16_FLASH128");
    result.minicpm_flash128_specialized_casts = enabled_unless_disabled(
        "MFQ_DISABLE_MINICPM_FLASH128_SPECIALIZED_CASTS");
    result.minicpm_bf16_gqa_decode = environment_flag(
        "MFQ_MINICPM_BF16_GQA_DECODE", true);
    result.minicpm_bf16_swiglu_fusion = enabled_unless_disabled(
        "MFQ_DISABLE_MINICPM_BF16_SWIGLU_FUSION");
    result.minicpm_bf16_residual_acc = enabled_unless_disabled(
        "MFQ_DISABLE_MINICPM_BF16_RESIDUAL_ACC");
    result.mma_attention = environment_flag("MFQ_MMA_ATTENTION", true);
    result.mma_attention_decode = environment_flag(
        "MFQ_MMA_ATTENTION_DECODE", true);
    result.attention_decode_aten = environment_flag(
        "MFQ_ATTENTION_DECODE_ATEN", false);
    result.attention_decode_split_k = environment_flag(
        "MFQ_ATTENTION_DECODE_SPLITK", true);
    result.deepseek_v41_engram_cache_rows = environment_size(
        "MFQ_DEEPSEEK_V41_ENGRAM_CACHE_ROWS", 16'384);
    result.moe_mapped_gather = environment_flag("MFQ_MOE_MAPPED_GATHER", false);
    result.moe_mapped_copy_blocks = environment_int(
        "MFQ_MOE_MAPPED_COPY_BLOCKS", 64, 4, 128);
    result.moe_ssd_io_workers = environment_int(
        "MFQ_MOE_SSD_IO_WORKERS", 8, 1, 64);
    result.moe_ssd_overlap = enabled_unless_disabled(
        "MFQ_DISABLE_MOE_SSD_OVERLAP");
    result.moe_ssd_ranges = enabled_unless_disabled(
        "MFQ_DISABLE_MOE_SSD_RANGES");
    return result;
}

void mfq_release_host_allocator_cache() noexcept {
#if defined(__GLIBC__)
    (void)malloc_trim(0);
#endif
}

using mfq_tensor_backend::indexing::Slice;

namespace mfq::cuda::internal {

PrefillCudaTimer::PrefillCudaTimer()
    : stream_(mfq_get_current_cuda_stream()) {
    MFQ_CUDA_CHECK(cudaEventCreate(&started_));
    try {
        MFQ_CUDA_CHECK(cudaEventCreate(&finished_));
        MFQ_CUDA_CHECK(cudaEventRecord(started_, stream_));
    } catch (...) {
        if (finished_ != nullptr) cudaEventDestroy(finished_);
        cudaEventDestroy(started_);
        finished_ = nullptr;
        started_ = nullptr;
        throw;
    }
}

PrefillCudaTimer::~PrefillCudaTimer() {
    if (finished_ != nullptr) cudaEventDestroy(finished_);
    if (started_ != nullptr) cudaEventDestroy(started_);
}

cudaEvent_t PrefillCudaTimer::finished_event() const {
    return finished_;
}

double PrefillCudaTimer::elapsed_ms() const {
    MFQ_CUDA_CHECK(cudaEventSynchronize(finished_));
    float elapsed = 0.0f;
    MFQ_CUDA_CHECK(cudaEventElapsedTime(&elapsed, started_, finished_));
    return static_cast<double>(elapsed);
}

} // namespace mfq::cuda::internal

bool CudaProfiler::selected(const std::string& name) const {
    if (filter.empty()) return true;

    const std::string_view filter_value(filter);
    size_t begin = 0;
    while (begin <= filter_value.size()) {
        const size_t end = filter_value.find(',', begin);
        const size_t count = end == std::string_view::npos
            ? filter_value.size() - begin
            : end - begin;
        if (filter_value.substr(begin, count) == name) return true;
        if (end == std::string_view::npos) break;
        begin = end + 1;
    }
    return false;
}

void CudaProfiler::reset() {
    for (auto& event : pending) {
        cudaEventDestroy(event.start);
        cudaEventDestroy(event.stop);
    }
    pending.clear();
    stats.clear();
    order.clear();
}

void CudaProfiler::report(const std::string& title) {
    if (!enabled) return;
    if (!pending.empty()) cudaEventSynchronize(pending.back().stop);
    for (auto& event : pending) {
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, event.start, event.stop);
        stats.at(event.name).ms += static_cast<double>(ms);
        cudaEventDestroy(event.start);
        cudaEventDestroy(event.stop);
    }
    pending.clear();
    std::cerr << "profile " << title << '\n';
    for (const auto& name : order) {
        const auto& stat = stats.at(name);
        std::cerr << "profile_item"
                  << " name=" << name
                  << " calls=" << stat.calls
                  << " cuda_ms=" << stat.ms
                  << " cuda_avg_ms="
                  << (stat.calls ? stat.ms / static_cast<double>(stat.calls) : 0.0)
                  << " wall_ms=" << stat.wall_ms
                  << " wall_avg_ms="
                  << (stat.calls ? stat.wall_ms / static_cast<double>(stat.calls) : 0.0)
                  << '\n';
    }
}

ModelParallelCollectiveRuntime::~ModelParallelCollectiveRuntime() {
    reset();
}

void ModelParallelCollectiveRuntime::reset() noexcept {
    // Release CUDA-owned state while every device context is still alive.
    // Static destruction is too late: the CUDA allocator may already be
    // shutting down when tensors on secondary model-parallel devices are
    // destroyed.
    for (int device : devices) {
        (void)cudaSetDevice(device);
        (void)cudaDeviceSynchronize();
    }
    reduction_buffers.clear();
#ifdef MFQ_HAVE_NCCL
    for (auto communicator : communicators) {
        if (communicator != nullptr) (void)ncclCommDestroy(communicator);
    }
    communicators.clear();
#endif
    for (size_t index = 0; index < devices.size(); ++index) {
        (void)cudaSetDevice(devices[index]);
        if (index < ready.size() && ready[index] != nullptr) {
            (void)cudaEventDestroy(ready[index]);
        }
        if (index < completed.size() && completed[index] != nullptr) {
            (void)cudaEventDestroy(completed[index]);
        }
    }
    devices.clear();
    streams.clear();
    ready.clear();
    completed.clear();
    collectives_enabled = false;
}

void ModelParallelCollectiveRuntime::configure(
        const std::vector<int>& requested_devices,
        bool allow_duplicate_devices) {
    reset();
    if (requested_devices.size() < 2 || allow_duplicate_devices) return;

    devices = requested_devices;
    streams.reserve(devices.size());
    ready.resize(devices.size(), nullptr);
    completed.resize(devices.size(), nullptr);
    reduction_buffers.resize(devices.size());
    for (size_t index = 0; index < devices.size(); ++index) {
        MfqCudaGuard guard(devices[index]);
        streams.push_back(mfq_get_stream_from_pool(false, devices[index]));
        MFQ_CUDA_CHECK(cudaEventCreateWithFlags(
            &ready[index], cudaEventDisableTiming));
        MFQ_CUDA_CHECK(cudaEventCreateWithFlags(
            &completed[index], cudaEventDisableTiming));
    }
#ifdef MFQ_HAVE_NCCL
    communicators.resize(devices.size(), nullptr);
    MFQ_NCCL_CHECK(ncclCommInitAll(
        communicators.data(),
        static_cast<int>(devices.size()),
        devices.data()));
    // Warm both directions before graph capture triggers NCCL's lazy setup.
    std::vector<mfq_tensor_backend::Tensor> p2p_warmup_buffers;
    p2p_warmup_buffers.reserve(devices.size());
    for (const int device : devices) {
        MfqCudaGuard guard(device);
        p2p_warmup_buffers.push_back(mfq_tensor_backend::empty(
            {1}, mfq_tensor_backend::TensorOptions()
                .device(mfq_tensor_backend::Device(
                    mfq_tensor_backend::kCUDA, device))
                .dtype(mfq_tensor_backend::kUInt8)));
    }
    for (size_t peer = 1; peer < devices.size(); ++peer) {
        MFQ_NCCL_CHECK(ncclGroupStart());
        MFQ_NCCL_CHECK(ncclSend(
            p2p_warmup_buffers[0].data_ptr(), 1, ncclUint8,
            static_cast<int>(peer), communicators[0], streams[0].stream()));
        MFQ_NCCL_CHECK(ncclRecv(
            p2p_warmup_buffers[peer].data_ptr(), 1, ncclUint8,
            0, communicators[peer], streams[peer].stream()));
        MFQ_NCCL_CHECK(ncclSend(
            p2p_warmup_buffers[peer].data_ptr(), 1, ncclUint8,
            0, communicators[peer], streams[peer].stream()));
        MFQ_NCCL_CHECK(ncclRecv(
            p2p_warmup_buffers[0].data_ptr(), 1, ncclUint8,
            static_cast<int>(peer), communicators[0], streams[0].stream()));
        MFQ_NCCL_CHECK(ncclGroupEnd());
    }
    for (size_t index = 0; index < devices.size(); ++index) {
        MfqCudaGuard guard(devices[index]);
        MFQ_CUDA_CHECK(cudaStreamSynchronize(streams[index].stream()));
    }
    collectives_enabled = true;
#endif
}

void LayerPlacementConfig::prepare(int64_t layers) {
    layer_devices.assign(static_cast<size_t>(layers), primary_device());
    if (!enabled()) return;
    const auto slices = mfq::plan_tensor_parallel_slices(
        layers, 1, devices, split);
    for (const auto& slice : slices) {
        for (int64_t layer = slice.begin; layer < slice.end; ++layer) {
            layer_devices.at(static_cast<size_t>(layer)) = slice.device;
        }
    }
}

int LayerPlacementConfig::device_for_layer(int64_t layer) const {
    if (layer < 0 || layer >= static_cast<int64_t>(layer_devices.size())) {
        throw std::runtime_error("layer-placement index is outside the model");
    }
    return layer_devices.at(static_cast<size_t>(layer));
}

KlMmqScope::KlMmqScope(KlMmqMode mode, CudaExecutionContext& context)
    : execution(context),
      previous_mode(context.kl_mmq_mode),
      previous_activation_quantize_calls(
          context.kl_mmq_activation_quantize_calls),
      previous_dense_calls(context.kl_mmq_dense_calls),
      previous_moe_calls(context.kl_mmq_moe_calls),
      previous_fallback_calls(context.kl_mmq_fallback_calls) {
    execution.kl_mmq_mode = mode;
    execution.kl_mmq_activation_quantize_calls = 0;
    execution.kl_mmq_dense_calls = 0;
    execution.kl_mmq_moe_calls = 0;
    execution.kl_mmq_fallback_calls = 0;
}

KlMmqScope::~KlMmqScope() {
    execution.kl_mmq_mode = previous_mode;
    execution.kl_mmq_activation_quantize_calls =
        previous_activation_quantize_calls;
    execution.kl_mmq_dense_calls = previous_dense_calls;
    execution.kl_mmq_moe_calls = previous_moe_calls;
    execution.kl_mmq_fallback_calls = previous_fallback_calls;
}

KlKvCacheCapacityScope::KlKvCacheCapacityScope(
        int64_t capacity, CudaExecutionContext& context)
    : execution(context), previous_capacity(context.kl_kv_cache_capacity) {
    execution.kl_kv_cache_capacity = capacity;
}

KlKvCacheCapacityScope::~KlKvCacheCapacityScope() {
    execution.kl_kv_cache_capacity = previous_capacity;
}

CudaExecutionContext::CudaExecutionContext()
    : config(load_cuda_execution_config()) {
    profiler.filter = config.profile_filter;
}

void CudaExecutionContext::reset() noexcept {
    profiler.reset();
    moe_expert_cache.reset();
    model_parallel_collectives.reset();
    tensor_parallel = {};
    expert_parallel = {};
    layer_placement = {};
    dsv4_cpu_offload_layers.clear();
    dsv4_cpu_offload_host_bytes = 0;
    n_gpu_layers = -1;
    dense_cpu_layer_count = 0;
    loading_cpu_layer = false;
    drop_file_cache = false;
    decode_graph_serial_branches = false;
    decode_graph_tp_projection_major = false;
    continuous_batch_cache_serial = false;
    moe_route_stats.clear();
    kl_mmq_mode = KlMmqMode::Default;
    kl_mmq_activation_quantize_calls = 0;
    kl_mmq_dense_calls = 0;
    kl_mmq_moe_calls = 0;
    kl_mmq_fallback_calls = 0;
    kl_kv_cache_capacity = 0;
    force_moe_pool_path = false;
    force_moe_unfused_reduce = false;
    force_moe_materialized_swiglu = false;
    force_moe_prefill_mma_off = false;
    moe_cache_registration_min_slots = 8;
    gemma_trace_layer = -1;
    gemma_stage_trace = nullptr;
}

bool model_parallel_enabled(const CudaExecutionContext& execution) {
    return execution.tensor_parallel.enabled() ||
        execution.expert_parallel.enabled();
}

const ParallelConfig & model_parallel_config(
        const CudaExecutionContext& execution) {
    return execution.tensor_parallel.enabled()
        ? execution.tensor_parallel
        : execution.expert_parallel;
}

const ParallelConfig & moe_parallel_config(
        const CudaExecutionContext& execution) {
    return execution.expert_parallel.enabled()
        ? execution.expert_parallel
        : execution.tensor_parallel;
}

int model_parallel_primary_device(const CudaExecutionContext& execution) {
    if (!execution.tensor_parallel.devices.empty()) {
        return execution.tensor_parallel.primary_device();
    }
    if (!execution.expert_parallel.devices.empty()) {
        return execution.expert_parallel.primary_device();
    }
    return 0;
}



bool model_parallel_cuda_graph_enabled(
        const CudaExecutionContext& execution) {
    if (!execution.tensor_parallel.enabled() &&
            !execution.expert_parallel.enabled()) {
        return true;
    }
    const bool enabled = execution.expert_parallel.enabled() &&
            !execution.tensor_parallel.enabled()
        ? execution.config.expert_parallel_cuda_graph
        : execution.config.tensor_parallel_cuda_graph;
    return execution.model_parallel_collectives.collectives_enabled && enabled;
}

size_t model_parallel_launch_index(
        const CudaExecutionConfig& config,
        size_t launch_position,
        size_t shard_count) {
    return config.model_parallel_peer_first_launch
        ? mfq::peer_first_parallel_launch_index(
            launch_position, shard_count)
        : launch_position;
}



int active_weight_load_device(const CudaExecutionContext& execution) {
    const auto& placement = execution.layer_placement;
    return placement.load_device >= 0
        ? placement.load_device
        : model_parallel_primary_device(execution);
}

const char * kl_mmq_mode_name(KlMmqMode mode) {
    switch (mode) {
        case KlMmqMode::Default: return "default";
        case KlMmqMode::Nint8One: return "nint8_1";
        case KlMmqMode::Fp16: return "fp16";
    }
    return "invalid";
}





mfq_tensor_backend::Tensor CudaExecutionContext::kl_mmq_prepare_activation(
        mfq_tensor_backend::Tensor x) {
    if (kl_mmq_mode != KlMmqMode::Nint8One) return x;
    const auto original_shape = x.sizes().vec();
    auto flat = x.reshape({-1, x.size(-1)}).contiguous();
    auto quantized = nint8_one_quantize_reconstruct_cuda(flat);
    ++kl_mmq_activation_quantize_calls;
    return quantized.at(3).reshape(original_shape).contiguous();
}



void record_moe_route_stats(
        CudaExecutionContext& execution,
        int layer,
        const mfq_tensor_backend::Tensor & ids,
        const mfq_tensor_backend::Tensor & weights,
        const mfq_tensor_backend::Tensor & output,
        int n_experts) {
    if (layer < 0 || execution.config.moe_route_stats_path.empty()) return;
    auto& stats = execution.moe_route_stats;
    auto found = stats.find(layer);
    if (found == stats.end()) {
        auto options = mfq_tensor_backend::TensorOptions()
            .device(ids.device()).dtype(mfq_tensor_backend::kFloat64);
        MoeRouteLayerStats value{
            mfq_tensor_backend::zeros({n_experts}, options),
            mfq_tensor_backend::zeros({n_experts}, options),
            mfq_tensor_backend::zeros({n_experts}, options),
            mfq_tensor_backend::zeros({n_experts}, options),
            mfq_tensor_backend::zeros({n_experts}, options),
        };
        found = stats.emplace(layer, std::move(value)).first;
    }
    auto flat_ids = ids.reshape({-1}).to(mfq_tensor_backend::kInt64);
    auto flat_weights = weights.reshape({-1}).to(mfq_tensor_backend::kFloat64);
    auto ones = mfq_tensor_backend::ones_like(flat_weights);
    found->second.counts.scatter_add_(0, flat_ids, ones);
    found->second.weight_sum.scatter_add_(0, flat_ids, flat_weights);
    found->second.weight_sq_sum.scatter_add_(
        0, flat_ids, flat_weights.square());
    if (execution.config.moe_route_output_energy) {
        auto energy = output.reshape({flat_ids.numel(), output.size(-1)})
            .to(mfq_tensor_backend::kFloat32).square().sum(1).to(mfq_tensor_backend::kFloat64);
        found->second.output_energy.scatter_add_(0, flat_ids, energy);
        found->second.weighted_output_energy.scatter_add_(
            0, flat_ids, energy * flat_weights.square());
    }
}

void write_moe_route_stats(
        const CudaExecutionConfig& config,
        const std::unordered_map<int, MoeRouteLayerStats>& stats) {
    if (config.moe_route_stats_path.empty() || stats.empty()) return;
    mfq_cuda_synchronize();
    std::filesystem::path path(config.moe_route_stats_path);
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error(
            "cannot write MoE route statistics: " + path.string());
    }
    out << "layer,expert,count,weight_sum,weight_sq_sum,"
           "output_energy,weighted_output_energy\n";
    std::vector<int> layers;
    layers.reserve(stats.size());
    for (const auto & item : stats) layers.push_back(item.first);
    std::sort(layers.begin(), layers.end());
    out << std::setprecision(17);
    for (int layer : layers) {
        const auto & value = stats.at(layer);
        auto counts = value.counts.to(mfq_tensor_backend::kCPU).contiguous();
        auto weight_sum = value.weight_sum.to(mfq_tensor_backend::kCPU).contiguous();
        auto weight_sq_sum = value.weight_sq_sum.to(mfq_tensor_backend::kCPU).contiguous();
        auto output_energy = value.output_energy.to(mfq_tensor_backend::kCPU).contiguous();
        auto weighted_output_energy =
            value.weighted_output_energy.to(mfq_tensor_backend::kCPU).contiguous();
        const auto n_experts = counts.numel();
        const double * count_data = counts.data_ptr<double>();
        const double * weight_data = weight_sum.data_ptr<double>();
        const double * weight_sq_data = weight_sq_sum.data_ptr<double>();
        const double * energy_data = output_energy.data_ptr<double>();
        const double * weighted_energy_data =
            weighted_output_energy.data_ptr<double>();
        for (int64_t expert = 0; expert < n_experts; ++expert) {
            out << layer << ',' << expert << ','
                << count_data[expert] << ','
                << weight_data[expert] << ','
                << weight_sq_data[expert] << ','
                << energy_data[expert] << ','
                << weighted_energy_data[expert] << '\n';
        }
    }
    std::cout << "moe_route_stats_path=" << path.string()
              << " layers=" << layers.size()
              << " output_energy="
              << (config.moe_route_output_energy ? 1 : 0) << "\n";
}

void trace_gemma_stage(
        CudaExecutionContext& execution,
        int layer,
        const char * name,
        const mfq_tensor_backend::Tensor & value) {
    if (execution.gemma_stage_trace != nullptr &&
            layer == execution.gemma_trace_layer) {
        execution.gemma_stage_trace->emplace_back(
            name, value.to(mfq_tensor_backend::kFloat32).contiguous().clone());
    }
}

void report_cuda_memory(
        const CudaExecutionConfig& config,
        const char * stage) {
    if (!config.report_cuda_memory) return;
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    MFQ_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    const auto stats = mfq_cuda_memory_stats(mfq_current_cuda_device());
    constexpr double mib = 1024.0 * 1024.0;
    std::cout << "cuda_memory_stage=" << stage
              << " free_mib=" << free_bytes / mib
              << " used_mib=" << (total_bytes - free_bytes) / mib
              << " total_mib=" << total_bytes / mib
              << " allocated_mib=" << stats.allocated_bytes / mib
              << " active_mib=" << stats.active_bytes / mib
              << " reserved_mib=" << stats.reserved_bytes / mib
              << " inactive_split_mib="
              << stats.inactive_split_bytes / mib
              << " requested_mib=" << stats.requested_bytes / mib
              << " allocations=" << stats.allocations
              << " segments=" << stats.segments
              << " retries=" << stats.retries
              << " ooms=" << stats.ooms << "\n";
}
