#include "cuda_execution.h"

#include "mfq_cuda_ops.h"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#if defined(__GLIBC__)
#include <malloc.h>
#endif

void mfq_set_env(const char * name, const char * value) {
#ifdef _WIN32
    if (_putenv_s(name, value ? value : "") != 0) {
        throw std::runtime_error(std::string("failed to update environment variable ") + name);
    }
#else
    const int status = value && value[0] != '\0'
        ? setenv(name, value, 1)
        : unsetenv(name);
    if (status != 0) {
        throw std::runtime_error(
            std::string("failed to update environment variable ") + name +
            ": " + std::strerror(errno));
    }
#endif
}

void mfq_release_host_allocator_cache() noexcept {
#if defined(__GLIBC__)
    (void)malloc_trim(0);
#endif
}

using mfq_tensor_backend::indexing::Slice;





CudaProfiler g_profiler;

bool g_force_moe_pool_path = false;
bool g_force_moe_unfused_reduce = false;
bool g_force_moe_materialized_swiglu = false;
bool g_force_moe_prefill_mma_off = false;
thread_local bool g_moe_continuous_batch_cache_serial = false;

KlMmqMode g_kl_mmq_mode = KlMmqMode::Default;
int64_t g_kl_mmq_activation_quantize_calls = 0;
int64_t g_kl_mmq_dense_calls = 0;
int64_t g_kl_mmq_moe_calls = 0;
int64_t g_kl_mmq_fallback_calls = 0;
int64_t g_kl_kv_cache_capacity = 0;
std::unordered_set<int> g_dsv4_cpu_offload_layers;
int64_t g_dsv4_cpu_offload_host_bytes = 0;
// Repeating-layer placement: the first layers stay on CPU and
// the last n_gpu_layers stay on CUDA.  -1 keeps the historical all-CUDA path.
int g_n_gpu_layers = -1;
int g_dense_cpu_layer_count = 0;
bool g_loading_cpu_layer = false;
class MoeExpertCache;
class MoeCachedSource;
int g_moe_cache_registration_min_slots = 8;
int g_gemma_trace_layer = -1;
std::vector<std::pair<std::string, mfq_tensor_backend::Tensor>> * g_gemma_stage_trace = nullptr;





ParallelConfig g_tensor_parallel;
ParallelConfig g_expert_parallel;

bool model_parallel_enabled() {
    return g_tensor_parallel.enabled() ||
        g_expert_parallel.enabled();
}

const ParallelConfig & model_parallel_config() {
    return g_tensor_parallel.enabled()
        ? g_tensor_parallel
        : g_expert_parallel;
}

const ParallelConfig & moe_parallel_config() {
    return g_expert_parallel.enabled()
        ? g_expert_parallel
        : g_tensor_parallel;
}

int model_parallel_primary_device() {
    if (!g_tensor_parallel.devices.empty()) {
        return g_tensor_parallel.primary_device();
    }
    if (!g_expert_parallel.devices.empty()) {
        return g_expert_parallel.primary_device();
    }
    return 0;
}



ModelParallelCollectiveRuntime g_model_parallel_collectives;

bool model_parallel_cuda_graph_enabled() {
    if (!model_parallel_enabled()) {
        return true;
    }
    const char * environment = std::getenv(
        "MFQ_MODEL_PARALLEL_CUDA_GRAPH");
    if (environment == nullptr) {
        environment = std::getenv(
            g_expert_parallel.enabled() && !g_tensor_parallel.enabled()
                ? "MFQ_EP_CUDA_GRAPH"
                : "MFQ_TP_CUDA_GRAPH");
    }
    return g_model_parallel_collectives.collectives_enabled &&
           (environment == nullptr || environment[0] != '0');
}

bool tensor_parallel_grouped_projections_enabled() {
    const char * environment = std::getenv(
        "MFQ_TP_GROUPED_PROJECTIONS");
    return environment == nullptr || std::atoi(environment) != 0;
}

bool tensor_parallel_shared_linear_attention_input_enabled() {
    const char * environment = std::getenv(
        "MFQ_TP_SHARED_LINEAR_ATTENTION_INPUT");
    return environment == nullptr || std::atoi(environment) != 0;
}

bool tensor_parallel_mirror_linear_attention_scalars_enabled() {
    const char * environment = std::getenv(
        "MFQ_TP_MIRROR_LINEAR_ATTENTION_SCALARS");
    return environment == nullptr || std::atoi(environment) != 0;
}

bool tensor_parallel_mirror_qwen35_attention_kv_enabled() {
    const char * environment = std::getenv(
        "MFQ_TP_MIRROR_QWEN35_ATTENTION_KV");
    return environment == nullptr || std::atoi(environment) != 0;
}

bool model_parallel_reduce_to_primary_enabled() {
    const char * environment = std::getenv(
        "MFQ_MODEL_PARALLEL_REDUCE_TO_PRIMARY");
    if (environment == nullptr) {
        environment = std::getenv("MFQ_TP_REDUCE_TO_PRIMARY");
    }
    return environment == nullptr || std::atoi(environment) != 0;
}

bool model_parallel_fp16_reduce_enabled() {
    const char * environment = std::getenv(
        "MFQ_MODEL_PARALLEL_FP16_REDUCE");
    if (environment == nullptr) {
        environment = std::getenv("MFQ_TP_FP16_REDUCE");
    }
    return environment == nullptr || std::atoi(environment) != 0;
}

bool model_parallel_peer_first_launch_enabled() {
    static const bool enabled = [] {
        const char * environment = std::getenv(
            "MFQ_MODEL_PARALLEL_PEER_FIRST_LAUNCH");
        if (environment == nullptr) {
            environment = std::getenv(
                "MFQ_TP_PEER_FIRST_LAUNCH");
        }
        return environment == nullptr || std::atoi(environment) != 0;
    }();
    return enabled;
}

size_t model_parallel_launch_index(
        size_t launch_position, size_t shard_count) {
    return model_parallel_peer_first_launch_enabled()
        ? mfq::peer_first_parallel_launch_index(
            launch_position, shard_count)
        : launch_position;
}



LayerPlacementConfig g_layer_placement;

int active_weight_load_device() {
    return g_layer_placement.load_device >= 0
        ? g_layer_placement.load_device
        : model_parallel_primary_device();
}

const char * kl_mmq_mode_name(KlMmqMode mode) {
    switch (mode) {
        case KlMmqMode::Default: return "default";
        case KlMmqMode::Nint8One: return "nint8_1";
        case KlMmqMode::Fp16: return "fp16";
    }
    return "invalid";
}





mfq_tensor_backend::Tensor kl_mmq_prepare_activation(mfq_tensor_backend::Tensor x) {
    if (g_kl_mmq_mode != KlMmqMode::Nint8One) {
        return x;
    }
    const auto original_shape = x.sizes().vec();
    auto flat = x.reshape({-1, x.size(-1)}).contiguous();
    auto quantized = nint8_one_quantize_reconstruct_cuda(flat);
    ++g_kl_mmq_activation_quantize_calls;
    return quantized.at(3).reshape(original_shape).contiguous();
}



static std::unordered_map<int, MoeRouteLayerStats> g_moe_route_stats;

const char * moe_route_stats_path() {
    const char * value = std::getenv("MFQ_MOE_ROUTE_STATS");
    return value != nullptr && value[0] != '\0' ? value : nullptr;
}

bool moe_route_output_energy_enabled() {
    const char * value = std::getenv("MFQ_MOE_ROUTE_OUTPUT_ENERGY");
    return value != nullptr && std::atoi(value) != 0;
}

void record_moe_route_stats(
        int layer,
        const mfq_tensor_backend::Tensor & ids,
        const mfq_tensor_backend::Tensor & weights,
        const mfq_tensor_backend::Tensor & output,
        int n_experts) {
    if (layer < 0 || moe_route_stats_path() == nullptr) return;
    auto found = g_moe_route_stats.find(layer);
    if (found == g_moe_route_stats.end()) {
        auto options = mfq_tensor_backend::TensorOptions()
            .device(ids.device()).dtype(mfq_tensor_backend::kFloat64);
        MoeRouteLayerStats value{
            mfq_tensor_backend::zeros({n_experts}, options),
            mfq_tensor_backend::zeros({n_experts}, options),
            mfq_tensor_backend::zeros({n_experts}, options),
            mfq_tensor_backend::zeros({n_experts}, options),
            mfq_tensor_backend::zeros({n_experts}, options),
        };
        found = g_moe_route_stats.emplace(layer, std::move(value)).first;
    }
    auto flat_ids = ids.reshape({-1}).to(mfq_tensor_backend::kInt64);
    auto flat_weights = weights.reshape({-1}).to(mfq_tensor_backend::kFloat64);
    auto ones = mfq_tensor_backend::ones_like(flat_weights);
    found->second.counts.scatter_add_(0, flat_ids, ones);
    found->second.weight_sum.scatter_add_(0, flat_ids, flat_weights);
    found->second.weight_sq_sum.scatter_add_(
        0, flat_ids, flat_weights.square());
    if (moe_route_output_energy_enabled()) {
        auto energy = output.reshape({flat_ids.numel(), output.size(-1)})
            .to(mfq_tensor_backend::kFloat32).square().sum(1).to(mfq_tensor_backend::kFloat64);
        found->second.output_energy.scatter_add_(0, flat_ids, energy);
        found->second.weighted_output_energy.scatter_add_(
            0, flat_ids, energy * flat_weights.square());
    }
}

void clear_moe_route_stats() {
    g_moe_route_stats.clear();
}

void write_moe_route_stats() {
    const char * path_value = moe_route_stats_path();
    if (path_value == nullptr || g_moe_route_stats.empty()) return;
    mfq_cuda_synchronize();
    std::filesystem::path path(path_value);
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
    layers.reserve(g_moe_route_stats.size());
    for (const auto & item : g_moe_route_stats) layers.push_back(item.first);
    std::sort(layers.begin(), layers.end());
    out << std::setprecision(17);
    for (int layer : layers) {
        const auto & value = g_moe_route_stats.at(layer);
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
              << (moe_route_output_energy_enabled() ? 1 : 0) << "\n";
}

void trace_gemma_stage(int layer, const char * name, const mfq_tensor_backend::Tensor & value) {
    if (g_gemma_stage_trace != nullptr && layer == g_gemma_trace_layer) {
        g_gemma_stage_trace->emplace_back(
            name, value.to(mfq_tensor_backend::kFloat32).contiguous().clone());
    }
}

bool gemma4_fused_norms_enabled() {
    const char * value = std::getenv("MFQ_GEMMA4_FUSED_NORMS");
    return value == nullptr || std::atoi(value) != 0;
}

void report_cuda_memory(const char * stage) {
    const char * enabled = std::getenv("MFQ_REPORT_CUDA_MEMORY");
    if (enabled == nullptr || std::atoi(enabled) == 0) return;
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

bool moe_small_glu_path_enabled(int tokens) {
    static const bool disabled = [] {
        const char * value = std::getenv("MFQ_DISABLE_MOE_SMALL_HETERO");
        return value != nullptr && std::atoi(value) != 0;
    }();
    return tokens == 1 || (tokens <= 4 && !disabled);
}
