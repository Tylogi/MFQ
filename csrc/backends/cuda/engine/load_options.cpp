#include "load_options.h"
#include "cuda_runtime_config.h"
#include "ops/cuda_execution.h"
#include "storage/moe/expert_cache.h"
#include "moe_cache_profile.h"
#include "native/tensor_backend.h"

#include <cuda_runtime_api.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace mfq::cuda::internal {
using mfq::engine::environment_uint64;
using mfq::engine::environment_enabled;
namespace {

std::int32_t graph_minimum(const char* name) {
    const auto value = std::max<std::uint64_t>(
        2, environment_uint64(name, 16));
    if (value > static_cast<std::uint64_t>(
            std::numeric_limits<std::int32_t>::max())) {
        throw std::runtime_error(std::string(name) + " exceeds int32");
    }
    return static_cast<std::int32_t>(value);
}

} // namespace

CudaRuntimeConfig resolve_cuda_runtime_config(
        const CudaEngineOptions& options) {
    CudaRuntimeConfig config;
    static_cast<mfq::engine::RuntimeConfig&>(config) =
        mfq::engine::resolve_runtime_config(options.prefill_chunk_size);
    static_cast<mfq::engine::ContinuousBatchConfig&>(config.continuous_batch) =
        mfq::engine::resolve_batch_config(options.continuous_batching, options.prefill_chunk_size);
    config.decode_graph.enabled = environment_enabled(
        "MFQ_RUNTIME_CUDA_GRAPH", true);
    config.decode_graph.trace = environment_enabled(
        "MFQ_RUNTIME_TRACE_CUDA_GRAPH", false);
    config.decode_graph.minimum_generation_tokens = graph_minimum(
        "MFQ_RUNTIME_CUDA_GRAPH_MIN_TOKENS");

    config.continuous_batch.greedy = environment_enabled(
        "MFQ_CONTINUOUS_BATCH_GREEDY", true);
    config.continuous_batch.cuda_graph =
        config.decode_graph.enabled && environment_enabled(
            "MFQ_CONTINUOUS_BATCH_CUDA_GRAPH", true);
    config.continuous_batch.paged_kv = environment_enabled(
        "MFQ_CONTINUOUS_PAGED_KV", true);
    config.continuous_batch.cuda_graph_minimum_tokens = graph_minimum(
        "MFQ_CONTINUOUS_BATCH_CUDA_GRAPH_MIN_TOKENS");
    return config;
}

static ParallelConfig validated_parallel_config(
        const CudaParallelOptions& options,
        const char* devices_option,
        const char* split_option,
        int available_devices,
        bool allow_duplicate_devices) {
    ParallelConfig config;
    config.devices = options.devices;
    config.split = options.split;
    config.allow_duplicate_devices = allow_duplicate_devices;
    if (config.devices.empty()) {
        if (!config.split.empty()) {
            throw std::runtime_error(
                std::string(split_option) + " requires " + devices_option);
        }
        return config;
    }
    if (config.devices.size() < 2) {
        throw std::runtime_error(
            std::string(devices_option) + " requires at least two devices");
    }

    std::unordered_set<int> unique_devices;
    for (const int device : config.devices) {
        if (device < 0 || device >= available_devices) {
            throw std::runtime_error(
                std::string(devices_option) +
                " CUDA device is unavailable: " +
                std::to_string(device));
        }
        const bool inserted = unique_devices.insert(device).second;
        if (!allow_duplicate_devices && !inserted) {
            throw std::runtime_error(
                std::string(devices_option) +
                " CUDA devices must be unique");
        }
    }
    for (const double weight : config.split) {
        if (!std::isfinite(weight) || weight <= 0.0) {
            throw std::runtime_error(
                std::string(split_option) +
                " values must be finite and positive");
        }
    }
    if (!config.split.empty() &&
            config.split.size() != config.devices.size()) {
        throw std::runtime_error(
            std::string(split_option) + " count must match " +
            devices_option + " devices");
    }
    return config;
}

static void print_parallel_config(
        const char * label,
        const ParallelConfig & config) {
    if (!config.enabled()) return;
    std::cerr << label << " devices=";
    for (size_t index = 0; index < config.devices.size(); ++index) {
        if (index) std::cerr << ',';
        std::cerr << config.devices[index];
    }
    if (!config.split.empty()) {
        std::cerr << " split=";
        for (size_t index = 0; index < config.split.size(); ++index) {
            if (index) std::cerr << ',';
            std::cerr << config.split[index];
        }
    }
    std::cerr << '\n';
}

static void configure_model_parallel(
        CudaExecutionContext& execution,
        const CudaParallelOptions& tensor_options,
        const CudaParallelOptions& expert_options,
        bool allow_duplicate_devices = false) {
    execution.model_parallel_collectives.reset();
    execution.tensor_parallel = {};
    execution.expert_parallel = {};

    int available = 0;
    MFQ_CUDA_CHECK(cudaGetDeviceCount(&available));
    execution.tensor_parallel = validated_parallel_config(
        tensor_options, "--tensor-parallel", "--tensor-split",
        available, allow_duplicate_devices);
    execution.expert_parallel = validated_parallel_config(
        expert_options, "--expert-parallel", "--expert-split",
        available, allow_duplicate_devices);

    if (execution.tensor_parallel.enabled() &&
            execution.expert_parallel.enabled() &&
            execution.tensor_parallel.devices !=
                execution.expert_parallel.devices) {
        throw std::runtime_error(
            "combined tensor and expert parallelism requires the same "
            "ordered CUDA device group");
    }
    if (!execution.tensor_parallel.enabled() &&
            !execution.expert_parallel.enabled()) {
        MFQ_CUDA_CHECK(cudaSetDevice(0));
        return;
    }

    const auto& config = execution.tensor_parallel.enabled()
        ? execution.tensor_parallel : execution.expert_parallel;
    std::unordered_set<int> unique_devices(
        config.devices.begin(), config.devices.end());
    for (const int source : unique_devices) {
        for (const int destination : unique_devices) {
            if (source == destination) continue;
            int can_access = 0;
            MFQ_CUDA_CHECK(cudaDeviceCanAccessPeer(
                &can_access, source, destination));
            if (!can_access) continue;
            MFQ_CUDA_CHECK(cudaSetDevice(source));
            const cudaError_t status = cudaDeviceEnablePeerAccess(
                destination, 0);
            if (status != cudaSuccess &&
                    status != cudaErrorPeerAccessAlreadyEnabled) {
                throw std::runtime_error(
                    "failed to enable model-parallel peer access from CUDA " +
                    std::to_string(source) + " to CUDA " +
                    std::to_string(destination) + ": " +
                    cudaGetErrorString(status));
            }
            if (status == cudaErrorPeerAccessAlreadyEnabled) {
                (void)cudaGetLastError();
            }
        }
    }
    execution.model_parallel_collectives.configure(
        config.devices, allow_duplicate_devices);
    MFQ_CUDA_CHECK(cudaSetDevice(config.primary_device()));
    print_parallel_config(
        "tensor_parallel", execution.tensor_parallel);
    print_parallel_config(
        "expert_parallel", execution.expert_parallel);
    std::cerr << "model_parallel ranks=" << config.devices.size()
              << " collective_backend="
              << (execution.model_parallel_collectives.collectives_enabled
                  ? "nccl" : "serial")
              << '\n';
}

static void configure_layer_placement(
        CudaExecutionContext& execution,
        const CudaParallelOptions& options) {
    auto& placement = execution.layer_placement;
    placement = {};
    if (options.devices.empty()) {
        if (!options.split.empty()) {
            throw std::runtime_error(
                "--layer-split requires --layer-parallel");
        }
        placement.devices = {
            model_parallel_primary_device(execution)};
        return;
    }
    if (execution.tensor_parallel.enabled() ||
            execution.expert_parallel.enabled()) {
        throw std::runtime_error(
            "--layer-parallel cannot be combined with tensor/expert parallelism");
    }

    int available = 0;
    MFQ_CUDA_CHECK(cudaGetDeviceCount(&available));
    auto parsed = validated_parallel_config(options,
        "--layer-parallel", "--layer-split", available, false);
    MFQ_CUDA_CHECK(cudaSetDevice(parsed.primary_device()));
    print_parallel_config("layer_parallel", parsed);
    placement.devices = std::move(parsed.devices);
    placement.split = std::move(parsed.split);
}

void setup_cuda_load(
        const mfq::cuda::CudaLoadOptions& options,
        CudaExecutionContext& execution) {
    execution.reset();
    if (options.n_gpu_layers) execution.n_gpu_layers = *options.n_gpu_layers;
    if (options.cpu_threads && *options.cpu_threads <= 0) {
        throw std::runtime_error("--threads must be positive");
    }
    if (options.cpu_threads) mfq_set_num_threads(*options.cpu_threads);
    configure_model_parallel(
        execution, options.tensor_parallel, options.expert_parallel,
        options.parallel_test_duplicates);
    configure_layer_placement(execution, options.layer_parallel);
    if (!options.cpu_offload_layers.empty()) {
        if (std::any_of(
                options.cpu_offload_layers.begin(),
                options.cpu_offload_layers.end(),
                [](int layer) { return layer < 0; })) {
            throw std::runtime_error(
                "--cpu-offload-layers must be non-negative");
        }
        execution.dsv4_cpu_offload_layers = {
            options.cpu_offload_layers.begin(),
            options.cpu_offload_layers.end()};
        std::vector<int> ordered(
            execution.dsv4_cpu_offload_layers.begin(),
            execution.dsv4_cpu_offload_layers.end());
        std::sort(ordered.begin(), ordered.end());
        std::cerr << "cpu_offload_layers=";
        for (size_t index = 0; index < ordered.size(); ++index) {
            if (index) std::cerr << ',';
            std::cerr << ordered[index];
        }
        std::cerr << std::endl;
    }
    if (options.n_gpu_layers && *options.n_gpu_layers < 0) {
        throw std::runtime_error("--n-gpu-layers must be non-negative");
    }
    if (options.n_gpu_layers && !options.cpu_offload_layers.empty()) {
        throw std::runtime_error(
            "--n-gpu-layers cannot be combined with --cpu-offload-layers");
    }
    if (options.moe_gpu_cache_gb < 0.0 ||
            !std::isfinite(options.moe_gpu_cache_gb)) {
        throw std::runtime_error(
            "--moe-gpu-cache-gb must be finite and non-negative");
    }
    if (options.moe_gpu_cache_gb > 0.0 &&
            !options.cpu_offload_layers.empty()) {
        throw std::runtime_error(
            "--moe-gpu-cache-gb cannot be combined with "
            "--cpu-offload-layers");
    }
    if (options.moe_gpu_cache_gb > 0.0 &&
            (execution.tensor_parallel.enabled() ||
             execution.expert_parallel.enabled())) {
        throw std::runtime_error(
            "--moe-gpu-cache-gb cannot be combined with "
            "tensor/expert parallelism");
    }
    if (!options.moe_cache_profile_path.empty() &&
            options.moe_gpu_cache_gb <= 0.0) {
        throw std::runtime_error(
            "--moe-cache-profile requires --moe-gpu-cache-gb");
    }
    if (options.moe_gpu_cache_gb > 0.0) {
        constexpr double gib =
            1024.0 * 1024.0 * 1024.0;
        const double bytes = options.moe_gpu_cache_gb * gib;
        if (bytes < 1.0 ||
                bytes >
                    static_cast<double>(
                        std::numeric_limits<int64_t>::max())) {
            throw std::runtime_error(
                "--moe-gpu-cache-gb is outside the supported range");
        }
        execution.moe_expert_cache =
            make_moe_expert_cache(
                static_cast<int64_t>(bytes), execution.config);
        if (!options.moe_cache_profile_path.empty()) {
            set_moe_expert_cache_profile(
                *execution.moe_expert_cache,
                mfq::load_moe_cache_profile(
                    options.moe_cache_profile_path));
        }
        execution.drop_file_cache = true;
    }
}

int with_cuda_load(
        const CudaLoadOptions& options,
        const std::function<int(CudaExecutionContext&)>& run) {
    CudaExecutionContext execution;
    CudaExecutionScope scope(execution);
    setup_cuda_load(options, execution);
    return run(execution);
}

} // namespace mfq::cuda::internal
