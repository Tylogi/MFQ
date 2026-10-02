#include "options.h"
#include "cuda_runtime_config.h"
#include "cuda_execution.h"
#include "storage/moe_expert_cache.h"
#include "moe_cache_profile.h"
#include "mfq_tensor_backend.h"
#include "quant_linear.h"
#include "tensor_parallel.h"

#include <cuda_runtime_api.h>
#include <algorithm>
#include <charconv>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace mfq::cuda::internal {
namespace {

int strict_int(const std::string& text, const char* option) {
    int result = 0;
    const auto [end, error] = std::from_chars(
        text.data(), text.data() + text.size(), result);
    if (error != std::errc{} || end != text.data() + text.size()) {
        throw std::runtime_error(std::string(option) + " requires integers");
    }
    return result;
}

double strict_double(const std::string& text, const char* option) {
    double result = 0.0;
    const auto [end, error] = std::from_chars(
        text.data(), text.data() + text.size(), result,
        std::chars_format::general);
    if (error != std::errc{} || end != text.data() + text.size() ||
            !std::isfinite(result)) {
        throw std::runtime_error(
            std::string(option) + " requires finite numbers");
    }
    return result;
}

std::uint64_t environment_uint64(
        const char* name, std::uint64_t fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') return fallback;
    std::uint64_t result = 0;
    const auto length = std::char_traits<char>::length(value);
    const auto [end, error] = std::from_chars(value, value + length, result);
    if (error != std::errc{} || end != value + length) {
        throw std::runtime_error(std::string("invalid ") + name);
    }
    return result;
}

bool environment_enabled(const char* name, bool fallback = true) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') return fallback;
    return strict_int(value, name) != 0;
}

std::filesystem::path default_prefix_cache_directory() {
    if (const char* configured =
            std::getenv("MFQ_RUNTIME_PREFIX_CACHE_DIR")) {
        if (configured[0] != '\0') return configured;
    }
#ifdef _WIN32
    if (const char* local = std::getenv("LOCALAPPDATA")) {
        if (local[0] != '\0') {
            return std::filesystem::path(local) /
                "TyloQuant" / "MFQ" / "prefix-cache";
        }
    }
#else
    if (const char* xdg = std::getenv("XDG_CACHE_HOME")) {
        if (xdg[0] != '\0') {
            return std::filesystem::path(xdg) /
                "tyloquant" / "mfq" / "prefix-cache";
        }
    }
    if (const char* home = std::getenv("HOME")) {
        if (home[0] != '\0') {
            return std::filesystem::path(home) / ".cache" /
                "tyloquant" / "mfq" / "prefix-cache";
        }
    }
#endif
    return std::filesystem::temp_directory_path() /
        "tyloquant-mfq-prefix-cache";
}

std::size_t checked_size(std::uint64_t value, const char* name) {
    if (value > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error(std::string(name) + " exceeds size_t");
    }
    return static_cast<std::size_t>(value);
}

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
    if (options.prefill_chunk_size <= 0) {
        throw std::invalid_argument("prefill chunk size must be positive");
    }
    if (options.continuous_batching < 0) {
        throw std::invalid_argument(
            "continuous batching capacity must be non-negative");
    }

    CudaRuntimeConfig config;
    config.generation.prefill_chunk_size = options.prefill_chunk_size;
    config.decode_graph.enabled = environment_enabled(
        "MFQ_RUNTIME_CUDA_GRAPH", true);
    config.decode_graph.trace = environment_enabled(
        "MFQ_RUNTIME_TRACE_CUDA_GRAPH", false);
    config.decode_graph.minimum_generation_tokens = graph_minimum(
        "MFQ_RUNTIME_CUDA_GRAPH_MIN_TOKENS");

    config.continuous_batch.scheduling.max_sequences =
        static_cast<std::size_t>(options.continuous_batching);
    config.continuous_batch.greedy = environment_enabled(
        "MFQ_CONTINUOUS_BATCH_GREEDY", true);
    config.continuous_batch.cuda_graph =
        config.decode_graph.enabled && environment_enabled(
            "MFQ_CONTINUOUS_BATCH_CUDA_GRAPH", true);
    config.continuous_batch.paged_kv = environment_enabled(
        "MFQ_CONTINUOUS_PAGED_KV", true);
    config.continuous_batch.cuda_graph_minimum_tokens = graph_minimum(
        "MFQ_CONTINUOUS_BATCH_CUDA_GRAPH_MIN_TOKENS");
    config.continuous_batch.prefill_token_budget =
        static_cast<std::int64_t>(std::min<std::uint64_t>(
            environment_uint64(
                "MFQ_CONTINUOUS_BATCH_PREFILL_TOKEN_BUDGET",
                static_cast<std::uint64_t>(options.prefill_chunk_size)),
            static_cast<std::uint64_t>(
                std::numeric_limits<std::int64_t>::max())));
    if (config.continuous_batch.prefill_token_budget < 1) {
        throw std::invalid_argument(
            "continuous batching prefill token budget must be positive");
    }

    auto& sessions = config.session_cache;
    sessions.snapshots.max_sessions = checked_size(environment_uint64(
        "MFQ_RUNTIME_MAX_KV_SESSIONS", sessions.snapshots.max_sessions),
        "MFQ_RUNTIME_MAX_KV_SESSIONS");
    sessions.snapshots.max_snapshots_per_session = checked_size(
        environment_uint64(
            "MFQ_RUNTIME_MAX_KV_SNAPSHOTS_PER_SESSION",
            sessions.snapshots.max_snapshots_per_session),
        "MFQ_RUNTIME_MAX_KV_SNAPSHOTS_PER_SESSION");
    sessions.snapshots.max_bytes = checked_size(environment_uint64(
        "MFQ_RUNTIME_KV_SESSION_BYTES", sessions.snapshots.max_bytes),
        "MFQ_RUNTIME_KV_SESSION_BYTES");
    sessions.trace = environment_enabled(
        "MFQ_RUNTIME_TRACE_SESSION_CACHE", false);

    auto& prefix = config.prefix_cache;
    prefix.directory = default_prefix_cache_directory();
    prefix.enabled = !environment_enabled(
        "MFQ_RUNTIME_DISABLE_PREFIX_CACHE", false);
    prefix.block_tokens = environment_uint64(
        "MFQ_RUNTIME_PREFIX_CACHE_BLOCK_TOKENS", prefix.block_tokens);
    if (prefix.block_tokens == 0 || prefix.block_tokens > 65536) {
        throw std::runtime_error(
            "MFQ_RUNTIME_PREFIX_CACHE_BLOCK_TOKENS must be in [1, 65536]");
    }
    prefix.disk_bytes = environment_uint64(
        "MFQ_RUNTIME_PREFIX_CACHE_DISK_BYTES", prefix.disk_bytes);
    prefix.hot_bytes = environment_uint64(
        "MFQ_RUNTIME_PREFIX_CACHE_HOT_BYTES", prefix.hot_bytes);
    prefix.pending_writes = checked_size(environment_uint64(
        "MFQ_RUNTIME_PREFIX_CACHE_PENDING_WRITES", prefix.pending_writes),
        "MFQ_RUNTIME_PREFIX_CACHE_PENDING_WRITES");
    prefix.pending_bytes = environment_uint64(
        "MFQ_RUNTIME_PREFIX_CACHE_PENDING_BYTES", prefix.pending_bytes);
    return config;
}

static std::vector<std::string> split_csv_values(
        const std::string & value,
        const char * option) {
    std::vector<std::string> result;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ',')) {
        const auto first = item.find_first_not_of(" \t\r\n");
        const auto last = item.find_last_not_of(" \t\r\n");
        if (first == std::string::npos) {
            throw std::runtime_error(
                std::string(option) + " contains an empty item");
        }
        result.push_back(item.substr(first, last - first + 1));
    }
    if (result.empty()) {
        throw std::runtime_error(
            std::string(option) +
            " requires at least one value");
    }
    if (value.ends_with(',')) {
        throw std::runtime_error(
            std::string(option) + " contains an empty item");
    }
    return result;
}

static ParallelConfig parse_parallel_config(
        const std::string & devices_arg,
        const std::string & split_arg,
        const char * devices_option,
        const char * split_option,
        int available_devices,
        bool allow_duplicate_devices) {
    ParallelConfig config;
    if (devices_arg.empty()) {
        if (!split_arg.empty()) {
            throw std::runtime_error(
                std::string(split_option) + " requires " + devices_option);
        }
        return config;
    }

    const auto values = split_csv_values(
        devices_arg, devices_option);
    if (values.size() == 1 &&
            devices_arg.find(',') == std::string::npos) {
        const int count = strict_int(values.front(), devices_option);
        if (count < 2) {
            throw std::runtime_error(
                std::string(devices_option) +
                " device count must be at least two");
        }
        config.devices.resize(static_cast<size_t>(count));
        std::iota(config.devices.begin(), config.devices.end(), 0);
    } else {
        for (const auto & item : values) {
            config.devices.push_back(strict_int(item, devices_option));
        }
        if (config.devices.size() < 2) {
            throw std::runtime_error(
                std::string(devices_option) +
                " requires at least two devices");
        }
    }
    config.allow_duplicate_devices = allow_duplicate_devices;

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

    if (!split_arg.empty()) {
        for (const auto & item :
             split_csv_values(split_arg, split_option)) {
            const double weight = strict_double(item, split_option);
            if (!std::isfinite(weight) || weight <= 0.0) {
                throw std::runtime_error(
                    std::string(split_option) +
                    " values must be finite and positive");
            }
            config.split.push_back(weight);
        }
        if (config.split.size() != config.devices.size()) {
            throw std::runtime_error(
                std::string(split_option) + " count must match " +
                devices_option + " devices");
        }
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
        const std::string & tensor_devices_arg,
        const std::string & tensor_split_arg,
        const std::string & expert_devices_arg,
        const std::string & expert_split_arg,
        bool allow_duplicate_devices = false) {
    execution.model_parallel_collectives.reset();
    execution.tensor_parallel = {};
    execution.expert_parallel = {};

    int available = 0;
    MFQ_CUDA_CHECK(cudaGetDeviceCount(&available));
    execution.tensor_parallel = parse_parallel_config(
        tensor_devices_arg, tensor_split_arg,
        "--tensor-parallel", "--tensor-split",
        available, allow_duplicate_devices);
    execution.expert_parallel = parse_parallel_config(
        expert_devices_arg, expert_split_arg,
        "--expert-parallel", "--expert-split",
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
        const std::string & devices_arg,
        const std::string & split_arg) {
    auto& placement = execution.layer_placement;
    placement = {};
    if (devices_arg.empty()) {
        if (!split_arg.empty()) {
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

    const auto device_values =
        split_csv_values(devices_arg, "--layer-parallel");
    if (device_values.size() == 1 &&
            devices_arg.find(',') == std::string::npos) {
        const int count = strict_int(
            device_values.front(), "--layer-parallel");
        if (count < 2) {
            throw std::runtime_error(
                "--layer-parallel device count must be at least 2");
        }
        placement.devices.resize(static_cast<size_t>(count));
        std::iota(
            placement.devices.begin(), placement.devices.end(), 0);
    } else {
        for (const auto & item : device_values) {
            placement.devices.push_back(
                strict_int(item, "--layer-parallel"));
        }
        if (placement.devices.size() < 2) {
            throw std::runtime_error(
                "--layer-parallel requires at least two devices");
        }
    }

    int available = 0;
    MFQ_CUDA_CHECK(cudaGetDeviceCount(&available));
    std::unordered_set<int> unique_devices;
    for (int device : placement.devices) {
        if (device < 0 || device >= available) {
            throw std::runtime_error(
                "layer-placement CUDA device is unavailable: " +
                std::to_string(device));
        }
        if (!unique_devices.insert(device).second) {
            throw std::runtime_error(
                "layer-placement CUDA devices must be unique");
        }
    }
    if (!split_arg.empty()) {
        for (const auto & item :
             split_csv_values(split_arg, "--layer-split")) {
            const double weight = strict_double(item, "--layer-split");
            if (weight <= 0.0) {
                throw std::runtime_error(
                    "--layer-split values must be positive");
            }
            placement.split.push_back(weight);
        }
        if (placement.split.size() != placement.devices.size()) {
            throw std::runtime_error(
                "--layer-split count must match --layer-parallel devices");
        }
    }
    MFQ_CUDA_CHECK(cudaSetDevice(placement.primary_device()));
    std::cerr << "layer_parallel devices=";
    for (size_t index = 0; index < placement.devices.size(); ++index) {
        if (index) std::cerr << ',';
        std::cerr << placement.devices[index];
    }
    if (!placement.split.empty()) {
        std::cerr << " split=";
        for (size_t index = 0; index < placement.split.size(); ++index) {
            if (index) std::cerr << ',';
            std::cerr << placement.split[index];
        }
    }
    std::cerr << '\n';
}

static std::unordered_set<int> parse_layer_ranges(
        const std::string & value) {
    std::unordered_set<int> result;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ',')) {
        item.erase(
            std::remove_if(
                item.begin(), item.end(),
                [](unsigned char ch) { return std::isspace(ch) != 0; }),
            item.end());
        if (item.empty()) {
            throw std::runtime_error(
                "--cpu-offload-layers contains an empty item");
        }
        const size_t dash = item.find('-');
        int first = 0;
        int last = 0;
        if (dash == std::string::npos) {
            first = last = strict_int(item, "--cpu-offload-layers");
        } else {
            if (dash == 0 || dash + 1 >= item.size() ||
                item.find('-', dash + 1) != std::string::npos) {
                throw std::runtime_error(
                    "invalid --cpu-offload-layers range: " + item);
            }
            first = strict_int(
                item.substr(0, dash), "--cpu-offload-layers");
            last = strict_int(
                item.substr(dash + 1), "--cpu-offload-layers");
        }
        if (first < 0 || last < first) {
            throw std::runtime_error(
                "invalid --cpu-offload-layers range: " + item);
        }
        for (int layer = first; layer <= last; ++layer) {
            result.insert(layer);
        }
    }
    if (result.empty()) {
        throw std::runtime_error(
            "--cpu-offload-layers requires at least one layer");
    }
    return result;
}

void setup_cuda_load(
        const mfq::cuda::CudaLoadOptions& options,
        CudaExecutionContext& execution) {
    execution.reset();
    const auto& tensor_parallel_arg = options.tensor_parallel_arg;
    const auto& tensor_split_arg = options.tensor_split_arg;
    const auto& expert_parallel_arg = options.expert_parallel_arg;
    const auto& expert_split_arg = options.expert_split_arg;
    const auto& parallel_test_duplicates = options.parallel_test_duplicates;
    const auto& layer_parallel_arg = options.layer_parallel_arg;
    const auto& layer_split_arg = options.layer_split_arg;
    const auto& cpu_offload_layers_arg = options.cpu_offload_layers_arg;
    const auto& moe_gpu_cache_gb = options.moe_gpu_cache_gb;
    const auto& moe_cache_profile_path = options.moe_cache_profile_path;
    const auto& n_gpu_layers_set = options.n_gpu_layers_set;
    const auto& n_gpu_layers = options.n_gpu_layers;
    const auto& cpu_threads_set = options.cpu_threads_set;
    const auto& cpu_threads = options.cpu_threads;
        if (n_gpu_layers_set) execution.n_gpu_layers = n_gpu_layers;
        if (cpu_threads_set && cpu_threads <= 0) {
            throw std::runtime_error("--threads must be positive");
        }
        if (cpu_threads > 0) {
            mfq_set_num_threads(cpu_threads);
        }
        configure_model_parallel(
            execution, tensor_parallel_arg, tensor_split_arg,
            expert_parallel_arg, expert_split_arg,
            parallel_test_duplicates);
        configure_layer_placement(
            execution, layer_parallel_arg, layer_split_arg);
        if (!cpu_offload_layers_arg.empty()) {
            execution.dsv4_cpu_offload_layers =
                parse_layer_ranges(cpu_offload_layers_arg);
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
        if (n_gpu_layers_set && execution.n_gpu_layers < 0) {
            throw std::runtime_error("--n-gpu-layers must be non-negative");
        }
        if (n_gpu_layers_set && !cpu_offload_layers_arg.empty()) {
            throw std::runtime_error(
                "--n-gpu-layers cannot be combined with --cpu-offload-layers");
        }
        if (moe_gpu_cache_gb < 0.0 ||
                !std::isfinite(moe_gpu_cache_gb)) {
            throw std::runtime_error(
                "--moe-gpu-cache-gb must be finite and non-negative");
        }
        if (moe_gpu_cache_gb > 0.0 &&
                !cpu_offload_layers_arg.empty()) {
            throw std::runtime_error(
                "--moe-gpu-cache-gb cannot be combined with "
                "--cpu-offload-layers");
        }
        if (moe_gpu_cache_gb > 0.0 &&
                (execution.tensor_parallel.enabled() ||
                 execution.expert_parallel.enabled())) {
            throw std::runtime_error(
                "--moe-gpu-cache-gb cannot be combined with "
                "tensor/expert parallelism");
        }
        if (!moe_cache_profile_path.empty() &&
                moe_gpu_cache_gb <= 0.0) {
            throw std::runtime_error(
                "--moe-cache-profile requires --moe-gpu-cache-gb");
        }
        if (moe_gpu_cache_gb > 0.0) {
            constexpr double gib =
                1024.0 * 1024.0 * 1024.0;
            const double bytes = moe_gpu_cache_gb * gib;
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
            if (!moe_cache_profile_path.empty()) {
                set_moe_expert_cache_profile(
                    *execution.moe_expert_cache,
                    mfq::load_moe_cache_profile(
                        moe_cache_profile_path));
            }
            execution.drop_file_cache = true;
        }
}

int with_cuda_load(
        const CudaLoadOptions& options,
        const std::function<int(CudaExecutionContext&)>& run) {
    CudaExecutionContext execution;
    setup_cuda_load(options, execution);
    return run(execution);
}

} // namespace mfq::cuda::internal
