#include "runner.h"
#include "cuda_execution.h"
#include "moe_expert_cache.h"
#include "moe_cache_profile.h"
#include "mfq_tensor_backend.h"
#include "tensor_parallel.h"

#include <cuda_runtime_api.h>
#include <algorithm>
#include <charconv>
#include <cctype>
#include <cmath>
#include <fstream>
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

} // namespace

std::vector<int64_t> parse_ids(const std::string & value) {
    std::vector<int64_t> ids;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ',')) {
        const auto first = item.find_first_not_of(" \t\r\n");
        const auto last = item.find_last_not_of(" \t\r\n");
        if (first == std::string::npos) {
            throw std::runtime_error("token ID lists cannot contain empty items");
        }
        item = item.substr(first, last - first + 1);
        int64_t id = 0;
        const auto [end, error] = std::from_chars(
            item.data(), item.data() + item.size(), id);
        if (error != std::errc{} || end != item.data() + item.size()) {
            throw std::runtime_error(
                "token ID lists must contain comma-separated integers");
        }
        ids.push_back(id);
    }
    if (ids.empty() || value.ends_with(',')) {
        throw std::runtime_error("token ID lists require at least one integer");
    }
    return ids;
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

KlMmqMode parse_kl_mmq_mode(const std::string & value) {
    if (value == "default") return KlMmqMode::Default;
    if (value == "nint8_1") return KlMmqMode::Nint8One;
    if (value == "fp16") return KlMmqMode::Fp16;
    throw std::runtime_error(
        "--kl-mmq must be default, nint8_1, or fp16");
}

std::vector<KlMmqMode> parse_kl_mmq_sequence(
        const std::string & value) {
    std::vector<KlMmqMode> modes;
    for (const auto & item :
         split_csv_values(value, "--kl-mmq-sequence")) {
        const KlMmqMode mode = parse_kl_mmq_mode(item);
        if (mode == KlMmqMode::Default) {
            throw std::runtime_error(
                "--kl-mmq-sequence accepts only nint8_1 and fp16");
        }
        if (std::find(modes.begin(), modes.end(), mode) != modes.end()) {
            throw std::runtime_error(
                "--kl-mmq-sequence contains a duplicate mode");
        }
        modes.push_back(mode);
    }
    return modes;
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
        const std::string & tensor_devices_arg,
        const std::string & tensor_split_arg,
        const std::string & expert_devices_arg,
        const std::string & expert_split_arg,
        bool allow_duplicate_devices = false) {
    g_model_parallel_collectives.reset();
    g_tensor_parallel = {};
    g_expert_parallel = {};

    int available = 0;
    MFQ_CUDA_CHECK(cudaGetDeviceCount(&available));
    g_tensor_parallel = parse_parallel_config(
        tensor_devices_arg, tensor_split_arg,
        "--tensor-parallel", "--tensor-split",
        available, allow_duplicate_devices);
    g_expert_parallel = parse_parallel_config(
        expert_devices_arg, expert_split_arg,
        "--expert-parallel", "--expert-split",
        available, allow_duplicate_devices);

    if (g_tensor_parallel.enabled() &&
            g_expert_parallel.enabled() &&
            g_tensor_parallel.devices != g_expert_parallel.devices) {
        throw std::runtime_error(
            "combined tensor and expert parallelism requires the same "
            "ordered CUDA device group");
    }
    if (!model_parallel_enabled()) {
        MFQ_CUDA_CHECK(cudaSetDevice(0));
        return;
    }

    const auto & config = model_parallel_config();
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
    g_model_parallel_collectives.configure(
        config.devices, allow_duplicate_devices);
    MFQ_CUDA_CHECK(cudaSetDevice(model_parallel_primary_device()));
    print_parallel_config("tensor_parallel", g_tensor_parallel);
    print_parallel_config("expert_parallel", g_expert_parallel);
    std::cerr << "model_parallel ranks=" << config.devices.size()
              << " collective_backend="
              << (g_model_parallel_collectives.collectives_enabled
                  ? "nccl" : "serial")
              << '\n';
}

static void configure_layer_placement(
        const std::string & devices_arg,
        const std::string & split_arg) {
    g_layer_placement = {};
    if (devices_arg.empty()) {
        if (!split_arg.empty()) {
            throw std::runtime_error(
                "--layer-split requires --layer-parallel");
        }
        return;
    }
    if (model_parallel_enabled()) {
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
        g_layer_placement.devices.resize(static_cast<size_t>(count));
        std::iota(
            g_layer_placement.devices.begin(),
            g_layer_placement.devices.end(), 0);
    } else {
        for (const auto & item : device_values) {
            g_layer_placement.devices.push_back(
                strict_int(item, "--layer-parallel"));
        }
        if (g_layer_placement.devices.size() < 2) {
            throw std::runtime_error(
                "--layer-parallel requires at least two devices");
        }
    }

    int available = 0;
    MFQ_CUDA_CHECK(cudaGetDeviceCount(&available));
    std::unordered_set<int> unique_devices;
    for (int device : g_layer_placement.devices) {
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
            g_layer_placement.split.push_back(weight);
        }
        if (g_layer_placement.split.size() !=
                g_layer_placement.devices.size()) {
            throw std::runtime_error(
                "--layer-split count must match --layer-parallel devices");
        }
    }
    MFQ_CUDA_CHECK(cudaSetDevice(g_layer_placement.primary_device()));
    std::cerr << "layer_parallel devices=";
    for (size_t index = 0;
         index < g_layer_placement.devices.size(); ++index) {
        if (index) std::cerr << ',';
        std::cerr << g_layer_placement.devices[index];
    }
    if (!g_layer_placement.split.empty()) {
        std::cerr << " split=";
        for (size_t index = 0;
             index < g_layer_placement.split.size(); ++index) {
            if (index) std::cerr << ',';
            std::cerr << g_layer_placement.split[index];
        }
    }
    std::cerr << '\n';
}

std::vector<int64_t> load_ids_file(const std::string & path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("cannot open --ids-file: " + path);
    const std::streamsize bytes = input.tellg();
    if (bytes <= 0 || bytes % static_cast<std::streamsize>(sizeof(int32_t)) != 0) {
        throw std::runtime_error("--ids-file must contain raw int32 token ids");
    }
    input.seekg(0);
    std::vector<int32_t> stored(
        static_cast<size_t>(bytes / static_cast<std::streamsize>(sizeof(int32_t))));
    input.read(reinterpret_cast<char *>(stored.data()), bytes);
    if (!input) throw std::runtime_error("truncated --ids-file: " + path);
    return std::vector<int64_t>(stored.begin(), stored.end());
}

std::unordered_set<int> parse_layer_ranges(
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

void setup_cuda_load(const mfq::cuda::CudaLoadOptions& options) {
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
        if (n_gpu_layers_set) g_n_gpu_layers = n_gpu_layers;
        if (cpu_threads_set && cpu_threads <= 0) {
            throw std::runtime_error("--threads must be positive");
        }
        if (cpu_threads > 0) {
            mfq_set_num_threads(cpu_threads);
        }
        configure_model_parallel(
            tensor_parallel_arg, tensor_split_arg,
            expert_parallel_arg, expert_split_arg,
            parallel_test_duplicates);
        configure_layer_placement(
            layer_parallel_arg, layer_split_arg);
        if (n_gpu_layers_set && g_n_gpu_layers < 0) {
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
                model_parallel_enabled()) {
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
            g_moe_expert_cache =
                make_moe_expert_cache(
                    static_cast<int64_t>(bytes));
            if (!moe_cache_profile_path.empty()) {
                set_moe_expert_cache_profile(
                    mfq::load_moe_cache_profile(
                        moe_cache_profile_path));
            }
            g_mfq_drop_file_cache = true;
        }
}

void reset_cuda_load() noexcept {
    g_model_parallel_collectives.reset();
}

} // namespace mfq::cuda::internal
