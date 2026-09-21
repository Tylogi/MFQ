#include "mfq_cuda_runtime.h"
#include "moe_expert_cache.h"
#include "cuda_execution.h"
#include "cuda_sampling.h"
#include "diagnostics/backend_checks.h"
#include "quant_linear.h"
#include "../models/qwen35/qwen35_linear_attention.h"
#include "../models/registry.h"
#include "mfq_tensor_backend.h"
#include "prepared_prompt.h"
#include "mfq/kernels/cuda/deepseek_v4_attention.h"
#include "mfq/kernels/cuda/deepseek_v4_hc.h"
#include "mfq/kernels/cuda/deepseek_v41.h"
#include "mfq/kernels/cuda/fp8_sq.h"
#include "mfq/kernels/cuda/mxfp4_sq.h"
#include "cuda_model_plan.h"
#include "mfq_cuda_mtp.h"
#include "mfq_cuda_paged_kv.h"
#include "mfq_cuda_ops.h"
#include "glm5_next/model.h"
#include "models/deepseek_v41.h"
#include "qwen4_exp/model.h"
#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>

#ifdef MFQ_HAVE_NCCL
#include <nccl.h>
#endif

#include "transport.h"
#include "mfq_format_compat.h"
#include "grid_vision.h"
#include "mfq_paged_prefix_cache.h"
#include "moe_cache_transfer.h"
#include "moe_cache_policy.h"
#include "moe_cache_profile.h"
#include "mfe_expert_store.h"
#include "tensor_parallel.h"
#include "nvq_codebooks.generated.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cctype>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <list>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(__GLIBC__)
#include <malloc.h>
#endif

#if defined(__x86_64__) && defined(__GNUC__)
#include <immintrin.h>
#define MFQ_CPU_X86_GNU 1
#endif

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif

#ifndef MFQ_CUDA_CHECK
#define MFQ_CUDA_CHECK(expr) do { \
    cudaError_t err__ = (expr); \
    if (err__ != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err__)); \
    } \
} while (0)
#endif

#ifdef MFQ_HAVE_NCCL
#define MFQ_NCCL_CHECK(expr) do { \
    ncclResult_t err__ = (expr); \
    if (err__ != ncclSuccess) { \
        throw std::runtime_error(std::string("NCCL error: ") + ncclGetErrorString(err__)); \
    } \
} while (0)
#endif

#include "causal_lm.h"
#include "runtime_components.h"

static std::vector<int64_t> parse_ids(const std::string & s) {
    std::vector<int64_t> ids;
    std::regex re("-?\\d+");
    for (auto it = std::sregex_iterator(s.begin(), s.end(), re); it != std::sregex_iterator(); ++it) {
        ids.push_back(std::stoll((*it)[0].str()));
    }
    if (ids.empty()) throw std::runtime_error("--ids must contain at least one token id");
    return ids;
}

static std::vector<std::string> split_csv_values(
        const std::string & value,
        const char * option) {
    std::vector<std::string> result;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ',')) {
        item.erase(
            std::remove_if(
                item.begin(), item.end(),
                [](unsigned char ch) {
                    return std::isspace(ch) != 0;
                }),
            item.end());
        if (item.empty()) {
            throw std::runtime_error(
                std::string(option) +
                " contains an empty item");
        }
        result.push_back(std::move(item));
    }
    if (result.empty()) {
        throw std::runtime_error(
            std::string(option) +
            " requires at least one value");
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
        const int count = std::stoi(values.front());
        if (count < 2) {
            throw std::runtime_error(
                std::string(devices_option) +
                " device count must be at least two");
        }
        config.devices.resize(static_cast<size_t>(count));
        std::iota(config.devices.begin(), config.devices.end(), 0);
    } else {
        for (const auto & item : values) {
            config.devices.push_back(std::stoi(item));
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
            const double weight = std::stod(item);
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
        const int count = std::stoi(device_values.front());
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
            g_layer_placement.devices.push_back(std::stoi(item));
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
            g_layer_placement.split.push_back(std::stod(item));
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

static std::vector<int64_t> load_ids_file(const std::string & path) {
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
            first = last = std::stoi(item);
        } else {
            if (dash == 0 || dash + 1 >= item.size() ||
                item.find('-', dash + 1) != std::string::npos) {
                throw std::runtime_error(
                    "invalid --cpu-offload-layers range: " + item);
            }
            first = std::stoi(item.substr(0, dash));
            last = std::stoi(item.substr(dash + 1));
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

template <typename Model>
static int run_prefill_sweep(
    Model& model,
    const std::vector<int64_t> & sizes,
    int repeats) {
    if (repeats < 1) throw std::runtime_error("--prefill-sweep-reps must be positive");
    const int64_t max_m = *std::max_element(sizes.begin(), sizes.end());
    if (max_m > model.max_position_embeddings()) {
        throw std::runtime_error("--prefill-sweep exceeds the configured context size");
    }

    std::vector<int64_t> token_ids((size_t)max_m);
    const int64_t token_span = std::max<int64_t>(
        1, std::min<int64_t>(1024, model.vocab_size() - 2));
    for (int64_t i = 0; i < max_m; ++i) token_ids[(size_t)i] = 1 + i % token_span;
    auto all_ids = mfq_tensor_backend::tensor(
        token_ids, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64).device(mfq_tensor_backend::kCUDA)).unsqueeze(0);

    for (int64_t m : sizes) {
        auto ids = all_ids.narrow(1, 0, m);
        for (int warmup = 0; warmup < 2; ++warmup) {
            model.reset(1);
            (void)model.last_logits(ids);
            mfq_cuda_synchronize();
        }

        std::vector<double> elapsed_ms;
        elapsed_ms.reserve((size_t)repeats);
        int64_t top = -1;
        for (int repeat = 0; repeat < repeats; ++repeat) {
            model.reset(1);
            auto started = std::chrono::steady_clock::now();
            auto logits = model.last_logits(ids);
            mfq_cuda_synchronize();
            auto ended = std::chrono::steady_clock::now();
            elapsed_ms.push_back(std::chrono::duration<double, std::milli>(ended - started).count());
            if (repeat == 0) top = logits.argmax(-1).template item<int64_t>();
        }
        std::sort(elapsed_ms.begin(), elapsed_ms.end());
        const double median_ms = elapsed_ms[elapsed_ms.size() / 2];
        std::cout << "prefill_sweep_m=" << m
                  << " median_ms=" << median_ms
                  << " min_ms=" << elapsed_ms.front()
                  << " max_ms=" << elapsed_ms.back()
                  << " tok_per_s=" << (1000.0 * (double)m / median_ms)
                  << " top=" << top << "\n";
    }
    return 0;
}

template <typename Model>
static int run_block_trace_compare(
    Model& test,
    const std::string & reference_model_path,
    const std::string & config_path,
    int64_t context_size,
    mfq_tensor_backend::Tensor ids)
{
    Model reference = [&]() {
        if (g_n_gpu_layers < 0) {
            return mfq::cuda::load_causal_lm<Model::backbone>(
                reference_model_path, config_path, context_size);
        }
        const int saved_n_gpu_layers = g_n_gpu_layers;
        const int saved_cpu_layers = g_dense_cpu_layer_count;
        g_n_gpu_layers = -1;
        try {
            auto loaded = mfq::cuda::load_causal_lm<Model::backbone>(
                reference_model_path, config_path, context_size);
            g_n_gpu_layers = saved_n_gpu_layers;
            g_dense_cpu_layer_count = saved_cpu_layers;
            return loaded;
        } catch (...) {
            g_n_gpu_layers = saved_n_gpu_layers;
            g_dense_cpu_layer_count = saved_cpu_layers;
            throw;
        }
    }();
    std::vector<mfq_tensor_backend::Tensor> test_trace;
    std::vector<mfq_tensor_backend::Tensor> reference_trace;

    test.reset(ids.size(0));
    auto test_hidden = test.hidden_forward(ids, mfq_nullopt, mfq_nullopt, &test_trace);
    reference.reset(ids.size(0));
    auto reference_hidden = reference.hidden_forward(
        ids, mfq_nullopt, mfq_nullopt, &reference_trace);
    mfq_cuda_synchronize();

    if (test_trace.size() != reference_trace.size()) {
        throw std::runtime_error("block trace stage count mismatch");
    }
    for (size_t i = 0; i < test_trace.size(); ++i) {
        auto ref = reference_trace[i].reshape({-1}).to(mfq_tensor_backend::kFloat64);
        auto got = test_trace[i].reshape({-1}).to(mfq_tensor_backend::kFloat64);
        if (ref.numel() != got.numel()) {
            throw std::runtime_error("block trace tensor size mismatch");
        }
        auto ref_norm = ref.norm();
        auto got_norm = got.norm();
        const double ref_norm_value = ref_norm.template item<double>();
        const double denominator = std::max(ref_norm_value, 1.0e-30);
        const double relative_l2 = (got - ref).norm().template item<double>() / denominator;
        const double cosine = mfq_tensor_backend::dot(ref, got).template item<double>() /
            std::max(ref_norm_value * got_norm.template item<double>(), 1.0e-30);
        const double norm_ratio = got_norm.template item<double>() / denominator;
        const double reference_rms = ref.square().mean().sqrt().template item<double>();
        const double test_rms = got.square().mean().sqrt().template item<double>();
        const std::string stage = i == 0
            ? "embedding"
            : "block_" + std::to_string(i - 1);
        std::cout << "block_trace stage=" << stage
                  << " relative_l2=" << relative_l2
                  << " cosine=" << cosine
                  << " norm_ratio=" << norm_ratio
                  << " reference_rms=" << reference_rms
                  << " test_rms=" << test_rms << "\n";
    }

    auto reference_logits = reference.lm_head.forward(reference_hidden).to(mfq_tensor_backend::kFloat32);
    auto test_logits = test.lm_head.forward(test_hidden).to(mfq_tensor_backend::kFloat32);
    double kl_sum = 0.0;
    int64_t same_top = 0;
    int64_t rows = 0;
    const int64_t tokens = reference_logits.numel() / reference_logits.size(-1);
    auto ref2 = reference_logits.reshape({tokens, -1});
    auto got2 = test_logits.reshape({tokens, -1});
    for (int64_t start = 0; start < tokens; start += 8) {
        const int64_t end = std::min(start + 8, tokens);
        auto ref_chunk = ref2.index({Slice(start, end)});
        auto got_chunk = got2.index({Slice(start, end)});
        auto ref_logp = mfq_tensor_backend::log_softmax(ref_chunk, -1);
        auto got_logp = mfq_tensor_backend::log_softmax(got_chunk, -1);
        kl_sum += (ref_logp.exp() * (ref_logp - got_logp)).sum(-1)
            .to(mfq_tensor_backend::kFloat64).sum().template item<double>();
        same_top += ref_chunk.argmax(-1).eq(got_chunk.argmax(-1)).sum().template item<int64_t>();
        rows += end - start;
    }
    const double logits_relative_l2 =
        (test_logits.to(mfq_tensor_backend::kFloat64) - reference_logits.to(mfq_tensor_backend::kFloat64)).norm().template item<double>() /
        std::max(reference_logits.to(mfq_tensor_backend::kFloat64).norm().template item<double>(), 1.0e-30);
    std::cout << "block_trace_logits kld=" << (kl_sum / std::max<int64_t>(rows, 1))
              << " same_top=" << ((double)same_top / std::max<int64_t>(rows, 1))
              << " relative_l2=" << logits_relative_l2 << "\n";
    return 0;
}

template <typename Model>
static int run_block_trace_dump(
    Model& model,
    const std::string & output_dir,
    mfq_tensor_backend::Tensor ids,
    int64_t token_start,
    int64_t token_count)
{
    const int64_t total_tokens = ids.size(1);
    if (token_start < 0 || token_start >= total_tokens) {
        throw std::runtime_error("--dump-block-trace-start is outside the token range");
    }
    if (token_count <= 0) token_count = total_tokens - token_start;
    if (token_count > total_tokens - token_start) {
        throw std::runtime_error("--dump-block-trace-count exceeds the token range");
    }
    const std::filesystem::path root(output_dir);
    std::error_code error;
    if (!std::filesystem::create_directories(root, error) || error) {
        throw std::runtime_error(
            "block trace output directory must be new: " + output_dir);
    }

    model.reset(ids.size(0));
    std::vector<mfq_tensor_backend::Tensor> trace;
    auto final_hidden = model.hidden_forward(
        ids, mfq_nullopt, mfq_nullopt, &trace);
    mfq_cuda_synchronize();

    auto ids_cpu = ids.to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt32).contiguous();
    {
        std::ofstream output(root / "tokens.i32", std::ios::binary);
        if (!output) throw std::runtime_error("cannot create block trace token file");
        output.write(
            reinterpret_cast<const char *>(ids_cpu.template data_ptr<int32_t>()),
            static_cast<std::streamsize>(ids_cpu.nbytes()));
        if (!output) throw std::runtime_error("failed to write block trace tokens");
    }

    std::ofstream metadata(root / "trace_meta.jsonl");
    if (!metadata) throw std::runtime_error("cannot create block trace metadata");
    auto token_slice = [&](mfq_tensor_backend::Tensor value) {
        if (value.dim() >= 3 && value.size(1) == total_tokens) {
            return value.narrow(1, token_start, token_count);
        }
        return value;
    };
    for (size_t index = 0; index < trace.size(); ++index) {
        auto value = token_slice(trace[index])
            .to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kFloat32).contiguous();
        const std::string stage = index == 0
            ? "embedding"
            : "block_" + std::to_string(index - 1);
        const std::filesystem::path file = root / (stage + ".f32");
        std::ofstream output(file, std::ios::binary);
        if (!output) {
            throw std::runtime_error("cannot create block trace tensor: " + file.string());
        }
        output.write(
            reinterpret_cast<const char *>(value.template data_ptr<float>()),
            static_cast<std::streamsize>(value.nbytes()));
        if (!output) {
            throw std::runtime_error("failed to write block trace tensor: " + file.string());
        }
        metadata << "{\"stage\":\"" << stage << "\",\"file\":\""
                 << file.filename().string() << "\",\"shape\":[";
        for (int64_t dim = 0; dim < value.dim(); ++dim) {
            if (dim) metadata << ',';
            metadata << value.size(dim);
        }
        metadata << "],\"dtype\":\"float32\"}\n";
        std::cout << "block_trace_dump stage=" << stage
                  << " values=" << value.numel() << "\n";
    }
    auto dump_terminal = [&](const std::string & stage, mfq_tensor_backend::Tensor value) {
        value = value.to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kFloat32).contiguous();
        const std::filesystem::path file = root / (stage + ".f32");
        std::ofstream output(file, std::ios::binary);
        if (!output) {
            throw std::runtime_error("cannot create block trace tensor: " + file.string());
        }
        output.write(
            reinterpret_cast<const char *>(value.template data_ptr<float>()),
            static_cast<std::streamsize>(value.nbytes()));
        if (!output) {
            throw std::runtime_error("failed to write block trace tensor: " + file.string());
        }
        metadata << "{\"stage\":\"" << stage << "\",\"file\":\""
                 << file.filename().string() << "\",\"shape\":[";
        for (int64_t dim = 0; dim < value.dim(); ++dim) {
            if (dim) metadata << ',';
            metadata << value.size(dim);
        }
        metadata << "],\"dtype\":\"float32\"}\n";
        metadata.flush();
        std::cout << "block_trace_dump stage=" << stage
                  << " values=" << value.numel() << "\n";
    };
    final_hidden = token_slice(final_hidden);
    dump_terminal("final_norm", final_hidden);
    auto logits = model.apply_final_logit_softcap(
        model.lm_head.forward(final_hidden));
    dump_terminal("logits", logits);
    metadata.flush();
    if (!metadata) throw std::runtime_error("failed to write block trace metadata");
    return 0;
}

template <typename Model>
static int run_dsv4_hc_model_compare(
    Model& model,
    mfq_tensor_backend::Tensor ids)
{
    std::vector<mfq_tensor_backend::Tensor> reference_trace;
    std::vector<mfq_tensor_backend::Tensor> candidate_trace;
    std::vector<mfq_tensor_backend::Tensor> repeat_trace;

    g_dsv4_fused_hc = false;
    model.reset(ids.size(0));
    auto reference_hidden = model.hidden_forward(
        ids, mfq_nullopt, mfq_nullopt, &reference_trace);
    auto reference_logits =
        model.lm_head.forward(reference_hidden).to(mfq_tensor_backend::kFloat32);

    g_dsv4_fused_hc = true;
    model.reset(ids.size(0));
    auto candidate_hidden = model.hidden_forward(
        ids, mfq_nullopt, mfq_nullopt, &candidate_trace);
    auto candidate_logits =
        model.lm_head.forward(candidate_hidden).to(mfq_tensor_backend::kFloat32);

    g_dsv4_fused_hc = false;
    model.reset(ids.size(0));
    auto repeat_hidden = model.hidden_forward(
        ids, mfq_nullopt, mfq_nullopt, &repeat_trace);
    auto repeat_logits =
        model.lm_head.forward(repeat_hidden).to(mfq_tensor_backend::kFloat32);
    g_dsv4_fused_hc = true;
    mfq_cuda_synchronize();

    if (reference_trace.size() != candidate_trace.size() ||
            reference_trace.size() != repeat_trace.size()) {
        throw std::runtime_error(
            "DeepSeek V4 HC trace stage count mismatch");
    }
    for (size_t index = 0; index < reference_trace.size(); ++index) {
        auto reference = reference_trace[index].reshape({-1});
        auto candidate = candidate_trace[index].reshape({-1});
        auto repeat = repeat_trace[index].reshape({-1});
        auto reference_f64 = reference.to(mfq_tensor_backend::kFloat64);
        auto candidate_f64 = candidate.to(mfq_tensor_backend::kFloat64);
        const double denominator = std::max(
            reference_f64.norm().template item<double>(), 1.0e-30);
        const std::string stage = index == 0
            ? "embedding"
            : "block_" + std::to_string(index - 1);
        std::cout << std::scientific << std::setprecision(9)
                  << "dsv4_hc_model_trace stage=" << stage
                  << " differing="
                  << candidate.ne(reference).sum().template item<int64_t>()
                  << " rel_l2="
                  << (candidate_f64 - reference_f64)
                         .norm().template item<double>() / denominator
                  << " mean_abs="
                  << (candidate_f64 - reference_f64)
                         .abs().mean().template item<double>()
                  << " max_abs="
                  << (candidate_f64 - reference_f64)
                         .abs().max().template item<double>()
                  << " repeat_differing="
                  << repeat.ne(reference).sum().template item<int64_t>()
                  << "\n";
    }

    auto reference_logp = mfq_tensor_backend::log_softmax(reference_logits, -1);
    auto candidate_logp = mfq_tensor_backend::log_softmax(candidate_logits, -1);
    const double kld_candidate_reference = (
        candidate_logp.exp() *
        (candidate_logp - reference_logp))
        .sum(-1).mean().template item<double>();
    const double kld_reference_candidate = (
        reference_logp.exp() *
        (reference_logp - candidate_logp))
        .sum(-1).mean().template item<double>();
    auto logit_diff = (
        candidate_logits.to(mfq_tensor_backend::kFloat64) -
        reference_logits.to(mfq_tensor_backend::kFloat64));
    std::cout << std::scientific << std::setprecision(9)
              << "dsv4_hc_model_logits"
              << " mean_kld_candidate_reference="
              << kld_candidate_reference
              << " mean_kld_reference_candidate="
              << kld_reference_candidate
              << " relative_l2="
              << logit_diff.norm().template item<double>() /
                    std::max(
                        reference_logits.to(mfq_tensor_backend::kFloat64)
                            .norm().template item<double>(),
                        1.0e-30)
              << " mean_abs="
              << logit_diff.abs().mean().template item<double>()
              << " max_abs="
              << logit_diff.abs().max().template item<double>()
              << " same_top="
              << candidate_logits.argmax(-1)
                     .eq(reference_logits.argmax(-1))
                     .to(mfq_tensor_backend::kFloat32).mean().template item<double>()
              << " repeat_logits_equal="
              << (repeat_logits.equal(reference_logits) ? 1 : 0)
              << "\n";
    return 0;
}

static std::vector<MfqCudaStream> make_cuda_graph_compute_streams(
        const MfqCudaStream& primary_stream) {
    if (!model_parallel_enabled()) {
        return {primary_stream};
    }
    const auto & config = model_parallel_config();
    std::vector<MfqCudaStream> streams;
    streams.reserve(config.devices.size());
    for (const int device : config.devices) {
        streams.push_back(
            device == primary_stream.device_index()
                ? primary_stream
                : mfq_get_stream_from_pool(false, device));
    }
    return streams;
}

static std::vector<MfqCudaStream> cuda_graph_participant_streams(
        const std::vector<MfqCudaStream>& compute_streams) {
    auto participants = compute_streams;
    participants.insert(
        participants.end(),
        g_model_parallel_collectives.streams.begin(),
        g_model_parallel_collectives.streams.end());
    return participants;
}

static std::vector<std::unique_ptr<MfqCudaStreamGuard>>
activate_cuda_graph_compute_streams(
        const std::vector<MfqCudaStream>& compute_streams) {
    std::vector<std::unique_ptr<MfqCudaStreamGuard>> guards;
    guards.reserve(compute_streams.size());
    for (const auto& stream : compute_streams) {
        guards.push_back(
            std::make_unique<MfqCudaStreamGuard>(stream));
    }
    return guards;
}

struct DecodeGraphCache {
    decltype(mfq_get_stream_from_pool(false)) stream;
    std::vector<MfqCudaStream> compute_streams;
    std::unique_ptr<MfqCudaGraph> graph;
    mfq_tensor_backend::Tensor static_input;
    mfq_tensor_backend::Tensor static_pos;
    mfq_tensor_backend::Tensor static_len;
    mfq_tensor_backend::Tensor static_step;
    mfq_tensor_backend::Tensor generated;
    mfq_tensor_backend::Tensor random;
    mfq_tensor_backend::Tensor counts;
    mfq_tensor_backend::Tensor static_next;
    int64_t generated_capacity = 0;
    int64_t planned_len = 0;
    bool greedy = false;
    double temperature = 0.0;
    int32_t top_k = 0;
    double top_p = 1.0;
    double presence_penalty = 0.0;
    double frequency_penalty = 0.0;
    double repetition_penalty = 1.0;
    uint64_t captures = 0;
    uint64_t reuses = 0;
    bool valid = false;

    explicit DecodeGraphCache(int64_t context_capacity)
        : stream(mfq_get_stream_from_pool(false)),
          generated_capacity(std::max<int64_t>(context_capacity, 2048)) {}

    void ensure_compute_streams() {
        if (!compute_streams.empty()) return;
        compute_streams =
            make_cuda_graph_compute_streams(stream);
    }

    std::vector<MfqCudaStream> graph_participant_streams() const {
        return cuda_graph_participant_streams(compute_streams);
    }

    bool matches(int64_t candidate_len, const MfqSamplingParams & sampling,
                 bool candidate_greedy) const {
        return valid && planned_len == candidate_len && greedy == candidate_greedy &&
               (candidate_greedy ||
                (temperature == sampling.temperature && top_k == sampling.top_k &&
                 top_p == sampling.top_p)) &&
               presence_penalty == sampling.presence_penalty &&
               frequency_penalty == sampling.frequency_penalty &&
               repetition_penalty == sampling.repetition_penalty;
    }

    void ensure_storage(int64_t vocab_size) {
        auto i64 = mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64).device(mfq_tensor_backend::kCUDA);
        if (!static_input.defined()) static_input = mfq_tensor_backend::empty({1, 1}, i64);
        if (!static_pos.defined()) static_pos = mfq_tensor_backend::empty({1}, i64);
        if (!static_len.defined()) static_len = mfq_tensor_backend::empty({1}, i64);
        if (!static_step.defined()) static_step = mfq_tensor_backend::empty({1}, i64);
        if (!generated.defined()) generated = mfq_tensor_backend::empty({generated_capacity}, i64);
        if (!random.defined()) {
            random = mfq_tensor_backend::empty(
                {1}, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kFloat32).device(mfq_tensor_backend::kCUDA));
        }
        if (!counts.defined()) {
            counts = mfq_tensor_backend::empty(
                {vocab_size}, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt32).device(mfq_tensor_backend::kCUDA));
        }
    }

    void invalidate() {
        if (graph) graph->reset();
        graph.reset();
        static_next = mfq_tensor_backend::Tensor();
        valid = false;
    }

    void set_key(int64_t candidate_len, const MfqSamplingParams & sampling,
                 bool candidate_greedy) {
        planned_len = candidate_len;
        greedy = candidate_greedy;
        temperature = sampling.temperature;
        top_k = sampling.top_k;
        top_p = sampling.top_p;
        presence_penalty = sampling.presence_penalty;
        frequency_penalty = sampling.frequency_penalty;
        repetition_penalty = sampling.repetition_penalty;
        valid = true;
    }
};

template <typename Model>
static void prepare_decode_graph_memory(Model& model, MfqCudaGraph& graph,
        const std::function<void()>& warmup,
        const std::vector<MfqCudaStream>& participant_streams = {}) {
    using Tensor = mfq_tensor_backend::Tensor;
    struct SavedRecurrentState {
        mfq::cuda::qwen35::LinearAttentionBlock* block;
        Tensor conv, gdn;
        const void* conv_address;
        const void* gdn_address;
    };
    // Snapshot before entering the private graph allocator. These copies are
    // temporary and must not become retained allocations in the captured pool.
    std::vector<SavedRecurrentState> saved;
    for (const auto& block : model.blocks) {
        if (auto* linear = dynamic_cast<
                mfq::cuda::qwen35::LinearAttentionBlock*>(block.get())) {
            MFQ_RUNTIME_CHECK(!linear->speculative_pending && linear->conv_state.defined() &&
                linear->gdn_state.defined(), "decode warmup requires confirmed recurrent state");
            saved.push_back({linear, linear->conv_state.clone(), linear->gdn_state.clone(),
                linear->conv_state.data_ptr(), linear->gdn_state.data_ptr()});
        }
    }
    const int64_t saved_position = model.cache_pos;
    auto restore = [&]() {
        for (const auto& state : saved) {
            MFQ_RUNTIME_CHECK(state.block->conv_state.data_ptr() == state.conv_address &&
                state.block->gdn_state.data_ptr() == state.gdn_address,
                "decode warmup changed recurrent storage addresses");
            state.block->conv_state.copy_(state.conv);
            state.block->gdn_state.copy_(state.gdn);
        }
        MFQ_RUNTIME_CHECK(model.cache_pos == saved_position, "static decode warmup changed cache position");
        if (participant_streams.empty()) {
            MFQ_CUDA_CHECK(cudaStreamSynchronize(
                mfq_get_current_cuda_stream().stream()));
            return;
        }
        for (const auto& participant : participant_streams) {
            MfqCudaGuard guard(participant.device_index());
            MFQ_CUDA_CHECK(cudaStreamSynchronize(
                participant.stream()));
        }
    };
    mfq_prepare_cuda_graph_memory(graph, participant_streams);
    if (!saved.empty() || Model::is_gemma4 || (Model::backbone == mfq::cuda::CudaBackbone::glm_dsa) || Model::is_minicpmo45) {
        // The first pass may initialize persistent CUDA/NCCL workspace state.
        // A second pass then records the complete set of reusable temporaries
        // needed by capture after those persistent allocations exist.
        for (int pass = 0; pass < 2; ++pass) {
            try {
                // Replay overwrites this same, still-unconfirmed KV position.
                warmup();
                restore();
            } catch (...) {
                restore();
                throw;
            }
        }
    }
}

static int64_t decode_graph_bucket(int64_t planned_len, int64_t context_capacity) {
    const int64_t quantum = planned_len <= 4096 ? 512 :
        (planned_len <= 16384 ? 1024 : 2048);
    const int64_t rounded = ((planned_len + quantum - 1) / quantum) * quantum;
    return std::min<int64_t>(rounded, context_capacity);
}

static bool trace_cuda_graph() {
    const char * value = std::getenv("MFQ_RUNTIME_TRACE_CUDA_GRAPH");
    return value != nullptr && std::atoi(value) != 0;
}

template <typename Model>
static mfq_tensor_backend::Tensor sample_token(
    Model& model,
    mfq_tensor_backend::Tensor ids,
    mfq::cuda::Sampler& sampler,
    mfq_tensor_backend::Tensor counts,
    const MfqTokenConstraintPtr & token_constraint,
    cudaEvent_t prefill_finished = nullptr)
{
    if (sampler.greedy() && !sampler.has_penalties() && !token_constraint) {
        auto next = model.next_token(ids);
        if (prefill_finished != nullptr) {
            MFQ_CUDA_CHECK(cudaEventRecord(
                prefill_finished, mfq_get_current_cuda_stream()));
        }
        return next;
    }

    auto logits = model.last_logits(ids).contiguous().view({1, -1});
    if (prefill_finished != nullptr) {
        MFQ_CUDA_CHECK(cudaEventRecord(
            prefill_finished, mfq_get_current_cuda_stream()));
    }
    return mfq::cuda::sample_logits(
        sampler, std::move(logits), counts, token_constraint);
}

template <typename Model>
static mfq_tensor_backend::Tensor prefill_tail(
    Model& model,
    mfq_tensor_backend::Tensor ids,
    int64_t chunk_size) {
    MFQ_RUNTIME_CHECK(
        chunk_size > 0,
        "runtime prefill chunk size must be positive");
    MFQ_RUNTIME_CHECK(
        ids.dim() == 2 && ids.size(0) == 1 && ids.size(1) > 0,
        "runtime prefill IDs must have shape [1, tokens]");
    int64_t offset = 0;
    while (ids.size(1) - offset > chunk_size) {
        (void)model.hidden_forward(
            ids.narrow(1, offset, chunk_size).contiguous());
        offset += chunk_size;
    }
    return offset == 0
        ? ids
        : ids.narrow(1, offset, ids.size(1) - offset).contiguous();
}

template <typename Model>
static mfq_tensor_backend::Tensor hidden_forward_chunked(
    Model& model,
    const mfq_tensor_backend::Tensor & ids,
    int64_t chunk_size,
    mfq_tensor_backend::Tensor * raw_hidden = nullptr) {
    MFQ_RUNTIME_CHECK(
        chunk_size > 0,
        "runtime prefill chunk size must be positive");
    MFQ_RUNTIME_CHECK(
        ids.dim() == 2 && ids.size(0) == 1 && ids.size(1) > 0,
        "runtime prefill IDs must have shape [1, tokens]");
    std::vector<mfq_tensor_backend::Tensor> raw_chunks;
    if (raw_hidden != nullptr) {
        raw_chunks.reserve(static_cast<std::size_t>(
            (ids.size(1) + chunk_size - 1) / chunk_size));
    }
    mfq_tensor_backend::Tensor hidden;
    for (int64_t offset = 0; offset < ids.size(1); offset += chunk_size) {
        const int64_t count = std::min(chunk_size, ids.size(1) - offset);
        mfq_tensor_backend::Tensor raw_chunk;
        hidden = model.hidden_forward(
            ids.narrow(1, offset, count).contiguous(),
            mfq_nullopt,
            mfq_nullopt,
            nullptr,
            mfq_nullopt,
            raw_hidden != nullptr ? &raw_chunk : nullptr);
        if (raw_hidden != nullptr) {
            raw_chunks.push_back(std::move(raw_chunk));
        }
    }
    if (raw_hidden != nullptr) {
        *raw_hidden = raw_chunks.size() == 1
            ? std::move(raw_chunks.front())
            : mfq_tensor_backend::cat(raw_chunks, 1).contiguous();
    }
    return hidden;
}

template <typename Model>
static mfq_tensor_backend::Tensor hidden_forward_prepared_chunked(
    Model& model,
    const mfq_tensor_backend::Tensor& ids,
    const CudaPreparedPrompt& prepared,
    int64_t chunk_size,
    mfq_tensor_backend::Tensor* raw_hidden = nullptr) {
    MFQ_RUNTIME_CHECK(
        chunk_size > 0 && prepared.transformed() &&
            ids.dim() == 2 && ids.size(0) == 1 && ids.size(1) > 0 &&
            prepared.embeddings.defined() && prepared.positions.defined() &&
            prepared.embeddings.dim() == 3 &&
            prepared.embeddings.size(0) == 1 &&
            prepared.embeddings.size(1) == ids.size(1) &&
            prepared.embeddings.size(2) == model.hidden_size() &&
            (prepared.positions.dim() == 1 ||
             prepared.positions.dim() == 2) &&
            prepared.positions.size(-1) == ids.size(1),
        "prepared CUDA prefill tensors disagree with prompt geometry");
    std::vector<mfq_tensor_backend::Tensor> raw_chunks;
    if (raw_hidden != nullptr) {
        raw_chunks.reserve(static_cast<std::size_t>(
            (ids.size(1) + chunk_size - 1) / chunk_size));
    }
    mfq_tensor_backend::Tensor hidden;
    for (int64_t offset = 0; offset < ids.size(1); offset += chunk_size) {
        const int64_t count = std::min(chunk_size, ids.size(1) - offset);
        mfq_tensor_backend::Tensor raw_chunk;
        hidden = model.hidden_forward_inputs(
            ids.narrow(1, offset, count).contiguous(),
            prepared.embeddings.narrow(1, offset, count).contiguous(),
            prepared.positions.narrow(-1, offset, count).contiguous(),
            mfq_nullopt, nullptr, mfq_nullopt, true, mfq_nullopt,
            raw_hidden != nullptr ? &raw_chunk : nullptr);
        if (raw_hidden != nullptr) raw_chunks.push_back(std::move(raw_chunk));
    }
    model.decode_position_delta = prepared.decode_position_delta;
    if (raw_hidden != nullptr) {
        *raw_hidden = raw_chunks.size() == 1
            ? std::move(raw_chunks.front())
            : mfq_tensor_backend::cat(raw_chunks, 1).contiguous();
    }
    return hidden;
}

class PrefillCudaTimer {
public:
    PrefillCudaTimer()
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

    ~PrefillCudaTimer() {
        if (finished_ != nullptr) cudaEventDestroy(finished_);
        if (started_ != nullptr) cudaEventDestroy(started_);
    }

    cudaEvent_t finished_event() const {
        return finished_;
    }

    double elapsed_ms() const {
        MFQ_CUDA_CHECK(cudaEventSynchronize(finished_));
        float elapsed = 0.0f;
        MFQ_CUDA_CHECK(cudaEventElapsedTime(&elapsed, started_, finished_));
        return static_cast<double>(elapsed);
    }

private:
    cudaStream_t stream_ = nullptr;
    cudaEvent_t started_ = nullptr;
    cudaEvent_t finished_ = nullptr;
};

static uint64_t cuda_cache_environment_bytes(
        const char * name, uint64_t fallback) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') return fallback;
    char * end = nullptr;
    const auto parsed = std::strtoull(value, &end, 10);
    if (end == value || *end != '\0') {
        throw std::runtime_error(std::string("invalid ") + name);
    }
    return parsed;
}

static std::filesystem::path default_cuda_prefix_cache_directory() {
    if (const char * configured =
            std::getenv("MFQ_RUNTIME_PREFIX_CACHE_DIR")) {
        if (configured[0] != '\0') return configured;
    }
#ifdef _WIN32
    if (const char * local = std::getenv("LOCALAPPDATA")) {
        if (local[0] != '\0') {
            return std::filesystem::path(local) /
                "TyloQuant" / "MFQ" / "prefix-cache";
        }
    }
#else
    if (const char * xdg = std::getenv("XDG_CACHE_HOME")) {
        if (xdg[0] != '\0') {
            return std::filesystem::path(xdg) /
                "tyloquant" / "mfq" / "prefix-cache";
        }
    }
    if (const char * home = std::getenv("HOME")) {
        if (home[0] != '\0') {
            return std::filesystem::path(home) / ".cache" /
                "tyloquant" / "mfq" / "prefix-cache";
        }
    }
#endif
    return std::filesystem::temp_directory_path() /
        "tyloquant-mfq-prefix-cache";
}

template <typename Model>
static std::string cuda_prefix_cache_compatibility_key(
        const mfq::ModelSource& source, const Model& model) {
    std::ostringstream key;
    key << "mfq-cuda-prefix-v1\n"
        << "codec=cuda-full-attention-kv-v1\n"
        << "context=" << model.max_position_embeddings() << '\n'
        << "architecture=" << source.architecture() << '\n';
    for (std::size_t index = 0; index < source.source_paths().size(); ++index) {
        const auto& path = source.source_paths()[index];
        std::error_code error;
        const auto size = std::filesystem::file_size(path, error);
        if (error) {
            throw std::runtime_error(
                "cannot identify model source for prefix cache: " +
                error.message());
        }
        const auto modified = std::filesystem::last_write_time(path, error);
        if (error) {
            throw std::runtime_error(
                "cannot identify model source timestamp: " +
                error.message());
        }
        const auto modified_ns = std::chrono::duration_cast<
            std::chrono::nanoseconds>(modified.time_since_epoch()).count();
        key << "source=" << index << ':' << path.filename().string() << ':'
            << size << ':' << modified_ns << '\n';
    }
    for (const auto& tensor : source.tensors()) {
        key << "tensor=" << tensor.name << ':' << tensor.dtype << ':'
            << tensor.nbytes << '\n';
    }
    return key.str();
}

template <typename Model>
static std::shared_ptr<mfq::cache::PagedPrefixCache>
make_cuda_paged_prefix_cache(
        const mfq::ModelSource& source, const Model& model) {
    const auto format = source.metadata().find("source.format");
    // ponytail: HF source fingerprints exclude config sidecars for now.
    if ((format != source.metadata().end() &&
         format->second == "hf-safetensors") ||
        !model.supports_paged_text_session_state()) {
        return {};
    }
    if (const char * disabled =
            std::getenv("MFQ_RUNTIME_DISABLE_PREFIX_CACHE")) {
        if (disabled[0] == '1') return {};
    }
    const auto block_size = cuda_cache_environment_bytes(
        "MFQ_RUNTIME_PREFIX_CACHE_BLOCK_TOKENS", 256);
    if (block_size == 0 || block_size > 65536) {
        throw std::runtime_error(
            "MFQ_RUNTIME_PREFIX_CACHE_BLOCK_TOKENS must be in [1, 65536]");
    }
    mfq::cache::PagedPrefixCacheConfig config;
    config.cache_dir = default_cuda_prefix_cache_directory();
    config.compatibility_key =
        cuda_prefix_cache_compatibility_key(source, model);
    config.block_size_tokens = static_cast<size_t>(block_size);
    config.max_disk_bytes = cuda_cache_environment_bytes(
        "MFQ_RUNTIME_PREFIX_CACHE_DISK_BYTES",
        100ULL * 1024ULL * 1024ULL * 1024ULL);
    config.max_hot_bytes = cuda_cache_environment_bytes(
        "MFQ_RUNTIME_PREFIX_CACHE_HOT_BYTES",
        2ULL * 1024ULL * 1024ULL * 1024ULL);
    config.max_pending_writes = static_cast<size_t>(
        cuda_cache_environment_bytes(
            "MFQ_RUNTIME_PREFIX_CACHE_PENDING_WRITES", 64));
    config.max_pending_bytes = cuda_cache_environment_bytes(
        "MFQ_RUNTIME_PREFIX_CACHE_PENDING_BYTES",
        512ULL * 1024ULL * 1024ULL);
    return std::make_shared<mfq::cache::PagedPrefixCache>(
        std::move(config));
}

class TextSessionCache {
private:
    struct PagedBinding {
        std::vector<mfq::cache::BlockHash> blocks;
        size_t tokens = 0;
        uint64_t last_used = 0;
    };

public:
    explicit TextSessionCache(
            std::shared_ptr<mfq::cache::PagedPrefixCache> paged_cache = {})
        : paged_cache_(std::move(paged_cache)) {
        const char * entries =
            std::getenv("MFQ_RUNTIME_MAX_KV_SESSIONS");
        if (entries != nullptr) {
            max_sessions_ = static_cast<size_t>(std::strtoull(
                entries, nullptr, 10));
        }
        const char * snapshots =
            std::getenv("MFQ_RUNTIME_MAX_KV_SNAPSHOTS_PER_SESSION");
        if (snapshots != nullptr) {
            max_snapshots_per_session_ = static_cast<size_t>(std::strtoull(
                snapshots, nullptr, 10));
        }
        const char * bytes =
            std::getenv("MFQ_RUNTIME_KV_SESSION_BYTES");
        if (bytes != nullptr) {
            max_bytes_ = static_cast<size_t>(std::strtoull(
                bytes, nullptr, 10));
        }
        const char * trace =
            std::getenv("MFQ_RUNTIME_TRACE_SESSION_CACHE");
        trace_ = trace != nullptr && trace[0] == '1';
        if (paged_cache_) {
            paged_disk_budget_ = cuda_cache_environment_bytes(
                "MFQ_RUNTIME_PREFIX_CACHE_DISK_BYTES",
                100ULL * 1024ULL * 1024ULL * 1024ULL);
            paged_hot_budget_ = cuda_cache_environment_bytes(
                "MFQ_RUNTIME_PREFIX_CACHE_HOT_BYTES",
                2ULL * 1024ULL * 1024ULL * 1024ULL);
        }
    }

    bool persistent_prefix_enabled() const noexcept {
        return static_cast<bool>(paged_cache_);
    }

    template <typename Model>
    size_t restore_best(
            Model& model,
            const std::string & requested_session,
            const std::vector<int64_t> & prompt,
            size_t maximum_prefix_tokens) {
        if (paged_cache_) {
            return restore_paged(
                model,
                requested_session,
                prompt,
                maximum_prefix_tokens);
        }
        if (requested_session.empty() || max_sessions_ == 0 ||
                max_snapshots_per_session_ == 0 ||
                max_bytes_ == 0 || !model.supports_text_session_state()) {
            return 0;
        }
        ++queries_;
        std::string selected_session;
        size_t selected_snapshot = 0;
        size_t selected_tokens = 0;
        for (const auto & [session_id, history] : states_) {
            for (size_t index = 0; index < history.size(); ++index) {
                const auto & tokens = history[index].tokens;
                if (tokens.empty() || tokens.size() >= prompt.size() ||
                        tokens.size() > maximum_prefix_tokens ||
                        tokens.size() < selected_tokens ||
                        !std::equal(
                            tokens.begin(), tokens.end(), prompt.begin())) {
                    continue;
                }
                const bool requested_tie =
                    tokens.size() == selected_tokens &&
                    session_id == requested_session &&
                    selected_session != requested_session;
                if (tokens.size() > selected_tokens || requested_tie) {
                    selected_session = session_id;
                    selected_snapshot = index;
                    selected_tokens = tokens.size();
                }
            }
        }
        if (selected_session.empty()) return 0;
        auto & selected = states_.at(selected_session)[selected_snapshot];
        try {
            model.restore_text_session_state(selected);
            selected.last_used = ++clock_;
            ++hits_;
            hit_tokens_ += selected_tokens;
            if (trace_) {
                std::cerr << "runtime_session_cache action=hit session="
                          << requested_session
                          << " source=" << selected_session
                          << " reused_tokens=" << selected_tokens
                          << " prefill_tokens="
                          << prompt.size() - selected_tokens << std::endl;
            }
            return selected_tokens;
        } catch (const std::exception & error) {
            erase_snapshot(selected_session, selected_snapshot, "invalidate");
            model.reset(1);
            std::cerr << "runtime_session_cache action=invalidate session="
                      << selected_session << " error=" << error.what()
                      << std::endl;
            return 0;
        }
    }

    void store(
            const std::string & session_id,
            TextSessionState state) {
        if (paged_cache_) {
            store_paged(session_id, state);
            return;
        }
        if (session_id.empty() || max_sessions_ == 0 ||
                max_snapshots_per_session_ == 0 || max_bytes_ == 0) {
            return;
        }
        if (state.bytes > max_bytes_) {
            if (trace_) {
                std::cerr << "runtime_session_cache action=skip session="
                          << session_id << " bytes=" << state.bytes
                          << " budget=" << max_bytes_ << std::endl;
            }
            return;
        }
        state.last_used = ++clock_;
        const uint64_t protected_clock = state.last_used;
        auto & history = states_[session_id];
        auto previous = std::find_if(
            history.begin(), history.end(),
            [&](const TextSessionState & saved) {
                return saved.tokens == state.tokens;
            });
        if (previous != history.end()) {
            bytes_ -= previous->bytes;
            *previous = std::move(state);
        } else {
            history.push_back(std::move(state));
        }
        const auto stored = std::find_if(
            history.begin(), history.end(),
            [&](const TextSessionState & saved) {
                return saved.last_used == protected_clock;
            });
        if (stored == history.end()) {
            throw std::runtime_error("stored session snapshot is unavailable");
        }
        bytes_ += stored->bytes;
        evict_history_to_limit(session_id, protected_clock);
        evict_to_budget(session_id, protected_clock);
        sync_telemetry();
        if (trace_) {
            const auto & saved_history = states_.at(session_id);
            const auto saved = std::find_if(
                saved_history.begin(), saved_history.end(),
                [&](const TextSessionState & candidate) {
                    return candidate.last_used == protected_clock;
                });
            if (saved == saved_history.end()) {
                throw std::runtime_error(
                    "protected session snapshot was evicted");
            }
            std::cerr << "runtime_session_cache action=store session="
                      << session_id << " tokens=" << saved->tokens.size()
                      << " bytes=" << saved->bytes
                      << " snapshots=" << saved_history.size()
                      << " total_bytes=" << bytes_ << std::endl;
        }
    }

    size_t fork_session(
            const std::string & source_session,
            const std::string & target_session) {
        if (paged_cache_) {
            const auto source = paged_bindings_.find(source_session);
            if (source == paged_bindings_.end() ||
                    source_session.empty() || target_session.empty() ||
                    source_session == target_session) {
                return 0;
            }
            bind_paged_session(
                target_session,
                source->second.blocks,
                source->second.tokens);
            return source->second.blocks.size();
        }
        if (source_session.empty() || target_session.empty() ||
                source_session == target_session || max_sessions_ == 0 ||
                max_snapshots_per_session_ == 0 || max_bytes_ == 0) {
            return 0;
        }
        const auto source = states_.find(source_session);
        if (source == states_.end()) return 0;
        std::vector<TextSessionState> copied = source->second;
        close_session(target_session);
        auto & target = states_[target_session];
        uint64_t protected_clock = 0;
        for (auto & snapshot : copied) {
            snapshot.last_used = ++clock_;
            protected_clock = snapshot.last_used;
            bytes_ += snapshot.bytes;
            target.push_back(std::move(snapshot));
        }
        evict_history_to_limit(target_session, protected_clock);
        evict_to_budget(target_session, protected_clock);
        const auto remaining = states_.find(target_session);
        const size_t copied_snapshots = remaining == states_.end()
            ? 0 : remaining->second.size();
        sync_telemetry();
        if (trace_) {
            std::cerr << "runtime_session_cache action=fork source="
                      << source_session << " target=" << target_session
                      << " snapshots=" << copied_snapshots
                      << " total_bytes=" << bytes_ << std::endl;
        }
        return copied_snapshots;
    }

    size_t close_session(const std::string & session_id) {
        if (paged_cache_) return close_paged_session(session_id);
        auto found = states_.find(session_id);
        if (found == states_.end()) return 0;
        const size_t released = found->second.size();
        size_t released_bytes = 0;
        for (const auto & snapshot : found->second) {
            released_bytes += snapshot.bytes;
        }
        bytes_ -= released_bytes;
        states_.erase(found);
        sync_telemetry();
        if (trace_) {
            std::cerr << "runtime_session_cache action=close session="
                      << session_id << " snapshots=" << released
                      << " bytes=" << released_bytes
                      << " total_bytes=" << bytes_ << std::endl;
        }
        return released;
    }

    std::vector<std::pair<std::string, double>> metrics() const {
        if (paged_cache_) {
            const auto value = paged_cache_->metrics();
            return {
                {"prefix_cache_queries", static_cast<double>(value.queries)},
                {"prefix_cache_hits", static_cast<double>(value.hits)},
                {"prefix_cache_hit_tokens", static_cast<double>(value.hit_tokens)},
                {"prefix_cache_sessions", static_cast<double>(metric_sessions_.load())},
                {"prefix_cache_snapshots", static_cast<double>(value.disk_blocks)},
                {"prefix_cache_tokens", static_cast<double>(metric_tokens_.load())},
                {"prefix_cache_bytes", static_cast<double>(value.hot_bytes)},
                {"prefix_cache_max_sessions", static_cast<double>(max_sessions_)},
                {"prefix_cache_max_snapshots_per_session", 1.0},
                {"prefix_cache_max_bytes", static_cast<double>(paged_hot_budget_)},
                {"prefix_cache_disk_blocks", static_cast<double>(value.disk_blocks)},
                {"prefix_cache_disk_bytes", static_cast<double>(value.disk_bytes)},
                {"prefix_cache_disk_max_bytes", static_cast<double>(paged_disk_budget_)},
                {"prefix_cache_hot_blocks", static_cast<double>(value.hot_blocks)},
                {"prefix_cache_hot_bytes", static_cast<double>(value.hot_bytes)},
                {"prefix_cache_pending_writes", static_cast<double>(value.pending_writes)},
                {"prefix_cache_pending_bytes", static_cast<double>(value.pending_bytes)},
                {"prefix_cache_pending_max_bytes", static_cast<double>(value.pending_max_bytes)},
                {"prefix_cache_writes", static_cast<double>(value.writes)},
                {"prefix_cache_deduplicated_writes", static_cast<double>(value.deduplicated_writes)},
                {"prefix_cache_disk_hits", static_cast<double>(value.disk_hits)},
                {"prefix_cache_hot_hits", static_cast<double>(value.hot_hits)},
                {"prefix_cache_evictions", static_cast<double>(value.evictions)},
                {"prefix_cache_corrupt_blocks", static_cast<double>(value.corrupt_blocks)},
            };
        }
        return {
            {"prefix_cache_queries", static_cast<double>(queries_.load())},
            {"prefix_cache_hits", static_cast<double>(hits_.load())},
            {"prefix_cache_hit_tokens", static_cast<double>(hit_tokens_.load())},
            {"prefix_cache_sessions", static_cast<double>(metric_sessions_.load())},
            {"prefix_cache_snapshots", static_cast<double>(metric_snapshots_.load())},
            {"prefix_cache_tokens", static_cast<double>(metric_tokens_.load())},
            {"prefix_cache_bytes", static_cast<double>(metric_bytes_.load())},
            {"prefix_cache_max_sessions", static_cast<double>(max_sessions_)},
            {"prefix_cache_max_snapshots_per_session",
                static_cast<double>(max_snapshots_per_session_)},
            {"prefix_cache_max_bytes", static_cast<double>(max_bytes_)},
        };
    }

    size_t clear_live_sessions() noexcept {
        if (paged_cache_) {
            const auto sessions = paged_bindings_.size();
            for (const auto & [session, binding] : paged_bindings_) {
                (void)session;
                paged_cache_->unpin(binding.blocks);
            }
            paged_bindings_.clear();
            sync_paged_telemetry();
            return sessions;
        }
        size_t snapshots = 0;
        for (const auto & [session_id, history] : states_) {
            (void)session_id;
            snapshots += history.size();
        }
        states_.clear();
        bytes_ = 0;
        sync_telemetry();
        return snapshots;
    }

    size_t clear() {
        if (paged_cache_) {
            clear_live_sessions();
            return paged_cache_->clear();
        }
        return clear_live_sessions();
    }

    uint64_t trim_hot(uint64_t target_bytes) {
        if (!paged_cache_) return 0;
        const auto released = paged_cache_->trim_hot(target_bytes);
        if (released > 0) mfq_release_host_allocator_cache();
        return released;
    }

private:
    template <typename Model>
    size_t restore_paged(
            Model& model,
            const std::string & requested_session,
            const std::vector<int64_t> & prompt,
            size_t maximum_prefix_tokens) {
        if (max_sessions_ == 0 || !model.supports_paged_text_session_state() ||
                prompt.size() < 2) {
            return 0;
        }
        const auto limit = std::min(
            maximum_prefix_tokens, prompt.size() - 1);
        std::vector<int64_t> candidate(
            prompt.begin(),
            prompt.begin() + static_cast<std::ptrdiff_t>(limit));
        auto match = paged_cache_->match(candidate, {}, false);
        if (match.matched_tokens == 0) {
            paged_cache_->record_match(0);
            return 0;
        }

        auto payloads = paged_cache_->load_prefix(match.blocks);
        if (payloads.empty()) {
            paged_cache_->record_match(0);
            return 0;
        }
        if (payloads.size() != match.blocks.size()) {
            match.blocks.resize(payloads.size());
            match.matched_tokens =
                payloads.size() * paged_cache_->block_size_tokens();
        }
        std::vector<int64_t> matched_tokens(
            prompt.begin(),
            prompt.begin() + static_cast<std::ptrdiff_t>(
                match.matched_tokens));
        std::optional<TextSessionState> state;
        try {
            state.emplace(decode_cuda_paged_session(
                payloads,
                matched_tokens,
                paged_cache_->block_size_tokens()));
        } catch (const std::exception & error) {
            if (!match.blocks.empty()) {
                paged_cache_->invalidate(match.blocks.back());
            }
            paged_cache_->record_match(0);
            model.reset(1);
            std::cerr
                << "runtime_session_cache backend=cuda "
                << "action=paged_codec_invalidate "
                << "session=" << requested_session
                << " error=" << error.what() << std::endl;
            return 0;
        }
        try {
            model.restore_text_session_state(*state);
            if (!requested_session.empty()) {
                bind_paged_session(
                    requested_session, match.blocks, match.matched_tokens);
            }
            paged_cache_->record_match(match.matched_tokens);
            if (trace_) {
                std::cerr
                    << "runtime_session_cache backend=cuda action=paged_hit "
                    << "session=" << requested_session
                    << " reused_tokens=" << match.matched_tokens
                    << " prefill_tokens="
                    << prompt.size() - match.matched_tokens << std::endl;
            }
            return match.matched_tokens;
        } catch (const std::exception & error) {
            paged_cache_->record_match(0);
            model.reset(1);
            std::cerr
                << "runtime_session_cache backend=cuda "
                << "action=paged_restore_failed "
                << "session=" << requested_session
                << " error=" << error.what() << std::endl;
            return 0;
        }
    }

    void store_paged(
            const std::string & session_id,
            const TextSessionState & state) {
        if (max_sessions_ == 0 || state.tokens.empty()) {
            return;
        }
        const auto block_size = paged_cache_->block_size_tokens();
        const auto full_blocks = state.tokens.size() / block_size;
        if (full_blocks == 0) return;

        auto existing = paged_cache_->match(state.tokens, {}, false);
        if (existing.blocks.size() > full_blocks) {
            throw std::runtime_error(
                "paged prefix match exceeds the CUDA session state");
        }
        auto blocks = std::move(existing.blocks);
        mfq::cache::BlockHash parent{};
        if (!blocks.empty()) parent = blocks.back();
        for (size_t index = blocks.size(); index < full_blocks; ++index) {
            const auto token_offset = index * block_size;
            auto payload = encode_cuda_paged_block(
                state, block_size, index);
            parent = paged_cache_->store(
                parent,
                state.tokens.data() + token_offset,
                block_size,
                std::move(payload));
            blocks.push_back(parent);
        }
        if (!session_id.empty()) {
            bind_paged_session(
                session_id,
                std::move(blocks),
                full_blocks * block_size);
        }
        if (trace_) {
            std::cerr
                << "runtime_session_cache backend=cuda action=paged_store "
                << "session=" << session_id
                << " tokens=" << full_blocks * block_size
                << " blocks=" << full_blocks << std::endl;
        }
    }

    void bind_paged_session(
            const std::string & session_id,
            std::vector<mfq::cache::BlockHash> blocks,
            size_t tokens) {
        close_paged_session(session_id);
        paged_cache_->pin(blocks);
        paged_bindings_[session_id] = PagedBinding{
            std::move(blocks), tokens, ++clock_};
        while (paged_bindings_.size() > max_sessions_) {
            auto victim = paged_bindings_.end();
            for (auto iterator = paged_bindings_.begin();
                    iterator != paged_bindings_.end(); ++iterator) {
                if (iterator->first == session_id) continue;
                if (victim == paged_bindings_.end() ||
                        iterator->second.last_used <
                            victim->second.last_used) {
                    victim = iterator;
                }
            }
            if (victim == paged_bindings_.end()) break;
            close_paged_session(victim->first);
        }
        sync_paged_telemetry();
    }

    size_t close_paged_session(const std::string & session_id) {
        auto found = paged_bindings_.find(session_id);
        if (found == paged_bindings_.end()) return 0;
        const auto blocks = found->second.blocks.size();
        paged_cache_->unpin(found->second.blocks);
        paged_bindings_.erase(found);
        sync_paged_telemetry();
        return blocks;
    }

    void sync_paged_telemetry() noexcept {
        size_t tokens = 0;
        for (const auto & [session, binding] : paged_bindings_) {
            (void)session;
            tokens += binding.tokens;
        }
        metric_sessions_.store(paged_bindings_.size());
        metric_tokens_.store(tokens);
    }

    void sync_telemetry() noexcept {
        size_t snapshots = 0;
        size_t tokens = 0;
        for (const auto & [session_id, history] : states_) {
            (void)session_id;
            snapshots += history.size();
            for (const auto & snapshot : history) {
                tokens += snapshot.tokens.size();
            }
        }
        metric_sessions_.store(states_.size());
        metric_snapshots_.store(snapshots);
        metric_tokens_.store(tokens);
        metric_bytes_.store(bytes_);
    }

    void evict_history_to_limit(
            const std::string & session_id,
            uint64_t protected_clock) {
        auto found = states_.find(session_id);
        while (found != states_.end() &&
                found->second.size() > max_snapshots_per_session_) {
            size_t victim = found->second.size();
            for (size_t index = 0; index < found->second.size(); ++index) {
                const auto & snapshot = found->second[index];
                if (snapshot.last_used == protected_clock) continue;
                if (victim == found->second.size() ||
                        snapshot.last_used <
                            found->second[victim].last_used) {
                    victim = index;
                }
            }
            if (victim == found->second.size()) break;
            erase_snapshot(session_id, victim, "history_evict");
            found = states_.find(session_id);
        }
    }

    void evict_to_budget(
            const std::string & protected_session,
            uint64_t protected_clock) {
        while (states_.size() > max_sessions_) {
            auto victim = states_.end();
            uint64_t victim_last_used = 0;
            for (auto it = states_.begin(); it != states_.end(); ++it) {
                if (it->first == protected_session) continue;
                uint64_t session_last_used = 0;
                for (const auto & snapshot : it->second) {
                    session_last_used = std::max(
                        session_last_used, snapshot.last_used);
                }
                if (victim == states_.end() ||
                        session_last_used < victim_last_used) {
                    victim = it;
                    victim_last_used = session_last_used;
                }
            }
            if (victim == states_.end()) break;
            close_session(victim->first);
        }
        while (bytes_ > max_bytes_) {
            std::string victim_session;
            size_t victim_snapshot = 0;
            uint64_t victim_last_used = 0;
            bool found_victim = false;
            for (const auto & [session_id, history] : states_) {
                for (size_t index = 0; index < history.size(); ++index) {
                    const auto & snapshot = history[index];
                    if (session_id == protected_session &&
                            snapshot.last_used == protected_clock) {
                        continue;
                    }
                    if (!found_victim ||
                            snapshot.last_used < victim_last_used) {
                        victim_session = session_id;
                        victim_snapshot = index;
                        victim_last_used = snapshot.last_used;
                        found_victim = true;
                    }
                }
            }
            if (!found_victim) break;
            erase_snapshot(victim_session, victim_snapshot, "budget_evict");
        }
    }

    void erase_snapshot(
            const std::string & session_id,
            size_t index,
            const char * action) {
        auto found = states_.find(session_id);
        if (found == states_.end() || index >= found->second.size()) return;
        const size_t removed_bytes = found->second[index].bytes;
        if (trace_) {
            std::cerr << "runtime_session_cache action=" << action
                      << " session=" << session_id
                      << " tokens=" << found->second[index].tokens.size()
                      << " bytes=" << removed_bytes << std::endl;
        }
        bytes_ -= removed_bytes;
        found->second.erase(found->second.begin() +
            static_cast<std::ptrdiff_t>(index));
        if (found->second.empty()) states_.erase(found);
        sync_telemetry();
    }

    std::unordered_map<
        std::string, std::vector<TextSessionState>> states_;
    std::unordered_map<std::string, PagedBinding> paged_bindings_;
    std::shared_ptr<mfq::cache::PagedPrefixCache> paged_cache_;
    uint64_t paged_disk_budget_ = 0;
    uint64_t paged_hot_budget_ = 0;
    size_t max_sessions_ = 4;
    size_t max_snapshots_per_session_ = 4;
    size_t max_bytes_ = 2ULL * 1024ULL * 1024ULL * 1024ULL;
    size_t bytes_ = 0;
    uint64_t clock_ = 0;
    std::atomic<uint64_t> queries_{0};
    std::atomic<uint64_t> hits_{0};
    std::atomic<uint64_t> hit_tokens_{0};
    std::atomic<size_t> metric_sessions_{0};
    std::atomic<size_t> metric_snapshots_{0};
    std::atomic<size_t> metric_tokens_{0};
    std::atomic<size_t> metric_bytes_{0};
    bool trace_ = false;
};



template <typename Model>
using PreparedPromptFactory =
    std::function<std::optional<CudaPreparedPrompt>(Model&)>;

template <typename Model>
static int32_t generate_tokens(
    Model& model,
    std::mutex & model_mutex,
    DecodeGraphCache & graph_cache,
    TextSessionCache & session_cache,
    const std::vector<int64_t> & prompt,
    const MfqSamplingParams & sampling,
    const MfqTokenCallback & on_token,
    const MfqPrefillCallback & on_prefill,
    const MfqPromptCachePlan & cache_plan,
    const MfqTokenConstraintPtr & token_constraint,
    MtpModule* mtp = nullptr,
    int64_t prefill_chunk_size = 2048,
    PreparedPromptFactory<Model> prepare_prompt = {})
{
    std::lock_guard<std::mutex> lock(model_mutex);
    const auto prepared = prepare_prompt
        ? prepare_prompt(model)
        : std::optional<CudaPreparedPrompt>{};
    if (prepared && prepared->token_ids != prompt) {
        throw std::invalid_argument(
            "prepared prompt token IDs disagree with the rendered prompt");
    }
    const bool transformed_prompt = prepared && prepared->transformed();
    if (mtp != nullptr) {
        mtp->last_stats = {};
        mtp->last_stats.available = true;
    }
    const char* mtp_reprefill = std::getenv("MFQ_RUNTIME_REPREFILL");
    const char* mtp_trace = std::getenv("MFQ_RUNTIME_TRACE_INCREMENTAL");
    if (mtp != nullptr && sampling.enable_mtp && sampling.max_tokens > 1 &&
        mfq_token_constraint_supports_speculation(token_constraint) &&
        !(mtp_reprefill != nullptr && mtp_reprefill[0] == '1') &&
        !(mtp_trace != nullptr && mtp_trace[0] == '1')) {
        // Predictor state is not in the persistent session snapshot contract.
        if constexpr (
                Model::backbone == mfq::cuda::CudaBackbone::generic_qwen ||
                Model::backbone == mfq::cuda::CudaBackbone::qwen4_exp ||
                Model::backbone == mfq::cuda::CudaBackbone::glm5_next ||
                Model::backbone == mfq::cuda::CudaBackbone::deepseek_v41) {
            return run_mtp_generation<Model::backbone>(
                model, *mtp, prompt, sampling, on_token, on_prefill,
                prefill_chunk_size, token_constraint,
                transformed_prompt ? &*prepared : nullptr);
        }
        throw std::runtime_error(
            "MTP is unavailable for this causal LM type");
    }
    auto options = mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64).device(mfq_tensor_backend::kCUDA);
    const size_t stable_prefix_tokens = transformed_prompt ? 0 : std::min(
        cache_plan.stable_prefix_tokens, prompt.size());
    const bool cache_enabled =
        !transformed_prompt && stable_prefix_tokens > 0 &&
        (!cache_plan.session_id.empty() ||
         session_cache.persistent_prefix_enabled()) &&
        model.supports_text_session_state();
    const size_t reused_tokens = cache_enabled
        ? session_cache.restore_best(
            model, cache_plan.session_id, prompt, stable_prefix_tokens)
        : 0;
    if (reused_tokens == 0) model.reset(1);
    auto full_ids = mfq_tensor_backend::tensor(prompt, options)
        .reshape({1, -1}).contiguous();
    auto ids = full_ids.narrow(
        1, static_cast<int64_t>(reused_tokens),
        static_cast<int64_t>(prompt.size() - reused_tokens)).contiguous();
    graph_cache.ensure_storage(model.vocab_size());
    auto random_host = mfq_tensor_backend::empty(
        {1}, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kFloat32).device(mfq_tensor_backend::kCPU).pinned_memory(true));
    auto random_cuda = mfq_tensor_backend::empty(
        {1}, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kFloat32).device(mfq_tensor_backend::kCUDA));
    mfq::cuda::Sampler sampler(
        sampling,
        mfq::cuda::SamplingOps(random_host, std::move(random_cuda)));
    const bool has_penalties = sampler.has_penalties();
    auto counts = has_penalties ? graph_cache.counts : mfq_tensor_backend::Tensor();
    if (has_penalties) {
        counts.zero_();
        sample_token_counts_add_cuda(counts, full_ids);
    }
    const auto store_session_snapshot = [&](size_t token_count) {
        if (!cache_enabled || model.cache_pos !=
                static_cast<int64_t>(token_count)) {
            return;
        }
        try {
            std::vector<int64_t> snapshot_tokens(
                prompt.begin(), prompt.begin() +
                    static_cast<std::ptrdiff_t>(token_count));
            session_cache.store(
                cache_plan.session_id,
                model.capture_text_session_state(snapshot_tokens));
        } catch (const std::exception & error) {
            std::cerr << "runtime_session_cache action=skip session="
                      << cache_plan.session_id
                      << " error=" << error.what() << std::endl;
        }
    };
    auto sample_first_token = [&]() {
        PrefillCudaTimer prefill_timer;
        if (cache_enabled && stable_prefix_tokens < prompt.size()) {
            if (reused_tokens < stable_prefix_tokens) {
                auto stable_suffix = full_ids.narrow(
                    1, static_cast<int64_t>(reused_tokens),
                    static_cast<int64_t>(
                        stable_prefix_tokens - reused_tokens)).contiguous();
                stable_suffix = prefill_tail(
                    model, std::move(stable_suffix), prefill_chunk_size);
                MfqOptional<mfq_tensor_backend::Tensor> stable_seq_len = mfq_nullopt;
                if (!Model::is_minicpmo45 && model.cache_pos > 0 &&
                        stable_suffix.size(1) == 1) {
                    stable_seq_len = mfq_tensor_backend::full(
                        {1}, model.cache_pos + 1, options);
                }
                (void)model.hidden_forward(
                    stable_suffix, mfq_nullopt, stable_seq_len);
            }
            store_session_snapshot(stable_prefix_tokens);
            ids = full_ids.narrow(
                1, static_cast<int64_t>(stable_prefix_tokens),
                static_cast<int64_t>(
                    prompt.size() - stable_prefix_tokens)).contiguous();
        }
        mfq_tensor_backend::Tensor next;
        if (transformed_prompt) {
            auto hidden = hidden_forward_prepared_chunked(
                model, full_ids, *prepared, prefill_chunk_size);
            auto logits = model.lm_head.forward(
                hidden.index({Slice(), -1, Slice()})
                    .to(mfq_tensor_backend::kFloat16).contiguous())
                .contiguous().view({1, -1});
            MFQ_CUDA_CHECK(cudaEventRecord(
                prefill_timer.finished_event(),
                mfq_get_current_cuda_stream()));
            next = mfq::cuda::sample_logits(
                sampler, std::move(logits), counts, token_constraint);
        } else {
            ids = prefill_tail(
                model, std::move(ids), prefill_chunk_size);
            next = sample_token(
                model, ids, sampler, counts, token_constraint,
                prefill_timer.finished_event());
        }
        const int64_t token = next.template item<int64_t>();
        const double prefill_ms = prefill_timer.elapsed_ms();
        if (stable_prefix_tokens == prompt.size()) {
            store_session_snapshot(stable_prefix_tokens);
        }
        if (on_prefill) {
            on_prefill(MfqPrefillTiming{
                prompt.size() - reused_tokens,
                prefill_ms,
                0.0,
                prefill_ms});
        }
        return std::make_pair(std::move(next), token);
    };
    const char * reprefill_env = std::getenv("MFQ_RUNTIME_REPREFILL");
    const bool reprefill = !transformed_prompt &&
        reprefill_env != nullptr && reprefill_env[0] == '1';
    std::vector<int64_t> history = prompt;
    const char * trace_incremental_env =
        std::getenv("MFQ_RUNTIME_TRACE_INCREMENTAL");
    const bool trace_incremental =
        trace_incremental_env != nullptr && trace_incremental_env[0] == '1';
    if (!transformed_prompt && trace_incremental && sampling.max_tokens > 0) {
        auto [first, first_token] = sample_first_token();
        if (!on_token(first_token)) return 1;

        std::vector<mfq_tensor_backend::Tensor> incremental_trace;
        std::vector<mfq_tensor_backend::Tensor> full_trace;
        std::vector<std::pair<std::string, mfq_tensor_backend::Tensor>> incremental_gemma_trace;
        std::vector<std::pair<std::string, mfq_tensor_backend::Tensor>> full_gemma_trace;
        const int64_t decode_len = model.cache_pos + 1;
        auto seq_len = mfq_tensor_backend::tensor({decode_len}, options);
        g_gemma_trace_layer = 0;
        g_gemma_stage_trace = &incremental_gemma_trace;
        auto incremental_hidden = model.hidden_forward(
            first.reshape({1, 1}), mfq_nullopt, seq_len, &incremental_trace);
        g_gemma_stage_trace = nullptr;

        history.push_back(first_token);
        model.reset(1);
        auto full_ids = mfq_tensor_backend::tensor(history, options).reshape({1, -1}).contiguous();
        g_gemma_stage_trace = &full_gemma_trace;
        auto full_hidden = model.hidden_forward(
            full_ids, mfq_nullopt, mfq_nullopt, &full_trace);
        g_gemma_stage_trace = nullptr;
        g_gemma_trace_layer = -1;
        mfq_cuda_synchronize();

        if (incremental_trace.size() != full_trace.size()) {
            throw std::runtime_error("incremental trace stage count mismatch");
        }
        for (size_t i = 0; i < incremental_trace.size(); ++i) {
            auto got = incremental_trace[i].reshape({-1}).to(mfq_tensor_backend::kFloat64);
            auto ref = full_trace[i].index({Slice(), -1, Slice()}).reshape({-1}).to(mfq_tensor_backend::kFloat64);
            const double denominator = std::max(ref.norm().template item<double>(), 1.0e-30);
            const double relative_l2 = (got - ref).norm().template item<double>() / denominator;
            const double cosine = mfq_tensor_backend::dot(got, ref).template item<double>() /
                std::max(got.norm().template item<double>() * denominator, 1.0e-30);
            std::cerr << "incremental_trace stage="
                      << (i == 0 ? "embedding" : "block_" + std::to_string(i - 1))
                      << " relative_l2=" << relative_l2
                      << " cosine=" << cosine << std::endl;
        }
        if (incremental_gemma_trace.size() != full_gemma_trace.size()) {
            throw std::runtime_error("incremental Gemma stage count mismatch");
        }
        for (size_t i = 0; i < incremental_gemma_trace.size(); ++i) {
            const auto & got_tensor = incremental_gemma_trace[i].second;
            const auto & full_tensor = full_gemma_trace[i].second;
            if (incremental_gemma_trace[i].first != full_gemma_trace[i].first ||
                full_tensor.numel() < got_tensor.numel()) {
                throw std::runtime_error("incremental Gemma stage layout mismatch");
            }
            auto got = got_tensor.reshape({-1}).to(mfq_tensor_backend::kFloat64);
            auto full_flat = full_tensor.reshape({-1});
            auto ref = full_flat.narrow(
                0, full_flat.numel() - got_tensor.numel(), got_tensor.numel()).to(mfq_tensor_backend::kFloat64);
            const double denominator = std::max(ref.norm().template item<double>(), 1.0e-30);
            const double relative_l2 = (got - ref).norm().template item<double>() / denominator;
            const double cosine = mfq_tensor_backend::dot(got, ref).template item<double>() /
                std::max(got.norm().template item<double>() * denominator, 1.0e-30);
            std::cerr << "incremental_gemma_trace stage="
                      << incremental_gemma_trace[i].first
                      << " relative_l2=" << relative_l2
                      << " cosine=" << cosine << std::endl;
        }
        auto incremental_logits = model.lm_head.forward(
            incremental_hidden.index({Slice(), -1, Slice()}).to(mfq_tensor_backend::kFloat16).contiguous());
        auto full_logits = model.lm_head.forward(
            full_hidden.index({Slice(), -1, Slice()}).to(mfq_tensor_backend::kFloat16).contiguous());
        const double logits_relative_l2 =
            (incremental_logits.to(mfq_tensor_backend::kFloat64) - full_logits.to(mfq_tensor_backend::kFloat64)).norm().template item<double>() /
            std::max(full_logits.to(mfq_tensor_backend::kFloat64).norm().template item<double>(), 1.0e-30);
        std::cerr << "incremental_trace logits_relative_l2=" << logits_relative_l2
                  << " incremental_top=" << incremental_logits.argmax(-1).template item<int64_t>()
                  << " full_top=" << full_logits.argmax(-1).template item<int64_t>() << std::endl;
        return 1;
    }

    const char * graph_env = std::getenv("MFQ_RUNTIME_CUDA_GRAPH");
    const bool graph_enabled =
        (graph_env == nullptr || graph_env[0] != '0') &&
        !Model::is_flash_next &&
        mfq_cuda_graph_capture_supported() &&
        g_dsv4_cpu_offload_layers.empty() &&
        g_dense_cpu_layer_count == 0 &&
        !g_moe_expert_cache &&
        model_parallel_cuda_graph_enabled();
    const char * graph_min_env =
        std::getenv("MFQ_RUNTIME_CUDA_GRAPH_MIN_TOKENS");
    const int32_t graph_min_tokens = graph_min_env != nullptr
        ? std::max<int32_t>(2, std::atoi(graph_min_env))
        : 16;
    // Grammar state advances on the CPU and may require a one-off full-logit
    // mask, so constrained requests cannot be replayed as a fixed CUDA graph.
    // Unconstrained decode keeps the existing graph fast path unchanged.
    const bool graph_eligible = !transformed_prompt && graph_enabled && !reprefill &&
        !token_constraint &&
        sampling.max_tokens >= graph_min_tokens &&
        sampling.max_tokens <= graph_cache.generated_capacity;
    if (graph_eligible) {
        const bool greedy = sampler.greedy();
        auto [first, first_token] = sample_first_token();
        int32_t generated = 1;
        if (!on_token(first_token) || generated >= sampling.max_tokens) return generated;

        graph_cache.ensure_compute_streams();
        MfqCudaGuard graph_device_guard(
            graph_cache.stream.device_index());
        auto graph_stream_guards =
            activate_cuda_graph_compute_streams(
                graph_cache.compute_streams);
        cudaStream_t graph_raw_stream = graph_cache.stream.stream();
        if (has_penalties) {
            sample_token_counts_add_cuda(graph_cache.counts, first.contiguous());
        }

        int64_t pos_h = model.cache_pos;
        int64_t len_h = pos_h + 1;
        int64_t step_h = 1;
        MFQ_CUDA_CHECK(cudaMemcpyAsync(
            graph_cache.static_input.template data_ptr<int64_t>(), first.template data_ptr<int64_t>(),
            sizeof(int64_t), cudaMemcpyDeviceToDevice, graph_raw_stream));
        MFQ_CUDA_CHECK(cudaMemcpyAsync(
            graph_cache.generated.template data_ptr<int64_t>(), first.template data_ptr<int64_t>(),
            sizeof(int64_t), cudaMemcpyDeviceToDevice, graph_raw_stream));
        MFQ_CUDA_CHECK(cudaMemcpyAsync(
            graph_cache.static_pos.template data_ptr<int64_t>(), &pos_h,
            sizeof(int64_t), cudaMemcpyHostToDevice, graph_raw_stream));
        MFQ_CUDA_CHECK(cudaMemcpyAsync(
            graph_cache.static_len.template data_ptr<int64_t>(), &len_h,
            sizeof(int64_t), cudaMemcpyHostToDevice, graph_raw_stream));
        MFQ_CUDA_CHECK(cudaMemcpyAsync(
            graph_cache.static_step.template data_ptr<int64_t>(), &step_h,
            sizeof(int64_t), cudaMemcpyHostToDevice, graph_raw_stream));
        *random_host.template data_ptr<float>() = 0.5f;
        MFQ_CUDA_CHECK(cudaMemcpyAsync(
            graph_cache.random.template data_ptr<float>(), random_host.template data_ptr<float>(),
            sizeof(float), cudaMemcpyHostToDevice, graph_raw_stream));
        MFQ_CUDA_CHECK(cudaStreamSynchronize(graph_raw_stream));

        auto sample_static = [&]() {
            if (greedy && !has_penalties) {
                return model.next_token_static(
                    graph_cache.static_input, graph_cache.static_pos, graph_cache.static_len);
            }
            auto logits = model.last_logits_static(
                    graph_cache.static_input, graph_cache.static_pos, graph_cache.static_len)
                .contiguous().view({1, -1});
            if (has_penalties) {
                logits = sampler.apply_penalties(
                    std::move(logits), graph_cache.counts);
            }
            if (greedy) {
                return sampler.ops().sample_greedy(std::move(logits));
            }
            return sampler.ops().sample_stochastic(
                std::move(logits), graph_cache.random, sampler.params());
        };

        const int64_t requested_len = model.cache_pos + sampling.max_tokens;
        const int64_t planned_len = decode_graph_bucket(
            requested_len, model.max_position_embeddings());
        const bool cache_hit = graph_cache.matches(planned_len, sampling, greedy);
        if (!cache_hit) {
            graph_cache.invalidate();
            mfq_cuda_empty_cache();
            g_decode_graph_attention_kv_len = planned_len;
            g_decode_graph_attention_parts = planned_len >= 192 ? (planned_len + 127) / 128 : 1;
            g_decode_graph_attention_parts = std::min<int64_t>(
                g_decode_graph_attention_parts, FullBlock::kDecodeAttentionMaxParts);
            try {
                DecodeGraphBranchScope branch_scope;
                graph_cache.graph = std::make_unique<MfqCudaGraph>();
                prepare_decode_graph_memory(model, *graph_cache.graph,
                    [&]() { (void)sample_static(); },
                    graph_cache.graph_participant_streams());

                graph_cache.graph->capture_begin();
                graph_cache.static_next = sample_static();
                if (has_penalties) {
                    sample_token_counts_add_cuda(
                        graph_cache.counts, graph_cache.static_next.contiguous());
                }
                decode_graph_commit_cuda(
                    graph_cache.static_next, graph_cache.generated,
                    graph_cache.static_step, graph_cache.static_input,
                    graph_cache.static_pos, graph_cache.static_len);
                graph_cache.graph->capture_end();
                graph_cache.set_key(planned_len, sampling, greedy);
                ++graph_cache.captures;
            } catch (...) {
                graph_cache.invalidate();
                g_decode_graph_attention_kv_len = 0;
                g_decode_graph_attention_parts = 0;
                throw;
            }
            g_decode_graph_attention_kv_len = 0;
            g_decode_graph_attention_parts = 0;
            report_cuda_memory("runtime_graph_capture");
        } else {
            ++graph_cache.reuses;
        }
        if (trace_cuda_graph()) {
            std::cerr << "runtime_cuda_graph action=" << (cache_hit ? "reuse" : "capture")
                      << " requested_len=" << requested_len
                      << " planned_len=" << planned_len
                      << " captures=" << graph_cache.captures
                      << " reuses=" << graph_cache.reuses << std::endl;
        }

        while (generated < sampling.max_tokens) {
            if (!greedy) {
                *random_host.template data_ptr<float>() = sampler.next_uniform_float();
                MFQ_CUDA_CHECK(cudaMemcpyAsync(
                    graph_cache.random.template data_ptr<float>(), random_host.template data_ptr<float>(),
                    sizeof(float), cudaMemcpyHostToDevice, graph_raw_stream));
            }
            graph_cache.graph->replay();
            const int64_t token = graph_cache.static_next.template item<int64_t>();
            ++generated;
            if (!on_token(token)) break;
        }
        model.cache_pos += generated - 1;
        return generated;
    }

    int32_t generated = 0;
    while (generated < sampling.max_tokens) {
        if (reprefill && generated > 0) {
            model.reset(1);
            ids = mfq_tensor_backend::tensor(history, options).reshape({1, -1}).contiguous();
        }
        mfq_tensor_backend::Tensor next;
        int64_t token = 0;
        if (generated == 0) {
            auto first = sample_first_token();
            next = std::move(first.first);
            token = first.second;
        } else {
            next = sample_token(
                model, ids, sampler, counts, token_constraint);
            token = next.template item<int64_t>();
        }
        ++generated;
        if (!on_token(token)) break;
        history.push_back(token);
        if (has_penalties) sample_token_counts_add_cuda(counts, next.contiguous());
        ids = next.reshape({1, 1});
    }
    return generated;
}

#include "qwen_continuous_batching.h"

// Real-weight correctness gate; does not require a tokenizer or start a runtime transport.
// It calls the same MTP generator used by the runtime, with synthetic token IDs.
static int run_qwen35_mtp_check(
        mfq::cuda::Qwen35CausalLm& model, Qwen35Mtp& mtp) {
    using Tensor = mfq_tensor_backend::Tensor;
    const MtpTarget target{
        [&model](Tensor ids) {
            return model.embed_forward(std::move(ids));
        },
        [&model](Tensor hidden) {
            return model.logits_from_hidden(std::move(hidden));
        },
        &model.rope,
    };
    // Identity projection isolates both dense gate modes and their dtype casts.
    const auto float_options = mfq_tensor_backend::TensorOptions()
        .device(mfq_tensor_backend::kCUDA).dtype(mfq_tensor_backend::kFloat32);
    std::vector<float> identity(33 * 33, 0.f), input(6 * 33), gate(6 * 33);
    for (int i = 0; i < 33; ++i) identity[i * 33 + i] = 1.f;
    for (int i = 0; i < 6 * 33; ++i) {
        input[i] = static_cast<float>(i % 33 - 16) / 16.f;
        gate[i] = static_cast<float>(i % 31 - 15) / 4.f;
    }
    auto test_input = mfq_tensor_backend::tensor(input, float_options).reshape({1, 6, 33});
    auto test_gate = mfq_tensor_backend::tensor(gate, float_options).reshape({1, 6, 33});
    for (auto dtype : {mfq_tensor_backend::kFloat32, mfq_tensor_backend::kFloat16,
                       mfq_tensor_backend::kBFloat16}) {
        QuantLinear linear;
        linear.kind = QuantLinearKind::Dense;
        linear.dense_small_m_rowwise = true;
        linear.dense = mfq_tensor_backend::tensor(identity, float_options)
            .reshape({33, 33}).to(dtype);
        const double tolerance = dtype == mfq_tensor_backend::kBFloat16 ? .008
            : dtype == mfq_tensor_backend::kFloat16 ? .001 : 2.e-6;
        for (int m = 1; m <= 6; ++m) for (int mode : {1, 2}) {
            auto actual = linear.forward_input_mul(test_input.narrow(1, 0, m),
                test_gate.narrow(1, 0, m), mode);
            MFQ_RUNTIME_CHECK(actual.scalar_type() == dtype && actual.size(1) == m,
                "dense gate output dtype or shape mismatch");
            auto values = actual.to(mfq_tensor_backend::kFloat32).cpu();
            for (int i = 0; i < m * 33; ++i) {
                const double activated = (mode == 1 ? 1. : gate[i]) / (1. + std::exp(-double(gate[i])));
                const double reference = double(input[i]) * activated;
                const double value = values.template data_ptr<float>()[i];
                MFQ_RUNTIME_CHECK(std::isfinite(value) &&
                    std::abs(value - reference) <= tolerance * (1. + std::abs(reference)),
                    "dense gate differs from CPU double identity oracle");
            }
        }
    }
    std::cout << "mtp_check dense_gate_cases=36 PASS\n";
    const auto options = mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA)
        .dtype(mfq_tensor_backend::kInt64);
    auto ids = [&](const std::vector<int64_t>& tokens) {
        return mfq_tensor_backend::tensor(tokens, options).reshape({1, -1});
    };
    bool numerical_mismatch = false;
    auto compare = [&](const Tensor& actual, const Tensor& reference, double tolerance,
                       const char* name) {
        MFQ_RUNTIME_CHECK(actual.sizes() == reference.sizes(), "MTP gate tensor shape mismatch");
        auto a = actual.contiguous().to(mfq_tensor_backend::kFloat32).cpu();
        auto r = reference.contiguous().to(mfq_tensor_backend::kFloat32).cpu();
        double squared = 0., norm = 0.;
        for (int64_t i = 0; i < a.numel(); ++i) {
            const double av = a.template data_ptr<float>()[i], rv = r.template data_ptr<float>()[i];
            MFQ_RUNTIME_CHECK(std::isfinite(av) && std::isfinite(rv), "nonfinite MTP gate value");
            squared += (av - rv) * (av - rv);
            norm += rv * rv;
        }
        const double rel = std::sqrt(squared / std::max(norm, 1.e-30));
        std::cout << "mtp_check " << name << " relative_l2=" << rel << '\n';
        numerical_mismatch = numerical_mismatch || rel > tolerance;
    };
    const std::vector<int64_t> prompt{100, 200, 300, 400, 500, 600, 700};
    // Explicitly test zero, partial, and full acceptance regardless of the
    // real predictor's acceptance rate.
    for (int accepted_drafts : {0, 1, 2}) {
        model.reset(1);
        (void)model.hidden_forward(ids(prompt));
        std::vector<Tensor> verify_trace;
        std::vector<std::pair<std::string, Tensor>> verify_stages;
        g_gemma_trace_layer = 3;
        g_gemma_stage_trace = &verify_stages;
        (void)model.hidden_forward(ids({37, 41, 43}), mfq_nullopt, mfq_nullopt,
            &verify_trace, mfq_nullopt, nullptr, 1);
        g_gemma_stage_trace = nullptr;
        if (accepted_drafts == 2) model.commit_speculative();
        else model.rollback_speculative(accepted_drafts);
        MFQ_RUNTIME_CHECK(
            model.cache_pos == static_cast<int64_t>(prompt.size()) +
                1 + accepted_drafts,
            "MTP resolution retained an incorrect logical cache length");
        auto actual = model.last_logits(ids({47})).clone();
        std::vector<std::pair<Tensor, Tensor>> states;
        for (auto& block : model.blocks) {
            if (auto* linear = dynamic_cast<
                    mfq::cuda::qwen35::LinearAttentionBlock*>(block.get()))
                states.emplace_back(linear->conv_state.clone(), linear->gdn_state.clone());
        }
        model.reset(1);
        (void)model.hidden_forward(ids(prompt));
        std::vector<Tensor> serial_trace;
        std::vector<std::pair<std::string, Tensor>> serial_stages;
        g_gemma_stage_trace = &serial_stages;
        (void)model.hidden_forward(ids({37}), mfq_nullopt, mfq_nullopt,
            &serial_trace);
        g_gemma_stage_trace = nullptr;
        MFQ_RUNTIME_CHECK(verify_stages.size() == serial_stages.size() && !verify_stages.empty(),
            "MTP full-attention stage trace mismatch");
        for (size_t stage = 0; stage < verify_stages.size(); ++stage) {
            MFQ_RUNTIME_CHECK(verify_stages[stage].first == serial_stages[stage].first,
                "MTP full-attention stage names differ");
            compare(verify_stages[stage].second.narrow(1, 0, 1), serial_stages[stage].second,
                .005, verify_stages[stage].first.c_str());
        }
        MFQ_RUNTIME_CHECK(verify_trace.size() == serial_trace.size(),
            "MTP diagnostic block trace size mismatch");
        for (size_t layer = 0; layer < verify_trace.size(); ++layer) {
            const auto label = "confirmed_prefix_block_" + std::to_string(layer);
            compare(verify_trace[layer].narrow(1, 0, 1), serial_trace[layer], .005,
                label.c_str());
        }
        if (accepted_drafts >= 1) (void)model.hidden_forward(ids({41}));
        if (accepted_drafts >= 2) (void)model.hidden_forward(ids({43}));
        auto reference = model.last_logits(ids({47}));
        const auto resolution_label =
            "resolution_" + std::to_string(accepted_drafts) + "_logits";
        compare(actual, reference, .005, resolution_label.c_str());
        size_t state = 0;
        for (auto& block : model.blocks) {
            if (auto* linear = dynamic_cast<
                    mfq::cuda::qwen35::LinearAttentionBlock*>(block.get())) {
                const auto conv_label = "conv_continuation_" + std::to_string(state);
                const auto gdn_label = "gdn_continuation_" + std::to_string(state);
                compare(states[state].first, linear->conv_state, .002, conv_label.c_str());
                compare(states[state].second, linear->gdn_state, .002, gdn_label.c_str());
                ++state;
            }
        }
    }
    MFQ_RUNTIME_CHECK(!numerical_mismatch, "MTP gate numerical mismatch (see per-layer diagnostics)");
    model.reset(1);
    Tensor raw;
    (void)model.hidden_forward(ids(prompt), mfq_nullopt, mfq_nullopt, nullptr, mfq_nullopt, &raw);
    for (int tokens = 2; tokens <= 6; ++tokens) {
        auto next = ids(prompt).narrow(1, 1, tokens);
        std::vector<std::pair<std::string, Tensor>> batch_stages, row_stages;
        g_gemma_trace_layer = 0;
        if (tokens == 2) g_gemma_stage_trace = &batch_stages;
        mtp.reset();
        auto batched = mtp.forward(
            target, raw.narrow(1, 0, tokens), next).clone();
        mtp.reset();
        if (tokens == 2) g_gemma_stage_trace = &row_stages;
        std::vector<Tensor> serial;
        for (int t = 0; t < tokens; ++t)
            serial.push_back(mtp.forward(
                target, raw.narrow(1, t, 1), next.narrow(1, t, 1)));
        g_gemma_stage_trace = nullptr;
        if (tokens == 2) {
            MFQ_RUNTIME_CHECK(row_stages.size() == 2 * batch_stages.size(),
                "MTP predictor diagnostic stage count mismatch");
            for (size_t stage = 0; stage < batch_stages.size(); ++stage) {
                auto reference = mfq_tensor_backend::cat({row_stages[stage].second,
                    row_stages[stage + batch_stages.size()].second}, 1);
                const auto label = "predictor_stage_" + batch_stages[stage].first;
                compare(batch_stages[stage].second, reference,
                    std::numeric_limits<double>::infinity(), label.c_str());
            }
        }
        compare(batched, mfq_tensor_backend::cat(serial, 1), .005, "predictor_batched_vs_serial");
    }
    MFQ_RUNTIME_CHECK(!numerical_mismatch, "MTP predictor gate numerical mismatch");
    uint64_t total_cycles = 0;
    for (const auto& input : std::vector<std::vector<int64_t>>{{1, 2, 3}, prompt, std::vector<int64_t>(17, 10)}) {
        MfqSamplingParams params;
        params.max_tokens = 32;
        params.temperature = 0.;
        params.top_k = 1;
        params.seed = 20260907;
        model.reset(1);
        auto current = ids(input);
        std::vector<int64_t> expected;
        for (int step = 0; step < params.max_tokens; ++step) {
            auto next = model.next_token(current);
            expected.push_back(next.template item<int64_t>());
            current = next.reshape({1, 1});
        }
        std::vector<int64_t> got;
        const int produced = run_mtp_generation<mfq::cuda::CudaBackbone::generic_qwen>(model, mtp, input, params,
            [&](int64_t token) { got.push_back(token); return true; }, {});
        total_cycles += mtp.last_cycles;
        std::cout << "mtp_check greedy_prompt_tokens=" << input.size() << " produced=" << produced
            << " exact=" << (got == expected) << " accepted=" << mtp.last_accepted
            << " rejected=" << mtp.last_rejected << '\n';
        MFQ_RUNTIME_CHECK(produced == params.max_tokens && got == expected,
            "MTP greedy generation differs from ordinary incremental decode");
        // Callback stop then a fresh request exercises state reset after an
        // early return, including stopping before a computed bonus is emitted.
        got.clear();
        const int stopped = run_mtp_generation<mfq::cuda::CudaBackbone::generic_qwen>(model, mtp, input, params,
            [&](int64_t token) { got.push_back(token); return got.size() < 3; }, {});
        MFQ_RUNTIME_CHECK(stopped == 3 && got == std::vector<int64_t>(expected.begin(), expected.begin() + 3),
            "MTP callback emitted extra or incorrect tokens");
    }
    MfqSamplingParams stochastic;
    stochastic.max_tokens = 8;
    stochastic.temperature = .8;
    stochastic.top_k = 100;
    stochastic.top_p = .95;
    stochastic.presence_penalty = .2;
    stochastic.frequency_penalty = .1;
    stochastic.repetition_penalty = 1.05;
    stochastic.seed = 20260907;
    const int produced = run_mtp_generation<mfq::cuda::CudaBackbone::generic_qwen>(model, mtp, prompt, stochastic,
        [](int64_t) { return true; }, {});
    MFQ_RUNTIME_CHECK(produced == 8 && mtp.last_cycles > 0 && total_cycles > 0,
        "MTP runtime gate did not execute speculative cycles");
    int64_t linear_layers = 0, ffn_batches = 0, projection_batches = 0;
    for (const auto& block : model.blocks) {
        if (auto* linear = dynamic_cast<const
                mfq::cuda::qwen35::LinearAttentionBlock*>(block.get())) {
            MFQ_RUNTIME_CHECK(
                linear->speculative_ffn_batches > 0 &&
                    linear->speculative_projection_batches > 0,
                "MTP gate did not exercise full-window recurrent batching");
            ++linear_layers;
            ffn_batches += linear->speculative_ffn_batches;
            projection_batches += linear->speculative_projection_batches;
        }
    }
    MFQ_RUNTIME_CHECK(
        linear_layers > 0, "MTP batching gate found no recurrent layers");
    std::cout << "mtp_check batched_linear_layers=" << linear_layers
        << " ffn_calls=" << ffn_batches
        << " projection_calls=" << projection_batches << '\n';
    std::cout << "mtp_check PASS full_chain=1 greedy_tokens=96 stochastic_smoke_tokens=8\n";
    return 0;
}

// Fixed synthetic-ID latency probe through the actual runtime dispatch. Model
// loading and the independent serial oracle are outside every timed request.
static int run_qwen35_mtp_bench(
        mfq::cuda::Qwen35CausalLm& model, Qwen35Mtp& mtp,
        bool enable_mtp, int generated_tokens, int repetitions) {
    using Clock = std::chrono::steady_clock;
    using Tensor = mfq_tensor_backend::Tensor;
    MFQ_RUNTIME_CHECK(
        generated_tokens >= 2 && repetitions > 0 && repetitions <= 100 &&
            model.max_position_embeddings() >= generated_tokens + 17,
        "MTP benchmark requires gen>=2, reps1-100 and context>=gen+17");
    DecodeGraphCache graph_cache(model.max_position_embeddings());
    TextSessionCache session_cache;
    std::mutex model_mutex;
    MfqSamplingParams params;
    params.max_tokens = generated_tokens;
    params.temperature = 0.;
    params.top_k = 1;
    params.top_p = 1.;
    params.enable_mtp = enable_mtp;
    params.seed = 20260907;
    const auto options = mfq_tensor_backend::TensorOptions()
        .device(mfq_tensor_backend::kCUDA).dtype(mfq_tensor_backend::kInt64);
    const char* graph_env = std::getenv("MFQ_RUNTIME_CUDA_GRAPH");
    const char* profiler_env = std::getenv("MFQ_CUDA_PROFILER_RANGE");
    const bool profiler_range = profiler_env != nullptr && std::atoi(profiler_env) != 0;
    std::cout << "mtp_bench config mode=" << (enable_mtp ? "mtp" : "ordinary")
        << " full_window_batching=1"
        << " runtime_graph=" << (graph_env == nullptr ? "default" : graph_env)
        << " gen=" << generated_tokens << " reps=" << repetitions
        << " warmup=1 seed=20260907 synthetic_ids=1 eos_stop=0 session_cache=0\n";
    const auto ms_between = [](Clock::time_point first, Clock::time_point last) {
        return std::chrono::duration<double, std::milli>(last - first).count();
    };
    for (const auto& prompt : std::vector<std::vector<int64_t>>{
            {1, 2, 3}, {100, 200, 300, 400, 500, 600, 700}, std::vector<int64_t>(17, 10)}) {
        model.reset(1);
        Tensor input = mfq_tensor_backend::tensor(prompt, options).reshape({1, -1}).contiguous();
        std::vector<int64_t> reference;
        reference.reserve(generated_tokens);
        for (int step = 0; step < generated_tokens; ++step) {
            auto next = model.next_token(input);
            reference.push_back(next.template item<int64_t>());
            input = next.reshape({1, 1});
        }
        mfq_cuda_synchronize();
        for (int repeat = -1; repeat < repetitions; ++repeat) {
            std::vector<int64_t> output;
            output.reserve(generated_tokens);
            MfqPrefillTiming prefill;
            Clock::time_point first_token;
            mfq_cuda_synchronize();
            // Capture only the actual request; its independent oracle, full
            // warmup, model load and context construction remain unprofiled.
            const bool capture_request = profiler_range && repeat >= 0;
            if (capture_request) {
                std::cout << "mtp_bench profiler_begin prompt=" << prompt.size()
                    << " repeat=" << repeat << '\n';
                MFQ_CUDA_CHECK(cudaProfilerStart());
            }
            const auto started = Clock::now();
            const int produced = generate_tokens(model, model_mutex, graph_cache,
                session_cache, prompt, params, [&](int64_t token) {
                    if (output.empty()) first_token = Clock::now();
                    output.push_back(token);
                    return true;
                }, [&](const MfqPrefillTiming& timing) { prefill = timing; }, {}, {}, &mtp);
            mfq_cuda_synchronize();
            const auto finished = Clock::now();
            if (capture_request) {
                MFQ_CUDA_CHECK(cudaProfilerStop());
                std::cout << "mtp_bench profiler_end prompt=" << prompt.size()
                    << " repeat=" << repeat << '\n';
            }
            MFQ_RUNTIME_CHECK(produced == generated_tokens && output == reference,
                "MTP benchmark output differs from ordinary serial oracle");
            MFQ_RUNTIME_CHECK(!enable_mtp || mtp.last_cycles > 0,
                "MTP benchmark did not execute speculative cycles");
            const double total_ms = ms_between(started, finished);
            const double first_ms = ms_between(started, first_token);
            const double decode_ms = ms_between(first_token, finished);
            MFQ_RUNTIME_CHECK(std::isfinite(total_ms) && total_ms > 0. &&
                std::isfinite(decode_ms) && decode_ms > 0., "invalid benchmark clock interval");
            std::cout << "mtp_bench sample mode=" << (enable_mtp ? "mtp" : "ordinary")
                << " prompt=" << prompt.size() << " gen=" << produced
                << " repeat=" << repeat << " warmup=" << (repeat < 0)
                << " total_ms=" << total_ms << " ttft_ms=" << first_ms
                << " decode_ms=" << decode_ms << " prefill_gpu_ms=" << prefill.llm_ms
                << " total_tps=" << produced * 1000. / total_ms
                << " decode_tps=" << (produced - 1) * 1000. / decode_ms
                << " exact=1 cycles=" << (enable_mtp ? mtp.last_cycles : 0)
                << " accepted=" << (enable_mtp ? mtp.last_accepted : 0)
                << " rejected=" << (enable_mtp ? mtp.last_rejected : 0)
                << " graph_captures=" << graph_cache.captures
                << " graph_reuses=" << graph_cache.reuses << '\n';
        }
    }
    std::cout << "mtp_bench PASS samples=" << repetitions * 3 << '\n';
    return 0;
}

static int32_t generate_multimodal_tokens(
    MiniCPMO45Runtime & runtime,
    std::mutex & model_mutex,
    const std::vector<int64_t> & prompt,
    const MfqVisionInput & vision,
    const MfqSamplingParams & sampling,
    const MfqTokenCallback & on_token,
    const MfqPrefillCallback & on_prefill,
    const MfqTokenConstraintPtr & token_constraint)
{
    std::lock_guard<std::mutex> lock(model_mutex);
    if (prompt.empty() || sampling.max_tokens < 0) {
        throw std::invalid_argument(
            "MiniCPM-o multimodal generation input is invalid");
    }
    if (sampling.max_tokens == 0) {
        runtime.language.reset(1);
        return 0;
    }
    const auto generation_limit = std::min<int32_t>(
        sampling.max_tokens,
        static_cast<int32_t>(
            runtime.language.max_position_embeddings() -
            static_cast<int64_t>(prompt.size()) + 1));
    if (generation_limit <= 0) {
        throw std::invalid_argument(
            "MiniCPM-o multimodal prompt exceeds the context capacity");
    }

    const auto cpu_i64 =
        mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64).device(mfq_tensor_backend::kCPU);
    const auto cuda_i64 =
        mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64).device(mfq_tensor_backend::kCUDA);
    auto input_ids = mfq_tensor_backend::tensor(prompt, cuda_i64)
        .reshape({1, -1}).contiguous();
    mfq_tensor_backend::Tensor pixels;
    mfq_tensor_backend::Tensor patch_mask;
    mfq_tensor_backend::Tensor target_sizes;
    mfq_tensor_backend::Tensor image_bounds;
    if (!vision.image_bounds.empty()) {
        pixels = mfq_tensor_backend::from_blob(
            const_cast<float *>(vision.pixel_values.data()),
            vision.pixel_shape,
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kFloat32).device(mfq_tensor_backend::kCPU))
            .clone();
        patch_mask = mfq_tensor_backend::from_blob(
            const_cast<uint8_t *>(vision.patch_mask.data()),
            vision.patch_mask_shape,
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kUInt8).device(mfq_tensor_backend::kCPU))
            .clone().to(mfq_tensor_backend::kBool);
        target_sizes = mfq_tensor_backend::from_blob(
            const_cast<int32_t *>(vision.target_sizes.data()),
            vision.target_sizes_shape,
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt32).device(mfq_tensor_backend::kCPU))
            .clone();
        image_bounds = mfq_tensor_backend::from_blob(
            const_cast<int64_t *>(vision.image_bounds.data()),
            std::vector<int64_t>{
                static_cast<int64_t>(vision.image_bounds.size() / 4), 4},
            cpu_i64).clone();
    }
    mfq_tensor_backend::Tensor audio_features;
    mfq_tensor_backend::Tensor audio_lengths;
    mfq_tensor_backend::Tensor audio_bounds;
    if (!vision.audio_bounds.empty()) {
        audio_features = mfq_tensor_backend::from_blob(
            const_cast<float *>(vision.audio_features.data()),
            vision.audio_features_shape,
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kFloat32).device(mfq_tensor_backend::kCPU))
            .clone();
        audio_lengths = mfq_tensor_backend::tensor(
            vision.audio_lengths, cpu_i64).contiguous();
        audio_bounds = mfq_tensor_backend::from_blob(
            const_cast<int64_t *>(vision.audio_bounds.data()),
            std::vector<int64_t>{
                static_cast<int64_t>(vision.audio_bounds.size() / 4), 4},
            cpu_i64).clone();
    }

    auto random_host = mfq_tensor_backend::empty(
        {1}, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kFloat32)
            .device(mfq_tensor_backend::kCPU).pinned_memory(true));
    auto random_cuda = mfq_tensor_backend::empty(
        {1}, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kFloat32)
            .device(mfq_tensor_backend::kCUDA));
    mfq::cuda::Sampler sampler(
        sampling,
        mfq::cuda::SamplingOps(
            std::move(random_host), std::move(random_cuda)));
    const bool has_penalties = sampler.has_penalties();
    auto counts = has_penalties
        ? mfq_tensor_backend::zeros(
              {runtime.language.vocab_size()},
              mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt32).device(mfq_tensor_backend::kCUDA))
        : mfq_tensor_backend::Tensor();
    if (has_penalties) {
        sample_token_counts_add_cuda(counts, input_ids);
    }

    PrefillCudaTimer prefill_timer;
    auto result = runtime.forward(
        input_ids,
        mfq_tensor_backend::Tensor(),
        mfq_tensor_backend::Tensor(),
        pixels,
        patch_mask,
        target_sizes,
        image_bounds,
        audio_features,
        audio_lengths,
        audio_bounds);
    auto logits = result.logits.index({Slice(), -1, Slice()})
        .contiguous().view({1, -1});
    MFQ_CUDA_CHECK(cudaEventRecord(
        prefill_timer.finished_event(),
        mfq_get_current_cuda_stream()));
    auto next = mfq::cuda::sample_logits(
        sampler, std::move(logits), counts, token_constraint);
    if (on_prefill) {
        const double model_ms = prefill_timer.elapsed_ms();
        // The current CUDA composite timer covers both the multimodal encoder
        // and language prefill. Keep that total explicit instead of falsely
        // presenting it as comparable language-model-only time.
        on_prefill(MfqPrefillTiming{
            prompt.size(),
            0.0,
            0.0,
            model_ms});
    }

    int32_t generated = 0;
    while (generated < generation_limit) {
        const int64_t token = next.template item<int64_t>();
        ++generated;
        if (!on_token(token) || generated >= generation_limit) break;
        if (has_penalties) {
            sample_token_counts_add_cuda(counts, next.contiguous());
        }
        next = sample_token(
            runtime.language, next.reshape({1, 1}), sampler, counts,
            token_constraint);
    }
    return generated;
}

#include "diagnostics/model_checks.h"

static MfqDuplexBackend make_cuda_minicpmo45_duplex_backend(
        MiniCPMO45Runtime & runtime,
        std::mutex & model_mutex,
        std::optional<MiniCPMO45DuplexSession> & session) {
    MfqDuplexBackend backend;
    backend.name = "cuda";
    backend.start = [&](const MfqDuplexSessionParams & parameters) {
        if (parameters.special_ids.size() != 15) {
            throw std::invalid_argument(
                "MiniCPM-o duplex requires 15 special token IDs");
        }
        if ((!parameters.greedy &&
                (!std::isfinite(parameters.temperature) ||
                 parameters.temperature <= 0.0)) ||
                parameters.top_k < 0 ||
                parameters.top_k >
                    std::min<int64_t>(
                        runtime.language.vocab_size(), 1024) ||
                !std::isfinite(parameters.top_p) ||
                parameters.top_p <= 0.0 || parameters.top_p > 1.0 ||
                !std::isfinite(parameters.listen_probability_scale) ||
                parameters.listen_probability_scale < 0.0 ||
                !std::isfinite(parameters.repetition_penalty) ||
                parameters.repetition_penalty <= 0.0 ||
                parameters.repetition_window <= 0 ||
                !std::isfinite(parameters.length_penalty) ||
                parameters.length_penalty <= 0.0 ||
                !std::isfinite(parameters.tts_temperature) ||
                parameters.tts_temperature <= 0.0 ||
                !std::isfinite(parameters.tts_repetition_penalty) ||
                parameters.tts_repetition_penalty <= 0.0) {
            throw std::invalid_argument(
                "MiniCPM-o duplex sampling configuration is invalid");
        }
        const int64_t audio_bos = parameters.special_ids.back();
        if (audio_bos < 0 || audio_bos >= 152064 ||
                std::any_of(
                    parameters.special_ids.begin(),
                    parameters.special_ids.end() - 1,
                    [&](int64_t token) {
                        return token < 0 ||
                            token >= runtime.language.vocab_size();
                    }) ||
                std::any_of(
                    parameters.forbidden_ids.begin(),
                    parameters.forbidden_ids.end(),
                    [&](int64_t token) {
                        return token < 0 ||
                            token >= runtime.language.vocab_size();
                    })) {
            throw std::invalid_argument(
                "MiniCPM-o duplex token ID is out of range");
        }
        if ((parameters.reference_audio_frames == 0) !=
                parameters.reference_audio_features.empty() ||
                parameters.reference_audio_frames < 0 ||
                (!parameters.reference_audio_features.empty() &&
                 (parameters.reference_audio_frames < 3 ||
                  parameters.reference_audio_features.size() !=
                    static_cast<size_t>(
                        parameters.reference_audio_frames) * 80))) {
            throw std::invalid_argument(
                "MiniCPM-o reference Mel geometry is invalid");
        }

        std::lock_guard<std::mutex> lock(model_mutex);
        MfqCudaGuard guard(
            g_layer_placement.primary_device());
        mfq_tensor_backend::manual_seed(static_cast<int64_t>(parameters.seed));
        mfq_cuda_manual_seed_all(parameters.seed);
        auto special_ids = MiniCPMO45DuplexSpecialIds::from_tensor(
            mfq_tensor_backend::tensor(
                parameters.special_ids,
                mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64)));
        session.reset();
        session.emplace(
            runtime, special_ids, parameters.forbidden_ids,
            parameters.greedy);
        session->temperature = parameters.temperature;
        session->top_k = parameters.top_k;
        session->top_p = parameters.top_p;
        session->listen_probability_scale =
            parameters.listen_probability_scale;
        session->repetition_penalty = parameters.repetition_penalty;
        session->repetition_window = parameters.repetition_window;
        session->length_penalty = parameters.length_penalty;
        session->tts_temperature = parameters.tts_temperature;
        session->tts_repetition_penalty =
            parameters.tts_repetition_penalty;

        const auto ids_tensor = [](const std::vector<int64_t> & values) {
            return values.empty()
                ? mfq_tensor_backend::Tensor()
                : mfq_tensor_backend::tensor(
                    values,
                    mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64));
        };
        mfq_tensor_backend::Tensor reference_features;
        if (!parameters.reference_audio_features.empty()) {
            reference_features = mfq_tensor_backend::tensor(
                parameters.reference_audio_features,
                mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kFloat32))
                .reshape({1, 80, parameters.reference_audio_frames});
        }
        session->prepare(
            ids_tensor(parameters.system_prefix),
            reference_features,
            ids_tensor(parameters.system_suffix));
        mfq_cuda_synchronize();
    };
    backend.step = [&](const MfqDuplexStepInput & input) {
        const bool has_audio = input.audio_frames > 0;
        const bool has_text = !input.text_tokens.empty();
        if (has_audio && input.audio_features.size() !=
                static_cast<size_t>(input.audio_frames) * 80) {
            throw std::invalid_argument(
                "MiniCPM-o duplex Mel geometry is invalid");
        }
        if (!has_audio && !has_text) {
            throw std::invalid_argument(
                "MiniCPM-o duplex step has no input");
        }
        if (input.max_new_speak_tokens < 2) {
            throw std::invalid_argument(
                "MiniCPM-o duplex generation requires at least two token slots");
        }

        std::lock_guard<std::mutex> lock(model_mutex);
        MfqCudaGuard guard(
            g_layer_placement.primary_device());
        if (!session) {
            throw std::runtime_error(
                "MiniCPM-o duplex session is not prepared");
        }
        mfq_tensor_backend::Tensor audio_features;
        if (has_audio) {
            audio_features = mfq_tensor_backend::tensor(
                input.audio_features,
                mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kFloat32))
                .reshape({1, 80, input.audio_frames});
        }
        mfq_tensor_backend::Tensor text_ids;
        if (has_text) {
            text_ids = mfq_tensor_backend::tensor(
                input.text_tokens,
                mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64))
                .reshape({1, static_cast<int64_t>(input.text_tokens.size())});
        }
        const auto started = std::chrono::steady_clock::now();
        auto result = session->run_step(
            {}, {}, {}, {}, audio_features,
            input.audio_prefix_extra_frames,
            input.audio_suffix_extra_frames,
            text_ids,
            input.max_new_speak_tokens,
            input.force_listen,
            input.force_speak);

        MfqDuplexStepResult response;
        auto generated = result.generated_ids
            .to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt64).contiguous().reshape({-1});
        const auto * generated_data = generated.template data_ptr<int64_t>();
        response.generated_tokens.assign(
            generated_data, generated_data + generated.numel());
        auto codes = result.tts_codes
            .to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt32).contiguous().reshape({-1});
        const auto * code_data = codes.template data_ptr<int32_t>();
        response.audio_tokens.assign(
            code_data, code_data + codes.numel());
        response.is_listen = result.is_listen;
        response.end_of_turn = result.end_of_turn;
        response.tts_force_flush = result.tts_force_flush;
        response.audio_chunk_index = session->audio_chunk_index;
        response.language_cache_position = runtime.language.cache_pos;
        response.audio_cache_position = runtime.audio.cache_length();
        response.tts_cache_position = runtime.tts.cache_position;
        response.inference_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count();
        return response;
    };
    backend.stop = [&]() {
        std::lock_guard<std::mutex> lock(model_mutex);
        MfqCudaGuard guard(
            g_layer_placement.primary_device());
        session.reset();
        runtime.language.reset(1);
        runtime.audio.reset();
        runtime.tts.reset(1);
        mfq_cuda_synchronize();
    };
    return backend;
}

#include "diagnostics/flash_next_mtp.h"

template <typename Model>
static int run_flash_next_check(Model& model) {
    namespace tb=mfq_tensor_backend;
    MFQ_RUNTIME_CHECK(
        Model::is_flash_next && model.vocab_size() >= 8 &&
            model.max_position_embeddings() >= 16,
        "Flash-Next diagnostic requires a Flash-Next text graph, vocab>=8 and context>=16");
    auto ids=tb::tensor(std::vector<int64_t>{1,2,3,4,5,6,7},
        tb::TensorOptions().device(tb::kCUDA).dtype(tb::kInt64)).reshape({1,7});
    const auto json_tensor=[](const tb::Tensor& value) {
        auto host=value.to(tb::kFloat32).contiguous().cpu();
        return nlohmann::json{{"shape",host.sizes().vec()},
            {"data",std::vector<float>(host.template data_ptr<float>(),host.template data_ptr<float>()+host.numel())}};
    };
    nlohmann::json result;
    model.reset(1);
    result["full"]=json_tensor(model.forward(ids));
    model.reset(1);
    std::vector<tb::Tensor> pieces;
    for (auto [begin,count] : std::vector<std::pair<int64_t,int64_t>>{{0,2},{2,1},{3,4}})
        pieces.push_back(model.forward(ids.narrow(1,begin,count)));
    result["chunked"]=json_tensor(tb::cat(pieces,1));
    for (bool accept : {false,true}) {
        const std::string name=accept ? "committed" : "rejected";
        model.reset(1);
        model.forward(ids.narrow(1,0,2));
        auto hidden=model.hidden_forward(ids.narrow(1,2,2),mfq_nullopt,mfq_nullopt,nullptr,mfq_nullopt,nullptr,1);
        result[name+"_verify"]=json_tensor(model.logits_from_hidden(hidden));
        if (accept) model.commit_speculative(); else model.rollback_speculative();
        MFQ_RUNTIME_CHECK(model.cache_pos==(accept?4:3),"Flash-Next transaction cache position mismatch");
        result[name]=json_tensor(model.forward(ids.narrow(1,5,1)));
        model.reset(1);
        model.forward(ids.narrow(1,0,2));
        model.forward(ids.narrow(1,2,accept?2:1));
        result[name+"_reference"]=json_tensor(model.forward(ids.narrow(1,5,1)));
    }
    model.reset(1);
    result["reset"]=json_tensor(model.forward(ids));
    if constexpr (Model::is_qwen4) {
        auto positions=tb::stack({ids.reshape({7})-1,ids.reshape({7})+1,ids.reshape({7})+3},0);
        model.reset(1);
        result["axis_full"]=json_tensor(model.logits_from_hidden(model.hidden_forward(ids,positions)));
        model.reset(1);pieces.clear();
        auto positions4=tb::cat({positions.narrow(0,0,1)+11,positions},0);
        for (auto [begin,count] : std::vector<std::pair<int64_t,int64_t>>{{0,2},{2,1},{3,4}})
            pieces.push_back(model.logits_from_hidden(model.hidden_forward(ids.narrow(1,begin,count),positions4.narrow(-1,begin,count))));
        result["axis_chunked"]=json_tensor(tb::cat(pieces,1));
        model.reset(1);
        result["batch"]=json_tensor(model.forward(ids.repeat({2,1})));
        result["batch_reset"]=json_tensor(model.forward(ids));
        model.reset(1);result["last"]=json_tensor(model.last_logits(ids));
    }
    result["architecture"] = model.graph.backbone;
    std::cout << "flash_next_check " << result.dump() << '\n';
    return 0;
}

int mfq::cuda::run_runtime(int argc, char ** argv) {
    struct ModelParallelCollectiveCleanup {
        ~ModelParallelCollectiveCleanup() {
            g_model_parallel_collectives.reset();
        }
    } model_parallel_collective_cleanup;
    try {
        std::string model_path, config_path, ids_arg, ids_file;
        std::string minicpmo_input_prefix, minicpmo_output_prefix;
        std::string minicpmo_duplex_input_prefix;
        std::string minicpmo_duplex_output_prefix;
        std::string check_linear, check_linear_cpu, check_linear_gate, kl_base;
        std::string kl_save_logits_f16;
        std::string check_tp_linear;
        std::string check_ep_moe;
        std::string check_tp_axis_arg = "output";
        std::string check_linear_group, check_q8_embedding;
        std::string check_gdn_input, check_gdn_output, check_gdn_state;
        std::string check_linear_conv_input, check_linear_conv_output;
        std::string check_mfe_tensor, check_dsv4_output_a;
        std::string kl_chunks_sequence_arg, kl_mmq_sequence_arg;
        std::string kl_evaluator_arg = "optimized";
        std::string kl_mmq_arg = "default";
        std::string prefill_sweep_arg, check_moe_tokens = "1,2,4,8,16,32,64,128,256";
        std::string block_trace_reference, block_trace_output;
        std::string transport_host = "127.0.0.1";
        std::string tokenizer_model, runtime_model_name = "mfq-model", transport_api_key;
        std::string check_tokenizer_text =
            "MFQ tokenizer check: hello, world! <think>";
        std::string runtime_sampling_profile;
        std::string cpu_offload_layers_arg;
        std::string moe_cache_profile_path;
        std::string tensor_parallel_arg;
        std::string tensor_split_arg;
        std::string expert_parallel_arg;
        std::string expert_split_arg;
        std::string layer_parallel_arg;
        std::string layer_split_arg;
        double moe_gpu_cache_gb = 0.0;
        int gen = 16;
        std::string bench_qwen35_mtp;
        int bench_qwen35_mtp_reps = 3;
        int cpu_threads = 0;
        int transport_port = 8080;
        int continuous_batching = 0;
        int64_t prefill_chunk_size = 2048;
        int64_t context_size = 0;
        int64_t minicpmo_tts_steps = 0;
        int64_t minicpmo_duplex_steps = 0;
        int64_t minicpmo_duplex_max_speak_tokens = 20;
        int64_t minicpmo_duplex_seed = 0;
        int64_t minicpmo_eval_vision_batch_size = 16;
        int kl_chunks = -1;
        int kl_score_count = -1;
        int64_t kl_n_batch = 0;
        KlReferenceContract kl_reference_contract;
        int kl_stream_layers = 0;
        int kl_stream_batch = 1;
        int64_t block_trace_start = 0;
        int64_t block_trace_count = 0;
        int prefill_repeat = 0;
        int prefill_sweep_reps = 5;
        int check_linear_m = 1;
        int check_linear_reps = 200;
        int check_tp_m = 1;
        int check_ep_moe_tokens = 1;
        int check_ep_moe_routes = 2;
        int check_gdn_tokens = 512;
        int check_gdn_q_heads = 16;
        int check_gdn_v_heads = 32;
        int check_gdn_head_dim = 128;
        int check_linear_conv_tokens = 512;
        int check_linear_conv_q_heads = 16;
        int check_linear_conv_v_heads = 32;
        int check_linear_conv_key_dim = 128;
        int check_linear_conv_value_dim = 128;
        int check_linear_conv_kernel = 4;
        int check_gemma_geglu_layer = -1;
        int check_gemma_geglu_reps = 100;
        int check_moe_layer = -1;
        int check_moe_reps = 100;
        int check_mfe_tokens = 1;
        int check_mfe_routes = 2;
        int check_mfe_reps = 100;
        int check_mfe_split_width = 0;
        int check_dsv4_output_a_batch = 1;
        int check_dsv4_output_a_reps = 200;
        int check_attention_decode = 0;
        int check_attention_reps = 200;
        int check_attention_head_dim = 256;
        int check_attention_window = 4096;
        int compare_mma_decode_steps = 1;
        int compare_mma_decode_planned_len = 0;
        bool profile = false;
        bool check_backend_bf16_add = false;
        bool check_backend_argmax = false;
        bool compare_mma_attention = false;
        bool compare_decode_splitk = false;
        bool compare_mma_decode = false;
        bool compare_nvq_vec4 = false;
        bool check_gemma4_swa = false;
        bool check_glm_dsa = false;
        bool check_dsv4_attention = false;
        bool check_dsv4_hc = false;
        bool check_deepseek_v41 = false;
        bool check_text_session_state = false;
        bool check_qwen35_mtp = false;
        bool check_continuous_batching = false;
        bool check_flash_next = false;
        bool check_flash_next_mtp = false;
        bool compare_dsv4_hc_ops = false;
        bool compare_dsv4_hc_model = false;
        bool check_attention_swa_decode = false;
        bool check_mfe_routed_input = false;
        bool check_mfe_benchmark_only = false;
        bool parallel_test_duplicates = false;
        bool transport_mode = false;
        bool stdio_mode = false;
        bool check_runtime_assets = false;
        bool check_mfq_container = false;
        bool minicpmo_duplex_greedy = false;
        bool minicpmo_eval_batch = false;
        bool n_gpu_layers_set = false;
        bool cpu_threads_set = false;
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            bool parsed_option = true;
            if (a == "--model" && i + 1 < argc) model_path = argv[++i];
            else if (a == "--config" && i + 1 < argc) config_path = argv[++i];
            else if (a == "--ids" && i + 1 < argc) ids_arg = argv[++i];
            else if (a == "--ids-file" && i + 1 < argc) ids_file = argv[++i];
            else if (a == "--minicpmo-input-prefix" && i + 1 < argc) {
                minicpmo_input_prefix = argv[++i];
            }
            else if (a == "--minicpmo-output-prefix" && i + 1 < argc) {
                minicpmo_output_prefix = argv[++i];
            }
            else if (a == "--minicpmo-tts-steps" && i + 1 < argc) {
                minicpmo_tts_steps = std::stoll(argv[++i]);
            }
            else if (a == "--minicpmo-eval-batch") {
                minicpmo_eval_batch = true;
            }
            else if (a == "--minicpmo-eval-vision-batch-size" &&
                    i + 1 < argc) {
                minicpmo_eval_vision_batch_size = std::stoll(argv[++i]);
            }
            else if (a == "--minicpmo-duplex-input-prefix" && i + 1 < argc) {
                minicpmo_duplex_input_prefix = argv[++i];
            }
            else if (a == "--minicpmo-duplex-output-prefix" && i + 1 < argc) {
                minicpmo_duplex_output_prefix = argv[++i];
            }
            else if (a == "--minicpmo-duplex-steps" && i + 1 < argc) {
                minicpmo_duplex_steps = std::stoll(argv[++i]);
            }
            else if (a == "--minicpmo-duplex-max-speak-tokens" && i + 1 < argc) {
                minicpmo_duplex_max_speak_tokens = std::stoll(argv[++i]);
            }
            else if (a == "--minicpmo-duplex-seed" && i + 1 < argc) {
                minicpmo_duplex_seed = std::stoll(argv[++i]);
            }
            else if (a == "--minicpmo-duplex-greedy") {
                minicpmo_duplex_greedy = true;
            }
            else if (a == "--check-linear" && i + 1 < argc) check_linear = argv[++i];
            else if (a == "--check-linear-cpu" && i + 1 < argc) {
                check_linear_cpu = argv[++i];
            }
            else if (a == "--check-tp-linear" && i + 1 < argc) {
                check_tp_linear = argv[++i];
            }
            else if (a == "--check-tp-axis" && i + 1 < argc) {
                check_tp_axis_arg = argv[++i];
            }
            else if (a == "--check-tp-m" && i + 1 < argc) {
                check_tp_m = std::stoi(argv[++i]);
            }
            else if ((a == "--check-ep-moe" ||
                    a == "--check-tp-moe") && i + 1 < argc) {
                check_ep_moe = argv[++i];
            }
            else if ((a == "--check-ep-moe-tokens" ||
                    a == "--check-tp-moe-tokens") && i + 1 < argc) {
                check_ep_moe_tokens = std::stoi(argv[++i]);
            }
            else if ((a == "--check-ep-moe-routes" ||
                    a == "--check-tp-moe-routes") && i + 1 < argc) {
                check_ep_moe_routes = std::stoi(argv[++i]);
            }
            else if (a == "--check-linear-group" && i + 1 < argc) {
                check_linear_group = argv[++i];
            }
            else if (a == "--check-q8-embedding" && i + 1 < argc) {
                check_q8_embedding = argv[++i];
            }
            else if (a == "--check-gdn-input" && i + 1 < argc) check_gdn_input = argv[++i];
            else if (a == "--check-gdn-output" && i + 1 < argc) check_gdn_output = argv[++i];
            else if (a == "--check-gdn-state" && i + 1 < argc) check_gdn_state = argv[++i];
            else if (a == "--check-gdn-tokens" && i + 1 < argc) check_gdn_tokens = std::stoi(argv[++i]);
            else if (a == "--check-gdn-q-heads" && i + 1 < argc) check_gdn_q_heads = std::stoi(argv[++i]);
            else if (a == "--check-gdn-v-heads" && i + 1 < argc) check_gdn_v_heads = std::stoi(argv[++i]);
            else if (a == "--check-gdn-head-dim" && i + 1 < argc) check_gdn_head_dim = std::stoi(argv[++i]);
            else if (a == "--check-linear-conv-input" && i + 1 < argc) check_linear_conv_input = argv[++i];
            else if (a == "--check-linear-conv-output" && i + 1 < argc) check_linear_conv_output = argv[++i];
            else if (a == "--check-linear-conv-tokens" && i + 1 < argc) check_linear_conv_tokens = std::stoi(argv[++i]);
            else if (a == "--check-linear-conv-q-heads" && i + 1 < argc) check_linear_conv_q_heads = std::stoi(argv[++i]);
            else if (a == "--check-linear-conv-v-heads" && i + 1 < argc) check_linear_conv_v_heads = std::stoi(argv[++i]);
            else if (a == "--check-linear-conv-key-dim" && i + 1 < argc) check_linear_conv_key_dim = std::stoi(argv[++i]);
            else if (a == "--check-linear-conv-value-dim" && i + 1 < argc) check_linear_conv_value_dim = std::stoi(argv[++i]);
            else if (a == "--check-linear-conv-kernel" && i + 1 < argc) check_linear_conv_kernel = std::stoi(argv[++i]);
            else if (a == "--check-linear-m" && i + 1 < argc) check_linear_m = std::stoi(argv[++i]);
            else if (a == "--check-linear-reps" && i + 1 < argc) {
                check_linear_reps = std::stoi(argv[++i]);
            }
            else if (a == "--check-linear-gate" && i + 1 < argc) check_linear_gate = argv[++i];
            else if (a == "--check-gemma-geglu-layer" && i + 1 < argc) check_gemma_geglu_layer = std::stoi(argv[++i]);
            else if (a == "--check-gemma-geglu-reps" && i + 1 < argc) check_gemma_geglu_reps = std::stoi(argv[++i]);
            else if (a == "--check-moe-layer" && i + 1 < argc) check_moe_layer = std::stoi(argv[++i]);
            else if (a == "--check-moe-tokens" && i + 1 < argc) check_moe_tokens = argv[++i];
            else if (a == "--check-moe-reps" && i + 1 < argc) check_moe_reps = std::stoi(argv[++i]);
            else if (a == "--check-mfe-tensor" && i + 1 < argc) check_mfe_tensor = argv[++i];
            else if (a == "--check-mfe-tokens" && i + 1 < argc) check_mfe_tokens = std::stoi(argv[++i]);
            else if (a == "--check-mfe-routes" && i + 1 < argc) check_mfe_routes = std::stoi(argv[++i]);
            else if (a == "--check-mfe-reps" && i + 1 < argc) check_mfe_reps = std::stoi(argv[++i]);
            else if (a == "--check-mfe-split-width" && i + 1 < argc) check_mfe_split_width = std::stoi(argv[++i]);
            else if (a == "--check-mfe-routed-input") check_mfe_routed_input = true;
            else if (a == "--check-mfe-benchmark-only") check_mfe_benchmark_only = true;
            else if (a == "--check-dsv4-output-a" && i + 1 < argc) check_dsv4_output_a = argv[++i];
            else if (a == "--check-dsv4-output-a-batch" && i + 1 < argc) check_dsv4_output_a_batch = std::stoi(argv[++i]);
            else if (a == "--check-dsv4-output-a-reps" && i + 1 < argc) check_dsv4_output_a_reps = std::stoi(argv[++i]);
            else if (a == "--check-attention-decode" && i + 1 < argc) check_attention_decode = std::stoi(argv[++i]);
            else if (a == "--check-attention-reps" && i + 1 < argc) check_attention_reps = std::stoi(argv[++i]);
            else if (a == "--check-attention-head-dim" && i + 1 < argc) check_attention_head_dim = std::stoi(argv[++i]);
            else if (a == "--check-attention-window" && i + 1 < argc) check_attention_window = std::stoi(argv[++i]);
            else if (a == "--check-attention-swa-decode") check_attention_swa_decode = true;
            else if (a == "--check-gemma4-swa") check_gemma4_swa = true;
            else if (a == "--check-glm-dsa") check_glm_dsa = true;
            else if (a == "--check-dsv4-attention") check_dsv4_attention = true;
            else if (a == "--check-dsv4-hc") check_dsv4_hc = true;
            else if (a == "--check-deepseek-v41") check_deepseek_v41 = true;
            else if (a == "--check-text-session-state") {
                check_text_session_state = true;
            }
            else if (a == "--check-qwen35-mtp") check_qwen35_mtp = true;
            else if (a == "--check-flash-next") check_flash_next = true;
            else if (a == "--check-flash-next-mtp") check_flash_next_mtp = true;
            else if (a == "--bench-qwen35-mtp" && i + 1 < argc) bench_qwen35_mtp = argv[++i];
            else if (a == "--bench-qwen35-mtp-reps" && i + 1 < argc) bench_qwen35_mtp_reps = std::stoi(argv[++i]);
            else if (a == "--compare-dsv4-hc-ops") compare_dsv4_hc_ops = true;
            else if (a == "--compare-dsv4-hc-model") compare_dsv4_hc_model = true;
            else parsed_option = false;
            if (parsed_option) continue;

            if (a == "--kl-base" && i + 1 < argc) kl_base = argv[++i];
            else if (a == "--kl-save-logits-f16" && i + 1 < argc) {
                kl_save_logits_f16 = argv[++i];
            }
            else if (a == "--kl-chunks" && i + 1 < argc) kl_chunks = std::stoi(argv[++i]);
            else if (a == "--kl-score-count" && i + 1 < argc) {
                kl_score_count = std::stoi(argv[++i]);
            }
            else if (a == "--kl-n-batch" && i + 1 < argc) {
                kl_n_batch = std::stoll(argv[++i]);
            }
            else if (a == "--kl-reference-n-batch" && i + 1 < argc) {
                kl_reference_contract.n_batch = std::stoll(argv[++i]);
            }
            else if (a == "--kl-reference-n-ubatch" && i + 1 < argc) {
                kl_reference_contract.n_ubatch = std::stoll(argv[++i]);
            }
            else if (a == "--kl-chunks-sequence" && i + 1 < argc) {
                kl_chunks_sequence_arg = argv[++i];
            }
            else if (a == "--kl-evaluator" && i + 1 < argc) {
                kl_evaluator_arg = argv[++i];
            }
            else if (a == "--kl-mmq" && i + 1 < argc) {
                kl_mmq_arg = argv[++i];
            }
            else if (a == "--kl-mmq-sequence" && i + 1 < argc) {
                kl_mmq_sequence_arg = argv[++i];
            }
            else if (a == "--kl-stream-layers" && i + 1 < argc) {
                kl_stream_layers = std::stoi(argv[++i]);
            }
            else if (a == "--kl-stream-batch" && i + 1 < argc) {
                kl_stream_batch = std::stoi(argv[++i]);
            }
            else if (a == "--compare-block-trace" && i + 1 < argc) block_trace_reference = argv[++i];
            else if (a == "--dump-block-trace" && i + 1 < argc) block_trace_output = argv[++i];
            else if (a == "--dump-block-trace-start" && i + 1 < argc) {
                block_trace_start = std::stoll(argv[++i]);
            }
            else if (a == "--dump-block-trace-count" && i + 1 < argc) {
                block_trace_count = std::stoll(argv[++i]);
            }
            else if (a == "--prefill-repeat" && i + 1 < argc) prefill_repeat = std::stoi(argv[++i]);
            else if (a == "--prefill-sweep" && i + 1 < argc) prefill_sweep_arg = argv[++i];
            else if (a == "--prefill-sweep-reps" && i + 1 < argc) prefill_sweep_reps = std::stoi(argv[++i]);
            else if (a == "--gen" && i + 1 < argc) gen = std::stoi(argv[++i]);
            else if ((a == "--threads" || a == "-t") && i + 1 < argc) {
                cpu_threads = std::stoi(argv[++i]);
                cpu_threads_set = true;
            }
            else if (a == "--transport" && i + 1 < argc) {
                const std::string transport = argv[++i];
                if (transport_mode) {
                    throw std::runtime_error("runtime transport was specified more than once");
                }
                if (transport != "stdio" && transport != "http") {
                    throw std::runtime_error("--transport must be stdio or http");
                }
                transport_mode = true;
                stdio_mode = transport == "stdio";
            }
            else if (a == "--host" && i + 1 < argc) transport_host = argv[++i];
            else if (a == "--port" && i + 1 < argc) transport_port = std::stoi(argv[++i]);
            else if (a == "--continuous-batching" && i + 1 < argc) {
                continuous_batching = std::stoi(argv[++i]);
            }
            else if (a == "--prefill-chunk-size" && i + 1 < argc) {
                prefill_chunk_size = std::stoll(argv[++i]);
            }
            else if (a == "--check-continuous-batching") {
                check_continuous_batching = true;
            }
            else if (a == "--ctx-size" && i + 1 < argc) context_size = std::stoll(argv[++i]);
            else if (a == "--cpu-offload-layers" && i + 1 < argc) {
                cpu_offload_layers_arg = argv[++i];
            }
            else if ((a == "--n-gpu-layers" || a == "-ngl") && i + 1 < argc) {
                g_n_gpu_layers = std::stoi(argv[++i]);
                n_gpu_layers_set = true;
            }
            else if (a == "--moe-gpu-cache-gb" && i + 1 < argc) {
                moe_gpu_cache_gb = std::stod(argv[++i]);
            }
            else if (a == "--moe-cache-profile" && i + 1 < argc) {
                moe_cache_profile_path = argv[++i];
            }
            else if ((a == "--tensor-parallel" ||
                    a == "--tensor-split" ||
                    a == "--expert-parallel" ||
                    a == "--expert-split") && i + 1 < argc) {
                std::string * destination =
                    a == "--tensor-parallel" ? &tensor_parallel_arg :
                    a == "--tensor-split" ? &tensor_split_arg :
                    a == "--expert-parallel" ? &expert_parallel_arg :
                    &expert_split_arg;
                *destination = argv[++i];
            }
            else if (a == "--layer-parallel" && i + 1 < argc) {
                layer_parallel_arg = argv[++i];
            }
            else if (a == "--layer-split" && i + 1 < argc) {
                layer_split_arg = argv[++i];
            }
            else if (a == "--parallel-test-duplicates" ||
                    a == "--tensor-parallel-test-duplicates") {
                parallel_test_duplicates = true;
            }
            else if (a == "--tokenizer" && i + 1 < argc) tokenizer_model = argv[++i];
            else if (a == "--check-runtime-assets") check_runtime_assets = true;
            else if (a == "--check-mfq-container") check_mfq_container = true;
            else if (a == "--check-tokenizer-text" && i + 1 < argc) {
                check_tokenizer_text = argv[++i];
            }
            else if (a == "--model-name" && i + 1 < argc) runtime_model_name = argv[++i];
            else if (a == "--api-key" && i + 1 < argc) transport_api_key = argv[++i];
            else if (a == "--sampling-profile" && i + 1 < argc) {
                runtime_sampling_profile = argv[++i];
            }
            else if (a == "--profile" ||
                    a == "--check-backend-bf16-add" ||
                    a == "--check-backend-argmax") {
                if (a == "--profile") profile = true;
                else if (a == "--check-backend-bf16-add") {
                    check_backend_bf16_add = true;
                } else {
                    check_backend_argmax = true;
                }
            }
            else if (a == "--compare-mma-attention") compare_mma_attention = true;
            else if (a == "--compare-decode-splitk") compare_decode_splitk = true;
            else if (a == "--compare-mma-decode") compare_mma_decode = true;
            else if (a == "--compare-mma-decode-steps" && i + 1 < argc) compare_mma_decode_steps = std::stoi(argv[++i]);
            else if (a == "--compare-mma-decode-planned-len" && i + 1 < argc) compare_mma_decode_planned_len = std::stoi(argv[++i]);
            else if (a == "--compare-nvq-vec4" || a == "--compare-niq-vec4") compare_nvq_vec4 = true;
            else {
                std::cerr << "usage: mfq-runtime --model MODEL_PATH [--config config.json] "
                             "(--ids 1,2,3 --gen 128 | --check-qwen35-mtp | --check-continuous-batching | --bench-qwen35-mtp ordinary|mtp | --transport stdio|http "
                             "[--host 127.0.0.1 --port 8080 --ctx-size 32768 --model-name name --tokenizer tokenizer.gguf "
                             "--continuous-batching 8 --prefill-chunk-size 2048 "
                             "--tensor-parallel 0,1 --tensor-split 1,1 "
                             "--expert-parallel 0,1 --expert-split 1,1 "
                             "--layer-parallel 0,1 --layer-split 1,1 "
                             "--n-gpu-layers 60 --threads 32 --cpu-offload-layers 0-7,12 --moe-gpu-cache-gb 8 "
                             "--moe-cache-profile profile.json "
                             "--api-key key --sampling-profile profile.json] | --kl-base reference.bin "
                             "[--kl-evaluator optimized|legacy --kl-chunks -1 "
                             "--kl-score-count N --kl-n-batch N "
                             "--kl-reference-n-batch N "
                             "--kl-reference-n-ubatch N --kl-mmq default|fp16|nint8_1] "
                             "[--kl-save-logits-f16 PATH])\n";
                return 2;
            }
        }
        if (stdio_mode) {
            prepare_mfq_stdio_transport();
        }
        if (cpu_threads_set && cpu_threads <= 0) {
            throw std::runtime_error("--threads must be positive");
        }
        if (continuous_batching < 0) {
            throw std::runtime_error(
                "--continuous-batching cannot be negative");
        }
        if (continuous_batching > 0 && !transport_mode) {
            throw std::runtime_error(
                "--continuous-batching requires --transport stdio|http");
        }
        if (prefill_chunk_size <= 0) {
            throw std::runtime_error(
                "--prefill-chunk-size must be positive");
        }
        if (cpu_threads > 0) {
            mfq_set_num_threads(cpu_threads);
        }
        if (check_backend_bf16_add) {
            return run_backend_bf16_add_check(4096, 10000);
        }
        if (check_backend_argmax) {
            return run_backend_argmax_check(151748, 2000);
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
        if (!check_linear_cpu.empty()) {
            if (model_path.empty()) {
                throw std::runtime_error("--check-linear-cpu requires --model");
            }
            int gate_mode = 0;
            if (check_linear_gate == "sigmoid") gate_mode = 1;
            else if (check_linear_gate == "silu") gate_mode = 2;
            else if (!check_linear_gate.empty()) {
                throw std::runtime_error("--check-linear-gate must be sigmoid or silu");
            }
            return run_cpu_linear_check(
                model_path, check_linear_cpu, check_linear_m,
                gate_mode, check_linear_reps);
        }
        if (!check_linear.empty()) {
            if (model_path.empty()) throw std::runtime_error("--check-linear requires --model");
            int gate_mode = 0;
            if (check_linear_gate == "sigmoid") gate_mode = 1;
            else if (check_linear_gate == "silu") gate_mode = 2;
            else if (!check_linear_gate.empty()) {
                throw std::runtime_error("--check-linear-gate must be sigmoid or silu");
            }
            return run_linear_check(
                model_path, check_linear, check_linear_m, gate_mode,
                check_linear_reps);
        }
        if (!check_tp_linear.empty()) {
            if (model_path.empty()) {
                throw std::runtime_error(
                    "--check-tp-linear requires --model");
            }
            const TensorParallelAxis axis =
                check_tp_axis_arg == "output"
                ? TensorParallelAxis::Output
                : check_tp_axis_arg == "input"
                ? TensorParallelAxis::Input
                : throw std::runtime_error(
                    "--check-tp-axis must be output or input");
            return run_tensor_parallel_linear_check(
                model_path, check_tp_linear,
                axis, check_tp_m);
        }
        if (!check_ep_moe.empty()) {
            if (model_path.empty()) {
                throw std::runtime_error(
                    "--check-ep-moe requires --model");
            }
            return run_expert_parallel_moe_check(
                model_path, check_ep_moe,
                check_ep_moe_tokens,
                check_ep_moe_routes);
        }
        if (!check_linear_group.empty()) {
            if (model_path.empty()) {
                throw std::runtime_error("--check-linear-group requires --model");
            }
            return run_linear_group_check(
                model_path, check_linear_group, check_linear_m,
                check_linear_reps);
        }
        if (!check_q8_embedding.empty()) {
            if (model_path.empty()) {
                throw std::runtime_error("--check-q8-embedding requires --model");
            }
            return run_q8_embedding_check(model_path, check_q8_embedding);
        }
        if (!check_gdn_input.empty()) {
            if (check_gdn_output.empty() || check_gdn_state.empty()) {
                throw std::runtime_error(
                    "--check-gdn-input requires --check-gdn-output and --check-gdn-state");
            }
            return run_gdn_operator_check(
                check_gdn_input, check_gdn_output, check_gdn_state,
                check_gdn_tokens, check_gdn_q_heads,
                check_gdn_v_heads, check_gdn_head_dim);
        }
        if (!check_linear_conv_input.empty()) {
            if (check_linear_conv_output.empty()) {
                throw std::runtime_error(
                    "--check-linear-conv-input requires --check-linear-conv-output");
            }
            return run_linear_conv_operator_check(
                check_linear_conv_input, check_linear_conv_output,
                check_linear_conv_tokens, check_linear_conv_q_heads,
                check_linear_conv_v_heads, check_linear_conv_key_dim,
                check_linear_conv_value_dim, check_linear_conv_kernel,
                1.0e-6);
        }
        if (check_gemma_geglu_layer >= 0) {
            if (model_path.empty()) {
                throw std::runtime_error("--check-gemma-geglu-layer requires --model");
            }
            return run_gemma_geglu_check(
                model_path, check_gemma_geglu_layer, check_gemma_geglu_reps);
        }
        if (check_mfq_container) {
            if (model_path.empty()) {
                throw std::runtime_error(
                    "--check-mfq-container requires --model");
            }
            auto model_source = mfq::open_model_source(model_path);
            const auto& mfq = *model_source;
            std::cout << "mfq_container_check=ok"
                      << " shards=" << mfq.source_paths().size()
                      << " tensors=" << mfq.tensors().size()
                      << " assets=" << mfq.assets().size()
                      << "\n";
            return 0;
        }
        if (check_runtime_assets) {
            if (model_path.empty()) {
                throw std::runtime_error(
                    "--check-runtime-assets requires --model");
            }
            auto model_source = mfq::open_model_source(model_path);
            const auto& mfq = *model_source;
            const auto config = mfq::models::ModelConfig::from_json(
                mfq::cuda::load_model_config_json(mfq, config_path));
            if (!mfq.has_asset(mfq::cuda::kTokenizerGgufAsset)) {
                throw std::runtime_error(
                    "model source has no tokenizer GGUF");
            }
            const auto tokenizer_blob =
                read_asset(mfq, mfq::cuda::kTokenizerGgufAsset);
            const auto embedded =
                probe_mfq_tokenizer(tokenizer_blob, check_tokenizer_text);
            if (embedded.vocab_size != config.vocab_size) {
                throw std::runtime_error(
                    "embedded tokenizer/model vocabulary mismatch: tokenizer=" +
                    std::to_string(embedded.vocab_size) + " model=" +
                    std::to_string(config.vocab_size));
            }
            std::cout << "runtime_assets_check=ok"
                      << " config=embedded"
                      << " tokenizer=embedded"
                      << " vocab_size=" << embedded.vocab_size
                      << " chat_template="
                      << (!embedded.chat_template.empty() ? 1 : 0)
                      << " bos=" << embedded.bos_token
                      << " eos=" << embedded.eos_token
                      << " eot=" << embedded.eot_token
                      << " pad=" << embedded.pad_token
                      << " token_count=" << embedded.tokens.size()
                      << "\n";
            if (!tokenizer_model.empty()) {
                const auto external =
                    probe_mfq_tokenizer(tokenizer_model, check_tokenizer_text);
                if (external.vocab_size != embedded.vocab_size ||
                    external.bos_token != embedded.bos_token ||
                    external.eos_token != embedded.eos_token ||
                    external.eot_token != embedded.eot_token ||
                    external.pad_token != embedded.pad_token ||
                    external.chat_template != embedded.chat_template ||
                    external.tokens != embedded.tokens) {
                    throw std::runtime_error(
                        "embedded tokenizer differs from external GGUF");
                }
                std::cout << "runtime_assets_external_match=ok\n";
            }
            return 0;
        }
        if (check_moe_layer >= 0) {
            if (model_path.empty()) {
                throw std::runtime_error("--check-moe-layer requires --model");
            }
            return run_moe_check(
                model_path, config_path, check_moe_layer, parse_ids(check_moe_tokens), check_moe_reps);
        }
        if (!check_mfe_tensor.empty()) {
            if (model_path.empty()) {
                throw std::runtime_error("--check-mfe-tensor requires --model");
            }
            KlMmqScope check_kl_mmq_scope(
                parse_kl_mmq_mode(kl_mmq_arg));
            const auto tensor_names =
                parse_tensor_names(check_mfe_tensor);
            if (g_moe_expert_cache &&
                    tensor_names.size() != 1) {
                throw std::runtime_error(
                    "cached --check-mfe-tensor accepts one tensor "
                    "per invocation");
            }
            for (const auto & tensor_name : tensor_names) {
                const int result = run_mfe_tensor_check(
                    model_path, tensor_name, check_mfe_tokens,
                    check_mfe_routes, check_mfe_reps,
                    check_mfe_split_width, check_mfe_routed_input,
                    check_mfe_benchmark_only);
                if (result != 0) return result;
            }
            return 0;
        }
        if (!check_dsv4_output_a.empty()) {
            if (model_path.empty()) {
                throw std::runtime_error("--check-dsv4-output-a requires --model");
            }
            return run_dsv4_output_a_check(
                model_path, check_dsv4_output_a,
                check_dsv4_output_a_batch, check_dsv4_output_a_reps);
        }
        if (check_attention_decode > 0) {
            return run_attention_decode_check(
                check_attention_decode, check_attention_reps,
                check_attention_head_dim, check_attention_swa_decode,
                check_attention_window);
        }
        if (check_gemma4_swa) {
            return run_gemma4_swa_check(check_attention_reps);
        }
        if (check_glm_dsa) {
            return run_glm_dsa_check(check_attention_reps);
        }
        if (check_dsv4_attention) {
            return run_dsv4_attention_check(check_attention_reps);
        }
        if (check_dsv4_hc) {
            return run_dsv4_hc_check(check_attention_reps);
        }
        if (check_deepseek_v41) {
            return mfq::cuda::deepseek_v41_runtime::run_self_check();
        }
        if (check_text_session_state) {
            return run_text_session_state_check();
        }
        if (!minicpmo_duplex_input_prefix.empty()) {
            if (model_path.empty() || minicpmo_duplex_output_prefix.empty()) {
                throw std::runtime_error(
                    "--minicpmo-duplex-input-prefix requires --model and "
                    "--minicpmo-duplex-output-prefix");
            }
            if (!minicpmo_input_prefix.empty() || transport_mode ||
                    !ids_arg.empty() || !ids_file.empty() ||
                    !kl_base.empty() || !prefill_sweep_arg.empty()) {
                throw std::runtime_error(
                    "MiniCPM-o duplex mode cannot be combined with "
                    "composite, token, transport, KL, or prefill modes");
            }
            return run_minicpmo45_duplex(
                model_path, config_path,
                minicpmo_duplex_input_prefix,
                minicpmo_duplex_output_prefix,
                context_size, minicpmo_duplex_steps,
                minicpmo_duplex_max_speak_tokens,
                minicpmo_duplex_greedy,
                minicpmo_duplex_seed);
        }
        if (minicpmo_eval_batch) {
            if (model_path.empty()) {
                throw std::runtime_error(
                    "--minicpmo-eval-batch requires --model");
            }
            if (!minicpmo_input_prefix.empty() || transport_mode ||
                    !ids_arg.empty() || !ids_file.empty() ||
                    !kl_base.empty() || !prefill_sweep_arg.empty() ||
                    !minicpmo_duplex_input_prefix.empty()) {
                throw std::runtime_error(
                    "MiniCPM-o eval mode cannot be combined with "
                    "composite, duplex, token, transport, KL, or prefill modes");
            }
            if (minicpmo_eval_vision_batch_size <= 0) {
                throw std::runtime_error(
                    "--minicpmo-eval-vision-batch-size must be positive");
            }
            return run_minicpmo45_eval_batch(
                model_path, config_path, context_size,
                minicpmo_eval_vision_batch_size);
        }
        if (!minicpmo_input_prefix.empty()) {
            if (model_path.empty() || minicpmo_output_prefix.empty()) {
                throw std::runtime_error(
                    "--minicpmo-input-prefix requires --model and "
                    "--minicpmo-output-prefix");
            }
            if (transport_mode || !ids_arg.empty() || !ids_file.empty() ||
                    !kl_base.empty() || !prefill_sweep_arg.empty()) {
                throw std::runtime_error(
                    "MiniCPM-o composite mode cannot be combined with "
                    "token, transport, KL, or prefill modes");
            }
            if (minicpmo_tts_steps < 0) {
                throw std::runtime_error(
                    "--minicpmo-tts-steps must be non-negative");
            }
            return run_minicpmo45_composite(
                model_path, config_path,
                minicpmo_input_prefix, minicpmo_output_prefix,
                context_size, minicpmo_tts_steps);
        }
        if (model_path.empty() ||
            (!transport_mode && !check_qwen35_mtp &&
                !check_continuous_batching && !check_flash_next &&
                !check_flash_next_mtp && bench_qwen35_mtp.empty() &&
                ids_arg.empty() && ids_file.empty() &&
                kl_base.empty() && prefill_sweep_arg.empty())) {
            std::cerr << "usage: mfq-runtime --model MODEL_PATH [--config config.json] "
                         "(--ids 1,2,3 --gen 128 | --check-qwen35-mtp | --check-continuous-batching | --bench-qwen35-mtp ordinary|mtp | --minicpmo-eval-batch "
                         "[--minicpmo-eval-vision-batch-size 16] | --transport stdio|http "
                         "[--host 127.0.0.1 --port 8080 --ctx-size 32768 --model-name name --tokenizer tokenizer.gguf "
                         "--api-key key] | --kl-base reference.bin "
                         "[--kl-evaluator optimized|legacy --kl-chunks -1 "
                         "--kl-score-count N --kl-n-batch N "
                         "--kl-reference-n-batch N "
                         "--kl-reference-n-ubatch N --kl-mmq default|fp16|nint8_1] "
                         ")\n";
            return 2;
        }
        if (context_size < 0) throw std::runtime_error("--ctx-size must be positive");
        if (transport_mode && context_size == 0) context_size = 32768;
        if (!bench_qwen35_mtp.empty()) {
            MFQ_RUNTIME_CHECK(bench_qwen35_mtp == "ordinary" || bench_qwen35_mtp == "mtp",
                "--bench-qwen35-mtp expects ordinary or mtp");
            MFQ_RUNTIME_CHECK(!transport_mode && !check_qwen35_mtp && ids_arg.empty() && ids_file.empty() &&
                kl_base.empty() && prefill_sweep_arg.empty(), "MTP benchmark cannot combine execution modes");
            if (context_size == 0) context_size = 512;
            MFQ_RUNTIME_CHECK(gen >= 2 && context_size >= static_cast<int64_t>(gen) + 17 &&
                bench_qwen35_mtp_reps > 0 && bench_qwen35_mtp_reps <= 100,
                "MTP benchmark requires gen>=2, reps1-100 and context>=gen+17");
            std::cout << std::unitbuf;
        }
        if (check_qwen35_mtp) {
            if (context_size == 0) context_size = 512;
            if (context_size < 64) throw std::runtime_error("MTP correctness gate requires --ctx-size >= 64");
            std::cout << std::unitbuf;
        }
        if (check_continuous_batching) {
            MFQ_RUNTIME_CHECK(
                !transport_mode && !check_qwen35_mtp &&
                !check_flash_next && !check_flash_next_mtp &&
                bench_qwen35_mtp.empty() && ids_arg.empty() &&
                ids_file.empty() && kl_base.empty() &&
                prefill_sweep_arg.empty(),
                "continuous batching check cannot combine execution modes");
            if (context_size == 0) context_size = 512;
            if (context_size < 32) {
                throw std::runtime_error(
                    "continuous batching check requires --ctx-size >= 32");
            }
            std::cout << std::unitbuf;
        }
        if (!cpu_offload_layers_arg.empty()) {
            g_dsv4_cpu_offload_layers =
                parse_layer_ranges(cpu_offload_layers_arg);
            std::vector<int> ordered(
                g_dsv4_cpu_offload_layers.begin(),
                g_dsv4_cpu_offload_layers.end());
            std::sort(ordered.begin(), ordered.end());
            std::cerr << "cpu_offload_layers=";
            for (size_t index = 0; index < ordered.size(); ++index) {
                if (index) std::cerr << ',';
                std::cerr << ordered[index];
            }
            std::cerr << std::endl;
        }
        std::vector<int64_t> prefill_sweep_sizes;
        if (!prefill_sweep_arg.empty()) {
            prefill_sweep_sizes = parse_ids(prefill_sweep_arg);
            if (std::any_of(prefill_sweep_sizes.begin(), prefill_sweep_sizes.end(),
                            [](int64_t value) { return value <= 0; })) {
                throw std::runtime_error("--prefill-sweep values must be positive");
            }
            if (context_size == 0) {
                context_size = *std::max_element(prefill_sweep_sizes.begin(), prefill_sweep_sizes.end());
            }
        }
        if ((!block_trace_reference.empty() || !block_trace_output.empty()) &&
                context_size == 0) {
            context_size = (int64_t)(
                ids_file.empty() ? parse_ids(ids_arg) : load_ids_file(ids_file)).size();
        }
        std::vector<int64_t> kl_chunks_sequence;
        const KlEvaluator kl_evaluator =
            parse_kl_evaluator(kl_evaluator_arg);
        const KlMmqMode kl_mmq_mode =
            parse_kl_mmq_mode(kl_mmq_arg);
        if (kl_chunks == 0 || kl_chunks < -1) {
            throw std::runtime_error(
                "--kl-chunks must be positive or -1 for all chunks");
        }
        if (kl_score_count == 0 || kl_score_count < -1) {
            throw std::runtime_error(
                "--kl-score-count must be positive or -1 for the stored count");
        }
        if (kl_n_batch < 0) {
            throw std::runtime_error(
                "--kl-n-batch must be non-negative; 0 uses n_ctx");
        }
        if (kl_reference_contract.n_batch < 0 ||
                kl_reference_contract.n_ubatch < 0) {
            throw std::runtime_error(
                "KL reference n_batch and n_ubatch must be non-negative");
        }
        if ((kl_reference_contract.n_batch == 0) !=
                (kl_reference_contract.n_ubatch == 0)) {
            throw std::runtime_error(
                "--kl-reference-n-batch and --kl-reference-n-ubatch must "
                "be supplied together");
        }
        if (kl_reference_contract.n_ubatch >
                kl_reference_contract.n_batch) {
            throw std::runtime_error(
                "--kl-reference-n-ubatch must not exceed "
                "--kl-reference-n-batch");
        }
        const auto active_env = [](const char * name) {
            const char * value = std::getenv(name);
            return value != nullptr && value[0] != '\0';
        };
        if (!kl_base.empty() && active_env("MFQ_KL_WINDOW_M")) {
            throw std::runtime_error(
                "MFQ_KL_WINDOW_M is disabled because it silently changes "
                "the metric; use --kl-score-count");
        }
        std::vector<KlMmqMode> kl_mmq_sequence;
        if (!kl_mmq_sequence_arg.empty()) {
            if (kl_mmq_arg != "default") {
                throw std::runtime_error(
                    "--kl-mmq and --kl-mmq-sequence are mutually exclusive");
            }
            kl_mmq_sequence =
                parse_kl_mmq_sequence(kl_mmq_sequence_arg);
        }
        if (kl_mmq_mode != KlMmqMode::Default &&
                (kl_base.empty() ||
                 kl_evaluator != KlEvaluator::Optimized)) {
            throw std::runtime_error(
                "--kl-mmq nint8_1/fp16 requires "
                "--kl-base and --kl-evaluator optimized");
        }
        if (!kl_mmq_sequence.empty() &&
                (kl_base.empty() ||
                 kl_evaluator != KlEvaluator::Optimized)) {
            throw std::runtime_error(
                "--kl-mmq-sequence requires "
                "--kl-base and --kl-evaluator optimized");
        }
        if (kl_n_batch != 0 &&
                (kl_base.empty() ||
                 kl_evaluator != KlEvaluator::Optimized ||
                 kl_stream_layers > 0)) {
            throw std::runtime_error(
                "--kl-n-batch requires non-streamed --kl-base with "
                "--kl-evaluator optimized");
        }
        std::unique_ptr<KlMmqScope> kl_mmq_scope;
        const KlMmqMode load_mmq_mode =
            kl_mmq_sequence.empty()
            ? kl_mmq_mode : kl_mmq_sequence.front();
        if (load_mmq_mode != KlMmqMode::Default) {
            kl_mmq_scope =
                std::make_unique<KlMmqScope>(load_mmq_mode);
        }
        if (!kl_chunks_sequence_arg.empty()) {
            kl_chunks_sequence = parse_ids(kl_chunks_sequence_arg);
            if (kl_base.empty() ||
                std::any_of(
                    kl_chunks_sequence.begin(), kl_chunks_sequence.end(),
                    [](int64_t value) { return value <= 0; })) {
                throw std::runtime_error(
                    "--kl-chunks-sequence requires --kl-base and "
                    "positive comma-separated chunk counts");
            }
        }
        g_profiler.enabled = false;
        mfq_tensor_backend::NoGradGuard no_grad;
        if (!kl_base.empty()) std::cout << std::unitbuf;
        if (!kl_base.empty()) {
            std::cout << "cpp_kl_contract"
                      << " evaluator=" << kl_evaluator_name(kl_evaluator)
                      << " mmq=" << kl_mmq_mode_name(kl_mmq_mode)
                      << " chunk_limit=" << kl_chunks
                      << " score_count_override=" << kl_score_count
                      << " requested_n_batch=" << kl_n_batch
                      << " reference_n_batch="
                      << kl_reference_contract.n_batch
                      << " reference_n_ubatch="
                      << kl_reference_contract.n_ubatch
                      << "\n";
        }
        if (!kl_base.empty() && kl_stream_layers > 0) {
            if (kl_evaluator != KlEvaluator::Legacy) {
                throw std::runtime_error(
                    "--kl-evaluator optimized is unavailable for streamed KL");
            }
            if (!kl_chunks_sequence.empty()) {
                throw std::runtime_error(
                    "--kl-chunks-sequence is unavailable for streamed KL");
            }
            if (g_moe_expert_cache) {
                throw std::runtime_error(
                    "--moe-gpu-cache-gb is unavailable for streamed KL");
            }
            return run_kl_eval_streamed(
                model_path, config_path, kl_base,
                kl_save_logits_f16, kl_chunks,
                kl_stream_layers, kl_stream_batch,
                kl_score_count, kl_reference_contract);
        }
        auto dispatch_source = mfq::open_model_source(model_path);
        if (transport_mode) {
            if (!config_path.empty()) {
                throw std::runtime_error(
                    "model runtime does not accept an external model config");
            }
            const auto& runtime_assets = *dispatch_source;
            if (!runtime_assets.has_asset(mfq::kModelConfigAsset) ||
                (!runtime_assets.has_asset(mfq::cuda::kTokenizerGgufAsset) &&
                 tokenizer_model.empty())) {
                throw std::runtime_error(
                    "model runtime requires model config and tokenizer GGUF");
            }
        }
        auto run_loaded = [&]<mfq::cuda::CudaBackbone Backbone>() -> int {
        using Model = mfq::cuda::CausalLmFor<Backbone>;
        auto t0 = std::chrono::steady_clock::now();
        const bool load_optional_components =
            transport_mode || check_qwen35_mtp || check_flash_next_mtp ||
            !bench_qwen35_mtp.empty();
        Model model = mfq::cuda::load_causal_lm<Backbone>(
            model_path,
            config_path,
            context_size,
            true,
            load_optional_components,
            dispatch_source);
        auto runtime_components =
            load_runtime_components(model, load_optional_components);
        if (moe_expert_cache_has_sources() &&
                !moe_expert_cache_finalized()) {
            finalize_moe_expert_cache();
        }
        mfq_cuda_synchronize();
        auto t1 = std::chrono::steady_clock::now();
        report_cuda_memory("loaded");
        if (check_continuous_batching) {
            if constexpr (
                    Backbone == mfq::cuda::CudaBackbone::generic_qwen) {
                return mfq::cuda::continuous::
                    run_qwen_continuous_batching_check(model);
            }
            throw std::runtime_error(
                "continuous batching requires Qwen35CausalLm");
        }
        if (check_flash_next) {
            if constexpr (
                    Backbone == mfq::cuda::CudaBackbone::qwen4_exp ||
                    Backbone == mfq::cuda::CudaBackbone::glm5_next) {
                return run_flash_next_check(model);
            }
            throw std::runtime_error(
                "Flash-Next diagnostic requires a Flash-Next causal LM");
        }
        if (check_flash_next_mtp) {
            if constexpr (
                    Backbone == mfq::cuda::CudaBackbone::qwen4_exp ||
                    Backbone == mfq::cuda::CudaBackbone::glm5_next) {
                using Predictor=std::conditional_t<
                    Backbone==mfq::cuda::CudaBackbone::qwen4_exp,
                    Qwen4ExpMtp,Glm5NextMtp>;
                auto* predictor=dynamic_cast<Predictor*>(runtime_components.mtp.get());
                MFQ_RUNTIME_CHECK(predictor != nullptr,
                    "Flash-Next MTP diagnostic requires a loaded predictor component");
                return run_flash_next_mtp_check(model,*predictor);
            }
            throw std::runtime_error(
                "Flash-Next MTP diagnostic requires a Flash-Next causal LM");
        }
        if (check_qwen35_mtp) {
            if constexpr (
                    Backbone == mfq::cuda::CudaBackbone::generic_qwen) {
                auto* predictor = dynamic_cast<Qwen35Mtp*>(
                    runtime_components.mtp.get());
                MFQ_RUNTIME_CHECK(predictor != nullptr,
                    "--check-qwen35-mtp requires a supported model containing MTP weights");
                return run_qwen35_mtp_check(model, *predictor);
            }
            throw std::runtime_error(
                "--check-qwen35-mtp requires Qwen35CausalLm");
        }
        if (!bench_qwen35_mtp.empty()) {
            if constexpr (
                    Backbone == mfq::cuda::CudaBackbone::generic_qwen) {
                auto* predictor = dynamic_cast<Qwen35Mtp*>(
                    runtime_components.mtp.get());
                MFQ_RUNTIME_CHECK(predictor != nullptr,
                    "--bench-qwen35-mtp requires a supported model containing MTP weights");
                return run_qwen35_mtp_bench(model, *predictor,
                    bench_qwen35_mtp == "mtp", gen,
                    bench_qwen35_mtp_reps);
            }
            throw std::runtime_error(
                "--bench-qwen35-mtp requires Qwen35CausalLm");
        }
        if (transport_mode) {
            Model& inference_model = runtime_components.language(model);
            if (transport_api_key.empty()) {
                const char * env_key = std::getenv("MFQ_API_KEY");
                if (env_key != nullptr) transport_api_key = env_key;
            }
            MfqHttpRuntimeTransportConfig transport_config;
            transport_config.host = transport_host;
            transport_config.port = transport_port;
            transport_config.model_name = runtime_model_name;
            transport_config.model_type = inference_model.model_type();
            const auto component_state = runtime_components.state();
            MfqModelCapabilities model_capabilities;
            model_capabilities.family = runtime_components.graph.architecture;
            model_capabilities.source = "model-graph+cuda-adapters";
            model_capabilities.text =
                runtime_components.graph.has_component("text") &&
                runtime_components.plan.backbone !=
                    mfq::cuda::CudaBackbone::unsupported;
            model_capabilities.image_input =
                component_state.vision_available;
            model_capabilities.video_input =
                component_state.vision_available &&
                !runtime_components.grid_vision.has_value();
            const bool composite_loaded =
                runtime_components.minicpmo.has_value();
            model_capabilities.audio_input = composite_loaded &&
                runtime_components.graph.has_component("audio_input");
            model_capabilities.audio_output = composite_loaded &&
                runtime_components.graph.has_component("audio_output");
            model_capabilities.full_duplex = composite_loaded &&
                runtime_components.graph.has_component("duplex");
            model_capabilities.mtp = component_state.mtp_available &&
                continuous_batching == 0;
            transport_config.model_capabilities = std::move(model_capabilities);
            const auto& runtime_assets = *inference_model.source;
            if (runtime_assets.has_asset(mfq::cuda::kTokenizerGgufAsset)) {
                transport_config.tokenizer_gguf =
                    read_asset(runtime_assets, mfq::cuda::kTokenizerGgufAsset);
            } else if (!tokenizer_model.empty()) {
                transport_config.tokenizer_model = tokenizer_model;
            } else {
                throw std::runtime_error(
                    "model runtime requires a tokenizer GGUF");
            }
            transport_config.api_key = transport_api_key;
            transport_config.max_context =
                inference_model.max_position_embeddings();
            transport_config.vocab_size = inference_model.vocab_size();
            const auto embedded_profile = runtime_assets.metadata().find(
                "runtime.sampling.v1");
            transport_config.runtime_profile = resolve_mfq_runtime_profile(
                model_path,
                runtime_components.graph.architecture,
                transport_config.model_type,
                transport_config.model_name,
                embedded_profile == runtime_assets.metadata().end()
                    ? std::string()
                    : embedded_profile->second,
                runtime_assets.has_asset(mfq::kModelConfigAsset)
                    ? read_asset_text(runtime_assets, mfq::kModelConfigAsset)
                    : std::string(),
                runtime_sampling_profile);
            std::mutex model_mutex;
            DecodeGraphCache decode_graph_cache(
                inference_model.max_position_embeddings());
            TextSessionCache text_session_cache(
                make_cuda_paged_prefix_cache(
                    runtime_assets, inference_model));
            std::unique_ptr<
                mfq::cuda::continuous::CudaContinuousBatcher>
                continuous_batcher;
            if (continuous_batching > 0) {
                if constexpr (
                        Backbone == mfq::cuda::CudaBackbone::generic_qwen) {
                    continuous_batcher = std::make_unique<
                        mfq::cuda::continuous::CudaContinuousBatcher>(
                            inference_model, model_mutex,
                            continuous_batching,
                            prefill_chunk_size);
                    std::cerr
                        << "continuous_batching enabled=1 max_sequences="
                        << continuous_batching
                        << " prefill_chunk_size=" << prefill_chunk_size
                        << " decode=target_only mtp=disabled"
                        << " paged_kv="
                        << (continuous_batcher->paged_kv_enabled() ? 1 : 0)
                        << " page_size="
                        << continuous_batcher->paged_kv_page_size()
                        << " prefix_cache=fresh_prefill\n";
                } else {
                    throw std::runtime_error(
                        "continuous batching requires Qwen35CausalLm");
                }
            }
            std::optional<MiniCPMO45DuplexSession> minicpmo_duplex_session;
            MfqDuplexBackend duplex_backend;
            if (runtime_components.minicpmo) {
                duplex_backend = make_cuda_minicpmo45_duplex_backend(
                    *runtime_components.minicpmo,
                    model_mutex,
                    minicpmo_duplex_session);
            }
            MfqMultimodalGenerateFn multimodal_generate;
            if (runtime_components.minicpmo) {
                multimodal_generate =
                    [&](const std::vector<int64_t> & prompt,
                        const MfqVisionInput & vision,
                        const MfqSamplingParams & sampling,
                        const MfqTokenCallback & on_token,
                        const MfqPrefillCallback & on_prefill,
                        const MfqTokenConstraintPtr & token_constraint) {
                        return generate_multimodal_tokens(
                            *runtime_components.minicpmo,
                            model_mutex,
                            prompt,
                            vision,
                            sampling,
                            on_token,
                            on_prefill,
                            token_constraint);
                    };
            } else if (runtime_components.grid_vision) {
                multimodal_generate =
                    [&](const std::vector<int64_t>& prompt,
                        const MfqVisionInput& vision,
                        const MfqSamplingParams& sampling,
                        const MfqTokenCallback& on_token,
                        const MfqPrefillCallback& on_prefill,
                        const MfqTokenConstraintPtr& token_constraint) {
                        return generate_tokens<Model>(
                            inference_model, model_mutex, decode_graph_cache,
                            text_session_cache, prompt, sampling, on_token,
                            on_prefill, {}, token_constraint,
                            continuous_batching == 0
                                ? runtime_components.mtp.get()
                                : nullptr,
                            prefill_chunk_size, [&](Model& language) {
                                return std::optional<CudaPreparedPrompt>{
                                    runtime_components.grid_vision->prepare(
                                        language, prompt, vision)};
                            });
                    };
            }
            MfqInferenceEngine inference_engine;
            inference_engine.generate =
                [&](const std::vector<int64_t> & prompt,
                                   const MfqSamplingParams & sampling,
                                   const MfqTokenCallback & on_token,
                                   const MfqPrefillCallback & on_prefill,
                                   const MfqPromptCachePlan & cache_plan,
                                   const MfqTokenConstraintPtr & token_constraint) {
                if (continuous_batcher) {
                    return continuous_batcher->submit(
                        prompt, sampling, on_token, on_prefill,
                        cache_plan, token_constraint);
                }
                return generate_tokens(
                    inference_model, model_mutex, decode_graph_cache,
                    text_session_cache, prompt, sampling,
                    on_token, on_prefill, cache_plan, token_constraint,
                    runtime_components.mtp.get(), prefill_chunk_size);
            };
            inference_engine.duplex = duplex_backend;
            inference_engine.session_control = {
                [&](const std::string & source_session_id,
                        const std::string & target_session_id) {
                    std::lock_guard<std::mutex> lock(model_mutex);
                    return text_session_cache.fork_session(
                        source_session_id, target_session_id);
                },
                [&](const std::string & session_id) {
                    std::lock_guard<std::mutex> lock(model_mutex);
                    return text_session_cache.close_session(session_id);
                },
                [&] {
                    return text_session_cache.metrics();
                },
                [&] {
                    std::lock_guard<std::mutex> lock(model_mutex);
                    return text_session_cache.clear();
                },
                [&](uint64_t target_bytes) {
                    return text_session_cache.trim_hot(target_bytes);
                },
            };
            inference_engine.multimodal_generate = multimodal_generate;
            inference_engine.runtime_metrics =
            [&runtime_components, &continuous_batcher, &model_mutex] {
                size_t free_bytes = 0;
                size_t total_bytes = 0;
                MFQ_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
                const auto stats = mfq_cuda_memory_stats(
                    mfq_current_cuda_device());
                const auto components = runtime_components.state();
                const bool mtp_available =
                    components.mtp_available && !continuous_batcher;
                std::vector<std::pair<std::string, double>> result{
                    {"device_free_bytes", static_cast<double>(free_bytes)},
                    {"device_total_bytes", static_cast<double>(total_bytes)},
                    {"cuda_allocated_bytes", static_cast<double>(
                        stats.allocated_bytes)},
                    {"cuda_reserved_bytes", static_cast<double>(
                        stats.reserved_bytes)},
                    {"vision_declared", components.vision_declared ? 1.0 : 0.0},
                    {"vision_supported", components.vision_supported ? 1.0 : 0.0},
                    {"vision_available", components.vision_available ? 1.0 : 0.0},
                    {"mtp_declared", components.mtp_declared ? 1.0 : 0.0},
                    {"mtp_supported", components.mtp_supported ? 1.0 : 0.0},
                    {"mtp_available", mtp_available ? 1.0 : 0.0},
                };
                std::unique_lock lock(model_mutex, std::try_to_lock);
                if (mtp_available && lock.owns_lock() &&
                    runtime_components.mtp) {
                    const auto& mtp_stats = runtime_components.mtp->last_stats;
                    result.emplace_back(
                        "mtp_used", mtp_stats.used ? 1.0 : 0.0);
                    result.emplace_back(
                        "mtp_cycles", static_cast<double>(mtp_stats.cycles));
                    result.emplace_back(
                        "mtp_drafted_tokens",
                        static_cast<double>(mtp_stats.drafted_tokens));
                    result.emplace_back(
                        "mtp_accepted_tokens",
                        static_cast<double>(mtp_stats.accepted_tokens));
                    result.emplace_back(
                        "mtp_acceptance_rate",
                        mtp_stats.drafted_tokens == 0
                            ? 0.0
                            : static_cast<double>(
                                  mtp_stats.accepted_tokens) /
                                  mtp_stats.drafted_tokens);
                    result.emplace_back(
                        "mtp_selected_depth",
                        static_cast<double>(mtp_stats.selected_depth));
                    for (std::size_t depth = 0;
                         depth < mtp_stats.depth_cycles.size(); ++depth) {
                        result.emplace_back(
                            "mtp_depth_" + std::to_string(depth) + "_cycles",
                            static_cast<double>(
                                mtp_stats.depth_cycles[depth]));
                    }
                    for (std::size_t position = 0;
                         position < mtp_stats.position_drafted.size();
                         ++position) {
                        result.emplace_back(
                            "mtp_position_" + std::to_string(position + 1) +
                                "_acceptance_rate",
                            mtp_stats.position_drafted[position] == 0
                                ? 0.0
                                : static_cast<double>(
                                      mtp_stats.position_accepted[position]) /
                                      mtp_stats.position_drafted[position]);
                    }
                    for (std::size_t depth = 0;
                         depth < mtp_stats.measured_depth_ms.size(); ++depth) {
                        result.emplace_back(
                            "mtp_depth_" + std::to_string(depth) +
                                "_cycle_ms",
                            mtp_stats.measured_depth_ms[depth]);
                    }
                }
                if (continuous_batcher) {
                    auto batching = continuous_batcher->metrics();
                    result.insert(
                        result.end(), batching.begin(), batching.end());
                }
                return result;
            };
            auto transport = stdio_mode
                ? make_mfq_stdio_transport(transport_config)
                : make_mfq_http_transport(transport_config);
            MfqRuntime runtime(
                std::move(inference_engine), std::move(transport));
            const int status = runtime.run();
            if (g_moe_expert_cache) {
                print_moe_expert_cache_stats(std::cout);
            }
            return status;
        }
        if (!kl_base.empty()) {
            if (!kl_mmq_sequence.empty()) {
                if (!kl_chunks_sequence.empty()) {
                    throw std::runtime_error(
                        "--kl-mmq-sequence cannot be combined with "
                        "--kl-chunks-sequence");
                }
                for (size_t index = 0;
                     index < kl_mmq_sequence.size(); ++index) {
                    std::cout << "cpp_kl_mmq_sequence_begin index=" << index
                              << " mmq="
                              << kl_mmq_mode_name(kl_mmq_sequence[index])
                              << " chunks=" << kl_chunks << "\n";
                    KlMmqScope run_scope(kl_mmq_sequence[index]);
                    const int status = run_kl_eval_batched(
                        model, kl_base, kl_chunks,
                        kl_n_batch, kl_score_count,
                        kl_reference_contract);
                    if (status != 0) return status;
                    std::cout << "cpp_kl_mmq_sequence_end index=" << index
                              << " mmq="
                              << kl_mmq_mode_name(kl_mmq_sequence[index])
                              << " chunks=" << kl_chunks << "\n";
                }
                return 0;
            }
            if (kl_chunks_sequence.empty()) {
                const int status = run_selected_kl_eval(
                    model, kl_base, kl_chunks,
                    kl_evaluator,
                    kl_n_batch, kl_score_count,
                    kl_reference_contract);
                if (g_moe_expert_cache) {
                    print_moe_expert_cache_stats(std::cout);
                }
                return status;
            }
            for (size_t index = 0; index < kl_chunks_sequence.size(); ++index) {
                const int chunks =
                    static_cast<int>(kl_chunks_sequence[index]);
                std::cout << "cpp_kl_sequence_begin index=" << index
                          << " chunks=" << chunks << "\n";
                const int status = run_selected_kl_eval(
                    model, kl_base, chunks,
                    kl_evaluator,
                    kl_n_batch, kl_score_count,
                    kl_reference_contract);
                if (status != 0) return status;
                std::cout << "cpp_kl_sequence_end index=" << index
                          << " chunks=" << chunks << "\n";
            }
            if (g_moe_expert_cache) {
                print_moe_expert_cache_stats(std::cout);
            }
            return 0;
        }
        if (!prefill_sweep_sizes.empty()) {
            const int status = run_prefill_sweep(
                model, prefill_sweep_sizes, prefill_sweep_reps);
            if (g_moe_expert_cache) {
                print_moe_expert_cache_stats(std::cout);
            }
            return status;
        }
        auto ids_vec = ids_file.empty() ? parse_ids(ids_arg) : load_ids_file(ids_file);
        auto ids = mfq_tensor_backend::tensor(ids_vec, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64).device(mfq_tensor_backend::kCUDA)).unsqueeze(0);
        if (compare_dsv4_hc_ops) {
            g_dsv4_compare_hc_ops = true;
            g_dsv4_fused_hc = false;
            model.reset(1);
            (void)model.forward(ids);
            mfq_cuda_synchronize();
            return 0;
        }
        if (compare_dsv4_hc_model) {
            return run_dsv4_hc_model_compare(model, ids);
        }
        if (!block_trace_reference.empty()) {
            return run_block_trace_compare(
                model, block_trace_reference, config_path, context_size, ids);
        }
        if (!block_trace_output.empty()) {
            return run_block_trace_dump(
                model, block_trace_output, ids,
                block_trace_start, block_trace_count);
        }
        if (profile) {
            model.reset(1);
            (void)model.next_token(ids);
            mfq_cuda_synchronize();
            model.reset(1);
            g_profiler.reset();
            g_profiler.enabled = true;
        }
        if (compare_decode_splitk) {
            auto run = [&](const char * split) {
                mfq_set_env("MFQ_ATTENTION_DECODE_SPLITK", split);
                model.reset(1);
                (void)model.hidden_forward(ids);
                const int64_t decode_len = model.cache_pos + 1;
                auto seq_len = mfq_tensor_backend::tensor({decode_len}, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64).device(mfq_tensor_backend::kCUDA));
                auto decode_id = ids.index({Slice(), -1}).reshape({1, 1});
                auto hidden = model.hidden_forward(decode_id, mfq_nullopt, seq_len);
                auto last = hidden.index({Slice(), -1, Slice()}).to(mfq_tensor_backend::kFloat16).contiguous();
                auto logits = model.lm_head.forward(last).to(mfq_tensor_backend::kFloat32);
                mfq_cuda_synchronize();
                return logits;
            };
            auto ref = run("0");
            auto test = run("1");
            auto ref_logp = mfq_tensor_backend::log_softmax(ref, -1);
            auto test_logp = mfq_tensor_backend::log_softmax(test, -1);
            auto kl = (ref_logp.exp() * (ref_logp - test_logp)).sum(-1);
            auto diff = (ref - test).abs();
            std::cout << "decode_splitk_compare_kl=" << kl.template item<float>() << "\n";
            std::cout << "decode_splitk_compare_rel=" << ((test - ref).norm() / ref.norm()).template item<float>() << "\n";
            std::cout << "decode_splitk_compare_mean_abs=" << diff.mean().template item<float>() << "\n";
            std::cout << "decode_splitk_compare_max_abs=" << diff.max().template item<float>() << "\n";
            std::cout << "decode_splitk_compare_same_top="
                      << (ref.argmax(-1).eq(test.argmax(-1)).template item<bool>() ? 1 : 0) << "\n";
            return 0;
        }
        if (compare_mma_decode) {
            if (compare_mma_decode_steps < 1) {
                throw std::runtime_error("--compare-mma-decode-steps must be positive");
            }
            auto cuda_i64 = mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64).device(mfq_tensor_backend::kCUDA);
            std::vector<mfq_tensor_backend::Tensor> reference_logits;
            std::vector<int64_t> teacher_tokens;
            reference_logits.reserve(compare_mma_decode_steps);
            teacher_tokens.reserve(compare_mma_decode_steps);
            mfq_set_env("MFQ_MMA_ATTENTION_DECODE", "0");
            model.reset(1);
            auto input = model.next_token(ids).reshape({1, 1});
            const int64_t initial_teacher_token = input.template item<int64_t>();
            for (int step = 0; step < compare_mma_decode_steps; ++step) {
                const int64_t decode_len = model.cache_pos + 1;
                auto seq_len = mfq_tensor_backend::tensor({decode_len}, cuda_i64);
                auto hidden = model.hidden_forward(input, mfq_nullopt, seq_len);
                auto last = hidden.index({Slice(), -1, Slice()}).to(mfq_tensor_backend::kFloat16).contiguous();
                auto logits = model.lm_head.forward(last).to(mfq_tensor_backend::kFloat32);
                reference_logits.push_back(logits.clone());
                const int64_t next = logits.argmax(-1).template item<int64_t>();
                teacher_tokens.push_back(next);
                input = mfq_tensor_backend::tensor({next}, cuda_i64).reshape({1, 1});
            }
            mfq_set_env("MFQ_MMA_ATTENTION_DECODE", "1");
            g_decode_graph_attention_kv_len = compare_mma_decode_planned_len > 0
                ? compare_mma_decode_planned_len
                : ids.size(1) + compare_mma_decode_steps;
            if (g_decode_graph_attention_kv_len >
                    model.max_position_embeddings()) {
                throw std::runtime_error("decode comparison planned length exceeds context capacity");
            }
            model.reset(1);
            (void)model.next_token(ids);
            input = mfq_tensor_backend::tensor({initial_teacher_token}, cuda_i64).reshape({1, 1});
            double kl_sum = 0.0;
            double kl_max = 0.0;
            double abs_sum = 0.0;
            double delta_sq_sum = 0.0;
            double reference_sq_sum = 0.0;
            double max_abs = 0.0;
            int same_top = 0;
            int first_top_difference = -1;
            int64_t values = 0;
            for (int step = 0; step < compare_mma_decode_steps; ++step) {
                const int64_t decode_len = model.cache_pos + 1;
                auto seq_len = mfq_tensor_backend::tensor({decode_len}, cuda_i64);
                auto hidden = model.hidden_forward(input, mfq_nullopt, seq_len);
                auto last = hidden.index({Slice(), -1, Slice()}).to(mfq_tensor_backend::kFloat16).contiguous();
                auto test = model.lm_head.forward(last).to(mfq_tensor_backend::kFloat32);
                const auto & ref = reference_logits[(size_t)step];
                auto ref_logp = mfq_tensor_backend::log_softmax(ref, -1);
                auto test_logp = mfq_tensor_backend::log_softmax(test, -1);
                const double kl = (ref_logp.exp() * (ref_logp - test_logp)).sum(-1).template item<double>();
                auto delta = test - ref;
                kl_sum += kl;
                kl_max = std::max(kl_max, kl);
                abs_sum += delta.abs().sum().template item<double>();
                delta_sq_sum += delta.square().sum().template item<double>();
                reference_sq_sum += ref.square().sum().template item<double>();
                max_abs = std::max(max_abs, delta.abs().max().template item<double>());
                values += delta.numel();
                const bool top_equal = test.argmax(-1).eq(ref.argmax(-1)).template item<bool>();
                same_top += top_equal ? 1 : 0;
                if (!top_equal && first_top_difference < 0) first_top_difference = step;
                input = mfq_tensor_backend::tensor({teacher_tokens[(size_t)step]}, cuda_i64).reshape({1, 1});
            }
            mfq_cuda_synchronize();
            g_decode_graph_attention_kv_len = 0;
            std::cout << std::setprecision(10)
                      << "mma_decode_compare_steps=" << compare_mma_decode_steps << "\n"
                      << "mma_decode_compare_mean_kl=" << kl_sum / compare_mma_decode_steps << "\n"
                      << "mma_decode_compare_max_kl=" << kl_max << "\n"
                      << "mma_decode_compare_rel=" << std::sqrt(delta_sq_sum / reference_sq_sum) << "\n"
                      << "mma_decode_compare_mean_abs=" << abs_sum / values << "\n"
                      << "mma_decode_compare_max_abs=" << max_abs << "\n"
                      << "mma_decode_compare_same_top=" << same_top << "\n"
                      << "mma_decode_compare_first_top_difference=" << first_top_difference << "\n";
            return 0;
        }
        if (compare_nvq_vec4) {
            auto run = [&](const char * enabled) {
                mfq_set_env("MFQ_NVQ_SWIGLU_VEC4", enabled);
                model.reset(1);
                auto logits = model.last_logits(ids).to(mfq_tensor_backend::kFloat32);
                mfq_cuda_synchronize();
                return logits;
            };
            auto ref = run("0");
            auto repeat = run("0");
            auto test = run("1");
            auto ref_logp = mfq_tensor_backend::log_softmax(ref, -1);
            auto repeat_logp = mfq_tensor_backend::log_softmax(repeat, -1);
            auto test_logp = mfq_tensor_backend::log_softmax(test, -1);
            auto repeat_kl = (ref_logp.exp() * (ref_logp - repeat_logp)).sum(-1);
            auto kl = (ref_logp.exp() * (ref_logp - test_logp)).sum(-1);
            auto diff = (ref - test).abs();
            auto repeat_diff = (ref - repeat).abs();
            std::cout << "nvq_vec4_repeat_kl=" << repeat_kl.template item<float>() << "\n";
            std::cout << "nvq_vec4_repeat_max_abs=" << repeat_diff.max().template item<float>() << "\n";
            std::cout << "nvq_vec4_compare_kl=" << kl.template item<float>() << "\n";
            std::cout << "nvq_vec4_compare_rel="
                      << ((test - ref).norm() / ref.norm()).template item<float>() << "\n";
            std::cout << "nvq_vec4_compare_mean_abs=" << diff.mean().template item<float>() << "\n";
            std::cout << "nvq_vec4_compare_max_abs=" << diff.max().template item<float>() << "\n";
            std::cout << "nvq_vec4_compare_same_top="
                      << (ref.argmax(-1).eq(test.argmax(-1)).template item<bool>() ? 1 : 0) << "\n";
            return 0;
        }
        if (prefill_repeat > 0) {
            const char * trace_env = std::getenv("MFQ_CHECK_PREFILL_REPEAT_TRACE");
            const bool trace_repeat = trace_env != nullptr && std::atoi(trace_env) != 0;
            std::vector<mfq_tensor_backend::Tensor> reference_trace;
            std::vector<std::pair<std::string, mfq_tensor_backend::Tensor>> reference_gemma_trace;
            for (int i = 0; i < prefill_repeat; ++i) {
                model.reset(1);
                auto run_t0 = std::chrono::steady_clock::now();
                std::vector<mfq_tensor_backend::Tensor> trace;
                std::vector<std::pair<std::string, mfq_tensor_backend::Tensor>> gemma_trace;
                mfq_tensor_backend::Tensor logits;
                if (trace_repeat) {
                    g_gemma_trace_layer = 0;
                    g_gemma_stage_trace = &gemma_trace;
                    auto hidden = model.hidden_forward(
                        ids, mfq_nullopt, mfq_nullopt, &trace);
                    g_gemma_stage_trace = nullptr;
                    g_gemma_trace_layer = -1;
                    auto last = hidden.index({Slice(), -1, Slice()})
                        .to(mfq_tensor_backend::kFloat16).contiguous();
                    logits = model.lm_head.forward(last);
                } else {
                    logits = model.last_logits(ids);
                }
                mfq_cuda_synchronize();
                auto run_t1 = std::chrono::steady_clock::now();
                std::cout << "prefill_repeat=" << (i + 1)
                          << " sec=" << std::chrono::duration<double>(run_t1 - run_t0).count()
                          << " top=" << logits.argmax(-1).template item<int64_t>() << "\n";
                if (trace_repeat) {
                    if (reference_trace.empty()) {
                        reference_trace = std::move(trace);
                        reference_gemma_trace = std::move(gemma_trace);
                    } else {
                        for (size_t stage = 0; stage < gemma_trace.size(); ++stage) {
                            auto diff = (gemma_trace[stage].second -
                                         reference_gemma_trace[stage].second).abs();
                            const float max_abs = diff.max().template item<float>();
                            if (max_abs != 0.0f) {
                                std::cout << "prefill_repeat_first_gemma_stage="
                                          << gemma_trace[stage].first
                                          << " max_abs=" << max_abs
                                          << " mean_abs=" << diff.mean().template item<float>() << "\n";
                                break;
                            }
                        }
                        for (size_t stage = 0; stage < trace.size(); ++stage) {
                            auto diff = (trace[stage] - reference_trace[stage]).abs();
                            const float max_abs = diff.max().template item<float>();
                            if (max_abs != 0.0f) {
                                std::cout << "prefill_repeat_first_difference=" << stage
                                          << " layer=" << static_cast<int64_t>(stage) - 1
                                          << " max_abs=" << max_abs
                                          << " mean_abs=" << diff.mean().template item<float>() << "\n";
                                break;
                            }
                        }
                    }
                }
            }
            return 0;
        }
        if (compare_mma_attention) {
            mfq_set_env("MFQ_MMA_ATTENTION", "0");
            mfq_set_env("MFQ_DISABLE_MINICPM_BF16_FLASH128", "1");
            auto ref = model.last_logits(ids).to(mfq_tensor_backend::kFloat32);
            mfq_cuda_synchronize();
            model.reset(1);
            mfq_set_env("MFQ_MMA_ATTENTION", "1");
            mfq_set_env("MFQ_DISABLE_MINICPM_BF16_FLASH128", "0");
            auto test = model.last_logits(ids).to(mfq_tensor_backend::kFloat32);
            mfq_cuda_synchronize();
            auto ref_logp = mfq_tensor_backend::log_softmax(ref, -1);
            auto test_logp = mfq_tensor_backend::log_softmax(test, -1);
            auto kl = (ref_logp.exp() * (ref_logp - test_logp)).sum(-1);
            auto diff = (ref - test).abs();
            auto same_top = ref.argmax(-1).eq(test.argmax(-1));
            std::cout << "attention_compare_kl=" << kl.template item<float>() << "\n";
            std::cout << "attention_compare_max_logit_abs=" << diff.max().template item<float>() << "\n";
            std::cout << "attention_compare_mean_logit_abs=" << diff.mean().template item<float>() << "\n";
            std::cout << "attention_compare_same_top=" << (same_top.template item<bool>() ? 1 : 0) << "\n";
            return 0;
        }
        g_profiler.reset();
        auto next = model.next_token(ids);
        mfq_cuda_synchronize();
        auto t2 = std::chrono::steady_clock::now();
        report_cuda_memory("prefill");
        const char * empty_cache_env = std::getenv("MFQ_EMPTY_CACHE_BEFORE_GRAPH");
        if (empty_cache_env != nullptr && std::atoi(empty_cache_env) != 0) {
            mfq_cuda_empty_cache();
            report_cuda_memory("prefill_empty_cache");
        }
        g_profiler.report("prefill");
        g_profiler.reset();
        if (gen == 0) return 0;
        auto generated_cuda = mfq_tensor_backend::empty({gen}, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64).device(mfq_tensor_backend::kCUDA));
        cudaStream_t stream = mfq_get_current_cuda_stream().stream();
        MFQ_CUDA_CHECK(cudaMemcpyAsync(generated_cuda.template data_ptr<int64_t>(), next.template data_ptr<int64_t>(),
                                   sizeof(int64_t), cudaMemcpyDeviceToDevice, stream));
        const char* graph_env = std::getenv("MFQ_CUDA_GRAPH");
        const char * profile_graph_env = std::getenv("MFQ_PROFILE_CUDA_GRAPH");
        const bool profile_cuda_graph = profile && profile_graph_env != nullptr &&
            std::atoi(profile_graph_env) != 0;
        bool use_cuda_graph =
            (graph_env == nullptr || graph_env[0] != '0') &&
            !Model::is_flash_next &&
            mfq_cuda_graph_capture_supported() &&
            g_dsv4_cpu_offload_layers.empty() &&
            g_dense_cpu_layer_count == 0 &&
            !g_moe_expert_cache &&
            model_parallel_cuda_graph_enabled() &&
            (!profile || profile_cuda_graph) && gen > 1;
        const char * cuda_profiler_env = std::getenv("MFQ_CUDA_PROFILER_RANGE");
        const bool cuda_profiler_range = cuda_profiler_env != nullptr &&
            std::atoi(cuda_profiler_env) != 0;
        auto decode_replay_t0 = t2;
        if (cuda_profiler_range) MFQ_CUDA_CHECK(cudaProfilerStart());
        if (use_cuda_graph) {
            auto static_input = mfq_tensor_backend::empty({1, 1}, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64).device(mfq_tensor_backend::kCUDA));
            auto static_pos = mfq_tensor_backend::empty({1}, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64).device(mfq_tensor_backend::kCUDA));
            auto static_len = mfq_tensor_backend::empty({1}, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64).device(mfq_tensor_backend::kCUDA));
            auto static_step = mfq_tensor_backend::empty({1}, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64).device(mfq_tensor_backend::kCUDA));
            auto graph_stream = mfq_get_stream_from_pool(false);
            MfqCudaGuard graph_device_guard(
                graph_stream.device_index());
            auto graph_compute_streams =
                make_cuda_graph_compute_streams(graph_stream);
            auto graph_stream_guards =
                activate_cuda_graph_compute_streams(
                    graph_compute_streams);
            cudaStream_t graph_raw_stream = graph_stream.stream();

            int64_t pos_h = model.cache_pos;
            int64_t len_h = pos_h + 1;
            int64_t step_h = 1;
            MFQ_CUDA_CHECK(cudaMemcpyAsync(static_input.template data_ptr<int64_t>(), next.template data_ptr<int64_t>(),
                                       sizeof(int64_t), cudaMemcpyDeviceToDevice, graph_raw_stream));
            MFQ_CUDA_CHECK(cudaMemcpyAsync(static_pos.template data_ptr<int64_t>(), &pos_h,
                                       sizeof(int64_t), cudaMemcpyHostToDevice, graph_raw_stream));
            MFQ_CUDA_CHECK(cudaMemcpyAsync(static_len.template data_ptr<int64_t>(), &len_h,
                                       sizeof(int64_t), cudaMemcpyHostToDevice, graph_raw_stream));
            MFQ_CUDA_CHECK(cudaMemcpyAsync(static_step.template data_ptr<int64_t>(), &step_h,
                                       sizeof(int64_t), cudaMemcpyHostToDevice, graph_raw_stream));
            MFQ_CUDA_CHECK(cudaStreamSynchronize(graph_raw_stream));

            MfqCudaGraph graph;
            mfq_tensor_backend::Tensor static_next;
            const int64_t planned_len = model.cache_pos + gen;
            g_decode_graph_attention_kv_len = planned_len;
            g_decode_graph_attention_parts = planned_len >= 192 ? (planned_len + 127) / 128 : 1;
            g_decode_graph_attention_parts = std::min<int64_t>(
                g_decode_graph_attention_parts, FullBlock::kDecodeAttentionMaxParts);
            {
                DecodeGraphBranchScope branch_scope;
                prepare_decode_graph_memory(model, graph, [&]() {
                    (void)model.next_token_static(static_input, static_pos, static_len);
                }, cuda_graph_participant_streams(
                    graph_compute_streams));
                g_profiler.reset();
                g_profiler.graph_events = profile_cuda_graph;
                graph.capture_begin();
                static_next = g_profiler.measure("decode.model_total", [&]() {
                    return model.next_token_static(static_input, static_pos, static_len);
                });
                g_profiler.measure("decode.commit", [&]() {
                    decode_graph_commit_cuda(
                        static_next, generated_cuda, static_step,
                        static_input, static_pos, static_len);
                    return 0;
                });
                graph.capture_end();
            }
            mfq_debug_dump_cuda_graph(graph);
            report_cuda_memory("graph_captured");
            g_decode_graph_attention_kv_len = 0;
            g_decode_graph_attention_parts = 0;

            decode_replay_t0 = std::chrono::steady_clock::now();
            for (int i = 1; i < gen; ++i) {
                graph.replay();
            }
            MFQ_CUDA_CHECK(cudaStreamSynchronize(graph_raw_stream));
        } else {
            for (int i = 1; i < gen; ++i) {
                next = g_profiler.measure("decode.eager_model", [&]() {
                    return model.next_token(next.view({1, 1}));
                });
                g_profiler.measure("decode.eager_commit", [&]() {
                    MFQ_CUDA_CHECK(cudaMemcpyAsync(
                        generated_cuda.template data_ptr<int64_t>() + i,
                        next.template data_ptr<int64_t>(), sizeof(int64_t),
                        cudaMemcpyDeviceToDevice, stream));
                    return 0;
                });
            }
        }
        mfq_cuda_synchronize();
        if (cuda_profiler_range) MFQ_CUDA_CHECK(cudaProfilerStop());
        auto t3 = std::chrono::steady_clock::now();
        g_profiler.report("decode");
        auto generated_tensor = generated_cuda.to(mfq_tensor_backend::kCPU).contiguous();
        auto generated_ptr = generated_tensor.template data_ptr<int64_t>();
        double load_s = std::chrono::duration<double>(t1 - t0).count();
        double prefill_s = std::chrono::duration<double>(t2 - t1).count();
        double decode_s = std::chrono::duration<double>(t3 - t2).count();
        double decode_replay_s = std::chrono::duration<double>(t3 - decode_replay_t0).count();
        std::cout << "load_sec=" << load_s << "\n";
        std::cout << "prefill_sec=" << prefill_s << "\n";
        std::cout << "decode_tokens=" << std::max(0, gen - 1) << "\n";
        std::cout << "decode_setup_sec=" << (decode_s - decode_replay_s) << "\n";
        std::cout << "decode_replay_sec=" << decode_replay_s << "\n";
        std::cout << "decode_sec=" << decode_s << "\n";
        if (gen > 1) std::cout << "decode_tok_per_s=" << (double)(gen - 1) / decode_s << "\n";
        if (gen > 1) std::cout << "decode_steady_tok_per_s=" << (double)(gen - 1) / decode_replay_s << "\n";
        std::cout << "generated_ids=";
        for (int64_t i = 0; i < generated_tensor.numel(); ++i) {
            if (i) std::cout << ",";
            std::cout << generated_ptr[i];
        }
        std::cout << "\n";
        if (g_moe_expert_cache) {
            print_moe_expert_cache_stats(std::cout);
        }
        return 0;
        };

        const auto dispatch = mfq::cuda::cuda_model_plan(
            dispatch_source->resolved_model_graph()).backbone;
        switch (dispatch) {
            case mfq::cuda::CudaBackbone::generic_qwen:
                return run_loaded.template operator()<
                    mfq::cuda::CudaBackbone::generic_qwen>();
            case mfq::cuda::CudaBackbone::minicpmo45:
                return run_loaded.template operator()<
                    mfq::cuda::CudaBackbone::minicpmo45>();
            case mfq::cuda::CudaBackbone::minicpmo_tts:
                return run_loaded.template operator()<
                    mfq::cuda::CudaBackbone::minicpmo_tts>();
            case mfq::cuda::CudaBackbone::gemma4:
                return run_loaded.template operator()<
                    mfq::cuda::CudaBackbone::gemma4>();
            case mfq::cuda::CudaBackbone::glm_dsa:
                return run_loaded.template operator()<
                    mfq::cuda::CudaBackbone::glm_dsa>();
            case mfq::cuda::CudaBackbone::glm5_next:
                return run_loaded.template operator()<
                    mfq::cuda::CudaBackbone::glm5_next>();
            case mfq::cuda::CudaBackbone::qwen4_exp:
                return run_loaded.template operator()<
                    mfq::cuda::CudaBackbone::qwen4_exp>();
            case mfq::cuda::CudaBackbone::deepseek_v4:
                return run_loaded.template operator()<
                    mfq::cuda::CudaBackbone::deepseek_v4>();
            case mfq::cuda::CudaBackbone::deepseek_v41:
                return run_loaded.template operator()<
                    mfq::cuda::CudaBackbone::deepseek_v41>();
            case mfq::cuda::CudaBackbone::unsupported:
                throw std::runtime_error("unsupported CUDA model backbone");
        }
        throw std::runtime_error("invalid CUDA model backbone");
    } catch (const MfqBackendError & e) {
        std::cerr << "backend_error: " << e.what() << "\n";
        return 1;
    } catch (const std::exception & e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
