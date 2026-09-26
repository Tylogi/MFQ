#pragma once

#include "mfq_tensor_backend.h"
#include "tensor_parallel.h"

#include <cuda_runtime_api.h>
#ifdef MFQ_HAVE_NCCL
#include <nccl.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
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
    std::unordered_map<std::string, ProfileStat> stats;
    std::vector<std::string> order;
    std::vector<PendingEvent> pending;

    bool selected(const std::string & name) const {
        const char * filter_value =
            std::getenv("MFQ_PROFILE_CUDA_FILTER");
        if (filter_value == nullptr || filter_value[0] == '\0') {
            return true;
        }
        const std::string_view filter(filter_value);
        size_t begin = 0;
        while (begin <= filter.size()) {
            const size_t end = filter.find(',', begin);
            const size_t count = end == std::string_view::npos
                ? filter.size() - begin
                : end - begin;
            if (filter.substr(begin, count) == name) {
                return true;
            }
            if (end == std::string_view::npos) {
                break;
            }
            begin = end + 1;
        }
        return false;
    }

    void reset() {
        for (auto & p : pending) {
            cudaEventDestroy(p.start);
            cudaEventDestroy(p.stop);
        }
        pending.clear();
        stats.clear();
        order.clear();
    }

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

    void report(const std::string & title) {
        if (!enabled) return;
        if (!pending.empty()) cudaEventSynchronize(pending.back().stop);
        for (auto & p : pending) {
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, p.start, p.stop);
            stats.at(p.name).ms += (double)ms;
            cudaEventDestroy(p.start);
            cudaEventDestroy(p.stop);
        }
        pending.clear();
        std::cerr << "profile " << title << "\n";
        for (const auto & name : order) {
            const auto & s = stats.at(name);
            std::cerr << "profile_item"
                      << " name=" << name
                      << " calls=" << s.calls
                      << " cuda_ms=" << s.ms
                      << " cuda_avg_ms=" << (s.calls ? s.ms / (double)s.calls : 0.0)
                      << " wall_ms=" << s.wall_ms
                      << " wall_avg_ms=" << (s.calls ? s.wall_ms / (double)s.calls : 0.0)
                      << "\n";
        }
    }
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

    ~ModelParallelCollectiveRuntime() {
        reset();
    }

    void reset() noexcept {
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
            if (communicator != nullptr) {
                (void)ncclCommDestroy(communicator);
            }
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

    void configure(
            const std::vector<int> & requested_devices,
            bool allow_duplicate_devices) {
        reset();
        if (requested_devices.size() < 2 || allow_duplicate_devices) {
            return;
        }
        devices = requested_devices;
        streams.reserve(devices.size());
        ready.resize(devices.size(), nullptr);
        completed.resize(devices.size(), nullptr);
        reduction_buffers.resize(devices.size());
        for (size_t index = 0; index < devices.size(); ++index) {
            MfqCudaGuard guard(devices[index]);
            streams.push_back(
                mfq_get_stream_from_pool(false, devices[index]));
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
        // NCCL initializes peer transports lazily.  That initialization can
        // allocate shared-memory control state, which CUDA forbids once graph
        // capture has started.  Exercise both directions between the primary
        // rank and every peer while ordinary stream execution is still active.
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
                static_cast<int>(peer), communicators[0],
                streams[0].stream()));
            MFQ_NCCL_CHECK(ncclRecv(
                p2p_warmup_buffers[peer].data_ptr(), 1, ncclUint8,
                0, communicators[peer], streams[peer].stream()));
            MFQ_NCCL_CHECK(ncclSend(
                p2p_warmup_buffers[peer].data_ptr(), 1, ncclUint8,
                0, communicators[peer], streams[peer].stream()));
            MFQ_NCCL_CHECK(ncclRecv(
                p2p_warmup_buffers[0].data_ptr(), 1, ncclUint8,
                static_cast<int>(peer), communicators[0],
                streams[0].stream()));
            MFQ_NCCL_CHECK(ncclGroupEnd());
        }
        for (size_t index = 0; index < devices.size(); ++index) {
            MfqCudaGuard guard(devices[index]);
            MFQ_CUDA_CHECK(cudaStreamSynchronize(streams[index].stream()));
        }
        collectives_enabled = true;
#endif
    }
};

int model_parallel_primary_device();

struct LayerPlacementConfig {
    std::vector<int> devices;
    std::vector<double> split;
    std::vector<int> layer_devices;
    int load_device = -1;

    bool enabled() const {
        return devices.size() > 1;
    }

    int primary_device() const {
        return devices.empty()
            ? model_parallel_primary_device()
            : devices.front();
    }

    void prepare(int64_t layers) {
        layer_devices.assign(
            static_cast<size_t>(layers), primary_device());
        if (!enabled()) return;
        const auto slices = mfq::plan_tensor_parallel_slices(
            layers, 1, devices, split);
        for (const auto & slice : slices) {
            for (int64_t layer = slice.begin; layer < slice.end; ++layer) {
                layer_devices.at(static_cast<size_t>(layer)) = slice.device;
            }
        }
    }

    int device_for_layer(int64_t layer) const {
        if (layer < 0 || layer >= static_cast<int64_t>(layer_devices.size())) {
            throw std::runtime_error("layer-placement index is outside the model");
        }
        return layer_devices.at(static_cast<size_t>(layer));
    }
};

extern CudaProfiler g_profiler;
extern bool g_force_moe_pool_path;
extern bool g_force_moe_unfused_reduce;
extern bool g_force_moe_materialized_swiglu;
extern bool g_force_moe_prefill_mma_off;
extern thread_local bool g_moe_continuous_batch_cache_serial;
extern KlMmqMode g_kl_mmq_mode;
extern int64_t g_kl_mmq_activation_quantize_calls;
extern int64_t g_kl_mmq_dense_calls;
extern int64_t g_kl_mmq_moe_calls;
extern int64_t g_kl_mmq_fallback_calls;
extern int64_t g_kl_kv_cache_capacity;
extern std::unordered_set<int> g_dsv4_cpu_offload_layers;
extern int64_t g_dsv4_cpu_offload_host_bytes;
extern int g_n_gpu_layers;
extern int g_dense_cpu_layer_count;
extern bool g_loading_cpu_layer;
extern int g_moe_cache_registration_min_slots;
extern int g_gemma_trace_layer;
extern std::vector<std::pair<std::string, mfq_tensor_backend::Tensor>>* g_gemma_stage_trace;
extern ParallelConfig g_tensor_parallel;
extern ParallelConfig g_expert_parallel;
extern ModelParallelCollectiveRuntime g_model_parallel_collectives;
extern LayerPlacementConfig g_layer_placement;

void mfq_set_env(const char* name, const char* value);
void mfq_release_host_allocator_cache() noexcept;
bool model_parallel_enabled();
const ParallelConfig& model_parallel_config();
const ParallelConfig& moe_parallel_config();
int model_parallel_primary_device();
bool model_parallel_cuda_graph_enabled();
bool tensor_parallel_grouped_projections_enabled();
bool tensor_parallel_shared_linear_attention_input_enabled();
bool tensor_parallel_mirror_linear_attention_scalars_enabled();
bool tensor_parallel_mirror_qwen35_attention_kv_enabled();
bool model_parallel_reduce_to_primary_enabled();
bool model_parallel_fp16_reduce_enabled();
bool model_parallel_peer_first_launch_enabled();
size_t model_parallel_launch_index(size_t launch_position, size_t shard_count);
int active_weight_load_device();
const char* kl_mmq_mode_name(KlMmqMode mode);
mfq_tensor_backend::Tensor kl_mmq_prepare_activation(mfq_tensor_backend::Tensor x);
const char* moe_route_stats_path();
bool moe_route_output_energy_enabled();
void record_moe_route_stats(
    int layer,
    const mfq_tensor_backend::Tensor& ids,
    const mfq_tensor_backend::Tensor& weights,
    const mfq_tensor_backend::Tensor& output,
    int n_experts);
void clear_moe_route_stats();
void write_moe_route_stats();
void trace_gemma_stage(int layer, const char* name, const mfq_tensor_backend::Tensor& value);
bool gemma4_fused_norms_enabled();
void report_cuda_memory(const char* stage);
bool moe_small_glu_path_enabled(int tokens);

struct KlMmqScope {
    KlMmqMode previous_mode;
    int64_t previous_activation_quantize_calls;
    int64_t previous_dense_calls;
    int64_t previous_moe_calls;
    int64_t previous_fallback_calls;

    explicit KlMmqScope(KlMmqMode mode)
        : previous_mode(g_kl_mmq_mode),
          previous_activation_quantize_calls(
              g_kl_mmq_activation_quantize_calls),
          previous_dense_calls(g_kl_mmq_dense_calls),
          previous_moe_calls(g_kl_mmq_moe_calls),
          previous_fallback_calls(g_kl_mmq_fallback_calls) {
        g_kl_mmq_mode = mode;
        g_kl_mmq_activation_quantize_calls = 0;
        g_kl_mmq_dense_calls = 0;
        g_kl_mmq_moe_calls = 0;
        g_kl_mmq_fallback_calls = 0;
    }

    ~KlMmqScope() {
        g_kl_mmq_mode = previous_mode;
        g_kl_mmq_activation_quantize_calls =
            previous_activation_quantize_calls;
        g_kl_mmq_dense_calls = previous_dense_calls;
        g_kl_mmq_moe_calls = previous_moe_calls;
        g_kl_mmq_fallback_calls = previous_fallback_calls;
    }
};

struct KlKvCacheCapacityScope {
    int64_t previous_capacity;

    explicit KlKvCacheCapacityScope(int64_t capacity)
        : previous_capacity(g_kl_kv_cache_capacity) {
        g_kl_kv_cache_capacity = capacity;
    }

    ~KlKvCacheCapacityScope() {
        g_kl_kv_cache_capacity = previous_capacity;
    }
};

struct MoeRouteLayerStats {
    mfq_tensor_backend::Tensor counts;
    mfq_tensor_backend::Tensor weight_sum;
    mfq_tensor_backend::Tensor weight_sq_sum;
    mfq_tensor_backend::Tensor output_energy;
    mfq_tensor_backend::Tensor weighted_output_energy;
};
