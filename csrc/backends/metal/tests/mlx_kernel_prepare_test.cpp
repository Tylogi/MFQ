#include "runtime/mlx_kernel_prepare.h"
#include "ops/mlx_nint.h"
#include "runtime/mlx_resident_budget.h"
#include "runtime/mlx_sampling.h"
#include "runtime/mlx_tensor.h"
#include <mlx/primitives.h>

#include <cstdint>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <future>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace mlx::core;

extern "C" __attribute__((noinline)) void mfq_kernel_prepare_phase(int phase) {
    std::cout << "kernel_prepare_phase=" << phase << std::endl;
}

namespace {
template <typename T>
void append(std::vector<std::uint8_t>& blob, T value) {
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(&value);
    blob.insert(blob.end(), bytes, bytes + sizeof(value));
}

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

struct ProbeState {
    std::atomic<int> active{0}, peak{0}, calls{0};
};

class PrepareProbe final : public UnaryPrimitive, public mfq::metal::MlxPreparableKernel {
public:
    PrepareProbe(std::shared_ptr<ProbeState> state, int key, bool fail)
        : UnaryPrimitive(default_stream(Device::gpu)), state_(std::move(state)), key_(key), fail_(fail) {}
    std::string preparation_key() const override { return "probe_" + std::to_string(key_); }
    void prepare_gpu() override {
        const int active = state_->active.fetch_add(1) + 1;
        auto peak = state_->peak.load();
        while (peak < active && !state_->peak.compare_exchange_weak(peak, active)) {}
        ++state_->calls;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        --state_->active;
        if (fail_) throw std::runtime_error("expected preparation failure");
    }
    void eval_cpu(const std::vector<array>&, array&) override { throw std::runtime_error("probe evaluated"); }
    void eval_gpu(const std::vector<array>&, array&) override { throw std::runtime_error("probe evaluated"); }
    const char* name() const override { return "PrepareProbe"; }
private:
    std::shared_ptr<ProbeState> state_;
    int key_;
    bool fail_;
};

void test_parallel_jobs() {
    constexpr int jobs = 32;
    auto state = std::make_shared<ProbeState>();
    {
        mfq::metal::MlxKernelPreparation preparation(128, 128);
        for (int key = 0; key < jobs; ++key) {
            for (int duplicate = 0; duplicate < 2; ++duplicate)
                preparation.add(array({1}, float32, std::make_shared<PrepareProbe>(state, key, false), {}));
        }
        require(preparation.size() == jobs, "preparation jobs were not deduplicated");
        preparation.finish();
    }
    require(state->calls.load() == jobs && state->active.load() == 0, "parallel preparation lost jobs");
    if (!std::getenv("MFQ_METAL_COMPILE_WORKERS") && std::thread::hardware_concurrency() > 1)
        require(state->peak.load() > 1 && state->peak.load() <= std::min<unsigned>(jobs,
            std::thread::hardware_concurrency()), "preparation exceeded available CPU concurrency");
    {
        mfq::metal::MlxKernelPreparation preparation(128, 128);
        for (int key = 0; key < 8; ++key)
            preparation.add(array({1}, float32, std::make_shared<PrepareProbe>(state, 99 + key, key == 0), {}));
        bool failed = false;
        try { preparation.finish(); }
        catch (const std::runtime_error& e) { failed = std::string(e.what()) == "expected preparation failure"; }
        require(failed && state->active.load() == 0, "preparation did not propagate its worker error");
        require(mfq::metal::MlxKernelPreparation::current() == nullptr, "failed preparation remained active");
    }
    std::cout << "parallel_prepare_peak=" << state->peak.load() << std::endl;
}

void test_shared_compilation() {
    std::atomic<int> builds{0};
    std::vector<std::future<MTL::ComputePipelineState*>> jobs;
    for (int i = 0; i < 8; ++i) jobs.push_back(std::async(std::launch::async, [&] {
        auto& device = mfq::metal::kernel_device(Device::gpu);
        auto* library = device.get_library("mfq_compile_cache_test_v1", [&] {
            ++builds;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            return std::string("#include <metal_stdlib>\nusing namespace metal;\n"
                "kernel void mfq_cache_unit(device float* x [[buffer(0)]], uint i [[thread_position_in_grid]]) { x[i] = 1.0f; }\n");
        });
        return device.get_kernel("mfq_cache_unit", library);
    }));
    auto* expected = jobs.front().get();
    for (std::size_t i = 1; i < jobs.size(); ++i)
        require(jobs[i].get() == expected, "concurrent preparation returned different pipelines");
    require(builds.load() == 1, "concurrent preparation compiled a library more than once");
    auto& device = mfq::metal::kernel_device(Device::gpu);
    bool failed = false;
    try {
        device.get_library("mfq_compile_cache_retry_v1", []() -> std::string {
            throw std::runtime_error("transient compilation failure");
        });
    } catch (const std::runtime_error&) { failed = true; }
    require(failed, "compilation failure was swallowed");
    auto* retry = device.get_library("mfq_compile_cache_retry_v1", [] {
        return std::string("#include <metal_stdlib>\nkernel void mfq_cache_retry() {}\n");
    });
    require(device.get_kernel("mfq_cache_retry", retry) != nullptr, "compilation failure could not be retried");
}

mfq::metal::MlxNintWeight weight(int output_size = 2) {
    std::vector<std::uint8_t> blob;
    append<std::uint8_t>(blob, 4);
    append<std::uint8_t>(blob, 1);
    append<std::int32_t>(blob, 64);
    append<std::int32_t>(blob, 0);
    append<std::int32_t>(blob, 64);
    append<std::uint32_t>(blob, 2);
    append<std::int64_t>(blob, output_size);
    append<std::int64_t>(blob, 64);
    append<std::uint32_t>(blob, output_size);
    append<std::uint32_t>(blob, 1);
    for (int row = 0; row < output_size; ++row) append<std::uint16_t>(blob, 0x3c00);
    for (int row = 0; row < output_size; ++row) append<std::uint16_t>(blob, 0);
    blob.insert(blob.end(), (output_size + 7) / 8, 0xff);
    blob.insert(blob.end(), (output_size + 7) / 8, 0);
    blob.insert(blob.end(), output_size * 32, 0x11);
    return mfq::metal::MlxNintWeight::from_blob(blob);
}

void test_model_projection_paths() {
    mfq::metal::MlxLinear first(weight(128)), second(weight(128));
    std::size_t all = 0;
    {
        mfq::metal::MlxKernelPreparation preparation(128, 128);
        mfq::metal::MlxProjectionBatch projections({&first, &second});
        all = preparation.size();
    }
    {
        mfq::metal::MlxKernelPreparation preparation(128, 128);
        preparation.use_model_paths();
        mfq::metal::MlxProjectionBatch projections({&first, &second});
        require(preparation.size() == 0, "model projections prepared unused constructor variants");
        const auto outputs = projections(zeros({1, 64}, float16));
        for (const auto& output : outputs) preparation.add(output);
        const auto selected = preparation.size();
        require(selected > 0 && selected < all, "model projections did not select execution paths");
        for (const auto& output : outputs) preparation.add(output);
        require(preparation.size() == selected, "model projections collected duplicate jobs");
        preparation.finish();
        for (const auto& output : outputs) require(!output.is_available(), "model preparation evaluated projections");
        std::cout << "projection_variants=" << all << " projection_selected=" << selected << std::endl;
    }
}
}

int main() {
    try {
#ifdef MFQ_MLX_METALLIB_DEFAULT
        metal::set_metallib_path(MFQ_MLX_METALLIB_DEFAULT);
#endif
        set_default_device(Device::gpu);
        test_parallel_jobs();
        test_shared_compilation();
        test_model_projection_paths();
        const auto packed = weight();
        const auto expanded = weight(129);
        mfq::metal::MlxKernelPreparation preparation(2048, 2048);
        const auto memory = get_active_memory();
        mfq_kernel_prepare_phase(1);
        std::vector<array> descriptions;
        for (const auto rows : preparation.row_buckets()) {
            descriptions.push_back(packed.matmul(zeros({rows, 64}, float16)));
            preparation.add(descriptions.back());
            descriptions.push_back(expanded.matmul(zeros({rows, 64}, float16)));
            preparation.add(descriptions.back());
            if (rows >= 64) {
                preparation.add(packed.matmul_packed(zeros({rows, 64}, float16)));
                preparation.add(expanded.matmul_packed(zeros({rows, 64}, float16)));
                preparation.add(packed.dequantize(float16));
                preparation.add(expanded.dequantize(float16));
            }
        }
        const auto logits = zeros({1, 13}, float32);
        auto counts = mfq::metal::sample_token_counts_add(zeros({13}, int32), zeros({8}, int32));
        auto adjusted = mfq::metal::sample_apply_penalties(logits, counts, 0.1, 0.1, 1.05);
        descriptions.push_back(mfq::metal::sample_greedy(adjusted));
        preparation.add(descriptions.back());
        require(preparation.size() >= 3, "no compilation descriptions collected");
        require(preparation.unsupported().empty(), "native compilation interface missing");
        const auto description_memory = get_active_memory();
        preparation.finish();
        require(mfq::metal::MlxKernelPreparation::current() == nullptr, "preparation still active after load");
        std::cout << "metadata_memory=" << description_memory - memory
                  << " compilation_memory=" << get_active_memory() - description_memory << std::endl;
        require(get_active_memory() == description_memory, "pure compilation allocated tensor buffers");
        for (const auto& description : descriptions)
            require(!description.is_available(), "pure compilation evaluated an output");
        mfq_kernel_prepare_phase(2);
        for (const int rows : {9, 17, 29, 64, 65, 127, 128, 129, 511, 512, 2039, 2048}) {
            for (const auto* projection : {&packed, &expanded}) {
                auto output = astype(projection->matmul(ones({rows, 64}, float16)), float32);
                output.eval();
                for (std::size_t index = 0; index < output.size(); ++index)
                    require(output.data<float>()[index] == 64.0f, "prepared projection is incorrect");
            }
        }
        {
            mfq::metal::MlxResidentBudgetScope denied({}, [](std::size_t) { return false; });
            for (const auto* projection : {&packed, &expanded}) {
                const auto fallback = projection->matmul(ones({2048, 64}, float16));
                require(all(equal(fallback, array(64.0f, float16))).item<bool>(),
                    "prepared low-memory fallback is incorrect");
            }
        }
        descriptions.back().eval();
        require(descriptions.back().item<int>() == 1, "prepared sampling is incorrect");
        mfq_kernel_prepare_phase(3);
        std::cout << "Metal pure kernel preparation passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
