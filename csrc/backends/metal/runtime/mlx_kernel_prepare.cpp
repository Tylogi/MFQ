#include "mlx_kernel_prepare.h"
#include "mfq_mlx_utils_embedded.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include <mlx/backend/metal/device.h>
#include <mlx/fast_primitives.h>
#include <mlx/version.h>

namespace mfq::metal {

thread_local MlxKernelPreparation* MlxKernelPreparation::active_ = nullptr;

MlxKernelPreparation::MlxKernelPreparation(int context, int chunk_size)
    : previous_(active_), context_(context) {
    if (context < 1 || chunk_size < 1)
        throw std::invalid_argument("invalid Metal kernel preparation geometry");
    for (const int requested : {1, 2, 3, 4, 5, 6, 8, 128, chunk_size}) {
        const int rows = std::min(context, requested);
        if (std::find(rows_.begin(), rows_.end(), rows) == rows_.end()) rows_.push_back(rows);
    }
    active_ = this;
}

MlxKernelPreparation::~MlxKernelPreparation() { active_ = previous_; }

MlxKernelPreparation* MlxKernelPreparation::current() noexcept {
    return active_ && !active_->collecting_ ? active_ : nullptr;
}

void MlxKernelPreparation::collect(const std::function<void()>& build) {
    const bool previous = collecting_;
    collecting_ = true;
    try { build(); }
    catch (...) { collecting_ = previous; throw; }
    collecting_ = previous;
}

void MlxKernelPreparation::add(const mlx::core::array& output) {
    std::unordered_set<const mlx::core::Primitive*> visited;
    const auto visit = [&](const auto& self, const mlx::core::array& value) -> void {
        if (!value.has_primitive()) return;
        const auto primitive = value.primitive_ptr();
        if (!visited.insert(primitive.get()).second) return;
        if (auto* native = dynamic_cast<MlxPreparableKernel*>(primitive.get())) {
            if (keys_.insert("native:" + native->preparation_key()).second)
                jobs_.push_back([primitive] {
                    dynamic_cast<MlxPreparableKernel&>(*primitive).prepare_gpu();
                });
        } else if (std::string_view(primitive->name()) == "CustomKernel") {
            const auto* custom = static_cast<mlx::core::fast::CustomKernel*>(primitive.get());
            const auto state = custom->state();
            const auto& name = std::get<0>(state);
            std::ostringstream library_key;
            library_key << name;
#if MLX_VERSION_NUMERIC >= 32001
            library_key << '_' << std::hex << std::hash<std::string>{}(std::get<1>(state))
                        << '_' << std::dec << std::get<10>(state);
#endif
            const auto library_name = library_key.str();
            if (keys_.insert("custom:" + library_name).second)
                jobs_.push_back([primitive, library_name] {
                    const auto state = static_cast<mlx::core::fast::CustomKernel&>(*primitive).state();
                    auto& device = mlx::core::metal::device(primitive->stream().device);
                    const auto& name = std::get<0>(state);
                    auto* library = device.get_library(library_name, mlx::core::CompileOptions(std::get<10>(state)),
                        [&] { return std::string(detail::kMlxUtilsSource) + std::get<1>(state); });
                    (void)device.get_kernel(name, library);
                });
        } else {
            const std::string name = primitive->name();
            if (name.find("Primitive") != std::string::npos) unsupported_.insert(name);
        }
        for (const auto& input : value.inputs()) self(self, input);
    };
    visit(visit, output);
}

void MlxKernelPreparation::collect_sampling(int vocab, const MlxSamplingParams& defaults) {
    collect([&] {
        for (const auto dtype : {mlx::core::float16, mlx::core::bfloat16, mlx::core::float32}) {
            for (int rows = 1; rows <= 6; ++rows) {
                const auto logits = mlx::core::zeros({rows, vocab}, dtype);
                const std::vector<float> random_values(rows, 0.5f);
                add(sample_greedy(logits));
                for (const auto& random : {mlx::core::zeros({rows}, mlx::core::float32),
                        mlx::core::array(random_values.begin(), mlx::core::Shape{rows})}) {
                    add(sample_softmax(logits, random));
                    add(sample(logits, defaults, random));
                    if (defaults.top_k > 0 && defaults.top_k <= 128)
                        add(sample_top_k_distribution(logits, random, 1.0, std::min(defaults.top_k, vocab)).sampled);
                }
                const auto counts = mlx::core::zeros({vocab}, mlx::core::int32);
                add(sample_apply_penalties(logits, counts, 0.1, 0.1, 1.05));
                for (const int count : {1, 8}) {
                    add(sample_token_counts_add(counts, mlx::core::zeros({count}, mlx::core::int32)));
                    const std::vector<std::int32_t> ids(count, 0);
                    add(sample_token_counts_add(counts, mlx::core::array(ids.begin(), mlx::core::Shape{count})));
                }
            }
        }
    });
}

void MlxKernelPreparation::finish() {
    const auto started = std::chrono::steady_clock::now();
    const auto total = jobs_.size();
    std::size_t completed = 0;
    for (const auto& job : jobs_) {
        std::cerr << "mfq_load_progress stage=compiling completed=" << completed
                  << " total=" << total << std::endl;
        job();
        ++completed;
    }
    std::cerr << "mfq_load_progress stage=compiling completed=" << completed
              << " total=" << total << std::endl;
    std::cerr << "mfq_kernel_prepare kernels=" << total << " unsupported=" << unsupported_.size()
              << " seconds=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()
              << std::endl;
    jobs_.clear();
    active_ = previous_;
}

} // namespace mfq::metal
