#pragma once
#include "setup.h"
#include "causal_lm.h"
#include "runtime_components.h"
#include "cuda_execution.h"
#include "mfq_cuda_mtp.h"
#include "tensor_parallel.h"
#include "moe_expert_cache.h"
#include <chrono>
#include <cstdint>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace mfq::cuda::internal {
std::vector<int64_t> parse_ids(const std::string& value);
KlMmqMode parse_kl_mmq_mode(const std::string& value);
std::vector<KlMmqMode> parse_kl_mmq_sequence(const std::string& value);
std::vector<int64_t> load_ids_file(const std::string& path);
std::unordered_set<int> parse_layer_ranges(const std::string& value);

template <typename F>
int with_loaded_cuda_model(mfq::cuda::CudaLoadOptions& options,
        bool load_optional_components, F&& run) {
    auto dispatch_source = mfq::open_model_source(options.model_path);
    auto run_loaded = [&]<mfq::cuda::CudaBackbone Backbone>() -> int {
        using Model = mfq::cuda::CausalLmFor<Backbone>;
        auto t0 = std::chrono::steady_clock::now();
        Model model = mfq::cuda::load_causal_lm<Backbone>(
            options.model_path,
            options.config_path,
            options.context_size,
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
        return run.template operator()<Backbone>(model, runtime_components, t0, t1);
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
 }

} // namespace mfq::cuda::internal
int run_qwen35_mtp_bench(mfq::cuda::Qwen35CausalLm& model, Qwen35Mtp& mtp,
    bool enable_mtp, int generated_tokens, int repetitions);
int run_cuda_continuous_batching_check(mfq::cuda::Qwen35CausalLm& model);
