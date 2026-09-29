#include "../components.h"

#include "models/minicpmo45/causal_lm.h"
#include "cuda_execution.h"
#include "cuda_sampling.h"
#include "generation.h"
#include "mfq_tensor_backend.h"
#include "tensor_parallel.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mfq::cuda::minicpmo45 {

using internal::PrefillCudaTimer;

namespace {
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

} // namespace

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
            runtime.language.execution->layer_placement.primary_device());
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
            runtime.language.execution->layer_placement.primary_device());
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
            runtime.language.execution->layer_placement.primary_device());
        session.reset();
        runtime.language.reset(1);
        runtime.audio.reset();
        runtime.tts.reset(1);
        mfq_cuda_synchronize();
    };
    return backend;
}


struct EngineComponents {
    explicit EngineComponents(mfq::cuda::MiniCPMO45CausalLm language)
        : runtime(MiniCPMO45Runtime::load_with_language(
              std::move(language))) {}

    MiniCPMO45Runtime runtime;
    std::optional<MiniCPMO45DuplexSession> duplex_session;
};

} // namespace mfq::cuda::minicpmo45

template <>
RuntimeComponents<mfq::cuda::MiniCPMO45CausalLm>
load_runtime_components(
        mfq::cuda::MiniCPMO45CausalLm& model,
        bool load_optional_components) {
    RuntimeComponents<mfq::cuda::MiniCPMO45CausalLm> result;
    result.graph = model.graph;
    result.plan = model.plan;
    if (!load_optional_components ||
            result.plan.vision == mfq::cuda::CudaVisionAdapter::none) {
        return result;
    }
    if (result.plan.vision !=
            mfq::cuda::CudaVisionAdapter::minicpmo45) {
        throw std::runtime_error(
            "unsupported MiniCPM-o CUDA vision adapter");
    }

    auto state = std::make_shared<
        mfq::cuda::minicpmo45::EngineComponents>(std::move(model));
    result.language_override = &state->runtime.language;
    result.engine_binder = [state](
            mfq::engine::Engine& engine,
            std::mutex& model_mutex) {
        engine.multimodal_generate = [state, &model_mutex](
                const std::vector<int64_t>& prompt,
                const MfqVisionInput& vision,
                const MfqSamplingParams& sampling,
                const MfqTokenCallback& on_token,
                const MfqPrefillCallback& on_prefill,
                const MfqPromptCachePlan&,
                const MfqTokenConstraintPtr& token_constraint,
                const MfqCancellationCheck&) {
            return mfq::cuda::minicpmo45::generate_multimodal_tokens(
                state->runtime,
                model_mutex,
                prompt,
                vision,
                sampling,
                on_token,
                on_prefill,
                token_constraint);
        };
        engine.duplex =
            mfq::cuda::minicpmo45::make_cuda_minicpmo45_duplex_backend(
                state->runtime,
                model_mutex,
                state->duplex_session);
    };
    result.vision_available = true;
    return result;
}

template <>
RuntimeComponents<mfq::cuda::MiniCPMOTtsCausalLm>
load_runtime_components(
        mfq::cuda::MiniCPMOTtsCausalLm& model,
        bool) {
    RuntimeComponents<mfq::cuda::MiniCPMOTtsCausalLm> result;
    result.graph = model.graph;
    result.plan = model.plan;
    return result;
}
