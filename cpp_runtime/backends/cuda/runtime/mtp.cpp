#include "mtp.h"

#include "cuda_execution.h"
#include "causal_lm.h"
#include "cuda_sampling.h"
#include "prepared_prompt.h"
#include "mfq_cuda_ops.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

namespace {

template <typename Model>
static mfq_tensor_backend::Tensor server_hidden_forward_chunked(
    Model& model,
    const mfq_tensor_backend::Tensor & ids,
    int64_t chunk_size,
    mfq_tensor_backend::Tensor * raw_hidden = nullptr) {
    MFQ_RUNTIME_CHECK(
        chunk_size > 0,
        "server prefill chunk size must be positive");
    MFQ_RUNTIME_CHECK(
        ids.dim() == 2 && ids.size(0) == 1 && ids.size(1) > 0,
        "server prefill IDs must have shape [1, tokens]");
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
static mfq_tensor_backend::Tensor server_hidden_forward_prepared_chunked(
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

class ServerPrefillCudaTimer {
public:
    ServerPrefillCudaTimer()
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

    ~ServerPrefillCudaTimer() {
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

struct CompactDistribution {
    std::vector<int32_t> tokens;
    std::vector<float> probabilities;

    float probability(int32_t token) const {
        const auto found = std::lower_bound(tokens.begin(), tokens.end(), token);
        return found == tokens.end() || *found != token
            ? 0.0f
            : probabilities[static_cast<size_t>(found - tokens.begin())];
    }
};

static void validate_compact_distribution(const CompactDistribution& distribution) {
    MFQ_RUNTIME_CHECK(
        !distribution.tokens.empty() &&
            distribution.tokens.size() == distribution.probabilities.size() &&
            std::is_sorted(distribution.tokens.begin(), distribution.tokens.end()),
        "invalid compact MTP distribution");
    double total = 0.0;
    for (const float probability : distribution.probabilities) {
        MFQ_RUNTIME_CHECK(
            std::isfinite(probability) && probability >= 0.0f,
            "invalid compact MTP probability");
        total += probability;
    }
    MFQ_RUNTIME_CHECK(
        std::abs(total - 1.0) <= 1.0e-5,
        "compact MTP distribution is not normalized");
}

static CompactDistribution compact_distribution_from_topk(
        const float* values,
        const int64_t* indices,
        int count,
        const MfqSamplingParams& sampling) {
    MFQ_RUNTIME_CHECK(
        count > 0 && std::isfinite(values[0]),
        "compact CUDA MTP logits have no finite maximum");
    std::vector<double> masses(static_cast<size_t>(count));
    double total = 0.0;
    for (int index = 0; index < count; ++index) {
        masses[static_cast<size_t>(index)] = std::exp(
            (static_cast<double>(values[index]) - values[0]) /
            sampling.temperature);
        total += masses[static_cast<size_t>(index)];
    }
    MFQ_RUNTIME_CHECK(
        total > 0.0 && std::isfinite(total),
        "invalid compact CUDA MTP softmax mass");
    int keep = count;
    double kept_mass = total;
    if (sampling.top_p < 1.0) {
        kept_mass = 0.0;
        for (int index = 0; index < count; ++index) {
            kept_mass += masses[static_cast<size_t>(index)];
            if (kept_mass >= sampling.top_p * total) {
                keep = index + 1;
                break;
            }
        }
    }
    std::vector<std::pair<int32_t, float>> entries;
    entries.reserve(static_cast<size_t>(keep));
    for (int index = 0; index < keep; ++index) {
        entries.emplace_back(
            static_cast<int32_t>(indices[index]),
            static_cast<float>(
                masses[static_cast<size_t>(index)] / kept_mass));
    }
    std::sort(entries.begin(), entries.end());
    CompactDistribution result;
    result.tokens.reserve(entries.size());
    result.probabilities.reserve(entries.size());
    for (const auto& [token, probability] : entries) {
        result.tokens.push_back(token);
        result.probabilities.push_back(probability);
    }
    validate_compact_distribution(result);
    return result;
}

static int32_t sample_compact(
        const CompactDistribution& distribution, double uniform) {
    validate_compact_distribution(distribution);
    uniform = std::clamp(uniform, 0.0, std::nextafter(1.0, 0.0));
    double cumulative = 0.0;
    int32_t fallback = distribution.tokens.back();
    for (size_t index = 0; index < distribution.tokens.size(); ++index) {
        if (distribution.probabilities[index] <= 0.0f) continue;
        fallback = distribution.tokens[index];
        cumulative += distribution.probabilities[index];
        if (uniform < cumulative) return fallback;
    }
    return fallback;
}

static CompactDistribution positive_residual(
        const CompactDistribution& target,
        const CompactDistribution& proposal) {
    CompactDistribution residual;
    residual.tokens.reserve(target.tokens.size() + proposal.tokens.size());
    residual.probabilities.reserve(residual.tokens.capacity());
    size_t target_index = 0;
    size_t proposal_index = 0;
    double total = 0.0;
    while (target_index < target.tokens.size() ||
            proposal_index < proposal.tokens.size()) {
        const int32_t token = proposal_index == proposal.tokens.size() ||
                (target_index < target.tokens.size() &&
                 target.tokens[target_index] < proposal.tokens[proposal_index])
            ? target.tokens[target_index]
            : proposal_index < proposal.tokens.size() &&
                    (target_index == target.tokens.size() ||
                     proposal.tokens[proposal_index] < target.tokens[target_index])
                ? proposal.tokens[proposal_index]
                : target.tokens[target_index];
        const float probability = std::max(
            0.0f, target.probability(token) - proposal.probability(token));
        if (probability > 0.0f) {
            residual.tokens.push_back(token);
            residual.probabilities.push_back(probability);
            total += probability;
        }
        if (target_index < target.tokens.size() &&
                target.tokens[target_index] == token) {
            ++target_index;
        }
        if (proposal_index < proposal.tokens.size() &&
                proposal.tokens[proposal_index] == token) {
            ++proposal_index;
        }
    }
    if (!(total > 0.0) || !std::isfinite(total)) return target;
    for (float& probability : residual.probabilities) {
        probability = static_cast<float>(probability / total);
    }
    return residual;
}

static mfq::cuda::mtp::ChainVerification verify_compact_chain(
        std::span<const int32_t> drafts,
        const std::vector<CompactDistribution>& proposals,
        const std::vector<CompactDistribution>& targets,
        std::span<const double> acceptance_uniforms,
        double sample_uniform) {
    MFQ_RUNTIME_CHECK(
        !drafts.empty() && proposals.size() == drafts.size() &&
            targets.size() == drafts.size() + 1 &&
            acceptance_uniforms.size() == drafts.size(),
        "compact MTP chain shapes are incompatible");
    for (size_t position = 0; position < drafts.size(); ++position) {
        validate_compact_distribution(proposals[position]);
        validate_compact_distribution(targets[position]);
        const double q = proposals[position].probability(drafts[position]);
        const double p = targets[position].probability(drafts[position]);
        MFQ_RUNTIME_CHECK(q > 0.0, "compact MTP draft has zero probability");
        const double acceptance = std::min(1.0, p / q);
        if (std::clamp(
                acceptance_uniforms[position], 0.0,
                std::nextafter(1.0, 0.0)) < acceptance) {
            continue;
        }
        return {
            position,
            sample_compact(
                positive_residual(targets[position], proposals[position]),
                sample_uniform),
            false};
    }
    return {
        drafts.size(), sample_compact(targets.back(), sample_uniform), true};
}

} // namespace

template <mfq::cuda::CudaBackbone Backbone>
int32_t run_mtp_generation(
        mfq::cuda::CausalLmFor<Backbone>& model, MtpModule& mtp,
        const std::vector<int64_t>& prompt, const MfqSamplingParams& sampling,
        const MfqTokenCallback& on_token, const MfqPrefillCallback& on_prefill,
        int64_t prefill_chunk_size,
        const MfqTokenConstraintPtr& token_constraint,
        const CudaPreparedPrompt* prepared) {
    using Tensor = mfq_tensor_backend::Tensor;
    using Clock = std::chrono::steady_clock;
    namespace policy = mfq::cuda::mtp;
    const MtpTarget target{
        [&model](Tensor ids) {
            return model.embed_forward(std::move(ids));
        },
        [&model](Tensor hidden) {
            return model.logits_from_hidden(std::move(hidden));
        },
        &model.rope,
    };
    MFQ_RUNTIME_CHECK(
        !prompt.empty() && prompt.size() <=
            static_cast<size_t>(model.max_position_embeddings()),
        "invalid MTP prompt length");
    for (auto token : prompt) {
        MFQ_RUNTIME_CHECK(
            token >= 0 && token < model.vocab_size(),
            "MTP prompt token outside vocabulary");
    }
    const bool transformed_prompt = prepared != nullptr && prepared->transformed();
    MFQ_RUNTIME_CHECK(
        prepared == nullptr || prepared->token_ids == prompt,
        "prepared CUDA MTP prompt disagrees with rendered token IDs");
    MFQ_RUNTIME_CHECK(
        !transformed_prompt ||
            (prepared->embeddings.defined() && prepared->positions.defined()),
        "prepared CUDA MTP prompt is missing embeddings or positions");
    MFQ_RUNTIME_CHECK(
        mfq_token_constraint_supports_speculation(token_constraint),
        "CUDA MTP token constraint must support allows/apply/accept/clone");
    mtp.last_stats = {};
    mtp.last_stats.available = true;
    mtp.last_stats.used = true;
    mtp.last_cycles = mtp.last_accepted = mtp.last_rejected = 0;
    const int64_t next_logical_position =
        static_cast<int64_t>(prompt.size()) +
        (transformed_prompt ? prepared->decode_position_delta : 0);
    MFQ_RUNTIME_CHECK(
        next_logical_position >= 0 &&
            next_logical_position <= model.max_position_embeddings(),
        "prepared CUDA MTP decode position is outside context capacity");
    const int64_t occupied_context = std::max<int64_t>(
        static_cast<int64_t>(prompt.size()), next_logical_position);
    const int32_t limit = static_cast<int32_t>(std::min<int64_t>(
        sampling.max_tokens,
        model.max_position_embeddings() - occupied_context));
    if (limit <= 0) return 0;
    const auto options = mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA)
        .dtype(mfq_tensor_backend::kInt64);
    auto ids_for = [&](std::vector<int64_t> tokens) {
        return mfq_tensor_backend::tensor(tokens, options).reshape({1, -1}).contiguous();
    };
    auto input_ids = ids_for(prompt);
    const int position_axes = transformed_prompt &&
            prepared->positions.dim() == 2 &&
            prepared->positions.size(0) == 3
        ? 3
        : 1;
    auto decode_positions = [&](int64_t cache_start, int64_t tokens) {
        MFQ_RUNTIME_CHECK(tokens > 0, "CUDA MTP position span must be positive");
        const int64_t logical_start = cache_start +
            (transformed_prompt ? prepared->decode_position_delta : 0);
        MFQ_RUNTIME_CHECK(
            logical_start >= 0 &&
                logical_start + tokens <= model.max_position_embeddings(),
            "CUDA MTP position span exceeds context capacity");
        auto result = mfq_tensor_backend::arange(
            logical_start, logical_start + tokens, options);
        return position_axes == 3
            ? result.reshape({1, tokens}).expand({3, tokens}).contiguous()
            : result;
    };
    auto predictor_step = [&](Tensor hidden, Tensor ids, Tensor positions) {
        return positions.defined()
            ? mtp.step_positioned(
                  target, std::move(hidden), std::move(ids),
                  std::move(positions))
            : mtp.step(target, std::move(hidden), std::move(ids));
    };
    auto random_host = mfq_tensor_backend::empty({1}, mfq_tensor_backend::TensorOptions()
        .device(mfq_tensor_backend::kCPU).dtype(mfq_tensor_backend::kFloat32).pinned_memory(true));
    auto random_gpu = mfq_tensor_backend::empty({1}, options.dtype(mfq_tensor_backend::kFloat32));
    mfq::cuda::Sampler sampler(
        sampling,
        mfq::cuda::SamplingOps(
            std::move(random_host), std::move(random_gpu)));
    const bool penalties = sampler.has_penalties();
    auto counts = penalties
        ? mfq_tensor_backend::zeros(
            {model.vocab_size()},
            options.dtype(mfq_tensor_backend::kInt32))
        : Tensor{};
    if (penalties) sample_token_counts_add_cuda(counts, input_ids);
    const bool greedy = sampler.greedy();
    auto sample_normal = [&](Tensor logits, Tensor token_counts) {
        if (penalties) logits = logits.clone();
        return static_cast<int32_t>(mfq::cuda::sample_logits(
            sampler, std::move(logits), token_counts).template item<int64_t>());
    };
    auto sample_constrained = [&](Tensor logits, Tensor token_counts,
                                  const MfqTokenConstraintPtr& constraint) {
        if (penalties) logits = logits.clone();
        auto sampler_constraint = constraint
            ? constraint->clone()
            : MfqTokenConstraintPtr{};
        return static_cast<int32_t>(mfq::cuda::sample_logits(
            sampler, std::move(logits), token_counts,
            sampler_constraint).template item<int64_t>());
    };
    auto probabilities = [&](Tensor logits, Tensor token_counts,
                             const MfqSamplingParams& parameters) {
        logits = logits.contiguous().reshape({1, -1});
        if (penalties) {
            logits = logits.clone();
            sample_apply_penalties_cuda(logits, token_counts,
                parameters.presence_penalty, parameters.frequency_penalty,
                parameters.repetition_penalty);
        }
        auto host = logits.to(mfq_tensor_backend::kFloat32).cpu().contiguous();
        return policy::distribution(std::span<const float>(host.template data_ptr<float>(), host.numel()),
            parameters.temperature, parameters.top_k, parameters.top_p);
    };
    auto compact_probabilities = [&](Tensor logits, Tensor token_counts,
                                     const MfqSamplingParams& parameters) {
        MFQ_RUNTIME_CHECK(
            parameters.top_k > 0 && parameters.top_k <= 64 &&
                parameters.temperature > 0.0,
            "compact CUDA MTP sampling requires top-k 1-64");
        logits = logits.contiguous().reshape({1, -1});
        if (penalties) {
            logits = logits.clone();
            sample_apply_penalties_cuda(logits, token_counts,
                parameters.presence_penalty, parameters.frequency_penalty,
                parameters.repetition_penalty);
        }
        auto selected = mfq_tensor_backend::topk(
            logits, parameters.top_k, -1, true, true);
        auto values = std::get<0>(selected)
            .to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kFloat32)
            .contiguous().reshape({-1});
        auto indices = std::get<1>(selected)
            .to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt64)
            .contiguous().reshape({-1});
        return compact_distribution_from_topk(
            values.template data_ptr<float>(), indices.template data_ptr<int64_t>(),
            static_cast<int>(values.numel()), parameters);
    };
    auto compact_probability_rows = [&](Tensor logits,
                                        const MfqSamplingParams& parameters) {
        MFQ_RUNTIME_CHECK(
            !penalties && logits.dim() == 2 &&
                parameters.top_k > 0 && parameters.top_k <= 64 &&
                parameters.temperature > 0.0,
            "batched compact CUDA MTP sampling geometry is invalid");
        logits = logits.contiguous();
        auto selected = mfq_tensor_backend::topk(
            logits, parameters.top_k, -1, true, true);
        auto values = std::get<0>(selected)
            .to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kFloat32)
            .contiguous();
        auto indices = std::get<1>(selected)
            .to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt64)
            .contiguous();
        const int64_t rows = values.size(0);
        const int64_t columns = values.size(1);
        const float* value_data = values.template data_ptr<float>();
        const int64_t* index_data = indices.template data_ptr<int64_t>();
        std::vector<CompactDistribution> result;
        result.reserve(static_cast<size_t>(rows));
        for (int64_t row = 0; row < rows; ++row) {
            result.push_back(compact_distribution_from_topk(
                value_data + row * columns,
                index_data + row * columns,
                static_cast<int>(columns), parameters));
        }
        return result;
    };
    auto logits_for = [&](Tensor normalized) {
        return model.logits_from_hidden((mtp.preserve_output_dtype()?normalized:
            normalized.to(mfq_tensor_backend::kFloat16)).contiguous());
    };
    int32_t generated = 0;
    auto emit = [&](int32_t token) {
        MFQ_RUNTIME_CHECK(
            token >= 0 && token < model.vocab_size(),
            "MTP sampled token outside vocabulary");
        if (token_constraint) token_constraint->accept(token);
        ++generated;
        if (penalties) sample_token_counts_add_cuda(counts, ids_for({token}));
        return !on_token || on_token(token);
    };
    struct DraftChain {
        std::vector<int32_t> tokens;
        Tensor device_tokens;
        std::vector<std::vector<float>> probabilities;
        std::vector<CompactDistribution> compact_probabilities;
    };
    const bool compact_stochastic =
        !greedy && sampling.top_k > 0 && sampling.top_k <= 64;
    auto draft_sampling = sampling;
    if (compact_stochastic) {
        draft_sampling.temperature = 0.6;
        draft_sampling.top_p = 0.95;
    }
    const int maximum_depth = std::min(
        mtp.maximum_draft_depth(),
        (!greedy && !compact_stochastic)
            ? 1
            : std::clamp<int>(sampling.mtp_max_draft_tokens, 1,
                              policy::kMaximumDraftDepth));
    policy::DepthController depth_controller(maximum_depth);

    auto generate = [&]() {
        model.reset(1);
        mtp.reset(1);
        ServerPrefillCudaTimer timer;
        Tensor raw;
        auto hidden = transformed_prompt
            ? server_hidden_forward_prepared_chunked(
                  model, input_ids, *prepared, prefill_chunk_size, &raw)
            : server_hidden_forward_chunked(
                  model, input_ids, prefill_chunk_size, &raw);
        auto logits = logits_for(hidden.narrow(1, hidden.size(1) - 1, 1));
        MFQ_CUDA_CHECK(cudaEventRecord(timer.finished_event(), mfq_get_current_cuda_stream()));
        auto initial_constraint = token_constraint
            ? token_constraint->clone()
            : MfqTokenConstraintPtr{};
        MFQ_RUNTIME_CHECK(
            !token_constraint ||
                (initial_constraint &&
                 mfq_token_constraint_supports_speculation(
                     initial_constraint)),
            "CUDA MTP token constraint clone is incomplete");
        int32_t pending = sample_constrained(
            logits, counts, initial_constraint);
        const double prefill_ms = timer.elapsed_ms();
        if (on_prefill) on_prefill(MfqPrefillTiming{prompt.size(), prefill_ms, 0., prefill_ms});
        if (!emit(pending) || generated == limit) return generated;
        auto constraint_cursor = token_constraint
            ? token_constraint->clone()
            : MfqTokenConstraintPtr{};
        MFQ_RUNTIME_CHECK(
            !token_constraint ||
                (constraint_cursor &&
                 mfq_token_constraint_supports_speculation(
                     constraint_cursor)),
            "CUDA MTP token constraint cursor is incomplete");

        if (mtp.blockwise_drafting()) {
            mtp.append_target_context(raw, 0);
        }

        if (mtp.teacher_forced_prompt_prime() && prompt.size() > 1) {
            constexpr int64_t chunk_size = 512;
            const int64_t pairs = raw.size(1) - 1;
            for (int64_t offset = 0; offset < pairs; offset += chunk_size) {
                const int64_t count = std::min(chunk_size, pairs - offset);
                (void)predictor_step(
                    raw.narrow(1, offset, count),
                    input_ids.narrow(1, offset + 1, count),
                    transformed_prompt
                        ? prepared->positions.narrow(
                              -1, offset + 1, count).contiguous()
                        : Tensor{});
            }
        }

        auto initial_hidden = raw.narrow(1, raw.size(1) - 1, 1);
        if (mtp.target_bootstrap_decode()) {
            if (mtp.teacher_forced_prompt_prime()) {
                // Some recurrent predictors consume the prompt/first-token
                // seam before the target advances to hidden(first_token).
                (void)predictor_step(
                    initial_hidden,
                    ids_for({pending}),
                    transformed_prompt
                        ? decode_positions(model.cache_pos, 1)
                        : Tensor{});
            }
            auto next_hidden = model.hidden_forward(
                ids_for({pending}), mfq_nullopt, mfq_nullopt,
                nullptr, mfq_nullopt, &raw);
            pending = sample_constrained(
                logits_for(next_hidden), counts, constraint_cursor);
            if (constraint_cursor) constraint_cursor->accept(pending);
            if (!emit(pending) || generated == limit) return generated;
            initial_hidden = raw.narrow(1, raw.size(1) - 1, 1);
        }

        int64_t predictor_history_position = mtp.cache_position();
        auto bounded_depth = [&](int desired) {
            const auto context_depth = std::max<int64_t>(
                0, model.max_position_embeddings() - model.cache_pos - 1);
            const auto output_depth = std::max<int64_t>(
                0, static_cast<int64_t>(limit - generated - 1));
            return static_cast<int>(std::min<int64_t>(
                desired, std::min(context_depth, output_depth)));
        };
        auto prepare_draft = [&](Tensor hidden_rows,
                                 const std::vector<int32_t>& next_ids,
                                 int requested_depth,
                                 bool initial) {
            MFQ_RUNTIME_CHECK(
                hidden_rows.dim() == 3 && hidden_rows.size(0) == 1 &&
                    hidden_rows.size(1) == static_cast<int64_t>(next_ids.size()) &&
                    !next_ids.empty() && requested_depth >= 0 &&
                    requested_depth <= maximum_depth,
                "CUDA MTP committed history is incompatible");
            mtp.trim_cache_to(predictor_history_position);
            auto prospective_counts = penalties ? counts.clone() : Tensor{};
            DraftChain result;
            result.tokens.reserve(static_cast<size_t>(requested_depth));
            result.probabilities.reserve(static_cast<size_t>(requested_depth));
            result.compact_probabilities.reserve(
                static_cast<size_t>(requested_depth));
            auto select_draft = [&](Tensor draft_logits) {
                draft_logits = draft_logits.reshape({1, -1});
                int32_t token = -1;
                if (greedy) {
                    token = sample_normal(draft_logits, prospective_counts);
                } else if (compact_stochastic) {
                    auto proposal = compact_probabilities(
                        draft_logits, prospective_counts, draft_sampling);
                    token = sample_compact(proposal, sampler.next_uniform());
                    result.compact_probabilities.push_back(
                        std::move(proposal));
                } else {
                    auto proposal = probabilities(
                        draft_logits, prospective_counts, draft_sampling);
                    token = policy::sample(proposal, sampler.next_uniform());
                    result.probabilities.push_back(std::move(proposal));
                }
                result.tokens.push_back(token);
                if (penalties) {
                    sample_token_counts_add_cuda(
                        prospective_counts, ids_for({token}));
                }
                return token;
            };
            if (mtp.blockwise_drafting()) {
                if (!initial) {
                    mtp.append_target_context(
                        hidden_rows, predictor_history_position);
                    predictor_history_position +=
                        static_cast<int64_t>(next_ids.size());
                }
                MFQ_RUNTIME_CHECK(
                    mtp.cache_position() == predictor_history_position,
                    "CUDA block predictor cache did not advance");
                if (requested_depth > 0) {
                    auto block = mtp.draft_block(
                        target,
                        ids_for({next_ids.back()}),
                        select_draft,
                        requested_depth);
                    MFQ_RUNTIME_CHECK(
                        block.tokens.numel() == requested_depth &&
                            block.logits.size(1) == requested_depth &&
                            block.confidence.numel() == requested_depth &&
                            result.tokens.size() ==
                                static_cast<size_t>(requested_depth),
                        "CUDA block predictor returned an incomplete draft");
                }
                return result;
            }
            std::vector<int64_t> shifted(next_ids.begin(), next_ids.end());
            auto head = predictor_step(
                std::move(hidden_rows),
                ids_for(std::move(shifted)),
                transformed_prompt
                    ? decode_positions(
                          predictor_history_position + 1,
                          static_cast<int64_t>(next_ids.size()))
                    : Tensor{});
            predictor_history_position += static_cast<int64_t>(next_ids.size());
            MFQ_RUNTIME_CHECK(
                mtp.cache_position() == predictor_history_position,
                "CUDA MTP predictor cache did not advance");
            auto sample_hidden = head.sample_hidden.narrow(
                1, head.sample_hidden.size(1) - 1, 1);
            auto chain_hidden = head.chain_hidden.narrow(
                1, head.chain_hidden.size(1) - 1, 1);
            const bool device_greedy =
                greedy && !penalties && !mtp.split_target_verification();
            std::vector<Tensor> device_tokens;
            device_tokens.reserve(static_cast<size_t>(requested_depth));
            for (int position = 0; position < requested_depth; ++position) {
                auto draft_logits = logits_for(sample_hidden).reshape({1, -1});
                Tensor device_token;
                int32_t token = -1;
                if (device_greedy) {
                    device_token = draft_logits.argmax(-1).contiguous();
                    device_tokens.push_back(device_token);
                } else {
                    token = select_draft(draft_logits);
                }
                if (position + 1 < requested_depth) {
                    auto next = predictor_step(
                        chain_hidden,
                        device_greedy
                            ? device_token.reshape({1, 1})
                            : ids_for({token}),
                        transformed_prompt
                            ? decode_positions(
                                  predictor_history_position + position + 1,
                                  1)
                            : Tensor{});
                    sample_hidden = std::move(next.sample_hidden);
                    chain_hidden = std::move(next.chain_hidden);
                }
            }
            if (device_greedy && !device_tokens.empty()) {
                result.device_tokens =
                    mfq_tensor_backend::cat(device_tokens, 0).contiguous();
                result.tokens.resize(
                    static_cast<size_t>(result.device_tokens.numel()), -1);
            }
            return result;
        };

        Tensor draft_hidden_rows = initial_hidden;
        std::vector<int32_t> draft_next_ids{pending};
        bool initial_draft = true;
        while (generated < limit) {
            const auto cycle_started = Clock::now();
            auto draft = prepare_draft(
                draft_hidden_rows, draft_next_ids,
                bounded_depth(depth_controller.depth()), initial_draft);
            initial_draft = false;
            const int draft_count = static_cast<int>(draft.tokens.size());
            Tensor verified_raw;
            Tensor verify_tensor;
            if (draft.device_tokens.defined()) {
                verify_tensor = mfq_tensor_backend::cat(
                    {ids_for({pending}),
                     draft.device_tokens.reshape({1, -1})},
                    1).contiguous();
            } else {
                std::vector<int64_t> verify_ids{pending};
                verify_ids.insert(
                    verify_ids.end(),
                    draft.tokens.begin(), draft.tokens.end());
                verify_tensor = ids_for(std::move(verify_ids));
            }
            Tensor verified;
            if (mtp.split_target_verification()) {
                Tensor pending_raw;
                auto pending_hidden = model.hidden_forward(
                    ids_for({pending}), mfq_nullopt, mfq_nullopt,
                    nullptr, mfq_nullopt, &pending_raw);
                if (draft_count > 0) {
                    model.begin_speculative_suffix(draft_count);
                    std::vector<int64_t> draft_ids(
                        draft.tokens.begin(), draft.tokens.end());
                    Tensor draft_raw;
                    auto draft_hidden = model.hidden_forward_speculative_suffix(
                        ids_for(std::move(draft_ids)), &draft_raw);
                    verified = mfq_tensor_backend::cat(
                        {pending_hidden, draft_hidden}, 1).contiguous();
                    verified_raw = mfq_tensor_backend::cat(
                        {pending_raw, draft_raw}, 1).contiguous();
                } else {
                    verified = std::move(pending_hidden);
                    verified_raw = std::move(pending_raw);
                }
            } else {
                verified = model.hidden_forward(
                    std::move(verify_tensor), mfq_nullopt, mfq_nullopt,
                    nullptr, mfq_nullopt, &verified_raw,
                    draft_count > 0 ? 1 : 0);
            }
            auto targets = logits_for(verified).reshape(
                {draft_count + 1, model.vocab_size()});

            ++mtp.last_stats.cycles;
            mtp.last_stats.drafted_tokens += static_cast<uint64_t>(draft_count);
            ++mtp.last_stats.depth_cycles.at(static_cast<size_t>(draft_count));
            for (int position = 0; position < draft_count; ++position) {
                ++mtp.last_stats.position_drafted.at(static_cast<size_t>(position));
            }

            auto row_counts = penalties ? counts.clone() : Tensor{};
            policy::ChainVerification result;
            if (draft_count == 0) {
                // No proposal distribution is needed: use the ordinary GPU
                // sampler rather than transferring a vocabulary row.
                result = {
                    0,
                    sample_normal(targets.narrow(0, 0, 1), row_counts),
                    true};
            } else if (greedy) {
                std::vector<int32_t> target_tokens;
                target_tokens.reserve(static_cast<size_t>(draft_count + 1));
                if (!penalties) {
                    auto device_targets = targets.argmax(-1).contiguous();
                    Tensor host_tokens;
                    if (draft.device_tokens.defined()) {
                        host_tokens = mfq_tensor_backend::cat(
                            {draft.device_tokens.reshape({-1}),
                             device_targets.reshape({-1})},
                            0).to(
                                mfq_tensor_backend::kCPU,
                                mfq_tensor_backend::kInt64)
                            .contiguous();
                    } else {
                        host_tokens = device_targets.to(
                            mfq_tensor_backend::kCPU,
                            mfq_tensor_backend::kInt64).contiguous();
                    }
                    const auto* data =
                        host_tokens.template data_ptr<int64_t>();
                    if (draft.device_tokens.defined()) {
                        draft.tokens.assign(data, data + draft_count);
                        data += draft_count;
                    }
                    target_tokens.assign(
                        data, data + draft_count + 1);
                } else {
                    for (int row = 0; row <= draft_count; ++row) {
                        target_tokens.push_back(sample_normal(
                            targets.narrow(0, row, 1), row_counts));
                        if (row < draft_count) {
                            sample_token_counts_add_cuda(
                                row_counts,
                                ids_for({draft.tokens[
                                    static_cast<size_t>(row)]}));
                        }
                    }
                }
                result = policy::verify_greedy(draft.tokens, target_tokens);
            } else if (compact_stochastic) {
                std::vector<CompactDistribution> target_probabilities;
                if (!penalties) {
                    // One device top-k and one compact host transfer for the
                    // complete verification window.
                    target_probabilities = compact_probability_rows(
                        targets, sampling);
                } else {
                    target_probabilities.reserve(
                        static_cast<size_t>(draft_count + 1));
                    for (int row = 0; row <= draft_count; ++row) {
                        target_probabilities.push_back(compact_probabilities(
                            targets.narrow(0, row, 1), row_counts, sampling));
                        if (row < draft_count) {
                            sample_token_counts_add_cuda(
                                row_counts,
                                ids_for({draft.tokens[static_cast<size_t>(row)]}));
                        }
                    }
                }
                std::vector<double> acceptance_uniforms(
                    static_cast<size_t>(draft_count));
                std::generate(
                    acceptance_uniforms.begin(), acceptance_uniforms.end(),
                    [&] { return sampler.next_uniform(); });
                result = verify_compact_chain(
                    draft.tokens, draft.compact_probabilities,
                    target_probabilities, acceptance_uniforms,
                    sampler.next_uniform());
            } else {
                std::vector<std::vector<float>> target_probabilities;
                target_probabilities.reserve(static_cast<size_t>(draft_count + 1));
                for (int row = 0; row <= draft_count; ++row) {
                    target_probabilities.push_back(probabilities(
                        targets.narrow(0, row, 1), row_counts, sampling));
                    if (row < draft_count && penalties) {
                        sample_token_counts_add_cuda(
                            row_counts, ids_for({draft.tokens[static_cast<size_t>(row)]}));
                    }
                }
                std::vector<double> acceptance_uniforms(
                    static_cast<size_t>(draft_count));
                std::generate(
                    acceptance_uniforms.begin(), acceptance_uniforms.end(),
                    [&] { return sampler.next_uniform(); });
                result = policy::verify_stochastic_chain(
                    draft.tokens, draft.probabilities, target_probabilities,
                    acceptance_uniforms, sampler.next_uniform());
            }

            int accepted = static_cast<int>(result.accepted_drafts);
            MFQ_RUNTIME_CHECK(
                accepted >= 0 && accepted <= draft_count &&
                    result.next_token >= 0 &&
                    result.next_token < model.vocab_size(),
                "CUDA MTP verification returned invalid data");
            if (constraint_cursor) {
                auto constraint_counts = penalties ? counts.clone() : Tensor{};
                bool corrected = false;
                for (int position = 0; position < accepted; ++position) {
                    const int32_t token =
                        draft.tokens[static_cast<size_t>(position)];
                    if (!constraint_cursor->allows(token)) {
                        accepted = position;
                        result.next_token = sample_constrained(
                            targets.narrow(0, position, 1),
                            constraint_counts,
                            constraint_cursor);
                        result.bonus = false;
                        corrected = true;
                        break;
                    }
                    constraint_cursor->accept(token);
                    if (penalties) {
                        sample_token_counts_add_cuda(
                            constraint_counts, ids_for({token}));
                    }
                }
                if (!corrected) {
                    if (!constraint_cursor->allows(result.next_token)) {
                        result.next_token = sample_constrained(
                            targets.narrow(0, accepted, 1),
                            constraint_counts,
                            constraint_cursor);
                        result.bonus = false;
                    }
                }
                constraint_cursor->accept(result.next_token);
                result.accepted_drafts = static_cast<size_t>(accepted);
            }
            mtp.last_stats.accepted_tokens += static_cast<uint64_t>(accepted);
            for (int position = 0; position < accepted; ++position) {
                ++mtp.last_stats.position_accepted.at(static_cast<size_t>(position));
            }

            int emitted_accepted = 0;
            bool continue_generation = true;
            for (int position = 0; position < accepted; ++position) {
                ++emitted_accepted;
                if (!emit(draft.tokens[static_cast<size_t>(position)])) {
                    continue_generation = false;
                    break;
                }
            }
            if (draft_count > 0) {
                if (emitted_accepted == draft_count) {
                    model.commit_speculative();
                } else {
                    const bool retained =
                        mtp.retains_partial_target_prefix();
                    model.rollback_speculative(
                        retained ? emitted_accepted : 0);
                    if (!retained && emitted_accepted > 0) {
                        std::vector<int64_t> replay(
                            draft.tokens.begin(),
                            draft.tokens.begin() + emitted_accepted);
                        (void)model.hidden_forward(ids_for(std::move(replay)));
                    }
                }
            }
            if (!continue_generation) return generated;
            if (!emit(result.next_token)) return generated;

            const double cycle_ms = std::chrono::duration<double, std::milli>(
                Clock::now() - cycle_started).count();
            depth_controller.observe(draft_count, accepted, cycle_ms);
            mtp.last_stats.selected_depth = depth_controller.depth();
            for (int depth = 0; depth <= depth_controller.maximum_depth(); ++depth) {
                if (const auto measured = depth_controller.measured_cycle_ms(depth)) {
                    mtp.last_stats.measured_depth_ms.at(
                        static_cast<size_t>(depth)) = *measured;
                }
            }

            if (draft_count > 0) {
                if (accepted == draft_count) {
                    mtp.last_accepted += static_cast<uint64_t>(accepted);
                } else {
                    mtp.last_accepted += static_cast<uint64_t>(accepted);
                    ++mtp.last_rejected;
                }
            }
            ++mtp.last_cycles;
            pending = result.next_token;
            draft_hidden_rows = verified_raw.narrow(1, 0, accepted + 1);
            draft_next_ids.clear();
            draft_next_ids.reserve(static_cast<size_t>(accepted + 1));
            draft_next_ids.insert(
                draft_next_ids.end(), draft.tokens.begin(),
                draft.tokens.begin() + accepted);
            draft_next_ids.push_back(pending);
        }
        return generated;
    };
    try {
        const auto result = generate();
        std::cerr << "mtp generated=" << result << " cycles=" << mtp.last_cycles
            << " drafted=" << mtp.last_stats.drafted_tokens
            << " accepted=" << mtp.last_accepted
            << " rejected=" << mtp.last_rejected
            << " depth=" << mtp.last_stats.selected_depth << '\n';
        return result;
    } catch (...) {
        // A failed partial pass must never become the next request's history.
        try { model.reset(1); mtp.reset(1); } catch (...) {}
        throw;
    }
}

#define MFQ_INSTANTIATE_MTP(BACKBONE)                                      \
    template int32_t run_mtp_generation<BACKBONE>(                         \
        mfq::cuda::CausalLmFor<BACKBONE>&, MtpModule&,                         \
        const std::vector<int64_t>&, const MfqSamplingParams&,              \
        const MfqTokenCallback&, const MfqPrefillCallback&, int64_t,         \
        const MfqTokenConstraintPtr&, const CudaPreparedPrompt*)

MFQ_INSTANTIATE_MTP(mfq::cuda::CudaBackbone::generic_qwen);
MFQ_INSTANTIATE_MTP(mfq::cuda::CudaBackbone::glm5_next);
MFQ_INSTANTIATE_MTP(mfq::cuda::CudaBackbone::qwen4_exp);
MFQ_INSTANTIATE_MTP(mfq::cuda::CudaBackbone::deepseek_v41);

#undef MFQ_INSTANTIATE_MTP
