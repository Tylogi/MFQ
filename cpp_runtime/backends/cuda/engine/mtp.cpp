#include "mtp.h"

#include "cuda_execution.h"
#include "models/causal_models.h"
#include "cuda_sampling.h"
#include "generation_policy.h"
#include "inference.h"
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

using mfq::cuda::internal::PrefillCudaTimer;

using mfq::cuda::internal::Generation;
using namespace mfq::engine;
template <typename Model>
Generation run_mtp_generation(Model& model, MtpModule& mtp, InferenceRequest& request,
        InferenceOutput& output, int64_t prefill_chunk_size,
        const CudaPreparedPrompt* prepared, std::size_t reused_tokens,
        mfq_tensor_backend::Tensor restored_last_hidden,
        mfq_tensor_backend::Tensor* session_last_hidden) {
    const auto& prompt = request.prompt;
    const auto& sampling = request.sampling;
    const auto& token_constraint = request.token_constraint;
    using Tensor = mfq_tensor_backend::Tensor;
    using Clock = std::chrono::steady_clock;
    namespace policy = mfq::engine::mtp;
    using policy::CompactDistribution;
    const MtpTarget target(model);
    if (session_last_hidden != nullptr) {
        *session_last_hidden = Tensor();
    }
    if (output.result.cancelled) co_return;
    MFQ_RUNTIME_CHECK(
        reused_tokens < prompt.size(),
        "restored MTP prefix must be shorter than the prompt");
    const bool transformed_prompt = prepared != nullptr && prepared->transformed();
    MFQ_RUNTIME_CHECK(
        prepared == nullptr || prepared->token_ids == prompt,
        "prepared CUDA MTP prompt disagrees with rendered token IDs");
    MFQ_RUNTIME_CHECK(
        !transformed_prompt ||
            (prepared->embeddings.defined() && prepared->positions.defined()),
        "prepared CUDA MTP prompt is missing embeddings or positions");
    MFQ_RUNTIME_CHECK(
        reused_tokens == 0 ||
            (mtp.supports_session_state() && restored_last_hidden.defined() &&
             model.cache_pos == static_cast<int64_t>(reused_tokens) &&
             mtp.cache_position() == static_cast<int64_t>(reused_tokens) - 1),
        "restored CUDA MTP session boundary is incompatible");
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
    const auto generation_plan = mfq::engine::plan_generation(
        prompt, model.vocab_size(), model.max_position_embeddings(),
        sampling.max_tokens, 0, occupied_context);
    const int32_t limit = generation_plan.generation_tokens;
    if (limit <= 0) co_return;
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
        return policy::compact_distribution_from_topk(
            values.template data_ptr<float>(),
            indices.template data_ptr<int64_t>(),
            static_cast<int>(values.numel()),
            parameters.temperature, parameters.top_p);
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
            result.push_back(policy::compact_distribution_from_topk(
                value_data + row * columns,
                index_data + row * columns,
                static_cast<int>(columns),
                parameters.temperature, parameters.top_p));
        }
        return result;
    };
    auto logits_for = [&](Tensor normalized) {
        return model.logits_from_hidden((mtp.preserve_output_dtype()?normalized:
            normalized.to(mfq_tensor_backend::kFloat16)).contiguous());
    };
    int32_t generated = 0;
    auto accept = [&](const TokenOutput& delta) {
        for (const auto token : delta.token_ids) {
            MFQ_RUNTIME_CHECK(token >= 0 && token < model.vocab_size(), "MTP sampled token outside vocabulary");
            if (token_constraint) token_constraint->accept(token);
            ++generated;
            if (penalties) sample_token_counts_add_cuda(counts, ids_for({token}));
        }
    };
    struct DraftChain {
        std::vector<int32_t> tokens;
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

    {
        if (reused_tokens == 0) {
            model.reset(1);
            mtp.reset(1);
        }
        Tensor raw, hidden;
        std::vector<Tensor> raw_chunks;
        double prefill_ms = 0.0;
        for (int64_t offset = static_cast<int64_t>(reused_tokens); offset < input_ids.size(1);) {
            if (output.result.cancelled) co_return;
            const auto chunk = next_prefill_chunk(input_ids.size(1), offset, prefill_chunk_size);
            PrefillCudaTimer timer;
            Tensor raw_chunk;
            auto ids = input_ids.narrow(1, chunk.offset, chunk.count).contiguous();
            hidden = transformed_prompt
                ? model.hidden_forward_inputs(ids,
                    prepared->embeddings.narrow(1, chunk.offset, chunk.count).contiguous(),
                    prepared->positions.narrow(-1, chunk.offset, chunk.count).contiguous(),
                    mfq_nullopt, nullptr, mfq_nullopt, true, mfq_nullopt, &raw_chunk)
                : model.hidden_forward(ids, mfq_nullopt, mfq_nullopt, nullptr, mfq_nullopt, &raw_chunk);
            raw_chunks.push_back(std::move(raw_chunk));
            offset += chunk.count;
            MFQ_CUDA_CHECK(cudaEventRecord(timer.finished_event(), mfq_get_current_cuda_stream()));
            prefill_ms += timer.elapsed_ms();
            co_yield PrefillProgress{{static_cast<std::size_t>(offset) - reused_tokens, prefill_ms, 0.0, prefill_ms}};
        }
        if (output.result.cancelled) co_return;
        if (transformed_prompt) model.decode_position_delta = prepared->decode_position_delta;
        raw = raw_chunks.size() == 1 ? std::move(raw_chunks.front()) : mfq_tensor_backend::cat(raw_chunks, 1).contiguous();
        raw_chunks.clear();
        auto committed_last_hidden = raw.narrow(
            1, raw.size(1) - 1, 1);
        auto finish = [&]() {
            output.metrics.mtp = mtp.last_stats;
            if (session_last_hidden != nullptr) {
                *session_last_hidden = committed_last_hidden;
            }
        };
        auto logits = logits_for(hidden.narrow(1, hidden.size(1) - 1, 1));
        auto initial_constraint = token_constraint
            ? token_constraint->clone()
            : MfqTokenConstraintPtr{};
        MFQ_RUNTIME_CHECK(
            !token_constraint ||
                initial_constraint,
            "CUDA MTP token constraint clone is incomplete");
        int32_t pending = sample_constrained(
            logits, counts, initial_constraint);

        MFQ_RUNTIME_CHECK(
            reused_tokens == 0 || !mtp.blockwise_drafting(),
            "blockwise CUDA MTP session restore is unsupported");
        if (mtp.blockwise_drafting()) {
            mtp.append_target_context(raw, 0);
        }

        if (mtp.teacher_forced_prompt_prime()) {
            const int64_t chunk_size = std::min<int64_t>(512, prefill_chunk_size);
            Tensor prime_hidden;
            int64_t prime_ids_offset = 1;
            int64_t pairs = raw.size(1) - 1;
            if (reused_tokens > 0) {
                prime_hidden = raw.size(1) == 1
                    ? restored_last_hidden
                    : mfq_tensor_backend::cat(
                          {restored_last_hidden,
                           raw.narrow(1, 0, raw.size(1) - 1)},
                          1).contiguous();
                prime_ids_offset = static_cast<int64_t>(reused_tokens);
                pairs = raw.size(1);
            } else if (pairs > 0) {
                prime_hidden = raw.narrow(1, 0, pairs);
            }
            for (int64_t offset = 0; offset < pairs;) {
                if (output.result.cancelled) co_return;
                const auto chunk = mfq::engine::next_prefill_chunk(
                    pairs, offset, chunk_size);
                (void)predictor_step(
                    prime_hidden.narrow(1, chunk.offset, chunk.count),
                    input_ids.narrow(
                        1, prime_ids_offset + chunk.offset, chunk.count),
                    transformed_prompt
                        ? prepared->positions.narrow(
                              -1, prime_ids_offset + chunk.offset,
                              chunk.count).contiguous()
                        : Tensor{});
                if (output.result.cancelled) co_return;
                offset += chunk.count;
                co_yield PrefillProgress{{prompt.size() - reused_tokens, prefill_ms, 0.0, prefill_ms}};
            }
        }

        if (output.result.cancelled) co_return;
        auto first = output.append(std::vector<int64_t>{pending});
        accept(first);
        finish();
        co_yield std::move(first);
        if (output.stopped()) co_return;
        auto constraint_cursor = token_constraint
            ? token_constraint->clone()
            : MfqTokenConstraintPtr{};
        MFQ_RUNTIME_CHECK(
            !token_constraint ||
                constraint_cursor,
            "CUDA MTP token constraint cursor is incomplete");

        auto initial_hidden = committed_last_hidden;
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
            committed_last_hidden = raw.narrow(
                1, raw.size(1) - 1, 1);
            pending = sample_constrained(
                logits_for(next_hidden), counts, constraint_cursor);
            if (constraint_cursor) constraint_cursor->accept(pending);
            auto delta = output.append(std::vector<int64_t>{pending});
            accept(delta);
            finish();
            co_yield std::move(delta);
            if (output.stopped()) co_return;
            initial_hidden = committed_last_hidden;
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
                    token = policy::sample_compact(
                        proposal, sampler.next_uniform());
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
                        requested_depth);
                    MFQ_RUNTIME_CHECK(
                        block.size(1) == requested_depth,
                        "CUDA block predictor returned an incomplete draft");
                    auto previous = ids_for({next_ids.back()});
                    for (int position = 0; position < requested_depth; ++position) {
                        const auto token = select_draft(mtp.draft_next(block.narrow(1, position, 1), previous));
                        previous = ids_for({token});
                    }
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
            for (int position = 0; position < requested_depth; ++position) {
                auto draft_logits = logits_for(sample_hidden).reshape({1, -1});
                const int32_t token = select_draft(draft_logits);
                if (position + 1 < requested_depth) {
                    auto next = predictor_step(
                        chain_hidden,
                        ids_for({token}),
                        transformed_prompt
                            ? decode_positions(
                                  predictor_history_position + position + 1,
                                  1)
                            : Tensor{});
                    sample_hidden = std::move(next.sample_hidden);
                    chain_hidden = std::move(next.chain_hidden);
                }
            }
            return result;
        };

        auto draft = prepare_draft(
            initial_hidden, {pending},
            bounded_depth(depth_controller.depth()), true);
        while (generated < limit) {
            if (output.stopped()) { finish(); co_return; }
            const auto cycle_started = Clock::now();
            const int draft_count = static_cast<int>(draft.tokens.size());
            Tensor verified_raw;
            std::vector<int64_t> verify_ids{pending};
            verify_ids.insert(
                verify_ids.end(), draft.tokens.begin(), draft.tokens.end());
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
                    ids_for(std::move(verify_ids)), mfq_nullopt, mfq_nullopt,
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
                for (int row = 0; row <= draft_count; ++row) {
                    target_tokens.push_back(sample_normal(
                        targets.narrow(0, row, 1), row_counts));
                    if (row < draft_count && penalties) {
                        sample_token_counts_add_cuda(
                            row_counts, ids_for({draft.tokens[static_cast<size_t>(row)]}));
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
                result = policy::verify_compact_chain(
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

            std::vector<int64_t> candidates(draft.tokens.begin(), draft.tokens.begin() + accepted);
            candidates.push_back(result.next_token);
            auto delta = output.append(candidates);
            accept(delta);
            const int emitted_accepted = std::min<int>(accepted, delta.token_ids.size());
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
            committed_last_hidden = verified_raw.narrow(
                1, emitted_accepted, 1);
            if (output.stopped()) {
                if (mtp.teacher_forced_prompt_prime() && !delta.token_ids.empty()) {
                    std::vector<int32_t> committed_ids(delta.token_ids.begin(), delta.token_ids.end());
                    (void)prepare_draft(verified_raw.narrow(1, 0, committed_ids.size()), committed_ids, 0, false);
                }
                finish();
                co_yield std::move(delta);
                co_return;
            }

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
            std::vector<int32_t> next_ids;
            next_ids.reserve(static_cast<size_t>(accepted + 1));
            next_ids.insert(
                next_ids.end(), draft.tokens.begin(),
                draft.tokens.begin() + accepted);
            next_ids.push_back(pending);
            draft = prepare_draft(
                verified_raw.narrow(1, 0, accepted + 1),
                next_ids,
                bounded_depth(depth_controller.depth()),
                false);
            finish();
            co_yield std::move(delta);
        }
        finish();
    }
}
#define MFQ_INSTANTIATE_MTP(MODEL) \
    template Generation run_mtp_generation<MODEL>(MODEL&, MtpModule&, \
        InferenceRequest&, InferenceOutput&, int64_t, const CudaPreparedPrompt*, \
        std::size_t, mfq_tensor_backend::Tensor, mfq_tensor_backend::Tensor*)

MFQ_INSTANTIATE_MTP(mfq::cuda::Qwen35CausalLm);
MFQ_INSTANTIATE_MTP(mfq::cuda::MiniCPMO45CausalLm);
MFQ_INSTANTIATE_MTP(mfq::cuda::MiniCPMOTtsCausalLm);
MFQ_INSTANTIATE_MTP(mfq::cuda::Gemma4CausalLm);
MFQ_INSTANTIATE_MTP(mfq::cuda::GlmDsaCausalLm);
MFQ_INSTANTIATE_MTP(mfq::cuda::Glm5CausalLm);
MFQ_INSTANTIATE_MTP(mfq::cuda::Qwen4CausalLm);
MFQ_INSTANTIATE_MTP(mfq::cuda::DeepseekV4CausalLm);
MFQ_INSTANTIATE_MTP(mfq::cuda::DeepseekV41CausalLm);

#undef MFQ_INSTANTIATE_MTP
