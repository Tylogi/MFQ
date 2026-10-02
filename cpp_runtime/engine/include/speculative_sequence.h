#pragma once

#include "generation_step.h"
#include "execution_check.h"
#include "mtp_policy.h"

#include <algorithm>
#include <chrono>
#include <vector>

namespace mfq::engine {

// The complete speculative transaction lives here. Ops submits native target /
// predictor work and numerical sampling; acceptance and commit order are shared.
template <class Ops>
Generation speculative_sequence(Ops ops, InferenceRequest &request, InferenceOutput &output,
    int64_t prefill_chunk_size, size_t reused_tokens, typename Ops::Tensor restored_last_hidden,
    typename Ops::Tensor *session_last_hidden) {
    const auto &prompt = request.prompt;
    const auto &sampling = request.sampling;
    const auto &token_constraint = request.token_constraint;
    using Tensor = typename Ops::Tensor;
    auto &model = ops.model;
    auto &mtp = ops.mtp;
    const auto *prepared = ops.prepared;
    using Clock = std::chrono::steady_clock;
    namespace policy = mfq::engine::mtp;
    using policy::CompactDistribution;
    if (session_last_hidden != nullptr) {
        *session_last_hidden = Tensor();
    }
    if (output.result.cancelled)
        co_return;
    require_execution(
        reused_tokens < prompt.size(), "restored MTP prefix must be shorter than the prompt");
    const bool transformed_prompt = prepared != nullptr && prepared->transformed();
    require_execution(prepared == nullptr || prepared->token_ids == prompt,
        "prepared MTP prompt disagrees with rendered token IDs");
    require_execution(!transformed_prompt ||
                          (ops.defined(prepared->embeddings) && ops.defined(prepared->positions)),
        "prepared MTP prompt is missing embeddings or positions");
    require_execution(
        reused_tokens == 0 || (mtp.supports_session_state() && ops.defined(restored_last_hidden) &&
                                  model.cache_pos == static_cast<int64_t>(reused_tokens) &&
                                  mtp.cache_position() == static_cast<int64_t>(reused_tokens) - 1),
        "restored MTP session boundary is incompatible");
    mtp.last_stats = {};
    mtp.last_stats.available = true;
    mtp.last_stats.used = true;
    mtp.last_cycles = mtp.last_accepted = mtp.last_rejected = 0;
    const int64_t next_logical_position =
        static_cast<int64_t>(prompt.size()) +
        (transformed_prompt ? prepared->decode_position_delta : 0);
    require_execution(
        next_logical_position >= 0 && next_logical_position <= model.max_position_embeddings(),
        "prepared MTP decode position is outside context capacity");
    const int64_t occupied_context =
        std::max<int64_t>(static_cast<int64_t>(prompt.size()), next_logical_position);
    const auto generation_plan = mfq::engine::plan_generation(prompt,
        model.vocab_size(),
        model.max_position_embeddings(),
        sampling.max_tokens,
        0,
        occupied_context);
    const int32_t limit = generation_plan.generation_tokens;
    if (limit <= 0)
        co_return;
    ops.initialize(prompt, sampling);
    auto &input_ids = ops.input_ids;
    auto &counts = ops.counts;
    auto &sampler = ops.sampler();
    const bool penalties = ops.penalties;
    const bool greedy = sampler.greedy();
    int32_t generated = 0;
    auto accept = [&](const TokenOutput &delta) {
        for (const auto token : delta.token_ids) {
            require_execution(
                token >= 0 && token < model.vocab_size(), "MTP sampled token outside vocabulary");
            if (token_constraint)
                token_constraint->accept(token);
            ++generated;
            if (penalties)
                ops.add_counts(counts, ops.ids_for({token}));
        }
    };
    struct DraftChain {
        std::vector<int32_t> tokens;
        std::vector<std::vector<float>> probabilities;
        std::vector<CompactDistribution> compact_probabilities;
    };
    const bool compact_stochastic = !greedy && sampling.top_k > 0 && sampling.top_k <= 64;
    auto draft_sampling = sampling;
    if (compact_stochastic) {
        draft_sampling.temperature = 0.6;
        draft_sampling.top_p = 0.95;
    }
    const int maximum_depth = std::min(mtp.maximum_draft_depth(),
        (!greedy && !compact_stochastic)
            ? 1
            : std::clamp<int>(sampling.mtp_max_draft_tokens, 1, policy::kMaximumDraftDepth));
    policy::DepthController depth_controller(maximum_depth);

    {
        if (reused_tokens == 0) {
            model.reset(1);
            mtp.reset(1);
        }
        Tensor raw, hidden;
        std::vector<Tensor> raw_chunks;
        double prefill_ms = 0.0;
        for (int64_t offset = static_cast<int64_t>(reused_tokens);
            offset < ops.size(input_ids, 1);) {
            if (output.result.cancelled)
                co_return;
            const auto chunk =
                next_prefill_chunk(ops.size(input_ids, 1), offset, prefill_chunk_size);
            auto step = ops.prefill(chunk);
            hidden = std::move(step.hidden);
            raw_chunks.push_back(std::move(step.raw));
            offset += chunk.count;
            prefill_ms += step.elapsed;
            co_yield PrefillProgress{
                {static_cast<std::size_t>(offset) - reused_tokens, prefill_ms, 0.0, prefill_ms}};
        }
        if (output.result.cancelled)
            co_return;
        if (transformed_prompt)
            model.decode_position_delta = prepared->decode_position_delta;
        raw =
            raw_chunks.size() == 1 ? std::move(raw_chunks.front()) : ops.concatenate(raw_chunks, 1);
        raw_chunks.clear();
        auto committed_last_hidden = ops.slice(raw, 1, ops.size(raw, 1) - 1, 1);
        auto finish = [&]() {
            output.metrics.mtp = mtp.last_stats;
            if (session_last_hidden != nullptr) {
                *session_last_hidden = committed_last_hidden;
            }
        };
        auto logits = ops.logits_for(ops.slice(hidden, 1, ops.size(hidden, 1) - 1, 1));
        auto initial_constraint =
            token_constraint ? token_constraint->clone() : MfqTokenConstraintPtr{};
        require_execution(
            !token_constraint || initial_constraint, "MTP token constraint clone is incomplete");
        int32_t pending = ops.sample_constrained(logits, counts, initial_constraint);

        require_execution(reused_tokens == 0 || !mtp.blockwise_drafting(),
            "blockwise MTP session restore is unsupported");
        if (mtp.blockwise_drafting()) {
            mtp.append_target_context(raw, 0);
        }

        if (mtp.teacher_forced_prompt_prime()) {
            const int64_t chunk_size = std::min<int64_t>(512, prefill_chunk_size);
            Tensor prime_hidden;
            int64_t prime_ids_offset = 1;
            int64_t pairs = ops.size(raw, 1) - 1;
            if (reused_tokens > 0) {
                prime_hidden =
                    ops.size(raw, 1) == 1
                        ? restored_last_hidden
                        : ops.concatenate(
                              {restored_last_hidden, ops.slice(raw, 1, 0, ops.size(raw, 1) - 1)},
                              1);
                prime_ids_offset = static_cast<int64_t>(reused_tokens);
                pairs = ops.size(raw, 1);
            } else if (pairs > 0) {
                prime_hidden = ops.slice(raw, 1, 0, pairs);
            }
            for (int64_t offset = 0; offset < pairs;) {
                if (output.result.cancelled)
                    co_return;
                const auto chunk = mfq::engine::next_prefill_chunk(pairs, offset, chunk_size);
                (void)ops.predictor_step(ops.slice(prime_hidden, 1, chunk.offset, chunk.count),
                    ops.slice(input_ids, 1, prime_ids_offset + chunk.offset, chunk.count),
                    transformed_prompt
                        ? ops.slice(
                              prepared->positions, -1, prime_ids_offset + chunk.offset, chunk.count)
                        : Tensor{});
                if (output.result.cancelled)
                    co_return;
                offset += chunk.count;
                co_yield PrefillProgress{
                    {prompt.size() - reused_tokens, prefill_ms, 0.0, prefill_ms}};
            }
        }

        if (output.result.cancelled)
            co_return;
        auto first = output.append(std::vector<int64_t>{pending});
        accept(first);
        finish();
        co_yield std::move(first);
        if (output.stopped())
            co_return;
        auto constraint_cursor =
            token_constraint ? token_constraint->clone() : MfqTokenConstraintPtr{};
        require_execution(
            !token_constraint || constraint_cursor, "MTP token constraint cursor is incomplete");

        auto initial_hidden = committed_last_hidden;
        if (mtp.target_bootstrap_decode()) {
            if (mtp.teacher_forced_prompt_prime()) {
                // Some recurrent predictors consume the prompt/first-token
                // seam before the target advances to hidden(first_token).
                (void)ops.predictor_step(initial_hidden,
                    ops.ids_for({pending}),
                    transformed_prompt ? ops.decode_positions(model.cache_pos, 1) : Tensor{});
            }
            auto next_hidden = ops.target_forward(ops.ids_for({pending}), &raw);
            committed_last_hidden = ops.slice(raw, 1, ops.size(raw, 1) - 1, 1);
            pending =
                ops.sample_constrained(ops.logits_for(next_hidden), counts, constraint_cursor);
            if (constraint_cursor)
                constraint_cursor->accept(pending);
            auto delta = output.append(std::vector<int64_t>{pending});
            accept(delta);
            finish();
            co_yield std::move(delta);
            if (output.stopped())
                co_return;
            initial_hidden = committed_last_hidden;
        }

        int64_t predictor_history_position = mtp.cache_position();
        auto bounded_depth = [&](int desired) {
            const auto context_depth =
                std::max<int64_t>(0, model.max_position_embeddings() - model.cache_pos - 1);
            const auto output_depth =
                std::max<int64_t>(0, static_cast<int64_t>(limit - generated - 1));
            return static_cast<int>(
                std::min<int64_t>(desired, std::min(context_depth, output_depth)));
        };
        auto prepare_draft = [&](Tensor hidden_rows,
                                 const std::vector<int32_t> &next_ids,
                                 int requested_depth,
                                 bool initial) {
            require_execution(
                ops.rank(hidden_rows) == 3 && ops.size(hidden_rows, 0) == 1 &&
                    ops.size(hidden_rows, 1) == static_cast<int64_t>(next_ids.size()) &&
                    !next_ids.empty() && requested_depth >= 0 && requested_depth <= maximum_depth,
                "MTP committed history is incompatible");
            mtp.trim_cache_to(predictor_history_position);
            auto prospective_counts = penalties ? ops.clone(counts) : Tensor{};
            DraftChain result;
            result.tokens.reserve(static_cast<size_t>(requested_depth));
            result.probabilities.reserve(static_cast<size_t>(requested_depth));
            result.compact_probabilities.reserve(static_cast<size_t>(requested_depth));
            auto select_draft = [&](Tensor draft_logits) {
                draft_logits = ops.reshape(draft_logits, {1, -1});
                int32_t token = -1;
                if (greedy) {
                    token = ops.sample_normal(draft_logits, prospective_counts);
                } else if (compact_stochastic) {
                    auto proposal =
                        ops.compact_probabilities(draft_logits, prospective_counts, draft_sampling);
                    token = policy::sample_compact(proposal, sampler.next_uniform());
                    result.compact_probabilities.push_back(std::move(proposal));
                } else {
                    auto proposal =
                        ops.probabilities(draft_logits, prospective_counts, draft_sampling);
                    token = policy::sample(proposal, sampler.next_uniform());
                    result.probabilities.push_back(std::move(proposal));
                }
                result.tokens.push_back(token);
                if (penalties) {
                    ops.add_counts(prospective_counts, ops.ids_for({token}));
                }
                return token;
            };
            if (mtp.blockwise_drafting()) {
                if (!initial) {
                    mtp.append_target_context(hidden_rows, predictor_history_position);
                    predictor_history_position += static_cast<int64_t>(next_ids.size());
                }
                require_execution(mtp.cache_position() == predictor_history_position,
                    "block predictor cache did not advance");
                if (requested_depth > 0) {
                    auto block = mtp.draft_block(
                        ops.target, ops.ids_for({next_ids.back()}), requested_depth);
                    require_execution(ops.size(block, 1) == requested_depth,
                        "block predictor returned an incomplete draft");
                    auto previous = ops.ids_for({next_ids.back()});
                    for (int position = 0; position < requested_depth; ++position) {
                        const auto token = select_draft(
                            mtp.draft_next(ops.slice(block, 1, position, 1), previous));
                        previous = ops.ids_for({token});
                    }
                }
                return result;
            }
            std::vector<int64_t> shifted(next_ids.begin(), next_ids.end());
            auto head = ops.predictor_step(std::move(hidden_rows),
                ops.ids_for(std::move(shifted)),
                transformed_prompt ? ops.decode_positions(predictor_history_position + 1,
                                         static_cast<int64_t>(next_ids.size()))
                                   : Tensor{});
            predictor_history_position += static_cast<int64_t>(next_ids.size());
            require_execution(mtp.cache_position() == predictor_history_position,
                "MTP predictor cache did not advance");
            auto sample_hidden =
                ops.slice(head.sample_hidden, 1, ops.size(head.sample_hidden, 1) - 1, 1);
            auto chain_hidden =
                ops.slice(head.chain_hidden, 1, ops.size(head.chain_hidden, 1) - 1, 1);
            for (int position = 0; position < requested_depth; ++position) {
                auto draft_logits = ops.reshape(ops.logits_for(sample_hidden), {1, -1});
                const int32_t token = select_draft(draft_logits);
                if (position + 1 < requested_depth) {
                    auto next = ops.predictor_step(chain_hidden,
                        ops.ids_for({token}),
                        transformed_prompt
                            ? ops.decode_positions(predictor_history_position + position + 1, 1)
                            : Tensor{});
                    sample_hidden = std::move(next.sample_hidden);
                    chain_hidden = std::move(next.chain_hidden);
                }
            }
            return result;
        };

        auto draft =
            prepare_draft(initial_hidden, {pending}, bounded_depth(depth_controller.depth()), true);
        while (generated < limit) {
            if (output.stopped()) {
                finish();
                co_return;
            }
            const auto cycle_started = Clock::now();
            const int draft_count = static_cast<int>(draft.tokens.size());
            Tensor verified_raw;
            std::vector<int64_t> verify_ids{pending};
            verify_ids.insert(verify_ids.end(), draft.tokens.begin(), draft.tokens.end());
            Tensor verified;
            if (mtp.split_target_verification()) {
                Tensor pending_raw;
                auto pending_hidden = ops.target_forward(ops.ids_for({pending}), &pending_raw);
                if (draft_count > 0) {
                    model.begin_speculative_suffix(draft_count);
                    std::vector<int64_t> draft_ids(draft.tokens.begin(), draft.tokens.end());
                    Tensor draft_raw;
                    auto draft_hidden =
                        ops.target_suffix(ops.ids_for(std::move(draft_ids)), &draft_raw);
                    verified = ops.concatenate({pending_hidden, draft_hidden}, 1);
                    verified_raw = ops.concatenate({pending_raw, draft_raw}, 1);
                } else {
                    verified = std::move(pending_hidden);
                    verified_raw = std::move(pending_raw);
                }
            } else {
                verified = ops.target_forward(
                    ops.ids_for(std::move(verify_ids)), &verified_raw, draft_count > 0 ? 1 : 0);
            }
            auto targets =
                ops.reshape(ops.logits_for(verified), {draft_count + 1, model.vocab_size()});

            ++mtp.last_stats.cycles;
            mtp.last_stats.drafted_tokens += static_cast<uint64_t>(draft_count);
            ++mtp.last_stats.depth_cycles.at(static_cast<size_t>(draft_count));
            for (int position = 0; position < draft_count; ++position) {
                ++mtp.last_stats.position_drafted.at(static_cast<size_t>(position));
            }

            auto row_counts = penalties ? ops.clone(counts) : Tensor{};
            policy::ChainVerification result;
            if (draft_count == 0) {
                // No proposal distribution is needed: use the ordinary GPU
                // sampler rather than transferring a vocabulary row.
                result = {0, ops.sample_normal(ops.slice(targets, 0, 0, 1), row_counts), true};
            } else if (greedy) {
                std::vector<int32_t> target_tokens;
                target_tokens.reserve(static_cast<size_t>(draft_count + 1));
                for (int row = 0; row <= draft_count; ++row) {
                    target_tokens.push_back(
                        ops.sample_normal(ops.slice(targets, 0, row, 1), row_counts));
                    if (row < draft_count && penalties) {
                        ops.add_counts(
                            row_counts, ops.ids_for({draft.tokens[static_cast<size_t>(row)]}));
                    }
                }
                result = policy::verify_greedy(draft.tokens, target_tokens);
            } else if (compact_stochastic) {
                std::vector<CompactDistribution> target_probabilities;
                if (!penalties) {
                    // One device top-k and one compact host transfer for the
                    // complete verification window.
                    target_probabilities = ops.compact_probability_rows(targets, sampling);
                } else {
                    target_probabilities.reserve(static_cast<size_t>(draft_count + 1));
                    for (int row = 0; row <= draft_count; ++row) {
                        target_probabilities.push_back(ops.compact_probabilities(
                            ops.slice(targets, 0, row, 1), row_counts, sampling));
                        if (row < draft_count) {
                            ops.add_counts(
                                row_counts, ops.ids_for({draft.tokens[static_cast<size_t>(row)]}));
                        }
                    }
                }
                std::vector<double> acceptance_uniforms(static_cast<size_t>(draft_count));
                std::generate(acceptance_uniforms.begin(), acceptance_uniforms.end(), [&] {
                    return sampler.next_uniform();
                });
                result = policy::verify_compact_chain(draft.tokens,
                    draft.compact_probabilities,
                    target_probabilities,
                    acceptance_uniforms,
                    sampler.next_uniform());
            } else {
                std::vector<std::vector<float>> target_probabilities;
                target_probabilities.reserve(static_cast<size_t>(draft_count + 1));
                for (int row = 0; row <= draft_count; ++row) {
                    target_probabilities.push_back(
                        ops.probabilities(ops.slice(targets, 0, row, 1), row_counts, sampling));
                    if (row < draft_count && penalties) {
                        ops.add_counts(
                            row_counts, ops.ids_for({draft.tokens[static_cast<size_t>(row)]}));
                    }
                }
                std::vector<double> acceptance_uniforms(static_cast<size_t>(draft_count));
                std::generate(acceptance_uniforms.begin(), acceptance_uniforms.end(), [&] {
                    return sampler.next_uniform();
                });
                result = policy::verify_stochastic_chain(draft.tokens,
                    draft.probabilities,
                    target_probabilities,
                    acceptance_uniforms,
                    sampler.next_uniform());
            }

            int accepted = static_cast<int>(result.accepted_drafts);
            require_execution(accepted >= 0 && accepted <= draft_count && result.next_token >= 0 &&
                                  result.next_token < model.vocab_size(),
                "MTP verification returned invalid data");
            if (constraint_cursor) {
                auto constraint_counts = penalties ? ops.clone(counts) : Tensor{};
                bool corrected = false;
                for (int position = 0; position < accepted; ++position) {
                    const int32_t token = draft.tokens[static_cast<size_t>(position)];
                    if (!constraint_cursor->allows(token)) {
                        accepted = position;
                        result.next_token =
                            ops.sample_constrained(ops.slice(targets, 0, position, 1),
                                constraint_counts,
                                constraint_cursor);
                        result.bonus = false;
                        corrected = true;
                        break;
                    }
                    constraint_cursor->accept(token);
                    if (penalties) {
                        ops.add_counts(constraint_counts, ops.ids_for({token}));
                    }
                }
                if (!corrected) {
                    if (!constraint_cursor->allows(result.next_token)) {
                        result.next_token =
                            ops.sample_constrained(ops.slice(targets, 0, accepted, 1),
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
                    const bool retained = mtp.retains_partial_target_prefix();
                    model.rollback_speculative(retained ? emitted_accepted : 0);
                    if (!retained && emitted_accepted > 0) {
                        std::vector<int64_t> replay(
                            draft.tokens.begin(), draft.tokens.begin() + emitted_accepted);
                        (void)ops.target_forward(ops.ids_for(std::move(replay)));
                    }
                }
            }
            committed_last_hidden = ops.slice(verified_raw, 1, emitted_accepted, 1);
            if (output.stopped()) {
                if (mtp.teacher_forced_prompt_prime() && !delta.token_ids.empty()) {
                    std::vector<int32_t> committed_ids(
                        delta.token_ids.begin(), delta.token_ids.end());
                    (void)prepare_draft(ops.slice(verified_raw, 1, 0, committed_ids.size()),
                        committed_ids,
                        0,
                        false);
                }
                finish();
                co_yield std::move(delta);
                co_return;
            }

            const double cycle_ms =
                std::chrono::duration<double, std::milli>(Clock::now() - cycle_started).count();
            depth_controller.observe(draft_count, accepted, cycle_ms);
            mtp.last_stats.selected_depth = depth_controller.depth();
            for (int depth = 0; depth <= depth_controller.maximum_depth(); ++depth) {
                if (const auto measured = depth_controller.measured_cycle_ms(depth)) {
                    mtp.last_stats.measured_depth_ms.at(static_cast<size_t>(depth)) = *measured;
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
            next_ids.insert(next_ids.end(), draft.tokens.begin(), draft.tokens.begin() + accepted);
            next_ids.push_back(pending);
            draft = prepare_draft(ops.slice(verified_raw, 1, 0, accepted + 1),
                next_ids,
                bounded_depth(depth_controller.depth()),
                false);
            finish();
            co_yield std::move(delta);
        }
        finish();
    }
}

} // namespace mfq::engine
