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
    typename Ops::Tensor *session_last_hidden, size_t stable_prefix_tokens = 0) {
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
    const bool positioned_prompt = transformed_prompt && ops.defined(prepared->positions);
    require_execution(prepared == nullptr || prepared->token_ids == prompt,
        "prepared MTP prompt disagrees with rendered token IDs");
    require_execution(!transformed_prompt ||
                          (ops.defined(prepared->embeddings) &&
                           (positioned_prompt || prepared->decode_position_delta==0)),
        "prepared MTP prompt is missing embeddings or valid position metadata");
    require_execution(
        reused_tokens == 0 || (mtp.supports_session_state() && ops.defined(restored_last_hidden) &&
                                  model.cache_pos == static_cast<int64_t>(reused_tokens) &&
                                  mtp.cache_position() == static_cast<int64_t>(reused_tokens) -
                                      (mtp.blockwise_drafting()?0:1)),
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
    if (compact_stochastic && mtp.dspark_policy()) {
        draft_sampling.temperature = 0.6;
        draft_sampling.top_p = 0.95;
    }
    const int maximum_depth = std::min(mtp.maximum_draft_depth(),
        (!greedy && !compact_stochastic)
            ? 1
            : std::clamp<int>(sampling.mtp_max_draft_tokens, 1, policy::kMaximumDraftDepth));
    policy::GenerationPolicy<policy::DsparkDepthController> depth_controller(
        maximum_depth, mtp.dspark_policy(), &mtp.policy_state,[&] {
            if constexpr(requires {ops.gradual_mtp_warmup();})return ops.gradual_mtp_warmup();
            return false;
        }(),[&] {
            if constexpr(requires {ops.metal_mtp_policy();})return ops.metal_mtp_policy();
            return false;
        }());
    mtp.last_stats.park_count=depth_controller.parks();

    {
        Tensor raw, hidden;
        std::vector<Tensor> raw_chunks;
        double prefill_ms = 0.0;
        double target_prefill_ms=0.0, predictor_prefill_ms=0.0, predictor_workspace_ms=0.0;
        const auto checkpoint=mtp.blockwise_drafting() && session_last_hidden &&
            stable_prefix_tokens>reused_tokens && stable_prefix_tokens<prompt.size()
            ?static_cast<int64_t>(stable_prefix_tokens):int64_t(0);
        bool prompt_context_seeded=false;
        const auto prime_blockwise=[&](Tensor context,int64_t position,bool seed) {
            if constexpr(requires { ops.blockwise_prompt_context(context,position,seed); }) {
                const auto elapsed=ops.blockwise_prompt_context(context,position,seed);
                prefill_ms+=elapsed;predictor_prefill_ms+=elapsed;
            } else if(seed) {
                if constexpr(requires { mtp.seed_target_context(context,position); })
                    mtp.seed_target_context(context,position);
                else mtp.append_target_context(context,0);
            } else mtp.append_target_context(context,position-ops.size(context,1));
        };
        const bool stream_prompt_history = mtp.teacher_forced_prompt_prime() && !mtp.blockwise_drafting() && [&] {
            if constexpr (requires { ops.stream_prompt_history(ops.size(input_ids, 1)); })
                return ops.stream_prompt_history(ops.size(input_ids, 1));
            return false;
        }();
        const int64_t predictor_chunk = [&] {
            if constexpr (requires { ops.predictor_prefill_chunk_size(prefill_chunk_size); })
                return ops.predictor_prefill_chunk_size(prefill_chunk_size);
            return std::min<int64_t>(512, prefill_chunk_size);
        }();
        auto prime_rows = [&](Tensor prime_hidden, int64_t ids_offset, int64_t pairs) {
            if (pairs <= 0) return;
            if constexpr (requires { ops.begin_predictor_prefill(pairs); }) {
                const auto elapsed=ops.begin_predictor_prefill(pairs);
                prefill_ms+=elapsed;predictor_workspace_ms+=elapsed;
            }
            for (int64_t offset = 0; offset < pairs;) {
                if (output.result.cancelled) return;
                const auto chunk = next_prefill_chunk(pairs, offset, predictor_chunk);
                auto prime = ops.slice(prime_hidden, 1, chunk.offset, chunk.count);
                auto ids = ops.slice(input_ids, 1, ids_offset + chunk.offset, chunk.count);
                auto positions = positioned_prompt
                    ? ops.slice(prepared->positions, -1, ids_offset + chunk.offset, chunk.count) : Tensor{};
                if constexpr (requires { ops.predictor_prefill(prime, ids, positions); }) {
                    const auto elapsed=ops.predictor_prefill(std::move(prime), std::move(ids), std::move(positions));
                    prefill_ms+=elapsed;predictor_prefill_ms+=elapsed;
                }
                else (void)ops.predictor_step(std::move(prime), std::move(ids), std::move(positions));
                offset += chunk.count;
            }
            if constexpr (requires { ops.end_predictor_prefill(); }) {
                const auto elapsed=ops.end_predictor_prefill();
                prefill_ms+=elapsed;predictor_workspace_ms+=elapsed;
            }
        };
        for (int64_t offset = static_cast<int64_t>(reused_tokens);
            offset < ops.size(input_ids, 1);) {
            if (output.result.cancelled)
                co_return;
            auto chunk =
                next_prefill_chunk(ops.size(input_ids, 1), offset, prefill_chunk_size);
            if(checkpoint>offset)chunk.count=std::min(chunk.count,checkpoint-offset);
            chunk.last=chunk.offset+chunk.count==ops.size(input_ids,1);
            auto step = [&] {
                PrefillActivity activity;
                return ops.prefill(chunk);
            }();
            hidden = std::move(step.hidden);
            if (stream_prompt_history) {
                if (offset == static_cast<int64_t>(reused_tokens) && reused_tokens > 0)
                    prime_rows(restored_last_hidden, offset, 1);
                // Each target row pairs with the next input ID, including
                // the boundary between target chunks. Retain just the final
                // row; a long prompt's multi-stream states can occupy GiBs.
                const auto pairs = std::min<int64_t>(chunk.count, ops.size(input_ids, 1) - offset - 1);
                prime_rows(ops.slice(step.raw, 1, 0, pairs), offset + 1, pairs);
                raw = ops.clone(ops.slice(step.raw, 1, ops.size(step.raw, 1) - 1, 1));
            } else raw_chunks.push_back(std::move(step.raw));
            offset += chunk.count;
            prefill_ms += step.elapsed;
            target_prefill_ms += step.elapsed;
            if(offset==checkpoint) {
                // Keep an exact checkpoint before the final prompt token so
                // an identical retry can reuse the prefix and recompute logits.
                auto context=raw_chunks.size()==1?std::move(raw_chunks.front()):ops.concatenate(raw_chunks,1);
                *session_last_hidden=ops.clone(ops.slice(context,1,ops.size(context,1)-1,1));
                prime_blockwise(context,offset,reused_tokens==0);
                prompt_context_seeded=true;raw_chunks.clear();
                co_yield PrefillProgress{
                    {static_cast<std::size_t>(offset)-reused_tokens,prefill_ms,0.0,prefill_ms},true};
                if(output.stopped())co_return;
                continue;
            }
            co_yield PrefillProgress{
                {static_cast<std::size_t>(offset) - reused_tokens, prefill_ms, 0.0, prefill_ms}};
        }
        if (output.result.cancelled)
            co_return;
        if (transformed_prompt)
            model.decode_position_delta = prepared->decode_position_delta;
        if (!stream_prompt_history)
            raw = raw_chunks.size() == 1 ? std::move(raw_chunks.front()) : ops.concatenate(raw_chunks, 1);
        raw_chunks.clear();
        auto committed_last_hidden = ops.slice(raw, 1, ops.size(raw, 1) - 1, 1);
        std::function<void()> flush_history;
        auto finish = [&]() {
            if (output.stopped() && flush_history) flush_history();
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

        if (mtp.blockwise_drafting()) {
            prime_blockwise(raw,ops.size(input_ids,1),!reused_tokens && !prompt_context_seeded);
        }

        if (mtp.teacher_forced_prompt_prime() && !stream_prompt_history) {
            const int64_t chunk_size = [&] {
                if constexpr (requires { ops.predictor_prefill_chunk_size(prefill_chunk_size); })
                    return ops.predictor_prefill_chunk_size(prefill_chunk_size);
                return std::min<int64_t>(512, prefill_chunk_size);
            }();
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
            if constexpr (requires { ops.begin_predictor_prefill(pairs); }) {
                const auto elapsed=ops.begin_predictor_prefill(pairs);
                prefill_ms+=elapsed;predictor_workspace_ms+=elapsed;
            }
            for (int64_t offset = 0; offset < pairs;) {
                if (output.result.cancelled)
                    co_return;
                const auto chunk = mfq::engine::next_prefill_chunk(pairs, offset, chunk_size);
                auto prime = ops.slice(prime_hidden, 1, chunk.offset, chunk.count);
                auto ids = ops.slice(input_ids, 1, prime_ids_offset + chunk.offset, chunk.count);
                auto positions = positioned_prompt
                    ? ops.slice(prepared->positions, -1, prime_ids_offset + chunk.offset, chunk.count)
                    : Tensor{};
                if constexpr (requires { ops.predictor_prefill(prime, ids, positions); }) {
                    const auto elapsed=ops.predictor_prefill(std::move(prime), std::move(ids), std::move(positions));
                    prefill_ms+=elapsed;predictor_prefill_ms+=elapsed;
                }
                else
                    (void)ops.predictor_step(std::move(prime), std::move(ids), std::move(positions));
                if (output.result.cancelled)
                    co_return;
                offset += chunk.count;
                if constexpr (requires { ops.end_predictor_prefill(); })
                    if(offset == pairs) {
                        const auto elapsed=ops.end_predictor_prefill();
                        prefill_ms+=elapsed;predictor_workspace_ms+=elapsed;
                    }
                co_yield PrefillProgress{
                    {prompt.size() - reused_tokens, prefill_ms, 0.0, prefill_ms}};
            }
        }

        if constexpr (requires { ops.trace_prompt_prefill(target_prefill_ms,predictor_prefill_ms,predictor_workspace_ms); })
            ops.trace_prompt_prefill(target_prefill_ms,predictor_prefill_ms,predictor_workspace_ms);
        if (output.result.cancelled)
            co_return;
        // Only one boundary row is needed after prompt priming. A contiguous
        // slice can still own the entire prompt allocation; detach it before
        // the first decode replay and release the large prefill tensors.
        committed_last_hidden = ops.clone(committed_last_hidden);
        raw = Tensor{};
        hidden = Tensor{};
        logits = Tensor{};
        if(session_last_hidden != nullptr) {
            *session_last_hidden = committed_last_hidden;
            // Initial prompt caching precedes token-one publication. Charge it
            // to TTFT rather than the interval between generated tokens.
            co_yield PrefillProgress{
                {prompt.size()-reused_tokens,prefill_ms,0.0,prefill_ms},true};
            if(output.stopped())co_return;
        }
        auto initial_hidden = committed_last_hidden;
        auto bootstrap = [&](Tensor token_counts,const MfqTokenConstraintPtr& cursor) {
            if(mtp.teacher_forced_prompt_prime()) {
                (void)ops.predictor_step(initial_hidden,ops.ids_for({pending}),
                    positioned_prompt ? ops.decode_positions(model.cache_pos,1) : Tensor{});
            }
            auto next_hidden=ops.target_forward(ops.ids_for({pending}),&raw);
            committed_last_hidden=ops.slice(raw,1,ops.size(raw,1)-1,1);
            pending=ops.sample_constrained(ops.logits_for(next_hidden),token_counts,cursor);
            if(cursor)cursor->accept(pending);
            initial_hidden=committed_last_hidden;
        };
        auto prefetch_committed_token=[&](int32_t token) {
            if(output.stopped())return;
            if constexpr(requires(Ops& value) {value.prefetch_committed_token(int32_t{});})
                ops.prefetch_committed_token(token);
        };
        std::vector<int64_t> first_tokens{pending};
        bool bootstrapped=false;
        if constexpr(requires { ops.bootstrap_before_publication(); }) {
            if(ops.bootstrap_before_publication() && mtp.target_bootstrap_decode() && limit>1) {
                // Inspect EOS, stop strings and parser boundaries on a copy.
                // The initial prompt snapshot above still belongs to the prompt
                // alone, and no output timestamp is recorded until publication.
                auto preview=output;
                (void)preview.append(first_tokens);
                if(!preview.stopped()) {
                    auto prospective_counts=penalties?ops.clone(counts):Tensor{};
                    if(penalties)ops.add_counts(prospective_counts,ops.ids_for(first_tokens));
                    auto cursor=token_constraint?token_constraint->clone():MfqTokenConstraintPtr{};
                    require_execution(!token_constraint || cursor,"MTP bootstrap constraint cursor is incomplete");
                    if(cursor)cursor->accept(pending);
                    bootstrap(std::move(prospective_counts),cursor);
                    first_tokens.push_back(pending);bootstrapped=true;
                }
            }
        }
        if(output.result.cancelled)co_return;
        auto first = output.append(first_tokens);
        accept(first);
        prefetch_committed_token(pending);
        finish();
        co_yield std::move(first);
        if (output.stopped())
            co_return;
        auto constraint_cursor =
            token_constraint ? token_constraint->clone() : MfqTokenConstraintPtr{};
        require_execution(
            !token_constraint || constraint_cursor, "MTP token constraint cursor is incomplete");

        if (mtp.target_bootstrap_decode() && !bootstrapped) {
            bootstrap(counts,constraint_cursor);
            auto delta = output.append(std::vector<int64_t>{pending});
            accept(delta);
            prefetch_committed_token(pending);
            finish();
            co_yield std::move(delta);
            if (output.stopped())
                co_return;
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
        std::vector<Tensor> pending_hidden;
        std::vector<int64_t> pending_ids;
        bool maintenance = false;
        flush_history = [&] {
            if (pending_ids.empty()) return;
            auto rows = pending_hidden.size() == 1 ? pending_hidden.front()
                : ops.concatenate(pending_hidden, 1);
            auto ids=ops.ids_for(pending_ids);
            auto positions=positioned_prompt ? ops.decode_positions(predictor_history_position + 1,
                static_cast<int64_t>(pending_ids.size())) : Tensor{};
            if constexpr(requires {ops.predictor_history_step(rows,ids,positions);})
                ops.predictor_history_step(std::move(rows),std::move(ids),std::move(positions));
            else (void)ops.predictor_step(std::move(rows),std::move(ids),std::move(positions));
            predictor_history_position += static_cast<int64_t>(pending_ids.size());
            pending_hidden.clear();
            pending_ids.clear();
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
            if constexpr(requires { ops.prefetch_target_first(next_ids.back()); }) {
                if(requested_depth>0)ops.prefetch_target_first(next_ids.back());
            }
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
            if (!depth_controller.dspark() && requested_depth == 0) {
                pending_hidden.push_back(hidden_rows);
                pending_ids.insert(pending_ids.end(), next_ids.begin(), next_ids.end());
                std::size_t threshold=512;
                if constexpr(requires { ops.predictor_history_flush_threshold(); })
                    threshold=ops.predictor_history_flush_threshold();
                if constexpr(requires { ops.predictor_parked_history_flush_threshold(); })
                    if(depth_controller.parked())threshold=ops.predictor_parked_history_flush_threshold();
                if (pending_ids.size() >= threshold) { maintenance = true; flush_history(); }
                return result;
            }
            std::vector<int64_t> shifted(next_ids.begin(), next_ids.end());
            if (!pending_ids.empty()) {
                maintenance = true;
                std::size_t merge_limit=0;
                if constexpr(requires { ops.predictor_history_merge_limit(); })
                    merge_limit=ops.predictor_history_merge_limit();
                if(pending_ids.size()+shifted.size()<=merge_limit) {
                    pending_hidden.push_back(hidden_rows);
                    hidden_rows=ops.concatenate(pending_hidden,1);
                    pending_ids.insert(pending_ids.end(),shifted.begin(),shifted.end());
                    shifted=std::move(pending_ids);pending_ids.clear();pending_hidden.clear();
                } else flush_history();
            }
            const auto history_rows=static_cast<int64_t>(shifted.size());
            auto head = ops.predictor_step(std::move(hidden_rows),
                ops.ids_for(std::move(shifted)),
                positioned_prompt ? ops.decode_positions(predictor_history_position + 1,
                                         history_rows)
                                   : Tensor{});
            predictor_history_position += history_rows;
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
                    if constexpr(requires(Ops& value,const std::vector<int32_t>& ids) {
                        value.prefetch_target_prefix(int32_t{},ids);
                    })
                        ops.prefetch_target_prefix(next_ids.back(),result.tokens);
                    auto next = ops.predictor_step(chain_hidden,
                        ops.ids_for({token}),
                        positioned_prompt
                            ? ops.decode_positions(predictor_history_position + position + 1, 1)
                            : Tensor{});
                    sample_hidden = std::move(next.sample_hidden);
                    chain_hidden = std::move(next.chain_hidden);
                }
            }
            return result;
        };

        auto cycle_started = Clock::now();
        auto draft =
            prepare_draft(initial_hidden, {pending}, bounded_depth(depth_controller.depth()), true);
        double draft_ms=std::chrono::duration<double,std::milli>(Clock::now()-cycle_started).count();
        mtp.last_stats.head_ms+=draft_ms;
        while (generated < limit) {
            if (output.stopped()) {
                finish();
                co_return;
            }
            if (depth_controller.parked()) {
                const auto plain_started=Clock::now();
                Tensor raw_step;
                auto hidden_step = ops.target_forward(ops.ids_for({pending}), &raw_step);
                const auto next = ops.sample_constrained(ops.logits_for(hidden_step), counts, constraint_cursor);
                mtp.last_stats.target_ms+=std::chrono::duration<double,std::milli>(Clock::now()-plain_started).count();
                if (constraint_cursor) constraint_cursor->accept(next);
                committed_last_hidden = ops.slice(raw_step, 1, ops.size(raw_step, 1)-1, 1);
                auto delta = output.append(std::vector<int64_t>{next});
                accept(delta);
                prefetch_committed_token(next);
                ++mtp.last_stats.standard_tokens;
                bool time_sample=!maintenance;
                if constexpr(requires { ops.take_target_maintenance(); }) {
                    const bool initialized=ops.take_target_maintenance();time_sample=time_sample && !initialized;
                }
                depth_controller.observe_plain_window(cycle_started,Clock::now(),time_sample);
                if(const auto measured=depth_controller.measured_cycle_ms(0))
                    mtp.last_stats.measured_depth_ms[0]=*measured;
                mtp.last_stats.selected_depth = depth_controller.depth();
                mtp.last_stats.reentry_probes = depth_controller.reentries();
                pending = next;
                finish();
                co_yield std::move(delta);
                cycle_started = Clock::now();
                maintenance = false;
                draft = prepare_draft(committed_last_hidden, {pending},
                    output.stopped()?0:bounded_depth(depth_controller.depth()), false);
                draft_ms=std::chrono::duration<double,std::milli>(Clock::now()-cycle_started).count();
                mtp.last_stats.head_ms+=draft_ms;
                if(output.stopped()){finish();co_return;}
                continue;
            }
            const int draft_count = static_cast<int>(draft.tokens.size());
            // Depth selection pays for the draft prepared before verification
            // as well. Keep that complete cycle when comparing with depth zero.
            const auto verify_started=Clock::now();
            Tensor verified_raw;
            std::vector<int64_t> verify_ids{pending};
            verify_ids.insert(verify_ids.end(), draft.tokens.begin(), draft.tokens.end());
            Tensor verified;
            const bool split_target=mtp.split_target_verification();
            if (split_target) {
                Tensor pending_raw;
                auto pending_hidden = ops.target_forward(ops.ids_for({pending}), &pending_raw);
                if (draft_count > 0) {
                    // The pending token is already committed. Journal only
                    // the suffix, using the same hooks as joint verification.
                    if constexpr(requires { ops.begin_target_routes(int{}); })
                        ops.begin_target_routes(draft_count);
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
                if constexpr(requires { ops.begin_target_routes(int{}); })
                    if(draft_count>0)ops.begin_target_routes(draft_count+1);
                verified = ops.target_forward(
                    ops.ids_for(std::move(verify_ids)), &verified_raw, draft_count > 0 ? 1 : 0);
            }
            auto targets =
                ops.reshape(ops.logits_for(verified), {draft_count + 1, model.vocab_size()});
            const auto verify_finished=Clock::now();

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
                if constexpr (requires { ops.greedy_tokens(targets); }) {
                    if (!penalties) target_tokens = ops.greedy_tokens(targets);
                }
                if (target_tokens.empty()) {
                    target_tokens.reserve(static_cast<size_t>(draft_count + 1));
                    for (int row = 0; row <= draft_count; ++row) {
                        target_tokens.push_back(
                            ops.sample_normal(ops.slice(targets, 0, row, 1), row_counts));
                        if (row < draft_count && penalties) {
                            ops.add_counts(
                                row_counts, ops.ids_for({draft.tokens[static_cast<size_t>(row)]}));
                        }
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
            mtp.last_stats.target_ms+=std::chrono::duration<double,std::milli>(Clock::now()-verify_started).count();
            mtp.last_stats.accepted_tokens += static_cast<uint64_t>(accepted);
            for (int position = 0; position < accepted; ++position) {
                ++mtp.last_stats.position_accepted.at(static_cast<size_t>(position));
            }

            std::vector<int64_t> candidates(draft.tokens.begin(), draft.tokens.begin() + accepted);
            candidates.push_back(result.next_token);
            auto delta = output.append(candidates);
            accept(delta);
            const int emitted_accepted = std::min<int>(accepted, delta.token_ids.size());
            const auto resolve_routes=[&](int rows) {
                if constexpr(requires { ops.resolve_target_routes(int{},bool{}); })
                    ops.resolve_target_routes(rows-int(split_target),output.result.cancelled);
            };
            const auto resolve_started=Clock::now();
            if (draft_count > 0) {
                if (emitted_accepted == draft_count) {
                    model.commit_speculative();
                    resolve_routes(emitted_accepted+1);
                } else {
                    const bool retained = [&] {
                        if constexpr(requires { ops.retains_partial_target_prefix(emitted_accepted); })
                            return ops.retains_partial_target_prefix(emitted_accepted);
                        return mtp.retains_partial_target_prefix();
                    }();
                    model.rollback_speculative(retained ? emitted_accepted : 0);
                    // A fallback replay observes the accepted rows itself.
                    // Publish only pending here so those rows count once.
                    resolve_routes(retained?emitted_accepted+1:1);
                    if (!retained && emitted_accepted > 0) {
                        std::vector<int64_t> replay(
                            draft.tokens.begin(), draft.tokens.begin() + emitted_accepted);
                        (void)ops.target_forward(ops.ids_for(std::move(replay)));
                    }
                }
            }else resolve_routes(1);
            committed_last_hidden = ops.slice(verified_raw, 1, emitted_accepted, 1);
            mtp.last_stats.rollback_ms+=std::chrono::duration<double,std::milli>(Clock::now()-resolve_started).count();
            if (output.stopped()) {
                if (mtp.blockwise_drafting()) {
                    // Block predictors retain the committed target rows rather
                    // than shifted input/hidden pairs. EOS can truncate an
                    // accepted draft, so use the target's actual cache boundary.
                    const auto rows=model.cache_pos-predictor_history_position;
                    require_execution(rows>=0 && rows<=ops.size(verified_raw,1),
                        "block predictor final target boundary is incompatible");
                    if(rows>0) {
                        mtp.append_target_context(ops.slice(verified_raw,1,0,rows),predictor_history_position);
                        predictor_history_position+=rows;
                    }
                } else if (mtp.teacher_forced_prompt_prime() && !delta.token_ids.empty()) {
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

            prefetch_committed_token(result.next_token);
            const auto cycle_finished=Clock::now();
            const double cycle_ms =
                std::chrono::duration<double, std::milli>(cycle_finished - cycle_started).count();
            if constexpr(requires { ops.trace_cycle(uint64_t{},int{},int{},double{},double{},double{}); })
                ops.trace_cycle(mtp.last_stats.cycles,draft_count,accepted,draft_ms,
                    std::chrono::duration<double,std::milli>(verify_finished-verify_started).count(),
                    std::chrono::duration<double,std::milli>(Clock::now()-verify_finished).count());
            bool time_sample=!maintenance;
            if constexpr(requires { ops.take_target_maintenance(); }) {
                const bool initialized=ops.take_target_maintenance();time_sample=time_sample && !initialized;
            }
            depth_controller.observe_window(draft_count,accepted,cycle_started,cycle_finished,time_sample);
            mtp.last_stats.park_count = depth_controller.parks();
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
            // Publish committed tokens before computing the next draft. A
            // cold next-width replay must not hold this completed batch back.
            // Resume timing after the consumer returns, excluding its delay.
            finish();
            co_yield std::move(delta);
            cycle_started = Clock::now();
            maintenance = false;
            draft = prepare_draft(ops.slice(verified_raw, 1, 0, accepted + 1),
                next_ids,
                output.stopped()?0:bounded_depth(depth_controller.depth()),
                false);
            draft_ms=std::chrono::duration<double,std::milli>(Clock::now()-cycle_started).count();
            mtp.last_stats.head_ms+=draft_ms;
            // Cancellation between yields still commits predictor history for
            // prefix restoration, without running another speculative chain.
            if(output.stopped()){finish();co_return;}
        }
        finish();
    }
}

} // namespace mfq::engine
