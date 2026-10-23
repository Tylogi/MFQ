#pragma once
#include "generation_step.h"
#include "kv_admission.h"
#include <deque>
#include <memory>
#include <unordered_map>

namespace mfq::engine {

// Physical row state and replay inputs. Lifecycle and output publication belong to the executor.
struct BatchRequest {
    BatchRequest(std::string id, const InferenceRequest& input)
        : id(std::move(id)), prompt(input.prompt), sampling(input.sampling),
          token_constraint(input.token_constraint), generation_limit(input.sampling.max_tokens) {}
    std::string id;
    std::vector<int64_t> prompt;
    MfqSamplingParams sampling;
    MfqTokenConstraintPtr token_constraint;
    int32_t generation_limit = 0, produced = 0;
    int64_t pending_token = 0;
    bool eligible = false;
    std::vector<int64_t> accepted_tokens;
    int64_t replay_offset = -1;
    int64_t replay_length() const { return prompt.size() + std::max<int64_t>(0, produced - 1); }
    int64_t prefill_offset = 0, cache_length = 0;
};

template <class Request> struct BatchState {
    std::deque<std::shared_ptr<Request>> prefilling;
    std::vector<std::shared_ptr<Request>> active;
};

template <class Sample> struct PrefillBatchResult {
    Sample sample;
    std::exception_ptr failure;
};

// Groups pending numerical operations. All rows run generate_sequence(), also
// used by B=1 execution; this component never accepts tokens or publishes events.
template <class Ops> class ContinuousBatch {
    using Request = typename Ops::Request;
    using Sample = typename Ops::Sample;
    using State = BatchState<Request>;
    struct Work {
        std::shared_ptr<Request> request;
        std::optional<PrefillChunk> prefill, completed_prefill;
        std::optional<Sample> result;
        std::exception_ptr failure;
        bool decode = false, active = false, admitted = false;
    };
    int64_t prefill_chunk_size, prefill_token_budget;
    Ops operations;
    State state;
    std::unordered_map<std::string, std::shared_ptr<Work>> requests;
    bool decode_next = true;
    NewTokenRatioTracker ratio;
    int64_t admission_deferrals = 0, retractions = 0;

    bool can_admit(const Work& candidate) {
        if constexpr (requires { operations.admission_available(); }) {
            if (candidate.admitted) return true;
            KvAdmissionBudget budget(operations.admission_available());
            bool occupied = false;
            for (const auto& [id, work] : requests) {
                if (!work->admitted || work->failure) continue;
                occupied = true;
                const auto& request = work->request;
                if (!budget.reserve(operations.admission_cost(request,
                        ratio.expected(request->generation_limit, request->produced)))) {
                    ++admission_deferrals;
                    return false;
                }
            }
            const auto& request = candidate.request;
            if (budget.reserve(operations.admission_cost(request,
                    ratio.expected(request->generation_limit, request->produced)))) return true;
            // One request must make progress when only its speculative decode
            // estimate is too large. The entire uncached prompt still must fit.
            if (!occupied) {
                KvAdmissionBudget solitary(operations.admission_available());
                if (solitary.reserve(operations.admission_cost(request, 1))) return true;
            }
            ++admission_deferrals;
            return false;
        }
        return true;
    }
    bool retract_one() {
        if constexpr (requires { operations.retract(state.active.front()); }) {
            if (state.active.size() <= 1) return false;
            for (const auto& request : retraction_order(state.active)) {
                auto& work = *requests.at(request->id);
                if (!work.decode) continue;
                const auto eligible_rows = std::count_if(state.active.begin(), state.active.end(),
                    [](const auto& row) { return row->eligible; });
                if (request->eligible && eligible_rows <= 1) continue;
                operations.retract(request);
                std::erase(state.active, request);
                work.active = work.admitted = false;
                request->eligible = false;
                request->replay_offset = 0;
                request->cache_length = 0;
                work.prefill = PrefillChunk{0, std::min(prefill_chunk_size, request->replay_length())};
                ++retractions;
                ratio.after_retraction(state.active);
                return true;
            }
        }
        return false;
    }
    int64_t admissions = 0, prefill_chunks = 0, prefill_yields = 0;

    int64_t cache_position() const {
        int64_t position = 0;
        for (const auto& request : state.active)
            position = std::max(position, request->cache_length);
        return position;
    }
    void recover(std::exception_ptr error) {
        State failed;
        for (const auto& [id, work] : requests) {
            work->failure = error;
            work->prefill.reset();
            work->decode = work->active = false;
            failed.prefilling.push_back(work->request);
        }
        state = {};
        operations.recover(failed);
    }
    void release(Work& work) {
        if (!requests.contains(work.request->id)) return;
        try {
            if (work.active) {
                std::erase(state.active, work.request);
                operations.retire({work.request}, cache_position());
            } else if (!work.failure) {
                operations.suspend_decode();
                operations.discard_prefill(work.request);
                operations.resume_decode(cache_position());
            }
        } catch (...) {
            auto error = std::current_exception();
            try { recover(error); } catch (...) { error = std::current_exception(); }
            requests.erase(work.request->id);
            std::rethrow_exception(error);
        }
        requests.erase(work.request->id);
        if (requests.empty()) ratio.reset();
    }
    struct SequenceOps {
        ContinuousBatch& batch;
        Work& work;
        PrefillChunk prefill_chunk(PrefillChunk chunk) {
            if constexpr (!requires { batch.operations.admission_available(); }) {
                if constexpr (requires { batch.operations.prefill_chunk(work.request, chunk); })
                    return batch.operations.prefill_chunk(work.request, chunk);
            }
            return chunk;
        }
        void schedule_prefill(PrefillChunk chunk) {
            work.result.reset();
            work.completed_prefill.reset();
            work.prefill = chunk;
        }
        PrefillChunk completed_prefill_chunk(PrefillChunk chunk) const {
            return work.completed_prefill.value_or(chunk);
        }
        void schedule_decode() { work.result.reset(); work.decode = true; }
        bool ready() const { return work.result.has_value() || work.failure; }
        const Sample& sample() const {
            if (work.failure) std::rethrow_exception(work.failure);
            return work.result.value();
        }
        double prefill(PrefillChunk) { return sample().timing.llm_ms; }
        int64_t first_token() { return sample().token; }
        int64_t advance() { return sample().token; }
        std::optional<Generation> generate_decode(InferenceOutput& output) {
            if constexpr (requires { batch.operations.generate_decode(work.request, output,
                    std::function<void()>{}, std::function<bool()>{}, std::function<void()>{}); })
                return batch.operations.generate_decode(work.request, output,
                    [this] { schedule_decode(); }, [this] { return ready(); },
                    [this] { (void)sample(); });
            return std::nullopt;
        }
        void accept(int64_t token) {
            auto& request = *work.request;
            // activate() initializes counts for the first sample.
            if (request.produced) batch.operations.accept(work.request, sample());
            if constexpr (requires { batch.operations.retract(work.request); })
                request.accepted_tokens.push_back(token);
            request.pending_token = token;
            ++request.produced;
        }
    };

  public:
    bool empty() const noexcept { return requests.empty(); }
    void invalidate_runtime_policy() {
        if constexpr (requires { operations.invalidate_runtime_policy(); }) operations.invalidate_runtime_policy();
    }
    void release_idle_buffers() {
        if(!empty())throw std::runtime_error("cannot release active batch buffers");
        if constexpr (requires { operations.release_idle_buffers(); })
            operations.release_idle_buffers();
        else
            operations.suspend_decode();
    }
    template <class... Args>
    explicit ContinuousBatch(int64_t chunk_size, int64_t token_budget, Args&&... args)
        : prefill_chunk_size(std::min(chunk_size, token_budget)), prefill_token_budget(token_budget),
          operations(std::forward<Args>(args)...) {
        if constexpr (requires { operations.admission_config(); })
            ratio = NewTokenRatioTracker(operations.admission_config());
        if (prefill_chunk_size <= 0)
            throw std::invalid_argument("continuous batching requires a positive prefill chunk");
    }

    Generation generate(std::string id, InferenceRequest& input, InferenceOutput& output) {
        if (output.stopped()) co_return;
        auto work = std::make_shared<Work>();
        work->request = std::make_shared<Request>(std::move(id), input);
        if (!requests.emplace(work->request->id, work).second)
            throw std::invalid_argument("duplicate batch request id");
        ExecutionCleanup cleanup{output.cleanup_failure, [&] { release(*work); }};
        SequenceOps ops{*this, *work};
        int64_t reused = 0, prefix = 0;
        if constexpr (requires { operations.prepare(work->request); })
            reused = operations.prepare(work->request);
        work->request->prefill_offset = reused;
        if constexpr (requires { operations.prefix_tokens(work->request); })
            prefix = operations.prefix_tokens(work->request);
        auto sequence = generate_sequence(ops, output, input.prompt.size(), reused, prefix, prefill_chunk_size);
        while (auto step = sequence.next()) co_yield std::move(step);
        cleanup.finish();
    }

    void step(const std::vector<std::string>& eligible) {
        for (const auto& request : state.active) {
            const auto& work = *requests.at(request->id);
            request->eligible = work.decode &&
                std::find(eligible.begin(), eligible.end(), request->id) != eligible.end();
        }
        auto prefill = requests.end();
        for (const auto& id : eligible) {
            auto it = requests.find(id);
            if (it != requests.end() && it->second->prefill && can_admit(*it->second)) {
                prefill = it;
                break;
            }
        }
        const bool rows_ready = std::all_of(state.active.begin(), state.active.end(),
            [&](const auto& request) {
                return std::find(eligible.begin(), eligible.end(), request->id) == eligible.end() ||
                    requests.at(request->id)->decode;
            });
        // Let eligible peers consume their previous sample and schedule the
        // next decode before executing it. Otherwise staggered coroutines
        // alternate single-row decodes and never form a stable batch.
        const bool has_decode = rows_ready && std::any_of(state.active.begin(), state.active.end(),
            [](const auto& request) { return request->eligible; });
        bool run_decode = has_decode && (prefill == requests.end() || decode_next);
        if constexpr (requires { operations.ready_decode(state, eligible); })
            run_decode = run_decode && operations.ready_decode(state, eligible);
        bool prefilling_operation = false;
        bool decoding_operation = false;
        std::vector<std::shared_ptr<Work>> prefill_work;
        try {
            if constexpr (requires { operations.synchronize(state); }) operations.synchronize(state);
            if (run_decode) {
                if constexpr (requires { operations.decode_has_capacity(state); }) {
                    while (!operations.decode_has_capacity(state)) {
                        if (!retract_one()) break;
                    }
                }
                decoding_operation = true;
                auto decoded = operations.decode(state);
                decoding_operation = false;
                ratio.decay_step();
                for (const auto& request : state.active) if (request->eligible) {
                    auto& work = *requests.at(request->id);
                    work.result = operations.sample(request, decoded);
                    if constexpr (requires { operations.cache_length(request); })
                        request->cache_length = operations.cache_length(request);
                    else ++request->cache_length;
                    work.decode = false;
                }
            } else if (prefill != requests.end()) {
                operations.suspend_decode();
                prefilling_operation = true;
                prefill_work.push_back(prefill->second);
                prefill->second->admitted = true;
                if constexpr (requires { operations.admission_available(); }) {
                    if (prefill->second->request->replay_offset < 0)
                        prefill->second->prefill = operations.prefill_chunk(
                            prefill->second->request, *prefill->second->prefill);
                }
                const bool replaying = prefill->second->request->replay_offset >= 0;
                using Item = std::pair<std::shared_ptr<Request>, PrefillChunk>;
                std::vector<Item> items{{prefill->second->request, *prefill->second->prefill}};
                if constexpr (requires { operations.prefill_batch(items); }) {
                    auto budget=prefill_chunk_size;
                    if constexpr (requires { operations.prefill_batch_limit(); }) budget=prefill_token_budget;
                    if constexpr(requires {operations.prefill_batch_token_budget();})
                        budget=std::min(budget,operations.prefill_batch_token_budget());
                    auto remaining = budget - items.front().second.count;
                    const auto limit = [&] {
                        if constexpr (requires { operations.prefill_batch_limit(); }) return operations.prefill_batch_limit();
                        else return eligible.size();
                    }();
                    for (const auto& id : eligible) {
                        auto it = requests.find(id);
                        if (it == requests.end() || !it->second->prefill ||
                            std::find(prefill_work.begin(),prefill_work.end(),it->second) != prefill_work.end()) continue;
                        if (replaying || it->second->request->replay_offset >= 0) continue;
                        auto chunk = *it->second->prefill;
                        if (items.size() >= limit || chunk.count > remaining || !can_admit(*it->second)) continue;
                        it->second->admitted = true;
                        if constexpr (requires { operations.admission_available(); })
                            chunk = operations.prefill_chunk(it->second->request, chunk);
                        remaining -= chunk.count;
                        prefill_work.push_back(it->second);
                        items.emplace_back(it->second->request,chunk);
                    }
                }
                PrefillActivity prefill_activity;
                std::vector<PrefillBatchResult<Sample>> samples;
                if constexpr (requires { operations.replay_prefill(items.front().first, items.front().second); }) {
                    if (replaying) {
                        operations.replay_prefill(items.front().first, items.front().second);
                        samples.push_back({Sample{}, {}});
                    }
                }
                if constexpr (requires { operations.prefill_batch(items); }) {
                    const bool use_batch = items.size() > 1 || !requires { operations.prefill_batch_limit(); };
                    if (use_batch && !replaying) {
                        auto results = operations.prefill_batch(items);
                        for (auto& result : results) {
                            if constexpr (requires { result.failure; result.sample; }) samples.push_back(std::move(result));
                            else samples.push_back({std::move(result), {}});
                        }
                    }
                }
                if (samples.empty()) {
                    const auto& [request,chunk]=items.front();
                    samples.push_back({operations.prefill(request, chunk),{}});
                }
                prefilling_operation = false;
                if (samples.size() != items.size()) throw std::runtime_error("invalid packed prefill result count");
                for (std::size_t row = 0; row < items.size(); ++row) {
                    auto& work = *prefill_work[row];
                    auto& request = work.request;
                    auto& result = samples[row];
                    const auto chunk = items[row].second;
                    if (replaying && !result.failure) {
                        request->replay_offset = chunk.offset + chunk.count;
                        request->cache_length = request->replay_offset;
                        ++prefill_chunks;
                        if (request->replay_offset == request->replay_length()) {
                            request->replay_offset = -1;
                            state.active.push_back(request);
                            work.active = true;
                            work.prefill.reset();
                        } else {
                            work.prefill = PrefillChunk{request->replay_offset,
                                std::min(prefill_chunk_size, request->replay_length() - request->replay_offset)};
                            ++prefill_yields;
                        }
                        continue;
                    }
                    if (result.failure) {
                        work.failure = result.failure;
                        operations.discard_prefill(request);
                    } else {
                        request->prefill_offset = chunk.offset + chunk.count;
                        ++prefill_chunks;
                        if (request->prefill_offset == static_cast<int64_t>(request->prompt.size())) {
                            request->cache_length = request->prompt.size();
                            operations.activate(request,result.sample);
                            state.active.push_back(request);
                            work.active = true;
                            ++admissions;
                        } else ++prefill_yields;
                        work.completed_prefill = chunk;
                        work.result = std::move(result.sample);
                    }
                    work.prefill.reset();
                }
                operations.resume_decode(cache_position());
            }
            decode_next = !decode_next;
        } catch (...) {
            if constexpr (requires { Ops::isolate_context_growth_failures; }) {
                bool capacity_failure = false;
                try { throw; }
                catch (const InferenceInputError& error) {
                    capacity_failure = error.code == InferenceInputErrorCode::ContextLength;
                } catch (...) {}
                if (Ops::isolate_context_growth_failures && decoding_operation && capacity_failure) {
                    if (retract_one()) return;
                    // Capacity checks run before numerical decode. Release the
                    // newest eligible request while preserving established rows.
                    const auto victim = std::find_if(state.active.rbegin(),state.active.rend(),
                        [](const auto& request) { return request->eligible; });
                    if (victim != state.active.rend()) {
                        auto request = *victim;
                        auto& work = *requests.at(request->id);
                        work.failure = std::current_exception(); work.decode = false;
                        std::erase(state.active,request);
                        try {
                            operations.retire({request},cache_position());
                            work.active = false;
                            return;
                        } catch (...) { recover(std::current_exception()); throw; }
                    }
                }
            }
            if constexpr (requires { Ops::isolate_prefill_failures; }) {
                if (Ops::isolate_prefill_failures && prefilling_operation && prefill != requests.end()) {
                    const auto error = std::current_exception();
                    try {
                        for (const auto& work : prefill_work) {
                            work->failure = error;
                            work->prefill.reset();
                            work->decode = false;
                            operations.discard_prefill(work->request);
                        }
                        operations.resume_decode(cache_position());
                        return;
                    } catch (...) { recover(std::current_exception()); throw; }
                }
            }
            const auto error = std::current_exception();
            if constexpr (requires { operations.recoverable_failure(error); }) {
                if (operations.recoverable_failure(error)) {
                    if (!run_decode && prefill != requests.end()) {
                        for (const auto& item : prefill_work) {
                            auto& work = *item;
                            work.failure = error;
                            work.prefill.reset();
                            work.decode = work.active = false;
                            std::erase(state.active, work.request);
                            operations.discard_prefill(work.request);
                        }
                        operations.resume_decode(cache_position());
                    } else recover(error);
                    return;
                }
            }
            recover(error);
            throw;
        }
    }

    Metrics metrics() const {
        auto result = operations.metrics();
        if constexpr (requires { operations.admission_available(); }) {
            result.emplace_back("continuous_batching_admission_deferred", admission_deferrals);
            result.emplace_back("continuous_batching_retractions", retractions);
            result.emplace_back("continuous_batching_new_token_ratio", ratio.current());
        }
        result.emplace_back("continuous_batching_active", state.active.size());
        result.emplace_back("continuous_batching_prefilling", requests.size() - state.active.size());
        result.emplace_back("continuous_batching_requests", admissions);
        result.emplace_back("continuous_batching_admissions", admissions);
        result.emplace_back("continuous_batching_prefill_chunks", prefill_chunks);
        result.emplace_back("continuous_batching_prefill_yields", prefill_yields);
        return result;
    }
};
} // namespace mfq::engine
