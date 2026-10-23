#pragma once
#include "plan.h"
#include "step.h"
#include "activity.h"
#include <memory>
#include <span>
#include <unordered_map>

namespace mfq::engine {

// Logical row metadata. Native request storage is associated with the execution
// implementation; no slot, tensor, KV or recurrent layout is prescribed here.
struct BatchRequest {
    BatchRequest(std::string id, const InferenceRequest& input)
        : id(std::move(id)), prompt(input.prompt), sampling(input.sampling),
          token_constraint(input.token_constraint), generation_limit(input.sampling.max_tokens) {}
    std::string id;
    std::vector<int64_t> prompt;
    mfq::SamplingParams sampling;
    MfqTokenConstraintPtr token_constraint;
    int32_t generation_limit = 0, produced = 0;
    int64_t pending_token = 0;
    int64_t prefill_offset = 0, cache_length = 0;
};

struct BatchSample {
    int64_t token = 0;
    mfq::PrefillTiming timing;
};

template <class Sample> struct PrefillBatchResult { Sample sample; std::exception_ptr failure; };

template <class Request> struct PrefillWork {
    const std::shared_ptr<Request>& request;
    PrefillChunk chunk;
};
template <class Request> struct DecodeWork {
    std::span<const std::shared_ptr<Request>> requests;
};

// One seam: execute a selected numerical quantum and return its results.
// execute(PrefillWork) completes that chunk, including native state switching.
// execute(DecodeWork) returns ordered samples, or nullopt for preparation with
// no committed rows. accept(), release() and recover() own native state only;
// output, cancellation and publication remain in the shared request lifecycle.
template <class Execution> class ContinuousBatch {
    using Request = typename Execution::Request;
    struct Work {
        std::shared_ptr<Request> request;
        std::optional<PrefillChunk> prefill;
        std::optional<BatchSample> result;
        std::exception_ptr failure;
        bool decode = false;
    };
    int64_t prefill_chunk_size, prefill_token_budget;
    Execution execution;
    std::vector<std::shared_ptr<Request>> active;
    std::unordered_map<std::string, std::shared_ptr<Work>> requests;
    bool decode_next = true;
    int64_t admissions = 0, prefill_chunks = 0, prefill_yields = 0;

    void recover(std::exception_ptr error) {
        std::vector<std::shared_ptr<Request>> failed;
        for (const auto& [id, work] : requests) {
            work->failure = error;
            work->prefill.reset();
            work->decode = false;
            failed.push_back(work->request);
        }
        active.clear();
        execution.recover(failed);
    }
    void release(Work& work) {
        if (!requests.contains(work.request->id)) return;
        try {
            std::erase(active, work.request);
            if (!work.failure) execution.release(work.request);
        } catch (...) {
            auto error = std::current_exception();
            try { recover(error); } catch (...) { error = std::current_exception(); }
            requests.erase(work.request->id);
            std::rethrow_exception(error);
        }
        requests.erase(work.request->id);
    }
    struct SequenceOps {
        ContinuousBatch& batch;
        Work& work;
        void schedule_prefill(PrefillChunk chunk) { work.result.reset(); work.prefill = chunk; }
        void schedule_decode() { work.result.reset(); work.decode = true; }
        bool ready() const { return work.result.has_value() || work.failure; }
        const BatchSample& sample() const {
            if (work.failure) std::rethrow_exception(work.failure);
            return work.result.value();
        }
        double prefill(PrefillChunk) { return sample().timing.llm_ms; }
        int64_t first_token() { return sample().token; }
        int64_t advance() { return sample().token; }
        void accept(int64_t token) {
            auto& request = *work.request;
            batch.execution.accept(work.request, sample());
            request.pending_token = token;
            ++request.produced;
        }
    };

  public:
    bool empty() const noexcept { return requests.empty(); }
    void invalidate_runtime_policy() {
        if constexpr(requires { execution.invalidate_runtime_policy(); })execution.invalidate_runtime_policy();
    }
    void release_idle_buffers() {
        if(!empty())throw std::runtime_error("cannot release active batch buffers");
        if constexpr(requires { execution.suspend_decode(); })execution.suspend_decode();
        else execution.recover(active);
    }
    template <class... Args>
    explicit ContinuousBatch(int64_t chunk_size, int64_t token_budget, Args&&... args)
        : prefill_chunk_size(std::min(chunk_size, token_budget)), prefill_token_budget(token_budget),
          execution(std::forward<Args>(args)...) {
        if constexpr(requires { execution.prefill_batch_token_budget(); })
            prefill_chunk_size=std::min(prefill_chunk_size,execution.prefill_batch_token_budget());
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
        auto sequence = generate_sequence(ops, output, input.prompt.size(), 0, 0, prefill_chunk_size);
        while (auto step = sequence.next()) co_yield std::move(step);
        cleanup.finish();
    }

    void step(const std::vector<std::string>& eligible) {
        const auto plan = plan_batch(eligible, decode_next);
        std::vector<std::pair<std::shared_ptr<Request>,PrefillChunk>> prefills;
        const auto finish_prefill = [&](const std::shared_ptr<Request>& request, PrefillChunk chunk, BatchSample sample) {
            auto& work=*requests.at(request->id);
            request->prefill_offset=chunk.offset+chunk.count;
            ++prefill_chunks;
            if(request->prefill_offset==static_cast<int64_t>(request->prompt.size())) {
                request->cache_length=request->prompt.size();active.push_back(request);++admissions;
            } else ++prefill_yields;
            work.result=std::move(sample);work.prefill.reset();
        };
        const auto fail = [&](const std::shared_ptr<Request>& request, std::exception_ptr failure) {
            execution.release(request);
            std::erase(active,request);
            auto& work=*requests.at(request->id);work.failure=failure;work.prefill.reset();work.decode=false;
        };
        try {
            if (!plan.decode.empty()) {
                for(const auto& request:active)
                    if(std::find(eligible.begin(),eligible.end(),request->id)!=eligible.end() && !requests.at(request->id)->decode)return;
                auto decoded = execution.execute(DecodeWork<Request>{plan.decode});
                if (!decoded) return; // Preparation committed no row.
                if (decoded->size() != plan.decode.size())
                    throw std::logic_error("batch decode result count disagrees with selected rows");
                for (size_t row = 0; row < plan.decode.size(); ++row) {
                    auto& request = plan.decode[row];
                    auto& work = *requests.at(request->id);
                    work.result = std::move((*decoded)[row]);
                    ++request->cache_length;
                    work.decode = false;
                }
            } else if (plan.prefill) {
                PrefillActivity prefill_activity;
                if constexpr(requires { execution.prefill_batch(prefills); }) {
                    std::size_t limit=eligible.size();
                    if constexpr(requires { execution.prefill_batch_limit(); })limit=execution.prefill_batch_limit();
                    auto budget=prefill_token_budget;
                    if constexpr(requires { execution.prefill_batch_token_budget(); })budget=std::min(budget,execution.prefill_batch_token_budget());
                    for(const auto& id:eligible) {
                        const auto it=requests.find(id);
                        if(it==requests.end() || !it->second->prefill || prefills.size()>=limit || budget<=0)continue;
                        auto chunk=*it->second->prefill;
                        if(chunk.count>budget)continue;
                        prefills.emplace_back(it->second->request,chunk);budget-=chunk.count;
                    }
                    if(prefills.empty())throw std::logic_error("packed prefill selected no work");
                    auto samples=execution.prefill_batch(prefills);
                    if(samples.size()!=prefills.size())throw std::logic_error("packed prefill result count disagrees with selected rows");
                    for(std::size_t row=0;row<samples.size();++row) {
                        if constexpr(requires { samples[row].failure; }) {
                            if(samples[row].failure)fail(prefills[row].first,samples[row].failure);
                            else finish_prefill(prefills[row].first,prefills[row].second,std::move(samples[row].sample));
                        } else finish_prefill(prefills[row].first,prefills[row].second,std::move(samples[row]));
                    }
                } else {
                    prefills.emplace_back(plan.prefill,plan.prefill_chunk);
                    auto sample=execution.execute(PrefillWork<Request>{plan.prefill,plan.prefill_chunk});
                    finish_prefill(plan.prefill,plan.prefill_chunk,std::move(sample));
                }
            }
            decode_next = !decode_next;
        } catch (...) {
            auto failure=std::current_exception();
            bool capacity=false;
            try {std::rethrow_exception(failure);}
            catch(const InferenceInputError& error) {capacity=error.code==InferenceInputErrorCode::ContextLength;}
            catch(...) {}
            bool isolate=false;
            if constexpr(requires { Execution::isolate_prefill_failures; })
                isolate=plan.prefill && Execution::isolate_prefill_failures;
            if constexpr(requires { Execution::isolate_context_growth_failures; })
                isolate|=capacity && !plan.decode.empty() && Execution::isolate_context_growth_failures;
            bool recoverable=false;
            if constexpr(requires { execution.recoverable_failure(failure); })
                recoverable=execution.recoverable_failure(failure);
            if(isolate || recoverable) {
                try {
                    if(recoverable && !plan.decode.empty()) { recover(failure); return; }
                    if(!plan.decode.empty())fail(plan.decode.back(),failure);
                    for(const auto& [request,chunk]:prefills)fail(request,failure);
                    return;
                } catch(...) {failure=std::current_exception();}
            }
            recover(failure);
            std::rethrow_exception(failure);
        }
    }

    // One physical decision per step, made by the shared BatchPlanner over
    // this batch's own facts; see batch_planner/batch_plan.h. step() above
    // only executes the result.
    using Plan = BatchPlan<Request>;
    Plan plan_batch(const std::vector<std::string>& eligible, bool prefer_decode) const {
        return mfq::engine::plan_batch(active, requests, eligible, prefer_decode);
    }

    Metrics metrics() const {
        auto result = execution.metrics();
        result.emplace_back("continuous_batching_active", active.size());
        result.emplace_back("continuous_batching_prefilling", requests.size() - active.size());
        result.emplace_back("continuous_batching_requests", admissions);
        result.emplace_back("continuous_batching_admissions", admissions);
        result.emplace_back("continuous_batching_prefill_chunks", prefill_chunks);
        result.emplace_back("continuous_batching_prefill_yields", prefill_yields);
        return result;
    }
};
} // namespace mfq::engine
