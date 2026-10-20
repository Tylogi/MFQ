#include "resource_error.h"
#include "qwen4_batch_execution.h"
#include "engine/batch_execution.h"
#include "storage/weights/moe/expert_cache.h"
#include "ops/execution/execution_options.h"
namespace mfq::cuda {
using internal::PrefillCudaTimer;
Qwen4BatchExecution::Qwen4BatchExecution(Qwen4CausalLm& model,
        CudaExecutionContext& execution, CudaContinuousBatchConfig config)
    : model_(model), execution_(execution), config_(std::move(config)) {
    MFQ_RUNTIME_CHECK(config_.max_sequences > 0 && config_.max_sequences <= 32,
                      "Flash-Next continuous batching supports 1..32 requests");
    model_.reset(1);
    if (config_.cuda_graph) graph_ = std::make_unique<Qwen4BatchDecodeGraph>(model_);
}
Qwen4BatchExecution::~Qwen4BatchExecution() = default;
int64_t Qwen4BatchExecution::vocab_size() const { return model_.vocab_size(); }
int64_t Qwen4BatchExecution::max_context() const { return model_.max_position_embeddings(); }
mfq::engine::Metrics Qwen4BatchExecution::metrics() const {
    std::size_t bytes = 0, contexts = 0;
    for (const auto& weak : requests_) if (const auto request = weak.lock(); request && request->cache) {
        bytes += request->cache->resident_bytes();
        contexts += request->cache->position() > 0;
    }
    return {{"continuous_batching_decode_batches",double(decode_batches_)},
        {"continuous_batching_decode_tokens",double(decode_tokens_)},
        {"continuous_batching_batched_greedy_batches",double(greedy_batches_)},
        {"continuous_batching_packed_prefill_batches",double(packed_prefill_batches_)},
        {"continuous_batching_packed_prefill_requests",double(packed_prefill_requests_)},
        {"continuous_batching_packed_prefill_tokens",double(packed_prefill_tokens_)},
        {"continuous_batching_moe_cached_row_serial",0.0},
        {"continuous_batching_cuda_graph_captures",double(graph_ ? graph_->captures() : 0)},
        {"continuous_batching_cuda_graph_replays",double(graph_ ? graph_->replays() : 0)},
        {"continuous_batching_request_kv_bytes",double(bytes)},
        {"continuous_batching_request_contexts",double(contexts)}};
}
void Qwen4BatchExecution::suspend_decode() { if (graph_) graph_->clear(); }
void Qwen4BatchExecution::resume_decode(int64_t) {}
static std::exception_ptr flash_next_prefill_failure(std::exception_ptr failure) {
    try {std::rethrow_exception(failure);}
    catch(const std::exception& error) {
        const std::string message=error.what();
        if(dynamic_cast<const MfqResourceExhausted*>(&error) ||
           message.starts_with("CUDA process VRAM limit exceeded:") ||
           message.starts_with("CUDA VRAM limit exceeded:") ||
           message.starts_with("CUDA pinned allocation failed:"))
            return std::make_exception_ptr(mfq::engine::InferenceInputError(
                mfq::engine::InferenceInputErrorCode::ContextLength,
                "insufficient GPU or pinned RAM for prefill workspace; reduce concurrent requests or the packed prefill token budget"));
    } catch(...) {}
    return failure;
}
void Qwen4BatchExecution::initialize_prefill(
        const std::shared_ptr<Request>& request, mfq::engine::PrefillChunk chunk) {
    if (!request->cache) {
        MFQ_RUNTIME_CHECK(chunk.offset == 0,"Flash-Next batch lost request state");
        std::erase_if(requests_,[](const auto& weak) { return weak.expired(); });
        const bool active=std::any_of(requests_.begin(),requests_.end(),[](const auto& weak) {
            const auto row=weak.lock();return row && row->cache;
        });
        if(!active) {
            // The preceding exclusive request's global KV/recurrent state is
            // unused while independent batch rows own their own context.
            qwen4_release_decode_graphs(model_,nullptr);
            model_.reset(1);
            release_moe_decode_buffers(execution_.stores.moe_expert_cache);
            default_context(execution_.layer_placement.primary_device())->reclaim_idle_memory();
        }
        requests_.push_back(request);
        request->cache = std::make_unique<Qwen4BatchRequestState>(model_);
        request->prefill_ids = mfq_tensor_backend::tensor(request->prompt,
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64)
                .device(mfq_tensor_backend::Device(mfq_tensor_backend::kCUDA,
                    execution_.layer_placement.primary_device()))).reshape({1,-1});
        request->sampler.emplace(request->sampling);
        if (request->sampler->has_penalties())
            request->counts = SamplingOps::token_counts(request->prefill_ids,model_.vocab_size());
    }
    MFQ_RUNTIME_CHECK(request->cache->position() == chunk.offset,"Flash-Next batch prefill position mismatch");
}
Qwen4BatchExecution::Sample Qwen4BatchExecution::finish_prefill(
        const std::shared_ptr<Request>& request,Tensor next) {
    if (!next.defined()) return {};
    const auto token=next.item<int64_t>();
    double milliseconds=request->packed_prefill_ms;
    for (const auto& measured:request->prefill_timers) milliseconds+=measured->elapsed_ms();
    request->prefill_timers.clear(); request->prefill_ids={}; request->packed_prefill_ms=0;
    request->pending=std::move(next);
    return {token,{request->prompt.size(),milliseconds,0.0,milliseconds}};
}
Qwen4BatchExecution::Sample Qwen4BatchExecution::prefill(
        const std::shared_ptr<Request>& request, mfq::engine::PrefillChunk chunk) {
    try {
    MfqCudaGuard guard(execution_.layer_placement.primary_device());
    initialize_prefill(request,chunk);
    auto timer = std::make_unique<PrefillCudaTimer>();
    auto hidden = request->cache->prefill(model_,request->prefill_ids.narrow(1,chunk.offset,chunk.count).contiguous());
    const auto finished = chunk.offset + chunk.count == static_cast<int64_t>(request->prompt.size());
    Tensor next;
    if (finished) next = sample_logits(*request->sampler,
        model_.logits_from_hidden(model_.last_hidden(std::move(hidden))).reshape({1,-1}),
        request->counts,request->token_constraint);
    MFQ_CUDA_CHECK(cudaEventRecord(timer->finished_event(),mfq_get_current_cuda_stream()));
    request->prefill_timers.push_back(std::move(timer));
    return finish_prefill(request,std::move(next));
    } catch(...) {std::rethrow_exception(flash_next_prefill_failure(std::current_exception()));}
}
std::size_t Qwen4BatchExecution::prefill_batch_limit() const {
    return runtime_options::batch_packed_prefill()?config_.max_sequences:1;
}
int64_t Qwen4BatchExecution::prefill_batch_token_budget() const {
    return runtime_options::batch_packed_prefill_token_budget();
}
std::vector<mfq::engine::PrefillBatchResult<Qwen4BatchExecution::Sample>>
Qwen4BatchExecution::prefill_batch(
        const std::vector<std::pair<std::shared_ptr<Request>,mfq::engine::PrefillChunk>>& items) {
    MfqCudaGuard guard(execution_.layer_placement.primary_device());
    qwen4_release_decode_graphs(model_,nullptr);
    model_.begin_forward(false);
    release_moe_decode_buffers(execution_.stores.moe_expert_cache);
    std::vector<mfq::engine::PrefillBatchResult<Sample>> result(items.size());
    std::vector<std::size_t> ready;
    for (std::size_t row=0;row<items.size();++row) {
        try { initialize_prefill(items[row].first,items[row].second); ready.push_back(row); }
        catch (...) { result[row].failure=flash_next_prefill_failure(std::current_exception()); }
    }
    const auto execute=[&](auto&& self,const std::vector<std::size_t>& rows)->void {
        if (rows.empty()) return;
        std::size_t growth=std::size_t(256)<<20;
        for (const auto row:rows) growth+=items[row].first->cache->prefill_growth_bytes(items[row].second.count);
        try { if(model_.reserve_kv_workspace)model_.reserve_kv_workspace(growth); }
        catch (const mfq::engine::InferenceInputError& error) {
            if(error.code!=mfq::engine::InferenceInputErrorCode::ContextLength)throw;
            if(rows.size()==1) {result[rows.front()].failure=std::current_exception();return;}
            const auto middle=rows.begin()+rows.size()/2;
            self(self,std::vector<std::size_t>(rows.begin(),middle));
            self(self,std::vector<std::size_t>(middle,rows.end()));
            return;
        }
        try {
            if(rows.size()==1) {
                const auto row=rows.front();
                result[row].sample=prefill(items[row].first,items[row].second);
                return;
            }
            std::vector<Qwen4BatchRequestState*> caches;
            std::vector<Tensor> chunks;
            int64_t tokens=0;
            for (const auto row:rows) {
                const auto& [request,chunk]=items[row];
                caches.push_back(request->cache.get());
                chunks.push_back(request->prefill_ids.narrow(1,chunk.offset,chunk.count).contiguous());
                tokens+=chunk.count;
            }
            PrefillCudaTimer timer;
            auto hidden=qwen4_batch_prefill(model_,caches,chunks);
            std::vector<Tensor> last;
            std::vector<std::size_t> finished;
            for(std::size_t i=0;i<rows.size();++i) {
                const auto& [request,chunk]=items[rows[i]];
                if(chunk.offset+chunk.count==static_cast<int64_t>(request->prompt.size())) {
                    last.push_back(hidden[i]);finished.push_back(rows[i]);
                }
            }
            std::vector<Tensor> next(items.size());
            if(!last.empty()) {
                auto logits=model_.logits_from_hidden(mfq_tensor_backend::cat(last,0)).reshape({int64_t(last.size()),-1});
                for(std::size_t i=0;i<finished.size();++i) {
                    const auto row=finished[i];const auto& request=items[row].first;
                    next[row]=sample_logits(*request->sampler,logits.narrow(0,i,1),request->counts,request->token_constraint);
                }
            }
            MFQ_CUDA_CHECK(cudaEventRecord(timer.finished_event(),mfq_get_current_cuda_stream()));
            const auto milliseconds=timer.elapsed_ms();
            for(const auto row:rows) {
                items[row].first->packed_prefill_ms+=milliseconds*items[row].second.count/tokens;
                result[row].sample=finish_prefill(items[row].first,std::move(next[row]));
            }
            ++packed_prefill_batches_;packed_prefill_requests_+=rows.size();packed_prefill_tokens_+=tokens;
        } catch (...) {
            // Kernels may have advanced attention state. Fail this group;
            // splitting/retrying is allowed only before numerical execution.
            const auto error=flash_next_prefill_failure(std::current_exception());
            for(const auto row:rows)result[row].failure=error;
        }
    };
    execute(execute,ready);
    return result;
}
void Qwen4BatchExecution::activate(const std::shared_ptr<Request>& request, const Sample& sampled) {
    accept(request,sampled);
}
void Qwen4BatchExecution::discard_prefill(const std::shared_ptr<Request>& request) {
    if (graph_) graph_->clear();
    finish_moe_expert_exchanges(execution_.stores.moe_expert_cache);
    request->cache.reset(); request->prefill_ids = {}; request->prefill_timers.clear(); request->packed_prefill_ms=0;
}
void Qwen4BatchExecution::retire(const std::vector<std::shared_ptr<Request>>& retired,int64_t) {
    for (const auto& request : retired) discard_prefill(request);
    const bool active = std::any_of(requests_.begin(),requests_.end(),[](const auto& weak) {
        const auto request = weak.lock();
        return request && request->cache;
    });
    if (!active && !model_.graph_registry) {
        // Batch widths have distinct transfer arenas and graph scratch. Return
        // them at idle while preserving resident experts and adaptation heat.
        release_moe_decode_buffers(execution_.stores.moe_expert_cache);
        default_context(execution_.layer_placement.primary_device())->reclaim_idle_memory();
    }
}
Qwen4BatchExecution::Decoded Qwen4BatchExecution::decode(State& state) {
    MfqCudaGuard guard(execution_.layer_placement.primary_device());
    std::vector<Qwen4BatchRequestState*> caches;
    std::vector<int64_t> tokens;
    for (const auto& request : state.active) {
        request->decode_row = -1;
        if (!request->eligible) continue;
        request->decode_row = static_cast<int64_t>(tokens.size());
        MFQ_RUNTIME_CHECK(request->cache && request->cache->position() == request->cache_length,
                          "Flash-Next batch decode position mismatch");
        caches.push_back(request->cache.get()); tokens.push_back(request->pending_token);
    }
    auto host_ids = mfq_tensor_backend::tensor(tokens,
        mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64)).reshape({-1,1});
    auto ids = host_ids.to(mfq_tensor_backend::Device(mfq_tensor_backend::kCUDA,
                                                   execution_.layer_placement.primary_device()));
    MoeBatchCacheScope moe_cache_scope(execution_,tokens.size() > 8);
    const bool greedy = config_.greedy && std::all_of(state.active.begin(),state.active.end(),[](const auto& request) {
        return !request->eligible || (request->sampler->greedy() && !request->sampler->has_penalties() && !request->token_constraint);
    });
    std::optional<Tensor> captured;
    try { if(graph_)captured=graph_->run(caches,ids,host_ids,greedy); }
    catch(const std::exception& error) {
        const std::string message=error.what();
        if(!dynamic_cast<const MfqResourceExhausted*>(&error) &&
           !message.starts_with("CUDA process VRAM limit exceeded:") &&
           !message.starts_with("CUDA VRAM limit exceeded:") &&
           !message.starts_with("CUDA pinned allocation failed:"))throw;
        throw mfq::engine::InferenceInputError(mfq::engine::InferenceInputErrorCode::ContextLength,
            "insufficient GPU or pinned RAM for the batch decode workspace; reduce concurrent requests");
    }
    auto logits = captured ? std::move(*captured) : qwen4_batch_decode(model_,caches,std::move(ids),host_ids);
    Tensor greedy_tokens;
    if (greedy) {
        greedy_tokens = captured ? logits.to(mfq_tensor_backend::kCPU).contiguous()
            : sample_greedy_cuda(logits.contiguous()).to(mfq_tensor_backend::kCPU).contiguous();
        ++greedy_batches_;
    }
    ++decode_batches_; decode_tokens_ += tokens.size();
    return {std::move(logits),std::move(greedy_tokens)};
}
Qwen4BatchExecution::Sample Qwen4BatchExecution::sample(
        const std::shared_ptr<Request>& request,const Decoded& decoded) {
    MFQ_RUNTIME_CHECK(request->decode_row >= 0,"Flash-Next sampled an inactive row");
    if(decoded.greedy_tokens.defined())return {decoded.greedy_tokens.data_ptr<int64_t>()[request->decode_row],{}};
    auto next = sample_logits(*request->sampler,decoded.logits.narrow(0,request->decode_row,1),
                             request->counts,request->token_constraint);
    const auto token = next.item<int64_t>();
    request->pending=std::move(next);
    return {token,{}};
}
void Qwen4BatchExecution::accept(const std::shared_ptr<Request>& request,const Sample& sampled) {
    if (request->counts.defined()) sample_token_counts_add_cuda(request->counts,request->pending.contiguous());
}
void Qwen4BatchExecution::recover(State& state) {
    for (const auto& request : state.active) discard_prefill(request);
    for (const auto& request : state.prefilling) discard_prefill(request);
}
Qwen4BatchExecution::Sample Qwen4BatchExecution::execute(mfq::engine::PrefillWork<Request> work) {
    return prefill(work.request,work.chunk);
}
std::optional<std::vector<Qwen4BatchExecution::Sample>> Qwen4BatchExecution::execute(mfq::engine::DecodeWork<Request> work) {
    State state;state.active.assign(work.requests.begin(),work.requests.end());
    const auto decoded=decode(state);
    std::vector<Sample> result;result.reserve(work.requests.size());
    for(const auto& request:work.requests)result.push_back(sample(request,decoded));
    return result;
}
void Qwen4BatchExecution::release(const std::shared_ptr<Request>& request) { retire({request},0); }
void Qwen4BatchExecution::recover(std::span<const std::shared_ptr<Request>> requests) {
    for(const auto& request:requests)discard_prefill(request);
}
} // namespace mfq::cuda
