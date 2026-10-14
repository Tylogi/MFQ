#include "speculative_sequence.h"
#include "mtp_batch_policy.h"

#include <iostream>
#include <thread>

namespace {
using namespace mfq::engine;

struct Tensor {
    std::vector<int64_t> shape;
    bool proposal = false;
    int64_t start = 0;
};

struct Model {
    int64_t cache_pos = 0, decode_position_delta = 0, speculative_start = 0;
    int vocab_size() const { return 3; }
    int max_position_embeddings() const { return 1024; }
    void begin_speculative_suffix(int) { speculative_start = cache_pos-1; }
    void commit_speculative() {}
    void rollback_speculative(int keep) { cache_pos = speculative_start+1+keep; }
};

struct Predictor {
    mtp::GenerationStats last_stats;
    mtp::PolicyState policy_state;
    uint64_t last_cycles = 0, last_accepted = 0, last_rejected = 0;
    int64_t position = 0;
    bool legacy = false;
    bool bootstrap = false;
    int maximum_draft_depth() const { return 3; }
    bool dspark_policy() const { return legacy; }
    bool blockwise_drafting() const { return false; }
    bool supports_session_state() const { return true; }
    bool teacher_forced_prompt_prime() const { return true; }
    bool target_bootstrap_decode() const { return bootstrap; }
    bool split_target_verification() const { return false; }
    bool retains_partial_target_prefix() const { return true; }
    int64_t cache_position() const { return position; }
    void trim_cache_to(int64_t value) { position = value; }
    void append_target_context(Tensor, int64_t) { throw std::logic_error("not blockwise"); }
    Tensor draft_block(int, Tensor, int) { throw std::logic_error("not blockwise"); }
    Tensor draft_next(Tensor, Tensor) { throw std::logic_error("not blockwise"); }
};

struct Prepared {
    std::vector<int64_t> token_ids;
    Tensor embeddings, positions;
    int decode_position_delta = 0;
    bool transformed() const { return false; }
};

struct Sampler {
    bool greedy() const { return true; }
    double next_uniform() { return 0.5; }
};

struct Ops {
    using Tensor = ::Tensor;
    Model& model;
    Predictor& mtp;
    const Prepared* prepared = nullptr;
    int target = 0;
    Tensor input_ids, counts;
    bool penalties = false;
    Sampler sampler_;
    void initialize(const std::vector<int64_t>& prompt, const MfqSamplingParams&) {
        input_ids = ids_for(prompt);
    }
    Sampler& sampler() { return sampler_; }
    static bool defined(const Tensor& t) { return !t.shape.empty(); }
    static int rank(const Tensor& t) { return t.shape.size(); }
    static int64_t size(const Tensor& t, int axis) {
        return t.shape.at(axis < 0 ? t.shape.size()+axis : axis);
    }
    static Tensor clone(const Tensor& t) { return t; }
    static Tensor slice(Tensor t, int axis, int64_t, int64_t length) {
        t.shape.at(axis < 0 ? t.shape.size()+axis : axis) = length;
        return t;
    }
    static Tensor reshape(Tensor t, std::initializer_list<int64_t> shape) {
        int64_t total = 1, fixed = 1;
        for (int64_t dim : t.shape) total *= dim;
        t.shape = shape;
        for (int64_t dim : t.shape) if (dim > 0) fixed *= dim;
        for (auto& dim : t.shape) if (dim < 0) dim = total/fixed;
        return t;
    }
    static Tensor concatenate(const std::vector<Tensor>& values, int axis) {
        auto result = values.at(0);
        result.shape.at(axis) = 0;
        for (const auto& value : values) result.shape.at(axis) += value.shape.at(axis);
        return result;
    }
    static Tensor ids_for(const std::vector<int64_t>& ids) {
        return {{1, static_cast<int64_t>(ids.size())}};
    }
    static void add_counts(Tensor, Tensor) {}
    static Tensor decode_positions(int64_t, int64_t count) { return {{count}}; }
    Tensor target_forward(Tensor ids, Tensor* raw = nullptr, int64_t confirmed = 0) {
        const auto count = size(ids, 1);
        if (confirmed) model.speculative_start = model.cache_pos;
        model.cache_pos += count;
        Tensor hidden{{1, count, 1}};
        if (raw) *raw = hidden;
        std::this_thread::sleep_for(std::chrono::milliseconds(count > 1 ? 12 : 1));
        return hidden;
    }
    Tensor target_suffix(Tensor ids, Tensor* raw) { return target_forward(ids, raw); }
    struct Prefill { Tensor hidden, raw; double elapsed; };
    Prefill prefill(PrefillChunk chunk) {
        auto hidden = target_forward({{1, chunk.count}});
        return {hidden, hidden, 1.0};
    }
    struct Head { Tensor sample_hidden, chain_hidden; };
    Head predictor_step(Tensor hidden, Tensor ids, Tensor) {
        mtp.position += size(ids, 1);
        hidden.proposal = true;
        return {hidden, hidden};
    }
    static Tensor logits_for(Tensor hidden) {
        hidden.shape.back() = 3;
        return hidden;
    }
    static int32_t sample_normal(Tensor logits, Tensor) { return logits.proposal ? 1 : 0; }
    static int32_t sample_constrained(Tensor logits, Tensor counts, const MfqTokenConstraintPtr&) {
        return sample_normal(logits, counts);
    }
    static std::vector<float> probabilities(Tensor logits, Tensor, const MfqSamplingParams&) {
        return logits.proposal ? std::vector<float>{0, 1, 0} : std::vector<float>{1, 0, 0};
    }
    static mtp::CompactDistribution compact_probabilities(Tensor logits, Tensor counts,
        const MfqSamplingParams& sampling) { return {{0, 1, 2}, probabilities(logits, counts, sampling)}; }
    static std::vector<mtp::CompactDistribution> compact_probability_rows(Tensor logits,
        const MfqSamplingParams& sampling) {
        return std::vector<mtp::CompactDistribution>(size(logits, 0), compact_probabilities(logits, {}, sampling));
    }
};

void run(bool dspark) {
    Model model;
    Predictor mtp;
    mtp.legacy = dspark;
    InferenceRequest request;
    request.chat = false;
    request.prompt = {0, 0, 0};
    request.sampling.max_tokens = 256;
    request.sampling.mtp_max_draft_tokens = 3;
    InferenceOutput output(request, nullptr, "mtp-test");
    Tensor last_hidden;
    auto generation = speculative_sequence(Ops{model, mtp}, request, output, 512, 0, Tensor{}, &last_hidden);
    std::vector<int64_t> tokens;
    while (auto step = generation.next()) {
        if (!step.value) continue;
        if (const auto* delta = std::get_if<TokenOutput>(&*step.value))
            tokens.insert(tokens.end(), delta->token_ids.begin(), delta->token_ids.end());
    }
    if (tokens != std::vector<int64_t>(256, 0) || model.cache_pos != 258 || !Ops::defined(last_hidden))
        throw std::runtime_error("shared MTP output/cache boundary mismatch");
    const auto& stats = output.metrics.mtp;
    if (dspark ? stats.park_count != 0 || stats.standard_tokens != 0 :
        stats.park_count == 0 || stats.standard_tokens == 0)
        throw std::runtime_error("shared MTP policy dispatch mismatch");
}

void run_chunked_prompt_history(bool streaming, int reused) {
    struct StreamOps : Ops {
        bool streaming;
        std::vector<int64_t>* boundaries;
        bool stream_prompt_history(int64_t) const { return streaming; }
        static Tensor slice(Tensor value, int axis, int64_t begin, int64_t length) {
            value=Ops::slice(value,axis,begin,length);value.start+=begin;return value;
        }
        Prefill prefill(PrefillChunk chunk) {
            auto result=Ops::prefill(chunk);result.raw.start=chunk.offset;return result;
        }
        double predictor_prefill(Tensor hidden, Tensor ids, Tensor positions) {
            if(hidden.start+1!=ids.start || size(hidden,1)!=size(ids,1))
                throw std::runtime_error("streamed prompt lost a hidden/input pair at a chunk or reused boundary");
            boundaries->push_back(model.cache_pos);
            (void)predictor_step(hidden,ids,positions);return 0;
        }
    };
    Model model;model.cache_pos=reused;
    Predictor mtp;mtp.position=reused?reused-1:0;
    InferenceRequest request;request.chat=false;request.prompt=std::vector<int64_t>(13,0);
    request.sampling.max_tokens=1;request.sampling.mtp_max_draft_tokens=3;
    InferenceOutput output(request,nullptr,"mtp-chunked-prompt-history");
    Tensor last_hidden,restored{{1,1,1},false,reused-1};
    std::vector<int64_t> boundaries;
    auto generation=speculative_sequence(StreamOps{Ops{model,mtp},streaming,&boundaries},
        request,output,4,reused,restored,&last_hidden);
    while(auto step=generation.next()) {}
    if(boundaries.empty() || (streaming?boundaries.front()>=13:boundaries.front()!=13) ||
        mtp.position!=12 || last_hidden.start!=12 || output.result.completion_tokens!=1)
        throw std::runtime_error("streamed prompt history changed the final cache boundary");
}

void run_prompt_snapshot_boundary() {
    Model model;Predictor mtp;
    InferenceRequest request;request.chat=false;request.prompt={0,0,0};
    request.sampling.max_tokens=8;request.sampling.mtp_max_draft_tokens=3;
    InferenceOutput output(request,nullptr,"mtp-snapshot-timing-test");
    Tensor last_hidden;
    auto generation=speculative_sequence(Ops{model,mtp},request,output,512,0,Tensor{},&last_hidden);
    bool saw_boundary=false;
    InferenceMetrics::Clock::time_point cache_finished;
    while(auto step=generation.next())if(step.value) {
        if(const auto* progress=std::get_if<PrefillProgress>(&*step.value);progress && progress->session_boundary) {
            if(saw_boundary || output.metrics.saw_token || model.cache_pos!=3 ||
                    mtp.cache_position()!=2 || !Ops::defined(last_hidden))
                throw std::runtime_error("MTP prompt snapshot boundary is incomplete");
            saw_boundary=true;
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            cache_finished=InferenceMetrics::Clock::now();
        }
        if(std::holds_alternative<TokenOutput>(*step.value) &&
                (!saw_boundary || output.metrics.first_token<cache_finished))
            throw std::runtime_error("initial snapshot entered decode timing");
    }
    if(!saw_boundary || output.result.completion_tokens!=8 || model.cache_pos!=10)
        throw std::runtime_error("prompt snapshot changed generation history");
}

void run_bootstrap_publication() {
    struct BootstrapOps : Ops {
        bool early;
        bool bootstrap_before_publication() const {return early;}
    };
    const auto run_bootstrap=[](bool early,int limit) {
        Model model;Predictor mtp;mtp.bootstrap=true;
        InferenceRequest request;request.chat=false;request.prompt={0,0,0};
        request.sampling.max_tokens=limit;request.sampling.mtp_max_draft_tokens=3;
        InferenceOutput output(request,nullptr,"mtp-bootstrap-publication-test");
        Tensor last_hidden;
        auto generation=speculative_sequence(BootstrapOps{{model,mtp},early},request,output,512,0,Tensor{},&last_hidden);
        std::vector<int64_t> tokens;bool boundary=false;
        while(auto step=generation.next())if(step.value) {
            if(const auto* progress=std::get_if<PrefillProgress>(&*step.value);progress && progress->session_boundary) {
                if(model.cache_pos!=3 || mtp.cache_position()!=2 || output.metrics.saw_token)
                    throw std::runtime_error("bootstrap changed prompt snapshot boundary");
                boundary=true;
            }
            if(const auto* delta=std::get_if<TokenOutput>(&*step.value)) {
                if(tokens.empty() && (!boundary || delta->token_ids.size()!=(early && limit>1?2:1)))
                    throw std::runtime_error("bootstrap publication emitted an incomplete first batch");
                tokens.insert(tokens.end(),delta->token_ids.begin(),delta->token_ids.end());
            }
        }
        if(tokens.size()!=std::size_t(limit) || model.cache_pos!=limit+2 || !Ops::defined(last_hidden))
            throw std::runtime_error("bootstrap publication changed output/cache length");
        return tokens;
    };
    if(run_bootstrap(false,8)!=run_bootstrap(true,8) || run_bootstrap(false,1)!=run_bootstrap(true,1))
        throw std::runtime_error("bootstrap publication changed sampled tokens");
}

void run_target_initialization() {
    struct InitializingOps : Ops {
        int* observations;
        bool take_target_maintenance() { ++*observations;return true; }
    };
    Model model;Predictor mtp;int observations=0;
    InferenceRequest request;request.chat=false;request.prompt={0,0,0};
    request.sampling.max_tokens=32;request.sampling.mtp_max_draft_tokens=3;
    InferenceOutput output(request,nullptr,"mtp-initialization-test");
    auto generation=speculative_sequence(InitializingOps{{model,mtp},&observations},request,output,512,0,Tensor{},nullptr);
    std::vector<int64_t> tokens;
    while(auto step=generation.next())if(step.value)
        if(const auto* delta=std::get_if<TokenOutput>(&*step.value))
            tokens.insert(tokens.end(),delta->token_ids.begin(),delta->token_ids.end());
    if(tokens!=std::vector<int64_t>(32,0) || model.cache_pos!=34 || observations==0)
        throw std::runtime_error("target initialization changed token/cache boundaries");
    for(double ms:output.metrics.mtp.measured_depth_ms)if(ms!=0)
        throw std::runtime_error("target initialization entered steady MTP timing");
}

void run_streaming_publication(bool accepted) {
    struct StreamingOps : Ops {
        int* predictor_calls;
        bool accepted;
        Head predictor_step(Tensor hidden,Tensor ids,Tensor positions) {
            ++*predictor_calls;return Ops::predictor_step(hidden,ids,positions);
        }
        std::vector<int32_t> greedy_tokens(Tensor logits) {
            return std::vector<int32_t>(size(logits,0),accepted?1:0);
        }
    };
    Model model;Predictor mtp;int calls=0;
    InferenceRequest request;request.chat=false;request.prompt={0,0,0};
    request.sampling.max_tokens=32;request.sampling.mtp_max_draft_tokens=3;
    InferenceOutput output(request,nullptr,"mtp-publication-test");
    Tensor last_hidden;
    auto generation=speculative_sequence(StreamingOps{{model,mtp},&calls,accepted},
        request,output,512,0,Tensor{},&last_hidden);
    int deltas=0;
    while(auto step=generation.next())if(step.value)
        if(const auto* delta=std::get_if<TokenOutput>(&*step.value);delta && !delta->token_ids.empty()) {
            if(++deltas==2)break;
        }
    // One prompt-priming call and three calls for the first draft. Computing
    // any part of the next speculative chain would delay this publication.
    if(deltas!=2 || calls!=4)
        throw std::runtime_error("completed MTP tokens waited for the next draft");
    output.result.cancelled=true;
    while(auto step=generation.next())if(step.value && std::holds_alternative<TokenOutput>(*step.value))
        throw std::runtime_error("cancelled MTP stream emitted another token batch");
    if(calls!=5 || mtp.cache_position()!=model.cache_pos || !Ops::defined(last_hidden))
        throw std::runtime_error("MTP cancellation failed to commit predictor history");
}

void run_history_batching() {
    struct HistoryOps : Ops {
        std::vector<int64_t>* widths;
        int* history_calls;
        bool gradual_mtp_warmup() const {return true;}
        std::size_t predictor_history_merge_limit() const {return 128;}
        std::size_t predictor_history_flush_threshold() const {return 4;}
        std::size_t predictor_parked_history_flush_threshold() const {return 16;}
        Head predictor_step(Tensor hidden,Tensor ids,Tensor positions) {
            if(size(hidden,1)!=size(ids,1))throw std::runtime_error("merged MTP history lost token alignment");
            widths->push_back(size(ids,1));
            return Ops::predictor_step(hidden,ids,positions);
        }
        void predictor_history_step(Tensor hidden,Tensor ids,Tensor positions) {
            if(size(hidden,1)!=size(ids,1))throw std::runtime_error("headless history lost token alignment");
            ++*history_calls;
            for(int64_t offset=0;offset<size(ids,1);offset+=4) {
                const auto rows=std::min<int64_t>(4,size(ids,1)-offset);
                widths->push_back(rows);
                (void)Ops::predictor_step(slice(hidden,1,offset,rows),slice(ids,1,offset,rows),positions);
            }
        }
        std::vector<int32_t> greedy_tokens(Tensor logits) {return std::vector<int32_t>(size(logits,0),1);}
    };
    Model model;Predictor mtp;std::vector<int64_t> widths;int history_calls=0;
    InferenceRequest request;request.chat=false;request.prompt={0,0,0};
    request.sampling.max_tokens=96;request.sampling.mtp_max_draft_tokens=3;
    InferenceOutput output(request,nullptr,"mtp-history-test");
    auto generation=speculative_sequence(HistoryOps{{model,mtp},&widths,&history_calls},request,output,512,0,Tensor{},nullptr);
    std::vector<int64_t> tokens;
    while(auto step=generation.next())if(step.value)
        if(const auto* delta=std::get_if<TokenOutput>(&*step.value))
            tokens.insert(tokens.end(),delta->token_ids.begin(),delta->token_ids.end());
    if(tokens.size()!=96 || tokens.front()!=0 || model.cache_pos!=98 || mtp.cache_position()!=98 ||
            output.metrics.mtp.depth_cycles[0]<3 || history_calls==0 ||
            std::find(widths.begin(),widths.end(),4)==widths.end() ||
            std::any_of(widths.begin(),widths.end(),[](auto n){return n>4;})) {
        std::cerr<<"history tokens="<<tokens.size()<<" model="<<model.cache_pos<<" predictor="<<mtp.cache_position()
            <<" baseline="<<output.metrics.mtp.depth_cycles[0]<<" widths=";
        for(auto n:widths)std::cerr<<n<<',';std::cerr<<std::endl;
        throw std::runtime_error("MTP history batching changed output/cache boundaries or left a re-entry burst");
    }
}

void run_batched_greedy(bool penalties) {
    struct BatchedOps : Ops {
        int* batches;
        bool use_penalties;
        void initialize(const std::vector<int64_t>& prompt, const MfqSamplingParams& sampling) {
            Ops::initialize(prompt, sampling);
            this->penalties = use_penalties;
        }
        std::vector<int32_t> greedy_tokens(Tensor logits) {
            ++*batches;
            return std::vector<int32_t>(size(logits, 0), 0);
        }
    };
    Model model; Predictor mtp; mtp.legacy = true; int batches = 0;
    InferenceRequest request; request.chat = false; request.prompt = {0, 0, 0};
    request.sampling.max_tokens = 32; request.sampling.mtp_max_draft_tokens = 3;
    InferenceOutput output(request, nullptr, "mtp-batched-greedy-test");
    auto generation = speculative_sequence(BatchedOps{{model, mtp}, &batches, penalties},
        request, output, 512, 0, Tensor{}, nullptr);
    std::vector<int64_t> tokens;
    while (auto step = generation.next()) if (step.value)
        if (const auto* delta = std::get_if<TokenOutput>(&*step.value))
            tokens.insert(tokens.end(), delta->token_ids.begin(), delta->token_ids.end());
    if (tokens != std::vector<int64_t>(32, 0) || model.cache_pos != 34 ||
        (penalties ? batches != 0 : batches == 0))
        throw std::runtime_error("batched greedy sampling changed rejection/cache/penalty behavior");
}

void batch_policy() {
    mtp::BatchDepthPolicy policy({"one", "two"}, 1);
    auto now = mtp::BatchDepthPolicy::Clock::time_point{};
    const std::array<int, 2> rejected{0, 0};
    const auto observe = [&](int depth, int milliseconds, bool stable = true, bool interrupted = false) {
        const auto started = now;
        now += std::chrono::milliseconds(milliseconds);
        policy.observe(depth, rejected, started, now, stable, interrupted);
    };
    observe(0, 500, false);
    observe(0, 500, true, true);
    if (policy.cost(0) || policy.depth()) throw std::runtime_error("interrupted batch polluted normal calibration");
    for (int index = 0; index < 5; ++index) observe(0, 10);
    if (policy.depth() != 1 || policy.cost(0) != 10)
        throw std::runtime_error("batch normal warmup/median mismatch");
    policy.interrupt();
    now += std::chrono::seconds(3);
    observe(0, 10);
    if (policy.cost(0) != 10) throw std::runtime_error("prefill gap entered batch timing");
    for (int index = 0; index < 30 && !policy.parks(); ++index) observe(1, 40);
    if (policy.depth() != 0 || policy.parks() != 1 || policy.remaining() != 128)
        throw std::runtime_error("losing batch MTP did not park");
    mtp::BatchParkMemory memory;
    memory.parked(policy);
    memory.tick();
    mtp::BatchDepthPolicy replacement({"three", "four"}, 1);
    memory.seed(replacement);
    if (replacement.remaining() != 127 || replacement.cooldown() != 256)
        throw std::runtime_error("batch membership reset park countdown");
    mtp::BatchDepthPolicy larger({"one", "two", "three"}, 1);
    memory.seed(larger);
    if (larger.remaining()) throw std::runtime_error("smaller-batch loss parked a larger cohort");
    mtp::BatchDepthPolicy smaller({"one"}, 1);
    memory.seed(smaller);
    if (smaller.remaining()) throw std::runtime_error("larger-batch loss parked a smaller cohort");
    mtp::BatchDepthPolicy mixed({"one", "two"}, 1, 1);
    memory.seed(mixed);
    if (mixed.remaining()) throw std::runtime_error("pure MTP loss parked a mixed cohort");
    mixed.start_parked(64);
    memory.parked(mixed);
    mtp::BatchDepthPolicy pure({"one", "two"}, 1);
    memory.seed(pure);
    if (pure.remaining() != 127) throw std::runtime_error("mixed MTP verdict replaced pure MTP verdict");
    mtp::BatchDepthPolicy deeper({"one", "two"}, 3);
    memory.seed(deeper);
    if (deeper.remaining()) throw std::runtime_error("different draft sweep inherited a park verdict");
    for (int index = 0; index < 127; ++index) memory.tick();
    mtp::BatchDepthPolicy expired({"one", "two"}, 1);
    memory.seed(expired);
    if (expired.remaining()) throw std::runtime_error("batch park verdict did not expire");
    for (int index = 0; index < 128; ++index) observe(0, 10);
    if (policy.remaining() || policy.depth() != 1 || policy.cost(1))
        throw std::runtime_error("batch MTP did not restart a clean depth sweep");
}
}

int main() {
    run(false);
    run(true);
    for(bool streaming:{false,true})for(int reused:{0,4})run_chunked_prompt_history(streaming,reused);
    run_prompt_snapshot_boundary();
    run_bootstrap_publication();
    run_target_initialization();
    run_streaming_publication(false);
    run_streaming_publication(true);
    run_history_batching();
    run_batched_greedy(false);
    run_batched_greedy(true);
    batch_policy();
    std::cout << "shared speculative sequence output/cache/dispatch tests passed\n";
}
