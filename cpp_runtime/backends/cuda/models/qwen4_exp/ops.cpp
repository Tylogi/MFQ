#include "../../runtime/execution_options.h"
#include "mfq_cuda_moe_ops.h"
#include "ops.h"
#include "model.h"
#include "mtp.h"
#include "storage/weight_loader.h"
#include "storage/moe_ffn_pipeline.h"
#include "storage/moe_expert_cache.h"
#include "runtime/nint_row_pipeline.h"
#include "runtime/decode_window.h"
#include "runtime/moe_pipeline.h"
#include "runtime/router_lookahead_audit.h"
#include <fstream>
#include <iostream>
#include "models/common/gated_mlp.h"
#include "models/common/moe.h"
#include "models/common/transformer_layer.h"
#include "storage/session_codec.h"
#include "gated_residual.h"
#include "gated_residual_fused.h"
#include "float_projection.h"
#include <cstdlib>

namespace mfq::cuda::qwen4_exp {
using Config = mfq::models::qwen4_exp::Config;

struct Gr {
    Tensor norm;
    weight_loader::ResidualLinear down, up, injection;
    int64_t hidden, streams;
    double eps;
    bool can_fuse(CudaExecutionContext& execution,const Tensor& x,const GatedResidualMarker& marker)const {
        if(!execution.config.gr_two_stage || marker || !execution.config.gr_fused_projections ||
            !execution.config.gr_native_projection_input || !down.weight || !up.weight || !injection.weight ||
            !down.weight->is_nint() || !up.weight->is_nint() || streams<1 || streams>4 ||
            x.dim()<1 || x.numel()!=hidden*streams || !x.is_cuda() || !x.is_contiguous() ||
            (x.scalar_type()!=tb::kFloat16 && x.scalar_type()!=tb::kFloat32) ||
            !norm.is_contiguous() || norm.scalar_type()!=tb::kFloat32)return false;
        if(!injection.weight->is_nint() && !(execution.config.gr_two_stage_dense_injection &&
            injection.weight->is_dense() && injection.weight->dense.is_contiguous() &&
            injection.weight->dense.scalar_type()==tb::kBFloat16))return false;
        const auto& w=up.weight->nint;
        return w.ng==1 || (w.out+3)/4>=nint_float_projection_resident_blocks(x.get_device());
    }
    std::vector<Tensor> fused(const Tensor& branch,const Tensor& residual,const Tensor& prior,bool after)const {
        const auto& projection=*injection.weight;
        return gated_residual_two_stage_cuda(branch,residual,prior,norm,down.weight->nint,up.weight->nint,
            projection.is_nint()?&projection.nint:nullptr,streams,eps,after,
            projection.is_dense()?projection.dense:Tensor{},injection.prepared_right);
    }
    std::vector<Tensor> pre(CudaExecutionContext &execution, const Tensor &x,
        const GatedResidualMarker& marker={}) const {
        if(can_fuse(execution,x,marker))return fused({},x,{},false);
        return gated_residual_pre_projected(execution, x, norm, down.forward, up.forward, injection.forward,
            hidden, streams, eps,up.mixed,marker,down.activated,injection.activated);
    }
    Tensor post(const Tensor &branch, const std::vector<Tensor> &inputs) const {
        return mfq_qwen4_exp::gated_residual_post(branch, inputs[1], inputs[2], streams);
    }
    std::vector<Tensor> pre_after(CudaExecutionContext& execution,const Tensor& branch,
        const std::vector<Tensor>& inputs,const GatedResidualMarker& marker={}) const {
        if(can_fuse(execution,inputs[1],marker) && branch.is_contiguous() && inputs[2].is_contiguous() &&
            (branch.scalar_type()==tb::kFloat16 || branch.scalar_type()==tb::kFloat32) &&
            (inputs[2].scalar_type()==tb::kFloat16 || inputs[2].scalar_type()==tb::kFloat32))
            return fused(branch,inputs[1],inputs[2],true);
        return gated_residual_pre_after_projected(execution,branch,inputs[1],inputs[2],norm,
            down.forward,up.forward,injection.forward,hidden,streams,eps,
            up.mixed,marker,down.activated,injection.activated);
    }
};

static Linear qwen_moe(weight_loader::Routed gate_up, weight_loader::Routed down,
                       Linear router, Linear shared_gate, Linear sg, Linear su, Linear sd,
                       const Config &c, MoeFfnPrefetch* prefetch=nullptr) {
    auto shared=[sg,su,sd,shared_gate](CudaExecutionContext& execution,const Tensor& source) {
        auto unfused=[](const auto&...){return std::optional<Tensor>{};};
        auto output=mfq::models::gated_mlp(source,false,0.0,unfused,unfused,
            [&](Tensor input){return std::array<Tensor,2>{sg(execution,input),su(execution,input)};},
            [](Tensor g,Tensor u,mfq::models::GatedActivation,double){return (g*tb::sigmoid(g))*u;},
            [&](Tensor hidden){return sd(execution,hidden);},unfused);
        return tb::sigmoid(shared_gate(execution,source))*output;
    };
    MoeFfnSharedWeights shared_weights;
    shared_weights.projections={weight_loader::linear_weight(sg),weight_loader::linear_weight(su),weight_loader::linear_weight(sd)};
    shared_weights.gate=[shared_gate](CudaExecutionContext& execution,const Tensor& input){return tb::sigmoid(shared_gate(execution,input));};
    auto pipeline=make_moe_ffn_pipeline(gate_up.projections,down.projections,shared,{},std::move(shared_weights),prefetch);
    return [gate_up, down, router, shared, pipeline, c](CudaExecutionContext &execution,
                                                               const Tensor &x) {
        auto source = x.reshape({-1, c.hidden}).to(tb::kFloat16);
        const auto routing = c.routing();
        return mfq::models::mixture_of_experts(
            [&] { return router(execution, source).contiguous(); },
            [&](Tensor logits) {
                return moe_topk_cuda(
                    logits, c.topk, routing.activation == mfq::models::RouterActivation::sigmoid,
                    routing.activation == mfq::models::RouterActivation::sqrt_softplus,
                    routing.normalize, routing.delayed_softmax, mfq_nullopt, 1e-20, routing.scale);
            },
            [&](const auto &) {
                return pipeline ? Tensor{} : shared(execution,source);
            },
            [&](const auto &selected) {
                if(pipeline)return pipeline(execution,source,selected[0].to(tb::kInt32).contiguous(),selected[1]);
                return mfq::models::routed_experts(
                    [&] { return gate_up(execution, source, selected[0]); },
                    [&](Tensor gu) {
                        auto gate = gu.narrow(-1, 0, c.moe_width),
                             up = gu.narrow(-1, c.moe_width, c.moe_width);
                        return (gate * tb::sigmoid(gate)) * up;
                    },
                    [&](Tensor hidden) { return down(execution, hidden, selected[0]); },
                    [&](Tensor pairs) {
                        auto reduced = tb::zeros({source.size(0), c.hidden},
                                                 source.options().dtype(tb::kFloat32));
                        for (int64_t r = 0; r < c.topk; ++r)
                            reduced = reduced + pairs.select(1, r).to(tb::kFloat32) *
                                                    selected[1].select(1, r).unsqueeze(-1);
                        return reduced.to(pairs.scalar_type());
                    });
            },
            [&](Tensor routed, Tensor shared_output, const auto &) {
                if(pipeline)return routed.reshape(x.sizes());
                return (routed + shared_output).reshape(x.sizes());
            });
    };
}

struct BlockLoader : weight_loader::Loader {
    using GdnWeights = qwen4_exp::GdnWeights;
    using QsaWeights = qwen4_exp::QsaWeights;
    using PleWeights = qwen4_exp::PleWeights;
    Linear* router_capture=nullptr;
    MoeFfnPrefetch* prefetch_capture=nullptr;
    struct Embedding {
        attention_ops::Embedding function;
        std::shared_ptr<mfq::NintRows> range;
        Tensor operator()(const Tensor& ids) const { return function(ids); }
    };

    auto residual_linear(const std::string &name) const {
        return weight_loader::residual_linear(execution, source, name);
    }
    static Gr residual(Tensor norm, weight_loader::ResidualLinear down,
        weight_loader::ResidualLinear up, weight_loader::ResidualLinear injection, const Config &c) {
        return {std::move(norm), std::move(down), std::move(up), std::move(injection),
                c.hidden, c.streams, c.eps};
    }
    static auto final_mixer(Gr weights) { return std::make_unique<Gr>(std::move(weights)); }
    auto gdn(GdnWeights weights, const Config &c) const {
        std::vector<Linear> projections{weights.qkv,weights.gate,weights.alpha,weights.beta};
        weights.input_projection=weight_loader::grouped_linear(execution,projections);
        weights.qkv=std::move(projections[0]);weights.gate=std::move(projections[1]);
        weights.alpha=std::move(projections[2]);weights.beta=std::move(projections[3]);
        return std::make_unique<Gdn>(std::move(weights), c.key_heads, c.value_heads, c.linear_width,
                                     c.kernel, c.eps, c.silu_gate,true,execution.config.gdn_transposed_state,
                                     execution.config.gdn_fused_output,true,execution.config.gdn_fused_preparation,
                                     execution.config.gdn_fused_core);
    }
    auto qsa(QsaWeights weights, const Config &c) const {
        std::vector<Linear> projections{weights.query,weights.key,weights.value,weights.index_query_key};
        weights.input_projection=weight_loader::grouped_linear(execution,projections);
        weights.query=std::move(projections[0]);weights.key=std::move(projections[1]);
        weights.value=std::move(projections[2]);weights.index_query_key=std::move(projections[3]);
        QsaConfig geometry{c.heads, c.kv_heads, c.width, c.index_heads, c.index_width,
                           c.pool, c.budget, c.maximum, c.eps};
        auto rotary = std::make_shared<RotaryEmbedding>(c.rotary, c.maximum, c.rope_base, c.sections, c.interleaved);
        return std::make_unique<Qsa>(std::move(weights), geometry, std::move(rotary));
    }
    auto moe(weight_loader::Routed gate_up, weight_loader::Routed down, Linear router,
                    Linear shared_gate, Linear sg, Linear su, Linear sd, const Config &c) const {
        if(router_capture)*router_capture=router;
        return qwen_moe(std::move(gate_up), std::move(down), std::move(router), std::move(shared_gate),
                        std::move(sg), std::move(su), std::move(sd), c, prefetch_capture);
    }
    auto embedding(const std::string &name) const {
        if (require_tensor(source, name).dtype == "NINT") {
            auto table = load_nint_row_table(source, name);
            Embedding lookup{[table](const Tensor &ids) {
                return nint_row_embedding_lookup(*table, ids);
            },table};
            return std::pair{std::move(lookup), std::array<int64_t, 2>{table->rows(), table->width()}};
        }
        auto weight = std::make_shared<QuantLinear>(load_quant_linear(execution, source, name));
        Embedding lookup{[weight](const Tensor &ids) { return quant_embedding_lookup(*weight, ids); },{}};
        return std::pair{std::move(lookup), std::array<int64_t, 2>{weight->out(), weight->neuron_len()}};
    }
    std::vector<int64_t> integers(const std::string &name) const {
        const auto &type = require_tensor(source, name).dtype;
        MFQ_RUNTIME_CHECK(type == "I64" || type == "I32",
                          "Qwen4 PLE metadata must be a dense integer array: ", name);
        auto host = load_dense_gpu(execution, source, name).to(tb::kInt64).contiguous().cpu();
        MFQ_RUNTIME_CHECK(host.dim() == 1, "Qwen4 PLE metadata must be a vector");
        return {host.data_ptr<int64_t>(), host.data_ptr<int64_t>() + host.numel()};
    }
    static auto ple(std::vector<Embedding> shards, int64_t rows, int64_t width, const Config &c,
                    std::vector<int64_t> multipliers, std::vector<int64_t> offsets,
                    std::vector<int64_t> sizes, PleWeights weights) {
        std::vector<attention_ops::Embedding> functions;std::vector<std::shared_ptr<mfq::NintRows>> ranges;
        for(auto& shard:shards){functions.push_back(std::move(shard.function));ranges.push_back(std::move(shard.range));}
        std::shared_ptr<NintRowPipeline> pipeline;
        if(std::all_of(ranges.begin(),ranges.end(),[](const auto& p){return bool(p);})) {
            pipeline=std::make_shared<NintRowPipeline>(std::move(ranges));
        }
        NgramEmbedding embedding(std::move(functions), rows, width, c.ngram, c.ngram_heads, c.eos,
                                 std::move(multipliers), std::move(offsets), std::move(sizes),std::move(pipeline));
        return std::make_unique<Ple>(std::move(embedding), std::move(weights), c.hidden, c.streams,
                                     c.ngram, c.eps);
    }
    std::unique_ptr<Qwen4Block> block(const Config &c, int layer, bool predictor);
};

struct Qwen4Block final : Block {
    using Tensor = mfq_tensor_backend::Tensor;
    mfq::models::qwen4_exp::Config config;
    Gr attention_gr, ffn_gr;
    Linear ffn,router;
    Linear next_router;
    MoeFfnPrefetch prefetch,next_prefetch;
    std::shared_ptr<RouterLookaheadInputs> router_audit;
    std::unique_ptr<Gdn> gdn;
    std::unique_ptr<Qsa> qsa;
    std::unique_ptr<Ple> ple;
    Qwen4Block(CudaExecutionContext &execution, const mfq::ModelSource &file,
               const Config &c, int layer, bool predictor = false) {
        BlockLoader loader{{execution, file, "qwen4_exp"},&router,&prefetch};
        mfq::models::qwen4_exp::load_block(*this, loader, c, layer, predictor);
    }
    void reset(int64_t) override {
        if (gdn)
            gdn->reset();
        if (qsa)
            qsa->reset();
        if (ple)
            ple->reset();
    }
    bool supports_speculation() const noexcept override { return true; }
    void prefetch_token_ids(const Tensor& ids) override { if(ple)ple->prefetch_tokens(ids); }
    void commit_speculative() override {
        if (gdn)
            gdn->commit();
        if (ple)
            ple->commit();
    }
    void rollback_speculative(int64_t keep) override {
        if (gdn)
            gdn->rollback();
        if (qsa)
            qsa->truncate(keep);
        if (ple)
            ple->rollback();
    }
    Tensor execute(CudaExecutionContext &execution, Tensor x, const Tensor &ids,
                   const Tensor &positions, const Tensor &full_positions, int64_t confirmed = 0,const Tensor& host_ids = {}) {
        return mfq::models::qwen4_exp::decoder_layer_chained(
            std::move(x), bool(ple), bool(gdn),
            [&](const Tensor &hidden) {
                return ple->forward(execution, hidden, ids, true, confirmed,host_ids);
            },
            [](Tensor hidden, Tensor positional) { return hidden + positional; },
            [&](const Tensor &hidden) { return attention_gr.pre(execution, hidden); },
            [&](Tensor branch) { return gdn->forward(execution, branch, true, confirmed); },
            [&](Tensor branch) {
                return qsa->forward(execution, branch, positions, full_positions, true);
            },
            [&](Tensor branch, const auto &mix) { return ffn_gr.pre_after(execution, branch, mix); },
            [&](Tensor branch) { return ffn(execution, branch); },
            [&](Tensor branch, const auto &mix) { return ffn_gr.post(branch, mix); });
    }
    Tensor forward(CudaExecutionContext &, Tensor, Tensor, int64_t, const MfqOptional<Tensor> &,
                   const RopeCache &, const MfqOptional<Tensor> & = mfq_nullopt,
                   const MfqOptional<Tensor> & = mfq_nullopt) override {
        throw std::runtime_error(
            "Qwen4 block requires the unified model position/PLE input lifecycle");
    }
    Tensor forward_context(CudaExecutionContext &execution, Tensor x, const Context &context,
                           const RopeCache &) override {
        return execute(execution, std::move(x), context.token_ids, context.positions,
                       context.full_positions, context.confirmed_prefix,context.host_token_ids);
    }
    std::vector<Tensor*> graph_warmup_state() override {
        std::vector<Tensor*> result;
        if(gdn)for(auto* value:gdn->graph_state())if(value->defined())result.push_back(value);
        if(ple)for(auto* value:ple->graph_state())if(value->defined())result.push_back(value);
        return result;
    }
    Tensor execute_graph(CudaExecutionContext& execution,Tensor x,const Tensor& ids,
                         const Tensor& positions,const Tensor& cache_positions,const Tensor& ple_rows,int layer) {
        const auto mark=[&](const char* phase){if(auto* window=DecodeWindow::recording())window->mark(layer,phase);};
        const bool detailed=mfq::cuda::runtime_options::mhc_timings();
        const auto marker=[&](const char* role)->GatedResidualMarker {
            if(!detailed || !DecodeWindow::recording())return {};
            return [&,role](const char* phase) {
                const auto name=std::string(role)+phase;mark(name.c_str());
            };
        };
        return mfq::models::qwen4_exp::decoder_layer_chained(std::move(x),bool(ple),bool(gdn),
            [&](const Tensor& hidden){return ple->forward_chunk(execution,hidden,ids,true,{},ple_rows);},
            [](Tensor hidden,Tensor positional){return hidden+positional;},
            [&](const Tensor& hidden){auto result=attention_gr.pre(execution,hidden,marker("attention_mhc_"));mark("attention_core_begin");return result;},
            [&](Tensor branch){auto result=gdn->forward_chunk(execution,branch,true);mark("attention_core_end");return result;},
            [&](Tensor branch){auto result=qsa->forward_graph(execution,branch,positions,cache_positions);mark("attention_core_end");return result;},
            [&](Tensor branch,const auto& mix){auto result=ffn_gr.pre_after(execution,branch,mix,marker("ffn_mhc_"));mark("ffn_pre_end");return result;},
            [&](Tensor branch) {
                DecodeWindow::Task prediction{};
                if(next_prefetch && branch.numel()==config.hidden) {
                    const auto routing=config.routing();
                    auto logits=next_router(execution,branch.reshape({1,config.hidden}).to(tb::kFloat16)).to(tb::kFloat32).contiguous();
                    auto selected=moe_topk_cuda(logits,2,
                        routing.activation==mfq::models::RouterActivation::sigmoid,
                        routing.activation==mfq::models::RouterActivation::sqrt_softplus,
                        routing.normalize,routing.delayed_softmax,mfq_nullopt,1e-20,routing.scale);
                    prediction=next_prefetch(selected[0].to(tb::kInt32).contiguous());
                }
                if(router_audit)router_audit->input(layer,branch);
                auto result=ffn(execution,branch);
                // Reserve current-layer readers before selecting speculative
                // slots. The route export itself precedes current FFN math.
                if(prediction.identity)DecodeWindow::recording()->enroll(std::move(prediction));
                return result;
            },
            [&](Tensor branch,const auto& mix){return ffn_gr.post(branch,mix);});
    }
};

std::unique_ptr<Qwen4Block> BlockLoader::block(const Config &c, int layer, bool predictor) {
    return std::make_unique<Qwen4Block>(execution, source, c, layer, predictor);
}

Qwen4ExpMtp::Qwen4ExpMtp() = default;
Qwen4ExpMtp::~Qwen4ExpMtp() = default;
Qwen4ExpMtp::Qwen4ExpMtp(Qwen4ExpMtp&&) noexcept = default;
Qwen4ExpMtp& Qwen4ExpMtp::operator=(Qwen4ExpMtp&&) noexcept = default;

std::optional<Qwen4ExpMtp> Qwen4ExpMtp::load_if_present(CudaExecutionContext &execution,
                                                      const mfq::ModelSource &file,
                                                      const mfq::models::qwen4_exp::Config &main) {
    Qwen4ExpMtp result;
    result.execution = &execution;
    BlockLoader loader{{execution, file, "qwen4_exp"}};
    if (!mfq::models::qwen4_exp::load_predictor(result, loader, main))
        return std::nullopt;
    return result;
}

void Qwen4ExpMtp::reset(int64_t next_batch) {
    MFQ_RUNTIME_CHECK(next_batch > 0, "Qwen4-Exp MTP batch must be positive");
    for (auto &layer : layers)
        layer->reset(next_batch);
    for (auto &pos : positions)
        pos = Tensor();
    std::fill(lengths.begin(), lengths.end(), 0);
    batch = next_batch;
}

std::pair<Tensor, Tensor> Qwen4ExpMtp::evaluate(const MtpTarget &target, const Tensor &hidden,
                                       const Tensor &ids, int64_t depth, bool cache,
                                       const Tensor &supplied_positions,
                                       const Tensor &supplied_embeddings) {
    namespace tb = mfq_tensor_backend;
    MFQ_RUNTIME_CHECK(ids.dim() == 2 && ids.size(0) > 0 && ids.size(1) > 0 &&
                          hidden.dim() == 3 && hidden.size(0) == ids.size(0) &&
                          hidden.size(1) == ids.size(1) &&
                          hidden.size(2) == hidden_norm.numel(),
                      "Qwen4-Exp MTP input geometry mismatch");
    const auto b = ids.size(0), t = ids.size(1);
    return mfq::models::projected_predictor(
        *this, b, t, depth, cache, [&](int64_t next) { reset(next); },
        [&] {
            auto embeds =
                supplied_embeddings.defined() ? supplied_embeddings : target.embed(ids);
            MFQ_RUNTIME_CHECK(embeds.sizes().vec() ==
                                  std::vector<int64_t>({b, t, config.hidden}),
                              "Qwen4-Exp MTP embedding shape mismatch");
            return embeds;
        },
        [&](mfq::models::PredictorCursor cursor) {
            auto current = supplied_positions.defined()
                               ? supplied_positions
                               : tb::arange(cursor.start, cursor.start + t,
                                            ids.options().dtype(tb::kInt32));
            if (current.dim() == 1)
                current = current.reshape({1, 1, t}).expand({3, 1, t}).contiguous();
            else if (current.dim() == 2)
                current = current.unsqueeze(1);
            if (current.dim() == 3 && current.size(0) == 4)
                current = current.narrow(0, 1, 3);
            MFQ_RUNTIME_CHECK(current.dim() == 3 && current.size(0) == 3 &&
                                  current.size(-1) == t &&
                                  (current.size(1) == 1 || current.size(1) == b),
                              "Qwen4 MTP positions require [T]/[3,T]/[3,B,T]");
            current = current.to(tb::kInt32).expand({3, b, t}).contiguous();
            auto full = cache && positions[cursor.layer].defined()
                            ? tb::cat({positions[cursor.layer], current}, -1)
                            : current;
            return std::array<Tensor, 2>{current, full};
        },
        [&](Tensor embeds, const auto &pos) {
            return embedding_fusion(*execution,
                                    rms_norm(embeds, embedding_norm + 1, config.eps));
        },
        [&] {
            return hidden_fusion(*execution,
                                 rms_norm(hidden, hidden_norm + 1, config.eps)
                                     .reshape({b, t, config.streams, config.hidden}));
        },
        [&](Tensor e, Tensor streams) {
            return (streams + e.unsqueeze(-2)).reshape({b, t, config.streams * config.hidden});
        },
        [&](Tensor x, int64_t layer, const auto &pos) {
            auto &block = *layers[layer];
            return mfq::models::qwen4_exp::decoder_layer(
                std::move(x), false, false, [](const Tensor &) { return Tensor{}; },
                [](Tensor value, Tensor) { return value; },
                [&](const Tensor &value) { return block.attention_gr.pre(*execution, value); },
                [](Tensor) -> Tensor { throw std::logic_error("Qwen4 MTP requires QSA"); },
                [&](Tensor branch) {
                    return block.qsa->forward(*execution, branch, pos[0], pos[1], cache);
                },
                [&](Tensor branch, const auto &mix) {
                    return block.attention_gr.post(branch, mix);
                },
                [&](const Tensor &value) { return block.ffn_gr.pre(*execution, value); },
                [&](Tensor branch) { return block.ffn(*execution, branch); },
                [&](Tensor branch, const auto &mix) { return block.ffn_gr.post(branch, mix); });
        },
        [&](const Tensor &multi) { return final_mixer->pre(*execution, multi)[0]; },
        [&](int64_t layer, const auto &pos) { positions[layer] = pos[1]; });
}

Tensor Qwen4ExpMtp::forward(const MtpTarget &target, Tensor hidden, Tensor ids) {
    return evaluate(target, hidden, ids).first;
}

MtpStep Qwen4ExpMtp::step(const MtpTarget &target, Tensor hidden, Tensor ids) {
    auto result = evaluate(target, hidden, ids);
    return {std::move(result.first), std::move(result.second)};
}

int64_t Qwen4ExpMtp::cache_position() const noexcept {
    return lengths.empty() ? 0 : lengths.front();
}

void Qwen4ExpMtp::trim_cache_to(int64_t position) {
    MFQ_RUNTIME_CHECK(position >= 0 && position <= cache_position(),
                      "Qwen4-Exp MTP cache trim position is invalid");
    constexpr size_t index = 0;
    if (index < layers.size() && layers[index]->qsa)
        layers[index]->qsa->truncate(position);
    if (positions[index].defined())
        positions[index] = positions[index].narrow(-1, 0, position);
    lengths[index] = position;
}

bool Qwen4ExpMtp::teacher_forced_prompt_prime() const noexcept { return true; }

bool Qwen4ExpMtp::target_bootstrap_decode() const noexcept { return true; }

bool Qwen4ExpMtp::preserve_output_dtype() const noexcept { return true; }

} // namespace mfq::cuda::qwen4_exp

namespace mfq::cuda {
struct Qwen4DecodeGraph {
    struct PleStage {
        qwen4_exp::Ple* ple;
        mfq_tensor_backend::Tensor rows;
        std::shared_ptr<NintRowStage> transfer;
        HostBuffer flags{128,true};
        uint32_t* device=nullptr;
        PleStage(qwen4_exp::Ple* p,int tokens,int hidden):ple(p) {
            charge_tensor_host_bytes(128);
            MFQ_CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&device),flags.data(),0));
            std::memset(flags.data(),0,128);publish_mapped_flag(host()+16);
            transfer=ple->graph_stage(tokens,mfq_tensor_backend::Device{mfq_tensor_backend::kCUDA,0});
            rows=transfer->output().reshape({1,tokens,hidden});
        }
        ~PleStage(){tensor_host_bytes.fetch_sub(128);}
        uint32_t* host(){return static_cast<uint32_t*>(flags.data());}
    };
    struct Cell {
        mfq_tensor_backend::Tensor ids,positions,cache_positions,host_ids,output;
        std::vector<std::unique_ptr<PleStage>> ple;
        DecodeWindow window;
        Cell(int tokens,int hidden,const std::vector<std::unique_ptr<Block>>& blocks,cudaStream_t stream)
            :window(stream,mfq::cuda::runtime_options::layer_timings()) {
            namespace tb=mfq_tensor_backend;
            const auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kInt64);
            ids=tb::zeros({1,tokens},options);positions=tb::zeros({tokens},options);cache_positions=tb::zeros({tokens},options);
            host_ids=tb::zeros({1,tokens},options.device(tb::kCPU));
            for(const auto& block:blocks) {
                auto& b=static_cast<qwen4_exp::Qwen4Block&>(*block);
                if(b.ple)ple.push_back(std::make_unique<PleStage>(b.ple.get(),tokens,hidden));
            }
        }
    };
    MfqCudaStream stream,ple_stream;
    std::map<std::pair<int,int>,std::unique_ptr<Cell>> cells;
    std::vector<qwen4_exp::Qsa*> qsa;
    bool failed=false;
    explicit Qwen4DecodeGraph(Qwen4CausalLm& model):stream(mfq_get_current_cuda_stream()),ple_stream(mfq_get_stream_from_pool()) {
        const bool enabled=model.router_lookahead.value_or(mfq::cuda::runtime_options::router_lookahead());
        for(std::size_t i=0;i<model.blocks.size();++i) {
            auto& current=static_cast<qwen4_exp::Qwen4Block&>(*model.blocks[i]);
            current.next_router={};current.next_prefetch={};
            if(enabled && i+1<model.blocks.size()) {
                auto& next=static_cast<qwen4_exp::Qwen4Block&>(*model.blocks[i+1]);
                MFQ_RUNTIME_CHECK(next.router && next.prefetch,"router lookahead requires the tiered FFN pipeline");
                current.next_router=next.router;current.next_prefetch=next.prefetch;
            }
        }
        for(const auto& block:model.blocks)if(auto* attention=static_cast<qwen4_exp::Qwen4Block&>(*block).qsa.get())qsa.push_back(attention);
    }
    ~Qwen4DecodeGraph() {(void)cudaStreamSynchronize(ple_stream.stream());cells.clear();}
    void restore_legacy() {cells.clear();for(auto* attention:qsa)attention->leave_graph();}

    mfq_tensor_backend::Tensor run(Qwen4CausalLm& model,mfq_tensor_backend::Tensor ids,int kind) {
        namespace tb=mfq_tensor_backend;
        if(failed)throw std::runtime_error("MFQ model session requires recovery after its previous error");
        if(stream.stream()!=mfq_current_cuda_stream())throw std::runtime_error("MFQ model window changed execution stream");
        const int tokens=static_cast<int>(ids.size(1));
        auto& owner=cells[{tokens,kind}];
        if(!owner)owner=std::make_unique<Cell>(tokens,model.config.hidden,model.blocks,stream.stream());
        auto& cell=*owner;
        cell.host_ids.copy_(ids.to(tb::kCPU,tb::kInt64));cell.ids.copy_(ids);
        cell.cache_positions.copy_(tb::arange(model.cache_pos,model.cache_pos+tokens,ids.options().dtype(tb::kInt64)));
        cell.positions.copy_(cell.cache_positions+model.decode_position_delta);
        try {
            finish_moe_expert_exchanges(model.execution->moe_expert_cache);
            if(!cell.window.valid()) {
                DecodeGraphBranchScope branch_scope(
                    model.execution->decode_graph_serial_branches);
                for(const auto& block:model.blocks) {
                    auto& b=static_cast<qwen4_exp::Qwen4Block&>(*block);
                    if(b.qsa)b.qsa->prepare_graph(model.positions);
                }
                struct State {tb::Tensor* target;tb::Tensor saved;const void* address;};
                std::vector<State> states;
                for(const auto& block:model.blocks)for(auto* state:block->graph_warmup_state())
                    states.push_back({state,state->clone(),state->data_ptr()});
                auto restore=[&] {
                    for(const auto& state:states) {
                        MFQ_RUNTIME_CHECK(state.target->data_ptr()==state.address,"MFQ capture changed recurrent storage");
                        state.target->copy_(state.saved);
                    }
                };
                cell.window.capture([&] {
                    cell.window.mark(-1,"window_begin");
                    auto hidden=model.adapter_prepare_hidden(model.embed_tokens(cell.ids),1,tokens);
                    std::size_t ple_index=0;int layer=0;
                    for(const auto& block:model.blocks) {
                        cell.window.mark(layer,"block_begin");
                        auto& b=static_cast<qwen4_exp::Qwen4Block&>(*block);tb::Tensor rows;
                        if(b.ple) {
                            auto* stage=cell.ple[ple_index++].get();rows=stage->rows;
                            cell.window.enroll({stage,
                                [&cell,stage] {std::memset(stage->flags.data(),0,128);stage->ple->prefetch_tokens(cell.host_ids);},
                                [this,&cell,stage] {
                                    wait_route_publication(stage->host(),stream.stream());
                                    MfqCudaStreamGuard guard(ple_stream);
                                    stage->ple->upload_graph(cell.host_ids,*stage->transfer);
                                    MFQ_CUDA_CHECK(cudaLaunchHostFunc(ple_stream.stream(),[](void* p){publish_mapped_flag(static_cast<uint32_t*>(p));},stage->host()+16));
                                    check_stream_progress(ple_stream.stream());
                                },{},[stage]{publish_mapped_flag(stage->host()+16);}});
                            signal_mapped_flag(stage->device,stream.stream());wait_mapped_flag(stage->device+16,stream.stream());
                            stage->transfer->decode();
                        }
                        cell.window.mark(layer,"attention_begin");
                        hidden=b.execute_graph(*model.execution,std::move(hidden),cell.ids,cell.positions,cell.cache_positions,rows,layer);
                        cell.window.mark(layer++,"block_end");
                    }
                    hidden=model.adapter_finalize_hidden(std::move(hidden),model.output_norm,1,tokens);
                    if(kind==2)cell.output=model.adapter_next_token(model.lm_head,model.last_hidden(std::move(hidden)));
                    else if(kind==1)cell.output=model.adapter_last_logits(model.lm_head,model.last_hidden(std::move(hidden)));
                    else cell.output=model.logits_from_hidden(std::move(hidden));
                    cell.window.mark(-1,"window_end");
                    auto audit=static_cast<qwen4_exp::Qwen4Block&>(*model.blocks.front()).router_audit;
                    if(audit)cell.window.enroll({audit.get(),[]{},[]{},[audit]{audit->finish_sample();},{}});
                },restore,[&]{cell.output={};});
            }
            cell.window.run();
            for(const auto& block:model.blocks) {
                auto& b=static_cast<qwen4_exp::Qwen4Block&>(*block);if(b.qsa)b.qsa->advance_graph(tokens);
            }
            model.cache_pos+=tokens;
            model.positions=tb::cat({model.positions,cell.positions},-1);model.batch=1;
            return cell.output;
        } catch(...) {failed=true;throw;}
    }
};

std::optional<mfq_tensor_backend::Tensor> qwen4_decode_graph_logits(Qwen4CausalLm& model,mfq_tensor_backend::Tensor ids,int kind) {
    if(!model.execution->config.moe_pipeline || !model.execution->moe_expert_cache || model.cache_pos==0 || model.speculative_start>=0)return {};
    if(ids.dim()==1)ids=ids.unsqueeze(0);
    if(ids.dim()!=2 || ids.size(0)!=1 || ids.size(1)<1 || ids.size(1)>8 || model.positions.dim()!=1)return {};
    MFQ_RUNTIME_CHECK(model.cache_pos+ids.size(1)<=model.max_position_embeddings(),"MFQ window exceeds context");
    if(!model.graph_registry)model.graph_registry=std::make_unique<Qwen4DecodeGraph>(model);
    return model.graph_registry->run(model,model.device_ids(std::move(ids)),kind);
}

void qwen4_set_rotary_fusion(Qwen4CausalLm& model, bool enabled) {
    MFQ_RUNTIME_CHECK(!model.graph_registry && model.cache_pos == 0,
                      "reset the model before switching rotary execution");
    mfq_cuda_synchronize();
    for (const auto& block : model.blocks) {
        auto* attention = static_cast<qwen4_exp::Qwen4Block&>(*block).qsa.get();
        if (attention) attention->set_rotary_fused(enabled);
    }
}

void qwen4_set_attention_grouping(Qwen4CausalLm& model, bool enabled) {
    MFQ_RUNTIME_CHECK(!model.graph_registry && model.cache_pos == 0,
                      "reset the model before switching attention projection groups");
    mfq_cuda_synchronize();
    for (const auto& block : model.blocks) {
        auto& attention = static_cast<qwen4_exp::Qwen4Block&>(*block);
        if (attention.gdn) attention.gdn->set_grouped_projection(enabled);
        if (attention.qsa) attention.qsa->set_grouped_projection(enabled);
    }
}

void qwen4_begin_router_lookahead_audit(Qwen4CausalLm& model,int samples) {
    if(model.blocks.size()<2 || model.graph_registry || samples<=0)
        throw std::invalid_argument("router lookahead audit requires a fresh multilayer decode session");
    auto audit=std::make_shared<RouterLookaheadInputs>(static_cast<int>(model.blocks.size()),
        static_cast<int>(model.config.hidden),samples);
    for(const auto& block:model.blocks) {
        auto& b=static_cast<qwen4_exp::Qwen4Block&>(*block);
        if(!b.router)throw std::runtime_error("router lookahead audit has no retained router");
        b.router_audit=audit;
    }
}
void qwen4_finish_router_lookahead_audit(Qwen4CausalLm& model,const std::string& prefix) {
    auto audit=static_cast<qwen4_exp::Qwen4Block&>(*model.blocks.front()).router_audit;
    if(!audit || !audit->samples())throw std::runtime_error("router lookahead audit captured no decode inputs");
    std::ofstream file(prefix+"-logits.f32",std::ios::binary);
    if(!file)throw std::runtime_error("cannot write router lookahead audit logits");
    for(std::size_t layer=1;layer<model.blocks.size();++layer) {
        auto& b=static_cast<qwen4_exp::Qwen4Block&>(*model.blocks[layer]);
        for(int input_layer:{static_cast<int>(layer)-1,static_cast<int>(layer)}) {
            auto input=audit->rows(input_layer).to(mfq_tensor_backend::kCUDA);
            auto logits=b.router(*model.execution,input).to(mfq_tensor_backend::kFloat32).cpu().contiguous();
            if(logits.dim()!=2 || logits.size(0)!=audit->samples() || logits.size(1)!=model.config.experts)
                throw std::runtime_error("router lookahead audit router shape mismatch");
            file.write(reinterpret_cast<const char*>(logits.data_ptr()),logits.numel()*logits.element_size());
            if(!file)throw std::runtime_error("router lookahead audit logits write failed");
        }
    }
    std::cout<<"router_lookahead_audit_samples="<<audit->samples()<<" layers="<<model.blocks.size()
        <<" hidden="<<model.config.hidden<<" experts="<<model.config.experts<<" topk="<<model.config.topk
        <<" scores=predicted_then_same_layer_oracle row_dtype=FP32\n";
}

Qwen4Model::Qwen4Model() = default;
Qwen4Model::~Qwen4Model() = default;
Qwen4Model::Qwen4Model(Qwen4Model&&) noexcept = default;
Qwen4Model& Qwen4Model::operator=(Qwen4Model&&) noexcept = default;

void Qwen4Model::adapter_validate_load_options() const {
    weight_loader::validate_load_options(*execution);
}

void Qwen4Model::adapter_load_final_state(const mfq::ModelSource &source,
                                          mfq_tensor_backend::Tensor &output_norm) {
    qwen4_exp::BlockLoader loader{{*execution, source, "qwen4_exp"}};
    final_mixer = loader.final_mixer(
        mfq::models::qwen4_exp::load_residual(loader, config, "model.mhc.pre", false));
    output_norm = mfq_tensor_backend::Tensor();
}

std::unique_ptr<Block> Qwen4Model::adapter_load_block(const mfq::ModelSource &source, int layer,
                                                      int, const std::string &) {
    return std::make_unique<qwen4_exp::Qwen4Block>(*execution, source, config, layer);
}

void Qwen4Model::adapter_reset(int64_t new_batch) {
    if(execution && execution->moe_expert_cache)finish_moe_expert_exchanges(execution->moe_expert_cache);
    graph_registry.reset();
    positions = mfq_tensor_backend::Tensor();
    batch = new_batch;
}
void Qwen4Model::adapter_begin_forward(bool) {
    if(graph_registry) {graph_registry->restore_legacy();graph_registry.reset();}
}

mfq_tensor_backend::Tensor
Qwen4Model::adapter_block_positions(const mfq_tensor_backend::Tensor &full_positions,
                                    const mfq_tensor_backend::Tensor &, int device) const {
    return tensor_to_cuda_device(execution->model_parallel_collectives, full_positions, device);
}

void Qwen4Model::adapter_finish_forward(const mfq_tensor_backend::Tensor &full_positions,
                                        int64_t new_batch, int64_t) {
    positions = full_positions;
    batch = new_batch;
}

mfq_tensor_backend::Tensor Qwen4Model::adapter_finalize_hidden(mfq_tensor_backend::Tensor hidden,
                                                               const mfq_tensor_backend::Tensor &,
                                                               int64_t, int64_t) const {
    return final_mixer->pre(*execution, hidden)[0];
}

mfq_tensor_backend::Tensor
Qwen4Model::adapter_last_logits(const QuantLinear &lm_head,
                                mfq_tensor_backend::Tensor hidden) const {
    return adapter_logits(lm_head, std::move(hidden));
}

mfq_tensor_backend::Tensor Qwen4Model::adapter_next_token(const QuantLinear &lm_head,
                                                          mfq_tensor_backend::Tensor hidden) const {
    return mfq_tensor_backend::argmax(adapter_logits(lm_head, std::move(hidden)), -1)
        .to(mfq_tensor_backend::kInt64);
}

void Qwen4Model::adapter_rollback_speculative(int64_t keep) {
    if (positions.defined())
        positions = positions.narrow(-1, 0, keep);
}

} // namespace mfq::cuda

namespace mfq::cuda {

template struct CudaSessionCodec<Qwen4Model>;

} // namespace mfq::cuda

namespace mfq::models {
template struct qwen4_exp::CausalLm<cuda::CudaCausalOps<cuda::Qwen4Model>>;
} // namespace mfq::models
