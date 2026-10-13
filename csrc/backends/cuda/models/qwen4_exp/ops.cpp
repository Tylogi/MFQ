#include "native/env.h"
#include "kernels/moe/ops.h"
#include "ops/execution/execution_options.h"
#include "kernels/shared_gate.h"
#include "kernels/sampling_ops.h"
#include "ops.h"
#include "model.h"
#include "mtp.h"
#include "storage/loaders/weight.h"
#include "storage/moe/ffn_pipeline.h"
#include "storage/moe/expert_cache.h"
#include "storage/nint_row_pipeline.h"
#include "ops/decode_window.h"
#include "kernels/moe/pipeline.h"
#include "ops/moe/router_lookahead_audit.h"
#include "common/gated_mlp.h"
#include "common/moe.h"
#include "common/transformer_layer.h"
#include "models/common/session_codec.h"
#include "ops/gated_residual.h"
#include "kernels/gated_residual_fused.h"
#include "kernels/float_projection.h"
#include "ops/quant/nint_prefill_cache.h"
#ifdef MFQ_NATIVE_CUDA_RUNTIME
#include "native/matmul_rows.h"
#endif
#include <cstdlib>
#include <chrono>
#include <fstream>
#include <iostream>

namespace mfq::cuda::qwen4_exp {
using Config = mfq::models::qwen4_exp::Config;

struct Gr {
    Tensor norm;
    weight_loader::ResidualLinear down, up, injection;
    int64_t hidden, streams;
    double eps;
    int64_t prefill_work_chunk(const Tensor& input)const {
        if(input.dim()!=3 || input.size(0)!=1)return 0;
        const auto* option=mfq::cuda::env_lookup("MFQ_GR_PREFILL_WORK_CHUNK");
#ifdef MFQ_NATIVE_CUDA_RUNTIME
        const int64_t chunk=option?std::atoll(option):8192;
        // Chat framing just beyond 8k needs no large temporary-memory cut.
        // Avoid extra concatenation there; longer inputs use 8k work tiles.
        if(!option && input.size(1)<=9216)return 0;
#else
        const int64_t chunk=option?std::atoll(option):0;
#endif
        return chunk>=256 && input.size(1)>chunk?chunk:0;
    }
    template<class Project>
    std::vector<Tensor> bounded_pre(const Tensor& input,int64_t chunk,bool reuse_input,Project project)const {
        const auto tokens=input.size(1);
        std::vector<Tensor> output(3);
        if(reuse_input)output[1]=input;
        for(int64_t begin=0;begin<tokens;) {
            auto count=std::min(chunk,tokens-begin);
            const auto tail=tokens-begin-count;
            // Both pieces must retain the main GR prefill kernels; a tiny
            // remainder selects different projection arithmetic.
            if(tail>0 && tail<128)count-=128-tail;
            auto part=project(begin,count);
            for(size_t index=0;index<part.size();++index) {
                if(!part[index].defined() || (reuse_input && index==1))continue;
                if(!output[index].defined()) {
                    auto shape=part[index].sizes().vec();shape[1]=tokens;
                    output[index]=tb::empty(shape,part[index].options());
                }
                output[index].narrow(1,begin,count).copy_(part[index]);
            }
            begin+=count;
        }
        return output;
    }
    bool can_fuse(CudaExecutionContext& execution,const Tensor& x)const {
        if(!execution.config.gr_two_stage || !execution.config.gr_fused_projections ||
            !execution.config.gr_native_projection_input || !down.weight || !up.weight || !injection.weight ||
            !down.weight->is_nint() || !up.weight->is_nint() || streams<1 || streams>4 ||
            x.dim()<1 || x.size(-1)!=hidden*streams || x.numel()<hidden*streams ||
            x.numel()/(hidden*streams)>8 || !x.is_cuda() || !x.is_contiguous() ||
            (x.scalar_type()!=tb::kFloat16 && x.scalar_type()!=tb::kFloat32) ||
            !norm.is_contiguous() || norm.scalar_type()!=tb::kFloat32)return false;
        if(!injection.weight->is_nint() && !(execution.config.gr_two_stage_dense_injection &&
            injection.weight->is_dense() && injection.weight->dense.is_contiguous() &&
            injection.weight->dense.scalar_type()==tb::kBFloat16))return false;
        const auto& w=up.weight->nint;
        const auto rows=x.numel()/(hidden*streams);
        const auto* batch=mfq::cuda::env_lookup("MFQ_GR_BATCH_TWO_STAGE");
        if(rows>1 && batch && batch[0]=='0')return false;
        return w.ng==1 || ((w.out+3)/4)*rows>=nint_float_projection_resident_blocks(x.get_device());
    }
    std::vector<Tensor> fused(const Tensor& branch,const Tensor& residual,const Tensor& prior,bool after,
        const GatedResidualMarker& marker={})const {
        const auto& projection=*injection.weight;
        return gated_residual_two_stage_cuda(branch,residual,prior,norm,down.weight->nint,up.weight->nint,
            projection.is_nint()?&projection.nint:nullptr,streams,eps,after,
            projection.is_dense()?projection.dense:Tensor{},injection.prepared_right,marker);
    }
    std::vector<Tensor> pre(CudaExecutionContext &execution, const Tensor &x,
        const GatedResidualMarker& marker={}) const {
        if(const auto chunk=prefill_work_chunk(x))
            return bounded_pre(x,chunk,true,[&](int64_t begin,int64_t count) {
                return pre(execution,x.narrow(1,begin,count),marker);
            });
        if(can_fuse(execution,x))return fused({},x,{},false,marker);
        return gated_residual_pre_projected(execution, x, norm, down.forward, up.forward, injection.forward,
            hidden, streams, eps,up.mixed,marker,down.activated,injection.activated);
    }
    Tensor post(const Tensor &branch, const std::vector<Tensor> &inputs) const {
        return mfq_qwen4_exp::gated_residual_post(branch, inputs[1], inputs[2], streams);
    }
    std::vector<Tensor> pre_after(CudaExecutionContext& execution,const Tensor& branch,
        const std::vector<Tensor>& inputs,const GatedResidualMarker& marker={}) const {
        if(const auto chunk=prefill_work_chunk(inputs[1]))
            return bounded_pre(inputs[1],chunk,false,[&](int64_t begin,int64_t count) {
                const std::vector<Tensor> part{{},inputs[1].narrow(1,begin,count),inputs[2].narrow(1,begin,count)};
                return pre_after(execution,branch.narrow(1,begin,count),part,marker);
            });
        if(can_fuse(execution,inputs[1]) && branch.is_contiguous() && inputs[2].is_contiguous() &&
            (branch.scalar_type()==tb::kFloat16 || branch.scalar_type()==tb::kFloat32) &&
            (inputs[2].scalar_type()==tb::kFloat16 || inputs[2].scalar_type()==tb::kFloat32))
            return fused(branch,inputs[1],inputs[2],true,marker);
        return gated_residual_pre_after_projected(execution,branch,inputs[1],inputs[2],norm,
            down.forward,up.forward,injection.forward,hidden,streams,eps,
            up.mixed,marker,down.activated,injection.activated);
    }
};

// Preserve router and shared-expert arithmetic while regrouping routed experts.
static thread_local int64_t qwen_prefill_reference_rows=0;
struct QwenPrefillReferenceRowsScope {
    int64_t previous;
    explicit QwenPrefillReferenceRowsScope(int64_t rows):previous(qwen_prefill_reference_rows) {
        qwen_prefill_reference_rows=rows;
    }
    ~QwenPrefillReferenceRowsScope(){qwen_prefill_reference_rows=previous;}
};
template<class Projection>
static Tensor qwen_prefill_preserve_rows(const Tensor& source,Projection project) {
    const auto rows=qwen_prefill_reference_rows;
    if(rows<=0 || source.size(0)<=rows)return project(source);
    Tensor output;
    for(int64_t begin=0;begin<source.size(0);begin+=rows) {
        const auto count=std::min<int64_t>(rows,source.size(0)-begin);
        auto part=project(source.narrow(0,begin,count));
        if(!output.defined()) {
            auto shape=part.sizes().vec();shape[0]=source.size(0);
            output=tb::empty(shape,part.options());
        }
        output.narrow(0,begin,count).copy_(part);
    }
    return output;
}

// This scope is active only while an entire GDN layer is evaluated. Retain
// each projection's original token count and therefore its GEMM arithmetic.
static thread_local int64_t qwen_gdn_prefill_projection_rows=0;
struct QwenGdnPrefillProjectionRowsScope {
    int64_t previous;
    explicit QwenGdnPrefillProjectionRowsScope(int64_t rows):previous(qwen_gdn_prefill_projection_rows) {
        qwen_gdn_prefill_projection_rows=rows;
    }
    ~QwenGdnPrefillProjectionRowsScope(){qwen_gdn_prefill_projection_rows=previous;}
};
static Linear qwen_gdn_preserve_projection_rows(Linear project) {
    return [project](CudaExecutionContext& execution,const Tensor& source) {
        const auto rows=qwen_gdn_prefill_projection_rows;
        if(rows<=0 || source.dim()!=3 || source.size(0)!=1 || source.size(1)<=rows)
            return project(execution,source);
#ifdef MFQ_NATIVE_CUDA_RUNTIME
        const auto* parallel_option=mfq::cuda::env_lookup("MFQ_GDN_PREFILL_PROJECTION_STREAMS");
        const int streams=parallel_option?std::atoi(parallel_option):2;
        const auto tail=source.size(1)%rows;
        if(streams>0 && (tail==0 || tail>8)) {
            const mfq::cuda::MatmulRowsScope gemm_rows(rows,streams);
            return project(execution,source);
        }
#endif
        Tensor output;
        for(int64_t begin=0;begin<source.size(1);begin+=rows) {
            const auto count=std::min<int64_t>(rows,source.size(1)-begin);
            auto part=project(execution,source.narrow(1,begin,count));
            if(!output.defined()) {
                auto shape=part.sizes().vec();shape[1]=source.size(1);
                output=tb::empty(shape,part.options());
            }
            output.narrow(1,begin,count).copy_(part);
        }
        return output;
    };
}
static LinearGroup qwen_gdn_preserve_projection_rows(LinearGroup project) {
    if(!project)return {};
    return [project](CudaExecutionContext& execution,const Tensor& source) {
        const auto rows=qwen_gdn_prefill_projection_rows;
        if(rows<=0 || source.dim()!=3 || source.size(0)!=1 || source.size(1)<=rows)
            return project(execution,source);
#ifdef MFQ_NATIVE_CUDA_RUNTIME
        const auto* parallel_option=mfq::cuda::env_lookup("MFQ_GDN_PREFILL_PROJECTION_STREAMS");
        const int streams=parallel_option?std::atoi(parallel_option):2;
        const auto tail=source.size(1)%rows;
        if(streams>0 && (tail==0 || tail>8)) {
            const mfq::cuda::MatmulRowsScope gemm_rows(rows,streams);
            return project(execution,source);
        }
#endif
        std::vector<Tensor> output;
        for(int64_t begin=0;begin<source.size(1);begin+=rows) {
            const auto count=std::min<int64_t>(rows,source.size(1)-begin);
            auto part=project(execution,source.narrow(1,begin,count));
            if(output.empty())for(const auto& value:part) {
                auto shape=value.sizes().vec();shape[1]=source.size(1);
                output.push_back(tb::empty(shape,value.options()));
            }
            MFQ_RUNTIME_CHECK(part.size()==output.size(),"GDN projection count changed across chunks");
            for(size_t i=0;i<part.size();++i)output[i].narrow(1,begin,count).copy_(part[i]);
        }
        return output;
    };
}

static Linear qwen_moe(weight_loader::Routed gate_up, weight_loader::Routed down,
                       Linear router, Linear shared_gate, Linear sg, Linear su, Linear sd,
                       const Config &c, MoeFfnPrefetch* prefetch=nullptr) {
    auto shared_raw=[sg,su,sd,shared_gate](CudaExecutionContext& execution,const Tensor& source) {
        auto unfused=[](const auto&...){return std::optional<Tensor>{};};
        auto output=mfq::models::gated_mlp(source,false,0.0,unfused,unfused,
            [&](Tensor input){return std::array<Tensor,2>{sg(execution,input),su(execution,input)};},
            [](Tensor g,Tensor u,mfq::models::GatedActivation,double){return (g*tb::sigmoid(g))*u;},
            [&](Tensor hidden){return sd(execution,hidden);},unfused);
        return tb::sigmoid(shared_gate(execution,source))*output;
    };
    const std::array<Linear,3> shared_projections{sg,su,sd};
    const bool shared_row_parallel=std::all_of(shared_projections.begin(),shared_projections.end(),
        [](const auto& projection) {
            const auto weight=weight_loader::linear_weight(projection);
            return weight && weight->is_nint() && !weight->tensor_parallel() && !weight->nint.q8_zero;
        });
    auto shared=[shared_raw,shared_projections,shared_gate,shared_row_parallel](CudaExecutionContext& execution,const Tensor& source) {
#ifdef MFQ_NATIVE_CUDA_RUNTIME
        const auto rows=qwen_prefill_reference_rows;
        const auto* parallel_option=mfq::cuda::env_lookup("MFQ_MOE_PREFILL_SHARED_STREAMS");
        const int streams=parallel_option?std::atoi(parallel_option):0;
        const auto* chain_option=mfq::cuda::env_lookup("MFQ_MOE_PREFILL_SHARED_CHAIN_STREAMS");
        const int chain_streams=chain_option?std::atoi(chain_option):4;
        const auto tail=rows>0?source.size(0)%rows:0;
        if(chain_streams>0 && shared_row_parallel && rows>8 && source.size(0)>=8192 &&
                source.size(0)>rows && (tail==0 || tail>8)) {
            // The first original chunk prepares all three cached weight
            // matrices. Worker pipelines then read them after the ready fence.
            auto first=shared_raw(execution,source.narrow(0,0,rows));
            auto shape=first.sizes().vec();shape[0]=source.size(0);
            auto output=tb::empty(shape,first.options());
            output.narrow(0,0,rows).copy_(first);
            mfq::cuda::submit_parallel_rows(source.get_device(),rows,source.size(0),rows,chain_streams,
                [&](int64_t begin,int64_t count) {
                    auto part=shared_raw(execution,source.narrow(0,begin,count));
                    output.narrow(0,begin,count).copy_(part);
                });
            return output;
        }
        if(streams>0 && shared_row_parallel && rows>8 && source.size(0)>rows && (tail==0 || tail>8)) {
            // Each GEMM retains its original row count. Only its submission
            // stream changes; activations remain in their canonical FP16 form.
            const auto project=[&](int index,const Tensor& input) {
                const mfq::cuda::MatmulRowsScope gemm_rows(rows,streams);
                return shared_projections[index](execution,input);
            };
            auto gate=project(0,source),up=project(1,source);
            auto output=project(2,(gate*tb::sigmoid(gate))*up);
            auto scale=qwen_prefill_preserve_rows(source,[&](const Tensor& part) {
                return tb::sigmoid(shared_gate(execution,part));
            });
            return scale*output;
        }
#endif
        return qwen_prefill_preserve_rows(source,[&](const Tensor& part){return shared_raw(execution,part);});
    };
    MoeFfnSharedWeights shared_weights;
    shared_weights.projections={weight_loader::linear_weight(sg),weight_loader::linear_weight(su),weight_loader::linear_weight(sd)};
    const auto shared_gate_weight=weight_loader::linear_weight(shared_gate);
    shared_weights.gate=[shared_gate,shared_gate_weight](CudaExecutionContext& execution,const Tensor& input) {
        const auto option=mfq::cuda::runtime_options::shared_gate_fused();
        if((!option || *option) && shared_gate_weight && shared_gate_weight->is_dense() &&
            !shared_gate_weight->tensor_parallel())
            if(auto output=try_shared_gate_sigmoid_cuda(input,shared_gate_weight->dense))return *output;
        return tb::sigmoid(shared_gate(execution,input));
    };
    auto pipeline=make_moe_ffn_pipeline(gate_up.projections,down.projections,shared,{},std::move(shared_weights),prefetch);
    return [gate_up, down, router, shared, pipeline, c](CudaExecutionContext &execution,
                                                               const Tensor &x) {
        auto source = x.reshape({-1, c.hidden}).to(tb::kFloat16);
        const bool use_pipeline=pipeline && (gate_up.projections.size()!=1 ||
            (source.size(0)<=8 && execution.config.moe_two_stage_ffn &&
             execution.kl_mmq.mode==KlMmqMode::Default));
        const auto routing = c.routing();
        return mfq::models::mixture_of_experts(
            [&] { return qwen_prefill_preserve_rows(source,[&](const Tensor& part){return router(execution,part);}).contiguous(); },
            [&](Tensor logits) {
                return moe_topk_cuda(
                    logits, c.topk, routing.activation == mfq::models::RouterActivation::sigmoid,
                    routing.activation == mfq::models::RouterActivation::sqrt_softplus,
                    routing.normalize, routing.delayed_softmax, mfq_nullopt, 1e-20, routing.scale);
            },
            [&](const auto &) {
                return use_pipeline ? Tensor{} : shared(execution,source);
            },
            [&](const auto &selected) {
                if(use_pipeline)return pipeline(execution,source,selected[0].to(tb::kInt32).contiguous(),selected[1]);
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
    auto gdn_weights(const std::string &p) const {
        const bool independent = has(p + ".query.weight") || has(p + ".key.weight") || has(p + ".value.weight");
        MFQ_RUNTIME_CHECK(!independent || !has(p + ".qkv.weight"), "ambiguous Qwen4 GDN input projections");
        GdnWeights w{independent ? Linear{} : linear(p + ".qkv.weight"), linear(p + ".gate.weight"),
            linear(p + ".alpha.weight"), linear(p + ".beta.weight"), linear(p + ".output.weight"),
            dense(p + ".conv.weight"), dense(p + ".dt_bias"), dense(p + ".a"), dense(p + ".norm.weight")};
        if (independent)
            for (const auto* name : {"query", "key", "value"})
                w.split_qkv.push_back(linear(p + "." + name + ".weight"));
        return w;
    }
    auto qsa_weights(const std::string &p) const {
        QsaWeights w{linear(p + ".query.weight"), linear(p + ".key.weight"), linear(p + ".value.weight"),
            linear(p + ".output.weight"), linear(p + ".indexer.query_key.weight"), dense(p + ".query_norm.weight"),
            dense(p + ".key_norm.weight"), dense(p + ".indexer.query_norm.weight"), dense(p + ".indexer.key_norm.weight")};
        if (has(p + ".gate.weight")) w.gate = linear(p + ".gate.weight");
        return w;
    }
    auto gdn(GdnWeights weights, const Config &c) const {
        std::vector<Linear> projections = weights.split_qkv.empty()
            ? std::vector<Linear>{weights.qkv} : weights.split_qkv;
        const bool independent = !weights.split_qkv.empty();
        projections.insert(projections.end(), {weights.gate,weights.alpha,weights.beta});
        auto group = weight_loader::grouped_linear(execution,projections);
        if (independent) {
            if (group)
                weights.input_projection = [group](CudaExecutionContext &e, const Tensor &x) {
                    auto p = group(e,x);
                    return std::vector<Tensor>{tb::cat({p[0],p[1],p[2]},-1),p[3],p[4],p[5]};
                };
            weights.qkv = [q=projections[0],k=projections[1],v=projections[2]](CudaExecutionContext &e, const Tensor &x) {
                return tb::cat({q(e,x),k(e,x),v(e,x)},-1);
            };
        } else {
            weights.input_projection = std::move(group);
            weights.qkv = std::move(projections[0]);
        }
        const auto start = independent ? 3 : 1;
        weights.gate=std::move(projections[start]);weights.alpha=std::move(projections[start+1]);
        weights.beta=std::move(projections[start+2]);
        weights.input_projection=qwen_gdn_preserve_projection_rows(std::move(weights.input_projection));
        weights.qkv=qwen_gdn_preserve_projection_rows(std::move(weights.qkv));
        weights.gate=qwen_gdn_preserve_projection_rows(std::move(weights.gate));
        weights.alpha=qwen_gdn_preserve_projection_rows(std::move(weights.alpha));
        weights.beta=qwen_gdn_preserve_projection_rows(std::move(weights.beta));
        weights.output=qwen_gdn_preserve_projection_rows(std::move(weights.output));
        return std::make_unique<Gdn>(std::move(weights), c.key_heads, c.value_heads, c.linear_width,
                                     c.kernel, c.eps, c.silu_gate,true,execution.config.gdn_transposed_state,
                                     execution.config.gdn_fused_output,true,execution.config.gdn_fused_preparation,
                                     execution.config.gdn_fused_core);
    }
    auto qsa(QsaWeights weights, const Config &c) const {
        std::vector<Linear> projections{weights.query,weights.key,weights.value,weights.index_query_key};
        if (weights.gate) projections.push_back(weights.gate);
        auto group = weight_loader::grouped_linear(execution,projections);
        if (weights.gate) {
            const auto heads = c.heads, width = c.width;
            const auto combine = [heads,width](const Tensor &q,const Tensor &g) {
                const auto b=q.size(0),t=q.size(1);
                return tb::cat({q.reshape({b,t,heads,width}),g.reshape({b,t,heads,width})},-1)
                    .reshape({b,t,2*heads*width});
            };
            if (group)
                weights.input_projection = [group,combine](CudaExecutionContext &e,const Tensor &x) {
                    auto p=group(e,x);p[0]=combine(p[0],p[4]);p.resize(4);return p;
                };
            projections[0] = [q=projections[0],g=projections[4],combine](CudaExecutionContext &e,const Tensor &x) {
                return combine(q(e,x),g(e,x));
            };
        } else weights.input_projection=std::move(group);
        weights.query=std::move(projections[0]);weights.key=std::move(projections[1]);
        weights.value=std::move(projections[2]);weights.index_query_key=std::move(projections[3]);
        QsaConfig geometry{c.heads, c.kv_heads, c.width, c.index_heads, c.index_width,
                           c.pool, c.budget, c.maximum, c.eps};
        auto rotary = std::make_shared<RotaryEmbedding>(c.rotary, c.maximum, c.rope_base, c.sections, c.interleaved);
        return std::make_unique<Qsa>(std::move(weights), geometry, std::move(rotary), true, true, execution.qsa_kv_store);
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

struct RecurrentGraphCheckpoint {
    Tensor gdn_conv,gdn_state,ple_conv;
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
                return execution.profiler.measure("prefill.ple",[&] {
                    return ple->forward(execution, hidden, ids, true, confirmed,host_ids);
                });
            },
            [](Tensor hidden, Tensor positional) { return hidden + positional; },
            [&](const Tensor &hidden) { return execution.profiler.measure("prefill.gr_attention",[&] {return attention_gr.pre(execution, hidden);}); },
            [&](Tensor branch) { return execution.profiler.measure("prefill.gdn",[&] {return gdn->forward(execution, branch, true, confirmed);}); },
            [&](Tensor branch) {
                return execution.profiler.measure("prefill.qsa",[&] {return qsa->forward(execution, branch, positions, full_positions, true);});
            },
            [&](Tensor branch, const auto &mix) { return execution.profiler.measure("prefill.gr_ffn_pre",[&] {return ffn_gr.pre_after(execution, branch, mix);}); },
            [&](Tensor branch) { return execution.profiler.measure("prefill.ffn",[&] {return ffn(execution, branch);}); },
            [&](Tensor branch, const auto &mix) { return execution.profiler.measure("prefill.gr_ffn_post",[&] {return ffn_gr.post(branch, mix);}); });
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
    Tensor prefill_ple(CudaExecutionContext& execution,Tensor x,const Context& context) {
        if(ple)x=x+execution.profiler.measure("prefill.ple",[&] {
            return ple->forward(execution,x,context.token_ids,true,0,context.host_token_ids);
        });
        return x;
    }
    Tensor prefill_attention_core(CudaExecutionContext& execution,const Tensor& input,const Context& context) {
        return gdn?
            execution.profiler.measure("prefill.gdn",[&] {return gdn->forward(execution,input,true,0);}):
            execution.profiler.measure("prefill.qsa",[&] {return qsa->forward(execution,input,context.positions,context.full_positions,true);});
    }
    std::vector<Tensor> prefill_attention(CudaExecutionContext& execution,Tensor x,const Context& context) {
        x=prefill_ple(execution,std::move(x),context);
        auto mix=execution.profiler.measure("prefill.gr_attention",[&] {return attention_gr.pre(execution,x);});
        auto branch=prefill_attention_core(execution,mix[0],context);
        return execution.profiler.measure("prefill.gr_ffn_pre",[&] {return ffn_gr.pre_after(execution,branch,mix);});
    }
    std::vector<Tensor*> graph_warmup_state() override {
        std::vector<Tensor*> result;
        if(gdn)for(auto* value:gdn->graph_state())if(value->defined())result.push_back(value);
        if(ple)for(auto* value:ple->graph_state())if(value->defined())result.push_back(value);
        return result;
    }
    Tensor execute_graph(CudaExecutionContext& execution,Tensor x,const Tensor& ids,
                         const Tensor& positions,const Tensor& cache_positions,const Tensor& ple_rows,int layer,
                         int64_t confirmed=0,RecurrentGraphCheckpoint* checkpoint=nullptr) {
        const auto mark=[&](const char* phase){if(auto* window=DecodeWindow::recording())window->mark(layer,phase);};
        const bool detailed=mfq::cuda::runtime_options::mhc_timings();
        const auto marker=[&](const char* role)->GatedResidualMarker {
            if(!detailed || !DecodeWindow::recording())return {};
            return [&,role](const char* phase) {
                const auto name=std::string(role)+phase;mark(name.c_str());
            };
        };
        return mfq::models::qwen4_exp::decoder_layer_chained(std::move(x),bool(ple),bool(gdn),
            [&](const Tensor& hidden){return confirmed
                ? ple->forward_graph(execution,hidden,ids,ple_rows,confirmed,checkpoint->ple_conv)
                : ple->forward_chunk(execution,hidden,ids,true,{},ple_rows);},
            [](Tensor hidden,Tensor positional){return hidden+positional;},
            [&](const Tensor& hidden){auto result=attention_gr.pre(execution,hidden,marker("attention_mhc_"));mark("attention_core_begin");return result;},
            [&](Tensor branch){auto result=confirmed
                ? gdn->forward_graph(execution,branch,confirmed,checkpoint->gdn_conv,checkpoint->gdn_state)
                : gdn->forward_chunk(execution,branch,true);mark("attention_core_end");return result;},
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

struct Qwen4MtpGraph {
    struct Cell {
        Tensor ids,hidden,positions,cache_positions,normalized,multi,logits,greedy_host;
        DecodeWindow window;
        Cell(const Tensor& source,const Tensor& token_ids,cudaStream_t stream)
            :ids(tb::empty_like(token_ids)),hidden(tb::empty_like(source)),
             positions(tb::empty({3,1,source.size(1)},token_ids.options().dtype(tb::kInt32))),
             cache_positions(tb::empty({source.size(1)},token_ids.options().dtype(tb::kInt64))),
             greedy_host(tb::empty({1},token_ids.options().device(tb::kCPU).dtype(tb::kInt64).pinned_memory(true))),window(stream) {}
    };
    MfqCudaStream stream=mfq_get_current_cuda_stream();
    std::map<std::pair<int,int>,std::unique_ptr<Cell>> cells;
    int64_t capacity=0;
    bool failed=false;
    void restore_legacy(Qwen4ExpMtp& predictor) {
        cells.clear();predictor.layers[0]->qsa->leave_graph();
    }
    MtpStep run(Qwen4ExpMtp& predictor,const MtpTarget& target,const Tensor& hidden,const Tensor& ids) {
        MFQ_RUNTIME_CHECK(!failed && stream.stream()==mfq_current_cuda_stream(),"MTP predictor graph is unavailable");
        auto& execution=*predictor.execution;auto& block=*predictor.layers[0];
        const int64_t tokens=ids.size(1),start=predictor.lengths[0],required=start+tokens;
        MFQ_RUNTIME_CHECK(required<=predictor.config.maximum,"MTP predictor exceeds context capacity");
        if(required>capacity) {
            finish_moe_expert_exchanges(execution.moe_expert_cache);restore_legacy(predictor);
            capacity=std::min<int64_t>(predictor.config.maximum,std::max<int64_t>(2048,capacity));
            while(capacity<required)capacity=std::min(predictor.config.maximum,capacity*2);
        }
        const auto key=std::make_pair(int(tokens),int(hidden.scalar_type()));
        auto& owner=cells[key];if(!owner)owner=std::make_unique<Cell>(hidden,ids,stream.stream());
        auto& cell=*owner;cell.hidden.copy_(hidden);cell.ids.copy_(ids);
        mfq_qwen4_exp::fill_decode_positions(cell.cache_positions,cell.positions,start);
        try {
            finish_moe_expert_exchanges(execution.moe_expert_cache);
            if(!cell.window.valid()) {
                DecodeGraphBranchScope branch_scope(execution.decode_graph_serial_branches);
                predictor.graph_initialized=true;
                block.qsa->prepare_graph(predictor.positions[0],capacity);
                cell.window.capture([&] {
                    auto embeddings=target.embed(cell.ids);
                    auto e=predictor.embedding_fusion(execution,rms_norm(embeddings,predictor.embedding_norm+1,predictor.config.eps));
                    auto h=predictor.hidden_fusion(execution,rms_norm(cell.hidden,predictor.hidden_norm+1,predictor.config.eps)
                        .reshape({1,tokens,predictor.config.streams,predictor.config.hidden}));
                    auto input=(h+e.unsqueeze(-2)).reshape({1,tokens,predictor.config.streams*predictor.config.hidden});
                    cell.multi=block.execute_graph(execution,std::move(input),cell.ids,cell.positions,cell.cache_positions,{},0);
                    cell.normalized=predictor.final_mixer->pre(execution,cell.multi)[0];
                    cell.logits=target.logits(cell.normalized.narrow(1,tokens-1,1).contiguous());
                    auto greedy=sample_greedy_cuda(cell.logits.reshape({1,-1}).contiguous());
                    MFQ_CUDA_CHECK(cudaMemcpyAsync(cell.greedy_host.data_ptr(),greedy.data_ptr(),sizeof(int64_t),
                        cudaMemcpyDeviceToHost,stream.stream()));
                },[]{},[&]{cell.multi={};cell.normalized={};cell.logits={};});
            }
            cell.window.run();++predictor.graph_replays;block.qsa->advance_graph(tokens);
            predictor.positions[0]=tb::cat({predictor.positions[0],cell.positions},-1);
            predictor.lengths[0]=required;predictor.batch=1;
            return {cell.normalized.clone(),cell.multi.clone(),cell.logits.clone(),
                static_cast<int32_t>(*cell.greedy_host.data_ptr<int64_t>())};
        } catch(...) {failed=true;throw;}
    }
};

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
    graph_registry.reset();
    graph_initialized=false;
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
    if(graph_registry) {graph_registry->restore_legacy(*this);graph_registry.reset();}
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
    const auto* graph=mfq::cuda::env_lookup("MFQ_MTP_PREDICTOR_GRAPH");
    if((!graph || graph[0]!='0') && execution->config.moe_pipeline && execution->config.moe_two_stage_ffn &&
        execution->kl_mmq.mode==KlMmqMode::Default && execution->moe_expert_cache &&
        !execution->qsa_kv_store && batch==1 && ids.dim()==2 && ids.size(0)==1 && ids.size(1)>0 && ids.size(1)<=8 &&
        hidden.dim()==3 && hidden.size(0)==1 && hidden.size(1)==ids.size(1) && hidden.size(2)==hidden_norm.numel() &&
        ids.is_cuda() && hidden.is_cuda() && !lengths.empty() && lengths[0]>0 &&
        layers[0]->qsa && layers[0]->prefetch && positions[0].dim()==3 && positions[0].size(1)==1) {
        if(!graph_registry)graph_registry=std::make_unique<Qwen4MtpGraph>();
        return graph_registry->run(*this,target,hidden,ids);
    }
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
        qwen4_exp::NgramEmbedding::Context prefix_context;
        HostBuffer flags{128,true};
        uint32_t* device=nullptr;
        bool profile=false;
        uint64_t calls=0,route_wait_ns=0,upload_ns=0,notify_ns=0;
        PleStage(qwen4_exp::Ple* p,int tokens,int hidden):ple(p) {
            const auto timing=mfq::cuda::runtime_options::trace_ple_timings();profile=timing && *timing;
            charge_tensor_host_bytes(128);
            MFQ_CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&device),flags.data(),0));
            std::memset(flags.data(),0,128);publish_mapped_flag(host()+16);
            transfer=ple->graph_stage(tokens,mfq_tensor_backend::Device{mfq_tensor_backend::kCUDA,0});
            rows=transfer->output().reshape({1,tokens,hidden});
        }
        ~PleStage(){
            if(profile)std::cerr<<"ple_stage_host mapped_rows="<<transfer->mapped_rows()<<" calls="<<calls
                <<" route_wait_ns="<<route_wait_ns<<" upload_ns="<<upload_ns<<" notify_ns="<<notify_ns<<'\n';
            tensor_host_bytes.fetch_sub(128);
        }
        uint32_t* host(){return static_cast<uint32_t*>(flags.data());}
    };
    struct Cell {
        mfq_tensor_backend::Tensor ids,positions,cache_positions,host_ids,output,raw_hidden,logits,greedy_host;
        std::vector<std::unique_ptr<PleStage>> ple;
        std::vector<qwen4_exp::RecurrentGraphCheckpoint> checkpoints;
        DecodeWindow window;
        Cell(int tokens,int hidden,const std::vector<std::unique_ptr<Block>>& blocks,cudaStream_t stream,int confirmed)
            :window(stream,mfq::cuda::runtime_options::layer_timings()) {
            namespace tb=mfq_tensor_backend;
            const auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kInt64);
            ids=tb::zeros({1,tokens},options);positions=tb::zeros({tokens},options);cache_positions=tb::zeros({tokens},options);
            host_ids=tb::zeros({1,tokens},options.device(tb::kCPU));
            greedy_host=tb::empty({tokens},options.device(tb::kCPU).pinned_memory(true));
            for(const auto& block:blocks) {
                auto& b=static_cast<qwen4_exp::Qwen4Block&>(*block);
                if(b.ple)ple.push_back(std::make_unique<PleStage>(b.ple.get(),tokens,hidden));
                if(confirmed) {
                    qwen4_exp::RecurrentGraphCheckpoint checkpoint;
                    if(b.gdn) {
                        const auto state=b.gdn->graph_state();
                        checkpoint.gdn_conv=tb::empty_like(*state[0]);checkpoint.gdn_state=tb::empty_like(*state[1]);
                    }
                    if(b.ple)checkpoint.ple_conv=tb::empty_like(b.ple->conv_state());
                    checkpoints.push_back(std::move(checkpoint));
                }
            }
        }
    };
    MfqCudaStream stream,ple_stream;
    std::map<std::tuple<int,int,int>,std::unique_ptr<Cell>> cells;
    std::vector<qwen4_exp::Qsa*> qsa;
    int64_t kv_capacity=0;
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

    mfq_tensor_backend::Tensor run(Qwen4CausalLm& model,mfq_tensor_backend::Tensor ids,int kind,
                                  mfq_tensor_backend::Tensor* raw_hidden=nullptr,int confirmed=0,bool* initialized=nullptr,
                                  const mfq_tensor_backend::Tensor& host_ids={},mfq_tensor_backend::Tensor* logits=nullptr,
                                  std::vector<int32_t>* greedy_tokens=nullptr) {
        namespace tb=mfq_tensor_backend;
        if(failed)throw std::runtime_error("MFQ model session requires recovery after its previous error");
        if(stream.stream()!=mfq_current_cuda_stream())throw std::runtime_error("MFQ model window changed execution stream");
        const int tokens=static_cast<int>(ids.size(1));
        release_moe_prefill_buffers(model.execution->moe_expert_cache);
        const auto required=model.cache_pos+tokens;
        MFQ_RUNTIME_CHECK(required<=model.max_position_embeddings(),"Qwen4 graph exceeds context capacity");
        if(required>kv_capacity) {
            finish_moe_expert_exchanges(model.execution->moe_expert_cache);
            restore_legacy();
            // Small growing chunks preserve dynamic KV allocation without
            // recapturing all MTP batch shapes at every few decode tokens.
            const int64_t chunk=raw_hidden?2048:256;
            kv_capacity=std::min<int64_t>(model.max_position_embeddings(),std::max<int64_t>(chunk,kv_capacity));
            while(kv_capacity<required)
                kv_capacity=std::min(model.max_position_embeddings(),kv_capacity>model.max_position_embeddings()/2
                    ? model.max_position_embeddings() : kv_capacity*2);
        }
        auto& owner=cells[{tokens,kind,confirmed}];
        if(!owner)owner=std::make_unique<Cell>(tokens,model.config.hidden,model.blocks,stream.stream(),confirmed);
        auto& cell=*owner;
        if(initialized)*initialized=!cell.window.valid();
        if(host_ids.defined()) {
            MFQ_RUNTIME_CHECK(!host_ids.is_cuda() && host_ids.scalar_type()==tb::kInt64 &&
                host_ids.sizes()==ids.sizes(),"Qwen4 graph host token geometry changed");
            cell.host_ids.copy_(host_ids);
        } else cell.host_ids.copy_(ids.to(tb::kCPU,tb::kInt64));
        cell.ids.copy_(ids);
        mfq_qwen4_exp::fill_decode_positions(cell.cache_positions,cell.positions,model.cache_pos,model.decode_position_delta);
        try {
            finish_moe_expert_exchanges(model.execution->moe_expert_cache);
            if(!cell.window.valid()) {
                DecodeGraphBranchScope branch_scope(
                    model.execution->decode_graph_serial_branches);
                for(const auto& block:model.blocks) {
                    auto& b=static_cast<qwen4_exp::Qwen4Block&>(*block);
                    if(b.qsa)b.qsa->prepare_graph(model.positions,kv_capacity);
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
                            cell.window.mark(layer,"ple_begin");
                            auto* stage=cell.ple[ple_index++].get();rows=stage->rows;
                            cell.window.enroll({stage,
                                [&cell,stage] {std::memset(stage->flags.data(),0,128);stage->ple->prefetch_tokens(cell.host_ids);},
                                [this,&cell,stage,confirmed] {
                                    using Clock=std::chrono::steady_clock;
                                    auto point=stage->profile?Clock::now():Clock::time_point{};
                                    const auto sample=[&](uint64_t& total) {
                                        if(!stage->profile)return;
                                        const auto now=Clock::now();
                                        total+=std::chrono::duration_cast<std::chrono::nanoseconds>(now-point).count();point=now;
                                    };
                                    wait_route_publication(stage->host(),stream.stream());
                                    sample(stage->route_wait_ns);
                                    MfqCudaStreamGuard guard(ple_stream);
                                    if(confirmed)stage->prefix_context=stage->ple->graph_prefix_context(cell.host_ids,confirmed);
                                    stage->ple->upload_graph(cell.host_ids,*stage->transfer);
                                    sample(stage->upload_ns);
                                    if(stage->transfer->mapped_rows())publish_mapped_flag(stage->host()+16);
                                    else {
                                        MFQ_CUDA_CHECK(cudaLaunchHostFunc(ple_stream.stream(),[](void* p){publish_mapped_flag(static_cast<uint32_t*>(p));},stage->host()+16));
                                        check_stream_progress(ple_stream.stream());
                                    }
                                    sample(stage->notify_ns);if(stage->profile)++stage->calls;
                                },{},[stage]{publish_mapped_flag(stage->host()+16);}});
                            signal_mapped_flag(stage->device,stream.stream());wait_mapped_flag(stage->device+16,stream.stream());
                            stage->transfer->decode();
                            cell.window.mark(layer,"ple_end");
                        }
                        cell.window.mark(layer,"attention_begin");
                        hidden=b.execute_graph(*model.execution,std::move(hidden),cell.ids,cell.positions,
                            cell.cache_positions,rows,layer,confirmed,confirmed?&cell.checkpoints[layer]:nullptr);
                        cell.window.mark(layer++,"block_end");
                    }
                    if(kind>=3)cell.raw_hidden=hidden.clone();
                    hidden=model.adapter_finalize_hidden(std::move(hidden),model.output_norm,1,tokens);
                    if(kind>=3) {
                        cell.output=std::move(hidden);
                        if(kind==4) {
                            cell.logits=model.logits_from_hidden(cell.output);
                            auto greedy=sample_greedy_cuda(cell.logits.reshape({tokens,-1}).contiguous());
                            MFQ_CUDA_CHECK(cudaMemcpyAsync(cell.greedy_host.data_ptr(),greedy.data_ptr(),tokens*sizeof(int64_t),
                                cudaMemcpyDeviceToHost,stream.stream()));
                        }
                    }
                    else if(kind==2)cell.output=model.adapter_next_token(model.lm_head,model.last_hidden(std::move(hidden)));
                    else if(kind==1)cell.output=model.adapter_last_logits(model.lm_head,model.last_hidden(std::move(hidden)));
                    else cell.output=model.logits_from_hidden(std::move(hidden));
                    cell.window.mark(-1,"window_end");
                    auto audit=static_cast<qwen4_exp::Qwen4Block&>(*model.blocks.front()).router_audit;
                    if(audit)cell.window.enroll({audit.get(),[]{},[]{},[audit]{audit->finish_sample();},{}});
                },restore,[&]{cell.output={};cell.raw_hidden={};cell.logits={};});
            }
            cell.window.run();
            if(confirmed) {
                // Captured copies save the confirmed prefix. Bind their owned
                // storage only after replay, so generic commit/rollback keeps
                // exactly the same transaction lifecycle as eager execution.
                std::size_t ple_index=0,layer=0;
                for(const auto& block:model.blocks) {
                    auto& b=static_cast<qwen4_exp::Qwen4Block&>(*block);
                    const auto& checkpoint=cell.checkpoints[layer++];
                    if(b.gdn)b.gdn->bind_graph_checkpoint(checkpoint.gdn_conv,checkpoint.gdn_state);
                    if(b.ple)b.ple->bind_graph_checkpoint(checkpoint.ple_conv,cell.ple[ple_index++]->prefix_context);
                }
                model.speculative_start=model.cache_pos;model.speculative_confirmed=confirmed;
            }
            for(const auto& block:model.blocks) {
                auto& b=static_cast<qwen4_exp::Qwen4Block&>(*block);if(b.qsa)b.qsa->advance_graph(tokens);
            }
            model.cache_pos+=tokens;
            model.positions=tb::cat({model.positions,cell.positions},-1);model.batch=1;
            // MTP can retain history until a later probe. Each step must own its
            // raw residual rather than alias the next replay's capture buffer.
            if(raw_hidden)*raw_hidden=cell.raw_hidden.clone();
            if(logits)*logits=cell.logits.clone();
            if(greedy_tokens) {
                const auto* first=cell.greedy_host.data_ptr<int64_t>();
                greedy_tokens->assign(first,first+tokens);
            }
            return cell.output;
        } catch(...) {failed=true;throw;}
    }
};

std::optional<mfq_tensor_backend::Tensor> qwen4_decode_graph_logits(Qwen4CausalLm& model,mfq_tensor_backend::Tensor ids,int kind) {
    if (model.execution->qsa_kv_store) return {};
    if(!model.execution->config.moe_pipeline || !model.execution->moe_expert_cache || model.cache_pos==0 || model.speculative_start>=0)return {};
    if(ids.dim()==1)ids=ids.unsqueeze(0);
    if(ids.dim()!=2 || ids.size(0)!=1 || ids.size(1)<1 || ids.size(1)>8 || model.positions.dim()!=1)return {};
    MFQ_RUNTIME_CHECK(model.cache_pos+ids.size(1)<=model.max_position_embeddings(),"MFQ window exceeds context");
    if(!model.graph_registry)model.graph_registry=std::make_unique<Qwen4DecodeGraph>(model);
    return model.graph_registry->run(model,model.device_ids(std::move(ids)),kind);
}

void qwen4_report_decode_graph(const Qwen4CausalLm& model,std::ostream& output) {
    if(!model.graph_registry)return;
    for(const auto& [key,cell]:model.graph_registry->cells)
        for(const auto& phase:cell->window.gpu_phases())if(phase.samples)
            output<<"mtp_graph_phase tokens="<<std::get<0>(key)<<" confirmed="<<std::get<2>(key)
                <<" layer="<<phase.layer<<" from="<<phase.from<<" to="<<phase.to
                <<" samples="<<phase.samples<<" total_ns="<<phase.total_ns<<'\n';
}

std::optional<mfq_tensor_backend::Tensor> qwen4_decode_graph_hidden(
        Qwen4CausalLm& model,mfq_tensor_backend::Tensor ids,mfq_tensor_backend::Tensor* raw_hidden,int64_t confirmed,
        bool* initialized,const mfq_tensor_backend::Tensor& host_ids,mfq_tensor_backend::Tensor* logits,
        std::vector<int32_t>* greedy_tokens) {
    if(initialized)*initialized=false;
    if(model.execution->qsa_kv_store || !model.execution->config.moe_pipeline ||
            !model.execution->moe_expert_cache || model.cache_pos==0 || model.speculative_start>=0)return {};
    if(ids.dim()==1)ids=ids.unsqueeze(0);
    if(ids.dim()!=2 || ids.size(0)!=1 || ids.size(1)<1 || ids.size(1)>8 || model.positions.dim()!=1 ||
        confirmed<0 || (confirmed && confirmed>=ids.size(1)))return {};
    for(const auto& block:model.blocks) {
        const auto& b=static_cast<const qwen4_exp::Qwen4Block&>(*block);
        if(!b.prefetch || (b.ple && !b.ple->graph_supported()))return {};
    }
    MFQ_RUNTIME_CHECK(model.cache_pos+ids.size(1)<=model.max_position_embeddings(),"MFQ window exceeds context");
    if(!model.graph_registry)model.graph_registry=std::make_unique<Qwen4DecodeGraph>(model);
    return model.graph_registry->run(model,model.device_ids(std::move(ids)),(logits || greedy_tokens)?4:3,raw_hidden,
        static_cast<int>(confirmed),initialized,host_ids,logits,greedy_tokens);
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

// Keep each compressed expert image resident while advancing all prompt chunks
// through that layer. Attention and PLE still advance in causal token order.
bool qwen4_layer_major_prefill_eligible(const Qwen4CausalLm& model, int64_t tokens) {
    const auto enabled=mfq::cuda::runtime_options::moe_prefill_layer_major();
    const auto whole=mfq::cuda::runtime_options::moe_prefill_layer();
    if((enabled && *enabled==0) || !whole || *whole==0 || tokens<=1024 ||
       model.cache_pos!=0 || model.speculative_start>=0 || !model.execution ||
       !model.execution->config.moe_pipeline || !model.execution->moe_expert_cache ||
       model.execution->dense_cpu_layer_count || model.execution->tensor_parallel.enabled() ||
       model.execution->expert_parallel.enabled())return false;
    const auto primary=model.execution->layer_placement.primary_device();
    for(const auto& block:model.blocks)
        if(block->cpu_offloaded || block->cuda_device!=primary)return false;
    return true;
}

mfq_tensor_backend::Tensor qwen4_layer_major_prefill_hidden(
        Qwen4CausalLm& model,mfq_tensor_backend::Tensor ids,int64_t compute_chunk,
        bool last_only,const std::function<bool()>& cancelled,mfq_tensor_backend::Tensor* raw_hidden) {
    namespace tb=mfq_tensor_backend;
    using Tensor=tb::Tensor;
    MFQ_RUNTIME_CHECK(ids.dim()==2 && ids.size(0)==1 &&
        qwen4_layer_major_prefill_eligible(model,ids.size(1)),
        "layer-major prefill requires a fresh single-device Qwen4 prompt and whole-layer cache");
    MFQ_RUNTIME_CHECK(compute_chunk>8,"layer-major compute chunk must exceed decode width");
    auto scope=model.execution_scope();
    ids=model.device_ids(std::move(ids));
    const auto tokens=ids.size(1);
    model.adapter_validate_forward(1,tokens,0,false,false,false);
    model.reset(1);
    model.adapter_begin_forward(raw_hidden!=nullptr);
    auto current=tb::arange(tokens,ids.options());
    if(model.decode_position_delta)current=current+model.decode_position_delta;
    auto prepared=model.adapter_prepare_positions(std::move(current),1,tokens);
    model.adapter_validate_positions(prepared.positions,1,tokens,false);
    auto cpu_ids=ids.to(tb::kCPU,tb::kInt64).contiguous();
    Tensor hidden=tb::empty({1,tokens,model.hidden_size()*model.hc_mult()},
                           ids.options().dtype(tb::kFloat16));
    for(int64_t begin=0;begin<tokens;begin+=compute_chunk) {
        if(cancelled && cancelled())return {};
        const auto count=std::min(compute_chunk,tokens-begin);
        auto input=model.adapter_prepare_hidden(model.embed_tokens(ids.narrow(1,begin,count)),1,count);
        MFQ_RUNTIME_CHECK(input.scalar_type()==hidden.scalar_type() &&
            input.size(-1)==hidden.size(-1),"layer-major embedding geometry changed");
        hidden.narrow(1,begin,count).copy_(input);
    }
    auto& profiler=model.execution->profiler;
    struct ProfileRestore {
        CudaProfiler& profiler;bool enabled;
        ~ProfileRestore(){profiler.enabled=enabled;}
    } restore{profiler,profiler.enabled};
    const auto trace_option=mfq::cuda::runtime_options::prefill_layer_major_trace();
    const bool trace=trace_option && *trace_option!=0;
    const auto batch_option=mfq::cuda::runtime_options::moe_prefill_expert_batch();
    const bool batch_experts=!batch_option || *batch_option!=0;
    const auto* bulk_gr_option=mfq::cuda::env_lookup("MFQ_GR_PREFILL_LAYER_BATCH");
    // Long prompts retain the established memory footprint unless explicitly
    // requested; the combined residual tensors have been verified to 16k.
    const bool bulk_gr=batch_experts && (bulk_gr_option?bulk_gr_option[0]!='0':tokens<=16384);
    const auto* bulk_gdn_option=mfq::cuda::env_lookup("MFQ_GDN_PREFILL_LAYER_BATCH");
#ifdef MFQ_NATIVE_CUDA_RUNTIME
    // Allow 8k chat framing while keeping the extra whole-sequence QKV and
    // gate tensors out of the 16k service path, which also retains MTP weights.
    const bool bulk_gdn=bulk_gr && (bulk_gdn_option?bulk_gdn_option[0]=='1':tokens<=9216);
#else
    const bool bulk_gdn=bulk_gr && bulk_gdn_option && bulk_gdn_option[0]=='1';
#endif
    const qwen4_exp::QwenPrefillReferenceRowsScope reference_rows(batch_experts?std::min<int64_t>(1024,compute_chunk):0);
    const auto batch_tokens_option=mfq::cuda::runtime_options::moe_prefill_expert_batch_tokens();
    const int64_t batch_tokens=batch_tokens_option?*batch_tokens_option:8192;
    const auto* append_tail_option=mfq::cuda::env_lookup("MFQ_MOE_PREFILL_APPEND_TAIL");
    const bool append_tail=append_tail_option && append_tail_option[0]=='1';
    const auto expert_batch_count=[&](int64_t remaining) {
        // Chat framing can leave a few tokens beyond the batch boundary.
        // Absorb at most one reference chunk instead of running a second
        // almost-empty routed GEMM for every layer.
        return append_tail && remaining<=batch_tokens+1024?remaining:std::min(batch_tokens,remaining);
    };
    MFQ_RUNTIME_CHECK(!batch_experts || batch_tokens>=1024,"expert prefill batch must cover at least 1024 tokens");
    const auto audit_option=mfq::cuda::runtime_options::prefill_expert_batch_audit();
    const bool audit_batch=audit_option && *audit_option!=0;
    const auto started=std::chrono::steady_clock::now();
    int layer=0;
    for(auto& block:model.blocks) {
        const auto* decode_reuse_option=mfq::cuda::env_lookup("MFQ_NINT_PREFILL_DECODE_REUSE");
        NintPrefillDecodeCacheScope decode_reuse(!decode_reuse_option || decode_reuse_option[0]=='1');
        Tensor next_hidden=hidden;
        Tensor ffn_inputs,ffn_injection;
        std::vector<Tensor> full_mix;
        Tensor full_branch;
        if(bulk_gr) {
            auto& qwen_block=static_cast<qwen4_exp::Qwen4Block&>(*block);
            Tensor source=hidden;
            if(qwen_block.ple) {
                for(int64_t begin=0;begin<tokens;begin+=compute_chunk) {
                    if(cancelled && cancelled())return {};
                    const auto count=std::min(compute_chunk,tokens-begin);
                    Block::Context context;
                    context.token_ids=ids.narrow(1,begin,count);
                    context.host_token_ids=cpu_ids.narrow(1,begin,count);
                    block->set_token_ids(context.token_ids);
                    block->prefetch_token_ids(context.host_token_ids);
                    auto input=qwen_block.prefill_ple(*model.execution,hidden.narrow(1,begin,count),context);
                    if(begin==0)source=tb::empty(hidden.sizes(),input.options());
                    source.narrow(1,begin,count).copy_(input);
                }
            }
            full_mix=profiler.measure("prefill.gr_attention",[&] {
                return qwen_block.attention_gr.pre(*model.execution,source);
            });
            if(bulk_gdn && qwen_block.gdn) {
                if(cancelled && cancelled())return {};
                const qwen4_exp::QwenGdnPrefillProjectionRowsScope projection_rows(compute_chunk);
                full_branch=profiler.measure("prefill.gdn",[&] {
                    return qwen_block.gdn->forward(*model.execution,full_mix[0],true,0);
                });
            }
        }
        for(int64_t begin=0;begin<tokens;begin+=compute_chunk) {
            if(cancelled && cancelled())return {};
            if(bulk_gdn && static_cast<qwen4_exp::Qwen4Block&>(*block).gdn)continue;
            const auto count=std::min(compute_chunk,tokens-begin);
            profiler.enabled=restore.enabled;
            Block::Context context;
            context.token_ids=ids.narrow(1,begin,count);
            context.host_token_ids=cpu_ids.narrow(1,begin,count);
            context.positions=prepared.positions.narrow(-1,begin,count);
            context.full_positions=prepared.full_positions.narrow(-1,0,begin+count);
            context.cache_position=begin;
            block->set_token_ids(context.token_ids);
            block->prefetch_token_ids(context.host_token_ids);
            auto input=hidden.narrow(1,begin,count);
            Tensor output;
            if(bulk_gr) {
                auto& qwen_block=static_cast<qwen4_exp::Qwen4Block&>(*block);
                auto branch=qwen_block.prefill_attention_core(*model.execution,full_mix[0].narrow(1,begin,count),context);
                if(!full_branch.defined()) {
                    auto shape=branch.sizes().vec();shape[1]=tokens;
                    full_branch=tb::empty(shape,branch.options());
                }
                full_branch.narrow(1,begin,count).copy_(branch);
                continue;
            } else if(batch_experts) {
                auto& qwen_block=static_cast<qwen4_exp::Qwen4Block&>(*block);
                auto mix=qwen_block.prefill_attention(*model.execution,input,context);
                if(!ffn_inputs.defined()) {
                    ffn_inputs=tb::empty({1,tokens,model.hidden_size()},mix[0].options().dtype(tb::kFloat16));
                    ffn_injection=tb::empty({1,tokens,mix[2].size(-1)},mix[2].options());
                }
                // qwen_moe casts the router/FFN source to half at this boundary.
                ffn_inputs.narrow(1,begin,count).copy_(mix[0].to(tb::kFloat16));
                ffn_injection.narrow(1,begin,count).copy_(mix[2]);
                output=std::move(mix[1]);
            } else output=block->forward_context(*model.execution,input,context,model.rope);
            // Retain the original rounding at every layer boundary. The next
            // chunk is independent of this input once its cache writes retire.
            MFQ_RUNTIME_CHECK(output.sizes()==input.sizes(),"layer-major decoder changed hidden shape");
            if(begin==0 && output.scalar_type()!=hidden.scalar_type()) {
                next_hidden=tb::empty(hidden.sizes(),output.options());
                if(trace)std::cout<<"prefill_layer_major_hidden layer="<<layer
                    <<" input_bytes="<<hidden.nbytes()<<" output_bytes="<<next_hidden.nbytes()<<std::endl;
            }
            MFQ_RUNTIME_CHECK(output.scalar_type()==next_hidden.scalar_type(),
                "layer-major decoder output dtype changed within a layer");
            next_hidden.narrow(1,begin,count).copy_(output);
        }
        // Attention's decoded matrices are no longer needed by FFN. Keep
        // reuse within each phase, without retaining both sets at peak memory.
        if(tokens>9216)decode_reuse.clear();
        if(bulk_gr) {
            auto& qwen_block=static_cast<qwen4_exp::Qwen4Block&>(*block);
            auto mix=profiler.measure("prefill.gr_ffn_pre",[&] {
                return qwen_block.ffn_gr.pre_after(*model.execution,full_branch,full_mix);
            });
            ffn_inputs=mix[0].to(tb::kFloat16);ffn_injection=std::move(mix[2]);
            next_hidden=std::move(mix[1]);
            full_mix.clear();full_branch={};
        }
        if(tokens>9216)decode_reuse.clear();
        // The new residual owns the next layer's values. Prior hidden readers
        // have already been submitted before this stream-ordered release.
        if(tokens>9216)hidden={};
        if(batch_experts) {
            if(cancelled && cancelled())return {};
            auto& qwen_block=static_cast<qwen4_exp::Qwen4Block&>(*block);
            profiler.enabled=restore.enabled;
            auto branch=profiler.measure("prefill.ffn",[&] {
                if(expert_batch_count(tokens)==tokens)return qwen_block.ffn(*model.execution,ffn_inputs);
                Tensor result;
                for(int64_t begin=0;begin<tokens;) {
                    if(cancelled && cancelled())return Tensor{};
                    auto count=expert_batch_count(tokens-begin);
                    const auto tail=tokens-begin-count;
                    // A tiny final batch would enter the decode FFN path and
                    // release the shared prefill images halfway through every
                    // layer. Keep both batches in prefill, including chat inputs
                    // just above the expert-batch boundary.
                    if(tail>0 && tail<=8)count-=16-tail;
                    auto part=qwen_block.ffn(*model.execution,ffn_inputs.narrow(1,begin,count));
                    if(!result.defined()) {
                        auto shape=part.sizes().vec();shape[1]=tokens;
                        result=tb::empty(shape,part.options());
                    }
                    result.narrow(1,begin,count).copy_(part);
                    begin+=count;
                }
                return result;
            });
            if(!branch.defined())return {};
            if(audit_batch) {
                auto expected=tb::empty(branch.sizes(),branch.options());
                for(int64_t begin=0;begin<tokens;begin+=compute_chunk) {
                    const auto count=std::min(compute_chunk,tokens-begin);
                    expected.narrow(1,begin,count).copy_(
                        qwen_block.ffn(*model.execution,ffn_inputs.narrow(1,begin,count)));
                }
                auto a=branch.to(tb::kCPU,tb::kFloat32).contiguous();
                auto b=expected.to(tb::kCPU,tb::kFloat32).contiguous();
                double error=0,scale=0,maximum=0;
                for(int64_t i=0;i<a.numel();++i) {
                    const double delta=double(a.data_ptr<float>()[i])-b.data_ptr<float>()[i];
                    error+=delta*delta;scale+=double(b.data_ptr<float>()[i])*b.data_ptr<float>()[i];
                    maximum=std::max(maximum,std::abs(delta));
                }
                std::cout<<"prefill_expert_batch_layer_oracle layer="<<layer<<" tokens="<<tokens
                    <<" relative_rms="<<std::sqrt(error/std::max(1e-30,scale))
                    <<" maximum_absolute="<<maximum<<std::endl;
                // Diagnostic only: continue with the reference so later
                // layers are tested with their original input distribution.
                branch=std::move(expected);
            }
            Tensor post_hidden=next_hidden;
            for(int64_t begin=0;begin<tokens;begin+=compute_chunk) {
                const auto count=std::min(compute_chunk,tokens-begin);
                std::vector<Tensor> mix{{},next_hidden.narrow(1,begin,count),ffn_injection.narrow(1,begin,count)};
                auto output=qwen_block.ffn_gr.post(branch.narrow(1,begin,count),mix);
                if(begin==0 && output.scalar_type()!=next_hidden.scalar_type())
                    post_hidden=tb::empty(next_hidden.sizes(),output.options());
                MFQ_RUNTIME_CHECK(output.scalar_type()==post_hidden.scalar_type(),
                    "expert-batched residual dtype changed");
                post_hidden.narrow(1,begin,count).copy_(output);
            }
            next_hidden=std::move(post_hidden);
        }
        hidden=std::move(next_hidden);
        ++layer;
        if(trace && (layer%4==0 || layer==int(model.blocks.size()))) {
            MFQ_CUDA_CHECK(cudaStreamSynchronize(mfq_get_current_cuda_stream()));
#ifdef MFQ_NATIVE_CUDA_RUNTIME
            const auto memory=mfq::cuda::default_context(ids.get_device())->memory_stats();
            const auto local=mfq::cuda::default_context(ids.get_device())->local_memory_usage();
            cudaMemPool_t pool=nullptr;uint64_t reserved=0,used=0;
            MFQ_CUDA_CHECK(cudaDeviceGetDefaultMemPool(&pool,ids.get_device()));
            MFQ_CUDA_CHECK(cudaMemPoolGetAttribute(pool,cudaMemPoolAttrReservedMemCurrent,&reserved));
            MFQ_CUDA_CHECK(cudaMemPoolGetAttribute(pool,cudaMemPoolAttrUsedMemCurrent,&used));
            std::cout<<"prefill_layer_major_memory tokens="<<tokens<<" chunk="<<compute_chunk
                <<" completed_layers="<<layer<<" owned_bytes="<<memory.allocated
                <<" process_local_bytes="<<local<<" pool_used_bytes="<<used<<" pool_reserved_bytes="<<reserved<<std::endl;
#endif
            std::cout<<"prefill_layer_major_progress tokens="<<tokens<<" chunk="<<compute_chunk
                <<" completed_layers="<<layer<<" total_layers="<<model.blocks.size()
                <<" elapsed_ms="<<std::chrono::duration<double,std::milli>(
                    std::chrono::steady_clock::now()-started).count()<<std::endl;
        }
    }
    profiler.enabled=restore.enabled;
    model.adapter_finish_forward(prepared.full_positions,1,tokens);
    model.cache_pos=tokens;
    if(raw_hidden)*raw_hidden=hidden;
    if(last_only) {
        // Use the same final GR dispatch and rounding as chunk-major prefill.
        const auto begin=((tokens-1)/compute_chunk)*compute_chunk;
        auto finalized=model.finalize_hidden(hidden.narrow(1,begin,tokens-begin),1,tokens-begin);
        return model.last_hidden(std::move(finalized)).unsqueeze(1);
    }
    Tensor finalized;
    for(int64_t begin=0;begin<tokens;begin+=compute_chunk) {
        const auto count=std::min(compute_chunk,tokens-begin);
        auto part=model.finalize_hidden(hidden.narrow(1,begin,count),1,count);
        if(!finalized.defined())finalized=tb::empty({1,tokens,part.size(-1)},part.options());
        finalized.narrow(1,begin,count).copy_(part);
    }
    return finalized;
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
