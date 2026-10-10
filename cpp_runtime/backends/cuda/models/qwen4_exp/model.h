#include "../../runtime/execution_options.h"
#pragma once
#include "cuda_execution.h"
#include "mfq_cuda_linear_attention_ops.h"
#include "mfq_cuda_norm_ops.h"
#include "models/qwen4_exp/causal_lm.h"
#include "models/qwen4_exp/ngram.h"
#include "core/attention.h"
#include "core/rope.h"
#include "mfq/kernels/cuda/qwen4_exp.h"
#include "runtime/nint_row_pipeline.h"
#include <array>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <numeric>
#include <set>

namespace mfq::cuda::qwen4_exp {
namespace tb = mfq_tensor_backend;
using attention_ops::Tensor;
using attention_ops::Linear;
using attention_ops::Embedding;
using attention_ops::rms_norm;
using attention_ops::SequenceCache;
using attention_ops::select_pooled_blocks;
using LinearGroup = std::function<std::vector<Tensor>(CudaExecutionContext&, const Tensor&)>;


struct GdnWeights {
    Linear qkv, gate, alpha, beta, output;
    Tensor conv, dt_bias, a_log, norm;
    LinearGroup input_projection;
};

class Gdn {
  public:
    Gdn(GdnWeights weights, int64_t key_heads, int64_t value_heads, int64_t width, int64_t kernel,
        double eps, bool silu_gate, bool fused_decode = true, bool transposed_state = true,
        bool fused_output = true, bool grouped_projection = true, bool fused_preparation = true,
        bool fused_core = true)
        : w_(std::move(weights)), nk_(key_heads), nv_(value_heads), d_(width), kernel_(kernel),
          eps_(eps), silu_gate_(silu_gate), fused_decode_(fused_decode),
          transposed_state_(fused_decode && transposed_state && (width==32 || width==64 || width==128)),
          fused_output_(fused_output), grouped_projection_(grouped_projection), fused_preparation_(fused_preparation),
          fused_core_(fused_core) {
        MFQ_RUNTIME_CHECK(nk_ > 0 && nv_ > 0 && nv_ % nk_ == 0 && d_ > 0 && kernel_ > 1 && eps_ > 0,
                          "invalid Qwen4 GDN head/convolution geometry");
    }
    void reset() {
        conv_ = Tensor();
        state_ = Tensor();
        core_workspace_ = Tensor();
        commit();
    }
    void commit() {
        saved_conv_ = Tensor();
        saved_state_ = Tensor();
    }
    void rollback() {
        MFQ_RUNTIME_CHECK(saved_conv_.defined(), "Qwen4 GDN has no speculative checkpoint");
        if(conv_.defined())conv_.copy_(saved_conv_);else conv_=saved_conv_;
        if(state_.defined())state_.copy_(saved_state_);else state_=saved_state_;
        commit();
    }
    const Tensor &conv_state() const { return conv_; }
    Tensor recurrent_state() const {
        return state_.defined() && transposed_state_ ? state_.transpose(-1,-2) : state_;
    }
    std::vector<Tensor*> graph_state() { return {&conv_,&state_}; }
    void set_grouped_projection(bool enabled) {
        MFQ_RUNTIME_CHECK(!conv_.defined() && !state_.defined(), "reset GDN before switching projection execution");
        grouped_projection_ = enabled;
    }
    Tensor forward(CudaExecutionContext &execution, const Tensor &x, bool cache,
                   int64_t confirmed = 0) {
        MFQ_RUNTIME_CHECK(x.is_cuda() && x.dim() == 3 && x.size(0) > 0,
                          "recurrent attention requires nonempty [B,T,H] input");
        return mfq::models::recurrent_window(
            x.size(1), cache, confirmed, saved_conv_.defined(),
            [&](int64_t start, int64_t count) {
                return forward_chunk(execution, x.narrow(1, start, count), cache);
            },
            [&] {
                saved_conv_ = conv_.clone();
                saved_state_ = state_.clone();
            },
            [&] { rollback(); },
            [](Tensor prefix, Tensor suffix) { return tb::cat({prefix, suffix}, 1); });
    }
    Tensor forward_chunk(CudaExecutionContext &execution, const Tensor &x, bool cache) {
        const auto b = x.size(0), t = x.size(1), kw = nk_ * d_, vw = nv_ * d_,
                   channels = 2 * kw + vw;
        auto options = x.options().dtype(tb::kFloat32);
        MFQ_RUNTIME_CHECK(!cache || !conv_.defined() ||
                              (conv_.size(0) == b && conv_.device() == x.device()),
                          "reset Qwen4 GDN before changing batch/device");
        auto previous =
            cache && conv_.defined() ? conv_ : tb::zeros({b, kernel_ - 1, channels}, options);
        std::vector<Tensor> projections;
        std::array<Tensor,2> prepared_gates;
        const auto prefill_option=mfq::cuda::runtime_options::gdn_prefill_fused();
        const bool prefill_fused=t>1 && fused_preparation_ && d_<=256 && kernel_<=8 &&
            (!prefill_option || *prefill_option!=0);
        if(cache && conv_.defined() && state_.defined() && fused_core_ && fused_decode_ &&
            fused_preparation_ && fused_output_ && transposed_state_ && t==1 &&
            (d_==32 || d_==64 || d_==128) &&
            (x.scalar_type()==tb::kFloat16 || x.scalar_type()==tb::kFloat32)) {
            if(grouped_projection_ && w_.input_projection)projections=w_.input_projection(execution,x);
            else projections={w_.qkv(execution,x),w_.gate(execution,x),w_.alpha(execution,x),w_.beta(execution,x)};
            MFQ_RUNTIME_CHECK(projections.size()==4,"GDN input projection count mismatch");
            if(projections[0].scalar_type()==tb::kFloat16 &&
                projections[1].scalar_type()==projections[2].scalar_type() &&
                projections[2].scalar_type()==projections[3].scalar_type() &&
                (projections[1].scalar_type()==tb::kFloat16 || projections[1].scalar_type()==tb::kFloat32)) {
                auto weight=w_.conv.to(tb::kFloat32).contiguous();
                if(weight.dim()==2 && weight.size(0)==channels && weight.size(1)==kernel_)
                    weight=weight.reshape({channels,1,kernel_});
                if(!core_workspace_.defined())core_workspace_=tb::zeros({b,nv_,d_+1},options);
                auto decoded=gdn_decode_core_cuda(projections[0].contiguous(),projections[1].contiguous(),
                    projections[2].reshape({b,t,nv_}),projections[3].reshape({b,t,nv_}),previous,state_,
                    weight,w_.dt_bias.to(tb::kFloat32).contiguous(),w_.a_log.to(tb::kFloat32).contiguous(),
                    w_.norm.to(tb::kFloat32).contiguous(),nk_,nv_,d_,1e-6,eps_,silu_gate_,
                    x.scalar_type()==tb::kFloat16,true,core_workspace_,
                    [] {const auto value=mfq::cuda::runtime_options::gdn_inplace_state();return !value || *value;}());
                auto output=w_.output(execution,decoded[0]);
                conv_.copy_(decoded[1]);
                if(state_.data_ptr()!=decoded[2].data_ptr())state_.copy_(decoded[2]);
                return output;
            }
        }
        return mfq::models::gated_delta_attention(
            cache,
            [&] {
                Tensor projected;
                if(!projections.empty())projected=projections[0];
                else if(grouped_projection_ && w_.input_projection) {
                    projections=w_.input_projection(execution,x);
                    MFQ_RUNTIME_CHECK(projections.size()==4,"GDN input projection count mismatch");
                    projected=projections[0];
                } else projected=w_.qkv(execution,x);
                MFQ_RUNTIME_CHECK(projected.sizes().vec() == std::vector<int64_t>({b, t, channels}),
                                  "Qwen4 GDN projection width mismatch");
                return projected;
            },
            [&](Tensor projected) {
                if(prefill_fused && projected.scalar_type()==tb::kFloat16) {
                    return execution.profiler.measure("prefill.gdn_conv_qkv",[&] {
                        auto weight=w_.conv.to(tb::kFloat32).contiguous();
                        if(weight.dim()==2 && weight.size(0)==channels && weight.size(1)==kernel_)
                            weight=weight.reshape({channels,1,kernel_});
                        auto qkv=linear_conv_qkv_prefill_cuda(previous,
                            projected.narrow(-1,0,2*kw).contiguous(),
                            projected.narrow(-1,2*kw,vw).contiguous(),weight,
                            tb::empty({0},options),nk_,nv_,d_,d_,1e-6);
                        return std::array<Tensor,4>{qkv[0],qkv[1],qkv[2],qkv[3]};
                    });
                }
                if (fused_decode_ && t == 1 && d_ <= 256 && projected.scalar_type() == tb::kFloat16) {
                    auto weight = w_.conv.to(tb::kFloat32).contiguous();
                    if (weight.dim() == 2 && weight.size(0) == channels && weight.size(1) == kernel_)
                        weight = weight.reshape({channels, 1, kernel_});
                    auto next_conv = previous.clone();
                    auto qk=projected.narrow(-1,0,2*kw).contiguous(),v=projected.narrow(-1,2*kw,vw).contiguous();
                    std::vector<Tensor> qkv;
                    if(fused_preparation_) {
                        auto alpha=projections.empty()?w_.alpha(execution,x):projections[2];
                        auto beta=projections.empty()?w_.beta(execution,x):projections[3];
                        if(alpha.scalar_type()!=beta.scalar_type() ||
                            (alpha.scalar_type()!=tb::kFloat16 && alpha.scalar_type()!=tb::kFloat32)) {
                            alpha=alpha.to(tb::kFloat32);beta=beta.to(tb::kFloat32);
                        }
                        qkv=linear_conv_qkv_gate_decode_cuda(next_conv,qk,v,weight,
                            alpha.reshape({b,t,nv_}),beta.reshape({b,t,nv_}),
                            w_.dt_bias.to(tb::kFloat32).contiguous(),w_.a_log.to(tb::kFloat32).contiguous(),
                            nk_,nv_,d_,d_,1e-6);
                        prepared_gates={qkv[3],qkv[4]};
                    } else qkv=linear_conv_qkv_decode_cuda(next_conv,qk,v,weight,
                        tb::empty({0}, options),nk_,nv_,d_,d_,1e-6);
                    return std::array<Tensor, 4>{qkv[0], qkv[1], qkv[2], next_conv};
                }
                auto joined = tb::cat({previous, projected.to(tb::kFloat32)}, 1).contiguous();
                auto convolved = ssm_conv_silu_cuda(joined, w_.conv.to(tb::kFloat32).contiguous(),
                                                    tb::empty({0}, options), t);
                auto next_conv = joined.narrow(1, t, kernel_ - 1).contiguous();
                const auto normalize = [](const Tensor &v) {
                    return (v / tb::clamp_min((v * v).sum(-1, true).sqrt(), 1e-6)).contiguous();
                };
                auto q = normalize(
                    convolved.narrow(-1, 0, kw).reshape({b, t, nk_, d_}).permute({0, 2, 1, 3}));
                auto k = normalize(
                    convolved.narrow(-1, kw, kw).reshape({b, t, nk_, d_}).permute({0, 2, 1, 3}));
                auto v = convolved.narrow(-1, 2 * kw, vw)
                             .reshape({b, t, nv_, d_})
                             .permute({0, 2, 1, 3})
                             .contiguous();
                return std::array<Tensor, 4>{q, k, v, next_conv};
            },
            [&] {
                if(prepared_gates[0].defined())return prepared_gates;
                auto alpha=projections.empty()?w_.alpha(execution,x):projections[2];
                auto raw_beta=projections.empty()?w_.beta(execution,x):projections[3];
                if ((fused_decode_ && t == 1) || prefill_fused) {
                    auto gates = linear_gate_beta_cuda(
                        alpha.to(tb::kFloat32).reshape({b, t, nv_}),
                        raw_beta.to(tb::kFloat32).reshape({b, t, nv_}),
                        w_.dt_bias.to(tb::kFloat32).contiguous(),
                        w_.a_log.to(tb::kFloat32).contiguous(), true);
                    return std::array<Tensor, 2>{gates[0], gates[1]};
                }
                auto gate_input = alpha.to(tb::kFloat32).reshape({b, t, nv_}) +
                                  w_.dt_bias.to(tb::kFloat32).reshape({1, 1, nv_});
                auto softplus =
                    tb::clamp_min(gate_input, 0) + tb::log1p(tb::exp(-gate_input.abs()));
                auto decay = (-w_.a_log.to(tb::kFloat32).exp().reshape({1, 1, nv_}) * softplus)
                                 .transpose(1, 2)
                                 .contiguous();
                auto beta = tb::sigmoid(raw_beta.to(tb::kFloat32).reshape({b, t, nv_}))
                                .transpose(1, 2)
                                .contiguous();
                return std::array<Tensor, 2>{decay, beta};
            },
            [&](const auto &convolution, const auto &gates) {
                auto q = convolution[0], k = convolution[1], v = convolution[2];
                auto decay = gates[0], beta = gates[1];
                auto initial = cache && state_.defined() ? state_ : Tensor{};
                Tensor attended, next_state;
                if (d_ == 32 || d_ == 64 || d_ == 128) {
                    auto result = execution.profiler.measure("prefill.gdn_recurrence",[&] {
                        return (transposed_state_ ? gdn_transposed_cuda : gdn_cuda)(
                            q, k, v, decay, beta,initial.defined() ? MfqOptional<Tensor>(initial) : mfq_nullopt);
                    });
                    attended = result[0];
                    next_state = result[1];
                } else {
                    auto ids = tb::arange(nv_, options.dtype(tb::kInt64)) / (nv_ / nk_);
                    q = q.index_select(1, ids.to(tb::kInt64));
                    k = k.index_select(1, ids.to(tb::kInt64));
                    next_state = initial.defined() ? initial : tb::zeros({b, nv_, d_, d_}, options);
                    std::vector<Tensor> out;
                    for (int64_t i = 0; i < t; ++i) {
                        auto ki = k.select(2, i), rate = decay.select(2, i).exp().unsqueeze(-1);
                        auto projected_key = (next_state * ki.unsqueeze(-1)).sum(-2);
                        auto delta = (v.select(2, i) - rate * projected_key) *
                                     beta.select(2, i).unsqueeze(-1);
                        next_state = rate.unsqueeze(-1) * next_state +
                                     ki.unsqueeze(-1) * delta.unsqueeze(-2);
                        out.push_back(((next_state * q.select(2, i).unsqueeze(-1)).sum(-2) /
                                       std::sqrt(double(d_)))
                                          .unsqueeze(2));
                    }
                    attended = tb::cat(out, 2);
                }
                return std::array<Tensor, 2>{attended, next_state};
            },
            [&](Tensor attended) {
                auto projected_gate=projections.empty()?w_.gate(execution,x):projections[1].contiguous();
                if(fused_decode_ && fused_output_ && attended.is_contiguous() &&
                    attended.scalar_type()==tb::kFloat32 && projected_gate.is_contiguous() &&
                    (projected_gate.scalar_type()==tb::kFloat16 || projected_gate.scalar_type()==tb::kFloat32) &&
                    (x.scalar_type()==tb::kFloat16 || x.scalar_type()==tb::kFloat32))
                    return gdn_rms_norm_gate_cuda(attended,projected_gate,
                        w_.norm.to(tb::kFloat32).contiguous(),eps_,silu_gate_,x.scalar_type()==tb::kFloat16);
                auto z = projected_gate.reshape({b, t, nv_, d_})
                             .permute({0, 2, 1, 3})
                             .to(tb::kFloat32);
                auto gate = tb::sigmoid(z);
                if (silu_gate_)
                    gate = z * gate;
                return rms_norm(attended, w_.norm, eps_) * gate;
            },
            [&](Tensor normalized) {
                if(normalized.dim()==3)return w_.output(execution,normalized);
                auto result = w_.output(
                    execution,
                    normalized.permute({0, 2, 1, 3}).reshape({b, t, vw}).to(x.scalar_type()));
                return result;
            },
            [&](const auto &convolution, const Tensor &next) {
                if(conv_.defined())conv_.copy_(convolution[3]);else conv_=convolution[3];
                if(state_.defined())state_.copy_(next);else state_=next;
            });
    }

  private:
    GdnWeights w_;
    int64_t nk_, nv_, d_, kernel_;
    double eps_;
    bool silu_gate_, fused_decode_,transposed_state_,fused_output_,grouped_projection_,fused_preparation_,fused_core_;
    Tensor conv_, state_, saved_conv_, saved_state_,core_workspace_;
};

// The upstream PLE hash is a CPU uint64 algorithm, including signed remainder
// after bitwise mixing and EOS segment boundaries. Keep shards independent.
class NgramEmbedding {
  public:
    using Context = mfq::models::qwen4_exp::NgramContext;
    NgramEmbedding(std::vector<Embedding> shards, int64_t rows, int64_t dimension, int64_t ngram,
                   int64_t heads_per_ngram, int64_t eos, std::vector<int64_t> multipliers,
                   std::vector<int64_t> offsets, std::vector<int64_t> vocab,std::shared_ptr<NintRowPipeline> pipeline = {})
        : shards_(std::move(shards)), rows_(rows), dimension_(dimension), ngram_(ngram),
          heads_(heads_per_ngram), eos_(eos), multipliers_(std::move(multipliers)),
          offsets_(std::move(offsets)), vocab_(std::move(vocab)),pipeline_(std::move(pipeline)) {
        MFQ_RUNTIME_CHECK(!shards_.empty() && rows_ > 0 && dimension_ > 0 && ngram_ > 1 &&
                              heads_ > 0 && multipliers_.size() == size_t(ngram_) &&
                              offsets_.size() == size_t((ngram_ - 1) * heads_) &&
                              vocab_.size() == offsets_.size(),
                          "invalid Qwen4 ngram metadata geometry");
        for (size_t h = 0; h < vocab_.size(); ++h)
            MFQ_RUNTIME_CHECK(vocab_[h] > 0 && offsets_[h] >= 0 &&
                                  offsets_[h] <= int64_t(shards_.size()) * rows_ - vocab_[h],
                              "Qwen4 ngram vocabulary lies outside embedding shards");
    }
    void reset() { context_ = {}; }
    const Context &context() const { return context_; }
    void restore(Context context) { context_ = std::move(context); }
    void prefetch(const Tensor& host_ids) {
        if(!pipeline_)return;
        MFQ_RUNTIME_CHECK(host_ids.is_cpu() && host_ids.dim()==2,"PLE prefetch requires host token IDs");
        auto host=host_ids.to(tb::kInt64).contiguous();const auto b=host.size(0),t=host.size(1);
        auto hashes=mfq::models::qwen4_exp::ngram_hashes(host.data_ptr<int64_t>(),b,t,ngram_,heads_,eos_,
            multipliers_,offsets_,vocab_,context_,true);
        pipeline_->issue(hashes.ids,{b,t,(ngram_-1)*heads_});
    }
    std::shared_ptr<NintRowStage> graph_stage(int64_t b,int64_t t,const tb::Device& device) {
        MFQ_RUNTIME_CHECK(pipeline_,"PLE graph requires canonical range rows");
        return pipeline_->make_stage({b,t,(ngram_-1)*heads_},device);
    }
    void upload_graph(const Tensor& host_ids,NintRowStage& stage) {
        MFQ_RUNTIME_CHECK(pipeline_ && host_ids.is_cpu() && host_ids.dim()==2,"PLE graph requires host token IDs");
        auto host=host_ids.to(tb::kInt64).contiguous();const auto b=host.size(0),t=host.size(1);
        auto hashes=mfq::models::qwen4_exp::ngram_hashes(host.data_ptr<int64_t>(),b,t,ngram_,heads_,eos_,
            multipliers_,offsets_,vocab_,context_,true);
        pipeline_->upload(stage,hashes.ids,{b,t,(ngram_-1)*heads_});
        context_=std::move(hashes.next);
    }
    Tensor forward(const Tensor &ids, bool cache, Tensor *hashed_ids = nullptr,const Tensor& host_ids = {}) {
        MFQ_RUNTIME_CHECK(ids.dim() == 2 && ids.size(0) > 0 && ids.size(1) > 0,
                          "Qwen4 ngram IDs must be [B,T]");
        const auto b = ids.size(0), t = ids.size(1), nh = (ngram_ - 1) * heads_;
        auto host = host_ids.defined() ? host_ids.to(tb::kInt64).contiguous() : ids.to(tb::kInt64).contiguous().cpu();
        MFQ_RUNTIME_CHECK(host.is_cpu() && host.sizes()==ids.sizes(),"PLE host token mirror differs");
        const auto *source = host.data_ptr<int64_t>();
        auto hashes = mfq::models::qwen4_exp::ngram_hashes(
            source, b, t, ngram_, heads_, eos_, multipliers_, offsets_, vocab_, context_, cache);
        auto &global = hashes.ids;
        if(pipeline_) {
            auto result=pipeline_->collect(global,{b,t,nh},ids.device());
            if(hashed_ids)*hashed_ids=tb::tensor(global).reshape({b,t,nh}).to(ids.device());
            if(cache)context_=std::move(hashes.next);
            return result.reshape({b,t,nh*dimension_});
        }
        auto global_tensor =
            tb::from_blob(global.data(), {b, t, nh}, tb::TensorOptions().dtype(tb::kInt64))
                .clone()
                .to(ids.device());
        if (hashed_ids)
            *hashed_ids = global_tensor;
        std::set<int64_t> used;
        for (auto value : global)
            used.insert(value / rows_);
        Tensor result;
        for (auto shard : used) {
            auto mask = (global_tensor >= shard * rows_) & (global_tensor < (shard + 1) * rows_);
            auto local =
                tb::where(mask, global_tensor - shard * rows_, tb::zeros_like(global_tensor));
            auto values = shards_.at(shard)(local);
            MFQ_RUNTIME_CHECK(values.sizes().vec() == std::vector<int64_t>({b, t, nh, dimension_}),
                              "Qwen4 ngram embedding shape mismatch");
            values = values * mask.unsqueeze(-1).to(values.scalar_type());
            result = result.defined() ? result + values : values;
        }
        if (cache)
            context_ = std::move(hashes.next);
        return result.reshape({b, t, nh * dimension_});
    }

  private:
    std::vector<Embedding> shards_;
    int64_t rows_, dimension_, ngram_, heads_, eos_;
    std::vector<int64_t> multipliers_, offsets_, vocab_;
    Context context_;
    std::shared_ptr<NintRowPipeline> pipeline_;
};

struct PleWeights {
    Linear key, value;
    Tensor key_norm, query_norm, conv_norm, conv;
};
class Ple {
  public:
    Ple(NgramEmbedding embedding, PleWeights weights, int64_t hidden, int64_t streams,
        int64_t dilation, double eps)
        : embedding_(std::move(embedding)), w_(std::move(weights)), hidden_(hidden),
          streams_(streams), dilation_(dilation), eps_(eps) {}
    void reset() {
        embedding_.reset();
        conv_ = Tensor();
        commit();
    }
    void commit() {
        saved_ = false;
        saved_conv_ = Tensor();
        saved_context_ = {};
    }
    void rollback() {
        MFQ_RUNTIME_CHECK(saved_, "Qwen4 PLE has no speculative checkpoint");
        if(conv_.defined())conv_.copy_(saved_conv_);else conv_=saved_conv_;
        embedding_.restore(saved_context_);
        commit();
    }
    const Tensor &conv_state() const { return conv_; }
    std::vector<Tensor*> graph_state() { return {&conv_}; }
    std::shared_ptr<NintRowStage> graph_stage(int64_t tokens,const tb::Device& device) {
        return embedding_.graph_stage(1,tokens,device);
    }
    void upload_graph(const Tensor& host_ids,NintRowStage& stage) {embedding_.upload_graph(host_ids,stage);}
    void prefetch_tokens(const Tensor& ids) { embedding_.prefetch(ids); }
    Tensor forward(CudaExecutionContext &execution, const Tensor &x, const Tensor &ids, bool cache,
                   int64_t confirmed = 0,const Tensor& host_ids = {}) {
        MFQ_RUNTIME_CHECK(x.dim() == 3 && ids.dim() == 2 && x.size(0) == ids.size(0) &&
                              x.size(1) == ids.size(1) && x.size(2) == hidden_ * streams_ &&
                              confirmed >= 0 && confirmed <= x.size(1) && (!confirmed || cache),
                          "invalid Qwen4 PLE input");
        return mfq::models::recurrent_window(
            x.size(1), cache, confirmed, saved_,
            [&](int64_t start, int64_t count) {
                return forward_chunk(execution, x.narrow(1, start, count),
                                     ids.narrow(1, start, count), cache,
                                     host_ids.defined() ? host_ids.narrow(1,start,count) : Tensor{});
            },
            [&] {
                saved_ = true;
                saved_conv_ = conv_.clone();
                saved_context_ = embedding_.context();
            },
            [&] { rollback(); },
            [](Tensor first, Tensor second) { return tb::cat({first, second}, 1); });
    }
    Tensor forward_chunk(CudaExecutionContext &execution, const Tensor &x, const Tensor &ids,
                         bool cache,const Tensor& host_ids = {},const Tensor& graph_embeddings = {}) {
        auto context = embedding_.context();
        const auto b = x.size(0), t = x.size(1);
        return mfq::models::qwen4_exp::position_embedding(
            cache, [&] { return graph_embeddings.defined() ? graph_embeddings : embedding_.forward(ids, cache,nullptr,host_ids); },
            [&](const Tensor &embeddings) {
                auto key = mfq_qwen4_exp::grouped_rms_norm(w_.key(execution, embeddings),
                                                           w_.key_norm, hidden_, eps_)
                               .reshape({b, t, streams_, hidden_});
                return key;
            },
            [&] {
                auto query = mfq_qwen4_exp::grouped_rms_norm(x, w_.query_norm, hidden_, eps_)
                                 .reshape({b, t, streams_, hidden_});
                return query;
            },
            [&](const Tensor &key, const Tensor &query, const Tensor &embeddings) {
                auto score = (key.to(tb::kFloat32) * query.to(tb::kFloat32)).sum(-1) /
                             std::sqrt(double(hidden_));
                auto sign = (score > 0).to(tb::kFloat32) - (score < 0).to(tb::kFloat32);
                auto root = sign * tb::clamp_min(score.abs(), 1e-6).sqrt();
                auto gated = (tb::sigmoid(root).unsqueeze(-1) *
                              w_.value(execution, embeddings).to(tb::kFloat32).unsqueeze(-2))
                                 .reshape({b, t, streams_ * hidden_});
                return gated;
            },
            [&](const Tensor &gated) {
                return mfq_qwen4_exp::grouped_rms_norm(gated, w_.conv_norm, hidden_, eps_);
            },
            [&](Tensor normalized) {
                return mfq_qwen4_exp::ple_dilated_conv_silu(
                    normalized, w_.conv,
                    cache && conv_.defined() ? std::optional<Tensor>(conv_) : std::nullopt,
                    dilation_);
            },
            [&](Tensor gated, Tensor convolved) { return (gated + convolved).to(x.scalar_type()); },
            [&](Tensor next) { if(conv_.defined())conv_.copy_(next);else conv_=std::move(next); },
            [&] { embedding_.restore(std::move(context)); });
    }

  private:
    NgramEmbedding embedding_;
    PleWeights w_;
    int64_t hidden_, streams_, dilation_;
    double eps_;
    Tensor conv_, saved_conv_;
    NgramEmbedding::Context saved_context_;
    bool saved_ = false;
};

struct QsaWeights {
    Linear query, key, value, output, index_query_key;
    Tensor query_norm, key_norm, index_query_norm, index_key_norm;
    LinearGroup input_projection;
};
struct QsaConfig {
    int64_t heads, kv_heads, width, index_heads, index_width, pool, budget, maximum;
    double eps;
};

class Qsa {
  public:
    Qsa(QsaWeights weights, QsaConfig config, std::shared_ptr<RotaryEmbedding> rotary,
        bool fused_projection = true, bool grouped_projection = true)
        : w_(std::move(weights)), c_(config), rotary_(std::move(rotary)), fused_projection_(fused_projection), grouped_projection_(grouped_projection),
          keys_(config.maximum, config.kv_heads * config.width),
          values_(config.maximum, config.kv_heads * config.width),
          index_(config.maximum, config.index_width) {
        MFQ_RUNTIME_CHECK(c_.heads > 0 && c_.kv_heads > 0 && c_.heads % c_.kv_heads == 0 &&
                              c_.width > 0 && c_.index_heads > 0 && c_.pool > 0 && c_.budget > 0 &&
                              c_.budget % c_.pool == 0 && rotary_,
                          "invalid Qwen4 QSA configuration");
    }
    int64_t position() const { return keys_.position(); }
    void set_grouped_projection(bool enabled) {
        MFQ_RUNTIME_CHECK(!graph_keys_.defined() && position()==0,"reset QSA before switching projection execution");
        grouped_projection_=enabled;
    }
    void set_rotary_fused(bool enabled) {
        MFQ_RUNTIME_CHECK(!graph_keys_.defined() && position() == 0,
                          "reset QSA before switching rotary execution");
        rotary_->set_fused(enabled);
    }
    void leave_graph() {
        if(!graph_keys_.defined())return;
        const auto n=position();
        keys_.replace_storage(graph_keys_.narrow(2,0,n).permute({0,2,1,3}).contiguous().reshape({1,n,c_.kv_heads*c_.width}));
        values_.replace_storage(graph_values_.narrow(2,0,n).permute({0,2,1,3}).contiguous().reshape({1,n,c_.kv_heads*c_.width}));
        graph_keys_={};graph_values_={};graph_pooled_={};
    }
    void reset() {
        keys_.reset();
        values_.reset();
        index_.reset();
        graph_keys_={};graph_values_={};graph_pooled_={};
    }
    void prepare_graph(const Tensor& full_positions) {
        if(graph_keys_.defined())return;
        MFQ_RUNTIME_CHECK(position()>0 && keys_.storage().size(0)==1,
            "QSA graph requires a prefilled single sequence");
        const auto n=position();
        graph_keys_=tb::zeros({1,c_.kv_heads,c_.maximum,c_.width},keys_.storage().options());
        graph_values_=tb::zeros_like(graph_keys_);
        graph_keys_.narrow(2,0,n).copy_(keys_.storage().narrow(1,0,n).reshape({1,n,c_.kv_heads,c_.width}).permute({0,2,1,3}));
        graph_values_.narrow(2,0,n).copy_(values_.storage().narrow(1,0,n).reshape({1,n,c_.kv_heads,c_.width}).permute({0,2,1,3}));
        index_.prepare_fixed();
        graph_pooled_=tb::zeros({1,(c_.maximum+c_.pool-1)/c_.pool,c_.index_width},index_.storage().options());
        const auto pools=n/c_.pool;
        if(pools) {
            auto pooled=index_.storage().narrow(1,0,pools*c_.pool).reshape({1,pools,c_.pool,c_.index_width}).to(tb::kFloat32).mean(2).to(index_.storage().scalar_type());
            pooled=rms_norm(pooled,w_.index_key_norm.to(tb::kFloat32)+1,c_.eps);
            auto starts=tb::arange(pools,full_positions.options().dtype(tb::kInt64))*c_.pool;
            graph_pooled_.narrow(1,0,pools).copy_(rotary_->forward(pooled.unsqueeze(1),full_positions.index_select(-1,starts)).squeeze(1));
        }
    }
    Tensor forward_graph(CudaExecutionContext& execution,const Tensor& hidden,
                         const Tensor& positions,const Tensor& cache_positions) {
        MFQ_RUNTIME_CHECK(graph_keys_.defined() && hidden.size(0)==1 && cache_positions.dim()==1,
            "QSA graph cache is not prepared");
        const auto t=hidden.size(1),pools=graph_pooled_.size(1);
        const bool dense_capacity=c_.maximum<=c_.budget;
        auto p=project(execution,hidden,positions,!dense_capacity,cache_positions,graph_keys_,graph_values_);
        if (!p.cache_written) {
            graph_keys_.index_copy_(2,cache_positions,p.key.permute({0,2,1,3}).to(tb::kFloat16));
            graph_values_.index_copy_(2,cache_positions,p.value.permute({0,2,1,3}).to(tb::kFloat16));
        }
        index_.append_fixed(p.raw,cache_positions);
        if(dense_capacity) {
            const auto columns=c_.budget+c_.pool-1;
            if(fused_projection_ && rotary_->fused() &&
                (hidden.scalar_type()==tb::kFloat16 || hidden.scalar_type()==tb::kFloat32)) {
                auto gated=mfq_qwen4_exp::causal_gqa_attention_gate(p.query,graph_keys_,graph_values_,
                    cache_positions,p.gate,columns,hidden.scalar_type()==tb::kFloat16);
                return w_.output(execution,gated.reshape({1,t,c_.heads*c_.width}));
            }
            auto absolute=cache_positions.reshape({1,t,1});
            auto indices=tb::arange(columns,cache_positions.options()).reshape({1,1,columns}).expand({1,t,columns});
            indices=tb::where(indices<=absolute,indices,tb::full_like(indices,-1)).to(tb::kInt32).contiguous();
            auto attended=mfq_qwen4_exp::sparse_gqa_attention(p.query,graph_keys_,graph_values_,indices);
            auto gated=attended.to(tb::kFloat32)*tb::sigmoid(p.gate.to(tb::kFloat32));
            return w_.output(execution,gated.reshape({1,t,c_.heads*c_.width}).to(hidden.scalar_type()));
        }
        auto block=(cache_positions/c_.pool).to(tb::kInt64);
        auto row=(block.unsqueeze(-1)*c_.pool+tb::arange(c_.pool,block.options())).clamp(0,c_.maximum-1);
        auto pooled=index_.storage().index_select(1,row.reshape({-1})).reshape({1,t,c_.pool,c_.index_width}).to(tb::kFloat32).mean(2).to(index_.storage().scalar_type());
        pooled=rms_norm(pooled,w_.index_key_norm.to(tb::kFloat32)+1,c_.eps);
        auto delta=positions-cache_positions;
        auto block_positions=block*c_.pool+delta;
        pooled=rotary_->forward(pooled.unsqueeze(1),block_positions).squeeze(1);
        graph_pooled_.index_copy_(1,block,pooled);
        auto scores=mfq_qwen4_exp::block_scores(p.iq,graph_pooled_);
        auto absolute=cache_positions.reshape({1,t,1});
        auto ends=(tb::arange(pools,cache_positions.options())*c_.pool+c_.pool-1).reshape({1,1,pools});
        auto visible=ends<=absolute;
        auto ranked=tb::where(visible,scores.to(tb::kFloat32),tb::full_like(scores,-1e30).to(tb::kFloat32));
        const auto count=std::min(c_.budget/c_.pool,pools),columns=c_.budget+c_.pool-1;
        auto indices=std::get<1>(tb::topk(ranked,count,-1,true,false));
        Tensor selected;
        const auto fused_selection=mfq::cuda::runtime_options::qsa_select_fused();
        if(!fused_selection || *fused_selection) {
            selected=mfq_qwen4_exp::qsa_selected_tokens(indices,cache_positions,c_.pool,c_.budget);
        } else {
        auto valid=visible.expand({1,t,pools}).gather(-1,indices).unsqueeze(-1).expand({1,t,count,c_.pool});
        auto expanded=indices.unsqueeze(-1)*c_.pool+tb::arange(c_.pool,indices.options());
        expanded=tb::where(valid,expanded,tb::full_like(expanded,-1));
        selected=tb::full({1,t,columns},-1,cache_positions.options());
        selected.narrow(-1,0,count*c_.pool).copy_(expanded.reshape({1,t,count*c_.pool}));
        if(c_.pool>1) {
            auto tail_count=(absolute+1).remainder(c_.pool);
            auto offsets=tb::arange(c_.pool-1,cache_positions.options()).reshape({1,1,c_.pool-1});
            auto tail=absolute+1-tail_count+offsets;
            selected.narrow(-1,c_.budget,c_.pool-1).copy_(tb::where(offsets<tail_count,tail,tb::full_like(tail,-1)));
        }
        // Dense-budget queries select their complete causal prefix. The same
        // native selected-attention kernel serves both paths inside one graph.
        auto dense=tb::arange(columns,cache_positions.options()).reshape({1,1,columns}).expand({1,t,columns});
        dense=tb::where(dense<=absolute,dense,tb::full_like(dense,-1));
        selected=tb::where(absolute+1<=c_.budget,dense,selected).to(tb::kInt32).contiguous();
        }
        Tensor gated;
        if(fused_projection_ && rotary_->fused() &&
            (hidden.scalar_type()==tb::kFloat16 || hidden.scalar_type()==tb::kFloat32)) {
            const auto sparse_gate=mfq::cuda::runtime_options::qsa_sparse_gate_fused();
            if(!sparse_gate || *sparse_gate)
                gated=mfq_qwen4_exp::sparse_gqa_attention_gate(p.query,graph_keys_,graph_values_,selected,
                    p.gate,hidden.scalar_type()==tb::kFloat16);
            else gated=mfq_qwen4_exp::attention_gate(
                mfq_qwen4_exp::sparse_gqa_attention(p.query,graph_keys_,graph_values_,selected),
                p.gate,hidden.scalar_type()==tb::kFloat16);
        } else {
            auto attended=mfq_qwen4_exp::sparse_gqa_attention(p.query,graph_keys_,graph_values_,selected);
            gated=(attended.to(tb::kFloat32)*tb::sigmoid(p.gate.to(tb::kFloat32))).to(hidden.scalar_type());
        }
        return w_.output(execution,gated.reshape({1,t,c_.heads*c_.width}));
    }
    void advance_graph(int64_t tokens) {keys_.advance_fixed(tokens);values_.advance_fixed(tokens);index_.advance_fixed(tokens);}
    void truncate(int64_t keep) {
        MFQ_RUNTIME_CHECK(keep >= 0 && keep <= keys_.position() && keep <= values_.position() &&
                              keep <= index_.position(),
                          "invalid Qwen4 QSA cache truncation");
        keys_.truncate(keep);
        values_.truncate(keep);
        index_.truncate(keep);
    }
    Tensor forward(CudaExecutionContext &execution, const Tensor &hidden,
                   const Tensor &current_positions, const Tensor &full_positions, bool use_cache,
                   std::vector<Tensor> *selection_trace = nullptr) {
        MFQ_RUNTIME_CHECK(hidden.is_cuda() && hidden.dim() == 3 && hidden.size(0) > 0 &&
                              hidden.size(1) > 0,
                          "Qwen4 QSA requires nonempty [B,T,H] input");
        const auto b = hidden.size(0), t = hidden.size(1), offset = use_cache ? position() : 0;
        const auto prefill_option=mfq::cuda::runtime_options::qsa_prefill_fused();
        const bool fused_prefill=t>8 && (!prefill_option || *prefill_option!=0);
        MFQ_RUNTIME_CHECK(
            t <= c_.maximum - offset && full_positions.size(-1) == offset + t &&
                (!use_cache || (offset == values_.position() && offset == index_.position())),
            "Qwen4 QSA cache/position geometry mismatch");
        return mfq::models::qwen4_exp::sparse_attention(
            offset + t, c_.budget, use_cache,
            [&] {
                if(fused_prefill)
                    return project(execution,hidden,current_positions,offset+t>c_.budget);
                auto pair = w_.query(execution, hidden).reshape({b, t, c_.heads, 2 * c_.width});
                auto gate = pair.narrow(-1, c_.width, c_.width);
                auto query = rms_norm(pair.narrow(-1, 0, c_.width),
                                      w_.query_norm.to(tb::kFloat32) + 1, c_.eps)
                                 .permute({0, 2, 1, 3});
                auto key =
                    rms_norm(w_.key(execution, hidden).reshape({b, t, c_.kv_heads, c_.width}),
                             w_.key_norm.to(tb::kFloat32) + 1, c_.eps)
                        .permute({0, 2, 1, 3});
                auto value = w_.value(execution, hidden).reshape({b, t, c_.kv_heads, c_.width});
                query = rotary_->forward(query, current_positions);
                key = rotary_->forward(key, current_positions).permute({0, 2, 1, 3});
                auto iqk = w_.index_query_key(execution, hidden);
                auto iq = rms_norm(iqk.narrow(-1, 0, c_.index_heads * c_.index_width)
                                       .reshape({b, t, c_.index_heads, c_.index_width}),
                                   w_.index_query_norm.to(tb::kFloat32) + 1, c_.eps)
                              .permute({0, 2, 1, 3});
                iq = rotary_->forward(iq, current_positions).permute({0, 2, 1, 3});
                auto raw = iqk.narrow(-1, c_.index_heads * c_.index_width, c_.index_width);
                return Projection{gate, query, key, value, iq, raw};
            },
            [&](Projection &p) {
                p.key = keys_.append(p.key.reshape({b, t, -1}))
                            .reshape({b, offset + t, c_.kv_heads, c_.width});
                p.value = values_.append(p.value.reshape({b, t, -1}))
                              .reshape({b, offset + t, c_.kv_heads, c_.width});
                p.raw = index_.append(p.raw);
            },
            [&](const Projection &p) {
                return mfq_qwen4_exp::dense_gqa_attention(p.query, p.key.permute({0, 2, 1, 3}),
                                                          p.value.permute({0, 2, 1, 3}), offset);
            },
            [&](const Projection &p) {
                const auto &raw = p.raw;
                const auto pools = (offset + t) / c_.pool;
                auto pooled = raw.narrow(1, 0, pools * c_.pool)
                                  .reshape({b, pools, c_.pool, c_.index_width})
                                  .to(tb::kFloat32)
                                  .mean(2)
                                  .to(raw.scalar_type());
                pooled = rms_norm(pooled, w_.index_key_norm.to(tb::kFloat32) + 1, c_.eps);
                auto starts = tb::arange(pools, raw.options().dtype(tb::kInt64)) * c_.pool;
                auto positions = full_positions.index_select(-1, starts);
                pooled = rotary_->forward(pooled.unsqueeze(1), positions).squeeze(1);
                return pooled;
            },
            [](const Projection &p, const Tensor &pooled) {
                return mfq_qwen4_exp::block_scores(p.iq, pooled);
            },
            [&](const Tensor &scores, const Projection &p, const Tensor &pooled) {
                auto ids =
                    select_pooled_blocks(scores, offset, offset + t, c_.pool, c_.budget, true);
                if (selection_trace)
                    *selection_trace = {scores, ids, p.iq, pooled};
                return ids;
            },
            [](const Projection &p, Tensor ids) {
                return mfq_qwen4_exp::sparse_gqa_attention(p.query, p.key.permute({0, 2, 1, 3}),
                                                           p.value.permute({0, 2, 1, 3}), ids);
            },
            [&](Tensor attended, const Projection &p) {
                auto gated = attended.to(tb::kFloat32) * tb::sigmoid(p.gate.to(tb::kFloat32));
                return w_.output(
                    execution, gated.reshape({b, t, c_.heads * c_.width}).to(hidden.scalar_type()));
            },
            [&] { truncate(offset); });
    }

  private:
    struct Projection {Tensor gate,query,key,value,iq,raw;bool cache_written=false;};
    Tensor project_norm(const Tensor& value,const Tensor& weight) {
        if(value.scalar_type()==tb::kFloat16 || value.scalar_type()==tb::kFloat32)
            return grouped_rms_norm_cuda(value.contiguous(),weight.to(tb::kFloat32).contiguous(),value.size(-1),c_.eps,1.0);
        return rms_norm(value,weight.to(tb::kFloat32)+1,c_.eps);
    }
    Projection project(CudaExecutionContext& execution,const Tensor& hidden,const Tensor& positions,
                       bool index_query_needed=true,const Tensor& cache_positions={},
                       const Tensor& key_cache={},const Tensor& value_cache={}) {
        const auto b=hidden.size(0),t=hidden.size(1);
        std::vector<Tensor> projections;
        if(grouped_projection_ && w_.input_projection) {
            projections=w_.input_projection(execution,hidden);
            MFQ_RUNTIME_CHECK(projections.size()==4,"QSA input projection count mismatch");
        }
        auto pair=(projections.empty()?w_.query(execution,hidden):projections[0]).reshape({b,t,c_.heads,2*c_.width});
        auto gate=pair.narrow(-1,c_.width,c_.width);
        auto key_source=(projections.empty()?w_.key(execution,hidden):projections[1]).reshape({b,t,c_.kv_heads,c_.width});
        auto value=(projections.empty()?w_.value(execution,hidden):projections[2]).reshape({b,t,c_.kv_heads,c_.width});
        const auto supported=[](const Tensor& x) { return x.scalar_type()==tb::kFloat16 || x.scalar_type()==tb::kFloat32; };
        const bool fused=fused_projection_ && rotary_->fused() && supported(pair) && supported(key_source) && supported(value);
        const bool cache_written=fused && key_cache.defined();
        auto iqk=projections.empty()?w_.index_query_key(execution,hidden):projections[3];
        Tensor query,key,iq;
        if(fused) {
            std::vector<Tensor> inputs{pair.narrow(-1,0,c_.width),key_source};
            std::vector<Tensor> weights{w_.query_norm,w_.key_norm};
            if(index_query_needed) {
                auto source=iqk.narrow(-1,0,c_.index_heads*c_.index_width).reshape({b,t,c_.index_heads,c_.index_width});
                if(supported(source)){inputs.push_back(source);weights.push_back(w_.index_query_norm);}
            }
            auto normalized=rotary_->forward_normalized_grouped(inputs,weights,positions,c_.eps,1,
                cache_written?key_cache:Tensor{},cache_written?cache_positions:Tensor{},
                cache_written?value:Tensor{},cache_written?value_cache:Tensor{});
            query=normalized[0];key=normalized[1];
            if(key.defined())key=key.permute({0,2,1,3});
            if(normalized.size()==3)iq=normalized[2].permute({0,2,1,3});
        } else {
            query=project_norm(pair.narrow(-1,0,c_.width),w_.query_norm).permute({0,2,1,3});
            key=project_norm(key_source,w_.key_norm).permute({0,2,1,3});
            query=rotary_->forward(query,positions);key=rotary_->forward(key,positions).permute({0,2,1,3});
        }
        if(index_query_needed && !iq.defined()) {
            auto source=iqk.narrow(-1,0,c_.index_heads*c_.index_width).reshape({b,t,c_.index_heads,c_.index_width});
            if(fused && supported(source))iq=rotary_->forward_normalized(source,w_.index_query_norm,positions,c_.eps).permute({0,2,1,3});
            else {
                iq=project_norm(source,w_.index_query_norm).permute({0,2,1,3});
                iq=rotary_->forward(iq,positions).permute({0,2,1,3});
            }
        }
        return {gate,query,key,value,iq,iqk.narrow(-1,c_.index_heads*c_.index_width,c_.index_width),cache_written};
    }
    QsaWeights w_;
    QsaConfig c_;
    std::shared_ptr<RotaryEmbedding> rotary_;
    bool fused_projection_;
    bool grouped_projection_;
    SequenceCache keys_, values_, index_;
    Tensor graph_keys_,graph_values_,graph_pooled_;
};
} // namespace mfq::cuda::qwen4_exp
