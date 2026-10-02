#pragma once
#include "models/qwen4_exp/causal_lm.h"
#include "models/qwen4_exp/ngram.h"
#include "../attention_ops.h"
#include "mfq/kernels/cuda/qwen4_exp.h"
#include <array>
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


struct GdnWeights {
    Linear qkv, gate, alpha, beta, output;
    Tensor conv, dt_bias, a_log, norm;
};

class Gdn {
  public:
    Gdn(GdnWeights weights, int64_t key_heads, int64_t value_heads, int64_t width, int64_t kernel,
        double eps, bool silu_gate)
        : w_(std::move(weights)), nk_(key_heads), nv_(value_heads), d_(width), kernel_(kernel),
          eps_(eps), silu_gate_(silu_gate) {
        MFQ_RUNTIME_CHECK(nk_ > 0 && nv_ > 0 && nv_ % nk_ == 0 && d_ > 0 && kernel_ > 1 && eps_ > 0,
                          "invalid Qwen4 GDN head/convolution geometry");
    }
    void reset() {
        conv_ = Tensor();
        state_ = Tensor();
        commit();
    }
    void commit() {
        saved_conv_ = Tensor();
        saved_state_ = Tensor();
    }
    void rollback() {
        MFQ_RUNTIME_CHECK(saved_conv_.defined(), "Qwen4 GDN has no speculative checkpoint");
        conv_ = saved_conv_;
        state_ = saved_state_;
        commit();
    }
    const Tensor &conv_state() const { return conv_; }
    const Tensor &recurrent_state() const { return state_; }
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
                saved_conv_ = conv_;
                saved_state_ = state_;
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
        return mfq::models::gated_delta_attention(
            cache,
            [&] {
                auto projected = w_.qkv(execution, x).to(tb::kFloat32);
                MFQ_RUNTIME_CHECK(projected.sizes().vec() == std::vector<int64_t>({b, t, channels}),
                                  "Qwen4 GDN projection width mismatch");
                return projected;
            },
            [&](Tensor projected) {
                auto joined = tb::cat({previous, projected}, 1).contiguous();
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
                auto gate_input = w_.alpha(execution, x).to(tb::kFloat32).reshape({b, t, nv_}) +
                                  w_.dt_bias.to(tb::kFloat32).reshape({1, 1, nv_});
                auto softplus =
                    tb::clamp_min(gate_input, 0) + tb::log1p(tb::exp(-gate_input.abs()));
                auto decay = (-w_.a_log.to(tb::kFloat32).exp().reshape({1, 1, nv_}) * softplus)
                                 .transpose(1, 2)
                                 .contiguous();
                auto beta = tb::sigmoid(w_.beta(execution, x).to(tb::kFloat32).reshape({b, t, nv_}))
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
                    auto result =
                        gdn_cuda(q, k, v, decay, beta,
                                 initial.defined() ? MfqOptional<Tensor>(initial) : mfq_nullopt);
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
                auto z = w_.gate(execution, x)
                             .reshape({b, t, nv_, d_})
                             .permute({0, 2, 1, 3})
                             .to(tb::kFloat32);
                auto gate = tb::sigmoid(z);
                if (silu_gate_)
                    gate = z * gate;
                return rms_norm(attended, w_.norm, eps_) * gate;
            },
            [&](Tensor normalized) {
                auto result = w_.output(
                    execution,
                    normalized.permute({0, 2, 1, 3}).reshape({b, t, vw}).to(x.scalar_type()));
                return result;
            },
            [&](const auto &convolution, const Tensor &next) {
                conv_ = convolution[3];
                state_ = next;
            });
    }

  private:
    GdnWeights w_;
    int64_t nk_, nv_, d_, kernel_;
    double eps_;
    bool silu_gate_;
    Tensor conv_, state_, saved_conv_, saved_state_;
};

// The upstream PLE hash is a CPU uint64 algorithm, including signed remainder
// after bitwise mixing and EOS segment boundaries. Keep shards independent.
class NgramEmbedding {
  public:
    using Context = mfq::models::qwen4_exp::NgramContext;
    NgramEmbedding(std::vector<Embedding> shards, int64_t rows, int64_t dimension, int64_t ngram,
                   int64_t heads_per_ngram, int64_t eos, std::vector<int64_t> multipliers,
                   std::vector<int64_t> offsets, std::vector<int64_t> vocab)
        : shards_(std::move(shards)), rows_(rows), dimension_(dimension), ngram_(ngram),
          heads_(heads_per_ngram), eos_(eos), multipliers_(std::move(multipliers)),
          offsets_(std::move(offsets)), vocab_(std::move(vocab)) {
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
    Tensor forward(const Tensor &ids, bool cache, Tensor *hashed_ids = nullptr) {
        MFQ_RUNTIME_CHECK(ids.dim() == 2 && ids.size(0) > 0 && ids.size(1) > 0,
                          "Qwen4 ngram IDs must be [B,T]");
        const auto b = ids.size(0), t = ids.size(1), nh = (ngram_ - 1) * heads_;
        auto host = ids.to(tb::kInt64).contiguous().cpu();
        const auto *source = host.data_ptr<int64_t>();
        auto hashes = mfq::models::qwen4_exp::ngram_hashes(
            source, b, t, ngram_, heads_, eos_, multipliers_, offsets_, vocab_, context_, cache);
        auto &global = hashes.ids;
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
        conv_ = saved_conv_;
        embedding_.restore(saved_context_);
        commit();
    }
    const Tensor &conv_state() const { return conv_; }
    Tensor forward(CudaExecutionContext &execution, const Tensor &x, const Tensor &ids, bool cache,
                   int64_t confirmed = 0) {
        MFQ_RUNTIME_CHECK(x.dim() == 3 && ids.dim() == 2 && x.size(0) == ids.size(0) &&
                              x.size(1) == ids.size(1) && x.size(2) == hidden_ * streams_ &&
                              confirmed >= 0 && confirmed <= x.size(1) && (!confirmed || cache),
                          "invalid Qwen4 PLE input");
        return mfq::models::recurrent_window(
            x.size(1), cache, confirmed, saved_,
            [&](int64_t start, int64_t count) {
                return forward_chunk(execution, x.narrow(1, start, count),
                                     ids.narrow(1, start, count), cache);
            },
            [&] {
                saved_ = true;
                saved_conv_ = conv_;
                saved_context_ = embedding_.context();
            },
            [&] { rollback(); },
            [](Tensor first, Tensor second) { return tb::cat({first, second}, 1); });
    }
    Tensor forward_chunk(CudaExecutionContext &execution, const Tensor &x, const Tensor &ids,
                         bool cache) {
        auto context = embedding_.context();
        const auto b = x.size(0), t = x.size(1);
        return mfq::models::qwen4_exp::position_embedding(
            cache, [&] { return embedding_.forward(ids, cache); },
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
            [&](Tensor next) { conv_ = std::move(next); },
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

// QSA uses the same rotate-half mapping for query/key/indexer and pooled keys.
// Positions describe logical token order independently of sparse pool IDs.
class Rotary {
  public:
    Rotary(int64_t dimension, int64_t maximum, double base, std::vector<int64_t> sections = {},
           bool interleaved = false)
        : dimension_(dimension), maximum_(maximum), base_(base), sections_(std::move(sections)),
          interleaved_(interleaved) {
        MFQ_RUNTIME_CHECK(dimension > 0 && dimension % 2 == 0 && maximum > 0 &&
                              std::isfinite(base) && base > 0,
                          "invalid Qwen4 rotary configuration");
        MFQ_RUNTIME_CHECK(
            (sections_.empty() && !interleaved_) ||
                (sections_.size() == 3 &&
                 std::all_of(sections_.begin(), sections_.end(), [](auto n) { return n >= 0; }) &&
                 std::accumulate(sections_.begin(), sections_.end(), int64_t(0)) == dimension / 2),
            "Qwen4 MRoPE sections disagree with rotary width");
    }
    Tensor forward(const Tensor &value, const Tensor &positions) const {
        MFQ_RUNTIME_CHECK(value.is_cuda() && value.dim() == 4 && value.size(-1) >= dimension_ &&
                              positions.is_cuda() && positions.device() == value.device() &&
                              positions.dim() >= 1 && positions.dim() <= 3 &&
                              positions.size(-1) == value.size(2),
                          "Qwen4 RoPE requires [B,H,T,D] with compatible positions");
        const auto b = value.size(0), t = value.size(2), pairs = dimension_ / 2;
        const auto axes = positions.dim() == 1 ? 1 : positions.size(0);
        const auto batches = positions.dim() < 3 ? 1 : positions.size(1);
        MFQ_RUNTIME_CHECK(axes > 0 && (batches == 1 || batches == b),
                          "Qwen4 RoPE position batch mismatch");
        auto input = value.scalar_type() == tb::kFloat32 ? value : value.to(tb::kFloat16);
        auto f = input.to(tb::kFloat32);
        auto pair = tb::arange(pairs, value.options().dtype(tb::kInt64));
        auto frequencies = tb::pow(tb::full({pairs}, base_, f.options()),
                                   -(pair.to(tb::kFloat32) * 2) / double(dimension_));
        auto axis_ids = tb::zeros({pairs}, pair.options());
        if (!sections_.empty()) {
            if (interleaved_) {
                auto remainder = pair.remainder(3);
                auto a1 = (remainder == 1) & (pair < sections_[1] * 3);
                auto a2 = (remainder == 2) & (pair < sections_[2] * 3);
                axis_ids = tb::where(a1, tb::full_like(pair, 1),
                                     tb::where(a2, tb::full_like(pair, 2), axis_ids));
            } else {
                axis_ids = tb::where(pair < sections_[0], axis_ids,
                                     tb::where(pair < sections_[0] + sections_[1],
                                               tb::full_like(pair, 1), tb::full_like(pair, 2)));
            }
            axis_ids = tb::where(axis_ids < axes, axis_ids, tb::zeros_like(axis_ids));
        }
        Tensor cosine, sine;
        for (int64_t axis = 0; axis < std::min<int64_t>(axes, 3); ++axis) {
            auto pos = (positions.dim() == 1 ? positions : positions.select(0, axis))
                           .reshape({batches, t})
                           .to(tb::kInt32)
                           .clamp(0, maximum_ - 1)
                           .to(tb::kFloat32);
            auto angles = pos.unsqueeze(-1) * frequencies;
            auto ca = tb::cos(angles), sa = tb::sin(angles);
            if (axis == 0) {
                cosine = ca;
                sine = sa;
            } else {
                cosine = tb::where(axis_ids == axis, ca, cosine);
                sine = tb::where(axis_ids == axis, sa, sine);
            }
        }
        cosine = cosine.unsqueeze(1);
        sine = sine.unsqueeze(1);
        auto first = f.narrow(-1, 0, pairs), second = f.narrow(-1, pairs, pairs);
        return tb::cat({first * cosine - second * sine, second * cosine + first * sine,
                        f.narrow(-1, dimension_, f.size(-1) - dimension_)},
                       -1)
            .to(input.scalar_type());
    }

  private:
    int64_t dimension_, maximum_;
    double base_;
    std::vector<int64_t> sections_;
    bool interleaved_;
};

struct QsaWeights {
    Linear query, key, value, output, index_query_key;
    Tensor query_norm, key_norm, index_query_norm, index_key_norm;
};
struct QsaConfig {
    int64_t heads, kv_heads, width, index_heads, index_width, pool, budget, maximum;
    double eps;
};

class Qsa {
  public:
    Qsa(QsaWeights weights, QsaConfig config, std::shared_ptr<Rotary> rotary)
        : w_(std::move(weights)), c_(config), rotary_(std::move(rotary)),
          keys_(config.maximum, config.kv_heads * config.width),
          values_(config.maximum, config.kv_heads * config.width),
          index_(config.maximum, config.index_width) {
        MFQ_RUNTIME_CHECK(c_.heads > 0 && c_.kv_heads > 0 && c_.heads % c_.kv_heads == 0 &&
                              c_.width > 0 && c_.index_heads > 0 && c_.pool > 0 && c_.budget > 0 &&
                              c_.budget % c_.pool == 0 && rotary_,
                          "invalid Qwen4 QSA configuration");
    }
    int64_t position() const { return keys_.position(); }
    void reset() {
        keys_.reset();
        values_.reset();
        index_.reset();
    }
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
        MFQ_RUNTIME_CHECK(
            t <= c_.maximum - offset && full_positions.size(-1) == offset + t &&
                (!use_cache || (offset == values_.position() && offset == index_.position())),
            "Qwen4 QSA cache/position geometry mismatch");
        struct Projection {
            Tensor gate, query, key, value, iq, raw;
        };
        return mfq::models::qwen4_exp::sparse_attention(
            offset + t, c_.budget, use_cache,
            [&] {
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
    QsaWeights w_;
    QsaConfig c_;
    std::shared_ptr<Rotary> rotary_;
    SequenceCache keys_, values_, index_;
};
} // namespace mfq::cuda::qwen4_exp
