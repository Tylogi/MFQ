#pragma once
#include "models/glm5_next/causal_lm.h"
#include "runtime.h"
#include <array>

namespace mfq::cuda::glm5_next {
struct KdaWeights {
    Linear query, key, value, beta, gate_a, gate_b, output;
    Tensor conv, forget_a, forget_b, dt_bias, a_log, output_norm;
};

class Kda {
  public:
    Kda(KdaWeights weights, int64_t heads, int64_t width, int64_t kernel, double lower_bound,
        double eps)
        : w_(std::move(weights)), heads_(heads), width_(width), kernel_(kernel),
          lower_bound_(lower_bound), eps_(eps) {
        MFQ_RUNTIME_CHECK(heads > 0 && width > 0 && kernel > 1 && std::isfinite(lower_bound) &&
                              std::isfinite(eps) && eps > 0,
                          "invalid GLM KDA configuration");
    }
    void reset() {
        conv_ = Tensor();
        recurrent_ = Tensor();
        commit();
    }
    const Tensor &conv_state() const { return conv_; }
    const Tensor &recurrent_state() const { return recurrent_; }
    bool pending() const { return rollback_conv_.defined(); }
    void commit() {
        rollback_conv_ = Tensor();
        rollback_recurrent_ = Tensor();
    }
    void rollback() {
        MFQ_RUNTIME_CHECK(pending(), "GLM KDA has no speculative checkpoint");
        conv_ = rollback_conv_;
        recurrent_ = rollback_recurrent_;
        commit();
    }
    Tensor forward(CudaExecutionContext &execution, const Tensor &hidden, bool use_cache,
                   int64_t confirmed = 0) {
        MFQ_RUNTIME_CHECK(hidden.is_cuda() && hidden.dim() == 3 && hidden.size(0) > 0,
                          "recurrent attention requires nonempty [B,T,H] input");
        return mfq::models::recurrent_window(
            hidden.size(1), use_cache, confirmed, pending(),
            [&](int64_t start, int64_t count) {
                return forward_chunk(execution, hidden.narrow(1, start, count), use_cache);
            },
            [&] {
                rollback_conv_ = conv_;
                rollback_recurrent_ = recurrent_;
            },
            [&] { rollback(); },
            [](Tensor prefix, Tensor suffix) { return tb::cat({prefix, suffix}, 1); });
    }
    Tensor forward_chunk(CudaExecutionContext &execution, const Tensor &hidden, bool use_cache) {
        const auto b = hidden.size(0), t = hidden.size(1), channels = heads_ * width_;
        const auto options = hidden.options().dtype(tb::kFloat32);
        MFQ_RUNTIME_CHECK(!use_cache || !conv_.defined() ||
                              (conv_.size(0) == b && conv_.device() == hidden.device()),
                          "reset GLM KDA before changing batch/device");
        auto previous = use_cache && conv_.defined()
                            ? conv_
                            : tb::zeros({b, kernel_ - 1, 3 * channels}, options);
        return mfq::models::gated_delta_attention(
            use_cache,
            [&] {
                auto projected = tb::cat({w_.query(execution, hidden), w_.key(execution, hidden),
                                          w_.value(execution, hidden)},
                                         -1)
                                     .to(tb::kFloat32);
                MFQ_RUNTIME_CHECK(projected.sizes().vec() ==
                                      std::vector<int64_t>({b, t, 3 * channels}),
                                  "GLM KDA projection shape mismatch");
                return projected;
            },
            [&](Tensor projected) {
                auto joined = tb::cat({previous, projected}, 1).contiguous();
                auto convolved = ssm_conv_silu_cuda(joined, w_.conv.to(tb::kFloat32).contiguous(),
                                                    tb::empty({0}, options), t);
                auto next_conv = joined.narrow(1, t, kernel_ - 1).contiguous();
                const auto heads = [&](int64_t begin) {
                    return convolved.narrow(-1, begin, channels)
                        .reshape({b, t, heads_, width_})
                        .permute({0, 2, 1, 3});
                };
                const auto normalize = [&](const Tensor &x) {
                    return (x / tb::clamp_min((x * x).sum(-1, true).sqrt(), 1e-6)).contiguous();
                };
                auto q = normalize(heads(0)), k = normalize(heads(channels)),
                     v = heads(2 * channels).contiguous();
                return std::array<Tensor, 4>{q, k, v, next_conv};
            },
            [&] {
                auto forget =
                    mfq_glm5_next::kda_forget_gate(hidden, w_.forget_a, w_.forget_b, w_.dt_bias,
                                                   w_.a_log, heads_, width_, lower_bound_)
                        .permute({0, 2, 1, 3})
                        .contiguous();
                auto beta = tb::sigmoid(w_.beta(execution, hidden).to(tb::kFloat32))
                                .transpose(1, 2)
                                .contiguous();
                return std::array<Tensor, 2>{forget, beta};
            },
            [&](const auto &convolution, const auto &gates) {
                auto q = convolution[0], k = convolution[1], v = convolution[2];
                auto forget = gates[0], beta = gates[1];
                Tensor attended, next_recurrent;
                auto initial = use_cache && recurrent_.defined() ? recurrent_ : Tensor{};
                if (width_ == 32 || width_ == 64 || width_ == 128) {
                    auto result =
                        gdn_cuda(q, k, v, forget, beta,
                                 initial.defined() ? MfqOptional<Tensor>(initial) : mfq_nullopt);
                    attended = result[0];
                    next_recurrent = result[1];
                } else {
                    // Public model dimensions use the fused kernel. Keep the exact
                    // recurrence available for small independent architecture fixtures.
                    next_recurrent = initial.defined()
                                         ? initial
                                         : tb::zeros({b, heads_, width_, width_}, options);
                    std::vector<Tensor> steps;
                    for (int64_t i = 0; i < t; ++i) {
                        auto ki = k.select(2, i);
                        next_recurrent = next_recurrent * forget.select(2, i).exp().unsqueeze(-1);
                        auto predicted = (next_recurrent * ki.unsqueeze(-1)).sum(-2);
                        auto delta = (v.select(2, i) - predicted) * beta.select(2, i).unsqueeze(-1);
                        next_recurrent = next_recurrent + ki.unsqueeze(-1) * delta.unsqueeze(-2);
                        steps.push_back((next_recurrent *
                                         (q.select(2, i) / std::sqrt(double(width_))).unsqueeze(-1))
                                            .sum(-2)
                                            .unsqueeze(2));
                    }
                    attended = tb::cat(steps, 2);
                }
                return std::array<Tensor, 2>{attended, next_recurrent};
            },
            [&](Tensor attended) {
                auto gate = w_.gate_b(execution, w_.gate_a(execution, hidden))
                                .reshape({b, t, heads_, width_})
                                .permute({0, 2, 1, 3});
                return rms_norm(attended, w_.output_norm, eps_) *
                       tb::sigmoid(gate.to(tb::kFloat32));
            },
            [&](Tensor normalized) {
                auto result = w_.output(execution, normalized.permute({0, 2, 1, 3})
                                                       .reshape({b, t, channels})
                                                       .to(hidden.scalar_type()));
                return result;
            },
            [&](const auto &convolution, const Tensor &next) {
                conv_ = convolution[3];
                recurrent_ = next;
            });
    }

  private:
    KdaWeights w_;
    int64_t heads_, width_, kernel_;
    double lower_bound_, eps_;
    Tensor conv_, recurrent_, rollback_conv_, rollback_recurrent_;
};

struct MlaWeights {
    Linear query_a, key_value_a, query_b, output, index_query, index_key, index_score;
    // Head-wise absorbed projections accept/return [B,T,heads,width].
    Linear embed_query, unembed_output;
    Tensor query_norm, latent_norm, index_norm, index_bias, index_gate, index_position;
};

struct MlaConfig {
    int64_t heads, nope, latent, value_width, index_heads, index_width, pool, budget, maximum;
    bool tail;
    double eps;
};

class SparseMla {
  public:
    SparseMla(MlaWeights weights, MlaConfig config)
        : w_(std::move(weights)), c_(config), latent_(config.maximum, config.latent),
          index_(config.maximum, 2 * config.index_width) {
        MFQ_RUNTIME_CHECK(c_.heads > 0 && c_.nope > 0 && c_.value_width > 0 && c_.index_heads > 0 &&
                              c_.pool > 0 && c_.budget > 0 && c_.budget % c_.pool == 0,
                          "invalid Flash-Next MLA configuration");
    }
    void reset() {
        latent_.reset();
        index_.reset();
    }
    int64_t position() const { return latent_.position(); }
    void truncate(int64_t keep) {
        MFQ_RUNTIME_CHECK(keep >= 0 && keep <= latent_.position() && keep <= index_.position(),
                          "invalid Flash-Next MLA cache truncation");
        latent_.truncate(keep);
        index_.truncate(keep);
    }
    Tensor forward(CudaExecutionContext &execution, const Tensor &hidden, bool use_cache) {
        MFQ_RUNTIME_CHECK(hidden.is_cuda() && hidden.dim() == 3 && hidden.size(0) > 0 &&
                              hidden.size(1) > 0,
                          "GLM MLA requires nonempty [B,T,H] input");
        const auto b = hidden.size(0), t = hidden.size(1), offset = use_cache ? position() : 0;
        MFQ_RUNTIME_CHECK(offset == (use_cache ? index_.position() : 0) && t <= c_.maximum - offset,
                          "GLM MLA cache position/capacity mismatch");
        struct Projection {
            Tensor rank, query, latent, index;
        };
        const double scale = 1.0 / std::sqrt(double(c_.nope));
        return mfq::models::glm5_next::sparse_mla(
            offset + t, c_.budget, use_cache,
            [&] {
                auto qr = rms_norm(w_.query_a(execution, hidden), w_.query_norm, c_.eps);
                auto query = w_.query_b(execution, qr).reshape({b, t, c_.heads, c_.nope});
                auto latent = rms_norm(w_.key_value_a(execution, hidden).narrow(-1, 0, c_.latent),
                                       w_.latent_norm, c_.eps);
                // Metal's indexer LayerNorm explicitly casts the source/output to F16.
                auto ik = w_.index_key(execution, hidden).to(tb::kFloat16).to(tb::kFloat32);
                auto centered = ik - ik.mean(-1, true);
                ik = (centered * tb::rsqrt((centered * centered).mean(-1, true) + 1e-6) *
                          w_.index_norm.to(tb::kFloat32) +
                      w_.index_bias.to(tb::kFloat32))
                         .to(tb::kFloat16);
                auto gate_dtype = hidden.scalar_type() == w_.index_gate.scalar_type()
                                      ? hidden.scalar_type()
                                      : tb::kFloat32;
                auto gates = tb::matmul(hidden.to(gate_dtype),
                                        w_.index_gate.to(gate_dtype).transpose(-1, -2));
                auto packed = tb::cat({ik.to(gates.scalar_type()), gates}, -1);
                return Projection{qr, query, latent, packed};
            },
            [&](Projection &p) {
                p.latent = latent_.append(p.latent);
                p.index = index_.append(p.index);
            },
            [&](const Projection &p) {
                return w_.embed_query(execution, p.query).permute({0, 2, 1, 3}).to(tb::kFloat32);
            },
            [&](const Tensor &query, const Projection &p) {
                return mfq_glm5_next::dense_mla_attention(query, p.latent, offset, scale);
            },
            [&](const Projection &p) {
                return mfq_glm5_next::kpool_states(
                    p.index.narrow(-1, 0, c_.index_width),
                    p.index.narrow(-1, c_.index_width, c_.index_width), w_.index_position, c_.pool);
            },
            [&](const Projection &p) {
                return w_.index_query(execution, p.rank)
                    .reshape({b, t, c_.index_heads, c_.index_width});
            },
            [&](Tensor query, Tensor pooled) {
                return mfq_glm5_next::kpool_scores(query, pooled,
                                                   w_.index_score(execution, hidden));
            },
            [&](Tensor scores) {
                return select_pooled_blocks(scores, offset, offset + t, c_.pool, c_.budget,
                                            c_.tail);
            },
            [&](const Tensor &query, const Projection &p, Tensor selected) {
                return mfq_glm5_next::sparse_mla_attention(query, p.latent, selected, scale);
            },
            [&](Tensor attended) {
                return w_.unembed_output(execution, attended.to(tb::kFloat16));
            },
            [&](Tensor value) {
                return w_.output(execution, value.reshape({b, t, c_.heads * c_.value_width}));
            },
            [&] {
                latent_.truncate(offset);
                index_.truncate(offset);
            });
    }

  private:
    MlaWeights w_;
    MlaConfig c_;
    SequenceCache latent_, index_;
};
} // namespace mfq::cuda::glm5_next
