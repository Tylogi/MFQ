#include "ops.h"
#include "model.h"
#include "mtp.h"
#include "storage/weight_loader.h"
#include "models/common/gated_mlp.h"
#include "models/common/moe.h"
#include "models/common/transformer_layer.h"
#include "../session_codec_impl.h"

namespace mfq::cuda::glm5_next {
using Config = mfq::models::glm5_next::Config;

static Linear headwise(weight_loader::Routed projection,int64_t heads,int64_t output) {
    return [projection=std::move(projection),heads,output](
            CudaExecutionContext& execution, const Tensor& x) {
        MFQ_RUNTIME_CHECK(x.dim()==4 && x.size(2)==heads,"GLM head-wise projection shape mismatch");
        const auto b=x.size(0),t=x.size(1),rows=b*t*heads;
        auto ids=tb::arange(rows,x.options().dtype(tb::kInt32)).remainder(heads).reshape({rows,1});
        return projection(execution,x.reshape({rows,x.size(-1)}),ids).reshape({b,t,heads,output});
    };
}

static Linear dense_ffn(Linear gate, Linear up, Linear down, double limit) {
    return [gate, up, down, limit](CudaExecutionContext &execution, const Tensor &x) {
        auto unfused = [](const auto &...) { return std::optional<Tensor>{}; };
        return mfq::models::gated_mlp(
            x, false, limit, unfused, unfused,
            [&](Tensor input) {
                return std::array<Tensor, 2>{gate(execution, input), up(execution, input)};
            },
            [](Tensor g, Tensor u, mfq::models::GatedActivation, double bound) {
                g = tb::clamp_max(g, bound);
                u = tb::clamp(u, -bound, bound);
                return (g * tb::sigmoid(g)) * u;
            },
            [&](Tensor hidden) { return down(execution, hidden); }, unfused);
    };
}

static Linear glm_moe(weight_loader::Routed gate_up, weight_loader::Routed down,
                      Linear router, Linear shared, Tensor bias, const Config &c) {
    return [gate_up, down, router, shared, bias, c](CudaExecutionContext &execution,
                                                    const Tensor &x) {
        auto source = x.reshape({-1, c.hidden}).to(tb::kFloat16);
        const auto routing = c.routing();
        return mfq::models::mixture_of_experts(
            [&] {
                return router(execution, source.to(tb::kFloat32)).to(tb::kFloat32).contiguous();
            },
            [&](Tensor logits) {
                return moe_topk_cuda(
                    logits, c.topk, routing.activation == mfq::models::RouterActivation::sigmoid,
                    routing.activation == mfq::models::RouterActivation::sqrt_softplus,
                    routing.normalize, routing.delayed_softmax, bias, 1e-20, routing.scale);
            },
            [&](const auto &) { return shared(execution, source); },
            [&](const auto &selected) {
                return mfq::models::routed_experts(
                    [&] { return gate_up(execution, source, selected[0]); },
                    [&](Tensor gu) {
                        auto gate =
                            tb::clamp_max(gu.narrow(-1, 0, c.moe_intermediate), c.swiglu_limit);
                        auto up = tb::clamp(gu.narrow(-1, c.moe_intermediate, c.moe_intermediate),
                                            -c.swiglu_limit, c.swiglu_limit);
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
                return (routed + shared_output).reshape(x.sizes());
            });
    };
}

struct Mhc {
    Tensor function, base, scale;
    std::vector<Tensor> pre(const Tensor &x, const mfq::models::glm5_next::Config &c) const {
        return mfq_glm5_next::mhc_pre(x, function, base, scale, c.sinkhorn, c.hc_eps, c.eps);
    }
};

struct BlockLoader : weight_loader::Loader {
    using KdaWeights = glm5_next::KdaWeights;
    using MlaWeights = glm5_next::MlaWeights;

    static Mhc mhc(Tensor function, Tensor base, Tensor scale) {
        return {std::move(function), std::move(base), std::move(scale)};
    }
    static Tensor concat(const std::vector<Tensor> &parts) { return tb::cat(parts, 0); }
    static auto headwise(weight_loader::Routed projection, int64_t heads, int64_t output) {
        return glm5_next::headwise(std::move(projection), heads, output);
    }
    static auto kda(KdaWeights weights, const Config &c) {
        return std::make_unique<Kda>(std::move(weights), c.kda_heads, c.kda_width, c.kda_kernel,
                                     c.lower_bound, c.eps);
    }
    static auto mla(MlaWeights weights, const Config &c) {
        MlaConfig geometry{c.heads, c.nope, c.latent, c.value_width, c.index_heads, c.index_width,
                           c.pool, c.budget, c.maximum, c.tail, c.eps};
        return std::make_unique<SparseMla>(std::move(weights), geometry);
    }
    static auto gated_mlp(Linear gate, Linear up, Linear down, double limit) {
        return dense_ffn(std::move(gate), std::move(up), std::move(down), limit);
    }
    static auto moe(weight_loader::Routed gate_up, weight_loader::Routed down, Linear router,
                    Linear shared, Tensor bias, const Config &c) {
        return glm_moe(std::move(gate_up), std::move(down), std::move(router), std::move(shared),
                       std::move(bias), c);
    }
};

struct Glm5NextBlock final : Block {
    using Tensor = mfq_tensor_backend::Tensor;
    mfq::models::glm5_next::Config config;
    Mhc attention_hc, ffn_hc;
    Tensor attention_norm, ffn_norm;
    Linear ffn;
    std::unique_ptr<Kda> kda;
    std::unique_ptr<SparseMla> mla;

    Glm5NextBlock(CudaExecutionContext &execution, const mfq::ModelSource &file,
                  const Config &c, int layer) {
        BlockLoader loader{{execution, file, "glm5_next"}};
        mfq::models::glm5_next::load_block(*this, loader, c, layer);
    }
    void reset(int64_t) override {
        if (kda)
            kda->reset();
        if (mla)
            mla->reset();
    }
    bool supports_speculation() const noexcept override { return true; }
    void commit_speculative() override {
        if (kda)
            kda->commit();
    }
    void rollback_speculative(int64_t keep) override {
        if (kda)
            kda->rollback();
        if (mla)
            mla->truncate(keep);
    }
    Tensor execute(CudaExecutionContext &execution, const Tensor &x, int64_t position,
                   int64_t confirmed = 0) {
        if (mla)
            MFQ_RUNTIME_CHECK(mla->position() == position,
                              "GLM MLA/model cache positions diverged");
        return mfq::models::glm5_next::decoder_layer(
            x, bool(kda), [&](const Tensor &hidden) { return attention_hc.pre(hidden, config); },
            [&](Tensor branch, int role) {
                return rms_norm(branch, role == 0 ? attention_norm : ffn_norm, config.eps);
            },
            [&](Tensor branch) { return kda->forward(execution, branch, true, confirmed); },
            [&](Tensor branch) { return mla->forward(execution, branch, true); },
            [](Tensor branch, const Tensor &residual, const auto &mix) {
                return mfq_glm5_next::mhc_post(branch, residual, mix[0], mix[1]);
            },
            [&](const Tensor &hidden) { return ffn_hc.pre(hidden, config); },
            [&](Tensor branch) { return ffn(execution, branch); });
    }
    Tensor forward(CudaExecutionContext &execution, Tensor x, Tensor, int64_t position,
                   const MfqOptional<Tensor> &, const RopeCache &,
                   const MfqOptional<Tensor> & = mfq_nullopt,
                   const MfqOptional<Tensor> &mask = mfq_nullopt) override {
        MFQ_RUNTIME_CHECK(!mask.has_value(), "GLM requires its causal unpadded attention geometry");
        return execute(execution, x, position);
    }
    Tensor forward_context(CudaExecutionContext &execution, Tensor x, const Context &context,
                           const RopeCache &) override {
        return execute(execution, x, context.cache_position, context.confirmed_prefix);
    }
};

Glm5NextMtp::Glm5NextMtp() = default;
Glm5NextMtp::~Glm5NextMtp() = default;
Glm5NextMtp::Glm5NextMtp(Glm5NextMtp&&) noexcept = default;
Glm5NextMtp& Glm5NextMtp::operator=(Glm5NextMtp&&) noexcept = default;

std::optional<Glm5NextMtp> Glm5NextMtp::load_if_present(CudaExecutionContext &execution,
                                                      const mfq::ModelSource &file,
                                                      const mfq::models::glm5_next::Config &main) {
    Glm5NextMtp result;
    result.execution = &execution;
    BlockLoader loader{{execution, file, "glm5_next"}};
    if (!mfq::models::glm5_next::load_predictor(result, loader, main))
        return std::nullopt;
    return result;
}

void Glm5NextMtp::reset(int64_t next_batch) {
    MFQ_RUNTIME_CHECK(next_batch > 0, "GLM5-Next MTP batch must be positive");
    for (auto &layer : layers)
        layer.attention->reset();
    std::fill(lengths.begin(), lengths.end(), 0);
    batch = next_batch;
}

std::pair<Tensor, Tensor> Glm5NextMtp::evaluate(const MtpTarget &target, const Tensor &hidden,
                                       const Tensor &ids, int64_t depth, bool cache,
                                       const Tensor &supplied_positions,
                                       const Tensor &supplied_embeddings) {
    namespace tb = mfq_tensor_backend;
    MFQ_RUNTIME_CHECK(ids.dim() == 2 && ids.size(0) > 0 && ids.size(1) > 0 &&
                          hidden.dim() == 3 && hidden.size(0) == ids.size(0) &&
                          hidden.size(1) == ids.size(1) &&
                          hidden.size(2) == hidden_norm.numel(),
                      "GLM5-Next MTP input geometry mismatch");
    const auto b = ids.size(0), t = ids.size(1);
    auto dtype = tb::kFloat16;
    return mfq::models::projected_predictor(
        *this, b, t, depth, cache, [&](int64_t next) { reset(next); },
        [&] {
            auto embeds =
                supplied_embeddings.defined() ? supplied_embeddings : target.embed(ids);
            MFQ_RUNTIME_CHECK(embeds.sizes().vec() ==
                                  std::vector<int64_t>({b, t, config.hidden}),
                              "GLM5-Next MTP embedding shape mismatch");
            return embeds;
        },
        [&](mfq::models::PredictorCursor cursor) {
            auto current = supplied_positions.defined()
                               ? supplied_positions
                               : tb::arange(cursor.start, cursor.start + t,
                                            ids.options().dtype(tb::kInt32));
            MFQ_RUNTIME_CHECK((current.dim() == 1 || current.dim() == 2) &&
                                  current.size(-1) == t &&
                                  (current.dim() == 1 || current.size(0) == b),
                              "GLM MTP positions require [T] or [B,T]");
            current = current.to(tb::kInt32);
            return current;
        },
        [&](Tensor embeds, const auto &current) {
            auto mask = (current == 0).reshape({current.dim() == 1 ? 1 : b, t, 1});
            auto masked = tb::where(mask, tb::zeros_like(embeds), embeds);
            return rms_norm(masked, embedding_norm, config.eps);
        },
        [&] { return rms_norm(hidden, hidden_norm, config.eps); },
        [&](Tensor e, Tensor streams) { return fusion(*execution, tb::cat({e, streams}, -1)); },
        [&](Tensor x, int64_t layer, const auto &pos) {
            auto &block = layers[layer];
            dtype = x.scalar_type() == tb::kFloat32 ? tb::kFloat32 : tb::kFloat16;
            return mfq::models::pre_norm_layer(
                std::move(x),
                [&](const Tensor &value, int stage) {
                    auto normalized = rms_norm(
                        value, stage == 0 ? block.attention_norm : block.ffn_norm, config.eps);
                    return stage == 0 ? normalized : normalized.to(dtype);
                },
                [&](Tensor value) {
                    return block.attention->forward(*execution, value, cache);
                },
                [&](Tensor value) { return block.ffn(*execution, value); },
                [](Tensor residual, Tensor branch) {
                    return residual.to(tb::kFloat32) + branch.to(tb::kFloat32);
                });
        },
        [&](const Tensor &multi) { return rms_norm(multi, output_norm, config.eps).to(dtype); },
        [&](int64_t layer, const auto &pos) {});
}

Tensor Glm5NextMtp::forward(const MtpTarget &target, Tensor hidden, Tensor ids) {
    return evaluate(target, hidden, ids).first;
}

MtpStep Glm5NextMtp::step(const MtpTarget &target, Tensor hidden, Tensor ids) {
    auto result = evaluate(target, hidden, ids);
    return {std::move(result.first), std::move(result.second)};
}

int64_t Glm5NextMtp::cache_position() const noexcept {
    return lengths.empty() ? 0 : lengths.front();
}

void Glm5NextMtp::trim_cache_to(int64_t position) {
    MFQ_RUNTIME_CHECK(position >= 0 && position <= cache_position(),
                      "GLM5-Next MTP cache trim position is invalid");
    constexpr size_t index = 0;
    if (index < layers.size() && layers[index].attention)
        layers[index].attention->truncate(position);
    lengths[index] = position;
}

bool Glm5NextMtp::teacher_forced_prompt_prime() const noexcept { return false; }

bool Glm5NextMtp::target_bootstrap_decode() const noexcept { return true; }

bool Glm5NextMtp::preserve_output_dtype() const noexcept { return true; }

} // namespace mfq::cuda::glm5_next

namespace mfq::cuda {

void Glm5Model::adapter_validate_load_options() const {
    weight_loader::validate_load_options(*execution);
}

std::unique_ptr<Block> Glm5Model::adapter_load_block(const mfq::ModelSource &source, int layer, int,
                                                     const std::string &) {
    return std::make_unique<glm5_next::Glm5NextBlock>(*execution, source, config, layer);
}

mfq_tensor_backend::Tensor Glm5Model::collapse_hidden(mfq_tensor_backend::Tensor hidden, int64_t,
                                                      int64_t) const {
    return hidden.mean(2);
}

mfq_tensor_backend::Tensor
Glm5Model::normalize_hidden(mfq_tensor_backend::Tensor hidden,
                            const mfq_tensor_backend::Tensor &output_norm, int64_t, int64_t) const {
    return glm5_next::rms_norm(hidden, output_norm, metadata.rms_norm_eps);
}

mfq_tensor_backend::Tensor
Glm5Model::adapter_raw_hidden(const mfq_tensor_backend::Tensor &,
                              const mfq_tensor_backend::Tensor &finalized) const {
    return finalized;
}

mfq_tensor_backend::Tensor Glm5Model::adapter_last_logits(const QuantLinear &lm_head,
                                                          mfq_tensor_backend::Tensor hidden) const {
    return adapter_logits(lm_head, std::move(hidden));
}

mfq_tensor_backend::Tensor Glm5Model::adapter_next_token(const QuantLinear &lm_head,
                                                         mfq_tensor_backend::Tensor hidden) const {
    return mfq_tensor_backend::argmax(adapter_logits(lm_head, std::move(hidden)), -1)
        .to(mfq_tensor_backend::kInt64);
}

} // namespace mfq::cuda

namespace mfq::cuda {

template struct CudaSessionCodec<Glm5Model>;

} // namespace mfq::cuda

namespace mfq::models {
template struct glm5_next::CausalLm<cuda::CudaCausalOps<cuda::Glm5Model>>;
} // namespace mfq::models
