#include "ops.h"
#include "model.h"
#include "mtp.h"
#include "storage/weight_loader.h"
#include "models/common/gated_mlp.h"
#include "models/common/moe.h"
#include "models/common/transformer_layer.h"
#include "../session_codec_impl.h"

namespace mfq::cuda::glm5_next {
using weight_loader::linear;
using weight_loader::dense;
using weight_loader::routed;
using weight_loader::routed_gate_up;

static Linear headwise(weight_loader::Routed projection,int64_t heads,int64_t output) {
    return [projection=std::move(projection),heads,output](
            CudaExecutionContext& execution, const Tensor& x) {
        MFQ_RUNTIME_CHECK(x.dim()==4 && x.size(2)==heads,"GLM head-wise projection shape mismatch");
        const auto b=x.size(0),t=x.size(1),rows=b*t*heads;
        auto ids=tb::arange(rows,x.options().dtype(tb::kInt32)).remainder(heads).reshape({rows,1});
        return projection(execution,x.reshape({rows,x.size(-1)}),ids).reshape({b,t,heads,output});
    };
}

static Linear dense_ffn(CudaExecutionContext &execution, const mfq::ModelSource &file,
                        const std::string &p, double limit) {
    auto gate = linear(execution, file, p + ".gate.weight"),
         up = linear(execution, file, p + ".up.weight"),
         down = linear(execution, file, p + ".down.weight");
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

static Linear glm_ffn(CudaExecutionContext &execution, const mfq::ModelSource &file,
                      const mfq::models::glm5_next::Config &c, int i,
                      const std::string &root = "model") {
    const auto p = root + ".block." + std::to_string(i) + ".mlp";
    if (c.dense_layer(i, root != "model"))
        return dense_ffn(execution, file, p, c.swiglu_limit);
    auto gate_up = routed_gate_up(execution, file, p, i, c.experts, c.moe_intermediate, c.hidden, "glm5_next");
    auto down = routed(execution, file, p + ".experts.down.weight", i, c.experts, c.hidden,
                       c.moe_intermediate, "glm5_next");
    auto router = linear(execution, file, p + ".router.weight"),
         shared = dense_ffn(execution, file, p + ".shared_expert", c.swiglu_limit);
    auto bias = dense(execution, file, p + ".router.bias").to(tb::kFloat32).contiguous();
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
    Mhc(CudaExecutionContext &execution, const mfq::ModelSource &file, const std::string &p)
        : function(dense(execution, file, p + ".function")),
          base(dense(execution, file, p + ".base")), scale(dense(execution, file, p + ".scale")) {}
    std::vector<Tensor> pre(const Tensor &x, const mfq::models::glm5_next::Config &c) const {
        return mfq_glm5_next::mhc_pre(x, function, base, scale, c.sinkhorn, c.hc_eps, c.eps);
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
                  const mfq::models::glm5_next::Config &c, int i)
        : config(c),
          attention_hc(execution, file, "model.block." + std::to_string(i) + ".attention.mhc.pre"),
          ffn_hc(execution, file, "model.block." + std::to_string(i) + ".mlp.mhc.pre") {
        const auto p = "model.block." + std::to_string(i);
        attention_norm = dense(execution, file, p + ".attention.norm.weight");
        ffn_norm = dense(execution, file, p + ".mlp.norm.weight");
        ffn = glm_ffn(execution, file, c, i);
        if (c.linear_layer(i)) {
            const auto a = p + ".linear_attention";
            KdaWeights w{linear(execution, file, a + ".query.weight"),
                         linear(execution, file, a + ".key.weight"),
                         linear(execution, file, a + ".value.weight"),
                         linear(execution, file, a + ".beta.weight"),
                         linear(execution, file, a + ".gate_a.weight"),
                         linear(execution, file, a + ".gate_b.weight"),
                         linear(execution, file, a + ".output.weight"),
                         tb::cat({dense(execution, file, a + ".query_conv.weight"),
                                  dense(execution, file, a + ".key_conv.weight"),
                                  dense(execution, file, a + ".value_conv.weight")},
                                 0),
                         dense(execution, file, a + ".forget_a.weight"),
                         dense(execution, file, a + ".forget_b.weight"),
                         dense(execution, file, a + ".dt_bias"),
                         dense(execution, file, a + ".a"),
                         dense(execution, file, a + ".output_norm.weight")};
            kda = std::make_unique<Kda>(std::move(w), c.kda_heads, c.kda_width, c.kda_kernel,
                                        c.lower_bound, c.eps);
        } else {
            const auto a = p + ".attention";
            MlaWeights w{linear(execution, file, a + ".query_a.weight"),
                         linear(execution, file, a + ".key_value_a.weight"),
                         linear(execution, file, a + ".query_b.weight"),
                         linear(execution, file, a + ".output.weight"),
                         linear(execution, file, a + ".indexer.query.weight"),
                         linear(execution, file, a + ".indexer.key.weight"),
                         linear(execution, file, a + ".indexer.score.weight"),
                         headwise(routed(execution, file, a + ".latent.query_embedding.weight", i,
                                         c.heads, c.latent, c.nope, "glm5_next"),
                                  c.heads, c.latent),
                         headwise(routed(execution, file, a + ".latent.output_unembedding.weight",
                                         i, c.heads, c.value_width, c.latent, "glm5_next"),
                                  c.heads, c.value_width),
                         dense(execution, file, a + ".query_a_norm.weight"),
                         dense(execution, file, a + ".key_value_a_norm.weight"),
                         dense(execution, file, a + ".indexer.key_norm.weight"),
                         dense(execution, file, a + ".indexer.key_norm.bias"),
                         dense(execution, file, a + ".indexer.pool.gate"),
                         dense(execution, file, a + ".indexer.pool.position")};
            MlaConfig mc{c.heads, c.nope,   c.latent,  c.value_width, c.index_heads, c.index_width,
                         c.pool,  c.budget, c.maximum, c.tail,        c.eps};
            mla = std::make_unique<SparseMla>(std::move(w), mc);
        }
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
    const bool any = std::any_of(file.tensors().begin(), file.tensors().end(),
                                 [](const mfq::TensorMetadata &tensor) {
                                     return tensor.name.rfind("predictor.", 0) == 0;
                                 });
    const auto count = main.predictor_layers;
    if (!has_tensor(file, "predictor.embedding_norm.weight") || count <= 0) {
        MFQ_RUNTIME_CHECK(
            !any, "GLM5-Next model source contains an incomplete or undeclared MTP head");
        return std::nullopt;
    }
    Glm5NextMtp result;
    result.execution = &execution;
    result.config = main;
    result.embedding_norm =
        dense(execution, file, "predictor.embedding_norm.weight").to(tb::kFloat32);
    result.hidden_norm =
        dense(execution, file, "predictor.hidden_norm.weight").to(tb::kFloat32);
    result.output_norm = dense(execution, file, "predictor.output_norm.weight");
    result.fusion = linear(execution, file, "predictor.fusion.weight");
    for (int64_t i = 0; i < count; ++i) {
        const auto p = "predictor.block." + std::to_string(i), a = p + ".attention";
        Layer layer;
        layer.attention_norm = dense(execution, file, a + ".norm.weight");
        layer.ffn_norm = dense(execution, file, p + ".mlp.norm.weight");
        layer.ffn = glm_ffn(execution, file, main, int(i), "predictor");
        MlaWeights w{linear(execution, file, a + ".query_a.weight"),
                     linear(execution, file, a + ".key_value_a.weight"),
                     linear(execution, file, a + ".query_b.weight"),
                     linear(execution, file, a + ".output.weight"),
                     linear(execution, file, a + ".indexer.query.weight"),
                     linear(execution, file, a + ".indexer.key.weight"),
                     linear(execution, file, a + ".indexer.score.weight"),
                     headwise(routed(execution, file, a + ".latent.query_embedding.weight",
                                     int(i), main.heads, main.latent, main.nope, "glm5_next"),
                              main.heads, main.latent),
                     headwise(routed(execution, file, a + ".latent.output_unembedding.weight",
                                     int(i), main.heads, main.value_width, main.latent, "glm5_next"),
                              main.heads, main.value_width),
                     dense(execution, file, a + ".query_a_norm.weight"),
                     dense(execution, file, a + ".key_value_a_norm.weight"),
                     dense(execution, file, a + ".indexer.key_norm.weight"),
                     dense(execution, file, a + ".indexer.key_norm.bias"),
                     dense(execution, file, a + ".indexer.pool.gate"),
                     dense(execution, file, a + ".indexer.pool.position")};
        MlaConfig mc{main.heads,       main.nope,        main.latent, main.value_width,
                     main.index_heads, main.index_width, main.pool,   main.budget,
                     main.maximum,     main.tail,        main.eps};
        layer.attention = std::make_unique<SparseMla>(std::move(w), mc);
        result.layers.push_back(std::move(layer));
    }
    MFQ_RUNTIME_CHECK(
        result.embedding_norm.dim() == 1 && result.embedding_norm.numel() == main.hidden &&
            result.hidden_norm.dim() == 1 && result.hidden_norm.numel() == main.hidden &&
            result.output_norm.numel() == main.hidden,
        "GLM5-Next MTP normalization width disagrees with backbone");
    result.lengths.resize(count, 0);
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
