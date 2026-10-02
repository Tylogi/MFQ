#include "ops.h"
#include "model.h"
#include "mtp.h"
#include "storage/weight_loader.h"
#include "models/common/gated_mlp.h"
#include "models/common/moe.h"
#include "models/common/transformer_layer.h"
#include "storage/session_codec.h"

namespace mfq::cuda::qwen4_exp {
using Config = mfq::models::qwen4_exp::Config;

struct Gr {
    Tensor norm, down, up, injection;
    int64_t hidden, streams;
    double eps;
    std::vector<Tensor> pre(const Tensor &x) const {
        return mfq_qwen4_exp::gated_residual_pre(
            x, norm, down, up,
            injection.defined() ? std::optional<Tensor>(injection) : std::nullopt, hidden, streams,
            eps);
    }
    Tensor post(const Tensor &branch, const std::vector<Tensor> &inputs) const {
        return mfq_qwen4_exp::gated_residual_post(branch, inputs[1], inputs[2], streams);
    }
};

static Linear qwen_moe(weight_loader::Routed gate_up, weight_loader::Routed down,
                       Linear router, Linear shared_gate, Linear sg, Linear su, Linear sd,
                       const Config &c) {
    return [gate_up, down, router, shared_gate, sg, su, sd, c](CudaExecutionContext &execution,
                                                               const Tensor &x) {
        auto source = x.reshape({-1, c.hidden}).to(tb::kFloat16);
        const auto routing = c.routing();
        return mfq::models::mixture_of_experts(
            [&] { return router(execution, source).to(tb::kFloat32).contiguous(); },
            [&](Tensor logits) {
                return moe_topk_cuda(
                    logits, c.topk, routing.activation == mfq::models::RouterActivation::sigmoid,
                    routing.activation == mfq::models::RouterActivation::sqrt_softplus,
                    routing.normalize, routing.delayed_softmax, mfq_nullopt, 1e-20, routing.scale);
            },
            [&](const auto &) {
                auto unfused = [](const auto &...) { return std::optional<Tensor>{}; };
                auto shared_output = mfq::models::gated_mlp(
                    source, false, 0.0, unfused, unfused,
                    [&](Tensor input) {
                        return std::array<Tensor, 2>{sg(execution, input), su(execution, input)};
                    },
                    [](Tensor g, Tensor u, mfq::models::GatedActivation, double) {
                        return (g * tb::sigmoid(g)) * u;
                    },
                    [&](Tensor hidden) { return sd(execution, hidden); }, unfused);
                return tb::sigmoid(shared_gate(execution, source)) *
                       shared_output;
            },
            [&](const auto &selected) {
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
                return (routed + shared_output).reshape(x.sizes());
            });
    };
}

struct BlockLoader : weight_loader::Loader {
    using GdnWeights = qwen4_exp::GdnWeights;
    using QsaWeights = qwen4_exp::QsaWeights;
    using PleWeights = qwen4_exp::PleWeights;
    using Embedding = attention_ops::Embedding;

    static Gr residual(Tensor norm, Tensor down, Tensor up, Tensor injection, const Config &c) {
        return {std::move(norm), std::move(down), std::move(up), std::move(injection),
                c.hidden, c.streams, c.eps};
    }
    static auto final_mixer(Gr weights) { return std::make_unique<Gr>(std::move(weights)); }
    static auto gdn(GdnWeights weights, const Config &c) {
        return std::make_unique<Gdn>(std::move(weights), c.key_heads, c.value_heads, c.linear_width,
                                     c.kernel, c.eps, c.silu_gate);
    }
    static auto qsa(QsaWeights weights, const Config &c) {
        QsaConfig geometry{c.heads, c.kv_heads, c.width, c.index_heads, c.index_width,
                           c.pool, c.budget, c.maximum, c.eps};
        auto rotary = std::make_shared<RotaryEmbedding>(c.rotary, c.maximum, c.rope_base, c.sections, c.interleaved);
        return std::make_unique<Qsa>(std::move(weights), geometry, std::move(rotary));
    }
    static auto moe(weight_loader::Routed gate_up, weight_loader::Routed down, Linear router,
                    Linear shared_gate, Linear sg, Linear su, Linear sd, const Config &c) {
        return qwen_moe(std::move(gate_up), std::move(down), std::move(router), std::move(shared_gate),
                        std::move(sg), std::move(su), std::move(sd), c);
    }
    auto embedding(const std::string &name) const {
        auto weight = std::make_shared<QuantLinear>(load_quant_linear(execution, source, name));
        Embedding lookup = [weight](const Tensor &ids) { return quant_embedding_lookup(*weight, ids); };
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
        NgramEmbedding embedding(std::move(shards), rows, width, c.ngram, c.ngram_heads, c.eos,
                                 std::move(multipliers), std::move(offsets), std::move(sizes));
        return std::make_unique<Ple>(std::move(embedding), std::move(weights), c.hidden, c.streams,
                                     c.ngram, c.eps);
    }
    std::unique_ptr<Qwen4Block> block(const Config &c, int layer, bool predictor);
};

struct Qwen4Block final : Block {
    using Tensor = mfq_tensor_backend::Tensor;
    mfq::models::qwen4_exp::Config config;
    Gr attention_gr, ffn_gr;
    Linear ffn;
    std::unique_ptr<Gdn> gdn;
    std::unique_ptr<Qsa> qsa;
    std::unique_ptr<Ple> ple;
    Qwen4Block(CudaExecutionContext &execution, const mfq::ModelSource &file,
               const Config &c, int layer, bool predictor = false) {
        BlockLoader loader{{execution, file, "qwen4_exp"}};
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
                   const Tensor &positions, const Tensor &full_positions, int64_t confirmed = 0) {
        return mfq::models::qwen4_exp::decoder_layer(
            std::move(x), bool(ple), bool(gdn),
            [&](const Tensor &hidden) {
                return ple->forward(execution, hidden, ids, true, confirmed);
            },
            [](Tensor hidden, Tensor positional) { return hidden + positional; },
            [&](const Tensor &hidden) { return attention_gr.pre(hidden); },
            [&](Tensor branch) { return gdn->forward(execution, branch, true, confirmed); },
            [&](Tensor branch) {
                return qsa->forward(execution, branch, positions, full_positions, true);
            },
            [&](Tensor branch, const auto &mix) { return attention_gr.post(branch, mix); },
            [&](const Tensor &hidden) { return ffn_gr.pre(hidden); },
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
                       context.full_positions, context.confirmed_prefix);
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
                [&](const Tensor &value) { return block.attention_gr.pre(value); },
                [](Tensor) -> Tensor { throw std::logic_error("Qwen4 MTP requires QSA"); },
                [&](Tensor branch) {
                    return block.qsa->forward(*execution, branch, pos[0], pos[1], cache);
                },
                [&](Tensor branch, const auto &mix) {
                    return block.attention_gr.post(branch, mix);
                },
                [&](const Tensor &value) { return block.ffn_gr.pre(value); },
                [&](Tensor branch) { return block.ffn(*execution, branch); },
                [&](Tensor branch, const auto &mix) { return block.ffn_gr.post(branch, mix); });
        },
        [&](const Tensor &multi) { return final_mixer->pre(multi)[0]; },
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
    positions = mfq_tensor_backend::Tensor();
    batch = new_batch;
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
    return final_mixer->pre(hidden)[0];
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
