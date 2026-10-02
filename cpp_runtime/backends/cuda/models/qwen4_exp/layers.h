#pragma once
#include "models/common/transformer_layer.h"

#include "loader.h"
#include "model.h"

namespace mfq::cuda::qwen4_exp {

struct Gr {
    Tensor norm,down,up,injection;
    int64_t hidden,streams;
    double eps;
    Gr(CudaExecutionContext &execution, const mfq::ModelSource &file,
        const mfq::models::qwen4_exp::Config &c, const std::string &p, bool combine = true)
        : norm(dense(execution, file, p + ".norm.weight").to(tb::kFloat32)),
          down(dense(execution, file, p + ".down.weight")),
          up(dense(execution, file, p + ".up.weight")), hidden(c.hidden), streams(c.streams),
          eps(c.eps) {
        if (combine)
            injection = dense(execution, file, p.substr(0, p.size() - 4) + ".post.inject.weight");
    }
    std::vector<Tensor> pre(const Tensor& x) const {
        return mfq_qwen4_exp::gated_residual_pre(x,norm,down,up,
            injection.defined()?std::optional<Tensor>(injection):std::nullopt,hidden,streams,eps);
    }
    Tensor post(const Tensor &branch, const std::vector<Tensor> &inputs) const {
        return mfq_qwen4_exp::gated_residual_post(branch, inputs[1], inputs[2], streams);
    }
};

inline Linear qwen_ffn(CudaExecutionContext &execution, const mfq::ModelSource &file,
    const mfq::models::qwen4_exp::Config &c, int i, const std::string &root = "model") {
    const auto p = root + ".block." + std::to_string(i) + ".mlp";
    auto gate_up = routed_gate_up(execution, file, p, i, c.experts, c.moe_width, c.hidden);
    auto down =
        routed(execution, file, p + ".experts.down.weight", i, c.experts, c.hidden, c.moe_width);
    auto router = linear(execution, file, p + ".router.weight"),
         shared_gate = linear(execution, file, p + ".shared_expert.router.weight");
    auto sg = linear(execution, file, p + ".shared_expert.gate.weight"),
         su = linear(execution, file, p + ".shared_expert.up.weight"),
         sd = linear(execution, file, p + ".shared_expert.down.weight");
    return [gate_up, down, router, shared_gate, sg, su, sd, c](
               CudaExecutionContext &execution, const Tensor &x) {
        auto source = x.reshape({-1, c.hidden}).to(tb::kFloat16);
        auto selected = moe_topk_cuda(router(execution, source).to(tb::kFloat32).contiguous(),
            c.topk,
            false,
            false,
            c.normalize_routes,
            false,
            mfq_nullopt,
            1e-20,
            1.0);
        auto gu = gate_up(execution, source, selected[0]);
        auto gate = gu.narrow(-1, 0, c.moe_width), up = gu.narrow(-1, c.moe_width, c.moe_width);
        auto pairs = down(execution, (gate * tb::sigmoid(gate)) * up, selected[0]);
        auto reduced = tb::zeros({source.size(0), c.hidden}, source.options().dtype(tb::kFloat32));
        for (int64_t r = 0; r < c.topk; ++r)
            reduced = reduced +
                      pairs.select(1, r).to(tb::kFloat32) * selected[1].select(1, r).unsqueeze(-1);
        auto g = sg(execution, source), u = su(execution, source);
        auto shared =
            tb::sigmoid(shared_gate(execution, source)) * sd(execution, (g * tb::sigmoid(g)) * u);
        return (reduced.to(pairs.scalar_type()) + shared).reshape(x.sizes());
    };
}

inline std::vector<int64_t> integers(CudaExecutionContext& execution, const mfq::ModelSource& file,const std::string& name) {
    const auto& type=require_tensor(file, name).dtype;
    MFQ_RUNTIME_CHECK(type=="I64" || type=="I32","Qwen4 PLE metadata must be a dense integer array: ",name);
    auto host=load_dense_gpu(execution, file,name).to(tb::kInt64).contiguous().cpu();
    MFQ_RUNTIME_CHECK(host.dim()==1,"Qwen4 PLE metadata must be a vector");
    return {host.data_ptr<int64_t>(), host.data_ptr<int64_t>() + host.numel()};
}

inline std::unique_ptr<Ple> qwen_ple(CudaExecutionContext &execution, const mfq::ModelSource &file,
    const mfq::models::qwen4_exp::Config &c, const std::string &p) {
    std::vector<Embedding> shards;
    int64_t rows = 0, width = c.hidden / ((c.ngram - 1) * c.ngram_heads);
    for (int64_t i = 0; i < c.shards; ++i) {
        const auto name = p + ".ngram.shard." + std::to_string(i) + ".weight";
        auto weight = std::make_shared<QuantLinear>(load_quant_linear(execution, file, name));
        std::vector<int64_t> shape{weight->out(), weight->neuron_len()};
        MFQ_RUNTIME_CHECK(
            shape.size() == 2 && shape[1] == width && shape[0] > 0 && (!rows || shape[0] == rows),
            "Qwen4 PLE embedding shard dimensions disagree");
        rows = shape[0];
        shards.push_back(
            [weight](const Tensor &ids) { return quant_embedding_lookup(*weight, ids); });
    }
    NgramEmbedding embedding(std::move(shards),rows,width,c.ngram,c.ngram_heads,c.eos,
        integers(execution,file,p+".ngram.layer_multipliers"),integers(execution,file,p+".ngram.head_offsets"),integers(execution,file,p+".ngram.head_vocab_sizes"));
    PleWeights w{linear(execution,file,p+".key.weight"),linear(execution,file,p+".value.weight"),
        dense(execution,file,p+".key_norm.weight"),dense(execution,file,p+".query_norm.weight"),dense(execution,file,p+".conv_norm.weight"),dense(execution,file,p+".conv.weight")};
    return std::make_unique<Ple>(
        std::move(embedding), std::move(w), c.hidden, c.streams, c.ngram, c.eps);
}

struct Qwen4Block final : Block {
    using Tensor = mfq_tensor_backend::Tensor;
    mfq::models::qwen4_exp::Config config;
    Gr attention_gr, ffn_gr;
    Linear ffn;
    std::unique_ptr<Gdn> gdn;
    std::unique_ptr<Qsa> qsa;
    std::unique_ptr<Ple> ple;
    Qwen4Block(CudaExecutionContext &execution, const mfq::ModelSource &file,
        const mfq::models::qwen4_exp::Config &c, int i, const std::string &root = "model")
        : config(c), attention_gr(execution, file, c,
                         root + ".block." + std::to_string(i) + ".attention.mhc.pre"),
          ffn_gr(execution, file, c, root + ".block." + std::to_string(i) + ".mlp.mhc.pre"),
          ffn(qwen_ffn(execution, file, c, i, root)) {
        const auto p = root + ".block." + std::to_string(i);
        if (root == "model" && c.layer_types.at(i) == "linear_attention") {
            const auto a = p + ".linear_attention";
            GdnWeights w{linear(execution, file, a + ".qkv.weight"),
                linear(execution, file, a + ".gate.weight"),
                linear(execution, file, a + ".alpha.weight"),
                linear(execution, file, a + ".beta.weight"),
                linear(execution, file, a + ".output.weight"),
                dense(execution, file, a + ".conv.weight"),
                dense(execution, file, a + ".dt_bias"),
                dense(execution, file, a + ".a"),
                dense(execution, file, a + ".norm.weight")};
            gdn = std::make_unique<Gdn>(std::move(w),
                c.key_heads,
                c.value_heads,
                c.linear_width,
                c.kernel,
                c.eps,
                c.silu_gate);
        } else {
            const auto a = p + ".attention";
            QsaWeights w{linear(execution, file, a + ".query.weight"),
                linear(execution, file, a + ".key.weight"),
                linear(execution, file, a + ".value.weight"),
                linear(execution, file, a + ".output.weight"),
                linear(execution, file, a + ".indexer.query_key.weight"),
                dense(execution, file, a + ".query_norm.weight"),
                dense(execution, file, a + ".key_norm.weight"),
                dense(execution, file, a + ".indexer.query_norm.weight"),
                dense(execution, file, a + ".indexer.key_norm.weight")};
            QsaConfig qc{c.heads,
                c.kv_heads,
                c.width,
                c.index_heads,
                c.index_width,
                c.pool,
                c.budget,
                c.maximum,
                c.eps};
            auto rotary = std::make_shared<Rotary>(
                c.rotary, c.maximum, c.rope_base, c.sections, c.interleaved);
            qsa = std::make_unique<Qsa>(std::move(w), qc, std::move(rotary));
        }
        if (root=="model" && std::find(c.ple_layers.begin(),c.ple_layers.end(),i+1)!=c.ple_layers.end()) ple=qwen_ple(execution,file,c,p+".position_embedding");
    }
    void reset(int64_t) override {if (gdn) gdn->reset();if (qsa) qsa->reset();if (ple) ple->reset();}
    bool supports_speculation() const noexcept override {return true;}
    void commit_speculative() override {if (gdn) gdn->commit();if (ple) ple->commit();}
    void rollback_speculative(int64_t keep) override {if (gdn) gdn->rollback();if (qsa) qsa->truncate(keep);if (ple) ple->rollback();}
    Tensor execute(CudaExecutionContext &execution, Tensor x, const Tensor &ids,
        const Tensor &positions, const Tensor &full_positions, int64_t confirmed = 0) {
        if (ple)
            x = x + ple->forward(execution, x, ids, true, confirmed);
        return mfq::models::hyperconnection_layer(std::move(x), [&](const Tensor &hidden) {
            return attention_gr.pre(hidden);
        }, [&](const auto &mix) {
            return gdn ? gdn->forward(execution, mix[0], true, confirmed)
                       : qsa->forward(execution, mix[0], positions, full_positions, true);
        }, [&](Tensor branch, const Tensor &, const auto &mix) {
            return attention_gr.post(branch, mix);
        }, [&](const Tensor &hidden, const auto &) {
            return ffn_gr.pre(hidden);
        }, [&](const auto &mix) {
            return ffn(execution, mix[0]);
        }, [&](Tensor branch, const Tensor &, const auto &mix) {
            return ffn_gr.post(branch, mix);
        }, [](const auto &) {});
    }
    Tensor forward(
        CudaExecutionContext&,
        Tensor,Tensor,int64_t,const MfqOptional<Tensor>&,
        const RopeCache&,const MfqOptional<Tensor>& = mfq_nullopt,
        const MfqOptional<Tensor>& = mfq_nullopt) override {
        throw std::runtime_error("Qwen4 block requires the unified model position/PLE input lifecycle");
    }
    Tensor forward_context(
        CudaExecutionContext& execution,
        Tensor x,const Context& context,const RopeCache&) override {
        return execute(execution,std::move(x),context.token_ids,context.positions,
            context.full_positions,context.confirmed_prefix);
    }
};

} // namespace mfq::cuda::qwen4_exp
