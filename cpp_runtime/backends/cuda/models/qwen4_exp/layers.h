#pragma once

#include "../../runtime/flash_next_loader.h"
#include "model.h"

namespace flash_runtime {
struct Gr {
    Tensor norm,down,up,injection;
    int64_t hidden,streams;
    double eps;
    Gr(const mfq::ModelSource& file,const mfq::models::flash_next::QwenConfig& c,const std::string& p,bool combine=true)
        : norm(dense(file,p+".norm.weight").to(tb::kFloat32)),down(dense(file,p+".down.weight")),
          up(dense(file,p+".up.weight")),hidden(c.hidden),streams(c.streams),eps(c.eps) {
        if (combine) injection=dense(file,p.substr(0,p.size()-4)+".post.inject.weight");
    }
    std::vector<Tensor> pre(const Tensor& x) const {
        return mfq_flash_next::qwen4_gated_residual_pre(x,norm,down,up,
            injection.defined()?std::optional<Tensor>(injection):std::nullopt,hidden,streams,eps);
    }
    Tensor post(const Tensor& branch,const std::vector<Tensor>& inputs) const {
        return mfq_flash_next::qwen4_gated_residual_post(branch,inputs[1],inputs[2],streams);
    }
};

inline Linear qwen_ffn(const mfq::ModelSource& file,const mfq::models::flash_next::QwenConfig& c,int i,const std::string& root="model") {
    const auto p=root+".block."+std::to_string(i)+".mlp";
    auto gate_up=routed_gate_up(file,p,i,c.experts,c.moe_width,c.hidden);
    auto down=routed(file,p+".experts.down.weight",i,c.experts,c.hidden,c.moe_width);
    auto router=linear(file,p+".router.weight"),shared_gate=linear(file,p+".shared_expert.router.weight");
    auto sg=linear(file,p+".shared_expert.gate.weight"),su=linear(file,p+".shared_expert.up.weight"),sd=linear(file,p+".shared_expert.down.weight");
    return [gate_up,down,router,shared_gate,sg,su,sd,c](const Tensor& x) {
        auto source=x.reshape({-1,c.hidden}).to(tb::kFloat16);
        auto selected=moe_topk_cuda(router(source).to(tb::kFloat32).contiguous(),c.topk,
            false,false,c.normalize_routes,false,mfq_nullopt,1e-20,1.0);
        auto gu=gate_up(source,selected[0]);
        auto gate=gu.narrow(-1,0,c.moe_width),up=gu.narrow(-1,c.moe_width,c.moe_width);
        auto pairs=down((gate*tb::sigmoid(gate))*up,selected[0]);
        auto reduced=tb::zeros({source.size(0),c.hidden},source.options().dtype(tb::kFloat32));
        for (int64_t r=0;r<c.topk;++r)
            reduced=reduced+pairs.select(1,r).to(tb::kFloat32)*selected[1].select(1,r).unsqueeze(-1);
        auto g=sg(source),u=su(source);
        auto shared=tb::sigmoid(shared_gate(source))*sd((g*tb::sigmoid(g))*u);
        return (reduced.to(pairs.scalar_type())+shared).reshape(x.sizes());
    };
}

inline std::vector<int64_t> integers(const mfq::ModelSource& file,const std::string& name) {
    const auto& type=require_tensor(file, name).dtype;
    MFQ_RUNTIME_CHECK(type=="I64" || type=="I32","Qwen4 PLE metadata must be a dense integer array: ",name);
    auto host=load_dense_gpu(file,name).to(tb::kInt64).contiguous().cpu();
    MFQ_RUNTIME_CHECK(host.dim()==1,"Qwen4 PLE metadata must be a vector");
    return {host.data_ptr<int64_t>(),host.data_ptr<int64_t>()+host.numel()};
}

inline std::unique_ptr<mfq::flash_next::Ple> qwen_ple(const mfq::ModelSource& file,const mfq::models::flash_next::QwenConfig& c,const std::string& p) {
    std::vector<Linear> shards;
    int64_t rows=0,width=c.hidden/((c.ngram-1)*c.ngram_heads);
    for (int64_t i=0;i<c.shards;++i) {
        const auto name=p+".ngram.shard."+std::to_string(i)+".weight";
        auto weight=std::make_shared<QuantLinear>(load_quant_linear(file,name));
        // QuantLinear reports logical dimensions regardless of storage format.
        std::vector<int64_t> shape{weight->out(),weight->neuron_len()};
        MFQ_RUNTIME_CHECK(shape.size()==2 && shape[1]==width && shape[0]>0 && (!rows || shape[0]==rows),
            "Qwen4 PLE embedding shard dimensions disagree");
        rows=shape[0];
        shards.push_back([weight](const Tensor& ids) {return quant_embedding_lookup(*weight,ids);});
    }
    mfq::flash_next::NgramEmbedding embedding(std::move(shards),rows,width,c.ngram,c.ngram_heads,c.eos,
        integers(file,p+".ngram.layer_multipliers"),integers(file,p+".ngram.head_offsets"),integers(file,p+".ngram.head_vocab_sizes"));
    mfq::flash_next::PleWeights w{linear(file,p+".key.weight"),linear(file,p+".value.weight"),
        dense(file,p+".key_norm.weight"),dense(file,p+".query_norm.weight"),dense(file,p+".conv_norm.weight"),dense(file,p+".conv.weight")};
    return std::make_unique<mfq::flash_next::Ple>(std::move(embedding),std::move(w),c.hidden,c.streams,c.ngram,c.eps);
}
} // namespace flash_runtime

struct Qwen4Block final : Block {
    using Tensor=mfq_tensor_backend::Tensor;
    mfq::models::flash_next::QwenConfig config;
    flash_runtime::Gr attention_gr,ffn_gr;
    mfq::flash_next::Linear ffn;
    std::unique_ptr<mfq::flash_next::Gdn> gdn;
    std::unique_ptr<mfq::flash_next::Qsa> qsa;
    std::unique_ptr<mfq::flash_next::Ple> ple;
    Qwen4Block(const mfq::ModelSource& file,const mfq::models::flash_next::QwenConfig& c,int i,const std::string& root="model")
        : config(c),attention_gr(file,c,root+".block."+std::to_string(i)+".attention.mhc.pre"),
          ffn_gr(file,c,root+".block."+std::to_string(i)+".mlp.mhc.pre"),ffn(flash_runtime::qwen_ffn(file,c,i,root)) {
        using namespace flash_runtime;
        const auto p=root+".block."+std::to_string(i);
        if (root=="model" && c.layer_types.at(i)=="linear_attention") {
            const auto a=p+".linear_attention";
            mfq::flash_next::GdnWeights w{linear(file,a+".qkv.weight"),linear(file,a+".gate.weight"),
                linear(file,a+".alpha.weight"),linear(file,a+".beta.weight"),linear(file,a+".output.weight"),
                dense(file,a+".conv.weight"),dense(file,a+".dt_bias"),dense(file,a+".a"),dense(file,a+".norm.weight")};
            gdn=std::make_unique<mfq::flash_next::Gdn>(std::move(w),c.key_heads,c.value_heads,c.linear_width,c.kernel,c.eps,c.silu_gate);
        } else {
            const auto a=p+".attention";
            mfq::flash_next::QsaWeights w{linear(file,a+".query.weight"),linear(file,a+".key.weight"),linear(file,a+".value.weight"),
                linear(file,a+".output.weight"),linear(file,a+".indexer.query_key.weight"),
                dense(file,a+".query_norm.weight"),dense(file,a+".key_norm.weight"),dense(file,a+".indexer.query_norm.weight"),dense(file,a+".indexer.key_norm.weight")};
            mfq::flash_next::QsaConfig qc{c.heads,c.kv_heads,c.width,c.index_heads,c.index_width,c.pool,c.budget,c.maximum,c.eps};
            auto rotary=std::make_shared<mfq::flash_next::Rotary>(c.rotary,c.maximum,c.rope_base,c.sections,c.interleaved);
            qsa=std::make_unique<mfq::flash_next::Qsa>(std::move(w),qc,std::move(rotary));
        }
        if (root=="model" && std::find(c.ple_layers.begin(),c.ple_layers.end(),i+1)!=c.ple_layers.end()) ple=qwen_ple(file,c,p+".position_embedding");
    }
    void reset(int64_t) override {if (gdn) gdn->reset();if (qsa) qsa->reset();if (ple) ple->reset();}
    bool supports_speculation() const noexcept override {return true;}
    void commit_speculative() override {if (gdn) gdn->commit();if (ple) ple->commit();}
    void rollback_speculative(int64_t keep) override {if (gdn) gdn->rollback();if (qsa) qsa->truncate(keep);if (ple) ple->rollback();}
    Tensor execute(Tensor x,const Tensor& ids,const Tensor& positions,const Tensor& full_positions,int64_t confirmed=0) {
        if (ple) x=x+ple->forward(x,ids,true,confirmed);
        auto first=attention_gr.pre(x);
        auto branch=gdn?gdn->forward(first[0],true,confirmed):qsa->forward(first[0],positions,full_positions,true);
        x=attention_gr.post(branch,first);
        auto second=ffn_gr.pre(x);
        return ffn_gr.post(ffn(second[0]),second);
    }
    Tensor forward(Tensor,Tensor,int64_t,const MfqOptional<Tensor>&,
        const RopeCache&,const MfqOptional<Tensor>& = mfq_nullopt,
        const MfqOptional<Tensor>& = mfq_nullopt) override {
        throw std::runtime_error("Qwen4 block requires the unified model position/PLE input lifecycle");
    }
    Tensor forward_context(
        Tensor x,const Context& context,const RopeCache&) override {
        return execute(std::move(x),context.token_ids,context.positions,
            context.full_positions,context.confirmed_prefix);
    }
};
