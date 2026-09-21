#pragma once

#include "../../runtime/flash_next_loader.h"
#include "model.h"

namespace flash_runtime {
inline Linear glm_ffn(const mfq::ModelSource& file,const mfq::models::flash_next::GlmConfig& c,int i,const std::string& root="model") {
    const auto p=root+".block."+std::to_string(i)+".mlp";
    if (root=="model" && c.mlp_types.at(i)=="dense") return dense_ffn(file,p,c.swiglu_limit);
    auto gate_up=routed_gate_up(file,p,i,c.experts,c.moe_intermediate,c.hidden);
    auto down=routed(file,p+".experts.down.weight",i,c.experts,c.hidden,c.moe_intermediate);
    auto router=linear(file,p+".router.weight"),shared=dense_ffn(file,p+".shared_expert",c.swiglu_limit);
    auto bias=dense(file,p+".router.bias").to(tb::kFloat32).contiguous();
    return [gate_up,down,router,shared,bias,c](const Tensor& x) {
        auto source=x.reshape({-1,c.hidden}).to(tb::kFloat16);
        auto selected=moe_topk_cuda(router(source.to(tb::kFloat32)).to(tb::kFloat32).contiguous(),
            c.topk,true,false,c.normalize_routes,false,bias,1e-20,c.router_scale);
        auto gu=gate_up(source,selected[0]);
        auto gate=tb::clamp_max(gu.narrow(-1,0,c.moe_intermediate),c.swiglu_limit);
        auto up=tb::clamp(gu.narrow(-1,c.moe_intermediate,c.moe_intermediate),-c.swiglu_limit,c.swiglu_limit);
        auto pairs=down((gate*tb::sigmoid(gate))*up,selected[0]);
        // Metal accumulates routes in FP32, then returns the pair dtype.
        auto reduced=tb::zeros({source.size(0),c.hidden},source.options().dtype(tb::kFloat32));
        for (int64_t r=0;r<c.topk;++r)
            reduced=reduced+pairs.select(1,r).to(tb::kFloat32)*selected[1].select(1,r).unsqueeze(-1);
        return (reduced.to(pairs.scalar_type())+shared(source)).reshape(x.sizes());
    };
}

struct Mhc {
    Tensor function,base,scale;
    explicit Mhc(const mfq::ModelSource& file,const std::string& p)
        : function(dense(file,p+".function")),base(dense(file,p+".base")),scale(dense(file,p+".scale")) {}
    std::vector<Tensor> pre(const Tensor& x,const mfq::models::flash_next::GlmConfig& c) const {
        return mfq_flash_next::glm5_mhc_pre(x,function,base,scale,c.sinkhorn,c.hc_eps,c.eps);
    }
};
} // namespace flash_runtime

struct Glm5NextBlock final : Block {
    using Tensor=mfq_tensor_backend::Tensor;
    mfq::models::flash_next::GlmConfig config;
    flash_runtime::Mhc attention_hc,ffn_hc;
    Tensor attention_norm,ffn_norm;
    mfq::flash_next::Linear ffn;
    std::unique_ptr<mfq::flash_next::Kda> kda;
    std::unique_ptr<mfq::flash_next::SparseMla> mla;

    Glm5NextBlock(const mfq::ModelSource& file,const mfq::models::flash_next::GlmConfig& c,int i)
        : config(c),attention_hc(file,"model.block."+std::to_string(i)+".attention.mhc.pre"),
          ffn_hc(file,"model.block."+std::to_string(i)+".mlp.mhc.pre") {
        using namespace flash_runtime;
        const auto p="model.block."+std::to_string(i);
        attention_norm=dense(file,p+".attention.norm.weight"); ffn_norm=dense(file,p+".mlp.norm.weight");
        ffn=glm_ffn(file,c,i);
        if (c.layer_types.at(i)=="linear_attention") {
            const auto a=p+".linear_attention";
            mfq::flash_next::KdaWeights w{linear(file,a+".query.weight"),linear(file,a+".key.weight"),
                linear(file,a+".value.weight"),linear(file,a+".beta.weight"),linear(file,a+".gate_a.weight"),
                linear(file,a+".gate_b.weight"),linear(file,a+".output.weight"),
                tb::cat({dense(file,a+".query_conv.weight"),dense(file,a+".key_conv.weight"),dense(file,a+".value_conv.weight")},0),
                dense(file,a+".forget_a.weight"),dense(file,a+".forget_b.weight"),dense(file,a+".dt_bias"),
                dense(file,a+".a"),dense(file,a+".output_norm.weight")};
            kda=std::make_unique<mfq::flash_next::Kda>(std::move(w),c.kda_heads,c.kda_width,c.kda_kernel,c.lower_bound,c.eps);
        } else {
            const auto a=p+".attention";
            mfq::flash_next::MlaWeights w{linear(file,a+".query_a.weight"),linear(file,a+".key_value_a.weight"),
                linear(file,a+".query_b.weight"),linear(file,a+".output.weight"),linear(file,a+".indexer.query.weight"),
                linear(file,a+".indexer.key.weight"),linear(file,a+".indexer.score.weight"),
                headwise(routed(file,a+".latent.query_embedding.weight",i,c.heads,c.latent,c.nope),c.heads,c.latent),
                headwise(routed(file,a+".latent.output_unembedding.weight",i,c.heads,c.value_width,c.latent),c.heads,c.value_width),
                dense(file,a+".query_a_norm.weight"),dense(file,a+".key_value_a_norm.weight"),
                dense(file,a+".indexer.key_norm.weight"),dense(file,a+".indexer.key_norm.bias"),
                dense(file,a+".indexer.pool.gate"),dense(file,a+".indexer.pool.position")};
            mfq::flash_next::MlaConfig mc{c.heads,c.nope,c.latent,c.value_width,c.index_heads,c.index_width,c.pool,c.budget,c.maximum,c.tail,c.eps};
            mla=std::make_unique<mfq::flash_next::SparseMla>(std::move(w),mc);
        }
    }
    void reset(int64_t) override { if (kda) kda->reset(); if (mla) mla->reset(); }
    bool supports_speculation() const noexcept override {return true;}
    void commit_speculative() override { if (kda) kda->commit(); }
    void rollback_speculative(int64_t keep) override { if (kda) kda->rollback(); if (mla) mla->truncate(keep); }
    Tensor execute(const Tensor& x,int64_t position,int64_t confirmed=0) {
        if (mla) MFQ_RUNTIME_CHECK(mla->position()==position,"GLM MLA/model cache positions diverged");
        auto first=attention_hc.pre(x,config);
        auto branch=mfq::flash_next::rms_norm(first[2],attention_norm,config.eps);
        branch=kda ? kda->forward(branch,true,confirmed) : mla->forward(branch,true);
        auto hidden=mfq_flash_next::glm5_mhc_post(branch,x,first[0],first[1]);
        auto second=ffn_hc.pre(hidden,config);
        branch=ffn(mfq::flash_next::rms_norm(second[2],ffn_norm,config.eps));
        return mfq_flash_next::glm5_mhc_post(branch,hidden,second[0],second[1]);
    }
    Tensor forward(Tensor x,Tensor,int64_t position,const MfqOptional<Tensor>&,
        const RopeCache&,const MfqOptional<Tensor>& = mfq_nullopt,
        const MfqOptional<Tensor>& mask = mfq_nullopt) override {
        MFQ_RUNTIME_CHECK(!mask.has_value(),"GLM Flash-Next requires its causal unpadded attention geometry");
        return execute(x,position);
    }
    Tensor forward_context(
        Tensor x,const Context& context,const RopeCache&) override {
        return execute(x,context.cache_position,context.confirmed_prefix);
    }
};

