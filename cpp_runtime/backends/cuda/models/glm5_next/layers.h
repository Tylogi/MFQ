#pragma once

#include "loader.h"
#include "model.h"

namespace mfq::cuda::glm5_next {

inline Linear glm_ffn(CudaExecutionContext& execution, const mfq::ModelSource& file,const mfq::models::glm5_next::Config& c,int i,const std::string& root="model") {
    const auto p=root+".block."+std::to_string(i)+".mlp";
    if (root=="model" && c.mlp_types.at(i)=="dense") return dense_ffn(execution,file,p,c.swiglu_limit);
    auto gate_up=routed_gate_up(execution,file,p,i,c.experts,c.moe_intermediate,c.hidden);
    auto down=routed(execution,file,p+".experts.down.weight",i,c.experts,c.hidden,c.moe_intermediate);
    auto router=linear(execution,file,p+".router.weight"),shared=dense_ffn(execution,file,p+".shared_expert",c.swiglu_limit);
    auto bias=dense(execution,file,p+".router.bias").to(tb::kFloat32).contiguous();
    return [gate_up,down,router,shared,bias,c](
            CudaExecutionContext& execution, const Tensor& x) {
        auto source=x.reshape({-1,c.hidden}).to(tb::kFloat16);
        auto selected=moe_topk_cuda(
            router(execution,source.to(tb::kFloat32))
                .to(tb::kFloat32).contiguous(),
            c.topk,true,false,c.normalize_routes,false,bias,1e-20,c.router_scale);
        auto gu=gate_up(execution,source,selected[0]);
        auto gate=tb::clamp_max(gu.narrow(-1,0,c.moe_intermediate),c.swiglu_limit);
        auto up=tb::clamp(gu.narrow(-1,c.moe_intermediate,c.moe_intermediate),-c.swiglu_limit,c.swiglu_limit);
        auto pairs=down(execution,(gate*tb::sigmoid(gate))*up,selected[0]);
        auto reduced=tb::zeros({source.size(0),c.hidden},source.options().dtype(tb::kFloat32));
        for (int64_t r=0;r<c.topk;++r)
            reduced=reduced+pairs.select(1,r).to(tb::kFloat32)*selected[1].select(1,r).unsqueeze(-1);
        return (reduced.to(pairs.scalar_type())+
            shared(execution,source)).reshape(x.sizes());
    };
}

struct Mhc {
    Tensor function,base,scale;
    Mhc(CudaExecutionContext& execution, const mfq::ModelSource& file,const std::string& p)
        : function(dense(execution,file,p+".function")),base(dense(execution,file,p+".base")),scale(dense(execution,file,p+".scale")) {}
    std::vector<Tensor> pre(const Tensor& x,const mfq::models::glm5_next::Config& c) const {
        return mfq_glm5_next::mhc_pre(x,function,base,scale,c.sinkhorn,c.hc_eps,c.eps);
    }
};

struct Glm5NextBlock final : Block {
    using Tensor=mfq_tensor_backend::Tensor;
    mfq::models::glm5_next::Config config;
    Mhc attention_hc,ffn_hc;
    Tensor attention_norm,ffn_norm;
    Linear ffn;
    std::unique_ptr<Kda> kda;
    std::unique_ptr<SparseMla> mla;

    Glm5NextBlock(CudaExecutionContext& execution, const mfq::ModelSource& file,const mfq::models::glm5_next::Config& c,int i)
        : config(c),attention_hc(execution,file,"model.block."+std::to_string(i)+".attention.mhc.pre"),
          ffn_hc(execution,file,"model.block."+std::to_string(i)+".mlp.mhc.pre") {
        const auto p="model.block."+std::to_string(i);
        attention_norm=dense(execution,file,p+".attention.norm.weight"); ffn_norm=dense(execution,file,p+".mlp.norm.weight");
        ffn=glm_ffn(execution,file,c,i);
        if (c.layer_types.at(i)=="linear_attention") {
            const auto a=p+".linear_attention";
            KdaWeights w{linear(execution,file,a+".query.weight"),linear(execution,file,a+".key.weight"),
                linear(execution,file,a+".value.weight"),linear(execution,file,a+".beta.weight"),linear(execution,file,a+".gate_a.weight"),
                linear(execution,file,a+".gate_b.weight"),linear(execution,file,a+".output.weight"),
                tb::cat({dense(execution,file,a+".query_conv.weight"),dense(execution,file,a+".key_conv.weight"),dense(execution,file,a+".value_conv.weight")},0),
                dense(execution,file,a+".forget_a.weight"),dense(execution,file,a+".forget_b.weight"),dense(execution,file,a+".dt_bias"),
                dense(execution,file,a+".a"),dense(execution,file,a+".output_norm.weight")};
            kda=std::make_unique<Kda>(std::move(w),c.kda_heads,c.kda_width,c.kda_kernel,c.lower_bound,c.eps);
        } else {
            const auto a=p+".attention";
            MlaWeights w{linear(execution,file,a+".query_a.weight"),linear(execution,file,a+".key_value_a.weight"),
                linear(execution,file,a+".query_b.weight"),linear(execution,file,a+".output.weight"),linear(execution,file,a+".indexer.query.weight"),
                linear(execution,file,a+".indexer.key.weight"),linear(execution,file,a+".indexer.score.weight"),
                headwise(routed(execution,file,a+".latent.query_embedding.weight",i,c.heads,c.latent,c.nope),c.heads,c.latent),
                headwise(routed(execution,file,a+".latent.output_unembedding.weight",i,c.heads,c.value_width,c.latent),c.heads,c.value_width),
                dense(execution,file,a+".query_a_norm.weight"),dense(execution,file,a+".key_value_a_norm.weight"),
                dense(execution,file,a+".indexer.key_norm.weight"),dense(execution,file,a+".indexer.key_norm.bias"),
                dense(execution,file,a+".indexer.pool.gate"),dense(execution,file,a+".indexer.pool.position")};
            MlaConfig mc{c.heads,c.nope,c.latent,c.value_width,c.index_heads,c.index_width,c.pool,c.budget,c.maximum,c.tail,c.eps};
            mla=std::make_unique<SparseMla>(std::move(w),mc);
        }
    }
    void reset(int64_t) override { if (kda) kda->reset(); if (mla) mla->reset(); }
    bool supports_speculation() const noexcept override {return true;}
    void commit_speculative() override { if (kda) kda->commit(); }
    void rollback_speculative(int64_t keep) override { if (kda) kda->rollback(); if (mla) mla->truncate(keep); }
    Tensor execute(
            CudaExecutionContext& execution,
            const Tensor& x,
            int64_t position,
            int64_t confirmed=0) {
        if (mla) MFQ_RUNTIME_CHECK(mla->position()==position,"GLM MLA/model cache positions diverged");
        auto first=attention_hc.pre(x,config);
        auto branch=rms_norm(first[2],attention_norm,config.eps);
        branch=kda
            ? kda->forward(execution,branch,true,confirmed)
            : mla->forward(execution,branch,true);
        auto hidden=mfq_glm5_next::mhc_post(branch,x,first[0],first[1]);
        auto second=ffn_hc.pre(hidden,config);
        branch=ffn(
            execution,rms_norm(second[2],ffn_norm,config.eps));
        return mfq_glm5_next::mhc_post(branch,hidden,second[0],second[1]);
    }
    Tensor forward(
        CudaExecutionContext& execution,
        Tensor x,Tensor,int64_t position,const MfqOptional<Tensor>&,
        const RopeCache&,const MfqOptional<Tensor>& = mfq_nullopt,
        const MfqOptional<Tensor>& mask = mfq_nullopt) override {
        MFQ_RUNTIME_CHECK(!mask.has_value(),"GLM requires its causal unpadded attention geometry");
        return execute(execution,x,position);
    }
    Tensor forward_context(
        CudaExecutionContext& execution,
        Tensor x,const Context& context,const RopeCache&) override {
        return execute(
            execution,x,context.cache_position,context.confirmed_prefix);
    }
};

} // namespace mfq::cuda::glm5_next
