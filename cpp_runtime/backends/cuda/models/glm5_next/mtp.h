#pragma once

#include "../../runtime/mtp.h"
#include "layers.h"

#include <algorithm>
#include <optional>
#include <utility>
#include <vector>

// GLM5-Next predictor shares the target embedding and output projection.
struct Glm5NextMtp final : MtpModule {
    using Tensor=mfq_tensor_backend::Tensor;
    using Linear=mfq::flash_next::Linear;
    struct Layer {
        Tensor attention_norm,ffn_norm;
        std::unique_ptr<mfq::flash_next::SparseMla> attention;
        Linear ffn;
    };

    mfq::models::flash_next::GlmConfig config;
    Tensor embedding_norm,hidden_norm,output_norm;
    Linear fusion;
    std::vector<Layer> layers;
    std::vector<int64_t> lengths;
    int64_t batch=0;

    static std::optional<Glm5NextMtp> load_if_present(
            const mfq::ModelSource& file,
            const mfq::models::flash_next::GlmConfig& main) {
        using namespace flash_runtime;
        const bool any=std::any_of(file.tensors().begin(),file.tensors().end(),
            [](const mfq::TensorMetadata& tensor) {return tensor.name.rfind("predictor.",0)==0;});
        const auto count=main.predictor_layers;
        if (!has_tensor(file,"predictor.embedding_norm.weight") || count<=0) {
            MFQ_RUNTIME_CHECK(!any,"GLM5-Next model source contains an incomplete or undeclared MTP head");
            return std::nullopt;
        }
        Glm5NextMtp result;
        result.config=main;
        result.embedding_norm=dense(file,"predictor.embedding_norm.weight").to(tb::kFloat32);
        result.hidden_norm=dense(file,"predictor.hidden_norm.weight").to(tb::kFloat32);
        result.output_norm=dense(file,"predictor.output_norm.weight");
        result.fusion=linear(file,"predictor.fusion.weight");
        for (int64_t i=0;i<count;++i) {
            const auto p="predictor.block."+std::to_string(i),a=p+".attention";
            Layer layer;layer.attention_norm=dense(file,a+".norm.weight");layer.ffn_norm=dense(file,p+".mlp.norm.weight");
            layer.ffn=glm_ffn(file,main,int(i),"predictor");
            mfq::flash_next::MlaWeights w{linear(file,a+".query_a.weight"),linear(file,a+".key_value_a.weight"),
                linear(file,a+".query_b.weight"),linear(file,a+".output.weight"),linear(file,a+".indexer.query.weight"),
                linear(file,a+".indexer.key.weight"),linear(file,a+".indexer.score.weight"),
                headwise(routed(file,a+".latent.query_embedding.weight",int(i),main.heads,main.latent,main.nope),main.heads,main.latent),
                headwise(routed(file,a+".latent.output_unembedding.weight",int(i),main.heads,main.value_width,main.latent),main.heads,main.value_width),
                dense(file,a+".query_a_norm.weight"),dense(file,a+".key_value_a_norm.weight"),
                dense(file,a+".indexer.key_norm.weight"),dense(file,a+".indexer.key_norm.bias"),
                dense(file,a+".indexer.pool.gate"),dense(file,a+".indexer.pool.position")};
            mfq::flash_next::MlaConfig mc{main.heads,main.nope,main.latent,main.value_width,main.index_heads,
                main.index_width,main.pool,main.budget,main.maximum,main.tail,main.eps};
            layer.attention=std::make_unique<mfq::flash_next::SparseMla>(std::move(w),mc);
            result.layers.push_back(std::move(layer));
        }
        MFQ_RUNTIME_CHECK(result.embedding_norm.dim()==1 && result.embedding_norm.numel()==main.hidden &&
            result.hidden_norm.dim()==1 && result.hidden_norm.numel()==main.hidden &&
            result.output_norm.numel()==main.hidden,"GLM5-Next MTP normalization width disagrees with backbone");
        result.lengths.resize(count,0);
        return result;
    }

    void reset(int64_t next_batch=1) override {
        MFQ_RUNTIME_CHECK(next_batch>0,"GLM5-Next MTP batch must be positive");
        for (auto& layer:layers) layer.attention->reset();
        std::fill(lengths.begin(),lengths.end(),0);batch=next_batch;
    }

    std::pair<Tensor,Tensor> evaluate(const MtpTarget& target,const Tensor& hidden,const Tensor& ids,int64_t depth=0,
        bool cache=true,const Tensor& supplied_positions={},const Tensor& supplied_embeddings={}) {
        namespace tb=mfq_tensor_backend;
        MFQ_RUNTIME_CHECK(ids.dim()==2 && ids.size(0)>0 && ids.size(1)>0 && hidden.dim()==3 &&
            hidden.size(0)==ids.size(0) && hidden.size(1)==ids.size(1) && hidden.size(2)==hidden_norm.numel(),
            "GLM5-Next MTP input geometry mismatch");
        const auto b=ids.size(0),t=ids.size(1),n=int64_t(lengths.size());
        const auto layer=(depth%n+n)%n;
        if (cache && batch && batch!=b) reset(b);
        const auto start=cache?lengths[layer]:0;
        MFQ_RUNTIME_CHECK(start+t<=config.maximum,"GLM5-Next MTP history exceeds context capacity");
        auto embeds=supplied_embeddings.defined()?supplied_embeddings:target.embed(ids);
        MFQ_RUNTIME_CHECK(embeds.sizes().vec()==std::vector<int64_t>({b,t,config.hidden}),
            "GLM5-Next MTP embedding shape mismatch");
        auto current=supplied_positions.defined()?supplied_positions:tb::arange(start,start+t,ids.options().dtype(tb::kInt32));
        MFQ_RUNTIME_CHECK((current.dim()==1 || current.dim()==2) && current.size(-1)==t &&
            (current.dim()==1 || current.size(0)==b),"GLM MTP positions require [T] or [B,T]");
        current=current.to(tb::kInt32);
        auto mask=(current==0).reshape({current.dim()==1?1:b,t,1});
        auto e=tb::where(mask,tb::zeros_like(embeds),embeds);
        auto x=fusion(tb::cat({mfq::flash_next::rms_norm(e,embedding_norm,config.eps),
            mfq::flash_next::rms_norm(hidden,hidden_norm,config.eps)},-1));
        auto& block=layers[layer];
        auto attention=block.attention->forward(mfq::flash_next::rms_norm(x,block.attention_norm,config.eps),cache);
        auto residual=x.to(tb::kFloat32)+attention.to(tb::kFloat32);
        const auto dtype=x.scalar_type()==tb::kFloat32?tb::kFloat32:tb::kFloat16;
        auto branch=mfq::flash_next::rms_norm(residual,block.ffn_norm,config.eps).to(dtype);
        auto multi=residual+block.ffn(branch).to(tb::kFloat32);
        auto output=mfq::flash_next::rms_norm(multi,output_norm,config.eps).to(dtype);
        if (cache) {batch=b;lengths[layer]=start+t;}
        return {output,multi};
    }

    Tensor forward(const MtpTarget& target,Tensor hidden,Tensor ids) override {return evaluate(target,hidden,ids).first;}
    MtpStep step(const MtpTarget& target,Tensor hidden,Tensor ids) override {
        auto result=evaluate(target,hidden,ids);return {std::move(result.first),std::move(result.second)};
    }
    int64_t cache_position() const noexcept override {return lengths.empty()?0:lengths.front();}
    void trim_cache_to(int64_t position) override {
        MFQ_RUNTIME_CHECK(position>=0 && position<=cache_position(),"GLM5-Next MTP cache trim position is invalid");
        constexpr size_t index=0;
        if (index<layers.size() && layers[index].attention) layers[index].attention->truncate(position);
        lengths[index]=position;
    }
    bool teacher_forced_prompt_prime() const noexcept override {return false;}
    bool target_bootstrap_decode() const noexcept override {return true;}
    bool preserve_output_dtype() const noexcept override {return true;}
};
