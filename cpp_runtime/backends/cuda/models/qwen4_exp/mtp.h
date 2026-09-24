#pragma once

#include "../../runtime/mtp.h"
#include "layers.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

// Qwen4-Exp predictor shares the target embedding and output projection.
struct Qwen4ExpMtp final : MtpModule {
    using Tensor=mfq_tensor_backend::Tensor;
    using Linear=mfq::flash_next::Linear;

    mfq::models::flash_next::QwenConfig config;
    Tensor embedding_norm,hidden_norm;
    Linear embedding_fusion,hidden_fusion;
    std::unique_ptr<flash_runtime::Gr> final_mixer;
    std::vector<std::unique_ptr<Qwen4Block>> layers;
    std::vector<Tensor> positions;
    std::vector<int64_t> lengths;
    int64_t batch=0;

    static std::optional<Qwen4ExpMtp> load_if_present(
            const mfq::ModelSource& file,
            const mfq::models::flash_next::QwenConfig& main) {
        using namespace flash_runtime;
        const bool any=std::any_of(file.tensors().begin(),file.tensors().end(),
            [](const mfq::TensorMetadata& tensor) {return tensor.name.rfind("predictor.",0)==0;});
        const auto count=main.predictor_layers;
        if (!has_tensor(file,"predictor.embedding_norm.weight") || count<=0) {
            MFQ_RUNTIME_CHECK(!any,"Qwen4-Exp model source contains an incomplete or undeclared MTP head");
            return std::nullopt;
        }
        Qwen4ExpMtp result;
        result.config=main;
        result.embedding_norm=dense(file,"predictor.embedding_norm.weight").to(tb::kFloat32);
        result.hidden_norm=dense(file,"predictor.hidden_norm.weight").to(tb::kFloat32);
        result.embedding_fusion=linear(file,"predictor.fusion.embedding.weight");
        result.hidden_fusion=linear(file,"predictor.fusion.hidden.weight");
        result.final_mixer=std::make_unique<Gr>(file,main,"predictor.mhc.pre",false);
        for (int64_t i=0;i<count;++i)
            result.layers.push_back(std::make_unique<Qwen4Block>(file,main,int(i),"predictor"));
        MFQ_RUNTIME_CHECK(result.embedding_norm.dim()==1 && result.embedding_norm.numel()==main.hidden &&
            result.hidden_norm.dim()==1 && result.hidden_norm.numel()==main.hidden*main.streams,
            "Qwen4-Exp MTP normalization width disagrees with backbone");
        result.positions.resize(count);result.lengths.resize(count,0);
        return result;
    }

    void reset(int64_t next_batch=1) override {
        MFQ_RUNTIME_CHECK(next_batch>0,"Qwen4-Exp MTP batch must be positive");
        for (auto& layer:layers) layer->reset(next_batch);
        for (auto& pos:positions) pos={};
        std::fill(lengths.begin(),lengths.end(),0);batch=next_batch;
    }

    std::pair<Tensor,Tensor> evaluate(const MtpTarget& target,const Tensor& hidden,const Tensor& ids,int64_t depth=0,
        bool cache=true,const Tensor& supplied_positions={},const Tensor& supplied_embeddings={}) {
        namespace tb=mfq_tensor_backend;
        MFQ_RUNTIME_CHECK(ids.dim()==2 && ids.size(0)>0 && ids.size(1)>0 && hidden.dim()==3 &&
            hidden.size(0)==ids.size(0) && hidden.size(1)==ids.size(1) && hidden.size(2)==hidden_norm.numel(),
            "Qwen4-Exp MTP input geometry mismatch");
        const auto b=ids.size(0),t=ids.size(1),n=int64_t(lengths.size());
        const auto layer=(depth%n+n)%n;
        if (cache && batch && batch!=b) reset(b);
        const auto start=cache?lengths[layer]:0;
        MFQ_RUNTIME_CHECK(start+t<=config.maximum,"Qwen4-Exp MTP history exceeds context capacity");
        auto embeds=supplied_embeddings.defined()?supplied_embeddings:target.embed(ids);
        MFQ_RUNTIME_CHECK(embeds.sizes().vec()==std::vector<int64_t>({b,t,config.hidden}),
            "Qwen4-Exp MTP embedding shape mismatch");
        auto current=supplied_positions.defined()?supplied_positions:tb::arange(start,start+t,ids.options().dtype(tb::kInt32));
        if (current.dim()==1) current=current.reshape({1,1,t}).expand({3,1,t}).contiguous();
        else if (current.dim()==2) current=current.unsqueeze(1);
        if (current.dim()==3 && current.size(0)==4) current=current.narrow(0,1,3);
        MFQ_RUNTIME_CHECK(current.dim()==3 && current.size(0)==3 && current.size(-1)==t &&
            (current.size(1)==1 || current.size(1)==b),"Qwen4 MTP positions require [T]/[3,T]/[3,B,T]");
        current=current.to(tb::kInt32).expand({3,b,t}).contiguous();
        auto full=cache && positions[layer].defined()?tb::cat({positions[layer],current},-1):current;
        auto e=embedding_fusion(mfq::flash_next::rms_norm(embeds,embedding_norm+1,config.eps));
        auto streams=hidden_fusion(mfq::flash_next::rms_norm(hidden,hidden_norm+1,config.eps)
            .reshape({b,t,config.streams,config.hidden}));
        auto x=(streams+e.unsqueeze(-2)).reshape({b,t,config.streams*config.hidden});
        auto& block=*layers[layer];
        auto first=block.attention_gr.pre(x);
        auto branch=block.qsa->forward(first[0],current,full,cache);
        x=block.attention_gr.post(branch,first);
        auto second=block.ffn_gr.pre(x);
        auto multi=block.ffn_gr.post(block.ffn(second[0]),second);
        auto output=final_mixer->pre(multi)[0];
        if (cache) {positions[layer]=full;batch=b;lengths[layer]=start+t;}
        return {output,multi};
    }

    Tensor forward(const MtpTarget& target,Tensor hidden,Tensor ids) override {return evaluate(target,hidden,ids).first;}
    MtpStep step(const MtpTarget& target,Tensor hidden,Tensor ids) override {
        auto result=evaluate(target,hidden,ids);return {std::move(result.first),std::move(result.second)};
    }
    int64_t cache_position() const noexcept override {return lengths.empty()?0:lengths.front();}
    void trim_cache_to(int64_t position) override {
        MFQ_RUNTIME_CHECK(position>=0 && position<=cache_position(),"Qwen4-Exp MTP cache trim position is invalid");
        constexpr size_t index=0;
        if (index<layers.size() && layers[index]->qsa) layers[index]->qsa->truncate(position);
        if (positions[index].defined()) positions[index]=positions[index].narrow(-1,0,position);
        lengths[index]=position;
    }
    bool teacher_forced_prompt_prime() const noexcept override {return true;}
    bool target_bootstrap_decode() const noexcept override {return true;}
    bool preserve_output_dtype() const noexcept override {return true;}
};
