#pragma once

#include "core/causal_model.h"
#include "core/block.h"
#include "models/qwen4_exp/causal_lm.h"
#include "models/qwen4_exp/config.h"
#include "quant_linear.h"

#include <memory>
#include <functional>

namespace mfq::cuda::qwen4_exp { struct Gr; }

namespace mfq::cuda {
struct Qwen4DecodeGraph;
void qwen4_begin_router_lookahead_audit(Qwen4CausalLm&,int samples);
void qwen4_finish_router_lookahead_audit(Qwen4CausalLm&,const std::string& prefix);
void qwen4_set_rotary_fusion(Qwen4CausalLm&, bool enabled);
void qwen4_set_attention_grouping(Qwen4CausalLm&, bool enabled);
std::optional<mfq_tensor_backend::Tensor> qwen4_decode_graph_logits(
    Qwen4CausalLm&,mfq_tensor_backend::Tensor,int);
bool qwen4_layer_major_prefill_eligible(const Qwen4CausalLm&, int64_t tokens);
mfq_tensor_backend::Tensor qwen4_layer_major_prefill_hidden(
    Qwen4CausalLm&, mfq_tensor_backend::Tensor ids, int64_t compute_chunk = 1024,
    bool last_only = true, const std::function<bool()>& cancelled = {});

struct Qwen4Model : CausalResources {
    Qwen4Model();
    ~Qwen4Model();
    Qwen4Model(Qwen4Model&&) noexcept;
    Qwen4Model& operator=(Qwen4Model&&) noexcept;
    template <class Backend> using CausalModel = mfq::models::qwen4_exp::CausalLm<Backend>;
    mfq::models::qwen4_exp::Config config;
    std::unique_ptr<qwen4_exp::Gr> final_mixer;
    mfq_tensor_backend::Tensor positions;
    int64_t batch = 0;
    std::optional<bool> router_lookahead;
    std::unique_ptr<Qwen4DecodeGraph> graph_registry;
    template<class Model> std::optional<mfq_tensor_backend::Tensor>
    adapter_graph_logits(Model& model,mfq_tensor_backend::Tensor ids,int kind) {
        if (ids.dim()==2 && ids.size(0)==1 &&
            qwen4_layer_major_prefill_eligible(model,ids.size(1))) {
            auto hidden=qwen4_layer_major_prefill_hidden(model,std::move(ids),1024,kind!=0);
            if(kind==2)return model.adapter_next_token(model.lm_head,model.last_hidden(std::move(hidden)));
            if(kind==1)return model.apply_final_logit_softcap(
                model.adapter_last_logits(model.lm_head,model.last_hidden(std::move(hidden))));
            return model.logits_from_hidden(std::move(hidden));
        }
        return qwen4_decode_graph_logits(model,std::move(ids),kind);
    }

    void adapter_validate_load_options() const;
    void adapter_load_final_state(const mfq::ModelSource &source,
                                  mfq_tensor_backend::Tensor &output_norm);
    std::unique_ptr<Block> adapter_load_block(const mfq::ModelSource &source, int layer, int device,
                                              const std::string &type);

    void adapter_reset(int64_t batch);
    void adapter_begin_forward(bool capture_raw_hidden);

    mfq_tensor_backend::Tensor
    adapter_block_positions(const mfq_tensor_backend::Tensor &full_positions,
                            const mfq_tensor_backend::Tensor &local_positions, int device) const;
    void adapter_finish_forward(const mfq_tensor_backend::Tensor &full_positions, int64_t batch,
                                int64_t tokens);
    mfq_tensor_backend::Tensor
    adapter_finalize_hidden(mfq_tensor_backend::Tensor hidden,
                            const mfq_tensor_backend::Tensor &output_norm, int64_t batch,
                            int64_t tokens) const;
    mfq_tensor_backend::Tensor adapter_last_logits(const QuantLinear &lm_head,
                                                   mfq_tensor_backend::Tensor hidden) const;
    mfq_tensor_backend::Tensor adapter_next_token(const QuantLinear &lm_head,
                                                  mfq_tensor_backend::Tensor hidden) const;
    void adapter_rollback_speculative(int64_t keep);
};

extern template struct CudaSessionCodec<Qwen4Model>;

} // namespace mfq::cuda

namespace mfq::models {
extern template struct qwen4_exp::CausalLm<cuda::CudaCausalOps<cuda::Qwen4Model>>;
} // namespace mfq::models
