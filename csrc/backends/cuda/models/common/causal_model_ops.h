#pragma once

#include "csrc/core/step_sequence.h"

#include "models/cuda_model_plan.h"
#include "block.h"
#include "resource_plan.h"
#include "model_source.h"
#include "common/causal_model.h"
#include "ops/quant/quant_linear.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

enum class TextSessionStateKind : int;
struct TextSessionState;
struct MtpModule;

namespace mfq::models::qwen35 {
template <class Backend> struct CausalLm;
}
namespace mfq::models::qwen4_exp {
template <class Backend> struct CausalLm;
}
namespace mfq::models::glm5_next {
template <class Backend> struct CausalLm;
}
namespace mfq::models::glm_dsa {
template <class Backend> struct CausalLm;
}
namespace mfq::models::gemma4 {
template <class Backend> struct CausalLm;
}
namespace mfq::models::deepseek_v4 {
template <class Backend> struct CausalLm;
}
namespace mfq::models::deepseek_v41 {
template <class Backend> struct CausalLm;
}
namespace mfq::models::minicpmo45 {
template <class Backend> struct CausalLm;
}
namespace mfq::models::minicpmo45 {
template <class Backend> struct TtsCausalLm;
}

namespace mfq::cuda {

template <typename Model>
Model load_causal_lm(CudaExecutionContext& execution, const std::string& model_path,
    const std::string& config_path, std::int64_t context_size_override = 0,
    bool load_blocks = true, bool defer_moe_cache_finalize = false,
    std::shared_ptr<const mfq::ModelSource> source = {}, bool load_input_output = true);

struct CausalResources {
    // The default family strategy: text only, plain execution paths. A family
    // that differs states its own `policy` and `plan_components` in its own
    // models/*/ops.{h,cpp}; nothing else in the backend states what a family is.
    static constexpr CudaFamilyPolicy policy{};
    // Which declared graph components this family has CUDA adapters for.
    static CudaModelPlan plan_components(const mfq::ModelGraph&) { return {}; }

    CudaResourcePlan resource_plan(const std::vector<std::unique_ptr<Block>>&,
        bool batch, std::size_t devices, const MtpModule*) const;
    mfq::models::CausalLmMetadata metadata;
    CudaExecutionContext *execution = nullptr;
    std::function<void()> restore_decode_workspace;
    template<class Model> std::optional<mfq_tensor_backend::Tensor>
    captured_logits(Model&,mfq_tensor_backend::Tensor,int) {return {};}

    void validate_load_options() const;
    void configure_rope(RopeCache &rope, mfq_tensor_backend::Device device) const;
    void load_final_state(const mfq::ModelSource &source,
                                  mfq_tensor_backend::Tensor &output_norm);
    void prepare_blocks(const mfq::ModelSource &source);

    mfq_tensor_backend::Tensor post_embed(mfq_tensor_backend::Tensor output) const;
    void reset_native(int64_t batch);
    bool requires_batch_reset(int64_t batch) const noexcept;

    mfq_tensor_backend::Tensor prepare_hidden(mfq_tensor_backend::Tensor hidden,
                                                      int64_t batch, int64_t tokens) const;
    void begin_forward(bool capture_raw_hidden);

    mfq_tensor_backend::Tensor
    block_positions(const mfq_tensor_backend::Tensor &full_positions,
                            const mfq_tensor_backend::Tensor &local_positions, int device) const;

    void finish_forward(const mfq_tensor_backend::Tensor &full_positions, int64_t batch,
                                int64_t tokens);

    mfq_tensor_backend::Tensor
    finalize_hidden(mfq_tensor_backend::Tensor hidden,
                            const mfq_tensor_backend::Tensor &output_norm, int64_t batch,
                            int64_t tokens) const;
    mfq_tensor_backend::Tensor
    extract_raw_hidden(const mfq_tensor_backend::Tensor &hidden,
                       const mfq_tensor_backend::Tensor &finalized) const;
    mfq_tensor_backend::Tensor logits_from_head(const QuantLinear &lm_head,
                                              mfq_tensor_backend::Tensor hidden) const;
    mfq_tensor_backend::Tensor last_logits_from_head(const QuantLinear &lm_head,
                                                   mfq_tensor_backend::Tensor hidden) const;
    mfq_tensor_backend::Tensor next_token_from_head(const QuantLinear &lm_head,
                                                  mfq_tensor_backend::Tensor hidden) const;

    void native_begin_speculation();
    void native_commit_speculation();
    void native_rollback_speculation(int64_t keep);
};

struct Qwen35Model;
struct MiniCPMO45Model;
struct MiniCPMOTtsModel;
struct Gemma4Model;
struct GlmDsaModel;
struct Glm5Model;
struct Qwen4Model;
struct DeepseekV4Model;
struct DeepseekV41Model;

template <typename Model> struct CudaCausalOps;
template <typename Model>
using CausalLm = typename Model::template CausalModel<CudaCausalOps<Model>>;

template <typename Model> struct CudaSessionCodec {
    using CausalModel = CausalLm<Model>;
    static TextSessionStateKind kind(const CausalModel &model);
    static bool supports_paged(const CausalModel &model);
    static mfq::StepSequence<TextSessionState> capture_steps(const CausalModel &model, const std::vector<int64_t> &tokens);
    static TextSessionState capture(const CausalModel &model, const std::vector<int64_t> &tokens);
    static mfq::StepSequence<std::monostate> restore_steps(CausalModel &model, const TextSessionState &state);
    static void restore(CausalModel &model, const TextSessionState &state);
};

struct CudaForwardPlan {
    int64_t kv_length = 0;
    int64_t attention_parts = 0;
};

// One loaded CUDA causal model, independent of which family instantiated it.
// Only MTP and the MiniCPM-o composite need to name a model without knowing its
// family, and both only need to run embed/logits. models::CausalModelBase
// already implements them, so no family writes code for this.
struct CudaCausalModel {
    using Tensor = mfq_tensor_backend::Tensor;
    virtual ~CudaCausalModel() = default;
    virtual Tensor embed_forward(Tensor ids) const = 0;
    virtual Tensor logits_from_hidden(Tensor hidden) = 0;
};

template <typename Model> struct CudaCausalOps : Model, CudaCausalModel {
    using Tensor = mfq_tensor_backend::Tensor;
    using ForwardPlan = CudaForwardPlan;
    using CausalModel = CausalLm<Model>;
    using SessionState = TextSessionState;
    using SessionStateKind = TextSessionStateKind;
    using SessionCodec = CudaSessionCodec<Model>;
    std::shared_ptr<const mfq::ModelSource> source;
    mfq::ModelGraph graph;
    CudaModelPlan plan;
    RopeCache rope;
    RopeCache cpu_rope;
    std::unordered_map<int, RopeCache> device_ropes;
    QuantLinear embed;
    std::vector<std::unique_ptr<Block>> blocks;
    mfq_tensor_backend::Tensor output_norm;
    QuantLinear lm_head;
    std::optional<Tensor> graph_logits(CausalModel& model,Tensor ids,int kind) {
        return static_cast<Model&>(model).captured_logits(model,std::move(ids),kind);
    }

    auto execution_scope() const {
        return CudaExecutionScope(*this->execution);
    }
    static auto block_scope(const std::unique_ptr<Block> &block) {
        return MfqCudaGuard(block->cuda_device);
    }
    static bool defined(const Tensor &value) { return value.defined(); }
    static Tensor position_axes(Tensor value, int64_t start, int64_t count) {
        return value.narrow(0, start, count);
    }
    static Tensor concat_positions(Tensor first, Tensor second) {
        // Generated text advances all semantic axes equally after an image.
        if (first.dim() > 1 && second.dim() == 1) {
            auto shape = first.sizes().vec();
            shape.back() = second.size(0);
            std::vector<int64_t> view(shape.size(),1);
            view.back() = second.size(0);
            second = second.reshape(view).expand(shape);
        }
        return mfq_tensor_backend::cat({first, second}, -1);
    }
    static int64_t rank(const Tensor &value) { return value.dim(); }
    static int64_t size(const Tensor &value, int axis) { return value.size(axis); }
    static Tensor batch_ids(Tensor ids) { return ids.unsqueeze(0); }
    Tensor device_ids(Tensor ids) const {
        return tensor_to_cuda_device(this->execution->model_parallel_collectives,
                                     ids.to(mfq_tensor_backend::kInt64),
                                     this->execution->layer_placement.primary_device())
            .contiguous();
    }
    Tensor embed_tokens(Tensor ids) const {
        return this->execution->profiler.measure(
            "model.embed", [&] { return this->post_embed(quant_embedding_lookup(embed, ids)); });
    }
    Tensor sequence_lengths(int64_t batch, int64_t end) const {
        auto scope = execution_scope();
        return mfq_tensor_backend::full({batch}, end,
                                        mfq_tensor_backend::TensorOptions()
                                            .dtype(mfq_tensor_backend::kInt64)
                                            .device(mfq_tensor_backend::kCUDA));
    }
    static Tensor last_hidden(Tensor hidden) { return hidden.index({Slice(), -1, Slice()}); }
    static Tensor softcap(Tensor logits, double cap) {
        return mfq_tensor_backend::tanh(logits / cap) * cap;
    }
    Tensor scale_hidden(Tensor hidden, double scale) const {
        return this->execution->profiler.measure("model.embed_scale",
                                                 [&] { return hidden * scale; });
    }
    static Tensor repeat_hidden(Tensor hidden, int64_t streams) {
        return hidden.to(mfq_tensor_backend::kFloat16).repeat({1, 1, streams});
    }
    static Tensor expand_hidden(Tensor hidden, int64_t batch, int64_t tokens, int64_t streams) {
        const auto width = hidden.size(-1);
        return hidden.to(mfq_tensor_backend::kFloat16)
            .unsqueeze(2)
            .expand({batch, tokens, streams, width})
            .contiguous();
    }
    struct ForwardOps : mfq::models::CausalForwardInputs<Tensor, ForwardPlan> {
        using Inputs = mfq::models::CausalForwardInputs<Tensor, ForwardPlan>;
        CausalModel &model;
        ForwardOps(CausalModel &model, Inputs input)
            : Inputs(std::move(input)), model(model),
              primary(model.execution->layer_placement.primary_device()) {}
        int primary;
        Tensor pos, full_positions, cache_positions, cpu_ids, cpu_pos, cpu_cache_positions;
        MfqOptional<Tensor> effective_attention_mask, cpu_attention_mask, cpu_seq_len;

        static int64_t size(const Tensor &value, int axis) { return value.size(axis); }
        static int64_t rank(const Tensor &value) { return value.dim(); }
        static int64_t elements(const Tensor &value) { return value.numel(); }

        Tensor device_ids(Tensor value) {
            return tensor_to_cuda_device(model.execution->model_parallel_collectives,
                                         value.to(mfq_tensor_backend::kInt64), primary)
                .contiguous();
        }
        void prepare_inputs() { ids = device_ids(ids); }
        bool has_mrope() const { return model.rope.sections.numel() > 0; }
        Tensor position_range(int64_t start, int64_t tokens) {
            return mfq_tensor_backend::arange(start, start + tokens,
                                              mfq_tensor_backend::TensorOptions()
                                                  .device(mfq_tensor_backend::kCUDA)
                                                  .dtype(mfq_tensor_backend::kInt64));
        }
        static Tensor offset_positions(Tensor positions, int64_t delta) {
            return positions + delta;
        }
        Tensor prepare_hidden(int64_t B, int64_t T) {
            if(model.execution->config.moe_pipeline) {
                cpu_ids=ids.to(mfq_tensor_backend::kCPU,mfq_tensor_backend::kInt64).contiguous();
                for(const auto& block:model.blocks)block->prefetch_token_ids(cpu_ids);
            }
            if (model.execution->dense_cpu_layer_count > 0) {
                cpu_ids = ids.to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt64).contiguous();
                cpu_pos = pos.to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt64).contiguous();
                cpu_cache_positions =
                    cache_positions.to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt64)
                        .contiguous();
                if (seq_len.has_value()) {
                    cpu_seq_len = seq_len.value()
                                      .to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt64)
                                      .contiguous();
                }
                if (effective_attention_mask.has_value()) {
                    cpu_attention_mask =
                        effective_attention_mask.value().to(mfq_tensor_backend::kCPU).contiguous();
                }
            }
            return model.prepare_hidden(
                tensor_to_cuda_device(model.execution->model_parallel_collectives, input_embeddings,
                                      primary)
                    .contiguous(),
                B, T);
        }
        Tensor layer(std::unique_ptr<Block> &b, Tensor x) {
            MfqCudaGuard block_guard(b->cuda_device);
            auto local_ids =
                b->cpu_offloaded
                    ? cpu_ids
                    : tensor_to_cuda_device(model.execution->model_parallel_collectives, ids,
                                            b->cuda_device);
            auto local_pos =
                b->cpu_offloaded
                    ? cpu_pos
                    : tensor_to_cuda_device(model.execution->model_parallel_collectives, pos,
                                            b->cuda_device);
            MfqOptional<mfq_tensor_backend::Tensor> local_cache_positions = mfq_nullopt;
            // Grid-MRoPE semantic coordinates never double as physical KV
            // slots, during either prepared prefill or delta-adjusted decode.
            if (model.passes_cache_positions(model.rope.sections.numel() > 0,
                                                   cache_positions_override.has_value())) {
                local_cache_positions =
                    b->cpu_offloaded
                        ? cpu_cache_positions
                        : tensor_to_cuda_device(model.execution->model_parallel_collectives,
                                                cache_positions, b->cuda_device);
            }
            MfqOptional<mfq_tensor_backend::Tensor> local_seq_len = mfq_nullopt;
            if (seq_len.has_value()) {
                local_seq_len =
                    b->cpu_offloaded
                        ? cpu_seq_len.value()
                        : tensor_to_cuda_device(model.execution->model_parallel_collectives,
                                                seq_len.value(), b->cuda_device);
            }
            MfqOptional<mfq_tensor_backend::Tensor> local_attention_mask = mfq_nullopt;
            if (effective_attention_mask.has_value()) {
                local_attention_mask =
                    b->cpu_offloaded
                        ? cpu_attention_mask.value()
                        : tensor_to_cuda_device(model.execution->model_parallel_collectives,
                                                effective_attention_mask.value(), b->cuda_device);
            }
            x = b->cpu_offloaded
                    ? x.to(mfq_tensor_backend::kCPU).contiguous()
                    : tensor_to_cuda_device(model.execution->model_parallel_collectives, x,
                                            b->cuda_device);
            b->set_token_ids(local_ids);
            const RopeCache &active_rope =
                b->cpu_offloaded
                    ? model.cpu_rope
                    : (model.device_ropes.empty() ? model.rope
                                                  : model.device_ropes.at(b->cuda_device));
            Block::Context context;
            context.token_ids = local_ids;
            context.host_token_ids=cpu_ids;
            context.positions = local_pos;
            context.full_positions =
                model.block_positions(full_positions, local_pos, b->cuda_device);
            context.cache_position = model.cache_pos;
            context.confirmed_prefix = confirmed_prefix;
            context.planned_kv_length = plan.kv_length;
            context.decode_attention_parts = plan.attention_parts;
            context.sequence_lengths = local_seq_len;
            context.cache_positions = local_cache_positions;
            if (model.passes_attention_mask()) {
                context.attention_mask = local_attention_mask;
            }
            x = b->forward_context(*model.execution, std::move(x), context, active_rope);
            return x;
        }
        void trace(const Tensor &x) {
            if (block_trace)
                block_trace->push_back(
                    tensor_to_cuda_device(model.execution->model_parallel_collectives, x, primary)
                        .to(mfq_tensor_backend::kFloat32)
                        .clone());
        }
        Tensor finish(Tensor x, int64_t B, int64_t T) {
            x = tensor_to_cuda_device(model.execution->model_parallel_collectives, x, primary);
            auto finalized = model.finalize_hidden(x, model.output_norm, B, T);
            if (raw_hidden)
                *raw_hidden = model.extract_raw_hidden(x, finalized);
            return finalized;
        }
    };
    static ForwardOps forward_ops(CausalModel &model, typename ForwardOps::Inputs input) {
        return ForwardOps(model, std::move(input));
    }
};

using Qwen35CausalLm = mfq::models::qwen35::CausalLm<CudaCausalOps<Qwen35Model>>;
using MiniCPMO45CausalLm = mfq::models::minicpmo45::CausalLm<CudaCausalOps<MiniCPMO45Model>>;
using MiniCPMOTtsCausalLm = mfq::models::minicpmo45::TtsCausalLm<CudaCausalOps<MiniCPMOTtsModel>>;
using Gemma4CausalLm = mfq::models::gemma4::CausalLm<CudaCausalOps<Gemma4Model>>;
using GlmDsaCausalLm = mfq::models::glm_dsa::CausalLm<CudaCausalOps<GlmDsaModel>>;
using Glm5CausalLm = mfq::models::glm5_next::CausalLm<CudaCausalOps<Glm5Model>>;
using Qwen4CausalLm = mfq::models::qwen4_exp::CausalLm<CudaCausalOps<Qwen4Model>>;
using DeepseekV4CausalLm = mfq::models::deepseek_v4::CausalLm<CudaCausalOps<DeepseekV4Model>>;
using DeepseekV41CausalLm = mfq::models::deepseek_v41::CausalLm<CudaCausalOps<DeepseekV41Model>>;

} // namespace mfq::cuda
