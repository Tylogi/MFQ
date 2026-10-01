#include "components.h"
#include <stdexcept>

template <typename Model>
RuntimeComponents<Model> load_runtime_components(
        Model& model,
        bool load_optional_components) {
    RuntimeComponents<Model> result;
    result.graph = model.graph;
    result.plan = model.plan;
    if (load_optional_components &&
            (result.plan.vision != mfq::cuda::CudaVisionAdapter::none ||
             result.plan.predictor != mfq::cuda::CudaPredictorAdapter::none)) {
        throw std::runtime_error(
            "CUDA component adapter is unsupported for this model");
    }
    return result;
}

template <typename Model>
std::unique_ptr<mfq::engine::ContinuousBatching>
make_cuda_continuous_batching(
        Model&,
        CudaExecutionContext&,
        std::mutex&,
        DecodeGraphCache&,
        mfq::cuda::internal::TextSessionCache&,
        RuntimeComponents<Model>&,
        const mfq::cuda::CudaRuntimeConfig&) {
    return {};
}

#define MFQ_INSTANTIATE_COMPONENTS(TYPE)                                  \
    template RuntimeComponents<TYPE> load_runtime_components(TYPE&, bool)

MFQ_INSTANTIATE_COMPONENTS(mfq::cuda::Gemma4CausalLm);
MFQ_INSTANTIATE_COMPONENTS(mfq::cuda::GlmDsaCausalLm);
MFQ_INSTANTIATE_COMPONENTS(mfq::cuda::DeepseekV4CausalLm);

#define MFQ_INSTANTIATE_BATCHING(TYPE)                                      \
    template std::unique_ptr<mfq::engine::ContinuousBatching>               \
    make_cuda_continuous_batching(                                          \
        TYPE&, CudaExecutionContext&, std::mutex&, DecodeGraphCache&,       \
        mfq::cuda::internal::TextSessionCache&, RuntimeComponents<TYPE>&,    \
        const mfq::cuda::CudaRuntimeConfig&)

MFQ_INSTANTIATE_BATCHING(mfq::cuda::MiniCPMO45CausalLm);
MFQ_INSTANTIATE_BATCHING(mfq::cuda::MiniCPMOTtsCausalLm);
MFQ_INSTANTIATE_BATCHING(mfq::cuda::Gemma4CausalLm);
MFQ_INSTANTIATE_BATCHING(mfq::cuda::GlmDsaCausalLm);
MFQ_INSTANTIATE_BATCHING(mfq::cuda::Glm5CausalLm);
MFQ_INSTANTIATE_BATCHING(mfq::cuda::Qwen4CausalLm);
MFQ_INSTANTIATE_BATCHING(mfq::cuda::DeepseekV4CausalLm);
MFQ_INSTANTIATE_BATCHING(mfq::cuda::DeepseekV41CausalLm);

#undef MFQ_INSTANTIATE_BATCHING
#undef MFQ_INSTANTIATE_COMPONENTS
