#include "models/deepseek_v4/ops.h"
#include "models/deepseek_v41/ops.h"
#include "models/gemma4/ops.h"
#include "models/glm5_next/ops.h"
#include "models/glm_dsa/ops.h"
#include "models/minicpmo45/ops.h"
#include "models/qwen35/ops.h"
#include "models/qwen4_exp/ops.h"

#include "models/common/model_loading.tpp"
#include "models/common/mtp.h"
#include "storage/text_session_cache_impl.h"

#define MFQ_BIND_CUDA_MODEL(TYPE) \
    template TYPE mfq::cuda::load_causal_lm<TYPE>(CudaExecutionContext&, const std::string&, const std::string&, int64_t, bool, bool, std::shared_ptr<const mfq::ModelSource>); \
    template mfq::StepSequence<mfq::cuda::internal::TextSessionRestore> mfq::cuda::internal::TextSessionCache::restore_steps(TYPE&, MtpModule*, const std::string&, const std::vector<int64_t>&, size_t, const std::string&); \
    template mfq::cuda::internal::TextSessionRestore mfq::cuda::internal::TextSessionCache::restore_best(TYPE&, MtpModule*, const std::string&, const std::vector<int64_t>&, size_t, const std::string&);

MFQ_BIND_CUDA_MODEL(mfq::cuda::Qwen35CausalLm)
MFQ_BIND_CUDA_MODEL(mfq::cuda::MiniCPMO45CausalLm)
MFQ_BIND_CUDA_MODEL(mfq::cuda::MiniCPMOTtsCausalLm)
MFQ_BIND_CUDA_MODEL(mfq::cuda::Gemma4CausalLm)
MFQ_BIND_CUDA_MODEL(mfq::cuda::GlmDsaCausalLm)
MFQ_BIND_CUDA_MODEL(mfq::cuda::Glm5CausalLm)
MFQ_BIND_CUDA_MODEL(mfq::cuda::Qwen4CausalLm)
MFQ_BIND_CUDA_MODEL(mfq::cuda::DeepseekV4CausalLm)
MFQ_BIND_CUDA_MODEL(mfq::cuda::DeepseekV41CausalLm)
