#pragma once

#include "models/causal_lm.h"
#include "models/mtp.h"
#include "mfq/runtime.h"

#include <cstdint>
#include <vector>

struct CudaPreparedPrompt;

template <typename Model>
int32_t run_mtp_generation(
    Model& model,
    MtpModule& mtp,
    const std::vector<int64_t>& prompt,
    const MfqSamplingParams& sampling,
    const MfqTokenCallback& on_token,
    const MfqPrefillCallback& on_prefill,
    int64_t prefill_chunk_size = 2048,
    const MfqTokenConstraintPtr& token_constraint = {},
    const CudaPreparedPrompt* prepared = nullptr,
    std::size_t reused_tokens = 0,
    const mfq_tensor_backend::Tensor& restored_last_hidden = {},
    mfq_tensor_backend::Tensor* session_last_hidden = nullptr,
    double multimodal_ms = 0.0,
    const MfqCancellationCheck& cancelled = {});
