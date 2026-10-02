#pragma once
#include "generation.h"
#include "models/causal_ops.h"
#include "models/mtp.h"

template <typename Model>
mfq::engine::Generation run_mtp_generation(
    Model& model, MtpModule& mtp, mfq::engine::InferenceRequest& request,
    mfq::engine::InferenceOutput& output, int64_t prefill_chunk_size = 2048,
    const CudaPreparedPrompt* prepared = nullptr, std::size_t reused_tokens = 0,
    mfq_tensor_backend::Tensor restored_last_hidden = {},
    mfq_tensor_backend::Tensor* session_last_hidden = nullptr);
