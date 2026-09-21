#pragma once

#include "causal_lm.h"
#include "mfq_cuda_mtp.h"
#include "mfq_tensor_backend.h"
#include "mfq/runtime.h"

#include <cstdint>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>

struct CudaPreparedPrompt;
struct RopeCache;

struct MtpTarget {
    std::function<mfq_tensor_backend::Tensor(mfq_tensor_backend::Tensor)> embed;
    std::function<mfq_tensor_backend::Tensor(mfq_tensor_backend::Tensor)> logits;
    const RopeCache* rope = nullptr;
};

struct MtpStep {
    mfq_tensor_backend::Tensor sample_hidden;
    mfq_tensor_backend::Tensor chain_hidden;
};

using MtpTokenSelector = std::function<std::int32_t(
    mfq_tensor_backend::Tensor)>;

struct MtpBlockDraft {
    mfq_tensor_backend::Tensor tokens;
    mfq_tensor_backend::Tensor logits;
    mfq_tensor_backend::Tensor confidence;
};

// Generation sees one predictor contract. Architecture-specific modules own their
// equations and cache layout; the runtime does not branch on architecture.
struct MtpModule {
    virtual ~MtpModule() = default;
    virtual void reset(int64_t batch = 1) = 0;
    virtual mfq_tensor_backend::Tensor forward(
        const MtpTarget& target,
        mfq_tensor_backend::Tensor previous_hidden,
        mfq_tensor_backend::Tensor next_ids) = 0;
    virtual MtpStep step(
        const MtpTarget& target,
        mfq_tensor_backend::Tensor previous_hidden,
        mfq_tensor_backend::Tensor next_ids) {
        auto hidden = forward(
            target, std::move(previous_hidden), std::move(next_ids));
        return {hidden, hidden};
    }
    virtual MtpStep step_positioned(
        const MtpTarget&,
        mfq_tensor_backend::Tensor,
        mfq_tensor_backend::Tensor,
        mfq_tensor_backend::Tensor) {
        throw std::runtime_error(
            "this CUDA MTP predictor does not accept explicit positions");
    }
    virtual int64_t cache_position() const noexcept = 0;
    virtual void trim_cache_to(int64_t position) = 0;
    virtual bool teacher_forced_prompt_prime() const noexcept = 0;
    virtual bool target_bootstrap_decode() const noexcept = 0;
    virtual bool preserve_output_dtype() const noexcept = 0;
    virtual int maximum_draft_depth() const noexcept {
        return mfq::cuda::mtp::kMaximumDraftDepth;
    }
    virtual bool blockwise_drafting() const noexcept { return false; }
    virtual bool split_target_verification() const noexcept { return false; }
    virtual bool retains_partial_target_prefix() const noexcept {
        return false;
    }
    virtual void append_target_context(
        mfq_tensor_backend::Tensor,
        std::int64_t) {
        throw std::runtime_error(
            "this CUDA MTP predictor has no target-context adapter");
    }
    virtual MtpBlockDraft draft_block(
        const MtpTarget&,
        mfq_tensor_backend::Tensor,
        const MtpTokenSelector&,
        int) {
        throw std::runtime_error(
            "this CUDA MTP predictor has no block-draft adapter");
    }

    mfq::cuda::mtp::GenerationStats last_stats;
    uint64_t last_cycles = 0;
    uint64_t last_accepted = 0;
    uint64_t last_rejected = 0;
};

template <mfq::cuda::CudaBackbone Backbone>
int32_t run_mtp_generation(
    mfq::cuda::CausalLmFor<Backbone>& model,
    MtpModule& mtp,
    const std::vector<int64_t>& prompt,
    const MfqSamplingParams& sampling,
    const MfqTokenCallback& on_token,
    const MfqPrefillCallback& on_prefill,
    int64_t prefill_chunk_size = 2048,
    const MfqTokenConstraintPtr& token_constraint = {},
    const CudaPreparedPrompt* prepared = nullptr);
