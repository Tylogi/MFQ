#pragma once

#include "storage/session_state.h"
#include "mtp_policy.h"
#include "mfq_tensor_backend.h"
#include "models/common/causal_model_ops.h"

#include <cstdint>
#include <variant>
#include <stdexcept>
#include <utility>

struct RopeCache;

struct MtpTarget {
    std::variant<mfq::cuda::Qwen35CausalLm*, mfq::cuda::MiniCPMO45CausalLm*,
        mfq::cuda::MiniCPMOTtsCausalLm*, mfq::cuda::Gemma4CausalLm*,
        mfq::cuda::GlmDsaCausalLm*, mfq::cuda::Glm5CausalLm*, mfq::cuda::Qwen4CausalLm*,
        mfq::cuda::DeepseekV4CausalLm*, mfq::cuda::DeepseekV41CausalLm*> model;
    const RopeCache* rope = nullptr;
    template <class Model> explicit MtpTarget(Model& target) : model(&target), rope(&target.rope) {}
    mfq_tensor_backend::Tensor embed(mfq_tensor_backend::Tensor ids) const;
    mfq_tensor_backend::Tensor logits(mfq_tensor_backend::Tensor hidden) const;
};

struct MtpStep {
    mfq_tensor_backend::Tensor sample_hidden;
    mfq_tensor_backend::Tensor chain_hidden;
};

// Architecture-specific predictors implement device math and cache state only.
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
        return mfq::engine::mtp::kMaximumDraftDepth;
    }
    virtual bool blockwise_drafting() const noexcept { return false; }
    virtual bool split_target_verification() const noexcept { return false; }
    virtual bool retains_partial_target_prefix() const noexcept {
        return false;
    }
    virtual bool supports_session_state() const noexcept { return false; }
    virtual MtpSessionState capture_session_state(
        std::int64_t,
        const mfq_tensor_backend::Tensor&) const {
        throw std::runtime_error(
            "this CUDA MTP predictor has no session state");
    }
    virtual void restore_session_state(const MtpSessionState&) {
        throw std::runtime_error(
            "this CUDA MTP predictor has no session state");
    }
    virtual void append_target_context(
        mfq_tensor_backend::Tensor,
        std::int64_t) {
        throw std::runtime_error(
            "this CUDA MTP predictor has no target-context adapter");
    }
    virtual mfq_tensor_backend::Tensor draft_block(
        const MtpTarget&,
        mfq_tensor_backend::Tensor,
        int) {
        throw std::runtime_error(
            "this CUDA MTP predictor has no block-draft adapter");
    }
    virtual mfq_tensor_backend::Tensor draft_next(
        mfq_tensor_backend::Tensor,
        mfq_tensor_backend::Tensor) {
        throw std::runtime_error("this CUDA MTP predictor has no block-draft adapter");
    }

    mfq::engine::mtp::GenerationStats last_stats;
    uint64_t last_cycles = 0;
    uint64_t last_accepted = 0;
    uint64_t last_rejected = 0;
};
