#pragma once

#include "../../ops/rope.h"

#include <cstdint>
#include <vector>

struct CudaExecutionContext;

mfq_tensor_backend::Tensor qwen_rms_norm(
    mfq_tensor_backend::Tensor x,
    mfq_tensor_backend::Tensor weight,
    double eps,
    double weight_offset);

mfq_tensor_backend::Tensor qwen_rms_norm_bf16(
    const CudaExecutionConfig& config,
    mfq_tensor_backend::Tensor x,
    mfq_tensor_backend::Tensor weight,
    double eps,
    double weight_offset);

struct Block {
    struct Context {
        mfq_tensor_backend::Tensor token_ids;
        mfq_tensor_backend::Tensor positions;
        mfq_tensor_backend::Tensor full_positions;
        int64_t cache_position = 0;
        int64_t confirmed_prefix = 0;
        int64_t planned_kv_length = 0;
        int64_t decode_attention_parts = 0;
        MfqOptional<mfq_tensor_backend::Tensor> sequence_lengths = mfq_nullopt;
        MfqOptional<mfq_tensor_backend::Tensor> cache_positions = mfq_nullopt;
        MfqOptional<mfq_tensor_backend::Tensor> attention_mask = mfq_nullopt;
    };

    int cuda_device = 0;
    bool cpu_offloaded = false;

    virtual ~Block() = default;
    virtual void reset(int64_t batch) = 0;
    virtual void set_token_ids(const mfq_tensor_backend::Tensor&) {}
    virtual bool supports_speculation() const noexcept { return false; }
    virtual std::vector<mfq_tensor_backend::Tensor*> graph_warmup_state() { return {}; }
    virtual void begin_speculative(int64_t) {}
    virtual void commit_speculative() {}
    virtual void rollback_speculative(int64_t) {}
    virtual mfq_tensor_backend::Tensor forward(
        CudaExecutionContext& execution,
        mfq_tensor_backend::Tensor x,
        mfq_tensor_backend::Tensor positions,
        int64_t cache_position,
        const MfqOptional<mfq_tensor_backend::Tensor>& sequence_lengths,
        const RopeCache& rope,
        const MfqOptional<mfq_tensor_backend::Tensor>& cache_positions = mfq_nullopt,
        const MfqOptional<mfq_tensor_backend::Tensor>& attention_mask = mfq_nullopt) = 0;
    virtual mfq_tensor_backend::Tensor forward_context(
        CudaExecutionContext& execution,
        mfq_tensor_backend::Tensor x,
        const Context& context,
        const RopeCache& rope);
};

using mfq_tensor_backend::indexing::Slice;
