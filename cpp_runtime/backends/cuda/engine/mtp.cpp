#include "mtp.h"
#include "speculative_sequence.h"

#include "cuda_execution.h"
#include "cuda_sampling.h"
#include "generation_policy.h"
#include "inference.h"
#include "mfq_cuda_ops.h"
#include "models/causal_models.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

using mfq::cuda::internal::PrefillCudaTimer;

using mfq::engine::Generation;
using namespace mfq::engine;
template <class Model> struct CudaMtpOps {
  using Tensor = mfq_tensor_backend::Tensor;
  Model &model;
  MtpModule &mtp;
  const CudaPreparedPrompt *prepared;
  MtpTarget target;
  bool transformed_prompt;
  int position_axes;
  const mfq_tensor_backend::TensorOptions options =
      mfq_tensor_backend::TensorOptions()
          .device(mfq_tensor_backend::kCUDA)
          .dtype(mfq_tensor_backend::kInt64);
  Tensor input_ids, counts;
  std::optional<mfq::cuda::Sampler> sampler_;
  bool penalties = false;

  CudaMtpOps(Model &model, MtpModule &mtp, const CudaPreparedPrompt *prepared)
      : model(model), mtp(mtp), prepared(prepared), target(model),
        transformed_prompt(prepared && prepared->transformed()),
        position_axes(transformed_prompt && prepared->positions.dim() == 2 &&
                              prepared->positions.size(0) == 3
                          ? 3
                          : 1) {}
  void initialize(const std::vector<int64_t> &prompt,
                  const MfqSamplingParams &sampling) {
    input_ids = ids_for(prompt);
    auto random_host =
        mfq_tensor_backend::empty({1}, mfq_tensor_backend::TensorOptions()
                                           .device(mfq_tensor_backend::kCPU)
                                           .dtype(mfq_tensor_backend::kFloat32)
                                           .pinned_memory(true));
    auto random_gpu = mfq_tensor_backend::empty(
        {1}, options.dtype(mfq_tensor_backend::kFloat32));
    sampler_.emplace(sampling, mfq::cuda::SamplingOps(std::move(random_host),
                                                      std::move(random_gpu)));
    penalties = sampler().has_penalties();
    if (penalties) {
      counts = mfq_tensor_backend::zeros(
          {model.vocab_size()}, options.dtype(mfq_tensor_backend::kInt32));
      add_counts(counts, input_ids);
    }
  }
  auto &sampler() { return *sampler_; }
  static void add_counts(Tensor counts, const Tensor &ids) {
    sample_token_counts_add_cuda(counts, ids);
  }
  static bool defined(const Tensor &value) { return value.defined(); }
  static int64_t size(const Tensor &value, int axis) {
    return value.size(axis);
  }
  static int rank(const Tensor &value) { return value.dim(); }
  static Tensor clone(const Tensor &value) { return value.clone(); }
  static Tensor slice(const Tensor &value, int axis, int64_t offset,
                      int64_t length) {
    return value.narrow(axis, offset, length).contiguous();
  }
  static Tensor reshape(Tensor value, std::initializer_list<int64_t> shape) {
    return value.reshape(shape);
  }
  static Tensor concatenate(const std::vector<Tensor> &values, int axis) {
    return mfq_tensor_backend::cat(values, axis).contiguous();
  }
  Tensor target_forward(Tensor ids, Tensor *raw = nullptr,
                        int64_t confirmed = 0) {
    return model.hidden_forward(ids, mfq_nullopt, mfq_nullopt, nullptr,
                                mfq_nullopt, raw, confirmed);
  }
  Tensor target_suffix(Tensor ids, Tensor *raw) {
    return model.hidden_forward_speculative_suffix(ids, raw);
  }
  struct PrefillResult {
    Tensor hidden, raw;
    double elapsed;
  };
  PrefillResult prefill(mfq::engine::PrefillChunk chunk) {
    PrefillCudaTimer timer;
    Tensor raw;
    auto ids = input_ids.narrow(1, chunk.offset, chunk.count).contiguous();
    auto hidden =
        transformed_prompt
            ? model.hidden_forward_inputs(
                  ids,
                  prepared->embeddings.narrow(1, chunk.offset, chunk.count)
                      .contiguous(),
                  prepared->positions.narrow(-1, chunk.offset, chunk.count)
                      .contiguous(),
                  mfq_nullopt, nullptr, mfq_nullopt, true, mfq_nullopt, &raw)
            : target_forward(ids, &raw);
    MFQ_CUDA_CHECK(
        cudaEventRecord(timer.finished_event(), mfq_get_current_cuda_stream()));
    return {std::move(hidden), std::move(raw), timer.elapsed_ms()};
  }
  Tensor ids_for(std::vector<int64_t> tokens) {
    return mfq_tensor_backend::tensor(tokens, options)
        .reshape({1, -1})
        .contiguous();
  }
  Tensor decode_positions(int64_t cache_start, int64_t tokens) {
    MFQ_RUNTIME_CHECK(tokens > 0, "CUDA MTP position span must be positive");
    const int64_t logical_start =
        cache_start +
        (transformed_prompt ? prepared->decode_position_delta : 0);
    MFQ_RUNTIME_CHECK(logical_start >= 0 && logical_start + tokens <=
                                                model.max_position_embeddings(),
                      "CUDA MTP position span exceeds context capacity");
    auto result = mfq_tensor_backend::arange(logical_start,
                                             logical_start + tokens, options);
    return position_axes == 3
               ? result.reshape({1, tokens}).expand({3, tokens}).contiguous()
               : result;
  }
  MtpStep predictor_step(Tensor hidden, Tensor ids, Tensor positions) {
    return positions.defined()
               ? mtp.step_positioned(target, std::move(hidden), std::move(ids),
                                     std::move(positions))
               : mtp.step(target, std::move(hidden), std::move(ids));
  }
  int32_t sample_normal(Tensor logits, Tensor token_counts) {
    if (penalties)
      logits = logits.clone();
    return static_cast<int32_t>(
        mfq::cuda::sample_logits(sampler(), std::move(logits), token_counts)
            .template item<int64_t>());
  }
  int32_t sample_constrained(Tensor logits, Tensor token_counts,
                             const MfqTokenConstraintPtr &constraint) {
    if (penalties)
      logits = logits.clone();
    auto sampler_constraint =
        constraint ? constraint->clone() : MfqTokenConstraintPtr{};
    return static_cast<int32_t>(
        mfq::cuda::sample_logits(sampler(), std::move(logits), token_counts,
                                 sampler_constraint)
            .template item<int64_t>());
  }
  std::vector<float> probabilities(Tensor logits, Tensor token_counts,
                                   const MfqSamplingParams &parameters) {
    logits = logits.contiguous().reshape({1, -1});
    if (penalties) {
      logits = logits.clone();
      sample_apply_penalties_cuda(
          logits, token_counts, parameters.presence_penalty,
          parameters.frequency_penalty, parameters.repetition_penalty);
    }
    auto host = logits.to(mfq_tensor_backend::kFloat32).cpu().contiguous();
    return mfq::engine::mtp::distribution(
        std::span<const float>(host.template data_ptr<float>(), host.numel()),
        parameters.temperature, parameters.top_k, parameters.top_p);
  }
  mfq::engine::mtp::CompactDistribution
  compact_probabilities(Tensor logits, Tensor token_counts,
                        const MfqSamplingParams &parameters) {
    MFQ_RUNTIME_CHECK(parameters.top_k > 0 && parameters.top_k <= 64 &&
                          parameters.temperature > 0.0,
                      "compact CUDA MTP sampling requires top-k 1-64");
    logits = logits.contiguous().reshape({1, -1});
    if (penalties) {
      logits = logits.clone();
      sample_apply_penalties_cuda(
          logits, token_counts, parameters.presence_penalty,
          parameters.frequency_penalty, parameters.repetition_penalty);
    }
    auto selected =
        mfq_tensor_backend::topk(logits, parameters.top_k, -1, true, true);
    auto values =
        std::get<0>(selected)
            .to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kFloat32)
            .contiguous()
            .reshape({-1});
    auto indices = std::get<1>(selected)
                       .to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt64)
                       .contiguous()
                       .reshape({-1});
    return mfq::engine::mtp::compact_distribution_from_topk(
        values.template data_ptr<float>(), indices.template data_ptr<int64_t>(),
        static_cast<int>(values.numel()), parameters.temperature,
        parameters.top_p);
  }
  std::vector<mfq::engine::mtp::CompactDistribution>
  compact_probability_rows(Tensor logits, const MfqSamplingParams &parameters) {
    MFQ_RUNTIME_CHECK(!penalties && logits.dim() == 2 && parameters.top_k > 0 &&
                          parameters.top_k <= 64 &&
                          parameters.temperature > 0.0,
                      "batched compact CUDA MTP sampling geometry is invalid");
    logits = logits.contiguous();
    auto selected =
        mfq_tensor_backend::topk(logits, parameters.top_k, -1, true, true);
    auto values =
        std::get<0>(selected)
            .to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kFloat32)
            .contiguous();
    auto indices = std::get<1>(selected)
                       .to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt64)
                       .contiguous();
    const int64_t rows = values.size(0);
    const int64_t columns = values.size(1);
    const float *value_data = values.template data_ptr<float>();
    const int64_t *index_data = indices.template data_ptr<int64_t>();
    std::vector<mfq::engine::mtp::CompactDistribution> result;
    result.reserve(static_cast<size_t>(rows));
    for (int64_t row = 0; row < rows; ++row) {
      result.push_back(mfq::engine::mtp::compact_distribution_from_topk(
          value_data + row * columns, index_data + row * columns,
          static_cast<int>(columns), parameters.temperature, parameters.top_p));
    }
    return result;
  }
  Tensor logits_for(Tensor normalized) {
    return model.logits_from_hidden(
        (mtp.preserve_output_dtype()
             ? normalized
             : normalized.to(mfq_tensor_backend::kFloat16))
            .contiguous());
  }
};
template <typename Model>
Generation
run_mtp_generation(Model &model, MtpModule &mtp, InferenceRequest &request,
                   InferenceOutput &output, int64_t prefill_chunk_size,
                   const CudaPreparedPrompt *prepared,
                   std::size_t reused_tokens,
                   mfq_tensor_backend::Tensor restored_last_hidden,
                   mfq_tensor_backend::Tensor *session_last_hidden) {
  return mfq::engine::speculative_sequence(
      CudaMtpOps<Model>{model, mtp, prepared}, request, output,
      prefill_chunk_size, reused_tokens, std::move(restored_last_hidden),
      session_last_hidden);
}
#define MFQ_INSTANTIATE_MTP(MODEL)                                             \
  template Generation run_mtp_generation<MODEL>(                               \
      MODEL &, MtpModule &, InferenceRequest &, InferenceOutput &, int64_t,    \
      const CudaPreparedPrompt *, std::size_t, mfq_tensor_backend::Tensor,     \
      mfq_tensor_backend::Tensor *)

MFQ_INSTANTIATE_MTP(mfq::cuda::Qwen35CausalLm);
MFQ_INSTANTIATE_MTP(mfq::cuda::MiniCPMO45CausalLm);
MFQ_INSTANTIATE_MTP(mfq::cuda::MiniCPMOTtsCausalLm);
MFQ_INSTANTIATE_MTP(mfq::cuda::Gemma4CausalLm);
MFQ_INSTANTIATE_MTP(mfq::cuda::GlmDsaCausalLm);
MFQ_INSTANTIATE_MTP(mfq::cuda::Glm5CausalLm);
MFQ_INSTANTIATE_MTP(mfq::cuda::Qwen4CausalLm);
MFQ_INSTANTIATE_MTP(mfq::cuda::DeepseekV4CausalLm);
MFQ_INSTANTIATE_MTP(mfq::cuda::DeepseekV41CausalLm);

#undef MFQ_INSTANTIATE_MTP
