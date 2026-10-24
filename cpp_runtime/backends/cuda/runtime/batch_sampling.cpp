#include "runtime/batch_sampling.h"

#include <algorithm>
#include <cstdlib>
#include <climits>

namespace mfq::cuda {

void BatchSampling::reserve(int rows, int vocabulary, mfq_tensor_backend::Device device) {
    MFQ_RUNTIME_CHECK(rows > 0 && rows <= 32767 && vocabulary > 0 && device.is_cuda(),
        "batch sampling: invalid capacity or device");
    if (capacity_ >= rows && vocabulary_ == vocabulary && metadata_cuda_.device() == device) return;
    const int64_t bytes = 2 * int64_t(rows) * sizeof(SamplingRow) + int64_t(rows) * vocabulary * sizeof(float);
    const auto host_options = mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCPU)
        .dtype(mfq_tensor_backend::kUInt8).pinned_memory(true);
    auto host = mfq_tensor_backend::empty({bytes}, host_options);
    auto cuda = mfq_tensor_backend::empty({bytes}, host_options.device(device).pinned_memory(false));
    auto result = mfq_tensor_backend::empty({2 * rows}, host_options.dtype(mfq_tensor_backend::kInt64));
    metadata_host_ = std::move(host);
    metadata_cuda_ = std::move(cuda);
    result_host_ = std::move(result);
    capacity_ = rows;
    vocabulary_ = vocabulary;
}

const SamplingRow* BatchSampling::metadata() const {
    return static_cast<const SamplingRow*>(metadata_cuda_.data_ptr());
}

const SamplingPlan& BatchSampling::prepare(const Tensor& logits, const std::vector<BatchSamplingInput>& inputs) {
    MFQ_RUNTIME_CHECK(logits.is_cuda() && logits.dim() == 2 && logits.size(1) <= INT_MAX &&
        logits.size(0) == int64_t(inputs.size()), "batch sampling: inputs must match CUDA logit rows");
    reserve(int(logits.size(0)), int(logits.size(1)), logits.device());
    inputs_ = inputs;
    plan_ = {};
    plan_.rows = int(inputs.size());
    plan_.candidates = plan_.rows;
    auto* rows = static_cast<SamplingRow*>(metadata_host_.data_ptr());
    for (int row = 0; row < 2 * capacity_; ++row) rows[row] = SamplingRow{};
    const int64_t mask_offset = 2 * int64_t(capacity_) * sizeof(SamplingRow);
    auto* masks = reinterpret_cast<float*>(metadata_host_.data_ptr<uint8_t>() + mask_offset);
    auto* device_masks = reinterpret_cast<float*>(metadata_cuda_.data_ptr<uint8_t>() + mask_offset);
    const auto* radix_option = std::getenv("MFQ_SAMPLE_RADIX_TOPK");
    const bool radix = !radix_option || radix_option[0] != '0';
    int mask_count = 0;
    for (int row = 0; row < plan_.rows; ++row) {
        auto& packed = rows[row];
        packed.source = row;
        const auto& input = inputs[row];
        if (!input.sampler) continue;
        auto& sampler = *input.sampler;
        const auto& params = sampler.params();
        const bool greedy = sampler.greedy();
        MFQ_RUNTIME_CHECK(greedy || (params.temperature > 0 && params.top_k <= vocabulary_ && params.top_k <= 1024),
            "batch sampling: invalid temperature or top-k");
        packed.temperature = float(params.temperature);
        packed.top_k = params.top_k;
        packed.top_p = float(params.top_p);
        if (greedy) {
            packed.kind = SamplingKind::greedy;
            plan_.greedy = true;
        } else if (params.top_k == 0) {
            packed.kind = SamplingKind::softmax;
            plan_.softmax = true;
        } else if (params.top_k <= 64 && radix) {
            packed.kind = SamplingKind::radix;
            plan_.radix_k = std::max(plan_.radix_k, params.top_k);
        } else {
            packed.kind = SamplingKind::fallback;
            plan_.fallback = true;
        }
        if (sampler.has_penalties()) {
            MFQ_RUNTIME_CHECK(input.counts.is_cuda() && input.counts.is_contiguous() &&
                input.counts.scalar_type() == mfq_tensor_backend::kInt32 && input.counts.dim() == 1 &&
                input.counts.numel() == vocabulary_ && input.counts.device() == logits.device(),
                "batch sampling: counts must be CUDA int32 [V] on the logits device");
            packed.counts = input.counts.data_ptr<int32_t>();
            packed.presence = float(params.presence_penalty);
            packed.frequency = float(params.frequency_penalty);
            packed.repetition = float(params.repetition_penalty);
            plan_.transform = true;
        }
        // Preserve mt19937_64/uniform_real_distribution and its draw count.
        // Only a rejected first candidate commits a second real RNG draw.
        if (!greedy) packed.random = sampler.next_uniform_float();
        if (input.constraint) {
            auto* mask = masks + int64_t(mask_count) * vocabulary_;
            std::fill_n(mask, vocabulary_, 0.0f);
            input.constraint->apply(mask, vocabulary_);
            packed.mask = device_masks + int64_t(mask_count++) * vocabulary_;
            packed.retry = plan_.candidates++;
            rows[packed.retry] = packed;
            rows[packed.retry].retry = -1;
            if (!greedy) {
                auto retry_sampler = sampler;
                rows[packed.retry].random = retry_sampler.next_uniform_float();
            }
            plan_.constrained = true;
            plan_.transform = true;
        }
    }
    const int64_t bytes = mask_offset + int64_t(mask_count) * vocabulary_ * sizeof(float);
    MFQ_CUDA_CHECK(cudaMemcpyAsync(metadata_cuda_.data_ptr(), metadata_host_.data_ptr(), bytes,
        cudaMemcpyHostToDevice, mfq_current_cuda_stream()));
    return plan_;
}

BatchSampling::Tensor BatchSampling::run(Tensor logits) const {
    return sample_batch_cuda(std::move(logits), metadata(), plan_);
}

BatchSamplingResult BatchSampling::finish(Tensor output) {
    // Replay can retain its captured retry lanes while constrained rows pause.
    const int values = output.defined() ? int(output.numel()) : 0;
    MFQ_RUNTIME_CHECK(output.is_cuda() && (values == plan_.rows || values == 2 * plan_.rows) &&
        (!plan_.constrained || values == 2 * plan_.rows) &&
        output.scalar_type() == mfq_tensor_backend::kInt64, "batch sampling: invalid output packet");
    auto host = result_host_.narrow(0, 0, values);
    MFQ_CUDA_CHECK(cudaMemcpyAsync(host.data_ptr(), output.data_ptr(), values * sizeof(int64_t),
        cudaMemcpyDeviceToHost, mfq_current_cuda_stream()));
    MFQ_CUDA_CHECK(cudaStreamSynchronize(mfq_current_cuda_stream()));
    const auto* tokens = host.data_ptr<int64_t>();
    for (int row = 0; row < plan_.rows; ++row) {
        const auto& input = inputs_[row];
        if (!input.sampler || !input.constraint) continue;
        if (tokens[plan_.rows + row] == 2) {
            throw std::runtime_error("chat-template grammar rejected every token");
        }
        if (tokens[plan_.rows + row] && !input.sampler->greedy()) input.sampler->next_uniform_float();
        if (!input.constraint->allows(tokens[row])) {
            throw std::runtime_error("CUDA constrained sampler returned an invalid token");
        }
        input.constraint->accept(tokens[row]);
    }
    return {output.narrow(0, 0, plan_.rows), host.narrow(0, 0, plan_.rows)};
}

BatchSamplingResult BatchSampling::sample(Tensor logits, const std::vector<BatchSamplingInput>& inputs) {
    logits = logits.contiguous();
    prepare(logits, inputs);
    return finish(run(std::move(logits)));
}

} // namespace mfq::cuda
