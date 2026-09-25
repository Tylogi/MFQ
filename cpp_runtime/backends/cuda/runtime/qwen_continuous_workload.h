#pragma once

#include "mfq/server.h"

#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace mfq::cuda {

// A small public surface for tools that exercise the Qwen CUDA scheduler.
// The model, serial generator and batcher remain owned by the runtime.
class QwenContinuousWorkload {
public:
    virtual ~QwenContinuousWorkload() = default;
    virtual int64_t vocab_size() const = 0;
    virtual int64_t max_position_embeddings() const = 0;
    virtual std::vector<int64_t> serial_generate(
        const std::vector<int64_t>& prompt,
        const MfqSamplingParams& sampling) = 0;
    virtual void start_batcher() = 0;
    virtual int32_t submit(
        const std::vector<int64_t>& prompt,
        const MfqSamplingParams& sampling,
        const MfqTokenCallback& on_token,
        const MfqPrefillCallback& on_prefill) = 0;
    virtual std::vector<std::pair<std::string, double>> metrics() const = 0;
};

using QwenWorkloadFn = std::function<int(QwenContinuousWorkload&)>;

int run_qwen_continuous_workload(
    const std::string& model_path,
    const std::string& config_path,
    int64_t context_size,
    int64_t prefill_chunk_size,
    const QwenWorkloadFn& run);

} // namespace mfq::cuda
