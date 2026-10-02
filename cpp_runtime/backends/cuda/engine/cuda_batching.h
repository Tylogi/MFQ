#pragma once
#include "request_executor.h"
#include "cuda_runtime_config.h"
#include <memory>
struct CudaExecutionContext;
namespace mfq::models { template <class Backend> struct CausalLm; }

namespace mfq::cuda {
struct Qwen35Model;
template <typename Model> struct CudaCausalOps;
template <typename Model> using CausalLm = mfq::models::CausalLm<CudaCausalOps<Model>>;
using Qwen35CausalLm = CausalLm<Qwen35Model>;
bool qwen_continuous_batch_cuda_graph_enabled(const Qwen35CausalLm&, const CudaContinuousBatchConfig&);
class QwenBatchExecutor final {
public:
    QwenBatchExecutor(Qwen35CausalLm&, CudaExecutionContext&, const CudaContinuousBatchConfig&, std::int64_t chunk);
    ~QwenBatchExecutor();
    void admit(std::string id, mfq::engine::ExecutionRequest&);
    void step(const std::vector<std::string>& eligible);
    std::vector<std::pair<std::string, double>> metrics() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace mfq::cuda
