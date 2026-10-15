#include "resource_plan.h"
#include "causal_model_ops.h"
#include "mtp.h"

#include <algorithm>
#include <stdexcept>

namespace mfq::cuda {

void CudaResourcePlan::add(int device, std::initializer_list<std::size_t> factors) {
    std::size_t bytes = 1;
    for (auto factor : factors) {
        if (factor && bytes > std::size_t(-1) / factor)
            throw std::overflow_error("model memory geometry overflows size_t");
        bytes *= factor;
    }
    auto& total = request_bytes.at(device);
    if (bytes > std::size_t(-1) - total)
        throw std::overflow_error("model memory geometry overflows size_t");
    total += bytes;
}

CudaResourcePlan CausalResources::resource_plan(
        const std::vector<std::unique_ptr<Block>>& blocks, bool batch,
        std::size_t devices, const MtpModule* predictor) const {
    CudaResourcePlan plan(devices);
    for (const auto& block : blocks)
        if (!block->cpu_offloaded) block->plan_resources(plan, metadata, batch);
    // Copy factors include active state, rollback/capture and one cached snapshot.
    std::size_t snapshot = std::size_t(-1);
    for (auto bytes : plan.request_bytes) if (bytes)
        snapshot = std::min(snapshot, bytes / (batch ? 4 : 3));
    plan.snapshot_bytes = snapshot == std::size_t(-1) ? 0 : snapshot;
    plan.add(execution->layer_placement.primary_device(), {8, std::size_t(metadata.vocab_size)});
    if (predictor) predictor->plan_resources(plan, metadata);
    return plan;
}

} // namespace mfq::cuda

void Block::plan_resources(mfq::cuda::CudaResourcePlan& plan,
        const mfq::models::CausalLmMetadata& metadata, bool) const {
    // ponytail: serial compressed caches use a dense envelope; replace with
    // family geometry when real-weight memory measurements require it.
    plan.add(cuda_device, {8, std::size_t(metadata.max_position_embeddings),
                          std::size_t(metadata.hidden_size)});
}
