#pragma once

#include "moe_cache_profile.h"

#include <cstdint>
#include <memory>
#include <ostream>
#include <string>
#include <vector>
#include <utility>

class MoeExpertCache;
class MoeQuantRangeSource;
struct CudaExecutionConfig;

// One cache per Engine load, shared by its target and optional predictor weights.
// Registered weights retain the cache; resetting the Engine's execution resources
// drops only its ownership. Cache APIs receive the resource, never the execution.
std::shared_ptr<MoeExpertCache> make_moe_expert_cache(
    std::int64_t bytes,
    const CudaExecutionConfig& config);
bool moe_expert_cache_has_sources(
    const std::shared_ptr<MoeExpertCache>& cache);
bool moe_expert_cache_finalized(
    const std::shared_ptr<MoeExpertCache>& cache);
std::vector<std::pair<std::string, double>> moe_expert_memory_metrics(
    const std::shared_ptr<MoeExpertCache>& cache);
void finalize_moe_expert_cache(
    const std::shared_ptr<MoeExpertCache>& cache);
double moe_expert_cache_ram_pcie_fraction(const std::shared_ptr<MoeExpertCache>& cache);
// Finish the adaptive transaction before a session reset or a prefill loan.
void finish_moe_expert_exchanges(const std::shared_ptr<MoeExpertCache>& cache);
// Return prompt-only ring and workspace storage before preparing decode graphs.
void release_moe_prefill_buffers(MoeExpertCache* cache);
inline void release_moe_prefill_buffers(const std::shared_ptr<MoeExpertCache>& cache) {
    release_moe_prefill_buffers(cache.get());
}
// Idle diagnostic passes retain primary weights, clear transfer-cache entries,
// and replay the CPU assignments captured by the first pass.
void prepare_moe_pipeline_comparison(const std::shared_ptr<MoeExpertCache>& cache, bool replay);
// The CPU policy pass uses natural routing and measured dispatch. Baseline
// passes retain their exact CPU assignments; report any policy route changes.
void prepare_moe_cpu_budget_comparison(const std::shared_ptr<MoeExpertCache>& cache,
    bool replay,bool transfer_budget);
// Compare synchronous/background calibration with fresh cost samples each pass.
void prepare_moe_cpu_calibration_comparison(const std::shared_ptr<MoeExpertCache>& cache,bool replay);
std::size_t finish_moe_pipeline_comparison(const std::shared_ptr<MoeExpertCache>& cache);
void print_moe_expert_cache_stats(
    const std::shared_ptr<MoeExpertCache>& cache,
    std::ostream& output);
void set_moe_expert_cache_profile(
    MoeExpertCache& cache,
    mfq::MoeCacheProfile profile);

struct MfeWeight;
struct MixedMoeRuntime;
namespace mfq::cuda { class MfeMxfp4ExpertStore; }

MfeWeight cache_moe_weight(
    const std::shared_ptr<MoeExpertCache>& cache,
    const std::string& name,
    const std::shared_ptr<MixedMoeRuntime>& runtime,
    int minimum_slots,
    int layer_id,
    const std::string& projection_role,
    std::shared_ptr<mfq::cuda::MfeMxfp4ExpertStore> range_store = {});

MfeWeight cache_quant_moe_weight(const std::shared_ptr<MoeExpertCache>& cache,
    const std::string& name,std::shared_ptr<MoeQuantRangeSource> source,
    int minimum_slots,int layer_id,const std::string& projection_role);
