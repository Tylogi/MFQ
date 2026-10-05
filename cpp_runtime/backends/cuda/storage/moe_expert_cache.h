#pragma once

#include "moe_cache_profile.h"

#include <cstdint>
#include <memory>
#include <ostream>
#include <string>

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
void finalize_moe_expert_cache(
    const std::shared_ptr<MoeExpertCache>& cache);
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
