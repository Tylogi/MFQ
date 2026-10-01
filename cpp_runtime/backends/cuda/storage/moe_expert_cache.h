#pragma once

#include "moe_cache_profile.h"

#include <cstdint>
#include <memory>
#include <ostream>

class MoeExpertCache;
struct CudaExecutionConfig;

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
