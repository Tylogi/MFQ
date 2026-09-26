#pragma once

#include "moe_cache_profile.h"

#include <cstdint>
#include <memory>
#include <ostream>

class MoeExpertCache;

extern std::shared_ptr<MoeExpertCache> g_moe_expert_cache;

std::shared_ptr<MoeExpertCache> make_moe_expert_cache(std::int64_t bytes);
bool moe_expert_cache_has_sources();
bool moe_expert_cache_finalized();
void finalize_moe_expert_cache();
void print_moe_expert_cache_stats(std::ostream& output);
void set_moe_expert_cache_profile(mfq::MoeCacheProfile profile);
