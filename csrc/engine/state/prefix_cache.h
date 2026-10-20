#pragma once
#include "prefix_blocks.h"
#include <mutex>

namespace mfq::cache {
struct PrefixMatch {
    std::size_t matched_tokens = 0;
    std::vector<BlockHash> blocks;
};

struct PagedPrefixCacheMetrics : PrefixBlockStoreMetrics {
    std::uint64_t queries=0,hits=0,hit_tokens=0;
};
// Logical prefix identity and matching own one host byte store.
class PagedPrefixCache {
private:
    // Counters outlive the storage worker and its reservation callbacks.
    mutable std::mutex metrics_mutex_;
    std::uint64_t queries_=0,hits_=0,hit_tokens_=0;
public:
    explicit PagedPrefixCache(PrefixBlockStoreConfig config);
    std::size_t block_size_tokens() const noexcept {return blocks.block_size_tokens();}
    const BlockHash& compatibility_hash() const noexcept {return blocks.compatibility_hash();}
    BlockHash block_hash(const BlockHash& parent,const std::int64_t* tokens,
        std::size_t count,std::string_view extra_key={}) const;
    PrefixMatch match(const std::vector<std::int64_t>& tokens,std::string_view extra_key={},bool record_query=true);
    void record_match(std::size_t matched_tokens);
    BlockHash store(const BlockHash& parent,const std::int64_t* tokens,std::size_t count,
        PagedPrefixPayload payload,std::string_view extra_key={});
    BlockHash replace(const BlockHash& parent,const std::int64_t* tokens,std::size_t count,
        PagedPrefixPayload payload,std::string_view extra_key={});
    PagedPrefixCacheMetrics metrics() const;
    PrefixBlockStore blocks;
};
} // namespace mfq::cache
