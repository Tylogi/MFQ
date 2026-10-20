#include "prefix_cache.h"
#include <utility>
namespace mfq::cache {
PagedPrefixCache::PagedPrefixCache(PrefixBlockStoreConfig config):blocks(std::move(config)) {}
BlockHash PagedPrefixCache::block_hash(const BlockHash& parent,const std::int64_t* tokens,
        std::size_t count,std::string_view extra_key) const {
    return prefix_block_hash(compatibility_hash(),block_size_tokens(),parent,tokens,count,extra_key);
}
PrefixMatch PagedPrefixCache::match(const std::vector<std::int64_t>& tokens,
        std::string_view extra_key,bool record_query) {
    PrefixMatch result;
    BlockHash parent{};
    const auto size=block_size_tokens();
    if(record_query) {std::lock_guard<std::mutex> lock(metrics_mutex_);++queries_;}
    for(std::size_t index=0;index<tokens.size()/size;++index) {
        const auto hash=block_hash(parent,tokens.data()+index*size,size,extra_key);
        if(!blocks.contains(hash,parent,size))break;
        parent=hash;
        result.blocks.push_back(hash);
        result.matched_tokens+=size;
    }
    for(const auto count:blocks.tail_lengths(parent,tokens.size()-result.matched_tokens)) {
        const auto hash=block_hash(parent,tokens.data()+result.matched_tokens,count,extra_key);
        if(!blocks.contains(hash,parent,count))continue;
        result.blocks.push_back(hash);
        result.matched_tokens+=count;
        break;
    }
    if(record_query && result.matched_tokens) {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        ++hits_;hit_tokens_+=result.matched_tokens;
    }
    return result;
}
void PagedPrefixCache::record_match(std::size_t tokens) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    ++queries_;
    if(tokens) {++hits_;hit_tokens_+=tokens;}
}
BlockHash PagedPrefixCache::store(const BlockHash& parent,const std::int64_t* tokens,
        std::size_t count,PagedPrefixPayload payload,std::string_view extra_key) {
    const auto hash=block_hash(parent,tokens,count,extra_key);
    return blocks.store(hash,parent,tokens,count,std::move(payload),extra_key,false);
}
BlockHash PagedPrefixCache::replace(const BlockHash& parent,const std::int64_t* tokens,
        std::size_t count,PagedPrefixPayload payload,std::string_view extra_key) {
    const auto hash=block_hash(parent,tokens,count,extra_key);
    return blocks.store(hash,parent,tokens,count,std::move(payload),extra_key,true);
}
PagedPrefixCacheMetrics PagedPrefixCache::metrics() const {
    PagedPrefixCacheMetrics result;
    static_cast<PrefixBlockStoreMetrics&>(result)=blocks.metrics();
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    result.queries=queries_;result.hits=hits_;result.hit_tokens=hit_tokens_;
    return result;
}
} // namespace mfq::cache
