#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace mfq::cache {

using BlockHash = std::array<std::uint8_t, 32>;
using PagedPrefixPayload =
    std::shared_ptr<const std::vector<std::uint8_t>>;

BlockHash sha256(const void* data, std::size_t size);
BlockHash sha256(std::string_view value);
std::string block_hash_hex(const BlockHash& hash);

struct PagedPrefixCacheConfig {
    std::filesystem::path cache_dir;
    std::string compatibility_key;
    std::size_t block_size_tokens = 256;
    std::uint64_t max_disk_bytes = 100ULL * 1024ULL * 1024ULL * 1024ULL;
    std::uint64_t max_hot_bytes = 0;
    std::size_t max_pending_writes = 64;
    // Zero forces writes onto the caller instead of retaining raw KV payloads
    // in the asynchronous queue.  The count limit remains a second guard.
    std::uint64_t max_pending_bytes = 512ULL * 1024ULL * 1024ULL;
    // Cold blocks are independent content-addressed files. Reading and
    // checksumming a matched chain in parallel substantially reduces restore
    // latency without involving a backend device or its command stream.
    std::size_t max_parallel_reads = 4;
};

struct PrefixMatch {
    std::size_t matched_tokens = 0;
    std::vector<BlockHash> blocks;
};

struct PagedPrefixCacheMetrics {
    std::uint64_t queries = 0;
    std::uint64_t hits = 0;
    std::uint64_t hit_tokens = 0;
    std::uint64_t disk_hits = 0;
    std::uint64_t hot_hits = 0;
    std::uint64_t writes = 0;
    std::uint64_t deduplicated_writes = 0;
    std::uint64_t failed_writes = 0;
    std::uint64_t corrupt_blocks = 0;
    std::uint64_t evictions = 0;
    std::uint64_t disk_blocks = 0;
    std::uint64_t disk_bytes = 0;
    std::uint64_t hot_blocks = 0;
    std::uint64_t hot_bytes = 0;
    std::uint64_t pending_writes = 0;
    std::uint64_t pending_bytes = 0;
    std::uint64_t pending_max_bytes = 0;
};

class PagedPrefixCache {
public:
    explicit PagedPrefixCache(PagedPrefixCacheConfig config);
    ~PagedPrefixCache();

    PagedPrefixCache(const PagedPrefixCache&) = delete;
    PagedPrefixCache& operator=(const PagedPrefixCache&) = delete;

    std::size_t block_size_tokens() const noexcept;
    const BlockHash& compatibility_hash() const noexcept;

    BlockHash block_hash(
        const BlockHash& parent,
        const std::int64_t* token_ids,
        std::size_t token_count,
        std::string_view extra_key = {}) const;

    PrefixMatch match(
        const std::vector<std::int64_t>& token_ids,
        std::string_view extra_key = {},
        bool record_query = true);

    // Record the effective prefix actually restored by a higher-level codec.
    // Hybrid caches may need to back off from a structurally matched KV chain
    // to the newest block carrying an exact recurrent-state checkpoint.
    void record_match(std::size_t matched_tokens);

    BlockHash store(
        const BlockHash& parent,
        const std::int64_t* token_ids,
        std::size_t token_count,
        std::shared_ptr<const std::vector<std::uint8_t>> payload,
        std::string_view extra_key = {});

    // Atomically refresh the payload associated with an existing token-block
    // hash. This is used to add an exact recurrent-state checkpoint when a
    // previously stored KV block later becomes a stable prompt boundary.
    BlockHash replace(
        const BlockHash& parent,
        const std::int64_t* token_ids,
        std::size_t token_count,
        std::shared_ptr<const std::vector<std::uint8_t>> payload,
        std::string_view extra_key = {});

    std::optional<std::vector<std::uint8_t>> load(const BlockHash& hash);

    // Load the longest readable prefix of a matched block chain. Payloads are
    // shared with the RAM hot tier instead of copied; cold files are opened,
    // read, and verified once, with bounded parallelism. The result stops at
    // the first missing or corrupt block.
    std::vector<PagedPrefixPayload> load_prefix(
        const std::vector<BlockHash>& blocks);

    // Remove a block that passed file-integrity checks but failed the
    // runtime codec's semantic validation. This rare path waits for an
    // in-flight write of the same content-addressed block to settle first.
    bool invalidate(const BlockHash& block);

    void pin(const std::vector<BlockHash>& blocks);
    void unpin(const std::vector<BlockHash>& blocks);
    // Reclaim the RAM tier without deleting durable SSD blocks or breaking
    // live RAM-only bindings. Returns the number of payload bytes released.
    std::uint64_t trim_hot(std::uint64_t target_bytes = 0);
    void flush();
    std::size_t clear();
    PagedPrefixCacheMetrics metrics() const;

private:
    class Implementation;
    std::unique_ptr<Implementation> implementation_;
};

} // namespace mfq::cache
