#include "prefix_blocks.h"
#include "disk_budget.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

#if defined(__APPLE__)
#include <CommonCrypto/CommonDigest.h>
#endif

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace mfq::cache {
namespace {

constexpr std::array<std::uint8_t, 8> kMagic{
    'M', 'F', 'Q', 'K', 'V', 'B', '1', 0};
constexpr std::uint32_t kFormatVersion = 1;
constexpr std::uint64_t kHeaderBytes =
    8 + 4 + 32 + 32 + 32 + 4 + 4 + 8 + 32;
constexpr std::uint64_t kDiskSafeDenominator = 10;

template <typename T>
void write_little_endian(std::ostream& output, T value) {
    std::array<std::uint8_t, sizeof(T)> bytes{};
    using Unsigned = std::make_unsigned_t<T>;
    const auto converted = static_cast<Unsigned>(value);
    for (std::size_t index = 0; index < bytes.size(); ++index) {
        bytes[index] = static_cast<std::uint8_t>(converted >> (index * 8));
    }
    output.write(
        reinterpret_cast<const char*>(bytes.data()),
        static_cast<std::streamsize>(bytes.size()));
}

template <typename T>
bool read_little_endian(std::istream& input, T& value) {
    std::array<std::uint8_t, sizeof(T)> bytes{};
    input.read(
        reinterpret_cast<char*>(bytes.data()),
        static_cast<std::streamsize>(bytes.size()));
    if (!input) return false;
    using Unsigned = std::make_unsigned_t<T>;
    Unsigned converted = 0;
    for (std::size_t index = 0; index < bytes.size(); ++index) {
        converted |= static_cast<Unsigned>(bytes[index]) << (index * 8);
    }
    value = static_cast<T>(converted);
    return true;
}

bool zero_hash(const BlockHash& hash) {
    return std::all_of(hash.begin(), hash.end(), [](std::uint8_t value) {
        return value == 0;
    });
}

struct HashHasher {
    std::size_t operator()(const BlockHash& hash) const noexcept {
        std::size_t value = 0;
        std::memcpy(&value, hash.data(), std::min(sizeof(value), hash.size()));
        return value;
    }
};

std::uint64_t monotonic_tick() {
    return static_cast<std::uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
}

bool sync_regular_file(const std::filesystem::path& path) {
#if defined(_WIN32)
    const int descriptor = _wopen(
        path.c_str(), _O_RDWR | _O_BINARY);
    if (descriptor < 0) return false;
    const bool synced = _commit(descriptor) == 0;
    (void)_close(descriptor);
    return synced;
#else
    const int descriptor = ::open(path.c_str(), O_RDONLY);
    if (descriptor < 0) return false;
#if defined(__APPLE__) && defined(F_FULLFSYNC)
    int result = ::fcntl(descriptor, F_FULLFSYNC);
    if (result != 0) result = ::fsync(descriptor);
#else
    const int result = ::fsync(descriptor);
#endif
    (void)::close(descriptor);
    return result == 0;
#endif
}

void sync_directory_best_effort(const std::filesystem::path& path) noexcept {
#if !defined(_WIN32)
    const int descriptor = ::open(path.c_str(), O_RDONLY);
    if (descriptor < 0) return;
    (void)::fsync(descriptor);
    (void)::close(descriptor);
#else
    (void)path;
#endif
}

} // namespace

class PrefixBlockStore::Implementation {
private:
    struct DiskEntry {
        std::filesystem::path path;
        BlockHash parent{};
        std::uint32_t token_count = 0;
        std::uint64_t payload_bytes = 0;
        std::uint64_t file_bytes = 0;
        std::uint64_t last_used = 0;
        std::size_t pins = 0;
    };

    struct HotEntry {
        std::shared_ptr<const std::vector<std::uint8_t>> payload;
        std::uint64_t last_used = 0;
    };

    struct WriteRequest {
        BlockHash hash{};
        BlockHash parent{};
        std::uint32_t token_count = 0;
        std::shared_ptr<const std::vector<std::uint8_t>> payload;
        std::uint64_t epoch = 0;
        bool replace_existing = false;
        std::vector<std::int64_t> tokens;
    };

    struct ParsedHeader {
        BlockHash compatibility{};
        BlockHash hash{};
        BlockHash parent{};
        std::uint32_t block_size = 0;
        std::uint32_t token_count = 0;
        std::uint64_t payload_bytes = 0;
        BlockHash payload_hash{};
    };

    enum class LoadSource {
        Hot,
        Pending,
        Disk,
    };

    struct LoadRequest {
        BlockHash hash{};
        LoadSource source = LoadSource::Disk;
        PagedPrefixPayload payload;
        std::filesystem::path path;
    };

public:
    explicit Implementation(PrefixBlockStoreConfig config)
        : config_(std::move(config)),
          compatibility_hash_(sha256(config_.compatibility_key)) {
        if (config_.cache_dir.empty()) {
            throw std::invalid_argument("paged prefix cache directory is empty");
        }
        if (config_.compatibility_key.empty()) {
            throw std::invalid_argument("paged prefix compatibility key is empty");
        }
        if (config_.block_size_tokens == 0 ||
            config_.block_size_tokens >
                static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
            throw std::invalid_argument("invalid paged prefix block size");
        }
        if (config_.max_parallel_reads == 0) {
            config_.max_parallel_reads = 1;
        }
        namespace_dir_ = config_.cache_dir /
            block_hash_hex(compatibility_hash_);
        std::error_code error;
        std::filesystem::create_directories(namespace_dir_, error);
        if (error) {
            throw std::runtime_error(
                "cannot create paged prefix cache directory: " +
                error.message());
        }
        std::filesystem::permissions(
            namespace_dir_,
            std::filesystem::perms::owner_all,
            std::filesystem::perm_options::replace,
            error);
        scan();
        if (config_.max_disk_bytes > 0) write_identity();
        worker_ = std::thread([this] { writer_loop(); });
    }

    ~Implementation() {
        if (config_.spill_hot_on_eviction) trim_hot(0);
        flush();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        work_available_.notify_all();
        if (worker_.joinable()) worker_.join();
    }

    std::size_t block_size_tokens() const noexcept {
        return config_.block_size_tokens;
    }

    const BlockHash& compatibility_hash() const noexcept {
        return compatibility_hash_;
    }







    bool contains(const BlockHash& hash,const BlockHash& parent,std::size_t count) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto disk=disk_.find(hash);
        if(disk!=disk_.end() && (disk->second.parent!=parent || disk->second.token_count!=count))
            erase_corrupt_locked(hash);
        return disk_.count(hash)!=0 || pending_.count(hash)!=0 || hot_.count(hash)!=0;
    }
    std::vector<std::size_t> tail_lengths(const BlockHash& parent,std::size_t maximum) const {
        std::vector<std::size_t> result;
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found=tail_lengths_.find(parent);
        if(found!=tail_lengths_.end())
            for(auto it=found->second.rbegin();it!=found->second.rend();++it)
                if(it->first<=maximum)result.push_back(it->first);
        return result;
    }

    BlockHash store(
        const BlockHash& hash,
        const BlockHash& parent,
        const std::int64_t* token_ids,
        std::size_t token_count,
        std::shared_ptr<const std::vector<std::uint8_t>> payload,
        std::string_view extra_key,
        bool replace_existing) {
        if (!payload) {
            throw std::invalid_argument("paged prefix payload is null");
        }
        if (!token_ids || !token_count || token_count > config_.block_size_tokens)
            throw std::invalid_argument("invalid paged prefix token block");
        std::uint64_t write_epoch = 0;
        const auto payload_bytes = static_cast<std::uint64_t>(
            payload->size());
        {
            std::unique_lock<std::mutex> lock(mutex_);
            if (clearing_ || (replace_existing && pending_.count(hash))) {
                ++metrics_.skipped_writes;
                return hash;
            }
            if (!replace_existing &&
                (disk_.count(hash) != 0 || pending_.count(hash) != 0)) {
                ++metrics_.deduplicated_writes;
                return hash;
            }
            schedule_tail_cleanup_locked(hash, parent, token_ids, token_count, extra_key);
            if (config_.spill_hot_on_eviction && payload_bytes <= config_.max_hot_bytes && config_.max_hot_bytes) {
                if (!replace_existing && hot_.count(hash)) {
                    hot_.at(hash).last_used = ++clock_;
                    ++metrics_.deduplicated_writes;
                    return hash;
                }
                deferred_[hash] = WriteRequest{hash, parent, static_cast<std::uint32_t>(token_count),
                    payload, cache_epoch_, replace_existing,
                    std::vector<std::int64_t>(token_ids, token_ids + token_count)};
                put_hot_locked(hash, std::move(payload));
                remember_tail_locked(hash, parent, token_count);
                prune_tail_cleanup_locked();
                trim_for_ram_pressure_locked();
                const bool drain = pending_write_bytes_ > config_.max_pending_bytes ||
                    writes_.size() >= config_.max_pending_writes;
                lock.unlock();
                if (drain) flush();
                return hash;
            }
            deferred_.erase(hash);
            if (config_.max_disk_bytes == 0) {
                if (!replace_existing && hot_.count(hash) != 0) {
                    ++metrics_.deduplicated_writes;
                    return hash;
                }
                put_hot_locked(hash, std::move(payload));
                if (hot_.count(hash) != 0) remember_tail_locked(hash, parent, token_count);
                prune_tail_cleanup_locked();
                sync_metrics_locked();
                return hash;
            }
            const auto pending_files = static_cast<std::uint64_t>(
                pending_.size()) + 1;
            const auto pending_headers = pending_files >
                    std::numeric_limits<std::uint64_t>::max() / kHeaderBytes
                ? std::numeric_limits<std::uint64_t>::max()
                : pending_files * kHeaderBytes;
            const auto pending_payloads = pending_write_bytes_ >
                    std::numeric_limits<std::uint64_t>::max() - payload_bytes
                ? std::numeric_limits<std::uint64_t>::max()
                : pending_write_bytes_ + payload_bytes;
            const auto reserved_bytes = pending_payloads >
                    std::numeric_limits<std::uint64_t>::max() - pending_headers
                ? std::numeric_limits<std::uint64_t>::max()
                : pending_payloads + pending_headers;
            put_hot_locked(hash, payload);
            write_epoch = cache_epoch_;
            const bool pending_bytes_full =
                config_.max_pending_bytes == 0 ||
                payload_bytes > config_.max_pending_bytes ||
                pending_write_bytes_ >
                    config_.max_pending_bytes - payload_bytes;
            if (writes_.size() >= config_.max_pending_writes ||
                pending_bytes_full) {
                ++metrics_.skipped_writes;
                sync_metrics_locked();
                return hash;
            } else {
                enforce_disk_budget_locked(reserved_bytes);
                writes_.push_back(WriteRequest{
                    hash,
                    parent,
                    static_cast<std::uint32_t>(token_count),
                    std::move(payload),
                    write_epoch,
                    replace_existing,
                    std::vector<std::int64_t>(token_ids, token_ids + token_count),
                });
                pending_[hash] = writes_.back().payload;
            }
            pending_write_bytes_ += payload_bytes;
            remember_tail_locked(hash, parent, token_count);
            sync_metrics_locked();
        }
        work_available_.notify_one();
        return hash;
    }

    std::optional<std::vector<std::uint8_t>> load(const BlockHash& hash) {
        const auto payloads = load_prefix({hash});
        if (payloads.empty()) return std::nullopt;
        return *payloads.front();
    }

    std::vector<PagedPrefixPayload> load_prefix(
        const std::vector<BlockHash>& hashes) {
        std::vector<LoadRequest> requests;
        requests.reserve(hashes.size());
        std::uint64_t load_epoch = 0;
        std::size_t payload_bytes = 0;
        try {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                load_epoch = cache_epoch_;
                for (const auto& hash : hashes) {
                    const auto hot = hot_.find(hash);
                    if (hot != hot_.end()) {
                        payload_bytes += hot->second.payload->size();
                        requests.push_back(LoadRequest{
                            hash,
                            LoadSource::Hot,
                            hot->second.payload,
                            {},
                        });
                        continue;
                    }
                    const auto queued = pending_.find(hash);
                    if (queued != pending_.end()) {
                        payload_bytes += queued->second->size();
                        requests.push_back(LoadRequest{
                            hash,
                            LoadSource::Pending,
                            queued->second,
                            {},
                        });
                        continue;
                    }
                    const auto found = disk_.find(hash);
                    if (found == disk_.end()) break;
                    payload_bytes += found->second.payload_bytes;
                    requests.push_back(LoadRequest{
                        hash,
                        LoadSource::Disk,
                        {},
                        found->second.path,
                    });
                    // Keep the durable copy alive after releasing mutex_: an
                    // overlapping write can enforce the SSD budget while
                    // these payloads are being read.
                    ++found->second.pins;
                }
            }

            if (config_.reserve_memory) config_.reserve_memory(payload_bytes);

            std::vector<std::optional<std::vector<std::uint8_t>>> cold_payloads(
                requests.size());
            std::vector<std::size_t> cold_indices;
            cold_indices.reserve(requests.size());
            for (std::size_t index = 0; index < requests.size(); ++index) {
                if (requests[index].source == LoadSource::Disk) {
                    cold_indices.push_back(index);
                }
            }

            const auto read_cold = [&](std::size_t cold_index) {
                const auto request_index = cold_indices[cold_index];
                const auto& request = requests[request_index];
                cold_payloads[request_index] = read_payload(
                    request.path, request.hash);
            };
            if (cold_indices.size() == 1 || config_.max_parallel_reads == 1) {
                for (std::size_t index = 0; index < cold_indices.size();
                     ++index) {
                    read_cold(index);
                }
            } else if (!cold_indices.empty()) {
                const auto worker_count = std::min(
                    cold_indices.size(), config_.max_parallel_reads);
                std::atomic<std::size_t> next{0};
                std::mutex error_mutex;
                std::exception_ptr first_error;
                std::vector<std::thread> readers;
                readers.reserve(worker_count);
                for (std::size_t worker = 0; worker < worker_count; ++worker) {
                    readers.emplace_back([&] {
                        while (true) {
                            const auto index = next.fetch_add(1);
                            if (index >= cold_indices.size()) break;
                            try {
                                read_cold(index);
                            } catch (...) {
                                std::lock_guard<std::mutex> lock(error_mutex);
                                if (!first_error) {
                                    first_error = std::current_exception();
                                }
                            }
                        }
                    });
                }
                for (auto& reader : readers) reader.join();
                if (first_error) std::rethrow_exception(first_error);
            }

            std::vector<PagedPrefixPayload> result;
            result.reserve(requests.size());
            std::lock_guard<std::mutex> lock(mutex_);
            if (load_epoch != cache_epoch_) {
                // clear() removed every entry carrying this load's leases.
                // A same-hash entry recreated in the new epoch must not be
                // charged for an older read.
                return result;
            }
            for (std::size_t index = 0; index < requests.size(); ++index) {
                auto& request = requests[index];
                if (request.source == LoadSource::Disk) {
                    if (!cold_payloads[index]) {
                        std::error_code error;
                        if (!std::filesystem::exists(request.path, error) && !error) {
                            const auto stale = disk_.find(request.hash);
                            if (stale != disk_.end()) {
                                disk_bytes_ -= stale->second.file_bytes;
                                disk_.erase(stale);
                                forget_tail_if_unused_locked(request.hash);
                                ++metrics_.evictions;
                            }
                        } else erase_corrupt_locked(request.hash);
                        break;
                    }
                    request.payload =
                        std::make_shared<const std::vector<std::uint8_t>>(
                            std::move(*cold_payloads[index]));
                    put_hot_locked(request.hash, request.payload);
                    const auto found = disk_.find(request.hash);
                    if (found != disk_.end()) {
                        found->second.last_used = ++clock_;
                    }
                    ++metrics_.disk_hits;
                } else {
                    if (request.source == LoadSource::Hot) {
                        const auto hot = hot_.find(request.hash);
                        if (hot != hot_.end()) {
                            hot->second.last_used = ++clock_;
                        }
                    }
                    ++metrics_.hot_hits;
                }
                const auto durable = disk_.find(request.hash);
                if (durable != disk_.end()) {
                    std::error_code error;
                    std::filesystem::last_write_time(durable->second.path,
                        std::filesystem::file_time_type::clock::now(), error);
                }
                result.push_back(std::move(request.payload));
            }
            release_disk_read_pins_locked(requests);
            return result;
        } catch (...) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (load_epoch == cache_epoch_) {
                release_disk_read_pins_locked(requests);
            }
            throw;
        }
    }

    bool invalidate(const BlockHash& hash) {
        std::unique_lock<std::mutex> lock(mutex_);
        writes_finished_.wait(lock, [this, &hash] {
            return pending_.count(hash) == 0;
        });
        return erase_corrupt_locked(hash);
    }

    void pin(const std::vector<BlockHash>& blocks) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& block : blocks) {
            ++pins_[block];
            auto found = disk_.find(block);
            if (found != disk_.end()) ++found->second.pins;
        }
    }

    void unpin(const std::vector<BlockHash>& blocks) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& block : blocks) {
            auto pinned = pins_.find(block);
            if (pinned != pins_.end()) {
                if (--pinned->second == 0) pins_.erase(pinned);
            }
            auto found = disk_.find(block);
            if (found != disk_.end() && found->second.pins > 0) {
                --found->second.pins;
            }
        }
        enforce_disk_budget_locked();
    }

    std::uint64_t reserve_ram(std::uint64_t allocation_bytes) {
        if (!config_.available_ram_bytes) return 0;
        const auto available = config_.available_ram_bytes();
        const auto reserve = config_.min_free_ram_bytes;
        const auto free = available > reserve ? available - reserve : 0;
        if (free >= allocation_bytes) return 0;
        const auto shortage = allocation_bytes - free;
        std::uint64_t target;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            target = hot_bytes_ > shortage ? hot_bytes_ - shortage : 0;
        }
        return trim_hot(target);
    }

    std::uint64_t trim_hot(std::uint64_t target_bytes) {
        std::unique_lock<std::mutex> lock(mutex_);
        const auto released = trim_hot_locked(target_bytes);
        sync_metrics_locked();
        lock.unlock();
        if (config_.spill_hot_on_eviction) flush();
        return released;
    }

    std::uint64_t set_hot_limit(std::uint64_t max_bytes) {
        std::unique_lock<std::mutex> lock(mutex_);
        config_.max_hot_bytes = max_bytes;
        metrics_.hot_pressure_bytes = 0;
        const auto released = trim_hot_locked(max_bytes);
        sync_metrics_locked();
        lock.unlock();
        if (config_.spill_hot_on_eviction) flush();
        return released;
    }

    void flush() {
        std::unique_lock<std::mutex> lock(mutex_);
        writes_finished_.wait(lock, [this] {
            return writes_.empty() && active_writes_ == 0 && !disk_budget_pending_;
        });
    }

    std::uint64_t set_disk_limit(std::uint64_t max_bytes) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (config_.max_disk_bytes == max_bytes) return 0;
        disk_identity_pending_ |= config_.max_disk_bytes == 0 && max_bytes > 0;
        config_.max_disk_bytes = max_bytes;
        disk_budget_pending_ = true;
        work_available_.notify_one();
        return 0;
    }

    std::size_t refresh_disk_index() {
        std::unique_lock<std::mutex> lock(mutex_);
        writes_finished_.wait(lock, [this] { return writes_.empty() && active_writes_ == 0; });
        std::size_t removed = 0;
        for (auto entry = disk_.begin(); entry != disk_.end();) {
            std::error_code error;
            if (std::filesystem::exists(entry->second.path, error) || error) { ++entry; continue; }
            const auto hash = entry->first;
            disk_bytes_ -= entry->second.file_bytes;
            entry = disk_.erase(entry);
            const auto hot = hot_.find(hash);
            if (hot != hot_.end()) {
                hot_bytes_ -= hot->second.payload->size();
                hot_.erase(hot);
            }
            deferred_.erase(hash);
            forget_tail_if_unused_locked(hash);
            ++removed;
        }
        sync_metrics_locked();
        return removed;
    }

    std::size_t clear() {
        std::unique_lock<std::mutex> lock(mutex_);
        writes_finished_.wait(lock, [this] { return !clearing_; });
        clearing_ = true;
        ++cache_epoch_;
        writes_.clear();
        pending_.clear();
        pending_write_bytes_ = 0;
        std::size_t removed = 0;
        for (const auto& [hash, entry] : disk_) {
            (void)hash;
            std::error_code error;
            if (std::filesystem::remove(entry.path, error) && !error) ++removed;
            remove_text(entry.path);
        }
        disk_.clear();
        hot_.clear();
        deferred_.clear();
        tail_cleanup_.clear();
        pending_.clear();
        pins_.clear();
        tail_lengths_.clear();
        tail_metadata_.clear();
        disk_bytes_ = 0;
        hot_bytes_ = 0;
        sync_metrics_locked();
        writes_finished_.wait(lock, [this] {
            return active_writes_ == 0;
        });
        clearing_ = false;
        lock.unlock();
        writes_finished_.notify_all();
        return removed;
    }

    PrefixBlockStoreMetrics metrics() const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto result = metrics_;
        result.disk_blocks = disk_.size();
        result.disk_bytes = disk_bytes_;
        result.hot_blocks = hot_.size();
        result.hot_bytes = hot_bytes_;
        result.blocks = disk_.size();
        for (const auto& [hash, entry] : hot_) if (!disk_.count(hash)) ++result.blocks;
        for (const auto& [hash, entry] : pending_) if (!disk_.count(hash) && !hot_.count(hash)) ++result.blocks;
        result.pending_writes = pending_.size();
        result.pending_bytes = pending_write_bytes_;
        result.pending_max_bytes = config_.max_pending_bytes;
        result.resident_bytes = hot_bytes_;
        for (const auto& [hash, entry] : hot_) result.resident_bytes += entry.payload->capacity() - entry.payload->size();
        for (const auto& [hash, payload] : pending_) {
            const auto hot = hot_.find(hash);
            if (hot == hot_.end() || hot->second.payload != payload) result.resident_bytes += payload->capacity();
        }
        return result;
    }

private:
    std::filesystem::path path_for(const BlockHash& hash) const {
        const auto text = block_hash_hex(hash);
        return namespace_dir_ / text.substr(0, 2) / (text + ".mfqkv");
    }

    static void remove_text(const std::filesystem::path& block) {
        std::error_code error;
        std::filesystem::remove(block.string() + ".tokens", error);
    }

    void write_text(const WriteRequest& request) {
        DiskWriteGuard space_guard(namespace_dir_, 108 + request.tokens.size() * 8, config_.min_disk_free_bytes);
        if (!space_guard.allowed()) return;
        const auto target = path_for(request.hash).string() + ".tokens";
        const auto temporary = target + ".tmp." + std::to_string(monotonic_tick());
        std::vector<std::uint8_t> bytes;
        bytes.reserve(request.tokens.size() * 8);
        for (const auto token : request.tokens) for (std::size_t i = 0; i < 8; ++i)
            bytes.push_back(static_cast<std::uint8_t>(static_cast<std::uint64_t>(token) >> (i * 8)));
        const auto checksum = sha256(bytes.data(), bytes.size());
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) return;
        output.write("MFQTXT1\0", 8);
        output.write(reinterpret_cast<const char*>(compatibility_hash_.data()), 32);
        output.write(reinterpret_cast<const char*>(request.hash.data()), 32);
        write_little_endian(output, static_cast<std::uint32_t>(request.tokens.size()));
        output.write(reinterpret_cast<const char*>(checksum.data()), 32);
        output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        output.close();
        std::error_code error;
        if (output) {
            std::filesystem::permissions(temporary, std::filesystem::perms::owner_read |
                std::filesystem::perms::owner_write, std::filesystem::perm_options::replace, error);
            if (!error) std::filesystem::rename(temporary, target, error);
        }
        std::filesystem::remove(temporary, error);
    }

    bool read_header(
        std::istream& input,
        ParsedHeader& header) const {
        std::array<std::uint8_t, 8> magic{};
        input.read(
            reinterpret_cast<char*>(magic.data()),
            static_cast<std::streamsize>(magic.size()));
        std::uint32_t version = 0;
        if (!input || magic != kMagic ||
            !read_little_endian(input, version) ||
            version != kFormatVersion) {
            return false;
        }
        input.read(
            reinterpret_cast<char*>(header.compatibility.data()), 32);
        input.read(reinterpret_cast<char*>(header.hash.data()), 32);
        input.read(reinterpret_cast<char*>(header.parent.data()), 32);
        if (!input || !read_little_endian(input, header.block_size) ||
            !read_little_endian(input, header.token_count) ||
            !read_little_endian(input, header.payload_bytes)) {
            return false;
        }
        input.read(reinterpret_cast<char*>(header.payload_hash.data()), 32);
        return input && header.compatibility == compatibility_hash_ &&
            header.block_size == config_.block_size_tokens &&
            header.token_count > 0 &&
            header.token_count <= header.block_size &&
            header.payload_bytes <=
                static_cast<std::uint64_t>(
                    std::numeric_limits<std::streamsize>::max());
    }

    bool read_header(
        const std::filesystem::path& path,
        ParsedHeader& header) const {
        std::ifstream input(path, std::ios::binary);
        return input && read_header(input, header);
    }

    std::optional<std::vector<std::uint8_t>> read_payload(
        const std::filesystem::path& path,
        const BlockHash& expected_hash) const {
        std::ifstream input(path, std::ios::binary);
        if (!input) return std::nullopt;
        ParsedHeader header;
        if (!read_header(input, header) || header.hash != expected_hash) {
            return std::nullopt;
        }
        const auto payload_start = input.tellg();
        input.seekg(0, std::ios::end);
        const auto file_end = input.tellg();
        if (payload_start != static_cast<std::streamoff>(kHeaderBytes) ||
            file_end < 0 ||
            header.payload_bytes >
                std::numeric_limits<std::uint64_t>::max() - kHeaderBytes ||
            static_cast<std::uint64_t>(file_end) !=
                kHeaderBytes + header.payload_bytes) {
            return std::nullopt;
        }
        input.seekg(payload_start);
        if (!input) return std::nullopt;
        std::vector<std::uint8_t> payload(
            static_cast<std::size_t>(header.payload_bytes));
        input.read(
            reinterpret_cast<char*>(payload.data()),
            static_cast<std::streamsize>(payload.size()));
        if (!input || sha256(payload.data(), payload.size()) !=
                header.payload_hash) {
            return std::nullopt;
        }
        return payload;
    }

    bool write_file(const WriteRequest& request) {
        const auto final_path = path_for(request.hash);
        std::error_code error;
        std::filesystem::create_directories(final_path.parent_path(), error);
        if (error) return false;
        const auto space = std::filesystem::space(final_path.parent_path(), error);
        const auto bytes = static_cast<std::uint64_t>(request.payload->size());
        const auto overhead = kHeaderBytes + 108 + request.tokens.size() * 8;
        if (error || space.available < config_.min_disk_free_bytes ||
            space.available - config_.min_disk_free_bytes < overhead ||
            bytes > space.available - config_.min_disk_free_bytes - overhead) {
            std::lock_guard<std::mutex> lock(mutex_);
            ++metrics_.low_disk_space_skips;
            return false;
        }
        const auto temporary = final_path.parent_path() /
            (".mfqkv.tmp." + std::to_string(monotonic_tick()) + "." +
             std::to_string(temp_counter_.fetch_add(1)));
        DiskWriteGuard space_guard(final_path.parent_path(), bytes + overhead, config_.min_disk_free_bytes);
        if (!space_guard.allowed()) return false;
        std::ofstream output(
            temporary,
            std::ios::binary | std::ios::trunc);
        if (!output) return false;
        const auto payload_hash = sha256(
            request.payload->data(), request.payload->size());
        output.write(
            reinterpret_cast<const char*>(kMagic.data()),
            static_cast<std::streamsize>(kMagic.size()));
        write_little_endian(output, kFormatVersion);
        output.write(
            reinterpret_cast<const char*>(compatibility_hash_.data()), 32);
        output.write(reinterpret_cast<const char*>(request.hash.data()), 32);
        output.write(reinterpret_cast<const char*>(request.parent.data()), 32);
        write_little_endian(
            output, static_cast<std::uint32_t>(config_.block_size_tokens));
        write_little_endian(output, request.token_count);
        write_little_endian(
            output, static_cast<std::uint64_t>(request.payload->size()));
        output.write(reinterpret_cast<const char*>(payload_hash.data()), 32);
        output.write(
            reinterpret_cast<const char*>(request.payload->data()),
            static_cast<std::streamsize>(request.payload->size()));
        output.flush();
        output.close();
        if (!output) {
            std::filesystem::remove(temporary, error);
            return false;
        }
        if (!sync_regular_file(temporary)) {
            std::filesystem::remove(temporary, error);
            return false;
        }
        std::filesystem::permissions(
            temporary,
            std::filesystem::perms::owner_read |
                std::filesystem::perms::owner_write,
            std::filesystem::perm_options::replace,
            error);
        error.clear();
        if (!request.replace_existing) {
            // Publish immutable blocks with create-if-absent semantics. A
            // second MFQ process may already have upgraded this same token
            // block with a recurrent checkpoint, which a plain POSIX rename
            // would silently overwrite with the smaller base payload.
            std::filesystem::create_hard_link(
                temporary, final_path, error);
            if (!error) {
                std::filesystem::remove(temporary, error);
                sync_directory_best_effort(final_path.parent_path());
                return true;
            }
            if (std::filesystem::exists(final_path)) {
                error.clear();
                std::filesystem::remove(temporary, error);
                return true;
            }
            std::filesystem::remove(temporary, error);
            return false;
        }
        std::filesystem::rename(temporary, final_path, error);
        if (error) {
            if (std::filesystem::exists(final_path)) {
                error.clear();
                std::filesystem::remove(final_path, error);
                if (!error) {
                    std::filesystem::rename(temporary, final_path, error);
                }
                if (!error) {
                    sync_directory_best_effort(final_path.parent_path());
                    return true;
                }
            }
            std::filesystem::remove(temporary, error);
            return false;
        }
        sync_directory_best_effort(final_path.parent_path());
        return true;
    }

    void write_identity() {
        DiskWriteGuard space_guard(namespace_dir_, 32768, config_.min_disk_free_bytes);
        if (!space_guard.allowed()) return;
        const auto target = namespace_dir_ / "identity.txt";
        const auto temporary = namespace_dir_ / (".identity.tmp." + std::to_string(monotonic_tick()));
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) return;
        const auto end = config_.compatibility_key.find("\ntensor=");
        output << "mfq-cache-identity-v1\nnamespace=" << block_hash_hex(compatibility_hash_) << '\n'
            << config_.compatibility_key.substr(0, std::min<std::size_t>(end, 30000)) << '\n';
        const auto model = config_.model_path.string();
        if (model.find_first_of("\r\n") == std::string::npos) output << "model_path=" << model << '\n';
        output.close();
        std::error_code error;
        if (output) {
            std::filesystem::permissions(temporary, std::filesystem::perms::owner_read |
                std::filesystem::perms::owner_write, std::filesystem::perm_options::replace, error);
            if (!error) std::filesystem::rename(temporary, target, error);
        }
        std::filesystem::remove(temporary, error);
    }

    void finish_write(const WriteRequest& request, bool success) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto pending = pending_.find(request.hash);
        if (pending != pending_.end()) {
            const auto bytes = static_cast<std::uint64_t>(
                pending->second->size());
            pending_write_bytes_ = bytes > pending_write_bytes_
                ? 0
                : pending_write_bytes_ - bytes;
            pending_.erase(pending);
        }
        if (success && request.epoch == cache_epoch_) {
            try { write_text(request); } catch (...) {}
            const auto path = path_for(request.hash);
            std::error_code error;
            const auto file_bytes = std::filesystem::file_size(path, error);
            if (!error) {
                auto previous = disk_.find(request.hash);
                std::size_t disk_pins = pins_.count(request.hash) == 0
                    ? 0
                    : pins_.at(request.hash);
                if (previous != disk_.end()) {
                    disk_pins = std::max(disk_pins, previous->second.pins);
                    disk_bytes_ -= previous->second.file_bytes;
                }
                disk_[request.hash] = DiskEntry{
                    path,
                    request.parent,
                    request.token_count,
                    request.payload->size(),
                    file_bytes,
                    ++clock_,
                    disk_pins,
                };
                disk_bytes_ += file_bytes;
                ++metrics_.writes;
                enforce_disk_budget_locked();
            } else {
                ++metrics_.failed_writes;
            }
        } else if (!success) {
            ++metrics_.failed_writes;
        } else {
            std::error_code error;
            const auto path = path_for(request.hash);
            std::filesystem::remove(path, error);
            remove_text(path);
            sync_directory_best_effort(path.parent_path());
        }
        forget_tail_if_unused_locked(request.hash);
        prune_tail_cleanup_locked();
        sync_metrics_locked();
        writes_finished_.notify_all();
    }

    void writer_loop() {
        while (true) {
            WriteRequest request;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                work_available_.wait_for(lock, std::chrono::milliseconds(250), [this] {
                    return stopping_ || !writes_.empty() || disk_budget_pending_;
                });
                if (!clearing_) trim_for_ram_pressure_locked();
                if (disk_budget_pending_) {
                    if (disk_identity_pending_) {
                        write_identity();
                        disk_identity_pending_ = false;
                    }
                    disk_budget_pending_ = enforce_disk_budget_locked(0, 1);
                    writes_finished_.notify_all();
                    continue;
                }
                if (stopping_ && writes_.empty()) return;
                if (writes_.empty()) continue;
                request = std::move(writes_.front());
                writes_.pop_front();
                ++active_writes_;
            }
            bool success = false;
            try {
                success = write_file(request);
            } catch (...) {
                success = false;
            }
            finish_write(request, success);
            {
                std::lock_guard<std::mutex> lock(mutex_);
                --active_writes_;
                writes_finished_.notify_all();
            }
        }
    }

    void scan() {
        struct ScannedEntry {
            BlockHash hash{};
            DiskEntry entry;
            std::filesystem::file_time_type modified{};
        };

        std::vector<ScannedEntry> scanned;
        std::error_code error;
        for (std::filesystem::recursive_directory_iterator iterator(
                 namespace_dir_,
                 std::filesystem::directory_options::skip_permission_denied,
                 error), end;
             iterator != end && !error;
             iterator.increment(error)) {
            if (!iterator->is_regular_file(error)) {
                error.clear();
                continue;
            }
            const auto filename = iterator->path().filename().string();
            if (filename.find(".tmp.") != std::string::npos) {
                std::filesystem::remove(iterator->path(), error);
                error.clear();
                continue;
            }
            if (
                iterator->path().extension() != ".mfqkv") {
                continue;
            }
            const auto reject = [&] {
                std::error_code remove_error;
                std::filesystem::remove(iterator->path(), remove_error);
                remove_text(iterator->path());
                ++metrics_.corrupt_blocks;
            };
            std::error_code symlink_error;
            if (iterator->is_symlink(symlink_error)) {
                reject();
                continue;
            }
            if (symlink_error) continue;
            ParsedHeader header;
            if (!read_header(iterator->path(), header) ||
                iterator->path().lexically_normal() !=
                    path_for(header.hash).lexically_normal()) {
                reject();
                continue;
            }
            const auto file_bytes = iterator->file_size(error);
            if (error) {
                error.clear();
                continue;
            }
            if (header.payload_bytes >
                    std::numeric_limits<std::uint64_t>::max() - kHeaderBytes ||
                file_bytes != kHeaderBytes + header.payload_bytes) {
                reject();
                continue;
            }
            const auto modified = iterator->last_write_time(error);
            const auto ordering_time = error
                ? std::filesystem::file_time_type::min()
                : modified;
            error.clear();
            scanned.push_back(ScannedEntry{
                header.hash,
                DiskEntry{
                    iterator->path(),
                    header.parent,
                    header.token_count,
                    header.payload_bytes,
                    file_bytes,
                    0,
                    0,
                },
                ordering_time,
            });
        }
        std::sort(
            scanned.begin(),
            scanned.end(),
            [](const ScannedEntry& left, const ScannedEntry& right) {
                if (left.modified != right.modified) {
                    return left.modified < right.modified;
                }
                return left.entry.path.native() < right.entry.path.native();
        });
        for (auto& item : scanned) {
            item.entry.last_used = ++clock_;
            const auto file_bytes = item.entry.file_bytes;
            const auto insertion = disk_.emplace(item.hash, std::move(item.entry));
            if (insertion.second) {
                disk_bytes_ += file_bytes;
                remember_tail_locked(item.hash, insertion.first->second.parent,
                    insertion.first->second.token_count);
            }
        }
        enforce_disk_budget_locked();
        sync_metrics_locked();
    }

    void put_hot_locked(
        const BlockHash& hash,
        std::shared_ptr<const std::vector<std::uint8_t>> payload) {
        auto previous = hot_.find(hash);
        if (previous != hot_.end()) {
            hot_bytes_ -= previous->second.payload->size();
            hot_.erase(previous);
        }
        const auto needed = hot_bytes_ + payload->size();
        if (needed > config_.max_hot_bytes)
            metrics_.hot_pressure_bytes = std::max(metrics_.hot_pressure_bytes, needed);
        if (config_.max_hot_bytes == 0 ||
            payload->size() > config_.max_hot_bytes) {
            sync_metrics_locked();
            return;
        }
        hot_[hash] = HotEntry{std::move(payload), ++clock_};
        hot_bytes_ += hot_[hash].payload->size();
        trim_hot_locked(config_.max_hot_bytes);
        sync_metrics_locked();
    }

    std::uint64_t trim_hot_locked(std::uint64_t target_bytes) {
        const auto before = hot_bytes_;
        while (hot_bytes_ > target_bytes && !hot_.empty()) {
            auto victim = hot_.end();
            for (auto iterator = hot_.begin(); iterator != hot_.end();
                 ++iterator) {
                // A block with no durable copy must remain available while a
                // live session pins it. Durable pinned blocks may still leave
                // the hot tier and be restored from SSD on demand.
                if (!config_.spill_hot_on_eviction && disk_.count(iterator->first) == 0 &&
                    pins_.count(iterator->first) != 0) {
                    continue;
                }
                if (victim == hot_.end() ||
                    iterator->second.last_used < victim->second.last_used) {
                    victim = iterator;
                }
            }
            if (victim == hot_.end()) break;
            hot_bytes_ -= victim->second.payload->size();
            const auto hash = victim->first;
            const auto dirty = deferred_.find(hash);
            if (dirty != deferred_.end()) {
                if (config_.max_disk_bytes > 0) {
                    pending_[hash] = dirty->second.payload;
                    pending_write_bytes_ += dirty->second.payload->size();
                    writes_.push_back(std::move(dirty->second));
                    work_available_.notify_one();
                }
                deferred_.erase(dirty);
            }
            hot_.erase(victim);
            forget_tail_if_unused_locked(hash);
        }
        return before - hot_bytes_;
    }


    void trim_for_ram_pressure_locked() {
        if (!config_.available_ram_bytes || hot_bytes_ == 0) return;
        const auto available = config_.available_ram_bytes();
        if (available >= config_.min_free_ram_bytes) return;
        const auto shortage = config_.min_free_ram_bytes - available;
        trim_hot_locked(hot_bytes_ > shortage ? hot_bytes_ - shortage : 0);
        sync_metrics_locked();
    }

    bool erase_corrupt_locked(const BlockHash& hash) {
        deferred_.erase(hash);
        bool removed = false;
        auto found = disk_.find(hash);
        if (found != disk_.end()) {
            std::error_code error;
            std::filesystem::remove(found->second.path, error);
            remove_text(found->second.path);
            disk_bytes_ -= found->second.file_bytes;
            disk_.erase(found);
            removed = true;
        }
        auto hot = hot_.find(hash);
        if (hot != hot_.end()) {
            hot_bytes_ -= hot->second.payload->size();
            hot_.erase(hot);
            removed = true;
        }
        if (removed) {
            forget_tail_if_unused_locked(hash);
            ++metrics_.corrupt_blocks;
            sync_metrics_locked();
        }
        return removed;
    }

    void release_disk_read_pins_locked(
        const std::vector<LoadRequest>& requests) {
        for (const auto& request : requests) {
            if (request.source != LoadSource::Disk) continue;
            auto found = disk_.find(request.hash);
            if (found != disk_.end() && found->second.pins > 0) {
                --found->second.pins;
            }
        }
        // A writer may have deferred LRU eviction while one of these files
        // was leased by the reader.
        enforce_disk_budget_locked();
    }

    std::uint64_t effective_disk_budget_locked() const {
        std::error_code error;
        const auto space = std::filesystem::space(namespace_dir_, error);
        if (error) return config_.max_disk_bytes;
        std::uint64_t managed_and_available = 0;
        try { managed_and_available = unreserved_disk_space(namespace_dir_, disk_bytes_); } catch (...) { return 0; }
        const auto safe_available = managed_and_available -
            managed_and_available / kDiskSafeDenominator;
        return std::min(config_.max_disk_bytes, safe_available);
    }

    bool enforce_disk_budget_locked(std::uint64_t reserved_bytes = 0,
        std::size_t max_evictions = std::numeric_limits<std::size_t>::max()) {
        const auto budget = effective_disk_budget_locked();
        const auto target = reserved_bytes >= budget ? 0 : budget - reserved_bytes;
        std::size_t evicted = 0;
        while (disk_bytes_ > target && !disk_.empty()) {
            auto victim = disk_.end();
            for (auto iterator = disk_.begin(); iterator != disk_.end();
                 ++iterator) {
                const auto binding = pins_.find(iterator->first);
                const auto session_pins = binding == pins_.end() ? 0 : binding->second;
                // Disposable session bindings must not hold SSD above quota,
                // but active file readers still need their pin until read ends.
                if (iterator->second.pins > (config_.spill_hot_on_eviction ? session_pins : 0) ||
                    pending_.count(iterator->first) != 0) {
                    continue;
                }
                if (victim == disk_.end() ||
                    iterator->second.last_used < victim->second.last_used) {
                    victim = iterator;
                }
            }
            if (victim == disk_.end()) break;
            std::error_code error;
            std::filesystem::remove(victim->second.path, error);
            if (error) break;
            remove_text(victim->second.path);
            disk_bytes_ -= victim->second.file_bytes;
            // Disk and RAM are independent tiers. A hot payload remains a
            // valid content-addressed hit after its durable copy is evicted.
            const auto hash = victim->first;
            if (config_.spill_hot_on_eviction && hot_.count(hash) && !deferred_.count(hash))
                deferred_[hash] = WriteRequest{hash, victim->second.parent, victim->second.token_count,
                    hot_.at(hash).payload, cache_epoch_, false, {}};
            disk_.erase(victim);
            forget_tail_if_unused_locked(hash);
            ++metrics_.evictions;
            if (++evicted >= max_evictions) break;
        }
        sync_metrics_locked();
        return disk_bytes_ > target && evicted == max_evictions;
    }

    void schedule_tail_cleanup_locked(const BlockHash& hash, const BlockHash& parent,
        const std::int64_t* tokens, std::size_t count, std::string_view extra_key) {
        if (config_.retain_tail_versions == 0 || count > config_.block_size_tokens) return;
        const auto tails = tail_lengths_.find(parent);
        if (tails == tail_lengths_.end()) return;
        std::size_t retained = 1;
        auto& victims = tail_cleanup_[hash];
        for (auto length = tails->second.rbegin(); length != tails->second.rend(); ++length) {
            if (length->first >= count) continue;
            const auto previous = prefix_block_hash(compatibility_hash_,config_.block_size_tokens,parent,tokens,length->first,extra_key);
            if (tail_metadata_.count(previous) == 0) continue;
            if (retained++ >= config_.retain_tail_versions) victims.push_back(previous);
        }
        if (victims.empty()) tail_cleanup_.erase(hash);
    }

    void prune_tail_cleanup_locked() {
        for (auto entry = tail_cleanup_.begin(); entry != tail_cleanup_.end();) {
            if (config_.max_disk_bytes != 0 && disk_.count(entry->first) == 0) {
                if (pending_.count(entry->first) != 0) { ++entry; continue; }
                entry = tail_cleanup_.erase(entry);
                continue;
            }
            if (config_.max_disk_bytes == 0 && hot_.count(entry->first) == 0) {
                entry = tail_cleanup_.erase(entry);
                continue;
            }
            auto& victims = entry->second;
            for (auto victim = victims.begin(); victim != victims.end();) {
                auto disk = disk_.find(*victim);
                if (pending_.count(*victim) != 0 || pins_.count(*victim) != 0 ||
                    (disk != disk_.end() && disk->second.pins != 0)) { ++victim; continue; }
                if (disk != disk_.end()) {
                    std::error_code error;
                    std::filesystem::remove(disk->second.path, error);
                    if (error) { ++victim; continue; }
                    remove_text(disk->second.path);
                    disk_bytes_ -= disk->second.file_bytes;
                    disk_.erase(disk);
                }
                auto hot = hot_.find(*victim);
                if (hot != hot_.end()) {
                    hot_bytes_ -= hot->second.payload->size();
                    hot_.erase(hot);
                }
                deferred_.erase(*victim);
                forget_tail_if_unused_locked(*victim);
                ++metrics_.superseded_blocks;
                victim = victims.erase(victim);
            }
            if (victims.empty()) entry = tail_cleanup_.erase(entry);
            else ++entry;
        }
    }

    void remember_tail_locked(const BlockHash& hash, const BlockHash& parent,
        std::size_t count) {
        if (count >= config_.block_size_tokens || tail_metadata_.count(hash) != 0) return;
        tail_metadata_.emplace(hash, std::make_pair(parent, count));
        ++tail_lengths_[parent][count];
    }

    void forget_tail_if_unused_locked(const BlockHash& hash) {
        if (disk_.count(hash) != 0 || pending_.count(hash) != 0 || hot_.count(hash) != 0) return;
        const auto metadata = tail_metadata_.find(hash);
        if (metadata == tail_metadata_.end()) return;
        auto parent = tail_lengths_.find(metadata->second.first);
        if (parent != tail_lengths_.end()) {
            auto length = parent->second.find(metadata->second.second);
            if (length != parent->second.end() && --length->second == 0)
                parent->second.erase(length);
            if (parent->second.empty()) tail_lengths_.erase(parent);
        }
        tail_metadata_.erase(metadata);
    }

    void sync_metrics_locked() {
        metrics_.disk_blocks = disk_.size();
        metrics_.disk_bytes = disk_bytes_;
        metrics_.hot_blocks = hot_.size();
        metrics_.hot_bytes = hot_bytes_;
        metrics_.pending_writes = pending_.size();
        metrics_.pending_bytes = pending_write_bytes_;
        metrics_.pending_max_bytes = config_.max_pending_bytes;
    }

    PrefixBlockStoreConfig config_;
    BlockHash compatibility_hash_{};
    std::filesystem::path namespace_dir_;
    mutable std::mutex mutex_;
    std::condition_variable work_available_;
    std::condition_variable writes_finished_;
    std::thread worker_;
    std::deque<WriteRequest> writes_;
    std::unordered_map<
        BlockHash,
        std::shared_ptr<const std::vector<std::uint8_t>>,
        HashHasher> pending_;
    std::unordered_map<BlockHash, DiskEntry, HashHasher> disk_;
    std::unordered_map<BlockHash, HotEntry, HashHasher> hot_;
    std::unordered_map<BlockHash, WriteRequest, HashHasher> deferred_;
    std::unordered_map<BlockHash, std::size_t, HashHasher> pins_;
    std::unordered_map<BlockHash, std::map<std::size_t, std::size_t>, HashHasher> tail_lengths_;
    std::unordered_map<BlockHash, std::pair<BlockHash, std::size_t>, HashHasher> tail_metadata_;
    std::unordered_map<BlockHash, std::vector<BlockHash>, HashHasher> tail_cleanup_;
    std::uint64_t disk_bytes_ = 0;
    std::uint64_t hot_bytes_ = 0;
    std::uint64_t pending_write_bytes_ = 0;
    std::uint64_t clock_ = 0;
    std::uint64_t cache_epoch_ = 0;
    std::size_t active_writes_ = 0;
    bool stopping_ = false;
    bool disk_budget_pending_ = false;
    bool disk_identity_pending_ = false;
    bool clearing_ = false;
    PrefixBlockStoreMetrics metrics_;
    std::atomic<std::uint64_t> temp_counter_{0};
};

PrefixBlockStore::PrefixBlockStore(PrefixBlockStoreConfig config)
    : implementation_(
          std::make_unique<Implementation>(std::move(config))) {}

PrefixBlockStore::~PrefixBlockStore() = default;

std::size_t PrefixBlockStore::block_size_tokens() const noexcept {
    return implementation_->block_size_tokens();
}

const BlockHash& PrefixBlockStore::compatibility_hash() const noexcept {
    return implementation_->compatibility_hash();
}











std::optional<std::vector<std::uint8_t>> PrefixBlockStore::load(
    const BlockHash& hash) {
    return implementation_->load(hash);
}

std::vector<PagedPrefixPayload> PrefixBlockStore::load_prefix(
    const std::vector<BlockHash>& blocks) {
    return implementation_->load_prefix(blocks);
}

bool PrefixBlockStore::invalidate(const BlockHash& block) {
    return implementation_->invalidate(block);
}

void PrefixBlockStore::pin(const std::vector<BlockHash>& blocks) {
    implementation_->pin(blocks);
}

void PrefixBlockStore::unpin(const std::vector<BlockHash>& blocks) {
    implementation_->unpin(blocks);
}

std::uint64_t PrefixBlockStore::set_hot_limit(std::uint64_t max_bytes) {
    return implementation_->set_hot_limit(max_bytes);
}

std::uint64_t PrefixBlockStore::set_disk_limit(std::uint64_t max_bytes) {
    return implementation_->set_disk_limit(max_bytes);
}

std::size_t PrefixBlockStore::refresh_disk_index() {
    return implementation_->refresh_disk_index();
}

std::uint64_t PrefixBlockStore::trim_hot(std::uint64_t target_bytes) {
    return implementation_->trim_hot(target_bytes);
}

std::uint64_t PrefixBlockStore::reserve_ram(std::uint64_t allocation_bytes) {
    return implementation_->reserve_ram(allocation_bytes);
}

void PrefixBlockStore::flush() {
    implementation_->flush();
}

std::size_t PrefixBlockStore::clear() {
    return implementation_->clear();
}

PrefixBlockStoreMetrics PrefixBlockStore::metrics() const {
    return implementation_->metrics();
}

} // namespace mfq::cache

namespace mfq::cache {
BlockHash PrefixBlockStore::store(const BlockHash& hash,const BlockHash& parent,
        const std::int64_t* tokens,std::size_t count,PagedPrefixPayload payload,
        std::string_view extra_key,bool replace_existing) {
    return implementation_->store(hash,parent,tokens,count,std::move(payload),extra_key,replace_existing);
}
bool PrefixBlockStore::contains(const BlockHash& hash,const BlockHash& parent,std::size_t count) {
    return implementation_->contains(hash,parent,count);
}
std::vector<std::size_t> PrefixBlockStore::tail_lengths(const BlockHash& parent,std::size_t maximum) const {
    return implementation_->tail_lengths(parent,maximum);
}
} // namespace mfq::cache
