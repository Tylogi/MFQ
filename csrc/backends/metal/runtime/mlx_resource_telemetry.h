#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace mfq::metal {

struct MlxResourceUsage {
    std::size_t cache_bytes = 0;
    std::size_t contexts = 0;
    std::size_t dynamic_weight_bytes = 0;
    std::size_t expert_payload_bytes = 0;
    std::size_t ple_payload_bytes = 0;
};

class MlxResourceTelemetry {
    struct Entry {
        std::mutex mutex;
        MlxResourceUsage value;
        std::function<MlxResourceUsage()> read;
    };
    struct Registry {
        std::mutex mutex;
        std::vector<std::weak_ptr<Entry>> entries;
    };
    static Registry& registry() {
        static Registry value;
        return value;
    }

    void ensure_entry() {
        if (entry_) return;
        entry_ = std::make_shared<Entry>();
        auto& state = registry();
        std::lock_guard lock(state.mutex);
        state.entries.push_back(entry_);
    }

public:
    static std::atomic<std::size_t>& ple_read_counter() {
        static std::atomic<std::size_t> value{0};
        return value;
    }

    MlxResourceTelemetry() = default;
    MlxResourceTelemetry(MlxResourceTelemetry&&) noexcept = default;
    MlxResourceTelemetry& operator=(MlxResourceTelemetry&&) noexcept = default;
    MlxResourceTelemetry(const MlxResourceTelemetry&) = default;
    MlxResourceTelemetry& operator=(const MlxResourceTelemetry&) = default;

    void set(MlxResourceUsage value) noexcept {
        if (!entry_ && value.cache_bytes == 0 && value.dynamic_weight_bytes == 0 &&
            value.expert_payload_bytes == 0 && value.ple_payload_bytes == 0) return;
        ensure_entry();
        std::lock_guard lock(entry_->mutex);
        entry_->value = value;
    }
    void bind(std::function<MlxResourceUsage()> read) {
        ensure_entry();
        std::lock_guard lock(entry_->mutex);
        entry_->read = std::move(read);
    }

    static MlxResourceUsage snapshot() {
        auto& state = registry();
        MlxResourceUsage result;
        std::vector<std::shared_ptr<Entry>> active;
        {
            std::lock_guard lock(state.mutex);
            auto& entries = state.entries;
            entries.erase(std::remove_if(entries.begin(), entries.end(),
                [](const auto& entry) { return entry.expired(); }), entries.end());
            for (const auto& weak : entries) if (auto entry = weak.lock()) active.push_back(std::move(entry));
        }
        for (const auto& entry : active) {
            std::lock_guard lock(entry->mutex);
            const auto value = entry->read ? entry->read() : entry->value;
            result.cache_bytes += value.cache_bytes;
            result.contexts = std::max(result.contexts, value.contexts);
            result.dynamic_weight_bytes += value.dynamic_weight_bytes;
            result.expert_payload_bytes += value.expert_payload_bytes;
            result.ple_payload_bytes += value.ple_payload_bytes;
        }
        return result;
    }

private:
    std::shared_ptr<Entry> entry_;
};

}
