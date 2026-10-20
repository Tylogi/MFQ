#include "runtime/moe_residency.h"
#include "storage/moe_host_cache_policy.h"

#include <iostream>
#include <stdexcept>

namespace {

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

} // namespace

int main() {
    try {
        using MemorySample = MoeResidencyManager::MemorySample;
        const MemorySample physical{1000, 8ull << 30};
        require(physical.available_host_bytes() == (8ull << 30),
            "platforms without commit accounting must keep physical headroom");
        const MemorySample commit_limited{1000, 8ull << 30, 3ull << 30};
        require(commit_limited.available_host_bytes() == (3ull << 30),
            "Windows commit headroom must cap host allocations");
        const MemorySample roomier_commit{1000, 8ull << 30, 12ull << 30};
        require(roomier_commit.available_host_bytes() == (8ull << 30),
            "commit headroom must not raise the physical limit");
        const MemorySample exhausted_commit{1000, 8ull << 30, 0};
        require(exhausted_commit.available_host_bytes() == 0,
            "zero commit headroom must prevent additional host allocations");
        require(moe_preload_fits_host(10ull << 30, 10ull << 30),
            "preload must fit the complete routed expert set in host RAM");
        require(!moe_preload_fits_host(10ull << 30, 9ull << 30),
            "preload must not assume only cold experts occupy host RAM");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
