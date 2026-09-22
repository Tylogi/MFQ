#include "../backends/cuda/runtime/paged_kv_allocator.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

} // namespace

int main() {
    using mfq::cuda::continuous::PagedKvPageAllocator;

    PagedKvPageAllocator allocator(8);
    const auto pages = allocator.allocate(8);
    require(pages == std::vector<std::int32_t>({0, 1, 2, 3, 4, 5, 6, 7}),
        "Paged KV allocator did not issue all pages monotonically");
    require(allocator.live_pages() == 8 &&
            allocator.peak_live_pages() == 8 &&
            allocator.high_watermark() == 8 &&
            allocator.allocation_count() == 8 &&
            allocator.reuse_count() == 0 &&
            allocator.release_count() == 0,
        "Paged KV allocator full-pool accounting mismatch");

    bool exhausted = false;
    try {
        (void)allocator.allocate(1);
    } catch (const std::runtime_error & error) {
        exhausted = std::string(error.what()) ==
            "Paged KV physical page pool exhausted";
    }
    require(exhausted, "Paged KV allocator did not report pool exhaustion");
    require(allocator.live_pages() == 8 &&
            allocator.peak_live_pages() == 8 &&
            allocator.high_watermark() == 8 &&
            allocator.allocation_count() == 8 &&
            allocator.reuse_count() == 0 &&
            allocator.release_count() == 0 &&
            std::all_of(pages.begin(), pages.end(),
                [&allocator](std::int32_t page) { return allocator.owns(page); }),
        "Paged KV allocator state changed after pool exhaustion");

    allocator.release({3});
    require(allocator.live_pages() == 7 &&
            allocator.peak_live_pages() == 8 &&
            allocator.high_watermark() == 8 &&
            allocator.allocation_count() == 8 &&
            allocator.reuse_count() == 0 &&
            allocator.release_count() == 1,
        "Paged KV allocator release accounting mismatch");

    bool invalid_release_rejected = false;
    try {
        allocator.release({3});
    } catch (const std::logic_error &) {
        invalid_release_rejected = true;
    }
    require(invalid_release_rejected &&
            allocator.live_pages() == 7 &&
            allocator.peak_live_pages() == 8 &&
            allocator.high_watermark() == 8 &&
            allocator.allocation_count() == 8 &&
            allocator.reuse_count() == 0 &&
            allocator.release_count() == 1,
        "Paged KV allocator accepted an invalid release");

    const auto reused = allocator.allocate(1);
    require(reused == std::vector<std::int32_t>({3}),
        "Paged KV allocator did not reuse the released page");
    require(allocator.live_pages() == 8 &&
            allocator.peak_live_pages() == 8 &&
            allocator.high_watermark() == 8 &&
            allocator.allocation_count() == 9 &&
            allocator.reuse_count() == 1 &&
            allocator.release_count() == 1,
        "Paged KV allocator reuse accounting mismatch");

    std::cout << "Paged KV allocator test passed\n";
    return 0;
}
