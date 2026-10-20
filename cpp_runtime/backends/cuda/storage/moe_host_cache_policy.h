#pragma once
#include <cstddef>

// Complete preload keeps a host image of every routed expert, including the
// experts mirrored in the GPU cache, so the RAM tier must fit the whole set.
// Both the server budget check and the native preload guard use this rule.
[[nodiscard]] constexpr bool moe_preload_fits_host(
        std::size_t expert_fields_bytes, std::size_t host_cache_bytes) noexcept {
    return expert_fields_bytes <= host_cache_bytes;
}
