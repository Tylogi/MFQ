#include "mlx_memory_residency.h"

#include <mlx/allocator.h>
#include <mlx/mlx.h>

#include <cassert>
#include <cstring>
#include <iostream>
#include <optional>

int main() {
    using mfq::metal::MlxMemoryResidency;
    using mlx::core::array;
    constexpr std::size_t limit = 128 << 20;
    constexpr int bytes = 96 << 20;
    mlx::core::set_default_device(mlx::core::Device::gpu);
    if (!__builtin_available(macOS 15, *)) return 0;
    MlxMemoryResidency::configure(limit);
    auto allocate = [](int size) {
        auto value = array(mlx::core::allocator::malloc(size),
            mlx::core::Shape{size}, mlx::core::uint8);
        std::memset(value.data<std::uint8_t>(), 1, size);
        return value;
    };
    std::optional<array> scratch{allocate(bytes)};
    std::optional<array> weights{allocate(bytes)};
    const auto saturated = MlxMemoryResidency::wired_bytes();
    scratch.reset();
    mlx::core::clear_cache();
    const auto before = MlxMemoryResidency::wired_bytes();
    assert(before < bytes);
    mlx::core::set_wired_limit(limit);
    assert(MlxMemoryResidency::wired_bytes() == before);
    MlxMemoryResidency::refresh();
    const auto after = MlxMemoryResidency::wired_bytes();
    assert(after >= bytes && after <= limit);
    std::optional<array> excess{allocate(bytes)};
    bool rejected = false;
    try { MlxMemoryResidency::refresh(); }
    catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
    excess.reset();
    weights.reset();
    mlx::core::clear_cache();
    mlx::core::set_wired_limit(0);
    std::cout << "Metal residency: saturated=" << saturated
              << " released=" << before << " rewired=" << after << '\n';
}
