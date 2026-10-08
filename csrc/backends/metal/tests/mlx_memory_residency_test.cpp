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
        mfq::metal::MlxWeightResidency::track(value);
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
    bool weight_budget_rejected = false;
    try { mfq::metal::MlxWeightResidency::finish_load(bytes - 1); }
    catch (const std::runtime_error&) { weight_budget_rejected = true; }
    assert(weight_budget_rejected);
    mfq::metal::MlxWeightResidency::begin_load();
    mfq::metal::MlxWeightResidency::track(*weights);
    auto alias = mlx::core::reshape(*weights, mlx::core::Shape{bytes / 2, 2});
    mlx::core::eval(alias);
    mfq::metal::MlxWeightResidency::track(alias);
    auto residency = mfq::metal::MlxWeightResidency::finish_load(limit);
    assert(residency && residency->bytes() == weights->buffer_size());
    assert(weights->data<std::uint8_t>()[0] == 1);
    assert(weights->data<std::uint8_t>()[bytes - 1] == 1);
    assert(!mfq::metal::MlxWeightResidency::finish_load(limit));
    alias = array({0}, mlx::core::uint8);
    std::optional<array> excess{allocate(bytes)};
    bool rejected = false;
    try { MlxMemoryResidency::refresh(); }
    catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
    excess.reset();
    weights.reset();
    mlx::core::clear_cache();
    for (int attempt = 0; attempt < 100 && residency->bytes() != 0; ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    assert(residency->bytes() == 0);
    residency.reset();
    mlx::core::set_wired_limit(0);
    assert(mlx::core::get_active_memory() < (1 << 20));
    std::cout << "Metal residency: saturated=" << saturated
              << " released=" << before << " rewired=" << after << '\n';
}
