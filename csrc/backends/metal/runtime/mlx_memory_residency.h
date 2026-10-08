#pragma once

#include "mlx_weight_residency.h"
#include <mlx/backend/metal/device.h>
#include <mlx/memory.h>

#include <atomic>
#include <cstddef>
#include <stdexcept>

namespace mfq::metal {

class MlxMemoryResidency {
public:
    static void configure(std::size_t limit) {
        if (limit == 0 || !available(mlx::core::metal::device(mlx::core::Device::gpu))) {
            throw std::runtime_error("Metal memory wiring is unavailable");
        }
        mlx::core::set_wired_limit(limit);
        limit_bytes().store(limit, std::memory_order_relaxed);
        MlxWeightResidency::begin_load();
    }

    static void refresh() {
        const auto limit = configured_limit();
        if (limit == 0) return;
        const auto active = mlx::core::get_active_memory();
        if (active > limit) {
            throw std::runtime_error(
                "Metal active memory exceeds the wired budget; "
                "reduce resident experts or unload an idle model");
        }
        mlx::core::set_wired_limit(limit - 1);
        mlx::core::set_wired_limit(limit);
        if (wired_bytes() < active) {
            throw std::runtime_error(
                "Metal residency does not cover the loaded allocations; "
                "reduce resident experts or unload an idle model");
        }
    }

    static std::size_t configured_limit() {
        return limit_bytes().load(std::memory_order_relaxed);
    }

    static std::size_t wired_bytes() {
        return allocated_bytes(mlx::core::metal::device(mlx::core::Device::gpu));
    }

private:
    static std::atomic<std::size_t>& limit_bytes() {
        static std::atomic<std::size_t> value{0};
        return value;
    }

    template <typename Device>
    static bool available(Device& device) {
        if constexpr (requires { device.residency_sets(); }) {
            return device.residency_sets().enabled();
        } else {
            return device.residency_set().mtl_residency_set() != nullptr;
        }
    }

    template <typename Device>
    static std::size_t allocated_bytes(Device& device) {
        if constexpr (requires { device.residency_sets(); }) {
            return device.residency_sets().wired_size();
        } else if (__builtin_available(macOS 15, *)) {
            if (const auto* set = device.residency_set().mtl_residency_set()) {
                return set->allocatedSize();
            }
        }
        return 0;
    }
};

}
