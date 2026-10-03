#pragma once

#include <cstddef>
#include <string>
#include <string_view>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

namespace mfq::metal {

inline const std::string& mlx_apple_chip_name() noexcept {
    static const std::string name = []() noexcept {
#if defined(__APPLE__)
        try {
            std::size_t size = 0;
            if (::sysctlbyname(
                    "machdep.cpu.brand_string",
                    nullptr,
                    &size,
                    nullptr,
                    0) != 0 || size <= 1) {
                return std::string{};
            }
            std::string value(size, '\0');
            if (::sysctlbyname(
                    "machdep.cpu.brand_string",
                    value.data(),
                    &size,
                    nullptr,
                    0) != 0) {
                return std::string{};
            }
            while (!value.empty() && value.back() == '\0') {
                value.pop_back();
            }
            return value;
        } catch (...) {
            return std::string{};
        }
#else
        return std::string{};
#endif
    }();
    return name;
}

inline bool mlx_apple_chip_is(std::string_view expected) noexcept {
    return mlx_apple_chip_name() == expected;
}

inline bool mlx_apple_chip_starts_with(std::string_view prefix) noexcept {
    return mlx_apple_chip_name().starts_with(prefix);
}

} // namespace mfq::metal
