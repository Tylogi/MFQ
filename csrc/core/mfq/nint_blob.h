#pragma once

#include <cstdint>

namespace mfq {

inline constexpr std::uint8_t kNintAdaptiveStorageFlag = 0x80;

constexpr bool nint_has_adaptive_storage(std::uint8_t raw_bits) noexcept {
    return (raw_bits & kNintAdaptiveStorageFlag) != 0;
}

constexpr std::uint8_t nint_logical_bits(std::uint8_t raw_bits) noexcept {
    return raw_bits & ~kNintAdaptiveStorageFlag;
}

} // namespace mfq
