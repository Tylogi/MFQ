#pragma once
#include <cstddef>
#include <cstdint>
namespace mfq {
struct ReadSpan { std::uint64_t offset=0;std::byte* destination=nullptr;std::size_t size=0; };
}
