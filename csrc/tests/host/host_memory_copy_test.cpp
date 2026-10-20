#include "host_memory_copy.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>

int main() {
    for (bool streaming : {false, true}) {
        for (std::size_t size : {0u, 1u, 4095u, 4096u, 8193u, (2u << 20) - 1, 2u << 20}) {
            std::vector<std::uint8_t> source(size + 7);
            for (std::size_t i = 0; i < source.size(); ++i)
                source[i] = static_cast<std::uint8_t>(i * 37 + i / 113);
            for (std::size_t offset : {0u, 1u, 31u, 63u}) {
                std::vector<std::uint8_t> destination(size + 128, 0x5a);
                {
                    mfq::cpu::HostMemoryCopyBatch batch(streaming);
                    batch.copy(destination.data() + offset, source.data() + 7, size);
                    batch.copy(nullptr, nullptr, 0);
                }
                assert(std::equal(source.begin() + 7, source.end(), destination.begin() + offset));
                assert(std::all_of(destination.begin(), destination.begin() + offset,
                    [](auto value) { return value == 0x5a; }));
                assert(std::all_of(destination.begin() + offset + size, destination.end(),
                    [](auto value) { return value == 0x5a; }));
            }
        }
    }
}
