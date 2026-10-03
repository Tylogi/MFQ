#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <sys/mman.h>

namespace mfq::metal::detail {

// MLX stores every individual shape axis as int32 even though array::size()
// and the underlying allocation are size_t. Packed runtime streams are flat
// byte arenas, so their logical rank is irrelevant to every consumer. Split a
// large arena over two exact factors instead of rejecting it merely because a
// single axis would exceed INT32_MAX. No padding is permitted: descriptor
// offsets and slot-count validation both use the exact flat element count.
struct PackedStorageLayout {
    int rows = 1;
    int columns = 1;

    constexpr bool is_matrix() const noexcept {
        return rows != 1;
    }
};

inline PackedStorageLayout packed_storage_layout(std::size_t elements) {
    if (elements == 0) {
        throw std::invalid_argument(
            "packed storage must contain at least one element");
    }
    constexpr auto max_axis = static_cast<std::size_t>(
        std::numeric_limits<std::int32_t>::max());
    if (elements <= max_axis) {
        return {1, static_cast<int>(elements)};
    }
    if (elements / max_axis > max_axis) {
        throw std::length_error(
            "packed storage exceeds the MLX multi-axis shape range");
    }

    const auto minimum_rows =
        elements / max_axis + (elements % max_axis != 0);
    for (auto rows = minimum_rows; rows <= elements / rows; ++rows) {
        if (elements % rows != 0) {
            continue;
        }
        const auto columns = elements / rows;
        if (rows <= max_axis && columns <= max_axis) {
            return {
                static_cast<int>(rows),
                static_cast<int>(columns),
            };
        }
    }
    throw std::length_error(
        "packed storage cannot be represented by exact MLX int32 axes");
}

// Large model-load scratch buffers must bypass Darwin malloc's large-object
// depot: freed multi-GiB vector capacities can otherwise remain resident and
// count against the model footprint. Anonymous mappings are returned to the
// kernel immediately when a staging vector dies.
template <typename T>
class MmapAllocator {
public:
    using value_type = T;
    using is_always_equal = std::true_type;

    MmapAllocator() noexcept = default;

    template <typename U>
    MmapAllocator(const MmapAllocator<U>&) noexcept {}

    T* allocate(std::size_t count) {
        if (count == 0) {
            return nullptr;
        }
        if (count > max_size()) {
            throw std::bad_array_new_length();
        }
        const auto bytes = count * sizeof(T);
        void* mapping = ::mmap(
            nullptr,
            bytes,
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS,
            -1,
            0);
        if (mapping == MAP_FAILED) {
            throw std::bad_alloc();
        }
        return static_cast<T*>(mapping);
    }

    void deallocate(T* pointer, std::size_t count) noexcept {
        if (pointer != nullptr && count != 0) {
            ::munmap(pointer, count * sizeof(T));
        }
    }

    constexpr std::size_t max_size() const noexcept {
        return std::numeric_limits<std::size_t>::max() / sizeof(T);
    }
};

template <typename T, typename U>
bool operator==(
    const MmapAllocator<T>&,
    const MmapAllocator<U>&) noexcept {
    return true;
}

template <typename T>
using StagingVector = std::vector<T, MmapAllocator<T>>;

}  // namespace mfq::metal::detail
