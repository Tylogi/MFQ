#include "packed_nint.h"
#include "mfq/host_parallel.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

#if defined(MFQ_CPU_PACKED_AVX2) && defined(_MSC_VER)
#include <intrin.h>
#endif

namespace mfq::cpu {
namespace {
void validate(const PackedNintView& w) {
    if (w.outputs < 0 || w.width <= 0 || w.group_size <= 0)
        throw std::invalid_argument("invalid packed NINT geometry");
    if (!w.outputs) return;
    if (!w.packed || !w.row_bits || !w.row_bit_offsets || !w.group_scale ||
        !w.group_min || !w.row_scale || !w.row_min)
        throw std::invalid_argument("missing packed NINT storage");
    if (w.packed_bytes > std::numeric_limits<std::uint64_t>::max() / 8)
        throw std::invalid_argument("packed NINT storage is too large");
    const auto capacity = static_cast<std::uint64_t>(w.packed_bytes) * 8;
    const auto width = static_cast<std::uint64_t>(w.width);
    for (std::int64_t row=0; row<w.outputs; ++row) {
        const auto bits = w.row_bits[row];
        const auto offset = w.row_bit_offsets[row];
        if (bits < 1 || bits > 8 || offset < 0 ||
            static_cast<std::uint64_t>(offset) > capacity ||
            width > (capacity - static_cast<std::uint64_t>(offset)) / bits)
            throw std::invalid_argument("packed NINT row exceeds storage");
    }
}

std::uint32_t code(const PackedNintView& w, std::uint64_t bit, int bits) {
    const auto byte = static_cast<std::size_t>(bit >> 3);
    const auto shift = static_cast<unsigned>(bit & 7);
    std::uint32_t word = w.packed[byte];
    if (shift + bits > 8) word |= std::uint32_t(w.packed[byte+1]) << 8;
    return (word >> shift) & ((1u << bits)-1);
}
} // namespace

bool packed_nint_has_avx2() {
    static const bool supported = [] {
#if defined(MFQ_CPU_PACKED_AVX2) && defined(_MSC_VER)
        int leaf[4];
        __cpuid(leaf, 0);
        if (leaf[0] < 7) return false;
        __cpuidex(leaf, 1, 0);
        if ((leaf[2] & ((1<<27) | (1<<28))) != ((1<<27) | (1<<28))) return false;
        if ((_xgetbv(0) & 6) != 6) return false;
        __cpuidex(leaf, 7, 0);
        return (leaf[1] & (1<<5)) != 0;
#elif defined(MFQ_CPU_PACKED_AVX2) && (defined(__GNUC__) || defined(__clang__))
        __builtin_cpu_init();
        return static_cast<bool>(__builtin_cpu_supports("avx2"));
#else
        return false;
#endif
    }();
    return supported;
}

void detail::packed_nint_scalar_rows(const PackedNintView& w, const float* input,
    std::int64_t batch, std::int64_t input_stride, float* output,
    std::int64_t output_stride, std::int64_t begin, std::int64_t end) {
    const auto groups = 1 + (w.width - 1) / w.group_size;
    std::vector<float> sums(static_cast<std::size_t>(batch));
    for (auto row=begin; row<end; ++row) {
        std::fill(sums.begin(), sums.end(), 0.0f);
        const int bits = w.row_bits[row];
        auto bit = static_cast<std::uint64_t>(w.row_bit_offsets[row]);
        for (std::int64_t group=0; group<groups; ++group) {
            const auto meta = row * groups + group;
            const float scale = w.row_scale[row] * w.group_scale[meta];
            const float minimum = w.row_min[row] * w.group_min[meta];
            const auto start = group * w.group_size;
            const auto stop = start + std::min(w.group_size, w.width - start);
            for (auto column=start; column<stop; ++column, bit+=bits) {
                // Separate decode operations match native NINT dequantization.
                const float decoded = scale * static_cast<float>(code(w, bit, bits)) - minimum;
                for (std::int64_t sample=0; sample<batch; ++sample)
                    sums[static_cast<std::size_t>(sample)] +=
                        decoded * input[sample*input_stride+column];
            }
        }
        for (std::int64_t sample=0; sample<batch; ++sample)
            output[sample*output_stride+row] = sums[static_cast<std::size_t>(sample)];
    }
}

void packed_nint_matmul(const PackedNintView& w, const float* input,
    std::int64_t batch, std::int64_t input_stride, float* output,
    std::int64_t output_stride, int threads, PackedNintKernel kernel) {
    validate(w);
    if (batch < 0 || input_stride < w.width || output_stride < w.outputs)
        throw std::invalid_argument("invalid packed NINT activation geometry");
    if (!batch || !w.outputs) return;
    if (!input || !output) throw std::invalid_argument("missing packed NINT activation storage");
    auto rows = detail::packed_nint_scalar_rows;
#ifdef MFQ_CPU_PACKED_AVX2
    if (kernel == PackedNintKernel::automatic && packed_nint_has_avx2())
        rows = detail::packed_nint_avx2_rows;
#endif
    host_parallel_for(0, w.outputs, 1, threads, [&](auto begin, auto end) {
        rows(w, input, batch, input_stride, output, output_stride, begin, end);
    });
}
} // namespace mfq::cpu
