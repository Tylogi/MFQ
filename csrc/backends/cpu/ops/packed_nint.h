#pragma once

#include <cstddef>
#include <cstdint>

namespace mfq::cpu {

// Borrowed, canonical NINT row storage. Group metadata has already been
// unpacked to bytes by the format loader; q codes remain compressed.
struct PackedNintView {
    const std::uint8_t* packed = nullptr;
    std::size_t packed_bytes = 0;
    const std::uint8_t* row_bits = nullptr;
    const std::int64_t* row_bit_offsets = nullptr;
    const std::uint8_t* group_scale = nullptr;
    const std::uint8_t* group_min = nullptr;
    const float* row_scale = nullptr;
    const float* row_min = nullptr;
    std::int64_t outputs = 0;
    std::int64_t width = 0;
    std::int64_t group_size = 0;
};

enum class PackedNintKernel { automatic, scalar };
bool packed_nint_has_avx2();
using PackedNintRows=void(*)(const PackedNintView&,const float*,
    std::int64_t,std::int64_t,float*,std::int64_t,std::int64_t,std::int64_t);
// Canonical row storage is validated by its loader; select once per expert job.
PackedNintRows packed_nint_rows_kernel() noexcept;

// FP32 activation -> FP32 output, without activation quantization or dense
// weight materialization. Strides are in float elements. Storage is borrowed
// until completion; callers must provide disjoint input and output storage.
void packed_nint_matmul(const PackedNintView& weight, const float* input,
    std::int64_t batch, std::int64_t input_stride, float* output,
    std::int64_t output_stride, int threads,
    PackedNintKernel kernel = PackedNintKernel::automatic);

namespace detail {
void packed_nint_scalar_rows(const PackedNintView&, const float*,
    std::int64_t batch, std::int64_t input_stride, float*,
    std::int64_t output_stride, std::int64_t begin, std::int64_t end);
void packed_nint_avx2_rows(const PackedNintView&, const float*,
    std::int64_t batch, std::int64_t input_stride, float*,
    std::int64_t output_stride, std::int64_t begin, std::int64_t end);
} // namespace detail
} // namespace mfq::cpu
