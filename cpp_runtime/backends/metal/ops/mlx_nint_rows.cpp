#include "mlx_nint_rows.h"

#include <optional>
#include <stdexcept>

namespace mfq::metal {
namespace {

constexpr const char* kRowDecodeHeader = R"METAL(
inline uint mfq_nint_row_bits(
    const device uchar* packed, uint offset, uint shift, uint index, uint bits) {
    uint bit = shift + index * bits;
    uint byte = offset + (bit >> 3u);
    uint value = uint(packed[byte]) | (uint(packed[byte + 1u]) << 8u);
    return (value >> (bit & 7u)) & ((1u << bits) - 1u);
}
)METAL";

constexpr const char* kRowDecode = R"METAL(
    uint column = thread_position_in_grid.x;
    uint row = thread_position_in_grid.y;
    if (column >= uint(WIDTH) || row >= uint(ROWS)) return;
    uint base = row * 6u;
    uint layout = descriptors[base + 3u];
    uint qbits = layout & 15u;
    uint kbits = (layout >> 4u) & 15u;
    uint group = column / descriptors[base + 4u];
    uint q = mfq_nint_row_bits(
        packed, descriptors[base], (layout >> 8u) & 7u, column, qbits);
    uint s = mfq_nint_row_bits(
        packed, descriptors[base + 1u], (layout >> 12u) & 7u, group, kbits);
    uint m = mfq_nint_row_bits(
        packed, descriptors[base + 2u], (layout >> 16u) & 7u, group, kbits);
    uint anchors = descriptors[base + 5u];
    float scale = float(as_type<half>(ushort(anchors & 65535u))) * float(s);
    float minimum = float(as_type<half>(ushort(anchors >> 16u))) * float(m);
    values[row * uint(WIDTH) + column] = T(scale * float(q) - minimum);
)METAL";

const mlx::core::fast::CustomKernelFunction& row_decode_kernel() {
    static const auto kernel = [] {
        mlx::core::CompileOptions options;
        options.math_mode = mlx::core::MathMode::Safe;
        return mlx::core::fast::metal_kernel(
            "mfq_cpp_nint_mapped_row_decode", {"packed", "descriptors"},
            {"values"}, kRowDecode, kRowDecodeHeader, true, false, options);
    }();
    return kernel;
}

} // namespace

mlx::core::array MlxNintRowBatch::decode(mlx::core::Dtype dtype) const {
    if (dtype != mlx::core::float16 && dtype != mlx::core::float32) {
        throw std::invalid_argument("NINT row decode requires F16 or F32 output");
    }
    validate();
    const mlx::core::array packed(this->packed().begin(), mlx::core::Shape{static_cast<int>(packed_nbytes())});
    const mlx::core::array descriptors(this->descriptors().begin(), mlx::core::Shape{static_cast<int>(rows()), 6});
    return row_decode_kernel()(
        {packed, descriptors}, {mlx::core::Shape{static_cast<int>(rows()), width()}}, {dtype},
        {width(), static_cast<int>(rows()), 1}, {128, 1, 1},
        {{"T", dtype}, {"WIDTH", width()}, {"ROWS", static_cast<int>(rows())}},
        std::nullopt, false, {}).front();
}

} // namespace mfq::metal
