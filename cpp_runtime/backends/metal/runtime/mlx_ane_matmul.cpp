#include "mlx_ane_matmul.h"
#include <mlx/allocator.h>
#include <mlx/backend/cpu/encoder.h>
#include <mlx/primitives.h>
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace mfq::metal {
namespace {
using namespace mlx::core;
constexpr int kTile = 4096;
int padded(int value) { return (value + 31) / 32 * 32; }

class AneMatmulPrimitive final : public UnaryPrimitive {
public:
    AneMatmulPrimitive(Stream stream, std::shared_ptr<MlxAneMatmul> engine)
        : UnaryPrimitive(stream), engine_(std::move(engine)) {}
    void eval_cpu(const std::vector<array>& inputs, array& output) override {
        output.set_data(allocator::malloc(output.nbytes()));
        cpu::get_command_encoder(stream()).dispatch([engine = engine_, inputs, output]() mutable {
            const auto result = engine->evaluate(inputs[0], inputs[1]);
            std::memcpy(output.data<float>(), result.data(), output.nbytes());
        });
    }
    void eval_gpu(const std::vector<array>&, array&) override {
        throw std::runtime_error("ANE training cannot execute on GPU");
    }
    std::vector<array> vjp(const std::vector<array>& inputs, const std::vector<array>& gradients,
        const std::vector<int>& argnums, const std::vector<array>&) override {
        std::vector<array> result;
        for (auto arg : argnums) {
            if (arg == 0) result.push_back((*engine_)(gradients[0], transpose(inputs[1], stream())));
            else if (arg == 1) result.push_back((*engine_)(transpose(inputs[0], stream()), gradients[0]));
            else throw std::invalid_argument("invalid ANE matmul gradient input");
        }
        return result;
    }
    const char* name() const override { return "AneMatmul"; }
private:
    std::shared_ptr<MlxAneMatmul> engine_;
};
}

mlx::core::Stream MlxAneMatmul::cpu() const {
    thread_local const auto stream = mlx::core::new_stream(mlx::core::Device(mlx::core::Device::cpu));
    return stream;
}

std::shared_ptr<AneMatmul> MlxAneMatmul::kernel(int rows, int inner, int columns) {
    std::lock_guard lock(mutex_);
    const auto key = std::tuple{rows, inner, columns};
    auto& result = kernels_[key];
    if (!result) result = std::make_shared<AneMatmul>(rows, inner, columns);
    return result;
}

void MlxAneMatmul::prepare(int rows, int inner, int columns) {
    if (rows <= 0 || inner <= 0 || columns <= 0) throw std::invalid_argument("invalid ANE matrix shape");
    for (int m = 0; m < rows; m += kTile)
        for (int k = 0; k < inner; k += kTile)
            for (int n = 0; n < columns; n += kTile)
                (void)kernel(padded(std::min(kTile, rows - m)), padded(std::min(kTile, inner - k)),
                    padded(std::min(kTile, columns - n)));
}

array MlxAneMatmul::operator()(const array& left, const array& right) {
    if (left.ndim() != 2 || right.ndim() != 2 || left.shape(1) != right.shape(0))
        throw std::invalid_argument("ANE training expects compatible rank-2 matrices");
    const auto stream = cpu();
    return array(Shape{left.shape(0), right.shape(1)}, float32,
        std::make_shared<AneMatmulPrimitive>(stream, shared_from_this()),
        {astype(left, float32, stream), astype(right, float32, stream)});
}

std::vector<float> MlxAneMatmul::evaluate(int rows, int inner, int columns,
    std::span<const float> left, std::span<const float> right) {
    if (left.size() != std::size_t(rows) * inner || right.size() != std::size_t(inner) * columns)
        throw std::invalid_argument("ANE matrix payload shape mismatch");
    return evaluate_strided(rows, inner, columns, left.data(), right.data(), inner, 1, columns, 1);
}

std::vector<float> MlxAneMatmul::evaluate(const array& left, const array& right) {
    if (left.ndim() != 2 || right.ndim() != 2 || left.shape(1) != right.shape(0) ||
        left.dtype() != float32 || right.dtype() != float32)
        throw std::invalid_argument("ANE matrix view shape mismatch");
    return evaluate_strided(left.shape(0), left.shape(1), right.shape(1), left.data<float>(), right.data<float>(),
        left.strides()[0], left.strides()[1], right.strides()[0], right.strides()[1]);
}

std::vector<float> MlxAneMatmul::evaluate_strided(int rows, int inner, int columns,
    const float* left, const float* right, std::size_t ls0, std::size_t ls1, std::size_t rs0, std::size_t rs1) {
    std::vector<float> result(std::size_t(rows) * columns);
    for (int m = 0; m < rows; m += kTile) for (int n = 0; n < columns; n += kTile) {
        const int height = std::min(kTile, rows - m), width = std::min(kTile, columns - n);
        for (int k = 0; k < inner; k += kTile) {
            const int depth = std::min(kTile, inner - k);
            const int pm = padded(height), pk = padded(depth), pn = padded(width);
            std::vector<float> a(std::size_t(pm) * pk), b(std::size_t(pk) * pn);
            for (int row = 0; row < height; ++row) {
                if (ls1 == 1) std::copy_n(left + (m + row) * ls0 + k, depth, a.data() + row * pk);
                else for (int column = 0; column < depth; ++column)
                    a[row * pk + column] = left[(m + row) * ls0 + (k + column) * ls1];
            }
            for (int r = 0; r < depth; r += 32) for (int c = 0; c < width; c += 32) {
                const int count = std::min(32, width - c), end = std::min(r + 32, depth);
                if (rs1 == 1) {
                    for (int row = r; row < end; ++row)
                        std::copy_n(right + (k + row) * rs0 + n + c, count, b.data() + row * pn + c);
                } else {
                    for (int column = c; column < c + count; ++column) for (int row = r; row < end; ++row)
                        b[row * pn + column] = right[(k + row) * rs0 + (n + column) * rs1];
                }
            }
            const auto part = (*kernel(pm, pk, pn))(a, b);
            for (int row = 0; row < height; ++row) for (int column = 0; column < width; ++column)
                result[(m + row) * columns + n + column] += part[row * pn + column];
        }
    }
    return result;
}

std::size_t MlxAneMatmul::bytes() const {
    std::lock_guard lock(mutex_);
    std::size_t size = 0;
    for (const auto& [key, value] : kernels_) if (value) size += value->bytes();
    return size;
}

} // namespace mfq::metal
