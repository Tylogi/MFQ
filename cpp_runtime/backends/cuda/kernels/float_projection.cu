#include "../ops/include/float_projection.h"
#include "../ops/include/nint.h"

#include <limits>

namespace {
__global__ void affine_projection(
    const float* input, const unsigned char* values,
    const unsigned char* row_bits, const int64_t* row_offsets,
    const unsigned char* scales, const unsigned char* minima,
    const float* row_scales, const float* row_minima, const __half* zero_scales,
    float* output, int64_t width, int64_t outputs, int64_t groups, int group_size, bool zero) {
    const int lane = threadIdx.x & 31;
    const int64_t neuron = int64_t(blockIdx.x) * 4 + (threadIdx.x >> 5);
    if (neuron >= outputs) return;
    const int64_t sample = blockIdx.y;
    const int bits = zero ? 8 : row_bits[neuron];
    float sum = 0;
    for (int64_t group = 0; group < groups; ++group) {
        const int64_t meta = neuron * groups + group;
        const float scale = zero ? __half2float(zero_scales[meta])
            : row_scales[neuron] * float(scales[meta]);
        const float minimum = zero ? 0 : row_minima[neuron] * float(minima[meta]);
        for (int j = lane; j < group_size; j += 32) {
            const int64_t column = group * group_size + j;
            if (column >= width) continue;
            float q;
            if (zero) {
                q = float(static_cast<signed char>(values[meta * group_size + j]));
            } else {
                const uint64_t bit = uint64_t(row_offsets[neuron]) + uint64_t(column) * bits;
                const int shift = int(bit & 7);
                unsigned packed = values[bit >> 3];
                if (shift + bits > 8) packed |= unsigned(values[(bit >> 3) + 1]) << 8;
                q = float((packed >> shift) & ((1u << bits) - 1));
            }
            // Separate affine rounding matches the canonical FP32 decoder.
            const float weight = __fsub_rn(__fmul_rn(scale, q), minimum);
            sum = fmaf(weight, input[sample * width + column], sum);
        }
    }
    for (int step = 16; step; step >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, step);
    if (!lane) output[sample * outputs + neuron] = sum;
}
} // namespace

mfq_tensor_backend::Tensor nint_float_projection_cuda(
        const NintWeight& weight, const mfq_tensor_backend::Tensor& input) {
    namespace tb = mfq_tensor_backend;
    MFQ_RUNTIME_CHECK(input.is_cuda() && weight.q_packed.is_cuda() &&
        input.device() == weight.q_packed.device() && input.dim() >= 1 &&
        input.size(-1) == weight.neuron_len && weight.neuron_len > 0 && weight.out > 0 &&
        weight.gs > 0 && weight.gs <= 64 &&
        weight.ng == (weight.neuron_len + weight.gs - 1) / weight.gs,
        "packed float projection dimensions/device disagree");
    MfqCudaGuard guard(input.device());
    auto x = input.to(tb::kFloat32).contiguous();
    auto shape = x.sizes().vec();
    const auto rows = x.numel() / weight.neuron_len;
    MFQ_RUNTIME_CHECK(rows <= 65535 && (weight.out + 3) / 4 <= std::numeric_limits<int>::max(),
        "packed float projection exceeds CUDA launch geometry");
    shape.back() = weight.out;
    auto result = tb::empty(shape, x.options());
    if (!rows) return result;
    affine_projection<<<dim3(unsigned((weight.out + 3) / 4), unsigned(rows)), 128, 0,
        mfq_current_cuda_stream()>>>(x.data_ptr<float>(), weight.q_packed.data_ptr<uint8_t>(),
        weight.q8_zero ? nullptr : weight.row_q_bits.data_ptr<uint8_t>(),
        weight.q8_zero ? nullptr : weight.row_q_bit_offsets.data_ptr<int64_t>(),
        weight.q8_zero ? nullptr : weight.sub_scale.data_ptr<uint8_t>(),
        weight.q8_zero ? nullptr : weight.sub_min.data_ptr<uint8_t>(),
        weight.q8_zero ? nullptr : weight.neuron_scale.data_ptr<float>(),
        weight.q8_zero ? nullptr : weight.neuron_min.data_ptr<float>(),
        weight.q8_zero ? weight.q8_zero_scale.data_ptr<__half>() : nullptr,
        result.data_ptr<float>(), weight.neuron_len, weight.out, weight.ng, int(weight.gs),
        weight.q8_zero);
    MFQ_CUDA_CHECK(cudaGetLastError());
    return result;
}
