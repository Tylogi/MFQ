#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

namespace {
template <int Bits>
__global__ void pack_bits(const int32_t* values, uint8_t* packed,
                          int64_t count, int64_t bytes) {
    const int64_t byte = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (byte >= bytes) return;
    const int64_t bit = byte * 8;
    const int64_t index = bit / Bits;
    const int shift = bit % Bits;
    uint32_t result = uint32_t(values[index]) >> shift;
    #pragma unroll
    for (int local = 1; local <= (7 + Bits - 1) / Bits; ++local) {
        const int left = local * Bits - shift;
        if (left < 8 && index + local < count) {
            result |= uint32_t(values[index + local]) << left;
        }
    }
    packed[byte] = uint8_t(result);
}

__global__ void pack_group64(const int32_t* state, const int32_t* indices,
                              const int32_t* signs, uint64_t* packed,
                              int64_t count, int64_t groups, int64_t vectors) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int64_t row = i / groups;
    const int64_t first = (i % groups) * 3;
    uint64_t record = uint64_t(state[i]) << 60;
    #pragma unroll
    for (int local = 0; local < 3; ++local) {
        if (first + local < vectors) {
            const int64_t offset = row * vectors + first + local;
            const uint32_t sign = signs[offset];
            const uint32_t sign8 = sign | ((__popc(sign) & 1) << 7);
            record |= (uint64_t(indices[offset]) | (uint64_t(sign8) << 12))
                      << (local * 20);
        }
    }
    packed[i] = record;
}

void check_values(const torch::Tensor& value) {
    TORCH_CHECK(value.is_cuda() && value.is_contiguous()
                && value.scalar_type() == torch::kInt32,
                "NVQ packing requires contiguous CUDA int32 values");
}
}  // namespace

torch::Tensor nvq_pack_bits_cuda(torch::Tensor values, int64_t bits) {
    check_values(values);
    TORCH_CHECK(bits >= 1 && bits <= 16, "NVQ packed width must be 1..16");
    const c10::cuda::CUDAGuard guard(values.device());
    const auto count = values.numel();
    const auto bytes = (count * bits + 7) / 8;
    auto output = torch::empty({bytes}, values.options().dtype(torch::kUInt8));
    if (!bytes) return output;
    const auto stream = at::cuda::getCurrentCUDAStream();
    const auto blocks = (bytes + 255) / 256;
    #define PACK_CASE(B) case B: pack_bits<B><<<blocks, 256, 0, stream>>>( \
        values.data_ptr<int32_t>(), output.data_ptr<uint8_t>(), count, bytes); break
    switch (bits) {
        PACK_CASE(1); PACK_CASE(2); PACK_CASE(3); PACK_CASE(4);
        PACK_CASE(5); PACK_CASE(6); PACK_CASE(7); PACK_CASE(8);
        PACK_CASE(9); PACK_CASE(10); PACK_CASE(11); PACK_CASE(12);
        PACK_CASE(13); PACK_CASE(14); PACK_CASE(15); PACK_CASE(16);
    }
    #undef PACK_CASE
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor nvq_pack_group64_cuda(torch::Tensor state, torch::Tensor indices,
                                   torch::Tensor signs, int64_t neuron_len) {
    check_values(state);
    check_values(indices);
    check_values(signs);
    TORCH_CHECK(neuron_len > 0, "NVQ neuron_len must be positive");
    const auto groups = (neuron_len + 23) / 24;
    const auto vectors = (neuron_len + 7) / 8;
    TORCH_CHECK(state.dim() == 2 && state.size(1) == groups,
                "NVQ group64 state shape mismatch");
    TORCH_CHECK(indices.dim() == 2 && indices.size(0) == state.size(0)
                && indices.size(1) == vectors && signs.sizes() == indices.sizes(),
                "NVQ group64 index/sign shape mismatch");
    TORCH_CHECK(indices.device() == state.device() && signs.device() == state.device(),
                "NVQ group64 tensors must share one CUDA device");
    const c10::cuda::CUDAGuard guard(state.device());
    const auto count = state.numel();
    auto output = torch::empty({count * 8}, state.options().dtype(torch::kUInt8));
    if (!count) return output;
    pack_group64<<<(count + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        state.data_ptr<int32_t>(), indices.data_ptr<int32_t>(), signs.data_ptr<int32_t>(),
        reinterpret_cast<uint64_t*>(output.data_ptr<uint8_t>()), count, groups, vectors);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
