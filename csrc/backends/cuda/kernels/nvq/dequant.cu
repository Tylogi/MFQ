// Inference-side NVQ decode. It lives apart from backward.cu so the native runtime
// build compiles dequant without the tool-only backward kernels.
#include "kernels/quant_ops.h"
#define MFQ_NVQ_DEQUANT 1
#include "matmul_impl.cuh"

mfq_tensor_backend::Tensor nvq_dequant_cuda(
    mfq_tensor_backend::Tensor indices,
    mfq_tensor_backend::Tensor aux,
    mfq_tensor_backend::Tensor sub_scale,
    mfq_tensor_backend::Tensor neuron_scale,
    mfq_tensor_backend::Tensor codebook,
    int64_t neuron_len,
    int64_t gs,
    int64_t sub_bits,
    int64_t format,
    int64_t sign_mode) {
    check_common(indices, aux, sub_scale, neuron_scale, codebook,
                 neuron_len, gs, sub_bits, format, sign_mode);
    const int N = static_cast<int>(neuron_scale.numel());
    const int K = static_cast<int>(neuron_len);
    const int ng = (K + static_cast<int>(gs) - 1) / static_cast<int>(gs);
    const int nvec = (K + (is_d4_format(format) ? 3 : 7)) /
                     (is_d4_format(format) ? 4 : 8);
    const int nsign = (K + 7) / 8;
    auto output = mfq_tensor_backend::empty({N, K}, neuron_scale.options().dtype(mfq_tensor_backend::kFloat16));
    const int block = 128;
    const dim3 grid((nsign + block - 1) / block, std::min(N, 65535));
    cudaStream_t stream = mfq_current_cuda_stream();
    launch_by_format(static_cast<int>(format), [&](auto tag) {
        constexpr int F = decltype(tag)::value;
        nvq_dequant_kernel<F><<<grid, block, 0, stream>>>(
            indices.data_ptr<uint8_t>(), indices.numel(),
            aux.data_ptr<uint8_t>(), aux.numel(),
            sub_scale.data_ptr<uint8_t>(), sub_scale.numel(),
            neuron_scale.data_ptr<float>(), codebook.data_ptr<int8_t>(),
            reinterpret_cast<__half *>(output.data_ptr<mfq_half>()),
            N, K, ng, nvec, nsign, static_cast<int>(sub_bits), static_cast<int>(sign_mode));
    });
    MFQ_RUNTIME_CHECK(cudaGetLastError() == cudaSuccess, "NVQ dequant kernel launch failed");
    return output;
}
