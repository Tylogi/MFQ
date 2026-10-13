#define MFQ_NVQ_DENSE 1
#include "native/env.h"
#include "nvq_matmul_impl.cuh"

mfq_tensor_backend::Tensor nvq_gemm_f16_cuda(
    mfq_tensor_backend::Tensor indices,
    mfq_tensor_backend::Tensor aux,
    mfq_tensor_backend::Tensor sub_scale,
    mfq_tensor_backend::Tensor neuron_scale,
    mfq_tensor_backend::Tensor codebook,
    mfq_tensor_backend::Tensor x,
    int64_t neuron_len,
    int64_t gs,
    int64_t sub_bits,
    int64_t format,
    int64_t sign_mode) {
    check_common(indices, aux, sub_scale, neuron_scale, codebook,
                 neuron_len, gs, sub_bits, format, sign_mode);
    MFQ_RUNTIME_CHECK(x.is_cuda() && x.scalar_type() == mfq_tensor_backend::kFloat16 &&
                x.is_contiguous() && x.dim() == 2,
                "NVQ FP16 GEMM x must be CUDA contiguous fp16 rank-2");
    const int M = static_cast<int>(x.size(0));
    const int K = static_cast<int>(neuron_len);
    const int N = static_cast<int>(neuron_scale.numel());
    const int ng = (K + static_cast<int>(gs) - 1) / static_cast<int>(gs);
    MFQ_RUNTIME_CHECK(M >= 16, "NVQ FP16 GEMM requires M >= 16");
    MFQ_RUNTIME_CHECK(x.size(1) == K, "NVQ FP16 GEMM x width must equal neuron_len");
    const int nvec = (K + (is_d4_format(format) ? 3 : 7)) /
                     (is_d4_format(format) ? 4 : 8);
    const int nsign = (K + 7) / 8;
    auto output = mfq_tensor_backend::empty({M, N}, x.options());
    cudaStream_t stream = mfq_current_cuda_stream();
    const dim3 block(32, 8);

#define NVQ_F16_GEMM_LAUNCH(MTILES_VALUE, NEED_CHECK_VALUE)                       \
    nvq_gemm_f16_gs24_kernel<F, MTILES_VALUE, NEED_CHECK_VALUE><<<                \
        dim3((N + 63) / 64, (M + 16 * MTILES_VALUE - 1) /                        \
                               (16 * MTILES_VALUE)),                              \
        block, 0, stream>>>(                                                       \
        indices.data_ptr<uint8_t>(), indices.numel(),                             \
        aux.data_ptr<uint8_t>(), aux.numel(),                                     \
        sub_scale.data_ptr<uint8_t>(), sub_scale.numel(),                         \
        neuron_scale.data_ptr<float>(), codebook.data_ptr<int8_t>(),              \
        reinterpret_cast<const __half *>(x.data_ptr<mfq_half>()),                 \
        reinterpret_cast<__half *>(output.data_ptr<mfq_half>()),                  \
        M, N, K, ng, nvec, nsign, static_cast<int>(sub_bits),                     \
        static_cast<int>(sign_mode))

    launch_by_format(static_cast<int>(format), [&](auto tag) {
        constexpr int F = decltype(tag)::value;
        if (M <= 16) {
            if (M == 16 && N % 64 == 0) NVQ_F16_GEMM_LAUNCH(1, false);
            else NVQ_F16_GEMM_LAUNCH(1, true);
        } else if (M <= 32) {
            if (M == 32 && N % 64 == 0) NVQ_F16_GEMM_LAUNCH(2, false);
            else NVQ_F16_GEMM_LAUNCH(2, true);
        } else if (M <= 48) {
            if (M == 48 && N % 64 == 0) NVQ_F16_GEMM_LAUNCH(3, false);
            else NVQ_F16_GEMM_LAUNCH(3, true);
        } else if (M <= 64) {
            if (M == 64 && N % 64 == 0) NVQ_F16_GEMM_LAUNCH(4, false);
            else NVQ_F16_GEMM_LAUNCH(4, true);
        } else if (M <= 80) {
            if (M == 80 && N % 64 == 0) NVQ_F16_GEMM_LAUNCH(5, false);
            else NVQ_F16_GEMM_LAUNCH(5, true);
        } else if (M <= 96) {
            if (M == 96 && N % 64 == 0) NVQ_F16_GEMM_LAUNCH(6, false);
            else NVQ_F16_GEMM_LAUNCH(6, true);
        } else if (M <= 112) {
            if (M == 112 && N % 64 == 0) NVQ_F16_GEMM_LAUNCH(7, false);
            else NVQ_F16_GEMM_LAUNCH(7, true);
        } else {
            if (M % 128 == 0 && N % 64 == 0) NVQ_F16_GEMM_LAUNCH(8, false);
            else NVQ_F16_GEMM_LAUNCH(8, true);
        }
    });
#undef NVQ_F16_GEMM_LAUNCH
    MFQ_RUNTIME_CHECK(cudaGetLastError() == cudaSuccess, "NVQ FP16 GEMM kernel launch failed");
    return output;
}

mfq_tensor_backend::Tensor nvq_gemv_m1_vec8_ws_cuda(
    mfq_tensor_backend::Tensor indices,
    mfq_tensor_backend::Tensor aux,
    mfq_tensor_backend::Tensor sub_scale,
    mfq_tensor_backend::Tensor neuron_scale,
    mfq_tensor_backend::Tensor codebook,
    mfq_tensor_backend::Tensor x,
    int64_t neuron_len,
    int64_t gs,
    int64_t sub_bits,
    int64_t format,
    int64_t sign_mode,
    int64_t nwarps,
    mfq_tensor_backend::Tensor qx,
    mfq_tensor_backend::Tensor xscale) {
    check_common(indices, aux, sub_scale, neuron_scale, codebook,
                 neuron_len, gs, sub_bits, format, sign_mode);
    MFQ_RUNTIME_CHECK(x.is_cuda() && x.scalar_type() == mfq_tensor_backend::kFloat16 && x.is_contiguous() &&
                x.dim() == 2 && x.size(0) == 1, "NVQ vec8 GEMV x must be CUDA contiguous fp16 [1,K]");
    MFQ_RUNTIME_CHECK(nwarps == 1 || nwarps == 2 || nwarps == 4 || nwarps == 8,
                "NVQ vec8 GEMV nwarps must be 1, 2, 4, or 8");
    const int K = static_cast<int>(neuron_len);
    const int N = static_cast<int>(neuron_scale.numel());
    const int ng = (K + static_cast<int>(gs) - 1) / static_cast<int>(gs);
    const int K_pad = ng * static_cast<int>(gs);
    MFQ_RUNTIME_CHECK(x.size(1) == K, "NVQ vec8 GEMV x width must equal neuron_len");
    MFQ_RUNTIME_CHECK(qx.is_cuda() && qx.scalar_type() == mfq_tensor_backend::kInt8 && qx.is_contiguous() &&
                qx.numel() >= K_pad, "NVQ vec8 GEMV qx workspace mismatch");
    MFQ_RUNTIME_CHECK(xscale.is_cuda() && xscale.scalar_type() == mfq_tensor_backend::kFloat32 && xscale.is_contiguous() &&
                xscale.numel() >= ng, "NVQ vec8 GEMV xscale workspace mismatch");
    const int nvec = (K + (is_d4_format(format) ? 3 : 7)) /
                     (is_d4_format(format) ? 4 : 8);
    const int nsign = (K + 7) / 8;
    auto output = mfq_tensor_backend::empty({1, N}, x.options());
    cudaStream_t stream = mfq_current_cuda_stream();
    nvq_quantize_x_gs24_kernel<<<dim3(1, ng), 32, 0, stream>>>(
        reinterpret_cast<const __half *>(x.data_ptr<mfq_half>()), qx.data_ptr<int8_t>(),
        xscale.data_ptr<float>(), 1, K, ng);
    switch (static_cast<int>(nwarps)) {
        case 1:
            launch_m1_vec8_by_format<1>(format, indices, aux, sub_scale, neuron_scale, codebook,
                                        qx, xscale, output, N, ng, nvec, nsign, sub_bits, sign_mode, stream);
            break;
        case 2:
            launch_m1_vec8_by_format<2>(format, indices, aux, sub_scale, neuron_scale, codebook,
                                        qx, xscale, output, N, ng, nvec, nsign, sub_bits, sign_mode, stream);
            break;
        case 4:
            launch_m1_vec8_by_format<4>(format, indices, aux, sub_scale, neuron_scale, codebook,
                                        qx, xscale, output, N, ng, nvec, nsign, sub_bits, sign_mode, stream);
            break;
        case 8:
            launch_m1_vec8_by_format<8>(format, indices, aux, sub_scale, neuron_scale, codebook,
                                        qx, xscale, output, N, ng, nvec, nsign, sub_bits, sign_mode, stream);
            break;
    }
    MFQ_RUNTIME_CHECK(cudaGetLastError() == cudaSuccess, "NVQ vec8 GEMV kernel launch failed");
    return output;
}

mfq_tensor_backend::Tensor nvq_gemv_batch_vec8_ws_cuda(
    mfq_tensor_backend::Tensor indices,
    mfq_tensor_backend::Tensor aux,
    mfq_tensor_backend::Tensor sub_scale,
    mfq_tensor_backend::Tensor neuron_scale,
    mfq_tensor_backend::Tensor codebook,
    mfq_tensor_backend::Tensor x,
    int64_t neuron_len,
    int64_t gs,
    int64_t sub_bits,
    int64_t format,
    int64_t sign_mode,
    int64_t nwarps,
    mfq_tensor_backend::Tensor qx,
    mfq_tensor_backend::Tensor xscale) {
    check_common(indices, aux, sub_scale, neuron_scale, codebook,
                 neuron_len, gs, sub_bits, format, sign_mode);
    MFQ_RUNTIME_CHECK(x.is_cuda() && x.scalar_type() == mfq_tensor_backend::kFloat16 && x.is_contiguous() && x.dim() == 2,
                "NVQ batch vec8 GEMV x must be CUDA contiguous fp16 rank-2");
    MFQ_RUNTIME_CHECK(nwarps == 2 || nwarps == 4 || nwarps == 8,
                "NVQ batch vec8 GEMV nwarps must be 2, 4, or 8");
    const int M = static_cast<int>(x.size(0));
    const int K = static_cast<int>(neuron_len);
    const int N = static_cast<int>(neuron_scale.numel());
    const int ng = (K + static_cast<int>(gs) - 1) / static_cast<int>(gs);
    const int K_pad = ng * static_cast<int>(gs);
    MFQ_RUNTIME_CHECK(M >= 2 && M <= 16, "NVQ batch vec8 GEMV supports M in [2,16]");
    MFQ_RUNTIME_CHECK(x.size(1) == K, "NVQ batch vec8 GEMV x width must equal neuron_len");
    MFQ_RUNTIME_CHECK(qx.is_cuda() && qx.scalar_type() == mfq_tensor_backend::kInt8 && qx.is_contiguous() &&
                qx.numel() >= static_cast<int64_t>(M) * K_pad, "NVQ batch vec8 GEMV qx workspace mismatch");
    MFQ_RUNTIME_CHECK(xscale.is_cuda() && xscale.scalar_type() == mfq_tensor_backend::kFloat32 && xscale.is_contiguous() &&
                xscale.numel() >= static_cast<int64_t>(M) * ng, "NVQ batch vec8 GEMV xscale workspace mismatch");
    const int nvec = (K + (is_d4_format(format) ? 3 : 7)) /
                     (is_d4_format(format) ? 4 : 8);
    const int nsign = (K + 7) / 8;
    auto output = mfq_tensor_backend::empty({M, N}, x.options());
    cudaStream_t stream = mfq_current_cuda_stream();
    nvq_quantize_x_gs24_kernel<<<dim3(M, ng), 32, 0, stream>>>(
        reinterpret_cast<const __half *>(x.data_ptr<mfq_half>()), qx.data_ptr<int8_t>(),
        xscale.data_ptr<float>(), M, K, ng);
#define NVQ_LAUNCH_BATCH_VEC8(NW)                                                       \
    do {                                                                                \
        if (M <= 2) {                                                                   \
            launch_batch_vec8_by_format<NW, 2>(                                        \
                format, indices, aux, sub_scale, neuron_scale, codebook, qx, xscale,   \
                output, M, N, ng, nvec, nsign, sub_bits, sign_mode, stream);           \
        } else if (M <= 4) {                                                            \
            launch_batch_vec8_by_format<NW, 4>(                                        \
                format, indices, aux, sub_scale, neuron_scale, codebook, qx, xscale,   \
                output, M, N, ng, nvec, nsign, sub_bits, sign_mode, stream);           \
        } else if (M <= 8) {                                                            \
            launch_batch_vec8_by_format<NW, 8>(                                        \
                format, indices, aux, sub_scale, neuron_scale, codebook, qx, xscale,   \
                output, M, N, ng, nvec, nsign, sub_bits, sign_mode, stream);           \
        } else if (M <= 12) {                                                           \
            launch_batch_vec8_by_format<NW, 12>(                                       \
                format, indices, aux, sub_scale, neuron_scale, codebook, qx, xscale,   \
                output, M, N, ng, nvec, nsign, sub_bits, sign_mode, stream);           \
        } else {                                                                        \
            launch_batch_vec8_by_format<NW, 16>(                                       \
                format, indices, aux, sub_scale, neuron_scale, codebook, qx, xscale,   \
                output, M, N, ng, nvec, nsign, sub_bits, sign_mode, stream);           \
        }                                                                               \
    } while (0)
    switch (static_cast<int>(nwarps)) {
        case 2: NVQ_LAUNCH_BATCH_VEC8(2); break;
        case 4: NVQ_LAUNCH_BATCH_VEC8(4); break;
        case 8: NVQ_LAUNCH_BATCH_VEC8(8); break;
    }
#undef NVQ_LAUNCH_BATCH_VEC8
    MFQ_RUNTIME_CHECK(cudaGetLastError() == cudaSuccess, "NVQ batch vec8 GEMV kernel launch failed");
    return output;
}

mfq_tensor_backend::Tensor nvq_gemv_ws_cuda(
    mfq_tensor_backend::Tensor indices,
    mfq_tensor_backend::Tensor aux,
    mfq_tensor_backend::Tensor sub_scale,
    mfq_tensor_backend::Tensor neuron_scale,
    mfq_tensor_backend::Tensor codebook,
    mfq_tensor_backend::Tensor x,
    int64_t neuron_len,
    int64_t gs,
    int64_t sub_bits,
    int64_t format,
    int64_t sign_mode,
    mfq_tensor_backend::Tensor qx,
    mfq_tensor_backend::Tensor xscale) {
    check_common(indices, aux, sub_scale, neuron_scale, codebook,
                 neuron_len, gs, sub_bits, format, sign_mode);
    MFQ_RUNTIME_CHECK(x.is_cuda() && x.scalar_type() == mfq_tensor_backend::kFloat16 && x.is_contiguous() && x.dim() == 2,
                "NVQ x must be CUDA contiguous fp16 rank-2");
    const int M = static_cast<int>(x.size(0));
    const int K = static_cast<int>(neuron_len);
    const int N = static_cast<int>(neuron_scale.numel());
    const int ng = (K + static_cast<int>(gs) - 1) / static_cast<int>(gs);
    const int K_pad = ng * static_cast<int>(gs);
    MFQ_RUNTIME_CHECK(x.size(1) == K, "NVQ x width must equal neuron_len");
    MFQ_RUNTIME_CHECK(M >= 1 && M <= 16, "NVQ direct GEMV supports M in [1,16]");
    MFQ_RUNTIME_CHECK(qx.is_cuda() && qx.scalar_type() == mfq_tensor_backend::kInt8 && qx.is_contiguous() &&
                qx.numel() >= static_cast<int64_t>(M) * K_pad, "NVQ qx workspace mismatch");
    MFQ_RUNTIME_CHECK(xscale.is_cuda() && xscale.scalar_type() == mfq_tensor_backend::kFloat32 && xscale.is_contiguous() &&
                xscale.numel() >= static_cast<int64_t>(M) * ng, "NVQ xscale workspace mismatch");
    const int nvec = (K + (is_d4_format(format) ? 3 : 7)) /
                     (is_d4_format(format) ? 4 : 8);
    const int nsign = (K + 7) / 8;
    auto output = mfq_tensor_backend::empty({M, N}, x.options());
    cudaStream_t stream = mfq_current_cuda_stream();
    nvq_quantize_x_gs24_kernel<<<dim3(M, ng), 32, 0, stream>>>(
        reinterpret_cast<const __half *>(x.data_ptr<mfq_half>()), qx.data_ptr<int8_t>(),
        xscale.data_ptr<float>(), M, K, ng);
    if (M == 1) {
        launch_selected_m1_vec8(
            static_cast<int>(format), indices, aux, sub_scale, neuron_scale, codebook,
            qx, xscale, output, N, ng, nvec, nsign,
            static_cast<int>(sub_bits), static_cast<int>(sign_mode), stream);
        MFQ_RUNTIME_CHECK(cudaGetLastError() == cudaSuccess, "NVQ vec8 GEMV kernel launch failed");
        return output;
    }
    launch_selected_batch_vec8(
        static_cast<int>(format), indices, aux, sub_scale, neuron_scale, codebook,
        qx, xscale, output, M, N, ng, nvec, nsign,
        static_cast<int>(sub_bits), static_cast<int>(sign_mode), stream);
    MFQ_RUNTIME_CHECK(cudaGetLastError() == cudaSuccess, "NVQ batch vec8 GEMV kernel launch failed");
    return output;
}

mfq_tensor_backend::Tensor nvq_gemv_qx_ws_cuda(
    mfq_tensor_backend::Tensor indices,
    mfq_tensor_backend::Tensor aux,
    mfq_tensor_backend::Tensor sub_scale,
    mfq_tensor_backend::Tensor neuron_scale,
    mfq_tensor_backend::Tensor codebook,
    int64_t neuron_len,
    int64_t gs,
    int64_t sub_bits,
    int64_t format,
    int64_t sign_mode,
    mfq_tensor_backend::Tensor qx,
    mfq_tensor_backend::Tensor xscale) {
    check_common(indices, aux, sub_scale, neuron_scale, codebook,
                 neuron_len, gs, sub_bits, format, sign_mode);
    MFQ_RUNTIME_CHECK(qx.is_cuda() && qx.scalar_type() == mfq_tensor_backend::kInt8 && qx.is_contiguous() && qx.dim() == 2,
                "NVQ prequantized qx must be CUDA contiguous int8 rank-2");
    MFQ_RUNTIME_CHECK(xscale.is_cuda() && xscale.scalar_type() == mfq_tensor_backend::kFloat32 && xscale.is_contiguous() &&
                xscale.dim() == 2, "NVQ prequantized xscale must be CUDA contiguous fp32 rank-2");
    const int M = static_cast<int>(qx.size(0));
    const int K = static_cast<int>(neuron_len);
    const int N = static_cast<int>(neuron_scale.numel());
    const int ng = (K + static_cast<int>(gs) - 1) / static_cast<int>(gs);
    const int K_pad = ng * static_cast<int>(gs);
    MFQ_RUNTIME_CHECK(M >= 1 && M <= 16, "NVQ prequantized GEMV supports M in [1,16]");
    MFQ_RUNTIME_CHECK(qx.size(1) >= K_pad, "NVQ prequantized qx width mismatch");
    MFQ_RUNTIME_CHECK(xscale.size(0) >= M && xscale.size(1) >= ng,
                "NVQ prequantized xscale shape mismatch");
    const int nvec = (K + (is_d4_format(format) ? 3 : 7)) /
                     (is_d4_format(format) ? 4 : 8);
    const int nsign = (K + 7) / 8;
    auto output = mfq_tensor_backend::empty(
        {M, N}, neuron_scale.options().dtype(mfq_tensor_backend::kFloat16));
    cudaStream_t stream = mfq_current_cuda_stream();
    if (M == 1) {
        launch_selected_m1_vec8(
            static_cast<int>(format), indices, aux, sub_scale, neuron_scale, codebook,
            qx, xscale, output, N, ng, nvec, nsign,
            static_cast<int>(sub_bits), static_cast<int>(sign_mode), stream);
    } else {
        launch_selected_batch_vec8(
            static_cast<int>(format), indices, aux, sub_scale, neuron_scale, codebook,
            qx, xscale, output, M, N, ng, nvec, nsign,
            static_cast<int>(sub_bits), static_cast<int>(sign_mode), stream);
    }
    MFQ_RUNTIME_CHECK(cudaGetLastError() == cudaSuccess, "NVQ prequantized GEMV kernel launch failed");
    return output;
}

mfq_tensor_backend::Tensor nvq_gemv_qx_residual_ws_cuda(
    mfq_tensor_backend::Tensor indices,
    mfq_tensor_backend::Tensor aux,
    mfq_tensor_backend::Tensor sub_scale,
    mfq_tensor_backend::Tensor neuron_scale,
    mfq_tensor_backend::Tensor codebook,
    int64_t neuron_len,
    int64_t gs,
    int64_t sub_bits,
    int64_t format,
    int64_t sign_mode,
    mfq_tensor_backend::Tensor qx,
    mfq_tensor_backend::Tensor xscale,
    mfq_tensor_backend::Tensor residual) {
    check_common(indices, aux, sub_scale, neuron_scale, codebook,
                 neuron_len, gs, sub_bits, format, sign_mode);
    MFQ_RUNTIME_CHECK(
        format == kNvq2JscXLGroupExec || format == kNvq3JscLGroupExec,
        "NVQ fused residual requires an aligned group execution format");
    MFQ_RUNTIME_CHECK(
        qx.is_cuda() && qx.scalar_type() == mfq_tensor_backend::kInt8 &&
        qx.is_contiguous() && qx.dim() == 2 && qx.size(0) == 1,
        "NVQ fused residual qx must be CUDA contiguous int8 [1,K]");
    MFQ_RUNTIME_CHECK(
        xscale.is_cuda() && xscale.scalar_type() == mfq_tensor_backend::kFloat32 &&
        xscale.is_contiguous() && xscale.dim() == 2 && xscale.size(0) >= 1,
        "NVQ fused residual xscale must be CUDA contiguous fp32 [1,ng]");
    const int K = static_cast<int>(neuron_len);
    const int N = static_cast<int>(neuron_scale.numel());
    const int ng = (K + static_cast<int>(gs) - 1) / static_cast<int>(gs);
    const int K_pad = ng * static_cast<int>(gs);
    MFQ_RUNTIME_CHECK(qx.size(1) >= K_pad, "NVQ fused residual qx width mismatch");
    MFQ_RUNTIME_CHECK(xscale.size(1) >= ng, "NVQ fused residual xscale width mismatch");
    MFQ_RUNTIME_CHECK(
        residual.is_cuda() && residual.scalar_type() == mfq_tensor_backend::kFloat16 &&
        residual.is_contiguous() && residual.dim() == 2 &&
        residual.size(0) == 1 && residual.size(1) == N,
        "NVQ fused residual must be CUDA contiguous fp16 [1,N]");

    const auto weight = make_device_weight(
        indices, aux, sub_scale, neuron_scale, codebook,
        K, static_cast<int>(gs), static_cast<int>(sub_bits),
        static_cast<int>(format), static_cast<int>(sign_mode));
    auto output = mfq_tensor_backend::empty_like(residual);
    cudaStream_t stream = mfq_current_cuda_stream();
    const auto * residual_ptr = reinterpret_cast<const __half *>(
        residual.data_ptr<mfq_half>());
    auto * output_ptr = reinterpret_cast<__half *>(
        output.data_ptr<mfq_half>());
    if (format == kNvq2JscXLGroupExec) {
        launch_aligned_group_m1<kNvq2JscXLGroupExec, true>(
            weight, qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),
            output_ptr, stream, residual_ptr);
    } else {
        launch_aligned_group_m1<kNvq3JscLGroupExec, true>(
            weight, qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),
            output_ptr, stream, residual_ptr);
    }
    MFQ_RUNTIME_CHECK(
        cudaGetLastError() == cudaSuccess,
        "NVQ fused residual GEMV kernel launch failed");
    return output;
}

mfq_tensor_backend::Tensor nvq_gemv_multi2_ws_cuda(
    mfq_tensor_backend::Tensor first_indices,
    mfq_tensor_backend::Tensor first_aux,
    mfq_tensor_backend::Tensor first_sub_scale,
    mfq_tensor_backend::Tensor first_neuron_scale,
    mfq_tensor_backend::Tensor first_codebook,
    mfq_tensor_backend::Tensor second_indices,
    mfq_tensor_backend::Tensor second_aux,
    mfq_tensor_backend::Tensor second_sub_scale,
    mfq_tensor_backend::Tensor second_neuron_scale,
    mfq_tensor_backend::Tensor second_codebook,
    mfq_tensor_backend::Tensor x,
    int64_t neuron_len,
    int64_t gs,
    int64_t first_sub_bits,
    int64_t first_format,
    int64_t first_sign_mode,
    int64_t second_sub_bits,
    int64_t second_format,
    int64_t second_sign_mode,
    mfq_tensor_backend::Tensor qx,
    mfq_tensor_backend::Tensor xscale) {
    check_common(first_indices, first_aux, first_sub_scale, first_neuron_scale, first_codebook,
                 neuron_len, gs, first_sub_bits, first_format, first_sign_mode);
    check_common(second_indices, second_aux, second_sub_scale, second_neuron_scale, second_codebook,
                 neuron_len, gs, second_sub_bits, second_format, second_sign_mode);
    MFQ_RUNTIME_CHECK(first_format == second_format,
                "NVQ multi-projection requires the same NVQ format");
    MFQ_RUNTIME_CHECK(x.is_cuda() && x.scalar_type() == mfq_tensor_backend::kFloat16 && x.is_contiguous() && x.dim() == 2,
                "NVQ multi-projection x must be CUDA contiguous fp16 rank-2");
    const int M = static_cast<int>(x.size(0));
    const int K = static_cast<int>(neuron_len);
    const int ng = (K + static_cast<int>(gs) - 1) / static_cast<int>(gs);
    const int K_pad = ng * static_cast<int>(gs);
    MFQ_RUNTIME_CHECK(M >= 1 && M <= 8, "NVQ multi-projection supports M in [1,8]");
    MFQ_RUNTIME_CHECK(x.size(1) == K, "NVQ multi-projection x width mismatch");
    MFQ_RUNTIME_CHECK(qx.is_cuda() && qx.scalar_type() == mfq_tensor_backend::kInt8 && qx.is_contiguous() &&
                qx.numel() >= static_cast<int64_t>(M) * K_pad,
                "NVQ multi-projection qx workspace mismatch");
    MFQ_RUNTIME_CHECK(xscale.is_cuda() && xscale.scalar_type() == mfq_tensor_backend::kFloat32 && xscale.is_contiguous() &&
                xscale.numel() >= static_cast<int64_t>(M) * ng,
                "NVQ multi-projection xscale workspace mismatch");
    const auto first = make_device_weight(
        first_indices, first_aux, first_sub_scale, first_neuron_scale, first_codebook,
        K, static_cast<int>(gs), static_cast<int>(first_sub_bits),
        static_cast<int>(first_format), static_cast<int>(first_sign_mode));
    const auto second = make_device_weight(
        second_indices, second_aux, second_sub_scale, second_neuron_scale, second_codebook,
        K, static_cast<int>(gs), static_cast<int>(second_sub_bits),
        static_cast<int>(second_format), static_cast<int>(second_sign_mode));
    auto output = mfq_tensor_backend::empty(
        {M, first.N + second.N}, x.options());
    cudaStream_t stream = mfq_current_cuda_stream();
    nvq_quantize_x_gs24_kernel<<<dim3(M, ng), 32, 0, stream>>>(
        reinterpret_cast<const __half *>(x.data_ptr<mfq_half>()), qx.data_ptr<int8_t>(),
        xscale.data_ptr<float>(), M, K, ng);
    launch_selected_multi2_vec8(
        static_cast<int>(first_format), first, second, qx, xscale, output, M, stream);
    MFQ_RUNTIME_CHECK(cudaGetLastError() == cudaSuccess, "NVQ multi-projection kernel launch failed");
    return output;
}

mfq_tensor_backend::Tensor nvq_gemv_swiglu_ws_cuda(
    mfq_tensor_backend::Tensor gate_indices,
    mfq_tensor_backend::Tensor gate_aux,
    mfq_tensor_backend::Tensor gate_sub_scale,
    mfq_tensor_backend::Tensor gate_neuron_scale,
    mfq_tensor_backend::Tensor gate_codebook,
    mfq_tensor_backend::Tensor up_indices,
    mfq_tensor_backend::Tensor up_aux,
    mfq_tensor_backend::Tensor up_sub_scale,
    mfq_tensor_backend::Tensor up_neuron_scale,
    mfq_tensor_backend::Tensor up_codebook,
    mfq_tensor_backend::Tensor x,
    int64_t neuron_len,
    int64_t gs,
    int64_t gate_sub_bits,
    int64_t gate_format,
    int64_t gate_sign_mode,
    int64_t up_sub_bits,
    int64_t up_format,
    int64_t up_sign_mode,
    mfq_tensor_backend::Tensor qx,
    mfq_tensor_backend::Tensor xscale) {
    check_common(gate_indices, gate_aux, gate_sub_scale, gate_neuron_scale, gate_codebook,
                 neuron_len, gs, gate_sub_bits, gate_format, gate_sign_mode);
    check_common(up_indices, up_aux, up_sub_scale, up_neuron_scale, up_codebook,
                 neuron_len, gs, up_sub_bits, up_format, up_sign_mode);
    MFQ_RUNTIME_CHECK(gate_format == up_format, "NVQ SwiGLU requires matching NVQ formats");
    MFQ_RUNTIME_CHECK(gate_neuron_scale.numel() == up_neuron_scale.numel(),
                "NVQ SwiGLU gate/up output widths must match");
    MFQ_RUNTIME_CHECK(x.is_cuda() && x.scalar_type() == mfq_tensor_backend::kFloat16 && x.is_contiguous() &&
                x.dim() == 2 && x.size(0) == 1,
                "NVQ SwiGLU x must be CUDA contiguous fp16 [1,K]");
    const int K = static_cast<int>(neuron_len);
    const int ng = (K + static_cast<int>(gs) - 1) / static_cast<int>(gs);
    const int K_pad = ng * static_cast<int>(gs);
    MFQ_RUNTIME_CHECK(x.size(1) == K, "NVQ SwiGLU x width mismatch");
    MFQ_RUNTIME_CHECK(qx.numel() >= K_pad && xscale.numel() >= ng,
                "NVQ SwiGLU workspace mismatch");
    const auto gate = make_device_weight(
        gate_indices, gate_aux, gate_sub_scale, gate_neuron_scale, gate_codebook,
        K, static_cast<int>(gs), static_cast<int>(gate_sub_bits),
        static_cast<int>(gate_format), static_cast<int>(gate_sign_mode));
    const auto up = make_device_weight(
        up_indices, up_aux, up_sub_scale, up_neuron_scale, up_codebook,
        K, static_cast<int>(gs), static_cast<int>(up_sub_bits),
        static_cast<int>(up_format), static_cast<int>(up_sign_mode));
    auto output = mfq_tensor_backend::empty({1, gate.N}, x.options());
    cudaStream_t stream = mfq_current_cuda_stream();
    nvq_quantize_x_gs24_kernel<<<dim3(1, ng), 32, 0, stream>>>(
        reinterpret_cast<const __half *>(x.data_ptr<mfq_half>()), qx.data_ptr<int8_t>(),
        xscale.data_ptr<float>(), 1, K, ng);
    if (gate_format == kNvq2JscXLGroupExec &&
        gate_sub_bits == 4 && up_sub_bits == 4) {
        aligned_group_swiglu_pair_kernel<kNvq2JscXLGroupExec, 4>
            <<<gate.N, dim3(32, 4), 0, stream>>>(
                gate, up, qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),
                reinterpret_cast<__half *>(output.data_ptr<mfq_half>()), nullptr);
    } else if (gate_format == kNvq3JscLGroupExec &&
               gate_sub_bits == 4 && up_sub_bits == 4) {
        aligned_group_swiglu_pair_kernel<kNvq3JscLGroupExec, 4>
            <<<gate.N, dim3(32, 4), 0, stream>>>(
                gate, up, qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),
                reinterpret_cast<__half *>(output.data_ptr<mfq_half>()), nullptr);
    } else if (gate_format == kNvq2 && gate_sub_bits == 4 && up_sub_bits == 4) {
        nvq_gemv_swiglu_pair_kernel<kNvq2, 4, false, true>
            <<<gate.N, dim3(32, 4), 0, stream>>>(
                gate, up, qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),
                reinterpret_cast<__half *>(output.data_ptr<mfq_half>()), nullptr);
    } else if (gate_format == kNvq2Exec && gate_sub_bits == 4 && up_sub_bits == 4) {
        nvq_gemv_swiglu_pair_kernel<kNvq2Exec, 4, false, true>
            <<<gate.N, dim3(32, 4), 0, stream>>>(
                gate, up, qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),
                reinterpret_cast<__half *>(output.data_ptr<mfq_half>()), nullptr);
    } else if (gate_format == kNvq2Jsc && gate_sub_bits == 4 && up_sub_bits == 4) {
        nvq_gemv_swiglu_pair_kernel<kNvq2Jsc, 4, false, true>
            <<<gate.N, dim3(32, 4), 0, stream>>>(
                gate, up, qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),
                reinterpret_cast<__half *>(output.data_ptr<mfq_half>()), nullptr);
    } else if (gate_format == kNvq2JscExec && gate_sub_bits == 4 && up_sub_bits == 4) {
        nvq_gemv_swiglu_pair_kernel<kNvq2JscExec, 4, false, true>
            <<<gate.N, dim3(32, 4), 0, stream>>>(
                gate, up, qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),
                reinterpret_cast<__half *>(output.data_ptr<mfq_half>()), nullptr);
    } else {
        launch_by_format(static_cast<int>(gate_format), [&](auto tag) {
            constexpr int F = decltype(tag)::value;
            nvq_gemv_swiglu_pair_kernel<F, 4><<<gate.N, dim3(32, 4), 0, stream>>>(
                gate, up, qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),
                reinterpret_cast<__half *>(output.data_ptr<mfq_half>()), nullptr);
        });
    }
    MFQ_RUNTIME_CHECK(cudaGetLastError() == cudaSuccess, "NVQ SwiGLU kernel launch failed");
    return output;
}

mfq_tensor_backend::Tensor nvq2_gemv_swiglu_vec4_ordered_ws_cuda(
    mfq_tensor_backend::Tensor gate_indices,
    mfq_tensor_backend::Tensor gate_aux,
    mfq_tensor_backend::Tensor gate_sub_scale,
    mfq_tensor_backend::Tensor gate_neuron_scale,
    mfq_tensor_backend::Tensor gate_codebook,
    mfq_tensor_backend::Tensor up_indices,
    mfq_tensor_backend::Tensor up_aux,
    mfq_tensor_backend::Tensor up_sub_scale,
    mfq_tensor_backend::Tensor up_neuron_scale,
    mfq_tensor_backend::Tensor up_codebook,
    mfq_tensor_backend::Tensor x,
    int64_t neuron_len,
    int64_t gs,
    int64_t gate_sub_bits,
    int64_t gate_format,
    int64_t gate_sign_mode,
    int64_t up_sub_bits,
    int64_t up_format,
    int64_t up_sign_mode,
    mfq_tensor_backend::Tensor qx,
    mfq_tensor_backend::Tensor xscale) {
    check_common(gate_indices, gate_aux, gate_sub_scale, gate_neuron_scale, gate_codebook,
                 neuron_len, gs, gate_sub_bits, gate_format, gate_sign_mode);
    check_common(up_indices, up_aux, up_sub_scale, up_neuron_scale, up_codebook,
                 neuron_len, gs, up_sub_bits, up_format, up_sign_mode);
    MFQ_RUNTIME_CHECK(gate_format == kNvq2 && up_format == kNvq2 &&
                gate_sub_bits == 4 && up_sub_bits == 4,
                "NVQ2 ordered vec4 SwiGLU requires NVQ2 sub_bits=4 weights");
    MFQ_RUNTIME_CHECK(gate_neuron_scale.numel() == up_neuron_scale.numel(),
                "NVQ2 ordered vec4 SwiGLU gate/up output widths must match");
    MFQ_RUNTIME_CHECK(x.is_cuda() && x.scalar_type() == mfq_tensor_backend::kFloat16 && x.is_contiguous() &&
                x.dim() == 2 && x.size(0) == 1 && x.size(1) == neuron_len,
                "NVQ2 ordered vec4 SwiGLU x must be CUDA contiguous fp16 [1,K]");
    const int K = static_cast<int>(neuron_len);
    const int ng = (K + static_cast<int>(gs) - 1) / static_cast<int>(gs);
    const int K_pad = ng * static_cast<int>(gs);
    const int nsign = (K + 7) / 8;
    MFQ_RUNTIME_CHECK(nsign % 4 == 0,
                "NVQ2 ordered vec4 SwiGLU requires a multiple of four 8-value vectors");
    MFQ_RUNTIME_CHECK(qx.numel() >= K_pad && xscale.numel() >= ng,
                "NVQ2 ordered vec4 SwiGLU workspace mismatch");
    const auto gate = make_device_weight(
        gate_indices, gate_aux, gate_sub_scale, gate_neuron_scale, gate_codebook,
        K, static_cast<int>(gs), 4, kNvq2, static_cast<int>(gate_sign_mode));
    const auto up = make_device_weight(
        up_indices, up_aux, up_sub_scale, up_neuron_scale, up_codebook,
        K, static_cast<int>(gs), 4, kNvq2, static_cast<int>(up_sign_mode));
    auto output = mfq_tensor_backend::empty({1, gate.N}, x.options());
    cudaStream_t stream = mfq_current_cuda_stream();
    nvq_quantize_x_gs24_kernel<<<dim3(1, ng), 32, 0, stream>>>(
        reinterpret_cast<const __half *>(x.data_ptr<mfq_half>()), qx.data_ptr<int8_t>(),
        xscale.data_ptr<float>(), 1, K, ng);
    const size_t shared_bytes = static_cast<size_t>(nsign) * sizeof(uint32_t);
    nvq2_gemv_swiglu_vec4_ordered_kernel<4>
        <<<gate.N, dim3(32, 4), shared_bytes, stream>>>(
            gate, up, qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),
            reinterpret_cast<__half *>(output.data_ptr<mfq_half>()), nullptr);
    MFQ_RUNTIME_CHECK(cudaGetLastError() == cudaSuccess,
                "NVQ2 ordered vec4 SwiGLU kernel launch failed");
    return output;
}

void nvq_ffn_swiglu_quant_ws_cuda(
    mfq_tensor_backend::Tensor gate_indices,
    mfq_tensor_backend::Tensor gate_aux,
    mfq_tensor_backend::Tensor gate_sub_scale,
    mfq_tensor_backend::Tensor gate_neuron_scale,
    mfq_tensor_backend::Tensor gate_codebook,
    mfq_tensor_backend::Tensor up_indices,
    mfq_tensor_backend::Tensor up_aux,
    mfq_tensor_backend::Tensor up_sub_scale,
    mfq_tensor_backend::Tensor up_neuron_scale,
    mfq_tensor_backend::Tensor up_codebook,
    mfq_tensor_backend::Tensor x,
    int64_t neuron_len,
    int64_t gs,
    int64_t gate_sub_bits,
    int64_t gate_format,
    int64_t gate_sign_mode,
    int64_t up_sub_bits,
    int64_t up_format,
    int64_t up_sign_mode,
    int64_t down_gs,
    mfq_tensor_backend::Tensor input_qx,
    mfq_tensor_backend::Tensor input_xscale,
    mfq_tensor_backend::Tensor output_qx,
    mfq_tensor_backend::Tensor output_xscale,
    mfq_tensor_backend::Tensor swiglu_scratch) {
    check_common(gate_indices, gate_aux, gate_sub_scale, gate_neuron_scale, gate_codebook,
                 neuron_len, gs, gate_sub_bits, gate_format, gate_sign_mode);
    check_common(up_indices, up_aux, up_sub_scale, up_neuron_scale, up_codebook,
                 neuron_len, gs, up_sub_bits, up_format, up_sign_mode);
    MFQ_RUNTIME_CHECK(gate_format == up_format, "NVQ fused FFN requires matching gate/up formats");
    MFQ_RUNTIME_CHECK(gate_neuron_scale.numel() == up_neuron_scale.numel(),
                "NVQ fused FFN gate/up widths must match");
    MFQ_RUNTIME_CHECK(x.is_cuda() && x.scalar_type() == mfq_tensor_backend::kFloat16 && x.is_contiguous() &&
                x.dim() == 2 && x.size(0) == 1 && x.size(1) == neuron_len,
                "NVQ fused FFN x must be CUDA contiguous fp16 [1,K]");
    MFQ_RUNTIME_CHECK(down_gs == 24 || down_gs == 28 || down_gs == 32,
                "NVQ fused FFN down gs must be 24, 28, or 32");
    const int K = static_cast<int>(neuron_len);
    const int ng = (K + static_cast<int>(gs) - 1) / static_cast<int>(gs);
    const int K_pad = ng * static_cast<int>(gs);
    const int N = static_cast<int>(gate_neuron_scale.numel());
    const int down_ng = (N + static_cast<int>(down_gs) - 1) / static_cast<int>(down_gs);
    const int down_k_pad = down_ng * static_cast<int>(down_gs);
    MFQ_RUNTIME_CHECK(input_qx.is_cuda() && input_qx.scalar_type() == mfq_tensor_backend::kInt8 &&
                input_qx.is_contiguous() && input_qx.numel() >= K_pad,
                "NVQ fused FFN input qx workspace mismatch");
    MFQ_RUNTIME_CHECK(input_xscale.is_cuda() && input_xscale.scalar_type() == mfq_tensor_backend::kFloat32 &&
                input_xscale.is_contiguous() && input_xscale.numel() >= ng,
                "NVQ fused FFN input xscale workspace mismatch");
    MFQ_RUNTIME_CHECK(output_qx.is_cuda() && output_qx.scalar_type() == mfq_tensor_backend::kInt8 &&
                output_qx.is_contiguous() && output_qx.numel() >= down_k_pad,
                "NVQ fused FFN output qx workspace mismatch");
    MFQ_RUNTIME_CHECK(output_xscale.is_cuda() && output_xscale.scalar_type() == mfq_tensor_backend::kFloat32 &&
                output_xscale.is_contiguous() && output_xscale.numel() >= down_ng,
                "NVQ fused FFN output xscale workspace mismatch");
    MFQ_RUNTIME_CHECK(swiglu_scratch.is_cuda() && swiglu_scratch.scalar_type() == mfq_tensor_backend::kFloat32 &&
                swiglu_scratch.is_contiguous() && swiglu_scratch.numel() >= N,
                "NVQ fused FFN SwiGLU scratch mismatch");
    const auto gate = make_device_weight(
        gate_indices, gate_aux, gate_sub_scale, gate_neuron_scale, gate_codebook,
        K, static_cast<int>(gs), static_cast<int>(gate_sub_bits),
        static_cast<int>(gate_format), static_cast<int>(gate_sign_mode));
    const auto up = make_device_weight(
        up_indices, up_aux, up_sub_scale, up_neuron_scale, up_codebook,
        K, static_cast<int>(gs), static_cast<int>(up_sub_bits),
        static_cast<int>(up_format), static_cast<int>(up_sign_mode));
    cudaStream_t stream = mfq_current_cuda_stream();
    nvq_quantize_x_gs24_kernel<<<dim3(1, ng), 32, 0, stream>>>(
        reinterpret_cast<const __half *>(x.data_ptr<mfq_half>()), input_qx.data_ptr<int8_t>(),
        input_xscale.data_ptr<float>(), 1, K, ng);
    bool aligned_independent_rows = false;
    const char * vec4_env = mfq::cuda::env_lookup("MFQ_NVQ_SWIGLU_VEC4");
    if (vec4_env == nullptr) vec4_env = mfq::cuda::env_lookup("MFQ_NIQ_SWIGLU_VEC4");
    const bool use_vec4 = vec4_env == nullptr || vec4_env[0] != '0';
    if (gate_format == kNvq2JscXLGroupExec &&
        gate_sub_bits == 4 && up_sub_bits == 4) {
        launch_aligned_group_multi2_m1<kNvq2JscXLGroupExec>(
            gate, up, input_qx.data_ptr<int8_t>(), input_xscale.data_ptr<float>(),
            reinterpret_cast<__half *>(swiglu_scratch.data_ptr<float>()), stream);
        aligned_independent_rows = true;
    } else if (gate_format == kNvq3JscLGroupExec &&
               gate_sub_bits == 4 && up_sub_bits == 4) {
        launch_aligned_group_multi2_m1<kNvq3JscLGroupExec>(
            gate, up, input_qx.data_ptr<int8_t>(), input_xscale.data_ptr<float>(),
            reinterpret_cast<__half *>(swiglu_scratch.data_ptr<float>()), stream);
        aligned_independent_rows = true;
    } else if (gate_format == kNvq2Exec && gate_sub_bits == 4 && up_sub_bits == 4) {
        nvq_gemv_swiglu_pair_kernel<kNvq2Exec, 4, true, true>
            <<<gate.N, dim3(32, 4), 0, stream>>>(
                gate, up, input_qx.data_ptr<int8_t>(), input_xscale.data_ptr<float>(),
                nullptr, swiglu_scratch.data_ptr<float>());
    } else if (gate_format == kNvq2JscExec && gate_sub_bits == 4 && up_sub_bits == 4) {
        nvq_gemv_swiglu_pair_kernel<kNvq2JscExec, 4, true, true>
            <<<gate.N, dim3(32, 4), 0, stream>>>(
                gate, up, input_qx.data_ptr<int8_t>(), input_xscale.data_ptr<float>(),
                nullptr, swiglu_scratch.data_ptr<float>());
    } else if (gate_format == kNvq2Jsc && gate_sub_bits == 4 && up_sub_bits == 4) {
        nvq_gemv_swiglu_pair_kernel<kNvq2Jsc, 4, true, true>
            <<<gate.N, dim3(32, 4), 0, stream>>>(
                gate, up, input_qx.data_ptr<int8_t>(), input_xscale.data_ptr<float>(),
                nullptr, swiglu_scratch.data_ptr<float>());
    } else if (use_vec4 && gate_format == kNvq2 && gate_sub_bits == 4 && up_sub_bits == 4 &&
        gate.nsign % 4 == 0 && up.nsign % 4 == 0) {
        const size_t shared_bytes = static_cast<size_t>(gate.nsign) * sizeof(uint32_t);
        nvq2_gemv_swiglu_vec4_ordered_kernel<4, true>
            <<<gate.N, dim3(32, 4), shared_bytes, stream>>>(
                gate, up, input_qx.data_ptr<int8_t>(), input_xscale.data_ptr<float>(),
                nullptr, swiglu_scratch.data_ptr<float>());
    } else if (gate_format == kNvq2 && gate_sub_bits == 4 && up_sub_bits == 4) {
        nvq_gemv_swiglu_pair_kernel<kNvq2, 4, true, true>
            <<<gate.N, dim3(32, 4), 0, stream>>>(
                gate, up, input_qx.data_ptr<int8_t>(), input_xscale.data_ptr<float>(),
                nullptr, swiglu_scratch.data_ptr<float>());
    } else {
        launch_by_format(static_cast<int>(gate_format), [&](auto tag) {
            constexpr int F = decltype(tag)::value;
            nvq_gemv_swiglu_pair_kernel<F, 4, true><<<gate.N, dim3(32, 4), 0, stream>>>(
                gate, up, input_qx.data_ptr<int8_t>(), input_xscale.data_ptr<float>(),
                nullptr, swiglu_scratch.data_ptr<float>());
        });
    }
    switch (static_cast<int>(down_gs)) {
        case 24:
            if (aligned_independent_rows) {
                nvq_swiglu_quantize_f16_pair_kernel<24><<<down_ng, 32, 0, stream>>>(
                    reinterpret_cast<const __half *>(swiglu_scratch.data_ptr<float>()),
                    output_qx.data_ptr<int8_t>(), output_xscale.data_ptr<float>(),
                    N, down_k_pad);
            } else {
                nvq_quantize_f32_kernel<24><<<down_ng, 32, 0, stream>>>(
                    swiglu_scratch.data_ptr<float>(), output_qx.data_ptr<int8_t>(),
                    output_xscale.data_ptr<float>(), N, down_k_pad);
            }
            break;
        case 28:
            if (aligned_independent_rows) {
                nvq_swiglu_quantize_f16_pair_kernel<28><<<down_ng, 32, 0, stream>>>(
                    reinterpret_cast<const __half *>(swiglu_scratch.data_ptr<float>()),
                    output_qx.data_ptr<int8_t>(), output_xscale.data_ptr<float>(),
                    N, down_k_pad);
            } else {
                nvq_quantize_f32_kernel<28><<<down_ng, 32, 0, stream>>>(
                    swiglu_scratch.data_ptr<float>(), output_qx.data_ptr<int8_t>(),
                    output_xscale.data_ptr<float>(), N, down_k_pad);
            }
            break;
        case 32:
            if (aligned_independent_rows) {
                nvq_swiglu_quantize_f16_pair_kernel<32><<<down_ng, 32, 0, stream>>>(
                    reinterpret_cast<const __half *>(swiglu_scratch.data_ptr<float>()),
                    output_qx.data_ptr<int8_t>(), output_xscale.data_ptr<float>(),
                    N, down_k_pad);
            } else {
                nvq_quantize_f32_kernel<32><<<down_ng, 32, 0, stream>>>(
                    swiglu_scratch.data_ptr<float>(), output_qx.data_ptr<int8_t>(),
                    output_xscale.data_ptr<float>(), N, down_k_pad);
            }
            break;
    }
    MFQ_RUNTIME_CHECK(cudaGetLastError() == cudaSuccess, "NVQ fused FFN kernel launch failed");
}

mfq_tensor_backend::Tensor nvq_gemv_gate_ws_cuda(
    mfq_tensor_backend::Tensor indices,
    mfq_tensor_backend::Tensor aux,
    mfq_tensor_backend::Tensor sub_scale,
    mfq_tensor_backend::Tensor neuron_scale,
    mfq_tensor_backend::Tensor codebook,
    mfq_tensor_backend::Tensor x,
    mfq_tensor_backend::Tensor gate,
    int64_t neuron_len,
    int64_t gs,
    int64_t sub_bits,
    int64_t format,
    int64_t sign_mode,
    int64_t mode,
    mfq_tensor_backend::Tensor qx,
    mfq_tensor_backend::Tensor xscale) {
    check_common(indices, aux, sub_scale, neuron_scale, codebook,
                 neuron_len, gs, sub_bits, format, sign_mode);
    MFQ_RUNTIME_CHECK(mode == 1 || mode == 2, "NVQ gate mode must be 1(sigmoid) or 2(silu)");
    MFQ_RUNTIME_CHECK(x.is_cuda() && x.scalar_type() == mfq_tensor_backend::kFloat16 && x.is_contiguous() && x.dim() == 2,
                "NVQ x must be CUDA contiguous fp16 rank-2");
    MFQ_RUNTIME_CHECK(gate.is_cuda() && gate.scalar_type() == mfq_tensor_backend::kFloat16 && gate.is_contiguous() &&
                gate.sizes() == x.sizes(), "NVQ gate must be CUDA contiguous fp16 with x shape");
    const int M = static_cast<int>(x.size(0));
    const int K = static_cast<int>(neuron_len);
    const int N = static_cast<int>(neuron_scale.numel());
    const int ng = (K + static_cast<int>(gs) - 1) / static_cast<int>(gs);
    const int K_pad = ng * static_cast<int>(gs);
    MFQ_RUNTIME_CHECK(x.size(1) == K, "NVQ x width must equal neuron_len");
    MFQ_RUNTIME_CHECK(M >= 1 && M <= 16, "NVQ gated direct GEMV supports M in [1,16]");
    MFQ_RUNTIME_CHECK(qx.is_cuda() && qx.scalar_type() == mfq_tensor_backend::kInt8 && qx.is_contiguous() &&
                qx.numel() >= static_cast<int64_t>(M) * K_pad, "NVQ qx workspace mismatch");
    MFQ_RUNTIME_CHECK(xscale.is_cuda() && xscale.scalar_type() == mfq_tensor_backend::kFloat32 && xscale.is_contiguous() &&
                xscale.numel() >= static_cast<int64_t>(M) * ng, "NVQ xscale workspace mismatch");
    const int nvec = (K + (is_d4_format(format) ? 3 : 7)) /
                     (is_d4_format(format) ? 4 : 8);
    const int nsign = (K + 7) / 8;
    auto output = mfq_tensor_backend::empty({M, N}, x.options());
    cudaStream_t stream = mfq_current_cuda_stream();
    if (mode == 1) {
        nvq_quantize_x_gate_gs24_kernel<1><<<dim3(M, ng), 32, 0, stream>>>(
            reinterpret_cast<const __half *>(x.data_ptr<mfq_half>()),
            reinterpret_cast<const __half *>(gate.data_ptr<mfq_half>()),
            qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), M, K, ng);
    } else {
        nvq_quantize_x_gate_gs24_kernel<2><<<dim3(M, ng), 32, 0, stream>>>(
            reinterpret_cast<const __half *>(x.data_ptr<mfq_half>()),
            reinterpret_cast<const __half *>(gate.data_ptr<mfq_half>()),
            qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), M, K, ng);
    }
    if (M == 1) {
        launch_selected_m1_vec8(
            static_cast<int>(format), indices, aux, sub_scale, neuron_scale, codebook,
            qx, xscale, output, N, ng, nvec, nsign,
            static_cast<int>(sub_bits), static_cast<int>(sign_mode), stream);
        MFQ_RUNTIME_CHECK(cudaGetLastError() == cudaSuccess, "NVQ gated vec8 GEMV kernel launch failed");
        return output;
    }
    launch_selected_batch_vec8(
        static_cast<int>(format), indices, aux, sub_scale, neuron_scale, codebook,
        qx, xscale, output, M, N, ng, nvec, nsign,
        static_cast<int>(sub_bits), static_cast<int>(sign_mode), stream);
    MFQ_RUNTIME_CHECK(cudaGetLastError() == cudaSuccess, "NVQ gated batch vec8 GEMV kernel launch failed");
    return output;
}

mfq_tensor_backend::Tensor nvq_mmq_ws_cuda(
    mfq_tensor_backend::Tensor indices,
    mfq_tensor_backend::Tensor aux,
    mfq_tensor_backend::Tensor sub_scale,
    mfq_tensor_backend::Tensor neuron_scale,
    mfq_tensor_backend::Tensor codebook,
    mfq_tensor_backend::Tensor x,
    int64_t neuron_len,
    int64_t gs,
    int64_t sub_bits,
    int64_t format,
    int64_t sign_mode,
    mfq_tensor_backend::Tensor qx,
    mfq_tensor_backend::Tensor xscale) {
    check_common(indices, aux, sub_scale, neuron_scale, codebook,
                 neuron_len, gs, sub_bits, format, sign_mode);
    MFQ_RUNTIME_CHECK(x.is_cuda() && x.scalar_type() == mfq_tensor_backend::kFloat16 && x.is_contiguous() && x.dim() == 2,
                "NVQ x must be CUDA contiguous fp16 rank-2");
    const int M = static_cast<int>(x.size(0));
    const int K = static_cast<int>(neuron_len);
    const int N = static_cast<int>(neuron_scale.numel());
    const int ng = (K + static_cast<int>(gs) - 1) / static_cast<int>(gs);
    const int K_pad = ng * static_cast<int>(gs);
    MFQ_RUNTIME_CHECK(x.size(1) == K, "NVQ x width must equal neuron_len");
    MFQ_RUNTIME_CHECK(M >= 4 && M <= 64, "NVQ MMA MMQ supports M in [4,64]");
    MFQ_RUNTIME_CHECK(qx.is_cuda() && qx.scalar_type() == mfq_tensor_backend::kInt8 && qx.is_contiguous() &&
                qx.numel() >= static_cast<int64_t>(M) * K_pad, "NVQ qx workspace mismatch");
    MFQ_RUNTIME_CHECK(xscale.is_cuda() && xscale.scalar_type() == mfq_tensor_backend::kFloat32 && xscale.is_contiguous() &&
                xscale.numel() >= static_cast<int64_t>(M) * ng, "NVQ xscale workspace mismatch");
    const int nvec = (K + (is_d4_format(format) ? 3 : 7)) /
                     (is_d4_format(format) ? 4 : 8);
    const int nsign = (K + 7) / 8;
    auto output = mfq_tensor_backend::empty({M, N}, x.options());
    cudaStream_t stream = mfq_current_cuda_stream();
    nvq_quantize_x_gs24_kernel<<<dim3(M, ng), 32, 0, stream>>>(
        reinterpret_cast<const __half *>(x.data_ptr<mfq_half>()), qx.data_ptr<int8_t>(),
        xscale.data_ptr<float>(), M, K, ng);
    const dim3 block(32, 8);
#define NVQ_MMQ_LAUNCH(MTILES_VALUE, NEED_CHECK_VALUE)                            \
    nvq_mmq_mma24_kernel<F, MTILES_VALUE, NEED_CHECK_VALUE><<<                   \
        dim3((N + 63) / 64, 1), block, 0, stream>>>(                             \
        indices.data_ptr<uint8_t>(), indices.numel(),                            \
        aux.data_ptr<uint8_t>(), aux.numel(),                                    \
        sub_scale.data_ptr<uint8_t>(), sub_scale.numel(),                        \
        neuron_scale.data_ptr<float>(), codebook.data_ptr<int8_t>(),             \
        qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),                         \
        reinterpret_cast<__half *>(output.data_ptr<mfq_half>()),                 \
        M, N, ng, nvec, nsign, static_cast<int>(sub_bits),                       \
        static_cast<int>(sign_mode))
    launch_by_format(static_cast<int>(format), [&](auto tag) {
        constexpr int F = decltype(tag)::value;
        if (M <= 16) {
            if (M == 16 && N % 64 == 0) NVQ_MMQ_LAUNCH(1, false);
            else NVQ_MMQ_LAUNCH(1, true);
        } else if (M <= 32) {
            if (M == 32 && N % 64 == 0) NVQ_MMQ_LAUNCH(2, false);
            else NVQ_MMQ_LAUNCH(2, true);
        } else if (M <= 48) {
            if (M == 48 && N % 64 == 0) NVQ_MMQ_LAUNCH(3, false);
            else NVQ_MMQ_LAUNCH(3, true);
        } else {
            if (M == 64 && N % 64 == 0) NVQ_MMQ_LAUNCH(4, false);
            else NVQ_MMQ_LAUNCH(4, true);
        }
    });
#undef NVQ_MMQ_LAUNCH
    MFQ_RUNTIME_CHECK(cudaGetLastError() == cudaSuccess, "NVQ MMA MMQ kernel launch failed");
    return output;
}

mfq_tensor_backend::Tensor nvq_mmq_gate_ws_cuda(
    mfq_tensor_backend::Tensor indices,
    mfq_tensor_backend::Tensor aux,
    mfq_tensor_backend::Tensor sub_scale,
    mfq_tensor_backend::Tensor neuron_scale,
    mfq_tensor_backend::Tensor codebook,
    mfq_tensor_backend::Tensor x,
    mfq_tensor_backend::Tensor gate,
    int64_t neuron_len,
    int64_t gs,
    int64_t sub_bits,
    int64_t format,
    int64_t sign_mode,
    int64_t mode,
    mfq_tensor_backend::Tensor qx,
    mfq_tensor_backend::Tensor xscale) {
    check_common(indices, aux, sub_scale, neuron_scale, codebook,
                 neuron_len, gs, sub_bits, format, sign_mode);
    MFQ_RUNTIME_CHECK(mode == 1 || mode == 2, "NVQ gate mode must be 1(sigmoid) or 2(silu)");
    MFQ_RUNTIME_CHECK(x.is_cuda() && x.scalar_type() == mfq_tensor_backend::kFloat16 && x.is_contiguous() && x.dim() == 2,
                "NVQ x must be CUDA contiguous fp16 rank-2");
    MFQ_RUNTIME_CHECK(gate.is_cuda() && gate.scalar_type() == mfq_tensor_backend::kFloat16 && gate.is_contiguous() &&
                gate.sizes() == x.sizes(), "NVQ gate must be CUDA contiguous fp16 with x shape");
    const int M = static_cast<int>(x.size(0));
    const int K = static_cast<int>(neuron_len);
    const int N = static_cast<int>(neuron_scale.numel());
    const int ng = (K + static_cast<int>(gs) - 1) / static_cast<int>(gs);
    const int K_pad = ng * static_cast<int>(gs);
    MFQ_RUNTIME_CHECK(x.size(1) == K, "NVQ x width must equal neuron_len");
    MFQ_RUNTIME_CHECK(M >= 4 && M <= 64, "NVQ gated MMA MMQ supports M in [4,64]");
    MFQ_RUNTIME_CHECK(qx.is_cuda() && qx.scalar_type() == mfq_tensor_backend::kInt8 && qx.is_contiguous() &&
                qx.numel() >= static_cast<int64_t>(M) * K_pad, "NVQ qx workspace mismatch");
    MFQ_RUNTIME_CHECK(xscale.is_cuda() && xscale.scalar_type() == mfq_tensor_backend::kFloat32 && xscale.is_contiguous() &&
                xscale.numel() >= static_cast<int64_t>(M) * ng, "NVQ xscale workspace mismatch");
    const int nvec = (K + (is_d4_format(format) ? 3 : 7)) /
                     (is_d4_format(format) ? 4 : 8);
    const int nsign = (K + 7) / 8;
    auto output = mfq_tensor_backend::empty({M, N}, x.options());
    cudaStream_t stream = mfq_current_cuda_stream();
    if (mode == 1) {
        nvq_quantize_x_gate_gs24_kernel<1><<<dim3(M, ng), 32, 0, stream>>>(
            reinterpret_cast<const __half *>(x.data_ptr<mfq_half>()),
            reinterpret_cast<const __half *>(gate.data_ptr<mfq_half>()),
            qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), M, K, ng);
    } else {
        nvq_quantize_x_gate_gs24_kernel<2><<<dim3(M, ng), 32, 0, stream>>>(
            reinterpret_cast<const __half *>(x.data_ptr<mfq_half>()),
            reinterpret_cast<const __half *>(gate.data_ptr<mfq_half>()),
            qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), M, K, ng);
    }
    const dim3 block(32, 8);
#define NVQ_GATED_MMQ_LAUNCH(MTILES_VALUE, NEED_CHECK_VALUE)                      \
    nvq_mmq_mma24_kernel<F, MTILES_VALUE, NEED_CHECK_VALUE><<<                   \
        dim3((N + 63) / 64, 1), block, 0, stream>>>(                             \
        indices.data_ptr<uint8_t>(), indices.numel(),                            \
        aux.data_ptr<uint8_t>(), aux.numel(),                                    \
        sub_scale.data_ptr<uint8_t>(), sub_scale.numel(),                        \
        neuron_scale.data_ptr<float>(), codebook.data_ptr<int8_t>(),             \
        qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),                         \
        reinterpret_cast<__half *>(output.data_ptr<mfq_half>()),                 \
        M, N, ng, nvec, nsign, static_cast<int>(sub_bits),                       \
        static_cast<int>(sign_mode))
    launch_by_format(static_cast<int>(format), [&](auto tag) {
        constexpr int F = decltype(tag)::value;
        if (M <= 16) {
            if (M == 16 && N % 64 == 0) NVQ_GATED_MMQ_LAUNCH(1, false);
            else NVQ_GATED_MMQ_LAUNCH(1, true);
        } else if (M <= 32) {
            if (M == 32 && N % 64 == 0) NVQ_GATED_MMQ_LAUNCH(2, false);
            else NVQ_GATED_MMQ_LAUNCH(2, true);
        } else if (M <= 48) {
            if (M == 48 && N % 64 == 0) NVQ_GATED_MMQ_LAUNCH(3, false);
            else NVQ_GATED_MMQ_LAUNCH(3, true);
        } else {
            if (M == 64 && N % 64 == 0) NVQ_GATED_MMQ_LAUNCH(4, false);
            else NVQ_GATED_MMQ_LAUNCH(4, true);
        }
    });
#undef NVQ_GATED_MMQ_LAUNCH
    MFQ_RUNTIME_CHECK(cudaGetLastError() == cudaSuccess, "NVQ gated MMA MMQ kernel launch failed");
    return output;
}

mfq_tensor_backend::Tensor nvq_embedding_lookup_cuda(
    mfq_tensor_backend::Tensor indices,
    mfq_tensor_backend::Tensor aux,
    mfq_tensor_backend::Tensor sub_scale,
    mfq_tensor_backend::Tensor neuron_scale,
    mfq_tensor_backend::Tensor codebook,
    mfq_tensor_backend::Tensor token_ids,
    int64_t neuron_len,
    int64_t gs,
    int64_t sub_bits,
    int64_t format,
    int64_t sign_mode) {
    check_common(indices, aux, sub_scale, neuron_scale, codebook,
                 neuron_len, gs, sub_bits, format, sign_mode);
    MFQ_RUNTIME_CHECK(token_ids.is_cuda() && token_ids.scalar_type() == mfq_tensor_backend::kInt64 && token_ids.is_contiguous(),
                "NVQ token_ids must be CUDA contiguous int64");
    const int tokens = static_cast<int>(token_ids.numel());
    const int vocab = static_cast<int>(neuron_scale.numel());
    const int K = static_cast<int>(neuron_len);
    const int ng = (K + static_cast<int>(gs) - 1) / static_cast<int>(gs);
    const int nvec = (K + (is_d4_format(format) ? 3 : 7)) /
                     (is_d4_format(format) ? 4 : 8);
    const int nsign = (K + 7) / 8;
    auto shape = token_ids.sizes().vec();
    shape.push_back(K);
    auto output = mfq_tensor_backend::empty(shape, neuron_scale.options().dtype(mfq_tensor_backend::kFloat16));
    const int64_t total = static_cast<int64_t>(tokens) * nsign;
    const int block = 256;
    const int grid = static_cast<int>(std::min<int64_t>((total + block - 1) / block, 65535));
    cudaStream_t stream = mfq_current_cuda_stream();
    launch_by_format(static_cast<int>(format), [&](auto tag) {
        constexpr int F = decltype(tag)::value;
        nvq_embedding_kernel<F><<<grid, block, 0, stream>>>(
            indices.data_ptr<uint8_t>(), indices.numel(), aux.data_ptr<uint8_t>(), aux.numel(),
            sub_scale.data_ptr<uint8_t>(), sub_scale.numel(), neuron_scale.data_ptr<float>(),
            codebook.data_ptr<int8_t>(), token_ids.data_ptr<int64_t>(),
            reinterpret_cast<__half *>(output.data_ptr<mfq_half>()),
            tokens, vocab, K, ng, nvec, nsign, static_cast<int>(sub_bits), static_cast<int>(sign_mode));
    });
    MFQ_RUNTIME_CHECK(cudaGetLastError() == cudaSuccess, "NVQ embedding kernel launch failed");
    return output;
}
