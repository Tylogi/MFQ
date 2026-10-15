#define MFQ_NVQ_MOE 1
#include "device/env.h"
#include "matmul_impl.cuh"

mfq_tensor_backend::Tensor nepq_dequant_cuda(
    mfq_tensor_backend::Tensor indices,
    mfq_tensor_backend::Tensor aux,
    mfq_tensor_backend::Tensor sub_scale,
    mfq_tensor_backend::Tensor neuron_scale,
    mfq_tensor_backend::Tensor table_pool,
    mfq_tensor_backend::Tensor bank_ids,
    int64_t neuron_len,
    int64_t sub_bits,
    int64_t format) {
    check_nepq_common(
        indices, aux, sub_scale, neuron_scale, table_pool, bank_ids,
        neuron_len, sub_bits, format);
    const int N = static_cast<int>(neuron_scale.numel());
    const int K = static_cast<int>(neuron_len);
    const int ng = (K + kGroupSize - 1) / kGroupSize;
    const int nvec = K / 8;
    const int nsign = (K + 7) / 8;
    const int nsuper = (ng + kNepqGroupsPerSupergroup - 1) /
        kNepqGroupsPerSupergroup;
    const int table_stride = static_cast<int>(table_pool.size(1));
    auto output = mfq_tensor_backend::empty({N, K}, neuron_scale.options().dtype(mfq_tensor_backend::kFloat16));
    const int block = 256;
    const int grid = static_cast<int>(std::min<int64_t>(
        (static_cast<int64_t>(N) * nsign + block - 1) / block, 65535));
    cudaStream_t stream = mfq_current_cuda_stream();
    launch_nepq_by_format(static_cast<int>(format), [&](auto tag) {
        constexpr int F = decltype(tag)::value;
        nepq_dequant_kernel<F><<<grid, block, 0, stream>>>(
            indices.data_ptr<uint8_t>(), indices.numel(),
            aux.data_ptr<uint8_t>(), aux.numel(),
            sub_scale.data_ptr<uint8_t>(), sub_scale.numel(),
            neuron_scale.data_ptr<float>(), table_pool.data_ptr<int8_t>(),
            bank_ids.data_ptr<uint8_t>(),
            reinterpret_cast<__half *>(output.data_ptr<mfq_half>()),
            N, K, ng, nvec, nsign, nsuper, table_stride,
            static_cast<int>(sub_bits), 0);
    });
    MFQ_RUNTIME_CHECK(cudaGetLastError() == cudaSuccess, "NEPQ dequant kernel launch failed");
    return output;
}

mfq_tensor_backend::Tensor nepq_gemv_ws_cuda(
    mfq_tensor_backend::Tensor indices,
    mfq_tensor_backend::Tensor aux,
    mfq_tensor_backend::Tensor sub_scale,
    mfq_tensor_backend::Tensor neuron_scale,
    mfq_tensor_backend::Tensor table_pool,
    mfq_tensor_backend::Tensor bank_ids,
    mfq_tensor_backend::Tensor x,
    int64_t neuron_len,
    int64_t sub_bits,
    int64_t format,
    mfq_tensor_backend::Tensor qx,
    mfq_tensor_backend::Tensor xscale) {
    check_nepq_common(
        indices, aux, sub_scale, neuron_scale, table_pool, bank_ids,
        neuron_len, sub_bits, format);
    MFQ_RUNTIME_CHECK(x.is_cuda() && x.scalar_type() == mfq_tensor_backend::kFloat16 &&
                x.is_contiguous() && x.dim() == 2,
                "NEPQ x must be CUDA contiguous fp16 rank-2");
    const int M = static_cast<int>(x.size(0));
    const int K = static_cast<int>(neuron_len);
    const int N = static_cast<int>(neuron_scale.numel());
    const int ng = (K + kGroupSize - 1) / kGroupSize;
    const int K_pad = ng * kGroupSize;
    MFQ_RUNTIME_CHECK(x.size(1) == K && M >= 1 && M <= 16,
                "NEPQ GEMV input must be [M,K] with M in [1,16]");
    MFQ_RUNTIME_CHECK(qx.is_cuda() && qx.scalar_type() == mfq_tensor_backend::kInt8 &&
                qx.is_contiguous() && qx.numel() >= static_cast<int64_t>(M) * K_pad,
                "NEPQ qx workspace mismatch");
    MFQ_RUNTIME_CHECK(xscale.is_cuda() && xscale.scalar_type() == mfq_tensor_backend::kFloat32 &&
                xscale.is_contiguous() && xscale.numel() >= static_cast<int64_t>(M) * ng,
                "NEPQ xscale workspace mismatch");
    const int nvec = K / 8;
    const int nsign = (K + 7) / 8;
    const int nsuper = (ng + kNepqGroupsPerSupergroup - 1) /
        kNepqGroupsPerSupergroup;
    const int table_stride = static_cast<int>(table_pool.size(1));
    auto output = mfq_tensor_backend::empty({M, N}, x.options());
    cudaStream_t stream = mfq_current_cuda_stream();
    nvq_quantize_x_gs24_kernel<<<dim3(M, ng), 32, 0, stream>>>(
        reinterpret_cast<const __half *>(x.data_ptr<mfq_half>()),
        qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), M, K, ng);

#define NEPQ_GEMV_LAUNCH(NWARPS_VALUE, MAX_M_VALUE)                               \
    nepq_gemv_batch_vec8_kernel<F, NWARPS_VALUE, MAX_M_VALUE><<<                 \
        N, dim3(32, NWARPS_VALUE), 0, stream>>>(                                  \
        indices.data_ptr<uint8_t>(), indices.numel(),                             \
        aux.data_ptr<uint8_t>(), aux.numel(),                                     \
        sub_scale.data_ptr<uint8_t>(), sub_scale.numel(),                         \
        neuron_scale.data_ptr<float>(), table_pool.data_ptr<int8_t>(),            \
        bank_ids.data_ptr<uint8_t>(), qx.data_ptr<int8_t>(),                      \
        xscale.data_ptr<float>(),                                                 \
        reinterpret_cast<__half *>(output.data_ptr<mfq_half>()),                  \
        M, N, ng, nvec, nsign, nsuper, table_stride,                              \
        static_cast<int>(sub_bits), 0)
    launch_nepq_by_format(static_cast<int>(format), [&](auto tag) {
        constexpr int F = decltype(tag)::value;
        const int nwarps = (format == kNvq1S && N <= 2048) ? 8 :
            ((format == kNpq0S && N <= 1024) ? 2 : 4);
        if (M == 1) {
            if (nwarps == 8) NEPQ_GEMV_LAUNCH(8, 1);
            else if (nwarps == 2) NEPQ_GEMV_LAUNCH(2, 1);
            else NEPQ_GEMV_LAUNCH(4, 1);
        } else if (M <= 2) {
            if (nwarps == 8) NEPQ_GEMV_LAUNCH(8, 2);
            else if (nwarps == 2) NEPQ_GEMV_LAUNCH(2, 2);
            else NEPQ_GEMV_LAUNCH(4, 2);
        } else if (M <= 4) {
            if (nwarps == 8) NEPQ_GEMV_LAUNCH(8, 4);
            else if (nwarps == 2) NEPQ_GEMV_LAUNCH(2, 4);
            else NEPQ_GEMV_LAUNCH(4, 4);
        } else if (M <= 8) {
            if (nwarps == 8) NEPQ_GEMV_LAUNCH(8, 8);
            else if (nwarps == 2) NEPQ_GEMV_LAUNCH(2, 8);
            else NEPQ_GEMV_LAUNCH(4, 8);
        } else {
            if (nwarps == 8) NEPQ_GEMV_LAUNCH(8, 16);
            else if (nwarps == 2) NEPQ_GEMV_LAUNCH(2, 16);
            else NEPQ_GEMV_LAUNCH(4, 16);
        }
    });
#undef NEPQ_GEMV_LAUNCH
    MFQ_RUNTIME_CHECK(cudaGetLastError() == cudaSuccess, "NEPQ GEMV kernel launch failed");
    return output;
}

mfq_tensor_backend::Tensor nepq_mmq_ws_cuda(
    mfq_tensor_backend::Tensor indices,
    mfq_tensor_backend::Tensor aux,
    mfq_tensor_backend::Tensor sub_scale,
    mfq_tensor_backend::Tensor neuron_scale,
    mfq_tensor_backend::Tensor table_pool,
    mfq_tensor_backend::Tensor bank_ids,
    mfq_tensor_backend::Tensor x,
    int64_t neuron_len,
    int64_t sub_bits,
    int64_t format,
    mfq_tensor_backend::Tensor qx,
    mfq_tensor_backend::Tensor xscale) {
    check_nepq_common(
        indices, aux, sub_scale, neuron_scale, table_pool, bank_ids,
        neuron_len, sub_bits, format);
    MFQ_RUNTIME_CHECK(x.is_cuda() && x.scalar_type() == mfq_tensor_backend::kFloat16 &&
                x.is_contiguous() && x.dim() == 2,
                "NEPQ x must be CUDA contiguous fp16 rank-2");
    const int M = static_cast<int>(x.size(0));
    const int K = static_cast<int>(neuron_len);
    const int N = static_cast<int>(neuron_scale.numel());
    const int ng = (K + kGroupSize - 1) / kGroupSize;
    const int K_pad = ng * kGroupSize;
    MFQ_RUNTIME_CHECK(x.size(1) == K && M >= 4 && M <= 64,
                "NEPQ MMQ input must be [M,K] with M in [4,64]");
    MFQ_RUNTIME_CHECK(qx.is_cuda() && qx.scalar_type() == mfq_tensor_backend::kInt8 &&
                qx.is_contiguous() && qx.numel() >= static_cast<int64_t>(M) * K_pad,
                "NEPQ qx workspace mismatch");
    MFQ_RUNTIME_CHECK(xscale.is_cuda() && xscale.scalar_type() == mfq_tensor_backend::kFloat32 &&
                xscale.is_contiguous() && xscale.numel() >= static_cast<int64_t>(M) * ng,
                "NEPQ xscale workspace mismatch");
    const int nvec = K / 8;
    const int nsign = (K + 7) / 8;
    const int nsuper = (ng + kNepqGroupsPerSupergroup - 1) /
        kNepqGroupsPerSupergroup;
    const int table_stride = static_cast<int>(table_pool.size(1));
    auto output = mfq_tensor_backend::empty({M, N}, x.options());
    cudaStream_t stream = mfq_current_cuda_stream();
    nvq_quantize_x_gs24_kernel<<<dim3(M, ng), 32, 0, stream>>>(
        reinterpret_cast<const __half *>(x.data_ptr<mfq_half>()),
        qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), M, K, ng);
    const dim3 block(32, 8);
#define NEPQ_MMQ_LAUNCH(MTILES_VALUE, NEED_CHECK_VALUE)                           \
    nepq_mmq_mma24_kernel<F, MTILES_VALUE, NEED_CHECK_VALUE><<<                  \
        dim3((N + 63) / 64, 1), block, 0, stream>>>(                             \
        indices.data_ptr<uint8_t>(), indices.numel(),                            \
        aux.data_ptr<uint8_t>(), aux.numel(),                                    \
        sub_scale.data_ptr<uint8_t>(), sub_scale.numel(),                        \
        neuron_scale.data_ptr<float>(), table_pool.data_ptr<int8_t>(),           \
        bank_ids.data_ptr<uint8_t>(), qx.data_ptr<int8_t>(),                     \
        xscale.data_ptr<float>(),                                                \
        reinterpret_cast<__half *>(output.data_ptr<mfq_half>()),                 \
        M, N, ng, nvec, nsign, nsuper, table_stride,                             \
        static_cast<int>(sub_bits), 0)
    launch_nepq_by_format(static_cast<int>(format), [&](auto tag) {
        constexpr int F = decltype(tag)::value;
        if (M <= 16) {
            if (M == 16 && N % 64 == 0) NEPQ_MMQ_LAUNCH(1, false);
            else NEPQ_MMQ_LAUNCH(1, true);
        } else if (M <= 32) {
            if (M == 32 && N % 64 == 0) NEPQ_MMQ_LAUNCH(2, false);
            else NEPQ_MMQ_LAUNCH(2, true);
        } else if (M <= 48) {
            if (M == 48 && N % 64 == 0) NEPQ_MMQ_LAUNCH(3, false);
            else NEPQ_MMQ_LAUNCH(3, true);
        } else {
            if (M == 64 && N % 64 == 0) NEPQ_MMQ_LAUNCH(4, false);
            else NEPQ_MMQ_LAUNCH(4, true);
        }
    });
#undef NEPQ_MMQ_LAUNCH
    MFQ_RUNTIME_CHECK(cudaGetLastError() == cudaSuccess, "NEPQ MMQ kernel launch failed");
    return output;
}

mfq_tensor_backend::Tensor nepq_gemm_f16_cuda(
    mfq_tensor_backend::Tensor indices,
    mfq_tensor_backend::Tensor aux,
    mfq_tensor_backend::Tensor sub_scale,
    mfq_tensor_backend::Tensor neuron_scale,
    mfq_tensor_backend::Tensor table_pool,
    mfq_tensor_backend::Tensor bank_ids,
    mfq_tensor_backend::Tensor x,
    int64_t neuron_len,
    int64_t sub_bits,
    int64_t format) {
    check_nepq_common(
        indices, aux, sub_scale, neuron_scale, table_pool, bank_ids,
        neuron_len, sub_bits, format);
    MFQ_RUNTIME_CHECK(x.is_cuda() && x.scalar_type() == mfq_tensor_backend::kFloat16 &&
                x.is_contiguous() && x.dim() == 2,
                "NEPQ x must be CUDA contiguous fp16 rank-2");
    const int M = static_cast<int>(x.size(0));
    const int K = static_cast<int>(neuron_len);
    const int N = static_cast<int>(neuron_scale.numel());
    const int ng = (K + kGroupSize - 1) / kGroupSize;
    MFQ_RUNTIME_CHECK(x.size(1) == K && M >= 16 && M <= 256,
                "NEPQ online GEMM input must be [M,K] with M in [16,256]");
    const int nvec = K / 8;
    const int nsign = (K + 7) / 8;
    const int nsuper = (ng + kNepqGroupsPerSupergroup - 1) /
        kNepqGroupsPerSupergroup;
    const int table_stride = static_cast<int>(table_pool.size(1));
    auto output = mfq_tensor_backend::empty({M, N}, x.options());
    cudaStream_t stream = mfq_current_cuda_stream();
    const dim3 block(32, 8);
#define NEPQ_GEMM_LAUNCH(MTILES_VALUE, NEED_CHECK_VALUE)                          \
    nepq_gemm_f16_gs24_kernel<F, MTILES_VALUE, NEED_CHECK_VALUE><<<              \
        dim3((N + 63) / 64, (M + 16 * MTILES_VALUE - 1) / (16 * MTILES_VALUE)),  \
        block, 0, stream>>>(                                                       \
        indices.data_ptr<uint8_t>(), indices.numel(),                             \
        aux.data_ptr<uint8_t>(), aux.numel(),                                     \
        sub_scale.data_ptr<uint8_t>(), sub_scale.numel(),                         \
        neuron_scale.data_ptr<float>(), table_pool.data_ptr<int8_t>(),            \
        bank_ids.data_ptr<uint8_t>(),                                             \
        reinterpret_cast<const __half *>(x.data_ptr<mfq_half>()),                 \
        reinterpret_cast<__half *>(output.data_ptr<mfq_half>()),                  \
        M, N, K, ng, nvec, nsign, nsuper, table_stride,                           \
        static_cast<int>(sub_bits), 0)
    launch_nepq_by_format(static_cast<int>(format), [&](auto tag) {
        constexpr int F = decltype(tag)::value;
        if (M <= 16) {
            if (M == 16 && N % 64 == 0 && ng % 4 == 0) NEPQ_GEMM_LAUNCH(1, false);
            else NEPQ_GEMM_LAUNCH(1, true);
        } else if (M <= 32) {
            if (M == 32 && N % 64 == 0 && ng % 4 == 0) NEPQ_GEMM_LAUNCH(2, false);
            else NEPQ_GEMM_LAUNCH(2, true);
        } else if (M <= 64) {
            if (M == 64 && N % 64 == 0 && ng % 4 == 0) NEPQ_GEMM_LAUNCH(4, false);
            else NEPQ_GEMM_LAUNCH(4, true);
        } else {
            if (M % 128 == 0 && N % 64 == 0 && ng % 4 == 0) {
                NEPQ_GEMM_LAUNCH(8, false);
            } else {
                NEPQ_GEMM_LAUNCH(8, true);
            }
        }
    });
#undef NEPQ_GEMM_LAUNCH
    MFQ_RUNTIME_CHECK(cudaGetLastError() == cudaSuccess, "NEPQ online GEMM kernel launch failed");
    return output;
}

mfq_tensor_backend::Tensor nepq_moe_grouped_matmul_ws_cuda(
    mfq_tensor_backend::Tensor indices,
    mfq_tensor_backend::Tensor aux,
    mfq_tensor_backend::Tensor sub_scale,
    mfq_tensor_backend::Tensor neuron_scale,
    mfq_tensor_backend::Tensor table_pool,
    mfq_tensor_backend::Tensor bank_ids,
    mfq_tensor_backend::Tensor grouped_table_pool,
    mfq_tensor_backend::Tensor x,
    mfq_tensor_backend::Tensor ids,
    int64_t n_experts,
    int64_t out_per_expert,
    int64_t neuron_len,
    int64_t sub_bits,
    int64_t format,
    mfq_tensor_backend::Tensor out,
    mfq_tensor_backend::Tensor qx,
    mfq_tensor_backend::Tensor xscale,
    mfq_tensor_backend::Tensor ids_dst,
    mfq_tensor_backend::Tensor expert_bounds,
    mfq_tensor_backend::Tensor tile_bounds,
    mfq_tensor_backend::Tensor tile_experts) {
    check_nepq_common(
        indices, aux, sub_scale, neuron_scale, table_pool, bank_ids,
        neuron_len, sub_bits, format);
    const int expected_grouped_stride = format == kNpq0S
        ? kNpq0STableBytes
        : static_cast<int>(table_pool.size(1));
    MFQ_RUNTIME_CHECK(
        grouped_table_pool.is_cuda() && grouped_table_pool.is_contiguous() &&
        grouped_table_pool.scalar_type() == mfq_tensor_backend::kInt8 &&
        grouped_table_pool.dim() == 2 &&
        grouped_table_pool.size(0) == table_pool.size(0) &&
        grouped_table_pool.size(1) == expected_grouped_stride,
        "NEPQ grouped table pool shape mismatch");
    MFQ_RUNTIME_CHECK(n_experts > 0 && n_experts <= 4096,
                "NEPQ expert count must be in [1,4096]");
    MFQ_RUNTIME_CHECK(out_per_expert > 0 &&
                neuron_scale.numel() == n_experts * out_per_expert,
                "NEPQ expert shape does not match the flattened row count");
    MFQ_RUNTIME_CHECK(ids.is_cuda() && ids.is_contiguous() &&
                ids.scalar_type() == mfq_tensor_backend::kInt32 && ids.dim() == 2,
                "NEPQ route IDs must be CUDA contiguous int32 [T,R]");
    MFQ_RUNTIME_CHECK(x.is_cuda() && x.is_contiguous() &&
                x.scalar_type() == mfq_tensor_backend::kFloat16 &&
                (x.dim() == 2 || x.dim() == 3) && x.size(-1) == neuron_len,
                "NEPQ routed input must be CUDA contiguous fp16 [T,K] or [T,R,K]");
    const int tokens = static_cast<int>(ids.size(0));
    const int routes = static_cast<int>(ids.size(1));
    const int pairs = tokens * routes;
    const bool routed_input = x.dim() == 3;
    MFQ_RUNTIME_CHECK(tokens > 0 && routes > 0, "NEPQ route dimensions must be nonzero");
    MFQ_RUNTIME_CHECK((!routed_input && x.size(0) == tokens) ||
                (routed_input && x.size(0) == tokens && x.size(1) == routes),
                "NEPQ input leading dimensions do not match routes");
    MFQ_RUNTIME_CHECK(out.is_cuda() && out.is_contiguous() &&
                out.scalar_type() == mfq_tensor_backend::kFloat16 && out.dim() == 3 &&
                out.size(0) == tokens && out.size(1) == routes &&
                out.size(2) == out_per_expert,
                "NEPQ output must be CUDA contiguous fp16 [T,R,O]");

    const int K = static_cast<int>(neuron_len);
    const int ng = (K + kGroupSize - 1) / kGroupSize;
    const int K_pad = ng * kGroupSize;
    const int input_rows = routed_input ? pairs : tokens;
    MFQ_RUNTIME_CHECK(qx.is_cuda() && qx.is_contiguous() &&
                qx.scalar_type() == mfq_tensor_backend::kInt8 && qx.dim() == 2 &&
                qx.size(0) >= input_rows && qx.size(1) >= K_pad,
                "NEPQ routed qx workspace mismatch");
    MFQ_RUNTIME_CHECK(xscale.is_cuda() && xscale.is_contiguous() &&
                xscale.scalar_type() == mfq_tensor_backend::kFloat32 && xscale.dim() == 2 &&
                xscale.size(0) >= input_rows && xscale.size(1) >= ng,
                "NEPQ routed xscale workspace mismatch");
    const int nvec = K / 8;
    const int nsign = (K + 7) / 8;
    const int nsuper = (ng + kNepqGroupsPerSupergroup - 1) /
        kNepqGroupsPerSupergroup;
    const int table_stride = static_cast<int>(table_pool.size(1));
    const int grouped_table_stride =
        static_cast<int>(grouped_table_pool.size(1));
    cudaStream_t stream = mfq_current_cuda_stream();
    nvq_quantize_x_gs24_kernel<<<dim3(input_rows, ng), 32, 0, stream>>>(
        reinterpret_cast<const __half *>(x.data_ptr<mfq_half>()),
        qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), input_rows, K, ng);

    if (tokens <= 8) {
        constexpr int rows_per_block = 2;
        const int row_blocks = (static_cast<int>(out_per_expert) + rows_per_block - 1) /
            rows_per_block;
#define NEPQ_MOE_SMALL_LAUNCH(NWARPS_VALUE)                                      \
        nepq_moe_mmvq_kernel<F, NWARPS_VALUE, rows_per_block><<<                 \
            dim3(row_blocks, pairs), dim3(32, NWARPS_VALUE), 0, stream>>>(       \
            indices.data_ptr<uint8_t>(), indices.numel(),                        \
            aux.data_ptr<uint8_t>(), aux.numel(),                                \
            sub_scale.data_ptr<uint8_t>(), sub_scale.numel(),                    \
            neuron_scale.data_ptr<float>(), table_pool.data_ptr<int8_t>(),       \
            bank_ids.data_ptr<uint8_t>(), qx.data_ptr<int8_t>(),                 \
            xscale.data_ptr<float>(), ids.data_ptr<int32_t>(),                   \
            nullptr,                                                             \
            reinterpret_cast<__half *>(out.data_ptr<mfq_half>()),                \
            pairs, routes, static_cast<int>(n_experts),                          \
            static_cast<int>(n_experts),                                         \
            static_cast<int>(out_per_expert), ng, nvec, nsign, nsuper,           \
            table_stride, static_cast<int>(sub_bits), routed_input)
        launch_nepq_by_format(static_cast<int>(format), [&](auto tag) {
            constexpr int F = decltype(tag)::value;
            if (K >= 4096) NEPQ_MOE_SMALL_LAUNCH(8);
            else NEPQ_MOE_SMALL_LAUNCH(4);
        });
#undef NEPQ_MOE_SMALL_LAUNCH
    } else {
        MFQ_RUNTIME_CHECK(ids_dst.is_cuda() && ids_dst.is_contiguous() &&
                    ids_dst.scalar_type() == mfq_tensor_backend::kInt32 && ids_dst.numel() >= pairs,
                    "NEPQ compact route map is missing");
        MFQ_RUNTIME_CHECK(expert_bounds.is_cuda() && expert_bounds.is_contiguous() &&
                    expert_bounds.scalar_type() == mfq_tensor_backend::kInt32 &&
                    expert_bounds.numel() >= n_experts + 1,
                    "NEPQ expert bounds are missing");
        MFQ_RUNTIME_CHECK(tile_bounds.is_cuda() && tile_bounds.is_contiguous() &&
                    tile_bounds.scalar_type() == mfq_tensor_backend::kInt32 &&
                    tile_bounds.numel() >= n_experts + 1,
                    "NEPQ tile bounds are missing");
        MFQ_RUNTIME_CHECK(tile_experts.is_cuda() && tile_experts.is_contiguous() &&
                    tile_experts.scalar_type() == mfq_tensor_backend::kInt32 &&
                    tile_experts.numel() >= pairs,
                    "NEPQ tile expert map is missing");
        constexpr int route_tile = 8;
        const int row_tiles = (static_cast<int>(out_per_expert) + 3) / 4;
        const int max_tiles = (pairs + route_tile - 1) / route_tile +
            static_cast<int>(n_experts);
        const int64_t max_tasks = static_cast<int64_t>(max_tiles) * row_tiles;
        const int blocks = static_cast<int>(std::min<int64_t>(max_tasks, 4096));
#define NEPQ_MOE_GROUPED_LAUNCH()                                                \
        nepq_moe_grouped_tile_kernel<F, route_tile><<<                           \
            blocks, dim3(32, 4), 0, stream>>>(                                   \
            indices.data_ptr<uint8_t>(), indices.numel(),                        \
            aux.data_ptr<uint8_t>(), aux.numel(),                                \
            sub_scale.data_ptr<uint8_t>(), sub_scale.numel(),                    \
            neuron_scale.data_ptr<float>(), grouped_table_pool.data_ptr<int8_t>(), \
            bank_ids.data_ptr<uint8_t>(), qx.data_ptr<int8_t>(),                 \
            xscale.data_ptr<float>(), nullptr, ids_dst.data_ptr<int32_t>(),      \
            expert_bounds.data_ptr<int32_t>(), tile_bounds.data_ptr<int32_t>(),  \
            tile_experts.data_ptr<int32_t>(),                                    \
            reinterpret_cast<__half *>(out.data_ptr<mfq_half>()),                \
            routes, static_cast<int>(n_experts),                                 \
            static_cast<int>(n_experts),                                         \
            static_cast<int>(out_per_expert), ng, nvec, nsign, nsuper,           \
            grouped_table_stride, static_cast<int>(sub_bits), max_tiles,         \
            row_tiles,                                                           \
            routed_input)
        launch_nepq_by_format(static_cast<int>(format), [&](auto tag) {
            constexpr int F = decltype(tag)::value;
            NEPQ_MOE_GROUPED_LAUNCH();
        });
#undef NEPQ_MOE_GROUPED_LAUNCH
    }
    MFQ_RUNTIME_CHECK(cudaGetLastError() == cudaSuccess,
                "NEPQ routed matmul kernel launch failed");
    return out;
}

// Route-compacted online-dequant FP16 Tensor Core path for mixed NVQ pools.
// Each task reuses one 64/128-column weight tile across 16 to 256 routed rows.
// Keep the format switch outside the inner K loop so one launch can serve
// every NVQ cohort without a runtime branch per decoded value.
constexpr int kNvqMoeF16PoolTileN = 64;
constexpr int kNvqMoeF16HeteroTileN = 128;
constexpr int kNvqMoeF16AllFormats = 0;
constexpr int kNvqMoeF16StandardFormats = 1;
constexpr int kNvqMoeF16ExtendedFormats = 2;
constexpr int kNvqMoeF16LegacyFormats = 3;

template <int GROUPS_PER_CHUNK>
struct NvqMoeF16KLayout {
    static constexpr int kTileK = kGroupSize * GROUPS_PER_CHUNK;
    static_assert(kTileK % 16 == 0);
    static constexpr int kStrideK = kTileK + 8;
};

template <int BM, int BN, int GROUPS_PER_CHUNK>
struct __align__(16) NvqMoeF16SharedStorage {
    static constexpr int kStrideK =
        NvqMoeF16KLayout<GROUPS_PER_CHUNK>::kStrideK;
    static constexpr int kOperandBytes =
        (BN + BM) * kStrideK * sizeof(__half);
    static constexpr int kOutputBytes = BM >= 128
        ? (BM / 64) * BN * 16 * sizeof(float)
        : (BN / 16) * BM * 16 * sizeof(float);
    static constexpr int kBytes =
        kOperandBytes > kOutputBytes ? kOperandBytes : kOutputBytes;
    uint8_t bytes[kBytes];
    int32_t pair_rows[BM];
    int32_t source_rows[BM];
};

static_assert(sizeof(NvqMoeF16SharedStorage<64, 128, 4>) <= 48 * 1024);
static_assert(sizeof(NvqMoeF16SharedStorage<128, 128, 2>) <= 48 * 1024);
static_assert(sizeof(NvqMoeF16SharedStorage<128, 64, 4>) <= 48 * 1024);

template <bool ASYNC_COPY, int BM, int GROUPS_PER_CHUNK,
          int THREAD_WARPS>
__device__ __forceinline__ void nvq_moe_load_activation_tile(
    const __half * x,
    const int32_t * source_rows,
    __half * activation_tile,
    int k_base,
    int K,
    int tid) {
    constexpr int kStrideK =
        NvqMoeF16KLayout<GROUPS_PER_CHUNK>::kStrideK;
    constexpr int kTileK =
        NvqMoeF16KLayout<GROUPS_PER_CHUNK>::kTileK;
    constexpr int kThreads = THREAD_WARPS * 32;
    constexpr int kVectorWidth = 8;
    static_assert(kTileK % kVectorWidth == 0);
    static_assert(kThreads % BM == 0);
    constexpr int kVectorsPerRow = kTileK / kVectorWidth;
    constexpr int kLanesPerRow = kThreads / BM;
    const int m_local = tid / kLanesPerRow;
    const int vector_lane = tid - m_local * kLanesPerRow;
    const int source_row = source_rows[m_local];
#pragma unroll
    for (int vector_local = vector_lane;
         vector_local < kVectorsPerRow;
         vector_local += kLanesPerRow) {
        const int k_local = vector_local * kVectorWidth;
        const int k = k_base + k_local;
        __half * destination =
            activation_tile + m_local * kStrideK + k_local;
        if (source_row < 0 || k >= K) {
            *reinterpret_cast<int4 *>(destination) =
                make_int4(0, 0, 0, 0);
        } else if ((K & 7) == 0 && k + 7 < K) {
            const __half * source =
                x + static_cast<int64_t>(source_row) * K + k;
            if constexpr (ASYNC_COPY) {
                mfq::cuda_detail::copy_16_async(destination, source);
            } else {
                *reinterpret_cast<int4 *>(destination) =
                    *reinterpret_cast<const int4 *>(source);
            }
        } else {
#pragma unroll
            for (int element = 0; element < kVectorWidth; ++element) {
                destination[element] = k + element < K
                    ? x[static_cast<int64_t>(source_row) * K + k + element]
                    : __float2half(0.0f);
            }
        }
    }
}

template <int FORMAT, int BM, int BN, int GROUPS_PER_CHUNK,
          int THREAD_WARPS, bool ASYNC_ACTIVATION = false>
__device__ __forceinline__ void nvq_moe_grouped_f16_task(
    const uint8_t * indices,
    int64_t indices_nbytes,
    const uint8_t * aux,
    int64_t aux_nbytes,
    const uint8_t * sub_scale,
    int64_t sub_scale_nbytes,
    const float * neuron_scale,
    const int8_t * codebook,
    const __half * x,
    const int32_t * ids_dst,
    __half * output,
    __half * weight_tile,
    __half * activation_tile,
    float * output_tile,
    int32_t * pair_rows,
    int32_t * source_rows,
    int routes,
    int out_per_expert,
    int K,
    int ng,
    int nvec,
    int nsign,
    int sub_bits,
    int sign_mode,
    bool routed_input,
    int local_expert,
    int first,
    int last,
    int n0) {
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid = warp * 32 + lane;
    using FragmentA = wmma::fragment<wmma::matrix_a, 16, 16, 16,
                                     __half, wmma::row_major>;
    using FragmentB = wmma::fragment<wmma::matrix_b, 16, 16, 16,
                                     __half, wmma::col_major>;
    using FragmentC = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

    static_assert(BM == 16 || BM == 32 || BM == 64 || BM == 128 || BM == 256);
    static_assert(BN == 64 || BN == 128);
    static_assert(GROUPS_PER_CHUNK == 2 || GROUPS_PER_CHUNK == 4);
    constexpr int kMFragments = BM / 16;
    constexpr int kNFragments = BN / 16;
    constexpr int kWarps = THREAD_WARPS;
    constexpr int kRowTeams = BM >= 128 ? BM / 64 : 1;
    constexpr int kComputeWarps = kRowTeams * kNFragments;
    static_assert(kWarps >= kComputeWarps);
    constexpr int kAccumulatorFragments = kMFragments / kRowTeams;
    constexpr int kThreads = kWarps * 32;
    constexpr int kTileK =
        NvqMoeF16KLayout<GROUPS_PER_CHUNK>::kTileK;
    constexpr int kStrideK =
        NvqMoeF16KLayout<GROUPS_PER_CHUNK>::kStrideK;
    if (tid < BM) {
        const int compact = first + tid;
        const int pair_index = compact < last ? ids_dst[compact] : -1;
        pair_rows[tid] = pair_index;
        source_rows[tid] = pair_index < 0
            ? -1 : (routed_input ? pair_index : pair_index / routes);
    }
    __syncthreads();
    FragmentC accumulators[kAccumulatorFragments];
#pragma unroll
    for (int fragment = 0;
         fragment < kAccumulatorFragments;
         ++fragment) {
        wmma::fill_fragment(accumulators[fragment], 0.0f);
    }
    const int chunks =
        (ng + GROUPS_PER_CHUNK - 1) / GROUPS_PER_CHUNK;
    for (int chunk = 0; chunk < chunks; ++chunk) {
        const int group_base = chunk * GROUPS_PER_CHUNK;
        const int k_base = group_base * kGroupSize;
        if constexpr (ASYNC_ACTIVATION) {
            nvq_moe_load_activation_tile<
                true, BM, GROUPS_PER_CHUNK, THREAD_WARPS>(
                x, source_rows, activation_tile, k_base, K, tid);
            mfq::cuda_detail::copy_async_commit();
        }
        constexpr int kWeightStates = BN * GROUPS_PER_CHUNK;
        // BM256 doubles the activation readers. Keep both warp teams useful
        // during dequantization by sharing the three vectors of each group.
        constexpr int kDecodeLanes = BM == 256 ? 2 : 1;
        constexpr int kWeightThreads = kThreads / kDecodeLanes;
        const int segment_lane = BM == 256 ? tid / kWeightThreads : 0;
#pragma unroll
        for (int weight_index = BM == 256 ? tid % kWeightThreads : tid;
             weight_index < kWeightStates;
            weight_index += kWeightThreads) {
            const int row_local =
                weight_index / GROUPS_PER_CHUNK;
            const int group_local = weight_index -
                row_local * GROUPS_PER_CHUNK;
            const int local_row = n0 + row_local;
            const int row = local_expert * out_per_expert + local_row;
            const int group = group_base + group_local;
            const bool valid_weight =
                local_row < out_per_expert && group < ng;
            DenseNvqGroup dense_record;
            const bool dense=dense_nvq_format(FORMAT) && aux_nbytes==0;
            uint32_t state = 0;
            float scale = 0.0f;
            uint32_t packed_signs = 0;
            uint64_t group_exec64 = 0;
            uint32_t group_exec96[3] = {0, 0, 0};
            int group_delta = 1;
            const int8_t * bank = codebook;
            if (valid_weight) {
                const int64_t sub_linear =
                    static_cast<int64_t>(row) * ng + group;
                if constexpr(dense_nvq_format(FORMAT)) {
                    if(dense) {
                        dense_record=load_dense_nvq_group<FORMAT>(indices,indices_nbytes,row,group,nsign,ng);
                        state=dense_record.state();
                    }else state=load_nvq_state<FORMAT>(indices,indices_nbytes,sub_scale,sub_scale_nbytes,
                        sub_linear,sub_bits,ng);
                }else state=load_nvq_state<FORMAT>(indices,indices_nbytes,sub_scale,sub_scale_nbytes,
                    sub_linear,sub_bits,ng);
                scale = format_scale<FORMAT>(
                    neuron_scale[row], state, codebook);
                if constexpr (is_e8_format(FORMAT) || is_d4_format(FORMAT)) {
                    bank = active_codebook<FORMAT>(codebook, state);
                }
                if constexpr (FORMAT == kNvq2JscXLGroupExec) {
                    group_exec64 = load_group_exec64(
                        indices, row, group, ng);
                } else if constexpr (FORMAT == kNvq3JscLGroupExec) {
                    load_group_exec96_words(
                        indices, row, group, ng, group_exec96);
                } else if constexpr (
                    FORMAT == kNvq2 || FORMAT == kNvq2Jsc ||
                    FORMAT == kNvq2JscL || FORMAT == kNvq2JscXL ||
                    FORMAT == kNvq3 || FORMAT == kNvq3Jsc ||
                    FORMAT == kNvq3Jsc2 || FORMAT == kNvq3Jsc512 ||
                    FORMAT == kNvq3JscL) {
                    const int64_t sign_linear =
                        static_cast<int64_t>(row) * nsign + group * 3;
                    if(!dense)packed_signs = load_packed_sign_group3(
                        aux, sign_linear * 7, aux_nbytes);
                } else if constexpr (
                    FORMAT == kNvq1S || FORMAT == kNvq1L) {
                    group_delta = load_packed_bits(
                        aux, sub_linear, 1, aux_nbytes) ? -1 : 1;
                    if constexpr (FORMAT == kNvq1S) {
                        bank = codebook +
                            (group_delta < 0 ? kNvq1SBankBytes : 0);
                    }
                }
            }
#pragma unroll
            for (int segment_local = 0;
                 segment_local < kChunksPerGroup / 2;
                 ++segment_local) {
                if constexpr (BM == 256) {
                    if (segment_local % kDecodeLanes != segment_lane) continue;
                }
                int2 packed = make_int2(0, 0);
                if constexpr(dense_nvq_format(FORMAT))if(valid_weight && dense && group*3+segment_local<nsign)
                    packed=decode_dense_nvq_vector<FORMAT>(dense_record,bank,segment_local,sign_mode);
                if (valid_weight && !dense) {
                    const auto decoded = load_nvq_group_vec8<FORMAT>(
                        indices, indices_nbytes, aux, aux_nbytes,
                        codebook, bank, row, group, segment_local,
                        ng, nvec, nsign, sign_mode, state,
                        packed_signs, group_exec64, group_exec96,
                        group_delta);
                    packed = decoded.values;
                    if constexpr (FORMAT == kNvq1S) {
                        packed.x = nvq1_s_scale_delta4(
                            packed.x, decoded.delta);
                        packed.y = nvq1_s_scale_delta4(
                            packed.y, decoded.delta);
                    } else if constexpr (FORMAT == kNvq1L) {
                        packed.x = nvq1_l_scale_delta4(
                            packed.x, decoded.delta);
                        packed.y = nvq1_l_scale_delta4(
                            packed.y, decoded.delta);
                    }
                }
#pragma unroll
                for (int pair = 0; pair < 4; ++pair) {
                    const int word = pair < 2 ? packed.x : packed.y;
                    const int shift = (pair & 1) * 16;
                    const int value0 = static_cast<int>(
                        static_cast<int8_t>((word >> shift) & 0xff));
                    const int value1 = static_cast<int>(
                        static_cast<int8_t>(
                            (word >> (shift + 8)) & 0xff));
                    *reinterpret_cast<__half2 *>(
                        weight_tile + row_local * kStrideK +
                            group_local * kGroupSize +
                            segment_local * 8 + pair * 2) =
                        __halves2half2(
                            __float2half(
                                scale * static_cast<float>(value0)),
                            __float2half(
                                scale * static_cast<float>(value1)));
                }
            }
        }

        if constexpr (ASYNC_ACTIVATION) {
            mfq::cuda_detail::copy_async_wait();
        } else {
            nvq_moe_load_activation_tile<
                false, BM, GROUPS_PER_CHUNK, THREAD_WARPS>(
                x, source_rows, activation_tile, k_base, K, tid);
        }
        __syncthreads();

        if (warp < kComputeWarps) {
            const int warp_n = BM >= 128 ? warp % kNFragments : warp;
            const int m_fragment_base = BM >= 128
                ? warp / kNFragments : 0;
#pragma unroll
            for (int k_local = 0;
                 k_local < kTileK;
                 k_local += 16) {
                if (BM < 64 || k_base + k_local < K) {
                    FragmentB weight_fragment;
                    wmma::load_matrix_sync(
                        weight_fragment,
                        weight_tile + warp_n * 16 * kStrideK + k_local,
                        kStrideK);
#pragma unroll
                    for (int fragment = 0;
                         fragment < kAccumulatorFragments;
                         ++fragment) {
                        const int m_fragment = m_fragment_base +
                            fragment * kRowTeams;
                        if constexpr (BM == 256) {
                            // Uniform within the warp; empty fragments have no
                            // output rows, and need no Tensor Core work.
                            if (m_fragment * 16 >= last - first) continue;
                        }
                        FragmentA activation_fragment;
                        wmma::load_matrix_sync(
                            activation_fragment,
                            activation_tile +
                                m_fragment * 16 * kStrideK + k_local,
                            kStrideK);
                        wmma::mma_sync(
                            accumulators[fragment], activation_fragment,
                            weight_fragment, accumulators[fragment]);
                    }
                }
            }
        }
        __syncthreads();
    }

    if constexpr (BM >= 128) {
        if (warp < kComputeWarps) {
            const int warp_n = warp % kNFragments;
            const int m_fragment_base = warp / kNFragments;
#pragma unroll
            for (int fragment = 0;
                 fragment < kAccumulatorFragments;
                 ++fragment) {
                const int m_fragment =
                    m_fragment_base + fragment * kRowTeams;
                wmma::store_matrix_sync(
                    output_tile + warp * 16 * 16,
                    accumulators[fragment],
                    16, wmma::mem_row_major);
                __syncwarp();
                for (int element = lane; element < 16 * 16; element += 32) {
                    const int m_local = m_fragment * 16 + element / 16;
                    const int n_local = element % 16;
                    const int local_row = n0 + warp_n * 16 + n_local;
                    const int pair_index = pair_rows[m_local];
                    if (pair_index >= 0 && local_row < out_per_expert) {
                        output[
                            static_cast<int64_t>(pair_index) * out_per_expert +
                            local_row] = __float2half(
                                output_tile[warp * 16 * 16 + element]);
                    }
                }
                __syncwarp();
            }
        }
    } else {
        if (warp < kNFragments) {
#pragma unroll
            for (int m_fragment = 0;
                 m_fragment < kMFragments;
                 ++m_fragment) {
                wmma::store_matrix_sync(
                    output_tile +
                        (warp * BM + m_fragment * 16) * 16,
                    accumulators[m_fragment],
                    16, wmma::mem_row_major);
            }
        }
        __syncthreads();
        if (warp < kNFragments) {
            for (int element = lane; element < BM * 16; element += 32) {
                const int m_local = element / 16;
                const int n_local = element % 16;
                const int local_row = n0 + warp * 16 + n_local;
                const int pair_index = pair_rows[m_local];
                if (pair_index >= 0 && local_row < out_per_expert) {
                    output[
                        static_cast<int64_t>(pair_index) * out_per_expert +
                        local_row] = __float2half(
                            output_tile[
                                (warp * BM + m_local) * 16 + n_local]);
                }
            }
        }
    }
    __syncthreads();
}

template <int FORMAT, int BM = 16>
__global__ void __launch_bounds__(256, 1) nvq_moe_grouped_f16_kernel(
    const uint8_t * indices,
    int64_t indices_nbytes,
    const uint8_t * aux,
    int64_t aux_nbytes,
    const uint8_t * sub_scale,
    int64_t sub_scale_nbytes,
    const float * neuron_scale,
    const int8_t * codebook,
    const __half * x,
    const int32_t * expert_local,
    const int32_t * ids_dst,
    const int32_t * expert_bounds,
    const int32_t * tile_bounds,
    const int32_t * tile_experts,
    __half * output,
    int routes,
    int global_experts,
    int pool_experts,
    int out_per_expert,
    int K,
    int ng,
    int nvec,
    int nsign,
    int sub_bits,
    int sign_mode,
    int max_tiles,
    bool routed_input) {
    constexpr int BN = kNvqMoeF16PoolTileN;
    constexpr int GROUPS_PER_CHUNK = 4;
    constexpr int kStrideK =
        NvqMoeF16KLayout<GROUPS_PER_CHUNK>::kStrideK;
    __shared__ NvqMoeF16SharedStorage<
        BM, BN, GROUPS_PER_CHUNK> shared;
    auto * weight_tile = reinterpret_cast<__half *>(shared.bytes);
    auto * activation_tile = weight_tile + BN * kStrideK;
    auto * output_tile = reinterpret_cast<float *>(shared.bytes);
    const int ntiles_n =
        (out_per_expert + BN - 1) / BN;
    const int64_t max_tasks =
        static_cast<int64_t>(max_tiles) * ntiles_n;
    for (int64_t task = blockIdx.x; task < max_tasks; task += gridDim.x) {
        const int fine_tile = static_cast<int>(task / ntiles_n);
        const int ntile = static_cast<int>(
            task - static_cast<int64_t>(fine_tile) * ntiles_n);
        if (fine_tile >= tile_bounds[global_experts]) continue;
        const int expert = tile_experts[fine_tile];
        const int local_fine_tile = fine_tile - tile_bounds[expert];
        if (local_fine_tile % (BM / 8) != 0) continue;
        const int local_expert = expert_local[expert];
        if (static_cast<unsigned int>(local_expert) >=
            static_cast<unsigned int>(pool_experts)) continue;
        const int first = expert_bounds[expert] + local_fine_tile * 8;
        const int last = min(
            first + BM, expert_bounds[expert + 1]);
        nvq_moe_grouped_f16_task<
            FORMAT, BM, BN, GROUPS_PER_CHUNK,
            (BM == 128 ? 16 : 8)>(
            indices, indices_nbytes, aux, aux_nbytes,
            sub_scale, sub_scale_nbytes, neuron_scale, codebook,
            x, ids_dst, output, weight_tile,
            activation_tile, output_tile, shared.pair_rows,
            shared.source_rows, routes,
            out_per_expert, K, ng, nvec, nsign, sub_bits, sign_mode,
            routed_input, local_expert, first, last,
            ntile * BN);
    }
}

template <int BM, int BN, int ROUTE_TILE_M, int GROUPS_PER_CHUNK,
          int FORMAT_GROUP, bool ASYNC_ACTIVATION = false>
__global__ void __launch_bounds__(
    (BM >= 128 ? BM / 64 : 1) * (BN / 16) * 32,
    BM == 128 && BN == 64 && GROUPS_PER_CHUNK == 2 &&
        FORMAT_GROUP != kNvqMoeF16AllFormats ? 3 :
    BM < 256 && (BN == 64 ||
        (FORMAT_GROUP != kNvqMoeF16AllFormats && GROUPS_PER_CHUNK == 2)
        ) ? 2 : 1)
nvq_moe_grouped_hetero_f16_kernel(
    const int64_t * weight_ptrs,
    const int64_t * weight_sizes,
    const int32_t * pool_params,
    const int32_t * expert_pool,
    const int32_t * expert_local,
    const __half * x,
    const int32_t * ids_dst,
    const int32_t * expert_bounds,
    const int32_t * tile_bounds,
    const int32_t * tile_experts,
    __half * output,
    int routes,
    int global_experts,
    int pool_count,
    int out_per_expert,
    int K,
    int max_tiles,
    bool routed_input) {
    static_assert(BM % ROUTE_TILE_M == 0);
    constexpr int kStrideK =
        NvqMoeF16KLayout<GROUPS_PER_CHUNK>::kStrideK;
    extern __shared__ __align__(16) uint8_t shared_bytes[];
    auto & shared = *reinterpret_cast<NvqMoeF16SharedStorage<
        BM, BN, GROUPS_PER_CHUNK> *>(shared_bytes);
    auto * weight_tile = reinterpret_cast<__half *>(shared.bytes);
    auto * activation_tile = weight_tile + BN * kStrideK;
    auto * output_tile = reinterpret_cast<float *>(shared.bytes);
    const int ntiles_n =
        (out_per_expert + BN - 1) / BN;
    const int64_t max_tasks =
        static_cast<int64_t>(max_tiles) * ntiles_n;
    for (int64_t task = blockIdx.x; task < max_tasks; task += gridDim.x) {
        const int fine_tile = static_cast<int>(task / ntiles_n);
        const int ntile = static_cast<int>(
            task - static_cast<int64_t>(fine_tile) * ntiles_n);
        if (fine_tile >= tile_bounds[global_experts]) continue;
        const int expert = tile_experts[fine_tile];
        const int local_fine_tile = fine_tile - tile_bounds[expert];
        if (local_fine_tile % (BM / ROUTE_TILE_M) != 0) continue;
        const int pool = expert_pool[expert];
        const int local_expert = expert_local[expert];
        if (static_cast<unsigned int>(pool) >=
            static_cast<unsigned int>(pool_count) || local_expert < 0) {
            continue;
        }
        const int64_t * pointers = weight_ptrs + static_cast<size_t>(pool) * 5;
        const int64_t * sizes = weight_sizes + static_cast<size_t>(pool) * 3;
        const int32_t * params = pool_params + static_cast<size_t>(pool) * 7;
        const int pool_experts = params[0];
        if (local_expert >= pool_experts) continue;
        const int first =
            expert_bounds[expert] + local_fine_tile * ROUTE_TILE_M;
        const int last = min(
            first + BM, expert_bounds[expert + 1]);
        const uint8_t * indices = reinterpret_cast<const uint8_t *>(
            static_cast<uintptr_t>(pointers[0]));
        const uint8_t * aux = reinterpret_cast<const uint8_t *>(
            static_cast<uintptr_t>(pointers[1]));
        const uint8_t * sub_scale = reinterpret_cast<const uint8_t *>(
            static_cast<uintptr_t>(pointers[2]));
        const float * neuron_scale = reinterpret_cast<const float *>(
            static_cast<uintptr_t>(pointers[3]));
        const int8_t * codebook = reinterpret_cast<const int8_t *>(
            static_cast<uintptr_t>(pointers[4]));
        const int ng = params[1];
        const int nvec = params[2];
        const int nsign = params[3];
        const int sub_bits = params[4];
        const int sign_mode = params[5];
        const int format = params[6];

#define NVQ_MOE_HETERO_F16_CASE(FORMAT_VALUE)                                  \
        case FORMAT_VALUE:                                                      \
            nvq_moe_grouped_f16_task<                                           \
                FORMAT_VALUE, BM, BN, GROUPS_PER_CHUNK,                         \
                (BM >= 128 ? BM / 64 : 1) * (BN / 16), ASYNC_ACTIVATION>(        \
                indices, sizes[0], aux, sizes[1], sub_scale, sizes[2],          \
                neuron_scale, codebook, x, ids_dst, output,                     \
                weight_tile, activation_tile, output_tile,                     \
                shared.pair_rows, shared.source_rows, routes,                   \
                out_per_expert, K, ng, nvec,                                    \
                nsign, sub_bits, sign_mode, routed_input, local_expert, first,  \
                last, ntile * BN);                                              \
            break
        if constexpr (FORMAT_GROUP == kNvqMoeF16StandardFormats) {
            switch (format) {
                NVQ_MOE_HETERO_F16_CASE(kNvq2);
                NVQ_MOE_HETERO_F16_CASE(kNvq3);
                NVQ_MOE_HETERO_F16_CASE(kNvq2Exec);
                NVQ_MOE_HETERO_F16_CASE(kNvq2Jsc);
                NVQ_MOE_HETERO_F16_CASE(kNvq2JscExec);
                NVQ_MOE_HETERO_F16_CASE(kNvq3Jsc);
                NVQ_MOE_HETERO_F16_CASE(kNvq3Jsc2);
            }
        } else if constexpr (
            FORMAT_GROUP == kNvqMoeF16ExtendedFormats) {
            switch (format) {
                NVQ_MOE_HETERO_F16_CASE(kNvq3Jsc512);
                NVQ_MOE_HETERO_F16_CASE(kNvq2JscL);
                NVQ_MOE_HETERO_F16_CASE(kNvq2JscXL);
                NVQ_MOE_HETERO_F16_CASE(kNvq3JscL);
                NVQ_MOE_HETERO_F16_CASE(kNvq2JscXLGroupExec);
                NVQ_MOE_HETERO_F16_CASE(kNvq3JscLGroupExec);
            }
        } else if constexpr (
            FORMAT_GROUP == kNvqMoeF16LegacyFormats) {
            switch (format) {
                NVQ_MOE_HETERO_F16_CASE(kNvq1L);
                NVQ_MOE_HETERO_F16_CASE(kNpq0L);
                NVQ_MOE_HETERO_F16_CASE(kNvq1S);
                NVQ_MOE_HETERO_F16_CASE(kNpq0S);
            }
        } else {
            switch (format) {
                NVQ_MOE_HETERO_F16_CASE(kNvq1L);
                NVQ_MOE_HETERO_F16_CASE(kNvq2);
                NVQ_MOE_HETERO_F16_CASE(kNvq3);
                NVQ_MOE_HETERO_F16_CASE(kNvq2Exec);
                NVQ_MOE_HETERO_F16_CASE(kNvq2Jsc);
                NVQ_MOE_HETERO_F16_CASE(kNvq2JscExec);
                NVQ_MOE_HETERO_F16_CASE(kNpq0L);
                NVQ_MOE_HETERO_F16_CASE(kNvq1S);
                NVQ_MOE_HETERO_F16_CASE(kNpq0S);
                NVQ_MOE_HETERO_F16_CASE(kNvq3Jsc);
                NVQ_MOE_HETERO_F16_CASE(kNvq3Jsc2);
                NVQ_MOE_HETERO_F16_CASE(kNvq3Jsc512);
                NVQ_MOE_HETERO_F16_CASE(kNvq2JscL);
                NVQ_MOE_HETERO_F16_CASE(kNvq2JscXL);
                NVQ_MOE_HETERO_F16_CASE(kNvq3JscL);
                NVQ_MOE_HETERO_F16_CASE(kNvq2JscXLGroupExec);
                NVQ_MOE_HETERO_F16_CASE(kNvq3JscLGroupExec);
            }
        }
#undef NVQ_MOE_HETERO_F16_CASE
    }
}

template <int FORMAT>
__global__ void __launch_bounds__(256, 1) nepq_moe_grouped_f16_kernel(
    const uint8_t * indices,
    int64_t indices_nbytes,
    const uint8_t * aux,
    int64_t aux_nbytes,
    const uint8_t * sub_scale,
    int64_t sub_scale_nbytes,
    const float * neuron_scale,
    const int8_t * table_pool,
    const uint8_t * bank_ids,
    const __half * x,
    const int32_t * expert_local,
    const int32_t * ids_dst,
    const int32_t * expert_bounds,
    const int32_t * tile_bounds,
    const int32_t * tile_experts,
    __half * output,
    int routes,
    int global_experts,
    int pool_experts,
    int out_per_expert,
    int K,
    int ng,
    int nvec,
    int nsign,
    int nsuper,
    int table_stride,
    int sub_bits,
    int max_tiles,
    bool routed_input) {
    constexpr int kTileM = 16;
    constexpr int kTileN = 64;
    constexpr int kGroupsPerChunk = kNepqGroupsPerSupergroup;
    constexpr int kTileK = kGroupSize * kGroupsPerChunk;
    constexpr int kStrideK = kTileK + 8;
    constexpr int kNFragments = kTileN / 16;
    constexpr int kFineTilesPerTask = kTileM / 8;

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid = warp * 32 + lane;
    const int ntiles_n = (out_per_expert + kTileN - 1) / kTileN;
    const int64_t max_tasks = static_cast<int64_t>(max_tiles) * ntiles_n;
    __shared__ __half weight_tile[kTileN][kStrideK];
    __shared__ __half activation_tile[kTileM][kStrideK];
    __shared__ float output_tile[kNFragments][16][16];

    using FragmentA = wmma::fragment<wmma::matrix_a, 16, 16, 16,
                                     __half, wmma::row_major>;
    using FragmentB = wmma::fragment<wmma::matrix_b, 16, 16, 16,
                                     __half, wmma::col_major>;
    using FragmentC = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

    for (int64_t task = blockIdx.x; task < max_tasks; task += gridDim.x) {
        const int fine_tile = static_cast<int>(task / ntiles_n);
        const int ntile = static_cast<int>(task -
            static_cast<int64_t>(fine_tile) * ntiles_n);
        if (fine_tile >= tile_bounds[global_experts]) continue;
        const int expert = tile_experts[fine_tile];
        const int local_fine_tile = fine_tile - tile_bounds[expert];
        if (local_fine_tile % kFineTilesPerTask != 0) continue;
        const int local_expert = expert_local[expert];
        if (static_cast<unsigned int>(local_expert) >=
            static_cast<unsigned int>(pool_experts)) continue;
        const int first = expert_bounds[expert] + local_fine_tile * 8;
        const int last = min(first + kTileM, expert_bounds[expert + 1]);
        const int n0 = ntile * kTileN;

        FragmentC accumulator;
        wmma::fill_fragment(accumulator, 0.0f);
        const int chunks = (ng + kGroupsPerChunk - 1) / kGroupsPerChunk;
        for (int chunk = 0; chunk < chunks; ++chunk) {
            const int group_base = chunk * kGroupsPerChunk;
            const int k_base = group_base * kGroupSize;
            const int row_local = tid / kGroupsPerChunk;
            const int group_local = tid - row_local * kGroupsPerChunk;
            const int local_row = n0 + row_local;
            const int row = local_expert * out_per_expert + local_row;
            const int group = group_base + group_local;
            const bool valid_weight =
                local_row < out_per_expert && group < ng;
            uint32_t state = 0;
            float scale = 0.0f;
            const int8_t * table = table_pool;
            if (valid_weight) {
                const int64_t sub_linear =
                    static_cast<int64_t>(row) * ng + group;
                state = load_nvq_state<FORMAT>(indices,indices_nbytes,sub_scale,sub_scale_nbytes,
                    sub_linear,sub_bits,ng);
                table = nepq_active_table(
                    table_pool, bank_ids, row, group,
                    nsuper, table_stride);
                scale = format_scale<FORMAT>(
                    neuron_scale[row], state, table);
            }
#pragma unroll
            for (int quartet = 0; quartet < kChunksPerGroup; ++quartet) {
                const int packed = valid_weight
                    ? decode_nepq_chunk4<FORMAT>(
                          indices, indices_nbytes, aux, aux_nbytes, table,
                          row, group, quartet, nvec, nsign, ng,
                          0, state)
                    : 0;
#pragma unroll
                for (int pair = 0; pair < 2; ++pair) {
                    const int shift = pair * 16;
                    const int value0 = static_cast<int>(
                        static_cast<int8_t>((packed >> shift) & 0xff));
                    const int value1 = static_cast<int>(
                        static_cast<int8_t>((packed >> (shift + 8)) & 0xff));
                    *reinterpret_cast<__half2 *>(
                        &weight_tile[row_local]
                            [group_local * kGroupSize + quartet * 4 + pair * 2]) =
                        __halves2half2(
                            __float2half(scale * static_cast<float>(value0)),
                            __float2half(scale * static_cast<float>(value1)));
                }
            }

            constexpr int kActivationPairs = kTileM * (kTileK / 2);
            for (int index = tid; index < kActivationPairs; index += 256) {
                const int m_local = index / (kTileK / 2);
                const int k_pair = index - m_local * (kTileK / 2);
                const int compact = first + m_local;
                const int k = k_base + k_pair * 2;
                __half2 values = __float2half2_rn(0.0f);
                if (compact < last) {
                    const int pair_index = ids_dst[compact];
                    const int source_row = routed_input
                        ? pair_index : pair_index / routes;
                    if (k + 1 < K) {
                        values = *reinterpret_cast<const __half2 *>(
                            x + static_cast<int64_t>(source_row) * K + k);
                    } else if (k < K) {
                        values = __halves2half2(
                            x[static_cast<int64_t>(source_row) * K + k],
                            __float2half(0.0f));
                    }
                }
                *reinterpret_cast<__half2 *>(
                    &activation_tile[m_local][k_pair * 2]) = values;
            }
            __syncthreads();

            if (warp < kNFragments) {
#pragma unroll
                for (int k_local = 0; k_local < kTileK; k_local += 16) {
                    FragmentA activation_fragment;
                    FragmentB weight_fragment;
                    wmma::load_matrix_sync(
                        activation_fragment, &activation_tile[0][k_local],
                        kStrideK);
                    wmma::load_matrix_sync(
                        weight_fragment, &weight_tile[warp * 16][k_local],
                        kStrideK);
                    wmma::mma_sync(
                        accumulator, activation_fragment,
                        weight_fragment, accumulator);
                }
            }
            __syncthreads();
        }

        if (warp < kNFragments) {
            wmma::store_matrix_sync(
                &output_tile[warp][0][0], accumulator,
                16, wmma::mem_row_major);
        }
        __syncthreads();
        if (warp < kNFragments) {
            for (int element = lane; element < 256; element += 32) {
                const int m_local = element / 16;
                const int n_local = element - m_local * 16;
                const int compact = first + m_local;
                const int local_row = n0 + warp * 16 + n_local;
                if (compact < last && local_row < out_per_expert) {
                    const int pair_index = ids_dst[compact];
                    output[static_cast<int64_t>(pair_index) * out_per_expert +
                           local_row] =
                        __float2half(output_tile[warp][m_local][n_local]);
                }
            }
        }
        __syncthreads();
    }
}

mfq_tensor_backend::Tensor nvq_moe_grouped_matmul_pool_f16_cuda(
    mfq_tensor_backend::Tensor indices, mfq_tensor_backend::Tensor aux, mfq_tensor_backend::Tensor sub_scale,
    mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor codebook, mfq_tensor_backend::Tensor x,
    mfq_tensor_backend::Tensor expert_local, int64_t n_experts, int64_t pool_experts,
    int64_t out_per_expert, int64_t neuron_len, int64_t gs,
    int64_t sub_bits, int64_t format, int64_t sign_mode,
    mfq_tensor_backend::Tensor out, mfq_tensor_backend::Tensor ids_dst, mfq_tensor_backend::Tensor expert_bounds,
    mfq_tensor_backend::Tensor tile_bounds, mfq_tensor_backend::Tensor tile_experts) {
    check_common(indices, aux, sub_scale, neuron_scale, codebook,
                 neuron_len, gs, sub_bits, format, sign_mode,true);
    MFQ_RUNTIME_CHECK(gs == kGroupSize, "NVQ routed FP16 requires gs24");
    MFQ_RUNTIME_CHECK(x.is_cuda() && x.is_contiguous() &&
                x.scalar_type() == mfq_tensor_backend::kFloat16 &&
                (x.dim() == 2 || x.dim() == 3) &&
                x.size(-1) == neuron_len,
                "NVQ routed FP16 input shape mismatch");
    const int routes = static_cast<int>(out.size(1));
    const int pairs = static_cast<int>(out.size(0)) * routes;
    const int K = static_cast<int>(neuron_len);
    const int ng = (K + kGroupSize - 1) / kGroupSize;
    const int nvec = (K + (is_d4_format(format) ? 3 : 7)) /
        (is_d4_format(format) ? 4 : 8);
    const int nsign = (K + 7) / 8;
    const int max_tiles = (pairs + 7) / 8 + static_cast<int>(n_experts);
    const int ntiles_n = (static_cast<int>(out_per_expert) + 63) / 64;
    const int blocks = static_cast<int>(std::min<int64_t>(
        static_cast<int64_t>(max_tiles) * ntiles_n, 4096));
    const dim3 threads(32, 8);
    const cudaStream_t stream = mfq_current_cuda_stream();
#define NVQ_MOE_F16_LAUNCH()                                                     \
    nvq_moe_grouped_f16_kernel<F><<<blocks, threads, 0, stream>>>(               \
        indices.data_ptr<uint8_t>(), indices.numel(),                            \
        aux.data_ptr<uint8_t>(), aux.numel(),                                    \
        sub_scale.data_ptr<uint8_t>(), sub_scale.numel(),                        \
        neuron_scale.data_ptr<float>(), codebook.data_ptr<int8_t>(),             \
        reinterpret_cast<const __half *>(x.data_ptr<mfq_half>()),                \
        expert_local.data_ptr<int32_t>(), ids_dst.data_ptr<int32_t>(),           \
        expert_bounds.data_ptr<int32_t>(), tile_bounds.data_ptr<int32_t>(),      \
        tile_experts.data_ptr<int32_t>(),                                        \
        reinterpret_cast<__half *>(out.data_ptr<mfq_half>()),                    \
        routes, static_cast<int>(n_experts), static_cast<int>(pool_experts),     \
        static_cast<int>(out_per_expert), K, ng, nvec, nsign,                   \
        static_cast<int>(sub_bits), static_cast<int>(sign_mode),                 \
        max_tiles, x.dim() == 3)
    launch_by_format(static_cast<int>(format), [&](auto tag) {
        constexpr int F = decltype(tag)::value;
        NVQ_MOE_F16_LAUNCH();
    });
#undef NVQ_MOE_F16_LAUNCH
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

mfq_tensor_backend::Tensor nvq_moe_grouped_matmul_hetero_f16_cuda(
    mfq_tensor_backend::Tensor weight_ptrs,
    mfq_tensor_backend::Tensor weight_sizes,
    mfq_tensor_backend::Tensor pool_params,
    mfq_tensor_backend::Tensor expert_pool,
    mfq_tensor_backend::Tensor expert_local,
    mfq_tensor_backend::Tensor x,
    int64_t n_experts,
    int64_t out_per_expert,
    int64_t neuron_len,
    int64_t route_tile_m,
    mfq_tensor_backend::Tensor out,
    mfq_tensor_backend::Tensor ids_dst,
    mfq_tensor_backend::Tensor expert_bounds,
    mfq_tensor_backend::Tensor tile_bounds,
    mfq_tensor_backend::Tensor tile_experts,
    int64_t format_group,
    bool masked_experts) {
    MFQ_RUNTIME_CHECK(
        n_experts > 0 && n_experts <= 4096,
        "NVQ heterogeneous expert count must be in [1,4096]");
    MFQ_RUNTIME_CHECK(
        out_per_expert > 0 && out_per_expert <= INT_MAX &&
            neuron_len > 0 && neuron_len <= INT_MAX,
        "NVQ heterogeneous dimensions must fit int32");
    MFQ_RUNTIME_CHECK(
        route_tile_m == 8 || route_tile_m == 64 || route_tile_m == 128,
        "NVQ heterogeneous route tile must be 8, 64, or 128");
    MFQ_RUNTIME_CHECK(
        format_group >= kNvqMoeF16AllFormats &&
            format_group <= kNvqMoeF16LegacyFormats,
        "NVQ heterogeneous FP16 format group is invalid");
    MFQ_RUNTIME_CHECK(
        weight_ptrs.is_cuda() && weight_ptrs.is_contiguous() &&
            weight_ptrs.scalar_type() == mfq_tensor_backend::kInt64 &&
            weight_ptrs.dim() == 2 && weight_ptrs.size(0) > 0 &&
            weight_ptrs.size(1) == 5,
        "NVQ heterogeneous weight pointers must be CUDA int64 [pools,5]");
    const int pools = static_cast<int>(weight_ptrs.size(0));
    MFQ_RUNTIME_CHECK(
        weight_sizes.is_cuda() && weight_sizes.is_contiguous() &&
            weight_sizes.scalar_type() == mfq_tensor_backend::kInt64 &&
            weight_sizes.dim() == 2 && weight_sizes.size(0) == pools &&
            weight_sizes.size(1) == 3,
        "NVQ heterogeneous weight sizes must be CUDA int64 [pools,3]");
    MFQ_RUNTIME_CHECK(
        pool_params.is_cuda() && pool_params.is_contiguous() &&
            pool_params.scalar_type() == mfq_tensor_backend::kInt32 &&
            pool_params.dim() == 2 && pool_params.size(0) == pools &&
            pool_params.size(1) == 7,
        "NVQ heterogeneous pool parameters must be CUDA int32 [pools,7]");
    MFQ_RUNTIME_CHECK(
        expert_pool.is_cuda() && expert_pool.is_contiguous() &&
            expert_pool.scalar_type() == mfq_tensor_backend::kInt32 &&
            expert_pool.numel() == n_experts,
        "NVQ heterogeneous expert-pool map must be CUDA int32 [experts]");
    MFQ_RUNTIME_CHECK(
        expert_local.is_cuda() && expert_local.is_contiguous() &&
            expert_local.scalar_type() == mfq_tensor_backend::kInt32 &&
            expert_local.numel() == n_experts,
        "NVQ heterogeneous expert-local map must be CUDA int32 [experts]");
    MFQ_RUNTIME_CHECK(
        x.is_cuda() && x.is_contiguous() &&
            x.scalar_type() == mfq_tensor_backend::kFloat16 &&
            (x.dim() == 2 || x.dim() == 3) &&
            x.size(-1) == neuron_len,
        "NVQ heterogeneous FP16 input shape mismatch");
    MFQ_RUNTIME_CHECK(
        out.is_cuda() && out.is_contiguous() &&
            out.scalar_type() == mfq_tensor_backend::kFloat16 &&
            out.dim() == 3 && out.size(2) == out_per_expert,
        "NVQ heterogeneous output must be CUDA FP16 [tokens,routes,out]");
    const int tokens = static_cast<int>(out.size(0));
    const int routes = static_cast<int>(out.size(1));
    const int pairs = tokens * routes;
    const bool routed_input = x.dim() == 3;
    MFQ_RUNTIME_CHECK(
        tokens > 8 && routes > 0 &&
            ((!routed_input && x.size(0) == tokens) ||
             (routed_input && x.size(0) == tokens && x.size(1) == routes)),
        "NVQ heterogeneous input leading dimensions do not match output");
    MFQ_RUNTIME_CHECK(
        ids_dst.is_cuda() && ids_dst.is_contiguous() &&
            ids_dst.scalar_type() == mfq_tensor_backend::kInt32 &&
            ids_dst.numel() >= pairs,
        "NVQ heterogeneous compact route map is missing");
    MFQ_RUNTIME_CHECK(
        expert_bounds.is_cuda() && expert_bounds.is_contiguous() &&
            expert_bounds.scalar_type() == mfq_tensor_backend::kInt32 &&
            expert_bounds.numel() >= n_experts + 1,
        "NVQ heterogeneous expert bounds are missing");
    MFQ_RUNTIME_CHECK(
        tile_bounds.is_cuda() && tile_bounds.is_contiguous() &&
            tile_bounds.scalar_type() == mfq_tensor_backend::kInt32 &&
            tile_bounds.numel() >= n_experts + 1,
        "NVQ heterogeneous tile bounds are missing");
    MFQ_RUNTIME_CHECK(
        tile_experts.is_cuda() && tile_experts.is_contiguous() &&
            tile_experts.scalar_type() == mfq_tensor_backend::kInt32 &&
            tile_experts.numel() >= pairs,
        "NVQ heterogeneous tile expert map is missing");
    for (const auto * tensor : {
             &weight_sizes, &pool_params, &expert_pool, &expert_local,
             &x, &out, &ids_dst, &expert_bounds, &tile_bounds,
             &tile_experts}) {
        MFQ_RUNTIME_CHECK(
            tensor->device() == weight_ptrs.device(),
            "NVQ heterogeneous tensors must share one CUDA device");
    }

    const int tile_m = static_cast<int>(route_tile_m);
    const int rows_per_expert = std::max(
        1, (pairs + static_cast<int>(n_experts) - 1) /
               static_cast<int>(n_experts));
    const int fine_bm = rows_per_expert <= 16
        ? 16
        : (rows_per_expert <= 32 ? 32 : 64);
    const int max_tiles =
        (pairs + tile_m - 1) / tile_m + static_cast<int>(n_experts);
    const int base_block_cap = pairs >= 32768 ? 8192 : 4096;
    const int64_t route_tiles =
        (pairs + tile_m - 1) / tile_m;
    const int narrow_ntiles_n =
        (static_cast<int>(out_per_expert) + 63) / 64;
    // Use two smaller resident CTAs only while each useful tile gets its own
    // CTA; longer routes retain the wider tile's activation reuse.
    const bool use_narrow_tile = tile_m == 128 &&
        route_tiles * narrow_ntiles_n <= base_block_cap;
    const int tile_n = use_narrow_tile
        ? 64 : kNvqMoeF16HeteroTileN;
    const int ntiles_n =
        (static_cast<int>(out_per_expert) + tile_n - 1) / tile_n;
    const int64_t useful_tiles =
        (pairs + tile_m - 1) / tile_m;
    const int64_t useful_tasks = useful_tiles * ntiles_n;
    int block_cap = std::max(
        base_block_cap,
        static_cast<int>(std::min<int64_t>(
            INT_MAX, (useful_tasks + 3) / 4)));
    const int64_t max_tasks =
        static_cast<int64_t>(max_tiles) * ntiles_n;
    if (max_tasks > block_cap) {
        const int average_route_tiles =
            (rows_per_expert + tile_m - 1) / tile_m;
        const int task_period = average_route_tiles * ntiles_n;
        // Rotate CTAs between expert phases when the grid stride would
        // otherwise repeat the same valid/empty task pattern.  For a fully
        // covered multi-pool tensor, one extra CTA breaks a repeated pool
        // phase without changing single-pool launches.
        if (task_period > 1 && block_cap % task_period == 0) {
            if (masked_experts && tile_m != 8 && task_period >= 64) {
                block_cap += task_period;
            } else if (tile_m == 8 && base_block_cap == 4096 &&
                       rows_per_expert > 16) {
                block_cap += std::min(16, std::max(1, task_period / 4));
            } else if (pools > 1 && task_period >= 256) {
                ++block_cap;
            }
        }
    }
    const int blocks = static_cast<int>(std::max<int64_t>(
        1, std::min<int64_t>(max_tasks, block_cap)));
#define NVQ_MOE_HETERO_F16_LAUNCH(                                             \
    BM_VALUE, BN_VALUE, ROUTE_VALUE, GROUP_VALUE, FORMAT_GROUP_VALUE)          \
    do {                                                                         \
        constexpr int kSharedBytes = sizeof(NvqMoeF16SharedStorage<             \
            BM_VALUE, BN_VALUE, GROUP_VALUE>);                                  \
        if constexpr (kSharedBytes > 48 * 1024) {                               \
            static const cudaError_t attribute_status = cudaFuncSetAttribute(   \
                nvq_moe_grouped_hetero_f16_kernel<                              \
                    BM_VALUE, BN_VALUE, ROUTE_VALUE, GROUP_VALUE,               \
                    FORMAT_GROUP_VALUE, true>,                                  \
                cudaFuncAttributeMaxDynamicSharedMemorySize, kSharedBytes);     \
            MFQ_RUNTIME_CHECK(                                                   \
                attribute_status == cudaSuccess,                                \
                "failed to opt in to the NVQ MoE shared-memory size");          \
        }                                                                        \
        nvq_moe_grouped_hetero_f16_kernel<                                      \
            BM_VALUE, BN_VALUE, ROUTE_VALUE, GROUP_VALUE,                       \
            FORMAT_GROUP_VALUE, true><<<                                        \
            blocks, dim3(32, (BM_VALUE >= 128 ? BM_VALUE / 64 : 1) *            \
                (BN_VALUE / 16)), kSharedBytes,                                 \
            mfq_current_cuda_stream()>>>(                                       \
        weight_ptrs.data_ptr<int64_t>(), weight_sizes.data_ptr<int64_t>(),      \
        pool_params.data_ptr<int32_t>(), expert_pool.data_ptr<int32_t>(),       \
        expert_local.data_ptr<int32_t>(),                                      \
        reinterpret_cast<const __half *>(x.data_ptr<mfq_half>()),              \
        ids_dst.data_ptr<int32_t>(), expert_bounds.data_ptr<int32_t>(),         \
        tile_bounds.data_ptr<int32_t>(), tile_experts.data_ptr<int32_t>(),      \
        reinterpret_cast<__half *>(out.data_ptr<mfq_half>()), routes,          \
        static_cast<int>(n_experts), pools,                                    \
        static_cast<int>(out_per_expert), static_cast<int>(neuron_len),        \
        max_tiles, routed_input);                                                \
    } while (false)
    // A bounded format switch fits the wide G2 task in 64 registers, so two
    // BM128 CTAs can reside together.  Cross-group mixtures keep the full
    // switch, G4 path, and its wider K reuse.
    const char* wide_rows_option = mfq::cuda::env_lookup("MFQ_NVQ_PREFILL_M256");
    const bool wide_rows = wide_rows_option && std::atoi(wide_rows_option) != 0;
    const char* narrow_g2_option = mfq::cuda::env_lookup("MFQ_NVQ_PREFILL_NARROW_G2");
    const bool narrow_g2 = narrow_g2_option && std::atoi(narrow_g2_option) != 0;
    const char* cohorts_option = mfq::cuda::env_lookup("MFQ_NVQ_PREFILL_COHORTS");
    const bool cohorts = !cohorts_option || std::atoi(cohorts_option) != 0;
    // Keep the M/N geometry and Tensor Core sum order. A shorter K tile
    // lowers decode liveness and shared storage, allowing three narrow CTAs.
    if (cohorts && tile_m == 128 && format_group == kNvqMoeF16AllFormats) {
        // Disjoint format switches write disjoint expert rows. Every task
        // retains the canonical Tensor Core K sequence, with no new maps
        // or weight buffers. Keep G4 for narrow tiles and the legacy decoder.
        if (use_narrow_tile) {
            NVQ_MOE_HETERO_F16_LAUNCH(128,64,128,4,kNvqMoeF16StandardFormats);
            NVQ_MOE_HETERO_F16_LAUNCH(128,64,128,4,kNvqMoeF16ExtendedFormats);
            NVQ_MOE_HETERO_F16_LAUNCH(128,64,128,4,kNvqMoeF16LegacyFormats);
        } else {
            NVQ_MOE_HETERO_F16_LAUNCH(128,128,128,2,kNvqMoeF16StandardFormats);
            NVQ_MOE_HETERO_F16_LAUNCH(128,128,128,2,kNvqMoeF16ExtendedFormats);
            NVQ_MOE_HETERO_F16_LAUNCH(128,128,128,4,kNvqMoeF16LegacyFormats);
        }
    } else if (narrow_g2 && tile_m == 128 && use_narrow_tile &&
            format_group == kNvqMoeF16StandardFormats) {
        NVQ_MOE_HETERO_F16_LAUNCH(
            128, 64, 128, 2, kNvqMoeF16StandardFormats);
    } else if (narrow_g2 && tile_m == 128 && use_narrow_tile &&
            format_group == kNvqMoeF16ExtendedFormats) {
        NVQ_MOE_HETERO_F16_LAUNCH(
            128, 64, 128, 2, kNvqMoeF16ExtendedFormats);
    } else if (wide_rows && tile_m == 128 && use_narrow_tile && rows_per_expert >= 128 &&
            out_per_expert <= neuron_len) {
        if (format_group == kNvqMoeF16StandardFormats) {
            NVQ_MOE_HETERO_F16_LAUNCH(
                256, 64, 128, 4, kNvqMoeF16StandardFormats);
        } else if (format_group == kNvqMoeF16ExtendedFormats) {
            NVQ_MOE_HETERO_F16_LAUNCH(
                256, 64, 128, 4, kNvqMoeF16ExtendedFormats);
        } else if (format_group == kNvqMoeF16LegacyFormats) {
            NVQ_MOE_HETERO_F16_LAUNCH(
                256, 64, 128, 4, kNvqMoeF16LegacyFormats);
        } else {
            NVQ_MOE_HETERO_F16_LAUNCH(
                256, 64, 128, 4, kNvqMoeF16AllFormats);
        }
    } else if (tile_m == 128 && use_narrow_tile &&
            format_group == kNvqMoeF16StandardFormats) {
        NVQ_MOE_HETERO_F16_LAUNCH(
            128, 64, 128, 4, kNvqMoeF16StandardFormats);
    } else if (tile_m == 128 && !use_narrow_tile &&
            format_group == kNvqMoeF16StandardFormats) {
        NVQ_MOE_HETERO_F16_LAUNCH(
            128, 128, 128, 2, kNvqMoeF16StandardFormats);
    } else if (tile_m == 128 && !use_narrow_tile &&
            format_group == kNvqMoeF16ExtendedFormats) {
        NVQ_MOE_HETERO_F16_LAUNCH(
            128, 128, 128, 2, kNvqMoeF16ExtendedFormats);
    } else if (tile_m == 128 && !use_narrow_tile &&
            format_group == kNvqMoeF16LegacyFormats) {
        NVQ_MOE_HETERO_F16_LAUNCH(
            128, 128, 128, 2, kNvqMoeF16LegacyFormats);
    } else if (tile_m == 128 && use_narrow_tile) {
        NVQ_MOE_HETERO_F16_LAUNCH(
            128, 64, 128, 4, kNvqMoeF16AllFormats);
    } else if (tile_m == 128) {
        NVQ_MOE_HETERO_F16_LAUNCH(
            128, 128, 128, 4, kNvqMoeF16AllFormats);
    } else if (tile_m == 64) {
        NVQ_MOE_HETERO_F16_LAUNCH(
            128, 128, 64, 4, kNvqMoeF16AllFormats);
    } else if (fine_bm == 16) {
        NVQ_MOE_HETERO_F16_LAUNCH(
            16, 128, 8, 4, kNvqMoeF16AllFormats);
    } else if (fine_bm == 32) {
        NVQ_MOE_HETERO_F16_LAUNCH(
            32, 128, 8, 4, kNvqMoeF16AllFormats);
    } else {
        NVQ_MOE_HETERO_F16_LAUNCH(
            64, 128, 8, 4, kNvqMoeF16AllFormats);
    }
#undef NVQ_MOE_HETERO_F16_LAUNCH
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

static mfq_tensor_backend::Tensor nvq_moe_grouped_matmul_hetero_impl(
    mfq_tensor_backend::Tensor weight_ptrs,
    mfq_tensor_backend::Tensor weight_sizes,
    mfq_tensor_backend::Tensor pool_params,
    mfq_tensor_backend::Tensor expert_pool,
    mfq_tensor_backend::Tensor expert_local,
    mfq_tensor_backend::Tensor x,
    mfq_tensor_backend::Tensor ids,
    int64_t n_experts,
    int64_t out_per_expert,
    int64_t neuron_len,
    bool input_quantized,
    mfq_tensor_backend::Tensor out,
    mfq_tensor_backend::Tensor qx,
    mfq_tensor_backend::Tensor xscale,mfq_tensor_backend::Tensor active_plan) {
    MFQ_RUNTIME_CHECK(
        n_experts > 0 && n_experts <= 4096,
        "NVQ heterogeneous expert count must be in [1,4096]");
    MFQ_RUNTIME_CHECK(
        out_per_expert > 0 && out_per_expert <= INT_MAX &&
            neuron_len > 0 && neuron_len <= INT_MAX,
        "NVQ heterogeneous dimensions must fit int32");
    MFQ_RUNTIME_CHECK(
        weight_ptrs.is_cuda() && weight_ptrs.is_contiguous() &&
            weight_ptrs.scalar_type() == mfq_tensor_backend::kInt64 &&
            weight_ptrs.dim() == 2 && weight_ptrs.size(0) > 0 &&
            weight_ptrs.size(1) == 5,
        "NVQ heterogeneous weight pointers must be CUDA int64 [pools,5]");
    const int pools = static_cast<int>(weight_ptrs.size(0));
    MFQ_RUNTIME_CHECK(
        weight_sizes.is_cuda() && weight_sizes.is_contiguous() &&
            weight_sizes.scalar_type() == mfq_tensor_backend::kInt64 &&
            weight_sizes.dim() == 2 && weight_sizes.size(0) == pools &&
            weight_sizes.size(1) == 3,
        "NVQ heterogeneous weight sizes must be CUDA int64 [pools,3]");
    MFQ_RUNTIME_CHECK(
        pool_params.is_cuda() && pool_params.is_contiguous() &&
            pool_params.scalar_type() == mfq_tensor_backend::kInt32 &&
            pool_params.dim() == 2 && pool_params.size(0) == pools &&
            pool_params.size(1) == 7,
        "NVQ heterogeneous pool parameters must be CUDA int32 [pools,7]");
    MFQ_RUNTIME_CHECK(
        expert_pool.is_cuda() && expert_pool.is_contiguous() &&
            expert_pool.scalar_type() == mfq_tensor_backend::kInt32 &&
            expert_pool.numel() == n_experts &&
            expert_local.is_cuda() && expert_local.is_contiguous() &&
            expert_local.scalar_type() == mfq_tensor_backend::kInt32 &&
            expert_local.numel() == n_experts,
        "NVQ heterogeneous expert maps must be CUDA int32 [experts]");
    MFQ_RUNTIME_CHECK(
        ids.is_cuda() && ids.is_contiguous() &&
            ids.scalar_type() == mfq_tensor_backend::kInt32 && ids.dim() == 2,
        "NVQ heterogeneous route IDs must be CUDA int32 [tokens,routes]");
    const int tokens = static_cast<int>(ids.size(0));
    const int routes = static_cast<int>(ids.size(1));
    const int pairs = tokens * routes;
    MFQ_RUNTIME_CHECK(
        tokens > 0 && tokens <= 8 && routes > 0,
        "NVQ heterogeneous MMVQ requires one to eight tokens");
    MFQ_RUNTIME_CHECK(
        x.is_cuda() && x.is_contiguous() &&
            x.scalar_type() == mfq_tensor_backend::kFloat16 &&
            (x.dim() == 2 || x.dim() == 3) &&
            (input_quantized || x.size(-1) == neuron_len),
        "NVQ heterogeneous routed input shape mismatch");
    const bool routed_input = x.dim() == 3;
    MFQ_RUNTIME_CHECK(
        (!routed_input && x.size(0) == tokens) ||
            (routed_input && x.size(0) == tokens && x.size(1) == routes),
        "NVQ heterogeneous input leading dimensions do not match routes");
    MFQ_RUNTIME_CHECK(
        out.is_cuda() && out.is_contiguous() &&
            out.scalar_type() == mfq_tensor_backend::kFloat16 &&
            out.dim() == 3 && out.size(0) == tokens &&
            out.size(1) == routes && out.size(2) == out_per_expert,
        "NVQ heterogeneous output must be CUDA FP16 [tokens,routes,out]");
    const int K = static_cast<int>(neuron_len);
    const int ng = (K + kGroupSize - 1) / kGroupSize;
    const int K_pad = ng * kGroupSize;
    const int input_rows = routed_input ? pairs : tokens;
    MFQ_RUNTIME_CHECK(
        qx.is_cuda() && qx.is_contiguous() &&
            qx.scalar_type() == mfq_tensor_backend::kInt8 && qx.dim() == 2 &&
            qx.size(0) >= input_rows && qx.size(1) >= K_pad,
        "NVQ heterogeneous qx workspace mismatch");
    MFQ_RUNTIME_CHECK(
        xscale.is_cuda() && xscale.is_contiguous() &&
            xscale.scalar_type() == mfq_tensor_backend::kFloat32 &&
            xscale.dim() == 2 && xscale.size(0) >= input_rows &&
            xscale.size(1) >= ng,
        "NVQ heterogeneous xscale workspace mismatch");
    for (const auto * tensor : {
             &weight_sizes, &pool_params, &expert_pool, &expert_local,
             &x, &ids, &out, &qx, &xscale}) {
        MFQ_RUNTIME_CHECK(
            tensor->device() == weight_ptrs.device(),
            "NVQ heterogeneous tensors must share one CUDA device");
    }

    if(active_plan.defined())MFQ_RUNTIME_CHECK(active_plan.is_cuda() && active_plan.is_contiguous() &&
        active_plan.scalar_type()==mfq_tensor_backend::kInt32 && active_plan.device()==x.device() &&
        active_plan.numel()==1+int64_t(pairs)*3,"NVQ active route plan geometry/device mismatch");
    const cudaStream_t stream = mfq_current_cuda_stream();
    if (!input_quantized) {
        nvq_quantize_x_gs24_kernel<<<dim3(input_rows, ng), 32, 0, stream>>>(
            reinterpret_cast<const __half *>(x.data_ptr<mfq_half>()),
            qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), input_rows, K, ng);
    }
#define NVQ_MOE_HETERO_MMVQ_LAUNCH(NWARPS_VALUE, ROWS_VALUE)                   \
    nvq_moe_mmvq_hetero_kernel<                                                \
        NWARPS_VALUE, ROWS_VALUE, false><<<                                    \
        dim3((static_cast<int>(out_per_expert) + ROWS_VALUE - 1) / ROWS_VALUE, pairs), \
        dim3(32, NWARPS_VALUE), 0, stream>>>(                                  \
        weight_ptrs.data_ptr<int64_t>(), weight_sizes.data_ptr<int64_t>(),     \
        pool_params.data_ptr<int32_t>(), expert_pool.data_ptr<int32_t>(),      \
        expert_local.data_ptr<int32_t>(), qx.data_ptr<int8_t>(),               \
        xscale.data_ptr<float>(), ids.data_ptr<int32_t>(),                     \
        reinterpret_cast<__half *>(out.data_ptr<mfq_half>()), pairs, routes,   \
        static_cast<int>(n_experts), pools, static_cast<int>(out_per_expert),  \
        routed_input)
    const char* grouped=mfq::cuda::env_lookup("MFQ_NVQ_MOE_GROUP_DOT");
    if(ng<=128 && out_per_expert<=4096 && (!grouped || grouped[0]!='0')) {
        mfq::cuda::NvqRoutedReuseParams reuse;
        reuse.pointers=weight_ptrs.data_ptr<int64_t>();reuse.sizes=weight_sizes.data_ptr<int64_t>();
        reuse.params=pool_params.data_ptr<int32_t>();reuse.expert_pool=expert_pool.data_ptr<int32_t>();
        reuse.expert_local=expert_local.data_ptr<int32_t>();reuse.ids=ids.data_ptr<int32_t>();
        reuse.active=active_plan.defined()?active_plan.data_ptr<int32_t>():nullptr;
        reuse.input=qx.data_ptr<int8_t>();reuse.scales=xscale.data_ptr<float>();
        reuse.output=reinterpret_cast<half*>(out.data_ptr<mfq_half>());
        reuse.experts=int(n_experts);reuse.pools=pools;reuse.rows=int(out_per_expert);
        reuse.routes=routes;reuse.pairs=pairs;reuse.down=routed_input;
        if(!mfq::cuda::nvq_launch_routed_reuse(reuse,stream))
            nvq_moe_groups_hetero_kernel<<<dim3((out_per_expert+3)/4,pairs),dim3(32,4),0,stream>>>(
            weight_ptrs.data_ptr<int64_t>(),weight_sizes.data_ptr<int64_t>(),pool_params.data_ptr<int32_t>(),
            expert_pool.data_ptr<int32_t>(),expert_local.data_ptr<int32_t>(),ids.data_ptr<int32_t>(),
            qx.data_ptr<int8_t>(),xscale.data_ptr<float>(),reinterpret_cast<half*>(out.data_ptr<mfq_half>()),
            active_plan.defined()?active_plan.data_ptr<int32_t>():nullptr,
            int(n_experts),pools,int(out_per_expert),routes,pairs,routed_input);
    }else if(active_plan.defined()) {
        const auto launch=[&](auto warp_tag) {
            constexpr int warps=decltype(warp_tag)::value;
            const int blocks=int(std::min<int64_t>(nvq_active_resident_blocks<warps>(),
                ((out_per_expert-1)/2+1)*pairs));
            nvq_moe_mmvq_hetero_kernel<warps,2,false,true><<<blocks,dim3(32,warps),0,stream>>>(
                weight_ptrs.data_ptr<int64_t>(),weight_sizes.data_ptr<int64_t>(),pool_params.data_ptr<int32_t>(),
                expert_pool.data_ptr<int32_t>(),expert_local.data_ptr<int32_t>(),qx.data_ptr<int8_t>(),
                xscale.data_ptr<float>(),ids.data_ptr<int32_t>(),reinterpret_cast<__half*>(out.data_ptr<mfq_half>()),
                pairs,routes,static_cast<int>(n_experts),pools,static_cast<int>(out_per_expert),routed_input,
                active_plan.data_ptr<int32_t>());
        };
        if(K>=4096)launch(std::integral_constant<int,8>{});else launch(std::integral_constant<int,4>{});
    } else if (K >= 4096) NVQ_MOE_HETERO_MMVQ_LAUNCH(8,2);
    else {
        // Keep each row's four-warp accumulation and reduction unchanged.
        // The opt-in sweep varies only the number of rows sharing a CTA.
        const auto* requested=mfq::cuda::env_lookup("MFQ_NVQ_MOE_ROW_TILE");
        const int rows=tokens==1 && requested ? std::atoi(requested) : 2;
        switch(rows) {
            case 1: NVQ_MOE_HETERO_MMVQ_LAUNCH(4,1);break;
            case 4: NVQ_MOE_HETERO_MMVQ_LAUNCH(4,4);break;
            case 8: NVQ_MOE_HETERO_MMVQ_LAUNCH(4,8);break;
            case 16: NVQ_MOE_HETERO_MMVQ_LAUNCH(4,16);break;
            case 32: NVQ_MOE_HETERO_MMVQ_LAUNCH(4,32);break;
            default: NVQ_MOE_HETERO_MMVQ_LAUNCH(4,2);break;
        }
    }
#undef NVQ_MOE_HETERO_MMVQ_LAUNCH
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

mfq_tensor_backend::Tensor nvq_moe_grouped_matmul_hetero_ws_cuda(
    mfq_tensor_backend::Tensor weight_ptrs,mfq_tensor_backend::Tensor weight_sizes,
    mfq_tensor_backend::Tensor pool_params,mfq_tensor_backend::Tensor expert_pool,
    mfq_tensor_backend::Tensor expert_local,mfq_tensor_backend::Tensor x,mfq_tensor_backend::Tensor ids,
    int64_t n_experts,int64_t out_per_expert,int64_t neuron_len,bool input_quantized,
    mfq_tensor_backend::Tensor out,mfq_tensor_backend::Tensor qx,mfq_tensor_backend::Tensor xscale) {
    return nvq_moe_grouped_matmul_hetero_impl(weight_ptrs,weight_sizes,pool_params,expert_pool,expert_local,
        x,ids,n_experts,out_per_expert,neuron_len,input_quantized,out,qx,xscale,{});
}
mfq_tensor_backend::Tensor nvq_moe_grouped_matmul_active_ws_cuda(
    mfq_tensor_backend::Tensor weight_ptrs,mfq_tensor_backend::Tensor weight_sizes,
    mfq_tensor_backend::Tensor pool_params,mfq_tensor_backend::Tensor expert_pool,
    mfq_tensor_backend::Tensor expert_local,mfq_tensor_backend::Tensor x,mfq_tensor_backend::Tensor ids,
    int64_t n_experts,int64_t out_per_expert,int64_t neuron_len,bool input_quantized,
    mfq_tensor_backend::Tensor out,mfq_tensor_backend::Tensor qx,mfq_tensor_backend::Tensor xscale,
    mfq_tensor_backend::Tensor active_plan) {
    MFQ_RUNTIME_CHECK(active_plan.defined(),"NVQ active route plan required");
    return nvq_moe_grouped_matmul_hetero_impl(weight_ptrs,weight_sizes,pool_params,expert_pool,expert_local,
        x,ids,n_experts,out_per_expert,neuron_len,input_quantized,out,qx,xscale,active_plan);
}

mfq_tensor_backend::Tensor nepq_moe_grouped_matmul_pool_f16_cuda(
    mfq_tensor_backend::Tensor indices, mfq_tensor_backend::Tensor aux, mfq_tensor_backend::Tensor sub_scale,
    mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor table_pool,
    mfq_tensor_backend::Tensor bank_ids, mfq_tensor_backend::Tensor x, mfq_tensor_backend::Tensor expert_local,
    int64_t n_experts, int64_t pool_experts, int64_t out_per_expert,
    int64_t neuron_len, int64_t sub_bits, int64_t format,
    mfq_tensor_backend::Tensor out, mfq_tensor_backend::Tensor ids_dst, mfq_tensor_backend::Tensor expert_bounds,
    mfq_tensor_backend::Tensor tile_bounds, mfq_tensor_backend::Tensor tile_experts) {
    check_nepq_common(indices, aux, sub_scale, neuron_scale, table_pool,
                      bank_ids, neuron_len, sub_bits, format);
    MFQ_RUNTIME_CHECK(x.is_cuda() && x.is_contiguous() &&
                x.scalar_type() == mfq_tensor_backend::kFloat16 &&
                (x.dim() == 2 || x.dim() == 3) &&
                x.size(-1) == neuron_len,
                "NEPQ routed FP16 input shape mismatch");
    const int routes = static_cast<int>(out.size(1));
    const int pairs = static_cast<int>(out.size(0)) * routes;
    const int K = static_cast<int>(neuron_len);
    const int ng = (K + kGroupSize - 1) / kGroupSize;
    const int nvec = K / 8;
    const int nsign = (K + 7) / 8;
    const int nsuper = (ng + kNepqGroupsPerSupergroup - 1) /
        kNepqGroupsPerSupergroup;
    const int table_stride = static_cast<int>(table_pool.size(1));
    const int max_tiles = (pairs + 7) / 8 + static_cast<int>(n_experts);
    const int ntiles_n = (static_cast<int>(out_per_expert) + 63) / 64;
    const int blocks = static_cast<int>(std::min<int64_t>(
        static_cast<int64_t>(max_tiles) * ntiles_n, 4096));
    const dim3 threads(32, 8);
    const cudaStream_t stream = mfq_current_cuda_stream();
#define NEPQ_MOE_F16_LAUNCH()                                                    \
    nepq_moe_grouped_f16_kernel<F><<<blocks, threads, 0, stream>>>(              \
        indices.data_ptr<uint8_t>(), indices.numel(),                            \
        aux.data_ptr<uint8_t>(), aux.numel(),                                    \
        sub_scale.data_ptr<uint8_t>(), sub_scale.numel(),                        \
        neuron_scale.data_ptr<float>(), table_pool.data_ptr<int8_t>(),           \
        bank_ids.data_ptr<uint8_t>(),                                            \
        reinterpret_cast<const __half *>(x.data_ptr<mfq_half>()),                \
        expert_local.data_ptr<int32_t>(), ids_dst.data_ptr<int32_t>(),           \
        expert_bounds.data_ptr<int32_t>(), tile_bounds.data_ptr<int32_t>(),      \
        tile_experts.data_ptr<int32_t>(),                                        \
        reinterpret_cast<__half *>(out.data_ptr<mfq_half>()),                    \
        routes, static_cast<int>(n_experts), static_cast<int>(pool_experts),     \
        static_cast<int>(out_per_expert), K, ng, nvec, nsign, nsuper,           \
        table_stride, static_cast<int>(sub_bits), max_tiles, x.dim() == 3)
    launch_nepq_by_format(static_cast<int>(format), [&](auto tag) {
        constexpr int F = decltype(tag)::value;
        NEPQ_MOE_F16_LAUNCH();
    });
#undef NEPQ_MOE_F16_LAUNCH
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

mfq_tensor_backend::Tensor nvq_moe_grouped_matmul_pool_ws_cuda(
    mfq_tensor_backend::Tensor indices,
    mfq_tensor_backend::Tensor aux,
    mfq_tensor_backend::Tensor sub_scale,
    mfq_tensor_backend::Tensor neuron_scale,
    mfq_tensor_backend::Tensor codebook,
    mfq_tensor_backend::Tensor x,
    mfq_tensor_backend::Tensor ids,
    mfq_tensor_backend::Tensor expert_local,
    int64_t n_experts,
    int64_t pool_experts,
    int64_t out_per_expert,
    int64_t neuron_len,
    int64_t gs,
    int64_t sub_bits,
    int64_t format,
    int64_t sign_mode,
    bool input_quantized,
    mfq_tensor_backend::Tensor out,
    mfq_tensor_backend::Tensor qx,
    mfq_tensor_backend::Tensor xscale,
    mfq_tensor_backend::Tensor ids_dst,
    mfq_tensor_backend::Tensor expert_bounds,
    mfq_tensor_backend::Tensor tile_bounds,
    mfq_tensor_backend::Tensor tile_experts) {
    check_common(
        indices, aux, sub_scale, neuron_scale, codebook,
        neuron_len, gs, sub_bits, format, sign_mode,true);
    MFQ_RUNTIME_CHECK(n_experts > 0 && n_experts <= 4096,
                "NVQ global expert count must be in [1,4096]");
    MFQ_RUNTIME_CHECK(
        pool_experts > 0 &&
        pool_experts <= static_cast<int64_t>(std::numeric_limits<int>::max()),
        "NVQ pool storage expert count must fit int32");
    MFQ_RUNTIME_CHECK(out_per_expert > 0 &&
                neuron_scale.numel() == pool_experts * out_per_expert,
                "NVQ pool shape does not match the flattened row count");
    MFQ_RUNTIME_CHECK(expert_local.is_cuda() && expert_local.is_contiguous() &&
                expert_local.scalar_type() == mfq_tensor_backend::kInt32 &&
                expert_local.dim() == 1 && expert_local.numel() >= n_experts,
                "NVQ expert-local map must be CUDA contiguous int32");
    MFQ_RUNTIME_CHECK(ids.is_cuda() && ids.is_contiguous() &&
                ids.scalar_type() == mfq_tensor_backend::kInt32 && ids.dim() == 2,
                "NVQ route IDs must be CUDA contiguous int32 [T,R]");
    MFQ_RUNTIME_CHECK(x.is_cuda() && x.is_contiguous() &&
                x.scalar_type() == mfq_tensor_backend::kFloat16 &&
                (x.dim() == 2 || x.dim() == 3) &&
                (input_quantized || x.size(-1) == neuron_len),
                "NVQ routed input must be CUDA contiguous fp16 [T,K] or [T,R,K]");
    const int tokens = static_cast<int>(ids.size(0));
    const int routes = static_cast<int>(ids.size(1));
    const int pairs = tokens * routes;
    const bool routed_input = x.dim() == 3;
    MFQ_RUNTIME_CHECK(tokens > 0 && routes > 0, "NVQ route dimensions must be nonzero");
    MFQ_RUNTIME_CHECK((!routed_input && x.size(0) == tokens) ||
                (routed_input && x.size(0) == tokens && x.size(1) == routes),
                "NVQ input leading dimensions do not match routes");
    MFQ_RUNTIME_CHECK(out.is_cuda() && out.is_contiguous() &&
                out.scalar_type() == mfq_tensor_backend::kFloat16 && out.dim() == 3 &&
                out.size(0) == tokens && out.size(1) == routes &&
                out.size(2) == out_per_expert,
                "NVQ output must be CUDA contiguous fp16 [T,R,O]");

    const int K = static_cast<int>(neuron_len);
    const int ng = (K + static_cast<int>(gs) - 1) / static_cast<int>(gs);
    const int K_pad = ng * static_cast<int>(gs);
    const int input_rows = routed_input ? pairs : tokens;
    MFQ_RUNTIME_CHECK(qx.is_cuda() && qx.is_contiguous() &&
                qx.scalar_type() == mfq_tensor_backend::kInt8 && qx.dim() == 2 &&
                qx.size(0) >= input_rows && qx.size(1) >= K_pad,
                "NVQ routed qx workspace mismatch");
    MFQ_RUNTIME_CHECK(xscale.is_cuda() && xscale.is_contiguous() &&
                xscale.scalar_type() == mfq_tensor_backend::kFloat32 && xscale.dim() == 2 &&
                xscale.size(0) >= input_rows && xscale.size(1) >= ng,
                "NVQ routed xscale workspace mismatch");
    const int nvec = (K + (is_d4_format(format) ? 3 : 7)) /
        (is_d4_format(format) ? 4 : 8);
    const int nsign = (K + 7) / 8;
    cudaStream_t stream = mfq_current_cuda_stream();
    if (!input_quantized) {
        nvq_quantize_x_gs24_kernel<<<dim3(input_rows, ng), 32, 0, stream>>>(
            reinterpret_cast<const __half *>(x.data_ptr<mfq_half>()),
            qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), input_rows, K, ng);
    }

    if(dense_nvq_format(int(format)) && aux.numel()==0) {
        const NvqDeviceWeight w{indices.data_ptr<uint8_t>(),indices.numel(),nullptr,0,nullptr,0,
            neuron_scale.data_ptr<float>(),codebook.data_ptr<int8_t>(),int(codebook.numel()),
            int(pool_experts*out_per_expert),ng,nvec,nsign,int(sub_bits),int(sign_mode)};
        launch_by_format(int(format),[&](auto tag) {
            constexpr int F=decltype(tag)::value;
            if constexpr(dense_nvq_format(F))
                nvq_moe_groups_kernel<F><<<dim3((out_per_expert+3)/4,pairs),dim3(32,4),0,stream>>>(
                    w,qx.data_ptr<int8_t>(),xscale.data_ptr<float>(),ids.data_ptr<int32_t>(),
                    expert_local.data_ptr<int32_t>(),reinterpret_cast<half*>(out.data_ptr<mfq_half>()),
                    int(n_experts),int(pool_experts),int(out_per_expert),routes,routed_input);
        });
        MFQ_CUDA_KERNEL_LAUNCH_CHECK();return out;
    }

    if (tokens <= 8) {
        const char * rows_env = mfq::cuda::env_lookup("MFQ_NVQ_MOE_ROWS_PER_BLOCK");
        int rows_per_block = rows_env == nullptr ? 2 : std::atoi(rows_env);
        if (rows_per_block != 1 && rows_per_block != 2 &&
                rows_per_block != 4 && rows_per_block != 8) {
            rows_per_block = 2;
        }
        const char * warps_env = mfq::cuda::env_lookup("MFQ_NVQ_MOE_WARPS");
        int warps_override = warps_env == nullptr ? 0 : std::atoi(warps_env);
        if (warps_override != 1 && warps_override != 2 &&
                warps_override != 4 && warps_override != 8) {
            warps_override = 0;
        }
        const char * exact_reduction_env =
            mfq::cuda::env_lookup("MFQ_NVQ_MOE_EXACT_REDUCTION");
        const bool exact_reduction =
            exact_reduction_env != nullptr &&
            std::atoi(exact_reduction_env) != 0;
        const char * exact_physical_env =
            mfq::cuda::env_lookup("MFQ_NVQ_MOE_EXACT_PHYSICAL_WARPS");
        const int exact_physical_warps =
            exact_physical_env != nullptr && std::atoi(exact_physical_env) == 4
            ? 4
            : 2;
        const char * share_group_env =
            mfq::cuda::env_lookup("MFQ_NVQ_MOE_SHARE_GROUP_STATE");
        const bool share_group_state =
            share_group_env != nullptr && std::atoi(share_group_env) != 0;
#define NVQ_MOE_SMALL_LAUNCH(NWARPS_VALUE, ROWS_VALUE, SHARE_VALUE)              \
        nvq_moe_mmvq_kernel<F, NWARPS_VALUE, ROWS_VALUE, SHARE_VALUE><<<         \
            dim3((static_cast<int>(out_per_expert) + ROWS_VALUE - 1) /          \
                     ROWS_VALUE,                                                 \
                 pairs),                                                         \
            dim3(32, NWARPS_VALUE), 0, stream>>>(                                \
            indices.data_ptr<uint8_t>(), indices.numel(),                        \
            aux.data_ptr<uint8_t>(), aux.numel(),                                \
            sub_scale.data_ptr<uint8_t>(), sub_scale.numel(),                    \
            neuron_scale.data_ptr<float>(), codebook.data_ptr<int8_t>(),         \
            qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),                     \
            ids.data_ptr<int32_t>(), expert_local.data_ptr<int32_t>(),           \
            reinterpret_cast<__half *>(out.data_ptr<mfq_half>()),                \
            pairs, routes, static_cast<int>(n_experts),                          \
            static_cast<int>(pool_experts), static_cast<int>(out_per_expert),    \
            ng, nvec, nsign, static_cast<int>(sub_bits),                         \
            static_cast<int>(sign_mode), routed_input)
#define NVQ_MOE_SELECTED_LAUNCH(NWARPS_VALUE, ROWS_VALUE)                        \
        do {                                                                      \
            if constexpr (F == kNvq2Exec || F == kNvq2JscExec) {                 \
                if (share_group_state) {                                          \
                    NVQ_MOE_SMALL_LAUNCH(NWARPS_VALUE, ROWS_VALUE, true);         \
                } else {                                                          \
                    NVQ_MOE_SMALL_LAUNCH(NWARPS_VALUE, ROWS_VALUE, false);        \
                }                                                                 \
            } else {                                                              \
                NVQ_MOE_SMALL_LAUNCH(NWARPS_VALUE, ROWS_VALUE, false);            \
            }                                                                     \
        } while (0)
#define NVQ_MOE_EXACT_LAUNCH(PHYSICAL_WARPS_VALUE, LOGICAL_WARPS_VALUE)          \
        nvq_moe_mmvq_exact_reduction_kernel<                                    \
            F, PHYSICAL_WARPS_VALUE, LOGICAL_WARPS_VALUE, 2><<<                 \
            dim3((static_cast<int>(out_per_expert) + 1) / 2, pairs),            \
            dim3(32, PHYSICAL_WARPS_VALUE), 0, stream>>>(                        \
            indices.data_ptr<uint8_t>(), indices.numel(),                        \
            aux.data_ptr<uint8_t>(), aux.numel(),                                \
            sub_scale.data_ptr<uint8_t>(), sub_scale.numel(),                    \
            neuron_scale.data_ptr<float>(), codebook.data_ptr<int8_t>(),         \
            qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),                     \
            ids.data_ptr<int32_t>(), expert_local.data_ptr<int32_t>(),           \
            reinterpret_cast<__half *>(out.data_ptr<mfq_half>()),                \
            pairs, routes, static_cast<int>(n_experts),                          \
            static_cast<int>(pool_experts), static_cast<int>(out_per_expert),    \
            ng, nvec, nsign, static_cast<int>(sub_bits),                         \
            static_cast<int>(sign_mode), routed_input)
        launch_by_format(static_cast<int>(format), [&](auto tag) {
            constexpr int F = decltype(tag)::value;
            const char* grouped=mfq::cuda::env_lookup("MFQ_NVQ_MOE_GROUP_DOT");
            if(!exact_reduction && ng<=128 && out_per_expert<=4096 && (!grouped || grouped[0]!='0')) {
                const NvqDeviceWeight w{indices.data_ptr<uint8_t>(),indices.numel(),aux.data_ptr<uint8_t>(),aux.numel(),
                    sub_scale.data_ptr<uint8_t>(),sub_scale.numel(),neuron_scale.data_ptr<float>(),codebook.data_ptr<int8_t>(),
                    int(codebook.numel()),int(pool_experts*out_per_expert),ng,nvec,nsign,int(sub_bits),int(sign_mode)};
                mfq::cuda::NvqRoutedReuseParams reuse;
                reuse.weight=w;reuse.format=F;reuse.expert_local=expert_local.data_ptr<int32_t>();
                reuse.ids=ids.data_ptr<int32_t>();reuse.input=qx.data_ptr<int8_t>();reuse.scales=xscale.data_ptr<float>();
                reuse.output=reinterpret_cast<half*>(out.data_ptr<mfq_half>());
                reuse.experts=int(n_experts);reuse.local_experts=int(pool_experts);reuse.rows=int(out_per_expert);
                reuse.routes=routes;reuse.pairs=pairs;reuse.down=routed_input;
                if(!mfq::cuda::nvq_launch_routed_reuse(reuse,stream))
                    nvq_moe_groups_kernel<F><<<dim3((out_per_expert+3)/4,pairs),dim3(32,4),0,stream>>>(
                    w,qx.data_ptr<int8_t>(),xscale.data_ptr<float>(),ids.data_ptr<int32_t>(),expert_local.data_ptr<int32_t>(),
                    reinterpret_cast<half*>(out.data_ptr<mfq_half>()),int(n_experts),int(pool_experts),int(out_per_expert),routes,routed_input);
                return;
            }
            if (exact_reduction) {
                if (K >= 4096) {
                    if (exact_physical_warps == 4) {
                        NVQ_MOE_EXACT_LAUNCH(4, 8);
                    } else {
                        NVQ_MOE_EXACT_LAUNCH(2, 8);
                    }
                } else {
                    NVQ_MOE_EXACT_LAUNCH(2, 4);
                }
                return;
            }
            if (warps_override == 1) {
                NVQ_MOE_SELECTED_LAUNCH(1, 2);
                return;
            }
            if (warps_override == 2) {
                if (rows_per_block == 4) {
                    NVQ_MOE_SELECTED_LAUNCH(2, 4);
                } else {
                    NVQ_MOE_SELECTED_LAUNCH(2, 2);
                }
                return;
            }
            if (warps_override == 4) {
                NVQ_MOE_SELECTED_LAUNCH(4, 2);
                return;
            }
            if (warps_override == 8) {
                NVQ_MOE_SELECTED_LAUNCH(8, 2);
                return;
            }
            if (K >= 4096) {
                if (rows_per_block == 1) NVQ_MOE_SELECTED_LAUNCH(8, 1);
                else if (rows_per_block == 4) NVQ_MOE_SELECTED_LAUNCH(8, 4);
                else if (rows_per_block == 8) NVQ_MOE_SELECTED_LAUNCH(8, 8);
                else NVQ_MOE_SELECTED_LAUNCH(8, 2);
            } else {
                if (rows_per_block == 1) NVQ_MOE_SELECTED_LAUNCH(4, 1);
                else if (rows_per_block == 4) NVQ_MOE_SELECTED_LAUNCH(4, 4);
                else if (rows_per_block == 8) NVQ_MOE_SELECTED_LAUNCH(4, 8);
                else NVQ_MOE_SELECTED_LAUNCH(4, 2);
            }
        });
#undef NVQ_MOE_EXACT_LAUNCH
#undef NVQ_MOE_SELECTED_LAUNCH
#undef NVQ_MOE_SMALL_LAUNCH
    } else {
        MFQ_RUNTIME_CHECK(ids_dst.is_cuda() && ids_dst.is_contiguous() &&
                    ids_dst.scalar_type() == mfq_tensor_backend::kInt32 &&
                    ids_dst.numel() >= pairs,
                    "NVQ compact route map is missing");
        MFQ_RUNTIME_CHECK(expert_bounds.is_cuda() && expert_bounds.is_contiguous() &&
                    expert_bounds.scalar_type() == mfq_tensor_backend::kInt32 &&
                    expert_bounds.numel() >= n_experts + 1,
                    "NVQ expert bounds are missing");
        MFQ_RUNTIME_CHECK(tile_bounds.is_cuda() && tile_bounds.is_contiguous() &&
                    tile_bounds.scalar_type() == mfq_tensor_backend::kInt32 &&
                    tile_bounds.numel() >= n_experts + 1,
                    "NVQ tile bounds are missing");
        MFQ_RUNTIME_CHECK(tile_experts.is_cuda() && tile_experts.is_contiguous() &&
                    tile_experts.scalar_type() == mfq_tensor_backend::kInt32 &&
                    tile_experts.numel() >= pairs,
                    "NVQ tile expert map is missing");
        constexpr int route_tile = 8;
        const int row_tiles = (static_cast<int>(out_per_expert) + 3) / 4;
        const int max_tiles = (pairs + route_tile - 1) / route_tile +
            static_cast<int>(n_experts);
        const int64_t max_tasks = static_cast<int64_t>(max_tiles) * row_tiles;
        const int blocks = static_cast<int>(
            std::min<int64_t>(max_tasks, 4096));
#define NVQ_MOE_GROUPED_LAUNCH()                                                 \
        nvq_moe_grouped_tile_kernel<F, route_tile><<<                            \
            blocks, dim3(32, 4), 0, stream>>>(                                   \
            indices.data_ptr<uint8_t>(), indices.numel(),                        \
            aux.data_ptr<uint8_t>(), aux.numel(),                                \
            sub_scale.data_ptr<uint8_t>(), sub_scale.numel(),                    \
            neuron_scale.data_ptr<float>(), codebook.data_ptr<int8_t>(),         \
            qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),                     \
            expert_local.data_ptr<int32_t>(), ids_dst.data_ptr<int32_t>(),       \
            expert_bounds.data_ptr<int32_t>(), tile_bounds.data_ptr<int32_t>(),  \
            tile_experts.data_ptr<int32_t>(),                                    \
            reinterpret_cast<__half *>(out.data_ptr<mfq_half>()),                \
            routes, static_cast<int>(n_experts),                                 \
            static_cast<int>(pool_experts), static_cast<int>(out_per_expert),    \
            ng, nvec, nsign, static_cast<int>(sub_bits),                         \
            static_cast<int>(sign_mode), max_tiles, row_tiles, routed_input)
        launch_by_format(static_cast<int>(format), [&](auto tag) {
            constexpr int F = decltype(tag)::value;
            NVQ_MOE_GROUPED_LAUNCH();
        });
#undef NVQ_MOE_GROUPED_LAUNCH
    }
    MFQ_RUNTIME_CHECK(cudaGetLastError() == cudaSuccess,
                "NVQ routed pool matmul kernel launch failed");
    return out;
}

mfq_tensor_backend::Tensor nepq_moe_grouped_matmul_pool_ws_cuda(
    mfq_tensor_backend::Tensor indices,
    mfq_tensor_backend::Tensor aux,
    mfq_tensor_backend::Tensor sub_scale,
    mfq_tensor_backend::Tensor neuron_scale,
    mfq_tensor_backend::Tensor table_pool,
    mfq_tensor_backend::Tensor bank_ids,
    mfq_tensor_backend::Tensor grouped_table_pool,
    mfq_tensor_backend::Tensor x,
    mfq_tensor_backend::Tensor ids,
    mfq_tensor_backend::Tensor expert_local,
    int64_t n_experts,
    int64_t pool_experts,
    int64_t out_per_expert,
    int64_t neuron_len,
    int64_t sub_bits,
    int64_t format,
    bool input_quantized,
    mfq_tensor_backend::Tensor out,
    mfq_tensor_backend::Tensor qx,
    mfq_tensor_backend::Tensor xscale,
    mfq_tensor_backend::Tensor ids_dst,
    mfq_tensor_backend::Tensor expert_bounds,
    mfq_tensor_backend::Tensor tile_bounds,
    mfq_tensor_backend::Tensor tile_experts) {
    check_nepq_common(
        indices, aux, sub_scale, neuron_scale, table_pool, bank_ids,
        neuron_len, sub_bits, format);
    const int expected_grouped_stride = format == kNpq0S
        ? kNpq0STableBytes
        : static_cast<int>(table_pool.size(1));
    MFQ_RUNTIME_CHECK(
        grouped_table_pool.is_cuda() && grouped_table_pool.is_contiguous() &&
        grouped_table_pool.scalar_type() == mfq_tensor_backend::kInt8 &&
        grouped_table_pool.dim() == 2 &&
        grouped_table_pool.size(0) == table_pool.size(0) &&
        grouped_table_pool.size(1) == expected_grouped_stride,
        "NEPQ grouped table pool shape mismatch");
    MFQ_RUNTIME_CHECK(n_experts > 0 && n_experts <= 4096,
                "NEPQ global expert count must be in [1,4096]");
    MFQ_RUNTIME_CHECK(pool_experts > 0 &&
                pool_experts <= static_cast<int64_t>(std::numeric_limits<int>::max()) &&
                out_per_expert > 0 &&
                neuron_scale.numel() == pool_experts * out_per_expert,
                "NEPQ pool shape does not match the flattened row count");
    MFQ_RUNTIME_CHECK(expert_local.is_cuda() && expert_local.is_contiguous() &&
                expert_local.scalar_type() == mfq_tensor_backend::kInt32 &&
                expert_local.dim() == 1 && expert_local.numel() >= n_experts,
                "NEPQ expert-local map must be CUDA contiguous int32");
    MFQ_RUNTIME_CHECK(ids.is_cuda() && ids.is_contiguous() &&
                ids.scalar_type() == mfq_tensor_backend::kInt32 && ids.dim() == 2,
                "NEPQ route IDs must be CUDA contiguous int32 [T,R]");
    MFQ_RUNTIME_CHECK(x.is_cuda() && x.is_contiguous() &&
                x.scalar_type() == mfq_tensor_backend::kFloat16 &&
                (x.dim() == 2 || x.dim() == 3) && x.size(-1) == neuron_len,
                "NEPQ routed input must be CUDA contiguous fp16 [T,K] or [T,R,K]");
    const int tokens = static_cast<int>(ids.size(0));
    const int routes = static_cast<int>(ids.size(1));
    const int pairs = tokens * routes;
    const bool routed_input = x.dim() == 3;
    MFQ_RUNTIME_CHECK(tokens > 0 && routes > 0, "NEPQ route dimensions must be nonzero");
    MFQ_RUNTIME_CHECK((!routed_input && x.size(0) == tokens) ||
                (routed_input && x.size(0) == tokens && x.size(1) == routes),
                "NEPQ input leading dimensions do not match routes");
    MFQ_RUNTIME_CHECK(out.is_cuda() && out.is_contiguous() &&
                out.scalar_type() == mfq_tensor_backend::kFloat16 && out.dim() == 3 &&
                out.size(0) == tokens && out.size(1) == routes &&
                out.size(2) == out_per_expert,
                "NEPQ output must be CUDA contiguous fp16 [T,R,O]");

    const int K = static_cast<int>(neuron_len);
    const int ng = (K + kGroupSize - 1) / kGroupSize;
    const int K_pad = ng * kGroupSize;
    const int input_rows = routed_input ? pairs : tokens;
    MFQ_RUNTIME_CHECK(qx.is_cuda() && qx.is_contiguous() &&
                qx.scalar_type() == mfq_tensor_backend::kInt8 && qx.dim() == 2 &&
                qx.size(0) >= input_rows && qx.size(1) >= K_pad,
                "NEPQ routed qx workspace mismatch");
    MFQ_RUNTIME_CHECK(xscale.is_cuda() && xscale.is_contiguous() &&
                xscale.scalar_type() == mfq_tensor_backend::kFloat32 && xscale.dim() == 2 &&
                xscale.size(0) >= input_rows && xscale.size(1) >= ng,
                "NEPQ routed xscale workspace mismatch");
    const int nvec = K / 8;
    const int nsign = (K + 7) / 8;
    const int nsuper = (ng + kNepqGroupsPerSupergroup - 1) /
        kNepqGroupsPerSupergroup;
    const int table_stride = static_cast<int>(table_pool.size(1));
    const int grouped_table_stride =
        static_cast<int>(grouped_table_pool.size(1));
    cudaStream_t stream = mfq_current_cuda_stream();
    if (!input_quantized) {
        nvq_quantize_x_gs24_kernel<<<dim3(input_rows, ng), 32, 0, stream>>>(
            reinterpret_cast<const __half *>(x.data_ptr<mfq_half>()),
            qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), input_rows, K, ng);
    }

    if (tokens <= 8) {
        constexpr int rows_per_block = 2;
        const int row_blocks =
            (static_cast<int>(out_per_expert) + rows_per_block - 1) /
            rows_per_block;
#define NEPQ_MOE_POOL_SMALL_LAUNCH(NWARPS_VALUE)                                 \
        nepq_moe_mmvq_kernel<F, NWARPS_VALUE, rows_per_block><<<                 \
            dim3(row_blocks, pairs), dim3(32, NWARPS_VALUE), 0, stream>>>(       \
            indices.data_ptr<uint8_t>(), indices.numel(),                        \
            aux.data_ptr<uint8_t>(), aux.numel(),                                \
            sub_scale.data_ptr<uint8_t>(), sub_scale.numel(),                    \
            neuron_scale.data_ptr<float>(), table_pool.data_ptr<int8_t>(),       \
            bank_ids.data_ptr<uint8_t>(), qx.data_ptr<int8_t>(),                 \
            xscale.data_ptr<float>(), ids.data_ptr<int32_t>(),                   \
            expert_local.data_ptr<int32_t>(),                                    \
            reinterpret_cast<__half *>(out.data_ptr<mfq_half>()),                \
            pairs, routes, static_cast<int>(n_experts),                          \
            static_cast<int>(pool_experts), static_cast<int>(out_per_expert),    \
            ng, nvec, nsign, nsuper, table_stride,                               \
            static_cast<int>(sub_bits), routed_input)
        launch_nepq_by_format(static_cast<int>(format), [&](auto tag) {
            constexpr int F = decltype(tag)::value;
            if (K >= 4096) NEPQ_MOE_POOL_SMALL_LAUNCH(8);
            else NEPQ_MOE_POOL_SMALL_LAUNCH(4);
        });
#undef NEPQ_MOE_POOL_SMALL_LAUNCH
    } else {
        MFQ_RUNTIME_CHECK(ids_dst.is_cuda() && ids_dst.is_contiguous() &&
                    ids_dst.scalar_type() == mfq_tensor_backend::kInt32 &&
                    ids_dst.numel() >= pairs,
                    "NEPQ compact route map is missing");
        MFQ_RUNTIME_CHECK(expert_bounds.is_cuda() && expert_bounds.is_contiguous() &&
                    expert_bounds.scalar_type() == mfq_tensor_backend::kInt32 &&
                    expert_bounds.numel() >= n_experts + 1,
                    "NEPQ expert bounds are missing");
        MFQ_RUNTIME_CHECK(tile_bounds.is_cuda() && tile_bounds.is_contiguous() &&
                    tile_bounds.scalar_type() == mfq_tensor_backend::kInt32 &&
                    tile_bounds.numel() >= n_experts + 1,
                    "NEPQ tile bounds are missing");
        MFQ_RUNTIME_CHECK(tile_experts.is_cuda() && tile_experts.is_contiguous() &&
                    tile_experts.scalar_type() == mfq_tensor_backend::kInt32 &&
                    tile_experts.numel() >= pairs,
                    "NEPQ tile expert map is missing");
        constexpr int route_tile = 8;
        const int row_tiles = (static_cast<int>(out_per_expert) + 3) / 4;
        const int max_tiles = (pairs + route_tile - 1) / route_tile +
            static_cast<int>(n_experts);
        const int64_t max_tasks = static_cast<int64_t>(max_tiles) * row_tiles;
        const int blocks = static_cast<int>(
            std::min<int64_t>(max_tasks, 4096));
#define NEPQ_MOE_POOL_GROUPED_LAUNCH()                                           \
        nepq_moe_grouped_tile_kernel<F, route_tile><<<                           \
            blocks, dim3(32, 4), 0, stream>>>(                                   \
            indices.data_ptr<uint8_t>(), indices.numel(),                        \
            aux.data_ptr<uint8_t>(), aux.numel(),                                \
            sub_scale.data_ptr<uint8_t>(), sub_scale.numel(),                    \
            neuron_scale.data_ptr<float>(), grouped_table_pool.data_ptr<int8_t>(), \
            bank_ids.data_ptr<uint8_t>(), qx.data_ptr<int8_t>(),                 \
            xscale.data_ptr<float>(), expert_local.data_ptr<int32_t>(),          \
            ids_dst.data_ptr<int32_t>(), expert_bounds.data_ptr<int32_t>(),      \
            tile_bounds.data_ptr<int32_t>(), tile_experts.data_ptr<int32_t>(),   \
            reinterpret_cast<__half *>(out.data_ptr<mfq_half>()),                \
            routes, static_cast<int>(n_experts),                                 \
            static_cast<int>(pool_experts), static_cast<int>(out_per_expert),    \
            ng, nvec, nsign, nsuper, grouped_table_stride,                       \
            static_cast<int>(sub_bits), max_tiles, row_tiles, routed_input)
        launch_nepq_by_format(static_cast<int>(format), [&](auto tag) {
            constexpr int F = decltype(tag)::value;
            NEPQ_MOE_POOL_GROUPED_LAUNCH();
        });
#undef NEPQ_MOE_POOL_GROUPED_LAUNCH
    }
    MFQ_RUNTIME_CHECK(cudaGetLastError() == cudaSuccess,
                "NEPQ routed pool matmul kernel launch failed");
    return out;
}
