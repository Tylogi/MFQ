// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Apple Inc.
// Adapted from jundot/omlx at a435a3736071286fdb2b74f486b3505f23f3c194.

constexpr const char* kQsaPrefillIndexerSource = R"MFQ_QSA(
using namespace mlx::steel;
struct DSATopKParams { int rows; int L; int K; int topk; bool causal_valid_prefix; };

template <typename T, int BM, int BN, int BK, int WM, int WN>
[[kernel, max_total_threads_per_threadgroup(WM* WN * 32)]] void
qwen4_qsa_indexer_score(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    device float* O [[buffer(2)]],
    const constant GEMMParams* params [[buffer(3)]],
    const constant int& mask_ratio [[buffer(4)]],
    const constant int& mask_q_offset [[buffer(5)]],
    const constant float& score_divisor [[buffer(6)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
  (void)lid;

  using gemm_kernel = GEMMKernel<
      T,
      T,
      BM,
      BN,
      BK,
      WM,
      WN,
      false,
      true,
      true,
      true,
      float>;
  using loader_a_t = typename gemm_kernel::loader_a_t;
  using loader_b_t = typename gemm_kernel::loader_b_t;
  using mma_t = typename gemm_kernel::mma_t;

  const int tid_y = ((tid.y) << params->swizzle_log) +
      ((tid.x) & ((1 << params->swizzle_log) - 1));
  const int tid_x = (tid.x) >> params->swizzle_log;
  if (params->tiles_n <= tid_x || params->tiles_m <= tid_y) {
    return;
  }

  constexpr int H = 4;
  const int M = params->M;
  const int N = params->N;
  const int D = params->K;
  const int c_row = tid_y * BM;
  const int c_col = tid_x * BN;
  const short tgp_bm = short(metal::min(BM, M - c_row));
  const short tgp_bn = short(metal::min(BN, N - c_col));

  Q += size_t(tid.z) * H * M * D;
  K += size_t(tid.z) * N * D;
  O += size_t(tid.z) * M * N + size_t(c_row) * params->ldd + c_col;

  threadgroup T As[gemm_kernel::tgp_mem_size_a];
  threadgroup T Bs[gemm_kernel::tgp_mem_size_b];
  thread mma_t mma_op(simd_group_id, simd_lane_id);

  float accum[decltype(mma_op.Ctile)::kElemsPerTile];
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < decltype(mma_op.Ctile)::kElemsPerTile; ++i) {
    accum[i] = 0.0f;
  }

  STEEL_PRAGMA_UNROLL
  for (short h = 0; h < H; ++h) {
    mma_op.Ctile.clear();
    const device T* A = Q + size_t(h) * M * D + size_t(c_row) * D;
    const device T* B = K + size_t(c_col) * D;
    thread loader_a_t loader_a(A, params->lda, As, simd_group_id, simd_lane_id);
    thread loader_b_t loader_b(B, params->ldb, Bs, simd_group_id, simd_lane_id);

    for (int d = 0; d < params->gemm_k_iterations_aligned; ++d) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (tgp_bm == BM) {
        loader_a.load_unsafe();
      } else {
        loader_a.load_safe(short2(BK, tgp_bm));
      }
      if (tgp_bn == BN) {
        loader_b.load_unsafe();
      } else {
        loader_b.load_safe(short2(BK, tgp_bn));
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(As, Bs);
      loader_a.next();
      loader_b.next();
    }
    threadgroup_barrier(mem_flags::mem_none);

    short ai = 0;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < decltype(mma_op.Ctile)::kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < decltype(mma_op.Ctile)::kTileCols; ++j) {
        thread const auto& frag = mma_op.Ctile.frag_at(i, j);
        STEEL_PRAGMA_UNROLL
        for (short e = 0; e < decltype(mma_op.Ctile)::kElemsPerFrag; ++e) {
          accum[ai++] += metal::max(frag[e], 0.0f);
        }
      }
    }
  }

  const float pooled_sentinel = as_type<float>(uint(0xFF7FFFFF));
  device float* Dst = O + size_t(mma_op.sm) * params->ldd + mma_op.sn;
  short ai = 0;
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < decltype(mma_op.Ctile)::kTileRows; ++i) {
    const int row = c_row + mma_op.sm + i * mma_t::TM_stride;
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < decltype(mma_op.Ctile)::kTileCols; ++j) {
      const int col_base = c_col + mma_op.sn + j * mma_t::TN_stride;
      const int out_base =
          (i * decltype(mma_op.Ctile)::kFragRows) * WM * params->ldd +
          (j * decltype(mma_op.Ctile)::kFragCols) * WN;
      STEEL_PRAGMA_UNROLL
      for (short e = 0; e < decltype(mma_op.Ctile)::kElemsPerFrag; ++e) {
        const int col = col_base + e;
        const bool pooled_masked =
            col >= (mask_q_offset + row + 1) / mask_ratio;
        const float value = pooled_masked
            ? pooled_sentinel
            : accum[ai] / score_divisor;
        if (row < M && col < N) {
          Dst[out_base + e] = value;
        }
        ai++;
      }
    }
  }
}

template <bool NUMERIC_ZERO_TIES>
METAL_FUNC uint dsa_ordered_key_32(float x) {
  uint bits = as_type<uint>(x);
  if (NUMERIC_ZERO_TIES && (bits & 0x7fffffffu) == 0u) {
    bits = 0u;
  }
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

template <int TOPK, int THREADS, bool HIGHEST_INDEX_TIES>
[[kernel, max_total_threads_per_threadgroup(THREADS)]] void
dspark_fp32_topk_indices(
    const device float* scores [[buffer(0)]],
    device uint* out [[buffer(1)]],
    const constant DSATopKParams* params [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]]) {
  if (row >= uint(params->rows)) {
    return;
  }

  constexpr uint kSimdgroups = THREADS / 32;
  threadgroup atomic_uint hist[256];
  threadgroup uint state[2];
  threadgroup uint partial_a[kSimdgroups];
  threadgroup uint partial_b[kSimdgroups];

  if (tid == 0) {
    state[0] = 0;
    state[1] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint K = uint(params->K);
  const uint segment = (K + THREADS - 1) / THREADS;
  const uint logical_tid = HIGHEST_INDEX_TIES ? THREADS - 1 - tid : tid;
  const uint start = logical_tid * segment;
  const uint stop = metal::min(start + segment, K);
  const uint count = start < K ? stop - start : 0u;
  const device float* row_scores = scores + size_t(row) * K;

  for (int shift = 24; shift >= 0; shift -= 8) {
    if (tid < 256) {
      atomic_store_explicit(&hist[tid], 0, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint prefix = state[0];
    for (uint step = 0; step < count; ++step) {
      const uint index = HIGHEST_INDEX_TIES ? stop - 1 - step : start + step;
      const uint key =
          dsa_ordered_key_32<HIGHEST_INDEX_TIES>(row_scores[index]);
      const bool prefix_match = shift == 24 ||
          (key >> uint(shift + 8)) == (prefix >> uint(shift + 8));
      if (prefix_match) {
        atomic_fetch_add_explicit(
            &hist[(key >> uint(shift)) & 255u], 1, memory_order_relaxed);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
      uint greater = state[1];
      uint selected_byte = 0;
      for (int byte = 255; byte >= 0; --byte) {
        const uint count =
            atomic_load_explicit(&hist[byte], memory_order_relaxed);
        if (greater + count >= uint(TOPK)) {
          selected_byte = uint(byte);
          break;
        }
        greater += count;
      }
      state[0] |= selected_byte << uint(shift);
      state[1] = greater;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const uint threshold = state[0];
  const uint greater_total = state[1];
  uint local_greater = 0;
  uint local_ties = 0;
  for (uint step = 0; step < count; ++step) {
    const uint index = HIGHEST_INDEX_TIES ? stop - 1 - step : start + step;
    const uint key =
        dsa_ordered_key_32<HIGHEST_INDEX_TIES>(row_scores[index]);
    local_greater += key > threshold ? 1u : 0u;
    local_ties += key == threshold ? 1u : 0u;
  }

  const uint lane = tid & 31u;
  const uint simd = tid >> 5;
  uint pre_greater = metal::simd_prefix_exclusive_sum(local_greater);
  uint pre_ties = metal::simd_prefix_exclusive_sum(local_ties);
  if (lane == 31) {
    partial_a[simd] = pre_greater + local_greater;
    partial_b[simd] = pre_ties + local_ties;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd == 0) {
    const uint group_greater = lane < kSimdgroups ? partial_a[lane] : 0u;
    const uint group_ties = lane < kSimdgroups ? partial_b[lane] : 0u;
    const uint group_pre_greater =
        metal::simd_prefix_exclusive_sum(group_greater);
    const uint group_pre_ties = metal::simd_prefix_exclusive_sum(group_ties);
    if (lane < kSimdgroups) {
      partial_a[lane] = group_pre_greater;
      partial_b[lane] = group_pre_ties;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  pre_greater += partial_a[simd];
  pre_ties += partial_b[simd];

  const uint tie_budget = uint(TOPK) - greater_total;
  const uint selected_ties = pre_ties >= tie_budget
      ? 0u
      : metal::min(local_ties, tie_budget - pre_ties);
  const uint local_selected = local_greater + selected_ties;

  uint pre_selected = metal::simd_prefix_exclusive_sum(local_selected);
  if (lane == 31) {
    partial_a[simd] = pre_selected + local_selected;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd == 0) {
    const uint group_selected = lane < kSimdgroups ? partial_a[lane] : 0u;
    const uint group_pre_selected =
        metal::simd_prefix_exclusive_sum(group_selected);
    if (lane < kSimdgroups) {
      partial_a[lane] = group_pre_selected;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint output_pos = partial_a[simd] + pre_selected;
  uint ties_seen = pre_ties;
  device uint* row_out = out + size_t(row) * TOPK;
  for (uint step = 0; step < count; ++step) {
    const uint index = HIGHEST_INDEX_TIES ? stop - 1 - step : start + step;
    const uint key =
        dsa_ordered_key_32<HIGHEST_INDEX_TIES>(row_scores[index]);
    bool selected = key > threshold;
    if (key == threshold) {
      selected = ties_seen < tie_budget;
      ties_seen++;
    }
    if (selected) {
      // The reversed traversal emits descending indices; mirror them so the
      // Qwen4 caller receives the chronological order without a sort.
      row_out[HIGHEST_INDEX_TIES ? uint(TOPK) - 1u - output_pos : output_pos] = index;
      output_pos++;
    }
  }
}

template [[host_name("mfq_qsa_prefill_scores_f16")]] [[kernel]] decltype(qwen4_qsa_indexer_score<half, 64, 64, 16, 2, 2>) qwen4_qsa_indexer_score<half, 64, 64, 16, 2, 2>;
template [[host_name("mfq_qsa_prefill_scores_bf16")]] [[kernel]] decltype(qwen4_qsa_indexer_score<bfloat16_t, 64, 64, 16, 2, 2>) qwen4_qsa_indexer_score<bfloat16_t, 64, 64, 16, 2, 2>;
template [[host_name("mfq_qsa_prefill_topk512")]] [[kernel]] decltype(dspark_fp32_topk_indices<512, 256, true>) dspark_fp32_topk_indices<512, 256, true>;
)MFQ_QSA";
