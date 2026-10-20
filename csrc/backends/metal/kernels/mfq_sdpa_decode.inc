// SPDX-License-Identifier: Apache-2.0
// Copyright © 2024 Apple Inc.
// Adapted from jundot/omlx decode_fast at 328edc756010693e61f8bd4e47f00996e0758176.

constexpr const char* kSdpaDecodeSource = R"MFQ_SDPA(
#include <metal_simdgroup>

using namespace metal;

constant bool has_mask [[function_constant(20)]];
constant bool query_transposed [[function_constant(21)]];
constant bool do_causal [[function_constant(22)]];
constant bool bool_mask [[function_constant(23)]];
constant bool float_mask [[function_constant(24)]];
constant bool has_sinks [[function_constant(25)]];
constant int blocks [[function_constant(26)]];
constant bool sparse_blocks [[function_constant(27)]];
constant int sparse_block_size [[function_constant(28)]];

inline long mfq_sdpa_sparse_token(const device int* ids, long stride, int visible,
    int complete, int valid_blocks, int block_tokens, int slot) {
  long token;
  if (slot < block_tokens) {
    const int index = slot / sparse_block_size;
    if (index >= valid_blocks) return -1;
    const int block = ids[long(index) * stride];
    if (block < 0) return -1;
    token = long(block) * sparse_block_size + slot % sparse_block_size;
  } else {
    token = long(complete) * sparse_block_size + slot - block_tokens;
  }
  return token < visible ? token : -1;
}

template <typename T, int N>
inline void mfq_sdpa_load_vec(thread T (&dst)[N], const device T* p) {
  if constexpr (N == 2 || N == 4) {
    *reinterpret_cast<thread vec<T, N>*>(dst) =
        *reinterpret_cast<const device vec<T, N>*>(p);
  } else if constexpr (N == 8 && sizeof(T) == 2) {
    *reinterpret_cast<thread vec<T, 8>*>(dst) =
        *reinterpret_cast<const device vec<T, 8>*>(p);
  } else if constexpr (N == 8) {
    *reinterpret_cast<thread vec<T, 4>*>(dst) =
        *reinterpret_cast<const device vec<T, 4>*>(p);
    *reinterpret_cast<thread vec<T, 4>*>(dst + 4) =
        *reinterpret_cast<const device vec<T, 4>*>(p + 4);
  } else {
#pragma unroll
    for (int i = 0; i < N; i++) {
      dst[i] = p[i];
    }
  }
}

template <typename T, int D, int V = D, int MAX_THREADS = 1024>
[[kernel, max_total_threads_per_threadgroup(MAX_THREADS)]] void mfq_sdpa_decode_2pass_1(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device float* out [[buffer(3)]],
    device float* sums [[buffer(4)]],
    device float* maxs [[buffer(5)]],
    const device int* selected_blocks [[buffer(6), function_constant(sparse_blocks)]],
    const constant int& N [[buffer(7)]],
    const constant size_t& k_head_stride [[buffer(8)]],
    const constant size_t& k_seq_stride [[buffer(9)]],
    const constant size_t& v_head_stride [[buffer(10)]],
    const constant size_t& v_seq_stride [[buffer(11)]],
    const constant float& scale [[buffer(12)]],
    const device bool* bmask [[buffer(13), function_constant(bool_mask)]],
    const device T* fmask [[buffer(14), function_constant(float_mask)]],
    const constant int& mask_kv_seq_stride
    [[buffer(15), function_constant(has_mask)]],
    const constant int& mask_q_seq_stride
    [[buffer(16), function_constant(has_mask)]],
    const constant int& mask_head_stride
    [[buffer(17), function_constant(has_mask)]],
    const device T* sinks [[buffer(18), function_constant(has_sinks)]],
    constant MfqSparseBlockGqaParams& sparse_params [[buffer(19), function_constant(sparse_blocks)]],
    uint3 tptg [[threads_per_threadgroup]],
    uint3 tidtg [[thread_position_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BD = 32;
  constexpr int TK = 4;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;

  typedef float U;

  thread U q[qk_per_thread];
  thread U o[v_per_thread] = {0};

  const int kv_head_idx = tid.x;
  const int batch_idx = tid.y;
  const int block_idx = tid.z;
  const int gqa_factor = tptg.y;
  const int q_seq_len = tptg.z;
  const int q_seq_idx = tidtg.z;
  const int q_head_idx = gqa_factor * kv_head_idx + tidtg.y;
  const int num_kv_heads = tpg.x;
  const int num_q_heads = num_kv_heads * gqa_factor;
  const int q_batch_head_idx = (batch_idx * num_q_heads + q_head_idx);
  const int o_offset = q_batch_head_idx * q_seq_len + q_seq_idx;
  int q_offset =
      query_transposed ? num_q_heads * q_seq_idx + q_batch_head_idx : o_offset;
  if (sparse_blocks) {
    q_offset = ((batch_idx / sparse_params.queries) * num_q_heads + q_head_idx)
        * sparse_params.queries + batch_idx % sparse_params.queries;
  }

  queries += q_offset * D + simd_lid * qk_per_thread;

  int sparse_visible = 0, sparse_complete = 0, sparse_valid = 0, sparse_tokens = 0;
  long sparse_stride = 0;
  if (sparse_blocks) {
    const int position = batch_idx % sparse_params.queries;
    const int visible = sparse_params.query_offset + position + 1;
    sparse_visible = min(sparse_params.keys, visible);
    sparse_complete = visible / sparse_block_size;
    sparse_valid = min(sparse_params.selected_blocks, sparse_complete);
    sparse_tokens = sparse_params.selected_blocks * sparse_block_size;
    sparse_stride = sparse_params.block_strides[2];
    selected_blocks += long(batch_idx / sparse_params.queries) * sparse_params.block_strides[0]
        + long(position) * sparse_params.block_strides[1];
  }

  const int chunk = (N + blocks - 1) / blocks;
  const int k_start = block_idx * chunk;
  const int k_end = min(N, k_start + chunk);

  const int kv_batch_head_idx = (sparse_blocks ? batch_idx / sparse_params.queries : batch_idx)
      * num_kv_heads + kv_head_idx;
  keys += (sparse_blocks ? (batch_idx / sparse_params.queries) * sparse_params.key_strides[0]
      + kv_head_idx * sparse_params.key_strides[1] : kv_batch_head_idx * k_head_stride)
      + (sparse_blocks ? 0 : k_start * k_seq_stride) +
      simd_lid * qk_per_thread;
  values += (sparse_blocks ? (batch_idx / sparse_params.queries) * sparse_params.value_strides[0]
      + kv_head_idx * sparse_params.value_strides[1] : kv_batch_head_idx * v_head_stride)
      + (sparse_blocks ? 0 : k_start * v_seq_stride) +
      simd_lid * v_per_thread;
  out += o_offset * blocks * V + block_idx * V + simd_lid * v_per_thread;
  if (bool_mask) {
    bmask += batch_idx * mask_head_stride +
        k_start * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
  }
  if (float_mask) {
    fmask += batch_idx * mask_head_stride +
        k_start * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
  }
  sums += o_offset * blocks + block_idx;
  maxs += o_offset * blocks + block_idx;

  {
    thread T qt[qk_per_thread];
    mfq_sdpa_load_vec<T, qk_per_thread>(qt, queries);
    for (int i = 0; i < qk_per_thread; i++) {
      q[i] = static_cast<U>(scale) * static_cast<U>(qt[i]);
    }
  }

  U max_score = Limits<U>::finite_min;
  U sum_exp_score = 0;
  if (has_sinks && block_idx == 0) {
    max_score = static_cast<U>(sinks[q_head_idx]);
    sum_exp_score = 1;
  }

  int i = k_start;
  for (; i + TK <= k_end; i += TK) {
    thread T ks[TK][qk_per_thread];
    U scores[TK];
    bool use_key[TK];
    long selected[TK];
#pragma unroll
    for (int t = 0; t < TK; t++) {
      selected[t] = sparse_blocks ? mfq_sdpa_sparse_token(selected_blocks, sparse_stride,
          sparse_visible, sparse_complete, sparse_valid, sparse_tokens, i + t) : t;
      mfq_sdpa_load_vec<T, qk_per_thread>(ks[t], keys + max(0L, selected[t]) * k_seq_stride);
    }
#pragma unroll
    for (int t = 0; t < TK; t++) {
      U score = 0;
      for (int j = 0; j < qk_per_thread; j++) {
        score += q[j] * static_cast<U>(ks[t][j]);
      }
      score = simd_sum(score);

      use_key[t] = !sparse_blocks || selected[t] >= 0;
      if (do_causal) {
        use_key[t] = i + t <= (N - q_seq_len + int(q_seq_idx));
      } else if (bool_mask) {
        use_key[t] = bmask[t * mask_kv_seq_stride];
      } else if (float_mask) {
        use_key[t] = fmask[t * mask_kv_seq_stride] >= Limits<T>::finite_min;
      }
      if (use_key[t] && float_mask) {
        score += static_cast<U>(fmask[t * mask_kv_seq_stride]);
      }
      scores[t] = use_key[t] ? score : Limits<U>::finite_min;
    }

    U tile_max = scores[0];
#pragma unroll
    for (int t = 1; t < TK; t++) {
      tile_max = max(tile_max, scores[t]);
    }
    U new_max = max(max_score, tile_max);

    thread T vs[TK][v_per_thread];
#pragma unroll
    for (int t = 0; t < TK; t++) {
      mfq_sdpa_load_vec<T, v_per_thread>(vs[t], values + max(0L, selected[t]) * v_seq_stride);
    }

    if (new_max > max_score) {
      U factor = fast::exp(max_score - new_max);
      sum_exp_score *= factor;
      for (int j = 0; j < v_per_thread; j++) {
        o[j] *= factor;
      }
      max_score = new_max;
    }

#pragma unroll
    for (int t = 0; t < TK; t++) {
      U p = use_key[t] ? fast::exp(scores[t] - max_score) : U(0);
      sum_exp_score += p;
#pragma unroll
      for (int j = 0; j < v_per_thread; j++) {
        o[j] += p * static_cast<U>(vs[t][j]);
      }
    }

    if (!sparse_blocks) {
      keys += TK * int(k_seq_stride);
      values += TK * int(v_seq_stride);
    }
    if (bool_mask) {
      bmask += TK * mask_kv_seq_stride;
    }
    if (float_mask) {
      fmask += TK * mask_kv_seq_stride;
    }
  }

  for (; i < k_end; i++) {
    const long selected = sparse_blocks ? mfq_sdpa_sparse_token(selected_blocks, sparse_stride,
        sparse_visible, sparse_complete, sparse_valid, sparse_tokens, i) : 0;
    bool use_key = !sparse_blocks || selected >= 0;
    if (do_causal) {
      use_key = i <= (N - q_seq_len + int(q_seq_idx));
    } else if (bool_mask) {
      use_key = bmask[0];
    } else if (float_mask) {
      use_key = (fmask[0] >= Limits<T>::finite_min);
    }
    if (use_key) {
      thread T k[qk_per_thread];
      mfq_sdpa_load_vec<T, qk_per_thread>(k, keys + selected * k_seq_stride);
      U score = 0;
      for (int j = 0; j < qk_per_thread; j++) {
        score += q[j] * static_cast<U>(k[j]);
      }
      score = simd_sum(score);

      if (float_mask) {
        score += fmask[0];
      }

      U new_max = max(max_score, score);
      U factor = fast::exp(max_score - new_max);
      U exp_score = fast::exp(score - new_max);

      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      thread T vt[v_per_thread];
      mfq_sdpa_load_vec<T, v_per_thread>(vt, values + selected * v_seq_stride);
      for (int j = 0; j < v_per_thread; j++) {
        o[j] = o[j] * factor + exp_score * static_cast<U>(vt[j]);
      }
    }

    if (!sparse_blocks) {
      keys += int(k_seq_stride);
      values += int(v_seq_stride);
    }
    if (bool_mask) {
      bmask += mask_kv_seq_stride;
    }
    if (float_mask) {
      fmask += mask_kv_seq_stride;
    }
  }

  if (simd_lid == 0) {
    sums[0] = sum_exp_score;
    maxs[0] = max_score;
  }

  for (int i = 0; i < v_per_thread; i++) {
    out[i] = o[i];
  }
}

template <typename T, int D>
[[kernel, max_total_threads_per_threadgroup(1024)]] void mfq_sdpa_decode_2pass_2(
    const device float* partials [[buffer(0)]],
    const device float* sums [[buffer(1)]],
    const device float* maxs [[buffer(2)]],
    device T* out [[buffer(3)]],
    const constant int& blocks [[buffer(4)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int elem_per_thread = D / BD;

  typedef float U;

  thread U o[elem_per_thread] = {0};
  threadgroup U outputs[BN * BD];

  const int head_idx = tid.x;
  const int q_seq_idx = tid.y;
  const int q_offset = head_idx * tpg.y + q_seq_idx;
  partials += q_offset * blocks * D + simd_gid * D + simd_lid * elem_per_thread;
  sums += q_offset * blocks;
  maxs += q_offset * blocks;
  out += q_offset * D + simd_gid * elem_per_thread;

  U sum_exp_score = 0.0;
  U max_score = Limits<U>::finite_min;

  for (int b = 0; b < blocks / BN; ++b) {
    max_score = max(max_score, maxs[simd_lid + BN * b]);
  }
  max_score = simd_max(max_score);

  for (int b = 0; b < blocks / BN; ++b) {
    U factor = fast::exp(maxs[simd_lid + BN * b] - max_score);
    sum_exp_score += factor * sums[simd_lid + BN * b];
  }
  sum_exp_score = simd_sum(sum_exp_score);

  for (int b = 0; b < blocks / BN; ++b) {
    U factor = fast::exp(maxs[simd_gid] - max_score);

    for (int i = 0; i < elem_per_thread; i++) {
      o[i] += factor * partials[i];
    }
    maxs += BN;
    sums += BN;
    partials += BN * D;
  }

  for (int i = 0; i < elem_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid]);
    o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (simd_lid == 0) {
    for (int i = 0; i < elem_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}
)MFQ_SDPA";
