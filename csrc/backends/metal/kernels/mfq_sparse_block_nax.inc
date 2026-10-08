// SPDX-License-Identifier: Apache-2.0
// Adapted from jundot/omlx qsa_nax.py, with FP16, batching and index bounds.
constexpr const char* kSparseBlockNaxSource = R"MFQ_METAL(

    constexpr int D = 256;
    constexpr int TDH = 8;
    constexpr int SB = 8;
#if MFQ_QSA_PREFILL
    threadgroup float xchg[16 * 32];
#else
    threadgroup float xchg[2][8 * 32];
#endif

    const int tq = int(threadgroup_position_in_grid.x);
    const int kvh = int(threadgroup_position_in_grid.y);
    const int batch = int(threadgroup_position_in_grid.z);
    q += size_t(batch) * q_strides[0];
    k += size_t(batch) * k_strides[0];
    v += size_t(batch) * v_strides[0];
    const ushort dh = simdgroup_index_in_threadgroup;
    const ushort lane = thread_index_in_simdgroup;
    const short qid = lane >> 2;
    const short fm = (qid & 4) | ((lane >> 1) & 3);
    const short fn = ((qid & 2) | (lane & 1)) * 4;

    const int q_offset = params[0];
    const int kL = params[1];
    const float scale2 = scale[0] * 1.44269504089f;
    const int p = q_offset + tq;
    const int complete = (p + 1) >> 2;
#if MFQ_QSA_PREFILL
    constexpr int nsel = TOPK;
#else
    const int nsel = min(TOPK, complete);
#endif
    const int ntail = p + 1 - (complete << 2);
    const int U = nsel + (ntail > 0 ? 1 : 0);
    const int nsteps = (U + SB - 1) / SB;
    const bool r1_ok = fm + 8 < GQA;
    const int h0 = kvh * GQA + fm;
    const int h1 = kvh * GQA + (r1_ok ? fm + 8 : fm);

    const int64_t sqh = q_strides[1], sql = q_strides[2];
    const uint skl = uint(k_strides[2]), svl = uint(v_strides[2]);
    const short fcol = 2 * fn;
    const device T* kb = (const device T*)k + kvh * k_strides[1] + dh * 128 + fcol;
    const device T* vb = (const device T*)v + kvh * v_strides[1] + dh * 128 + fcol;
    const device T* qp0 = (const device T*)q + h0 * sqh + tq * sql + dh * 128 + fcol;
    const device T* qp1 = (const device T*)q + h1 * sqh + tq * sql + dh * 128 + fcol;
    const device int* blk = sel + (size_t(batch) * size_t(params[2]) + size_t(tq)) * TOPK;

    constexpr auto qk_desc = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, 32, false, true, true, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<qk_desc, metal::execution_simdgroup> qk_op;
    constexpr auto pv_desc = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, 32, false, false, true, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<pv_desc, metal::execution_simdgroup> pv_op;

    auto qa_ct = qk_op.template get_left_input_cooperative_tensor<T, T, float>();
    auto kb_ct = qk_op.template get_right_input_cooperative_tensor<T, T, float>();
    using qa_t = metal::remove_addrspace_t<decltype(qa_ct)>;
    using kb_t = metal::remove_addrspace_t<decltype(kb_ct)>;
    auto pa_ct = pv_op.template get_left_input_cooperative_tensor<half, T, float>();
    auto vb_ct = pv_op.template get_right_input_cooperative_tensor<half, T, float>();
    using pa_t = metal::remove_addrspace_t<decltype(pa_ct)>;
    using vb_t = metal::remove_addrspace_t<decltype(vb_ct)>;

    auto of0 = pv_op.template get_destination_cooperative_tensor<pa_t, vb_t, float>();
    auto of1 = pv_op.template get_destination_cooperative_tensor<pa_t, vb_t, float>();
    auto of2 = pv_op.template get_destination_cooperative_tensor<pa_t, vb_t, float>();
    auto of3 = pv_op.template get_destination_cooperative_tensor<pa_t, vb_t, float>();
    UNROLL for (short i = 0; i < 16; ++i) {
      of0[i] = 0.0f;
      of1[i] = 0.0f;
      of2[i] = 0.0f;
      of3[i] = 0.0f;
    }
    float max_s[2] = {-FLT_MAX, -FLT_MAX};
    float sum_s[2] = {0.0f, 0.0f};

    auto blk_at = [&](int u) -> int { return u < nsel ? blk[u] : complete; };
    int nrow_b[4];
    int ncol_n[2];
    auto fetch = [&](int u0) {
      UNROLL for (short r = 0; r < 4; ++r) {
        const int u = u0 + ((16 * (r >> 1) + fm + 8 * (r & 1)) >> 2);
#if MFQ_QSA_PREFILL
        nrow_b[r] = u < U ? blk_at(u) : 0;
#else
        nrow_b[r] = u < U ? clamp(blk_at(u), 0, (kL - 1) / 4) : 0;
#endif
      }
      UNROLL for (short f = 0; f < 2; ++f) {
        const int u = u0 + 4 * f + (fn >> 2);
#if MFQ_QSA_PREFILL
        ncol_n[f] = u < nsel ? 4 : (u < U ? ntail : 0);
#else
        const int block = u < U ? blk_at(u) : -1;
        const int first = block >= 0 && block <= (kL - 1) / 4 ? block * 4 : kL;
        ncol_n[f] = u < U ? clamp(min(p + 1, kL) - first, 0, 4) : 0;
#endif
      }
    };
    fetch(0);

    for (int step = 0; step < nsteps; ++step) {

      uint krow[4];
      UNROLL for (short r = 0; r < 4; ++r) {
#if MFQ_QSA_PREFILL
        krow[r] = uint(nrow_b[r] * 4 + (fm & 3));
        if (step == TOPK / SB) krow[r] = min(krow[r], uint(kL - 1));
#else
        krow[r] = uint(min(nrow_b[r] * 4 + (fm & 3), kL - 1));
#endif
      }
      int nvis[2];
      UNROLL for (short f = 0; f < 2; ++f) {
        nvis[f] = ncol_n[f];
      }
      if (step + 1 < nsteps) {
        fetch((step + 1) * SB);
      }

      auto s_ct = qk_op.template get_destination_cooperative_tensor<qa_t, kb_t, float>();
      UNROLL for (short i = 0; i < 16; ++i) {
        s_ct[i] = 0.0f;
      }
      UNROLL for (short jj = 0; jj < TDH / 2; ++jj) {
        const vec<T, 8> qa = *(const device vec<T, 8>*)(qp0 + 32 * jj);
        const vec<T, 8> qb = r1_ok ? *(const device vec<T, 8>*)(qp1 + 32 * jj) : vec<T, 8>(0);
        const vec<T, 8> a0 = *(const device vec<T, 8>*)(kb + (krow[0] * skl + 32 * jj));
        const vec<T, 8> a1 = *(const device vec<T, 8>*)(kb + (krow[1] * skl + 32 * jj));
        const vec<T, 8> a2 = *(const device vec<T, 8>*)(kb + (krow[2] * skl + 32 * jj));
        const vec<T, 8> a3 = *(const device vec<T, 8>*)(kb + (krow[3] * skl + 32 * jj));
        UNROLL for (short j = 0; j < 4; ++j) {
          qa_ct[j] = qa[j];
          qa_ct[4 + j] = qb[j];
          qa_ct[8 + j] = qa[4 + j];
          qa_ct[12 + j] = qb[4 + j];
          kb_ct[j] = a0[j];
          kb_ct[4 + j] = a1[j];
          kb_ct[8 + j] = a2[j];
          kb_ct[12 + j] = a3[j];
          kb_ct[16 + j] = a0[4 + j];
          kb_ct[20 + j] = a1[4 + j];
          kb_ct[24 + j] = a2[4 + j];
          kb_ct[28 + j] = a3[4 + j];
        }
        qk_op.run(qa_ct, kb_ct, s_ct);
      }
      vec<float, 8> s[2];
      UNROLL for (short i = 0; i < 8; ++i) {
        s[0][i] = s_ct[i];
        s[1][i] = s_ct[8 + i];
      }

#if MFQ_QSA_PREFILL
      if (dh == 0) {
        UNROLL for (short f = 0; f < 2; ++f) {
          UNROLL for (short i = 0; i < 8; ++i) {
            xchg[lane * 16 + f * 8 + i] = s[f][i];
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (dh == 1) {
        UNROLL for (short f = 0; f < 2; ++f) {
          UNROLL for (short i = 0; i < 8; ++i) {
            s[f][i] += xchg[lane * 16 + f * 8 + i];
            xchg[lane * 16 + f * 8 + i] = s[f][i];
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (dh == 0) {
        UNROLL for (short f = 0; f < 2; ++f) {
          UNROLL for (short i = 0; i < 8; ++i) {
            s[f][i] = xchg[lane * 16 + f * 8 + i];
          }
        }
      }
#else
      UNROLL for (short f = 0; f < 2; ++f) {
        threadgroup float* mine = xchg[dh];
        const threadgroup float* peer = xchg[1 - dh];
        UNROLL for (short i = 0; i < 8; ++i) {
          mine[lane * 8 + i] = s[f][i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        UNROLL for (short i = 0; i < 8; ++i) {
          s[f][i] += peer[lane * 8 + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
#endif

      UNROLL for (short f = 0; f < 2; ++f) {
        UNROLL for (short j = 0; j < 4; ++j) {
#if MFQ_QSA_PREFILL
          const bool ok = step < TOPK / SB || j < nvis[f];
#else
          const bool ok = j < nvis[f];
#endif
          s[f][j] = ok ? s[f][j] * scale2 : -INFINITY;
          s[f][4 + j] = ok ? s[f][4 + j] * scale2 : -INFINITY;
        }
      }

      float factor[2];
      UNROLL for (short i = 0; i < 2; ++i) {
        float m = -INFINITY;
        UNROLL for (short f = 0; f < 2; ++f) {
          m = max(m, max(max(s[f][4 * i], s[f][4 * i + 1]), max(s[f][4 * i + 2], s[f][4 * i + 3])));
        }
        m = max(m, simd_shuffle_xor(m, ushort(1)));
        m = max(m, simd_shuffle_xor(m, ushort(8)));
        const float new_max = max(max_s[i], m);
        float rs = 0.0f;
        UNROLL for (short f = 0; f < 2; ++f) {
          UNROLL for (short j = 0; j < 4; ++j) {
            s[f][4 * i + j] = fast::exp2(s[f][4 * i + j] - new_max);
            rs += s[f][4 * i + j];
          }
        }
        rs += simd_shuffle_xor(rs, ushort(1));
        rs += simd_shuffle_xor(rs, ushort(8));
        factor[i] = fast::exp2(max_s[i] - new_max);
        max_s[i] = new_max;
        sum_s[i] = sum_s[i] * factor[i] + rs;
      }

      if (simd_any(factor[0] != 1.0f || factor[1] != 1.0f)) {
        UNROLL for (short h = 0; h < 2; ++h) {
          UNROLL for (short j = 0; j < 4; ++j) {
            of0[8 * h + j] *= factor[0];
            of0[8 * h + 4 + j] *= factor[1];
            of1[8 * h + j] *= factor[0];
            of1[8 * h + 4 + j] *= factor[1];
            of2[8 * h + j] *= factor[0];
            of2[8 * h + 4 + j] *= factor[1];
            of3[8 * h + j] *= factor[0];
            of3[8 * h + 4 + j] *= factor[1];
          }
        }
      }

#if MFQ_QSA_PREFILL
      UNROLL for (short id = 0; id < TDH; id += 2) {
        const vec<T, 8> ra = *(const device vec<T, 8>*)(vb + (krow[0] * svl + 16 * id));
        const vec<T, 8> rb = *(const device vec<T, 8>*)(vb + (krow[1] * svl + 16 * id));
        const vec<T, 8> rc = *(const device vec<T, 8>*)(vb + (krow[2] * svl + 16 * id));
        const vec<T, 8> rd = *(const device vec<T, 8>*)(vb + (krow[3] * svl + 16 * id));
        UNROLL for (short j = 0; j < 4; ++j) {
          vb_ct[j] = ra[j];
          vb_ct[4 + j] = rb[j];
          vb_ct[8 + j] = ra[4 + j];
          vb_ct[12 + j] = rb[4 + j];
          vb_ct[16 + j] = rc[j];
          vb_ct[20 + j] = rd[j];
          vb_ct[24 + j] = rc[4 + j];
          vb_ct[28 + j] = rd[4 + j];
        }
        UNROLL for (short i = 0; i < 8; ++i) {
          pa_ct[i] = half(s[0][i]);
          pa_ct[8 + i] = half(s[1][i]);
        }
        if (id == 0) {
          pv_op.run(pa_ct, vb_ct, of0);
        } else if (id == 2) {
          pv_op.run(pa_ct, vb_ct, of1);
        } else if (id == 4) {
          pv_op.run(pa_ct, vb_ct, of2);
        } else {
          pv_op.run(pa_ct, vb_ct, of3);
        }
      }
#else
      UNROLL for (short f = 0; f < 2; ++f) {
        vec<half, 8> pc[2];
        {
          vec<float, 8> rest = s[f];
          UNROLL for (short tt = 0; tt < 2; ++tt) {
            UNROLL for (short i = 0; i < 8; ++i) {
              const half piece = half(rest[i]);
              pc[tt][i] = piece;
              rest[i] -= float(piece);
            }
          }
        }
        UNROLL for (short id = 0; id < TDH; id += 2) {
          const vec<T, 8> ra = *(const device vec<T, 8>*)(vb + (krow[2 * f] * svl + 16 * id));
          const vec<T, 8> rb = *(const device vec<T, 8>*)(vb + (krow[2 * f + 1] * svl + 16 * id));
          UNROLL for (short j = 0; j < 4; ++j) {
            vb_ct[j] = ra[j];
            vb_ct[4 + j] = rb[j];
            vb_ct[8 + j] = ra[4 + j];
            vb_ct[12 + j] = rb[4 + j];
            vb_ct[16 + j] = ra[j];
            vb_ct[20 + j] = rb[j];
            vb_ct[24 + j] = ra[4 + j];
            vb_ct[28 + j] = rb[4 + j];
          }
          UNROLL for (short i = 0; i < 8; ++i) {
            pa_ct[i] = pc[0][i];
            pa_ct[8 + i] = pc[1][i];
          }
          if (id == 0) {
            pv_op.run(pa_ct, vb_ct, of0);
          } else if (id == 2) {
            pv_op.run(pa_ct, vb_ct, of1);
          } else if (id == 4) {
            pv_op.run(pa_ct, vb_ct, of2);
          } else {
            pv_op.run(pa_ct, vb_ct, of3);
          }
        }
      }
#endif
    }

    UNROLL for (short i = 0; i < 2; ++i) {
      if (i == 1 && !r1_ok) {
        continue;
      }
      const float rr = sum_s[i] > 0.0f ? 1.0f / sum_s[i] : 0.0f;
      device T* o = (device T*)out + ((size_t(batch) * size_t(params[2]) + size_t(tq)) * (2 * GQA) + (i == 0 ? h0 : h1)) * D + dh * 128 + fcol;
      UNROLL for (short jj = 0; jj < TDH / 2; ++jj) {
        vec<T, 8> w;
        UNROLL for (short j = 0; j < 4; ++j) {
          float e0, e1;
          if (jj == 0) {
            e0 = of0[4 * i + j];
            e1 = of0[8 + 4 * i + j];
          } else if (jj == 1) {
            e0 = of1[4 * i + j];
            e1 = of1[8 + 4 * i + j];
          } else if (jj == 2) {
            e0 = of2[4 * i + j];
            e1 = of2[8 + 4 * i + j];
          } else {
            e0 = of3[4 * i + j];
            e1 = of3[8 + 4 * i + j];
          }
          w[j] = T(e0 * rr);
          w[4 + j] = T(e1 * rr);
        }
        *(device vec<T, 8>*)(o + 32 * jj) = w;
      }
    }

)MFQ_METAL";
