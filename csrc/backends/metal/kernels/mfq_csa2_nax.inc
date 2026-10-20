constexpr char kCsa2NaxRead[] = R"CSA2(
inline int csa2_nax_row(uint query, uint slot, constant int* meta, const device int* ci) {
    if (slot >= uint(meta[1] + meta[2])) return -1;
    if (slot < uint(meta[1])) return csa2_window_index(query, slot, meta);
    const int row = ci[query * meta[2] + slot - meta[1]];
    return row >= 0 && row < meta[4] ? meta[3] + row : -1;
}
inline vec<half, 8> csa2_nax_read(const device uchar* window, const device uchar* pooled,
    int row, int first, constant int* meta) {
    if (row < 0) return vec<half, 8>(0);
    const bool compressed = row >= meta[3];
#ifdef CSA2_UNPACKED
    const device half* data = reinterpret_cast<const device half*>(compressed ? pooled : window);
    return *(const device vec<half, 8>*)(data + size_t(compressed ? row - meta[3] : row) * 512 + first);
#else
    const device uchar* data = compressed ? pooled + size_t(row - meta[3]) * 288 : window + size_t(row) * 528;
    const float scale = compressed ? v41_fp8(data[256 + first / 16])
        : as_type<float>(uint(data[512 + first / 32]) << 23);
    vec<half, 8> value;
    UNROLL for (short i = 0; i < 8; ++i) {
        const int d = first + i;
        const float decoded = compressed ? v41_fp4((data[d / 2] >> ((d & 1) * 4)) & 15) : v41_fp8(data[d]);
        value[i] = half(decoded * scale);
    }
    return value;
#endif
}
)CSA2";
constexpr char kCsa2Nax[] = R"CSA2(

    constexpr int D = 512;
    using T = half;
    threadgroup float xchg[4][8 * 32];
    const int H = meta[0], splits = meta[5];
    const int tq = int(threadgroup_position_in_grid.x) / splits;
    const int partition = int(threadgroup_position_in_grid.x) % splits;
    const int first_head = int(threadgroup_position_in_grid.y) * 16;
    const ushort dh = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
    const short qid = lane >> 2;
    const short fm = (qid & 4) | ((lane >> 1) & 3);
    const short fn = ((qid & 2) | (lane & 1)) * 4;
    const short fcol = 2 * fn;
    const int h0 = first_head + fm, h1 = h0 + 8;
    constexpr bool r1_ok = true;
    const float scale2 = scalep[0] * M_LOG2E_F;
    const int nsteps = (meta[1] + meta[2] + 31) / 32;
    const device T* qp0 = q + (size_t(tq) * H + h0) * D + dh * 128 + fcol;
    const device T* qp1 = q + (size_t(tq) * H + h1) * D + dh * 128 + fcol;
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

    int nrow_b[4];
    bool visible[2][4];
    auto fetch = [&](int step) {
      UNROLL for (short r = 0; r < 4; ++r)
        nrow_b[r] = csa2_nax_row(tq, step * 32 + 16 * (r >> 1) + fm + 8 * (r & 1), meta, ci);
      UNROLL for (short f = 0; f < 2; ++f)
        UNROLL for (short j = 0; j < 4; ++j)
          visible[f][j] = csa2_nax_row(tq, step * 32 + 16 * f + fn + j, meta, ci) >= 0;
    };
    fetch(partition);

    for (int step = partition; step < nsteps; step += splits) {

      bool valid[2][4];
      UNROLL for (short f = 0; f < 2; ++f)
        UNROLL for (short j = 0; j < 4; ++j) valid[f][j] = visible[f][j];
      int rows[4];
      UNROLL for (short r = 0; r < 4; ++r) rows[r] = nrow_b[r];
      if (step + splits < nsteps) fetch(step + splits);
      auto s_ct = qk_op.template get_destination_cooperative_tensor<qa_t, kb_t, float>();
      UNROLL for (short i = 0; i < 16; ++i) {
        s_ct[i] = 0.0f;
      }
      UNROLL for (short jj = 0; jj < 8 / 2; ++jj) {
        const vec<T, 8> qa = *(const device vec<T, 8>*)(qp0 + 32 * jj);
        const vec<T, 8> qb = r1_ok ? *(const device vec<T, 8>*)(qp1 + 32 * jj) : vec<T, 8>(0);
        const vec<T, 8> a0 = csa2_nax_read(window, pooled, rows[0], dh * 128 + fcol + 32 * jj, meta);
        const vec<T, 8> a1 = csa2_nax_read(window, pooled, rows[1], dh * 128 + fcol + 32 * jj, meta);
        const vec<T, 8> a2 = csa2_nax_read(window, pooled, rows[2], dh * 128 + fcol + 32 * jj, meta);
        const vec<T, 8> a3 = csa2_nax_read(window, pooled, rows[3], dh * 128 + fcol + 32 * jj, meta);
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

      UNROLL for (short f = 0; f < 2; ++f) {
        threadgroup float* mine = xchg[dh];
        UNROLL for (short i = 0; i < 8; ++i) {
          mine[lane * 8 + i] = s[f][i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        UNROLL for (short i = 0; i < 8; ++i) {
          UNROLL for (short peer = 0; peer < 4; ++peer)
            if (peer != dh) s[f][i] += xchg[peer][lane * 8 + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }

      UNROLL for (short f = 0; f < 2; ++f) {
        UNROLL for (short j = 0; j < 4; ++j) {
          const bool ok = valid[f][j];
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
        UNROLL for (short id = 0; id < 8; id += 2) {
          const vec<T, 8> ra = csa2_nax_read(window, pooled, rows[2 * f], dh * 128 + fcol + 16 * id, meta);
          const vec<T, 8> rb = csa2_nax_read(window, pooled, rows[2 * f + 1], dh * 128 + fcol + 16 * id, meta);
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
    }

    UNROLL for (short i = 0; i < 2; ++i) {
      if (i == 1 && !r1_ok) {
        continue;
      }
      const size_t row = size_t(tq) * H + (i == 0 ? h0 : h1);
      using Output = float;
      const bool final = splits == 1 && meta[7] > 8;
      float rr = 1.0f;
      if (final) {
        const float maximum = max(max_s[i] * M_LN2_F, sink[i == 0 ? h0 : h1]);
        const float correction = exp(max_s[i] * M_LN2_F - maximum);
        rr = correction / (sum_s[i] * correction + exp(sink[i == 0 ? h0 : h1] - maximum));
      }
      if (!final && dh == 0 && fcol == 0) {
        partial[(row * splits + partition) * (D + 2) + D] = max_s[i] * M_LN2_F;
        partial[(row * splits + partition) * (D + 2) + D + 1] = sum_s[i];
      }
      device Output* o = partial;
      if (!final) o += (row * splits + partition) * (D + 2) + dh * 128 + fcol;
      UNROLL for (short jj = 0; jj < 8 / 2; ++jj) {
        vec<Output, 8> w;
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
          w[j] = Output(e0 * rr);
          w[4 + j] = Output(e1 * rr);
        }
        if (final) {
          *(device vec<half, 8>*)(final_output + row * D + dh * 128 + fcol + 32 * jj) = vec<half, 8>(w);
        } else {
          *(device vec<Output, 8>*)(o + 32 * jj) = w;
        }
      }
    }


)CSA2";
