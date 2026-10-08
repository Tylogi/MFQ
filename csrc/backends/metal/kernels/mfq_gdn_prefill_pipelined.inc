// SPDX-License-Identifier: Apache-2.0
// Adapted from jundot/omlx gated_delta_pipelined, with MFQ layouts and scaling.

std::string gdn_pipeline_load_vec(
    const char* source, const char* target, const std::string& token) {
    std::string code;
    for (int i = 0; i < 4; ++i) {
        code += std::string(target) + "[" + std::to_string(i) +
            "] = *(const threadgroup float4*)(&" + source + "[" + token +
            "][4 * (seg + 8 * " + std::to_string(i) + ")]);\n";
    }
    return code;
}

std::string gdn_pipeline_reduce_one(const std::string& token) {
    return "{\nfloat yq = pq_prev;\n"
        "yq += simd_shuffle_down(yq, 4);\n"
        "yq += simd_shuffle_down(yq, 2);\n"
        "yq += simd_shuffle_down(yq, 1);\n"
        "if (seg == 0) y_s[" + token + "][rg] = yq * output_scale;\n}\n";
}

std::string gdn_pipeline_reduce_four(int token) {
    return "{\nconst bool qb2 = (seg & 4) != 0, qb1 = (seg & 2) != 0;\n"
        "float qa = qb2 ? pq4[2] : pq4[0];\n"
        "float qb = qb2 ? pq4[3] : pq4[1];\n"
        "qa += simd_shuffle_xor(qb2 ? pq4[0] : pq4[2], 4);\n"
        "qb += simd_shuffle_xor(qb2 ? pq4[1] : pq4[3], 4);\n"
        "float qx = qb1 ? qb : qa;\n"
        "qx += simd_shuffle_xor(qb1 ? qa : qb, 2);\n"
        "qx += simd_shuffle_xor(qx, 1);\n"
        "if ((seg & 1) == 0) y_s[" + std::to_string(token) +
        " + (qb2 ? 2 : 0) + (qb1 ? 1 : 0)][rg] = qx * output_scale;\n}\n";
}

std::string gdn_pipeline_step(int token) {
    const bool full = token >= 0;
    const std::string t = full ? std::to_string(token) : "token";
    std::string code = "{\nfloat4 kc[4];\n"
        "for (int i = 0; i < 4; ++i) kc[i] = k_next[i];\n"
        "const float gc = g_next, bc = b_next, vc = v_next;\n";
    if (!full || token < 11) {
        if (!full) code += "if (token + 1 < tt) {\n";
        code += gdn_pipeline_load_vec("k_s", "k_next", "(" + t + ") + 1");
        code += "g_next = g_s[(" + t + ") + 1];\n"
            "b_next = b_s[(" + t + ") + 1];\n"
            "v_next = v_s[(" + t + ") + 1][rg];\n";
        if (!full) code += "}\n";
    }
    code += "float2 a2 = 0.0f;\n"
        "for (int i = 0; i < 4; ++i) {\n"
        "st[i] = st[i] * gc;\n"
        "a2 = fma(st[i].xy, kc[i].xy, a2);\n"
        "a2 = fma(st[i].zw, kc[i].zw, a2);\n}\n";
    if (!full) {
        code += "if (token > 0) " + gdn_pipeline_reduce_one("token - 1");
    } else if (token > 0 && token % 4 == 0) {
        code += gdn_pipeline_reduce_four(token - 4);
    }
    code += "float p = a2.x + a2.y;\n"
        "p += simd_shuffle_xor(p, 4);\n"
        "p += simd_shuffle_xor(p, 2);\n"
        "p += simd_shuffle_xor(p, 1);\nfloat4 qc[4];\n";
    code += gdn_pipeline_load_vec("q_s", "qc", t);
    code += "const float delta = (vc - p) * bc;\nfloat2 o2 = 0.0f;\n"
        "for (int i = 0; i < 4; ++i) {\n"
        "st[i] = fma(kc[i], float4(delta), st[i]);\n"
        "o2 = fma(st[i].xy, qc[i].xy, o2);\n"
        "o2 = fma(st[i].zw, qc[i].zw, o2);\n}\n";
    code += full ? "pq4[" + std::to_string(token % 4) + "] = o2.x + o2.y;\n"
        : "pq_prev = o2.x + o2.y;\n";
    return code + "}\n";
}

std::string gdn_pipeline_source() {
    std::string code = R"METAL(
    const int TOKENS = v_shape[2];
    constexpr int TB = 12, DB = 16, NT = 128;
    const int tid = thread_index_in_threadgroup;
    const int hv = threadgroup_position_in_grid.y;
    const int b = threadgroup_position_in_grid.z;
    const int hk = TILED_HEADS ? hv % HQ : hv / (HV / HQ);
    const int dv0 = threadgroup_position_in_grid.x * DB;
    const int rg = tid / 8, seg = tid % 8;
    const float output_scale = rsqrt(float(D));
    threadgroup float k_s[TB][D], q_s[TB][D];
    threadgroup float v_s[TB][DB], y_s[TB][DB];
    threadgroup float g_s[TB], b_s[TB];
    const device float* k_base = k + ((size_t)b * HQ + hk) * TOKENS * D;
    const device float* q_base = q + ((size_t)b * HQ + hk) * TOKENS * D;
    const device float* v_base = v + ((size_t)b * HV + hv) * TOKENS * D + dv0;
    device float* y_base = out + ((size_t)b * HV + hv) * TOKENS * D + dv0;
    float4 st[4];
    {
        const device float4* s = (const device float4*)(
            state_in + (((size_t)b * HV + hv) * D + dv0 + rg) * D);
        for (int i = 0; i < 4; ++i) st[i] = s[seg + 8 * i];
    }
    constexpr int NKQ = (TB * D / 4 + NT - 1) / NT;
    constexpr int NV = (TB * DB / 4 + NT - 1) / NT;
    float4 pk[NKQ], pq[NKQ], pv[NV];
    float pg = 0.0f, pb = 0.0f;
#define GDN_PIPE_FETCH(T0N) { \
        const int ttn = min(TB, TOKENS - (T0N)); \
        for (int j = 0; j < NKQ; ++j) { \
            const int p = tid + j * NT; \
            if (p < ttn * (D / 4)) { \
                const int r = p / (D / 4), c4 = p % (D / 4); \
                const size_t off = (size_t)((T0N) + r) * D + 4 * c4; \
                pk[j] = *(const device float4*)(k_base + off); \
                pq[j] = *(const device float4*)(q_base + off); \
            } \
        } \
        for (int j = 0; j < NV; ++j) { \
            const int p = tid + j * NT; \
            if (p < ttn * (DB / 4)) { \
                const int r = p / (DB / 4), c4 = p % (DB / 4); \
                pv[j] = *(const device float4*)(v_base + \
                    (size_t)((T0N) + r) * D + 4 * c4); \
            } \
        } \
        if (tid < ttn) { \
            pg = exp(g[((size_t)b * HV + hv) * TOKENS + (T0N) + tid]); \
            pb = beta[((size_t)b * HV + hv) * TOKENS + (T0N) + tid]; \
        } \
    }
    GDN_PIPE_FETCH(0)
    float4 k_next[4];
    float g_next = 0.0f, b_next = 0.0f, v_next = 0.0f, pq_prev = 0.0f;
    float pq4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int t0 = 0; t0 < TOKENS; t0 += TB) {
        const int tt = min(TB, TOKENS - t0);
        for (int j = 0; j < NKQ; ++j) {
            const int p = tid + j * NT;
            if (p < tt * (D / 4)) {
                const int r = p / (D / 4), c4 = p % (D / 4);
                *(threadgroup float4*)(&k_s[r][4 * c4]) = pk[j];
                *(threadgroup float4*)(&q_s[r][4 * c4]) = pq[j];
            }
        }
        for (int j = 0; j < NV; ++j) {
            const int p = tid + j * NT;
            if (p < tt * (DB / 4)) {
                const int r = p / (DB / 4), c4 = p % (DB / 4);
                *(threadgroup float4*)(&v_s[r][4 * c4]) = pv[j];
            }
        }
        if (tid < tt) { g_s[tid] = pg; b_s[tid] = pb; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t0 + TB < TOKENS) GDN_PIPE_FETCH(t0 + TB)
)METAL";
    code += gdn_pipeline_load_vec("k_s", "k_next", "0");
    code += "g_next = g_s[0]; b_next = b_s[0]; v_next = v_s[0][rg];\n"
        "if (tt == TB) {\n";
    for (int token = 0; token < 12; ++token) code += gdn_pipeline_step(token);
    code += gdn_pipeline_reduce_four(8);
    code += "} else {\nfor (int token = 0; token < tt; ++token) ";
    code += gdn_pipeline_step(-1);
    code += gdn_pipeline_reduce_one("tt - 1");
    code += R"METAL(
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int p = tid; p < tt * (DB / 4); p += NT) {
            const int r = p / (DB / 4), c4 = p % (DB / 4);
            *(device float4*)(y_base + (size_t)(t0 + r) * D + 4 * c4) =
                *(threadgroup float4*)(&y_s[r][4 * c4]);
        }
    }
#undef GDN_PIPE_FETCH
    {
        device float4* s = (device float4*)(
            state_out + (((size_t)b * HV + hv) * D + dv0 + rg) * D);
        for (int i = 0; i < 4; ++i) s[seg + 8 * i] = st[i];
    }
)METAL";
    return code;
}
