// Keep each value-column recurrence in registers. Several columns share q/k,
// gate, and token-loop work without changing FP32 reduction order.
template<int D, int COLUMNS, bool TRANSPOSED_STATE, bool TILED_HEADS, bool PIPELINED=false>
__global__ void gdn_prefill_columns_kernel(
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    const float* __restrict__ g, const float* __restrict__ beta,
    const float* __restrict__ s_in, float* __restrict__ out, float* __restrict__ s_out,
    int Hq, int Hv, int T) {
    constexpr int ROWS = D / 32, WARPS = 4;
    const int bh = blockIdx.x, hv = bh % Hv, b = bh / Hv;
    const int hq = TILED_HEADS ? hv % Hq : hv / (Hv / Hq);
    const int lane = threadIdx.x;
    const int first_col = (blockIdx.y * WARPS + threadIdx.y) * COLUMNS;
    const size_t state_off = size_t(bh) * D * D;
    const size_t q_seq_off = size_t(b * Hq + hq) * T;
    const size_t v_seq_off = size_t(bh) * T;
    float state[COLUMNS][ROWS];
    #pragma unroll
    for (int c = 0; c < COLUMNS; ++c) {
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const int row = r * 32 + lane, col = first_col + c;
            const size_t index = TRANSPOSED_STATE ? size_t(col) * D + row : size_t(row) * D + col;
            state[c][r] = s_in ? s_in[state_off + index] : 0.0f;
        }
    }
    const float scale = 1.0f / sqrtf(float(D));
    float cached_key[ROWS],cached_query[ROWS];
    if constexpr(PIPELINED) {
        #pragma unroll
        for(int r=0;r<ROWS;++r) {
            const int row=r*32+lane;
            cached_key[r]=k[q_seq_off*D+row];
            cached_query[r]=q[q_seq_off*D+row];
        }
    }
    for (int t = 0; t < T; ++t) {
        float key[ROWS], query[ROWS], kv[COLUMNS] = {}, attended[COLUMNS] = {};
        if constexpr(PIPELINED) {
            #pragma unroll
            for(int r=0;r<ROWS;++r) {
                key[r]=cached_key[r];query[r]=cached_query[r];
            }
            // Current q/k remain live while the next token's loads overlap
            // the unchanged FP32 recurrence and warp reductions.
            if(t+1<T) {
                #pragma unroll
                for(int r=0;r<ROWS;++r) {
                    const int row=r*32+lane;
                    cached_key[r]=k[(q_seq_off+t+1)*D+row];
                    cached_query[r]=q[(q_seq_off+t+1)*D+row];
                }
            }
            #pragma unroll
            for(int r=0;r<ROWS;++r) {
                #pragma unroll
                for(int c=0;c<COLUMNS;++c)kv[c]+=state[c][r]*key[r];
            }
        } else {
            #pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                const int row = r * 32 + lane;
                key[r] = k[(q_seq_off + t) * D + row];
                query[r] = q[(q_seq_off + t) * D + row];
                #pragma unroll
                for (int c = 0; c < COLUMNS; ++c) kv[c] += state[c][r] * key[r];
            }
        }
        const float decay = expf(g[v_seq_off + t]);
        const float bt = beta[v_seq_off + t];
        float delta[COLUMNS];
        #pragma unroll
        for (int c = 0; c < COLUMNS; ++c) {
            kv[c] = warp_sum(kv[c]);
            delta[c] = (v[(v_seq_off + t) * D + first_col + c] - decay * kv[c]) * bt;
        }
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            #pragma unroll
            for (int c = 0; c < COLUMNS; ++c) {
                state[c][r] = decay * state[c][r] + key[r] * delta[c];
                attended[c] += state[c][r] * query[r];
            }
        }
        #pragma unroll
        for (int c = 0; c < COLUMNS; ++c) {
            attended[c] = warp_sum(attended[c]);
            if (lane == 0) out[(v_seq_off + t) * D + first_col + c] = attended[c] * scale;
        }
    }
    #pragma unroll
    for (int c = 0; c < COLUMNS; ++c) {
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const int row = r * 32 + lane, col = first_col + c;
            const size_t index = TRANSPOSED_STATE ? size_t(col) * D + row : size_t(row) * D + col;
            s_out[state_off + index] = state[c][r];
        }
    }
}

template<bool TRANSPOSED_STATE, bool TILED_HEADS>
void launch_gdn_prefill_columns(int columns, int B, int Hq, int Hv, int T,
    const float* q, const float* k, const float* v, const float* g, const float* beta,
    const float* state, float* out, float* next_state, cudaStream_t stream) {
    const dim3 block(32, 4);
    const auto* pipeline_option=std::getenv("MFQ_GDN_PREFILL_PIPELINED");
    // The verified single-sequence 48/64-head grids do not lose resident
    // blocks from the extra q/k registers. Other shapes retain the old kernel.
    const bool pipelined=pipeline_option?std::atoi(pipeline_option)!=0:
        B==1 && (Hv==48 || Hv==64);
    if(columns==4 && pipelined) {
        gdn_prefill_columns_kernel<128,4,TRANSPOSED_STATE,TILED_HEADS,true>
            <<<dim3(B*Hv,128/(4*4)),block,0,stream>>>
            (q,k,v,g,beta,state,out,next_state,Hq,Hv,T);
    } else if (columns == 4) {
        gdn_prefill_columns_kernel<128, 4, TRANSPOSED_STATE, TILED_HEADS>
            <<<dim3(B * Hv, 128 / (4 * 4)), block, 0, stream>>>
            (q, k, v, g, beta, state, out, next_state, Hq, Hv, T);
    } else {
        gdn_prefill_columns_kernel<128, 2, TRANSPOSED_STATE, TILED_HEADS>
            <<<dim3(B * Hv, 128 / (4 * 2)), block, 0, stream>>>
            (q, k, v, g, beta, state, out, next_state, Hq, Hv, T);
    }
}
