// Tile the existing SoA inputs directly. State and all FP32 reductions retain
// the column kernel's register layout/order; no packed QKV tensor is allocated.
__device__ __forceinline__ void gdn_prefill_cp16(float* dst,const float* src) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::
        "r"(unsigned(__cvta_generic_to_shared(dst))),"l"(src):"memory");
#else
    *reinterpret_cast<float4*>(dst)=*reinterpret_cast<const float4*>(src);
#endif
}
__device__ __forceinline__ void gdn_prefill_cp4(float* dst,const float* src) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::
        "r"(unsigned(__cvta_generic_to_shared(dst))),"l"(src):"memory");
#else
    *dst=*src;
#endif
}
__device__ __forceinline__ void gdn_prefill_cp_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;" ::: "memory");
#endif
}
__device__ __forceinline__ void gdn_prefill_cp_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 1;" ::: "memory");
#endif
}

template<bool TRANSPOSED_STATE,bool TILED_HEADS,bool PREPARED_DECAY>
__global__ void gdn_prefill_staged_kernel(
    const float* __restrict__ q,const float* __restrict__ k,const float* __restrict__ v,
    const float* __restrict__ g,const float* __restrict__ beta,
    const float* __restrict__ s_in,float* __restrict__ out,float* __restrict__ s_out,
    int Hq,int Hv,int T) {
    constexpr int D=128,COLUMNS=4,ROWS=4,WARPS=4,TB=8,VC=WARPS*COLUMNS,NT=32*WARPS;
    __shared__ __align__(16) float sq[2][TB][D],sk[2][TB][D],sv[2][TB][VC];
    __shared__ float sg[2][TB],sb[2][TB];
    const int bh=blockIdx.x,hv=bh%Hv,b=bh/Hv;
    const int hq=TILED_HEADS?hv%Hq:hv/(Hv/Hq);
    const int lane=threadIdx.x,warp=threadIdx.y,tid=warp*32+lane;
    const int block_col=blockIdx.y*VC,first_col=block_col+warp*COLUMNS;
    const size_t state_off=size_t(bh)*D*D,q_seq_off=size_t(b*Hq+hq)*T,v_seq_off=size_t(bh)*T;
    float state[COLUMNS][ROWS];
    #pragma unroll
    for(int c=0;c<COLUMNS;++c) {
        #pragma unroll
        for(int r=0;r<ROWS;++r) {
            const int row=r*32+lane,col=first_col+c;
            const size_t index=TRANSPOSED_STATE?size_t(col)*D+row:size_t(row)*D+col;
            state[c][r]=s_in?s_in[state_off+index]:0.f;
        }
    }
    const auto stage=[&](int tile) {
        const int bb=tile&1,t0=tile*TB;
        for(int p=tid;p<TB*2*(D/4);p+=NT) {
            const int i=p/(2*(D/4)),w=p%(2*(D/4)),is_key=w/(D/4),row=(w%(D/4))*4;
            if(t0+i<T)gdn_prefill_cp16(is_key?&sk[bb][i][row]:&sq[bb][i][row],
                (is_key?k:q)+(q_seq_off+t0+i)*D+row);
        }
        for(int p=tid;p<TB*(VC/4);p+=NT) {
            const int i=p/(VC/4),col=(p%(VC/4))*4;
            if(t0+i<T)gdn_prefill_cp16(&sv[bb][i][col],v+(v_seq_off+t0+i)*D+block_col+col);
        }
        if(tid<2*TB) {
            const int i=tid%TB;
            if(t0+i<T)gdn_prefill_cp4(tid<TB?&sg[bb][i]:&sb[bb][i],
                (tid<TB?g:beta)+v_seq_off+t0+i);
        }
    };
    const int tiles=(T+TB-1)/TB;
    if(tiles)stage(0);
    gdn_prefill_cp_commit();
    const float scale=1.f/sqrtf(float(D));
    for(int tile=0;tile<tiles;++tile) {
        if(tile+1<tiles)stage(tile+1);
        // An empty final group also keeps wait_group 1 valid for the last tile.
        gdn_prefill_cp_commit();gdn_prefill_cp_wait();__syncthreads();
        const int bb=tile&1,count=min(TB,T-tile*TB);
        for(int i=0;i<count;++i) {
            float key[ROWS],query[ROWS],kv[COLUMNS]={},attended[COLUMNS]={};
            #pragma unroll
            for(int r=0;r<ROWS;++r) {
                const int row=r*32+lane;
                key[r]=sk[bb][i][row];query[r]=sq[bb][i][row];
                #pragma unroll
                for(int c=0;c<COLUMNS;++c)kv[c]+=state[c][r]*key[r];
            }
            const float decay=PREPARED_DECAY?sg[bb][i]:expf(sg[bb][i]),bt=sb[bb][i];
            float delta[COLUMNS];
            #pragma unroll
            for(int c=0;c<COLUMNS;++c) {
                kv[c]=warp_sum(kv[c]);
                delta[c]=(sv[bb][i][warp*COLUMNS+c]-decay*kv[c])*bt;
            }
            #pragma unroll
            for(int r=0;r<ROWS;++r) {
                #pragma unroll
                for(int c=0;c<COLUMNS;++c) {
                    state[c][r]=decay*state[c][r]+key[r]*delta[c];
                    attended[c]+=state[c][r]*query[r];
                }
            }
            #pragma unroll
            for(int c=0;c<COLUMNS;++c) {
                attended[c]=warp_sum(attended[c]);
                if(lane==0)out[(v_seq_off+tile*TB+i)*D+first_col+c]=attended[c]*scale;
            }
        }
        // All warps finish reading this buffer before the next reuse.
        __syncthreads();
    }
    #pragma unroll
    for(int c=0;c<COLUMNS;++c) {
        #pragma unroll
        for(int r=0;r<ROWS;++r) {
            const int row=r*32+lane,col=first_col+c;
            const size_t index=TRANSPOSED_STATE?size_t(col)*D+row:size_t(row)*D+col;
            s_out[state_off+index]=state[c][r];
        }
    }
}
