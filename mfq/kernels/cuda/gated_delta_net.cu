// Gated DeltaNet CUDA kernel (ggml gated_delta_net.cu).
//
// grid = B*H (one block per (batch, head)); state S[D,D] in shared memory, T loop in-kernel.
// Recurrence (matches ops.cpp / Python reference):
//     S <- decay*S ; delta = (v - S^T k) * beta ; S <- S + k (x) delta ; o = scale * (S^T q)
// scale = 1/sqrt(D). Scalar gate and per-dim (KDA) gate; D in {32,64,128} (D=128 needs the
// 64KB shared-mem opt-in via cudaFuncSetAttribute).

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda/atomic>
#include <cmath>
#include <limits>
#include "mfq_tensor_backend.h"
#include <vector>
#include <cstdlib>

#include "reduce.cuh"

template <int D>
__global__ void gdn_kernel(
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    const float* __restrict__ g, const float* __restrict__ beta,
    const float* __restrict__ s_in,
    float* __restrict__ out, float* __restrict__ s_out,
    int H, int T, int kda)
{
    int bh = blockIdx.x;
    int b = bh / H, h = bh % H;
    int j = threadIdx.x;
    if (j >= D) return;

    extern __shared__ float S[];   // S[i*D + j] = state[i][j]

    if (s_in) {
        const float* src = s_in + ((size_t)(b * H + h)) * D * D;
        #pragma unroll 4
        for (int i = 0; i < D; ++i) S[i * D + j] = src[i * D + j];
    } else {
        #pragma unroll 4
        for (int i = 0; i < D; ++i) S[i * D + j] = 0.0f;
    }

    const float scale = 1.0f / sqrtf((float)D);
    const size_t bhoff = (size_t)(b * H + h) * T;
    const float* qb = q + bhoff * D;
    const float* kb = k + bhoff * D;
    const float* vb = v + bhoff * D;
    const float* gb = kda ? g + bhoff * D : g + bhoff;
    const float* bb = beta + bhoff;
    float* ob = out + bhoff * D;

    for (int t = 0; t < T; ++t) {
        const float* qt = qb + (size_t)t * D;
        const float* kt = kb + (size_t)t * D;
        const float* vt = vb + (size_t)t * D;
        float bt = bb[t];

        // gated decay
        if (kda) {
            #pragma unroll 4
            for (int i = 0; i < D; ++i) S[i * D + j] *= expf(gb[t * D + i]);
        } else {
            float dec = expf(gb[t]);
            #pragma unroll 4
            for (int i = 0; i < D; ++i) S[i * D + j] *= dec;
        }
        __syncthreads();

        // delta[j] = (v[j] - sum_i S[i][j]*k[i]) * beta
        float stk = 0.f;
        #pragma unroll 4
        for (int i = 0; i < D; ++i) stk += S[i * D + j] * kt[i];
        float delta = (vt[j] - stk) * bt;

        // S[i][j] += k[i] * delta[j]
        #pragma unroll 4
        for (int i = 0; i < D; ++i) S[i * D + j] += kt[i] * delta;
        __syncthreads();

        // o[j] = scale * sum_i S[i][j]*q[i]
        float o = 0.f;
        #pragma unroll 4
        for (int i = 0; i < D; ++i) o += S[i * D + j] * qt[i];
        ob[(size_t)t * D + j] = o * scale;
    }

    float* dst = s_out + ((size_t)(b * H + h)) * D * D;
    #pragma unroll 4
    for (int i = 0; i < D; ++i) dst[i * D + j] = S[i * D + j];
}

template <int D, int BD>
__global__ void gdn_column_kernel(
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    const float* __restrict__ g, const float* __restrict__ beta,
    const float* __restrict__ s_in,
    float* __restrict__ out, float* __restrict__ s_out,
    int H, int T, int kda)
{
    constexpr int NW = BD / 32;
    int bid = blockIdx.x;
    int j = bid % D;
    int bh = bid / D;
    int b = bh / H;
    int h = bh % H;
    int tid = threadIdx.x;
    const float scale = 1.0f / sqrtf((float)D);
    const size_t state_off = ((size_t)(b * H + h)) * D * D;
    const size_t bhoff = (size_t)(b * H + h) * T;
    const float* qb = q + bhoff * D;
    const float* kb = k + bhoff * D;
    const float* vb = v + bhoff * D;
    const float* gb = kda ? g + bhoff * D : g + bhoff;
    const float* bb = beta + bhoff;
    float* ob = out + bhoff * D;
    float* so = s_out + state_off;

    for (int i = tid; i < D; i += BD) {
        so[i * D + j] = s_in ? s_in[state_off + i * D + j] : 0.0f;
    }
    __syncthreads();

    for (int t = 0; t < T; ++t) {
        const float* qt = qb + (size_t)t * D;
        const float* kt = kb + (size_t)t * D;
        const float* vt = vb + (size_t)t * D;
        float bt = bb[t];

        float local = 0.0f;
        for (int i = tid; i < D; i += BD) {
            float sij = so[i * D + j];
            float dec = kda ? expf(gb[(size_t)t * D + i]) : expf(gb[t]);
            sij *= dec;
            so[i * D + j] = sij;
            local += sij * kt[i];
        }
        float stk = block_sum<NW>(local);
        float delta = (vt[j] - stk) * bt;
        __syncthreads();

        for (int i = tid; i < D; i += BD) {
            so[i * D + j] += kt[i] * delta;
        }
        __syncthreads();

        local = 0.0f;
        for (int i = tid; i < D; i += BD) {
            local += so[i * D + j] * qt[i];
        }
        float o = block_sum<NW>(local);
        if (tid == 0) {
            ob[(size_t)t * D + j] = o * scale;
        }
        __syncthreads();
    }
}

template <int D, bool KDA, int WARPS, bool TRANSPOSED_STATE=false, bool TILED_HEADS=false>
__global__ void gdn_warp_column_kernel(
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    const float* __restrict__ g, const float* __restrict__ beta,
    const float* __restrict__ s_in,
    float* __restrict__ out, float* __restrict__ s_out,
    int Hq, int Hv, int T)
{
    constexpr int WARP = 32;
    constexpr int ROWS = (D + WARP - 1) / WARP;
    int bh = blockIdx.x;
    int b = bh / Hv;
    int hv = bh % Hv;
    int hq = TILED_HEADS ? hv % Hq : hv / (Hv / Hq);
    int lane = threadIdx.x;
    int warp = threadIdx.y;
    int col = blockIdx.y * WARPS + warp;
    if (col >= D) {
        return;
    }

    const size_t state_off = ((size_t)(b * Hv + hv)) * D * D;
    const size_t q_seq_off = ((size_t)(b * Hq + hq)) * T;
    const size_t v_seq_off = ((size_t)(b * Hv + hv)) * T;
    float s_shard[ROWS];

    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        int i = r * WARP + lane;
        size_t state_idx = TRANSPOSED_STATE ? (size_t)col * D + i : (size_t)i * D + col;
        s_shard[r] = (i < D && s_in) ? s_in[state_off + state_idx] : 0.0f;
    }

    const float scale = 1.0f / sqrtf((float)D);
    for (int t = 0; t < T; ++t) {
        const float* qt = q + (q_seq_off + t) * D;
        const float* kt = k + (q_seq_off + t) * D;
        const float* vt = v + (v_seq_off + t) * D;
        const float* gt = KDA ? g + (v_seq_off + t) * D : g + (v_seq_off + t);
        float bt = beta[v_seq_off + t];

        float k_reg[ROWS];
        float q_reg[ROWS];
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            int i = r * WARP + lane;
            k_reg[r] = (i < D) ? kt[i] : 0.0f;
            q_reg[r] = (i < D) ? qt[i] : 0.0f;
        }

        float kv = 0.0f;
        if constexpr (KDA) {
            #pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                int i = r * WARP + lane;
                float dec = (i < D) ? expf(gt[i]) : 0.0f;
                kv += dec * s_shard[r] * k_reg[r];
            }
        } else {
            #pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                kv += s_shard[r] * k_reg[r];
            }
        }
        kv = warp_sum(kv);

        float delta;
        if constexpr (KDA) {
            delta = (vt[col] - kv) * bt;
        } else {
            float dec = expf(*gt);
            delta = (vt[col] - dec * kv) * bt;
        }

        float o = 0.0f;
        if constexpr (KDA) {
            #pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                int i = r * WARP + lane;
                float dec = (i < D) ? expf(gt[i]) : 0.0f;
                s_shard[r] = dec * s_shard[r] + k_reg[r] * delta;
                o += s_shard[r] * q_reg[r];
            }
        } else {
            float dec = expf(*gt);
            #pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                s_shard[r] = dec * s_shard[r] + k_reg[r] * delta;
                o += s_shard[r] * q_reg[r];
            }
        }
        o = warp_sum(o);
        if (lane == 0) {
            out[(v_seq_off + t) * D + col] = o * scale;
        }
    }

    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        int i = r * WARP + lane;
        if (i < D) {
            size_t state_idx = TRANSPOSED_STATE ? (size_t)col * D + i : (size_t)i * D + col;
            s_out[state_off + state_idx] = s_shard[r];
        }
    }
}

#define LAUNCH_GDN(DVAL)                                                            \
    do {                                                                            \
        if (shmem > 49152)                                                          \
            cudaFuncSetAttribute((const void*)gdn_kernel<DVAL>,                     \
                cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);               \
        gdn_kernel<DVAL><<<B * H, DVAL, shmem, stream>>>(                           \
            qd, kd, vd, gd, bd, sd, od, sod, H, T, kda);                            \
    } while (0)

#define LAUNCH_GDN_COL(DVAL)                                                        \
    do {                                                                            \
        gdn_column_kernel<DVAL, 128><<<B * H * DVAL, 128, 0, stream>>>(              \
            qd, kd, vd, gd, bd, sd, od, sod, H, T, kda);                            \
    } while (0)

#define LAUNCH_GDN_WARP(DVAL, KDA_VAL, TRANSPOSED_VAL, TILED_VAL)                     \
    do {                                                                              \
        constexpr int WARPS = 4;                                                      \
        dim3 grid(B * Hv, (DVAL + WARPS - 1) / WARPS, 1);                             \
        dim3 block(32, WARPS, 1);                                                     \
        gdn_warp_column_kernel<DVAL, KDA_VAL, WARPS, TRANSPOSED_VAL, TILED_VAL><<<grid, block, 0, stream>>>( \
            qd, kd, vd, gd, bd, sd, od, sod, Hq, Hv, T);                              \
    } while (0)

#define DISPATCH_GDN_WARP(TILED_VAL)                                                   \
    do {                                                                               \
        if (transposed_state) {                                                        \
            if (kda) {                                                                 \
                if (D == 32) { LAUNCH_GDN_WARP(32, true, true, TILED_VAL); }           \
                else if (D == 64) { LAUNCH_GDN_WARP(64, true, true, TILED_VAL); }      \
                else if (D == 128) { LAUNCH_GDN_WARP(128, true, true, TILED_VAL); }    \
                else MFQ_RUNTIME_CHECK(false, "GDN CUDA: D must be in {32,64,128}, got ", D); \
            } else {                                                                   \
                if (D == 32) { LAUNCH_GDN_WARP(32, false, true, TILED_VAL); }          \
                else if (D == 64) { LAUNCH_GDN_WARP(64, false, true, TILED_VAL); }     \
                else if (D == 128) { LAUNCH_GDN_WARP(128, false, true, TILED_VAL); }   \
                else MFQ_RUNTIME_CHECK(false, "GDN CUDA: D must be in {32,64,128}, got ", D); \
            }                                                                          \
        } else if (kda) {                                                              \
            if (D == 32) { LAUNCH_GDN_WARP(32, true, false, TILED_VAL); }              \
            else if (D == 64) { LAUNCH_GDN_WARP(64, true, false, TILED_VAL); }         \
            else if (D == 128) { LAUNCH_GDN_WARP(128, true, false, TILED_VAL); }       \
            else MFQ_RUNTIME_CHECK(false, "GDN CUDA: D must be in {32,64,128}, got ", D);   \
        } else {                                                                       \
            if (D == 32) { LAUNCH_GDN_WARP(32, false, false, TILED_VAL); }             \
            else if (D == 64) { LAUNCH_GDN_WARP(64, false, false, TILED_VAL); }        \
            else if (D == 128) { LAUNCH_GDN_WARP(128, false, false, TILED_VAL); }      \
            else MFQ_RUNTIME_CHECK(false, "GDN CUDA: D must be in {32,64,128}, got ", D);   \
        }                                                                              \
    } while (0)

static std::vector<mfq_tensor_backend::Tensor> gdn_cuda_impl(
    mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor k, mfq_tensor_backend::Tensor v,
    mfq_tensor_backend::Tensor g, mfq_tensor_backend::Tensor beta, MfqOptional<mfq_tensor_backend::Tensor> state,
    bool inplace_state, bool transposed_state=false, bool tiled_heads=false)
{
    MFQ_RUNTIME_CHECK(q.is_cuda() && q.is_contiguous() && q.scalar_type() == mfq_tensor_backend::kFloat32,
                "q must be cuda contiguous f32");
    int B = q.size(0), Hq = q.size(1), T = q.size(2), D = q.size(3);
    int Hv = v.size(1);
    int H = Hv;
    int kda = (g.dim() == 4) ? 1 : 0;
    auto opts = q.options();
    MFQ_RUNTIME_CHECK(k.is_cuda() && k.is_contiguous() && k.scalar_type() == mfq_tensor_backend::kFloat32,
                "k must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(v.is_cuda() && v.is_contiguous() && v.scalar_type() == mfq_tensor_backend::kFloat32,
                "v must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(k.size(0) == B && k.size(1) == Hq && k.size(2) == T && k.size(3) == D,
                "GDN CUDA: k shape mismatch");
    MFQ_RUNTIME_CHECK(v.size(0) == B && v.size(2) == T && v.size(3) == D,
                "GDN CUDA: v shape mismatch");
    MFQ_RUNTIME_CHECK(Hv % Hq == 0, "GDN CUDA: value heads must be divisible by key heads");
    MFQ_RUNTIME_CHECK(beta.is_cuda() && beta.is_contiguous() && beta.scalar_type() == mfq_tensor_backend::kFloat32,
                "beta must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(beta.dim() == 3 && beta.size(0) == B && beta.size(1) == Hv && beta.size(2) == T,
                "GDN CUDA: beta must be [B,Hv,T]");
    MFQ_RUNTIME_CHECK(g.is_cuda() && g.is_contiguous() && g.scalar_type() == mfq_tensor_backend::kFloat32,
                "g must be cuda contiguous f32");
    if (kda) {
        MFQ_RUNTIME_CHECK(g.size(0) == B && g.size(1) == Hv && g.size(2) == T && g.size(3) == D,
                    "GDN CUDA: KDA g must be [B,Hv,T,D]");
    } else {
        MFQ_RUNTIME_CHECK(g.dim() == 3 && g.size(0) == B && g.size(1) == Hv && g.size(2) == T,
                    "GDN CUDA: g must be [B,Hv,T]");
    }
    auto out = mfq_tensor_backend::empty({B, Hv, T, D}, opts);
    MFQ_RUNTIME_CHECK(!inplace_state || (state.has_value() && state->defined() && state->numel() > 0),
                "GDN CUDA inplace: state is required");
    auto s_out = inplace_state ? state.value() : mfq_tensor_backend::empty({B, Hv, D, D}, opts);
    MFQ_RUNTIME_CHECK(s_out.is_cuda() && s_out.is_contiguous() && s_out.scalar_type() == mfq_tensor_backend::kFloat32,
                "GDN CUDA: output state must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(s_out.size(0) == B && s_out.size(1) == Hv && s_out.size(2) == D && s_out.size(3) == D,
                "GDN CUDA: state shape must be [B,Hv,D,D]");

    const float *qd = q.data_ptr<float>(), *kd = k.data_ptr<float>(), *vd = v.data_ptr<float>();
    const float *gd = g.data_ptr<float>(), *bd = beta.data_ptr<float>();
    auto initial_state = state.has_value() && state->defined() && state->numel() > 0
        ? state->contiguous() : mfq_tensor_backend::Tensor{};
    const float* sd = initial_state.defined() ? initial_state.data_ptr<float>() : nullptr;
    float* od = out.data_ptr<float>();
    float* sod = s_out.data_ptr<float>();
    cudaStream_t stream = mfq_current_cuda_stream();
    int shmem = D * D * (int)sizeof(float);

    const char* col_env = std::getenv("MFQ_GDN_COLUMN");
    const char* warp_env = std::getenv("MFQ_GDN_WARP");
    bool use_warp = transposed_state || !(warp_env && warp_env[0] == '0');
    if (use_warp) {
        if (tiled_heads) { DISPATCH_GDN_WARP(true); }
        else { DISPATCH_GDN_WARP(false); }
    } else if (T <= 4 && col_env && col_env[0] == '1') {
        MFQ_RUNTIME_CHECK(Hq == Hv, "GDN CUDA column path requires repeated q/k heads");
        if (D == 32) { LAUNCH_GDN_COL(32); }
        else if (D == 64) { LAUNCH_GDN_COL(64); }
        else if (D == 128) { LAUNCH_GDN_COL(128); }
        else MFQ_RUNTIME_CHECK(false, "GDN CUDA: D must be in {32,64,128}, got ", D);
    } else {
        MFQ_RUNTIME_CHECK(Hq == Hv, "GDN CUDA shared-memory path requires repeated q/k heads");
        if (D == 32) { LAUNCH_GDN(32); }
        else if (D == 64) { LAUNCH_GDN(64); }
        else if (D == 128) { LAUNCH_GDN(128); }
        else MFQ_RUNTIME_CHECK(false, "GDN CUDA: D must be in {32,64,128}, got ", D);
    }

    return {out, s_out};
}

std::vector<mfq_tensor_backend::Tensor> gdn_cuda(
    mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor k, mfq_tensor_backend::Tensor v,
    mfq_tensor_backend::Tensor g, mfq_tensor_backend::Tensor beta, MfqOptional<mfq_tensor_backend::Tensor> state)
{
    return gdn_cuda_impl(q, k, v, g, beta, state, false, false);
}

std::vector<mfq_tensor_backend::Tensor> gdn_inplace_cuda(
    mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor k, mfq_tensor_backend::Tensor v,
    mfq_tensor_backend::Tensor g, mfq_tensor_backend::Tensor beta, mfq_tensor_backend::Tensor state)
{
    return gdn_cuda_impl(q, k, v, g, beta, state, true, false);
}

std::vector<mfq_tensor_backend::Tensor> gdn_transposed_cuda(
    mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor k, mfq_tensor_backend::Tensor v,
    mfq_tensor_backend::Tensor g, mfq_tensor_backend::Tensor beta, MfqOptional<mfq_tensor_backend::Tensor> state)
{
    return gdn_cuda_impl(q, k, v, g, beta, state, false, true);
}

std::vector<mfq_tensor_backend::Tensor> gdn_inplace_transposed_cuda(
    mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor k, mfq_tensor_backend::Tensor v,
    mfq_tensor_backend::Tensor g, mfq_tensor_backend::Tensor beta, mfq_tensor_backend::Tensor state)
{
    return gdn_cuda_impl(q, k, v, g, beta, state, true, true);
}

std::vector<mfq_tensor_backend::Tensor> gdn_inplace_tiled_cuda(
    mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor k, mfq_tensor_backend::Tensor v,
    mfq_tensor_backend::Tensor g, mfq_tensor_backend::Tensor beta, mfq_tensor_backend::Tensor state)
{
    return gdn_cuda_impl(q, k, v, g, beta, state, true, false, true);
}

std::vector<mfq_tensor_backend::Tensor> gdn_inplace_transposed_tiled_cuda(
    mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor k, mfq_tensor_backend::Tensor v,
    mfq_tensor_backend::Tensor g, mfq_tensor_backend::Tensor beta, mfq_tensor_backend::Tensor state)
{
    return gdn_cuda_impl(q, k, v, g, beta, state, true, true, true);
}

template<class Gate,class Output>
__global__ void gdn_decode_core_kernel(const __half* qkv,const Gate* output_gate,
    const Gate* alpha,const Gate* beta,const float* convolution_state,const float* recurrent_state,
    const float* convolution_weight,const float* dt_bias,const float* a_log,const float* norm_weight,
    Output* output,float* convolution_out,float* recurrent_out,int key_heads,int value_heads,
    int width,int taps,int layout,float convolution_eps,double norm_eps,bool silu_gate,
    bool transposed_state,int64_t as0,int64_t as2,int64_t bs0,int64_t bs2,
    float* workspace) {
    constexpr int tiles=4;
    const int bh=blockIdx.x/tiles,tile=blockIdx.x%tiles;
    const int batch=bh/value_heads,head=bh%value_heads;
    const int key_head=head/(value_heads/key_heads),tid=threadIdx.x;
    const int qk_width=2*key_heads*width,channels=qk_width+value_heads*width;
    __shared__ float query[128],key[128],value[128],attended[128];
    __shared__ float qsum[256],ksum[256],norm_sum[128],decay_value,beta_value;
    __shared__ bool final_tile;
    float q=0,k=0,v=0;
    if(tid<width) {
        const int qc=key_head*width+tid,kc=key_heads*width+qc,vc=qk_width+head*width+tid;
        const size_t state_base=size_t(batch)*(taps-1)*channels;
        for(int tap=0;tap<taps-1;++tap) {
            const size_t wq=layout==0?size_t(qc)*taps+tap:size_t(tap)*channels+qc;
            const size_t wk=layout==0?size_t(kc)*taps+tap:size_t(tap)*channels+kc;
            const size_t wv=layout==0?size_t(vc)*taps+tap:size_t(tap)*channels+vc;
            q+=convolution_state[state_base+size_t(tap)*channels+qc]*convolution_weight[wq];
            k+=convolution_state[state_base+size_t(tap)*channels+kc]*convolution_weight[wk];
            v+=convolution_state[state_base+size_t(tap)*channels+vc]*convolution_weight[wv];
        }
        const float q_current=__half2float(qkv[size_t(batch)*channels+qc]);
        const float k_current=__half2float(qkv[size_t(batch)*channels+kc]);
        const float v_current=__half2float(qkv[size_t(batch)*channels+vc]);
        const size_t wq=layout==0?size_t(qc)*taps+taps-1:size_t(taps-1)*channels+qc;
        const size_t wk=layout==0?size_t(kc)*taps+taps-1:size_t(taps-1)*channels+kc;
        const size_t wv=layout==0?size_t(vc)*taps+taps-1:size_t(taps-1)*channels+vc;
        q+=q_current*convolution_weight[wq];
        k+=k_current*convolution_weight[wk];
        v+=v_current*convolution_weight[wv];
        q=q/(1.f+expf(-q));k=k/(1.f+expf(-k));v=v/(1.f+expf(-v));
        query[tid]=q;key[tid]=k;value[tid]=v;
        for(int tap=0;tap<taps-2;++tap) {
            if(tile==0 && head%(value_heads/key_heads)==0) {
                convolution_out[state_base+size_t(tap)*channels+qc]=convolution_state[state_base+size_t(tap+1)*channels+qc];
                convolution_out[state_base+size_t(tap)*channels+kc]=convolution_state[state_base+size_t(tap+1)*channels+kc];
            }
            if(tile==0)convolution_out[state_base+size_t(tap)*channels+vc]=convolution_state[state_base+size_t(tap+1)*channels+vc];
        }
        if(tile==0 && head%(value_heads/key_heads)==0) {
            convolution_out[state_base+size_t(taps-2)*channels+qc]=q_current;
            convolution_out[state_base+size_t(taps-2)*channels+kc]=k_current;
        }
        if(tile==0)convolution_out[state_base+size_t(taps-2)*channels+vc]=v_current;
    }
    if(tid<256) {qsum[tid]=tid<width?q*q:0.f;ksum[tid]=tid<width?k*k:0.f;}
    if(tid==0) {
        const float a=static_cast<float>(alpha[int64_t(batch)*as0+int64_t(head)*as2])+dt_bias[head];
        const float sp=fmaxf(a,0.f)+log1pf(expf(-fabsf(a)));
        const float gate_log=sp*-expf(a_log[head]);
        decay_value=expf(gate_log);
        const float raw_beta=static_cast<float>(beta[int64_t(batch)*bs0+int64_t(head)*bs2]);
        beta_value=1.f/(1.f+expf(-raw_beta));
    }
    __syncthreads();
    for(int stride=128;stride>=32;stride>>=1) {
        if(tid<stride) {qsum[tid]+=qsum[tid+stride];ksum[tid]+=ksum[tid+stride];}
        __syncthreads();
    }
    if(tid<32) {
        float qr=qsum[tid],kr=ksum[tid];
        #pragma unroll
        for(int stride=16;stride>0;stride>>=1) {
            const float qother=__shfl_down_sync(0xffffffff,qr,stride);
            const float kother=__shfl_down_sync(0xffffffff,kr,stride);
            if(tid<stride) {qr+=qother;kr+=kother;}
        }
        if(tid==0) {qsum[0]=qr;ksum[0]=kr;}
    }
    __syncthreads();
    if(tid<width) {
        query[tid]*=1.f/fmaxf(sqrtf(qsum[0]),convolution_eps);
        key[tid]*=1.f/fmaxf(sqrtf(ksum[0]),convolution_eps);
    }
    __syncthreads();
    const int lane=tid&31,warp=tid>>5,warps=blockDim.x>>5;
    const int rows=(width+31)/32;
    const size_t state_head=(size_t(batch)*value_heads+head)*width*width;
    const float decay=decay_value,bt=beta_value,scale=1.f/sqrtf(float(width));
    const int columns_per_tile=(width+tiles-1)/tiles;
    const int column_end=min(width,(tile+1)*columns_per_tile);
    for(int column=tile*columns_per_tile+warp;column<column_end;column+=warps) {
        float states[4],keys[4],queries[4];
        #pragma unroll
        for(int row=0;row<4;++row) if(row<rows) {
            const int i=row*32+lane;
            const size_t index=transposed_state?size_t(column)*width+i:size_t(i)*width+column;
            states[row]=recurrent_state[state_head+index];
            keys[row]=key[i];queries[row]=query[i];
        }
        float projected=0.f;
        #pragma unroll
        for(int row=0;row<4;++row)if(row<rows)projected+=states[row]*keys[row];
        projected=warp_sum(projected);
        const float delta=(value[column]-decay*projected)*bt;
        float result=0.f;
        #pragma unroll
        for(int row=0;row<4;++row) if(row<rows) {
            const int i=row*32+lane;
            const size_t index=transposed_state?size_t(column)*width+i:size_t(i)*width+column;
            // Preserve the original warp-column kernel's contracted operand
            // order, including its one-row and multiple-row distinction.
            states[row]=rows==1
                ? __fmaf_rn(keys[row],delta,__fmul_rn(decay,states[row]))
                : __fmaf_rn(decay,states[row],__fmul_rn(keys[row],delta));
            result+=states[row]*queries[row];
            recurrent_out[state_head+index]=states[row];
        }
        result=warp_sum(result);
        if(lane==0) {
            const float computed=result*scale;
            if(tiles==1)attended[column]=computed;
            else workspace[size_t(bh)*(width+1)+column]=computed;
        }
    }
    if(tiles>1) {
        __threadfence();
        __syncthreads();
        if(tid==0) {
            auto* counter=reinterpret_cast<uint32_t*>(workspace+size_t(bh)*(width+1)+width);
            cuda::atomic_ref<uint32_t,cuda::thread_scope_device> progress(*counter);
            final_tile=progress.fetch_add(1u,cuda::std::memory_order_acq_rel)==unsigned(tiles-1);
        }
        __syncthreads();
        if(!final_tile)return;
        if(tid<width)attended[tid]=workspace[size_t(bh)*(width+1)+tid];
    }
    __syncthreads();
    if(tid<width)norm_sum[tid]=__fadd_rn(0.f,__fmul_rn(attended[tid],attended[tid]));
    __syncthreads();
    for(int stride=width/2;stride>=32;stride>>=1) {
        if(tid<stride)norm_sum[tid]=__fadd_rn(norm_sum[tid],norm_sum[tid+stride]);
        __syncthreads();
    }
    if(tid<32) {
        float partial=norm_sum[tid];
        #pragma unroll
        for(int stride=16;stride>0;stride>>=1) {
            const float other=__shfl_down_sync(0xffffffff,partial,stride);
            if(tid<stride)partial=__fadd_rn(partial,other);
        }
        if(tid==0)norm_sum[0]=partial;
    }
    __syncthreads();
    if(tid<width) {
        const float mean=__fdiv_rn(norm_sum[0],float(width));
        const float variance=float(double(mean)+norm_eps);
        const float inverse=__fdiv_rn(1.f,__fsqrt_rn(variance));
        const size_t index=(size_t(batch)*value_heads+head)*width+tid;
        const float z=static_cast<float>(output_gate[index]);
        const float sigmoid=__fdiv_rn(1.f,__fadd_rn(1.f,expf(-z)));
        const float gate=silu_gate?__fmul_rn(z,sigmoid):sigmoid;
        const float scaled=__fmul_rn(__fmul_rn(attended[tid],inverse),norm_weight[tid]);
        output[index]=static_cast<Output>(__fmul_rn(scaled,gate));
    }
    if(tiles>1) {
        __syncthreads();
        if(tid==0) {
            auto* counter=reinterpret_cast<uint32_t*>(workspace+size_t(bh)*(width+1)+width);
            cuda::atomic_ref<uint32_t,cuda::thread_scope_device> progress(*counter);
            progress.store(0u,cuda::std::memory_order_release);
        }
    }
}

std::vector<mfq_tensor_backend::Tensor> gdn_decode_core_cuda(
    mfq_tensor_backend::Tensor qkv,mfq_tensor_backend::Tensor output_gate,
    mfq_tensor_backend::Tensor alpha,mfq_tensor_backend::Tensor beta,
    mfq_tensor_backend::Tensor convolution_state,mfq_tensor_backend::Tensor recurrent_state,
    mfq_tensor_backend::Tensor convolution_weight,mfq_tensor_backend::Tensor dt_bias,
    mfq_tensor_backend::Tensor a_log,mfq_tensor_backend::Tensor norm_weight,
    int64_t key_heads,int64_t value_heads,int64_t width,
    double convolution_eps,double norm_eps,bool silu_gate,bool output_half,
    bool transposed_state,mfq_tensor_backend::Tensor workspace) {
    constexpr int threads=256,tiles=4;
    namespace tb=mfq_tensor_backend;
    MFQ_RUNTIME_CHECK(width==32 || width==64 || width==128,"GDN decode core width unsupported");
    MFQ_RUNTIME_CHECK(threads==256 || threads==512 || threads==1024,"GDN decode core block width unsupported");
    MFQ_RUNTIME_CHECK(tiles==4,"GDN decode core tile count unsupported");
    MFQ_RUNTIME_CHECK(qkv.dim()==3 && qkv.size(0)>0 && key_heads>0 && value_heads>0 &&
        value_heads%key_heads==0 && key_heads<=std::numeric_limits<int>::max()/width &&
        value_heads<=std::numeric_limits<int>::max()/width,
        "GDN decode core heads/batch disagree");
    const int64_t batch=qkv.size(0),channels=(2*key_heads+value_heads)*width,taps=convolution_state.size(1)+1;
    MFQ_RUNTIME_CHECK(channels<=std::numeric_limits<int>::max() && taps>1 &&
        taps<=std::numeric_limits<int>::max() && batch<=std::numeric_limits<int>::max()/value_heads/tiles &&
        std::isfinite(convolution_eps) && convolution_eps>0 && std::isfinite(norm_eps) && norm_eps>0,
        "GDN decode core geometry/epsilon exceeds supported bounds");
    MFQ_RUNTIME_CHECK(qkv.is_cuda() && qkv.is_contiguous() && qkv.scalar_type()==tb::kFloat16 &&
        qkv.sizes().vec()==std::vector<int64_t>({batch,1,channels}),"GDN decode core qkv disagrees");
    const auto device=qkv.device();
    const auto real=[&](const tb::Tensor& t){return t.is_cuda() && t.device()==device && t.is_contiguous() && t.scalar_type()==tb::kFloat32;};
    MFQ_RUNTIME_CHECK(real(convolution_state) && real(recurrent_state) && real(convolution_weight) &&
        real(dt_bias) && real(a_log) && real(norm_weight),"GDN decode core parameters disagree");
    MFQ_RUNTIME_CHECK(convolution_state.sizes().vec()==std::vector<int64_t>({batch,taps-1,channels}) &&
        recurrent_state.sizes().vec()==std::vector<int64_t>({batch,value_heads,width,width}),"GDN decode core state shape disagrees");
    const bool channels_first=convolution_weight.sizes().vec()==std::vector<int64_t>({channels,1,taps});
    MFQ_RUNTIME_CHECK(channels_first || convolution_weight.sizes().vec()==std::vector<int64_t>({taps,channels}),"GDN decode core conv weight disagrees");
    MFQ_RUNTIME_CHECK(output_gate.is_cuda() && alpha.is_cuda() && beta.is_cuda() &&
        output_gate.device()==device && alpha.device()==device && beta.device()==device &&
        output_gate.is_contiguous() && output_gate.sizes().vec()==std::vector<int64_t>({batch,1,value_heads*width}) &&
        alpha.sizes().vec()==std::vector<int64_t>({batch,1,value_heads}) && beta.sizes()==alpha.sizes() &&
        output_gate.scalar_type()==alpha.scalar_type() && alpha.scalar_type()==beta.scalar_type() &&
        (alpha.scalar_type()==tb::kFloat16 || alpha.scalar_type()==tb::kFloat32),"GDN decode core gate inputs disagree");
    MFQ_RUNTIME_CHECK(dt_bias.numel()==value_heads && a_log.numel()==value_heads && norm_weight.numel()==width,
        "GDN decode core vector width disagrees");
    MfqCudaGuard guard(device);
    auto output=tb::empty({batch,1,value_heads*width},qkv.options().dtype(output_half?tb::kFloat16:tb::kFloat32));
    auto conv=tb::empty_like(convolution_state),state=tb::empty_like(recurrent_state);
    if(tiles>1 && !workspace.defined())workspace=tb::zeros({batch,value_heads,width+1},convolution_state.options());
    MFQ_RUNTIME_CHECK(real(workspace) && workspace.sizes().vec()==std::vector<int64_t>({batch,value_heads,width+1}),
        "GDN decode core workspace shape/device/dtype disagrees");
    const auto launch=[&]<class Gate,class Output>() {
        gdn_decode_core_kernel<Gate,Output><<<unsigned(batch*value_heads*tiles),threads,0,mfq_current_cuda_stream()>>>(
            reinterpret_cast<const __half*>(qkv.data_ptr<mfq_half>()),output_gate.data_ptr<Gate>(),alpha.data_ptr<Gate>(),beta.data_ptr<Gate>(),
            convolution_state.data_ptr<float>(),recurrent_state.data_ptr<float>(),convolution_weight.data_ptr<float>(),
            dt_bias.data_ptr<float>(),a_log.data_ptr<float>(),norm_weight.data_ptr<float>(),output.data_ptr<Output>(),
            conv.data_ptr<float>(),state.data_ptr<float>(),int(key_heads),int(value_heads),int(width),int(taps),channels_first?0:1,
            float(convolution_eps),norm_eps,silu_gate,transposed_state,alpha.stride(0),alpha.stride(2),beta.stride(0),beta.stride(2),
            workspace.defined()?workspace.data_ptr<float>():nullptr);
    };
    if(alpha.scalar_type()==tb::kFloat16) {
        if(output_half)launch.template operator()<mfq_half,mfq_half>();else launch.template operator()<mfq_half,float>();
    } else {
        if(output_half)launch.template operator()<float,mfq_half>();else launch.template operator()<float,float>();
    }
    MFQ_CUDA_CHECK(cudaGetLastError());
    return {output,conv,state};
}
