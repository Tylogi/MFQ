// Fused causal depthwise SSM conv + SiLU.
// conv_input [B, K - 1 + T, C], weight [C,1,K], [C,K], or [K,C], output [B,T,C].

#include <cuda_runtime.h>
#include "mfq_tensor_backend.h"
#include <cuda_fp16.h>
#include <vector>

#include "reduce.cuh"

constexpr int SSM_CONV_BD = 256;

__device__ inline float silu_f32(float x)
{
    return x / (1.0f + expf(-x));
}

__global__ void ssm_conv_silu_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int B,
    int T,
    int C,
    int K,
    int weight_layout,
    int has_bias)
{
    int b = blockIdx.x;
    int c = blockIdx.y * blockDim.x + threadIdx.x;
    if (b >= B || c >= C) {
        return;
    }
    for (int t = 0; t < T; ++t) {
        float sum = 0.0f;
        for (int j = 0; j < K; ++j) {
            float xv = x[((size_t)b * (T + K - 1) + t + j) * C + c];
            float wv = weight_layout == 0 ? w[(size_t)c * K + j] : w[(size_t)j * C + c];
            sum += xv * wv;
        }
        if (has_bias) {
            sum += bias[c];
        }
        out[((size_t)b * T + t) * C + c] = silu_f32(sum);
    }
}

mfq_tensor_backend::Tensor ssm_conv_silu_cuda(mfq_tensor_backend::Tensor conv_input, mfq_tensor_backend::Tensor weight, mfq_tensor_backend::Tensor bias, int64_t n_tokens)
{
    MFQ_RUNTIME_CHECK(conv_input.is_cuda() && conv_input.is_contiguous() && conv_input.scalar_type() == mfq_tensor_backend::kFloat32,
                "ssm_conv_silu: conv_input must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(weight.is_cuda() && weight.is_contiguous() && weight.scalar_type() == mfq_tensor_backend::kFloat32,
                "ssm_conv_silu: weight must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(bias.is_cuda() && bias.is_contiguous() && bias.scalar_type() == mfq_tensor_backend::kFloat32,
                "ssm_conv_silu: bias must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(conv_input.dim() == 3, "ssm_conv_silu: conv_input must be [B,K-1+T,C]");
    int B = (int)conv_input.size(0);
    int C = (int)conv_input.size(2);
    int has_bias = (int)(bias.numel() > 0);
    MFQ_RUNTIME_CHECK(!has_bias || (bias.dim() == 1 && bias.size(0) == C), "ssm_conv_silu: bias must be [C]");
    int K = 0;
    int layout = 0; // 0: channel-major [C,1,K]/[C,K], 1: [K,C]
    if (weight.dim() == 3) {
        MFQ_RUNTIME_CHECK(weight.size(0) == C && weight.size(1) == 1, "ssm_conv_silu: [C,1,K] weight mismatch");
        K = (int)weight.size(2);
        layout = 0;
    } else {
        MFQ_RUNTIME_CHECK(weight.dim() == 2, "ssm_conv_silu: weight must be [C,1,K], [C,K], or [K,C]");
        if (weight.size(0) == C) {
            K = (int)weight.size(1);
            layout = 0;
        } else {
            MFQ_RUNTIME_CHECK(weight.size(1) == C, "ssm_conv_silu: 2D weight must be [C,K] or [K,C]");
            K = (int)weight.size(0);
            layout = 1;
        }
    }
    int T = (int)n_tokens;
    MFQ_RUNTIME_CHECK(T > 0 && conv_input.size(1) == T + K - 1, "ssm_conv_silu: conv_input length must be K-1+T");
    auto out = mfq_tensor_backend::empty({B, T, C}, conv_input.options());
    dim3 blocks(B, (C + SSM_CONV_BD - 1) / SSM_CONV_BD);
    ssm_conv_silu_kernel<<<blocks, SSM_CONV_BD, 0, mfq_current_cuda_stream()>>>(
        conv_input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), out.data_ptr<float>(),
        B, T, C, K, layout, has_bias);
    return out;
}

__global__ void ssm_conv_silu_decode_kernel(
    float* __restrict__ state,
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int B,
    int C,
    int K,
    int weight_layout,
    int has_bias)
{
    int b = blockIdx.x;
    int c = blockIdx.y * blockDim.x + threadIdx.x;
    if (b >= B || c >= C) {
        return;
    }
    size_t state_base = ((size_t)b * (size_t)(K - 1)) * (size_t)C + (size_t)c;
    float sum = 0.0f;
    for (int j = 0; j < K - 1; ++j) {
        float xv = state[state_base + (size_t)j * (size_t)C];
        float wv = weight_layout == 0 ? w[(size_t)c * K + j] : w[(size_t)j * C + c];
        sum += xv * wv;
    }
    float x_cur = x[(size_t)b * (size_t)C + (size_t)c];
    float wv = weight_layout == 0 ? w[(size_t)c * K + (K - 1)] : w[(size_t)(K - 1) * C + c];
    sum += x_cur * wv;
    if (has_bias) {
        sum += bias[c];
    }
    out[(size_t)b * (size_t)C + (size_t)c] = silu_f32(sum);
    for (int j = 0; j < K - 2; ++j) {
        state[state_base + (size_t)j * (size_t)C] = state[state_base + (size_t)(j + 1) * (size_t)C];
    }
    state[state_base + (size_t)(K - 2) * (size_t)C] = x_cur;
}

mfq_tensor_backend::Tensor ssm_conv_silu_decode_cuda(mfq_tensor_backend::Tensor state, mfq_tensor_backend::Tensor x, mfq_tensor_backend::Tensor weight, mfq_tensor_backend::Tensor bias)
{
    MFQ_RUNTIME_CHECK(state.is_cuda() && state.is_contiguous() && state.scalar_type() == mfq_tensor_backend::kFloat32,
                "ssm_conv_silu_decode: state must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == mfq_tensor_backend::kFloat32,
                "ssm_conv_silu_decode: x must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(weight.is_cuda() && weight.is_contiguous() && weight.scalar_type() == mfq_tensor_backend::kFloat32,
                "ssm_conv_silu_decode: weight must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(bias.is_cuda() && bias.is_contiguous() && bias.scalar_type() == mfq_tensor_backend::kFloat32,
                "ssm_conv_silu_decode: bias must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(state.dim() == 3, "ssm_conv_silu_decode: state must be [B,K-1,C]");
    MFQ_RUNTIME_CHECK(x.dim() == 3 && x.size(1) == 1, "ssm_conv_silu_decode: x must be [B,1,C]");
    int B = (int)state.size(0);
    int K = (int)state.size(1) + 1;
    int C = (int)state.size(2);
    MFQ_RUNTIME_CHECK(x.size(0) == B && x.size(2) == C, "ssm_conv_silu_decode: x shape mismatch");
    int has_bias = (int)(bias.numel() > 0);
    MFQ_RUNTIME_CHECK(!has_bias || (bias.dim() == 1 && bias.size(0) == C), "ssm_conv_silu_decode: bias must be [C]");
    int layout = 0;
    if (weight.dim() == 3) {
        MFQ_RUNTIME_CHECK(weight.size(0) == C && weight.size(1) == 1 && weight.size(2) == K,
                    "ssm_conv_silu_decode: [C,1,K] weight mismatch");
        layout = 0;
    } else {
        MFQ_RUNTIME_CHECK(weight.dim() == 2,
                    "ssm_conv_silu_decode: weight must be [C,1,K], [C,K], or [K,C]");
        if (weight.size(0) == C && weight.size(1) == K) {
            layout = 0;
        } else {
            MFQ_RUNTIME_CHECK(weight.size(0) == K && weight.size(1) == C,
                        "ssm_conv_silu_decode: 2D weight must be [C,K] or [K,C]");
            layout = 1;
        }
    }
    auto out = mfq_tensor_backend::empty({B, 1, C}, x.options());
    dim3 blocks(B, (C + SSM_CONV_BD - 1) / SSM_CONV_BD);
    ssm_conv_silu_decode_kernel<<<blocks, SSM_CONV_BD, 0, mfq_current_cuda_stream()>>>(
        state.data_ptr<float>(), x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        out.data_ptr<float>(), B, C, K, layout, has_bias);
    return out;
}

__device__ inline float ssm_decode_cur(
    const __half* __restrict__ qk,
    const __half* __restrict__ v,
    int b,
    int c,
    int qkC,
    int vsz)
{
    if (c < qkC) {
        return __half2float(qk[(size_t)b * (size_t)qkC + (size_t)c]);
    }
    return __half2float(v[(size_t)b * (size_t)vsz + (size_t)(c - qkC)]);
}

template <int BD>
__device__ __forceinline__ void ssm_conv_qk_norm_decode(
    float* __restrict__ state,
    const __half* __restrict__ qk,
    const __half* __restrict__ v,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ q,
    float* __restrict__ k,
    int B,
    int nk,
    int nv,
    int dk,
    int dv,
    int K,
    int weight_layout,
    int has_bias,
    float eps, int row)
{
    int b = row / (2 * nk);
    int rem = row - b * 2 * nk;
    int which = rem / nk;
    int h = rem - which * nk;
    int tid = threadIdx.x;
    int qkC = 2 * nk * dk;
    int vsz = nv * dv;
    int C = qkC + nv * dv;
    int base_c = (which * nk + h) * dk;

    __shared__ float vals[BD];
    __shared__ float sums[BD];
    float val = 0.0f;
    if (tid < dk) {
        int c = base_c + tid;
        size_t state_base = ((size_t)b * (size_t)(K - 1)) * (size_t)C + (size_t)c;
        float sum = 0.0f;
        for (int j = 0; j < K - 1; ++j) {
            float xv = state[state_base + (size_t)j * (size_t)C];
            float wv = weight_layout == 0 ? w[(size_t)c * K + j] : w[(size_t)j * C + c];
            sum += xv * wv;
        }
        float x_cur = ssm_decode_cur(qk, v, b, c, qkC, vsz);
        float wv = weight_layout == 0 ? w[(size_t)c * K + (K - 1)] : w[(size_t)(K - 1) * C + c];
        sum += x_cur * wv;
        if (has_bias) {
            sum += bias[c];
        }
        val = silu_f32(sum);
        vals[tid] = val;
        for (int j = 0; j < K - 2; ++j) {
            state[state_base + (size_t)j * (size_t)C] = state[state_base + (size_t)(j + 1) * (size_t)C];
        }
        state[state_base + (size_t)(K - 2) * (size_t)C] = x_cur;
    }
    float ssq = (tid < dk) ? val * val : 0.0f;
    sums[tid] = ssq;
    __syncthreads();
    for (int stride = BD / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sums[tid] += sums[tid + stride];
        }
        __syncthreads();
    }
    float inv = 1.0f / fmaxf(sqrtf(sums[0]), eps);
    if (tid < dk) {
        float outv = vals[tid] * inv;
        float* dst = which == 0 ? q : k;
        dst[((size_t)b * nk + h) * dk + tid] = outv;
    }
}

__device__ __forceinline__ void ssm_conv_v_decode(
    float* __restrict__ state,
    const __half* __restrict__ qk,
    const __half* __restrict__ v_in,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ v_out,
    int B,
    int nk,
    int nv,
    int dk,
    int dv,
    int K,
    int weight_layout,
    int has_bias, int b, int idx)
{
    int vsz = nv * dv;
    if (b >= B || idx >= vsz) {
        return;
    }
    int qkC = 2 * nk * dk;
    int C = qkC + vsz;
    int c = qkC + idx;
    size_t state_base = ((size_t)b * (size_t)(K - 1)) * (size_t)C + (size_t)c;
    float sum = 0.0f;
    for (int j = 0; j < K - 1; ++j) {
        float xv = state[state_base + (size_t)j * (size_t)C];
        float wv = weight_layout == 0 ? w[(size_t)c * K + j] : w[(size_t)j * C + c];
        sum += xv * wv;
    }
    float x_cur = __half2float(v_in[(size_t)b * (size_t)vsz + (size_t)idx]);
    float wv = weight_layout == 0 ? w[(size_t)c * K + (K - 1)] : w[(size_t)(K - 1) * C + c];
    sum += x_cur * wv;
    if (has_bias) {
        sum += bias[c];
    }
    v_out[(size_t)b * (size_t)vsz + (size_t)idx] = silu_f32(sum);
    for (int j = 0; j < K - 2; ++j) {
        state[state_base + (size_t)j * (size_t)C] = state[state_base + (size_t)(j + 1) * (size_t)C];
    }
    state[state_base + (size_t)(K - 2) * (size_t)C] = x_cur;
}

template<int BD>
__global__ void ssm_conv_qk_norm_decode_kernel(float* state,const __half* qk,const __half* v,
    const float* w,const float* bias,float* q,float* k,int B,int nk,int nv,int dk,int dv,
    int K,int layout,int has_bias,float eps) {
    ssm_conv_qk_norm_decode<BD>(state,qk,v,w,bias,q,k,B,nk,nv,dk,dv,K,layout,has_bias,eps,blockIdx.x);
}
__global__ void ssm_conv_v_decode_kernel(float* state,const __half* qk,const __half* v,
    const float* w,const float* bias,float* output,int B,int nk,int nv,int dk,int dv,
    int K,int layout,int has_bias) {
    ssm_conv_v_decode(state,qk,v,w,bias,output,B,nk,nv,dk,dv,K,layout,has_bias,
        blockIdx.x,blockIdx.y*blockDim.x+threadIdx.x);
}

template<class Gate>
__global__ void ssm_conv_qkv_gate_decode_kernel(float* state,const __half* qk,const __half* v,
    const float* weight,const Gate* alpha,const Gate* beta,const float* dt_bias,const float* a_log,
    float* q,float* k,float* output,float* decay,float* beta_out,
    int B,int nk,int nv,int dk,int dv,int K,int layout,float eps,
    int64_t as0,int64_t as2,int64_t bs0,int64_t bs2) {
    const int v_blocks=(nv*dv+255)/256;
    if(blockIdx.x<unsigned(2*nk)) {
        ssm_conv_qk_norm_decode<256>(state,qk,v,weight,nullptr,q,k,B,nk,nv,dk,dv,K,layout,0,
            eps,blockIdx.y*2*nk+blockIdx.x);
    } else if(blockIdx.x<unsigned(2*nk+v_blocks)) {
        ssm_conv_v_decode(state,qk,v,weight,nullptr,output,B,nk,nv,dk,dv,K,layout,0,
            blockIdx.y,(blockIdx.x-2*nk)*256+threadIdx.x);
    } else {
        const int head=(blockIdx.x-2*nk-v_blocks)*256+threadIdx.x,b=blockIdx.y;
        if(head<nv) {
            const float a=static_cast<float>(alpha[int64_t(b)*as0+int64_t(head)*as2])+dt_bias[head];
            const float sp=fmaxf(a,0.0f)+log1pf(expf(-fabsf(a)));
            decay[int64_t(b)*nv+head]=sp*-expf(a_log[head]);
            const float bv=static_cast<float>(beta[int64_t(b)*bs0+int64_t(head)*bs2]);
            beta_out[int64_t(b)*nv+head]=1.0f/(1.0f+expf(-bv));
        }
    }
}

std::vector<mfq_tensor_backend::Tensor> linear_conv_qkv_gate_decode_cuda(
    mfq_tensor_backend::Tensor state,mfq_tensor_backend::Tensor qk,mfq_tensor_backend::Tensor v,
    mfq_tensor_backend::Tensor weight,mfq_tensor_backend::Tensor alpha,mfq_tensor_backend::Tensor beta,
    mfq_tensor_backend::Tensor dt_bias,mfq_tensor_backend::Tensor a_log,
    int64_t nk,int64_t nv,int64_t dk,int64_t dv,double eps) {
    namespace tb=mfq_tensor_backend;
    MFQ_RUNTIME_CHECK(state.is_cuda() && state.is_contiguous() && state.scalar_type()==tb::kFloat32 &&
        state.dim()==3 && state.size(1)>0 && nk>0 && nv>0 && nv%nk==0 && dk>0 && dk<=256 && dv>0,
        "GDN combined decode state/geometry disagrees");
    const int B=int(state.size(0)),K=int(state.size(1))+1,C=int(2*nk*dk+nv*dv);
    const auto same_device=[&](const tb::Tensor& x){return x.is_cuda() && x.device()==state.device();};
    MFQ_RUNTIME_CHECK(same_device(qk) && same_device(v) && qk.is_contiguous() && v.is_contiguous() &&
        qk.scalar_type()==tb::kFloat16 && v.scalar_type()==tb::kFloat16 &&
        qk.sizes().vec()==std::vector<int64_t>({B,1,2*nk*dk}) &&
        v.sizes().vec()==std::vector<int64_t>({B,1,nv*dv}) && state.size(2)==C,
        "GDN combined decode projection shape/dtype disagrees");
    MFQ_RUNTIME_CHECK(same_device(alpha) && same_device(beta) && alpha.scalar_type()==beta.scalar_type() &&
        (alpha.scalar_type()==tb::kFloat16 || alpha.scalar_type()==tb::kFloat32) &&
        alpha.sizes().vec()==std::vector<int64_t>({B,1,nv}) && beta.sizes()==alpha.sizes(),
        "GDN combined decode gates disagree");
    MFQ_RUNTIME_CHECK(same_device(weight) && same_device(dt_bias) && same_device(a_log) &&
        weight.is_contiguous() && dt_bias.is_contiguous() && a_log.is_contiguous() &&
        weight.scalar_type()==tb::kFloat32 && dt_bias.scalar_type()==tb::kFloat32 && a_log.scalar_type()==tb::kFloat32 &&
        dt_bias.numel()==nv && a_log.numel()==nv,"GDN combined decode parameters disagree");
    const bool channels_first=weight.sizes().vec()==std::vector<int64_t>({C,1,K});
    MFQ_RUNTIME_CHECK(channels_first || weight.sizes().vec()==std::vector<int64_t>({K,C}),
        "GDN combined decode convolution shape disagrees");
    MfqCudaGuard guard(state.device());auto options=state.options();
    auto q=tb::empty({B,nk,1,dk},options),k=tb::empty_like(q),output=tb::empty({B,nv,1,dv},options);
    auto decay=tb::empty({B,nv,1},options),beta_out=tb::empty_like(decay);
    const dim3 grid(unsigned(2*nk+(nv*dv+255)/256+(nv+255)/256),unsigned(B));
    const auto launch=[&](auto tag) {
        using Gate=decltype(tag);
        ssm_conv_qkv_gate_decode_kernel<Gate><<<grid,256,0,mfq_current_cuda_stream()>>>(
            state.data_ptr<float>(),reinterpret_cast<const __half*>(qk.data_ptr<mfq_half>()),
            reinterpret_cast<const __half*>(v.data_ptr<mfq_half>()),weight.data_ptr<float>(),
            alpha.data_ptr<Gate>(),beta.data_ptr<Gate>(),dt_bias.data_ptr<float>(),a_log.data_ptr<float>(),
            q.data_ptr<float>(),k.data_ptr<float>(),output.data_ptr<float>(),decay.data_ptr<float>(),beta_out.data_ptr<float>(),
            B,int(nk),int(nv),int(dk),int(dv),K,channels_first?0:1,float(eps),
            alpha.stride(0),alpha.stride(2),beta.stride(0),beta.stride(2));
    };
    if(alpha.scalar_type()==tb::kFloat16)launch(mfq_half{});else launch(float{});
    MFQ_CUDA_CHECK(cudaGetLastError());return {q,k,output,decay,beta_out};
}

std::vector<mfq_tensor_backend::Tensor> linear_conv_qkv_decode_cuda(
    mfq_tensor_backend::Tensor state,
    mfq_tensor_backend::Tensor qk,
    mfq_tensor_backend::Tensor v,
    mfq_tensor_backend::Tensor weight,
    mfq_tensor_backend::Tensor bias,
    int64_t nk,
    int64_t nv,
    int64_t dk,
    int64_t dv,
    double eps)
{
    MFQ_RUNTIME_CHECK(state.is_cuda() && state.is_contiguous() && state.scalar_type() == mfq_tensor_backend::kFloat32,
                "linear_conv_qkv_decode: state must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(qk.is_cuda() && qk.is_contiguous() && qk.scalar_type() == mfq_tensor_backend::kHalf,
                "linear_conv_qkv_decode: qk must be cuda contiguous f16");
    MFQ_RUNTIME_CHECK(v.is_cuda() && v.is_contiguous() && v.scalar_type() == mfq_tensor_backend::kHalf,
                "linear_conv_qkv_decode: v must be cuda contiguous f16");
    MFQ_RUNTIME_CHECK(weight.is_cuda() && weight.is_contiguous() && weight.scalar_type() == mfq_tensor_backend::kFloat32,
                "linear_conv_qkv_decode: weight must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(bias.is_cuda() && bias.is_contiguous() && bias.scalar_type() == mfq_tensor_backend::kFloat32,
                "linear_conv_qkv_decode: bias must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(state.dim() == 3, "linear_conv_qkv_decode: state must be [B,K-1,C]");
    MFQ_RUNTIME_CHECK(qk.dim() == 3 && qk.size(1) == 1, "linear_conv_qkv_decode: qk must be [B,1,2*nk*dk]");
    MFQ_RUNTIME_CHECK(v.dim() == 3 && v.size(1) == 1, "linear_conv_qkv_decode: v must be [B,1,nv*dv]");
    int B = (int)state.size(0);
    int K = (int)state.size(1) + 1;
    int qkC = (int)(2 * nk * dk);
    int vsz = (int)(nv * dv);
    int C = qkC + vsz;
    MFQ_RUNTIME_CHECK(qk.size(0) == B && qk.size(2) == qkC, "linear_conv_qkv_decode: qk shape mismatch");
    MFQ_RUNTIME_CHECK(v.size(0) == B && v.size(2) == vsz, "linear_conv_qkv_decode: v shape mismatch");
    MFQ_RUNTIME_CHECK(state.size(2) == C, "linear_conv_qkv_decode: state width mismatch");
    MFQ_RUNTIME_CHECK(nv % nk == 0, "linear_conv_qkv_decode: nv must be divisible by nk");
    MFQ_RUNTIME_CHECK(dk <= 256, "linear_conv_qkv_decode: dk > 256 is unsupported");
    int has_bias = (int)(bias.numel() > 0);
    MFQ_RUNTIME_CHECK(!has_bias || (bias.dim() == 1 && bias.size(0) == C), "linear_conv_qkv_decode: bias must be [C]");
    int layout = 0;
    if (weight.dim() == 3) {
        MFQ_RUNTIME_CHECK(weight.size(0) == C && weight.size(1) == 1 && weight.size(2) == K,
                    "linear_conv_qkv_decode: [C,1,K] weight mismatch");
        layout = 0;
    } else {
        MFQ_RUNTIME_CHECK(weight.dim() == 2 && weight.size(0) == K && weight.size(1) == C,
                    "linear_conv_qkv_decode: weight must be [C,1,K] or [K,C]");
        layout = 1;
    }
    auto opts = state.options();
    auto q = mfq_tensor_backend::empty({B, (int)nk, 1, (int)dk}, opts);
    auto k = mfq_tensor_backend::empty({B, (int)nk, 1, (int)dk}, opts);
    auto vo = mfq_tensor_backend::empty({B, (int)nv, 1, (int)dv}, opts);
    cudaStream_t stream = mfq_current_cuda_stream();
    ssm_conv_qk_norm_decode_kernel<256><<<B * 2 * (int)nk, 256, 0, stream>>>(
        state.data_ptr<float>(), reinterpret_cast<const __half*>(qk.data_ptr<mfq_half>()),
        reinterpret_cast<const __half*>(v.data_ptr<mfq_half>()), weight.data_ptr<float>(),
        bias.data_ptr<float>(), q.data_ptr<float>(), k.data_ptr<float>(), B, (int)nk, (int)nv,
        (int)dk, (int)dv, K, layout, has_bias, (float)eps);
    dim3 v_blocks(B, (vsz + SSM_CONV_BD - 1) / SSM_CONV_BD);
    ssm_conv_v_decode_kernel<<<v_blocks, SSM_CONV_BD, 0, stream>>>(
        state.data_ptr<float>(), reinterpret_cast<const __half*>(qk.data_ptr<mfq_half>()),
        reinterpret_cast<const __half*>(v.data_ptr<mfq_half>()), weight.data_ptr<float>(),
        bias.data_ptr<float>(), vo.data_ptr<float>(), B, (int)nk, (int)nv, (int)dk, (int)dv,
        K, layout, has_bias);
    return {q, k, vo};
}

template <int BD>
__global__ void linear_conv_qk_norm_prefill_kernel(
    const float* __restrict__ state,
    const __half* __restrict__ qk,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ q,
    float* __restrict__ k,
    int B, int T, int nk, int dk, int C, int K,
    int weight_layout, int has_bias, float eps)
{
    int row = blockIdx.x;
    int h = row % nk;
    int r = row / nk;
    int which = r & 1;
    r >>= 1;
    int t = r % T;
    int b = r / T;
    int tid = threadIdx.x;
    int qkC = 2 * nk * dk;
    int c = (which * nk + h) * dk + tid;
    float val = 0.0f;
    if (tid < dk) {
        float sum = 0.0f;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            if (j >= K) break;
            int token = t + j - (K - 1);
            float xv;
            if (token < 0) {
                xv = state[((size_t)b * (K - 1) + (token + K - 1)) * C + c];
            } else {
                xv = __half2float(qk[((size_t)b * T + token) * qkC + c]);
            }
            float wv = weight_layout == 0 ? w[(size_t)c * K + j] : w[(size_t)j * C + c];
            sum += xv * wv;
        }
        if (has_bias) sum += bias[c];
        val = silu_f32(sum);
    }
    float ssq = block_sum<BD / 32>(tid < dk ? val * val : 0.0f);
    float inv = 1.0f / fmaxf(sqrtf(ssq), eps);
    if (tid < dk) {
        float* dst = which == 0 ? q : k;
        dst[(((size_t)b * nk + h) * T + t) * dk + tid] = val * inv;
    }
}

__global__ void linear_conv_v_prefill_kernel(
    const float* __restrict__ state,
    const __half* __restrict__ v_in,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ v_out,
    int B, int T, int nk, int nv, int dk, int dv, int C, int K,
    int weight_layout, int has_bias)
{
    int bt = blockIdx.x;
    int b = bt / T;
    int t = bt - b * T;
    int vi = blockIdx.y * blockDim.x + threadIdx.x;
    int vsz = nv * dv;
    if (b >= B || vi >= vsz) return;
    int qkC = 2 * nk * dk;
    int c = qkC + vi;
    float sum = 0.0f;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        if (j >= K) break;
        int token = t + j - (K - 1);
        float xv;
        if (token < 0) {
            xv = state[((size_t)b * (K - 1) + (token + K - 1)) * C + c];
        } else {
            xv = __half2float(v_in[((size_t)b * T + token) * vsz + vi]);
        }
        float wv = weight_layout == 0 ? w[(size_t)c * K + j] : w[(size_t)j * C + c];
        sum += xv * wv;
    }
    if (has_bias) sum += bias[c];
    int h = vi / dv;
    int d = vi - h * dv;
    v_out[(((size_t)b * nv + h) * T + t) * dv + d] = silu_f32(sum);
}

__device__ __forceinline__ float linear_prefill_input(
    const __half* qk, const __half* v, int b, int t, int c,
    int T, int qkC, int vsz)
{
    if (c < qkC) return __half2float(qk[((size_t)b * T + t) * qkC + c]);
    return __half2float(v[((size_t)b * T + t) * vsz + (c - qkC)]);
}

__global__ void linear_conv_state_prefill_kernel(
    const float* __restrict__ old_state,
    const __half* __restrict__ qk,
    const __half* __restrict__ v,
    float* __restrict__ new_state,
    int B, int T, int qkC, int vsz, int C, int K)
{
    size_t total = (size_t)B * (K - 1) * C;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += (size_t)gridDim.x * blockDim.x) {
        int c = (int)(idx % C);
        size_t br = idx / C;
        int j = (int)(br % (K - 1));
        int b = (int)(br / (K - 1));
        int combined = T + j;
        if (combined < K - 1) {
            new_state[idx] = old_state[((size_t)b * (K - 1) + combined) * C + c];
        } else {
            int token = combined - (K - 1);
            new_state[idx] = linear_prefill_input(qk, v, b, token, c, T, qkC, vsz);
        }
    }
}

std::vector<mfq_tensor_backend::Tensor> linear_conv_qkv_prefill_cuda(
    mfq_tensor_backend::Tensor state,
    mfq_tensor_backend::Tensor qk,
    mfq_tensor_backend::Tensor v,
    mfq_tensor_backend::Tensor weight,
    mfq_tensor_backend::Tensor bias,
    int64_t nk,
    int64_t nv,
    int64_t dk,
    int64_t dv,
    double eps)
{
    MFQ_RUNTIME_CHECK(state.is_cuda() && state.is_contiguous() && state.scalar_type() == mfq_tensor_backend::kFloat32,
                "linear_conv_qkv_prefill: state must be contiguous CUDA f32");
    MFQ_RUNTIME_CHECK(qk.is_cuda() && qk.is_contiguous() && qk.scalar_type() == mfq_tensor_backend::kHalf,
                "linear_conv_qkv_prefill: qk must be contiguous CUDA f16");
    MFQ_RUNTIME_CHECK(v.is_cuda() && v.is_contiguous() && v.scalar_type() == mfq_tensor_backend::kHalf,
                "linear_conv_qkv_prefill: v must be contiguous CUDA f16");
    MFQ_RUNTIME_CHECK(weight.is_cuda() && weight.is_contiguous() && weight.scalar_type() == mfq_tensor_backend::kFloat32,
                "linear_conv_qkv_prefill: weight must be contiguous CUDA f32");
    MFQ_RUNTIME_CHECK(bias.is_cuda() && bias.is_contiguous() && bias.scalar_type() == mfq_tensor_backend::kFloat32,
                "linear_conv_qkv_prefill: bias must be contiguous CUDA f32");
    MFQ_RUNTIME_CHECK(state.dim() == 3 && qk.dim() == 3 && v.dim() == 3,
                "linear_conv_qkv_prefill: state/qk/v must be rank 3");
    int B = (int)qk.size(0);
    int T = (int)qk.size(1);
    int K = (int)state.size(1) + 1;
    int qkC = (int)(2 * nk * dk);
    int vsz = (int)(nv * dv);
    int C = qkC + vsz;
    MFQ_RUNTIME_CHECK(T > 1 && qk.size(2) == qkC && v.size(0) == B && v.size(1) == T && v.size(2) == vsz,
                "linear_conv_qkv_prefill: qk/v shape mismatch");
    MFQ_RUNTIME_CHECK(state.size(0) == B && state.size(2) == C, "linear_conv_qkv_prefill: state shape mismatch");
    MFQ_RUNTIME_CHECK(dk <= 256 && K <= 8, "linear_conv_qkv_prefill: requires dk<=256 and K<=8");
    int layout = 0;
    if (weight.dim() == 3) {
        MFQ_RUNTIME_CHECK(weight.size(0) == C && weight.size(1) == 1 && weight.size(2) == K,
                    "linear_conv_qkv_prefill: [C,1,K] weight mismatch");
    } else {
        MFQ_RUNTIME_CHECK(weight.dim() == 2 && weight.size(0) == K && weight.size(1) == C,
                    "linear_conv_qkv_prefill: weight must be [C,1,K] or [K,C]");
        layout = 1;
    }
    int has_bias = (int)(bias.numel() > 0);
    MFQ_RUNTIME_CHECK(!has_bias || (bias.dim() == 1 && bias.size(0) == C), "linear_conv_qkv_prefill: bias shape mismatch");

    auto opts = state.options();
    auto qo = mfq_tensor_backend::empty({B, (int)nk, T, (int)dk}, opts);
    auto ko = mfq_tensor_backend::empty_like(qo);
    auto vo = mfq_tensor_backend::empty({B, (int)nv, T, (int)dv}, opts);
    auto new_state = mfq_tensor_backend::empty_like(state);
    cudaStream_t stream = mfq_current_cuda_stream();
    linear_conv_qk_norm_prefill_kernel<256><<<B * T * 2 * (int)nk, 256, 0, stream>>>(
        state.data_ptr<float>(), reinterpret_cast<const __half*>(qk.data_ptr<mfq_half>()),
        weight.data_ptr<float>(), bias.data_ptr<float>(), qo.data_ptr<float>(), ko.data_ptr<float>(),
        B, T, (int)nk, (int)dk, C, K, layout, has_bias, (float)eps);
    linear_conv_v_prefill_kernel<<<dim3(B * T, (vsz + 255) / 256), 256, 0, stream>>>(
        state.data_ptr<float>(), reinterpret_cast<const __half*>(v.data_ptr<mfq_half>()),
        weight.data_ptr<float>(), bias.data_ptr<float>(), vo.data_ptr<float>(),
        B, T, (int)nk, (int)nv, (int)dk, (int)dv, C, K, layout, has_bias);
    int state_total = B * (K - 1) * C;
    linear_conv_state_prefill_kernel<<<(state_total + 255) / 256, 256, 0, stream>>>(
        state.data_ptr<float>(), reinterpret_cast<const __half*>(qk.data_ptr<mfq_half>()),
        reinterpret_cast<const __half*>(v.data_ptr<mfq_half>()), new_state.data_ptr<float>(),
        B, T, qkC, vsz, C, K);
    return {qo, ko, vo, new_state};
}
