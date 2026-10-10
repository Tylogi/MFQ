// Compressed GR tiles with three TF32 products: Ah*Bh + Al*Bh + Ah*Bl.
// The residual products retain near-FP32 precision without dense weights.
#include <mma.h>
namespace {
template<class Input,bool AlignedQ8,bool Zero>
__global__ void gr_prefill_tf32_kernel(const Input* input,
        const uint8_t* packed,const uint8_t* row_bits,const int64_t* row_offsets,
        const uint8_t* scales,const uint8_t* minima,const float* row_scales,
        const float* row_minima,const __half* zero_scales,float* output,
        int rows,int width,int outputs,int groups,int gs) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__>=800
    using namespace nvcuda;
    constexpr int BM=32,BN=32,BK=32,Pad=40;
    __shared__ __align__(32) float ah[BM][Pad],al[BM][Pad],bh[BN][Pad],bl[BN][Pad],result[BM][BN];
    const int tid=threadIdx.x,warp=tid/32,wm=(warp/2)*16,wn=(warp%2)*16;
    const int row0=int(blockIdx.y)*BM,out0=int(blockIdx.x)*BN;
    wmma::fragment<wmma::accumulator,16,16,8,float> total;
    wmma::fill_fragment(total,0.0f);
    for(int k0=0;k0<width;k0+=BK) {
        wmma::fragment<wmma::accumulator,16,16,8,float> high,low;
        wmma::fill_fragment(high,0.0f);wmma::fill_fragment(low,0.0f);
        for(int i=tid;i<BM*BK;i+=128) {
            const int r=i/BK,c=i%BK;
            const float value=row0+r<rows && k0+c<width ? float(input[int64_t(row0+r)*width+k0+c]) : 0;
            const float top=wmma::__float_to_tf32(value);
            ah[r][c]=top;al[r][c]=wmma::__float_to_tf32(__fsub_rn(value,top));
        }
        for(int i=tid;i<BN*BK;i+=128) {
            const int r=i/BK,c=i%BK,neuron=out0+r,column=k0+c;
            float value=0;
            if(neuron<outputs && column<width) {
                const int64_t meta=int64_t(neuron)*groups+column/gs;
                float q,scale,minimum;
                if constexpr(Zero) {
                    q=float(static_cast<int8_t>(packed[meta*gs+column%gs]));
                    scale=__half2float(zero_scales[meta]);minimum=0;
                } else {
                    const int bits=AlignedQ8?8:row_bits[neuron];
                    const uint64_t bit=uint64_t(row_offsets[neuron])+uint64_t(column)*bits;
                    unsigned word=packed[bit>>3];const int shift=int(bit&7);
                    if(shift+bits>8)word|=unsigned(packed[(bit>>3)+1])<<8;
                    q=float((word>>shift)&((1u<<bits)-1));
                    scale=row_scales[neuron]*float(scales[meta]);
                    minimum=row_minima[neuron]*float(minima[meta]);
                }
                value=__fsub_rn(__fmul_rn(scale,q),minimum);
            }
            const float top=wmma::__float_to_tf32(value);
            bh[r][c]=top;bl[r][c]=wmma::__float_to_tf32(__fsub_rn(value,top));
        }
        __syncthreads();
        #pragma unroll
        for(int k=0;k<BK;k+=8) {
            wmma::fragment<wmma::matrix_a,16,16,8,wmma::precision::tf32,wmma::row_major> a_hi,a_lo;
            wmma::fragment<wmma::matrix_b,16,16,8,wmma::precision::tf32,wmma::col_major> b_hi,b_lo;
            wmma::load_matrix_sync(a_hi,&ah[wm][k],Pad);
            wmma::load_matrix_sync(b_hi,&bh[wn][k],Pad);
            wmma::load_matrix_sync(b_lo,&bl[wn][k],Pad);
            wmma::mma_sync(low,a_hi,b_lo,low);
            if constexpr(!std::is_same_v<Input,__half>) {
                wmma::load_matrix_sync(a_lo,&al[wm][k],Pad);
                wmma::mma_sync(low,a_lo,b_hi,low);
            }
            wmma::mma_sync(high,a_hi,b_hi,high);
        }
        #pragma unroll
        for(int i=0;i<total.num_elements;++i)total.x[i]+=high.x[i]+low.x[i];
        __syncthreads();
    }
    wmma::store_matrix_sync(&result[wm][wn],total,BN,wmma::mem_row_major);
    __syncthreads();
    for(int i=tid;i<BM*BN;i+=128) {
        const int m=i/BN,n=i%BN;
        if(row0+m<rows && out0+n<outputs)output[int64_t(row0+m)*outputs+out0+n]=result[m][n];
    }
#endif
}
} // namespace
