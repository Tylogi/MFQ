#include "native/env.h"
#include "gr_prefill_blas.h"
#include <cuda_fp16.h>
#include "gr_prefill_decode.cuh"
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
namespace tb=mfq_tensor_backend;

// Each projection decodes one transient FP32 matrix. Affine weight values
// retain their original bits; SGEMM changes only the FP32 reduction order.
template<bool Aligned,bool Zero>
__global__ void decode_gr_blas_workspace_kernel(const uint8_t* packed,const uint8_t* bits,const int64_t* offsets,
 const uint8_t* scales,const uint8_t* minima,const float* outer_scales,const float* outer_minima,
 const __half* zero_scales,float* output,int width,int outputs,int groups,int gs) {
 const int groups4=(width+3)/4,index=int(blockIdx.x)*blockDim.x+threadIdx.x;
 const int row=index/groups4,column=index%groups4*4;if(row>=outputs)return;
 float values[4];gr_prefill_decode_four<Aligned,Zero>(values,packed,bits,offsets,scales,minima,
  outer_scales,outer_minima,zero_scales,row,column,width,groups,gs);
 #pragma unroll
 for(int i=0;i<4;++i)if(column+i<width)output[int64_t(row)*width+column+i]=values[i];
}
tb::Tensor decode_gr_blas_workspace(const NintWeight& w) {
 auto dense=tb::empty({w.out,w.neuron_len},w.q_packed.options().dtype(tb::kFloat32));
 const auto launch=[&](auto a,auto z) {
  constexpr bool aligned=decltype(a)::value,zero=decltype(z)::value;
  decode_gr_blas_workspace_kernel<aligned,zero><<<unsigned((w.out*((w.neuron_len+3)/4)+255)/256),256,0,mfq_current_cuda_stream()>>>(
   w.q_packed.data_ptr<uint8_t>(),zero?nullptr:w.row_q_bits.data_ptr<uint8_t>(),zero?nullptr:w.row_q_bit_offsets.data_ptr<int64_t>(),
   zero?nullptr:w.sub_scale.data_ptr<uint8_t>(),zero?nullptr:w.sub_min.data_ptr<uint8_t>(),
   zero?nullptr:w.neuron_scale.data_ptr<float>(),zero?nullptr:w.neuron_min.data_ptr<float>(),
   zero?w.q8_zero_scale.data_ptr<__half>():nullptr,dense.data_ptr<float>(),int(w.neuron_len),int(w.out),int(w.ng),int(w.gs));
 };
 if(w.q8_zero)launch(std::false_type{},std::true_type{});
 else if(w.aligned_q8)launch(std::true_type{},std::false_type{});
 else launch(std::false_type{},std::false_type{});
 MFQ_CUDA_CHECK(cudaGetLastError());return dense;
}
tb::Tensor gr_prefill_blas_fp32(const NintWeight& w,const tb::Tensor& input) {
 const MfqCudaGuard guard(input.device());
 auto decoded=decode_gr_blas_workspace(w);
 const auto rows=input.numel()/w.neuron_len;
 auto x=input.reshape({rows,w.neuron_len}),right=decoded.transpose(0,1);
 // Convert large Half inputs in bounded tiles. Contiguous conversion now
 // avoids strided indexing; long GR contractions can use SGEMM too without
 // retaining a full-sequence FP32 input alongside the residual and norm.
 const auto tile=input.scalar_type()==tb::kFloat32?rows:
  std::max<int64_t>(1,std::min<int64_t>(8192,(int64_t(320)<<20)/(int64_t(sizeof(float))*w.neuron_len)));
 tb::Tensor output;
 if(rows<=tile)output=tb::matmul(x.to(tb::kFloat32),right);
 else {
  output=tb::empty({rows,w.out},input.options().dtype(tb::kFloat32));
  for(int64_t first=0;first<rows;first+=tile) {
   const auto count=std::min(tile,rows-first);
   auto projected=tb::matmul(x.narrow(0,first,count).to(tb::kFloat32),right);
   output.narrow(0,first,count).copy_(projected);
  }
 }
 auto shape=input.sizes().vec();shape.back()=w.out;
 static std::atomic<bool> traced{false};
 if(!traced.exchange(true))std::fprintf(stderr,"gr_prefill_blas_dispatch scope=first_projection enabled=1 rows=%lld outputs=%lld width=%lld half=%d weight_bytes=%llu\n",
  (long long)rows,(long long)w.out,(long long)w.neuron_len,int(input.scalar_type()==tb::kFloat16),(unsigned long long)decoded.nbytes());
 return output.reshape(shape);
}

namespace {
bool gr_blas_mode() {
 const auto* value=mfq::cuda::env_lookup("MFQ_GR_PREFILL_MATMUL");return !value || std::atoi(value)==32;
}
template<class Gate,class Value>
__global__ void gr_blas_mix_into(const float* projection,const Value* normalized,
 std::conditional_t<std::is_same_v<Gate,float> || std::is_same_v<Value,float>,float,mfq_half>* output,
 int64_t count,int hidden,int streams) {
 using Output=std::conditional_t<std::is_same_v<Gate,float> || std::is_same_v<Value,float>,float,mfq_half>;
 for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=int64_t(gridDim.x)*blockDim.x) {
  const int64_t base=(i/hidden)*hidden*streams+i%hidden;float sum=0;
  for(int stream=0;stream<streams;++stream) {
   const auto at=base+int64_t(stream)*hidden;
   const Gate projected=static_cast<Gate>(projection[at]);
   const Gate gate=static_cast<Gate>(1.f/(1.f+expf(-float(projected))));
   const Output product=static_cast<Output>(__fmul_rn(float(gate),float(normalized[at])));
   sum=__fadd_rn(sum,float(product));
  }
  output[i]=static_cast<Output>(sum/streams);
 }
}
}
bool gr_prefill_blas_selected(const NintWeight& w,const tb::Tensor& input) {
 // Decode/MTP windows keep their existing kernels. Bound decoded weight
 // workspace independently of sequence length for other model geometries.
 return gr_blas_mode() && input.is_cuda() && input.is_contiguous() && input.dim()>0 &&
  (input.scalar_type()==tb::kFloat32 || (input.scalar_type()==tb::kFloat16 && w.out>=32)) &&
  input.size(-1)>0 && input.numel()/input.size(-1)>=128 &&
  w.out>0 && w.neuron_len>0 && w.out<=(int64_t(64)<<20)/sizeof(float)/w.neuron_len;
}
tb::Tensor gr_prefill_blas_mix(const NintWeight& w,const tb::Tensor& input,
 const tb::Tensor& normalized,int64_t streams) {
 const MfqCudaGuard guard(input.device());
 const auto rows=input.numel()/w.neuron_len,hidden=w.out/streams;
 auto dense=decode_gr_blas_workspace(w);auto right=dense.transpose(0,1);
 auto x=input.reshape({rows,w.neuron_len}),values=normalized.reshape({rows,w.out});
 auto shape=input.sizes().vec();shape.back()=hidden;
 const auto dtype=input.scalar_type()==tb::kFloat32 || normalized.scalar_type()==tb::kFloat32?tb::kFloat32:tb::kFloat16;
 auto output=tb::empty(shape,input.options().dtype(dtype));auto flat=output.reshape({rows,hidden});
 int64_t tile=1024;
 // The expanded projection is consumed immediately by the mixing epilogue.
 // Its scratch stays bounded as the prompt grows to 8K/16K and beyond.
 while(tile>1 && tile*w.out*int64_t(sizeof(float))>(int64_t(64)<<20))tile/=2;
 while(tile>1 && tile*w.neuron_len*int64_t(sizeof(float))>(int64_t(320)<<20))tile/=2;
 for(int64_t first=0;first<rows;first+=tile) {
  const auto count=std::min(tile,rows-first);
  auto projected=tb::matmul(x.narrow(0,first,count).to(tb::kFloat32),right);
  const auto elements=count*hidden;
  const auto launch=[&](auto gate,auto type) {
   using Gate=decltype(gate);using Value=decltype(type);
   using Output=std::conditional_t<std::is_same_v<Gate,float> || std::is_same_v<Value,float>,float,mfq_half>;
   gr_blas_mix_into<Gate,Value><<<unsigned((elements+255)/256),256,0,mfq_current_cuda_stream()>>>(
    projected.data_ptr<float>(),values.data_ptr<Value>()+first*w.out,
    flat.data_ptr<Output>()+first*hidden,elements,int(hidden),int(streams));
  };
  const auto typed=[&](auto gate) {
   if(normalized.scalar_type()==tb::kFloat16)launch(gate,mfq_half{});else launch(gate,float{});
  };
  if(input.scalar_type()==tb::kFloat16)typed(mfq_half{});else typed(float{});
 }
 MFQ_CUDA_CHECK(cudaGetLastError());
 static std::atomic<bool> traced{false};
 if(!traced.exchange(true))std::fprintf(stderr,"gr_prefill_blas_dispatch scope=first_mix enabled=1 rows=%lld outputs=%lld width=%lld half=%d tile=%lld weight_bytes=%llu\n",
  (long long)rows,(long long)w.out,(long long)w.neuron_len,int(input.scalar_type()==tb::kFloat16),(long long)tile,(unsigned long long)dense.nbytes());
 return output;
}
void prepare_gr_prefill_blas_runtime() {
 if(!gr_blas_mode())return;
 // Charge lazy SGEMM modules before expert arenas consume the VRAM budget.
 const auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat32);
 const int64_t shapes[3][2]={{320,10240},{10240,320},{4,10240}};
 for(const auto& shape:shapes) {
  auto a=tb::zeros({128,shape[1]},options),b=tb::zeros({shape[1],shape[0]},options);
  (void)tb::matmul(a,b);
 }
}
