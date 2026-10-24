#include "mfq_tensor_backend.h"
#include "mfq_cuda_linear_attention_ops.h"
#include "mfq_cuda_context.h"
#include <vector>
#include <cmath>
#include <cstring>
#include <iostream>
#include <algorithm>
namespace tb=mfq_tensor_backend;
using tb::Tensor;
Tensor data(std::vector<int64_t> shape,int salt,float scale,tb::ScalarType dtype=tb::kFloat32) {
 int64_t n=1;for(auto d:shape)n*=d;std::vector<float> values(n);
 for(int64_t i=0;i<n;++i)values[i]=std::sin(float(i)*.173f+salt)*scale;
 return tb::tensor(values).reshape(shape).to(tb::kCUDA).to(dtype);
}
void exact(const Tensor& a,const Tensor& b,const char* label) {
 auto x=a.contiguous().cpu(),y=b.contiguous().cpu();
 if(x.scalar_type()!=y.scalar_type() || x.sizes()!=y.sizes())throw std::runtime_error(std::string(label)+" shape/dtype differs");
 if(std::memcmp(x.data_ptr(),y.data_ptr(),x.nbytes())) {
  auto xf=x.to(tb::kFloat32),yf=y.to(tb::kFloat32);int64_t first=0;double peak=0;
  for(int64_t i=0;i<xf.numel();++i)if(xf.data_ptr<float>()[i]!=yf.data_ptr<float>()[i]) {
   if(peak==0)first=i;peak=std::max(peak,std::abs(double(xf.data_ptr<float>()[i])-yf.data_ptr<float>()[i]));
  }
  throw std::runtime_error(std::string(label)+" bytes differ first="+std::to_string(first)+" max="+std::to_string(peak));
 }
}
void independent_window_prefixes() {
    constexpr int tokens = 4, nk = 1, nv = 2, width = 128, channels = (2 * nk + nv) * width;
    auto convolution = data({3, 3, channels}, 721, .2f);
    auto recurrent = data({3, nv, width, width}, 722, .1f);
    auto initial_convolution = convolution.clone();
    auto initial_recurrent = recurrent.clone();
    auto qkv = data({2, tokens, channels}, 723, .3f, tb::kFloat16);
    auto gate = data({2, tokens, nv * width}, 724, .4f, tb::kFloat16);
    auto alpha = data({2, tokens, nv}, 725, .2f, tb::kFloat16);
    auto beta = data({2, tokens, nv}, 726, .2f, tb::kFloat16);
    auto weight = data({channels, 1, 4}, 727, .2f);
    auto bias = data({nv}, 728, .2f), a = data({nv}, 729, .2f), norm = data({width}, 730, .2f);
    for (int row : {1, 0}) {
        auto conv = convolution.narrow(0, row, 1);
        auto state = recurrent.narrow(0, row, 1);
        std::vector<Tensor> checkpoints_conv, checkpoints_state;
        for (int token = 1; token < tokens; ++token) {
            checkpoints_conv.push_back(tb::empty_like(conv));
            checkpoints_state.push_back(tb::empty_like(state));
        }
        auto output = gdn_decode_window_cuda(qkv.narrow(0,row,1), gate.narrow(0,row,1),
            alpha.narrow(0,row,1), beta.narrow(0,row,1), conv, state, weight, bias, a, norm,
            nk, nv, 1, 1e-6, 1e-6, true, true, true, checkpoints_conv, checkpoints_state);
        conv.copy_(output[1]);
        const int retained = row ? 3 : 1;
        conv.copy_(checkpoints_conv[retained - 1]);
        state.copy_(checkpoints_state[retained - 1]);
        auto expected_conv = initial_convolution.narrow(0,row,1).clone();
        auto expected_state = initial_recurrent.narrow(0,row,1).clone();
        auto workspace = tb::zeros({1,nv,width + 1}, expected_conv.options());
        for (int token = 0; token < retained; ++token) {
            auto decoded = gdn_decode_core_cuda(qkv.narrow(0,row,1).narrow(1,token,1),
                gate.narrow(0,row,1).narrow(1,token,1), alpha.narrow(0,row,1).narrow(1,token,1),
                beta.narrow(0,row,1).narrow(1,token,1), expected_conv, expected_state, weight, bias, a, norm,
                nk, nv, width, 1e-6, 1e-6, true, true, true, workspace, true);
            expected_conv.copy_(decoded[1]);
        }
        exact(conv, expected_conv, "independent accepted convolution prefix");
        exact(state, expected_state, "independent accepted recurrent prefix");
    }
    exact(convolution.narrow(0,2,1), initial_convolution.narrow(0,2,1), "paused convolution");
    exact(recurrent.narrow(0,2,1), initial_recurrent.narrow(0,2,1), "paused recurrence");
}

int main(int argc,char** argv)try {
 independent_window_prefixes();
 const bool benchmark=argc==2 && std::string(argv[1])=="--bench";
 const auto context=mfq::cuda::default_context(mfq_current_cuda_device());
 const auto stream=mfq_current_cuda_stream();
 int cases=0,replays=0;
 for(int tokens:{2,3,4,6,8})for(int combination=0;combination<16;++combination) {
  const int nk=combination&1?16:2,nv=nk*(combination&2?3:1),channels=(2*nk+nv)*128;
  const auto gate_type=combination&4?tb::kFloat32:tb::kFloat16;
  const bool output_half=!(combination&8),transposed=!(combination&8),silu=combination&2;
  const int confirmed=combination&8?2:1;
  const double eps=combination&8?.013456789123:1e-6;
  auto qkv=data({1,tokens,channels+17},151,.4f,tb::kFloat16).narrow(2,0,channels);
  auto gate=data({1,tokens,nv*128+7},152,88,gate_type).narrow(2,0,nv*128);
  const float gate_amplitude=gate_type==tb::kFloat32?88.f:4.f;
  auto alpha=data({1,tokens,nv*2},153,gate_amplitude,gate_type).slice(2,0,nv*2,2);
  auto beta=data({1,tokens,nv*2},154,gate_amplitude,gate_type).slice(2,0,nv*2,2);
  auto initial_conv=data({1,3,channels},155,.3f),initial_state=data({1,nv,128,128},156,.17f);
  auto weight=data(combination&8?std::vector<int64_t>{4,channels}:std::vector<int64_t>{channels,1,4},157,.19f);
  auto bias=data({nv},158,1.3f),a=data({nv},159,.3f),norm=data({128},160,1.3f);
  Tensor conv[2]={initial_conv.clone(),initial_conv.clone()},state[2]={initial_state.clone(),initial_state.clone()};
  auto workspace=tb::zeros({1,nv,129},conv[0].options());
  std::vector<Tensor> cc[2],cs[2];
  for(int kind=0;kind<2;++kind)for(int i=0;i<tokens-confirmed;++i){cc[kind].push_back(tb::empty_like(conv[kind]));cs[kind].push_back(tb::empty_like(state[kind]));}
  const auto old=[&] {
   std::vector<Tensor> outputs;
   for(int t=0;t<tokens;++t) {
    const int cp=t+1-confirmed;
    auto result=cp>=0 && t+1<tokens?
     gdn_decode_core_checkpoint_cuda(qkv.narrow(1,t,1),gate.narrow(1,t,1),alpha.narrow(1,t,1),beta.narrow(1,t,1),
      conv[0],state[0],weight,bias,a,norm,nk,nv,128,1e-6,eps,silu,output_half,transposed,workspace,true,cc[0][cp],cs[0][cp]):
     gdn_decode_core_cuda(qkv.narrow(1,t,1),gate.narrow(1,t,1),alpha.narrow(1,t,1),beta.narrow(1,t,1),
      conv[0],state[0],weight,bias,a,norm,nk,nv,128,1e-6,eps,silu,output_half,transposed,workspace,true);
    if(result[1].data_ptr()!=conv[0].data_ptr())conv[0].copy_(result[1]);outputs.push_back(result[0]);
   }
   return tb::cat(outputs,1);
  };
  const auto candidate=[&] {
   auto result=gdn_decode_window_cuda(qkv,gate,alpha,beta,conv[1],state[1],weight,bias,a,norm,
    nk,nv,confirmed,1e-6,eps,silu,output_half,transposed,cc[1],cs[1]);
   conv[1].copy_(result[1]);return result[0];
  };
  const auto reset=[&](int k){conv[k].copy_(initial_conv);state[k].copy_(initial_state);};
  const auto check=[&](const Tensor& actual,const Tensor& expected) {
   exact(actual,expected,"output");exact(state[1],state[0],"state");exact(conv[1],conv[0],"conv");
   for(int i=0;i<tokens-confirmed;++i){exact(cc[1][i],cc[0][i],"checkpoint conv");exact(cs[1][i],cs[0][i],"checkpoint state");}
  };
  auto expected=old(),actual=candidate();check(actual,expected);++cases;
  if(combination==3) {
   mfq::cuda::Graph oldgraph,newgraph;Tensor oldout,newout;
   oldgraph.prepare_memory();oldout=old();mfq_cuda_synchronize();oldout={};reset(0);
   oldgraph.capture_begin();oldout=old();oldgraph.capture_end();
   {
    mfq::cuda::GraphWarmupScope warm(context,stream);
    newout=candidate();mfq_cuda_synchronize();newout={};reset(1);
   }
   newgraph.capture_begin_shared();newout=candidate();newgraph.capture_end();
   auto base=qkv.clone();reset(0);reset(1);
   for(float amplitude:{.3f,.8f,1.1f}) {
    qkv.copy_(base*amplitude);oldgraph.replay();newgraph.replay();check(newout,oldout);++replays;
   }
   if(benchmark) {
   std::vector<float> times[2];
   for(int sample=0;sample<5;++sample)for(int order=0;order<2;++order) {
    const int kind=(sample+order)%2;reset(kind);mfq_cuda_synchronize();
    cudaEvent_t start,end;MFQ_CUDA_CHECK(cudaEventCreate(&start));MFQ_CUDA_CHECK(cudaEventCreate(&end));
    MFQ_CUDA_CHECK(cudaEventRecord(start,stream));
    for(int i=0;i<100;++i)(kind?newgraph:oldgraph).replay();
    MFQ_CUDA_CHECK(cudaEventRecord(end,stream));MFQ_CUDA_CHECK(cudaEventSynchronize(end));
    float ms;MFQ_CUDA_CHECK(cudaEventElapsedTime(&ms,start,end));times[kind].push_back(ms*10);
    MFQ_CUDA_CHECK(cudaEventDestroy(start));MFQ_CUDA_CHECK(cudaEventDestroy(end));
   }
   for(auto& v:times)std::sort(v.begin(),v.end());
   std::cout<<"gdn_register_window tokens="<<tokens<<" heads="<<nv<<" old_us="<<times[0][2]<<" new_us="<<times[1][2]<<std::endl;
   }
  }
 }
 mfq_cuda_synchronize();
 std::cout<<"gdn_register_window exact_cases="<<cases<<" changing_graph="<<replays<<" PASS\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
