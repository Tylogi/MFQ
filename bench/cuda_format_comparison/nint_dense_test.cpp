#include "mfq_cuda_quant_ops.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace tb=mfq_tensor_backend;
static void reference(bool enabled) {
#ifdef _WIN32
    _putenv_s("MFQ_NINT_DENSE_REFERENCE",enabled ? "1" : "");
    _putenv_s("MFQ_NINT_SINGLE_ROW","0");
#else
    if(enabled)setenv("MFQ_NINT_DENSE_REFERENCE","1",1);
    else unsetenv("MFQ_NINT_DENSE_REFERENCE");
    setenv("MFQ_NINT_SINGLE_ROW","0",1);
#endif
}
static void test(int gs,int k,int n,int m,bool padding,float amplitude) {
    const int groups=(k+gs-1)/gs,width=groups*gs;
    std::vector<uint8_t> bits(n),scales(n*groups),minima(n*groups);
    std::vector<int64_t> offsets(n);
    std::vector<float> ns(n),nm(n),input(m*k);
    int64_t bit_count=0;
    for(int row=0;row<n;++row) {
        bits[row]=1+row%8;offsets[row]=bit_count;bit_count+=int64_t(width)*bits[row];
        ns[row]=float(row%5+1)*.000123f;nm[row]=float(row%7-3)*.000743f;
        for(int g=0;g<groups;++g) {
            scales[row*groups+g]=(row*11+g*7)%127;
            minima[row*groups+g]=(row*3+g*17)%127;
        }
    }
    std::vector<uint8_t> packed((bit_count+7)/8+(padding?8:0));
    for(int row=0;row<n;++row)for(int col=0;col<width;++col) {
        const unsigned code=(row*17+col*23+col/3)&((1u<<bits[row])-1);
        for(int b=0;b<bits[row];++b) {
            const int64_t bit=offsets[row]+int64_t(col)*bits[row]+b;
            packed[bit/8]|=((code>>b)&1u)<<(bit%8);
        }
    }
    for(int i=0;i<m*k;++i)input[i]=amplitude*(std::sin(float(i+7)*.193f)+.3f*std::cos(float(i+11)*.071f));
    auto q=tb::tensor(packed).to(tb::kCUDA),qb=tb::tensor(bits).to(tb::kCUDA);
    auto qo=tb::tensor(offsets).to(tb::kCUDA),s=tb::tensor(scales).reshape({n,groups}).to(tb::kCUDA);
    auto mn=tb::tensor(minima).reshape({n,groups}).to(tb::kCUDA);
    auto scale=tb::tensor(ns).to(tb::kCUDA),minimum=tb::tensor(nm).to(tb::kCUDA);
    auto x=tb::tensor(input).reshape({m,k}).to(tb::kCUDA,tb::kFloat16);
    auto qx=tb::empty({m,width},q.options().dtype(tb::kInt8));
    auto xs=tb::empty({m,groups},q.options().dtype(tb::kFloat32));
    reference(false);
    auto actual=nint_matmul_ws_cuda(q,qb,qo,s,mn,scale,minimum,x,gs,qx,xs).to(tb::kCPU,tb::kFloat32);
    auto decoded=nint_decode_cuda(q,qb,qo,s,mn,scale,minimum,k,gs).to(tb::kCPU);
    reference(true);
    auto original=nint_decode_cuda(q,qb,qo,s,mn,scale,minimum,k,gs).to(tb::kCPU);
    if(std::memcmp(decoded.data_ptr(),original.data_ptr(),size_t(decoded.numel())*decoded.element_size()))
        throw std::runtime_error("vector decode changed packed weight values");
    auto codes=qx.to(tb::kCPU),anchors=xs.to(tb::kCPU);
    for(int a=0;a<m;++a)for(int row=0;row<n;++row) {
        double expected=0,absolute_sum=0;
        for(int group=0;group<groups;++group) {
            int dot=0,sum=0;
            for(int i=0;i<gs;++i) {
                const int col=group*gs+i;
                const int w=(row*17+col*23+col/3)&((1u<<bits[row])-1);
                const int v=codes.data_ptr<int8_t>()[a*width+col];dot+=w*v;sum+=v;
            }
            const double value=double(anchors.data_ptr<float>()[a*groups+group])*
                (double(ns[row])*scales[row*groups+group]*dot-double(nm[row])*minima[row*groups+group]*sum);
            expected+=value;absolute_sum+=std::abs(value);
        }
        const double value=actual.data_ptr<float>()[a*n+row];
        // FP16 rounding plus an FP32 reduction error bound, also near cancellation.
        const double tolerance=std::abs(expected)*.0005+absolute_sum*2e-6+6e-8;
        if(!std::isfinite(value) || std::abs(value-expected)>tolerance)
            throw std::runtime_error("FP64 reference mismatch gs="+std::to_string(gs)+" K="+std::to_string(k)+
                " M="+std::to_string(m)+" row="+std::to_string(row)+" got="+std::to_string(value)+
                " expected="+std::to_string(expected));
    }
}
int main()try {
    int devices=0;
    const auto status=cudaGetDeviceCount(&devices);
    if(status==cudaErrorNoDevice || status==cudaErrorInsufficientDriver ||
       (status==cudaSuccess && devices==0))return 77;
    if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
    int count=0;
    for(int gs:{24,28})for(int k:{1,24,25,28,29,640,2560})
        for(int m:{1,2,3,4,5,7,8})for(bool pad:{false,true})for(float amplitude:{0.f,.13f,8.f}) {
            test(gs,k,17,m,pad,amplitude);++count;
        }
    std::cout<<"NINT dense FP64 reference and exact decode: "<<count<<" cases PASS\n";
    return 0;
}catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
