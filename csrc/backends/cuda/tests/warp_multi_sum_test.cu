#include "warp_multi_sum.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {
void check(cudaError_t result) {
    if(result!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(result));
}
template<int Count>
__global__ void compare(const float* input,std::uint32_t* output,int cases) {
    const int lane=int(threadIdx.x)&31;
    const int sample=(int(blockIdx.x)*int(blockDim.x)+int(threadIdx.x))/32;
    if(sample>=cases)return;
    float reference[Count],candidate[Count];
#pragma unroll
    for(int row=0;row<Count;++row) {
        const float value=input[(sample*Count+row)*32+lane];
        reference[row]=candidate[row]=value;
    }
#pragma unroll
    for(int row=0;row<Count;++row) {
#pragma unroll
        for(int offset=16;offset>0;offset/=2)
            reference[row]+=__shfl_xor_sync(0xffffffffu,reference[row],offset);
    }
    mfq::cuda::warp_multi_sum::broadcast(candidate,lane);
#pragma unroll
    for(int row=0;row<Count;++row) {
        const int index=((sample*Count+row)*32+lane)*4;
        output[index]=__float_as_uint(reference[row]);
        output[index+1]=__float_as_uint(candidate[row]);
        output[index+2]=__half_as_ushort(__float2half_rn(reference[row]));
        output[index+3]=__half_as_ushort(__float2half_rn(candidate[row]));
    }
}
std::uint32_t next(std::uint32_t& seed) {
    seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;return seed;
}
template<int Count>
void run() {
    constexpr int cases=257;
    std::vector<float> input(cases*Count*32);
    std::uint32_t seed=0x981ba51du+Count;
    for(int sample=0;sample<cases;++sample)for(int row=0;row<Count;++row)
        for(int lane=0;lane<32;++lane) {
            std::uint32_t bits=next(seed);
            // Finite exponents with wide dynamic range, then cancellation,
            // signed zeros, subnormals, and inactive-row zero padding.
            bits=(bits&0x807fffffu)|((32u+next(seed)%160u)<<23);
            if(sample%7==0)bits=(lane&1?0x80000000u:0u)|0x3f000000u;
            if(sample%11==0)bits=(bits&0x80000000u);
            if(sample%13==0)bits&=0x807fffffu;
            if(sample%17==0 && row==Count-1)bits=0;
            std::memcpy(&input[(sample*Count+row)*32+lane],&bits,4);
        }
    const std::size_t result_count=input.size()*4;
    float* device_input=nullptr;std::uint32_t* device_output=nullptr;
    check(cudaMalloc(&device_input,input.size()*sizeof(float)));
    check(cudaMalloc(&device_output,result_count*sizeof(std::uint32_t)));
    check(cudaMemcpy(device_input,input.data(),input.size()*sizeof(float),cudaMemcpyHostToDevice));
    compare<Count><<<(cases+7)/8,256>>>(device_input,device_output,cases);
    check(cudaGetLastError());
    std::vector<std::uint32_t> output(result_count);
    check(cudaMemcpy(output.data(),device_output,result_count*sizeof(std::uint32_t),cudaMemcpyDeviceToHost));
    check(cudaFree(device_output));check(cudaFree(device_input));
    for(std::size_t index=0;index<result_count;index+=4)
        if(output[index]!=output[index+1] || output[index+2]!=output[index+3]) {
            std::fprintf(stderr,"rows=%d item=%zu reference=%08x candidate=%08x half=%04x/%04x\n",
                Count,index/4,output[index],output[index+1],output[index+2],output[index+3]);
            throw std::runtime_error("multi-row butterfly changed result bits");
        }
    std::printf("rows=%d cases=%d float_and_half_all_lanes=%zu exact PASS\n",Count,cases,input.size());
}
}
int main() {
    try {
        run<1>();run<2>();run<4>();run<8>();run<16>();run<32>();
        return 0;
    }catch(const std::exception& error) {
        std::fprintf(stderr,"%s\n",error.what());return 1;
    }
}
