#include "../../mfq/kernels/cuda/packed_nvq.cuh"
#include "../../mfq/kernels/cuda/nvq_byte_signs.cuh"
#include "../../mfq/kernels/cuda/nvq_cooperative_reads.cuh"
#include <algorithm>
#include <array>
#include <cstdio>
#include <vector>

struct ReadCase {int offset,bytes,base_bit;};
template<int Bits,int Lanes>
__global__ void check_reads(const uint8_t* fixture,const ReadCase* cases,int count,uint64_t* output) {
    const int thread=int(blockIdx.x)*blockDim.x+threadIdx.x;
    const int index=thread/Lanes,lane=thread&(Lanes-1);
    if(index>=count)return;
    const auto c=cases[index];
    output[thread]=mfq::cuda::nvq_cooperative_window<Bits,Lanes>(
        fixture+c.offset,c.bytes,c.base_bit,lane);
}
template<int Bits,int Lanes>
bool verify_reads(const std::vector<uint8_t>& fixture,const std::vector<ReadCase>& cases,
        const uint8_t* device_fixture,const ReadCase* device_cases,uint64_t* device_output) {
    const int count=int(cases.size());
    check_reads<Bits,Lanes><<<(count*Lanes+127)/128,128>>>(device_fixture,device_cases,count,device_output);
    std::vector<uint64_t> values(size_t(count)*Lanes);
    const auto status=cudaMemcpy(values.data(),device_output,values.size()*sizeof(uint64_t),cudaMemcpyDeviceToHost);
    if(status!=cudaSuccess){std::fprintf(stderr,"packed read check: %s\n",cudaGetErrorString(status));return false;}
    for(int i=0;i<count;++i)for(int lane=0;lane<Lanes;++lane) {
        const auto c=cases[i];uint64_t expected=0;
        // Independent bit-at-a-time oracle, with zero bits beyond the declared payload.
        for(int bit=0;bit<Bits;++bit) {
            const int position=c.base_bit+lane*Bits+bit;
            if(position<c.bytes*8)
                expected|=uint64_t((fixture[c.offset+position/8]>>(position%8))&1u)<<bit;
        }
        if(values[size_t(i)*Lanes+lane]!=expected) {
            std::fprintf(stderr,"packed read mismatch Bits=%d Lanes=%d case=%d lane=%d offset=%d bytes=%d base_bit=%d\n",
                Bits,Lanes,i,lane,c.offset,c.bytes,c.base_bit);return false;
        }
    }
    return true;
}
bool verify_cooperative_reads() {
    std::vector<uint8_t> fixture(517);
    for(size_t i=0;i<fixture.size();++i)fixture[i]=uint8_t((i*73+(i*i*19)+37)&255);
    std::vector<ReadCase> cases;
    for(int offset=0;offset<4;++offset)for(int bytes:{0,1,2,3,4,5,7,8,15,16,31,32,63,64,127,129,257,513}) {
        for(int base=0;base<64;++base)cases.push_back({offset,bytes,base});
        for(int delta=-7;delta<=7;++delta)cases.push_back({offset,bytes,std::max(0,bytes*8+delta)});
    }
    uint8_t* device_fixture=nullptr;ReadCase* device_cases=nullptr;uint64_t* device_output=nullptr;
    auto status=cudaMalloc(&device_fixture,fixture.size());
    if(status==cudaSuccess)status=cudaMalloc(&device_cases,cases.size()*sizeof(ReadCase));
    if(status==cudaSuccess)status=cudaMalloc(&device_output,cases.size()*32*sizeof(uint64_t));
    if(status==cudaSuccess)status=cudaMemcpy(device_fixture,fixture.data(),fixture.size(),cudaMemcpyHostToDevice);
    if(status==cudaSuccess)status=cudaMemcpy(device_cases,cases.data(),cases.size()*sizeof(ReadCase),cudaMemcpyHostToDevice);
    bool passed=status==cudaSuccess;
    if(!passed)std::fprintf(stderr,"packed read setup: %s\n",cudaGetErrorString(status));
#define CHECK_READS(B) \
    if(passed)passed=verify_reads<B,16>(fixture,cases,device_fixture,device_cases,device_output); \
    if(passed)passed=verify_reads<B,32>(fixture,cases,device_fixture,device_cases,device_output)
    CHECK_READS(21);CHECK_READS(24);CHECK_READS(27);CHECK_READS(30);CHECK_READS(32);
    CHECK_READS(33);CHECK_READS(36);CHECK_READS(48);CHECK_READS(54);CHECK_READS(60);
#undef CHECK_READS
    cudaFree(device_fixture);cudaFree(device_cases);cudaFree(device_output);
    if(passed)std::printf("NVQ_COOPERATIVE_READS %zu cases, %zu packed windows, CPU exact PASS\n",
        cases.size(),cases.size()*48*10);
    return passed;
}

__global__ void check_signs(int4* output) {
    const int i=int(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=65536)return;
    const uint32_t signs=uint32_t(i>>8);
    uint32_t a=0,b=0;
    for(int byte=0;byte<4;++byte) {
        a|=uint32_t((i+17*byte)&255)<<(byte*8);
        b|=uint32_t((i+17*(byte+4))&255)<<(byte*8);
    }
    const int2 values=make_int2(int(a),int(b));
    const int2 old=mfq::cuda::packed_nvq::apply_sign8(values,signs);
    const int2 candidate=mfq::cuda::nvq_apply_byte_signs8(values,signs);
    output[i]=make_int4(old.x,old.y,candidate.x,candidate.y);
}
int main() {
    int4* device=nullptr;
    auto status=cudaMalloc(&device,65536*sizeof(int4));
    if(status!=cudaSuccess){std::fprintf(stderr,"cudaMalloc: %s\n",cudaGetErrorString(status));return 1;}
    check_signs<<<256,256>>>(device);
    std::vector<int4> values(65536);
    status=cudaMemcpy(values.data(),device,values.size()*sizeof(int4),cudaMemcpyDeviceToHost);
    cudaFree(device);
    if(status!=cudaSuccess){std::fprintf(stderr,"sign check: %s\n",cudaGetErrorString(status));return 1;}
    for(int i=0;i<65536;++i) {
        std::array<uint32_t,2> expected{};
        for(int byte=0;byte<8;++byte) {
            const int v=(i+17*byte)&255;
            const int signed_value=(i&(1<<(byte+8)))?-v:v;
            expected[byte/4]|=uint32_t(signed_value&255)<<((byte%4)*8);
        }
        const auto& row=values[i];
        if(uint32_t(row.x)!=expected[0] || uint32_t(row.y)!=expected[1] ||
           uint32_t(row.z)!=expected[0] || uint32_t(row.w)!=expected[1]) {
            std::fprintf(stderr,"byte sign mismatch at %d\n",i);return 1;
        }
    }
    std::puts("NVQ_BYTE_SIGNS 65536 words, 524288 byte checks, all 256 values and 256 sign masks, CPU exact PASS");
    return verify_cooperative_reads()?0:1;
}
