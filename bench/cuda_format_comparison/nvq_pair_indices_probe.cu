#include "timing.h"
#include "mfq/kernels/cuda/packed_nvq_pair_indices.cuh"
#include <array>
#include <random>

namespace fb=format_bench;
namespace pn=mfq::cuda::packed_nvq;
template<int Bits,bool Paired,bool Verify>
__global__ void read_indices(const uint8_t* data,int64_t bytes,uint2* output,int rows,int vectors) {
    const int row=blockIdx.x*8+threadIdx.x/16,lane=threadIdx.x&15;
    if(row>=rows)return;
    uint2 sum=make_uint2(0,0);
    for(int pair=lane;pair<(vectors+1)/2;pair+=16) {
        const int64_t index=int64_t(row)*vectors+pair*2;
        const bool complete=pair*2+1<vectors;
        uint2 value;
        if constexpr(Paired)value=pn::load_d4_index_pair<Bits>(data,index,bytes,complete);
        else value=make_uint2(pn::load_packed_bits(data,index*Bits,Bits,bytes),
            complete?pn::load_packed_bits(data,(index+1)*Bits,Bits,bytes):0);
        if constexpr(Verify)output[int64_t(row)*((vectors+1)/2)+pair]=value;
        else {sum.x+=value.x;sum.y+=value.y;}
    }
    if constexpr(!Verify)output[int64_t(row)*16+lane]=sum;
}
template<int Bits>
void run(cudaStream_t stream) {
    constexpr int rows=17,max_vectors=65;
    constexpr int max_bytes=(rows*max_vectors*Bits+7)/8;
    uint8_t* source=nullptr;uint2* output=nullptr;
    fb::check(cudaMalloc(&source,max_bytes+8));
    fb::check(cudaMalloc(&output,rows*((max_vectors+1)/2)*sizeof(uint2)));
    std::mt19937 random(0x124519u+Bits);
    std::size_t checked=0;
    for(int prefix=0;prefix<8;++prefix)for(int vectors=1;vectors<=max_vectors;++vectors) {
        const int bytes=(rows*vectors*Bits+7)/8;
        std::vector<uint8_t> values(bytes);
        for(auto& v:values)v=uint8_t(random());
        fb::check(cudaMemcpyAsync(source+prefix,values.data(),bytes,cudaMemcpyHostToDevice,stream));
        read_indices<Bits,true,true><<<(rows+7)/8,128,0,stream>>>(source+prefix,bytes,output,rows,vectors);
        std::vector<uint2> actual(rows*((vectors+1)/2));
        fb::check(cudaMemcpyAsync(actual.data(),output,actual.size()*sizeof(uint2),cudaMemcpyDeviceToHost,stream));
        fb::check(cudaStreamSynchronize(stream));
        const auto scalar=[&](int64_t index) {
            uint32_t result=0;
            for(int bit=0;bit<Bits;++bit) {
                const auto position=index*Bits+bit;
                result|=((values[position/8]>>(position%8))&1u)<<bit;
            }
            return result;
        };
        for(int row=0;row<rows;++row)for(int pair=0;pair<(vectors+1)/2;++pair) {
            const auto a=actual[row*((vectors+1)/2)+pair];
            if(a.x!=scalar(int64_t(row)*vectors+pair*2) ||
                a.y!=(pair*2+1<vectors?scalar(int64_t(row)*vectors+pair*2+1):0))
                throw std::runtime_error("paired D4 reader differs from CPU bit oracle");
            ++checked;
        }
    }
    fb::check(cudaFree(source));fb::check(cudaFree(output));
    constexpr int timed_rows=65536,vectors=640;
    constexpr int64_t bytes=int64_t(timed_rows)*vectors*Bits/8;
    fb::check(cudaMalloc(&source,bytes));
    fb::check(cudaMalloc(&output,int64_t(timed_rows)*16*sizeof(uint2)));
    std::vector<uint8_t> values(bytes);
    for(auto& v:values)v=uint8_t(random());
    fb::check(cudaMemcpyAsync(source,values.data(),bytes,cudaMemcpyHostToDevice,stream));
    std::vector<uint2> expected(int64_t(timed_rows)*16),actual(expected.size());
    for(int variant:{0,1,0}) {
        cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
        fb::check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        if(variant)read_indices<Bits,true,false><<<(timed_rows+7)/8,128,0,stream>>>(source,bytes,output,timed_rows,vectors);
        else read_indices<Bits,false,false><<<(timed_rows+7)/8,128,0,stream>>>(source,bytes,output,timed_rows,vectors);
        fb::check(cudaStreamEndCapture(stream,&graph));
        fb::check(cudaGraphInstantiate(&executable,graph,0));
        fb::check(cudaGraphLaunch(executable,stream));
        fb::check(cudaMemcpyAsync(actual.data(),output,actual.size()*sizeof(uint2),cudaMemcpyDeviceToHost,stream));
        fb::check(cudaStreamSynchronize(stream));
        if(!variant)expected=actual;
        else for(std::size_t i=0;i<actual.size();++i)
            if(actual[i].x!=expected[i].x || actual[i].y!=expected[i].y)
                throw std::runtime_error("paired D4 streaming checksum differs");
        const auto samples=fb::measure([&]{fb::check(cudaGraphLaunch(executable,stream));},stream,1);
        const auto us=fb::median(samples);
        std::cout<<"READER bits="<<Bits<<" paired="<<variant<<" bytes="<<bytes<<" us="<<us
            <<" GBs="<<bytes/(us*1000.)<<" oracle_pairs="<<checked<<'\n'<<std::flush;
        fb::check(cudaGraphExecDestroy(executable));fb::check(cudaGraphDestroy(graph));
    }
    fb::check(cudaFree(source));fb::check(cudaFree(output));
}
int main()try {
    cudaStream_t stream=nullptr;fb::check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    run<9>(stream);run<10>(stream);fb::check(cudaStreamDestroy(stream));
    return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
