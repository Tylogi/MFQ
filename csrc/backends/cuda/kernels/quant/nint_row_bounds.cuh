#pragma once

template<int GS,int Bits,typename Offset=uint64_t>
__device__ __forceinline__ bool nint_row_reads_fit(NintWholeArgs a,int row) {
    constexpr Offset ReadBytes=4*((GS*Bits+31)/32+1);
    const Offset first=Offset(a.offsets[row]),bytes=Offset(a.bytes);
    const Offset last=first+Offset(a.groups-1)*GS*Bits;
    const Offset start=(last>>3)&~Offset(3);
    return a.groups>0 && last>=first && start<=bytes && bytes-start>=ReadBytes;
}

// The caller proves the final aligned read fits once for this immutable row.
// Group starts are monotonic, so all earlier groups fit too. A single decode
// body handles both row-proven reads and the checked final-row fallback.
template<int GS,int Bits,typename Offset=uint64_t>
struct NintRowBoundGroup {
    static constexpr int Words=(GS*Bits+31)/32;
    uint32_t normalized[Words+1];
    __device__ __forceinline__ NintRowBoundGroup(const uint8_t* stream,Offset bit,Offset bytes,bool row_safe=false) {
        const Offset start=(bit>>3)&~Offset(3);
        const int shift=int(bit&31u);
        uint32_t packed[Words+1];
        if(row_safe) {
            #pragma unroll
            for(int i=0;i<=Words;++i)packed[i]=__ldg(reinterpret_cast<const uint32_t*>(stream+start)+i);
        }else {
            #pragma unroll
            for(int i=0;i<=Words;++i) {
                const Offset at=start+4*i;
                if(at+4<=bytes)packed[i]=__ldg(reinterpret_cast<const uint32_t*>(stream+at));
                else {
                    uint32_t word=0;
                    #pragma unroll
                    for(int b=0;b<4;++b)if(at+b<bytes)word|=uint32_t(stream[at+b])<<(8*b);
                    packed[i]=word;
                }
            }
        }
        #pragma unroll
        for(int i=0;i<Words;++i)normalized[i]=__funnelshift_r(packed[i],packed[i+1],shift);
        normalized[Words]=0;
    }
    template<int Chunk> __device__ __forceinline__ int codes() const {
        constexpr int bit=Chunk*4*Bits,index=bit/32,shift=bit%32;
        const uint32_t packed=__funnelshift_r(normalized[index],normalized[index+1],shift);
        const uint32_t pairs=__byte_perm(packed,packed>>(2*Bits),0x5410);
        return int(__byte_perm(pairs,pairs>>Bits,0x6240)&(((1u<<Bits)-1u)*0x01010101u));
    }
};
