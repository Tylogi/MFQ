#pragma once
// Each lane owns an original NINT GS24/28 group. Read its packed words once,
// normalize the bit alignment in registers, then extract each original int4.
// This is temporary register state, not an expanded weight representation.
template<int GS,int Bits,bool SharedSource=false,typename Offset=uint64_t>
struct NintWholeGroup {
    static constexpr int Words=(GS*Bits+31)/32;
    uint32_t normalized[Words+1];
    __device__ __forceinline__ NintWholeGroup(const uint8_t* stream,Offset bit,Offset bytes,bool group_load=false) {
        const Offset start=(bit>>3)&~Offset(3);
        const int shift=int(bit&31u);
        uint32_t packed[Words+1];
        if(group_load && start+4*(Words+1)<=bytes) {
            // One proof covers every packed word. The final, incomplete group
            // retains byte-checked loads without padding or weight expansion.
#pragma unroll
            for(int i=0;i<=Words;++i) {
                const auto* p=reinterpret_cast<const uint32_t*>(stream+start)+i;
                packed[i]=SharedSource?*p:__ldg(p);
            }
        } else {
#pragma unroll
        for(int i=0;i<=Words;++i) {
            const Offset at=start+4*i;
            if(at+4<=bytes) {
                const auto* p=reinterpret_cast<const uint32_t*>(stream+at);
                packed[i]=SharedSource?*p:__ldg(p);
            }
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
