#include "nvq_group.h"

#include <algorithm>
#include <cstring>
#include <immintrin.h>

namespace mfq::cpu::detail {
namespace {
inline std::uint32_t bits_at(const std::uint8_t* data,std::int64_t bytes,
        std::int64_t bit,int bits) {
    const auto offset=bit>>3;
    std::uint32_t word=0;
    if (bytes-offset>=4) std::memcpy(&word,data+offset,4);
    else std::memcpy(&word,data+offset,static_cast<std::size_t>(bytes-offset));
    return (word>>(bit&7))&((1u<<bits)-1u);
}

inline std::uint32_t parity7(std::uint32_t value) {
    value^=value>>4; value^=value>>2; value^=value>>1;
    return value&1u;
}

inline float reduce(__m256 value) {
    auto sum=_mm_add_ps(_mm256_castps256_ps128(value),_mm256_extractf128_ps(value,1));
    sum=_mm_add_ps(sum,_mm_movehl_ps(sum,sum));
    sum=_mm_add_ss(sum,_mm_shuffle_ps(sum,sum,1));
    return _mm_cvtss_f32(sum);
}

#ifdef _MSC_VER
#define MFQ_NVQ_INLINE __forceinline
#else
#define MFQ_NVQ_INLINE inline __attribute__((always_inline))
#endif
template<int Format>
MFQ_NVQ_INLINE void decode_group(const NvqDecodeView& w,std::int64_t row,int group,
        std::uint32_t state,int valid,float scale,__m256 decoded[3]) {
    constexpr bool d4=Format==3 || Format==10 || Format==11 || Format==12 || Format==15;
    constexpr bool delta_format=Format==1 || Format==8;
    constexpr int bits=Format==1 ? 11 : Format==7 ? 7 :
        (Format==8 || Format==12) ? 9 : Format==9 ? 6 :
        (Format==13 || Format==15) ? 10 : Format==14 ? 12 : 8;
    const auto* bank=w.banks[state];
    int delta=1;
    if constexpr (delta_format) {
        const bool negative=bits_at(w.aux,w.aux_bytes,row*w.groups+group,1)!=0;
        delta=negative ? -1 : 1;
        if constexpr (Format==8) bank+=int(negative)*512*8;
    }
    std::uint32_t group_signs=0;
    if constexpr(!delta_format && Format!=7 && Format!=9) {
        group_signs=bits_at(w.aux,w.aux_bytes,(row*w.nsign+group*3)*7,21);
    }
    std::uint64_t group_indices=0;
    if constexpr(bits!=8 && Format!=7) {
        const auto first=(row*w.nvec+(d4?group*6:group*3))*bits;
        const auto byte=first>>3;
        const int shift=static_cast<int>(first&7);
        const auto available=w.index_bytes-byte;
        if(available>=8)std::memcpy(&group_indices,w.indices+byte,8);
        else std::memcpy(&group_indices,w.indices+byte,static_cast<std::size_t>(available));
        group_indices>>=shift;
        if constexpr((d4?6:3)*bits>57) {
            if(shift && (d4?6:3)*bits+shift>64 && available>8)
                group_indices|=static_cast<std::uint64_t>(w.indices[byte+8])<<(64-shift);
        }
    }
    const auto multiplier=_mm256_set1_ps(scale);
    const int chunks=(valid+7)/8;
    for (int chunk=0; chunk<chunks; ++chunk) {
        const int vector8=group*3+chunk;
        const int vector=d4 ? vector8*2 : vector8;
        const auto linear=row*w.nvec+vector;
        const std::uint32_t code=bits==8 ? w.indices[linear] :
            Format==7 ? bits_at(w.indices,w.index_bytes,linear*bits,bits) :
            static_cast<std::uint32_t>((group_indices>>(bits*chunk*(d4?2:1)))&((1u<<bits)-1u));
        __m128i values;
        if constexpr (Format==7 || d4) {
            std::uint64_t word=0;
            if constexpr (Format==7) {
                std::memcpy(&word,bank+(code&7u)*4,4);
                std::memcpy(reinterpret_cast<char*>(&word)+4,
                    bank+256+state*32+(code>>3)*4,4);
            } else {
                std::memcpy(&word,bank+std::int64_t(code)*4,4);
                if (vector+1<w.nvec) {
                    const std::uint32_t second=bits==8 ? w.indices[linear+1] :
                        static_cast<std::uint32_t>((group_indices>>(bits*(chunk*2+1)))&((1u<<bits)-1u));
                    std::memcpy(reinterpret_cast<char*>(&word)+4,bank+std::int64_t(second)*4,4);
                }
            }
            values=_mm_cvtsi64_si128(static_cast<long long>(word));
        } else values=_mm_loadl_epi64(reinterpret_cast<const __m128i*>(bank+std::int64_t(code)*8));
        if constexpr (delta_format) {
            constexpr int shift=Format==1 ? 3 : 5;
            // Each byte wraps independently, matching the canonical int8
            // decoder. Masking removes bits shifted from its adjacent byte.
            const auto mask=_mm_set1_epi8(static_cast<char>(0xffu<<shift));
            values=_mm_and_si128(_mm_slli_epi16(values,shift),mask);
            values=_mm_add_epi8(values,_mm_set1_epi8(static_cast<char>(delta*(Format==1 ? 1 : 5))));
        } else if constexpr (Format!=7 && Format!=9) {
            const auto mask7=(group_signs>>(chunk*7))&127u;
            const auto last=parity7(mask7)^((Format==2 && w.sign_mode!=0) ? ((code>>7)&1u) : 0u);
            const auto mask8=mask7|(last<<7);
            const auto selected=_mm_and_si128(_mm_set1_epi8(static_cast<char>(mask8)),
                _mm_setr_epi8(1,2,4,8,16,32,64,static_cast<char>(128),0,0,0,0,0,0,0,0));
            const auto positive=_mm_cmpeq_epi8(selected,_mm_setzero_si128());
            const auto signs=_mm_or_si128(_mm_andnot_si128(positive,_mm_set1_epi8(-1)),_mm_set1_epi8(1));
            values=_mm_sign_epi8(values,signs);
        }
        decoded[chunk]=_mm256_mul_ps(multiplier,_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(values)));
    }
}
#undef MFQ_NVQ_INLINE

template<int Format>
void group_dot(const NvqDecodeView& w,std::int64_t row,int group,std::uint32_t state,
        const float* input,std::int64_t batch,std::int64_t stride,int valid,
        float scale,float* accumulators) {
    if (valid==0 || batch==0) return;
    __m256 decoded[3];
    decode_group<Format>(w,row,group,state,valid,scale,decoded);
    const int chunks=(valid+7)/8;
    for (std::int64_t sample=0; sample<batch; ++sample) {
        auto sum=_mm256_setzero_ps();
        for (int chunk=0; chunk<chunks; ++chunk) {
            const int count=valid-chunk*8;
            __m256 x;
            if (count>=8) x=_mm256_loadu_ps(input+sample*stride+chunk*8);
            else {
                const auto mask=_mm256_cmpgt_epi32(_mm256_set1_epi32(count),_mm256_setr_epi32(0,1,2,3,4,5,6,7));
                x=_mm256_maskload_ps(input+sample*stride+chunk*8,mask);
            }
            sum=_mm256_add_ps(sum,_mm256_mul_ps(decoded[chunk],x));
        }
        accumulators[sample]+=reduce(sum);
    }
}

template<int Format>
void rows_dot(const NvqDecodeView& w,const float* input,std::int64_t batch,
        std::int64_t input_stride,float* output,std::int64_t output_stride,
        std::int64_t begin,std::int64_t end) {
    for (auto neuron=begin; neuron<end; ++neuron) {
        for (std::int64_t sample0=0; sample0<batch; sample0+=4) {
            const int count=static_cast<int>(std::min<std::int64_t>(4,batch-sample0));
            __m256 sums[4];
            for (int sample=0; sample<count; ++sample) sums[sample]=_mm256_setzero_ps();
            for (int group=0; group<w.groups; ++group) {
                const auto state=bits_at(w.states,w.state_bytes,
                    (neuron*w.groups+group)*w.state_bits,w.state_bits);
                const auto start=std::int64_t(group)*24;
                const int valid=static_cast<int>(std::min<std::int64_t>(24,w.width-start));
                __m256 decoded[3];
                decode_group<Format>(w,neuron,group,state,valid,
                    w.anchors[neuron]*w.multipliers[state],decoded);
                for (int chunk=0; chunk<(valid+7)/8; ++chunk) {
                    const int remaining=valid-chunk*8;
                    const auto mask=_mm256_cmpgt_epi32(_mm256_set1_epi32(remaining),
                        _mm256_setr_epi32(0,1,2,3,4,5,6,7));
                    for (int sample=0; sample<count; ++sample) {
                        const auto* values=input+(sample0+sample)*input_stride+start+chunk*8;
                        const auto x=remaining>=8 ? _mm256_loadu_ps(values) : _mm256_maskload_ps(values,mask);
                        sums[sample]=_mm256_add_ps(sums[sample],_mm256_mul_ps(decoded[chunk],x));
                    }
                }
            }
            for (int sample=0; sample<count; ++sample)
                output[(sample0+sample)*output_stride+neuron]=reduce(sums[sample]);
        }
    }
}

template<int Format>
void rows_dot_single(const NvqDecodeView& w,const float* input,std::int64_t batch,
        std::int64_t input_stride,float* output,std::int64_t output_stride,
        std::int64_t begin,std::int64_t end) {
    if(batch!=1) {
        rows_dot<Format>(w,input,batch,input_stride,output,output_stride,begin,end);return;
    }
    const auto full_groups=static_cast<int>(w.width/24);
    const int tail=static_cast<int>(w.width%24);
    for(auto neuron=begin;neuron<end;++neuron) {
        auto sum=_mm256_setzero_ps();
        // Keep one accumulator in a register. Full groups have three fixed
        // vectors; the final partial group retains the original masked reads.
        for(int group=0;group<full_groups;++group) {
            const auto state=bits_at(w.states,w.state_bytes,
                (neuron*w.groups+group)*w.state_bits,w.state_bits);
            __m256 decoded[3];
            decode_group<Format>(w,neuron,group,state,24,
                w.anchors[neuron]*w.multipliers[state],decoded);
            const auto* x=input+std::int64_t(group)*24;
            sum=_mm256_add_ps(sum,_mm256_mul_ps(decoded[0],_mm256_loadu_ps(x)));
            sum=_mm256_add_ps(sum,_mm256_mul_ps(decoded[1],_mm256_loadu_ps(x+8)));
            sum=_mm256_add_ps(sum,_mm256_mul_ps(decoded[2],_mm256_loadu_ps(x+16)));
        }
        if(tail) {
            const auto state=bits_at(w.states,w.state_bytes,
                (neuron*w.groups+full_groups)*w.state_bits,w.state_bits);
            __m256 decoded[3];
            decode_group<Format>(w,neuron,full_groups,state,tail,
                w.anchors[neuron]*w.multipliers[state],decoded);
            for(int chunk=0;chunk<(tail+7)/8;++chunk) {
                const int remaining=tail-chunk*8;
                const auto* x=input+std::int64_t(full_groups)*24+chunk*8;
                const auto mask=_mm256_cmpgt_epi32(_mm256_set1_epi32(remaining),
                    _mm256_setr_epi32(0,1,2,3,4,5,6,7));
                const auto values=remaining>=8 ? _mm256_loadu_ps(x) : _mm256_maskload_ps(x,mask);
                sum=_mm256_add_ps(sum,_mm256_mul_ps(decoded[chunk],values));
            }
        }
        output[neuron]=reduce(sum);
    }
}
} // namespace

NvqGroupDot nvq_group_dot_avx2(int format) noexcept {
    switch (format) {
#define MFQ_NVQ_CPU_CASE(F) case F: return group_dot<F>;
        MFQ_NVQ_CPU_CASE(1) MFQ_NVQ_CPU_CASE(2) MFQ_NVQ_CPU_CASE(3)
        MFQ_NVQ_CPU_CASE(5) MFQ_NVQ_CPU_CASE(7) MFQ_NVQ_CPU_CASE(8)
        MFQ_NVQ_CPU_CASE(9) MFQ_NVQ_CPU_CASE(10) MFQ_NVQ_CPU_CASE(11)
        MFQ_NVQ_CPU_CASE(12) MFQ_NVQ_CPU_CASE(13) MFQ_NVQ_CPU_CASE(14)
        MFQ_NVQ_CPU_CASE(15)
#undef MFQ_NVQ_CPU_CASE
        default: return nullptr;
    }
}
NvqRowsDot nvq_rows_dot_avx2(int format) noexcept {
    switch (format) {
#define MFQ_NVQ_CPU_CASE(F) case F: return rows_dot<F>;
        MFQ_NVQ_CPU_CASE(1) MFQ_NVQ_CPU_CASE(2) MFQ_NVQ_CPU_CASE(3)
        MFQ_NVQ_CPU_CASE(5) MFQ_NVQ_CPU_CASE(7) MFQ_NVQ_CPU_CASE(8)
        MFQ_NVQ_CPU_CASE(9) MFQ_NVQ_CPU_CASE(10) MFQ_NVQ_CPU_CASE(11)
        MFQ_NVQ_CPU_CASE(12) MFQ_NVQ_CPU_CASE(13) MFQ_NVQ_CPU_CASE(14)
        MFQ_NVQ_CPU_CASE(15)
#undef MFQ_NVQ_CPU_CASE
        default: return nullptr;
    }
}
NvqRowsDot nvq_rows_dot_single_avx2(int format) noexcept {
    switch(format) {
#define MFQ_NVQ_CPU_SINGLE_CASE(F) case F: return rows_dot_single<F>;
        MFQ_NVQ_CPU_SINGLE_CASE(1) MFQ_NVQ_CPU_SINGLE_CASE(2) MFQ_NVQ_CPU_SINGLE_CASE(3)
        MFQ_NVQ_CPU_SINGLE_CASE(5) MFQ_NVQ_CPU_SINGLE_CASE(7) MFQ_NVQ_CPU_SINGLE_CASE(8)
        MFQ_NVQ_CPU_SINGLE_CASE(9) MFQ_NVQ_CPU_SINGLE_CASE(10) MFQ_NVQ_CPU_SINGLE_CASE(11)
        MFQ_NVQ_CPU_SINGLE_CASE(12) MFQ_NVQ_CPU_SINGLE_CASE(13) MFQ_NVQ_CPU_SINGLE_CASE(14)
        MFQ_NVQ_CPU_SINGLE_CASE(15)
#undef MFQ_NVQ_CPU_SINGLE_CASE
        default:return nullptr;
    }
}
} // namespace mfq::cpu::detail
