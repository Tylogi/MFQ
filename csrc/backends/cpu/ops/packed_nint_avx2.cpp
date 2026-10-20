#include "packed_nint.h"

#include <algorithm>
#include <cstring>
#include <immintrin.h>

namespace mfq::cpu::detail {
namespace {
__m256i decode8(const PackedNintView& w, std::uint64_t bit, int bits, int valid=8) {
    const auto byte = static_cast<std::size_t>(bit >> 3);
    const int shift = static_cast<int>(bit & 7);
    if (bits == 8 && !shift) {
        if (w.packed_bytes-byte>=8)
            return _mm256_cvtepu8_epi32(_mm_loadl_epi64(
                reinterpret_cast<const __m128i*>(w.packed+byte)));
        std::uint64_t word=0;
        std::memcpy(&word,w.packed+byte,w.packed_bytes-byte);
        return _mm256_cvtepu8_epi32(_mm_cvtsi64_si128(word));
    }
    if (bits == 4 && !shift && w.packed_bytes-byte>=4) {
        std::uint32_t word;
        std::memcpy(&word, w.packed+byte, sizeof(word));
        return _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32(word),
            _mm256_setr_epi32(0,4,8,12,16,20,24,28)), _mm256_set1_epi32(15));
    }
    if (bits < 8 || valid < 8) {
        // Eight adjacent q<=7 codes occupy at most 63 bits. One contiguous
        // load replaces eight overlapping gathers, including q=5/gs=28.
        std::uint64_t word=0;
        if (w.packed_bytes-byte>=8) std::memcpy(&word,w.packed+byte,8);
        else std::memcpy(&word,w.packed+byte,w.packed_bytes-byte);
        const auto low=_mm256_set1_epi32(static_cast<std::uint32_t>(word));
        const auto high=_mm256_set1_epi32(static_cast<std::uint32_t>(word>>32));
        const auto positions=_mm256_setr_epi32(shift,shift+bits,shift+2*bits,
            shift+3*bits,shift+4*bits,shift+5*bits,shift+6*bits,shift+7*bits);
        const auto boundary=_mm256_set1_epi32(32);
        // AVX2 shifts >=32 produce zero, so the two halves and crossing
        // fields combine without branches or reading beyond the packed end.
        const auto first=_mm256_srlv_epi32(low,positions);
        const auto second=_mm256_srlv_epi32(high,_mm256_sub_epi32(positions,boundary));
        const auto crossing=_mm256_sllv_epi32(high,_mm256_sub_epi32(boundary,positions));
        return _mm256_and_si256(_mm256_or_si256(first,_mm256_or_si256(second,crossing)),
            _mm256_set1_epi32((1<<bits)-1));
    }
    const auto last_byte = static_cast<std::size_t>((bit+7*bits) >> 3);
    if (w.packed_bytes >= 4 && last_byte <= w.packed_bytes-4) {
        const auto positions = _mm256_setr_epi32(shift,shift+bits,shift+2*bits,
            shift+3*bits,shift+4*bits,shift+5*bits,shift+6*bits,shift+7*bits);
        const auto bytes = _mm256_srli_epi32(positions,3);
        const auto words = _mm256_i32gather_epi32(
            reinterpret_cast<const int*>(w.packed+byte), bytes, 1);
        return _mm256_and_si256(_mm256_srlv_epi32(words,
            _mm256_and_si256(positions,_mm256_set1_epi32(7))),
            _mm256_set1_epi32((1<<bits)-1));
    }
    // A gather reads four bytes per lane; the final packed row may have no
    // padding and must never read across the allocation or mapped-file end.
    alignas(32) int values[8];
    for (int lane=0; lane<8; ++lane) {
        const auto position = bit+lane*bits;
        const auto offset = static_cast<std::size_t>(position>>3);
        const auto residual = static_cast<int>(position&7);
        std::uint32_t word = w.packed[offset];
        if (residual+bits>8) word |= std::uint32_t(w.packed[offset+1])<<8;
        values[lane] = (word>>residual)&((1<<bits)-1);
    }
    return _mm256_load_si256(reinterpret_cast<const __m256i*>(values));
}

float reduce(__m256 value) {
    auto sum = _mm_add_ps(_mm256_castps256_ps128(value),_mm256_extractf128_ps(value,1));
    sum = _mm_add_ps(sum,_mm_movehl_ps(sum,sum));
    sum = _mm_add_ss(sum,_mm_shuffle_ps(sum,sum,1));
    return _mm_cvtss_f32(sum);
}
} // namespace

void packed_nint_avx2_rows(const PackedNintView& w, const float* input,
    std::int64_t batch, std::int64_t input_stride, float* output,
    std::int64_t output_stride, std::int64_t begin, std::int64_t end) {
    const auto groups = 1 + (w.width-1)/w.group_size;
    if(batch==1) {
        for(auto row=begin;row<end;++row) {
            auto sum=_mm256_setzero_ps();
            const int bits=w.row_bits[row];
            auto bit=static_cast<std::uint64_t>(w.row_bit_offsets[row]);
            for(std::int64_t group=0;group<groups;++group) {
                const auto meta=row*groups+group;
                const auto vscale=_mm256_set1_ps(w.row_scale[row]*w.group_scale[meta]);
                const auto vminimum=_mm256_set1_ps(w.row_min[row]*w.group_min[meta]);
                auto column=group*w.group_size;
                const auto stop=column+std::min(w.group_size,w.width-column);
                for(;column+8<=stop;column+=8,bit+=8*bits) {
                    const auto decoded=_mm256_sub_ps(_mm256_mul_ps(vscale,
                        _mm256_cvtepi32_ps(decode8(w,bit,bits))),vminimum);
                    sum=_mm256_add_ps(sum,_mm256_mul_ps(decoded,_mm256_loadu_ps(input+column)));
                }
                if(column<stop) {
                    const int valid=static_cast<int>(stop-column);
                    const auto mask=_mm256_cmpgt_epi32(_mm256_set1_epi32(valid),
                        _mm256_setr_epi32(0,1,2,3,4,5,6,7));
                    const auto decoded=_mm256_and_ps(_mm256_castsi256_ps(mask),
                        _mm256_sub_ps(_mm256_mul_ps(vscale,
                            _mm256_cvtepi32_ps(decode8(w,bit,bits,valid))),vminimum));
                    sum=_mm256_add_ps(sum,_mm256_mul_ps(decoded,_mm256_maskload_ps(input+column,mask)));
                    bit+=static_cast<std::uint64_t>(valid)*bits;
                }
            }
            output[row]=reduce(sum);
        }
        return;
    }
    for (auto row=begin; row<end; ++row) {
        const int bits = w.row_bits[row];
        for (std::int64_t sample0=0; sample0<batch; sample0+=4) {
            const auto count = std::min<std::int64_t>(4,batch-sample0);
            __m256 accumulators[4];
            for (int sample=0; sample<count; ++sample) accumulators[sample] = _mm256_setzero_ps();
            auto bit = static_cast<std::uint64_t>(w.row_bit_offsets[row]);
            for (std::int64_t group=0; group<groups; ++group) {
                const auto meta = row*groups+group;
                const float scale = w.row_scale[row]*w.group_scale[meta];
                const float minimum = w.row_min[row]*w.group_min[meta];
                const auto vscale = _mm256_set1_ps(scale);
                const auto vminimum = _mm256_set1_ps(minimum);
                auto column = group*w.group_size;
                const auto stop = column+std::min(w.group_size,w.width-column);
                for (; column+8<=stop; column+=8, bit+=8*bits) {
                    const auto decoded = _mm256_sub_ps(_mm256_mul_ps(vscale,
                        _mm256_cvtepi32_ps(decode8(w,bit,bits))),vminimum);
                    for (int sample=0; sample<count; ++sample) {
                        const auto activation = _mm256_loadu_ps(
                            input+(sample0+sample)*input_stride+column);
                        accumulators[sample] = _mm256_add_ps(accumulators[sample],
                            _mm256_mul_ps(decoded,activation));
                    }
                }
                if (column<stop) {
                    const int valid=static_cast<int>(stop-column);
                    const auto mask=_mm256_cmpgt_epi32(_mm256_set1_epi32(valid),
                        _mm256_setr_epi32(0,1,2,3,4,5,6,7));
                    const auto decoded=_mm256_and_ps(_mm256_castsi256_ps(mask),
                        _mm256_sub_ps(_mm256_mul_ps(vscale,
                            _mm256_cvtepi32_ps(decode8(w,bit,bits,valid))),vminimum));
                    for (int sample=0; sample<count; ++sample) {
                        const auto activation=_mm256_maskload_ps(
                            input+(sample0+sample)*input_stride+column,mask);
                        accumulators[sample]=_mm256_add_ps(accumulators[sample],
                            _mm256_mul_ps(decoded,activation));
                    }
                    bit+=static_cast<std::uint64_t>(valid)*bits;
                }
            }
            for (int sample=0; sample<count; ++sample)
                output[(sample0+sample)*output_stride+row] = reduce(accumulators[sample]);
        }
    }
}
} // namespace mfq::cpu::detail
