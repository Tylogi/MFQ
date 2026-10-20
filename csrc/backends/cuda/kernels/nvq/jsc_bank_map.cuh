#pragma once
#include "kernels/quant/packed_nvq.cuh"

namespace mfq::cuda::packed_nvq {
// Mode 1 is checked against all sixteen LUT entries during unpacking. In a
// two-bank D4 table the FP16 scales are exact integers 1..8. Inspect the mode
// once per projection, outside the compressed-group loop.
template<int Format>
__device__ __forceinline__ bool jsc_d4_analytic_scales(const int8_t* metadata) {
    if constexpr(Format==kNvq3Jsc || Format==kNvq3Jsc512 || Format==kNvq3JscL)
        return uint8_t(metadata[1])==2 && uint8_t(metadata[3])==1;
    return false;
}

// The validated JSC table has sixteen states and at most four banks. A
// descriptor can retain the complete mapping in one uint32_t. This removes
// the state-dependent metadata load without changing the codebook or scales.
template<int Format>
__device__ __forceinline__ const int8_t* packed_jsc_codebook(
        const int8_t* metadata,uint32_t bank_map,uint32_t state) {
    constexpr bool d4=Format==kNvq3Jsc || Format==kNvq3Jsc512 || Format==kNvq3JscL;
    static_assert(d4 || Format==kNvq2Jsc || Format==kNvq2JscL || Format==kNvq2JscXL);
    constexpr int bank_bytes=(1<<format_index_bits(Format))*(d4?4:8);
    return metadata+kJscMetadataBytes+((bank_map>>(2*state))&3u)*bank_bytes;
}
}
