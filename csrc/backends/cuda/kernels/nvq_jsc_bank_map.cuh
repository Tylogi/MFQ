#pragma once
#include "packed_nvq.cuh"

namespace mfq::cuda::packed_nvq {
// The validated JSC table has sixteen states and at most four banks. A
// descriptor can retain the complete mapping in one uint32_t. This removes
// the state-dependent metadata load without changing the codebook or scales.
template<int Format>
__device__ __forceinline__ const int8_t* packed_jsc_codebook(
        const int8_t* metadata,uint32_t bank_map,uint32_t state) {
    static_assert(Format==kNvq2Jsc || Format==kNvq2JscL || Format==kNvq2JscXL);
    constexpr int bank_bytes=(1<<format_index_bits(Format))*8;
    return metadata+kJscMetadataBytes+((bank_map>>(2*state))&3u)*bank_bytes;
}
}
