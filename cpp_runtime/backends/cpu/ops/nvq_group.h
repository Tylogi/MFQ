#pragma once

#include <cstdint>

namespace mfq::cpu {
// Borrowed canonical storage and the small state tables prepared by the
// format loader. No expanded weight matrix is retained by this kernel.
struct NvqDecodeView {
    const std::uint8_t* indices;
    const std::uint8_t* aux;
    std::int64_t index_bytes;
    std::int64_t aux_bytes;
    int nvec;
    int nsign;
    int groups;
    int sign_mode;
    const std::int8_t* banks[16]{};
    float multipliers[16]{};
    const std::uint8_t* states=nullptr;
    std::int64_t state_bytes=0;
    const float* anchors=nullptr;
    std::int64_t width=0;
    int state_bits=0;
};

using NvqGroupDot=void(*)(const NvqDecodeView&,std::int64_t,int,std::uint32_t,
    const float*,std::int64_t,std::int64_t,int,float,float*);
// The loader validates the borrowed storage. Inputs begin at the selected
// group, valid is 0..24, and each sample has at least valid readable floats.

// Null selects the portable caller's scalar decoder. Dispatch is performed
// outside the group loop and checks both CPU and OS AVX state support.
NvqGroupDot nvq_group_dot_kernel(int format) noexcept;
using NvqRowsDot=void(*)(const NvqDecodeView&,const float*,std::int64_t,std::int64_t,
    float*,std::int64_t,std::int64_t,std::int64_t);
NvqRowsDot nvq_rows_dot_kernel(int format) noexcept;
// Compact state/index/sign records for JSC E8/D4. Uses indices/index_bytes
// and the same anchors/state tables; aux and states are not accessed.
NvqRowsDot nvq_dense_rows_dot_kernel(int format) noexcept;
} // namespace mfq::cpu
