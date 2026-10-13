#pragma once

// Compact NPQ/NVQ inference kernels.
//
// The persistent GPU representation keeps every variable-width stream packed:
//   NVQ1-L: 11-bit ternary codebook indices, 1-bit group delta signs
//   NVQ2: 8-bit E8 indices, 7-bit even-parity signs per 8 weights
//   NVQ3: 8-bit D4 indices per 4 weights, 7-bit even-parity signs per 8 weights
// All profiles share one fp32 neuron anchor and one packed relative scale per
// gs=24 group. Decode/small-batch matmul quantizes activations per group and
// performs signed int8 dot products with __dp4a or integer MMA. The medium-M
// path expands only one 64x96 fp16 weight tile at a time for Tensor Core GEMM.

#include "native/env.h"

#include "nvq_e8_dense_group.cuh"
#include "nvq_d4_dense_group.cuh"
#include <cuda_runtime.h>
#include "csrc/backends/cuda/native/tensor_backend.h"
#include <cuda_fp16.h>
#include <mma.h>
#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <map>
#include <type_traits>

#if defined(MFQ_NVQ_BACKWARD)
#include "packed_backward.cuh"
#endif
#include "packed_nvq.cuh"
namespace mfq::cuda::packed_nvq {
__host__ __device__ constexpr bool dense_nvq_format(int format) {
    return format==5 || format==10 || format==11 || format==12 ||
        format==13 || format==14 || format==15;
}
struct DenseNvqGroup {
    uint32_t lo=0,mid=0,hi=0;
    __device__ uint32_t state() const {return lo&15u;}
};
template<int Format>
__device__ __forceinline__ DenseNvqGroup load_dense_nvq_group(
        const uint8_t* stream,int64_t bytes,int row,int group,int pairs,int groups) {
    static_assert(dense_nvq_format(Format));
    constexpr int bits=format_index_bits(Format);
    if constexpr(is_d4_format(Format)) {
        const auto r=load_dense_d4_group<bits>(stream,bytes,row,group,pairs,groups);
        return {r.lo,r.mid,r.hi};
    }else {
        const auto r=load_dense_e8_group<bits>(stream,bytes,row,group,pairs,groups);
        return {uint32_t(r.bits),uint32_t(r.bits>>32),0};
    }
}
template<int Format,int Segment>
__device__ __forceinline__ int2 decode_dense_nvq_vector(
        DenseNvqGroup r,const int8_t* bank,int sign_mode) {
    static_assert(dense_nvq_format(Format));
    constexpr int bits=format_index_bits(Format);
    if constexpr(is_d4_format(Format))
        return decode_dense_d4_vector<Segment>(DenseD4Group<bits>{r.lo,r.mid,r.hi},bank);
    else return decode_dense_e8_vector<Segment>(DenseE8Group<bits>{uint64_t(r.lo)|(uint64_t(r.mid)<<32)},bank,sign_mode);
}
template<int Format>
__device__ __forceinline__ int2 decode_dense_nvq_vector(
        DenseNvqGroup r,const int8_t* bank,int segment,int sign_mode) {
    if(segment==0)return decode_dense_nvq_vector<Format,0>(r,bank,sign_mode);
    if(segment==1)return decode_dense_nvq_vector<Format,1>(r,bank,sign_mode);
    return decode_dense_nvq_vector<Format,2>(r,bank,sign_mode);
}
} // namespace mfq::cuda::packed_nvq
#include "nvq_routed_reuse.cuh"
#include "async_copy.cuh"

using namespace nvcuda;

namespace {

using namespace mfq::cuda::packed_nvq;

#if defined(MFQ_NVQ_BACKWARD)
template <int FORMAT>
__global__ void nvq_dequant_kernel(
    const uint8_t * indices,
    int64_t indices_nbytes,
    const uint8_t * aux,
    int64_t aux_nbytes,
    const uint8_t * sub_scale,
    int64_t sub_scale_nbytes,
    const float * neuron_scale,
    const int8_t * codebook,
    __half * weight,
    int N,
    int K,
    int ng,
    int nvec,
    int nsign,
    int sub_bits,
    int sign_mode) {
    const int segment = blockIdx.x * blockDim.x + threadIdx.x;
    if (segment >= (K + 7) / 8) return;
    for (int row = blockIdx.y; row < N; row += gridDim.y) {
        const int k0 = segment * 8;
        const int group = k0 / kGroupSize;
        const int64_t sub_linear = static_cast<int64_t>(row) * ng + group;
        const uint32_t sub = load_nvq_state<FORMAT>(indices,indices_nbytes,sub_scale,sub_scale_nbytes,
                    sub_linear,sub_bits,ng);
        const float scale = format_scale<FORMAT>(neuron_scale[row], sub, codebook);
        auto decoded = load_nvq_vec8<FORMAT>(
            indices, indices_nbytes, aux, aux_nbytes, codebook,
            row, segment, group, ng, nvec, nsign, sign_mode, sub);
        if constexpr (FORMAT == kNvq1L) {
            decoded.values.x = nvq1_l_scale_delta4(decoded.values.x, decoded.delta);
            decoded.values.y = nvq1_l_scale_delta4(decoded.values.y, decoded.delta);
        } else if constexpr (FORMAT == kNvq1S) {
            decoded.values.x = nvq1_s_scale_delta4(decoded.values.x, decoded.delta);
            decoded.values.y = nvq1_s_scale_delta4(decoded.values.y, decoded.delta);
        }
        half values[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int packed = i < 4 ? decoded.values.x : decoded.values.y;
            const int value = int(int8_t((packed >> (8 * (i & 3))) & 0xff));
            values[i] = __float2half(scale * float(value));
        }
        if ((K & 7) == 0) {
            reinterpret_cast<uint4*>(weight + int64_t(row) * K)[segment] = make_uint4(
                (uint32_t(__half_as_ushort(values[0])) | (uint32_t(__half_as_ushort(values[1])) << 16)),
                (uint32_t(__half_as_ushort(values[2])) | (uint32_t(__half_as_ushort(values[3])) << 16)),
                (uint32_t(__half_as_ushort(values[4])) | (uint32_t(__half_as_ushort(values[5])) << 16)),
                (uint32_t(__half_as_ushort(values[6])) | (uint32_t(__half_as_ushort(values[7])) << 16)));
        } else {
#pragma unroll
            for (int i = 0; i < 8; ++i)
                if (k0 + i < K) weight[int64_t(row) * K + k0 + i] = values[i];
        }
    }
}

#endif

#if defined(MFQ_NVQ_DENSE) || defined(MFQ_NVQ_MOE)
__global__ void nvq_quantize_x_gs24_kernel(
    const __half * x,
    int8_t * qx,
    float * xscale,
    int M,
    int K,
    int ng) {
    const int m = blockIdx.x;
    const int group = blockIdx.y;
    const int lane = threadIdx.x;
    if (group >= ng) return;
    const int k = group * kGroupSize + lane;
    const bool valid = lane < kGroupSize && k < K;
    const float value = valid ? __half2float(x[static_cast<int64_t>(m) * K + k]) : 0.0f;
    float maximum = fabsf(value);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
    }
    const float scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
    int q = 0;
    if (valid) {
        q = static_cast<int>(roundf(value / scale));
        q = q < -127 ? -127 : (q > 127 ? 127 : q);
    }
    if (lane == 0) xscale[static_cast<int64_t>(m) * ng + group] = scale;
    if (lane < kGroupSize) {
        qx[(static_cast<int64_t>(m) * ng + group) * kGroupSize + lane] =
            valid ? static_cast<int8_t>(q) : static_cast<int8_t>(0);
    }
}

#endif

#if defined(MFQ_NVQ_DENSE)
template <int MODE>
__global__ void nvq_quantize_x_gate_gs24_kernel(
    const __half * x,
    const __half * gate,
    int8_t * qx,
    float * xscale,
    int M,
    int K,
    int ng) {
    const int m = blockIdx.x;
    const int group = blockIdx.y;
    const int lane = threadIdx.x;
    const int k = group * kGroupSize + lane;
    const bool valid = lane < kGroupSize && k < K;
    float value = 0.0f;
    if (valid) {
        const int64_t offset = static_cast<int64_t>(m) * K + k;
        const float gate_value = __half2float(gate[offset]);
        const float sigmoid = 1.0f / (1.0f + expf(-gate_value));
        const float multiplier = MODE == 1 ? sigmoid : gate_value * sigmoid;
        value = __half2float(x[offset]) * multiplier;
    }
    float maximum = fabsf(value);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
    }
    const float scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
    int q = 0;
    if (valid) {
        q = static_cast<int>(roundf(value / scale));
        q = q < -127 ? -127 : (q > 127 ? 127 : q);
    }
    if (lane == 0) xscale[static_cast<int64_t>(m) * ng + group] = scale;
    if (lane < kGroupSize) {
        qx[(static_cast<int64_t>(m) * ng + group) * kGroupSize + lane] =
            valid ? static_cast<int8_t>(q) : static_cast<int8_t>(0);
    }
}

template <int FORMAT>
__device__ __forceinline__ float nvq_vec8_dot(
    const uint8_t * indices,
    int64_t indices_nbytes,
    const uint8_t * aux,
    int64_t aux_nbytes,
    const int8_t * codebook,
    const int8_t * activation,
    int row,
    int segment,
    int group,
    int ng,
    int nvec,
    int nsign,
    int sign_mode,
    uint32_t state) {
    return dot_nvq_vec8<FORMAT>(
        load_nvq_vec8<FORMAT>(
            indices, indices_nbytes, aux, aux_nbytes, codebook,
            row, segment, group, ng, nvec, nsign, sign_mode, state),
        activation);
}

template <int FORMAT>
__device__ __forceinline__ float nvq_vec8_scaled_fma(
    const NvqDeviceWeight & weight,
    const int8_t * activation,
    const float * activation_scale,
    int row,
    int segment,
    float acc) {
    const int group = segment / 3;
    const int k = group * kGroupSize + (segment - group * 3) * 8;
    const int64_t sub_linear = static_cast<int64_t>(row) * weight.ng + group;
    const uint32_t sub = load_nvq_state<FORMAT>(weight.indices,weight.indices_nbytes,
        weight.sub_scale,weight.sub_scale_nbytes,sub_linear,weight.sub_bits,weight.ng);
    const float dot = nvq_vec8_dot<FORMAT>(
        weight.indices, weight.indices_nbytes,
        weight.aux, weight.aux_nbytes,
        weight.codebook, activation + k,
        row, segment, group, weight.ng, weight.nvec, weight.nsign,
        weight.sign_mode, sub);
    const float weight_scale = (FORMAT == kNvq1L || FORMAT == kNvq1S)
        ? weight.neuron_scale[row] * static_cast<float>(sub)
        : format_scale<FORMAT>(weight.neuron_scale[row], sub, weight.codebook);
    return fmaf(weight_scale * activation_scale[group], dot, acc);
}

template <bool EXEC_LAYOUT = false, bool JSC = false>
__device__ __forceinline__ float nvq2_vec8_scaled_fma_s4(
    const NvqDeviceWeight & weight,
    const int8_t * activation,
    const float * activation_scale,
    int64_t index_row,
    int64_t sign_row,
    int64_t sub_row,
    float anchor,
    int segment,
    float acc) {
    const int group = segment / 3;
    const int k = group * kGroupSize + (segment - group * 3) * 8;
    uint32_t index;
    uint32_t mask8;
    if constexpr (EXEC_LAYOUT) {
        const uint16_t metadata =
            reinterpret_cast<const uint16_t *>(weight.indices)[index_row + segment];
        index = metadata & 0xffu;
        mask8 = metadata >> 8;
    } else {
        index = weight.indices[index_row + segment];
        const int64_t sign_linear = sign_row + segment;
        const int64_t sign_bit = sign_linear * 7;
        const int64_t sign_byte = sign_bit >> 3;
        const int sign_shift = static_cast<int>(sign_bit & 7);
        uint32_t sign_word = weight.aux[sign_byte];
        if (sign_byte + 1 < weight.aux_nbytes) {
            sign_word |= static_cast<uint32_t>(weight.aux[sign_byte + 1]) << 8;
        }
        const uint32_t mask7 = (sign_word >> sign_shift) & 0x7fu;
        const int last = parity7(mask7) ^
            (weight.sign_mode ? ((index >> 7) & 1u) : 0u);
        mask8 = mask7 | (static_cast<uint32_t>(last) << 7);
    }
    const uint32_t sub = load_packed_4(weight.sub_scale, sub_row + group);
    const int8_t * bank = JSC
        ? active_codebook<kNvq2Jsc>(weight.codebook, sub)
        : weight.codebook;
    const int2 code = reinterpret_cast<const int2 *>(bank)[index];
    const int2 values = EXEC_LAYOUT ? apply_sign8_exec(code, mask8) : apply_sign8(code, mask8);
    const int2 x = *reinterpret_cast<const int2 *>(activation + k);
    const int dot = __dp4a(values.y, x.y, __dp4a(values.x, x.x, 0));
    const float weight_scale = JSC
        ? format_scale<kNvq2Jsc>(anchor, sub, weight.codebook)
        : anchor * static_cast<float>(sub);
    return fmaf(weight_scale * activation_scale[group], static_cast<float>(dot), acc);
}

__device__ __forceinline__ uint32_t load_nvq_sign7(
    const uint8_t * aux,
    int64_t aux_nbytes,
    int64_t sign_linear) {
    const int64_t sign_bit = sign_linear * 7;
    const int64_t sign_byte = sign_bit >> 3;
    const int sign_shift = static_cast<int>(sign_bit & 7);
    uint32_t word = aux[sign_byte];
    if (sign_byte + 1 < aux_nbytes) {
        word |= static_cast<uint32_t>(aux[sign_byte + 1]) << 8;
    }
    return (word >> sign_shift) & 0x7fu;
}

template <bool JSC = false, bool ANALYTIC2 = false>
__device__ __forceinline__ float nvq3_vec8_scaled_fma_s4(
    const NvqDeviceWeight & weight,
    const int8_t * activation,
    const float * activation_scale,
    int64_t index_row,
    int64_t sign_row,
    int64_t sub_row,
    float anchor,
    int segment,
    float acc) {
    const int group = segment / 3;
    const int k = group * kGroupSize + (segment - group * 3) * 8;
    const int vector4 = segment * 2;
    const uint32_t index0 = weight.indices[index_row + vector4];
    const uint32_t index1 = vector4 + 1 < weight.nvec
        ? weight.indices[index_row + vector4 + 1]
        : 0;
    const uint32_t mask7 = load_nvq_sign7(weight.aux, weight.aux_nbytes, sign_row + segment);
    const uint32_t mask8 = mask7 | (static_cast<uint32_t>(parity7(mask7)) << 7);
    const uint32_t sub = load_packed_4(weight.sub_scale, sub_row + group);
    const uint8_t * metadata = reinterpret_cast<const uint8_t *>(weight.codebook);
    const int8_t * bank = weight.codebook;
    if constexpr (JSC) {
        const uint32_t bank_id = ANALYTIC2
            ? sub & 1u
            : metadata[kJscBankMapOffset + sub];
        bank = weight.codebook + kJscMetadataBytes + bank_id * kJscD4CodebookBytes;
    }
    const int2 values = apply_sign8(
        make_int2(
            reinterpret_cast<const int *>(bank)[index0],
            reinterpret_cast<const int *>(bank)[index1]),
        mask8);
    const int2 x = *reinterpret_cast<const int2 *>(activation + k);
    const int dot = __dp4a(values.y, x.y, __dp4a(values.x, x.x, 0));
    float weight_scale;
    if constexpr (ANALYTIC2) {
        weight_scale = anchor * static_cast<float>((sub >> 1) + 1);
    } else if constexpr (JSC) {
        const __half * lut = reinterpret_cast<const __half *>(metadata + kJscLutOffset);
        weight_scale = anchor * __half2float(lut[sub]);
    } else {
        weight_scale = anchor * static_cast<float>(sub);
    }
    return fmaf(weight_scale * activation_scale[group], static_cast<float>(dot), acc);
}

template <bool EXEC_LAYOUT = false, bool JSC = false>
__device__ __forceinline__ void nvq2_vec8_pair_scaled_fma_s4(
    const NvqDeviceWeight & gate,
    const NvqDeviceWeight & up,
    const int8_t * activation,
    const float * activation_scale,
    int64_t index_row,
    int64_t sign_row,
    int64_t sub_row,
    float gate_anchor,
    float up_anchor,
    int segment,
    float & gate_acc,
    float & up_acc) {
    const int group = segment / 3;
    const int k = group * kGroupSize + (segment - group * 3) * 8;
    const int2 x = *reinterpret_cast<const int2 *>(activation + k);
    const float x_scale = activation_scale[group];
    uint32_t gate_index;
    uint32_t up_index;
    uint32_t gate_mask8;
    uint32_t up_mask8;
    if constexpr (EXEC_LAYOUT) {
        const uint16_t gate_metadata =
            reinterpret_cast<const uint16_t *>(gate.indices)[index_row + segment];
        const uint16_t up_metadata =
            reinterpret_cast<const uint16_t *>(up.indices)[index_row + segment];
        gate_index = gate_metadata & 0xffu;
        up_index = up_metadata & 0xffu;
        gate_mask8 = gate_metadata >> 8;
        up_mask8 = up_metadata >> 8;
    } else {
        const int64_t sign_linear = sign_row + segment;
        const int64_t sign_bit = sign_linear * 7;
        const int64_t sign_byte = sign_bit >> 3;
        const int sign_shift = static_cast<int>(sign_bit & 7);
        uint32_t gate_sign_word = gate.aux[sign_byte];
        uint32_t up_sign_word = up.aux[sign_byte];
        if (sign_byte + 1 < gate.aux_nbytes) {
            gate_sign_word |= static_cast<uint32_t>(gate.aux[sign_byte + 1]) << 8;
            up_sign_word |= static_cast<uint32_t>(up.aux[sign_byte + 1]) << 8;
        }
        const uint32_t gate_mask7 = (gate_sign_word >> sign_shift) & 0x7fu;
        const uint32_t up_mask7 = (up_sign_word >> sign_shift) & 0x7fu;
        gate_index = gate.indices[index_row + segment];
        up_index = up.indices[index_row + segment];
        const int gate_last = parity7(gate_mask7) ^
            (gate.sign_mode ? ((gate_index >> 7) & 1u) : 0u);
        const int up_last = parity7(up_mask7) ^
            (up.sign_mode ? ((up_index >> 7) & 1u) : 0u);
        gate_mask8 = gate_mask7 | (static_cast<uint32_t>(gate_last) << 7);
        up_mask8 = up_mask7 | (static_cast<uint32_t>(up_last) << 7);
    }
    const uint32_t gate_sub = load_packed_4(gate.sub_scale, sub_row + group);
    const uint32_t up_sub = load_packed_4(up.sub_scale, sub_row + group);
    const int8_t * gate_bank = JSC
        ? active_codebook<kNvq2Jsc>(gate.codebook, gate_sub)
        : gate.codebook;
    const int8_t * up_bank = JSC
        ? active_codebook<kNvq2Jsc>(up.codebook, up_sub)
        : up.codebook;
    const int2 gate_code = reinterpret_cast<const int2 *>(gate_bank)[gate_index];
    const int2 up_code = reinterpret_cast<const int2 *>(up_bank)[up_index];
    const int2 gate_values = EXEC_LAYOUT
        ? apply_sign8_exec(gate_code, gate_mask8)
        : apply_sign8(gate_code, gate_mask8);
    const int2 up_values = EXEC_LAYOUT
        ? apply_sign8_exec(up_code, up_mask8)
        : apply_sign8(up_code, up_mask8);
    const int gate_dot = __dp4a(
        gate_values.y, x.y, __dp4a(gate_values.x, x.x, 0));
    const int up_dot = __dp4a(
        up_values.y, x.y, __dp4a(up_values.x, x.x, 0));
    const float gate_scale = JSC
        ? format_scale<kNvq2Jsc>(gate_anchor, gate_sub, gate.codebook)
        : gate_anchor * static_cast<float>(gate_sub);
    const float up_scale = JSC
        ? format_scale<kNvq2Jsc>(up_anchor, up_sub, up.codebook)
        : up_anchor * static_cast<float>(up_sub);
    gate_acc = fmaf(gate_scale * x_scale, static_cast<float>(gate_dot), gate_acc);
    up_acc = fmaf(up_scale * x_scale, static_cast<float>(up_dot), up_acc);
}

// llama.cpp-style MMVQ schedule: one output row per block, multiple warps
// split K, and each lane consumes one complete 8-value VQ segment.
template <int FORMAT, int NWARPS, bool FAST_S4 = false>
__global__ void __launch_bounds__(NWARPS * 32, 1) nvq_gemv_m1_vec8_kernel(
    const uint8_t * indices,
    int64_t indices_nbytes,
    const uint8_t * aux,
    int64_t aux_nbytes,
    const uint8_t * sub_scale,
    int64_t sub_scale_nbytes,
    const float * neuron_scale,
    const int8_t * codebook,
    const int8_t * qx,
    const float * xscale,
    __half * output,
    int N,
    int ng,
    int nvec,
    int nsign,
    int sub_bits,
    int sign_mode) {
    const int row = blockIdx.x;
    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    if (row >= N) return;

    float acc = 0.0f;
    [[maybe_unused]] const NvqDeviceWeight fast_weight{
        indices, indices_nbytes, aux, aux_nbytes,
        sub_scale, sub_scale_nbytes, neuron_scale, codebook,
        0, N, ng, nvec, nsign, 4, sign_mode};
    [[maybe_unused]] const int64_t fast_index_row = static_cast<int64_t>(row) * nvec;
    [[maybe_unused]] const int64_t fast_sign_row = static_cast<int64_t>(row) * nsign;
    [[maybe_unused]] const int64_t fast_sub_row = static_cast<int64_t>(row) * ng;
    [[maybe_unused]] const float fast_anchor = neuron_scale[row];
    for (int segment = warp * 32 + lane;
         segment < nsign;
         segment += NWARPS * 32) {
        if constexpr (FAST_S4) {
            if constexpr (
                FORMAT == kNvq2 || FORMAT == kNvq2Exec ||
                FORMAT == kNvq2Jsc || FORMAT == kNvq2JscExec) {
                acc = nvq2_vec8_scaled_fma_s4<
                    FORMAT == kNvq2Exec || FORMAT == kNvq2JscExec,
                    FORMAT == kNvq2Jsc || FORMAT == kNvq2JscExec>(
                    fast_weight, qx, xscale,
                    fast_index_row, fast_sign_row, fast_sub_row, fast_anchor,
                    segment, acc);
            } else {
                acc = nvq3_vec8_scaled_fma_s4<
                    FORMAT == kNvq3Jsc || FORMAT == kNvq3Jsc2,
                    FORMAT == kNvq3Jsc2>(
                    fast_weight, qx, xscale,
                    fast_index_row, fast_sign_row, fast_sub_row, fast_anchor,
                    segment, acc);
            }
        } else {
            const int group = segment / 3;
            const int segment_in_group = segment - group * 3;
            const int k = group * kGroupSize + segment_in_group * 8;
            const int64_t sub_linear = static_cast<int64_t>(row) * ng + group;
            const uint32_t sub = load_nvq_state<FORMAT>(indices,indices_nbytes,sub_scale,sub_scale_nbytes,
                    sub_linear,sub_bits,ng);
            const float dot = nvq_vec8_dot<FORMAT>(
                indices, indices_nbytes, aux, aux_nbytes, codebook, qx + k,
                row, segment, group, ng, nvec, nsign, sign_mode, sub);
            const float weight_scale = (FORMAT == kNvq1L || FORMAT == kNvq1S)
                ? neuron_scale[row] * static_cast<float>(sub)
                : format_scale<FORMAT>(neuron_scale[row], sub, codebook);
            acc = fmaf(weight_scale * xscale[group], dot, acc);
        }
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, offset);
    }
    __shared__ float partial[NWARPS];
    if (lane == 0) partial[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        acc = lane < NWARPS ? partial[lane] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_xor_sync(0xffffffff, acc, offset);
        }
        if (lane == 0) output[row] = __float2half(acc);
    }
}

template <int NWARPS>
__global__ void __launch_bounds__(NWARPS * 32, 1) nvq3j2_gemv_m1_group_kernel(
    const uint8_t * indices,
    const uint8_t * aux,
    int64_t aux_nbytes,
    const uint8_t * sub_scale,
    const float * neuron_scale,
    const int8_t * metadata,
    const int8_t * qx,
    const float * xscale,
    __half * output,
    int N,
    int ng,
    int nvec,
    int nsign) {
    const int row = blockIdx.x;
    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    if (row >= N) return;

    const int64_t index_row = static_cast<int64_t>(row) * nvec;
    const int64_t sign_row = static_cast<int64_t>(row) * nsign;
    const int64_t sub_row = static_cast<int64_t>(row) * ng;
    const float anchor = neuron_scale[row];
    float acc = 0.0f;
    for (int group = warp * 32 + lane; group < ng; group += NWARPS * 32) {
        const uint32_t state = load_packed_4(sub_scale, sub_row + group);
        const int8_t * bank = metadata + kJscMetadataBytes
            + (state & 1u) * kJscD4CodebookBytes;
        int group_dot = 0;
#pragma unroll
        for (int local = 0; local < 3; ++local) {
            const int segment = group * 3 + local;
            if (segment >= nsign) continue;
            const int vector4 = segment * 2;
            const uint32_t index0 = indices[index_row + vector4];
            const uint32_t index1 = vector4 + 1 < nvec
                ? indices[index_row + vector4 + 1]
                : 0;
            const uint32_t mask7 = load_nvq_sign7(aux, aux_nbytes, sign_row + segment);
            const uint32_t mask8 = mask7
                | (static_cast<uint32_t>(parity7(mask7)) << 7);
            const int2 values = apply_sign8(
                make_int2(
                    reinterpret_cast<const int *>(bank)[index0],
                    reinterpret_cast<const int *>(bank)[index1]),
                mask8);
            const int2 x = *reinterpret_cast<const int2 *>(
                qx + group * kGroupSize + local * 8);
            group_dot = __dp4a(
                values.y, x.y, __dp4a(values.x, x.x, group_dot));
        }
        const float weight_scale = anchor * static_cast<float>((state >> 1) + 1);
        acc = fmaf(
            weight_scale * xscale[group], static_cast<float>(group_dot), acc);
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, offset);
    }
    __shared__ float partial[NWARPS];
    if (lane == 0) partial[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        acc = lane < NWARPS ? partial[lane] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_xor_sync(0xffffffff, acc, offset);
        }
        if (lane == 0) output[row] = __float2half(acc);
    }
}

__device__ __forceinline__ uint32_t extract_group_exec96_bits(
    const uint32_t (&words)[3], int bit, int bits) {
    return extract_group_exec96_bits_ptr(words, bit, bits);
}

__device__ __forceinline__ int aligned_group_d4_dot_words(
    const NvqDeviceWeight & weight,
    int group,
    const int8_t * qx,
    const uint32_t (&words)[3],
    uint32_t & state) {
    state = extract_group_exec96_bits(words, 84, 4);
    const int8_t * bank = active_codebook<kNvq3JscLGroupExec>(
        weight.codebook, state);
    const uint32_t index00 = extract_group_exec96_bits(words, 0, 10);
    const uint32_t index01 = extract_group_exec96_bits(words, 10, 10);
    const uint32_t index10 = extract_group_exec96_bits(words, 20, 10);
    const uint32_t index11 = extract_group_exec96_bits(words, 30, 10);
    const uint32_t index20 = extract_group_exec96_bits(words, 40, 10);
    const uint32_t index21 = extract_group_exec96_bits(words, 50, 10);
    const uint32_t mask0 = extract_group_exec96_bits(words, 60, 8);
    const uint32_t mask1 = extract_group_exec96_bits(words, 68, 8);
    const uint32_t mask2 = extract_group_exec96_bits(words, 76, 8);
    const int2 values0 = apply_sign8(
        make_int2(
            reinterpret_cast<const int *>(bank)[index00],
            reinterpret_cast<const int *>(bank)[index01]),
        mask0);
    const int2 values1 = apply_sign8(
        make_int2(
            reinterpret_cast<const int *>(bank)[index10],
            reinterpret_cast<const int *>(bank)[index11]),
        mask1);
    const int2 values2 = apply_sign8(
        make_int2(
            reinterpret_cast<const int *>(bank)[index20],
            reinterpret_cast<const int *>(bank)[index21]),
        mask2);
    const int8_t * activation = qx + group * kGroupSize;
    const int2 x0 = *reinterpret_cast<const int2 *>(activation);
    const int2 x1 = *reinterpret_cast<const int2 *>(activation + 8);
    const int2 x2 = *reinterpret_cast<const int2 *>(activation + 16);
    const int dot0 = __dp4a(values0.y, x0.y, __dp4a(values0.x, x0.x, 0));
    const int dot1 = __dp4a(values1.y, x1.y, __dp4a(values1.x, x1.x, 0));
    const int dot2 = __dp4a(values2.y, x2.y, __dp4a(values2.x, x2.x, 0));
    return dot0 + dot1 + dot2;
}

template <int FORMAT>
__device__ __forceinline__ int aligned_group_dot(
    const NvqDeviceWeight & weight,
    int row,
    int group,
    const int8_t * qx,
    uint32_t & state) {
    static_assert(
        FORMAT == kNvq2JscXLGroupExec ||
        FORMAT == kNvq3JscLGroupExec);
    if constexpr (FORMAT == kNvq2JscXLGroupExec) {
        const uint64_t metadata = load_group_exec64(
            weight.indices, row, group, weight.ng);
        state = static_cast<uint32_t>(metadata >> 60);
        const int8_t * bank = active_codebook<FORMAT>(weight.codebook, state);
        int dot = 0;
#pragma unroll
        for (int local = 0; local < 3; ++local) {
            const int segment = group * 3 + local;
            if (segment >= weight.nsign) continue;
            const uint32_t segment_metadata =
                (metadata >> (local * 20)) & 0xfffffu;
            const uint32_t index = segment_metadata & 0xfffu;
            const uint32_t mask8 = segment_metadata >> 12;
            const int2 values = apply_sign8(
                reinterpret_cast<const int2 *>(bank)[index], mask8);
            const int2 x = *reinterpret_cast<const int2 *>(
                qx + group * kGroupSize + local * 8);
            dot = __dp4a(values.y, x.y, __dp4a(values.x, x.x, dot));
        }
        return dot;
    } else {
        uint32_t words[3];
        load_group_exec96_words(
            weight.indices, row, group, weight.ng, words);
        return aligned_group_d4_dot_words(weight, group, qx, words, state);
    }
}

template <int FORMAT>
__device__ __forceinline__ void aligned_group_pair_dot(
    const NvqDeviceWeight & gate,
    const NvqDeviceWeight & up,
    int row,
    int group,
    const int8_t * qx,
    uint32_t & gate_state,
    uint32_t & up_state,
    int & gate_dot,
    int & up_dot) {
    static_assert(
        FORMAT == kNvq2JscXLGroupExec ||
        FORMAT == kNvq3JscLGroupExec);
    gate_dot = 0;
    up_dot = 0;
    if constexpr (FORMAT == kNvq2JscXLGroupExec) {
        const uint64_t gate_metadata = load_group_exec64(
            gate.indices, row, group, gate.ng);
        const uint64_t up_metadata = load_group_exec64(
            up.indices, row, group, up.ng);
        gate_state = static_cast<uint32_t>(gate_metadata >> 60);
        up_state = static_cast<uint32_t>(up_metadata >> 60);
        const int8_t * gate_bank = active_codebook<FORMAT>(
            gate.codebook, gate_state);
        const int8_t * up_bank = active_codebook<FORMAT>(
            up.codebook, up_state);
#pragma unroll
        for (int local = 0; local < 3; ++local) {
            const int segment = group * 3 + local;
            if (segment >= gate.nsign) continue;
            const uint32_t gate_segment =
                (gate_metadata >> (local * 20)) & 0xfffffu;
            const uint32_t up_segment =
                (up_metadata >> (local * 20)) & 0xfffffu;
            const uint32_t gate_index = gate_segment & 0xfffu;
            const uint32_t up_index = up_segment & 0xfffu;
            const uint32_t gate_mask = gate_segment >> 12;
            const uint32_t up_mask = up_segment >> 12;
            const int2 gate_values = apply_sign8(
                reinterpret_cast<const int2 *>(gate_bank)[gate_index],
                gate_mask);
            const int2 up_values = apply_sign8(
                reinterpret_cast<const int2 *>(up_bank)[up_index],
                up_mask);
            const int2 x = *reinterpret_cast<const int2 *>(
                qx + group * kGroupSize + local * 8);
            gate_dot = __dp4a(
                gate_values.y, x.y,
                __dp4a(gate_values.x, x.x, gate_dot));
            up_dot = __dp4a(
                up_values.y, x.y,
                __dp4a(up_values.x, x.x, up_dot));
        }
    } else {
        uint32_t gate_words[3];
        uint32_t up_words[3];
        load_group_exec96_words(
            gate.indices, row, group, gate.ng, gate_words);
        load_group_exec96_words(
            up.indices, row, group, up.ng, up_words);
        gate_state = extract_group_exec96_bits(gate_words, 84, 4);
        up_state = extract_group_exec96_bits(up_words, 84, 4);
        const int8_t * gate_bank = active_codebook<FORMAT>(
            gate.codebook, gate_state);
        const int8_t * up_bank = active_codebook<FORMAT>(
            up.codebook, up_state);
#pragma unroll
        for (int local = 0; local < 3; ++local) {
            const int segment = group * 3 + local;
            if (segment >= gate.nsign) continue;
            const int vector4 = segment * 2;
            const uint32_t gate_index0 = extract_group_exec96_bits(
                gate_words, local * 20, 10);
            const uint32_t gate_index1 = vector4 + 1 < gate.nvec
                ? extract_group_exec96_bits(
                    gate_words, local * 20 + 10, 10)
                : 0;
            const uint32_t up_index0 = extract_group_exec96_bits(
                up_words, local * 20, 10);
            const uint32_t up_index1 = vector4 + 1 < up.nvec
                ? extract_group_exec96_bits(
                    up_words, local * 20 + 10, 10)
                : 0;
            const uint32_t gate_mask = extract_group_exec96_bits(
                gate_words, 60 + local * 8, 8);
            const uint32_t up_mask = extract_group_exec96_bits(
                up_words, 60 + local * 8, 8);
            const int2 gate_values = apply_sign8(
                make_int2(
                    reinterpret_cast<const int *>(gate_bank)[gate_index0],
                    reinterpret_cast<const int *>(gate_bank)[gate_index1]),
                gate_mask);
            const int2 up_values = apply_sign8(
                make_int2(
                    reinterpret_cast<const int *>(up_bank)[up_index0],
                    reinterpret_cast<const int *>(up_bank)[up_index1]),
                up_mask);
            const int2 x = *reinterpret_cast<const int2 *>(
                qx + group * kGroupSize + local * 8);
            gate_dot = __dp4a(
                gate_values.y, x.y,
                __dp4a(gate_values.x, x.x, gate_dot));
            up_dot = __dp4a(
                up_values.y, x.y,
                __dp4a(up_values.x, x.x, up_dot));
        }
    }
}

template <
    int FORMAT, int NWARPS, int ROWS_PER_BLOCK = 1,
    bool ADD_RESIDUAL = false>
__global__ void __launch_bounds__(NWARPS * ROWS_PER_BLOCK * 32, 1)
aligned_group_gemv_m1_kernel(
    NvqDeviceWeight weight,
    const int8_t * qx,
    const float * xscale,
    __half * output,
    const __half * residual) {
    const int row_slot = threadIdx.y / NWARPS;
    const int warp = threadIdx.y - row_slot * NWARPS;
    const int row = blockIdx.x * ROWS_PER_BLOCK + row_slot;
    const int lane = threadIdx.x;
    const bool valid = row < weight.N;

    const float anchor = valid ? weight.neuron_scale[row] : 0.0f;
    float acc = 0.0f;
    if (valid) {
        for (int group = warp * 32 + lane;
             group < weight.ng;
             group += NWARPS * 32) {
            uint32_t state = 0;
            const int dot = aligned_group_dot<FORMAT>(
                weight, row, group, qx, state);
            acc = fmaf(
                format_scale<FORMAT>(anchor, state, weight.codebook) *
                    xscale[group],
                static_cast<float>(dot), acc);
        }
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, offset);
    }
    __shared__ float partial[ROWS_PER_BLOCK][NWARPS];
    if (lane == 0) partial[row_slot][warp] = acc;
    __syncthreads();
    if (warp == 0) {
        acc = lane < NWARPS ? partial[row_slot][lane] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_xor_sync(0xffffffff, acc, offset);
        }
        if (lane == 0 && valid) {
            const __half projected = __float2half(acc);
            if constexpr (ADD_RESIDUAL) {
                output[row] = __hadd(residual[row], projected);
            } else {
                output[row] = projected;
            }
        }
    }
}

template <int FORMAT, int NWARPS, int ROWS_PER_BLOCK = 1>
__global__ void __launch_bounds__(NWARPS * ROWS_PER_BLOCK * 32, 1)
aligned_group_multi2_m1_kernel(
    NvqDeviceWeight first,
    NvqDeviceWeight second,
    const int8_t * qx,
    const float * xscale,
    __half * output) {
    const int row_slot = threadIdx.y / NWARPS;
    const int warp = threadIdx.y - row_slot * NWARPS;
    const int combined_row = blockIdx.x * ROWS_PER_BLOCK + row_slot;
    const int total_rows = first.N + second.N;
    const bool valid = combined_row < total_rows;
    const bool use_second = combined_row >= first.N;
    const NvqDeviceWeight weight = use_second ? second : first;
    const int row = use_second ? combined_row - first.N : combined_row;
    const int lane = threadIdx.x;

    const float anchor = valid ? weight.neuron_scale[row] : 0.0f;
    float acc = 0.0f;
    if (valid) {
        for (int group = warp * 32 + lane;
             group < weight.ng;
             group += NWARPS * 32) {
            uint32_t state = 0;
            const int dot = aligned_group_dot<FORMAT>(weight, row, group, qx, state);
            acc = fmaf(
                format_scale<FORMAT>(anchor, state, weight.codebook) * xscale[group],
                static_cast<float>(dot), acc);
        }
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, offset);
    }
    __shared__ float partial[ROWS_PER_BLOCK][NWARPS];
    if (lane == 0) partial[row_slot][warp] = acc;
    __syncthreads();
    if (warp == 0) {
        acc = lane < NWARPS ? partial[row_slot][lane] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_xor_sync(0xffffffff, acc, offset);
        }
        if (lane == 0 && valid) output[combined_row] = __float2half(acc);
    }
}

template <int FORMAT>
__device__ __forceinline__ int aligned_group_dot_preloaded(
    const NvqDeviceWeight & weight,
    const int8_t * codebook,
    int row,
    int group,
    const int2 & x0,
    const int2 & x1,
    const int2 & x2,
    uint32_t & state) {
    static_assert(
        FORMAT == kNvq2JscXLGroupExec ||
        FORMAT == kNvq3JscLGroupExec);
    if constexpr (FORMAT == kNvq2JscXLGroupExec) {
        const uint64_t metadata = load_group_exec64(
            weight.indices, row, group, weight.ng);
        state = static_cast<uint32_t>(metadata >> 60);
        const int8_t * bank = active_codebook<FORMAT>(codebook, state);
        const uint32_t segment0 = static_cast<uint32_t>(metadata) & 0xfffffu;
        const uint32_t segment1 = static_cast<uint32_t>(metadata >> 20) & 0xfffffu;
        const uint32_t segment2 = static_cast<uint32_t>(metadata >> 40) & 0xfffffu;
        const int2 values0 = apply_sign8(
            reinterpret_cast<const int2 *>(bank)[segment0 & 0xfffu],
            segment0 >> 12);
        const int2 values1 = apply_sign8(
            reinterpret_cast<const int2 *>(bank)[segment1 & 0xfffu],
            segment1 >> 12);
        const int2 values2 = apply_sign8(
            reinterpret_cast<const int2 *>(bank)[segment2 & 0xfffu],
            segment2 >> 12);
        int dot = __dp4a(values0.y, x0.y, __dp4a(values0.x, x0.x, 0));
        dot = __dp4a(values1.y, x1.y, __dp4a(values1.x, x1.x, dot));
        return __dp4a(values2.y, x2.y, __dp4a(values2.x, x2.x, dot));
    } else {
        uint32_t words[3];
        load_group_exec96_words(
            weight.indices, row, group, weight.ng, words);
        state = extract_group_exec96_bits(words, 84, 4);
        const int8_t * bank = active_codebook<FORMAT>(codebook, state);
        const uint32_t index00 = extract_group_exec96_bits(words, 0, 10);
        const uint32_t index01 = extract_group_exec96_bits(words, 10, 10);
        const uint32_t index10 = extract_group_exec96_bits(words, 20, 10);
        const uint32_t index11 = extract_group_exec96_bits(words, 30, 10);
        const uint32_t index20 = extract_group_exec96_bits(words, 40, 10);
        const uint32_t index21 = extract_group_exec96_bits(words, 50, 10);
        const int2 values0 = apply_sign8(
            make_int2(
                reinterpret_cast<const int *>(bank)[index00],
                reinterpret_cast<const int *>(bank)[index01]),
            extract_group_exec96_bits(words, 60, 8));
        const int2 values1 = apply_sign8(
            make_int2(
                reinterpret_cast<const int *>(bank)[index10],
                reinterpret_cast<const int *>(bank)[index11]),
            extract_group_exec96_bits(words, 68, 8));
        const int2 values2 = apply_sign8(
            make_int2(
                reinterpret_cast<const int *>(bank)[index20],
                reinterpret_cast<const int *>(bank)[index21]),
            extract_group_exec96_bits(words, 76, 8));
        const int dot0 = __dp4a(values0.y, x0.y, __dp4a(values0.x, x0.x, 0));
        const int dot1 = __dp4a(values1.y, x1.y, __dp4a(values1.x, x1.x, 0));
        const int dot2 = __dp4a(values2.y, x2.y, __dp4a(values2.x, x2.x, 0));
        return dot0 + dot1 + dot2;
    }
}

__device__ __forceinline__ int aligned_group_dot_preloaded_e8_stage3_metadata(
    const int8_t * global_codebook,
    const int8_t * shared_codebook,
    bool table_is_staged,
    uint64_t metadata,
    const int2 & x0,
    const int2 & x1,
    const int2 & x2,
    uint32_t & state) {
    state = static_cast<uint32_t>(metadata >> 60);
    const uint32_t bank_id = state & 3u;
    const bool use_staged_bank = table_is_staged && bank_id < 3;
    const int8_t * bank = use_staged_bank
        ? shared_codebook + bank_id * kJscE8Codebook4096Bytes
        : global_codebook + kJscMetadataBytes +
            bank_id * kJscE8Codebook4096Bytes;
    const uint32_t segment0 = static_cast<uint32_t>(metadata) & 0xfffffu;
    const uint32_t segment1 = static_cast<uint32_t>(metadata >> 20) & 0xfffffu;
    const uint32_t segment2 = static_cast<uint32_t>(metadata >> 40) & 0xfffffu;
    const uint32_t index0 = segment0 & 0xfffu;
    const uint32_t index1 = segment1 & 0xfffu;
    const uint32_t index2 = segment2 & 0xfffu;
    int2 raw0;
    int2 raw1;
    int2 raw2;
    if (use_staged_bank) {
        const int * plane = reinterpret_cast<const int *>(bank);
        constexpr int plane_entries = 4096;
        raw0 = make_int2(plane[index0], plane[plane_entries + index0]);
        raw1 = make_int2(plane[index1], plane[plane_entries + index1]);
        raw2 = make_int2(plane[index2], plane[plane_entries + index2]);
    } else if (table_is_staged && bank_id == 3) {
        const int * partial = reinterpret_cast<const int *>(
            shared_codebook + 3 * kJscE8Codebook4096Bytes);
        const int2 * global = reinterpret_cast<const int2 *>(bank);
        raw0 = index0 < kJscE8PartialEntries
            ? make_int2(
                partial[index0], partial[kJscE8PartialEntries + index0])
            : global[index0];
        raw1 = index1 < kJscE8PartialEntries
            ? make_int2(
                partial[index1], partial[kJscE8PartialEntries + index1])
            : global[index1];
        raw2 = index2 < kJscE8PartialEntries
            ? make_int2(
                partial[index2], partial[kJscE8PartialEntries + index2])
            : global[index2];
    } else {
        raw0 = reinterpret_cast<const int2 *>(bank)[index0];
        raw1 = reinterpret_cast<const int2 *>(bank)[index1];
        raw2 = reinterpret_cast<const int2 *>(bank)[index2];
    }
    const int2 values0 = apply_sign8(raw0, segment0 >> 12);
    const int2 values1 = apply_sign8(raw1, segment1 >> 12);
    const int2 values2 = apply_sign8(raw2, segment2 >> 12);
    int dot = __dp4a(values0.y, x0.y, __dp4a(values0.x, x0.x, 0));
    dot = __dp4a(values1.y, x1.y, __dp4a(values1.x, x1.x, dot));
    return __dp4a(values2.y, x2.y, __dp4a(values2.x, x2.x, dot));
}

__device__ __forceinline__ int aligned_group_dot_preloaded_e8_stage3(
    const NvqDeviceWeight & weight,
    const int8_t * global_codebook,
    const int8_t * shared_codebook,
    bool table_is_staged,
    int row,
    int group,
    const int2 & x0,
    const int2 & x1,
    const int2 & x2,
    uint32_t & state) {
    const uint64_t metadata = load_group_exec64(
        weight.indices, row, group, weight.ng);
    return aligned_group_dot_preloaded_e8_stage3_metadata(
        global_codebook, shared_codebook, table_is_staged, metadata,
        x0, x1, x2, state);
}

// Decode geometry: every warp owns several output rows and reuses
// the same quantized activation values across those rows.  The compact D4
// tables fit in shared memory and are staged once per CTA; the 128 KiB E8
// tables remain in the read-only cache path.
template <
    int FORMAT, int NWARPS, int ROWS_PER_WARP, bool MULTI2,
    bool STAGE_E8_THREE_BANKS = false, bool ADD_RESIDUAL = false>
__global__ void __launch_bounds__(NWARPS * 32, 1)
aligned_group_row_tiled_m1_kernel(
    NvqDeviceWeight first,
    NvqDeviceWeight second,
    const int8_t * qx,
    const float * xscale,
    __half * output,
    const __half * residual) {
    static_assert(!ADD_RESIDUAL || !MULTI2);
    static_assert(NWARPS == 16 || NWARPS == 32);
    static_assert(
        ROWS_PER_WARP == 1 || ROWS_PER_WARP == 2 || ROWS_PER_WARP == 4);
    static_assert(
        !STAGE_E8_THREE_BANKS ||
        (FORMAT == kNvq2JscXLGroupExec && NWARPS == 32 &&
         (ROWS_PER_WARP == 1 || ROWS_PER_WARP == 4)));
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int linear_thread = warp * 32 + lane;
    const int block_threads = NWARPS * 32;
    const int k_pad = first.ng * kGroupSize;
    const int scale_offset = (k_pad + 3) & ~3;
    const int codebook_offset = scale_offset + first.ng * sizeof(float);
    extern __shared__ uint8_t shared_storage[];
    int8_t * shared_qx = reinterpret_cast<int8_t *>(shared_storage);
    float * shared_xscale = reinterpret_cast<float *>(
        shared_storage + scale_offset);

    if constexpr (!STAGE_E8_THREE_BANKS) {
        for (int item = linear_thread; item < k_pad; item += block_threads) {
            shared_qx[item] = qx[item];
        }
        for (int item = linear_thread; item < first.ng; item += block_threads) {
            shared_xscale[item] = xscale[item];
        }
    }

    const int8_t * first_codebook = first.codebook;
    const int8_t * second_codebook = second.codebook;
    const int rows_per_block = NWARPS * ROWS_PER_WARP;
    const int block_row_base = blockIdx.x * rows_per_block;
    const bool stage_second_table = MULTI2 && block_row_base >= first.N;
    const int8_t * staged_global_codebook = stage_second_table
        ? second.codebook
        : first.codebook;
    constexpr int stage_min_rows = ROWS_PER_WARP == 1 ? 4096 : 12288;
    const bool stage_active_table = STAGE_E8_THREE_BANKS &&
        (stage_second_table
            ? second.N >= stage_min_rows
            : first.N >= stage_min_rows);
    const int8_t * shared_e8_codebook = nullptr;
    if constexpr (STAGE_E8_THREE_BANKS) {
        int8_t * staged = reinterpret_cast<int8_t *>(shared_storage);
        constexpr int entries_per_bank = 4096;
        constexpr int staged_entries = 3 * entries_per_bank;
        if (stage_active_table) {
            for (int item = linear_thread;
                 item < staged_entries;
                 item += block_threads) {
                const int bank = item / entries_per_bank;
                const int entry = item - bank * entries_per_bank;
                const int2 value = reinterpret_cast<const int2 *>(
                    staged_global_codebook + kJscMetadataBytes +
                    bank * kJscE8Codebook4096Bytes)[entry];
                int * destination = reinterpret_cast<int *>(
                    staged + bank * kJscE8Codebook4096Bytes);
                destination[entry] = value.x;
                destination[entries_per_bank + entry] = value.y;
            }
            int * partial = reinterpret_cast<int *>(
                staged + 3 * kJscE8Codebook4096Bytes);
            for (int entry = linear_thread;
                 entry < kJscE8PartialEntries;
                 entry += block_threads) {
                const int2 value = reinterpret_cast<const int2 *>(
                    staged_global_codebook + kJscMetadataBytes +
                    3 * kJscE8Codebook4096Bytes)[entry];
                partial[entry] = value.x;
                partial[kJscE8PartialEntries + entry] = value.y;
            }
        }
        shared_e8_codebook = staged;
    }
    if constexpr (FORMAT == kNvq3JscLGroupExec) {
        int8_t * shared_first_codebook = reinterpret_cast<int8_t *>(
            shared_storage + codebook_offset);
        for (int item = linear_thread;
             item < first.codebook_nbytes;
             item += block_threads) {
            shared_first_codebook[item] = first.codebook[item];
        }
        first_codebook = shared_first_codebook;
        if constexpr (MULTI2) {
            int8_t * shared_second_codebook =
                shared_first_codebook + first.codebook_nbytes;
            for (int item = linear_thread;
                 item < second.codebook_nbytes;
                 item += block_threads) {
                shared_second_codebook[item] = second.codebook[item];
            }
            second_codebook = shared_second_codebook;
        }
    }
    __syncthreads();

    const int total_rows = MULTI2 ? first.N + second.N : first.N;
    const int row_base = blockIdx.x * rows_per_block + warp;
    float accumulators[ROWS_PER_WARP] = {};
    float anchors[ROWS_PER_WARP] = {};
#pragma unroll
    for (int item = 0; item < ROWS_PER_WARP; ++item) {
        const int combined_row = row_base + item * NWARPS;
        if (combined_row < total_rows) {
            if constexpr (MULTI2) {
                anchors[item] = combined_row < first.N
                    ? first.neuron_scale[combined_row]
                    : second.neuron_scale[combined_row - first.N];
            } else {
                anchors[item] = first.neuron_scale[combined_row];
            }
        }
    }

    for (int group = lane; group < first.ng; group += 32) {
        const int8_t * activation =
            (STAGE_E8_THREE_BANKS ? qx : shared_qx) + group * kGroupSize;
        const int2 x0 = *reinterpret_cast<const int2 *>(activation);
        const int2 x1 = *reinterpret_cast<const int2 *>(activation + 8);
        const int2 x2 = *reinterpret_cast<const int2 *>(activation + 16);
        const float activation_scale = STAGE_E8_THREE_BANKS
            ? xscale[group]
            : shared_xscale[group];
        if constexpr (STAGE_E8_THREE_BANKS) {
            {
                uint64_t metadata[ROWS_PER_WARP] = {};
                int rows[ROWS_PER_WARP] = {};
                bool use_second[ROWS_PER_WARP] = {};
                bool valid[ROWS_PER_WARP] = {};
#pragma unroll
                for (int item = 0; item < ROWS_PER_WARP; ++item) {
                    const int combined_row = row_base + item * NWARPS;
                    valid[item] = combined_row < total_rows;
                    use_second[item] = combined_row >= first.N;
                    rows[item] = use_second[item]
                        ? combined_row - first.N
                        : combined_row;
                    if (valid[item]) {
                        const uint8_t * indices = use_second[item]
                            ? second.indices
                            : first.indices;
                        metadata[item] = load_group_exec64(
                            indices, rows[item], group, first.ng);
                    }
                }
#pragma unroll
                for (int item = 0; item < ROWS_PER_WARP; ++item) {
                    if (!valid[item]) continue;
                    const int8_t * active_table = use_second[item]
                        ? second_codebook
                        : first_codebook;
                    uint32_t state = 0;
                    const int dot =
                        aligned_group_dot_preloaded_e8_stage3_metadata(
                            active_table, shared_e8_codebook,
                            stage_active_table &&
                                active_table == staged_global_codebook,
                            metadata[item], x0, x1, x2, state);
                    accumulators[item] = fmaf(
                        format_scale<FORMAT>(
                            anchors[item], state, active_table) *
                            activation_scale,
                        static_cast<float>(dot), accumulators[item]);
                }
            }
        } else {
#pragma unroll
            for (int item = 0; item < ROWS_PER_WARP; ++item) {
                const int combined_row = row_base + item * NWARPS;
                if (combined_row >= total_rows) continue;
                uint32_t state = 0;
                int dot = 0;
                const int8_t * active_table = first_codebook;
                if constexpr (MULTI2) {
                    if (combined_row < first.N) {
                        if constexpr (STAGE_E8_THREE_BANKS) {
                            dot = aligned_group_dot_preloaded_e8_stage3(
                                first, first_codebook, shared_e8_codebook,
                                stage_active_table &&
                                    first_codebook == staged_global_codebook,
                                combined_row, group, x0, x1, x2, state);
                        } else {
                            dot = aligned_group_dot_preloaded<FORMAT>(
                                first, first_codebook, combined_row, group,
                                x0, x1, x2, state);
                        }
                    } else {
                        const int row = combined_row - first.N;
                        active_table = second_codebook;
                        if constexpr (STAGE_E8_THREE_BANKS) {
                            dot = aligned_group_dot_preloaded_e8_stage3(
                                second, second_codebook, shared_e8_codebook,
                                stage_active_table &&
                                    second_codebook == staged_global_codebook,
                                row, group, x0, x1, x2, state);
                        } else {
                            dot = aligned_group_dot_preloaded<FORMAT>(
                                second, second_codebook, row, group,
                                x0, x1, x2, state);
                        }
                    }
                } else {
                    if constexpr (STAGE_E8_THREE_BANKS) {
                        dot = aligned_group_dot_preloaded_e8_stage3(
                            first, first_codebook, shared_e8_codebook,
                            stage_active_table,
                            combined_row, group, x0, x1, x2, state);
                    } else {
                        dot = aligned_group_dot_preloaded<FORMAT>(
                            first, first_codebook, combined_row, group,
                            x0, x1, x2, state);
                    }
                }
                accumulators[item] = fmaf(
                    format_scale<FORMAT>(anchors[item], state, active_table) *
                        activation_scale,
                    static_cast<float>(dot), accumulators[item]);
            }
        }
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
        for (int item = 0; item < ROWS_PER_WARP; ++item) {
            accumulators[item] += __shfl_xor_sync(
                0xffffffff, accumulators[item], offset);
        }
    }
    if constexpr (ADD_RESIDUAL) {
        auto * projected_rows = reinterpret_cast<__half *>(shared_storage);
        // Every warp must finish its codebook reads before this storage is
        // reused for the coalesced residual epilogue.
        __syncthreads();
        if (lane == 0) {
#pragma unroll
            for (int item = 0; item < ROWS_PER_WARP; ++item) {
                const int local_row = warp + item * NWARPS;
                const int combined_row = block_row_base + local_row;
                if (combined_row < total_rows) {
                    projected_rows[local_row] =
                        __float2half(accumulators[item]);
                }
            }
        }
        __syncthreads();
        if (linear_thread < rows_per_block) {
            const int combined_row = block_row_base + linear_thread;
            if (combined_row < total_rows) {
                output[combined_row] = __hadd(
                    residual[combined_row], projected_rows[linear_thread]);
            }
        }
    } else if (lane == 0) {
#pragma unroll
        for (int item = 0; item < ROWS_PER_WARP; ++item) {
            const int combined_row = row_base + item * NWARPS;
            if (combined_row < total_rows) {
                output[combined_row] = __float2half(accumulators[item]);
            }
        }
    }
}

template <int FORMAT, int NWARPS, bool OUTPUT_F32 = false>
__global__ void __launch_bounds__(NWARPS * 32, 1) aligned_group_swiglu_pair_kernel(
    NvqDeviceWeight gate,
    NvqDeviceWeight up,
    const int8_t * qx,
    const float * xscale,
    __half * output_h,
    float * output_f) {
    const int row = blockIdx.x;
    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    if (row >= gate.N) return;

    const float gate_anchor = gate.neuron_scale[row];
    const float up_anchor = up.neuron_scale[row];
    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    for (int group = warp * 32 + lane;
         group < gate.ng;
         group += NWARPS * 32) {
        uint32_t gate_state = 0;
        uint32_t up_state = 0;
        int gate_dot = 0;
        int up_dot = 0;
        aligned_group_pair_dot<FORMAT>(
            gate, up, row, group, qx,
            gate_state, up_state, gate_dot, up_dot);
        const float activation_scale = xscale[group];
        gate_acc = fmaf(
            format_scale<FORMAT>(gate_anchor, gate_state, gate.codebook) *
                activation_scale,
            static_cast<float>(gate_dot), gate_acc);
        up_acc = fmaf(
            format_scale<FORMAT>(up_anchor, up_state, up.codebook) *
                activation_scale,
            static_cast<float>(up_dot), up_acc);
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        gate_acc += __shfl_xor_sync(0xffffffff, gate_acc, offset);
        up_acc += __shfl_xor_sync(0xffffffff, up_acc, offset);
    }
    __shared__ float partial[2][NWARPS];
    if (lane == 0) {
        partial[0][warp] = gate_acc;
        partial[1][warp] = up_acc;
    }
    __syncthreads();
    if (warp == 0) {
        gate_acc = lane < NWARPS ? partial[0][lane] : 0.0f;
        up_acc = lane < NWARPS ? partial[1][lane] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            gate_acc += __shfl_xor_sync(0xffffffff, gate_acc, offset);
            up_acc += __shfl_xor_sync(0xffffffff, up_acc, offset);
        }
        if (lane == 0) {
            const float gate_value = __half2float(__float2half(gate_acc));
            const float up_value = __half2float(__float2half(up_acc));
            const float sigmoid = 1.0f / (1.0f + expf(-gate_value));
            const float value = up_value * (gate_value * sigmoid);
            if constexpr (OUTPUT_F32) output_f[row] = value;
            else output_h[row] = __float2half(value);
        }
    }
}

// Decode a group once; its three integer segment dots share the scale/state.
// Four warps cover a long projection, while short rows use independent warps.
template<int FORMAT, int Rows, int Warps>
__global__ void __launch_bounds__(128) nvq_gemv_group_kernel(
    NvqDeviceWeight w, const int8_t* qx, const float* xscale,
    half* output, int M) {
    const int lane = threadIdx.x, warp = threadIdx.y;
    const int row = blockIdx.x * (4 / Warps) + warp / Warps;
    const bool valid_row = row < w.N;
    float acc[Rows] = {};
    const float anchor = valid_row ? w.neuron_scale[row] : 0.0f;
    for (int group = valid_row ? (warp % Warps) * 32 + lane : w.ng;
         group < w.ng; group += Warps * 32) {
        const int64_t state_index = int64_t(row) * w.ng + group;
        const uint32_t state = load_nvq_state<FORMAT>(w.indices,w.indices_nbytes,w.sub_scale,
            w.sub_scale_nbytes,state_index,w.sub_bits,w.ng);
        const float scale = (FORMAT == kNvq1L || FORMAT == kNvq1S)
            ? anchor * float(state) : format_scale<FORMAT>(anchor, state, w.codebook);
        uint64_t group_indices = 0;
        uint32_t group_signs = 0;
        bool group_words = false;
        if constexpr (FORMAT == kNvq2JscL || FORMAT == kNvq2JscXL) {
            constexpr int bits = format_index_bits(FORMAT);
            const int64_t bit = (int64_t(row) * w.nvec + group * 3) * bits;
            const int64_t byte = (bit >> 3) & ~int64_t(3);
            // Three 10/12-bit indices always fit these two aligned words.
            // The final unpadded bytes retain the checked scalar decoder.
            group_words = (reinterpret_cast<uintptr_t>(w.indices) & 3u) == 0 && byte + 8 <= w.indices_nbytes;
            if (group_words) {
                const auto* words = reinterpret_cast<const uint32_t*>(w.indices + byte);
                group_indices = (uint64_t(words[0]) | (uint64_t(words[1]) << 32)) >> (bit & 31);
                group_signs = load_packed_sign_group3(w.aux,
                    (int64_t(row) * w.nsign + group * 3) * 7, w.aux_nbytes);
            }
        }
        float dot[Rows] = {};
#pragma unroll
        for (int segment = 0; segment < 3; ++segment) {
            const int vector = group * 3 + segment;
            NvqVec8Values<FORMAT> weight;
            if ((FORMAT == kNvq2JscL || FORMAT == kNvq2JscXL) && group_words) {
                constexpr int bits = format_index_bits(FORMAT);
                const uint32_t index = (group_indices >> (segment * bits)) & ((1u << bits) - 1u);
                const uint32_t mask7 = (group_signs >> (segment * 7)) & 127u;
                const uint32_t last = parity7(mask7) ^ (w.sign_mode ? ((index >> 7) & 1u) : 0u);
                const uint32_t mask8 = mask7 | (last << 7);
                const auto* bank = active_codebook<FORMAT>(w.codebook, state);
                weight = {apply_sign8(reinterpret_cast<const int2*>(bank)[index], mask8), 0, vector < w.nvec};
            } else {
                weight = load_nvq_vec8<FORMAT>(
                    w.indices, w.indices_nbytes, w.aux, w.aux_nbytes, w.codebook,
                    row, vector, group, w.ng, w.nvec, w.nsign, w.sign_mode, state);
            }
#pragma unroll
            for (int m = 0; m < Rows; ++m)
                if (m < M) dot[m] += dot_nvq_vec8<FORMAT>(weight,
                    qx + (int64_t(m) * w.ng + group) * kGroupSize + segment * 8);
        }
#pragma unroll
        for (int m = 0; m < Rows; ++m)
            if (m < M) acc[m] = fmaf(scale * xscale[int64_t(m) * w.ng + group], dot[m], acc[m]);
    }
#pragma unroll
    for (int m = 0; m < Rows; ++m) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            acc[m] += __shfl_xor_sync(0xffffffffu, acc[m], offset);
    }
    if constexpr (Warps == 1) {
#pragma unroll
        for (int m = 0; m < Rows; ++m)
            if (lane == 0 && valid_row && m < M)
                output[int64_t(m) * w.N + row] = __float2half(acc[m]);
    } else {
        __shared__ float partial[Rows][4];
#pragma unroll
        for (int m = 0; m < Rows; ++m)
            if (lane == 0) partial[m][warp] = acc[m];
        __syncthreads();
        if (warp == 0) {
#pragma unroll
            for (int m = 0; m < Rows; ++m) {
                float value = lane < Warps ? partial[m][lane] : 0.0f;
#pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    value += __shfl_xor_sync(0xffffffffu, value, offset);
                if (lane == 0 && valid_row && m < M)
                    output[int64_t(m) * w.N + row] = __float2half(value);
            }
        }
    }
}

template <int FORMAT, int NWARPS, int MAX_M>
__global__ void __launch_bounds__(NWARPS * 32, 1) nvq_gemv_batch_vec8_kernel(
    const uint8_t * indices,
    int64_t indices_nbytes,
    const uint8_t * aux,
    int64_t aux_nbytes,
    const uint8_t * sub_scale,
    int64_t sub_scale_nbytes,
    const float * neuron_scale,
    const int8_t * codebook,
    const int8_t * qx,
    const float * xscale,
    __half * output,
    int M,
    int N,
    int ng,
    int nvec,
    int nsign,
    int sub_bits,
    int sign_mode) {
    const int row = blockIdx.x;
    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    if (row >= N) return;

    float acc[MAX_M];
#pragma unroll
    for (int m = 0; m < MAX_M; ++m) acc[m] = 0.0f;
    for (int segment = warp * 32 + lane; segment < nsign; segment += NWARPS * 32) {
        const int group = segment / 3;
        const int segment_in_group = segment - group * 3;
        const int k = group * kGroupSize + segment_in_group * 8;
        const int64_t sub_linear = static_cast<int64_t>(row) * ng + group;
        const uint32_t sub = load_nvq_state<FORMAT>(indices,indices_nbytes,sub_scale,sub_scale_nbytes,
                    sub_linear,sub_bits,ng);
        const auto weight = load_nvq_vec8<FORMAT>(
            indices, indices_nbytes, aux, aux_nbytes, codebook,
            row, segment, group, ng, nvec, nsign, sign_mode, sub);
        const float weight_scale = (FORMAT == kNvq1L || FORMAT == kNvq1S)
            ? neuron_scale[row] * static_cast<float>(sub)
            : format_scale<FORMAT>(neuron_scale[row], sub, codebook);
#pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < M) {
                const float dot = dot_nvq_vec8<FORMAT>(
                    weight, qx + (static_cast<int64_t>(m) * ng * kGroupSize + k));
                acc[m] = fmaf(
                    weight_scale * xscale[static_cast<int64_t>(m) * ng + group], dot, acc[m]);
            }
        }
    }

#pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc[m] += __shfl_xor_sync(0xffffffff, acc[m], offset);
        }
    }
    __shared__ float partial[MAX_M][NWARPS];
    if (lane == 0) {
#pragma unroll
        for (int m = 0; m < MAX_M; ++m) partial[m][warp] = acc[m];
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            acc[m] = lane < NWARPS ? partial[m][lane] : 0.0f;
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                acc[m] += __shfl_xor_sync(0xffffffff, acc[m], offset);
            }
            if (lane == 0 && m < M) {
                output[static_cast<int64_t>(m) * N + row] = __float2half(acc[m]);
            }
        }
    }
}

// One grid covers two projections with independent packed streams/codebooks.
// The activation is quantized once before this kernel.
template <int FORMAT, int NWARPS, int MAX_M, bool FAST_NVQ2_S4 = false>
__global__ void __launch_bounds__(NWARPS * 32, 1) nvq_gemv_multi2_vec8_kernel(
    NvqDeviceWeight first,
    NvqDeviceWeight second,
    const int8_t * qx,
    const float * xscale,
    __half * output,
    int M) {
    const int combined_row = blockIdx.x;
    const bool use_second = combined_row >= first.N;
    const NvqDeviceWeight weight = use_second ? second : first;
    const int row = use_second ? combined_row - first.N : combined_row;
    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    if (row >= weight.N) return;

    float acc[MAX_M];
#pragma unroll
    for (int m = 0; m < MAX_M; ++m) acc[m] = 0.0f;
    [[maybe_unused]] const int64_t fast_index_row = static_cast<int64_t>(row) * weight.nvec;
    [[maybe_unused]] const int64_t fast_sign_row = static_cast<int64_t>(row) * weight.nsign;
    [[maybe_unused]] const int64_t fast_sub_row = static_cast<int64_t>(row) * weight.ng;
    [[maybe_unused]] const float fast_anchor = weight.neuron_scale[row];
    for (int segment = warp * 32 + lane;
         segment < weight.nsign;
         segment += NWARPS * 32) {
#pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < M) {
                const int8_t * qrow = qx + static_cast<int64_t>(m) * weight.ng * kGroupSize;
                const float * srow = xscale + static_cast<int64_t>(m) * weight.ng;
                if constexpr (FAST_NVQ2_S4) {
                    acc[m] = nvq2_vec8_scaled_fma_s4<
                        FORMAT == kNvq2Exec || FORMAT == kNvq2JscExec,
                        FORMAT == kNvq2Jsc || FORMAT == kNvq2JscExec>(
                        weight, qrow, srow,
                        fast_index_row, fast_sign_row, fast_sub_row, fast_anchor,
                        segment, acc[m]);
                } else {
                    acc[m] = nvq_vec8_scaled_fma<FORMAT>(
                        weight, qrow, srow, row, segment, acc[m]);
                }
            }
        }
    }

#pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc[m] += __shfl_xor_sync(0xffffffff, acc[m], offset);
        }
    }
    __shared__ float partial[MAX_M][NWARPS];
    if (lane == 0) {
#pragma unroll
        for (int m = 0; m < MAX_M; ++m) partial[m][warp] = acc[m];
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            acc[m] = lane < NWARPS ? partial[m][lane] : 0.0f;
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                acc[m] += __shfl_xor_sync(0xffffffff, acc[m], offset);
            }
            if (lane == 0 && m < M) {
                output[static_cast<int64_t>(m) * (first.N + second.N) + combined_row] =
                    __float2half(acc[m]);
            }
        }
    }
}

// NINT-style paired projection: each block computes matching gate/up rows and
// applies SwiGLU after the same fp16 rounding as the materialized path.
template <int FORMAT, int NWARPS, bool OUTPUT_F32 = false, bool FAST_NVQ2_S4 = false>
__global__ void __launch_bounds__(NWARPS * 32, 1) nvq_gemv_swiglu_pair_kernel(
    NvqDeviceWeight gate,
    NvqDeviceWeight up,
    const int8_t * qx,
    const float * xscale,
    __half * output_h,
    float * output_f) {
    const int row = blockIdx.x;
    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    if (row >= gate.N) return;

    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    [[maybe_unused]] const int64_t fast_index_row = static_cast<int64_t>(row) * gate.nvec;
    [[maybe_unused]] const int64_t fast_sign_row = static_cast<int64_t>(row) * gate.nsign;
    [[maybe_unused]] const int64_t fast_sub_row = static_cast<int64_t>(row) * gate.ng;
    const float fast_gate_anchor = gate.neuron_scale[row];
    const float fast_up_anchor = up.neuron_scale[row];
    for (int segment = warp * 32 + lane;
        segment < gate.nsign;
         segment += NWARPS * 32) {
        if constexpr (FAST_NVQ2_S4) {
            nvq2_vec8_pair_scaled_fma_s4<
                FORMAT == kNvq2Exec || FORMAT == kNvq2JscExec,
                FORMAT == kNvq2Jsc || FORMAT == kNvq2JscExec>(
                gate, up, qx, xscale,
                fast_index_row, fast_sign_row, fast_sub_row,
                fast_gate_anchor, fast_up_anchor, segment, gate_acc, up_acc);
        } else {
            gate_acc = nvq_vec8_scaled_fma<FORMAT>(
                gate, qx, xscale, row, segment, gate_acc);
            up_acc = nvq_vec8_scaled_fma<FORMAT>(
                up, qx, xscale, row, segment, up_acc);
        }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        gate_acc += __shfl_xor_sync(0xffffffff, gate_acc, offset);
        up_acc += __shfl_xor_sync(0xffffffff, up_acc, offset);
    }
    __shared__ float partial[2][NWARPS];
    if (lane == 0) {
        partial[0][warp] = gate_acc;
        partial[1][warp] = up_acc;
    }
    __syncthreads();
    if (warp == 0) {
        gate_acc = lane < NWARPS ? partial[0][lane] : 0.0f;
        up_acc = lane < NWARPS ? partial[1][lane] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            gate_acc += __shfl_xor_sync(0xffffffff, gate_acc, offset);
            up_acc += __shfl_xor_sync(0xffffffff, up_acc, offset);
        }
        if (lane == 0) {
            const float gate_value = __half2float(__float2half(gate_acc));
            const float up_value = __half2float(__float2half(up_acc));
            const float sigmoid = 1.0f / (1.0f + expf(-gate_value));
            const float value = up_value * (gate_value * sigmoid);
            if constexpr (OUTPUT_F32) {
                output_f[row] = value;
            } else {
                output_h[row] = __float2half(value);
            }
        }
    }
}

__device__ __forceinline__ uint32_t load_u8x4(const uint8_t * data) {
    return static_cast<uint32_t>(data[0]) |
           (static_cast<uint32_t>(data[1]) << 8) |
           (static_cast<uint32_t>(data[2]) << 16) |
           (static_cast<uint32_t>(data[3]) << 24);
}

// Decode four adjacent vectors per thread, then replay the original MMVQ FMA
// order from shared memory. This keeps the packed-load savings without changing
// the quantized activation produced for Wdown.
template <int NWARPS, bool OUTPUT_F32 = false>
__global__ void __launch_bounds__(NWARPS * 32, 1) nvq2_gemv_swiglu_vec4_ordered_kernel(
    NvqDeviceWeight gate,
    NvqDeviceWeight up,
    const int8_t * qx,
    const float * xscale,
    __half * output_h,
    float * output_f) {
    const int row = blockIdx.x;
    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    const int thread = warp * 32 + lane;
    if (row >= gate.N) return;

    extern __shared__ uint32_t metadata[];
    const int64_t index_row = static_cast<int64_t>(row) * gate.nvec;
    const int64_t sign_row = static_cast<int64_t>(row) * gate.nsign;
    const int units = gate.nsign / 4;
    for (int unit = thread; unit < units; unit += NWARPS * 32) {
        const int segment_base = unit * 4;
        const uint32_t gate_indices = *reinterpret_cast<const uint32_t *>(
            gate.indices + index_row + segment_base);
        const uint32_t up_indices = *reinterpret_cast<const uint32_t *>(
            up.indices + index_row + segment_base);
        const int64_t sign_bit = (sign_row + segment_base) * 7;
        const int64_t sign_byte = sign_bit >> 3;
        const int sign_shift = static_cast<int>(sign_bit & 7);
        const uint32_t gate_signs = load_u8x4(gate.aux + sign_byte) >> sign_shift;
        const uint32_t up_signs = load_u8x4(up.aux + sign_byte) >> sign_shift;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint32_t gate_index = (gate_indices >> (8 * i)) & 0xffu;
            const uint32_t up_index = (up_indices >> (8 * i)) & 0xffu;
            const uint32_t gate_mask7 = (gate_signs >> (7 * i)) & 0x7fu;
            const uint32_t up_mask7 = (up_signs >> (7 * i)) & 0x7fu;
            metadata[segment_base + i] = gate_index | (up_index << 8) |
                (gate_mask7 << 16) | (up_mask7 << 23);
        }
    }
    __syncthreads();

    const int64_t sub_row = static_cast<int64_t>(row) * gate.ng;
    const float gate_anchor = gate.neuron_scale[row];
    const float up_anchor = up.neuron_scale[row];
    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    for (int segment = thread; segment < gate.nsign; segment += NWARPS * 32) {
        const uint32_t meta = metadata[segment];
        const uint32_t gate_index = meta & 0xffu;
        const uint32_t up_index = (meta >> 8) & 0xffu;
        const uint32_t gate_mask7 = (meta >> 16) & 0x7fu;
        const uint32_t up_mask7 = (meta >> 23) & 0x7fu;
        const int gate_last = parity7(gate_mask7) ^
            (gate.sign_mode ? ((gate_index >> 7) & 1u) : 0u);
        const int up_last = parity7(up_mask7) ^
            (up.sign_mode ? ((up_index >> 7) & 1u) : 0u);
        const int2 gate_values = apply_sign8(
            reinterpret_cast<const int2 *>(gate.codebook)[gate_index],
            gate_mask7 | (static_cast<uint32_t>(gate_last) << 7));
        const int2 up_values = apply_sign8(
            reinterpret_cast<const int2 *>(up.codebook)[up_index],
            up_mask7 | (static_cast<uint32_t>(up_last) << 7));
        const int2 x = *reinterpret_cast<const int2 *>(qx + segment * 8);
        const int gate_dot = __dp4a(
            gate_values.y, x.y, __dp4a(gate_values.x, x.x, 0));
        const int up_dot = __dp4a(
            up_values.y, x.y, __dp4a(up_values.x, x.x, 0));
        const int group = segment / 3;
        const float activation_scale = xscale[group];
        const uint32_t gate_sub = load_packed_4(gate.sub_scale, sub_row + group);
        const uint32_t up_sub = load_packed_4(up.sub_scale, sub_row + group);
        const float gate_scale = gate_anchor * static_cast<float>(gate_sub);
        const float up_scale = up_anchor * static_cast<float>(up_sub);
        gate_acc = fmaf(
            gate_scale * activation_scale, static_cast<float>(gate_dot), gate_acc);
        up_acc = fmaf(
            up_scale * activation_scale, static_cast<float>(up_dot), up_acc);
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        gate_acc += __shfl_xor_sync(0xffffffff, gate_acc, offset);
        up_acc += __shfl_xor_sync(0xffffffff, up_acc, offset);
    }
    __shared__ float partial[2][NWARPS];
    if (lane == 0) {
        partial[0][warp] = gate_acc;
        partial[1][warp] = up_acc;
    }
    __syncthreads();
    if (warp == 0) {
        gate_acc = lane < NWARPS ? partial[0][lane] : 0.0f;
        up_acc = lane < NWARPS ? partial[1][lane] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            gate_acc += __shfl_xor_sync(0xffffffff, gate_acc, offset);
            up_acc += __shfl_xor_sync(0xffffffff, up_acc, offset);
        }
        if (lane == 0) {
            const float gate_value = __half2float(__float2half(gate_acc));
            const float up_value = __half2float(__float2half(up_acc));
            const float sigmoid = 1.0f / (1.0f + expf(-gate_value));
            const float value = up_value * (gate_value * sigmoid);
            if constexpr (OUTPUT_F32) {
                output_f[row] = value;
            } else {
                output_h[row] = __float2half(value);
            }
        }
    }
}

template <int GS>
__global__ void nvq_quantize_f32_kernel(
    const float * input,
    int8_t * output_qx,
    float * output_xscale,
    int K,
    int K_pad) {
    const int group = blockIdx.x;
    const int lane = threadIdx.x;
    const int k = group * GS + lane;
    const bool valid = lane < GS && k < K;
    const float value = valid ? input[k] : 0.0f;
    float maximum = fabsf(value);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
    }
    const float scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
    int quant = 0;
    if (valid) {
        quant = static_cast<int>(roundf(value / scale));
        quant = quant < -127 ? -127 : (quant > 127 ? 127 : quant);
    }
    if (lane == 0) output_xscale[group] = scale;
    if (lane < GS && k < K_pad) output_qx[k] = static_cast<int8_t>(quant);
}

// The aligned group layout is faster when gate/up rows remain independent:
// 2N blocks expose both projections to the scheduler and keep the GEMV at the
// lower single-projection register count.  The existing N-float workspace is
// exactly large enough for the two fp16 projections, so this adds no workspace.
template <int GS>
__global__ void nvq_swiglu_quantize_f16_pair_kernel(
    const __half * gate_up,
    int8_t * output_qx,
    float * output_xscale,
    int K,
    int K_pad) {
    const int group = blockIdx.x;
    const int lane = threadIdx.x;
    if (group * GS >= K_pad) return;
    const int k = group * GS + lane;
    const bool valid = lane < GS && k < K;
    float value = 0.0f;
    if (valid) {
        const float gate_value = __half2float(gate_up[k]);
        const float up_value = __half2float(gate_up[K + k]);
        const float sigmoid = 1.0f / (1.0f + expf(-gate_value));
        value = up_value * (gate_value * sigmoid);
    }
    float maximum = fabsf(value);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
    }
    const float scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
    int quant = 0;
    if (valid) {
        quant = static_cast<int>(roundf(value / scale));
        quant = quant < -127 ? -127 : (quant > 127 ? 127 : quant);
    }
    if (lane == 0) output_xscale[group] = scale;
    if (lane < GS && k < K_pad) output_qx[k] = static_cast<int8_t>(quant);
}

// Computes gate/up, applies SwiGLU, and writes the q8 activation layout used
// directly by the following NINT/NVQ down projection.
template <int FORMAT, int DOWN_GS>
__global__ void __launch_bounds__(1024, 1) nvq_ffn_swiglu_quant_kernel(
    NvqDeviceWeight gate,
    NvqDeviceWeight up,
    const int8_t * input_qx,
    const float * input_xscale,
    int8_t * output_qx,
    float * output_xscale,
    int down_k_pad) {
    const int output_group = blockIdx.x;
    const int element = threadIdx.y;
    const int lane = threadIdx.x;
    const int row = output_group * DOWN_GS + element;

    __shared__ float values[DOWN_GS];
    __shared__ float scale;

    float value = 0.0f;
    if (row < gate.N) {
        float gate_acc = 0.0f;
        float up_acc = 0.0f;
        for (int segment = lane; segment < gate.nsign; segment += 32) {
            gate_acc = nvq_vec8_scaled_fma<FORMAT>(
                gate, input_qx, input_xscale, row, segment, gate_acc);
            up_acc = nvq_vec8_scaled_fma<FORMAT>(
                up, input_qx, input_xscale, row, segment, up_acc);
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            gate_acc += __shfl_xor_sync(0xffffffff, gate_acc, offset);
            up_acc += __shfl_xor_sync(0xffffffff, up_acc, offset);
        }
        if (lane == 0) {
            const float gate_value = __half2float(__float2half(gate_acc));
            const float up_value = __half2float(__float2half(up_acc));
            const float sigmoid = 1.0f / (1.0f + expf(-gate_value));
            value = up_value * (gate_value * sigmoid);
        }
    }
    if (lane == 0) values[element] = value;
    __syncthreads();

    if (element == 0) {
        float amax = lane < DOWN_GS ? fabsf(values[lane]) : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, offset));
        }
        if (lane == 0) {
            scale = amax > 0.0f ? amax / 127.0f : 1.0f;
            output_xscale[output_group] = scale;
        }
    }
    __syncthreads();

    if (lane == 0) {
        int quant = 0;
        if (row < gate.N) {
            quant = static_cast<int>(roundf(values[element] / scale));
            quant = max(-127, min(127, quant));
        }
        const int offset = output_group * DOWN_GS + element;
        if (offset < down_k_pad) output_qx[offset] = static_cast<int8_t>(quant);
    }
}

template <int FORMAT>
__global__ void nvq_embedding_kernel(
    const uint8_t * indices,
    int64_t indices_nbytes,
    const uint8_t * aux,
    int64_t aux_nbytes,
    const uint8_t * sub_scale,
    int64_t sub_scale_nbytes,
    const float * neuron_scale,
    const int8_t * codebook,
    const int64_t * token_ids,
    __half * output,
    int tokens,
    int vocab,
    int K,
    int ng,
    int nvec,
    int nsign,
    int sub_bits,
    int sign_mode) {
    const int segments = (K + 7) / 8;
    const int64_t total = static_cast<int64_t>(tokens) * segments;
    for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int token = static_cast<int>(linear / segments);
        const int segment = static_cast<int>(linear - static_cast<int64_t>(token) * segments);
        const int row = static_cast<int>(token_ids[token]);
        if (row < 0 || row >= vocab) continue;
        const int k0 = segment * 8;
        const int group = k0 / kGroupSize;
        const int chunk0 = (k0 - group * kGroupSize) / 4;
        const int64_t sub_linear = static_cast<int64_t>(row) * ng + group;
        const uint32_t sub = load_nvq_state<FORMAT>(indices,indices_nbytes,sub_scale,sub_scale_nbytes,
                    sub_linear,sub_bits,ng);
        const float scale = format_scale<FORMAT>(neuron_scale[row], sub, codebook);
        const int lo = decode_chunk4<FORMAT>(
            indices, indices_nbytes, aux, aux_nbytes, codebook,
            row, group, chunk0, nvec, nsign, ng, sign_mode, sub);
        const int hi = decode_chunk4<FORMAT>(
            indices, indices_nbytes, aux, aux_nbytes, codebook,
            row, group, chunk0 + 1, nvec, nsign, ng, sign_mode, sub);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int k = k0 + i;
            if (k < K) {
                const int packed = i < 4 ? lo : hi;
                const int byte = (packed >> (8 * (i & 3))) & 0xff;
                const int value = static_cast<int>(static_cast<int8_t>(byte));
                output[static_cast<int64_t>(token) * K + k] =
                    __float2half(scale * static_cast<float>(value));
            }
        }
    }
}

#endif

#if defined(MFQ_NVQ_DENSE) || defined(MFQ_NVQ_MOE)
__device__ __forceinline__ int mma168_i(int fragment) {
    return ((fragment / 2) * 8) + (threadIdx.x / 4);
}

__device__ __forceinline__ int mma168_j(int fragment) {
    return ((threadIdx.x % 4) * 2) + (fragment % 2);
}

__device__ __forceinline__ void load_mma_a_m16n8k32(int (&a)[4], const int * ptr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
        : "l"(ptr));
}

__device__ __forceinline__ void load_mma_b_m16n8k32(int (&b)[2], const int * ptr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
        : "=r"(b[0]), "=r"(b[1])
        : "l"(ptr));
}

__device__ __forceinline__ void mma_m16n8k32_s8(
    int (&d)[4], const int (&a)[4], const int (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Integer Tensor Core path for M=4..64. Each gs24 quantization group is
// represented as one transient K32 tile in shared memory. The MMA dot is
// rescaled per group before accumulation, preserving NVQ's group scales.
#endif

#if defined(MFQ_NVQ_DENSE)
template <int FORMAT, int MTILES, bool NEED_CHECK>
__global__ void __launch_bounds__(256) nvq_mmq_mma24_kernel(
    const uint8_t * indices,
    int64_t indices_nbytes,
    const uint8_t * aux,
    int64_t aux_nbytes,
    const uint8_t * sub_scale,
    int64_t sub_scale_nbytes,
    const float * neuron_scale,
    const int8_t * codebook,
    const int8_t * qx,
    const float * xscale,
    __half * output,
    int M,
    int N,
    int ng,
    int nvec,
    int nsign,
    int sub_bits,
    int sign_mode) {
    constexpr int kChunkGroups = 8;
    constexpr int kGroupInts = 8;
    constexpr int kStride = kChunkGroups * kGroupInts + 4;
    constexpr int kTileM = 16 * MTILES;
    constexpr int kRowsPerWarp = 8;
    constexpr int kWarps = 8;
    constexpr int kTileN = kRowsPerWarp * kWarps;

    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    const int tid = warp * 32 + lane;
    const int m0 = blockIdx.y * kTileM;
    const int n0 = blockIdx.x * kTileN;

    __shared__ int activation[kTileM][kStride];
    __shared__ int weight[kWarps][kRowsPerWarp][kStride];
    __shared__ __half weight_scale[kChunkGroups][kTileN];
    __shared__ float activation_scale[kChunkGroups][kTileM];

    float acc[MTILES][4];
#pragma unroll
    for (int mt = 0; mt < MTILES; ++mt) {
#pragma unroll
        for (int fragment = 0; fragment < 4; ++fragment) acc[mt][fragment] = 0.0f;
    }

    for (int group_base = 0; group_base < ng; group_base += kChunkGroups) {
        const int active_groups = min(kChunkGroups, ng - group_base);

        for (int index = tid; index < kTileM * kChunkGroups * kGroupInts;
             index += kWarps * 32) {
            const int m_local = index / (kChunkGroups * kGroupInts);
            const int rest = index - m_local * kChunkGroups * kGroupInts;
            const int group_local = rest / kGroupInts;
            const int quartet = rest - group_local * kGroupInts;
            const int m = m0 + m_local;
            int value = 0;
            if (group_local < active_groups && quartet < kChunksPerGroup &&
                (!NEED_CHECK || m < M)) {
                const int group = group_base + group_local;
                value = load_i8x4(
                    qx + (static_cast<int64_t>(m) * ng + group) * kGroupSize + quartet * 4);
            }
            activation[m_local][group_local * kGroupInts + quartet] = value;
        }

        constexpr int kWeightTasks = kRowsPerWarp * kChunkGroups;
        for (int task = lane; task < kWeightTasks; task += 32) {
            const int row_local = task / kChunkGroups;
            const int group_local = task - row_local * kChunkGroups;
            const int row = n0 + warp * kRowsPerWarp + row_local;
            const int group = group_base + group_local;
            const bool valid = group_local < active_groups && (!NEED_CHECK || row < N);
            uint32_t sub = 0;
            if (valid) {
                const int64_t sub_linear = static_cast<int64_t>(row) * ng + group;
                sub = load_nvq_state<FORMAT>(indices,indices_nbytes,sub_scale,sub_scale_nbytes,
                    sub_linear,sub_bits,ng);
            }
#pragma unroll
            for (int quartet = 0; quartet < kChunksPerGroup; ++quartet) {
                weight[warp][row_local][group_local * kGroupInts + quartet] = valid
                    ? decode_chunk4<FORMAT>(
                          indices, indices_nbytes, aux, aux_nbytes, codebook,
                          row, group, quartet, nvec, nsign, ng, sign_mode, sub)
                    : 0;
            }
            weight[warp][row_local][group_local * kGroupInts + 6] = 0;
            weight[warp][row_local][group_local * kGroupInts + 7] = 0;
            float scale = 0.0f;
            if (valid) {
                scale = format_scale<FORMAT>(neuron_scale[row], sub, codebook);
            }
            weight_scale[group_local][warp * kRowsPerWarp + row_local] = __float2half(scale);
        }

        for (int index = tid; index < kTileM * kChunkGroups; index += kWarps * 32) {
            const int m_local = index / kChunkGroups;
            const int group_local = index - m_local * kChunkGroups;
            const int m = m0 + m_local;
            activation_scale[group_local][m_local] =
                group_local < active_groups && (!NEED_CHECK || m < M)
                ? xscale[static_cast<int64_t>(m) * ng + group_base + group_local]
                : 0.0f;
        }
        __syncthreads();

#pragma unroll
        for (int group_local = 0; group_local < kChunkGroups; ++group_local) {
            int a[MTILES][4];
#pragma unroll
            for (int mt = 0; mt < MTILES; ++mt) {
                load_mma_a_m16n8k32(
                    a[mt],
                    &activation[mt * 16 + lane % 16]
                               [group_local * kGroupInts + (lane / 16) * 4]);
            }
            int b[2];
            load_mma_b_m16n8k32(
                b,
                &weight[warp][lane % 8]
                       [group_local * kGroupInts + (((lane / 8) * 4) & 7)]);
#pragma unroll
            for (int mt = 0; mt < MTILES; ++mt) {
                int dot[4] = {0, 0, 0, 0};
                mma_m16n8k32_s8(dot, a[mt], b);
#pragma unroll
                for (int fragment = 0; fragment < 4; ++fragment) {
                    const int m_local = mt * 16 + mma168_i(fragment);
                    const int row_local = warp * kRowsPerWarp + mma168_j(fragment);
                    const int m = m0 + m_local;
                    const int row = n0 + row_local;
                    if (!NEED_CHECK || (m < M && row < N)) {
                        const float scale = activation_scale[group_local][m_local]
                            * __half2float(weight_scale[group_local][row_local]);
                        acc[mt][fragment] = fmaf(scale, static_cast<float>(dot[fragment]),
                                                acc[mt][fragment]);
                    }
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int mt = 0; mt < MTILES; ++mt) {
#pragma unroll
        for (int fragment = 0; fragment < 4; ++fragment) {
            const int m = m0 + mt * 16 + mma168_i(fragment);
            const int row = n0 + warp * kRowsPerWarp + mma168_j(fragment);
            if (!NEED_CHECK || (m < M && row < N)) {
                output[static_cast<int64_t>(m) * N + row] = __float2half(acc[mt][fragment]);
            }
        }
    }
}

// Online-dequant FP16 Tensor Core path for M=16..256. Four gs24 groups form
// one K=96 chunk. A block expands a 64-row weight tile into shared memory and
// reuses it across 16..128 activation rows without materializing full weights.
template <int FORMAT, int MTILES, bool NEED_CHECK>
__global__ void __launch_bounds__(256) nvq_gemm_f16_gs24_kernel(
    const uint8_t * indices,
    int64_t indices_nbytes,
    const uint8_t * aux,
    int64_t aux_nbytes,
    const uint8_t * sub_scale,
    int64_t sub_scale_nbytes,
    const float * neuron_scale,
    const int8_t * codebook,
    const __half * x,
    __half * output,
    int M,
    int N,
    int K,
    int ng,
    int nvec,
    int nsign,
    int sub_bits,
    int sign_mode) {
    constexpr int kGroupsPerChunk = 4;
    constexpr int kTileK = kGroupSize * kGroupsPerChunk;
    constexpr int kStrideK = kTileK + 8;
    constexpr int kTileM = 16 * MTILES;
    constexpr int kTileN = 64;
    constexpr int kWarps = 8;
    constexpr int kWmmaM = 16;
    constexpr int kWmmaN = 16;
    constexpr int kWmmaK = 16;
    constexpr int kNFragments = kTileN / kWmmaN;
    constexpr int kAccumulatorsPerWarp = (MTILES + 1) / 2;

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid = warp * 32 + lane;
    const int m0 = blockIdx.y * kTileM;
    const int n0 = blockIdx.x * kTileN;
    const int warp_m0 = warp / kNFragments;
    const int warp_n = warp % kNFragments;

    __shared__ __half weight_tile[kTileN][kStrideK];
    __shared__ __half activation_tile[kTileM][kStrideK];
    __shared__ float output_tile[kNFragments][kWmmaM][kWmmaN];

    using FragmentA = wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK,
                                     __half, wmma::row_major>;
    using FragmentB = wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK,
                                     __half, wmma::col_major>;
    using FragmentC = wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float>;
    FragmentC accumulators[kAccumulatorsPerWarp];
#pragma unroll
    for (int accumulator = 0; accumulator < kAccumulatorsPerWarp; ++accumulator) {
        wmma::fill_fragment(accumulators[accumulator], 0.0f);
    }

    const int chunks = (ng + kGroupsPerChunk - 1) / kGroupsPerChunk;
    for (int chunk = 0; chunk < chunks; ++chunk) {
        const int group_base = chunk * kGroupsPerChunk;
        const int k_base = group_base * kGroupSize;

        // There are exactly 64 * 4 row/group tasks, one per block thread.
        const int row_local = tid / kGroupsPerChunk;
        const int group_local = tid - row_local * kGroupsPerChunk;
        const int row = n0 + row_local;
        const int group = group_base + group_local;
        const bool valid_weight = group < ng && (!NEED_CHECK || row < N);
        uint32_t state = 0;
        float scale = 0.0f;
        if (valid_weight) {
            const int64_t sub_linear = static_cast<int64_t>(row) * ng + group;
            state = load_nvq_state<FORMAT>(indices,indices_nbytes,sub_scale,sub_scale_nbytes,
                    sub_linear,sub_bits,ng);
            scale = format_scale<FORMAT>(neuron_scale[row], state, codebook);
        }
#pragma unroll
        for (int quartet = 0; quartet < kChunksPerGroup; ++quartet) {
            const int packed = valid_weight
                ? decode_chunk4<FORMAT>(
                      indices, indices_nbytes, aux, aux_nbytes, codebook,
                      row, group, quartet, nvec, nsign, ng, sign_mode, state)
                : 0;
#pragma unroll
            for (int pair = 0; pair < 2; ++pair) {
                const int shift = pair * 16;
                const int value0 = static_cast<int>(
                    static_cast<int8_t>((packed >> shift) & 0xff));
                const int value1 = static_cast<int>(
                    static_cast<int8_t>((packed >> (shift + 8)) & 0xff));
                const __half2 values = __halves2half2(
                    __float2half(scale * static_cast<float>(value0)),
                    __float2half(scale * static_cast<float>(value1)));
                *reinterpret_cast<__half2 *>(
                    &weight_tile[row_local]
                                [group_local * kGroupSize + quartet * 4 + pair * 2]) = values;
            }
        }

        constexpr int kActivationPairs = kTileM * (kTileK / 2);
        for (int index = tid; index < kActivationPairs; index += kWarps * 32) {
            const int m_local = index / (kTileK / 2);
            const int pair = index - m_local * (kTileK / 2);
            const int m = m0 + m_local;
            const int k = k_base + pair * 2;
            __half2 values = __float2half2_rn(0.0f);
            if ((!NEED_CHECK || m < M) && k + 1 < K && (K & 1) == 0) {
                values = *reinterpret_cast<const __half2 *>(
                    x + static_cast<int64_t>(m) * K + k);
            } else if ((!NEED_CHECK || m < M) && k < K) {
                const __half value0 = x[static_cast<int64_t>(m) * K + k];
                const __half value1 = k + 1 < K
                    ? x[static_cast<int64_t>(m) * K + k + 1]
                    : __float2half(0.0f);
                values = __halves2half2(value0, value1);
            }
            *reinterpret_cast<__half2 *>(
                &activation_tile[m_local][pair * 2]) = values;
        }
        __syncthreads();

        const bool warp_active = warp_m0 < MTILES;
#pragma unroll
        for (int k_local = 0; k_local < kTileK; k_local += kWmmaK) {
            if (warp_active) {
                FragmentB weight_fragment;
                wmma::load_matrix_sync(
                    weight_fragment,
                    &weight_tile[warp_n * kWmmaN][k_local],
                    kStrideK);
#pragma unroll
                for (int accumulator = 0;
                     accumulator < kAccumulatorsPerWarp;
                     ++accumulator) {
                    const int m_fragment = warp_m0 + accumulator * 2;
                    if (m_fragment < MTILES) {
                        FragmentA activation_fragment;
                        wmma::load_matrix_sync(
                            activation_fragment,
                            &activation_tile[m_fragment * kWmmaM][k_local],
                            kStrideK);
                        wmma::mma_sync(
                            accumulators[accumulator], activation_fragment,
                            weight_fragment, accumulators[accumulator]);
                    }
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int accumulator = 0;
         accumulator < kAccumulatorsPerWarp;
         ++accumulator) {
#pragma unroll
        for (int warp_row = 0; warp_row < 2; ++warp_row) {
            const int m_fragment = warp_m0 + accumulator * 2;
            const bool owns_fragment = warp_m0 == warp_row && m_fragment < MTILES;
            if (owns_fragment) {
                wmma::store_matrix_sync(
                    &output_tile[warp_n][0][0], accumulators[accumulator],
                    kWmmaN, wmma::mem_row_major);
            }
            __syncthreads();
            if (owns_fragment) {
                const int output_m0 = m0 + m_fragment * kWmmaM;
                const int output_n0 = n0 + warp_n * kWmmaN;
#pragma unroll
                for (int element = lane; element < kWmmaM * kWmmaN; element += 32) {
                    const int local_m = element / kWmmaN;
                    const int local_n = element - local_m * kWmmaN;
                    const int m = output_m0 + local_m;
                    const int n = output_n0 + local_n;
                    if (!NEED_CHECK || (m < M && n < N)) {
                        output[static_cast<int64_t>(m) * N + n] =
                            __float2half(output_tile[warp_n][local_m][local_n]);
                    }
                }
            }
            __syncthreads();
        }
    }
}

#endif

#if defined(MFQ_NVQ_MOE)
template <int FORMAT>
__global__ void nepq_dequant_kernel(
    const uint8_t * indices,
    int64_t indices_nbytes,
    const uint8_t * aux,
    int64_t aux_nbytes,
    const uint8_t * sub_scale,
    int64_t sub_scale_nbytes,
    const float * neuron_scale,
    const int8_t * table_pool,
    const uint8_t * bank_ids,
    __half * weight,
    int N,
    int K,
    int ng,
    int nvec,
    int nsign,
    int nsuper,
    int table_stride,
    int sub_bits,
    int sign_mode) {
    const int64_t total = static_cast<int64_t>(N) * nsign;
    for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int row = static_cast<int>(linear / nsign);
        const int segment = static_cast<int>(linear - static_cast<int64_t>(row) * nsign);
        const int group = segment / 3;
        const int segment_in_group = segment - group * 3;
        const int k0 = group * kGroupSize + segment_in_group * 8;
        const int64_t sub_linear = static_cast<int64_t>(row) * ng + group;
        const uint32_t sub = load_nvq_state<FORMAT>(indices,indices_nbytes,sub_scale,sub_scale_nbytes,
                    sub_linear,sub_bits,ng);
        const int8_t * table = nepq_active_table(
            table_pool, bank_ids, row, group, nsuper, table_stride);
        const auto values = load_nepq_vec8<FORMAT>(
            indices, indices_nbytes, aux, aux_nbytes, table,
            row, segment, group, ng, nvec, nsign, sign_mode, sub);
        const float scale = (FORMAT == kNvq1L || FORMAT == kNvq1S)
            ? neuron_scale[row] * static_cast<float>(sub)
            : format_scale<FORMAT>(neuron_scale[row], sub, table);
        const int2 packed = values.values;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int k = k0 + i;
            if (k >= K) continue;
            const int word = i < 4 ? packed.x : packed.y;
            float value = static_cast<float>(
                static_cast<int8_t>((word >> (8 * (i & 3))) & 0xff));
            if constexpr (FORMAT == kNvq1L) {
                value += 0.125f * static_cast<float>(values.delta);
            }
            if constexpr (FORMAT == kNvq1S) {
                value += 0.15625f * static_cast<float>(values.delta);
            }
            weight[static_cast<int64_t>(row) * K + k] = __float2half(scale * value);
        }
    }
}

template <int FORMAT, int NWARPS, int MAX_M>
__global__ void __launch_bounds__(NWARPS * 32, 1) nepq_gemv_batch_vec8_kernel(
    const uint8_t * indices,
    int64_t indices_nbytes,
    const uint8_t * aux,
    int64_t aux_nbytes,
    const uint8_t * sub_scale,
    int64_t sub_scale_nbytes,
    const float * neuron_scale,
    const int8_t * table_pool,
    const uint8_t * bank_ids,
    const int8_t * qx,
    const float * xscale,
    __half * output,
    int M,
    int N,
    int ng,
    int nvec,
    int nsign,
    int nsuper,
    int table_stride,
    int sub_bits,
    int sign_mode) {
    const int row = blockIdx.x;
    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    if (row >= N) return;

    float acc[MAX_M];
#pragma unroll
    for (int m = 0; m < MAX_M; ++m) acc[m] = 0.0f;
    for (int segment = warp * 32 + lane;
         segment < nsign;
         segment += NWARPS * 32) {
        const int group = segment / 3;
        const int segment_in_group = segment - group * 3;
        const int k = group * kGroupSize + segment_in_group * 8;
        const int64_t sub_linear = static_cast<int64_t>(row) * ng + group;
        const uint32_t sub = load_nvq_state<FORMAT>(indices,indices_nbytes,sub_scale,sub_scale_nbytes,
                    sub_linear,sub_bits,ng);
        const int8_t * table = nepq_active_table(
            table_pool, bank_ids, row, group, nsuper, table_stride);
        const auto weight = load_nepq_vec8<FORMAT>(
            indices, indices_nbytes, aux, aux_nbytes, table,
            row, segment, group, ng, nvec, nsign, sign_mode, sub);
        const float weight_scale = (FORMAT == kNvq1L || FORMAT == kNvq1S)
            ? neuron_scale[row] * static_cast<float>(sub)
            : format_scale<FORMAT>(neuron_scale[row], sub, table);
#pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < M) {
                const float dot = dot_nvq_vec8<FORMAT>(
                    weight, qx + (static_cast<int64_t>(m) * ng * kGroupSize + k));
                acc[m] = fmaf(
                    weight_scale * xscale[static_cast<int64_t>(m) * ng + group],
                    dot,
                    acc[m]);
            }
        }
    }

#pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc[m] += __shfl_xor_sync(0xffffffff, acc[m], offset);
        }
    }
    __shared__ float partial[MAX_M][NWARPS];
    if (lane == 0) {
#pragma unroll
        for (int m = 0; m < MAX_M; ++m) partial[m][warp] = acc[m];
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            acc[m] = lane < NWARPS ? partial[m][lane] : 0.0f;
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                acc[m] += __shfl_xor_sync(0xffffffff, acc[m], offset);
            }
            if (lane == 0 && m < M) {
                output[static_cast<int64_t>(m) * N + row] = __float2half(acc[m]);
            }
        }
    }
}

template <int FORMAT, int MTILES, bool NEED_CHECK>
__global__ void __launch_bounds__(256) nepq_mmq_mma24_kernel(
    const uint8_t * indices,
    int64_t indices_nbytes,
    const uint8_t * aux,
    int64_t aux_nbytes,
    const uint8_t * sub_scale,
    int64_t sub_scale_nbytes,
    const float * neuron_scale,
    const int8_t * table_pool,
    const uint8_t * bank_ids,
    const int8_t * qx,
    const float * xscale,
    __half * output,
    int M,
    int N,
    int ng,
    int nvec,
    int nsign,
    int nsuper,
    int table_stride,
    int sub_bits,
    int sign_mode) {
    constexpr int kChunkGroups = 8;
    constexpr int kGroupInts = 8;
    constexpr int kStride = kChunkGroups * kGroupInts + 4;
    constexpr int kTileM = 16 * MTILES;
    constexpr int kRowsPerWarp = 8;
    constexpr int kWarps = 8;
    constexpr int kTileN = kRowsPerWarp * kWarps;

    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    const int tid = warp * 32 + lane;
    const int m0 = blockIdx.y * kTileM;
    const int n0 = blockIdx.x * kTileN;

    __shared__ int activation[kTileM][kStride];
    __shared__ int weight[kWarps][kRowsPerWarp][kStride];
    __shared__ __half weight_scale[kChunkGroups][kTileN];
    __shared__ float activation_scale[kChunkGroups][kTileM];

    float acc[MTILES][4];
#pragma unroll
    for (int mt = 0; mt < MTILES; ++mt) {
#pragma unroll
        for (int fragment = 0; fragment < 4; ++fragment) acc[mt][fragment] = 0.0f;
    }

    for (int group_base = 0; group_base < ng; group_base += kChunkGroups) {
        const int active_groups = min(kChunkGroups, ng - group_base);
        for (int index = tid; index < kTileM * kChunkGroups * kGroupInts;
             index += kWarps * 32) {
            const int m_local = index / (kChunkGroups * kGroupInts);
            const int rest = index - m_local * kChunkGroups * kGroupInts;
            const int group_local = rest / kGroupInts;
            const int quartet = rest - group_local * kGroupInts;
            const int m = m0 + m_local;
            int value = 0;
            if (group_local < active_groups && quartet < kChunksPerGroup &&
                (!NEED_CHECK || m < M)) {
                const int group = group_base + group_local;
                value = load_i8x4(
                    qx + (static_cast<int64_t>(m) * ng + group) * kGroupSize
                    + quartet * 4);
            }
            activation[m_local][group_local * kGroupInts + quartet] = value;
        }

        constexpr int kWeightTasks = kRowsPerWarp * kChunkGroups;
        for (int task = lane; task < kWeightTasks; task += 32) {
            const int row_local = task / kChunkGroups;
            const int group_local = task - row_local * kChunkGroups;
            const int row = n0 + warp * kRowsPerWarp + row_local;
            const int group = group_base + group_local;
            const bool valid = group_local < active_groups && (!NEED_CHECK || row < N);
            uint32_t sub = 0;
            const int8_t * table = table_pool;
            if (valid) {
                const int64_t sub_linear = static_cast<int64_t>(row) * ng + group;
                sub = load_nvq_state<FORMAT>(indices,indices_nbytes,sub_scale,sub_scale_nbytes,
                    sub_linear,sub_bits,ng);
                table = nepq_active_table(
                    table_pool, bank_ids, row, group, nsuper, table_stride);
            }
#pragma unroll
            for (int quartet = 0; quartet < kChunksPerGroup; ++quartet) {
                weight[warp][row_local][group_local * kGroupInts + quartet] = valid
                    ? decode_nepq_chunk4<FORMAT>(
                          indices, indices_nbytes, aux, aux_nbytes, table,
                          row, group, quartet, nvec, nsign, ng, sign_mode, sub)
                    : 0;
            }
            weight[warp][row_local][group_local * kGroupInts + 6] = 0;
            weight[warp][row_local][group_local * kGroupInts + 7] = 0;
            const float scale = valid
                ? format_scale<FORMAT>(neuron_scale[row], sub, table)
                : 0.0f;
            weight_scale[group_local][warp * kRowsPerWarp + row_local] = __float2half(scale);
        }

        for (int index = tid; index < kTileM * kChunkGroups; index += kWarps * 32) {
            const int m_local = index / kChunkGroups;
            const int group_local = index - m_local * kChunkGroups;
            const int m = m0 + m_local;
            activation_scale[group_local][m_local] =
                group_local < active_groups && (!NEED_CHECK || m < M)
                ? xscale[static_cast<int64_t>(m) * ng + group_base + group_local]
                : 0.0f;
        }
        __syncthreads();

#pragma unroll
        for (int group_local = 0; group_local < kChunkGroups; ++group_local) {
            int a[MTILES][4];
#pragma unroll
            for (int mt = 0; mt < MTILES; ++mt) {
                load_mma_a_m16n8k32(
                    a[mt],
                    &activation[mt * 16 + lane % 16]
                               [group_local * kGroupInts + (lane / 16) * 4]);
            }
            int b[2];
            load_mma_b_m16n8k32(
                b,
                &weight[warp][lane % 8]
                       [group_local * kGroupInts + (((lane / 8) * 4) & 7)]);
#pragma unroll
            for (int mt = 0; mt < MTILES; ++mt) {
                int dot[4] = {0, 0, 0, 0};
                mma_m16n8k32_s8(dot, a[mt], b);
#pragma unroll
                for (int fragment = 0; fragment < 4; ++fragment) {
                    const int m_local = mt * 16 + mma168_i(fragment);
                    const int row_local = warp * kRowsPerWarp + mma168_j(fragment);
                    const int m = m0 + m_local;
                    const int row = n0 + row_local;
                    if (!NEED_CHECK || (m < M && row < N)) {
                        const float scale = activation_scale[group_local][m_local]
                            * __half2float(weight_scale[group_local][row_local]);
                        acc[mt][fragment] = fmaf(
                            scale, static_cast<float>(dot[fragment]), acc[mt][fragment]);
                    }
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int mt = 0; mt < MTILES; ++mt) {
#pragma unroll
        for (int fragment = 0; fragment < 4; ++fragment) {
            const int m = m0 + mt * 16 + mma168_i(fragment);
            const int row = n0 + warp * kRowsPerWarp + mma168_j(fragment);
            if (!NEED_CHECK || (m < M && row < N)) {
                output[static_cast<int64_t>(m) * N + row] = __float2half(acc[mt][fragment]);
            }
        }
    }
}

template <int FORMAT, int MTILES, bool NEED_CHECK>
__global__ void __launch_bounds__(256) nepq_gemm_f16_gs24_kernel(
    const uint8_t * indices,
    int64_t indices_nbytes,
    const uint8_t * aux,
    int64_t aux_nbytes,
    const uint8_t * sub_scale,
    int64_t sub_scale_nbytes,
    const float * neuron_scale,
    const int8_t * table_pool,
    const uint8_t * bank_ids,
    const __half * x,
    __half * output,
    int M,
    int N,
    int K,
    int ng,
    int nvec,
    int nsign,
    int nsuper,
    int table_stride,
    int sub_bits,
    int sign_mode) {
    constexpr int kGroupsPerChunk = kNepqGroupsPerSupergroup;
    constexpr int kTileK = kGroupSize * kGroupsPerChunk;
    constexpr int kStrideK = kTileK + 8;
    constexpr int kTileM = 16 * MTILES;
    constexpr int kTileN = 64;
    constexpr int kWarps = 8;
    constexpr int kWmmaM = 16;
    constexpr int kWmmaN = 16;
    constexpr int kWmmaK = 16;
    constexpr int kNFragments = kTileN / kWmmaN;
    constexpr int kAccumulatorsPerWarp = (MTILES + 1) / 2;

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid = warp * 32 + lane;
    const int m0 = blockIdx.y * kTileM;
    const int n0 = blockIdx.x * kTileN;
    const int warp_m0 = warp / kNFragments;
    const int warp_n = warp % kNFragments;

    __shared__ __half weight_tile[kTileN][kStrideK];
    __shared__ __half activation_tile[kTileM][kStrideK];
    __shared__ float output_tile[kNFragments][kWmmaM][kWmmaN];

    using FragmentA = wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK,
                                     __half, wmma::row_major>;
    using FragmentB = wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK,
                                     __half, wmma::col_major>;
    using FragmentC = wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float>;
    FragmentC accumulators[kAccumulatorsPerWarp];
#pragma unroll
    for (int accumulator = 0; accumulator < kAccumulatorsPerWarp; ++accumulator) {
        wmma::fill_fragment(accumulators[accumulator], 0.0f);
    }

    const int chunks = (ng + kGroupsPerChunk - 1) / kGroupsPerChunk;
    for (int chunk = 0; chunk < chunks; ++chunk) {
        const int group_base = chunk * kGroupsPerChunk;
        const int k_base = group_base * kGroupSize;
        const int row_local = tid / kGroupsPerChunk;
        const int group_local = tid - row_local * kGroupsPerChunk;
        const int row = n0 + row_local;
        const int group = group_base + group_local;
        const bool valid_weight = group < ng && (!NEED_CHECK || row < N);
        uint32_t state = 0;
        float scale = 0.0f;
        const int8_t * table = table_pool;
        if (valid_weight) {
            const int64_t sub_linear = static_cast<int64_t>(row) * ng + group;
            state = load_nvq_state<FORMAT>(indices,indices_nbytes,sub_scale,sub_scale_nbytes,
                    sub_linear,sub_bits,ng);
            table = nepq_active_table(
                table_pool, bank_ids, row, group, nsuper, table_stride);
            scale = format_scale<FORMAT>(neuron_scale[row], state, table);
        }
#pragma unroll
        for (int quartet = 0; quartet < kChunksPerGroup; ++quartet) {
            const int packed = valid_weight
                ? decode_nepq_chunk4<FORMAT>(
                      indices, indices_nbytes, aux, aux_nbytes, table,
                      row, group, quartet, nvec, nsign, ng, sign_mode, state)
                : 0;
#pragma unroll
            for (int pair = 0; pair < 2; ++pair) {
                const int shift = pair * 16;
                float value0 = static_cast<float>(
                    static_cast<int8_t>((packed >> shift) & 0xff));
                float value1 = static_cast<float>(
                    static_cast<int8_t>((packed >> (shift + 8)) & 0xff));
                const __half2 values = __halves2half2(
                    __float2half(scale * value0), __float2half(scale * value1));
                *reinterpret_cast<__half2 *>(
                    &weight_tile[row_local]
                                [group_local * kGroupSize + quartet * 4 + pair * 2]) = values;
            }
        }

        constexpr int kActivationPairs = kTileM * (kTileK / 2);
        for (int index = tid; index < kActivationPairs; index += kWarps * 32) {
            const int m_local = index / (kTileK / 2);
            const int pair = index - m_local * (kTileK / 2);
            const int m = m0 + m_local;
            const int k = k_base + pair * 2;
            __half2 values = __float2half2_rn(0.0f);
            if ((!NEED_CHECK || m < M) && k + 1 < K && (K & 1) == 0) {
                values = *reinterpret_cast<const __half2 *>(
                    x + static_cast<int64_t>(m) * K + k);
            } else if ((!NEED_CHECK || m < M) && k < K) {
                const __half value0 = x[static_cast<int64_t>(m) * K + k];
                const __half value1 = k + 1 < K
                    ? x[static_cast<int64_t>(m) * K + k + 1]
                    : __float2half(0.0f);
                values = __halves2half2(value0, value1);
            }
            *reinterpret_cast<__half2 *>(
                &activation_tile[m_local][pair * 2]) = values;
        }
        __syncthreads();

        const bool warp_active = warp_m0 < MTILES;
#pragma unroll
        for (int k_local = 0; k_local < kTileK; k_local += kWmmaK) {
            if (warp_active) {
                FragmentB weight_fragment;
                wmma::load_matrix_sync(
                    weight_fragment, &weight_tile[warp_n * kWmmaN][k_local], kStrideK);
#pragma unroll
                for (int accumulator = 0;
                     accumulator < kAccumulatorsPerWarp;
                     ++accumulator) {
                    const int m_fragment = warp_m0 + accumulator * 2;
                    if (m_fragment < MTILES) {
                        FragmentA activation_fragment;
                        wmma::load_matrix_sync(
                            activation_fragment,
                            &activation_tile[m_fragment * kWmmaM][k_local],
                            kStrideK);
                        wmma::mma_sync(
                            accumulators[accumulator], activation_fragment,
                            weight_fragment, accumulators[accumulator]);
                    }
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int accumulator = 0; accumulator < kAccumulatorsPerWarp; ++accumulator) {
#pragma unroll
        for (int warp_row = 0; warp_row < 2; ++warp_row) {
            const int m_fragment = warp_m0 + accumulator * 2;
            const bool owns_fragment = warp_m0 == warp_row && m_fragment < MTILES;
            if (owns_fragment) {
                wmma::store_matrix_sync(
                    &output_tile[warp_n][0][0], accumulators[accumulator],
                    kWmmaN, wmma::mem_row_major);
            }
            __syncthreads();
            if (owns_fragment) {
                const int output_m0 = m0 + m_fragment * kWmmaM;
                const int output_n0 = n0 + warp_n * kWmmaN;
#pragma unroll
                for (int element = lane; element < kWmmaM * kWmmaN; element += 32) {
                    const int local_m = element / kWmmaN;
                    const int local_n = element - local_m * kWmmaN;
                    const int m = output_m0 + local_m;
                    const int n = output_n0 + local_n;
                    if (!NEED_CHECK || (m < M && n < N)) {
                        output[static_cast<int64_t>(m) * N + n] =
                            __float2half(output_tile[warp_n][local_m][local_n]);
                    }
                }
            }
            __syncthreads();
        }
    }
}

template<int Format>
__device__ __forceinline__ float nvq_dense_routed_group_dot(
        const NvqDeviceWeight& w,int row,const int8_t* input,const float* scales) {
    float sum=0.f;
    for(int group=threadIdx.x;group<w.ng;group+=32) {
        const auto record=load_dense_nvq_group<Format>(w.indices,w.indices_nbytes,row,group,w.nsign,w.ng);
        const auto state=record.state();
        const auto* bank=active_codebook<Format>(w.codebook,state);
        int dot=0;
#pragma unroll
        for(int segment=0;segment<3;++segment)if(group*3+segment<w.nsign) {
            const auto values=decode_dense_nvq_vector<Format>(record,bank,segment,w.sign_mode);
            const auto x=*reinterpret_cast<const int2*>(input+group*24+segment*8);
            dot=__dp4a(values.y,x.y,__dp4a(values.x,x.x,dot));
        }
        sum=fmaf(format_scale<Format>(w.neuron_scale[row],state,w.codebook)*scales[group],float(dot),sum);
    }
#pragma unroll
    for(int offset=16;offset>0;offset>>=1)sum+=__shfl_xor_sync(0xffffffffu,sum,offset);
    return sum;
}

template<int Format>
__device__ __forceinline__ float nvq_routed_group_dot(
        const NvqDeviceWeight& w,int row,const int8_t* input,const float* scales) {
    if constexpr(dense_nvq_format(Format))if(w.aux_nbytes==0)
        return nvq_dense_routed_group_dot<Format>(w,row,input,scales);
    float sum=0.f;
    for(int group=threadIdx.x;group<w.ng;group+=32) {
        const int64_t state_index=int64_t(row)*w.ng+group;
        const uint32_t state=load_nvq_state<Format>(w.indices,w.indices_nbytes,w.sub_scale,
            w.sub_scale_nbytes,state_index,w.sub_bits,w.ng);
        const float scale=(Format==kNvq1L || Format==kNvq1S) ? w.neuron_scale[row]*float(state) :
            format_scale<Format>(w.neuron_scale[row],state,w.codebook);
        const int8_t* bank=active_codebook<Format>(w.codebook,state);
        uint64_t packed=0;uint32_t signs=0,exec96[3]{};int delta=0;
        if constexpr(Format==kNvq1L || Format==kNvq1S) {
            if(w.aux_nbytes==0)packed=load_nvq1_record64<Format>(w.indices,w.indices_nbytes,row,group,w.ng);
            const int negative=w.aux_nbytes==0?int((packed>>(Format==kNvq1S?31:36))&1u):
                load_packed_bits(w.aux,state_index,1,w.aux_nbytes);
            delta=negative?-1:1;
            if constexpr(Format==kNvq1S)bank=w.codebook+negative*kNvq1SBankBytes;
        }else if constexpr(Format==kNvq2JscXLGroupExec) {
            packed=load_group_exec64(w.indices,row,group,w.ng);
        }else if constexpr(Format==kNvq3JscLGroupExec) {
            load_group_exec96_words(w.indices,row,group,w.ng,exec96);
        }else if constexpr(Format!=kNvq2Exec && Format!=kNvq2JscExec && Format!=kNpq0L && Format!=kNpq0S) {
            const int64_t sign_bit=(int64_t(row)*w.nsign+group*3)*7;
            signs=w.ng>32 ? load_packed_sign_group3_words(w.aux,sign_bit,w.aux_nbytes) :
                load_packed_sign_group3(w.aux,sign_bit,w.aux_nbytes);
        }
        if constexpr(Format!=kNvq2Exec && Format!=kNvq2JscExec && Format!=kNpq0L && Format!=kNpq0S &&
                     Format!=kNvq2JscXLGroupExec && Format!=kNvq3JscLGroupExec) {
            constexpr int bits=format_index_bits(Format),vectors=is_d4_format(Format)?6:3;
            if(!((Format==kNvq1L || Format==kNvq1S) && w.aux_nbytes==0))
                packed=load_packed_group_window<bits*vectors>(w.indices,(int64_t(row)*w.nvec+group*vectors)*bits,w.indices_nbytes);
        }
        int dot=0,activation_sum=0;
#pragma unroll
        for(int segment=0;segment<3;++segment) {
            const auto weight=load_nvq_group_vec8<Format,true>(w.indices,w.indices_nbytes,w.aux,w.aux_nbytes,
                w.codebook,bank,row,group,segment,w.ng,w.nvec,w.nsign,w.sign_mode,state,signs,packed,exec96,delta);
            const int2 activation=*reinterpret_cast<const int2*>(input+group*24+segment*8);
            if(weight.valid) {
                dot=__dp4a(weight.values.y,activation.y,__dp4a(weight.values.x,activation.x,dot));
                if constexpr(Format==kNvq1L || Format==kNvq1S)
                    activation_sum=__dp4a(0x01010101,activation.y,__dp4a(0x01010101,activation.x,activation_sum));
            }
        }
        float value=float(dot);
        if constexpr(Format==kNvq1L)value+=0.125f*float(delta*activation_sum);
        if constexpr(Format==kNvq1S)value+=0.15625f*float(delta*activation_sum);
        sum=fmaf(scale*scales[group],value,sum);
    }
#pragma unroll
    for(int offset=16;offset>0;offset>>=1)sum+=__shfl_xor_sync(0xffffffffu,sum,offset);
    return sum;
}

template<int Format>
__global__ void __launch_bounds__(128) nvq_moe_groups_kernel(
        NvqDeviceWeight w,const int8_t* input,const float* scales,const int32_t* ids,
        const int32_t* local_map,half* output,int experts,int local_experts,int rows,int routes,bool down) {
    const int pair=blockIdx.y,row=blockIdx.x*4+threadIdx.y;
    if(row>=rows)return;
    const int expert=ids[pair];
    if(unsigned(expert)>=unsigned(experts))return;
    const int local=local_map[expert];if(unsigned(local)>=unsigned(local_experts))return;
    const int source=down?pair:pair/routes;
    const float value=nvq_routed_group_dot<Format>(w,local*rows+row,input+int64_t(source)*w.ng*24,scales+int64_t(source)*w.ng);
    if(threadIdx.x==0)output[int64_t(pair)*rows+row]=__float2half(value);
}

__global__ void __launch_bounds__(128) nvq_moe_groups_hetero_kernel(
        const int64_t* pointers,const int64_t* sizes,const int32_t* params,
        const int32_t* expert_pool,const int32_t* expert_local,const int32_t* ids,
        const int8_t* input,const float* scales,half* output,const int32_t* active,
        int experts,int pools,int rows,int routes,int pairs,bool down) {
    const int pair=blockIdx.y,row=blockIdx.x*4+threadIdx.y;
    if(row>=rows)return;
    const int expert=ids[pair];if(unsigned(expert)>=unsigned(experts))return;
    const int pool=active?active[1+pair]:expert_pool[expert];
    const int local=active?active[1+pairs+pair]:expert_local[expert];
    if(unsigned(pool)>=unsigned(pools) || local<0)return;
    const auto* p=pointers+int64_t(pool)*5;const auto* s=sizes+int64_t(pool)*3;
    const auto* g=params+int64_t(pool)*7;if(local>=g[0])return;
    const NvqDeviceWeight w{reinterpret_cast<const uint8_t*>(p[0]),s[0],reinterpret_cast<const uint8_t*>(p[1]),s[1],
        reinterpret_cast<const uint8_t*>(p[2]),s[2],reinterpret_cast<const float*>(p[3]),reinterpret_cast<const int8_t*>(p[4]),
        0,g[0]*rows,g[1],g[2],g[3],g[4],g[5]};
    const int source=down?pair:pair/routes;float value=0.f;
#define MFQ_NVQ_ROUTED_GROUP_CASE(F) case F: value=nvq_routed_group_dot<F>(w,local*rows+row, \
        input+int64_t(source)*w.ng*24,scales+int64_t(source)*w.ng);break
    switch(g[6]) {
        MFQ_NVQ_ROUTED_GROUP_CASE(1);MFQ_NVQ_ROUTED_GROUP_CASE(2);MFQ_NVQ_ROUTED_GROUP_CASE(3);
        MFQ_NVQ_ROUTED_GROUP_CASE(4);MFQ_NVQ_ROUTED_GROUP_CASE(5);MFQ_NVQ_ROUTED_GROUP_CASE(6);
        MFQ_NVQ_ROUTED_GROUP_CASE(7);MFQ_NVQ_ROUTED_GROUP_CASE(8);MFQ_NVQ_ROUTED_GROUP_CASE(9);
        MFQ_NVQ_ROUTED_GROUP_CASE(10);MFQ_NVQ_ROUTED_GROUP_CASE(11);MFQ_NVQ_ROUTED_GROUP_CASE(12);
        MFQ_NVQ_ROUTED_GROUP_CASE(13);MFQ_NVQ_ROUTED_GROUP_CASE(14);MFQ_NVQ_ROUTED_GROUP_CASE(15);
        MFQ_NVQ_ROUTED_GROUP_CASE(16);MFQ_NVQ_ROUTED_GROUP_CASE(17);
    }
#undef MFQ_NVQ_ROUTED_GROUP_CASE
    if(threadIdx.x==0)output[int64_t(pair)*rows+row]=__float2half(value);
}

template <int FORMAT, int NWARPS, int ROWS_PER_BLOCK, bool SHARE_GROUP_STATE>
__device__ __forceinline__ void nvq_moe_mmvq_task(
    const uint8_t * indices,
    int64_t indices_nbytes,
    const uint8_t * aux,
    int64_t aux_nbytes,
    const uint8_t * sub_scale,
    int64_t sub_scale_nbytes,
    const float * neuron_scale,
    const int8_t * codebook,
    const int8_t * qx,
    const float * xscale,
    __half * output,
    float * partial,
    int pair,
    int routes,
    int out_per_expert,
    int ng,
    int nvec,
    int nsign,
    int sub_bits,
    int sign_mode,
    bool routed_input,
    int source_row,
    int local_expert,
    int row0) {
    const int warp = threadIdx.y;
    const int lane = threadIdx.x;

    float acc[ROWS_PER_BLOCK];
#pragma unroll
    for (int row_local = 0; row_local < ROWS_PER_BLOCK; ++row_local) {
        acc[row_local] = 0.0f;
    }
    for (int segment = warp * 32 + lane;
         segment < nsign;
         segment += NWARPS * 32) {
        const int group = segment / 3;
        const int segment_in_group = segment - group * 3;
        const unsigned int active_mask = __activemask();
        const int group_leader = lane - segment_in_group;
        const int source_lane = group_leader >= 0 ? group_leader : lane;
        const int k = group * kGroupSize + segment_in_group * 8;
        const int8_t * activation =
            qx + (static_cast<int64_t>(source_row) * ng * kGroupSize + k);
        float activation_scale;
        if constexpr (
            SHARE_GROUP_STATE &&
            (FORMAT == kNvq2Exec || FORMAT == kNvq2JscExec)) {
            const float loaded_scale = lane == source_lane
                ? xscale[static_cast<int64_t>(source_row) * ng + group]
                : 0.0f;
            activation_scale = __shfl_sync(
                active_mask, loaded_scale, source_lane);
        } else {
            activation_scale =
                xscale[static_cast<int64_t>(source_row) * ng + group];
        }
#pragma unroll
        for (int row_local = 0; row_local < ROWS_PER_BLOCK; ++row_local) {
            const int local_row = row0 + row_local;
            if (local_row >= out_per_expert) continue;
            const int row = local_expert * out_per_expert + local_row;
            const int64_t sub_linear = static_cast<int64_t>(row) * ng + group;
            uint32_t sub;
            float weight_scale;
            if constexpr (
                SHARE_GROUP_STATE &&
                (FORMAT == kNvq2Exec || FORMAT == kNvq2JscExec)) {
                const uint32_t loaded_sub = lane == source_lane
                    ? load_nvq_state<FORMAT>(indices,indices_nbytes,sub_scale,sub_scale_nbytes,
                    sub_linear,sub_bits,ng)
                    : 0;
                sub = __shfl_sync(active_mask, loaded_sub, source_lane);
                const float loaded_weight_scale = lane == source_lane
                    ? format_scale<FORMAT>(
                        neuron_scale[row], loaded_sub, codebook)
                    : 0.0f;
                weight_scale = __shfl_sync(
                    active_mask, loaded_weight_scale, source_lane);
            } else {
                sub = load_nvq_state<FORMAT>(indices,indices_nbytes,sub_scale,sub_scale_nbytes,
                    sub_linear,sub_bits,ng);
                weight_scale = (FORMAT == kNvq1L || FORMAT == kNvq1S)
                    ? neuron_scale[row] * static_cast<float>(sub)
                    : format_scale<FORMAT>(neuron_scale[row], sub, codebook);
            }
            const auto weight = load_nvq_vec8<FORMAT>(
                indices, indices_nbytes, aux, aux_nbytes, codebook,
                row, segment, group, ng, nvec, nsign, sign_mode, sub);
            acc[row_local] = fmaf(
                weight_scale * activation_scale,
                dot_nvq_vec8<FORMAT>(weight, activation),
                acc[row_local]);
        }
    }

#pragma unroll
    for (int row_local = 0; row_local < ROWS_PER_BLOCK; ++row_local) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc[row_local] += __shfl_xor_sync(0xffffffff, acc[row_local], offset);
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int row_local = 0; row_local < ROWS_PER_BLOCK; ++row_local) {
            partial[row_local * NWARPS + warp] = acc[row_local];
        }
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int row_local = 0; row_local < ROWS_PER_BLOCK; ++row_local) {
            float value = lane < NWARPS
                ? partial[row_local * NWARPS + lane]
                : 0.0f;
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                value += __shfl_xor_sync(0xffffffff, value, offset);
            }
            const int local_row = row0 + row_local;
            if (lane == 0 && local_row < out_per_expert) {
                output[static_cast<int64_t>(pair) * out_per_expert + local_row] =
                    __float2half(value);
            }
        }
    }
}

template <int FORMAT, int NWARPS, int ROWS_PER_BLOCK, bool SHARE_GROUP_STATE>
__global__ void __launch_bounds__(NWARPS * 32) nvq_moe_mmvq_kernel(
    const uint8_t * indices,
    int64_t indices_nbytes,
    const uint8_t * aux,
    int64_t aux_nbytes,
    const uint8_t * sub_scale,
    int64_t sub_scale_nbytes,
    const float * neuron_scale,
    const int8_t * codebook,
    const int8_t * qx,
    const float * xscale,
    const int32_t * ids,
    const int32_t * expert_local,
    __half * output,
    int pairs,
    int routes,
    int global_experts,
    int pool_experts,
    int out_per_expert,
    int ng,
    int nvec,
    int nsign,
    int sub_bits,
    int sign_mode,
    bool routed_input) {
    const int pair = blockIdx.y;
    if (pair >= pairs) return;
    const int expert = ids[pair];
    if (static_cast<unsigned int>(expert) >=
        static_cast<unsigned int>(global_experts)) {
        return;
    }
    const int local_expert = expert_local[expert];
    if (static_cast<unsigned int>(local_expert) >=
        static_cast<unsigned int>(pool_experts)) {
        return;
    }
    __shared__ float partial[ROWS_PER_BLOCK][NWARPS];
    nvq_moe_mmvq_task<
        FORMAT, NWARPS, ROWS_PER_BLOCK, SHARE_GROUP_STATE>(
        indices, indices_nbytes, aux, aux_nbytes,
        sub_scale, sub_scale_nbytes, neuron_scale, codebook,
        qx, xscale, output, &partial[0][0], pair, routes,
        out_per_expert, ng, nvec, nsign, sub_bits, sign_mode,
        routed_input, routed_input ? pair : pair / routes,
        local_expert, blockIdx.x * ROWS_PER_BLOCK);
}

template <int NWARPS, int ROWS_PER_BLOCK, bool SHARE_GROUP_STATE,bool ActiveRows=false>
__global__ void __launch_bounds__(NWARPS * 32)
nvq_moe_mmvq_hetero_kernel(
    const int64_t * weight_ptrs,
    const int64_t * weight_sizes,
    const int32_t * pool_params,
    const int32_t * expert_pool,
    const int32_t * expert_local,
    const int8_t * qx,
    const float * xscale,
    const int32_t * ids,
    __half * output,
    int pairs,
    int routes,
    int global_experts,
    int pool_count,
    int out_per_expert,
    bool routed_input,const int32_t* active_plan=nullptr) {
    const int row_tiles=(out_per_expert-1)/ROWS_PER_BLOCK+1;
    const int64_t total=ActiveRows?int64_t(active_plan[0])*row_tiles:1;
    for(int64_t task=ActiveRows?blockIdx.x:0;task<total;task+=ActiveRows?gridDim.x:1) {
    const int pair = ActiveRows?active_plan[1+2*pairs+task/row_tiles]:int(blockIdx.y);
    const int row0=ActiveRows?int(task%row_tiles)*ROWS_PER_BLOCK:int(blockIdx.x)*ROWS_PER_BLOCK;
    if (pair >= pairs) continue;
    const int expert = ids[pair];
    if (static_cast<unsigned int>(expert) >=
        static_cast<unsigned int>(global_experts)) {
        continue;
    }
    const int pool = ActiveRows?active_plan[1+pair]:expert_pool[expert];
    const int local_expert = ActiveRows?active_plan[1+pairs+pair]:expert_local[expert];
    if (static_cast<unsigned int>(pool) >=
            static_cast<unsigned int>(pool_count) || local_expert < 0) {
        continue;
    }
    const int64_t * pointers =
        weight_ptrs + static_cast<size_t>(pool) * 5;
    const int64_t * sizes =
        weight_sizes + static_cast<size_t>(pool) * 3;
    const int32_t * params =
        pool_params + static_cast<size_t>(pool) * 7;
    if (local_expert >= params[0]) continue;
    const uint8_t * indices = reinterpret_cast<const uint8_t *>(
        static_cast<uintptr_t>(pointers[0]));
    const uint8_t * aux = reinterpret_cast<const uint8_t *>(
        static_cast<uintptr_t>(pointers[1]));
    const uint8_t * sub_scale = reinterpret_cast<const uint8_t *>(
        static_cast<uintptr_t>(pointers[2]));
    const float * neuron_scale = reinterpret_cast<const float *>(
        static_cast<uintptr_t>(pointers[3]));
    const int8_t * codebook = reinterpret_cast<const int8_t *>(
        static_cast<uintptr_t>(pointers[4]));
    const int ng = params[1];
    const int nvec = params[2];
    const int nsign = params[3];
    const int sub_bits = params[4];
    const int sign_mode = params[5];
    const int format = params[6];
    __shared__ float partial[ROWS_PER_BLOCK][NWARPS];

#define NVQ_MOE_HETERO_MMVQ_CASE(FORMAT_VALUE)                                 \
    case FORMAT_VALUE:                                                          \
        nvq_moe_mmvq_task<                                                      \
            FORMAT_VALUE, NWARPS, ROWS_PER_BLOCK, SHARE_GROUP_STATE>(           \
            indices, sizes[0], aux, sizes[1], sub_scale, sizes[2],              \
            neuron_scale, codebook, qx, xscale, output, &partial[0][0],         \
            pair, routes, out_per_expert, ng, nvec, nsign, sub_bits,           \
            sign_mode, routed_input, routed_input ? pair : pair / routes,       \
            local_expert, row0);                                              \
        break
    switch (format) {
        NVQ_MOE_HETERO_MMVQ_CASE(kNvq1L);
        NVQ_MOE_HETERO_MMVQ_CASE(kNvq2);
        NVQ_MOE_HETERO_MMVQ_CASE(kNvq3);
        NVQ_MOE_HETERO_MMVQ_CASE(kNvq2Exec);
        NVQ_MOE_HETERO_MMVQ_CASE(kNvq2Jsc);
        NVQ_MOE_HETERO_MMVQ_CASE(kNvq2JscExec);
        NVQ_MOE_HETERO_MMVQ_CASE(kNpq0L);
        NVQ_MOE_HETERO_MMVQ_CASE(kNvq1S);
        NVQ_MOE_HETERO_MMVQ_CASE(kNpq0S);
        NVQ_MOE_HETERO_MMVQ_CASE(kNvq3Jsc);
        NVQ_MOE_HETERO_MMVQ_CASE(kNvq3Jsc2);
        NVQ_MOE_HETERO_MMVQ_CASE(kNvq3Jsc512);
        NVQ_MOE_HETERO_MMVQ_CASE(kNvq2JscL);
        NVQ_MOE_HETERO_MMVQ_CASE(kNvq2JscXL);
        NVQ_MOE_HETERO_MMVQ_CASE(kNvq3JscL);
        NVQ_MOE_HETERO_MMVQ_CASE(kNvq2JscXLGroupExec);
        NVQ_MOE_HETERO_MMVQ_CASE(kNvq3JscLGroupExec);
    }
#undef NVQ_MOE_HETERO_MMVQ_CASE
    if constexpr(ActiveRows)__syncthreads();
    }
}

template<int Warps>
int nvq_active_resident_blocks() {
    int device=0;
    MFQ_RUNTIME_CHECK(cudaGetDevice(&device)==cudaSuccess,"NVQ active device query failed");
    static thread_local std::map<int,int> cached;
    if(const auto found=cached.find(device);found!=cached.end())return found->second;
    cudaDeviceProp properties{};int active=0;
    MFQ_RUNTIME_CHECK(cudaGetDeviceProperties(&properties,device)==cudaSuccess,
        "NVQ active device properties unavailable");
    MFQ_RUNTIME_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,
        nvq_moe_mmvq_hetero_kernel<Warps,2,false,true>,Warps*32,0)==cudaSuccess && active>0,
        "NVQ active kernel occupancy unavailable");
    const int blocks=active*properties.multiProcessorCount;
    cached.emplace(device,blocks);return blocks;
}

template <int FORMAT, int PHYSICAL_WARPS, int LOGICAL_WARPS, int ROWS_PER_BLOCK>
__global__ void __launch_bounds__(PHYSICAL_WARPS * 32)
nvq_moe_mmvq_exact_reduction_kernel(
    const uint8_t * indices,
    int64_t indices_nbytes,
    const uint8_t * aux,
    int64_t aux_nbytes,
    const uint8_t * sub_scale,
    int64_t sub_scale_nbytes,
    const float * neuron_scale,
    const int8_t * codebook,
    const int8_t * qx,
    const float * xscale,
    const int32_t * ids,
    const int32_t * expert_local,
    __half * output,
    int pairs,
    int routes,
    int global_experts,
    int pool_experts,
    int out_per_expert,
    int ng,
    int nvec,
    int nsign,
    int sub_bits,
    int sign_mode,
    bool routed_input) {
    static_assert(LOGICAL_WARPS % PHYSICAL_WARPS == 0);
    constexpr int kAccumulators = LOGICAL_WARPS / PHYSICAL_WARPS;
    const int pair = blockIdx.y;
    const int row0 = blockIdx.x * ROWS_PER_BLOCK;
    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    if (pair >= pairs) return;
    const int expert = ids[pair];
    if (static_cast<unsigned int>(expert) >=
        static_cast<unsigned int>(global_experts)) {
        return;
    }
    const int local_expert = expert_local[expert];
    if (static_cast<unsigned int>(local_expert) >=
        static_cast<unsigned int>(pool_experts)) {
        return;
    }
    const int source_row = routed_input ? pair : pair / routes;

    float acc[ROWS_PER_BLOCK][kAccumulators];
#pragma unroll
    for (int row_local = 0; row_local < ROWS_PER_BLOCK; ++row_local) {
#pragma unroll
        for (int accumulator = 0; accumulator < kAccumulators; ++accumulator) {
            acc[row_local][accumulator] = 0.0f;
        }
    }
#pragma unroll
    for (int accumulator = 0; accumulator < kAccumulators; ++accumulator) {
        const int logical_warp = warp + accumulator * PHYSICAL_WARPS;
        for (int segment = logical_warp * 32 + lane;
             segment < nsign;
             segment += LOGICAL_WARPS * 32) {
            const int group = segment / 3;
            const int segment_in_group = segment - group * 3;
            const int k = group * kGroupSize + segment_in_group * 8;
            const int8_t * activation =
                qx + (static_cast<int64_t>(source_row) * ng * kGroupSize + k);
            const float activation_scale =
                xscale[static_cast<int64_t>(source_row) * ng + group];
#pragma unroll
            for (int row_local = 0; row_local < ROWS_PER_BLOCK; ++row_local) {
                const int local_row = row0 + row_local;
                if (local_row >= out_per_expert) continue;
                const int row = local_expert * out_per_expert + local_row;
                const int64_t sub_linear =
                    static_cast<int64_t>(row) * ng + group;
                const uint32_t sub = load_nvq_state<FORMAT>(indices,indices_nbytes,sub_scale,sub_scale_nbytes,
                    sub_linear,sub_bits,ng);
                const auto weight = load_nvq_vec8<FORMAT>(
                    indices, indices_nbytes, aux, aux_nbytes, codebook,
                    row, segment, group, ng, nvec, nsign, sign_mode, sub);
                const float weight_scale =
                    (FORMAT == kNvq1L || FORMAT == kNvq1S)
                    ? neuron_scale[row] * static_cast<float>(sub)
                    : format_scale<FORMAT>(neuron_scale[row], sub, codebook);
                acc[row_local][accumulator] = fmaf(
                    weight_scale * activation_scale,
                    dot_nvq_vec8<FORMAT>(weight, activation),
                    acc[row_local][accumulator]);
            }
        }
    }

#pragma unroll
    for (int row_local = 0; row_local < ROWS_PER_BLOCK; ++row_local) {
#pragma unroll
        for (int accumulator = 0; accumulator < kAccumulators; ++accumulator) {
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                acc[row_local][accumulator] += __shfl_xor_sync(
                    0xffffffff, acc[row_local][accumulator], offset);
            }
        }
    }
    __shared__ float partial[ROWS_PER_BLOCK][LOGICAL_WARPS];
    if (lane == 0) {
#pragma unroll
        for (int row_local = 0; row_local < ROWS_PER_BLOCK; ++row_local) {
#pragma unroll
            for (int accumulator = 0; accumulator < kAccumulators; ++accumulator) {
                const int logical_warp = warp + accumulator * PHYSICAL_WARPS;
                partial[row_local][logical_warp] =
                    acc[row_local][accumulator];
            }
        }
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int row_local = 0; row_local < ROWS_PER_BLOCK; ++row_local) {
            float value = lane < LOGICAL_WARPS
                ? partial[row_local][lane]
                : 0.0f;
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                value += __shfl_xor_sync(0xffffffff, value, offset);
            }
            const int local_row = row0 + row_local;
            if (lane == 0 && local_row < out_per_expert) {
                output[static_cast<int64_t>(pair) * out_per_expert + local_row] =
                    __float2half(value);
            }
        }
    }
}

template <int FORMAT, int TILE_M>
__global__ void __launch_bounds__(128) nvq_moe_grouped_tile_kernel(
    const uint8_t * indices,
    int64_t indices_nbytes,
    const uint8_t * aux,
    int64_t aux_nbytes,
    const uint8_t * sub_scale,
    int64_t sub_scale_nbytes,
    const float * neuron_scale,
    const int8_t * codebook,
    const int8_t * qx,
    const float * xscale,
    const int32_t * expert_local,
    const int32_t * ids_dst,
    const int32_t * expert_bounds,
    const int32_t * tile_bounds,
    const int32_t * tile_experts,
    __half * output,
    int routes,
    int global_experts,
    int pool_experts,
    int out_per_expert,
    int ng,
    int nvec,
    int nsign,
    int sub_bits,
    int sign_mode,
    int max_tiles,
    int row_tiles,
    bool routed_input) {
    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    const int64_t max_tasks = static_cast<int64_t>(max_tiles) * row_tiles;
    for (int64_t task = blockIdx.x; task < max_tasks; task += gridDim.x) {
        const int row_tile = static_cast<int>(task % row_tiles);
        const int tile = static_cast<int>(task / row_tiles);
        if (tile >= tile_bounds[global_experts]) continue;
        const int expert = tile_experts[tile];
        const int local_expert = expert_local[expert];
        if (static_cast<unsigned int>(local_expert) >=
            static_cast<unsigned int>(pool_experts)) {
            continue;
        }
        const int local_tile = tile - tile_bounds[expert];
        const int first = expert_bounds[expert] + local_tile * TILE_M;
        const int last = min(first + TILE_M, expert_bounds[expert + 1]);
        const int local_row = row_tile * 4 + warp;
        if (local_row >= out_per_expert) continue;
        const int row = local_expert * out_per_expert + local_row;

        float acc[TILE_M];
#pragma unroll
        for (int item = 0; item < TILE_M; ++item) acc[item] = 0.0f;
        for (int segment = lane; segment < nsign; segment += 32) {
            const int group = segment / 3;
            const int segment_in_group = segment - group * 3;
            const int k = group * kGroupSize + segment_in_group * 8;
            const int64_t sub_linear = static_cast<int64_t>(row) * ng + group;
            const uint32_t sub = load_nvq_state<FORMAT>(indices,indices_nbytes,sub_scale,sub_scale_nbytes,
                    sub_linear,sub_bits,ng);
            const auto weight = load_nvq_vec8<FORMAT>(
                indices, indices_nbytes, aux, aux_nbytes, codebook,
                row, segment, group, ng, nvec, nsign, sign_mode, sub);
            const float weight_scale = (FORMAT == kNvq1L || FORMAT == kNvq1S)
                ? neuron_scale[row] * static_cast<float>(sub)
                : format_scale<FORMAT>(neuron_scale[row], sub, codebook);
#pragma unroll
            for (int item = 0; item < TILE_M; ++item) {
                const int compact = first + item;
                if (compact >= last) continue;
                const int pair = ids_dst[compact];
                const int source_row = routed_input ? pair : pair / routes;
                const int8_t * activation =
                    qx + (static_cast<int64_t>(source_row) * ng * kGroupSize + k);
                acc[item] = fmaf(
                    weight_scale *
                        xscale[static_cast<int64_t>(source_row) * ng + group],
                    dot_nvq_vec8<FORMAT>(weight, activation),
                    acc[item]);
            }
        }
#pragma unroll
        for (int item = 0; item < TILE_M; ++item) {
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                acc[item] += __shfl_xor_sync(0xffffffff, acc[item], offset);
            }
            if (lane == 0 && first + item < last) {
                const int pair = ids_dst[first + item];
                output[static_cast<int64_t>(pair) * out_per_expert + local_row] =
                    __float2half(acc[item]);
            }
        }
    }
}

template <int FORMAT, int NWARPS, int ROWS_PER_BLOCK>
__global__ void __launch_bounds__(NWARPS * 32) nepq_moe_mmvq_kernel(
    const uint8_t * indices,
    int64_t indices_nbytes,
    const uint8_t * aux,
    int64_t aux_nbytes,
    const uint8_t * sub_scale,
    int64_t sub_scale_nbytes,
    const float * neuron_scale,
    const int8_t * table_pool,
    const uint8_t * bank_ids,
    const int8_t * qx,
    const float * xscale,
    const int32_t * ids,
    const int32_t * expert_local,
    __half * output,
    int pairs,
    int routes,
    int global_experts,
    int pool_experts,
    int out_per_expert,
    int ng,
    int nvec,
    int nsign,
    int nsuper,
    int table_stride,
    int sub_bits,
    bool routed_input) {
    const int pair = blockIdx.y;
    const int row0 = blockIdx.x * ROWS_PER_BLOCK;
    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    if (pair >= pairs) return;
    const int expert = ids[pair];
    if (static_cast<unsigned int>(expert) >=
        static_cast<unsigned int>(global_experts)) {
        return;
    }
    const int local_expert = expert_local == nullptr ? expert : expert_local[expert];
    if (static_cast<unsigned int>(local_expert) >=
        static_cast<unsigned int>(pool_experts)) {
        return;
    }
    const int source_row = routed_input ? pair : pair / routes;

    float acc[ROWS_PER_BLOCK];
#pragma unroll
    for (int row_local = 0; row_local < ROWS_PER_BLOCK; ++row_local) {
        acc[row_local] = 0.0f;
    }
    for (int segment = warp * 32 + lane;
         segment < nsign;
         segment += NWARPS * 32) {
        const int group = segment / 3;
        const int segment_in_group = segment - group * 3;
        const int k = group * kGroupSize + segment_in_group * 8;
        const int8_t * activation =
            qx + (static_cast<int64_t>(source_row) * ng * kGroupSize + k);
        const float activation_scale =
            xscale[static_cast<int64_t>(source_row) * ng + group];
#pragma unroll
        for (int row_local = 0; row_local < ROWS_PER_BLOCK; ++row_local) {
            const int local_row = row0 + row_local;
            if (local_row >= out_per_expert) continue;
            const int row = local_expert * out_per_expert + local_row;
            const int64_t sub_linear = static_cast<int64_t>(row) * ng + group;
            const uint32_t sub = load_nvq_state<FORMAT>(indices,indices_nbytes,sub_scale,sub_scale_nbytes,
                    sub_linear,sub_bits,ng);
            const int8_t * table = nepq_active_table(
                table_pool, bank_ids, row, group, nsuper, table_stride);
            const auto weight = load_nepq_vec8<FORMAT>(
                indices, indices_nbytes, aux, aux_nbytes, table,
                row, segment, group, ng, nvec, nsign, 0, sub);
            const float weight_scale = (FORMAT == kNvq1L || FORMAT == kNvq1S)
                ? neuron_scale[row] * static_cast<float>(sub)
                : format_scale<FORMAT>(neuron_scale[row], sub, table);
            acc[row_local] = fmaf(
                weight_scale * activation_scale,
                dot_nvq_vec8<FORMAT>(weight, activation),
                acc[row_local]);
        }
    }

#pragma unroll
    for (int row_local = 0; row_local < ROWS_PER_BLOCK; ++row_local) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc[row_local] += __shfl_xor_sync(0xffffffff, acc[row_local], offset);
        }
    }
    __shared__ float partial[ROWS_PER_BLOCK][NWARPS];
    if (lane == 0) {
#pragma unroll
        for (int row_local = 0; row_local < ROWS_PER_BLOCK; ++row_local) {
            partial[row_local][warp] = acc[row_local];
        }
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int row_local = 0; row_local < ROWS_PER_BLOCK; ++row_local) {
            float value = lane < NWARPS ? partial[row_local][lane] : 0.0f;
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                value += __shfl_xor_sync(0xffffffff, value, offset);
            }
            const int local_row = row0 + row_local;
            if (lane == 0 && local_row < out_per_expert) {
                output[static_cast<int64_t>(pair) * out_per_expert + local_row] =
                    __float2half(value);
            }
        }
    }
}

template <int FORMAT, int TILE_M>
__global__ void __launch_bounds__(128) nepq_moe_grouped_tile_kernel(
    const uint8_t * indices,
    int64_t indices_nbytes,
    const uint8_t * aux,
    int64_t aux_nbytes,
    const uint8_t * sub_scale,
    int64_t sub_scale_nbytes,
    const float * neuron_scale,
    const int8_t * table_pool,
    const uint8_t * bank_ids,
    const int8_t * qx,
    const float * xscale,
    const int32_t * expert_local,
    const int32_t * ids_dst,
    const int32_t * expert_bounds,
    const int32_t * tile_bounds,
    const int32_t * tile_experts,
    __half * output,
    int routes,
    int global_experts,
    int pool_experts,
    int out_per_expert,
    int ng,
    int nvec,
    int nsign,
    int nsuper,
    int table_stride,
    int sub_bits,
    int max_tiles,
    int row_tiles,
    bool routed_input) {
    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    const int64_t max_tasks = static_cast<int64_t>(max_tiles) * row_tiles;
    for (int64_t task = blockIdx.x; task < max_tasks; task += gridDim.x) {
        const int row_tile = static_cast<int>(task % row_tiles);
        const int tile = static_cast<int>(task / row_tiles);
        if (tile >= tile_bounds[global_experts]) continue;
        const int expert = tile_experts[tile];
        const int local_expert =
            expert_local == nullptr ? expert : expert_local[expert];
        if (static_cast<unsigned int>(local_expert) >=
            static_cast<unsigned int>(pool_experts)) {
            continue;
        }
        const int local_tile = tile - tile_bounds[expert];
        const int first = expert_bounds[expert] + local_tile * TILE_M;
        const int last = min(first + TILE_M, expert_bounds[expert + 1]);
        const int local_row = row_tile * 4 + warp;
        if (local_row >= out_per_expert) continue;
        const int row = local_expert * out_per_expert + local_row;

        float acc[TILE_M];
#pragma unroll
        for (int item = 0; item < TILE_M; ++item) acc[item] = 0.0f;
        for (int segment = lane; segment < nsign; segment += 32) {
            const int group = segment / 3;
            const int segment_in_group = segment - group * 3;
            const int k = group * kGroupSize + segment_in_group * 8;
            const int64_t sub_linear = static_cast<int64_t>(row) * ng + group;
            const uint32_t sub = load_nvq_state<FORMAT>(indices,indices_nbytes,sub_scale,sub_scale_nbytes,
                    sub_linear,sub_bits,ng);
            const int8_t * table = nepq_active_table(
                table_pool, bank_ids, row, group, nsuper, table_stride);
            const auto weight = load_nvq_vec8<FORMAT>(
                indices, indices_nbytes, aux, aux_nbytes, table,
                row, segment, group, ng, nvec, nsign, 0, sub);
            const float weight_scale = (FORMAT == kNvq1L || FORMAT == kNvq1S)
                ? neuron_scale[row] * static_cast<float>(sub)
                : format_scale<FORMAT>(neuron_scale[row], sub, table);
#pragma unroll
            for (int item = 0; item < TILE_M; ++item) {
                const int compact = first + item;
                if (compact >= last) continue;
                const int pair = ids_dst[compact];
                const int source_row = routed_input ? pair : pair / routes;
                const int8_t * activation =
                    qx + (static_cast<int64_t>(source_row) * ng * kGroupSize + k);
                acc[item] = fmaf(
                    weight_scale * xscale[static_cast<int64_t>(source_row) * ng + group],
                    dot_nvq_vec8<FORMAT>(weight, activation),
                    acc[item]);
            }
        }
#pragma unroll
        for (int item = 0; item < TILE_M; ++item) {
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                acc[item] += __shfl_xor_sync(0xffffffff, acc[item], offset);
            }
            if (lane == 0 && first + item < last) {
                const int pair = ids_dst[first + item];
                output[static_cast<int64_t>(pair) * out_per_expert + local_row] =
                    __float2half(acc[item]);
            }
        }
    }
}

#endif

void check_nepq_common(
    const mfq_tensor_backend::Tensor & indices,
    const mfq_tensor_backend::Tensor & aux,
    const mfq_tensor_backend::Tensor & sub_scale,
    const mfq_tensor_backend::Tensor & neuron_scale,
    const mfq_tensor_backend::Tensor & table_pool,
    const mfq_tensor_backend::Tensor & bank_ids,
    int64_t neuron_len,
    int64_t sub_bits,
    int64_t format) {
    MFQ_RUNTIME_CHECK(
        format == kNvq1L || format == kNpq0L ||
        format == kNvq1S || format == kNpq0S,
        "NEPQ base format must be NVQ1-L/NPQ0-L/NVQ1-S/NPQ0-S");
    MFQ_RUNTIME_CHECK(neuron_len > 0 && neuron_len % 8 == 0,
                "NEPQ neuron_len must be a positive multiple of 8");
    const int expected_sub_bits = format == kNvq1S ? 4 :
        (format == kNpq0S ? 2 : 3);
    MFQ_RUNTIME_CHECK(sub_bits == expected_sub_bits, "NEPQ state width mismatch");
    for (const auto * tensor : {&indices, &aux, &sub_scale, &bank_ids}) {
        MFQ_RUNTIME_CHECK(tensor->is_cuda() && tensor->scalar_type() == mfq_tensor_backend::kUInt8 &&
                    tensor->is_contiguous(),
                    "NEPQ packed streams must be CUDA contiguous uint8");
    }
    MFQ_RUNTIME_CHECK(neuron_scale.is_cuda() && neuron_scale.scalar_type() == mfq_tensor_backend::kFloat32 &&
                neuron_scale.is_contiguous() && neuron_scale.dim() == 1,
                "NEPQ neuron_scale must be CUDA contiguous float32 rank-1");
    MFQ_RUNTIME_CHECK(table_pool.is_cuda() && table_pool.scalar_type() == mfq_tensor_backend::kInt8 &&
                table_pool.is_contiguous() && table_pool.dim() == 2 &&
                table_pool.size(0) >= 1 && table_pool.size(0) <= 256,
                "NEPQ table pool must be CUDA contiguous int8 [1..256,stride]");
    const int table_stride = format == kNvq1L ? 2048 * 8 :
        (format == kNvq1S ? 1024 * 8 :
         (format == kNpq0L ? kNpq0LTableBytes : kNepq0SCompactTableBytes));
    MFQ_RUNTIME_CHECK(table_pool.size(1) == table_stride, "NEPQ table stride mismatch");
    const int64_t N = neuron_scale.numel();
    const int64_t ng = (neuron_len + kGroupSize - 1) / kGroupSize;
    const int64_t nvec = neuron_len / 8;
    const int64_t nsuper = (ng + kNepqGroupsPerSupergroup - 1) /
        kNepqGroupsPerSupergroup;
    MFQ_RUNTIME_CHECK(bank_ids.dim() == 2 && bank_ids.size(0) == N &&
                bank_ids.size(1) == nsuper,
                "NEPQ bank selectors must have shape [rows,ceil(groups/4)]");
    const int index_bits = format == kNvq1L ? 11 :
        (format == kNvq1S ? 9 : (format == kNpq0L ? 7 : 6));
    const int64_t index_bytes = (N * nvec * index_bits + 7) / 8;
    const bool delta_format = format == kNvq1L || format == kNvq1S;
    const int64_t aux_bytes = delta_format ? (N * ng + 7) / 8 : 0;
    const int64_t sub_bytes = (N * ng * sub_bits + 7) / 8;
    MFQ_RUNTIME_CHECK(indices.numel() == index_bytes, "NEPQ index stream length mismatch");
    MFQ_RUNTIME_CHECK(aux.numel() == aux_bytes, "NEPQ aux stream length mismatch");
    MFQ_RUNTIME_CHECK(sub_scale.numel() == sub_bytes, "NEPQ state stream length mismatch");
}

void check_common(
    const mfq_tensor_backend::Tensor & indices,
    const mfq_tensor_backend::Tensor & aux,
    const mfq_tensor_backend::Tensor & sub_scale,
    const mfq_tensor_backend::Tensor & neuron_scale,
    const mfq_tensor_backend::Tensor & codebook,
    int64_t neuron_len,
    int64_t gs,
    int64_t sub_bits,
    int64_t format,
    int64_t sign_mode,bool allow_dense=false) {
    MFQ_RUNTIME_CHECK(format >= kNvq1L && format <= kNvq3JscLGroupExec,
                "NVQ format must be in [1,17]");
    MFQ_RUNTIME_CHECK(gs == kGroupSize, "NVQ CUDA kernels currently require gs=24");
    MFQ_RUNTIME_CHECK(sub_bits >= 1 && sub_bits <= 8, "NVQ sub_bits must be in [1,8]");
    MFQ_RUNTIME_CHECK(neuron_len > 0, "NVQ neuron_len must be positive");
    MFQ_RUNTIME_CHECK(sign_mode == 0 ||
                (sign_mode == 1 && (format == kNvq2 || format == kNvq2Exec)),
                "index-parity sign mode is valid only for NVQ2");
    for (const auto * tensor : {&indices, &aux, &sub_scale}) {
        MFQ_RUNTIME_CHECK(tensor->is_cuda() && tensor->scalar_type() == mfq_tensor_backend::kUInt8 && tensor->is_contiguous(),
                    "NVQ packed streams must be CUDA contiguous uint8");
    }
    MFQ_RUNTIME_CHECK(neuron_scale.is_cuda() && neuron_scale.scalar_type() == mfq_tensor_backend::kFloat32 &&
                neuron_scale.is_contiguous(), "NVQ neuron_scale must be CUDA contiguous float32");
    MFQ_RUNTIME_CHECK(codebook.is_cuda() && codebook.scalar_type() == mfq_tensor_backend::kInt8 && codebook.is_contiguous(),
                "NVQ codebook must be CUDA contiguous int8");
    if (format == kNvq1L) {
        MFQ_RUNTIME_CHECK(codebook.dim() == 2 && codebook.size(0) == 2048 && codebook.size(1) == 8,
                    "NVQ1-L codebook must have shape [2048,8]");
    } else if (format == kNvq1S) {
        MFQ_RUNTIME_CHECK(codebook.dim() == 2 && codebook.size(0) == 1024 && codebook.size(1) == 8,
                    "NVQ1-S codebook must have shape [1024,8]");
        MFQ_RUNTIME_CHECK(sub_bits == 4 && sign_mode == 0,
                    "NVQ1-S requires a 4-bit scale and no sign mode");
    } else if (format == kNvq2 || format == kNvq2Exec) {
        MFQ_RUNTIME_CHECK(codebook.dim() == 2 && codebook.size(0) == 256 && codebook.size(1) == 8,
                    "NVQ2 codebook must have shape [256,8]");
    } else if (format == kNvq2Jsc || format == kNvq2JscExec) {
        MFQ_RUNTIME_CHECK(codebook.dim() == 1 &&
                    (codebook.numel() == kJscMetadataBytes + kJscE8CodebookBytes ||
                     codebook.numel() == kJscMetadataBytes + 2 * kJscE8CodebookBytes ||
                     codebook.numel() == kJscMetadataBytes + 4 * kJscE8CodebookBytes),
                    "NVQ2J codebook metadata has an invalid size");
        MFQ_RUNTIME_CHECK(sub_bits == 4 && sign_mode == 0,
                    "NVQ2J requires a 4-bit state and even-parity signs");
    } else if (
        format == kNvq2JscL || format == kNvq2JscXL ||
        format == kNvq2JscXLGroupExec) {
        const int e8_bank_bytes = format == kNvq2JscL
            ? kJscE8Codebook1024Bytes
            : kJscE8Codebook4096Bytes;
        MFQ_RUNTIME_CHECK(codebook.dim() == 1 &&
                    (codebook.numel() == kJscMetadataBytes + e8_bank_bytes ||
                     codebook.numel() == kJscMetadataBytes + 2 * e8_bank_bytes ||
                     codebook.numel() == kJscMetadataBytes + 4 * e8_bank_bytes),
                    "extended NVQ2J codebook metadata has an invalid size");
        MFQ_RUNTIME_CHECK(sub_bits == 4 && sign_mode == 0,
                    "extended NVQ2J requires a 4-bit state and even-parity signs");
    } else if (
        format == kNvq3Jsc || format == kNvq3Jsc2 ||
        format == kNvq3Jsc512 || format == kNvq3JscL ||
        format == kNvq3JscLGroupExec) {
        const int d4_bank_bytes = format == kNvq3Jsc512
            ? kJscD4Codebook512Bytes
            : (format == kNvq3JscL || format == kNvq3JscLGroupExec
               ? kJscD4Codebook1024Bytes
               : kJscD4CodebookBytes);
        MFQ_RUNTIME_CHECK(codebook.dim() == 1 &&
                    (codebook.numel() == kJscMetadataBytes + d4_bank_bytes ||
                     codebook.numel() == kJscMetadataBytes + 2 * d4_bank_bytes ||
                     codebook.numel() == kJscMetadataBytes + 4 * d4_bank_bytes),
                    "NVQ3J codebook metadata has an invalid size");
        if (format == kNvq3Jsc2) {
            MFQ_RUNTIME_CHECK(
                codebook.numel() == kJscMetadataBytes + 2 * kJscD4CodebookBytes,
                "NVQ3J analytic-2 requires exactly two D4 banks");
        }
        MFQ_RUNTIME_CHECK(sub_bits == 4 && sign_mode == 0,
                    "NVQ3J requires a 4-bit state and even-parity signs");
    } else if (format == kNpq0L) {
        MFQ_RUNTIME_CHECK(codebook.dim() == 1 && codebook.numel() == kNpq0LTableBytes,
                    "NPQ0-L codebook metadata must contain 832 bytes");
        MFQ_RUNTIME_CHECK(sub_bits == 3 && sign_mode == 0,
                    "NPQ0-L requires a 3-bit state and no sign stream");
    } else if (format == kNpq0S) {
        MFQ_RUNTIME_CHECK(codebook.dim() == 1 && codebook.numel() == kNpq0STableBytes,
                    "NPQ0-S expanded PQ3+3 runtime LUT must contain 2112 bytes");
        MFQ_RUNTIME_CHECK(sub_bits == 2 && sign_mode == 0,
                    "NPQ0-S requires a 2-bit state and no sign stream");
    } else {
        MFQ_RUNTIME_CHECK(codebook.dim() == 2 && codebook.size(0) == 256 && codebook.size(1) == 4,
                    "NVQ3 codebook must have shape [256,4]");
    }

    const int64_t N = neuron_scale.numel();
    const int64_t ng = (neuron_len + gs - 1) / gs;
    const int64_t nvec = (neuron_len + (is_d4_format(format) ? 3 : 7)) /
                         (is_d4_format(format) ? 4 : 8);
    const int64_t nsign = (neuron_len + 7) / 8;
    const bool exec_layout = format == kNvq2Exec || format == kNvq2JscExec;
    const bool group_exec_layout =
        format == kNvq2JscXLGroupExec || format == kNvq3JscLGroupExec;
    const int index_bits = format_index_bits(format);
    const bool dense=allow_dense && dense_nvq_format(int(format)) && aux.numel()==0 && sub_scale.numel()==0;
    if(dense) {
        MFQ_RUNTIME_CHECK(sub_bits==4 && (!is_d4_format(int(format)) || nvec==2*nsign),
            "compact NVQ requires complete D4 index pairs and 4-bit state");
        const int64_t bytes=(N*(nvec*index_bits+nsign*7+ng*4)+7)/8;
        MFQ_RUNTIME_CHECK(indices.numel()==bytes,"compact NVQ stream length mismatch");
        return;
    }
    const bool nvq1_records=(format==kNvq1L || format==kNvq1S) && aux.numel()==0;
    if(nvq1_records)MFQ_RUNTIME_CHECK(sub_bits==(format==kNvq1S?4:3),"NVQ1 record state width mismatch");
    const int64_t index_bytes = nvq1_records ? N*ng*(format==kNvq1S?4:5) : group_exec_layout
        ? N * ng * (format == kNvq2JscXLGroupExec ? 8 : 12)
        : (N * nvec * index_bits + 7) / 8;
    const bool delta_format = format == kNvq1L || format == kNvq1S;
    const bool no_aux_format = format == kNpq0L || format == kNpq0S;
    const int aux_bits = delta_format ? 1 : (no_aux_format ? 0 : 7);
    const int64_t aux_count = delta_format ? N * ng :
        (no_aux_format ? 0 : N * nsign);
    const int64_t aux_bytes = exec_layout || group_exec_layout || nvq1_records
        ? 0
        : (aux_count * aux_bits + 7) / 8;
    const int64_t sub_bytes = nvq1_records || (group_exec_layout && sub_scale.numel()==0)?
        0:(N * ng * sub_bits + 7) / 8;
    MFQ_RUNTIME_CHECK(indices.numel() == index_bytes, "NVQ index stream length mismatch");
    MFQ_RUNTIME_CHECK(aux.numel() == aux_bytes, "NVQ aux stream length mismatch");
    MFQ_RUNTIME_CHECK(sub_scale.numel() == sub_bytes, "NVQ sub-scale stream length mismatch");
}

template <typename Launch>
void launch_by_format(int format, Launch && launch) {
    switch (format) {
        case kNvq1L: launch(std::integral_constant<int, kNvq1L>{}); break;
        case kNvq2: launch(std::integral_constant<int, kNvq2>{}); break;
        case kNvq3: launch(std::integral_constant<int, kNvq3>{}); break;
        case kNvq2Exec: launch(std::integral_constant<int, kNvq2Exec>{}); break;
        case kNvq2Jsc: launch(std::integral_constant<int, kNvq2Jsc>{}); break;
        case kNvq2JscExec: launch(std::integral_constant<int, kNvq2JscExec>{}); break;
        case kNpq0L: launch(std::integral_constant<int, kNpq0L>{}); break;
        case kNvq1S: launch(std::integral_constant<int, kNvq1S>{}); break;
        case kNpq0S: launch(std::integral_constant<int, kNpq0S>{}); break;
        case kNvq3Jsc: launch(std::integral_constant<int, kNvq3Jsc>{}); break;
        case kNvq3Jsc2: launch(std::integral_constant<int, kNvq3Jsc2>{}); break;
        case kNvq3Jsc512:
            launch(std::integral_constant<int, kNvq3Jsc512>{});
            break;
        case kNvq2JscL:
            launch(std::integral_constant<int, kNvq2JscL>{});
            break;
        case kNvq2JscXL:
            launch(std::integral_constant<int, kNvq2JscXL>{});
            break;
        case kNvq3JscL:
            launch(std::integral_constant<int, kNvq3JscL>{});
            break;
        case kNvq2JscXLGroupExec:
            launch(std::integral_constant<int, kNvq2JscXLGroupExec>{});
            break;
        case kNvq3JscLGroupExec:
            launch(std::integral_constant<int, kNvq3JscLGroupExec>{});
            break;
        default: MFQ_RUNTIME_CHECK(false, "unsupported NVQ format");
    }
}

template <typename Launch>
void launch_nepq_by_format(int format, Launch && launch) {
    switch (format) {
        case kNvq1L: launch(std::integral_constant<int, kNvq1L>{}); break;
        case kNpq0L: launch(std::integral_constant<int, kNpq0L>{}); break;
        case kNvq1S: launch(std::integral_constant<int, kNvq1S>{}); break;
        case kNpq0S: launch(std::integral_constant<int, kNpq0S>{}); break;
        default: MFQ_RUNTIME_CHECK(false, "unsupported NEPQ base format");
    }
}

#if defined(MFQ_NVQ_DENSE)
int aligned_group_rows_per_block(int format) {
    const char * value = mfq::cuda::env_lookup(
        format == kNvq2JscXLGroupExec
            ? "MFQ_NVQ_GROUP_ROWS_E8"
            : "MFQ_NVQ_GROUP_ROWS_D4");
    if (value == nullptr) value = mfq::cuda::env_lookup("MFQ_NVQ_GROUP_ROWS");
    if (value != nullptr && value[0] == '4') return 4;
    if (value != nullptr && value[0] == '2') return 2;
    return 1;
}

int aligned_group_row_tile(int format) {
    const char * value = mfq::cuda::env_lookup(
        format == kNvq2JscXLGroupExec
            ? "MFQ_NVQ_ROW_TILE_E8"
            : "MFQ_NVQ_ROW_TILE_D4");
    if (value == nullptr) value = mfq::cuda::env_lookup("MFQ_NVQ_ROW_TILE");
    return value == nullptr ? 0 : std::atoi(value);
}

bool aligned_group_e8_down_w32r1_enabled() {
    static const bool value = [] {
        const char * text = mfq::cuda::env_lookup("MFQ_NVQ_E8_STAGE_DOWN_W32R1");
        return text == nullptr || text[0] != '0';
    }();
    return value;
}

template <
    int FORMAT, int NWARPS, int ROWS_PER_WARP, bool MULTI2,
    bool STAGE_E8_THREE_BANKS = false, bool ADD_RESIDUAL = false>
void launch_aligned_group_row_tiled_m1(
    const NvqDeviceWeight & first,
    const NvqDeviceWeight & second,
    const int8_t * qx,
    const float * xscale,
    __half * output,
    cudaStream_t stream,
    const __half * residual = nullptr) {
    static_assert(!ADD_RESIDUAL || !MULTI2);
    const int total_rows = MULTI2 ? first.N + second.N : first.N;
    const int rows_per_block = NWARPS * ROWS_PER_WARP;
    size_t shared_bytes = 0;
    if constexpr (STAGE_E8_THREE_BANKS) {
        shared_bytes =
            3 * kJscE8Codebook4096Bytes + kJscE8PartialBytes;
        static const cudaError_t attribute_status = cudaFuncSetAttribute(
            aligned_group_row_tiled_m1_kernel<
                FORMAT, NWARPS, ROWS_PER_WARP, MULTI2, true, ADD_RESIDUAL>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(shared_bytes));
        MFQ_RUNTIME_CHECK(
            attribute_status == cudaSuccess,
            "failed to opt in to the E8 row-tiled shared-memory size");
    } else {
        const int k_pad = first.ng * kGroupSize;
        const int scale_offset = (k_pad + 3) & ~3;
        shared_bytes = static_cast<size_t>(scale_offset) +
            static_cast<size_t>(first.ng) * sizeof(float);
        if constexpr (FORMAT == kNvq3JscLGroupExec) {
            shared_bytes += static_cast<size_t>(first.codebook_nbytes);
            if constexpr (MULTI2) {
                shared_bytes += static_cast<size_t>(second.codebook_nbytes);
            }
        }
    }
    aligned_group_row_tiled_m1_kernel<
        FORMAT, NWARPS, ROWS_PER_WARP, MULTI2,
        STAGE_E8_THREE_BANKS, ADD_RESIDUAL>
        <<<(total_rows + rows_per_block - 1) / rows_per_block,
           dim3(32, NWARPS), shared_bytes, stream>>>(
            first, second, qx, xscale, output, residual);
}

template <int FORMAT, bool ADD_RESIDUAL = false>
void launch_aligned_group_m1(
    const NvqDeviceWeight & weight,
    const int8_t * qx,
    const float * xscale,
    __half * output,
    cudaStream_t stream,
    const __half * residual = nullptr) {
    const int tile = aligned_group_row_tile(FORMAT);
    if constexpr (FORMAT == kNvq2JscXLGroupExec) {
        if (tile == 3243 && weight.codebook_nbytes >=
                kJscMetadataBytes + 4 * kJscE8Codebook4096Bytes) {
            if (weight.N >= 12288) {
                launch_aligned_group_row_tiled_m1<
                    FORMAT, 32, 4, false, true, ADD_RESIDUAL>(
                    weight, weight, qx, xscale, output, stream, residual);
                return;
            }
            if (weight.N >= 4096 && aligned_group_e8_down_w32r1_enabled()) {
                launch_aligned_group_row_tiled_m1<
                    FORMAT, 32, 1, false, true, ADD_RESIDUAL>(
                    weight, weight, qx, xscale, output, stream, residual);
                return;
            }
        }
        if (tile == 164) {
            launch_aligned_group_row_tiled_m1<
                FORMAT, 16, 4, false, false, ADD_RESIDUAL>(
                weight, weight, qx, xscale, output, stream, residual);
            return;
        }
        if (tile == 324) {
            launch_aligned_group_row_tiled_m1<
                FORMAT, 32, 4, false, false, ADD_RESIDUAL>(
                weight, weight, qx, xscale, output, stream, residual);
            return;
        }
    } else {
        if (tile == 162) {
            launch_aligned_group_row_tiled_m1<
                FORMAT, 16, 2, false, false, ADD_RESIDUAL>(
                weight, weight, qx, xscale, output, stream, residual);
            return;
        }
        if (tile == 322) {
            launch_aligned_group_row_tiled_m1<
                FORMAT, 32, 2, false, false, ADD_RESIDUAL>(
                weight, weight, qx, xscale, output, stream, residual);
            return;
        }
    }
    const int rows = aligned_group_rows_per_block(FORMAT);
    if (rows == 4) {
        aligned_group_gemv_m1_kernel<FORMAT, 4, 4, ADD_RESIDUAL>
            <<<(weight.N + 3) / 4, dim3(32, 16), 0, stream>>>(
                weight, qx, xscale, output, residual);
    } else if (rows == 2) {
        aligned_group_gemv_m1_kernel<FORMAT, 4, 2, ADD_RESIDUAL>
            <<<(weight.N + 1) / 2, dim3(32, 8), 0, stream>>>(
                weight, qx, xscale, output, residual);
    } else {
        aligned_group_gemv_m1_kernel<FORMAT, 4, 1, ADD_RESIDUAL>
            <<<weight.N, dim3(32, 4), 0, stream>>>(
                weight, qx, xscale, output, residual);
    }
}

template <int FORMAT>
void launch_aligned_group_multi2_m1(
    const NvqDeviceWeight & first,
    const NvqDeviceWeight & second,
    const int8_t * qx,
    const float * xscale,
    __half * output,
    cudaStream_t stream) {
    const int blocks = first.N + second.N;
    const int tile = aligned_group_row_tile(FORMAT);
    if constexpr (FORMAT == kNvq2JscXLGroupExec) {
        if (tile == 3243 &&
            (first.N >= 12288 || second.N >= 12288) &&
            first.codebook_nbytes >=
                kJscMetadataBytes + 4 * kJscE8Codebook4096Bytes &&
            second.codebook_nbytes >=
                kJscMetadataBytes + 4 * kJscE8Codebook4096Bytes) {
            launch_aligned_group_row_tiled_m1<FORMAT, 32, 4, true, true>(
                first, second, qx, xscale, output, stream);
            return;
        }
        if (tile == 164) {
            launch_aligned_group_row_tiled_m1<FORMAT, 16, 4, true>(
                first, second, qx, xscale, output, stream);
            return;
        }
        if (tile == 324) {
            launch_aligned_group_row_tiled_m1<FORMAT, 32, 4, true>(
                first, second, qx, xscale, output, stream);
            return;
        }
    } else {
        if (tile == 162) {
            launch_aligned_group_row_tiled_m1<FORMAT, 16, 2, true>(
                first, second, qx, xscale, output, stream);
            return;
        }
        if (tile == 322) {
            launch_aligned_group_row_tiled_m1<FORMAT, 32, 2, true>(
                first, second, qx, xscale, output, stream);
            return;
        }
    }
    const int rows = aligned_group_rows_per_block(FORMAT);
    if (rows == 4) {
        aligned_group_multi2_m1_kernel<FORMAT, 4, 4>
            <<<(blocks + 3) / 4, dim3(32, 16), 0, stream>>>(
                first, second, qx, xscale, output);
    } else if (rows == 2) {
        aligned_group_multi2_m1_kernel<FORMAT, 4, 2>
            <<<(blocks + 1) / 2, dim3(32, 8), 0, stream>>>(
                first, second, qx, xscale, output);
    } else {
        aligned_group_multi2_m1_kernel<FORMAT, 4, 1>
            <<<blocks, dim3(32, 4), 0, stream>>>(
                first, second, qx, xscale, output);
    }
}

template <int NWARPS>
void launch_m1_vec8_by_format(
    int format,
    const mfq_tensor_backend::Tensor & indices,
    const mfq_tensor_backend::Tensor & aux,
    const mfq_tensor_backend::Tensor & sub_scale,
    const mfq_tensor_backend::Tensor & neuron_scale,
    const mfq_tensor_backend::Tensor & codebook,
    const mfq_tensor_backend::Tensor & qx,
    const mfq_tensor_backend::Tensor & xscale,
    mfq_tensor_backend::Tensor & output,
    int N,
    int ng,
    int nvec,
    int nsign,
    int sub_bits,
    int sign_mode,
    cudaStream_t stream) {
    if (format == kNvq2 && sub_bits == 4) {
        nvq_gemv_m1_vec8_kernel<kNvq2, NWARPS, true><<<N, dim3(32, NWARPS), 0, stream>>>(
            indices.data_ptr<uint8_t>(), indices.numel(), aux.data_ptr<uint8_t>(), aux.numel(),
            sub_scale.data_ptr<uint8_t>(), sub_scale.numel(), neuron_scale.data_ptr<float>(),
            codebook.data_ptr<int8_t>(), qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),
            reinterpret_cast<__half *>(output.data_ptr<mfq_half>()),
            N, ng, nvec, nsign, sub_bits, sign_mode);
        return;
    }
    if (format == kNvq2Exec && sub_bits == 4) {
        nvq_gemv_m1_vec8_kernel<kNvq2Exec, NWARPS, true><<<N, dim3(32, NWARPS), 0, stream>>>(
            indices.data_ptr<uint8_t>(), indices.numel(), aux.data_ptr<uint8_t>(), aux.numel(),
            sub_scale.data_ptr<uint8_t>(), sub_scale.numel(), neuron_scale.data_ptr<float>(),
            codebook.data_ptr<int8_t>(), qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),
            reinterpret_cast<__half *>(output.data_ptr<mfq_half>()),
            N, ng, nvec, nsign, sub_bits, sign_mode);
        return;
    }
    if (format == kNvq2Jsc && sub_bits == 4) {
        nvq_gemv_m1_vec8_kernel<kNvq2Jsc, NWARPS, true><<<N, dim3(32, NWARPS), 0, stream>>>(
            indices.data_ptr<uint8_t>(), indices.numel(), aux.data_ptr<uint8_t>(), aux.numel(),
            sub_scale.data_ptr<uint8_t>(), sub_scale.numel(), neuron_scale.data_ptr<float>(),
            codebook.data_ptr<int8_t>(), qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),
            reinterpret_cast<__half *>(output.data_ptr<mfq_half>()),
            N, ng, nvec, nsign, sub_bits, sign_mode);
        return;
    }
    if (format == kNvq2JscExec && sub_bits == 4) {
        nvq_gemv_m1_vec8_kernel<kNvq2JscExec, NWARPS, true><<<N, dim3(32, NWARPS), 0, stream>>>(
            indices.data_ptr<uint8_t>(), indices.numel(), aux.data_ptr<uint8_t>(), aux.numel(),
            sub_scale.data_ptr<uint8_t>(), sub_scale.numel(), neuron_scale.data_ptr<float>(),
            codebook.data_ptr<int8_t>(), qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),
            reinterpret_cast<__half *>(output.data_ptr<mfq_half>()),
            N, ng, nvec, nsign, sub_bits, sign_mode);
        return;
    }
    if (format == kNvq3 && sub_bits == 4) {
        nvq_gemv_m1_vec8_kernel<kNvq3, NWARPS, true><<<N, dim3(32, NWARPS), 0, stream>>>(
            indices.data_ptr<uint8_t>(), indices.numel(), aux.data_ptr<uint8_t>(), aux.numel(),
            sub_scale.data_ptr<uint8_t>(), sub_scale.numel(), neuron_scale.data_ptr<float>(),
            codebook.data_ptr<int8_t>(), qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),
            reinterpret_cast<__half *>(output.data_ptr<mfq_half>()),
            N, ng, nvec, nsign, sub_bits, sign_mode);
        return;
    }
    if (format == kNvq3Jsc && sub_bits == 4) {
        nvq_gemv_m1_vec8_kernel<kNvq3Jsc, NWARPS, true><<<N, dim3(32, NWARPS), 0, stream>>>(
            indices.data_ptr<uint8_t>(), indices.numel(), aux.data_ptr<uint8_t>(), aux.numel(),
            sub_scale.data_ptr<uint8_t>(), sub_scale.numel(), neuron_scale.data_ptr<float>(),
            codebook.data_ptr<int8_t>(), qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),
            reinterpret_cast<__half *>(output.data_ptr<mfq_half>()),
            N, ng, nvec, nsign, sub_bits, sign_mode);
        return;
    }
    if (format == kNvq3Jsc2 && sub_bits == 4) {
        nvq3j2_gemv_m1_group_kernel<NWARPS><<<N, dim3(32, NWARPS), 0, stream>>>(
            indices.data_ptr<uint8_t>(), aux.data_ptr<uint8_t>(), aux.numel(),
            sub_scale.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(),
            codebook.data_ptr<int8_t>(), qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),
            reinterpret_cast<__half *>(output.data_ptr<mfq_half>()),
            N, ng, nvec, nsign);
        return;
    }
    if (format == kNvq2JscXLGroupExec || format == kNvq3JscLGroupExec) {
        const NvqDeviceWeight weight{
            indices.data_ptr<uint8_t>(), indices.numel(),
            aux.data_ptr<uint8_t>(), aux.numel(),
            sub_scale.data_ptr<uint8_t>(), sub_scale.numel(),
            neuron_scale.data_ptr<float>(), codebook.data_ptr<int8_t>(),
            static_cast<int>(codebook.numel()),
            N, ng, nvec, nsign, sub_bits, sign_mode};
        if (format == kNvq2JscXLGroupExec) {
            launch_aligned_group_m1<kNvq2JscXLGroupExec>(
                weight, qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),
                reinterpret_cast<__half *>(output.data_ptr<mfq_half>()), stream);
        } else {
            launch_aligned_group_m1<kNvq3JscLGroupExec>(
                weight, qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),
                reinterpret_cast<__half *>(output.data_ptr<mfq_half>()), stream);
        }
        return;
    }
    launch_by_format(format, [&](auto tag) {
        constexpr int F = decltype(tag)::value;
        nvq_gemv_m1_vec8_kernel<F, NWARPS><<<N, dim3(32, NWARPS), 0, stream>>>(
            indices.data_ptr<uint8_t>(), indices.numel(), aux.data_ptr<uint8_t>(), aux.numel(),
            sub_scale.data_ptr<uint8_t>(), sub_scale.numel(), neuron_scale.data_ptr<float>(),
            codebook.data_ptr<int8_t>(), qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),
            reinterpret_cast<__half *>(output.data_ptr<mfq_half>()),
            N, ng, nvec, nsign, sub_bits, sign_mode);
    });
}

template <int NWARPS, int MAX_M>
void launch_batch_vec8_by_format(
    int format,
    const mfq_tensor_backend::Tensor & indices,
    const mfq_tensor_backend::Tensor & aux,
    const mfq_tensor_backend::Tensor & sub_scale,
    const mfq_tensor_backend::Tensor & neuron_scale,
    const mfq_tensor_backend::Tensor & codebook,
    const mfq_tensor_backend::Tensor & qx,
    const mfq_tensor_backend::Tensor & xscale,
    mfq_tensor_backend::Tensor & output,
    int M,
    int N,
    int ng,
    int nvec,
    int nsign,
    int sub_bits,
    int sign_mode,
    cudaStream_t stream) {
    launch_by_format(format, [&](auto tag) {
        constexpr int F = decltype(tag)::value;
        nvq_gemv_batch_vec8_kernel<F, NWARPS, MAX_M><<<N, dim3(32, NWARPS), 0, stream>>>(
            indices.data_ptr<uint8_t>(), indices.numel(), aux.data_ptr<uint8_t>(), aux.numel(),
            sub_scale.data_ptr<uint8_t>(), sub_scale.numel(), neuron_scale.data_ptr<float>(),
            codebook.data_ptr<int8_t>(), qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),
            reinterpret_cast<__half *>(output.data_ptr<mfq_half>()),
            M, N, ng, nvec, nsign, sub_bits, sign_mode);
    });
}

bool launch_group_vec8(
    int format, const mfq_tensor_backend::Tensor& indices,
    const mfq_tensor_backend::Tensor& aux, const mfq_tensor_backend::Tensor& sub_scale,
    const mfq_tensor_backend::Tensor& neuron_scale, const mfq_tensor_backend::Tensor& codebook,
    const mfq_tensor_backend::Tensor& qx, const mfq_tensor_backend::Tensor& xscale,
    mfq_tensor_backend::Tensor& output,
    int M, int N, int ng, int nvec, int nsign, int sub_bits, int sign_mode, cudaStream_t stream) {
    if (M > 4 || ng > 128 || N < 128 || N > 4096 ||
        (sub_bits != 4 && format != kNvq1L) || mfq::cuda::env_lookup("MFQ_NVQ_DENSE_REFERENCE")) return false;
    if (M == 1 && ng > 32 &&
        (format == kNvq2Jsc || format == kNvq3Jsc || format == kNvq3JscL)) return false;
    const NvqDeviceWeight weight{indices.data_ptr<uint8_t>(), indices.numel(),
        aux.data_ptr<uint8_t>(), aux.numel(), sub_scale.data_ptr<uint8_t>(), sub_scale.numel(),
        neuron_scale.data_ptr<float>(), codebook.data_ptr<int8_t>(), int(codebook.numel()),
        N, ng, nvec, nsign, sub_bits, sign_mode};
    launch_by_format(format, [&](auto tag) {
        const auto launch = [&](auto rows, auto warps) {
            constexpr int rows_per_block = 4 / decltype(warps)::value;
            nvq_gemv_group_kernel<decltype(tag)::value, decltype(rows)::value, decltype(warps)::value><<<
                (N + rows_per_block - 1) / rows_per_block, dim3(32, 4), 0, stream>>>(
                weight, qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),
                reinterpret_cast<half*>(output.data_ptr<mfq_half>()), M);
        };
        const auto select_warps = [&](auto rows) {
            if (ng <= 32) launch(rows, std::integral_constant<int, 1>{});
            else launch(rows, std::integral_constant<int, 4>{});
        };
        if (M == 1) select_warps(std::integral_constant<int, 1>{});
        else if (M <= 2) select_warps(std::integral_constant<int, 2>{});
        else select_warps(std::integral_constant<int, 4>{});
    });
    return true;
}

void launch_selected_batch_vec8(
    int format,
    const mfq_tensor_backend::Tensor & indices,
    const mfq_tensor_backend::Tensor & aux,
    const mfq_tensor_backend::Tensor & sub_scale,
    const mfq_tensor_backend::Tensor & neuron_scale,
    const mfq_tensor_backend::Tensor & codebook,
    const mfq_tensor_backend::Tensor & qx,
    const mfq_tensor_backend::Tensor & xscale,
    mfq_tensor_backend::Tensor & output,
    int M,
    int N,
    int ng,
    int nvec,
    int nsign,
    int sub_bits,
    int sign_mode,
    cudaStream_t stream) {
#define NVQ_SELECTED_BATCH(NW, MAX_M)                                                   \
    launch_batch_vec8_by_format<NW, MAX_M>(                                            \
        format, indices, aux, sub_scale, neuron_scale, codebook, qx, xscale, output,  \
        M, N, ng, nvec, nsign, sub_bits, sign_mode, stream)
    if (launch_group_vec8(format, indices, aux, sub_scale, neuron_scale, codebook,
            qx, xscale, output, M, N, ng, nvec, nsign, sub_bits, sign_mode, stream)) return;
    const bool compact_s = format == kNvq1S || format == kNpq0S;
    const bool compact_s_small = compact_s && N <= 1024;
    if (M <= 2) {
        if (compact_s_small) {
            NVQ_SELECTED_BATCH(8, 2);
        } else if (compact_s) {
            NVQ_SELECTED_BATCH(4, 2);
        } else if (is_e8_format(format)) {
            if (N <= 1024) {
                NVQ_SELECTED_BATCH(8, 2);
            } else {
                NVQ_SELECTED_BATCH(4, 2);
            }
        } else if (is_d4_format(format)) {
            NVQ_SELECTED_BATCH(2, 2);
        } else {
            NVQ_SELECTED_BATCH(8, 2);
        }
    } else if (M <= 4) {
        if (compact_s_small) {
            NVQ_SELECTED_BATCH(8, 4);
        } else if (compact_s) {
            NVQ_SELECTED_BATCH(4, 4);
        } else if (is_e8_format(format)) {
            if (N <= 1024) {
                NVQ_SELECTED_BATCH(8, 4);
            } else {
                NVQ_SELECTED_BATCH(4, 4);
            }
        } else if (is_d4_format(format)) {
            NVQ_SELECTED_BATCH(4, 4);
        } else {
            NVQ_SELECTED_BATCH(8, 4);
        }
    } else if (M <= 8 && compact_s && (compact_s_small || ng >= 384)) {
        NVQ_SELECTED_BATCH(8, 8);
    } else if (M <= 8 && compact_s) {
        NVQ_SELECTED_BATCH(4, 8);
    } else if (M <= 8 && is_e8_format(format)) {
        if (N <= 1024) {
            NVQ_SELECTED_BATCH(8, 8);
        } else {
            NVQ_SELECTED_BATCH(4, 8);
        }
    } else if (M <= 8) {
        NVQ_SELECTED_BATCH(8, 8);
    } else if (M <= 12 && compact_s_small) {
        NVQ_SELECTED_BATCH(8, 12);
    } else if (M <= 12 && compact_s) {
        NVQ_SELECTED_BATCH(4, 12);
    } else if (M <= 12 && is_e8_format(format)) {
        if (M == 12 && N <= 1024) {
            NVQ_SELECTED_BATCH(8, 12);
        } else {
            NVQ_SELECTED_BATCH(4, 12);
        }
    } else if (M <= 12) {
        NVQ_SELECTED_BATCH(8, 12);
    } else if (compact_s_small) {
        NVQ_SELECTED_BATCH(8, 16);
    } else if (compact_s) {
        NVQ_SELECTED_BATCH(4, 16);
    } else if (is_e8_format(format)) {
        if (N <= 1024) {
            NVQ_SELECTED_BATCH(8, 16);
        } else {
            NVQ_SELECTED_BATCH(4, 16);
        }
    } else {
        NVQ_SELECTED_BATCH(8, 16);
    }
#undef NVQ_SELECTED_BATCH
}

void launch_selected_m1_vec8(
    int format,
    const mfq_tensor_backend::Tensor & indices,
    const mfq_tensor_backend::Tensor & aux,
    const mfq_tensor_backend::Tensor & sub_scale,
    const mfq_tensor_backend::Tensor & neuron_scale,
    const mfq_tensor_backend::Tensor & codebook,
    const mfq_tensor_backend::Tensor & qx,
    const mfq_tensor_backend::Tensor & xscale,
    mfq_tensor_backend::Tensor & output,
    int N,
    int ng,
    int nvec,
    int nsign,
    int sub_bits,
    int sign_mode,
    cudaStream_t stream) {
    if (launch_group_vec8(format, indices, aux, sub_scale, neuron_scale, codebook,
            qx, xscale, output, 1, N, ng, nvec, nsign, sub_bits, sign_mode, stream)) return;
    if (format == kNvq1S && N <= 2048) {
        launch_m1_vec8_by_format<8>(
            format, indices, aux, sub_scale, neuron_scale, codebook, qx, xscale,
            output, N, ng, nvec, nsign, sub_bits, sign_mode, stream);
    } else if (format == kNpq0S && N <= 1024) {
        launch_m1_vec8_by_format<2>(
            format, indices, aux, sub_scale, neuron_scale, codebook, qx, xscale,
            output, N, ng, nvec, nsign, sub_bits, sign_mode, stream);
    } else {
        launch_m1_vec8_by_format<4>(
            format, indices, aux, sub_scale, neuron_scale, codebook, qx, xscale,
            output, N, ng, nvec, nsign, sub_bits, sign_mode, stream);
    }
}

NvqDeviceWeight make_device_weight(
    const mfq_tensor_backend::Tensor & indices,
    const mfq_tensor_backend::Tensor & aux,
    const mfq_tensor_backend::Tensor & sub_scale,
    const mfq_tensor_backend::Tensor & neuron_scale,
    const mfq_tensor_backend::Tensor & codebook,
    int neuron_len,
    int gs,
    int sub_bits,
    int format,
    int sign_mode) {
    const int ng = (neuron_len + gs - 1) / gs;
    return NvqDeviceWeight{
        indices.data_ptr<uint8_t>(), indices.numel(),
        aux.data_ptr<uint8_t>(), aux.numel(),
        sub_scale.data_ptr<uint8_t>(), sub_scale.numel(),
        neuron_scale.data_ptr<float>(), codebook.data_ptr<int8_t>(),
        static_cast<int>(codebook.numel()),
        static_cast<int>(neuron_scale.numel()), ng,
        (neuron_len + (is_d4_format(format) ? 3 : 7)) /
            (is_d4_format(format) ? 4 : 8),
        (neuron_len + 7) / 8, sub_bits, sign_mode};
}

template <int FORMAT, int NWARPS, int MAX_M, bool FAST_NVQ2_S4 = false>
void launch_multi2_vec8(
    const NvqDeviceWeight & first,
    const NvqDeviceWeight & second,
    const mfq_tensor_backend::Tensor & qx,
    const mfq_tensor_backend::Tensor & xscale,
    mfq_tensor_backend::Tensor & output,
    int M,
    cudaStream_t stream) {
    nvq_gemv_multi2_vec8_kernel<FORMAT, NWARPS, MAX_M, FAST_NVQ2_S4>
        <<<first.N + second.N, dim3(32, NWARPS), 0, stream>>>(
            first, second, qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),
            reinterpret_cast<__half *>(output.data_ptr<mfq_half>()), M);
}

void launch_selected_multi2_vec8(
    int format,
    const NvqDeviceWeight & first,
    const NvqDeviceWeight & second,
    const mfq_tensor_backend::Tensor & qx,
    const mfq_tensor_backend::Tensor & xscale,
    mfq_tensor_backend::Tensor & output,
    int M,
    cudaStream_t stream) {
#define NVQ_MULTI2(F, NW, MAX_M) \
    launch_multi2_vec8<F, NW, MAX_M>(first, second, qx, xscale, output, M, stream)
#define NVQ_MULTI2_FAST(F, NW, MAX_M) \
    launch_multi2_vec8<F, NW, MAX_M, true>(first, second, qx, xscale, output, M, stream)
    const bool fast_nvq2_s4 =
        (format == kNvq2 || format == kNvq2Exec ||
         format == kNvq2Jsc || format == kNvq2JscExec) &&
        first.sub_bits == 4 && second.sub_bits == 4;
    if (M == 1) {
        switch (format) {
            case kNvq1L: NVQ_MULTI2(kNvq1L, 4, 1); break;
            case kNvq2:
                if (fast_nvq2_s4) NVQ_MULTI2_FAST(kNvq2, 4, 1);
                else NVQ_MULTI2(kNvq2, 4, 1);
                break;
            case kNvq3: NVQ_MULTI2(kNvq3, 4, 1); break;
            case kNvq2Exec:
                if (fast_nvq2_s4) NVQ_MULTI2_FAST(kNvq2Exec, 4, 1);
                else NVQ_MULTI2(kNvq2Exec, 4, 1);
                break;
            case kNvq2Jsc:
                if (fast_nvq2_s4) NVQ_MULTI2_FAST(kNvq2Jsc, 4, 1);
                else NVQ_MULTI2(kNvq2Jsc, 4, 1);
                break;
            case kNvq2JscExec:
                if (fast_nvq2_s4) NVQ_MULTI2_FAST(kNvq2JscExec, 4, 1);
                else NVQ_MULTI2(kNvq2JscExec, 4, 1);
                break;
            case kNpq0L: NVQ_MULTI2(kNpq0L, 4, 1); break;
            case kNvq1S: NVQ_MULTI2(kNvq1S, 4, 1); break;
            case kNpq0S: NVQ_MULTI2(kNpq0S, 4, 1); break;
            case kNvq3Jsc: NVQ_MULTI2(kNvq3Jsc, 4, 1); break;
            case kNvq3Jsc2: NVQ_MULTI2(kNvq3Jsc2, 4, 1); break;
            case kNvq3Jsc512: NVQ_MULTI2(kNvq3Jsc512, 4, 1); break;
            case kNvq2JscL: NVQ_MULTI2(kNvq2JscL, 4, 1); break;
            case kNvq2JscXL: NVQ_MULTI2(kNvq2JscXL, 4, 1); break;
            case kNvq3JscL: NVQ_MULTI2(kNvq3JscL, 4, 1); break;
            case kNvq2JscXLGroupExec:
                launch_aligned_group_multi2_m1<kNvq2JscXLGroupExec>(
                    first, second, qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),
                    reinterpret_cast<__half *>(output.data_ptr<mfq_half>()), stream);
                break;
            case kNvq3JscLGroupExec:
                launch_aligned_group_multi2_m1<kNvq3JscLGroupExec>(
                    first, second, qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),
                    reinterpret_cast<__half *>(output.data_ptr<mfq_half>()), stream);
                break;
        }
    } else if (M <= 4) {
        if (format == kNvq1L || format == kNvq1S) {
            if (format == kNvq1S) NVQ_MULTI2(kNvq1S, 8, 4);
            else NVQ_MULTI2(kNvq1L, 8, 4);
        } else if (is_e8_format(format)) {
            if (format == kNvq2Exec) NVQ_MULTI2(kNvq2Exec, 4, 4);
            else if (format == kNvq2Jsc) NVQ_MULTI2(kNvq2Jsc, 4, 4);
            else if (format == kNvq2JscExec) NVQ_MULTI2(kNvq2JscExec, 4, 4);
            else if (format == kNvq2JscL) NVQ_MULTI2(kNvq2JscL, 4, 4);
            else if (format == kNvq2JscXL) NVQ_MULTI2(kNvq2JscXL, 4, 4);
            else if (format == kNvq2JscXLGroupExec) {
                NVQ_MULTI2(kNvq2JscXLGroupExec, 4, 4);
            }
            else NVQ_MULTI2(kNvq2, 4, 4);
        } else if (format == kNpq0L || format == kNpq0S) {
            if (format == kNpq0S) NVQ_MULTI2(kNpq0S, 4, 4);
            else NVQ_MULTI2(kNpq0L, 4, 4);
        } else if (M <= 2) {
            if (format == kNvq3Jsc2) NVQ_MULTI2(kNvq3Jsc2, 2, 4);
            else if (format == kNvq3Jsc) NVQ_MULTI2(kNvq3Jsc, 2, 4);
            else if (format == kNvq3Jsc512) NVQ_MULTI2(kNvq3Jsc512, 2, 4);
            else if (format == kNvq3JscL) NVQ_MULTI2(kNvq3JscL, 2, 4);
            else if (format == kNvq3JscLGroupExec) {
                NVQ_MULTI2(kNvq3JscLGroupExec, 2, 4);
            }
            else NVQ_MULTI2(kNvq3, 2, 4);
        } else {
            if (format == kNvq3Jsc2) NVQ_MULTI2(kNvq3Jsc2, 4, 4);
            else if (format == kNvq3Jsc) NVQ_MULTI2(kNvq3Jsc, 4, 4);
            else if (format == kNvq3Jsc512) NVQ_MULTI2(kNvq3Jsc512, 4, 4);
            else if (format == kNvq3JscL) NVQ_MULTI2(kNvq3JscL, 4, 4);
            else if (format == kNvq3JscLGroupExec) {
                NVQ_MULTI2(kNvq3JscLGroupExec, 4, 4);
            }
            else NVQ_MULTI2(kNvq3, 4, 4);
        }
    } else if (format == kNvq1L || format == kNvq1S) {
        if (format == kNvq1S) NVQ_MULTI2(kNvq1S, 8, 8);
        else NVQ_MULTI2(kNvq1L, 8, 8);
    } else if (is_e8_format(format)) {
        if (format == kNvq2Exec) NVQ_MULTI2(kNvq2Exec, 4, 8);
        else if (format == kNvq2Jsc) NVQ_MULTI2(kNvq2Jsc, 4, 8);
        else if (format == kNvq2JscExec) NVQ_MULTI2(kNvq2JscExec, 4, 8);
        else if (format == kNvq2JscL) NVQ_MULTI2(kNvq2JscL, 4, 8);
        else if (format == kNvq2JscXL) NVQ_MULTI2(kNvq2JscXL, 4, 8);
        else if (format == kNvq2JscXLGroupExec) {
            NVQ_MULTI2(kNvq2JscXLGroupExec, 4, 8);
        }
        else NVQ_MULTI2(kNvq2, 4, 8);
    } else if (format == kNpq0L || format == kNpq0S) {
        if (format == kNpq0S) NVQ_MULTI2(kNpq0S, 4, 8);
        else NVQ_MULTI2(kNpq0L, 4, 8);
    } else {
        if (format == kNvq3Jsc2) NVQ_MULTI2(kNvq3Jsc2, 8, 8);
        else if (format == kNvq3Jsc) NVQ_MULTI2(kNvq3Jsc, 8, 8);
        else if (format == kNvq3Jsc512) NVQ_MULTI2(kNvq3Jsc512, 8, 8);
        else if (format == kNvq3JscL) NVQ_MULTI2(kNvq3JscL, 8, 8);
        else if (format == kNvq3JscLGroupExec) {
            NVQ_MULTI2(kNvq3JscLGroupExec, 8, 8);
        }
        else NVQ_MULTI2(kNvq3, 8, 8);
    }
#undef NVQ_MULTI2
#undef NVQ_MULTI2_FAST
}

#endif

}  // namespace
