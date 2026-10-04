#pragma once

#include "tensor.h"
#include "context.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mfq::cuda::native_ops_detail {

inline std::size_t normalize_dimension(std::int64_t dimension, std::size_t rank) {
    const auto normalized = dimension < 0
        ? dimension + static_cast<std::int64_t>(rank)
        : dimension;
    if (normalized < 0 || normalized >= static_cast<std::int64_t>(rank)) {
        throw std::out_of_range("tensor dimension is out of range");
    }
    return static_cast<std::size_t>(normalized);
}

inline bool floating(ScalarType type) {
    return type == kFloat16 || type == kBFloat16 ||
           type == kFloat32 || type == kFloat64;
}

inline ScalarType promote(ScalarType left, ScalarType right) {
    if (left == right) return left;
    if (left == kFloat64 || right == kFloat64) return kFloat64;
    if (left == kFloat32 || right == kFloat32) return kFloat32;
    if ((left == kFloat16 && right == kBFloat16) ||
        (left == kBFloat16 && right == kFloat16)) return kFloat32;
    if (left == kBFloat16 || right == kBFloat16) return kBFloat16;
    if (left == kFloat16 || right == kFloat16) return kFloat16;
    if (left == kInt64 || right == kInt64) return kInt64;
    if (left == kInt32 || right == kInt32) return kInt32;
    if (left == kInt16 || right == kInt16) return kInt16;
    if (left == kInt8 || right == kInt8) return kInt8;
    if (left == kUInt8 || right == kUInt8) return kUInt8;
    return kBool;
}

inline TensorView align_for_broadcast(
    const Tensor& input,
    std::span<const std::int64_t> output_shape) {
    if (input.dim() > static_cast<std::int64_t>(output_shape.size())) {
        throw std::invalid_argument("broadcast rank mismatch");
    }
    auto result = input.view_descriptor();
    std::array<std::int64_t, kMaximumTensorRank> sizes{};
    std::array<std::int64_t, kMaximumTensorRank> strides{};
    const auto offset = output_shape.size() - static_cast<std::size_t>(input.dim());
    for (std::size_t dimension = 0; dimension < output_shape.size(); ++dimension) {
        sizes[dimension] = output_shape[dimension];
        if (dimension < offset) {
            strides[dimension] = 0;
            continue;
        }
        const auto source_dimension = dimension - offset;
        const auto source_extent = input.size(static_cast<std::int64_t>(source_dimension));
        if (source_extent != output_shape[dimension] && source_extent != 1) {
            throw std::invalid_argument("tensor dimensions cannot be broadcast");
        }
        strides[dimension] = source_extent == 1
            ? 0
            : input.stride(static_cast<std::int64_t>(source_dimension));
    }
    result.rank = static_cast<std::uint8_t>(output_shape.size());
    result.sizes = sizes;
    result.strides = strides;
    return result;
}

inline std::vector<std::int64_t> broadcast_shape(const Tensor& left, const Tensor& right) {
    const auto rank = static_cast<std::size_t>(std::max(left.dim(), right.dim()));
    if (rank > kMaximumTensorRank) throw std::invalid_argument("broadcast rank exceeds ABI");
    std::vector<std::int64_t> result(rank, 1);
    for (std::size_t reverse = 0; reverse < rank; ++reverse) {
        const auto left_dimension = left.dim() - 1 - static_cast<std::int64_t>(reverse);
        const auto right_dimension = right.dim() - 1 - static_cast<std::int64_t>(reverse);
        const auto l = left_dimension >= 0 ? left.size(left_dimension) : 1;
        const auto r = right_dimension >= 0 ? right.size(right_dimension) : 1;
        if (l != r && l != 1 && r != 1) {
            throw std::invalid_argument("tensor dimensions cannot be broadcast");
        }
        result[rank - 1 - reverse] = std::max(l, r);
    }
    return result;
}

inline bool same_shape(const Tensor& left, const Tensor& right) {
    if (left.dim() != right.dim()) {
        return false;
    }
    for (std::int64_t dimension = 0; dimension < left.dim(); ++dimension) {
        if (left.size(dimension) != right.size(dimension)) {
            return false;
        }
    }
    return true;
}

__device__ inline std::int64_t tensor_offset(const TensorView& view, std::int64_t linear) {
    std::int64_t offset = 0;
    for (std::size_t reverse = view.rank; reverse > 0; --reverse) {
        const auto dimension = reverse - 1;
        const auto coordinate = linear % view.sizes[dimension];
        linear /= view.sizes[dimension];
        offset += coordinate * view.strides[dimension];
    }
    return offset;
}

template <typename Value>
__device__ double load_number(const Value* values, std::int64_t index) {
    if constexpr (std::is_same_v<Value, __half>) {
        return static_cast<double>(__half2float(values[index]));
    } else if constexpr (std::is_same_v<Value, __nv_bfloat16>) {
        return static_cast<double>(__bfloat162float(values[index]));
    } else {
        return static_cast<double>(values[index]);
    }
}

template <typename Value>
__device__ Value store_number(double value) {
    if constexpr (std::is_same_v<Value, __half>) {
        return __float2half_rn(static_cast<float>(value));
    } else if constexpr (std::is_same_v<Value, __nv_bfloat16>) {
        return __float2bfloat16_rn(static_cast<float>(value));
    } else {
        return static_cast<Value>(value);
    }
}

template <typename Function>
void dispatch_numeric(ScalarType type, Function&& function) {
    switch (type) {
        case kBool: function.template operator()<bool>(); break;
        case kUInt8: function.template operator()<std::uint8_t>(); break;
        case kInt8: function.template operator()<std::int8_t>(); break;
        case kInt16: function.template operator()<std::int16_t>(); break;
        case kInt32: function.template operator()<std::int32_t>(); break;
        case kInt64: function.template operator()<std::int64_t>(); break;
        case kFloat16: function.template operator()<__half>(); break;
        case kBFloat16: function.template operator()<__nv_bfloat16>(); break;
        case kFloat32: function.template operator()<float>(); break;
        case kFloat64: function.template operator()<double>(); break;
        case kFloat8E4M3FN:
            throw std::invalid_argument("generic native op does not accept FP8 storage");
    }
}

inline std::pair<int, int> launch_geometry(std::int64_t elements) {
    constexpr int threads = 256;
    const int blocks = static_cast<int>(std::min<std::int64_t>(
        4096, std::max<std::int64_t>(1, (elements + threads - 1) / threads)));
    return {blocks, threads};
}


extern std::atomic<std::int64_t> native_random_seed;

}  // namespace mfq::cuda::native_ops_detail
