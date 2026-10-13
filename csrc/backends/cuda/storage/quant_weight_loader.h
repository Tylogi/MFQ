#pragma once

#include "ops/fp8_sq.h"
#include "ops/mx.h"
#include "ops/mxfp4_sq.h"

#include <cstdint>
#include <vector>

// Payload bytes -> CPU weight description -> device weights for the dense
// quant formats. The matmul/dequant entry points stay in ops/.

Fp8SqWeight to_device_fp8_sq(
    std::string_view dtype,
    const std::vector<uint8_t>& payload,
    bool cuda,
    int device = -1);
Mxfp4SqWeight to_device_mxfp4_sq(
    const std::vector<uint8_t>& payload,
    bool cuda,
    int device = -1);

size_t checked_mxfp8_size(
    uint64_t left, uint64_t right, const char* label);
Mxfp8Cpu unpack_mxfp8(const std::vector<uint8_t>& blob);
Mxfp4Cpu unpack_mxfp4(const std::vector<uint8_t>& blob);
mfq_tensor_backend::Tensor dequant_mxfp4_cpu(const Mxfp4Cpu& source);
Mxfp4Cpu select_mxfp4_cpu_rows(
    const Mxfp4Cpu& source, const std::vector<int64_t>& rows);
Mxfp4Cpu slice_mxfp4_cpu(
    const Mxfp4Cpu& source, TensorParallelAxis axis, int64_t begin, int64_t end);
Mxfp8Cpu slice_mxfp8_cpu(
    const Mxfp8Cpu& source, TensorParallelAxis axis, int64_t begin, int64_t end);
Mxfp8Weight to_device_mxfp8(
    const Mxfp8Cpu& source, bool cuda, int device = -1);
Mxfp8Weight to_cuda_device_mxfp8(const Mxfp8Cpu& source, int device);
Mxfp4Weight to_device_mxfp4(
    const Mxfp4Cpu& source, bool cuda, int device = -1);
