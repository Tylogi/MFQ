#pragma once

#include "ops/fp8_sq.h"
#include "ops/mx.h"
#include "ops/mxfp4_sq.h"
#include "ops/nint.h"
#include "ops/vq.h"

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

void require_tp_row_major_weight(
    const std::vector<int64_t>& shape,
    int axis,
    int out,
    int neuron_len,
    const char* format);
NintCpu unpack_nint(const std::vector<uint8_t>& blob);
Nint8ZeroCpu unpack_nint8_zero(const std::vector<uint8_t>& blob);
NintCpu slice_nint_cpu_output(
    const NintCpu& source, int64_t begin, int64_t end);
NintCpu slice_nint_cpu_input_groups(
    const NintCpu& source, int64_t begin, int64_t end);
NintCpu select_nint_cpu_rows(
    const NintCpu& source, const std::vector<int64_t>& rows);
Nint8ZeroCpu slice_nint8_zero_cpu_output(
    const Nint8ZeroCpu& source, int64_t begin, int64_t end);
Nint8ZeroCpu slice_nint8_zero_cpu_input_groups(
    const Nint8ZeroCpu& source, int64_t begin, int64_t end);
Nint8ZeroCpu select_nint8_zero_cpu_rows(
    const Nint8ZeroCpu& source, const std::vector<int64_t>& rows);
NintWeight to_device_nint(const NintCpu& source, bool cuda);
NintWeight to_gpu_nint(const NintCpu& source);
NintWeight to_cuda_device_nint(const NintCpu& source, int device);
NintWeight to_cpu_nint(const NintCpu& source);
NintWeight to_device_mfe_nint(
    const NintCpu& source, int local_experts, int out_per_expert, bool cuda);
NintWeight to_device_nint8_zero(const Nint8ZeroCpu& source, bool cuda);
NintWeight to_gpu_nint8_zero(const Nint8ZeroCpu& source);
NintWeight to_cuda_device_nint8_zero(
    const Nint8ZeroCpu& source, int device);
NintWeight to_cpu_nint8_zero(const Nint8ZeroCpu& source);
mfq_tensor_backend::Tensor dequant_nint8_zero_cpu(
    const Nint8ZeroCpu& source);

NvqCpu unpack_nvq(const std::vector<uint8_t>& blob, const std::string& dtype);
NvqCpu slice_nvq_cpu(
    const NvqCpu& source, TensorParallelAxis axis, int64_t begin, int64_t end);
NvqCpu select_nvq_cpu_rows(
    const NvqCpu& source, const std::vector<int64_t>& rows);
NvqWeight to_device_nvq(
    const NvqCpu& source,
    bool cuda,
    const CudaExecutionConfig& config = {});
NvqWeight to_gpu_nvq(
    const NvqCpu& source,
    const CudaExecutionConfig& config = {});
NvqWeight to_cuda_device_nvq(
    const NvqCpu& source,
    int device,
    const CudaExecutionConfig& config = {});
NvqWeight to_cpu_nvq(const NvqCpu& source);
NepqCpu unpack_nepq(
    const std::vector<uint8_t>& blob,
    const std::string& dtype,
    const std::vector<uint8_t>& runtime_payload);
NepqCpu select_nepq_cpu_rows(
    const NepqCpu& source,
    const std::vector<int64_t>& rows,
    int output_per_expert,
    int selected_experts = -1);
NepqWeight to_device_nepq(const NepqCpu& source, bool cuda);
NepqWeight to_gpu_nepq(const NepqCpu& source);
NepqWeight to_cpu_nepq(const NepqCpu& source);
