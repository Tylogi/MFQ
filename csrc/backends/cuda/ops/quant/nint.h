#pragma once

#include "native/tensor/backend.h"
#include "native/nint_weight.h"
#include "mfq/quant/nint_rows.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct CudaProfiler;

mfq_tensor_backend::Tensor nint_row_embedding_lookup(
    const mfq::NintRows& table, const mfq_tensor_backend::Tensor& ids);
mfq_tensor_backend::Tensor nint_sharded_embedding_lookup(
    const std::vector<std::shared_ptr<mfq::NintRows>>& tables,
    const std::vector<int64_t>& ids,const std::vector<int64_t>& shape,
    const mfq_tensor_backend::Device& device);


mfq_tensor_backend::Tensor nint_matmul(
    CudaProfiler& profiler,
    const NintWeight& weight,
    mfq_tensor_backend::Tensor input);
mfq_tensor_backend::Tensor nint_matmul_input_mul(
    CudaProfiler& profiler,
    const NintWeight& weight,
    mfq_tensor_backend::Tensor input,
    mfq_tensor_backend::Tensor gate,
    int mode);
mfq_tensor_backend::Tensor nint_matmul_input_mul_f32(
    CudaProfiler& profiler,
    const NintWeight& weight,
    mfq_tensor_backend::Tensor input,
    mfq_tensor_backend::Tensor gate,
    int mode);
mfq_tensor_backend::Tensor nint_matmul_swiglu(
    CudaProfiler& profiler,
    const NintWeight& weight,
    mfq_tensor_backend::Tensor input);
mfq_tensor_backend::Tensor nint_matmul_geglu(
    CudaProfiler& profiler,
    const NintWeight& weight,
    mfq_tensor_backend::Tensor input);

struct NintCpu {
    int format_version = 2;
    int bits = 0;
    int sub_bits = 0;
    int gs = 0;
    int axis = 0;
    int neuron_len = 0;
    std::vector<int64_t> shape;
    int out = 0;
    int ng = 0;
    std::vector<uint8_t> q_packed;
    std::vector<uint8_t> row_q_bits;
    std::vector<int64_t> row_q_bit_offsets;
    std::vector<uint8_t> row_sub_bits;
    std::vector<uint8_t> sub_scale;
    std::vector<uint8_t> sub_min;
    std::vector<uint16_t> neuron_scale_h;
    std::vector<uint16_t> neuron_min_h;
    double aggregate_bpw = 0.0;
    double distribution_entropy = 0.0;
};

struct Nint8ZeroCpu {
    int axis = 0;
    int neuron_len = 0;
    std::vector<int64_t> shape;
    int out = 0;
    int ng = 0;
    std::vector<uint8_t> q;
    std::vector<uint16_t> scale_h;
};

struct CpuQuantizedActivation {
    int64_t rows = 0;
    int64_t groups = 0;
    int64_t group_size = 0;
    std::vector<int8_t> values;
    std::vector<float> scales;
    std::vector<int32_t> sums;
};

mfq_tensor_backend::Tensor pad_last(
    mfq_tensor_backend::Tensor input, int64_t target);
uint32_t cpu_load_packed_bits(
    const uint8_t* data, int64_t nbytes, int64_t bit, int bits);
int32_t cpu_dot_s8_s8_64(
    const int8_t* left, const int8_t* right, int32_t right_sum);
int32_t cpu_dot_u8_s8_64(
    const uint8_t* left, const int8_t* right);
void cpu_unpack_nint_group(
    const uint8_t* packed, int bits, int valid, uint8_t* values);
void cpu_unpack_nint_group_at_bit_offset(
    const uint8_t* packed, uint64_t bit_offset, int bits, int valid, uint8_t* values);
CpuQuantizedActivation cpu_quantize_activation(
    mfq_tensor_backend::Tensor input, int64_t width, int64_t group_size, bool with_sums);
float cpu_half_from_bytes(const int8_t* bytes, int64_t offset);
