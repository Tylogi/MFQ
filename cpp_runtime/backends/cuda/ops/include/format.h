#pragma once

#include "mfq_cuda_ops.h"
#include "mfq/model_source.h"

#include <cstdint>
#include <initializer_list>
#include <istream>
#include <string>
#include <string_view>
#include <vector>

const mfq::TensorMetadata& require_tensor(
    const mfq::ModelSource& source, std::string_view name);
std::vector<std::uint8_t> read_tensor(
    const mfq::ModelSource& source, std::string_view name);
std::int64_t read_i64_from(
    const std::vector<std::uint8_t>& bytes, std::size_t& offset);
std::uint32_t read_u32_from(
    const std::vector<std::uint8_t>& bytes, std::size_t& offset);

namespace mfq::cuda::quant_format {

uint32_t read_u32(std::istream& input);
uint64_t read_u64(std::istream& input);
int32_t read_i32_from(const std::vector<uint8_t>& bytes, size_t& offset);
uint16_t read_u16_from(const std::vector<uint8_t>& bytes, size_t& offset);
uint64_t read_u64_from(const std::vector<uint8_t>& bytes, size_t& offset);
std::string read_str(std::istream& input);
std::vector<uint8_t> unpack_bits(
    const std::vector<uint8_t>& blob,
    size_t& offset,
    size_t count,
    int bits);

mfq_tensor_backend::Tensor cpu_u8_tensor(
    const std::vector<uint8_t>& values,
    std::initializer_list<int64_t> shape);
mfq_tensor_backend::Tensor cpu_i64_tensor(
    const std::vector<int64_t>& values,
    std::initializer_list<int64_t> shape);
mfq_tensor_backend::Tensor cpu_i32_tensor(
    const std::vector<int32_t>& values,
    std::initializer_list<int64_t> shape);
mfq_tensor_backend::Tensor cpu_f16_tensor(
    const std::vector<uint16_t>& values,
    std::initializer_list<int64_t> shape);
mfq_tensor_backend::Tensor cpu_f16_to_f32_tensor(
    const std::vector<uint16_t>& values, int64_t count);
mfq_tensor_backend::Tensor cpu_i8_tensor(
    const std::vector<int8_t>& values,
    std::initializer_list<int64_t> shape);

} // namespace mfq::cuda::quant_format
