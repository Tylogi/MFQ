#pragma once
#include "mfq_tensor_backend.h"
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

// Keep the complete 64-bit bit offset and both FP32 anchors. This experiment
// changes their placement only, including anchors that cannot be stored in Half.
inline mfq_tensor_backend::Tensor make_nint_row_metadata(
        const mfq_tensor_backend::Tensor& offsets,
        const mfq_tensor_backend::Tensor& scales,
        const mfq_tensor_backend::Tensor& minima,
        const mfq_tensor_backend::Device& device) {
    namespace tb=mfq_tensor_backend;
    if(offsets.is_cuda() || scales.is_cuda() || minima.is_cuda() ||
            !offsets.is_contiguous() || !scales.is_contiguous() || !minima.is_contiguous() ||
            offsets.scalar_type()!=tb::kInt64 || scales.scalar_type()!=tb::kFloat32 ||
            minima.scalar_type()!=tb::kFloat32 || offsets.numel()!=scales.numel() ||
            offsets.numel()!=minima.numel())throw std::runtime_error("invalid NINT row record source");
    std::vector<int32_t> records(size_t(offsets.numel())*4);
    for(int64_t row=0;row<offsets.numel();++row) {
        const uint64_t offset=uint64_t(offsets.data_ptr<int64_t>()[row]);
        const uint32_t low=uint32_t(offset),high=uint32_t(offset>>32);
        std::memcpy(records.data()+row*4,&low,4);
        std::memcpy(records.data()+row*4+1,&high,4);
        std::memcpy(records.data()+row*4+2,scales.data_ptr<float>()+row,4);
        std::memcpy(records.data()+row*4+3,minima.data_ptr<float>()+row,4);
    }
    return tb::tensor(records).reshape({offsets.numel(),4}).to(device).contiguous();
}
