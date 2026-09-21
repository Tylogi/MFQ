#pragma once

#include "cuda_execution.h"
#include "mfq_cuda_ops.h"

#include <cstdint>
#include <string>
#include <vector>

struct TpqWeight {
    bool int4 = false;
    mfq_tensor_backend::Tensor packed;
    mfq_tensor_backend::Tensor scales;
    mfq_tensor_backend::Tensor codebook;
    int64_t out = 0;
    int64_t neuron_len = 0;
    int group_size = 0;
    int vector_size = 0;
    int index_bits = 0;
};

mfq_tensor_backend::Tensor tpq_matmul(
    const TpqWeight& weight, mfq_tensor_backend::Tensor input);

struct TpqLinear {
    TpqWeight weight;

    mfq_tensor_backend::Tensor forward(mfq_tensor_backend::Tensor x) const {
        auto shape = x.sizes().vec();
        auto y = tpq_matmul(
            weight, x.reshape({-1, x.size(-1)}));
        shape.back() = y.size(-1);
        return y.reshape(shape);
    }
};

struct TpqCpu {
    bool int4 = false;
    int64_t out = 0;
    int64_t neuron_len = 0;
    int group_size = 0;
    int vector_size = 0;
    int index_bits = 0;
    int codebook_entries = 0;
    std::vector<uint8_t> packed;
    std::vector<uint16_t> scales_h;
    std::vector<float> codebook;
};

bool is_tpq_pq_dtype(const std::string& dtype);
TpqCpu unpack_tpq_pq(
    const std::vector<uint8_t>& blob, const std::string& dtype);
TpqCpu unpack_tpq_int4(const std::vector<uint8_t>& blob);
TpqCpu slice_tpq_cpu(
    const TpqCpu& source, TensorParallelAxis axis, int64_t begin, int64_t end);
TpqCpu select_tpq_cpu_rows(
    const TpqCpu& source, const std::vector<int64_t>& rows);
TpqWeight to_device_tpq(
    const TpqCpu& source, bool cuda, int device = -1);
