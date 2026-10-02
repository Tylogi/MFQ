#pragma once

#include "mfq_tensor_backend.h"

#include <cstdint>
#include <vector>

struct CudaExecutionConfig;

struct RopeCache {
    mfq_tensor_backend::Tensor cos;
    mfq_tensor_backend::Tensor sin;
    mfq_tensor_backend::Tensor sections;
    mfq_tensor_backend::Tensor empty_sections;
    mfq_tensor_backend::Tensor interleaved_order;
    mfq_tensor_backend::Tensor interleaved_inverse;
    int64_t rotary_dim = 0;

    RopeCache() = default;
    RopeCache(
        int64_t max_positions,
        int64_t dim,
        double base,
        int64_t frequency_dim = 0,
        int64_t active_pairs = -1,
        mfq_tensor_backend::Device device =
            mfq_tensor_backend::Device(mfq_tensor_backend::kCUDA),
        bool official_reciprocal_frequencies = false);

    void configure_mrope(
        const std::vector<int64_t>& configured_sections,
        bool interleaved,
        int64_t configured_rotary_dim,
        mfq_tensor_backend::Device device);
    mfq_tensor_backend::Tensor apply(
        mfq_tensor_backend::Tensor x,
        mfq_tensor_backend::Tensor pos,
        bool grid_mrope_positions = false) const;
    mfq_tensor_backend::Tensor apply_bf16(
        const CudaExecutionConfig& config,
        mfq_tensor_backend::Tensor x,
        mfq_tensor_backend::Tensor pos) const;
};

namespace mfq::cuda {

// Computes rotate-half frequencies for logical [axis,batch,token] positions.
// Uses FP32 for F32 inputs and FP16 otherwise, without a resident table.
class RotaryEmbedding {
public:
    RotaryEmbedding(int64_t dimension, int64_t maximum, double base,
                    std::vector<int64_t> sections = {}, bool interleaved = false);
    mfq_tensor_backend::Tensor forward(const mfq_tensor_backend::Tensor& value,
                                       const mfq_tensor_backend::Tensor& positions) const;
private:
    int64_t dimension_, maximum_;
    double base_;
    std::vector<int64_t> sections_;
    bool interleaved_;
};

} // namespace mfq::cuda
