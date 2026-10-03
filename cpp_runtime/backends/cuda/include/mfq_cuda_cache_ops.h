#pragma once

#include "mfq_tensor_backend.h"

#include <cstdint>
#include <vector>

mfq_tensor_backend::Tensor glm_dsa_cache_write_cuda(
    mfq_tensor_backend::Tensor cache, mfq_tensor_backend::Tensor values, mfq_tensor_backend::Tensor positions);
std::vector<mfq_tensor_backend::Tensor> kv_cache_write_cuda(mfq_tensor_backend::Tensor k_cache, mfq_tensor_backend::Tensor v_cache,
                                               mfq_tensor_backend::Tensor k, mfq_tensor_backend::Tensor v, mfq_tensor_backend::Tensor positions);
std::vector<mfq_tensor_backend::Tensor> kv_cache_write_ring_cuda(
    mfq_tensor_backend::Tensor k_cache, mfq_tensor_backend::Tensor v_cache,
    mfq_tensor_backend::Tensor k, mfq_tensor_backend::Tensor v, int64_t position_start);
std::vector<mfq_tensor_backend::Tensor> kv_cache_write_ring_positions_cuda(
    mfq_tensor_backend::Tensor k_cache, mfq_tensor_backend::Tensor v_cache,
    mfq_tensor_backend::Tensor k, mfq_tensor_backend::Tensor v, mfq_tensor_backend::Tensor positions);
