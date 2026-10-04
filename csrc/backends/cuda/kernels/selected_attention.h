#pragma once

#include "csrc/backends/cuda/native/tensor_backend.h"

#include <initializer_list>

namespace mfq_selected_attention {
using mfq_tensor_backend::Tensor;

void values(std::initializer_list<const Tensor*> tensors);
Tensor promoted_matmul(const Tensor& left, const Tensor& right);
Tensor dots(const Tensor& query, const Tensor& keys);
Tensor dense(const Tensor& query, const Tensor& key, const Tensor& value,
    int64_t offset, double scale);
Tensor sparse(const Tensor& query, const Tensor& key, const Tensor& value,
    const Tensor& indices, double scale, bool key_is_cache);

} // namespace mfq_selected_attention
