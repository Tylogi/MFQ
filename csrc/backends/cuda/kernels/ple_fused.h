#pragma once
#include "native/tensor/backend.h"
#include <optional>
#include <vector>

namespace mfq_qwen4_exp {
using mfq_tensor_backend::Tensor;
std::vector<Tensor> ple_gate_norm_cuda(const Tensor& key,const Tensor& query,
    const Tensor& value,const Tensor& norm,int64_t hidden,int64_t streams,double eps);
std::vector<Tensor> ple_conv_add_cuda(const Tensor& normalized,const Tensor& weight,
    const std::optional<Tensor>& previous,int64_t dilation,const Tensor& gated,bool half_output);
}
