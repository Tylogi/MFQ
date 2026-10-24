#pragma once
#include "mfq_tensor_backend.h"
#include "mfq/model_source.h"
#include "tq_projection.h"

struct TqWeight {
    mfq_tensor_backend::Tensor image;
    ProjectionView view{};
    std::int64_t out=0, neuron_len=0;
    mfq_tensor_backend::Tensor forward(mfq_tensor_backend::Tensor input) const;
    mfq_tensor_backend::Tensor materialize(mfq_tensor_backend::ScalarType type) const;
};

TqWeight load_tq_weight(const mfq::ModelSource&,const std::string&,int device);
TqWeight to_device_tq(const std::vector<std::uint8_t>&,std::int64_t columns,int device);
bool tq_q8_enabled();
std::vector<mfq_tensor_backend::Tensor> tq_forward_group(
    const std::vector<const TqWeight*>&,mfq_tensor_backend::Tensor);
mfq_tensor_backend::Tensor tq_swiglu(
    const TqWeight&,const TqWeight&,mfq_tensor_backend::Tensor);
mfq_tensor_backend::Tensor tq_forward_residual(
    const TqWeight&,mfq_tensor_backend::Tensor,mfq_tensor_backend::Tensor);
