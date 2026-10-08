#pragma once
#include "mfq_tensor_backend.h"
#include <memory>
#include <stdexcept>
#include <vector>

namespace mfq::cuda {
// Diagnostic snapshots are copied after a completed decode window. They retain
// the inputs available before each FFN without changing routing or cache policy.
class RouterLookaheadInputs {
public:
    RouterLookaheadInputs(int layers,int hidden,int samples)
        :layers_(layers),hidden_(hidden),limit_(samples) {
        if(layers<=0 || hidden<=0 || samples<=0)
            throw std::invalid_argument("invalid router lookahead audit geometry");
        latest_=mfq_tensor_backend::empty({layers,hidden},mfq_tensor_backend::TensorOptions()
            .device(mfq_tensor_backend::kCUDA).dtype(mfq_tensor_backend::kFloat16));
        samples_.reserve(samples);
    }
    void input(int layer,const mfq_tensor_backend::Tensor& value) {
        if(layer<0 || layer>=layers_ || !value.is_cuda() || value.numel()!=hidden_)
            throw std::invalid_argument("router lookahead audit requires a single CUDA input row");
        latest_.narrow(0,layer,1).copy_(value.reshape({1,hidden_}).to(mfq_tensor_backend::kFloat16));
    }
    void finish_sample() {
        if(samples_.size()>=static_cast<std::size_t>(limit_))
            throw std::runtime_error("router lookahead audit sample limit exceeded");
        samples_.push_back(latest_.cpu().contiguous());
    }
    int samples() const {return static_cast<int>(samples_.size());}
    mfq_tensor_backend::Tensor rows(int layer) const {
        if(layer<0 || layer>=layers_ || samples_.empty())
            throw std::invalid_argument("router lookahead audit has no input rows");
        std::vector<mfq_tensor_backend::Tensor> values;values.reserve(samples_.size());
        for(const auto& sample:samples_)values.push_back(sample.narrow(0,layer,1));
        return mfq_tensor_backend::cat(values,0).contiguous();
    }
private:
    int layers_,hidden_,limit_;
    mfq_tensor_backend::Tensor latest_;
    std::vector<mfq_tensor_backend::Tensor> samples_;
};
}
