#include "mfq_cuda_shared_gate.h"
#include "mfq_cuda_context.h"
#include "shared_gate_fused.cuh"

std::optional<mfq_tensor_backend::Tensor> try_shared_gate_sigmoid_cuda(
        const mfq_tensor_backend::Tensor& input,
        const mfq_tensor_backend::Tensor& weight) {
    namespace tb=mfq_tensor_backend;
    if(!input.defined() || !weight.defined() || !input.is_cuda() || !weight.is_cuda() ||
        input.get_device()!=weight.get_device() || !input.is_contiguous() || !weight.is_contiguous() ||
        input.scalar_type()!=tb::kFloat16 || weight.scalar_type()!=tb::kBFloat16 ||
        input.dim()!=2 || weight.dim()!=2 || weight.size(0)!=1 ||
        input.size(0)<1 || input.size(0)>8 || input.size(1)<1 || input.size(1)>4096 ||
        input.size(1)!=weight.size(1))return std::nullopt;
    auto output=tb::empty({input.size(0),1},weight.options());
    mfq::cuda::DeviceGuard guard(input.get_device());
    const auto stream=mfq::cuda::current_stream(input.get_device()).stream();
    mfq::cuda::shared_gate_detail::project_sigmoid<128,true,false>
        <<<int(input.size(0)),128,0,stream>>>(
            static_cast<const __half*>(input.data_ptr()),
            static_cast<const __nv_bfloat16*>(weight.data_ptr()),nullptr,
            static_cast<__nv_bfloat16*>(output.data_ptr()),int(input.size(1)));
    MFQ_NATIVE_CUDA_CHECK(cudaGetLastError());
    return output;
}
