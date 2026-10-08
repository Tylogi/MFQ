#include "gated_residual.h"
#include "qwen4_exp.h"
#include "cuda_execution.h"

namespace {
std::vector<mfq_tensor_backend::Tensor> residual_pre(
        CudaExecutionContext& execution, const mfq_tensor_backend::Tensor& input,
        const mfq_tensor_backend::Tensor& norm, const GatedResidualProjection& down,
        const GatedResidualProjection& up, const GatedResidualProjection& inject,
        int64_t hidden, int64_t streams, double eps,
        const GatedResidualMixProjection& mixed_projection,const GatedResidualMarker& marker,
        const GatedResidualActivationProjection& activated_down,
        const GatedResidualActivationProjection& activated_injection,
        const mfq_tensor_backend::Tensor& prepared_norm = {}) {
    namespace tb = mfq_tensor_backend;
    MFQ_RUNTIME_CHECK(input.dim() >= 1 && hidden > 0 && streams > 0 &&
        input.size(-1) / streams == hidden && input.size(-1) % streams == 0 && down && up,
        "gated-residual projection input dimensions disagree");
    if(marker && !prepared_norm.defined())marker("begin");
    auto normalized = prepared_norm.defined()?prepared_norm:
        mfq_qwen4_exp::grouped_rms_norm(input, norm, hidden, eps);
    if(marker)marker("norm");
    const bool fused=execution.config.gr_fused_projections;
    const bool epilogue=fused && execution.config.gr_fused_projection_activation;
    const bool down_activated=epilogue && bool(activated_down);
    auto low = down_activated?activated_down(execution,normalized,streams,false):down(execution, normalized);
    if(marker)marker("down");
    MFQ_RUNTIME_CHECK(low.dim() == input.dim() && low.size(-1) > 0,
        "gated-residual bottleneck rank disagrees");
    if(!down_activated) {
        if(fused)low=mfq_qwen4_exp::gated_residual_bottleneck(low,streams);
        else {low=low/streams;low=low*tb::sigmoid(low);}
    }
    if(marker)marker("bottleneck");
    auto mixed = [&] {
        if(fused && mixed_projection)return mixed_projection(execution,low,normalized,streams);
        auto mixing=up(execution,low);
        MFQ_RUNTIME_CHECK(mixing.sizes()==input.sizes(),"gated-residual mixing shape disagrees");
        return mfq_qwen4_exp::gated_residual_mix(mixing,normalized,streams);
    }();
    if(marker)marker("up_mix");
    tb::Tensor injection;
    if (inject) {
        if(epilogue && activated_injection) {
            injection=activated_injection(execution,normalized,streams,true);
            if(marker)marker("inject");
        } else {
            auto projected=inject(execution,normalized);
            if(marker)marker("inject");
            injection=fused?mfq_qwen4_exp::gated_residual_injection(projected,streams):
                2.0*tb::sigmoid(projected/streams);
        }
        auto expected = input.sizes().vec();
        expected.back() = streams;
        MFQ_RUNTIME_CHECK(injection.sizes().vec() == expected,
            "gated-residual injection shape disagrees");
    }
    if(marker)marker("end");
    return {mixed, input, injection};
}
}

std::vector<mfq_tensor_backend::Tensor> gated_residual_pre_projected(
    CudaExecutionContext& execution,const mfq_tensor_backend::Tensor& input,
    const mfq_tensor_backend::Tensor& norm,const GatedResidualProjection& down,
    const GatedResidualProjection& up,const GatedResidualProjection& inject,
    int64_t hidden,int64_t streams,double eps,const GatedResidualMixProjection& mixed,
    const GatedResidualMarker& marker,const GatedResidualActivationProjection& activated_down,
    const GatedResidualActivationProjection& activated_injection) {
    return residual_pre(execution,input,norm,down,up,inject,hidden,streams,eps,
        mixed,marker,activated_down,activated_injection);
}

std::vector<mfq_tensor_backend::Tensor> gated_residual_pre_after_projected(
    CudaExecutionContext& execution,const mfq_tensor_backend::Tensor& branch,
    const mfq_tensor_backend::Tensor& residual,const mfq_tensor_backend::Tensor& prior_injection,
    const mfq_tensor_backend::Tensor& norm,const GatedResidualProjection& down,
    const GatedResidualProjection& up,const GatedResidualProjection& inject,
    int64_t hidden,int64_t streams,double eps,const GatedResidualMixProjection& mixed,
    const GatedResidualMarker& marker,const GatedResidualActivationProjection& activated_down,
    const GatedResidualActivationProjection& activated_injection) {
    if(!execution.config.gr_fused_projections || !execution.config.gr_fused_post_norm) {
        auto input=mfq_qwen4_exp::gated_residual_post(branch,residual,prior_injection,streams);
        return residual_pre(execution,input,norm,down,up,inject,hidden,streams,eps,
            mixed,marker,activated_down,activated_injection);
    }
    MFQ_RUNTIME_CHECK(branch.size(-1)==hidden,"gated-residual chained hidden width disagrees");
    if(marker)marker("begin");
    auto prepared=mfq_qwen4_exp::gated_residual_post_norm(branch,residual,prior_injection,norm,streams,eps);
    return residual_pre(execution,prepared[0],norm,down,up,inject,hidden,streams,eps,
        mixed,marker,activated_down,activated_injection,prepared[1]);
}
