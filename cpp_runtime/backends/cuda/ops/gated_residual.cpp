#include "gated_residual.h"
#include "qwen4_exp.h"

std::vector<mfq_tensor_backend::Tensor> gated_residual_pre_projected(
        CudaExecutionContext& execution, const mfq_tensor_backend::Tensor& input,
        const mfq_tensor_backend::Tensor& norm, const GatedResidualProjection& down,
        const GatedResidualProjection& up, const GatedResidualProjection& inject,
        int64_t hidden, int64_t streams, double eps) {
    namespace tb = mfq_tensor_backend;
    MFQ_RUNTIME_CHECK(input.dim() >= 1 && hidden > 0 && streams > 0 &&
        input.size(-1) / streams == hidden && input.size(-1) % streams == 0 && down && up,
        "gated-residual projection input dimensions disagree");
    auto normalized = mfq_qwen4_exp::grouped_rms_norm(input, norm, hidden, eps);
    auto low = down(execution, normalized) / streams;
    MFQ_RUNTIME_CHECK(low.dim() == input.dim() && low.size(-1) > 0,
        "gated-residual bottleneck rank disagrees");
    low = low * tb::sigmoid(low);
    auto mixing = tb::sigmoid(up(execution, low));
    MFQ_RUNTIME_CHECK(mixing.sizes() == input.sizes(), "gated-residual mixing shape disagrees");
    auto shape = input.sizes().vec();
    shape.back() = streams;
    shape.push_back(hidden);
    auto mixed = (mixing.reshape(shape) * normalized.reshape(shape)).mean(-2);
    tb::Tensor injection;
    if (inject) {
        injection = 2.0 * tb::sigmoid(inject(execution, normalized) / streams);
        auto expected = input.sizes().vec();
        expected.back() = streams;
        MFQ_RUNTIME_CHECK(injection.sizes().vec() == expected,
            "gated-residual injection shape disagrees");
    }
    return {mixed, input, injection};
}
