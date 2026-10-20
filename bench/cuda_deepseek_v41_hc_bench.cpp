#include "models/deepseek_v41/ops.h"
#include <chrono>
#include <iostream>

namespace {
using namespace mfq::cuda::deepseek_v41_runtime;

MhcResult composed_collapse(const Tensor& residual, const Tensor& previous_pre,
                           const Tensor& function, const Tensor& scale,
                           const Tensor& base, const Tensor& norm) {
    auto flat = residual.flatten(2).to(mfq_tensor_backend::kFloat32);
    auto inverse = mfq_tensor_backend::rsqrt(flat.square().mean(-1, true) + 1.e-20);
    auto mixes = mfq_tensor_backend::matmul(flat * inverse, function.transpose(0, 1));
    auto expansion = dsv4_hc_pre_cuda(residual, mixes.contiguous(), scale, base, 20, 1.e-6);
    auto next = mfq_tensor_backend::sigmoid(mixes.slice(-1, 0, 4) * scale.index({0}) + base.slice(0, 0, 4)) + 1.e-6;
    auto branch = (previous_pre.unsqueeze(-1) * residual.to(mfq_tensor_backend::kFloat32)).sum(2);
    return {weighted_rms(branch, norm, 1.e-20), next.contiguous(), expansion.at(1), expansion.at(2)};
}

template<class F> double elapsed_ms(F&& function, int repetitions) {
    for (int index = 0; index < 8; ++index) function();
    auto stream = mfq_current_cuda_stream();
    cudaEvent_t begin, end;
    MFQ_CUDA_CHECK(cudaEventCreate(&begin));
    MFQ_CUDA_CHECK(cudaEventCreate(&end));
    MFQ_CUDA_CHECK(cudaEventRecord(begin, stream));
    for (int index = 0; index < repetitions; ++index) function();
    MFQ_CUDA_CHECK(cudaEventRecord(end, stream));
    MFQ_CUDA_CHECK(cudaEventSynchronize(end));
    float milliseconds = 0;
    MFQ_CUDA_CHECK(cudaEventElapsedTime(&milliseconds, begin, end));
    MFQ_CUDA_CHECK(cudaEventDestroy(begin));
    MFQ_CUDA_CHECK(cudaEventDestroy(end));
    return milliseconds / repetitions;
}
}

int main(int argc,char** argv) try {
    const std::string mode=argc==2?argv[1]:"";
    const bool legacy_only=mode=="--three-stage-only";
    const bool current_only=mode=="--current-only" || mode=="--crossover" || legacy_only;
    namespace tb = mfq_tensor_backend;
    const auto fp32 = tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat32);
    CudaProfiler profiler;
    mfq::cuda::deepseek_v41_runtime::Block block;
    block.config.hc_mult = 4;
    block.config.hc_sinkhorn_iters = 20;
    block.config.hc_eps = 1.e-6;
    block.config.rms_eps = 1.e-20;
    const std::vector<int> cases=legacy_only?std::vector<int>{8,16,32}:mode=="--crossover"?std::vector<int>{8,16}:
        current_only?std::vector<int>{1,2,6,32}:std::vector<int>{1,2,6,128,1024,4096};
    for (const int tokens : cases) {
        auto residual = tb::randn({1, tokens, 4, 5120}, fp32).to(tb::kFloat16).contiguous();
        auto pre = tb::full({1, tokens, 4}, .25, fp32);
        auto function = tb::randn({24, 4 * 5120}, fp32) / std::sqrt(4.0 * 5120);
        auto scale = tb::ones({3}, fp32), base = tb::zeros({24}, fp32), norm = tb::ones({5120}, fp32);
        auto old_path = [&] { return composed_collapse(residual, pre, function, scale, base, norm); };
        auto new_path = [&] { return block.collapse(profiler, residual, pre, function, scale, base, norm, "hc"); };
        auto three_stage = [&] {
            auto prepared=deepseek_v41_hc_prepare_cuda(residual,pre,norm,1.e-20);
            auto mixes=tb::matmul(prepared[0],function.transpose(0,1));
            auto coeff=dsv4_hc_coefficients_cuda(mixes.contiguous(),scale,base,20,1.e-6);
            return MhcResult{prepared[1],coeff[0],coeff[1],coeff[2]};
        };
        auto reference = old_path(), candidate = new_path();
        for (const auto& pair : {std::pair{reference.branch, candidate.branch},
                                 std::pair{reference.next_pre, candidate.next_pre},
                                 std::pair{reference.post, candidate.post},
                                 std::pair{reference.combination, candidate.combination}}) {
            const double error = (pair.first.to(tb::kFloat32) - pair.second.to(tb::kFloat32)).abs().max().item<double>();
            if (!std::isfinite(error) || error > .0021) throw std::runtime_error("mHC composed/fused output mismatch");
        }
        const int repetitions = tokens <= 6 ? 200 : tokens <= 128 ? 80 : tokens <= 1024 ? 15 : 5;
        if(current_only) {
            const auto fused=legacy_only?elapsed_ms(three_stage,repetitions):elapsed_ms(new_path,repetitions);
            std::cout<<"{\"tokens\":"<<tokens<<",\"fused_ms\":"<<fused
                     <<",\"dispatch\":\""<<(legacy_only || tokens>8?"three_stage":"split_projection_two_stage")<<"\"}"<<std::endl;
            continue;
        }
        const double previous = elapsed_ms(old_path, repetitions), fused = elapsed_ms(new_path, repetitions);
        std::cout << "{\"tokens\":" << tokens << ",\"previous_ms\":" << previous
                  << ",\"fused_ms\":" << fused << ",\"speedup\":" << previous / fused << "}" << std::endl;
    }
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
