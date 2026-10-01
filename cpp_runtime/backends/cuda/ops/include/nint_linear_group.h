#pragma once

#include "quant_linear_common.h"

struct NintLinearGroup {
    NintWeight w;
    std::vector<NintWeight> projection_w;
    std::vector<NintWeight> split_w;
    std::vector<std::vector<int64_t>> split_outs;
    std::vector<int64_t> outs;
    mutable std::shared_ptr<CudaIndependentBranchExecutor>
        branch_executor =
            std::make_shared<CudaIndependentBranchExecutor>();
    std::vector<mfq_tensor_backend::Tensor> forward(
            CudaExecutionContext& execution,
            mfq_tensor_backend::Tensor x) const {
        auto shape = x.sizes().vec();
        std::vector<mfq_tensor_backend::Tensor> parts;
        auto xf = x.reshape({-1, x.size(-1)});
        if (w.q8_zero && xf.size(0) > 64 &&
                projection_w.size() == outs.size()) {
            parts.reserve(projection_w.size());
            for (const auto & projection : projection_w) {
                parts.push_back(run_nint_linear(
                    execution, projection, xf));
            }
        } else if (!split_w.empty()) {
            parts.reserve(outs.size());
            std::vector<mfq_tensor_backend::Tensor> grouped_outputs;
            const bool parallel =
                decode_branch_parallel_enabled(execution.decode_graph_serial_branches, xf.size(0)) &&
                branch_executor->run(
                    split_w.size(),
                    [&](size_t index) {
                        return run_nint_linear(
                            execution, split_w[index], xf);
                    },
                    grouped_outputs);
            for (size_t i = 0; i < split_w.size(); ++i) {
                auto y = parallel
                    ? grouped_outputs[i]
                    : run_nint_linear(execution, split_w[i], xf);
                auto ys = y.split_with_sizes(split_outs[i], -1);
                for (auto & p : ys) parts.push_back(p);
            }
        } else {
            auto y = run_nint_linear(
                execution, w, x.reshape({-1, x.size(-1)}));
            parts = y.split_with_sizes(outs, -1);
        }
        for (auto & p : parts) {
            auto s = shape;
            s.back() = p.size(-1);
            p = p.reshape(s);
        }
        return parts;
    }
    mfq_tensor_backend::Tensor forward_swiglu(
            CudaProfiler& profiler,
            mfq_tensor_backend::Tensor x) const {
        if (!split_w.empty() || outs.size() != 2 || outs[0] != outs[1]) {
            throw std::runtime_error("NINT SwiGLU fusion requires one packed [gate, up] group");
        }
        auto shape = x.sizes().vec();
        auto y = nint_matmul_swiglu(
            profiler, w, x.reshape({-1, x.size(-1)}));
        shape.back() = y.size(-1);
        return y.reshape(shape);
    }
    mfq_tensor_backend::Tensor forward_geglu(
            CudaProfiler& profiler,
            mfq_tensor_backend::Tensor x) const {
        if (!split_w.empty() || outs.size() != 2 || outs[0] != outs[1]) {
            throw std::runtime_error("NINT GeGLU fusion requires one packed [gate, up] group");
        }
        auto shape = x.sizes().vec();
        auto y = nint_matmul_geglu(
            profiler, w, x.reshape({-1, x.size(-1)}));
        shape.back() = y.size(-1);
        return y.reshape(shape);
    }
};

enum class QuantLinearKind {
    Nint,
    Nvq,
    Mxfp4,
    Mxfp4Sq,
    Fp8Sq,
    Mxfp8,
    Dense,
};

