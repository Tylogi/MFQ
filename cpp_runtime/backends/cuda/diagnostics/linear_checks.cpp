#include "storage/weight_loader.h"
#include "model_checks.h"

#include "quant_linear.h"
#include "cuda_execution.h"
#include "mfq_cuda_activation_ops.h"
#include "mfq_cuda_linear_attention_ops.h"
#include "mfq_cuda_quant_ops.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using mfq_tensor_backend::indexing::Slice;

int run_linear_check(
    CudaExecutionContext& execution,
    const std::string & model_path,
    const std::string & name,
    int M,
    int gate_mode,
    int reps) {
    auto model_source = mfq::open_model_source(model_path);
    const auto& mfq = *model_source;
    auto linear = load_quant_linear(execution, mfq, name);
    MFQ_RUNTIME_CHECK(M >= 1 && M <= 4096, "--check-linear-m must be in [1, 4096]");
    MFQ_RUNTIME_CHECK(reps >= 1, "--check-linear-reps must be positive");
    int64_t neuron_len = linear.neuron_len();
    auto x = mfq_tensor_backend::arange((int64_t)M * neuron_len,
                           mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA).dtype(mfq_tensor_backend::kFloat32))
                 .reshape({M, neuron_len});
    x = (x.remainder(97) - 48) / 512.0;
    auto xh = x.to(mfq_tensor_backend::kFloat16).contiguous();
    mfq_tensor_backend::Tensor gateh;
    if (gate_mode != 0) {
        MFQ_RUNTIME_CHECK(gate_mode == 1 || gate_mode == 2, "linear check gate mode must be 0, 1, or 2");
        gateh = ((mfq_tensor_backend::arange((int64_t)M * neuron_len, x.options()).reshape({M, neuron_len})
                    .remainder(53) - 26) / 16.0)
                    .to(mfq_tensor_backend::kFloat16).contiguous();
    }
    auto run = [&]() {
        return gate_mode == 0
            ? linear.forward(execution, xh)
            : linear.forward_input_mul(
                execution, xh, gateh, gate_mode);
    };
    const char * check_bf16_output_env =
        std::getenv("MFQ_CHECK_LINEAR_BF16_OUTPUT");
    if (check_bf16_output_env != nullptr &&
            check_bf16_output_env[0] == '1') {
        MFQ_RUNTIME_CHECK(
            gate_mode == 0,
            "direct BF16 linear check does not support input gating");
        auto bf16_input = x.to(mfq_tensor_backend::kBFloat16).contiguous();
        auto reference = linear.forward(execution, bf16_input)
            .to(mfq_tensor_backend::kBFloat16).contiguous();
        auto candidate = linear.forward_bf16_output(
            execution, bf16_input);
        auto difference =
            (candidate.to(mfq_tensor_backend::kFloat32) -
             reference.to(mfq_tensor_backend::kFloat32)).abs();
        std::cout << "linear_bf16_output_check"
                  << " max_abs="
                  << difference.max().item<double>()
                  << " equal="
                  << (candidate.equal(reference) ? 1 : 0)
                  << "\n";
        MFQ_RUNTIME_CHECK(
            candidate.equal(reference),
            "direct BF16 NINT output differs from FP16-then-BF16 reference");
    }
    mfq_tensor_backend::Tensor y_test;
    const int warmups = std::min(30, std::max(1, reps));
    for (int i = 0; i < warmups; ++i) y_test = run();
    mfq_cuda_synchronize();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    auto stream = mfq_get_current_cuda_stream().stream();
    cudaEventRecord(start, stream);
    for (int i = 0; i < reps; ++i) y_test = run();
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    y_test = y_test.to(mfq_tensor_backend::kFloat32);

    const NintWeight * nint = linear.is_nint() ? &linear.nint : nullptr;
    const NvqWeight * nvq = linear.is_nvq() ? &linear.nvq : nullptr;
    const Mxfp4Weight * mxfp4 = linear.is_mxfp4()
        ? &linear.mxfp4.weight : nullptr;
    const Mxfp4SqWeight * mxfp4_sq = linear.is_mxfp4_sq()
        ? &linear.mxfp4_sq.weight : nullptr;
    const Fp8SqWeight * fp8_sq = linear.is_fp8_sq()
        ? &linear.fp8_sq.weight : nullptr;
    const Mxfp8Weight * mxfp8 = linear.is_mxfp8()
        ? &linear.mxfp8.weight : nullptr;
    auto ww = quant_linear_reference_weight(linear);
    mfq_tensor_backend::Tensor ref_input = xh;
    if (gate_mode == 1) ref_input = xh * mfq_tensor_backend::sigmoid(gateh);
    else if (gate_mode == 2) ref_input = xh * mfq_tensor_backend::silu(gateh);
    auto y_ref = mfq_tensor_backend::matmul(ref_input, ww.transpose(0, 1)).to(mfq_tensor_backend::kFloat32);
    mfq_cuda_synchronize();
    auto diff = (y_test - y_ref).abs();
    double per_ms = (double)elapsed_ms / (double)reps;
    double weight_bytes = nint != nullptr
        ? (double)nint->q_packed.numel() +
          (nint->q8_zero
              ? (double)nint->q8_zero_scale.numel() * sizeof(mfq_half)
              : (double)(nint->sub_scale.numel() +
                         nint->sub_min.numel()) +
                (double)(nint->neuron_scale.numel() +
                         nint->neuron_min.numel()) * sizeof(float))
        : nvq != nullptr
        ? (double)nvq->indices_packed.numel() + (double)nvq->aux_packed.numel() +
          (double)nvq->sub_scale_packed.numel() +
          (double)nvq->neuron_scale.numel() * sizeof(float) + (double)nvq->codebook.numel()
        : mxfp4 != nullptr
        ? (double)mxfp4->values.numel() + (double)mxfp4->scales.numel()
        : mxfp4_sq != nullptr
        ? (double)mxfp4_sq->blob.numel()
        : fp8_sq != nullptr
        ? (double)fp8_sq->blob.numel()
        : (double)mxfp8->values.numel() + (double)mxfp8->scales.numel();
    std::cout << "shape=" << y_ref.sizes() << "\n";
    if (nint != nullptr) {
        if (nint->q8_zero) {
            std::cout << "dtype=NINT8-0";
        } else {
            std::cout << "dtype=NINT format_version="
                      << nint->format_version
                      << " aggregate_bpw=" << nint->aggregate_bpw
                      << " distribution_entropy="
                      << nint->distribution_entropy
                      << " nominal_q=" << nint->bits;
        }
        std::cout << " gs=" << nint->gs << " m=" << M << "\n";
    } else if (nvq != nullptr) {
        std::cout << "dtype=NVQ profile=" << nvq->format << " gs=" << nvq->gs
                  << " sub_bits=" << nvq->sub_bits << " m=" << M << "\n";
    } else if (mxfp4 != nullptr) {
        std::cout << "dtype=MXFP4 block=1x32 m=" << M << "\n";
    } else if (mxfp4_sq != nullptr) {
        std::cout << "dtype=MXFP4-SQ format_version="
                  << mxfp4_sq->format_version
                  << " aggregate_bpw=" << mxfp4_sq->aggregate_bpw
                  << " distribution_entropy="
                  << mxfp4_sq->distribution_entropy;
        if (mxfp4_sq->bits != 0) {
            std::cout << " uniform_q=" << mxfp4_sq->bits;
        }
        std::cout << " block=1x32 m=" << M << "\n";
    } else if (fp8_sq != nullptr) {
        std::cout << "dtype=" << fp8_sq->dtype
                  << " format_version=" << fp8_sq->format_version
                  << " aggregate_bpw=" << fp8_sq->aggregate_bpw
                  << " distribution_entropy="
                  << fp8_sq->distribution_entropy
                  << " block=" << fp8_sq->block_rows
                  << "x" << fp8_sq->block_columns
                  << " m=" << M << "\n";
    } else {
        std::cout << "dtype=MXFP8 block=128x128 m=" << M << "\n";
    }
    if (gate_mode != 0) std::cout << "gate=" << (gate_mode == 1 ? "sigmoid" : "silu") << "\n";
    std::cout << "production_ms=" << per_ms << "\n";
    std::cout << "production_weight_gbps=" << weight_bytes / (per_ms * 1.0e6) << "\n";
    std::cout << "production_rel=" << ((y_test - y_ref).norm() / y_ref.norm()).item<float>() << "\n";
    std::cout << "production_mean_abs=" << diff.mean().item<float>() << "\n";
    std::cout << "production_max_abs=" << diff.max().item<float>() << "\n";
    return 0;
}

int run_cpu_linear_check(
        CudaExecutionContext& execution,
        const std::string & model_path,
        const std::string & name,
        int rows,
        int gate_mode,
        int reps) {
    MFQ_RUNTIME_CHECK(rows >= 1 && rows <= 4096, "--check-linear-m must be in [1, 4096]");
    MFQ_RUNTIME_CHECK(reps >= 1, "--check-linear-reps must be positive");
    auto model_source = mfq::open_model_source(model_path);
    const auto& mfq = *model_source;
    execution.loading_cpu_layer = true;
    QuantLinear cpu_linear;
    try {
        cpu_linear = load_quant_linear(execution, mfq, name);
    } catch (...) {
        execution.loading_cpu_layer = false;
        throw;
    }
    execution.loading_cpu_layer = false;
    auto cuda_linear = load_quant_linear(execution, mfq, name);
    const int64_t width = cpu_linear.neuron_len();
    auto x = mfq_tensor_backend::arange(
        static_cast<int64_t>(rows) * width,
        mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCPU).dtype(mfq_tensor_backend::kFloat32))
        .reshape({rows, width});
    x = ((x.remainder(97) - 48) / 512.0)
        .to(mfq_tensor_backend::kFloat16).contiguous();
    mfq_tensor_backend::Tensor gate;
    if (gate_mode != 0) {
        MFQ_RUNTIME_CHECK(gate_mode == 1 || gate_mode == 2, "linear check gate mode must be 0, 1, or 2");
        gate = ((mfq_tensor_backend::arange(
            static_cast<int64_t>(rows) * width,
            mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCPU).dtype(mfq_tensor_backend::kFloat32))
            .reshape({rows, width}).remainder(53) - 26) / 16.0)
            .to(mfq_tensor_backend::kFloat16).contiguous();
    }
    auto run_cpu = [&]() {
        return gate_mode == 0
            ? cpu_linear.forward(execution, x)
            : cpu_linear.forward_input_mul(
                execution, x, gate, gate_mode);
    };
    mfq_tensor_backend::Tensor actual = run_cpu();
    const auto start = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < reps; ++iteration) {
        actual = run_cpu();
    }
    const auto end = std::chrono::steady_clock::now();
    const double cpu_ms = std::chrono::duration<double, std::milli>(
        end - start).count() / static_cast<double>(reps);

    auto cuda_x = x.to(mfq_tensor_backend::kCUDA).contiguous();
    mfq_tensor_backend::Tensor cuda_gate;
    if (gate_mode != 0) cuda_gate = gate.to(mfq_tensor_backend::kCUDA).contiguous();
    auto reference = gate_mode == 0
        ? cuda_linear.forward(execution, cuda_x)
        : cuda_linear.forward_input_mul(
            execution, cuda_x, cuda_gate, gate_mode);
    mfq_cuda_synchronize();
    reference = reference.to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kFloat32).contiguous();
    actual = actual.to(mfq_tensor_backend::kFloat32).contiguous();
    auto difference = (actual - reference).abs();
    std::cout << "cpu_linear=" << name << "\n"
              << "shape=" << actual.sizes() << "\n"
              << "cpu_ms=" << cpu_ms << "\n"
              << "relative_error="
              << ((actual - reference).norm() / reference.norm()).item<float>() << "\n"
              << "mean_abs=" << difference.mean().item<float>() << "\n"
              << "max_abs=" << difference.max().item<float>() << "\n";
    return 0;
}

int run_tensor_parallel_linear_check(
        CudaExecutionContext& execution,
        const std::string & model_path,
        const std::string & name,
        TensorParallelAxis axis,
        int M) {
    MFQ_RUNTIME_CHECK(
        execution.tensor_parallel.enabled(),
        "--check-tp-linear requires --tensor-parallel");
    MFQ_RUNTIME_CHECK(
        axis == TensorParallelAxis::Output ||
        axis == TensorParallelAxis::Input,
        "--check-tp-axis must be output or input");
    MFQ_RUNTIME_CHECK(
        M >= 1 && M <= 4096,
        "--check-tp-m must be in [1, 4096]");
    auto model_source = mfq::open_model_source(model_path);
    const auto& mfq = *model_source;
    const ParallelConfig saved = execution.tensor_parallel;
    execution.tensor_parallel = {};
    execution.tensor_parallel.devices = {saved.primary_device()};
    QuantLinear full;
    try {
        full = load_quant_linear(execution,
            mfq, name, TensorParallelAxis::Mirrored);
    } catch (...) {
        execution.tensor_parallel = saved;
        throw;
    }
    execution.tensor_parallel = saved;
    auto sharded = load_quant_linear(execution,
        mfq, name, axis);
    MFQ_RUNTIME_CHECK(
        sharded.tensor_parallel(),
        "tensor-parallel diagnostic did not create shards");

    const int64_t width = full.neuron_len();
    auto x = mfq_tensor_backend::arange(
        static_cast<int64_t>(M) * width,
        mfq_tensor_backend::TensorOptions()
            .device(mfq_tensor_backend::Device(
                mfq_tensor_backend::kCUDA,
                saved.primary_device()))
            .dtype(mfq_tensor_backend::kFloat32))
        .reshape({M, width});
    x = ((x.remainder(127) - 63) / 384.0)
        .to(mfq_tensor_backend::kFloat16).contiguous();
    auto reference = full.forward(
        execution, x).to(mfq_tensor_backend::kFloat32);
    auto test = sharded.forward(
        execution, x).to(mfq_tensor_backend::kFloat32);
    mfq_cuda_synchronize();
    const auto difference = (test - reference).abs();
    const double denominator =
        std::max(
            reference.norm().item<double>(),
            1.0e-30);
    const double relative =
        (test - reference).norm().item<double>() /
        denominator;
    const double mean_abs =
        difference.mean().item<double>();
    const double max_abs =
        difference.max().item<double>();
    std::cout
        << "tensor_parallel_check=1"
        << " tensor=" << name
        << " axis="
        << (axis == TensorParallelAxis::Output
            ? "output" : "input")
        << " logical_shape=[" << full.out()
        << ',' << full.neuron_len() << ']'
        << " shards="
        << sharded.tensor_parallel_shards.size()
        << " m=" << M
        << " relative=" << relative
        << " mean_abs=" << mean_abs
        << " max_abs=" << max_abs
        << '\n';
    const double tolerance = full.is_mxfp8()
        ? 5.0e-4
        : axis == TensorParallelAxis::Output
        ? 1.0e-6 : 5.0e-3;
    if (!mfq_tensor_backend::isfinite(test).all().item<bool>() ||
        relative > tolerance) {
        throw std::runtime_error(
            "tensor-parallel linear numerical check failed");
    }
    return 0;
}

std::vector<std::string> parse_tensor_names(const std::string & value) {
    std::vector<std::string> names;
    size_t begin = 0;
    while (begin <= value.size()) {
        const size_t end = value.find(',', begin);
        const size_t count =
            end == std::string::npos ? value.size() - begin : end - begin;
        const std::string name = value.substr(begin, count);
        if (name.empty()) {
            throw std::runtime_error("empty tensor name in comma-separated list");
        }
        names.push_back(name);
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    return names;
}

int run_linear_group_check(
        CudaExecutionContext& execution,
        const std::string & model_path,
        const std::string & names_arg,
        int M,
        int reps) {
    MFQ_RUNTIME_CHECK(M >= 1 && M <= 4096, "--check-linear-m must be in [1, 4096]");
    MFQ_RUNTIME_CHECK(reps >= 1, "--check-linear-reps must be positive");
    const auto names = parse_tensor_names(names_arg);
    MFQ_RUNTIME_CHECK(names.size() >= 2, "--check-linear-group requires at least two tensors");
    auto model_source = mfq::open_model_source(model_path);
    const auto& mfq = *model_source;
    const char * preserve_env =
        std::getenv("MFQ_CHECK_LINEAR_GROUP_PRESERVE");
    const bool preserve_projection_boundaries =
        preserve_env != nullptr && preserve_env[0] == '1';
    const char * bf16_env =
        std::getenv("MFQ_CHECK_LINEAR_GROUP_BF16");
    const bool check_bf16 =
        bf16_env != nullptr && bf16_env[0] == '1';
    auto group = load_quant_group(execution,
        mfq, names, names.size(), nullptr,
        preserve_projection_boundaries);
    const int64_t width = group.nint_grouped
        ? (group.nint.split_w.empty()
            ? group.nint.w.neuron_len
            : group.nint.split_w.front().neuron_len)
        : group.layers.front().neuron_len();
    auto sequence = mfq_tensor_backend::arange(
        (int64_t)M * width,
        mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA).dtype(mfq_tensor_backend::kFloat32));
    auto x = (((sequence.remainder(257) - 128.0) / 127.0) +
              0.03125 * mfq_tensor_backend::sin(sequence * 0.015625))
                 .to(check_bf16
                     ? mfq_tensor_backend::kBFloat16
                     : mfq_tensor_backend::kFloat16)
                 .reshape({M, width})
                 .contiguous();
    auto actual = group.forward(execution, x);
    const char * check_bf16_swiglu_env =
        std::getenv("MFQ_CHECK_BF16_SWIGLU");
    if (check_bf16_swiglu_env != nullptr &&
            check_bf16_swiglu_env[0] == '1') {
        MFQ_RUNTIME_CHECK(
            actual.size() == 2 &&
            actual[0].scalar_type() == mfq_tensor_backend::kBFloat16 &&
            actual[1].scalar_type() == mfq_tensor_backend::kBFloat16,
            "BF16 SwiGLU check requires two BF16 projection outputs");
        auto reference =
            (mfq_tensor_backend::silu(actual[0]) * actual[1]).contiguous();
        auto candidate = silu_mul_cuda(
            actual[0].contiguous(), actual[1].contiguous());
        auto difference =
            (candidate.to(mfq_tensor_backend::kFloat32) -
             reference.to(mfq_tensor_backend::kFloat32)).abs();
        std::cout << "bf16_swiglu_check"
                  << " max_abs="
                  << difference.max().item<double>()
                  << " equal="
                  << (candidate.equal(reference) ? 1 : 0)
                  << "\n";
        MFQ_RUNTIME_CHECK(
            candidate.equal(reference),
            "fused BF16 SwiGLU differs from the official BF16 expression");
    }
    mfq_cuda_synchronize();
    const auto started = std::chrono::steady_clock::now();
    for (int rep = 0; rep < reps; ++rep) {
        actual = group.forward(execution, x);
    }
    mfq_cuda_synchronize();
    const double elapsed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    std::cout << "linear_group_timing"
              << " preserve=" << (preserve_projection_boundaries ? 1 : 0)
              << " m=" << M
              << " reps=" << reps
              << " mean_ms=" << elapsed_ms / reps << '\n';
    MFQ_RUNTIME_CHECK(actual.size() == names.size(), "linear group output count mismatch");
    std::vector<mfq_tensor_backend::Tensor> graph_actual;
    if (M == 1 && decode_branch_parallel_enabled(
                    execution.config, execution.decode_graph_serial_branches, M)) {
        mfq_cuda_synchronize();
        const auto graph_stream =
            mfq_get_stream_from_pool(false);
        MfqCudaStreamGuard graph_guard(
            graph_stream);
        MfqCudaGraph graph;
        mfq_prepare_cuda_graph_memory(graph);
        graph_actual = group.forward(execution, x);
        MFQ_CUDA_CHECK(cudaStreamSynchronize(graph_stream.stream()));
        graph_actual.clear();
        graph.capture_begin();
        graph_actual = group.forward(execution, x);
        graph.capture_end();
        graph.replay();
        MFQ_CUDA_CHECK(cudaStreamSynchronize(
            graph_stream.stream()));
        MFQ_RUNTIME_CHECK(
            graph_actual.size() == actual.size(),
            "linear group CUDA Graph output count mismatch");
        for (size_t index = 0;
             index < graph_actual.size(); ++index) {
            const auto difference =
                (graph_actual[index].to(mfq_tensor_backend::kFloat32) -
                 actual[index].to(mfq_tensor_backend::kFloat32)).abs();
            const double maximum =
                difference.max().item<double>();
            std::cout
                << "linear_group_graph_check"
                << " tensor=" << names[index]
                << " max_abs=" << maximum << "\n";
            MFQ_RUNTIME_CHECK(
                maximum == 0.0,
                "linear group CUDA Graph replay differs from eager output");
        }
    }
    mfq_disable_tf32_cublas();
    std::vector<mfq_tensor_backend::Tensor> dense_weights;
    std::vector<mfq_tensor_backend::Tensor> separate_dense_references;
    std::vector<mfq_tensor_backend::Tensor> fp32_references;
    std::vector<mfq_tensor_backend::Tensor> separate_production;
    dense_weights.reserve(names.size());
    separate_dense_references.reserve(names.size());
    fp32_references.reserve(names.size());
    separate_production.reserve(names.size());
    for (size_t index = 0; index < names.size(); ++index) {
        auto linear = load_quant_linear(execution, mfq, names[index]);
        MFQ_RUNTIME_CHECK(
            linear.is_nint(),
            "--check-linear-group currently requires NINT tensors");
        const auto & weight = linear.nint;
        auto dense = weight.q8_zero
            ? nint8_zero_dequant_cuda(
                  weight.q_packed, weight.q8_zero_scale, weight.neuron_len)
            : nint_decode_cuda(
                  weight.q_packed, weight.row_q_bits,
                  weight.row_q_bit_offsets, weight.sub_scale,
                  weight.sub_min, weight.neuron_scale,
                  weight.neuron_min, weight.neuron_len, weight.gs);
        auto separate = linear.forward(execution, x);
        if (actual[index].scalar_type() == mfq_tensor_backend::kBFloat16) {
            separate = separate.to(mfq_tensor_backend::kBFloat16);
        }
        separate_production.push_back(separate.to(mfq_tensor_backend::kFloat32));
        auto dense_for_x = dense.to(x.scalar_type());
        separate_dense_references.push_back(
            mfq_tensor_backend::matmul(x, dense_for_x.transpose(0, 1))
                .to(mfq_tensor_backend::kFloat32));
        fp32_references.push_back(mfq_tensor_backend::matmul(
            x.to(mfq_tensor_backend::kFloat32),
            dense.to(mfq_tensor_backend::kFloat32).transpose(0, 1)));
        dense_weights.push_back(std::move(dense));
    }
    auto combined_dense = mfq_tensor_backend::cat(dense_weights, 0)
        .to(x.scalar_type()).contiguous();
    auto combined_output =
        mfq_tensor_backend::matmul(x, combined_dense.transpose(0, 1)).to(mfq_tensor_backend::kFloat32);
    auto combined_references =
        combined_output.split_with_sizes(group.outs, -1);
    for (size_t index = 0; index < names.size(); ++index) {
        auto combined_reference = combined_references[index];
        auto separate_dense_reference = separate_dense_references[index];
        auto fp32_reference = fp32_references[index];
        auto separate_candidate = separate_production[index];
        auto candidate = actual[index].to(mfq_tensor_backend::kFloat32);
        auto combined_dense_difference = candidate - combined_reference;
        auto grouped_vs_separate = candidate - separate_candidate;
        auto separate_dense_difference =
            separate_candidate - separate_dense_reference;
        auto grouped_fp32_difference = candidate - fp32_reference;
        auto separate_fp32_difference = separate_candidate - fp32_reference;
        const double grouped_vs_separate_rel =
            (grouped_vs_separate.norm() /
             separate_candidate.norm().clamp_min(1.0e-30)).item<double>();
        const double grouped_vs_separate_snr =
            grouped_vs_separate_rel == 0.0
                ? std::numeric_limits<double>::infinity()
                : -20.0 * std::log10(grouped_vs_separate_rel);
        const double grouped_fp32_rel =
            (grouped_fp32_difference.norm() /
             fp32_reference.norm().clamp_min(1.0e-30)).item<double>();
        const double separate_fp32_rel =
            (separate_fp32_difference.norm() /
             fp32_reference.norm().clamp_min(1.0e-30)).item<double>();
        std::cout << std::fixed << std::setprecision(9)
                  << "linear_group_check"
                  << " tensor=" << names[index]
                  << " m=" << M
                  << " n=" << combined_reference.size(1)
                  << " k=" << width
                  << " grouped_vs_combined_dense_rel="
                  << (combined_dense_difference.norm() /
                      combined_reference.norm().clamp_min(1.0e-30)).item<double>()
                  << " grouped_vs_separate_rel=" << grouped_vs_separate_rel
                  << " grouped_vs_separate_snr_db=" << grouped_vs_separate_snr
                  << " grouped_vs_separate_mean_abs="
                  << grouped_vs_separate.abs().mean().item<double>()
                  << " grouped_vs_separate_max_abs="
                  << grouped_vs_separate.abs().max().item<double>()
                  << " separate_vs_separate_dense_rel="
                  << (separate_dense_difference.norm() /
                      separate_dense_reference.norm().clamp_min(1.0e-30)).item<double>()
                  << " grouped_vs_fp32_rel=" << grouped_fp32_rel
                  << " grouped_vs_fp32_snr_db="
                  << (grouped_fp32_rel == 0.0
                          ? std::numeric_limits<double>::infinity()
                          : -20.0 * std::log10(grouped_fp32_rel))
                  << " separate_vs_fp32_rel=" << separate_fp32_rel
                  << " separate_vs_fp32_snr_db="
                  << (separate_fp32_rel == 0.0
                          ? std::numeric_limits<double>::infinity()
                          : -20.0 * std::log10(separate_fp32_rel))
                  << "\n";
    }
    return 0;
}

static mfq_tensor_backend::Tensor read_f32_tensor(
        const std::filesystem::path & path,
        const std::vector<int64_t> & shape) {
    const int64_t count = std::accumulate(
        shape.begin(), shape.end(), int64_t{1}, std::multiplies<int64_t>());
    std::vector<float> values(static_cast<size_t>(count));
    std::ifstream input(path, std::ios::binary);
    MFQ_RUNTIME_CHECK(input, "failed to open ", path.string());
    input.read(
        reinterpret_cast<char *>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(float)));
    MFQ_RUNTIME_CHECK(
        input.gcount() ==
            static_cast<std::streamsize>(values.size() * sizeof(float)),
        "short f32 tensor read from ", path.string());
    return mfq_tensor_backend::from_blob(
               values.data(), shape,
               mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kFloat32))
        .clone()
        .to(mfq_tensor_backend::kCUDA)
        .contiguous();
}

static void write_f32_tensor(
        const std::filesystem::path & path,
        const mfq_tensor_backend::Tensor & tensor) {
    auto host = tensor.to(mfq_tensor_backend::kCPU).to(mfq_tensor_backend::kFloat32).contiguous();
    std::ofstream output(path, std::ios::binary);
    MFQ_RUNTIME_CHECK(output, "failed to create ", path.string());
    output.write(
        reinterpret_cast<const char *>(host.data_ptr<float>()),
        static_cast<std::streamsize>(host.numel() * sizeof(float)));
    MFQ_RUNTIME_CHECK(output, "failed to write ", path.string());
}

int run_gdn_operator_check(
        const std::string & input_dir,
        const std::string & output_path,
        const std::string & state_path,
        int64_t tokens,
        int64_t q_heads,
        int64_t v_heads,
        int64_t head_dim) {
    MFQ_RUNTIME_CHECK(
        tokens >= 1 && q_heads >= 1 && v_heads >= q_heads && head_dim >= 1,
        "invalid GDN diagnostic shape");
    MFQ_RUNTIME_CHECK(
        v_heads % q_heads == 0,
        "GDN diagnostic value heads must be divisible by query heads");
    const std::filesystem::path root(input_dir);
    auto q = read_f32_tensor(
        root / "q.bin", {1, tokens, q_heads, head_dim})
                 .permute({0, 2, 1, 3})
                 .contiguous();
    auto k = read_f32_tensor(
        root / "k.bin", {1, tokens, q_heads, head_dim})
                 .permute({0, 2, 1, 3})
                 .contiguous();
    auto v = read_f32_tensor(
        root / "v.bin", {1, tokens, v_heads, head_dim})
                 .permute({0, 2, 1, 3})
                 .contiguous();
    auto g = read_f32_tensor(
        root / "g.bin", {1, tokens, v_heads})
                 .permute({0, 2, 1})
                 .contiguous();
    auto beta = read_f32_tensor(
        root / "beta.bin", {1, tokens, v_heads})
                    .permute({0, 2, 1})
                    .contiguous();
    auto state = read_f32_tensor(
        root / "state.bin", {1, v_heads, head_dim, head_dim});
    auto result = gdn_inplace_transposed_tiled_cuda(
        q, k, v, g, beta, state);
    write_f32_tensor(output_path, result[0]);
    write_f32_tensor(state_path, result[1]);
    std::cout << "mfq_gdn_operator"
              << " t=" << tokens
              << " hq=" << q_heads
              << " hv=" << v_heads
              << " d=" << head_dim
              << " output=" << output_path
              << " state=" << state_path << "\n";
    return 0;
}

int run_linear_conv_operator_check(
        const std::string & input_dir,
        const std::string & output_dir,
        int64_t tokens,
        int64_t q_heads,
        int64_t v_heads,
        int64_t key_dim,
        int64_t value_dim,
        int64_t kernel_size,
        double eps) {
    MFQ_RUNTIME_CHECK(
        tokens >= 2 && q_heads >= 1 && v_heads >= 1 &&
            key_dim >= 1 && value_dim >= 1 &&
            kernel_size >= 2 && kernel_size <= 8,
        "invalid linear-conv diagnostic shape");
    const int64_t qk_width = 2 * q_heads * key_dim;
    const int64_t v_width = v_heads * value_dim;
    const int64_t channels = qk_width + v_width;
    const std::filesystem::path input_root(input_dir);
    const std::filesystem::path output_root(output_dir);
    std::filesystem::create_directories(output_root);
    auto state = read_f32_tensor(
        input_root / "state.bin",
        {1, kernel_size - 1, channels});
    auto qk = read_f32_tensor(
                  input_root / "qk.bin",
                  {1, tokens, qk_width})
                  .to(mfq_tensor_backend::kHalf)
                  .contiguous();
    auto v = read_f32_tensor(
                 input_root / "v.bin",
                 {1, tokens, v_width})
                 .to(mfq_tensor_backend::kHalf)
                 .contiguous();
    auto weight = read_f32_tensor(
        input_root / "weight.bin",
        {channels, 1, kernel_size});
    mfq_tensor_backend::Tensor bias;
    if (std::filesystem::exists(input_root / "bias.bin")) {
        bias = read_f32_tensor(
            input_root / "bias.bin", {channels});
    } else {
        bias = mfq_tensor_backend::empty(
            {0},
            mfq_tensor_backend::TensorOptions()
                .device(mfq_tensor_backend::kCUDA)
                .dtype(mfq_tensor_backend::kFloat32));
    }
    auto result = linear_conv_qkv_prefill_cuda(
        state, qk, v, weight, bias,
        q_heads, v_heads, key_dim, value_dim, eps);
    write_f32_tensor(output_root / "q.bin", result[0]);
    write_f32_tensor(output_root / "k.bin", result[1]);
    write_f32_tensor(output_root / "v.bin", result[2]);
    write_f32_tensor(output_root / "state.bin", result[3]);
    std::cout << "mfq_linear_conv_operator"
              << " t=" << tokens
              << " hq=" << q_heads
              << " hv=" << v_heads
              << " dk=" << key_dim
              << " dv=" << value_dim
              << " kernel=" << kernel_size
              << " output=" << output_dir << "\n";
    return 0;
}

int run_q8_embedding_check(
        CudaExecutionContext& execution,
        const std::string & model_path,
        const std::string & name) {
    auto model_source = mfq::open_model_source(model_path);
    const auto& mfq = *model_source;
    auto linear = load_quant_linear(execution, mfq, name);
    MFQ_RUNTIME_CHECK(
        linear.is_nint() && linear.nint.q8_zero,
        "--check-q8-embedding requires an NINT8-0 tensor");
    const int64_t vocab = linear.nint.out;
    std::vector<int64_t> host_ids = {
        0,
        std::min<int64_t>(1, vocab - 1),
        std::min<int64_t>(106, vocab - 1),
        std::min<int64_t>(12345, vocab - 1),
        std::min<int64_t>(255999, vocab - 1),
        vocab - 1,
    };
    auto ids = mfq_tensor_backend::from_blob(
        host_ids.data(), {(int64_t)host_ids.size()},
        mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64))
                   .clone()
                   .to(mfq_tensor_backend::kCUDA)
                   .contiguous();
    auto candidate = nint8_zero_embedding_lookup_cuda(
        linear.nint.q_packed, linear.nint.q8_zero_scale,
        ids, linear.nint.neuron_len);
    auto dense = nint8_zero_dequant_cuda(
        linear.nint.q_packed, linear.nint.q8_zero_scale,
        linear.nint.neuron_len);
    auto reference = dense.index_select(0, ids);
    auto difference =
        candidate.to(mfq_tensor_backend::kFloat32) - reference.to(mfq_tensor_backend::kFloat32);
    std::cout << std::fixed << std::setprecision(9)
              << "q8_embedding_check"
              << " tensor=" << name
              << " ids=" << host_ids.size()
              << " vocab=" << vocab
              << " width=" << linear.nint.neuron_len
              << " equal=" << (candidate.equal(reference) ? 1 : 0)
              << " rel="
              << (difference.norm() /
                  reference.to(mfq_tensor_backend::kFloat32).norm()).item<double>()
              << " mean_abs=" << difference.abs().mean().item<double>()
              << " max_abs=" << difference.abs().max().item<double>()
              << "\n";
    return 0;
}

int run_dsv4_output_a_check(
        CudaExecutionContext& execution,
        const std::string & model_path,
        const std::string & name,
        int batch,
        int reps) {
    constexpr int64_t kGroups = 8;
    MFQ_RUNTIME_CHECK(batch > 0 && reps > 0, "DSV4 output_a check requires positive batch and reps");
    auto model_source = mfq::open_model_source(model_path);
    const auto& mfq = *model_source;
    auto linear = load_quant_linear(execution,
        mfq, name, TensorParallelAxis::Input);
    const bool supported_nint =
        linear.is_nint() &&
        linear.nint.bits == 8 &&
        linear.nint.gs == 48;
    MFQ_RUNTIME_CHECK(
        supported_nint || linear.is_mxfp8(),
        "DSV4 output_a check requires NINT8 gs48 or MXFP8");
    MFQ_RUNTIME_CHECK(
        linear.out() % kGroups == 0,
        "DSV4 output_a rows must divide eight groups");

    const int64_t width = linear.neuron_len();
    const int64_t rows_per_group = linear.out() / kGroups;
    auto sequence = mfq_tensor_backend::arange(
        (int64_t)batch * kGroups * width,
        mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA).dtype(mfq_tensor_backend::kFloat32));
    auto grouped = (
        (sequence.remainder(257) - 128.0) / 127.0 +
        0.03125 * mfq_tensor_backend::sin(sequence * 0.015625))
        .to(mfq_tensor_backend::kFloat16)
        .reshape({batch, kGroups, width})
        .contiguous();

    auto legacy = [&]() {
        auto expanded = linear.forward(
            execution, grouped.reshape({batch * kGroups, width}))
            .reshape({batch, kGroups, kGroups, rows_per_group});
        std::vector<mfq_tensor_backend::Tensor> diagonal;
        diagonal.reserve(kGroups);
        for (int64_t group = 0; group < kGroups; ++group) {
            diagonal.push_back(
                expanded.index({Slice(), group, group, Slice()}));
        }
        return mfq_tensor_backend::stack(diagonal, 1).reshape({batch, linear.out()});
    };
    auto groupwise = [&]() {
        return linear.is_mxfp8()
            ? linear.forward_mxfp8_groupwise(
                execution, grouped, kGroups)
            : nint_matmul_groupwise_u8(
                execution.profiler, linear.nint, grouped, kGroups);
    };
    auto time_ms = [&](auto && fn) {
        mfq_tensor_backend::Tensor output;
        for (int warmup = 0; warmup < 10; ++warmup) output = fn();
        mfq_cuda_synchronize();
        cudaEvent_t start, stop;
        MFQ_CUDA_CHECK(cudaEventCreate(&start));
        MFQ_CUDA_CHECK(cudaEventCreate(&stop));
        auto stream = mfq_get_current_cuda_stream().stream();
        MFQ_CUDA_CHECK(cudaEventRecord(start, stream));
        for (int iteration = 0; iteration < reps; ++iteration) output = fn();
        MFQ_CUDA_CHECK(cudaEventRecord(stop, stream));
        MFQ_CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed = 0.0f;
        MFQ_CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
        MFQ_CUDA_CHECK(cudaEventDestroy(start));
        MFQ_CUDA_CHECK(cudaEventDestroy(stop));
        return std::pair<float, mfq_tensor_backend::Tensor>(elapsed / reps, output);
    };

    auto legacy_result = time_ms(legacy);
    auto groupwise_result = time_ms(groupwise);
    auto reference = legacy_result.second.to(mfq_tensor_backend::kFloat32);
    auto candidate = groupwise_result.second.to(mfq_tensor_backend::kFloat32);
    auto diff = (candidate - reference).abs();
    const float relative =
        ((candidate - reference).norm() / reference.norm()).item<float>();
    std::cout << std::fixed << std::setprecision(9)
              << "dsv4_output_a_check"
              << " tensor=" << name
              << " format=" << (linear.is_mxfp8() ? "MXFP8" : "NINT8")
              << " batch=" << batch
              << " groups=" << kGroups
              << " rows_per_group=" << rows_per_group
              << " k=" << width
              << " equal=" << (candidate.equal(reference) ? 1 : 0)
              << " rel=" << relative
              << " mean_abs=" << diff.mean().item<float>()
              << " max_abs=" << diff.max().item<float>()
              << " legacy_ms=" << legacy_result.first
              << " groupwise_ms=" << groupwise_result.first
              << " speedup=" << legacy_result.first / groupwise_result.first
              << " checksum=" << candidate.sum().item<double>()
              << "\n";
    if (linear.is_mxfp8()) {
        MFQ_RUNTIME_CHECK(
            mfq_tensor_backend::isfinite(candidate).all().item<bool>() &&
                relative <= 5.0e-4f,
            "DSV4 MXFP8 groupwise output_a exceeded the FP16 GEMM "
            "reduction-order tolerance");
    } else {
        MFQ_RUNTIME_CHECK(
            candidate.equal(reference),
            "DSV4 NINT groupwise output_a must be bit-exact with the "
            "legacy path");
    }
    return 0;
}

int run_gemma_geglu_check(
    CudaExecutionContext& execution,
    const std::string & model_path,
    int layer,
    int reps) {
    if (layer < 0 || reps < 1) {
        throw std::runtime_error("Gemma GeGLU check requires a nonnegative layer and positive reps");
    }
    auto model_source = mfq::open_model_source(model_path);
    const auto& mfq = *model_source;
    mfq::cuda::validate_model_source(mfq);
    const std::string prefix =
        "model.block." + std::to_string(layer) + ".mlp.";
    auto gate_up = load_quant_group(execution,
        mfq, {prefix + "gate.weight", prefix + "up.weight"}, 2);
    auto down = load_quant_linear(execution, mfq, prefix + "down.weight");
    if (!gate_up.nint_grouped || !gate_up.nint.split_w.empty() || !down.is_nint() ||
        gate_up.outs.size() != 2 || gate_up.outs[0] != gate_up.outs[1]) {
        throw std::runtime_error("Gemma GeGLU check requires packed NINT gate/up and NINT down tensors");
    }

    const int64_t hidden = gate_up.nint.w.neuron_len;
    auto xf = mfq_tensor_backend::arange(
        hidden, mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA).dtype(mfq_tensor_backend::kFloat32));
    auto x = (((xf.remainder(257) - 128.0) / 64.0) +
              0.125 * mfq_tensor_backend::sin(xf * 0.03125)).to(mfq_tensor_backend::kFloat16).reshape({1, hidden}).contiguous();

    auto materialized_activation = [&]() {
        auto parts = gate_up.forward(execution, x);
        return gelu_mul_cuda(parts[0].contiguous(), parts[1].contiguous());
    };
    auto materialized = [&]() {
        return down.forward(execution, materialized_activation());
    };
    auto fused_activation = [&]() {
        return gate_up.forward_geglu(execution, x);
    };
    auto fused = [&]() {
        return down.forward(execution, fused_activation());
    };

    auto reference_activation = materialized_activation();
    auto reference_output = down.forward(execution, reference_activation);
    std::cout << "gemma_geglu_check layer=" << layer
              << " gate_up_bits=" << gate_up.nint.w.bits
              << " gate_up_gs=" << gate_up.nint.w.gs
              << " gate_out=" << gate_up.outs[0]
              << " up_out=" << gate_up.outs[1]
              << " down_bits=" << down.nint.bits
              << " down_gs=" << down.nint.gs << "\n";
    auto report = [&](const char * name, mfq_tensor_backend::Tensor value, mfq_tensor_backend::Tensor reference) {
        auto got = value.to(mfq_tensor_backend::kFloat64);
        auto ref = reference.to(mfq_tensor_backend::kFloat64);
        const double ref_norm = std::max(ref.norm().item<double>(), 1.0e-30);
        const double got_norm = std::max(got.norm().item<double>(), 1.0e-30);
        std::cout << "gemma_geglu_check path=" << name
                  << " relative_l2=" << (got - ref).norm().item<double>() / ref_norm
                  << " cosine=" << mfq_tensor_backend::dot(got.reshape({-1}), ref.reshape({-1})).item<double>() /
                         (got_norm * ref_norm)
                  << " max_abs=" << (got - ref).abs().max().item<double>() << "\n";
    };

    report("activation_combined", fused_activation(), reference_activation);
    report("output_combined", fused(), reference_output);
    report("activation_pair", fused_activation(), reference_activation);
    report("output_pair", fused(), reference_output);

    auto time_ms = [&](auto && fn) {
        for (int i = 0; i < 10; ++i) (void)fn();
        mfq_cuda_synchronize();
        cudaEvent_t start, stop;
        MFQ_CUDA_CHECK(cudaEventCreate(&start));
        MFQ_CUDA_CHECK(cudaEventCreate(&stop));
        auto stream = mfq_get_current_cuda_stream().stream();
        MFQ_CUDA_CHECK(cudaEventRecord(start, stream));
        for (int i = 0; i < reps; ++i) (void)fn();
        MFQ_CUDA_CHECK(cudaEventRecord(stop, stream));
        MFQ_CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed = 0.0f;
        MFQ_CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
        MFQ_CUDA_CHECK(cudaEventDestroy(start));
        MFQ_CUDA_CHECK(cudaEventDestroy(stop));
        return elapsed / reps;
    };
    const float materialized_ms = time_ms(materialized);
    const float combined_ms = time_ms(fused);
    const float pair_ms = time_ms(fused);
    std::cout << "gemma_geglu_check layer=" << layer
              << " gate_bits=" << gate_up.nint.w.bits
              << " gate_gs=" << gate_up.nint.w.gs
              << " down_bits=" << down.nint.bits
              << " down_gs=" << down.nint.gs
              << " materialized_ms=" << materialized_ms
              << " combined_ms=" << combined_ms
              << " pair_ms=" << pair_ms << "\n";
    return 0;
}


