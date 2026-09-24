#pragma once

#include "runtime/generation.h"
#include "diagnostics/flash_next_mtp.h"
#include "quant_linear.h"
#include "registry.h"
#include "cuda_execution.h"
#include "qwen35/qwen35_linear_attention.h"
#include "mfq/kernels/cuda/deepseek_v41.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mfq::cuda::diagnostics {

using namespace mfq::cuda::internal;

template <typename Model>
static int run_prefill_sweep(
    Model& model,
    const std::vector<int64_t> & sizes,
    int repeats) {
    if (repeats < 1) throw std::runtime_error("--prefill-sweep-reps must be positive");
    const int64_t max_m = *std::max_element(sizes.begin(), sizes.end());
    if (max_m > model.max_position_embeddings()) {
        throw std::runtime_error("--prefill-sweep exceeds the configured context size");
    }

    std::vector<int64_t> token_ids((size_t)max_m);
    const int64_t token_span = std::max<int64_t>(
        1, std::min<int64_t>(1024, model.vocab_size() - 2));
    for (int64_t i = 0; i < max_m; ++i) token_ids[(size_t)i] = 1 + i % token_span;
    auto all_ids = mfq_tensor_backend::tensor(
        token_ids, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64).device(mfq_tensor_backend::kCUDA)).unsqueeze(0);

    for (int64_t m : sizes) {
        auto ids = all_ids.narrow(1, 0, m);
        for (int warmup = 0; warmup < 2; ++warmup) {
            model.reset(1);
            (void)model.last_logits(ids);
            mfq_cuda_synchronize();
        }

        std::vector<double> elapsed_ms;
        elapsed_ms.reserve((size_t)repeats);
        int64_t top = -1;
        for (int repeat = 0; repeat < repeats; ++repeat) {
            model.reset(1);
            auto started = std::chrono::steady_clock::now();
            auto logits = model.last_logits(ids);
            mfq_cuda_synchronize();
            auto ended = std::chrono::steady_clock::now();
            elapsed_ms.push_back(std::chrono::duration<double, std::milli>(ended - started).count());
            if (repeat == 0) top = logits.argmax(-1).template item<int64_t>();
        }
        std::sort(elapsed_ms.begin(), elapsed_ms.end());
        const double median_ms = elapsed_ms[elapsed_ms.size() / 2];
        std::cout << "prefill_sweep_m=" << m
                  << " median_ms=" << median_ms
                  << " min_ms=" << elapsed_ms.front()
                  << " max_ms=" << elapsed_ms.back()
                  << " tok_per_s=" << (1000.0 * (double)m / median_ms)
                  << " top=" << top << "\n";
    }
    return 0;
}

template <typename Model>
static int run_block_trace_compare(
    Model& test,
    const std::string & reference_model_path,
    const std::string & config_path,
    int64_t context_size,
    mfq_tensor_backend::Tensor ids)
{
    Model reference = [&]() {
        if (g_n_gpu_layers < 0) {
            return mfq::cuda::load_causal_lm<Model::backbone>(
                reference_model_path, config_path, context_size);
        }
        const int saved_n_gpu_layers = g_n_gpu_layers;
        const int saved_cpu_layers = g_dense_cpu_layer_count;
        g_n_gpu_layers = -1;
        try {
            auto loaded = mfq::cuda::load_causal_lm<Model::backbone>(
                reference_model_path, config_path, context_size);
            g_n_gpu_layers = saved_n_gpu_layers;
            g_dense_cpu_layer_count = saved_cpu_layers;
            return loaded;
        } catch (...) {
            g_n_gpu_layers = saved_n_gpu_layers;
            g_dense_cpu_layer_count = saved_cpu_layers;
            throw;
        }
    }();
    std::vector<mfq_tensor_backend::Tensor> test_trace;
    std::vector<mfq_tensor_backend::Tensor> reference_trace;

    test.reset(ids.size(0));
    auto test_hidden = test.hidden_forward(ids, mfq_nullopt, mfq_nullopt, &test_trace);
    reference.reset(ids.size(0));
    auto reference_hidden = reference.hidden_forward(
        ids, mfq_nullopt, mfq_nullopt, &reference_trace);
    mfq_cuda_synchronize();

    if (test_trace.size() != reference_trace.size()) {
        throw std::runtime_error("block trace stage count mismatch");
    }
    for (size_t i = 0; i < test_trace.size(); ++i) {
        auto ref = reference_trace[i].reshape({-1}).to(mfq_tensor_backend::kFloat64);
        auto got = test_trace[i].reshape({-1}).to(mfq_tensor_backend::kFloat64);
        if (ref.numel() != got.numel()) {
            throw std::runtime_error("block trace tensor size mismatch");
        }
        auto ref_norm = ref.norm();
        auto got_norm = got.norm();
        const double ref_norm_value = ref_norm.template item<double>();
        const double denominator = std::max(ref_norm_value, 1.0e-30);
        const double relative_l2 = (got - ref).norm().template item<double>() / denominator;
        const double cosine = mfq_tensor_backend::dot(ref, got).template item<double>() /
            std::max(ref_norm_value * got_norm.template item<double>(), 1.0e-30);
        const double norm_ratio = got_norm.template item<double>() / denominator;
        const double reference_rms = ref.square().mean().sqrt().template item<double>();
        const double test_rms = got.square().mean().sqrt().template item<double>();
        const std::string stage = i == 0
            ? "embedding"
            : "block_" + std::to_string(i - 1);
        std::cout << "block_trace stage=" << stage
                  << " relative_l2=" << relative_l2
                  << " cosine=" << cosine
                  << " norm_ratio=" << norm_ratio
                  << " reference_rms=" << reference_rms
                  << " test_rms=" << test_rms << "\n";
    }

    auto reference_logits = reference.lm_head.forward(reference_hidden).to(mfq_tensor_backend::kFloat32);
    auto test_logits = test.lm_head.forward(test_hidden).to(mfq_tensor_backend::kFloat32);
    double kl_sum = 0.0;
    int64_t same_top = 0;
    int64_t rows = 0;
    const int64_t tokens = reference_logits.numel() / reference_logits.size(-1);
    auto ref2 = reference_logits.reshape({tokens, -1});
    auto got2 = test_logits.reshape({tokens, -1});
    for (int64_t start = 0; start < tokens; start += 8) {
        const int64_t end = std::min(start + 8, tokens);
        auto ref_chunk = ref2.index({Slice(start, end)});
        auto got_chunk = got2.index({Slice(start, end)});
        auto ref_logp = mfq_tensor_backend::log_softmax(ref_chunk, -1);
        auto got_logp = mfq_tensor_backend::log_softmax(got_chunk, -1);
        kl_sum += (ref_logp.exp() * (ref_logp - got_logp)).sum(-1)
            .to(mfq_tensor_backend::kFloat64).sum().template item<double>();
        same_top += ref_chunk.argmax(-1).eq(got_chunk.argmax(-1)).sum().template item<int64_t>();
        rows += end - start;
    }
    const double logits_relative_l2 =
        (test_logits.to(mfq_tensor_backend::kFloat64) - reference_logits.to(mfq_tensor_backend::kFloat64)).norm().template item<double>() /
        std::max(reference_logits.to(mfq_tensor_backend::kFloat64).norm().template item<double>(), 1.0e-30);
    std::cout << "block_trace_logits kld=" << (kl_sum / std::max<int64_t>(rows, 1))
              << " same_top=" << ((double)same_top / std::max<int64_t>(rows, 1))
              << " relative_l2=" << logits_relative_l2 << "\n";
    return 0;
}

template <typename Model>
static int run_block_trace_dump(
    Model& model,
    const std::string & output_dir,
    mfq_tensor_backend::Tensor ids,
    int64_t token_start,
    int64_t token_count)
{
    const int64_t total_tokens = ids.size(1);
    if (token_start < 0 || token_start >= total_tokens) {
        throw std::runtime_error("--dump-block-trace-start is outside the token range");
    }
    if (token_count <= 0) token_count = total_tokens - token_start;
    if (token_count > total_tokens - token_start) {
        throw std::runtime_error("--dump-block-trace-count exceeds the token range");
    }
    const std::filesystem::path root(output_dir);
    std::error_code error;
    if (!std::filesystem::create_directories(root, error) || error) {
        throw std::runtime_error(
            "block trace output directory must be new: " + output_dir);
    }

    model.reset(ids.size(0));
    std::vector<mfq_tensor_backend::Tensor> trace;
    auto final_hidden = model.hidden_forward(
        ids, mfq_nullopt, mfq_nullopt, &trace);
    mfq_cuda_synchronize();

    auto ids_cpu = ids.to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt32).contiguous();
    {
        std::ofstream output(root / "tokens.i32", std::ios::binary);
        if (!output) throw std::runtime_error("cannot create block trace token file");
        output.write(
            reinterpret_cast<const char *>(ids_cpu.template data_ptr<int32_t>()),
            static_cast<std::streamsize>(ids_cpu.nbytes()));
        if (!output) throw std::runtime_error("failed to write block trace tokens");
    }

    std::ofstream metadata(root / "trace_meta.jsonl");
    if (!metadata) throw std::runtime_error("cannot create block trace metadata");
    auto token_slice = [&](mfq_tensor_backend::Tensor value) {
        if (value.dim() >= 3 && value.size(1) == total_tokens) {
            return value.narrow(1, token_start, token_count);
        }
        return value;
    };
    for (size_t index = 0; index < trace.size(); ++index) {
        auto value = token_slice(trace[index])
            .to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kFloat32).contiguous();
        const std::string stage = index == 0
            ? "embedding"
            : "block_" + std::to_string(index - 1);
        const std::filesystem::path file = root / (stage + ".f32");
        std::ofstream output(file, std::ios::binary);
        if (!output) {
            throw std::runtime_error("cannot create block trace tensor: " + file.string());
        }
        output.write(
            reinterpret_cast<const char *>(value.template data_ptr<float>()),
            static_cast<std::streamsize>(value.nbytes()));
        if (!output) {
            throw std::runtime_error("failed to write block trace tensor: " + file.string());
        }
        metadata << "{\"stage\":\"" << stage << "\",\"file\":\""
                 << file.filename().string() << "\",\"shape\":[";
        for (int64_t dim = 0; dim < value.dim(); ++dim) {
            if (dim) metadata << ',';
            metadata << value.size(dim);
        }
        metadata << "],\"dtype\":\"float32\"}\n";
        std::cout << "block_trace_dump stage=" << stage
                  << " values=" << value.numel() << "\n";
    }
    auto dump_terminal = [&](const std::string & stage, mfq_tensor_backend::Tensor value) {
        value = value.to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kFloat32).contiguous();
        const std::filesystem::path file = root / (stage + ".f32");
        std::ofstream output(file, std::ios::binary);
        if (!output) {
            throw std::runtime_error("cannot create block trace tensor: " + file.string());
        }
        output.write(
            reinterpret_cast<const char *>(value.template data_ptr<float>()),
            static_cast<std::streamsize>(value.nbytes()));
        if (!output) {
            throw std::runtime_error("failed to write block trace tensor: " + file.string());
        }
        metadata << "{\"stage\":\"" << stage << "\",\"file\":\""
                 << file.filename().string() << "\",\"shape\":[";
        for (int64_t dim = 0; dim < value.dim(); ++dim) {
            if (dim) metadata << ',';
            metadata << value.size(dim);
        }
        metadata << "],\"dtype\":\"float32\"}\n";
        metadata.flush();
        std::cout << "block_trace_dump stage=" << stage
                  << " values=" << value.numel() << "\n";
    };
    final_hidden = token_slice(final_hidden);
    dump_terminal("final_norm", final_hidden);
    auto logits = model.apply_final_logit_softcap(
        model.lm_head.forward(final_hidden));
    dump_terminal("logits", logits);
    metadata.flush();
    if (!metadata) throw std::runtime_error("failed to write block trace metadata");
    return 0;
}

template <typename Model>
static int run_dsv4_hc_model_compare(
    Model& model,
    mfq_tensor_backend::Tensor ids)
{
    std::vector<mfq_tensor_backend::Tensor> reference_trace;
    std::vector<mfq_tensor_backend::Tensor> candidate_trace;
    std::vector<mfq_tensor_backend::Tensor> repeat_trace;

    g_dsv4_fused_hc = false;
    model.reset(ids.size(0));
    auto reference_hidden = model.hidden_forward(
        ids, mfq_nullopt, mfq_nullopt, &reference_trace);
    auto reference_logits =
        model.lm_head.forward(reference_hidden).to(mfq_tensor_backend::kFloat32);

    g_dsv4_fused_hc = true;
    model.reset(ids.size(0));
    auto candidate_hidden = model.hidden_forward(
        ids, mfq_nullopt, mfq_nullopt, &candidate_trace);
    auto candidate_logits =
        model.lm_head.forward(candidate_hidden).to(mfq_tensor_backend::kFloat32);

    g_dsv4_fused_hc = false;
    model.reset(ids.size(0));
    auto repeat_hidden = model.hidden_forward(
        ids, mfq_nullopt, mfq_nullopt, &repeat_trace);
    auto repeat_logits =
        model.lm_head.forward(repeat_hidden).to(mfq_tensor_backend::kFloat32);
    g_dsv4_fused_hc = true;
    mfq_cuda_synchronize();

    if (reference_trace.size() != candidate_trace.size() ||
            reference_trace.size() != repeat_trace.size()) {
        throw std::runtime_error(
            "DeepSeek V4 HC trace stage count mismatch");
    }
    for (size_t index = 0; index < reference_trace.size(); ++index) {
        auto reference = reference_trace[index].reshape({-1});
        auto candidate = candidate_trace[index].reshape({-1});
        auto repeat = repeat_trace[index].reshape({-1});
        auto reference_f64 = reference.to(mfq_tensor_backend::kFloat64);
        auto candidate_f64 = candidate.to(mfq_tensor_backend::kFloat64);
        const double denominator = std::max(
            reference_f64.norm().template item<double>(), 1.0e-30);
        const std::string stage = index == 0
            ? "embedding"
            : "block_" + std::to_string(index - 1);
        std::cout << std::scientific << std::setprecision(9)
                  << "dsv4_hc_model_trace stage=" << stage
                  << " differing="
                  << candidate.ne(reference).sum().template item<int64_t>()
                  << " rel_l2="
                  << (candidate_f64 - reference_f64)
                         .norm().template item<double>() / denominator
                  << " mean_abs="
                  << (candidate_f64 - reference_f64)
                         .abs().mean().template item<double>()
                  << " max_abs="
                  << (candidate_f64 - reference_f64)
                         .abs().max().template item<double>()
                  << " repeat_differing="
                  << repeat.ne(reference).sum().template item<int64_t>()
                  << "\n";
    }

    auto reference_logp = mfq_tensor_backend::log_softmax(reference_logits, -1);
    auto candidate_logp = mfq_tensor_backend::log_softmax(candidate_logits, -1);
    const double kld_candidate_reference = (
        candidate_logp.exp() *
        (candidate_logp - reference_logp))
        .sum(-1).mean().template item<double>();
    const double kld_reference_candidate = (
        reference_logp.exp() *
        (reference_logp - candidate_logp))
        .sum(-1).mean().template item<double>();
    auto logit_diff = (
        candidate_logits.to(mfq_tensor_backend::kFloat64) -
        reference_logits.to(mfq_tensor_backend::kFloat64));
    std::cout << std::scientific << std::setprecision(9)
              << "dsv4_hc_model_logits"
              << " mean_kld_candidate_reference="
              << kld_candidate_reference
              << " mean_kld_reference_candidate="
              << kld_reference_candidate
              << " relative_l2="
              << logit_diff.norm().template item<double>() /
                    std::max(
                        reference_logits.to(mfq_tensor_backend::kFloat64)
                            .norm().template item<double>(),
                        1.0e-30)
              << " mean_abs="
              << logit_diff.abs().mean().template item<double>()
              << " max_abs="
              << logit_diff.abs().max().template item<double>()
              << " same_top="
              << candidate_logits.argmax(-1)
                     .eq(reference_logits.argmax(-1))
                     .to(mfq_tensor_backend::kFloat32).mean().template item<double>()
              << " repeat_logits_equal="
              << (repeat_logits.equal(reference_logits) ? 1 : 0)
              << "\n";
    return 0;
}

// Real-weight correctness gate; does not require a tokenizer or start a runtime transport.
// It calls the same MTP generator used by the runtime, with synthetic token IDs.
static int run_qwen35_mtp_check(
        mfq::cuda::Qwen35CausalLm& model, Qwen35Mtp& mtp) {
    using Tensor = mfq_tensor_backend::Tensor;
    const MtpTarget target{
        [&model](Tensor ids) {
            return model.embed_forward(std::move(ids));
        },
        [&model](Tensor hidden) {
            return model.logits_from_hidden(std::move(hidden));
        },
        &model.rope,
    };
    // Identity projection isolates both dense gate modes and their dtype casts.
    const auto float_options = mfq_tensor_backend::TensorOptions()
        .device(mfq_tensor_backend::kCUDA).dtype(mfq_tensor_backend::kFloat32);
    std::vector<float> identity(33 * 33, 0.f), input(6 * 33), gate(6 * 33);
    for (int i = 0; i < 33; ++i) identity[i * 33 + i] = 1.f;
    for (int i = 0; i < 6 * 33; ++i) {
        input[i] = static_cast<float>(i % 33 - 16) / 16.f;
        gate[i] = static_cast<float>(i % 31 - 15) / 4.f;
    }
    auto test_input = mfq_tensor_backend::tensor(input, float_options).reshape({1, 6, 33});
    auto test_gate = mfq_tensor_backend::tensor(gate, float_options).reshape({1, 6, 33});
    for (auto dtype : {mfq_tensor_backend::kFloat32, mfq_tensor_backend::kFloat16,
                       mfq_tensor_backend::kBFloat16}) {
        QuantLinear linear;
        linear.kind = QuantLinearKind::Dense;
        linear.dense_small_m_rowwise = true;
        linear.dense = mfq_tensor_backend::tensor(identity, float_options)
            .reshape({33, 33}).to(dtype);
        const double tolerance = dtype == mfq_tensor_backend::kBFloat16 ? .008
            : dtype == mfq_tensor_backend::kFloat16 ? .001 : 2.e-6;
        for (int m = 1; m <= 6; ++m) for (int mode : {1, 2}) {
            auto actual = linear.forward_input_mul(test_input.narrow(1, 0, m),
                test_gate.narrow(1, 0, m), mode);
            MFQ_RUNTIME_CHECK(actual.scalar_type() == dtype && actual.size(1) == m,
                "dense gate output dtype or shape mismatch");
            auto values = actual.to(mfq_tensor_backend::kFloat32).cpu();
            for (int i = 0; i < m * 33; ++i) {
                const double activated = (mode == 1 ? 1. : gate[i]) / (1. + std::exp(-double(gate[i])));
                const double reference = double(input[i]) * activated;
                const double value = values.template data_ptr<float>()[i];
                MFQ_RUNTIME_CHECK(std::isfinite(value) &&
                    std::abs(value - reference) <= tolerance * (1. + std::abs(reference)),
                    "dense gate differs from CPU double identity oracle");
            }
        }
    }
    std::cout << "mtp_check dense_gate_cases=36 PASS\n";
    const auto options = mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA)
        .dtype(mfq_tensor_backend::kInt64);
    auto ids = [&](const std::vector<int64_t>& tokens) {
        return mfq_tensor_backend::tensor(tokens, options).reshape({1, -1});
    };
    bool numerical_mismatch = false;
    auto compare = [&](const Tensor& actual, const Tensor& reference, double tolerance,
                       const char* name) {
        MFQ_RUNTIME_CHECK(actual.sizes() == reference.sizes(), "MTP gate tensor shape mismatch");
        auto a = actual.contiguous().to(mfq_tensor_backend::kFloat32).cpu();
        auto r = reference.contiguous().to(mfq_tensor_backend::kFloat32).cpu();
        double squared = 0., norm = 0.;
        for (int64_t i = 0; i < a.numel(); ++i) {
            const double av = a.template data_ptr<float>()[i], rv = r.template data_ptr<float>()[i];
            MFQ_RUNTIME_CHECK(std::isfinite(av) && std::isfinite(rv), "nonfinite MTP gate value");
            squared += (av - rv) * (av - rv);
            norm += rv * rv;
        }
        const double rel = std::sqrt(squared / std::max(norm, 1.e-30));
        std::cout << "mtp_check " << name << " relative_l2=" << rel << '\n';
        numerical_mismatch = numerical_mismatch || rel > tolerance;
    };
    const std::vector<int64_t> prompt{100, 200, 300, 400, 500, 600, 700};
    // Explicitly test zero, partial, and full acceptance regardless of the
    // real predictor's acceptance rate.
    for (int accepted_drafts : {0, 1, 2}) {
        model.reset(1);
        (void)model.hidden_forward(ids(prompt));
        std::vector<Tensor> verify_trace;
        std::vector<std::pair<std::string, Tensor>> verify_stages;
        g_gemma_trace_layer = 3;
        g_gemma_stage_trace = &verify_stages;
        (void)model.hidden_forward(ids({37, 41, 43}), mfq_nullopt, mfq_nullopt,
            &verify_trace, mfq_nullopt, nullptr, 1);
        g_gemma_stage_trace = nullptr;
        if (accepted_drafts == 2) model.commit_speculative();
        else model.rollback_speculative(accepted_drafts);
        MFQ_RUNTIME_CHECK(
            model.cache_pos == static_cast<int64_t>(prompt.size()) +
                1 + accepted_drafts,
            "MTP resolution retained an incorrect logical cache length");
        auto actual = model.last_logits(ids({47})).clone();
        std::vector<std::pair<Tensor, Tensor>> states;
        for (auto& block : model.blocks) {
            if (auto* linear = dynamic_cast<
                    mfq::cuda::qwen35::LinearAttentionBlock*>(block.get()))
                states.emplace_back(linear->conv_state.clone(), linear->gdn_state.clone());
        }
        model.reset(1);
        (void)model.hidden_forward(ids(prompt));
        std::vector<Tensor> serial_trace;
        std::vector<std::pair<std::string, Tensor>> serial_stages;
        g_gemma_stage_trace = &serial_stages;
        (void)model.hidden_forward(ids({37}), mfq_nullopt, mfq_nullopt,
            &serial_trace);
        g_gemma_stage_trace = nullptr;
        MFQ_RUNTIME_CHECK(verify_stages.size() == serial_stages.size() && !verify_stages.empty(),
            "MTP full-attention stage trace mismatch");
        for (size_t stage = 0; stage < verify_stages.size(); ++stage) {
            MFQ_RUNTIME_CHECK(verify_stages[stage].first == serial_stages[stage].first,
                "MTP full-attention stage names differ");
            compare(verify_stages[stage].second.narrow(1, 0, 1), serial_stages[stage].second,
                .005, verify_stages[stage].first.c_str());
        }
        MFQ_RUNTIME_CHECK(verify_trace.size() == serial_trace.size(),
            "MTP diagnostic block trace size mismatch");
        for (size_t layer = 0; layer < verify_trace.size(); ++layer) {
            const auto label = "confirmed_prefix_block_" + std::to_string(layer);
            compare(verify_trace[layer].narrow(1, 0, 1), serial_trace[layer], .005,
                label.c_str());
        }
        if (accepted_drafts >= 1) (void)model.hidden_forward(ids({41}));
        if (accepted_drafts >= 2) (void)model.hidden_forward(ids({43}));
        auto reference = model.last_logits(ids({47}));
        const auto resolution_label =
            "resolution_" + std::to_string(accepted_drafts) + "_logits";
        compare(actual, reference, .005, resolution_label.c_str());
        size_t state = 0;
        for (auto& block : model.blocks) {
            if (auto* linear = dynamic_cast<
                    mfq::cuda::qwen35::LinearAttentionBlock*>(block.get())) {
                const auto conv_label = "conv_continuation_" + std::to_string(state);
                const auto gdn_label = "gdn_continuation_" + std::to_string(state);
                compare(states[state].first, linear->conv_state, .002, conv_label.c_str());
                compare(states[state].second, linear->gdn_state, .002, gdn_label.c_str());
                ++state;
            }
        }
    }
    MFQ_RUNTIME_CHECK(!numerical_mismatch, "MTP gate numerical mismatch (see per-layer diagnostics)");
    model.reset(1);
    Tensor raw;
    (void)model.hidden_forward(ids(prompt), mfq_nullopt, mfq_nullopt, nullptr, mfq_nullopt, &raw);
    for (int tokens = 2; tokens <= 6; ++tokens) {
        auto next = ids(prompt).narrow(1, 1, tokens);
        std::vector<std::pair<std::string, Tensor>> batch_stages, row_stages;
        g_gemma_trace_layer = 0;
        if (tokens == 2) g_gemma_stage_trace = &batch_stages;
        mtp.reset();
        auto batched = mtp.forward(
            target, raw.narrow(1, 0, tokens), next).clone();
        mtp.reset();
        if (tokens == 2) g_gemma_stage_trace = &row_stages;
        std::vector<Tensor> serial;
        for (int t = 0; t < tokens; ++t)
            serial.push_back(mtp.forward(
                target, raw.narrow(1, t, 1), next.narrow(1, t, 1)));
        g_gemma_stage_trace = nullptr;
        if (tokens == 2) {
            MFQ_RUNTIME_CHECK(row_stages.size() == 2 * batch_stages.size(),
                "MTP predictor diagnostic stage count mismatch");
            for (size_t stage = 0; stage < batch_stages.size(); ++stage) {
                auto reference = mfq_tensor_backend::cat({row_stages[stage].second,
                    row_stages[stage + batch_stages.size()].second}, 1);
                const auto label = "predictor_stage_" + batch_stages[stage].first;
                compare(batch_stages[stage].second, reference,
                    std::numeric_limits<double>::infinity(), label.c_str());
            }
        }
        compare(batched, mfq_tensor_backend::cat(serial, 1), .005, "predictor_batched_vs_serial");
    }
    MFQ_RUNTIME_CHECK(!numerical_mismatch, "MTP predictor gate numerical mismatch");
    uint64_t total_cycles = 0;
    for (const auto& input : std::vector<std::vector<int64_t>>{{1, 2, 3}, prompt, std::vector<int64_t>(17, 10)}) {
        MfqSamplingParams params;
        params.max_tokens = 32;
        params.temperature = 0.;
        params.top_k = 1;
        params.seed = 20260907;
        model.reset(1);
        auto current = ids(input);
        std::vector<int64_t> expected;
        for (int step = 0; step < params.max_tokens; ++step) {
            auto next = model.next_token(current);
            expected.push_back(next.template item<int64_t>());
            current = next.reshape({1, 1});
        }
        std::vector<int64_t> got;
        const int produced = run_mtp_generation<mfq::cuda::CudaBackbone::generic_qwen>(model, mtp, input, params,
            [&](int64_t token) { got.push_back(token); return true; }, {});
        total_cycles += mtp.last_cycles;
        std::cout << "mtp_check greedy_prompt_tokens=" << input.size() << " produced=" << produced
            << " exact=" << (got == expected) << " accepted=" << mtp.last_accepted
            << " rejected=" << mtp.last_rejected << '\n';
        MFQ_RUNTIME_CHECK(produced == params.max_tokens && got == expected,
            "MTP greedy generation differs from ordinary incremental decode");
        // Callback stop then a fresh request exercises state reset after an
        // early return, including stopping before a computed bonus is emitted.
        got.clear();
        const int stopped = run_mtp_generation<mfq::cuda::CudaBackbone::generic_qwen>(model, mtp, input, params,
            [&](int64_t token) { got.push_back(token); return got.size() < 3; }, {});
        MFQ_RUNTIME_CHECK(stopped == 3 && got == std::vector<int64_t>(expected.begin(), expected.begin() + 3),
            "MTP callback emitted extra or incorrect tokens");
    }
    MfqSamplingParams stochastic;
    stochastic.max_tokens = 8;
    stochastic.temperature = .8;
    stochastic.top_k = 100;
    stochastic.top_p = .95;
    stochastic.presence_penalty = .2;
    stochastic.frequency_penalty = .1;
    stochastic.repetition_penalty = 1.05;
    stochastic.seed = 20260907;
    const int produced = run_mtp_generation<mfq::cuda::CudaBackbone::generic_qwen>(model, mtp, prompt, stochastic,
        [](int64_t) { return true; }, {});
    MFQ_RUNTIME_CHECK(produced == 8 && mtp.last_cycles > 0 && total_cycles > 0,
        "MTP runtime gate did not execute speculative cycles");
    int64_t linear_layers = 0, ffn_batches = 0, projection_batches = 0;
    for (const auto& block : model.blocks) {
        if (auto* linear = dynamic_cast<const
                mfq::cuda::qwen35::LinearAttentionBlock*>(block.get())) {
            MFQ_RUNTIME_CHECK(
                linear->speculative_ffn_batches > 0 &&
                    linear->speculative_projection_batches > 0,
                "MTP gate did not exercise full-window recurrent batching");
            ++linear_layers;
            ffn_batches += linear->speculative_ffn_batches;
            projection_batches += linear->speculative_projection_batches;
        }
    }
    MFQ_RUNTIME_CHECK(
        linear_layers > 0, "MTP batching gate found no recurrent layers");
    std::cout << "mtp_check batched_linear_layers=" << linear_layers
        << " ffn_calls=" << ffn_batches
        << " projection_calls=" << projection_batches << '\n';
    std::cout << "mtp_check PASS full_chain=1 greedy_tokens=96 stochastic_smoke_tokens=8\n";
    return 0;
}

template <typename Model>
static int run_flash_next_check(Model& model) {
    namespace tb=mfq_tensor_backend;
    MFQ_RUNTIME_CHECK(
        Model::is_flash_next && model.vocab_size() >= 8 &&
            model.max_position_embeddings() >= 16,
        "Flash-Next diagnostic requires a Flash-Next text graph, vocab>=8 and context>=16");
    auto ids=tb::tensor(std::vector<int64_t>{1,2,3,4,5,6,7},
        tb::TensorOptions().device(tb::kCUDA).dtype(tb::kInt64)).reshape({1,7});
    const auto json_tensor=[](const tb::Tensor& value) {
        auto host=value.to(tb::kFloat32).contiguous().cpu();
        return nlohmann::json{{"shape",host.sizes().vec()},
            {"data",std::vector<float>(host.template data_ptr<float>(),host.template data_ptr<float>()+host.numel())}};
    };
    nlohmann::json result;
    model.reset(1);
    result["full"]=json_tensor(model.forward(ids));
    model.reset(1);
    std::vector<tb::Tensor> pieces;
    for (auto [begin,count] : std::vector<std::pair<int64_t,int64_t>>{{0,2},{2,1},{3,4}})
        pieces.push_back(model.forward(ids.narrow(1,begin,count)));
    result["chunked"]=json_tensor(tb::cat(pieces,1));
    for (bool accept : {false,true}) {
        const std::string name=accept ? "committed" : "rejected";
        model.reset(1);
        model.forward(ids.narrow(1,0,2));
        auto hidden=model.hidden_forward(ids.narrow(1,2,2),mfq_nullopt,mfq_nullopt,nullptr,mfq_nullopt,nullptr,1);
        result[name+"_verify"]=json_tensor(model.logits_from_hidden(hidden));
        if (accept) model.commit_speculative(); else model.rollback_speculative();
        MFQ_RUNTIME_CHECK(model.cache_pos==(accept?4:3),"Flash-Next transaction cache position mismatch");
        result[name]=json_tensor(model.forward(ids.narrow(1,5,1)));
        model.reset(1);
        model.forward(ids.narrow(1,0,2));
        model.forward(ids.narrow(1,2,accept?2:1));
        result[name+"_reference"]=json_tensor(model.forward(ids.narrow(1,5,1)));
    }
    model.reset(1);
    result["reset"]=json_tensor(model.forward(ids));
    if constexpr (Model::is_qwen4) {
        auto positions=tb::stack({ids.reshape({7})-1,ids.reshape({7})+1,ids.reshape({7})+3},0);
        model.reset(1);
        result["axis_full"]=json_tensor(model.logits_from_hidden(model.hidden_forward(ids,positions)));
        model.reset(1);pieces.clear();
        auto positions4=tb::cat({positions.narrow(0,0,1)+11,positions},0);
        for (auto [begin,count] : std::vector<std::pair<int64_t,int64_t>>{{0,2},{2,1},{3,4}})
            pieces.push_back(model.logits_from_hidden(model.hidden_forward(ids.narrow(1,begin,count),positions4.narrow(-1,begin,count))));
        result["axis_chunked"]=json_tensor(tb::cat(pieces,1));
        model.reset(1);
        result["batch"]=json_tensor(model.forward(ids.repeat({2,1})));
        result["batch_reset"]=json_tensor(model.forward(ids));
        model.reset(1);result["last"]=json_tensor(model.last_logits(ids));
    }
    result["architecture"] = model.graph.backbone;
    std::cout << "flash_next_check " << result.dump() << '\n';
    return 0;
}

} // namespace mfq::cuda::diagnostics
