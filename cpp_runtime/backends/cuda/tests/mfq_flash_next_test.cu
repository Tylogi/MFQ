// Native ABI numerical-test bridge: tests/test_cuda_flash_next.py supplies the
// same NumPy oracle cases to this executable and the production Torch module.
#include "mfq_cuda_attention_ops.h"
#include "mfq_cuda_moe_ops.h"
#include "mfq/kernels/cuda/glm5_next.h"
#include "mfq/kernels/cuda/qwen4_exp.h"
#include "glm5_next/model.h"
#include "qwen4_exp/model.h"
#include "cuda_execution.h"
#include "mfq_cuda_context.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>

mfq_tensor_backend::Tensor attention_glm_mla_sparse_cuda(
    mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor kv,
    mfq_tensor_backend::Tensor indices, mfq_tensor_backend::Tensor meta,
    double scale);
mfq_tensor_backend::Tensor attention_dsv4_sparse_cuda(
    mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor kv,
    mfq_tensor_backend::Tensor indices, mfq_tensor_backend::Tensor mask,
    mfq_tensor_backend::Tensor sinks, mfq_tensor_backend::Tensor meta,
    double scale);

namespace {
using namespace mfq::cuda;
using Json = nlohmann::json;

Tensor unfused_grouped_rms_norm(const Tensor& value, const Tensor& weight,
    int64_t group, double eps) {
    auto shape = value.sizes().vec();
    shape.back() /= group;
    shape.push_back(group);
    auto source = value.to(kFloat32).reshape(shape);
    auto normalized = source * rsqrt((source * source).mean(-1, true) + eps);
    return (normalized.reshape(value.sizes()) * (1.0 + weight.to(kFloat32)))
        .to(value.scalar_type());
}

void check_grouped_rms_norm() {
    const Device gpu{DeviceType::cuda, 0};
    // Rounding eps to float first changes this small activation by one ULP.
    auto tiny = tensor<float>({1.0980109436786734e-05f}).to(gpu);
    auto zero = zeros({1}, tiny.options());
    auto expected_tiny = unfused_grouped_rms_norm(tiny, zero, 1, 1e-6);
    MFQ_RUNTIME_CHECK(expected_tiny.equal(mfq_qwen4_exp::grouped_rms_norm(tiny, zero, 1, 1e-6)) &&
        !expected_tiny.equal(unfused_grouped_rms_norm(tiny, zero, 1, double(float(1e-6)))),
        "Qwen grouped RMSNorm lost FP64 epsilon rounding");
    for (auto dtype : {kFloat32, kFloat16, kBFloat16})
    for (int group : {1, 7, 32, 33, 64, 65, 128, 129, 640, 2560})
    for (int tokens : {1, 23}) for (float magnitude : {1.f, 0.001f, 0.00001f}) {
        const int width = group * 4;
        std::vector<float> values(tokens * width), scales(width);
        for (size_t i = 0; i < values.size(); ++i)
            values[i] = magnitude * (int((i * 17) % 257) - 128) / 11.0f;
        for (size_t i = 0; i < scales.size(); ++i)
            scales[i] = (int(i % 19) - 9) / 17.0f;
        auto input = tensor(values).to(gpu, dtype).reshape({1, tokens, width});
        auto weight = tensor(scales).to(gpu, dtype);
        for (bool strided : {false, true}) {
            auto x = strided ? input.transpose(-1, -2).contiguous().transpose(-1, -2) : input;
            auto expected = unfused_grouped_rms_norm(x, weight, group, 1e-6).cpu();
            auto actual = mfq_qwen4_exp::grouped_rms_norm(x, weight, group, 1e-6).cpu();
            MFQ_RUNTIME_CHECK(std::memcmp(actual.data_ptr(), expected.data_ptr(), actual.nbytes()) == 0,
                "Qwen grouped RMSNorm output bits changed: group=", group, ", tokens=", tokens);
        }
    }
}

void check_gated_residual_post() {
    const Device gpu{DeviceType::cuda, 0};
    for (auto dtype : {kFloat32, kFloat16, kBFloat16})
    for (int hidden : {7, 2560}) for (int tokens : {1, 23}) {
        constexpr int streams = 4;
        std::vector<float> values(tokens * hidden), residuals(tokens * hidden * streams), gates(tokens * streams);
        for (size_t i = 0; i < values.size(); ++i)
            values[i] = (int(i * 17 % 257) - 128) / 11.0f;
        for (size_t i = 0; i < residuals.size(); ++i)
            residuals[i] = (int(i * 13 % 193) - 96) / 7.0f;
        for (size_t i = 0; i < gates.size(); ++i)
            gates[i] = (int(i % 19) - 9) / 17.0f;
        // The separately rounded product differs from FMA at this element.
        values[0] = 0x1.000002p+0f; gates[0] = 0x1.fffffep-1f; residuals[0] = -1.0f;
        auto branch = tensor(values).to(gpu, dtype).reshape({1, tokens, hidden});
        auto residual = tensor(residuals).to(gpu).reshape({1, tokens, hidden * streams});
        auto injection = tensor(gates).to(gpu).reshape({1, tokens, streams});
        for (bool strided : {false, true}) {
            auto layout = [strided](const Tensor& value) {
                return strided ? value.transpose(-1, -2).contiguous().transpose(-1, -2) : value;
            };
            auto b = layout(branch), r = layout(residual), g = layout(injection);
            auto expected = (r + (b.unsqueeze(-2) * g.unsqueeze(-1)).reshape(r.sizes())).cpu();
            auto actual = mfq_qwen4_exp::gated_residual_post(b, r, g, streams).cpu();
            MFQ_RUNTIME_CHECK(actual.scalar_type() == expected.scalar_type() &&
                std::memcmp(actual.data_ptr(), expected.data_ptr(), actual.nbytes()) == 0,
                "Qwen gated residual post output bits changed");
        }
    }
}

Tensor unfused_moe_reduce(const Tensor& pairs, const Tensor& weights) {
    auto result = zeros({pairs.size(0), pairs.size(2)}, weights.options());
    for (int64_t route = 0; route < pairs.size(1); ++route)
        result = result + pairs.select(1, route).to(kFloat32) *
            weights.select(1, route).unsqueeze(-1);
    return result.to(pairs.scalar_type());
}

void check_moe_reduce() {
    const Device gpu{DeviceType::cuda, 0};
    for (int tokens : {1, 4, 33}) for (int width : {7, 2560}) {
        constexpr int routes = 10;
        std::vector<float> values(tokens * routes * width), scales(tokens * routes);
        for (size_t i = 0; i < values.size(); ++i)
            values[i] = (static_cast<int>((i * 17) % 257) - 128) / 11.0f;
        for (size_t i = 0; i < scales.size(); ++i)
            scales[i] = static_cast<float>(i % 9 + 1) / 45.0f;
        auto pairs = tensor(values).to(gpu, kFloat16).reshape({tokens, routes, width});
        auto weights = tensor(scales).to(gpu).reshape({tokens, routes});
        auto expected = unfused_moe_reduce(pairs, weights).cpu();
        auto actual = moe_weighted_reduce_cuda(pairs, weights, true).cpu();
        MFQ_RUNTIME_CHECK(std::memcmp(actual.data_ptr(), expected.data_ptr(), actual.nbytes()) == 0,
            "Qwen MoE reduction changed FP16 output bits");
    }
    // Separate products cancel exactly; an FMA retains one FP16 subnormal.
    auto pairs = tensor<float>({-1025.f, 1025.f}).to(gpu, kFloat16).reshape({1, 2, 1});
    auto weights = tensor<float>({0.500000059604644775390625f, 0.500000059604644775390625f})
        .to(gpu).reshape({1, 2});
    MFQ_RUNTIME_CHECK(moe_weighted_reduce_cuda(pairs, weights, true).to(kFloat32).item<float>() == 0.f &&
        moe_weighted_reduce_cuda(pairs, weights).to(kFloat32).item<float>() != 0.f,
        "Qwen MoE product rounding regressed or legacy FMA behavior changed");
}

void check_qwen_moe_topk() {
    const Device gpu{DeviceType::cuda, 0};
    std::vector<float> values(512);
    for (int index = 0; index < 512; ++index)
        values[index] = static_cast<float>((index * 97) % 509 - 254) / 37.0f;
    auto logits = tensor(values).reshape({1, 512}).to(gpu);
    auto actual = moe_topk_cuda(
        logits, 10, false, false, true, false, mfq_nullopt, 1e-20, 1.0);
    auto expected = topk(softmax(logits, -1), 10, -1, true, true);
    auto expected_weights = std::get<0>(expected);
    expected_weights = expected_weights / expected_weights.sum(-1, true);
    MFQ_RUNTIME_CHECK(
        actual[0].equal(std::get<1>(expected).to(kInt32)),
        "Qwen 512x10 MoE TopK indices changed");
    const float maximum_error =
        (actual[1] - expected_weights).abs().max().item<float>();
    MFQ_RUNTIME_CHECK(
        maximum_error <= 2e-6f,
        "Qwen 512x10 MoE TopK weights differ: ", maximum_error);
}

void require_constant(
    const Tensor& value, float expected, float tolerance, const char * name) {
    const auto host = value.to(kFloat32).contiguous().cpu();
    float maximum_error = 0.0f;
    for (int64_t index = 0; index < host.numel(); ++index) {
        maximum_error = std::max(
            maximum_error,
            std::abs(host.data_ptr<float>()[index] - expected));
    }
    if (maximum_error > tolerance) {
        throw std::runtime_error(
            std::string(name) + " mismatch, max_abs=" +
            std::to_string(maximum_error));
    }
}

void require_qsa_gqa_means(
    const Tensor& value, float first, float second, float tolerance) {
    const auto host = value.to(kFloat32).contiguous().cpu();
    float maximum_error = 0.0f;
    for (int head = 0; head < 24; ++head) {
        const float expected = head < 12 ? first : second;
        for (int width = 0; width < 256; ++width) {
            maximum_error = std::max(
                maximum_error,
                std::abs(host.data_ptr<float>()[head * 256 + width] - expected));
        }
    }
    if (maximum_error > tolerance) {
        throw std::runtime_error(
            "QSA sparse GQA mismatch, max_abs=" +
            std::to_string(maximum_error));
    }
}

void check_shared_sparse_attention() {
    const auto float_cuda = TensorOptions{}.dtype(kFloat32).device(kCUDA);
    const auto half_cuda = TensorOptions{}.dtype(kFloat16).device(kCUDA);
    const auto int_cuda = TensorOptions{}.dtype(kInt32).device(kCUDA);
    constexpr int selected_count = 2048;
    std::vector<int32_t> selected(selected_count);
    float glm_expected = 0.0f;
    for (int index = 0; index < selected_count; ++index) {
        selected[index] = (index * 37 + 11) % 256;
        glm_expected += static_cast<float>(selected[index]);
    }
    glm_expected /= selected.size();
    const auto indices = tensor(selected)
        .reshape({1, 1, selected_count}).to(int_cuda);
    auto meta = empty({4 << 20}, float_cuda);

    const auto qsa_q = zeros({1, 24, 1, 256}, float_cuda);
    const auto qsa_k = zeros({1, 2, 256, 256}, half_cuda);
    const auto qsa_v = arange(512, float_cuda)
        .reshape({1, 2, 256, 1}).expand({1, 2, 256, 256})
        .contiguous().to(kFloat16);
    require_qsa_gqa_means(
        mfq_qwen4_exp::sparse_gqa_attention(
            qsa_q, qsa_k, qsa_v, indices),
        glm_expected, glm_expected + 256.0f, 6.0e-2f);

    const auto glm_q = zeros({1, 64, 1, 576}, float_cuda);
    const auto glm_kv = arange(256, float_cuda)
        .reshape({1, 256, 1}).expand({1, 256, 576}).contiguous().to(kFloat16);
    require_constant(
        attention_glm_mla_sparse_cuda(glm_q, glm_kv, indices, meta, 1.0),
        glm_expected, 2.0e-2f, "GLM sparse MLA");

    const auto dsv_q = zeros({1, 64, 1, 512}, float_cuda);
    const auto dsv_kv = arange(256, float_cuda)
        .reshape({1, 256, 1}).expand({1, 256, 512}).contiguous().to(kFloat16);
    std::vector<float> mask_values(selected_count, 0.0f);
    float dsv_sum = 0.0f;
    int dsv_count = 0;
    for (int index = 0; index < selected_count; ++index) {
        if (index % 5 == 0) {
            mask_values[index] = -std::numeric_limits<float>::infinity();
        } else {
            dsv_sum += static_cast<float>(selected[index]);
            ++dsv_count;
        }
    }
    const auto mask = tensor(mask_values)
        .reshape({1, 1, selected_count}).to(half_cuda);
    const auto sinks = zeros({64}, float_cuda);
    require_constant(
        attention_dsv4_sparse_cuda(
            dsv_q, dsv_kv, indices, mask, sinks, meta, 1.0),
        dsv_sum / static_cast<float>(dsv_count + 1),
        6.0e-2f, "DSV4 sparse attention");

    const auto all_invalid_mask = full(
        {1, 1, selected_count},
        -std::numeric_limits<float>::infinity(), half_cuda);
    const auto all_invalid_sinks = full(
        {64}, -std::numeric_limits<float>::infinity(), float_cuda);
    require_constant(
        attention_dsv4_sparse_cuda(
            dsv_q, dsv_kv, indices, all_invalid_mask,
            all_invalid_sinks, meta, 1.0),
        0.0f, 0.0f, "fully masked DSV4 sparse attention");
}

Tensor input(const Json& j) {
    if (j.is_null()) return {};
    const auto shape = j.at("shape").get<std::vector<int64_t>>();
    const auto dtype = j.value("dtype", "float32");
    auto host = empty(shape, TensorOptions{}.dtype(dtype == "int64" ? kInt64 : kFloat32));
    if (dtype == "int64") {
        const auto data = j.at("data").get<std::vector<int64_t>>();
        if (data.size() != static_cast<size_t>(host.numel())) throw std::runtime_error("fixture size mismatch");
        std::copy(data.begin(), data.end(), host.data_ptr<int64_t>());
    } else {
        const auto data = j.at("data").get<std::vector<float>>();
        if (data.size() != static_cast<size_t>(host.numel())) throw std::runtime_error("fixture size mismatch");
        std::copy(data.begin(), data.end(), host.data_ptr<float>());
    }
    auto value = host.to(Device{DeviceType::cuda, 0});
    if (dtype == "float16") value = value.to(kFloat16);
    else if (dtype == "bfloat16") value = value.to(kBFloat16);
    else if (dtype != "float32" && dtype != "int64") throw std::runtime_error("unknown fixture dtype");
    if (j.value("noncontiguous", false) && value.dim() >= 2)
        value = value.transpose(-1, -2).contiguous().transpose(-1, -2);
    return value;
}

std::vector<Tensor> run(const std::string& op, const std::vector<Tensor>& a, const Json& p) {
    if (op == "moe_reduce") return {moe_weighted_reduce_cuda(a.at(0), a.at(1), true)};
    if (op == "moe_reduce_reference") return {unfused_moe_reduce(a.at(0), a.at(1))};
    CudaExecutionContext execution;
    const auto optional = [&](size_t i) -> std::optional<Tensor> {
        return a.at(i).defined() ? std::optional<Tensor>(a.at(i)) : std::nullopt;
    };
    std::optional<double> scale;
    if (p.contains("scale") && !p.at("scale").is_null()) scale = p.at("scale").get<double>();
    if (op == "runtime_gdn") {
        const auto linear = [&](int index) -> mfq::cuda::qwen4_exp::Linear {
            auto w=a.at(index);return [w](CudaExecutionContext&, const Tensor& x) {return matmul(x.to(w.scalar_type()),w.transpose(-1,-2));};
        };
        mfq::cuda::qwen4_exp::GdnWeights w{linear(1),linear(2),linear(3),linear(4),linear(5),a.at(6),a.at(7),a.at(8),a.at(9)};
        mfq::cuda::qwen4_exp::Gdn block(std::move(w),p.at("key_heads"),p.at("value_heads"),p.at("width"),
            p.at("kernel"),p.value("eps",1e-6),p.value("silu_gate",false));
        std::vector<Tensor> out;
        for (const auto& step:p.at("steps")) {
            if (step.value("reset",false)) block.reset();
            else if (step.value("commit",false)) block.commit();
            else if (step.value("rollback",false)) block.rollback();
            else {
                out.push_back(block.forward(execution,a.at(0).narrow(1,step.at("begin"),step.at("count")),step.value("cache",true),step.value("confirmed",0)));
                out.push_back(block.conv_state());out.push_back(block.recurrent_state());
            }
        }
        return out;
    }
    if (op == "runtime_ngram" || op == "runtime_ple") {
        const auto weights=a.at(op=="runtime_ngram"?1:2);
        std::vector<mfq::cuda::qwen4_exp::Embedding> shards;
        for (int64_t i=0;i<weights.size(0);++i) {
            auto w=weights.select(0,i);
            shards.push_back([w](const Tensor& ids) {
                auto shape=ids.sizes().vec();shape.push_back(w.size(1));
                return w.index_select(0,ids.reshape({-1}).to(kInt64)).reshape(shape);
            });
        }
        mfq::cuda::qwen4_exp::NgramEmbedding embedding(std::move(shards),weights.size(1),weights.size(2),p.at("ngram"),
            p.at("heads_per_ngram"),p.at("eos"),p.at("multipliers").get<std::vector<int64_t>>(),
            p.at("offsets").get<std::vector<int64_t>>(),p.at("vocab").get<std::vector<int64_t>>());
        std::vector<Tensor> out;
        if (op=="runtime_ngram") {
            for (const auto& step:p.at("steps")) {
                if (step.value("reset",false)) embedding.reset();
                else {
                    Tensor ids;
                    out.push_back(embedding.forward(a.at(0).narrow(1,step.at("begin"),step.at("count")),step.value("cache",true),&ids));
                    out.push_back(ids);
                }
            }
        } else {
            const auto linear=[&](int i)->mfq::cuda::qwen4_exp::Linear {
                auto w=a.at(i);return [w](CudaExecutionContext&, const Tensor& x) {return matmul(x.to(w.scalar_type()),w.transpose(-1,-2));};
            };
            mfq::cuda::qwen4_exp::PleWeights w{linear(3),linear(4),a.at(5),a.at(6),a.at(7),a.at(8)};
            mfq::cuda::qwen4_exp::Ple block(std::move(embedding),std::move(w),p.at("hidden"),p.at("streams"),p.at("ngram"),p.value("eps",1e-6));
            for (const auto& step:p.at("steps")) {
                if (step.value("reset",false)) block.reset();
                else if (step.value("commit",false)) block.commit();
                else if (step.value("rollback",false)) block.rollback();
                else {
                    const int64_t begin=step.at("begin"),count=step.at("count");
                    out.push_back(block.forward(execution,a.at(0).narrow(1,begin,count),a.at(1).narrow(1,begin,count),step.value("cache",true),step.value("confirmed",0)));
                    out.push_back(block.conv_state());
                }
            }
        }
        return out;
    }
    if (op == "runtime_rotary") {
        mfq::cuda::RotaryEmbedding rotary(p.at("rotary"),p.at("maximum"),p.value("base",1e7),
            p.value("sections",std::vector<int64_t>{}),p.value("interleaved",false));
        return {rotary.forward(a.at(0),a.at(1))};
    }
    if (op == "runtime_qsa") {
        const auto linear = [&](int index) -> mfq::cuda::qwen4_exp::Linear {
            auto weight=a.at(index);
            return [weight](CudaExecutionContext&, const Tensor& x) {return matmul(x.to(weight.scalar_type()),weight.transpose(-1,-2));};
        };
        auto rotary=std::make_shared<mfq::cuda::RotaryEmbedding>(p.at("rotary"),p.at("maximum"),p.value("base",1e7),
            p.value("sections",std::vector<int64_t>{}),p.value("interleaved",false));
        mfq::cuda::qwen4_exp::QsaWeights weights{linear(1),linear(2),linear(3),linear(4),linear(5),a.at(6),a.at(7),a.at(8),a.at(9)};
        mfq::cuda::qwen4_exp::QsaConfig config{p.at("heads"),p.at("kv_heads"),p.at("width"),p.at("index_heads"),
            p.at("index_width"),p.at("pool"),p.at("budget"),p.at("maximum"),p.value("eps",1e-6)};
        mfq::cuda::qwen4_exp::Qsa block(std::move(weights),config,rotary);
        Tensor history;
        std::vector<Tensor> out;
        for (const auto& step:p.at("steps")) {
            if (step.value("reset",false)) {block.reset();history={};}
            else if (step.contains("truncate")) {
                block.truncate(step.at("truncate"));
                history=history.narrow(-1,0,step.at("truncate"));
            } else {
                auto pos=a.at(10).narrow(-1,step.at("begin"),step.at("count"));
                const bool cache=step.value("cache",true);
                auto full=cache && history.defined()?cat({history,pos},-1):pos;
                std::vector<Tensor> trace;
                out.push_back(block.forward(execution,a.at(0).narrow(1,step.at("begin"),step.at("count")),pos,full,cache,
                    p.value("trace",false)?&trace:nullptr));
                out.insert(out.end(),trace.begin(),trace.end());
                if (cache) history=full;
            }
        }
        return out;
    }
    if (op == "runtime_select_pooled_blocks") return {mfq::cuda::attention_ops::select_pooled_blocks(
        a.at(0), p.at("query_offset"), p.at("logical_length"), p.at("pool"), p.at("budget"), p.at("tail"))};
    if (op == "runtime_sequence_cache") {
        mfq::cuda::attention_ops::SequenceCache cache(p.at("maximum"), a.at(0).size(-1));
        std::vector<Tensor> out;
        for (const auto& step : p.at("steps")) {
            if (step.contains("truncate")) cache.truncate(step.at("truncate"));
            else if (step.value("reset", false)) cache.reset();
            else out.push_back(cache.append(a.at(0).narrow(1, step.at("begin"), step.at("count"))).clone());
        }
        return out;
    }
    if (op == "runtime_kda") {
        const auto linear = [&](int index) -> mfq::cuda::glm5_next::Linear {
            auto weight = a.at(index);
            return [weight](CudaExecutionContext&, const Tensor& x) { return matmul(x.to(weight.scalar_type()), weight.transpose(-1,-2)); };
        };
        mfq::cuda::glm5_next::KdaWeights weights{linear(1),linear(2),linear(3),linear(4),linear(5),linear(6),linear(7),
            a.at(8),a.at(9),a.at(10),a.at(11),a.at(12),a.at(13)};
        mfq::cuda::glm5_next::Kda block(std::move(weights), p.at("heads"), p.at("width"), p.at("kernel"),
            p.value("lower_bound", -5.0), p.value("eps", 1e-5));
        std::vector<Tensor> out;
        for (const auto& step : p.at("steps")) {
            if (step.value("rollback", false)) block.rollback();
            else if (step.value("commit", false)) block.commit();
            else if (step.value("reset", false)) block.reset();
            else {
                out.push_back(block.forward(execution,a.at(0).narrow(1, step.at("begin"), step.at("count")),
                    step.value("cache", true), step.value("confirmed", 0)));
                out.push_back(block.conv_state());
                out.push_back(block.recurrent_state());
            }
        }
        return out;
    }
    if (op == "qwen4_grouped_rms_norm") return {mfq_qwen4_exp::grouped_rms_norm(a.at(0), a.at(1), p.at("group_size"), p.value("eps", 1e-6))};
    if (op == "qwen4_gated_residual_pre") return mfq_qwen4_exp::gated_residual_pre(a.at(0), a.at(1), a.at(2), a.at(3), optional(4), p.at("hidden_size"), p.value("hc_count", 4), p.value("eps", 1e-6));
    if (op == "qwen4_gated_residual_post") return {mfq_qwen4_exp::gated_residual_post(a.at(0), a.at(1), a.at(2), p.value("hc_count", 4))};
    if (op == "glm5_mhc_pre") return mfq_glm5_next::mhc_pre(a.at(0), a.at(1), a.at(2), a.at(3), p.value("sinkhorn_iterations", 20), p.value("hc_eps", 1e-6), p.value("rms_eps", 1e-5));
    if (op == "glm5_mhc_post") return {mfq_glm5_next::mhc_post(a.at(0), a.at(1), a.at(2), a.at(3))};
    if (op == "glm5_kda_forget_gate") return {mfq_glm5_next::kda_forget_gate(a.at(0), a.at(1), a.at(2), a.at(3), a.at(4), p.at("num_heads"), p.at("head_dim"), p.value("lower_bound", -5.0))};
    if (op == "qsa_block_scores") return {mfq_qwen4_exp::block_scores(a.at(0), a.at(1))};
    if (op == "glm5_kpool_scores") return {mfq_glm5_next::kpool_scores(a.at(0), a.at(1), a.at(2))};
    if (op == "glm5_kpool_states") return {mfq_glm5_next::kpool_states(a.at(0), a.at(1), a.at(2), p.value("pool_size", 4))};
    if (op == "qwen4_ple_dilated_conv_silu") return mfq_qwen4_exp::ple_dilated_conv_silu(a.at(0), a.at(1), optional(2), p.at("dilation"));
    if (op == "qwen4_dense_gqa_attention") return {mfq_qwen4_exp::dense_gqa_attention(a.at(0), a.at(1), a.at(2), p.at("query_offset"))};
    if (op == "qwen4_sparse_gqa_attention") return {mfq_qwen4_exp::sparse_gqa_attention(a.at(0), a.at(1), a.at(2), a.at(3))};
    if (op == "glm5_dense_mla_attention") return {mfq_glm5_next::dense_mla_attention(a.at(0), a.at(1), p.at("query_offset"), scale)};
    if (op == "glm5_sparse_mla_attention") return {mfq_glm5_next::sparse_mla_attention(a.at(0), a.at(1), a.at(2), scale)};
    throw std::runtime_error("unknown Flash-Next operation");
}

Json output(const std::vector<Tensor>& tensors) {
    Json result = Json::array();
    for (const auto& value : tensors) {
        if (!value.defined()) { result.push_back(nullptr); continue; }
        auto host = value.to(kFloat32).contiguous().cpu();
        std::vector<float> data(host.numel());
        if (host.numel()) std::copy_n(host.data_ptr<float>(), host.numel(), data.data());
        result.push_back({{"shape", value.sizes().vec()}, {"data", data},
            {"dtype", value.scalar_type() == kFloat16 ? "float16" :
                value.scalar_type() == kBFloat16 ? "bfloat16" : "float32"}});
    }
    return result;
}
} // namespace

int main(int argc, char** argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    try {
        const auto context = mfq::cuda::default_context(0);
        auto stream = mfq::cuda::stream_from_pool(false, 0);
        mfq::cuda::StreamGuard stream_guard(stream);
        if (argc == 1) {
            auto q = mfq::cuda::ones({1,1,2,4}, mfq::cuda::TensorOptions{}.device(mfq::cuda::kCUDA));
            auto k = mfq::cuda::ones({1,3,4}, q.options());
            auto got = mfq_qwen4_exp::block_scores(q,k).cpu();
            for (int i = 0; i < 3; ++i)
                if (got.data_ptr<float>()[i] != 4.f) throw std::runtime_error("QSA smoke mismatch");
            check_shared_sparse_attention();
            check_moe_reduce();
            check_qwen_moe_topk();
            check_grouped_rms_norm();
            check_gated_residual_post();
            std::cout << "Flash-Next and shared sparse-attention native smoke passed\n";
            return 0;
        }
        if (std::string(argv[1]) != "--json") throw std::runtime_error("expected --json");
        std::string line;
        while (std::getline(std::cin, line)) {
            try {
                const auto request = Json::parse(line);
                std::vector<Tensor> tensors;
                for (const auto& item : request.at("inputs")) tensors.push_back(input(item));
                const auto op = request.at("op").get<std::string>();
                const auto params = request.value("params", Json::object());
                const auto execute = [&] { return run(op, tensors, params); };
                std::vector<Tensor> result;
                mfq::cuda::Graph graph;
                if (request.value("graph", false)) {
                    result = execute();
                    MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(mfq::cuda::current_stream(0).stream()));
                    graph.prepare_memory();
                    result = execute();
                    MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(mfq::cuda::current_stream(0).stream()));
                    result.clear();
                    graph.capture_begin();
                    result = execute();
                    graph.capture_end();
                    graph.replay();
                    graph.replay();
                    if (request.contains("replay_inputs")) {
                        const auto& updates = request.at("replay_inputs");
                        if (updates.size() != tensors.size()) throw std::runtime_error("replay fixture count mismatch");
                        for (size_t i = 0; i < tensors.size(); ++i) {
                            if (tensors[i].defined()) tensors[i].copy_(input(updates.at(i)));
                        }
                        graph.replay();
                    }
                } else result = execute();
                std::cout << Json({{"outputs", output(result)}}).dump() << std::endl;
            } catch (const std::exception& e) {
                std::cout << Json({{"error", e.what()}}).dump() << std::endl;
            }
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Flash-Next native test failed: " << e.what() << '\n';
        return 1;
    }
}
