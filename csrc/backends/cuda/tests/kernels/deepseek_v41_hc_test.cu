#include "kernels/deepseek_v4_hc.h"
#include "kernels/deepseek_v4_attention.h"
#include "kernels/deepseek_v41.h"
#include "native/context.h"
#include "native/tensor/tensor.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace mfq::cuda;

void close(float actual, float expected, float tolerance, const char* name) {
    if (!std::isfinite(actual) ||
        std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(
            std::string(name) + " mismatch: actual=" +
            std::to_string(actual) + " expected=" +
            std::to_string(expected));
    }
}

void check_single_pass_prepare(int batch, int tokens, int hidden) {
    const auto gpu = Device{DeviceType::cuda, 0};
    const int rows = batch * tokens;
    std::vector<float> input(rows * 4 * hidden), previous(rows * 4), gamma(hidden);
    for (std::size_t i = 0; i < input.size(); ++i) input[i] = float(int(i * 17 % 97) - 48) / 64;
    for (std::size_t i = 0; i < previous.size(); ++i) previous[i] = .1f + float(i % 7) * .07f;
    for (int i = 0; i < hidden; ++i) gamma[i] = .5f + float(i % 13) / 16;
    auto residual = tensor(input).reshape({batch, tokens, 4, hidden}).to(gpu).to(kFloat16).contiguous();
    auto pre = tensor(previous).reshape({batch, tokens, 4}).to(gpu).contiguous();
    auto weight = tensor(gamma).to(gpu).contiguous();
    auto prepared = deepseek_v41_hc_prepare_cuda(residual, pre, weight, 1.e-20);
    auto input_only = deepseek_v41_hc_input_cuda(residual, pre, weight, 1.e-20);
    if (!input_only.equal(prepared.at(1)))
        throw std::runtime_error("mHC encoder projection input changed without coefficient prediction");
    auto normalized = prepared.at(0).cpu();
    auto branch = prepared.at(1).to(kFloat32).cpu();
    for (int row = 0; row < rows; ++row) {
        double flat_square = 0, branch_square = 0;
        std::vector<float> mixed(hidden, 0);
        for (int source = 0; source < 4; ++source) {
            for (int col = 0; col < hidden; ++col) {
                const auto index = (row * 4 + source) * hidden + col;
                const auto value = __half2float(__float2half_rn(input[index]));
                flat_square += double(value) * value;
                mixed[col] += value * previous[row * 4 + source];
            }
        }
        for (auto value : mixed) branch_square += double(value) * value;
        const auto flat_inverse = float(1 / std::sqrt(flat_square / (4 * hidden) + 1.e-20));
        const auto branch_inverse = float(1 / std::sqrt(branch_square / hidden + 1.e-20));
        for (int source = 0; source < 4; ++source) {
            for (int col = 0; col < hidden; ++col) {
                const auto index = (row * 4 + source) * hidden + col;
                const auto value = __half2float(__float2half_rn(input[index]));
                close(normalized.data_ptr<float>()[index], value * flat_inverse, 3.e-6f,
                      "single-pass mHC projection input");
            }
        }
        for (int col = 0; col < hidden; ++col) {
            const auto expected = __half2float(__float2half_rn(mixed[col] * branch_inverse * gamma[col]));
            close(branch.data_ptr<float>()[row * hidden + col], expected, 2.e-3f,
                  "single-pass mHC previous-weight branch");
        }
    }
    std::vector<float> mix(rows * 24), scales{.25f, -.5f, .75f}, bases(24);
    for (std::size_t i = 0; i < mix.size(); ++i) mix[i] = float(int(i * 11 % 37) - 18) / 5;
    for (int i = 0; i < 24; ++i) bases[i] = float(i - 12) / 9;
    auto mixes = tensor(mix).reshape({batch, tokens, 24}).to(gpu).contiguous();
    auto scale = tensor(scales).to(gpu).contiguous();
    auto base = tensor(bases).to(gpu).contiguous();
    auto reference = dsv4_hc_pre_cuda(residual, mixes, scale, base, 20, 1.e-6);
    auto coefficients = dsv4_hc_coefficients_cuda(mixes, scale, base, 20, 1.e-6);
    if (!coefficients.at(1).equal(reference.at(1)) || !coefficients.at(2).equal(reference.at(2)))
        throw std::runtime_error("mHC coefficient-only path changed post/Sinkhorn values");
    auto function=mfq_tensor_backend::randn({24,4*hidden},pre.options())/std::sqrt(double(4*hidden));
    auto partials=deepseek_v41_hc_project_cuda(residual,pre,function);
    auto fused=deepseek_v41_hc_finalize_cuda(residual,pre,weight,partials,scale,base,20,1.e-20,1.e-6);
    auto projected=mfq_tensor_backend::matmul(prepared[0],function.transpose(0,1));
    auto projected_coefficients=dsv4_hc_coefficients_cuda(projected.contiguous(),scale,base,20,1.e-6);
    for(int index=0;index<4;++index) {
        auto expected=index==0?prepared[1]:projected_coefficients[index-1];
        const auto error=(fused[index].to(kFloat32)-expected.to(kFloat32)).abs().max().item<double>();
        if(!std::isfinite(error) || error>(index==0?.0021:.00003))
            throw std::runtime_error("two-stage mHC changed branch or coefficient values");
    }
    const auto next_pre = coefficients.at(0).cpu();
    for (int row = 0; row < rows; ++row) for (int col = 0; col < 4; ++col) {
        const auto affine = mix[row * 24 + col] * scales[0] + bases[col];
        close(next_pre.data_ptr<float>()[row * 4 + col], 1.f / (1.f + std::exp(-affine)) + 1.e-6f,
              3.e-7f, "single-pass mHC next weights");
    }
}

void check_candidate_scores(int heads) {
    namespace tb=mfq_tensor_backend;
    auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat32);
    auto q=tb::randn({2,3,heads,128},options).to(tb::kFloat16).contiguous();
    auto k=tb::randn({2,131,128},options).to(tb::kFloat16).contiguous();
    auto weights=tb::randn({2,3,heads},options).to(tb::kFloat16).contiguous();
    std::vector<int64_t> ids(2*3*79);
    for(size_t i=0;i<ids.size();++i)ids[i]=i%13==0?-1:int64_t((i*17)%131);
    auto selected=tb::tensor(ids).reshape({2,3,79}).to(tb::kCUDA).contiguous();
    auto full=dsv4_indexer_scores_cuda(q,k,weights,126,1);
    auto safe=selected.clamp(0.,130.);
    auto expected=tb::where(selected>=0.,full.gather(-1,safe),
                              -std::numeric_limits<float>::infinity());
    auto actual=dsv4_indexer_candidate_scores_cuda(q,k,weights,selected,126,1);
    if(!actual.equal(expected))throw std::runtime_error("C4A candidate scores differ from full scoring");
}

void check_v41_geometry() {
    constexpr std::int64_t batch = 1;
    constexpr std::int64_t tokens = 2;
    constexpr std::int64_t streams = 4;
    constexpr std::int64_t hidden = 5120;
    constexpr float eps = 1.e-6f;

    std::vector<float> values(batch * tokens * streams * hidden);
    for (std::int64_t index = 0;
         index < static_cast<std::int64_t>(values.size()); ++index) {
        values[static_cast<std::size_t>(index)] =
            static_cast<float>((index * 17) % 97 - 48) / 64.0f;
    }
    const auto gpu = Device{DeviceType::cuda, 0};
    auto residual = tensor(values)
        .reshape({batch, tokens, streams, hidden})
        .to(gpu).to(kFloat16).contiguous();
    auto mixes = zeros(
        {batch, tokens, 24},
        TensorOptions{}.device(gpu).dtype(kFloat32));
    auto scale = ones({3}, mixes.options());
    auto base = zeros({24}, mixes.options());

    const auto expansion = dsv4_hc_pre_cuda(
        residual, mixes, scale, base, 3, eps);
    if (expansion.size() != 3 ||
        expansion[0].sizes().vec() !=
            std::vector<std::int64_t>({batch, tokens, hidden}) ||
        expansion[1].sizes().vec() !=
            std::vector<std::int64_t>({batch, tokens, streams}) ||
        expansion[2].sizes().vec() !=
            std::vector<std::int64_t>({batch, tokens, streams, streams})) {
        throw std::runtime_error("V4.1 mHC output geometry mismatch");
    }
    default_context(0)->stream().synchronize();

    const auto rounded = residual.to(kFloat32).cpu();
    const auto reduced = expansion[0].to(kFloat32).cpu();
    const auto post = expansion[1].cpu();
    const auto combination = expansion[2].cpu();
    for (std::int64_t row = 0; row < batch * tokens; ++row) {
        for (std::int64_t feature = 0; feature < hidden; ++feature) {
            float expected = 0.0f;
            for (std::int64_t stream = 0; stream < streams; ++stream) {
                expected += (0.5f + eps) * rounded.data_ptr<float>()[
                    (row * streams + stream) * hidden + feature];
            }
            expected = __half2float(__float2half_rn(expected));
            close(
                reduced.data_ptr<float>()[row * hidden + feature],
                expected, 2.e-3f, "mHC collapse");
        }
        for (std::int64_t stream = 0; stream < streams; ++stream) {
            close(post.data_ptr<float>()[row * streams + stream],
                  1.0f, 2.e-6f, "mHC post scale");
        }
        for (std::int64_t index = 0; index < streams * streams; ++index) {
            close(combination.data_ptr<float>()[
                      row * streams * streams + index],
                  0.25f, 4.e-6f, "mHC Sinkhorn matrix");
        }
    }

    auto branch = expansion[0].contiguous();
    auto output = dsv4_hc_post_cuda(
        branch, residual, expansion[1], expansion[2]);
    default_context(0)->stream().synchronize();
    const auto output_host = output.to(kFloat32).cpu();
    for (std::int64_t row = 0; row < batch * tokens; ++row) {
        for (std::int64_t destination = 0;
             destination < streams; ++destination) {
            for (std::int64_t feature = 0; feature < hidden; ++feature) {
                float expected = reduced.data_ptr<float>()[
                    row * hidden + feature];
                for (std::int64_t source = 0; source < streams; ++source) {
                    expected += 0.25f * rounded.data_ptr<float>()[
                        (row * streams + source) * hidden + feature];
                }
                expected = __half2float(__float2half_rn(expected));
                close(output_host.data_ptr<float>()[
                          (row * streams + destination) * hidden + feature],
                      expected, 2.e-3f, "mHC expansion");
            }
        }
    }
}

float reference_e4m3(float value) {
    const float magnitude = std::min(std::abs(value), 448.0f);
    float quantized = 0.0f;
    if (magnitude < std::ldexp(1.0f, -6)) {
        quantized = std::nearbyint(magnitude * 512.0f) / 512.0f;
    } else {
        const float exponent = std::floor(std::log2(magnitude));
        const float step = std::exp2(exponent - 3.0f);
        quantized = std::min(
            std::nearbyint(magnitude / step) * step, 448.0f);
    }
    return std::copysign(quantized, value);
}

float reference_e2m1(float value, float scale) {
    const float normalized = std::clamp(value / scale, -6.0f, 6.0f);
    const float magnitude = std::abs(normalized);
    float quantized = 0.0f;
    if (magnitude <= 0.25f) quantized = 0.0f;
    else if (magnitude < 0.75f) quantized = 0.5f;
    else if (magnitude <= 1.25f) quantized = 1.0f;
    else if (magnitude < 1.75f) quantized = 1.5f;
    else if (magnitude <= 2.5f) quantized = 2.0f;
    else if (magnitude < 3.5f) quantized = 3.0f;
    else if (magnitude <= 5.0f) quantized = 4.0f;
    else quantized = 6.0f;
    return std::copysign(quantized * scale, normalized);
}

void check_released_activation_boundaries() {
    std::vector<float> values(64);
    for (int index = 0; index < 32; ++index) {
        values[static_cast<std::size_t>(index)] =
            index == 31 ? 448.0f : static_cast<float>(index - 15) / 13.0f;
        values[static_cast<std::size_t>(32 + index)] =
            index == 31 ? -28.0f : static_cast<float>(index - 15) / 17.0f;
    }
    const auto gpu = Device{DeviceType::cuda, 0};
    auto input = tensor(values).reshape({2, 32}).to(gpu).to(kFloat16).contiguous();
    const auto rounded = input.to(kFloat32).cpu();
    const auto got8 = deepseek_v41_mxfp8_e4m3_sim_cuda(input)
        .to(kFloat32).cpu();
    for (int group = 0; group < 2; ++group) {
        float maximum = 0.0f;
        for (int lane = 0; lane < 32; ++lane) {
            maximum = std::max(maximum, std::abs(
                rounded.data_ptr<float>()[group * 32 + lane]));
        }
        const float scale = std::exp2(std::ceil(std::log2(
            std::max(maximum, 448.0f * std::ldexp(1.0f, -126)) / 448.0f)));
        for (int lane = 0; lane < 32; ++lane) {
            const float source = rounded.data_ptr<float>()[group * 32 + lane];
            const float expected = __half2float(__float2half_rn(
                reference_e4m3(source / scale) * scale));
            close(got8.data_ptr<float>()[group * 32 + lane],
                  expected, 0.0f, "MXFP8 activation boundary");
        }
    }

    std::vector<float> fp4_values(32);
    for (int lane = 0; lane < 16; ++lane) {
        fp4_values[static_cast<std::size_t>(lane)] =
            lane == 15 ? 6.0f : static_cast<float>(lane - 7) / 2.0f;
        fp4_values[static_cast<std::size_t>(16 + lane)] =
            lane == 15 ? -3.0f : static_cast<float>(lane - 7) / 4.0f;
    }
    input = tensor(fp4_values).reshape({2, 16})
        .to(gpu).to(kFloat16).contiguous();
    const auto fp4_rounded = input.to(kFloat32).cpu();
    const auto got4 = deepseek_v41_mxfp4_e4m3_scale_sim_cuda(input)
        .to(kFloat32).cpu();
    for (int group = 0; group < 2; ++group) {
        float maximum = 0.0f;
        for (int lane = 0; lane < 16; ++lane) {
            maximum = std::max(maximum, std::abs(
                fp4_rounded.data_ptr<float>()[group * 16 + lane]));
        }
        const float raw_scale = std::max(
            maximum, 6.0f * std::ldexp(1.0f, -9)) / 6.0f;
        const float scale = std::max(
            reference_e4m3(raw_scale), std::ldexp(1.0f, -9));
        for (int lane = 0; lane < 16; ++lane) {
            const float source = fp4_rounded.data_ptr<float>()[
                group * 16 + lane];
            const float expected = __half2float(__float2half_rn(
                reference_e2m1(source, scale)));
            close(got4.data_ptr<float>()[group * 16 + lane],
                  expected, 0.0f, "MXFP4 activation boundary");
        }
    }
}

void check_v41_indexer_geometry() {
    constexpr std::int64_t batch = 1;
    constexpr std::int64_t tokens = 2;
    constexpr std::int64_t heads = 32;
    constexpr std::int64_t width = 128;
    constexpr std::int64_t keys = 65;
    const auto gpu = Device{DeviceType::cuda, 0};
    const auto half_gpu = TensorOptions{}.device(gpu).dtype(kFloat16);
    auto query = full(
        {batch, tokens, heads, width}, 0.5, half_gpu);
    std::vector<float> key_values(keys * width);
    for (std::int64_t key = 0; key < keys; ++key) {
        std::fill_n(
            key_values.begin() + key * width,
            width,
            static_cast<float>(key + 1) / 128.0f);
    }
    auto key = tensor(key_values).reshape({batch, keys, width})
        .to(gpu).to(kFloat16).contiguous();
    auto weights = full(
        {batch, tokens, heads}, 1.0 / static_cast<double>(heads),
        half_gpu);
    auto scores = dsv4_indexer_scores_cuda(
        query, key, weights, 4, 2);
    default_context(0)->stream().synchronize();
    const auto host = scores.to(kFloat32).cpu();
    for (std::int64_t token = 0; token < tokens; ++token) {
        const std::int64_t visible = (4 + token + 1) / 2;
        for (std::int64_t key_index = 0; key_index < keys; ++key_index) {
            const float actual = host.data_ptr<float>()[
                token * keys + key_index];
            if (key_index >= visible) {
                if (!std::isinf(actual) || actual >= 0.0f) {
                    throw std::runtime_error(
                        "V4.1 indexer causal mask mismatch");
                }
                continue;
            }
            const float expected = __half2float(__float2half_rn(
                0.5f * static_cast<float>(key_index + 1) / 64.0f));
            close(actual, expected, 2.e-4f, "V4.1 32-head index score");
        }
    }

    auto selected = dsv4_topk512_cuda(scores);
    if (selected.sizes().vec() !=
            std::vector<std::int64_t>({batch, tokens, keys})) {
        throw std::runtime_error(
            "small V4.1 index pools retained padded duplicate routes");
    }
    const auto ids = selected.to(kInt32).cpu();
    for (std::int64_t row = 0; row < tokens; ++row) {
        for (std::int64_t index = 0; index < keys; ++index) {
            if (ids.data_ptr<std::int32_t>()[row * keys + index] != index) {
                throw std::runtime_error(
                    "small V4.1 index pool order mismatch");
            }
        }
    }
}

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    try {
        check_single_pass_prepare(1, 1, 5120);
        check_single_pass_prepare(2, 3, 33);
        check_single_pass_prepare(1, 17, 5120);
        check_candidate_scores(32);
        check_candidate_scores(64);
        check_v41_geometry();
        check_released_activation_boundaries();
        check_v41_indexer_geometry();
        std::cout << "DeepSeek-V4.1 CUDA primitive tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "DeepSeek-V4.1 CUDA mHC test failed: "
                  << error.what() << '\n';
        return 1;
    }
}
