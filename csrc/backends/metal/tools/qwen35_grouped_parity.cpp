// Parity check: grouped multi-row projections vs independent per-tensor
// GEMVs on tensors loaded from a real MFQ container. Exercises the exact
// QKV groups the Qwen3.5 runtime builds for prefill/verify rows.
#include "mfq_container.h"
#include "mlx_grouped_linear.h"
#include "mlx_legacy_tensor_compat.h"
#include "mlx_tensor.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <mlx/mlx.h>

namespace {

using mlx::core::Shape;
using mlx::core::array;
using mfq::metal::MfqContainer;
using mfq::metal::MlxGroupedLinear;
using mfq::metal::MlxGroupedLinearWeightRef;
using mfq::metal::MlxLinear;

array make_input(int width, int rows, float scale) {
    std::vector<float> values(
        static_cast<std::size_t>(rows) * static_cast<std::size_t>(width));
    // Deterministic unit-RMS pattern: post-RMSNorm activations have RMS 1.
    double sum_squares = 0.0;
    for (std::size_t index = 0; index < values.size(); ++index) {
        const double phase =
            0.7 * static_cast<double>(((index * 2654435761u) >> 16) % 65536)
            / 32768.0 - 1.0;
        sum_squares += phase * phase;
        values[index] = static_cast<float>(phase);
    }
    const double rms = std::sqrt(sum_squares / static_cast<double>(values.size()));
    for (auto& value : values) {
        value = static_cast<float>(static_cast<double>(value) / rms * scale);
    }
    return mlx::core::astype(
        array(values.begin(), Shape{1, rows, width}),
        mlx::core::float16);
}

double max_abs(const array& actual, const array& expected) {
    auto a = mlx::core::astype(
        mlx::core::reshape(actual, Shape{static_cast<int>(actual.size())}),
        mlx::core::float32);
    auto e = mlx::core::astype(
        mlx::core::reshape(expected, Shape{static_cast<int>(expected.size())}),
        mlx::core::float32);
    mlx::core::eval(a, e);
    const auto* actual_data = a.data<float>();
    const auto* expected_data = e.data<float>();
    double worst = 0.0;
    for (std::size_t index = 0; index < a.size(); ++index) {
        const double difference = std::abs(
            static_cast<double>(actual_data[index]) -
            static_cast<double>(expected_data[index]));
        worst = std::max(worst, difference);
    }
    return worst;
}

void check_group(
    const MfqContainer& model,
    const std::string& label,
    const std::vector<std::string>& names,
    int hidden,
    const std::vector<int>& rows_list) {
    std::vector<MlxLinear> linears;
    for (const auto& name : names) {
        try {
            linears.push_back(MlxLinear::load(model, name));
        } catch (const std::exception&) {
            std::cout << label << ": member " << name
                      << " is not present in this checkpoint; skipped\n";
            return;
        }
    }
    std::vector<MlxGroupedLinearWeightRef> refs;
    for (const auto& linear : linears) {
        auto ref = linear.grouped_weight_ref();
        if (!ref) {
            std::cout << label << ": member " << names[refs.size()]
                      << " has no packed weight; skipped\n";
            return;
        }
        refs.push_back(*ref);
    }
    MlxGroupedLinear grouped(refs);

    std::cout << label << " (members:";
    for (const auto& name : names) {
        std::cout << " " << name.substr(name.find_last_of('.') + 1);
    }
    std::cout << ")\n";
    for (const int rows : rows_list) {
        const auto input = make_input(hidden, rows, 1.0f);
        std::vector<array> grouped_outputs;
        std::vector<array> per_tensor;
        bool threw = false;
        try {
            grouped_outputs = grouped(input);
            mlx::core::eval(grouped_outputs);
        } catch (const std::exception& error) {
            std::cout << "  rows=" << rows << " grouped threw: "
                      << error.what() << "\n";
            threw = true;
        }
        for (auto& linear : linears) {
            per_tensor.push_back(linear(
                mlx::core::reshape(input, Shape{rows, hidden})));
        }
        mlx::core::eval(per_tensor);
        if (threw) {
            continue;
        }
        std::cout << "  rows=" << rows;
        double worst = 0.0;
        for (std::size_t index = 0; index < per_tensor.size(); ++index) {
            const double difference = max_abs(
                grouped_outputs[index],
                mlx::core::reshape(
                    per_tensor[index],
                    Shape{1, rows, static_cast<int>(per_tensor[index].shape(-1))}));
            std::cout << " [" << names[index].substr(names[index].find_last_of('.') + 1)
                      << " max=" << difference << "]";
            worst = std::max(worst, difference);
        }
        std::cout << (worst > 0.5 ? "   <-- UNFAITHFUL" : "") << "\n";
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr
            << "usage: qwen35_grouped_parity <container.mfq> [block]\n";
        return 2;
    }
    try {
        MfqContainer model(argv[1]);
        mfq::metal::install_legacy_tensor_compatibility(model);
        const int block = argc >= 3 ? std::stoi(argv[2]) : 11;
        const int hidden = 1024;
        const std::vector<int> rows_list{1, 2, 3, 4, 6, 8};
        check_group(
            model,
            "mixed-width QKV block " + std::to_string(block),
            {
                "model.block." + std::to_string(block) + ".attention.query.weight",
                "model.block." + std::to_string(block) + ".attention.key.weight",
                "model.block." + std::to_string(block) + ".attention.value.weight",
            },
            hidden,
            rows_list);
        check_group(
            model,
            "homogeneous QKV block 7",
            {
                "model.block.7.attention.query.weight",
                "model.block.7.attention.key.weight",
                "model.block.7.attention.value.weight",
            },
            hidden,
            rows_list);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
}
