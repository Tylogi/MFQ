#include "mfq_container.h"
#include "mlx_grouped_linear.h"
#include "mlx_vq.h"
#include "mlx_fp8_sq.h"
#include "mlx_mxfp4_sq.h"
#include "mlx_kernel_prepare.h"

#include <mlx/mlx.h>
#include <mlx/stream.h>
#include <mlx/fast_primitives.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <variant>
#include <unordered_set>
#include <vector>

int main(int argc, char** argv) {
    try {
        if (argc < 3 || argc > 6) {
            throw std::runtime_error("MODEL [--repetitions=N] [--dequantize] TENSOR [TENSOR]");
        }
        using namespace mlx::core;
        using Clock = std::chrono::steady_clock;
        using Json = nlohmann::json;
        int repetitions = 1024;
        int first_tensor = 2;
        bool dequantize = false;
        bool repetitions_set = false;
        while (first_tensor < argc) {
            const std::string option(argv[first_tensor]);
            if (option.starts_with("--repetitions=")) {
                repetitions = std::stoi(option.substr(14));
                repetitions_set = true;
            } else if (option == "--dequantize") {
                dequantize = true;
            } else {
                break;
            }
            ++first_tensor;
        }
        if (dequantize && !repetitions_set) repetitions = 8;
        if (repetitions < 1 || repetitions > 4096 || argc - first_tensor < 1 || argc - first_tensor > 2) {
            throw std::runtime_error("repetitions must be 1..4096 with one or two tensors");
        }
        if (dequantize && (repetitions > 32 || argc - first_tensor != 1)) {
            throw std::runtime_error("dequantization requires one tensor and at most 32 repetitions");
        }
        const mfq::metal::MfqContainer model(argv[1]);
        using Weight = std::variant<mfq::metal::MlxVqWeight,
            mfq::metal::MlxMxfp4SqWeight, mfq::metal::MlxFp8SqWeight>;
        std::vector<Weight> weights;
        weights.reserve(argc - first_tensor);
        std::uint64_t payload_bytes = 0;
        for (int argument = first_tensor; argument < argc; ++argument) {
            const auto& dtype = model.record(argv[argument]).dtype;
            const auto blob = model.read(argv[argument]);
            payload_bytes += blob.size();
            if (mfq::metal::is_fp8_sq_dtype(dtype)) {
                weights.emplace_back(mfq::metal::MlxFp8SqWeight::from_blob(dtype, blob));
            } else if (mfq::metal::is_mxfp4_sq_dtype(dtype)) {
                weights.emplace_back(mfq::metal::MlxMxfp4SqWeight::from_blob(blob));
            } else {
                weights.emplace_back(mfq::metal::MlxVqWeight::from_blob(dtype, blob));
            }
        }
        std::vector<mfq::metal::MlxGroupedLinearWeightRef> refs;
        for (const auto& weight : weights) {
            if (std::visit([](const auto& value) { return value.input_size(); }, weight)
                != std::visit([](const auto& value) { return value.input_size(); }, weights.front())) {
                throw std::runtime_error("input widths differ");
            }
            std::visit([&](const auto& value) { refs.emplace_back(&value); }, weight);
        }
        std::optional<mfq::metal::MlxGroupedLinear> grouped;
        if (weights.size() > 1) grouped.emplace(std::move(refs));
        const int width = std::visit([](const auto& value) { return value.input_size(); }, weights.front());
        std::uint64_t resident_bytes = 0;
        for (const auto& weight : weights)
            resident_bytes += std::visit([](const auto& value) { return value.packed_nbytes(); }, weight);
        const int output_width = std::visit([](const auto& value) { return value.output_size(); }, weights.front());
        const auto dequantized_bytes = static_cast<std::uint64_t>(width) * output_width * 2u;
        const auto retained_outputs = std::max<std::uint64_t>(repetitions + 2u, 5u);
        if (dequantize && dequantized_bytes > (1ull << 30u) / retained_outputs) {
            throw std::runtime_error("dequantization benchmark output working set exceeds 1 GiB; reduce repetitions");
        }
        const std::string label = std::visit([&](const auto& weight) {
            using W = std::decay_t<decltype(weight)>;
            if constexpr (std::is_same_v<W, mfq::metal::MlxVqWeight>) {
                return weight.format_label();
            } else {
                return model.record(argv[first_tensor]).dtype;
            }
        }, weights.front());
        for (int rows = 1; rows <= (dequantize ? 1 : 6); ++rows) {
            std::vector<float> values(rows * width);
            for (std::size_t index = 0; index < values.size(); ++index) {
                values[index] = static_cast<float>(static_cast<int>((index * 17 + 7) % 43) - 21) / 256.0f;
            }
            const auto input = astype(array(values.begin(), Shape{rows, width}), float16);
            eval(input);
            const auto dispatch = [&]() {
                if (dequantize) {
                    return std::vector<array>{std::visit([](const auto& value) {
                        return value.dequantize(float16);
                    }, weights.front())};
                }
                return grouped ? grouped->matmul(input)
                    : std::vector<array>{std::visit([&](const auto& value) { return value.matmul(input); }, weights.front())};
            };
            const auto build_batch = [&]() {
                std::vector<array> outputs;
                outputs.reserve(repetitions * weights.size());
                for (int iteration = 0; iteration < repetitions; ++iteration) {
                    const auto result = dispatch();
                    outputs.insert(outputs.end(), result.begin(), result.end());
                }
                return outputs;
            };
            std::vector<std::string> kernel_keys;
            std::unordered_set<const Primitive*> visited;
            const auto collect_kernels = [&](const auto& self, const array& value) -> void {
                if (!value.has_primitive()) return;
                const auto primitive = value.primitive_ptr();
                if (!visited.insert(primitive.get()).second) return;
                if (auto* native = dynamic_cast<mfq::metal::MlxPreparableKernel*>(primitive.get()))
                    kernel_keys.push_back(native->preparation_key());
                if (std::string_view(primitive->name()) == "CustomKernel")
                    kernel_keys.push_back(std::get<0>(
                        static_cast<mlx::core::fast::CustomKernel*>(primitive.get())->state()));
                for (const auto& dependency : value.inputs()) self(self, dependency);
            };
            for (const auto& output : dispatch()) collect_kernels(collect_kernels, output);
            eval(dispatch());
            synchronize();
            double warm_ms = 0.0;
            while (warm_ms < 50.0) {
                auto outputs = build_batch();
                const auto started = Clock::now();
                eval(outputs);
                synchronize();
                warm_ms += std::chrono::duration<double, std::milli>(Clock::now() - started).count();
            }
            std::uint64_t bytes = 0;
            const auto packed_bytes = [](std::uint64_t count, int bits) { return (count * bits + 7) / 8; };
            for (const auto& variant : weights) std::visit([&](const auto& weight) {
                const auto output = static_cast<std::uint64_t>(weight.output_size());
                using W = std::decay_t<decltype(weight)>;
                if constexpr (std::is_same_v<W, mfq::metal::MlxVqWeight>) {
                    bytes += output * 4;
                    if (weight.execution_layout() == 1) {
                        bytes += output * weight.groups() * 8;
                    } else {
                        bytes += packed_bytes(output * weight.vectors(), weight.index_bits());
                        bytes += packed_bytes(output * weight.groups(), weight.state_bits());
                        bytes += weight.aux_mode() == 3
                            ? packed_bytes(output * weight.groups(), 1)
                            : packed_bytes(output * ((width + 7) / 8), 7);
                    }
                } else if constexpr (std::is_same_v<W, mfq::metal::MlxFp8SqWeight>) {
                    const auto& layout = weight.wire_layout();
                    bytes += layout.bytes - layout.symbols + output * 5;
                } else {
                    const auto& layout = weight.wire_layout();
                    bytes += layout.bytes - layout.symbols + output * 9;
                }
            }, variant);
            if (dequantize) {
                bytes = std::visit([](const auto& value) { return value.packed_nbytes(); }, weights.front())
                    + static_cast<std::uint64_t>(width) * output_width * 2u;
            }
            std::vector<double> timings(3);
            std::vector<std::vector<array>> checked_outputs(3);
            for (int round = 0; round < 3; ++round) {
                auto outputs = build_batch();
                const auto started = Clock::now();
                eval(outputs);
                synchronize();
                timings[round] = std::chrono::duration<double, std::milli>(Clock::now() - started).count() / repetitions;
                checked_outputs[round].assign(outputs.end() - weights.size(), outputs.end());
            }
            for (int round = 0; round < 3; ++round) {
                const double ms = timings[round];
                std::uint64_t hash = 1469598103934665603ull;
                double checksum = 0.0;
                for (std::size_t projection = 0; projection < weights.size(); ++projection) {
                    auto checked = contiguous(astype(checked_outputs[round][projection], float32));
                    eval(checked);
                    const auto* data = checked.data<float>();
                    for (std::size_t index = 0; index < checked.size(); ++index) {
                        if (!std::isfinite(data[index])) throw std::runtime_error("non-finite output");
                        checksum += data[index] * static_cast<double>(index % 251 + 1);
                    }
                    const auto* raw = reinterpret_cast<const std::uint8_t*>(data);
                    for (std::size_t index = 0; index < checked.nbytes(); ++index) {
                        hash = (hash ^ raw[index]) * 1099511628211ull;
                    }
                }
                std::cout << Json{{"label", label}, {"tensor", argv[first_tensor]},
                    {"operation", dequantize ? "dequantize" : "matmul"},
                    {"mode", grouped ? "grouped" : "standalone"}, {"M", rows}, {"round", round},
                    {"ms", ms}, {"logical_bytes", bytes}, {"effective_GB_s", bytes / (ms * 1e6)},
                    {"input", width}, {"output", output_width}, {"projections", weights.size()},
                    {"payload_bytes", payload_bytes}, {"resident_bytes", resident_bytes},
                    {"kernel_keys", kernel_keys},
                    {"hash", std::to_string(hash)}, {"checksum", checksum}, {"repetitions", repetitions},
                    {"timing", "eval-prebuilt-graphs; host submission and synchronization included"},
                    {"bytes_scope", dequantize
                        ? "packed decoder inputs plus decoded output footprint; execution-only records excluded; not measured traffic"
                        : "executed packed streams; MMQ reuses weights across M; not physical DRAM"}}.dump() << std::endl;
            }
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
