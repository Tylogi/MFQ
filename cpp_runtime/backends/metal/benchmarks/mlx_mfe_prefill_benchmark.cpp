#include "mfq_container.h"
#include "mlx_kernel_prepare.h"
#include "mlx_moe.h"
#include "mlx_moe_ops.h"

#include <mlx/mlx.h>
#include <mlx/primitives.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <sstream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    try {
        if (argc < 5 || argc > 9)
            throw std::runtime_error("MODEL PREFIX ROWS REPETITIONS [BLOCK_ROWS [uniform|skewed|file:ROUTES_JSON [EXPERT_IDS [PROJECTION]]]]");
        using namespace mlx::core;
        const mfq::metal::MfqContainer model(argv[1]);
        const int rows = std::stoi(argv[3]);
        const int repetitions = std::stoi(argv[4]);
        const int block_rows = argc >= 6 ? std::stoi(argv[5]) : 0;
        const std::string routing = argc >= 7 ? argv[6] : "uniform";
        const std::string selected_projection = argc == 9 ? argv[8] : "all";
        const bool replay = routing.starts_with("file:");
        if (routing != "uniform" && routing != "skewed" && !replay)
            throw std::runtime_error("invalid benchmark routing");
        if (selected_projection != "all" && selected_projection != "gate"
            && selected_projection != "up" && selected_projection != "down"
            && selected_projection != "gate_up")
            throw std::runtime_error("invalid benchmark projection");
        if (rows < 1 || rows > 32768 || repetitions < 1 || repetitions > 256)
            throw std::runtime_error("invalid benchmark dimensions");
        const auto document = nlohmann::json::parse(model.read("__mfq_asset__/model_config.json"));
        const auto& config = document.contains("text_config") ? document.at("text_config") : document;
        const int routes = config.at("num_experts_per_tok").get<int>();
        std::vector<std::int32_t> replay_selection;
        if (replay) {
            std::ifstream stream(routing.substr(5));
            if (!stream) throw std::runtime_error("cannot open benchmark routes");
            const auto route_data = nlohmann::json::parse(stream);
            replay_selection = route_data.at("selection").get<std::vector<std::int32_t>>();
            if (route_data.at("rows").get<int>() != rows
                || route_data.at("routes").get<int>() != routes
                || replay_selection.size() != std::size_t(rows) * std::size_t(routes))
                throw std::runtime_error("benchmark routes shape mismatch");
        }
        for (const auto* projection : {"gate", "up", "down", "gate_up"}) {
            if (selected_projection != "all" && selected_projection != projection) continue;
            const bool fused = std::strcmp(projection, "gate_up") == 0;
            const bool routed = std::strcmp(projection, "down") == 0;
            const std::string root = std::string(argv[2]) + ".experts.";
            const auto first = model.read(root + (fused ? "gate" : projection) + ".weight");
            const auto second = fused ? model.read(root + "up.weight") : std::vector<std::uint8_t>{};
            const std::vector<std::span<const std::uint8_t>> blobs = fused
                ? std::vector<std::span<const std::uint8_t>>{first, second}
                : std::vector<std::span<const std::uint8_t>>{first};
            const auto weight = mfq::metal::MlxMfeWeight::from_projection_blobs(blobs);
            std::vector<int> candidates;
            if (argc >= 8) {
                std::istringstream stream(argv[7]);
                std::string token;
                while (std::getline(stream, token, ',')) {
                    std::size_t end = 0;
                    const int id = std::stoi(token, &end);
                    if (end != token.size() || id < 0 || id >= weight.experts()
                        || std::find(candidates.begin(), candidates.end(), id) != candidates.end())
                        throw std::runtime_error("invalid benchmark expert ID");
                    candidates.push_back(id);
                }
                if (candidates.empty()) throw std::runtime_error("no benchmark experts");
            } else {
                candidates.resize(weight.experts());
                std::iota(candidates.begin(), candidates.end(), 0);
            }
            int stride = 37;
            while (std::gcd(stride, int(candidates.size())) != 1) ++stride;
            std::vector<float> values(rows * (routed ? routes : 1) * weight.neuron_len());
            for (std::size_t index = 0; index < values.size(); ++index)
                values[index] = float(int((index * 13 + 5) % 47) - 23) / 4096.0f;
            const auto x = astype(array(values.begin(), routed
                ? Shape{rows, routes, weight.neuron_len()} : Shape{rows, weight.neuron_len()}), float16);
            std::vector<std::int32_t> selection(rows * routes);
            for (std::size_t index = 0; index < selection.size(); ++index) {
                if (replay) {
                    const auto id = replay_selection[index];
                    if (id < 0 || id >= weight.experts())
                        throw std::runtime_error("invalid replay expert ID");
                    selection[index] = id;
                    continue;
                }
                const auto token = index / routes;
                const auto route = routing == "skewed" && token % 4 != 0
                    ? (token % 8) * routes + index % routes : index;
                selection[index] = candidates[(route * stride + 7) % candidates.size()];
            }
            const auto ids = array(selection.begin(), Shape{rows, routes});
            const auto order = astype(argsort(reshape(ids, {-1})), int32);
            const auto plan = weight.build_grouped_mmq_plan(ids, order,
                block_rows > 0 ? block_rows
                    : weight.recommended_grouped_mmq_block_rows(rows * routes, fused));
            const auto execute = [&] {
                return weight.routed_matmul_sorted(x, ids, order, false, fused, 0.0f, &plan);
            };
            std::set<std::string> kernels;
            std::size_t activation_variant_bytes = 0;
            std::function<void(const array&)> visit = [&](const array& node) {
                if (!node.has_primitive()) return;
                if (const auto* kernel = dynamic_cast<const mfq::metal::MlxPreparableKernel*>(&node.primitive()))
                    kernels.insert(kernel->preparation_key());
                if (std::strcmp(node.primitive().name(), "GroupedMmqPrimitive") == 0
                    && node.inputs().size() == 28)
                    activation_variant_bytes = std::max(activation_variant_bytes, node.inputs()[22].nbytes());
                for (const auto& input : node.inputs()) visit(input);
            };
            const auto inspected = execute();
            const bool fused_dispatch = fused && inspected.has_primitive()
                && std::strcmp(inspected.primitive().name(), "GroupedMmqPrimitive") == 0;
            visit(inspected);
            eval(x, ids, order, plan.block_meta, plan.block_count);
            for (int warmup = 0; warmup < 2; ++warmup) { eval(execute()); synchronize(); }
            std::vector<double> timings;
            for (int round = 0; round < 3; ++round) {
                const auto start = std::chrono::steady_clock::now();
                for (int repeat = 0; repeat < repetitions; ++repeat) { eval(execute()); synchronize(); }
                const auto milliseconds = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - start).count() / repetitions;
                timings.push_back(milliseconds);
            }
            const auto output = contiguous(astype(execute(), float32));
            eval(output);
            double split_reference_error = 0.0;
            double split_reference_limit = 0.0;
            if (fused) {
                const auto separate = weight.routed_matmul_sorted(
                    x, ids, order, false, false, 0.0f, &plan);
                const auto parts = split(separate, 2, 1);
                const auto reference = contiguous(astype(
                    mfq::metal::moe_swiglu_pair(parts[0], parts[1]), float32));
                eval(reference);
                const auto* expected = reference.data<float>();
                const auto* actual = output.data<float>();
                double maximum = 0.0;
                for (std::size_t index = 0; index < output.size(); ++index) {
                    maximum = std::max(maximum, std::abs(double(expected[index])));
                    split_reference_error = std::max(split_reference_error,
                        std::abs(double(actual[index]) - expected[index]));
                }
                split_reference_limit = 0.005 * maximum + 1e-7;
                if (split_reference_error > split_reference_limit)
                    throw std::runtime_error("fused Gate/Up differs from split reference");
            }
            const auto* data = output.data<float>();
            std::uint64_t hash = 1469598103934665603ull;
            for (std::size_t index = 0; index < output.size(); ++index) {
                if (!std::isfinite(data[index])) throw std::runtime_error("nonfinite output");
                std::uint32_t bits;
                std::memcpy(&bits, data + index, sizeof(bits));
                for (int byte = 0; byte < 4; ++byte) { hash ^= (bits >> (byte * 8)) & 255; hash *= 1099511628211ull; }
            }
            for (std::size_t round = 0; round < timings.size(); ++round) {
                std::cout << nlohmann::json{{"projection", projection}, {"rows", rows},
                    {"routes", routes}, {"experts", weight.experts()}, {"round", round},
                    {"candidate_experts", candidates.size()},
                    {"kernels", kernels},
                    {"fused_dispatch", fused_dispatch},
                    {"split_reference_error", split_reference_error},
                    {"split_reference_limit", split_reference_limit},
                    {"activation_variant_bytes", activation_variant_bytes},
                    {"ms", timings[round]}, {"repetitions", repetitions}, {"hash", std::to_string(hash)},
                    {"input", weight.neuron_len()}, {"output", weight.out_per_expert()},
                    {"payload_bytes", first.size() + second.size()}, {"resident_bytes", weight.packed_nbytes()},
                    {"block_rows", plan.block_rows},
                    {"routing", routing},
                    {"input_mode", routed ? "routed" : "shared"},
                    {"timing", "graph-build+eval+synchronize; warmed; route plan excluded"}}
                    .dump() << std::endl;
            }
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
