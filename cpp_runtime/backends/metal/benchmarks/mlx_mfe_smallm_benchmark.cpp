#include "mfq_container.h"
#include "mlx_moe.h"
#include "mlx_kernel_prepare.h"
#include "mlx_vq.h"

#include <mlx/mlx.h>
#include <mlx/fast_primitives.h>
#include <mlx/stream.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <vector>

std::vector<bool> reusable_vq_experts(std::span<const std::uint8_t> blob, int experts) {
    std::vector<bool> reusable(experts, false);
    if (blob.size() < 20 || (std::memcmp(blob.data(), "MFE1", 4) != 0
            && std::memcmp(blob.data(), "NIM2", 4) != 0)) return reusable;
    std::size_t offset = 4;
    const auto read = [&]<typename T>() {
        if (offset > blob.size() || sizeof(T) > blob.size() - offset)
            throw std::runtime_error("truncated benchmark MFE metadata");
        T value;
        std::memcpy(&value, blob.data() + offset, sizeof(T));
        offset += sizeof(T);
        return value;
    };
    read.operator()<std::uint32_t>();
    read.operator()<std::uint32_t>();
    read.operator()<std::uint32_t>();
    const auto pools = read.operator()<std::uint32_t>();
    for (std::uint32_t pool = 0; pool < pools; ++pool) {
        const auto count = read.operator()<std::uint32_t>();
        const auto dtype_size = read.operator()<std::uint32_t>();
        const auto payload_size = read.operator()<std::uint64_t>();
        const auto runtime_size = read.operator()<std::uint64_t>();
        std::vector<std::int32_t> ids(count);
        for (auto& id : ids) id = read.operator()<std::int32_t>();
        const auto total = std::uint64_t(dtype_size) + runtime_size + payload_size;
        if (offset > blob.size() || total > blob.size() - offset)
            throw std::runtime_error("truncated benchmark MFE pool");
        const std::string_view dtype(reinterpret_cast<const char*>(blob.data() + offset), dtype_size);
        offset += dtype_size;
        const auto runtime = blob.subspan(offset, runtime_size);
        offset += runtime_size;
        const auto payload = blob.subspan(offset, payload_size);
        offset += payload_size;
        if (!mfq::metal::is_vq_dtype(dtype)) continue;
        const auto profile = mfq::metal::inspect_vq_blob(dtype, payload, runtime).format_label;
        const bool eligible = profile == "NVQ1-S" || profile == "NVQ1-L"
            || profile == "NVQ2J" || profile == "NVQ2J-L" || profile == "NVQ2J-XL"
            || profile == "NVQ3J" || profile == "NVQ3J-512" || profile == "NVQ3J-L";
        for (auto id : ids) {
            if (id < 0 || id >= experts) throw std::runtime_error("invalid benchmark expert ID");
            reusable[id] = eligible;
        }
    }
    return reusable;
}

int check_vq(char** argv) {
    using namespace mlx::core;
    const mfq::metal::MfqContainer model(argv[2]);
    const auto weight = mfq::metal::MlxMfeWeight::from_blob(model.read(argv[3]));
    const mfq::metal::MfqContainer direct_model(argv[4]);
    const auto direct = mfq::metal::MlxVqWeight::from_blob(
        direct_model.record(argv[5]).dtype, direct_model.read(argv[5]));
    if (direct.input_size() != weight.neuron_len() || direct.output_size() != weight.out_per_expert()) {
        throw std::runtime_error("VQ reference geometry differs from routed weight");
    }
    const std::string mode(argv[6]);
    if (mode != "shared" && mode != "routed") {
        throw std::runtime_error("input mode must be shared or routed");
    }
    const auto dense = contiguous(direct.dequantize(float32));
    eval(dense);
    const auto* weights = dense.data<float>();
    const int routes = std::min(10, weight.experts());
    const int width = weight.neuron_len();
    const int output = weight.out_per_expert();
    for (const auto dtype : {float16, float32}) {
        for (int rows = 1; rows <= 6; ++rows) {
            std::vector<float> values(rows * (mode == "routed" ? routes : 1) * width);
            for (std::size_t index = 0; index < values.size(); ++index) {
                values[index] = float(int((index * 17 + 7) % 43) - 21) / 256.0f;
            }
            std::vector<std::int32_t> ids(rows * routes);
            for (std::size_t index = 0; index < ids.size(); ++index) ids[index] = index % routes;
            const auto input = astype(array(values.begin(), mode == "routed"
                ? Shape{rows, routes, width} : Shape{rows, width}), dtype);
            const auto actual = contiguous(astype(weight.routed_matmul(input,
                array(ids.begin(), Shape{rows, routes})), float32));
            eval(actual);
            std::vector<float> reference(rows * routes * output);
            for (int row = 0; row < rows; ++row) {
                for (int route = 0; route < routes; ++route) {
                    const auto* source = values.data() + (row * (mode == "routed" ? routes : 1)
                        + (mode == "routed" ? route : 0)) * width;
                    for (int column = 0; column < output; ++column) {
                        double sum = 0.0;
                        for (int k = 0; k < width; ++k) sum += double(source[k]) * weights[column * width + k];
                        reference[(row * routes + route) * output + column] = float(sum);
                    }
                }
            }
            const auto rounded = contiguous(astype(astype(array(reference.begin(),
                Shape{rows, routes, output}), dtype), float32));
            eval(rounded);
            double maximum = 0.0, squared = 0.0, reference_squared = 0.0;
            for (std::size_t index = 0; index < reference.size(); ++index) {
                const auto difference = double(actual.data<float>()[index]) - rounded.data<float>()[index];
                if (!std::isfinite(difference)) throw std::runtime_error("non-finite VQ check");
                maximum = std::max(maximum, std::abs(difference));
                squared += difference * difference;
                reference_squared += double(reference[index]) * reference[index];
            }
            std::cout << nlohmann::json{{"operation", "numerical-check"}, {"M", rows},
                {"dtype", dtype == float16 ? "F16" : "F32"}, {"mode", mode},
                {"max_abs", maximum}, {"rms", std::sqrt(squared / reference.size())},
                {"relative_l2", std::sqrt(squared / std::max(reference_squared, 1e-30))},
                {"reference", "same packed VQ payload; F32 decode; CPU F64 accumulation; output-type rounding"}}
                .dump() << std::endl;
        }
    }
    return 0;
}

int main(int argc, char** argv) {
    try {
        if (argc == 7 && std::string_view(argv[1]) == "--check-vq") return check_vq(argv);
        if (argc < 5) {
            throw std::runtime_error(
                "MODEL TENSOR shared|routed LABEL:BYTES_PER_EXPERT:ID,... [...]");
        }
        using namespace mlx::core;
        using Json = nlohmann::json;
        using Clock = std::chrono::steady_clock;
        const std::string mode(argv[3]);
        if (mode != "shared" && mode != "routed") {
            throw std::runtime_error("input mode must be shared or routed");
        }
        const mfq::metal::MfqContainer model(argv[1]);
        const auto blob = model.read(argv[2]);
        const auto weight = mfq::metal::MlxMfeWeight::from_blob(blob);
        const auto reusable = reusable_vq_experts(blob, weight.experts());
        int repetitions = 128;
        int selected_rows = 0;
        int first_selection = 4;
        bool use_route_groups = false;
        while (first_selection < argc) {
            const std::string argument(argv[first_selection]);
            if (argument == "--route-groups") use_route_groups = true;
            else if (argument.starts_with("--repetitions=")) repetitions = std::stoi(argument.substr(14));
            else if (argument.starts_with("--rows=")) selected_rows = std::stoi(argument.substr(7));
            else break;
            ++first_selection;
        }
        if (repetitions < 1 || repetitions > 4096 || first_selection >= argc)
            throw std::runtime_error("repetitions must be 1..4096 with an expert selection");
        if (selected_rows < 0 || selected_rows > 6)
            throw std::runtime_error("rows must be 1..6, or 0 for all");
        for (int argument = first_selection; argument < argc; ++argument) {
            const std::string spec(argv[argument]);
            const auto first = spec.find(':');
            const auto second = spec.find(':', first + 1);
            if (first == std::string::npos || second == std::string::npos) {
                throw std::runtime_error("invalid expert selection");
            }
            const auto label = spec.substr(0, first);
            const auto bytes = std::stoull(spec.substr(first + 1, second - first - 1));
            std::vector<std::int32_t> candidates;
            std::istringstream list(spec.substr(second + 1));
            for (std::string id; std::getline(list, id, ',');) {
                const auto value = std::stoi(id);
                if (value < 0 || value >= weight.experts()) {
                    throw std::runtime_error("expert ID out of range");
                }
                candidates.push_back(value);
            }
            if (candidates.empty() || bytes == 0) {
                throw std::runtime_error("empty expert selection or byte count");
            }
            const int routes = std::min(10, static_cast<int>(candidates.size()));
            for (int rows = selected_rows == 0 ? 1 : selected_rows;
                 rows <= (selected_rows == 0 ? 6 : selected_rows); ++rows) {
                synchronize();
                clear_cache();
                std::vector<float> values(static_cast<std::size_t>(rows)
                    * (mode == "routed" ? routes : 1) * weight.neuron_len());
                for (std::size_t index = 0; index < values.size(); ++index) {
                    values[index] = static_cast<float>(
                        static_cast<int>((index * 17 + 7) % 43) - 21) / 256.0f;
                }
                const auto input = astype(array(values.begin(), mode == "routed"
                    ? Shape{rows, routes, weight.neuron_len()}
                    : Shape{rows, weight.neuron_len()}), float16);
                const int cycle = static_cast<int>(candidates.size())
                    / std::gcd(static_cast<int>(candidates.size()), routes);
                std::vector<array> selections;
                std::vector<array> route_groups;
                for (int step = 0; step < cycle; ++step) {
                    std::vector<std::int32_t> ids(rows * routes);
                    for (int index = 0; index < rows * routes; ++index) {
                        ids[index] = candidates[(step * routes + index) % candidates.size()];
                    }
                    selections.emplace_back(ids.begin(), Shape{rows, routes});
                    if (use_route_groups && rows > 1) {
                        std::vector<std::int32_t> groups(ids.size() * 8, -1);
                        for (std::size_t index = 0; index < ids.size(); ++index) {
                            int matches = 0;
                            for (std::size_t route = 0; route < ids.size(); ++route)
                                if (ids[route] == ids[index]) {
                                    if (matches == 6) throw std::runtime_error("route group exceeds Top-K uniqueness bound");
                                    groups[index * 8 + 2 + matches++] = int(route);
                                }
                            groups[index * 8] = ids[index];
                            groups[index * 8 + 1] = matches;
                        }
                        route_groups.emplace_back(groups.begin(), Shape{rows, routes, 8});
                    }
                }
                eval(input);
                eval(selections);
                eval(route_groups);
                const auto routed = [&](int step) {
                    return weight.routed_matmul(input, selections[step % cycle], route_groups.empty()
                        ? nullptr : &route_groups[step % cycle]);
                };
                std::vector<std::string> kernel_keys;
                std::unordered_set<const Primitive*> visited;
                const auto collect_kernels = [&](const auto& self, const array& value) -> void {
                    if (!value.has_primitive()) return;
                    const auto primitive = value.primitive_ptr();
                    if (!visited.insert(primitive.get()).second) return;
                    if (auto* native = dynamic_cast<mfq::metal::MlxPreparableKernel*>(primitive.get())) {
                        kernel_keys.push_back(native->preparation_key());
                    }
                    if (std::string_view(primitive->name()) == "CustomKernel")
                        kernel_keys.push_back(std::get<0>(
                            static_cast<mlx::core::fast::CustomKernel*>(primitive.get())->state()));
                    for (const auto& dependency : value.inputs()) self(self, dependency);
                };
                collect_kernels(collect_kernels, routed(0));
                const bool route_reuse = rows > 1 && std::any_of(kernel_keys.begin(), kernel_keys.end(),
                    [](const auto& key) { return key.starts_with("mfq_native_mfe_v"); });
                double decoded_blocks = 0.0;
                double distinct_experts = 0.0;
                for (int step = 0; step < repetitions; ++step) {
                    std::unordered_map<std::int32_t, int> frequencies;
                    for (int index = 0; index < rows * routes; ++index)
                        ++frequencies[candidates[((step % cycle) * routes + index) % candidates.size()]];
                    distinct_experts += frequencies.size();
                    for (const auto& [expert, count] : frequencies)
                        decoded_blocks += route_reuse && reusable[expert] ? (count + 5) / 6 : count;
                }
                decoded_blocks /= repetitions;
                distinct_experts /= repetitions;
                for (int step = 0; step < cycle + 4; ++step) {
                    auto output = routed(step);
                    eval(output);
                }
                synchronize();
                double warm_ms = 0.0;
                while (warm_ms < 50.0) {
                    std::vector<array> outputs;
                    for (int step = 0; step < repetitions; ++step) {
                        outputs.push_back(routed(step));
                    }
                    const auto started = Clock::now();
                    eval(outputs);
                    synchronize();
                    warm_ms += std::chrono::duration<double, std::milli>(
                        Clock::now() - started).count();
                }
                for (int round = 0; round < 3; ++round) {
                    std::vector<array> outputs;
                    outputs.reserve(repetitions);
                    for (int step = 0; step < repetitions; ++step) {
                        outputs.push_back(routed(step));
                    }
                    const auto started = Clock::now();
                    eval(outputs);
                    synchronize();
                    const double ms = std::chrono::duration<double, std::milli>(
                        Clock::now() - started).count() / repetitions;
                    auto checked = contiguous(astype(outputs.back(), float32));
                    eval(checked);
                    std::uint64_t hash = 1469598103934665603ull;
                    double checksum = 0.0;
                    const auto* data = checked.data<float>();
                    for (std::size_t index = 0; index < checked.size(); ++index) {
                        if (!std::isfinite(data[index])) {
                            throw std::runtime_error("non-finite projection output");
                        }
                        checksum += data[index] * static_cast<double>(index % 251 + 1);
                    }
                    const auto* raw = reinterpret_cast<const std::uint8_t*>(data);
                    for (std::size_t index = 0; index < checked.nbytes(); ++index) {
                        hash = (hash ^ raw[index]) * 1099511628211ull;
                    }
                    const auto logical_bytes = bytes * decoded_blocks;
                    std::cout << Json{
                        {"label", label}, {"tensor", argv[2]}, {"M", rows},
                        {"round", round}, {"ms", ms}, {"logical_bytes", logical_bytes},
                        {"effective_GB_s", logical_bytes / (ms * 1e6)},
                        {"routes", routes}, {"candidates", candidates.size()},
                        {"distinct_experts", distinct_experts}, {"decoded_expert_blocks", decoded_blocks},
                        {"requested_projection_bytes", bytes * rows * routes},
                        {"kernel_keys", kernel_keys},
                        {"input", weight.neuron_len()}, {"output", weight.out_per_expert()},
                        {"payload_bytes", blob.size()}, {"resident_bytes", weight.packed_nbytes()},
                        {"shape", weight.out_per_expert() > weight.neuron_len()
                            ? "expansion" : "contraction"},
                        {"hash", std::to_string(hash)}, {"checksum", checksum},
                        {"repetitions", repetitions},
                        {"timing", "eval-prebuilt-graphs; host submission and synchronization included"},
                        {"allocation_cache", "cleared before each shape; warmed before timing"},
                        {"bytes_scope", "packed execution streams per decoder block; cross-M reuse accounted; not physical DRAM"}
                    }.dump() << std::endl;
                }
            }
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
