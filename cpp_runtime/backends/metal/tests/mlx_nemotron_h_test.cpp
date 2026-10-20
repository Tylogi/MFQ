// Nemotron-H Metal correctness bridge.
//
// Loads a synthetic MFQ fixture and prints the same JSON documents the CUDA
// diagnostics emit, so tests/test_metal_nemotron_*.py can compare the native
// Metal graph against the shared NumPy oracle.
#include "mlx_nemotron_h_causal_lm.h"
#include "mlx_linear_attention.h"
#include "grid_vision.h"
#include "mfq_container.h"
#include "nemotron_h_model.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <mlx/mlx.h>

namespace {

using mlx::core::Shape;
using mlx::core::array;
using mfq::metal::MlxNemotronHCausalLm;
using mfq::metal::MlxSamplingParams;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

nlohmann::json json_tensor(const array& value) {
    auto host = mlx::core::contiguous(
        mlx::core::astype(value, mlx::core::float32));
    host.eval();
    const auto* data = host.data<float>();
    return nlohmann::json{
        {"shape", host.shape()},
        {"data", std::vector<float>(data, data + host.size())},
    };
}

array prompt_ids(const std::vector<std::int32_t>& values) {
    return array(
        values.begin(), Shape{1, static_cast<int>(values.size())},
        mlx::core::int32);
}

array deterministic_hidden(int tokens, int width) {
    std::vector<float> values(static_cast<std::size_t>(tokens * width));
    for (std::size_t index = 0; index < values.size(); ++index) {
        values[index] =
            static_cast<float>(index % 31) * 0.01f - 0.15f;
    }
    return array(values.begin(), Shape{1, tokens, width});
}

void compare(
    const array& actual,
    const array& expected,
    const char* label) {
    auto left = mlx::core::contiguous(mlx::core::astype(actual, mlx::core::float32));
    auto right = mlx::core::contiguous(mlx::core::astype(expected, mlx::core::float32));
    left.eval();
    right.eval();
    require(left.shape() == right.shape(), std::string(label) + " changed its shape");
    const auto* a = left.data<float>();
    const auto* b = right.data<float>();
    double worst = 0.0;
    for (std::size_t index = 0; index < left.size(); ++index) {
        const double error = std::abs(double(a[index]) - b[index]) /
            (0.003 + 0.003 * std::abs(double(b[index])));
        worst = std::max(worst, error);
    }
    require(worst <= 1.0, std::string(label) + " changed its value");
}

int run_text_check(MlxNemotronHCausalLm& model) {
    require(model.config().vocab_size >= 8 &&
                model.config().max_position_embeddings >= 16,
            "Nemotron-H diagnostic requires vocab>=8 and context>=16");
    const auto ids = prompt_ids({1, 2, 3, 4, 5, 6, 7});
    nlohmann::json result;
    model.reset_cache(1);
    auto full = model.forward_embeddings_with_hidden(
        model.embed_tokens(ids), true);
    result["full_hidden"] = json_tensor(full.second);
    result["full"] = json_tensor(full.first);
    model.reset_cache(1);
    std::vector<array> pieces;
    for (const auto& [begin, count] :
         std::vector<std::pair<int, int>>{{0, 2}, {2, 1}, {3, 4}}) {
        pieces.push_back(model.forward(
            mlx::core::slice(ids, Shape{0, begin}, Shape{1, begin + count}), true));
    }
    result["chunked"] = json_tensor(
        mlx::core::concatenate(std::move(pieces), 1));
    model.reset_cache(1);
    model.forward(mlx::core::slice(ids, Shape{0, 0}, Shape{1, 5}), true);
    pieces.clear();
    for (int token = 5; token < 7; ++token) {
        pieces.push_back(model.forward(
            mlx::core::slice(ids, Shape{0, token}, Shape{1, token + 1}), true));
    }
    result["decode"] = json_tensor(mlx::core::concatenate(std::move(pieces), 1));
    model.reset_cache(1);
    model.forward(mlx::core::slice(ids, Shape{0, 0}, Shape{1, 4}), true);
    const std::vector<std::int64_t> prompt{1, 2, 3, 4};
    auto snapshot = model.capture_text_session_state(prompt);
    auto continued = model.forward(
        mlx::core::slice(ids, Shape{0, 4}, Shape{1, 5}), true);
    model.restore_text_session_state(snapshot);
    result["restored"] = json_tensor(model.forward(
        mlx::core::slice(ids, Shape{0, 4}, Shape{1, 5}), true));
    result["continued"] = json_tensor(continued);
    for (bool accept : {false, true}) {
        const std::string name = accept ? "committed" : "rejected";
        model.reset_cache(1);
        model.forward(mlx::core::slice(ids, Shape{0, 0}, Shape{1, 2}), true);
        auto window = model.forward_verified(
            mlx::core::slice(ids, Shape{0, 2}, Shape{1, 4}), 1);
        result[name + "_verify"] = json_tensor(window.first);
        if (accept) {
            model.commit_speculative();
        } else {
            model.rollback_speculative(0, 1);
        }
        require(model.cache_position() == (accept ? 4 : 3),
                "Nemotron-H transaction cache position mismatch");
        result[name] = json_tensor(model.forward(
            mlx::core::slice(ids, Shape{0, 5}, Shape{1, 6}), true));
        model.reset_cache(1);
        model.forward(mlx::core::slice(ids, Shape{0, 0}, Shape{1, 2}), true);
        model.forward(
            mlx::core::slice(ids, Shape{0, 2}, Shape{1, accept ? 4 : 3}), true);
        result[name + "_reference"] = json_tensor(model.forward(
            mlx::core::slice(ids, Shape{0, 5}, Shape{1, 6}), true));
    }
    model.reset_cache(2);
    result["batch"] = json_tensor(model.forward(mlx::core::repeat(ids, 2, 0), true));
    model.reset_cache(1);
    result["reset"] = json_tensor(model.forward(ids, true));
    result["architecture"] = "nemotron_h";
    std::cout << "nemotron_h_check " << result.dump() << '\n';
    return 0;
}

std::vector<std::int64_t> generate_tokens(
    MlxNemotronHCausalLm& model,
    const std::vector<std::int64_t>& prompt,
    bool enable_mtp) {
    MlxSamplingParams sampling;
    sampling.temperature = 0.0;
    sampling.top_k = 1;
    sampling.top_p = 1.0;
    sampling.enable_mtp = enable_mtp;
    model.reset_cache(1);
    if (model.mtp()) model.mtp()->reset_cache(1);
    std::vector<std::int64_t> tokens;
    model.generate(
        prompt,
        sampling,
        32,
        [&](std::int64_t token) {
            tokens.push_back(token);
            return true;
        });
    return tokens;
}

int run_mtp_check(MlxNemotronHCausalLm& model) {
    require(model.config().vocab_size >= 8 &&
                model.config().max_position_embeddings >= 32,
            "Nemotron-H MTP diagnostic requires vocab>=8 and context>=32");
    auto& mtp = *model.mtp();
    const auto ids = prompt_ids({1, 2, 3, 4, 5, 6, 7});
    const int hidden = static_cast<int>(model.config().hidden_size);
    const auto previous = deterministic_hidden(7, hidden);
    const auto& embedding = model.embedding();
    nlohmann::json result;
    result["architecture"] = "nemotron_h";
    result["previous"] = json_tensor(previous);
    result["ids"] = json_tensor(mlx::core::astype(ids, mlx::core::float32));
    result["embedding"] = json_tensor(model.embed_tokens(ids));
    mtp.reset_cache(1);
    const auto full = mtp.forward(previous, ids, embedding);
    result["head"] = json_tensor(full);
    result["logits"] = json_tensor(model.project_logits(full));
    mtp.reset_cache(1);
    std::vector<array> pieces;
    for (const auto& [begin, count] :
         std::vector<std::pair<int, int>>{{0, 2}, {2, 1}, {3, 4}}) {
        pieces.push_back(mtp.forward(
            mlx::core::slice(previous, Shape{0, begin, 0},
                             Shape{1, begin + count, hidden}),
            mlx::core::slice(ids, Shape{0, begin}, Shape{1, begin + count}),
            embedding));
    }
    compare(mlx::core::concatenate(std::move(pieces), 1), full, "chunked forward");
    mtp.reset_cache(1);
    (void)mtp.forward(
        mlx::core::slice(previous, Shape{0, 0, 0}, Shape{1, 4, hidden}),
        mlx::core::slice(ids, Shape{0, 0}, Shape{1, 4}), embedding);
    const int committed = mtp.cache_position();
    auto rejected = mtp.forward(
        mlx::core::slice(previous, Shape{0, 4, 0}, Shape{1, 5, hidden}),
        mlx::core::slice(ids, Shape{0, 4}, Shape{1, 5}), embedding);
    mtp.trim_cache_to(committed);
    auto replayed = mtp.forward(
        mlx::core::slice(previous, Shape{0, 4, 0}, Shape{1, 5, hidden}),
        mlx::core::slice(ids, Shape{0, 4}, Shape{1, 5}), embedding);
    compare(replayed, rejected, "rollback replay");
    const auto tail = mlx::core::concatenate(
        {replayed,
         mtp.forward(
             mlx::core::slice(previous, Shape{0, 5, 0}, Shape{1, 7, hidden}),
             mlx::core::slice(ids, Shape{0, 5}, Shape{1, 7}), embedding)},
        1);
    compare(tail,
            mlx::core::slice(full, Shape{0, 4, 0}, Shape{1, 7, hidden}),
            "continuation after rollback");
    result["replayed"] = json_tensor(replayed);
    result["rollback"] = json_tensor(
        mlx::core::slice(full, Shape{0, 4, 0}, Shape{1, 5, hidden}));
    mtp.reset_cache(1);
    (void)mtp.forward(
        mlx::core::slice(previous, Shape{0, 0, 0}, Shape{1, 4, hidden}),
        mlx::core::slice(ids, Shape{0, 0}, Shape{1, 4}), embedding);
    auto committed_draft = mtp.forward(
        mlx::core::slice(previous, Shape{0, 4, 0}, Shape{1, 5, hidden}),
        mlx::core::slice(ids, Shape{0, 4}, Shape{1, 5}), embedding);
    auto committed_next = mtp.forward(
        mlx::core::slice(previous, Shape{0, 5, 0}, Shape{1, 7, hidden}),
        mlx::core::slice(ids, Shape{0, 5}, Shape{1, 7}), embedding);
    result["committed"] = json_tensor(
        mlx::core::concatenate({committed_draft, committed_next}, 1));
    mtp.reset_cache(1);
    auto first = mtp.forward(
        mlx::core::slice(previous, Shape{0, 0, 0}, Shape{1, 1, hidden}),
        mlx::core::slice(ids, Shape{0, 0}, Shape{1, 1}), embedding);
    auto second = mtp.forward(
        first, mlx::core::slice(ids, Shape{0, 1}, Shape{1, 2}), embedding);
    result["chain_first"] = json_tensor(first);
    result["chain_second"] = json_tensor(second);
    model.reset_cache(1);
    auto target = model.forward_embeddings_with_hidden(
        model.embed_tokens(ids), true);
    result["target_hidden"] = json_tensor(target.second);
    result["target_normalized"] = json_tensor(target.second);
    mtp.reset_cache(1);
    result["head_from_target"] = json_tensor(mtp.forward(
        mlx::core::slice(target.second, Shape{0, 0, 0}, Shape{1, 6, hidden}),
        mlx::core::slice(ids, Shape{0, 1}, Shape{1, 7}), embedding));
    const std::vector<std::int64_t> prompt{1, 2, 3, 4, 5};
    const auto speculative = generate_tokens(model, prompt, true);
    result["greedy_mtp"] = speculative;
    result["cycles"] = model.last_mtp_stats().cycles;
    const auto plain = generate_tokens(model, prompt, false);
    result["greedy_plain"] = plain;
    require(speculative == plain,
            "Nemotron-H MTP greedy generation differs from ordinary decode");
    const auto disabled = generate_tokens(model, prompt, false);
    result["greedy_disabled"] = disabled;
    require(disabled == plain, "disabled Nemotron-H MTP changed the output");
    std::cout << "nemotron_h_mtp_check " << result.dump() << '\n';
    return 0;
}

// Two-step decode through the shared cached-depthwise convolution. The fast
// single-token kernel must match the generic causal reference for the
// Nemotron-H geometry (float32, 160 channels, kernel 4).
int run_cached_conv_check() {
    using namespace mlx::core;
    constexpr int channels = 160;
    constexpr int kernel = 4;
    constexpr int state_length = kernel - 1;
    std::vector<float> state_values(
        static_cast<std::size_t>(state_length * channels));
    std::vector<float> input_values(static_cast<std::size_t>(channels));
    std::vector<float> weight_values(
        static_cast<std::size_t>(channels * kernel));
    std::vector<float> bias_values(static_cast<std::size_t>(channels));
    for (std::size_t index = 0; index < state_values.size(); ++index) {
        state_values[index] = static_cast<float>(index % 37) * 0.01f - 0.15f;
    }
    for (std::size_t index = 0; index < input_values.size(); ++index) {
        input_values[index] = static_cast<float>(index % 29) * 0.02f - 0.2f;
    }
    for (std::size_t index = 0; index < weight_values.size(); ++index) {
        weight_values[index] = static_cast<float>(index % 17) * 0.003f - 0.02f;
    }
    for (std::size_t index = 0; index < bias_values.size(); ++index) {
        bias_values[index] = static_cast<float>(index % 11) * 0.001f - 0.004f;
    }
    const auto input = array(
        input_values.begin(), Shape{1, 1, channels}, mlx::core::float32);
    const auto weight = array(
        weight_values.begin(), Shape{channels, 1, kernel}, mlx::core::float32);
    const auto bias = array(
        bias_values.begin(), Shape{channels}, mlx::core::float32);
    auto state = array(
        state_values.begin(), Shape{1, state_length, channels},
        mlx::core::float32);
    for (int step = 0; step < 2; ++step) {
        const auto combined = concatenate({state, input}, 1);
        auto reference = zeros(Shape{1, 1, channels}, mlx::core::float32);
        for (int tap = 0; tap < kernel; ++tap) {
            const auto source = slice(
                combined, Shape{0, tap, 0}, Shape{1, tap + 1, channels});
            const auto coefficient = reshape(
                slice(weight, Shape{0, 0, tap}, Shape{channels, 1, tap + 1}),
                Shape{1, 1, channels});
            reference = reference + source * coefficient;
        }
        reference = reference + reshape(bias, Shape{1, 1, channels});
        reference = reference * sigmoid(reference);
        auto actual = mfq::metal::cached_depthwise_conv_silu(
            input, weight, state, 1, bias);
        compare(actual.output, reference, "cached depthwise output");
        compare(actual.state,
                slice(combined, Shape{0, 1, 0}, Shape{1, 1 + state_length, channels}),
                "cached depthwise state");
        state = actual.state;
    }
    std::cout << "cached_conv_check ok" << '\n';
    return 0;
}

// Deterministic patch values shared with the Python oracle: k / 64 is
// exactly representable, so both sides feed the tower identical pixels.
float vision_pixel(std::int64_t index) {
    return static_cast<float>((index * 37 + 11) % 101) / 64.0f;
}

int run_vision_check(MlxNemotronHCausalLm& model) {
    const auto& vision = *model.vision();
    const auto& config = vision.config();
    require(config.image_token_id >= 1 && model.config().vocab_size > 3,
            "Nemotron-H vision diagnostic requires image/vocab token ids");
    const std::vector<std::pair<std::int64_t, std::int64_t>> image_grids{
        {4, 6}, {6, 2}};
    std::vector<std::int64_t> token_ids{1, 2};
    std::vector<mfq::GridShape> grids;
    std::int64_t patches = 0;
    for (const auto& [rows, columns] : image_grids) {
        const auto count = rows * columns;
        grids.push_back(mfq::GridShape{
            1, static_cast<std::int32_t>(rows), static_cast<std::int32_t>(columns)});
        patches += count;
        const auto tokens = count / (config.downsample_factor * config.downsample_factor);
        token_ids.insert(token_ids.end(), static_cast<std::size_t>(tokens),
                         config.image_token_id);
        token_ids.push_back(3);
    }
    std::vector<float> pixel_values(
        static_cast<std::size_t>(patches * config.patch_width()));
    for (std::size_t index = 0; index < pixel_values.size(); ++index) {
        pixel_values[index] = vision_pixel(static_cast<std::int64_t>(index));
    }
    mfq::metal::MlxGridMediaInput media;
    media.processor = vision.input_contract();
    media.pixel_values = array(
        pixel_values.begin(),
        Shape{static_cast<int>(patches), static_cast<int>(config.patch_width())},
        mlx::core::float32);
    media.image_grids = std::move(grids);
    media.media_grids = media.image_grids;
    media.media_types.assign(media.image_grids.size(), 1);
    model.reset_cache(1);
    const auto prepared = model.prepare_multimodal_prompt(token_ids, media);
    require(prepared.embeddings.has_value() && prepared.positions.has_value(),
            "Nemotron-H vision prompt was not transformed");
    require(prepared.embeddings->shape(1) ==
                static_cast<int>(token_ids.size()),
            "Nemotron-H vision prompt shape disagrees with the prompt length");
    nlohmann::json result;
    result["architecture"] = "nemotron_h";
    result["image_grids"] = nlohmann::json::array();
    for (const auto& [rows, columns] : image_grids) {
        result["image_grids"].push_back({rows, columns});
    }
    result["patch_width"] = config.patch_width();
    result["hidden_size"] = config.hidden_size;
    result["downsample_factor"] = config.downsample_factor;
    result["position_side"] = config.position_side();
    result["summary_tokens"] = config.summary_tokens();
    result["heads"] = config.num_heads;
    result["depth"] = config.depth;
    result["image_token_id"] = config.image_token_id;
    result["layer_norm_eps"] = config.layer_norm_eps;
    result["projector_norm_eps"] = config.projector_norm_eps;
    result["ids"] = token_ids;
    result["pixels"] = pixel_values;
    std::vector<array> feature_rows;
    std::int64_t feature_offset = 0;
    for (const auto& [rows, columns] : image_grids) {
        const auto count = rows * columns;
        auto image = mlx::core::slice(
            media.pixel_values,
            Shape{static_cast<int>(feature_offset), 0},
            Shape{static_cast<int>(feature_offset + count),
                  static_cast<int>(config.patch_width())});
        feature_rows.push_back(vision.tower().features(image, rows, columns));
        feature_offset += count;
    }
    result["features"] = json_tensor(
        mlx::core::concatenate(std::move(feature_rows), 0));
    {
        auto first = mlx::core::slice(
            media.pixel_values, Shape{0, 0},
            Shape{static_cast<int>(image_grids.front().first *
                                   image_grids.front().second),
                  static_cast<int>(config.patch_width())});
        result["tower_input"] = json_tensor(vision.tower().embedding_input(
            first, image_grids.front().first, image_grids.front().second));
    }
    result["embeddings"] = json_tensor(*prepared.embeddings);
    result["positions"] = json_tensor(
        mlx::core::astype(*prepared.positions, mlx::core::float32));
    bool rejected = false;
    try {
        auto short_prompt = token_ids;
        short_prompt.erase(
            std::remove(short_prompt.begin(), short_prompt.end(), config.image_token_id),
            short_prompt.end());
        (void)model.prepare_multimodal_prompt(short_prompt, media);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    result["placeholder_mismatch_rejected"] = rejected;
    std::cout << "nemotron_h_vision_check " << result.dump() << '\n';
    return 0;
}

struct Arguments {
    std::string model;
    std::string check;
    int context = 0;
};

Arguments parse_arguments(int argc, char** argv) {
    Arguments result;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--model" && index + 1 < argc) {
            result.model = argv[++index];
        } else if (argument == "--ctx-size" && index + 1 < argc) {
            result.context = std::stoi(argv[++index]);
        } else if (argument == "--check-cached-conv") {
            result.check = argument;
        } else if (argument == "--check-nemotron-h" ||
                   argument == "--check-nemotron-h-mtp" ||
                   argument == "--check-nemotron-h-vision") {
            result.check = argument;
        } else {
            throw std::runtime_error("unknown argument: " + argument);
        }
    }
    if (result.check == "--check-cached-conv") return result;
    if (result.model.empty() || result.check.empty()) {
        throw std::runtime_error(
            "usage: mfq-metal-nemotron-test --model PATH [--ctx-size N] "
            "--check-nemotron-h|--check-nemotron-h-mtp");
    }
    return result;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto arguments = parse_arguments(argc, argv);
        const auto stream =
            mlx::core::new_thread_unsafe_stream(mlx::core::Device::gpu);
        mlx::core::set_default_stream(stream);
        if (arguments.check == "--check-cached-conv") {
            return run_cached_conv_check();
        }
        const mfq::metal::MfqContainer container(arguments.model);
        auto model = MlxNemotronHCausalLm::load(container, arguments.context);
        if (arguments.check == "--check-nemotron-h") {
            return run_text_check(model);
        }
        if (arguments.check == "--check-nemotron-h-vision") {
            if (!model.vision()) {
                throw std::runtime_error(
                    "--check-nemotron-h-vision requires a model containing "
                    "RADIO vision weights");
            }
            return run_vision_check(model);
        }
        if (!model.mtp()) {
            throw std::runtime_error(
                "--check-nemotron-h-mtp requires a model containing MTP weights");
        }
        return run_mtp_check(model);
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
