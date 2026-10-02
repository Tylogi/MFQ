#pragma once
#include "config.h"
#include "models/common/causal_forward.h"
#include "models/common/causal_model.h"
#include "models/common/gated_mlp.h"
#include "models/common/transformer_layer.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <span>
#include <vector>

namespace mfq::models::minicpmo45 {

struct TtsSampling {
    int64_t eos_token = 6561, minimum_steps = 50, top_k = 25, minimum_keep = 0;
    double temperature = 0.8, top_p = 0.85, repetition_penalty = 1.05, min_p = 0.0;
    void validate(int64_t steps, int64_t vocabulary) const {
        require_model(steps > 0 && minimum_steps >= 0 && eos_token >= 0 && eos_token < vocabulary &&
                          std::isfinite(temperature) && temperature > 0.0 && top_p > 0.0 &&
                          top_p <= 1.0 && min_p >= 0.0 && min_p <= 1.0 && top_k >= 3 &&
                          top_k <= vocabulary && std::isfinite(repetition_penalty) &&
                          repetition_penalty > 0.0 && minimum_keep >= 0 && minimum_keep <= top_k,
                      "MiniCPM-o TTS generation limits are invalid");
    }
};

// Official TTS filters before temperature scaling; the reference evaluator
// scales first and keeps at least minimum_keep tokens. These orders differ
// from the text sampler and must not be interchanged.
template <class Ops>
auto sample_tts(Ops &ops, typename Ops::Tensor logits, const TtsSampling &config,
                std::span<const typename Ops::Tensor> history, int64_t step, bool evaluator) {
    if (evaluator)
        logits = ops.temperature(std::move(logits), config.temperature);
    if (!history.empty() && config.repetition_penalty != 1.0)
        logits =
            ops.penalties(std::move(logits), history.last(std::min<size_t>(16, history.size())),
                          config.repetition_penalty);
    if (step < config.minimum_steps)
        ops.mask_eos(logits, config.eos_token);
    if (evaluator)
        return ops.reference(std::move(logits), config.top_k, config.top_p, config.minimum_keep);
    logits = ops.top_k(std::move(logits), config.top_k);
    if (config.top_p < 1.0)
        logits = ops.top_p(std::move(logits), config.top_p);
    if (config.min_p > 0.0)
        logits = ops.min_p(std::move(logits), config.min_p);
    return ops.sample(ops.temperature(std::move(logits), config.temperature));
}

struct PatchPositions {
    std::vector<int64_t> ids;
    bool all_active = true;
};
inline PatchPositions patch_positions(int64_t batch, int64_t patches, int64_t side,
                                      const int64_t *sizes, const bool *mask) {
    require_model(batch > 0 && patches > 0 && side > 0, "invalid patch geometry");
    PatchPositions result{std::vector<int64_t>(batch * patches, 0), true};
    for (int64_t b = 0; b < batch; ++b) {
        const auto height = sizes[2 * b], width = sizes[2 * b + 1];
        require_model(height > 0 && width > 0 && height <= patches / width,
                      "MiniCPM-o target patch size is invalid");
        int64_t active = 0;
        for (int64_t patch = 0; patch < patches; ++patch) {
            if (!mask[b * patches + patch]) {
                result.all_active = false;
                continue;
            }
            require_model(active < height * width,
                          "MiniCPM-o patch mask has too many active entries");
            const auto row = std::min(side - 1, (active / width) * side / height);
            const auto column = std::min(side - 1, (active % width) * side / width);
            result.ids[b * patches + patch] = row * side + column;
            ++active;
        }
        require_model(active == height * width,
                      "MiniCPM-o patch mask active count disagrees with target size");
    }
    return result;
}

inline int64_t pooled_audio_length(int64_t frames) {
    const auto convolved = (frames - 1) / 2 + 1;
    const auto pooled = (convolved - 5) / 5 + 1;
    require_model(frames > 0 && pooled > 0, "MiniCPM-o audio length is too short");
    return pooled;
}
inline constexpr int64_t audio_chunk_frames = 50;

template <class Tensor, class Query, class Key, class Value, class Cache, class Attend,
          class Output>
auto encoder_attention(Tensor hidden, bool cached, Query query, Key key, Value value, Cache cache,
                       Attend attend, Output output) {
    auto q = query(hidden);
    auto k = key(hidden);
    auto v = value(hidden);
    if (cached)
        cache(k, v);
    return output(attend(std::move(q), std::move(k), std::move(v)));
}

template <class Tensor, class Layers, class Conv1, class Gelu, class Conv2, class Prepare,
          class Layer, class Normalize, class Project1, class Relu, class Project2, class Pool>
auto audio_encoder(Tensor features, Layers &layers, Conv1 conv1, Gelu gelu, Conv2 conv2,
                   Prepare prepare, Layer layer, Normalize normalize, Project1 project1, Relu relu,
                   Project2 project2, Pool pool) {
    auto hidden = gelu(conv1(std::move(features)));
    hidden = gelu(conv2(std::move(hidden)));
    auto prepared = prepare(std::move(hidden));
    hidden = layer_stack(std::move(prepared[0]), layers, [&](auto &block, auto value) {
        return layer(block, std::move(value), prepared[1]);
    });
    return pool(mlp(normalize(std::move(hidden)), project1, relu, project2));
}

template <class Tensor, class Layers, class Patch, class Prepare, class Layer, class Normalize>
auto vision_encoder(Tensor pixels, Layers &layers, Patch patch, Prepare prepare, Layer layer,
                    Normalize normalize) {
    auto hidden = prepare(patch(std::move(pixels)));
    return normalize(layer_stack(std::move(hidden), layers, layer));
}

template <class ProjectKv, class NormKv, class NormQ, class Query, class Key, class Value,
          class Attend, class Output, class NormOutput, class Final>
auto resample(ProjectKv project_kv, NormKv norm_kv, NormQ norm_q, Query query, Key key, Value value,
              Attend attend, Output output, NormOutput norm_output, Final final) {
    auto kv = norm_kv(project_kv());
    auto q = query(norm_q());
    auto k = key(kv);
    auto v = value(kv);
    return final(norm_output(output(attend(std::move(q), std::move(k), std::move(v)))));
}

// Text-conditioned TTS has its own embedding and output head, but the decoder
// traversal and logical position transitions are the same on every backend.
template <class Tensor, class Layers, class Reset, class Prepare, class Positions, class Layer,
          class Normalize>
auto tts_forward(Tensor embeddings, int64_t batch, int64_t tokens, int64_t capacity,
                 int64_t &cache_position, Layers &layers, Reset reset, Prepare prepare,
                 Positions positions, Layer layer, Normalize normalize) {
    require_model(batch > 0 && tokens > 0 && cache_position >= 0 && cache_position <= capacity &&
                      tokens <= capacity - cache_position,
                  "MiniCPM-o TTS context exceeds capacity or has invalid geometry");
    if (cache_position == 0)
        reset();
    auto pos = positions();
    auto hidden = layer_stack(prepare(std::move(embeddings)), layers, [&](auto &block, auto value) {
        return layer(block, std::move(value), pos);
    });
    cache_position += tokens;
    return normalize(std::move(hidden));
}

inline void set_standard_metadata(mfq::models::CausalLmMetadata &metadata,
                                  const mfq::models::ModelConfig &config) {
    metadata.vocab_size = config.vocab_size;
    metadata.hidden_size = config.hidden_size;
    metadata.num_hidden_layers = config.num_hidden_layers;
    metadata.num_attention_heads = config.num_attention_heads;
    metadata.num_key_value_heads = config.num_key_value_heads;
    metadata.head_dim = config.head_dim;
    metadata.max_position_embeddings = config.max_position_embeddings;
    metadata.rotary_dim = config.rotary_dim;
    metadata.rope_base = config.rope_base;
    metadata.rms_norm_eps = config.rms_norm_eps;
    metadata.tie_word_embeddings = config.tie_word_embeddings;
    metadata.model_type = config.model_type;
    metadata.layer_types = config.layer_types;
}

template <class Backend> struct CausalLm : models::CausalModelBase<Backend, CausalLm<Backend>> {
    using Tensor = typename Backend::Tensor;
    template <class Graph, class Source>
    void adapter_load_config(std::string_view payload, const Graph &graph, const Source &source) {
        auto &config = this->config;
        auto &metadata = this->metadata;

        config = Config::from_json(payload);
        if (config.model_type.empty())
            config.model_type = graph.architecture;
        config.rotary_dim = config.head_dim;
        config.layer_types.assign(static_cast<std::size_t>(config.num_hidden_layers),
                                  "full_attention");
        if (config.hidden_size != 4096 || config.intermediate_size != 12288 ||
            config.num_hidden_layers != 36 || config.num_attention_heads != 32 ||
            config.num_key_value_heads != 8 || config.head_dim != 128 ||
            config.hidden_act != "silu" || config.attention_bias || config.use_sliding_window) {
            throw std::runtime_error("unsupported MiniCPM-o 4.5 Qwen3 configuration");
        }
        set_standard_metadata(metadata, config);
        metadata.rope_interleaved = true;
        metadata.decode_graph_double_warmup = true;
    }
    void adapter_set_max_position_embeddings(int64_t value) {
        this->config.max_position_embeddings = value;
    }
    bool adapter_uses_decode_sequence_length() const noexcept { return false; }
    bool adapter_pass_cache_positions(bool, bool) const noexcept { return true; }
    bool adapter_pass_attention_mask() const noexcept { return true; }
};

template <class Backend>
struct TtsCausalLm : models::CausalModelBase<Backend, TtsCausalLm<Backend>> {
    using Tensor = typename Backend::Tensor;
    template <class Graph, class Source>
    void adapter_load_config(std::string_view payload, const Graph &graph, const Source &source) {
        auto &config = this->config;
        auto &metadata = this->metadata;

        config = mfq::models::ModelConfig::from_json(payload);
        set_standard_metadata(metadata, config);
        metadata.norm_weight_offset = 1.0;
    }
    void adapter_set_max_position_embeddings(int64_t value) {
        this->config.max_position_embeddings = value;
    }
};

} // namespace mfq::models::minicpmo45
