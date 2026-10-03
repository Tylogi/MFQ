#pragma once

#include "models/common/moe.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace mfq::models::qwen4_exp {

struct Config {
    int64_t vocab, hidden, layers, maximum, heads, kv_heads, width, rotary, interval, streams, rank;
    int64_t key_heads, value_heads, linear_width, kernel, experts, topk, moe_width, shared_width;
    int64_t index_heads, index_width, pool, budget, ple_kernel, ngram, ngram_heads, shards,
        predictor_layers, eos;
    double eps, rope_base;
    bool interleaved, normalize_routes, silu_gate, tied_embeddings, dedicated_predictor_embeddings;
    std::vector<int64_t> sections, ple_layers;
    std::vector<std::string> layer_types;
    bool linear_layer(int64_t layer, bool predictor = false) const {
        return !predictor && layer_types.at(layer) == "linear_attention";
    }
    bool position_embedding_layer(int64_t layer, bool predictor = false) const {
        return !predictor &&
               std::find(ple_layers.begin(), ple_layers.end(), layer + 1) != ple_layers.end();
    }
    MoeRouting routing() const { return {RouterActivation::softmax, normalize_routes, false, 1.0}; }
    static Config from_json(std::string_view payload);

    template <class Source> static Config from_source(const Source &source) {
        const auto graph = source.resolved_model_graph();
        if (graph.backbone != "qwen4_exp") {
            throw std::runtime_error("Qwen4-Exp loading requires a qwen4_exp model graph");
        }
        return from_json(source.model_config_json());
    }
    static Config parse(const nlohmann::json &outer) {
        const auto &text = outer.contains("text_config") ? outer.at("text_config") : outer;
        const auto outer_type = outer.value("model_type", std::string{});
        if ((outer_type != "qwen4_exp" && outer_type != "qwen4_exp_text") ||
            text.value("model_type", outer_type) != "qwen4_exp_text")
            throw std::runtime_error("expected Qwen4-Exp text config");
        const auto positive = [&](const char *key) {
            const auto &v = text.at(key);
            if (!v.is_number_integer() || v.get<int64_t>() <= 0)
                throw std::runtime_error(std::string("invalid Qwen4 positive integer: ") + key);
            return v.get<int64_t>();
        };
        Config c{};
        c.vocab = positive("vocab_size");
        c.hidden = positive("hidden_size");
        c.layers = positive("num_hidden_layers");
        c.maximum = positive("max_position_embeddings");
        c.heads = positive("num_attention_heads");
        c.kv_heads = positive("num_key_value_heads");
        c.width = positive("head_dim");
        c.interval = positive("full_attention_interval");
        c.streams = positive("hc_count");
        c.rank = positive("hc_lowrank");
        c.layer_types = text.at("layer_types").get<std::vector<std::string>>();
        if (c.layer_types.size() != static_cast<std::size_t>(c.layers))
            throw std::runtime_error("Qwen4 layer schedule must describe every layer");
        for (int64_t i = 0; i < c.layers; ++i)
            if (c.layer_types[i] !=
                ((i + 1) % c.interval == 0 ? "full_attention" : "linear_attention"))
                throw std::runtime_error("Qwen4 attention schedule disagrees with interval");
        auto rope = text.value("rope_parameters", nlohmann::json::object());
        auto partial =
            text.value("partial_rotary_factor", rope.value("partial_rotary_factor", 1.0));
        if (!std::isfinite(partial) || partial <= 0 || partial > 1)
            throw std::runtime_error("invalid Qwen4 rotary fraction");
        auto rotation = c.width * partial, lower = std::floor(rotation);
        c.rotary = int64_t(lower) +
                   (rotation - lower > .5 || (rotation - lower == .5 && int64_t(lower) % 2));
        c.sections = rope.value("mrope_section", std::vector<int64_t>{});
        c.rope_base = rope.value("rope_theta", 1e7);
        c.interleaved = rope.value("mrope_interleaved", false);
        int64_t section_sum = 0;
        for (auto n : c.sections) {
            if (n < 0)
                throw std::runtime_error("negative Qwen4 rotary section");
            section_sum += n;
        }
        if (c.rotary <= 0 || c.rotary % 2 ||
            (!c.sections.empty() && (c.sections.size() != 3 || section_sum * 2 != c.rotary)) ||
            (c.interleaved && c.sections.empty()) || !std::isfinite(c.rope_base) ||
            c.rope_base <= 0)
            throw std::runtime_error("Qwen4 rotary geometry mismatch");
        c.key_heads = positive("linear_num_key_heads");
        c.value_heads = positive("linear_num_value_heads");
        c.linear_width = positive("linear_key_head_dim");
        c.kernel = positive("linear_conv_kernel_dim");
        c.experts = positive("num_experts");
        c.topk = positive("num_experts_per_tok");
        c.moe_width = positive("moe_intermediate_size");
        c.shared_width = positive("shared_expert_intermediate_size");
        c.index_heads = positive("indexer_n_heads");
        c.index_width = positive("indexer_head_dim");
        c.pool = positive("indexer_compress_ratio");
        c.budget = positive("indexer_budget");
        c.ple_kernel = positive("ple_conv_kernel_size");
        c.ngram = positive("ngram_size");
        c.ngram_heads = positive("heads_per_ngram");
        c.shards = positive("split_ngram_parts");
        positive("ngram_vocab_size_base");
        if (c.heads % c.kv_heads || c.value_heads % c.key_heads ||
            c.linear_width != positive("linear_value_head_dim") || c.kernel < 2 || c.streams < 2 ||
            positive("ple_embed_dim") != c.hidden || c.ngram < 2 ||
            c.hidden % ((c.ngram - 1) * c.ngram_heads) || positive("indexer_kv_heads") != 1 ||
            c.budget % c.pool || c.topk > std::min<int64_t>(16, c.experts) || c.experts > 4096 ||
            c.index_width < c.rotary)
            throw std::runtime_error("Qwen4 attention/GR/PLE/routing geometry mismatch");
        c.ple_layers = text.value("ple_layer_ids", std::vector<int64_t>{});
        for (auto i : c.ple_layers)
            if (i < 1 || i > c.layers || c.layer_types[i - 1] != "linear_attention")
                throw std::runtime_error("Qwen4 PLE requires one-indexed linear-attention layers");
        auto gate = text.value("output_gate_type", std::string{});
        if (gate.empty())
            gate = text.value("hidden_act", std::string("silu"));
        if ((gate != "silu" && gate != "sigmoid") ||
            text.value("hidden_act", std::string("silu")) != "silu" ||
            text.value("attention_bias", false))
            throw std::runtime_error("unsupported Qwen4 activation/attention semantics");
        c.silu_gate = gate == "silu";
        c.eps = text.value("rms_norm_eps", 1e-6);
        if (!std::isfinite(c.eps) || c.eps <= 0)
            throw std::runtime_error("invalid Qwen4 norm epsilon");
        c.normalize_routes = text.value("norm_topk_prob", true);
        c.tied_embeddings = text.value("tie_word_embeddings", false);
        c.predictor_layers = text.value("mtp_num_hidden_layers", int64_t(0));
        if (c.predictor_layers < 0 ||
            (text.contains("mtp") && text.at("mtp").is_object() &&
             text.at("mtp").value("num_hidden_layers", c.predictor_layers) != c.predictor_layers))
            throw std::runtime_error("Qwen4 predictor layer counts disagree");
        c.dedicated_predictor_embeddings = text.value("mtp_use_dedicated_embeddings", false);
        auto eos = text.value("eos_token_id", outer.value("eos_token_id", nlohmann::json(nullptr)));
        if (!c.ple_layers.empty() && (eos.is_null() || (eos.is_array() && eos.empty())))
            throw std::runtime_error("Qwen4 PLE requires the configured EOS token ID");
        c.eos = eos.is_array()  ? (eos.empty() ? 0 : eos.at(0).get<int64_t>())
                : eos.is_null() ? 0
                                : eos.get<int64_t>();
        return c;
    }
};

} // namespace mfq::models::qwen4_exp
