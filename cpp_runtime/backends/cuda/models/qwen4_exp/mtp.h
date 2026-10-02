#pragma once

#include "../mtp.h"
#include "layers.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace mfq::cuda::qwen4_exp {

// Qwen4-Exp predictor shares the target embedding and output projection.
struct Qwen4ExpMtp final : MtpModule {
    using Tensor = mfq_tensor_backend::Tensor;

    mfq::models::qwen4_exp::Config config;
    Tensor embedding_norm, hidden_norm;
    Linear embedding_fusion, hidden_fusion;
    std::unique_ptr<Gr> final_mixer;
    std::vector<std::unique_ptr<Qwen4Block>> layers;
    std::vector<Tensor> positions;
    std::vector<int64_t> lengths;
    CudaExecutionContext *execution = nullptr;
    int64_t batch = 0;

    static std::optional<Qwen4ExpMtp> load_if_present(CudaExecutionContext &execution,
                                                      const mfq::ModelSource &file,
                                                      const mfq::models::qwen4_exp::Config &main) {
        const bool any = std::any_of(file.tensors().begin(), file.tensors().end(),
                                     [](const mfq::TensorMetadata &tensor) {
                                         return tensor.name.rfind("predictor.", 0) == 0;
                                     });
        const auto count = main.predictor_layers;
        if (!has_tensor(file, "predictor.embedding_norm.weight") || count <= 0) {
            MFQ_RUNTIME_CHECK(
                !any, "Qwen4-Exp model source contains an incomplete or undeclared MTP head");
            return std::nullopt;
        }
        Qwen4ExpMtp result;
        result.execution = &execution;
        result.config = main;
        result.embedding_norm =
            dense(execution, file, "predictor.embedding_norm.weight").to(tb::kFloat32);
        result.hidden_norm =
            dense(execution, file, "predictor.hidden_norm.weight").to(tb::kFloat32);
        result.embedding_fusion = linear(execution, file, "predictor.fusion.embedding.weight");
        result.hidden_fusion = linear(execution, file, "predictor.fusion.hidden.weight");
        result.final_mixer =
            std::make_unique<Gr>(execution, file, main, "predictor.mhc.pre", false);
        for (int64_t i = 0; i < count; ++i)
            result.layers.push_back(
                std::make_unique<Qwen4Block>(execution, file, main, int(i), "predictor"));
        MFQ_RUNTIME_CHECK(result.embedding_norm.dim() == 1 &&
                              result.embedding_norm.numel() == main.hidden &&
                              result.hidden_norm.dim() == 1 &&
                              result.hidden_norm.numel() == main.hidden * main.streams,
                          "Qwen4-Exp MTP normalization width disagrees with backbone");
        result.positions.resize(count);
        result.lengths.resize(count, 0);
        return result;
    }

    void reset(int64_t next_batch = 1) override {
        MFQ_RUNTIME_CHECK(next_batch > 0, "Qwen4-Exp MTP batch must be positive");
        for (auto &layer : layers)
            layer->reset(next_batch);
        for (auto &pos : positions)
            pos = Tensor();
        std::fill(lengths.begin(), lengths.end(), 0);
        batch = next_batch;
    }

    std::pair<Tensor, Tensor> evaluate(const MtpTarget &target, const Tensor &hidden,
                                       const Tensor &ids, int64_t depth = 0, bool cache = true,
                                       const Tensor &supplied_positions = {},
                                       const Tensor &supplied_embeddings = {}) {
        namespace tb = mfq_tensor_backend;
        MFQ_RUNTIME_CHECK(ids.dim() == 2 && ids.size(0) > 0 && ids.size(1) > 0 &&
                              hidden.dim() == 3 && hidden.size(0) == ids.size(0) &&
                              hidden.size(1) == ids.size(1) &&
                              hidden.size(2) == hidden_norm.numel(),
                          "Qwen4-Exp MTP input geometry mismatch");
        const auto b = ids.size(0), t = ids.size(1);
        return mfq::models::projected_predictor(
            *this, b, t, depth, cache, [&](int64_t next) { reset(next); },
            [&] {
                auto embeds =
                    supplied_embeddings.defined() ? supplied_embeddings : target.embed(ids);
                MFQ_RUNTIME_CHECK(embeds.sizes().vec() ==
                                      std::vector<int64_t>({b, t, config.hidden}),
                                  "Qwen4-Exp MTP embedding shape mismatch");
                return embeds;
            },
            [&](mfq::models::PredictorCursor cursor) {
                auto current = supplied_positions.defined()
                                   ? supplied_positions
                                   : tb::arange(cursor.start, cursor.start + t,
                                                ids.options().dtype(tb::kInt32));
                if (current.dim() == 1)
                    current = current.reshape({1, 1, t}).expand({3, 1, t}).contiguous();
                else if (current.dim() == 2)
                    current = current.unsqueeze(1);
                if (current.dim() == 3 && current.size(0) == 4)
                    current = current.narrow(0, 1, 3);
                MFQ_RUNTIME_CHECK(current.dim() == 3 && current.size(0) == 3 &&
                                      current.size(-1) == t &&
                                      (current.size(1) == 1 || current.size(1) == b),
                                  "Qwen4 MTP positions require [T]/[3,T]/[3,B,T]");
                current = current.to(tb::kInt32).expand({3, b, t}).contiguous();
                auto full = cache && positions[cursor.layer].defined()
                                ? tb::cat({positions[cursor.layer], current}, -1)
                                : current;
                return std::array<Tensor, 2>{current, full};
            },
            [&](Tensor embeds, const auto &pos) {
                return embedding_fusion(*execution,
                                        rms_norm(embeds, embedding_norm + 1, config.eps));
            },
            [&] {
                return hidden_fusion(*execution,
                                     rms_norm(hidden, hidden_norm + 1, config.eps)
                                         .reshape({b, t, config.streams, config.hidden}));
            },
            [&](Tensor e, Tensor streams) {
                return (streams + e.unsqueeze(-2)).reshape({b, t, config.streams * config.hidden});
            },
            [&](Tensor x, int64_t layer, const auto &pos) {
                auto &block = *layers[layer];
                return mfq::models::qwen4_exp::decoder_layer(
                    std::move(x), false, false, [](const Tensor &) { return Tensor{}; },
                    [](Tensor value, Tensor) { return value; },
                    [&](const Tensor &value) { return block.attention_gr.pre(value); },
                    [](Tensor) -> Tensor { throw std::logic_error("Qwen4 MTP requires QSA"); },
                    [&](Tensor branch) {
                        return block.qsa->forward(*execution, branch, pos[0], pos[1], cache);
                    },
                    [&](Tensor branch, const auto &mix) {
                        return block.attention_gr.post(branch, mix);
                    },
                    [&](const Tensor &value) { return block.ffn_gr.pre(value); },
                    [&](Tensor branch) { return block.ffn(*execution, branch); },
                    [&](Tensor branch, const auto &mix) { return block.ffn_gr.post(branch, mix); });
            },
            [&](const Tensor &multi) { return final_mixer->pre(multi)[0]; },
            [&](int64_t layer, const auto &pos) { positions[layer] = pos[1]; });
    }

    Tensor forward(const MtpTarget &target, Tensor hidden, Tensor ids) override {
        return evaluate(target, hidden, ids).first;
    }
    MtpStep step(const MtpTarget &target, Tensor hidden, Tensor ids) override {
        auto result = evaluate(target, hidden, ids);
        return {std::move(result.first), std::move(result.second)};
    }
    int64_t cache_position() const noexcept override {
        return lengths.empty() ? 0 : lengths.front();
    }
    void trim_cache_to(int64_t position) override {
        MFQ_RUNTIME_CHECK(position >= 0 && position <= cache_position(),
                          "Qwen4-Exp MTP cache trim position is invalid");
        constexpr size_t index = 0;
        if (index < layers.size() && layers[index]->qsa)
            layers[index]->qsa->truncate(position);
        if (positions[index].defined())
            positions[index] = positions[index].narrow(-1, 0, position);
        lengths[index] = position;
    }
    bool teacher_forced_prompt_prime() const noexcept override { return true; }
    bool target_bootstrap_decode() const noexcept override { return true; }
    bool preserve_output_dtype() const noexcept override { return true; }
};

} // namespace mfq::cuda::qwen4_exp
