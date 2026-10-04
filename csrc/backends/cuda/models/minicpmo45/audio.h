#pragma once

#include "models/minicpmo45/causal_lm.h"
#include "vision.h"

struct MiniCPMO45WhisperAttention {
    MiniCPMO45Linear q;
    MiniCPMO45Linear k;
    MiniCPMO45Linear v;
    MiniCPMO45Linear output;
    int64_t heads = 16;
    int64_t head_dim = 64;

    static MiniCPMO45WhisperAttention load(CudaExecutionContext &execution,
                                           const mfq::ModelSource &mfq, const std::string &prefix) {
        MiniCPMO45WhisperAttention result;
        result.q = MiniCPMO45Linear::load(execution, mfq, prefix + ".query");
        result.k = MiniCPMO45Linear::load(execution, mfq, prefix + ".key", false);
        result.v = MiniCPMO45Linear::load(execution, mfq, prefix + ".value");
        result.output = MiniCPMO45Linear::load(execution, mfq, prefix + ".output");
        return result;
    }

    mfq_tensor_backend::Tensor forward(CudaExecutionContext &execution,
                                       mfq_tensor_backend::Tensor input,
                                       MfqOptional<mfq_tensor_backend::Tensor> mask,
                                       mfq_tensor_backend::Tensor *key_cache = nullptr,
                                       mfq_tensor_backend::Tensor *value_cache = nullptr) const {
        const int64_t batch = input.size(0);
        const int64_t tokens = input.size(1);
        auto reshape = [&](mfq_tensor_backend::Tensor value) {
            return value.reshape({batch, tokens, heads, head_dim}).transpose(1, 2).contiguous();
        };
        using Tensor = mfq_tensor_backend::Tensor;
        return mfq::models::minicpmo45::encoder_attention(
            input, key_cache && value_cache,
            [&](const Tensor &x) { return reshape(q.forward(execution, x)); },
            [&](const Tensor &x) { return reshape(k.forward(execution, x)); },
            [&](const Tensor &x) { return reshape(v.forward(execution, x)); },
            [&](Tensor &key, Tensor &value) {
                if (key_cache->defined()) {
                    key = mfq_tensor_backend::cat({*key_cache, key}, 2).contiguous();
                    value = mfq_tensor_backend::cat({*value_cache, value}, 2).contiguous();
                }
                *key_cache = key;
                *value_cache = value;
            },
            [&](Tensor query, Tensor key, Tensor value) {
                std::optional<mfq_tensor_backend::Tensor> attention_mask = std::nullopt;
                if (mask.has_value()) {
                    attention_mask = mask.value().to(query.scalar_type());
                }
                return mfq_scaled_dot_product_attention(query, key, value, attention_mask, 0.0,
                                                        false, std::nullopt, false)
                    .contiguous();
            },
            [&](Tensor attended) {
                attended = attended.transpose(1, 2).reshape({batch, tokens, heads * head_dim});
                return output.forward(execution, attended);
            });
    }
};

struct MiniCPMO45WhisperLayer {
    MiniCPMO45WhisperAttention attention;
    mfq_tensor_backend::Tensor attention_norm_weight;
    mfq_tensor_backend::Tensor attention_norm_bias;
    mfq_tensor_backend::Tensor final_norm_weight;
    mfq_tensor_backend::Tensor final_norm_bias;
    MiniCPMO45Linear fc1;
    MiniCPMO45Linear fc2;
    mfq_tensor_backend::Tensor key_cache;
    mfq_tensor_backend::Tensor value_cache;

    static MiniCPMO45WhisperLayer load(CudaExecutionContext &execution, const mfq::ModelSource &mfq,
                                       int index) {
        const std::string prefix = "audio.block." + std::to_string(index);
        MiniCPMO45WhisperLayer result;
        result.attention = MiniCPMO45WhisperAttention::load(execution, mfq, prefix + ".attention");
        result.attention_norm_weight =
            load_dense_native_gpu(execution, mfq, prefix + ".attention.norm.weight");
        result.attention_norm_bias =
            load_dense_native_gpu(execution, mfq, prefix + ".attention.norm.bias");
        result.final_norm_weight =
            load_dense_native_gpu(execution, mfq, prefix + ".mlp.norm.weight");
        result.final_norm_bias = load_dense_native_gpu(execution, mfq, prefix + ".mlp.norm.bias");
        result.fc1 = MiniCPMO45Linear::load(execution, mfq, prefix + ".mlp.up");
        result.fc2 = MiniCPMO45Linear::load(execution, mfq, prefix + ".mlp.down");
        return result;
    }

    void reset() {
        key_cache = mfq_tensor_backend::Tensor();
        value_cache = mfq_tensor_backend::Tensor();
    }

    mfq_tensor_backend::Tensor forward(CudaExecutionContext &execution,
                                       mfq_tensor_backend::Tensor input,
                                       MfqOptional<mfq_tensor_backend::Tensor> mask,
                                       bool use_cache) {
        using Tensor = mfq_tensor_backend::Tensor;
        return mfq::models::pre_norm_layer(
            input,
            [&](const Tensor &x, int stage) {
                return minicpmo45_layer_norm(
                    x, stage == 0 ? attention_norm_weight : final_norm_weight,
                    stage == 0 ? attention_norm_bias : final_norm_bias, 1e-5);
            },
            [&](Tensor x) {
                return attention.forward(execution, x, mask, use_cache ? &key_cache : nullptr,
                                         use_cache ? &value_cache : nullptr);
            },
            [&](Tensor x) {
                return mfq::models::mlp(
                    x, [&](Tensor x) { return fc1.forward(execution, x); },
                    [](Tensor x) { return mfq_tensor_backend::gelu(x); },
                    [&](Tensor x) { return fc2.forward(execution, x); });
            },
            [](Tensor x, Tensor branch) { return (x + branch).contiguous(); });
    }
};

struct MiniCPMO45AudioEncoder {
    mfq_tensor_backend::Tensor conv1_weight;
    mfq_tensor_backend::Tensor conv1_bias;
    mfq_tensor_backend::Tensor conv2_weight;
    mfq_tensor_backend::Tensor conv2_bias;
    mfq_tensor_backend::Tensor position_embedding;
    std::vector<MiniCPMO45WhisperLayer> layers;
    mfq_tensor_backend::Tensor final_norm_weight;
    mfq_tensor_backend::Tensor final_norm_bias;
    MiniCPMO45Linear projector1;
    MiniCPMO45Linear projector2;

    static MiniCPMO45AudioEncoder load(CudaExecutionContext &execution,
                                       const mfq::ModelSource &mfq) {
        MiniCPMO45AudioEncoder result;
        result.conv1_weight =
            load_dense_native_gpu(execution, mfq, "audio.patch_embedding.conv1.weight");
        result.conv1_bias =
            load_dense_native_gpu(execution, mfq, "audio.patch_embedding.conv1.bias");
        result.conv2_weight =
            load_dense_native_gpu(execution, mfq, "audio.patch_embedding.conv2.weight");
        result.conv2_bias =
            load_dense_native_gpu(execution, mfq, "audio.patch_embedding.conv2.bias");
        result.position_embedding =
            load_dense_native_gpu(execution, mfq, "audio.position_embedding.weight");
        result.final_norm_weight =
            load_dense_native_gpu(execution, mfq, "audio.output_norm.weight");
        result.final_norm_bias = load_dense_native_gpu(execution, mfq, "audio.output_norm.bias");
        result.projector1 = MiniCPMO45Linear::load(execution, mfq, "audio.projector.input");
        result.projector2 = MiniCPMO45Linear::load(execution, mfq, "audio.projector.output");
        result.layers.reserve(24);
        for (int index = 0; index < 24; ++index) {
            result.layers.push_back(MiniCPMO45WhisperLayer::load(execution, mfq, index));
        }
        if (result.conv1_weight.sizes().vec() != std::vector<int64_t>({1024, 80, 3}) ||
            result.conv2_weight.sizes().vec() != std::vector<int64_t>({1024, 1024, 3}) ||
            result.position_embedding.sizes().vec() != std::vector<int64_t>({1500, 1024})) {
            throw std::runtime_error("MiniCPM-o Whisper tensor shapes disagree with version 4.5");
        }
        return result;
    }

    void reset() {
        for (auto &layer : layers)
            layer.reset();
    }

    int64_t cache_length() const {
        if (layers.empty() || !layers.front().key_cache.defined())
            return 0;
        return layers.front().key_cache.size(2);
    }

    mfq::StepSequence<mfq_tensor_backend::Tensor> forward_streaming_steps(CudaExecutionContext &execution,
                                                 mfq_tensor_backend::Tensor features,
                                                 int64_t prefix_extra_frames,
                                                 int64_t suffix_extra_frames) {
        if (features.dim() != 3 || features.size(0) != 1 || features.size(1) != 80 ||
            prefix_extra_frames < 0 || suffix_extra_frames < 0) {
            throw std::runtime_error("MiniCPM-o streaming audio expects [1,80,frames] and "
                                     "non-negative extra-frame counts");
        }
        const int64_t conv_tokens_before_crop = (features.size(2) - 1) / 2 + 1;
        if (cache_length() + conv_tokens_before_crop >= position_embedding.size(0)) {
            reset();
        }

        using Tensor = mfq_tensor_backend::Tensor;
        auto sequence = mfq::models::minicpmo45::audio_encoder(
            features, layers,
            [&](Tensor features) {
                return mfq_tensor_backend::conv1d(
                    features.to(conv1_weight.scalar_type()), conv1_weight, conv1_bias,
                    std::vector<int64_t>{1}, std::vector<int64_t>{1}, std::vector<int64_t>{1}, 1);
            },
            [](Tensor x) { return mfq_tensor_backend::gelu(x); },
            [&](Tensor hidden) {
                return mfq_tensor_backend::conv1d(hidden, conv2_weight, conv2_bias,
                                                  std::vector<int64_t>{2}, std::vector<int64_t>{1},
                                                  std::vector<int64_t>{1}, 1);
            },
            [&](Tensor hidden) {
                const int64_t prefix_to_remove =
                    prefix_extra_frames > 0 ? (prefix_extra_frames + 1) / 2 : 0;
                const int64_t suffix_to_remove =
                    suffix_extra_frames > 0 ? (suffix_extra_frames + 1) / 2 : 0;
                if (prefix_to_remove + suffix_to_remove >= hidden.size(2)) {
                    throw std::runtime_error(
                        "MiniCPM-o streaming audio extra context removes every frame");
                }
                if (prefix_to_remove > 0) {
                    hidden = hidden.narrow(2, prefix_to_remove, hidden.size(2) - prefix_to_remove);
                }
                if (suffix_to_remove > 0) {
                    hidden = hidden.narrow(2, 0, hidden.size(2) - suffix_to_remove);
                }
                hidden = hidden.transpose(1, 2).contiguous();

                const int64_t past = cache_length();
                const int64_t tokens = hidden.size(1);
                if (past + tokens > position_embedding.size(0)) {
                    throw std::runtime_error(
                        "MiniCPM-o streaming Whisper position cache exceeds 1500 frames");
                }
                hidden = hidden + position_embedding.narrow(0, past, tokens);
                auto attention_mask =
                    mfq_tensor_backend::zeros({1, 1, tokens, past + tokens}, hidden.options());
                return std::array<Tensor, 2>{hidden, attention_mask};
            },
            [&](auto &layer, Tensor hidden, Tensor mask) {
                return layer.forward(execution, hidden, mask, true);
            },
            [&](Tensor hidden) {
                return minicpmo45_layer_norm(hidden, final_norm_weight, final_norm_bias, 1e-5);
            },
            [&](Tensor hidden) { return projector1.forward(execution, hidden); },
            [](Tensor hidden) { return mfq_tensor_backend::relu(hidden); },
            [&](Tensor hidden) { return projector2.forward(execution, hidden); },
            [&](Tensor hidden) {
                if (hidden.size(1) < 5) {
                    throw std::runtime_error(
                        "MiniCPM-o streaming audio chunk is too short for stride-5 pooling");
                }
                return mfq_tensor_backend::avg_pool1d(
                           hidden.transpose(1, 2), std::vector<int64_t>{5}, std::vector<int64_t>{5})
                    .transpose(1, 2)
                    .contiguous();
            });
        while (auto step = sequence.next()) co_yield std::move(step);
    }

    mfq_tensor_backend::Tensor forward_streaming(CudaExecutionContext &execution,
        mfq_tensor_backend::Tensor features, int64_t prefix, int64_t suffix) {
        return mfq::finish_steps(forward_streaming_steps(execution, std::move(features), prefix, suffix));
    }

    static std::vector<int64_t> pooled_lengths(mfq_tensor_backend::Tensor raw_lengths) {
        auto lengths =
            raw_lengths.to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt64).contiguous();
        const auto *values = lengths.data_ptr<int64_t>();
        std::vector<int64_t> result(static_cast<size_t>(lengths.numel()));
        for (int64_t index = 0; index < lengths.numel(); ++index) {
            result[static_cast<size_t>(index)] =
                mfq::models::minicpmo45::pooled_audio_length(values[index]);
        }
        return result;
    }

    mfq::StepSequence<mfq_tensor_backend::Tensor> forward_steps(
        CudaExecutionContext &execution,
        mfq_tensor_backend::Tensor features,
        mfq_tensor_backend::Tensor raw_lengths,
        bool use_cache = false) {
        if (features.dim() != 3 || features.size(1) != 80 || raw_lengths.dim() != 1 ||
            raw_lengths.size(0) != features.size(0)) {
            throw std::runtime_error("MiniCPM-o audio input geometry is invalid");
        }
        using Tensor = mfq_tensor_backend::Tensor;
        auto encoding = mfq::models::minicpmo45::audio_encoder(
            features, layers,
            [&](Tensor features) {
                return mfq_tensor_backend::conv1d(
                    features.to(conv1_weight.scalar_type()), conv1_weight, conv1_bias,
                    std::vector<int64_t>{1}, std::vector<int64_t>{1}, std::vector<int64_t>{1}, 1);
            },
            [](Tensor x) { return mfq_tensor_backend::gelu(x); },
            [&](Tensor hidden) {
                return mfq_tensor_backend::conv1d(hidden, conv2_weight, conv2_bias,
                                                  std::vector<int64_t>{2}, std::vector<int64_t>{1},
                                                  std::vector<int64_t>{1}, 1);
            },
            [&](Tensor hidden) {
                hidden = hidden.transpose(1, 2).contiguous();
                const int64_t tokens = hidden.size(1);
                int64_t past = 0;
                if (use_cache && !layers.empty() && layers.front().key_cache.defined()) {
                    past = layers.front().key_cache.size(2);
                }
                if (past + tokens > position_embedding.size(0)) {
                    throw std::runtime_error(
                        "MiniCPM-o Whisper position cache exceeds 1500 frames");
                }
                hidden = hidden + position_embedding.narrow(0, past, tokens);

                // Preserve the official graph: its padding comparison uses the raw
                // Whisper feature lengths against the post-convolution sequence.
                auto length_tensor =
                    raw_lengths.to(mfq_tensor_backend::kCUDA, mfq_tensor_backend::kInt64)
                        .contiguous();
                auto key_positions = mfq_tensor_backend::arange(
                    past + tokens, mfq_tensor_backend::TensorOptions()
                                       .device(mfq_tensor_backend::kCUDA)
                                       .dtype(mfq_tensor_backend::kInt64));
                auto valid_keys = key_positions.unsqueeze(0) < (length_tensor + past).unsqueeze(1);
                auto query_positions =
                    mfq_tensor_backend::arange(past, past + tokens,
                                               mfq_tensor_backend::TensorOptions()
                                                   .device(mfq_tensor_backend::kCUDA)
                                                   .dtype(mfq_tensor_backend::kInt64));
                // Official MiniCPM-o 4.5 uses audio_chunk_length=1.0. Whisper emits
                // 50 encoder frames per second, and every query can see its current
                // 50-frame chunk plus all preceding chunks.
                auto chunk_visible =
                    key_positions.unsqueeze(0) <
                    ((query_positions / mfq::models::minicpmo45::audio_chunk_frames + 1) *
                     mfq::models::minicpmo45::audio_chunk_frames)
                        .unsqueeze(1);
                auto visible = valid_keys.unsqueeze(1) & chunk_visible.unsqueeze(0);
                auto attention_mask =
                    mfq_tensor_backend::zeros({features.size(0), 1, tokens, past + tokens},
                                              mfq_tensor_backend::TensorOptions()
                                                  .device(mfq_tensor_backend::kCUDA)
                                                  .dtype(mfq_tensor_backend::kFloat32))
                        .masked_fill(visible.logical_not().unsqueeze(1),
                                     -std::numeric_limits<float>::infinity());
                return std::array<Tensor, 2>{hidden, attention_mask};
            },
            [&](auto &layer, Tensor hidden, Tensor mask) {
                return layer.forward(execution, hidden, mask, use_cache);
            },
            [&](Tensor hidden) {
                return minicpmo45_layer_norm(hidden, final_norm_weight, final_norm_bias, 1e-5);
            },
            [&](Tensor hidden) { return projector1.forward(execution, hidden); },
            [](Tensor hidden) { return mfq_tensor_backend::relu(hidden); },
            [&](Tensor hidden) { return projector2.forward(execution, hidden); },
            [&](Tensor hidden) {
                return mfq_tensor_backend::avg_pool1d(
                           hidden.transpose(1, 2), std::vector<int64_t>{5}, std::vector<int64_t>{5})
                    .transpose(1, 2)
                    .contiguous();
            });
        while (auto step = encoding.next()) co_yield std::move(step);
    }

    mfq_tensor_backend::Tensor forward(CudaExecutionContext& execution,
        mfq_tensor_backend::Tensor features, mfq_tensor_backend::Tensor lengths, bool use_cache = false) {
        return mfq::finish_steps(
            forward_steps(execution, std::move(features), std::move(lengths), use_cache));
    }
};
