#pragma once

#include "tts.h"

inline auto minicpmo45_parse_bounds(mfq_tensor_backend::Tensor bounds, const char *label) {
    if (!bounds.defined() || bounds.numel() == 0)
        return std::vector<mfq::models::minicpmo45::MediaBound>{};
    if (bounds.dim() != 2 || bounds.size(1) != 4)
        throw std::runtime_error(std::string("MiniCPM-o ") + label +
                                 " bounds must have shape [count,4]");
    bounds = bounds.to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt64).contiguous();
    return mfq::models::minicpmo45::media_bounds(
        std::span<const int64_t>(bounds.data_ptr<int64_t>(), bounds.numel()));
}

struct MiniCPMO45Runtime {
    mfq::cuda::MiniCPMO45CausalLm language;
    MiniCPMO45VisionEncoder vision;
    MiniCPMO45Resampler resampler;
    MiniCPMO45AudioEncoder audio;
    MiniCPMO45TtsDecoder tts;

    static MiniCPMO45Runtime load(CudaExecutionContext &execution, const std::string &model_path,
                                  const std::string &config_path, int64_t context_size) {
        return load_with_language(mfq::cuda::load_causal_lm<mfq::cuda::MiniCPMO45CausalLm>(
            execution, model_path, config_path, context_size));
    }

    static MiniCPMO45Runtime load_with_language(mfq::cuda::MiniCPMO45CausalLm language) {
        MiniCPMO45Runtime result;
        result.language = std::move(language);
        auto &execution = *result.language.execution;
        auto &placement = execution.layer_placement;
        placement.load_device = placement.primary_device();
        MfqCudaGuard guard(placement.primary_device());
        const auto &mfq = *result.language.source;
        result.vision = MiniCPMO45VisionEncoder::load(execution, mfq);
        result.resampler = MiniCPMO45Resampler::load(execution, mfq);
        result.audio = MiniCPMO45AudioEncoder::load(execution, mfq);
        result.tts = MiniCPMO45TtsDecoder::load(*result.language.execution, mfq);
        return result;
    }

    struct EncodeOps {
        using Tensor = mfq_tensor_backend::Tensor;
        MiniCPMO45Runtime &runtime;
        static int64_t rank(const Tensor &t) { return t.dim(); }
        static int64_t size(const Tensor &t, int axis) { return t.size(axis); }
        static bool defined(const Tensor &t) { return t.defined(); }
        static Tensor batch_ids(Tensor t) { return t.unsqueeze(0); }
        static Tensor device_ids(Tensor t) {
            return t.to(mfq_tensor_backend::kCUDA, mfq_tensor_backend::kInt64).contiguous();
        }
        Tensor embed(Tensor ids) { return runtime.language.embed_forward(ids); }
        Tensor vision(Tensor pixels, Tensor mask, Tensor sizes) {
            return runtime.vision.forward(*runtime.language.execution,
                                          pixels.to(mfq_tensor_backend::kCUDA), mask, sizes);
        }
        Tensor resample(Tensor hidden, Tensor sizes) {
            return runtime.resampler.forward(*runtime.language.execution, hidden, sizes);
        }
        void reset_audio() { runtime.audio.reset(); }
        Tensor audio(Tensor features, Tensor lengths) {
            return runtime.audio.forward(*runtime.language.execution,
                                         features.to(mfq_tensor_backend::kCUDA), lengths, false);
        }
        static auto audio_lengths(Tensor lengths) {
            return MiniCPMO45AudioEncoder::pooled_lengths(lengths);
        }
        static void scatter(Tensor &target, const Tensor &source,
                            const mfq::models::minicpmo45::MediaBound &bound) {
            target.index({bound.batch, Slice(bound.begin, bound.end), Slice()})
                .copy_(source.index({bound.source, Slice(0, bound.end - bound.begin), Slice()})
                           .to(target.scalar_type()));
        }
    };

    auto encode(mfq_tensor_backend::Tensor input_ids, mfq_tensor_backend::Tensor pixels,
                mfq_tensor_backend::Tensor patch_mask, mfq_tensor_backend::Tensor target_sizes,
                mfq_tensor_backend::Tensor image_bounds, mfq_tensor_backend::Tensor audio_features,
                mfq_tensor_backend::Tensor audio_lengths, mfq_tensor_backend::Tensor audio_bounds) {
        EncodeOps ops{*this};
        mfq::models::minicpmo45::MultimodalInputs<mfq_tensor_backend::Tensor> input{
            input_ids,
            pixels,
            patch_mask,
            target_sizes,
            audio_features,
            audio_lengths,
            minicpmo45_parse_bounds(image_bounds, "image"),
            minicpmo45_parse_bounds(audio_bounds, "audio")};
        return mfq::models::minicpmo45::encode(ops, input);
    }

    auto forward(mfq_tensor_backend::Tensor input_ids, mfq_tensor_backend::Tensor position_ids,
                 mfq_tensor_backend::Tensor attention_mask, mfq_tensor_backend::Tensor pixels,
                 mfq_tensor_backend::Tensor patch_mask, mfq_tensor_backend::Tensor target_sizes,
                 mfq_tensor_backend::Tensor image_bounds, mfq_tensor_backend::Tensor audio_features,
                 mfq_tensor_backend::Tensor audio_lengths,
                 mfq_tensor_backend::Tensor audio_bounds) {
        if (input_ids.dim() == 1)
            input_ids = input_ids.unsqueeze(0);
        input_ids =
            input_ids.to(mfq_tensor_backend::kCUDA, mfq_tensor_backend::kInt64).contiguous();
        auto result = encode(input_ids, pixels, patch_mask, target_sizes, image_bounds,
                             audio_features, audio_lengths, audio_bounds);
        return mfq::models::minicpmo45::multimodal_forward(
            language, input_ids, std::move(result),
            position_ids.defined() ? std::optional{position_ids} : std::nullopt,
            attention_mask.defined() ? std::optional{attention_mask} : std::nullopt);
    }
};

struct MiniCPMO45DuplexSpecialIds {
    int64_t unit_start = -1;
    int64_t unit_end = -1;
    int64_t image_start = -1;
    int64_t image_end = -1;
    int64_t slice_start = -1;
    int64_t slice_end = -1;
    int64_t listen = -1;
    int64_t speak = -1;
    int64_t tts_bos = -1;
    int64_t tts_eos = -1;
    int64_t chunk_eos = -1;
    int64_t chunk_tts_eos = -1;
    int64_t turn_eos = -1;
    int64_t tts_pad = -1;
    int64_t audio_bos = -1;

    static MiniCPMO45DuplexSpecialIds from_tensor(mfq_tensor_backend::Tensor value) {
        value = value.to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt64).contiguous().reshape({-1});
        if (value.numel() != 15) {
            throw std::runtime_error(
                "MiniCPM-o duplex special_ids must contain 15 token IDs");
        }
        const auto * ids = value.data_ptr<int64_t>();
        MiniCPMO45DuplexSpecialIds result;
        result.unit_start = ids[0];
        result.unit_end = ids[1];
        result.image_start = ids[2];
        result.image_end = ids[3];
        result.slice_start = ids[4];
        result.slice_end = ids[5];
        result.listen = ids[6];
        result.speak = ids[7];
        result.tts_bos = ids[8];
        result.tts_eos = ids[9];
        result.chunk_eos = ids[10];
        result.chunk_tts_eos = ids[11];
        result.turn_eos = ids[12];
        result.tts_pad = ids[13];
        result.audio_bos = ids[14];
        const int64_t minimum = *std::min_element(ids, ids + 15);
        if (minimum < 0) {
            throw std::runtime_error(
                "MiniCPM-o duplex special token IDs must be non-negative");
        }
        return result;
    }

    bool is_chunk_terminator(int64_t token) const {
        return token == listen || token == chunk_eos ||
            token == chunk_tts_eos;
    }

    bool is_special(int64_t token) const {
        const std::array<int64_t, 15> values = {
            unit_start, unit_end, image_start, image_end,
            slice_start, slice_end, listen, speak, tts_bos,
            tts_eos, chunk_eos, chunk_tts_eos, turn_eos,
            tts_pad, audio_bos};
        return std::find(values.begin(), values.end(), token) != values.end();
    }
};

struct MiniCPMO45DuplexStepResult {
    mfq_tensor_backend::Tensor decision_logits;
    mfq_tensor_backend::Tensor audio_embeddings;
    mfq_tensor_backend::Tensor generated_ids;
    mfq_tensor_backend::Tensor tts_codes;
    bool is_listen = false;
    bool end_of_turn = false;
    bool tts_force_flush = false;
};

struct MiniCPMO45DuplexSession {
    MiniCPMO45Runtime & runtime;
    MiniCPMO45DuplexSpecialIds ids;
    std::vector<int64_t> forbidden_ids;
    std::vector<int64_t> generated_text_ids;
    int64_t audio_chunk_index = 0;
    int64_t tts_text_start_position = 0;
    bool current_turn_ended = true;
    bool greedy = false;
    double temperature = 0.7;
    int64_t top_k = 100;
    double top_p = 0.8;
    double listen_probability_scale = 1.0;
    double repetition_penalty = 1.05;
    int64_t repetition_window = 512;
    double length_penalty = 1.0;
    double tts_temperature = 0.8;
    double tts_repetition_penalty = 1.05;

    MiniCPMO45DuplexSession(
            MiniCPMO45Runtime & runtime_,
            MiniCPMO45DuplexSpecialIds ids_,
            std::vector<int64_t> forbidden_ids_,
            bool greedy_)
        : runtime(runtime_), ids(ids_),
          forbidden_ids(std::move(forbidden_ids_)), greedy(greedy_) {
        if (std::find(forbidden_ids.begin(), forbidden_ids.end(),
                ids.chunk_eos) == forbidden_ids.end()) {
            forbidden_ids.push_back(ids.chunk_eos);
        }
        if (std::find(forbidden_ids.begin(), forbidden_ids.end(),
                ids.tts_pad) == forbidden_ids.end()) {
            forbidden_ids.push_back(ids.tts_pad);
        }
    }

    void prepare(
            mfq_tensor_backend::Tensor system_prefix_ids,
            mfq_tensor_backend::Tensor reference_audio_features = mfq_tensor_backend::Tensor(),
            mfq_tensor_backend::Tensor system_suffix_ids = mfq_tensor_backend::Tensor()) {
        runtime.language.reset(1);
        runtime.audio.reset();
        runtime.tts.reset(1);
        generated_text_ids.clear();
        audio_chunk_index = 0;
        tts_text_start_position = 0;
        current_turn_ended = true;
        if (system_prefix_ids.defined() &&
                system_prefix_ids.numel() > 0) {
            feed_ids(system_prefix_ids);
        }
        if (reference_audio_features.defined()) {
            if (reference_audio_features.dim() != 3 ||
                    reference_audio_features.size(0) != 1 ||
                    reference_audio_features.size(1) != 80 ||
                    reference_audio_features.size(2) < 3) {
                throw std::runtime_error(
                    "MiniCPM-o duplex reference audio expects [1,80,frames]");
            }
            auto raw_lengths = mfq_tensor_backend::tensor(
                std::vector<int64_t>{reference_audio_features.size(2)},
                mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64));
            auto embeddings = runtime.audio.forward(
                *runtime.language.execution,
                reference_audio_features.to(mfq_tensor_backend::kCUDA),
                raw_lengths, false);
            feed_embeddings(embeddings);
            runtime.audio.reset();
        }
        if (system_suffix_ids.defined() &&
                system_suffix_ids.numel() > 0) {
            feed_ids(system_suffix_ids);
        }
    }

    std::pair<mfq_tensor_backend::Tensor, mfq_tensor_backend::Tensor> feed_embeddings(
            mfq_tensor_backend::Tensor embeddings,
            mfq_tensor_backend::Tensor token_ids = mfq_tensor_backend::Tensor()) {
        if (embeddings.dim() == 2) embeddings = embeddings.unsqueeze(0);
        if (embeddings.dim() != 3 || embeddings.size(0) != 1 ||
                embeddings.size(2) != runtime.language.hidden_size()) {
            throw std::runtime_error(
                "MiniCPM-o duplex embeddings must have shape [1,tokens,4096]");
        }
        if (!token_ids.defined()) {
            token_ids = mfq_tensor_backend::zeros(
                {1, embeddings.size(1)},
                mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64)
                    .device(mfq_tensor_backend::kCUDA));
        } else {
            if (token_ids.dim() == 1) token_ids = token_ids.unsqueeze(0);
            if (token_ids.dim() != 2 || token_ids.size(0) != 1 ||
                    token_ids.size(1) != embeddings.size(1)) {
                throw std::runtime_error(
                    "MiniCPM-o duplex token IDs do not match embeddings");
            }
            token_ids = token_ids.to(mfq_tensor_backend::kCUDA, mfq_tensor_backend::kInt64).contiguous();
        }
        auto hidden = runtime.language.hidden_forward_inputs(
            token_ids, embeddings.to(mfq_tensor_backend::kBFloat16).contiguous());
        auto logits = runtime.language.logits_from_hidden(
            hidden.index({Slice(), -1, Slice()}));
        return {logits, hidden};
    }

    std::pair<mfq_tensor_backend::Tensor, mfq_tensor_backend::Tensor> feed_ids(mfq_tensor_backend::Tensor token_ids) {
        if (token_ids.dim() == 0) token_ids = token_ids.reshape({1});
        if (token_ids.dim() == 1) token_ids = token_ids.unsqueeze(0);
        token_ids = token_ids.to(mfq_tensor_backend::kCUDA, mfq_tensor_backend::kInt64).contiguous();
        return feed_embeddings(
            runtime.language.embed_forward(token_ids), token_ids);
    }

    std::pair<mfq_tensor_backend::Tensor, mfq_tensor_backend::Tensor> feed_id(int64_t token) {
        return feed_ids(mfq_tensor_backend::tensor(
            std::vector<int64_t>{token},
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64)
                .device(mfq_tensor_backend::kCUDA)));
    }

    mfq_tensor_backend::Tensor apply_text_sampling_filters(mfq_tensor_backend::Tensor logits) const {
        logits = logits.clone().to(mfq_tensor_backend::kFloat32);
        if (!generated_text_ids.empty() && repetition_penalty != 1.0) {
            const size_t begin = generated_text_ids.size() >
                    static_cast<size_t>(repetition_window)
                ? generated_text_ids.size() -
                    static_cast<size_t>(repetition_window)
                : 0;
            std::vector<int64_t> recent(
                generated_text_ids.begin() +
                    static_cast<std::ptrdiff_t>(begin),
                generated_text_ids.end());
            std::sort(recent.begin(), recent.end());
            recent.erase(std::unique(recent.begin(), recent.end()), recent.end());
            auto recent_tensor = mfq_tensor_backend::tensor(
                recent, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64)
                    .device(logits.device()));
            auto selected = logits.index_select(1, recent_tensor);
            selected = repetition_penalty > 1.0
                ? selected / repetition_penalty
                : selected * (1.0 / repetition_penalty);
            logits.index_copy_(1, recent_tensor, selected);
        }
        if (length_penalty != 1.0) {
            auto selected = logits.index({Slice(), ids.turn_eos});
            selected = mfq_tensor_backend::where(
                selected > 0,
                selected / length_penalty,
                selected * length_penalty);
            logits.index_put_({Slice(), ids.turn_eos}, selected);
        }
        if (listen_probability_scale != 1.0) {
            logits.index_put_(
                {Slice(), ids.listen},
                logits.index({Slice(), ids.listen}) *
                    listen_probability_scale);
        }
        if (!forbidden_ids.empty()) {
            auto forbidden = mfq_tensor_backend::tensor(
                forbidden_ids,
                mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64)
                    .device(logits.device()));
            logits.index_fill_(
                1, forbidden,
                -std::numeric_limits<float>::infinity());
        }
        if (greedy) return logits;
        logits = logits / temperature;
        if (top_k > 0 && top_k < logits.size(1)) {
            auto values = std::get<0>(
                mfq_tensor_backend::topk(
                    logits, top_k, -1, true, true));
            auto threshold = values.select(1, top_k - 1).unsqueeze(1);
            logits = logits.masked_fill(
                logits < threshold,
                -std::numeric_limits<float>::infinity());
        }
        if (top_p < 1.0) {
            auto sorted = mfq_tensor_backend::sort(logits, -1, true);
            auto sorted_logits = std::get<0>(sorted);
            auto sorted_indices = std::get<1>(sorted);
            auto cumulative = mfq_tensor_backend::cumsum(
                mfq_tensor_backend::softmax(sorted_logits, -1), -1);
            auto remove = cumulative > top_p;
            auto shifted = mfq_tensor_backend::zeros_like(remove);
            shifted.narrow(1, 1, remove.size(1) - 1)
                .copy_(remove.narrow(1, 0, remove.size(1) - 1));
            sorted_logits = sorted_logits.masked_fill(
                shifted, -std::numeric_limits<float>::infinity());
            logits = mfq_tensor_backend::full_like(
                logits, -std::numeric_limits<float>::infinity());
            logits.scatter_(1, sorted_indices, sorted_logits);
        }
        return logits;
    }

    int64_t sample_text_token(mfq_tensor_backend::Tensor logits, bool force_listen) const {
        if (force_listen) return ids.listen;
        auto original = logits.to(mfq_tensor_backend::kFloat32);
        auto first = greedy
            ? mfq_tensor_backend::argmax(original, -1)
            : mfq_tensor_backend::multinomial(mfq_tensor_backend::softmax(original, -1), 1)
                .reshape({1});
        const int64_t first_id = first.item<int64_t>();
        if (first_id == ids.chunk_eos) return first_id;
        auto filtered = apply_text_sampling_filters(original);
        auto sampled = greedy
            ? mfq_tensor_backend::argmax(filtered, -1)
            : mfq_tensor_backend::multinomial(mfq_tensor_backend::softmax(filtered, -1), 1)
                .reshape({1});
        return sampled.item<int64_t>();
    }

    MiniCPMO45DuplexStepResult run_step(
            mfq_tensor_backend::Tensor pixels,
            mfq_tensor_backend::Tensor patch_mask,
            mfq_tensor_backend::Tensor target_sizes,
            mfq_tensor_backend::Tensor image_slice_counts,
            mfq_tensor_backend::Tensor audio_features,
            int64_t audio_prefix_extra_frames,
            int64_t audio_suffix_extra_frames,
            mfq_tensor_backend::Tensor text_ids,
            int64_t max_new_speak_tokens,
            bool force_listen,
            bool force_speak = false) {
        if (max_new_speak_tokens < 2) {
            throw std::runtime_error(
                "MiniCPM-o duplex generation requires at least two token slots");
        }
        auto pending = feed_id(ids.unit_start);
        mfq_tensor_backend::Tensor generation_logits;
        bool has_content = false;
        mfq_tensor_backend::Tensor audio_embeddings;

        if (pixels.defined()) {
            if (!patch_mask.defined() || !target_sizes.defined()) {
                throw std::runtime_error(
                    "MiniCPM-o duplex image pixels require patch mask and target sizes");
            }
            auto vision_states = runtime.vision.forward(
                *runtime.language.execution,
                pixels.to(mfq_tensor_backend::kCUDA), patch_mask, target_sizes);
            auto image_embeddings = runtime.resampler.forward(
                *runtime.language.execution, vision_states, target_sizes);
            std::vector<int64_t> counts;
            if (image_slice_counts.defined()) {
                auto count_tensor = image_slice_counts
                    .to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt64).contiguous().reshape({-1});
                const auto * values = count_tensor.data_ptr<int64_t>();
                counts.assign(values, values + count_tensor.numel());
            } else {
                counts.assign(
                    static_cast<size_t>(image_embeddings.size(0)), 1);
            }
            int64_t offset = 0;
            for (const int64_t count : counts) {
                if (count <= 0 || offset + count > image_embeddings.size(0)) {
                    throw std::runtime_error(
                        "MiniCPM-o duplex image slice counts are invalid");
                }
                pending = feed_id(ids.image_start);
                pending = feed_embeddings(image_embeddings.index({offset}));
                pending = feed_id(ids.image_end);
                ++offset;
                for (int64_t slice = 1; slice < count; ++slice) {
                    pending = feed_id(ids.slice_start);
                    pending = feed_embeddings(image_embeddings.index({offset}));
                    pending = feed_id(ids.slice_end);
                    ++offset;
                }
            }
            if (offset != image_embeddings.size(0)) {
                throw std::runtime_error(
                    "MiniCPM-o duplex image slice counts do not cover embeddings");
            }
            generation_logits = pending.first;
            has_content = true;
        }

        if (audio_features.defined()) {
            audio_embeddings = runtime.audio.forward_streaming(
                *runtime.language.execution,
                audio_features.to(mfq_tensor_backend::kCUDA),
                audio_prefix_extra_frames,
                audio_suffix_extra_frames);
            pending = feed_embeddings(audio_embeddings);
            generation_logits = pending.first;
            ++audio_chunk_index;
            has_content = true;
        }

        if (text_ids.defined() && text_ids.numel() > 0) {
            pending = feed_ids(text_ids);
            if (!generation_logits.defined()) {
                generation_logits = pending.first;
            }
            has_content = true;
        }
        if (!has_content) {
            throw std::runtime_error(
                "MiniCPM-o duplex step contains no image, audio, or text input");
        }
        if (pixels.defined() && !audio_features.defined()) {
            ++audio_chunk_index;
        }

        MiniCPMO45DuplexStepResult result;
        result.decision_logits = generation_logits;
        result.audio_embeddings = audio_embeddings;
        std::vector<int64_t> generated;
        std::vector<int64_t> spoken_ids;
        std::vector<mfq_tensor_backend::Tensor> spoken_hidden;
        auto logits = generation_logits;
        bool force_current = force_listen;
        for (int64_t index = 0; index < max_new_speak_tokens; ++index) {
            if (index == max_new_speak_tokens - 1) {
                feed_id(ids.chunk_eos);
                generated.push_back(ids.chunk_eos);
                break;
            }
            const bool forced_decision = force_current;
            int64_t token = sample_text_token(logits, force_current);
            force_current = false;
            if (!forced_decision && token != ids.chunk_eos) {
                generated_text_ids.push_back(token);
            }
            if (!forced_decision && token == ids.listen &&
                    (!current_turn_ended || force_speak)) {
                token = ids.tts_bos;
            }
            generated.push_back(token);
            result.is_listen = token == ids.listen;
            if (ids.is_chunk_terminator(token)) {
                pending = feed_id(token);
                break;
            }

            current_turn_ended = false;
            pending = feed_id(token);
            logits = pending.first;
            result.end_of_turn = token == ids.turn_eos;
            if (result.end_of_turn) current_turn_ended = true;
            if (index != 0) {
                spoken_ids.push_back(token);
                spoken_hidden.push_back(
                    pending.second.index({Slice(), -1, Slice()}));
            }
        }
        feed_id(ids.unit_end);

        result.generated_ids = mfq_tensor_backend::tensor(
            generated,
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64)
                .device(mfq_tensor_backend::kCUDA));
        if (!result.is_listen) {
            mfq_tensor_backend::Tensor text_id_tensor;
            mfq_tensor_backend::Tensor hidden_tensor;
            if (spoken_ids.empty()) {
                text_id_tensor = mfq_tensor_backend::empty(
                    {1, 0}, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64)
                        .device(mfq_tensor_backend::kCUDA));
                hidden_tensor = mfq_tensor_backend::empty(
                    {1, 0, runtime.language.hidden_size()},
                    mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kBFloat16)
                        .device(mfq_tensor_backend::kCUDA));
            } else {
                text_id_tensor = mfq_tensor_backend::tensor(
                    spoken_ids,
                    mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64)
                        .device(mfq_tensor_backend::kCUDA)).unsqueeze(0);
                hidden_tensor = mfq_tensor_backend::stack(spoken_hidden, 1)
                    .to(mfq_tensor_backend::kBFloat16).contiguous();
            }
            auto condition = runtime.tts.duplex_condition(
                text_id_tensor, hidden_tensor, ids.audio_bos);
            const bool first_tts_chunk = tts_text_start_position == 0;
            if (first_tts_chunk) {
                runtime.tts.reset(1);
                result.tts_force_flush = true;
            }
            if (runtime.tts.cache_position != tts_text_start_position) {
                throw std::runtime_error(
                    "MiniCPM-o duplex TTS cache position is inconsistent");
            }
            const int64_t minimum_codes =
                result.end_of_turn || first_tts_chunk ? 0 : 26;
            auto tts_result = runtime.tts.generate_duplex_chunk(
                condition, 26, minimum_codes, 6561,
                tts_temperature, tts_repetition_penalty);
            result.tts_codes = tts_result.codes;
            if (result.end_of_turn) {
                runtime.tts.reset(1);
                tts_text_start_position = 0;
            } else {
                tts_text_start_position +=
                    condition.size(1) + result.tts_codes.size(1);
                if (runtime.tts.cache_position != tts_text_start_position) {
                    throw std::runtime_error(
                        "MiniCPM-o duplex TTS cache did not advance exactly");
                }
            }
        } else {
            result.tts_codes = mfq_tensor_backend::empty(
                {1, 0, 1},
                mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64)
                    .device(mfq_tensor_backend::kCUDA));
        }
        return result;
    }
};

inline mfq_tensor_backend::Tensor minicpmo45_load_tensor(
        const std::string & path,
        bool required) {
    if (!std::filesystem::is_regular_file(path)) {
        if (required) {
            throw std::runtime_error(
                "MiniCPM-o tensor file is missing: " + path);
        }
        return mfq_tensor_backend::Tensor();
    }
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error(
            "failed to open MiniCPM-o tensor file: " + path);
    }
    const auto end = input.tellg();
    if (end <= 0) {
        throw std::runtime_error(
            "MiniCPM-o tensor file is empty: " + path);
    }
    std::vector<char> bytes(static_cast<size_t>(end));
    input.seekg(0, std::ios::beg);
    input.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!input) {
        throw std::runtime_error(
            "failed to read MiniCPM-o tensor file: " + path);
    }
    auto value = mfq_tensor_backend::pickle_load(bytes);
    if (!value.isTensor()) {
        throw std::runtime_error(
            "MiniCPM-o input is not a tensor: " + path);
    }
    return value.toTensor();
}

inline void minicpmo45_save_tensor(
        const mfq_tensor_backend::Tensor & value,
        const std::string & path) {
    if (!value.defined()) return;
    const std::filesystem::path output(path);
    if (output.has_parent_path()) {
        std::filesystem::create_directories(output.parent_path());
    }
    minicpmo45_write_pickle_tensor(value, path);
}

inline int64_t minicpmo45_load_optional_scalar(
        const std::string & path,
        int64_t fallback) {
    auto value = minicpmo45_load_tensor(path, false);
    if (!value.defined()) return fallback;
    if (value.numel() != 1) {
        throw std::runtime_error(
            "MiniCPM-o scalar tensor must contain one value: " + path);
    }
    return value.to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt64).item<int64_t>();
}

inline std::string minicpmo45_duplex_step_prefix(
        const std::string & prefix,
        int64_t step) {
    std::ostringstream stream;
    stream << prefix << ".step" << std::setfill('0')
           << std::setw(4) << step;
    return stream.str();
}

int run_minicpmo45_duplex(
    CudaExecutionContext& execution,
    const std::string& model_path,
    const std::string& config_path,
    const std::string& input_prefix,
    const std::string& output_prefix,
    std::int64_t context_size,
    std::int64_t steps,
    std::int64_t max_speak_tokens,
    bool greedy,
    std::int64_t seed);
int run_minicpmo45_eval_batch(
    CudaExecutionContext& execution,
    const std::string& model_path,
    const std::string& config_path,
    std::int64_t context_size,
    std::int64_t vision_batch_size);
int run_minicpmo45_composite(
    CudaExecutionContext& execution,
    const std::string& model_path,
    const std::string& config_path,
    const std::string& input_prefix,
    const std::string& output_prefix,
    std::int64_t context_size,
    std::int64_t tts_steps);
