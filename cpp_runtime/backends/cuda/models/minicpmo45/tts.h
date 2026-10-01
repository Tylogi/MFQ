#pragma once

#include "audio.h"

struct MiniCPMO45TtsDecoder {
    CudaExecutionContext* execution = nullptr;
    mfq::models::ModelConfig config;
    RopeCache rope;
    QuantLinear text_embedding;
    QuantLinear code_embedding;
    MiniCPMO45Linear semantic_projector1;
    MiniCPMO45Linear semantic_projector2;
    MiniCPMO45Linear speaker_projector1;
    MiniCPMO45Linear speaker_projector2;
    std::vector<std::unique_ptr<Block>> blocks;
    mfq_tensor_backend::Tensor output_norm;
    mfq_tensor_backend::Tensor code_head;
    int64_t cache_position = 0;

    static mfq::models::ModelConfig make_config() {
        mfq::models::ModelConfig result;
        result.model_type = "minicpmtts";
        result.hidden_size = 768;
        result.intermediate_size = 3072;
        result.num_hidden_layers = 20;
        result.num_attention_heads = 12;
        result.num_key_value_heads = 12;
        result.head_dim = 64;
        result.rotary_dim = 64;
        result.max_position_embeddings = 4096;
        result.rope_base = 10000.0;
        result.rms_norm_eps = 1e-6;
        result.layer_types.assign(20, "full_attention");
        return result;
    }

    static MiniCPMO45TtsDecoder load(
            CudaExecutionContext& execution,
            const mfq::ModelSource& mfq) {
        MiniCPMO45TtsDecoder result;
        result.execution = &execution;
        result.config = make_config();
        result.rope = RopeCache(
            result.config.max_position_embeddings,
            result.config.rotary_dim,
            result.config.rope_base);
        result.text_embedding = load_quant_linear(execution, 
            mfq, "tts.text_embedding.weight");
        result.code_embedding = load_quant_linear(execution, 
            mfq, "tts.code_embedding.0.weight");
        result.semantic_projector1 = MiniCPMO45Linear::load(execution, 
            mfq, "tts.semantic_projector.input");
        result.semantic_projector2 = MiniCPMO45Linear::load(execution, 
            mfq, "tts.semantic_projector.output");
        result.speaker_projector1 = MiniCPMO45Linear::load(execution, 
            mfq, "tts.speaker_projector.input");
        result.speaker_projector2 = MiniCPMO45Linear::load(execution, 
            mfq, "tts.speaker_projector.output");
        result.output_norm = load_dense_native_gpu(execution, 
            mfq, "tts.output_norm.weight");
        auto head_g = load_dense_native_gpu(execution, 
            mfq, "tts.code_output.0.weight_norm.magnitude");
        auto head_v = load_dense_native_gpu(execution, 
            mfq, "tts.code_output.0.weight_norm.direction");
        if (head_g.sizes().vec() != std::vector<int64_t>({6562, 1}) ||
                head_v.sizes().vec() != std::vector<int64_t>({6562, 768}) ||
                result.text_embedding.out() != 152064 ||
                result.text_embedding.neuron_len() != 768 ||
                result.code_embedding.out() != 6562 ||
                result.code_embedding.neuron_len() != 768) {
            throw std::runtime_error(
                "MiniCPM-o TTS tensor shapes disagree with version 4.5");
        }
        auto head_v_f32 = head_v.to(mfq_tensor_backend::kFloat32);
        auto row_norm = mfq_tensor_backend::sqrt(
            mfq_tensor_backend::sum(head_v_f32.square(), -1, true))
            .clamp_min(1e-12);
        result.code_head = (
            head_g.to(mfq_tensor_backend::kFloat32) * head_v_f32 /
            row_norm).to(head_v.scalar_type()).contiguous();
        result.blocks.reserve(20);
        for (int index = 0; index < 20; ++index) {
            auto block = load_transformer_block(
                execution, mfq, result.config, index,
                "full_attention", false, "tts");
            static_cast<FullBlock&>(*block).norm_weight_offset = 0.0;
            block->cuda_device =
                execution.layer_placement.primary_device();
            result.blocks.push_back(std::move(block));
        }
        return result;
    }

    void reset(int64_t batch) {
        cache_position = 0;
        for (auto & block : blocks) block->reset(batch);
    }

    mfq_tensor_backend::Tensor semantic_projection(mfq_tensor_backend::Tensor hidden) const {
        auto projected = semantic_projector1.forward(
            *execution, hidden);
        projected = mfq_tensor_backend::relu(projected);
        projected = semantic_projector2.forward(
            *execution, projected);
        auto norm = mfq_tensor_backend::sqrt(
            mfq_tensor_backend::sum(projected.to(mfq_tensor_backend::kFloat32).square(), -1, true))
            .clamp_min(1e-12);
        return (projected / norm.to(projected.scalar_type())).contiguous();
    }

    mfq_tensor_backend::Tensor speaker_projection(mfq_tensor_backend::Tensor hidden) const {
        return speaker_projector2.forward(
            *execution,
            mfq_tensor_backend::relu(
                speaker_projector1.forward(*execution, hidden)));
    }

    mfq_tensor_backend::Tensor condition(
            mfq_tensor_backend::Tensor text_ids,
            mfq_tensor_backend::Tensor language_hidden) const {
        if (text_ids.dim() == 1) text_ids = text_ids.unsqueeze(0);
        if (language_hidden.dim() == 2) {
            language_hidden = language_hidden.unsqueeze(0);
        }
        if (text_ids.dim() != 2 || language_hidden.dim() != 3 ||
                text_ids.size(0) != 1 || language_hidden.size(0) != 1 ||
                text_ids.size(1) != language_hidden.size(1) ||
                language_hidden.size(2) != 4096) {
            throw std::runtime_error(
                "MiniCPM-o TTS condition expects one aligned text/hidden span");
        }
        auto text = minicpmo45_embedding(text_embedding, text_ids);
        auto semantic = semantic_projection(language_hidden)
            .to(text.scalar_type());
        auto merged = (text + semantic).contiguous();
        auto suffix_ids = mfq_tensor_backend::tensor(
            std::vector<int64_t>{151692, 151687},
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64)
                .device(mfq_tensor_backend::kCUDA)).reshape({1, 2});
        auto suffix = minicpmo45_embedding(
            text_embedding, suffix_ids, text.scalar_type());
        return mfq_tensor_backend::cat({merged, suffix}, 1).contiguous();
    }

    mfq_tensor_backend::Tensor duplex_condition(
            mfq_tensor_backend::Tensor text_ids,
            mfq_tensor_backend::Tensor language_hidden,
            int64_t audio_bos_token) const {
        if (text_ids.dim() == 1) text_ids = text_ids.unsqueeze(0);
        if (language_hidden.dim() == 2) {
            language_hidden = language_hidden.unsqueeze(0);
        }
        if (text_ids.dim() != 2 || language_hidden.dim() != 3 ||
                text_ids.size(0) != 1 || language_hidden.size(0) != 1 ||
                text_ids.size(1) != language_hidden.size(1) ||
                language_hidden.size(2) != 4096 ||
                audio_bos_token < 0 || audio_bos_token >= text_embedding.out()) {
            throw std::runtime_error(
                "MiniCPM-o duplex TTS condition has invalid geometry or audio BOS");
        }
        mfq_tensor_backend::Tensor condition;
        if (text_ids.size(1) > 0) {
            auto text = minicpmo45_embedding(text_embedding, text_ids);
            auto semantic = semantic_projection(language_hidden)
                .to(text.scalar_type());
            condition = (text + semantic).contiguous();
        }
        auto bos_ids = mfq_tensor_backend::tensor(
            std::vector<int64_t>{audio_bos_token},
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64)
                .device(mfq_tensor_backend::kCUDA)).reshape({1, 1});
        auto bos = minicpmo45_embedding(
            text_embedding, bos_ids, mfq_tensor_backend::kBFloat16);
        return condition.defined()
            ? mfq_tensor_backend::cat({condition, bos}, 1).contiguous()
            : bos.contiguous();
    }

    mfq_tensor_backend::Tensor hidden_forward(mfq_tensor_backend::Tensor input_embeddings) {
        if (input_embeddings.dim() != 3 ||
                input_embeddings.size(2) != config.hidden_size) {
            throw std::runtime_error(
                "MiniCPM-o TTS embeddings must be [batch,tokens,768]");
        }
        const int64_t batch = input_embeddings.size(0);
        const int64_t tokens = input_embeddings.size(1);
        if (cache_position == 0) reset(batch);
        if (cache_position + tokens > config.max_position_embeddings) {
            throw std::runtime_error(
                "MiniCPM-o TTS context exceeds 4096 tokens");
        }
        auto positions = mfq_tensor_backend::arange(
            cache_position, cache_position + tokens,
            mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA)
                .dtype(mfq_tensor_backend::kInt64));
        MfqOptional<mfq_tensor_backend::Tensor> sequence_length = mfq_nullopt;
        if (cache_position > 0) {
            sequence_length = mfq_tensor_backend::full(
                {batch}, cache_position + tokens,
                mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA)
                    .dtype(mfq_tensor_backend::kInt64));
        }
        auto hidden = input_embeddings.to(mfq_tensor_backend::kBFloat16).contiguous();
        for (auto & block : blocks) {
            hidden = block->forward(
                *execution, hidden, positions, cache_position,
                sequence_length, rope);
        }
        cache_position += tokens;
        auto flat = hidden.reshape(
            {batch * tokens, config.hidden_size});
        auto normalized_f32 = flat.to(mfq_tensor_backend::kFloat32);
        normalized_f32 = normalized_f32 * mfq_tensor_backend::rsqrt(
            mfq_tensor_backend::mean(normalized_f32.square(), -1, true) +
            config.rms_norm_eps);
        return (
            normalized_f32.to(flat.scalar_type()) *
            output_norm.to(flat.scalar_type()))
            .reshape({batch, tokens, config.hidden_size})
            .contiguous();
    }

    mfq_tensor_backend::Tensor logits(mfq_tensor_backend::Tensor hidden) const {
        return mfq_tensor_backend::matmul(
            hidden.to(code_head.scalar_type()),
            code_head.transpose(0, 1));
    }

    struct ChunkResult {
        mfq_tensor_backend::Tensor codes;
        bool finished = false;
    };

    ChunkResult generate_duplex_chunk(
            mfq_tensor_backend::Tensor condition_embeddings,
            int64_t max_new_tokens,
            int64_t minimum_new_tokens,
            int64_t eos_token,
            double temperature,
            double repetition_penalty) {
        if (condition_embeddings.dim() != 3 ||
                condition_embeddings.size(0) != 1 ||
                condition_embeddings.size(2) != config.hidden_size ||
                max_new_tokens <= 0 || minimum_new_tokens < 0 ||
                eos_token < 0 || eos_token >= code_embedding.out() ||
                temperature <= 0.0 || repetition_penalty <= 0.0) {
            throw std::runtime_error(
                "MiniCPM-o duplex TTS generation limits are invalid");
        }
        std::vector<mfq_tensor_backend::Tensor> sampled;
        sampled.reserve(static_cast<size_t>(max_new_tokens));
        auto current = condition_embeddings.to(mfq_tensor_backend::kBFloat16).contiguous();
        bool finished = false;
        for (int64_t step = 0; step < max_new_tokens; ++step) {
            auto hidden = hidden_forward(current);
            auto step_logits = logits(
                hidden.index({Slice(), -1, Slice()}))
                .to(mfq_tensor_backend::kFloat32) / temperature;
            if (!sampled.empty() && repetition_penalty != 1.0) {
                const size_t begin = sampled.size() > 16
                    ? sampled.size() - 16 : 0;
                auto counts = mfq_tensor_backend::zeros_like(step_logits);
                for (size_t index = begin; index < sampled.size(); ++index) {
                    counts.scatter_add_(
                        1, sampled[index].reshape({1, 1}),
                        mfq_tensor_backend::ones({1, 1}, counts.options()));
                }
                auto alpha = mfq_tensor_backend::pow(
                    mfq_tensor_backend::full_like(counts, repetition_penalty), counts);
                step_logits = mfq_tensor_backend::where(
                    step_logits < 0,
                    step_logits * alpha,
                    step_logits / alpha);
            }
            if (step < minimum_new_tokens) {
                step_logits.index_put_(
                    {Slice(), eos_token},
                    -std::numeric_limits<float>::infinity());
            }
            auto token = mfq_tensor_backend::multinomial(
                mfq_tensor_backend::softmax(step_logits, -1), 1)
                .reshape({1}).to(mfq_tensor_backend::kInt64);
            sampled.push_back(token);
            finished = token.eq(eos_token).all().item<bool>();
            if (finished) break;
            current = minicpmo45_embedding(
                code_embedding, token.unsqueeze(1));
        }
        const int64_t returned = std::max<int64_t>(
            0, static_cast<int64_t>(sampled.size()) - 1);
        mfq_tensor_backend::Tensor codes;
        if (returned > 0) {
            codes = mfq_tensor_backend::stack(sampled, 1)
                .narrow(1, 0, returned)
                .unsqueeze(-1).contiguous();
        } else {
            codes = mfq_tensor_backend::empty(
                {1, 0, 1},
                mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA)
                    .dtype(mfq_tensor_backend::kInt64));
        }
        return {codes, finished};
    }

    mfq_tensor_backend::Tensor generate_official(
            mfq_tensor_backend::Tensor condition_embeddings,
            int64_t steps,
            int64_t eos_token = 6561,
            int64_t minimum_steps = 50,
            double temperature = 0.8,
            double top_p = 0.85,
            int64_t top_k = 25,
            double repetition_penalty = 1.05,
            std::vector<mfq_tensor_backend::Tensor> * logits_trace = nullptr,
            double min_p = 0.0,
            std::mt19937 * evaluator_rng = nullptr,
            int64_t evaluator_min_keep = 0) {
        if (steps <= 0 || minimum_steps < 0 ||
                eos_token < 0 || eos_token >= 6562 ||
                temperature <= 0.0 || top_p <= 0.0 || top_p > 1.0 ||
                min_p < 0.0 || min_p > 1.0 ||
                top_k < 3 || top_k > 6562 || repetition_penalty <= 0.0 ||
                evaluator_min_keep < 0 || evaluator_min_keep > top_k) {
            throw std::runtime_error(
                "MiniCPM-o TTS generation limits are invalid");
        }
        if (condition_embeddings.size(0) != 1) {
            throw std::runtime_error(
                "official MiniCPM-o TTS generation requires batch size one");
        }
        reset(condition_embeddings.size(0));
        std::vector<mfq_tensor_backend::Tensor> generated;
        bool hit_eos = false;
        generated.reserve(static_cast<size_t>(steps));
        auto current = condition_embeddings;
        for (int64_t step = 0; step < steps; ++step) {
            auto hidden = hidden_forward(current);
            auto raw_step_logits = logits(
                hidden.index({Slice(), -1, Slice()}))
                .to(mfq_tensor_backend::kFloat32);
            if (logits_trace != nullptr) {
                logits_trace->push_back(raw_step_logits.clone());
            }
            auto step_logits = evaluator_rng != nullptr
                ? raw_step_logits / temperature
                : raw_step_logits;
            if (!generated.empty()) {
                if (repetition_penalty != 1.0) {
                    auto counts = mfq_tensor_backend::zeros_like(step_logits);
                    const size_t begin = generated.size() > 16
                        ? generated.size() - 16 : 0;
                    for (size_t index = begin; index < generated.size(); ++index) {
                        counts.scatter_add_(
                            1, generated[index].reshape({1, 1}),
                            mfq_tensor_backend::ones(
                                {1, 1}, counts.options()));
                    }
                    auto alpha = mfq_tensor_backend::pow(
                        mfq_tensor_backend::full_like(counts, repetition_penalty), counts);
                    step_logits = mfq_tensor_backend::where(
                        step_logits < 0,
                        step_logits * alpha,
                        step_logits / alpha);
                }
            }
            if (step < minimum_steps) {
                step_logits.index_put_({Slice(), eos_token},
                    -std::numeric_limits<float>::infinity());
            }
            if (evaluator_rng != nullptr) {
                auto cpu_logits = step_logits
                    .reshape({-1}).to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kFloat32)
                    .contiguous();
                const auto * values = cpu_logits.data_ptr<float>();
                const int64_t vocabulary = cpu_logits.numel();
                float maximum = values[0];
                for (int64_t index = 1; index < vocabulary; ++index) {
                    maximum = std::max(maximum, values[index]);
                }
                std::vector<std::pair<float, int64_t>> probabilities;
                probabilities.reserve(static_cast<size_t>(vocabulary));
                float sum = 0.0F;
                for (int64_t index = 0; index < vocabulary; ++index) {
                    const float probability = std::exp(values[index] - maximum);
                    probabilities.emplace_back(probability, index);
                    sum += probability;
                }
                for (auto & probability : probabilities) {
                    probability.first /= sum;
                }
                std::sort(probabilities.begin(), probabilities.end(),
                    [](const auto & left, const auto & right) {
                        return left.first > right.first;
                    });
                std::vector<float> kept_probabilities;
                std::vector<int64_t> kept_indices;
                kept_probabilities.reserve(static_cast<size_t>(top_k));
                kept_indices.reserve(static_cast<size_t>(top_k));
                float cumulative = 0.0F;
                for (const auto & probability : probabilities) {
                    if (static_cast<int64_t>(kept_probabilities.size()) <
                            evaluator_min_keep ||
                            (cumulative < static_cast<float>(top_p) &&
                             static_cast<int64_t>(kept_probabilities.size()) <
                                 top_k)) {
                        cumulative += probability.first;
                        kept_probabilities.push_back(probability.first);
                        kept_indices.push_back(probability.second);
                    } else {
                        break;
                    }
                }
                float kept_sum = std::accumulate(
                    kept_probabilities.begin(), kept_probabilities.end(), 0.0F);
                for (auto & probability : kept_probabilities) {
                    probability /= kept_sum;
                }
                std::uniform_real_distribution<float> distribution(0.0F, 1.0F);
                const float sample = distribution(*evaluator_rng);
                float selected_sum = 0.0F;
                int64_t selected = kept_indices.back();
                for (size_t index = 0; index < kept_probabilities.size(); ++index) {
                    selected_sum += kept_probabilities[index];
                    if (sample <= selected_sum) {
                        selected = kept_indices[index];
                        break;
                    }
                }
                auto token = mfq_tensor_backend::tensor(
                    std::vector<int64_t>{selected},
                    mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA)
                        .dtype(mfq_tensor_backend::kInt64));
                generated.push_back(token);
                if (token.eq(eos_token).all().item<bool>()) {
                    hit_eos = true;
                    break;
                }
                current = minicpmo45_embedding(
                    code_embedding, token.unsqueeze(1));
                continue;
            }
            if (top_k < step_logits.size(1)) {
                auto top_values = std::get<0>(
                    mfq_tensor_backend::topk(
                        step_logits, top_k, -1, true, true));
                auto threshold = top_values.select(1, top_k - 1)
                    .unsqueeze(1);
                step_logits = step_logits.masked_fill(
                    step_logits < threshold,
                    -std::numeric_limits<float>::infinity());
            }
            if (top_p < 1.0) {
                auto sorted = mfq_tensor_backend::sort(
                    step_logits, -1, true);
                auto sorted_logits = std::get<0>(sorted);
                auto sorted_indices = std::get<1>(sorted);
                auto cumulative = mfq_tensor_backend::cumsum(
                    mfq_tensor_backend::softmax(sorted_logits, -1), -1);
                auto keep = cumulative <= top_p;
                keep.narrow(1, 1, keep.size(1) - 1).copy_(
                    keep.narrow(1, 0, keep.size(1) - 1).clone());
                keep.index_put_({Slice(), 0}, true);
                sorted_logits = sorted_logits.masked_fill(
                    keep.logical_not(),
                    -std::numeric_limits<float>::infinity());
                step_logits = mfq_tensor_backend::full_like(
                    step_logits,
                    -std::numeric_limits<float>::infinity());
                step_logits.scatter_(1, sorted_indices, sorted_logits);
            }
            if (min_p > 0.0) {
                auto max_logits = std::get<0>(
                    mfq_tensor_backend::max(step_logits, -1, true));
                auto remove =
                    step_logits < max_logits + std::log(min_p);
                step_logits = step_logits.masked_fill(
                    remove,
                    -std::numeric_limits<float>::infinity());
            }
            step_logits = step_logits / temperature;
            auto token = mfq_tensor_backend::multinomial(
                mfq_tensor_backend::softmax(step_logits, -1), 1)
                .reshape({1}).to(mfq_tensor_backend::kInt64);
            generated.push_back(token);
            if (token.eq(eos_token).all().item<bool>()) {
                hit_eos = true;
                break;
            }
            current = minicpmo45_embedding(
                code_embedding, token.unsqueeze(1));
        }
        auto sampled = mfq_tensor_backend::stack(generated, 1);
        const int64_t returned = sampled.size(1) - (hit_eos ? 1 : 0);
        return sampled.narrow(1, 0, returned)
            .unsqueeze(-1).contiguous();
    }
};

