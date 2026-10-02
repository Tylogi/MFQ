#pragma once

#include "storage/weight_loader.h"

#include "audio.h"
#include <random>
#include "generation_policy.h"
#include "sampling.h"
#include "core/full_block.h"
#include "ops.h"

struct MiniCPMO45TtsSamplingOps {
    using Tensor = mfq_tensor_backend::Tensor;
    std::mt19937 *evaluator_rng = nullptr;
    static Tensor temperature(Tensor scores, double value) { return scores / value; }
    static Tensor penalties(Tensor scores, std::span<const Tensor> history, double penalty) {
        auto counts = mfq_tensor_backend::zeros_like(scores);
        for (const auto &token : history)
            counts.scatter_add_(1, token.reshape({1, 1}),
                                mfq_tensor_backend::ones({1, 1}, counts.options()));
        auto alpha =
            mfq_tensor_backend::pow(mfq_tensor_backend::full_like(counts, penalty), counts);
        return mfq_tensor_backend::where(scores < 0, scores * alpha, scores / alpha);
    }
    static void mask_eos(Tensor &scores, int64_t token) {
        scores.index_put_({Slice(), token}, -std::numeric_limits<float>::infinity());
    }
    Tensor reference(Tensor scores, int64_t top_k, double top_p, int64_t minimum_keep) const {
        auto host = scores.reshape({-1})
                        .to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kFloat32)
                        .contiguous();
        auto token = mfq::engine::sample_top_k_top_p(
            std::span<const float>(host.data_ptr<float>(), host.numel()), top_k, top_p,
            minimum_keep, *evaluator_rng);
        return mfq_tensor_backend::tensor(std::vector<int64_t>{token},
                                          mfq_tensor_backend::TensorOptions()
                                              .device(mfq_tensor_backend::kCUDA)
                                              .dtype(mfq_tensor_backend::kInt64));
    }
    static Tensor top_k(Tensor scores, int64_t count) {
        if (count >= scores.size(1))
            return scores;
        auto values = std::get<0>(mfq_tensor_backend::topk(scores, count, -1, true, true));
        auto threshold = values.select(1, count - 1).unsqueeze(1);
        return scores.masked_fill(scores < threshold, -std::numeric_limits<float>::infinity());
    }
    static Tensor top_p(Tensor scores, double value) {
        auto sorted = mfq_tensor_backend::sort(scores, -1, true);
        auto logits = std::get<0>(sorted);
        auto indices = std::get<1>(sorted);
        auto cumulative = mfq_tensor_backend::cumsum(mfq_tensor_backend::softmax(logits, -1), -1);
        auto keep = cumulative <= value;
        keep.narrow(1, 1, keep.size(1) - 1).copy_(keep.narrow(1, 0, keep.size(1) - 1).clone());
        keep.index_put_({Slice(), 0}, true);
        logits = logits.masked_fill(keep.logical_not(), -std::numeric_limits<float>::infinity());
        scores = mfq_tensor_backend::full_like(scores, -std::numeric_limits<float>::infinity());
        scores.scatter_(1, indices, logits);
        return scores;
    }
    static Tensor min_p(Tensor scores, double value) {
        auto maximum = std::get<0>(mfq_tensor_backend::max(scores, -1, true));
        return scores.masked_fill(scores < maximum + std::log(value),
                                  -std::numeric_limits<float>::infinity());
    }
    static Tensor sample(Tensor scores) {
        return mfq_tensor_backend::multinomial(mfq_tensor_backend::softmax(scores, -1), 1)
            .reshape({1})
            .to(mfq_tensor_backend::kInt64);
    }
};

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


    static MiniCPMO45TtsDecoder load(
            CudaExecutionContext& execution,
            const mfq::ModelSource& mfq) {
        MiniCPMO45TtsDecoder result;
        result.execution = &execution;
        result.config = mfq::models::minicpmo45::tts_decoder_config();
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
            auto block = mfq::cuda::minicpmo45::load_language_block(
                execution, mfq, result.config, index,
                "full_attention", mfq::models::minicpmo45::LanguageComponent::tts, "tts");
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
        return mfq::models::minicpmo45::tts_forward(
            std::move(input_embeddings), batch, tokens, config.max_position_embeddings,
            cache_position, blocks,
            [&] { reset(batch); },
            [&](auto value) { return value.to(mfq_tensor_backend::kBFloat16).contiguous(); },
            [&] {
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
                return std::make_pair(positions, sequence_length);
            },
            [&](auto& block, auto hidden, const auto& positions) {
                return block->forward(*execution, hidden, positions.first, cache_position,
                                      positions.second, rope);
            },
            [&](auto hidden) {
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
            });
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
        mfq_tensor_backend::Tensor condition_embeddings, int64_t steps, int64_t eos_token = 6561,
        int64_t minimum_steps = 50, double temperature = 0.8, double top_p = 0.85,
        int64_t top_k = 25, double repetition_penalty = 1.05,
        std::vector<mfq_tensor_backend::Tensor> *logits_trace = nullptr, double min_p = 0.0,
        std::mt19937 *evaluator_rng = nullptr, int64_t evaluator_min_keep = 0) {
        const mfq::models::minicpmo45::TtsSampling config{eos_token,          minimum_steps, top_k,
                                                          evaluator_min_keep, temperature,   top_p,
                                                          repetition_penalty, min_p};
        config.validate(steps, 6562);
        if (condition_embeddings.dim() != 3 || condition_embeddings.size(0) != 1)
            throw std::runtime_error("official MiniCPM-o TTS generation requires batch size one");
        reset(1);
        using Tensor = mfq_tensor_backend::Tensor;
        std::vector<Tensor> generated;
        generated.reserve(static_cast<size_t>(steps));
        auto current = condition_embeddings;
        MiniCPMO45TtsSamplingOps sampling_ops{evaluator_rng};
        const auto result = mfq::engine::generate_tokens(
            steps,
            [&](int64_t step) {
                auto hidden = hidden_forward(current);
                auto raw_step_logits =
                    logits(hidden.index({Slice(), -1, Slice()})).to(mfq_tensor_backend::kFloat32);
                if (logits_trace)
                    logits_trace->push_back(raw_step_logits.clone());
                return mfq::models::minicpmo45::sample_tts(
                    sampling_ops, std::move(raw_step_logits), config,
                    std::span<const Tensor>(generated), step, evaluator_rng != nullptr);
            },
            [&](const Tensor &token, int64_t) { generated.push_back(token); },
            [&](const Tensor &token) { return token.eq(eos_token).all().item<bool>(); },
            [&](const Tensor &token) {
                current = minicpmo45_embedding(code_embedding, token.unsqueeze(1));
            });
        auto sampled = mfq_tensor_backend::stack(generated, 1);
        return sampled.narrow(1, 0, result.tokens - (result.hit_eos ? 1 : 0))
            .unsqueeze(-1)
            .contiguous();
    }
};
