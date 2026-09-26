#include "model_checks.h"

#include "../../models/registry.h"
#include "../../ops/cuda_quantized_ops.h"
#include "../cuda_execution.h"
#include "../cuda_transformer.h"
#include "mfq/kernels/cuda/deepseek_v4_attention.h"
#include "mfq/kernels/cuda/deepseek_v4_hc.h"
#include "mfq/kernels/cuda/deepseek_v41.h"
#include "mfq_cuda_ops.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

struct KlEvalChunk {
    std::vector<int32_t> tokens;
    std::vector<float> target_log_probs;
    int target_start = 0;
    int score_count = 0;
    std::streamoff row_offset = 0;
};

static void validate_kl_execution_geometry(
        int64_t execution_n_batch,
        int64_t execution_n_ubatch,
        const KlReferenceContract & reference_contract,
        const char * execution_path) {
    if (reference_contract.n_batch == 0) return;
    if (execution_n_batch != reference_contract.n_batch ||
            execution_n_ubatch != reference_contract.n_ubatch) {
        std::ostringstream message;
        message << execution_path
                << " KL execution geometry n_batch=" << execution_n_batch
                << " n_ubatch=" << execution_n_ubatch
                << " does not match reference n_batch="
                << reference_contract.n_batch
                << " n_ubatch=" << reference_contract.n_ubatch;
        throw std::runtime_error(message.str());
    }
}

KlEvaluator parse_kl_evaluator(const std::string & value) {
    if (value == "legacy") return KlEvaluator::Legacy;
    if (value == "optimized") return KlEvaluator::Optimized;
    throw std::runtime_error(
        "--kl-evaluator must be legacy or optimized");
}

KlMmqMode parse_kl_mmq_mode(const std::string & value) {
    if (value == "default") return KlMmqMode::Default;
    if (value == "nint8_1") return KlMmqMode::Nint8One;
    if (value == "fp16") return KlMmqMode::Fp16;
    throw std::runtime_error(
        "--kl-mmq must be default, nint8_1, or fp16");
}

std::vector<KlMmqMode> parse_kl_mmq_sequence(
        const std::string & value) {
    std::vector<KlMmqMode> modes;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ',')) {
        item.erase(
            std::remove_if(
                item.begin(), item.end(),
                [](unsigned char ch) { return std::isspace(ch) != 0; }),
            item.end());
        if (item.empty()) {
            throw std::runtime_error(
                "--kl-mmq-sequence contains an empty item");
        }
        const KlMmqMode mode = parse_kl_mmq_mode(item);
        if (mode == KlMmqMode::Default) {
            throw std::runtime_error(
                "--kl-mmq-sequence accepts only nint8_1 and fp16");
        }
        if (std::find(modes.begin(), modes.end(), mode) != modes.end()) {
            throw std::runtime_error(
                "--kl-mmq-sequence contains a duplicate mode");
        }
        modes.push_back(mode);
    }
    if (modes.empty()) {
        throw std::runtime_error("--kl-mmq-sequence cannot be empty");
    }
    return modes;
}

const char * kl_evaluator_name(KlEvaluator evaluator) {
    return evaluator == KlEvaluator::Legacy ? "legacy" : "optimized";
}

template <typename Model>
static int run_kl_eval(
        Model& model,
        const std::string & path,
        int max_chunks,
        KlEvaluator evaluator,
        int score_override,
        const KlReferenceContract & reference_contract) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open KL logits file: " + path);
    char magic[8];
    int32_t n_vocab = 0;
    f.read(magic, sizeof(magic));
    if (!f) {
        throw std::runtime_error("invalid KL logits header");
    }
    const std::string reference_format(magic, sizeof(magic));
    std::vector<KlEvalChunk> eval_chunks;
    if (reference_format == "_logits_") {
        uint32_t n_ctx = 0;
        int32_t n_chunks = 0;
        f.read(reinterpret_cast<char*>(&n_ctx), sizeof(n_ctx));
        f.read(reinterpret_cast<char*>(&n_vocab), sizeof(n_vocab));
        f.read(reinterpret_cast<char*>(&n_chunks), sizeof(n_chunks));
        if (!f || n_ctx < 2 || n_vocab <= 0 || n_chunks <= 0) {
            throw std::runtime_error("invalid legacy KL logits header");
        }
        std::vector<int32_t> tokens((size_t)n_ctx * n_chunks);
        f.read(
            reinterpret_cast<char*>(tokens.data()),
            (std::streamsize)(tokens.size() * sizeof(int32_t)));
        if (!f) throw std::runtime_error("truncated legacy KL token header");
        const int first = (int)n_ctx / 2;
        const int score_count = (int)n_ctx - 1 - first;
        eval_chunks.resize((size_t)n_chunks);
        const int32_t legacy_bos = tokens[0];
        int legacy_bos_replacements = 0;
        for (int ci = 0; ci < n_chunks; ++ci) {
            auto begin = tokens.begin() + (size_t)ci * n_ctx;
            eval_chunks[(size_t)ci].tokens.assign(begin, begin + n_ctx);
            // The reference perplexity evaluator temporarily replaces every Gemma context
            // chunk's first corpus token with BOS before evaluating it, then
            // serializes the original corpus tokens in this legacy header.
            // Qwen legacy references use the serialized token directly.
            if (Model::is_gemma4 &&
                    eval_chunks[(size_t)ci].tokens[0] != legacy_bos) {
                eval_chunks[(size_t)ci].tokens[0] = legacy_bos;
                ++legacy_bos_replacements;
            }
            eval_chunks[(size_t)ci].target_start = first + 1;
            eval_chunks[(size_t)ci].score_count = score_count;
        }
        std::cout << "cpp_kl_legacy_chunk_bos="
                  << (Model::is_gemma4 ? legacy_bos : -1)
                  << " replacements=" << legacy_bos_replacements
                  << " model_type=" << model.model_type() << "\n";
    } else if (reference_format == "_logit2_" || reference_format == "_logit3_") {
        const bool has_exact_target_log_probs = reference_format == "_logit3_";
        uint32_t vocab = 0;
        uint32_t n_chunks = 0;
        f.read(reinterpret_cast<char*>(&vocab), sizeof(vocab));
        f.read(reinterpret_cast<char*>(&n_chunks), sizeof(n_chunks));
        if (!f || vocab == 0 || vocab > (uint32_t)std::numeric_limits<int32_t>::max() ||
            n_chunks == 0 || n_chunks > (1u << 20)) {
            throw std::runtime_error("invalid trace KL logits header");
        }
        n_vocab = (int32_t)vocab;
        eval_chunks.resize((size_t)n_chunks);
        std::vector<uint32_t> token_counts(n_chunks);
        for (uint32_t ci = 0; ci < n_chunks; ++ci) {
            uint32_t target_start = 0;
            uint32_t score_count = 0;
            f.read(reinterpret_cast<char*>(&token_counts[ci]), sizeof(uint32_t));
            f.read(reinterpret_cast<char*>(&target_start), sizeof(uint32_t));
            f.read(reinterpret_cast<char*>(&score_count), sizeof(uint32_t));
            if (!f || token_counts[ci] < 2 || target_start < 1 ||
                target_start >= token_counts[ci] || score_count < 1 ||
                score_count > token_counts[ci] - target_start) {
                throw std::runtime_error("invalid trace KL chunk descriptor");
            }
            eval_chunks[ci].target_start = (int)target_start;
            eval_chunks[ci].score_count = (int)score_count;
        }
        for (uint32_t ci = 0; ci < n_chunks; ++ci) {
            auto & chunk = eval_chunks[ci];
            chunk.tokens.resize(token_counts[ci]);
            f.read(
                reinterpret_cast<char*>(chunk.tokens.data()),
                (std::streamsize)(chunk.tokens.size() * sizeof(int32_t)));
            if (!f) throw std::runtime_error("truncated trace KL token header");
        }
        if (has_exact_target_log_probs) {
            for (auto & chunk : eval_chunks) {
                chunk.target_log_probs.resize((size_t)chunk.score_count);
                f.read(
                    reinterpret_cast<char*>(chunk.target_log_probs.data()),
                    (std::streamsize)(chunk.target_log_probs.size() * sizeof(float)));
                if (!f) throw std::runtime_error("truncated trace KL target log probabilities");
                for (float value : chunk.target_log_probs) {
                    if (!std::isfinite(value) || value > 0.0f) {
                        throw std::runtime_error("invalid trace KL target log probability");
                    }
                }
            }
        }
    } else {
        throw std::runtime_error("invalid KL logits magic");
    }
    const std::streamoff data_offset = f.tellg();
    const int nv = 2 * ((n_vocab + 1) / 2) + 4;
    std::streamoff row_offset = data_offset;
    for (auto & chunk : eval_chunks) {
        chunk.row_offset = row_offset;
        row_offset += (std::streamoff)chunk.score_count * nv * (std::streamoff)sizeof(uint16_t);
    }
    f.seekg(0, std::ios::end);
    if (!f || f.tellg() < row_offset) {
        throw std::runtime_error("truncated KL logits rows");
    }
    const int available_chunks = (int)eval_chunks.size();
    const int chunks = max_chunks < 0
        ? available_chunks
        : std::min(max_chunks, available_chunks);
    if (chunks <= 0) throw std::runtime_error("KL evaluation requires at least one chunk");
    for (int ci = 0; ci < chunks; ++ci) {
        if ((int64_t)eval_chunks[(size_t)ci].tokens.size() >
                model.max_position_embeddings()) {
            throw std::runtime_error(
                "KL reference exceeds model context capacity");
        }
    }
    const int64_t execution_n_batch =
        (int64_t)eval_chunks[0].tokens.size();
    if (reference_contract.n_batch != 0) {
        for (int ci = 0; ci < chunks; ++ci) {
            const int64_t execution_tokens =
                (int64_t)eval_chunks[(size_t)ci].tokens.size();
            validate_kl_execution_geometry(
                execution_tokens, execution_tokens,
                reference_contract, "single-sequence");
        }
    }
    constexpr int KL_BATCH = 8;
    double kld_sum = 0.0;
    double reverse_kld_sum = 0.0;
    double bf16_ce_sum = 0.0;
    double mfq_ce_sum = 0.0;
    int64_t same_top = 0;
    int64_t count = 0;
    const bool optimized = evaluator == KlEvaluator::Optimized;
    auto started = std::chrono::steady_clock::now();
    std::cout << "cpp_kl_execution evaluator="
              << kl_evaluator_name(evaluator)
              << " graph=single_sequence"
              << " available_chunks=" << available_chunks
              << " selected_chunks=" << chunks
              << " execution_n_seq=1"
              << " execution_n_batch=" << execution_n_batch
              << " execution_n_ubatch=" << execution_n_batch
              << " score_count_override=" << score_override
              << " reference_n_batch=" << reference_contract.n_batch
              << " reference_n_ubatch=" << reference_contract.n_ubatch
              << "\n";

    for (int ci = 0; ci < chunks; ++ci) {
        const auto & chunk = eval_chunks[(size_t)ci];
        const int stored_score_count = chunk.score_count;
        const int score_count = score_override < 0 ? stored_score_count : score_override;
        if (score_count > stored_score_count) {
            throw std::runtime_error(
                "--kl-score-count exceeds the stored chunk score count");
        }
        const int first = chunk.target_start - 1;
        model.reset(1);
        std::vector<int64_t> chunk_tokens(chunk.tokens.size());
        for (size_t j = 0; j < chunk.tokens.size(); ++j) {
            chunk_tokens[j] = chunk.tokens[j];
        }
        auto ids = mfq_tensor_backend::from_blob(chunk_tokens.data(), {1, (int64_t)chunk_tokens.size()},
                                    mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64)).clone().to(mfq_tensor_backend::kCUDA);
        mfq_tensor_backend::Tensor pred;
        if (optimized) {
            auto hidden = model.hidden_forward(ids);
            auto selected_hidden = hidden.index({
                0, Slice(first, first + score_count), Slice()
            }).contiguous();
            pred = model.logits_from_hidden(selected_hidden);
        } else if (score_count < stored_score_count) {
            (void)model.hidden_forward(ids.index({Slice(), Slice(0, first)}));
            auto logits = model.forward(ids.index({Slice(), Slice(first, first + score_count)}));
            pred = logits.index({0, Slice(), Slice()});
        } else {
            auto logits = model.forward(ids);
            pred = logits.index({0, Slice(first, first + score_count), Slice()});
        }
        if (pred.size(1) != n_vocab) throw std::runtime_error("KL vocab size mismatch");

        mfq_tensor_backend::Tensor optimized_kld_sum;
        mfq_tensor_backend::Tensor optimized_reverse_kld_sum;
        mfq_tensor_backend::Tensor optimized_bf16_ce_sum;
        mfq_tensor_backend::Tensor optimized_mfq_ce_sum;
        mfq_tensor_backend::Tensor optimized_same_top;
        if (optimized) {
            const auto cuda = mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA);
            optimized_kld_sum = mfq_tensor_backend::zeros(
                {}, cuda.dtype(mfq_tensor_backend::kFloat64));
            optimized_reverse_kld_sum = mfq_tensor_backend::zeros(
                {}, cuda.dtype(mfq_tensor_backend::kFloat64));
            optimized_bf16_ce_sum = mfq_tensor_backend::zeros(
                {}, cuda.dtype(mfq_tensor_backend::kFloat64));
            optimized_mfq_ce_sum = mfq_tensor_backend::zeros(
                {}, cuda.dtype(mfq_tensor_backend::kFloat64));
            optimized_same_top = mfq_tensor_backend::zeros(
                {}, cuda.dtype(mfq_tensor_backend::kInt64));
        }
        f.seekg(chunk.row_offset);
        for (int s = 0; s < score_count; s += KL_BATCH) {
            const int b = std::min(KL_BATCH, score_count - s);
            std::vector<uint16_t> rows((size_t)b * nv);
            f.read(reinterpret_cast<char*>(rows.data()), (std::streamsize)(rows.size() * sizeof(uint16_t)));
            if (!f) throw std::runtime_error("truncated KL logits data");

            std::vector<float> scales(b), mins(b);
            std::vector<int32_t> codes((size_t)b * n_vocab);
            for (int r = 0; r < b; ++r) {
                const uint16_t* row = rows.data() + (size_t)r * nv;
                std::memcpy(&scales[r], row + 0, sizeof(float));
                std::memcpy(&mins[r], row + 2, sizeof(float));
                for (int v = 0; v < n_vocab; ++v) codes[(size_t)r * n_vocab + v] = row[4 + v];
            }
            auto scale = mfq_tensor_backend::from_blob(scales.data(), {b}, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kFloat32)).clone().to(mfq_tensor_backend::kCUDA);
            auto min_lp = mfq_tensor_backend::from_blob(mins.data(), {b}, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kFloat32)).clone().to(mfq_tensor_backend::kCUDA);
            auto base_codes = mfq_tensor_backend::from_blob(
                codes.data(), {b, n_vocab}, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt32))
                .clone().to(mfq_tensor_backend::kCUDA);
            auto base_logp = base_codes.to(mfq_tensor_backend::kFloat32) * scale.unsqueeze(1) +
                min_lp.unsqueeze(1);
            auto q = pred.index({Slice(s, s + b), Slice()}).to(mfq_tensor_backend::kFloat32);
            auto lse = mfq_tensor_backend::logsumexp(q, -1);
            auto quant_logp = q - lse.unsqueeze(1);
            auto normalized_base_logp =
                base_logp - mfq_tensor_backend::logsumexp(base_logp, -1, true);
            auto p_base = mfq_tensor_backend::exp(base_logp).masked_fill(base_codes.eq(0), 0.0f);
            auto kld = (p_base * (base_logp - q + lse.unsqueeze(1))).sum(-1);
            auto reverse_kld =
                (mfq_tensor_backend::exp(quant_logp) * (quant_logp - normalized_base_logp)).sum(-1);
            if (optimized) {
                optimized_kld_sum.add_(kld.to(mfq_tensor_backend::kFloat64).sum());
                optimized_reverse_kld_sum.add_(
                    reverse_kld.to(mfq_tensor_backend::kFloat64).sum());
            } else {
                kld_sum += kld.to(mfq_tensor_backend::kFloat64).sum().item<double>();
                reverse_kld_sum +=
                    reverse_kld.to(mfq_tensor_backend::kFloat64).sum().item<double>();
            }
            std::vector<int64_t> target_ids(b);
            for (int r = 0; r < b; ++r) {
                target_ids[r] = chunk_tokens[(size_t)chunk.target_start + s + r];
            }
            auto target = mfq_tensor_backend::from_blob(target_ids.data(), {b, 1},
                                            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64))
                              .clone().to(mfq_tensor_backend::kCUDA);
            if (chunk.target_log_probs.empty()) {
                auto batch_bf16_ce =
                    -base_logp.gather(1, target).to(mfq_tensor_backend::kFloat64).sum();
                if (optimized) {
                    optimized_bf16_ce_sum.add_(batch_bf16_ce);
                } else {
                    bf16_ce_sum += batch_bf16_ce.item<double>();
                }
            } else {
                for (int r = 0; r < b; ++r) {
                    bf16_ce_sum -= chunk.target_log_probs[(size_t)s + r];
                }
            }
            auto batch_mfq_ce = (lse.unsqueeze(1) - q.gather(1, target))
                                     .to(mfq_tensor_backend::kFloat64).sum();
            auto batch_same_top =
                q.argmax(-1).eq(base_logp.argmax(-1)).sum();
            if (optimized) {
                optimized_mfq_ce_sum.add_(batch_mfq_ce);
                optimized_same_top.add_(batch_same_top);
            } else {
                mfq_ce_sum += batch_mfq_ce.item<double>();
                same_top += batch_same_top.item<int64_t>();
            }
            count += b;
        }
        if (optimized) {
            kld_sum += optimized_kld_sum.item<double>();
            reverse_kld_sum += optimized_reverse_kld_sum.item<double>();
            if (chunk.target_log_probs.empty()) {
                bf16_ce_sum += optimized_bf16_ce_sum.item<double>();
            }
            mfq_ce_sum += optimized_mfq_ce_sum.item<double>();
            same_top += optimized_same_top.item<int64_t>();
        }
        mfq_cuda_synchronize();
        std::cout << "cpp_kl_chunk=" << (ci + 1)
                  << " mean=" << (kld_sum / (double)count)
                  << " mean_kld_q_ref=" << (reverse_kld_sum / (double)count)
                  << " same_top=" << ((double)same_top / (double)count) << "\n";
    }
    auto ended = std::chrono::steady_clock::now();
    std::cout << "cpp_kl_result chunks=" << chunks
              << " scored_tokens=" << count
              << " sec=" << std::chrono::duration<double>(ended - started).count()
              << " kld=" << (kld_sum / (double)count)
              << " mean_kld_q_ref=" << (reverse_kld_sum / (double)count)
              << " bf16_ce=" << (bf16_ce_sum / (double)count)
              << " mfq_ce=" << (mfq_ce_sum / (double)count)
              << " bf16_ppl=" << std::exp(bf16_ce_sum / (double)count)
              << " mfq_ppl=" << std::exp(mfq_ce_sum / (double)count)
              << " kld_pct_bf16=" << (100.0 * kld_sum / bf16_ce_sum)
              << " same_top=" << ((double)same_top / (double)count)
              << " same_top_count=" << same_top
              << " reference_format="
              << (reference_format == "_logit3_" ? "trace_v3" :
                  (reference_format == "_logit2_" ? "trace_v2" : "legacy"))
              << " execution=" << kl_evaluator_name(evaluator)
              << " graph=single_sequence"
              << " execution_n_seq=1"
              << " execution_n_batch=" << execution_n_batch
              << " execution_n_ubatch=" << execution_n_batch
              << " score_count_override=" << score_override
              << " reference_n_batch=" << reference_contract.n_batch
              << " reference_n_ubatch=" << reference_contract.n_ubatch
              << "\n";
    return 0;
}

struct StreamedKlInput {
    std::string reference_format;
    int32_t n_vocab = 0;
    int nv = 0;
    std::vector<KlEvalChunk> chunks;
};

static StreamedKlInput load_streamed_kl_input(
    const std::string & path,
    int max_chunks) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open KL logits file: " + path);
    char magic[8];
    f.read(magic, sizeof(magic));
    if (!f) throw std::runtime_error("invalid KL logits header");

    StreamedKlInput input;
    input.reference_format.assign(magic, sizeof(magic));
    if (input.reference_format == "_logits_") {
        uint32_t n_ctx = 0;
        int32_t n_chunks = 0;
        f.read(reinterpret_cast<char*>(&n_ctx), sizeof(n_ctx));
        f.read(reinterpret_cast<char*>(&input.n_vocab), sizeof(input.n_vocab));
        f.read(reinterpret_cast<char*>(&n_chunks), sizeof(n_chunks));
        if (!f || n_ctx < 2 || input.n_vocab <= 0 || n_chunks <= 0) {
            throw std::runtime_error("invalid legacy KL logits header");
        }
        std::vector<int32_t> tokens((size_t)n_ctx * n_chunks);
        f.read(
            reinterpret_cast<char*>(tokens.data()),
            (std::streamsize)(tokens.size() * sizeof(int32_t)));
        if (!f) throw std::runtime_error("truncated legacy KL token header");
        const int first = (int)n_ctx / 2;
        const int score_count = (int)n_ctx - 1 - first;
        input.chunks.resize((size_t)n_chunks);
        for (int ci = 0; ci < n_chunks; ++ci) {
            auto begin = tokens.begin() + (size_t)ci * n_ctx;
            input.chunks[(size_t)ci].tokens.assign(begin, begin + n_ctx);
            input.chunks[(size_t)ci].target_start = first + 1;
            input.chunks[(size_t)ci].score_count = score_count;
        }
    } else if (
        input.reference_format == "_logit2_" ||
        input.reference_format == "_logit3_") {
        const bool has_exact_target_log_probs =
            input.reference_format == "_logit3_";
        uint32_t vocab = 0;
        uint32_t n_chunks = 0;
        f.read(reinterpret_cast<char*>(&vocab), sizeof(vocab));
        f.read(reinterpret_cast<char*>(&n_chunks), sizeof(n_chunks));
        if (!f || vocab == 0 ||
            vocab > (uint32_t)std::numeric_limits<int32_t>::max() ||
            n_chunks == 0 || n_chunks > (1u << 20)) {
            throw std::runtime_error("invalid trace KL logits header");
        }
        input.n_vocab = (int32_t)vocab;
        input.chunks.resize((size_t)n_chunks);
        std::vector<uint32_t> token_counts(n_chunks);
        for (uint32_t ci = 0; ci < n_chunks; ++ci) {
            uint32_t target_start = 0;
            uint32_t score_count = 0;
            f.read(reinterpret_cast<char*>(&token_counts[ci]), sizeof(uint32_t));
            f.read(reinterpret_cast<char*>(&target_start), sizeof(uint32_t));
            f.read(reinterpret_cast<char*>(&score_count), sizeof(uint32_t));
            if (!f || token_counts[ci] < 2 || target_start < 1 ||
                target_start >= token_counts[ci] || score_count < 1 ||
                score_count > token_counts[ci] - target_start) {
                throw std::runtime_error("invalid trace KL chunk descriptor");
            }
            input.chunks[ci].target_start = (int)target_start;
            input.chunks[ci].score_count = (int)score_count;
        }
        for (uint32_t ci = 0; ci < n_chunks; ++ci) {
            auto & chunk = input.chunks[ci];
            chunk.tokens.resize(token_counts[ci]);
            f.read(
                reinterpret_cast<char*>(chunk.tokens.data()),
                (std::streamsize)(chunk.tokens.size() * sizeof(int32_t)));
            if (!f) throw std::runtime_error("truncated trace KL token header");
        }
        if (has_exact_target_log_probs) {
            for (auto & chunk : input.chunks) {
                chunk.target_log_probs.resize((size_t)chunk.score_count);
                f.read(
                    reinterpret_cast<char*>(chunk.target_log_probs.data()),
                    (std::streamsize)(
                        chunk.target_log_probs.size() * sizeof(float)));
                if (!f) {
                    throw std::runtime_error(
                        "truncated trace KL target log probabilities");
                }
                for (float value : chunk.target_log_probs) {
                    if (!std::isfinite(value) || value > 0.0f) {
                        throw std::runtime_error(
                            "invalid trace KL target log probability");
                    }
                }
            }
        }
    } else {
        throw std::runtime_error("invalid KL logits magic");
    }

    const std::streamoff data_offset = f.tellg();
    input.nv = 2 * ((input.n_vocab + 1) / 2) + 4;
    std::streamoff row_offset = data_offset;
    for (auto & chunk : input.chunks) {
        chunk.row_offset = row_offset;
        row_offset += (std::streamoff)chunk.score_count * input.nv *
            (std::streamoff)sizeof(uint16_t);
    }
    f.seekg(0, std::ios::end);
    if (!f || f.tellg() < row_offset) {
        throw std::runtime_error("truncated KL logits rows");
    }
    const int available_chunks = (int)input.chunks.size();
    const int chunks = max_chunks < 0
        ? available_chunks
        : std::min(max_chunks, available_chunks);
    if (chunks <= 0) {
        throw std::runtime_error("KL evaluation requires at least one chunk");
    }
    input.chunks.resize((size_t)chunks);
    return input;
}

static mfq_tensor_backend::Tensor streamed_kl_ids(
    const StreamedKlInput & input,
    int begin,
    int count,
    int64_t n_ctx) {
    std::vector<int64_t> token_ids((size_t)count * n_ctx);
    for (int bi = 0; bi < count; ++bi) {
        const auto & source = input.chunks[(size_t)(begin + bi)].tokens;
        for (int64_t ti = 0; ti < n_ctx; ++ti) {
            token_ids[(size_t)bi * n_ctx + (size_t)ti] =
                source[(size_t)ti];
        }
    }
    return mfq_tensor_backend::from_blob(
        token_ids.data(), {count, n_ctx},
        mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64))
        .clone().to(mfq_tensor_backend::kCUDA);
}

struct KlEvalMetrics {
    double kld_sum = 0.0;
    double reverse_kld_sum = 0.0;
    double reference_ce_sum = 0.0;
    double quant_ce_sum = 0.0;
    int64_t same_top = 0;
    int64_t count = 0;
};

static void accumulate_streamed_kl_chunk(
    std::ifstream & f,
    const StreamedKlInput & input,
    int chunk_index,
    mfq_tensor_backend::Tensor pred,
    int score_count,
    KlEvalMetrics & metrics) {
    constexpr int KL_BATCH = 8;
    const auto & chunk = input.chunks[(size_t)chunk_index];
    f.seekg(chunk.row_offset);
    for (int s = 0; s < score_count; s += KL_BATCH) {
        const int b = std::min(KL_BATCH, score_count - s);
        std::vector<uint16_t> rows((size_t)b * input.nv);
        f.read(
            reinterpret_cast<char*>(rows.data()),
            (std::streamsize)(rows.size() * sizeof(uint16_t)));
        if (!f) throw std::runtime_error("truncated KL logits data");

        std::vector<float> scales(b), mins(b);
        std::vector<int32_t> codes((size_t)b * input.n_vocab);
        for (int r = 0; r < b; ++r) {
            const uint16_t * row = rows.data() + (size_t)r * input.nv;
            std::memcpy(&scales[r], row + 0, sizeof(float));
            std::memcpy(&mins[r], row + 2, sizeof(float));
            for (int v = 0; v < input.n_vocab; ++v) {
                codes[(size_t)r * input.n_vocab + v] = row[4 + v];
            }
        }
        auto scale = mfq_tensor_backend::from_blob(
            scales.data(), {b},
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kFloat32))
            .clone().to(mfq_tensor_backend::kCUDA);
        auto min_lp = mfq_tensor_backend::from_blob(
            mins.data(), {b},
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kFloat32))
            .clone().to(mfq_tensor_backend::kCUDA);
        auto base_codes = mfq_tensor_backend::from_blob(
            codes.data(), {b, input.n_vocab},
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt32))
            .clone().to(mfq_tensor_backend::kCUDA);
        auto base_logp = base_codes.to(mfq_tensor_backend::kFloat32) *
            scale.unsqueeze(1) + min_lp.unsqueeze(1);
        auto q = pred.index({Slice(s, s + b), Slice()}).to(mfq_tensor_backend::kFloat32);
        auto lse = mfq_tensor_backend::logsumexp(q, -1);
        auto quant_logp = q - lse.unsqueeze(1);
        auto normalized_base_logp =
            base_logp - mfq_tensor_backend::logsumexp(base_logp, -1, true);
        auto p_base =
            mfq_tensor_backend::exp(base_logp).masked_fill(base_codes.eq(0), 0.0f);
        auto kld =
            (p_base * (base_logp - q + lse.unsqueeze(1))).sum(-1);
        auto reverse_kld =
            (mfq_tensor_backend::exp(quant_logp) *
             (quant_logp - normalized_base_logp)).sum(-1);
        metrics.kld_sum +=
            kld.to(mfq_tensor_backend::kFloat64).sum().item<double>();
        metrics.reverse_kld_sum +=
            reverse_kld.to(mfq_tensor_backend::kFloat64).sum().item<double>();

        std::vector<int64_t> target_ids(b);
        for (int r = 0; r < b; ++r) {
            target_ids[r] =
                chunk.tokens[(size_t)chunk.target_start + s + r];
        }
        auto target = mfq_tensor_backend::from_blob(
            target_ids.data(), {b, 1},
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64))
            .clone().to(mfq_tensor_backend::kCUDA);
        if (chunk.target_log_probs.empty()) {
            metrics.reference_ce_sum +=
                -base_logp.gather(1, target)
                    .to(mfq_tensor_backend::kFloat64).sum().item<double>();
        } else {
            for (int r = 0; r < b; ++r) {
                metrics.reference_ce_sum -=
                    chunk.target_log_probs[(size_t)s + r];
            }
        }
        metrics.quant_ce_sum +=
            (lse.unsqueeze(1) - q.gather(1, target))
                .to(mfq_tensor_backend::kFloat64).sum().item<double>();
        metrics.same_top +=
            q.argmax(-1).eq(base_logp.argmax(-1)).sum().item<int64_t>();
        metrics.count += b;
    }
}

template <typename Model>
int run_kl_eval_batched(
    Model& model,
    const std::string & reference_path,
    int max_chunks,
    int64_t requested_n_batch,
    int score_override,
    const KlReferenceContract & reference_contract) {
    auto input = load_streamed_kl_input(reference_path, max_chunks);
    const int chunks = (int)input.chunks.size();
    const int64_t n_ctx = (int64_t)input.chunks[0].tokens.size();
    const int64_t n_batch = requested_n_batch == 0
        ? (reference_contract.n_batch == 0
              ? n_ctx : reference_contract.n_batch)
        : requested_n_batch;
    if (n_ctx > model.max_position_embeddings()) {
        throw std::runtime_error(
            "KL reference exceeds model context capacity");
    }
    if (n_ctx <= 0 || n_batch < n_ctx || n_batch % n_ctx != 0) {
        throw std::runtime_error(
            "optimized KL requires --kl-n-batch to be at least n_ctx "
            "and exactly divisible by n_ctx");
    }
    const int kl_n_seq = std::max<int64_t>(
        1, n_batch / n_ctx);
    validate_kl_execution_geometry(
        n_batch, n_batch, reference_contract, "optimized");
    KlKvCacheCapacityScope kv_cache_capacity_scope(n_ctx);
    const int target_start = input.chunks[0].target_start;
    int score_count = input.chunks[0].score_count;
    for (const auto & chunk : input.chunks) {
        if ((int64_t)chunk.tokens.size() != n_ctx ||
            chunk.target_start != target_start ||
            chunk.score_count != score_count) {
            throw std::runtime_error(
                "optimized KL requires uniform chunk geometry");
        }
    }
    if (score_override >= 0) {
        score_count = score_override;
        if (score_count < 1 ||
            score_count > input.chunks[0].score_count) {
            throw std::runtime_error(
                "--kl-score-count exceeds the stored chunk score count");
        }
    }
    if (input.n_vocab != model.vocab_size()) {
        throw std::runtime_error("KL vocab size mismatch");
    }
    int bos_replacements = 0;
    int32_t legacy_bos = -1;
    if (input.reference_format == "_logits_" &&
        Model::is_gemma4) {
        legacy_bos = input.chunks[0].tokens[0];
        for (auto & chunk : input.chunks) {
            if (chunk.tokens[0] != legacy_bos) {
                chunk.tokens[0] = legacy_bos;
                ++bos_replacements;
            }
        }
    }

    std::ifstream reference(reference_path, std::ios::binary);
    if (!reference) {
        throw std::runtime_error(
            "cannot reopen KL logits file: " + reference_path);
    }
    const int first = target_start - 1;
    KlEvalMetrics metrics;
    auto started = std::chrono::steady_clock::now();
    std::cout << "cpp_kl_execution evaluator=optimized"
              << " graph=batched_contexts"
              << " n_batch=" << n_batch
              << " n_ctx=" << n_ctx
              << " n_seq=" << kl_n_seq
              << " score_count=" << score_count
              << " score_count_override=" << score_override
              << " reference_n_batch=" << reference_contract.n_batch
              << " reference_n_ubatch=" << reference_contract.n_ubatch
              << " logits_start=" << first
              << " bos=" << legacy_bos
              << " bos_replacements=" << bos_replacements
              << "\n";

    for (int begin = 0; begin < chunks; begin += kl_n_seq) {
        const int n_seq_batch =
            std::min(kl_n_seq, chunks - begin);
        auto ids = streamed_kl_ids(
            input, begin, n_seq_batch, n_ctx);
        model.reset(n_seq_batch);
        auto hidden = model.hidden_forward(ids);
        for (int bi = 0; bi < n_seq_batch; ++bi) {
            auto selected_hidden = hidden.index({
                bi, Slice(first, first + score_count), Slice()
            }).contiguous();
            auto pred =
                model.logits_from_hidden(selected_hidden);
            accumulate_streamed_kl_chunk(
                reference, input, begin + bi, pred,
                score_count, metrics);
            std::cout << "cpp_kl_chunk=" << (begin + bi + 1)
                      << " mean="
                      << (metrics.kld_sum / (double)metrics.count)
                      << " mean_kld_q_ref="
                      << (metrics.reverse_kld_sum /
                          (double)metrics.count)
                      << " same_top="
                      << ((double)metrics.same_top /
                          (double)metrics.count)
                      << "\n";
        }
        mfq_cuda_synchronize();
    }
    auto ended = std::chrono::steady_clock::now();
    std::cout << "cpp_kl_result chunks=" << chunks
              << " scored_tokens=" << metrics.count
              << " sec="
              << std::chrono::duration<double>(
                     ended - started).count()
              << " kld="
              << (metrics.kld_sum / (double)metrics.count)
              << " mean_kld_q_ref="
              << (metrics.reverse_kld_sum / (double)metrics.count)
              << " bf16_ce="
              << (metrics.reference_ce_sum / (double)metrics.count)
              << " mfq_ce="
              << (metrics.quant_ce_sum / (double)metrics.count)
              << " bf16_ppl="
              << std::exp(
                     metrics.reference_ce_sum /
                     (double)metrics.count)
              << " mfq_ppl="
              << std::exp(
                     metrics.quant_ce_sum /
                     (double)metrics.count)
              << " kld_pct_bf16="
              << (100.0 * metrics.kld_sum /
                  metrics.reference_ce_sum)
              << " same_top="
              << ((double)metrics.same_top /
                  (double)metrics.count)
              << " same_top_count=" << metrics.same_top
              << " reference_format="
              << (input.reference_format == "_logit3_"
                      ? "trace_v3"
                      : (input.reference_format == "_logit2_"
                             ? "trace_v2" : "legacy"))
               << " execution=optimized"
               << " graph=batched_contexts"
               << " n_ctx=" << n_ctx
               << " n_batch=" << n_batch
               << " n_seq=" << kl_n_seq
               << " score_count=" << score_count
               << " score_count_override=" << score_override
               << " reference_n_batch=" << reference_contract.n_batch
               << " reference_n_ubatch=" << reference_contract.n_ubatch
               << "\n";
    std::cout << "cpp_kl_mmq"
              << " mmq=" << kl_mmq_mode_name(g_kl_mmq_mode)
              << " activation_quantize_calls="
              << g_kl_mmq_activation_quantize_calls
              << " dense_calls=" << g_kl_mmq_dense_calls
              << " moe_calls=" << g_kl_mmq_moe_calls
              << " fallback_calls=" << g_kl_mmq_fallback_calls
              << "\n";
    return 0;
}

template <typename Model>
int run_selected_kl_eval(
    Model& model,
    const std::string & reference_path,
    int max_chunks,
    KlEvaluator evaluator,
    int64_t requested_n_batch,
    int score_override,
    const KlReferenceContract & reference_contract) {
    return evaluator == KlEvaluator::Optimized
        ? run_kl_eval_batched(
              model, reference_path, max_chunks,
              requested_n_batch, score_override,
              reference_contract)
        : run_kl_eval(
              model, reference_path, max_chunks, evaluator,
              score_override, reference_contract);
}

int run_kl_eval_streamed(
    const std::string & model_path,
    const std::string & config_path,
    const std::string & reference_path,
    const std::string & logits_output_path,
    int max_chunks,
    int layer_group,
    int chunk_batch,
    int score_override,
    const KlReferenceContract & reference_contract) {
    if (layer_group < 1 || chunk_batch < 1) {
        throw std::runtime_error(
            "streamed KL layer group and chunk batch must be positive");
    }
    clear_moe_route_stats();
    MfqDropFileCacheGuard drop_cache_guard(true);
    auto input = load_streamed_kl_input(reference_path, max_chunks);
    const int chunks = (int)input.chunks.size();
    const int64_t n_ctx = (int64_t)input.chunks[0].tokens.size();
    const int64_t execution_n_seq = std::min(chunk_batch, chunks);
    const int64_t execution_n_batch = execution_n_seq * n_ctx;
    validate_kl_execution_geometry(
        execution_n_batch, execution_n_batch,
        reference_contract, "streamed");
    const int target_start = input.chunks[0].target_start;
    int score_count = input.chunks[0].score_count;
    for (const auto & chunk : input.chunks) {
        if ((int64_t)chunk.tokens.size() != n_ctx ||
            chunk.target_start != target_start ||
            chunk.score_count != score_count) {
            throw std::runtime_error(
                "streamed KL currently requires uniform chunk geometry");
        }
    }
    if (score_override >= 0) {
        score_count = score_override;
        if (score_count < 1 ||
            score_count > input.chunks[0].score_count) {
            throw std::runtime_error(
                "--kl-score-count exceeds the stored chunk score count");
        }
    }
    const int first = target_start - 1;

    std::ofstream logits_output;
    std::filesystem::path logits_final;
    std::filesystem::path logits_partial;
    if (!logits_output_path.empty()) {
        logits_final = std::filesystem::path(logits_output_path);
        logits_partial = logits_final;
        logits_partial += ".partial";
        if (std::filesystem::exists(logits_final) ||
                std::filesystem::exists(logits_partial)) {
            throw std::runtime_error(
                "refusing to overwrite saved KL logits: " +
                logits_output_path);
        }
        if (!logits_final.parent_path().empty()) {
            std::filesystem::create_directories(
                logits_final.parent_path());
        }
        logits_output.open(logits_partial, std::ios::binary);
        if (!logits_output) {
            throw std::runtime_error(
                "cannot create saved KL logits: " +
                logits_partial.string());
        }
        const char magic[8] = {'_', 'm', 'f', 'q', 'f', '1', '6', '_'};
        const int32_t header[5] = {
            chunks,
            static_cast<int32_t>(n_ctx),
            target_start,
            score_count,
            input.n_vocab,
        };
        logits_output.write(magic, sizeof(magic));
        logits_output.write(
            reinterpret_cast<const char *>(header), sizeof(header));
    }

    auto started = std::chrono::steady_clock::now();
    auto model = mfq::cuda::load_causal_lm<
        mfq::cuda::CudaBackbone::deepseek_v4>(
            model_path, config_path, n_ctx, false);
    if (chunk_batch > 16) {
        throw std::runtime_error(
            "DeepSeek V4 streamed KL chunk batch must not exceed 16");
    }
    if (model.vocab_size() != input.n_vocab) {
        throw std::runtime_error("KL vocab size mismatch");
    }
    const auto& mfq = *model.source;

    const int64_t hidden_bytes =
        (int64_t)chunks * n_ctx * model.hc_mult() *
        model.hidden_size() * (int64_t)sizeof(mfq_half);
    auto hidden_cpu = mfq_tensor_backend::empty(
        {chunks, n_ctx, model.hc_mult(), model.hidden_size()},
        mfq_tensor_backend::TensorOptions()
            .device(mfq_tensor_backend::kCPU)
            .dtype(mfq_tensor_backend::kFloat16)
            .pinned_memory(true));
    std::cout << "cpp_kl_stream_begin chunks=" << chunks
              << " ctx=" << n_ctx
              << " score_count=" << score_count
              << " score_count_override=" << score_override
              << " reference_n_batch=" << reference_contract.n_batch
              << " reference_n_ubatch=" << reference_contract.n_ubatch
              << " layer_group=" << layer_group
              << " chunk_batch=" << chunk_batch
              << " execution_n_seq=" << execution_n_seq
              << " execution_n_batch=" << execution_n_batch
              << " execution_n_ubatch=" << execution_n_batch
              << " hidden_bytes=" << hidden_bytes << "\n";

    for (int begin = 0; begin < chunks; begin += chunk_batch) {
        const int count = std::min(chunk_batch, chunks - begin);
        auto ids = streamed_kl_ids(input, begin, count, n_ctx);
        auto x = model.embed_forward(ids);
        x = x.unsqueeze(2)
            .expand({
                count, n_ctx, model.hc_mult(), model.hidden_size()})
            .contiguous();
        hidden_cpu.narrow(0, begin, count).copy_(x, true);
        mfq_cuda_synchronize();
    }
    std::cout << "cpp_kl_stream_phase=embedding completed_chunks="
              << chunks << "\n";

    for (int layer_begin = 0;
        layer_begin < model.num_hidden_layers();
         layer_begin += layer_group) {
        const int layer_end = std::min(
            layer_begin + layer_group,
            static_cast<int>(model.num_hidden_layers()));
        auto group_started = std::chrono::steady_clock::now();
        auto dsv4_state = std::make_shared<Dsv4SharedState>();
        std::vector<std::unique_ptr<Block>> blocks;
        blocks.reserve((size_t)(layer_end - layer_begin));
        for (int layer = layer_begin; layer < layer_end; ++layer) {
            std::cerr << "stream loading layer " << layer << " "
                      << model.layer_type(layer) << std::endl;
            blocks.push_back(mfq::cuda::deepseek_v4::load_block(
                mfq, model.config, layer,
                std::string(model.layer_type(layer)), dsv4_state));
        }
        for (int begin = 0; begin < chunks; begin += chunk_batch) {
            const int count = std::min(chunk_batch, chunks - begin);
            auto ids = streamed_kl_ids(input, begin, count, n_ctx);
            auto pos = mfq_tensor_backend::arange(
                0, n_ctx,
                mfq_tensor_backend::TensorOptions()
                    .device(mfq_tensor_backend::kCUDA).dtype(mfq_tensor_backend::kInt64));
            auto x = hidden_cpu.narrow(0, begin, count).to(mfq_tensor_backend::kCUDA);
            for (auto & block : blocks) {
                block->reset(count);
                block->set_token_ids(ids);
                x = block->forward(
                    x, pos, 0, mfq_nullopt, model.rope);
            }
            hidden_cpu.narrow(0, begin, count).copy_(x, true);
            mfq_cuda_synchronize();
        }
        blocks.clear();
        dsv4_state.reset();
        mfq_cuda_empty_cache();
        auto group_ended = std::chrono::steady_clock::now();
        std::cout << "cpp_kl_stream_layers begin=" << layer_begin
                  << " end=" << layer_end
                  << " sec="
                  << std::chrono::duration<double>(
                         group_ended - group_started).count()
                  << "\n";
    }

    std::ifstream reference(reference_path, std::ios::binary);
    if (!reference) {
        throw std::runtime_error(
            "cannot reopen KL logits file: " + reference_path);
    }
    KlEvalMetrics metrics;
    for (int begin = 0; begin < chunks; begin += chunk_batch) {
        const int count = std::min(chunk_batch, chunks - begin);
        auto x = hidden_cpu.narrow(0, begin, count).to(mfq_tensor_backend::kCUDA);
        auto y = model.finalize_hidden(x, count, n_ctx);
        auto selected =
            y.index({Slice(), Slice(first, first + score_count), Slice()});
        auto logits = model.apply_final_logit_softcap(
            model.lm_head.forward(selected));
        if (logits_output.is_open()) {
            auto saved = logits.to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kFloat16)
                             .contiguous();
            logits_output.write(
                reinterpret_cast<const char *>(
                    saved.data_ptr<mfq_half>()),
                static_cast<std::streamsize>(
                    saved.numel() * sizeof(mfq_half)));
            if (!logits_output) {
                throw std::runtime_error(
                    "failed while writing saved KL logits: " +
                    logits_partial.string());
            }
        }
        for (int bi = 0; bi < count; ++bi) {
            accumulate_streamed_kl_chunk(
                reference, input, begin + bi,
                logits.index({bi, Slice(), Slice()}),
                score_count, metrics);
            std::cout << "cpp_kl_chunk=" << (begin + bi + 1)
                      << " mean="
                      << (metrics.kld_sum / (double)metrics.count)
                      << " mean_kld_q_ref="
                      << (metrics.reverse_kld_sum / (double)metrics.count)
                      << " same_top="
                      << ((double)metrics.same_top /
                          (double)metrics.count)
                      << "\n";
        }
        mfq_cuda_synchronize();
    }
    auto ended = std::chrono::steady_clock::now();
    if (logits_output.is_open()) {
        logits_output.close();
        if (!logits_output) {
            throw std::runtime_error(
                "failed to finalize saved KL logits: " +
                logits_partial.string());
        }
        std::filesystem::rename(logits_partial, logits_final);
    }
    std::cout << "cpp_kl_result chunks=" << chunks
              << " scored_tokens=" << metrics.count
              << " sec="
              << std::chrono::duration<double>(ended - started).count()
              << " kld="
              << (metrics.kld_sum / (double)metrics.count)
              << " mean_kld_q_ref="
              << (metrics.reverse_kld_sum / (double)metrics.count)
              << " bf16_ce="
              << (metrics.reference_ce_sum / (double)metrics.count)
              << " mfq_ce="
              << (metrics.quant_ce_sum / (double)metrics.count)
              << " bf16_ppl="
              << std::exp(
                  metrics.reference_ce_sum / (double)metrics.count)
              << " mfq_ppl="
              << std::exp(metrics.quant_ce_sum / (double)metrics.count)
              << " kld_pct_bf16="
              << (100.0 * metrics.kld_sum / metrics.reference_ce_sum)
              << " same_top="
              << ((double)metrics.same_top / (double)metrics.count)
              << " reference_format="
              << (input.reference_format == "_logit3_"
                  ? "trace_v3"
                  : (input.reference_format == "_logit2_"
                     ? "trace_v2" : "legacy"))
              << " execution=streamed_layer_groups"
              << " graph=streamed_layer_groups"
              << " n_ctx=" << n_ctx
              << " execution_n_seq=" << execution_n_seq
              << " execution_n_batch=" << execution_n_batch
              << " execution_n_ubatch=" << execution_n_batch
              << " score_count=" << score_count
              << " score_count_override=" << score_override
              << " reference_n_batch=" << reference_contract.n_batch
              << " reference_n_ubatch=" << reference_contract.n_ubatch
              << "\n";
    write_moe_route_stats();
    return 0;
}

int run_linear_check(
    const std::string & model_path,
    const std::string & name,
    int M,
    int gate_mode,
    int reps) {
    auto model_source = mfq::open_model_source(model_path);
    const auto& mfq = *model_source;
    auto linear = load_quant_linear(mfq, name);
    MFQ_RUNTIME_CHECK(M >= 1 && M <= 4096, "--check-linear-m must be in [1, 4096]");
    MFQ_RUNTIME_CHECK(reps >= 1, "--check-linear-reps must be positive");
    int64_t neuron_len = linear.neuron_len();
    auto x = mfq_tensor_backend::arange((int64_t)M * neuron_len,
                           mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA).dtype(mfq_tensor_backend::kFloat32))
                 .reshape({M, neuron_len});
    x = (x.remainder(97) - 48) / 512.0;
    auto xh = x.to(mfq_tensor_backend::kFloat16).contiguous();
    mfq_tensor_backend::Tensor gateh;
    if (gate_mode != 0) {
        MFQ_RUNTIME_CHECK(gate_mode == 1 || gate_mode == 2, "linear check gate mode must be 0, 1, or 2");
        gateh = ((mfq_tensor_backend::arange((int64_t)M * neuron_len, x.options()).reshape({M, neuron_len})
                    .remainder(53) - 26) / 16.0)
                    .to(mfq_tensor_backend::kFloat16).contiguous();
    }
    auto run = [&]() {
        return gate_mode == 0 ? linear.forward(xh) : linear.forward_input_mul(xh, gateh, gate_mode);
    };
    const char * check_bf16_output_env =
        std::getenv("MFQ_CHECK_LINEAR_BF16_OUTPUT");
    if (check_bf16_output_env != nullptr &&
            check_bf16_output_env[0] == '1') {
        MFQ_RUNTIME_CHECK(
            gate_mode == 0,
            "direct BF16 linear check does not support input gating");
        auto bf16_input = x.to(mfq_tensor_backend::kBFloat16).contiguous();
        auto reference = linear.forward(bf16_input)
            .to(mfq_tensor_backend::kBFloat16).contiguous();
        auto candidate = linear.forward_bf16_output(bf16_input);
        auto difference =
            (candidate.to(mfq_tensor_backend::kFloat32) -
             reference.to(mfq_tensor_backend::kFloat32)).abs();
        std::cout << "linear_bf16_output_check"
                  << " max_abs="
                  << difference.max().item<double>()
                  << " equal="
                  << (candidate.equal(reference) ? 1 : 0)
                  << "\n";
        MFQ_RUNTIME_CHECK(
            candidate.equal(reference),
            "direct BF16 NINT output differs from FP16-then-BF16 reference");
    }
    mfq_tensor_backend::Tensor y_test;
    const int warmups = std::min(30, std::max(1, reps));
    for (int i = 0; i < warmups; ++i) y_test = run();
    mfq_cuda_synchronize();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    auto stream = mfq_get_current_cuda_stream().stream();
    cudaEventRecord(start, stream);
    for (int i = 0; i < reps; ++i) y_test = run();
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    y_test = y_test.to(mfq_tensor_backend::kFloat32);

    const NintWeight * nint = linear.is_nint() ? &linear.nint.w : nullptr;
    const NvqWeight * nvq = linear.is_nvq() ? &linear.nvq.w : nullptr;
    const Mxfp4Weight * mxfp4 = linear.is_mxfp4()
        ? &linear.mxfp4.weight : nullptr;
    const Mxfp4SqWeight * mxfp4_sq = linear.is_mxfp4_sq()
        ? &linear.mxfp4_sq.weight : nullptr;
    const Fp8SqWeight * fp8_sq = linear.is_fp8_sq()
        ? &linear.fp8_sq.weight : nullptr;
    const Mxfp8Weight * mxfp8 = linear.is_mxfp8()
        ? &linear.mxfp8.weight : nullptr;
    const TpqWeight * tpq = linear.is_tpq()
        ? &linear.tpq.weight : nullptr;
    auto ww = quant_linear_reference_weight(linear);
    mfq_tensor_backend::Tensor ref_input = xh;
    if (gate_mode == 1) ref_input = xh * mfq_tensor_backend::sigmoid(gateh);
    else if (gate_mode == 2) ref_input = xh * mfq_tensor_backend::silu(gateh);
    ref_input = ref_input.to(ww.scalar_type()).contiguous();
    auto y_ref = mfq_tensor_backend::matmul(ref_input, ww.transpose(0, 1)).to(mfq_tensor_backend::kFloat32);
    mfq_cuda_synchronize();
    auto diff = (y_test - y_ref).abs();
    double per_ms = (double)elapsed_ms / (double)reps;
    double weight_bytes = nint != nullptr
        ? (double)nint->q_packed.numel() +
          (nint->q8_zero
              ? (double)nint->q8_zero_scale.numel() * sizeof(mfq_half)
              : (double)(nint->sub_scale.numel() +
                         nint->sub_min.numel()) +
                (double)(nint->neuron_scale.numel() +
                         nint->neuron_min.numel()) * sizeof(float))
        : nvq != nullptr
        ? (double)nvq->indices_packed.numel() + (double)nvq->aux_packed.numel() +
          (double)nvq->sub_scale_packed.numel() +
          (double)nvq->neuron_scale.numel() * sizeof(float) + (double)nvq->codebook.numel()
        : mxfp4 != nullptr
        ? (double)mxfp4->values.numel() + (double)mxfp4->scales.numel()
        : mxfp4_sq != nullptr
        ? (double)mxfp4_sq->blob.numel()
        : fp8_sq != nullptr
        ? (double)fp8_sq->blob.numel()
        : tpq != nullptr
        ? (double)tpq->packed.numel() +
          (double)tpq->scales.numel() * sizeof(mfq_half) +
          (double)tpq->codebook.numel() * sizeof(float)
        : (double)mxfp8->values.numel() + (double)mxfp8->scales.numel();
    std::cout << "shape=" << y_ref.sizes() << "\n";
    if (nint != nullptr) {
        if (nint->q8_zero) {
            std::cout << "dtype=NINT8-0";
        } else {
            std::cout << "dtype=NINT format_version="
                      << nint->format_version
                      << " aggregate_bpw=" << nint->aggregate_bpw
                      << " distribution_entropy="
                      << nint->distribution_entropy
                      << " nominal_q=" << nint->bits;
        }
        std::cout << " gs=" << nint->gs << " m=" << M << "\n";
    } else if (nvq != nullptr) {
        std::cout << "dtype=NVQ profile=" << nvq->format << " gs=" << nvq->gs
                  << " sub_bits=" << nvq->sub_bits << " m=" << M << "\n";
    } else if (mxfp4 != nullptr) {
        std::cout << "dtype=MXFP4 block=1x32 m=" << M << "\n";
    } else if (mxfp4_sq != nullptr) {
        std::cout << "dtype=MXFP4-SQ format_version="
                  << mxfp4_sq->format_version
                  << " aggregate_bpw=" << mxfp4_sq->aggregate_bpw
                  << " distribution_entropy="
                  << mxfp4_sq->distribution_entropy;
        if (mxfp4_sq->bits != 0) {
            std::cout << " uniform_q=" << mxfp4_sq->bits;
        }
        std::cout << " block=1x32 m=" << M << "\n";
    } else if (fp8_sq != nullptr) {
        std::cout << "dtype=" << fp8_sq->dtype
                  << " format_version=" << fp8_sq->format_version
                  << " aggregate_bpw=" << fp8_sq->aggregate_bpw
                  << " distribution_entropy="
                  << fp8_sq->distribution_entropy
                  << " block=" << fp8_sq->block_rows
                  << "x" << fp8_sq->block_columns
                  << " m=" << M << "\n";
    } else if (tpq != nullptr) {
        std::cout << "dtype=" << (tpq->int4 ? "TPQ-I4G64" : "TPQ-PQ")
                  << " m=" << M << "\n";
    } else {
        std::cout << "dtype=MXFP8 block=128x128 m=" << M << "\n";
    }
    if (gate_mode != 0) std::cout << "gate=" << (gate_mode == 1 ? "sigmoid" : "silu") << "\n";
    std::cout << "production_ms=" << per_ms << "\n";
    std::cout << "production_weight_gbps=" << weight_bytes / (per_ms * 1.0e6) << "\n";
    std::cout << "production_rel=" << ((y_test - y_ref).norm() / y_ref.norm()).item<float>() << "\n";
    std::cout << "production_mean_abs=" << diff.mean().item<float>() << "\n";
    std::cout << "production_max_abs=" << diff.max().item<float>() << "\n";
    return 0;
}

int run_cpu_linear_check(
        const std::string & model_path,
        const std::string & name,
        int rows,
        int gate_mode,
        int reps) {
    MFQ_RUNTIME_CHECK(rows >= 1 && rows <= 4096, "--check-linear-m must be in [1, 4096]");
    MFQ_RUNTIME_CHECK(reps >= 1, "--check-linear-reps must be positive");
    auto model_source = mfq::open_model_source(model_path);
    const auto& mfq = *model_source;
    g_loading_cpu_layer = true;
    auto cpu_linear = load_quant_linear(mfq, name);
    g_loading_cpu_layer = false;
    auto cuda_linear = load_quant_linear(mfq, name);
    const int64_t width = cpu_linear.neuron_len();
    auto x = mfq_tensor_backend::arange(
        static_cast<int64_t>(rows) * width,
        mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCPU).dtype(mfq_tensor_backend::kFloat32))
        .reshape({rows, width});
    x = ((x.remainder(97) - 48) / 512.0)
        .to(mfq_tensor_backend::kFloat16).contiguous();
    mfq_tensor_backend::Tensor gate;
    if (gate_mode != 0) {
        MFQ_RUNTIME_CHECK(gate_mode == 1 || gate_mode == 2, "linear check gate mode must be 0, 1, or 2");
        gate = ((mfq_tensor_backend::arange(
            static_cast<int64_t>(rows) * width,
            mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCPU).dtype(mfq_tensor_backend::kFloat32))
            .reshape({rows, width}).remainder(53) - 26) / 16.0)
            .to(mfq_tensor_backend::kFloat16).contiguous();
    }
    auto run_cpu = [&]() {
        return gate_mode == 0
            ? cpu_linear.forward(x)
            : cpu_linear.forward_input_mul(x, gate, gate_mode);
    };
    mfq_tensor_backend::Tensor actual = run_cpu();
    const auto start = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < reps; ++iteration) {
        actual = run_cpu();
    }
    const auto end = std::chrono::steady_clock::now();
    const double cpu_ms = std::chrono::duration<double, std::milli>(
        end - start).count() / static_cast<double>(reps);

    auto cuda_x = x.to(mfq_tensor_backend::kCUDA).contiguous();
    mfq_tensor_backend::Tensor cuda_gate;
    if (gate_mode != 0) cuda_gate = gate.to(mfq_tensor_backend::kCUDA).contiguous();
    auto reference = gate_mode == 0
        ? cuda_linear.forward(cuda_x)
        : cuda_linear.forward_input_mul(cuda_x, cuda_gate, gate_mode);
    mfq_cuda_synchronize();
    reference = reference.to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kFloat32).contiguous();
    actual = actual.to(mfq_tensor_backend::kFloat32).contiguous();
    auto difference = (actual - reference).abs();
    std::cout << "cpu_linear=" << name << "\n"
              << "shape=" << actual.sizes() << "\n"
              << "cpu_ms=" << cpu_ms << "\n"
              << "relative_error="
              << ((actual - reference).norm() / reference.norm()).item<float>() << "\n"
              << "mean_abs=" << difference.mean().item<float>() << "\n"
              << "max_abs=" << difference.max().item<float>() << "\n";
    return 0;
}

int run_tensor_parallel_linear_check(
        const std::string & model_path,
        const std::string & name,
        TensorParallelAxis axis,
        int M) {
    MFQ_RUNTIME_CHECK(
        g_tensor_parallel.enabled(),
        "--check-tp-linear requires --tensor-parallel");
    MFQ_RUNTIME_CHECK(
        axis == TensorParallelAxis::Output ||
        axis == TensorParallelAxis::Input,
        "--check-tp-axis must be output or input");
    MFQ_RUNTIME_CHECK(
        M >= 1 && M <= 4096,
        "--check-tp-m must be in [1, 4096]");
    auto model_source = mfq::open_model_source(model_path);
    const auto& mfq = *model_source;
    const ParallelConfig saved =
        g_tensor_parallel;
    g_tensor_parallel = {};
    g_tensor_parallel.devices = {
        saved.primary_device()};
    auto full = load_quant_linear(
        mfq, name, TensorParallelAxis::Mirrored);
    g_tensor_parallel = saved;
    auto sharded = load_quant_linear(
        mfq, name, axis);
    MFQ_RUNTIME_CHECK(
        sharded.tensor_parallel(),
        "tensor-parallel diagnostic did not create shards");

    const int64_t width = full.neuron_len();
    auto x = mfq_tensor_backend::arange(
        static_cast<int64_t>(M) * width,
        mfq_tensor_backend::TensorOptions()
            .device(mfq_tensor_backend::Device(
                mfq_tensor_backend::kCUDA,
                saved.primary_device()))
            .dtype(mfq_tensor_backend::kFloat32))
        .reshape({M, width});
    x = ((x.remainder(127) - 63) / 384.0)
        .to(mfq_tensor_backend::kFloat16).contiguous();
    auto reference = full.forward(x).to(mfq_tensor_backend::kFloat32);
    auto test = sharded.forward(x).to(mfq_tensor_backend::kFloat32);
    mfq_cuda_synchronize();
    const auto difference = (test - reference).abs();
    const double denominator =
        std::max(
            reference.norm().item<double>(),
            1.0e-30);
    const double relative =
        (test - reference).norm().item<double>() /
        denominator;
    const double mean_abs =
        difference.mean().item<double>();
    const double max_abs =
        difference.max().item<double>();
    std::cout
        << "tensor_parallel_check=1"
        << " tensor=" << name
        << " axis="
        << (axis == TensorParallelAxis::Output
            ? "output" : "input")
        << " logical_shape=[" << full.out()
        << ',' << full.neuron_len() << ']'
        << " shards="
        << sharded.tensor_parallel_shards.size()
        << " m=" << M
        << " relative=" << relative
        << " mean_abs=" << mean_abs
        << " max_abs=" << max_abs
        << '\n';
    const double tolerance = full.is_mxfp8()
        ? 5.0e-4
        : axis == TensorParallelAxis::Output
        ? 1.0e-6 : 5.0e-3;
    if (!mfq_tensor_backend::isfinite(test).all().item<bool>() ||
        relative > tolerance) {
        throw std::runtime_error(
            "tensor-parallel linear numerical check failed");
    }
    return 0;
}

std::vector<std::string> parse_tensor_names(const std::string & value) {
    std::vector<std::string> names;
    size_t begin = 0;
    while (begin <= value.size()) {
        const size_t end = value.find(',', begin);
        const size_t count =
            end == std::string::npos ? value.size() - begin : end - begin;
        const std::string name = value.substr(begin, count);
        if (name.empty()) {
            throw std::runtime_error("empty tensor name in comma-separated list");
        }
        names.push_back(name);
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    return names;
}

int run_linear_group_check(
        const std::string & model_path,
        const std::string & names_arg,
        int M,
        int reps) {
    MFQ_RUNTIME_CHECK(M >= 1 && M <= 4096, "--check-linear-m must be in [1, 4096]");
    MFQ_RUNTIME_CHECK(reps >= 1, "--check-linear-reps must be positive");
    const auto names = parse_tensor_names(names_arg);
    MFQ_RUNTIME_CHECK(names.size() >= 2, "--check-linear-group requires at least two tensors");
    auto model_source = mfq::open_model_source(model_path);
    const auto& mfq = *model_source;
    const char * preserve_env =
        std::getenv("MFQ_CHECK_LINEAR_GROUP_PRESERVE");
    const bool preserve_projection_boundaries =
        preserve_env != nullptr && preserve_env[0] == '1';
    const char * bf16_env =
        std::getenv("MFQ_CHECK_LINEAR_GROUP_BF16");
    const bool check_bf16 =
        bf16_env != nullptr && bf16_env[0] == '1';
    auto group = load_quant_group(
        mfq, names, names.size(), nullptr,
        preserve_projection_boundaries);
    const int64_t width = group.nint_grouped
        ? (group.nint.split_w.empty()
            ? group.nint.w.neuron_len
            : group.nint.split_w.front().neuron_len)
        : group.layers.front().neuron_len();
    auto sequence = mfq_tensor_backend::arange(
        (int64_t)M * width,
        mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA).dtype(mfq_tensor_backend::kFloat32));
    auto x = (((sequence.remainder(257) - 128.0) / 127.0) +
              0.03125 * mfq_tensor_backend::sin(sequence * 0.015625))
                 .to(check_bf16
                     ? mfq_tensor_backend::kBFloat16
                     : mfq_tensor_backend::kFloat16)
                 .reshape({M, width})
                 .contiguous();
    auto actual = group.forward(x);
    const char * check_bf16_swiglu_env =
        std::getenv("MFQ_CHECK_BF16_SWIGLU");
    if (check_bf16_swiglu_env != nullptr &&
            check_bf16_swiglu_env[0] == '1') {
        MFQ_RUNTIME_CHECK(
            actual.size() == 2 &&
            actual[0].scalar_type() == mfq_tensor_backend::kBFloat16 &&
            actual[1].scalar_type() == mfq_tensor_backend::kBFloat16,
            "BF16 SwiGLU check requires two BF16 projection outputs");
        auto reference =
            (mfq_tensor_backend::silu(actual[0]) * actual[1]).contiguous();
        auto candidate = silu_mul_cuda(
            actual[0].contiguous(), actual[1].contiguous());
        auto difference =
            (candidate.to(mfq_tensor_backend::kFloat32) -
             reference.to(mfq_tensor_backend::kFloat32)).abs();
        std::cout << "bf16_swiglu_check"
                  << " max_abs="
                  << difference.max().item<double>()
                  << " equal="
                  << (candidate.equal(reference) ? 1 : 0)
                  << "\n";
        MFQ_RUNTIME_CHECK(
            candidate.equal(reference),
            "fused BF16 SwiGLU differs from the official BF16 expression");
    }
    mfq_cuda_synchronize();
    const auto started = std::chrono::steady_clock::now();
    for (int rep = 0; rep < reps; ++rep) {
        actual = group.forward(x);
    }
    mfq_cuda_synchronize();
    const double elapsed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    std::cout << "linear_group_timing"
              << " preserve=" << (preserve_projection_boundaries ? 1 : 0)
              << " m=" << M
              << " reps=" << reps
              << " mean_ms=" << elapsed_ms / reps << '\n';
    MFQ_RUNTIME_CHECK(actual.size() == names.size(), "linear group output count mismatch");
    std::vector<mfq_tensor_backend::Tensor> graph_actual;
    if (M == 1 && decode_branch_parallel_enabled(M)) {
        mfq_cuda_synchronize();
        const auto graph_stream =
            mfq_get_stream_from_pool(false);
        MfqCudaStreamGuard graph_guard(
            graph_stream);
        MfqCudaGraph graph;
        mfq_prepare_cuda_graph_memory(graph);
        graph_actual = group.forward(x);
        MFQ_CUDA_CHECK(cudaStreamSynchronize(graph_stream.stream()));
        graph_actual.clear();
        graph.capture_begin();
        graph_actual = group.forward(x);
        graph.capture_end();
        graph.replay();
        MFQ_CUDA_CHECK(cudaStreamSynchronize(
            graph_stream.stream()));
        MFQ_RUNTIME_CHECK(
            graph_actual.size() == actual.size(),
            "linear group CUDA Graph output count mismatch");
        for (size_t index = 0;
             index < graph_actual.size(); ++index) {
            const auto difference =
                (graph_actual[index].to(mfq_tensor_backend::kFloat32) -
                 actual[index].to(mfq_tensor_backend::kFloat32)).abs();
            const double maximum =
                difference.max().item<double>();
            std::cout
                << "linear_group_graph_check"
                << " tensor=" << names[index]
                << " max_abs=" << maximum << "\n";
            MFQ_RUNTIME_CHECK(
                maximum == 0.0,
                "linear group CUDA Graph replay differs from eager output");
        }
    }
    mfq_disable_tf32_cublas();
    std::vector<mfq_tensor_backend::Tensor> dense_weights;
    std::vector<mfq_tensor_backend::Tensor> separate_dense_references;
    std::vector<mfq_tensor_backend::Tensor> fp32_references;
    std::vector<mfq_tensor_backend::Tensor> separate_production;
    dense_weights.reserve(names.size());
    separate_dense_references.reserve(names.size());
    fp32_references.reserve(names.size());
    separate_production.reserve(names.size());
    for (size_t index = 0; index < names.size(); ++index) {
        auto linear = load_quant_linear(mfq, names[index]);
        MFQ_RUNTIME_CHECK(
            linear.is_nint(),
            "--check-linear-group currently requires NINT tensors");
        const auto & weight = linear.nint.w;
        auto dense = weight.q8_zero
            ? nint8_zero_dequant_cuda(
                  weight.q_packed, weight.q8_zero_scale, weight.neuron_len)
            : nint_decode_cuda(
                  weight.q_packed, weight.row_q_bits,
                  weight.row_q_bit_offsets, weight.sub_scale,
                  weight.sub_min, weight.neuron_scale,
                  weight.neuron_min, weight.neuron_len, weight.gs);
        auto separate = linear.forward(x);
        if (actual[index].scalar_type() == mfq_tensor_backend::kBFloat16) {
            separate = separate.to(mfq_tensor_backend::kBFloat16);
        }
        separate_production.push_back(separate.to(mfq_tensor_backend::kFloat32));
        auto dense_for_x = dense.to(x.scalar_type());
        separate_dense_references.push_back(
            mfq_tensor_backend::matmul(x, dense_for_x.transpose(0, 1))
                .to(mfq_tensor_backend::kFloat32));
        fp32_references.push_back(mfq_tensor_backend::matmul(
            x.to(mfq_tensor_backend::kFloat32),
            dense.to(mfq_tensor_backend::kFloat32).transpose(0, 1)));
        dense_weights.push_back(std::move(dense));
    }
    auto combined_dense = mfq_tensor_backend::cat(dense_weights, 0)
        .to(x.scalar_type()).contiguous();
    auto combined_output =
        mfq_tensor_backend::matmul(x, combined_dense.transpose(0, 1)).to(mfq_tensor_backend::kFloat32);
    auto combined_references =
        combined_output.split_with_sizes(group.outs, -1);
    for (size_t index = 0; index < names.size(); ++index) {
        auto combined_reference = combined_references[index];
        auto separate_dense_reference = separate_dense_references[index];
        auto fp32_reference = fp32_references[index];
        auto separate_candidate = separate_production[index];
        auto candidate = actual[index].to(mfq_tensor_backend::kFloat32);
        auto combined_dense_difference = candidate - combined_reference;
        auto grouped_vs_separate = candidate - separate_candidate;
        auto separate_dense_difference =
            separate_candidate - separate_dense_reference;
        auto grouped_fp32_difference = candidate - fp32_reference;
        auto separate_fp32_difference = separate_candidate - fp32_reference;
        const double grouped_vs_separate_rel =
            (grouped_vs_separate.norm() /
             separate_candidate.norm().clamp_min(1.0e-30)).item<double>();
        const double grouped_vs_separate_snr =
            grouped_vs_separate_rel == 0.0
                ? std::numeric_limits<double>::infinity()
                : -20.0 * std::log10(grouped_vs_separate_rel);
        const double grouped_fp32_rel =
            (grouped_fp32_difference.norm() /
             fp32_reference.norm().clamp_min(1.0e-30)).item<double>();
        const double separate_fp32_rel =
            (separate_fp32_difference.norm() /
             fp32_reference.norm().clamp_min(1.0e-30)).item<double>();
        std::cout << std::fixed << std::setprecision(9)
                  << "linear_group_check"
                  << " tensor=" << names[index]
                  << " m=" << M
                  << " n=" << combined_reference.size(1)
                  << " k=" << width
                  << " grouped_vs_combined_dense_rel="
                  << (combined_dense_difference.norm() /
                      combined_reference.norm().clamp_min(1.0e-30)).item<double>()
                  << " grouped_vs_separate_rel=" << grouped_vs_separate_rel
                  << " grouped_vs_separate_snr_db=" << grouped_vs_separate_snr
                  << " grouped_vs_separate_mean_abs="
                  << grouped_vs_separate.abs().mean().item<double>()
                  << " grouped_vs_separate_max_abs="
                  << grouped_vs_separate.abs().max().item<double>()
                  << " separate_vs_separate_dense_rel="
                  << (separate_dense_difference.norm() /
                      separate_dense_reference.norm().clamp_min(1.0e-30)).item<double>()
                  << " grouped_vs_fp32_rel=" << grouped_fp32_rel
                  << " grouped_vs_fp32_snr_db="
                  << (grouped_fp32_rel == 0.0
                          ? std::numeric_limits<double>::infinity()
                          : -20.0 * std::log10(grouped_fp32_rel))
                  << " separate_vs_fp32_rel=" << separate_fp32_rel
                  << " separate_vs_fp32_snr_db="
                  << (separate_fp32_rel == 0.0
                          ? std::numeric_limits<double>::infinity()
                          : -20.0 * std::log10(separate_fp32_rel))
                  << "\n";
    }
    return 0;
}

static mfq_tensor_backend::Tensor read_f32_tensor(
        const std::filesystem::path & path,
        const std::vector<int64_t> & shape) {
    const int64_t count = std::accumulate(
        shape.begin(), shape.end(), int64_t{1}, std::multiplies<int64_t>());
    std::vector<float> values(static_cast<size_t>(count));
    std::ifstream input(path, std::ios::binary);
    MFQ_RUNTIME_CHECK(input, "failed to open ", path.string());
    input.read(
        reinterpret_cast<char *>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(float)));
    MFQ_RUNTIME_CHECK(
        input.gcount() ==
            static_cast<std::streamsize>(values.size() * sizeof(float)),
        "short f32 tensor read from ", path.string());
    return mfq_tensor_backend::from_blob(
               values.data(), shape,
               mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kFloat32))
        .clone()
        .to(mfq_tensor_backend::kCUDA)
        .contiguous();
}

static void write_f32_tensor(
        const std::filesystem::path & path,
        const mfq_tensor_backend::Tensor & tensor) {
    auto host = tensor.to(mfq_tensor_backend::kCPU).to(mfq_tensor_backend::kFloat32).contiguous();
    std::ofstream output(path, std::ios::binary);
    MFQ_RUNTIME_CHECK(output, "failed to create ", path.string());
    output.write(
        reinterpret_cast<const char *>(host.data_ptr<float>()),
        static_cast<std::streamsize>(host.numel() * sizeof(float)));
    MFQ_RUNTIME_CHECK(output, "failed to write ", path.string());
}

int run_gdn_operator_check(
        const std::string & input_dir,
        const std::string & output_path,
        const std::string & state_path,
        int64_t tokens,
        int64_t q_heads,
        int64_t v_heads,
        int64_t head_dim) {
    MFQ_RUNTIME_CHECK(
        tokens >= 1 && q_heads >= 1 && v_heads >= q_heads && head_dim >= 1,
        "invalid GDN diagnostic shape");
    MFQ_RUNTIME_CHECK(
        v_heads % q_heads == 0,
        "GDN diagnostic value heads must be divisible by query heads");
    const std::filesystem::path root(input_dir);
    auto q = read_f32_tensor(
        root / "q.bin", {1, tokens, q_heads, head_dim})
                 .permute({0, 2, 1, 3})
                 .contiguous();
    auto k = read_f32_tensor(
        root / "k.bin", {1, tokens, q_heads, head_dim})
                 .permute({0, 2, 1, 3})
                 .contiguous();
    auto v = read_f32_tensor(
        root / "v.bin", {1, tokens, v_heads, head_dim})
                 .permute({0, 2, 1, 3})
                 .contiguous();
    auto g = read_f32_tensor(
        root / "g.bin", {1, tokens, v_heads})
                 .permute({0, 2, 1})
                 .contiguous();
    auto beta = read_f32_tensor(
        root / "beta.bin", {1, tokens, v_heads})
                    .permute({0, 2, 1})
                    .contiguous();
    auto state = read_f32_tensor(
        root / "state.bin", {1, v_heads, head_dim, head_dim});
    auto result = gdn_inplace_transposed_tiled_cuda(
        q, k, v, g, beta, state);
    write_f32_tensor(output_path, result[0]);
    write_f32_tensor(state_path, result[1]);
    std::cout << "mfq_gdn_operator"
              << " t=" << tokens
              << " hq=" << q_heads
              << " hv=" << v_heads
              << " d=" << head_dim
              << " output=" << output_path
              << " state=" << state_path << "\n";
    return 0;
}

int run_linear_conv_operator_check(
        const std::string & input_dir,
        const std::string & output_dir,
        int64_t tokens,
        int64_t q_heads,
        int64_t v_heads,
        int64_t key_dim,
        int64_t value_dim,
        int64_t kernel_size,
        double eps) {
    MFQ_RUNTIME_CHECK(
        tokens >= 2 && q_heads >= 1 && v_heads >= 1 &&
            key_dim >= 1 && value_dim >= 1 &&
            kernel_size >= 2 && kernel_size <= 8,
        "invalid linear-conv diagnostic shape");
    const int64_t qk_width = 2 * q_heads * key_dim;
    const int64_t v_width = v_heads * value_dim;
    const int64_t channels = qk_width + v_width;
    const std::filesystem::path input_root(input_dir);
    const std::filesystem::path output_root(output_dir);
    std::filesystem::create_directories(output_root);
    auto state = read_f32_tensor(
        input_root / "state.bin",
        {1, kernel_size - 1, channels});
    auto qk = read_f32_tensor(
                  input_root / "qk.bin",
                  {1, tokens, qk_width})
                  .to(mfq_tensor_backend::kHalf)
                  .contiguous();
    auto v = read_f32_tensor(
                 input_root / "v.bin",
                 {1, tokens, v_width})
                 .to(mfq_tensor_backend::kHalf)
                 .contiguous();
    auto weight = read_f32_tensor(
        input_root / "weight.bin",
        {channels, 1, kernel_size});
    mfq_tensor_backend::Tensor bias;
    if (std::filesystem::exists(input_root / "bias.bin")) {
        bias = read_f32_tensor(
            input_root / "bias.bin", {channels});
    } else {
        bias = mfq_tensor_backend::empty(
            {0},
            mfq_tensor_backend::TensorOptions()
                .device(mfq_tensor_backend::kCUDA)
                .dtype(mfq_tensor_backend::kFloat32));
    }
    auto result = linear_conv_qkv_prefill_cuda(
        state, qk, v, weight, bias,
        q_heads, v_heads, key_dim, value_dim, eps);
    write_f32_tensor(output_root / "q.bin", result[0]);
    write_f32_tensor(output_root / "k.bin", result[1]);
    write_f32_tensor(output_root / "v.bin", result[2]);
    write_f32_tensor(output_root / "state.bin", result[3]);
    std::cout << "mfq_linear_conv_operator"
              << " t=" << tokens
              << " hq=" << q_heads
              << " hv=" << v_heads
              << " dk=" << key_dim
              << " dv=" << value_dim
              << " kernel=" << kernel_size
              << " output=" << output_dir << "\n";
    return 0;
}

int run_q8_embedding_check(
        const std::string & model_path,
        const std::string & name) {
    auto model_source = mfq::open_model_source(model_path);
    const auto& mfq = *model_source;
    auto linear = load_quant_linear(mfq, name);
    MFQ_RUNTIME_CHECK(
        linear.is_nint() && linear.nint.w.q8_zero,
        "--check-q8-embedding requires an NINT8-0 tensor");
    const int64_t vocab = linear.nint.w.out;
    std::vector<int64_t> host_ids = {
        0,
        std::min<int64_t>(1, vocab - 1),
        std::min<int64_t>(106, vocab - 1),
        std::min<int64_t>(12345, vocab - 1),
        std::min<int64_t>(255999, vocab - 1),
        vocab - 1,
    };
    auto ids = mfq_tensor_backend::from_blob(
        host_ids.data(), {(int64_t)host_ids.size()},
        mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64))
                   .clone()
                   .to(mfq_tensor_backend::kCUDA)
                   .contiguous();
    auto candidate = nint8_zero_embedding_lookup_cuda(
        linear.nint.w.q_packed, linear.nint.w.q8_zero_scale,
        ids, linear.nint.w.neuron_len);
    auto dense = nint8_zero_dequant_cuda(
        linear.nint.w.q_packed, linear.nint.w.q8_zero_scale,
        linear.nint.w.neuron_len);
    auto reference = dense.index_select(0, ids);
    auto difference =
        candidate.to(mfq_tensor_backend::kFloat32) - reference.to(mfq_tensor_backend::kFloat32);
    std::cout << std::fixed << std::setprecision(9)
              << "q8_embedding_check"
              << " tensor=" << name
              << " ids=" << host_ids.size()
              << " vocab=" << vocab
              << " width=" << linear.nint.w.neuron_len
              << " equal=" << (candidate.equal(reference) ? 1 : 0)
              << " rel="
              << (difference.norm() /
                  reference.to(mfq_tensor_backend::kFloat32).norm()).item<double>()
              << " mean_abs=" << difference.abs().mean().item<double>()
              << " max_abs=" << difference.abs().max().item<double>()
              << "\n";
    return 0;
}

int run_dsv4_output_a_check(
        const std::string & model_path,
        const std::string & name,
        int batch,
        int reps) {
    constexpr int64_t kGroups = 8;
    MFQ_RUNTIME_CHECK(batch > 0 && reps > 0, "DSV4 output_a check requires positive batch and reps");
    auto model_source = mfq::open_model_source(model_path);
    const auto& mfq = *model_source;
    auto linear = load_quant_linear(
        mfq, name, TensorParallelAxis::Input);
    const bool supported_nint =
        linear.is_nint() &&
        linear.nint.w.bits == 8 &&
        linear.nint.w.gs == 48;
    MFQ_RUNTIME_CHECK(
        supported_nint || linear.is_mxfp8(),
        "DSV4 output_a check requires NINT8 gs48 or MXFP8");
    MFQ_RUNTIME_CHECK(
        linear.out() % kGroups == 0,
        "DSV4 output_a rows must divide eight groups");

    const int64_t width = linear.neuron_len();
    const int64_t rows_per_group = linear.out() / kGroups;
    auto sequence = mfq_tensor_backend::arange(
        (int64_t)batch * kGroups * width,
        mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA).dtype(mfq_tensor_backend::kFloat32));
    auto grouped = (
        (sequence.remainder(257) - 128.0) / 127.0 +
        0.03125 * mfq_tensor_backend::sin(sequence * 0.015625))
        .to(mfq_tensor_backend::kFloat16)
        .reshape({batch, kGroups, width})
        .contiguous();

    auto legacy = [&]() {
        auto expanded = linear.forward(
            grouped.reshape({batch * kGroups, width}))
            .reshape({batch, kGroups, kGroups, rows_per_group});
        std::vector<mfq_tensor_backend::Tensor> diagonal;
        diagonal.reserve(kGroups);
        for (int64_t group = 0; group < kGroups; ++group) {
            diagonal.push_back(
                expanded.index({Slice(), group, group, Slice()}));
        }
        return mfq_tensor_backend::stack(diagonal, 1).reshape({batch, linear.out()});
    };
    auto groupwise = [&]() {
        return linear.is_mxfp8()
            ? linear.forward_mxfp8_groupwise(grouped, kGroups)
            : nint_matmul_groupwise_u8(
                linear.nint.w, grouped, kGroups);
    };
    auto time_ms = [&](auto && fn) {
        mfq_tensor_backend::Tensor output;
        for (int warmup = 0; warmup < 10; ++warmup) output = fn();
        mfq_cuda_synchronize();
        cudaEvent_t start, stop;
        MFQ_CUDA_CHECK(cudaEventCreate(&start));
        MFQ_CUDA_CHECK(cudaEventCreate(&stop));
        auto stream = mfq_get_current_cuda_stream().stream();
        MFQ_CUDA_CHECK(cudaEventRecord(start, stream));
        for (int iteration = 0; iteration < reps; ++iteration) output = fn();
        MFQ_CUDA_CHECK(cudaEventRecord(stop, stream));
        MFQ_CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed = 0.0f;
        MFQ_CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
        MFQ_CUDA_CHECK(cudaEventDestroy(start));
        MFQ_CUDA_CHECK(cudaEventDestroy(stop));
        return std::pair<float, mfq_tensor_backend::Tensor>(elapsed / reps, output);
    };

    auto legacy_result = time_ms(legacy);
    auto groupwise_result = time_ms(groupwise);
    auto reference = legacy_result.second.to(mfq_tensor_backend::kFloat32);
    auto candidate = groupwise_result.second.to(mfq_tensor_backend::kFloat32);
    auto diff = (candidate - reference).abs();
    const float relative =
        ((candidate - reference).norm() / reference.norm()).item<float>();
    std::cout << std::fixed << std::setprecision(9)
              << "dsv4_output_a_check"
              << " tensor=" << name
              << " format=" << (linear.is_mxfp8() ? "MXFP8" : "NINT8")
              << " batch=" << batch
              << " groups=" << kGroups
              << " rows_per_group=" << rows_per_group
              << " k=" << width
              << " equal=" << (candidate.equal(reference) ? 1 : 0)
              << " rel=" << relative
              << " mean_abs=" << diff.mean().item<float>()
              << " max_abs=" << diff.max().item<float>()
              << " legacy_ms=" << legacy_result.first
              << " groupwise_ms=" << groupwise_result.first
              << " speedup=" << legacy_result.first / groupwise_result.first
              << " checksum=" << candidate.sum().item<double>()
              << "\n";
    if (linear.is_mxfp8()) {
        MFQ_RUNTIME_CHECK(
            mfq_tensor_backend::isfinite(candidate).all().item<bool>() &&
                relative <= 5.0e-4f,
            "DSV4 MXFP8 groupwise output_a exceeded the FP16 GEMM "
            "reduction-order tolerance");
    } else {
        MFQ_RUNTIME_CHECK(
            candidate.equal(reference),
            "DSV4 NINT groupwise output_a must be bit-exact with the "
            "legacy path");
    }
    return 0;
}

int run_gemma_geglu_check(
    const std::string & model_path,
    int layer,
    int reps) {
    if (layer < 0 || reps < 1) {
        throw std::runtime_error("Gemma GeGLU check requires a nonnegative layer and positive reps");
    }
    auto model_source = mfq::open_model_source(model_path);
    const auto& mfq = *model_source;
    mfq::cuda::validate_model_source(mfq);
    const std::string prefix =
        "model.block." + std::to_string(layer) + ".mlp.";
    auto gate_up = load_quant_group(
        mfq, {prefix + "gate.weight", prefix + "up.weight"}, 2);
    auto down = load_quant_linear(mfq, prefix + "down.weight");
    if (!gate_up.nint_grouped || !gate_up.nint.split_w.empty() || !down.is_nint() ||
        gate_up.outs.size() != 2 || gate_up.outs[0] != gate_up.outs[1]) {
        throw std::runtime_error("Gemma GeGLU check requires packed NINT gate/up and NINT down tensors");
    }

    const int64_t hidden = gate_up.nint.w.neuron_len;
    auto xf = mfq_tensor_backend::arange(
        hidden, mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA).dtype(mfq_tensor_backend::kFloat32));
    auto x = (((xf.remainder(257) - 128.0) / 64.0) +
              0.125 * mfq_tensor_backend::sin(xf * 0.03125)).to(mfq_tensor_backend::kFloat16).reshape({1, hidden}).contiguous();

    auto materialized_activation = [&]() {
        auto parts = gate_up.forward(x);
        return gelu_mul_cuda(parts[0].contiguous(), parts[1].contiguous());
    };
    auto materialized = [&]() { return down.forward(materialized_activation()); };
    auto fused_activation = [&]() { return gate_up.forward_geglu(x); };
    auto fused = [&]() { return down.forward(fused_activation()); };

    auto reference_activation = materialized_activation();
    auto reference_output = down.forward(reference_activation);
    std::cout << "gemma_geglu_check layer=" << layer
              << " gate_up_bits=" << gate_up.nint.w.bits
              << " gate_up_gs=" << gate_up.nint.w.gs
              << " gate_out=" << gate_up.outs[0]
              << " up_out=" << gate_up.outs[1]
              << " down_bits=" << down.nint.w.bits
              << " down_gs=" << down.nint.w.gs << "\n";
    auto report = [&](const char * name, mfq_tensor_backend::Tensor value, mfq_tensor_backend::Tensor reference) {
        auto got = value.to(mfq_tensor_backend::kFloat64);
        auto ref = reference.to(mfq_tensor_backend::kFloat64);
        const double ref_norm = std::max(ref.norm().item<double>(), 1.0e-30);
        const double got_norm = std::max(got.norm().item<double>(), 1.0e-30);
        std::cout << "gemma_geglu_check path=" << name
                  << " relative_l2=" << (got - ref).norm().item<double>() / ref_norm
                  << " cosine=" << mfq_tensor_backend::dot(got.reshape({-1}), ref.reshape({-1})).item<double>() /
                         (got_norm * ref_norm)
                  << " max_abs=" << (got - ref).abs().max().item<double>() << "\n";
    };

    mfq_set_env("MFQ_NINT_GLU_COMBINED", "1");
    report("activation_combined", fused_activation(), reference_activation);
    report("output_combined", fused(), reference_output);
    mfq_set_env("MFQ_NINT_GLU_COMBINED", "0");
    report("activation_pair", fused_activation(), reference_activation);
    report("output_pair", fused(), reference_output);
    mfq_set_env("MFQ_NINT_GLU_COMBINED", "");

    auto time_ms = [&](auto && fn) {
        for (int i = 0; i < 10; ++i) (void)fn();
        mfq_cuda_synchronize();
        cudaEvent_t start, stop;
        MFQ_CUDA_CHECK(cudaEventCreate(&start));
        MFQ_CUDA_CHECK(cudaEventCreate(&stop));
        auto stream = mfq_get_current_cuda_stream().stream();
        MFQ_CUDA_CHECK(cudaEventRecord(start, stream));
        for (int i = 0; i < reps; ++i) (void)fn();
        MFQ_CUDA_CHECK(cudaEventRecord(stop, stream));
        MFQ_CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed = 0.0f;
        MFQ_CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
        MFQ_CUDA_CHECK(cudaEventDestroy(start));
        MFQ_CUDA_CHECK(cudaEventDestroy(stop));
        return elapsed / reps;
    };
    const float materialized_ms = time_ms(materialized);
    mfq_set_env("MFQ_NINT_GLU_COMBINED", "1");
    const float combined_ms = time_ms(fused);
    mfq_set_env("MFQ_NINT_GLU_COMBINED", "0");
    const float pair_ms = time_ms(fused);
    mfq_set_env("MFQ_NINT_GLU_COMBINED", "");
    std::cout << "gemma_geglu_check layer=" << layer
              << " gate_bits=" << gate_up.nint.w.bits
              << " gate_gs=" << gate_up.nint.w.gs
              << " down_bits=" << down.nint.w.bits
              << " down_gs=" << down.nint.w.gs
              << " materialized_ms=" << materialized_ms
              << " combined_ms=" << combined_ms
              << " pair_ms=" << pair_ms << "\n";
    return 0;
}


static double mfe_weight_bytes(const MfeWeight & weight) {
    double bytes = static_cast<double>(weight.mixed_weight_bytes);
    for (const auto & shard :
         weight.expert_parallel_shards) {
        if (shard.weight) {
            bytes += mfe_weight_bytes(
                *shard.weight);
        }
    }
    for (const auto & pool : weight.pools) {
        bytes += static_cast<double>(pool.weight.q_packed.numel()) * pool.weight.q_packed.element_size();
        bytes += static_cast<double>(pool.weight.row_q_bits.numel()) * pool.weight.row_q_bits.element_size();
        bytes += static_cast<double>(pool.weight.row_q_bit_offsets.numel()) * pool.weight.row_q_bit_offsets.element_size();
        bytes += static_cast<double>(pool.weight.sub_scale.numel()) * pool.weight.sub_scale.element_size();
        bytes += static_cast<double>(pool.weight.sub_min.numel()) * pool.weight.sub_min.element_size();
        bytes += static_cast<double>(pool.weight.neuron_scale.numel()) * pool.weight.neuron_scale.element_size();
        bytes += static_cast<double>(pool.weight.neuron_min.numel()) * pool.weight.neuron_min.element_size();
    }
    return bytes;
}

int run_expert_parallel_moe_check(
        const std::string & model_path,
        const std::string & tensor_name,
        int tokens,
        int routes) {
    MFQ_RUNTIME_CHECK(
        moe_parallel_config().enabled(),
        "--check-ep-moe requires --expert-parallel or --tensor-parallel");
    MFQ_RUNTIME_CHECK(
        tokens >= 1 && tokens <= 4096,
        "--check-ep-moe-tokens must be in [1, 4096]");
    MFQ_RUNTIME_CHECK(
        routes >= 1,
        "--check-ep-moe-routes must be positive");
    auto model_source = mfq::open_model_source(model_path);
    const auto& mfq = *model_source;
    const std::string role =
        tensor_name.find("gate_up") != std::string::npos
        ? "gate_up" : "diagnostic";
    const ParallelConfig saved_tensor = g_tensor_parallel;
    const ParallelConfig saved_expert = g_expert_parallel;
    const ParallelConfig saved = moe_parallel_config();
    g_tensor_parallel = {};
    g_expert_parallel = {};
    g_expert_parallel.devices = {saved.primary_device()};
    auto full = load_mfe_gpu(
        mfq, tensor_name, false, 0, role);
    g_tensor_parallel = saved_tensor;
    g_expert_parallel = saved_expert;
    auto sharded = load_mfe_gpu(
        mfq, tensor_name, false, 0, role);
    MFQ_RUNTIME_CHECK(
        sharded.expert_parallel(),
        "expert-parallel MoE diagnostic did not create shards");
    MFQ_RUNTIME_CHECK(
        full.n_experts == sharded.n_experts &&
        full.out_per_expert == sharded.out_per_expert &&
        full.neuron_len == sharded.neuron_len,
        "expert-parallel MoE metadata differs");
    MFQ_RUNTIME_CHECK(
        routes <= full.n_experts,
        "--check-ep-moe-routes exceeds the expert count");

    MfqCudaGuard primary_guard(
        saved.primary_device());
    const int64_t count =
        static_cast<int64_t>(tokens) *
        full.neuron_len;
    auto sequence = mfq_tensor_backend::arange(
        count,
        mfq_tensor_backend::TensorOptions()
            .device(mfq_tensor_backend::Device(
                mfq_tensor_backend::kCUDA,
                saved.primary_device()))
            .dtype(mfq_tensor_backend::kFloat32));
    auto x = (
        (sequence.remainder(257) - 128.0) / 127.0 +
        0.03125 * mfq_tensor_backend::sin(sequence * 0.015625))
        .to(mfq_tensor_backend::kFloat16)
        .reshape({tokens, full.neuron_len})
        .contiguous();
    std::vector<int32_t> host_ids(
        static_cast<size_t>(tokens) *
        static_cast<size_t>(routes));
    for (int token = 0; token < tokens; ++token) {
        for (int route = 0; route < routes; ++route) {
            host_ids[
                static_cast<size_t>(token) * routes +
                static_cast<size_t>(route)] =
                (token * routes + route * 3) %
                full.n_experts;
        }
    }
    auto ids = mfq_tensor_backend::from_blob(
        host_ids.data(), {tokens, routes},
        mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt32))
        .clone()
        .to(mfq_tensor_backend::Device(
            mfq_tensor_backend::kCUDA,
            saved.primary_device()))
        .contiguous();
    auto route =
        build_moe_route_plan(
            ids, full.n_experts);
    auto reference =
        full.forward(x, route)
            .to(mfq_tensor_backend::kFloat32);
    auto test =
        sharded.forward(x, route)
            .to(mfq_tensor_backend::kFloat32);
    mfq_cuda_synchronize();
    auto difference = test - reference;
    const double denominator =
        std::max(
            reference.norm().item<double>(),
            1.0e-30);
    const double relative =
        difference.norm().item<double>() /
        denominator;
    const double mean_abs =
        difference.abs().mean().item<double>();
    const double max_abs =
        difference.abs().max().item<double>();
    const int64_t differing =
        test.ne(reference).sum().item<int64_t>();
    std::cout
        << "expert_parallel_moe_check=1"
        << " tensor=" << tensor_name
        << " tokens=" << tokens
        << " routes=" << routes
        << " experts=" << full.n_experts
        << " logical_shape=["
        << full.out_per_expert << ','
        << full.neuron_len << ']'
        << " shards="
        << sharded.expert_parallel_shards.size()
        << " differing=" << differing
        << " relative=" << relative
        << " mean_abs=" << mean_abs
        << " max_abs=" << max_abs
        << '\n';
    if (!mfq_tensor_backend::isfinite(test).all().item<bool>() ||
        relative > 1.0e-6) {
        throw std::runtime_error(
            "expert-parallel MoE numerical check failed");
    }
    return 0;
}

int run_mfe_tensor_check(
        const std::string & model_path,
        const std::string & tensor_name,
        int tokens,
        int routes,
        int reps,
        int split_width,
        bool routed_input,
        bool benchmark_only) {
    const int max_tokens = benchmark_only ? 131072 : 4096;
    if (tokens < 1 || tokens > max_tokens || routes < 1 || reps < 1 ||
            split_width < 0) {
        throw std::runtime_error("MFE tensor check dimensions are invalid");
    }
    if (benchmark_only && split_width != 0) {
        throw std::runtime_error(
            "MFE benchmark-only mode does not support split checks");
    }
    auto model_source = mfq::open_model_source(model_path);
    const auto& mfq = *model_source;
    auto weight = load_mfe_gpu(
        mfq, tensor_name, true, 0, "diagnostic");
    if (g_moe_expert_cache &&
            !moe_expert_cache_finalized()) {
        finalize_moe_expert_cache();
    }
    if (routes > weight.n_experts) {
        throw std::runtime_error("MFE tensor check routes exceed expert count");
    }
    if (split_width > 0 &&
            (split_width >= weight.out_per_expert || weight.mixed_forward)) {
        throw std::runtime_error(
            "MFE merged/split check requires a resident NINT tensor "
            "and an interior split width");
    }
    const int64_t count =
        (int64_t)tokens * (routed_input ? routes : 1) * weight.neuron_len;
    const auto input_shape = routed_input
        ? std::vector<int64_t>{tokens, routes, weight.neuron_len}
        : std::vector<int64_t>{tokens, weight.neuron_len};
    mfq_tensor_backend::Tensor x;
    if (benchmark_only) {
        x = mfq_tensor_backend::zeros(
            input_shape,
            mfq_tensor_backend::TensorOptions()
                .device(mfq_tensor_backend::kCUDA)
                .dtype(mfq_tensor_backend::kFloat16));
    } else {
        auto sequence = mfq_tensor_backend::arange(
            count,
            mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA).dtype(mfq_tensor_backend::kFloat32));
        x = (
            (sequence.remainder(257) - 128.0) / 127.0 +
            0.03125 * mfq_tensor_backend::sin(sequence * 0.015625))
            .to(mfq_tensor_backend::kFloat16)
            .reshape(input_shape)
            .contiguous();
    }
    std::vector<int32_t> host_ids((size_t)tokens * routes);
    for (int token = 0; token < tokens; ++token) {
        for (int route = 0; route < routes; ++route) {
            host_ids[(size_t)token * routes + route] =
                (token * routes + route * 3) % weight.n_experts;
        }
    }
    auto ids = mfq_tensor_backend::from_blob(
        host_ids.data(), {tokens, routes},
        mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt32))
        .clone().to(mfq_tensor_backend::kCUDA).contiguous();
    auto route = build_moe_route_plan(ids, weight.n_experts);
    mfq_tensor_backend::Tensor output;
    for (int warmup = 0; warmup < 5; ++warmup) {
        output = weight.forward(x, route);
        if (warmup == 0) weight.prefetch(route);
    }
    mfq_cuda_synchronize();
    cudaEvent_t start, stop;
    MFQ_CUDA_CHECK(cudaEventCreate(&start));
    MFQ_CUDA_CHECK(cudaEventCreate(&stop));
    auto stream = mfq_get_current_cuda_stream().stream();
    MFQ_CUDA_CHECK(cudaEventRecord(start, stream));
    for (int index = 0; index < reps; ++index) output = weight.forward(x, route);
    MFQ_CUDA_CHECK(cudaEventRecord(stop, stream));
    MFQ_CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed = 0.0f;
    MFQ_CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
    MFQ_CUDA_CHECK(cudaEventDestroy(start));
    MFQ_CUDA_CHECK(cudaEventDestroy(stop));
    if (benchmark_only) {
        std::cout << std::fixed << std::setprecision(9)
                  << "mfe_tensor_benchmark"
                  << " tensor=" << tensor_name
                  << " tokens=" << tokens
                  << " routes=" << routes
                  << " experts=" << weight.n_experts
                  << " out=" << weight.out_per_expert
                  << " k=" << weight.neuron_len
                  << " routed_input=" << (routed_input ? 1 : 0)
                  << " mixed=" << (weight.mixed_forward ? 1 : 0)
                  << " weight_bytes=" << mfe_weight_bytes(weight)
                  << " cuda_ms=" << elapsed / reps
                  << '\n';
        return 0;
    }
    double dense_reference_rel = -1.0;
    double dense_reference_mean_abs = -1.0;
    double dense_reference_max_abs = -1.0;
    if (split_width == 0) {
        auto reference = mfe_dense_reference(
            mfq, tensor_name, x, host_ids,
            tokens, routes, routed_input);
        mfq_cuda_synchronize();
        auto actual_f32 =
            output.reshape({tokens * routes, weight.out_per_expert})
                .to(mfq_tensor_backend::kFloat32);
        auto reference_f32 = reference.to(mfq_tensor_backend::kFloat32);
        auto difference = actual_f32 - reference_f32;
        dense_reference_rel =
            (difference.norm() / reference_f32.norm()).item<double>();
        dense_reference_mean_abs =
            difference.abs().mean().item<double>();
        dense_reference_max_abs =
            difference.abs().max().item<double>();
    }
    auto output_f32 = output.to(mfq_tensor_backend::kFloat32);
    if (!mfq_tensor_backend::isfinite(output_f32).all().item<bool>()) {
        throw std::runtime_error("MFE tensor check produced a non-finite value");
    }
    const double checksum = output_f32.sum().item<double>();
    const double squared_checksum = output_f32.square().sum().item<double>();
    auto flat = output_f32.cpu().reshape({-1});
    std::cout << std::fixed << std::setprecision(9)
              << "mfe_tensor_check"
              << " tensor=" << tensor_name
              << " tokens=" << tokens
              << " routes=" << routes
              << " experts=" << weight.n_experts
              << " out=" << weight.out_per_expert
              << " k=" << weight.neuron_len
              << " routed_input=" << (routed_input ? 1 : 0)
              << " mixed=" << (weight.mixed_forward ? 1 : 0)
              << " weight_bytes=" << mfe_weight_bytes(weight)
              << " cuda_ms=" << elapsed / reps
              << " dense_reference_rel=" << dense_reference_rel
              << " dense_reference_mean_abs=" << dense_reference_mean_abs
              << " dense_reference_max_abs=" << dense_reference_max_abs
              << " checksum=" << checksum
              << " sqsum=" << squared_checksum
              << " values=";
    const int64_t shown = std::min<int64_t>(flat.numel(), 128);
    const float * values = flat.data_ptr<float>();
    for (int64_t index = 0; index < shown; ++index) {
        if (index) std::cout << ",";
        std::cout << values[index];
    }
    std::cout << "\n";

    if (split_width > 0) {
        auto run_segment = [&](int width, int row_offset) {
            return weight.forward(x, route)
                .narrow(2, row_offset, width).contiguous();
        };
        mfq_tensor_backend::Tensor merged;
        mfq_tensor_backend::Tensor left;
        mfq_tensor_backend::Tensor right;
        for (int warmup = 0; warmup < 3; ++warmup) {
            merged = run_segment(weight.out_per_expert, 0);
            left = run_segment(split_width, 0);
            right = run_segment(
                weight.out_per_expert - split_width, split_width);
        }
        mfq_cuda_synchronize();
        auto split = mfq_tensor_backend::cat({left, right}, 2).contiguous();
        auto difference =
            merged.to(mfq_tensor_backend::kFloat32) - split.to(mfq_tensor_backend::kFloat32);
        const int64_t left_differing =
            merged.slice(2, 0, split_width).ne(left).sum().item<int64_t>();
        const int64_t right_differing =
            merged.slice(2, split_width, weight.out_per_expert)
                .ne(right).sum().item<int64_t>();
        const int64_t differing = left_differing + right_differing;
        const double merged_norm = merged.to(mfq_tensor_backend::kFloat32).norm().item<double>();
        std::cout << std::scientific << std::setprecision(9)
                  << "mfe_merged_split_check"
                  << " tensor=" << tensor_name
                  << " path=nint_matmul_kernel"
                  << " tokens=" << tokens
                  << " routes=" << routes
                  << " experts=" << weight.n_experts
                  << " bm=" << (tokens <= 128 ? 64 : (tokens <= 512 ? 32 : 64))
                  << " bn=64"
                  << " k=" << weight.neuron_len
                  << " full_width=" << weight.out_per_expert
                  << " split_width=" << split_width
                  << " right_width=" << weight.out_per_expert - split_width
                  << " weight_out_stride=" << weight.out_per_expert
                  << " left_row_offset=0"
                  << " right_row_offset=" << split_width
                  << " equal=" << (merged.equal(split) ? 1 : 0)
                  << " differing=" << differing
                  << " values=" << merged.numel()
                  << " left_differing=" << left_differing
                  << " right_differing=" << right_differing
                  << " rel_l2="
                  << (merged_norm == 0.0
                          ? 0.0
                          : difference.norm().item<double>() / merged_norm)
                  << " mean_abs=" << difference.abs().mean().item<double>()
                  << " max_abs=" << difference.abs().max().item<double>()
                  << "\n";
    }

    const char * warp_ab_env = std::getenv("MFQ_CHECK_NVQ_MOE_WARP_AB");
    if (warp_ab_env != nullptr && std::atoi(warp_ab_env) != 0) {
        const char * original_env = std::getenv("MFQ_NVQ_MOE_WARPS");
        const bool had_original_env = original_env != nullptr;
        const std::string original_value = had_original_env ? original_env : "";
        const char * original_exact_env =
            std::getenv("MFQ_NVQ_MOE_EXACT_REDUCTION");
        const bool had_original_exact_env = original_exact_env != nullptr;
        const std::string original_exact_value =
            had_original_exact_env ? original_exact_env : "";
        auto set_env = [](const char * name, const char * value) {
#ifdef _WIN32
            _putenv_s(name, value);
#else
            setenv(name, value, 1);
#endif
        };
        auto restore_env = [&](const char * name, bool existed, const std::string & value) {
#ifdef _WIN32
            _putenv_s(name, existed ? value.c_str() : "");
#else
            if (existed) {
                setenv(name, value.c_str(), 1);
            } else {
                unsetenv(name);
            }
#endif
        };
        set_env("MFQ_NVQ_MOE_WARPS", "0");
        set_env("MFQ_NVQ_MOE_EXACT_REDUCTION", "1");
        auto candidate = weight.forward(x, route);
        set_env("MFQ_NVQ_MOE_EXACT_REDUCTION", "0");
        auto baseline = weight.forward(x, route);
        mfq_cuda_synchronize();
        restore_env(
            "MFQ_NVQ_MOE_WARPS", had_original_env, original_value);
        restore_env(
            "MFQ_NVQ_MOE_EXACT_REDUCTION",
            had_original_exact_env, original_exact_value);
        auto candidate_f32 = candidate.to(mfq_tensor_backend::kFloat32);
        auto baseline_f32 = baseline.to(mfq_tensor_backend::kFloat32);
        auto diff = candidate_f32 - baseline_f32;
        const double baseline_norm = baseline_f32.norm().item<double>();
        std::cout << std::scientific << std::setprecision(9)
                  << "nvq_moe_warp_ab"
                  << " candidate_physical_warps=2"
                  << " baseline_warps=" << (weight.neuron_len >= 4096 ? 8 : 4)
                  << " equal=" << (candidate.equal(baseline) ? 1 : 0)
                  << " differing=" << candidate.ne(baseline).sum().item<int64_t>()
                  << " rel_l2=" << diff.norm().item<double>() / baseline_norm
                  << " mean_abs=" << diff.abs().mean().item<double>()
                  << " max_abs=" << diff.abs().max().item<double>()
                  << "\n";
    }

    const char * clamped_ab_env =
        std::getenv("MFQ_CHECK_MFE_CLAMPED_SWIGLU_AB");
    if (clamped_ab_env != nullptr && weight.supports_clamped_swiglu()) {
        const double limit = std::atof(clamped_ab_env);
        if (!std::isfinite(limit) || limit <= 0.0) {
            throw std::runtime_error(
                "MFQ_CHECK_MFE_CLAMPED_SWIGLU_AB must be positive");
        }
        const int64_t gate_up_count =
            static_cast<int64_t>(tokens) * routes * 2 * weight.neuron_len;
        auto gate_up_sequence = mfq_tensor_backend::arange(
            gate_up_count,
            mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA).dtype(mfq_tensor_backend::kFloat32));
        auto gate_up = (
            12.0 * mfq_tensor_backend::sin(gate_up_sequence * 0.013671875) +
            0.5 * mfq_tensor_backend::cos(gate_up_sequence * 0.00390625))
            .to(mfq_tensor_backend::kFloat16)
            .reshape({tokens, routes, 2 * weight.neuron_len})
            .contiguous();
        auto run_candidate = [&]() {
            return weight.forward_clamped_swiglu(gate_up, route, limit);
        };
        auto run_baseline = [&]() {
            const int64_t width = weight.neuron_len;
            auto gate = mfq_tensor_backend::clamp_max(
                gate_up.slice(-1, 0, width).to(mfq_tensor_backend::kFloat32), limit);
            auto up = mfq_tensor_backend::clamp(
                gate_up.slice(-1, width, 2 * width).to(mfq_tensor_backend::kFloat32),
                -limit, limit);
            auto hidden = (mfq_tensor_backend::silu(gate) * up)
                .to(mfq_tensor_backend::kFloat16).contiguous();
            return weight.forward(hidden, route);
        };
        mfq_tensor_backend::Tensor candidate;
        mfq_tensor_backend::Tensor baseline;
        for (int warmup = 0; warmup < 5; ++warmup) {
            candidate = run_candidate();
            baseline = run_baseline();
        }
        mfq_cuda_synchronize();
        auto time_ms = [&](auto && fn) {
            cudaEvent_t begin, end;
            MFQ_CUDA_CHECK(cudaEventCreate(&begin));
            MFQ_CUDA_CHECK(cudaEventCreate(&end));
            auto cuda_stream = mfq_get_current_cuda_stream().stream();
            MFQ_CUDA_CHECK(cudaEventRecord(begin, cuda_stream));
            mfq_tensor_backend::Tensor value;
            for (int iteration = 0; iteration < reps; ++iteration) {
                value = fn();
            }
            MFQ_CUDA_CHECK(cudaEventRecord(end, cuda_stream));
            MFQ_CUDA_CHECK(cudaEventSynchronize(end));
            float elapsed_ms = 0.0f;
            MFQ_CUDA_CHECK(cudaEventElapsedTime(
                &elapsed_ms, begin, end));
            MFQ_CUDA_CHECK(cudaEventDestroy(begin));
            MFQ_CUDA_CHECK(cudaEventDestroy(end));
            return std::pair<double, mfq_tensor_backend::Tensor>(
                elapsed_ms / reps, std::move(value));
        };
        auto candidate_timing = time_ms(run_candidate);
        auto baseline_timing = time_ms(run_baseline);
        candidate = std::move(candidate_timing.second);
        baseline = std::move(baseline_timing.second);
        auto candidate_f32 = candidate.to(mfq_tensor_backend::kFloat32);
        auto baseline_f32 = baseline.to(mfq_tensor_backend::kFloat32);
        auto diff = candidate_f32 - baseline_f32;
        const double baseline_norm = baseline_f32.norm().item<double>();
        std::cout << std::scientific << std::setprecision(9)
                  << "mfe_clamped_swiglu_ab"
                  << " limit=" << limit
                  << " candidate_ms=" << candidate_timing.first
                  << " baseline_ms=" << baseline_timing.first
                  << " speedup=" << baseline_timing.first / candidate_timing.first
                  << " equal=" << (candidate.equal(baseline) ? 1 : 0)
                  << " differing=" << candidate.ne(baseline).sum().item<int64_t>()
                  << " rel_l2=" << diff.norm().item<double>() / baseline_norm
                  << " mean_abs=" << diff.abs().mean().item<double>()
                  << " max_abs=" << diff.abs().max().item<double>()
                  << "\n";
    }
    if (g_moe_expert_cache) {
        print_moe_expert_cache_stats(std::cout);
    }
    return 0;
}

static int run_gemma_moe_check(
        const mfq::ModelSource & mfq,
        const mfq::models::gemma4::Config & config,
        int layer,
        const std::vector<int64_t> & token_sizes,
    int reps) {
    const std::string prefix =
        "model.block." + std::to_string(layer) + ".mlp.experts.";
    auto gate_up = load_mfe_gpu(
        mfq, prefix + "gate_up.weight",
        true, layer, "gate_up");
    auto down = load_mfe_gpu(
        mfq, prefix + "down.weight",
        true, layer, "down");
    if (g_moe_expert_cache &&
            !moe_expert_cache_finalized()) {
        finalize_moe_expert_cache();
    }
    const int routes = static_cast<int>(config.num_experts_per_tok);
    const int experts = static_cast<int>(config.num_experts);
    MFQ_RUNTIME_CHECK(routes > 0 && routes <= 8 && experts > routes,
        "Gemma MoE benchmark requires 1..8 routes and more experts than routes");
    const char * dense_reference_env =
        std::getenv("MFQ_CHECK_GEMMA_MOE_DENSE_REFERENCE");
    const bool dense_reference_enabled =
        dense_reference_env != nullptr && std::atoi(dense_reference_env) != 0;
    mfq_tensor_backend::Tensor dense_gate_up;
    mfq_tensor_backend::Tensor dense_down;
    if (dense_reference_enabled) {
        dense_gate_up = materialize_mfe_dense(
            mfq, prefix + "gate_up.weight");
        dense_down = materialize_mfe_dense(
            mfq, prefix + "down.weight");
    }
    std::cout << "gemma_moe_bench_config"
              << " layer=" << layer
              << " experts=" << experts
              << " top_k=" << routes
              << " hidden=" << gate_up.neuron_len
              << " intermediate=" << down.neuron_len
              << " gate_up_pools=" << gate_up.pools.size()
              << " down_pools=" << down.pools.size()
              << " routed_weight_bytes=" << std::fixed << std::setprecision(0)
              << mfe_weight_bytes(gate_up) + mfe_weight_bytes(down)
              << "\n";

    auto stream = mfq_get_current_cuda_stream().stream();
    auto time_ms = [&](auto && fn, int iterations) {
        mfq_tensor_backend::Tensor output;
        for (int warmup = 0; warmup < 5; ++warmup) output = fn();
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, stream);
        for (int iteration = 0; iteration < iterations; ++iteration) output = fn();
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float elapsed = 0.0f;
        cudaEventElapsedTime(&elapsed, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return std::pair<double, mfq_tensor_backend::Tensor>(elapsed / iterations, output);
    };

    auto topk_logits = mfq_tensor_backend::randn(
        {1, experts}, mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA).dtype(mfq_tensor_backend::kFloat32));
    auto selected = moe_topk_cuda(
        topk_logits, routes, false, false, false, true, mfq_nullopt, 1e-20, 1.0);
    auto reference = mfq_tensor_backend::topk(topk_logits, routes, 1, true, true);
    auto reference_weights = mfq_tensor_backend::softmax(std::get<0>(reference), 1);
    mfq_cuda_synchronize();
    auto topk_timing = time_ms([&]() {
        return moe_topk_cuda(
            topk_logits, routes, false, false, false, true,
            mfq_nullopt, 1e-20, 1.0).at(1);
    }, reps);
    std::cout << std::fixed << std::setprecision(6)
              << "gemma_topk_check"
              << " ids_equal="
              << (selected.at(0).equal(std::get<1>(reference).to(mfq_tensor_backend::kInt32)) ? 1 : 0)
              << " weights_max_abs="
              << (selected.at(1) - reference_weights).abs().max().item<float>()
              << " cuda_ms=" << topk_timing.first << "\n";

    mfq_tensor_backend::manual_seed(20260721 + layer);
    for (int64_t tokens : token_sizes) {
        auto x = mfq_tensor_backend::randn(
            {tokens, gate_up.neuron_len},
            mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA).dtype(mfq_tensor_backend::kFloat16));
        std::vector<int32_t> host_ids(static_cast<size_t>(tokens) * routes);
        for (int64_t token = 0; token < tokens; ++token) {
            for (int route_index = 0; route_index < routes; ++route_index) {
                host_ids[static_cast<size_t>(token) * routes + route_index] =
                    static_cast<int32_t>(
                        (token * routes + route_index) % experts);
            }
        }
        auto ids = mfq_tensor_backend::from_blob(
            host_ids.data(), {tokens, routes}, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt32))
            .clone().to(mfq_tensor_backend::kCUDA).contiguous();
        auto weights = mfq_tensor_backend::full(
            {tokens, routes}, 1.0 / routes,
            mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA).dtype(mfq_tensor_backend::kFloat32));
        auto forward_materialized = [&]() {
            auto route = build_moe_route_plan(ids, experts);
            const bool projection_bundle_prefetched =
                prefetch_cached_moe_projection_bundle(
                    gate_up, down, route);
            auto gate_pair = gate_up.forward(x, route);
            auto hidden = moe_geglu_split_cuda(gate_pair);
            if (!projection_bundle_prefetched) down.prefetch(route);
            auto down_pair = down.forward(hidden, route);
            return moe_weighted_reduce_cuda(down_pair, weights);
        };
        auto forward_gate_glu = [&]() {
            auto route = build_moe_route_plan(ids, experts);
            const bool projection_bundle_prefetched =
                prefetch_cached_moe_projection_bundle(
                    gate_up, down, route);
            auto hidden = gate_up.forward_glu_output(x, route, true);
            if (!projection_bundle_prefetched) down.prefetch(route);
            auto down_pair = down.forward(hidden, route);
            return moe_weighted_reduce_cuda(down_pair, weights);
        };
        const bool gate_glu_supported = tokens <= 4;
        auto forward = [&]() {
            return gate_glu_supported
                ? forward_gate_glu()
                : forward_materialized();
        };
        auto fused_check = forward();
        auto materialized_check = forward_materialized();
        auto gate_glu_check = gate_glu_supported
            ? forward_gate_glu()
            : fused_check;
        mfq_cuda_synchronize();
        auto fused_diff = (fused_check - materialized_check).abs().to(mfq_tensor_backend::kFloat32);
        auto fused_time = time_ms(forward, reps);
        auto materialized_time = time_ms(forward_materialized, reps);
        auto gate_glu_time = gate_glu_supported
            ? time_ms(forward_gate_glu, reps)
            : fused_time;
        auto gate_glu_diff = (gate_glu_check - materialized_check).abs().to(mfq_tensor_backend::kFloat32);
        std::cout << std::fixed << std::setprecision(6)
                  << "gemma_moe_geglu_quant_fusion"
                  << " tokens=" << tokens
                  << " equal=" << (fused_check.equal(materialized_check) ? 1 : 0)
                  << " max_abs=" << fused_diff.max().item<float>()
                  << " fused_ms=" << fused_time.first
                  << " materialized_ms=" << materialized_time.first
                  << " speedup=" << materialized_time.first / fused_time.first
                  << " gate_glu_supported=" << (gate_glu_supported ? 1 : 0)
                  << " gate_glu_equal=" << (gate_glu_check.equal(materialized_check) ? 1 : 0)
                  << " gate_glu_max_abs=" << gate_glu_diff.max().item<float>()
                  << " gate_glu_ms=" << gate_glu_time.first << "\n";

        if (tokens == 1) {
            auto stage_route = build_moe_route_plan(ids, experts);
            auto stage_hidden = gate_up.forward_glu_output(x, stage_route, true);
            auto stage_down = down.forward(stage_hidden, stage_route);
            mfq_cuda_synchronize();
            auto gate_stage = time_ms(
                [&]() { return gate_up.forward_glu_output(x, stage_route, true); }, reps);
            auto down_stage = time_ms(
                [&]() { return down.forward(stage_hidden, stage_route); }, reps);
            auto reduce_stage = time_ms(
                [&]() { return moe_weighted_reduce_cuda(stage_down, weights); }, reps);
            std::cout << "gemma_moe_stage"
                      << " gate_up_geglu_ms=" << gate_stage.first
                      << " down_ms=" << down_stage.first
                      << " reduce_ms=" << reduce_stage.first << "\n";
        }

        if (tokens != 1) {
            g_force_moe_prefill_mma_off = false;
            auto mma = time_ms(forward, reps);
            auto mma_first = mma.second.clone();
            auto mma_repeat = forward().clone();
            mfq_cuda_synchronize();
            g_force_moe_prefill_mma_off = true;
            auto baseline = time_ms(forward, reps);
            mfq_cuda_synchronize();
            g_force_moe_prefill_mma_off = false;
            auto repeat_diff = (mma_repeat - mma_first).abs().to(mfq_tensor_backend::kFloat32);
            auto baseline_diff = (mma_first - baseline.second).abs().to(mfq_tensor_backend::kFloat32);
            std::cout << std::setprecision(6)
                      << "gemma_moe_prefill_ab"
                      << " tokens=" << tokens
                      << " mma_ms=" << mma.first
                      << " baseline_ms=" << baseline.first
                      << " speedup=" << baseline.first / mma.first
                      << " repeat_equal=" << (mma_repeat.equal(mma_first) ? 1 : 0)
                      << " repeat_max_abs=" << repeat_diff.max().item<float>()
                      << " baseline_rel="
                      << ((mma_first - baseline.second).to(mfq_tensor_backend::kFloat32).norm() /
                          baseline.second.to(mfq_tensor_backend::kFloat32).norm()).item<float>()
                      << " baseline_max_abs=" << baseline_diff.max().item<float>()
                      << "\n";
            if (dense_reference_enabled) {
                auto dense_route_forward = [&](const mfq_tensor_backend::Tensor & dense,
                                               const mfq_tensor_backend::Tensor & input) {
                    auto result = mfq_tensor_backend::empty(
                        {tokens, routes, dense.size(1)},
                        input.options().dtype(mfq_tensor_backend::kFloat16));
                    for (int route_index = 0; route_index < routes; ++route_index) {
                        const int expert = (route_index * 17) % experts;
                        auto selected = input.dim() == 3
                            ? input.select(1, route_index)
                            : input;
                        result.select(1, route_index).copy_(mfq_tensor_backend::matmul(
                            selected,
                            dense.index({expert}).transpose(0, 1)));
                    }
                    return result;
                };
                auto dense_gate_pair =
                    dense_route_forward(dense_gate_up, x);
                auto dense_hidden =
                    moe_geglu_split_cuda(dense_gate_pair);
                auto dense_down_pair =
                    dense_route_forward(dense_down, dense_hidden);
                auto dense_output =
                    moe_weighted_reduce_cuda(dense_down_pair, weights);
                auto dense_difference =
                    mma_first.to(mfq_tensor_backend::kFloat32) -
                    dense_output.to(mfq_tensor_backend::kFloat32);
                const double dense_norm =
                    dense_output.to(mfq_tensor_backend::kFloat32).norm().item<double>();
                std::cout << std::scientific << std::setprecision(9)
                          << "gemma_moe_dense_reference"
                          << " tokens=" << tokens
                          << " rel_l2="
                          << dense_difference.norm().item<double>() / dense_norm
                          << " mean_abs="
                          << dense_difference.abs().mean().item<double>()
                          << " max_abs="
                          << dense_difference.abs().max().item<double>()
                          << "\n";
            }
        }
    }
    if (g_moe_expert_cache) {
        print_moe_expert_cache_stats(std::cout);
    }
    return 0;
}

int run_moe_check(
        const std::string & model_path,
        const std::string & config_path,
        int layer,
        const std::vector<int64_t> & token_sizes,
        int reps) {
    if (layer < 0) throw std::runtime_error("--check-moe-layer must be nonnegative");
    if (reps < 1) throw std::runtime_error("--check-moe-reps must be positive");
    if (token_sizes.empty() || std::any_of(token_sizes.begin(), token_sizes.end(),
            [](int64_t value) { return value < 1 || value > 4096; })) {
        throw std::runtime_error("--check-moe-tokens values must be in [1, 4096]");
    }
    auto model_source = mfq::open_model_source(model_path);
    const auto& mfq = *model_source;
    mfq::cuda::validate_model_source(mfq);
    const auto graph = mfq.resolved_model_graph();
    const auto plan = mfq::cuda::cuda_model_plan(graph);
    const auto payload = mfq::cuda::load_model_config_json(
        mfq, config_path);
    FFN ffn;
    int64_t hidden_size = 0;
    switch (plan.backbone) {
        case mfq::cuda::CudaBackbone::gemma4: {
            const auto config =
                mfq::models::gemma4::Config::from_json(payload);
            if (layer >= config.num_hidden_layers) {
                throw std::runtime_error(
                    "MoE benchmark layer is out of range");
            }
            return run_gemma_moe_check(
                mfq, config, layer, token_sizes, reps);
        }
        case mfq::cuda::CudaBackbone::glm_dsa: {
            const auto config =
                mfq::models::glm_dsa::Config::from_json(payload);
            if (layer >= config.num_hidden_layers) {
                throw std::runtime_error(
                    "MoE benchmark layer is out of range");
            }
            mfq::cuda::glm_dsa::load_ffn(
                mfq, config, layer, ffn);
            hidden_size = config.hidden_size;
            break;
        }
        case mfq::cuda::CudaBackbone::deepseek_v4: {
            const auto config =
                mfq::models::deepseek_v4::Config::from_json(payload);
            if (layer >= config.num_hidden_layers) {
                throw std::runtime_error(
                    "MoE benchmark layer is out of range");
            }
            auto block = mfq::cuda::deepseek_v4::load_block(
                mfq, config, layer, "deepseek_v4",
                std::make_shared<Dsv4SharedState>());
            ffn = std::move(static_cast<Dsv4Block&>(*block).ffn);
            hidden_size = config.hidden_size;
            break;
        }
        case mfq::cuda::CudaBackbone::deepseek_v41: {
            const auto config =
                mfq::models::deepseek_v41::Config::from_json(payload);
            if (layer >= config.n_layers) {
                throw std::runtime_error(
                    "MoE benchmark layer is out of range");
            }
            ffn = mfq::cuda::deepseek_v41_runtime::load_moe(
                mfq, config, layer);
            hidden_size = config.hidden;
            break;
        }
        default:
            throw std::runtime_error(
                "selected CUDA backbone has no FFN MoE benchmark adapter");
    }
    if (!ffn.is_moe) {
        throw std::runtime_error(
            "selected layer does not contain MFE MoE weights");
    }
    if (g_moe_expert_cache &&
            !moe_expert_cache_finalized()) {
        finalize_moe_expert_cache();
    }
    const double routed_weight_bytes =
        mfe_weight_bytes(ffn.moe_gate_up) + mfe_weight_bytes(ffn.moe_down);
    std::cout << "moe_bench_config"
              << " layer=" << layer
              << " experts=" << ffn.moe_gate_up.n_experts
              << " top_k=" << ffn.moe_top_k
              << " hidden=" << ffn.moe_gate_up.neuron_len
              << " intermediate=" << ffn.moe_down.neuron_len
              << " gate_up_pools=" << ffn.moe_gate_up.pools.size()
              << " down_pools=" << ffn.moe_down.pools.size()
              << " routed_weight_bytes=" << std::fixed << std::setprecision(0)
              << routed_weight_bytes << "\n";

    mfq_tensor_backend::manual_seed(20260720 + layer);
    mfq_tensor_backend::Tensor output;
    for (int64_t tokens : token_sizes) {
        auto x = mfq_tensor_backend::randn(
            {tokens, hidden_size},
            mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA).dtype(mfq_tensor_backend::kFloat16));
        for (int warmup = 0; warmup < 10; ++warmup) output = ffn.forward(x);
        mfq_cuda_synchronize();

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        auto stream = mfq_get_current_cuda_stream().stream();
        auto wall_start = std::chrono::steady_clock::now();
        cudaEventRecord(start, stream);
        for (int iteration = 0; iteration < reps; ++iteration) output = ffn.forward(x);
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        auto wall_stop = std::chrono::steady_clock::now();
        float elapsed_ms = 0.0f;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        const double cuda_ms = static_cast<double>(elapsed_ms) / reps;
        const double wall_ms =
            std::chrono::duration<double, std::milli>(wall_stop - wall_start).count() / reps;
        const double checksum = output.to(mfq_tensor_backend::kFloat32).sum().item<double>();
        if (!std::isfinite(checksum)) throw std::runtime_error("non-finite MoE benchmark output");
        std::cout << std::setprecision(6)
                  << "moe_bench_result"
                  << " layer=" << layer
                  << " tokens=" << tokens
                  << " reps=" << reps
                  << " cuda_ms=" << cuda_ms
                  << " wall_ms=" << wall_ms
                  << " tokens_per_second=" << (1000.0 * tokens / cuda_ms)
                  << " checksum=" << checksum << "\n";

        const char * prefill_ab_env = std::getenv("MFQ_CHECK_MOE_PREFILL_MMA_AB");
        if (prefill_ab_env != nullptr && std::atoi(prefill_ab_env) != 0 && tokens >= 9) {
            auto candidate = output.clone();
            g_force_moe_prefill_mma_off = true;
            auto baseline = ffn.forward(x);
            mfq_cuda_synchronize();
            g_force_moe_prefill_mma_off = false;
            auto diff = (candidate - baseline).to(mfq_tensor_backend::kFloat32);
            const double baseline_norm = baseline.to(mfq_tensor_backend::kFloat32).norm().item<double>();
            std::cout << "moe_prefill_mma_ab"
                      << " tokens=" << tokens
                      << " equal=" << (candidate.equal(baseline) ? 1 : 0)
                      << " differing=" << candidate.ne(baseline).sum().item<int64_t>()
                      << " rel_l2=" << (diff.norm().item<double>() / baseline_norm)
                      << " mean_abs=" << diff.abs().mean().item<float>()
                      << " max_abs=" << diff.abs().max().item<float>()
                      << "\n";
        }

        const char * exact_env = std::getenv("MFQ_CHECK_MOE_POOL_EXACT");
        if (exact_env != nullptr && std::atoi(exact_env) != 0 && tokens <= 8) {
            auto candidate = output;
            g_force_moe_pool_path = true;
            g_force_moe_unfused_reduce = true;
            g_force_moe_materialized_swiglu = true;
            auto baseline = ffn.forward(x);
            mfq_cuda_synchronize();
            g_force_moe_pool_path = false;
            g_force_moe_unfused_reduce = false;
            g_force_moe_materialized_swiglu = false;
            auto diff = (candidate - baseline).abs().to(mfq_tensor_backend::kFloat32);
            std::cout << "moe_exact_result"
                      << " equal=" << (candidate.equal(baseline) ? 1 : 0)
                      << " differing=" << candidate.ne(baseline).sum().item<int64_t>()
                      << " max_abs=" << diff.max().item<float>()
                      << "\n";
        }

        g_profiler.reset();
        g_profiler.enabled = true;
        const int profile_reps = std::min(reps, 10);
        for (int iteration = 0; iteration < profile_reps; ++iteration) output = ffn.forward(x);
        mfq_cuda_synchronize();
        g_profiler.report("moe_layer" + std::to_string(layer) + "_m" + std::to_string(tokens));
        g_profiler.enabled = false;
        g_profiler.reset();
    }
    if (g_moe_expert_cache) {
        print_moe_expert_cache_stats(std::cout);
    }
    return 0;
}

int run_attention_decode_check(int length, int reps, int D, bool sliding, int window) {
    if (length < 1 || length > 262144) {
        throw std::runtime_error("--check-attention-decode must be in [1, 262144]");
    }
    if (reps < 1) throw std::runtime_error("--check-attention-reps must be positive");
    constexpr int B = 1;
    constexpr int Hq = 16;
    constexpr int max_parts = 256;
    if (D != 256 && D != 512) {
        throw std::runtime_error("--check-attention-head-dim must be 256 or 512");
    }
    if (sliding && D != 256) {
        throw std::runtime_error("SWA decode check currently requires head_dim 256");
    }
    const int Hk = sliding ? 8 : 2;
    const int kv_tile = D == 512 ? 32 : 64;
    const int visible_len = sliding ? std::min(length, window) : length;
    const int max_seq = sliding ? window : (length + kv_tile - 1) / kv_tile * kv_tile;
    const int parts = std::min(max_parts, std::max(1, (length + 127) / 128));
    auto cuda = mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA);
    mfq_tensor_backend::manual_seed(20260720);
    auto q = mfq_tensor_backend::randn({B, Hq, 1, D}, cuda.dtype(mfq_tensor_backend::kFloat32));
    auto k = mfq_tensor_backend::randn({B, Hk, max_seq, D}, cuda.dtype(mfq_tensor_backend::kFloat16));
    auto v = mfq_tensor_backend::randn({B, Hk, max_seq, D}, cuda.dtype(mfq_tensor_backend::kFloat16));
    auto seq_len = mfq_tensor_backend::tensor({length}, cuda.dtype(mfq_tensor_backend::kInt64));
    auto partial_o = mfq_tensor_backend::empty({B * Hq, max_parts, D}, cuda.dtype(mfq_tensor_backend::kFloat32));
    auto partial_m = mfq_tensor_backend::empty({B * Hq, max_parts}, cuda.dtype(mfq_tensor_backend::kFloat32));
    auto partial_l = mfq_tensor_backend::empty({B * Hq, max_parts}, cuda.dtype(mfq_tensor_backend::kFloat32));
    auto qh = q.to(mfq_tensor_backend::kFloat16);
    const int mask_stride = (visible_len + kv_tile - 1) / kv_tile * kv_tile;
    const int ntiles_kv = (visible_len + kv_tile - 1) / kv_tile;
    const int64_t max_blocks = B * Hk * ntiles_kv;
    const int64_t meta_float2 = max_blocks * 8 * (2 + D / 2);
    auto mask = mfq_tensor_backend::empty({B, mask_stride}, cuda.dtype(mfq_tensor_backend::kFloat16));
    auto kv_max = mfq_tensor_backend::empty({B}, cuda.dtype(mfq_tensor_backend::kInt32));
    auto meta = mfq_tensor_backend::empty({2 * meta_float2}, cuda.dtype(mfq_tensor_backend::kFloat32));
    const double scale = 1.0 / std::sqrt((double)D);
    auto run_ref = [&]() {
        if (sliding) {
            return attention_cache_swa_planned_cuda(
                qh, k, v, seq_len, scale, window, visible_len);
        }
        return parts > 1
            ? attention_cache_decode_split_cuda(
                  qh, k, v, seq_len, scale, partial_o, partial_m, partial_l, parts)
            : attention_cache_decode_cuda(qh, k, v, seq_len, scale);
    };
    auto run_test = [&]() {
        if (sliding) {
            return mfq_attention_mma256_swa_decode_cuda(
                q, k, v, seq_len, scale, visible_len, mask, kv_max, meta);
        }
        return D == 512
            ? mfq_attention_mma512_decode_cuda(
                q, k, v, seq_len, scale, length, mask, kv_max, meta)
            : mfq_attention_mma256_decode_cuda(
                q, k, v, seq_len, scale, length, mask, kv_max, meta);
    };
    mfq_tensor_backend::Tensor ref, test;
    for (int i = 0; i < 10; ++i) {
        ref = run_ref();
        test = run_test();
    }
    mfq_cuda_synchronize();
    auto time_ms = [&](auto && fn) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        auto stream = mfq_get_current_cuda_stream().stream();
        cudaEventRecord(start, stream);
        for (int i = 0; i < reps; ++i) fn();
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float elapsed = 0.0f;
        cudaEventElapsedTime(&elapsed, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return elapsed / reps;
    };
    const float ref_ms = time_ms(run_ref);
    const float test_ms = time_ms(run_test);
    ref = run_ref().to(mfq_tensor_backend::kFloat32);
    test = run_test().permute({0, 2, 1, 3}).contiguous();
    mfq_cuda_synchronize();
    auto diff = (test - ref).abs();
    std::cout << "attention_decode_check mode=" << (sliding ? "swa" : "full")
              << " head_dim=" << D << " length=" << length
              << " parts=" << parts << " ref_ms=" << ref_ms
              << " test_ms=" << test_ms
              << " speedup=" << ref_ms / test_ms << "\n";
    std::cout << "attention_decode_rel="
              << ((test - ref).norm() / ref.norm()).item<float>() << "\n";
    std::cout << "attention_decode_mean_abs=" << diff.mean().item<float>() << "\n";
    std::cout << "attention_decode_max_abs=" << diff.max().item<float>() << "\n";
    std::cout << "attention_decode_test_finite="
              << (mfq_tensor_backend::isfinite(test).all().item<bool>() ? 1 : 0) << "\n";
    return 0;
}

int run_gemma4_swa_check(int reps) {
    if (reps < 1) throw std::runtime_error("--check-attention-reps must be positive");
    constexpr int B = 1;
    constexpr int Hq = 32;
    constexpr int Hk = 16;
    constexpr int D = 256;
    const double scale = 1.0 / std::sqrt((double)D);
    auto cuda = mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA);
    mfq_tensor_backend::manual_seed(20260721);

    struct Shape {
        int tokens;
        int window;
    };
    const Shape shapes[] = {
        {1, 1}, {17, 5}, {33, 17}, {73, 32}, {256, 128}, {1024, 1024},
    };
    double worst_rel = 0.0;
    double worst_mean_abs = 0.0;
    double worst_max_abs = 0.0;
    for (const auto shape : shapes) {
        auto q = mfq_tensor_backend::randn({B, Hq, shape.tokens, D}, cuda.dtype(mfq_tensor_backend::kFloat32));
        auto k = mfq_tensor_backend::randn({B, Hk, shape.tokens, D}, cuda.dtype(mfq_tensor_backend::kFloat16));
        auto v = mfq_tensor_backend::randn({B, Hk, shape.tokens, D}, cuda.dtype(mfq_tensor_backend::kFloat16));
        auto ref = attention_swa_cuda(
            q, k.to(mfq_tensor_backend::kFloat32), v.to(mfq_tensor_backend::kFloat32), scale, shape.window);
        auto test = mfq_attention_mma256_swa_cuda(q, k, v, scale, shape.window)
            .permute({0, 2, 1, 3}).contiguous();
        mfq_cuda_synchronize();
        auto diff = (test - ref).abs();
        const double rel = ((test - ref).norm() / ref.norm()).item<double>();
        const double mean_abs = diff.mean().item<double>();
        const double max_abs = diff.max().item<double>();
        const bool finite = mfq_tensor_backend::isfinite(test).all().item<bool>();
        worst_rel = std::max(worst_rel, rel);
        worst_mean_abs = std::max(worst_mean_abs, mean_abs);
        worst_max_abs = std::max(worst_max_abs, max_abs);
        std::cout << "gemma4_swa_check tokens=" << shape.tokens
                  << " window=" << shape.window
                  << " rel=" << rel
                  << " mean_abs=" << mean_abs
                  << " max_abs=" << max_abs
                  << " finite=" << (finite ? 1 : 0) << "\n";
        if (!finite || rel > 0.02) {
            throw std::runtime_error("Gemma4 SWA FlashAttention numerical check failed");
        }
    }

    struct FullAttentionShape {
        int tokens;
        int kv_heads;
    };
    for (const auto shape : {
            FullAttentionShape{33, 4}, FullAttentionShape{256, 4},
            FullAttentionShape{65, 2}, FullAttentionShape{256, 2}}) {
        constexpr int full_hq = 16;
        const int tokens = shape.tokens;
        const int full_hk = shape.kv_heads;
        auto q = mfq_tensor_backend::randn({B, full_hq, tokens, D}, cuda.dtype(mfq_tensor_backend::kFloat32));
        auto k = mfq_tensor_backend::randn({B, full_hk, tokens, D}, cuda.dtype(mfq_tensor_backend::kFloat16));
        auto v = mfq_tensor_backend::randn({B, full_hk, tokens, D}, cuda.dtype(mfq_tensor_backend::kFloat16));
        auto ref = attention_cuda(
            q, k.to(mfq_tensor_backend::kFloat32), v.to(mfq_tensor_backend::kFloat32), scale, true);
        auto test = mfq_attention_mma256_cuda(q, k, v, scale)
            .permute({0, 2, 1, 3}).contiguous();
        mfq_cuda_synchronize();
        auto diff = (test - ref).abs();
        const double rel = ((test - ref).norm() / ref.norm()).item<double>();
        const bool finite = mfq_tensor_backend::isfinite(test).all().item<bool>();
        std::cout << "flash256_full_check tokens=" << tokens
                  << " gqa=" << full_hq / full_hk
                  << " rel=" << rel
                  << " mean_abs=" << diff.mean().item<double>()
                  << " max_abs=" << diff.max().item<double>()
                  << " finite=" << (finite ? 1 : 0) << "\n";
        if (!finite || rel > 0.02) {
            throw std::runtime_error("full FlashAttention numerical regression failed");
        }
    }

    for (const int tokens : {65, 256}) {
        constexpr int full_hq = 16;
        constexpr int full_hk = 2;
        constexpr int full_d = 128;
        const double full_scale = 1.0 / std::sqrt((double)full_d);
        auto q = mfq_tensor_backend::randn(
            {B, full_hq, tokens, full_d},
            cuda.dtype(mfq_tensor_backend::kFloat32));
        auto k = mfq_tensor_backend::randn(
            {B, full_hk, tokens, full_d},
            cuda.dtype(mfq_tensor_backend::kFloat16));
        auto v = mfq_tensor_backend::randn(
            {B, full_hk, tokens, full_d},
            cuda.dtype(mfq_tensor_backend::kFloat16));
        auto ref = attention_cuda(
            q, k.to(mfq_tensor_backend::kFloat32),
            v.to(mfq_tensor_backend::kFloat32), full_scale, true);
        auto test = mfq_attention_mma128_cuda(q, k, v, full_scale)
            .permute({0, 2, 1, 3}).contiguous();
        mfq_cuda_synchronize();
        auto diff = (test - ref).abs();
        const double rel = ((test - ref).norm() / ref.norm()).item<double>();
        const bool finite = mfq_tensor_backend::isfinite(test).all().item<bool>();
        std::cout << "flash128_full_check tokens=" << tokens
                  << " gqa=" << full_hq / full_hk
                  << " rel=" << rel
                  << " mean_abs=" << diff.mean().item<double>()
                  << " max_abs=" << diff.max().item<double>()
                  << " finite=" << (finite ? 1 : 0) << "\n";
        if (!finite || rel > 0.02) {
            throw std::runtime_error(
                "D128 full FlashAttention numerical regression failed");
        }
    }

    for (const int tokens : {256}) {
        constexpr int full_hq = 16;
        constexpr int full_hk = 2;
        constexpr int full_d = 512;
        auto q = mfq_tensor_backend::randn({B, full_hq, tokens, full_d}, cuda.dtype(mfq_tensor_backend::kFloat32));
        auto k = mfq_tensor_backend::randn({B, full_hk, tokens, full_d}, cuda.dtype(mfq_tensor_backend::kFloat16));
        auto v = mfq_tensor_backend::randn({B, full_hk, tokens, full_d}, cuda.dtype(mfq_tensor_backend::kFloat16));
        auto ref = attention_cuda(
            q, k.to(mfq_tensor_backend::kFloat32), v.to(mfq_tensor_backend::kFloat32), 1.0, true);
        auto test = mfq_attention_mma512_cuda(q, k, v, 1.0)
            .permute({0, 2, 1, 3}).contiguous();
        mfq_cuda_synchronize();
        auto diff = (test - ref).abs();
        const double rel = ((test - ref).norm() / ref.norm()).item<double>();
        const bool finite = mfq_tensor_backend::isfinite(test).all().item<bool>();
        std::cout << "gemma4_flash512_check tokens=" << tokens
                  << " rel=" << rel
                  << " mean_abs=" << diff.mean().item<double>()
                  << " max_abs=" << diff.max().item<double>()
                  << " finite=" << (finite ? 1 : 0) << "\n";
        if (!finite || rel > 0.02) {
            throw std::runtime_error("Gemma4 D512 FlashAttention numerical check failed");
        }
    }

    constexpr int bench_tokens = 256;
    constexpr int bench_window = 128;
    auto q = mfq_tensor_backend::randn({B, Hq, bench_tokens, D}, cuda.dtype(mfq_tensor_backend::kFloat32));
    auto k = mfq_tensor_backend::randn({B, Hk, bench_tokens, D}, cuda.dtype(mfq_tensor_backend::kFloat16));
    auto v = mfq_tensor_backend::randn({B, Hk, bench_tokens, D}, cuda.dtype(mfq_tensor_backend::kFloat16));
    auto kf = k.to(mfq_tensor_backend::kFloat32);
    auto vf = v.to(mfq_tensor_backend::kFloat32);
    auto run_ref = [&]() { return attention_swa_cuda(q, kf, vf, scale, bench_window); };
    auto run_test = [&]() {
        return mfq_attention_mma256_swa_cuda(q, k, v, scale, bench_window);
    };
    for (int i = 0; i < 10; ++i) {
        run_ref();
        run_test();
    }
    mfq_cuda_synchronize();
    auto time_ms = [&](auto && fn) {
        cudaEvent_t start, stop;
        MFQ_CUDA_CHECK(cudaEventCreate(&start));
        MFQ_CUDA_CHECK(cudaEventCreate(&stop));
        auto stream = mfq_get_current_cuda_stream().stream();
        MFQ_CUDA_CHECK(cudaEventRecord(start, stream));
        for (int i = 0; i < reps; ++i) fn();
        MFQ_CUDA_CHECK(cudaEventRecord(stop, stream));
        MFQ_CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed = 0.0f;
        MFQ_CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
        MFQ_CUDA_CHECK(cudaEventDestroy(start));
        MFQ_CUDA_CHECK(cudaEventDestroy(stop));
        return elapsed / reps;
    };
    const float ref_ms = time_ms(run_ref);
    const float test_ms = time_ms(run_test);
    std::cout << "gemma4_swa_bench tokens=" << bench_tokens
              << " window=" << bench_window
              << " generic_ms=" << ref_ms
              << " mma_attention_ms=" << test_ms
              << " speedup=" << ref_ms / test_ms << "\n";

    std::cout << "gemma4_swa_worst_rel=" << worst_rel
              << " worst_mean_abs=" << worst_mean_abs
              << " worst_max_abs=" << worst_max_abs << "\n";
    return 0;
}

int run_glm_dsa_check(int reps) {
    if (reps < 1) throw std::runtime_error("--check-attention-reps must be positive");
    mfq_tensor_backend::NoGradGuard no_grad;
    auto cuda = mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA);
    mfq_tensor_backend::manual_seed(20260722);

    {
        constexpr int B = 2, H = 3, T = 5, D = 128, RD = 64;
        auto x = mfq_tensor_backend::randn({B, H, T, D}, cuda.dtype(mfq_tensor_backend::kFloat32));
        auto pos = mfq_tensor_backend::arange(7, 7 + T, cuda.dtype(mfq_tensor_backend::kInt64));
        auto freq = mfq_tensor_backend::pow(
            mfq_tensor_backend::full({RD / 2}, 8000000.0, cuda.dtype(mfq_tensor_backend::kFloat32)),
            -mfq_tensor_backend::arange(0, RD, 2, cuda.dtype(mfq_tensor_backend::kFloat32)) / (double)RD);
        auto angle = mfq_tensor_backend::arange(32, cuda.dtype(mfq_tensor_backend::kFloat32)).unsqueeze(1) * freq.unsqueeze(0);
        auto cos = mfq_tensor_backend::cos(angle).contiguous();
        auto sin = mfq_tensor_backend::sin(angle).contiguous();
        auto test = glm_interleaved_rope_cuda(x, pos, cos, sin, RD);
        auto paired = x.index({Slice(), Slice(), Slice(), Slice(0, RD)})
            .reshape({B, H, T, RD / 2, 2});
        auto c = cos.index_select(0, pos).reshape({1, 1, T, RD / 2});
        auto s = sin.index_select(0, pos).reshape({1, 1, T, RD / 2});
        auto rotated = mfq_tensor_backend::stack({
            paired.select(-1, 0) * c - paired.select(-1, 1) * s,
            paired.select(-1, 1) * c + paired.select(-1, 0) * s,
        }, -1).reshape({B, H, T, RD});
        auto ref = mfq_tensor_backend::cat({
            rotated, x.index({Slice(), Slice(), Slice(), Slice(RD, D)})}, -1);
        const double max_abs = (test - ref).abs().max().item<double>();
        std::cout << "glm_rope_max_abs=" << max_abs << "\n";
        if (max_abs > 2e-6) throw std::runtime_error("GLM interleaved RoPE numerical check failed");
    }

    {
        constexpr int ROWS = 17, D = 128;
        auto x = mfq_tensor_backend::randn({ROWS, D}, cuda.dtype(mfq_tensor_backend::kFloat16));
        auto weight = mfq_tensor_backend::randn({D}, cuda.dtype(mfq_tensor_backend::kFloat32));
        auto bias = mfq_tensor_backend::randn({D}, cuda.dtype(mfq_tensor_backend::kFloat32));
        auto test = glm_dsa_indexer_layer_norm_cuda(x, weight, bias, 1e-5);
        auto ref = mfq_tensor_backend::layer_norm(
            x.to(mfq_tensor_backend::kFloat32), {D}, weight, bias, 1e-5).to(mfq_tensor_backend::kFloat16);
        const double max_abs = (test.to(mfq_tensor_backend::kFloat32) - ref.to(mfq_tensor_backend::kFloat32))
            .abs().max().item<double>();
        std::cout << "glm_indexer_layer_norm_max_abs=" << max_abs << "\n";
        if (max_abs > 0.004) {
            throw std::runtime_error("GLM indexer LayerNorm numerical check failed");
        }
    }

    {
        constexpr int ROWS = 3, EXPERTS = 256, TOPK = 8;
        auto logits = mfq_tensor_backend::randn({ROWS, EXPERTS}, cuda.dtype(mfq_tensor_backend::kFloat32));
        auto bias = mfq_tensor_backend::randn({EXPERTS}, cuda.dtype(mfq_tensor_backend::kFloat32));
        auto selected = moe_topk_cuda(
            logits, TOPK, true, false, true, false, bias, 1e-20, 2.5);
        auto sigmoid = mfq_tensor_backend::sigmoid(logits);
        auto expected_topk = mfq_tensor_backend::topk(
            sigmoid + bias.unsqueeze(0), TOPK, 1, true, true);
        auto expected_ids = std::get<1>(expected_topk);
        auto expected_weights = sigmoid.gather(1, expected_ids);
        expected_weights = expected_weights /
            expected_weights.sum(1, true).clamp_min(1e-20) * 2.5;
        const bool ids_equal = selected[0].to(mfq_tensor_backend::kInt64).equal(expected_ids);
        const double max_abs = (selected[1] - expected_weights)
            .abs().max().item<double>();
        std::cout << "glm_moe_route_ids_equal=" << (ids_equal ? 1 : 0)
                  << " max_abs=" << max_abs << "\n";
        if (!ids_equal || max_abs > 2e-6) {
            throw std::runtime_error("GLM MoE routing numerical check failed");
        }
    }

    {
        constexpr int B = 1, M = 3, K = 2112, H = 32, D = 128;
        auto q = mfq_tensor_backend::randn({B, M, H, D}, cuda.dtype(mfq_tensor_backend::kFloat16));
        auto k = mfq_tensor_backend::randn({B, K, D}, cuda.dtype(mfq_tensor_backend::kFloat16));
        auto weights = mfq_tensor_backend::randn({B, M, H}, cuda.dtype(mfq_tensor_backend::kFloat32));
        const int offset = K - M;
        auto test = glm_dsa_indexer_scores_cuda(q, k, weights, offset, K);
        auto heads = mfq_tensor_backend::einsum(
            "bmhd,bkd->bmhk", {q.to(mfq_tensor_backend::kFloat32), k.to(mfq_tensor_backend::kFloat32)}) /
            std::sqrt(128.0);
        auto ref = (mfq_tensor_backend::relu(heads) * weights.unsqueeze(-1)).sum(2) / std::sqrt(32.0);
        auto key_pos = mfq_tensor_backend::arange(K, cuda.dtype(mfq_tensor_backend::kInt64)).reshape({1, 1, K});
        auto query_pos = mfq_tensor_backend::arange(offset, offset + M, cuda.dtype(mfq_tensor_backend::kInt64)).reshape({1, M, 1});
        ref = ref.masked_fill(key_pos > query_pos, -std::numeric_limits<float>::infinity());
        auto finite = mfq_tensor_backend::isfinite(ref);
        auto diff = mfq_tensor_backend::where(finite, (test - ref).abs(), mfq_tensor_backend::zeros_like(ref));
        const double rel = mfq_tensor_backend::where(finite, test - ref, mfq_tensor_backend::zeros_like(ref)).norm().item<double>() /
            std::max(mfq_tensor_backend::where(finite, ref, mfq_tensor_backend::zeros_like(ref)).norm().item<double>(), 1e-30);
        const double max_abs = diff.max().item<double>();
        std::cout << "glm_indexer_rel=" << rel << " max_abs=" << max_abs << "\n";
        if (!mfq_tensor_backend::isneginf(test.masked_select(~finite)).all().item<bool>() || rel > 0.003) {
            throw std::runtime_error("GLM indexer score numerical check failed");
        }
    }

    {
        constexpr int B = 1, M = 1, VISIBLE = 2112, PLANNED = 2176;
        constexpr int H = 32, D = 128;
        auto q = mfq_tensor_backend::randn({B, M, H, D}, cuda.dtype(mfq_tensor_backend::kFloat16));
        auto k = mfq_tensor_backend::randn({B, PLANNED, D}, cuda.dtype(mfq_tensor_backend::kFloat16));
        auto weights = mfq_tensor_backend::randn({B, M, H}, cuda.dtype(mfq_tensor_backend::kFloat32));
        auto seq_len = mfq_tensor_backend::tensor({VISIBLE}, cuda.dtype(mfq_tensor_backend::kInt64));
        auto test = glm_dsa_indexer_scores_decode_cuda(
            q, k, weights, seq_len, PLANNED);
        auto ref = (mfq_tensor_backend::relu(mfq_tensor_backend::einsum(
            "bmhd,bkd->bmhk",
            {q.to(mfq_tensor_backend::kFloat32), k.to(mfq_tensor_backend::kFloat32)}) / std::sqrt(128.0)) *
            weights.unsqueeze(-1)).sum(2) / std::sqrt(32.0);
        ref.index({Slice(), Slice(), Slice(VISIBLE, PLANNED)})
            .fill_(-std::numeric_limits<float>::infinity());
        auto finite = mfq_tensor_backend::isfinite(ref);
        const double rel = mfq_tensor_backend::where(
            finite, test - ref, mfq_tensor_backend::zeros_like(ref)).norm().item<double>() /
            std::max(mfq_tensor_backend::where(
                finite, ref, mfq_tensor_backend::zeros_like(ref)).norm().item<double>(), 1e-30);
        const bool future_inf = mfq_tensor_backend::isneginf(
            test.index({Slice(), Slice(), Slice(VISIBLE, PLANNED)})).all().item<bool>();
        std::cout << "glm_indexer_decode_rel=" << rel
                  << " future_inf=" << (future_inf ? 1 : 0) << "\n";
        if (rel > 0.003 || !future_inf) {
            throw std::runtime_error("GLM decode indexer numerical check failed");
        }
    }

    {
        constexpr int B = 1, H = 64, T = 33, DQ = 576, DV = 512;
        const double scale = 1.0 / std::sqrt(256.0);
        auto q = mfq_tensor_backend::randn({B, H, T, DQ}, cuda.dtype(mfq_tensor_backend::kFloat32));
        auto kv = mfq_tensor_backend::randn({B, 1, T, DQ}, cuda.dtype(mfq_tensor_backend::kFloat16));
        auto kv_cache = mfq_tensor_backend::zeros({B, 1, 64, DQ}, cuda.dtype(mfq_tensor_backend::kFloat16));
        kv_cache.index({Slice(), Slice(), Slice(0, T), Slice()}).copy_(kv);
        auto mask = mfq_tensor_backend::empty({T, 64}, cuda.dtype(mfq_tensor_backend::kFloat16));
        auto kv_max = mfq_tensor_backend::empty({B * ((T + 3) / 4)}, cuda.dtype(mfq_tensor_backend::kInt32));
        auto meta = mfq_tensor_backend::empty({8 * 1024 * 1024}, cuda.dtype(mfq_tensor_backend::kFloat32));
        auto test = attention_glm_mla576_cached_cuda(
            q, kv_cache, T, mask, kv_max, meta, scale);
        auto q_ref = q.to(mfq_tensor_backend::kFloat16).to(mfq_tensor_backend::kFloat32);
        auto k_ref = kv.to(mfq_tensor_backend::kFloat32).expand({B, H, T, DQ});
        auto scores = mfq_tensor_backend::matmul(q_ref, k_ref.transpose(-1, -2)) * scale;
        auto causal = mfq_tensor_backend::ones({T, T}, cuda.dtype(mfq_tensor_backend::kBool)).tril();
        scores = scores.masked_fill(~causal, -std::numeric_limits<float>::infinity());
        auto ref = mfq_tensor_backend::matmul(
            mfq_tensor_backend::softmax(scores, -1),
            k_ref.index({Slice(), Slice(), Slice(), Slice(0, DV)}))
            .permute({0, 2, 1, 3}).contiguous();
        const double rel = (test - ref).norm().item<double>() / ref.norm().item<double>();
        const double max_abs = (test - ref).abs().max().item<double>();
        std::cout << "glm_dense_mla_rel=" << rel << " max_abs=" << max_abs << "\n";
        if (!mfq_tensor_backend::isfinite(test).all().item<bool>() || rel > 0.02) {
            throw std::runtime_error("GLM dense MLA numerical check failed");
        }
    }

    {
        constexpr int B = 1, H = 64, VISIBLE = 33, PLANNED = 64;
        constexpr int DQ = 576, DV = 512;
        const double scale = 1.0 / std::sqrt(256.0);
        auto q = mfq_tensor_backend::randn({B, H, 1, DQ}, cuda.dtype(mfq_tensor_backend::kFloat32));
        auto kv = mfq_tensor_backend::randn({B, 1, PLANNED, DQ}, cuda.dtype(mfq_tensor_backend::kFloat16));
        auto seq_len = mfq_tensor_backend::tensor({VISIBLE}, cuda.dtype(mfq_tensor_backend::kInt64));
        auto mask = mfq_tensor_backend::empty({B, PLANNED}, cuda.dtype(mfq_tensor_backend::kFloat16));
        auto kv_max = mfq_tensor_backend::empty({B}, cuda.dtype(mfq_tensor_backend::kInt32));
        auto meta = mfq_tensor_backend::empty({8 * 1024 * 1024}, cuda.dtype(mfq_tensor_backend::kFloat32));
        auto test = attention_glm_mla576_decode_cuda(
            q, kv, seq_len, scale, PLANNED, mask, kv_max, meta);
        auto selected = kv.index({Slice(), Slice(), Slice(0, VISIBLE), Slice()})
            .to(mfq_tensor_backend::kFloat32).expand({B, H, VISIBLE, DQ});
        auto ref = mfq_tensor_backend::matmul(
            mfq_tensor_backend::softmax(mfq_tensor_backend::matmul(q, selected.transpose(-1, -2)) * scale, -1),
            selected.index({Slice(), Slice(), Slice(), Slice(0, DV)}))
            .permute({0, 2, 1, 3}).contiguous();
        const double rel = (test - ref).norm().item<double>() /
            std::max(ref.norm().item<double>(), 1e-30);
        std::cout << "glm_dense_decode_rel=" << rel << "\n";
        if (!mfq_tensor_backend::isfinite(test).all().item<bool>() || rel > 0.02) {
            throw std::runtime_error("GLM dense decode numerical check failed");
        }
    }

    constexpr int B = 1, H = 64, M = 3, K = 2112, TOPK = 2048, DQ = 576, DV = 512;
    const double scale = 1.0 / std::sqrt(256.0);
    auto q = mfq_tensor_backend::randn({B, H, M, DQ}, cuda.dtype(mfq_tensor_backend::kFloat32));
    auto kv = mfq_tensor_backend::randn({B, K, DQ}, cuda.dtype(mfq_tensor_backend::kFloat16));
    std::vector<mfq_tensor_backend::Tensor> rows;
    rows.reserve(M);
    for (int row = 0; row < M; ++row) {
        rows.push_back(mfq_tensor_backend::randperm(K - M + row + 1, cuda.dtype(mfq_tensor_backend::kInt64))
            .narrow(0, 0, TOPK).to(mfq_tensor_backend::kInt32));
    }
    auto indices = mfq_tensor_backend::stack(rows, 0).unsqueeze(0).contiguous();
    auto meta = mfq_tensor_backend::empty({8 * 1024 * 1024}, cuda.dtype(mfq_tensor_backend::kFloat32));
    auto run_sparse = [&]() {
        return attention_glm_mla_sparse_cuda(q, kv, indices, meta, scale);
    };
    auto test = run_sparse();
    std::vector<mfq_tensor_backend::Tensor> refs;
    refs.reserve(M);
    auto q_ref = q.to(mfq_tensor_backend::kFloat16).to(mfq_tensor_backend::kFloat32);
    for (int row = 0; row < M; ++row) {
        auto idx = indices.index({0, row}).to(mfq_tensor_backend::kInt64);
        auto selected = kv.index_select(1, idx).index({0}).to(mfq_tensor_backend::kFloat32);
        auto score = mfq_tensor_backend::matmul(q_ref.index({0, Slice(), row}), selected.transpose(0, 1)) * scale;
        refs.push_back(mfq_tensor_backend::matmul(
            mfq_tensor_backend::softmax(score, -1), selected.index({Slice(), Slice(0, DV)})));
    }
    auto ref = mfq_tensor_backend::stack(refs, 0).unsqueeze(0);
    mfq_cuda_synchronize();
    const double rel = (test - ref).norm().item<double>() / ref.norm().item<double>();
    const double max_abs = (test - ref).abs().max().item<double>();
    std::cout << "glm_sparse_mla_rel=" << rel << " max_abs=" << max_abs << "\n";
    if (!mfq_tensor_backend::isfinite(test).all().item<bool>() || rel > 0.02) {
        throw std::runtime_error("GLM sparse MLA numerical check failed");
    }

    auto time_ms = [&](auto && fn) {
        cudaEvent_t start, stop;
        MFQ_CUDA_CHECK(cudaEventCreate(&start));
        MFQ_CUDA_CHECK(cudaEventCreate(&stop));
        auto stream = mfq_get_current_cuda_stream().stream();
        MFQ_CUDA_CHECK(cudaEventRecord(start, stream));
        for (int i = 0; i < reps; ++i) fn();
        MFQ_CUDA_CHECK(cudaEventRecord(stop, stream));
        MFQ_CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed = 0.0f;
        MFQ_CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
        MFQ_CUDA_CHECK(cudaEventDestroy(start));
        MFQ_CUDA_CHECK(cudaEventDestroy(stop));
        return elapsed / reps;
    };
    for (int i = 0; i < 5; ++i) run_sparse();
    const float sparse_ms = time_ms(run_sparse);
    std::cout << "glm_sparse_mla_m=" << M << " k=" << K
              << " topk=" << TOPK << " cuda_ms=" << sparse_ms << "\n";
    return 0;
}

int run_dsv4_hc_check(int reps) {
    if (reps < 1) {
        throw std::runtime_error("--check-attention-reps must be positive");
    }
    mfq_tensor_backend::NoGradGuard no_grad;
    auto cuda = mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA);
    constexpr int64_t hc = 4;
    constexpr int64_t hidden = 4096;
    constexpr int64_t mix_width = 24;
    constexpr double eps = 1e-6;

    auto x_sequence = mfq_tensor_backend::arange(
        hc * hidden, cuda.dtype(mfq_tensor_backend::kFloat32));
    auto x = (
        (x_sequence.remainder(251) - 125.0) / 64.0 +
        0.125 * mfq_tensor_backend::sin(x_sequence * 0.015625))
        .to(mfq_tensor_backend::kFloat16)
        .reshape({1, 1, hc, hidden})
        .contiguous();
    auto function_sequence = mfq_tensor_backend::arange(
        mix_width * hc * hidden, cuda.dtype(mfq_tensor_backend::kFloat32));
    auto function = (
        0.003 * mfq_tensor_backend::sin(function_sequence * 0.001953125) +
        0.001 * mfq_tensor_backend::cos(function_sequence * 0.0078125))
        .reshape({mix_width, hc * hidden})
        .contiguous();
    auto scale = (
        0.7 + 0.2 * mfq_tensor_backend::arange(3, cuda.dtype(mfq_tensor_backend::kFloat32)))
        .contiguous();
    auto base = (
        0.1 * mfq_tensor_backend::sin(
            mfq_tensor_backend::arange(
                mix_width, cuda.dtype(mfq_tensor_backend::kFloat32)) * 0.3125))
        .contiguous();
    auto flat = x.flatten(2).to(mfq_tensor_backend::kFloat32);
    auto inverse_rms = mfq_tensor_backend::rsqrt(
        flat.square().mean(-1, true) + eps);
    auto mixes = (
        mfq_tensor_backend::matmul(flat, function.transpose(0, 1)) * inverse_rms)
        .contiguous();

    auto reference_split = dsv4_hc_split_sinkhorn(
        mixes, scale, base, hc, 20, eps);
    auto reference_reduced = (
        reference_split.at(0).unsqueeze(-1) *
        flat.reshape({1, 1, hc, hidden}))
        .sum(2).to(mfq_tensor_backend::kFloat16).contiguous();
    auto candidate = dsv4_hc_pre_cuda(
        x, mixes, scale, base, 20, eps);

    auto direct = x.index(
        {Slice(), Slice(), 0, Slice()}).contiguous();
    auto reference_post = (
        reference_split.at(1).unsqueeze(-1) *
            direct.unsqueeze(-2) +
        (reference_split.at(2).unsqueeze(-1) *
            x.to(mfq_tensor_backend::kFloat32).unsqueeze(-2)).sum(2))
        .to(mfq_tensor_backend::kFloat16).contiguous();
    auto candidate_post = dsv4_hc_post_cuda(
        direct, x, candidate.at(1), candidate.at(2));
    mfq_cuda_synchronize();

    auto compare = [](const char * name,
                      mfq_tensor_backend::Tensor reference,
                      mfq_tensor_backend::Tensor value) {
        auto reference_f32 = reference.to(mfq_tensor_backend::kFloat32);
        auto value_f32 = value.to(mfq_tensor_backend::kFloat32);
        auto diff = value_f32 - reference_f32;
        const double norm = reference_f32.norm().item<double>();
        std::cout << std::scientific << std::setprecision(9)
                  << "dsv4_hc_ab tensor=" << name
                  << " equal=" << (value.equal(reference) ? 1 : 0)
                  << " differing="
                  << value.ne(reference).sum().item<int64_t>()
                  << " rel_l2="
                  << (norm == 0.0
                      ? 0.0
                      : diff.norm().item<double>() / norm)
                  << " mean_abs=" << diff.abs().mean().item<double>()
                  << " max_abs=" << diff.abs().max().item<double>()
                  << "\n";
    };
    compare("reduced", reference_reduced, candidate.at(0));
    compare("post", reference_split.at(1), candidate.at(1));
    compare("combination", reference_split.at(2), candidate.at(2));
    compare("hc_post", reference_post, candidate_post);

    auto time_ms = [&](auto && fn) {
        for (int warmup = 0; warmup < 10; ++warmup) fn();
        mfq_cuda_synchronize();
        cudaEvent_t begin, end;
        MFQ_CUDA_CHECK(cudaEventCreate(&begin));
        MFQ_CUDA_CHECK(cudaEventCreate(&end));
        auto stream = mfq_get_current_cuda_stream().stream();
        MFQ_CUDA_CHECK(cudaEventRecord(begin, stream));
        for (int iteration = 0; iteration < reps; ++iteration) fn();
        MFQ_CUDA_CHECK(cudaEventRecord(end, stream));
        MFQ_CUDA_CHECK(cudaEventSynchronize(end));
        float elapsed = 0.0f;
        MFQ_CUDA_CHECK(cudaEventElapsedTime(&elapsed, begin, end));
        MFQ_CUDA_CHECK(cudaEventDestroy(begin));
        MFQ_CUDA_CHECK(cudaEventDestroy(end));
        return elapsed / reps;
    };
    const float reference_pre_ms = time_ms([&]() {
        auto split = dsv4_hc_split_sinkhorn(
            mixes, scale, base, hc, 20, eps);
        return (
            split.at(0).unsqueeze(-1) *
            flat.reshape({1, 1, hc, hidden}))
            .sum(2).to(mfq_tensor_backend::kFloat16).contiguous();
    });
    const float candidate_pre_ms = time_ms([&]() {
        return dsv4_hc_pre_cuda(
            x, mixes, scale, base, 20, eps);
    });
    const float reference_post_ms = time_ms([&]() {
        return (
            reference_split.at(1).unsqueeze(-1) *
                direct.unsqueeze(-2) +
            (reference_split.at(2).unsqueeze(-1) *
                x.to(mfq_tensor_backend::kFloat32).unsqueeze(-2)).sum(2))
            .to(mfq_tensor_backend::kFloat16).contiguous();
    });
    const float candidate_post_ms = time_ms([&]() {
        return dsv4_hc_post_cuda(
            direct, x, candidate.at(1), candidate.at(2));
    });
    std::cout << std::fixed << std::setprecision(9)
              << "dsv4_hc_timing"
              << " reference_pre_ms=" << reference_pre_ms
              << " candidate_pre_ms=" << candidate_pre_ms
              << " pre_speedup=" << reference_pre_ms / candidate_pre_ms
              << " reference_post_ms=" << reference_post_ms
              << " candidate_post_ms=" << candidate_post_ms
              << " post_speedup=" << reference_post_ms / candidate_post_ms
              << "\n";
    return 0;
}

int run_dsv4_attention_check(int reps) {
    if (reps < 1) {
        throw std::runtime_error("--check-attention-reps must be positive");
    }
    mfq_tensor_backend::NoGradGuard no_grad;
    auto cuda = mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA);
    mfq_tensor_backend::manual_seed(20260723);

    {
        constexpr int EXPERTS = 256, TOPK = 6;
        auto logits = mfq_tensor_backend::randn({1, EXPERTS}, cuda.dtype(mfq_tensor_backend::kFloat32)) * 3.0;
        auto bias = mfq_tensor_backend::randn({EXPERTS}, cuda.dtype(mfq_tensor_backend::kFloat32)) * 0.05;
        auto selected = moe_topk_cuda(
            logits, TOPK, false, true, true, false, bias, 1e-20, 1.5);
        auto transformed = mfq_tensor_backend::sqrt(mfq_tensor_backend::where(
            logits > 20.0, logits, mfq_tensor_backend::log1p(mfq_tensor_backend::exp(logits))));
        auto expected_topk = mfq_tensor_backend::topk(
            transformed + bias.unsqueeze(0), TOPK, 1, true, true);
        auto expected_ids = std::get<1>(expected_topk);
        auto expected_weights = transformed.gather(1, expected_ids);
        expected_weights = expected_weights /
            expected_weights.sum(1, true).clamp_min(1e-20) * 1.5;
        const bool ids_equal =
            selected.at(0).to(mfq_tensor_backend::kInt64).equal(expected_ids);
        const double max_abs = (selected.at(1) - expected_weights)
            .abs().max().item<double>();

        auto hash_ids = mfq_tensor_backend::randint(
            0, EXPERTS, {7, TOPK}, cuda.dtype(mfq_tensor_backend::kInt32));
        auto hash_logits =
            mfq_tensor_backend::randn({7, EXPERTS}, cuda.dtype(mfq_tensor_backend::kFloat32)) * 3.0;
        auto hash_weights = moe_sqrtsoftplus_weights_cuda(
            hash_logits, hash_ids, 1e-20, 1.5);
        auto hash_transformed = mfq_tensor_backend::sqrt(mfq_tensor_backend::where(
            hash_logits > 20.0, hash_logits,
            mfq_tensor_backend::log1p(mfq_tensor_backend::exp(hash_logits))));
        auto expected_hash = hash_transformed.gather(
            1, hash_ids.to(mfq_tensor_backend::kInt64));
        expected_hash = expected_hash /
            expected_hash.sum(1, true).clamp_min(1e-20) * 1.5;
        const double hash_max_abs = (hash_weights - expected_hash)
            .abs().max().item<double>();
        std::cout << "dsv4_moe_route_ids_equal=" << (ids_equal ? 1 : 0)
                  << " max_abs=" << max_abs
                  << " hash_max_abs=" << hash_max_abs << "\n";
        if (!ids_equal || max_abs > 2e-6 || hash_max_abs > 2e-6) {
            throw std::runtime_error(
                "DeepSeek V4 MoE routing numerical check failed");
        }
    }

    constexpr int B = 1;
    constexpr int D = 512;
    constexpr int RD = 64;
    constexpr double eps = 1e-6;
    auto norm = mfq_tensor_backend::randn({D}, cuda.dtype(mfq_tensor_backend::kFloat32));

    {
        constexpr int W = 2, R = 128;
        auto kv = mfq_tensor_backend::randn({B, W, R, D}, cuda.dtype(mfq_tensor_backend::kFloat16));
        auto gate = mfq_tensor_backend::randn({B, W, R, D}, cuda.dtype(mfq_tensor_backend::kFloat16));
        auto ape = mfq_tensor_backend::randn({R, D}, cuda.dtype(mfq_tensor_backend::kFloat32));
        auto empty = mfq_tensor_backend::empty({0}, cuda.dtype(mfq_tensor_backend::kFloat16));
        auto positions = mfq_tensor_backend::arange(W, cuda.dtype(mfq_tensor_backend::kInt64)).reshape({B, W});
        auto cos = mfq_tensor_backend::ones({W + 1, RD / 2}, cuda.dtype(mfq_tensor_backend::kFloat32));
        auto sin = mfq_tensor_backend::zeros_like(cos);
        auto test = dsv4_compress_cuda(
            kv, gate, ape, norm, empty, empty, positions, cos, sin,
            R, false, 1, eps);
        auto score = gate.to(mfq_tensor_backend::kFloat32) +
            ape.reshape({1, 1, R, D});
        auto pooled = (kv.to(mfq_tensor_backend::kFloat32) *
            mfq_tensor_backend::softmax(score, 2)).sum(2);
        auto ref = pooled * mfq_tensor_backend::rsqrt(
            pooled.square().mean(-1, true) + eps) * norm;
        ref = ref.to(mfq_tensor_backend::kBFloat16).to(mfq_tensor_backend::kFloat32);
        auto ref_nope = ref.slice(-1, 0, D - RD)
            .reshape({-1, (D - RD) / 64, 64});
        auto fp8_scale = ref_nope.abs().amax(-1, true)
            .clamp_min(1e-4) / 448.0;
        ref_nope = (ref_nope / fp8_scale)
            .clamp(-448.0, 448.0)
            .to(mfq_float8_e4m3fn)
            .to(mfq_tensor_backend::kFloat32) * fp8_scale;
        ref.slice(-1, 0, D - RD).copy_(
            ref_nope.reshape({B, W, D - RD}));
        ref = ref.to(mfq_tensor_backend::kBFloat16)
            .to(mfq_tensor_backend::kFloat16).to(mfq_tensor_backend::kFloat32);
        const double rel =
            (test.to(mfq_tensor_backend::kFloat32) - ref).norm().item<double>() /
            std::max(ref.norm().item<double>(), 1e-30);
        std::cout << "dsv4_hca_compressor_rel=" << rel << "\n";
        if (rel > 0.002) {
            throw std::runtime_error(
                "DeepSeek V4 ratio-128 compressor numerical check failed");
        }
    }

    {
        constexpr int W = 3, R = 4, OD = 2 * D;
        auto kv = mfq_tensor_backend::randn({B, W, R, OD}, cuda.dtype(mfq_tensor_backend::kFloat16));
        auto gate = mfq_tensor_backend::randn({B, W, R, OD}, cuda.dtype(mfq_tensor_backend::kFloat16));
        auto prev_kv = mfq_tensor_backend::randn({B, R, D}, cuda.dtype(mfq_tensor_backend::kFloat16));
        auto prev_gate = mfq_tensor_backend::randn({B, R, D}, cuda.dtype(mfq_tensor_backend::kFloat16));
        auto ape = mfq_tensor_backend::randn({R, OD}, cuda.dtype(mfq_tensor_backend::kFloat32));
        auto positions = mfq_tensor_backend::arange(W, cuda.dtype(mfq_tensor_backend::kInt64)).reshape({B, W});
        auto cos = mfq_tensor_backend::ones({W + 1, RD / 2}, cuda.dtype(mfq_tensor_backend::kFloat32));
        auto sin = mfq_tensor_backend::zeros_like(cos);
        auto test = dsv4_compress_cuda(
            kv, gate, ape, norm, prev_kv, prev_gate, positions,
            cos, sin, R, true, 0, eps);
        std::vector<mfq_tensor_backend::Tensor> ref_rows;
        for (int w = 0; w < W; ++w) {
            auto left_kv = w == 0
                ? prev_kv
                : kv.index({Slice(), w - 1, Slice(), Slice(0, D)});
            auto left_gate = w == 0
                ? prev_gate
                : gate.index({Slice(), w - 1, Slice(), Slice(0, D)});
            auto right_kv =
                kv.index({Slice(), w, Slice(), Slice(D, OD)});
            auto right_gate =
                gate.index({Slice(), w, Slice(), Slice(D, OD)});
            auto values = mfq_tensor_backend::cat({left_kv, right_kv}, 1)
                .to(mfq_tensor_backend::kFloat32);
            auto score = mfq_tensor_backend::cat({
                left_gate.to(mfq_tensor_backend::kFloat32) +
                    ape.index({Slice(), Slice(0, D)}).unsqueeze(0),
                right_gate.to(mfq_tensor_backend::kFloat32) +
                    ape.index({Slice(), Slice(D, OD)}).unsqueeze(0)}, 1);
            auto pooled = (values * mfq_tensor_backend::softmax(score, 1)).sum(1);
            ref_rows.push_back(
                pooled * mfq_tensor_backend::rsqrt(
                    pooled.square().mean(-1, true) + eps) * norm);
        }
        auto ref = mfq_tensor_backend::stack(ref_rows, 1)
            .to(mfq_tensor_backend::kFloat16).to(mfq_tensor_backend::kFloat32);
        const double rel =
            (test.to(mfq_tensor_backend::kFloat32) - ref).norm().item<double>() /
            std::max(ref.norm().item<double>(), 1e-30);
        std::cout << "dsv4_csa_overlap_compressor_rel=" << rel << "\n";
        if (rel > 0.002) {
            throw std::runtime_error(
                "DeepSeek V4 ratio-4 overlap compressor numerical check failed");
        }
    }

    {
        constexpr int T = 12, R = 4, W = T / R, OD = 2 * D;
        auto kv = mfq_tensor_backend::randn({B, T, OD}, cuda.dtype(mfq_tensor_backend::kFloat32));
        auto gate = mfq_tensor_backend::randn({B, T, OD}, cuda.dtype(mfq_tensor_backend::kFloat32));
        auto ape = mfq_tensor_backend::randn({R, OD}, cuda.dtype(mfq_tensor_backend::kFloat32));
        auto empty = mfq_tensor_backend::empty({0}, cuda.dtype(mfq_tensor_backend::kFloat16));
        auto positions = mfq_tensor_backend::arange(W, cuda.dtype(mfq_tensor_backend::kInt64)).reshape({B, W});
        auto cos = mfq_tensor_backend::ones({W + 1, RD / 2}, cuda.dtype(mfq_tensor_backend::kFloat32));
        auto sin = mfq_tensor_backend::zeros_like(cos);
        auto batch = dsv4_compress_cuda(
            kv.reshape({B, W, R, OD}).contiguous(),
            gate.reshape({B, W, R, OD}).contiguous(),
            ape, norm, empty, empty, positions, cos, sin,
            R, true, 1, eps);
        auto state_kv = mfq_tensor_backend::zeros(
            {B, R, OD}, cuda.dtype(mfq_tensor_backend::kFloat32));
        auto state_gate = mfq_tensor_backend::zeros_like(state_kv);
        auto prev_kv = mfq_tensor_backend::zeros(
            {B, R, D}, cuda.dtype(mfq_tensor_backend::kFloat32));
        auto prev_gate = mfq_tensor_backend::zeros_like(prev_kv);
        auto pool = mfq_tensor_backend::zeros(
            {B, W, D}, cuda.dtype(mfq_tensor_backend::kFloat16));
        auto seq_len = mfq_tensor_backend::zeros({B}, cuda.dtype(mfq_tensor_backend::kInt64));
        for (int t = 0; t < T; ++t) {
            seq_len.fill_(t + 1);
            dsv4_decode_pool_update_cuda(
                kv.narrow(1, t, 1).contiguous(),
                gate.narrow(1, t, 1).contiguous(),
                ape, norm, state_kv, state_gate, prev_kv, prev_gate,
                pool, seq_len, cos, sin, R, true, 1, eps);
        }
        const double rel =
            (pool.to(mfq_tensor_backend::kFloat32) - batch.to(mfq_tensor_backend::kFloat32))
                .norm().item<double>() /
            std::max(batch.to(mfq_tensor_backend::kFloat32).norm().item<double>(), 1e-30);
        std::cout << "dsv4_decode_pool_state_rel=" << rel << "\n";
        if (rel > 0.002) {
            throw std::runtime_error(
                "DeepSeek V4 decode compressor state check failed");
        }
    }

    {
        constexpr int ID = 128, T = 12, R = 4, W = T / R;
        constexpr int OD = 2 * ID;
        auto index_norm = mfq_tensor_backend::randn(
            {ID}, cuda.dtype(mfq_tensor_backend::kFloat32));
        auto kv = mfq_tensor_backend::randn(
            {B, T, OD}, cuda.dtype(mfq_tensor_backend::kFloat32));
        auto gate = mfq_tensor_backend::randn(
            {B, T, OD}, cuda.dtype(mfq_tensor_backend::kFloat32));
        auto ape = mfq_tensor_backend::randn(
            {R, OD}, cuda.dtype(mfq_tensor_backend::kFloat32));
        auto empty = mfq_tensor_backend::empty({0}, cuda.dtype(mfq_tensor_backend::kFloat16));
        auto positions = mfq_tensor_backend::arange(
            W, cuda.dtype(mfq_tensor_backend::kInt64)).reshape({B, W});
        auto cos = mfq_tensor_backend::ones(
            {W + 1, RD / 2}, cuda.dtype(mfq_tensor_backend::kFloat32));
        auto sin = mfq_tensor_backend::zeros_like(cos);
        auto batch = dsv4_compress_cuda(
            kv.reshape({B, W, R, OD}).contiguous(),
            gate.reshape({B, W, R, OD}).contiguous(),
            ape, index_norm, empty, empty, positions, cos, sin,
            R, true, 2, eps);
        auto state_kv = mfq_tensor_backend::zeros(
            {B, R, OD}, cuda.dtype(mfq_tensor_backend::kFloat32));
        auto state_gate = mfq_tensor_backend::zeros_like(state_kv);
        auto prev_kv = mfq_tensor_backend::zeros(
            {B, R, ID}, cuda.dtype(mfq_tensor_backend::kFloat32));
        auto prev_gate = mfq_tensor_backend::zeros_like(prev_kv);
        auto pool = mfq_tensor_backend::zeros(
            {B, W, ID}, cuda.dtype(mfq_tensor_backend::kFloat16));
        auto seq_len = mfq_tensor_backend::zeros(
            {B}, cuda.dtype(mfq_tensor_backend::kInt64));
        for (int t = 0; t < T; ++t) {
            seq_len.fill_(t + 1);
            dsv4_decode_pool_update_cuda(
                kv.narrow(1, t, 1).contiguous(),
                gate.narrow(1, t, 1).contiguous(),
                ape, index_norm, state_kv, state_gate,
                prev_kv, prev_gate, pool, seq_len, cos, sin,
                R, true, 2, eps);
        }
        const double rel =
            (pool.to(mfq_tensor_backend::kFloat32) - batch.to(mfq_tensor_backend::kFloat32))
                .norm().item<double>() /
            std::max(
                batch.to(mfq_tensor_backend::kFloat32).norm().item<double>(),
                1e-30);
        std::cout << "dsv4_indexer_pool_state_rel=" << rel << "\n";
        if (rel > 0.002) {
            throw std::runtime_error(
                "DeepSeek V4 indexer compressor state check failed");
        }
    }

    {
        constexpr int ID = 128, T = 14, R = 4, OD = 2 * ID;
        auto index_norm = mfq_tensor_backend::randn(
            {ID}, cuda.dtype(mfq_tensor_backend::kFloat32));
        auto kv = mfq_tensor_backend::randn(
            {B, T, OD}, cuda.dtype(mfq_tensor_backend::kFloat32));
        auto gate = mfq_tensor_backend::randn(
            {B, T, OD}, cuda.dtype(mfq_tensor_backend::kFloat32));
        Dsv4RopeTable rope;
        rope.cos = mfq_tensor_backend::ones(
            {T + 1, RD / 2}, cuda.dtype(mfq_tensor_backend::kFloat32));
        rope.sin = mfq_tensor_backend::zeros_like(rope.cos);
        rope.negative_sin = -rope.sin;

        Dsv4PoolState batched;
        batched.ratio = R;
        batched.head_dim = ID;
        batched.overlap = true;
        batched.cache_quant_mode = 2;
        batched.ape = mfq_tensor_backend::randn(
            {R, OD}, cuda.dtype(mfq_tensor_backend::kFloat32));
        batched.norm = index_norm;
        batched.reset(B, T);

        Dsv4PoolState decoded;
        decoded.ratio = batched.ratio;
        decoded.head_dim = batched.head_dim;
        decoded.overlap = batched.overlap;
        decoded.cache_quant_mode = batched.cache_quant_mode;
        decoded.ape = batched.ape;
        decoded.norm = batched.norm;
        decoded.reset(B, T);

        const int64_t windows = batched.prefill(kv, gate, rope);
        auto seq_len = mfq_tensor_backend::zeros(
            {B}, cuda.dtype(mfq_tensor_backend::kInt64));
        for (int t = 0; t < T; ++t) {
            seq_len.fill_(t + 1);
            decoded.update(
                kv.narrow(1, t, 1).contiguous(),
                gate.narrow(1, t, 1).contiguous(),
                seq_len, rope);
        }
        const auto pool_width = T / R;
        auto reference_pool = decoded.pool.narrow(1, 0, pool_width);
        auto candidate_pool = batched.pool.narrow(1, 0, pool_width);
        const double pool_rel =
            (candidate_pool.to(mfq_tensor_backend::kFloat32) -
             reference_pool.to(mfq_tensor_backend::kFloat32)).norm().item<double>() /
            std::max(
                reference_pool.to(mfq_tensor_backend::kFloat32).norm().item<double>(),
                1e-30);
        const double state_rel =
            (batched.state_kv - decoded.state_kv)
                .norm().item<double>() /
            std::max(decoded.state_kv.norm().item<double>(), 1e-30);
        const double gate_rel =
            (batched.state_gate - decoded.state_gate)
                .norm().item<double>() /
            std::max(decoded.state_gate.norm().item<double>(), 1e-30);
        const double previous_rel =
            (batched.previous_kv - decoded.previous_kv)
                .norm().item<double>() /
            std::max(decoded.previous_kv.norm().item<double>(), 1e-30);
        const double previous_gate_rel =
            (batched.previous_gate - decoded.previous_gate)
                .norm().item<double>() /
            std::max(decoded.previous_gate.norm().item<double>(), 1e-30);
        std::cout << "dsv4_pool_prefill_windows=" << windows
                  << " pool_rel=" << pool_rel
                  << " state_rel=" << state_rel
                  << " gate_rel=" << gate_rel
                  << " previous_rel=" << previous_rel
                  << " previous_gate_rel=" << previous_gate_rel << "\n";
        if (windows != pool_width || pool_rel > 0.002 ||
            state_rel > 1e-7 || gate_rel > 1e-7 ||
            previous_rel > 1e-7 || previous_gate_rel > 1e-7) {
            throw std::runtime_error(
                "DeepSeek V4 batched compressor prefill state check failed");
        }
    }

    {
        constexpr int ROWS = 9, WIDTH = 128;
        auto input = mfq_tensor_backend::randn(
            {ROWS, WIDTH}, cuda.dtype(mfq_tensor_backend::kFloat16)) * 2.0;
        auto test = dsv4_fp4_sim_cuda(input.contiguous());
        auto grouped = input.to(mfq_tensor_backend::kFloat32)
            .reshape({ROWS, WIDTH / 32, 32});
        auto scale = mfq_tensor_backend::exp2(mfq_tensor_backend::ceil(mfq_tensor_backend::log2(
            grouped.abs().amax(-1, true)
                .clamp_min(6.0 * std::ldexp(1.0, -126)) / 6.0)));
        auto normalized = (grouped / scale).clamp(-6.0, 6.0);
        auto magnitude = normalized.abs();
        auto quantized = mfq_tensor_backend::where(
            magnitude <= 0.25, mfq_tensor_backend::zeros_like(magnitude),
            mfq_tensor_backend::where(
                magnitude < 0.75, mfq_tensor_backend::full_like(magnitude, 0.5),
                mfq_tensor_backend::where(
                    magnitude <= 1.25, mfq_tensor_backend::ones_like(magnitude),
                    mfq_tensor_backend::where(
                        magnitude < 1.75,
                        mfq_tensor_backend::full_like(magnitude, 1.5),
                        mfq_tensor_backend::where(
                            magnitude <= 2.5,
                            mfq_tensor_backend::full_like(magnitude, 2.0),
                            mfq_tensor_backend::where(
                                magnitude < 3.5,
                                mfq_tensor_backend::full_like(magnitude, 3.0),
                                mfq_tensor_backend::where(
                                    magnitude <= 5.0,
                                    mfq_tensor_backend::full_like(magnitude, 4.0),
                                    mfq_tensor_backend::full_like(
                                        magnitude, 6.0))))))));
        quantized = mfq_tensor_backend::where(
            normalized < 0, -quantized, quantized);
        auto reference = (quantized * scale)
            .reshape({ROWS, WIDTH}).to(mfq_tensor_backend::kFloat16);
        const double max_abs = (test - reference)
            .abs().max().item<double>();
        std::cout << "dsv4_fp4_sim_max_abs=" << max_abs << "\n";
        if (max_abs != 0.0) {
            throw std::runtime_error(
                "DeepSeek V4 FP4 activation simulation check failed");
        }
    }

    {
        constexpr int M = 3, K = 768, H = 64, ID = 128;
        auto q = mfq_tensor_backend::randn({B, M, H, ID}, cuda.dtype(mfq_tensor_backend::kFloat16));
        auto k = mfq_tensor_backend::randn({B, K, ID}, cuda.dtype(mfq_tensor_backend::kFloat16));
        auto weights = mfq_tensor_backend::randn({B, M, H}, cuda.dtype(mfq_tensor_backend::kFloat16));
        auto test = dsv4_indexer_scores_cuda(q, k, weights, 4096, 4);
        auto dot = mfq_tensor_backend::einsum(
            "bmhd,bkd->bmhk",
            {q.to(mfq_tensor_backend::kFloat32), k.to(mfq_tensor_backend::kFloat32)});
        auto ref = (mfq_tensor_backend::relu(dot) *
            weights.to(mfq_tensor_backend::kFloat32).unsqueeze(-1)).sum(2) /
            std::sqrt(static_cast<double>(H * ID));
        ref = ref.to(mfq_tensor_backend::kFloat16);
        const double rel =
            (test.to(mfq_tensor_backend::kFloat32) - ref.to(mfq_tensor_backend::kFloat32))
                .norm().item<double>() /
            std::max(ref.to(mfq_tensor_backend::kFloat32).norm().item<double>(), 1e-30);
        auto selected = dsv4_topk512_cuda(test);
        auto expected_topk = mfq_tensor_backend::topk(test, 512, -1, true, false);
        auto expected_scores = std::get<0>(expected_topk);
        auto expected = std::get<1>(expected_topk);
        auto selected_i64 = selected.to(mfq_tensor_backend::kInt64);
        auto selected_sorted = std::get<0>(
            mfq_tensor_backend::sort(selected_i64, -1));
        auto expected_sorted = std::get<0>(
            mfq_tensor_backend::sort(expected.to(mfq_tensor_backend::kInt64), -1));
        const bool topk_id_equal = selected_sorted.equal(expected_sorted);
        const bool topk_ids_valid =
            (selected_i64 >= 0).all().item<bool>() &&
            (selected_i64 < test.size(-1)).all().item<bool>();
        const bool topk_ids_unique =
            selected_sorted.slice(-1, 1, selected_sorted.size(-1))
                .ne(selected_sorted.slice(-1, 0, selected_sorted.size(-1) - 1))
                .all().item<bool>();
        bool topk_scores_equal = false;
        if (topk_ids_valid) {
            auto selected_scores = test.gather(-1, selected_i64);
            auto selected_score_sorted = std::get<0>(
                mfq_tensor_backend::sort(selected_scores, -1));
            auto expected_score_sorted = std::get<0>(
                mfq_tensor_backend::sort(expected_scores, -1));
            topk_scores_equal =
                selected_score_sorted.equal(expected_score_sorted);
        }
        std::cout << "dsv4_indexer_rel=" << rel
                  << " topk_id_set_equal=" << (topk_id_equal ? 1 : 0)
                  << " topk_score_multiset_equal="
                  << (topk_scores_equal ? 1 : 0)
                  << " topk_ids_valid=" << (topk_ids_valid ? 1 : 0)
                  << " topk_ids_unique=" << (topk_ids_unique ? 1 : 0)
                  << "\n";
        if (rel > 0.004 || !topk_scores_equal ||
            !topk_ids_valid || !topk_ids_unique) {
            throw std::runtime_error(
                "DeepSeek V4 indexer/top-k numerical check failed");
        }
    }

    constexpr int H = 64;
    constexpr int M = 3;
    constexpr int HISTORY = 127;
    constexpr int POOL = 800;
    constexpr int TOPK = 512;
    constexpr int WINDOW = 128;
    const double scale = 1.0 / std::sqrt(static_cast<double>(D));
    auto q = mfq_tensor_backend::randn({B, H, M, D}, cuda.dtype(mfq_tensor_backend::kFloat32));
    auto raw = mfq_tensor_backend::randn({B, HISTORY + M, D}, cuda.dtype(mfq_tensor_backend::kFloat16));
    auto pooled = mfq_tensor_backend::randn({B, POOL, D}, cuda.dtype(mfq_tensor_backend::kFloat16));
    auto kv = mfq_tensor_backend::cat({raw, pooled}, 1).contiguous();
    std::vector<mfq_tensor_backend::Tensor> topk_rows;
    for (int row = 0; row < M; ++row) {
        topk_rows.push_back(
            mfq_tensor_backend::randperm(POOL, cuda.dtype(mfq_tensor_backend::kInt64))
                .narrow(0, 0, TOPK).to(mfq_tensor_backend::kInt32));
    }
    auto topk = mfq_tensor_backend::stack(topk_rows, 0).unsqueeze(0).contiguous();
    auto plan = dsv4_build_prefill_plan_cuda(
        topk, 4096, HISTORY, POOL, 4, WINDOW);
    {
        auto seq_len = mfq_tensor_backend::tensor({4097}, cuda.dtype(mfq_tensor_backend::kInt64));
        auto decode_plan = dsv4_build_decode_plan_cuda(
            topk.narrow(1, 0, 1).contiguous(),
            seq_len, POOL, 4, WINDOW);
        auto expected_local =
            mfq_tensor_backend::arange(4097 - WINDOW, 4097, cuda.dtype(mfq_tensor_backend::kInt64))
                .remainder(WINDOW);
        const bool local_equal = decode_plan[0]
            .index({0, 0, Slice(0, WINDOW)})
            .to(mfq_tensor_backend::kInt64).equal(expected_local);
        const bool pooled_equal = decode_plan[0]
            .index({0, 0, Slice(WINDOW, WINDOW + TOPK)})
            .equal(topk.index({0, 0}) + WINDOW);
        const bool mask_clear =
            (decode_plan[1] == 0).all().item<bool>();
        std::cout << "dsv4_decode_plan="
                  << (local_equal && pooled_equal && mask_clear ? 1 : 0)
                  << "\n";
        if (!local_equal || !pooled_equal || !mask_clear) {
            throw std::runtime_error(
                "DeepSeek V4 decode cache plan check failed");
        }
    }
    auto sinks = mfq_tensor_backend::randn({H}, cuda.dtype(mfq_tensor_backend::kFloat32));
    auto meta = mfq_tensor_backend::empty(
        {8 * 1024 * 1024}, cuda.dtype(mfq_tensor_backend::kFloat32));
    auto run_sparse = [&]() {
        return attention_dsv4_sparse_cuda(
            q, kv, plan[0], plan[1], sinks, meta, scale);
    };
    auto test = run_sparse();
    std::vector<mfq_tensor_backend::Tensor> ref_rows;
    auto q_ref = q.to(mfq_tensor_backend::kFloat16).to(mfq_tensor_backend::kFloat32);
    for (int row = 0; row < M; ++row) {
        auto idx = plan[0].index({0, row}).to(mfq_tensor_backend::kInt64);
        auto selected = kv.index_select(1, idx).index({0})
            .to(mfq_tensor_backend::kFloat32);
        auto score = mfq_tensor_backend::matmul(
            q_ref.index({0, Slice(), row}),
            selected.transpose(0, 1)) * scale;
        score = score + plan[1].index({0, row})
            .to(mfq_tensor_backend::kFloat32).unsqueeze(0);
        auto logits = mfq_tensor_backend::cat({score, sinks.unsqueeze(1)}, 1);
        auto probabilities = mfq_tensor_backend::softmax(logits, -1)
            .index({Slice(), Slice(0, score.size(1))});
        ref_rows.push_back(mfq_tensor_backend::matmul(probabilities, selected));
    }
    auto ref = mfq_tensor_backend::stack(ref_rows, 0).unsqueeze(0);
    mfq_cuda_synchronize();
    const double rel = (test - ref).norm().item<double>() /
        std::max(ref.norm().item<double>(), 1e-30);
    const double max_abs = (test - ref).abs().max().item<double>();
    std::cout << "dsv4_sparse_attention_rel=" << rel
              << " max_abs=" << max_abs << "\n";
    if (!mfq_tensor_backend::isfinite(test).all().item<bool>() || rel > 0.02) {
        throw std::runtime_error(
            "DeepSeek V4 sparse attention numerical check failed");
    }

    auto time_ms = [&](auto && fn) {
        cudaEvent_t start, stop;
        MFQ_CUDA_CHECK(cudaEventCreate(&start));
        MFQ_CUDA_CHECK(cudaEventCreate(&stop));
        auto stream = mfq_get_current_cuda_stream().stream();
        for (int i = 0; i < 5; ++i) fn();
        MFQ_CUDA_CHECK(cudaEventRecord(start, stream));
        for (int i = 0; i < reps; ++i) fn();
        MFQ_CUDA_CHECK(cudaEventRecord(stop, stream));
        MFQ_CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed = 0.0f;
        MFQ_CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
        MFQ_CUDA_CHECK(cudaEventDestroy(start));
        MFQ_CUDA_CHECK(cudaEventDestroy(stop));
        return elapsed / reps;
    };
    std::cout << "dsv4_sparse_attention_m=" << M
              << " selected=" << plan[0].size(2)
              << " cuda_ms=" << time_ms(run_sparse) << "\n";
    return 0;
}

int run_text_session_state_check() {
    MfqCudaGuard guard(0);
    auto cuda_float = mfq_tensor_backend::TensorOptions()
        .device(mfq_tensor_backend::kCUDA).dtype(mfq_tensor_backend::kFloat32);
    const auto values = [&](std::vector<int64_t> shape,
                            mfq_tensor_backend::ScalarType dtype,
                            float offset) {
        int64_t elements = 1;
        for (int64_t extent : shape) elements *= extent;
        return (mfq_tensor_backend::arange(elements, cuda_float) + offset)
            .reshape(shape).to(dtype).contiguous();
    };
    const auto require_equal = [](bool condition, const char * message) {
        if (!condition) throw std::runtime_error(message);
    };

    mfq::cuda::DeepseekV4CausalLm dsv4_model;
    auto dsv4 = std::make_unique<Dsv4Block>();
    auto * dsv4_block = dsv4.get();
    dsv4_block->cuda_device = 0;
    dsv4_block->shared_state = std::make_shared<Dsv4SharedState>();
    dsv4_block->shared_state->attention_meta = mfq_tensor_backend::empty({1}, cuda_float);
    dsv4_block->shared_state->hadamard_signs = mfq_tensor_backend::ones(
        {128}, cuda_float.dtype(mfq_tensor_backend::kInt8));
    dsv4_block->local_cache = values(
        {1, 128, 8}, mfq_tensor_backend::kFloat16, 1.0f);
    const auto initialize_dsv4_pool = [&](Dsv4PoolState & pool,
                                          int64_t head_dim,
                                          bool overlap,
                                          float offset) {
        pool.ratio = 4;
        pool.head_dim = head_dim;
        pool.overlap = overlap;
        pool.cache_quant_mode = overlap ? 2 : 0;
        pool.capacity = 16;
        const int64_t state_width = overlap ? 2 * head_dim : head_dim;
        pool.state_kv = values(
            {1, pool.ratio, state_width}, mfq_tensor_backend::kFloat32, offset);
        pool.state_gate = values(
            {1, pool.ratio, state_width}, mfq_tensor_backend::kFloat32, offset + 100.0f);
        if (overlap) {
            pool.previous_kv = values(
                {1, pool.ratio, head_dim}, mfq_tensor_backend::kFloat32, offset + 200.0f);
            pool.previous_gate = values(
                {1, pool.ratio, head_dim}, mfq_tensor_backend::kFloat32, offset + 300.0f);
        }
        pool.pool = values(
            {1, pool.capacity, head_dim}, mfq_tensor_backend::kFloat16, offset + 400.0f);
    };
    initialize_dsv4_pool(
        dsv4_block->compressor, 8, false, 10.0f);
    initialize_dsv4_pool(
        dsv4_block->indexer_compressor, 4, true, 20.0f);
    dsv4_model.blocks.push_back(std::move(dsv4));
    dsv4_model.cache_pos = 10;
    std::vector<int64_t> tokens(10);
    std::iota(tokens.begin(), tokens.end(), 0);
    const TextSessionState dsv4_state =
        dsv4_model.capture_text_session_state(tokens);
    dsv4_block->local_cache.zero_();
    dsv4_block->compressor.state_kv.zero_();
    dsv4_block->compressor.state_gate.zero_();
    dsv4_block->compressor.pool.zero_();
    dsv4_block->indexer_compressor.state_kv.zero_();
    dsv4_block->indexer_compressor.state_gate.zero_();
    dsv4_block->indexer_compressor.previous_kv.zero_();
    dsv4_block->indexer_compressor.previous_gate.zero_();
    dsv4_block->indexer_compressor.pool.zero_();
    dsv4_model.cache_pos = 0;
    dsv4_model.restore_text_session_state(dsv4_state);
    const auto & saved_dsv4 = dsv4_state.dsv4_blocks.at(0);
    require_equal(
        mfq_tensor_backend::equal(dsv4_block->local_cache, saved_dsv4.local_cache),
        "DeepSeek V4 local session cache restore failed");
    const auto check_dsv4_pool = [&](const Dsv4PoolState & restored,
                                     const Dsv4PoolSessionState & saved) {
        require_equal(
            mfq_tensor_backend::equal(restored.state_kv, saved.state_kv) &&
            mfq_tensor_backend::equal(restored.state_gate, saved.state_gate),
            "DeepSeek V4 compressor recurrent state restore failed");
        if (saved.overlap) {
            require_equal(
                mfq_tensor_backend::equal(restored.previous_kv, saved.previous_kv) &&
                mfq_tensor_backend::equal(restored.previous_gate, saved.previous_gate),
                "DeepSeek V4 overlap compressor state restore failed");
        }
        require_equal(
            restored.capacity == saved.capacity &&
            mfq_tensor_backend::equal(
                restored.pool.narrow(1, 0, saved.pool.size(1)),
                saved.pool),
            "DeepSeek V4 compressed pool restore failed");
    };
    check_dsv4_pool(
        dsv4_block->compressor, saved_dsv4.compressor);
    check_dsv4_pool(
        dsv4_block->indexer_compressor,
        saved_dsv4.indexer_compressor);
    require_equal(
        dsv4_model.cache_pos == 10 && dsv4_state.bytes > 0,
        "DeepSeek V4 session position restore failed");

    mfq::cuda::GlmDsaCausalLm glm_model;
    auto glm_shared = std::make_shared<GlmDsaSharedState>();
    std::vector<GlmDsaBlock *> glm_blocks;
    for (int index = 0; index < 2; ++index) {
        auto glm = std::make_unique<GlmDsaBlock>();
        glm->cuda_device = 0;
        glm->full_indexer = index == 0;
        glm->shared_state = glm_shared;
        glm->kv_cache = values(
            {1, 1, 16, 6}, mfq_tensor_backend::kFloat16,
            1000.0f + index * 100.0f);
        if (glm->full_indexer) {
            glm->index_cache = values(
                {1, 16, 4}, mfq_tensor_backend::kFloat16, 2000.0f);
        }
        glm_blocks.push_back(glm.get());
        glm_model.blocks.push_back(std::move(glm));
    }
    glm_model.cache_pos = 10;
    const TextSessionState glm_state =
        glm_model.capture_text_session_state(tokens);
    for (auto * glm : glm_blocks) {
        glm->kv_cache.zero_();
        if (glm->index_cache.defined()) glm->index_cache.zero_();
    }
    glm_shared->topk_indices = mfq_tensor_backend::ones(
        {1, 1, 1}, cuda_float.dtype(mfq_tensor_backend::kInt32));
    glm_model.cache_pos = 0;
    glm_model.restore_text_session_state(glm_state);
    for (size_t index = 0; index < glm_blocks.size(); ++index) {
        const auto & saved = glm_state.glm_dsa_blocks.at(index);
        require_equal(
            glm_blocks[index]->kv_cache.size(2) == saved.kv_capacity &&
            mfq_tensor_backend::equal(
                glm_blocks[index]->kv_cache.narrow(
                    2, 0, saved.kv_cache.size(2)),
                saved.kv_cache),
            "GLM DSA MLA session cache restore failed");
        if (saved.full_indexer) {
            require_equal(
                glm_blocks[index]->index_cache.size(1) ==
                    saved.index_capacity &&
                mfq_tensor_backend::equal(
                    glm_blocks[index]->index_cache.narrow(
                        1, 0, saved.index_cache.size(1)),
                    saved.index_cache),
                "GLM DSA index session cache restore failed");
        }
    }
    require_equal(
        glm_model.cache_pos == 10 && glm_state.bytes > 0 &&
        !glm_shared->topk_indices.defined(),
        "GLM DSA session metadata restore failed");

    std::cout << "text_session_state_check dsv4=1 glm_dsa=1\n";
    return 0;
}

#define MFQ_INSTANTIATE_KL(TYPE)                                           \
    template int run_kl_eval_batched<TYPE>(                               \
        TYPE&, const std::string&, int, std::int64_t, int,                 \
        const KlReferenceContract&);                                       \
    template int run_selected_kl_eval<TYPE>(                              \
        TYPE&, const std::string&, int, KlEvaluator, std::int64_t, int,    \
        const KlReferenceContract&)

MFQ_INSTANTIATE_KL(mfq::cuda::Qwen35CausalLm);
MFQ_INSTANTIATE_KL(mfq::cuda::MiniCPMO45CausalLm);
MFQ_INSTANTIATE_KL(mfq::cuda::MiniCPMOTtsCausalLm);
MFQ_INSTANTIATE_KL(mfq::cuda::Gemma4CausalLm);
MFQ_INSTANTIATE_KL(mfq::cuda::GlmDsaCausalLm);
MFQ_INSTANTIATE_KL(mfq::cuda::Glm5CausalLm);
MFQ_INSTANTIATE_KL(mfq::cuda::Qwen4CausalLm);
MFQ_INSTANTIATE_KL(mfq::cuda::DeepseekV4CausalLm);
MFQ_INSTANTIATE_KL(mfq::cuda::DeepseekV41CausalLm);

#undef MFQ_INSTANTIATE_KL

