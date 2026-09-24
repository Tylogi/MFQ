#pragma once

#include "kl_reference_contract.h"
#include "runtime/causal_lm.h"

#include <cstdint>
#include <string>

enum class KlEvaluator {
    Legacy,
    Optimized,
};

const char* kl_evaluator_name(KlEvaluator evaluator);

template <typename Model>
int run_kl_eval_batched(
    Model& model,
    const std::string& reference_path,
    int max_chunks,
    std::int64_t requested_n_batch,
    int score_override,
    const KlReferenceContract& reference_contract);

template <typename Model>
int run_selected_kl_eval(
    Model& model,
    const std::string& reference_path,
    int max_chunks,
    KlEvaluator evaluator,
    std::int64_t requested_n_batch,
    int score_override,
    const KlReferenceContract& reference_contract);

int run_kl_eval_streamed(
    const std::string& model_path,
    const std::string& config_path,
    const std::string& reference_path,
    const std::string& logits_output_path,
    int max_chunks,
    int layer_group,
    int chunk_batch,
    int score_override,
    const KlReferenceContract& reference_contract);
