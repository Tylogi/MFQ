#pragma once

#include "../native/tensor_backend.h"

#include <cstdint>
#include <vector>

void decode_graph_commit_cuda(mfq_tensor_backend::Tensor next, mfq_tensor_backend::Tensor generated, mfq_tensor_backend::Tensor step,
                              mfq_tensor_backend::Tensor input, mfq_tensor_backend::Tensor pos, mfq_tensor_backend::Tensor len);
mfq_tensor_backend::Tensor sample_greedy_cuda(mfq_tensor_backend::Tensor logits);
mfq_tensor_backend::Tensor sample_softmax_cuda(mfq_tensor_backend::Tensor logits, mfq_tensor_backend::Tensor random, double temperature);
mfq_tensor_backend::Tensor sample_top_k_top_p_cuda(
    mfq_tensor_backend::Tensor logits, mfq_tensor_backend::Tensor random, double temperature, int64_t top_k, double top_p);
void sample_token_counts_add_cuda(mfq_tensor_backend::Tensor counts, mfq_tensor_backend::Tensor tokens);
mfq_tensor_backend::Tensor sample_apply_penalties_cuda(
    mfq_tensor_backend::Tensor logits, mfq_tensor_backend::Tensor counts,
    double presence_penalty, double frequency_penalty, double repetition_penalty);
