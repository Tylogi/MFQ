#pragma once

#include "mfq_tensor_backend.h"

// Chunk algebra uses scalar log gates, FP32 state/output, and the same grouped
// or tiled key-head mapping as the recurrent GDN operator. Unsupported devices
// and per-key gates retain the recurrent implementation.
bool gdn_prefill_chunk_enabled(int tokens, bool scalar_gate);
// Returns an empty result when the transient chunk workspace cannot fit.
// No recurrent state has been modified in that case.
std::vector<mfq_tensor_backend::Tensor> gdn_prefill_chunk_cuda(
    mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor k, mfq_tensor_backend::Tensor v,
    mfq_tensor_backend::Tensor g, mfq_tensor_backend::Tensor beta,
    MfqOptional<mfq_tensor_backend::Tensor> state, bool inplace, bool transposed, bool tiled,
    mfq_tensor_backend::Tensor lengths = {});
