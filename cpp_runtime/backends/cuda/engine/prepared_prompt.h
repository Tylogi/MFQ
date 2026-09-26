#pragma once

#include <cstdint>
#include <string>
#include <vector>

// Backend-wide result of optional input-component preparation. Generation
// consumes this seam without knowing which component produced the embeddings.
struct CudaPreparedPrompt {
    std::vector<std::int64_t> token_ids;
    mfq_tensor_backend::Tensor embeddings;
    mfq_tensor_backend::Tensor positions;
    int decode_position_delta = 0;
    std::string cache_key;

    bool transformed() const noexcept {
        return embeddings.defined() || positions.defined() ||
            decode_position_delta != 0;
    }
};
