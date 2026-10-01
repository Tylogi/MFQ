#pragma once

#include "../causal_lm.h"
#include "models/include/gemma4.h"

#include <memory>
#include <string>

namespace mfq { class ModelSource; }
struct Block;

namespace mfq::cuda::gemma4 {

using Config = mfq::models::gemma4::Config;

std::unique_ptr<::Block> load_block(
    CudaExecutionContext& execution,
    const mfq::ModelSource& source,
    const Config& config,
    int layer,
    const std::string& type);

} // namespace mfq::cuda::gemma4

namespace mfq::cuda {

struct Gemma4Model : CausalLmArchitecture {
    mfq::models::gemma4::Config config;
    double embed_scale = 1.0;

    void adapter_load_config(
        std::string_view payload,
        const mfq::ModelGraph& graph,
        const mfq::ModelSource& source);
    std::unique_ptr<Block> adapter_load_block(
        const mfq::ModelSource& source,
        int layer,
        int device,
        const std::string& type);
    void adapter_set_max_position_embeddings(int64_t value) {
        config.max_position_embeddings = value;
    }
    mfq_tensor_backend::Tensor adapter_prepare_hidden(
        mfq_tensor_backend::Tensor hidden,
        int64_t batch,
        int64_t tokens) const;
};

extern template struct CudaSessionCodec<Gemma4Model>;

extern template struct CausalLm<Gemma4Model>;

} // namespace mfq::cuda
