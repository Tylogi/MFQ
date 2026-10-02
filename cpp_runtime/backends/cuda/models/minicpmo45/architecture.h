#pragma once

#include "../causal_ops.h"
#include "../full_attention_session_codec.h"
#include "inference.h"
#include "mfq_cuda_ops.h"
#include "models/block.h"
#include "models/minicpmo45/causal_lm.h"
#include "models/minicpmo45/config.h"
#include "quant_linear.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mfq::cuda {

struct MiniCPMO45Model : CausalLmArchitecture {
    template <class Backend> using CausalModel = mfq::models::minicpmo45::CausalLm<Backend>;
    mfq::models::minicpmo45::Config config;

    bool adapter_uses_common_rope() const noexcept;
    bool adapter_supports_dense_cpu_offload() const noexcept;
    std::unique_ptr<Block> adapter_load_block(const mfq::ModelSource &source, int layer, int device,
                                              const std::string &type);

    mfq_tensor_backend::Tensor adapter_embed(mfq_tensor_backend::Tensor output) const;
    MfqOptional<mfq_tensor_backend::Tensor>
    adapter_attention_mask(MfqOptional<mfq_tensor_backend::Tensor> mask, int64_t tokens,
                           int64_t cache_position) const;
    mfq_tensor_backend::Tensor adapter_prepare_hidden(mfq_tensor_backend::Tensor hidden,
                                                      int64_t batch, int64_t tokens) const;
    mfq_tensor_backend::Tensor
    adapter_finalize_hidden(mfq_tensor_backend::Tensor hidden,
                            const mfq_tensor_backend::Tensor &output_norm, int64_t batch,
                            int64_t tokens) const;
    mfq_tensor_backend::Tensor adapter_logits(const QuantLinear &lm_head,
                                              mfq_tensor_backend::Tensor hidden) const;
    mfq_tensor_backend::Tensor adapter_last_logits(const QuantLinear &lm_head,
                                                   mfq_tensor_backend::Tensor hidden) const;
    mfq_tensor_backend::Tensor adapter_next_token(const QuantLinear &lm_head,
                                                  mfq_tensor_backend::Tensor hidden) const;
};

template <>
struct CudaSessionCodec<MiniCPMO45Model> : FullAttentionSessionCodec<MiniCPMO45Model> {};

extern template struct FullAttentionSessionCodec<MiniCPMO45Model>;

struct MiniCPMOTtsModel : CausalLmArchitecture {
    template <class Backend> using CausalModel = mfq::models::minicpmo45::TtsCausalLm<Backend>;
    mfq::models::ModelConfig config;

    bool adapter_uses_common_rope() const noexcept;
    bool adapter_supports_dense_cpu_offload() const noexcept;
    std::unique_ptr<Block> adapter_load_block(const mfq::ModelSource &source, int layer, int device,
                                              const std::string &type);
};

template <>
struct CudaSessionCodec<MiniCPMOTtsModel> : FullAttentionSessionCodec<MiniCPMOTtsModel> {};

extern template struct FullAttentionSessionCodec<MiniCPMOTtsModel>;

} // namespace mfq::cuda

inline constexpr const char *MINICPMO45_RESAMPLER_POS_EMBED_ASSET =
    "__mfq_asset__/minicpmo45-resampler-pos-embed-v1.bf16";

namespace mfq::models {
extern template struct minicpmo45::CausalLm<cuda::CudaCausalOps<cuda::MiniCPMO45Model>>;
extern template struct minicpmo45::TtsCausalLm<cuda::CudaCausalOps<cuda::MiniCPMOTtsModel>>;
} // namespace mfq::models
