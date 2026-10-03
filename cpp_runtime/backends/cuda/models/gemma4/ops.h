#pragma once

#include "core/causal_model.h"
#include "models/gemma4/causal_lm.h"
#include "models/gemma4/config.h"

#include <memory>
#include <string>

namespace mfq {
class ModelSource;
}
struct Block;

namespace mfq::cuda::gemma4 {

using Config = mfq::models::gemma4::Config;

std::unique_ptr<::Block> load_block(CudaExecutionContext &execution, const mfq::ModelSource &source,
                                    const Config &config, int layer, const std::string &type);

} // namespace mfq::cuda::gemma4

namespace mfq::cuda {

struct Gemma4Model : CausalResources {
    template <class Backend> using CausalModel = mfq::models::gemma4::CausalLm<Backend>;
    mfq::models::gemma4::Config config;
    double embed_scale = 1.0;

    std::unique_ptr<Block> adapter_load_block(const mfq::ModelSource &source, int layer, int device,
                                              const std::string &type);
};

extern template struct CudaSessionCodec<Gemma4Model>;

} // namespace mfq::cuda

namespace mfq::models {
extern template struct gemma4::CausalLm<cuda::CudaCausalOps<cuda::Gemma4Model>>;
} // namespace mfq::models
