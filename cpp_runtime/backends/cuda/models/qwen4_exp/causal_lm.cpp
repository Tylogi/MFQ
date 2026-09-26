#include "causal_lm.h"

namespace mfq::cuda::qwen4_exp {

std::unique_ptr<::Block> load_block(
        const mfq::ModelSource& source,
        const mfq::models::qwen4_exp::Config& config,
        int layer) {
    return std::make_unique<Qwen4Block>(source, config, layer);
}

std::unique_ptr<Gr> load_final_mixer(
        const mfq::ModelSource& source,
        const mfq::models::qwen4_exp::Config& config) {
    return std::make_unique<Gr>(
        source, config, "model.mhc.pre", false);
}

} // namespace mfq::cuda::qwen4_exp
