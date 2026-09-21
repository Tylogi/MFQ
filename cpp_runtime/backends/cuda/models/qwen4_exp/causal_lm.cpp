#include "causal_lm.h"

namespace mfq::cuda::flash_next {

std::unique_ptr<::Block> load_block(
        const mfq::ModelSource& source,
        const mfq::models::flash_next::QwenConfig& config,
        int layer) {
    return std::make_unique<::Qwen4Block>(source, config, layer);
}

std::unique_ptr<::flash_runtime::Gr> load_final_mixer(
        const mfq::ModelSource& source,
        const mfq::models::flash_next::QwenConfig& config) {
    return std::make_unique<::flash_runtime::Gr>(
        source, config, "model.mhc.pre", false);
}

} // namespace mfq::cuda::flash_next
