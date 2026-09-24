#include "causal_lm.h"

namespace mfq::cuda::flash_next {

std::unique_ptr<::Block> load_block(
        const mfq::ModelSource& source,
        const mfq::models::flash_next::GlmConfig& config,
        int layer) {
    return std::make_unique<::Glm5NextBlock>(source, config, layer);
}

} // namespace mfq::cuda::flash_next
