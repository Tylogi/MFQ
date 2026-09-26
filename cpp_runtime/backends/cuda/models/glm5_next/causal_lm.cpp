#include "causal_lm.h"

namespace mfq::cuda::glm5_next {

std::unique_ptr<::Block> load_block(
        const mfq::ModelSource& source,
        const mfq::models::glm5_next::Config& config,
        int layer) {
    return std::make_unique<Glm5NextBlock>(source, config, layer);
}

} // namespace mfq::cuda::glm5_next
