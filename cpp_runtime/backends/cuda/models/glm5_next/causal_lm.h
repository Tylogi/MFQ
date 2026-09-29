#pragma once

#include "models/transformer.h"
#include "models/include/glm5_next.h"
#include "layers.h"

#include <memory>

namespace mfq::cuda::glm5_next {

std::unique_ptr<::Block> load_block(
    const mfq::ModelSource& source,
    const mfq::models::glm5_next::Config& config,
    int layer);
Tensor finalize_hidden(
    const Tensor& hidden,
    const Tensor& output_norm,
    double epsilon);

} // namespace mfq::cuda::glm5_next
