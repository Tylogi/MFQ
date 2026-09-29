#pragma once

#include "models/transformer.h"
#include "models/include/qwen4_exp.h"
#include "layers.h"

#include <memory>

namespace mfq::cuda::qwen4_exp {

std::unique_ptr<::Block> load_block(
    const mfq::ModelSource& source,
    const mfq::models::qwen4_exp::Config& config,
    int layer);
std::unique_ptr<Gr> load_final_mixer(
    const mfq::ModelSource& source,
    const mfq::models::qwen4_exp::Config& config);
Tensor finalize_hidden(const Gr& mixer, const Tensor& hidden);

} // namespace mfq::cuda::qwen4_exp
