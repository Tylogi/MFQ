#pragma once

#include "models/deepseek_v4/ops.h"
#include "models/deepseek_v41/ops.h"
#include "models/gemma4/ops.h"
#include "models/glm5_next/ops.h"
#include "models/glm_dsa/ops.h"
#include "models/minicpmo45/ops.h"
#include "models/qwen35/ops.h"
#include "models/qwen4_exp/ops.h"

// The CUDA model family set, and the eight family headers that compose it.
// Adding a family edits one row here plus the family's own model directory; the
// load, generation, MTP and KL instantiation sites expand from this list.
// A graph's family is whichever listed type's accepts_backbone() takes
// graph.backbone, so there is no backbone table to keep in sync.
#define MFQ_CUDA_EACH_MODEL(X)                                  \
    X(mfq::cuda::Qwen35CausalLm)                                \
    X(mfq::cuda::MiniCPMO45CausalLm)                            \
    X(mfq::cuda::MiniCPMOTtsCausalLm)                           \
    X(mfq::cuda::Gemma4CausalLm)                                \
    X(mfq::cuda::GlmDsaCausalLm)                                \
    X(mfq::cuda::Glm5CausalLm)                                  \
    X(mfq::cuda::Qwen4CausalLm)                                 \
    X(mfq::cuda::DeepseekV4CausalLm)                            \
    X(mfq::cuda::DeepseekV41CausalLm)
