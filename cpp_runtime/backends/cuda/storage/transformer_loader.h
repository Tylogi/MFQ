#pragma once

#include "core/block.h"
#include "core/ffn.h"
#include "models/common/model_config.h"

#include <memory>
#include <string>
#include <string_view>

namespace mfq { class ModelSource; }

std::unique_ptr<Block> load_transformer_block(
    CudaExecutionContext& execution,
    const mfq::ModelSource& source,
    const mfq::models::ModelConfig& config,
    int layer,
    const std::string& type,
    bool minicpmo45 = false,
    std::string_view tensor_root = "model");

struct MoeWeightLoadOptions {
    int layer;
    bool cpu_offloaded = false;
    bool router_bias_required = false;
    std::size_t shared_gate_up_compatible_prefix = 2;
};

FFN load_moe_weights(
    CudaExecutionContext& execution,
    const mfq::ModelSource& source,
    std::string_view prefix,
    const MoeWeightLoadOptions& options);
FFN load_ffn(
    CudaExecutionContext& execution,
    const mfq::ModelSource& source,
    const mfq::models::ModelConfig& config,
    int layer,
    bool minicpmo45 = false,
    std::string_view tensor_root = "model");
void load_important_neuron_branch(
    CudaExecutionContext& execution,
    const mfq::ModelSource& source,
    int64_t hidden_size,
    int64_t intermediate_size,
    FFN& ffn,
    const std::string& down_name,
    const std::string& gate_name,
    const std::string& up_name);
