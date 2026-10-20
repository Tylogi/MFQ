#include "core/axial_patch_vision.h"

#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mfq::cuda::axial_patch_vision {
namespace {

using Tensor = mfq_tensor_backend::Tensor;
using vision_ops::Affine;
using vision_ops::RmsNorm;
using vision_ops::VisionAttention;

struct VisionBlock {
    RmsNorm norm1;
    VisionAttention attention;
    RmsNorm norm2;
    Affine gate_up;
    Affine down;

    template <class Visitor>
    void visit_tensors(Visitor visit) {
        norm1.visit_tensors(visit);
        norm2.visit_tensors(visit);
        attention.qkv.visit_tensors(visit);
        attention.output.visit_tensors(visit);
        gate_up.visit_tensors(visit);
        down.visit_tensors(visit);
    }
};

std::string block_prefix(int layer) {
    return "vision.block." + std::to_string(layer);
}

} // namespace

struct CudaAxialPatchVisionTower::Impl {
    Config config;
    Affine patch_embedding;
    std::vector<VisionBlock> blocks;
    RmsNorm output_norm;
    Affine aligner_input;
    Affine aligner_output;

    Tensor encode(
        CudaExecutionContext& execution,
        Tensor patches,
        int height,
        int width) const {
        if (height <= 0 || width <= 0 ||
            height > std::numeric_limits<int>::max() / width) {
            throw std::invalid_argument(
                "invalid axial patch vision grid");
        }
        const int count = height * width;
        const int patch_width = static_cast<int>(config.patch_width());
        if (patches.dim() == 4) {
            patches = patches.reshape({count, patch_width});
        }
        if (patches.dim() != 2 || patches.size(0) != count ||
            patches.size(1) != patch_width) {
            throw std::invalid_argument(
                "axial patch vision patch geometry mismatch");
        }
        std::vector<std::int32_t> positions(
            static_cast<std::size_t>(count * 2));
        for (int row = 0; row < height; ++row) {
            for (int column = 0; column < width; ++column) {
                const auto token =
                    static_cast<std::size_t>(row * width + column);
                positions[token * 2] = row;
                positions[token * 2 + 1] = column;
            }
        }
        const std::vector<std::int32_t> segments{
            static_cast<std::int32_t>(count)};
        auto hidden = patch_embedding(
            execution, patches.to(mfq_tensor_backend::kFloat16));
        for (const auto& block : blocks) {
            auto residual = hidden;
            auto attended = block.attention(
                execution, block.norm1(hidden), positions, segments);
            hidden = residual + attended.to(residual.scalar_type());
            residual = hidden;
            auto gate_up = block.gate_up(
                execution, block.norm2(hidden));
            auto parts = gate_up.chunk(2, -1);
            auto activated =
                mfq_tensor_backend::silu(parts.at(0)) * parts.at(1);
            hidden = residual +
                block.down(execution, activated)
                    .to(residual.scalar_type());
        }
        return align(execution, output_norm(hidden), height, width);
    }

    Tensor align(
        CudaExecutionContext& execution,
        Tensor hidden,
        int height,
        int width) const {
        const int ratio = static_cast<int>(config.downsample_ratio);
        const auto grid =
            mfq::models::axial_patch_vision::aligned_grid(
                height, width, ratio);
        const auto indices =
            mfq::models::axial_patch_vision::gather_indices(
                height, width, ratio);
        const std::int64_t hidden_width = config.hidden;
        const auto cpu = mfq_tensor_backend::TensorOptions()
                             .dtype(mfq_tensor_backend::kInt32)
                             .device(mfq_tensor_backend::kCPU);
        auto index = mfq_tensor_backend::tensor(indices, cpu)
                         .to(mfq_tensor_backend::kInt64)
                         .to(hidden.device())
                         .contiguous();
        auto padded = mfq_tensor_backend::cat(
            {
                hidden,
                mfq_tensor_backend::zeros(
                    {1, hidden_width}, hidden.options()),
            },
            0);
        auto gathered = padded.index_select(0, index).reshape(
            {grid.count(), ratio * ratio, hidden_width});
        gathered = gathered.transpose(1, 2)
                       .reshape({grid.count(), hidden_width * ratio * ratio});
        auto projected = aligner_input(execution, gathered);
        auto activated = mfq_tensor_backend::gelu(
                             projected.to(mfq_tensor_backend::kFloat32),
                             "none")
                             .to(projected.scalar_type());
        return aligner_output(execution, activated);
    }
};

CudaAxialPatchVisionTower CudaAxialPatchVisionTower::load(
    CudaExecutionContext& execution,
    const mfq::ModelSource& model,
    const Config& config) {
    config.validate();
    const auto hidden = config.hidden;
    const auto intermediate = config.intermediate;
    const auto patch_width = config.patch_width();
    const auto ratio = config.downsample_ratio;

    auto patch_embedding = Affine::load(
        execution, model,
        "vision.patch_embedding.weight",
        "vision.patch_embedding.bias");
    auto output_norm = RmsNorm::load(
        execution, model,
        "vision.output_norm.weight",
        hidden,
        config.rms_eps);
    auto aligner_input = Affine::load(
        execution, model,
        "vision.aligner.input.weight",
        "vision.aligner.input.bias");
    auto aligner_output = Affine::load(
        execution, model,
        "vision.aligner.output.weight",
        "vision.aligner.output.bias");
    if (patch_embedding.linear.neuron_len() != patch_width ||
        patch_embedding.linear.out() != hidden ||
        output_norm.weight.size(0) != hidden ||
        aligner_input.linear.neuron_len() != hidden * ratio * ratio ||
        aligner_input.linear.out() != config.output_hidden ||
        aligner_output.linear.neuron_len() != config.output_hidden ||
        aligner_output.linear.out() != config.output_hidden) {
        throw std::runtime_error(
            "axial patch vision top-level tensor geometry mismatch");
    }

    std::vector<VisionBlock> blocks;
    blocks.reserve(static_cast<std::size_t>(config.layers));
    for (int layer = 0; layer < config.layers; ++layer) {
        const auto prefix = block_prefix(layer);
        VisionBlock block{
            RmsNorm::load(
                execution, model,
                prefix + ".norm1.weight", hidden, config.rms_eps),
            {
                Affine::load(
                    execution, model,
                    prefix + ".attention.qkv.weight",
                    prefix + ".attention.qkv.bias"),
                Affine::load(
                    execution, model,
                    prefix + ".attention.output.weight",
                    prefix + ".attention.output.bias"),
                hidden,
                config.heads,
                config.head_dim(),
                config.rope_theta,
            },
            RmsNorm::load(
                execution, model,
                prefix + ".norm2.weight", hidden, config.rms_eps),
            Affine::load(
                execution, model,
                prefix + ".mlp.gate_up.weight",
                prefix + ".mlp.gate_up.bias"),
            Affine::load(
                execution, model,
                prefix + ".mlp.down.weight",
                prefix + ".mlp.down.bias"),
        };
        if (block.norm1.weight.size(0) != hidden ||
            block.attention.qkv.linear.neuron_len() != hidden ||
            block.attention.qkv.linear.out() != 3 * hidden ||
            block.attention.output.linear.neuron_len() != hidden ||
            block.attention.output.linear.out() != hidden ||
            block.norm2.weight.size(0) != hidden ||
            block.gate_up.linear.neuron_len() != hidden ||
            block.gate_up.linear.out() != 2 * intermediate ||
            block.down.linear.neuron_len() != intermediate ||
            block.down.linear.out() != hidden) {
            throw std::runtime_error(
                "axial patch vision block geometry mismatch");
        }
        blocks.push_back(std::move(block));
    }
    return CudaAxialPatchVisionTower(std::make_unique<Impl>(Impl{
        config,
        std::move(patch_embedding),
        std::move(blocks),
        std::move(output_norm),
        std::move(aligner_input),
        std::move(aligner_output),
    }));
}

CudaAxialPatchVisionTower::CudaAxialPatchVisionTower(
    std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

CudaAxialPatchVisionTower::CudaAxialPatchVisionTower(
    CudaAxialPatchVisionTower&&) noexcept = default;

CudaAxialPatchVisionTower& CudaAxialPatchVisionTower::operator=(
    CudaAxialPatchVisionTower&&) noexcept = default;

CudaAxialPatchVisionTower::~CudaAxialPatchVisionTower() = default;

Tensor CudaAxialPatchVisionTower::encode(
    CudaExecutionContext& execution,
    Tensor patches,
    int height,
    int width) const {
    return impl_->encode(execution, std::move(patches), height, width);
}

const Config& CudaAxialPatchVisionTower::config() const noexcept {
    return impl_->config;
}

} // namespace mfq::cuda::axial_patch_vision
