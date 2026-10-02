#include "rope.h"

#include "cuda_execution.h"
#include "mfq_cuda_ops.h"

#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

RopeCache::RopeCache(
        int64_t max_positions,
        int64_t dim,
        double base,
        int64_t frequency_dim,
        int64_t active_pairs,
        mfq_tensor_backend::Device device,
        bool official_reciprocal_frequencies)
    : rotary_dim(dim) {
    int64_t half = rotary_dim / 2;
    const int64_t denominator = frequency_dim > 0 ? frequency_dim : rotary_dim;
    if (active_pairs < 0) active_pairs = half;
    if (active_pairs > half) {
        throw std::runtime_error("RoPE active pair count exceeds rotary dimension");
    }
    auto opts = mfq_tensor_backend::TensorOptions().device(device).dtype(mfq_tensor_backend::kFloat32);
    auto pos = mfq_tensor_backend::arange(max_positions, opts);
    auto ar = mfq_tensor_backend::arange(0, rotary_dim, 2, opts);
    auto freq = mfq_tensor_backend::pow(
        mfq_tensor_backend::full({half}, base, opts),
        -ar / (double)denominator);
    if (official_reciprocal_frequencies) {
        auto cpu_opts = mfq_tensor_backend::TensorOptions()
            .device(mfq_tensor_backend::kCPU).dtype(mfq_tensor_backend::kFloat32);
#ifdef MFQ_NATIVE_CUDA_RUNTIME
        std::vector<float> official_values(static_cast<size_t>(half));
        const float official_base = static_cast<float>(base);
        const float official_denominator = static_cast<float>(denominator);
        for (int64_t index = 0; index < half; ++index) {
            const float exponent =
                static_cast<float>(index * 2) / official_denominator;
            official_values[static_cast<size_t>(index)] =
                1.0f / std::pow(official_base, exponent);
        }
        auto official_freq = mfq_tensor_backend::tensor(
            official_values, cpu_opts);
#else
        auto exponent = mfq_tensor_backend::arange(0, rotary_dim, 2, cpu_opts) /
            (double)denominator;
        auto official_freq = mfq_tensor_backend::reciprocal(
            mfq_tensor_backend::pow(mfq_tensor_backend::full({half}, base, cpu_opts), exponent));
#endif
        freq.copy_(official_freq);
    }
    if (active_pairs < half) {
        auto pair = mfq_tensor_backend::arange(half, opts);
        freq = mfq_tensor_backend::where(pair < active_pairs, freq, mfq_tensor_backend::zeros_like(freq));
    }
    auto ang = pos.unsqueeze(1) * freq.unsqueeze(0);
    cos = mfq_tensor_backend::cos(ang).contiguous();
    sin = mfq_tensor_backend::sin(ang).contiguous();
    sections = mfq_tensor_backend::empty({0}, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64).device(mfq_tensor_backend::kCPU));
    empty_sections = sections;
}

void RopeCache::configure_mrope(
        const std::vector<int64_t>& configured_sections,
        bool interleaved,
        int64_t configured_rotary_dim,
        mfq_tensor_backend::Device device) {
        if (configured_sections.empty()) return;
        auto execution_sections = configured_sections;
        if (interleaved) {
            std::array<std::vector<int64_t>, 3> pairs;
            for (int64_t pair = 0; pair < configured_rotary_dim / 2; ++pair) {
                const int64_t residue = pair % 3;
                const int axis = residue == 1 &&
                        pair < configured_sections[1] * 3
                    ? 1
                    : (residue == 2 && pair < configured_sections[2] * 3
                        ? 2 : 0);
                pairs[static_cast<size_t>(axis)].push_back(pair);
            }
            execution_sections = {
                static_cast<int64_t>(pairs[0].size()),
                static_cast<int64_t>(pairs[1].size()),
                static_cast<int64_t>(pairs[2].size())};
            std::vector<int64_t> order;
            for (const auto& axis : pairs) {
                order.insert(order.end(), axis.begin(), axis.end());
            }
            std::vector<int64_t> inverse(order.size());
            for (size_t index = 0; index < order.size(); ++index) {
                inverse[static_cast<size_t>(order[index])] =
                    static_cast<int64_t>(index);
            }
            interleaved_order = mfq_tensor_backend::tensor(
                order, mfq_tensor_backend::TensorOptions()
                    .dtype(mfq_tensor_backend::kInt64)
                    .device(mfq_tensor_backend::kCPU)).to(device);
            interleaved_inverse = mfq_tensor_backend::tensor(
                inverse, mfq_tensor_backend::TensorOptions()
                    .dtype(mfq_tensor_backend::kInt64)
                    .device(mfq_tensor_backend::kCPU)).to(device);
        }
        sections = mfq_tensor_backend::tensor(
            execution_sections,
            mfq_tensor_backend::TensorOptions()
                .dtype(mfq_tensor_backend::kInt64)
                .device(mfq_tensor_backend::kCPU));
    }

mfq_tensor_backend::Tensor RopeCache::apply(
        mfq_tensor_backend::Tensor x,
        mfq_tensor_backend::Tensor pos,
        bool grid_mrope_positions) const {
        if (!x.is_cuda()) {
            MFQ_RUNTIME_CHECK(
                !cos.is_cuda() && !sin.is_cuda(),
                "CPU RoPE requires a CPU-resident table");
            auto positions = pos.contiguous().to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt64)
                .clamp(0, cos.size(0) - 1);
            mfq_tensor_backend::Tensor selected_cos;
            mfq_tensor_backend::Tensor selected_sin;
            if (positions.dim() == 1) {
                selected_cos = cos.index_select(0, positions);
                selected_sin = sin.index_select(0, positions);
            } else if (!grid_mrope_positions && positions.dim() == 2) {
                const int64_t batches = positions.size(0);
                const int64_t tokens = positions.size(1);
                selected_cos = cos.index_select(0, positions.reshape({-1}))
                    .reshape({batches, tokens, rotary_dim / 2})
                    .unsqueeze(1);
                selected_sin = sin.index_select(0, positions.reshape({-1}))
                    .reshape({batches, tokens, rotary_dim / 2})
                    .unsqueeze(1);
            } else if (grid_mrope_positions && sections.numel() == 3) {
                const int64_t * section = sections.data_ptr<int64_t>();
                std::vector<mfq_tensor_backend::Tensor> cos_parts;
                std::vector<mfq_tensor_backend::Tensor> sin_parts;
                int64_t offset = 0;
                for (int axis = 0; axis < 3; ++axis) {
                    const int64_t count = section[axis];
                    auto active = positions.select(0, axis);
                    cos_parts.push_back(
                        cos.index_select(0, active).narrow(1, offset, count));
                    sin_parts.push_back(
                        sin.index_select(0, active).narrow(1, offset, count));
                    offset += count;
                }
                selected_cos = mfq_tensor_backend::cat(cos_parts, 1);
                selected_sin = mfq_tensor_backend::cat(sin_parts, 1);
            } else {
                throw std::runtime_error(
                    "multi-axis CPU RoPE requires three position sections");
            }
            auto xf = x.contiguous().to(mfq_tensor_backend::kFloat32);
            const int64_t half = rotary_dim / 2;
            auto first = xf.narrow(-1, 0, half);
            auto second = xf.narrow(-1, half, half);
            auto output = xf.clone();
            output.narrow(-1, 0, half).copy_(
                first * selected_cos - second * selected_sin);
            output.narrow(-1, half, half).copy_(
                second * selected_cos + first * selected_sin);
            return output;
        }
        auto source = x.contiguous().to(mfq_tensor_backend::kFloat32);
        auto active_cos = cos;
        auto active_sin = sin;
        const bool interleaved = grid_mrope_positions &&
            interleaved_order.defined();
        if (interleaved) {
            const int64_t half = rotary_dim / 2;
            std::vector<mfq_tensor_backend::Tensor> parts{
                source.narrow(-1, 0, half).index_select(-1, interleaved_order),
                source.narrow(-1, half, half).index_select(-1, interleaved_order)};
            if (source.size(-1) > rotary_dim) {
                parts.push_back(source.narrow(
                    -1, rotary_dim, source.size(-1) - rotary_dim));
            }
            source = mfq_tensor_backend::cat(parts, -1).contiguous();
            active_cos = cos.index_select(1, interleaved_order).contiguous();
            active_sin = sin.index_select(1, interleaved_order).contiguous();
        }
        auto output = rope_table_cuda(
            source,
            pos.contiguous().to(
                mfq_tensor_backend::kCUDA, mfq_tensor_backend::kInt64),
            active_cos, active_sin, rotary_dim,
            grid_mrope_positions ? sections : empty_sections);
        if (interleaved) {
            const int64_t half = rotary_dim / 2;
            std::vector<mfq_tensor_backend::Tensor> parts{
                output.narrow(-1, 0, half).index_select(
                    -1, interleaved_inverse),
                output.narrow(-1, half, half).index_select(
                    -1, interleaved_inverse)};
            if (output.size(-1) > rotary_dim) {
                parts.push_back(output.narrow(
                    -1, rotary_dim, output.size(-1) - rotary_dim));
            }
            output = mfq_tensor_backend::cat(parts, -1).contiguous();
        }
        return output;
    }

mfq_tensor_backend::Tensor RopeCache::apply_bf16(
        const CudaExecutionConfig& config,
        mfq_tensor_backend::Tensor x,
        mfq_tensor_backend::Tensor pos) const {
        MFQ_RUNTIME_CHECK(
            sections.numel() == 0,
            "Qwen3 BF16 RoPE does not support multi-axis sections");
        if (x.is_cuda() && pos.dim() == 1 &&
                config.minicpm_fused_bf16_rope) {
            return rope_table_bf16_cuda(
                x.contiguous().to(mfq_tensor_backend::kBFloat16),
                pos.contiguous().to(cos.device(), mfq_tensor_backend::kInt64),
                cos, sin, rotary_dim);
        }
        auto positions = pos.contiguous().to(cos.device(), mfq_tensor_backend::kInt64)
            .clamp(0, cos.size(0) - 1);
        mfq_tensor_backend::Tensor selected_cos;
        mfq_tensor_backend::Tensor selected_sin;
        if (positions.dim() == 1) {
            selected_cos = cos.index_select(0, positions)
                .to(mfq_tensor_backend::kBFloat16).unsqueeze(0).unsqueeze(0);
            selected_sin = sin.index_select(0, positions)
                .to(mfq_tensor_backend::kBFloat16).unsqueeze(0).unsqueeze(0);
        } else if (positions.dim() == 2) {
            const int64_t batches = positions.size(0);
            const int64_t tokens = positions.size(1);
            selected_cos = cos.index_select(0, positions.reshape({-1}))
                .reshape({batches, tokens, rotary_dim / 2})
                .to(mfq_tensor_backend::kBFloat16).unsqueeze(1);
            selected_sin = sin.index_select(0, positions.reshape({-1}))
                .reshape({batches, tokens, rotary_dim / 2})
                .to(mfq_tensor_backend::kBFloat16).unsqueeze(1);
        } else {
            throw std::runtime_error(
                "Qwen3 BF16 RoPE positions must be rank one or two");
        }
        auto xb = x.contiguous().to(mfq_tensor_backend::kBFloat16);
        const int64_t half = rotary_dim / 2;
        auto first = xb.narrow(-1, 0, half);
        auto second = xb.narrow(-1, half, half);
        auto output = xb.clone();
        output.narrow(-1, 0, half).copy_(
            first * selected_cos - second * selected_sin);
        output.narrow(-1, half, half).copy_(
            second * selected_cos + first * selected_sin);
        return output.contiguous();
    }

