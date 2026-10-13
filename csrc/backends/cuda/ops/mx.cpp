#include "kernels/quant_ops.h"
#include "mx.h"
#include "format.h"

using mfq_tensor_backend::indexing::Slice;
using namespace mfq::cuda::quant_format;

mfq_tensor_backend::Tensor mxfp8_matmul(
        CudaProfiler& profiler,
        const Mxfp8Weight & weight,
        mfq_tensor_backend::Tensor x) {
    x = x.contiguous().to(mfq_tensor_backend::kFloat16);
    MFQ_RUNTIME_CHECK(
        x.dim() == 2 && x.size(1) == weight.neuron_len,
        "MXFP8 activation width mismatch");
    if (!x.is_cuda()) {
        MFQ_RUNTIME_CHECK(
            !weight.values.is_cuda() && !weight.scales.is_cuda(),
            "CPU MXFP8 matmul requires CPU-resident weights");
        const int64_t rows = x.size(0);
        const int64_t outputs = weight.out;
        const int64_t width = weight.neuron_len;
        const int64_t scale_columns = width / 128;
        const auto * input = x.data_ptr<mfq_half>();
        const auto * values = weight.values.data_ptr<uint8_t>();
        const auto * scales = weight.scales.data_ptr<uint8_t>();
        auto result = mfq_tensor_backend::empty(
            {rows, outputs},
            mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCPU).dtype(mfq_tensor_backend::kFloat16));
        auto * output = result.data_ptr<mfq_half>();
        mfq_parallel_for(0, outputs, 1, [&](int64_t begin, int64_t end) {
            std::vector<float> accumulators(static_cast<size_t>(rows));
            for (int64_t neuron = begin; neuron < end; ++neuron) {
                std::fill(accumulators.begin(), accumulators.end(), 0.0f);
                for (int64_t column = 0; column < width; ++column) {
                    const uint8_t raw = values[neuron * width + column];
                    const unsigned exponent =
                        (static_cast<unsigned>(raw) >> 3u) & 15u;
                    const unsigned mantissa = static_cast<unsigned>(raw) & 7u;
                    float decoded = exponent == 0u
                        ? std::ldexp(float(mantissa) * 0.125f, -6)
                        : std::ldexp(
                            1.0f + float(mantissa) * 0.125f,
                            static_cast<int>(exponent) - 7);
                    if ((raw & 128u) != 0u) decoded = -decoded;
                    const uint8_t raw_scale = scales[
                        (neuron / 128) * scale_columns + column / 128];
                    const float weight_value = decoded * std::ldexp(
                        1.0f, static_cast<int>(raw_scale) - 127);
                    for (int64_t row = 0; row < rows; ++row) {
                        accumulators[static_cast<size_t>(row)] = std::fma(
                            static_cast<float>(input[row * width + column]),
                            weight_value,
                            accumulators[static_cast<size_t>(row)]);
                    }
                }
                for (int64_t row = 0; row < rows; ++row) {
                    output[row * outputs + neuron] = mfq_half(
                        accumulators[static_cast<size_t>(row)]);
                }
            }
        });
        return result;
    }
    if (x.size(0) <= 8) {
        return profiler.measure("mxfp8.small_m", [&]() {
            return mxfp8_small_m_cuda(
                weight.values, weight.scales, x);
        });
    }
    return profiler.measure("mxfp8.packed_matmul", [&]() {
        return mxfp8_matmul_f16_cuda(
            weight.values, weight.scales, x);
    });
}

mfq_tensor_backend::Tensor mxfp8_matmul_f32(
        CudaProfiler& profiler,
        const Mxfp8Weight & weight,
        mfq_tensor_backend::Tensor x) {
    x = x.contiguous().to(mfq_tensor_backend::kFloat16);
    MFQ_RUNTIME_CHECK(
        x.dim() == 2 && x.size(1) == weight.neuron_len,
        "MXFP8 FP32-output activation width mismatch");
    if (x.size(0) <= 8) {
        return profiler.measure("mxfp8.small_m_f32", [&]() {
            return mxfp8_small_m_f32_cuda(
                weight.values, weight.scales, x);
        });
    }
    return profiler.measure("mxfp8.gemm_f32", [&]() {
        return mxfp8_gemm_f32_cuda(
            weight.values, weight.scales, x);
    });
}

mfq_tensor_backend::Tensor mxfp8_cpu_reference(
        const Mxfp8Weight & weight) {
    auto values = weight.values.to(mfq_tensor_backend::kCPU).contiguous();
    auto scales = weight.scales.to(mfq_tensor_backend::kCPU).contiguous();
    const auto * value_bytes = values.data_ptr<uint8_t>();
    const auto * scale_bytes = scales.data_ptr<uint8_t>();
    const int64_t scale_columns = weight.neuron_len / 128;
    std::vector<float> dense(
        static_cast<size_t>(weight.out * weight.neuron_len));
    for (int64_t row = 0; row < weight.out; ++row) {
        for (int64_t column = 0;
             column < weight.neuron_len; ++column) {
            const uint8_t raw = value_bytes[
                static_cast<size_t>(row * weight.neuron_len + column)];
            const unsigned exponent =
                (static_cast<unsigned>(raw) >> 3u) & 15u;
            const unsigned mantissa =
                static_cast<unsigned>(raw) & 7u;
            float value;
            if (exponent == 15u && mantissa == 7u) {
                value = std::numeric_limits<float>::quiet_NaN();
            } else {
                value = exponent == 0u
                    ? std::ldexp(float(mantissa) * 0.125f, -6)
                    : std::ldexp(
                        1.0f + float(mantissa) * 0.125f,
                        static_cast<int>(exponent) - 7);
                if ((raw & 128u) != 0u) value = -value;
            }
            const uint8_t raw_scale = scale_bytes[
                static_cast<size_t>(
                    (row / 128) * scale_columns + column / 128)];
            const float scale = raw_scale == 255u
                ? std::numeric_limits<float>::quiet_NaN()
                : std::ldexp(1.0f, int(raw_scale) - 127);
            dense[static_cast<size_t>(
                row * weight.neuron_len + column)] = value * scale;
        }
    }
    return mfq_tensor_backend::from_blob(
        dense.data(), {weight.out, weight.neuron_len},
        mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kFloat32))
        .clone().to(weight.values.device())
        .to(mfq_tensor_backend::kFloat16).contiguous();
}

mfq_tensor_backend::Tensor mxfp8_groupwise_matmul(
        CudaProfiler& profiler,
        const Mxfp8Weight & weight,
        mfq_tensor_backend::Tensor grouped,
        int64_t groups) {
    grouped = grouped.contiguous().to(mfq_tensor_backend::kFloat16);
    MFQ_RUNTIME_CHECK(
        grouped.dim() == 3 && grouped.size(1) == groups &&
            grouped.size(2) == weight.neuron_len &&
            weight.out % groups == 0,
        "MXFP8 groupwise projection geometry mismatch");
    const int64_t outputs_per_group = weight.out / groups;
    MFQ_RUNTIME_CHECK(
        outputs_per_group % 128 == 0,
        "MXFP8 groupwise output width must preserve scale blocks");
    if (grouped.size(0) <= 8) {
        return profiler.measure("mxfp8.groupwise_small_m", [&]() {
            return mxfp8_groupwise_small_m_cuda(
                weight.values, weight.scales, grouped, groups);
        });
    }
    std::vector<mfq_tensor_backend::Tensor> outputs;
    outputs.reserve(static_cast<size_t>(groups));
    const int64_t scale_rows_per_group = outputs_per_group / 128;
    for (int64_t group = 0; group < groups; ++group) {
        Mxfp8Weight shard;
        shard.out = outputs_per_group;
        shard.neuron_len = weight.neuron_len;
        shard.values = weight.values.narrow(
            0, group * outputs_per_group,
            outputs_per_group).contiguous();
        shard.scales = weight.scales.narrow(
            0, group * scale_rows_per_group,
            scale_rows_per_group).contiguous();
        outputs.push_back(mxfp8_matmul(
            profiler, shard,
            grouped.select(1, group).contiguous()));
    }
    return mfq_tensor_backend::cat(outputs, -1).contiguous();
}

mfq_tensor_backend::Tensor mxfp8_groupwise_matmul_f32(
        CudaProfiler& profiler,
        const Mxfp8Weight & weight,
        mfq_tensor_backend::Tensor grouped,
        int64_t groups) {
    grouped = grouped.contiguous().to(mfq_tensor_backend::kFloat16);
    MFQ_RUNTIME_CHECK(
        grouped.dim() == 3 && grouped.size(1) == groups &&
            grouped.size(2) == weight.neuron_len &&
            weight.out % groups == 0,
        "MXFP8 groupwise FP32-output projection geometry mismatch");
    const int64_t outputs_per_group = weight.out / groups;
    MFQ_RUNTIME_CHECK(
        outputs_per_group % 128 == 0,
        "MXFP8 groupwise output width must preserve scale blocks");
    if (grouped.size(0) <= 8) {
        return profiler.measure(
            "mxfp8.groupwise_small_m_f32", [&]() {
                return mxfp8_groupwise_small_m_f32_cuda(
                    weight.values, weight.scales, grouped, groups);
            });
    }
    std::vector<mfq_tensor_backend::Tensor> outputs;
    outputs.reserve(static_cast<size_t>(groups));
    const int64_t scale_rows_per_group = outputs_per_group / 128;
    for (int64_t group = 0; group < groups; ++group) {
        Mxfp8Weight shard;
        shard.out = outputs_per_group;
        shard.neuron_len = weight.neuron_len;
        shard.values = weight.values.narrow(
            0, group * outputs_per_group,
            outputs_per_group).contiguous();
        shard.scales = weight.scales.narrow(
            0, group * scale_rows_per_group,
            scale_rows_per_group).contiguous();
        outputs.push_back(mxfp8_matmul_f32(
            profiler, shard,
            grouped.select(1, group).contiguous()));
    }
    return mfq_tensor_backend::cat(outputs, -1).contiguous();
}

mfq_tensor_backend::Tensor mxfp4_matmul(
        const Mxfp4Weight & weight,
        mfq_tensor_backend::Tensor x) {
    x = x.contiguous().to(mfq_tensor_backend::kFloat16);
    MFQ_RUNTIME_CHECK(
        x.dim() == 2 && x.size(1) == weight.neuron_len,
        "MXFP4 activation width mismatch");
    if (x.is_cuda()) {
        MFQ_RUNTIME_CHECK(
            weight.values.is_cuda() && weight.scales.is_cuda(),
            "CUDA MXFP4 matmul requires CUDA-resident weights");
        return mxfp4_matmul_f16_cuda(
            weight.values, weight.scales, x);
    }
    MFQ_RUNTIME_CHECK(
        !weight.values.is_cuda() && !weight.scales.is_cuda(),
        "CPU MXFP4 matmul requires CPU-resident weights");
    const int64_t rows = x.size(0);
    const int64_t outputs = weight.out;
    const int64_t width = weight.neuron_len;
    const int64_t packed_columns = width / 2;
    const int64_t scale_columns = width / 32;
    const auto * input = x.data_ptr<mfq_half>();
    const auto * values = weight.values.data_ptr<uint8_t>();
    const auto * scales = weight.scales.data_ptr<uint8_t>();
    auto result = mfq_tensor_backend::empty(
        {rows, outputs},
        mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCPU).dtype(mfq_tensor_backend::kFloat16));
    auto * output = result.data_ptr<mfq_half>();
    static constexpr float magnitude[8] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    };
    mfq_parallel_for(0, outputs, 1, [&](int64_t begin, int64_t end) {
        std::vector<float> accumulators(static_cast<size_t>(rows));
        for (int64_t neuron = begin; neuron < end; ++neuron) {
            std::fill(accumulators.begin(), accumulators.end(), 0.0f);
            for (int64_t column = 0; column < width; ++column) {
                const uint8_t packed = values[
                    neuron * packed_columns + column / 2];
                const uint8_t code = static_cast<uint8_t>(
                    (packed >> ((column & 1) * 4)) & 15u);
                float decoded = magnitude[code & 7u];
                if ((code & 8u) != 0u) decoded = -decoded;
                const uint8_t raw_scale = scales[
                    neuron * scale_columns + column / 32];
                const float weight_value = decoded * std::ldexp(
                    1.0f, static_cast<int>(raw_scale) - 127);
                for (int64_t row = 0; row < rows; ++row) {
                    accumulators[static_cast<size_t>(row)] = std::fma(
                        static_cast<float>(input[row * width + column]),
                        weight_value,
                        accumulators[static_cast<size_t>(row)]);
                }
            }
            for (int64_t row = 0; row < rows; ++row) {
                output[row * outputs + neuron] = mfq_half(
                    accumulators[static_cast<size_t>(row)]);
            }
        }
    });
    return result;
}
