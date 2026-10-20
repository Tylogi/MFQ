#include "cpu_projection_rows.h"
#include "kernels/quant_ops.h"
#include "mx.h"
#include "ops/format.h"

using mfq_tensor_backend::indexing::Slice;
using namespace mfq::cuda::quant_format;

CpuProjectionRows make_cpu_projection_rows(const Mxfp4Weight& w) {
    MFQ_RUNTIME_CHECK(!w.values.is_cuda() && !w.scales.is_cuda() && w.neuron_len>0 &&
        w.neuron_len%32==0 && w.values.numel()==w.out*(w.neuron_len/2) &&
        w.scales.numel()==w.out*(w.neuron_len/32),"invalid CPU MXFP4 projection fields");
    CpuProjectionRows plan;plan.width=int(w.neuron_len);plan.outputs=int(w.out);
    const auto* values=w.values.data_ptr<uint8_t>();const auto* scales=w.scales.data_ptr<uint8_t>();
    const int width=plan.width;
    plan.scalar_rows=[values,scales,width](const float* input,int batch,int input_stride,
            float* output,int output_stride,int begin,int end) {
        static constexpr float table[16]={0,.5f,1,1.5f,2,3,4,6,0,-.5f,-1,-1.5f,-2,-3,-4,-6};
        for(int row=begin;row<end;++row)for(int token=0;token<batch;++token) {
            float sum=0;
            for(int group=0;group<width/32;++group) {
                const float scale=std::ldexp(1.0f,int(scales[std::size_t(row)*(width/32)+group])-127);
                const auto* packed=values+std::size_t(row)*(width/2)+group*16;
                const auto* x=input+std::size_t(token)*input_stride+group*32;
                for(int i=0;i<16;++i) {
                    sum+=x[2*i]*float(mfq_half(table[packed[i]&15]*scale));
                    sum+=x[2*i+1]*float(mfq_half(table[packed[i]>>4]*scale));
                }
            }
            output[std::size_t(token)*output_stride+row]=sum;
        }
    };
    return plan;
}

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
        const int64_t scale_columns = weight.scales.size(1);
        const int64_t block_columns=width/scale_columns;
        const int64_t block_rows=weight.scales.size(0)==outputs?1:block_columns;
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
                        (neuron / block_rows) * scale_columns + column / block_columns];
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
    const int64_t scale_columns = scales.size(1);
    const int64_t block_columns=weight.neuron_len/scale_columns;
    const int64_t block_rows=scales.size(0)==weight.out?1:block_columns;
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
                    (row / block_rows) * scale_columns + column / block_columns)];
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
        outputs_per_group % (weight.scales.size(0)==weight.out?1:weight.neuron_len/weight.scales.size(1)) == 0,
        "MXFP8 groupwise output width must preserve scale blocks");
    if (grouped.size(0) <= 8) {
        return profiler.measure("mxfp8.groupwise_small_m", [&]() {
            return mxfp8_groupwise_small_m_cuda(
                weight.values, weight.scales, grouped, groups);
        });
    }
    std::vector<mfq_tensor_backend::Tensor> outputs;
    outputs.reserve(static_cast<size_t>(groups));
    const int64_t block_rows=weight.scales.size(0)==weight.out?1:weight.neuron_len/weight.scales.size(1);
    const int64_t scale_rows_per_group = outputs_per_group / block_rows;
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
        outputs_per_group % (weight.scales.size(0)==weight.out?1:weight.neuron_len/weight.scales.size(1)) == 0,
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
    const int64_t block_rows=weight.scales.size(0)==weight.out?1:weight.neuron_len/weight.scales.size(1);
    const int64_t scale_rows_per_group = outputs_per_group / block_rows;
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
