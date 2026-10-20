#include "quant.h"

#include "ops/format.h"

using namespace mfq::cuda::quant_format;

Fp8SqWeight to_device_fp8_sq(
        std::string_view dtype,
        const std::vector<uint8_t> & payload,
        bool cuda,
        int device) {
    const auto layout = mfq::fp8sq::parse(
        dtype, payload.data(), payload.size());
    const auto rows = mfq::fp8sq::row_metadata(payload.data(), layout);
    Fp8SqWeight result;
    result.dtype = std::string(dtype);
    result.out = layout.outputs;
    result.neuron_len = layout.width;
    result.block_rows = layout.block_rows;
    result.block_columns = layout.block_columns;
    result.scale_rows = layout.scale_rows;
    result.scale_columns = layout.scale_columns;
    result.scale_kind = static_cast<int64_t>(layout.scale_kind);
    result.palettes_offset = static_cast<int64_t>(layout.palettes);
    result.symbols_offset = static_cast<int64_t>(layout.symbols);
    result.scales_offset = static_cast<int64_t>(layout.scales);
    result.format_version = layout.version;
    result.aggregate_bpw = static_cast<double>(
        layout.bytes - mfq::fp8sq::kHeaderBytes) * 8.0 /
        (static_cast<double>(layout.outputs) * layout.width);
    std::array<size_t, 8> q_counts{};
    std::vector<int32_t> symbol_offsets(rows.symbol_byte_offsets.size());
    for (size_t row = 0; row < rows.q.size(); ++row) {
        const int q = rows.q[row];
        ++q_counts[static_cast<size_t>(q - 1)];
        if (rows.symbol_byte_offsets[row] >
                static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
            throw std::runtime_error(
                "FP8-SQ runtime row metadata exceeds int32 limits");
        }
        symbol_offsets[row] =
            static_cast<int32_t>(rows.symbol_byte_offsets[row]);
    }
    for (const auto count : q_counts) {
        if (count == 0) continue;
        const double probability =
            static_cast<double>(count) / layout.outputs;
        result.distribution_entropy -= probability * std::log2(probability);
    }
    result.blob = cpu_u8_tensor(
        payload, {static_cast<int64_t>(payload.size())});
    result.row_q = cpu_u8_tensor(rows.q, {layout.outputs});
    result.row_symbol_byte_offsets = cpu_i32_tensor(
        symbol_offsets, {layout.outputs});
    if (cuda) {
        const int target_device = device >= 0
            ? device : mfq_current_cuda_device();
        MfqCudaGuard guard(target_device);
        const auto target = mfq_tensor_backend::Device(
            mfq_tensor_backend::kCUDA, target_device);
        result.blob = result.blob.to(target, false, false).contiguous();
        result.row_q = result.row_q.to(target, false, false).contiguous();
        result.row_symbol_byte_offsets =
            result.row_symbol_byte_offsets.to(
                target, false, false).contiguous();
    }
    return result;
}

Mxfp4SqWeight to_device_mxfp4_sq(
        const std::vector<uint8_t> & payload,
        bool cuda,
        int device) {
    const auto layout = mfq::sq::parse(payload.data(), payload.size());
    const auto rows = mfq::sq::row_metadata(payload.data(), layout);
    Mxfp4SqWeight result;
    result.bits = layout.bits;
    result.out = layout.outputs;
    result.neuron_len = layout.width;
    result.matrix_scale_base = layout.base;
    result.sq4_rows = layout.sq4_rows;
    result.format_version = layout.version;
    result.aggregate_bpw = static_cast<double>(layout.bytes - 24) * 8.0 /
        (static_cast<double>(layout.outputs) * layout.width);
    std::array<size_t, 4> q_counts{};
    std::vector<int32_t> symbol_offsets(rows.symbol_byte_offsets.size());
    std::vector<int32_t> auxiliary_rows(rows.auxiliary_rows.size());
    for (size_t row = 0; row < rows.q.size(); ++row) {
        const int q = rows.q[row];
        result.q_sum += q;
        ++q_counts[static_cast<size_t>(q - 1)];
        if (rows.symbol_byte_offsets[row] >
                static_cast<uint32_t>(std::numeric_limits<int32_t>::max()) ||
                rows.auxiliary_rows[row] >
                static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
            throw std::runtime_error(
                "MXFP4-SQ runtime row metadata exceeds int32 limits");
        }
        symbol_offsets[row] =
            static_cast<int32_t>(rows.symbol_byte_offsets[row]);
        auxiliary_rows[row] =
            static_cast<int32_t>(rows.auxiliary_rows[row]);
    }
    for (const auto count : q_counts) {
        if (count == 0) continue;
        const double probability =
            static_cast<double>(count) / layout.outputs;
        result.distribution_entropy -= probability * std::log2(probability);
    }
    result.blob = cpu_u8_tensor(
        payload,
        {static_cast<int64_t>(payload.size())});
    result.row_q = cpu_u8_tensor(rows.q, {layout.outputs});
    result.row_symbol_byte_offsets = cpu_i32_tensor(
        symbol_offsets, {layout.outputs});
    result.row_auxiliary = cpu_i32_tensor(
        auxiliary_rows, {layout.outputs});
    if (cuda) {
        const int target_device = device >= 0
            ? device : mfq_current_cuda_device();
        MfqCudaGuard guard(target_device);
        result.blob = result.blob.to(
            mfq_tensor_backend::Device(
                mfq_tensor_backend::kCUDA,
                target_device),
            false,
            false).contiguous();
        result.row_q = result.row_q.to(
            mfq_tensor_backend::Device(
                mfq_tensor_backend::kCUDA,
                target_device),
            false,
            false).contiguous();
        result.row_symbol_byte_offsets =
            result.row_symbol_byte_offsets.to(
                mfq_tensor_backend::Device(
                    mfq_tensor_backend::kCUDA,
                    target_device),
                false,
                false).contiguous();
        result.row_auxiliary = result.row_auxiliary.to(
            mfq_tensor_backend::Device(
                mfq_tensor_backend::kCUDA,
                target_device),
            false,
            false).contiguous();
    }
    return result;
}


