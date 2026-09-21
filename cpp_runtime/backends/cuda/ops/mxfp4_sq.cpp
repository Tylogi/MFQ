#include "mxfp4_sq.h"

namespace {
mfq_tensor_backend::Tensor cpu_u8_tensor(
        const std::vector<uint8_t>& values,
        std::initializer_list<int64_t> shape) {
    return mfq_tensor_backend::from_blob(
        const_cast<uint8_t*>(values.data()), shape,
        mfq_tensor_backend::TensorOptions().dtype(
            mfq_tensor_backend::kUInt8)).clone();
}

mfq_tensor_backend::Tensor cpu_i32_tensor(
        const std::vector<int32_t>& values,
        std::initializer_list<int64_t> shape) {
    return mfq_tensor_backend::from_blob(
        const_cast<int32_t*>(values.data()), shape,
        mfq_tensor_backend::TensorOptions().dtype(
            mfq_tensor_backend::kInt32)).clone();
}
} // namespace

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
