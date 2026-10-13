#include "quant_weight_loader.h"

#include "kernels/quant_ops.h"
#include "ops/format.h"

using mfq_tensor_backend::indexing::Slice;
using namespace mfq::cuda::quant_format;

size_t checked_mxfp8_size(
        uint64_t left, uint64_t right,
        const char * label) {
    if (left == 0 || right == 0 ||
            left > std::numeric_limits<size_t>::max() / right) {
        throw std::runtime_error(
            std::string("invalid MXFP8 ") + label);
    }
    return static_cast<size_t>(left * right);
}

Mxfp8Cpu unpack_mxfp8(
        const std::vector<uint8_t> & blob) {
    constexpr size_t kHeaderBytes = 56;
    if (blob.size() < kHeaderBytes ||
            std::memcmp(blob.data(), "MXT1", 4) != 0 ||
            blob[4] != 1 || blob[5] != 8 ||
            blob[6] != 0 || blob[7] != 0) {
        throw std::runtime_error("invalid MXFP8 payload header");
    }
    size_t offset = 8;
    const uint64_t rows = read_u64_from(blob, offset);
    const uint64_t columns = read_u64_from(blob, offset);
    const uint64_t storage_rows = read_u64_from(blob, offset);
    const uint64_t storage_columns = read_u64_from(blob, offset);
    const uint64_t scale_rows = read_u64_from(blob, offset);
    const uint64_t scale_columns = read_u64_from(blob, offset);
    if (rows > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
            columns > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
            columns % 128 != 0 ||
            storage_rows != rows || storage_columns != columns ||
            scale_rows != (rows + 127) / 128 ||
            scale_columns != columns / 128) {
        throw std::runtime_error("invalid MXFP8 payload geometry");
    }
    const size_t value_bytes = checked_mxfp8_size(
        storage_rows, storage_columns, "value size");
    const size_t scale_bytes = checked_mxfp8_size(
        scale_rows, scale_columns, "scale size");
    if (value_bytes > blob.size() - offset ||
            scale_bytes != blob.size() - offset - value_bytes) {
        throw std::runtime_error("invalid MXFP8 payload length");
    }
    Mxfp8Cpu result;
    result.out = static_cast<int64_t>(rows);
    result.neuron_len = static_cast<int64_t>(columns);
    result.values.assign(
        blob.begin() + static_cast<ptrdiff_t>(offset),
        blob.begin() + static_cast<ptrdiff_t>(offset + value_bytes));
    offset += value_bytes;
    result.scales.assign(
        blob.begin() + static_cast<ptrdiff_t>(offset),
        blob.end());
    if (std::any_of(
            result.values.begin(), result.values.end(),
            [](uint8_t value) { return (value & 0x7fu) == 0x7fu; })) {
        throw std::runtime_error("MXFP8 payload contains an E4M3 NaN code");
    }
    if (std::find(result.scales.begin(), result.scales.end(), 255u) !=
            result.scales.end()) {
        throw std::runtime_error("MXFP8 payload contains an E8M0 NaN scale");
    }
    return result;
}

Mxfp4Cpu unpack_mxfp4(
        const std::vector<uint8_t> & blob) {
    constexpr size_t kHeaderBytes = 56;
    if (blob.size() < kHeaderBytes ||
            std::memcmp(blob.data(), "MXT1", 4) != 0 ||
            blob[4] != 1 || blob[5] != 4 ||
            blob[6] != 0 || blob[7] != 0) {
        throw std::runtime_error("invalid MXFP4 payload header");
    }
    size_t offset = 8;
    const uint64_t rows = read_u64_from(blob, offset);
    const uint64_t columns = read_u64_from(blob, offset);
    const uint64_t storage_rows = read_u64_from(blob, offset);
    const uint64_t storage_columns = read_u64_from(blob, offset);
    const uint64_t scale_rows = read_u64_from(blob, offset);
    const uint64_t scale_columns = read_u64_from(blob, offset);
    if (rows > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
            columns > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
            columns % 32 != 0 || storage_rows != rows ||
            storage_columns != columns / 2 || scale_rows != rows ||
            scale_columns != columns / 32) {
        throw std::runtime_error("invalid MXFP4 payload geometry");
    }
    const size_t value_bytes = checked_mxfp8_size(
        storage_rows, storage_columns, "MXFP4 value size");
    const size_t scale_bytes = checked_mxfp8_size(
        scale_rows, scale_columns, "MXFP4 scale size");
    if (value_bytes > blob.size() - offset ||
            scale_bytes != blob.size() - offset - value_bytes) {
        throw std::runtime_error("invalid MXFP4 payload length");
    }
    Mxfp4Cpu result;
    result.out = static_cast<int64_t>(rows);
    result.neuron_len = static_cast<int64_t>(columns);
    result.values.assign(
        blob.begin() + static_cast<ptrdiff_t>(offset),
        blob.begin() + static_cast<ptrdiff_t>(offset + value_bytes));
    offset += value_bytes;
    result.scales.assign(
        blob.begin() + static_cast<ptrdiff_t>(offset), blob.end());
    if (std::find(result.scales.begin(), result.scales.end(), 255u) !=
            result.scales.end()) {
        throw std::runtime_error("MXFP4 payload contains an E8M0 NaN scale");
    }
    return result;
}

mfq_tensor_backend::Tensor dequant_mxfp4_cpu(const Mxfp4Cpu & source) {
    auto dense = mfq_tensor_backend::empty(
        {source.out, source.neuron_len},
        mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCPU).dtype(mfq_tensor_backend::kFloat32));
    float * destination = dense.data_ptr<float>();
    static constexpr float magnitude[8] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    };
    const int64_t packed_columns = source.neuron_len / 2;
    const int64_t scale_columns = source.neuron_len / 32;
    for (int64_t row = 0; row < source.out; ++row) {
        for (int64_t column = 0; column < source.neuron_len; ++column) {
            const uint8_t packed = source.values[
                static_cast<size_t>(row * packed_columns + column / 2)];
            const uint8_t code = static_cast<uint8_t>(
                (packed >> ((column & 1) * 4)) & 15u);
            const uint8_t raw_scale = source.scales[
                static_cast<size_t>(row * scale_columns + column / 32)];
            const float scale = raw_scale == 255u
                ? std::numeric_limits<float>::quiet_NaN()
                : std::ldexp(1.0f, static_cast<int>(raw_scale) - 127);
            const float value = magnitude[code & 7u] * scale;
            destination[row * source.neuron_len + column] =
                (code & 8u) == 0u ? value : -value;
        }
    }
    return dense.to(mfq_tensor_backend::kFloat16);
}

Mxfp4Cpu select_mxfp4_cpu_rows(
        const Mxfp4Cpu & source,
        const std::vector<int64_t> & rows) {
    Mxfp4Cpu result;
    result.out = static_cast<int64_t>(rows.size());
    result.neuron_len = source.neuron_len;
    const size_t value_row = static_cast<size_t>(source.neuron_len / 2);
    const size_t scale_row = static_cast<size_t>(source.neuron_len / 32);
    result.values.resize(rows.size() * value_row);
    result.scales.resize(rows.size() * scale_row);
    for (size_t destination = 0; destination < rows.size(); ++destination) {
        const int64_t source_row = rows[destination];
        if (source_row < 0 || source_row >= source.out) {
            throw std::runtime_error("MXFP4 selected row is out of range");
        }
        std::memcpy(
            result.values.data() + destination * value_row,
            source.values.data() + static_cast<size_t>(source_row) * value_row,
            value_row);
        std::memcpy(
            result.scales.data() + destination * scale_row,
            source.scales.data() + static_cast<size_t>(source_row) * scale_row,
            scale_row);
    }
    return result;
}

Mxfp4Cpu slice_mxfp4_cpu(
        const Mxfp4Cpu & source,
        TensorParallelAxis axis,
        int64_t begin,
        int64_t end) {
    if (axis == TensorParallelAxis::Output) {
        if (begin < 0 || begin >= end || end > source.out) {
            throw std::runtime_error("invalid MXFP4 output shard");
        }
        std::vector<int64_t> rows(static_cast<size_t>(end - begin));
        std::iota(rows.begin(), rows.end(), begin);
        return select_mxfp4_cpu_rows(source, rows);
    }
    if (axis != TensorParallelAxis::Input || begin < 0 || begin >= end ||
            end > source.neuron_len || begin % 32 != 0 || end % 32 != 0) {
        throw std::runtime_error(
            "MXFP4 input shards must preserve 32-value scale blocks");
    }
    Mxfp4Cpu result;
    result.out = source.out;
    result.neuron_len = end - begin;
    const size_t source_value_row =
        static_cast<size_t>(source.neuron_len / 2);
    const size_t destination_value_row =
        static_cast<size_t>(result.neuron_len / 2);
    const size_t source_scale_row =
        static_cast<size_t>(source.neuron_len / 32);
    const size_t destination_scale_row =
        static_cast<size_t>(result.neuron_len / 32);
    result.values.resize(static_cast<size_t>(result.out) * destination_value_row);
    result.scales.resize(static_cast<size_t>(result.out) * destination_scale_row);
    for (int64_t row = 0; row < result.out; ++row) {
        std::memcpy(
            result.values.data() + static_cast<size_t>(row) * destination_value_row,
            source.values.data() + static_cast<size_t>(row) * source_value_row +
                static_cast<size_t>(begin / 2),
            destination_value_row);
        std::memcpy(
            result.scales.data() + static_cast<size_t>(row) * destination_scale_row,
            source.scales.data() + static_cast<size_t>(row) * source_scale_row +
                static_cast<size_t>(begin / 32),
            destination_scale_row);
    }
    return result;
}
Mxfp8Cpu slice_mxfp8_cpu(
        const Mxfp8Cpu & source,
        TensorParallelAxis axis,
        int64_t begin,
        int64_t end) {
    if (axis != TensorParallelAxis::Output &&
            axis != TensorParallelAxis::Input) {
        throw std::runtime_error("MXFP8 slicing requires output or input axis");
    }
    const int64_t extent = axis == TensorParallelAxis::Output
        ? source.out : source.neuron_len;
    if (begin < 0 || begin >= end || end > extent ||
            begin % 128 != 0 ||
            (end != extent && end % 128 != 0)) {
        throw std::runtime_error(
            "MXFP8 tensor-parallel shards must preserve 128-element blocks");
    }
    Mxfp8Cpu result;
    if (axis == TensorParallelAxis::Output) {
        result.out = end - begin;
        result.neuron_len = source.neuron_len;
        const size_t value_begin =
            static_cast<size_t>(begin * source.neuron_len);
        const size_t value_end =
            static_cast<size_t>(end * source.neuron_len);
        result.values.assign(
            source.values.begin() + static_cast<ptrdiff_t>(value_begin),
            source.values.begin() + static_cast<ptrdiff_t>(value_end));
        const int64_t scale_columns = source.neuron_len / 128;
        const size_t scale_begin =
            static_cast<size_t>((begin / 128) * scale_columns);
        const size_t scale_end = static_cast<size_t>(
            ((end + 127) / 128) * scale_columns);
        result.scales.assign(
            source.scales.begin() + static_cast<ptrdiff_t>(scale_begin),
            source.scales.begin() + static_cast<ptrdiff_t>(scale_end));
        return result;
    }

    result.out = source.out;
    result.neuron_len = end - begin;
    result.values.resize(
        static_cast<size_t>(result.out * result.neuron_len));
    for (int64_t row = 0; row < source.out; ++row) {
        std::memcpy(
            result.values.data() +
                static_cast<size_t>(row * result.neuron_len),
            source.values.data() +
                static_cast<size_t>(row * source.neuron_len + begin),
            static_cast<size_t>(result.neuron_len));
    }
    const int64_t source_scale_columns = source.neuron_len / 128;
    const int64_t result_scale_columns = result.neuron_len / 128;
    const int64_t scale_rows = (source.out + 127) / 128;
    result.scales.resize(
        static_cast<size_t>(scale_rows * result_scale_columns));
    for (int64_t row = 0; row < scale_rows; ++row) {
        std::memcpy(
            result.scales.data() +
                static_cast<size_t>(row * result_scale_columns),
            source.scales.data() + static_cast<size_t>(
                row * source_scale_columns + begin / 128),
            static_cast<size_t>(result_scale_columns));
    }
    return result;
}
Mxfp8Weight to_device_mxfp8(
        const Mxfp8Cpu & source,
        bool cuda,
        int device) {
    Mxfp8Weight result;
    result.out = source.out;
    result.neuron_len = source.neuron_len;
    result.values = cpu_u8_tensor(
        source.values,
        {source.out, source.neuron_len});
    result.scales = cpu_u8_tensor(
        source.scales,
        {(source.out + 127) / 128, source.neuron_len / 128});
    if (cuda) {
        const int target_device = device >= 0
            ? device : mfq_current_cuda_device();
        MfqCudaGuard guard(target_device);
        const auto target = mfq_tensor_backend::Device(mfq_tensor_backend::kCUDA, target_device);
        result.values = result.values.to(target, false, false).contiguous();
        result.scales = result.scales.to(target, false, false).contiguous();
    }
    return result;
}

Mxfp8Weight to_cuda_device_mxfp8(
        const Mxfp8Cpu & source,
        int device) {
    return to_device_mxfp8(source, true, device);
}
Mxfp4Weight to_device_mxfp4(
        const Mxfp4Cpu & source, bool cuda, int device) {
    Mxfp4Weight result;
    result.out = source.out;
    result.neuron_len = source.neuron_len;
    result.values = cpu_u8_tensor(
        source.values, {source.out, source.neuron_len / 2});
    result.scales = cpu_u8_tensor(
        source.scales, {source.out, source.neuron_len / 32});
    if (cuda) {
        const int target_device = device >= 0
            ? device : mfq_current_cuda_device();
        MfqCudaGuard guard(target_device);
        const auto target = mfq_tensor_backend::Device(mfq_tensor_backend::kCUDA, target_device);
        result.values = result.values.to(target, false, false).contiguous();
        result.scales = result.scales.to(target, false, false).contiguous();
    }
    return result;
}

