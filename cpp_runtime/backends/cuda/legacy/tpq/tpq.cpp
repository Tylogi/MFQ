#include "tpq.h"
#include "format.h"
#include "nint.h"
#include "mx.h"

using mfq_tensor_backend::indexing::Slice;
using namespace mfq::cuda::quant_format;

mfq_tensor_backend::Tensor tpq_matmul(
        const TpqWeight & weight,
        mfq_tensor_backend::Tensor x) {
    x = x.contiguous().to(mfq_tensor_backend::kFloat16);
    MFQ_RUNTIME_CHECK(
        x.dim() == 2 && x.size(1) == weight.neuron_len,
        "TPQ activation width mismatch");
    if (x.is_cuda()) {
        MFQ_RUNTIME_CHECK(weight.packed.is_cuda(),
            "CUDA TPQ matmul requires CUDA-resident weights");
        return weight.int4
            ? tpq_int4_matmul_f16_cuda(
                weight.packed, weight.scales, x, weight.group_size)
            : tpq_pq_matmul_f16_cuda(
                weight.packed, weight.codebook, x,
                weight.out, weight.neuron_len,
                weight.vector_size, weight.index_bits);
    }
    MFQ_RUNTIME_CHECK(!weight.packed.is_cuda(),
        "CPU TPQ matmul requires CPU-resident weights");
    const int64_t rows = x.size(0);
    const int64_t outputs = weight.out;
    const int64_t width = weight.neuron_len;
    const auto * input = x.data_ptr<mfq_half>();
    const auto * packed = weight.packed.data_ptr<uint8_t>();
    auto result = mfq_tensor_backend::empty(
        {rows, outputs},
        mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCPU).dtype(mfq_tensor_backend::kFloat16));
    auto * output = result.data_ptr<mfq_half>();
    const auto * scales = weight.int4
        ? weight.scales.data_ptr<mfq_half>() : nullptr;
    const auto * codebook = weight.int4
        ? nullptr : weight.codebook.data_ptr<float>();
    const size_t packed_size = static_cast<size_t>(weight.packed.numel());
    mfq_parallel_for(0, outputs, 1, [&](int64_t begin, int64_t end) {
        std::vector<float> accumulators(static_cast<size_t>(rows));
        for (int64_t neuron = begin; neuron < end; ++neuron) {
            std::fill(accumulators.begin(), accumulators.end(), 0.0f);
            if (weight.int4) {
                const int64_t packed_columns = width / 2;
                const int64_t scale_columns = width / weight.group_size;
                for (int64_t column = 0; column < width; ++column) {
                    const uint8_t byte = packed[
                        neuron * packed_columns + column / 2];
                    const int code = static_cast<int>(
                        (byte >> ((column & 1) * 4)) & 15u) - 8;
                    const float value = static_cast<float>(code) *
                        static_cast<float>(scales[
                            neuron * scale_columns +
                            column / weight.group_size]);
                    for (int64_t row = 0; row < rows; ++row) {
                        accumulators[static_cast<size_t>(row)] = std::fma(
                            static_cast<float>(input[row * width + column]),
                            value, accumulators[static_cast<size_t>(row)]);
                    }
                }
            } else {
                const int64_t vectors = width / weight.vector_size;
                for (int64_t vector = 0; vector < vectors; ++vector) {
                    const size_t linear = static_cast<size_t>(
                        neuron * vectors + vector);
                    const size_t bit = linear *
                        static_cast<size_t>(weight.index_bits);
                    const size_t byte = bit >> 3;
                    const int shift = static_cast<int>(bit & 7);
                    uint32_t code = byte < packed_size ? packed[byte] : 0u;
                    if (byte + 1 < packed_size) {
                        code |= static_cast<uint32_t>(packed[byte + 1]) << 8;
                    }
                    if (byte + 2 < packed_size) {
                        code |= static_cast<uint32_t>(packed[byte + 2]) << 16;
                    }
                    code = (code >> shift) &
                        ((1u << weight.index_bits) - 1u);
                    const float * values = codebook +
                        static_cast<size_t>(code) * weight.vector_size;
                    for (int element = 0;
                         element < weight.vector_size; ++element) {
                        const int64_t column =
                            vector * weight.vector_size + element;
                        for (int64_t row = 0; row < rows; ++row) {
                            accumulators[static_cast<size_t>(row)] = std::fma(
                                static_cast<float>(input[row * width + column]),
                                values[element],
                                accumulators[static_cast<size_t>(row)]);
                        }
                    }
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
bool is_tpq_pq_dtype(const std::string & dtype) {
    return dtype == "TPQ-X" || dtype == "TPQ-W" ||
        dtype == "TPQ-V" || dtype == "TPQ-VV" ||
        dtype == "TPQ-P" || dtype == "TPQ-PVQ";
}

static uint32_t read_tpq_index_cpu(
        const std::vector<uint8_t> & packed,
        size_t linear,
        int bits) {
    const size_t bit = linear * static_cast<size_t>(bits);
    const size_t byte = bit >> 3;
    const int shift = static_cast<int>(bit & 7);
    uint32_t value = byte < packed.size() ? packed[byte] : 0u;
    if (byte + 1 < packed.size()) {
        value |= static_cast<uint32_t>(packed[byte + 1]) << 8;
    }
    if (byte + 2 < packed.size()) {
        value |= static_cast<uint32_t>(packed[byte + 2]) << 16;
    }
    return (value >> shift) & ((1u << bits) - 1u);
}

TpqCpu unpack_tpq_pq(
        const std::vector<uint8_t> & blob,
        const std::string & dtype) {
    constexpr size_t kHeaderBytes = 24;
    if (!is_tpq_pq_dtype(dtype) || blob.size() < kHeaderBytes ||
            std::memcmp(blob.data(), "CPQ1", 4) != 0 || blob[4] != 1) {
        throw std::runtime_error("invalid TPQ-PQ payload header");
    }
    const int tier = blob[5];
    const int vector_size = blob[6];
    const int index_bits = blob[7];
    const int expected_tier = dtype == "TPQ-X" ? 1 :
        (dtype == "TPQ-W" ? 2 :
         (dtype == "TPQ-V" ? 3 :
          (dtype == "TPQ-VV" ? 4 : 5)));
    size_t offset = 8;
    const int32_t axis = read_i32_from(blob, offset);
    const int32_t neuron_len = read_i32_from(blob, offset);
    const uint32_t ndim = read_u32_from(blob, offset);
    const uint32_t codebook_entries = read_u32_from(blob, offset);
    if (tier != expected_tier || axis != 0 || ndim != 2 ||
            neuron_len <= 0 || vector_size <= 0 ||
            neuron_len % vector_size != 0 ||
            index_bits < 8 || index_bits > 16 ||
            codebook_entries <= 1 ||
            codebook_entries > (1u << index_bits)) {
        throw std::runtime_error("invalid TPQ-PQ payload geometry");
    }
    const uint64_t rows_u64 = read_u64_from(blob, offset);
    const uint64_t columns_u64 = read_u64_from(blob, offset);
    const uint32_t rows_tail = read_u32_from(blob, offset);
    if (rows_u64 == 0 ||
            rows_u64 > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
            columns_u64 != static_cast<uint64_t>(neuron_len) ||
            rows_tail != rows_u64) {
        throw std::runtime_error("inconsistent TPQ-PQ matrix dimensions");
    }
    const size_t codebook_values = checked_mxfp8_size(
        codebook_entries, static_cast<uint64_t>(vector_size),
        "TPQ-PQ codebook size");
    const size_t codebook_bytes = checked_mxfp8_size(
        codebook_values, sizeof(float), "TPQ-PQ codebook bytes");
    if (codebook_bytes > blob.size() - offset) {
        throw std::runtime_error("truncated TPQ-PQ codebook");
    }
    TpqCpu result;
    result.out = static_cast<int64_t>(rows_u64);
    result.neuron_len = neuron_len;
    result.vector_size = vector_size;
    result.index_bits = index_bits;
    result.codebook_entries = static_cast<int>(codebook_entries);
    result.codebook.resize(codebook_values);
    std::memcpy(result.codebook.data(), blob.data() + offset, codebook_bytes);
    offset += codebook_bytes;
    if (std::any_of(
            result.codebook.begin(), result.codebook.end(),
            [](float value) { return !std::isfinite(value); })) {
        throw std::runtime_error("TPQ-PQ codebook must be finite");
    }
    const size_t vectors = static_cast<size_t>(neuron_len / vector_size);
    const size_t index_count = checked_mxfp8_size(
        rows_u64, vectors, "TPQ-PQ index count");
    if (index_count > (std::numeric_limits<size_t>::max() - 7) /
            static_cast<size_t>(index_bits)) {
        throw std::runtime_error("TPQ-PQ index stream is too large");
    }
    const size_t index_bytes =
        (index_count * static_cast<size_t>(index_bits) + 7) / 8;
    if (index_bytes != blob.size() - offset) {
        throw std::runtime_error("invalid TPQ-PQ index stream length");
    }
    result.packed.assign(
        blob.begin() + static_cast<ptrdiff_t>(offset), blob.end());
    for (size_t linear = 0; linear < index_count; ++linear) {
        if (read_tpq_index_cpu(result.packed, linear, index_bits) >=
                codebook_entries) {
            throw std::runtime_error(
                "TPQ-PQ index references a missing codeword");
        }
    }
    return result;
}

TpqCpu unpack_tpq_int4(const std::vector<uint8_t> & blob) {
    constexpr size_t kHeaderBytes = 48;
    if (blob.size() < kHeaderBytes ||
            std::memcmp(blob.data(), "CI41", 4) != 0 || blob[4] != 1 ||
            blob[5] != 0 || blob[6] != 0 || blob[7] != 0) {
        throw std::runtime_error("invalid TPQ-I4 payload header");
    }
    size_t offset = 8;
    const uint32_t group_size = read_u32_from(blob, offset);
    const int32_t axis = read_i32_from(blob, offset);
    const int32_t neuron_len = read_i32_from(blob, offset);
    const uint32_t ndim = read_u32_from(blob, offset);
    const uint64_t rows_u64 = read_u64_from(blob, offset);
    const uint64_t columns_u64 = read_u64_from(blob, offset);
    const uint32_t rows_tail = read_u32_from(blob, offset);
    const uint32_t groups_tail = read_u32_from(blob, offset);
    if (group_size != 64 || axis != 0 || ndim != 2 || neuron_len <= 0 ||
            neuron_len % static_cast<int>(group_size) != 0 ||
            columns_u64 != static_cast<uint64_t>(neuron_len) ||
            rows_u64 == 0 || rows_tail != rows_u64 ||
            groups_tail != static_cast<uint32_t>(neuron_len) / group_size ||
            rows_u64 > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        throw std::runtime_error("invalid TPQ-I4 payload geometry");
    }
    const size_t packed_bytes = checked_mxfp8_size(
        rows_u64, static_cast<uint64_t>(neuron_len / 2),
        "TPQ-I4 packed size");
    const size_t scale_count = checked_mxfp8_size(
        rows_u64, groups_tail, "TPQ-I4 scale count");
    const size_t scale_bytes = checked_mxfp8_size(
        scale_count, sizeof(uint16_t), "TPQ-I4 scale bytes");
    if (packed_bytes > blob.size() - offset ||
            scale_bytes != blob.size() - offset - packed_bytes) {
        throw std::runtime_error("invalid TPQ-I4 payload length");
    }
    TpqCpu result;
    result.int4 = true;
    result.out = static_cast<int64_t>(rows_u64);
    result.neuron_len = neuron_len;
    result.group_size = static_cast<int>(group_size);
    result.packed.assign(
        blob.begin() + static_cast<ptrdiff_t>(offset),
        blob.begin() + static_cast<ptrdiff_t>(offset + packed_bytes));
    offset += packed_bytes;
    result.scales_h.resize(scale_count);
    std::memcpy(result.scales_h.data(), blob.data() + offset, scale_bytes);
    for (uint16_t raw : result.scales_h) {
        const bool nonzero_negative =
            (raw & 0x8000u) != 0 && (raw & 0x7fffu) != 0;
        if ((raw & 0x7c00u) == 0x7c00u || nonzero_negative) {
            throw std::runtime_error(
                "TPQ-I4 scales must be finite and non-negative");
        }
    }
    return result;
}

static void write_tpq_index_cpu(
        std::vector<uint8_t> & packed,
        size_t linear,
        int bits,
        uint32_t value) {
    const size_t bit = linear * static_cast<size_t>(bits);
    for (int index = 0; index < bits; ++index) {
        if (((value >> index) & 1u) != 0) {
            packed[(bit + static_cast<size_t>(index)) >> 3] |=
                static_cast<uint8_t>(1u << ((bit + index) & 7));
        }
    }
}

TpqCpu slice_tpq_cpu(
        const TpqCpu & source,
        TensorParallelAxis axis,
        int64_t begin,
        int64_t end) {
    if (begin < 0 || begin >= end) {
        throw std::runtime_error("invalid TPQ tensor-parallel shard");
    }
    if (axis == TensorParallelAxis::Output) {
        if (end > source.out) {
            throw std::runtime_error("invalid TPQ output shard");
        }
        TpqCpu result = source;
        result.out = end - begin;
        if (source.int4) {
            const size_t packed_row = static_cast<size_t>(source.neuron_len / 2);
            const size_t scale_row = static_cast<size_t>(
                source.neuron_len / source.group_size);
            result.packed.assign(
                source.packed.begin() + static_cast<ptrdiff_t>(begin * packed_row),
                source.packed.begin() + static_cast<ptrdiff_t>(end * packed_row));
            result.scales_h.assign(
                source.scales_h.begin() + static_cast<ptrdiff_t>(begin * scale_row),
                source.scales_h.begin() + static_cast<ptrdiff_t>(end * scale_row));
        } else {
            const size_t vectors = static_cast<size_t>(
                source.neuron_len / source.vector_size);
            const size_t count = static_cast<size_t>(result.out) * vectors;
            result.packed.assign(
                (count * source.index_bits + 7) / 8, 0);
            for (size_t row = 0; row < static_cast<size_t>(result.out); ++row) {
                for (size_t vector = 0; vector < vectors; ++vector) {
                    const uint32_t code = read_tpq_index_cpu(
                        source.packed,
                        (static_cast<size_t>(begin) + row) * vectors + vector,
                        source.index_bits);
                    write_tpq_index_cpu(
                        result.packed, row * vectors + vector,
                        source.index_bits, code);
                }
            }
        }
        return result;
    }
    const int granularity = source.int4
        ? source.group_size : source.vector_size;
    if (axis != TensorParallelAxis::Input || end > source.neuron_len ||
            begin % granularity != 0 || end % granularity != 0) {
        throw std::runtime_error(
            "TPQ input shard does not preserve packed vector boundaries");
    }
    TpqCpu result = source;
    result.neuron_len = end - begin;
    if (source.int4) {
        const size_t source_packed_row =
            static_cast<size_t>(source.neuron_len / 2);
        const size_t result_packed_row =
            static_cast<size_t>(result.neuron_len / 2);
        const size_t source_scale_row = static_cast<size_t>(
            source.neuron_len / source.group_size);
        const size_t result_scale_row = static_cast<size_t>(
            result.neuron_len / result.group_size);
        result.packed.resize(static_cast<size_t>(result.out) * result_packed_row);
        result.scales_h.resize(static_cast<size_t>(result.out) * result_scale_row);
        for (int64_t row = 0; row < result.out; ++row) {
            std::memcpy(
                result.packed.data() + static_cast<size_t>(row) * result_packed_row,
                source.packed.data() + static_cast<size_t>(row) * source_packed_row +
                    static_cast<size_t>(begin / 2),
                result_packed_row);
            std::memcpy(
                result.scales_h.data() + static_cast<size_t>(row) * result_scale_row,
                source.scales_h.data() + static_cast<size_t>(row) * source_scale_row +
                    static_cast<size_t>(begin / source.group_size),
                result_scale_row * sizeof(uint16_t));
        }
    } else {
        const size_t source_vectors = static_cast<size_t>(
            source.neuron_len / source.vector_size);
        const size_t first_vector = static_cast<size_t>(begin / source.vector_size);
        const size_t result_vectors = static_cast<size_t>(
            result.neuron_len / result.vector_size);
        const size_t count = static_cast<size_t>(result.out) * result_vectors;
        result.packed.assign((count * source.index_bits + 7) / 8, 0);
        for (size_t row = 0; row < static_cast<size_t>(result.out); ++row) {
            for (size_t vector = 0; vector < result_vectors; ++vector) {
                const uint32_t code = read_tpq_index_cpu(
                    source.packed,
                    row * source_vectors + first_vector + vector,
                    source.index_bits);
                write_tpq_index_cpu(
                    result.packed, row * result_vectors + vector,
                    source.index_bits, code);
            }
        }
    }
    return result;
}
TpqWeight to_device_tpq(
        const TpqCpu & source, bool cuda, int device) {
    TpqWeight result;
    result.int4 = source.int4;
    result.out = source.out;
    result.neuron_len = source.neuron_len;
    result.group_size = source.group_size;
    result.vector_size = source.vector_size;
    result.index_bits = source.index_bits;
    result.packed = cpu_u8_tensor(
        source.packed,
        source.int4
            ? std::initializer_list<int64_t>{
                source.out, source.neuron_len / 2}
            : std::initializer_list<int64_t>{
                static_cast<int64_t>(source.packed.size())});
    if (source.int4) {
        result.scales = cpu_f16_tensor(
            source.scales_h,
            {source.out, source.neuron_len / source.group_size});
    } else {
        result.codebook = mfq_tensor_backend::from_blob(
            const_cast<float *>(source.codebook.data()),
            {source.codebook_entries, source.vector_size},
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kFloat32)).clone();
    }
    if (cuda) {
        const int target_device = device >= 0
            ? device : mfq_current_cuda_device();
        MfqCudaGuard guard(target_device);
        const auto target = mfq_tensor_backend::Device(mfq_tensor_backend::kCUDA, target_device);
        result.packed = result.packed.to(target, false, false).contiguous();
        if (source.int4) {
            result.scales = result.scales.to(target, false, false).contiguous();
        } else {
            result.codebook = result.codebook.to(target, false, false).contiguous();
        }
    }
    return result;
}

TpqCpu select_tpq_cpu_rows(
        const TpqCpu & source,
        const std::vector<int64_t> & rows) {
    if (rows.empty()) {
        throw std::runtime_error("cannot create an empty TPQ MoE shard");
    }
    TpqCpu result = source;
    result.out = static_cast<int64_t>(rows.size());
    if (source.int4) {
        const size_t packed_row = static_cast<size_t>(source.neuron_len / 2);
        const size_t scale_row = static_cast<size_t>(
            source.neuron_len / source.group_size);
        result.packed.resize(rows.size() * packed_row);
        result.scales_h.resize(rows.size() * scale_row);
        for (size_t destination = 0; destination < rows.size(); ++destination) {
            const int64_t source_row = rows[destination];
            if (source_row < 0 || source_row >= source.out) {
                throw std::runtime_error("TPQ MoE shard row is out of range");
            }
            std::memcpy(
                result.packed.data() + destination * packed_row,
                source.packed.data() +
                    static_cast<size_t>(source_row) * packed_row,
                packed_row);
            std::memcpy(
                result.scales_h.data() + destination * scale_row,
                source.scales_h.data() +
                    static_cast<size_t>(source_row) * scale_row,
                scale_row * sizeof(uint16_t));
        }
        return result;
    }
    const size_t vectors = static_cast<size_t>(
        source.neuron_len / source.vector_size);
    const size_t count = rows.size() * vectors;
    result.packed.assign((count * source.index_bits + 7) / 8, 0);
    for (size_t destination = 0; destination < rows.size(); ++destination) {
        const int64_t source_row = rows[destination];
        if (source_row < 0 || source_row >= source.out) {
            throw std::runtime_error("TPQ MoE shard row is out of range");
        }
        for (size_t vector = 0; vector < vectors; ++vector) {
            const uint32_t code = read_tpq_index_cpu(
                source.packed,
                static_cast<size_t>(source_row) * vectors + vector,
                source.index_bits);
            write_tpq_index_cpu(
                result.packed, destination * vectors + vector,
                source.index_bits, code);
        }
    }
    return result;
}
