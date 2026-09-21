#include "format.h"

#include <algorithm>
#include <cstring>
#include <limits>

namespace mfq::cuda::quant_format {

uint32_t read_u32(std::istream & is) {
    uint32_t v = 0;
    is.read(reinterpret_cast<char *>(&v), sizeof(v));
    if (!is) throw std::runtime_error("unexpected EOF reading u32");
    return v;
}

uint64_t read_u64(std::istream & is) {
    uint64_t v = 0;
    is.read(reinterpret_cast<char *>(&v), sizeof(v));
    if (!is) throw std::runtime_error("unexpected EOF reading u64");
    return v;
}

int32_t read_i32_from(const std::vector<uint8_t> & b, size_t & off) {
    int32_t v = 0;
    std::memcpy(&v, b.data() + off, sizeof(v));
    off += sizeof(v);
    return v;
}

uint16_t read_u16_from(const std::vector<uint8_t> & b, size_t & off) {
    uint16_t v = 0;
    std::memcpy(&v, b.data() + off, sizeof(v));
    off += sizeof(v);
    return v;
}
uint64_t read_u64_from(const std::vector<uint8_t> & b, size_t & off) {
    uint64_t v = 0;
    std::memcpy(&v, b.data() + off, sizeof(v));
    off += sizeof(v);
    return v;
}

std::string read_str(std::istream & is) {
    uint32_t n = read_u32(is);
    std::string s(n, '\0');
    is.read(s.data(), n);
    if (!is) throw std::runtime_error("unexpected EOF reading string");
    return s;
}

std::vector<uint8_t> unpack_bits(const std::vector<uint8_t> & blob, size_t & off, size_t count, int bits) {
    if (bits < 1 || bits > 8 ||
            count > (std::numeric_limits<size_t>::max() - 7) /
                static_cast<size_t>(bits)) {
        throw std::runtime_error("invalid packed-bit stream dimensions");
    }
    const size_t required =
        (count * static_cast<size_t>(bits) + 7) / 8;
    if (off > blob.size() || required > blob.size() - off) {
        throw std::runtime_error("truncated packed-bit stream");
    }
    std::vector<uint8_t> out(count);
    if (bits == 8) {
        std::copy(blob.begin() + (ptrdiff_t)off, blob.begin() + (ptrdiff_t)(off + count), out.begin());
        off += count;
        return out;
    }
    if (bits == 4) {
        size_t nbytes = (count + 1) / 2;
        for (size_t i = 0; i < count; ++i) {
            uint8_t p = blob[off + i / 2];
            out[i] = (i & 1) ? (p >> 4) : (p & 0x0f);
        }
        off += nbytes;
        return out;
    }
    size_t nbytes = (count * (size_t)bits + 7) / 8;
    for (size_t i = 0; i < count; ++i) {
        uint32_t v = 0;
        size_t bit0 = i * (size_t)bits;
        for (int j = 0; j < bits; ++j) {
            size_t bit = bit0 + (size_t)j;
            uint8_t by = blob[off + bit / 8];
            v |= ((by >> (bit & 7)) & 1u) << j;
        }
        out[i] = (uint8_t)v;
    }
    off += nbytes;
    return out;
}
mfq_tensor_backend::Tensor cpu_u8_tensor(const std::vector<uint8_t> & v, std::initializer_list<int64_t> shape) {
    return mfq_tensor_backend::from_blob((void *)v.data(), shape, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kUInt8)).clone();
}

mfq_tensor_backend::Tensor cpu_i64_tensor(
        const std::vector<int64_t> & v,
        std::initializer_list<int64_t> shape) {
    return mfq_tensor_backend::from_blob(
        (void *)v.data(), shape,
        mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64)).clone();
}

mfq_tensor_backend::Tensor cpu_i32_tensor(
        const std::vector<int32_t> & v,
        std::initializer_list<int64_t> shape) {
    return mfq_tensor_backend::from_blob(
        (void *)v.data(), shape,
        mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt32)).clone();
}

mfq_tensor_backend::Tensor cpu_f16_tensor(
        const std::vector<uint16_t> & v,
        std::initializer_list<int64_t> shape) {
    return mfq_tensor_backend::from_blob(
        (void *)v.data(), shape,
        mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kFloat16)).clone();
}

mfq_tensor_backend::Tensor cpu_f16_to_f32_tensor(const std::vector<uint16_t> & v, int64_t n) {
    auto h = mfq_tensor_backend::from_blob((void *)v.data(), {n}, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kFloat16)).clone();
    return h.to(mfq_tensor_backend::kFloat32).contiguous();
}
mfq_tensor_backend::Tensor cpu_i8_tensor(const std::vector<int8_t> & v, std::initializer_list<int64_t> shape) {
    return mfq_tensor_backend::from_blob((void *)v.data(), shape, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt8)).clone();
}

} // namespace mfq::cuda::quant_format
