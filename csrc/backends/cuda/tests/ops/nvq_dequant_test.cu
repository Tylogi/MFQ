// Numerical check for the NVQ dequant unit in kernels/nvq/dequant.cu.
// The host reference decodes the packed streams directly; the embedding kernel
// is a second, independent device decode path over the same records.
#include "kernels/quant_ops.h"

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
using namespace mfq::cuda;

constexpr int kNvq2 = 2;  // E8: one 8-byte codebook entry per eight values.
constexpr int kNvq3 = 3;  // D4: two 4-byte codebook entries per eight values.
constexpr int kGroupSize = 24;

void require(bool condition, const char *message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void write_bits(std::vector<std::uint8_t> &data, std::int64_t bit, int bits,
                std::uint32_t value) {
    for (int i = 0; i < bits; ++i) {
        if ((value >> i) & 1u) {
            data[(bit + i) >> 3] |= static_cast<std::uint8_t>(1u << ((bit + i) & 7));
        }
    }
}

// Mirrors load_packed_bits() in kernels/quant/packed_nvq.cuh.
std::uint32_t read_bits(const std::vector<std::uint8_t> &data, std::int64_t bit, int bits) {
    std::uint32_t word = 0;
    for (int i = 0; i < 3; ++i) {
        const std::int64_t byte = (bit >> 3) + i;
        if (byte >= static_cast<std::int64_t>(data.size())) break;
        word |= static_cast<std::uint32_t>(data[byte]) << (8 * i);
    }
    return (word >> (bit & 7)) & ((1u << bits) - 1u);
}

int parity7(std::uint32_t mask) {
    int bits = 0;
    for (int i = 0; i < 7; ++i) bits += int((mask >> i) & 1u);
    return bits & 1;
}

struct Streams {
    int format, sub_bits, rows, width, groups, vectors, signs, entry_bytes;
    std::vector<std::uint8_t> indices, aux, sub_scale;
    std::vector<std::int8_t> codebook;
    std::vector<float> neuron_scale;
};

Streams make_streams(int format, int sub_bits, int width, int rows) {
    Streams s;
    s.format = format;
    s.sub_bits = sub_bits;
    s.rows = rows;
    s.width = width;
    s.groups = (width + kGroupSize - 1) / kGroupSize;
    s.entry_bytes = format == kNvq3 ? 4 : 8;
    s.vectors = (width + (format == kNvq3 ? 3 : 7)) / (format == kNvq3 ? 4 : 8);
    s.signs = (width + 7) / 8;
    s.indices.assign(static_cast<std::size_t>(rows) * s.vectors, 0);
    for (int row = 0; row < rows; ++row) {
        for (int vector = 0; vector < s.vectors; ++vector) {
            s.indices[std::size_t(row) * s.vectors + vector] =
                static_cast<std::uint8_t>((row * 37 + vector * 91 + 5) & 0xff);
        }
    }
    s.aux.assign((std::size_t(rows) * s.signs * 7 + 7) / 8, 0);
    for (int row = 0; row < rows; ++row) {
        for (int segment = 0; segment < s.signs; ++segment) {
            write_bits(s.aux, (std::int64_t(row) * s.signs + segment) * 7, 7,
                       static_cast<std::uint32_t>((row * 13 + segment * 43) & 0x7f));
        }
    }
    s.sub_scale.assign((std::size_t(rows) * s.groups * sub_bits + 7) / 8, 0);
    for (int row = 0; row < rows; ++row) {
        for (int group = 0; group < s.groups; ++group) {
            write_bits(s.sub_scale, (std::int64_t(row) * s.groups + group) * sub_bits,
                       sub_bits, static_cast<std::uint32_t>((row + group * 3) % (1 << sub_bits)));
        }
    }
    s.codebook.assign(std::size_t(256) * s.entry_bytes, 0);
    for (std::size_t i = 0; i < s.codebook.size(); ++i) {
        s.codebook[i] = static_cast<std::int8_t>(int((i * 29 + 7) % 201) - 100);
    }
    s.neuron_scale.assign(rows, 0.0f);
    for (int row = 0; row < rows; ++row) {
        s.neuron_scale[row] = 0.0013f * float((row % 7) - 3);
    }
    return s;
}

// Host decode of the same records: group state -> scale, codebook entry, sign bits.
std::vector<std::uint16_t> reference_halfs(const Streams &s) {
    std::vector<std::uint16_t> out(std::size_t(s.rows) * s.width);
    for (int row = 0; row < s.rows; ++row) {
        for (int segment = 0; segment < s.signs; ++segment) {
            const int k0 = segment * 8;
            const int group = k0 / kGroupSize;
            const std::uint32_t state = read_bits(
                s.sub_scale, (std::int64_t(row) * s.groups + group) * s.sub_bits, s.sub_bits);
            const float scale = s.neuron_scale[row] * float(state);
            const std::uint32_t mask7 = read_bits(
                s.aux, (std::int64_t(row) * s.signs + segment) * 7, 7);
            const std::uint32_t mask8 = mask7 | (static_cast<std::uint32_t>(parity7(mask7)) << 7);
            for (int i = 0; i < 8; ++i) {
                const int k = k0 + i;
                if (k >= s.width) break;
                const int vector = s.format == kNvq3 ? segment * 2 + i / 4 : segment;
                const std::uint32_t index = s.indices[std::size_t(row) * s.vectors + vector];
                const int slot = s.format == kNvq3 ? i % 4 : i;
                std::int8_t byte = s.codebook[std::size_t(index) * s.entry_bytes + slot];
                // apply_sign8(): two's-complement negate for every set sign bit.
                if ((mask8 >> i) & 1u) byte = static_cast<std::int8_t>(-static_cast<int>(byte));
                const float value = scale * float(static_cast<int>(byte));
                out[std::size_t(row) * s.width + k] = __half_as_ushort(__float2half(value));
            }
        }
    }
    return out;
}

std::vector<std::uint16_t> host_halfs(const Tensor &halfs) {
    require(halfs.scalar_type() == kFloat16, "NVQ dequant must return Half");
    const auto host = halfs.to(kCPU).contiguous();
    std::vector<std::uint16_t> out(static_cast<std::size_t>(host.numel()));
    std::memcpy(out.data(), host.data_ptr(), out.size() * sizeof(std::uint16_t));
    return out;
}

void check_case(int format, int sub_bits, int width, int rows, int &cases) {
    const Device gpu{DeviceType::cuda, 0};
    const Streams s = make_streams(format, sub_bits, width, rows);
    const auto indices = tensor(s.indices).to(gpu);
    const auto aux = tensor(s.aux).to(gpu);
    const auto sub_scale = tensor(s.sub_scale).to(gpu);
    const auto neuron_scale = tensor(s.neuron_scale).to(gpu);
    const auto codebook =
        tensor(s.codebook).reshape({256, s.entry_bytes}).to(gpu);

    const auto weight = nvq_dequant_cuda(indices, aux, sub_scale, neuron_scale, codebook,
                                         width, kGroupSize, sub_bits, format, 0);
    require(weight.sizes().size() == 2 && weight.size(0) == rows && weight.size(1) == width,
            "NVQ dequant output shape mismatch");
    const auto actual = host_halfs(weight);
    const auto expected = reference_halfs(s);
    require(actual == expected, "NVQ dequant differs from the host packed-stream reference");

    // The embedding kernel decodes the same records through a different kernel.
    std::vector<std::int64_t> token_ids(rows);
    for (int row = 0; row < rows; ++row) token_ids[row] = row;
    const auto rows_out = nvq_embedding_lookup_cuda(
        indices, aux, sub_scale, neuron_scale, codebook, tensor(token_ids).to(gpu),
        width, kGroupSize, sub_bits, format, 0);
    const auto embedded = host_halfs(rows_out);
    require(embedded == actual, "NVQ embedding rows differ from the dequantized weight rows");
    ++cases;
}

void check_rejections(int &rejections) {
    const Device gpu{DeviceType::cuda, 0};
    const Streams s = make_streams(kNvq3, 4, 24, 2);
    const auto indices = tensor(s.indices).to(gpu);
    const auto aux = tensor(s.aux).to(gpu);
    const auto sub_scale = tensor(s.sub_scale).to(gpu);
    const auto neuron_scale = tensor(s.neuron_scale).to(gpu);
    const auto codebook = tensor(s.codebook).reshape({256, 4}).to(gpu);
    const auto run = [&](int64_t width, int64_t gs, int64_t format,
                         const Tensor &packed_indices) {
        return nvq_dequant_cuda(packed_indices, aux, sub_scale, neuron_scale, codebook,
                                width, gs, s.sub_bits, format, 0);
    };
    const auto rejects = [&](const char *label, auto &&body) {
        bool thrown = false;
        try {
            body();
        } catch (const std::runtime_error &) {
            thrown = true;
        }
        require(thrown, label);
        ++rejections;
    };
    rejects("NVQ dequant accepted a group size other than 24", [&] { run(24, 32, kNvq3, indices); });
    rejects("NVQ dequant accepted an out-of-range format", [&] { run(24, kGroupSize, 0, indices); });
    rejects("NVQ dequant accepted a short index stream",
            [&] { run(24, kGroupSize, kNvq3, indices.narrow(0, 0, indices.numel() - 1)); });
    rejects("NVQ dequant accepted a short sign stream",
            [&] { return nvq_dequant_cuda(indices, aux.narrow(0, 0, aux.numel() - 1), sub_scale,
                                          neuron_scale, codebook, 24, kGroupSize, s.sub_bits,
                                          kNvq3, 0); });
}

}  // namespace

int main() {
    try {
        int devices = 0;
        const auto status = cudaGetDeviceCount(&devices);
        if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
                (status == cudaSuccess && devices == 0)) {
            return 77;
        }
        MFQ_NATIVE_CUDA_CHECK(status);
        int cases = 0, rejections = 0;
        for (int format : {kNvq3, kNvq2}) {
            for (int sub_bits : {1, 4}) {
                // A multiple of eight and a tail width, plus a multi-group row.
                for (int width : {24, 20, 60}) {
                    check_case(format, sub_bits, width, 3, cases);
                }
            }
        }
        check_rejections(rejections);
        std::cout << "NVQ dequant bit-exact host reference cases=" << cases
                  << " rejections=" << rejections << " PASS\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
