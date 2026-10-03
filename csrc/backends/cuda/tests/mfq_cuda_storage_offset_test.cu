#include "mfq_cuda_cache_ops.h"
#include "mfq_cuda_quant_ops.h"

#include <array>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>

using namespace mfq_tensor_backend;

static void check(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

static void exact(const Tensor& actual, const Tensor& expected) {
    auto a = actual.cpu().contiguous(), b = expected.cpu().contiguous();
    check(a.sizes() == b.sizes() && a.scalar_type() == b.scalar_type() &&
              std::memcmp(a.data_ptr(), b.data_ptr(), a.nbytes()) == 0,
          "offset kernel output differs from CPU assignment");
}

struct OffsetTensor {
    Tensor storage, view;
    int offset;

    OffsetTensor(const std::vector<int64_t>& shape, ScalarType dtype, int start)
        : offset(start) {
        const auto count = std::accumulate(shape.begin(), shape.end(), int64_t{1},
                                           std::multiplies<int64_t>{});
        storage = full({start + count + 8}, -100, TensorOptions().device(kCUDA).dtype(dtype));
        view = storage.narrow(0, start, count).view(shape);
        check(view.is_contiguous() &&
                  view.data_ptr() == static_cast<char*>(storage.data_ptr()) + start * view.element_size(),
              "fixture lost its contiguous storage offset");
    }

    void guards() const {
        auto host = storage.to(kFloat32).cpu();
        const auto* values = host.data_ptr<float>();
        for (int64_t i = 0; i < host.numel(); ++i)
            if (i < offset || i >= offset + view.numel())
                check(values[i] == -100, "kernel overwrote storage outside the view");
    }
};

static void embedding_cases() {
    constexpr int vocabulary = 7;
    for (auto dtype : {kFloat16, kFloat32})
        for (int width : {32, 33, 2056})
            for (int tokens : {1, 63, 64, 65})
                for (int offset : {1, 8}) {
                    OffsetTensor weight({vocabulary, width}, dtype, offset);
                    std::vector<float> values(vocabulary * width);
                    for (size_t i = 0; i < values.size(); ++i) values[i] = int(i % 113) - 56;
                    weight.view.copy_(tensor(values).to(kCUDA, dtype).reshape({vocabulary, width}));
                    std::vector<int64_t> ids(tokens);
                    for (int t = 0; t < tokens; ++t) ids[t] = t % (vocabulary + 2) - 1;
                    OffsetTensor indices({tokens}, kInt64, 1);
                    indices.view.copy_(tensor(ids));
                    std::vector<float> expected(tokens * width, 0);
                    for (int t = 0; t < tokens; ++t)
                        if (ids[t] >= 0 && ids[t] < vocabulary)
                            std::copy_n(values.data() + ids[t] * width, width,
                                        expected.data() + t * width);
                    exact(embedding_lookup_cuda(weight.view, indices.view),
                          tensor(expected).reshape({tokens, width}).to(dtype));
                    weight.guards();
                    indices.guards();
                }
}

static void kv_case(ScalarType dtype, int tokens, int width, int unaligned, int mode,
                    bool batched_positions) {
    constexpr int batch = 2, heads = 2;
    const int capacity = mode ? 5 : tokens + 3;
    std::array<int, 4> offsets{8, 8, 8, 8};
    if (unaligned >= 0) offsets[unaligned] = 1;
    OffsetTensor k({batch, heads, tokens, width}, dtype, offsets[0]);
    OffsetTensor v({batch, heads, tokens, width}, dtype, offsets[1]);
    OffsetTensor kc({batch, heads, capacity, width}, dtype, offsets[2]);
    OffsetTensor vc({batch, heads, capacity, width}, dtype, offsets[3]);
    std::vector<float> keys(k.view.numel()), values(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        keys[i] = int(i % 127) - 63;
        values[i] = int(i % 97) - 48;
    }
    k.view.copy_(tensor(keys).to(kCUDA, dtype).reshape(k.view.sizes().vec()));
    v.view.copy_(tensor(values).to(kCUDA, dtype).reshape(v.view.sizes().vec()));
    std::vector<int64_t> positions((batched_positions ? batch : 1) * tokens);
    for (size_t i = 0; i < positions.size(); ++i) {
        const int t = i % tokens;
        positions[i] = mode ? 3 + t : tokens - t + int(i / tokens);
        if (!mode && t == 0) positions[i] = -1;
        if (!mode && t == 1) positions[i] = capacity;
    }
    auto shape = batched_positions ? std::vector<int64_t>{batch, tokens}
                                   : std::vector<int64_t>{tokens};
    OffsetTensor pos(shape, kInt64, 1);
    pos.view.copy_(tensor(positions).reshape(shape));
    std::vector<float> expected_k(kc.view.numel(), -100), expected_v(vc.view.numel(), -100);
    for (int b = 0; b < batch; ++b)
        for (int h = 0; h < heads; ++h)
            for (int t = mode ? std::max(0, tokens - capacity) : 0; t < tokens; ++t) {
                const auto p = positions[(batched_positions ? b * tokens : 0) + t];
                if (p < 0 || (!mode && p >= capacity)) continue;
                const auto src = ((b * heads + h) * tokens + t) * width;
                const auto dst = ((b * heads + h) * capacity + p % capacity) * width;
                std::copy_n(keys.data() + src, width, expected_k.data() + dst);
                std::copy_n(values.data() + src, width, expected_v.data() + dst);
            }
    const auto result = mode == 1 ? kv_cache_write_ring_cuda(kc.view, vc.view, k.view, v.view, 3)
                        : mode == 2 ? kv_cache_write_ring_positions_cuda(kc.view, vc.view, k.view, v.view, pos.view)
                                    : kv_cache_write_cuda(kc.view, vc.view, k.view, v.view, pos.view);
    check(result[0].data_ptr() == kc.view.data_ptr() && result[1].data_ptr() == vc.view.data_ptr(),
          "KV writer replaced the offset cache");
    exact(kc.view, tensor(expected_k).reshape(kc.view.sizes().vec()).to(dtype));
    exact(vc.view, tensor(expected_v).reshape(vc.view.sizes().vec()).to(dtype));
    exact(k.view, tensor(keys).to(kCUDA, dtype).reshape(k.view.sizes().vec()).to(dtype));
    exact(v.view, tensor(values).to(kCUDA, dtype).reshape(v.view.sizes().vec()).to(dtype));
    for (const auto* tensor : {&k, &v, &kc, &vc, &pos}) tensor->guards();
}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    try {
        embedding_cases();
        for (auto dtype : {kFloat16, kBFloat16, kFloat32})
            for (int width : {8, 9})
                for (int unaligned : {-1, 0, 1, 2, 3}) {
                    // B*H*T straddles the 4096-row vector dispatch threshold.
                    for (int tokens : {3, 1023, 1024, 1025})
                        for (bool batched : {false, true})
                            kv_case(dtype, tokens, width, unaligned, 0, batched);
                    for (int mode : {1, 2})
                        for (int tokens : {3, 9})
                            kv_case(dtype, tokens, width, unaligned, mode, false);
                }
        std::cout << "embedding and KV storage-offset checks passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
