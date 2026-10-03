#include "mfq_cuda_quant_ops.h"
#include "../../../tests/nint_row_fixture.h"
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace {
using namespace mfq::cuda;
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void check_batch(const mfq::NintRowBatch& batch, int& cases, bool capture) {
    batch.validate();
    const Device gpu{kCUDA, 0};
    auto packed = tensor(batch.packed()).to(gpu);
    std::vector<std::int32_t> words(batch.descriptors().size());
    std::memcpy(words.data(), batch.descriptors().data(), words.size() * 4);
    auto descriptors = tensor(words).reshape({static_cast<int64_t>(batch.rows()), 6}).to(gpu);
    std::vector<float> reference;
    for (std::size_t row = 0; row < batch.rows(); ++row)
        for (int col = 0; col < batch.width(); ++col)
            reference.push_back(mfq::test::row_value(batch, row, col));
    auto expected = tensor(reference).reshape({static_cast<int64_t>(batch.rows()), batch.width()}).to(gpu, kFloat16);
    const auto verify = [&](const Tensor& result) {
        auto a = result.cpu().contiguous(), b = expected.cpu().contiguous();
        require(a.sizes() == b.sizes() && a.scalar_type() == kFloat16, "selected row shape/dtype differs");
        require(std::memcmp(a.data_ptr(), b.data_ptr(), a.nbytes()) == 0, "selected row scalar oracle differs");
    };
    verify(nint_selected_rows_cuda(packed, descriptors, batch.width()));
    ++cases;
    if (capture && mfq_cuda_graph_capture_supported()) {
        const auto stream = mfq_get_stream_from_pool(false);
        MfqCudaStreamGuard guard(stream);
        MfqCudaGraph graph;
        mfq_prepare_cuda_graph_memory(graph);
        Tensor result = nint_selected_rows_cuda(packed, descriptors, batch.width());
        MFQ_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
        result = Tensor{};
        graph.capture_begin();
        result = nint_selected_rows_cuda(packed, descriptors, batch.width());
        graph.capture_end();
        for (int i = 0; i < 3; ++i) {
            graph.replay(); MFQ_CUDA_CHECK(cudaStreamSynchronize(stream.stream())); verify(result);
        }
    }
}
void check(int width, int gs, int k, bool adaptive, int& cases) {
    auto f = mfq::test::fixture(529, width, gs, k, adaptive);
    const std::uint16_t scales[]{0x0001, 0x0355, 0x2355, 0x3555, 0xb855};
    const std::uint16_t minima[]{0x0001, 0x0255, 0x2155, 0x3555, 0xb455};
    for (int row = 0; row < 529; ++row) {
        std::memcpy(f.blob.data() + 42 + row * 2, &scales[row % 5], 2);
        std::memcpy(f.blob.data() + 42 + 529 * 2 + row * 2, &minima[row % 5], 2);
    }
    mfq::NintRows table(f.blob.data(), f.blob.size());
    mfq::NintRowBatch batch;
    for (int row : {528, 257, 255, 256, 0, 3, 528}) table.append_row(row, batch);
    for (int row = 0; row < 32; ++row) table.append_row(row, batch);
    check_batch(batch, cases, true);
}
}
int main() {
    try {
        int devices = 0;
        const auto status = cudaGetDeviceCount(&devices);
        if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
            (status == cudaSuccess && devices == 0)) return 77;
        MFQ_CUDA_CHECK(status);
        int cases = 0;
        for (const auto width : {53, 160})
            for (const auto gs : {5, 24})
                for (const auto k : {2, 6})
                    for (const auto adaptive : {false, true}) check(width, gs, k, adaptive, cases);
        const auto a = mfq::test::fixture(513, 160, 24, 2);
        const auto b = mfq::test::fixture(513, 160, 7, 6);
        mfq::NintRows first(a.blob.data(), a.blob.size()), second(b.blob.data(), b.blob.size());
        mfq::NintRowBatch cross;
        first.append_row(512, cross); second.append_row(257, cross); first.append_row(0, cross);
        check_batch(cross, cases, false);
        const Device gpu{kCUDA, 0};
        auto empty = nint_selected_rows_cuda(tensor(std::vector<std::uint8_t>{}).to(gpu),
            tensor(std::vector<std::int32_t>{}).reshape({0,6}).to(gpu), 160);
        require(empty.sizes().vec() == std::vector<int64_t>({0,160}), "empty row batch shape");
        std::cout << "NINT packed selected-row CUDA cases=" << cases << " (including fractional/subnormal anchors) passed\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
