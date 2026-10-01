#include "mfq_cuda_ops.h"
#include "mfq/kernels/cuda/fp8_sq.h"
#include "mfq/kernels/cuda/mxfp4_sq.h"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
using namespace mfq::cuda;
const Device gpu{DeviceType::cuda, 0};
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void environment(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}
void close(const Tensor& actual, const Tensor& reference, float atol, float rtol) {
    require(actual.sizes() == reference.sizes(), "shape mismatch");
    auto a = actual.contiguous().to(kCPU, kFloat32);
    auto b = reference.contiguous().to(kCPU, kFloat32);
    for (std::int64_t i = 0; i < a.numel(); ++i) {
        const float x = a.data_ptr<float>()[i], y = b.data_ptr<float>()[i];
        if (!std::isfinite(x) || std::abs(x - y) > atol + rtol * std::abs(y)) {
            throw std::runtime_error("numerical mismatch at " + std::to_string(i) +
                ": " + std::to_string(x) + " vs " + std::to_string(y));
        }
    }
}
Tensor pattern(std::vector<std::int64_t> shape, float scale, int phase = 0) {
    std::int64_t count = 1;
    for (auto d : shape) count *= d;
    std::vector<float> values(count);
    for (std::int64_t i = 0; i < count; ++i) values[i] = scale * std::sin(float(i * 17 + phase) * 0.137f);
    return tensor(values).reshape(shape).to(gpu);
}
template <typename F> float milliseconds(F run) {
    run();
    cudaEvent_t start, stop;
    MFQ_NATIVE_CUDA_CHECK(cudaEventCreate(&start));
    MFQ_NATIVE_CUDA_CHECK(cudaEventCreate(&stop));
    const auto stream = mfq_current_cuda_stream();
    MFQ_NATIVE_CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < 10; ++i) run();
    MFQ_NATIVE_CUDA_CHECK(cudaEventRecord(stop, stream));
    MFQ_NATIVE_CUDA_CHECK(cudaEventSynchronize(stop));
    float ms;
    MFQ_NATIVE_CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return ms / 10;
}
void fp8(bool mx) {
    const std::string dtype = mx ? "MXFP8-SQ" : "FP8-128SQ";
    const int n = mx ? 17 : 257, k = mx ? 96 : 257;
    const int br = mx ? 1 : 128, bc = mx ? 32 : 128;
    const auto q_bytes = (mfq::fp8sq::packed_nbytes(n, 3) + 3) & ~std::size_t{3};
    const auto palettes = 44 + q_bytes, symbols = palettes + 256;
    std::size_t scales = symbols;
    for (int row = 0; row < n; ++row) scales += mfq::fp8sq::packed_nbytes(k, row % 8 + 1);
    const int sr = (n + br - 1) / br, sc = (k + bc - 1) / bc;
    std::vector<std::uint8_t> raw(scales + sr * sc * (mx ? 1 : 2), 0);
    raw[0] = mx ? 'M' : 'F'; raw[1] = '8'; raw[2] = 'S'; raw[3] = 'Q'; raw[4] = 1;
    raw[5] = mx ? 1 : 2; raw[8] = br; raw[10] = bc;
    const auto write = [&](int at, std::uint64_t value) {
        for (int i = 0; i < 8; ++i) raw[at + i] = value >> (i * 8);
    };
    write(12, n); write(20, k); write(28, sr); write(36, sc);
    const auto put = [&](std::size_t at, std::size_t index, unsigned value, int bits) {
        for (int bit = 0; bit < bits; ++bit) {
            const auto position = index * bits + bit;
            raw[at + position / 8] |= ((value >> bit) & 1) << (position % 8);
        }
    };
    for (int bits = 1; bits <= 7; ++bits) for (int j = 0; j < (1 << bits); ++j)
        raw[palettes + (1 << bits) - 2 + j] = j == 127 ? 128 : j;
    auto offset = symbols;
    for (int row = 0; row < n; ++row) {
        const int bits = row % 8 + 1;
        put(44, row, bits - 1, 3);
        for (int col = 0; col < k; ++col) {
            unsigned code = (row * 13 + col * 31) & ((1 << bits) - 1);
            if (bits == 8 && (code & 127) == 127) code ^= 1;
            put(offset, col, code, bits);
        }
        offset += mfq::fp8sq::packed_nbytes(k, bits);
    }
    for (int i = 0; i < sr * sc; ++i) {
        if (mx) raw[scales + i] = 120 + i % 3;
        else { raw[scales + 2 * i] = 128; raw[scales + 2 * i + 1] = 60; }
    }
    const auto decode = [&](const std::vector<std::uint8_t>& bytes) {
        const auto layout = mfq::fp8sq::parse(dtype, bytes.data(), bytes.size());
        const auto metadata = mfq::fp8sq::row_metadata(bytes.data(), layout);
        auto blob = tensor(bytes).to(gpu), q = tensor(metadata.q).to(gpu);
        std::vector<std::int32_t> offsets(metadata.symbol_byte_offsets.begin(), metadata.symbol_byte_offsets.end());
        auto off = tensor(offsets).to(gpu);
        auto dense = mx ? mxfp8_sq_dequant_cuda(blob,q,off,layout.outputs,layout.width,br,bc,
            layout.scale_rows,layout.scale_columns,layout.palettes,layout.symbols,layout.scales,true)
            : fp8_128_sq_dequant_cuda(blob,q,off,layout.outputs,layout.width,2,layout.palettes,layout.symbols,layout.scales,true);
        auto host = dense.to(kCPU);
        for (int row = 0; row < layout.outputs; ++row) for (int col = 0; col < layout.width; ++col)
            require(host.data_ptr<float>()[row * layout.width + col] == mfq::fp8sq::decode_cpu(
                bytes.data(), layout, metadata.q[row], offsets[row], row, col), "FP8 CPU decode differs");
        return dense;
    };
    const auto original = decode(raw);
    const int row_begin = mx ? 8 : 128, col_begin = mx ? 32 : 128;
    const auto sliced = mfq::fp8sq::slice(dtype, raw, row_begin, n, col_begin, k);
    close(decode(sliced), original.narrow(0, row_begin, n-row_begin).narrow(1,col_begin,k-col_begin), 0, 0);
}

void mxfp4_benchmark(int n, int m = 128) {
    environment("MFQ_FORCE_SQ_TENSOR_CORE", "1");
    constexpr int k = 4096;
    const auto layout = mfq::sq::adaptive_layout(n, k, 120, n / 4 * 10, n / 4);
    std::vector<std::uint8_t> bytes(layout.bytes, 0);
    bytes[0]='S'; bytes[1]='Q'; bytes[2]='V'; bytes[3]='2'; bytes[4]=2; bytes[5]=120;
    for (int i=0;i<8;++i) { bytes[8+i]=std::uint64_t(n)>>(8*i); bytes[16+i]=std::uint64_t(k)>>(8*i); }
    auto offset = layout.symbols;
    for (int row=0;row<n;++row) {
        const int bits=row%4+1;
        mfq::sq::write_packed_bits(bytes,layout.q_selectors,row,2,bits-1);
        for (int col=0;col<k;++col) mfq::sq::write_packed_bits(bytes,offset,col,bits,(col*17+row*13)&((1<<bits)-1));
        offset+=std::size_t(k)*bits/8;
    }
    std::fill(bytes.begin()+layout.native_scales, bytes.end(), 120);
    const auto rows=mfq::sq::row_metadata(bytes.data(),layout);
    auto blob=tensor(bytes).to(gpu), q=tensor(rows.q).to(gpu);
    auto offsets=tensor(std::vector<std::int32_t>(rows.symbol_byte_offsets.begin(),rows.symbol_byte_offsets.end())).to(gpu);
    auto auxiliary=tensor(std::vector<std::int32_t>(rows.auxiliary_rows.begin(),rows.auxiliary_rows.end())).to(gpu);
    auto x=pattern({m,k},.5f).to(kFloat16);
    auto run=[&] { return mxfp4_sq_matmul_cuda(blob,q,offsets,auxiliary,x,0,n,k,120,n/4*10,n/4); };
    environment("MFQ_DISABLE_SQ_TENSOR_CORE","1");
    auto reference=run(); const float baseline=milliseconds(run);
    environment("MFQ_DISABLE_SQ_TENSOR_CORE","0");
    close(run(),reference,.02f,.006f);
    std::cout << "MXFP4-SQ M="<<m<<" N="<<n<<" K="<<k<<" tensor_core_ms="<<milliseconds(run)<<" scalar_ms="<<baseline<<'\n';
}

void native_benchmark() {
    auto selection=pattern({8,65536},1.0f);
    for (int count : {64, 1024, 2048}) {
        auto selected=[&] { return topk(selection,count,-1,true,true); };
        auto sorted=[&] { return sort(selection,-1,true); };
        const auto fast=selected(), ref=sorted();
        close(std::get<0>(fast),std::get<0>(ref).narrow(-1,0,count),0,0);
        close(std::get<1>(fast).to(kFloat32),std::get<1>(ref).narrow(-1,0,count).to(kFloat32),0,0);
        if (default_context(0)->supports_async_allocations()) {
            auto stream = stream_from_pool(false, 0);
            StreamGuard guard(stream);
            Graph graph;
            std::tuple<Tensor, Tensor> captured;
            graph.prepare_memory();
            captured = selected();
            MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
            captured = {};
            graph.capture_begin();
            captured = selected();
            graph.capture_end();
            graph.replay(); graph.replay();
            MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
            close(std::get<0>(captured), std::get<0>(fast), 0, 0);
            close(std::get<1>(captured).to(kFloat32), std::get<1>(fast).to(kFloat32), 0, 0);
        }
        std::cout << "TopK rows=8 width=65536 k="<<count<<" tiled_ms="<<milliseconds(selected)<<" full_sort_ms="<<milliseconds(sorted)<<'\n';
    }
    auto a=pattern({32,128,128},.5f).to(kBFloat16), b=pattern({32,128,128},.5f,7).to(kBFloat16);
    auto run=[&] { return matmul(a,b); };
    environment("MFQ_DISABLE_NATIVE_PARALLEL_BATCH_MATMUL","1");
    environment("MFQ_DISABLE_NATIVE_STRIDED_BATCH_MATMUL","1");
    const float loop=milliseconds(run);
    environment("MFQ_DISABLE_NATIVE_STRIDED_BATCH_MATMUL","0");
    std::cout << "BMM B=32 M=N=K=128 strided_ms="<<milliseconds(run)<<" loop_ms="<<loop<<'\n';
}
} // namespace

int main() try {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    fp8(true); fp8(false);
    for (int m : {9, 32, 128}) for (int n : {512,1024,2048,4096}) mxfp4_benchmark(n,m);
    native_benchmark();
    std::cout << "PASS SQ operators and native benchmarks\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
