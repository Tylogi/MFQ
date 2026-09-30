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
void nint(int m, int n, int gs, int groups) {
    const int width = gs * groups - 1;
    std::vector<std::uint8_t> bits(n), bytes((std::size_t(n) * gs * groups * 8 + 7) / 8 + 8, 0);
    std::vector<std::int64_t> offsets(n);
    std::uint64_t position = 0;
    for (int row = 0; row < n; ++row) {
        const int q = bits[row] = 1 + row % 8;
        offsets[row] = position;
        for (int k = 0; k < gs * groups; ++k) {
            const unsigned code = (k * 31 + row * 17) & ((1 << q) - 1);
            for (int j = 0; j < q; ++j, ++position) bytes[position / 8] |= ((code >> j) & 1) << (position % 8);
        }
    }
    auto packed = tensor(bytes).to(gpu), q = tensor(bits).to(gpu), off = tensor(offsets).to(gpu);
    auto sub = ones({n, groups}, TensorOptions{}.device(gpu).dtype(kUInt8));
    auto ns = full({n}, 0.001, TensorOptions{}.device(gpu).dtype(kFloat32));
    auto nm = ns * 7.0;
    auto x = pattern({m, width}, 0.5f).to(kFloat16);
    auto run = [&] { return nint_matmul_ws_cuda(packed, q, off, sub, sub, ns, nm, x, gs, {}, {}); };
    auto reference = [&] { return matmul(x, nint_decode_cuda(packed, q, off, sub, sub, ns, nm, width, gs).transpose(0, 1)); };
    close(run(), reference(), 0.002f, 0.002f);
    environment("MFQ_NINT_PANEL_PREFILL", "1");
    close(run(), reference(), 0.002f, 0.002f);
    const float panel_ms = n >= 512 ? milliseconds(run) : 0;
    environment("MFQ_NINT_FUSED_PREFILL", "1");
    close(run(), reference(), 0.002f, 0.002f);
    if (n >= 512) std::cout << "NINT M=" << m << " N=" << n << " K=" << width
        << " packed_ms=" << milliseconds(run) << " panel_ms=" << panel_ms
        << " decode_gemm_ms=" << milliseconds(reference) << '\n';
    environment("MFQ_NINT_FUSED_PREFILL", "0");
    environment("MFQ_NINT_PANEL_PREFILL", "0");
}
void gdn(int d, int t, bool transpose, bool tiled, bool initial, bool benchmark = false) {
    const int hq = benchmark ? 8 : 2, hv = benchmark ? 16 : 4;
    auto q = pattern({1, hq, t, d}, 0.07f);
    auto k = pattern({1, hq, t, d}, 0.09f, 13);
    auto v = pattern({1, hv, t, d}, 0.25f, 7);
    auto g = pattern({1, hv, t}, 0.025f) - 0.05;
    auto beta = pattern({1, hv, t}, 0.2f) + 0.5;
    auto state = initial ? pattern({1, hv, d, d}, 0.025f) : zeros({1, hv, d, d}, q.options());
    if (transpose) state = state.transpose(-2, -1).contiguous();
    auto run = [&] {
        auto local = state.clone();
        if (transpose) return tiled ? gdn_inplace_transposed_tiled_cuda(q,k,v,g,beta,local)
            : gdn_inplace_transposed_cuda(q,k,v,g,beta,local);
        return tiled ? gdn_inplace_tiled_cuda(q,k,v,g,beta,local) : gdn_inplace_cuda(q,k,v,g,beta,local);
    };
    environment("MFQ_GDN_CHUNKED", "0");
    const auto reference = run();
    const float baseline_ms = benchmark ? milliseconds(run) : 0;
    environment("MFQ_GDN_CHUNKED", "1");
    const auto result = run();
    close(result[0], reference[0], 2e-5f, 2e-4f);
    close(result[1], reference[1], 2e-5f, 2e-4f);
    if (d == 128 && !benchmark && initial && transpose && !tiled && default_context(0)->supports_async_allocations()) {
        auto stream = stream_from_pool(false, 0);
        StreamGuard guard(stream);
        Graph graph;
        std::vector<Tensor> captured;
        graph.prepare_memory();
        captured = run();
        MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
        captured.clear();
        graph.capture_begin();
        captured = run();
        graph.capture_end();
        graph.replay(); graph.replay();
        MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
        close(captured[0], reference[0], 2e-5f, 2e-4f);
        close(captured[1], reference[1], 2e-5f, 2e-4f);
    }
    if (benchmark) std::cout << "GDN T=" << t << " D=" << d << " chunk_ms=" << milliseconds(run)
        << " recurrent_ms=" << baseline_ms << '\n';
}

void fp8(bool mx, bool benchmark = false) {
    environment("MFQ_FP8_SQ_FUSED_PREFILL", "1");
    const std::string dtype = mx ? "MXFP8-SQ" : "FP8-128SQ";
    const int n = benchmark ? 4096 : mx ? 17 : 257, k = benchmark ? 4096 : mx ? 96 : 257;
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
    const auto decode = [&](const std::vector<std::uint8_t>& bytes, bool run_matmul) {
        const auto layout = mfq::fp8sq::parse(dtype, bytes.data(), bytes.size());
        const auto metadata = mfq::fp8sq::row_metadata(bytes.data(), layout);
        auto blob = tensor(bytes).to(gpu), q = tensor(metadata.q).to(gpu);
        std::vector<std::int32_t> offsets(metadata.symbol_byte_offsets.begin(), metadata.symbol_byte_offsets.end());
        auto off = tensor(offsets).to(gpu);
        auto dense = mx ? mxfp8_sq_dequant_cuda(blob,q,off,layout.outputs,layout.width,br,bc,
            layout.scale_rows,layout.scale_columns,layout.palettes,layout.symbols,layout.scales,true)
            : fp8_128_sq_dequant_cuda(blob,q,off,layout.outputs,layout.width,2,layout.palettes,layout.symbols,layout.scales,true);
        auto host = dense.to(kCPU);
        if (!benchmark) for (int row = 0; row < layout.outputs; ++row) for (int col = 0; col < layout.width; ++col)
            require(host.data_ptr<float>()[row * layout.width + col] == mfq::fp8sq::decode_cpu(
                bytes.data(), layout, metadata.q[row], offsets[row], row, col), "FP8 CPU decode differs");
        if (run_matmul) {
            auto x = pattern({benchmark ? 128 : 65, layout.width}, 0.5f).to(kFloat16);
            auto run = [&] { return mx ? mxfp8_sq_matmul_cuda(blob,q,off,x,layout.outputs,layout.width,br,bc,
                layout.scale_rows,layout.scale_columns,layout.palettes,layout.symbols,layout.scales)
                : fp8_128_sq_matmul_cuda(blob,q,off,x,layout.outputs,layout.width,2,layout.palettes,layout.symbols,layout.scales); };
            close(run(), matmul(x, dense.to(kFloat16).transpose(0,1)), 0.02f, 0.002f);
            if (benchmark) {
                const float packed=milliseconds(run);
                environment("MFQ_FP8_SQ_FUSED_PREFILL","0");
                std::cout << dtype << " M=128 N="<<layout.outputs<<" K="<<layout.width<<" tensor_core_ms="<<packed<<" decode_gemm_ms="<<milliseconds(run)<<'\n';
                environment("MFQ_FP8_SQ_FUSED_PREFILL","1");
            }
        }
        return dense;
    };
    const auto original = decode(raw, true);
    if (benchmark) return;
    const int row_begin = mx ? 8 : 128, col_begin = mx ? 32 : 128;
    const auto sliced = mfq::fp8sq::slice(dtype, raw, row_begin, n, col_begin, k);
    close(decode(sliced, true), original.narrow(0, row_begin, n-row_begin).narrow(1,col_begin,k-col_begin), 0, 0);
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
    for (int gs : {5, 28, 64}) for (int m : {9, 33, 65}) nint(m, 9, gs, 3);
    nint(128, 512, 32, 32);
    nint(128, 4096, 32, 128);
    nint(9, 4096, 32, 128);
    nint(32, 4096, 32, 128);
    for (int d : {32, 64, 128}) for (bool transpose : {false, true})
        for (bool tiled : {false, true}) for (bool initial : {false, true}) gdn(d, 137, transpose, tiled, initial);
    gdn(128, 2048, true, false, true, true);
    fp8(true); fp8(false);
    fp8(true,true); fp8(false,true);
    for (int m : {9, 32, 128}) for (int n : {512,1024,2048,4096}) mxfp4_benchmark(n,m);
    native_benchmark();
    std::cout << "PASS packed prefill and chunked GDN\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
