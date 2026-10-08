#include "mfq_cuda_quant_ops.h"

#include <cuda_runtime_api.h>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
using namespace mfq::cuda;
void single_path(bool enabled) {
#ifdef _WIN32
    _putenv_s("MFQ_NINT_SINGLE_ROW",enabled ? "1" : "0");
#else
    setenv("MFQ_NINT_SINGLE_ROW",enabled ? "1" : "0",1);
#endif
}

void require(bool condition, const char *message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void exact_half(const Tensor &actual, const Tensor &expected) {
    require(actual.sizes() == expected.sizes() &&
            actual.scalar_type() == kFloat16 &&
            expected.scalar_type() == kFloat16, "NINT output contract mismatch");
    auto a = actual.to(kCPU).contiguous();
    auto b = expected.to(kCPU).contiguous();
    require(std::memcmp(a.data_ptr(), b.data_ptr(),
            static_cast<std::size_t>(a.numel()) * sizeof(std::uint16_t)) == 0,
            "NINT q8 output differs from the original generic path");
}

void check_case(int gs, int ng, int m, int width_tail, int storage_offset,
                int mode, int &cases, int &graphs) {
    const Device gpu{DeviceType::cuda, 0};
    const auto options = TensorOptions().device(gpu);
    constexpr int n = 5;  // Exercises a partial four-warp output block.
    const int k = gs * ng;
    const int stride = (k + 3) / 4 * 4;
    std::vector<std::uint8_t> bytes(n * stride + storage_offset, 0);
    std::vector<std::int64_t> offsets(n);
    for (int row = 0; row < n; ++row) {
        offsets[row] = static_cast<std::int64_t>(row) * stride * 8;
        for (int col = 0; col < k; ++col) {
            bytes[storage_offset + row * stride + col] =
                static_cast<std::uint8_t>((row * 97 + col * 31) % 256);
        }
    }
    auto q = tensor(bytes).to(gpu).narrow(0, storage_offset, n * stride);
    auto bits = tensor(std::vector<std::uint8_t>(n, 8)).to(gpu);
    auto off = tensor(offsets).to(gpu);
    std::vector<std::uint8_t> scales(n * ng), minima(n * ng);
    for (int i = 0; i < n * ng; ++i) {
        scales[i] = static_cast<std::uint8_t>(i % 13);
        minima[i] = static_cast<std::uint8_t>(i % 7);
    }
    auto s = tensor(scales).reshape({n, ng}).to(gpu);
    auto mn = tensor(minima).reshape({n, ng}).to(gpu);
    auto ns = tensor<float>({0.0001f, 0.0002f, 0.0003f, 0.0004f, 0.0005f}).to(gpu);
    auto nm = tensor<float>({0.01f, 0.02f, 0.03f, 0.04f, 0.05f}).to(gpu);
    std::vector<float> input(m * (k - width_tail));
    for (std::size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(static_cast<int>(i % 37) - 18) / 8;
    }
    auto x = tensor(input).reshape({m, k - width_tail}).to(gpu, kFloat16);
    for (std::size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(static_cast<int>(i % 53) - 26) / 2;
    }
    auto gate = tensor(input).reshape({m, k - width_tail}).to(gpu, kFloat16);
    auto qx = empty({m, k}, options.dtype(kInt8));
    auto xs = empty({m, ng}, options.dtype(kFloat32));
    auto run = [&](bool fast) {
        if (mode == 0) {
            return fast
                ? nint_matmul_q8_ws_cuda(q, bits, off, s, mn, ns, nm, x, gs, qx, xs)
                : nint_matmul_ws_cuda(q, bits, off, s, mn, ns, nm, x, gs, qx, xs);
        }
        return fast
            ? nint_matmul_input_mul_q8_ws_cuda(
                q, bits, off, s, mn, ns, nm, x, gate, mode, gs, qx, xs)
            : nint_matmul_input_mul_ws_cuda(
                q, bits, off, s, mn, ns, nm, x, gate, mode, gs, qx, xs);
    };
    const auto expected = run(false);
    const auto actual = run(true);
    MFQ_NATIVE_CUDA_CHECK(cudaDeviceSynchronize());
    exact_half(actual, expected);
    ++cases;

    // Compare both specialization branches during capture/replay. Odd group
    // widths and offset storage also exercise the existing generic fallback.
    if (ng == 3 && width_tail == 0 && storage_offset == 0 && (m == 1 || m == 8)) {
        const auto stream = mfq_get_stream_from_pool(false);
        MfqCudaStreamGuard guard(stream);
        MfqCudaGraph graph;
        mfq_prepare_cuda_graph_memory(graph);
        Tensor result = run(true);
        MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
        result = Tensor();
        graph.capture_begin();
        result = run(true);
        graph.capture_end();
        for (int repeat = 0; repeat < 3; ++repeat) {
            graph.replay();
            MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
            exact_half(result, expected);
        }
        ++graphs;
    }
}
void check_single(int gs,int width,int preset,int storage_offset,int padding,int n,int& cases) {
    const Device gpu{DeviceType::cuda,0};
    const int ng=(width+gs-1)/gs,k=ng*gs;
    std::vector<std::uint8_t> row_bits(n);
    std::vector<std::int64_t> offsets(n);
    std::uint64_t total=5; // Non-byte-aligned rows exercise every word shift.
    for(int row=0;row<n;++row) {
        row_bits[row]=preset ? preset : 1+row%8;
        offsets[row]=static_cast<std::int64_t>(total);total+=std::uint64_t(k)*row_bits[row];
    }
    const auto packed_bytes=static_cast<std::size_t>((total+7)/8)+padding;
    std::vector<std::uint8_t> bytes(storage_offset+packed_bytes,0);
    for(int row=0;row<n;++row)for(int column=0;column<k;++column) {
        const auto value=std::uint32_t((row*97+column*31+19)&((1u<<row_bits[row])-1));
        const auto bit=std::uint64_t(offsets[row])+std::uint64_t(column)*row_bits[row];
        for(int b=0;b<row_bits[row];++b)if((value>>b)&1u)
            bytes[storage_offset+(bit+b)/8]|=std::uint8_t(1u<<((bit+b)&7));
    }
    auto q=tensor(bytes).to(gpu).narrow(0,storage_offset,packed_bytes);
    auto bits=tensor(row_bits).to(gpu),off=tensor(offsets).to(gpu);
    std::vector<std::uint8_t> scales(n*ng),minima(n*ng);
    std::vector<float> outer(n),minimum(n),input(width);
    for(int i=0;i<n*ng;++i){scales[i]=std::uint8_t((i*13)%37);minima[i]=std::uint8_t(i%19);}
    for(int row=0;row<n;++row){outer[row]=.000123f*float(row%13-6);minimum[row]=.004731f*float(row%5-2);}
    for(int column=0;column<width;++column)input[column]=std::sin(float(column+7)*.193f)*.31f;
    auto s=tensor(scales).reshape({n,ng}).to(gpu),mn=tensor(minima).reshape({n,ng}).to(gpu);
    auto ns=tensor(outer).to(gpu),nm=tensor(minimum).to(gpu);
    auto x=tensor(input).reshape({1,width}).to(gpu,kFloat16);
    const auto options=TensorOptions().device(gpu);
    auto qx=empty({1,k},options.dtype(kInt8)),xs=empty({1,ng},options.dtype(kFloat32));
    const auto run=[&]{return nint_matmul_ws_cuda(q,bits,off,s,mn,ns,nm,x,gs,qx,xs);};
    single_path(false);auto expected=run();
    single_path(true);auto actual=run();exact_half(actual,expected);++cases;
    const auto stream=mfq_get_stream_from_pool(false);MfqCudaStreamGuard guard(stream);
    MfqCudaGraph graph;mfq_prepare_cuda_graph_memory(graph);
    actual=run();MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));actual=Tensor{};
    graph.capture_begin();actual=run();graph.capture_end();
    for(int step=0;step<2;++step) {
        for(int column=0;column<width;++column)input[column]=std::sin(float(column+step*31)*.173f)*.13f;
        x.copy_(tensor(input).reshape({1,width}).to(gpu,kFloat16));
        single_path(false);expected=run();
        graph.replay();MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));exact_half(actual,expected);++cases;
    }
    single_path(false);
}

void check_float_input(int gs,int rows,int kind,int mode,int& ordinary,int& changing) {
    const Device gpu{DeviceType::cuda,0};
    const auto options=TensorOptions().device(gpu);
    constexpr int n=9,ng=3;
    const int padded=gs*ng,width=padded-1;
    std::vector<std::uint8_t> row_bits(n),packed;
    std::vector<std::int64_t> offsets(n);
    std::uint64_t end=kind==0?5:0;
    for(int row=0;row<n;++row) {
        row_bits[row]=kind==0?1+row%8:8;
        if(kind!=0)end=(end+31)/32*32;
        offsets[row]=end;end+=std::uint64_t(padded)*row_bits[row];
    }
    packed.resize((end+7)/8);
    for(int row=0;row<n;++row)for(int col=0;col<padded;++col) {
        const unsigned value=(row*97+col*31+19)&((1u<<row_bits[row])-1);
        const auto start=std::uint64_t(offsets[row])+std::uint64_t(col)*row_bits[row];
        for(int b=0;b<row_bits[row];++b)if((value>>b)&1u)
            packed[(start+b)/8]|=std::uint8_t(1u<<((start+b)&7));
    }
    auto q=tensor(packed).to(gpu),bits=tensor(row_bits).to(gpu),off=tensor(offsets).to(gpu);
    auto sub=tensor(std::vector<std::uint8_t>(n*ng,7)).reshape({n,ng}).to(gpu);
    auto minima=tensor(std::vector<std::uint8_t>(n*ng,3)).reshape({n,ng}).to(gpu);
    auto scale=tensor(std::vector<float>(n,1e-9f)).to(gpu);
    auto minimum=tensor(std::vector<float>(n,2e-9f)).to(gpu);
    if(kind==2) {
        std::vector<std::uint8_t> codes(n*ng*32);
        for(size_t i=0;i<codes.size();++i)codes[i]=std::uint8_t(i*31+17);
        q=tensor(codes).reshape({n,ng,32}).to(gpu);
        scale=tensor(std::vector<float>(n*ng,1e-6f)).reshape({n,ng}).to(gpu,kFloat16);
    }
    auto x=empty({rows,width},options.dtype(kFloat32));
    auto gate=empty({rows,width},options.dtype(kind==1?kFloat16:kFloat32));
    Tensor qx[2]={empty({rows,padded},options.dtype(kInt8)),empty({rows,padded},options.dtype(kInt8))};
    Tensor xs[2]={empty({rows,ng},options.dtype(kFloat32)),empty({rows,ng},options.dtype(kFloat32))};
    const auto change=[&](int step) {
        const float midpoint=1.f+1.f/2048.f;
        const std::vector<float> edges={0.f,-0.f,std::ldexp(1.f,-25),-std::ldexp(1.f,-25),
            std::ldexp(3.f,-25),std::ldexp(1.f,-24),std::ldexp(1.f,-14),
            std::nextafter(midpoint,0.f),midpoint,std::nextafter(midpoint,2.f),
            1.f+3.f/2048.f,-midpoint,127.4999695f,65504.f,-65504.f};
        std::vector<float> values(rows*width),gates(rows*width);
        for(size_t i=0;i<values.size();++i) {
            values[i]=step==0?edges[i%edges.size()]:step==1?0.f:
                std::sin(float(i+step*31)*.193f)*(step==2?.031339f:31.733f);
            gates[i]=step==1?0.f:float(int(i%7)-3)*29.333f;
        }
        x.copy_(tensor(values).reshape({rows,width}).to(gpu));
        gate.copy_(tensor(gates).reshape({rows,width}).to(gpu,gate.scalar_type()));
    };
    const auto run=[&](bool direct) {
        const int slot=direct?1:0;
        auto activation=direct?x:x.to(kFloat16);
        if(kind==2)return nint8_zero_gemv_ws_cuda(q,scale,activation,qx[slot],xs[slot]);
        if(mode==0)return kind==1
            ?nint_matmul_q8_ws_cuda(q,bits,off,sub,minima,scale,minimum,activation,gs,qx[slot],xs[slot])
            :nint_matmul_ws_cuda(q,bits,off,sub,minima,scale,minimum,activation,gs,qx[slot],xs[slot]);
        auto gates=direct?gate:gate.to(kFloat16);
        return kind==1
            ?nint_matmul_input_mul_q8_ws_cuda(q,bits,off,sub,minima,scale,minimum,activation,gates,mode,gs,qx[slot],xs[slot])
            :nint_matmul_input_mul_ws_cuda(q,bits,off,sub,minima,scale,minimum,activation,gates,mode,gs,qx[slot],xs[slot]);
    };
    const auto bytes=[](const Tensor& actual,const Tensor& expected) {
        require(actual.sizes()==expected.sizes() && actual.scalar_type()==expected.scalar_type(),
            "NINT Float quantizer workspace contract mismatch");
        auto a=actual.cpu().contiguous(),b=expected.cpu().contiguous();
        require(std::memcmp(a.data_ptr(),b.data_ptr(),a.numel()*a.element_size())==0,
            "NINT Float quantizer codes/scales differ from separately rounded Half input");
    };
    change(0);auto expected=run(false);auto actual=run(true);
    exact_half(actual,expected);bytes(qx[1],qx[0]);bytes(xs[1],xs[0]);++ordinary;
    const auto stream=mfq_get_stream_from_pool(false);MfqCudaStreamGuard guard(stream);
    MfqCudaGraph graph;mfq_prepare_cuda_graph_memory(graph);
    actual=run(true);MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));actual=Tensor{};
    graph.capture_begin();actual=run(true);graph.capture_end();
    for(int step=1;step<=3;++step) {
        change(step);expected=run(false);graph.replay();
        MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
        exact_half(actual,expected);bytes(qx[1],qx[0]);bytes(xs[1],xs[0]);++changing;
    }
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
        int cases = 0, graphs = 0;
        int float_ordinary=0,float_changing=0;
        for(int gs:{5,24,28,32,48,64})for(int rows:{1,3,8})
            for(int kind:{0,1})for(int mode:{0,1,2})
                check_float_input(gs,rows,kind,mode,float_ordinary,float_changing);
        for(int rows:{1,3,8})check_float_input(32,rows,2,0,float_ordinary,float_changing);
        std::cout<<"NINT Float activation rounding ordinary="<<float_ordinary
                 <<" changing_graph="<<float_changing<<" quantized/scales/output bytes exact PASS\n";
        int single_cases=0;
        for(int gs:{5,24,28,32,48,64})for(int width:{53,640,2560})
            for(int preset:{0,4,5,8})for(int offset:{0,1})for(int padding:{0,8})
                check_single(gs,width,preset,offset,padding,9,single_cases);
        for(int gs:{24,28}) {
            check_single(gs,2560,5,0,8,640,single_cases);
            check_single(gs,640,0,0,8,2560,single_cases);
            check_single(gs,2560,5,0,8,6144,single_cases);
        }
        std::cout<<"NINT single-row original Half bits ordinary/changing graphs cases="<<single_cases<<" PASS\n";
        for (int gs : {4, 12, 28, 32, 48, 64, 5, 47, 63}) {
            for (int ng : {1, 3, 22}) {
                for (int m = 1; m <= 8; ++m) {
                    for (int tail : {0, 1}) {
                        for (int offset : {0, 1}) {
                            for (int mode : {0, 1, 2}) {
                                check_case(gs, ng, m, tail, offset, mode, cases, graphs);
                            }
                        }
                    }
                }
            }
        }
        std::cout << "NINT q8 bit-exact FP16 cases=" << cases
                  << " graph_cases=" << graphs << " replays_per_case=3\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
