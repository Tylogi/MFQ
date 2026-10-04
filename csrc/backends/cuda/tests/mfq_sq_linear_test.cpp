#include "../ops/quant_linear.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <numeric>
#include <stdexcept>

namespace {
using namespace mfq::cuda;
using namespace mfq::cuda::internal;
Mxfp4SqWeight weight(const std::vector<std::uint8_t>& bytes, bool cuda) {
    const auto layout = mfq::sq::parse(bytes.data(), bytes.size());
    const auto rows = mfq::sq::row_metadata(bytes.data(), layout);
    Mxfp4SqWeight result;
    result.blob = tensor(bytes);
    result.row_q = tensor(rows.q);
    result.row_symbol_byte_offsets = tensor(std::vector<std::int32_t>(rows.symbol_byte_offsets.begin(), rows.symbol_byte_offsets.end()));
    result.row_auxiliary = tensor(std::vector<std::int32_t>(rows.auxiliary_rows.begin(), rows.auxiliary_rows.end()));
    if (cuda) {
        const Device gpu{DeviceType::cuda, 0};
        result.blob = result.blob.to(gpu); result.row_q = result.row_q.to(gpu);
        result.row_symbol_byte_offsets = result.row_symbol_byte_offsets.to(gpu);
        result.row_auxiliary = result.row_auxiliary.to(gpu);
    }
    result.bits=layout.bits; result.out=layout.outputs; result.neuron_len=layout.width;
    result.matrix_scale_base=layout.base; result.sq4_rows=layout.sq4_rows;
    result.q_sum=std::accumulate(rows.q.begin(),rows.q.end(),std::int64_t{0});
    return result;
}
std::vector<float> host(const Tensor& value) {
    auto cpu=value.contiguous().to(kCPU,kFloat32);
    return {cpu.data_ptr<float>(),cpu.data_ptr<float>()+cpu.numel()};
}
void close(const Tensor& value, const std::vector<float>& reference, float tolerance) {
    auto actual=host(value);
    if (actual.size()!=reference.size()) throw std::runtime_error("SQ linear output shape differs");
    for (std::size_t i=0;i<actual.size();++i)
        if (!std::isfinite(actual[i]) || std::abs(actual[i]-reference[i])>tolerance)
            throw std::runtime_error("SQ linear numerical mismatch at "+std::to_string(i));
}
void fp8_cpu(CudaExecutionContext& execution, bool mx) {
    const int n = 3, k = mx ? 32 : 128;
    const int br = mx ? 1 : 128, bc = k, sr = mx ? n : 1;
    const int symbols = 44 + 4 + 256, scales = symbols + n * k;
    std::vector<std::uint8_t> bytes(scales + sr * (mx ? 1 : 2), 0);
    bytes[0]=mx?'M':'F'; bytes[1]='8'; bytes[2]='S'; bytes[3]='Q'; bytes[4]=1;
    bytes[5]=mx?1:3; bytes[8]=br; bytes[10]=bc;
    const auto write=[&](int at, std::uint64_t v) { for (int i=0;i<8;++i) bytes[at+i]=v>>(i*8); };
    write(12,n); write(20,k); write(28,sr); write(36,1);
    bytes[44]=255; bytes[45]=1; // three q=8 selectors
    for (int q=1;q<=7;++q) for (int j=0;j<(1<<q);++j)
        bytes[48+(1<<q)-2+j]=j==127?128:j;
    std::vector<float> dense(n*k), values(3*k), expected(3*n,0);
    for (int r=0;r<n;++r) for (int c=0;c<k;++c) {
        const bool negative=(r+c)%3==0;
        bytes[symbols+r*k+c]=negative?0xb8:0x38;
        dense[r*k+c]=negative?-1.0f:1.0f;
    }
    if (mx) std::fill(bytes.begin()+scales,bytes.end(),127);
    else bytes[scales+1]=0x3c;
    for (int i=0;i<3*k;++i) values[i]=float(i%11-5)/16;
    for (int m=0;m<3;++m) for (int r=0;r<n;++r) for (int c=0;c<k;++c)
        expected[m*n+r]+=values[m*k+c]*dense[r*k+c];
    Fp8SqWeight w;
    w.dtype=mx?"MXFP8-SQ":"FP8-128SQ"; w.blob=tensor(bytes);
    const auto layout=mfq::fp8sq::parse(w.dtype,bytes.data(),bytes.size());
    const auto rows=mfq::fp8sq::row_metadata(bytes.data(),layout);
    w.row_q=tensor(rows.q);
    w.row_symbol_byte_offsets=tensor(std::vector<std::int32_t>(rows.symbol_byte_offsets.begin(),rows.symbol_byte_offsets.end()));
    w.out=n; w.neuron_len=k;
    QuantLinear layer; layer.kind=QuantLinearKind::Fp8Sq; layer.fp8_sq.weight=w;
    for (auto dtype : {kFloat32,kFloat16}) close(layer.fp8_sq.forward(tensor(values).reshape({3,k}).to(dtype)),expected,0);
    close(quant_linear_reference_weight(layer),dense,0);
    QuantLinearGroup group; group.layers.push_back(layer); group.outs={n};
    close(make_fp32_quant_group(execution, group).w,dense,0);
}

}

int main(int argc, char** argv) try {
    if (argc!=2) throw std::runtime_error("expected SQ fixture directory");
    const std::string prefix=std::string(argv[1])+"/sqv2-n7-k96-b120";
    std::ifstream packed(prefix+".sq",std::ios::binary), dense_file(prefix+".f32",std::ios::binary);
    if (!packed || !dense_file) throw std::runtime_error("missing SQ fixture");
    const std::vector<std::uint8_t> bytes{std::istreambuf_iterator<char>(packed),{}};
    std::vector<float> dense(7*96); dense_file.read(reinterpret_cast<char*>(dense.data()),dense.size()*sizeof(float));
    std::vector<float> values(3*96), expected(3*7,0);
    for (std::size_t i=0;i<values.size();++i) values[i]=float(int(i%17)-8)/16;
    for (int m=0;m<3;++m) for (int n=0;n<7;++n) for (int k=0;k<96;++k)
        expected[m*7+n]+=values[m*96+k]*dense[n*96+k];
    auto x=tensor(values).reshape({3,96});
    CudaExecutionContext execution;
    Mxfp4SqLinear cpu{weight(bytes,false)};
    close(cpu.forward(x),expected,1e-5f);
    QuantLinear cpu_layer; cpu_layer.kind=QuantLinearKind::Mxfp4Sq; cpu_layer.mxfp4_sq=cpu;
    QuantLinearGroup cpu_group; cpu_group.layers.push_back(cpu_layer); cpu_group.outs={7};
    close(make_fp32_quant_group(execution, cpu_group).w,dense,0);
    fp8_cpu(execution, true); fp8_cpu(execution, false);
    int devices=0;
    if (cudaGetDeviceCount(&devices)!=cudaSuccess || devices==0) return 77;
    auto device_x=x.to(Device{DeviceType::cuda,0}).to(kFloat16);
    // Two logical shards on one GPU exercise packed slicing, dispatch and both
    // gather/reduce paths. The separate multi-device transport is unchanged.
    for (const auto axis : {TensorParallelAxis::Output,TensorParallelAxis::Input}) {
        QuantLinear layer;
        layer.kind=QuantLinearKind::Mxfp4Sq; layer.tensor_parallel_axis=axis;
        layer.logical_out=7; layer.logical_neuron_len=96;
        for (int index=0;index<2;++index) {
            QuantLinearShard shard;
            shard.kind=layer.kind; shard.device=0;
            shard.output_begin=axis==TensorParallelAxis::Output?(index==0?0:3):0;
            shard.output_end=axis==TensorParallelAxis::Output?(index==0?3:7):7;
            shard.input_begin=axis==TensorParallelAxis::Input?(index==0?0:32):0;
            shard.input_end=axis==TensorParallelAxis::Input?(index==0?32:96):96;
            std::vector<std::int64_t> rows(shard.output_end-shard.output_begin);
            std::iota(rows.begin(),rows.end(),shard.output_begin);
            shard.mxfp4_sq.weight=weight(mfq::sq::select_rows(bytes,rows,shard.input_begin,shard.input_end),true);
            layer.tensor_parallel_shards.push_back(std::move(shard));
        }
        close(layer.forward_tensor_parallel_flat(
            execution, device_x, mfq_nullopt, 0), expected, 0.002f);
    }
    std::cout<<"PASS SQ CPU linear and packed input/output shards\n";
    return 0;
} catch (const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
