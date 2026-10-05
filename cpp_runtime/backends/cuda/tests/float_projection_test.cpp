#include "float_projection.h"
#include "gated_residual.h"
#include "cuda_execution.h"
#include "storage/weight_loader.h"
#include "qwen4_exp.h"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace tb = mfq_tensor_backend;
namespace {
template<class T> tb::Tensor upload(const std::vector<T>& values, tb::ScalarType dtype,
        const std::vector<int64_t>& shape) {
    auto host = tb::empty({int64_t(values.size())}, tb::TensorOptions().device(tb::kCPU).dtype(dtype));
    std::memcpy(host.data_ptr(), values.data(), values.size()*sizeof(T));
    return host.reshape(shape).to(tb::kCUDA);
}

struct Fixture { NintWeight weight; std::vector<float> dense; };
Fixture make_weight(int out, int width, int group_size, bool zero = false) {
    Fixture fixture;
    auto& w = fixture.weight;
    w.out=out; w.neuron_len=width; w.gs=group_size; w.ng=(width+group_size-1)/group_size;
    w.q8_zero=zero;
    std::vector<uint8_t> bits(out), scales(out*w.ng), minima(out*w.ng);
    std::vector<int64_t> offsets(out);
    std::vector<float> row_scale(out), row_min(out), zero_scale(out*w.ng);
    int64_t total_bits=0;
    for (int row=0; row<out; ++row) {
        bits[row]=1+row%8;
        offsets[row]=total_bits;
        total_bits+=w.ng*group_size*(zero ? 8 : bits[row]);
        row_scale[row]=float(1+row%13)*0.0001f;
        row_min[row]=float(1+row%7)*0.00001f;
    }
    std::vector<uint8_t> packed((total_bits+7)/8+8,0);
    fixture.dense.resize(out*width);
    for (int row=0; row<out; ++row) for (int64_t g=0; g<w.ng; ++g) {
        const auto meta=row*w.ng+g;
        scales[meta]=1+(row+g*7)%24;
        minima[meta]=(row+g)%4;
        zero_scale[meta]=float(mfq_half(float(1+meta%13)*0.003f));
        for (int j=0; j<group_size; ++j) {
            const int64_t col=g*group_size+j;
            const unsigned q=zero ? unsigned((row*17+col*3)%255) : unsigned((row*17+col*3)% (1<<bits[row]));
            if (zero) packed[meta*group_size+j]=uint8_t(q);
            else {
                const auto bit=offsets[row]+col*bits[row];
                for (int b=0; b<bits[row]; ++b)
                    packed[(bit+b)/8] |= uint8_t(((q>>b)&1) << ((bit+b)%8));
            }
            if (col<width) fixture.dense[row*width+col]=zero
                ? zero_scale[meta]*float(static_cast<int8_t>(q))
                : (row_scale[row]*float(scales[meta]))*float(q)-row_min[row]*float(minima[meta]);
        }
    }
    w.q_packed=upload(packed,tb::kUInt8,{int64_t(packed.size())});
    if (zero) w.q8_zero_scale=upload(zero_scale,tb::kFloat32,{out,w.ng}).to(tb::kFloat16);
    else {
        w.row_q_bits=upload(bits,tb::kUInt8,{out});
        w.row_q_bit_offsets=upload(offsets,tb::kInt64,{out});
        w.sub_scale=upload(scales,tb::kUInt8,{out,w.ng});
        w.sub_min=upload(minima,tb::kUInt8,{out,w.ng});
        w.neuron_scale=upload(row_scale,tb::kFloat32,{out});
        w.neuron_min=upload(row_min,tb::kFloat32,{out});
    }
    return fixture;
}

void compare(const tb::Tensor& actual,const std::vector<float>& expected,const std::string& name) {
    if (actual.scalar_type()!=tb::kFloat32 || actual.numel()!=int64_t(expected.size()))
        throw std::runtime_error(name+" output dtype/shape mismatch");
    auto host=actual.cpu().contiguous();
    for (size_t i=0; i<expected.size(); ++i) {
        const double value=host.data_ptr<float>()[i], ref=expected[i];
        if (!std::isfinite(value) || std::abs(value-ref)>1e-5+1e-4*std::abs(ref))
            throw std::runtime_error(name+" FP32 projection differs from FP64 oracle at "+std::to_string(i));
    }
}

void synthetic(int outputs,int width,int group,int batch,bool zero) {
    auto fixture=make_weight(outputs,width,group,zero);
    std::vector<float> input(batch*width), expected(batch*outputs);
    for (size_t i=0; i<input.size(); ++i) input[i]=std::sin(float(i)*0.031f)*0.1f;
    for (int sample=0; sample<batch; ++sample) for (int row=0; row<outputs; ++row) {
        double sum=0;
        for (int k=0; k<width; ++k) sum+=double(input[sample*width+k])*fixture.dense[row*width+k];
        expected[sample*outputs+row]=float(sum);
    }
    auto x=upload(input,tb::kFloat32,{batch,width});
    compare(nint_float_projection_cuda(fixture.weight,x),expected,"synthetic");
}

std::vector<float> read(const std::filesystem::path& path,size_t count) {
    std::vector<float> values(count);
    std::ifstream stream(path,std::ios::binary);
    stream.read(reinterpret_cast<char*>(values.data()),count*sizeof(float));
    if (!stream || stream.peek()!=std::char_traits<char>::eof()) throw std::runtime_error("invalid float oracle");
    return values;
}
void canonical(const std::filesystem::path& root) {
    CudaExecutionContext execution;
    std::ifstream manifest(root/"cases.txt");
    int out,width,batch,cases=0;
    std::string name;
    while (manifest>>name>>out>>width>>batch) {
        auto source=mfq::open_model_source((root/(name+".mfq")).string());
        if (require_tensor(*source,"linear.weight").dtype!="NINT") continue;
        auto projection=mfq::cuda::weight_loader::residual_linear(execution,*source,"linear.weight");
        auto x=upload(read(root/(name+".input.f32"),batch*width),tb::kFloat32,{batch,width});
        compare(projection(execution,x),read(root/(name+".expected.f32"),batch*out),name);
        std::cout<<"float_projection "<<name<<" PASS\n";
        ++cases;
    }
    if (!manifest.eof() || !cases) throw std::runtime_error("invalid canonical projection fixtures");
    std::cout<<"canonical_float_cases="<<cases<<'\n';
}

void residual() {
    constexpr int hidden=16, streams=4, width=hidden*streams;
    auto down=make_weight(8,width,32), up=make_weight(width,8,32), inject=make_weight(streams,width,32);
    auto dense=[](const Fixture& f) { return upload(f.dense,tb::kFloat32,{f.weight.out,f.weight.neuron_len}); };
    std::vector<float> input(3*width), norm(width);
    for (size_t i=0; i<input.size(); ++i) input[i]=std::sin(float(i)*0.031f);
    auto x=upload(input,tb::kFloat32,{1,3,width}), n=upload(norm,tb::kFloat32,{width});
    auto projection=[](const Fixture& f)->GatedResidualProjection {
        return [w=f.weight](CudaExecutionContext&,const tb::Tensor& x) { return nint_float_projection_cuda(w,x); };
    };
    CudaExecutionContext execution;
    const auto actual=gated_residual_pre_projected(execution,x,n,projection(down),projection(up),
        projection(inject),hidden,streams,1e-6);
    const auto expected=mfq_qwen4_exp::gated_residual_pre(x,n,dense(down),dense(up),dense(inject),hidden,streams,1e-6);
    for (int i : {0,1,2}) {
        auto host=expected[i].cpu().contiguous();
        compare(actual[i],{host.data_ptr<float>(),host.data_ptr<float>()+host.numel()},"packed gated residual");
    }
}
}

int main(int argc,char** argv) try {
    if (argc>2) throw std::runtime_error("expected optional canonical fixture directory");
    if (argc==2) canonical(argv[1]);
    else {
        for (int group : {32,48,64}) for (int batch : {1,3,11}) synthetic(17,77,group,batch,false);
        for (int batch : {1,3,11}) synthetic(17,77,32,batch,true);
        for (auto shape : {std::pair<int,int>{320,10240},{10240,320},{4,10240}})
            synthetic(shape.first,shape.second,64,2,false);
        residual();
        std::cout<<"float projection: adaptive q1..8, group32/48/64, tails, signed q8, GR shapes and GR composition PASS\n";
    }
    mfq_cuda_synchronize();
    return 0;
} catch (const std::exception& error) {
    std::cerr<<"float projection: "<<error.what()<<'\n';
    return 1;
}
