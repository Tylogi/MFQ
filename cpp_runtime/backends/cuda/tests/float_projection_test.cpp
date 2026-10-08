#include "float_projection.h"
#include "gated_residual.h"
#include "gated_residual_fused.h"
#include "cuda_execution.h"
#include "storage/weight_loader.h"
#include "selected_attention.h"
#include "qwen4_exp.h"
#include "runtime/decode_window.h"

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
Fixture make_weight(int out, int width, int group_size, bool zero = false,bool aligned_q8=false) {
    Fixture fixture;
    auto& w = fixture.weight;
    w.out=out; w.neuron_len=width; w.gs=group_size; w.ng=(width+group_size-1)/group_size;
    w.q8_zero=zero;
    std::vector<uint8_t> bits(out), scales(out*w.ng), minima(out*w.ng);
    std::vector<int64_t> offsets(out);
    std::vector<float> row_scale(out), row_min(out), zero_scale(out*w.ng);
    int64_t total_bits=0;
    for (int row=0; row<out; ++row) {
        bits[row]=aligned_q8?8:1+row%8;
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
    w.aligned_q8=aligned_q8 && !zero && group_size%4==0;
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

void exact(const tb::Tensor& actual,const tb::Tensor& expected,const std::string& name) {
    if(actual.sizes()!=expected.sizes() || actual.scalar_type()!=expected.scalar_type())
        throw std::runtime_error(name+" dtype/shape differs");
    auto a=actual.to(tb::kFloat32).cpu().contiguous(),b=expected.to(tb::kFloat32).cpu().contiguous();
    for(int64_t i=0;i<a.numel();++i)if(!std::isfinite(a.data_ptr<float>()[i]) ||
        std::memcmp(a.data_ptr<float>()+i,b.data_ptr<float>()+i,sizeof(float)))
        throw std::runtime_error(name+" differs in FP32 bits at "+std::to_string(i)+" actual="+
            std::to_string(a.data_ptr<float>()[i])+" expected="+std::to_string(b.data_ptr<float>()[i]));
}

void projection_mix(bool aligned_q8,bool zero=false,int hidden=2560,int group_size=48,int rank=320,int streams=4) {
    auto fixture=make_weight(hidden*streams,rank,group_size,zero,aligned_q8);
    std::vector<float> source(rank*6),values(hidden*streams*6);
    for(size_t i=0;i<source.size();++i)source[i]=std::sin(float(i)*.031f)*.13f;
    for(size_t i=0;i<values.size();++i)values[i]=std::sin(float(i)*.173f)*.7f;
    for(auto gate_dtype:{tb::kFloat16,tb::kFloat32})for(auto value_dtype:{tb::kFloat16,tb::kFloat32}) {
        auto input=upload(source,tb::kFloat32,{2,3,rank}).to(gate_dtype);
        auto normalized=upload(values,tb::kFloat32,{2,3,hidden*streams}).to(value_dtype);
        const auto compact=[&](const tb::Tensor& x,const tb::Tensor& n,bool native=true,bool fixed=true) {
            return nint_float_projection_mix_cuda(fixture.weight,x,n,streams,native,fixed,true);
        };
        auto reference=[&](const tb::Tensor& x,const tb::Tensor& n) {
            auto projected=nint_float_projection_cuda(fixture.weight,x,true,false);
            if(x.scalar_type()==tb::kFloat16)projected=projected.to(tb::kFloat16);
            return mfq_qwen4_exp::gated_residual_mix(projected,n,streams);
        };
        exact(compact(input,normalized),
            reference(input,normalized),"packed Up/mix prefill");
        exact(compact(input,normalized),
            nint_float_projection_mix_cuda(fixture.weight,input,normalized,streams,true,true,false),"compact Up/mix vs original CTA");
        exact(compact(input,normalized),compact(input,normalized,false),"native-input Up/mix");
        exact(compact(input,normalized),compact(input,normalized,true,false),"fixed-group Up/mix");
        exact(compact(input.transpose(0,1),normalized.transpose(0,1)),
            reference(input.transpose(0,1),normalized.transpose(0,1)),"strided packed Up/mix");
        auto x=input.narrow(0,0,1).narrow(1,0,1).clone();
        auto n=normalized.narrow(0,0,1).narrow(1,0,1).clone();
        tb::Tensor result;
        mfq::cuda::DecodeWindow window(mfq_current_cuda_stream());
        window.capture([&]{result=compact(x,n);},[]{},[&]{result={};});
        for(int token=0;token<3;++token) {
            x.copy_(input.narrow(0,0,1).narrow(1,token,1));
            n.copy_(normalized.narrow(0,0,1).narrow(1,token,1));
            window.run();exact(result,reference(x,n),"packed Up/mix changing-input graph");
            exact(result,nint_float_projection_mix_cuda(fixture.weight,x,n,streams,true,true,false),"compact Up/mix original graph result");
        }
    }
}

void native_projection_inputs() {
    int cases=0;
    for(int mode:{0,1,2})for(int group:{32,48,64})
        for(auto shape:{std::pair<int,int>{320,10240},{10240,320},{4,10240}}) {
            auto fixture=make_weight(shape.first,shape.second,group,mode==2,mode==1);
            for(int rows:{1,3,35})for(auto dtype:{tb::kFloat16,tb::kFloat32}) {
                std::vector<float> values(rows*shape.second);
                for(size_t i=0;i<values.size();++i)values[i]=std::sin(float(i)*.031f)*.13f;
                auto x=upload(values,tb::kFloat32,{rows,shape.second}).to(dtype);
                const auto check=[&](const tb::Tensor& actual,const tb::Tensor& input) {
                    auto expected=nint_float_projection_cuda(fixture.weight,input,true,false);
                    exact(actual,expected,"native-input projection");
                    exact(actual,nint_float_projection_cuda(fixture.weight,input,true,true,false),"original runtime-group projection");
                    auto a=actual.cpu().contiguous(),b=expected.cpu().contiguous();
                    if(std::memcmp(a.data_ptr(),b.data_ptr(),a.numel()*sizeof(float)))
                        throw std::runtime_error("native-input projection bytes differ");
                };
                check(nint_float_projection_cuda(fixture.weight,x),x);
                auto source=x.narrow(0,0,1).clone();tb::Tensor output;
                mfq::cuda::DecodeWindow window(mfq_current_cuda_stream());
                window.capture([&]{output=nint_float_projection_cuda(fixture.weight,source);},[]{},[&]{output={};});
                for(int token=0;token<3;++token) {
                    source.copy_(x.narrow(0,token%rows,1)*(1.0+token*.25));
                    window.run();check(output,source);
                }
                ++cases;
            }
        }
    std::cout<<"native_projection_input_cases="<<cases<<" original FP32 conversion/reduction bytes exact, real GR shapes, changing-input graphs PASS\n";
}

void projection_activations() {
    int cases=0;
    for(auto shape:{std::pair<int,int>{320,10240},{4,10240},{17,77}})for(int profile:{0,1,2}) {
        const int group=profile==0?32:profile==1?48:64;
        auto fixture=make_weight(shape.first,shape.second,group,profile==2,profile==1);
        for(int rows:{1,3})for(auto dtype:{tb::kFloat16,tb::kFloat32})
            for(bool injection:{false,true})for(bool parallel:{false,true}) {
                std::vector<float> values(rows*shape.second);
                for(size_t i=0;i<values.size();++i)values[i]=std::sin(float(i)*.037f)*.31f;
                auto x=upload(values,tb::kFloat32,{rows,shape.second}).to(dtype);
                const auto reference=[&](bool native) {
                    auto projected=nint_float_projection_cuda(fixture.weight,x,parallel,native);
                    if(dtype==tb::kFloat16)projected=projected.to(tb::kFloat16);
                    return injection?mfq_qwen4_exp::gated_residual_injection(projected,4):
                        mfq_qwen4_exp::gated_residual_bottleneck(projected,4);
                };
                for(bool native:{false,true})exact(
                    nint_float_projection_activation_cuda(fixture.weight,x,4,injection,parallel,native),
                    reference(native),"packed projection activation original split/cast/gate");
                tb::Tensor output;mfq::cuda::DecodeWindow window(mfq_current_cuda_stream());
                window.capture([&]{output=nint_float_projection_activation_cuda(fixture.weight,x,4,injection,parallel);},
                    []{},[&]{output={};});
                for(int step=0;step<3;++step) {
                    x.copy_(upload(values,tb::kFloat32,{rows,shape.second}).to(dtype)*(1.0+step*.25));
                    window.run();exact(output,reference(true),"packed projection activation changing-input graph");
                }
                ++cases;
            }
    }
    std::cout<<"projection_activation_cases="<<cases<<" original split order, Half/Float cast/division/gate bits, variable/aligned/signed q8, real Down/inject and tails, changing-input graphs exact PASS\n";
}

void fixed_group_projections() {
    int cases=0;
    for(int group:{32,48,64}) {
        const int width=group*160;
        auto fixture=make_weight(320,width,group,false,true);
        for(auto dtype:{tb::kFloat16,tb::kFloat32}) {
            std::vector<float> values(width);
            for(int i=0;i<width;++i)values[i]=std::sin(float(i)*.037f)*.31f;
            auto x=upload(values,tb::kFloat32,{1,width}).to(dtype);
            const auto reference=[&] {return nint_float_projection_cuda(fixture.weight,x,true,true,false);};
            exact(nint_float_projection_cuda(fixture.weight,x),reference(),"complete-group original projection");
            tb::Tensor output;mfq::cuda::DecodeWindow window(mfq_current_cuda_stream());
            window.capture([&]{output=nint_float_projection_cuda(fixture.weight,x);},[]{},[&]{output={};});
            for(int step=0;step<3;++step) {
                x.copy_(upload(values,tb::kFloat32,{1,width}).to(dtype)*(1.0+step*.25));
                window.run();exact(output,reference(),"complete-group changing-input projection graph");
            }
            ++cases;
        }
    }
    std::cout<<"fixed_group_projection_cases="<<cases<<" groups32/48/64 Half/Float original runtime-group bytes exact PASS\n";
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
    compare(nint_float_projection_cuda(fixture.weight,x,false),expected,"original unsplit projection");
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
    int out,width,batch,cases=0,activation_cases=0,dense_cases=0,compact_cases=0;
    std::string name;
    while (manifest>>name>>out>>width>>batch) {
        auto source=mfq::open_model_source((root/(name+".mfq")).string());
        if (require_tensor(*source,"linear.weight").dtype!="NINT") {
            execution.config.gr_prepared_dense_projection=false;
            auto unprepared=mfq::cuda::weight_loader::residual_linear(execution,*source,"linear.weight");
            execution.config.gr_prepared_dense_projection=true;
            auto projection=mfq::cuda::weight_loader::residual_linear(execution,*source,"linear.weight");
            if(projection.mixed || projection.activated)throw std::runtime_error("dense residual acquired packed callbacks");
            auto dense=mfq::cuda::weight_loader::dense(execution,*source,"linear.weight");
            auto right=dense.transpose(-1,-2);
            auto x=upload(read(root/(name+".input.f32"),batch*width),tb::kFloat32,{batch,width});
            for(auto dtype:{tb::kFloat16,tb::kBFloat16,tb::kFloat32})for(int rows:{1,batch}) {
                auto input=x.to(dtype).narrow(0,0,rows).clone();
                const auto reference=[&](const tb::Tensor& value) {
                    return mfq_selected_attention::promoted_matmul(value,right);
                };
                execution.config.gr_prepared_dense_projection=false;
                exact(projection(execution,input),reference(input),name+" original dense projection");
                execution.config.gr_prepared_dense_projection=true;
                exact(projection(execution,input),reference(input),name+" prepared dense projection");
                exact(unprepared(execution,input),reference(input),name+" disabled loading remains original after flag change");
                auto strided=tb::cat({input,input},0).reshape({2,rows,width}).transpose(0,1);
                exact(projection(execution,strided),reference(strided),name+" prepared dense strided input");
                tb::Tensor output;mfq::cuda::DecodeWindow window(mfq_current_cuda_stream());
                window.capture([&]{output=projection(execution,input);},[]{},[&]{output={};});
                for(int step=0;step<3;++step) {
                    input.copy_(x.to(dtype).narrow(0,0,rows)*(1.0+step*.25));window.run();
                    exact(output,reference(input),name+" prepared dense changing-input graph");
                }
                ++dense_cases;
            }
            ++cases;
            continue;
        }
        auto projection=mfq::cuda::weight_loader::residual_linear(execution,*source,"linear.weight");
        auto x=upload(read(root/(name+".input.f32"),batch*width),tb::kFloat32,{batch,width});
        compare(projection(execution,x),read(root/(name+".expected.f32"),batch*out),name);
        for(auto dtype:{tb::kFloat16,tb::kFloat32}) {
            auto input=x.to(dtype);
            execution.config.gr_native_projection_input=false;auto reference=projection(execution,input);
            execution.config.gr_native_projection_input=true;
            exact(projection(execution,input),reference,name+" native input vs original projection");
        }
        if(out>width && out%4==0) {
            if(!projection.mixed)throw std::runtime_error("packed residual has no mixed projection");
            std::vector<float> values(batch*out);
            for(size_t i=0;i<values.size();++i)values[i]=std::sin(float(i)*.173f)*.7f;
            const auto normalized=upload(values,tb::kFloat32,{batch,out});
            for(auto gate_dtype:{tb::kFloat16,tb::kFloat32})for(auto value_dtype:{tb::kFloat16,tb::kFloat32})
                for(int rows:{1,batch}) {
                    auto input=x.to(gate_dtype).narrow(0,0,rows).clone();
                    auto norm=normalized.to(value_dtype).narrow(0,0,rows).clone();
                    const auto check=[&](const tb::Tensor& compact) {
                        execution.config.gr_compact_mix=false;
                        exact(compact,projection.mixed(execution,input,norm,4),name+" compact Up/mix original CTA");
                        exact(compact,mfq_qwen4_exp::gated_residual_mix(projection(execution,input),norm,4),name+" compact Up/mix composed oracle");
                        execution.config.gr_compact_mix=true;
                    };
                    execution.config.gr_compact_mix=true;
                    check(projection.mixed(execution,input,norm,4));
                    tb::Tensor output;mfq::cuda::DecodeWindow window(mfq_current_cuda_stream());
                    window.capture([&]{output=projection.mixed(execution,input,norm,4);},[]{},[&]{output={};});
                    for(int step=0;step<3;++step) {
                        input.copy_(x.to(gate_dtype).narrow(0,0,rows)*(1.0+step*.25));
                        norm.copy_(normalized.to(value_dtype).narrow(0,0,rows)*(1.0+step*.125));
                        window.run();check(output);
                    }
                    ++compact_cases;
                }
            execution.config.gr_compact_mix=true;
        }
        if(out<width)for(auto dtype:{tb::kFloat16,tb::kFloat32})for(bool injection:{false,true}) {
            if(!projection.activated)throw std::runtime_error("packed residual has no activation projection");
            auto input=x.to(dtype);
            const auto reference=[&] {
                auto value=projection(execution,input);
                return injection?mfq_qwen4_exp::gated_residual_injection(value,4):
                    mfq_qwen4_exp::gated_residual_bottleneck(value,4);
            };
            exact(projection.activated(execution,input,4,injection),reference(),name+" original projection activation");
            tb::Tensor output;mfq::cuda::DecodeWindow window(mfq_current_cuda_stream());
            window.capture([&]{output=projection.activated(execution,input,4,injection);},[]{},[&]{output={};});
            for(int step=0;step<3;++step) {
                input.copy_(x.to(dtype)*(1.0+step*.25));window.run();
                exact(output,reference(),name+" actual projection activation graph");
            }
            ++activation_cases;
        }
        std::cout<<"float_projection "<<name<<" PASS\n";
        ++cases;
    }
    if (!manifest.eof() || !cases) throw std::runtime_error("invalid canonical projection fixtures");
    std::cout<<"canonical_float_cases="<<cases<<'\n';
    std::cout<<"canonical_compact_mix_cases="<<compact_cases<<" actual released Up matrices, mixed Half/Float, t1/3, original CTA/composed bits and changing graphs exact PASS\n";
    std::cout<<"canonical_projection_activation_cases="<<activation_cases<<" actual matrix original projection/cast/gate bits exact PASS\n";
    std::cout<<"canonical_prepared_dense_cases="<<dense_cases<<" original promoted transpose/matmul bits, Half/BFloat/Float input, t1/3, strided inputs and changing graphs exact PASS\n";
}

void residual() {
    constexpr int hidden=16, streams=4, width=hidden*streams;
    auto down=make_weight(8,width,32), up=make_weight(width,8,32), inject=make_weight(streams,width,32);
    auto dense=[](const Fixture& f) { return upload(f.dense,tb::kFloat32,{f.weight.out,f.weight.neuron_len}); };
    std::vector<float> input(3*width), norm(width);
    for (size_t i=0; i<input.size(); ++i) input[i]=std::sin(float(i)*0.031f);
    for(size_t i=0;i<norm.size();++i)norm[i]=1.0f+std::sin(float(i)*.017f)*.13f;
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

void residual_post_norm() {
    int cases=0,graph_cases=0;
    for(int hidden:{31,32,33,64,65,128,129,2560})
    for(auto branch_dtype:{tb::kFloat16,tb::kFloat32})
    for(auto residual_dtype:{tb::kFloat16,tb::kFloat32})
    for(auto injection_dtype:{tb::kFloat16,tb::kFloat32})
    for(auto norm_dtype:{tb::kFloat16,tb::kFloat32}) {
        const int streams=hidden%2?3:4,tokens=hidden%2?3:1,width=hidden*streams;
        std::vector<float> branches(2*tokens*hidden),residuals(2*tokens*width),injections(2*tokens*streams),weights(width);
        for(size_t i=0;i<branches.size();++i)branches[i]=i%19?std::sin(float(i)*.173f)*1.31f:-0.f;
        for(size_t i=0;i<residuals.size();++i)residuals[i]=i%17?std::sin(float(i)*.037f)*.29f:0.f;
        for(size_t i=0;i<injections.size();++i)injections[i]=i%5?std::sin(float(i)*.73f)*1.37f:-0.f;
        for(size_t i=0;i<weights.size();++i)weights[i]=std::sin(float(i)*.13f)*.41f;
        auto branch=upload(branches,tb::kFloat32,{2,tokens,hidden}).to(branch_dtype);
        auto residual=upload(residuals,tb::kFloat32,{2,tokens,width}).to(residual_dtype);
        auto injection=upload(injections,tb::kFloat32,{2,tokens,streams}).to(injection_dtype);
        auto norm=upload(weights,tb::kFloat32,{width}).to(norm_dtype);
        const auto check=[&](const std::vector<tb::Tensor>& actual) {
            auto updated=mfq_qwen4_exp::gated_residual_post(branch,residual,injection,streams);
            auto normalized=mfq_qwen4_exp::grouped_rms_norm(updated,norm,hidden,1e-6);
            exact(actual[0],updated,"chained residual dtype promotion and rounding");
            exact(actual[1],normalized,"chained residual/grouped RMSNorm");
        };
        check(mfq_qwen4_exp::gated_residual_post_norm(branch,residual,injection,norm,streams,1e-6));++cases;
        std::vector<tb::Tensor> output;
        mfq::cuda::DecodeWindow window(mfq_current_cuda_stream());
        window.capture([&]{output=mfq_qwen4_exp::gated_residual_post_norm(branch,residual,injection,norm,streams,1e-6);},
            []{},[&]{output.clear();});
        for(float amplitude:{0.f,.13f,8.f}) {
            branch.copy_(upload(branches,tb::kFloat32,{2,tokens,hidden}).to(branch_dtype)*amplitude);
            window.run();check(output);++graph_cases;
        }
    }
    std::cout<<"gr_post_norm_cases="<<cases<<" graph_cases="<<graph_cases
        <<" mixed Half/Float, norm dtypes, widths31/32/33/64/65/128/129/2560, streams3/4, batch2, signed zero exact PASS\n";
}

void residual_model() {
    constexpr int hidden=2560,streams=4,width=hidden*streams,rank=320;
    auto down=make_weight(rank,width,48,false,true),up=make_weight(width,rank,48,false,true),
        inject=make_weight(streams,width,48,false,true);
    const auto projection=[](const Fixture& f)->GatedResidualProjection {
        return [w=f.weight](CudaExecutionContext& execution,const tb::Tensor& x) {
            auto output=nint_float_projection_cuda(w,x,true,execution.config.gr_native_projection_input);
            return x.scalar_type()==tb::kFloat16?output.to(tb::kFloat16):output;
        };
    };
    GatedResidualMixProjection mixed=[w=up.weight](CudaExecutionContext& execution,const tb::Tensor& x,
        const tb::Tensor& normalized,int64_t count) {
        return nint_float_projection_mix_cuda(w,x,normalized,count,execution.config.gr_native_projection_input,
            execution.config.gr_fixed_group_projection,execution.config.gr_compact_mix);
    };
    const auto activated=[](const Fixture& f)->GatedResidualActivationProjection {
        return [w=f.weight](CudaExecutionContext& execution,const tb::Tensor& x,int64_t count,bool injection) {
            return nint_float_projection_activation_cuda(w,x,count,injection,true,execution.config.gr_native_projection_input);
        };
    };
    std::vector<float> values(3*width),norm(width);
    for(size_t i=0;i<values.size();++i)values[i]=std::sin(float(i)*.031f)*.4f;
    for(size_t i=0;i<norm.size();++i)norm[i]=1.0f+std::sin(float(i)*.017f)*.13f;
    auto input=upload(values,tb::kFloat32,{1,3,width}),n=upload(norm,tb::kFloat32,{width});
    CudaExecutionContext execution;
    const auto forward=[&](const tb::Tensor& x) {
        return gated_residual_pre_projected(execution,x,n,projection(down),projection(up),
            projection(inject),hidden,streams,1e-6,mixed,{},activated(down),activated(inject));
    };
    int cases=0;const auto norm_source=n;
    for(auto norm_dtype:{tb::kFloat16,tb::kFloat32})for(auto dtype:{tb::kFloat16,tb::kFloat32}) {
        n=norm_source.to(norm_dtype);
        auto x=input.to(dtype);
        execution.config.gr_fused_projections=false;execution.config.gr_native_projection_input=false;auto reference=forward(x);
        execution.config.gr_fused_projections=true;execution.config.gr_native_projection_input=true;
        execution.config.gr_compact_mix=true;auto actual=forward(x);
        for(int i=0;i<3;++i)exact(actual[i],reference[i],"four-stream packed GR composition");
        execution.config.gr_fused_projection_activation=false;auto separate=forward(x);
        execution.config.gr_fused_projection_activation=true;
        for(int i=0;i<3;++i)exact(actual[i],separate[i],"four-stream packed GR projection activation");
        auto source=x.narrow(1,0,1).clone();std::vector<tb::Tensor> result;
        mfq::cuda::DecodeWindow window(mfq_current_cuda_stream());
        window.capture([&]{result=forward(source);},[]{},[&]{result.clear();});
        for(int token=0;token<3;++token) {
            source.copy_(x.narrow(1,token,1));window.run();
            execution.config.gr_fused_projections=false;execution.config.gr_native_projection_input=false;reference=forward(source);
            execution.config.gr_fused_projections=true;execution.config.gr_native_projection_input=true;
            for(int i=0;i<3;++i)exact(result[i],reference[i],"four-stream packed GR graph composition");
        }
        auto branch=x.narrow(1,0,1).narrow(-1,0,hidden).contiguous().to(tb::kFloat16);
        auto prior_injection=upload(std::vector<float>{.37f,-.83f,1.03f,-0.f},tb::kFloat32,{1,1,streams});
        const auto after=[&] {
            return gated_residual_pre_after_projected(execution,branch,source,prior_injection,n,
                projection(down),projection(up),projection(inject),hidden,streams,1e-6,
                mixed,{},activated(down),activated(inject));
        };
        execution.config.gr_fused_post_norm=false;auto separate_after=after();
        execution.config.gr_fused_post_norm=true;auto fused_after=after();
        for(int i=0;i<3;++i)exact(fused_after[i],separate_after[i],"chained full GR composition");
        std::vector<tb::Tensor> chained;
        mfq::cuda::DecodeWindow chain_window(mfq_current_cuda_stream());
        chain_window.capture([&]{chained=after();},[]{},[&]{chained.clear();});
        for(int token=0;token<3;++token) {
            source.copy_(x.narrow(1,token,1));
            branch.copy_(source.narrow(-1,0,hidden).contiguous().to(tb::kFloat16));
            chain_window.run();execution.config.gr_fused_post_norm=false;separate_after=after();
            execution.config.gr_fused_post_norm=true;
            for(int i=0;i<3;++i)exact(chained[i],separate_after[i],"chained full GR changing graph");
        }
        ++cases;
    }
    std::cout<<"gr_nonzero_composition_cases="<<cases<<" four-stream2560/rank320 mixed Half/Float norm/input, ordinary and changing-input graphs exact PASS\n";
    std::cout<<"gr_projection_activation_cases="<<cases<<" original separate projection activation, nonzero norm, full GR graph bits exact PASS\n";
    std::cout<<"gr_chained_model_cases="<<cases<<" actual2560/rank320, original post/pre composition and changing graph exact PASS\n";
}
}

namespace {
void complete_two_stage_residual() {
    int ordinary=0,graphs=0;
    for(int group:{5,24,28,32,48,64})for(int profile:{0,1,2})for(int streams:{1,3,4}) {
        const int hidden=4096/streams,width=hidden*streams,rank=17;
        auto down=make_weight(rank,width,group,profile==2,profile==1);
        auto up=make_weight(width,rank,group,profile==2,profile==1);
        auto inject=make_weight(streams,width,group==5?24:5,profile==2,profile==1);
        std::vector<float> data(width),gamma(width),gate(streams),branch_data(hidden);
        for(int j=0;j<width;++j) {data[j]=std::sin(j*.047f)*.73f;gamma[j]=std::cos(j*.023f)*.19f;}
        for(int j=0;j<hidden;++j)branch_data[j]=std::sin(j*.017f)*.21f;
        for(int j=0;j<streams;++j)gate[j]=.13f+j*.19f;
        auto norm=upload(gamma,tb::kFloat32,{width});
        CudaExecutionContext execution;
        const auto projection=[](const NintWeight& w)->GatedResidualProjection {
            return [&w](CudaExecutionContext&,const tb::Tensor& x) {
                auto y=nint_float_projection_cuda(w,x);
                return x.scalar_type()==tb::kFloat16?y.to(tb::kFloat16):y;
            };
        };
        const auto activated=[](const NintWeight& w)->GatedResidualActivationProjection {
            return [&w](CudaExecutionContext&,const tb::Tensor& x,int64_t streams,bool injection) {
                return nint_float_projection_activation_cuda(w,x,streams,injection);
            };
        };
        GatedResidualMixProjection mixed=[&](CudaExecutionContext&,const tb::Tensor& x,const tb::Tensor& n,int64_t s) {
            return nint_float_projection_mix_cuda(up.weight,x,n,s,true,true,true);
        };
        for(int kind:{0,6})for(bool after:{false,true}) {
            auto residual=upload(data,tb::kFloat32,{1,1,width}).to(kind&2?tb::kFloat32:tb::kFloat16);
            auto branch=upload(branch_data,tb::kFloat32,{1,1,hidden}).to(tb::kFloat16);
            auto prior=upload(gate,tb::kFloat32,{1,1,streams}).to(kind&4?tb::kFloat32:tb::kFloat16);
            const auto reference=[&] {
                if(after)return gated_residual_pre_after_projected(execution,branch,residual,prior,norm,
                    projection(down.weight),projection(up.weight),projection(inject.weight),hidden,streams,1e-6,
                    mixed,{},activated(down.weight),activated(inject.weight));
                return gated_residual_pre_projected(execution,residual,norm,projection(down.weight),projection(up.weight),
                    projection(inject.weight),hidden,streams,1e-6,mixed,{},activated(down.weight),activated(inject.weight));
            };
            const auto candidate=[&] {return gated_residual_two_stage_cuda(branch,residual,prior,norm,
                down.weight,up.weight,&inject.weight,streams,1e-6,after);};
            auto expected=reference(),result=candidate();
            for(int j=0;j<3;++j)exact(result[j],expected[j],"complete two-stage residual ordinary");
            ++ordinary;
            mfq::cuda::DecodeWindow window(mfq_current_cuda_stream());
            window.capture([&]{result=candidate();},[]{},[&]{result.clear();});
            for(int step=0;step<3;++step) {
                residual.copy_(upload(data,tb::kFloat32,{1,1,width}).to(residual.scalar_type())*(step*.5));
                branch.copy_(upload(branch_data,tb::kFloat32,{1,1,hidden}).to(tb::kFloat16)*(1+step*.5));
                window.run();expected=reference();
                for(int j=0;j<3;++j)exact(result[j],expected[j],"complete two-stage residual changing graph");
                ++graphs;
            }
        }
    }
    std::cout<<"complete_two_stage_residual ordinary="<<ordinary<<" changing_graph="<<graphs
        <<" exact; runtime groups5/24/28/32/48/64, variable/aligned/signed q8, differing injection group, streams1/3/4, tails and pre/after PASS\n";
}
}

int main(int argc,char** argv) try {
    if (argc>2) throw std::runtime_error("expected optional canonical fixture directory");
    auto context=mfq::cuda::default_context(mfq_current_cuda_device());
    const auto stream=mfq_current_cuda_stream();context->begin_graph_pool(stream);
    if (argc==2) canonical(argv[1]);
    else {
        for (int group : {32,48,64}) for (int batch : {1,3,11}) synthetic(17,77,group,batch,false);
        for (int batch : {1,3,11}) synthetic(17,77,32,batch,true);
        for (auto shape : {std::pair<int,int>{320,10240},{10240,320},{4,10240}})
            for(int batch:{1,2})synthetic(shape.first,shape.second,64,batch,false);
        residual();
        residual_model();
        residual_post_norm();
        complete_two_stage_residual();
        native_projection_inputs();
        fixed_group_projections();
        projection_activations();
        projection_mix(false);projection_mix(true);projection_mix(false,true);projection_mix(false,false,17);
        for(int group:{32,48,64})projection_mix(true,false,2560,group,group==48?336:320);
        projection_mix(true,false,2561,48,320,3);
        std::cout<<"compact_mix_tail_cases=4 three-stream odd hidden2561, mixed Half/Float, original CTA and changing graphs exact PASS\n";
        std::cout<<"fixed_group_mix_cases=12 groups32/48/64 four-stream2560 mixed Half/Float input/value, ordinary and changing-input graphs exact PASS\n";
        std::cout<<"float projection: adaptive q1..8, group32/48/64, tails, signed q8, GR shapes and GR composition PASS\n";
        std::cout<<"GR packed Up/mix: mixed dtypes, aligned q8, signed q8, prefill, split fallback and changing-input graphs exact PASS\n";
    }
    mfq_cuda_synchronize();
    context->end_graph_pool(stream);
    return 0;
} catch (const std::exception& error) {
    std::cerr<<"float projection: "<<error.what()<<'\n';
    return 1;
}
