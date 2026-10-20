#include "timing.h"
#include "cuda_execution.h"
#include "storage/weight_loader.h"
#include "mfq/model_source.h"
#include "mfq/kernels/cuda/qwen4_exp.h"
#include "mfq/kernels/cuda/ple_fused.h"
#include "mfq_cuda_linear_attention_ops.h"
#include "mfq_cuda_norm_ops.h"
#include "mfq_cuda_moe_ops.h"
#include "mfq_cuda_quant_ops.h"
#include "float_projection.h"
#include "selected_attention.h"
#include <cstring>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <set>

namespace tb=mfq_tensor_backend;
namespace fb=format_bench;
namespace weight_loader=mfq::cuda::weight_loader;

tb::Tensor nint_matmul_fixed_bits_ws_cuda(tb::Tensor,tb::Tensor,tb::Tensor,tb::Tensor,tb::Tensor,
    tb::Tensor,tb::Tensor,tb::Tensor,int64_t,tb::Tensor,tb::Tensor,int);

static void option(const char* name,const char* value) {
    if(std::strcmp(name,"MFQ_NINT_WARP_ROWS_AUTO")==0) {
        option("MFQ_NINT_WARP_ROWS",value[0]=='0'?"":"1");return;
    }
    if(std::strcmp(name,"MFQ_NINT_BEST")==0) {
        for(const char* key:{"MFQ_NINT_ACTIVATION_SUM","MFQ_NINT_FIXED_BITS","MFQ_NINT_INTERLEAVED","MFQ_NINT_FIXED_OUTPUT","MFQ_NINT_OUTPUT_TWO_WARPS","MFQ_NINT_NARROW_OFFSETS","MFQ_NINT_FIXED_EXACT_ROWS"})
            option(key,value[0]=='0'?"0":"");
        return;
    }
#ifdef _WIN32
    _putenv_s(name,value);
#else
    setenv(name,value,1);
#endif
}
static void compare(const tb::Tensor& actual,const tb::Tensor& expected,bool exact) {
    auto a=actual.contiguous().cpu(),b=expected.contiguous().cpu();
    if(a.sizes()!=b.sizes() || a.scalar_type()!=b.scalar_type())throw std::runtime_error("projection geometry changed");
    if(exact) {
        if(std::memcmp(a.data_ptr(),b.data_ptr(),a.nbytes()))throw std::runtime_error("projection Half bits changed");
        return;
    }
    a=a.to(tb::kFloat32);b=b.to(tb::kFloat32);
    const auto* av=a.data_ptr<float>();const auto* bv=b.data_ptr<float>();
    double error=0,signal=0,maximum=0;
    for(int64_t i=0;i<a.numel();++i) {
        const double diff=std::abs(double(av[i])-bv[i]);
        if(!std::isfinite(av[i]) || !std::isfinite(bv[i]) || diff>0.002+0.002*std::abs(double(bv[i])))
            throw std::runtime_error("grouped projection exceeds FP16 reduction tolerance");
        error+=diff*diff;signal+=double(bv[i])*bv[i];maximum=std::max(maximum,diff);
    }
    const double nrmse=std::sqrt(error/std::max(signal,1e-20));
    if(nrmse>0.0003)throw std::runtime_error("grouped projection relative error exceeds reduction tolerance");
    std::cout<<"NUMERICAL nrmse="<<nrmse<<" max_absolute="<<maximum<<'\n';
}
static void joined_qkv(CudaExecutionContext& execution,const mfq::ModelSource& model) {
    const std::string root="model.block.0.linear_attention.";
    if(!has_tensor(model,root+"query.weight"))return;
    std::vector<weight_loader::Linear> separate;
    for(const char* role:{"query","key","value","gate","alpha","beta"})
        separate.push_back(weight_loader::linear(execution,model,root+role+".weight"));
    auto grouped=weight_loader::grouped_linear(execution,separate,3);
    if(!grouped)throw std::runtime_error("real GDN grouped projection unavailable");
    const int64_t width=weight_loader::linear_weight(separate.front())->neuron_len();
    int checks=0;
    for(int rows:{1,2,3,4,5,6,8,16}) {
        auto x=tb::tensor(fb::input(rows,int(width))).reshape({1,rows,width}).to(tb::kCUDA,tb::kFloat16);
        const auto reference=[&] {
            std::vector<tb::Tensor> p;for(auto& projection:separate)p.push_back(projection(execution,x));
            return std::vector<tb::Tensor>{tb::cat({p[0],p[1],p[2]},-1),p[3],p[4],p[5]};
        };
        MfqCudaGraph graph;mfq_prepare_cuda_graph_memory(graph);
        auto output=grouped(execution,x);fb::check(cudaStreamSynchronize(mfq_current_cuda_stream()));output.clear();
        graph.capture_begin();output=grouped(execution,x);graph.capture_end();
        for(int step=0;step<3;++step) {
            auto input=fb::input(rows,int(width));for(auto& v:input)v*=float(step*step);
            x.copy_(tb::tensor(input).reshape(x.sizes()).to(tb::kCUDA,tb::kFloat16));
            auto expected=reference();graph.replay();
            if(output.size()!=expected.size())throw std::runtime_error("joined QKV output count changed");
            for(size_t i=0;i<output.size();++i){compare(output[i],expected[i],true);++checks;}
        }
    }
    std::cout<<"VERIFY_PACKED_QKV checks="<<checks<<" PASS\n"<<std::flush;
}

static int gdn_projection(CudaExecutionContext& execution,const mfq::ModelSource& model,const char* sweep_option="MFQ_NINT_GROUP_LOAD") {
    const bool float_input=std::getenv("MFQ_PROBE_FLOAT_INPUT")!=nullptr;
    const auto input_dtype=float_input?tb::kFloat32:tb::kFloat16;
    struct Bank {weight_loader::LinearGroup input;weight_loader::Linear output;tb::Tensor x,y;};
    std::vector<Bank> banks;
    for(int layer:{0,1,8,9,16,17,24,25}) {
        const auto root="model.block."+std::to_string(layer)+".linear_attention.";
        std::vector<weight_loader::Linear> projections;
        if(has_tensor(model,root+"query.weight")) {
            for(const char* role:{"query","key","value"})
                projections.push_back(weight_loader::linear(execution,model,root+role+".weight"));
        } else projections.push_back(weight_loader::linear(execution,model,root+"qkv.weight"));
        const size_t join=projections.size();
        const auto width=weight_loader::linear_weight(projections.front())->neuron_len();
        for(const char* role:{"gate","alpha","beta"})
            projections.push_back(weight_loader::linear(execution,model,root+role+".weight"));
        auto output=weight_loader::linear(execution,model,root+"output.weight");
        const auto output_width=weight_loader::linear_weight(output)->neuron_len();
        banks.push_back({weight_loader::grouped_linear(execution,projections,join),output,
            tb::empty({1,6,width},tb::TensorOptions().dtype(input_dtype).device(tb::kCUDA)),
            tb::empty({1,6,output_width},tb::TensorOptions().dtype(input_dtype).device(tb::kCUDA))});
        if(!banks.back().input)throw std::runtime_error("GDN grouped probe unavailable");
    }
    int checks=0;
    for(int rows:{1,2,3,4,5,6})for(int stage:{0,1})for(const char* variant:{"0","1","0"}) {
        option(sweep_option,variant);
        const auto run=[&](size_t bank) {
            auto& b=banks[bank];
            if(stage==0)return b.input(execution,b.x.narrow(1,0,rows));
            return std::vector<tb::Tensor>{b.output(execution,b.y.narrow(1,0,rows))};
        };
        MfqCudaGraph graph;mfq_prepare_cuda_graph_memory(graph);
        std::vector<std::vector<tb::Tensor>> outputs;
        for(size_t i=0;i<banks.size();++i)outputs.push_back(run(i));
        fb::check(cudaStreamSynchronize(mfq_current_cuda_stream()));outputs.clear();
        graph.capture_begin();for(size_t i=0;i<banks.size();++i)outputs.push_back(run(i));graph.capture_end();
        for(int step=0;step<3;++step) {
            for(auto& b:banks)for(auto* value:{&b.x,&b.y}) {
                auto input=fb::input(6,int(value->size(-1)));for(auto& v:input)v*=float(step*step);
                value->copy_(tb::tensor(input).reshape(value->sizes()).to(tb::kCUDA,input_dtype));
            }
            option(sweep_option,"0");
            std::vector<std::vector<tb::Tensor>> expected;
            for(size_t i=0;i<banks.size();++i) {
                auto p=run(i);for(auto& y:p)y=y.contiguous().cpu();expected.push_back(std::move(p));
            }
            graph.replay();
            for(size_t i=0;i<banks.size();++i)for(size_t j=0;j<outputs[i].size();++j) {
                const bool q8_small=std::strcmp(sweep_option,"MFQ_NINT_Q8_SMALL_WARPS")==0 && stage==0 && j>=2;
                compare(outputs[i][j],expected[i][j],!q8_small);++checks;
            }
        }
        const auto samples=fb::measure([&]{graph.replay();},mfq_current_cuda_stream(),banks.size());
        std::cout<<std::setprecision(10)<<"{\"stage\":\"gdn_"<<(stage==0?"input":"output")<<"\",\"M\":"<<rows
            <<",\"variant\":"<<variant<<",\"us\":"<<fb::median(samples)<<",\"samples_us\":[";
        for(size_t i=0;i<samples.size();++i)std::cout<<(i?",":"")<<samples[i];std::cout<<"]}\n"<<std::flush;
    }
    std::cout<<"VERIFY_GDN_PROJECTIONS checks="<<checks<<" PASS\n"<<std::flush;return 0;
}

static int packed_projection(CudaExecutionContext& execution,const mfq::ModelSource& model,bool router) {
    struct Bank {std::array<weight_loader::LinearGroup,2> grouped;weight_loader::Linear route;tb::Tensor x;};
    std::vector<Bank> banks;
    for(int layer:router?std::vector<int>{0,4,8,12,16,20,24,28,32,36,40,44}:std::vector<int>{0,1,8,9,16,17,24,25}) {
        const auto root="model.block."+std::to_string(layer)+(router?".mlp.":".linear_attention.");
        Bank b;
        int64_t width=0;
        if(router) {
            b.route=weight_loader::linear(execution,model,root+"router.weight");
            width=weight_loader::linear_weight(b.route)->neuron_len();
        }else {
            std::vector<weight_loader::Linear> original;
            for(const char* role:{"query","key","value","gate","alpha","beta"})
                original.push_back(weight_loader::linear(execution,model,root+role+".weight"));
            width=weight_loader::linear_weight(original.front())->neuron_len();
            for(int variant=0;variant<2;++variant) {
                // grouped_linear replaces these functions with views of its
                // packed storage. Each factory must start from original rows.
                auto functions=original;
                b.grouped[variant]=weight_loader::grouped_linear(execution,functions,variant?3:1);
                if(!b.grouped[variant])throw std::runtime_error("GDN packed AB unavailable");
            }
        }
        b.x=tb::tensor(fb::input(6,int(width))).reshape({1,6,width}).to(tb::kCUDA,tb::kFloat16);
        banks.push_back(std::move(b));
    }
    auto context=mfq::cuda::default_context(0);
    context->begin_graph_pool(mfq_current_cuda_stream());
    int checks=0;
    for(int rows:{1,2,5,6}) {
        const auto run=[&](size_t bank,int variant) {
            auto& b=banks[bank];const auto x=b.x.narrow(1,0,rows);
            if(router)return std::vector<tb::Tensor>{b.route(execution,x)};
            auto p=b.grouped[variant](execution,x);
            if(!variant)p={tb::cat({p[0],p[1],p[2]},-1),p[3],p[4],p[5]};
            // This is the storage handed to the real GDN short-token kernels.
            for(auto& y:p)y=y.contiguous();
            return p;
        };
        std::array<MfqCudaGraph,2> graphs;
        std::array<std::vector<std::vector<tb::Tensor>>,2> output;
        context->begin_graph_warmup(mfq_current_cuda_stream());
        for(int variant=0;variant<2;++variant) {
            if(router)option("MFQ_NINT_Q8_DECODE",variant?"3":"0");
            for(size_t i=0;i<banks.size();++i)output[variant].push_back(run(i,variant));
            fb::check(cudaStreamSynchronize(mfq_current_cuda_stream()));
        }
        for(auto& outputs:output)outputs.clear();
        for(int variant=0;variant<2;++variant) {
            if(router)option("MFQ_NINT_Q8_DECODE",variant?"3":"0");
            graphs[variant].capture_begin_shared();
            for(size_t i=0;i<banks.size();++i)output[variant].push_back(run(i,variant));
            graphs[variant].capture_end();
        }
        for(int step=0;step<3;++step) {
            for(auto& b:banks) {
                auto values=fb::input(6,int(b.x.size(-1)));for(auto& v:values)v*=float(step*step);
                b.x.copy_(tb::tensor(values).reshape(b.x.sizes()).to(tb::kCUDA,tb::kFloat16));
            }
            for(auto& graph:graphs)graph.replay();
            for(size_t i=0;i<banks.size();++i)for(size_t j=0;j<output[0][i].size();++j) {
                compare(output[1][i][j],output[0][i][j],!router);++checks;
            }
        }
        for(int repeat=0;repeat<3;++repeat)for(int offset=0;offset<2;++offset) {
            const int variant=(repeat+offset)%2;
            const auto samples=fb::measure([&]{graphs[variant].replay();},mfq_current_cuda_stream(),banks.size());
            std::cout<<std::setprecision(10)<<"{\"stage\":\""<<(router?"router":"gdn_packed")<<"\",\"M\":"<<rows
                <<",\"variant\":"<<variant<<",\"repeat\":"<<repeat<<",\"us\":"<<fb::median(samples)<<"}\n"<<std::flush;
        }
    }
    context->end_graph_pool(mfq_current_cuda_stream());
    std::cout<<"VERIFY_PACKED_AB checks="<<checks<<" PASS\n"<<std::flush;return 0;
}

static int ple_fusion(CudaExecutionContext& execution,const mfq::ModelSource& model) {
    const std::string root="model.block.1.position_embedding.";
    auto key=weight_loader::linear(execution,model,root+"key.weight");
    auto value=weight_loader::linear(execution,model,root+"value.weight");
    auto kn=load_dense_gpu(execution,model,root+"key_norm.weight");
    auto qn=load_dense_gpu(execution,model,root+"query_norm.weight");
    auto cn=load_dense_gpu(execution,model,root+"conv_norm.weight");
    auto weight=load_dense_gpu(execution,model,root+"conv.weight");
    const int64_t width=weight_loader::linear_weight(key)->neuron_len(),hidden=qn.numel()/4;
    const int64_t taps=weight.size(-1),dilation=3,history=(taps-1)*dilation;
    auto context=mfq::cuda::default_context(0);context->begin_graph_pool(mfq_current_cuda_stream());
    int checks=0;
    for(int rows:{1,2,3,4,5,6,8,32})for(bool half:{false,true}) {
        auto x=tb::tensor(fb::input(rows,int(hidden*4))).reshape({1,rows,hidden*4}).to(tb::kCUDA,half?tb::kFloat16:tb::kFloat32);
        auto embeddings=tb::tensor(fb::input(rows,int(width))).reshape({1,rows,width}).to(tb::kCUDA,tb::kFloat16);
        auto previous=tb::tensor(fb::input(int(history),int(hidden*4))).reshape({1,history,hidden*4}).to(tb::kCUDA,tb::kFloat32);
        const auto run=[&](int candidate) {
            auto k=mfq_qwen4_exp::grouped_rms_norm(key(execution,embeddings),kn,hidden,1e-6);
            auto q=mfq_qwen4_exp::grouped_rms_norm(x,qn,hidden,1e-6);
            auto v=value(execution,embeddings);
            tb::Tensor gated,normalized;
            if(candidate) {
                auto pair=mfq_qwen4_exp::ple_gate_norm_cuda(k,q,v,cn.to(tb::kFloat32).contiguous(),hidden,4,1e-6);
                gated=pair[0];normalized=pair[1];
            }else {
                auto score=(k.to(tb::kFloat32).reshape({1,rows,4,hidden})*
                    q.to(tb::kFloat32).reshape({1,rows,4,hidden})).sum(-1)/std::sqrt(double(hidden));
                auto sign=(score>0).to(tb::kFloat32)-(score<0).to(tb::kFloat32);
                auto root=sign*tb::clamp_min(score.abs(),1e-6).sqrt();
                gated=(tb::sigmoid(root).unsqueeze(-1)*v.to(tb::kFloat32).unsqueeze(-2)).reshape({1,rows,4*hidden});
                normalized=mfq_qwen4_exp::grouped_rms_norm(gated,cn,hidden,1e-6);
            }
            auto out=candidate?mfq_qwen4_exp::ple_conv_add_cuda(normalized,weight,previous,dilation,gated,half):
                mfq_qwen4_exp::ple_dilated_conv_silu(normalized,weight,previous,dilation);
            if(!candidate)out[0]=(gated+out[0]).to(x.scalar_type());
            return std::vector<tb::Tensor>{gated,normalized,out[0],out[1]};
        };
        std::array<MfqCudaGraph,2> graphs;std::array<std::vector<tb::Tensor>,2> outputs;
        context->begin_graph_warmup(mfq_current_cuda_stream());
        for(int candidate=0;candidate<2;++candidate){outputs[candidate]=run(candidate);fb::check(cudaStreamSynchronize(mfq_current_cuda_stream()));}
        outputs={};
        for(int candidate=0;candidate<2;++candidate){graphs[candidate].capture_begin_shared();outputs[candidate]=run(candidate);graphs[candidate].capture_end();}
        for(int step=0;step<3;++step) {
            auto values=fb::input(rows,int(hidden*4));for(auto& v:values)v*=step==0?0.f:float(step*step);
            x.copy_(tb::tensor(values).reshape(x.sizes()).to(tb::kCUDA,x.scalar_type()));
            for(auto& graph:graphs)graph.replay();
            for(size_t field=0;field<outputs[0].size();++field){compare(outputs[1][field],outputs[0][field],true);++checks;}
            previous.copy_(outputs[0][3]);
        }
        std::array<std::vector<double>,2> samples;
        for(int repeat=0;repeat<3;++repeat)for(int offset=0;offset<2;++offset) {
            const int candidate=(repeat+offset)%2;
            samples[candidate].push_back(fb::median(fb::measure([&]{graphs[candidate].replay();},mfq_current_cuda_stream(),1)));
        }
        std::cout<<"PLE_AB M="<<rows<<" half="<<half<<" old_us="<<fb::median(samples[0])
            <<" new_us="<<fb::median(samples[1])<<" BYTE_EXACT\n"<<std::flush;
    }
    context->end_graph_pool(mfq_current_cuda_stream());
    std::cout<<"VERIFY_PLE checks="<<checks<<" BYTE_EXACT PASS\n";return 0;
}

static int head_projection(CudaExecutionContext& execution,const mfq::ModelSource& model,const char* sweep) {
    auto head=load_quant_linear(execution,model,"model.output.weight");
    if(!head.is_nint() || !head.nint.aligned_q8 || head.nint.gs!=48)
        throw std::runtime_error("head probe requires aligned GS48 Q8");
    auto context=mfq::cuda::default_context(0);
    context->begin_graph_pool(mfq_current_cuda_stream());
    const auto options=tb::TensorOptions().dtype(tb::kFloat16).device(tb::kCUDA);
    int checks=0;
    for(int rows:{1,2,3,4,5,6}) {
        auto x=tb::empty({1,rows,head.neuron_len()},options);
        const auto set=[&](int variant) {
            option("MFQ_NINT_Q8_DECODE","2");
            const bool u8s8=std::strcmp(sweep,"u8s8")==0;
            option("MFQ_NINT_Q8_U8S8",u8s8 && variant?"1":"0");
            if(u8s8) {
                option("MFQ_NINT_Q8_SUM","");option("MFQ_NINT_Q8_VECTOR","");option("MFQ_NINT_Q8_EXACT_ROWS","");return;
            }
            const bool exact=std::strcmp(sweep,"exact")==0,new_best=std::strcmp(sweep,"new-best")==0;
            option("MFQ_NINT_Q8_SUM",exact || std::strcmp(sweep,"vector")==0?"0":variant?"1":"0");
            option("MFQ_NINT_Q8_VECTOR",exact || std::strcmp(sweep,"sum")==0?"0":variant?"1":"0");
            option("MFQ_NINT_Q8_EXACT_ROWS",(exact || new_best) && variant?"1":"0");
        };
        std::array<MfqCudaGraph,2> graphs;
        std::array<tb::Tensor,2> output;
        context->begin_graph_warmup(mfq_current_cuda_stream());
        for(int candidate=0;candidate<2;++candidate) {
            set(candidate);output[candidate]=head.forward(execution,x);
            fb::check(cudaStreamSynchronize(mfq_current_cuda_stream()));
        }
        output={};
        for(int candidate=0;candidate<2;++candidate) {
            set(candidate);graphs[candidate].capture_begin_shared();
            output[candidate]=head.forward(execution,x);graphs[candidate].capture_end();
        }
        for(int step=0;step<3;++step) {
            auto values=fb::input(rows,int(head.neuron_len()));
            for(size_t i=0;i<values.size();++i)values[i]=step==0?0.f:
                values[i]*float(step*step)+std::sin(float(i)*.019f+step)*.21f;
            x.copy_(tb::tensor(values).reshape(x.sizes()).to(tb::kCUDA,tb::kFloat16));
            for(auto& graph:graphs)graph.replay();
            compare(output[1],output[0],true);++checks;
        }
        std::array<std::vector<double>,2> samples;
        for(int repeat=0;repeat<3;++repeat)for(int offset=0;offset<2;++offset) {
            const int candidate=(repeat+offset)%2;
            samples[candidate].push_back(fb::median(fb::measure([&]{graphs[candidate].replay();},
                mfq_current_cuda_stream(),1)));
        }
        std::cout<<"HEAD_AB sweep="<<sweep<<" M="<<rows<<" old_us="<<fb::median(samples[0])
            <<" new_us="<<fb::median(samples[1])<<" BYTE_EXACT\n"<<std::flush;
    }
    context->end_graph_pool(mfq_current_cuda_stream());
    std::cout<<"VERIFY_HEAD_AB checks="<<checks<<" BYTE_EXACT PASS\n"<<std::flush;
    return 0;
}

static int gdn_strided(CudaExecutionContext& execution) {
    constexpr int nk=16,nv=96,width=128,channels=(2*nk+nv)*width,taps=4;
    auto context=mfq::cuda::default_context(0);context->begin_graph_pool(mfq_current_cuda_stream());
    const auto options=tb::TensorOptions().dtype(tb::kFloat32).device(tb::kCUDA);
    const auto values=[&](std::vector<int64_t> shape,int seed,float amplitude,tb::ScalarType dtype) {
        int64_t elements=1;for(auto size:shape)elements*=size;
        std::vector<float> data(elements);
        for(int64_t i=0;i<elements;++i)data[i]=std::sin(float(i+seed)*.193f)*amplitude;
        return tb::tensor(data).reshape(shape).to(tb::kCUDA,dtype);
    };
    int checks=0;
    for(int mode:{0,1})for(int rows:{1,2,3,4,5,6})for(bool gate_float:{false,true}) {
        if(mode==1 && rows==1)continue;
        const int batch=mode==0?rows:1,tokens=mode==0?1:rows;
        const auto dtype=gate_float?tb::kFloat32:tb::kFloat16;
        auto packed=values({batch,tokens,channels+128},11,.31f,tb::kFloat16);
        auto projected=values({batch,tokens,nv*width+2*nv+128},12,.31f,dtype);
        auto qkv=packed.narrow(-1,0,channels),gate=projected.narrow(-1,0,nv*width);
        auto alpha=projected.narrow(-1,nv*width,nv),beta=projected.narrow(-1,nv*(width+1),nv);
        auto conv=values({batch,taps-1,channels},13,.1f,tb::kFloat32);
        auto state=values({batch,nv,width,width},14,.03f,tb::kFloat32);
        auto weight=values({channels,1,taps},15,.13f,tb::kFloat32);
        auto bias=values({nv},16,.3f,tb::kFloat32),a=values({nv},17,.3f,tb::kFloat32);
        auto norm=values({width},18,1.3f,tb::kFloat32);
        auto workspace=tb::zeros({batch,nv,width+1},options);
        const auto compute=[&](bool direct) {
            option("MFQ_GDN_NORM_WARP_REDUCTION",direct?"1":"0");
            auto q=direct?qkv:qkv.contiguous(),z=direct?gate:gate.contiguous();
            if(mode==0)return gdn_decode_core_cuda(q,z,alpha,beta,conv,state,weight,bias,a,norm,
                nk,nv,width,1e-6,1e-6,false,gate_float?false:true,true,workspace,false);
            auto p=linear_conv_qkv_gate_short_cuda(conv,q,weight,direct?alpha:alpha.contiguous(),
                direct?beta:beta.contiguous(),bias,a,nk,nv,width,1e-6);
            auto r=gdn_transposed_cuda(p[0],p[1],p[2],p[4],p[5],state);
            auto y=gdn_rms_norm_gate_cuda(r[0],z,norm,1e-6,false,gate_float?false:true);
            p.push_back(y);p.push_back(r[1]);return p;
        };
        std::array<MfqCudaGraph,2> graphs;std::array<std::vector<tb::Tensor>,2> out;
        context->begin_graph_warmup(mfq_current_cuda_stream());
        for(int variant=0;variant<2;++variant)out[variant]=compute(variant!=0);
        fb::check(cudaStreamSynchronize(mfq_current_cuda_stream()));out={};
        for(int variant=0;variant<2;++variant) {
            graphs[variant].capture_begin_shared();out[variant]=compute(variant!=0);graphs[variant].capture_end();
        }
        for(int step=0;step<3;++step) {
            packed.copy_(values(packed.sizes().vec(),31+step,step*.21f,tb::kFloat16));
            projected.copy_(values(projected.sizes().vec(),41+step,step*3.f,dtype));
            conv.copy_(values(conv.sizes().vec(),51+step,step*.11f,tb::kFloat32));
            for(auto& graph:graphs)graph.replay();
            for(size_t field=0;field<out[0].size();++field){compare(out[1][field],out[0][field],true);++checks;}
        }
        // Timing the frequent M1, B2 and MTP M5/M6 cases is sufficient.
        if(rows==1 || rows==2 || rows==5 || rows==6) {
            std::array<std::vector<double>,2> samples;
            for(int repeat=0;repeat<2;++repeat)for(int offset=0;offset<2;++offset) {
                const int variant=(repeat+offset)%2;
                samples[variant].push_back(fb::median(fb::measure([&]{graphs[variant].replay();},mfq_current_cuda_stream(),1)));
            }
            std::cout<<"GDN_STRIDE_AB mode="<<mode<<" M="<<rows<<" gate_float="<<gate_float
                <<" old_us="<<fb::median(samples[0])<<" new_us="<<fb::median(samples[1])<<" BYTE_EXACT\n"<<std::flush;
        }
    }
    context->end_graph_pool(mfq_current_cuda_stream());
    std::cout<<"VERIFY_GDN_STRIDE checks="<<checks<<" BYTE_EXACT PASS\n";return 0;
}

static int router_short_batch() {
    auto context=mfq::cuda::default_context(0);context->begin_graph_pool(mfq_current_cuda_stream());
    int checks=0;
    for(int rows:{1,2,3,4,5,6})for(bool half:{false,true})for(int setting=0;setting<3;++setting) {
        auto x=tb::empty({rows,512},tb::TensorOptions().device(tb::kCUDA).dtype(half?tb::kFloat16:tb::kFloat32));
        const float floor=setting==0?1e-20f:setting==1?.5f:1e-7f;
        const float scale=setting==0?1.f:setting==1?2.75f:0.f;
        const auto run=[&](bool candidate) {
            option("MFQ_MOE_TOPK_512_BATCH",candidate?"1":"0");
            return moe_topk_cuda(x,10,false,false,true,false,mfq_nullopt,floor,scale);
        };
        std::array<MfqCudaGraph,2> graphs;std::array<std::vector<tb::Tensor>,2> out;
        context->begin_graph_warmup(mfq_current_cuda_stream());
        for(int variant=0;variant<2;++variant)out[variant]=run(variant!=0);
        fb::check(cudaStreamSynchronize(mfq_current_cuda_stream()));out={};
        for(int variant=0;variant<2;++variant) {
            graphs[variant].capture_begin_shared();out[variant]=run(variant!=0);graphs[variant].capture_end();
        }
        for(int pattern=0;pattern<12;++pattern)for(int step=0;step<3;++step) {
            std::vector<float> values(rows*512);
            for(size_t i=0;i<values.size();++i) {
                const int expert=int(i%512),row=int(i/512);
                float v=std::sin(float(i)*.319f+float(step+row)*.721f)*float(step+1);
                if(pattern==0)v=0.f;
                if(pattern==1)v=float((expert+step)%7-3);
                if(pattern==2)v=float(expert)*.003f+step;
                if(pattern==3)v=-float(expert)*.003f-step;
                if(pattern==4)v=(expert+step)%37==0?1000.f:-1000.f;
                if(pattern==5 && (expert+step)%11==0)v=std::numeric_limits<float>::quiet_NaN();
                if(pattern==6)v=std::numeric_limits<float>::quiet_NaN();
                if(pattern==7 && expert==step)v=std::numeric_limits<float>::infinity();
                if(pattern==8)v=-std::numeric_limits<float>::infinity();
                if(pattern==9)v=(expert&1)?-0.f:0.f;
                if(pattern==11)v*=1000.f;
                values[i]=v;
            }
            x.copy_(tb::tensor(values).reshape(x.sizes()).to(tb::kCUDA,x.scalar_type()));
            for(auto& graph:graphs)graph.replay();
            for(int field=0;field<2;++field){compare(out[1][field],out[0][field],true);++checks;}
        }
        if(setting==0 && (rows==2 || rows==5 || rows==6)) {
            x.copy_(tb::tensor(fb::input(rows,512)).reshape(x.sizes()).to(tb::kCUDA,x.scalar_type()));
            std::array<std::vector<double>,2> samples;
            for(int repeat=0;repeat<2;++repeat)for(int offset=0;offset<2;++offset) {
                const int variant=(repeat+offset)%2;
                samples[variant].push_back(fb::median(fb::measure([&]{graphs[variant].replay();},mfq_current_cuda_stream(),1)));
            }
            std::cout<<"ROUTER_BATCH_AB M="<<rows<<" half="<<half<<" old_us="<<fb::median(samples[0])
                <<" new_us="<<fb::median(samples[1])<<" BYTE_EXACT\n"<<std::flush;
        }
    }
    context->end_graph_pool(mfq_current_cuda_stream());
    std::cout<<"VERIFY_ROUTER_BATCH checks="<<checks<<" BYTE_EXACT PASS\n";return 0;
}

static int gs_projection(CudaExecutionContext& execution,const mfq::ModelSource& model) {
    const auto* setting=std::getenv("MFQ_GS_PROBE_OPTION");
    const char* sweep=setting?setting:"MFQ_NINT_OUTPUT_NARROW";
    const bool q8_small=std::strcmp(sweep,"MFQ_NINT_Q8_SMALL_WARPS")==0;
    const bool q8_exact=std::strcmp(sweep,"MFQ_NINT_Q8_EXACT_ROWS")==0;
    const bool q8=q8_small || q8_exact;
    const auto* selected_rows=std::getenv("MFQ_GS_PROBE_ROWS");
    const bool small_groups=std::strcmp(sweep,"MFQ_NINT_FIXED_SMALL_GROUPS")==0;
    const bool fused=std::strcmp(sweep,"MFQ_NINT_FUSED_FIXED_BITS")==0;
    const bool timing=std::getenv("MFQ_GS_PROBE_NO_TIMING")==nullptr;
    const auto input_dtype=std::getenv("MFQ_PROBE_FLOAT_INPUT")?tb::kFloat32:tb::kFloat16;
    const auto batches=selected_rows?std::vector<int>{std::atoi(selected_rows)}:
        small_groups?std::vector<int>{1,2,3,4,5,6}:std::vector<int>{1,2,5,6};
    std::set<std::array<int,4>> geometries;
    auto context=mfq::cuda::default_context(0);
    context->begin_graph_pool(mfq_current_cuda_stream());
    int checks=0;
    for(const auto& tensor:model.tensors()) {
        if(tensor.dtype!="NINT" || tensor.name.find("model.block.")!=0 ||
           tensor.name.find(".ngram.")!=std::string::npos)continue;
        const auto bytes=model.read(tensor.name,0,14);
        int gs=0,width=0;std::memcpy(&gs,bytes.data()+2,4);std::memcpy(&width,bytes.data()+10,4);
        if(q8?gs!=48:(gs!=24 && gs!=28))continue;
        const auto prefix=model.read(tensor.name,14,4);
        uint32_t ndim=0;std::memcpy(&ndim,prefix.data(),4);
        const auto shape=model.read(tensor.name,18+ndim*8,4);
        int out=0;std::memcpy(&out,shape.data(),4);
        if(q8_small && out>128)continue;
        if(q8_exact && (out<=128 || out>16384 || width!=2560))continue;
        if(std::strcmp(sweep,"MFQ_NINT_FIXED_SMALL_GROUPS")==0 && (width+gs-1)/gs>32)continue;
        if(std::strcmp(sweep,"MFQ_NINT_FIXED_LONG_GROUPS")==0 && (width!=6144 || out!=2560))continue;
        if(fused && (out!=2560 || (width!=640 && width!=6144)))continue;
        if(!geometries.insert({gs,int(uint8_t(bytes[0])&127),width,out}).second)continue;
        auto projection=weight_loader::linear(execution,model,tensor.name);
        auto weight=weight_loader::linear_weight(projection);
        std::cout<<"GS_WEIGHT name="<<tensor.name<<" gs="<<gs<<" bits="<<weight->nint.uniform_qbits
            <<" width="<<width<<" out="<<out<<'\n'<<std::flush;
        for(int rows:batches) {
            auto x=tb::tensor(fb::input(rows,width)).reshape({1,rows,width}).to(tb::kCUDA,input_dtype);
            auto gate=tb::tensor(fb::input(rows,width)).reshape(x.sizes()).to(tb::kCUDA,input_dtype);
            const auto compute=[&] {
                return fused?weight->forward_input_mul(execution,x,gate,width==640?2:1):projection(execution,x);
            };
            std::array<tb::Tensor,2> outputs;
            std::array<MfqCudaGraph,2> graphs;
            context->begin_graph_warmup(mfq_current_cuda_stream());
            for(int variant=0;variant<2;++variant) {
                option(sweep,variant?"1":"0");
                outputs[variant]=compute();fb::check(cudaStreamSynchronize(mfq_current_cuda_stream()));
            }
            outputs={};
            for(int variant=0;variant<2;++variant) {
                option(sweep,variant?"1":"0");
                graphs[variant].capture_begin_shared();outputs[variant]=compute();graphs[variant].capture_end();
            }
            for(int step=0;step<3;++step) {
                auto input=fb::input(rows,width);
                for(auto& value:input)value*=float(step*step);
                x.copy_(tb::tensor(input).reshape(x.sizes()).to(tb::kCUDA,input_dtype));
                if(fused) {
                    for(auto& value:input)value=std::sin(value*17.f+step)*float(step+1);
                    gate.copy_(tb::tensor(input).reshape(gate.sizes()).to(tb::kCUDA,input_dtype));
                }
                for(auto& graph:graphs)graph.replay();
                compare(outputs[1],outputs[0],!q8_small);++checks;
            }
            if(!timing)continue;
            std::array<std::vector<double>,2> samples;
            for(int repeat=0;repeat<2;++repeat)for(int offset=0;offset<2;++offset) {
                const int variant=(repeat+offset)%2;
                samples[variant].push_back(fb::median(fb::measure([&]{graphs[variant].replay();},mfq_current_cuda_stream(),1)));
            }
            std::cout<<std::setprecision(10)<<"GS_AB option="<<sweep<<" gs="<<gs<<" bits="<<weight->nint.uniform_qbits
                <<" N="<<out<<" K="<<width<<" M="<<rows<<" old_us="<<fb::median(samples[0])
                <<" new_us="<<fb::median(samples[1])<<'\n'<<std::flush;
        }
    }
    context->end_graph_pool(mfq_current_cuda_stream());
    if(geometries.empty())throw std::runtime_error("no stored GS projection selected");
    std::cout<<"VERIFY_GS_PROJECTIONS geometries="<<geometries.size()<<" checks="<<checks<<" PASS\n";
    return 0;
}

static int final_bf16_gr(CudaExecutionContext& execution,const mfq::ModelSource& model) {
    auto norm=load_dense_gpu(execution,model,"model.mhc.pre.norm.weight").to(tb::kFloat32).contiguous();
    auto down=load_quant_linear(execution,model,"model.mhc.pre.down.weight").dense;
    auto up=load_quant_linear(execution,model,"model.mhc.pre.up.weight").dense;
    std::cout<<"FINAL_GR_WEIGHT down_dtype="<<int(down.scalar_type())<<" down_N="<<down.size(0)
        <<" down_K="<<down.size(1)<<" up_dtype="<<int(up.scalar_type())<<" up_N="<<up.size(0)
        <<" up_K="<<up.size(1)<<'\n'<<std::flush;
    if(down.scalar_type()!=tb::kBFloat16 || up.scalar_type()!=tb::kBFloat16)
        throw std::runtime_error("final GR test requires real BF16 Down/Up");
    struct Bank {tb::Tensor down,up,input;};
    std::vector<Bank> banks;
    for(int i=0;i<4;++i)banks.push_back({i?down.clone():down,i?up.clone():up,{}});
    int checks=0;
    auto context=mfq::cuda::default_context(mfq_current_cuda_device());
    context->begin_graph_pool(mfq_current_cuda_stream());
    for(auto dtype:{tb::kFloat16,tb::kFloat32})for(int rows:{1,2,3,4,5,6}) {
        for(auto& b:banks)b.input=tb::empty({1,rows,10240},norm.options().dtype(dtype));
        const auto compute=[&](const Bank& b,int variant) {
            auto normalized=mfq_qwen4_exp::grouped_rms_norm(b.input,norm,2560,1e-6);
            auto low=variant?try_gr_bf16_down_cuda(normalized,b.down,4):
                mfq_qwen4_exp::gated_residual_bottleneck(mfq_selected_attention::promoted_matmul(normalized,b.down.transpose(-1,-2)),4);
            if(!low.defined())throw std::runtime_error("BF16 Down did not dispatch");
            auto output=variant?try_gr_bf16_up_mix_cuda(low,normalized,b.up,4):
                mfq_qwen4_exp::gated_residual_mix(mfq_selected_attention::promoted_matmul(low,b.up.transpose(-1,-2)),normalized,4);
            if(!output.defined())throw std::runtime_error("BF16 Up did not dispatch");
            return std::vector<tb::Tensor>{low,output};
        };
        std::array<MfqCudaGraph,2> graphs;
        std::array<std::vector<std::vector<tb::Tensor>>,2> outputs;
        context->begin_graph_warmup(mfq_current_cuda_stream());
        for(int variant=0;variant<2;++variant)for(const auto& b:banks)
            outputs[variant].push_back(compute(b,variant));
        fb::check(cudaStreamSynchronize(mfq_current_cuda_stream()));
        outputs={};
        for(int variant=0;variant<2;++variant) {
            graphs[variant].capture_begin_shared();
            for(const auto& b:banks)outputs[variant].push_back(compute(b,variant));
            graphs[variant].capture_end();
        }
        for(int step=0;step<3;++step) {
            auto values=fb::input(rows,10240);for(auto& value:values)value*=float(step*step);
            auto input=tb::tensor(values).reshape({1,rows,10240}).to(tb::kCUDA,dtype);
            for(auto& b:banks)b.input.copy_(input);
            for(auto& graph:graphs)graph.replay();
            for(size_t i=0;i<banks.size();++i)for(int stage=0;stage<2;++stage) {
                auto actual=outputs[1][i][stage].cpu().contiguous(),expected=outputs[0][i][stage].cpu().contiguous();
                if(actual.sizes()!=expected.sizes() || actual.scalar_type()!=expected.scalar_type())
                    throw std::runtime_error("BF16 final GR changed output type/shape");
                double error=0,signal=0,maximum=0;
                for(int64_t j=0;j<actual.numel();++j) {
                    const double a=actual.data_ptr<float>()[j],e=expected.data_ptr<float>()[j],diff=std::abs(a-e);
                    if(!std::isfinite(a) || !std::isfinite(e) || diff>.0002+.0001*std::abs(e))
                        throw std::runtime_error("BF16 final GR absolute error exceeds original Float tolerance");
                    error+=diff*diff;signal+=e*e;maximum=std::max(maximum,diff);
                }
                const double relative=std::sqrt(error/std::max(signal,1e-30));
                if(relative>.0001)throw std::runtime_error("BF16 final GR relative error exceeds original Float tolerance");
                std::cout<<"FINAL_GR_ERROR M="<<rows<<" half="<<(dtype==tb::kFloat16)<<" stage="<<stage
                    <<" relative="<<relative<<" absolute="<<maximum<<'\n';++checks;
            }
        }
        std::array<std::vector<double>,2> times;
        for(int repeat=0;repeat<2;++repeat)for(int offset=0;offset<2;++offset) {
            const int variant=(repeat+offset)%2;
            times[variant].push_back(fb::median(fb::measure([&]{graphs[variant].replay();},mfq_current_cuda_stream(),banks.size())));
        }
        std::cout<<std::setprecision(10)<<"FINAL_GR_AB M="<<rows<<" half="<<(dtype==tb::kFloat16)
            <<" old_us="<<fb::median(times[0])<<" new_us="<<fb::median(times[1])<<" weight_banks=4\n"<<std::flush;
    }
    context->end_graph_pool(mfq_current_cuda_stream());
    std::cout<<"VERIFY_FINAL_GR_BF16 checks="<<checks<<" original_float_tolerance=1 PASS\n";return 0;
}

static int row_bounds_boundaries(CudaExecutionContext& execution) {
    int checks=0;
    for(int gs:{24,28})for(int groups:{33,92,107,220,256})for(int salt=0;salt<8;++salt)for(int padding:{0,8})for(int tail:{0,13}) {
        std::cout<<"ROW_BOUND_FIXTURE gs="<<gs<<" groups="<<groups<<" offset="<<salt<<" padding="<<padding<<" tail="<<tail<<std::endl;
        constexpr int out=17;const int bits=gs==24?6:5;
        const int width=groups*gs-tail;
        NintWeight w;w.gs=gs;w.ng=groups;w.out=out;w.neuron_len=width;w.uniform_qbits=bits;w.bits=bits;
        std::vector<uint8_t> row_bits(out,bits),scale(out*groups),minimum(out*groups);
        std::vector<int64_t> offsets(out);std::vector<float> outer(out),outer_min(out);
        const int64_t bit_count=salt+int64_t(out)*groups*gs*bits;
        std::vector<uint8_t> packed((bit_count+7)/8+padding);
        for(size_t i=0;i<packed.size();++i)packed[i]=uint8_t(i*173+i/7);
        for(int row=0;row<out;++row) {
            offsets[row]=salt+int64_t(row)*groups*gs*bits;
            outer[row]=float(row%5-2)*.000031f;outer_min[row]=float(row%7-3)*.000027f;
        }
        for(size_t i=0;i<scale.size();++i){scale[i]=uint8_t(i*31);minimum[i]=uint8_t(i*17);}
        w.q_packed=tb::tensor(packed).to(tb::kCUDA);w.row_q_bits=tb::tensor(row_bits).to(tb::kCUDA);
        w.row_q_bit_offsets=tb::tensor(offsets).to(tb::kCUDA);
        w.sub_scale=tb::tensor(scale).reshape({out,groups}).to(tb::kCUDA);
        w.sub_min=tb::tensor(minimum).reshape({out,groups}).to(tb::kCUDA);
        w.neuron_scale=tb::tensor(outer).to(tb::kCUDA);w.neuron_min=tb::tensor(outer_min).to(tb::kCUDA);
        for(int rows:{1,2,5,6}) {
            auto x=tb::zeros({rows,width},w.neuron_scale.options().dtype(tb::kFloat16));
            auto& ws=w.workspace(rows);
            const auto run=[&]{return nint_matmul_fixed_bits_ws_cuda(w.q_packed,w.row_q_bits,
                w.row_q_bit_offsets,w.sub_scale,w.sub_min,w.neuron_scale,w.neuron_min,
                x,w.gs,ws.qx,ws.xscale,w.uniform_qbits);};
            option("MFQ_NINT_ROW_BOUNDS","1");
            MfqCudaGraph graph;mfq_prepare_cuda_graph_memory(graph);
            auto y=run();fb::check(cudaStreamSynchronize(mfq_current_cuda_stream()));y={};
            graph.capture_begin();y=run();graph.capture_end();
            for(int step=0;step<3;++step) {
                auto values=fb::input(rows,width);for(auto& v:values)v*=float(step*step);
                x.copy_(tb::tensor(values).reshape(x.sizes()).to(tb::kCUDA,tb::kFloat16));
                option("MFQ_NINT_ROW_BOUNDS","0");const auto expected=run().cpu();
                graph.replay();compare(y,expected,true);++checks;
            }
        }
    }
    std::cout<<"VERIFY_ROW_BOUNDS checks="<<checks<<" gs24/6+gs28/5 groups33/92/107/220/256 offsets0..7 unpadded/padded tails0/13 changing_graph PASS\n"<<std::flush;
    return 0;
}

int main(int argc,char** argv)try {
    if(argc<2 || argc>3)throw std::runtime_error("usage: nint-dense-decode-probe model [--verify-only]");
    const bool verify_only=argc==3 && std::strcmp(argv[2],"--verify-only")==0;
    auto context=mfq::cuda::default_context(0);auto stream=mfq_get_stream_from_pool(false);MfqCudaStreamGuard guard(stream);
    CudaExecutionContext execution;execution.decode_graph_serial_branches=true;
    auto model=mfq::open_model_source(argv[1]);
    if(argc==3 && std::strcmp(argv[2],"--final-bf16-gr")==0)return final_bf16_gr(execution,*model);
    if(argc==3 && std::strcmp(argv[2],"--gs-projections")==0)return gs_projection(execution,*model);
    if(argc==3 && std::strcmp(argv[2],"--router-short-batch")==0)return router_short_batch();
    if(argc==3 && std::strcmp(argv[2],"--gdn-strided")==0)return gdn_strided(execution);
    if(argc==3 && std::strcmp(argv[2],"--ple-fused")==0)return ple_fusion(execution,*model);
    if(argc==3 && std::strcmp(argv[2],"--head-u8s8")==0)return head_projection(execution,*model,"u8s8");
    if(argc==3 && std::strcmp(argv[2],"--gdn-warp-rows-auto")==0)return gdn_projection(execution,*model,"MFQ_NINT_WARP_ROWS_AUTO");
    if(argc==3 && std::strcmp(argv[2],"--gdn-output-one-warp")==0)return gdn_projection(execution,*model,"MFQ_NINT_OUTPUT_ONE_WARP");
    if(argc==3 && std::strcmp(argv[2],"--gdn-fixed-92")==0)return gdn_projection(execution,*model,"MFQ_NINT_FIXED_92_GROUPS");
    if(argc==3 && std::strcmp(argv[2],"--gdn-row-bounds")==0)return gdn_projection(execution,*model,"MFQ_NINT_ROW_BOUNDS");
    if(argc==3 && std::strcmp(argv[2],"--row-bounds-boundaries")==0)return row_bounds_boundaries(execution);
    if(argc==3 && std::strcmp(argv[2],"--gdn-float-whole")==0)return gdn_projection(execution,*model,"MFQ_NINT_FLOAT_WHOLE_GROUP");
    if(argc==3 && std::strcmp(argv[2],"--gdn-q8-small")==0)return gdn_projection(execution,*model,"MFQ_NINT_Q8_SMALL_WARPS");
    if(argc==3 && std::strcmp(argv[2],"--gdn-group-quantize")==0)return gdn_projection(execution,*model,"MFQ_NINT_GROUP_QUANTIZE");
    if(argc==3 && std::strcmp(argv[2],"--gdn-dualgroups")==0)return gdn_projection(execution,*model,"MFQ_NINT_OUTPUT_DUAL_GROUP");
    if(argc==3 && std::strcmp(argv[2],"--head-sum")==0)return head_projection(execution,*model,"sum");
    if(argc==3 && std::strcmp(argv[2],"--head-vector")==0)return head_projection(execution,*model,"vector");
    if(argc==3 && std::strcmp(argv[2],"--head-both")==0)return head_projection(execution,*model,"both");
    if(argc==3 && std::strcmp(argv[2],"--head-exact")==0)return head_projection(execution,*model,"exact");
    if(argc==3 && std::strcmp(argv[2],"--head-new-best")==0)return head_projection(execution,*model,"new-best");
    if(argc==3 && std::strcmp(argv[2],"--gdn-formats")==0) {
        for(int layer:{0,1,8,9,16,17,24,25}) {
            const auto root="model.block."+std::to_string(layer)+".linear_attention.";
            auto output=weight_loader::linear(execution,*model,root+"output.weight");
            const auto w=weight_loader::linear_weight(output);
            std::cout<<"GDN_OUTPUT_FORMAT layer="<<layer<<" bits="<<w->nint.bits
                <<" uniform="<<w->nint.uniform_qbits<<" gs="<<w->nint.gs<<" groups="<<w->nint.ng<<'\n';
        }
        return 0;
    }
    if(argc==3 && std::strcmp(argv[2],"--gdn")==0)return gdn_projection(execution,*model);
    if(argc==3 && std::strcmp(argv[2],"--gdn-row-pairs")==0)return gdn_projection(execution,*model,"MFQ_NINT_ROW_PAIRS");
    if(argc==3 && std::strcmp(argv[2],"--gdn-interleaved")==0)return gdn_projection(execution,*model,"MFQ_NINT_INTERLEAVED");
    if(argc==3 && std::strcmp(argv[2],"--gdn-narrow")==0)return gdn_projection(execution,*model,"MFQ_NINT_NARROW_OFFSETS");
    if(argc==3 && std::strcmp(argv[2],"--gdn-fixed-exact")==0)return gdn_projection(execution,*model,"MFQ_NINT_FIXED_EXACT_ROWS");
    if(argc==3 && std::strcmp(argv[2],"--gdn-best")==0)return gdn_projection(execution,*model,"MFQ_NINT_BEST");
    if(argc==3 && std::strcmp(argv[2],"--gdn-fixed-output")==0)return gdn_projection(execution,*model,"MFQ_NINT_FIXED_OUTPUT");
    if(argc==3 && std::strcmp(argv[2],"--gdn-output-warps")==0)return gdn_projection(execution,*model,"MFQ_NINT_OUTPUT_TWO_WARPS");
    if(argc==3 && std::strcmp(argv[2],"--gdn-fixed")==0)return gdn_projection(execution,*model,"MFQ_NINT_FIXED_BITS");
    if(argc==3 && std::strcmp(argv[2],"--gdn-sums")==0)return gdn_projection(execution,*model,"MFQ_NINT_ACTIVATION_SUM");
    if(argc==3 && std::strcmp(argv[2],"--gdn-warps")==0)return gdn_projection(execution,*model,"MFQ_NINT_SHORT_WARPS");
    if(argc==3 && std::strcmp(argv[2],"--gdn-staged")==0)return gdn_projection(execution,*model,"MFQ_NINT_PACKED_STAGE");
    if(argc==3 && std::strcmp(argv[2],"--gdn-warp-rows")==0)return gdn_projection(execution,*model,"MFQ_NINT_WARP_ROWS");
    if(argc==3 && std::strcmp(argv[2],"--gdn-packed")==0)return packed_projection(execution,*model,false);
    if(argc==3 && std::strcmp(argv[2],"--router")==0)return packed_projection(execution,*model,true);
    option("MFQ_NINT_Q8_DECODE","0");joined_qkv(execution,*model);
    auto head=load_quant_linear(execution,*model,"model.output.weight");
    if(!head.is_nint() || !head.nint.aligned_q8 || head.nint.gs!=48)throw std::runtime_error("real head is not aligned GS48 NINT8");
    std::cout<<"HEAD N="<<head.out()<<" K="<<head.neuron_len()<<" groups="<<head.nint.ng<<" bytes="<<require_tensor(*model,"model.output.weight").nbytes<<'\n';
    int checks=0;
    for(int rows:{1,2,3,4,5,6}) {
        auto x=tb::tensor(fb::input(rows,int(head.neuron_len()))).reshape({1,rows,head.neuron_len()}).to(tb::kCUDA,tb::kFloat16);
        for(const char* variant:{"0","1","2","0"}) {
            option("MFQ_NINT_Q8_DECODE",variant);
            MfqCudaGraph graph;mfq_prepare_cuda_graph_memory(graph);
            auto y=head.forward(execution,x);fb::check(cudaStreamSynchronize(mfq_current_cuda_stream()));y={};
            graph.capture_begin();y=head.forward(execution,x);graph.capture_end();
            for(int step=0;step<3;++step) {
                auto input=fb::input(rows,int(head.neuron_len()));
                for(size_t i=0;i<input.size();++i)input[i]=step==0?0.f:input[i]*float(step*step)+std::sin(float(i)*.019f+step)*.21f;
                x.copy_(tb::tensor(input).reshape(x.sizes()).to(tb::kCUDA,tb::kFloat16));
                option("MFQ_NINT_Q8_DECODE","0");auto reference=head.forward(execution,x);
                graph.replay();compare(y,reference,variant[0]!='2');++checks;
            }
            if(!verify_only) {
                const auto samples=fb::measure([&]{graph.replay();},stream.stream(),1);
                std::cout<<std::setprecision(10)<<"{\"stage\":\"head\",\"M\":"<<rows<<",\"variant\":"<<variant
                    <<",\"us\":"<<fb::median(samples)<<",\"samples_us\":[";
                for(size_t i=0;i<samples.size();++i)std::cout<<(i?",":"")<<samples[i];
                std::cout<<"]}\n"<<std::flush;
            }
        }
    }
    std::cout<<"VERIFY_HEAD checks="<<checks<<" PASS\n"<<std::flush;
    return 0;
}catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
