#include "models/qwen4_exp/model.h"
#include "runtime/decode_window.h"
#include "mfq_cuda_context.h"
#include "cuda_execution.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <stdexcept>
namespace tb=mfq_tensor_backend;
using tb::Tensor;
using namespace mfq::cuda;
using namespace mfq::cuda::qwen4_exp;
namespace {
Tensor data(std::vector<int64_t> shape,int salt,float scale=.03f) {
    int64_t n=1;for(auto dim:shape)n*=dim;std::vector<float> values(n);
    for(int64_t i=0;i<n;++i)values[i]=std::sin(float(i)*.173f+salt)*scale;
    return tb::tensor(values).reshape(shape).to(tb::kFloat16).to(tb::kCUDA);
}
Linear linear(int in,int out,int salt) {
    auto w=data({out,in},salt);
    return [w](CudaExecutionContext&,const Tensor& x){return tb::matmul(x.to(w.scalar_type()),w.transpose(-1,-2));};
}
void equal(const Tensor& a,const Tensor& b,const char* label,float tolerance=2e-4f,float relative=5e-3f) {
    auto x=a.contiguous().to(tb::kCPU).to(tb::kFloat32),y=b.contiguous().to(tb::kCPU).to(tb::kFloat32);
    if(x.sizes()!=y.sizes())throw std::runtime_error(std::string(label)+" shape mismatch");
    for(int64_t i=0;i<x.numel();++i)if(!std::isfinite(x.data_ptr<float>()[i]) ||
        std::abs(x.data_ptr<float>()[i]-y.data_ptr<float>()[i])>tolerance+relative*std::abs(y.data_ptr<float>()[i]))
        throw std::runtime_error(std::string(label)+" differs at "+std::to_string(i)+" actual="+std::to_string(x.data_ptr<float>()[i])+" expected="+std::to_string(y.data_ptr<float>()[i]));
}
struct Cell {
    Tensor x,pos,output;
    DecodeWindow window;
    Cell(int t,int width):window(mfq_current_cuda_stream()) {
        x=data({1,t,width},0);pos=tb::zeros({t},x.options().dtype(tb::kInt64));
    }
};
void qsa_case(int width,int heads,int maximum,int budget=8) {
    CudaExecutionContext execution;constexpr int hidden=32,kv=2,iw=8,ih=2,pool=4;
    auto rotary=std::make_shared<RotaryEmbedding>(std::min(width,8),maximum,10000.);
    auto norm=data({width},1,0),inorm=data({iw},1,0);
    QsaWeights weights{linear(hidden,heads*2*width,1),linear(hidden,kv*width,2),linear(hidden,kv*width,3),
        linear(heads*width,hidden,4),linear(hidden,(ih+1)*iw,5),norm,norm,inorm,inorm};
    QsaConfig cfg{heads,kv,width,ih,iw,pool,budget,maximum,1e-6};
    Qsa actual(weights,cfg,rotary),oracle(weights,cfg,rotary);
    auto inputs=data({1,maximum,hidden},17,.4f);
    auto history=tb::arange(6,inputs.options().dtype(tb::kInt64));
    actual.forward(execution,inputs.narrow(1,0,6),history,history,true);
    oracle.forward(execution,inputs.narrow(1,0,6),history,history,true);
    actual.prepare_graph(history);
    std::map<int,std::unique_ptr<Cell>> cells;
    for(int t:{1,3,8,1,3,1,8,1,3,3,1}) {
        const auto pos=actual.position();if(pos+t>maximum)continue;
        auto positions=tb::arange(pos,pos+t,history.options());
        auto& cell=cells[t];if(!cell)cell=std::make_unique<Cell>(t,hidden);
        cell->x.copy_(inputs.narrow(1,pos,t));cell->pos.copy_(positions);
        cell->window.capture([&]{cell->output=actual.forward_graph(execution,cell->x,cell->pos,cell->pos);},[]{},[&]{cell->output={};});
        cell->window.run();actual.advance_graph(t);
        history=tb::cat({history,positions},-1);
        auto expected=oracle.forward(execution,inputs.narrow(1,pos,t),positions,history,true);
        std::cout<<"QSA width="<<width<<" position="<<pos<<" tokens="<<t<<" actual0="<<float(cell->output.to(tb::kCPU).data_ptr<mfq_half>()[0])<<" expected0="<<float(expected.to(tb::kCPU).data_ptr<mfq_half>()[0])<<std::endl;
        equal(cell->output,expected,"QSA changing-position graph");
    }
    actual.truncate(13);oracle.truncate(13);history=history.narrow(-1,0,13);
    auto& cell=cells[3];cell->x.copy_(inputs.narrow(1,13,3));cell->pos.copy_(tb::arange(13,16,history.options()));
    cell->window.run();actual.advance_graph(3);history=tb::cat({history,cell->pos},-1);
    equal(cell->output,oracle.forward(execution,inputs.narrow(1,13,3),cell->pos,history,true),"QSA rollback replay");
    cells.clear();actual.leave_graph();
    auto pos=tb::arange(16,17,history.options());history=tb::cat({history,pos},-1);
    equal(actual.forward(execution,inputs.narrow(1,16,1),pos,history,true),
        oracle.forward(execution,inputs.narrow(1,16,1),pos,history,true),"QSA graph to ordinary cache");
}
void layer_profile_case() {
    auto input=data({1,1,2560},29);auto output=tb::empty_like(input);
    DecodeWindow window(mfq_current_cuda_stream(),true);
    window.capture([&] {
        window.mark(-1,"begin");window.mark(0,"copy_begin");output.copy_(input);window.mark(0,"copy_end");
    },[]{});
    for(int replay=0;replay<3;++replay) {
        input.copy_(data({1,1,2560},replay+30));window.run();equal(output,input,"profiled replay output",0);
    }
    const auto phases=window.gpu_phases();
    if(phases.size()!=2 || phases[0].from_layer!=-1 || phases[1].to!="copy_end")
        throw std::runtime_error("GPU profile phase order changed in capture");
    for(const auto& phase:phases)if(phase.samples!=3 || phase.min_ns<0 || phase.max_ns<phase.min_ns || phase.total_ns<phase.max_ns)
        throw std::runtime_error("GPU profile lost captured replay timing events");
}
void grouped_norm_case() {
    constexpr int hidden=2560,streams=4;
    for(auto dtype:{tb::kFloat16,tb::kFloat32}) {
        auto input=data({2,3,hidden*streams},91,.4f).to(dtype);
        auto weight=data({hidden*streams},92,.3f).to(tb::kFloat32);
        auto shape=std::vector<int64_t>{2,3,streams,hidden};
        auto source=input.to(tb::kFloat32).reshape(shape);
        auto expected=(source*tb::rsqrt((source*source).mean(-1,true)+1e-6)).reshape(input.sizes());
        expected=(expected*(1.0+weight)).to(dtype);
        equal(mfq_qwen4_exp::grouped_rms_norm(input,weight,hidden,1e-6),expected,"four-stream grouped RMSNorm",0,0);
        auto strided=input.transpose(0,1);
        equal(mfq_qwen4_exp::grouped_rms_norm(strided,weight,hidden,1e-6),expected.transpose(0,1),"strided grouped RMSNorm");
    }
}
void gated_residual_case() {
    constexpr int hidden=2560,streams=4;
    for(auto dtype:{tb::kFloat16,tb::kFloat32,tb::kBFloat16})for(int width:{4,320}) {
        auto projection=data({2,3,width},103,88.f).to(dtype);
        auto low=projection/streams;
        auto activated=low*tb::sigmoid(low),injected=2.0*tb::sigmoid(low);
        equal(mfq_qwen4_exp::gated_residual_bottleneck(projection,streams),activated,"GR bottleneck activation",0,0);
        equal(mfq_qwen4_exp::gated_residual_injection(projection,streams),injected,"GR injection activation",0,0);
        auto strided=projection.transpose(0,1),strided_low=strided/streams;
        equal(mfq_qwen4_exp::gated_residual_bottleneck(strided,streams),strided_low*tb::sigmoid(strided_low),
            "strided GR bottleneck activation",0,0);
        equal(mfq_qwen4_exp::gated_residual_injection(strided,streams),2.0*tb::sigmoid(strided_low),
            "strided GR injection activation",0,0);
    }
    for(auto value_dtype:{tb::kFloat16,tb::kFloat32})for(auto gate_dtype:{tb::kFloat16,tb::kFloat32}) {
        auto normalized=data({2,3,hidden*streams},97,.4f).to(value_dtype);
        auto gate=data({2,3,hidden*streams},98,7.f).to(gate_dtype);
        auto shape=std::vector<int64_t>{2,3,streams,hidden};
        auto expected=(tb::sigmoid(gate).reshape(shape)*normalized.reshape(shape)).mean(-2);
        const bool fp32=expected.scalar_type()==tb::kFloat32;
        equal(mfq_qwen4_exp::gated_residual_mix(gate,normalized,streams),expected,"GR gate/mix/reduce",
            0,0);
        equal(mfq_qwen4_exp::gated_residual_mix(gate.transpose(0,1),normalized.transpose(0,1),streams),
            expected.transpose(0,1),"strided GR mix",fp32?2e-6f:2e-4f,fp32?2e-6f:5e-3f);
    }
    for(auto dtype:{tb::kFloat16,tb::kFloat32}) {
        auto branch=data({2,3,hidden},99,.4f).to(dtype);
        auto residual=data({2,3,hidden*streams},100,.4f).to(dtype);
        auto injection=data({2,3,streams},101,1.3f).to(dtype);
        auto expected=residual+(branch.unsqueeze(-2)*injection.unsqueeze(-1)).reshape(residual.sizes());
        equal(mfq_qwen4_exp::gated_residual_post(branch,residual,injection,streams),expected,"GR residual injection",0,0);
        equal(mfq_qwen4_exp::gated_residual_post(branch.transpose(0,1),residual.transpose(0,1),injection.transpose(0,1),streams),
            expected.transpose(0,1),"strided GR injection",0,0);
    }
}
Tensor composed_gdn_output(const Tensor& attended,const Tensor& projection,const Tensor& weight,
    double eps,bool silu_gate,bool output_half) {
    const auto batch=attended.size(0),heads=attended.size(1),tokens=attended.size(2),width=attended.size(3);
    auto z=projection.reshape({batch,tokens,heads,width}).permute({0,2,1,3}).to(tb::kFloat32);
    auto gate=tb::sigmoid(z);if(silu_gate)gate=z*gate;
    auto normalized=attention_ops::rms_norm(attended,weight,eps)*gate;
    return normalized.permute({0,2,1,3}).reshape({batch,tokens,heads*width})
        .to(output_half?tb::kFloat16:tb::kFloat32);
}
void gdn_output_case() {
    int cases=0;
    for(int width:{17,32,64,128,129,320})for(int tokens:{1,3,8,35})
        for(auto gate_dtype:{tb::kFloat16,tb::kFloat32})for(bool silu:{false,true})for(bool half:{false,true}) {
        const int batch=tokens==3?2:1,heads=48;
        auto attended=data({batch,heads,tokens,width},132,.17f).to(tb::kFloat32);
        auto projection=data({batch,tokens,heads*width},133,88.f).to(gate_dtype);
        auto weight=data({width},134,1.3f).to(tb::kFloat32);
        const double eps=half?1e-6:0.013456789123;
        auto output=gdn_rms_norm_gate_cuda(attended,projection,weight,eps,silu,half);
        equal(output,composed_gdn_output(attended,projection,weight,eps,silu,half),"GDN fused RMSNorm/gate/layout",0,0);
        if(output.scalar_type()!=(half?tb::kFloat16:tb::kFloat32) || !output.is_contiguous())
            throw std::runtime_error("GDN fused output dtype/layout changed");
        ++cases;
    }
    for(int tokens:{1,3,8})for(auto dtype:{tb::kFloat16,tb::kFloat32}) {
        auto attended=data({1,48,tokens,128},141,.00013f).to(tb::kFloat32);
        auto projection=data({1,tokens,48*128},142,14.f).to(dtype);
        auto weight=data({128},143,1.1f).to(tb::kFloat32);
        Tensor output;DecodeWindow window(mfq_current_cuda_stream());
        window.capture([&]{output=gdn_rms_norm_gate_cuda(attended,projection,weight,1e-6,true,true);},[]{},[&]{output={};});
        for(int replay=0;replay<3;++replay) {
            attended.copy_(replay==1?tb::zeros_like(attended):data(attended.sizes().vec(),144+replay,replay==2?.17f:.00013f).to(tb::kFloat32));
            projection.copy_(data(projection.sizes().vec(),147+replay,88.f).to(dtype));
            window.run();equal(output,composed_gdn_output(attended,projection,weight,1e-6,true,true),
                "GDN changing-input fused output graph",0,0);++cases;
        }
    }
    std::cout<<"GDN fused output composed exact cases="<<cases<<" PASS\n";
}
void gdn_case(bool model_shape = false) {
    CudaExecutionContext execution;
    const int hidden=model_shape?2560:32,d=model_shape?128:32,nk=model_shape?16:1,nv=model_shape?48:2;
    const int channels=(2*nk+nv)*d;
    auto gpu=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat32);
    GdnWeights weights{linear(hidden,channels,1),linear(hidden,nv*d,2),linear(hidden,nv,3),linear(hidden,nv,4),
        linear(nv*d,hidden,5),data({channels,4},6),tb::zeros({nv},gpu),tb::zeros({nv},gpu),tb::ones({d},gpu)};
    Gdn actual(weights,nk,nv,d,4,1e-6,false),oracle(weights,nk,nv,d,4,1e-6,false,false);
    Gdn row_reference(weights,nk,nv,d,4,1e-6,false,true,false,false,true,false);
    const int prefix=model_shape?2:5;
    auto input=data({1,model_shape?5:32,hidden},12,.5f);
    actual.forward_chunk(execution,input.narrow(1,0,prefix),true);oracle.forward_chunk(execution,input.narrow(1,0,prefix),true);
    row_reference.forward_chunk(execution,input.narrow(1,0,prefix),true);
    equal(actual.recurrent_state(),row_reference.recurrent_state(),"GDN prefill transposed state",0,0);
    const auto* conv=actual.conv_state().data_ptr();const auto* recurrent=actual.recurrent_state().data_ptr();
    std::map<int,std::unique_ptr<Cell>> cells;int position=prefix;
    const std::vector<int> lengths=model_shape?std::vector<int>{1,1,1}:std::vector<int>{1,3,8,1,3,1,3};
    for(int t:lengths) {
        auto& cell=cells[t];if(!cell)cell=std::make_unique<Cell>(t,hidden);cell->x.copy_(input.narrow(1,position,t));
        if(!cell->window.valid()) {
            auto saved_conv=actual.conv_state().clone(),saved_state=actual.graph_state()[1]->clone();
            cell->window.capture([&]{cell->output=actual.forward_chunk(execution,cell->x,true);},[&]{
                actual.graph_state()[0]->copy_(saved_conv);actual.graph_state()[1]->copy_(saved_state);
            },[&]{cell->output={};});
        }
        cell->window.run();equal(cell->output,oracle.forward_chunk(execution,input.narrow(1,position,t),true),"GDN changing-input graph");
        equal(cell->output,row_reference.forward_chunk(execution,input.narrow(1,position,t),true),"GDN transposed recurrence output",0,0);
        equal(actual.recurrent_state(),row_reference.recurrent_state(),"GDN transposed recurrence state",0,0);
        equal(actual.conv_state(),oracle.conv_state(),"GDN convolution state");
        equal(actual.recurrent_state(),oracle.recurrent_state(),"GDN recurrent state");
        if(actual.conv_state().data_ptr()!=conv || actual.recurrent_state().data_ptr()!=recurrent)
            throw std::runtime_error("GDN graph changed recurrent addresses");
        position+=t;
    }
}

void gdn_preparation_case() {
    int ordinary=0,changing=0;
    const auto bits=[](const Tensor& a,const Tensor& b,const char* label) {
        auto x=a.cpu().contiguous(),y=b.cpu().contiguous();
        if(x.sizes()!=y.sizes() || x.scalar_type()!=y.scalar_type() ||
            std::memcmp(x.data_ptr(),y.data_ptr(),x.numel()*x.element_size()))
            throw std::runtime_error(std::string(label)+" dtype/shape/bytes differ");
    };
    for(int width:{32,64,128,256})for(int batch:{1,2})for(bool channel_first:{false,true})
    for(auto gate_dtype:{tb::kFloat16,tb::kFloat32}) {
        const int nk=width==128?16:2,nv=nk*3,dv=width==32?17:width,K=width==64?7:4,C=2*nk*width+nv*dv;
        auto qk=data({batch,1,2*nk*width},311,.31f),v=data({batch,1,nv*dv},312,.23f);
        auto alpha=data({batch,1,nv*2},313,88.f).to(gate_dtype).narrow(-1,0,nv);
        auto beta=data({batch,1,nv*2},314,88.f).to(gate_dtype).narrow(-1,0,nv);
        auto dt=data({nv},315,.31f).to(tb::kFloat32),a=data({nv},316,.37f).to(tb::kFloat32);
        auto weight=data({C,K},317,.09f).to(tb::kFloat32);
        weight=channel_first?weight.reshape({C,1,K}):weight.transpose(0,1).contiguous();
        auto state=data({batch,K-1,C},318,.13f).to(tb::kFloat32),reference_state=state.clone();
        const auto separate=[&] {
            auto qkv=linear_conv_qkv_decode_cuda(reference_state,qk,v,weight,tb::empty({0},dt.options()),nk,nv,width,dv,1e-6);
            auto gates=linear_gate_beta_cuda(alpha.to(tb::kFloat32),beta.to(tb::kFloat32),dt,a,true);
            qkv.push_back(gates[0]);qkv.push_back(gates[1]);return qkv;
        };
        const auto combined=[&]{return linear_conv_qkv_gate_decode_cuda(state,qk,v,weight,alpha,beta,dt,a,nk,nv,width,dv,1e-6);};
        auto expected=separate(),actual=combined();
        for(int i=0;i<5;++i)bits(actual[i],expected[i],"GDN combined preparation ordinary");
        bits(state,reference_state,"GDN combined convolution state");++ordinary;
        std::vector<Tensor> output;DecodeWindow window(mfq_current_cuda_stream());auto saved=state.clone();
        window.capture([&]{output=combined();},[&]{state.copy_(saved);},[&]{output.clear();});
        for(float amplitude:{0.f,.13f,8.f}) {
            qk.copy_(data(qk.sizes().vec(),321,amplitude));v.copy_(data(v.sizes().vec(),322,amplitude));
            alpha.copy_(data(alpha.sizes().vec(),323,amplitude*11).to(gate_dtype));
            beta.copy_(data(beta.sizes().vec(),324,amplitude*11).to(gate_dtype));
            reference_state.copy_(state);expected=separate();window.run();
            for(int i=0;i<5;++i)bits(output[i],expected[i],"GDN combined preparation changing graph");
            bits(state,reference_state,"GDN changing convolution state");++changing;
        }
    }
    std::cout<<"GDN combined preparation ordinary="<<ordinary<<" changing_graph="<<changing
        <<" all5outputs/state bytes exact, Half/Float strided gates, batch1/2, layouts2, widths32/64/128/256 PASS\n";
}
void gdn_core_case() {
    int ordinary=0,changing=0;
    const auto bits=[](const Tensor& a,const Tensor& b,const char* label) {
        auto x=a.contiguous().cpu(),y=b.contiguous().cpu();
        if(x.sizes()!=y.sizes() || x.scalar_type()!=y.scalar_type() ||
            std::memcmp(x.data_ptr(),y.data_ptr(),x.numel()*x.element_size()))
            throw std::runtime_error(std::string(label)+" dtype/shape/bytes differ");
    };
    for(int width:{32,64,128})for(int combination=0;combination<16;++combination) {
        const int batch=1+(combination&1),nk=combination&2?16:2,nv=nk*3,taps=combination&4?7:4;
        const int channels=(2*nk+nv)*width;
        const auto dtype=combination&8?tb::kFloat32:tb::kFloat16;
        const bool half=combination&1,silu=combination&2,transposed=combination&4;
        const double eps=half?1e-6:.013456789123;
        auto qkv=data({batch,1,channels},151,.4f);
        auto gate=data({batch,1,nv*width},152,88.f).to(dtype);
        auto alpha=data({batch,1,nv*2},153,88.f).to(dtype).slice(-1,0,2*nv,2);
        auto beta=data({batch,1,nv*2},154,88.f).to(dtype).slice(-1,0,2*nv,2);
        auto conv=data({batch,taps-1,channels},155,.3f).to(tb::kFloat32);
        auto state=data({batch,nv,width,width},156,.17f).to(tb::kFloat32);
        auto weight=data(combination&1?std::vector<int64_t>{taps,channels}:std::vector<int64_t>{channels,1,taps},157,.19f).to(tb::kFloat32);
        auto bias=data({nv},158,1.3f).to(tb::kFloat32),a=data({nv},159,.3f).to(tb::kFloat32);
        auto norm=data({width},160,1.3f).to(tb::kFloat32);
        auto workspace=tb::zeros({batch,nv,width+1},conv.options());
        auto expected_counter=tb::zeros({batch,nv},conv.options());
        const auto reference=[&] {
            auto next=conv.clone();
            auto prepared=linear_conv_qkv_gate_decode_cuda(next,qkv.narrow(-1,0,2*nk*width).contiguous(),
                qkv.narrow(-1,2*nk*width,nv*width).contiguous(),weight,alpha,beta,bias,a,nk,nv,width,width,1e-6);
            auto result=(transposed?gdn_transposed_cuda:gdn_cuda)(prepared[0],prepared[1],prepared[2],prepared[3],prepared[4],state);
            return std::vector<Tensor>{gdn_rms_norm_gate_cuda(result[0],gate,norm,eps,silu,half),next,result[1]};
        };
        const auto fused=[&] {return gdn_decode_core_cuda(qkv,gate,alpha,beta,conv,state,weight,bias,a,norm,
            nk,nv,width,1e-6,eps,silu,half,transposed,workspace);};
        const auto check=[&](const std::vector<Tensor>& result) {
            auto expected=reference();for(size_t i=0;i<3;++i)bits(result[i],expected[i],"GDN complete core exact");
            bits(workspace.select(-1,width),expected_counter,"GDN reusable counter reset");
        };
        check(fused());++ordinary;
        std::vector<Tensor> result;DecodeWindow window(mfq_current_cuda_stream());
        window.capture([&]{result=fused();},[]{},[&]{result.clear();});
        for(int step=0;step<3;++step) {
            const float amplitude=step==0?0.f:step==1?.13f:8.f;
            qkv.copy_(data(qkv.sizes().vec(),161+step,amplitude));
            conv.copy_(data(conv.sizes().vec(),164+step,amplitude).to(tb::kFloat32));
            state.copy_(data(state.sizes().vec(),167+step,amplitude).to(tb::kFloat32));
            window.run();check(result);++changing;
        }
    }
    std::cout<<"GDN complete core ordinary="<<ordinary<<" changing_graph="<<changing
        <<" output/conv/FP32state bytes exact; reused counters reset; widths32/64/128, batch1/2, layouts2, strided Half/Float gates PASS\n";
}
}
int main()try {
    const auto stream=mfq_current_cuda_stream();auto context=default_context(mfq_current_cuda_device());context->begin_graph_pool(stream);
    layer_profile_case();grouped_norm_case();gated_residual_case();gdn_output_case();gdn_preparation_case();gdn_core_case();qsa_case(8,4,40);qsa_case(256,24,41);qsa_case(256,24,512,2048);gdn_case();gdn_case(true);
    MFQ_CUDA_CHECK(cudaStreamSynchronize(stream));context->end_graph_pool(stream);
    std::cout<<"MFQ window QSA/GDN changing-position, rollback and fixed-state checks PASS\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
