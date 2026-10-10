#include "models/qwen4_exp/model.h"
#include "runtime/decode_window.h"
#include "mfq_cuda_context.h"
#include "cuda_execution.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <stdexcept>
#include <cstdlib>
namespace tb=mfq_tensor_backend;
using tb::Tensor;
using namespace mfq::cuda;
using namespace mfq::cuda::qwen4_exp;
namespace {
class GdnTileScope {
    std::string previous_;
    bool present_;
    static void set(const char* value) {
#ifdef _WIN32
        if(_putenv_s("MFQ_GDN_DECODE_TILES",value)!=0)throw std::runtime_error("cannot set GDN tile option");
#else
        const int result=*value?setenv("MFQ_GDN_DECODE_TILES",value,1):unsetenv("MFQ_GDN_DECODE_TILES");
        if(result!=0)throw std::runtime_error("cannot set GDN tile option");
#endif
    }
public:
    explicit GdnTileScope(int tiles):present_(std::getenv("MFQ_GDN_DECODE_TILES")!=nullptr) {
        if(present_)previous_=std::getenv("MFQ_GDN_DECODE_TILES");
        set(std::to_string(tiles).c_str());
    }
    ~GdnTileScope(){
        try{set(present_?previous_.c_str():"");}catch(...){}
    }
};
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
void segmented_sort_graph_case() {
    int ordinary=0,changing=0;
    for(const auto shape:std::vector<std::vector<int64_t>>{{1,1024},{3,2051},{256,1}})
    for(bool descending:{false,true}) {
        const auto rows=shape[0],columns=shape[1];
        const auto make_values=[&](int salt) {
            std::vector<float> values(rows*columns);
            for(int64_t row=0;row<rows;++row)for(int64_t column=0;column<columns;++column)
                values[row*columns+column]=float((column*37+row*13+salt*11)%23-11)*.125f;
            return values;
        };
        auto input=tb::tensor(make_values(0)).reshape(shape).to(tb::kCUDA);
        const auto check=[&](const Tensor& keys,const Tensor& indices,int salt) {
            const auto values=make_values(salt);
            auto actual_keys=keys.contiguous().cpu(),actual_indices=indices.contiguous().cpu();
            if(actual_keys.sizes()!=input.sizes() || actual_indices.sizes()!=input.sizes() ||
                actual_keys.scalar_type()!=tb::kFloat32 || actual_indices.scalar_type()!=tb::kInt64)
                throw std::runtime_error("segmented sort output geometry differs");
            for(int64_t row=0;row<rows;++row) {
                std::vector<int64_t> order(columns);
                for(int64_t column=0;column<columns;++column)order[column]=column;
                std::stable_sort(order.begin(),order.end(),[&](int64_t left,int64_t right) {
                    return descending?values[row*columns+left]>values[row*columns+right]:
                        values[row*columns+left]<values[row*columns+right];
                });
                for(int64_t column=0;column<columns;++column) {
                    const auto offset=row*columns+column;
                    if(actual_indices.data_ptr<int64_t>()[offset]!=order[column] ||
                        actual_keys.data_ptr<float>()[offset]!=values[row*columns+order[column]])
                        throw std::runtime_error("segmented sort stable values/indices differ");
                }
            }
        };
        auto result=tb::sort(input,-1,descending);
        check(std::get<0>(result),std::get<1>(result),0);++ordinary;
        Tensor keys,indices;DecodeWindow window(mfq_current_cuda_stream());
        window.capture([&]{auto sorted=tb::sort(input,-1,descending);
            keys=std::get<0>(sorted);indices=std::get<1>(sorted);},[]{},[&]{keys={};indices={};});
        for(int salt:{1,2,3}) {
            input.copy_(tb::tensor(make_values(salt)).reshape(shape).to(tb::kCUDA));
            window.run();check(keys,indices,salt);++changing;
        }
    }
    std::cout<<"Native segmented sort ordinary="<<ordinary<<" changing_graph="<<changing
        <<" stable values/indices exact; ties, multiple rows, one-column final offset PASS\n";
}
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
void gdn_prefill_case() {
    int cases=0;
    for(int width:{32,64,128})for(bool model_shape:{false,true}) {
        if(model_shape && width!=128)continue;
        const int hidden=model_shape?2560:96,nk=model_shape?16:2,nv=3*nk,channels=(2*nk+nv)*width;
        auto gpu=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat32);
        GdnWeights weights{linear(hidden,channels,331),linear(hidden,nv*width,332),
            linear(hidden,nv,333),linear(hidden,nv,334),linear(nv*width,hidden,335),
            data({channels,4},336).to(tb::kFloat32),tb::zeros({nv},gpu),tb::zeros({nv},gpu),tb::ones({width},gpu)};
        Gdn actual(weights,nk,nv,width,4,1e-6,true,true,true,true,false,true),
            oracle(weights,nk,nv,width,4,1e-6,true,false,false,false,false,false);
        CudaExecutionContext execution;
        for(int tokens:{31,256,1024,1,257}) {
            auto input=data({1,tokens,hidden},337+tokens,.13f);
            auto expected=oracle.forward_chunk(execution,input,true);
            auto result=actual.forward_chunk(execution,input,true);
            equal(result,expected,"GDN fused prefill output");
            equal(actual.conv_state(),oracle.conv_state(),"GDN fused prefill convolution state",0,0);
            equal(actual.recurrent_state(),oracle.recurrent_state(),"GDN fused prefill recurrent state");
            ++cases;
        }
    }
    std::cout<<"GDN fused prefill cases="<<cases<<" widths32/64/128, model heads16/48, continued state and prefill/decode transition PASS\n";
}
void gdn_core_case(bool inplace=false) {
    int ordinary=0,changing=0;
    const auto bits=[](const Tensor& a,const Tensor& b,const char* label) {
        auto x=a.contiguous().cpu(),y=b.contiguous().cpu();
        if(x.sizes()!=y.sizes() || x.scalar_type()!=y.scalar_type() ||
            std::memcmp(x.data_ptr(),y.data_ptr(),x.numel()*x.element_size()))
            throw std::runtime_error(std::string(label)+" dtype/shape/bytes differ");
    };
    for(int width:{32,64,128})for(int combination=0;combination<32;++combination) {
        const int batch=1+(combination&1),nk=combination&2?16:2,nv=nk*3,taps=combination&16?7:4;
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
        const auto fused=[&] {
            auto input_state=inplace?state.clone():state;
            auto result=gdn_decode_core_cuda(qkv,gate,alpha,beta,conv,input_state,weight,bias,a,norm,
                nk,nv,width,1e-6,eps,silu,half,transposed,workspace,inplace);
            if(inplace && result[2].data_ptr()!=input_state.data_ptr())
                throw std::runtime_error("GDN inplace state storage was replaced");
            return result;
        };
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
    std::cout<<"GDN complete core inplace="<<inplace<<" ordinary="<<ordinary<<" changing_graph="<<changing
        <<" output/conv/FP32state bytes exact; reused counters reset; widths32/64/128, batch1/2, layouts2, strided Half/Float gates PASS\n";
}
void gdn_core_benchmark(bool compare_geometry=false,bool compare_inplace=false) {
    constexpr int nk=16,nv=48,width=128,channels=(2*nk+nv)*width,taps=4,replays=1024;
    auto qkv=data({1,1,channels},151,.4f),gate=data({1,1,nv*width},152,.5f);
    auto alpha=data({1,1,nv},153,.5f),beta=data({1,1,nv},154,.5f);
    auto conv=data({1,taps-1,channels},155,.3f).to(tb::kFloat32);
    auto state=data({1,nv,width,width},156,.17f).to(tb::kFloat32);
    auto weight=data({channels,1,taps},157,.19f).to(tb::kFloat32);
    auto bias=data({nv},158,1.3f).to(tb::kFloat32),a=data({nv},159,.3f).to(tb::kFloat32);
    auto norm=data({width},160,1.3f).to(tb::kFloat32);
    auto workspace=tb::zeros({1,nv,width+1},conv.options());
    const auto initial_state=state.clone(),initial_conv=conv.clone();
    const auto stream=mfq_current_cuda_stream();
    const std::vector<int> tile_cases=(compare_geometry || compare_inplace)?std::vector<int>{4,4,4,4,4,4}:std::vector<int>{4,2,1,1,2,4};
    int arm=0;
    for(int tiles:tile_cases) {
        const char* fixed_setting=std::getenv("MFQ_GDN_FIXED_GEOMETRY");
        const bool fixed=compare_geometry?arm%3==1:(!fixed_setting || fixed_setting[0]!='0');
        const bool inplace=compare_inplace && arm%3==1;
        if(compare_inplace){state.copy_(initial_state);conv.copy_(initial_conv);}
        if(compare_geometry) {
#ifdef _WIN32
            _putenv_s("MFQ_GDN_FIXED_GEOMETRY",fixed?"1":"0");
#else
            setenv("MFQ_GDN_FIXED_GEOMETRY",fixed?"1":"0",1);
#endif
        }
        ++arm;
        GdnTileScope setting(tiles);std::vector<Tensor> result;
        const auto compute=[&]{result=gdn_decode_core_cuda(qkv,gate,alpha,beta,conv,state,weight,bias,a,norm,
            nk,nv,width,1e-6,1e-6,false,true,true,workspace,inplace);
            if(compare_inplace) {
                conv.copy_(result[1]);
                if(!inplace)state.copy_(result[2]);
            }
        };
        {
            GraphWarmupScope warmup(default_context(mfq_current_cuda_device()),stream);compute();
        }
        MFQ_CUDA_CHECK(cudaStreamSynchronize(stream));result.clear();
        mfq::cuda::Graph graph;graph.capture_begin_shared();compute();graph.capture_end();
        for(int i=0;i<128;++i)graph.replay();
        MFQ_CUDA_CHECK(cudaStreamSynchronize(stream));
        cudaEvent_t begin=nullptr,end=nullptr;
        MFQ_CUDA_CHECK(cudaEventCreate(&begin));MFQ_CUDA_CHECK(cudaEventCreate(&end));
        MFQ_CUDA_CHECK(cudaEventRecord(begin,stream));
        for(int i=0;i<replays;++i)graph.replay();
        MFQ_CUDA_CHECK(cudaEventRecord(end,stream));MFQ_CUDA_CHECK(cudaEventSynchronize(end));
        float milliseconds=0;MFQ_CUDA_CHECK(cudaEventElapsedTime(&milliseconds,begin,end));
        MFQ_CUDA_CHECK(cudaEventDestroy(begin));MFQ_CUDA_CHECK(cudaEventDestroy(end));
        const double microseconds=double(milliseconds)*1000/replays;
        const double state_bytes=double(state.numel()*state.element_size())*2;
        std::cout<<"gdn_core_bench tiles="<<tiles<<" batch=1 key_heads="<<nk<<" value_heads="<<nv
            <<" width="<<width<<" transposed_state=1 fixed_geometry="<<fixed<<" inplace_state="<<inplace<<" us="<<microseconds
            <<" state_effective_GBps="<<state_bytes/(microseconds*1000)<<" replays="<<replays
            <<" benchmark_mode=queued_graph\n";
    }
}
Tensor composed_qsa_selection(const Tensor& blocks,const Tensor& positions,int64_t pool,
    int64_t budget,int64_t pools,const Tensor& existing_visible=Tensor{}) {
    const auto batch=blocks.size(0),tokens=blocks.size(1),count=blocks.size(2),columns=budget+pool-1;
    auto indices=blocks.to(tb::kInt64),absolute=positions.reshape({1,tokens,1});
    auto visible=existing_visible;
    if(!visible.defined()) {
        auto ends=(tb::arange(pools,positions.options())*pool+pool-1).reshape({1,1,pools});
        visible=ends<=absolute;
    }
    auto selected=tb::full({batch,tokens,columns},-1,positions.options());
    if(count) {
        auto valid=visible.expand({batch,tokens,pools}).gather(-1,indices).unsqueeze(-1).expand({batch,tokens,count,pool});
        auto expanded=indices.unsqueeze(-1)*pool+tb::arange(pool,indices.options());
        expanded=tb::where(valid,expanded,tb::full_like(expanded,-1));
        selected.narrow(-1,0,count*pool).copy_(expanded.reshape({batch,tokens,count*pool}));
    }
    if(pool>1) {
        auto tail_count=(absolute+1).remainder(pool);
        auto offsets=tb::arange(pool-1,positions.options()).reshape({1,1,pool-1});
        auto tail=absolute+1-tail_count+offsets;
        selected.narrow(-1,budget,pool-1).copy_(tb::where(offsets<tail_count,tail,tb::full_like(tail,-1))
            .expand({batch,tokens,pool-1}));
    }
    auto dense=tb::arange(columns,positions.options()).reshape({1,1,columns}).expand({batch,tokens,columns});
    dense=tb::where(dense<=absolute,dense,tb::full_like(dense,-1));
    return tb::where(absolute+1<=budget,dense,selected).to(tb::kInt32).contiguous();
}
void qsa_selection_case() {
    int ordinary=0,changing=0;
    const auto bits=[](const Tensor& actual,const Tensor& expected) {
        auto x=actual.contiguous().cpu(),y=expected.contiguous().cpu();
        if(x.sizes()!=y.sizes() || x.scalar_type()!=y.scalar_type() ||
            std::memcmp(x.data_ptr(),y.data_ptr(),x.numel()*x.element_size()))
            throw std::runtime_error("QSA fused token selection dtype/shape/bytes differ");
    };
    for(int pool:{1,4,16})for(int budget:{5,32,2048})for(int batch:{1,2})for(int mode=0;mode<8;++mode) {
        constexpr int tokens=7;
        const int capacity=budget+pool*7+3,pools=(capacity+pool-1)/pool,count=std::min(budget/pool,pools);
        const bool strided=mode&4;
        std::vector<int64_t> values(batch*tokens*count*(strided?2:1));
        for(size_t i=0;i<values.size();++i)values[i]=(i*3+mode)%pools;
        auto blocks=tb::tensor(values).reshape({batch,tokens,count*(strided?2:1)}).to(tb::kCUDA);
        if(!(mode&1))blocks=blocks.to(tb::kInt32);
        if(strided)blocks=blocks.slice(-1,0,count*2,2);
        const std::vector<int64_t> base_positions{0,1,budget-1,budget,budget+1,capacity-1,-1};
        std::vector<int64_t> storage(tokens*(strided?2:1));
        for(int i=0;i<tokens;++i)storage[i*(strided?2:1)]=base_positions[i];
        auto positions=tb::tensor(storage).to(tb::kCUDA);
        if(!(mode&2))positions=positions.to(tb::kInt32);
        if(strided)positions=positions.slice(0,0,tokens*2,2);
        const auto reference=[&]{return composed_qsa_selection(blocks,positions,pool,budget,pools);};
        const auto fused=[&]{return mfq_qwen4_exp::qsa_selected_tokens(blocks,positions,pool,budget);};
        try{bits(fused(),reference());}catch(const std::exception& e){
            throw std::runtime_error("QSA selection pool="+std::to_string(pool)+" budget="+std::to_string(budget)+
                " batch="+std::to_string(batch)+" mode="+std::to_string(mode)+": "+e.what());
        }
        ++ordinary;
        Tensor result;DecodeWindow window(mfq_current_cuda_stream());
        window.capture([&]{result=fused();},[]{},[&]{result={};});
        for(int replay=0;replay<3;++replay) {
            std::vector<int64_t> moved(tokens);
            for(int i=0;i<tokens;++i)moved[i]=base_positions[(i+replay+1)%tokens];
            positions.copy_(tb::tensor(moved).to(tb::kCUDA).to(positions.scalar_type()));
            window.run();bits(result,reference());++changing;
        }
    }
    std::cout<<"QSA fused token selection ordinary="<<ordinary<<" changing_graph="<<changing
        <<" bytes exact; dense prefix, sparse blocks, invisible/padded blocks, partial tail, count0, strided Int32/Int64 PASS\n";
}
void qsa_selection_benchmark() {
    constexpr int replays=256,pools=1024;
    const auto stream=mfq_current_cuda_stream();
    for(int tokens:{1,3,8})for(int pool:{4,16})for(int budget:{512,2048}) {
        const int count=budget/pool;
        std::vector<int64_t> values(tokens*count);
        for(size_t i=0;i<values.size();++i)values[i]=(i*37+5)%pools;
        auto blocks=tb::tensor(values).reshape({1,tokens,count}).to(tb::kCUDA);
        auto positions=tb::full({tokens},budget+17,blocks.options());
        auto visible=(tb::arange(pools,positions.options())*pool+pool-1).reshape({1,1,pools})
            <=positions.reshape({1,tokens,1});
        auto expected=composed_qsa_selection(blocks,positions,pool,budget,pools,visible).cpu().contiguous();
        for(bool fused:{false,true,false,false,true,false}) {
            Tensor output;const auto compute=[&]{output=fused?
                mfq_qwen4_exp::qsa_selected_tokens(blocks,positions,pool,budget):
                composed_qsa_selection(blocks,positions,pool,budget,pools,visible);};
            {
                GraphWarmupScope warmup(default_context(mfq_current_cuda_device()),stream);compute();
            }
            MFQ_CUDA_CHECK(cudaStreamSynchronize(stream));output={};
            mfq::cuda::Graph graph;graph.capture_begin_shared();compute();graph.capture_end();
            for(int i=0;i<32;++i)graph.replay();
            MFQ_CUDA_CHECK(cudaStreamSynchronize(stream));
            auto actual=output.cpu().contiguous();
            if(actual.sizes()!=expected.sizes() || actual.scalar_type()!=expected.scalar_type() ||
                std::memcmp(actual.data_ptr(),expected.data_ptr(),actual.numel()*actual.element_size()))
                throw std::runtime_error("QSA selection benchmark original bytes differ");
            cudaEvent_t begin=nullptr,end=nullptr;
            MFQ_CUDA_CHECK(cudaEventCreate(&begin));MFQ_CUDA_CHECK(cudaEventCreate(&end));
            MFQ_CUDA_CHECK(cudaEventRecord(begin,stream));
            for(int i=0;i<replays;++i)graph.replay();
            MFQ_CUDA_CHECK(cudaEventRecord(end,stream));MFQ_CUDA_CHECK(cudaEventSynchronize(end));
            float milliseconds=0;MFQ_CUDA_CHECK(cudaEventElapsedTime(&milliseconds,begin,end));
            MFQ_CUDA_CHECK(cudaEventDestroy(begin));MFQ_CUDA_CHECK(cudaEventDestroy(end));
            std::cout<<"qsa_selection_bench tokens="<<tokens<<" pool="<<pool<<" budget="<<budget
                <<" fused="<<fused<<" us="<<double(milliseconds)*1000/replays
                <<" replays="<<replays<<" scope=block_expand/visibility/tail/dense_prefix/cast queued_graph bytes_exact=1\n";
        }
    }
}
void qsa_sparse_gate_case() {
    int ordinary=0,changing=0;
    const auto bits=[](const Tensor& actual,const Tensor& expected) {
        auto x=actual.contiguous().cpu(),y=expected.contiguous().cpu();
        if(x.sizes()!=y.sizes() || x.scalar_type()!=y.scalar_type() ||
            std::memcmp(x.data_ptr(),y.data_ptr(),x.numel()*x.element_size()))
            throw std::runtime_error("QSA sparse fused gate dtype/shape/bytes differ");
    };
    for(int tokens:{1,3,8})for(int count:{1,31,32,33,127,512,2051})for(int mode=0;mode<8;++mode) {
        const int batch=tokens==3?2:1,heads=count<127 || count==2051?24:48,kv=2,width=256,capacity=257;
        const auto gate_dtype=mode&1?tb::kFloat32:tb::kFloat16;
        const bool strided=mode&2,half_output=mode&4;
        auto query=data({batch,heads,tokens,width*2},381,.3f).slice(-1,0,width*2,2);
        if(mode&1)query=query.to(tb::kFloat32);
        auto key=data({batch,kv,capacity,width},382,.17f),value=data(key.sizes().vec(),383,.19f);
        auto gate=data({batch,tokens,heads,width*(strided?2:1)},384,88.f).to(gate_dtype);
        if(strided)gate=gate.slice(-1,0,width*2,2);
        std::vector<int64_t> host_indices(batch*tokens*count);
        for(size_t i=0;i<host_indices.size();++i)
            host_indices[i]=i%7==0?-1:i%11==0?capacity+3:int64_t((i*13+i/count)%capacity);
        auto indices=tb::tensor(host_indices).reshape({batch,tokens,count}).to(tb::kCUDA);
        if(!(mode&1))indices=indices.to(tb::kInt32);
        const auto reference=[&]{return mfq_qwen4_exp::attention_gate(
            mfq_qwen4_exp::sparse_gqa_attention(query,key,value,indices),gate,half_output);};
        const auto fused=[&]{return mfq_qwen4_exp::sparse_gqa_attention_gate(
            query,key,value,indices,gate,half_output);};
        bits(fused(),reference());++ordinary;
        Tensor result;DecodeWindow window(mfq_current_cuda_stream());
        window.capture([&]{result=fused();},[]{},[&]{result={};});
        for(int replay=0;replay<3;++replay) {
            query.copy_(data(query.sizes().vec(),385+replay,replay==0?0.f:.31f).to(query.scalar_type()));
            gate.copy_(data(gate.sizes().vec(),388+replay,replay==2?0.f:88.f).to(gate_dtype));
            if(replay==0)indices.fill_(-1);
            else if(replay==1)indices.copy_(tb::tensor(host_indices).reshape(indices.sizes()).to(tb::kCUDA).to(indices.scalar_type()));
            else indices.fill_(capacity+7);
            window.run();bits(result,reference());++changing;
        }
    }
    // Preserve fallback behavior for other widths, BF16 gates and an empty KV cache.
    for(int width:{64,256})for(int capacity:{0,17}) {
        auto query=data({1,4,1,width},397,.3f),key=data({1,2,capacity,width},398,.17f);
        auto gate=data({1,1,4,width},399,88.f).to(tb::kBFloat16);
        auto indices=tb::zeros({1,1,33},query.options().dtype(tb::kInt32));
        for(bool half_output:{false,true}) {
            bits(mfq_qwen4_exp::sparse_gqa_attention_gate(query,key,key,indices,gate,half_output),
                mfq_qwen4_exp::attention_gate(mfq_qwen4_exp::sparse_gqa_attention(query,key,key,indices),gate,half_output));
            ++ordinary;
        }
    }
    std::cout<<"QSA sparse fused gate ordinary="<<ordinary<<" changing_graph="<<changing
        <<" bytes exact; partial/whole tiles, invalid/repeated indices, Half/Float strided query/gates, BF16/empty-cache fallbacks PASS\n";
}
void qsa_sparse_gate_benchmark(bool compare_query=false) {
    constexpr int width=256,kv=2,capacity=4096,replays=256;
    const auto stream=mfq_current_cuda_stream();
    for(int heads:{24,48})for(int tokens:{1,3})for(int count:{32,512,2048}) {
        auto query=data({1,heads,tokens,width},401,.3f),key=data({1,kv,capacity,width},402,.17f);
        auto value=data(key.sizes().vec(),403,.19f),gate=data({1,tokens,heads,width},404,.7f);
        std::vector<int32_t> host_indices(tokens*count);
        for(int token=0;token<tokens;++token)for(int i=0;i<count;++i)
            host_indices[token*count+i]=((i/16*37+token)%(capacity/16))*16+i%16;
        auto indices=tb::tensor(host_indices).reshape({1,tokens,count}).to(tb::kCUDA);
        auto expected=mfq_qwen4_exp::attention_gate(
            mfq_qwen4_exp::sparse_gqa_attention(query,key,value,indices),gate,true).cpu().contiguous();
        for(bool fused:{false,true,false,false,true,false}) {
            if(compare_query) {
#ifdef _WIN32
                _putenv_s("MFQ_QSA_SPARSE_QUERY_FUSED",fused?"1":"0");
#else
                setenv("MFQ_QSA_SPARSE_QUERY_FUSED",fused?"1":"0",1);
#endif
            }
            Tensor output;
            const auto compute=[&]{output=(fused || compare_query)?
                mfq_qwen4_exp::sparse_gqa_attention_gate(query,key,value,indices,gate,true):
                mfq_qwen4_exp::attention_gate(mfq_qwen4_exp::sparse_gqa_attention(query,key,value,indices),gate,true);};
            {
                GraphWarmupScope warmup(default_context(mfq_current_cuda_device()),stream);compute();
            }
            MFQ_CUDA_CHECK(cudaStreamSynchronize(stream));output={};
            mfq::cuda::Graph graph;graph.capture_begin_shared();compute();graph.capture_end();
            for(int i=0;i<32;++i)graph.replay();
            MFQ_CUDA_CHECK(cudaStreamSynchronize(stream));
            auto actual=output.cpu().contiguous();
            if(actual.sizes()!=expected.sizes() || actual.scalar_type()!=expected.scalar_type() ||
                std::memcmp(actual.data_ptr(),expected.data_ptr(),actual.numel()*actual.element_size()))
                throw std::runtime_error("QSA sparse gate benchmark original bytes differ");
            cudaEvent_t begin=nullptr,end=nullptr;
            MFQ_CUDA_CHECK(cudaEventCreate(&begin));MFQ_CUDA_CHECK(cudaEventCreate(&end));
            MFQ_CUDA_CHECK(cudaEventRecord(begin,stream));
            for(int i=0;i<replays;++i)graph.replay();
            MFQ_CUDA_CHECK(cudaEventRecord(end,stream));MFQ_CUDA_CHECK(cudaEventSynchronize(end));
            float milliseconds=0;MFQ_CUDA_CHECK(cudaEventElapsedTime(&milliseconds,begin,end));
            MFQ_CUDA_CHECK(cudaEventDestroy(begin));MFQ_CUDA_CHECK(cudaEventDestroy(end));
            std::cout<<(compare_query?"qsa_sparse_query_bench heads=":"qsa_sparse_gate_bench heads=")<<heads<<" kv_heads="<<kv<<" tokens="<<tokens
                <<" width="<<width<<" selected="<<count<<" capacity="<<capacity<<" fused="<<fused
                <<" us="<<double(milliseconds)*1000/replays<<" replays="<<replays
                <<" scope=query_cast/indices/MMA/reduction/gate queued_graph bytes_exact=1\n";
        }
    }
}
}
#include "qsa_prefill_test.inc"
#include "gdn_prefill_columns_test.inc"
#include "gdn_prefill_pipeline_test.inc"
int main(int argc,char** argv)try {
    const auto stream=mfq_current_cuda_stream();auto context=default_context(mfq_current_cuda_device());context->begin_graph_pool(stream);
    if(argc==2 && std::string(argv[1])=="--gdn-core-check") {
        for(int tiles:{4,2,1}) {
            GdnTileScope setting(tiles);std::cout<<"gdn_decode_tiles="<<tiles<<'\n';gdn_core_case();
        }
    } else if(argc==2 && std::string(argv[1])=="--gdn-inplace-check") {
        for(int tiles:{4,2,1}){GdnTileScope setting(tiles);gdn_core_case(true);}
    } else if(argc==2 && std::string(argv[1])=="--gdn-core-bench")gdn_core_benchmark();
    else if(argc==2 && std::string(argv[1])=="--gdn-fixed-bench")gdn_core_benchmark(true);
    else if(argc==2 && std::string(argv[1])=="--gdn-inplace-bench")gdn_core_benchmark(false,true);
    else if(argc==2 && std::string(argv[1])=="--gdn-prefill-check")gdn_prefill_case();
    else if(argc==2 && std::string(argv[1])=="--gdn-prefill-columns-check")gdn_prefill_columns_case();
    else if(argc==2 && std::string(argv[1])=="--gdn-prefill-pipeline-default-check")gdn_prefill_pipeline_case(true);
    else if(argc==2 && std::string(argv[1])=="--gdn-prefill-pipeline-check")gdn_prefill_pipeline_case();
    else if(argc==2 && std::string(argv[1])=="--gdn-prefill-columns-bench")gdn_prefill_columns_benchmark();
    else if(argc==2 && std::string(argv[1])=="--qsa-prefill-check")qsa_prefill_fusion_case();
    else if(argc==2 && std::string(argv[1])=="--qsa-sparse-gate-check")qsa_sparse_gate_case();
    else if(argc==2 && std::string(argv[1])=="--qsa-sparse-gate-bench")qsa_sparse_gate_benchmark();
    else if(argc==2 && std::string(argv[1])=="--qsa-sparse-query-bench")qsa_sparse_gate_benchmark(true);
    else if(argc==2 && std::string(argv[1])=="--qsa-select-check")qsa_selection_case();
    else if(argc==2 && std::string(argv[1])=="--qsa-select-bench")qsa_selection_benchmark();
    else if(argc==2 && std::string(argv[1])=="--segmented-sort-check")segmented_sort_graph_case();
    else if(argc==1) {
        layer_profile_case();grouped_norm_case();gated_residual_case();gdn_output_case();gdn_preparation_case();gdn_core_case();segmented_sort_graph_case();qsa_sparse_gate_case();qsa_selection_case();qsa_case(8,4,40);qsa_case(256,24,41);qsa_case(256,24,512,2048);qsa_case(256,24,4096,2048);gdn_case();gdn_case(true);
    } else throw std::runtime_error("unknown attention test argument");
    MFQ_CUDA_CHECK(cudaStreamSynchronize(stream));context->end_graph_pool(stream);
    if(argc==1)std::cout<<"MFQ window QSA/GDN changing-position, rollback and fixed-state checks PASS\n";
    else if(std::string(argv[1])=="--gdn-core-check" || std::string(argv[1])=="--gdn-inplace-check")std::cout<<"MFQ GDN tile 4/2/1 exact checks PASS\n";
    else if(std::string(argv[1])=="--qsa-prefill-check")std::cout<<"MFQ QSA prefill fusion checks PASS\n";
    else if(std::string(argv[1])=="--qsa-sparse-gate-check")std::cout<<"MFQ QSA sparse gate exact checks PASS\n";
    else if(std::string(argv[1])=="--qsa-select-check")std::cout<<"MFQ QSA token selection exact checks PASS\n";
    else if(std::string(argv[1])=="--segmented-sort-check")std::cout<<"MFQ segmented sort graph checks PASS\n";
    else std::cout<<"MFQ attention benchmark completed\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
