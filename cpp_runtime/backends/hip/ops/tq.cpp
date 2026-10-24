#include "tq.h"
#include "tq_input_gate.h"
#include "tq_residual.h"
#include "mfq/tq_rows.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <atomic>
#include <cstdio>

namespace tb=mfq_tensor_backend;
namespace {
int dtype(tb::ScalarType type) {
    if(type==tb::kFloat16)return 0;
    if(type==tb::kBFloat16)return 1;
    if(type==tb::kFloat32)return 2;
    throw std::invalid_argument("TQ HIP activation dtype must be FP16, BF16 or FP32");
}
void status(int code) {MFQ_CUDA_CHECK(static_cast<hipError_t>(code));}
void trace(int role,const char* name,const tb::Tensor& x) {
    static const bool enabled=[] {
        const char* option=std::getenv("MFQ_HIP_TQ_TRACE");
        return option && std::strcmp(option,"1")==0;
    }();
    if(!enabled || x.numel()!=x.size(-1))return;
    static std::atomic<unsigned> seen{0};
    const int type=dtype(x.scalar_type());const unsigned bit=1u<<(role*3+type);
    if(!(seen.fetch_or(bit)&bit))std::fprintf(stderr,"tq_hip_path role=%s activation=%s tokens=1 input_columns=%lld\n",
        name,type==0?"FP16":type==1?"BF16":"FP32",static_cast<long long>(x.size(-1)));
}
}

bool tq_q8_enabled() {
    static const bool enabled=[] {
        const char* option=std::getenv("MFQ_HIP_TQ_Q8");
        return option && std::strcmp(option,"1")==0;
    }();
    return enabled;
}

tb::Tensor tq_gate_input(tb::Tensor input,tb::Tensor gate,int mode) {
    if(!input.is_cuda() || !gate.is_cuda() || input.device()!=gate.device() ||
       input.sizes()!=gate.sizes() || (mode!=1 && mode!=2))
        throw std::invalid_argument("TQ input gate requires matching GPU tensors and sigmoid/SiLU");
    const MfqCudaGuard guard(input.get_device());
    const bool double_math=gate.scalar_type()==input.scalar_type() && !gate.is_contiguous();
    const auto x=input.contiguous(),g=gate.contiguous();
    auto result=tb::empty(x.sizes(),x.options());
    status(mfq_tq_input_gate(x.data_ptr(),g.data_ptr(),result.data_ptr(),x.numel(),
        dtype(x.scalar_type()),dtype(g.scalar_type()),mode,double_math,mfq_current_cuda_stream()));
    return result;
}

std::vector<tb::Tensor> tq_forward_group(
        const std::vector<const TqWeight*>& weights,tb::Tensor input) {
    if(weights.empty() || weights.size()>4 || !input.is_cuda())
        throw std::invalid_argument("TQ projection group requires 1..4 GPU weights");
    const auto x=input.contiguous();
    if(x.numel()!=weights[0]->neuron_len || x.size(-1)!=weights[0]->neuron_len)
        throw std::invalid_argument("TQ fused group requires one input token");
    const MfqCudaGuard guard(x.get_device());
    trace(1,"group",x);
    std::vector<tb::Tensor> result;
    for(const auto* w:weights) {
        if(w->neuron_len!=x.size(-1) || w->image.get_device()!=x.get_device())
            throw std::invalid_argument("TQ grouped weight geometry or device differs");
        auto shape=x.sizes().vec();shape.back()=w->out;
        result.push_back(tb::empty(shape,x.options()));
    }
    // Schedule tiny gate projections early so their few blocks overlap the
    // larger projections. Output tensors retain the caller's original order.
    std::vector<int> order;for(int i=0;i<int(weights.size());++i)order.push_back(i);
    std::stable_sort(order.begin(),order.end(),[&](int a,int b){return (weights[a]->out<=64)>(weights[b]->out<=64);});
    TqProjectionPlan plan{};plan.count=weights.size();
    for(int i=0;i<plan.count;++i) {
        const int j=order[i];plan.jobs[i]={weights[j]->view,result[j].data_ptr(),0};
    }
    status(mfq_tq_group(plan,x.data_ptr(),dtype(x.scalar_type()),mfq_current_cuda_stream()));
    return result;
}

tb::Tensor tq_swiglu(const TqWeight& gate,const TqWeight& up,tb::Tensor input) {
    if(!input.is_cuda() || input.numel()!=gate.neuron_len || input.size(-1)!=gate.neuron_len ||
       gate.neuron_len!=up.neuron_len || gate.out!=up.out ||
       input.get_device()!=gate.image.get_device() || input.get_device()!=up.image.get_device())
        throw std::invalid_argument("TQ fused SwiGLU geometry or device differs");
    const MfqCudaGuard guard(input.get_device());
    const auto x=input.contiguous();auto shape=x.sizes().vec();shape.back()=gate.out;
    trace(2,"swiglu",x);
    auto result=tb::empty(shape,x.options());
    status(mfq_tq_swiglu(gate.view,up.view,x.data_ptr(),result.data_ptr(),
        dtype(x.scalar_type()),mfq_current_cuda_stream()));
    return result;
}

tb::Tensor tq_forward_residual(const TqWeight& weight,tb::Tensor input,tb::Tensor residual) {
    if(!input.is_cuda() || !residual.is_cuda() || input.scalar_type()!=residual.scalar_type() ||
       input.numel()!=weight.neuron_len ||
       residual.numel()!=weight.out || input.size(-1)!=weight.neuron_len ||
       residual.size(-1)!=weight.out || input.get_device()!=weight.image.get_device() ||
       residual.get_device()!=input.get_device())
        throw std::invalid_argument("TQ fused residual requires compatible single-token tensors");
    const MfqCudaGuard guard(input.get_device());
    const auto x=input.contiguous(),skip=residual.contiguous();
    trace(3,"residual",x);
    auto shape=x.sizes().vec();shape.back()=weight.out;
    auto result=tb::empty(shape,x.options());
    status(mfq_tq_residual(weight.view,x.data_ptr(),skip.data_ptr(),result.data_ptr(),
        dtype(x.scalar_type()),mfq_current_cuda_stream()));
    return result;
}

TqWeight to_device_tq(const std::vector<std::uint8_t>& blob,std::int64_t columns,int device) {
    const mfq::TqLayout m(blob.data(),blob.size());
    if(columns<1 || columns>m.width || m.rows>INT_MAX || m.width>INT_MAX)
        throw std::invalid_argument("invalid TQ HIP logical shape");
    // Validate descriptors, complete packed coverage, and finite positive scales
    // before uploading. The retained tensor keeps its ordinary RTRELL05 layout.
    mfq::TqRows rows(blob.size(),[&](std::uint64_t offset,std::byte* out,std::size_t size){
        if(offset>blob.size() || size>blob.size()-offset)throw std::out_of_range("TQ read");
        std::memcpy(out,blob.data()+offset,size);
    });
    (void)rows.slice_rows(0,m.rows);
    if(m.columns)for(std::size_t i=0;i<m.width;++i) {
        const auto bits=mfq::TqLayout::integer(blob.data()+m.scales+i*2,2);
        if(!bits || bits>0x7bff)throw std::invalid_argument("invalid TQ column scale");
    }
    const auto words=(m.packed+3)&~std::size_t(3),relative=words+m.packed_bytes;
    std::vector<std::uint8_t> image(relative+32);
    std::memcpy(image.data(),blob.data(),m.packed);
    std::memcpy(image.data()+words,blob.data()+m.packed,m.packed_bytes);
    constexpr std::uint16_t gains[]{0x3400,0x34d0,0x35ca,0x36f7,0x3830,0x390a,0x3a10,0x3b4b,
        0x3c63,0x3d47,0x3e59,0x3fa4,0x4098,0x4187,0x42a6,0x4400};
    std::memcpy(image.data()+relative,gains,sizeof(gains));
    const MfqCudaGuard guard(device);
    TqWeight result;
    result.image=tb::from_blob(image.data(),{static_cast<std::int64_t>(image.size())},
        tb::TensorOptions().dtype(tb::kUInt8)).clone().to(tb::Device(tb::kCUDA,device)).contiguous();
    const auto* p=result.image.data_ptr<std::uint8_t>();
    int gslog=0;for(auto gs=m.gs;gs>1;gs>>=1)++gslog;
    result.view={reinterpret_cast<const std::uint32_t*>(p+words),
        reinterpret_cast<const std::uint64_t*>(p+64),p+m.scales,p+relative,nullptr,
        int(m.rows),int(m.width),int(columns),gslog,int(m.columns)};
    result.out=m.rows;result.neuron_len=columns;
    return result;
}

TqWeight load_tq_weight(const mfq::ModelSource& source,const std::string& name,int device) {
    const auto* record=source.find_tensor(name);
    if(!record || record->dtype!="TQ")throw std::invalid_argument("expected TQ tensor: "+name);
    std::vector<std::int64_t> shape;
    for(const auto* key:{"tq_shapes","tensor_shapes"}) {
        const auto entry=source.metadata().find(key);
        if(entry==source.metadata().end())continue;
        const auto shapes=nlohmann::json::parse(entry->second);
        if(!shapes.is_object())throw std::invalid_argument("invalid TQ shape metadata");
        if(shapes.contains(name)){shape=shapes.at(name).get<std::vector<std::int64_t>>();break;}
    }
    if(shape.empty() && record->shape)shape=*record->shape;
    if(shape.size()!=2 || shape[0]<1 || shape[1]<1)
        throw std::invalid_argument("TQ HIP projection requires logical [out,in] shape: "+name);
    const auto bytes=source.read(name);
    std::vector<std::uint8_t> blob(bytes.size());
    std::memcpy(blob.data(),bytes.data(),bytes.size());
    auto result=to_device_tq(blob,shape[1],device);
    if(result.out!=shape[0])throw std::invalid_argument("TQ output rows differ: "+name);
    return result;
}

tb::Tensor TqWeight::materialize(tb::ScalarType type) const {
    if(type!=tb::kFloat16 && type!=tb::kFloat32)
        throw std::invalid_argument("TQ exact materialization requires FP16 or FP32");
    const MfqCudaGuard guard(image.get_device());
    auto result=tb::empty({out,neuron_len},image.options().dtype(type));
    status(mfq_tq_materialize(view,result.data_ptr(),dtype(type),mfq_current_cuda_stream()));
    return result;
}

tb::Tensor TqWeight::forward(tb::Tensor input) const {
    if(input.dim()<1 || input.size(-1)!=neuron_len || !input.is_cuda() ||
       input.get_device()!=image.get_device())
        throw std::invalid_argument("TQ HIP projection input shape or device differs");
    const int type=dtype(input.scalar_type());
    const MfqCudaGuard guard(image.get_device());
    const auto x=input.contiguous();
    const auto tokens=x.numel()/neuron_len;
    trace(0,"projection",x);
    auto shape=x.sizes().vec();shape.back()=out;
    if(!tokens)return tb::empty(shape,x.options());
    // Same geometry policy as the Python HIP operator. Dense storage is
    // temporary per invocation; compressed weights remain the model storage.
    const auto area=out*neuron_len;
    const bool small=tokens<=8 || (tokens<=32 && out<=64);
    // Match CUDA's optional Q8 activation algorithm. Exact FP16 weight
    // reconstruction remains the default and is used for calibration.
    const bool q8=tq_q8_enabled();
    if(q8 && tokens<=4 && out>64 && type!=1 && (view.columns || view.gslog>=4)) {
        auto result=tb::empty(shape,x.options());
        const auto bytes=tokens*((neuron_len+511)/512*512/16)*24;
        auto scratch=tb::empty({bytes},x.options().dtype(tb::kUInt8));
        status(mfq_tq_q8_tokens(view,x.data_ptr(),result.data_ptr(),int(tokens),type,
            scratch.data_ptr(),bytes,mfq_current_cuda_stream()));
        return result;
    }
    const bool staged=!small && neuron_len>64 && area>=16384 &&
        (neuron_len>=512 || tokens*area>=(1LL<<27));
    if(staged) {
        const auto compute=type==0?tb::kFloat16:tb::kFloat32;
        auto w=materialize(compute);
        return tb::linear(x.reshape({tokens,neuron_len}).to(compute),w)
            .to(x.scalar_type()).reshape(shape);
    }
    auto result=tb::empty(shape,x.options());
    const auto bytes=x.element_size();
    for(std::int64_t start=0;start<tokens;start+=65535*16) {
        status(mfq_tq_forward(view,static_cast<const std::uint8_t*>(x.data_ptr())+start*neuron_len*bytes,
            static_cast<std::uint8_t*>(result.data_ptr())+start*out*bytes,
            int(std::min<std::int64_t>(tokens-start,65535*16)),type,mfq_current_cuda_stream()));
    }
    return result;
}
