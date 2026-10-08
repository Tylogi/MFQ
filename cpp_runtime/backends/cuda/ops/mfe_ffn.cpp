#include "../runtime/execution_options.h"
#include "mfe_ffn_runtime.h"
#include <cstring>
#include <cstdlib>
#include <map>

namespace {
namespace tb=mfq_tensor_backend;
using View=mfq::cuda::MfePackedProjection;
template<class T>tb::Tensor upload(const std::vector<T>& values) {
    auto host=tb::empty({int64_t(values.size()*sizeof(T))},tb::TensorOptions().device(tb::kCPU).dtype(tb::kUInt8));
    std::memcpy(host.data_ptr(),values.data(),values.size()*sizeof(T));return host.to(tb::kCUDA);
}
View nint_view(const NintWeight& w,int rows,int experts,std::vector<tb::Tensor>& owners) {
    View result;result.family=1;result.output_rows=rows;result.local_experts=experts;
    result.input_width=int(w.neuron_len);result.groups=int(w.ng);result.group_size=int(w.gs);
    result.q_expert_stride=int(w.q_expert_stride);
    const std::array<tb::Tensor,7> fields={w.q_packed,w.row_q_bits,w.row_q_bit_offsets,w.sub_scale,w.sub_min,w.neuron_scale,w.neuron_min};
    for(int i=0;i<7;++i){result.fields[i]=fields[i].data_ptr();owners.push_back(fields[i]);}
    return result;
}
View pool_view(const MixedMoePool& pool,int rows,std::vector<tb::Tensor>& owners) {
    View result;
    if(pool.family==MixedMoeFamily::Nint)result=nint_view(pool.nint,rows,pool.local_experts,owners);
    else if(pool.family==MixedMoeFamily::Nvq) {
        const auto& w=pool.nvq;result.family=2;result.output_rows=rows;result.local_experts=pool.local_experts;
        result.input_width=int(w.neuron_len);result.groups=int(w.ng);result.group_size=int(w.gs);
        result.format=int(w.kernel_format);result.sub_bits=int(w.sub_bits);result.sign_mode=int(w.sign_mode);
        const bool d4=result.format==3 || result.format==10 || result.format==11 ||
            result.format==12 || result.format==15 || result.format==17;
        result.nvec=(result.input_width+(d4?3:7))/(d4?4:8);result.nsign=(result.input_width+7)/8;
        const std::array<tb::Tensor,5> fields={w.indices_packed,w.aux_packed,w.sub_scale_packed,w.neuron_scale,w.codebook};
        for(int i=0;i<5;++i){result.fields[i]=fields[i].data_ptr();owners.push_back(fields[i]);}
        for(int i=0;i<3;++i)result.sizes[i]=fields[i].numel()*fields[i].element_size();
    } else throw std::invalid_argument("two-stage MFE requires native NINT/NVQ projections");
    if(result.group_size<1 || result.group_size>64 || result.groups<=0)
        throw std::invalid_argument("two-stage MFE input group geometry unsupported");
    if(pool.family==MixedMoeFamily::Nvq && (result.group_size!=24 || result.format<1 || result.format>17))
        throw std::invalid_argument("two-stage MFE NVQ execution layout unsupported");
    result.expert_local=pool.expert_local.data_ptr<int32_t>();owners.push_back(pool.expert_local);return result;
}
}
MfeFfnRuntime::MfeFfnRuntime(const std::array<MixedMoeRuntime*,3>& projections,
        const tb::Tensor& input,const tb::Tensor& ids,const tb::Tensor& weights,const tb::Tensor& sigmoid,
        const std::array<const NintWeight*,3>& shared,const tb::Tensor& shared_gate,
        int output_columns,int intermediate_columns,const std::array<std::vector<int32_t>,3>* cohort_by_expert,bool output_pairs)
        :sigmoid_(sigmoid),input_(input),ids_(ids),weights_(weights) {
    if(!projections[0] || !projections[1] || !projections[2] || !input.is_cuda() || input.dim()!=2 ||
        input.scalar_type()!=tb::kFloat16 || !input.is_contiguous() || ids.dim()!=2 || ids.scalar_type()!=tb::kInt32 ||
        !ids.is_cuda() || ids.get_device()!=input.get_device() || !weights.is_cuda() ||
        weights.get_device()!=input.get_device() || !ids.is_contiguous() || ids.size(0)!=input.size(0) || weights.sizes()!=ids.sizes() ||
        weights.scalar_type()!=tb::kFloat32 || !weights.is_contiguous() || !sigmoid.is_cuda() ||
        sigmoid.get_device()!=input.get_device() || !sigmoid.is_contiguous() ||
        sigmoid.scalar_type()!=tb::kFloat16 || sigmoid.dim()!=2 || sigmoid.size(0)!=2 || sigmoid.size(1)!=65536)
        throw std::invalid_argument("two-stage MFE tensor contract mismatch");
    batch_.tokens=int(input.size(0));batch_.routes=int(ids.size(1));batch_.experts=projections[0]->n_experts;
    batch_.input_width=int(input.size(1));
    batch_.intermediate=intermediate_columns?intermediate_columns:projections[0]->out_per_expert;
    batch_.output_width=output_columns?output_columns:projections[2]->out_per_expert;
    if(batch_.tokens<=0 || batch_.routes<=0 || batch_.experts<=0 || batch_.intermediate<=0 || batch_.output_width<=0 ||
        batch_.intermediate>projections[0]->out_per_expert || batch_.output_width>projections[2]->out_per_expert)
        throw std::invalid_argument("two-stage MFE dimensions invalid");
    std::vector<View> views;std::vector<int32_t> mapping;
    std::map<std::pair<int,int>,std::pair<tb::Tensor,tb::Tensor>> quantizers;
    const auto attach_input=[&](View& d) {
        const auto key=std::make_pair(d.groups,d.group_size);
        auto it=quantizers.find(key);
        if(it==quantizers.end()) {
            auto q=tb::empty({batch_.tokens,d.groups*d.group_size},input.options().dtype(tb::kInt8));
            auto s=tb::empty({batch_.tokens,d.groups},input.options().dtype(tb::kFloat32));
            it=quantizers.emplace(key,std::make_pair(q,s)).first;
        }
        d.fields[7]=it->second.first.data_ptr();d.fields[8]=it->second.second.data_ptr();
    };
    for(int projection=0;projection<3;++projection) {
        auto& runtime=*projections[projection];
        if(runtime.n_experts!=batch_.experts || (projection<2 && runtime.neuron_len!=batch_.input_width) ||
            (projection==1 && runtime.out_per_expert!=projections[0]->out_per_expert))
            throw std::invalid_argument("two-stage MFE projection dimensions mismatch");
        batch_.projection_offsets[projection]=int(views.size());batch_.projection_counts[projection]=int(runtime.pools.size());
        std::vector<int32_t> indices(batch_.experts,-1);
        for(std::size_t p=0;p<runtime.pools.size();++p) {
            auto d=pool_view(runtime.pools[p],runtime.out_per_expert,owners_);
            if(projection<2)attach_input(d);
            else {
                if(d.groups!=(batch_.intermediate+d.group_size-1)/d.group_size)
                    throw std::invalid_argument("two-stage MFE intermediate changes group topology");
                batch_.quantized_stride=std::max(batch_.quantized_stride,d.groups*d.group_size);
                batch_.scale_stride=std::max(batch_.scale_stride,d.groups);
            }
            if(!cohort_by_expert) {
                const auto local=runtime.pools[p].expert_local.to(tb::kCPU).contiguous();
                const auto* values=local.data_ptr<int32_t>();
                for(int e=0;e<batch_.experts;++e)if(values[e]>=0) {
                    if(indices[e]>=0)throw std::invalid_argument("two-stage MFE expert has overlapping pools");
                    indices[e]=int32_t(p);
                }
            }
            views.push_back(d);
        }
        if(cohort_by_expert)indices=(*cohort_by_expert)[projection];
        if(indices.size()!=std::size_t(batch_.experts))throw std::invalid_argument("two-stage MFE static expert mapping invalid");
        mapping.insert(mapping.end(),indices.begin(),indices.end());
    }
    if(shared[0] || shared[1] || shared[2]) {
        if(!shared[0] || !shared[1] || !shared[2] || !shared_gate.defined() || shared_gate.numel()!=batch_.tokens)
            throw std::invalid_argument("two-stage MFE shared projection incomplete");
        batch_.shared_intermediate=int(shared[0]->out);
        if(shared[1]->out!=shared[0]->out || shared[0]->neuron_len!=batch_.input_width ||
            shared[1]->neuron_len!=batch_.input_width || shared[2]->neuron_len!=shared[0]->out ||
            shared[2]->out<batch_.output_width)throw std::invalid_argument("two-stage MFE shared geometry invalid");
        for(int p=0;p<3;++p) {
            auto d=nint_view(*shared[p],int(shared[p]->out),1,owners_);d.expert_local=nullptr;
            if(d.group_size<1 || d.group_size>64 || d.groups<=0)
                throw std::invalid_argument("two-stage MFE shared input group geometry unsupported");
            if(p<2)attach_input(d);
            else {batch_.quantized_stride=std::max(batch_.quantized_stride,d.groups*d.group_size);
                  batch_.scale_stride=std::max(batch_.scale_stride,d.groups);}
            views.push_back(d);
        }
        if(!shared_gate.is_cuda() || shared_gate.get_device()!=input.get_device() || shared_gate.numel()!=batch_.tokens ||
            (shared_gate.scalar_type()!=tb::kFloat16 && shared_gate.scalar_type()!=tb::kBFloat16 && shared_gate.scalar_type()!=tb::kFloat32))
            throw std::invalid_argument("two-stage MFE shared gate device/dtype unsupported");
        shared_gate_=shared_gate.contiguous();
        batch_.shared_product_half=shared_gate.scalar_type()==tb::kFloat16;
        batch_.shared_gate_bfloat=shared_gate.scalar_type()==tb::kBFloat16;
        batch_.output_float=!batch_.shared_product_half;
    }
    if(batch_.routes+(batch_.shared_intermediate?1:0)>32)
        throw std::invalid_argument("two-stage MFE route warps exceed one CUDA block");
    host_views_=views;host_cohorts_=mapping;
    descriptors_=upload(views);cohorts_=tb::tensor(mapping).to(tb::kCUDA);
    std::vector<mfq::cuda::MfeInputQuantization> quantization;
    for(const auto& entry:quantizers) {
        const auto& q=entry.second.first;const auto& s=entry.second.second;
        quantization.push_back({q.data_ptr<int8_t>(),s.data_ptr<float>(),entry.first.first,entry.first.second});
        owners_.push_back(q);owners_.push_back(s);quantization_groups_+=entry.first.first;
    }
    quantization_count_=int(quantization.size());quantization_=upload(quantization);
    batch_.hidden_stride=std::max(batch_.intermediate,batch_.shared_intermediate);
    const int slots=batch_.tokens*(batch_.routes+(shared[0]?1:0));
    hidden_=tb::empty({slots,batch_.hidden_stride},input.options());
    quantized_=tb::empty({slots,batch_.quantized_stride},input.options().dtype(tb::kInt8));
    scales_=tb::empty({slots,batch_.scale_stride},input.options().dtype(tb::kFloat32));
    completion_=tb::empty({slots,batch_.scale_stride},input.options().dtype(tb::kInt32));
    if(output_pairs && shared[0])throw std::invalid_argument("two-stage MFE pair output cannot contain shared experts");
    batch_.output_pairs=output_pairs;
    output_=tb::empty(output_pairs?std::vector<int64_t>{batch_.tokens,batch_.routes,batch_.output_width}:
        std::vector<int64_t>{batch_.tokens,batch_.output_width},input.options().dtype(batch_.output_float?tb::kFloat32:tb::kFloat16));
    batch_.projections=reinterpret_cast<const View*>(descriptors_.data_ptr());
    batch_.cohort_by_expert=cohorts_.data_ptr<int32_t>();
    if(shared[0])batch_.shared=batch_.projections+views.size()-3;
    batch_.input=reinterpret_cast<const __half*>(input_.data_ptr());batch_.ids=ids_.data_ptr<int32_t>();
    batch_.route_weights=weights_.data_ptr<float>();batch_.sigmoid_table=reinterpret_cast<const __half*>(sigmoid_.data_ptr());
    batch_.shared_gate=shared_gate_.defined()?shared_gate_.data_ptr():nullptr;
    batch_.hidden=reinterpret_cast<__half*>(hidden_.data_ptr());batch_.hidden_quantized=quantized_.data_ptr<int8_t>();
    batch_.hidden_scales=scales_.data_ptr<float>();batch_.completion=reinterpret_cast<uint32_t*>(completion_.data_ptr<int32_t>());
    batch_.output=output_.data_ptr();
}
void MfeFfnRuntime::cpu_results(const tb::Tensor& kinds,const tb::Tensor& pairs,const uint32_t* ready) {
    if(!kinds.is_cuda() || !pairs.is_cuda() || kinds.get_device()!=input_.get_device() ||
        pairs.get_device()!=input_.get_device() || kinds.numel()!=batch_.experts || kinds.scalar_type()!=tb::kInt32 || !kinds.is_contiguous() ||
        pairs.dim()!=3 || pairs.size(0)!=batch_.tokens || pairs.size(1)!=batch_.routes || pairs.size(2)!=batch_.output_width ||
        pairs.scalar_type()!=tb::kFloat16 || !pairs.is_contiguous())throw std::invalid_argument("two-stage MFE CPU result contract mismatch");
    kinds_=kinds;cpu_=pairs;batch_.kinds=kinds.data_ptr<int32_t>();batch_.cpu_pairs=reinterpret_cast<const __half*>(pairs.data_ptr());batch_.cpu_ready=ready;
}
void MfeFfnRuntime::asynchronous(const int32_t* kinds,const uint32_t* plan,const uint32_t* transfer,
        const uint32_t* cpu_ready,const uint32_t* abort,const int32_t* indices,const void* cpu_pairs) {
    if(!kinds || !plan || !transfer || !cpu_ready || !abort || !indices || !cpu_pairs)
        throw std::invalid_argument("two-stage MFE asynchronous resources incomplete");
    transferred_=tb::empty({int64_t(batch_.tokens)*batch_.routes*3*int64_t(sizeof(View))},input_.options().dtype(tb::kUInt8));
    kinds_=tb::empty({batch_.experts},input_.options().dtype(tb::kInt32));
    transfer_indices_=tb::empty({batch_.experts},kinds_.options());
    aborted_=tb::empty({1},kinds_.options());
    resident_pairs_=tb::empty({batch_.tokens,batch_.routes+(batch_.shared?1:0),batch_.output_width},input_.options());
    batch_.resident_pairs=reinterpret_cast<__half*>(resident_pairs_.data_ptr());
    batch_.transferred=reinterpret_cast<const View*>(transferred_.data_ptr());
    batch_.transfer_index=transfer_indices_.data_ptr<int32_t>();batch_.host_transfer_index=indices;
    batch_.kinds=kinds_.data_ptr<int32_t>();batch_.host_kinds=kinds;
    batch_.plan_ready=plan;batch_.transfer_ready=transfer;batch_.cpu_ready=cpu_ready;
    batch_.aborted=reinterpret_cast<uint32_t*>(aborted_.data_ptr<int32_t>());batch_.host_aborted=abort;
    batch_.cpu_pairs=reinterpret_cast<const __half*>(cpu_pairs);
    batch_.resident_plan_overlap=mfq::cuda::runtime_options::resident_plan_overlap();
}
mfq::cuda::MfePackedProjection MfeFfnRuntime::expert_view(int p,int expert,
        const std::vector<int64_t>& bytes,int slot)const {
    if(p<0 || p>2 || expert<0 || expert>=batch_.experts)throw std::invalid_argument("MFE transfer projection/expert invalid");
    const int pool=host_cohorts_[p*batch_.experts+expert];
    if(pool<0 || pool>=batch_.projection_counts[p])throw std::invalid_argument("MFE transfer cohort missing");
    auto view=host_views_[batch_.projection_offsets[p]+pool];
    if(bytes.size()!=std::size_t(view.family==1?7:4))throw std::invalid_argument("MFE transfer field count differs");
    for(std::size_t f=0;f<bytes.size();++f)view.fields[f]=slot>=0?
        static_cast<const unsigned char*>(view.fields[f])+int64_t(slot)*bytes[f]:nullptr;
    if(view.family==2)for(int i=0;i<3;++i)view.sizes[i]=bytes[i];
    view.expert_local=nullptr;view.local_experts=1;view.fallback=-1;return view;
}
void MfeFfnRuntime::prepare(){mfq::cuda::mfe_ffn_prepare(batch_,reinterpret_cast<const mfq::cuda::MfeInputQuantization*>(quantization_.data_ptr()),
    quantization_count_,quantization_groups_,mfq_current_cuda_stream());MFQ_CUDA_KERNEL_LAUNCH_CHECK();}
void MfeFfnRuntime::gate_up(){mfq::cuda::mfe_ffn_gate_up(batch_,mfq_current_cuda_stream());MFQ_CUDA_KERNEL_LAUNCH_CHECK();}
void MfeFfnRuntime::resident(){mfq::cuda::mfe_ffn_resident(batch_,mfq_current_cuda_stream());MFQ_CUDA_KERNEL_LAUNCH_CHECK();}
void MfeFfnRuntime::wait_transfer(){mfq::cuda::mfe_ffn_wait_transfer(batch_,mfq_current_cuda_stream());MFQ_CUDA_KERNEL_LAUNCH_CHECK();}
void MfeFfnRuntime::transferred(){mfq::cuda::mfe_ffn_transferred(batch_,mfq_current_cuda_stream());MFQ_CUDA_KERNEL_LAUNCH_CHECK();}
void MfeFfnRuntime::wait_cpu(){mfq::cuda::mfe_ffn_wait_cpu(batch_,mfq_current_cuda_stream());MFQ_CUDA_KERNEL_LAUNCH_CHECK();}
void MfeFfnRuntime::down_reduce_compute(){mfq::cuda::mfe_ffn_down_reduce_compute(batch_,mfq_current_cuda_stream());MFQ_CUDA_KERNEL_LAUNCH_CHECK();}
void MfeFfnRuntime::down_reduce(){mfq::cuda::mfe_ffn_down_reduce(batch_,mfq_current_cuda_stream());MFQ_CUDA_KERNEL_LAUNCH_CHECK();}
