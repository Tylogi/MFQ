#include "../runtime/execution_options.h"
#include "mfe_ffn_runtime.h"
#include "mfe_e8_narrow_bounds.h"
#include "mfe_ffn_launch.h"
#include "mfe_ffn_compact.h"
#include "mfe_ffn_prepare.h"
#include "mfe_ffn_token_groups.h"
#include "mfe_ffn_shared.h"
#include "mfe_nvq_dense_bounds.h"
#include "mfq_cuda_nint_route_hint.h"
#include "runtime/decode_window.h"
#include <algorithm>
#include <atomic>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <map>

namespace mfq::cuda {
bool mfe_ffn_compact_transferred_fused(const MfeFfnBatch&,cudaStream_t,const int32_t*);
}

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
    result.sizes[0]=w.q_packed.numel()*w.q_packed.element_size();
    return result;
}
View pool_view(const MixedMoePool& pool,int rows,std::vector<tb::Tensor>& owners) {
    View result;
    if(pool.family==MixedMoeFamily::Nint)result=nint_view(pool.nint,rows,pool.local_experts,owners);
    else if(pool.family==MixedMoeFamily::Nvq) {
        const auto& w=pool.nvq;result.family=2;result.output_rows=rows;result.local_experts=pool.local_experts;
        const auto word_signs=mfq::cuda::runtime_options::mfe_nvq_word_signs();
        result.nvq_word_signs=!word_signs || *word_signs;
        result.input_width=int(w.neuron_len);result.groups=int(w.ng);result.group_size=int(w.gs);
        result.format=int(w.kernel_format);result.sub_bits=int(w.sub_bits);result.sign_mode=int(w.sign_mode);
        const bool d4=result.format==3 || result.format==10 || result.format==11 ||
            result.format==12 || result.format==15 || result.format==17;
        result.e8_dense_groups=w.dense_groups && !d4;
        result.d4_dense_groups=w.dense_groups && d4;
        const auto bank_map_option=mfq::cuda::runtime_options::mfe_nvq_bank_map();
        result.nvq_bank_map=bank_map_option && !*bank_map_option?0:w.bank_map;
        if(d4 && result.nvq_bank_map) {
            static std::atomic<uint32_t> reported{0};
            const uint32_t bit=1u<<result.format;
            if(!(reported.fetch_or(bit,std::memory_order_relaxed)&bit))std::fprintf(stderr,
                "mfe_d4_bank_map_dispatch scope=first_descriptor enabled=1 format=%d map=0x%08x\n",
                result.format,result.nvq_bank_map);
        }
        result.nvec=(result.input_width+(d4?3:7))/(d4?4:8);result.nsign=(result.input_width+7)/8;
        const std::array<tb::Tensor,5> fields={w.indices_packed,w.aux_packed,w.sub_scale_packed,w.neuron_scale,w.codebook};
        for(int i=0;i<5;++i){result.fields[i]=fields[i].data_ptr();owners.push_back(fields[i]);}
        for(int i=0;i<3;++i)result.sizes[i]=fields[i].numel()*fields[i].element_size();
        if(w.integer_codebook.defined()) {
            result.fields[5]=w.integer_codebook.data_ptr();owners.push_back(w.integer_codebook);
        }
        if(w.decode_records.defined()) {
            result.fields[6]=w.decode_records.data_ptr();owners.push_back(w.decode_records);
            result.decode_record_bytes=w.decode_records.numel();
        }
    } else if(pool.family==MixedMoeFamily::Mxfp4) {
        const auto& w=pool.mxfp4;
        result.family=3;result.output_rows=rows;result.local_experts=pool.local_experts;
        result.input_width=int(w.neuron_len);result.group_size=32;result.groups=result.input_width/32;
        const std::array<tb::Tensor,2> fields{w.values,w.scales};
        for(int i=0;i<2;++i) {
            result.fields[i]=fields[i].data_ptr();result.sizes[i]=fields[i].nbytes();owners.push_back(fields[i]);
        }
    } else throw std::invalid_argument("two-stage MFE requires native NINT/NVQ/MXFP4 projections");
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
        int output_columns,int intermediate_columns,const std::array<std::vector<int32_t>,3>* cohort_by_expert,bool output_pairs,
        bool combined_gate_up,int token_group_capacity_hint,double swiglu_limit,
        bool ungated_experts,bool ungated_shared)
        :sigmoid_(sigmoid),input_(input),ids_(ids),weights_(weights),combined_gate_up_(combined_gate_up) {
    // A relu-squared bank has no gate projection: projection 0 stays empty and
    // every dimension comes from Up. The shared view applies the same rule.
    ungated_experts_=ungated_experts;
    const int first_projection=ungated_experts?1:0;
    if((!projections[0] && !ungated_experts) || !projections[1] || !projections[2] || !input.is_cuda() || input.dim()!=2 ||
        input.scalar_type()!=tb::kFloat16 || !input.is_contiguous() || ids.dim()!=2 || ids.scalar_type()!=tb::kInt32 ||
        !ids.is_cuda() || ids.get_device()!=input.get_device() || !weights.is_cuda() ||
        weights.get_device()!=input.get_device() || !ids.is_contiguous() || ids.size(0)!=input.size(0) || weights.sizes()!=ids.sizes() ||
        weights.scalar_type()!=tb::kFloat32 || !weights.is_contiguous() || !sigmoid.is_cuda() ||
        sigmoid.get_device()!=input.get_device() || !sigmoid.is_contiguous() ||
        sigmoid.scalar_type()!=tb::kFloat16 || sigmoid.dim()!=2 || sigmoid.size(0)!=2 || sigmoid.size(1)!=65536)
        throw std::invalid_argument("two-stage MFE tensor contract mismatch");
    batch_.swiglu_limit=float(swiglu_limit);
    batch_.expert_activation=ungated_experts?1:0;
    batch_.shared_activation=ungated_shared?1:0;
    batch_.tokens=int(input.size(0));batch_.routes=int(ids.size(1));batch_.experts=projections[first_projection]->n_experts;
    batch_.input_width=int(input.size(1));
    batch_.warp_multi_sum=mfq::cuda::runtime_options::warp_multi_sum();
    batch_.intermediate=intermediate_columns?intermediate_columns:projections[first_projection]->out_per_expert;
    batch_.output_width=output_columns?output_columns:projections[2]->out_per_expert;
    if(batch_.tokens<=0 || batch_.routes<=0 || batch_.experts<=0 || batch_.intermediate<=0 || batch_.output_width<=0 ||
        batch_.intermediate>projections[first_projection]->out_per_expert ||
        batch_.output_width>projections[2]->out_per_expert)
        throw std::invalid_argument("two-stage MFE dimensions invalid");
    std::vector<View> views;std::vector<int32_t> mapping;
    // Projection 0 has no descriptors when ungated; keep the expert stride so
    // phase-1/2 cohort lookups remain at their original indices.
    if(ungated_experts)mapping.assign(batch_.experts,-1);
    batch_.nvq_format_mask=0;
    const bool nint_whole=mfq::cuda::mfe_ffn_nint_whole_requested();
    auto& quantizers=input_quantizers_;
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
    const auto late_option=mfq::cuda::runtime_options::mfe_nint_late_scale();
    const auto grouped_option=mfq::cuda::runtime_options::nint_group_dot();
    const auto hint_option=mfq::cuda::runtime_options::nint_route_hint();
    const auto warp_option=mfq::cuda::runtime_options::nint_route_warps();
    const bool align_late=late_option && *late_option && batch_.tokens<=8 &&
        (!grouped_option || *grouped_option) && (!hint_option || *hint_option) &&
        (!warp_option || *warp_option!=4);
    batch_.projection_offsets[0]=0;batch_.projection_counts[0]=0;
    for(int projection=first_projection;projection<3;++projection) {
        auto& runtime=*projections[projection];
        if(runtime.n_experts!=batch_.experts || (projection<2 && runtime.neuron_len!=batch_.input_width) ||
            (projection==1 && runtime.out_per_expert!=projections[1]->out_per_expert))
            throw std::invalid_argument("two-stage MFE projection dimensions mismatch");
        batch_.projection_offsets[projection]=int(views.size());batch_.projection_counts[projection]=int(runtime.pools.size());
        std::vector<int32_t> indices(batch_.experts,-1);
        int nint_pools=0;
        for(const auto& pool:runtime.pools)nint_pools+=pool.family==MixedMoeFamily::Nint;
        for(std::size_t p=0;p<runtime.pools.size();++p) {
            auto d=pool_view(runtime.pools[p],runtime.out_per_expert,owners_);
            // Long GS24/28 GU rows and GS28 Down improved complete FFN tests.
            // GS24 Down keeps the original decoder, including shared q6.
            d.nint_whole=nint_whole && d.family==1 &&
                (d.group_size==28 || (projection<2 && d.group_size==24));
            if(align_late && d.family==1 && d.group_size!=24 && d.group_size!=28)
                d.nint_late_bits=-1; // The ordinary routed kernel uses four-value chunks here.
            else if(align_late && d.family==1 && nint_pools==1 && (d.group_size==24 || d.group_size==28)) {
                const auto& w=runtime.pools[p].nint;
                const auto direct_option=mfq::cuda::runtime_options::nint_direct_pool();
                const int width=direct_option && !*direct_option?d.input_width:d.groups*d.group_size;
                if(w.bits>=4 && w.bits<=6 && nint_late_scale_enabled(width,int(w.bits),input.get_device()))
                    d.nint_late_bits=int(w.bits);
            }
            if(d.family==2)batch_.nvq_format_mask|=1u<<d.format;
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
        const auto dense_math_option=mfq::cuda::runtime_options::mfe_shared_dense_math();
        batch_.shared_dense_math=dense_math_option && *dense_math_option;
        const int shared_input_projection=ungated_shared?1:0;
        if(!shared[shared_input_projection] || !shared[1] || !shared[2] ||
            !shared_gate.defined() || shared_gate.numel()!=batch_.tokens)
            throw std::invalid_argument("two-stage MFE shared projection incomplete");
        shared_short_encoding_=std::all_of(shared.begin(),shared.end(),[](const NintWeight* w) {
            return !w || w->gs==24 || w->gs==28 || w->gs==32;
        });
        batch_.shared_intermediate=int(shared[1]->out);
        if((!ungated_shared && shared[0]->out!=shared[1]->out) ||
            shared[shared_input_projection]->neuron_len!=batch_.input_width ||
            shared[1]->neuron_len!=batch_.input_width || shared[2]->neuron_len!=shared[1]->out ||
            shared[2]->out<batch_.output_width)throw std::invalid_argument("two-stage MFE shared geometry invalid");
        for(int p=0;p<3;++p) {
            // An ungated shared expert has no gate; reuse Up's descriptor so the
            // packed layout stays three slots while the kernel reads one bank.
            const NintWeight& source=*shared[p?p:(ungated_shared?1:0)];
            auto d=nint_view(source,int(source.out),1,owners_);d.expert_local=nullptr;
            const auto small_whole=mfq::cuda::runtime_options::mfe_shared_small_whole();
            const bool small_down=p==2 && d.group_size==24 && d.groups<=32 &&
                (small_whole?*small_whole:(batch_.tokens==1 || batch_.tokens==5 || batch_.tokens==6));
            d.nint_whole=nint_whole && (d.group_size==28 || (p<2 && d.group_size==24) || small_down);
            const auto grouped=mfq::cuda::runtime_options::nint_group_dot();
            if(batch_.shared_dense_math && source.aligned_q8)
                d.dense_reduction_warps=-1; // Dense q8 uses four-value chunk accumulation.
            else if(batch_.shared_dense_math && (!grouped || *grouped) &&
                !mfq::cuda::runtime_options::nint_dense_reference() && !source.aligned_q8 &&
                (d.group_size==24 || d.group_size==28) &&
                (reinterpret_cast<std::uintptr_t>(d.fields[0])&3u)==0)
                d.dense_reduction_warps=d.groups<=32?1:4;
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
    batch_.default_math=std::all_of(views.begin(),views.end(),[](const View& d) {
        return d.dense_reduction_warps==0 && d.nint_late_bits==0;
    });
    constexpr uint32_t compact_formats=(1u<<5)|(1u<<10)|(1u<<11)|(1u<<12)|(1u<<13)|(1u<<14)|(1u<<15);
    batch_.compact_nvq=(batch_.nvq_format_mask&compact_formats) &&
        std::all_of(views.begin(),views.end(),[](const View& d) {
            if(d.family!=2 || !(compact_formats&(1u<<d.format)))return true;
            return mfq::cuda::mfe_nvq_dense_narrow_eligible(d);
        });
    batch_.e8_narrow=(batch_.nvq_format_mask&((1u<<5)|(1u<<13)|(1u<<14))) &&
        std::all_of(views.begin(),views.end(),[](const View& d) {
            return mfq::cuda::mfe_e8_narrow_view_eligible(d);
        });
    host_views_=views;host_cohorts_=mapping;
    if(combined_gate_up_) {
        if(projections[0]->out_per_expert!=2*batch_.intermediate || batch_.intermediate%2)
            throw std::invalid_argument("combined MFE Gate/Up requires two equal aligned row ranges");
        // Bounds for specialized decoders describe an entire projection. The
        // combined row view retains its full expert stride but computes half.
        batch_.compact_nvq=false;batch_.e8_narrow=false;
        for(int i=0;i<batch_.projection_counts[1];++i) {
            const int index=batch_.projection_offsets[1]+i;
            views[index]=projection_view(1,views[index]);
        }
    }
    constexpr uint32_t grouped_formats=0xfd22u;
    const bool grouped_math=!combined_gate_up_ && batch_.default_math &&
        !batch_.shared_dense_math && !batch_.warp_multi_sum &&
        !(batch_.nvq_format_mask&~grouped_formats) &&
        std::all_of(views.begin(),views.end(),[](const View& d) {
            return d.family==2 || d.family==3 || (d.family==1 &&
                (d.group_size==24 || d.group_size==28 || d.group_size==32));
        });
    batch_.token_group_compact=grouped_math &&
        (!(batch_.nvq_format_mask&compact_formats) || batch_.compact_nvq);
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
    const int pairs=batch_.tokens*batch_.routes;
    if(pairs<=mfq::cuda::kMfeExpertPlanMaxPairs) {
        expert_plan_storage_=tb::empty({3*pairs+2*batch_.experts+1},input.options().dtype(tb::kInt32));
        auto* scratch=expert_plan_storage_.data_ptr<int32_t>();
        expert_plan_={scratch,scratch+pairs,scratch+2*pairs,scratch+2*pairs+batch_.experts,
            scratch+2*pairs+2*batch_.experts,scratch+3*pairs+2*batch_.experts};
    }
    // Pure NINT speculative batches retain the measured faster routed kernels;
    // independent eight-row decode tiles may request expert reuse explicitly.
    const bool has_mxfp4=std::any_of(views.begin(),views.end(),[](const View& view){return view.family==3;});
    if(batch_.tokens>=2 && batch_.tokens<=32 &&
       (batch_.nvq_format_mask || has_mxfp4 || batch_.tokens>8 || token_group_capacity_hint>=4)) {
        // Three/four-row compact batches reuse decoded weights across the
        // complete small window. Other shapes retain their previous capacity.
        const auto* wide_compact=std::getenv("MFQ_MFE_TOKEN_WIDE_COMPACT");
        const bool wide_default=(!wide_compact || wide_compact[0]!='0') && grouped_math &&
            (batch_.tokens==3 || batch_.tokens==4);
        batch_.token_group_capacity=batch_.tokens<=8?(wide_default?4:2):8;
        if(token_group_capacity_hint==4 || token_group_capacity_hint==8)
            batch_.token_group_capacity=token_group_capacity_hint;
        if(const auto capacity=mfq::cuda::runtime_options::mfe_token_group_capacity()) {
            if(*capacity>mfq::cuda::kMfeSharedGroupEncoding)
                batch_.token_group_capacity=shared[0] && batch_.tokens>=3 && batch_.tokens<=8?*capacity:2;
            else if(*capacity>0)batch_.token_group_capacity=*capacity;
            else if(batch_.tokens>=3 && batch_.tokens<=6)
                batch_.token_group_capacity=batch_.tokens<=4?4:8;
        }
        const auto* split_shared=std::getenv("MFQ_MFE_SPLIT_SHARED_GROUPS");
        split_shared_groups_=shared[0] && batch_.token_group_capacity==4 && batch_.tokens<=8 &&
            split_shared && split_shared[0]!='0';
        const int plan_capacity=split_shared_groups_ || batch_.token_group_capacity>mfq::cuda::kMfeSharedGroupEncoding?
            2:batch_.token_group_capacity;
        batch_.token_groups=batch_.tokens*batch_.routes+
            (shared[0]?(batch_.tokens+plan_capacity-1)/plan_capacity:0);
        token_group_counts_=tb::empty({batch_.token_groups},input.options().dtype(tb::kInt32));
        token_group_slots_=tb::empty({batch_.token_groups,8},input.options().dtype(tb::kInt32));
        batch_.token_group_counts=token_group_counts_.data_ptr<int32_t>();
        batch_.token_group_slots=token_group_slots_.data_ptr<int32_t>();
        const auto aligned_gu=mfq::cuda::runtime_options::mfe_token_aligned_gu();
        // Short grouped mixed batches are dominated by singleton/pair groups.
        // Their smaller independent Gate/Up CTAs outperform the aligned CTA,
        // including its fused quantizer. Keep the existing explicit override.
        const bool short_grouped=batch_.tokens==4 && grouped_math && batch_.nvq_format_mask;
        batch_.token_group_aligned_gu=batch_.tokens<=8 &&
            (aligned_gu?*aligned_gu:batch_.tokens==4 && !short_grouped);
        if(short_grouped && !aligned_gu && !batch_.token_group_aligned_gu) {
            static std::atomic<bool> reported{false};
            if(!reported.exchange(true,std::memory_order_relaxed))std::fprintf(stderr,
                "mfe_short_gu_dispatch scope=first_eligible_batch enabled=1 tokens=4 aligned=0 capacity=%d mask=%u\n",
                batch_.token_group_capacity,batch_.nvq_format_mask);
        }
        const auto aligned_down=[](const View& d) {
            return d.group_size>=2 && d.group_size<=32 && d.group_size%2==0;
        };
        for(int i=0;i<batch_.projection_counts[2];++i)
            batch_.token_group_aligned_gu &= aligned_down(views[batch_.projection_offsets[2]+i]);
        if(shared[0])batch_.token_group_aligned_gu &= aligned_down(views.back());
        // Shared-only wide kernels use the separate hidden quantizer. Their
        // CTAs own individual outputs rather than complete quantization groups.
        batch_.token_group_aligned_gu &= batch_.token_group_capacity<=mfq::cuda::kMfeSharedGroupEncoding;
        if(batch_.token_group_capacity>mfq::cuda::kMfeSharedGroupEncoding) {
            static std::atomic<bool> reported{false};
            if(!reported.exchange(true,std::memory_order_relaxed))std::fprintf(stderr,
                "mfe_shared_group_dispatch scope=first_encoded_batch enabled=1 tokens=%d routed_capacity=2 shared_capacity=%d\n",
                batch_.tokens,batch_.token_group_capacity-mfq::cuda::kMfeSharedGroupEncoding);
        }
        const auto compact_plan=mfq::cuda::runtime_options::mfe_token_compact_plan();
        if(!compact_plan || *compact_plan) {
            token_group_buckets_=tb::empty({4,batch_.token_groups+1},input.options().dtype(tb::kInt32));
            batch_.token_group_buckets=token_group_buckets_.data_ptr<int32_t>();
        }
        resident_pairs_=tb::empty({batch_.tokens,batch_.routes+(shared[0]?1:0),batch_.output_width},input.options());
        batch_.resident_pairs=reinterpret_cast<__half*>(resident_pairs_.data_ptr());
        const auto* profile=std::getenv("MFQ_PROFILE_SSD_EXPERTS");
        const auto* enabled=std::getenv("MFQ_MFE_TOKEN_GROUPS");
        const auto* group_dot=std::getenv("MFQ_MFE_GROUP_DOT");
        if(profile && profile[0]=='1' && grouped_math && (!enabled || enabled[0]!='0') &&
                (!group_dot || group_dot[0]!='0') && std::any_of(views.begin(),views.end(),[](const View& d) {
                    return d.family==2 && (compact_formats&(1u<<d.format)) &&
                        !d.e8_dense_groups && !d.d4_dense_groups;
                })) {
            static std::atomic<bool> reported{false};
            if(!reported.exchange(true,std::memory_order_relaxed))std::fprintf(stderr,
                "mfe_raw_nvq_token_reuse_dispatch scope=first_eligible_batch enabled=1 tokens=%d capacity=%d mask=%u expanded=0\n",
                batch_.tokens,batch_.token_group_capacity,batch_.nvq_format_mask);
        }
    }
}
void MfeFfnRuntime::cpu_results(const tb::Tensor& kinds,const tb::Tensor& pairs,const uint32_t* ready) {
    if(!kinds.is_cuda() || !pairs.is_cuda() || kinds.get_device()!=input_.get_device() ||
        pairs.get_device()!=input_.get_device() || kinds.numel()!=batch_.experts || kinds.scalar_type()!=tb::kInt32 || !kinds.is_contiguous() ||
        pairs.dim()!=3 || pairs.size(0)!=batch_.tokens || pairs.size(1)!=batch_.routes || pairs.size(2)!=batch_.output_width ||
        pairs.scalar_type()!=tb::kFloat16 || !pairs.is_contiguous())throw std::invalid_argument("two-stage MFE CPU result contract mismatch");
    kinds_=kinds;cpu_=pairs;batch_.kinds=kinds.data_ptr<int32_t>();batch_.cpu_pairs=reinterpret_cast<const __half*>(pairs.data_ptr());batch_.cpu_ready=ready;
}
void MfeFfnRuntime::bind_window_inputs(const tb::Tensor& input,const tb::Tensor& ids,const tb::Tensor& weights) {
    if(!mfq::cuda::DecodeWindow::recording() ||
        input.sizes()!=input_.sizes() || ids.sizes()!=ids_.sizes() || weights.sizes()!=weights_.sizes() ||
        !input.is_cuda() || !ids.is_cuda() || !weights.is_cuda() ||
        input.device()!=input_.device() || ids.device()!=ids_.device() || weights.device()!=weights_.device() ||
        input.scalar_type()!=tb::kFloat16 || ids.scalar_type()!=tb::kInt32 || weights.scalar_type()!=tb::kFloat32 ||
        !input.is_contiguous() || !ids.is_contiguous() || !weights.is_contiguous())
        throw std::invalid_argument("two-stage MFE window input view contract mismatch");
    // Graph capture records the producer's addresses. Do not retain its Tensor
    // owners: warmup must release temporary allocations at the same point as
    // capture, allowing the graph pool to reproduce their lifetimes.
    batch_.input=reinterpret_cast<const __half*>(input.data_ptr());
    batch_.ids=ids.data_ptr<int32_t>();batch_.route_weights=weights.data_ptr<float>();
}
void MfeFfnRuntime::use_staged_inputs() {
    batch_.input=reinterpret_cast<const __half*>(input_.data_ptr());
    batch_.ids=ids_.data_ptr<int32_t>();batch_.route_weights=weights_.data_ptr<float>();
}
void MfeFfnRuntime::bind_shared_gate(const tb::Tensor& gate) {
    if(!gate.defined()) {
        bound_shared_gate_={};batch_.shared_gate=shared_gate_.defined()?shared_gate_.data_ptr():nullptr;
        return;
    }
    if(!shared_gate_.defined() || !gate.defined() || !gate.is_contiguous() ||
        gate.device()!=shared_gate_.device() || gate.scalar_type()!=shared_gate_.scalar_type() ||
        gate.numel()!=batch_.tokens)throw std::invalid_argument("MFE shared gate view mismatch");
    // Each decode-window cell owns its runtime and this graph allocation.
    // Retaining the tensor prevents graph scratch from recycling the result.
    bound_shared_gate_=gate;batch_.shared_gate=bound_shared_gate_.data_ptr();
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
mfq::cuda::MfePackedProjection MfeFfnRuntime::projection_view(int p,View view)const {
    if(!combined_gate_up_ || p!=1)return view;
    const auto shift=[&](int field,int64_t bytes) {
        if(view.fields[field])view.fields[field]=static_cast<const uint8_t*>(view.fields[field])+bytes;
    };
    const int64_t rows=batch_.intermediate;
    if(view.family==1) {
        // Packed bit offsets stay absolute within the original expert payload.
        shift(1,rows);shift(2,rows*sizeof(int64_t));
        shift(3,rows*view.groups);shift(4,rows*view.groups);
        shift(5,rows*sizeof(float));shift(6,rows*sizeof(float));
    } else {
        for(int f=0;f<3;++f) {
            if(view.local_experts<=0 || view.sizes[f]%(2*view.local_experts))
                throw std::invalid_argument("combined NVQ row range is not byte aligned");
            shift(f,view.sizes[f]/view.local_experts/2);
        }
        shift(3,rows*sizeof(float));
        if(view.fields[6])shift(6,view.decode_record_bytes/view.local_experts/2);
    }
    return view;
}
mfq::cuda::MfePackedProjection MfeFfnRuntime::expert_view(int p,int expert,
        const std::vector<int64_t>& bytes,int slot)const {
    if(p<0 || p>2 || expert<0 || expert>=batch_.experts)throw std::invalid_argument("MFE transfer projection/expert invalid");
    // Ungated banks have no Gate descriptor; Gate transfers alias Up.
    if(ungated_experts_ && p==0)p=1;
    const int pool=host_cohorts_[p*batch_.experts+expert];
    if(pool<0 || pool>=batch_.projection_counts[p])throw std::invalid_argument("MFE transfer cohort missing");
    auto view=host_views_[batch_.projection_offsets[p]+pool];
    // The unfused transfer runtime contains both staged and retained NINT
    // pools, so it does not select the single-pool late-scale hint. This
    // also applies to resident fields retained in a partially cold bundle.
    if(view.nint_late_bits>0)view.nint_late_bits=0;
    if(bytes.size()!=std::size_t(view.family==1?7:view.family==3?2:4))throw std::invalid_argument("MFE transfer field count differs");
    for(std::size_t f=0;f<bytes.size();++f)view.fields[f]=slot>=0?
        static_cast<const unsigned char*>(view.fields[f])+int64_t(slot)*bytes[f]:nullptr;
    if(view.family==2)for(int i=0;i<3;++i)view.sizes[i]=bytes[i];
    if(view.family==3)for(int i=0;i<2;++i)view.sizes[i]=bytes[i];
    if(view.family==1) {
        view.sizes[0]=bytes[0];
        if(view.q_expert_stride>bytes[0])throw std::invalid_argument("MFE transferred NINT payload is shorter than its stride");
    }
    // Transferred canonical fields are matrix-local; optional resident record
    // copies have a different row origin and stay out of this transfer view.
    if(view.family==2){view.fields[6]=nullptr;view.decode_record_bytes=0;}
    view.expert_local=nullptr;view.local_experts=1;view.fallback=-1;
    if(batch_.e8_narrow && !mfq::cuda::mfe_e8_narrow_view_eligible(view,false))
        throw std::invalid_argument("MFE transferred E8 buffers violate narrow metadata contract");
    if(batch_.compact_nvq && view.family==2 &&
            ((1u<<view.format)&((1u<<5)|(1u<<10)|(1u<<11)|(1u<<12)|(1u<<13)|(1u<<14)|(1u<<15))) &&
            !mfq::cuda::mfe_nvq_dense_narrow_eligible(view,false))
        throw std::invalid_argument("MFE transferred compact buffers violate local decoder contract");
    return view;
}
void MfeFfnRuntime::trace_dispatch() {
    if(dispatch_trace_reported_)return;
    dispatch_trace_reported_=true;
    const auto trace=mfq::cuda::runtime_options::trace_mfe_dispatch_requested();
    if(!trace || !*trace)return;
    const auto grouped=mfq::cuda::runtime_options::mfe_group_dot();
    const bool group_dot=!grouped || *grouped;
    const bool gu_requested=mfq::cuda::mfe_ffn_e8_narrow_requested(batch_,false);
    const bool down_requested=mfq::cuda::mfe_ffn_e8_narrow_requested(batch_,true);
    const bool eligible=batch_.e8_narrow && batch_.default_math && group_dot;
    std::fprintf(stderr,"mfe_compact_dispatch runtime=%p scope=first_prepare bounds=%d gu_requested=%d down_requested=%d mask=%u async=%d tokens=%d token_capacity=%d\n",
        static_cast<void*>(this),int(batch_.compact_nvq),
        int(mfq::cuda::mfe_ffn_compact_requested(batch_,false)),
        int(mfq::cuda::mfe_ffn_compact_requested(batch_,true)),batch_.nvq_format_mask,int(batch_.plan_ready!=nullptr),
        batch_.tokens,batch_.token_group_capacity);
    // Report the host launch decision after all asynchronous resources have
    // been bound. One record describes a runtime, not token/route execution
    // counts; graph replay does not call this method again.
    std::fprintf(stderr,"mfe_e8_dispatch runtime=%p scope=first_prepare tokens=%d routes=%d async=%d capture=%d mask=%u bounds=%d default_math=%d grouped=%d gu_requested=%d down_requested=%d gu_selected=%d down_selected=%d\n",
        static_cast<void*>(this),batch_.tokens,batch_.routes,int(batch_.plan_ready!=nullptr),
        int(mfq::cuda::DecodeWindow::recording()!=nullptr),batch_.nvq_format_mask,int(batch_.e8_narrow),
        int(batch_.default_math),int(group_dot),int(gu_requested),int(down_requested),
        int(eligible && gu_requested),int(eligible && down_requested));
    for(int format:{5,13,14}) {
        int count=0,rejected=0,max_experts=0;
        for(const auto& d:host_views_)if(d.family==2 && d.format==format) {
            ++count;rejected+=!mfq::cuda::mfe_e8_narrow_view_eligible(d);
            max_experts=std::max(max_experts,d.local_experts);
        }
        if(count)std::fprintf(stderr,"mfe_e8_pool_proof runtime=%p format=%d views=%d rejected=%d max_experts=%d\n",
            static_cast<void*>(this),format,count,rejected,max_experts);
    }
}
void MfeFfnRuntime::enable_cached_dispatch(const mfq::cuda::MfeCachedExpert* mapped,
        const std::array<std::vector<std::vector<int64_t>>,3>& bytes,uint32_t* snapshot_ready) {
    if(!mapped || !snapshot_ready || cached_dispatch_.mapped || !batch_.resident_plan_overlap || batch_.tokens*batch_.routes>256)
        throw std::invalid_argument("MFE cached resident dispatch contract invalid");
    std::vector<View> views;
    int offsets[3]{},counts[3]{},original_offsets[3]{};
    for(int p=0;p<3;++p) {
        const int count=batch_.projection_counts[p];
        original_offsets[p]=batch_.projection_offsets[p];
        // An ungated bank stores only Up and Down: projection 0 keeps its empty
        // layout and has no cohort count or per-expert slots to install.
        if(ungated_experts_ && p==0) {
            offsets[p]=int(views.size());counts[p]=0;cached_dispatch_.primary_counts[p]=0;continue;
        }
        int64_t capacity=0;
        for(int c=0;c<count;++c)capacity+=host_views_[batch_.projection_offsets[p]+c].local_experts;
        cached_dispatch_.primary_incomplete|=capacity<batch_.experts;
        if(bytes[p].size()!=std::size_t(count))throw std::invalid_argument("MFE cache cohort geometry differs");
        offsets[p]=int(views.size());counts[p]=count+batch_.experts;cached_dispatch_.primary_counts[p]=count;
        for(int c=0;c<count;++c)views.push_back(projection_view(p,host_views_[batch_.projection_offsets[p]+c]));
        for(int e=0;e<batch_.experts;++e) {
            const int c=host_cohorts_[p*batch_.experts+e];
            views.push_back(unsigned(c)<unsigned(count)?projection_view(p,expert_view(p,e,bytes[p].at(c),-1)):View{});
        }
    }
    const int shared_offset=int(views.size());
    if(batch_.shared)views.insert(views.end(),host_views_.end()-3,host_views_.end());
    descriptors_=upload(views);primary_cohorts_=cohorts_;
    cohorts_=tb::tensor(host_cohorts_).to(tb::kCUDA);
    for(int p=0;p<3;++p){batch_.projection_offsets[p]=offsets[p];batch_.projection_counts[p]=counts[p];}
    batch_.projections=reinterpret_cast<const View*>(descriptors_.data_ptr());
    batch_.cohort_by_expert=cohorts_.data_ptr<int32_t>();
    if(batch_.shared)batch_.shared=batch_.projections+shared_offset;
    // Keep the original, unshifted Up templates for expert_view().
    // expert_view is used by initialization and optional diagnostics after this
    // call. Its cohort indices still refer to the original primary templates.
    auto original=std::move(host_views_);
    for(int p=0;p<3;++p)for(int c=0;c<cached_dispatch_.primary_counts[p];++c)
        views[offsets[p]+c]=original[original_offsets[p]+c];
    host_views_=std::move(views);
    cached_dispatch_.mapped=mapped;cached_dispatch_.primary_cohorts=primary_cohorts_.data_ptr<int32_t>();
    cached_dispatch_.projections=reinterpret_cast<View*>(descriptors_.data_ptr());
    cached_snapshot_tags_=tb::zeros({batch_.experts},input_.options().dtype(tb::kInt64));
    cached_dispatch_.snapshot_tags=reinterpret_cast<uint64_t*>(cached_snapshot_tags_.data_ptr<int64_t>());
    cached_dispatch_.snapshot_ready=snapshot_ready;
    if(batch_.tokens==1 && batch_.routes<=32)
        cached_transfer_routes_=tb::empty({batch_.routes+2},input_.options().dtype(tb::kInt32));
}
bool MfeFfnRuntime::try_fuse_prepare_shared_gate(const NintWeight& w,const tb::Tensor& output) {
    if(!cached_dispatch_.mapped || !batch_.shared || batch_.tokens<1 || batch_.tokens>6 ||
        !w.aligned_q8 || w.q8_zero || w.out!=1 || w.gs!=48 || w.ng<1 || w.ng>64 ||
        w.neuron_len!=batch_.input_width || w.neuron_len>w.ng*48 || !output.is_cuda() ||
        output.scalar_type()!=tb::kFloat16 || !output.is_contiguous() || output.numel()!=batch_.tokens ||
        output.device()!=input_.device())return false;
    if(const auto* option=std::getenv("MFQ_SHARED_GATE_PREPARE");option && option[0]=='0')return false;
    if(const auto* option=std::getenv("MFQ_NINT_Q8_DECODE");option && std::atoi(option)!=2)return false;
    if(const auto* option=std::getenv(batch_.tokens==1?"MFQ_NINT_Q8_ORDERED":"MFQ_NINT_Q8_SMALL_WARPS");option && option[0]=='0')return false;
    for(const auto* field:{&w.q_packed,&w.row_q_bit_offsets,&w.sub_scale,&w.sub_min,&w.neuron_scale,&w.neuron_min})
        if(!field->is_cuda() || !field->is_contiguous() || field->device()!=input_.device())return false;
    const auto [qx,xs]=reserve_input_quantization(int(w.ng),48);
    prepared_gate_terms_=tb::empty({batch_.tokens,w.ng,batch_.tokens==1?12:1},input_.options().dtype(tb::kFloat32));
    prepared_gate_completed_=tb::zeros({batch_.tokens},input_.options().dtype(tb::kInt32));
    auto& gate=cached_dispatch_.gate;
    gate.activation=qx.data_ptr<int8_t>();gate.activation_scales=xs.data_ptr<float>();
    gate.packed=w.q_packed.data_ptr<uint8_t>();gate.offset=w.row_q_bit_offsets.data_ptr<int64_t>();
    gate.scales=w.sub_scale.data_ptr<uint8_t>();gate.minima=w.sub_min.data_ptr<uint8_t>();
    gate.outer_scale=w.neuron_scale.data_ptr<float>();gate.outer_minimum=w.neuron_min.data_ptr<float>();
    gate.terms=prepared_gate_terms_.data_ptr<float>();gate.completed=reinterpret_cast<uint32_t*>(prepared_gate_completed_.data_ptr<int32_t>());
    gate.output=reinterpret_cast<__half*>(output.data_ptr());gate.groups=int(w.ng);
    int prefix=0,gate_end=0;
    for(const auto& [geometry,tensors]:input_quantizers_) {
        prefix+=batch_.tokens*geometry.first;
        if(tensors.first.data_ptr()==qx.data_ptr())gate_end=prefix;
    }
    gate.quantization_total=prefix;gate.quantization_rotation=gate_end%prefix;
    for(const auto* field:{&w.q_packed,&w.row_q_bit_offsets,&w.sub_scale,&w.sub_min,&w.neuron_scale,&w.neuron_min})owners_.push_back(*field);
    shared_gate_=output;bind_shared_gate(output);
    return true;
}
void MfeFfnRuntime::prepare(int32_t* published_ids,uint32_t* ready){
    trace_dispatch();
    if(ready && published_ids)start_shared(true);
    auto prepared=batch_;
    const bool shared_plan=batch_.token_group_capacity>mfq::cuda::kMfeSharedGroupEncoding;
    if(shared_plan)prepared.token_group_capacity=2;
    if(const auto setting=mfq::cuda::runtime_options::mfe_token_groups())
        if(!*setting)prepared.token_group_counts=nullptr;
    const auto* warp_prepare=std::getenv("MFQ_MFE_PREPARE_WARP");
    const bool use_warp=!warp_prepare || warp_prepare[0]!='0';
    // Small windows retain linear grouping to avoid radix overhead on repeated routes.
    // The explicit option keeps both paths available for same-process probes.
    const bool planned=mfq::cuda::runtime_options::mfe_expert_plan().value_or(batch_.tokens*batch_.routes>=128);
    const auto plan=planned && (prepared.token_group_counts || (ready && cached_dispatch_.mapped))?
        expert_plan_:mfq::cuda::MfeExpertPlan{};
    const auto* quantization=reinterpret_cast<const mfq::cuda::MfeInputQuantization*>(quantization_.data_ptr());
    if(ready && cached_dispatch_.mapped) {
        auto cache=cached_dispatch_;
        // The existing shared event also protects its scalar gate. Its
        // ordered readout is needed at final combination, after join_shared().
        if(shared_in_flight_ && early_shared_gate_.activation)cache.gate={};
        mfq::cuda::mfe_ffn_prepare_publish_cached(prepared,quantization,quantization_count_,quantization_groups_,
            published_ids,ready,cache,mfq_current_cuda_stream(),plan);
    }
    else if(ready)mfq::cuda::mfe_ffn_prepare_publish(prepared,quantization,quantization_count_,quantization_groups_,
        published_ids,ready,mfq_current_cuda_stream(),plan);
    else if(use_warp)mfq::cuda::mfe_ffn_prepare_warp(prepared,quantization,quantization_count_,quantization_groups_,
        mfq_current_cuda_stream(),plan);
    else mfq::cuda::mfe_ffn_prepare(prepared,quantization,quantization_count_,quantization_groups_,
        mfq_current_cuda_stream());
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    if((shared_plan || split_shared_groups_) && prepared.token_group_counts) {
        auto shared_batch=batch_;
        if(split_shared_groups_)shared_batch.token_group_capacity=mfq::cuda::kMfeSharedGroupEncoding+2;
        mfq::cuda::mfe_ffn_token_group_shared_plan(shared_batch,mfq_current_cuda_stream());
        MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    }
}
std::pair<tb::Tensor,tb::Tensor> MfeFfnRuntime::reserve_input_quantization(int groups,int group_size) {
    if(group_size<4 || group_size>64 || groups!=(batch_.input_width+group_size-1)/group_size)
        throw std::invalid_argument("MFE additional input quantization geometry is invalid");
    const auto key=std::make_pair(groups,group_size);
    if(const auto found=input_quantizers_.find(key);found!=input_quantizers_.end())return found->second;
    cudaStreamCaptureStatus capture;
    MFQ_CUDA_CHECK(cudaStreamIsCapturing(mfq_current_cuda_stream(),&capture));
    if(capture!=cudaStreamCaptureStatusNone)throw std::logic_error("MFE input quantization must be reserved before capture");
    const auto q=tb::empty({batch_.tokens,groups*group_size},input_.options().dtype(tb::kInt8));
    const auto s=tb::empty({batch_.tokens,groups},input_.options().dtype(tb::kFloat32));
    input_quantizers_.emplace(key,std::make_pair(q,s));
    std::vector<mfq::cuda::MfeInputQuantization> views;
    quantization_groups_=0;
    for(const auto& [geometry,tensors]:input_quantizers_) {
        views.push_back({tensors.first.data_ptr<int8_t>(),tensors.second.data_ptr<float>(),geometry.first,geometry.second});
        quantization_groups_+=geometry.first;
    }
    quantization_=upload(views);quantization_count_=int(views.size());
    return {q,s};
}
void MfeFfnRuntime::gate_up(){mfq::cuda::mfe_ffn_gate_up(batch_,mfq_current_cuda_stream());MFQ_CUDA_KERNEL_LAUNCH_CHECK();}
void MfeFfnRuntime::resident(){
    start_shared(false);
    mfq::cuda::mfe_ffn_resident(arithmetic_batch(),mfq_current_cuda_stream());
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
}
void MfeFfnRuntime::start_shared(bool early){
    if(shared_in_flight_)return;
    if(early) {
        const auto* setting=std::getenv("MFQ_MFE_SHARED_EARLY");
        if(batch_.tokens!=1 || !batch_.plan_ready || (setting && setting[0]=='0'))return;
    }
    const auto* option=std::getenv("MFQ_MOE_SHARED_STREAM");
    const auto grouped=mfq::cuda::runtime_options::mfe_group_dot();
    const auto token_groups=mfq::cuda::runtime_options::mfe_token_groups();
    const int shared_capacity=split_shared_groups_?2:batch_.token_group_capacity>mfq::cuda::kMfeSharedGroupEncoding?
        batch_.token_group_capacity-mfq::cuda::kMfeSharedGroupEncoding:batch_.token_group_capacity;
    const bool grouped_shared=batch_.token_group_counts && shared_capacity>0 && shared_capacity<=8 &&
        (!token_groups || *token_groups);
    if((batch_.tokens==1 || grouped_shared) && batch_.shared && batch_.resident_pairs && batch_.default_math && (!grouped || *grouped) &&
            (!option || option[0]!='0')) {
        if(early && !early_shared_descriptors_.defined()) {
            cudaStreamCaptureStatus capture;
            MFQ_CUDA_CHECK(cudaStreamIsCapturing(mfq_current_cuda_stream(),&capture));
            if(capture!=cudaStreamCaptureStatusNone)return;
            std::array<View,3> views;
            std::copy_n(host_views_.end()-3,3,views.begin());
            std::map<std::pair<int,int>,std::pair<tb::Tensor,tb::Tensor>> inputs;
            std::vector<mfq::cuda::MfeInputQuantization> quantizers;
            for(int p=0;p<2;++p) {
                auto& d=views[p];const auto key=std::make_pair(d.groups,d.group_size);
                auto found=inputs.find(key);
                if(found==inputs.end()) {
                    auto q=tb::empty({d.groups*d.group_size},input_.options().dtype(tb::kInt8));
                    auto s=tb::empty({d.groups},input_.options().dtype(tb::kFloat32));
                    found=inputs.emplace(key,std::make_pair(q,s)).first;
                    early_shared_inputs_.push_back(q);early_shared_inputs_.push_back(s);
                    quantizers.push_back({q.data_ptr<int8_t>(),s.data_ptr<float>(),d.groups,d.group_size});
                    early_shared_groups_+=d.groups;
                }
                d.fields[7]=found->second.first.data_ptr();d.fields[8]=found->second.second.data_ptr();
            }
            if(cached_dispatch_.gate.activation) {
                early_shared_gate_=cached_dispatch_.gate;
                const auto key=std::make_pair(early_shared_gate_.groups,48);
                auto found=inputs.find(key);
                if(found==inputs.end()) {
                    auto q=tb::empty({early_shared_gate_.groups*48},input_.options().dtype(tb::kInt8));
                    auto s=tb::empty({early_shared_gate_.groups},input_.options().dtype(tb::kFloat32));
                    found=inputs.emplace(key,std::make_pair(q,s)).first;
                    early_shared_inputs_.push_back(q);early_shared_inputs_.push_back(s);
                    quantizers.push_back({q.data_ptr<int8_t>(),s.data_ptr<float>(),early_shared_gate_.groups,48});
                    early_shared_groups_+=early_shared_gate_.groups;
                }
                early_shared_gate_.activation=found->second.first.data_ptr<int8_t>();
                early_shared_gate_.activation_scales=found->second.second.data_ptr<float>();
            }
            early_shared_descriptors_=upload(std::vector<View>(views.begin(),views.end()));
            early_shared_quantizers_=upload(quantizers);early_shared_quantizer_count_=int(quantizers.size());
        }
        if(!shared_stream_) {
            // Initialization is outside graph capture: pipeline warmup owns
            // these resources before recording the steady decode graph.
            shared_stream_=std::make_unique<mfq::cuda::Stream>(input_.get_device());
            shared_fork_=std::make_unique<mfq::cuda::Event>(cudaEventDisableTiming);
            shared_done_=std::make_unique<mfq::cuda::Event>(cudaEventDisableTiming);
        }
        MFQ_CUDA_CHECK(cudaEventRecord(shared_fork_->get(),mfq_current_cuda_stream()));
        MFQ_CUDA_CHECK(cudaStreamWaitEvent(shared_stream_->get(),shared_fork_->get(),0));
        auto shared_batch=batch_;
        if(early) {
            shared_batch.shared=reinterpret_cast<const View*>(early_shared_descriptors_.data_ptr());
            mfq::cuda::mfe_ffn_shared_prepare_input(batch_.input,batch_.input_width,
                reinterpret_cast<const mfq::cuda::MfeInputQuantization*>(early_shared_quantizers_.data_ptr()),
                early_shared_quantizer_count_,early_shared_groups_,shared_stream_->get());
        }
        if(split_shared_groups_)shared_batch.token_group_capacity=mfq::cuda::kMfeSharedGroupEncoding+2;
        const auto* short_option=std::getenv("MFQ_MFE_SHARED_SHORT");
        const bool scalar_gate=early && early_shared_gate_.activation;
        const bool short_shared=!scalar_gate && shared_short_encoding_ && (!short_option || short_option[0]!='0') &&
            mfq::cuda::mfe_ffn_shared_short_resident(shared_batch,shared_stream_->get());
        if(scalar_gate)mfq::cuda::mfe_ffn_shared_resident_with_gate(shared_batch,early_shared_gate_,shared_stream_->get());
        else if(!short_shared)mfq::cuda::mfe_ffn_shared_resident(shared_batch,shared_stream_->get());
        if(short_shared && !shared_short_reported_) {
            shared_short_reported_=true;
            std::fprintf(stderr,"mfe_shared_short_dispatch scope=first_short_shared enabled=1 tokens=%d capacity=%d groups=%d\n",
                batch_.tokens,shared_capacity,(batch_.tokens+shared_capacity-1)/shared_capacity);
        }
        MFQ_CUDA_KERNEL_LAUNCH_CHECK();
        MFQ_CUDA_CHECK(cudaEventRecord(shared_done_->get(),shared_stream_->get()));
        shared_in_flight_=true;
        if(early && !shared_early_reported_) {
            shared_early_reported_=true;
            std::fprintf(stderr,"mfe_shared_early_dispatch scope=first_prepare enabled=1 tokens=1 quantizers=%d groups=%d scalar_gate_async=%d scalar_gate_fused=%d\n",
                early_shared_quantizer_count_,early_shared_groups_,int(early_shared_gate_.activation!=nullptr),int(scalar_gate));
        }
        if(!shared_stream_reported_) {
            shared_stream_reported_=true;
            std::fprintf(stderr,"mfe_shared_stream_dispatch scope=first_resident enabled=1 tokens=%d routes=%d shared_width=%d gu_quant_fused=%d\n",
                batch_.tokens,batch_.routes,batch_.shared_intermediate,int(batch_.tokens==1));
        }
    }
}
void MfeFfnRuntime::join_shared(){
    if(shared_in_flight_)MFQ_CUDA_CHECK(cudaStreamWaitEvent(mfq_current_cuda_stream(),shared_done_->get(),0));
}
void MfeFfnRuntime::wait_transfer(const void* mapped_table,void* snapshot,uint64_t capacity,
        const void* mapped_descriptors,uint64_t descriptor_capacity_words){
    const auto* route_list=std::getenv("MFQ_MFE_TRANSFER_ROUTE_LIST");
    const auto* fused_hidden=std::getenv("MFQ_MFE_TRANSFER_HIDDEN_FUSED");
    cached_transfer_routes_ready_=cached_transfer_routes_.defined() && snapshot && mapped_table &&
        transfer_route_plan_ && mapped_descriptors && (!route_list || route_list[0]!='0') &&
        (!fused_hidden || fused_hidden[0]!='0');
    if(snapshot && (!transfer_route_plan_ || !mapped_table))
        throw std::invalid_argument("MFE transfer snapshot requires its mapped route packet and table");
    if(snapshot && transfer_route_plan_)mfq::cuda::mfe_ffn_wait_transfer_packet_snapshot(batch_,transfer_route_plan_,
        mapped_table,snapshot,capacity,mfq_current_cuda_stream(),mapped_descriptors,
        mapped_descriptors?transfer_buffer().data_ptr():nullptr,descriptor_capacity_words,cached_dispatch_.mapped!=nullptr,
        cached_transfer_routes_ready_?cached_transfer_routes_.data_ptr<int32_t>():nullptr);
    else if(transfer_route_plan_)mfq::cuda::mfe_ffn_wait_transfer_packet(batch_,transfer_route_plan_,mfq_current_cuda_stream());
    else mfq::cuda::mfe_ffn_wait_transfer(batch_,mfq_current_cuda_stream());
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
}
void MfeFfnRuntime::transferred(){
    const auto stream=mfq_current_cuda_stream();
    const auto b=arithmetic_batch();
    if(!cached_dispatch_.mapped || !mfq::cuda::mfe_ffn_compact_transferred_fused(b,stream,
            cached_transfer_routes_ready_?cached_transfer_routes_.data_ptr<int32_t>():nullptr))
        mfq::cuda::mfe_ffn_transferred(b,stream);
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
}
void MfeFfnRuntime::wait_cpu(){mfq::cuda::mfe_ffn_wait_cpu(batch_,mfq_current_cuda_stream());MFQ_CUDA_KERNEL_LAUNCH_CHECK();}
void MfeFfnRuntime::down_reduce_compute(){
    join_shared();
    const auto b=arithmetic_batch();const auto stream=mfq_current_cuda_stream();
    const auto* cached=cached_transfer_routes_ready_?
        reinterpret_cast<const uint32_t*>(cached_transfer_routes_.data_ptr<int32_t>()+batch_.routes+1):nullptr;
    if(!cached || !mfq::cuda::mfe_ffn_compact_cached_down(b,cached,stream))
        mfq::cuda::mfe_ffn_down_reduce_compute(b,stream);
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();shared_in_flight_=false;
}
void MfeFfnRuntime::down_reduce(){wait_cpu();down_reduce_compute();}
