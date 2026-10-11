#pragma once
#include "timing.h"
#include "mfq_cuda_quant_ops.h"
#include "storage/moe_quant_range_source.h"
#include "mfq/model_source.h"
#include <cstring>
#include <filesystem>
#include <map>
#include <memory>
#include <type_traits>

namespace tb=mfq_tensor_backend;
namespace fb=format_bench;
namespace fs=std::filesystem;

static std::vector<tb::Tensor*> fields(MixedMoePool& pool) {
    if(pool.family==MixedMoeFamily::Nint)
        return {&pool.nint.q_packed,&pool.nint.row_q_bits,&pool.nint.row_q_bit_offsets,
            &pool.nint.sub_scale,&pool.nint.sub_min,&pool.nint.neuron_scale,&pool.nint.neuron_min};
    if(pool.family==MixedMoeFamily::Nvq)
        return {&pool.nvq.indices_packed,&pool.nvq.aux_packed,&pool.nvq.sub_scale_packed,
            &pool.nvq.neuron_scale,&pool.nvq.codebook};
    throw std::runtime_error("unsupported released format in benchmark");
}
static std::size_t storage(MixedMoePool& pool) {
    std::size_t bytes=0;
    for(auto* field:fields(pool))if(field->defined() && field!=&pool.nvq.codebook)
        bytes+=field->numel()*field->element_size();
    if(pool.family==MixedMoeFamily::Nint && pool.nint.route_metadata.defined())
        bytes+=pool.nint.route_metadata.numel()*pool.nint.route_metadata.element_size();
    return bytes;
}
static void verify_bench_nvq_records(const NvqWeight& canonical,const NvqWeight& converted) {
    const char* setting=std::getenv("MFQ_BENCH_VERIFY_RECORD_LAYOUT");
    if(!setting || setting[0]=='0')return;
    auto raw=canonical;
    for(auto* field:{&raw.indices_packed,&raw.aux_packed,&raw.sub_scale_packed,&raw.neuron_scale,&raw.codebook})
        *field=field->to(tb::kCUDA).contiguous();
    const auto dequant=[](const NvqWeight& w) {
        return nvq_dequant_cuda(w.indices_packed,w.aux_packed,w.sub_scale_packed,w.neuron_scale,
            w.codebook,w.neuron_len,w.gs,w.sub_bits,w.kernel_format,w.sign_mode).to(tb::kCPU);
    };
    const auto expected_weight=dequant(raw),actual_weight=dequant(converted);
    if(expected_weight.sizes()!=actual_weight.sizes() ||
        std::memcmp(expected_weight.data_ptr(),actual_weight.data_ptr(),actual_weight.numel()*actual_weight.element_size()))
        throw std::runtime_error("extended execution records changed dequantized weight bits");
    CudaExecutionContext execution;double worst=0;
    for(int batch:{1,3,8,16,64}) {
        auto x=tb::tensor(fb::input(batch,int(raw.neuron_len))).reshape({batch,raw.neuron_len}).to(tb::kCUDA,tb::kFloat16);
        auto a=nvq_matmul(execution.profiler,raw,x).to(tb::kCPU,tb::kFloat32);
        auto b=nvq_matmul(execution.profiler,converted,x).to(tb::kCPU,tb::kFloat32);
        double error=0,norm=0,largest=0,max_error=0;
        for(int64_t i=0;i<a.numel();++i) {
            const double av=a.data_ptr<float>()[i],bv=b.data_ptr<float>()[i];
            if(!std::isfinite(bv))throw std::runtime_error("nonfinite extended record matrix output");
            error+=(av-bv)*(av-bv);norm+=av*av;
            largest=std::max(largest,std::abs(av));max_error=std::max(max_error,std::abs(av-bv));
        }
        const double relative=std::sqrt(error/std::max(norm,1e-30));
        if(relative>2e-4 || max_error>largest*2e-3+6e-8)
            throw std::runtime_error("extended records changed matrix result beyond FP32 reduction tolerance");
        worst=std::max(worst,relative);
    }
    std::cout<<"VERIFY_EXEC_RECORDS format="<<converted.kernel_format<<" rows="<<converted.out
        <<" weight_bitexact matrix_checks=5 M=1/3/8/16/64 max_relative_l2="<<worst<<" PASS\n";
}
// Use the production lossless execution-layout conversion while retaining
// CPU cohorts, so rotating-bank sizes count every executed record byte.
static void prepare_bench_execution_layout(MixedMoePool& pool) {
    const char* setting=std::getenv("MFQ_BENCH_EXTENDED_RECORDS");
    if(!setting || setting[0]=='0' || pool.family!=MixedMoeFamily::Nvq ||
       (pool.nvq.kernel_format!=14 && pool.nvq.kernel_format!=15))return;
    auto& w=pool.nvq;NvqCpu c;
    c.format=int(w.format);c.sign_mode=int(w.sign_mode);c.sub_bits=int(w.sub_bits);
    c.gs=int(w.gs);c.neuron_len=int(w.neuron_len);c.out=int(w.out);c.ng=int(w.ng);c.shape=w.shape;
    c.nvec=(c.neuron_len+(c.format==15?3:7))/(c.format==15?4:8);c.nsign=(c.neuron_len+7)/8;
    const auto copy=[](const tb::Tensor& tensor,auto& output) {
        using Value=typename std::decay_t<decltype(output)>::value_type;
        const auto host=tensor.to(tb::kCPU).contiguous();output.resize(host.numel());
        if(!output.empty())std::memcpy(output.data(),host.data_ptr(),output.size()*sizeof(Value));
    };
    copy(w.indices_packed,c.indices_packed);copy(w.aux_packed,c.aux_packed);
    copy(w.sub_scale_packed,c.sub_scale_packed);copy(w.codebook,c.codebook);
    copy(w.neuron_scale.to(tb::kFloat16),c.neuron_scale_h);
    CudaExecutionConfig config;config.nvq_extended_group_exec=true;
    auto converted=to_device_nvq(c,true,config);
    verify_bench_nvq_records(w,converted);
    w=std::move(converted);
    for(auto* field:fields(pool))if(field->defined())*field=field->to(tb::kCPU).contiguous();
    w.workspaces.clear();
}
// E8 execution books are reordered using the cohort's index distribution.
// Convert compatible experts together to retain the production shared book.
static void prepare_bench_execution_layout_batch(std::vector<MixedMoePool>& pools) {
    const char* setting=std::getenv("MFQ_BENCH_EXTENDED_RECORDS");
    if(!setting || setting[0]=='0')return;
    std::vector<bool> used(pools.size());
    for(size_t first=0;first<pools.size();++first)if(!used[first]) {
        const auto& weight=pools[first].nvq;
        if(pools[first].family!=MixedMoeFamily::Nvq ||
            (weight.kernel_format!=14 && weight.kernel_format!=15))continue;
        std::vector<size_t> members;
        for(size_t i=first;i<pools.size();++i)if(!used[i] && pools[i].family==MixedMoeFamily::Nvq) {
            const auto& other=pools[i].nvq;
            if(other.kernel_format==weight.kernel_format && other.neuron_len==weight.neuron_len &&
                other.out==weight.out && other.gs==weight.gs && other.sub_bits==weight.sub_bits &&
                other.sign_mode==weight.sign_mode && other.codebook.sizes()==weight.codebook.sizes() &&
                !std::memcmp(other.codebook.data_ptr(),weight.codebook.data_ptr(),weight.codebook.numel())) {
                members.push_back(i);used[i]=true;
            }
        }
        auto combined=pools[first];auto combined_fields=fields(combined);
        for(size_t f=0;f<combined_fields.size();++f)if(combined_fields[f]!=&combined.nvq.codebook) {
            std::vector<tb::Tensor> parts;
            for(size_t i:members)parts.push_back(*fields(pools[i])[f]);
            *combined_fields[f]=tb::cat(parts,0);
        }
        combined.nvq.out=weight.out*members.size();
        combined.nvq.shape={combined.nvq.out,combined.nvq.neuron_len};
        prepare_bench_execution_layout(combined);
        for(size_t position=0;position<members.size();++position) {
            auto& target=pools[members[position]];
            const auto shape=target.nvq.shape;const auto rows=target.nvq.out;
            target.nvq=combined.nvq;target.nvq.out=rows;target.nvq.shape=shape;
            auto target_fields=fields(target);
            for(size_t f=0;f<target_fields.size();++f)if(target_fields[f]!=&target.nvq.codebook) {
                const auto bytes=combined_fields[f]->numel()/members.size();
                *target_fields[f]=combined_fields[f]->narrow(0,int64_t(position)*bytes,bytes).contiguous();
            }
        }
    }
}
static MixedMoePool upload(const MixedMoePool& source,const tb::Tensor& shared_codebook={},bool dense=true) {
    auto result=source;
    for(auto* field:fields(result))if(field->defined()) {
        if(field==&result.nvq.codebook && shared_codebook.defined())*field=shared_codebook;
        else *field=field->to(tb::kCUDA).contiguous();
    }
    if(dense && result.family==MixedMoeFamily::Nint)result.nint.q_packed=result.nint.q_packed.reshape({-1});
    if(result.family==MixedMoeFamily::Nvq)prepare_nvq_decode_records(result.nvq,true);
    return result;
}
static tb::Tensor decoded(const MixedMoePool& pool) {
    if(pool.family==MixedMoeFamily::Nint) {
        const auto& w=pool.nint;
        return nint_decode_cuda(w.q_packed,w.row_q_bits,w.row_q_bit_offsets,w.sub_scale,w.sub_min,
            w.neuron_scale,w.neuron_min,w.neuron_len,w.gs);
    }
    const auto& w=pool.nvq;
    return nvq_dequant_cuda(w.indices_packed,w.aux_packed,w.sub_scale_packed,w.neuron_scale,
        w.codebook,w.neuron_len,w.gs,w.sub_bits,w.kernel_format,w.sign_mode);
}
static void write_tensor(const fs::path& path,const tb::Tensor& tensor) {
    const auto host=tensor.to(tb::kCPU).to(tb::kFloat32).contiguous();
    const auto* data=host.data_ptr<float>();
    for(int64_t i=0;i<host.numel();++i)if(!std::isfinite(data[i]))
        throw std::runtime_error("nonfinite benchmark value: "+path.string());
    std::ofstream output(path,std::ios::binary);
    output.write(reinterpret_cast<const char*>(data),host.numel()*sizeof(float));
    if(!output)throw std::runtime_error("cannot save benchmark data: "+path.string());
}
