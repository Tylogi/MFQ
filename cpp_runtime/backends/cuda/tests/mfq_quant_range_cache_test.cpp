#include "storage/weight_loader.h"
#include "storage/moe_expert_cache.h"
#include "storage/moe_quant_range_source.h"
#include "cuda_execution.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace tb=mfq_tensor_backend;
namespace {
// Models may support immediate reads without an independently owned reader.
// The loader must keep their existing cache path, while real I/O failures
// continue to propagate rather than switching to whole-pool materialization.
class WithoutOwningReader final : public mfq::ModelSource {
public:
    explicit WithoutOwningReader(std::shared_ptr<const mfq::ModelSource> source):source_(std::move(source)) {}
    const std::vector<std::filesystem::path>& source_paths() const noexcept override { return source_->source_paths(); }
    std::string_view architecture() const noexcept override { return source_->architecture(); }
    const std::unordered_map<std::string,std::string>& metadata() const noexcept override { return source_->metadata(); }
    const std::vector<mfq::TensorMetadata>& tensors() const noexcept override { return source_->tensors(); }
    const mfq::TensorMetadata* find_tensor(std::string_view name) const noexcept override { return source_->find_tensor(name); }
    void read_range_into(std::string_view name,std::uint64_t offset,std::byte* data,std::size_t size) const override {
        source_->read_range_into(name,offset,data,size);
    }
    const std::vector<std::string>& assets() const noexcept override { return source_->assets(); }
    bool has_asset(std::string_view name) const noexcept override { return source_->has_asset(name); }
    std::vector<std::byte> read_asset(std::string_view name) const override { return source_->read_asset(name); }
    std::optional<mfq::ModelGraph> model_graph() const override { return source_->model_graph(); }
private:
    std::shared_ptr<const mfq::ModelSource> source_;
};

std::vector<float> read(const std::filesystem::path& path,std::size_t count) {
    std::ifstream stream(path,std::ios::binary);
    std::vector<float> values(count);
    stream.read(reinterpret_cast<char*>(values.data()),count*sizeof(float));
    if (!stream || stream.peek()!=std::char_traits<char>::eof())
        throw std::runtime_error("invalid independent FP32 fixture: "+path.string());
    return values;
}

std::int64_t slot_bytes(const MixedMoePool& pool) {
    std::vector<tb::Tensor> fields;
    if (pool.family==MixedMoeFamily::Nint)
        fields={pool.nint.q_packed,pool.nint.row_q_bits,pool.nint.row_q_bit_offsets,
            pool.nint.sub_scale,pool.nint.sub_min,pool.nint.neuron_scale,pool.nint.neuron_min};
    else fields={pool.nvq.indices_packed,pool.nvq.aux_packed,pool.nvq.sub_scale_packed,pool.nvq.neuron_scale};
    std::int64_t total=0;
    for (const auto& field:fields) total+=field.numel()*field.element_size();
    return total;
}

MoeRoutePlan route(const std::vector<std::int32_t>& ids,int tokens,int routes) {
    return build_moe_route_plan(tb::tensor(ids,tb::TensorOptions().dtype(tb::kInt32))
        .reshape({tokens,routes}).to(tb::kCUDA),3);
}

void verify_cpu(const tb::Tensor& value,const std::vector<float>& reference,
        const std::vector<std::int32_t>& ids,int tokens,int routes,int output) {
    if (!value.is_cuda() || value.dim()!=3 || value.size(0)!=tokens ||
            value.size(1)!=routes || value.size(2)!=output)
        throw std::runtime_error("CPU cold projection output placement/shape mismatch");
    MFQ_CUDA_CHECK(cudaDeviceSynchronize());
    const auto actual=value.to(tb::kCPU).to(tb::kFloat32).contiguous();
    for (int sample=0; sample<tokens; ++sample) for (int selected=0; selected<routes; ++selected)
        for (int row=0; row<output; ++row) {
            const double expected=reference[(sample*3+ids[sample*routes+selected])*output+row];
            const double got=actual.data_ptr<float>()[(sample*routes+selected)*output+row];
            if (!std::isfinite(got) || std::abs(got-expected)>1e-5+1e-3*std::abs(expected))
                throw std::runtime_error("selected CPU expert differs from independent FP64 oracle at "+
                    std::to_string(sample)+","+std::to_string(selected)+","+std::to_string(row)+
                    ", error="+std::to_string(std::abs(got-expected)));
        }
}

void verify_gpu(const tb::Tensor& cached,const tb::Tensor& original) {
    MFQ_CUDA_CHECK(cudaDeviceSynchronize());
    const auto actual=cached.to(tb::kCPU).contiguous();
    const auto expected=original.to(tb::kCPU).contiguous();
    if (actual.sizes()!=expected.sizes() || actual.scalar_type()!=tb::kFloat16 ||
            expected.scalar_type()!=tb::kFloat16 ||
            std::memcmp(actual.data_ptr(),expected.data_ptr(),actual.numel()*sizeof(mfq_half)))
        throw std::runtime_error("cached GPU expert differs from the canonical full GPU dispatch");
}
}

// This gate exercises production cache registration, misses, hits, eviction,
// failed-read rollback, and original-FP32 CPU fallback on actual layer shapes.
// GPU results must equal the existing canonical GPU dispatch bit-for-bit;
// CPU results are checked against an independent Python codec + FP64 oracle.
int main(int argc,char** argv) try {
    if (argc<2 || argc>4) throw std::runtime_error("expected fixture directory [--host-cache] [--benchmark]");
    bool host_cache=false,benchmark=false;
    for (int argument=2; argument<argc; ++argument) {
        const std::string option=argv[argument];
        if (option=="--host-cache") host_cache=true;
        else if (option=="--benchmark") benchmark=true;
        else throw std::runtime_error("unknown range cache test option");
    }
    const std::filesystem::path root(argv[1]);
    std::ifstream manifest(root/"cases.txt");
    if (!manifest) throw std::runtime_error("missing range cache fixture manifest");
    CudaExecutionContext execution;
    std::string name;
    int output,width,tokens,cases=0;
    while (manifest>>name>>output>>width>>tokens) {
        if (tokens!=3) throw std::runtime_error("range cache gate requires three independent samples");
        auto model=mfq::open_model_source((root/(name+".mfq")).string());
        const auto record=require_tensor(*model,"linear.weight");
        const auto reader=model->tensor_reader("linear.weight");
        std::uint64_t bytes_read=0;
        bool fail_next=false;
        auto store=std::make_shared<mfq::MfeQuantExpertStore>(record.nbytes,
            [reader,&bytes_read,&fail_next](std::size_t offset,std::uint8_t* data,std::size_t bytes) {
                if (fail_next) { fail_next=false; throw std::runtime_error("injected expert read failure"); }
                bytes_read+=bytes;
                reader(offset,reinterpret_cast<std::byte*>(data),bytes);
            });
        auto source=std::make_shared<MoeQuantRangeSource>(store);
        if (bytes_read>=record.nbytes*2/3)
            throw std::runtime_error("range cache registration materialized the entire expert pool");
        std::int64_t bytes=0;
        for (const auto& pool:source->metadata()->pools) bytes+=slot_bytes(pool);
        execution.config.moe_host_cache_bytes=host_cache ? 3*bytes : 0;
        auto cache=make_moe_expert_cache(bytes,execution.config);
        auto cached=cache_quant_moe_weight(cache,"linear.weight",source,1,0,"gate");
        finalize_moe_expert_cache(cache);
        // The baseline follows the same canonical field layout and kernels.
        // It is deliberately independent of the range source and cache.
        auto baseline=stage_cpu_mixed_moe(make_mixed_moe_runtime(load_mfe_cpu(*model,"linear.weight"),false),execution.config);
        model.reset(); // The owning range reader must survive its ModelSource.
        const auto inputs=read(root/(name+".input.f32"),std::size_t(tokens)*width);
        const auto reference=read(root/(name+".expected.f32"),std::size_t(tokens)*3*output);
        auto x=tb::tensor(inputs).reshape({tokens,width}).to(tb::kFloat16).to(tb::kCUDA);
        const std::vector<std::int32_t> all={2,0,1,0,1,2,1,2,0};
        const auto cold_route=route(all,tokens,3);
        verify_cpu(cached.forward(execution,x,cold_route),reference,all,tokens,3,output);
        if (source->cpu_projections()!=3) throw std::runtime_error("cold route did not use selected CPU projections");
        const auto cold_read_bytes=bytes_read;
        const auto cold_materializations=source->materializations();
        // Routed inputs use their own route-major row instead of token-major.
        auto routed=x.unsqueeze(1).expand({tokens,3,width}).contiguous();
        verify_cpu(cached.forward(execution,routed,cold_route),reference,all,tokens,3,output);
        if (source->cpu_projections()!=6) throw std::runtime_error("routed CPU input did not use selected experts");
        if (host_cache && (cold_read_bytes!=bytes_read || cold_materializations!=source->materializations()))
            throw std::runtime_error("CPU cold cache hit reread or decoded expert weights");
        for (int expert:{2,0,1,2}) {
            const auto selected=route(std::vector<std::int32_t>(tokens,expert),tokens,1);
            const auto before=bytes_read;
            auto original=baseline.forward(execution,x,selected);
            verify_gpu(cached.forward(execution,x,selected),original);
            if (!host_cache && bytes_read==before) throw std::runtime_error("GPU cache miss did not read the selected expert");
            if (host_cache && bytes_read!=before) throw std::runtime_error("RAM to GPU promotion reread SSD weights");
            const auto after=bytes_read;
            verify_gpu(cached.forward(execution,x,selected),original);
            if (bytes_read!=after) throw std::runtime_error("GPU cache hit reread expert weights");
            if (host_cache) for (int id=0; id<3; ++id) {
                const bool retained=source->host_cache()->contains(source->host_key(id));
                if (retained==source->gpu_resident(id))
                    throw std::runtime_error("RAM cold and GPU hot cache residency is inconsistent");
            }
        }
        const auto selected=route(std::vector<std::int32_t>(tokens,0),tokens,1);
        const auto projected_cpu=source->cpu_projections();
        if (host_cache) {
            const auto before=bytes_read;
            const std::vector<std::int32_t> cold_ids(tokens,0);
            verify_cpu(source->forward_cpu(execution,x,selected),reference,cold_ids,tokens,1,output);
            if (bytes_read!=before) throw std::runtime_error("GPU-evicted expert did not execute directly from RAM");
            // Force the existing I/O failure gate to exercise a true SSD miss.
            source->host_cache()->erase(source->host_key(0));
        }
        fail_next=true;
        bool failed=false;
        try { cached.forward(execution,x,selected); }
        catch (const std::runtime_error& error) { failed=std::string(error.what())=="injected expert read failure"; }
        if (!failed || fail_next) throw std::runtime_error("expert read failure was hidden or not exercised");
        verify_gpu(cached.forward(execution,x,selected),baseline.forward(execution,x,selected));
        if (source->cpu_projections()!=projected_cpu+(host_cache ? 1 : 0))
            throw std::runtime_error("GPU hot routes unexpectedly used CPU projections");
        // Exercise the public loader used by the native attention adapters,
        // including its owning ModelSource range callback after model release.
        auto loaded_model=mfq::open_model_source((root/(name+".mfq")).string());
        execution.moe_expert_cache=make_moe_expert_cache(bytes,execution.config);
        execution.moe_cache_registration_min_slots=1;
        mfq::cuda::weight_loader::validate_load_options(execution);
        auto loaded=load_mfe_gpu(execution,*loaded_model,"linear.weight",true,0,"gate");
        finalize_moe_expert_cache(execution.moe_expert_cache);
        loaded_model.reset();
        execution.moe_expert_cache.reset(); // MfeWeight retains its cache owner.
        verify_gpu(loaded.forward(execution,x,selected),baseline.forward(execution,x,selected));
        verify_cpu(loaded.forward(execution,x,cold_route),reference,all,tokens,3,output);
        if (!cases) {
            WithoutOwningReader borrowed(mfq::open_model_source((root/(name+".mfq")).string()));
            execution.moe_expert_cache=make_moe_expert_cache(bytes,execution.config);
            auto compatible=load_mfe_gpu(execution,borrowed,"linear.weight",true,0,"gate");
            finalize_moe_expert_cache(execution.moe_expert_cache);
            execution.moe_expert_cache.reset();
            verify_gpu(compatible.forward(execution,x,selected),baseline.forward(execution,x,selected));
        }
        MFQ_CUDA_CHECK(cudaDeviceSynchronize());
        const auto stats=source->host_cache()->stats();
        if (stats.managed_peak_bytes>stats.budget_bytes || stats.managed_bytes>stats.budget_bytes || stats.transient_bytes)
            throw std::runtime_error("retained host expert budget or transient lifetime violated");
        if (host_cache && (!stats.gpu_demotions || !stats.promotions || !stats.hits))
            throw std::runtime_error("tiered expert migration was not exercised");
        std::cout<<name<<" shape="<<output<<'x'<<width<<" cache_slots=1 slot_bytes="<<bytes
            <<" cpu_projections="<<source->cpu_projections()<<" read_bytes="<<bytes_read
            <<" host_managed_peak="<<stats.managed_peak_bytes<<" host_budget="<<stats.budget_bytes
            <<" host_hits="<<stats.hits<<" host_demotions="<<stats.gpu_demotions<<std::endl;
        if (benchmark) {
            // Include activation/ID readback, expert materialization on misses,
            // original-FP32 CPU math and result upload. This is a component
            // measurement with a warmed filesystem, not model tokens/second.
            const auto cpu_route=route(std::vector<std::int32_t>(tokens,2),tokens,1);
            if (source->gpu_resident(2)) throw std::runtime_error("benchmark expert must be cold");
            const auto before_read=bytes_read,before_decode=source->materializations();
            tb::Tensor result;
            const auto started=std::chrono::steady_clock::now();
            for (int repeat=0; repeat<3; ++repeat) {
                result=source->forward_cpu(execution,x,cpu_route);
                MFQ_CUDA_CHECK(cudaDeviceSynchronize());
            }
            const auto elapsed=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-started).count()/3;
            verify_cpu(result,reference,std::vector<std::int32_t>(tokens,2),tokens,1,output);
            std::cout<<"cpu_source_benchmark "<<name<<" host_cache="<<host_cache<<" samples="<<tokens
                <<" repeats=3 us_per_projection="<<elapsed<<" read_bytes="<<bytes_read-before_read
                <<" materializations="<<source->materializations()-before_decode<<std::endl;
        }
        ++cases;
    }
    if (!manifest.eof() || !cases) throw std::runtime_error("invalid range cache manifest");
    std::cout<<"quantized range GPU cache + CPU cold cases passed="<<cases<<'\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr<<error.what()<<'\n';
    return 1;
}
